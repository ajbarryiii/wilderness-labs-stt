"""MB1 (kea-b) encoder in MLX on the M1 Pro GPU: exact low-bit weights, FP16 activations, per-bucket timing.

Self-contained (numpy + mlx only; no import from the ternary tree). The graph mirrors
finetune/parakeet-ternary/ios/mlxarm/encoder.py (WP6b, gated exact against the reference for M_P2, the same
FastConformer architecture: 24 layers, d 1024, 8 heads, no xscale, rel-pos attention, conv k 9, BN folded),
with LocalFlow's buckets 4 / 8 / 15 / 30 s (401 / 801 / 1501 / 3001 mel frames -> 51 / 101 / 188 / 376
encoder frames) and the 1-bit export format parakeet-binary-v1 (codes uint8 [out, ceil(in / 8)], LSB first,
bit 1 = +1; FP32 row scale s).

Weight encodings (every encoder projection; linear_pos is folded into per-bucket position tables in FP32 with
FP16-rounded scales, as the Core ML build does):
  q2g<G>  MLX affine 2-bit, group G in {32, 64, 128}: q = 2 * bit in {0, 2}, scales FP16(s), biases -FP16(s),
          so dequantize = q * s - s = +-FP16(s) exactly (checked bit for bit by `exact`).
  b1      custom 1-bit Metal kernel (group-of-4 LUT, kernels/b1_mlx.py) through mx.fast.metal_kernel; weights = the export
          bits as uint32 words + FP16(s) per row (1 bit + 16 / K bits per weight).
  dense   diagnostic only: FP16 codes x FP16(s) (1.2 GB, more memory than the ternary model; MLX's own GEMM
          speed of light on these shapes).
MLX 0.32.3 has no bits = 1 affine kernels (quantize refuses bits = 1; quantized_matmul / dequantize with
bits = 1 fail with "Unable to load kernel affine_qmv_float16_t_gs_*_b_1"); see README.md.

Usage (Mac, through macguard):
  python mlx_encoder.py exact  --export EXPORT [--group 64]
  python mlx_encoder.py b1check --export EXPORT                (custom 1-bit kernel exactness + encoder agreement)
  python mlx_encoder.py time   --export EXPORT --encoding q2g64 [--buckets 4 8 15 30] [--warmup 5] [--runs 30]
  python mlx_encoder.py profile --export EXPORT --encoding q2g64     (projections-only vs full encoder)
Synthetic inputs only (seeded Gaussian mel features); no audio.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import struct
import sys
import time

import numpy as np

D_MODEL, HEADS, D_K, SUB_CH, KERNEL, N_LAYERS = 1024, 8, 128, 256, 9, 24
BUCKETS = {4: 401, 8: 801, 15: 1501, 30: 3001}  # LocalFlow buckets: seconds -> mel frames
NEG = -10000.0
LN_EPS = 1e-5
LINEARS = ("feed_forward1.linear1", "feed_forward1.linear2", "self_attn.linear_q", "self_attn.linear_k",
           "self_attn.linear_v", "self_attn.linear_out", "conv.pointwise_conv1", "conv.pointwise_conv2",
           "feed_forward2.linear1", "feed_forward2.linear2")


def encoder_frames(mel_frames: int) -> int:
    t = mel_frames
    for _ in range(3):
        t = (t - 1) // 2 + 1
    return t


MAX_T = encoder_frames(max(BUCKETS.values()))  # 376


# --- export -------------------------------------------------------------------------------------------------

class Export:
    """Minimal safetensors reader (numpy memmap) for parakeet-binary-v1."""

    DT = {"F16": np.float16, "F32": np.float32, "U8": np.uint8, "I64": np.int64}

    def __init__(self, path: str) -> None:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.offset = 8 + n
        self.path = path
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        if self.header.get("__metadata__", {}).get("format") != "parakeet-binary-v1":
            raise ValueError("not a parakeet-binary-v1 export")

    def tensor(self, key: str) -> np.ndarray:
        h = self.header[key]
        a, b = h["data_offsets"]
        raw = self.mm[self.offset + a: self.offset + b]
        return np.frombuffer(raw.tobytes(), dtype=self.DT[h["dtype"]]).reshape(h["shape"])

    def floating(self, key: str) -> np.ndarray:
        return self.tensor(key).astype(np.float32)

    def binary(self, key: str) -> tuple[np.ndarray, np.ndarray, int]:
        """(bits uint8 [out, ceil(in / 8)], FP32 scale [out], in-features)."""
        bits = self.tensor(key + ".codes")
        scale = self.tensor(key + ".scale").astype(np.float32)
        n_in = bits.shape[1] * 8  # every encoder projection has in-features divisible by 8 (1024 / 4096)
        return bits, scale, n_in


def unpack_bits(bits: np.ndarray, n_in: int) -> np.ndarray:
    """uint8 [out, ceil(in / 8)] LSB first -> uint8 [out, in] in {0, 1}."""
    return np.unpackbits(bits, axis=1, bitorder="little")[:, :n_in]


def codes_of(bits: np.ndarray, n_in: int) -> np.ndarray:
    return unpack_bits(bits, n_in).astype(np.int8) * 2 - 1


def fp16_scale(scale: np.ndarray) -> np.ndarray:
    s16 = scale.astype(np.float16)
    if not np.isfinite(s16).all() or (s16 <= 0).any() or (s16 < np.finfo(np.float16).tiny).any():
        raise ValueError("a row scale is not a normal positive FP16 number")
    return s16


def rel_positional_embedding(length: int) -> np.ndarray:
    """NeMo RelPositionalEncoding pos_emb, positions length-1 .. -(length-1): [2L-1, d] float32."""
    positions = np.arange(length - 1, -length, -1, dtype=np.float32)[:, None]
    div = np.exp(np.arange(0, D_MODEL, 2, dtype=np.float32) * np.float32(-(math.log(10000.0) / D_MODEL)))
    pe = np.zeros((positions.shape[0], D_MODEL), dtype=np.float32)
    pe[:, 0::2] = np.sin(positions * div)
    pe[:, 1::2] = np.cos(positions * div)
    return pe


def fold_positions(w_pos: np.ndarray, length: int = MAX_T) -> np.ndarray:
    """linear_pos(pos_emb) as [8, 128, 2 length - 1] float32."""
    p = rel_positional_embedding(length) @ w_pos.astype(np.float32).T
    return np.ascontiguousarray(p.reshape(2 * length - 1, HEADS, D_K).transpose(1, 2, 0))


# --- 2-bit affine packing (MLX layout) -----------------------------------------------------------------------

def pack_q2(q: np.ndarray) -> np.ndarray:
    """q [out, in] in {0..3} -> uint32 [out, in / 16], element j of a word in bits 2j..2j+1."""
    out, n = q.shape
    words = q.reshape(out, n // 16, 16).astype(np.uint32) << (2 * np.arange(16, dtype=np.uint32))
    return np.bitwise_or.reduce(words, axis=-1).astype(np.uint32)


def q2_tensors(bits: np.ndarray, scale: np.ndarray, n_in: int, group: int) -> dict:
    s16 = fp16_scale(scale)
    q = unpack_bits(bits, n_in) * 2  # {0, 2}: -1 -> 0, +1 -> 2
    scales = np.repeat(s16[:, None], n_in // group, axis=1).astype(np.float16)
    return {"wq": pack_q2(q), "scales": scales, "biases": (-scales).astype(np.float16)}


def b1_words(bits: np.ndarray) -> np.ndarray:
    """Export bytes [out, in / 8] -> uint32 [out, in / 32] (little endian: code j in bit j % 32 of word j // 32)."""
    return np.ascontiguousarray(bits).view(np.uint32).copy()


# --- weights ------------------------------------------------------------------------------------------------

class Weights:
    def __init__(self, export: Export, encoding: str, precision: str = "fp16") -> None:
        import mlx.core as mx

        self.encoding, self.precision = encoding, precision
        self.np_dt = np.float16 if precision == "fp16" else np.float32
        self.dt = mx.float16 if precision == "fp16" else mx.float32
        t0 = time.time()
        src = export

        def dense(key):
            return mx.array(src.floating(f"encoder.{key}").astype(self.np_dt))

        def conv2d(key):  # torch [out, in/g, kh, kw] -> MLX [out, kh, kw, in/g]
            return mx.array(src.floating(f"encoder.{key}").transpose(0, 2, 3, 1).astype(self.np_dt))

        self.sub = {"w0": conv2d("pre_encode.conv.0.weight"), "b0": dense("pre_encode.conv.0.bias")}
        for stage, (dw, pw) in enumerate(((2, 3), (5, 6)), start=1):
            self.sub[f"dw{stage}"] = conv2d(f"pre_encode.conv.{dw}.weight")
            self.sub[f"dwb{stage}"] = dense(f"pre_encode.conv.{dw}.bias")
            self.sub[f"pw{stage}"] = conv2d(f"pre_encode.conv.{pw}.weight")
            self.sub[f"pwb{stage}"] = dense(f"pre_encode.conv.{pw}.bias")
        self.sub["out_w"] = dense("pre_encode.out.weight")
        self.sub["out_b"] = dense("pre_encode.out.bias")
        self.layers, pos_full = [], []
        self.tiles = None
        if encoding == "b1":
            from kernels import b1_mlx

            bench = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "metal_bench.json")
            if os.path.exists(bench):
                self.tiles = {f"{k[0]}x{k[1]}@{k[2]}": v[0] + "_" + "_".join(map(str, v[1]))
                              for k, v in b1_mlx.load_tiles(bench, "lut").items()}
        for i in range(N_LAYERS):
            p = f"layers.{i}."
            L = {}
            for norm in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
                L[norm] = (dense(p + norm + ".weight"), dense(p + norm + ".bias"))
            L["pos_bias_u"] = dense(p + "self_attn.pos_bias_u")
            L["pos_bias_v"] = dense(p + "self_attn.pos_bias_v")
            for name in LINEARS:
                bits, scale, n_in = src.binary(f"encoder.{p}{name}")
                L[name] = self._linear(bits, scale, n_in)
            w = src.floating(f"encoder.{p}conv.depthwise_conv.weight").astype(np.float64)
            g, b = (src.floating(f"encoder.{p}conv.batch_norm.{k}").astype(np.float64) for k in ("weight", "bias"))
            mean, var = (src.floating(f"encoder.{p}conv.batch_norm.running_{k}").astype(np.float64)
                         for k in ("mean", "var"))
            factor = g / np.sqrt(var + 1e-5)
            L["dw_w"] = mx.array((w * factor[:, None, None]).astype(np.float32).transpose(0, 2, 1).astype(self.np_dt))
            L["dw_b"] = mx.array((b - mean * factor).astype(np.float32).astype(self.np_dt))
            bits, scale, n_in = src.binary(f"encoder.{p}self_attn.linear_pos")
            wpos = codes_of(bits, n_in).astype(np.float32) * fp16_scale(scale).astype(np.float32)[:, None]
            pos_full.append(fold_positions(wpos))
            self.layers.append(L)
        self.pos = {}
        for bkt, f in BUCKETS.items():
            t = encoder_frames(f)
            self.pos[bkt] = [mx.array(np.ascontiguousarray(pf[..., MAX_T - t: MAX_T + t - 1]).astype(self.np_dt))
                             for pf in pos_full]
        mx.eval(self.sub, self.layers, self.pos)
        self.load_seconds = time.time() - t0

    def _linear(self, bits, scale, n_in):
        import mlx.core as mx

        enc = self.encoding
        if enc.startswith("q2g"):
            qt = q2_tensors(bits, scale, n_in, int(enc[3:]))
            return ("q2", int(enc[3:]), mx.array(qt["wq"]), mx.array(qt["scales"].astype(self.np_dt)),
                    mx.array(qt["biases"].astype(self.np_dt)))
        if enc == "b1":
            return ("b1", n_in, mx.array(b1_words(bits)), mx.array(fp16_scale(scale).astype(self.np_dt)))
        if enc == "dense":
            w = codes_of(bits, n_in).astype(np.float16) * fp16_scale(scale)[:, None]
            return ("dense", mx.array(np.ascontiguousarray(w.T).astype(self.np_dt)))
        raise KeyError(enc)

    def weight_bytes(self) -> dict:
        proj = 0
        for L in self.layers:
            for name in LINEARS:
                proj += sum(a.nbytes for a in L[name][1:] if hasattr(a, "nbytes"))
        other = sum(a.nbytes for a in self.sub.values())
        for L in self.layers:
            for k, v in L.items():
                if k in LINEARS:
                    continue
                other += sum(a.nbytes for a in v) if isinstance(v, tuple) else v.nbytes
        pos = sum(a.nbytes for tabs in self.pos.values() for a in tabs)
        return {"projections": proj, "other": other, "position_tables_all_buckets": pos,
                "total": proj + other + pos}


# --- graph --------------------------------------------------------------------------------------------------

def qlinear(x, w):
    import mlx.core as mx

    kind = w[0]
    if kind == "q2":
        _, group, wq, scales, biases = w
        return mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=group, bits=2)
    if kind == "b1":
        from kernels import b1_mlx

        _, n_in, words, s16 = w
        return b1_mlx.b1_linear(x, words, s16, n_in)
    if kind == "dense":
        return x @ w[1]
    raise KeyError(kind)


def _halve(L):
    return (L - 1) // 2 + 1


def _rel_shift(x, t: int):
    import mlx.core as mx

    h = x.shape[0]
    x = mx.pad(x, [(0, 0), (0, 0), (1, 0)])
    x = x.reshape(h, 2 * t, t)[:, 1:, :]
    return x.reshape(h, t, 2 * t - 1)[:, :, :t]


def _layer(W: Weights, L: dict, x, att_bad, pad_bad, pos, t: int):
    import mlx.core as mx

    dt = W.dt

    def ln(v, norm):
        g, b = L[norm]
        return mx.fast.layer_norm(v, g, b, LN_EPS)

    def ff(v, which):
        h = qlinear(v, L[f"feed_forward{which}.linear1"])
        h = h * mx.sigmoid(h)
        h = qlinear(h, L[f"feed_forward{which}.linear2"])
        return h * mx.array(0.5, dtype=dt)

    res = x + ff(ln(x, "norm_feed_forward1"), 1)
    xa = ln(res, "norm_self_att")
    q = qlinear(xa, L["self_attn.linear_q"]).reshape(t, HEADS, D_K)
    k = qlinear(xa, L["self_attn.linear_k"]).reshape(t, HEADS, D_K).transpose(1, 0, 2)
    v = qlinear(xa, L["self_attn.linear_v"]).reshape(t, HEADS, D_K).transpose(1, 0, 2)
    qu = (q + L["pos_bias_u"]).transpose(1, 0, 2)
    qv = (q + L["pos_bias_v"]).transpose(1, 0, 2)
    ac = qu @ k.transpose(0, 2, 1)
    bd = _rel_shift(qv @ pos, t)
    scores = (ac + bd) * mx.array(1.0 / math.sqrt(D_K), dtype=dt)
    scores = mx.where(att_bad, mx.array(NEG, dtype=dt), scores)
    attn = mx.where(att_bad, mx.array(0, dtype=dt), mx.softmax(scores, axis=-1, precise=True))
    out = (attn @ v).transpose(1, 0, 2).reshape(t, D_MODEL)
    res = res + qlinear(out, L["self_attn.linear_out"])
    h = qlinear(ln(res, "norm_conv"), L["conv.pointwise_conv1"])
    h = h[:, :D_MODEL] * mx.sigmoid(h[:, D_MODEL:])
    h = mx.where(pad_bad[:, None], mx.array(0, dtype=dt), h)
    h = mx.conv1d(h[None], L["dw_w"], stride=1, padding=(KERNEL - 1) // 2, groups=D_MODEL)[0] + L["dw_b"]
    h = h * mx.sigmoid(h)
    res = res + qlinear(h, L["conv.pointwise_conv2"])
    res = res + ff(ln(res, "norm_feed_forward2"), 2)
    return ln(res, "norm_out")


def forward(W: Weights, mel, mel_length, bucket: int, n_layers: int = N_LAYERS):
    """mel [128, F_b] (compute dtype), mel_length int32 [] -> (encoder float32 [1024, T_b], length)."""
    import mlx.core as mx

    dt = W.dt
    f = mel.shape[1]
    x = mel.T[None, :, :, None]
    L = mel_length
    x = x * (mx.arange(f) < L).astype(dt)[None, :, None, None]
    x = mx.conv2d(x, W.sub["w0"], stride=2, padding=1) + W.sub["b0"]
    x = mx.maximum(x, mx.array(0, dtype=dt))
    L = _halve(L)
    for stage in (1, 2):
        x = x * (mx.arange(x.shape[1]) < L).astype(dt)[None, :, None, None]
        x = mx.conv2d(x, W.sub[f"dw{stage}"], stride=2, padding=1, groups=SUB_CH) + W.sub[f"dwb{stage}"]
        x = mx.conv2d(x, W.sub[f"pw{stage}"]) + W.sub[f"pwb{stage}"]
        x = mx.maximum(x, mx.array(0, dtype=dt))
        L = _halve(L)
    t = x.shape[1]
    x = x * (mx.arange(t) < L).astype(dt)[None, :, None, None]
    x = x.transpose(0, 1, 3, 2).reshape(t, SUB_CH * x.shape[2])
    x = x @ W.sub["out_w"].T + W.sub["out_b"]
    valid = mx.arange(t) < L
    att_bad = mx.logical_not(valid[:, None] & valid[None, :])[None]
    pad_bad = mx.logical_not(valid)
    for i in range(n_layers):
        x = _layer(W, W.layers[i], x, att_bad, pad_bad, W.pos[bucket][i], t)
    return x.T.astype(mx.float32), L


def projections_only(W: Weights, xs: dict):
    """The encoder's 240 projection matmuls on fixed inputs (same shapes and order), summed so all are live."""
    import mlx.core as mx

    acc = []
    for L in W.layers:
        for name in LINEARS:
            k_in = 4096 if name.endswith("linear2") and name.startswith("feed") else 1024
            acc.append(qlinear(xs[k_in], L[name]))
    return [a.sum() for a in acc]


# --- commands -----------------------------------------------------------------------------------------------

def synthetic_mel(bucket: int, dt, seed: int = 0):
    f = BUCKETS[bucket]
    rng = np.random.default_rng(seed + bucket)
    return rng.standard_normal((128, f), dtype=np.float32).astype(dt)


def peak_memory():
    import mlx.core as mx

    get = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return get()


def reset_peak():
    import mlx.core as mx

    f = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
    f()


def active_memory():
    import mlx.core as mx

    get = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
    return get()


def cmd_exact(args) -> dict:
    """Gate: every projection's dequantized 2-bit weights == codes x FP16(s) bit for bit (FP16 and FP32)."""
    import mlx.core as mx

    ex = Export(args.export)
    n, bad = 0, []
    for i in range(N_LAYERS):
        for name in LINEARS + ("self_attn.linear_pos",):
            key = f"encoder.layers.{i}.{name}"
            bits, scale, n_in = ex.binary(key)
            codes = codes_of(bits, n_in)
            s16 = fp16_scale(scale)
            for dt, ndt in ((mx.float16, np.float16), (mx.float32, np.float32)):
                qt = q2_tensors(bits, scale, n_in, args.group)
                deq = mx.dequantize(mx.array(qt["wq"]), mx.array(qt["scales"].astype(ndt)),
                                    mx.array(qt["biases"].astype(ndt)), group_size=args.group, bits=2)
                got = np.array(deq.astype(dt))
                want = (codes.astype(ndt) * s16.astype(ndt)[:, None]).astype(ndt)
                if got.dtype != want.dtype or not np.array_equal(got.view(np.uint8), want.view(np.uint8)):
                    bad.append(f"{key} {ndt.__name__}")
            # independent unpack of the packed words
            qt = q2_tensors(bits, scale, n_in, args.group)
            w = qt["wq"]
            q = ((w[..., None] >> (2 * np.arange(16, dtype=np.uint32))) & np.uint32(3)).reshape(w.shape[0], -1)
            if not np.array_equal(q.astype(np.int16) - 1, codes.astype(np.int16)):
                bad.append(f"{key} unpack")
            # 1-bit words: bit j of word j // 32 == code j
            words = b1_words(bits)
            ub = ((words[..., None] >> np.arange(32, dtype=np.uint32)) & np.uint32(1)).reshape(words.shape[0], -1)
            if not np.array_equal(ub.astype(np.int16) * 2 - 1, codes.astype(np.int16)):
                bad.append(f"{key} b1words")
            n += 1
    return {"command": "exact", "group": args.group, "modules": n, "failures": bad, "pass": not bad}


def cmd_b1check(args) -> dict:
    """b1 kernel exactness on real MB1 modules: integer activations in [-3, 3] make every sum exact in FP32, so the
    FP16 output must equal FP16(FP32(sum) * FP32(FP16(s))) bit for bit; then whole-encoder agreement of b1 / q2 FP16
    against the q2 FP32 graph (same weights, FP32 activations) on synthetic mel input."""
    import mlx.core as mx

    from kernels import b1_mlx

    ex = Export(args.export)
    rng = np.random.default_rng(7)
    cases, bad = [], 0
    kernels = {"lut": [("lut", (8, 1, 4)), ("lut", (16, 1, 4)), ("lut", (8, 2, 4))], "mma": [("mma", (2, 4, 2, 2))]}
    for i in (0, 11, 23):
        for name in LINEARS:
            bits, scale, n_in = ex.binary(f"encoder.layers.{i}.{name}")
            codes = codes_of(bits, n_in).astype(np.int64)
            s16 = fp16_scale(scale)
            for m in (7, 51, 101, 188, 376):
                x = rng.integers(-3, 4, size=(m, n_in)).astype(np.float16)
                want = ((x.astype(np.int64) @ codes.T).astype(np.float32) * s16.astype(np.float32)[None]).astype(np.float16)
                for tiles in kernels.values():
                    for tile in tiles:
                        b1_mlx.set_tiles({})
                        b1_mlx.DEFAULT = tile
                        y = np.array(b1_mlx.b1_linear(mx.array(x), mx.array(b1_words(bits)), mx.array(s16), n_in))
                        mism = int((y.view(np.uint16) != want.view(np.uint16)).sum())
                        bad += mism
                        cases.append({"layer": i, "module": name, "M": m, "kernel": tile[0] + "_" + "_".join(map(str, tile[1])),
                                      "mismatches": mism})
    b1_mlx.DEFAULT = ("lut", (8, 1, 4))
    agree = {}
    ref = Weights(ex, "q2g64", "fp32")
    for enc in ("q2g64", "b1"):
        W = Weights(ex, enc, "fp16")
        for b in (4, 15):
            mel32 = synthetic_mel(b, np.float32)
            r, _ = forward(ref, mx.array(mel32), mx.array(BUCKETS[b], dtype=mx.int32), b)
            o, _ = forward(W, mx.array(mel32.astype(np.float16)), mx.array(BUCKETS[b], dtype=mx.int32), b)
            r, o = np.array(r).astype(np.float64), np.array(o).astype(np.float64)
            agree[f"{enc}_fp16_vs_q2g64_fp32_b{b}"] = {
                "rel_frobenius": float(np.linalg.norm(o - r) / np.linalg.norm(r)),
                "max_abs": float(np.abs(o - r).max()), "finite": bool(np.isfinite(o).all())}
        del W
    return {"command": "b1check", "linear_cases": len(cases), "linear_mismatches": bad, "cases": cases,
            "encoder_agreement": agree, "pass": bad == 0}


def cmd_time(args) -> dict:
    import mlx.core as mx

    ex = Export(args.export)
    reset_peak()
    W = Weights(ex, args.encoding, args.precision)
    after_load = active_memory()
    wb = W.weight_bytes()
    fns = {b: mx.compile((lambda b: (lambda mel, length: forward(W, mel, length, b)))(b)) for b in args.buckets}
    res = {"command": "time", "encoding": args.encoding, "precision": args.precision, "mlx": mx.__version__,
           "load_s": round(W.load_seconds, 2), "active_after_load_bytes": after_load, "weight_bytes": wb,
           "b1_tiles": W.tiles,
           "warmup": args.warmup, "runs": args.runs, "rounds": args.rounds, "buckets": {}}
    times = {b: [] for b in args.buckets}
    order = list(args.buckets)
    for rnd in range(args.rounds):
        for b in (order if rnd % 2 == 0 else order[::-1]):
            mel = mx.array(synthetic_mel(b, W.np_dt))
            length = mx.array(BUCKETS[b], dtype=mx.int32)
            for _ in range(args.warmup):
                out, _l = fns[b](mel, length)
                mx.eval(out)
            mx.synchronize()
            for _ in range(args.runs):
                t0 = time.perf_counter()
                out, _l = fns[b](mel, length)
                mx.eval(out)
                mx.synchronize()
                times[b].append((time.perf_counter() - t0) * 1e3)
            if not np.isfinite(np.array(out)).all():
                raise RuntimeError(f"non-finite encoder output at bucket {b}")
    for b in args.buckets:
        ts = sorted(times[b])
        res["buckets"][str(b)] = {"frames": encoder_frames(BUCKETS[b]), "median_ms": round(statistics.median(ts), 2),
                                  "p10_ms": round(ts[len(ts) // 10], 2), "p90_ms": round(ts[(9 * len(ts)) // 10], 2),
                                  "min_ms": round(ts[0], 2), "n": len(ts)}
    res["peak_mlx_bytes"] = peak_memory()
    res["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    return res


def cmd_profile(args) -> dict:
    """Projections-only time (240 matmuls on fixed inputs, one compiled graph) vs the full encoder."""
    import mlx.core as mx

    ex = Export(args.export)
    W = Weights(ex, args.encoding, args.precision)
    res = {"command": "profile", "encoding": args.encoding, "buckets": {}}
    for b in args.buckets:
        t = encoder_frames(BUCKETS[b])
        rng = np.random.default_rng(b)
        xs = {k: mx.array(rng.standard_normal((t, k), dtype=np.float32).astype(W.np_dt)) for k in (1024, 4096)}
        proj = mx.compile(lambda a, c: projections_only(W, {1024: a, 4096: c}))
        full = mx.compile((lambda b: (lambda mel, length: forward(W, mel, length, b)))(b))
        mel = mx.array(synthetic_mel(b, W.np_dt))
        length = mx.array(BUCKETS[b], dtype=mx.int32)

        def run(fn, *a):
            ts = []
            for i in range(args.warmup + args.runs):
                t0 = time.perf_counter()
                o = fn(*a)
                mx.eval(o)
                mx.synchronize()
                if i >= args.warmup:
                    ts.append((time.perf_counter() - t0) * 1e3)
            return statistics.median(ts)

        p = run(proj, xs[1024], xs[4096])
        f_ = run(full, mel, length)
        res["buckets"][str(b)] = {"frames": t, "projections_ms": round(p, 2), "full_ms": round(f_, 2),
                                  "rest_ms": round(f_ - p, 2),
                                  "projection_tflops": round(2 * 579.0e6 * t / (p * 1e-3) / 1e12, 2)}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("exact", "b1check", "time", "profile"))
    ap.add_argument("--export", required=True)
    ap.add_argument("--encoding", default="q2g64")
    ap.add_argument("--precision", default="fp16", choices=("fp16", "fp32"))
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--buckets", type=int, nargs="+", default=list(BUCKETS))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    res = {"exact": cmd_exact, "b1check": cmd_b1check, "time": cmd_time, "profile": cmd_profile}[args.command](args)
    text = json.dumps(res, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    return 0 if res.get("pass", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
