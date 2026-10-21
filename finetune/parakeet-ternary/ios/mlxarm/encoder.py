"""The M_P2 encoder in MLX on the GPU, with exact 2-bit affine ternary weights (DESIGN.md "E", "MLX encoding").

Weights. Every ternary module (codes C in {-1, 0, +1} [out, in], FP32 row scales s) becomes MLX's affine
2-bit quantized format, built explicitly (MLX's own quantizer is not used):
  q = C + 1 in {0, 1, 2}, packed 16 per uint32, element j of a word in bits 2j..2j+1 (LSB first);
  scales[r, g] = FP16(s_r) and biases[r, g] = -FP16(s_r) for every group g of row r (group size GROUP);
so mx.dequantize gives q * s - s = C * FP16(s), exactly (gate 2 checks it bit for bit). The matmuls are
mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=GROUP, bits=2). Floating tensors
(subsampling, norms, depthwise convs with BatchNorm folded, positional biases) are dense in the compute dtype.
Precision "fp16": FP16 activations and FP16 scales (the arm); "fp32": the same graph and the same packed
weights with FP32 activations, scales and biases equal to FP16(s) (gate 4a's FP32 path).

Graph and masking: mil/MASKING.md (the S0 contract), mirrored from mil/encoder.py: mel [128, F_b] with
frames >= M zero, mel_length M; the subsampling input, both depthwise-stage inputs and the output are masked
with L0..L3 (L -> (L - 1) // 2 + 1); attention bad[i, j] = not (valid(i) and valid(j)) with -10000 fill and
zeroed probabilities; the conv module zeroes padded frames after the GLU before the depthwise k = 9 conv;
relative positions are the middle 2T - 1 columns of the 15 s linear_pos table (FP32 with FP16-rounded scales,
then the compute dtype). Softmax uses MLX's precise mode (FP32 accumulation inside the kernel, FP16 output).
Lengths enter the compiled function as an int32 array, so one compiled graph serves every length of a bucket.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import IOS  # noqa: F401  (sys.path)

GROUP = 64
BITS = 2
D_MODEL, HEADS, D_K, SUB_CH, KERNEL, N_LAYERS = 1024, 8, 128, 256, 9, 24
MAX_T = 188
BUCKETS = {2: 201, 4: 401, 8: 801, 15: 1501}
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


def bucket_of(samples: int) -> int:
    return next(b for b in BUCKETS if samples <= 16000 * b)


# --- exact 2-bit affine packing ----------------------------------------------------------------------------

def pack_q(q: np.ndarray) -> np.ndarray:
    """q uint [out, in] in {0, 1, 2} -> uint32 [out, in / 16], element j of each word in bits 2j (LSB first)."""
    out, n = q.shape
    if n % 16:
        raise ValueError("in-features must be a multiple of 16")
    words = q.reshape(out, n // 16, 16).astype(np.uint32) << (2 * np.arange(16, dtype=np.uint32))
    return np.bitwise_or.reduce(words, axis=-1).astype(np.uint32)


def unpack_q(wq: np.ndarray, n: int) -> np.ndarray:
    """Inverse of pack_q (independent numpy unpacking for gate 2)."""
    w = np.asarray(wq, dtype=np.uint32)
    q = (w[..., None] >> (2 * np.arange(16, dtype=np.uint32))) & np.uint32(3)
    return q.reshape(w.shape[0], -1)[:, :n].astype(np.uint8)


def quantized(codes: np.ndarray, scale: np.ndarray, dtype) -> dict:
    """The arm's quantized tensors of one module (numpy): wq uint32, scales / biases [out, in / GROUP] (dtype)."""
    from mil.weights import fp16_scale

    if codes.shape[1] % GROUP:
        raise ValueError("in-features must be a multiple of the group size")
    s16 = fp16_scale(scale)
    groups = codes.shape[1] // GROUP
    scales = np.repeat(s16[:, None], groups, axis=1).astype(dtype)
    return {"wq": pack_q((codes + 1).astype(np.uint8)), "scales": scales, "biases": (-scales).astype(dtype),
            "shape": codes.shape}


# --- weights ------------------------------------------------------------------------------------------------

class Weights:
    """M_P2's encoder as MLX arrays in one precision ("fp16" or "fp32"), read from mil.weights.Source."""

    def __init__(self, model: str = "mp2", precision: str = "fp16", n_layers: int = N_LAYERS, source=None) -> None:
        import mlx.core as mx

        from mil.encoder import fold_positions
        from mil.weights import Source, fp16_scale

        self.precision, self.n_layers = precision, n_layers
        self.np_dt = np.float16 if precision == "fp16" else np.float32
        self.dt = mx.float16 if precision == "fp16" else mx.float32
        self.source = source or Source(model)
        src = self.source
        t0 = time.time()

        def dense(key: str):
            return mx.array(src.floating(f"encoder.{key}").astype(self.np_dt))

        def conv2d(key: str):  # torch [out, in/g, kh, kw] -> MLX [out, kh, kw, in/g]
            return mx.array(src.floating(f"encoder.{key}").transpose(0, 2, 3, 1).astype(self.np_dt))

        self.sub = {"w0": conv2d("pre_encode.conv.0.weight"), "b0": dense("pre_encode.conv.0.bias")}
        for stage, (dw, pw) in enumerate(((2, 3), (5, 6)), start=1):
            self.sub[f"dw{stage}"] = conv2d(f"pre_encode.conv.{dw}.weight")
            self.sub[f"dwb{stage}"] = dense(f"pre_encode.conv.{dw}.bias")
            self.sub[f"pw{stage}"] = conv2d(f"pre_encode.conv.{pw}.weight")
            self.sub[f"pwb{stage}"] = dense(f"pre_encode.conv.{pw}.bias")
        self.sub["out_w"] = dense("pre_encode.out.weight")
        self.sub["out_b"] = dense("pre_encode.out.bias")
        self.layers, self.pos_full, self.packed = [], [], {}
        for i in range(n_layers):
            p = f"layers.{i}."
            L = {}
            for norm in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
                L[norm] = (dense(p + norm + ".weight"), dense(p + norm + ".bias"))
            L["pos_bias_u"] = dense(p + "self_attn.pos_bias_u")
            L["pos_bias_v"] = dense(p + "self_attn.pos_bias_v")
            for name in LINEARS:
                codes, scale = src.ternary(f"encoder.{p}{name}")
                if codes.ndim != 2:
                    codes = codes.reshape(codes.shape[0], -1)
                qt = quantized(codes, scale, self.np_dt)
                self.packed[p + name] = qt
                L[name] = (mx.array(qt["wq"]), mx.array(qt["scales"]), mx.array(qt["biases"]))
            # depthwise conv with BatchNorm folded (as mil.encoder.ArmProvider.depthwise), MLX [1024, 9, 1]
            w = src.floating(f"encoder.{p}conv.depthwise_conv.weight").astype(np.float64)
            g, b = (src.floating(f"encoder.{p}conv.batch_norm.{k}").astype(np.float64) for k in ("weight", "bias"))
            mean, var = (src.floating(f"encoder.{p}conv.batch_norm.running_{k}").astype(np.float64) for k in ("mean", "var"))
            factor = g / np.sqrt(var + 1e-5)
            L["dw_w"] = mx.array((w * factor[:, None, None]).astype(np.float32).transpose(0, 2, 1).astype(self.np_dt))
            L["dw_b"] = mx.array((b - mean * factor).astype(np.float32).astype(self.np_dt))
            codes, scale = src.ternary(f"encoder.{p}self_attn.linear_pos")
            wpos = codes.astype(np.float32) * fp16_scale(scale).astype(np.float32)[:, None]  # codes x FP16(s)
            self.pos_full.append(fold_positions(wpos)[0])  # [8, 128, 375] float32
            self.layers.append(L)
        self.pos = {}  # bucket -> [layer] MLX [8, 128, 2T - 1]
        for b, f in BUCKETS.items():
            t = encoder_frames(f)
            self.pos[b] = [mx.array(np.ascontiguousarray(pf[..., MAX_T - t: MAX_T + t - 1]).astype(self.np_dt))
                           for pf in self.pos_full]
        mx.eval(self.sub, self.layers, self.pos)
        self.load_seconds = time.time() - t0


# --- graph --------------------------------------------------------------------------------------------------

def _qlinear(x, w):
    import mlx.core as mx

    wq, scales, biases = w
    return mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=GROUP, bits=BITS)


def _halve(L):
    return (L - 1) // 2 + 1


def _rel_shift(x, t: int):
    """x [H, T, 2T - 1] -> [H, T, T] (reference rel_shift followed by [..., :T])."""
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
        h = _qlinear(v, L[f"feed_forward{which}.linear1"])
        h = h * mx.sigmoid(h)
        h = _qlinear(h, L[f"feed_forward{which}.linear2"])
        return h * mx.array(0.5, dtype=dt)

    res = x + ff(ln(x, "norm_feed_forward1"), 1)
    xa = ln(res, "norm_self_att")
    q = _qlinear(xa, L["self_attn.linear_q"]).reshape(t, HEADS, D_K)
    k = _qlinear(xa, L["self_attn.linear_k"]).reshape(t, HEADS, D_K).transpose(1, 0, 2)
    v = _qlinear(xa, L["self_attn.linear_v"]).reshape(t, HEADS, D_K).transpose(1, 0, 2)
    qu = (q + L["pos_bias_u"]).transpose(1, 0, 2)
    qv = (q + L["pos_bias_v"]).transpose(1, 0, 2)
    ac = qu @ k.transpose(0, 2, 1)
    bd = _rel_shift(qv @ pos, t)
    scores = (ac + bd) * mx.array(1.0 / math.sqrt(D_K), dtype=dt)
    scores = mx.where(att_bad, mx.array(NEG, dtype=dt), scores)
    attn = mx.where(att_bad, mx.array(0, dtype=dt), mx.softmax(scores, axis=-1, precise=True))
    out = (attn @ v).transpose(1, 0, 2).reshape(t, D_MODEL)
    res = res + _qlinear(out, L["self_attn.linear_out"])
    h = _qlinear(ln(res, "norm_conv"), L["conv.pointwise_conv1"])            # [T, 2048]
    h = h[:, :D_MODEL] * mx.sigmoid(h[:, D_MODEL:])                          # GLU
    h = mx.where(pad_bad[:, None], mx.array(0, dtype=dt), h)
    h = mx.conv1d(h[None], L["dw_w"], stride=1, padding=(KERNEL - 1) // 2, groups=D_MODEL)[0] + L["dw_b"]
    h = h * mx.sigmoid(h)
    res = res + _qlinear(h, L["conv.pointwise_conv2"])
    res = res + ff(ln(res, "norm_feed_forward2"), 2)
    return ln(res, "norm_out")


def forward(W: Weights, mel, mel_length, bucket: int):
    """mel [128, F_b] (compute dtype), mel_length int32 [] -> (encoder float32 [1024, T_b], encoder_length int32 [])."""
    import mlx.core as mx

    dt = W.dt
    f = mel.shape[1]
    x = mel.T[None, :, :, None]                                              # NHWC [1, F, 128, 1]
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
    x = x.transpose(0, 1, 3, 2).reshape(t, SUB_CH * x.shape[2])               # [T, 256 x 16] channel-major
    x = x @ W.sub["out_w"].T + W.sub["out_b"]
    valid = mx.arange(t) < L
    att_bad = mx.logical_not(valid[:, None] & valid[None, :])[None]           # [1, T, T]
    pad_bad = mx.logical_not(valid)
    for i in range(W.n_layers):
        x = _layer(W, W.layers[i], x, att_bad, pad_bad, W.pos[bucket][i], t)
    return x.T.astype(mx.float32), L


class Encoder:
    """Compiled per-bucket encoder functions on the GPU (MLX compiles once per input shape)."""

    def __init__(self, W: Weights, compile: bool = True) -> None:
        import mlx.core as mx

        self.W = W
        self.fns = {}
        for b in BUCKETS:
            fn = (lambda b: (lambda mel, length: forward(W, mel, length, b)))(b)
            self.fns[b] = mx.compile(fn) if compile else fn

    def mel_input(self, features: np.ndarray, bucket: int):
        """Reference features [128, M + 1] (frame M zero) zero-padded to [128, F_b] in the compute dtype."""
        import mlx.core as mx

        f = BUCKETS[bucket]
        mel = np.zeros((128, f), dtype=self.W.np_dt)
        mel[:, :features.shape[1]] = features.astype(self.W.np_dt)
        return mx.array(mel)

    def __call__(self, mel, mel_length: int, bucket: int):
        import mlx.core as mx

        out, length = self.fns[bucket](mel, mx.array(mel_length, dtype=mx.int32))
        mx.eval(out, length)
        return out, int(length.item())
