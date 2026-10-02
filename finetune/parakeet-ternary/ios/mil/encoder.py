"""Plain-layout FastConformer encoder of Parakeet-TDT 0.6B v2 as MIL programs (DESIGN.md "D. Graph layout", plain).

The graph mirrors reference.py (NeMo ConformerEncoder at inference) op for op, in FP16, with the masking
contract of mil/MASKING.md:
  mel [1, 128, F] fp32, mel_length [1] int32 -> cast fp16 -> transpose [1, F, 128] -> [1, 1, F, 128]
  subsampling (dw_striding 8x): x * mask(L0) -> conv 3x3/2 + ReLU -> x * mask(L1) -> dw 3x3/2 -> pw 1x1 ->
    ReLU -> x * mask(L2) -> dw 3x3/2 -> pw 1x1 -> ReLU -> x * mask(L3) -> [1, T, 4096] -> linear -> [1, T, 1024]
    (L_{k+1} = floor((L_k - 1) / 2) + 1; equal on every frame to the reference's mask before each layer,
    since masks before 1x1 convs and ReLUs only touch frames a later mask zeroes again)
  24 x ConformerLayer: x + 0.5 FF1(LN x); + RelPosMHSA(LN x); + Conv(LN x); + 0.5 FF2(LN x); LN
    RelPosMHSA: q, k, v projections; (q + u) k^T; rel_shift((q + v) P) with P the per-bucket folded
    linear_pos(pos_emb) table [1, 8, 128, 2T - 1] (FP32 numpy, cast to FP16); (ac + bd) * 1/sqrt(128);
    select(key/query mask, -1e4); softmax; select(mask, 0); attn v; linear_out
    Conv: pointwise_conv1 (1x1) -> GLU -> select(pad mask, 0) -> pad 4/4 -> depthwise k=9 with BatchNorm
    folded into weight and bias (FP32/FP64 numpy, cast) -> SiLU -> pointwise_conv2
  -> transpose [1, 1024, T] -> cast fp32: "encoder"; "encoder_length" = L3 (int32 [1]).
Ternary modules go through the arm's encoding (encodings.py); linear_pos is folded into the position
table and is therefore not arm-encoded in any arm. Frames at or beyond encoder_length are unspecified.

Length variants: fixed (F = 1501), multifunction (functions b2, b4, b8, b15 with F = 201/401/801/1501,
weights shared through coremltools' cross-function constant deduplication), enumerated (one function,
mel EnumeratedShapes over the four F; the graph is shape-generic and the position table is a slice of the
15 s table at runtime-known offsets).
"""
from __future__ import annotations

import math

import numpy as np

from . import encodings
from .weights import N_LAYERS, Source, fp16_scale

D_MODEL, HEADS, D_K, FF = 1024, 8, 128, 4096
MEL = 128
SUB_CH = 256
KERNEL = 9
BN_EPS = 1e-5
LN_EPS = 1e-5
NEG = -10000.0  # reference.INF_VAL
BUCKETS = {2: 201, 4: 401, 8: 801, 15: 1501}  # bucket seconds -> allocated mel frames (N_alloc // 160 + 1)
MAX_T = 188
SITES = {"feed_forward1.linear1": "ff1_in", "feed_forward1.linear2": "ff1_mid", "self_attn.linear_q": "att_in",
         "self_attn.linear_k": "att_in", "self_attn.linear_v": "att_in", "self_attn.linear_out": "att_out",
         "conv.pointwise_conv1": "conv_in", "conv.pointwise_conv2": "conv_mid", "feed_forward2.linear1": "ff2_in",
         "feed_forward2.linear2": "ff2_mid"}
CONV_MODULES = ("conv.pointwise_conv1", "conv.pointwise_conv2")


def encoder_frames(mel_frames: int) -> int:
    t = mel_frames
    for _ in range(3):
        t = (t - 1) // 2 + 1
    return t


def rel_positional_embedding(length: int) -> np.ndarray:
    """reference.rel_positional_embedding(length, 1024)[0] in float32 numpy (same float32 operations)."""
    import torch

    import reference

    return reference.rel_positional_embedding(length, D_MODEL)[0].numpy()


def fold_positions(w_pos: np.ndarray, length: int = MAX_T) -> np.ndarray:
    """linear_pos(pos_emb) for `length` frames as [1, 8, 128, 2 length - 1] float32 (matmul in FP32)."""
    import torch

    pos = torch.from_numpy(rel_positional_embedding(length))  # [2L-1, 1024]
    p = pos @ torch.from_numpy(w_pos.astype(np.float32)).T     # [2L-1, 1024]
    return p.view(2 * length - 1, HEADS, D_K).permute(1, 2, 0).unsqueeze(0).contiguous().numpy()


def slice_positions(table: np.ndarray, t: int) -> np.ndarray:
    """The 2t - 1 middle columns of a MAX_T table: positions t - 1 .. -(t - 1)."""
    return np.ascontiguousarray(table[..., MAX_T - t: MAX_T + t - 1])


# --- weight providers --------------------------------------------------------------------------------------

class ArmProvider:
    """Encoder weights of a benchmark model in one arm's encoding (cached, so functions share arrays)."""

    def __init__(self, source: Source, arm: str, n_layers: int = N_LAYERS, act_scales: dict | None = None) -> None:
        if arm not in encodings.ENCODER_ARMS + ("F32",):
            raise KeyError(arm)
        self.source, self.arm, self.n_layers = source, arm, n_layers
        self.act_scales = act_scales or {}
        if arm == "C5" and not act_scales:
            raise ValueError("C5 needs activation scales (reference_cache calibrate)")
        self._modules: dict[str, encodings.Encoded] = {}
        self._pos: dict[int, np.ndarray] = {}
        self._dense: dict[str, np.ndarray] = {}
        self.opset = "iOS26"  # iOS26 target (iOS18 op definitions); see README WP3 "opset"
        # "F32" is a diagnostic arm (mil/diag.py): the same graph in FP32 with dense codes x FP16(s) weights
        self.dt, self.np = ("fp32", np.float32) if arm == "F32" else ("fp16", np.float16)

    def dense(self, key: str) -> np.ndarray:
        if key not in self._dense:
            self._dense[key] = self.source.floating(f"encoder.{key}").astype(self.np)
        return self._dense[key]

    weight = dense

    def depthwise(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        key = f"layers.{i}.depthwise"
        if key + ".weight" not in self._dense:
            p = f"encoder.layers.{i}.conv."
            w = self.source.floating(p + "depthwise_conv.weight").astype(np.float64)
            g, b = (self.source.floating(p + f"batch_norm.{k}").astype(np.float64) for k in ("weight", "bias"))
            mean, var = (self.source.floating(p + f"batch_norm.running_{k}").astype(np.float64) for k in ("mean", "var"))
            factor = g / np.sqrt(var + BN_EPS)
            self._dense[key + ".weight"] = (w * factor[:, None, None]).astype(np.float32).astype(self.np)
            self._dense[key + ".bias"] = (b - mean * factor).astype(np.float32).astype(self.np)
        return self._dense[key + ".weight"], self._dense[key + ".bias"]

    def module(self, i: int, suffix: str) -> encodings.Encoded:
        key = f"layers.{i}.{suffix}"
        if key not in self._modules:
            codes, scale = self.source.ternary(f"encoder.{key}")
            self._modules[key] = encodings.encode(self.arm, codes, scale, 3 if suffix in CONV_MODULES else 2)
        return self._modules[key]

    def project(self, x, i: int, suffix: str):
        return encodings.apply(self.module(i, suffix), x, f"l{i}_{suffix.replace('.', '_')}")

    def act(self, x, i: int, site: str):
        if self.arm != "C5":
            return x
        return encodings.quantize_activation(x, self.act_scales[f"layers.{i}.{site}"], f"l{i}_{site}")

    def pos_full(self, i: int) -> np.ndarray:
        if i not in self._pos:
            codes, scale = self.source.ternary(f"encoder.layers.{i}.self_attn.linear_pos")
            w = codes.astype(np.float32) * fp16_scale(scale).astype(np.float32)[:, None]  # codes x FP16(s)
            self._pos[i] = fold_positions(w).astype(self.np)
        return self._pos[i]

    def pos_table(self, i: int, t: int):
        return slice_positions(self.pos_full(i), t)

    def manifest(self) -> dict:
        mods = list(self._modules.values())
        by_kind: dict[str, dict] = {}
        for key, enc in self._modules.items():
            kind = key.split(".", 2)[2]
            by_kind.setdefault(kind, {"count": 0, "inputs": enc.describe()})["count"] += 1
        totals: dict[str, int] = {}
        for enc in mods:
            for k, v in enc.nbytes().items():
                totals[k] = totals.get(k, 0) + v
        return {"arm": self.arm, "chain": encodings.CHAINS[encodings.family(self.arm)], "modules": len(mods),
                "constexpr_inputs_by_module": by_kind, "encoded_bytes_by_const": totals,
                "encoded_bytes_total": sum(totals.values()),
                "position_tables": {"layers": len(self._pos), "dtype": "float16", "full_shape": [1, HEADS, D_K, 2 * MAX_T - 1],
                                    "note": "linear_pos folded: pos_emb @ (codes x FP16(s))^T in FP32, cast to FP16; "
                                            "per bucket the middle 2T-1 columns"},
                "activation_quantization": ({"sites": len(self.act_scales), "scheme": "per-tensor symmetric int8, "
                                             "zero point 0, scale = max|x| / 127 over the calibration clips"}
                                            if self.arm == "C5" else None)}


class G0Provider:
    """G0: C0's own encoder tensors (6-bit palettized LUTs and indices, dense FP16 consts), iOS17 opset."""

    def __init__(self, c0, n_layers: int = N_LAYERS) -> None:
        self.c0, self.n_layers, self.arm, self.opset = c0, n_layers, "G0", "iOS17"
        self.dt, self.np = "fp16", np.float16
        self._vars: dict[str, object] = {}

    def _get(self, role: str, name: str):
        from coremltools.converters.mil import Builder as mb

        entry = self.c0[role]
        if entry[0] == "dense":
            return entry[1]
        _, packed, lut, shape, _ = entry
        return mb.constexpr_lut_to_dense(indices=packed, lut=lut, shape=shape, name=name)

    def dense(self, key: str):
        return self._get(key, key.replace(".", "_") + "_g0")

    def weight(self, key: str):
        return self._get(key, key.replace(".", "_") + "_g0")

    def depthwise(self, i: int):
        return (self._get(f"layers.{i}.depthwise.weight", f"l{i}_dw_g0"), self.c0[f"layers.{i}.depthwise.bias"][1])

    def project(self, x, i: int, suffix: str):
        from coremltools.converters.mil import Builder as mb

        name = f"l{i}_{suffix.replace('.', '_')}"
        w = self._get(f"layers.{i}.{suffix}.weight", name + "_weight")
        return mb.conv(x=x, weight=w, name=name + "_mm") if suffix in CONV_MODULES else mb.linear(x=x, weight=w, name=name + "_mm")

    def act(self, x, i, site):
        return x

    def pos_table(self, i: int, t: int):
        if t != MAX_T:
            raise ValueError("G0 has C0's fixed 15 s window only")
        return self._get(f"layers.{i}.pos_table", f"l{i}_pos_g0")

    def manifest(self) -> dict:
        lut = [e for e in self.c0.entries.values() if e[0] == "lut"]
        return {"arm": "G0", "chain": encodings.CHAINS["G0"], "c0": self.c0.provenance,
                "lut_tensors": len(lut), "index_bytes": int(sum(e[1].size for e in lut)),
                "lut_bytes": int(sum(e[2].size * 2 for e in lut)),
                "constexpr_inputs": {"indices": "uint8 packed 6-bit (iOS16 layout)", "lut": "float16 [64]",
                                     "shape": "uint32"},
                "note": "all 294 palettized tensors (incl. subsampling, depthwise with BatchNorm folded, folded "
                        "linear_pos tables [1, 8, 128, 375]) and 320 dense FP16 consts taken from C0's weight.bin"}


# --- graph -----------------------------------------------------------------------------------------------

class _Dims:
    """Static ints when the mel length is fixed, MIL scalars when it is symbolic (enumerated shapes)."""

    def __init__(self, static: bool) -> None:
        self.static = static

    def dim(self, x, axis: int):
        from coremltools.converters.mil import Builder as mb

        if self.static:
            return int(x.shape[axis])
        return mb.gather(x=mb.shape(x=x), indices=np.int32(axis), axis=0)

    def vec(self, items: list):
        """A shape/begin/end vector from ints and int32 scalars."""
        from coremltools.converters.mil import Builder as mb

        if all(isinstance(v, (int, np.integer)) for v in items):
            return [int(v) for v in items]
        parts = [np.array([v], dtype=np.int32) if isinstance(v, (int, np.integer)) else mb.reshape(x=v, shape=[1])
                 for v in items]
        return mb.concat(values=parts, axis=0)

    def arange(self, n):
        from coremltools.converters.mil import Builder as mb

        if isinstance(n, (int, np.integer)):
            return np.arange(int(n), dtype=np.int32)
        return mb.range_1d(start=np.int32(0), end=n, step=np.int32(1))

    def add(self, a, b):
        from coremltools.converters.mil import Builder as mb

        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            return int(a) + int(b)
        return mb.add(x=a, y=np.int32(b) if isinstance(b, (int, np.integer)) else b)

    def mul(self, a, k: int):
        from coremltools.converters.mil import Builder as mb

        return int(a) * k if isinstance(a, (int, np.integer)) else mb.mul(x=a, y=np.int32(k))

    def sub_from(self, k: int, a):
        from coremltools.converters.mil import Builder as mb

        return k - int(a) if isinstance(a, (int, np.integer)) else mb.sub(x=np.int32(k), y=a)


def _halve_length(L, name: str):
    """floor((L - 1) / 2) + 1 on int32 [1]: one kernel-3 stride-2 pad-1 stage (reference Subsampling)."""
    from coremltools.converters.mil import Builder as mb

    return mb.add(x=mb.floor_div(x=mb.sub(x=L, y=np.int32(1)), y=np.int32(2)), y=np.int32(1), name=name)


def _time_mask(d: _Dims, n, L, name: str, dt: str = "fp16"):
    """[1, 1, n, 1] = (t < L) in the compute dtype."""
    from coremltools.converters.mil import Builder as mb

    valid = mb.less(x=d.arange(n), y=L)
    return mb.expand_dims(x=mb.cast(x=valid, dtype=dt), axes=[0, 1, 3], name=name)


def _rel_shift(d: _Dims, x, name: str, zero=np.float16(0)):
    """reference.RelPositionAttention.rel_shift followed by [..., :T]: x [1, H, T, 2T - 1] -> [1, H, T, T]."""
    from coremltools.converters.mil import Builder as mb

    t = d.dim(x, 2)
    x = mb.pad(x=x, pad=[0, 0, 0, 0, 0, 0, 1, 0], mode="constant", constant_val=zero, name=name + "_pad")
    x = mb.reshape(x=x, shape=d.vec([1, HEADS, -1, t]), name=name + "_view")
    x = mb.slice_by_index(x=x, begin=[0, 0, 1, 0], end=[0, 0, 0, 0], end_mask=[True, True, True, True],
                          name=name + "_drop")
    x = mb.reshape(x=x, shape=d.vec([1, HEADS, t, -1]), name=name + "_back")
    return mb.slice_by_index(x=x, begin=[0, 0, 0, 0], end=d.vec([1, HEADS, t, t]), name=name)


def _layer(d: _Dims, P, i: int, x, att_bad, pad_bad, pos):
    from coremltools.converters.mil import Builder as mb

    def ln(v, norm):
        return mb.layer_norm(x=v, axes=[-1], gamma=P.dense(f"layers.{i}.{norm}.weight"),
                             beta=P.dense(f"layers.{i}.{norm}.bias"), epsilon=P.np(LN_EPS),
                             name=f"l{i}_{norm}")

    def ff(v, which):
        h = P.project(P.act(v, i, f"{which}_in"), i, f"feed_forward{which[-1]}.linear1")
        h = mb.silu(x=h, name=f"l{i}_{which}_silu")
        h = P.project(P.act(h, i, f"{which}_mid"), i, f"feed_forward{which[-1]}.linear2")
        return mb.mul(x=h, y=P.np(0.5), name=f"l{i}_{which}_half")

    # macaron FF1
    res = mb.add(x=x, y=ff(ln(x, "norm_feed_forward1"), "ff1"), name=f"l{i}_res_ff1")
    # rel-pos MHSA
    xa = P.act(ln(res, "norm_self_att"), i, "att_in")
    heads = [1, -1, HEADS, D_K]
    q = mb.reshape(x=P.project(xa, i, "self_attn.linear_q"), shape=heads, name=f"l{i}_q")
    k = mb.transpose(x=mb.reshape(x=P.project(xa, i, "self_attn.linear_k"), shape=heads), perm=[0, 2, 1, 3], name=f"l{i}_k")
    v = mb.transpose(x=mb.reshape(x=P.project(xa, i, "self_attn.linear_v"), shape=heads), perm=[0, 2, 1, 3], name=f"l{i}_v")
    qu = mb.transpose(x=mb.add(x=q, y=P.dense(f"layers.{i}.self_attn.pos_bias_u")), perm=[0, 2, 1, 3], name=f"l{i}_qu")
    qv = mb.transpose(x=mb.add(x=q, y=P.dense(f"layers.{i}.self_attn.pos_bias_v")), perm=[0, 2, 1, 3], name=f"l{i}_qv")
    ac = mb.matmul(x=qu, y=k, transpose_y=True, name=f"l{i}_ac")
    bd = _rel_shift(d, mb.matmul(x=qv, y=pos, name=f"l{i}_bd_raw"), f"l{i}_bd", P.np(0))
    scores = mb.mul(x=mb.add(x=ac, y=bd), y=P.np(1.0 / math.sqrt(D_K)), name=f"l{i}_scores")
    scores = mb.select(cond=att_bad, a=P.np(NEG), b=scores, name=f"l{i}_scores_masked")
    attn = mb.select(cond=att_bad, a=P.np(0), b=mb.softmax(x=scores, axis=-1), name=f"l{i}_attn")
    out = mb.matmul(x=attn, y=v, name=f"l{i}_attn_v")
    out = mb.reshape(x=mb.transpose(x=out, perm=[0, 2, 1, 3]), shape=[1, -1, D_MODEL], name=f"l{i}_attn_merge")
    out = P.project(P.act(out, i, "att_out"), i, "self_attn.linear_out")
    res = mb.add(x=res, y=out, name=f"l{i}_res_att")
    # conv module
    xc = mb.transpose(x=P.act(ln(res, "norm_conv"), i, "conv_in"), perm=[0, 2, 1], name=f"l{i}_conv_in")
    h = P.project(xc, i, "conv.pointwise_conv1")
    a, b = mb.split(x=h, num_splits=2, axis=1, name=f"l{i}_glu_split")
    h = mb.mul(x=a, y=mb.sigmoid(x=b), name=f"l{i}_glu")
    h = mb.select(cond=pad_bad, a=P.np(0), b=h, name=f"l{i}_conv_masked")
    h = mb.pad(x=h, pad=[0, 0, 0, 0, (KERNEL - 1) // 2, (KERNEL - 1) // 2], mode="constant",
               constant_val=P.np(0), name=f"l{i}_dw_pad")
    dw_w, dw_b = P.depthwise(i)
    h = mb.conv(x=h, weight=dw_w, bias=dw_b, groups=D_MODEL, pad_type="valid", name=f"l{i}_dw")
    h = mb.silu(x=h, name=f"l{i}_dw_silu")
    h = P.project(P.act(h, i, "conv_mid"), i, "conv.pointwise_conv2")
    res = mb.add(x=res, y=mb.transpose(x=h, perm=[0, 2, 1]), name=f"l{i}_res_conv")
    # macaron FF2, output norm
    res = mb.add(x=res, y=ff(ln(res, "norm_feed_forward2"), "ff2"), name=f"l{i}_res_ff2")
    return ln(res, "norm_out")


def forward(P, mel, mel_length, static: bool, t_static: int | None = None):
    """The encoder body inside a MIL function context. Returns (encoder fp32 [1, 1024, T], encoder_length)."""
    from coremltools.converters.mil import Builder as mb

    d = _Dims(static)
    x = mb.cast(x=mel, dtype=P.dt, name="mel_" + P.dt)
    x = mb.expand_dims(x=mb.transpose(x=x, perm=[0, 2, 1]), axes=[1], name="mel_nchw")  # [1, 1, F, 128]
    L = mel_length
    # subsampling with MaskedConvSequential masking
    x = mb.mul(x=x, y=_time_mask(d, d.dim(x, 2), L, "mask0", P.dt), name="sub_in_masked")
    x = mb.conv(x=x, weight=P.weight("pre_encode.conv.0.weight"), bias=P.dense("pre_encode.conv.0.bias"),
                strides=[2, 2], pad_type="custom", pad=[1, 1, 1, 1], name="sub_conv0")
    x = mb.relu(x=x, name="sub_relu0")
    L = _halve_length(L, "len1")
    for stage, (dw, pw) in enumerate(((2, 3), (5, 6)), start=1):
        x = mb.mul(x=x, y=_time_mask(d, d.dim(x, 2), L, f"mask{stage}", P.dt), name=f"sub_masked{stage}")
        x = mb.conv(x=x, weight=P.weight(f"pre_encode.conv.{dw}.weight"), bias=P.dense(f"pre_encode.conv.{dw}.bias"),
                    strides=[2, 2], pad_type="custom", pad=[1, 1, 1, 1], groups=SUB_CH, name=f"sub_dw{stage}")
        x = mb.conv(x=x, weight=P.weight(f"pre_encode.conv.{pw}.weight"), bias=P.dense(f"pre_encode.conv.{pw}.bias"),
                    name=f"sub_pw{stage}")
        x = mb.relu(x=x, name=f"sub_relu{stage}")
        L = _halve_length(L, f"len{stage + 1}")
    x = mb.mul(x=x, y=_time_mask(d, d.dim(x, 2), L, "mask3", P.dt), name="sub_out_masked")
    x = mb.reshape(x=mb.transpose(x=x, perm=[0, 2, 1, 3]), shape=[1, -1, SUB_CH * 16], name="sub_flat")
    x = mb.linear(x=x, weight=P.weight("pre_encode.out.weight"), bias=P.dense("pre_encode.out.bias"), name="sub_out")
    length = mb.identity(x=L, name="encoder_length")
    # masks: valid = t < L; attention mask (query and key), conv pad mask
    t = d.dim(x, 1)
    valid = mb.less(x=d.arange(t), y=L, name="valid")
    pair = mb.logical_and(x=mb.expand_dims(x=valid, axes=[1]), y=mb.expand_dims(x=valid, axes=[0]))
    att_bad = mb.expand_dims(x=mb.logical_not(x=pair), axes=[0, 1], name="att_mask")      # [1, 1, T, T]
    pad_bad = mb.expand_dims(x=mb.logical_not(x=valid), axes=[0, 1], name="pad_mask")     # [1, 1, T]
    for i in range(P.n_layers):
        if static:
            pos = P.pos_table(i, t)
        else:  # the 15 s table sliced at runtime-known offsets: columns MAX_T - T .. MAX_T + T - 2
            full = P.pos_full(i)
            pos = mb.slice_by_index(x=full, begin=d.vec([0, 0, 0, d.sub_from(MAX_T, t)]),
                                    end=d.vec([1, HEADS, D_K, d.add(t, MAX_T - 1)]), name=f"l{i}_pos")
        x = _layer(d, P, i, x, att_bad, pad_bad, pos)
    out = mb.cast(x=mb.transpose(x=x, perm=[0, 2, 1]), dtype="fp32", name="encoder")
    return out, length


def function(P, mel_frames: int | None, opset):
    """One MIL function: mel [1, 128, mel_frames] (None: symbolic) + mel_length -> encoder, encoder_length."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import get_new_symbol, types

    frames = mel_frames if mel_frames is not None else get_new_symbol()

    @mb.function(input_specs=[mb.TensorSpec(shape=(1, MEL, frames), dtype=types.fp32),
                              mb.TensorSpec(shape=(1,), dtype=types.int32)], opset_version=opset)
    def encoder(mel, mel_length):
        return forward(P, mel, mel_length, static=mel_frames is not None)

    return encoder
