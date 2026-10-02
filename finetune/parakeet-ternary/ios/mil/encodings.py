"""Encoder weight encodings (DESIGN.md "Encoder weight encodings (C)"): constexpr chains and effective matrices.

encode(arm, codes, scale, rank) turns one ternary module (int8 codes [out, in] in {-1, 0, 1}, FP32 row
scales [out]) into an Encoded: the numpy constants of the arm's constexpr chain (shapes for rank 2
[out, in] linear weights or rank 3 [out, in, 1] pointwise-conv weights), the post-matmul step (C7/C8) and
the byte accounting. apply() emits the chain and the matmul inside a MIL builder context. Scales are
FP16(s) in every arm; rounding FP32 -> FP16 is gate 3's term, not the encoding's.

| arm    | chain                                                                                  |
| C1     | const fp16 W = codes x FP16(s)                                                         |
| C3/C5  | constexpr_blockwise_shift_scale(data int8 codes, scale fp16 [out, 1])                  |
| C4     | constexpr_lut_to_dense(uint2 codes + 1, lut fp16 [1,1,4,1] = {-1,0,+1,0})              |
|        |   -> constexpr_blockwise_shift_scale(scale fp16 [out, 1])                              |
| C7     | constexpr_lut_to_dense as C4 (no scale); matmul; mul by FP16(s) per output row         |
| C8     | constexpr_lut_to_dense(uint1 [2 out, in] = [P; N], lut {0, 1}); matmul; split; sub; mul|
| C6s(g) | constexpr_lut_to_sparse(mask uint1 = codes != 0, nonzero uint(log2 2g) = 2k + [c<0],   |
|        |   lut fp16 [out/g, 1, 2g, 1] = {+s_r0, -s_r0, ..., +s_r(g-1), -s_r(g-1)})              |
|        |   -> constexpr_sparse_to_dense(both outputs)                                           |
| C6d(g) | constexpr_lut_to_dense(uintN, lut fp16 [out/g, 1, 2^N, 1] = {0, +s_r0, -s_r0, ...},    |
|        |   N = 4 for g = 4 (9 used), 6 for g = 8 (17 used; 5-bit indices do not exist)           |
C5 adds int8 activation quantization (quantize -> dequantize, per-tensor symmetric, calibrated) on the
matmul input; it is exploratory (speed only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .weights import fp16_scale

ENCODER_ARMS = ("C1", "C3", "C4", "C7", "C8", "C6s2", "C6s4", "C6s8", "C6d4", "C6d8", "C5")
EXACT = {"C1", "C3", "C4", "C7", "C8", "C6s2", "C6s4", "C6s8", "C6d4", "C6d8"}
CHAINS = {
    "C1": "const fp16 (scale folded into weight)",
    "C3": "constexpr_blockwise_shift_scale(int8 codes, fp16 per-row scale)",
    "C4": "constexpr_lut_to_dense(uint2, fp16 LUT {-1,0,+1,0}) -> constexpr_blockwise_shift_scale(fp16 per-row scale)",
    "C7": "constexpr_lut_to_dense(uint2, fp16 LUT {-1,0,+1,0}); matmul; mul(fp16 per-row scale)",
    "C8": "constexpr_lut_to_dense(uint1 [P; N], fp16 LUT {0,1}); matmul; split; sub; mul(fp16 per-row scale)",
    "C6s": "constexpr_lut_to_sparse(uint1 mask, uintB nonzero indices, fp16 grouped LUT {+-s_r}) -> constexpr_sparse_to_dense",
    "C6d": "constexpr_lut_to_dense(uintB, fp16 grouped LUT {0, +-s_r})",
    "C5": "C3 weights; quantize(int8, per-tensor) -> dequantize on the activation",
    "G0": "C0's iOS16 constexpr_lut_to_dense (packed 6-bit uint8 indices, fp16 LUT[64]) verbatim",
}


def family(arm: str) -> str:
    return arm[:3] if arm.startswith("C6") else arm


def group_size(arm: str) -> int:
    return int(arm[3:]) if arm.startswith("C6") else 0


def _sub_byte(name: str):
    from coremltools.converters.mil.mil import types

    return types.nptype_from_builtin(types.string_to_builtin(name))


def packed_bytes(count: int, bits: int) -> int:
    return math.ceil(count * bits / 8)


@dataclass
class Encoded:
    """One module's constants. consts: name -> numpy array (sub-byte dtypes carry coremltools metadata)."""
    arm: str
    out: int
    inp: int
    rank: int
    consts: dict = field(default_factory=dict)
    bits: dict = field(default_factory=dict)   # const name -> element bit width as stored
    post_scale: np.ndarray | None = None       # FP16 per-row scale applied after the matmul (C7, C8)

    def nbytes(self) -> dict:
        """Stored bytes per const (sub-byte packed) plus post scale."""
        out = {k: packed_bytes(v.size, self.bits[k]) for k, v in self.consts.items()}
        if self.post_scale is not None:
            out["post_scale"] = self.post_scale.size * 2
        return out

    def describe(self) -> dict:
        """Literal shape and dtype of every constexpr input (manifest)."""
        def dt(k, v):
            return f"uint{self.bits[k]}" if v.dtype == np.uint8 and self.bits[k] < 8 else str(v.dtype)
        d = {k: {"dtype": dt(k, v), "shape": list(v.shape)} for k, v in self.consts.items()}
        if self.post_scale is not None:
            d["post_scale"] = {"dtype": "float16", "shape": list(self.post_scale.shape)}
        return d


def _shape(rank: int, *dims: int) -> tuple:
    """[out, in] weights get rank 2; pointwise convs [out, in, 1] rank 3 (the LUT/scale get a 1 inserted)."""
    return dims if rank == 2 else dims[:2] + (1,) + dims[2:]


def encode(arm: str, codes: np.ndarray, scale: np.ndarray, rank: int) -> Encoded:
    out, inp = codes.shape
    s16 = fp16_scale(scale)
    fam = family(arm)
    enc = Encoded(arm, out, inp, rank)
    w_shape = _shape(rank, out, inp)
    if fam == "C1":
        enc.consts["weight"] = (codes.astype(np.float16) * s16[:, None]).reshape(w_shape)
        enc.bits["weight"] = 16
    elif fam in ("C3", "C5"):
        enc.consts["data"] = codes.reshape(w_shape)
        enc.bits["data"] = 8
        enc.consts["scale"] = s16.reshape(_shape(rank, out, 1))
        enc.bits["scale"] = 16
    elif fam in ("C4", "C7"):
        enc.consts["indices"] = (codes + 1).astype(np.uint8).reshape(w_shape).astype(_sub_byte("uint2"))
        enc.bits["indices"] = 2
        enc.consts["lut"] = np.array([-1, 0, 1, 0], dtype=np.float16).reshape(_shape(rank, 1, 1, 4, 1))
        enc.bits["lut"] = 16
        if fam == "C4":
            enc.consts["scale"] = s16.reshape(_shape(rank, out, 1))
            enc.bits["scale"] = 16
        else:
            enc.post_scale = s16
    elif fam == "C8":
        planes = np.concatenate([(codes == 1), (codes == -1)], axis=0).astype(np.uint8)
        enc.consts["indices"] = planes.reshape(_shape(rank, 2 * out, inp)).astype(_sub_byte("uint1"))
        enc.bits["indices"] = 1
        enc.consts["lut"] = np.array([0, 1], dtype=np.float16).reshape(_shape(rank, 1, 1, 2, 1))
        enc.bits["lut"] = 16
        enc.post_scale = s16
    elif fam == "C6s":
        g = group_size(arm)
        nbits = int(math.log2(2 * g))
        mask = codes != 0
        k = (np.arange(out) % g)[:, None]
        idx = (2 * k + (codes < 0)).astype(np.uint8)
        enc.consts["indices_mask"] = mask.astype(np.uint8).reshape(w_shape).astype(_sub_byte("uint1"))
        enc.bits["indices_mask"] = 1
        enc.consts["indices_nonzero_data"] = idx[mask].astype(_sub_byte(f"uint{nbits}"))
        enc.bits["indices_nonzero_data"] = nbits
        lut = np.empty((out // g, 2 * g), dtype=np.float16)
        lut[:, 0::2] = s16.reshape(out // g, g)
        lut[:, 1::2] = -s16.reshape(out // g, g)
        enc.consts["lut"] = lut.reshape(_shape(rank, out // g, 1, 2 * g, 1))
        enc.bits["lut"] = 16
    elif fam == "C6d":
        g = group_size(arm)
        nbits = {4: 4, 8: 6}[g]
        k = (np.arange(out) % g)[:, None]
        idx = np.where(codes == 0, 0, 1 + 2 * k + (codes < 0)).astype(np.uint8)
        enc.consts["indices"] = idx.reshape(w_shape).astype(_sub_byte(f"uint{nbits}"))
        enc.bits["indices"] = nbits
        lut = np.zeros((out // g, 2 ** nbits), dtype=np.float16)
        lut[:, 1:2 * g + 1:2] = s16.reshape(out // g, g)
        lut[:, 2:2 * g + 2:2] = -s16.reshape(out // g, g)
        enc.consts["lut"] = lut.reshape(_shape(rank, out // g, 1, 2 ** nbits, 1))
        enc.bits["lut"] = 16
    else:
        raise KeyError(f"unknown arm {arm}")
    return enc


def weight_var(enc: Encoded, name: str):
    """Emit the constexpr chain (or const) of enc in the current MIL context; returns the dense weight Var."""
    from coremltools.converters.mil import Builder as mb

    fam = family(enc.arm)
    c = enc.consts
    if fam == "C1":
        return mb.const(val=c["weight"], name=name)
    if fam in ("C3", "C5"):
        return mb.constexpr_blockwise_shift_scale(data=c["data"], scale=c["scale"], name=name)
    if fam in ("C4", "C7", "C8"):
        dense = mb.constexpr_lut_to_dense(indices=c["indices"], lut=c["lut"],
                                          name=name if fam != "C4" else name + "_codes")
        if fam == "C4":
            return mb.constexpr_blockwise_shift_scale(data=dense, scale=c["scale"], name=name)
        return dense
    if fam == "C6s":
        mask, nonzero = mb.constexpr_lut_to_sparse(indices_mask=c["indices_mask"],
                                                   indices_nonzero_data=c["indices_nonzero_data"], lut=c["lut"],
                                                   name=name + "_sparse")
        return mb.constexpr_sparse_to_dense(nonzero_data=nonzero, mask=mask, name=name)
    if fam == "C6d":
        return mb.constexpr_lut_to_dense(indices=c["indices"], lut=c["lut"], name=name)
    raise KeyError(enc.arm)


def apply(enc: Encoded, x, name: str):
    """x [1, T, in] (rank-2 weights, linear) or [1, in, T] (rank-3 weights, 1x1 conv) -> output with the arm's
    post-matmul steps. (C5's activation quantization is emitted by the caller, once per shared input.)"""
    from coremltools.converters.mil import Builder as mb

    fam = family(enc.arm)
    w = weight_var(enc, name + "_weight")
    y = (mb.linear(x=x, weight=w, name=name + "_mm") if enc.rank == 2
         else mb.conv(x=x, weight=w, name=name + "_mm"))
    if enc.post_scale is None:
        return y
    axis = -1 if enc.rank == 2 else 1
    if fam == "C8":
        p, n = mb.split(x=y, num_splits=2, axis=axis, name=name + "_pn")
        y = mb.sub(x=p, y=n, name=name + "_diff")
    s = enc.post_scale if enc.rank == 2 else enc.post_scale.reshape(1, -1, 1)
    return mb.mul(x=y, y=s, name=name + "_scaled")


def quantize_activation(x, act_scale: float, name: str):
    """Per-tensor symmetric int8 fake-quantization pair (the pattern Core ML lowers to int8 activations)."""
    from coremltools.converters.mil import Builder as mb

    scale = np.float16(act_scale)
    zp = np.int8(0)
    q = mb.quantize(input=x, scale=scale, zero_point=zp, output_dtype="int8", name=name + "_aq")
    return mb.dequantize(input=q, scale=scale, zero_point=zp, name=name + "_adq")


# --- gate 2: effective matrices -----------------------------------------------------------------------------

def reference_matrix(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """codes x FP16(s), FP16 [out, in]: the bits every exact arm must reproduce."""
    return (codes.astype(np.float16) * fp16_scale(scale)[:, None])


def bits_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.ascontiguousarray(a, dtype=np.float16)
    b = np.ascontiguousarray(b, dtype=np.float16)
    return a.shape == b.shape and np.array_equal(a.view(np.uint16), b.view(np.uint16))
