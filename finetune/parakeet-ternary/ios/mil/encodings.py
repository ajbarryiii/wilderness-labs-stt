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

1-bit arms (binary models such as MB1 / kea-b: codes in {-1, +1}, never 0; encode() refuses a 0 code):
| B1L    | constexpr_lut_to_dense(uint1 = [c > 0], lut fp16 [out, 1, 2, 1] = {-s_r, +s_r})            |
| B1S    | constexpr_lut_to_dense(uint1 = [c > 0], lut fp16 [1, 1, 2, 1] = {-1, +1})                  |
|        |   -> constexpr_blockwise_shift_scale(scale fp16 [out, 1])                    (C4 analogue) |
| B1Q    | as B1S with an int8 LUT {-1, +1} (int8 dense codes into the shift/scale)                  |
| B1P    | constexpr_lut_to_dense as B1S (no scale); matmul; mul by FP16(s) per row     (C7 analogue) |
| Bg(g)  | constexpr_lut_to_dense(uintB = 2k + [c < 0], B = log2 2g, lut fp16 [out/g, 1, 2g, 1] =     |
|        |   {+s_r0, -s_r0, ..., +s_r(g-1), -s_r(g-1)}), g = 2, 4, 8 (B = 2, 3, 4)   (C6d analogue)  |
| Bi4    | constexpr_blockwise_shift_scale(data int4 codes, scale fp16 [out, 1])          (C3 analogue) |
| B1V(V) | constexpr_lut_to_dense(uintV [out/V, in] = sum_t [c(gV+t) > 0] << t, vector_axis 0,        |
|        |   lut fp16 [out/V, 1, 2^V, V]: entry e of group g = (+-s_(gV+t) by bit t of e)_t), V = 2, 4 |
|        |   (vector palettization over V rows: 1 bit per weight, row scales inside the LUT)          |

Timing-only 0/1 arms (binary models; NOT exact: the -sum(x) correction of w = 2b - 1 is deliberately left out,
so outputs are wrong; they only time a 0/1 matrix with half its entries zero, b = [c > 0]):
| Z1Q    | constexpr_lut_to_dense(uint1 = b, int8 lut {0, +1}) -> constexpr_blockwise_shift_scale(2 FP16(s)) |
|        |   (B1Q's chain with zero values in the dense matrix)                                             |
| Z1S(g) | constexpr_lut_to_sparse(mask uint1 = b, nonzero uint(log2 g) = row % g, lut fp16 [out/g, 1, g, 1] |
|        |   = {2 s_r0, ..., 2 s_r(g-1)}) -> constexpr_sparse_to_dense, g = 2, 8              (C6s analogue) |

Corrected 0/1 arm (binary models; exact in real arithmetic, not bit-exact in FP16): with w = s (2b - 1),
  W x = 2 s (b (x - m)) + m (s sum_j c_j),   m = mean of x over its features (per frame),
because b (x - m) sums a zero-mean vector, so the -sum(x - m) term of 2b - 1 vanishes. Centring keeps the 0/1
matmul's result the size of the true output (no FP16 cancellation for inputs with a large common mean).
| Z1C    | Z1Q's weight (uint1 b, int8 lut {0, +1}, shift/scale 2 FP16(s)); reduce_mean, sub before the matmul; |
|        |   add m x row_sum after, row_sum = FP16(FP16(s_r) sum_j c_rj) per output row                        |
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .weights import fp16_scale

BINARY_ARMS = ("B1L", "B1S", "B1Q", "B1P", "Bg2", "Bg4", "Bg8", "Bi4", "B1V2", "B1V4")  # 1-bit models only ({-1, +1} codes)
TIMING_ARMS = ("Z1Q", "Z1S2", "Z1S8")  # 1-bit models only; timing only (not exact, see the module table)
CORRECTED_ARMS = ("Z1C",)  # 1-bit models only; exact in real arithmetic, not in FP16
ENCODER_ARMS = (("C1", "C3", "C4", "C7", "C8", "C6s2", "C6s4", "C6s8", "C6d4", "C6d8", "C5") + BINARY_ARMS
                + TIMING_ARMS + CORRECTED_ARMS)
EXACT = {"C1", "C3", "C4", "C7", "C8", "C6s2", "C6s4", "C6s8", "C6d4", "C6d8"} | set(BINARY_ARMS)
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
    "F32": "diagnostic: FP32 graph, dense FP32 const = codes x FP16(s)",
    "B1L": "constexpr_lut_to_dense(uint1, fp16 per-row LUT {-s_r,+s_r})",
    "B1S": "constexpr_lut_to_dense(uint1, fp16 LUT {-1,+1}) -> constexpr_blockwise_shift_scale(fp16 per-row scale)",
    "B1Q": "constexpr_lut_to_dense(uint1, int8 LUT {-1,+1}) -> constexpr_blockwise_shift_scale(fp16 per-row scale)",
    "B1P": "constexpr_lut_to_dense(uint1, fp16 LUT {-1,+1}); matmul; mul(fp16 per-row scale)",
    "Bg": "constexpr_lut_to_dense(uintB, fp16 grouped LUT {+-s_r} over g rows, B = log2 2g)",
    "Bi4": "constexpr_blockwise_shift_scale(int4 +-1 codes, fp16 per-row scale)",
    "B1V": "constexpr_lut_to_dense(uintV over V rows, vector_axis 0, fp16 grouped LUT of 2^V V-vectors {+-s_r})",
    "Z1Q": "timing only: constexpr_lut_to_dense(uint1 b, int8 LUT {0,+1}) -> constexpr_blockwise_shift_scale(2 fp16 s)",
    "Z1C": "constexpr_lut_to_dense(uint1 b, int8 LUT {0,+1}) -> constexpr_blockwise_shift_scale(2 fp16 s); "
           "x - mean(x) before the matmul, + mean(x) x fp16 row_sum after",
    "Z1S": "timing only: constexpr_lut_to_sparse(uint1 mask b, uint(log2 g) row-in-group, fp16 LUT {2 s_r}) -> constexpr_sparse_to_dense",
}
BINARY_FAMILIES = {"B1L", "B1S", "B1Q", "B1P", "Bg", "Bi4", "B1V"}


def family(arm: str) -> str:
    if arm.startswith("C6") or arm.startswith("B1V") or arm.startswith("Z1S"):
        return arm[:3]
    return "Bg" if arm.startswith("Bg") else arm


def group_size(arm: str) -> int:
    if arm.startswith("C6") or arm.startswith("B1V") or arm.startswith("Z1S"):
        return int(arm[3:])
    return int(arm[2:]) if arm.startswith("Bg") else 0


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
    row_sum: np.ndarray | None = None          # per-row s_r sum_j c_rj, added times the input mean (Z1C)

    def nbytes(self) -> dict:
        """Stored bytes per const (sub-byte packed) plus post scale."""
        out = {k: packed_bytes(v.size, self.bits[k]) for k, v in self.consts.items()}
        if self.post_scale is not None:
            out["post_scale"] = self.post_scale.nbytes
        if self.row_sum is not None:
            out["row_sum"] = self.row_sum.nbytes
        return out

    def describe(self) -> dict:
        """Literal shape and dtype of every constexpr input (manifest)."""
        def dt(k, v):
            if self.bits[k] < 8 and v.dtype in (np.uint8, np.int8):
                return f"{'u' if v.dtype == np.uint8 else ''}int{self.bits[k]}"
            return str(v.dtype)
        d = {k: {"dtype": dt(k, v), "shape": list(v.shape)} for k, v in self.consts.items()}
        if self.post_scale is not None:
            d["post_scale"] = {"dtype": str(self.post_scale.dtype), "shape": list(self.post_scale.shape)}
        return d


def _shape(rank: int, *dims: int) -> tuple:
    """[out, in] weights get rank 2; pointwise convs [out, in, 1] rank 3 and 1x1 conv2d [out, in, 1, 1] rank 4
    (the ANE layout); the LUT/scale get the same 1s inserted after their first two dimensions."""
    return dims if rank == 2 else dims[:2] + (1,) * (rank - 2) + dims[2:]


def encode(arm: str, codes: np.ndarray, scale: np.ndarray, rank: int, dtype=np.float16) -> Encoded:
    """dtype float32 builds the same chain with FP32 LUT/scale/weight values (the FP16-rounded scales, exactly
    representable): the gate-4a FP32-compute build of the arm's graph topology."""
    out, inp = codes.shape
    s16 = fp16_scale(scale).astype(dtype)
    dt = dtype
    fam = family(arm)
    enc = Encoded(arm, out, inp, rank)
    w_shape = _shape(rank, out, inp)
    if fam in BINARY_FAMILIES:
        _encode_binary(enc, fam, codes, s16, dt, w_shape)
    elif fam in ("Z1Q", "Z1S", "Z1C"):
        _encode_zero_one(enc, fam, codes, s16, dt, w_shape)
    elif fam == "C1":
        enc.consts["weight"] = (codes.astype(dt) * s16[:, None]).reshape(w_shape)
        enc.bits["weight"] = 8 * np.dtype(dt).itemsize
    elif fam == "F32":  # diagnostic (mil/diag.py): FP32 graph, dense codes x FP16(s) in FP32
        enc.consts["weight"] = (codes.astype(np.float32) * s16.astype(np.float32)[:, None]).reshape(w_shape)
        enc.bits["weight"] = 32
    elif fam in ("C3", "C5"):
        enc.consts["data"] = codes.reshape(w_shape)
        enc.bits["data"] = 8
        enc.consts["scale"] = s16.reshape(_shape(rank, out, 1))
        enc.bits["scale"] = 8 * np.dtype(dt).itemsize
    elif fam in ("C4", "C7"):
        enc.consts["indices"] = (codes + 1).astype(np.uint8).reshape(w_shape).astype(_sub_byte("uint2"))
        enc.bits["indices"] = 2
        enc.consts["lut"] = np.array([-1, 0, 1, 0], dtype=dt).reshape(_shape(rank, 1, 1, 4, 1))
        enc.bits["lut"] = 8 * np.dtype(dt).itemsize
        if fam == "C4":
            enc.consts["scale"] = s16.reshape(_shape(rank, out, 1))
            enc.bits["scale"] = 8 * np.dtype(dt).itemsize
        else:
            enc.post_scale = s16
    elif fam == "C8":
        planes = np.concatenate([(codes == 1), (codes == -1)], axis=0).astype(np.uint8)
        enc.consts["indices"] = planes.reshape(_shape(rank, 2 * out, inp)).astype(_sub_byte("uint1"))
        enc.bits["indices"] = 1
        enc.consts["lut"] = np.array([0, 1], dtype=dt).reshape(_shape(rank, 1, 1, 2, 1))
        enc.bits["lut"] = 8 * np.dtype(dt).itemsize
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
        lut = np.empty((out // g, 2 * g), dtype=dt)
        lut[:, 0::2] = s16.reshape(out // g, g)
        lut[:, 1::2] = -s16.reshape(out // g, g)
        enc.consts["lut"] = lut.reshape(_shape(rank, out // g, 1, 2 * g, 1))
        enc.bits["lut"] = 8 * np.dtype(dt).itemsize
    elif fam == "C6d":
        g = group_size(arm)
        nbits = {4: 4, 8: 6}[g]
        k = (np.arange(out) % g)[:, None]
        idx = np.where(codes == 0, 0, 1 + 2 * k + (codes < 0)).astype(np.uint8)
        enc.consts["indices"] = idx.reshape(w_shape).astype(_sub_byte(f"uint{nbits}"))
        enc.bits["indices"] = nbits
        lut = np.zeros((out // g, 2 ** nbits), dtype=dt)
        lut[:, 1:2 * g + 1:2] = s16.reshape(out // g, g)
        lut[:, 2:2 * g + 2:2] = -s16.reshape(out // g, g)
        enc.consts["lut"] = lut.reshape(_shape(rank, out // g, 1, 2 ** nbits, 1))
        enc.bits["lut"] = 8 * np.dtype(dt).itemsize
    else:
        raise KeyError(f"unknown arm {arm}")
    return enc


def _encode_binary(enc: Encoded, fam: str, codes: np.ndarray, s16: np.ndarray, dt, w_shape: tuple) -> None:
    """The 1-bit arms' constants (see the module table); codes must be +-1."""
    if not np.isin(codes, (-1, 1)).all():
        raise ValueError(f"{enc.arm} encodes 1-bit models only: codes must be -1 or +1")
    out, rank = enc.out, enc.rank
    fbits = 8 * np.dtype(dt).itemsize
    if fam == "Bi4":
        enc.consts["data"] = codes.reshape(w_shape).astype(_sub_byte("int4"))
        enc.bits["data"] = 4
        enc.consts["scale"] = s16.reshape(_shape(rank, out, 1))
        enc.bits["scale"] = fbits
        return
    if fam == "B1V":
        v = group_size(enc.arm)
        if out % v:
            raise ValueError(f"{enc.arm}: {out} rows are not a multiple of the vector size {v}")
        pos = (codes > 0).astype(np.uint8).reshape(out // v, v, -1)
        idx = sum(pos[:, t, :] << t for t in range(v)).astype(np.uint8)
        enc.consts["indices"] = idx.reshape(_shape(rank, out // v, codes.shape[1])).astype(_sub_byte(f"uint{v}"))
        enc.bits["indices"] = v
        entries = np.arange(2 ** v)
        signs = np.where((entries[:, None] >> np.arange(v)[None, :]) & 1, 1, -1)       # [2^v, v]
        lut = (signs[None, :, :] * s16.reshape(out // v, 1, v)).astype(dt)            # [out/v, 2^v, v]
        enc.consts["lut"] = lut.reshape(_shape(rank, out // v, 1, 2 ** v, v))
        enc.bits["lut"] = fbits
        return
    if fam == "Bg":
        g = group_size(enc.arm)
        if out % g:
            raise ValueError(f"{enc.arm}: {out} rows are not a multiple of the group size {g}")
        nbits = int(math.log2(2 * g))
        k = (np.arange(out) % g)[:, None]
        idx = (2 * k + (codes < 0)).astype(np.uint8)
        enc.consts["indices"] = idx.reshape(w_shape).astype(_sub_byte(f"uint{nbits}"))
        enc.bits["indices"] = nbits
        lut = np.empty((out // g, 2 * g), dtype=dt)
        lut[:, 0::2] = s16.reshape(out // g, g)
        lut[:, 1::2] = -s16.reshape(out // g, g)
        enc.consts["lut"] = lut.reshape(_shape(rank, out // g, 1, 2 * g, 1))
        enc.bits["lut"] = fbits
        return
    enc.consts["indices"] = (codes > 0).astype(np.uint8).reshape(w_shape).astype(_sub_byte("uint1"))
    enc.bits["indices"] = 1
    if fam == "B1L":
        lut = np.stack([-s16, s16], axis=1).astype(dt)
        enc.consts["lut"] = lut.reshape(_shape(rank, out, 1, 2, 1))
        enc.bits["lut"] = fbits
        return
    lut_dt = np.int8 if fam == "B1Q" else dt
    enc.consts["lut"] = np.array([-1, 1], dtype=lut_dt).reshape(_shape(rank, 1, 1, 2, 1))
    enc.bits["lut"] = 8 * np.dtype(lut_dt).itemsize
    if fam == "B1P":
        enc.post_scale = s16
    else:
        enc.consts["scale"] = s16.reshape(_shape(rank, out, 1))
        enc.bits["scale"] = fbits


def _encode_zero_one(enc: Encoded, fam: str, codes: np.ndarray, s16: np.ndarray, dt, w_shape: tuple) -> None:
    """The timing-only 0/1 arms (module table): b = [c > 0] with the doubled row scale; codes must be +-1."""
    if not np.isin(codes, (-1, 1)).all():
        raise ValueError(f"{enc.arm} encodes 1-bit models only: codes must be -1 or +1")
    out, rank = enc.out, enc.rank
    b = (codes > 0).astype(np.uint8)
    s2 = (2 * s16.astype(np.float32)).astype(dt)  # exact: doubling an FP16 value
    if fam in ("Z1Q", "Z1C"):
        if fam == "Z1C":  # exact in FP32 (11-bit scale x |sum| <= 2^11), then rounded once to the graph dtype
            enc.row_sum = (s16.astype(np.float32) * codes.astype(np.int32).sum(axis=1).astype(np.float32)).astype(dt)
        enc.consts["indices"] = b.reshape(w_shape).astype(_sub_byte("uint1"))
        enc.bits["indices"] = 1
        enc.consts["lut"] = np.array([0, 1], dtype=np.int8).reshape(_shape(rank, 1, 1, 2, 1))
        enc.bits["lut"] = 8
        enc.consts["scale"] = s2.reshape(_shape(rank, out, 1))
        enc.bits["scale"] = 8 * np.dtype(dt).itemsize
        return
    g = group_size(enc.arm)
    if out % g:
        raise ValueError(f"{enc.arm}: {out} rows are not a multiple of the group size {g}")
    nbits = int(math.log2(g))
    mask = b.astype(bool)
    k = np.broadcast_to((np.arange(out) % g)[:, None], codes.shape).astype(np.uint8)
    enc.consts["indices_mask"] = b.reshape(w_shape).astype(_sub_byte("uint1"))
    enc.bits["indices_mask"] = 1
    enc.consts["indices_nonzero_data"] = k[mask].astype(_sub_byte(f"uint{nbits}"))
    enc.bits["indices_nonzero_data"] = nbits
    enc.consts["lut"] = s2.reshape(out // g, g).reshape(_shape(rank, out // g, 1, g, 1))
    enc.bits["lut"] = 8 * np.dtype(dt).itemsize


def weight_var(enc: Encoded, name: str):
    """Emit the constexpr chain (or const) of enc in the current MIL context; returns the dense weight Var."""
    from coremltools.converters.mil import Builder as mb

    fam = family(enc.arm)
    c = enc.consts
    if fam in ("C1", "F32"):
        return mb.const(val=c["weight"], name=name)
    if fam in ("C3", "C5"):
        return mb.constexpr_blockwise_shift_scale(data=c["data"], scale=c["scale"], name=name)
    if fam in ("C4", "C7", "C8"):
        dense = mb.constexpr_lut_to_dense(indices=c["indices"], lut=c["lut"],
                                          name=name if fam != "C4" else name + "_codes")
        if fam == "C4":
            return mb.constexpr_blockwise_shift_scale(data=dense, scale=c["scale"], name=name)
        return dense
    if fam in ("C6s", "Z1S"):
        mask, nonzero = mb.constexpr_lut_to_sparse(indices_mask=c["indices_mask"],
                                                   indices_nonzero_data=c["indices_nonzero_data"], lut=c["lut"],
                                                   name=name + "_sparse")
        return mb.constexpr_sparse_to_dense(nonzero_data=nonzero, mask=mask, name=name)
    if fam == "C6d":
        return mb.constexpr_lut_to_dense(indices=c["indices"], lut=c["lut"], name=name)
    if fam in ("B1L", "Bg", "B1P"):
        return mb.constexpr_lut_to_dense(indices=c["indices"], lut=c["lut"], name=name)
    if fam in ("B1S", "B1Q", "Z1Q", "Z1C"):
        dense = mb.constexpr_lut_to_dense(indices=c["indices"], lut=c["lut"], name=name + "_codes")
        return mb.constexpr_blockwise_shift_scale(data=dense, scale=c["scale"], name=name)
    if fam == "Bi4":
        return mb.constexpr_blockwise_shift_scale(data=c["data"], scale=c["scale"], name=name)
    if fam == "B1V":
        return mb.constexpr_lut_to_dense(indices=c["indices"], lut=c["lut"], vector_axis=np.int32(0), name=name)
    raise KeyError(enc.arm)


def apply(enc: Encoded, x, name: str):
    """x [1, T, in] (rank-2 weights, linear) or [1, in, T] (rank-3 weights, 1x1 conv) -> output with the arm's
    post-matmul steps. (C5's activation quantization is emitted by the caller, once per shared input.)"""
    from coremltools.converters.mil import Builder as mb

    fam = family(enc.arm)
    w = weight_var(enc, name + "_weight")
    if fam == "Z1C":
        feat = -1 if enc.rank == 2 else 1
        m = mb.reduce_mean(x=x, axes=[feat], keep_dims=True, name=name + "_mean")
        xc = mb.sub(x=x, y=m, name=name + "_centred")
        y = (mb.linear(x=xc, weight=w, name=name + "_mm") if enc.rank == 2
             else mb.conv(x=xc, weight=w, name=name + "_mm"))
        c = enc.row_sum if enc.rank == 2 else enc.row_sum.reshape((1, -1) + (1,) * (enc.rank - 2))
        return mb.add(x=y, y=mb.mul(x=m, y=c, name=name + "_corr"), name=name + "_corrected")
    y = (mb.linear(x=x, weight=w, name=name + "_mm") if enc.rank == 2
         else mb.conv(x=x, weight=w, name=name + "_mm"))
    if enc.post_scale is None:
        return y
    axis = -1 if enc.rank == 2 else 1
    if fam == "C8":
        p, n = mb.split(x=y, num_splits=2, axis=axis, name=name + "_pn")
        y = mb.sub(x=p, y=n, name=name + "_diff")
    s = enc.post_scale if enc.rank == 2 else enc.post_scale.reshape((1, -1) + (1,) * (enc.rank - 2))
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
