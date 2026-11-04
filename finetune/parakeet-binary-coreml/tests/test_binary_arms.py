"""Contract of the 1-bit encoder arms (mil/encodings.py BINARY_ARMS) on invented +-1 matrices.

Contracts per arm:
- gate 2: coremltools' own constexpr decompression of the arm's chain (and, for B1P, the post-matmul
  per-row multiplier) reproduces codes x FP16(s) bit for bit, both on the authored program and after
  conversion at iOS26 (DEFAULT pipeline, FP16), save and reload of the .mlpackage (sub-byte packing);
- the chain stays as authored after conversion (op types and literal dtypes/shapes);
- byte accounting: Encoded.nbytes() is the packed size of each const at its stored bit width;
- binary arms refuse codes outside {-1, +1}.
"""
from __future__ import annotations

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import common  # noqa: F401  (paths)
from common import synthetic_codes

OUT, IN = 16, 24
# arm -> (constexpr chain from the matmul weight back to the stored constants, post-matmul op, literal of each const)
EXPECTED = {
    "B1L": (["constexpr_lut_to_dense"], None, {"indices": ("uint1", [OUT, IN]), "lut": ("fp16", [OUT, 1, 2, 1])}),
    "B1S": (["constexpr_lut_to_dense", "constexpr_blockwise_shift_scale"], None,
            {"indices": ("uint1", [OUT, IN]), "lut": ("fp16", [1, 1, 2, 1]), "scale": ("fp16", [OUT, 1])}),
    "B1Q": (["constexpr_lut_to_dense", "constexpr_blockwise_shift_scale"], None,
            {"indices": ("uint1", [OUT, IN]), "lut": ("int8", [1, 1, 2, 1]), "scale": ("fp16", [OUT, 1])}),
    "B1P": (["constexpr_lut_to_dense"], "mul", {"indices": ("uint1", [OUT, IN]), "lut": ("fp16", [1, 1, 2, 1])}),
    "Bg2": (["constexpr_lut_to_dense"], None, {"indices": ("uint2", [OUT, IN]), "lut": ("fp16", [OUT // 2, 1, 4, 1])}),
    "Bg4": (["constexpr_lut_to_dense"], None, {"indices": ("uint3", [OUT, IN]), "lut": ("fp16", [OUT // 4, 1, 8, 1])}),
    "Bg8": (["constexpr_lut_to_dense"], None, {"indices": ("uint4", [OUT, IN]), "lut": ("fp16", [OUT // 8, 1, 16, 1])}),
    "Bi4": (["constexpr_blockwise_shift_scale"], None, {"data": ("int4", [OUT, IN]), "scale": ("fp16", [OUT, 1])}),
    # vector palettization along the output axis: one index per V rows x 1 column, LUT of 2^V V-vectors per group
    "B1V2": (["constexpr_lut_to_dense"], None, {"indices": ("uint2", [OUT // 2, IN]), "lut": ("fp16", [OUT // 2, 1, 4, 2])}),
    "B1V4": (["constexpr_lut_to_dense"], None, {"indices": ("uint4", [OUT // 4, IN]), "lut": ("fp16", [OUT // 4, 1, 16, 4])}),
}


def expected_bytes(arm: str, out: int, inp: int) -> dict:
    n = out * inp
    if arm == "B1L":
        return {"indices": math.ceil(n / 8), "lut": out * 2 * 2}
    if arm == "B1S":
        return {"indices": math.ceil(n / 8), "lut": 2 * 2, "scale": out * 2}
    if arm == "B1Q":
        return {"indices": math.ceil(n / 8), "lut": 2, "scale": out * 2}
    if arm == "B1P":
        return {"indices": math.ceil(n / 8), "lut": 2 * 2, "post_scale": out * 2}
    if arm.startswith("Bg"):
        g = int(arm[2:])
        return {"indices": math.ceil(n * int(math.log2(2 * g)) / 8), "lut": (out // g) * 2 * g * 2}
    if arm == "Bi4":
        return {"data": math.ceil(n * 4 / 8), "scale": out * 2}
    if arm.startswith("B1V"):
        v = int(arm[3:])
        return {"indices": math.ceil(n / 8), "lut": (out // v) * 2 ** v * v * 2}
    raise KeyError(arm)


def fixture(seed: int = 1):
    rng = np.random.default_rng(seed)
    codes = synthetic_codes(rng, OUT, IN)
    codes[0] = 1   # an all +1 and an all -1 row
    codes[1] = -1
    # scales across the FP16 normal range, including values that FP32 -> FP16 rounding changes
    scale = (rng.random(OUT) * 0.2 + 1e-3).astype(np.float32)
    scale[2] = np.float32(6.2e-5)  # just above the smallest normal FP16
    scale[3] = np.float32(3.14159e2)
    return codes, scale


def program(enc, rank: int):
    """One matmul with the arm's weight (linear for rank 2, 1x1 conv for rank 3), iOS26, FP16 inside."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from mil import encodings

    shape = (1, 5, IN) if rank == 2 else (1, IN, 5)

    @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp32)], opset_version=ct.target.iOS26)
    def prog(x):
        y = encodings.apply(enc, mb.cast(x=x, dtype="fp16"), "m")
        return mb.cast(x=y, dtype="fp32", name="y")

    return prog


def effective(mm_op, fam_post: str | None) -> np.ndarray:
    from mil import probes

    w = probes._dense(mm_op.weight).astype(np.float16)
    w = w.reshape(w.shape[0], -1)
    if fam_post is None:
        return w
    mul = next(c for c in mm_op.outputs[0].child_ops if c.op_type == "mul")
    s = (mul.y if mul.x is mm_op.outputs[0] else mul.x).val.reshape(-1).astype(np.float16)
    return (w.astype(np.float32) * s.astype(np.float32)[:, None]).astype(np.float16)


def matmul_ops(prog):
    return [op for op in next(iter(prog.functions.values())).operations if op.op_type in ("linear", "conv")]


class BinaryArmTests(unittest.TestCase):
    def test_registered(self):
        from mil import encodings

        self.assertEqual(set(encodings.BINARY_ARMS), set(EXPECTED))
        for arm in EXPECTED:
            self.assertIn(arm, encodings.ENCODER_ARMS)
            self.assertIn(arm, encodings.EXACT)
            self.assertIn(encodings.family(arm), encodings.CHAINS)

    def test_refuses_non_binary_codes(self):
        from mil import encodings

        codes, scale = fixture()
        codes[4, 5] = 0
        for arm in EXPECTED:
            with self.assertRaises(ValueError, msg=arm):
                encodings.encode(arm, codes, scale, 2)

    def test_byte_accounting_and_literals(self):
        from mil import encodings

        codes, scale = fixture()
        for arm, (_, _, literals) in EXPECTED.items():
            enc = encodings.encode(arm, codes, scale, 2)
            self.assertEqual(enc.nbytes(), expected_bytes(arm, OUT, IN), arm)
            desc = enc.describe()
            for k, (dt, shape) in literals.items():
                want = {"fp16": "float16", "int8": "int8"}.get(dt, dt)
                self.assertEqual(desc[k], {"dtype": want, "shape": shape}, f"{arm} {k}")

    def test_gate2_authored_program(self):
        from mil import encodings

        codes, scale = fixture()
        ref = encodings.reference_matrix(codes, scale)
        for arm, (_, post, _) in EXPECTED.items():
            for rank in (2, 3):
                enc = encodings.encode(arm, codes, scale, rank)
                (mm,) = matmul_ops(program(enc, rank))
                self.assertTrue(encodings.bits_equal(effective(mm, post), ref), f"{arm} rank {rank}")

    def test_gate2_and_chain_after_conversion_save_and_reload(self):
        import coremltools as ct
        from coremltools.converters.mil.mil import types
        from mil import encodings, probes

        codes, scale = fixture(2)
        ref = encodings.reference_matrix(codes, scale)
        tmp = Path(tempfile.mkdtemp())
        try:
            for arm, (chain, post, literals) in EXPECTED.items():
                for rank in (2, 3):
                    enc = encodings.encode(arm, codes, scale, rank)
                    model = ct.convert(program(enc, rank), convert_to="mlprogram",
                                       minimum_deployment_target=ct.target.iOS26,
                                       compute_precision=ct.precision.FLOAT16, pass_pipeline=ct.PassPipeline.DEFAULT,
                                       compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
                    pkg = tmp / f"{arm}-{rank}.mlpackage"
                    model.save(str(pkg))
                    loaded = probes.load_program(pkg)
                    (mm,) = matmul_ops(loaded)
                    self.assertEqual(probes._chain(mm.weight), chain, f"{arm} rank {rank}")
                    children = [c.op_type for c in mm.outputs[0].child_ops]
                    self.assertEqual("mul" in children, post == "mul", f"{arm} rank {rank}: {children}")
                    stored = {}
                    op = mm.weight.op
                    while op is not None and op.op_type.startswith("constexpr_"):
                        for k, v in op.inputs.items():
                            if hasattr(v, "dtype") and v.op is not None and v.op.op_type == "const":
                                stored[k] = (types.builtin_to_string(v.dtype), [int(d) for d in v.shape])
                        nxt = [v for v in op.inputs.values() if hasattr(v, "op") and v.op is not None
                               and v.op.op_type.startswith("constexpr_")]
                        op = nxt[0].op if nxt else None
                    for k, (dt, shape) in literals.items():
                        key = {"indices": "indices", "lut": "lut", "scale": "scale", "data": "data"}[k]
                        want_shape = shape if rank == 2 else shape[:2] + [1] + shape[2:]
                        self.assertEqual(stored.get(key), (dt, want_shape), f"{arm} rank {rank} {k}: {stored}")
                    self.assertTrue(encodings.bits_equal(effective(mm, post), ref), f"{arm} rank {rank}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)



class TimingArmTests(unittest.TestCase):
    """Timing-only 0/1 arms (encodings.TIMING_ARMS): the dense weight is b x 2 FP16(s), b = [c > 0] (the -sum(x)
    correction of w = 2b - 1 is not emitted); never registered as exact; 1-bit codes only."""

    def test_registered_not_exact(self):
        from mil import encodings

        self.assertEqual(set(encodings.TIMING_ARMS), {"Z1Q", "Z1S2", "Z1S8"})
        for arm in encodings.TIMING_ARMS:
            self.assertIn(arm, encodings.ENCODER_ARMS)
            self.assertNotIn(arm, encodings.EXACT)
            self.assertIn(encodings.family(arm), encodings.CHAINS)

    def test_refuses_non_binary_codes(self):
        from mil import encodings

        codes, scale = fixture()
        codes[4, 5] = 0
        for arm in encodings.TIMING_ARMS:
            with self.assertRaises(ValueError, msg=arm):
                encodings.encode(arm, codes, scale, 2)

    def test_dense_weight_is_zero_one_times_doubled_scale(self):
        from mil import encodings
        from mil.weights import fp16_scale

        codes, scale = fixture(3)
        s16 = fp16_scale(scale).astype(np.float16)
        want = np.where(codes > 0, (2 * s16.astype(np.float32)).astype(np.float16)[:, None], np.float16(0))
        for arm in encodings.TIMING_ARMS:
            for rank in (2, 3):
                enc = encodings.encode(arm, codes, scale, rank)
                (mm,) = matmul_ops(program(enc, rank))
                self.assertTrue(encodings.bits_equal(effective(mm, None), want), f"{arm} rank {rank}")
            if arm.startswith("Z1S"):
                self.assertEqual(enc.consts["indices_nonzero_data"].size, int((codes > 0).sum()), arm)



def run_matmul(enc, rank: int, x: np.ndarray, fp16: bool) -> np.ndarray:
    """Convert one arm-encoded matmul and run it on the CPU (FP16 or FP32 compute); x is [1, T, IN] for rank 2."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from mil import encodings

    xin = x if rank == 2 else np.ascontiguousarray(x.transpose(0, 2, 1))

    @mb.program(input_specs=[mb.TensorSpec(shape=xin.shape, dtype=types.fp32)], opset_version=ct.target.iOS26)
    def prog(v):
        h = mb.cast(x=v, dtype="fp16") if fp16 else v
        y = encodings.apply(enc, h, "m")
        return mb.cast(x=y, dtype="fp32", name="y") if fp16 else mb.identity(x=y, name="y")

    model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS26,
                       compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
                       compute_units=ct.ComputeUnit.CPU_ONLY)
    y = np.asarray(model.predict({next(iter(prog.functions["main"].inputs)): xin})["y"], np.float64)
    return y if rank == 2 else y.transpose(0, 2, 1)


@unittest.skipUnless(sys.platform == "darwin", "runs Core ML predictions")
class CorrectedArmTests(unittest.TestCase):
    """Z1C: the 1-bit weight stored as b = [c > 0] (Z1Q's chain, half of every matrix zero) with the centred
    correction, y = 2 FP16(s) (b (x - m)) + m (FP16(s) sum_j c_j), m = the input's mean over its features.
    Equal to codes x FP16(s) applied to x in exact arithmetic; not bit-exact in FP16 (never registered exact)."""

    def test_registered_not_exact(self):
        from mil import encodings

        self.assertIn("Z1C", encodings.CORRECTED_ARMS)
        self.assertIn("Z1C", encodings.ENCODER_ARMS)
        self.assertNotIn("Z1C", encodings.EXACT)
        self.assertIn(encodings.family("Z1C"), encodings.CHAINS)
        codes, scale = fixture()
        codes[4, 5] = 0
        with self.assertRaises(ValueError):
            encodings.encode("Z1C", codes, scale, 2)

    def test_stored_bytes_are_one_bit_plus_row_constants(self):
        from mil import encodings

        codes, scale = fixture()
        enc = encodings.encode("Z1C", codes, scale, 2)
        self.assertEqual(enc.nbytes(), {"indices": math.ceil(OUT * IN / 8), "lut": 2, "scale": OUT * 2,
                                        "row_sum": OUT * 2})

    def test_fp32_equals_the_binary_matmul(self):
        from mil import encodings
        from mil.weights import fp16_scale

        codes, scale = fixture(4)
        w = codes.astype(np.float64) * fp16_scale(scale).astype(np.float64)[:, None]
        rng = np.random.default_rng(5)
        x = (rng.standard_normal((1, 7, IN)) * 3 + 20).astype(np.float32)  # large common mean
        want = x.astype(np.float64) @ w.T
        for rank in (2, 3):
            enc = encodings.encode("Z1C", codes, scale, rank, dtype=np.float32)
            got = run_matmul(enc, rank, x, fp16=False)
            np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * np.abs(want).max(), err_msg=f"rank {rank}")

    def test_fp16_error_comparable_to_b1q_with_large_mean_inputs(self):
        from mil import encodings
        from mil.weights import fp16_scale

        # a realistic layer shape and scale range (1024 inputs, s ~ 0.01-0.03): with a common input mean of 40
        # the uncorrected 0/1 sum is ~30x the true output, the FP16 cancellation case
        rng = np.random.default_rng(7)
        inp = 1024
        codes = synthetic_codes(rng, OUT, inp)
        scale = (rng.random(OUT) * 0.02 + 0.01).astype(np.float32)
        w = codes.astype(np.float64) * fp16_scale(scale).astype(np.float64)[:, None]
        for mean in (0.0, 40.0):
            x = (rng.standard_normal((1, 9, inp)) + mean).astype(np.float16).astype(np.float32)
            want = x.astype(np.float64) @ w.T
            err = {}
            for arm in ("B1Q", "Z1C", "Z1Q"):
                got = run_matmul(encodings.encode(arm, codes, scale, 2), 2, x, fp16=True)
                err[arm] = float(np.linalg.norm(got - want) / np.linalg.norm(want))
            self.assertLess(err["Z1C"], max(2 * err["B1Q"], 2e-3), f"mean {mean}: {err}")
            self.assertGreater(err["Z1Q"], 0.1, f"mean {mean}: the uncorrected 0/1 arm must be wrong ({err})")


if __name__ == "__main__":
    unittest.main()
