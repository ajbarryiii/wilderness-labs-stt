"""Contract tests (CPU): the 8-bit parts of stage 4 step 0 (keab-prime plan "Step 0" and "Use", "Code and review":
8-bit parts).

- INT8 per-channel codes and scales against an independent NumPy reference, including an all-zero channel, the
  1e-30 floor and rounding ties; W8 = fp16(q * s_c);
- exact FP16 -> FP32 widening of every finite FP16 value;
- the frozen decoder quantized once (and refused a second time), in place;
- pre_encode.out computing with W8 of its latent weight, with a straight-through gradient to it;
- under S2, binary modules computing with fp32(fp16(alpha)), with a straight-through gradient;
- the scope parts of the real module names, the Kea-B and decoder-only scopes;
- palettization (informational): at most 256 FP16 values per tensor, deterministic, exact on <= 256 values;
- the x1 per-frame activation quantizer emulated for the informational W8A8 measurement.
"""
from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from decoder_fixtures import tiny_model
from training import eightbit as eb


def np_int8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent reference: (q int8, s float32, W8 float16) of a float32 matrix."""
    w = w.astype(np.float32)
    s = np.maximum(np.abs(w).max(axis=1) / np.float32(127.0), np.float32(1e-30)).astype(np.float32)
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)          # rint: round half to even
    w8 = (q.astype(np.float32) * s[:, None]).astype(np.float32).astype(np.float16)
    return q, s, w8


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view({torch.float16: torch.int16, torch.float32: torch.int32,
                                torch.bfloat16: torch.int16}[t.dtype])


class Int8Format(unittest.TestCase):
    def test_hand_example_ties_zero_channel_and_floor(self):
        w = torch.tensor([[127.0, 2.5, 3.5, -2.5, 0.5, -1.5],      # s = 1: ties go to even
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],          # all-zero channel
                          [-254.0, 1.0, 3.0, 0.0, -1.0, 2.0],      # s = 2, negative maximum
                          [1e-36, -1e-36, 0.0, 0.0, 0.0, 0.0]])    # max/127 below the floor
        q, s = eb.int8_quantize(w)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(s.dtype, torch.float32)
        self.assertEqual(q.tolist(), [[127, 2, 4, -2, 0, -2], [0] * 6, [-127, 0, 2, 0, 0, 1], [0] * 6])
        self.assertEqual(s[0].item(), 1.0)
        self.assertEqual(s[2].item(), 2.0)
        floor = torch.tensor(1e-30, dtype=torch.float32)
        self.assertTrue(torch.equal(s[1], floor) and torch.equal(s[3], floor))
        w8 = eb.w8(q, s)
        self.assertEqual(w8.dtype, torch.float16)
        self.assertTrue(torch.equal(w8[1], torch.zeros(6, dtype=torch.float16)))
        self.assertTrue(torch.equal(w8[3], torch.zeros(6, dtype=torch.float16)))
        self.assertFalse(torch.signbit(w8[1]).any(), "an all-zero channel stores +0")

    def test_random_matrices_match_the_numpy_reference_bit_for_bit(self):
        g = torch.Generator().manual_seed(1)
        for rows, cols, scale in ((7, 13, 1.0), (64, 640, 0.02), (33, 4096, 3e-4), (5, 9, 1e4)):
            w = torch.randn(rows, cols, generator=g) * scale
            w[rows // 2] = 0.0
            q, s = eb.int8_quantize(w)
            rq, rs, rw8 = np_int8(w.numpy())
            self.assertTrue(np.array_equal(q.numpy(), rq))
            self.assertTrue(np.array_equal(s.numpy().view(np.int32), rs.view(np.int32)))
            self.assertTrue(np.array_equal(eb.w8(q, s).numpy().view(np.int16), rw8.view(np.int16)))
            self.assertEqual(int(q.abs().max()), 127)
            nz = w.abs().amax(1) > 0
            self.assertTrue(torch.all(q.abs().amax(1)[nz] == 127), "every nonzero channel's maximum maps to 127")

    def test_w8_is_the_fp16_rounding_of_the_fp32_product(self):
        q = torch.tensor([[3, -127, 1, 0]], dtype=torch.int8)
        s = torch.tensor([1.0 / 3.0], dtype=torch.float32)
        want = (q.to(torch.float32) * s[:, None]).to(torch.float16)
        self.assertTrue(torch.equal(bits(eb.w8(q, s)), bits(want)))
        self.assertTrue(torch.equal(eb.w8_of(torch.tensor([[127.0, -1.0]])),
                                    torch.tensor([[127.0, -1.0]], dtype=torch.float16)))   # s = 1: exact

    def test_input_is_quantized_in_fp32_and_must_be_a_matrix(self):
        w = torch.randn(4, 8, dtype=torch.float64)
        q, s = eb.int8_quantize(w)
        rq, rs, _ = np_int8(w.float().numpy())
        self.assertTrue(np.array_equal(q.numpy(), rq) and np.array_equal(s.numpy(), rs))
        with self.assertRaises(ValueError):
            eb.int8_quantize(torch.randn(8))
        with self.assertRaises(ValueError):
            eb.int8_quantize(torch.tensor([[float("nan"), 1.0]]))


class Widening(unittest.TestCase):
    def test_every_finite_fp16_value_widens_exactly(self):
        allbits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
        half = allbits.view(torch.float16)
        finite = torch.isfinite(half)
        self.assertEqual(int(finite.sum()), 63488)
        wide = eb.widen(half[finite])
        self.assertEqual(wide.dtype, torch.float32)
        self.assertTrue(torch.equal(bits(wide.to(torch.float16)), bits(half[finite])))
        exact = half[finite].numpy().astype(np.float64)
        self.assertTrue(np.array_equal(wide.numpy().astype(np.float64), exact))
        with self.assertRaises(ValueError):
            eb.widen(torch.ones(2, dtype=torch.float32))


class FrozenDecoder(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model(seed=3)
        self.state = {k: v.clone() for k, v in self.model.state_dict().items()}

    def test_scope_parts_are_the_real_module_names(self):
        self.assertEqual(eb.PARTS["prediction_network"], (
            "decoder.prediction.embed.weight",
            "decoder.prediction.dec_rnn.lstm.weight_ih_l0", "decoder.prediction.dec_rnn.lstm.weight_hh_l0",
            "decoder.prediction.dec_rnn.lstm.weight_ih_l1", "decoder.prediction.dec_rnn.lstm.weight_hh_l1"))
        self.assertEqual(eb.PARTS["joint"], ("joint.enc.weight", "joint.pred.weight", "joint.joint_net.2.weight"))
        self.assertEqual(eb.PARTS["pre_encode_out"], ("encoder.pre_encode.out.weight",))
        self.assertEqual(eb.check_parts(self.model), {p: list(k) for p, k in eb.PARTS.items()})
        self.assertEqual(eb.KEAB_SCOPES, {"none": (), "S1": ("prediction_network",),
                                          "S2": ("prediction_network", "joint", "pre_encode_out")})
        self.assertEqual(eb.DECODER_ONLY_SCOPES, {"none": (), "S1": ("prediction_network",),
                                                  "S2": ("prediction_network", "joint")})

    def test_quantized_once_in_place_with_the_codes_recorded(self):
        params = {k: self.model.get_parameter(k) for k in eb.keys_of(("prediction_network", "joint"))}
        record = eb.quantize_frozen(self.model, ("prediction_network", "joint"))
        self.assertEqual(sorted(record), sorted(params))
        for key, p in params.items():
            self.assertIs(self.model.get_parameter(key), p, "the Parameter object is kept (LSTM flat weights)")
            q, s = eb.int8_quantize(self.state[key])
            self.assertTrue(torch.equal(record[key][0], q) and torch.equal(record[key][1], s))
            self.assertEqual(p.dtype, torch.float32)
            self.assertTrue(torch.equal(p.detach(), eb.w8(q, s).float()))
        for key, value in self.model.state_dict().items():           # nothing else changed (biases included)
            if key not in params:
                self.assertTrue(torch.equal(value, self.state[key]), key)
        with self.assertRaises(ValueError):
            eb.quantize_frozen(self.model, ("prediction_network",))
        with self.assertRaises(ValueError):
            eb.palettize_frozen(self.model, ("joint",))
        self.assertEqual(sorted(eb.frozen_int8(self.model)), sorted(params))

    def test_decoder_only_scopes(self):
        eb.apply_decoder_scope(self.model, "S2", "int8")
        self.assertEqual(sorted(eb.frozen_int8(self.model)), sorted(eb.keys_of(("prediction_network", "joint"))))
        pre = self.model.get_parameter("encoder.pre_encode.out.weight")
        self.assertTrue(torch.equal(pre.detach(), self.state["encoder.pre_encode.out.weight"]))
        other = tiny_model(seed=3)
        eb.apply_decoder_scope(other, "none", "int8")
        self.assertEqual(eb.frozen_int8(other), {})
        with self.assertRaises(ValueError):
            eb.apply_decoder_scope(tiny_model(seed=3), "S3", "int8")


class PreEncodeOut(unittest.TestCase):
    def test_forward_uses_w8_and_the_gradient_passes_straight_through(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(32, 12)
        wrapped = eb.W8Linear.from_linear(linear)
        self.assertIs(wrapped.weight, linear.weight)
        self.assertIs(wrapped.bias, linear.bias)
        x = torch.randn(3, 5, 32, requires_grad=True)
        y = wrapped(x)
        w8 = eb.w8_of(linear.weight).float().requires_grad_(True)
        x2 = x.detach().clone().requires_grad_(True)
        ref = F.linear(x2, w8, linear.bias.detach())
        self.assertTrue(torch.equal(y, ref))
        self.assertTrue(torch.equal(wrapped.forward_weight(), eb.w8_of(linear.weight).float()))
        g = torch.randn_like(y)
        y.backward(g)
        ref.backward(g)
        self.assertTrue(torch.equal(linear.weight.grad, w8.grad), "identity gradient to the latent weight")
        self.assertTrue(torch.equal(x.grad, x2.grad))
        q, s = wrapped.int8()
        rq, rs = eb.int8_quantize(linear.weight)
        self.assertTrue(torch.equal(q, rq) and torch.equal(s, rs))

    def test_state_dict_keys_are_unchanged(self):
        model = tiny_model(seed=1)
        keys = list(model.state_dict())
        eb.wrap_pre_encode_out(model)
        self.assertIsInstance(model.encoder.pre_encode.out, eb.W8Linear)
        self.assertEqual(list(model.state_dict()), keys)
        with self.assertRaises(ValueError):
            eb.wrap_pre_encode_out(model)


class HalfScale(unittest.TestCase):
    def setUp(self):
        import binary_quant
        self.bq = binary_quant
        torch.manual_seed(2)

    def check(self, module, matrix, x, call):
        codes, scale = self.bq.binary_quantize(matrix)
        alpha = scale.to(torch.float16).to(torch.float32)
        weight = self.bq.dequantize(codes, alpha)
        self.assertTrue(torch.equal(module.forward_weight(), weight))
        got_codes, got_scale = module.quantized_weight()
        self.assertTrue(torch.equal(got_codes, codes))
        self.assertTrue(torch.equal(got_scale, alpha))
        self.assertTrue(torch.equal(got_scale.to(torch.float16).float(), got_scale), "exactly representable in FP16")
        self.assertFalse(torch.equal(alpha, scale), "the test needs a scale that FP16 rounds")
        y = module(x)
        w_ref = weight.clone().requires_grad_(True)
        ref = call(x, w_ref)
        self.assertTrue(torch.equal(y, ref))
        g = torch.randn_like(y)
        y.backward(g)
        ref.backward(g)
        grad = module.weight.grad.squeeze(2) if module.weight.grad.dim() == 3 else module.weight.grad
        self.assertTrue(torch.equal(grad, w_ref.grad), "identity gradient to the latent weight")

    def test_linear(self):
        w = torch.nn.Parameter(torch.randn(16, 24) * 0.013)
        m = eb.HalfScaleBinaryLinear(w, None, "fp32")
        self.assertIsInstance(m, self.bq.BinaryLinear)
        self.check(m, w, torch.randn(2, 3, 24), lambda x, wt: F.linear(x, wt))

    def test_pointwise_conv(self):
        w = torch.nn.Parameter(torch.randn(16, 24, 1) * 0.021)
        m = eb.HalfScaleBinaryPointwiseConv1d(w, None, "fp32")
        self.assertIsInstance(m, self.bq.BinaryPointwiseConv1d)
        self.check(m, w.squeeze(2), torch.randn(2, 24, 7), lambda x, wt: F.conv1d(x, wt.unsqueeze(2)))

    def test_fraction_below_one_and_fp16_underflow_are_refused(self):
        m = eb.HalfScaleBinaryLinear(torch.nn.Parameter(torch.randn(4, 8)), None, "fp32")
        m.fraction = 0.5
        with self.assertRaises(ValueError):
            m(torch.randn(1, 8))
        tiny = eb.HalfScaleBinaryLinear(torch.nn.Parameter(torch.full((2, 8), 1e-9)), None, "fp32")
        with self.assertRaises(ValueError):
            tiny.quantized_weight()

    def test_swap_keeps_parameters(self):
        layer = self.bq.BinaryLinear(torch.nn.Parameter(torch.randn(8, 16)), None, "ternary")
        model = torch.nn.Module()
        model.add_module("lin", layer)
        self.assertEqual(eb.half_scale_binary(model), 1)
        self.assertIsInstance(model.lin, eb.HalfScaleBinaryLinear)
        self.assertIs(model.lin.weight, layer.weight)
        self.assertEqual(model.lin.ramp, "ternary")
        with self.assertRaises(ValueError):
            eb.half_scale_binary(model)


class KeabScope(unittest.TestCase):
    def binary_model(self):
        model = tiny_model(seed=4)
        eb.binarize(model)
        return model

    def test_s2_training_form(self):
        model = self.binary_model()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        form = eb.apply_keab_scope(model, "S2")
        self.assertEqual(form, {"scope": "S2", "frozen": sorted(eb.keys_of(("prediction_network", "joint"))),
                                "pre_encode_out": "w8_ste", "binary_scales": "fp16"})
        self.assertEqual(eb.keab_form(model), form)
        self.assertIsInstance(model.encoder.pre_encode.out, eb.W8Linear)
        self.assertTrue(torch.equal(model.encoder.pre_encode.out.weight.detach(),
                                    before["encoder.pre_encode.out.weight"]), "the latent weight is kept")
        import binary_quant
        names = binary_quant.quantized_module_names(model)
        self.assertEqual(len(names), 22)
        self.assertTrue(all(type(model.get_submodule(n)) in eb.HALF_SCALE_TYPES for n in names))
        with self.assertRaises(ValueError):
            eb.apply_keab_scope(model, "S1")

    def test_s1_and_none_forms(self):
        m1 = self.binary_model()
        self.assertEqual(eb.apply_keab_scope(m1, "S1"),
                         {"scope": "S1", "frozen": sorted(eb.keys_of(("prediction_network",))),
                          "pre_encode_out": "fp", "binary_scales": "fp32"})
        m0 = self.binary_model()
        self.assertEqual(eb.apply_keab_scope(m0, "none"),
                         {"scope": "none", "frozen": [], "pre_encode_out": "fp", "binary_scales": "fp32"})

    def test_keab_scope_needs_binary_modules_at_fraction_one(self):
        with self.assertRaises(ValueError):
            eb.apply_keab_scope(tiny_model(seed=4), "S1")
        model = self.binary_model()
        import binary_quant
        binary_quant.set_weight_fraction(model, 0.5)
        with self.assertRaises(ValueError):
            eb.apply_keab_scope(model, "S1")

    def test_palette_form(self):
        model = self.binary_model()
        form = eb.apply_keab_palette(model, "S2")
        self.assertEqual(form["scope"], "S2")
        self.assertEqual(form["binary_scales"], "fp16")
        self.assertEqual(sorted(eb.frozen_palette(model)), sorted(eb.keys_of(eb.KEAB_SCOPES["S2"])))
        for key in eb.keys_of(eb.KEAB_SCOPES["S2"]):
            w = model.get_parameter(key).detach()
            self.assertLessEqual(len(torch.unique(w)), 256)
            self.assertTrue(torch.equal(w.to(torch.float16).float(), w))


class Palettization(unittest.TestCase):
    def test_at_most_256_fp16_values_deterministic(self):
        g = torch.Generator().manual_seed(5)
        w = torch.randn(300, 70, generator=g) * 0.05
        p1, info = eb.palettize(w)
        p2, _ = eb.palettize(w.clone())
        self.assertEqual(p1.dtype, torch.float16)
        self.assertEqual(p1.shape, w.shape)
        self.assertTrue(torch.equal(bits(p1), bits(p2)))
        self.assertLessEqual(len(torch.unique(p1)), 256)
        self.assertLessEqual(info["entries"], 256)
        self.assertTrue(info["converged"])

    def test_close_to_the_mse_optimum_on_a_layer_sized_tensor(self):
        w = torch.randn(640, 640, generator=torch.Generator().manual_seed(8)).double()
        p, info = eb.palettize(w.float())
        self.assertTrue(info["converged"])
        mse = (p.double() - w).pow(2).mean().item()
        lloyd_max = 3 ** 0.5 * torch.pi / 2 / 256 ** 2          # asymptotic optimum for a unit Gaussian
        self.assertLess(mse, 1.1 * lloyd_max)
        lo, hi = w.min().item(), w.max().item()
        step = (hi - lo) / 255
        grid = torch.round((w - lo) / step) * step + lo
        self.assertLess(mse, (grid - w).pow(2).mean().item(), "k-means beats a uniform 256-level grid")

    def test_every_entry_is_used_on_a_peaked_tensor(self):
        w = torch.randn(640, 640, generator=torch.Generator().manual_seed(9)) ** 5       # most mass in a few bins
        _, info = eb.palettize(w)
        self.assertEqual(info["entries"], 256)

    def test_exact_on_few_values(self):
        vals = torch.tensor([-0.5, -0.125, 0.0, 0.25, 1.0])
        w = vals[torch.randint(0, 5, (40, 9), generator=torch.Generator().manual_seed(6))]
        p, info = eb.palettize(w)
        self.assertTrue(torch.equal(p.float(), w))
        self.assertEqual(info["entries"], 5)

    def test_converged_centroids_are_the_means_of_their_members(self):
        w = torch.randn(1000, generator=torch.Generator().manual_seed(7)).reshape(10, 100)
        lut, idx, info = eb.palette_table(w)
        self.assertTrue(info["converged"])
        x = w.double().flatten()
        for j in range(len(lut)):
            members = x[idx.flatten().long() == j]
            if len(members):
                self.assertAlmostEqual(float(info["centroids"][j]), float(members.mean()), places=12)


class X1ActivationEmulation(unittest.TestCase):
    def test_zero_rows_floor_ties_and_axis(self):
        x = torch.tensor([[[0.0, 0.0, 0.0, 0.0],
                           [127.0, 2.5, -3.5, 0.5],
                           [1e-36, 0.0, -1e-36, 0.0]]])
        y = eb.x1_fake_quant(x)
        self.assertTrue(torch.equal(y[0, 0], torch.zeros(4)))
        self.assertEqual(y[0, 1].tolist(), [127.0, 2.0, -4.0, 0.0])
        self.assertTrue(torch.equal(y[0, 2], torch.zeros(4)))
        z = torch.randn(2, 5, 64)
        z[1, 3] *= 1000
        y = eb.x1_fake_quant(z)
        s = (z.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-30)
        self.assertTrue(torch.equal(y, torch.round(z / s) * s), "per frame across the last (channel) axis")

    def test_bf16_in_bf16_out_computed_in_fp32(self):
        x = (torch.randn(3, 64) * 4).to(torch.bfloat16)
        y = eb.x1_fake_quant(x)
        self.assertEqual(y.dtype, torch.bfloat16)
        x32 = x.float()
        s = (x32.abs().amax(-1, keepdim=True) / 127.0).clamp_min(1e-30)
        self.assertTrue(torch.equal(bits(y), bits((torch.round(x32 / s) * s).to(torch.bfloat16))))


if __name__ == "__main__":
    unittest.main()
