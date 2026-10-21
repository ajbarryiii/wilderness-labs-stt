"""Contract tests 1-5 (plans/pilot.md "Contract tests"): binary quantizer, STE, the two ramps
and the quantized set. Standalone modules for the math; a tiny NeMo model for the
ternary-ramp equivalence on a whole encoder; the full-size architecture (random weights) for
the 264-module set.

Run (from finetune/parakeet-binary): ./python -m unittest discover -s tests -v (via ./heavy)
"""
from __future__ import annotations

import copy
import re
import unittest

import binary_fixtures as fx
import torch
import torch.nn.functional as F
from torch import nn

import binary_quant
import quant
from binary_quant import (BinaryLinear, BinaryPointwiseConv1d, binary_quantize, code_histogram,
                          parameter_accounting, quantize_parakeet, quantized_module_names,
                          set_weight_fraction, weight_fraction)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _pair(kind: str, ramp: str, seed: int = 0):
    """(float reference, binary module holding a copy of its parameters, input)."""
    torch.manual_seed(seed)
    if kind == "linear":
        ref = nn.Linear(13, 6, bias=True)
        x = torch.randn(3, 5, 13, generator=_generator(seed + 1))
        return ref, BinaryLinear.from_linear(copy.deepcopy(ref), ramp), x
    ref = nn.Conv1d(13, 6, kernel_size=1, bias=True)
    x = torch.randn(3, 13, 9, generator=_generator(seed + 1))
    return ref, BinaryPointwiseConv1d.from_conv(copy.deepcopy(ref), ramp), x


def _apply(kind: str, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """The float module's own call with an [out, in] weight."""
    return F.linear(x, weight, bias) if kind == "linear" else F.conv1d(x, weight.unsqueeze(2), bias)


def _b_hat(module) -> torch.Tensor:
    codes, scale = module.quantized_weight()
    return codes.float() * scale[:, None]


# ----------------------------------------------------------------------------- contract 1

class BinaryQuantizerTest(unittest.TestCase):
    def test_codes_are_plus_and_minus_one_and_zero_maps_to_plus_one(self) -> None:
        weight = torch.randn(9, 31, generator=_generator(0))
        weight[0, :5] = 0.0
        weight[1, :3] = -0.0
        weight[2] = 0.0  # an all-zero row
        codes, scale = binary_quantize(weight)
        self.assertEqual((codes.dtype, scale.dtype), (torch.int8, torch.float32))
        self.assertEqual((tuple(codes.shape), tuple(scale.shape)), ((9, 31), (9,)))
        self.assertEqual(set(codes.unique().tolist()), {-1, 1})
        expected = torch.where(weight >= 0, 1, -1).to(torch.int8)  # exact zero (either sign) -> +1
        self.assertTrue(torch.equal(codes, expected))
        self.assertTrue((codes[0, :5] == 1).all() and (codes[1, :3] == 1).all() and (codes[2] == 1).all())

    def test_scale_is_row_mean_abs_clamped(self) -> None:
        weight = torch.randn(7, 64, generator=_generator(1))
        weight[3] = 0.0
        weight[4] = 1e-12
        _, scale = binary_quantize(weight)
        expected = weight.double().abs().mean(1).clamp_min(1e-8).float()
        torch.testing.assert_close(scale, expected, rtol=1e-6, atol=0)
        self.assertEqual(scale[3].item(), torch.tensor(1e-8).item())
        self.assertEqual(scale[4].item(), torch.tensor(1e-8).item())

    def test_rejects_non_2d(self) -> None:
        for shape in ((8,), (4, 8, 1), (2, 2, 2, 2)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                binary_quantize(torch.randn(*shape))

    def test_no_gradient_reaches_the_scale(self) -> None:
        weight = torch.randn(4, 8, requires_grad=True)
        codes, scale = binary_quantize(weight)
        self.assertFalse(codes.requires_grad or scale.requires_grad)


# ----------------------------------------------------------------------------- contract 2

class SteTest(unittest.TestCase):
    def test_forward_is_exactly_codes_times_scale_at_fraction_1(self) -> None:
        for kind in ("linear", "conv"):
            for ramp in binary_quant.RAMPS:
                with self.subTest(kind=kind, ramp=ramp):
                    _, module, x = _pair(kind, ramp)
                    self.assertEqual(module.fraction, 1.0)  # default
                    self.assertTrue(torch.equal(module(x), _apply(kind, x, _b_hat(module), module.bias)))

    def test_identity_gradient_to_the_latent_at_every_fraction(self) -> None:
        for kind in ("linear", "conv"):
            for ramp in binary_quant.RAMPS:
                for fraction in (0.0, 0.1, 0.5, 0.9, 1.0):
                    with self.subTest(kind=kind, ramp=ramp, fraction=fraction):
                        _, module, x = _pair(kind, ramp)
                        module.fraction = fraction
                        out = module(x)
                        g = torch.randn(out.shape, generator=_generator(2))
                        (out * g).sum().backward()
                        # The gradient of the same loss with respect to the forward weight, taken
                        # as a leaf (for a linear map it does not depend on the weight's value).
                        w_eff = module.matrix().detach().clone().requires_grad_()
                        (_apply(kind, x, w_eff, module.bias) * g).sum().backward()
                        expected = w_eff.grad if kind == "linear" else w_eff.grad.unsqueeze(2)
                        torch.testing.assert_close(module.weight.grad, expected, rtol=1e-6, atol=1e-6)


# ----------------------------------------------------------------------------- contract 3

class FloatRampTest(unittest.TestCase):
    def test_fraction_0_is_the_original_module_bit_for_bit(self) -> None:
        for kind in ("linear", "conv"):
            with self.subTest(kind=kind):
                ref, module, x = _pair(kind, "fp32")
                module.fraction = 0.0
                self.assertTrue(torch.equal(module(x), ref(x)))

    def test_fraction_1_is_the_binary_forward(self) -> None:
        for kind in ("linear", "conv"):
            with self.subTest(kind=kind):
                _, module, x = _pair(kind, "fp32")
                module.fraction = 0.0
                module.fraction = 1.0
                self.assertTrue(torch.equal(module(x), _apply(kind, x, _b_hat(module), module.bias)))

    def test_between_is_the_lerp_from_the_latent(self) -> None:
        _, module, x = _pair("linear", "fp32")
        module.fraction = 0.3
        expected = torch.lerp(module.weight.detach(), _b_hat(module), 0.3)
        torch.testing.assert_close(module(x), F.linear(x, expected, module.bias), rtol=1e-6, atol=1e-6)


# ----------------------------------------------------------------------------- contract 4

class TernaryRampTest(unittest.TestCase):
    def _ternary(self, ref: nn.Module):
        twin = copy.deepcopy(ref)
        return (quant.TernaryLinear.from_linear(twin) if isinstance(ref, nn.Linear)
                else quant.TernaryPointwiseConv1d.from_conv(twin))

    def test_fraction_0_is_the_ternary_module_bit_for_bit(self) -> None:
        for kind in ("linear", "conv"):
            with self.subTest(kind=kind):
                ref, module, x = _pair(kind, "ternary")
                ternary = self._ternary(ref)  # fraction 1: M1's forward
                module.fraction = 0.0
                self.assertTrue(torch.equal(module(x), ternary(x)))
                self.assertFalse(torch.equal(module(x), ref(x)))

    def test_fraction_1_is_the_binary_forward(self) -> None:
        for kind in ("linear", "conv"):
            with self.subTest(kind=kind):
                _, module, x = _pair(kind, "ternary")
                module.fraction = 1.0
                self.assertTrue(torch.equal(module(x), _apply(kind, x, _b_hat(module), module.bias)))

    def test_between_is_the_lerp_from_ternary(self) -> None:
        ref, module, x = _pair("linear", "ternary")
        t_hat = quant.dequantize(*quant.ternary_quantize(ref.weight))
        module.fraction = 0.6
        expected = torch.lerp(t_hat, _b_hat(module), 0.6)
        torch.testing.assert_close(module(x), F.linear(x, expected, module.bias), rtol=1e-6, atol=1e-6)

    def test_whole_encoder_at_fraction_0_matches_the_ternary_model(self) -> None:
        """Encoder outputs of a tiny model: binary with the ternary ramp at f = 0 equals the
        ternary model (f = 1) bit for bit; at f = 1 it differs."""
        state = fx.tiny_model(seed=3).state_dict()
        binary, ternary = fx.tiny_model(seed=4), fx.tiny_model(seed=5)
        binary.load_state_dict(state)
        ternary.load_state_dict(state)
        with fx.tiny_module_count():
            quantize_parakeet(binary, "ternary")
            quant.quantize_parakeet(ternary)
        audio = 0.1 * torch.randn(2, 16000 * 2, generator=_generator(6))
        lengths = torch.tensor([32000, 24000])
        with torch.no_grad():
            expected, expected_len = ternary.forward(input_signal=audio, input_signal_length=lengths)
            self.assertEqual(set_weight_fraction(binary, 0.0), fx.TINY_MODULES)
            got, got_len = binary.forward(input_signal=audio, input_signal_length=lengths)
            self.assertTrue(torch.equal(got_len, expected_len))
            self.assertTrue(torch.equal(got, expected))
            set_weight_fraction(binary, 1.0)
            self.assertFalse(torch.equal(binary.forward(input_signal=audio, input_signal_length=lengths)[0],
                                         expected))


class FractionTest(unittest.TestCase):
    def test_fraction_and_ramp_validation(self) -> None:
        module = BinaryLinear.from_linear(nn.Linear(4, 4), "fp32")
        for bad in (-0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                module.fraction = bad
        with self.assertRaises(TypeError):
            module.fraction = True
        with self.assertRaises(ValueError):
            BinaryLinear.from_linear(nn.Linear(4, 4), "none")
        for conv in (nn.Conv1d(4, 4, 3), nn.Conv1d(4, 4, 1, groups=4), nn.Conv1d(4, 4, 1, stride=2)):
            with self.assertRaises(ValueError):
                BinaryPointwiseConv1d.from_conv(conv, "fp32")


# ----------------------------------------------------------------------------- contract 5

class QuantizedSetTest(unittest.TestCase):
    """The full Parakeet-TDT-0.6B-v2 architecture from the pinned config (random weights)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = fx.nemo_model(tiny=False)
        cls.before = {k: tuple(v.shape) for k, v in cls.model.state_dict().items()}
        cls.pattern_names = sorted(n for n, _ in cls.model.named_modules()
                                   if quant.QUANTIZED_PATTERN.fullmatch(n))
        cls.parameters_before = list(cls.model.parameters())
        cls.names = quantize_parakeet(cls.model, "fp32")

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.model, cls.parameters_before

    def test_same_264_modules_as_the_ternary_pattern(self) -> None:
        self.assertIs(binary_quant.QUANTIZED_PATTERN, quant.QUANTIZED_PATTERN)
        expected = sorted(f"encoder.layers.{i}.{s}" for i in range(24) for s in quant.PER_LAYER)
        self.assertEqual(self.pattern_names, expected)
        self.assertEqual(self.names, expected)
        self.assertEqual(quantized_module_names(self.model), expected)
        self.assertEqual(len(self.names), 264)
        forbidden = re.compile(r"(^decoder|^joint|pre_encode|depthwise|^preprocessor)")
        self.assertFalse([n for n in self.names if forbidden.search(n)])
        convs = [n for n in self.names if isinstance(self.model.get_submodule(n), BinaryPointwiseConv1d)]
        self.assertEqual(len(convs), 48)

    def test_state_dict_keys_shapes_and_parameters_unchanged(self) -> None:
        after = {k: tuple(v.shape) for k, v in self.model.state_dict().items()}
        self.assertEqual(list(after), list(self.before))
        self.assertEqual(after, self.before)
        self.assertEqual([id(p) for p in self.model.parameters()], [id(p) for p in self.parameters_before])
        accounting = parameter_accounting(self.model)
        self.assertEqual(accounting["total_parameters"], quant.TOTAL_PARAMETERS)
        self.assertEqual(accounting["binary_parameters"],
                         24 * (4 * 4096 * 1024 + 5 * 1024 * 1024 + 3 * 1024 * 1024))
        self.assertEqual(accounting["binary_parameters"] + accounting["float_parameters"],
                         quant.TOTAL_PARAMETERS)

    def test_code_histogram_overall_and_per_family(self) -> None:
        hist = code_histogram(self.model)
        self.assertEqual(hist["zero"], 0.0)
        self.assertAlmostEqual(hist["minus_one"] + hist["plus_one"], 1.0, places=12)
        self.assertEqual(set(hist["per_family"]), {"feed_forward", "self_attn", "pointwise_conv"})
        for family in hist["per_family"].values():
            self.assertAlmostEqual(family["minus_one"] + family["plus_one"], 1.0, places=12)
        self.assertEqual(len(hist["per_module"]), 264)
        self.assertEqual(weight_fraction(self.model), 1.0)

    def test_requantizing_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            quantize_parakeet(self.model, "fp32")
        tiny = fx.tiny_model()
        with fx.tiny_module_count():
            quant.quantize_parakeet(tiny)
            with self.assertRaises(ValueError):  # already ternary
                quantize_parakeet(tiny, "ternary")


if __name__ == "__main__":
    unittest.main()
