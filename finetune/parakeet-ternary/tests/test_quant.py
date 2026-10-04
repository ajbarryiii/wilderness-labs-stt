"""Quantizer, STE, ramp and module-swap tests. Standalone modules for the math; the real
Parakeet checkpoint (CPU) pins the quantized set, counts and fraction-0 exactness.

Run: finetune/parakeet-ternary/python -m unittest discover -s finetune/parakeet-ternary/tests -v
"""
from __future__ import annotations

import copy
import re
import unittest

import torch
import torch.nn.functional as F
from torch import nn

import paths
import quant
from quant import (QUANTIZED_PATTERN, TernaryLinear, TernaryPointwiseConv1d, code_histogram, dequantize,
                   parameter_accounting, quantize_parakeet, quantized_module_names, set_weight_fraction,
                   ternary_quantize, weight_fraction)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class QuantizerReuseTest(unittest.TestCase):
    def test_quantizer_is_the_whisper_implementation(self) -> None:
        wq = quant.load_whisper_module("quant")
        self.assertIs(ternary_quantize, wq.ternary_quantize)
        self.assertIs(dequantize, wq.dequantize)
        self.assertIs(quant._effective_weight, wq._effective_weight)
        self.assertTrue(str(quant.WHISPER_DIR).endswith("finetune/whisper-ternary"))

    def test_codes_and_scale(self) -> None:
        weight = torch.randn(7, 13, generator=_generator(0))
        codes, scale = ternary_quantize(weight)
        self.assertEqual((codes.dtype, scale.dtype, tuple(scale.shape)), (torch.int8, torch.float32, (7,)))
        torch.testing.assert_close(scale, weight.double().abs().mean(1).float())
        ratio = weight / scale[:, None]
        expected = torch.where(ratio.abs() > 0.5, ratio.sign(), torch.zeros_like(ratio)).to(torch.int8)
        self.assertTrue(torch.equal(codes, expected))


class SteTest(unittest.TestCase):
    def _pair(self, kind: str):
        torch.manual_seed(0)
        if kind == "linear":
            ref = nn.Linear(12, 6, bias=True)
            x = torch.randn(3, 5, 12, generator=_generator(1))
            return ref, TernaryLinear.from_linear(copy.deepcopy(ref)), x
        ref = nn.Conv1d(12, 6, kernel_size=1, bias=True)
        x = torch.randn(3, 12, 9, generator=_generator(1))
        return ref, TernaryPointwiseConv1d.from_conv(copy.deepcopy(ref)), x

    def _w_hat(self, module) -> torch.Tensor:
        return dequantize(*module.quantized_weight())

    def test_forward_is_exactly_w_hat_at_fraction_1(self) -> None:
        for kind in ("linear", "conv"):
            with self.subTest(kind=kind):
                _, module, x = self._pair(kind)
                w_hat = self._w_hat(module)
                expected = (F.linear(x, w_hat, module.bias) if kind == "linear"
                            else F.conv1d(x, w_hat.unsqueeze(2), module.bias))
                self.assertTrue(torch.equal(module(x), expected))

    def test_forward_is_exactly_the_float_module_at_fraction_0(self) -> None:
        for kind in ("linear", "conv"):
            with self.subTest(kind=kind):
                ref, module, x = self._pair(kind)
                module.fraction = 0.0
                self.assertTrue(torch.equal(module(x), ref(x)))

    def test_identity_gradient_to_latent_weight(self) -> None:
        for kind in ("linear", "conv"):
            for fraction in (0.0, 0.3, 1.0):
                with self.subTest(kind=kind, fraction=fraction):
                    _, module, x = self._pair(kind)
                    module.fraction = fraction
                    g = torch.randn_like(module(x), generator=_generator(2))
                    (module(x) * g).sum().backward()
                    # The gradient of the same loss through a plain layer with weight W_eff (treated as a leaf).
                    w_eff = torch.lerp(module.matrix().detach(), self._w_hat(module), fraction).requires_grad_()
                    out = (F.linear(x, w_eff, module.bias) if kind == "linear"
                           else F.conv1d(x, w_eff.unsqueeze(2), module.bias))
                    (out * g).sum().backward()
                    expected = w_eff.grad if kind == "linear" else w_eff.grad.unsqueeze(2)
                    torch.testing.assert_close(module.weight.grad, expected, rtol=1e-6, atol=1e-6)

    def test_parameter_objects_and_shapes_are_kept(self) -> None:
        conv = nn.Conv1d(8, 4, kernel_size=1, bias=False)
        module = TernaryPointwiseConv1d.from_conv(conv)
        self.assertIs(module.weight, conv.weight)
        self.assertEqual(tuple(module.weight.shape), (4, 8, 1))
        codes, scale = module.quantized_weight()
        self.assertEqual((tuple(codes.shape), tuple(scale.shape)), ((4, 8), (4,)))

    def test_rejects_non_pointwise_conv(self) -> None:
        for conv in (nn.Conv1d(4, 4, 3), nn.Conv1d(4, 4, 1, groups=4), nn.Conv1d(4, 4, 1, stride=2)):
            with self.assertRaises(ValueError):
                TernaryPointwiseConv1d.from_conv(conv)

    def test_fraction_validation(self) -> None:
        module = TernaryLinear.from_linear(nn.Linear(4, 4))
        for bad in (-0.1, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                module.fraction = bad
        with self.assertRaises(TypeError):
            module.fraction = True


class RealModelTest(unittest.TestCase):
    """One CPU load of the real checkpoint for the whole class."""

    @classmethod
    def setUpClass(cls) -> None:
        import evaluate
        cls.model = evaluate.load_pretrained("cpu")
        generator = _generator(3)
        cls.audio = 0.1 * torch.randn(2, 16000 * 3, generator=generator)
        cls.lengths = torch.tensor([16000 * 3, 16000 * 2])
        with torch.no_grad():
            cls.float_encoded, cls.float_len = cls.model.forward(input_signal=cls.audio,
                                                                 input_signal_length=cls.lengths)
        cls.float_names = {n for n, _ in cls.model.named_modules()}
        cls.float_keys = list(cls.model.state_dict().keys())
        cls.names = quantize_parakeet(cls.model)

    def test_quantized_set_is_exactly_the_design_list(self) -> None:
        expected = sorted(f"encoder.layers.{i}.{suffix}" for i in range(24) for suffix in quant.PER_LAYER)
        self.assertEqual(self.names, expected)
        self.assertEqual(len(self.names), 264)
        matched = [n for n in self.float_names if QUANTIZED_PATTERN.fullmatch(n)]
        self.assertEqual(sorted(matched), expected)
        forbidden = re.compile(r"(^decoder|^joint|pre_encode|depthwise|^preprocessor)")
        self.assertFalse([n for n in self.names if forbidden.search(n)])
        types = {type(self.model.get_submodule(n)).__name__ for n in self.names}
        self.assertEqual(types, {"TernaryLinear", "TernaryPointwiseConv1d"})
        convs = [n for n in self.names if isinstance(self.model.get_submodule(n), TernaryPointwiseConv1d)]
        self.assertEqual(len(convs), 48)

    def test_state_dict_keys_and_counts_unchanged(self) -> None:
        self.assertEqual(list(self.model.state_dict().keys()), self.float_keys)
        accounting = parameter_accounting(self.model)
        self.assertEqual(accounting["total_parameters"], quant.TOTAL_PARAMETERS)
        self.assertEqual(accounting["ternary_parameters"] + accounting["float_parameters"], quant.TOTAL_PARAMETERS)
        self.assertEqual(accounting["ternary_parameters"], 24 * (4 * 4096 * 1024 + 5 * 1024 * 1024 + 3 * 1024 * 1024))

    def test_fraction_0_reproduces_float_encoder_exactly(self) -> None:
        self.assertEqual(set_weight_fraction(self.model, 0.0), 264)
        self.assertEqual(weight_fraction(self.model), 0.0)
        try:
            with torch.no_grad():
                encoded, length = self.model.forward(input_signal=self.audio, input_signal_length=self.lengths)
            self.assertTrue(torch.equal(length, self.float_len))
            self.assertTrue(torch.equal(encoded, self.float_encoded))
            set_weight_fraction(self.model, 1.0)
            with torch.no_grad():
                quantized, _ = self.model.forward(input_signal=self.audio, input_signal_length=self.lengths)
            self.assertFalse(torch.equal(quantized, self.float_encoded))
        finally:
            set_weight_fraction(self.model, 1.0)

    def test_requantize_refused_and_histogram(self) -> None:
        with self.assertRaises(ValueError):
            quantize_parakeet(self.model)
        hist = code_histogram(self.model)
        self.assertAlmostEqual(hist["minus_one"] + hist["zero"] + hist["plus_one"], 1.0, places=9)
        self.assertEqual(len(hist["per_module"]), 264)
        self.assertEqual(quantized_module_names(self.model), self.names)

    def test_disagreeing_fractions_raise(self) -> None:
        try:
            self.model.get_submodule(self.names[0]).fraction = 0.5
            with self.assertRaises(ValueError):
                weight_fraction(self.model)
        finally:
            set_weight_fraction(self.model, 1.0)


if __name__ == "__main__":
    unittest.main()
