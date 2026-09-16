"""CPU numerical/causality checks; no downloaded data or model artifacts."""

import math
import unittest

import torch
from torch.nn import functional as F

from binary_stt.features import CausalLogMel
from binary_stt.model import (
    BinaryCTCModel,
    BinaryLinear,
    ModelConfig,
    activation_quantize,
    binary_sign,
    chunk_attention_mask,
)


def tiny_config(**overrides):
    values = dict(
        d_model=32, ff_dim=128, num_layers=2, num_heads=4,
        stem_channels=16, vocab_size=32, dropout=0.0,
        chunk_size=4, left_context=8, activation_checkpointing=True,
    )
    values.update(overrides)
    return values


class BinaryOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)

    def test_sign_zero_and_activation_ste(self):
        value = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
        threshold = torch.zeros(3, requires_grad=True)
        actual = activation_quantize(value, threshold, 1.0)
        torch.testing.assert_close(actual, torch.tensor([-1.0, 1.0, 1.0]))
        actual.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones(3))
        torch.testing.assert_close(threshold.grad, -torch.ones(3))

    def test_popcount_algebra_with_partial_word(self):
        layer = BinaryLinear(37, 5)
        with torch.no_grad():
            layer.weight[0, 0] = 0
            layer.threshold.copy_(torch.linspace(-0.3, 0.3, 37))
            layer.log_scale.copy_(torch.arange(5) * math.log(2))
        layer.set_quantization(1.0, 1.0)
        value = torch.randn(3, 37)
        activation_bits = value >= layer.threshold
        weight_bits = layer.weight >= 0
        expected = torch.empty(3, 5)
        for row in range(3):
            for channel in range(5):
                differences = 0
                for start in range(0, 37, 32):
                    a = sum(int(bit) << k for k, bit in enumerate(activation_bits[row, start:start + 32]))
                    w = sum(int(bit) << k for k, bit in enumerate(weight_bits[channel, start:start + 32]))
                    differences += (a ^ w).bit_count()
                expected[row, channel] = (37 - 2 * differences) * layer.output_scale[channel]
        torch.testing.assert_close(layer(value), expected, atol=0, rtol=0)

    def test_weight_ste_is_identity_and_scales_learn(self):
        layer = BinaryLinear(4, 3)
        layer.set_quantization(1.0, 1.0)
        value = torch.tensor([[2.0, -1.0, 0.0, 4.0], [-2.0, -3.0, 1.0, 0.0]])
        output = layer(value)
        expected_scale_gradient = output.detach().sum(0)
        output.sum().backward()
        expected_weight_gradient = binary_sign(value).sum(0).expand(3, -1)
        torch.testing.assert_close(layer.weight.grad, expected_weight_gradient)
        torch.testing.assert_close(layer.log_scale.grad, expected_scale_gradient)
        self.assertTrue(torch.isfinite(layer.threshold.grad).all())

    def test_blend_endpoints_and_partial_weight_ste(self):
        layer = BinaryLinear(4, 3)
        value = torch.randn(2, 4)
        layer.set_quantization(0.0, 0.0)
        torch.testing.assert_close(layer(value), F.linear(value, layer.weight))
        layer.set_quantization(1.0, 0.0)
        torch.testing.assert_close(layer(value), F.linear(value, binary_sign(layer.weight)) * layer.output_scale)
        layer.set_quantization(0.25, 0.0)
        expected_weight = 0.75 * layer.weight + 0.25 * binary_sign(layer.weight) * layer.output_scale[:, None]
        torch.testing.assert_close(layer.effective_weight(), expected_weight)
        layer(value).sum().backward()
        torch.testing.assert_close(layer.weight.grad, value.sum(0).expand(3, -1))
        with self.assertRaises(ValueError):
            layer.set_quantization(float("nan"), 1)

    def test_optional_centering_matches_global_mean(self):
        layer = BinaryLinear(2, 2, center_weights=True)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 10.0]]))
            layer.log_scale.zero_()
        layer.set_quantization(1.0, 1.0)
        expected = binary_sign(layer.weight - layer.weight.mean())
        torch.testing.assert_close(layer.effective_weight(), expected)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_exact_default_parameter_count_without_allocation(self):
        with torch.device("meta"):
            model = BinaryCTCModel(ModelConfig())
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 488270080)
        binary_matrices = [module for module in model.modules() if isinstance(module, BinaryLinear)]
        self.assertEqual(len(binary_matrices), 160)
        self.assertEqual(sum(module.weight.numel() for module in binary_matrices), 482344960)
        self.assertTrue(all(module.weight.dtype == torch.float32 for module in binary_matrices))

    def test_padding_invariance_and_length_rounding(self):
        model = BinaryCTCModel(tiny_config()).eval()
        model.set_quantization(1.0, 1.0)
        original = torch.randn(1, 80, 41)
        padded = torch.cat((original, 100 * torch.randn(1, 80, 40)), dim=-1)
        with torch.no_grad():
            alone, alone_lengths = model(original, torch.tensor([41]))
            together, together_lengths = model(padded, torch.tensor([41]))
        torch.testing.assert_close(alone_lengths, torch.tensor([6]))
        torch.testing.assert_close(alone_lengths, together_lengths)
        torch.testing.assert_close(alone, together[:, :6], atol=2e-5, rtol=2e-5)
        self.assertEqual(torch.count_nonzero(together[:, 6:]).item(), 0)

    def test_chunk_mask_has_current_chunk_and_fixed_previous_cache(self):
        mask = chunk_attention_mask(12, 4, 3)
        self.assertTrue(mask[:4, :4].all())
        self.assertFalse(mask[:4, 4:].any())
        self.assertEqual(mask[8].nonzero().flatten().tolist(), list(range(5, 12)))
        self.assertTrue(torch.equal(mask[8], mask[11]))

    def test_future_chunks_cannot_change_completed_chunk(self):
        # Eight encoded prefix frames end a complete 4-frame chunk. A frame
        # within an unfinished chunk intentionally may see its chunk mates.
        model = BinaryCTCModel(tiny_config()).eval()
        model.set_quantization(1.0, 1.0)
        prefix = torch.randn(1, 80, 64)
        longer = torch.cat((prefix, torch.randn(1, 80, 40)), dim=-1)
        with torch.no_grad():
            prefix_logits, _ = model(prefix, torch.tensor([64]))
            longer_logits, _ = model(longer, torch.tensor([104]))
        torch.testing.assert_close(prefix_logits, longer_logits[:, :8], atol=2e-5, rtol=2e-5)

    def test_single_frame_chunks_are_causal(self):
        model = BinaryCTCModel(tiny_config(chunk_size=1)).eval()
        features = torch.randn(1, 80, 57)
        altered = features.clone()
        altered[:, :, 33:] = torch.randn_like(altered[:, :, 33:]) * 10
        with torch.no_grad():
            original, _ = model(features, torch.tensor([57]))
            updated, _ = model(altered, torch.tensor([57]))
        # Encoded frame 4 ends at input feature 32 after three causal strides.
        torch.testing.assert_close(original[:, :5], updated[:, :5], atol=2e-5, rtol=2e-5)

    def test_checkpointed_w1a1_ctc_backward(self):
        model = BinaryCTCModel(tiny_config()).train()
        model.set_quantization(1.0, 1.0)
        logits, lengths = model(torch.randn(2, 80, 65), torch.tensor([65, 49]))
        targets = torch.tensor([1, 2, 3, 4, 5])
        loss = F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), targets,
                          lengths, torch.tensor([3, 2]), blank=0, zero_infinity=False)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_invalid_lengths_and_configuration_fail(self):
        model = BinaryCTCModel(tiny_config())
        for invalid in (torch.tensor([0]), torch.tensor([17]), torch.tensor([1.0])):
            with self.assertRaises(ValueError):
                model(torch.randn(1, 80, 16), invalid)
        with self.assertRaises(ValueError):
            BinaryCTCModel(tiny_config(num_heads=3))
        with self.assertRaises(ValueError):
            BinaryCTCModel(tiny_config(unknown_field=7))


class FeatureTests(unittest.TestCase):
    def test_prefix_stability_and_lengths(self):
        extractor = CausalLogMel()
        torch.manual_seed(11)
        prefix = torch.randn(1025)
        extended = torch.cat((prefix, torch.randn(1733)))
        short = extractor(prefix)
        long = extractor(extended)
        self.assertEqual(short.shape, (80, 7))
        self.assertEqual(long.shape[-1], extractor.feature_lengths(extended.numel()))
        # GEMM/FFT batching can change FP32 roundoff with sequence length.
        torch.testing.assert_close(short, long[:, :short.shape[-1]], rtol=1e-6, atol=5e-7)

    def test_silence_is_finite_and_wrong_rate_is_rejected(self):
        extractor = CausalLogMel()
        output = extractor(torch.zeros(2, 320))
        self.assertEqual(output.shape, (2, 80, 2))
        self.assertTrue(torch.isfinite(output).all())
        with self.assertRaises(ValueError):
            extractor(torch.zeros(320), sample_rate=8000)
        with self.assertRaises(ValueError):
            extractor(torch.zeros(0))


if __name__ == "__main__":
    unittest.main()
