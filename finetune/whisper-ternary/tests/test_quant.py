"""Quantizer, STE and module-swap tests. A tiny random Whisper config keeps these CPU-only and fast;
one test loads the real tiny.en checkpoint to pin the DESIGN.md counts."""
from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F
from torch import nn
from transformers import WhisperConfig, WhisperForConditionalGeneration

import paths
from quant import (TernaryEmbedding, TernaryLinear, code_histogram, dequantize, parameter_accounting,
                   quantize_model, quantized_module_names, set_weight_fraction, ternary_quantize,
                   weight_fraction)

TINY_EN_PARAMETERS = 37_760_256


def tiny_model(seed: int = 0) -> WhisperForConditionalGeneration:
    torch.manual_seed(seed)
    config = WhisperConfig(
        vocab_size=100, d_model=32, encoder_layers=2, decoder_layers=2, encoder_attention_heads=2,
        decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=8,
        max_source_positions=50, max_target_positions=20, pad_token_id=99, bos_token_id=98,
        eos_token_id=99, decoder_start_token_id=97, suppress_tokens=[], begin_suppress_tokens=[])
    return WhisperForConditionalGeneration(config).eval()


def expected_projection_names(encoder_layers: int, decoder_layers: int) -> list[str]:
    proj = ("q_proj", "k_proj", "v_proj", "out_proj")
    names = [f"model.encoder.layers.{i}.{m}" for i in range(encoder_layers)
             for m in [f"self_attn.{p}" for p in proj] + ["fc1", "fc2"]]
    names += [f"model.decoder.layers.{i}.{m}" for i in range(decoder_layers)
              for m in [f"{a}.{p}" for a in ("self_attn", "encoder_attn") for p in proj] + ["fc1", "fc2"]]
    return sorted(names)


def batch(model: WhisperForConditionalGeneration) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(2, model.config.num_mel_bins, 2 * model.config.max_source_positions, generator=generator)
    ids = torch.randint(0, 97, (2, 6), generator=generator)
    ids[:, 0] = model.config.decoder_start_token_id
    return features, ids


class TernaryQuantizeTest(unittest.TestCase):
    def test_codes_are_ternary_and_scale_is_row_absmean(self) -> None:
        weight = torch.randn(7, 13, generator=torch.Generator().manual_seed(0))
        codes, scale = ternary_quantize(weight)
        self.assertEqual((codes.dtype, scale.dtype, tuple(scale.shape)), (torch.int8, torch.float32, (7,)))
        self.assertTrue(set(codes.unique().tolist()) <= {-1, 0, 1})
        torch.testing.assert_close(scale, weight.double().abs().mean(1).float())
        ratio = weight / scale[:, None]
        expected = torch.where(ratio.abs() > 0.5, ratio.sign(), torch.zeros_like(ratio)).to(torch.int8)
        self.assertTrue(torch.equal(codes, expected))

    def test_zero_row_gets_floor_scale_and_zero_codes(self) -> None:
        codes, scale = ternary_quantize(torch.zeros(2, 5))
        self.assertTrue(torch.equal(codes, torch.zeros(2, 5, dtype=torch.int8)))
        self.assertTrue(torch.equal(scale, torch.full((2,), 1e-8)))

    def test_exact_ternary_matrix_round_trips(self) -> None:
        generator = torch.Generator().manual_seed(2)
        codes = (torch.randint(0, 2, (5, 16), generator=generator) * 2 - 1).to(torch.int8)
        scale = torch.tensor([0.25, 0.5, 0.375, 1.5, 3.0])
        weight = dequantize(codes, scale)
        got_codes, got_scale = ternary_quantize(weight)
        self.assertTrue(torch.equal(got_codes, codes))
        self.assertTrue(torch.equal(got_scale, scale))
        self.assertTrue(torch.equal(dequantize(got_codes, got_scale), weight))

    def test_rows_with_zero_codes_keep_codes_but_shrink_scale(self) -> None:
        # Why reconstruction_check does not re-run absmean on a dequantized weight.
        codes = torch.tensor([[1, 0, -1, 0], [1, 1, -1, 1]], dtype=torch.int8)
        scale = torch.tensor([0.5, 0.25])
        got_codes, got_scale = ternary_quantize(dequantize(codes, scale))
        self.assertTrue(torch.equal(got_codes, codes))
        self.assertTrue(torch.equal(got_scale, torch.tensor([0.25, 0.25])))

    def test_bf16_input_is_quantized_in_float32(self) -> None:
        weight = torch.randn(6, 9, generator=torch.Generator().manual_seed(3)).bfloat16()
        codes, scale = ternary_quantize(weight)
        ref_codes, ref_scale = ternary_quantize(weight.float())
        self.assertEqual((codes.dtype, scale.dtype, dequantize(codes, scale).dtype),
                         (torch.int8, torch.float32, torch.float32))
        self.assertTrue(torch.equal(codes, ref_codes) and torch.equal(scale, ref_scale))


class SteTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.linear = nn.Linear(13, 7)
        self.x = torch.randn(5, 13)
        self.target = torch.randn(5, 7)

    def reference(self) -> nn.Linear:
        ref = nn.Linear(13, 7)
        with torch.no_grad():
            ref.weight.copy_(dequantize(*ternary_quantize(self.linear.weight)))
            ref.bias.copy_(self.linear.bias)
        return ref

    def test_from_linear_shares_parameters_and_forward_uses_dequantized_weight(self) -> None:
        layer = TernaryLinear.from_linear(self.linear)
        self.assertIs(layer.weight, self.linear.weight)
        self.assertIs(layer.bias, self.linear.bias)
        self.assertEqual((layer.in_features, layer.out_features), (13, 7))
        expected = F.linear(self.x, dequantize(*ternary_quantize(self.linear.weight)), self.linear.bias)
        self.assertTrue(torch.equal(layer(self.x), expected))
        no_bias = TernaryLinear.from_linear(nn.Linear(13, 7, bias=False))
        self.assertIsNone(no_bias.bias)
        self.assertEqual(list(no_bias.state_dict()), ["weight"])

    def test_forward_weight_is_bit_exact_dequantized_weight(self) -> None:
        eye = torch.eye(13)
        w_hat = dequantize(*ternary_quantize(self.linear.weight))
        layer = TernaryLinear.from_linear(self.linear)
        self.assertTrue(torch.equal(layer(eye), F.linear(eye, w_hat, self.linear.bias)))
        self.assertTrue(torch.equal(TernaryLinear.from_parameter(self.linear.weight, None)(eye), w_hat.T))

    def test_gradient_is_identity_ste(self) -> None:
        layer, ref = TernaryLinear.from_linear(self.linear), self.reference()
        F.mse_loss(layer(self.x), self.target).backward()
        F.mse_loss(ref(self.x), self.target).backward()
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        self.assertFalse(layer.quantized_weight()[1].requires_grad)
        self.assertTrue(torch.equal(layer.weight.grad, ref.weight.grad))
        self.assertTrue(torch.equal(layer.bias.grad, ref.bias.grad))

    def test_bf16_autocast_on_cpu(self) -> None:
        layer, ref = TernaryLinear.from_linear(self.linear), self.reference()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out, ref_out = layer(self.x), ref(self.x)
        self.assertEqual(out.dtype, torch.bfloat16)
        F.mse_loss(out.float(), self.target).backward()
        F.mse_loss(ref_out.float(), self.target).backward()
        self.assertEqual(layer.weight.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(layer.weight.grad).all())
        torch.testing.assert_close(layer.weight.grad, ref.weight.grad, rtol=2e-2, atol=1e-3)

    def test_embedding_ste_and_padding(self) -> None:
        emb = nn.Embedding(20, 8, padding_idx=3)
        ternary = TernaryEmbedding.from_embedding(emb)
        self.assertIs(ternary.weight, emb.weight)
        self.assertEqual((ternary.num_embeddings, ternary.embedding_dim, ternary.padding_idx), (20, 8, 3))
        ref = nn.Embedding(20, 8, padding_idx=3)
        with torch.no_grad():
            ref.weight.copy_(dequantize(*ternary.quantized_weight()))
        ids = torch.tensor([[3, 1, 5, 3], [7, 3, 19, 0]])
        out = ternary(ids)
        self.assertTrue(torch.equal(out, ref(ids)))
        out.square().sum().backward()
        ref(ids).square().sum().backward()
        torch.testing.assert_close(ternary.weight.grad, ref.weight.grad)
        self.assertEqual(float(ternary.weight.grad[3].abs().sum()), 0.0)


class WeightFractionTest(unittest.TestCase):
    """Progressive quantization (DESIGN.md Revision 2): forward weight lerp(W, W_hat_ste, fraction)."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.linear = nn.Linear(13, 7)
        self.emb = nn.Embedding(20, 8, padding_idx=3)
        self.x = torch.randn(5, 13)
        self.ids = torch.tensor([[3, 1, 5, 3], [7, 3, 19, 0]])
        self.target = torch.randn(5, 7)

    def test_default_fraction_is_one(self) -> None:
        self.assertEqual(TernaryLinear.from_linear(self.linear).fraction, 1.0)
        self.assertEqual(TernaryEmbedding.from_embedding(self.emb).fraction, 1.0)
        model = tiny_model()
        quantize_model(model, include_embedding=True)
        self.assertEqual(weight_fraction(model), 1.0)

    def test_fraction_zero_is_the_plain_layer_bit_for_bit(self) -> None:
        layer = TernaryLinear.from_linear(self.linear)
        layer.fraction = 0.0
        self.assertTrue(torch.equal(layer(self.x), F.linear(self.x, self.linear.weight, self.linear.bias)))
        emb = TernaryEmbedding.from_embedding(self.emb)
        emb.fraction = 0.0
        self.assertTrue(torch.equal(emb(self.ids), F.embedding(self.ids, self.emb.weight, padding_idx=3)))

    def test_fraction_zero_model_is_the_unquantized_model(self) -> None:
        for include_embedding in (False, True):
            with self.subTest(include_embedding=include_embedding):
                reference = tiny_model()
                model = copy.deepcopy(reference)
                quantize_model(model, include_embedding=include_embedding)
                features, ids = batch(model)
                with torch.no_grad():
                    expected = reference(input_features=features, decoder_input_ids=ids).logits
                    set_weight_fraction(model, 0.0)
                    self.assertTrue(torch.equal(model(input_features=features, decoder_input_ids=ids).logits,
                                                expected))
                    set_weight_fraction(model, 1.0)  # sanity: quantization is really active at 1
                    self.assertFalse(torch.equal(model(input_features=features, decoder_input_ids=ids).logits,
                                                 expected))

    def test_fraction_half_is_the_midpoint(self) -> None:
        w = self.linear.weight.detach()
        w_hat = dequantize(*ternary_quantize(w))
        layer = TernaryLinear.from_parameter(self.linear.weight, None)
        layer.fraction = 0.5
        effective = layer(torch.eye(13)).T  # one nonzero product per output: exactly the forward weight
        torch.testing.assert_close(effective, 0.5 * (w + w_hat), rtol=1e-6, atol=1e-7)
        self.assertFalse(torch.equal(effective, w) or torch.equal(effective, w_hat))
        emb = TernaryEmbedding.from_embedding(self.emb)
        emb.fraction = 0.5
        e = self.emb.weight.detach()
        torch.testing.assert_close(emb(torch.arange(20)), 0.5 * (e + dequantize(*ternary_quantize(e))),
                                   rtol=1e-6, atol=1e-7)

    def test_gradient_matches_the_equivalent_dense_layer(self) -> None:
        # The latent gradient is the identity STE for every fraction: it equals the gradient of a dense
        # layer whose weight is the effective weight. Exact at 0 and 1; lerp's backward
        # g*(1-f) + g*f rounds within an ulp in between.
        for fraction in (0.0, 0.3, 1.0):
            with self.subTest(fraction=fraction):
                linear = copy.deepcopy(self.linear)
                layer = TernaryLinear.from_linear(linear)
                layer.fraction = fraction
                dense = nn.Linear(13, 7)
                with torch.no_grad():
                    dense.weight.copy_(torch.lerp(linear.weight, dequantize(*ternary_quantize(linear.weight)),
                                                  fraction))
                    dense.bias.copy_(linear.bias)
                out, ref = layer(self.x), dense(self.x)
                self.assertTrue(torch.equal(out, ref))
                F.mse_loss(out, self.target).backward()
                F.mse_loss(ref, self.target).backward()
                self.assertTrue(torch.allclose(linear.weight.grad, dense.weight.grad, rtol=1e-6, atol=1e-10))
                if fraction in (0.0, 1.0):
                    self.assertTrue(torch.equal(linear.weight.grad, dense.weight.grad))
                self.assertTrue(torch.equal(linear.bias.grad, dense.bias.grad))
                self.assertFalse(layer.quantized_weight()[1].requires_grad)

                emb = copy.deepcopy(self.emb)
                ternary = TernaryEmbedding.from_embedding(emb)
                ternary.fraction = fraction
                dense_emb = nn.Embedding(20, 8, padding_idx=3)
                with torch.no_grad():
                    dense_emb.weight.copy_(torch.lerp(emb.weight, dequantize(*ternary_quantize(emb.weight)),
                                                      fraction))
                out, ref = ternary(self.ids), dense_emb(self.ids)
                self.assertTrue(torch.equal(out, ref))
                (out.square() * 0.5).sum().backward()
                (ref.square() * 0.5).sum().backward()
                self.assertTrue(torch.allclose(emb.weight.grad, dense_emb.weight.grad, rtol=1e-6, atol=1e-10))
                if fraction in (0.0, 1.0):
                    self.assertTrue(torch.equal(emb.weight.grad, dense_emb.weight.grad))
                self.assertEqual(float(emb.weight.grad[3].abs().sum()), 0.0)

    def test_set_and_get_helpers(self) -> None:
        for include_embedding, count in ((False, 32), (True, 34)):
            with self.subTest(include_embedding=include_embedding):
                model = tiny_model()
                names = quantize_model(model, include_embedding=include_embedding)
                keys = list(model.state_dict())
                self.assertEqual(set_weight_fraction(model, 0.25), count)
                self.assertEqual(len(names), count)  # both halves of the tie are set
                self.assertEqual(weight_fraction(model), 0.25)
                self.assertTrue(all(model.get_submodule(n).fraction == 0.25 for n in names))
                self.assertEqual(set_weight_fraction(model, 0), count)
                self.assertIs(type(weight_fraction(model)), float)
                self.assertEqual(weight_fraction(model), 0.0)
                set_weight_fraction(model, 1)
                self.assertEqual(weight_fraction(model), 1.0)
                self.assertEqual(list(model.state_dict()), keys)  # fraction is not state

    def test_helper_validation(self) -> None:
        model = tiny_model()
        quantize_model(model, include_embedding=True)
        set_weight_fraction(model, 0.5)
        for bad in (-0.1, 1.0001, 2, float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                set_weight_fraction(model, bad)
        for bad in ("0.5", None, True, torch.tensor(0.5)):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                set_weight_fraction(model, bad)
        self.assertEqual(weight_fraction(model), 0.5)  # rejected calls changed nothing
        for empty in (tiny_model(), nn.Linear(2, 2)):
            with self.assertRaisesRegex(ValueError, "no ternary modules"):
                set_weight_fraction(empty, 0.5)
            with self.assertRaisesRegex(ValueError, "no ternary modules"):
                weight_fraction(empty)
        model.get_submodule("model.encoder.layers.0.fc1").fraction = 0.75
        with self.assertRaisesRegex(ValueError, "disagree"):
            weight_fraction(model)
        layer = TernaryLinear.from_linear(self.linear)
        with self.assertRaises(ValueError):
            layer.fraction = 1.5
        self.assertEqual(layer.fraction, 1.0)

    def test_codes_histogram_and_accounting_ignore_fraction(self) -> None:
        model = tiny_model()
        names = quantize_model(model, include_embedding=True)
        before = {n: model.get_submodule(n).quantized_weight() for n in names}
        histogram, accounting = code_histogram(model), parameter_accounting(model)
        for fraction in (0.0, 0.3):
            set_weight_fraction(model, fraction)
            for name in names:
                codes, scale = model.get_submodule(name).quantized_weight()
                self.assertTrue(torch.equal(codes, before[name][0]) and torch.equal(scale, before[name][1]), name)
            self.assertEqual(code_histogram(model), histogram)
            self.assertEqual(parameter_accounting(model), accounting)


class QuantizeModelTest(unittest.TestCase):
    def test_projection_arm_swaps_exactly_the_design_set(self) -> None:
        model = tiny_model()
        before = {name: param for name, param in model.named_parameters()}
        keys = list(model.state_dict())
        total = sum(p.numel() for p in model.parameters())
        names = quantize_model(model, include_embedding=False)
        self.assertEqual(names, expected_projection_names(2, 2))
        self.assertEqual(len(names), 2 * 6 + 2 * 10)
        self.assertEqual(quantized_module_names(model), names)
        for name in names:
            self.assertIsInstance(model.get_submodule(name), TernaryLinear)
        after = dict(model.named_parameters())
        self.assertEqual(after.keys(), before.keys())
        self.assertTrue(all(after[name] is before[name] for name in before))
        self.assertEqual(list(model.state_dict()), keys)
        for name, module in model.named_modules():
            if name not in names:
                self.assertNotIsInstance(module, (TernaryLinear, TernaryEmbedding), name)
        encoder = model.model.encoder
        self.assertIsInstance(encoder.conv1, nn.Conv1d)
        self.assertIsInstance(encoder.conv2, nn.Conv1d)
        self.assertIs(type(model.model.decoder.embed_tokens), nn.Embedding)
        self.assertIs(type(model.proj_out), nn.Linear)
        self.assertIs(model.proj_out.weight, model.model.decoder.embed_tokens.weight)
        accounting = parameter_accounting(model)
        ternary = sum(model.get_submodule(name).weight.numel() for name in names)
        self.assertEqual(accounting, {"ternary_parameters": ternary, "residual_parameters": total - ternary,
                                      "total_parameters": total})

    def test_double_quantization_raises(self) -> None:
        for include_embedding in (False, True):
            model = tiny_model()
            quantize_model(model, include_embedding=include_embedding)
            with self.assertRaises(ValueError):
                quantize_model(model, include_embedding=False)

    def test_embedding_arm_keeps_one_tied_latent(self) -> None:
        model = tiny_model()
        embedding = model.model.decoder.embed_tokens.weight
        total = sum(p.numel() for p in model.parameters())
        names = quantize_model(model, include_embedding=True)
        self.assertEqual(names, sorted(expected_projection_names(2, 2) + ["model.decoder.embed_tokens", "proj_out"]))
        model.tie_weights()
        decoder_embed = model.model.decoder.embed_tokens
        self.assertIs(model.proj_out.weight, embedding)
        self.assertIs(decoder_embed.weight, embedding)
        self.assertIs(model.get_input_embeddings(), decoder_embed)
        self.assertIs(model.get_output_embeddings(), model.proj_out)
        self.assertIsInstance(decoder_embed, TernaryEmbedding)
        self.assertEqual((model.proj_out.out_features, decoder_embed.padding_idx), (100, 99))
        accounting = parameter_accounting(model)
        self.assertEqual(accounting["total_parameters"], total)
        self.assertEqual(accounting["ternary_parameters"],
                         embedding.numel() + sum(model.get_submodule(n).weight.numel()
                                                 for n in expected_projection_names(2, 2)))
        features, ids = batch(model)
        with torch.no_grad():
            hidden = model.model(input_features=features, decoder_input_ids=ids).last_hidden_state
            logits = model(input_features=features, decoder_input_ids=ids).logits
            dequantized = dequantize(*ternary_quantize(embedding))
            torch.testing.assert_close(logits, F.linear(hidden, dequantized))
            torch.testing.assert_close(decoder_embed(ids), F.embedding(ids, dequantized))

    def test_histogram_counts_the_tie_once(self) -> None:
        model = tiny_model()
        quantize_model(model, include_embedding=True)
        histogram = code_histogram(model)
        self.assertNotIn("proj_out", histogram["per_layer"])
        self.assertEqual(len(histogram["per_layer"]), 2 * 6 + 2 * 10 + 1)
        self.assertAlmostEqual(histogram["minus_one"] + histogram["zero"] + histogram["plus_one"], 1.0)
        codes = torch.cat([model.get_submodule(n).quantized_weight()[0].flatten() for n in histogram["per_layer"]])
        self.assertEqual(codes.numel(), parameter_accounting(model)["ternary_parameters"])
        self.assertAlmostEqual(histogram["zero"], float((codes == 0).double().mean()))
        self.assertAlmostEqual(histogram["plus_one"], float((codes == 1).double().mean()))

    def test_training_step_reaches_every_latent(self) -> None:
        for fraction in (0.0, 0.3, 1.0):
            model = tiny_model().train()
            quantize_model(model, include_embedding=True)
            set_weight_fraction(model, fraction)
            features, ids = batch(model)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                loss = model(input_features=features, labels=ids[:, 1:]).loss
            loss.backward()
            for name in quantized_module_names(model):
                grad = model.get_submodule(name).weight.grad
                self.assertTrue(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0,
                                (fraction, name))


@unittest.skipUnless((paths.MODEL_DIR / "model.safetensors").is_file(), "tiny.en checkpoint not present")
class RealTinyEnTest(unittest.TestCase):
    def load(self) -> WhisperForConditionalGeneration:
        return WhisperForConditionalGeneration.from_pretrained(
            paths.MODEL_DIR, local_files_only=True, use_safetensors=True, dtype=torch.float32)

    def test_design_counts_and_tie(self) -> None:
        model = self.load()
        c = model.config
        projection = c.encoder_layers * (4 * c.d_model ** 2 + 2 * c.d_model * c.encoder_ffn_dim) \
            + c.decoder_layers * (8 * c.d_model ** 2 + 2 * c.d_model * c.decoder_ffn_dim)
        names = quantize_model(model, include_embedding=False)
        self.assertEqual(names, expected_projection_names(4, 4))
        self.assertEqual(len(names), 64)
        self.assertEqual(parameter_accounting(model), {
            "ternary_parameters": projection, "residual_parameters": TINY_EN_PARAMETERS - projection,
            "total_parameters": TINY_EN_PARAMETERS})

        model = self.load()
        embedding = model.model.decoder.embed_tokens.weight
        names = quantize_model(model, include_embedding=True)
        self.assertEqual(len(names), 66)
        self.assertIs(model.proj_out.weight, embedding)
        model.tie_weights()
        self.assertIs(model.proj_out.weight, model.model.decoder.embed_tokens.weight)
        self.assertIs(model.get_output_embeddings().weight, embedding)
        self.assertEqual(parameter_accounting(model)["ternary_parameters"], projection + embedding.numel())
        self.assertEqual(parameter_accounting(model)["total_parameters"], TINY_EN_PARAMETERS)

    def test_ramp_starts_at_the_pretrained_model(self) -> None:
        reference = self.load().eval()
        features = torch.randn(1, reference.config.num_mel_bins, 3000, generator=torch.Generator().manual_seed(0))
        ids = torch.tensor([[reference.config.decoder_start_token_id, 50362, 383, 3290]])
        with torch.no_grad():
            expected = reference(input_features=features, decoder_input_ids=ids).logits
        for include_embedding, count in ((False, 64), (True, 66)):
            model = self.load().eval()
            quantize_model(model, include_embedding=include_embedding)
            self.assertEqual(set_weight_fraction(model, 0.0), count)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(input_features=features, decoder_input_ids=ids).logits, expected))
            self.assertEqual(set_weight_fraction(model, 1.0), count)
            self.assertEqual(weight_fraction(model), 1.0)


if __name__ == "__main__":
    unittest.main()
