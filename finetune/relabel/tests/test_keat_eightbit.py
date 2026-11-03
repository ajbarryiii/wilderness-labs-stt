"""Contract tests (CPU, the tiny model): the ternary 8-bit training forms of stage 5, training/keat_eightbit.py
(keat-prime plan "Step 0" and "Code and review", "The S2 ternary form").

- the S2 ternary form: the codes are quant.ternary_quantize's from the FP32 scale (never from the FP16-rounded one),
  the forward weight is codes * fp32(fp16(s)) bit for bit, the gradient to the latent weight is the identity, and a
  weight fraction below 1 is refused;
- the scopes are stage 4's (S1 = prediction network; S2 = S1 + joint + pre_encode.out with FP16 ternary row scales),
  built from stage 4's type-agnostic INT8 pieces (eightbit.quantize_frozen, W8Linear) on a ternary model;
- keat_form reads a form back and refuses a state that does not match one (a binary model, a weight changed after
  load, mixed scales, a palettized part); a second form is refused;
- ternarize swaps every module of the quantized pattern for its ternary module, keeping the Parameters.
"""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

import quant  # finetune/parakeet-ternary/quant.py
from decoder_fixtures import tiny_model
from training import eightbit as eb
from training import keat_eightbit as teb

TINY_MODULES = 22   # 2 layers x 11


def ternary_model(seed: int = 11):
    model = tiny_model(seed=seed)
    teb.ternarize(model)
    return model.eval()


def form_model(scope: str, seed: int = 11):
    model = ternary_model(seed)
    teb.apply_keat_scope(model, scope)
    return model


def boundary_row(n: int = 1024, seed: int = 5) -> torch.Tensor:
    """A [1, n] FP32 row with one entry between 0.5 * s and 0.5 * fp32(fp16(s)), where s = mean|W|: its code is 0
    under one scale and nonzero under the other (so a form that rounds the scale before coding is caught)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(1, n, generator=g) * 2 - 1
    for _ in range(50):
        s = w.abs().mean(dim=1)
        s16 = s.to(torch.float16).to(torch.float32)
        if torch.equal(s, s16):
            w[0, 1:] *= 1.0001
            continue
        w[0, 0] = 0.25 * (s[0] + s16[0])
    return w


class HalfScaleForm(unittest.TestCase):
    def test_codes_from_the_fp32_scale_and_forward_codes_times_fp16_scale(self):
        model = form_model("S2")
        names = quant.quantized_module_names(model)
        self.assertEqual(len(names), TINY_MODULES)
        for name in names:
            module = model.get_submodule(name)
            self.assertIsInstance(module, teb.HALF_SCALE_TYPES)
            matrix = module.matrix().detach()
            codes32, s32 = quant.ternary_quantize(matrix)
            half = s32.to(torch.float16).to(torch.float32)
            codes, scale = module.quantized_weight()
            self.assertTrue(torch.equal(codes, codes32), name)
            self.assertEqual(scale.dtype, torch.float32)
            self.assertTrue(torch.equal(scale, half), name)
            forward = module.forward_weight().detach()
            self.assertTrue(torch.equal(forward, codes32.float() * half[:, None]), name)

    def test_codes_are_not_taken_from_the_rounded_scale(self):
        w = boundary_row()
        s = w.abs().mean(dim=1)
        s16 = s.to(torch.float16).to(torch.float32)
        from_fp32 = torch.round(w / s[:, None]).clamp(-1, 1)
        from_fp16 = torch.round(w / s16[:, None]).clamp(-1, 1)
        self.assertFalse(torch.equal(from_fp32, from_fp16), "the fixture must separate the two codings")
        module = teb.HalfScaleTernaryLinear(torch.nn.Parameter(w.clone()), None)
        codes, scale = module.quantized_weight()
        self.assertTrue(torch.equal(codes.float(), from_fp32))
        self.assertTrue(torch.equal(scale, s16))
        self.assertTrue(torch.equal(module.forward_weight().detach(), from_fp32 * s16[:, None]))

    def test_module_forward_uses_the_forward_weight(self):
        model = form_model("S2")
        x = torch.randn(3, 5, 64, generator=torch.Generator().manual_seed(1))
        lin = model.get_submodule("encoder.layers.0.feed_forward1.linear1")
        self.assertTrue(torch.equal(lin(x), F.linear(x, lin.forward_weight(), lin.bias)))
        conv = model.get_submodule("encoder.layers.0.conv.pointwise_conv1")
        xc = torch.randn(2, 64, 7, generator=torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(conv(xc), F.conv1d(xc, conv.forward_weight().unsqueeze(2), conv.bias)))

    def test_identity_gradient_to_the_latent_weight(self):
        model = form_model("S2")
        for name in ("encoder.layers.0.self_attn.linear_q", "encoder.layers.1.conv.pointwise_conv2"):
            module = model.get_submodule(name)
            module.weight.grad = None
            g = torch.randn(module.forward_weight().shape, generator=torch.Generator().manual_seed(3))
            module.forward_weight().backward(g)
            grad = module.weight.grad
            grad = grad.squeeze(2) if grad.dim() == 3 else grad
            self.assertTrue(torch.equal(grad, g), name)

    def test_fraction_below_one_is_refused(self):
        model = form_model("S2")
        module = model.get_submodule("encoder.layers.0.feed_forward2.linear2")
        module.fraction = 0.5
        with self.assertRaises(ValueError):
            module.forward_weight()
        with self.assertRaises(ValueError):
            module(torch.zeros(1, 2, module.in_features))
        ramped = ternary_model()
        quant.set_weight_fraction(ramped, 0.75)
        with self.assertRaises(ValueError):
            teb.apply_keat_scope(ramped, "S2")
        with self.assertRaises(ValueError):
            teb.apply_keat_scope(ramped, "none")

    def test_same_parameters_after_the_swap(self):
        model = ternary_model()
        before = {n: (model.get_submodule(n).weight, model.get_submodule(n).bias)
                  for n in quant.quantized_module_names(model)}
        teb.apply_keat_scope(model, "S2")
        for n, (w, b) in before.items():
            self.assertIs(model.get_submodule(n).weight, w)
            self.assertIs(model.get_submodule(n).bias, b)
        self.assertEqual(quant.quantized_module_names(model), sorted(before))


class Scopes(unittest.TestCase):
    def test_scopes_are_stage_4s(self):
        self.assertEqual(teb.KEAT_SCOPES, eb.KEAB_SCOPES)
        self.assertEqual(tuple(teb.KEAT_SCOPES), eb.SCOPES)

    def test_each_scope_reads_back(self):
        for scope in eb.SCOPES:
            with self.subTest(scope=scope):
                model = ternary_model()
                latent_pre = model.encoder.pre_encode.out.weight.detach().clone()
                pred = {k: model.get_parameter(k).detach().clone() for k in eb.keys_of(("prediction_network", "joint"))}
                form = teb.apply_keat_scope(model, scope)
                self.assertEqual(form, teb.keat_form(model))
                self.assertEqual(form["scope"], scope)
                parts = eb.KEAB_SCOPES[scope]
                frozen = eb.keys_of([p for p in parts if p in eb.FROZEN_PARTS])
                self.assertEqual(form["frozen"], sorted(frozen))
                self.assertEqual(form["ternary_scales"], "fp16" if scope == "S2" else "fp32")
                self.assertEqual(form["pre_encode_out"], "w8_ste" if scope == "S2" else "fp")
                for key, before in pred.items():
                    now = model.get_parameter(key).detach()
                    if key in frozen:
                        self.assertTrue(torch.equal(now, eb.w8_of(before).to(torch.float32)), key)
                    else:
                        self.assertTrue(torch.equal(now, before), key)
                pre = model.encoder.pre_encode.out
                if scope == "S2":
                    self.assertIsInstance(pre, eb.W8Linear)
                    self.assertTrue(torch.equal(pre.weight.detach(), latent_pre), "the latent weight is kept")
                    self.assertTrue(torch.equal(pre.forward_weight().detach(), eb.w8_of(latent_pre).float()))
                else:
                    self.assertNotIsInstance(pre, eb.W8Linear)
                for n in quant.quantized_module_names(model):
                    module = model.get_submodule(n)
                    self.assertEqual(isinstance(module, teb.HALF_SCALE_TYPES), scope == "S2", n)
                    if scope != "S2":
                        codes, s = quant.ternary_quantize(module.matrix())
                        self.assertTrue(torch.equal(module.quantized_weight()[1], s))

    def test_unknown_scope_is_refused(self):
        with self.assertRaises(ValueError):
            teb.apply_keat_scope(ternary_model(), "S3")


class Refusals(unittest.TestCase):
    def test_binary_model_is_not_a_keat_form(self):
        model = tiny_model(seed=11)
        eb.binarize(model)
        with self.assertRaises(ValueError):
            teb.apply_keat_scope(model, "S1")
        with self.assertRaises(ValueError):
            teb.keat_form(model)

    def test_second_form_is_refused(self):
        for scope in ("S1", "S2"):
            model = form_model(scope)
            with self.assertRaises(ValueError):
                teb.apply_keat_scope(model, scope)

    def test_weight_changed_after_load_is_refused(self):
        model = form_model("S1")
        with torch.no_grad():
            model.get_parameter("decoder.prediction.embed.weight")[0, 0] += 1.0
        with self.assertRaises(ValueError):
            teb.keat_form(model)

    def test_mixed_scales_are_refused(self):
        model = form_model("S2")
        name = quant.quantized_module_names(model)[0]
        module = model.get_submodule(name)
        plain = (quant.TernaryPointwiseConv1d if isinstance(module, quant.TernaryPointwiseConv1d)
                 else quant.TernaryLinear)(module.weight, module.bias)
        model.set_submodule(name, plain)
        with self.assertRaises(ValueError):
            teb.keat_form(model)

    def test_palettized_part_is_refused(self):
        model = ternary_model()
        eb.palettize_frozen(model, ("prediction_network",))
        with self.assertRaises(ValueError):
            teb.keat_form(model)
        with self.assertRaises(ValueError):
            teb.apply_keat_scope(model, "S1")


class Ternarize(unittest.TestCase):
    def test_every_pattern_module_keeps_its_parameters(self):
        model = tiny_model(seed=3)
        targets = sorted(n for n, _ in model.named_modules() if quant.QUANTIZED_PATTERN.fullmatch(n))
        params = {n: model.get_submodule(n).weight for n in targets}
        names = teb.ternarize(model)
        self.assertEqual(names, targets)
        self.assertEqual(len(names), TINY_MODULES)
        for n in names:
            module = model.get_submodule(n)
            self.assertIsInstance(module, quant.TERNARY_TYPES)
            self.assertIs(module.weight, params[n])
            self.assertEqual(module.fraction, 1.0)
        with self.assertRaises(ValueError):
            teb.ternarize(model)


if __name__ == "__main__":
    unittest.main()
