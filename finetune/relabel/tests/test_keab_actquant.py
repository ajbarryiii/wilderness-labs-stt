"""Contract tests (CPU): the stage 4 activation quantizer (keab-prime plan "Activation quantization", "Code and review":
the quantizer and the quantization boundary). The kernel comparison is tests/test_keab_actquant_gpu.py (CUDA).

- the formula: per frame across input channels, s = max(max|x| / 127, 1e-30), q = round-half-even(x / s), x̂ = q s;
  zero rows (s = 1e-30, q = 0), tiny inputs (the floor), rounding ties (half to even), large inputs;
- the channel axis: the last axis of a linear module's [..., K] input; the channel axis of a pointwise convolution's
  [B, C, T] input, per (b, t); frames in [B, T] order;
- FP32 computation from BF16 and FP32 inputs, the output in the input's dtype;
- the straight-through gradient (the identity to the input);
- the boundary on a NeMo conformer (the tiny model with every binary module): exactly one quantizer on the input of
  every binary module, none on pre_encode.out, the prediction or joint networks; the quantized tensor is the
  LayerNorm output for feed_forward1/2.linear1, conv.pointwise_conv1 and self_attn.linear_q, the post-SiLU input of
  linear2 and pointwise_conv2, the shared normalized input of linear_q, linear_k and linear_v (quantized
  identically), the positional embeddings for linear_pos.
"""
from __future__ import annotations

import unittest

import torch

import train  # noqa: F401  finetune/parakeet-ternary first
from decoder_fixtures import tiny_model
from training import actquant
from training import eightbit as eb

import binary_quant  # noqa: E402  (on sys.path through training.eightbit)


def manual(x: torch.Tensor):
    """The formula row by row in Python floats (FP32 rounding of the division emulated by torch on one value)."""
    rows = x.float().reshape(-1, x.shape[-1])
    q, s = [], []
    for row in rows:
        m = float(row.abs().max())
        scale = max(float(torch.tensor(m, dtype=torch.float32) / torch.tensor(127.0)), 1e-30)
        scale = float(torch.tensor(scale, dtype=torch.float32))
        q.append(torch.round(row / torch.tensor(scale, dtype=torch.float32)))
        s.append(scale)
    return torch.stack(q).to(torch.int8), torch.tensor(s, dtype=torch.float32)


class Formula(unittest.TestCase):
    def test_zero_rows_tiny_inputs_ties_and_large_values(self):
        x = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                     # a zero row
                          [1e-35, -1e-34, 0.0, 3e-36, 0.0, 0.0],              # max/127 below the floor
                          [127.0, 0.5, 1.5, 2.5, -0.5, -2.5],                 # s = 1: ties to even
                          [254.0, 1.0, 3.0, 5.0, -1.0, -7.0],                 # s = 2: ties at x / s = k + 0.5
                          [3e30, -1e30, 2e29, 0.0, 1.0, -3e30]])              # large values
        y, q, s = actquant.reference(x)
        floor = torch.tensor(actquant.FLOOR, dtype=torch.float32).item()          # 1e-30 in FP32
        self.assertEqual(s[0].item(), floor)
        self.assertTrue(torch.equal(q[0], torch.zeros(6, dtype=torch.int8)))
        self.assertTrue(torch.equal(y[0], torch.zeros(6)))
        self.assertEqual(s[1].item(), floor)
        self.assertEqual(q[1].tolist(), [round(1e-35 / s[1].item()), round(-1e-34 / s[1].item()), 0, 0, 0, 0])
        self.assertEqual(s[2].item(), 1.0)
        self.assertEqual(q[2].tolist(), [127, 0, 2, 2, 0, -2])                # round half to even
        self.assertEqual(s[3].item(), 2.0)
        self.assertEqual(q[3].tolist(), [127, 0, 2, 2, 0, -4])
        self.assertEqual(q[4].abs().max().item(), 127)
        mq, ms = manual(x)
        self.assertTrue(torch.equal(q, mq))
        self.assertTrue(torch.equal(s, ms))
        self.assertTrue(torch.equal(y, (q.float() * s[:, None])))

    def test_random_inputs_match_the_formula(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, 7, 33, generator=g) * torch.logspace(-3, 3, 33)
        y, q, s = actquant.reference(x)
        mq, ms = manual(x)
        self.assertTrue(torch.equal(q, mq))
        self.assertTrue(torch.equal(s, ms))
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.equal(y, (q.float() * s[:, None]).reshape(x.shape)))

    def test_the_channel_axis_and_the_frame_order(self):
        g = torch.Generator().manual_seed(1)
        conv = torch.randn(2, 5, 3, generator=g)                              # [B, C, T]
        y, q, s = actquant.reference(conv, actquant.CONV_DIM)
        frames = conv.transpose(1, 2).reshape(-1, 5)                         # [B T, C], frame m = b T + t
        mq, ms = manual(frames)
        self.assertTrue(torch.equal(q, mq))
        self.assertTrue(torch.equal(s, ms))
        self.assertTrue(torch.equal(y, (mq.float() * ms[:, None]).reshape(2, 3, 5).transpose(1, 2)))
        lin = conv.transpose(1, 2).contiguous()                               # the same frames as [B, T, K]
        y2, q2, s2 = actquant.reference(lin)
        self.assertTrue(torch.equal(q2, q) and torch.equal(s2, s))
        self.assertTrue(torch.equal(y2.transpose(1, 2), y))
        with self.assertRaises(ValueError):
            actquant.quantize(torch.ones(2, 3, 4, 5), 1)                    # not a binary module's layout

    def test_fp32_computation_from_bf16_and_fp32_with_the_output_in_the_input_dtype(self):
        g = torch.Generator().manual_seed(2)
        x32 = torch.randn(3, 64, generator=g) * 7
        xb = x32.to(torch.bfloat16)
        y, q, s = actquant.quantize(xb)
        q32, s32 = manual(xb.float())                                         # FP32 arithmetic on the BF16 values
        self.assertTrue(torch.equal(q, q32) and torch.equal(s, s32))
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(y, (q32.float() * s32[:, None]).to(torch.bfloat16)))
        y32, _, _ = actquant.quantize(x32)
        self.assertEqual(y32.dtype, torch.float32)
        self.assertEqual(actquant.fake_quant(xb).dtype, torch.bfloat16)

    def test_the_straight_through_gradient(self):
        for dtype, dim, shape in ((torch.float32, -1, (2, 3, 8)), (torch.bfloat16, -1, (5, 8)),
                                  (torch.float32, 1, (2, 8, 3))):
            x = (torch.randn(*shape) * 3).to(dtype).requires_grad_(True)
            w = torch.randn(*shape).to(dtype)
            y = actquant.fake_quant(x, dim)
            self.assertFalse(torch.equal(y.detach(), x.detach()))            # it does quantize
            (y * w).sum().backward()
            self.assertEqual(x.grad.dtype, dtype)
            self.assertTrue(torch.equal(x.grad, w))                          # identity: the upstream gradient


class Boundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = tiny_model(3)
        cls.names = eb.binarize(cls.model, "ternary")
        eb.apply_keab_scope(cls.model, "S2")                                  # pre_encode.out is a W8Linear here
        actquant.install(cls.model)

    def test_one_quantizer_on_every_binary_module_and_none_elsewhere(self):
        found = actquant.hooks(self.model)
        self.assertEqual(sorted(found), self.names)
        self.assertTrue(all(len(v) == 1 for v in found.values()))
        self.assertEqual(len(self.names), 11 * len(self.model.encoder.layers))
        for name in ("encoder.pre_encode.out", "encoder.pre_encode", "decoder", "joint", "joint.enc"):
            self.assertNotIn(name, found)
        self.assertFalse(any(n.startswith(("decoder", "joint", "encoder.pre_encode")) for n in found))
        dims = {n: h[0].dim for n, h in found.items()}
        self.assertTrue(all(dims[n] == actquant.CONV_DIM for n in self.names if "pointwise_conv" in n))
        self.assertTrue(all(dims[n] == actquant.LINEAR_DIM for n in self.names if "pointwise_conv" not in n))
        self.assertEqual(actquant.verify(self.model)["modules"], len(self.names))
        with self.assertRaises(ValueError):
            actquant.install(self.model)                                     # never twice
        bare = tiny_model(4)
        eb.binarize(bare, "ternary")
        with self.assertRaises(ValueError):
            actquant.verify(bare)                                             # none installed
        with self.assertRaises(ValueError):
            actquant.install(bare, expected=binary_quant.EXPECTED_MODULES)   # the production count is 264

    def capture(self):
        """One eval-mode forward: every quantizer's input and output, and the reference tensors of the boundary."""
        seen, ref = {}, {}
        original = actquant.Hook.__call__

        def record(hook, module, inputs):
            out = original(hook, module, inputs)
            seen[hook.name] = (inputs[0].detach().clone(), out[0].detach().clone())
            return out

        handles = []
        for i, layer in enumerate(self.model.encoder.layers):
            p = f"encoder.layers.{i}"
            for mod, key in ((layer.norm_feed_forward1, "norm_ff1"), (layer.norm_feed_forward2, "norm_ff2"),
                             (layer.norm_conv, "norm_conv"), (layer.norm_self_att, "norm_att"),
                             (layer.feed_forward1.linear1, "lin1_ff1"), (layer.feed_forward2.linear1, "lin1_ff2"),
                             (layer.conv.batch_norm, "bn_conv")):          # (the Swish module may be shared)
                handles.append(mod.register_forward_hook(
                    lambda m, a, o, k=f"{p}.{key}": ref.__setitem__(k, o.detach().clone())))
            handles.append(layer.self_attn.register_forward_pre_hook(
                lambda m, a, kw, k=f"{p}.pos_emb": ref.__setitem__(k, kw["pos_emb"].detach().clone()),
                with_kwargs=True))
        self.model.eval()
        try:
            actquant.Hook.__call__ = record
            with torch.no_grad():
                audio = torch.randn(2, 24000)
                self.model(input_signal=audio, input_signal_length=torch.tensor([24000, 17000]))
        finally:
            actquant.Hook.__call__ = original
            for h in handles:
                h.remove()
        with torch.no_grad():                                                 # post-SiLU: each module's own SiLU
            for i, layer in enumerate(self.model.encoder.layers):
                p = f"encoder.layers.{i}"
                ref[f"{p}.silu_ff1"] = layer.feed_forward1.activation(ref[f"{p}.lin1_ff1"])
                ref[f"{p}.silu_ff2"] = layer.feed_forward2.activation(ref[f"{p}.lin1_ff2"])
                ref[f"{p}.silu_conv"] = layer.conv.activation(ref[f"{p}.bn_conv"])
        return seen, ref

    def test_the_quantized_tensor_is_the_one_the_kernel_quantizes(self):
        seen, ref = self.capture()
        self.assertEqual(sorted(seen), self.names)
        for i in range(len(self.model.encoder.layers)):
            p = f"encoder.layers.{i}"
            eq = lambda a, b: self.assertTrue(torch.equal(a, b), (p, a.shape, b.shape))   # noqa: E731
            eq(seen[f"{p}.feed_forward1.linear1"][0], ref[f"{p}.norm_ff1"])            # after the fused LayerNorm
            eq(seen[f"{p}.feed_forward2.linear1"][0], ref[f"{p}.norm_ff2"])
            eq(seen[f"{p}.conv.pointwise_conv1"][0], ref[f"{p}.norm_conv"].transpose(1, 2))
            eq(seen[f"{p}.self_attn.linear_q"][0], ref[f"{p}.norm_att"])
            for kv in ("linear_k", "linear_v"):                                         # the shared input
                eq(seen[f"{p}.self_attn.{kv}"][0], ref[f"{p}.norm_att"])
                eq(seen[f"{p}.self_attn.{kv}"][1], seen[f"{p}.self_attn.linear_q"][1])  # quantized identically
            eq(seen[f"{p}.feed_forward1.linear2"][0], ref[f"{p}.silu_ff1"])            # post-SiLU (eval: no dropout)
            eq(seen[f"{p}.feed_forward2.linear2"][0], ref[f"{p}.silu_ff2"])
            eq(seen[f"{p}.conv.pointwise_conv2"][0], ref[f"{p}.silu_conv"])
            eq(seen[f"{p}.self_attn.linear_pos"][0], ref[f"{p}.pos_emb"])              # the positional embeddings
            conv_in, conv_out = seen[f"{p}.conv.pointwise_conv1"]
            y, _, _ = actquant.reference(conv_in, actquant.CONV_DIM)                   # across channels, per frame
            eq(conv_out, y)
            lin_in, lin_out = seen[f"{p}.feed_forward1.linear1"]
            eq(lin_out, actquant.reference(lin_in)[0])

    def test_every_forward_quantizes_every_module_once(self):
        hooks = {n: h[0] for n, h in actquant.hooks(self.model).items()}
        before = {n: h.calls for n, h in hooks.items()}
        self.model.eval()
        with torch.no_grad():
            self.model(input_signal=torch.randn(1, 16000), input_signal_length=torch.tensor([16000]))
        self.assertTrue(all(hooks[n].calls == before[n] + 1 for n in hooks))
