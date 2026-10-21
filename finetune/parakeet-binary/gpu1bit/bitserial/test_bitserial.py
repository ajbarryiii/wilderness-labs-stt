"""Agent C's own correctness tests for BitSerialLayer (before/alongside H's contract tests).

Run on the GPU under the lock:
  timeout 600 finetune/parakeet-binary/gpu1bit/shared/python \
      finetune/parakeet-binary/gpu1bit/bitserial/run_locked.py C:tests -m unittest gpu1bit.bitserial.test_bitserial -v
"""
from __future__ import annotations

import unittest

import torch

from gpu1bit.shared import api
from gpu1bit.bitserial.layer import BitSerialLayer, IMPLEMENTATION
from gpu1bit.bitserial.kernels import Config

DEV = 'cuda'
TOL = 2e-6  # relative Frobenius error vs FP64 on Gaussian data; FP32 accumulation gives ~1e-7


def random_weights(n, k, *, bias=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (n, k), generator=g, dtype=torch.int8) * 2 - 1).to(torch.int8)
    codes = api.pack_signs(signs)
    scale = torch.rand(n, generator=g) * 0.05 + 0.001
    b = torch.randn(n, generator=g) * 0.1 if bias else None
    return codes, scale, b


def rel(y, ref):
    return api.relative_error(y, ref)


class Semantics(unittest.TestCase):
    def test_interface(self):
        self.assertIs(IMPLEMENTATION, BitSerialLayer)
        codes, scale, _ = random_weights(8, 8)
        layer = BitSerialLayer.prepare(codes, scale, None, 8, DEV)
        self.assertIsInstance(layer, api.BinaryLayer)

    def test_integer_exact(self):
        for (m, n, k) in [(1, 1, 1), (3, 5, 7), (17, 33, 65), (38, 1024, 1024), (5, 64, 4096)]:
            codes, _, _ = random_weights(n, k, seed=m + n + k)
            scale = torch.ones(n)
            layer = BitSerialLayer.prepare(codes, scale, None, k, DEV)
            x = torch.randint(-8, 9, (m, k), device=DEV).float()
            y = layer.linear(x)
            ref = api.reference_linear(x, codes, scale, None, k)
            self.assertTrue(torch.equal(y.double(), ref), (m, n, k))

    def test_weight_bytes_and_ownership(self):
        for n, k, bias in [(1024, 1024, False), (7, 37, True), (3, 1, False)]:
            codes, scale, b = random_weights(n, k, bias=bias)
            codes_gpu = codes.to(DEV)
            layer = BitSerialLayer.prepare(codes_gpu, scale.to(DEV), None if b is None else b.to(DEV), k, DEV)
            self.assertEqual(layer.weight_bytes(), n * api.packed_row_bytes(k) + 4 * n + (4 * n if bias else 0))
            self.assertLessEqual(layer.weight_bytes(), api.weight_budget_bytes(n, k, bias))
            self.assertNotEqual(layer.codes.data_ptr(), codes_gpu.data_ptr())

    def test_rejects(self):
        codes, scale, _ = random_weights(4, 13)
        bad = codes.clone()
        bad[0, -1] |= 0x80  # tail bit of K=13 (bits 13..15 unused)
        with self.assertRaises(ValueError):
            BitSerialLayer.prepare(bad, scale, None, 13, DEV)
        layer = BitSerialLayer.prepare(codes, scale, None, 13, DEV)
        with self.assertRaises(ValueError):
            layer.linear(torch.randn(2, 12, device=DEV))
        with self.assertRaises(ValueError):
            layer.linear(torch.randn(2, 13, device=DEV, dtype=torch.float64))
        with self.assertRaises(ValueError):
            layer.linear(torch.randn(2, 13))
        with self.assertRaises(ValueError):
            layer.conv(torch.randn(2, 13, device=DEV))
        x = torch.randn(2, 13, device=DEV, requires_grad=True)
        with self.assertRaises(RuntimeError):
            layer.linear(x)
        with torch.no_grad():
            self.assertFalse(layer.linear(x).requires_grad)


class Numerics(unittest.TestCase):
    def check(self, layer, x, kind, codes, scale, bias, k, tol=TOL):
        y = api.call(layer, kind, x)
        ref = api.reference(kind, x, codes, scale, bias, k)
        self.assertEqual(tuple(y.shape), tuple(ref.shape))
        self.assertTrue(torch.isfinite(y).all())
        e = rel(y, ref)
        self.assertLess(e, tol, (kind, tuple(x.shape), x.stride(), e))
        return y

    def test_linear_shapes(self):
        shapes = [(1, 8, 8), (2, 5, 37), (3, 130, 100), (38, 1024, 1024), (126, 4096, 1024), (126, 1024, 4096),
                  (376, 2048, 1024), (65, 3072, 1024), (13, 1000, 999), (1, 1024, 4096), (7, 33, 4100)]
        for (m, n, k) in shapes:
            codes, scale, _ = random_weights(n, k, seed=m * 7 + n + k)
            layer = BitSerialLayer.prepare(codes, scale, None, k, DEV)
            with torch.inference_mode():
                self.check(layer, torch.randn(m, k, device=DEV), 'linear', codes, scale, None, k)

    def test_bias_and_leading_dims(self):
        codes, scale, b = random_weights(96, 160, bias=True, seed=3)
        layer = BitSerialLayer.prepare(codes, scale, b, 160, DEV)
        with torch.no_grad():
            for shape in [(160,), (4, 160), (2, 9, 160), (2, 3, 5, 160), (0, 160), (2, 0, 160)]:
                x = torch.randn(*shape, device=DEV)
                y = layer.linear(x)
                self.assertEqual(tuple(y.shape), (*shape[:-1], 96))
                if x.numel():
                    ref = api.reference_linear(x, codes, scale, b, 160)
                    self.assertLess(rel(y, ref), TOL)

    def test_conv_layouts_and_batch(self):
        for (b, n, k, t) in [(1, 1024, 1024, 38), (3, 2048, 1024, 126), (2, 64, 37, 5), (1, 128, 1024, 376)]:
            codes, scale, _ = random_weights(n, k, seed=b + t)
            layer = BitSerialLayer.prepare(codes, scale, None, k, DEV)
            with torch.inference_mode():
                contiguous = torch.randn(b, k, t, device=DEV)                 # [B, K, T], T contiguous
                nemo = torch.randn(b, t, k, device=DEV).transpose(1, 2)       # transposed view (NeMo)
                sliced = torch.randn(b, 2 * k, t + 3, device=DEV)[:, ::2, 1:t + 1]  # generic strides
                for x in (contiguous, nemo, sliced):
                    y = self.check(layer, x, 'conv', codes, scale, None, k)
                    lin = layer.linear(x.transpose(1, 2))
                    self.assertTrue(torch.equal(y.transpose(1, 2), lin))

    def test_strided_linear(self):
        codes, scale, _ = random_weights(256, 1024, seed=9)
        layer = BitSerialLayer.prepare(codes, scale, None, 1024, DEV)
        base = torch.randn(4, 50, 2048, device=DEV)
        with torch.no_grad():
            for x in (base[:, :, ::2], base[:, :, 1:1025], base.transpose(0, 1)[:, :, :1024],
                      torch.randn(1024, 70, device=DEV).t()):
                self.check(layer, x, 'linear', codes, scale, None, 1024)

    def test_activations(self):
        codes, scale, b = random_weights(256, 512, bias=True, seed=11)
        layer = BitSerialLayer.prepare(codes, scale, b, 512, DEV)
        x = torch.randn(40, 512, device=DEV)
        with torch.no_grad():
            ref = api.reference_linear(x, codes, scale, b, 512)
            y = layer.linear(x, activation='silu')
            self.assertLess(rel(y, torch.nn.functional.silu(ref)), TOL)
            y = layer.linear(x, input_activation='silu')
            ref_in = api.reference_linear(torch.nn.functional.silu(x.double()), codes, scale, b, 512)
            self.assertLess(rel(y, ref_in), TOL)

    def test_every_config(self):
        m, n, k = 70, 384, 1024
        codes, scale, _ = random_weights(n, k, seed=21)
        layer = BitSerialLayer.prepare(codes, scale, None, k, DEV)
        x = torch.randn(m, k, device=DEV)
        xt = torch.randn(k, m, device=DEV).t()
        ref = api.reference_linear(x, codes, scale, None, k)
        ref_t = api.reference_linear(xt, codes, scale, None, k)
        configs = [Config(bm, bn, bk, tm, tn, split) for bm, bn, bk, tm, tn, split in
                   [(16, 64, 32, 4, 4, 1), (16, 128, 64, 4, 8, 3), (32, 64, 32, 4, 8, 2), (32, 128, 32, 8, 4, 8),
                    (64, 128, 32, 8, 8, 1), (64, 64, 64, 8, 8, 4), (64, 128, 32, 4, 8, 5), (128, 128, 32, 8, 8, 2)]]
        for config in configs:
            if not config.valid():
                continue
            layer.config_override = config
            with torch.no_grad():
                self.assertLess(rel(layer.linear(x), ref), TOL, config)
                self.assertLess(rel(layer.linear(xt), ref_t), TOL, config)
        layer.config_override = None

    def test_special_values(self):
        codes, scale, b = random_weights(64, 300, bias=True, seed=5)
        layer = BitSerialLayer.prepare(codes, scale, b, 300, DEV)
        with torch.no_grad():
            y = layer.linear(torch.zeros(3, 300, device=DEV))
            self.assertTrue(torch.equal(y, b.to(DEV).expand(3, -1)))   # silence -> bias exactly
            x = torch.zeros(2, 300, device=DEV)
            x[0, 17] = 3.0e4
            x[1, 299] = -1.0e-30
            self.check(layer, x, 'linear', codes, scale, b, 300)
            x = torch.randn(5, 300, device=DEV) * torch.logspace(-20, 20, 300, device=DEV)
            self.check(layer, x, 'linear', codes, scale, b, 300, tol=1e-5)


class Project(unittest.TestCase):
    """layer.project == provider semantics (ExpandedProjection.__call__), checked against FP64."""

    def test_project_matches_reference(self):
        import torch.nn.functional as F
        for (kind, b, t, n, k) in [('linear', 1, 38, 4096, 1024), ('conv', 1, 38, 2048, 1024),
                                   ('conv', 2, 13, 96, 1024), ('linear', 3, 7, 64, 200)]:
            codes, scale, bias = random_weights(n, k, bias=True, seed=t + n)
            layer = BitSerialLayer.prepare(codes, scale, bias, k, DEV)
            norm = torch.nn.LayerNorm(k).to(DEV)
            with torch.no_grad():
                norm.weight.copy_(torch.rand(k) + 0.5)
                norm.bias.copy_(torch.randn(k) * 0.1)
            for use_norm, act, in_act in [(True, '', ''), (True, 'silu', ''), (False, '', 'silu'),
                                          (False, '', ''), (True, 'silu', 'silu')]:
                with torch.inference_mode():
                    base = torch.randn(b, t, k, device=DEV) * 3 + 1
                    x = base.transpose(1, 2) if kind == 'conv' else base
                    y = layer.project(x, conv=kind == 'conv', components=3, activation=act,
                                      norm=norm if use_norm else None, input_activation=in_act)
                    z = base.double()
                    if use_norm:
                        z = F.layer_norm(z, (k,), norm.weight.double(), norm.bias.double(), norm.eps)
                    if in_act:
                        z = F.silu(z)
                    ref = api.reference_linear(z, codes, scale, bias, k)
                    if act:
                        ref = F.silu(ref)
                    if kind == 'conv':
                        ref = ref.transpose(1, 2)
                    self.assertEqual(tuple(y.shape), tuple(ref.shape))
                    self.assertLess(rel(y, ref), TOL, (kind, use_norm, act, in_act))


class Execution(unittest.TestCase):
    def setUp(self):
        self.codes, self.scale, _ = random_weights(1024, 1024, seed=31)
        self.layer = BitSerialLayer.prepare(self.codes, self.scale, None, 1024, DEV)

    def test_deterministic(self):
        x = torch.randn(126, 1024, device=DEV)
        with torch.no_grad():
            ys = [self.layer.linear(x) for _ in range(3)]
        for y in ys[1:]:
            self.assertTrue(torch.equal(y, ys[0]))
        self.assertNotEqual(ys[0].data_ptr(), ys[1].data_ptr())

    def test_graph_capture(self):
        for kind, x in (('linear', torch.randn(126, 1024, device=DEV)),
                        ('conv', torch.randn(1, 38, 1024, device=DEV).transpose(1, 2))):
            with torch.inference_mode():
                api.call(self.layer, kind, x)  # warm-up compiles
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = api.call(self.layer, kind, x)
                x.copy_(torch.randn_like(x))
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(out, api.call(self.layer, kind, x)), kind)
                del graph

    def test_alternate_stream(self):
        x = torch.randn(38, 1024, device=DEV)
        with torch.no_grad():
            expect = self.layer.linear(x)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                y = self.layer.linear(x)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
        self.assertTrue(torch.equal(y, expect))


if __name__ == '__main__':
    unittest.main()
