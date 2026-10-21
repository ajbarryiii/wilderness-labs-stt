"""Agent A's own correctness tests for TritonBF16Layer / TritonFP16x2Layer (before H's contract tests).

GPU: run through run_tests.py, which holds the GPU lock for the whole process:
    timeout 600 gpu1bit/shared/python -m gpu1bit.triton_bf16.run_tests
"""
from __future__ import annotations

import unittest

import torch

from gpu1bit.shared import api
from gpu1bit.triton_bf16 import kernels
from gpu1bit.triton_bf16.layer import TritonBF16Layer

DEV = torch.device("cuda")
MODES = ("bf16x3", "bf16x3l", "fp16x2", "fp16x2p", "bf16x3p")


def make(n, k, seed=0, bias=False, scale_kind="random"):
    g = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (n, k), generator=g) * 2 - 1).to(torch.int8)
    codes = api.pack_signs(signs)
    if scale_kind == "one":
        scale = torch.ones(n)
    else:
        scale = torch.rand(n, generator=g) * 0.05 + 0.001
    b = torch.randn(n, generator=g) * 0.1 if bias else None
    return codes, scale.float(), b


def activations(shape, seed=1, outliers=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g)
    if outliers:
        x = x * torch.exp(torch.randn(shape, generator=g))       # heavy tails, wide magnitude range
    return x.to(DEV)


class Layer(unittest.TestCase):
    def check(self, layer, kind, x, codes, scale, bias, k, tol):
        y = api.call(layer, kind, x)
        ref = api.reference(kind, x, codes, scale, bias, k)
        self.assertEqual(y.shape, ref.shape)
        self.assertEqual(y.dtype, torch.float32)
        self.assertTrue(torch.isfinite(y).all())
        err = api.relative_error(y, ref)
        self.assertLess(err, tol, f"{kind} {tuple(x.shape)} rel err {err:.3e}")
        return y, ref

    def test_integer_exact(self):
        # Integer data, scale 1: every exact result is an integer; elementwise |err| < 0.2.
        for mode in MODES:
            for n, k in [(16, 64), (65, 1000), (8, 4096)]:
                codes, scale, _ = make(n, k, scale_kind="one")
                layer = TritonBF16Layer.prepare(codes, scale, None, k, DEV, mode=mode)
                x = torch.randint(-50, 51, (37, k), generator=torch.Generator().manual_seed(n)).float().to(DEV)
                y = layer.linear(x)
                ref = api.reference_linear(x, codes, scale, None, k)
                self.assertLess((y.double() - ref).abs().max().item(), 0.2, (mode, n, k))

    def test_ragged_shapes_and_bias(self):
        for mode in MODES:
            for n, k, m, bias in [(1, 1, 1, False), (3, 5, 7, True), (17, 33, 65, False), (64, 100, 1, True),
                                  (65, 1023, 129, False), (130, 257, 300, True), (1024, 4096, 38, False)]:
                codes, scale, b = make(n, k, seed=n + k, bias=bias)
                layer = TritonBF16Layer.prepare(codes, scale, b, k, DEV, mode=mode)
                self.assertLessEqual(layer.weight_bytes(), api.weight_budget_bytes(n, k, bias))
                self.check(layer, "linear", activations((m, k), seed=m), codes, scale, b, k, 2e-6)

    def test_linear_ranks_and_strides(self):
        n, k = 96, 200
        codes, scale, b = make(n, k, bias=True)
        for mode in MODES:
            layer = TritonBF16Layer.prepare(codes, scale, b, k, DEV, mode=mode)
            base = activations((3, 40, 2 * k), seed=3)
            cases = [activations((k,), seed=4), activations((5, k), seed=5), base[..., :k], base[:, ::2, k:],
                     activations((2, 3, 7, k), seed=6), activations((k, 9), seed=7).t(),
                     activations((4, k, 11), seed=8).transpose(1, 2)]
            for x in cases:
                self.check(layer, "linear", x, codes, scale, b, k, 2e-6)

    def test_conv_layouts_and_batch(self):
        n, k = 130, 192
        codes, scale, b = make(n, k, seed=9, bias=True)
        for mode in MODES:
            layer = TritonBF16Layer.prepare(codes, scale, b, k, DEV, mode=mode)
            for bsz, t in [(1, 38), (3, 17), (2, 1)]:
                contiguous = activations((bsz, k, t), seed=t)                 # [B, K, T], T fastest
                transposed = activations((bsz, t, k), seed=t + 1).transpose(1, 2)  # NeMo's view
                sliced = activations((bsz, k, 2 * t), seed=t + 2)[:, :, ::2]
                for x in (contiguous, transposed, sliced):
                    y, _ = self.check(layer, "conv", x, codes, scale, b, k, 2e-6)
                    self.assertEqual(tuple(y.shape), (bsz, n, t))

    def test_empty(self):
        codes, scale, _ = make(32, 64)
        layer = TritonBF16Layer.prepare(codes, scale, None, 64, DEV)
        self.assertEqual(tuple(layer.linear(torch.empty(0, 64, device=DEV)).shape), (0, 32))
        self.assertEqual(tuple(layer.linear(torch.empty(2, 0, 64, device=DEV)).shape), (2, 0, 32))
        self.assertEqual(tuple(layer.conv(torch.empty(1, 64, 0, device=DEV)).shape), (1, 32, 0))

    def test_split_k_serial_counters_and_determinism(self):
        n, k = 1024, 4096
        codes, scale, _ = make(n, k, seed=11)
        x = activations((126, k), seed=12)
        ref = api.reference_linear(x, codes, scale, None, k)
        for mode in MODES:
            for config in [(32, 64, 64, 8, 4, 2, "serial"), (64, 64, 128, 4, 4, 2, "serial"),
                           (32, 64, 64, 8, 4, 2, "workspace"), (16, 128, 64, 16, 4, 3, "serial")]:
                layer = TritonBF16Layer.prepare(codes, scale, None, k, DEV, mode=mode, configs={126: config},
                                                counters=True)
                outs = [layer.linear(x) for _ in range(5)]
                torch.cuda.synchronize()
                self.assertEqual(int(layer.counters.abs().sum()), 0, "counters must be reset")
                for o in outs[1:]:
                    self.assertTrue(torch.equal(o, outs[0]), (mode, config))
                self.assertLess(api.relative_error(outs[0], ref), 2e-6, (mode, config))

    def test_graph_capture_replay(self):
        n, k = 2048, 1024
        codes, scale, b = make(n, k, seed=13, bias=True)
        for mode in MODES:
            for kind, shape in [("linear", (1, 38, k)), ("conv", (1, k, 126))]:
                layer = TritonBF16Layer.prepare(codes, scale, b, k, DEV, mode=mode)
                x = activations(shape, seed=14)
                api.call(layer, kind, x)            # eager warm-up (compile, scratch)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(graph):
                        y = api.call(layer, kind, x)
                torch.cuda.current_stream().wait_stream(stream)
                for seed in (15, 16, 17):
                    x.copy_(activations(shape, seed=seed))
                    graph.replay()
                    eager = api.call(layer, kind, x)
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(y, eager), (mode, kind))
                del graph

    def test_alternate_stream(self):
        n, k = 256, 512
        codes, scale, _ = make(n, k, seed=18)
        layer = TritonBF16Layer.prepare(codes, scale, None, k, DEV)
        x = activations((50, k), seed=19)
        eager = layer.linear(x)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            y = layer.linear(x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(y, eager))

    def test_rejects(self):
        codes, scale, _ = make(8, 13)
        layer = TritonBF16Layer.prepare(codes, scale, None, 13, DEV)
        with self.assertRaises(ValueError):
            layer.linear(torch.zeros(4, 13, device=DEV, dtype=torch.float16))
        with self.assertRaises(ValueError):
            layer.linear(torch.zeros(4, 12, device=DEV))
        with self.assertRaises(ValueError):
            layer.conv(torch.zeros(4, 13, device=DEV))
        with self.assertRaises(RuntimeError):
            layer.linear(torch.zeros(4, 13, device=DEV, requires_grad=True))
        bad = codes.clone()
        bad[0, -1] |= 0x80                       # nonzero tail bit (K = 13 uses 5 bits of byte 1)
        with self.assertRaises(ValueError):
            TritonBF16Layer.prepare(bad, scale, None, 13, DEV)
        with torch.no_grad():
            self.assertFalse(layer.linear(torch.zeros(4, 13, device=DEV, requires_grad=True)).requires_grad)

    def test_output_is_new(self):
        codes, scale, _ = make(64, 64)
        layer = TritonBF16Layer.prepare(codes, scale, None, 64, DEV)
        x = activations((10, 64))
        a, b = layer.linear(x), layer.linear(x)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        self.assertNotEqual(a.data_ptr(), x.data_ptr())

    def test_wide_range_and_cancellation(self):
        n, k = 64, 1024
        codes, scale, _ = make(n, k, seed=20)
        signs = api.unpack_signs(codes, k)
        for mode in MODES:
            layer = TritonBF16Layer.prepare(codes, scale, None, k, DEV, mode=mode)
            # Rows spanning 1e-30 .. 1e30 in magnitude (per row), zero rows, single nonzero entries.
            x = torch.zeros(6, k)
            x[0] = torch.randn(k) * 1e30
            x[1] = torch.randn(k) * 1e-30
            x[2] = torch.randn(k) * torch.logspace(-20, 20, k)
            x[4, 17] = 3.0
            x[5] = signs[0].float() * torch.rand(k)  # row 0 of W: exact large positive sum
            x = x.to(DEV)
            y = layer.linear(x)
            ref = api.reference_linear(x, codes, scale, None, k)
            self.assertTrue(torch.isfinite(y).all(), mode)
            self.assertTrue(torch.equal(y[3], torch.zeros_like(y[3])), mode)
            for r in range(6):
                if r == 3:
                    continue
                err = api.relative_error(y[r], ref[r])
                self.assertLess(err, 2e-6, (mode, r, err))


class Project(unittest.TestCase):
    """layer.project (M1 provider semantics, gpu1bit/shared/provider.py) vs an FP64 composition."""

    def reference(self, x, codes, scale, bias, k, conv, norm, input_activation, activation):
        xx = (x.transpose(1, 2) if conv else x).double()
        if norm is not None:
            xx = torch.nn.functional.layer_norm(xx, (k,), norm.weight.double(), norm.bias.double(), norm.eps)
        if input_activation == "silu":
            xx = torch.nn.functional.silu(xx)
        y = api.reference_linear(xx, codes, scale, bias, k)
        if activation == "silu":
            y = torch.nn.functional.silu(y)
        return y.transpose(1, 2) if conv else y

    def test_project_semantics(self):
        torch.manual_seed(0)
        for mode in ("bf16x3", "fp16x2", "bf16x3p", "fp16x2p"):
            for n, k, conv, use_norm, ia, act, shape in [
                    (3072, 1024, False, True, "", "", (1, 38, 1024)),          # fused QKV with norm
                    (4096, 1024, False, True, "", "silu", (1, 126, 1024)),     # ff linear1
                    (1024, 4096, False, False, "", "", (1, 38, 4096)),         # ff linear2
                    (2048, 1024, True, True, "", "", (1, 1024, 38)),           # pointwise_conv1
                    (1024, 1024, True, False, "silu", "", (1, 1024, 126)),     # pointwise_conv2
                    (96, 200, False, True, "silu", "silu", (2, 17, 200)),      # batch 2, ragged
                    (130, 192, True, True, "silu", "silu", (3, 192, 9))]:
                codes, scale, b = make(n, k, seed=n + k, bias=n % 2 == 0)
                layer = TritonBF16Layer.prepare(codes, scale, b, k, DEV, mode=mode)
                norm = None
                if use_norm:
                    norm = torch.nn.LayerNorm(k).to(DEV)
                    with torch.no_grad():
                        norm.weight.copy_(torch.rand(k) + 0.5)
                        norm.bias.copy_(torch.randn(k) * 0.1)
                if conv:   # pw1 gets NeMo's transposed view, pw2 the depthwise output [B, K, T]
                    x = (activations((shape[0], shape[2], k)).transpose(1, 2) if use_norm
                         else activations(shape))
                else:
                    x = activations(shape) + 0.3       # nonzero row means exercise the LayerNorm
                with torch.inference_mode():
                    y = layer.project(x, conv=conv, components=3, activation=act, norm=norm, input_activation=ia)
                ref = self.reference(x, codes, scale, b, k, conv, norm, ia, act)
                self.assertEqual(tuple(y.shape), tuple(ref.shape))
                err = api.relative_error(y, ref)
                self.assertLess(err, 1e-6, (mode, n, k, conv, use_norm, ia, act, err))


class PDL(unittest.TestCase):
    """Programmatic dependent launch inside an op (kernels.PDL): same results, graph-capturable."""

    def test_pdl_matches_and_captures(self):
        old = kernels.PDL
        try:
            n, k = 1024, 1024
            codes, scale, b = make(n, k, seed=31, bias=True)
            norm = torch.nn.LayerNorm(k).to(DEV)
            x = activations((1, 38, k), seed=32) + 0.2
            for mode, split_cfg in [(mode, cfg) for mode in ("bf16x3", "bf16x3p") for cfg in
                                    [(16, 64, 64, 12, 4, 3, "workspace", 1, 0), (16, 64, 64, 1, 4, 3, "workspace", 1, 0)]]:
                layer = TritonBF16Layer.prepare(codes, scale, b, k, DEV, mode=mode, configs={38: split_cfg})
                with torch.inference_mode():
                    kernels.PDL = False
                    plain = layer.project(x, norm=norm, activation="silu")
                    kernels.PDL = True
                    outs = [layer.project(x, norm=norm, activation="silu") for _ in range(3)]
                    torch.cuda.synchronize()
                    for o in outs:
                        self.assertTrue(torch.equal(o, plain), split_cfg)
                    graph = torch.cuda.CUDAGraph()
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        with torch.cuda.graph(graph):
                            y = layer.project(x, norm=norm, activation="silu")
                    torch.cuda.current_stream().wait_stream(s)
                    for seed in (33, 34):
                        x.copy_(activations((1, 38, k), seed=seed))
                        graph.replay()
                        kernels.PDL = False
                        ref = layer.project(x, norm=norm, activation="silu")
                        kernels.PDL = True
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(y, ref), split_cfg)
                    del graph
        finally:
            kernels.PDL = old


if __name__ == "__main__":
    unittest.main()
