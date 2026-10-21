"""Contract of the exact deferred split-K (review c1 finding 1): kernels_exact XCHG 5 + row_exact.

The deferred value is bitwise the SPLIT == 1 result: for every slice configuration, the integer slices
finished by _row_epilogue_exact give exactly the outputs (y, and the fused next-layer q / qs) of
b1_dot's exact in-kernel finish followed by M1's own _row_epilogue with SPLIT = 1, including Codex's
counterexample and cancellations placed across every slice boundary. The b01 form gives identical
integer slices. CUDA-graph replay equals eager. M1's unpatched finish_residual refuses the slices.

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.gpulock gpu1bit.int8_mma.test_exact
"""
from __future__ import annotations

import itertools
import types
import unittest

import torch

from gpu1bit.int8_mma import configs, hybrid_exact, kernels_exact, row_exact
from gpu1bit.int8_mma.kernels import DotConfig
from gpu1bit.int8_mma.layer import Int8MmaLayer, m1_quantize
from gpu1bit.int8_mma.layer_b01 import Int8MmaB01Layer
from gpu1bit.shared import api, env

_LOCK = None

# XCHG 5 configs (wm, wn, rm, rn, stages, cluster, 5, wk, 0): the tuned table plus WK groups, 2-CTA
# M tiles, odd slice counts and wide clusters.
EXTRA = [DotConfig(1, 4, 1, 2, 2, 2, 5), DotConfig(1, 8, 1, 1, 3, 8, 5), DotConfig(1, 4, 2, 2, 2, 3, 5, 2),
         DotConfig(2, 2, 1, 3, 2, 5, 5), DotConfig(1, 4, 1, 4, 4, 6, 5), DotConfig(1, 2, 1, 4, 3, 7, 5, 2)]


def tuned_configs():
    return list(dict.fromkeys(c for c in hybrid_exact.TUNED_EXACT_DEFERRED.values() if c is not None))


def setUpModule():
    global _LOCK
    if not getattr(env.gpu_lock, "_held", False):
        _LOCK = env.gpu_lock("B:exact deferred tests")
        _LOCK.__enter__()


def tearDownModule():
    if _LOCK is not None:
        _LOCK.__exit__(None, None, None)


def weights(n, k, seed, *, plus_one=False, unit_scale=False, bias=False):
    g = torch.Generator().manual_seed(seed)
    signs = torch.ones(n, k, dtype=torch.int8) if plus_one else torch.randint(0, 2, (n, k), generator=g, dtype=torch.int8) * 2 - 1
    scale = torch.ones(n) if unit_scale else torch.rand(n, generator=g) * 0.05 + 1e-3
    b = torch.randn(n, generator=g) * 0.1 if bias else None
    return api.pack_signs(signs), scale, b


def provider():
    return types.SimpleNamespace(row_warps="adaptive", prequantized={},
                                 null=torch.empty(0, device="cuda", dtype=torch.uint8))


def norm_module(n, seed):
    g = torch.Generator().manual_seed(seed)
    ln = torch.nn.LayerNorm(n)
    with torch.no_grad():
        ln.weight.copy_(1 + 0.1 * torch.randn(n, generator=g))
        ln.bias.copy_(0.1 * torch.randn(n, generator=g))
    return ln.cuda()


def reference(layer, q, qs, m, shape, residual, factor, norm, out_norm, order):
    """b1_dot's exact in-kernel finish (TUNED_M config) + M1's _row_epilogue with SPLIT = 1."""
    from inference.row_fusion import DeferredProjection
    n, k = layer.out_features, layer.in_features
    value = torch.empty(shape, device="cuda")
    layer.dot(q, qs, k, m, value.view(m, n), config=configs.choose(m, n, k), order=order)
    p = provider()
    y = row_exact._ORIGINAL(p, DeferredProjection(value, layer.scale, layer.bias, 1, shape), residual, factor, norm, out_norm)
    return y, p.prequantized


def candidate(layer, q, qs, m, shape, residual, factor, norm, out_norm, order, config):
    n, k = layer.out_features, layer.in_features
    cfg = kernels_exact.ExactDeferredConfig.of(config)
    slices = torch.empty(cfg.cluster, 2, m, n, device="cuda", dtype=torch.int32)
    hybrid_exact.dot_slices(layer, q, qs, k, m, slices, cfg, order)
    p = provider()
    value = row_exact.ExactDeferredProjection(slices, layer.scale, layer.bias, cfg.cluster, shape, qs, order)
    y = row_exact.finish_residual_exact(p, value, residual, factor, norm, out_norm)
    return y, p.prequantized, slices


def assert_same(test, ref, got, what):
    y0, pre0 = ref
    y1, pre1 = got[:2]
    test.assertTrue(torch.equal(y0, y1), f"{what}: y differs, max |d| {(y0 - y1).abs().max().item():.3e}")
    test.assertEqual(len(pre0), len(pre1), what)
    for (q0, s0), (q1, s1) in zip(pre0.values(), pre1.values()):
        test.assertTrue(torch.equal(q0, q1), f"{what}: fused q differs")
        test.assertTrue(torch.equal(s0, s1), f"{what}: fused qs differs")


class ExactDeferred(unittest.TestCase):
    def test_codex_counterexample(self):
        m, n, k = 126, 1024, 4096
        codes, scale, _ = weights(n, k, 0, plus_one=True, unit_scale=True)
        x = torch.zeros(1, m, k, device="cuda")
        x[..., 0], x[..., 128], x[..., 1024] = 16256.0, 2.0 ** -12, -16256.0
        q, qs = m1_quantize(x)
        zeros = torch.zeros(1, m, n, device="cuda")
        for cls in (Int8MmaLayer, Int8MmaB01Layer):
            layer = cls.prepare(codes, scale, None, k, "cuda")
            ref = reference(layer, q, qs, m, (1, m, n), zeros, 1.0, None, None, 0)
            self.assertTrue(torch.equal(ref[0], torch.full_like(ref[0], 2.0 ** -12)))
            for cfg in tuned_configs() + EXTRA:
                got = candidate(layer, q, qs, m, (1, m, n), zeros, 1.0, None, None, 0, cfg)
                assert_same(self, ref, got, f"{cls.__name__} {cfg}")
            y = hybrid_exact.deferred_project(layer, x)  # the provider path as one op (default config)
            self.assertTrue(torch.equal(y, ref[0].view(1, m, n)))
            # control: the frozen FP32-slice path loses the residual on this input (the input is not vacuous)
            frozen = hybrid_exact.deferred_project(layer, x, exact=False)
            self.assertTrue(torch.equal(frozen, torch.zeros_like(frozen)))

    def test_split_boundary_cancellation(self):
        """+A and a small d in one slice, -A at the first column of every other slice (and at M1's
        interleaved tile boundary 128), all-+1 weights: the result is d's contribution, never 0."""
        for k in (1024, 4096):
            n = 1024
            codes, scale, _ = weights(n, k, 1, plus_one=True, unit_scale=True)
            layers = [cls.prepare(codes, scale, None, k, "cuda") for cls in (Int8MmaLayer, Int8MmaB01Layer)]
            kb = -(-k // 128)
            for cfg in tuned_configs() + EXTRA:
                c = cfg.cluster
                if c > kb:
                    continue
                cols = sorted({(kb * r // c) * 128 for r in range(1, c)} | {128})
                rows = []
                for col, (big, small) in itertools.product(cols, ((16256.0, 2.0 ** -12), (1016.0, 2.0 ** -20), (31.75, 1.0e-3))):
                    rows.append({0: big, 1: small, col: -big})                 # small in the first slice
                    rows.append({0: big, col: -big, col + 1: small})           # small in the later slice
                    rows.append({col - 1: big, col: -big, 5: small})           # straddling the boundary
                m = len(rows)
                x = torch.zeros(1, m, k, device="cuda")
                for i, entries in enumerate(rows):
                    for j, v in entries.items():
                        x[0, i, j] = v
                q, qs = m1_quantize(x)
                zeros = torch.zeros(1, m, n, device="cuda")
                for layer in layers:
                    ref = reference(layer, q, qs, m, (1, m, n), zeros, 1.0, None, None, 0)
                    self.assertTrue(bool((ref[0] != 0).all()), "exact SPLIT == 1 keeps every small term")
                    got = candidate(layer, q, qs, m, (1, m, n), zeros, 1.0, None, None, 0, cfg)
                    assert_same(self, ref, got, f"K={k} {cfg} {type(layer).__name__}")

    def test_bitwise_random(self):
        """Random weights / activations with outliers; every M tail, bias, norms, factor, both orders."""
        epilogues = [(False, False, False, 1.0), (True, True, False, 0.5), (True, True, True, 0.5), (False, False, True, 1.0),
                     (False, True, False, 0.5), (False, True, True, 1.0), (True, False, False, 1.0), (False, True, True, 0.5)]
        shapes = [(1024, 1024), (1024, 4096), (1001, 1088)]
        for (n, k), m in itertools.product(shapes, (1, 17, 38, 126, 130, 192, 193, 256, 300, 368)):
            for e, (bias, with_norm, with_out, factor) in enumerate(epilogues):
                if m not in (17, 126, 193, 300) and e not in (0, 2, 7):
                    continue
                seed = n + k + m + e
                codes, scale, b = weights(n, k, seed, bias=bias)
                layer = Int8MmaLayer.prepare(codes, scale, b, k, "cuda")
                g = torch.Generator(device="cuda").manual_seed(seed)
                x = torch.randn(1, m, k, device="cuda", generator=g)
                x[..., 7] *= 300.0
                x[..., k // 2] *= -50.0
                residual = torch.randn(1, m, n, device="cuda", generator=g)
                norm = norm_module(n, seed) if with_norm else None
                out_norm = norm_module(n, seed + 1) if with_out else None
                q, qs = m1_quantize(x)
                chosen = hybrid_exact.choose_exact_deferred(m, n, k)
                cfgs = [chosen] if chosen is not None else []
                cfgs += [EXTRA[(m + e) % len(EXTRA)], EXTRA[(m + e + 3) % len(EXTRA)]]
                for order in ((0, 1) if e == 2 else (0,)):
                    ref = reference(layer, q, qs, m, (1, m, n), residual, factor, norm, out_norm, order)
                    for cfg in cfgs:
                        got = candidate(layer, q, qs, m, (1, m, n), residual, factor, norm, out_norm, order, cfg)
                        assert_same(self, ref, got, f"N={n} K={k} M={m} epilogue {e} order {order} {cfg}")

    def test_b01_slices_identical(self):
        for (n, k), m in itertools.product([(1024, 1024), (1024, 4096), (1001, 1088)], (1, 38, 130, 368)):
            codes, scale, b = weights(n, k, m + k, bias=True)
            pm1 = Int8MmaLayer.prepare(codes, scale, b, k, "cuda")
            b01 = Int8MmaB01Layer.prepare(codes, scale, b, k, "cuda")
            x = torch.randn(1, m, k, device="cuda")
            x[..., 3] = 1e3
            q, qs = m1_quantize(x)
            for cfg in tuned_configs() + EXTRA:
                cfg = kernels_exact.ExactDeferredConfig.of(cfg)
                s0 = torch.empty(cfg.cluster, 2, m, n, device="cuda", dtype=torch.int32)
                s1 = torch.empty_like(s0)
                hybrid_exact.dot_slices(pm1, q, qs, k, m, s0, cfg)
                hybrid_exact.dot_slices(b01, q, qs, k, m, s1, cfg)
                self.assertTrue(torch.equal(s0, s1), f"N={n} K={k} M={m} {cfg}")

    def test_cuda_graph_replay(self):
        m, n, k = 126, 1024, 4096
        codes, scale, b = weights(n, k, 5, bias=True)
        layer = Int8MmaLayer.prepare(codes, scale, b, k, "cuda")
        norm = norm_module(n, 5)
        x = torch.randn(1, m, k, device="cuda")
        residual = torch.randn(1, m, n, device="cuda")
        cfg = hybrid_exact.choose_exact_deferred(m, n, k)
        self.assertIsNotNone(cfg)
        p = provider()

        def step():
            q, qs = m1_quantize(x)
            slices = torch.empty(cfg.cluster, 2, m, n, device="cuda", dtype=torch.int32)
            hybrid_exact.dot_slices(layer, q, qs, k, m, slices, cfg)
            y = row_exact.finish_residual_exact(p, row_exact.ExactDeferredProjection(
                slices, layer.scale, layer.bias, cfg.cluster, (1, m, n), qs, 0), residual, 1.0, norm)
            q2, qs2 = p.prequantized.popitem()[1]
            return y, q2, qs2

        step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = step()
        for trial in range(3):
            x.copy_(torch.randn_like(x) * (1 + trial))
            residual.copy_(torch.randn_like(residual))
            graph.replay()
            torch.cuda.synchronize()
            eager = step()
            for a, e in zip(out, eager):
                self.assertTrue(torch.equal(a, e), f"replay {trial}")

    def test_unpatched_m1_epilogue_refuses_slices(self):
        m, n, k = 38, 1024, 1024
        codes, scale, _ = weights(n, k, 6)
        layer = Int8MmaLayer.prepare(codes, scale, None, k, "cuda")
        q, qs = m1_quantize(torch.randn(1, m, k, device="cuda"))
        cfg = kernels_exact.ExactDeferredConfig.of(EXTRA[0])
        slices = torch.empty(cfg.cluster, 2, m, n, device="cuda", dtype=torch.int32)
        hybrid_exact.dot_slices(layer, q, qs, k, m, slices, cfg)
        value = row_exact.ExactDeferredProjection(slices, layer.scale, None, cfg.cluster, (1, m, n), qs, 0)
        with self.assertRaises(RuntimeError):
            row_exact._ORIGINAL(provider(), value, torch.zeros(1, m, n, device="cuda"), 1.0)
        from inference import row_fusion  # the routed name finishes it exactly
        y = row_fusion.finish_residual(provider(), value, torch.zeros(1, m, n, device="cuda"), 1.0)
        ref = reference(layer, q, qs, m, (1, m, n), torch.zeros(1, m, n, device="cuda"), 1.0, None, None, 0)
        self.assertTrue(torch.equal(y, ref[0]))


if __name__ == "__main__":
    unittest.main()
