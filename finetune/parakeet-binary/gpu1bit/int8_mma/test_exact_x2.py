"""Unit contracts of B-hybrid-x2's exact deferred path (plans/x2-default.md X1, X2, R8, Z5), x2 port of test_exact.

Reference ("the same-stack non-deferred x2 path"): the layer's in-kernel 2-part b1_dot (configs.choose
tiles) finished by M1's own _row_epilogue with SPLIT = 1. Candidate: kernels_exact_x2's XCHG 5 slices
finished by row_exact_x2._row_epilogue_exact_parts. Contracts:
  * bitwise equality of y and the fused next-layer q / qs for every row-epilogue flag combination the
    pipeline can use (bias, LayerNorm + quantization, out-norm, factor 0.5 / 1), both contraction
    orders, 4 and 8 warps (M <= 192 / > 192), M tails, N x K in 1024 x 1024, 1024 x 4096, 1001 x 1088,
    every tuned XCHG 5 config and test_exact.EXTRA (C = 2..8, WK 2, 2-warp M tiles);
  * Codex's c1 counterexample and +A / d / -A cancellations across every slice boundary at x2;
  * P = 3 through the x2 code equals B's x3 exact path (row_exact) bitwise (the generalisation is a no-op);
  * R8: the deferred value of both routes, and the provider's non-deferred calls, equal M1's
    int8_matmul at components = 2, SPLIT = 1 elementwise in one of its two contraction forms;
  * per-projection parity with int8_mma:x2 (layer_parts project, the provider-x2 arm's op) on identical
    projection inputs, through HybridProjectionX2: non-deferred bitwise, deferred value bitwise;
  * deferred equals non-deferred bitwise on deferred_a3's synthetic case set at components 2 (Z5);
  * the third part is never consumed: garbage codes / NaN scale in part 2 of a cached activation and
    garbage in the packed lane 2 of the slices leave every output bitwise unchanged (and a control
    poisoning part 1 does change it);
  * CUDA-graph replay with changed input equals eager; refusals (components != 2, x3 layers, tile
    config, M1's unpatched finish_residual on x2 slices).

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.gpulock gpu1bit.int8_mma.test_exact_x2
"""
from __future__ import annotations

import itertools
import types
import unittest

import torch

from gpu1bit.int8_mma import configs, hybrid_exact, hybrid_x2, kernels_exact, row_exact, row_exact_x2
from gpu1bit.int8_mma.kernels import DotConfig
from gpu1bit.int8_mma.layer import Int8MmaLayer, m1_quantize
from gpu1bit.int8_mma.test_exact import EXTRA, norm_module, tuned_configs, weights
from gpu1bit.shared import api, env

_LOCK = None
EPILOGUES = [(False, False, False, 1.0), (True, True, False, 0.5), (True, True, True, 0.5), (False, False, True, 1.0),
             (False, True, False, 0.5), (False, True, True, 1.0), (True, False, False, 1.0), (False, True, True, 0.5),
             (True, False, True, 0.5), (True, True, False, 1.0), (False, False, False, 0.5), (True, True, True, 1.0)]
M1_CONFIGS = [(32, 64, 64, 1, 4, 3), (32, 64, 128, 1, 4, 2), (16, 32, 128, 1, 4, 2)]


def setUpModule():
    global _LOCK
    hybrid_x2.parts()
    row_exact_x2.install()
    if not getattr(env.gpu_lock, "_held", False):
        _LOCK = env.gpu_lock("B:x2 exact deferred tests")
        _LOCK.__enter__()


def tearDownModule():
    if _LOCK is not None:
        _LOCK.__exit__(None, None, None)


def x2_layer(codes, scale, bias, k):
    return hybrid_x2.x2_layer_class().prepare(codes, scale, bias, k, "cuda")


def provider():
    return types.SimpleNamespace(row_warps="adaptive", prequantized={}, null=torch.empty(0, device="cuda", dtype=torch.uint8))


def configs_for(k):
    kb = -(-k // 128)
    return [c for c in tuned_configs() + EXTRA if c.cluster <= kb]


def reference(layer, q, qs, m, shape, residual, factor, norm, out_norm, order):
    """In-kernel 2-part b1_dot + M1's _row_epilogue with SPLIT = 1."""
    from inference.row_fusion import DeferredProjection
    n, k = layer.out_features, layer.in_features
    value = torch.empty(shape, device="cuda")
    layer.dot(q, qs, k, m, value.view(m, n), config=configs.choose(m, n, k), order=order, parts=q.shape[0])
    p = provider()
    y = row_exact._ORIGINAL(p, DeferredProjection(value, layer.scale, layer.bias, 1, shape), residual, factor, norm, out_norm)
    return y, p.prequantized


def candidate(layer, q, qs, m, shape, residual, factor, norm, out_norm, order, config, poison_lane2=False):
    n, k = layer.out_features, layer.in_features
    cfg = kernels_exact.ExactDeferredConfig.of(config)
    slices = torch.empty(cfg.cluster, 2, m, n, device="cuda", dtype=torch.int32)
    hybrid_x2.dot_slices_x2(layer, q, qs, k, m, slices, cfg, order)
    if poison_lane2:  # packed bits 42..63 (lane 2) live in bits 10..31 of the high word
        g = torch.Generator(device="cuda").manual_seed(7)
        junk = torch.randint(-2**31, 2**31 - 1, slices[:, 1].shape, device="cuda", dtype=torch.int64, generator=g)
        slices[:, 1].bitwise_xor_((junk.to(torch.int32) & ~0x3FF))
    p = provider()
    value = row_exact_x2.ExactDeferredProjectionX2(slices, layer.scale, layer.bias, cfg.cluster, shape, qs, order, q.shape[0])
    y = row_exact_x2.finish_residual_exact_x2(p, value, residual, factor, norm, out_norm)
    return y, p.prequantized


def assert_same(test, ref, got, what):
    y0, pre0 = ref
    y1, pre1 = got
    test.assertTrue(torch.equal(y0, y1), f"{what}: y differs, max |d| {(y0 - y1).abs().max().item():.3e}")
    test.assertEqual(len(pre0), len(pre1), what)
    for (q0, s0), (q1, s1) in zip(pre0.values(), pre1.values()):
        test.assertTrue(torch.equal(q0, q1), f"{what}: fused q differs")
        test.assertTrue(torch.equal(s0, s1), f"{what}: fused qs differs")


def assert_either(test, ref, y0, y1, what):
    ok = (ref == y0) | (ref == y1)
    test.assertTrue(bool(ok.all()), f"{what}: {int((~ok).sum())} of {ok.numel()} elements match neither form")


def one_layer_provider(layer, kind="linear"):
    """HybridProjectionX2 over a one-layer table (as H's deferred_a3 builds it) and the module handle."""
    from gpu1bit.shared.provider import Table, _handle
    table = Table()
    table.add("layer", layer, kind, layer.out_features, layer.in_features)
    return hybrid_x2.HybridProjectionX2(table, {}), types.SimpleNamespace(packed_t=_handle(0, layer.out_features))


def call(prov, mod, x, conv=False, defer=False, activation="", norm=None, input_activation=""):
    """The provider call R-hybrid's stack makes; a DeferredProjection is finished by the routed
    inference.row_fusion.finish_residual with a zero residual and factor 1 (= the projection value)."""
    from inference import row_fusion
    prov.defer_epilogue = defer
    try:
        v = prov(x, mod, conv, 2, activation, norm, None, input_activation)
    finally:
        prov.defer_epilogue = False
    if isinstance(v, row_fusion.DeferredProjection):
        y = row_fusion.finish_residual(prov, v, torch.zeros(v.shape, device="cuda"), 1.0)
        return (y.transpose(1, 2) if conv else y.reshape(*x.shape[:-1], y.shape[-1])), type(v).__name__, v.split
    return v, "plain", None


class ExactDeferredX2(unittest.TestCase):
    def test_codex_counterexample_x2(self):
        m, n, k = 126, 1024, 4096
        codes, scale, _ = weights(n, k, 0, plus_one=True, unit_scale=True)
        x = torch.zeros(1, m, k, device="cuda")
        x[..., 0], x[..., 128], x[..., 1024] = 16256.0, 2.0 ** -12, -16256.0
        q, qs = hybrid_x2.quantize_x2(x)
        zeros = torch.zeros(1, m, n, device="cuda")
        layer = x2_layer(codes, scale, None, k)
        ref = reference(layer, q, qs, m, (1, m, n), zeros, 1.0, None, None, 0)
        self.assertTrue(torch.equal(ref[0], torch.full_like(ref[0], 2.0 ** -12)))
        for cfg in configs_for(k):
            assert_same(self, ref, candidate(layer, q, qs, m, (1, m, n), zeros, 1.0, None, None, 0, cfg), f"{cfg}")
        self.assertTrue(torch.equal(hybrid_x2.deferred_project_x2(layer, x), ref[0]))

    def test_split_boundary_cancellation_x2(self):
        for k in (1024, 4096):
            n = 1024
            codes, scale, _ = weights(n, k, 1, plus_one=True, unit_scale=True)
            layer = x2_layer(codes, scale, None, k)
            kb = -(-k // 128)
            for cfg in configs_for(k):
                c = cfg.cluster
                cols = sorted({(kb * r // c) * 128 for r in range(1, c)} | {128})
                rows = []
                for col, (big, small) in itertools.product(cols, ((16256.0, 2.0 ** -12), (1016.0, 2.0 ** -20), (31.75, 1.0e-3))):
                    rows.append({0: big, 1: small, col: -big})
                    rows.append({0: big, col: -big, col + 1: small})
                    rows.append({col - 1: big, col: -big, 5: small})
                m = len(rows)
                x = torch.zeros(1, m, k, device="cuda")
                for i, entries in enumerate(rows):
                    for j, v in entries.items():
                        x[0, i, j] = v
                q, qs = hybrid_x2.quantize_x2(x)
                zeros = torch.zeros(1, m, n, device="cuda")
                ref = reference(layer, q, qs, m, (1, m, n), zeros, 1.0, None, None, 0)
                self.assertTrue(bool((ref[0] != 0).all()), "exact SPLIT == 1 at x2 keeps every small term")
                assert_same(self, ref, candidate(layer, q, qs, m, (1, m, n), zeros, 1.0, None, None, 0, cfg), f"K={k} {cfg}")

    def test_bitwise_every_flag_order_tail_config(self):
        shapes = [(1024, 1024), (1024, 4096), (1001, 1088)]
        for (n, k), m in itertools.product(shapes, (1, 17, 38, 126, 130, 192, 193, 256, 300, 368)):
            for e, (bias, with_norm, with_out, factor) in enumerate(EPILOGUES):
                if m not in (17, 126, 193, 300) and e not in (0, 2, 7, 11):
                    continue
                seed = n + k + m + e
                codes, scale, b = weights(n, k, seed, bias=bias)
                layer = x2_layer(codes, scale, b, k)
                g = torch.Generator(device="cuda").manual_seed(seed)
                x = torch.randn(1, m, k, device="cuda", generator=g)
                x[..., 7] *= 300.0
                x[..., k // 2] *= -50.0
                residual = torch.randn(1, m, n, device="cuda", generator=g)
                norm = norm_module(n, seed) if with_norm else None
                out_norm = norm_module(n, seed + 1) if with_out else None
                q, qs = hybrid_x2.quantize_x2(x)
                chosen = hybrid_exact.choose_exact_deferred(m, n, k)
                pool = configs_for(k)
                cfgs = ([chosen] if chosen is not None else []) + [pool[(m + e) % len(pool)], pool[(m + e + 3) % len(pool)]]
                for order in ((0, 1) if e in (2, 11) else (0,)):
                    ref = reference(layer, q, qs, m, (1, m, n), residual, factor, norm, out_norm, order)
                    for cfg in cfgs:
                        got = candidate(layer, q, qs, m, (1, m, n), residual, factor, norm, out_norm, order, cfg)
                        assert_same(self, ref, got, f"N={n} K={k} M={m} epilogue {e} order {order} {cfg}")

    def test_parts3_equals_x3_exact_path(self):
        for (n, k), m in itertools.product([(1024, 1024), (1024, 4096)], (17, 126, 300)):
            codes, scale, b = weights(n, k, 5 * m + k, bias=True)
            layer = Int8MmaLayer.prepare(codes, scale, b, k, "cuda")
            x = torch.randn(1, m, k, device="cuda")
            residual = torch.randn(1, m, n, device="cuda")
            norm = norm_module(n, m)
            q, qs = m1_quantize(x)
            for cfg in configs_for(k)[:5]:
                c = kernels_exact.ExactDeferredConfig.of(cfg)
                s0 = torch.empty(c.cluster, 2, m, n, device="cuda", dtype=torch.int32)
                s1 = torch.empty_like(s0)
                hybrid_exact.dot_slices(layer, q, qs, k, m, s0, c)
                hybrid_x2.dot_slices_x2(layer, q, qs, k, m, s1, c)
                self.assertTrue(torch.equal(s0, s1), f"P=3 slices N={n} K={k} M={m} {cfg}")
                p0, p1 = provider(), provider()
                y0 = row_exact.finish_residual_exact(p0, row_exact.ExactDeferredProjection(s0, layer.scale, layer.bias, c.cluster,
                                                                                          (1, m, n), qs, 0), residual, 0.5, norm)
                y1 = row_exact_x2.finish_residual_exact_x2(p1, row_exact_x2.ExactDeferredProjectionX2(
                    s1, layer.scale, layer.bias, c.cluster, (1, m, n), qs, 0, 3), residual, 0.5, norm)
                assert_same(self, (y0, p0.prequantized), (y1, p1.prequantized), f"P=3 epilogue N={n} K={k} M={m} {cfg}")

    def test_r8_oracle_both_routes(self):
        """Deferred value (XCHG 5 and in-kernel routes) vs M1's int8_matmul(components=2, SPLIT=1)."""
        from gpu1bit.shared.m1 import derived_packed_linear
        from inference.fused_kernels import int8_matmul
        for (n, k), m in itertools.product([(1024, 1024), (1024, 4096)], (1, 17, 38, 126, 130, 368)):
            codes, scale, b = weights(n, k, 11 * m + k, bias=(m % 2 == 0))
            mod = derived_packed_linear(codes, scale, b, k)
            layer = x2_layer(codes, scale, b, k)
            x = torch.randn(1, m, k, device="cuda")
            x[..., 3] *= 200.0
            q, qs = hybrid_x2.quantize_x2(x)
            zeros = torch.zeros(1, m, n, device="cuda")
            routes = []
            for order in (0, 1):
                routes.append([candidate(layer, q, qs, m, (1, m, n), zeros, 1.0, None, None, order, cfg)[0]
                               for cfg in configs_for(k)[:3]])
                routes.append([reference(layer, q, qs, m, (1, m, n), zeros, 1.0, None, None, order)[0]])
            for cfg in M1_CONFIGS:
                ref = int8_matmul(x, mod, False, 2, "", None, cfg, "")
                for i in range(len(routes[0])):
                    assert_either(self, ref, routes[0][i], routes[2][i], f"XCHG 5 N={n} K={k} M={m} M1 {cfg}")
                assert_either(self, ref, routes[1][0], routes[3][0], f"in-kernel N={n} K={k} M={m} M1 {cfg}")

    def test_provider_parity_and_oracle(self):
        """HybridProjectionX2 per projection: equal to int8_mma:x2 project (same layer, identical inputs) and
        to M1's int8_matmul at components = 2 in one contraction form; deferred value = project, both routes."""
        from gpu1bit.shared.m1 import derived_packed_linear
        from inference.fused_kernels import int8_matmul
        n, k = 1024, 1024
        ln = norm_module(k, 3)
        for m, (act, use_norm, in_act, conv, bias) in itertools.product(
                (17, 38, 126, 300), [("", True, "", False, False), ("silu", False, "", False, True), ("", False, "silu", True, True),
                                     ("silu", True, "", False, False), ("", False, "", True, False), ("", False, "", False, True)]):
            codes, scale, b = weights(n, k, 3 * m + int(conv), bias=bias)
            layer = x2_layer(codes, scale, b, k)
            prov, handle = one_layer_provider(layer, "conv" if conv else "linear")
            x = torch.randn(1, m, k, device="cuda")
            x[..., 9] *= 100.0
            if conv:
                x = x.transpose(1, 2).contiguous()
            kw = dict(conv=conv, activation=act, norm=ln if use_norm else None, input_activation=in_act)
            want = layer.project(x, components=2, **kw)
            got, route, _ = call(prov, handle, x, **kw)
            self.assertEqual(route, "plain")
            self.assertTrue(torch.equal(got, want), f"non-deferred M={m} {kw}")
            mod = derived_packed_linear(codes, scale, b, k, conv=conv)
            xx = x.transpose(1, 2) if conv else x
            qq, ss = hybrid_x2.quantize_x2(xx, kw["norm"], in_act)
            out = torch.empty(qq.shape[1], n, device="cuda")
            layer.dot(qq, ss, k, qq.shape[1], out, silu=act == "silu", order=1, parts=2)  # the other form
            other = out.view(1, -1, n).transpose(1, 2) if conv else out.view(1, -1, n)
            for cfg in M1_CONFIGS[:2]:
                ref = int8_matmul(x, mod, conv, 2, act, kw["norm"], cfg, in_act)
                assert_either(self, ref, want, other, f"R8 provider M={m} {kw}")
            if act == "" and not use_norm and not in_act:
                for split in (True, False):
                    prov.split_deferred = split
                    got_d, route_d, s = call(prov, handle, x, conv=conv, defer=True)
                    chosen = hybrid_exact.choose_exact_deferred(m, n, k)
                    self.assertEqual(route_d, "ExactDeferredProjectionX2" if split and chosen is not None else "DeferredProjection")
                    self.assertTrue(torch.equal(got_d, want), f"deferred split={split} M={m} conv={conv}")
                prov.split_deferred = True

    def test_refusals(self):
        n, k, m = 1024, 1024, 17
        codes, scale, _ = weights(n, k, 9)
        layer = x2_layer(codes, scale, None, k)
        prov, handle = one_layer_provider(layer)
        x = torch.randn(1, m, k, device="cuda")
        for comp in (1, 3):
            with self.assertRaises(ValueError):
                prov(x, handle, False, comp)
        with self.assertRaises(ValueError):
            prov(x, handle, False, 2, "", None, (32, 64, 64, 1, 4, 3), "")
        from gpu1bit.shared.provider import Table
        t = Table()
        t.add("layer", Int8MmaLayer.prepare(codes, scale, None, k, "cuda"), "linear", n, k)
        with self.assertRaises(ValueError):
            hybrid_x2.HybridProjectionX2(t, {})
        q, qs = hybrid_x2.quantize_x2(x)
        cfg = kernels_exact.ExactDeferredConfig.of(EXTRA[0])
        slices = torch.empty(cfg.cluster, 2, m, n, device="cuda", dtype=torch.int32)
        hybrid_x2.dot_slices_x2(layer, q, qs, k, m, slices, cfg)
        value = row_exact_x2.ExactDeferredProjectionX2(slices, layer.scale, None, cfg.cluster, (1, m, n), qs, 0, 2)
        zeros = torch.zeros(1, m, n, device="cuda")
        with self.assertRaises(RuntimeError):
            row_exact._ORIGINAL(provider(), value, zeros, 1.0)          # M1's original: .partial raises
        with self.assertRaises(RuntimeError):
            value.partial
        with self.assertRaises(TypeError):                              # B's x3 exact finish refuses x2 slices
            row_exact.finish_residual_exact(provider(), value, zeros, 1.0)
        self.assertFalse(isinstance(value, row_exact.ExactDeferredProjection), "x3 router must not take x2 slices")
        from inference import row_fusion
        self.assertTrue(row_exact_x2.installed())
        y = row_fusion.finish_residual(provider(), value, zeros, 1.0)
        ref = reference(layer, q, qs, m, (1, m, n), zeros, 1.0, None, None, 0)
        self.assertTrue(torch.equal(y, ref[0]))

    def test_third_part_is_never_consumed(self):
        """Poison: part 2 of a cached activation (random codes, NaN scale) and lane 2 of the packed slices."""
        from inference import row_fusion
        n, k = 1024, 4096
        for m in (38, 126, 300):
            codes, scale, b = weights(n, k, m, bias=True)
            layer = x2_layer(codes, scale, b, k)
            prov, handle = one_layer_provider(layer)
            norm_ = norm_module(k, m)
            g = torch.Generator(device="cuda").manual_seed(m)
            x = torch.randn(1, m, k, device="cuda", generator=g)
            residual = torch.randn(1, m, k, device="cuda", generator=g)
            src = torch.randn(1, m, k, device="cuda", generator=g)

            def consume(poison_part, split, seed):
                """M1's row epilogue caches a 3-part activation; the x2 provider consumes it (deferred call)."""
                from inference.row_fusion import DeferredProjection
                ones = torch.ones(k, device="cuda")
                prov.prequantized.clear()
                y = row_exact._ORIGINAL(prov, DeferredProjection(src.clone(), ones, None, 1, (1, m, k)), residual, 1.0, norm_)
                q3, qs3 = prov.prequantized[(y.data_ptr(), m, k, id(norm_))]
                if poison_part is not None:
                    gg = torch.Generator(device="cuda").manual_seed(seed)
                    q3[poison_part].copy_(torch.randint(-128, 128, q3[poison_part].shape, device="cuda", dtype=torch.int8,
                                                        generator=gg))
                    qs3[poison_part].fill_(float("nan"))
                prov.split_deferred, prov.defer_epilogue = split, True
                try:
                    v = prov(y, handle, False, 2, "", norm_, None, "")
                finally:
                    prov.split_deferred, prov.defer_epilogue = True, False
                self.assertFalse(prov.prequantized, "the cached activation was consumed")
                return type(v).__name__, row_fusion.finish_residual(prov, v, torch.zeros(v.shape, device="cuda"), 1.0)

            def cached_run(poison_part):
                return [consume(poison_part, split, 90 + i) for i, split in enumerate((True, False))]

            clean, poisoned, control = cached_run(None), cached_run(2), cached_run(1)
            for (r0, y0), (r1, y1), (_, y2) in zip(clean, poisoned, control):
                self.assertEqual(r0, r1)
                self.assertTrue(torch.equal(y0, y1), f"M={m} {r0}: part 2 was consumed")
                self.assertFalse(torch.equal(y0, y2), f"M={m} {r0}: control (part 1 poisoned) did not change the output")
            self.assertEqual({r for r, _ in clean}, {"ExactDeferredProjectionX2", "DeferredProjection"} if
                             hybrid_exact.choose_exact_deferred(m, n, k) is not None else {"DeferredProjection"})
            # lane 2 of the packed slices
            q, qs = hybrid_x2.quantize_x2(x)
            zeros = torch.zeros(1, m, n, device="cuda")
            for cfg in configs_for(k)[:4]:
                a = candidate(layer, q, qs, m, (1, m, n), residual[..., :n].contiguous(), 0.5, norm_module(n, 1), None, 0, cfg)
                c = candidate(layer, q, qs, m, (1, m, n), residual[..., :n].contiguous(), 0.5, norm_module(n, 1), None, 0, cfg,
                              poison_lane2=True)
                assert_same(self, a, c, f"lane 2 poison M={m} {cfg}")

    def test_deferred_equals_non_deferred_on_deferred_a3_cases(self):
        """Z5 at components 2: deferred_a3's synthetic case set, deferred == non-deferred bitwise."""
        from gpu1bit.shared import deferred_a3 as D
        cases = D.synthetic_cases()
        self.assertGreater(len(cases), 100)
        self.assertIn("codex_c1_counterexample", {c["name"] for c in cases})
        routes = {}
        for i, case in enumerate(cases):
            w, x = D.case_data(case, 9000 + i)
            layer = x2_layer(w["codes"], w["scale"], w["bias"], w["K"])
            prov, handle = one_layer_provider(layer, case["kind"])
            conv = case["kind"] == "conv"
            plain, r0, _ = call(prov, handle, x, conv=conv, defer=False)
            deferred, r1, _ = call(prov, handle, x, conv=conv, defer=True)
            self.assertEqual(r0, "plain")
            self.assertIn(r1, ("ExactDeferredProjectionX2", "DeferredProjection"))
            self.assertTrue(torch.equal(plain, deferred), f"{case['name']}: deferred != non-deferred at x2")
            routes[r1] = routes.get(r1, 0) + 1
        self.assertGreater(routes.get("ExactDeferredProjectionX2", 0), 0, routes)
        self.assertGreater(routes.get("DeferredProjection", 0), 0, routes)
        print(f"deferred_a3 synthetic cases at x2: {len(cases)}, routes {routes}", flush=True)

    def test_cuda_graph_replay_changed_input(self):
        m, n, k = 126, 1024, 4096
        codes, scale, b = weights(n, k, 5, bias=True)
        layer = x2_layer(codes, scale, b, k)
        norm = norm_module(n, 5)
        x = torch.randn(1, m, k, device="cuda")
        residual = torch.randn(1, m, n, device="cuda")
        cfg = hybrid_exact.choose_exact_deferred(m, n, k)
        self.assertIsNotNone(cfg)
        p = provider()

        def step():
            q, qs = hybrid_x2.quantize_x2(x)
            slices = torch.empty(cfg.cluster, 2, m, n, device="cuda", dtype=torch.int32)
            hybrid_x2.dot_slices_x2(layer, q, qs, k, m, slices, cfg)
            y = row_exact_x2.finish_residual_exact_x2(p, row_exact_x2.ExactDeferredProjectionX2(
                slices, layer.scale, layer.bias, cfg.cluster, (1, m, n), qs, 0, 2), residual, 1.0, norm)
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


if __name__ == "__main__":
    unittest.main()
