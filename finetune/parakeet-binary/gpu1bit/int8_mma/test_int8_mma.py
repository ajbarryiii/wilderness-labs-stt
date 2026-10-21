"""Agent B's own correctness checks (besides H's contract tests): run with
    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.gpulock gpu1bit.int8_mma.test_int8_mma
from finetune/parakeet-binary: one GPU lock held until the process exits (plans/gpu-kernel.md A9)."""
from __future__ import annotations

import itertools
import unittest

import torch
import triton

from gpu1bit.int8_mma import configs, kernels
from gpu1bit.int8_mma.kernels import DotConfig
from gpu1bit.int8_mma.layer import Int8MmaLayer, quantize
from gpu1bit.shared import api, env

_LOCK = None


def setUpModule():
    """Prefer `gpu1bit/shared/python -m gpu1bit.int8_mma.gpulock` (lock held until process exit, A9);
    plain unittest takes the lock here for the module's lifetime."""
    global _LOCK
    if not getattr(env.gpu_lock, "_held", False):
        _LOCK = env.gpu_lock("B:own correctness tests")
        _LOCK.__enter__()
    torch.backends.cuda.matmul.allow_tf32 = False


def tearDownModule():
    if _LOCK is not None:
        _LOCK.__exit__(None, None, None)


def make_layer(n, k, bias=False, seed=0, special_rows=True):
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n, k), generator=g, dtype=torch.int8) * 2 - 1
    if special_rows and n >= 3:
        signs[0] = 1
        signs[1] = -1
    codes = api.pack_signs(signs)
    scale = torch.rand(n, generator=g) * 0.05 + 1e-3
    b = torch.randn(n, generator=g) * 0.1 if bias else None
    return codes, scale, b, Int8MmaLayer.prepare(codes, scale, b, k, "cuda")


def m1_quantize(x3):
    """M1's _quantize on [B, T, K] (any strides) -> q [3, M, K], qs [3, M]."""
    from inference.fused_kernels import _quantize
    b, t, k = x3.shape
    m = b * t
    q = torch.empty(3, m, k, device="cuda", dtype=torch.int8)
    qs = torch.empty(3, m, device="cuda")
    _quantize[(m,)](x3, q, qs, None, None, m, k, t, *x3.stride(), triton.next_power_of_2(k), 3, False, 0.,
                    INPUT_SILU=False, enable_fp_fusion=False)
    return q, qs


def m1_dot_split1(q, qs, signs, scale, bias, silu=False, bk=64):
    """M1's expanded INT8x3 kernel (_int8_dot, DENSE column-major INT8 weights), SPLIT = 1."""
    from inference.fused_kernels import _int8_dot
    _, m, k = q.shape
    n = signs.shape[0]
    w = signs.to("cuda").contiguous()  # [N, K] row-major == column-major [K, N]
    y = torch.empty(m, n, device="cuda")
    bm, bn = 32, 64
    _int8_dot[(triton.cdiv(m, bm), triton.cdiv(n, bn), 1)](
        q, qs, w, scale, bias, y, y, m, n, k, bias is not None, bm, bn, bk, 1, 3,
        ACTIVATION="silu" if silu else "", DENSE=True, COLUMN=True, num_warps=4, num_stages=2)
    return y


class LayoutTests(unittest.TestCase):
    def test_relayout_round_trip(self):
        for n, k in [(8, 128), (1024, 1024), (24, 1000), (3, 129), (17, 4096)]:
            bits = torch.randint(0, 2, (n, k), dtype=torch.uint8, device="cuda")
            words = kernels.relayout(bits)
            self.assertEqual(tuple(words.shape), (-(-n // 8), -(-k // 128), 32))
            self.assertTrue(torch.equal(kernels.unlayout(words, n, k), bits))

    def test_fragment_word_semantics(self):
        """One hand-checked bit: w[n = 8t + g, k = 128b + 32j + 16r + 4s + i] is bit 8i + 2j + r of lane 4g + s."""
        n, k = 16, 256
        for (t, g, b, j, r, s, i) in [(1, 3, 1, 2, 1, 3, 2), (0, 0, 0, 0, 0, 0, 0), (1, 7, 1, 3, 1, 3, 3)]:
            bits = torch.zeros(n, k, dtype=torch.uint8, device="cuda")
            bits[8 * t + g, 128 * b + 32 * j + 16 * r + 4 * s + i] = 1
            words = kernels.relayout(bits)
            expect = torch.zeros_like(words)
            expect[t, b, 4 * g + s] = 1 << (8 * i + 2 * j + r) if 8 * i + 2 * j + r < 31 else -(2**31)
            self.assertTrue(torch.equal(words, expect))

    def test_weight_budget(self):
        for n, k in [(1024, 1024), (4096, 1024), (1024, 4096), (2048, 1024), (1000, 1000), (37, 33), (1, 5)]:
            _, _, _, layer = make_layer(n, k)
            self.assertLessEqual(layer.weight_bytes(), api.weight_budget_bytes(n, k, False))
        _, _, _, layer = make_layer(1024, 1024)
        self.assertTrue(layer.fast)
        self.assertEqual(layer.weight_bytes(), 1024 * 128 + 4 * 1024)


class QuantizeTests(unittest.TestCase):
    @torch.inference_mode()
    def test_bitwise_equal_to_m1_quantize(self):
        for b, t, k in [(1, 38, 1024), (2, 17, 4096), (1, 5, 1000), (3, 7, 33)]:
            x = torch.randn(b, t, k, device="cuda") * torch.logspace(-3, 3, k, device="cuda")
            x[0, 0] = 0
            for view in [x, x.transpose(1, 2).contiguous().transpose(1, 2), x[:, :, :]]:
                q1, s1 = m1_quantize(view)
                q2, s2, ldq = quantize(view, (view.stride(0), view.stride(1), view.stride(2), t), b * t, k)
                self.assertTrue(torch.equal(s1, s2))
                self.assertTrue(torch.equal(q1, q2[:, :, :k]))
                self.assertFalse(q2[:, :, k:].any())


class DotTests(unittest.TestCase):
    CONFIGS = [DotConfig(1, 4, 3, 2, 2, 1), DotConfig(1, 4, 3, 2, 2, 8), DotConfig(1, 2, 1, 4, 3, 4),
               DotConfig(2, 2, 2, 2, 3, 2), DotConfig(1, 4, 4, 2, 3, 1), DotConfig(1, 8, 2, 1, 2, 3),
               DotConfig(2, 4, 1, 2, 2, 5), DotConfig(1, 4, 3, 2, 2, 8, 1), DotConfig(1, 2, 1, 4, 3, 3, 1),
               DotConfig(1, 4, 1, 1, 3, 1, 0, 2), DotConfig(1, 2, 3, 1, 2, 2, 1, 2), DotConfig(2, 2, 1, 2, 3, 4, 0, 2),
               DotConfig(1, 4, 2, 2, 2, 1, 0, 2), DotConfig(1, 2, 1, 2, 2, 3, 1, 4), DotConfig(1, 4, 3, 2, 2, 8, 3),
               DotConfig(1, 8, 1, 1, 3, 4, 3, 2), DotConfig(2, 2, 1, 3, 2, 3, 3),
               DotConfig(1, 4, 1, 2, 4, 1, 0, 2, 1), DotConfig(1, 8, 2, 3, 3, 1, 0, 1, 1), DotConfig(1, 4, 3, 2, 2, 1, 0, 1, 1)]
    FP32_SPLIT = [DotConfig(1, 4, 3, 2, 2, 8, 2), DotConfig(1, 8, 2, 1, 3, 3, 2), DotConfig(1, 4, 1, 2, 2, 4, 2, 2)]
    DEFERRED = [DotConfig(1, 4, 1, 2, 2, 4, 4), DotConfig(1, 8, 1, 1, 3, 8, 4), DotConfig(1, 4, 2, 2, 2, 3, 4, 2),
                DotConfig(1, 8, 1, 2, 4, 4, 4, 1, 1)]

    def assert_m1_elementwise(self, layer, q, qs, k, m, expect, msg, **kw):
        """Integer accumulators identical to M1's, and every element equal to M1's FP32 epilogue under
        one of its two contraction forms. M1's Triton compiles SLP-vectorize the epilogue into f32x2
        pairs whose halves contract differently (kernels.py), so no single form matches a whole tile."""
        n = expect.shape[1]
        got = {}
        for order in (0, 1):
            y = torch.full((m, n), float("nan"), device="cuda")
            layer.dot(q, qs, k, m, y, order=order, **kw)
            got[order] = y
        ok = (got[0] == expect) | (got[1] == expect)
        self.assertTrue(bool(ok.all()), f"{msg}: {(~ok).sum().item()} elements differ, max "
                        f"{torch.minimum((got[0] - expect).abs(), (got[1] - expect).abs()).max().item()}")
        return got

    @torch.inference_mode()
    def test_m1_expanded_split1_elementwise(self):
        for (n, k), m, bias, silu in itertools.product([(1024, 1024), (256, 4096), (2048, 512)], [1, 38, 126],
                                                       [False, True], [False, True]):
            codes, scale, b, layer = make_layer(n, k, bias, seed=n + k + m)
            x = torch.randn(1, m, k, device="cuda")
            q, qs = m1_quantize(x)
            signs = api.unpack_signs(codes, k)
            for bk in (64, 128):
                expect = m1_dot_split1(q, qs, signs, layer.scale, layer.bias, silu, bk=bk)
                for cfg in self.CONFIGS:
                    self.assert_m1_elementwise(layer, q, qs, k, m, expect,
                                               f"{n}x{k} m={m} bias={bias} silu={silu} bk={bk} {cfg.label()}",
                                               silu=silu, config=cfg)

    @torch.inference_mode()
    def test_fp32_split_exchange_is_deterministic_and_accurate(self):
        """xchg 2 (FP32 partial per cluster rank, M1's split arithmetic): run-to-run bitwise identical,
        close to the exact INT32 result, and within FP64-reference accuracy of it."""
        for (n, k), m, silu in itertools.product([(1024, 1024), (2048, 4096), (300, 1040)], [1, 38, 130], [False, True]):
            codes, scale, b, layer = make_layer(n, k, bias=n == 300, seed=n + m)
            x = torch.randn(1, m, k, device="cuda")
            q, qs = m1_quantize(x)
            exact = torch.empty(m, n, device="cuda")
            layer.dot(q, qs, k, m, exact, silu=silu, config=DotConfig(1, 4, 1, 2, 2, 1))
            for cfg in self.FP32_SPLIT:
                y1 = torch.full((m, n), float("nan"), device="cuda")
                y2 = torch.full((m, n), float("nan"), device="cuda")
                layer.dot(q, qs, k, m, y1, silu=silu, config=cfg)
                layer.dot(q, qs, k, m, y2, silu=silu, config=cfg)
                self.assertTrue(torch.equal(y1, y2), cfg.label())
                rel = ((y1 - exact).norm() / exact.norm()).item()
                self.assertLess(rel, 3e-7, f"{cfg.label()} {n}x{k} m={m}: {rel}")

    @torch.inference_mode()
    def test_deferred_slices_sum_to_the_projection(self):
        """xchg 4: S FP32 slices of M1's split partial; R-hybrid's _row_epilogue arithmetic on them
        ((0 + P_0) + P_1 + ...) * alpha matches the exact projection closely and deterministically."""
        for (n, k), m in itertools.product([(1024, 1024), (1024, 4096)], [1, 38, 130]):
            codes, scale, b, layer = make_layer(n, k, seed=n + k + m)
            x = torch.randn(1, m, k, device="cuda")
            q, qs = m1_quantize(x)
            exact = torch.empty(m, n, device="cuda")
            layer.dot(q, qs, k, m, exact, config=DotConfig(1, 4, 1, 2, 2, 1))
            for cfg in self.DEFERRED:
                parts = torch.full((cfg.cluster, m, n), float("nan"), device="cuda")
                layer.dot(q, qs, k, m, parts[0], config=cfg)
                again = torch.full_like(parts, float("nan"))
                layer.dot(q, qs, k, m, again[0], config=cfg)
                self.assertTrue(torch.equal(parts, again), cfg.label())
                total = torch.zeros(m, n, device="cuda")
                for p in parts:
                    total = total + p
                y = total * layer.scale
                rel = ((y - exact).norm() / exact.norm()).item()
                self.assertLess(rel, 3e-7, f"{cfg.label()} {n}x{k} m={m}: {rel}")

    @torch.inference_mode()
    def test_fp64_reference_shapes_strides_batches(self):
        cases = [(1024, 1024, (1, 38)), (4096, 1024, (2, 21)), (1024, 4096, (1, 126)), (2048, 1024, (3, 5)),
                 (1000, 1000, (2, 9)), (37, 33, (1, 3)), (1, 5, (4, 2)), (24, 129, (1, 1)), (3072, 1024, (1, 376))]
        for n, k, (bb, t) in cases:
            codes, scale, b, layer = make_layer(n, k, bias=(n % 2 == 0), seed=n * 7 + k)
            base = torch.randn(bb, t, k, device="cuda", dtype=torch.float64).float()
            inputs = {"contiguous": base, "sliced": torch.randn(bb, t, k + 7, device="cuda")[..., 3:3 + k],
                      "2d": base.reshape(bb * t, k), "4d": base.reshape(1, bb, t, k),
                      "transposed-storage": base.transpose(0, 1).contiguous().transpose(0, 1)}
            for label, x in inputs.items():
                y = layer.linear(x)
                self.assertEqual(tuple(y.shape), (*x.shape[:-1], n))
                self.assertTrue(y.is_contiguous())
                ref = api.reference_linear(x, codes, scale, b, k)
                e = api.errors("linear", x, y, ref, scale)
                self.assertTrue(e["finite"])
                self.assertLess(e["relative_frobenius"], 2e-6, f"{n}x{k} {label} {e}")
            xc = base.transpose(1, 2)  # NeMo's conv input: [B, K, T] view of [B, T, K]
            for label, x in {"conv-view": xc, "conv-contiguous": xc.contiguous()}.items():
                y = layer.conv(x)
                self.assertEqual(tuple(y.shape), (bb, n, t))
                ref = api.reference_conv(x, codes, scale, b, k)
                e = api.errors("conv", x, y, ref, scale)
                self.assertLess(e["relative_frobenius"], 2e-6, f"{n}x{k} {label} {e}")
                self.assertTrue(torch.equal(y, layer.linear(xc.transpose(1, 2)).transpose(1, 2)))

    @torch.inference_mode()
    def test_integer_semantics_and_silence(self):
        n, k = 96, 300
        codes, scale, b, layer = make_layer(n, k, bias=True)
        scale_int = torch.ones(n)
        layer_int = Int8MmaLayer.prepare(codes, scale_int, torch.arange(n, dtype=torch.float32), k, "cuda")
        x = torch.randint(-20, 21, (5, k), device="cuda").float()
        ref = api.reference_linear(x, codes, scale_int, torch.arange(n, dtype=torch.float32), k)
        self.assertLess((layer_int.linear(x).double() - ref).abs().max().item(), 0.2)
        zero = torch.zeros(4, k, device="cuda")
        self.assertTrue(torch.equal(layer.linear(zero), layer.bias.expand(4, n)))

    @torch.inference_mode()
    def test_graph_capture_replay_and_streams(self):
        for n, k, m in [(1024, 1024, 38), (2048, 1024, 126), (37, 33, 3)]:
            codes, scale, b, layer = make_layer(n, k, seed=3)
            x = torch.randn(1, m, k, device="cuda")
            xc = torch.randn(1, k, m, device="cuda")
            layer.linear(x), layer.conv(xc)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y = layer.linear(x)
                yc = layer.conv(xc)
            for _ in range(2):
                x.copy_(torch.randn_like(x))
                xc.copy_(torch.randn_like(xc))
                pressure = torch.empty(32 * 1024 * 1024, device="cuda")
                pressure.fill_(7)
                del pressure
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(y, layer.linear(x)))
                self.assertTrue(torch.equal(yc, layer.conv(xc)))
            del graph
            side = torch.cuda.Stream()
            expect = layer.linear(x)
            torch.cuda.synchronize()
            with torch.cuda.stream(side):
                got = layer.linear(x)
            side.synchronize()
            self.assertTrue(torch.equal(got, expect))

    @torch.inference_mode()
    def test_project_matches_r_expanded_call_signatures(self):
        """provider.py's fused op vs R-expanded's ExpandedProjection on every call form the transforms make."""
        from inference.encoder4 import ExpandedProjection
        from gpu1bit.shared.m1 import derived_packed_linear
        reference = ExpandedProjection("triton_col_tuned")
        norm = torch.nn.LayerNorm(1024).cuda()
        norm.weight.uniform_(0.5, 1.5)
        norm.bias.uniform_(-0.2, 0.2)
        calls = [(3072, 1024, False, "", norm, ""), (1024, 1024, False, "", None, ""),
                 (4096, 1024, False, "silu", norm, ""), (1024, 4096, False, "", None, ""),
                 (2048, 1024, True, "", norm, ""), (1024, 1024, True, "", None, "silu")]
        alive = []  # ExpandedProjection caches INT8 weights by packed_t.data_ptr(): never free a module
        for (n, k, conv, act, nrm, inact), (b, t) in itertools.product(calls, [(1, 38), (2, 63)]):
            codes, scale, bias, layer = make_layer(n, k, seed=n + t)
            mod = derived_packed_linear(codes, scale, bias, k, conv=conv)
            alive.append(mod)
            base = torch.randn(b, t, k, device="cuda")
            x = base.transpose(1, 2) if conv else base  # NeMo's conv input view
            if conv and inact:
                x = torch.randn(b, k, t, device="cuda")  # pointwise_conv2 gets contiguous [B, C, T]
            got = layer.project(x, conv=conv, components=3, activation=act, norm=nrm, input_activation=inact)
            want = reference(x, mod, conv, 3, act, nrm, None, inact)
            self.assertEqual(got.shape, want.shape)
            rel = ((got - want).norm() / want.norm()).item()
            self.assertLess(rel, 1e-6, f"{n}x{k} conv={conv} act={act} norm={nrm is not None} in={inact}: {rel}")

    def test_rejections_and_empty(self):
        _, _, _, layer = make_layer(64, 128)
        with torch.inference_mode():
            self.assertEqual(tuple(layer.linear(torch.empty(0, 128, device="cuda")).shape), (0, 64))
            self.assertEqual(tuple(layer.conv(torch.empty(2, 128, 0, device="cuda")).shape), (2, 64, 0))
            with self.assertRaises(ValueError):
                layer.linear(torch.randn(3, 127, device="cuda"))
            with self.assertRaises(ValueError):
                layer.linear(torch.randn(3, 128, device="cuda", dtype=torch.float16))
            with self.assertRaises(ValueError):
                layer.conv(torch.randn(3, 128, device="cuda"))
        x = torch.randn(3, 128, device="cuda", requires_grad=True)
        with self.assertRaises(RuntimeError):
            layer.linear(x)
        with torch.no_grad():
            self.assertFalse(layer.linear(x).requires_grad)

    @torch.inference_mode()
    def test_default_configs_for_model_shapes(self):
        for (n, k), m in itertools.product([(1024, 1024), (3072, 1024), (4096, 1024), (2048, 1024), (1024, 4096)],
                                           [38, 75, 126, 251, 376, 751]):
            codes, scale, b, layer = make_layer(n, k, seed=m)
            x = torch.randn(1, m, k, device="cuda")
            q, qs = m1_quantize(x)
            expect = m1_dot_split1(q, qs, api.unpack_signs(codes, k), layer.scale, None, bk=128)
            self.assert_m1_elementwise(layer, q, qs, k, m, expect, f"{n}x{k} m={m} {configs.choose(m, n, k).label()}")


if __name__ == "__main__":
    unittest.main()
