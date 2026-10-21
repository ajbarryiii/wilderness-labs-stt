"""int8_mma:b01 contract: bitwise equal to the frozen int8_mma on the same inputs, for every config.

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.gpulock gpu1bit.int8_mma.test_b01
"""
from __future__ import annotations

import itertools
import unittest

import torch

from gpu1bit.int8_mma.kernels import DotConfig
from gpu1bit.int8_mma.layer import Int8MmaLayer
from gpu1bit.int8_mma.layer_b01 import Int8MmaB01Layer
from gpu1bit.int8_mma.test_int8_mma import m1_quantize
from gpu1bit.shared import api, env

_LOCK = None

# Every exchange mode, WK groups, cluster sizes, prefetch, tiles; (wm, wn, rm, rn, stages, cluster, xchg, wk, pf)
CONFIGS = [DotConfig(1, 4, 1, 2, 2, 1), DotConfig(1, 4, 3, 2, 2, 1, 0, 1, 1), DotConfig(2, 2, 2, 2, 3, 1),
           DotConfig(1, 4, 1, 2, 4, 1, 0, 2, 1), DotConfig(1, 2, 2, 1, 2, 1, 0, 3), DotConfig(1, 2, 1, 1, 2, 1, 0, 4),
           DotConfig(1, 4, 3, 2, 2, 8), DotConfig(1, 4, 1, 1, 4, 3, 1), DotConfig(1, 2, 1, 4, 3, 4, 1, 2),
           DotConfig(1, 4, 3, 2, 2, 8, 3), DotConfig(1, 8, 1, 1, 3, 4, 3, 2), DotConfig(2, 2, 1, 3, 2, 3, 3),
           DotConfig(1, 4, 1, 3, 3, 8, 3, 1), DotConfig(1, 8, 2, 3, 3, 1, 0, 1, 1), DotConfig(1, 8, 1, 2, 2, 5, 1),
           DotConfig(1, 4, 3, 2, 2, 8, 2), DotConfig(1, 4, 1, 2, 2, 4, 2, 2)]
DEFERRED = [DotConfig(1, 4, 1, 3, 3, 4, 4, 1, 1), DotConfig(1, 8, 1, 1, 3, 8, 4), DotConfig(1, 4, 2, 2, 2, 3, 4, 2),
            DotConfig(1, 4, 1, 4, 4, 2, 4, 2, 1)]


def setUpModule():
    global _LOCK
    if not getattr(env.gpu_lock, "_held", False):
        _LOCK = env.gpu_lock("B:b01 tests")
        _LOCK.__enter__()


def tearDownModule():
    if _LOCK is not None:
        _LOCK.__exit__(None, None, None)


def layers(n, k, *, signs=None, bias=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    if signs is None:
        signs = torch.randint(0, 2, (n, k), generator=g, dtype=torch.int8) * 2 - 1
    codes = api.pack_signs(signs)
    scale = torch.rand(n, generator=g) * 0.05 + 1e-3
    b = torch.randn(n, generator=g) * 0.1 if bias else None
    return (Int8MmaLayer.prepare(codes, scale, b, k, "cuda"), Int8MmaB01Layer.prepare(codes, scale, b, k, "cuda"))


def both(frozen, b01, q, qs, ldq, m, shape, **kw):
    y0 = torch.full(shape, float("nan"), device="cuda")
    y1 = torch.full(shape, float("nan"), device="cuda")
    frozen.dot(q, qs, ldq, m, y0.view(-1, shape[-1]) if len(shape) == 2 else y0[0], **kw)
    b01.dot(q, qs, ldq, m, y1.view(-1, shape[-1]) if len(shape) == 2 else y1[0], **kw)
    torch.cuda.synchronize()
    return y0, y1


class B01Tests(unittest.TestCase):
    def assert_same(self, y0, y1, msg):
        self.assertFalse(torch.isnan(y1).any(), f"{msg}: unwritten outputs")
        self.assertTrue(torch.equal(y0, y1), f"{msg}: {(y0 != y1).sum().item()} differ, max {(y0 - y1).abs().max().item()}")

    @torch.inference_mode()
    def test_every_config_bitwise_equal_to_frozen(self):
        """split-K (cluster 1-8, every exchange), WK 1-4, PF, tiles, K and M tails, bias, SiLU."""
        for (n, k), m, bias, silu in itertools.product([(1024, 1024), (256, 4096), (1000, 1040), (96, 1008)],
                                                       [1, 17, 38, 130], [False, True], [False, True]):
            frozen, b01 = layers(n, k, bias=bias, seed=n + k + m)
            q, qs = m1_quantize(torch.randn(1, m, k, device="cuda"))
            for cfg in CONFIGS:
                y0, y1 = both(frozen, b01, q, qs, k, m, (m, n), silu=silu, config=cfg)
                self.assert_same(y0, y1, f"{n}x{k} m={m} bias={bias} silu={silu} {cfg.label()}")

    @torch.inference_mode()
    def test_row_sum_extremes(self):
        """All q = +127 / -127 / alternating, weights all +1 / all -1 / random, K up to 4096 (|acc| = 127 K)."""
        for (n, k), m in itertools.product([(64, 4096), (64, 1024), (24, 1040)], [1, 16, 37]):
            g = torch.Generator().manual_seed(k + m)
            weight_sets = {"plus": torch.ones(n, k, dtype=torch.int8), "minus": -torch.ones(n, k, dtype=torch.int8),
                           "random": torch.randint(0, 2, (n, k), generator=g, dtype=torch.int8) * 2 - 1}
            qsets = {"plus127": torch.full((3, m, k), 127, dtype=torch.int8),
                     "minus127": torch.full((3, m, k), -127, dtype=torch.int8),
                     "alternating": (torch.arange(k) % 2 * 254 - 127).to(torch.int8).expand(3, m, k).contiguous(),
                     "random": torch.randint(-127, 128, (3, m, k), generator=g, dtype=torch.int8)}
            qs = torch.rand(3, m, generator=g).cuda() * 1e-2 + 1e-4
            for (wname, signs), (qname, q) in itertools.product(weight_sets.items(), qsets.items()):
                frozen, b01 = layers(n, k, signs=signs)
                q = q.cuda()
                for cfg in [DotConfig(1, 4, 1, 2, 2, 1), DotConfig(1, 2, 1, 1, 2, 4, 3, 2), DotConfig(1, 4, 1, 1, 2, 8, 1),
                            DotConfig(1, 2, 1, 1, 2, 1, 0, 4, 1), DotConfig(1, 4, 3, 2, 2, 1, 0, 1, 1)]:
                    y0, y1 = both(frozen, b01, q, qs, k, m, (m, n), config=cfg)
                    self.assert_same(y0, y1, f"{n}x{k} m={m} w={wname} q={qname} {cfg.label()}")
                if wname == "plus" and qname == "plus127":  # exact value check: y = 127 K (sum of scales) alpha
                    ref = (127.0 * k * qs.sum(0))[:, None].double() * frozen.scale.double()[None, :]
                    self.assertLess(((y1.double() - ref).abs() / ref.abs()).max().item(), 1e-6)

    @torch.inference_mode()
    def test_cluster_wk_split_variation_against_one_frozen_output(self):
        """One input, every split structure: all b01 outputs equal ONE frozen output bitwise."""
        n, k, m = 1024, 4096, 38
        frozen, b01 = layers(n, k, seed=11)
        q, qs = m1_quantize(torch.randn(1, m, k, device="cuda"))
        y0 = torch.empty(m, n, device="cuda")
        frozen.dot(q, qs, k, m, y0, config=DotConfig(1, 4, 1, 2, 2, 1))
        for cfg in [c for c in CONFIGS if c.exact] + [DotConfig(1, 4, 1, 1, 2, c, x, w, p)
                                                    for c, x, w, p in itertools.product([1, 2, 4, 8], [1, 3], [1, 2, 4], [0, 1])
                                                    if not (c == 1 and x != 1)]:
            y1 = torch.full((m, n), float("nan"), device="cuda")
            b01.dot(q, qs, k, m, y1, config=cfg)
            self.assert_same(y0, y1, cfg.label())

    @torch.inference_mode()
    def test_deferred_slices_bitwise_equal(self):
        """xchg 4 (B-hybrid's deferred calls): each FP32 slice equals the frozen slice."""
        for (n, k), m in itertools.product([(1024, 1024), (1024, 4096)], [1, 38, 130]):
            frozen, b01 = layers(n, k, seed=n + m)
            q, qs = m1_quantize(torch.randn(1, m, k, device="cuda"))
            for cfg in DEFERRED:
                p0 = torch.full((cfg.cluster, m, n), float("nan"), device="cuda")
                p1 = torch.full_like(p0, float("nan"))
                frozen.dot(q, qs, k, m, p0[0], config=cfg)
                b01.dot(q, qs, k, m, p1[0], config=cfg)
                torch.cuda.synchronize()
                self.assert_same(p0, p1, f"{n}x{k} m={m} {cfg.label()}")

    @torch.inference_mode()
    def test_project_linear_conv_bitwise_equal(self):
        """The provider op (norm / SiLU fusions, conv views) and BinaryLayer calls with tuned configs."""
        norm = torch.nn.LayerNorm(1024).cuda()
        norm.weight.uniform_(0.5, 1.5)
        calls = [(3072, 1024, False, "", norm, ""), (1024, 1024, False, "", None, ""), (4096, 1024, False, "silu", norm, ""),
                 (1024, 4096, False, "", None, ""), (2048, 1024, True, "", norm, ""), (1024, 1024, True, "", None, "silu")]
        for (n, k, conv, act, nrm, inact), (b, t) in itertools.product(calls, [(1, 38), (1, 126), (1, 368), (2, 63)]):
            frozen, b01 = layers(n, k, seed=n + t)
            base = torch.randn(b, t, k, device="cuda")
            x = torch.randn(b, k, t, device="cuda") if (conv and inact) else (base.transpose(1, 2) if conv else base)
            kw = dict(conv=conv, components=3, activation=act, norm=nrm, input_activation=inact)
            self.assert_same(frozen.project(x, **kw), b01.project(x, **kw), f"project {n}x{k} {b}x{t} conv={conv}")
            self.assert_same(frozen.linear(base), b01.linear(base), f"linear {n}x{k} {b}x{t}")
            xc = base.transpose(1, 2)
            self.assert_same(frozen.conv(xc), b01.conv(xc), f"conv {n}x{k} {b}x{t}")
        for n, k in [(1000, 1000), (37, 33), (1, 5)]:  # ragged; small ones use the MB1-layout fallback
            frozen, b01 = layers(n, k, bias=True, seed=n)
            x = torch.randn(2, 9, k, device="cuda")
            self.assert_same(frozen.linear(x), b01.linear(x), f"ragged {n}x{k}")
            self.assertEqual(frozen.weight_bytes(), b01.weight_bytes())


if __name__ == "__main__":
    unittest.main()
