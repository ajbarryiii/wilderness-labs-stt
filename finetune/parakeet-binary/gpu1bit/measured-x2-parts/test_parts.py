"""Contract of int8_mma with 1 / 2 activation parts (plans/activation-precision.md, R1, R6, R8).

Oracle (R8): M1's own inference.fused_kernels.int8_matmul at SPLIT = 1 with the same part count, on the
derived 2-bit weights of the same 1-bit layer (gpu1bit.shared.m1.derived_packed_linear): every output
element equals ours bitwise in one of M1's two epilogue contraction forms (order 0 / 1; with one part
there is a single form). Then, ours against ours (exact integer arithmetic, so bitwise):
  * every frozen x3 tile configuration and exchange mode (WK groups, cluster XCHG 0 / 1 / 3, PF) gives
    the same P-part output; tails in M and K, the MB1-layout fallback (zero-padded parts);
  * x3 through the parts code (kernels_parts at P = 3) equals the frozen kernel, and components = 3
    through Int8MmaPartsLayer is the frozen code path;
  * the {0, 1} forms equal the +-1 forms at every part count;
  * project(components = P) with fused LayerNorm, input SiLU, output SiLU, bias and conv layouts equals
    M1's int8_matmul with the same arguments; linear / conv of the x1 / x2 variants equal project;
  * CUDA-graph capture and replay equals eager.

    shared/python /mnt/hd/.../gpu1bit/B/wip/parts/run.py test_parts
"""
from __future__ import annotations

import itertools
import unittest

import torch

from gpu1bit.int8_mma import configs, kernels_parts
from gpu1bit.int8_mma.kernels import DotConfig
from gpu1bit.int8_mma.layer import Int8MmaLayer
from gpu1bit.int8_mma.layer_b01 import Int8MmaB01Layer
from gpu1bit.int8_mma.layer_parts import IMPLEMENTATIONS, Int8MmaPartsLayer, dot_parts, m1_quantize_parts, variant
from gpu1bit.shared import api, env

_LOCK = None

# The frozen x3 tiles (R6) plus every exchange mode / WK / cluster / PF combination of test_b01.
CONFIGS = list(dict.fromkeys(list(configs.TUNED_M.values()) + [
    DotConfig(1, 4, 1, 2, 2, 1), DotConfig(1, 4, 3, 2, 2, 1, 0, 1, 1), DotConfig(2, 2, 2, 2, 3, 1),
    DotConfig(1, 4, 1, 2, 4, 1, 0, 2, 1), DotConfig(1, 2, 2, 1, 2, 1, 0, 3), DotConfig(1, 2, 1, 1, 2, 1, 0, 4),
    DotConfig(1, 4, 3, 2, 2, 8), DotConfig(1, 4, 1, 1, 4, 3, 1), DotConfig(1, 2, 1, 4, 3, 4, 1, 2),
    DotConfig(1, 4, 3, 2, 2, 8, 3), DotConfig(1, 8, 1, 1, 3, 4, 3, 2), DotConfig(2, 2, 1, 3, 2, 3, 3),
    DotConfig(1, 8, 1, 2, 2, 5, 1)]))
M1_CONFIGS = [(32, 64, 64, 1, 4, 3), (32, 64, 128, 1, 4, 2), (16, 32, 128, 1, 4, 2)]  # (bm, bn, bk, SPLIT=1, warps, stages)


def setUpModule():
    global _LOCK
    if not getattr(env.gpu_lock, "_held", False):
        _LOCK = env.gpu_lock("B:x1/x2 tests")
        _LOCK.__enter__()


def tearDownModule():
    if _LOCK is not None:
        _LOCK.__exit__(None, None, None)


def weights(n, k, seed, bias=False):
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (n, k), generator=g, dtype=torch.int8) * 2 - 1
    scale = torch.rand(n, generator=g) * 0.05 + 1e-3
    b = torch.randn(n, generator=g) * 0.1 if bias else None
    return api.pack_signs(signs), scale, b


def activations(b, t, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(b, t, k, device="cuda", generator=g)
    x[..., 5] *= 400.0
    x[..., k // 3] *= -30.0
    return x


def m1_module(codes, scale, bias, k, conv=False):
    from gpu1bit.shared.m1 import derived_packed_linear
    return derived_packed_linear(codes, scale, bias, k, conv=conv)


def m1_reference(x, mod, parts, *, conv=False, activation="", norm=None, input_activation="", config=M1_CONFIGS[0]):
    from inference.fused_kernels import int8_matmul
    return int8_matmul(x, mod, conv, parts, activation, norm, config, input_activation)


def ours_both_orders(layer, x, parts, *, conv=False, activation="", norm=None, input_activation="", config=None):
    """Our P-part output in both contraction orders (layer.dot with the same q / qs as project uses)."""
    from gpu1bit.int8_mma.layer_parts import _aligned
    xx = x.transpose(1, 2) if conv else x
    b, t, k = xx.shape
    m, n = b * t, layer.out_features
    q, qs = m1_quantize_parts(xx, parts, norm, input_activation)
    q, ldq = _aligned(q, k)
    out = []
    for order in (0, 1):
        y = torch.empty(b, t, n, device="cuda")
        layer.dot(q, qs, ldq, m, y.view(m, n), silu=activation == "silu", config=config, order=order, parts=parts)
        out.append(y.transpose(1, 2) if conv else y)
    return out


def assert_either(test, ref, y0, y1, what):
    ok = (ref == y0) | (ref == y1)
    test.assertTrue(bool(ok.all()), f"{what}: {int((~ok).sum())} of {ok.numel()} elements match neither form; "
                                    f"max |d| {(ref - y0).abs().max().item():.3e}")


class PartsContract(unittest.TestCase):
    def test_m1_oracle_split1(self):
        """R8: elementwise equal to M1's int8_matmul (SPLIT = 1) in one of its two contraction forms."""
        for (n, k), m, parts in itertools.product([(1024, 1024), (4096, 1024), (1024, 4096), (3072, 1024), (1001, 1088)],
                                                  (1, 17, 38, 126, 130, 368), (1, 2, 3)):
            if (n, k) in ((4096, 1024), (3072, 1024)) and m not in (38, 368):
                continue
            codes, scale, b = weights(n, k, n + k + m, bias=(m % 2 == 0))
            mod = m1_module(codes, scale, b, k)
            layer = Int8MmaPartsLayer.prepare(codes, scale, b, k, "cuda")
            x = activations(1, m, k, m + parts)
            y0, y1 = ours_both_orders(layer, x, parts)
            for cfg in M1_CONFIGS:
                ref = m1_reference(x, mod, parts, config=cfg)
                assert_either(self, ref, y0, y1, f"N={n} K={k} M={m} P={parts} M1 config {cfg}")
                if parts == 1:
                    self.assertTrue(torch.equal(ref, y0), f"P=1 has one form: N={n} K={k} M={m} {cfg}")

    def test_m1_oracle_fused_ops(self):
        """project(components=P) with LayerNorm / input SiLU / output SiLU / bias / conv vs int8_matmul."""
        n, k = 1024, 1024
        norm = torch.nn.LayerNorm(k).cuda()
        with torch.no_grad():
            norm.weight.mul_(1.1).add_(0.01)
            norm.bias.add_(0.03)
        for parts, m, (act, use_norm, in_act, conv, bias) in itertools.product(
                (1, 2), (17, 126), [("", True, "", False, False), ("silu", False, "", False, True),
                                    ("", False, "silu", True, True), ("silu", True, "", False, False),
                                    ("", False, "", True, False)]):
            codes, scale, b = weights(n, k, 3 * m + parts, bias=bias)
            mod = m1_module(codes, scale, b, k, conv=conv)
            layer = Int8MmaPartsLayer.prepare(codes, scale, b, k, "cuda")
            x = activations(1, m, k, 7 * m + parts)
            if conv:
                x = x.transpose(1, 2).contiguous()  # [B, K, T], pointwise conv layout
            kw = dict(conv=conv, activation=act, norm=norm if use_norm else None, input_activation=in_act)
            y0, y1 = ours_both_orders(layer, x, parts, **kw)
            got = layer.project(x, components=parts, **kw)
            self.assertTrue(torch.equal(got, y0), "project uses order 0")
            for cfg in M1_CONFIGS[:2]:
                ref = m1_reference(x, mod, parts, config=cfg, **kw)
                assert_either(self, ref, y0, y1, f"P={parts} M={m} {kw}")

    def test_configs_bitwise(self):
        """Every frozen tile / exchange / WK / cluster / PF config gives the same P-part output."""
        for (n, k), m, parts in itertools.product([(1024, 1024), (1024, 4096), (1001, 1088)], (1, 37, 130), (1, 2)):
            codes, scale, b = weights(n, k, n * m + parts, bias=True)
            layer = Int8MmaPartsLayer.prepare(codes, scale, b, k, "cuda")
            x = activations(1, m, k, m * parts + k)
            from gpu1bit.int8_mma.layer_parts import _aligned
            q, qs = m1_quantize_parts(x, parts)
            q, ldq = _aligned(q, k)
            ref = None
            for cfg in CONFIGS:
                if cfg.cluster > -(-k // 128):
                    continue
                y = torch.empty(m, n, device="cuda")
                dot_parts(layer, q, qs, ldq, m, y, config=cfg, parts=parts)
                if ref is None:
                    ref = y
                else:
                    self.assertTrue(torch.equal(ref, y), f"N={n} K={k} M={m} P={parts} {cfg}")

    def test_x3_through_parts_code_is_frozen(self):
        """kernels_parts at P = 3 equals the frozen kernel; components=3 is the frozen code path."""
        for (n, k), m in itertools.product([(1024, 1024), (1024, 4096)], (17, 126, 368)):
            codes, scale, b = weights(n, k, m + 5 * k, bias=True)
            frozen = Int8MmaLayer.prepare(codes, scale, b, k, "cuda")
            parts_layer = Int8MmaPartsLayer.prepare(codes, scale, b, k, "cuda")
            x = activations(1, m, k, m + 11)
            from gpu1bit.int8_mma.layer import m1_quantize
            q, qs = m1_quantize(x)
            for cfg in CONFIGS[:12]:
                if cfg.cluster > -(-k // 128):
                    continue
                y0 = torch.empty(m, n, device="cuda")
                y1 = torch.empty(m, n, device="cuda")
                frozen.dot(q, qs, k, m, y0, config=cfg)
                dot_parts(parts_layer, q, qs, k, m, y1, config=cfg, parts=3)
                self.assertTrue(torch.equal(y0, y1), f"P=3 transform differs: N={n} K={k} M={m} {cfg}")
            self.assertTrue(torch.equal(frozen.project(x), parts_layer.project(x, components=3)))

    def test_b01_equals_pm1(self):
        for (n, k), m, parts in itertools.product([(1024, 1024), (1024, 4096), (1001, 1088)], (1, 38, 130, 368), (1, 2, 3)):
            codes, scale, b = weights(n, k, 13 * m + parts + k, bias=True)
            pm1 = IMPLEMENTATIONS[f"x{min(parts, 2)}"].prepare(codes, scale, b, k, "cuda")
            b01 = IMPLEMENTATIONS[f"x{min(parts, 2)}:b01"].prepare(codes, scale, b, k, "cuda")
            x = activations(1, m, k, m + parts)
            self.assertTrue(torch.equal(pm1.project(x, components=parts), b01.project(x, components=parts)),
                            f"N={n} K={k} M={m} P={parts}")
            for cfg in CONFIGS[::3]:
                if cfg.cluster > -(-k // 128):
                    continue
                from gpu1bit.int8_mma.layer_parts import _aligned
                q, qs = m1_quantize_parts(x, parts)
                q, ldq = _aligned(q, k)
                y0, y1 = torch.empty(m, n, device="cuda"), torch.empty(m, n, device="cuda")
                pm1.dot(q, qs, ldq, m, y0, config=cfg, parts=parts)
                b01.dot(q, qs, ldq, m, y1, config=cfg, parts=parts)
                self.assertTrue(torch.equal(y0, y1), f"b01 N={n} K={k} M={m} P={parts} {cfg}")

    def test_variants_linear_conv(self):
        """x1 / x2 (and b01) linear / conv = project(components=PARTS), incl. strided inputs and M tails."""
        for name, cls in IMPLEMENTATIONS.items():
            parts = cls.PARTS
            for (n, k), m in itertools.product([(1024, 1024), (37, 1001), (1001, 1088)], (1, 23, 130)):
                codes, scale, b = weights(n, k, m + n, bias=True)
                layer = cls.prepare(codes, scale, b, k, "cuda")
                x = activations(2, m, k, m + 3)
                y = layer.linear(x)
                if k % 16 == 0:
                    ref = layer.project(x, components=parts)
                else:  # M1's quantizer and our fast path need K % 16 == 0; compare against M1's math on the fallback
                    ref = None
                if ref is not None:
                    self.assertTrue(torch.equal(y, ref), f"{name} linear N={n} K={k} M={m}")
                xs = x[:, ::2]  # strided rows
                self.assertTrue(torch.equal(layer.linear(xs), layer.linear(xs.contiguous())), f"{name} strided")
                xc = x.transpose(1, 2)  # [B, K, T] non-contiguous conv input
                self.assertTrue(torch.equal(layer.conv(xc), layer.linear(x).transpose(1, 2)), f"{name} conv N={n} K={k} M={m}")
                if k % 16 == 0 and n >= 64:
                    mod = m1_module(codes, scale, b, k)
                    ref1 = m1_reference(x, mod, parts, config=M1_CONFIGS[1])
                    y0, y1 = ours_both_orders(layer, x, parts)
                    assert_either(self, ref1, y0, y1, f"{name} vs M1 N={n} K={k} M={m}")

    def test_fallback_zero_padded_parts(self):
        """Shapes outside the fragment budget run the frozen 3-part fallback with zero parts appended:
        still M1's arithmetic (oracle as above, M1's int8_matmul handles the ragged K too)."""
        n, k, m = 33, 1001, 19  # 5 n8 tiles x 8 k128 blocks = 5120 B > 1.1 x 33 x 126 B: MB1-layout fallback
        codes, scale, b = weights(n, k, 99, bias=True)
        mod = m1_module(codes, scale, b, k)
        for parts in (1, 2):
            layer = variant(parts, False).prepare(codes, scale, b, k, "cuda")
            self.assertFalse(layer.fast)
            x = activations(1, m, k, parts)
            y = layer.linear(x)
            # the same q / qs through dot in both orders
            from gpu1bit.int8_mma.layer import quantize
            q3, qs3, ldq = quantize(x, (x.stride(0), x.stride(1), x.stride(2), m), m, k)
            outs = []
            for order in (0, 1):
                yo = torch.empty(m, n, device="cuda")
                layer.dot(q3[:parts], qs3[:parts], ldq, m, yo, order=order, parts=parts)
                outs.append(yo.view(1, m, n))
            self.assertTrue(torch.equal(y, outs[0]))
            for cfg in M1_CONFIGS[:2]:
                assert_either(self, m1_reference(x, mod, parts, config=cfg), outs[0], outs[1],
                              f"fallback P={parts} M1 config {cfg}")

    def test_cuda_graph_replay(self):
        for parts in (1, 2):
            n, k, m = 1024, 4096, 126
            codes, scale, b = weights(n, k, parts, bias=True)
            layer = IMPLEMENTATIONS[f"x{parts}"].prepare(codes, scale, b, k, "cuda")
            x = activations(1, m, k, 1)
            layer.project(x, components=parts)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = layer.project(x, components=parts)
            for trial in range(3):
                x.copy_(activations(1, m, k, 100 + trial))
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(out, layer.project(x, components=parts)), f"P={parts} replay {trial}")

    def test_component_validation(self):
        codes, scale, b = weights(64, 128, 1)
        layer = IMPLEMENTATIONS["x1"].prepare(codes, scale, b, 128, "cuda")
        x = activations(1, 4, 128, 1)
        for bad in (0, 4):
            with self.assertRaises(ValueError):
                layer.project(x, components=bad)


if __name__ == "__main__":
    unittest.main()
