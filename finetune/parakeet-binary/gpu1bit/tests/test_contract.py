"""Contract tests every BinaryLayer implementation must pass (gpu1bit/shared/api.py docstring;
plans/gpu-kernel.md A3, A5, A8). One TestCase class per registered implementation (registry.py).

Checks, in A8's terms: ragged K (not a multiple of 8/32), N and M (1, odd, > tile); bit order
(exact integer semantics) and nonzero-tail rejection; bias / no bias; batch > 1; every input
stride pattern the transforms produce (contiguous, transposed conv view, sliced) plus 1-D/2-D/4-D
linear inputs; M = 0; CUDA-graph capture and replay with changed contents equal to eager, bitwise;
alternate stream; no output aliasing; grad-enabled input rejected; determinism; per-layer storage
(code storage <= floor(1.10 N ceil(K/8)), honest weight_bytes, no hidden allocations); synthetic
numerics (A3: gaussian, ragged, wide dynamic range, exact cancellation, silence -> y = bias
exactly, single nonzero input, M = 1) against 1.5x the worse of M1's R-bf16x3 and R-expanded
INT8x3 on the same inputs. Real-data numerics: test_numerics_real.py. Provider coverage and the
whole-model A5 audit: test_provider.py. compute-sanitizer: shared/sanitize.py (if available).
"""
from __future__ import annotations

import gc
import unittest

import torch

from gpu1bit.shared import api, audit, registry

DEVICE = torch.device("cuda")
SLACK = api.UNACCOUNTED_SLACK_BYTES


# --- data ----------------------------------------------------------------------------------------

def make_weights(n, k, seed, *, integer=False, bias=False, scale_range=(0.01, 0.05)):
    g = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (n, k), generator=g, dtype=torch.int8) * 2 - 1).to(torch.int8)
    if integer:
        scale = torch.tensor([0.5, 1.0, 2.0, 4.0])[torch.randint(0, 4, (n,), generator=g)]
        b = torch.randint(-5, 6, (n,), generator=g).float() if bias else None
    else:
        lo, hi = scale_range
        scale = lo + (hi - lo) * torch.rand(n, generator=g)
        b = torch.randn(n, generator=g) * 0.1 if bias else None
    return {"signs": signs, "codes": api.pack_signs(signs), "scale": scale.float().contiguous(), "bias": b, "N": n, "K": k}


def make_input(kind, layout, shape, k, seed, *, integer=False, device=DEVICE):
    """An input of logical `shape` (linear [..., K] / conv [B, K, T]) with the requested stride pattern."""
    g = torch.Generator().manual_seed(seed)

    def fill(s):
        return (torch.randint(-3, 4, s, generator=g).float() if integer else torch.randn(s, generator=g)).to(device)

    if kind == "linear":
        if layout == "contiguous":
            return fill(shape)
        if layout == "transposed":          # NeMo-style [B, T, K] view of [B, K, T] storage
            b, t, kk = shape
            return fill((b, kk, t)).transpose(1, 2)
        if layout == "k_sliced":
            return fill((*shape[:-1], 2 * shape[-1]))[..., ::2]
        if layout == "rows_sliced":
            b, t, kk = shape
            return fill((b, 2 * t, kk))[:, ::2, :]
    else:
        b, kk, t = shape
        if layout == "contiguous":           # pointwise_conv2's input
            return fill(shape)
        if layout == "transposed":           # pointwise_conv1's input: x.transpose(1, 2) of [B, T, K]
            return fill((b, t, kk)).transpose(1, 2)
        if layout == "t_sliced":
            return fill((b, kk, 2 * t))[..., ::2]
        if layout == "k_sliced":
            return fill((b, 2 * kk, t))[:, ::2, :]
    raise ValueError(f"{kind} layout {layout}")


SEMANTIC_LINEAR = [  # (N, K, x shape, layout, bias)
    (3, 1, (1, 1), "contiguous", False), (5, 7, (2, 3, 7), "contiguous", True), (17, 9, (9,), "contiguous", False),
    (64, 31, (1, 38, 31), "transposed", True), (1, 33, (131, 33), "contiguous", False),
    (100, 100, (2, 5, 3, 100), "contiguous", True), (1000, 1023, (1, 7, 1023), "k_sliced", False),
    (1024, 1025, (1, 1, 1025), "contiguous", True), (3072, 1024, (1, 38, 1024), "contiguous", False),
    (1024, 4096, (2, 131, 4096), "rows_sliced", True), (37, 4097, (3, 4097), "contiguous", False),
    (4096, 1024, (1, 126, 1024), "transposed", False), (2049, 64, (1, 377, 64), "contiguous", True),
]
SEMANTIC_CONV = [  # (N, K, (B, K, T), layout, bias)
    (5, 7, (1, 7, 3), "contiguous", False), (64, 33, (2, 33, 38), "transposed", True),
    (2048, 1024, (1, 1024, 38), "transposed", False), (1024, 1024, (2, 1024, 17), "contiguous", True),
    (9, 1025, (3, 1025, 5), "t_sliced", False), (1, 8, (1, 8, 1), "contiguous", True),
    (1024, 1024, (1, 1024, 126), "k_sliced", False), (2048, 1024, (1, 1024, 368), "transposed", True),
]
# (name, kind, N, K, x shape, layout, bias, generator)
PRECISION = [
    ("gaussian", "linear", 1024, 1024, (1, 38, 1024), "contiguous", False, "gaussian"),
    ("gaussian_conv", "conv", 2048, 1024, (1, 1024, 38), "transposed", False, "gaussian"),
    ("ragged", "linear", 1000, 1023, (3, 7, 1023), "contiguous", True, "gaussian"),
    ("ragged_conv", "conv", 9, 1025, (2, 1025, 5), "t_sliced", True, "gaussian"),
    ("wide_range", "linear", 1024, 4096, (1, 38, 4096), "contiguous", False, "wide"),
    ("cancellation", "linear", 512, 1024, (1, 38, 1024), "contiguous", False, "cancel"),
    ("silence", "linear", 1024, 1024, (1, 38, 1024), "contiguous", True, "zero"),
    ("silence_conv", "conv", 1024, 1024, (1, 1024, 38), "contiguous", True, "zero"),
    ("silence_no_bias", "linear", 4096, 1024, (2, 19, 1024), "contiguous", False, "zero"),
    ("single_nonzero", "linear", 4096, 1024, (1, 38, 1024), "contiguous", False, "onehot"),
    ("one_row", "linear", 1024, 1024, (1, 1, 1024), "contiguous", False, "gaussian"),
    ("batch3_conv", "conv", 1024, 1024, (3, 1024, 17), "transposed", False, "gaussian"),
]


def precision_data(case, seed):
    name, kind, n, k, shape, layout, bias, gen = case
    w = make_weights(n, k, seed, bias=bias)
    x = make_input(kind, layout, shape, k, seed + 1)
    g = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        if gen == "wide":
            x.mul_((10.0 ** (8 * torch.rand(x.shape, generator=g) - 4)).to(x.device))
        elif gen == "zero":
            x.zero_()
        elif gen == "onehot":
            rows = x.reshape(-1, k) if kind == "linear" and x.is_contiguous() else None
            x.zero_()
            pos = torch.randint(0, k, (rows.shape[0],), generator=g)
            rows[torch.arange(rows.shape[0]), pos.to(x.device)] = (torch.randn(rows.shape[0], generator=g) * 3).to(x.device)
        elif gen == "cancel":
            # columns 2i and 2i+1 have equal weights for i < (K-8)/2; x carries +a / -a there (|a| ~ 1e3),
            # so those terms cancel exactly and y_ref is the small remainder of the last 8 columns.
            pairs = (k - 8) // 2
            signs = w["signs"].clone()
            signs[:, 1:2 * pairs:2] = signs[:, 0:2 * pairs:2]
            w = {**w, "signs": signs, "codes": api.pack_signs(signs)}
            a = torch.randn(x.shape[:-1] + (pairs,), generator=g).to(x.device) * 1e3
            x[..., 0:2 * pairs:2] = a
            x[..., 1:2 * pairs:2] = -a
            x[..., 2 * pairs:] = (torch.randn(x.shape[:-1] + (k - 2 * pairs,), generator=g) * 1e-3).to(x.device)
    return w, x


# --- helpers -------------------------------------------------------------------------------------

def prepare(cls, w, device=DEVICE):
    return cls.prepare(w["codes"], w["scale"], w["bias"], w["K"], device)


def run(layer, kind, x):
    return api.call(layer, kind, x)


def reference(kind, x, w):
    return api.reference(kind, x, w["codes"], w["scale"], w["bias"], w["K"])


def storages_overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.numel() == 0 or b.numel() == 0:
        return False
    sa, sb = a.untyped_storage(), b.untyped_storage()
    a0, a1 = sa.data_ptr(), sa.data_ptr() + sa.nbytes()
    b0, b1 = sb.data_ptr(), sb.data_ptr() + sb.nbytes()
    return a0 < b1 and b0 < a1


def capture_and_replay(fn, static_x, new_values):
    """Warm up on a side stream, capture, then for each new value: copy, replay, return outputs."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(static_x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(static_x)
    results = []
    for value in new_values:
        static_x.copy_(value)
        graph.replay()
        torch.cuda.synchronize()
        results.append(out.clone())
    del graph
    return results


_M1_CACHE: dict = {}


def m1_thresholds(case, w, x):
    """api.numerics_thresholds from M1's R-bf16x3 and R-expanded INT8x3 on (w, x); cached per case."""
    from gpu1bit.shared import m1
    key = case[0]
    if key not in _M1_CACHE:
        kind = case[1]
        ops = m1.LayerOps(w["codes"], w["scale"], w["bias"], w["K"])
        ref = reference(kind, x, w)
        errs = {name: api.errors(kind, x, f(x, kind == "conv"), ref, w["scale"]) for name, f in ops.ops().items()}
        _M1_CACHE[key] = (errs, api.numerics_thresholds(errs))
        del ops
    return _M1_CACHE[key]


# --- the contract --------------------------------------------------------------------------------

class Contract:
    impl_name: str = ""
    impl = None

    def setUp(self):
        if isinstance(self.impl, Exception):
            self.fail(str(self.impl))
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    # structure -----------------------------------------------------------------------------------
    def test_protocol(self):
        w = make_weights(8, 16, 1)
        layer = prepare(self.impl, w)
        self.assertIsInstance(layer, api.BinaryLayer)
        self.assertIsInstance(layer.weight_bytes(), int)

    def test_prepare_rejects_invalid_weights(self):
        w = make_weights(4, 9, 2)
        bad_tail = w["codes"].clone()
        bad_tail[2, 1] |= 0b100  # bit 10 of a K = 9 row
        cases = {
            "tail bits": (bad_tail, w["scale"], None, 9),
            "codes dtype": (w["codes"].to(torch.int8), w["scale"], None, 9),
            "codes width": (w["codes"], w["scale"], None, 17),
            "scale shape": (w["codes"], torch.ones(5), None, 9),
            "scale dtype": (w["codes"], w["scale"].double(), None, 9),
            "scale non-finite": (w["codes"], torch.tensor([1.0, float("inf"), 1.0, 1.0]), None, 9),
            "bias shape": (w["codes"], w["scale"], torch.ones(3), 9),
            "bias dtype": (w["codes"], w["scale"], torch.ones(4, dtype=torch.float64), 9),
        }
        for label, args in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                self.impl.prepare(*args, DEVICE)

    def test_prepare_does_not_modify_or_alias_inputs(self):
        w = make_weights(33, 70, 3, bias=True)
        for device in ("cpu", "cuda"):
            codes, scale, bias = (t.to(device).clone() for t in (w["codes"], w["scale"], w["bias"]))
            before = [t.clone() for t in (codes, scale, bias)]
            layer = self.impl.prepare(codes, scale, bias, 70, DEVICE)
            torch.cuda.synchronize()
            for t, b in zip((codes, scale, bias), before):
                self.assertTrue(torch.equal(t, b), "prepare modified its inputs")
            if device == "cuda":
                x = make_input("linear", "contiguous", (4, 70), 70, 4)
                y0 = run(layer, "linear", x).clone()
                codes.zero_(); scale.zero_(); bias.zero_()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(run(layer, "linear", x), y0), "layer aliases the caller's tensors")

    def test_semantics_integer_exact(self):
        """Every element within 0.2 of the exact FP64 result (results lie on a 0.5 grid)."""
        for i, (n, k, shape, layout, bias) in enumerate(SEMANTIC_LINEAR):
            with self.subTest(kind="linear", n=n, k=k, shape=shape, layout=layout, bias=bias):
                w = make_weights(n, k, 100 + i, integer=True, bias=bias)
                layer = prepare(self.impl, w)
                x = make_input("linear", layout, shape, k, 200 + i, integer=True)
                y = run(layer, "linear", x)
                self.assertEqual(tuple(y.shape), (*shape[:-1], n))
                err = (y.double() - reference("linear", x, w)).abs().max().item()
                self.assertLess(err, 0.2)
        for i, (n, k, shape, layout, bias) in enumerate(SEMANTIC_CONV):
            with self.subTest(kind="conv", n=n, k=k, shape=shape, layout=layout, bias=bias):
                w = make_weights(n, k, 300 + i, integer=True, bias=bias)
                layer = prepare(self.impl, w)
                x = make_input("conv", layout, shape, k, 400 + i, integer=True)
                y = run(layer, "conv", x)
                self.assertEqual(tuple(y.shape), (shape[0], n, shape[2]))
                err = (y.double() - reference("conv", x, w)).abs().max().item()
                self.assertLess(err, 0.2)

    def test_empty_inputs(self):
        w = make_weights(12, 20, 5, bias=True)
        layer = prepare(self.impl, w)
        for shape in [(0, 20), (2, 0, 20)]:
            with self.subTest(shape=shape):
                y = layer.linear(torch.empty(shape, device=DEVICE))
                self.assertEqual(tuple(y.shape), (*shape[:-1], 12))
        for shape in [(0, 20, 5), (2, 20, 0)]:
            with self.subTest(shape=shape):
                y = layer.conv(torch.empty(shape, device=DEVICE))
                self.assertEqual(tuple(y.shape), (shape[0], 12, shape[2]))

    def test_invalid_inputs_raise(self):
        w = make_weights(12, 20, 6)
        layer = prepare(self.impl, w)
        bad = {"cpu": torch.randn(3, 20), "fp16": torch.randn(3, 20, device=DEVICE, dtype=torch.float16),
               "fp64": torch.randn(3, 20, device=DEVICE, dtype=torch.float64),
               "wrong K": torch.randn(3, 21, device=DEVICE)}
        for label, x in bad.items():
            with self.subTest(op="linear", case=label), self.assertRaises(ValueError):
                layer.linear(x)
        bad_conv = {"rank 2": torch.randn(20, 5, device=DEVICE), "wrong K": torch.randn(1, 21, 5, device=DEVICE),
                    "cpu": torch.randn(1, 20, 5), "fp16": torch.randn(1, 20, 5, device=DEVICE, dtype=torch.float16)}
        for label, x in bad_conv.items():
            with self.subTest(op="conv", case=label), self.assertRaises(ValueError):
                layer.conv(x)

    def test_inference_only(self):
        w = make_weights(12, 20, 7)
        layer = prepare(self.impl, w)
        x = torch.randn(2, 5, 20, device=DEVICE, requires_grad=True)
        with torch.enable_grad():
            with self.assertRaises(RuntimeError):
                layer.linear(x)
            with self.assertRaises(RuntimeError):
                layer.conv(x.transpose(1, 2))
        with torch.no_grad():
            self.assertFalse(layer.linear(x).requires_grad)
        with torch.inference_mode():
            xi = torch.randn(2, 5, 20, device=DEVICE)
            self.assertFalse(layer.linear(xi).requires_grad)
            self.assertFalse(layer.conv(xi.transpose(1, 2)).requires_grad)

    def test_outputs_are_new_and_do_not_alias(self):
        w = make_weights(1024, 1024, 8)
        layer = prepare(self.impl, w)
        for kind, layout, shape in [("linear", "contiguous", (1, 38, 1024)), ("conv", "transposed", (1, 1024, 38)),
                                    ("conv", "contiguous", (1, 1024, 38))]:
            with self.subTest(kind=kind, layout=layout):
                x = make_input(kind, layout, shape, 1024, 9)
                x_copy = x.clone()
                y1 = run(layer, kind, x)
                y1_copy = y1.clone()
                y2 = run(layer, kind, x)
                torch.cuda.synchronize()
                self.assertEqual(y1.dtype, torch.float32)
                self.assertEqual(y1.device, x.device)
                if kind == "linear":
                    self.assertTrue(y1.is_contiguous())
                self.assertFalse(storages_overlap(y1, x), "output aliases the input")
                self.assertFalse(storages_overlap(y1, y2), "two calls returned the same buffer")
                self.assertTrue(torch.equal(y1, y1_copy), "an earlier output changed")
                self.assertTrue(torch.equal(x, x_copy), "the input changed")

    def test_deterministic(self):
        for kind, n, k, shape, layout in [("linear", 1024, 1024, (1, 38, 1024), "contiguous"),
                                          ("linear", 1024, 4096, (1, 126, 4096), "contiguous"),
                                          ("conv", 2048, 1024, (1, 1024, 38), "transposed"),
                                          ("linear", 3072, 1024, (1, 368, 1024), "contiguous")]:
            with self.subTest(kind=kind, n=n, k=k, shape=shape):
                w = make_weights(n, k, 10)
                layer = prepare(self.impl, w)
                x = make_input(kind, layout, shape, k, 11)
                ys = [run(layer, kind, x).clone() for _ in range(3)]
                self.assertTrue(all(torch.equal(ys[0], y) for y in ys[1:]))

    def test_cuda_graph_replay_equals_eager(self):
        for kind, n, k, shape, layout in [("linear", 1024, 1024, (1, 38, 1024), "contiguous"),
                                          ("linear", 4096, 1024, (1, 126, 1024), "contiguous"),
                                          ("linear", 1024, 4096, (1, 38, 4096), "contiguous"),
                                          ("conv", 2048, 1024, (1, 1024, 38), "transposed"),
                                          ("conv", 1024, 1024, (1, 1024, 126), "contiguous"),
                                          ("linear", 37, 1001, (2, 5, 1001), "transposed")]:
            with self.subTest(kind=kind, n=n, k=k, shape=shape, layout=layout):
                w = make_weights(n, k, 12, bias=(n == 37))
                layer = prepare(self.impl, w)
                static = make_input(kind, layout, shape, k, 13)
                values = [make_input(kind, "contiguous", tuple(static.shape), k, 14 + j) for j in range(2)]
                replayed = capture_and_replay(lambda x: run(layer, kind, x), static, values)
                for value, got in zip(values, replayed):
                    static.copy_(value)
                    eager = run(layer, kind, static)
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(got, eager), "graph replay differs from eager")

    def test_alternate_stream(self):
        w = make_weights(2048, 1024, 15)
        layer = prepare(self.impl, w)
        x = make_input("conv", "transposed", (1, 1024, 38), 1024, 16)
        expected = run(layer, "conv", x).clone()
        xl = make_input("linear", "contiguous", (1, 38, 1024), 1024, 17)
        expected_l = run(layer, "linear", xl).clone()
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            x2 = torch.empty_like(x.transpose(1, 2).contiguous()).transpose(1, 2)
            x2.copy_(x)
            got = run(layer, "conv", x2)
            got_l = run(layer, "linear", xl.clone())
        s.synchronize()
        self.assertTrue(torch.equal(got, expected))
        self.assertTrue(torch.equal(got_l, expected_l))

    # storage -------------------------------------------------------------------------------------
    def test_storage_budget_and_honest_weight_bytes(self):
        from gpu1bit.shared import fixtures
        names = ["encoder.layers.0.self_attn.linear_out", "encoder.layers.0.feed_forward1.linear1",
                 "encoder.layers.0.feed_forward1.linear2", "encoder.layers.0.conv.pointwise_conv1",
                 "encoder.layers.0.conv.pointwise_conv2"]
        real = fixtures.mb1_layers(names)
        q = fixtures.mb1_layers([f"encoder.layers.0.self_attn.linear_{p}" for p in "qkv"])
        fused = {"name": "qkv", "codes": torch.cat([q[n]["codes"] for n in q]), "scale": torch.cat([q[n]["scale"] for n in q]),
                 "bias": None, "N": 3072, "K": 1024, "kind": "linear"}
        for w in list(real.values()) + [fused, make_weights(37, 1001, 18, bias=True) | {"kind": "linear"}]:
            with self.subTest(layer=w.get("name", "random"), n=w["N"], k=w["K"]):
                gc.collect()
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                layer = prepare(self.impl, w)
                torch.cuda.synchronize()
                gc.collect()
                grown = torch.cuda.memory_allocated() - before
                reported = layer.weight_bytes()
                found = audit.layer_storage(layer, w["scale"], w["bias"])
                self.assertLessEqual(found["code_bytes"], api.code_budget_bytes(w["N"], w["K"]),
                                     f"code storage {found['code_bytes']} over budget: {found['storages']}")
                self.assertLessEqual(reported, api.weight_budget_bytes(w["N"], w["K"], w["bias"] is not None))
                self.assertGreaterEqual(reported, found["total_bytes"], "weight_bytes() under-reports reachable storage")
                self.assertLessEqual(grown, reported + SLACK, f"prepare allocated {grown} B, weight_bytes() says {reported}")
                for m in (1, 38, 126, 368):
                    for kind in ("linear", "conv"):
                        x = make_input(kind, "contiguous", (1, m, w["K"]) if kind == "linear" else (1, w["K"], m), w["K"], 19)
                        run(layer, kind, x)
                        del x
                torch.cuda.synchronize()
                gc.collect()
                grown = torch.cuda.memory_allocated() - before
                self.assertLessEqual(grown, reported + SLACK, f"calls left {grown} B allocated; weight_bytes() {reported}")
                del layer

    # numerics ------------------------------------------------------------------------------------
    def test_synthetic_numerics_vs_m1(self):
        for i, case in enumerate(PRECISION):
            name, kind = case[0], case[1]
            with self.subTest(case=name):
                w, x = precision_data(case, 1000 + 10 * i)
                m1_errors, thresholds = m1_thresholds(case, w, x)
                layer = prepare(self.impl, w)
                y = run(layer, kind, x)
                got = api.errors(kind, x, y, reference(kind, x, w), w["scale"])
                self.assertTrue(got["finite"], "non-finite output")
                if case[7] == "zero":
                    expected = (w["bias"] if w["bias"] is not None else torch.zeros(w["N"])).to(DEVICE)
                    expected = expected.view(1, -1, 1) if kind == "conv" else expected
                    self.assertTrue(torch.equal(y, expected.expand_as(y)), "silence must give y = bias exactly")
                for metric, limit in thresholds.items():
                    self.assertLessEqual(got[metric], limit, f"{metric} {got[metric]:.3e} > 1.5 x M1 worse "
                                         f"({m1_errors['r_bf16x3'][metric]:.3e}, {m1_errors['r_expanded_int8x3'][metric]:.3e})")


def _register():
    for name, impl in registry.discover().items():
        cls_name = "Contract_" + "".join(c if c.isalnum() else "_" for c in name)
        globals()[cls_name] = type(cls_name, (Contract, unittest.TestCase), {"impl_name": name, "impl": impl})


_register()

if __name__ == "__main__":
    unittest.main()
