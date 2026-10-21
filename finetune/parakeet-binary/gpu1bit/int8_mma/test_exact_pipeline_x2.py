"""Pipeline contracts of B-hybrid-x2 on MB1, dev audio only (plans/x2-default.md X1, X2, Z6).

The x2 stack is hybrid_x2.exact_model_x2: PartCount(probe(HybridProjectionX2), 2) inside
exact_hybrid_pipeline_x2; the x3 reference is the deployed B-hybrid (HybridProjectionExact over
int8_mma, exact_hybrid_pipeline) with the same probe. The probe records each call as the provider
receives it and forwards attribute writes. Paths (H's test_partcount protocol plus a replay with changed
input): s1 eager warm-up + capture, s1 replay, s1 replay with changed input (same length), s2-s4 first
calls, s3 replay with changed input, s5 on the eager fallback after cache overflow, s1 / s3 direct
eager. The shapes put both deferred routes (and so both row paths) under capture. Models are built one
at a time.

X1 coverage (X2Coverage):
  * the adapter is H's fulldev.PartCount, and a write to it reaches the provider;
  * every call reaches the inner provider with components 2; PartCount's record shows the requested
    count, 3 for every call made at layer_fusions4.py:72 (pointwise_conv2, found by caller frame);
  * per path, the call sequence (site, caller file:line, N, K, conv, activation, norm, input
    activation, input shape / strides, capturing, defer flag, no-defer membership, cached activation
    hit, route) equals x3's; the route table (deferred XCHG 5 / in-kernel finish / plain) is x3's and
    contains both deferred routes; the inner provider's defer flag is set exactly where x3's is, and
    every deferrable call deferred;
  * replay makes no call; 4 graph entries after 4 shapes and after the overflow; outputs finite,
    lengths equal x3's; replayed == direct eager; x2 outputs differ from x3's;
  * no referrer of M1's original finish_residual outside row_exact (H's routing property).
X1 poison (X2Coverage.test_third_part_poison): a second x2 model whose probe overwrites part 2 of every
cached activation (random codes, NaN scale) before the provider consumes it, and whose outer
finish_residual wrapper scrambles lane 2 of every XCHG 5 slice, gives bitwise the clean model's outputs
on every path; cached activations produced by both row paths (exact and SPLIT = 1 fallback) are
poisoned, eagerly and under capture; a control that poisons part 1 on a direct eager call changes the
output.
X2 (X2Pipeline): with split_deferred True vs False (every deferred call finished in-kernel by M1's
SPLIT = 1 row epilogue), encoder outputs and text / token IDs / timestamps are bitwise equal on dev_24
3 / 10 / 30 s; each XCHG 5 call's finished y and next-layer q / qs equal the in-kernel x2 reference on
the same q / qs; both deferred routes occur.

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.gpulock gpu1bit.int8_mma.test_exact_pipeline_x2
"""
from __future__ import annotations

import contextlib
import gc
import json
import os
import sys
import unittest

import torch

from gpu1bit.int8_mma import hybrid_exact, hybrid_x2, row_exact, row_exact_x2
from gpu1bit.shared import env

_LOCK = None
SITE_SKIP = ("gpu1bit/shared/fulldev.py", "gpu1bit/int8_mma/test_exact_pipeline_x2.py")
PHASES = ("s1-first", "s1-replay", "s1-replay-changed", "s2-first", "s3-first", "s4-first", "s3-replay-changed",
          "s5-overflow", "s1-eager", "s1-eager-changed", "s3-eager-changed")
OUT_KEYS = ("s1", "s1-replay", "s1-replay-changed", "s2", "s3", "s4", "s3-replay-changed", "s5", "s1-eager",
            "s1-eager-changed", "s3-eager-changed")
SIG_KEYS = ("phase", "capturing", "site", "caller", "N", "K", "conv", "activation", "norm", "input_activation",
            "x_shape", "x_stride", "defer", "no_defer", "cached", "route")
LF72 = ("layer_fusions4.py", 72)


def setUpModule():
    global _LOCK
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    hybrid_x2.parts()
    if not getattr(env.gpu_lock, "_held", False):
        _LOCK = env.gpu_lock("B:x2 pipeline tests")
        _LOCK.__enter__()


def tearDownModule():
    if _LOCK is not None:
        _LOCK.__exit__(None, None, None)


def adapter_class():
    """H's fulldev.PartCount, required to forward attribute writes to the provider (X1)."""
    from gpu1bit.shared.fulldev import PartCount

    class _Inner:
        defer_epilogue = False
    inner = _Inner()
    pc = PartCount(inner, 2)
    pc.defer_epilogue = True
    if inner.defer_epilogue is not True:
        raise AssertionError("gpu1bit.shared.fulldev.PartCount does not forward attribute writes (plans/x2-default.md X1)")
    return PartCount, "fulldev.PartCount"


def route_of(v) -> str:
    from inference.row_fusion import DeferredProjection
    if isinstance(v, (row_exact_x2.ExactDeferredProjectionX2, row_exact.ExactDeferredProjection)):
        return "xchg5"
    if isinstance(v, DeferredProjection):
        return f"in-kernel(split={v.split})"
    return "plain"


def caller_site() -> str:
    f = sys._getframe(2)
    while f is not None and f.f_code.co_filename.endswith(SITE_SKIP):
        f = f.f_back
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}" if f is not None else "?"


class Probe:
    """Transparent provider wrapper: records each call as the provider receives it, forwards writes,
    and (poison=True) overwrites part 2 of the cached activation the call is about to consume."""
    _OWN = ("inner", "calls", "phase", "names", "poison", "poison_part", "garbage", "tags", "poisoned")

    def __init__(self, inner):
        for k, v in dict(inner=inner, calls=[], phase=None, names={}, poison=False, poison_part=2, garbage=None,
                         tags={}, poisoned=[]).items():
            object.__setattr__(self, k, v)

    def __setattr__(self, name, value):
        if name in Probe._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        inner = self.inner
        xx = x.transpose(1, 2) if conv else x
        if xx.dim() == 2:
            xx = xx.unsqueeze(0)
        key = (xx.data_ptr(), xx.shape[0] * xx.shape[1], xx.shape[2], id(norm)) if norm is not None else None
        cached = key is not None and key in inner.prequantized
        if cached and self.poison:
            q3, qs3 = inner.prequantized[key]
            p = self.poison_part
            numel = q3[p].numel()
            q3[p].copy_(self.garbage[:numel].view_as(q3[p]))
            qs3[p].fill_(float("nan"))
            self.poisoned.append({"phase": self.phase, "capturing": torch.cuda.is_current_stream_capturing(),
                                  "producer": self.tags.get(key, "?"), "part": p})
        rec = {"phase": self.phase, "capturing": torch.cuda.is_current_stream_capturing(),
               "site": self.names.get(id(mod), f"unnamed:{getattr(mod, 'out_features', None)}"), "caller": caller_site(),
               "N": getattr(mod, "out_features", None), "K": getattr(mod, "in_features", None), "conv": bool(conv),
               "components": components, "activation": activation,
               "norm": None if norm is None else self.names.get(id(norm), "unnamed-norm"),
               "input_activation": input_activation, "x_shape": list(x.shape), "x_stride": list(x.stride()),
               "defer": bool(inner.defer_epilogue), "no_defer": mod.packed_t.data_ptr() in inner.no_defer_weights,
               "cached": cached}
        out = inner(x, mod, conv, components, activation, norm, config, input_activation)
        rec["route"] = route_of(out)
        self.calls.append(rec)
        return out


@contextlib.contextmanager
def finish_wrapper(probe):
    """Outermost wrapper of inference.row_fusion.finish_residual (looked up at call time): tags each new
    cached activation with the row path that produced it and, for a poisoning probe, scrambles lane 2
    (packed bits 42..63 = high-word bits 10..31) of every x2 XCHG 5 slice before it is finished."""
    from inference import row_fusion
    below = row_fusion.finish_residual
    junk = {}

    def finish_residual(provider, value, residual, factor, norm=None, out_norm=None):
        exact = isinstance(value, row_exact_x2.ExactDeferredProjectionX2)
        if exact and probe.poison:
            hi = value.slices[:, 1]
            g = junk.get(hi.numel())
            if g is None:
                gen = torch.Generator(device="cuda").manual_seed(hi.numel())
                g = junk[hi.numel()] = (torch.randint(-2**31, 2**31 - 1, (hi.numel(),), device="cuda", dtype=torch.int64,
                                                      generator=gen).to(torch.int32) & ~0x3FF)
            hi.bitwise_xor_(g.view_as(hi))
        before = set(provider.prequantized)
        y = below(provider, value, residual, factor, norm, out_norm)
        for key in set(provider.prequantized) - before:
            probe.tags[key] = "exact" if exact else "split1"
        return y

    row_fusion.finish_residual = finish_residual
    try:
        yield
    finally:
        row_fusion.finish_residual = below


def shapes() -> list[dict]:
    """Five dev_24 utterances of distinct sample counts: the two shortest 3 s ones (M 38: their deferred calls
    take XCHG 5 at K 1024 and 4096), the two shortest 10 s ones (M 125: in-kernel at K 1024, XCHG 5 at K 4096),
    so both row paths occur under capture; then the shortest 30 s one (M 269: in-kernel), which overflows
    the 4-entry cache and runs eagerly."""
    from gpu1bit.shared import workloads
    by_target: dict = {}
    for r in sorted(workloads.load()["dev_24"], key=lambda r: (r["samples"], r["id"])):
        group = by_target.setdefault(r["target_seconds"], [])
        if r["samples"] not in {g["samples"] for g in group}:
            group.append(r)
    out = by_target[3][:2] + by_target[10][:2] + by_target[30][:1]
    if len(out) != 5 or len({r["samples"] for r in out}) != 5:
        raise AssertionError("expected five dev_24 utterances of distinct lengths")
    return out


def protocol(model, probe) -> dict:
    import evaluate as ev
    recs = shapes()
    cache = model.forward
    if type(cache).__name__ != "PipelineGraphCache":
        raise AssertionError(f"expected the deployed pipeline graph cache, got {type(cache).__name__}")
    cache.entries.clear()
    probe.names = {id(m): n for n, m in model.named_modules()}
    out = {}

    def run(phase, rec, eager=False, scale=1.0):
        audio, length = ev.audio_batch([rec])
        probe.phase = phase
        fn = cache.model_forward if eager else model.forward
        enc, enc_len = fn(input_signal=(audio * scale).cuda(), input_signal_length=length.cuda())
        torch.cuda.synchronize()
        probe.phase = None
        return enc.clone().cpu(), enc_len.clone().cpu()

    with torch.inference_mode(), ev.strict_fp32("cuda"), finish_wrapper(probe):
        out["s1"] = run("s1-first", recs[0])
        out["s1-replay"] = run("s1-replay", recs[0])
        out["s1-replay-changed"] = run("s1-replay-changed", recs[0], scale=0.5)
        for j in (1, 2, 3):
            out[f"s{j + 1}"] = run(f"s{j + 1}-first", recs[j])
        out["s3-replay-changed"] = run("s3-replay-changed", recs[2], scale=0.5)
        out["entries_after_4"] = len(cache.entries)
        out["s5"] = run("s5-overflow", recs[4])
        out["entries_after_5"] = len(cache.entries)
        out["s1-eager"] = run("s1-eager", recs[0], eager=True)
        out["s1-eager-changed"] = run("s1-eager-changed", recs[0], eager=True, scale=0.5)
        out["s3-eager-changed"] = run("s3-eager-changed", recs[2], eager=True, scale=0.5)
        if probe.poison:  # control: poison part 1 (which x2 reads) on a direct eager call
            probe.poison_part = 1
            out["control-part1"] = run("control", recs[0], eager=True)
            probe.poison_part = 2
    return out


@contextlib.contextmanager
def x3_model(probe_factory):
    """The deployed B-hybrid (x3) with a probe: hybrid_exact.exact_model with the probe inserted."""
    import evaluate as ev
    from gpu1bit.int8_mma.layer import Int8MmaLayer
    from gpu1bit.shared import provider as P
    model, table = P.load_binary_model(env.MB1_EXPORT, Int8MmaLayer)
    with ev.inference_settings(model):
        model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
        inner = hybrid_exact.HybridProjectionExact(table, P.module_names(model))
        probe = probe_factory(inner)
        with hybrid_exact.exact_hybrid_pipeline(model, probe):
            yield model, {"inner": inner, "probe": probe, "partcount": None}


@contextlib.contextmanager
def x2_model(probe_factory, adapter, split_deferred=True):
    made = {}

    def factory(inner):
        made["probe"] = probe_factory(inner)
        return made["probe"]
    with hybrid_x2.exact_model_x2(split_deferred=split_deferred, probe=factory, partcount=adapter) as (model, info):
        yield model, {**info, "probe": made["probe"]}


def run_stack(builder, poison=False) -> dict:
    with builder(Probe) as (model, info):
        probe = info["probe"]
        if info.get("partcount") is not None:
            info["partcount"].record = []
        no_defer = len(info["inner"].no_defer_weights)
        if poison:
            gen = torch.Generator(device="cuda").manual_seed(1234)
            probe.garbage = torch.randint(-128, 128, (2048 * 4096,), device="cuda", dtype=torch.int8, generator=gen)
            probe.poison = True
        out = protocol(model, probe)
        out["calls"] = list(probe.calls)
        out["poisoned"] = list(probe.poisoned)
        out["adapter_record"] = None if info.get("partcount") is None else list(info["partcount"].record)
        out["no_defer_weights"] = no_defer
        out["inner_type"] = type(info["inner"]).__name__
        out["defer_after"] = bool(info["inner"].defer_epilogue)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def signature(calls, phase):
    return [tuple(json.dumps(c[k]) for k in SIG_KEYS) for c in calls if c["phase"] == phase]


def offenders_of_m1_original() -> list[str]:
    from gpu1bit.tests.test_finish_residual_routing import m1_original, offenders
    return offenders(m1_original())


class X2Coverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter, cls.adapter_name = adapter_class()
        cls.x3 = run_stack(x3_model)
        cls.x2 = run_stack(lambda pf: x2_model(pf, cls.adapter))
        cls.routing_offenders = offenders_of_m1_original()

    def test_adapter_and_components(self):
        x2, x3 = self.x2, self.x3
        self.assertEqual(x2["inner_type"], "HybridProjectionX2")
        self.assertTrue(x2["calls"] and x3["calls"])
        self.assertEqual({c["components"] for c in x3["calls"]}, {3})
        self.assertEqual({c["components"] for c in x2["calls"]}, {2}, "every call reaches the provider with 2 parts")
        rec = x2["adapter_record"]
        self.assertEqual(len(rec), len(x2["calls"]), "every call passes the adapter")
        self.assertEqual({e for _, e in rec}, {2})
        lf72 = [i for i, c in enumerate(x2["calls"]) if c["caller"] == f"{LF72[0]}:{LF72[1]}"]
        self.assertTrue(lf72, "no call from layer_fusions4.py:72")
        self.assertTrue(all(x2["calls"][i]["site"].endswith("conv.pointwise_conv2") for i in lf72))
        self.assertEqual({rec[i] for i in lf72}, {(3, 2)}, "layer_fusions4.py:72 asks for 3 and gets 2")
        self.assertTrue(all(x3["calls"][i]["caller"] == f"{LF72[0]}:{LF72[1]}" for i in lf72))
        print(json.dumps({"adapter": self.adapter_name, "calls": len(x2["calls"]), "lf72_calls": len(lf72),
                          "requested": sorted({r for r, _ in rec})}), flush=True)

    def test_paths_routes_and_state_equal_x3(self):
        x2, x3 = self.x2, self.x3
        for phase in PHASES:
            with self.subTest(phase=phase):
                self.assertEqual(signature(x2["calls"], phase), signature(x3["calls"], phase))
        routes = {c["route"] for c in x2["calls"]}
        self.assertIn("xchg5", routes)
        self.assertIn("in-kernel(split=1)", routes)
        deferrable = [c for c in x2["calls"] if c["defer"] and c["N"] == 1024 and not c["no_defer"] and c["activation"] == ""]
        self.assertTrue(deferrable)
        self.assertTrue(all(c["route"] != "plain" for c in deferrable), "a deferrable call did not defer")
        self.assertTrue(any(c["defer"] for c in x2["calls"]), "the inner provider's defer flag was never set")
        self.assertEqual(x2["no_defer_weights"], x3["no_defer_weights"])
        self.assertGreater(x2["no_defer_weights"], 0)
        self.assertFalse(x2["defer_after"])
        first = [c for c in x2["calls"] if c["phase"] == "s1-first"]
        self.assertTrue(any(c["capturing"] for c in first) and any(not c["capturing"] for c in first))
        self.assertEqual(signature(x2["calls"], "s1-replay"), [])
        self.assertEqual(signature(x2["calls"], "s1-replay-changed"), [])
        self.assertEqual(signature(x2["calls"], "s3-replay-changed"), [])
        over = [c for c in x2["calls"] if c["phase"] == "s5-overflow"]
        self.assertTrue(over and not any(c["capturing"] for c in over))
        self.assertEqual((x2["entries_after_4"], x2["entries_after_5"]), (4, 4))
        self.assertEqual(self.routing_offenders, [], "a referrer of M1's original finish_residual outside row_exact")
        table = {}
        for c in x2["calls"]:
            table[c["route"]] = table.get(c["route"], 0) + 1
        print(json.dumps({"route_table": table}), flush=True)

    def test_outputs(self):
        x2, x3 = self.x2, self.x3
        for key in OUT_KEYS:
            enc, length = x2[key]
            self.assertTrue(bool(torch.isfinite(enc).all()), key)
            self.assertTrue(torch.equal(length, x3[key][1]), f"{key}: lengths differ from x3")
        self.assertTrue(torch.equal(x2["s1-replay"][0], x2["s1-eager"][0]))
        self.assertTrue(torch.equal(x2["s1-replay"][0], x2["s1"][0]))
        self.assertTrue(torch.equal(x2["s1-replay-changed"][0], x2["s1-eager-changed"][0]), "replay with changed input")
        self.assertTrue(torch.equal(x2["s3-replay-changed"][0], x2["s3-eager-changed"][0]), "replay with changed input (s3)")
        self.assertFalse(torch.equal(x2["s1-replay-changed"][0], x2["s1-replay"][0]))
        for key in ("s1-replay", "s5"):
            self.assertFalse(torch.equal(x2[key][0], x3[key][0]), f"{key}: x2 output equals x3's")
        t = int(x3["s1-eager"][1][0])
        rel = ((x2["s1-eager"][0][..., :t] - x3["s1-eager"][0][..., :t]).norm() / x3["s1-eager"][0][..., :t].norm()).item()
        record = {"adapter": self.adapter_name, "calls_by_phase": {p: len(signature(x2["calls"], p)) for p in PHASES},
                  "s1_rel_frobenius_vs_x3": rel}
        d = env.ARTIFACTS_ROOT / "B" / "x2"
        d.mkdir(parents=True, exist_ok=True)
        (d / "pipeline-coverage.json").write_text(json.dumps(record, indent=1) + "\n")
        print(json.dumps(record), flush=True)

    def test_third_part_poison(self):
        clean = self.x2
        dirty = run_stack(lambda pf: x2_model(pf, self.adapter), poison=True)
        for key in OUT_KEYS:
            with self.subTest(key=key):
                self.assertTrue(torch.equal(clean[key][0], dirty[key][0]), f"{key}: the poisoned third part changed the output")
                self.assertTrue(torch.equal(clean[key][1], dirty[key][1]))
        self.assertFalse(torch.equal(dirty["control-part1"][0], clean["s1-eager"][0]), "control: part 1 poison had no effect")
        poisoned = [p for p in dirty["poisoned"] if p["part"] == 2]
        producers = {(p["producer"], p["capturing"]) for p in poisoned}
        for want in (("exact", False), ("exact", True), ("split1", False), ("split1", True)):
            self.assertIn(want, producers, f"no poisoned consumption of a {want[0]} activation (capturing={want[1]})")
        phases = {p["phase"] for p in poisoned}
        self.assertTrue({"s1-first", "s3-first", "s5-overflow", "s1-eager", "s3-eager-changed"} <= phases, phases)
        counts = {}
        for p in poisoned:
            k = f"{p['phase']}|{p['producer']}|{'capture' if p['capturing'] else 'eager'}"
            counts[k] = counts.get(k, 0) + 1
        print(json.dumps({"poisoned_consumptions": counts}), flush=True)


class X2Pipeline(unittest.TestCase):
    """X2: deferred (both routes) vs the same stack with every deferred call finished in-kernel."""

    def test_split_vs_in_kernel_bitwise(self):
        import evaluate as ev
        from gpu1bit.shared import bench, workloads
        from inference.row_fusion import DeferredProjection
        adapter, adapter_name = adapter_class()
        picks = {}
        for rec in workloads.load()["dev_24"]:
            picks.setdefault(rec["target_seconds"], rec)
        records = [picks[t] for t in sorted(picks)]
        self.assertGreaterEqual(len(records), 3)
        results, checks = {}, {"exact_calls_checked": 0, "routes": {}}
        stash = {}
        original_dot = hybrid_x2.dot_slices_x2

        def dot_slices_x2(layer, q, qs, ldq, m, slices, config, order=0):
            original_dot(layer, q, qs, ldq, m, slices, config, order)
            stash[slices.data_ptr()] = (layer, q, qs)

        def per_call_check(inner_finish):
            def finish_residual(provider, value, residual, factor, norm=None, out_norm=None):
                y = inner_finish(provider, value, residual, factor, norm, out_norm)
                r = route_of(value)
                checks["routes"][r] = checks["routes"].get(r, 0) + 1
                if isinstance(value, row_exact_x2.ExactDeferredProjectionX2) and not torch.cuda.is_current_stream_capturing():
                    layer, q, qs = stash.pop(value.slices.data_ptr())
                    n, k, m = layer.out_features, layer.in_features, q.shape[1]
                    ref_v = torch.empty(value.shape, device="cuda")
                    layer.dot(q, qs, k, m, ref_v.view(m, n), config=hybrid_x2.configs.choose(m, n, k), parts=2)

                    class P:
                        row_warps, null, prequantized = provider.row_warps, provider.null, {}
                    ref = row_exact._ORIGINAL(P, DeferredProjection(ref_v, value.scale, value.bias, 1, value.shape),
                                              residual, factor, norm, out_norm)
                    self.assertTrue(torch.equal(y, ref), "per-call y")
                    if norm is not None:
                        key = (y.data_ptr(), m, n, id(norm))
                        q1, s1 = provider.prequantized[key]
                        q0, s0 = next(iter(P.prequantized.values()))
                        self.assertTrue(torch.equal(q0, q1) and torch.equal(s0, s1), "per-call next-layer q / qs")
                    checks["exact_calls_checked"] += 1
                return y
            return finish_residual

        from inference import row_fusion
        for split in (True, False):
            with torch.inference_mode(), ev.strict_fp32("cuda"), \
                    x2_model(lambda inner: inner, adapter, split_deferred=split) as (model, info):
                hybrid_x2.dot_slices_x2 = dot_slices_x2
                below = row_fusion.finish_residual
                row_fusion.finish_residual = per_call_check(below)
                try:
                    for rec in records:
                        audio, length = ev.audio_batch([rec])
                        feats, flen = model.preprocessor(input_signal=audio.cuda(), length=length.cuda())
                        enc, enc_len = model.encoder(audio_signal=feats, length=flen)
                        torch.cuda.synchronize()
                        text = bench.recognize(model, audio, length)
                        state = bench.decoded_state(model)
                        results[(split, rec["id"])] = (enc.cpu(), enc_len.cpu(), text, state)
                finally:
                    row_fusion.finish_residual = below
                    hybrid_x2.dot_slices_x2 = original_dot
                del model
            gc.collect()
            torch.cuda.empty_cache()
        for rec in records:
            a, b = results[(True, rec["id"])], results[(False, rec["id"])]
            with self.subTest(id=rec["id"]):
                self.assertTrue(torch.equal(a[1], b[1]))
                self.assertTrue(torch.equal(a[0], b[0]), f"{rec['id']}: max |d| {(a[0] - b[0]).abs().max().item():.3e}")
                self.assertEqual((a[2], a[3]), (b[2], b[3]), "text / tokens / timestamps")
        self.assertGreater(checks["exact_calls_checked"], 0)
        self.assertGreater(checks["routes"].get("xchg5", 0), 0)
        self.assertGreater(checks["routes"].get("in-kernel(split=1)", 0), 0)
        print(json.dumps({"adapter": adapter_name, **checks, "utterances": [r["id"] for r in records]}), flush=True)


if __name__ == "__main__":
    unittest.main()
