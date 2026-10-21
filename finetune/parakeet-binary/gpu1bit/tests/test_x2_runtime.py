"""GPU contracts of the x2 deployment path, in one process holding one lock (gpu1bit.tests.run):

    timeout 600 gpu1bit/shared/python -m gpu1bit.tests.run --label H:x2-runtime gpu1bit.tests.test_x2_runtime

X1 on the deployed loader (plans/x2-default.md X1, Z6): MB1 through gpu1bit.runtime.load_binary_packed under
shared/selected-x2.json, at parts 2 (PartCount(HybridProjectionX2, 2)) and at the parts-3 override
(HybridProjectionExact, unwrapped), each with a transparent recording probe between the adapter and the
provider (the loader's _provider_hook). One protocol on five dev_24 utterances of distinct lengths:
first call (eager warm-up and capture) of four shapes, replay, replay with changed input, the fifth
shape (cache overflow, eager), direct eager calls. Contracts:
  - attribute writes through the loader's provider reach the inner provider (defer_epilogue), and the
    x2 pipeline's state changes exactly as x3's (inner defer flag per call, no-defer membership);
  - per phase, the x2 call signature equals x3's: site, shapes, capture state, defer flag, no-defer
    membership and the route (plain / XCHG 5 deferred / in-kernel finish); replay makes no calls;
  - every deferrable call defers (defer flag set, N 1024, no activation, not a no-defer weight);
  - every x2 call reaches the provider with components 2 (R1), PartCount records requested 3 for the
    layer_fusions4.py second pointwise convolution and effective 2 everywhere;
  - graph entries: 4 after four shapes, still 4 after the fifth (eager).
Encoder length (testscore's structural check, X6/Z1): x2.testscore.expected_encoder_length equals the
encoder's enc_len on all 24 timing workloads, for MB1 (x2 loader) and M1 (runtime_m1 x2 loader).
P-kernel at x2 (the x2 benchmark's kernel step): bench.candidate_op of the int8_mma:x2 layer at
part_count("int8_mma:x2") = 2 runs every op at the dev clip M values with a finite output within 1e-2
relative Frobenius of the FP64 reference, and differs from int8_mma's x3 op (that layer called with 3
parts falls back to the x3 path; recorded, not asserted).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import unittest

import torch

from gpu1bit.shared import env

PHASES = ("s1-first", "s1-replay", "s1-replay-changed", "s2-first", "s3-first", "s4-first", "s5-overflow",
          "s1-eager", "s3-eager-changed")
SIG = ("phase", "capturing", "site", "N", "K", "conv", "activation", "x_shape", "defer", "no_defer", "route")
SKIP_FRAMES = ("gpu1bit/shared/fulldev.py", "gpu1bit/tests/test_x2_runtime.py")


def route_of(v) -> str:
    from inference.row_fusion import DeferredProjection
    from gpu1bit.int8_mma import row_exact, row_exact_x2
    if isinstance(v, (row_exact_x2.ExactDeferredProjectionX2, row_exact.ExactDeferredProjection)):
        return "xchg5"
    if isinstance(v, DeferredProjection):
        return f"in-kernel(split={v.split})"
    return "plain"


def caller() -> str:
    f = sys._getframe(2)
    while f is not None and f.f_code.co_filename.endswith(SKIP_FRAMES):
        f = f.f_back
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}" if f is not None else "?"


class Probe:
    """Transparent: forwards reads and writes to `inner`, records each call as the provider receives it."""
    OWN = ("inner", "calls", "phase", "names")

    def __init__(self, inner):
        for k, v in dict(inner=inner, calls=[], phase=None, names={}).items():
            object.__setattr__(self, k, v)

    def __setattr__(self, name, value):
        if name in Probe.OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, x, mod, conv=False, components=3, activation="", norm=None, config=None, input_activation=""):
        inner = self.inner
        rec = {"phase": self.phase, "capturing": torch.cuda.is_current_stream_capturing(),
               "site": self.names.get(id(mod), "?"), "caller": caller(), "N": getattr(mod, "out_features", None),
               "K": getattr(mod, "in_features", None),
               "conv": bool(conv), "activation": activation, "components": components, "x_shape": list(x.shape),
               "defer": bool(inner.defer_epilogue), "no_defer": mod.packed_t.data_ptr() in inner.no_defer_weights}
        out = inner(x, mod, conv, components, activation, norm, config, input_activation)
        rec["route"] = route_of(out)
        self.calls.append(rec)
        return out


def shapes() -> list[dict]:
    """Five dev_24 utterances of distinct sample counts: the two shortest of the 3 s and of the 10 s targets
    (both deferred routes occur under capture), then the shortest 30 s one (the fifth shape: overflow)."""
    from gpu1bit.shared import workloads
    by: dict = {}
    for r in sorted(workloads.load()["dev_24"], key=lambda r: (r["samples"], r["id"])):
        g = by.setdefault(r["target_seconds"], [])
        if r["samples"] not in {x["samples"] for x in g}:
            g.append(r)
    out = by[3][:2] + by[10][:2] + by[30][:1]
    assert len({r["samples"] for r in out}) == 5
    return out


def protocol(model, probe) -> dict:
    import evaluate as ev
    recs = shapes()
    cache = model.forward
    assert type(cache).__name__ == "PipelineGraphCache", type(cache).__name__
    cache.entries.clear()
    probe.names = {id(m): n for n, m in model.named_modules()}
    out = {}

    def run(phase, rec, eager=False, scale=1.0):
        audio, length = ev.audio_batch([rec])
        probe.phase = phase
        fn = cache.model_forward if eager else model.forward
        enc, _ = fn(input_signal=(audio * scale).cuda(), input_signal_length=length.cuda())
        torch.cuda.synchronize()
        probe.phase = None
        return enc.clone().cpu()

    with torch.inference_mode(), ev.strict_fp32("cuda"):
        out["s1"] = run("s1-first", recs[0])
        out["s1-replay"] = run("s1-replay", recs[0])
        run("s1-replay-changed", recs[0], scale=0.5)
        for j in (1, 2, 3):
            run(f"s{j + 1}-first", recs[j])
        out["entries_after_4"] = len(cache.entries)
        run("s5-overflow", recs[4])
        out["entries_after_5"] = len(cache.entries)
        out["s1-eager"] = run("s1-eager", recs[0], eager=True)
        run("s3-eager-changed", recs[2], eager=True, scale=0.5)
    return out


def _timing() -> list[dict]:
    from gpu1bit.shared import workloads
    doc = workloads.load()
    return list({r["id"]: r for r in doc["dev_clips"] + doc["dev_24"]}.values())


def encoder_lengths(model) -> list[dict]:
    """(actual enc_len, testscore.expected_encoder_length) on the 24 timing workloads (eager, no captures)."""
    import evaluate as ev
    from gpu1bit.x2 import testscore
    rows = []
    with torch.inference_mode(), ev.strict_fp32("cuda"):
        for rec in _timing():
            audio, length = ev.audio_batch([rec])
            _, enc_len = model.forward.model_forward(input_signal=audio.cuda(), input_signal_length=length.cuda())
            rows.append({"id": rec["id"], "samples": int(length[0]), "enc_len": int(enc_len[0]),
                         "expected": testscore.expected_encoder_length(model, int(length[0]))})
    return rows


class DeployedX1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nemo.utils import logging
        from gpu1bit import runtime
        from gpu1bit.shared import fulldev, seldoc
        logging.setLevel(logging.ERROR)
        cls.runs = {}
        for parts in (3, 2):
            probes = []

            def hook(inner):
                probes.append(Probe(inner))
                return probes[-1]
            model = runtime.load_binary_packed(env.MB1_EXPORT, selection=seldoc.CANDIDATE, parts=parts, _provider_hook=hook)
            try:
                rt = runtime.runtime_of(model)
                prov, probe = rt.provider, probes[0]
                info = {"parts": rt.parts, "provider": type(prov).__name__, "inner": type(probe.inner).__name__,
                        "kernel": rt.kernel}
                before = probe.inner.defer_epilogue                      # write forwarding through the loader's chain
                prov.defer_epilogue = not before
                info["write_reaches_inner"] = probe.inner.defer_epilogue is (not before)
                prov.defer_epilogue = before
                if isinstance(prov, fulldev.PartCount):
                    prov.record = []
                out = protocol(model, probe)
                out["calls"] = list(probe.calls)
                out["record"] = list(prov.record) if isinstance(prov, fulldev.PartCount) else None
                out["defer_after"] = bool(probe.inner.defer_epilogue)
                out["info"] = info
                if parts == 2:
                    out["encoder_lengths"] = encoder_lengths(model)
                cls.runs[parts] = out
            finally:
                runtime.close_binary_packed(model)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        x2, x3 = cls.runs[2], cls.runs[3]
        summary = {p: {"info": cls.runs[p]["info"], "calls_by_phase": {ph: sum(c["phase"] == ph for c in cls.runs[p]["calls"])
                                                                       for ph in PHASES},
                       "routes": {r: sum(c["route"] == r for c in cls.runs[p]["calls"])
                                  for r in sorted({c["route"] for c in cls.runs[p]["calls"]})}} for p in (2, 3)}
        (env.artifacts("x2") / "x1-deployed-loader.json").write_text(json.dumps(summary, indent=1) + "\n")

    @staticmethod
    def sig(calls, phase):
        return [tuple(json.dumps(c[k]) for k in SIG) for c in calls if c["phase"] == phase]

    def test_loader_chain(self):
        self.assertEqual(self.runs[2]["info"]["provider"], "PartCount")
        self.assertEqual(self.runs[2]["info"]["inner"], "HybridProjectionX2")
        self.assertEqual(self.runs[2]["info"]["kernel"], "int8_mma:x2")
        self.assertEqual(self.runs[3]["info"]["inner"], "HybridProjectionExact")
        for p in (2, 3):
            self.assertTrue(self.runs[p]["info"]["write_reaches_inner"], p)

    def test_signatures_and_routes_equal_x3(self):
        x2, x3 = self.runs[2]["calls"], self.runs[3]["calls"]
        for phase in PHASES:
            with self.subTest(phase=phase):
                self.assertEqual(self.sig(x2, phase), self.sig(x3, phase))
        for phase in ("s1-replay", "s1-replay-changed"):
            self.assertEqual(self.sig(x2, phase), [], f"{phase}: replay makes no provider calls")
        routes = {c["route"] for c in x2}
        self.assertTrue({"plain", "xchg5"} <= routes and any(r.startswith("in-kernel") for r in routes), routes)
        self.assertEqual(self.runs[2]["defer_after"], self.runs[3]["defer_after"])

    def test_every_deferrable_call_defers(self):
        for p in (2, 3):
            deferrable = [c for c in self.runs[p]["calls"]
                          if c["defer"] and c["N"] == 1024 and c["activation"] == "" and not c["no_defer"]]
            self.assertTrue(deferrable, p)
            self.assertEqual([c for c in deferrable if c["route"] == "plain"], [], p)

    def test_components_two(self):
        calls, record = self.runs[2]["calls"], self.runs[2]["record"]
        self.assertEqual({c["components"] for c in calls}, {2})
        self.assertEqual(len(record), len(calls))
        self.assertEqual({eff for _, eff in record}, {2})
        lf72 = [c for c in calls if c["caller"] == "layer_fusions4.py:72"]
        self.assertTrue(lf72, "the second pointwise convolution was called")
        requested = [req for (req, _), c in zip(record, calls) if c["caller"] == "layer_fusions4.py:72"]
        self.assertEqual(set(requested), {3}, "layer_fusions4.py:72 requests 3 parts; PartCount makes it 2")

    def test_graph_entries(self):
        for p in (2, 3):
            self.assertEqual((self.runs[p]["entries_after_4"], self.runs[p]["entries_after_5"]), (4, 4), p)

    def test_replay_and_eager_agree(self):
        for p in (2, 3):
            self.assertTrue(torch.equal(self.runs[p]["s1"], self.runs[p]["s1-replay"]), p)
            torch.testing.assert_close(self.runs[p]["s1-eager"], self.runs[p]["s1"], atol=1e-6, rtol=1e-5)

    def test_encoder_length_mb1(self):
        rows = self.runs[2]["encoder_lengths"]
        self.assertEqual(len(rows), 24)
        self.assertEqual([r for r in rows if r["enc_len"] != r["expected"]], [])


class EncoderLengthM1(unittest.TestCase):
    def test_encoder_length_m1(self):
        from gpu1bit import runtime_m1
        from gpu1bit.shared import seldoc
        model = runtime_m1.load_ternary_packed(selection=seldoc.CANDIDATE)
        try:
            self.assertEqual(runtime_m1.runtime_of(model).parts, 2)
            rows = encoder_lengths(model)
        finally:
            runtime_m1.close_ternary_packed(model)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        (env.artifacts("x2") / "encoder-lengths-m1.json").write_text(json.dumps(rows, indent=1) + "\n")
        self.assertEqual(len(rows), 24)
        self.assertEqual([r for r in rows if r["enc_len"] != r["expected"]], [])


class KernelOpX2(unittest.TestCase):
    def test_candidate_op_at_two_parts(self):
        from gpu1bit.actprec import parts
        from gpu1bit.shared import api, bench, registry
        parts.load_parts()
        x2, x3 = registry.get("int8_mma:x2"), registry.get("int8_mma")
        self.assertEqual((bench.part_count("int8_mma:x2"), bench.part_count("int8_mma")), (2, 3))
        rows = []
        with torch.inference_mode():
            torch.backends.cuda.matmul.allow_tf32 = False
            for op in bench.OPS:
                w = bench.op_weights(op)
                norm = bench._norm_module(bench.OPS[op][4])
                x = bench.op_input(op, 0)
                ref = bench.op_reference(op, x, w, norm)
                y2 = bench.candidate_op(x2, op, w, norm, components=2)[0](x)
                y2_as3 = bench.candidate_op(x2, op, w, norm, components=3)[0](x)
                y3 = bench.candidate_op(x3, op, w, norm, components=3)[0](x)
                torch.cuda.synchronize()
                rows.append({"op": op, "rel_x2": api.relative_error(y2, ref), "rel_x3": api.relative_error(y3, ref),
                             "x2_is_x3": bool(torch.equal(y2, y3)), "x2_layer_at_3_is_x3": bool(torch.equal(y2_as3, y3)),
                             "finite": bool(torch.isfinite(y2).all())})
        (env.artifacts("x2") / "kernel-op-x2.json").write_text(json.dumps(rows, indent=1) + "\n")
        for r in rows:
            with self.subTest(op=r["op"]):
                self.assertTrue(r["finite"])
                self.assertLess(r["rel_x2"], 1e-2)
                self.assertFalse(r["x2_is_x3"], "the x2 op must not be the x3 path")


if __name__ == "__main__":
    unittest.main()
