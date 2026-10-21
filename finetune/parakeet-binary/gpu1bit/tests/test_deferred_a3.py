"""A3 contract for DEFERRED providers (Codex code review c1, finding 1); shared/deferred_a3.py.

  - harness validity: the frozen B-hybrid provider (FP32 deferred slices) must FAIL Codex's
    counterexample (and must actually defer it), so the check is not vacuous;
  - every provider in DEFERRED_PROVIDERS of the module GPU1BIT_DEFERRED_MODULE (default
    gpu1bit.int8_mma.deferred, B's fixed path) must pass A3 on all 264 captured layers x 15 dev fixture
    utterances (deferred exactly where the pipeline defers) and on every split-boundary cancellation
    case, Codex's included, against 1.5 x the worse of M1's two references, both metrics, no floor.
    A provider registered with "components": 2 (plans/x2-default.md Z5) is qualified at x2 on the same
    real and synthetic case set, criterion: its deferred value equals its in-kernel value bitwise
    (GPU1BIT_DEFERRED_MODULE=gpu1bit.int8_mma.deferred_x2).
Run through dispatch so b01 is registered:
    timeout 600 gpu1bit/shared/python -m gpu1bit.shared.dispatch --patches gpu1bit.int8_mma.b01_run,<B's patch> \\
        tests --label H:deferred gpu1bit.tests.test_deferred_a3
"""
from __future__ import annotations

import importlib
import json
import os
import unittest

from gpu1bit.shared import deferred_a3, env

MODULE = os.environ.get("GPU1BIT_DEFERRED_MODULE", "gpu1bit.int8_mma.deferred")


def _providers() -> dict:
    try:
        return dict(importlib.import_module(MODULE).DEFERRED_PROVIDERS)
    except ModuleNotFoundError:
        return {}


class HarnessValidity(unittest.TestCase):
    def test_frozen_b_hybrid_fails_codex_counterexample(self):
        import torch
        build, _ = deferred_a3.provider_spec("B-hybrid@frozen")
        with torch.inference_mode():
            case = deferred_a3.CODEX
            w, x = deferred_a3.case_data(case, 1)
            y, deferred = deferred_a3.run_layer(build(w, "linear"), "linear", x, True)
            torch.cuda.synchronize()
        self.assertTrue(deferred, "the frozen provider must defer this N=1024 call")
        self.assertTrue(bool((y == 0).all()), "the frozen FP32 slices lose the 2^-12 residual (Codex c1)")
        report = deferred_a3.qualify("B-hybrid@frozen", real=False)
        codex = next(r for r in report["synthetic"]["rows"] if r["case"] == "codex_c1_counterexample")
        self.assertFalse(codex["ok"])
        self.assertEqual(codex["m1"]["r_expanded_int8x3"]["relative_frobenius"], 0.0)


class DeferredProviders(unittest.TestCase):
    def test_every_registered_provider_passes(self):
        providers = _providers()
        if not providers:
            self.skipTest(f"{MODULE} not available yet (B's exact deferred path)")
        for name, entry in providers.items():
            components = int(entry.get("components", 3))   # each provider at its registered part count (x2 plan Z5)
            with self.subTest(provider=name, components=components):
                report = deferred_a3.qualify(name, MODULE, components=components)
                out = env.artifacts("deferred") / f"test-{name.replace(':', '_')}.json"
                out.write_text(json.dumps(report) + "\n")
                self.assertEqual(report["real"]["evaluations"], 264 * 15)
                self.assertEqual(report["real"]["deferral_mismatches"], 0)
                self.assertEqual(report["real"]["failures"], 0, report["real"].get("worst_ratio"))
                self.assertEqual(report["synthetic"]["cases"], len(deferred_a3.synthetic_cases()))
                self.assertEqual(report["synthetic"]["not_deferred"], 0)
                self.assertEqual(report["synthetic"]["failures"], 0,
                                 [r["case"] for r in report["synthetic"]["rows"] if not r["ok"]][:5])
                if components == 2:                         # x2 criterion: bitwise deferred == in-kernel
                    self.assertEqual(report["real"]["inexact"], 0)
                    self.assertEqual(report["synthetic"]["inexact"], 0)


if __name__ == "__main__":
    unittest.main()
