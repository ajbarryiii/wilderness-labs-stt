"""A3 numerics on real data: every one of MB1's 264 layers, on its actual input captured from MB1's
FP32 encoder for each of the 15 frozen dev fixture utterances (5 dev corpora x 3, 10, 30 s), in the
stride pattern NeMo passed it. Per layer and utterance: finite; relative Frobenius error and
elementwise guard each <= 1.5 x the worse of M1's R-bf16x3 and R-expanded INT8x3 errors on the same
weights (derived 2-bit layout) and inputs (precomputed by shared/fixtures.py m1-errors, hash-checked).

A full report per implementation is written to .../gpu1bit/H/numerics/<impl>.json.
GPU1BIT_FIXTURE_UTTERANCES="0-4" limits the utterances (default all 15).
"""
from __future__ import annotations

import gc
import json
import os
import time
import unittest

import torch

from gpu1bit.shared import api, env, fixtures, registry


def _selection() -> list[int]:
    spec = os.environ.get("GPU1BIT_FIXTURE_UTTERANCES", "0-14")
    a, b = (int(v) for v in spec.split("-"))
    return list(range(a, b + 1))


class RealNumerics:
    impl_name = ""
    impl = None

    def test_all_layers_all_fixture_utterances(self):
        if isinstance(self.impl, Exception):
            self.fail(str(self.impl))
        torch.backends.cuda.matmul.allow_tf32 = False
        thresholds = fixtures.m1_thresholds()
        idx = fixtures.index()
        weights = fixtures.mb1_layers()
        layers = {name: self.impl.prepare(w["codes"], w["scale"], w["bias"], w["K"], "cuda") for name, w in weights.items()}
        rows, failures = [], []
        started = time.time()
        with torch.inference_mode():
            for i in _selection():
                entry = idx["utterances"][i]
                inputs = fixtures.utterance_inputs(i)
                limits = thresholds[entry["file"]]
                for name, w in weights.items():
                    x = inputs[name]
                    y = api.call(layers[name], w["kind"], x)
                    got = api.errors(w["kind"], x, y, api.reference(w["kind"], x, w["codes"], w["scale"], w["bias"], w["K"]),
                                     w["scale"])
                    ok = api.within(got, limits[name])
                    rows.append({"utterance": entry["file"], "id": entry["id"], "layer": name, "M": entry["layers"][name]["M"],
                                 **got, "limits": limits[name], "ok": ok})
                    if not ok:
                        failures.append(rows[-1])
                del inputs
                gc.collect()
        out = env.artifacts("numerics") / f"{self.impl_name.replace(':', '_')}.json"
        worst = {m: max(r[m] / r["limits"][m] if r["limits"][m] > 0 else (0 if r[m] == 0 else float("inf")) for r in rows)
                 for m in ("relative_frobenius", "elementwise_guard")}
        out.write_text(json.dumps({"impl": self.impl_name, "source": str(registry.source_path(self.impl_name)),
                                   "source_sha256": env.sha256_file(registry.source_path(self.impl_name)),
                                   "evaluations": len(rows), "failures": len(failures),
                                   "worst_ratio_to_limit": worst, "seconds": time.time() - started,
                                   "rows": rows}) + "\n")
        self.assertEqual(len(rows), 264 * len(_selection()))
        self.assertFalse(failures, f"{len(failures)} layer evaluations over the A3 limits, first: "
                                   f"{json.dumps(failures[:3])[:1500]} (report {out})")


def _register():
    for name, impl in registry.discover().items():
        cls_name = "RealNumerics_" + "".join(c if c.isalnum() else "_" for c in name)
        globals()[cls_name] = type(cls_name, (RealNumerics, unittest.TestCase), {"impl_name": name, "impl": impl})


_register()

if __name__ == "__main__":
    unittest.main()
