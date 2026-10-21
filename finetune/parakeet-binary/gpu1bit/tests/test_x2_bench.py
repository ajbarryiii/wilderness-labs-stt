"""Contract tests of the x2 benchmark roster (gpu1bit/x2/bench_roster.py; plans/x2-default.md X5, Y4; CPU):
  - the roster b1 to b6 names model, export, stack, parts and the timing variant of every arm:
    b1 MB1 1-bit B-hybrid x2, b2 MB1 1-bit B-hybrid x3, b3 MB1 derived R-expanded x3, b4 M1 R-expanded x2,
    b5 M1 R-expanded x3, b6 M1 R-hybrid x3.
  - the comparison pairs are frozen: b1-b2 and b4-b5 (part count only); b1-b6, b4-b6, b1-b4 and b1-b3
    (change of deployed runtime).
  - labels: the report refuses an arm whose recorded model, or complete export identity, does not match
    its roster row (a swapped export, a wrong model, a missing record).
  - reading (Y4): every pair reports median paired energy and median latency separately, with counts,
    and no automatic consequence. The report states plainly where b4 uses more energy or has higher
    latency than b6.
  - the timing patch (gpu1bit/x2/bench_patch.py) serves exactly the six roster variants and delegates
    everything else; it refuses a roster variant unless the selection in use is selected-x2.json's content (Y1).

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_x2_bench
"""
from __future__ import annotations

import contextlib
import unittest

from gpu1bit.shared import env, fulldev
from gpu1bit.x2 import bench_roster as BR


def _rows(energy: dict[str, float], latency: dict[str, float]) -> list[dict]:
    return [{"index": i, "target_seconds": (3, 10, 30)[i // 8],
             "summary": {v: {"joules_per_iteration": energy[v] * (1 + 0.001 * (i % 3)), "median_ms": latency[v],
                             "average_watts": 300.0} for v in energy}} for i in range(24)]


class Roster(unittest.TestCase):
    def test_rows(self):
        r = {b["id"]: b for b in BR.BENCH_ROSTER}
        self.assertEqual(sorted(r), ["b1", "b2", "b3", "b4", "b5", "b6"])
        want = {"b1": ("MB1", "B-hybrid", 2, "MB1-B-hybrid-x2", "mb1"), "b2": ("MB1", "B-hybrid", 3, "MB1-B-hybrid-x3", "mb1"),
                "b3": ("MB1", "R-expanded", 3, "R-expanded", "derived"), "b4": ("M1", "R-expanded", 2, "M1-x2", "m1"),
                "b5": ("M1", "R-expanded", 3, "M1-x3", "m1"), "b6": ("M1", "R-hybrid (ptx_row_best)", 3, "M1-R-hybrid-x3", "m1")}
        for b, (model, stack, parts, variant, export) in want.items():
            self.assertEqual((r[b]["model"], r[b]["stack"], r[b]["parts"], r[b]["variant"], r[b]["export_key"]),
                             (model, stack, parts, variant, export), b)
        self.assertIn("a287e977", r["b4"]["export"])
        self.assertEqual(BR.export_dir("b3"), env.DERIVED_EXPORT)
        self.assertEqual(BR.export_dir("b1"), env.MB1_EXPORT)
        self.assertEqual(BR.export_dir("b6"), env.M1_EXPORT)

    def test_pairs_frozen(self):
        self.assertEqual([(a, b) for a, b, _ in BR.PAIRS],
                         [("b1", "b2"), ("b4", "b5"), ("b1", "b6"), ("b4", "b6"), ("b1", "b4"), ("b1", "b3")])


class Labels(unittest.TestCase):
    def infos(self):
        return {b["variant"]: {"roster_id": b["id"], "model": b["model"], "export": fulldev.export_identity(BR.export_dir(b["id"]))}
                for b in BR.BENCH_ROSTER}

    def test_consistent_labels(self):
        BR.verify_labels(self.infos())

    def test_refusals(self):
        for name, edit in {"swapped export": lambda i: i["M1-x2"].update(export=i["R-expanded"]["export"]),
                           "wrong model": lambda i: i["R-expanded"].update(model="M1"),
                           "missing arm": lambda i: i.pop("M1-R-hybrid-x3"),
                           "wrong roster id": lambda i: i["MB1-B-hybrid-x2"].update(roster_id="b2")}.items():
            with self.subTest(case=name):
                infos = self.infos()
                edit(infos)
                with self.assertRaises(ValueError):
                    BR.verify_labels(infos)


class Reading(unittest.TestCase):
    def test_pairs_report_energy_and_latency_separately(self):
        e = {"MB1-B-hybrid-x2": 0.90, "MB1-B-hybrid-x3": 1.00, "R-expanded": 1.10, "M1-x2": 1.05, "M1-x3": 1.15,
             "M1-R-hybrid-x3": 1.00}
        lat = {"MB1-B-hybrid-x2": 4.0, "MB1-B-hybrid-x3": 4.1, "R-expanded": 4.6, "M1-x2": 4.3, "M1-x3": 4.5,
               "M1-R-hybrid-x3": 4.2}
        out = BR.describe_pairs(_rows(e, lat))
        p = out["pairs"]["b4-b6"]
        self.assertGreater(p["energy"]["median_pct"], 0)
        self.assertGreater(p["latency"]["median_pct"], 0)
        self.assertTrue(any("b4" in s and "more energy" in s for s in out["statements"]))
        self.assertTrue(any("b4" in s and "higher latency" in s for s in out["statements"]))
        self.assertNotIn("decision", out)
        self.assertEqual(set(out["pairs"]), {"b1-b2", "b4-b5", "b1-b6", "b4-b6", "b1-b4", "b1-b3"})
        lat["M1-x2"] = 4.0
        e["M1-x2"] = 0.95
        out = BR.describe_pairs(_rows(e, lat))
        self.assertFalse(any("b4" in s for s in out["statements"]))


class Patch(unittest.TestCase):
    def test_serves_the_roster_and_delegates(self):
        from gpu1bit.x2 import bench_patch as BP
        self.assertEqual(set(BP.VARIANTS), {b["variant"] for b in BR.BENCH_ROSTER})
        seen = []

        @contextlib.contextmanager
        def previous(name, *, max_graphs=4):
            seen.append(name)
            yield "m", {"kind": "previous"}
        saved = BP._previous
        BP._previous = previous
        try:
            with BP.variant_model("int8_mma") as pair:
                self.assertEqual(pair, ("m", {"kind": "previous"}))
        finally:
            BP._previous = saved
        self.assertEqual(seen, ["int8_mma"])

    def test_roster_runs_only_under_the_candidate_selection(self):
        """Y1: the benchmark passes selected-x2.json; a roster variant under any other selection document
        (e.g. the x3 pointer before promotion) is refused before any model work."""
        import json
        import os
        import tempfile
        from pathlib import Path
        from gpu1bit.x2 import bench_patch as BP
        saved = os.environ.get("GPU1BIT_SELECTION")
        other = Path(tempfile.mkdtemp(dir=str(env.ARTIFACTS_ROOT / "H"))) / "x3-like.json"   # not the candidate's content
        other.write_text(json.dumps({"about": "an x3-like document", "kernel": "int8_mma", "runtime": "hybrid"}))
        os.environ["GPU1BIT_SELECTION"] = str(other)
        try:
            for variant in BP.VARIANTS:
                with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "selected-x2.json"):
                    with BP.variant_model(variant):
                        pass
        finally:
            if saved is None:
                os.environ.pop("GPU1BIT_SELECTION", None)
            else:
                os.environ["GPU1BIT_SELECTION"] = saved
            import shutil
            shutil.rmtree(other.parent, ignore_errors=True)
