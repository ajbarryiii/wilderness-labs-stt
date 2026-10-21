"""Tests of the stage analysis (Codex code review c1 findings 2, 3, 4). NixOS:
    finetune/parakeet-binary/python cpu1bit/shared/tests/test_analysis.py
"""
from __future__ import annotations

import os
import random
import shutil
import statistics
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mac"))
import analysis as A  # noqa: E402
import design  # noqa: E402
import runner  # noqa: E402

CLIPS = [f"0{i}" for i in range(8)]


def windows(values: dict, lo_f=0.99, hi_f=1.01) -> list:
    return [{"clip": c, "energy_j_per_replay": {"lower": v * lo_f, "est": v, "upper": v * hi_f},
             "latency_s_per_replay": v / 20} for c, v in values.items()]


def make_stage(d: str, arms: list, replicates: int, value_fn, valid_fn=lambda b, n: True, fp="fp",
               bounds=(0.99, 1.01), stage="x"):
    blocks = design.design(arms, replicates, 7, stage, {"threads": 8, "components": 2})
    runner.write_new(os.path.join(d, "stage.json"), {"fingerprint_sha256": fp, "fingerprint": {}})
    runner.write_new(os.path.join(d, "design.json"), {"fingerprint_sha256": fp, "blocks": blocks})

    def run_one(b, stem):
        n = int(stem.rsplit(".att", 1)[1])
        vals = {c: value_fn(b, c) for c in CLIPS}
        ok = valid_fn(b, n)
        return {"analysis": {"valid": ok, "windows": windows(vals, *bounds),
                             "block_invalid_reasons": [] if ok else ["synthetic disturbance: foreign busy 0.9 cores"]}}
    return runner.execute_stage(blocks, d, fp, run_one, lambda x: None, max_attempts=2, passes=1)


class Loading(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_codex_repro_no_pairing_across_attempts(self):
        """Review c1 finding 3: baseline from a partial retry must never pair with the candidate of an
        invalid earlier attempt."""
        blocks = design.design(["B", "C"], 1, 7, "x", {"threads": 8, "components": 2})[:2]   # one sequence
        runner.write_new(os.path.join(self.d, "stage.json"), {"fingerprint_sha256": "fp", "fingerprint": {}})
        runner.write_new(os.path.join(self.d, "design.json"), {"fingerprint_sha256": "fp", "blocks": blocks})
        bB = next(b for b in blocks if b["arm"] == "B")
        bC = next(b for b in blocks if b["arm"] == "C")

        def put(b, n, valid, v):
            runner.write_new(os.path.join(self.d, f"{b['block_id']}.att{n}.json"),
                             {"block": b, "attempt": n, "fingerprint_sha256": "fp",
                              "analysis": {"valid": valid, "windows": windows({c: v for c in CLIPS})}})
        runner.write_new(os.path.join(self.d, "seq-r0-s0.att0.start.json"), {})
        put(bB, 0, False, 10.0)
        put(bC, 0, True, 4.0)
        runner.write_new(os.path.join(self.d, "seq-r0-s0.att0.end.json"), {"valid": False})
        runner.write_new(os.path.join(self.d, "seq-r0-s0.att1.start.json"), {})
        put(bB, 1, True, 10.0)                       # partial retry: candidate never ran
        with self.assertRaises(A.Incomplete):
            A.load_stage(self.d)
        runner.write_new(os.path.join(self.d, "seq-r0-s0.att2.start.json"), {})
        put(bB, 2, True, 10.0)
        put(bC, 2, True, 9.0)
        runner.write_new(os.path.join(self.d, "seq-r0-s0.att2.end.json"), {"valid": True})
        data = A.load_stage(self.d)
        self.assertEqual(data["attempts"], {"0/0": 2})
        self.assertAlmostEqual(A.statistic(A.paired(data, "B", "C")), -10.0)   # not the spurious -60%

    def test_incomplete_design_is_refused(self):
        """Review c1 finding 4: analysis refuses until every required sequence is valid."""
        st = make_stage(self.d, ["B", "C", "D"], 1, lambda b, c: 10.0, valid_fn=lambda b, n: b["sequence"] != 3)
        self.assertEqual(st["status"], "incomplete")
        with self.assertRaises(A.Incomplete):
            A.load_stage(self.d)

    def test_other_fingerprint_is_never_selected(self):
        make_stage(self.d, ["B", "C"], 1, lambda b, c: 10.0)
        p = os.path.join(self.d, "stage.json")
        os.chmod(p, 0o644)
        os.remove(p)
        runner.write_new(p, {"fingerprint_sha256": "other", "fingerprint": {}})
        with self.assertRaises(A.Incomplete):
            A.load_stage(self.d)

    def test_screen_statistic_and_pick(self):
        eff = {"B": 1.0, "C": 0.9, "D": 1.05}
        make_stage(self.d, ["B", "C", "D"], 1, lambda b, c: 10.0 * eff[b["arm"]] * (1 + int(c) / 100))
        out = A.screen(A.load_stage(self.d), "B", ["C", "D"])
        self.assertAlmostEqual(out["statistic_pct_energy"]["C"], -10.0)
        self.assertAlmostEqual(out["statistic_pct_energy"]["D"], 5.0)
        self.assertEqual(out["confirmation_candidate"], "C")


class Bootstrap(unittest.TestCase):
    def test_matches_an_independent_loop_implementation(self):
        rng = np.random.default_rng(3)
        D = rng.normal(0, 1, (6, 2, 8)) + rng.normal(0, 2, (1, 1, 8))
        lo, hi, s = A.bootstrap(D, draws=500, seed=11)
        r2 = np.random.default_rng(11)
        reps = r2.integers(0, 6, size=(500, 6))
        clips = r2.integers(0, 8, size=(500, 8))
        for b in range(500):
            clip_means = [statistics.mean(statistics.mean(D[r, :, c]) for r in reps[b]) for c in clips[b]]
            self.assertAlmostEqual(s[b], statistics.median(clip_means), places=10)
        self.assertEqual(A.bootstrap(D, draws=500, seed=11)[:2], (lo, hi))   # deterministic

    def test_interval_is_the_95_percent_percentile_interval(self):
        """Plan review p8 major 2: the frozen interval is the 2.5th and 97.5th percentiles of the draws."""
        rng = np.random.default_rng(5)
        D = rng.normal(-2, 1, (6, 2, 8)) + rng.normal(0, 1, (6, 1, 1))
        lo, hi, s = A.bootstrap(D, draws=2000, seed=13)
        self.assertEqual(len(s), 2000)
        self.assertEqual((lo, hi), tuple(float(v) for v in np.percentile(s, [2.5, 97.5])))
        self.assertLess(lo, float(np.percentile(s, 5)))                # not a 90% interval
        self.assertGreater(hi, float(np.percentile(s, 95)))

    def test_constant_data_and_seed(self):
        D = np.full((6, 2, 8), -3.0)
        lo, hi, _ = A.bootstrap(D)
        self.assertEqual((lo, hi), (-3.0, -3.0))
        self.assertEqual(A.BOOT_SEED, 20261020)
        self.assertEqual(A.BOOT_DRAWS, 10_000)

    def test_null_simulation_is_calibrated_for_moderate_spread(self):
        sig = {"sigma_replicate": 1.0, "sigma_clip": 2.0, "sigma_noise": 1.0}
        cov = A.simulate(sig, 6, 2, 8, sims=300, draws=2000)
        self.assertGreater(cov["coverage"], 0.80)
        self.assertLess(cov["false_improvement_rate"], 0.10)

    def test_variance_components_recover_magnitudes(self):
        rng = np.random.default_rng(5)
        Y = rng.normal(0, 3.0, (200, 1)) + rng.normal(0, 1.0, (1, 8)) + rng.normal(0, 0.5, (200, 8))
        v = A.variance_components(Y)
        self.assertAlmostEqual(v["sigma_replicate"], 3.0, delta=0.4)
        self.assertAlmostEqual(v["sigma_noise"], 0.5, delta=0.05)

    def test_verdict_fallback_and_rule(self):
        ok = {"coverage": 0.95, "false_improvement_rate": 0.02}
        self.assertTrue(A.verdict((-5.0, -1.0), ok)["improvement"])
        self.assertEqual(A.verdict((-5.0, 0.5), ok)["verdict"], "baseline wins")
        for bad in ({"coverage": 0.92, "false_improvement_rate": 0.02}, {"coverage": 0.95, "false_improvement_rate": 0.04}):
            v = A.verdict((-5.0, -1.0), bad)
            self.assertTrue(v["descriptive_only"])
            self.assertFalse(v["improvement"])


class Confirmation(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.scr = os.path.join(self.d, "screen")
        self.conf = os.path.join(self.d, "confirm")
        os.makedirs(self.scr)
        os.makedirs(self.conf)

    def tearDown(self):
        shutil.rmtree(self.d)

    def build(self, effect: float, bounds):
        rng = random.Random(9)
        clip_eff = {c: 1 + 0.1 * int(c) for c in CLIPS}
        rep_eff = {}

        def value(b, c):
            k = (b["stage"], b["replicate"], b["sequence"])
            rep_eff.setdefault(k, 1 + rng.gauss(0, 0.01))
            return 10.0 * clip_eff[c] * rep_eff[k] * (effect if b["arm"] == "C" else 1.0) * (1 + rng.gauss(0, 0.005))
        make_stage(self.scr, ["B", "C", "D"], 1, value, stage="screen")
        make_stage(self.conf, ["B", "C"], 6, value, bounds=bounds, stage="confirm")

    def test_m10_verdict_uses_candidate_upper_vs_baseline_lower(self):
        """Review c1 finding 2: -10% on the point estimate, but bounds of +-8% per window."""
        self.build(0.90, (0.92, 1.08))
        out = A.confirm(self.conf, self.scr, "B", "C", sims=200)
        self.assertLess(out["ci95_energy_est"][1], 0.0)
        self.assertGreater(out["m10"]["ci95"][1], 0.0)
        self.assertFalse(out["improvement"])
        self.assertEqual(out["replicates"], 6)
        self.assertEqual(out["bootstrap"], {"draws": 10_000, "seed": 20261020, "type": "crossed: replicates x shared clip vector"})

    def test_improvement_with_tight_bounds(self):
        self.build(0.90, (0.995, 1.005))
        out = A.confirm(self.conf, self.scr, "B", "C", sims=200)
        self.assertLess(out["m10"]["ci95"][1], 0.0)
        if not out["descriptive_only"]:
            self.assertTrue(out["improvement"])
        self.assertIn("coverage", out["coverage_check"])


if __name__ == "__main__":
    unittest.main()
