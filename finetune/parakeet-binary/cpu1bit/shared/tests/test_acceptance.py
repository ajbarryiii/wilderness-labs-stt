"""Contract tests for rejection, selection and the acceptance audit (plans/cpu-benchmark-mac.md M14.7 as
amended by M15.3-M15.4). Written before the implementation, pending plan review p6. Changes from the M14
version, and why:
- imputation (M15.3, p5 major 3): every rejected sequence attempt is imputed at the prespecified +50%
  envelope (analysis.ENVELOPE_PCT), not at the observed range; the observed-range tests are replaced, and
  Codex's p5 example (-5% accepted, +45% rejected) is added: it must not be declared;
- simulation (M15.4, p5 major 4): the monotonicity test is replaced by the rejection-rate grid with an
  independent and an adversarial dependent rejection model; the pass criterion is the maximum
  false-improvement rate over the grid and coverage at r = 0;
- the verdict text states its conditions.
Unchanged: Fisher, the audit and its flags, the <= 2 rejection cap (analysis.MAX_REJECTED), Clopper-Pearson.
M16 changes (p6): the two ConfirmFlags fixtures use a -70% improvement (with -20% the +50% imputation makes
every replicate mean positive, so no conforming implementation could pass them; p6 major 4); added the
stress family (rare unfavourable outcomes, always rejected, zero-rejection realisations) as a reported
limitation, and the verdict text must state its conditions (A1, A2, A3, envelope, generating model, stress).
  analysis.stress_family(sig, R, S, C, sims=, draws=, seed=)
      -> [{"p", "unfavourable_pct", "favourable_pct", "false_improvement_rate", "zero_rejection_rate"}]

Contract (shared/analysis.py):
  analysis.acceptance(stage_dir) -> {"attempts", "overall", "by_arm", "by_position", "sequence_attempts",
                                     "sequence_rejected", "arm_p", "position_p", "flags", "rejected"}
  analysis.fisher_exact_2x2(a, b, c, d) -> two-sided p
  analysis.clopper_pearson_upper(k, n, conf=0.95)
  analysis.simulate(sig, R, S, C, sims=, draws=, seed=, reject_rate=0.0, model="independent"|"adversarial",
                    envelope_pct=analysis.ENVELOPE_PCT or None)   one grid point of the full stage-2 gate
  analysis.simulate_grid(sig, R, S, C, r_upper, sims=, draws=, seed=)
      -> {"grid": [{"r", "model", "false_improvement_rate", "coverage"}], "max_false_improvement",
          "coverage_r0", "pass"}
  analysis.confirm(...) -> {"m10" (imputed, the gate), "m10_without_imputation", "imputed_sequences",
                            "imputed_change_pct", "acceptance", "simulation", "descriptive_only",
                            "improvement", "verdict", "reasons", ...}

    finetune/parakeet-binary/python cpu1bit/shared/tests/test_acceptance.py
"""
from __future__ import annotations

import math
import os
import random
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mac"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mac", "tests"))
sys.path.insert(0, HERE)
import analysis as A  # noqa: E402
from test_analysis import CLIPS, make_stage  # noqa: E402


def fisher_reference(a, b, c, d) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] by enumeration (sum of tables no more likely)."""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2

    def prob(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)
    p0 = prob(a)
    return min(1.0, sum(prob(x) for x in range(max(0, c1 - r2), min(r1, c1) + 1) if prob(x) <= p0 * (1 + 1e-9)))


class Fisher(unittest.TestCase):
    def test_matches_enumeration(self):
        for t in ((24, 0, 20, 4), (10, 2, 3, 9), (24, 1, 24, 1), (0, 5, 5, 0), (12, 12, 12, 12)):
            self.assertAlmostEqual(A.fisher_exact_2x2(*t), fisher_reference(*t), places=12, msg=str(t))


class Audit(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_all_accepted_has_no_flags(self):
        make_stage(self.d, ["B", "C"], 6, lambda b, c: 10.0, stage="confirm")
        acc = A.acceptance(self.d)
        self.assertEqual(acc["overall"], 1.0)
        self.assertEqual((acc["flags"], acc["rejected"]), ([], []))
        self.assertEqual(set(acc["by_arm"]), {"B", "C"})

    def test_low_overall_acceptance_is_flagged(self):
        rng = random.Random(3)
        make_stage(self.d, ["B", "C"], 6, lambda b, c: 10.0, valid_fn=lambda b, n: rng.random() > 0.15, stage="confirm")
        acc = A.acceptance(self.d)
        self.assertLess(acc["overall"], A.ACCEPT_MIN)
        self.assertTrue(any("acceptance" in f for f in acc["flags"]), acc["flags"])

    def test_arm_dependent_acceptance_is_flagged(self):
        """Every candidate block's first attempt fails, baseline blocks never do."""
        make_stage(self.d, ["B", "C"], 6, lambda b, c: 10.0,
                   valid_fn=lambda b, n: not (b["arm"] == "C" and n == 0), stage="confirm")
        acc = A.acceptance(self.d)
        self.assertTrue(any("arm" in f for f in acc["flags"]), acc["flags"])

    def test_rejected_attempts_are_listed_with_their_reasons(self):
        first = lambda b, n: b["replicate"] == 2 and b["sequence"] == 1 and n == 0  # noqa: E731
        make_stage(self.d, ["B", "C"], 6, lambda b, c: 10.0, valid_fn=lambda b, n: not first(b, n), stage="confirm")
        acc = A.acceptance(self.d)
        self.assertEqual(len(acc["rejected"]), 1)
        r = acc["rejected"][0]
        self.assertEqual((r["replicate"], r["sequence"], r["attempt"]), (2, 1, 0))
        self.assertTrue(any("foreign busy" in x for x in r["reasons"]), r)

    def test_counts_every_attempt_including_invalid_ones(self):
        first = lambda b, n: b["replicate"] == 0 and b["sequence"] == 0 and n == 0  # noqa: E731
        make_stage(self.d, ["B", "C"], 6, lambda b, c: 10.0, valid_fn=lambda b, n: not first(b, n), stage="confirm")
        acc = A.acceptance(self.d)
        valid = sum(v["valid"] for v in acc["by_arm"].values())
        total = sum(v["attempts"] for v in acc["by_arm"].values())
        self.assertEqual(total, acc["attempts"])
        self.assertEqual(total, 24 + 2)          # 12 sequences x 2 blocks, plus the failed attempt's 2 blocks
        self.assertEqual(total - valid, 2)
        self.assertEqual((acc["sequence_attempts"], acc["sequence_rejected"]), (13, 1))


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


class ClopperPearson(unittest.TestCase):
    def test_upper_bound(self):
        self.assertAlmostEqual(A.clopper_pearson_upper(0, 12), 1 - 0.05 ** (1 / 12), places=9)
        for k, n in ((1, 13), (2, 14), (5, 40)):
            u = A.clopper_pearson_upper(k, n)
            self.assertAlmostEqual(binom_cdf(k, n, u), 0.05, places=6)   # P(X <= k | p = u) = alpha
        self.assertEqual(A.clopper_pearson_upper(12, 12), 1.0)


class RejectionSimulation(unittest.TestCase):
    SIG = {"sigma_replicate": 1.0, "sigma_clip": 2.0, "sigma_noise": 1.0}

    def test_no_rejection_equals_the_plain_simulation(self):
        a = A.simulate(self.SIG, 6, 2, 8, sims=200, draws=1000, seed=5)
        b = A.simulate(self.SIG, 6, 2, 8, sims=200, draws=1000, seed=5, reject_rate=0.0, model="adversarial")
        self.assertEqual((a["coverage"], a["false_improvement_rate"]), (b["coverage"], b["false_improvement_rate"]))

    def test_grid_covers_zero_to_the_upper_rate_for_both_models(self):
        g = A.simulate_grid(self.SIG, 6, 2, 8, r_upper=0.22, sims=100, draws=500, seed=3)
        rs = sorted({p["r"] for p in g["grid"]})
        self.assertEqual(rs[0], 0.0)
        self.assertAlmostEqual(rs[-1], 0.22)
        self.assertGreaterEqual(len(rs), 5)
        self.assertEqual({p["model"] for p in g["grid"]}, {"independent", "adversarial"})
        self.assertEqual(g["max_false_improvement"], max(p["false_improvement_rate"] for p in g["grid"]))
        plain = A.simulate(self.SIG, 6, 2, 8, sims=100, draws=500, seed=3)
        self.assertEqual(g["coverage_r0"], plain["coverage"])
        self.assertEqual(g["pass"], g["max_false_improvement"] <= A.FALSE_IMPROVEMENT_MAX
                         and g["coverage_r0"] >= A.COVERAGE_MIN)

    def test_adversarial_rejection_bites_without_the_envelope_and_not_with_it(self):
        """Rejecting the attempts least favourable to the candidate biases the stage statistic toward the
        candidate when the rejected attempts are simply re-measured (dropped); independent rejection does
        not; the +50% imputation reverses the bias and adds no false improvements.
        Corrected 2026-10-21 (coordinator-approved): the earlier assertion `dropped false-improvement rate
        > base + 0.02` is infeasible under the plan's own gate. At this noise level the crossed interval is
        very conservative (base false-improvement rate 0-0.0075, coverage >= 0.9975), at most 2 re-drawn
        attempts of 12 move the statistic by about 0.25 pp against about 1.3 pp of slack, and the
        2-rejection cap removes about 36% of the stages at r = 0.15. The mechanism is checked on the point
        statistic instead (measured: base -0.04, adversarial-dropped -0.27, independent-dropped -0.04,
        imputed +4.97 pp)."""
        noisy = {"sigma_replicate": 0.2, "sigma_clip": 0.5, "sigma_noise": 3.0}   # attempt-level noise dominates
        kw = dict(sims=400, draws=1000, seed=21)
        base = A.simulate(noisy, 6, 2, 8, **kw)
        dropped = A.simulate(noisy, 6, 2, 8, reject_rate=0.15, model="adversarial", envelope_pct=None, **kw)
        indep = A.simulate(noisy, 6, 2, 8, reject_rate=0.15, model="independent", envelope_pct=None, **kw)
        imputed = A.simulate(noisy, 6, 2, 8, reject_rate=0.15, model="adversarial", **kw)
        self.assertLess(dropped["mean_statistic"], base["mean_statistic"] - 0.1)
        self.assertAlmostEqual(indep["mean_statistic"], base["mean_statistic"], delta=0.1)
        self.assertGreater(imputed["mean_statistic"], base["mean_statistic"])
        self.assertLessEqual(imputed["false_improvement_rate"], base["false_improvement_rate"] + 0.01)

    def test_heavy_rejection_declares_nothing(self):
        """At a 50% attempt rejection rate nearly every simulated 12-sequence stage has > 2 rejections."""
        r = A.simulate(self.SIG, 6, 2, 8, sims=200, draws=1000, seed=7, reject_rate=0.5, model="adversarial")
        self.assertLessEqual(r["false_improvement_rate"], 0.01)

    def test_codex_c3_example_is_flagged(self):
        """Two equally frequent states with candidate changes +10% / -10% and block acceptance 55% / 92%:
        the true mean is 0 but accepted two-block sequences average about -4.73%. Rule (a) flags it."""
        rng = random.Random(9)
        acc, kept = [], []
        for _ in range(20000):
            plus = rng.random() < 0.5
            oks = [rng.random() < (0.55 if plus else 0.92) for _ in range(2)]
            acc += oks
            if all(oks):
                kept.append(10.0 if plus else -10.0)
        self.assertAlmostEqual(sum(kept) / len(kept), -4.73, delta=0.4)
        self.assertLess(sum(acc) / len(acc), A.ACCEPT_MIN)


class StressFamily(unittest.TestCase):
    """M16.3: rare unfavourable outcomes (+45%) that are always rejected; a known limitation, reported,
    not a pass criterion."""

    def test_cases_and_zero_rejection_realisations(self):
        sig = {"sigma_replicate": 0.5, "sigma_clip": 0.5, "sigma_noise": 0.5}
        cases = {round(c["p"], 2): c for c in A.stress_family(sig, 6, 2, 8, sims=300, draws=500, seed=4)}
        self.assertEqual(set(cases), {0.05, 0.10})
        for p, c in cases.items():
            self.assertEqual(c["unfavourable_pct"], 45.0)
            self.assertAlmostEqual(c["favourable_pct"], -p * 45.0 / (1 - p), places=6)   # mean change 0
            self.assertAlmostEqual(c["zero_rejection_rate"], (1 - p) ** 12, delta=0.08)
        self.assertGreater(cases[0.10]["false_improvement_rate"], 0.15)   # the documented limitation

    def test_grid_pass_ignores_the_stress_family(self):
        sig = {"sigma_replicate": 1.0, "sigma_clip": 2.0, "sigma_noise": 1.0}
        g = A.simulate_grid(sig, 6, 2, 8, r_upper=0.22, sims=100, draws=500, seed=3)
        self.assertNotIn("stress", {p["model"] for p in g["grid"]})


class Imputation(unittest.TestCase):
    """Every rejected confirmation sequence attempt is imputed at the +50% envelope (M15.3)."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.scr = os.path.join(self.d, "screen")
        self.conf = os.path.join(self.d, "confirm")
        os.makedirs(self.scr)
        os.makedirs(self.conf)

    def tearDown(self):
        shutil.rmtree(self.d)

    def build(self, effect: float, reject: int):
        """Paired changes are exactly `effect` in every accepted sequence (a host factor shared by both
        arms of a sequence spreads the energies by +-5%); `reject` replicates lose their first attempt."""
        clip_eff = {c: 1 + 0.1 * int(c) for c in CLIPS}
        rng = random.Random(11)
        host = {}

        def value(b, c):
            k = (b["stage"], b["replicate"], b["sequence"])
            host.setdefault(k, 1 + rng.uniform(-0.05, 0.05))
            return 10.0 * clip_eff[c] * host[k] * (effect if b["arm"] == "C" else 1.0)
        make_stage(self.scr, ["B", "C", "D"], 1, value, stage="screen")
        first = lambda b, n: b["replicate"] < reject and b["sequence"] == 0 and n == 0  # noqa: E731
        make_stage(self.conf, ["B", "C"], 6, value, bounds=(0.999, 1.001), stage="confirm",
                   valid_fn=lambda b, n: not first(b, n))
        return A.confirm(self.conf, self.scr, "B", "C", sims=100)

    def test_envelope_is_fifty_percent(self):
        self.assertEqual(A.ENVELOPE_PCT, 50.0)

    def test_no_rejection_no_imputation(self):
        out = self.build(0.99, reject=0)
        self.assertEqual(out["imputed_sequences"], 0)
        self.assertEqual(out["m10"]["ci95"], out["m10_without_imputation"]["ci95"])

    def test_imputation_removes_a_marginal_improvement(self):
        out = self.build(0.99, reject=1)
        self.assertEqual((out["imputed_sequences"], out["imputed_change_pct"]), (1, 50.0))
        self.assertLess(out["m10_without_imputation"]["ci95"][1], 0.0)
        self.assertGreaterEqual(out["m10"]["ci95"][1], 0.0)
        self.assertFalse(out["improvement"])

    def test_codex_p5_selection_example_is_not_declared(self):
        """p5 major 3: accepted attempts show -5%, the +45% attempts are rejected (here 2, within the
        cap). Observed-range imputation would change nothing; the +50% envelope must stop the claim."""
        out = self.build(0.95, reject=2)
        self.assertLess(out["m10_without_imputation"]["ci95"][1], 0.0)
        self.assertGreaterEqual(out["m10"]["ci95"][1], 0.0)
        self.assertFalse(out["improvement"])

    def test_a_large_improvement_survives_one_or_two_rejections(self):
        for k in (1, 2):
            if k > 1:                       # fresh stage directories per case (records are write-once)
                self.tearDown()
                self.setUp()
            out = self.build(0.70, reject=k)
            self.assertEqual(out["imputed_sequences"], k)
            self.assertLess(out["m10"]["ci95"][1], 0.0)
            self.assertGreaterEqual(out["m10"]["ci95"][1], out["m10_without_imputation"]["ci95"][1])
            self.assertTrue(out["improvement"], (k, out["reasons"]))
            self.assertFalse(out["descriptive_only"])
            self.assertTrue(out["acceptance"]["flags"])        # flagged, not gating
            self.assertIn("conditional", out["verdict"])
            self.assertIn("50%", out["verdict"])

    def test_zero_rejection_mixture_is_declared_with_its_conditions_quoted(self):
        """p6 major 3: a confirmation that happens to see none of the rare unfavourable attempts (all -5%,
        zero rejections) is declared; the verdict must quote the limitation and every condition."""
        out = self.build(0.95, reject=0)
        self.assertTrue(out["improvement"], out["reasons"])
        v = out["verdict"]
        for needle in ("A1", "A2", "A3", "50%", "generating model", "rare unfavourable"):
            self.assertIn(needle, v, needle)

    def test_three_rejections_are_descriptive(self):
        out = self.build(0.70, reject=A.MAX_REJECTED + 1)
        self.assertEqual(out["imputed_sequences"], 3)
        self.assertTrue(out["descriptive_only"])
        self.assertFalse(out["improvement"])
        self.assertTrue(any("rejected" in r for r in out["reasons"]), out["reasons"])


class ConfirmFlags(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.scr = os.path.join(self.d, "screen")
        self.conf = os.path.join(self.d, "confirm")
        os.makedirs(self.scr)
        os.makedirs(self.conf)

    def tearDown(self):
        shutil.rmtree(self.d)

    def run_confirm(self, valid_fn, effect=0.3):
        """Candidate at `effect` x baseline (default -70%: survives +50% imputation of half the sequences)."""
        clip_eff = {c: 1 + 0.1 * int(c) for c in CLIPS}
        value = lambda b, c: 10.0 * clip_eff[c] * (effect if b["arm"] == "C" else 1.0)  # noqa: E731
        make_stage(self.scr, ["B", "C", "D"], 1, value, stage="screen")
        make_stage(self.conf, ["B", "C"], 6, value, bounds=(0.999, 1.001), stage="confirm", valid_fn=valid_fn)
        return A.confirm(self.conf, self.scr, "B", "C", sims=100)

    def test_flags_do_not_gate(self):
        """Two rejected attempts, both caused by candidate blocks (arm-dependent): flagged, still inferential."""
        out = self.run_confirm(lambda b, n: not (b["arm"] == "C" and b["replicate"] < 2 and b["sequence"] == 0 and n == 0))
        self.assertTrue(out["acceptance"]["flags"])
        self.assertLess(out["m10"]["ci95"][1], 0.0)
        self.assertTrue(out["improvement"], out["reasons"])

    def test_many_rejections_are_descriptive_whatever_the_numbers(self):
        out = self.run_confirm(lambda b, n: not (b["arm"] == "C" and n == 0))    # every sequence's first attempt
        self.assertEqual(out["imputed_sequences"], 12)
        self.assertLess(out["m10"]["ci95"][1], 0.0)
        self.assertTrue(out["descriptive_only"])
        self.assertFalse(out["improvement"])
        self.assertTrue(any("rejected" in r for r in out["reasons"]), out["reasons"])


class HostCovariatesFromRunnerRecords(unittest.TestCase):
    """Codex code review c4 minor: the acceptance audit reads the host covariates from the schema the runner
    actually writes (the idle reference's p_ref_w), checked on records carrying runner.analyse_block output."""

    def test_idle_power_and_p_cluster_share_are_read(self):
        import runner
        import test_runner
        real = test_runner.Block().analyse()
        self.assertTrue(real["valid"])
        p_ref = real["idle"]["reference"]["p_ref_w"]
        d = tempfile.mkdtemp()
        try:
            import design
            blocks = design.design(["B", "C"], 1, 7, "x", {"threads": 8, "components": 2})
            runner.write_new(os.path.join(d, "stage.json"), {"fingerprint_sha256": "fp", "fingerprint": {}})
            runner.write_new(os.path.join(d, "design.json"), {"fingerprint_sha256": "fp", "blocks": blocks})
            rec = {"analysis": real, "quiet": {"ok": True, "load1": 0.7, "checks": {"thermal_nominal": True}},
                   "started_unix": 1.0, "ended_unix": 2.0}
            runner.execute_stage(blocks, d, "fp", lambda b, stem: dict(rec), lambda x: None, max_attempts=1, passes=1)
            host = A.acceptance(d)["host_state"]
            self.assertEqual(host["idle_power_w"]["valid"]["n"], len(blocks), host["idle_power_w"])
            self.assertAlmostEqual(host["idle_power_w"]["valid"]["mean"], p_ref)
            self.assertEqual(host["p_cluster_idle_share"]["valid"]["n"], len(blocks))
            self.assertEqual(host["load1"]["valid"]["n"], len(blocks))
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
