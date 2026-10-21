"""Contract tests of the activation-precision reading (gpu1bit/actprec_report.py `reading`;
plans/activation-precision.md "Reading", R5 and R7), on synthetic summaries (CPU):
  - recognition-equivalent only if the mean is within 0.02 points (both directions), no set is worse by
    more than 0.10 points (a better set never counts against it), token mismatches are at most 0.5%
    pooled AND at most 1.0% in every set (R7);
  - otherwise a trade-off;
  - INVALID (not a trade-off) whenever R5 fails: a non-finite encoder output or a decoding error, any
    encoder length differing from x3 on full dev, or a failing 512-sample encoder check. This holds even
    if every WER criterion passes.
  - (ReportBinding, review c1 majors 1 and 3) report() reads only identity-bound, complete artifacts
    (summaries, comparisons with both identities, encoder checks with all three identities), refuses
    any mismatch, and leaves xN arms NOT CLASSIFIED when their x3 reference fails the R2 gate.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_actprec_report
"""
from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from pathlib import Path

from gpu1bit.shared import env, fulldev

sys.path.insert(0, str(env.GPU1BIT.parent))
import gpu1bit.actprec_report as R  # noqa: E402

SETS = fulldev.DEV_SETS


def _summary(wers, valid=True, empty=0):
    return {"dev_wer_percent": dict(zip(SETS, wers)), "dev_mean_wer_percent": sum(wers) / len(wers),
            "structurally_valid": valid, "empty_hypotheses": {s: empty for s in SETS}}


def _compare(mismatches, n=1000, lengths=None):
    sets = {s: {"utterances": n, "token_mismatches": m, "mismatch_rate": m / n, "text_mismatches": m,
                "encoder_length_differences": (lengths or {}).get(s, [])} for s, m in zip(SETS, mismatches)}
    total = sum(mismatches)
    return {"sets": sets, "pooled": {"utterances": n * len(SETS), "token_mismatches": total, "mismatch_rate": total / (n * len(SETS))}}


REF = _summary([2.0, 4.0, 10.0, 6.0, 8.0])
ENC_OK = {"not_finite": [], "length_differences": []}


class Reading(unittest.TestCase):
    def verdict(self, wers, mismatches, valid=True, lengths=None, enc=ENC_OK):
        return R.reading(_summary(wers, valid), REF, _compare(mismatches, lengths=lengths), enc)["verdict"]

    def test_equivalent_at_the_limits(self):
        self.assertEqual(self.verdict([2.02, 4.0, 10.0, 6.0, 8.0], [5, 5, 5, 5, 5]), "recognition-equivalent")
        self.assertEqual(self.verdict([2.1, 3.98, 9.98, 5.98, 7.98], [10, 0, 0, 0, 0]), "recognition-equivalent")

    def test_mean_limit_is_two_sided(self):
        self.assertEqual(self.verdict([2.11, 4.0, 10.0, 6.0, 8.0], [0] * 5), "trade-off")      # +0.022 mean
        self.assertEqual(self.verdict([1.89, 4.0, 10.0, 6.0, 8.0], [0] * 5), "trade-off")      # -0.022 mean

    def test_one_set_worse_by_more_than_0_10(self):
        self.assertEqual(self.verdict([2.11, 3.97, 9.97, 5.97, 8.0], [0] * 5), "trade-off")
        better = R.reading(_summary([1.8, 4.0, 10.0, 6.0, 8.15]), REF, _compare([0] * 5), ENC_OK)
        self.assertFalse(better["criteria"]["no_set_worse_than_0.10"])                  # 8.15 vs 8.0 is +0.15
        ok = R.reading(_summary([1.9, 4.0, 10.0, 6.0, 8.1]), REF, _compare([0] * 5), ENC_OK)
        self.assertTrue(ok["criteria"]["no_set_worse_than_0.10"])                      # a better set never counts

    def test_pooled_and_per_set_churn(self):
        self.assertEqual(self.verdict([2.0, 4.0, 10.0, 6.0, 8.0], [6, 6, 6, 6, 6]), "trade-off")   # 0.6% pooled
        r = R.reading(_summary([2.0, 4.0, 10.0, 6.0, 8.0]), REF, _compare([11, 0, 0, 0, 0]), ENC_OK)
        self.assertTrue(r["criteria"]["pooled_mismatch_le_0.5%"])                       # 0.22% pooled
        self.assertFalse(r["criteria"]["each_set_mismatch_le_1.0%"])                    # 1.1% in one set (R7)
        self.assertEqual(r["verdict"], "trade-off")
        self.assertAlmostEqual(r["token_mismatch_rate"][SETS[0]], 0.011)

    def test_structural_failures_invalidate(self):
        good = ([2.0, 4.0, 10.0, 6.0, 8.0], [0] * 5)
        self.assertEqual(self.verdict(*good, valid=False), "INVALID")
        self.assertEqual(self.verdict(*good, lengths={SETS[2]: ["ami:x"]}), "INVALID")
        self.assertEqual(self.verdict(*good, enc={"not_finite": ["a"], "length_differences": []}), "INVALID")
        self.assertEqual(self.verdict(*good, enc={"not_finite": [], "length_differences": ["a"]}), "INVALID")
        bad_wer = ([3.0, 5.0, 11.0, 7.0, 9.0], [500] * 5)
        self.assertEqual(self.verdict(*bad_wer, valid=False), "INVALID", "invalid, not a trade-off")

    def test_inputs_unchanged(self):
        a, c = _summary([2.0, 4.0, 10.0, 6.0, 8.0]), _compare([1] * 5)
        a0, c0, ref0 = copy.deepcopy(a), copy.deepcopy(c), copy.deepcopy(REF)
        R.reading(a, REF, c, ENC_OK)
        self.assertEqual((a, c, REF), (a0, c0, ref0))


class ReportBinding(unittest.TestCase):
    """Review c1 major 1 (R4) and major 3 (R2), on a synthetic run with real frozen identities:
      - report() reads only artifacts bound to the frozen, re-verified identities of the run's arms:
        summaries (arm name, full subset, complete, utterance counts, identity), comparison artifacts
        (run, arm and reference names, full subset, BOTH identities, utterance counts), and encoder
        checks (run, model, the three identities, 512 utterances, sample subset);
      - any mismatch is refused (no reading is produced);
      - a failed or invalid x3 reference leaves that model's xN arms NOT CLASSIFIED (R2), with the reason;
      - (review c2) so does a non-finite x3 encoder output on the 512 sample, and the encoder check
        itself refuses to start (no lock, no build) against a reference that fails the R2 gate."""

    def setUp(self):
        import shutil  # noqa: F401
        from gpu1bit.actprec import parts
        self.parts = parts
        self.run = f"test-actprec-report-{os.getpid()}"
        self.ctx = parts.installed()
        self.ctx.__enter__()
        self.ids = {arm: fulldev.freeze(self.run, arm, "full") for arms in R.ARMS.values() for arm in arms.values()}
        self.counts = {n: len(fulldev.subset_records(n, "full")) for n in SETS}
        for model, arms in R.ARMS.items():
            for n, arm in arms.items():
                self.write_summary(arm, [2.0, 4.0, 10.0, 6.0, 8.0] if n != 1 else [2.1, 4.1, 10.1, 6.1, 8.1])
            for n in (2, 1):
                self.write_compare(arms[n], arms[3], [0, 1, 0, 0, 0] if n == 2 else [200, 200, 200, 200, 200])
            self.write_encoder(model)

    def tearDown(self):
        import shutil
        self.ctx.__exit__(None, None, None)
        shutil.rmtree(fulldev.ROOT / self.run, ignore_errors=True)
        for p in R.out_dir().glob(f"*{self.run}*"):
            p.unlink()

    def write_summary(self, arm_name, wers, **overrides):
        arm = arm_name
        spec = fulldev.parse_arm(arm)
        s = {"run": self.run, "arm": arm, "subset": "full", "spec": spec, "identity": self.ids[arm], "complete": True,
             "dev_wer_percent": dict(zip(SETS, wers)), "dev_mean_wer_percent": sum(wers) / len(wers),
             "utterances": dict(self.counts), "empty_hypotheses": {n: 0 for n in SETS},
             "structurally_valid": True, "structural": {}}
        if spec["parts"] == 3:
            s["cross_check"] = {"pass": True}
        s.update(overrides)
        (fulldev.arm_dir(self.run, arm, "full") / "summary.json").write_text(json.dumps(s))

    def write_compare(self, arm_name, ref_name, mismatches, **overrides):
        arm, ref = arm_name, ref_name
        sets = {n: {"utterances": self.counts[n], "token_mismatches": m, "mismatch_rate": m / self.counts[n],
                    "text_mismatches": m, "encoder_length_differences": [], "mismatch_ids": []}
                for n, m in zip(SETS, mismatches)}
        total = sum(self.counts.values())
        c = {"run": self.run, "arm": arm, "ref": ref, "subset": "full", "identities": {"arm": self.ids[arm], "ref": self.ids[ref]},
             "sets": sets, "pooled": {"utterances": total, "token_mismatches": sum(mismatches), "mismatch_rate": sum(mismatches) / total}}
        c.update(overrides)
        d = fulldev.ROOT / self.run / "compare"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{arm}--vs--{ref}@full.json").write_text(json.dumps(c))

    def write_encoder(self, model_name, **overrides):
        model = model_name
        arms = R.ARMS[model]
        part = {"utterances": 512, "not_finite": [], "length_differences": [], "rel_frobenius_valid": {"p50": 1e-4, "max": 2e-4}}
        doc = {"run": self.run, "model": model, "subset": "sample", "arms": {str(n): a for n, a in arms.items()},
               "identities": {str(n): self.ids[a] for n, a in arms.items()}, "x3_not_finite": [],
               "summary": {"x2": dict(part), "x1": dict(part)}}
        doc.update(overrides)
        (R.out_dir() / f"encoder-check-{self.run}-{model}.json").write_text(json.dumps(doc))

    def report(self):
        import types
        return R.report(types.SimpleNamespace(run=self.run))

    def test_consistent_inputs_are_read(self):
        out = self.report()
        for model in R.ARMS:
            self.assertEqual(out["models"][model]["x2"]["verdict"], "recognition-equivalent")
            self.assertEqual(out["models"][model]["x1"]["verdict"], "trade-off")
            self.assertEqual(out["models"][model]["x2"]["identity"], self.ids[R.ARMS[model][2]])
            self.assertEqual(out["models"][model]["x3"]["identity"], self.ids[R.ARMS[model][3]])

    def test_summary_mismatches_are_refused(self):
        arm = "MB1-int8_mma-x2"
        for bad in ({"identity": "0" * 64}, {"arm": "MB1-int8_mma-x1"}, {"subset": "sample"}, {"complete": False},
                    {"utterances": {n: 1 for n in SETS}}):
            with self.subTest(bad=bad):
                self.write_summary(arm, [2.0, 4.0, 10.0, 6.0, 8.0], **bad)
                with self.assertRaises((ValueError, SystemExit)):
                    self.report()
        self.write_summary(arm, [2.0, 4.0, 10.0, 6.0, 8.0])
        self.report()

    def test_compare_mismatches_are_refused(self):
        arm, ref = "M1-x1", "M1-x3"
        for bad in ({"identities": {"arm": self.ids[arm], "ref": "0" * 64}}, {"identities": None}, {"ref": "M1-x2"},
                    {"subset": "sample"}, {"run": "elsewhere"},
                    {"pooled": {"utterances": 10, "token_mismatches": 0, "mismatch_rate": 0.0}}):
            with self.subTest(bad=bad):
                self.write_compare(arm, ref, [200] * 5, **bad)
                with self.assertRaises((ValueError, SystemExit)):
                    self.report()
        self.write_compare(arm, ref, [200] * 5)
        self.report()

    def test_encoder_check_mismatches_are_refused(self):
        for bad in ({"identities": {"3": "0" * 64, "2": "0" * 64, "1": "0" * 64}}, {"model": "M1"}, {"subset": "full"},
                    {"run": "elsewhere"}):
            with self.subTest(bad=bad):
                self.write_encoder("MB1", **bad)
                with self.assertRaises((ValueError, SystemExit)):
                    self.report()
        (R.out_dir() / f"encoder-check-{self.run}-MB1.json").unlink()
        with self.assertRaises((ValueError, SystemExit, FileNotFoundError)):
            self.report()
        self.write_encoder("MB1")
        self.report()

    def test_failed_reference_leaves_xn_unclassified(self):
        for bad in ({"cross_check": {"pass": False}}, {"structurally_valid": False}, {"identity": "0" * 64},
                    {"run": "ap-some-other-run"}):                                  # review c3: a foreign-run x3
            with self.subTest(bad=bad):
                self.write_summary("M1-x3", [2.0, 4.0, 10.0, 6.0, 8.0], **bad)
                out = self.report()
                for n in ("x2", "x1"):
                    self.assertEqual(out["models"]["M1"][n]["verdict"], "NOT CLASSIFIED")
                    self.assertIn("R2", out["models"]["M1"][n]["reason"])
                self.assertEqual(out["models"]["MB1"]["x2"]["verdict"], "recognition-equivalent")
        self.write_summary("M1-x3", [2.0, 4.0, 10.0, 6.0, 8.0])

    def test_nonfinite_x3_encoder_sample_withholds_classification(self):
        """Review c2 failure injection: x3's encoder output is not finite on one 512-sample utterance."""
        self.write_encoder("M1", x3_not_finite=["librispeech:1462-170142-0028"])
        out = self.report()
        for n in ("x2", "x1"):
            self.assertEqual(out["models"]["M1"][n]["verdict"], "NOT CLASSIFIED")
            self.assertIn("not finite", out["models"]["M1"][n]["reason"])
        self.assertFalse(out["models"]["M1"]["x3"]["valid_reference"])
        self.assertEqual(out["models"]["MB1"]["x2"]["verdict"], "recognition-equivalent")
        doc = json.loads((R.out_dir() / f"encoder-check-{self.run}-M1.json").read_text())
        del doc["x3_not_finite"]
        (R.out_dir() / f"encoder-check-{self.run}-M1.json").write_text(json.dumps(doc))
        with self.assertRaises(ValueError):                    # the x3 finiteness record is required
            self.report()

    def test_encoder_check_needs_a_valid_reference(self):
        """Review c2: the encoder check takes no lock and builds nothing against a failed reference."""
        import types

        class _Locked(Exception):
            pass
        requested = []

        def lock(label):
            requested.append(label)
            raise _Locked(label)
        saved = R.env.gpu_lock
        R.env.gpu_lock = lock
        try:
            self.write_summary("M1-x3", [2.0, 4.0, 10.0, 6.0, 8.0], cross_check={"pass": False})
            with self.assertRaisesRegex(SystemExit, "R2"):
                R.encoder_check(types.SimpleNamespace(run=self.run, model="M1"))
            self.assertEqual(requested, [])
            self.write_summary("M1-x3", [2.0, 4.0, 10.0, 6.0, 8.0])
            with self.assertRaises(_Locked):
                R.encoder_check(types.SimpleNamespace(run=self.run, model="M1"))
            self.assertEqual(len(requested), 1)
        finally:
            R.env.gpu_lock = saved


class TimingSummary(unittest.TestCase):
    """The activation-precision timing summary (plans/activation-precision.md Measurements 2):
      - the comparisons are x2 and x1 against x3 of the same model, and MB1-xN against M1-xN;
      - each is the median of per-utterance paired changes (a - b) / b, with the count of utterances where
        a is lower, for energy, latency and average power;
      - the summary refuses an artifact that is not bound to the roster's identities and references."""

    def test_pairs(self):
        self.assertEqual(R.TIMING_PAIRS, [("M1-x2", "M1-x3"), ("M1-x1", "M1-x3"),
                                          ("MB1-int8_mma-x2", "MB1-int8_mma-x3"), ("MB1-int8_mma-x1", "MB1-int8_mma-x3"),
                                          ("MB1-int8_mma-x3", "M1-x3"), ("MB1-int8_mma-x2", "M1-x2"),
                                          ("MB1-int8_mma-x1", "M1-x1")])

    def test_paired_changes(self):
        rows = [{"target_seconds": 3, "summary": {"a": {"joules_per_iteration": 0.9, "median_ms": 1.1, "average_watts": 300.0},
                                                  "b": {"joules_per_iteration": 1.0, "median_ms": 1.0, "average_watts": 310.0}}},
                {"target_seconds": 3, "summary": {"a": {"joules_per_iteration": 0.8, "median_ms": 0.9, "average_watts": 330.0},
                                                  "b": {"joules_per_iteration": 1.0, "median_ms": 1.0, "average_watts": 300.0}}},
                {"target_seconds": 10, "summary": {"a": {"joules_per_iteration": 1.1, "median_ms": 1.0, "average_watts": 300.0},
                                                   "b": {"joules_per_iteration": 1.0, "median_ms": 1.0, "average_watts": 300.0}}}]
        out = R.paired_changes(rows, "a", "b")
        self.assertAlmostEqual(out["energy"]["median_pct"], -10.0)
        self.assertEqual(out["energy"]["lower"], 2)
        self.assertAlmostEqual(out["latency"]["median_pct"], 0.0)
        self.assertEqual(out["latency"]["lower"], 1)
        self.assertEqual(out["average_power"]["lower"], 1)
        self.assertEqual(out["n"], 3)

    def test_unbound_artifacts_are_refused(self):
        import types
        steps = [{"name": "paired-full-0-3", "kind": "paired", "output": Path("/nonexistent")}]
        T = types.SimpleNamespace(identity_guard=lambda roster: {"identities": {}}, make_binding=lambda r, g: {},
                                  timing_steps=lambda run, roster: steps, _sidecar=lambda root, s: {},
                                  reusable=lambda s, rec, bind, roster: False)
        with self.assertRaisesRegex(ValueError, "not bound"):
            R.bound_timing_docs({"run": "x", "variants": []}, Path("/nonexistent"), T)
