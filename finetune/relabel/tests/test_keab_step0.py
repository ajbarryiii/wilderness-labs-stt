"""Contract tests (CPU): the stage 4 step 0 driver keab_step0.py (keab-prime plan "Step 0", "Scoring and gate").

- the preregistered rule per model (H4 cost <= 0.03, interval upper bound <= 0.06, no H4 set worse by > 0.10) and
  the choice of the largest scope passing on both models;
- the MB1 x1 check against RESULTS.md's x1 arm, within 0.02 points per set;
- token-sequence mismatch against the unquantized scoring, and the token IDs of a batched decode put back in
  record order;
- a variant's summary on synthetic records (H4 cost and its paired cluster-bootstrap interval, YODAS, domain WER
  and term recall);
- smoke outputs never inside the production root; the variants of each model.
"""
from __future__ import annotations

import unittest

import keab_step0 as k0
import train_b0prime


class Rule(unittest.TestCase):
    def test_each_threshold(self):
        ok = k0.rule(0.03, 0.06, {"a": 0.10, "b": -0.2})
        self.assertTrue(ok["pass"])
        self.assertEqual(ok["worst_set"], "a")
        self.assertFalse(k0.rule(0.031, 0.05, {"a": 0.0})["checks"]["h4_cost"])
        self.assertFalse(k0.rule(0.01, 0.061, {"a": 0.0})["checks"]["ci_upper"])
        self.assertFalse(k0.rule(0.01, 0.05, {"a": 0.101, "b": 0.0})["checks"]["worst_set"])
        self.assertFalse(k0.rule(0.01, 0.05, {"a": 0.101})["pass"])

    def test_largest_scope_passing_on_both_models(self):
        self.assertEqual(k0.choose({"MB1": {"S1": True, "S2": True}, "B0p": {"S1": True, "S2": True}}), "S2")
        self.assertEqual(k0.choose({"MB1": {"S1": True, "S2": False}, "B0p": {"S1": True, "S2": True}}), "S1")
        self.assertEqual(k0.choose({"MB1": {"S1": False, "S2": False}, "B0p": {"S1": True, "S2": True}}), "none")
        self.assertEqual(k0.choose({"MB1": {"S1": True}, "B0p": {"S1": True, "S2": True}}), "S1")

    def test_constants(self):
        self.assertEqual(k0.H4, tuple(train_b0prime.H4))
        self.assertEqual(k0.TERM_KIND, train_b0prime.TERM_KIND)
        self.assertEqual((k0.EVAL_BATCH, k0.EVAL_MAX_SECONDS), (train_b0prime.td.EVAL_BATCH,
                                                                train_b0prime.td.EVAL_MAX_SECONDS))
        self.assertEqual(k0.MODEL_VARIANTS["B0p"], ("none", "S1-int8", "S2-int8", "S1-palette", "S2-palette"))
        self.assertIn("S2-int8-w8a8", k0.MODEL_VARIANTS["MB1"])
        with self.assertRaises(ValueError):
            k0.check_variant("B0p", "S2-int8-w8a8")


class FullRun(unittest.TestCase):
    def test_rule_variants_on_all_sets_information_on_dev(self):
        plan = k0.full_plan()
        self.assertEqual(len(plan), 11)
        for model, variant, groups in plan:
            want = ("dev", "domain") if variant in ("none", "S1-int8", "S2-int8") else ("dev",)
            self.assertEqual(groups, want, (model, variant))
        self.assertEqual([v for m, v, _ in plan if m == "MB1"], list(k0.MODEL_VARIANTS["MB1"]))
        self.assertEqual(plan[0][:2], ("MB1", "none"))
        self.assertIn("dev sets only", k0.FULL_NOTE)


class Mb1Check(unittest.TestCase):
    def test_within_002_per_set(self):
        same = k0.check_against_recorded(dict(k0.RECORDED_X1))
        self.assertTrue(same["pass"])
        self.assertAlmostEqual(same["mean"], k0.RECORDED_X1_MEAN, places=12)
        moved = dict(k0.RECORDED_X1, ami_dev=k0.RECORDED_X1["ami_dev"] + 0.021)
        self.assertFalse(k0.check_against_recorded(moved)["pass"])
        edge = dict(k0.RECORDED_X1, yodas_dev=k0.RECORDED_X1["yodas_dev"] - 0.0199)
        self.assertTrue(k0.check_against_recorded(edge)["pass"])

    def test_recorded_values_are_results_md_x1_arm(self):
        self.assertEqual(round(k0.RECORDED_X1_MEAN, 4), 6.3333)
        self.assertAlmostEqual(sum(k0.RECORDED_X1.values()) / 5, k0.RECORDED_X1_MEAN, places=12)


class Tokens(unittest.TestCase):
    def test_mismatch(self):
        a = [{"id": "x", "tokens": [1, 2]}, {"id": "y", "tokens": [3]}]
        b = [{"id": "y", "tokens": [3]}, {"id": "x", "tokens": [1, 2, 4]}]
        self.assertEqual(k0.mismatch(a, b), {"utterances": 2, "mismatches": 1, "rate": 0.5})
        with self.assertRaises(ValueError):
            k0.mismatch(a, b[:1])

    def test_batched_tokens_back_in_record_order(self):
        import evaluate
        manifest = [{"id": f"u{i}", "duration": d} for i, d in enumerate([3.0, 9.0, 1.0, 9.0, 5.0])]
        records = [{"id": r["id"]} for r in manifest]
        batches = evaluate.make_batches(manifest, k0.EVAL_BATCH, k0.EVAL_MAX_SECONDS)
        captured = [[[100 + i] for i in b] for b in batches]
        k0.attach_tokens(records, manifest, captured)
        self.assertEqual([r["tokens"] for r in records], [[100], [101], [102], [103], [104]])
        with self.assertRaises(ValueError):
            k0.attach_tokens([{"id": r["id"]} for r in manifest], manifest, captured + [[[0]]])


def records(set_name: str, errors: list[int], words: int = 10, tokens=None) -> list[dict]:
    ids = {"librispeech_dev_clean": "{}-1-{}", "librispeech_dev_other": "{}-2-{}",
           "ami_dev": "ami:AMI_M{}_H00_{}", "voxpopuli_dev": "voxpopuli:2020010{}-0900-PLENARY-{}",
           "yodas_dev": "yodas:{}-{}", "med-dev-1": "m:{}-{}"}[set_name]
    out = []
    for i, e in enumerate(errors):
        out.append({"id": ids.format(i % 3 + 1, i), "ref_norm": " ".join(["w"] * words), "hyp_norm": "",
                    "S": e, "D": 0, "I": 0, "scored": True, "tokens": tokens[i] if tokens else [e]})
    return out


def result(recs, term=None):
    errors = sum(r["S"] for r in recs)
    words = sum(len(r["ref_norm"].split()) for r in recs)
    out = {"records": recs, "wer": {"wer": errors / words}, "utterances": len(recs)}
    if term:
        out["term_recall"] = {"occurrences": term[1], "reproduced": term[0], "recall": term[0] / term[1]}
    return out


class Summary(unittest.TestCase):
    def test_variant_against_none(self):
        n = 12
        base = {s: result(records(s, [1] * n)) for s in k0.DEV_SETS}
        quant = {s: result(records(s, [1] * n)) for s in k0.DEV_SETS}
        quant["ami_dev"] = result(records("ami_dev", [1] * (n - 1) + [2]))      # one more error in 120 words
        base["med-dev-1"] = result(records("med-dev-1", [0] * 4), term=(9, 10))
        quant["med-dev-1"] = result(records("med-dev-1", [0, 0, 0, 1]), term=(8, 10))
        s = k0.summarize_variant("MB1", "S1-int8", quant, base, {"dev-medical": ["med-dev-1"]})
        self.assertAlmostEqual(s["delta"]["ami_dev"], 100 / 120)
        self.assertAlmostEqual(s["h4_delta"], 100 / 120 / 4)
        self.assertAlmostEqual(s["rule"]["h4_cost"], 100 / 120 / 4)
        self.assertEqual(s["rule"]["worst_set"], "ami_dev")
        self.assertFalse(s["rule"]["pass"])
        lo, hi = s["h4_ci"]
        self.assertLessEqual(lo, s["rule"]["h4_cost"])
        self.assertGreaterEqual(hi, s["rule"]["h4_cost"])
        self.assertEqual(s["token_mismatch"]["ami_dev"]["mismatches"], 1)
        self.assertEqual(s["token_mismatch_dev_pooled"]["mismatches"], 1)
        self.assertAlmostEqual(s["domain"]["dev-medical"]["mean_delta"], 100 / 40)
        self.assertAlmostEqual(s["domain"]["dev-medical"]["pooled_term_recall_delta"], -0.1)

    def test_identical_scoring_costs_nothing(self):
        base = {s: result(records(s, [i % 3 for i in range(9)])) for s in k0.DEV_SETS}
        s = k0.summarize_variant("B0p", "S2-int8", base, base, {})
        self.assertEqual(s["h4_delta"], 0.0)
        self.assertEqual(s["h4_ci"], [0.0, 0.0])
        self.assertTrue(s["rule"]["pass"])
        self.assertEqual(s["token_mismatch_dev_pooled"]["rate"], 0.0)


class SecondBaseline(unittest.TestCase):
    def test_w8a8_is_also_compared_with_s2(self):
        self.assertEqual(k0.ALSO_AGAINST, {"S2-int8-w8a8": "S2-int8"})
        n = 12
        none = {s: result(records(s, [1] * n)) for s in k0.DEV_SETS}
        s2 = {s: result(records(s, [1] * n)) for s in k0.DEV_SETS}
        s2["ami_dev"] = result(records("ami_dev", [1] * (n - 1) + [2]))           # S2 costs one error on AMI
        w8a8 = {s: result(records(s, [1] * n)) for s in k0.DEV_SETS}
        w8a8["ami_dev"] = result(records("ami_dev", [1] * (n - 2) + [2, 2]))      # W8A8 one more
        out = k0.compare_variant("MB1", "S2-int8-w8a8", w8a8, {"none": none, "S2-int8": s2}, {})
        self.assertAlmostEqual(out["delta"]["ami_dev"], 200 / 120)
        self.assertAlmostEqual(out["against"]["S2-int8"]["delta"]["ami_dev"], 100 / 120)
        self.assertEqual(out["against"]["S2-int8"]["token_mismatch"]["ami_dev"]["mismatches"], 1)
        self.assertEqual(out["token_mismatch"]["ami_dev"]["mismatches"], 2)
        plain = k0.compare_variant("MB1", "S1-int8", s2, {"none": none, "S2-int8": s2}, {})
        self.assertNotIn("against", plain)


class PaletteReport(unittest.TestCase):
    def test_iteration_limit_and_non_converged_tensors_are_reported(self):
        import torch
        from training import eightbit as eb
        w = torch.randn(400, 400, generator=torch.Generator().manual_seed(3))
        _, _, info = eb.palette_table(w, max_iterations=2)
        self.assertEqual(info["iterations"], 2)
        self.assertFalse(info["converged"])
        summary = k0.palette_summary({"a.weight": {k: v for k, v in info.items() if k != "centroids"},
                                      "b.weight": {"entries": 5, "iterations": 0, "converged": True,
                                                   "max_abs_error": 0.0}})
        self.assertEqual(summary["not_converged"], ["a.weight"])
        self.assertEqual(summary["tensors"]["a.weight"]["iterations"], 2)
        self.assertEqual(summary["max_iterations"], eb.PALETTE_MAX_ITERATIONS)


class PaletteResume(unittest.TestCase):
    def test_palette_info_survives_a_stop_before_run_json(self):
        import shutil
        import tempfile
        from pathlib import Path
        from unittest import mock
        import relabel_paths
        tmp = Path(tempfile.mkdtemp(prefix="keab-pal-", dir=relabel_paths.ARTIFACTS / "tmp"))
        try:
            with mock.patch.object(k0, "out_root", lambda limit: tmp):
                d = k0.scores_dir("B0p", "S1-palette", 8)
                d.mkdir(parents=True)
                info = {"decoder.prediction.embed.weight": {"entries": 256, "iterations": 1000, "converged": False,
                                                            "max_abs_error": 0.1}}
                k0.write_json(d / "librispeech_dev_clean.json", {"records": [], "palette": info})
                self.assertFalse((d / "run.json").exists())
                got = k0.b0p_palette(8)
            self.assertEqual(got["S1-palette"]["not_converged"], ["decoder.prediction.embed.weight"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class LockWait(unittest.TestCase):
    def test_long_wait_logs_every_chunk_then_takes_the_lock(self):
        import contextlib
        from unittest import mock
        import score_teacher
        calls, held = [], []

        @contextlib.contextmanager
        def fake_lock(label, wait_s):
            calls.append(wait_s)
            if len(calls) < 3:
                raise score_teacher.LockBusy("held by synthetic")
            held.append(label)
            yield

        with mock.patch.object(score_teacher, "gpu_lock", fake_lock), mock.patch.object(k0, "log") as logged:
            with k0.waiting_lock("x", 3600.0, chunk=1.0):
                self.assertEqual(held, ["x"])
        self.assertEqual(calls, [1.0, 1.0, 1.0])
        self.assertEqual(logged.call_count, 2, "one line per busy chunk keeps the job from looking stalled")

    def test_gives_up_after_the_wait(self):
        import contextlib
        from unittest import mock
        import score_teacher

        @contextlib.contextmanager
        def busy(label, wait_s):
            raise score_teacher.LockBusy("held")
            yield

        with mock.patch.object(score_teacher, "gpu_lock", busy), mock.patch.object(k0, "log"):
            with self.assertRaises(score_teacher.LockBusy):
                with k0.waiting_lock("x", 0.0, chunk=1.0):
                    pass


class Paths(unittest.TestCase):
    def test_smoke_never_in_production(self):
        prod = k0.out_root(None)
        smoke = k0.out_root(20)
        self.assertFalse(smoke.is_relative_to(prod))
        self.assertTrue(k0.scores_dir("MB1", "none", 20).is_relative_to(smoke))
        self.assertTrue(k0.scores_dir("B0p", "S1-int8", None).is_relative_to(prod))
        with self.assertRaises(ValueError):
            k0.out_root(0)
        self.assertEqual(k0.view_dir("S2-int8").parent.name, "S2-int8")
        with self.assertRaises(ValueError):
            k0.view_dir("S2-int8-w8a8")

    def test_set_groups(self):
        sets = k0.expand_sets(["dev", "domain"])
        self.assertEqual([n for n, g in sets if g == "dev"], list(k0.DEV_SETS))
        self.assertEqual([n for n, g in sets if g != "dev"],
                         ["med-dev-1", "med-dev-2", "med-dev-3", "tech-dev-1", "tech-dev-2"])
        with self.assertRaises(ValueError):
            k0.expand_sets(["test"])


if __name__ == "__main__":
    unittest.main()
