"""Orchestration tests (plans/pilot.md "Arms", "Selection and reporting"): the launch commands,
the dry run, and the preregistered selection and reporting rules on hand-made results."""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import unittest
from unittest import mock

import binary_fixtures as fx

import binary_paths
import binary_sweep
import paths

SETS = list(paths.DEV_SETS)


def result(mean: float, offsets: tuple = (0.0, 0.01, -0.01, 0.02, -0.02)) -> dict:
    """A summary-like record whose per-set WERs average to `mean`."""
    return {"dev_mean_wer": mean, "dev_wer": {s: mean + o for s, o in zip(SETS, offsets)}}


REFERENCES = {"b0": {"dev_mean_wer": 0.0558, **{"dev_wer": {s: 0.0558 for s in SETS}}},
              "ternary_pilot": result(0.0685), "m1": result(0.0609)}


class CommandTest(unittest.TestCase):
    def test_arms_and_train_commands(self) -> None:
        self.assertEqual(list(binary_sweep.ARMS), ["A", "B5", "B2", "C"])
        expected = {"A": ("fp32", "binary", "5e-4", "pilot-binA-fp32-lr5e-4"),
                    "B5": ("m1", "binary", "5e-4", "pilot-binB-m1-lr5e-4"),
                    "B2": ("m1", "binary", "2e-4", "pilot-binB-m1-lr2e-4"),
                    "C": ("m1", "ternary", "2e-4", "pilot-terC-m1-lr2e-4")}
        for name, (init, quantizer, lr, run) in expected.items():
            arm = binary_sweep.ARMS[name]
            self.assertEqual((arm.init, arm.quantizer, arm.lr, arm.run_name), (init, quantizer, lr, run))
            cmd = binary_sweep.train_command(binary_sweep.Plan(), arm)
            self.assertEqual(cmd[:2], [str(binary_paths.HERE / "python"), str(binary_paths.HERE / "binary_train.py")])
            self.assertEqual(cmd[2:], ["--init", init, "--quantizer", quantizer, "--arm", "P2", "--lr", lr,
                                       "--run-name", run, "--train", "pilot", "--max-steps", "12000",
                                       "--batch-seconds", "600", "--eval-every", "1200", "--select", "final",
                                       "--stream", "hub", "--grad-checkpointing"])

    def test_dry_run_prints_every_command_in_order_and_writes_nothing(self) -> None:
        root = fx.TEST_ROOT / "dry-run"
        shutil.rmtree(root, ignore_errors=True)
        runs, evals = root / "runs", root / "eval"
        out = io.StringIO()
        with mock.patch.object(binary_paths, "RUNS", runs), mock.patch.object(binary_paths, "EVAL", evals), \
                contextlib.redirect_stdout(out):
            binary_sweep.main(["--phase", "pilot", "--dry-run", "--allow-unconfined"])
        lines = [line for line in out.getvalue().splitlines() if "/python " in line]
        order = [("binary_eval.py", "G-B"), ("binary_eval.py", "Z-A"), ("binary_eval.py", "Z-B"),
                 ("binary_train.py", "pilot-binA-fp32-lr5e-4"), ("binary_train.py", "pilot-binB-m1-lr5e-4"),
                 ("binary_train.py", "pilot-binB-m1-lr2e-4"), ("binary_train.py", "pilot-terC-m1-lr2e-4")]
        self.assertEqual(len(lines), len(order), out.getvalue())
        for line, (script, word) in zip(lines, order):
            self.assertIn(script, line)
            self.assertIn(word, line.split())
        self.assertFalse(runs.exists() or evals.exists())
        shutil.rmtree(root, ignore_errors=True)


class SelectionTest(unittest.TestCase):
    def decide(self, a, b5, b2, c):
        return binary_sweep.decide({"A": result(a), "B5": result(b5), "B2": result(b2), "C": result(c)},
                                   REFERENCES)

    def test_b_is_the_better_m1_arm_and_a_clear_winner_wins(self) -> None:
        s = self.decide(0.0900, 0.0800, 0.0790, 0.0600)
        self.assertEqual(s["starting_point"]["B"], "B2")
        self.assertFalse(s["starting_point"]["tie"])
        self.assertEqual(s["starting_point"]["winner"], "B2")
        s = self.decide(0.0770, 0.0800, 0.0790, 0.0600)  # A below B by 0.20 points
        self.assertEqual(s["starting_point"]["winner"], "A")
        self.assertFalse(s["starting_point"]["tie"])

    def test_tie_within_0_15_points_prefers_a(self) -> None:
        s = self.decide(0.0800, 0.0790, 0.0799, 0.0600)  # A - B = 0.10 points
        self.assertTrue(s["starting_point"]["tie"])
        self.assertEqual(s["starting_point"]["winner"], "A")
        s = self.decide(0.0800, 0.0784, 0.0799, 0.0600)  # 0.16 points: B wins
        self.assertFalse(s["starting_point"]["tie"])
        self.assertEqual(s["starting_point"]["winner"], "B5")

    def test_c_is_never_a_candidate(self) -> None:
        s = self.decide(0.0900, 0.0850, 0.0860, 0.0500)
        self.assertNotIn(s["starting_point"]["winner"], ("C",))
        self.assertEqual(s["starting_point"]["winner"], "B5")

    def test_binary_cost_and_continuation_control(self) -> None:
        s = self.decide(0.0900, 0.0800, 0.0790, 0.0600)
        cost = s["binary_cost"]
        self.assertAlmostEqual(cost["A_minus_ternary_pilot"]["mean"], 0.0900 - 0.0685)
        self.assertAlmostEqual(cost["B2_minus_C"]["mean"], 0.0790 - 0.0600)
        self.assertAlmostEqual(cost["B_minus_M1"]["mean"], 0.0790 - 0.0609)
        for entry in cost.values():
            self.assertEqual(set(entry["per_set"]), set(SETS))
        self.assertAlmostEqual(cost["B2_minus_C"]["per_set"][SETS[1]], (0.0790 + 0.01) - (0.0600 + 0.01))
        control = s["continuation_control"]
        self.assertAlmostEqual(control["C_minus_M1"]["mean"], 0.0600 - 0.0609)
        self.assertEqual(control["reading"], "tie")  # 0.09 points
        self.assertEqual(self.decide(0.09, 0.08, 0.079, 0.0590)["continuation_control"]["reading"],
                         "M1 still improving")  # 0.19 points below
        self.assertEqual(self.decide(0.09, 0.08, 0.079, 0.0630)["continuation_control"]["reading"],
                         "worse than M1")

    def test_recommendation_threshold_and_stop_rule(self) -> None:
        s = self.decide(0.0830, 0.0900, 0.0900, 0.0600)  # winner A at 8.30% <= 6.85 + 1.5
        self.assertEqual(s["recommendation"]["proposal"], "binary main run")
        self.assertEqual(s["recommendation"]["start"], "A")
        s = self.decide(0.0900, 0.0860, 0.0870, 0.0600)  # winner B5 at 8.60%
        self.assertEqual(s["recommendation"]["proposal"], "mixed pilot")
        self.assertFalse(s["stop_rule"]["triggered"])
        s = self.decide(0.20, 0.30, 0.25, 0.0600)  # every candidate above 3 x 5.58%; C does not count
        self.assertTrue(s["stop_rule"]["triggered"])
        self.assertAlmostEqual(s["stop_rule"]["threshold_dev_mean_wer"], 3 * 0.0558)
        self.assertEqual(s["recommendation"]["proposal"], "stop")


MODE = "no_while_loops"
GRAPHS = {"teacher": MODE, "student": MODE, "rebuilt": MODE, "dev-subset": MODE, "full-dev": MODE}


def arm_summary(arm, **changes) -> dict:
    """The fields validate_binary reads from a finished arm's summary.json."""
    summary = {"binary": {"init": arm.init, "quantizer": arm.quantizer, "ramp_kind": arm.ramp_kind,
                          "code_balance": {"minus_one": 0.5, "zero": 0.0, "plus_one": 0.5},
                          "decoder_graphs": dict(GRAPHS),
                          "timing": {"train_wall_s": 100.0, "sessions": 1, "step": 12000}},
               "ramp_steps": 0 if arm.ramp_kind == "none" else 3000,
               "reconstruction": {"codes_exact": True, "scales_exact": True}}
    for path, value in changes.items():
        target, *keys = path.split(".")
        node = summary
        for key in [target, *keys][:-1]:
            node = node[key]
        if value is None:
            node.pop([target, *keys][-1], None)
        else:
            node[[target, *keys][-1]] = value
    return summary


class ValidationTest(unittest.TestCase):
    """Review round 1, findings 1-2: a run without an exact reconstruction record or decoded in
    another CUDA-graph mode is never reused or selected."""

    def problems(self, name: str, **changes) -> list[str]:
        arm = binary_sweep.ARMS[name]
        expected = binary_sweep.BinaryExpected(binary_sweep.Plan(), arm, None)
        return binary_sweep.validate_binary(arm_summary(arm, **changes), expected)

    def test_complete_records_pass(self) -> None:
        for name in binary_sweep.ARMS:
            self.assertEqual(self.problems(name), [], name)

    def test_inexact_or_missing_reconstruction_is_rejected(self) -> None:
        for name in ("A", "C"):
            for change in ({"reconstruction.codes_exact": False}, {"reconstruction.scales_exact": False},
                           {"reconstruction.scales_exact": None}, {"reconstruction": None}):
                with self.subTest(arm=name, change=change):
                    self.assertTrue(self.problems(name, **change))

    def test_other_decoder_modes_are_rejected(self) -> None:
        for change in ({"binary.decoder_graphs.teacher": "full_graph"}, {"binary.decoder_graphs.rebuilt": None},
                       {"binary.decoder_graphs.full-dev": "no_graphs"}, {"binary.decoder_graphs": None}):
            with self.subTest(change=change):
                self.assertTrue(self.problems("B5", **change))

    def test_zero_step_points_need_exact_reconstruction_and_the_decoder_mode(self) -> None:
        sources = {"x.py": "0" * 64}
        base = {"point": "Z-B", "limit": None, "dev_wer": {s: 0.5 for s in SETS}, "source_hashes": sources,
                "reconstruction": {"codes_exact": True, "scales_exact": True},
                "decoder_graphs": {"qat": MODE, "rebuilt": MODE, "full-dev": MODE}}
        self.assertEqual(binary_sweep.validate_point("Z-B", base, sources), [])
        for broken in ({**base, "reconstruction": {"codes_exact": False, "scales_exact": True}},
                       {k: v for k, v in base.items() if k != "reconstruction"},
                       {**base, "decoder_graphs": {"qat": MODE, "rebuilt": "full_graph", "full-dev": MODE}},
                       {**base, "limit": 16}):
            with self.subTest(broken=broken):
                self.assertTrue(binary_sweep.validate_point("Z-B", broken, sources))
        gate = {**{k: v for k, v in base.items() if k != "reconstruction"}, "point": "G-B",
                "decoder_graphs": {"student": MODE, "full-dev": MODE}, "gate": {"pass": True}}
        self.assertEqual(binary_sweep.validate_point("G-B", gate, sources), [])
        self.assertTrue(binary_sweep.validate_point("G-B", {**gate, "decoder_graphs": {}}, sources))


class ReportTest(unittest.TestCase):
    def test_training_time_comes_from_the_seam_and_waits_from_step_accumulators(self) -> None:
        """Review round 1, finding 7: train.py's step_time_s covers only the steps since its timer
        was last reset (evaluation, checkpoint, resume), so the report takes the training wall time
        the seam records; stream waits are per-step means over the steps since the previous metrics
        line (train.py's accumulators survive checkpoints), so mean x step difference is exact."""
        run = fx.TEST_ROOT / "report-run"
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir(parents=True)
        lines = [{"step": 20, "loss": 1.0, "step_time_s": 1.0, "data_wait_s_per_step": 0.1},
                 {"step": 40, "loss": 1.0, "step_time_s": 1.0, "data_wait_s_per_step": 0.2},
                 {"step": 40, "eval": "dev-subset", "dev_subset_mean_wer": 0.3, "dev_subset_wer": {}},
                 # resumed from the step-47 checkpoint: step_time_s covers 13 steps, the wait 20
                 {"step": 60, "loss": 1.0, "step_time_s": 18.79, "data_wait_s_per_step": 0.3}]
        (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
        summary = {"run_name": "r", "lr": 5e-4, "dev_wer": {}, "dev_mean_wer": 0.1, "resumes": 1,
                   "started_utc": "2026-10-16T00:00:00Z", "finished_utc": "2026-10-16T01:00:00Z",
                   "export_bytes": {"file_bytes": 10, "packed_code_bytes": 5},
                   "binary": {"timing": {"train_wall_s": 1234.5, "sessions": 2, "step": 60}}}
        report = binary_sweep.arm_report(run, summary)
        self.assertEqual(report["train_wall_s"], 1234.5)
        self.assertEqual(report["train_sessions"], 2)
        self.assertAlmostEqual(report["stream_wait_s"], 0.1 * 20 + 0.2 * 20 + 0.3 * 20)
        self.assertEqual(report["wall_time_s"], 3600.0)
        self.assertNotIn("train_time_s_from_metrics", report)
        self.assertEqual([c["step"] for c in report["dev_subset_curve"]], [40])
        shutil.rmtree(run, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
