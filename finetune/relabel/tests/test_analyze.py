"""Analysis contracts (diagnostic plan Step 2 and "Decision") on hand-built records, plus an end-to-end
run on the real B0, M1 and MB1 records with stand-in teachers built from them (tmp paths only).

Run from finetune/relabel:  ./python -m unittest discover -s tests -v   (CPU only)
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import analyze  # noqa: E402
import evaluate  # noqa: E402
import relabel_paths  # noqa: E402

TMP = relabel_paths.ARTIFACTS / "tmp" / "tests"
QWEN, GRANITE, CANARY = relabel_paths.REAL_TEACHERS
VOTE = relabel_paths.VOTE


def rec(uid: str, ref: str, hyp: str, scored: bool = True) -> dict:
    s, d, i = evaluate.edit_counts(ref.split(), hyp.split())
    return {"id": uid, "ref": ref, "hyp": hyp, "ref_norm": ref, "hyp_norm": hyp, "S": s, "D": d, "I": i,
            "scored": scored, "duration": 1.0}


def write_model(root: Path, name: str, per_set: dict[str, list[dict]], teacher: bool = False,
                rate: float = 100.0) -> Path:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    for set_name, records in per_set.items():
        (out / f"{set_name}.json").write_text(json.dumps({"set": set_name, "limit": None, "config": {"limit": None},
                                                          "records": records}))
    if teacher:
        (out / "summary.json").write_text(json.dumps({
            "status": "scored", "complete": True, "config": {"limit": None}, "sets": {n: {} for n in per_set},
            "teacher": {"name": name}, "throughput": {"audio_hours_per_gpu_hour": rate}}))
    return out


# Hand-built example (one set; every set gets the same utterances unless overridden):
#   u1 ref "a b c d e f g h i j" (10 words), substitutions only
#   u2 ref "a b" (2 words), insertions in gap 1
#   u3 empty reference (scored = false): must not change any Step 2 number
REF1, REF2 = "a b c d e f g h i j", "a b"
HYPS = {
    "B0": ["a X c Y e f g h i j", "a u u b", "mm"],
    "M1": ["a X c d e f Z h i j", "a u b", ""],
    "MB1": ["a X c Y e f g h i W", "a u u u b", ""],
    QWEN: ["a b c Q e f g h i j", "a b", "mm hmm"],   # T1
    GRANITE: ["a X c Y e f g h i j", "a u u b", ""],  # B0's text
    CANARY: ["a X c Y e f g h i j", "a u u b", ""],   # B0's text
}
# Expected (points of 12 reference words; c and o fractions):
#   err B0 4, M1 3, MB1 6, T1 1; shared M1 2, MB1 4; fix(M1,T1) 2, fix(MB1,T1) 3; new(T1) 1
EXPECTED = {"wer": {"B0": 100 * 4 / 12, "M1": 25.0, "MB1": 50.0, QWEN: 100 / 12},
            "c": {"M1": 0.5, "MB1": 1.0}, "o": {"M1": 1 / 3, "MB1": 1 / 3},
            "delta": {QWEN: 25.0, GRANITE: 0.0, CANARY: 0.0, VOTE: 0.0},
            "F": {"M1": {QWEN: 100 * 2 / 12, GRANITE: 0.0, VOTE: 0.0}, "MB1": {QWEN: 25.0, GRANITE: 0.0}},
            "N": {QWEN: 100 / 12, GRANITE: 0.0, VOTE: 0.0}}


def build(root: Path, yodas_qwen: list[str] | None = None, drop: str | None = None) -> dict[str, Path]:
    if root.exists():
        shutil.rmtree(root)
    dirs = {}
    for model, hyps in HYPS.items():
        if model == drop:
            continue
        per_set = {}
        for set_name in relabel_paths.SETS:
            h = yodas_qwen if (set_name == relabel_paths.YODAS and model == QWEN and yodas_qwen) else hyps
            per_set[set_name] = [rec(f"{set_name}:u1", REF1, h[0]), rec(f"{set_name}:u2", REF2, h[1]),
                                 rec(f"{set_name}:u3", "", h[2], scored=False)]
        dirs[model] = write_model(root / ("students" if model in analyze.RECORDED else "teachers"), model,
                                  per_set, teacher=model not in analyze.RECORDED)
    return {"B0": dirs["B0"], "M1": dirs["M1"], "MB1": dirs["MB1"], "teachers": root / "teachers"}


class HandBuiltTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        relabel_paths.require_mount()
        cls.dirs = build(TMP / "hand")
        cls.report = analyze.analyze(cls.dirs, cls.dirs["teachers"], resamples=50)

    def test_quantities_per_set_and_h4(self) -> None:
        for where in [self.report["sets"]["ami_dev"], self.report["h4"], self.report["yodas"]]:
            for key, expected in EXPECTED.items():
                for k, v in expected.items():
                    if isinstance(v, dict):
                        for t, x in v.items():
                            self.assertAlmostEqual(where[key][k][t], x, places=9, msg=f"{key} {k} {t}")
                    else:
                        self.assertAlmostEqual(where[key][k], v, places=9, msg=f"{key} {k}")

    def test_unscored_records_are_excluded(self) -> None:
        s = self.report["sets"]["ami_dev"]
        self.assertEqual(s["utterances"], 3)
        self.assertEqual(s["utterances_scored"], 2)
        self.assertEqual(s["ref_words"], 12)
        self.assertAlmostEqual(s["insertion_rate"]["B0"], 100 * 2 / 12)  # u3's "mm" is not counted
        out = self.report["secondary"]["excluded_reference_output"]["ami_dev"]
        self.assertEqual(out["B0"]["nonempty_share"], 1.0)
        self.assertEqual(out["M1"]["nonempty_share"], 0.0)
        self.assertEqual(out[QWEN]["excluded"], 1)

    def test_vote_takes_the_majority_and_is_a_teacher(self) -> None:
        self.assertIn(VOTE, self.report["teachers"])
        self.assertEqual(self.report["vote_sources"]["ami_dev"], {QWEN: 0, GRANITE: 3, CANARY: 0})
        self.assertIn(VOTE, self.report["decision"]["teachers"])

    def test_f_never_exceeds_shared_errors(self) -> None:
        for s_name, s in self.report["sets"].items():
            for student in analyze.STUDENTS:
                shared_points = 100 * s["c"][student] * s["wer"]["B0"] / 100
                for t in self.report["teachers"]:
                    self.assertLessEqual(s["F"][student][t], shared_points + 1e-9)

    def test_bootstrap_is_deterministic(self) -> None:
        again = analyze.analyze(self.dirs, self.dirs["teachers"], resamples=50)
        self.assertEqual(again["h4"]["ci90"], self.report["h4"]["ci90"])


class YodasExcludedTest(unittest.TestCase):
    def test_yodas_changes_do_not_reach_h4_or_the_decision(self) -> None:
        self.assertNotIn(relabel_paths.YODAS, relabel_paths.H4)
        dirs = build(TMP / "yodas", yodas_qwen=["q q q q q q q q q q", "z z z", ""])
        report = analyze.analyze(dirs, dirs["teachers"], resamples=50)
        self.assertNotIn(relabel_paths.YODAS, report["h4"]["sets"])
        self.assertAlmostEqual(report["h4"]["delta"][QWEN], 25.0)
        self.assertLess(report["yodas"]["delta"][QWEN], 0)
        self.assertEqual(report["decision"]["teachers"][QWEN]["delta"], report["h4"]["delta"][QWEN])


class MissingTeacherTest(unittest.TestCase):
    def test_missing_teacher_is_not_scored_and_vote_not_computed(self) -> None:
        dirs = build(TMP / "missing", drop=CANARY)
        report = analyze.analyze(dirs, dirs["teachers"], resamples=20)
        self.assertNotIn(CANARY, report["teachers"])
        self.assertNotIn(VOTE, report["teachers"])
        self.assertIn(CANARY, report["not_scored"])
        self.assertIn(VOTE, report["not_scored"])

    def test_not_scorable_summary_is_reported(self) -> None:
        dirs = build(TMP / "gated")
        (dirs["teachers"] / CANARY / "summary.json").write_text(json.dumps(
            {"status": "not_scorable", "complete": False, "reason": "gated"}))
        report = analyze.analyze(dirs, dirs["teachers"], resamples=20)
        self.assertIn("gated", report["not_scored"][CANARY])


class TeacherStateTest(unittest.TestCase):
    def test_set_written_with_another_configuration_is_not_scored(self) -> None:
        dirs = build(TMP / "mixed")
        teacher = dirs["teachers"] / QWEN
        summary = json.loads((teacher / "summary.json").read_text())
        summary["config"] = {"limit": None, "batch_size": 32}
        summary["sets"] = {n: {} for n in relabel_paths.SETS}
        (teacher / "summary.json").write_text(json.dumps(summary))
        for name in relabel_paths.SETS:
            data = json.loads((teacher / f"{name}.json").read_text())
            data["config"] = {"limit": None, "batch_size": 16 if name == "ami_dev" else 32}
            (teacher / f"{name}.json").write_text(json.dumps(data))
        ok, reason, _ = analyze.teacher_state(teacher, relabel_paths.SETS)
        self.assertFalse(ok)
        self.assertIn("ami_dev", reason)
        data = json.loads((teacher / "ami_dev.json").read_text())
        data["config"]["batch_size"] = 32
        (teacher / "ami_dev.json").write_text(json.dumps(data))
        self.assertTrue(analyze.teacher_state(teacher, relabel_paths.SETS)[0])


class CheckTest(unittest.TestCase):
    def test_tampered_recorded_sdi_stops_the_analysis(self) -> None:
        dirs = build(TMP / "tamper")
        path = dirs["M1"] / "voxpopuli_dev.json"
        data = json.loads(path.read_text())
        data["records"][0]["S"] += 1
        path.write_text(json.dumps(data))
        with self.assertRaises(analyze.AnalysisError):
            analyze.analyze(dirs, dirs["teachers"], resamples=10)
        rc = analyze.main(["--teachers-root", str(dirs["teachers"]), "--b0", str(dirs["B0"]), "--m1", str(dirs["M1"]),
                           "--mb1", str(dirs["MB1"]), "--out", str(TMP / "tamper" / "report.json"), "--resamples", "10"])
        self.assertEqual(rc, 2)
        self.assertFalse((TMP / "tamper" / "report.json").exists())

    def test_production_report_needs_production_inputs(self) -> None:
        dirs = build(TMP / "guard")
        with self.assertRaises(SystemExit):
            analyze.main(["--teachers-root", str(dirs["teachers"]), "--out", str(relabel_paths.REPORT)])


def h4_rows(rows: dict[str, tuple[float, float, float, float]]):
    """rows: teacher -> (delta, ci lo, ci hi, F(MB1)); returns decide()'s h4 and h4_ci inputs."""
    h4 = {"delta": {t: r[0] for t, r in rows.items()}, "F": {"MB1": {t: r[3] for t, r in rows.items()}}}
    ci = {"delta": {t: [r[1], r[2]] for t, r in rows.items()}}
    return h4, ci


class DecisionTest(unittest.TestCase):
    def decide(self, rows, rates=None):
        h4, ci = h4_rows(rows)
        return analyze.decide(h4, ci, list(rows), rates or {t: 100.0 for t in rows})

    def test_gates(self) -> None:
        d = self.decide({QWEN: (0.30, 0.01, 0.6, 0.15),        # both thresholds exactly met: passes
                         GRANITE: (0.29, 0.10, 0.5, 0.40),     # delta below 0.30
                         CANARY: (0.50, -0.01, 1.0, 0.40),     # interval includes 0
                         VOTE: (0.50, 0.20, 0.8, 0.14)})       # F below 0.15
        self.assertEqual(d["passing"], [QWEN])
        self.assertTrue(d["write_labeling_plan"])
        self.assertEqual(d["chosen"], QWEN)

    def test_none_passes(self) -> None:
        d = self.decide({QWEN: (0.2, 0.1, 0.3, 0.5), GRANITE: (1.0, 0.5, 1.5, 0.1)})
        self.assertFalse(d["write_labeling_plan"])
        self.assertIsNone(d["chosen"])

    def test_largest_f_outside_near_tie(self) -> None:
        d = self.decide({QWEN: (1.0, 0.5, 1.5, 0.20), CANARY: (1.0, 0.5, 1.5, 0.26)})
        self.assertEqual(d["chosen"], CANARY)

    def test_near_tie_prefers_single_teacher_over_vote(self) -> None:
        d = self.decide({VOTE: (1.0, 0.5, 1.5, 0.30), CANARY: (1.0, 0.5, 1.5, 0.26)})
        self.assertEqual(d["chosen"], CANARY)

    def test_near_tie_prefers_better_private_conversational_wer(self) -> None:
        d = self.decide({QWEN: (1.0, 0.5, 1.5, 0.25), CANARY: (1.0, 0.5, 1.5, 0.30),
                         GRANITE: (1.0, 0.5, 1.5, 0.29)})
        self.assertEqual(d["chosen"], QWEN)  # 12.14 < 13.41 < 13.79, all within 0.05 of 0.30

    def test_near_tie_then_throughput(self) -> None:
        saved = dict(analyze.PRIVATE_CONVERSATIONAL_WER)
        try:
            analyze.PRIVATE_CONVERSATIONAL_WER.update({QWEN: 13.0, CANARY: 13.0})
            d = self.decide({QWEN: (1.0, 0.5, 1.5, 0.30), CANARY: (1.0, 0.5, 1.5, 0.28)},
                            rates={QWEN: 100.0, CANARY: 400.0})
            self.assertEqual(d["chosen"], CANARY)
        finally:
            analyze.PRIVATE_CONVERSATIONAL_WER.clear(); analyze.PRIVATE_CONVERSATIONAL_WER.update(saved)


@unittest.skipUnless(all(d.exists() for d in relabel_paths.STUDENT_RECORDS.values()), "real B0/M1/MB1 records not found under PARAKEET_TERNARY_ROOT / PARAKEET_BINARY_ROOT (private/local.env)")
class RealRecordsEndToEndTest(unittest.TestCase):
    """Stand-in teachers from the real records: qwen3 := M1, granite := MB1, canary := B0 (tmp paths only)."""

    @classmethod
    def setUpClass(cls) -> None:
        relabel_paths.require_mount()
        root = TMP / "real"
        if root.exists():
            shutil.rmtree(root)
        cls.stand_in = {QWEN: "M1", GRANITE: "MB1", CANARY: "B0"}
        for teacher, model in cls.stand_in.items():
            out = root / teacher
            out.mkdir(parents=True)
            for name in relabel_paths.SETS:
                data = json.loads((relabel_paths.STUDENT_RECORDS[model] / f"{name}.json").read_text())
                data["config"] = {"limit": None}
                (out / f"{name}.json").write_text(json.dumps(data))
            (out / "summary.json").write_text(json.dumps({
                "status": "scored", "complete": True, "config": {"limit": None}, "teacher": {"name": teacher},
                "sets": {n: {} for n in relabel_paths.SETS}, "throughput": {"audio_hours_per_gpu_hour": 1000.0}}))
        cls.out = root / "report.json"
        rc = analyze.main(["--teachers-root", str(root), "--out", str(cls.out), "--resamples", "30"])
        assert rc == 0
        cls.report = json.loads(cls.out.read_text())

    def test_report_shape(self) -> None:
        r = self.report
        self.assertEqual(r["teachers"], [QWEN, GRANITE, CANARY, VOTE])
        self.assertEqual(r["h4"]["sets"], relabel_paths.H4)
        self.assertTrue(r["check"]["sdi_reproduced"])
        self.assertEqual(set(r["sets"]), set(relabel_paths.SETS))
        self.assertIn("verdict", r["decision"])
        self.assertEqual(r["ami"], r["sets"][relabel_paths.AMI])
        self.assertEqual(r["yodas"], r["sets"][relabel_paths.YODAS])

    def test_wer_matches_recorded_counts(self) -> None:
        for name in relabel_paths.SETS:
            for model in analyze.RECORDED:
                rows = json.loads((relabel_paths.STUDENT_RECORDS[model] / f"{name}.json").read_text())["records"]
                rows = [x for x in rows if x["scored"]]
                wer = 100 * sum(x["S"] + x["D"] + x["I"] for x in rows) / sum(len(x["ref_norm"].split()) for x in rows)
                self.assertAlmostEqual(self.report["sets"][name]["wer"][model], wer, places=9)

    def test_identities(self) -> None:
        for where in list(self.report["sets"].values()) + [self.report["h4"]]:
            self.assertAlmostEqual(where["delta"][CANARY], 0.0)      # canary := B0
            self.assertAlmostEqual(where["N"][CANARY], 0.0)
            self.assertAlmostEqual(where["F"]["MB1"][CANARY], 0.0)
            self.assertAlmostEqual(where["F"]["M1"][QWEN], 0.0)      # a model never fixes its own errors
            self.assertAlmostEqual(where["delta"][QWEN], where["wer"]["B0"] - where["wer"]["M1"], places=9)
            for s in analyze.STUDENTS:
                self.assertGreaterEqual(where["c"][s], 0.0)
                self.assertLessEqual(where["c"][s], 1.0)

    def test_production_report_untouched(self) -> None:
        self.assertNotEqual(self.out.resolve(), relabel_paths.REPORT.resolve())


if __name__ == "__main__":
    unittest.main()
