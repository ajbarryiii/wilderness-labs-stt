"""encoder_probe_report contracts (encoder-probe-ctc plan "Analysis" v2.1 and "Readout") on synthetic records:
the population rule; WER, Delta_enc and s per set and on the H4 mean; s only where |WER(B0) - WER(T*)| >= 0.30;
the paired cluster bootstrap (training/bootstrap.py defaults, clusters from training/devsplit.cluster_of, the same
draw for every model, s per replicate); YODAS without an interval; the preregistered readout; implementation checks
4 and 5; the secondary quantities; the production-report guard.

Run from finetune/relabel:  ./python -m unittest discover -s tests -v   (CPU only)
"""
from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

import encoder_probe_report as rep  # noqa: E402
import evaluate  # noqa: E402
import relabel_paths  # noqa: E402
from training import bootstrap, devsplit  # noqa: E402

TMP = relabel_paths.ARTIFACTS / "tmp" / "tests" / "encoder-probe-report"
MODELS = ("B0", "T*", "char", "bpe")

# One id pattern per set, as the real manifests have them (devsplit.cluster_of reads the cluster from the id).
ID = {"librispeech_dev_clean": "librispeech:{c}-100-{k:04d}", "librispeech_dev_other": "librispeech:{c}-200-{k:04d}",
      "ami_dev": "ami:AMI_ES20{c}a_H00_FEE041_{k:07d}_0000001", "voxpopuli_dev": "voxpopuli:2009020{c}-0900-PLENARY-{k}-en_x",
      "yodas_dev": "yodas:en000_{c}_{k}"}
REF = "a b c d e f g h i j"  # 10 words


def rec(uid: str, ref: str, hyp: str, scored: bool = True) -> dict:
    s, d, i = evaluate.edit_counts(ref.split(), hyp.split())
    return {"id": uid, "ref": ref, "hyp": hyp, "ref_norm": ref, "hyp_norm": hyp, "S": s, "D": d, "I": i,
            "scored": scored, "duration": 1.0}


def hyp_with(errors: int, kind: str = "sub") -> str:
    words = REF.split()
    if kind == "sub":
        return " ".join(["x"] * errors + words[errors:])
    if kind == "del":
        return " ".join(words[errors:])
    return " ".join(words + ["y"] * errors)


def build(errors: dict[str, dict[str, list[int]]], clusters: int = 3, per_cluster: int = 2,
          unscored: dict[str, list[str]] | None = None) -> dict[str, dict[str, list[dict]]]:
    """{model: {set: records}}: errors[model][set] lists each scored utterance's substitutions, utterances k
    in cluster k // per_cluster; plus, per model, unscored (empty-reference) records with the given hypotheses."""
    out = {m: {} for m in MODELS}
    for name in relabel_paths.SETS:
        n = clusters * per_cluster
        for m in MODELS:
            errs = errors.get(m, {}).get(name, [0] * n)
            rows = [rec(ID[name].format(c=k // per_cluster + 1, k=k), REF, hyp_with(e)) for k, e in enumerate(errs)]
            for j, h in enumerate((unscored or {}).get(m, [])):
                rows.append(rec(ID[name].format(c=1, k=900 + j), "", h, scored=False))
            out[m][name] = sorted(rows, key=lambda r: r["id"])
    return out


def write(root: Path, models: dict[str, dict[str, list[dict]]], checks: dict | None = None,
          status: str = "scored", limit: int | None = None) -> dict[str, Path]:
    if root.exists():
        shutil.rmtree(root)
    dirs = {"B0": root / "b0", "T*": root / "tstar", "probe": root / "probe"}
    config = {"limit": limit, "probe": "encoder-probe-ctc"}
    for m, directory in (("B0", dirs["B0"]), ("T*", dirs["T*"]), ("char", dirs["probe"] / "char"),
                         ("bpe", dirs["probe"] / "bpe")):
        directory.mkdir(parents=True)
        for name, rows in models[m].items():
            body = {"set": name, "limit": limit, "config": config if m in ("char", "bpe") else {"limit": None},
                    "records": rows}
            (directory / f"{name}.json").write_text(json.dumps(body))
    (dirs["T*"] / "summary.json").write_text(json.dumps({
        "status": "scored", "complete": True, "config": {"limit": None}, "sets": {n: {} for n in relabel_paths.SETS},
        "teacher": {"name": "granite-speech-5.0-2b"}, "throughput": {"audio_hours_per_gpu_hour": 1.0}}))
    ok = {"passed": True, "hooks_passive": {"passed": True}, "pre_feedback": {"passed": True},
          "determinism": {"passed": True}}
    (dirs["probe"] / "summary.json").write_text(json.dumps({
        "status": status, "complete": status == "scored", "config": config, "checks": checks or ok,
        "heads": {h: sorted(relabel_paths.SETS) for h in ("char", "bpe")}}))
    return dirs


def base_errors() -> dict:
    """Per set (6 utterances of 10 words in 3 clusters): B0 6 errors (WER 10.0), T* 3 (5.0), char 6 (10.0),
    bpe 2 (3.33)."""
    return {"B0": {n: [1, 1, 1, 1, 1, 1] for n in relabel_paths.SETS},
            "T*": {n: [1, 0, 1, 0, 1, 0] for n in relabel_paths.SETS},
            "char": {n: [2, 2, 1, 1, 0, 0] for n in relabel_paths.SETS},
            "bpe": {n: [0, 0, 1, 0, 1, 0] for n in relabel_paths.SETS}}


class PointTest(unittest.TestCase):
    def setUp(self) -> None:
        relabel_paths.require_mount()
        self.models = build(base_errors(), unscored={"B0": ["mm"], "T*": [""], "char": ["uh huh"], "bpe": ["yeah"]})
        self.dirs = write(TMP / "point", self.models)
        self.report = rep.analyze(self.dirs, replicates=200)

    def test_population_is_scored_records_word_weighted(self) -> None:
        s = self.report["sets"]["ami_dev"]
        self.assertEqual(s["utterances"], 7)
        self.assertEqual(s["utterances_scored"], 6)
        self.assertEqual(s["reference_words"], 60)
        self.assertAlmostEqual(s["wer"]["B0"], 10.0)
        self.assertAlmostEqual(s["wer"]["T*"], 5.0)
        self.assertAlmostEqual(s["wer"]["char"], 10.0)
        self.assertAlmostEqual(s["wer"]["bpe"], 10 / 3)

    def test_delta_enc_and_s_per_set_and_on_h4(self) -> None:
        s = self.report["sets"]["librispeech_dev_clean"]
        self.assertAlmostEqual(s["delta_enc"]["bpe"]["point"], 10.0 - 10 / 3)
        self.assertAlmostEqual(s["delta_enc"]["char"]["point"], 0.0)
        self.assertAlmostEqual(s["s"]["bpe"]["point"], (10 / 3 - 5.0) / (10.0 - 5.0))
        self.assertAlmostEqual(s["s"]["char"]["point"], 1.0)
        h4 = self.report["h4"]
        self.assertEqual(h4["sets"], relabel_paths.H4)
        self.assertAlmostEqual(h4["wer"]["bpe"], 10 / 3)
        self.assertAlmostEqual(h4["delta_enc"]["bpe"]["point"], 10.0 - 10 / 3)
        self.assertAlmostEqual(h4["s"]["bpe"]["point"], (10 / 3 - 5.0) / 5.0)

    def test_yodas_is_reported_without_an_interval(self) -> None:
        y = self.report["yodas"]
        self.assertAlmostEqual(y["wer"]["bpe"], 10 / 3)
        self.assertAlmostEqual(y["delta_enc"]["bpe"]["point"], 10.0 - 10 / 3)
        self.assertIsNone(y["delta_enc"]["bpe"]["ci90"])
        self.assertIsNone(y["s"]["bpe"]["ci90"])
        self.assertNotIn("yodas_dev", self.report["bootstrap"]["clusters"])

    def test_secondary_rates_and_backchannel_output(self) -> None:
        s = self.report["sets"]["ami_dev"]
        self.assertAlmostEqual(s["rates"]["B0"]["substitution"], 10.0)
        self.assertAlmostEqual(s["rates"]["B0"]["deletion"], 0.0)
        self.assertAlmostEqual(s["rates"]["B0"]["insertion"], 0.0)
        self.assertAlmostEqual(s["empty_hypothesis_rate"]["bpe"], 0.0)
        ami = self.report["secondary"]["ami_unscored_nonempty"]
        self.assertEqual(ami["T*"]["nonempty_share"], 0.0)
        self.assertEqual(ami["bpe"]["nonempty_share"], 1.0)
        self.assertEqual(ami["bpe"]["excluded"], 1)
        self.assertIn("char_minus_bpe", s)
        self.assertAlmostEqual(s["char_minus_bpe"]["point"], 10.0 - 10 / 3)

    def test_rates_count_deletions_and_insertions(self) -> None:
        models = build(base_errors())
        for name in relabel_paths.SETS:
            models["char"][name] = [rec(r["id"], REF, hyp_with(1, "del")) for r in models["char"][name]]
            models["bpe"][name] = [rec(r["id"], REF, hyp_with(2, "ins")) for r in models["bpe"][name]]
            models["bpe"][name][0] = rec(models["bpe"][name][0]["id"], REF, "")
        report = rep.analyze(write(TMP / "rates", models), replicates=50)
        s = report["sets"]["voxpopuli_dev"]
        self.assertAlmostEqual(s["rates"]["char"]["deletion"], 10.0)
        self.assertAlmostEqual(s["rates"]["bpe"]["insertion"], 100 * 10 / 60)
        self.assertAlmostEqual(s["rates"]["bpe"]["deletion"], 100 * 10 / 60)
        self.assertAlmostEqual(s["empty_hypothesis_rate"]["bpe"], 1 / 6)


class SRuleTest(unittest.TestCase):
    def test_s_only_where_the_teacher_gain_is_at_least_0_30_points(self) -> None:
        errors = base_errors()
        # dev_other: B0 and T* tie (|gain| < 0.30): no per-set s, but it still enters H4 means
        errors["T*"]["librispeech_dev_other"] = [1, 1, 1, 1, 1, 1]
        report = rep.analyze(write(TMP / "srule", build(errors)), replicates=100)
        other = report["sets"]["librispeech_dev_other"]
        self.assertFalse(other["s"]["bpe"]["eligible"])
        self.assertIsNone(other["s"]["bpe"]["point"])
        self.assertIsNone(other["s"]["bpe"]["ci90"])
        self.assertTrue(report["sets"]["ami_dev"]["s"]["bpe"]["eligible"])
        h4 = report["h4"]
        b0, t, b = (h4["wer"][m] for m in ("B0", "T*", "bpe"))
        self.assertAlmostEqual(h4["s"]["bpe"]["point"], (b - t) / (b0 - t))
        self.assertIsNotNone(h4["s"]["bpe"]["ci90"])

    def test_s_threshold_is_inclusive(self) -> None:
        self.assertTrue(rep.s_eligible(10.0, 9.7))
        self.assertTrue(rep.s_eligible(9.7, 10.0))
        self.assertFalse(rep.s_eligible(10.0, 9.71))


class BootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        relabel_paths.require_mount()
        self.models = build(base_errors())
        self.dirs = write(TMP / "boot", self.models)

    def test_defaults_are_the_bootstrap_module_defaults(self) -> None:
        self.assertEqual(rep.REPLICATES, bootstrap.REPLICATES)
        self.assertEqual(rep.REPLICATES, 10_000)
        self.assertEqual(rep.SEED, bootstrap.SEED)
        self.assertEqual(rep.CI, bootstrap.CI)
        self.assertEqual(rep.CI, 0.90)

    def test_clusters_come_from_devsplit(self) -> None:
        tables = rep.tables({m: self.models[m] for m in MODELS}, relabel_paths.H4)
        self.assertEqual(sorted(tables), sorted(relabel_paths.H4))
        for name, t in tables.items():
            expected = sorted({devsplit.cluster_of(name, r["id"]) for r in self.models["B0"][name] if r["scored"]})
            self.assertEqual(t.clusters, expected)
            self.assertEqual(len(t.clusters), 3)

    def test_delta_intervals_are_bootstrap_paired(self) -> None:
        report = rep.analyze(self.dirs, replicates=300)
        tables = rep.tables({m: self.models[m] for m in MODELS}, relabel_paths.H4)
        theirs = bootstrap.paired(tables, {"bpe": ("B0", "bpe"), "char": ("B0", "char")}, relabel_paths.H4,
                                  replicates=300)
        for head in ("bpe", "char"):
            self.assertEqual(report["h4"]["delta_enc"][head]["ci90"], theirs["mean"][head]["ci"])
            for name in relabel_paths.H4:
                self.assertEqual(report["sets"][name]["delta_enc"][head]["ci90"], theirs["sets"][name][head]["ci"])

    def test_one_draw_serves_every_model(self) -> None:
        tables = rep.tables({m: self.models[m] for m in MODELS}, relabel_paths.H4)
        reps = rep.replicate_wers(tables, list(MODELS), replicates=400, seed=7)
        theirs = bootstrap.paired(tables, {"d": ("B0", "bpe"), "e": ("T*", "char")}, relabel_paths.H4,
                                  replicates=400, seed=7)
        diff = np.mean([reps[n]["B0"] - reps[n]["bpe"] for n in relabel_paths.H4], axis=0)
        self.assertEqual(bootstrap._ci(diff, 0.90), theirs["mean"]["d"]["ci"])
        diff = reps["ami_dev"]["T*"] - reps["ami_dev"]["char"]
        self.assertEqual(bootstrap._ci(diff, 0.90), theirs["sets"]["ami_dev"]["e"]["ci"])

    def test_clusters_not_utterances_are_resampled(self) -> None:
        errors = {"B0": {n: [2, 2, 0, 0, 0, 0] for n in relabel_paths.SETS}}  # all of B0's errors in cluster 1
        tables = rep.tables({m: build(errors)[m] for m in MODELS}, relabel_paths.H4)
        reps = rep.replicate_wers(tables, ["B0"], replicates=500, seed=1)
        # 3 clusters of 20 words; cluster 1 holds 4 errors: a replicate's WER is 100 * 4j / 60 for j draws of it
        allowed = {100 * 4 * j / 60 for j in range(4)}
        values = set(np.round(reps["ami_dev"]["B0"], 9))
        self.assertLessEqual(values, {round(v, 9) for v in allowed})
        self.assertGreater(len(values), 1)

    def test_s_per_replicate_is_degenerate_for_the_endpoints(self) -> None:
        models = build(base_errors())
        models["bpe"] = models["T*"]   # s = 0 in every replicate
        models["char"] = models["B0"]  # s = 1 in every replicate
        report = rep.analyze(write(TMP / "ends", models), replicates=200)
        for scope in (report["h4"], report["sets"]["ami_dev"]):
            self.assertEqual(scope["s"]["bpe"]["ci90"], [0.0, 0.0])
            self.assertEqual(scope["s"]["char"]["ci90"], [1.0, 1.0])
            self.assertEqual(scope["delta_enc"]["char"]["ci90"], [0.0, 0.0])

    def test_deterministic(self) -> None:
        a = rep.analyze(self.dirs, replicates=200)
        b = rep.analyze(self.dirs, replicates=200)
        for r in (a, b):
            r.pop("created")
        self.assertEqual(a, b)


class ReadoutTest(unittest.TestCase):
    def test_preregistered_rule(self) -> None:
        self.assertEqual(rep.readout(0.35, [0.10, 0.60]), "A")
        self.assertEqual(rep.readout(0.30, [0.01, 0.55]), "A")
        self.assertEqual(rep.readout(0.35, [-0.05, 0.70]), "B")
        self.assertEqual(rep.readout(0.29, [0.10, 0.50]), "B")
        self.assertEqual(rep.readout(-0.30, [-0.55, -0.01]), "C")
        self.assertEqual(rep.readout(-0.50, [-0.90, 0.02]), "B")
        self.assertEqual(rep.readout(0.0, [-0.1, 0.1]), "B")

    def test_readout_uses_the_bpe_head_on_h4(self) -> None:
        relabel_paths.require_mount()
        report = rep.analyze(write(TMP / "readout", build(base_errors())), replicates=200)
        d = report["readout"]
        self.assertEqual(d["head"], "bpe")
        self.assertEqual(d["delta_enc"], report["h4"]["delta_enc"]["bpe"]["point"])
        self.assertEqual(d["ci90"], report["h4"]["delta_enc"]["bpe"]["ci90"])
        self.assertEqual(d["label"], rep.readout(d["delta_enc"], d["ci90"]))
        self.assertIsNone(d["withheld"])


class ChecksTest(unittest.TestCase):
    def setUp(self) -> None:
        relabel_paths.require_mount()

    def test_check_4_dev_clean_above_10_percent_withholds_the_readout(self) -> None:
        errors = base_errors()
        errors["char"]["librispeech_dev_clean"] = [2, 2, 2, 1, 1, 1]  # 15%
        report = rep.analyze(write(TMP / "check4", build(errors)), replicates=50)
        c4 = report["checks"]["gross_sanity"]
        self.assertFalse(c4["passed"])
        self.assertAlmostEqual(c4["wer"]["char"], 15.0)
        self.assertTrue(c4["heads"]["bpe"]["passed"])
        self.assertEqual(len(c4["heads"]["char"]["examples"]), 6)  # at most 20 (only 6 here)
        self.assertIsNone(report["readout"]["label"])
        self.assertIn("gross_sanity", report["readout"]["withheld"])

    def test_failed_scorer_checks_stop_the_analysis(self) -> None:
        bad = {"passed": False, "hooks_passive": {"passed": True}, "pre_feedback": {"passed": False},
               "determinism": {"passed": True}}
        with self.assertRaises(rep.AnalysisError):
            rep.analyze(write(TMP / "check2", build(base_errors()), checks=bad), replicates=50)

    def test_check_5_recorded_counts_must_recompute(self) -> None:
        models = build(base_errors())
        models["B0"]["ami_dev"][0]["S"] += 1
        with self.assertRaises(rep.AnalysisError):
            rep.analyze(write(TMP / "check5", models), replicates=50)
        models = build(base_errors())
        models["T*"]["voxpopuli_dev"][2]["I"] = 3
        with self.assertRaises(rep.AnalysisError):
            rep.analyze(write(TMP / "check5b", models), replicates=50)

    def test_check_5_passes_and_is_recorded(self) -> None:
        report = rep.analyze(write(TMP / "check5ok", build(base_errors())), replicates=50)
        c5 = report["checks"]["records_consistent"]
        self.assertTrue(c5["passed"])
        self.assertEqual(c5["records_checked"]["B0"]["ami_dev"], 6)

    def test_smoke_or_incomplete_probe_runs_are_refused(self) -> None:
        with self.assertRaises(rep.AnalysisError):
            rep.analyze(write(TMP / "smoke", build(base_errors()), status="smoke", limit=3), replicates=50)
        with self.assertRaises(rep.AnalysisError):
            rep.analyze(write(TMP / "partial", build(base_errors()), status="partial"), replicates=50)

    def test_mismatched_utterances_are_refused(self) -> None:
        models = build(base_errors())
        models["bpe"]["ami_dev"] = models["bpe"]["ami_dev"][:-1]
        with self.assertRaises(rep.AnalysisError):
            rep.analyze(write(TMP / "mismatch", models), replicates=50)


class MainTest(unittest.TestCase):
    def test_production_report_needs_production_inputs_and_settings(self) -> None:
        relabel_paths.require_mount()
        dirs = write(TMP / "main", build(base_errors()))
        with self.assertRaises(SystemExit):
            rep.main(["--probe", str(dirs["probe"]), "--b0", str(dirs["B0"]), "--tstar", str(dirs["T*"]),
                      "--out", str(rep.REPORT)])
        with self.assertRaises(SystemExit):
            rep.main(["--replicates", "100", "--out", str(rep.REPORT)])

    def test_main_writes_a_report_elsewhere(self) -> None:
        relabel_paths.require_mount()
        dirs = write(TMP / "main2", build(base_errors()))
        out = TMP / "main2" / "report.json"
        rc = rep.main(["--probe", str(dirs["probe"]), "--b0", str(dirs["B0"]), "--tstar", str(dirs["T*"]),
                       "--out", str(out), "--replicates", "100"])
        self.assertEqual(rc, 0)
        self.assertIn(json.loads(out.read_text())["readout"]["label"], ("A", "B", "C"))

    def test_analysis_stops_with_exit_2(self) -> None:
        models = build(base_errors())
        models["B0"]["ami_dev"][0]["D"] += 1
        dirs = write(TMP / "main3", models)
        rc = rep.main(["--probe", str(dirs["probe"]), "--b0", str(dirs["B0"]), "--tstar", str(dirs["T*"]),
                       "--out", str(TMP / "main3" / "report.json"), "--replicates", "50"])
        self.assertEqual(rc, 2)
        self.assertFalse((TMP / "main3" / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
