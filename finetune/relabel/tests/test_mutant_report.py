"""mutant_report contracts (mutant plan "Evaluation" and "Readout"), on synthetic inputs: the plateau, control-recovery,
extension and A/B/C/unresolved rules, the T5 sign-consistency label, the H4 and T5 cluster mappings and paired draws
(one draw per set for every arm; an interval for a T5 set only with at least 10 clusters; none for tech-dev-1, the
T5 mean or YODAS), and (v3.1) T5's mean without tech-dev-2 for every comparison.

CPU only. Run from finetune/relabel:  ./python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import shutil
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

import evaluate  # noqa: E402
import mutant_report as mr  # noqa: E402
import relabel_paths  # noqa: E402
from training import bootstrap, devsplit, t5clusters  # noqa: E402

TMP = relabel_paths.ARTIFACTS / "tmp" / "tests" / "mutant-report"


def curve(main, ext=None):
    """Held-out evals: main = (25%, 50%, 75%, 100%) WERs at steps 25, 50, 75, 100; ext = the extension's end."""
    out = {"main": [{"step": s, "wer": w} for s, w in zip((25, 50, 75, 100), main)], "S": 100}
    out["extension"] = None if ext is None else [{"step": 150, "wer": ext}]
    return out


class TestRules(unittest.TestCase):
    def test_plateau(self):
        self.assertTrue(mr.plateau(curve((9, 8, 7.0, 6.90)), extended=False)["plateaued"])
        self.assertFalse(mr.plateau(curve((9, 8, 7.0, 6.89)), extended=False)["plateaued"])
        self.assertTrue(mr.plateau(curve((9, 8, 7.0, 7.2)), extended=False)["plateaued"])  # got worse: no gain
        p = mr.plateau(curve((9, 8, 7.0, 6.5), ext=6.45), extended=True)
        self.assertEqual((p["from_step"], p["to_step"]), (100, 150))
        self.assertTrue(p["plateaued"])
        self.assertFalse(mr.plateau(curve((9, 8, 7.0, 6.5), ext=6.3), extended=True)["plateaued"])

    def test_recovery(self):
        self.assertTrue(mr.recovered(5.90))
        self.assertTrue(mr.recovered(5.60))
        self.assertFalse(mr.recovered(5.91))

    def test_verdict(self):
        self.assertEqual(mr.verdict(0.30, [0.05, 0.6]), "A")
        self.assertEqual(mr.verdict(0.35, [-0.05, 0.8]), "B")
        self.assertEqual(mr.verdict(0.29, [0.1, 0.5]), "B")
        self.assertEqual(mr.verdict(-0.30, [-0.6, -0.01]), "C")
        self.assertEqual(mr.verdict(-0.5, [-0.9, 0.0]), "B")

    def test_t5_label(self):
        self.assertEqual(mr.t5_label("A", 0.31), "consistent on T5")
        self.assertEqual(mr.t5_label("A", 0.29), "H4 only")
        self.assertEqual(mr.t5_label("A", -0.5), "H4 only")
        self.assertEqual(mr.t5_label("C", -0.4), "consistent on T5")
        self.assertEqual(mr.t5_label("C", 0.4), "H4 only")
        self.assertIsNone(mr.t5_label("B", 1.0))

    def state(self, **kw):
        base = {"extended": False, "recovery": True, "plateaued": {a: True for a in mr.ARMS},
                "delta": {"D-T": (0.5, [0.2, 0.8]), "D-Q": (-0.1, [-0.4, 0.2])},
                "t5_delta": {"D-T": 0.4, "D-Q": -0.2}}
        base.update(kw)
        return base

    def test_readout_final_after_main(self):
        r = mr.readout(self.state())
        self.assertEqual(r["status"], "final")
        self.assertEqual(r["verdicts"]["D-T"]["label"], "A")
        self.assertEqual(r["verdicts"]["D-T"]["t5"], "consistent on T5")
        self.assertEqual(r["verdicts"]["D-Q"]["label"], "B")
        self.assertIsNone(r["verdicts"]["D-Q"]["t5"])

    def test_extension_when_recovery_fails_or_an_arm_has_not_plateaued(self):
        r = mr.readout(self.state(recovery=False))
        self.assertEqual(r["status"], "extension required")
        self.assertIsNone(r["verdicts"])
        r = mr.readout(self.state(plateaued={"D-B": True, "D-T": False, "D-Q": True}))
        self.assertEqual(r["status"], "extension required")

    def test_after_the_extension(self):
        r = mr.readout(self.state(extended=True))
        self.assertEqual(r["status"], "final")
        self.assertEqual(r["verdicts"]["D-T"]["label"], "A")
        r = mr.readout(self.state(extended=True, recovery=False, plateaued={"D-B": True, "D-T": True, "D-Q": False}))
        self.assertEqual(r["status"], "unresolved")
        for x in ("D-T", "D-Q"):
            self.assertEqual(r["verdicts"][x]["label"], "unresolved")
        self.assertEqual(r["failed"], {"control_recovery": ["D-B"], "plateau": ["D-Q"]})


class TestClusters(unittest.TestCase):
    def test_t5_clusters_from_the_id_prefix(self):
        self.assertEqual(t5clusters.cluster_of("tech-dev-2", "tech-dev-2:call-17-0007"), "call-17")
        self.assertEqual(t5clusters.cluster_of("stem-cs", "stem-cs:course-a_lecture-05-0012"),
                         "course-a_lecture-05")
        with self.assertRaises(ValueError):
            t5clusters.cluster_of("tech-dev-1", "tech-dev-1:dev-0-0000")
        with self.assertRaises(ValueError):
            t5clusters.cluster_of("stem-cs", "stem-math:course-b_lecture-05-0001")

    def test_interval_eligibility(self):
        ids = {"tech-dev-1": [f"tech-dev-1:dev-{i}-0000" for i in range(30)],
               "tech-dev-2": [f"tech-dev-2:{c}-{k:04d}" for c in range(12) for k in range(3)],
               "stem-chem": [f"stem-chem:L{c}-{k:04d}" for c in range(2) for k in range(5)],
               "stem-cs": [f"stem-cs:L{c}-{k:04d}" for c in range(4) for k in range(5)],
               "stem-math": [f"stem-math:L{c}-{k:04d}" for c in range(9) for k in range(5)]}
        got = t5clusters.interval_sets(ids)
        self.assertEqual({k: (v["clusters"], v["interval"]) for k, v in got.items()},
                         {"tech-dev-1": (None, False), "tech-dev-2": (12, True), "stem-chem": (2, False),
                          "stem-cs": (4, False), "stem-math": (9, False)})

    def test_t5_means(self):
        values = {"tech-dev-1": 1.0, "tech-dev-2": 10.0, "stem-chem": 2.0, "stem-cs": 3.0, "stem-math": 4.0}
        self.assertEqual(t5clusters.mean(values), 4.0)
        self.assertEqual(t5clusters.mean(values, t5clusters.SETS_WITHOUT), 2.5)
        self.assertNotIn("tech-dev-2", t5clusters.SETS_WITHOUT)


# --- a synthetic world: records of three arms and B0 on H4, YODAS and T5 ---------------------------------------

def make_ids():
    ids = {"librispeech_dev_clean": [f"librispeech:{s}-{c}-{k:04d}" for s in range(20) for c in (1,) for k in range(4)],
           "librispeech_dev_other": [f"librispeech:{s}-{c}-{k:04d}" for s in range(40, 55) for c in (2,) for k in range(4)],
           "ami_dev": [f"ami:AMI_ES20{m:02d}a_H0{h}_X_{k:07d}_{k + 9:07d}" for m in range(12) for h in range(2)
                       for k in range(3)],
           "voxpopuli_dev": [f"voxpopuli:2018{m:02d}01-0900-PLENARY-{k}-en_x_{k}" for m in range(1, 13) for k in range(4)],
           "yodas_dev": [f"yodas:y{k}" for k in range(30)],
           "tech-dev-1": [f"tech-dev-1:dev-{k}-0000" for k in range(25)],
           "tech-dev-2": [f"tech-dev-2:{4000000 + c}-{k:04d}" for c in range(14) for k in range(5)],
           "stem-chem": [f"stem-chem:L{c}-{k:04d}" for c in range(2) for k in range(10)],
           "stem-cs": [f"stem-cs:L{c}-{k:04d}" for c in range(4) for k in range(6)],
           "stem-math": [f"stem-math:L{c}-{k:04d}" for c in range(3) for k in range(7)]}
    return ids


WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()


def records(ids, error_rate, seed, unscored=()):
    rng = np.random.default_rng(seed)
    out = []
    for i in sorted(ids):
        ref = " ".join(WORDS[(len(i) + k) % len(WORDS)] for k in range(8))
        words = ref.split()
        hyp = [w if rng.random() > error_rate else "wrong" for w in words]
        hyp_text = "" if i in unscored else " ".join(hyp)
        ref_text = "ignore time segment in scoring" if i in unscored else ref
        out.append({"id": i, "ref": ref_text, "hyp": hyp_text, **evaluate.score(ref_text, hyp_text), "duration": 2.0})
    return out


def write_world(root, rates, b0_rate=0.06, extension=False):
    ids = make_ids()
    unscored = {ids["ami_dev"][0]}
    dirs = {}
    for k, arm in enumerate(mr.ARMS):
        d = root / arm
        for s, sid in ids.items():
            body = {"set": s, "limit": None, "records": records(sid, rates[arm], 100 + k * 13 + len(s), unscored)}
            (d / ("extension" if extension else "")).mkdir(parents=True, exist_ok=True)
            (d / ("extension" if extension else "") / f"{s}.json").write_text(json.dumps(body))
        dirs[arm] = d / "extension" if extension else d
    b0 = {}
    for s, sid in ids.items():
        p = root / "B0" / f"{s}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"set": s, "limit": None, "records": records(sid, b0_rate, 7 + len(s), unscored)}))
        b0[s] = p.parent
    return dirs, b0, ids


class TestAnalyze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)
        cls.dirs, cls.b0, cls.ids = write_world(TMP / "w", {"D-B": 0.05, "D-T": 0.03, "D-Q": 0.05})
        curves = {a: curve((9, 8, 7.0, 6.95)) for a in mr.ARMS}
        cls.report = mr.analyze(cls.dirs, cls.b0, curves, extended=False, replicates=400)

    def test_population_and_wers(self):
        r = self.report
        for s in mr.H4 + [mr.YODAS] + list(t5clusters.SETS):
            for m in mr.MODELS:
                rows = json.loads(((self.b0[s] if m == mr.B0 else self.dirs[m]) / f"{s}.json").read_text())["records"]
                scored = [x for x in rows if x["scored"]]
                want = 100 * sum(x["S"] + x["D"] + x["I"] for x in scored) / sum(len(x["ref_norm"].split())
                                                                                  for x in scored)
                self.assertAlmostEqual(r["sets"][s]["wer"][m], want)
        self.assertAlmostEqual(r["h4"]["wer"]["D-T"], np.mean([r["sets"][s]["wer"]["D-T"] for s in mr.H4]))

    def test_t5_rows_with_and_without_tech_dev_2(self):
        r = self.report
        for m in mr.MODELS:
            self.assertAlmostEqual(r["t5"]["wer"][m], np.mean([r["sets"][s]["wer"][m] for s in t5clusters.SETS]))
            self.assertAlmostEqual(r["t5_without_tech_dev_2"]["wer"][m],
                                   np.mean([r["sets"][s]["wer"][m] for s in t5clusters.SETS_WITHOUT]))
        self.assertEqual(r["t5_without_tech_dev_2"]["sets"], list(t5clusters.SETS_WITHOUT))
        self.assertEqual(set(r["t5_without_tech_dev_2"]["contrasts"]), set(mr.CONTRASTS))
        for name, (a, b) in mr.CONTRASTS.items():
            for key, sets in (("t5", t5clusters.SETS), ("t5_without_tech_dev_2", t5clusters.SETS_WITHOUT)):
                want = np.mean([r["sets"][s]["wer"][a] - r["sets"][s]["wer"][b] for s in sets])
                self.assertAlmostEqual(r[key]["contrasts"][name]["point"], want)
                self.assertIsNone(r[key]["contrasts"][name]["ci90"])

    def test_intervals_where_the_plan_gives_them(self):
        r = self.report
        for s in mr.H4:
            self.assertIsNotNone(r["sets"][s]["contrasts"]["D-T"]["ci90"])
            self.assertEqual(r["sets"][s]["clusters"], len({devsplit.cluster_of(s, i) for i in self.ids[s]}))
        self.assertIsNotNone(r["h4"]["contrasts"]["D-T"]["ci90"])
        self.assertIsNotNone(r["sets"]["tech-dev-2"]["contrasts"]["D-T"]["ci90"])
        self.assertEqual(r["sets"]["tech-dev-2"]["clusters"], 14)
        for s in ("tech-dev-1", "stem-chem", "stem-cs", "stem-math", mr.YODAS):
            self.assertIsNone(r["sets"][s]["contrasts"]["D-T"]["ci90"], s)
        self.assertIsNone(r["sets"]["tech-dev-1"]["clusters"])
        self.assertEqual(r["sets"]["stem-cs"]["clusters"], 4)

    def test_paired_draws_match_bootstrap_and_are_shared_by_every_arm(self):
        tabs = {s: bootstrap.SetTable.from_records(
            {m: json.loads(((self.b0[s] if m == mr.B0 else self.dirs[m]) / f"{s}.json").read_text())["records"]
             for m in mr.MODELS}, cluster_of=lambda i, s=s: devsplit.cluster_of(s, i)) for s in mr.H4}
        want = bootstrap.paired(tabs, mr.CONTRASTS, mr.H4, replicates=400)
        for name in mr.CONTRASTS:
            self.assertEqual(self.report["h4"]["contrasts"][name]["ci90"], want["mean"][name]["ci"])
        # identical records in two arms: every replicate's difference is 0
        dirs, b0, _ = write_world(TMP / "same", {"D-B": 0.05, "D-T": 0.05, "D-Q": 0.04})
        shutil.rmtree(dirs["D-T"])
        shutil.copytree(dirs["D-B"], dirs["D-T"])
        r = mr.analyze(dirs, b0, {a: curve((9, 8, 7, 7)) for a in mr.ARMS}, extended=False, replicates=200)
        self.assertEqual(r["h4"]["contrasts"]["D-T"]["ci90"], [0.0, 0.0])
        self.assertEqual(r["sets"]["tech-dev-2"]["contrasts"]["D-T"]["ci90"], [0.0, 0.0])

    def test_readout_uses_h4_and_the_five_set_t5_mean(self):
        r = self.report
        st = r["readout"]["inputs"]
        self.assertAlmostEqual(st["delta"]["D-T"][0], r["h4"]["wer"]["D-B"] - r["h4"]["wer"]["D-T"])
        self.assertAlmostEqual(st["t5_delta"]["D-T"], r["t5"]["wer"]["D-B"] - r["t5"]["wer"]["D-T"])
        self.assertEqual(st["recovery"], r["h4"]["wer"]["D-B"] <= mr.RECOVERY_MAX + mr.EPS)

    def test_mismatched_ids_are_refused(self):
        dirs, b0, ids = write_world(TMP / "bad", {"D-B": 0.05, "D-T": 0.03, "D-Q": 0.05})
        p = dirs["D-Q"] / "ami_dev.json"
        body = json.loads(p.read_text())
        body["records"] = body["records"][1:]
        p.write_text(json.dumps(body))
        with self.assertRaises(mr.AnalysisError):
            mr.analyze(dirs, b0, {a: curve((9, 8, 7, 7)) for a in mr.ARMS}, extended=False, replicates=50)

    def test_smoke_records_are_refused(self):
        dirs, b0, ids = write_world(TMP / "lim", {"D-B": 0.05, "D-T": 0.03, "D-Q": 0.05})
        p = dirs["D-T"] / "stem-cs.json"
        body = json.loads(p.read_text())
        body["limit"] = 5
        p.write_text(json.dumps(body))
        with self.assertRaises(mr.AnalysisError):
            mr.analyze(dirs, b0, {a: curve((9, 8, 7, 7)) for a in mr.ARMS}, extended=False, replicates=50)


if __name__ == "__main__":
    unittest.main()
