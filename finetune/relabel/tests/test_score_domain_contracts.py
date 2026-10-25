"""score_domain contracts (labeling plan Part A "Critical-term recall", "Baselines", "reported separately,
and as an unweighted mean per set"): exact term reproduction in the aligned hypothesis, per-part and
unweighted per-set means, only B0/M1/MB1 scored, smoke output never on the production path, no run without
the parakeet GPU lock, a failed dev cross-check writes nothing, complete builds only, and resumable parts.
Stand-in model and decoder (evaluate.evaluate_set is replaced); no GPU.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p "test_score_domain_*" -v
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import unittest
import uuid
from unittest import mock

import devsets
import evaluate
import relabel_paths
import score_domain as sd
import terms
from devsets import layout

REAL_ROOT = layout.root("RELABEL_ROOT")


def record(rid: str, ref: str, hyp: str, scored: bool = True) -> dict:
    return {"id": rid, "ref": ref, "hyp": hyp, **evaluate.score(ref, hyp), "duration": 2.0,
            **({} if scored else {"scored": False})}


class TermRecallTest(unittest.TestCase):
    def setUp(self) -> None:
        self.m = terms.Matcher(["ketamine", "tranexamic acid", "blood pressure"], evaluate.normalize)

    def recall(self, ref: str, hyp: str):
        return sd.term_occurrences(evaluate.normalize(ref), evaluate.normalize(hyp), self.m)

    def test_exact_reproduction_in_the_alignment(self) -> None:
        ref = "Give ketamine and tranexamic acid now."
        self.assertEqual(self.recall(ref, ref), [(1, 2, True), (3, 5, True)])
        self.assertEqual(self.recall(ref, "give ketamine and tranexamic asset now"), [(1, 2, True), (3, 5, False)])
        self.assertEqual(self.recall(ref, "give ketamine and tranexamic uh huh acid now")[1][2], False)  # inside
        self.assertEqual(self.recall(ref, "give ketamine and tranexamic acid really now")[1][2], True)  # boundary
        self.assertEqual(self.recall(ref, "give and tranexamic acid now")[0][2], False)  # deleted
        self.assertEqual(self.recall("blood pressure is low", "blood pressure is low"), [(0, 2, True)])
        self.assertEqual(self.recall("nothing here", "nothing here"), [])

    def test_recall_counts_scored_records_only(self) -> None:
        records = [record("a", "ketamine and tranexamic acid", "ketamine and tranexamic asset"),
                   record("b", "blood pressure", "blood pressure"),
                   {**record("c", "ketamine", "no"), "scored": False}]
        result = sd.term_recall(records, self.m)
        self.assertEqual(result, {"occurrences": 3, "reproduced": 2, "recall": 2 / 3})
        self.assertEqual(records[0]["terms"], [[0, 1, True], [2, 4, False]])
        self.assertNotIn("terms", records[2])
        self.assertIsNone(sd.term_recall([record("d", "plain words", "plain words")], self.m)["recall"])


def fake_result(part: str, ref_words: int, errors: int, utterances: int, recall=None) -> dict:
    result = {"source": {"sha256": "x"}, "utterances": utterances, "audio_hours": utterances / 1000, "limit": None,
              "wer": {"wer": errors / ref_words, "substitutions": errors, "deletions": 0, "insertions": 0,
                      "ref_words": ref_words, "utterances": utterances}}
    if recall is not None:
        result["term_recall"] = {"occurrences": recall[1], "reproduced": recall[0], "recall": recall[0] / recall[1]}
    return result


class SummaryTest(unittest.TestCase):
    def test_parts_and_unweighted_set_means(self) -> None:
        results = {"med-dev-1": fake_result("med-dev-1", 1000, 100, 50, (90, 100)),
                   "med-dev-2": fake_result("med-dev-2", 100000, 2000, 5000, (900, 1000)),
                   "med-dev-3": fake_result("med-dev-3", 10000, 500, 400, (5, 10)),
                   "tech-dev-1": fake_result("tech-dev-1", 2000, 100, 100),
                   "tech-dev-2": fake_result("tech-dev-2", 50000, 5000, 3000)}
        s = sd.summarize("B0", results, list(devsets.DEFAULT_SETS))
        self.assertAlmostEqual(s["parts"]["med-dev-2"]["wer"], 2.0)
        self.assertAlmostEqual(s["sets"]["dev-medical"]["mean_wer"], (10.0 + 2.0 + 5.0) / 3)
        self.assertAlmostEqual(s["sets"]["dev-technical"]["mean_wer"], (5.0 + 10.0) / 2)
        self.assertAlmostEqual(s["sets"]["dev-medical"]["mean_term_recall"], (0.9 + 0.9 + 0.5) / 3)
        self.assertEqual(s["sets"]["dev-medical"]["pooled_term_recall"]["occurrences"], 1110)
        self.assertNotIn("mean_term_recall", s["sets"]["dev-technical"])

    def test_a_set_with_a_missing_part_has_no_mean(self) -> None:
        results = {"med-dev-1": fake_result("med-dev-1", 1000, 100, 50, (1, 2))}
        self.assertEqual(sd.summarize("B0", results, ["dev-medical"])["sets"], {})


class PathsAndArgsTest(unittest.TestCase):
    def test_only_the_three_students(self) -> None:
        self.assertEqual(sd.MODELS, ("B0", "M1", "MB1"))
        for name in ("qwen3-asr-1.7b", "T*", "vote"):
            with self.assertRaises(ValueError):
                sd.output_dir(name, None, None)
        with self.assertRaises(SystemExit):
            sd.parse_args(["--model", "qwen3-asr-1.7b"])

    def test_smoke_never_on_the_production_path(self) -> None:
        for model in sd.MODELS:
            prod = sd.output_dir(model, None, None)
            self.assertEqual(prod, layout.scores_root() / model)
            for limit, build in ((8, None), (None, 2), (4, 2)):
                smoke = sd.output_dir(model, limit, build)
                self.assertFalse(smoke.is_relative_to(layout.production_root()))
                self.assertTrue(smoke.is_relative_to(layout.smoke_scores_root()))

    def test_cpu_only_for_limited_runs(self) -> None:
        with self.assertRaises(SystemExit):
            sd.parse_args(["--model", "B0", "--device", "cpu"])
        self.assertEqual(sd.parse_args(["--model", "B0", "--device", "cpu", "--limit", "4"]).limit, 4)


class RunFixture(unittest.TestCase):
    """A complete fake build in a temporary $RELABEL_ROOT and a stand-in decoder."""

    def setUp(self) -> None:
        layout.require_mount()
        self.tmp = REAL_ROOT / "tmp" / "tests" / f"score-domain-{uuid.uuid4().hex[:8]}"
        self.env = mock.patch.dict(os.environ, {"RELABEL_ROOT": str(self.tmp)})
        self.env.start()
        self.lock = self.tmp / "gpu.lock"
        self.lay = layout.Layout.production()
        for part in devsets.PARTS:
            rows = [{"audio_filepath": f"/nonexistent/{part}/{i}.flac", "duration": 2.0,
                     "text": f"give ketamine now {i}", "id": f"{part}:r-{i:04d}", "source": part, "part": part}
                    for i in range(3)]
            manifest = self.lay.manifest(part)
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
            meta = {"complete": True, "config_hash": "h", "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
            self.lay.meta(part).write_text(json.dumps(meta))
        self.decoded = []
        patches = [
            mock.patch.object(relabel_paths, "GPU_LOCK", self.lock),
            mock.patch.object(sd, "describe", side_effect=lambda m: {"sha256": f"w-{m}", "model": m}),
            mock.patch.object(sd, "load_model", return_value=(object(), "no_while_loops")),
            mock.patch.object(sd, "require_mode", return_value="no_while_loops"),
            mock.patch.object(sd.evaluate, "evaluate_set", side_effect=self.fake_evaluate_set),
            mock.patch("torch.cuda.is_available", return_value=True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.check = {"pass": True, "set": "ami_dev", "wer": 1.0, "recorded_wer": 1.0, "identical_hypotheses": 1,
                      "utterances": 1, "difference_points": 0.0}
        p = mock.patch.object(sd, "crosscheck", side_effect=lambda *a: dict(self.check))
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self) -> None:
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake_evaluate_set(self, model, name, manifest, source, batch_size, limit, device, max_batch_seconds):
        self.decoded.append(name)
        self.assertEqual((batch_size, max_batch_seconds), (64, 1200.0))
        rows = evaluate.read_manifest(manifest)[:limit]
        records = [record(r["id"], r["text"], r["text"].replace("ketamine", "ketamin") if i == 0 else r["text"])
                   for i, r in enumerate(rows)]
        return {"source": source, "set": name, "utterances": len(records), "limit": limit, "records": records,
                "audio_hours": len(records) * 2 / 3600, "wer": evaluate.wer_summary(records)}


class RunTest(RunFixture):
    def test_full_run_then_resume(self) -> None:
        self.assertEqual(sd.score("M1", list(devsets.DEFAULT_SETS), None, None, "cuda", 5, False), 0)
        out = layout.scores_root() / "M1"
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(sorted(summary["parts"]), sorted(p for s in devsets.DEFAULT_SETS for p in devsets.SETS[s]))
        self.assertAlmostEqual(summary["parts"]["med-dev-1"]["term_recall"]["recall"], 2 / 3)
        self.assertNotIn("term_recall", summary["parts"]["tech-dev-1"])
        self.assertIn("crosscheck", summary)
        part = json.loads((out / "med-dev-1.json").read_text())
        self.assertEqual(part["config"]["weights_sha256"], "w-M1")
        self.assertIn("terms", part["records"][0])
        self.decoded.clear()
        self.assertEqual(sd.score("M1", list(devsets.DEFAULT_SETS), None, None, "cuda", 5, False), 0)
        self.assertEqual(self.decoded, [])
        self.assertEqual(json.loads((out / "summary.json").read_text())["parts"], summary["parts"])

    def test_refuses_without_the_gpu_lock(self) -> None:
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock, "a+") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(sd.score("B0", ["dev-medical"], None, None, "cuda", 0.2, False), sd.LOCK_BUSY_EXIT)
        self.assertEqual(self.decoded, [])
        self.assertFalse((layout.scores_root() / "B0" / "med-dev-1.json").exists())

    def test_failed_crosscheck_writes_no_scores(self) -> None:
        self.check.update({"pass": False, "difference_points": 0.5})
        with self.assertRaises(SystemExit):
            sd.score("MB1", list(devsets.DEFAULT_SETS), None, None, "cuda", 5, False)
        self.assertEqual(self.decoded, [])
        out = layout.scores_root() / "MB1"
        self.assertEqual(sorted(p.name for p in out.glob("*-dev-*.json")), [])
        self.assertEqual(json.loads((out / "summary.json").read_text())["status"], "in_progress")

    def test_limited_run_goes_to_the_smoke_path(self) -> None:
        self.assertEqual(sd.score("B0", ["dev-technical"], 2, None, "cuda", 5, False), 0)
        self.assertFalse(layout.scores_root().exists())
        out = layout.smoke_scores_root() / "B0-limit2"
        self.assertEqual(json.loads((out / "tech-dev-1.json").read_text())["utterances"], 2)

    def test_a_part_without_term_occurrences_completes(self) -> None:
        for part in devsets.SETS["dev-medical"]:
            rows = [json.loads(line) for line in self.lay.manifest(part).read_text().splitlines()]
            manifest = self.lay.manifest(part)
            manifest.write_text("".join(json.dumps({**r, "text": "plain words only"}) + "\n" for r in rows))
            meta = json.loads(self.lay.meta(part).read_text())
            meta["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.lay.meta(part).write_text(json.dumps(meta))
        self.assertEqual(sd.score("B0", ["dev-medical"], 1, None, "cuda", 5, False), 0)
        summary = json.loads((layout.smoke_scores_root() / "B0-limit1" / "summary.json").read_text())
        self.assertEqual(summary["status"], "complete")
        self.assertIsNone(summary["parts"]["med-dev-1"]["term_recall"]["recall"])
        self.assertIsNone(summary["sets"]["dev-medical"]["mean_term_recall"])
        self.assertEqual(sd.score("B0", ["dev-medical"], None, None, "cuda", 5, False), 0)
        self.assertIn("| B0 | med-dev-1 |", sd.table())

    def test_incomplete_build_is_refused(self) -> None:
        self.lay.meta("med-dev-2").unlink()
        with self.assertRaises(SystemExit):
            sd.score("B0", ["dev-medical"], None, None, "cuda", 5, False)
        with open(self.lay.manifest("tech-dev-1"), "a") as handle:
            handle.write("\n")
        with self.assertRaises(SystemExit):
            sd.score("B0", ["dev-technical"], None, None, "cuda", 5, False)
        self.assertEqual(self.decoded, [])


if __name__ == "__main__":
    unittest.main()
