"""score_teacher contracts (diagnostic plan Step 1 and "Resources"): smoke output never lands on the
production path; no run without the parakeet GPU lock; records in the existing per-utterance schema,
with B0's short-clip padding, and throughput fields. CPU only: a stand-in teacher, no model download.

Run from finetune/relabel:  ./python -m unittest discover -s tests -v
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import evaluate  # noqa: E402
import relabel_paths  # noqa: E402
import score_teacher  # noqa: E402
import teachers  # noqa: E402

TMP = relabel_paths.ARTIFACTS / "tmp" / "tests" / "score"
RECORD_KEYS = {"id", "ref", "hyp", "ref_norm", "hyp_norm", "S", "D", "I", "scored", "duration"}


class OutputPathTest(unittest.TestCase):
    def test_smoke_path_never_equals_production_path(self) -> None:
        for name in relabel_paths.REAL_TEACHERS:
            production = relabel_paths.output_dir(name, None)
            self.assertEqual(production, relabel_paths.DIAGNOSTIC / name)
            for limit in (1, 8, 100000):
                smoke = relabel_paths.output_dir(name, limit)
                self.assertNotEqual(smoke, production)
                self.assertFalse(smoke.is_relative_to(relabel_paths.DIAGNOSTIC))
                self.assertTrue(smoke.is_relative_to(relabel_paths.SMOKE))

    def test_vote_and_unknown_names_have_no_scoring_path(self) -> None:
        for name in (relabel_paths.VOTE, "report.json", "../x"):
            with self.assertRaises(ValueError):
                relabel_paths.output_dir(name, None)

    def test_only_dev_sets(self) -> None:
        self.assertEqual(score_teacher.expand_sets(["dev"]), relabel_paths.SETS)
        with self.assertRaises(SystemExit):
            score_teacher.expand_sets(["librispeech_clean"])


class LockTest(unittest.TestCase):
    def setUp(self) -> None:
        relabel_paths.require_mount()
        TMP.mkdir(parents=True, exist_ok=True)
        self.lock = TMP / "gpu.lock"
        self.patch = mock.patch.object(relabel_paths, "GPU_LOCK", self.lock)
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()

    def test_refuses_to_run_while_the_lock_is_held(self) -> None:
        with open(self.lock, "a+") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            with mock.patch.object(teachers.Qwen3ASR, "load", side_effect=AssertionError("loaded without the lock")), \
                 mock.patch.object(score_teacher, "_run_locked", side_effect=AssertionError("ran without the lock")):
                rc = score_teacher.main(["--teacher", relabel_paths.REAL_TEACHERS[0], "--limit", "1",
                                         "--lock-wait", "0.2"])
            fcntl.flock(holder, fcntl.LOCK_UN)
        self.assertEqual(rc, score_teacher.LOCK_BUSY_EXIT)

    def test_runs_inside_the_lock(self) -> None:
        seen = {}

        def probe(*args, **kwargs):
            with open(self.lock, "a+") as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    seen["held"] = False
                    fcntl.flock(other, fcntl.LOCK_UN)
                except BlockingIOError:
                    seen["held"] = True
            return 0

        with mock.patch.object(score_teacher, "_run_locked", side_effect=probe):
            rc = score_teacher.main(["--teacher", relabel_paths.REAL_TEACHERS[0], "--limit", "1", "--lock-wait", "1"])
        self.assertEqual(rc, 0)
        self.assertTrue(seen["held"])


class StandIn(teachers.Teacher):
    name = "qwen3-asr-1.7b"
    hf_id = "stand-in"
    revision = "0" * 40
    license = "none"
    library = "test"
    batch_size = 2

    def _load(self) -> None:
        self.seen = []

    def transcribe(self, clips):
        self.seen.append([len(c) for c in clips])
        return [f"Hello, World {len(c)}." for c in clips], [False] * len(clips)

    def info(self) -> dict:
        return {"name": self.name, "revision": self.revision}


class ScoreSetTest(unittest.TestCase):
    def test_records_schema_padding_and_throughput(self) -> None:
        relabel_paths.require_mount()
        root = TMP / "set"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        rows = []
        for k, seconds in enumerate([0.01, 1.0, 0.5]):
            path = root / f"u{k}.flac"
            sf.write(path, np.zeros(int(seconds * 16000), dtype=np.float32), 16000)
            rows.append({"audio_filepath": str(path), "duration": seconds, "text": f"HELLO WORLD {k}", "id": f"t:u{k}"})
        manifest = root / "toy.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        teacher = StandIn()
        teacher.load("cpu")
        args = score_teacher.parse_args(["--teacher", relabel_paths.REAL_TEACHERS[0], "--limit", "3"])
        args.batch_size, args.max_batch_seconds = 2, 600.0
        result = score_teacher.score_set(teacher, "toy", manifest, args, torch.device("cpu"), teacher.info())
        self.assertEqual([r["id"] for r in result["records"]], ["t:u0", "t:u1", "t:u2"])
        for r in result["records"]:
            self.assertEqual(set(r), RECORD_KEYS)
            self.assertEqual({k: r[k] for k in ("ref_norm", "hyp_norm", "S", "D", "I", "scored")},
                             evaluate.score(r["ref"], r["hyp"]))
        self.assertEqual(result["padded_short_ids"], ["t:u0"])          # 0.01 s < MIN_DECODE_SECONDS
        self.assertEqual(result["records"][0]["hyp"], f"Hello, World {evaluate.MIN_DECODE_SAMPLES}.")
        self.assertEqual(teacher.seen, [[16000, 8000], [evaluate.MIN_DECODE_SAMPLES]])  # longest first, 2 per batch
        for key in ("batch_size", "audio_load_s", "inference_wall_s", "audio_hours", "audio_hours_per_gpu_hour",
                    "peak_gpu_memory_gb", "oom_splits"):
            self.assertIn(key, result["throughput"])
        self.assertEqual(result["wer"], evaluate.wer_summary(result["records"]))


class OomStandIn(StandIn):
    def transcribe(self, clips):
        if len(clips) > 1:
            raise torch.cuda.OutOfMemoryError("stand-in: batch too large")
        return super().transcribe(clips)


class EntryPointTest(unittest.TestCase):
    """score_teacher.main end to end on CPU (--device cpu smoke) with a stand-in teacher and real dev audio."""

    def setUp(self) -> None:
        relabel_paths.require_mount()
        self.root = TMP / "entry"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.patches = [mock.patch.object(relabel_paths, "SMOKE", self.root / "smoke"),
                        mock.patch.object(relabel_paths, "DIAGNOSTIC", self.root / "diagnostic"),
                        mock.patch.object(relabel_paths, "GPU_LOCK", self.root / "gpu.lock")]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()

    def run_main(self, cls, extra=()):
        with mock.patch.object(teachers, "get", lambda name: cls()):
            return score_teacher.main(["--teacher", relabel_paths.REAL_TEACHERS[0], "--limit", "3",
                                       "--sets", "voxpopuli_dev", "yodas_dev", "--device", "cpu",
                                       "--min-available-gb", "0", *extra])

    def test_cpu_smoke_writes_every_set_and_the_summary(self) -> None:
        self.assertEqual(self.run_main(StandIn), 0)
        out = self.root / "smoke" / f"{relabel_paths.REAL_TEACHERS[0]}-limit3-cpu"
        for name in ("voxpopuli_dev", "yodas_dev"):
            data = json.loads((out / f"{name}.json").read_text())
            self.assertEqual(len(data["records"]), 3)
            self.assertEqual(set(data["records"][0]), RECORD_KEYS)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["status"], "smoke")
        self.assertFalse(summary["complete"])
        self.assertIsNone(summary["gpu_lock"])
        self.assertFalse((self.root / "diagnostic").exists())  # the (redirected) production root is never written
        # resumable: a second run with the same configuration skips both sets
        with mock.patch.object(StandIn, "transcribe", side_effect=AssertionError("re-decoded")):
            self.assertEqual(self.run_main(StandIn), 0)

    def test_interrupted_rerun_does_not_leave_a_scored_summary(self) -> None:
        self.assertEqual(self.run_main(StandIn), 0)
        out = self.root / "smoke" / f"{relabel_paths.REAL_TEACHERS[0]}-limit3-cpu"

        class Dies(StandIn):
            def transcribe(self, clips):
                raise KeyboardInterrupt  # stopped during the first set of a rerun with another batch size

        with self.assertRaises(KeyboardInterrupt):
            self.run_main(Dies, ["--batch-size", "1"])
        self.assertEqual(json.loads((out / "summary.json").read_text())["status"], "in_progress")

    def test_oom_batches_are_split_and_counted(self) -> None:
        self.assertEqual(self.run_main(OomStandIn, ["--batch-size", "3"]), 0)
        out = self.root / "smoke" / f"{relabel_paths.REAL_TEACHERS[0]}-limit3-cpu"
        data = json.loads((out / "voxpopuli_dev.json").read_text())
        self.assertTrue(all(r["hyp"].startswith("Hello, World") for r in data["records"]))
        self.assertEqual(data["throughput"]["oom_splits"], 2)  # 3 -> 1 + 2, 2 -> 1 + 1
        self.assertEqual(data["throughput"]["batch_calls"], 3)

    def test_low_host_memory_refuses_to_load(self) -> None:
        with mock.patch.object(score_teacher, "available_memory_gb", return_value=1.0), \
             mock.patch.object(StandIn, "_load", side_effect=AssertionError("loaded")):
            self.assertEqual(self.run_main(StandIn, ["--min-available-gb", "16"]), score_teacher.LOW_MEMORY_EXIT)


if __name__ == "__main__":
    unittest.main()
