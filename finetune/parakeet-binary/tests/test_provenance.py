"""Contract test 10 (plans/pilot.md "Contract tests"): the reused parakeet-ternary and
whisper-ternary sources are byte-identical to what M1 recorded (summary.json "source_hashes"),
so the binary pilot trains and scores with M1's code."""
from __future__ import annotations

import hashlib
import json
import unittest

import binary_fixtures  # noqa: F401  (CPU only)

import binary_paths
import paths

EXPECTED_SOURCES = {"parakeet-ternary/train.py", "parakeet-ternary/quant.py", "parakeet-ternary/stream.py",
                    "parakeet-ternary/teacher.py", "parakeet-ternary/evaluate.py",
                    "parakeet-ternary/export.py", "parakeet-ternary/data.py", "parakeet-ternary/paths.py",
                    "whisper-ternary/quant.py", "whisper-ternary/wer.py"}


class ProvenanceTest(unittest.TestCase):
    def test_reused_sources_hash_to_m1_summary(self) -> None:
        paths.require_mount()
        summary = json.loads((binary_paths.M1_RUN / "summary.json").read_text())
        recorded = summary["source_hashes"]
        self.assertEqual(set(recorded), EXPECTED_SOURCES)
        finetune = paths.REPO / "finetune"
        for name, digest in sorted(recorded.items()):
            with self.subTest(name):
                self.assertEqual(hashlib.sha256((finetune / name).read_bytes()).hexdigest(), digest)

    def test_imports_resolve_to_the_recorded_files(self) -> None:
        import evaluate
        import export
        import quant
        import stream
        import teacher
        import train
        finetune = (paths.REPO / "finetune").resolve()
        for module in (train, quant, stream, teacher, evaluate, export, paths):
            with self.subTest(module.__name__):
                self.assertEqual(module.__file__, str(finetune / "parakeet-ternary" / f"{module.__name__}.py"))
        self.assertEqual(str(quant.WHISPER_DIR.resolve()), str(finetune / "whisper-ternary"))


    def test_binary_runs_record_the_runtime_wrappers(self) -> None:
        """Review round 1, finding 3: the interpreter wrapper (it decides the library path, hence
        the decoder backend) and ./heavy are training and evaluation sources: recorded at every
        start, resume and finish and checked by the sweep."""
        import binary_sweep
        import binary_train
        import train
        for name in ("parakeet-binary/python", "parakeet-binary/heavy"):
            self.assertIn(name, binary_paths.BINARY_SOURCES)
            self.assertIn(name, binary_sweep.TRAINING_SOURCES)
            self.assertIn(name, binary_sweep.EVAL_SOURCES)
        with binary_train.seam("binary"):
            recorded = train.training_source_hashes()
        self.assertEqual(recorded["parakeet-binary/python"],
                         hashlib.sha256((binary_paths.HERE / "python").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
