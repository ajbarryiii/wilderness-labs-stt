"""pull / resume / pull (Codex code review c2 finding 3). NixOS:
    finetune/parakeet-binary/python cpu1bit/shared/tests/test_pull.py
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mac"))
import design  # noqa: E402
import pull  # noqa: E402
import runner  # noqa: E402


def listing(root: Path) -> dict:
    out = {}
    for p in root.rglob("*"):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class PullResumePull(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.mac = self.root / "mac"
        self.nix = self.root / "nix"
        self.mac.mkdir()
        self.nix.mkdir()
        self.blocks = design.design(["B", "C", "D"], 1, 7, "x", {"threads": 8, "components": 2})

    def tearDown(self):
        shutil.rmtree(self.root)

    def run_stage(self, valid_seq: set, log_lines: list[str]):
        """One runner invocation: its own log, raw streams written before each record."""
        log = self.mac / f"runner-{len(list(self.mac.glob('runner-*')))}.log.jsonl"
        with open(log, "a") as f:
            for line in log_lines:
                f.write(line + "\n")

        def run_one(b, stem):
            Path(stem + ".powermetrics.plist0").write_bytes(b"raw " + b["block_id"].encode())
            Path(stem + ".stderr").write_bytes(b"")
            return {"analysis": {"valid": b["sequence"] in valid_seq, "windows": []}}
        return runner.execute_stage(self.blocks, str(self.mac), "fp", run_one, lambda d: None, max_attempts=1, passes=1)

    def test_pull_resume_pull(self):
        runner.write_new(str(self.mac / "stage.json"), {"fingerprint_sha256": "fp"})
        st = self.run_stage({0, 1, 2, 3, 4}, ['{"event": "start"}'])
        self.assertEqual(st["status"], "incomplete")
        with open(self.mac / "runner-0.log.jsonl", "a") as f:     # still running when pulled: partial line
            f.write('{"event": "seq')
        Path(self.mac / "x-r0-s5-p0-late.att1.powermetrics.plist0").write_bytes(b"in progress")
        c1 = pull.merge(self.mac, self.nix, listing(self.mac))
        self.assertEqual(c1.get("skip"), 1)                        # the in-progress raw stream
        self.assertFalse((self.nix / "x-r0-s5-p0-late.att1.powermetrics.plist0").exists())
        with open(self.mac / "runner-0.log.jsonl", "a") as f:     # the first invocation finishes its log
            f.write('uence_attempt"}\n')
        (self.mac / "x-r0-s5-p0-late.att1.powermetrics.plist0").unlink()
        st2 = self.run_stage({0, 1, 2, 3, 4, 5}, ['{"event": "resume"}'])
        self.assertEqual(st2["status"], "complete")
        c2 = pull.merge(self.mac, self.nix, listing(self.mac))
        self.assertEqual(c2.get("extend"), 1)
        self.assertGreater(c2.get("copy", 0), 0)
        self.assertEqual(listing(self.mac), listing(self.nix))
        c3 = pull.merge(self.mac, self.nix, listing(self.mac))      # idempotent
        self.assertEqual(set(c3), {"same"})

    def test_conflicts_abort_without_changes(self):
        runner.write_new(str(self.mac / "stage.json"), {"fingerprint_sha256": "fp"})
        self.run_stage({0, 1, 2, 3, 4, 5}, ['{"event": "start"}'])
        pull.merge(self.mac, self.nix, listing(self.mac))
        before = listing(self.nix)
        rec = next(self.mac.glob("x-r0-s0-*.att0.json"))
        os.chmod(rec, 0o644)
        rec.write_text('{"tampered": true}')
        Path(self.mac / "new-evidence.json").write_text("{}")
        with self.assertRaises(pull.PullConflict):
            pull.merge(self.mac, self.nix, listing(self.mac))
        self.assertEqual(listing(self.nix), before)                # nothing applied
        self._check_log_rewrite_conflict()

    def _check_log_rewrite_conflict(self):
        log = next(self.mac.glob("runner-*.log.jsonl"))
        nix_log = self.nix / log.name
        log.write_text('{"event": "rewritten"}\n')
        with self.assertRaises(pull.PullConflict):
            pull.merge(self.mac, self.nix, {log.name: listing(self.mac)[log.name]})
        self.assertTrue(nix_log.read_text().startswith('{"event": "start"}'))

    def test_classification(self):
        names = {"a.att0.json", "a.att0.powermetrics.plist0", "b.att0.stderr", "runner-1.log.jsonl", ".tmp-a.json.1.2"}
        self.assertEqual(pull.classify("a.att0.powermetrics.plist0", names), "evidence")
        self.assertEqual(pull.classify("b.att0.stderr", names), "skip")
        self.assertEqual(pull.classify("runner-1.log.jsonl", names), "log")
        self.assertEqual(pull.classify(".tmp-a.json.1.2", names), "skip")
        self.assertEqual(pull.classify("design.json", names), "evidence")


class AtomicRecords(unittest.TestCase):
    def test_write_new_is_atomic_and_exclusive(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "r.json")
            runner.write_new(p, {"x": 1})
            with self.assertRaises(FileExistsError):
                runner.write_new(p, {"x": 2})
            self.assertEqual(runner.read_json(p), {"x": 1})
            self.assertEqual(os.listdir(d), ["r.json"])            # no temporaries left behind
        finally:
            shutil.rmtree(d)


class MacRoot(unittest.TestCase):
    def test_env_override_names_the_mac_results_dir(self):
        old, pull.MAC_RESULTS = pull.MAC_RESULTS, None
        os.environ["CPU1BIT_MAC_ROOT"] = "/Users/mac/root"
        try:
            self.assertEqual(str(pull.mac_results()), "/Users/mac/root/results")
        finally:
            os.environ.pop("CPU1BIT_MAC_ROOT")
            pull.MAC_RESULTS = old


if __name__ == "__main__":
    unittest.main()
