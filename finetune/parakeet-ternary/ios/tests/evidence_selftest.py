"""Archive/eligibility safety tests: no models, gates, sweeps or training."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

IOS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IOS))
import artifacts
from mil import evidence, eligibility


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        # Real C4 passing record and all prerequisites, isolated on the artifact disk.
        self.original_ios = evidence.IOS
        self.entry_map = evidence.index()["records"]
        self.key = "results/eligibility/mp2-C4-multi-ane.json"
        self.rec = json.loads(evidence.read_text(IOS / self.key))
        self.data = {k: evidence.read_bytes(IOS / k) for k in [self.key, *self.rec["inputs"]]}
        self.tmp = Path(tempfile.mkdtemp(prefix="evidence-test-", dir=artifacts.root()))
        self.ios = self.tmp / "ios"
        self.archive = self.tmp / "archive"
        self.archive.mkdir()
        records = {}
        for k, data in self.data.items():
            p = self.ios / k
            p.parent.mkdir(parents=True, exist_ok=True)
            if k.startswith("results/"):
                entry = self.entry_map[k]
                dest = self.archive / entry["archive"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                records[k] = entry
                p.write_bytes(evidence.summary(data, entry))
            else:
                p.write_bytes(data)
        (self.ios / "results/INDEX.json").write_bytes(evidence.index_bytes({"version": 1, "records": records}))
        self.patches = [
            patch.object(evidence, "IOS", self.ios), patch.object(evidence, "RESULTS", self.ios / "results"),
            patch.object(evidence, "INDEX", self.ios / "results/INDEX.json"),
            patch.object(eligibility, "IOS", self.ios), patch.object(eligibility, "ELIG", self.ios / "results/eligibility"),
            patch.dict(os.environ, {"IOS_RESULTS_ROOT": str(self.archive)}),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        shutil.rmtree(self.tmp)

    def check(self):
        return eligibility.check("mp2", "C4", "multi", "ane")

    def archive_path(self, key):
        return self.archive / self.entry_map[key]["archive"]

    def test_archived_record_passes(self):
        self.assertTrue(self.check()["timing_allowed"])
        evidence.verify()

    def test_missing_record_refused(self):
        self.archive_path(self.key).unlink()
        with self.assertRaises(eligibility.Ineligible):
            self.check()

    def test_tampered_record_refused(self):
        self.archive_path(self.key).write_bytes(b"tampered")
        with self.assertRaises(eligibility.Ineligible):
            self.check()

    def test_missing_input_refused(self):
        key = next(k for k in self.rec["inputs"] if k.startswith("results/"))
        self.archive_path(key).unlink()
        with self.assertRaises(eligibility.Ineligible):
            self.check()

    def test_tampered_input_refused(self):
        key = next(k for k in self.rec["inputs"] if k.startswith("results/"))
        self.archive_path(key).write_bytes(b"tampered")
        with self.assertRaises(eligibility.Ineligible):
            self.check()

    def test_stale_input_digest_refused(self):
        # Publish changed input with a valid new index digest; the old eligibility
        # record must still refuse it using its independently embedded input digest.
        key = next(k for k in self.rec["inputs"] if k.startswith("results/"))
        evidence.write_text(self.ios / key, '{"changed": true}\n')
        with self.assertRaises(eligibility.Ineligible):
            self.check()

    def test_summary_never_authorizes_timing(self):
        (self.ios / self.key).write_text('{"timing_allowed": true}\n')
        self.archive_path(self.key).unlink()
        with self.assertRaises(eligibility.Ineligible):
            self.check()

    def test_writer_and_withdrawal(self):
        p = self.ios / "results/gates/new.json"
        text = json.dumps({"pass": True, "cases": list(range(10000))})
        evidence.write_text(p, text)
        self.assertEqual(evidence.read_text(p), text)
        self.assertLess(p.stat().st_size, len(text))
        self.assertIn(p, list(evidence.glob(p.parent, "*.json")))
        evidence.verify()
        evidence.unlink(p)
        self.assertFalse(evidence.exists(p))
        p.write_text('{"pass": true}')  # stale display cannot resurrect it
        with self.assertRaises(evidence.EvidenceError):
            evidence.read_text(p)

    def test_export_is_consistent_during_publication(self):
        import tarfile
        import hashlib
        bundle = self.tmp / "interleaved.tar.gz"
        original = evidence._read_entry
        changed = False
        def interleave(key, entry):
            nonlocal changed
            if not changed:
                changed = True
                p = self.ios / next(k for k in self.rec["inputs"] if k.startswith("results/"))
                evidence.write_text(p, '{"new": true}')
            return original(key, entry)
        with patch.object(evidence, "_read_entry", side_effect=interleave):
            evidence.export_bundle(bundle)
        with tarfile.open(bundle) as tar:
            doc = json.load(tar.extractfile("INDEX.json"))
            for entry in doc["records"].values():
                data = tar.extractfile(entry["archive"]).read()
                self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
                self.assertEqual(len(data), entry["bytes"])

    def test_bundle_roundtrip_and_collision(self):
        bundle = self.tmp / "bundle.tar.gz"
        evidence.export_bundle(bundle)
        target = self.tmp / "imported"
        with patch.dict(os.environ, {"IOS_RESULTS_ROOT": str(target)}):
            evidence.import_bundle(bundle)
            self.assertTrue(self.check()["timing_allowed"])
            evidence.verify()
            (target / self.entry_map[self.key]["archive"]).write_bytes(b"collision")
            with self.assertRaises(evidence.EvidenceError):
                evidence.import_bundle(bundle)

    def test_nonfinite_failure_archived(self):
        p = self.ios / "results/gates/failure.json"
        text = json.dumps({"pass": False, "rel": float("inf"), "abs": float("nan")})
        evidence.write_text(p, text)
        self.assertEqual(evidence.read_text(p), text)
        self.assertEqual(json.loads(p.read_text())["rel"], "inf")
        evidence.verify()

    def test_aggregate_rows_retained(self):
        data = json.dumps({"rows": [{"arm": str(i)} for i in range(1000)]}).encode()
        entry = {"sha256": "a" * 64, "bytes": len(data), "archive": "commit/wp5/sweep.json"}
        display = json.loads(evidence.summary(data, entry))
        self.assertEqual(len(display["rows"]), 1000)

    def test_bad_root_refused(self):
        with patch.dict(os.environ, {"IOS_RESULTS_ROOT": str(self.ios)}):
            with self.assertRaises(RuntimeError):
                evidence.root()

    def test_quarantine_refused(self):
        with self.assertRaises(evidence.EvidenceError):
            evidence.read_text(self.ios / "results/eligibility/pipelines/quarantine/fake.json")

    def test_escape_refused(self):
        doc = evidence.index()
        doc["records"][self.key]["archive"] = "../outside.json"
        evidence.INDEX.write_bytes(evidence.index_bytes(doc))
        with self.assertRaises(eligibility.Ineligible):
            self.check()


if __name__ == "__main__":
    unittest.main(verbosity=2)
