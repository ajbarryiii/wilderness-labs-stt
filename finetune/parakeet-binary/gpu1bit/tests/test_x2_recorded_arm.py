"""Contract tests of gpu1bit/x2/recorded_arm.py (plans/x2-default.md Z4, Z7; CPU):
  - The real pinned arms of ap-20261020r4 (M1-x2, MB1-int8_mma-x2) verify as recorded. This includes
    Z7's case: M1-x2's last chunk is recorded at 21:49:36, and yodas_dev.jsonl has ctime 21:49:36.79,
    which is accepted (cutoff finished + 1 s, exclusive). The identity is not recomputed from current
    sources.
  - Refusals, on a copy of a real arm:
    - a modified row file (pin mismatch);
    - a row file not pinned;
    - a mixed identity (a row of another identity);
    - a summary of another identity;
    - an incomplete arm (a missing row, an incomplete summary);
    - a row file changed after the last chunk + 1 s;
    - an x3 reference that did not pass R2;
    - a reference whose pinned summary changed.
  - The cross-run comparison stores both runs and both identities and is descriptive only.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_x2_recorded_arm
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from gpu1bit.shared import env, fulldev
from gpu1bit.x2 import recorded_arm as RA

RUN = "ap-20261020r4"


class RealPinnedArms(unittest.TestCase):
    def test_pins_cover_the_compared_arms(self):
        pins = RA.plan_pins()
        for arm in ("M1-x2", "MB1-int8_mma-x2"):
            for n in fulldev.DEV_SETS:
                self.assertIn(f"{RUN}/{arm}/{n}.jsonl", pins)
        for ref in ("M1-x3", "MB1-int8_mma-x3"):
            self.assertIn(f"{RUN}/{ref}/summary.json", pins)

    def test_real_arms_verify_including_the_z7_case(self):
        pins = RA.plan_pins()
        for arm in ("M1-x2", "MB1-int8_mma-x2"):
            with self.subTest(arm=arm):
                out = RA.verify_recorded_arm(RUN, arm, pins)
                self.assertEqual(sum(len(r) for r in out["rows"].values()), 22418)
                self.assertIn("not recomputed", out["binding"])
        d = fulldev.ROOT / RUN / "M1-x2"
        chunks = [json.loads(l) for l in (d / "chunks.jsonl").read_text().splitlines() if l.strip()]
        self.assertEqual(chunks[-1]["finished"], "2026-10-20T21:49:36")
        ct = os.stat(d / "yodas_dev.jsonl").st_ctime
        self.assertGreater(ct, time.mktime(time.strptime("2026-10-20T21:49:36", "%Y-%m-%dT%H:%M:%S")))


class Refusals(unittest.TestCase):
    """Each case works on a copy of the real arms (rows re-pinned for the copy unless the case is a pin
    mismatch); the copy is rewritten, so the ctime cutoff is moved with an injected chunk time."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(dir=str(env.ARTIFACTS_ROOT / "H")))
        for arm in ("M1-x2", "M1-x3"):
            shutil.copytree(fulldev.ROOT / RUN / arm, cls.tmp / RUN / arm)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.work = Path(tempfile.mkdtemp(dir=str(self.tmp)))
        for arm in ("M1-x2", "M1-x3"):
            shutil.copytree(self.tmp / RUN / arm, self.work / RUN / arm)
        self.d = self.work / RUN / "M1-x2"
        self.future_chunk()

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def future_chunk(self):
        """The copy's files are new; move the recorded chunk end after them (setup only)."""
        p = self.d / "chunks.jsonl"
        chunks = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        chunks[-1]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() + 120))
        p.write_text("".join(json.dumps(c) + "\n" for c in chunks))

    def pins(self):
        out = {}
        for arm in ("M1-x2", "M1-x3"):
            for f in (self.work / RUN / arm).iterdir():
                if f.suffix in (".jsonl", ".json") and f.name != "chunks.jsonl":
                    out[f"{RUN}/{arm}/{f.name}"] = RA._sha(f)
        return out

    def verify(self, pins=None):
        return RA.verify_recorded_arm(RUN, "M1-x2", self.pins() if pins is None else pins, root=self.work)

    def test_copy_verifies(self):
        self.verify()

    def test_modified_row_file(self):
        pins = self.pins()
        p = self.d / "ami_dev.jsonl"
        rows = p.read_text().splitlines()
        r = json.loads(rows[5])
        r["hyp"] = r["hyp"] + " x"
        rows[5] = json.dumps(r)
        p.write_text("\n".join(rows) + "\n")
        with self.assertRaisesRegex(ValueError, "pin"):
            self.verify(pins)

    def test_unpinned_row_file(self):
        pins = self.pins()
        del pins[f"{RUN}/M1-x2/yodas_dev.jsonl"]
        with self.assertRaisesRegex(ValueError, "not pinned"):
            self.verify(pins)

    def test_mixed_identity(self):
        p = self.d / "voxpopuli_dev.jsonl"
        rows = p.read_text().splitlines()
        r = json.loads(rows[0])
        r["identity"] = "0" * 64
        rows[0] = json.dumps(r)
        p.write_text("\n".join(rows) + "\n")
        with self.assertRaisesRegex(ValueError, "other identities"):
            self.verify()

    def test_summary_of_another_identity(self):
        s = json.loads((self.d / "summary.json").read_text())
        s["identity"] = "1" * 64
        (self.d / "summary.json").write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError, "summary identity"):
            self.verify()

    def test_incomplete_arm(self):
        p = self.d / "yodas_dev.jsonl"
        p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "rows"):
            self.verify()
        shutil.copy(self.tmp / RUN / "M1-x2" / "yodas_dev.jsonl", p)
        s = json.loads((self.d / "summary.json").read_text())
        s["complete"] = False
        (self.d / "summary.json").write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError, "complete"):
            self.verify()

    def test_row_file_changed_after_the_last_chunk(self):
        p = self.d / "chunks.jsonl"
        chunks = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        chunks[-1]["finished"] = "2026-10-20T21:49:36"           # the recorded time; the copy is newer
        p.write_text("".join(json.dumps(c) + "\n" for c in chunks))
        with self.assertRaisesRegex(ValueError, "after the last chunk"):
            self.verify()

    def test_reference_without_r2(self):
        rp = self.work / RUN / "M1-x3" / "summary.json"
        s = json.loads(rp.read_text())
        s["cross_check"]["pass"] = False
        rp.write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError, "R2"):
            self.verify()

    def test_reference_pin_mismatch(self):
        pins = self.pins()
        rp = self.work / RUN / "M1-x3" / "summary.json"
        rp.write_text(rp.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "reference"):
            self.verify(pins)


class CrossRun(unittest.TestCase):
    """A current arm (a fresh run with the recorded rows re-stamped under a current identity) against the
    recorded arm: both runs and identities are stored; identical rows give 0 differences."""

    def setUp(self):
        self.run = f"test-x2-crossrun-{os.getpid()}"

    def tearDown(self):
        shutil.rmtree(fulldev.ROOT / self.run, ignore_errors=True)

    def test_both_identities_are_stored(self):
        from gpu1bit.actprec import parts
        with parts.installed():
            h = fulldev.freeze(self.run, "M1-x2", "full")
            for n in fulldev.DEV_SETS:
                rows = fulldev.load_rows(fulldev.ROOT / RUN / "M1-x2" / f"{n}.jsonl")
                with open(fulldev.work_path(self.run, "M1-x2", n), "w") as f:
                    for r in rows:
                        f.write(json.dumps(dict(r, identity=h)) + "\n")
            out = RA.compare_with_recorded(self.run, "M1-x2", RUN, "M1-x2")
        self.assertTrue(out["descriptive_only"])
        self.assertEqual(out["current"], {"run": self.run, "arm": "M1-x2", "identity": h})
        self.assertEqual(out["recorded"]["run"], RUN)
        self.assertEqual(out["recorded"]["identity"], json.loads((fulldev.ROOT / RUN / "M1-x2" / "identity.json").read_text())["hash"])
        self.assertEqual(out["pooled"], {"utterances": 22418, "differing": 0})
