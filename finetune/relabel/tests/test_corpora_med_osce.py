"""med-osce contracts (labeling plan "New corpora"):

- fetch once: the archive is verified against the registry's size and MD5 (a mismatch is refused and nothing
  is extracted); only the layout's audio and transcript members are written; a done marker makes it once;
- references: speaker labels (with the variants seen in the data) are removed, continuation lines kept;
- segmentation like Part A: 1-30 s segments with stable ids "med-osce:<recording>-<k>", one FLAC per
  segment whose length is the manifest duration, drops counted; the manifest is frozen (a built corpus is
  never re-segmented) and its SHA-256 is recorded in status.json;
- every record carries its aligner's identity; a record made by another aligner is re-segmented, never
  reused, and the frozen meta and status name the one aligner that made the whole manifest; --batch caps
  the recordings segmented per call and the manifest is frozen only when all are done;
- the device the aligner's emissions were computed on (identity "device") is not part of that identity:
  records that differ only in it are reused (a CPU run resumed on the GPU), the meta and status name the
  aligner without a device, and count the recordings per device ("aligner_devices").

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_corpora_*' -v
"""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from corpora import med_osce
from devsets.align import UnitSpan

LAYOUT = {"audio_dir": "Data/Audio Recordings", "audio_ext": ".wav", "transcript_dir": "Data/Clean Transcripts",
          "transcript_ext": ".txt"}
SENTENCES = ["Hello there, what brings you in today?", "I have had chest pain since last night.",
             "Where is the pain located?", "It is on the left side of my chest.",
             "Does anything make it better or worse?", "Lying down makes it worse."]


class FakeAligner:
    """Spreads the units evenly over the audio (score 1.0), like a perfect aligner on uniform speech."""
    min_score = 0.35

    def __init__(self, name: str = "fake-A", device: str | None = None):
        self.identity = {"aligner": name, **({"device": device} if device else {})}
        self.calls = 0

    def align(self, audio, units):
        self.calls += 1
        d = len(audio) / 16000
        n = len(units)
        return [UnitSpan(i * d / n, (i + 0.8) * d / n, None if u is None else 1.0) for i, u in enumerate(units)]


def transcript(k: int) -> str:
    lines = []
    for i in range(k):
        lab = "D:" if i % 2 == 0 else "P;"
        lines.append(f"{lab} {SENTENCES[i % len(SENTENCES)]}")
        lines.append("")
    return "\n".join(lines)


def make_zip(path: Path, recs: dict[str, tuple[float, str]], extra: dict[str, bytes] | None = None) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for rec, (seconds, text) in recs.items():
            buf = io.BytesIO()
            rng = np.random.default_rng(len(rec))
            sf.write(buf, (0.1 * rng.standard_normal(int(seconds * 22050))).astype(np.float32), 22050,
                     format="WAV", subtype="PCM_16")
            z.writestr(f"Data/Audio Recordings/{rec}.wav", buf.getvalue())
            z.writestr(f"Data/Clean Transcripts/{rec}.txt", text.encode())
        for name, data in (extra or {}).items():
            z.writestr(name, data)


def registry_for(path: Path) -> dict:
    data = path.read_bytes()
    return {"file": "Data.zip", "download_url": "unused", "size": len(data),
            "md5": hashlib.md5(data).hexdigest(), "layout": LAYOUT}


class ReferenceTextTest(unittest.TestCase):
    def test_labels_removed_continuations_kept(self):
        raw = ("D: What brought you in?\n\nP; Chest pain,\nsince last night.\nD\" OK.\nDL OK, any allergies?\n"
               "P No.\np: Fine.\nDo you smoke?\nD-dimer was normal.\n")
        text, counts = med_osce.reference_text(raw)
        self.assertEqual(text, "What brought you in? Chest pain, since last night. OK. OK, any allergies? No. Fine. "
                               "Do you smoke? D-dimer was normal.")
        self.assertEqual(counts, {"labelled_lines": 6, "continuation_lines": 3})

    def test_cp1252_fallback(self):
        self.assertEqual(med_osce.decode_text("D: café ’".encode("cp1252")), "D: café ’")


class FetchSegmentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.src = self.d / "src.zip"
        make_zip(self.src, {"CAR0001": (40.0, transcript(12)), "RES0002": (25.0, transcript(6)),
                            "ONLYAUDIO": (5.0, "")},
                 extra={"Data/Other/readme.txt": b"x", "Data/Audio Recordings/../evil.wav": b"x"})
        with zipfile.ZipFile(self.src, "a") as z:   # a transcript without audio
            z.writestr("Data/Clean Transcripts/ONLYTEXT.txt", b"D: hi")
        self.reg = registry_for(self.src)
        self.cdir = self.d / "med-osce"
        self.downloads = 0

    def tearDown(self):
        self.tmp.cleanup()

    def downloader(self, url, dest):
        self.downloads += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.src.read_bytes())

    def test_fetch_verifies_and_extracts_layout_only(self):
        s = med_osce.fetch(self.reg, self.cdir, self.downloader)
        self.assertEqual(s["paired"], 3)
        self.assertEqual(s["transcript_only"], ["ONLYTEXT"])
        self.assertEqual(s["md5"], self.reg["md5"])
        written = sorted(p.relative_to(self.cdir / "raw" / "extracted").as_posix()
                         for p in (self.cdir / "raw" / "extracted").rglob("*") if p.is_file())
        self.assertEqual(written, ["audio/CAR0001.wav", "audio/ONLYAUDIO.wav", "audio/RES0002.wav",
                                   "transcripts/CAR0001.txt", "transcripts/ONLYAUDIO.txt",
                                   "transcripts/ONLYTEXT.txt", "transcripts/RES0002.txt"])
        med_osce.fetch(self.reg, self.cdir, self.downloader)
        self.assertEqual(self.downloads, 1)

    def test_fetch_refuses_bad_md5(self):
        bad = dict(self.reg, md5="0" * 32)
        with self.assertRaises(RuntimeError):
            med_osce.fetch(bad, self.cdir, self.downloader)
        self.assertFalse((self.cdir / "raw" / "extracted").exists())
        self.assertFalse((self.cdir / "raw" / "fetch.done.json").exists())

    def test_segments_manifest_frozen(self):
        med_osce.fetch(self.reg, self.cdir, self.downloader)
        status = med_osce.segment_all(self.cdir, FakeAligner())
        self.assertEqual(status["status"], "built")
        manifest = self.cdir / "manifest.jsonl"
        self.assertEqual(status["sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
        rows = [json.loads(x) for x in manifest.read_text().splitlines()]
        self.assertTrue(rows)
        self.assertEqual([r["id"] for r in rows], sorted(r["id"] for r in rows))
        self.assertEqual({r["recording"] for r in rows}, {"CAR0001", "RES0002"})
        for r in rows:
            self.assertRegex(r["id"], r"^med-osce:(CAR0001|RES0002)-\d{4}$")
            self.assertTrue(1.0 <= r["duration"] <= 30.0)
            info = sf.info(r["audio_filepath"])
            self.assertEqual((info.samplerate, info.channels), (16000, 1))
            self.assertAlmostEqual(info.frames / 16000, r["duration"], places=6)
            self.assertNotRegex(r["text"], r"(^|\s)[DP][:;]")
            self.assertEqual(r["source"], "med-osce")
        joined = " ".join(r["text"] for r in rows if r["recording"] == "RES0002")
        self.assertEqual(joined, " ".join(SENTENCES))
        meta = json.loads((self.cdir / "manifest.meta.json").read_text())
        self.assertEqual(meta["segments"], len(rows))
        self.assertIn("empty_reference", meta["drops"])   # ONLYAUDIO has an empty transcript
        before = manifest.stat().st_mtime_ns
        self.assertEqual(med_osce.segment_all(self.cdir, None), status)   # frozen: never re-segmented
        self.assertEqual(manifest.stat().st_mtime_ns, before)

    def test_aligner_identity_and_batches(self):
        med_osce.fetch(self.reg, self.cdir, self.downloader)
        part = med_osce.segment_all(self.cdir, FakeAligner("fake-A"), batch=1)
        self.assertEqual(part["status"], "partial")
        self.assertFalse((self.cdir / "manifest.jsonl").exists())
        self.assertEqual(len(list((self.cdir / "records").glob("*.json"))), 1)
        status = med_osce.segment_all(self.cdir, FakeAligner("fake-B"))
        self.assertEqual(status["status"], "built")
        self.assertEqual(status["aligner"], {"aligner": "fake-B"})
        for r in (self.cdir / "records").glob("*.json"):
            self.assertEqual(json.loads(r.read_text())["aligner"], {"aligner": "fake-B"})
        meta = json.loads((self.cdir / "manifest.meta.json").read_text())
        self.assertEqual(meta["aligner"], {"aligner": "fake-B"})

    def test_mixed_aligners_refused(self):
        med_osce.fetch(self.reg, self.cdir, self.downloader)
        med_osce.segment_all(self.cdir, FakeAligner("fake-A"))
        (self.cdir / "status.json").unlink()
        rec = self.cdir / "records" / "CAR0001.json"
        r = json.loads(rec.read_text())
        r["aligner"] = {"aligner": "other"}
        rec.write_text(json.dumps(r))
        with self.assertRaises(RuntimeError):
            med_osce.segment_all(self.cdir, None)

    def test_resume_reproduces_manifest(self):
        med_osce.fetch(self.reg, self.cdir, self.downloader)
        first = med_osce.segment_all(self.cdir, FakeAligner())
        text = (self.cdir / "manifest.jsonl").read_text()
        (self.cdir / "status.json").unlink()
        (self.cdir / "manifest.jsonl").unlink()
        again = med_osce.segment_all(self.cdir, None)   # all records exist: no alignment needed
        self.assertEqual((self.cdir / "manifest.jsonl").read_text(), text)
        self.assertEqual(again["sha256"], first["sha256"])

    def test_device_is_not_part_of_the_aligner_identity(self):
        med_osce.fetch(self.reg, self.cdir, self.downloader)
        part = med_osce.segment_all(self.cdir, FakeAligner("fake-A", "cpu"), batch=1)
        self.assertEqual(part["status"], "partial")
        (first,) = (self.cdir / "records").glob("*.json")
        before = first.read_bytes()
        gpu = FakeAligner("fake-A", "cuda")
        status = med_osce.segment_all(self.cdir, gpu)   # resumes: the CPU record is reused, not re-segmented
        self.assertEqual(status["status"], "built")
        self.assertEqual(first.read_bytes(), before)
        records = {r.stem: json.loads(r.read_text()) for r in (self.cdir / "records").glob("*.json")}
        self.assertEqual(len(records), 3)
        self.assertEqual(gpu.calls, sum(1 for r in records.values() if r["aligner"]["device"] == "cuda"
                                        and r["rows"]))
        self.assertEqual(records[first.stem]["aligner"], {"aligner": "fake-A", "device": "cpu"})
        meta = json.loads((self.cdir / "manifest.meta.json").read_text())
        for doc in (status, meta):
            self.assertEqual(doc["aligner"], {"aligner": "fake-A"})
            self.assertEqual(doc["aligner_devices"], {"cpu": 1, "cuda": 2})

    def test_other_identity_change_still_resegments_across_devices(self):
        med_osce.fetch(self.reg, self.cdir, self.downloader)
        med_osce.segment_all(self.cdir, FakeAligner("fake-A", "cpu"), batch=1)
        status = med_osce.segment_all(self.cdir, FakeAligner("fake-B", "cuda"))
        self.assertEqual(status["aligner"], {"aligner": "fake-B"})
        self.assertEqual(status["aligner_devices"], {"cuda": 3})


if __name__ == "__main__":
    unittest.main()
