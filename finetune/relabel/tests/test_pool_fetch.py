"""Text cache contracts (labeling plan Part B "Text"; ids and file lists of parakeet-ternary stream.py).

- One cache row per source row, with its row group, recording id, order, duration in seconds, raw text and
  normalized text (evaluation normalizer).
- Recordings: yodas = original_audio_id ordered by original_audio_offset; peoples_speech = the row id
  without its '_<segment>.flac' suffix, ordered by segment.
- Resumable: a file with a done marker is not read again; a failing file gets no marker.
- The production file lists are the stream's (same files_sha256 as its sources.json).

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_pool_*' -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

import evaluate
from pool import common, fetch
from test_pool_fixtures import source_parquet


class FetchFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_yodas_rows(self):
        src = self.dir / "y.parquet"
        rows = source_parquet(src, "yodas", [("vidA", ["Hello, Dr. Smith!", "It costs 5 dollars."]),
                                             ("vid_B-x", ["Second recording.", "", "More words here."])],
                              rows_per_group=2)
        entry = {"path": "data/en000/asr_only/00000001.parquet", "local_path": str(src)}
        out = self.dir / "cache"
        s = fetch.fetch_file("yodas", entry, out)
        cache = pq.read_table(out / "en000-00000001.parquet").to_pylist()
        self.assertEqual(len(cache), len(rows))
        self.assertEqual(s["rows"], len(rows))
        for i, (c, r) in enumerate(zip(cache, rows)):
            self.assertEqual(c["row"], i)
            self.assertEqual(c["row_group"], i // 2)
            self.assertEqual(c["recording"], r["original_audio_id"])
            self.assertEqual(c["order"], r["original_audio_offset"])
            self.assertAlmostEqual(c["duration"], r["duration"], places=4)
            self.assertEqual(c["text"], r["text"].strip())
            self.assertEqual(c["norm"], evaluate.normalize(r["text"]).strip() if r["text"].strip() else "")
        self.assertEqual(s["empty_text_rows"], 1)
        self.assertEqual(s["recordings"], 2)

    def test_peoples_speech_recording_and_seconds(self):
        src = self.dir / "p.parquet"
        rows = source_parquet(src, "peoples_speech", [("itemA_SLASH_fileA_DOT_mp3", ["a b", "c d", "e"]),
                                                      ("itemA_SLASH_fileB_DOT_mp3", ["f"])], rows_per_group=3)
        entry = {"path": "clean/train-00001-of-00804.parquet", "local_path": str(src)}
        fetch.fetch_file("peoples_speech", entry, self.dir / "cache")
        cache = pq.read_table(self.dir / "cache" / "train-00001-of-00804.parquet").to_pylist()
        self.assertEqual([c["recording"] for c in cache], ["itemA_SLASH_fileA_DOT_mp3"] * 3 + ["itemA_SLASH_fileB_DOT_mp3"])
        self.assertEqual([c["order"] for c in cache], [0.0, 1.0, 2.0, 0.0])
        for c, r in zip(cache, rows):
            self.assertAlmostEqual(c["duration"], r["duration_ms"] / 1000, places=3)

    def test_unparseable_id_fails_without_marker(self):
        src = self.dir / "p.parquet"
        source_parquet(src, "peoples_speech", [("x", ["a"])])
        import pyarrow as pa
        t = pq.read_table(src).set_column(0, "id", pa.array(["no-suffix"]))
        pq.write_table(t, src)
        entry = {"path": "clean/train-00002-of-00804.parquet", "local_path": str(src)}
        with self.assertRaises(ValueError):
            fetch.fetch_file("peoples_speech", entry, self.dir / "cache")
        self.assertFalse((self.dir / "cache" / "train-00002-of-00804.done.json").exists())

    def test_done_marker_skips_reading(self):
        src = self.dir / "y.parquet"
        source_parquet(src, "yodas", [("v", ["a b c"])])
        entry = {"path": "data/en000/asr_only/00000003.parquet", "local_path": str(src)}
        first = fetch.fetch_file("yodas", entry, self.dir / "cache")
        os.remove(src)
        self.assertEqual(fetch.fetch_file("yodas", entry, self.dir / "cache"), first)

    def test_run_fetch_resumes_and_pins_listing(self):
        entries = []
        for i in range(3):
            src = self.dir / f"y{i}.parquet"
            source_parquet(src, "yodas", [(f"v{i}", ["words here"])], seed=i)
            entries.append({"path": f"data/en000/asr_only/0000001{i}.parquet", "local_path": str(src)})
        out = self.dir / "cache"
        (out).mkdir()
        # A failing file: no marker, reported, others done.
        bad = {"path": "data/en000/asr_only/00000019.parquet", "local_path": str(self.dir / "missing.parquet")}
        status = fetch.run_fetch("yodas", entries + [bad], out, workers=0, threads=1)
        self.assertEqual(status["done"], 3)
        self.assertEqual([f["path"] for f in status["failed"]], [bad["path"]])
        self.assertTrue(json.loads((out / "files.json").read_text())["listing"])
        with self.assertRaises(RuntimeError):
            fetch.run_fetch("yodas", entries, out, workers=0, threads=1)  # a different list, same directory


class ProductionListsTest(unittest.TestCase):
    """The production file lists equal the stream's (reads the cached listings; no network)."""

    def test_same_files_as_the_stream(self):
        recorded = json.loads((common.root("PARAKEET_TERNARY_ROOT") / "data" / "stream" / "sources.json").read_text())
        for source in ("yodas", "peoples_speech"):
            files = common.training_files(source)
            self.assertEqual(len(files), recorded["sources"][source]["files"])
            self.assertEqual(common.stream_files_sha256(source, files), recorded["sources"][source]["files_sha256"])
        import data
        all_files, reserved = data.yodas_reserved()
        reserved_paths = {sorted(f["path"] for f in all_files)[i] for i in reserved}
        self.assertFalse(reserved_paths & {f["path"] for f in common.training_files("yodas")})


if __name__ == "__main__":
    unittest.main()
