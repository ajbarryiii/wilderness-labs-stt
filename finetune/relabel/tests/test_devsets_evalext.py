"""eval-ext contracts (the held-out extension set, eval only): its parts sit outside the default dev sets;
pre-cut clips without an id column get row-position ids; captioned lectures (media + SubRip) become
long-form recordings whose transcript is the caption text without speaker labels; archive downloads are
checked against the pinned size, MD5 and SHA-1.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p "test_devsets_*" -v
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import unittest
import uuid
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

import devsets
from devsets import fetch, layout, sources

REAL_ROOT = layout.root("RELABEL_ROOT")

SRT = """1
00:00:00,000 --> 00:00:03,500
The following content is provided
under a Creative Commons license.

2
00:00:03,500 --> 00:00:06,000
PROFESSOR: So today we talk about <i>insulin</i>.

3
00:00:06,000 --> 00:00:08,000
[LAUGHTER]

4
00:00:08,000 --> 00:00:10,000
AUDIENCE: Is it a hormone?
DNA replication -- right.
"""


class SetsTest(unittest.TestCase):
    def test_eval_ext_is_a_set_outside_the_default_scoring(self) -> None:
        self.assertEqual(devsets.SETS["eval-ext"], ("medical-b", "lectures-a"))
        self.assertEqual(devsets.DEFAULT_SETS, ("dev-medical", "dev-technical"))
        self.assertNotIn("eval-ext", devsets.DEFAULT_SETS)
        self.assertEqual(devsets.set_of("lectures-a"), "eval-ext")


class SrtTest(unittest.TestCase):
    def test_caption_text_without_speaker_labels(self) -> None:
        self.assertEqual(sources.srt_text(SRT),
                         "The following content is provided under a Creative Commons license. "
                         "So today we talk about <i>insulin</i>. [LAUGHTER] Is it a hormone? "
                         "DNA replication -- right.")


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        layout.require_mount()
        self.tmp = REAL_ROOT / "tmp" / "tests" / f"evalext-{uuid.uuid4().hex[:8]}"
        self.tmp.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)


class RowIdTest(Fixture):
    def test_row_position_ids_are_unique_across_files(self) -> None:
        files = []
        for shard in range(2):
            buf = io.BytesIO()
            sf.write(buf, np.zeros(16000 * 2, dtype=np.float32), 16000, format="FLAC")
            path = self.tmp / f"test-0000{shard}-of-00002.parquet"
            pq.write_table(pa.table({"audio": [{"bytes": buf.getvalue(), "path": "0.0_2.0.ogg"}] * 3,
                                     "text": ["a b c"] * 3}), path, row_group_size=2)
            files.append(path)
        src = sources.Source("medical-b", {"layout": "parquet-clips", "id_column": sources.ROW_ID,
                                           "audio_column": "audio", "text_column": "text"}, files)
        items, drops = src.items()
        self.assertEqual([i.id for i in items], [f"test-0000{s}-of-00002-r{r:05d}" for s in range(2) for r in range(3)])
        self.assertEqual(drops, {})
        rec = src.load(items[4])
        self.assertEqual((rec.id, rec.transcript[0].text), ("test-00001-of-00002-r00001", "a b c"))


class LectureTest(Fixture):
    def test_media_with_captions_become_long_form_recordings(self) -> None:
        try:
            ffmpeg = sources.ffmpeg_path()
        except RuntimeError:
            self.skipTest("no ffmpeg ($FFMPEG or $RELABEL_ROOT/deps/ffmpeg-headless-bin)")
        wav = self.tmp / "tone.wav"
        t = np.arange(int(44100 * 3.0)) / 44100
        sf.write(wav, np.stack([0.1 * np.sin(2 * np.pi * 300 * t)] * 2, axis=1), 44100)
        media = self.tmp / "item" / "lec01_300k.mp4"
        media.parent.mkdir()
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(wav), "-c:a", "aac", str(media)], check=True,
                       env=sources.ffmpeg_env())
        (media.parent / "lec01_300k.srt").write_text(SRT)
        orphan = media.parent / "lec02_300k.mp4"
        shutil.copy(media, orphan)
        src = sources.Source("lectures-a", {"layout": "srt-longform", "media_suffix": ".mp4"},
                             [media, media.with_suffix(".srt"), orphan])
        items, _ = src.items()
        self.assertEqual([i.id for i in items], ["lec01_300k", "lec02_300k"])
        rec = src.load(items[0])
        self.assertIsInstance(rec.transcript, str)
        self.assertTrue(rec.transcript.startswith("The following content") and "PROFESSOR" not in rec.transcript)
        self.assertAlmostEqual(len(rec.audio) / 16000, 3.0, delta=0.1)
        self.assertEqual(rec.audio.dtype, np.float32)
        with self.assertRaisesRegex(ValueError, "missing_transcript"):
            src.load(items[1])


class ArchiveDownloadTest(Fixture):
    def test_download_is_checked_against_pinned_hashes(self) -> None:
        data = os.urandom(5000)
        origin = self.tmp / "origin.bin"
        origin.write_bytes(data)
        pinned = {"size": len(data), "md5": hashlib.md5(data).hexdigest(), "sha1": hashlib.sha1(data).hexdigest()}
        dest = self.tmp / "got" / "x.bin"
        fetch.download_checked(origin.as_uri(), dest, **pinned)
        self.assertEqual(dest.read_bytes(), data)
        for key, bad in (("md5", "0" * 32), ("sha1", "0" * 40), ("size", 1)):
            other = self.tmp / "got" / f"bad-{key}.bin"
            with self.assertRaises(OSError):
                fetch.download_checked(origin.as_uri(), other, **{**pinned, key: bad}, attempts=1)
            self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
