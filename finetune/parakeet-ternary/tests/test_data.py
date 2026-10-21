"""Unit tests for data.py: manifests, selection rules and audio conversion (no network)."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402
import paths  # noqa: E402


def _row(i: str, duration: float = 2.0, text: str | None = "hello") -> dict:
    return {"audio_filepath": f"/abs/{i}.flac", "duration": duration, "text": text, "id": i, "source": "test"}


class TempManifests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(paths, "MANIFESTS", Path(self._tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)


class ManifestTests(TempManifests):
    def test_round_trip_sorted(self):
        rows = [_row("b", 1.5), _row("a", 2.25, None), _row("c", 30.0, "x y")]
        meta = data.write_manifest("rt", rows, {"source": "test"})
        loaded = data.load_manifest("rt")
        self.assertEqual(loaded, sorted(rows, key=lambda r: r["id"]))
        self.assertEqual([r["id"] for r in loaded], ["a", "b", "c"])
        self.assertEqual(meta["utterances"], 3)
        self.assertAlmostEqual(meta["hours"], (1.5 + 2.25 + 30.0) / 3600, places=4)
        for line in (Path(self._tmp.name) / "rt.jsonl").read_text().splitlines():
            self.assertEqual(list(json.loads(line)), list(data.MANIFEST_KEYS))

    def test_meta_sha256_matches_file(self):
        data.write_manifest("sha", [_row("x"), _row("y")])
        meta = data.load_meta("sha")
        self.assertEqual(meta["sha256"], data.sha256_file(Path(self._tmp.name) / "sha.jsonl"))

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            data.write_manifest("dup", [_row("a"), _row("a")])

    def test_bad_rows_rejected(self):
        for bad in ({**_row("a"), "extra": 1}, {**_row("a"), "audio_filepath": "rel.flac"},
                    {**_row("a"), "duration": 0.0}, {**_row("a"), "duration": 2}):
            with self.assertRaises(ValueError):
                data.write_manifest("bad", [bad])

    def test_filter_records_counts_and_dedup(self):
        recs = [{"id": "a", "audio_filepath": "/a.flac", "duration": 0.99, "text": "t", "drop": None},
                {"id": "b", "audio_filepath": "/b.flac", "duration": 1.0, "text": "", "drop": None},
                {"id": "c", "audio_filepath": "/c.flac", "duration": 30.01, "text": "t", "drop": None},
                {"id": "d", "audio_filepath": None, "duration": None, "text": "t", "drop": "decode_error"},
                {"id": "b", "audio_filepath": "/b2.flac", "duration": 5.0, "text": "t", "drop": None},
                {"id": "e", "audio_filepath": "/e.flac", "duration": 5.0, "text": "t", "drop": None,
                 "group": "dev-rec"}]
        rows, drops = data.filter_records(recs, "s", duration_filter=True, exclude_groups={"dev-rec"})
        self.assertEqual([r["id"] for r in rows], ["b"])
        self.assertIsNone(rows[0]["text"])
        self.assertEqual(drops, {"decode_error": 1, "dev_recording_overlap": 1, "duplicate_id": 1,
                                 "too_long": 1, "too_short": 1})
        rows, drops = data.filter_records(recs[:3], "s", duration_filter=False)
        self.assertEqual(len(rows), 3)


class SelectionTests(unittest.TestCase):
    def test_duration_boundaries(self):
        f = data.duration_drop_reason
        self.assertEqual(f(0.999999), "too_short")
        self.assertIsNone(f(1.0))
        self.assertIsNone(f(15.0))
        self.assertIsNone(f(30.0))
        self.assertEqual(f(30.000001), "too_long")
        self.assertEqual(f(0.0), "empty_audio")
        self.assertEqual(f(None), "empty_audio")
        self.assertEqual(f(float("nan")), "empty_audio")

    def test_every_kth_deterministic(self):
        rows = [_row(f"u{i:05d}") for i in range(1003)]
        shuffled = [rows[i] for i in np.random.default_rng(0).permutation(len(rows))]
        a, b = data.every_kth(rows, 400), data.every_kth(shuffled, 400)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 400)
        self.assertEqual([r["id"] for r in a], [f"u{i:05d}" for i in range(0, 800, 2)])
        self.assertEqual(len(data.every_kth(rows[:300], 400)), 300)

    def test_reserved_shards_deterministic(self):
        names = [f"data/en{i // 500:03d}/asr_only/{i % 500:08d}.parquet" for i in range(18496)]
        r1 = data.reserve_shards(names, 16, seed=paths.SEED)
        r2 = data.reserve_shards(list(reversed(names)), 16, seed=paths.SEED)
        self.assertEqual(r1, r2)
        self.assertEqual(len(set(r1)), 16)
        self.assertEqual(r1, sorted(r1))
        self.assertNotEqual(r1, data.reserve_shards(names, 16, seed=paths.SEED + 1))
        with self.assertRaises(ValueError):
            data.reserve_shards(names + names[:1], 16)


class AudioTests(unittest.TestCase):
    @staticmethod
    def _encode(x: np.ndarray, sr: int, fmt: str = "WAV", subtype: str = "PCM_16") -> bytes:
        buf = io.BytesIO()
        sf.write(buf, x, sr, format=fmt, subtype=subtype)
        return buf.getvalue()

    def test_resample_44k_sine_length_and_frequency(self):
        sr, seconds, freq = 44100, 2.0, 1000.0
        t = np.arange(int(sr * seconds)) / sr
        x = 0.5 * np.sin(2 * np.pi * freq * t)
        pcm, reason, passthrough = data.convert(self._encode(np.stack([x, x], 1), sr))
        self.assertIsNone(reason)
        self.assertFalse(passthrough)
        self.assertEqual(pcm.dtype, np.int16)
        self.assertLessEqual(abs(pcm.size - int(seconds * data.SR)), 1)
        y = pcm.astype(np.float64) / 32768
        spectrum = np.abs(np.fft.rfft(y * np.hanning(y.size)))
        peak = np.argmax(spectrum) * data.SR / y.size
        self.assertAlmostEqual(peak, freq, delta=data.SR / y.size)
        self.assertAlmostEqual(np.sqrt(np.mean(y[1000:-1000] ** 2)), 0.5 / np.sqrt(2), delta=0.01)

    def test_resample_function_direct(self):
        x = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050).astype(np.float32)
        y = data.resample(x, 22050, 16000)
        self.assertLessEqual(abs(y.size - 16000), 1)

    def test_flac_passthrough_and_float_conversion(self):
        x = (0.3 * np.sin(np.arange(16000) / 5)).astype(np.float32)
        flac = self._encode(x, 16000, "FLAC", "PCM_16")
        pcm, reason, passthrough = data.convert(flac)
        self.assertIsNone(reason)
        self.assertTrue(passthrough)
        pcm2, reason, passthrough = data.convert(self._encode(x, 16000, "WAV", "FLOAT"))
        self.assertIsNone(reason)
        self.assertFalse(passthrough)
        self.assertLessEqual(np.max(np.abs(pcm.astype(int) - pcm2.astype(int))), 1)
        loud, reason, _ = data.convert(self._encode(np.full(1600, 1.5, np.float32), 16000, "WAV", "FLOAT"))
        self.assertIsNone(reason)
        self.assertEqual(int(loud.max()), 32767)

    def test_bad_audio_reasons(self):
        self.assertEqual(data.convert(b"not audio")[1], "decode_error")
        self.assertEqual(data.convert(self._encode(np.zeros(1600, np.int16), 16000))[1], "silent_audio")
        nan = np.full(1600, np.nan, np.float32)
        self.assertEqual(data.convert(self._encode(nan, 16000, "WAV", "FLOAT"))[1], "nonfinite_audio")

    def test_write_flac_round_trip(self):
        x = (1000 * np.sin(np.arange(32000) / 7)).astype(np.int16)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a.flac"
            data.write_flac(out, x)
            y, sr = sf.read(out, dtype="int16")
            info = sf.info(out)
        self.assertEqual((sr, info.channels, info.format, info.subtype), (16000, 1, "FLAC", "PCM_16"))
        np.testing.assert_array_equal(x, y)

    def test_safe_name(self):
        self.assertEqual(data.safe_name("20160526-10:18:50_2"), "20160526-10_18_50_2")
        self.assertEqual(data.safe_name("a/b c.flac"), "a_b_c.flac")


if __name__ == "__main__":
    unittest.main()
