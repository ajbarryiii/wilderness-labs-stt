"""Normalizer, edit counts, scoring rules, batching and test-set manifest schema (CPU only, no model load)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import evaluate
import paths
import testsets


class NormalizerTest(unittest.TestCase):
    def test_sample_strings(self) -> None:
        cases = {
            "Hello, World!": "hello world",
            "Mr. Smith paid $20 on Jan. 3rd.": "mister smith paid $20 on jan 3rd",
            "I won't colour it, it's twenty-one percent.": "i will not color it it is 21%",
            "Um, uh, well...": "well .",  # leaderboard behavior: an ellipsis leaves " ."
            "Um.": "",
            "THE QUICK BROWN FOX": "the quick brown fox",
            "In ' 22 we'll grow.": "in 22 we will grow",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(evaluate.normalize(text), expected)

    def test_pinned_mapping(self) -> None:
        self.assertEqual(len(json.loads(evaluate.NORMALIZER_JSON.read_text())), 1740)


class EditCountsTest(unittest.TestCase):
    def test_cases(self) -> None:
        ec = evaluate.edit_counts
        self.assertEqual(ec([], []), (0, 0, 0))
        self.assertEqual(ec("a b c".split(), "a b c".split()), (0, 0, 0))
        self.assertEqual(ec("a b c".split(), "a x c".split()), (1, 0, 0))
        self.assertEqual(ec("a b c".split(), "a c".split()), (0, 1, 0))
        self.assertEqual(ec("a b c".split(), "a b c d".split()), (0, 0, 1))
        self.assertEqual(ec("a b".split(), []), (0, 2, 0))
        self.assertEqual(ec([], "a b".split()), (0, 0, 2))
        s, d, i = ec("the cat sat on the mat".split(), "a cat sat the mat today".split())
        self.assertEqual(s + d + i, 3)

    def test_corpus_wer_excludes_empty_references(self) -> None:
        records = [{"id": "a", **evaluate.score("hello world", "hello word")},
                   {"id": "b", **evaluate.score("Um.", "um okay")},
                   {"id": "c", **evaluate.score("ignore time segment in scoring", "anything")}]
        self.assertEqual([r["scored"] for r in records], [True, False, False])
        summary = evaluate.wer_summary(records)
        self.assertEqual((summary["wer"], summary["ref_words"], summary["excluded_empty_reference"]), (0.5, 2, 2))

    def test_utterance_wer(self) -> None:
        self.assertEqual(evaluate.utterance_wer("Hello there.", "hello there"), 0.0)
        self.assertEqual(evaluate.utterance_wer("a b c d", "a b"), 0.5)
        self.assertEqual(evaluate.utterance_wer("Um.", ""), 0.0)
        self.assertEqual(evaluate.utterance_wer("Um.", "yes"), float("inf"))


class BatchingTest(unittest.TestCase):
    def test_make_batches(self) -> None:
        records = [{"id": f"u{i}", "duration": d} for i, d in enumerate([1.0, 30.0, 5.0, 29.0, 2.0])]
        batches = evaluate.make_batches(records, batch_size=2, max_batch_seconds=40.0)
        self.assertEqual(batches, [[1], [3, 2], [4, 0]])
        self.assertEqual(sorted(i for b in batches for i in b), list(range(5)))


class ShortUtteranceTest(unittest.TestCase):
    """Utterances under MIN_DECODE_SECONDS are zero-padded to it (ami_dev has a 0.020 s one)."""

    def _write(self, directory: Path, name: str, seconds: float) -> dict:
        samples = round(seconds * paths.SAMPLE_RATE)
        audio = 0.01 * np.sin(np.arange(samples, dtype=np.float32) / 7.0)
        path = directory / f"{name}.flac"
        sf.write(path, audio, paths.SAMPLE_RATE, subtype="PCM_16")
        return {"id": name, "audio_filepath": str(path), "duration": samples / paths.SAMPLE_RATE, "text": "x"}

    def test_pad_short(self) -> None:
        self.assertEqual(evaluate.MIN_DECODE_SAMPLES, 480)
        clip = np.ones(320, dtype=np.float32)
        padded, was = evaluate.pad_short(clip)
        self.assertTrue(was)
        self.assertEqual(len(padded), 480)
        self.assertTrue(np.array_equal(padded[:320], clip) and not padded[320:].any())
        clip = np.ones(640, dtype=np.float32)
        same, was = evaluate.pad_short(clip)
        self.assertFalse(was)
        self.assertIs(same, clip)

    def test_short_clip_is_padded_and_decodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [self._write(Path(tmp), "short", 0.02), self._write(Path(tmp), "ok", 0.04)]
            idx, audio, lengths, padded = evaluate._BatchSet(records, [[0, 1]])[0]
            self.assertEqual(lengths.tolist(), [480, 640])  # 0.02 s padded to 0.03 s; 0.04 s untouched
            self.assertEqual(padded, [0])
            model = evaluate.load_pretrained("cpu")
            stats: dict = {}
            out = evaluate.decode(model, records, batch_size=2, device="cpu", workers=0, stats=stats)
            self.assertEqual([r["id"] for r in out], ["short", "ok"])
            self.assertEqual((stats["padded_short"], stats["padded_short_ids"]), (1, ["short"]))


class TestsetManifestTest(unittest.TestCase):
    def test_schema(self) -> None:
        found = 0
        for name in paths.TEST_SETS:
            path = testsets.manifest_path(name)
            if not path.exists():
                continue
            found += 1
            with self.subTest(set=name):
                records = testsets.read_manifest(path)
                meta = json.loads(path.with_suffix(".meta.json").read_text())
                self.assertEqual(len(records), meta["utterances"])
                self.assertEqual(meta["revision"], paths.ESB_REVISION)
                self.assertEqual(meta["manifest_sha256"], testsets.sha256_file(path))
                ids = [r["id"] for r in records]
                self.assertEqual(ids, sorted(ids))
                self.assertEqual(len(set(ids)), len(ids))
                for r in records[:: max(1, len(records) // 50)]:
                    self.assertEqual(list(r), ["audio_filepath", "duration", "text", "id", "source"])
                    self.assertEqual(r["source"], name)
                    self.assertIsInstance(r["text"], str)
                    self.assertGreater(r["duration"], 0)
                    self.assertTrue(r["audio_filepath"].endswith(".flac"))
                    self.assertTrue(r["audio_filepath"].startswith(str(paths.AUDIO / "test" / name)))
                audio = evaluate.load_audio(records[0]["audio_filepath"])
                self.assertAlmostEqual(len(audio) / paths.SAMPLE_RATE, records[0]["duration"], places=6)
        if not found:
            self.skipTest("no test-set manifests yet (run testsets.py)")


if __name__ == "__main__":
    unittest.main()
