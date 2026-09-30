"""transcribe.py: audio reading, windowing, duration cap, processor directory selection."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile
from transformers import WhisperProcessor

import paths
import transcribe

SR = paths.SAMPLE_RATE


class ReadAudio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data, rate=SR):
        path = self.dir / name
        soundfile.write(str(path), data, rate)
        return path

    def test_mono(self):
        data = np.linspace(-0.5, 0.5, SR, dtype=np.float32)
        audio = transcribe.read_audio(self.write("m.wav", data))
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (SR,))

    def test_stereo_is_averaged(self):
        left = np.full(SR, 0.5, dtype=np.float32)
        right = np.full(SR, -0.25, dtype=np.float32)
        audio = transcribe.read_audio(self.write("s.wav", np.stack([left, right], axis=1)))
        self.assertEqual(audio.shape, (SR,))
        self.assertTrue(np.allclose(audio, 0.125, atol=1e-4))

    def test_other_rate_refused_with_hint(self):
        path = self.write("r.wav", np.zeros(44100, dtype=np.float32), rate=44100)
        with self.assertRaisesRegex(ValueError, "ffmpeg"):
            transcribe.read_audio(path)

    def test_empty_refused(self):
        with self.assertRaises(ValueError):
            transcribe.read_audio(self.write("e.wav", np.zeros(0, dtype=np.float32)))


class Windows(unittest.TestCase):
    def test_exact_multiple(self):
        chunks = transcribe.windows(np.zeros(60 * SR, dtype=np.float32))
        self.assertEqual([len(c) for c in chunks], [30 * SR, 30 * SR])

    def test_remainder_and_short(self):
        self.assertEqual([len(c) for c in transcribe.windows(np.zeros(31 * SR))], [30 * SR, SR])
        self.assertEqual([len(c) for c in transcribe.windows(np.zeros(5))], [5])

    def test_concatenation_preserves_samples(self):
        audio = np.arange(65 * SR, dtype=np.float32)
        self.assertTrue(np.array_equal(np.concatenate(transcribe.windows(audio)), audio))


class DurationCap(unittest.TestCase):
    def test_boundary(self):
        limit = 14  # ceil(4.5 * 2.0) + 5
        at = " ".join(f"w{i}" for i in range(limit))
        over = at + " extra"
        self.assertEqual(transcribe.cap_words(at, 2.0), at)
        self.assertEqual(transcribe.cap_words(over, 2.0), at)

    def test_ceil_on_fractional_seconds(self):
        # ceil(4.5 * 1.1) + 5 = ceil(4.95) + 5 = 10
        text = " ".join(str(i) for i in range(12))
        self.assertEqual(len(transcribe.cap_words(text, 1.1).split()), 10)


class ProcessorDir(unittest.TestCase):
    """Uses real tokenizer files copied from the pinned checkpoint, not stubs."""

    def bundle(self, tmp: str, names: list[str]) -> Path:
        for name in names:
            shutil.copy(paths.MODEL_DIR / name, Path(tmp) / name)
        return Path(tmp)

    def test_fast_tokenizer_bundle_is_used_and_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.bundle(tmp, ["preprocessor_config.json", "tokenizer_config.json",
                                  "tokenizer.json", "special_tokens_map.json",
                                  "added_tokens.json", "normalizer.json"])
            self.assertEqual(transcribe.processor_dir(d), d)
            processor = WhisperProcessor.from_pretrained(d, local_files_only=True)
            self.assertEqual(processor.tokenizer("hello world").input_ids,
                             [50257, 50362, 31373, 995, 50256])

    def test_slow_tokenizer_bundle_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.bundle(tmp, ["preprocessor_config.json", "tokenizer_config.json",
                                  "vocab.json", "merges.txt"])
            self.assertEqual(transcribe.processor_dir(d), d)

    def test_incomplete_bundle_falls_back_to_pinned_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self.bundle(tmp, ["preprocessor_config.json", "tokenizer_config.json",
                                  "vocab.json"])  # no merges.txt, no tokenizer.json
            self.assertEqual(transcribe.processor_dir(d), paths.MODEL_DIR)

    def test_empty_dir_falls_back_to_pinned_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(transcribe.processor_dir(Path(tmp)), paths.MODEL_DIR)


class Args(unittest.TestCase):
    def test_defaults(self):
        args = transcribe.parse_args(["a.wav"])
        self.assertEqual(args.export, transcribe.DEFAULT_EXPORT)
        self.assertFalse(args.compare_fp32)
        self.assertFalse(args.duration_cap)
        self.assertEqual(args.device, "cpu")


if __name__ == "__main__":
    unittest.main()
