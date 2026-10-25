"""Forced-aligner contracts (labeling plan Part A: "cut with a forced aligner independent of every evaluated
model ... a segment whose alignment fails is dropped"): on emissions whose truth is known, the block
aligner recovers every unit's time over recordings much longer than one block, also when the speech is
sparse; text that does not match the audio scores below the acceptance threshold; units that cannot fit
the remaining audio come back failed. One test loads the pinned MMS checkpoint (frame count, chunking).

Run from finetune/relabel:  ./python -m unittest discover -s tests -p "test_devsets_*" -v
"""
from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from devsets import align

LETTERS = "abcdefghijklmnopqrstuvwxyz"
DICTIONARY = {c: i + 1 for i, c in enumerate(LETTERS)}
BLANK, STAR, C = 0, 27, 28


def emission_for(words: list[str], gaps: list[int], lead: int = 50, tail: int = 100, spacing: int = 3):
    """Log-probs [T, C] with each character spiking for one frame, `spacing` frames apart within a word,
    `gaps[i]` blank frames after word i; returns (emission, true first/last char frame per word)."""
    frames = lead + sum(len(w) * spacing for w in words) + sum(gaps) + tail
    probs = torch.full((frames, C), 0.02 / (C - 2))
    probs[:, BLANK] = 0.98
    truth, t = [], lead
    for word, gap in zip(words, gaps):
        first = t
        for ch in word:
            probs[t] = 0.05 / (C - 2)
            probs[t, BLANK] = 0.10
            probs[t, DICTIONARY[ch]] = 0.85
            t += spacing
        truth.append((first, t - spacing))
        t += gap
    logp = probs.log()
    logp[:, STAR] = 0.0  # as torchaudio's MMS_FA with_star
    return logp, truth


def random_words(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(LETTERS) for _ in range(rng.randint(2, 9))) for _ in range(n)]


class BlockAlignerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.aligner = align.BlockAligner(DICTIONARY, STAR, BLANK)

    def assert_recovers(self, words, gaps) -> list:
        emission, truth = emission_for(words, gaps)
        spans = self.aligner.align_emission(emission, list(words))
        self.assertEqual(len(spans), len(words))
        for span, (first, last) in zip(spans, truth):
            self.assertAlmostEqual(span.start / align.FRAME_SECONDS, first, delta=1.01)
            self.assertAlmostEqual(span.end / align.FRAME_SECONDS, last + 1, delta=1.01)
            self.assertGreaterEqual(span.score, align.MIN_SCORE)
        return spans

    def test_long_dense_recording(self) -> None:
        rng = random.Random(1)
        words = random_words(900, 1)
        gaps = [rng.randint(3, 20) for _ in words]
        emission, _ = emission_for(words, gaps)
        self.assertGreater(emission.shape[0] * align.FRAME_SECONDS, 5 * align.BLOCK_SECONDS)
        self.assert_recovers(words, gaps)

    def test_sparse_speech(self) -> None:
        rng = random.Random(2)
        words = random_words(300, 2)
        gaps = [rng.randint(3, 15) if i % 8 else rng.randint(400, 1500) for i in range(len(words))]
        self.assert_recovers(words, gaps)

    def test_wildcard_units_and_untranscribed_audio(self) -> None:
        words = random_words(120, 3)
        gaps = [10] * len(words)
        emission, truth = emission_for(words, gaps)
        units = list(words)
        units[40] = None   # a markup-only token: matches its word's audio
        del units[80]      # a word missing from the transcript
        del truth[80]
        spans = self.aligner.align_emission(emission, units)
        for u, (span, (first, last)) in enumerate(zip(spans, truth)):
            if units[u] is None:
                self.assertIsNone(span.score)
                continue
            self.assertAlmostEqual(span.start / align.FRAME_SECONDS, first, delta=1.01)

    def test_mismatched_text_scores_below_threshold(self) -> None:
        words = random_words(200, 4)
        emission, _ = emission_for(words, [12] * len(words))
        spans = self.aligner.align_emission(emission, random_words(200, 5))
        mean = sum(s.score for s in spans) / len(spans)
        self.assertLess(mean, align.MIN_SCORE)

    def test_units_that_cannot_fit_fail(self) -> None:
        words = random_words(10, 6)
        emission, _ = emission_for(words, [5] * len(words), tail=5)
        spans = self.aligner.align_emission(emission, words + random_words(200, 7))
        self.assertTrue(all(s.score == 0.0 for s in spans[-150:]))


class MMSCheckpointTest(unittest.TestCase):
    """Loads the pinned torchaudio MMS_FA checkpoint (cached under $RELABEL_ROOT/caches/torch)."""

    def test_pinned_checkpoint_and_chunked_frames(self) -> None:
        aligner = align.MMSAligner(threads=4)
        self.assertEqual(aligner.identity["sha256"], align.CHECKPOINT_SHA256)
        rng = np.random.default_rng(0)
        for seconds in (0.5, 19.99, 45.3):
            audio = (0.01 * rng.standard_normal(int(seconds * align.SAMPLE_RATE))).astype(np.float32)
            emission = aligner.emission(audio)
            self.assertEqual(tuple(emission.shape), (align.frame_count(len(audio)), C + 1))
            self.assertTrue(torch.isfinite(emission).all())
        short = (0.01 * rng.standard_normal(10 * align.SAMPLE_RATE)).astype(np.float32)
        with torch.inference_mode():
            whole, _ = aligner.model(torch.from_numpy(short).unsqueeze(0))
        self.assertTrue(torch.allclose(aligner.emission(short), whole[0], atol=1e-5))


if __name__ == "__main__":
    unittest.main()
