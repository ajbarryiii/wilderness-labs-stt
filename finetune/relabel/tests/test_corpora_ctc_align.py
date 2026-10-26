"""Training-aligner contracts (corpora/ctc_align.py): Part A's block algorithm over a character CTC model.

- the model's log-probabilities, computed in Part A's chunks with context, land on the right absolute frames;
- the word delimiter is folded into the blank and never targeted; a wildcard column of log-probability 0 is
  appended;
- unit targets are the characters' class indices (wildcard units: the wildcard; unknown characters refused);
- end to end, words are placed at their true times and score at or above min_score, while a unit whose word is
  not in the audio scores below it (so Part A's drop rule works unchanged).

No model is downloaded: the CTC model is a stand-in whose output is a function of the absolute frame, which
the test audio encodes.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_corpora_*' -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from corpora import ctc_align
from devsets import align as part_a

VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, **{c: 5 + i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")}}
C = len(VOCAB)


def frame_audio(frames: int) -> np.ndarray:
    """Audio of exactly `frames` frames whose samples encode the absolute frame (value f / 1e4 over hop f)."""
    n = (frames - 1) * part_a.HOP + part_a.RECEPTIVE_FIELD
    return (np.arange(n) // part_a.HOP).astype(np.float32) / 1e4


def model_from_table(table: torch.Tensor):
    """Stand-in forward: frame k of a chunk returns row f of `table`, f = the frame encoded at k * HOP."""
    def forward(x):
        rows = [int(round(float(x[0, k * part_a.HOP]) * 1e4)) for k in range(part_a.frame_count(x.shape[1]))]
        return table[rows].unsqueeze(0)
    return forward


def table_of(frames: int, label_of) -> torch.Tensor:
    t = torch.full((frames, C), math.log(1e-4))
    for f in range(frames):
        t[f, label_of(f)] = 0.0
    return t.log_softmax(-1)


class EmissionTest(unittest.TestCase):
    def test_chunked_frames_delimiter_and_wildcard(self):
        frames = 2600   # 52 s: three chunks with context
        table = table_of(frames, lambda f: 5 + f % 26 if f % 7 else 4)   # every 7th frame: the delimiter
        al = ctc_align.CTCAligner(model_from_table(table), VOCAB)
        em = al.emission(frame_audio(frames))
        self.assertEqual(tuple(em.shape), (frames, C + 1))
        self.assertTrue(torch.all(em[:, al.star] == 0))
        self.assertTrue(torch.all(em[:, 4] < -1e3))   # the delimiter is never a target
        labels = em[:, :C].argmax(-1)
        for f in range(frames):
            self.assertEqual(int(labels[f]), 0 if f % 7 == 0 else 5 + f % 26, f)   # delimiter -> blank
        folded = torch.logaddexp(table[:, 0], table[:, 4])
        self.assertTrue(torch.allclose(em[:, 0], folded, atol=1e-5))

    def test_targets(self):
        al = ctc_align.CTCAligner(lambda x: None, VOCAB)
        self.assertEqual(al.targets("ab'"), [VOCAB["A"], VOCAB["B"], VOCAB["'"]])
        self.assertEqual(al.targets(None), [al.star])
        self.assertEqual((al.blank, al.star), (0, C))
        with self.assertRaises(ValueError):
            al.targets("a1")


class EndToEndTest(unittest.TestCase):
    def test_words_found_at_their_times_and_missing_word_scores_low(self):
        # Ten 3-letter words spoken at frames 100 k .. 100 k + 2 (k = 1..10), blank elsewhere.
        frames = 1200
        words = ["cat", "dog", "sun", "map", "pen", "box", "cup", "hat", "jam", "kit"]
        spoken = {}
        for k, w in enumerate(words, 1):
            for j, ch in enumerate(w):
                spoken[100 * k + 3 * j] = VOCAB[ch.upper()]
        al = ctc_align.CTCAligner(model_from_table(table_of(frames, lambda f: spoken.get(f, 0))), VOCAB)
        units = words[:5] + ["zzz"] + words[5:]
        spans = al.align(frame_audio(frames), units)
        self.assertEqual(len(spans), len(units))
        for w, s in zip(units, spans):
            if w == "zzz":
                self.assertLess(s.score, al.min_score)
                continue
            truth = 100 * (words.index(w) + 1) * part_a.FRAME_SECONDS
            self.assertAlmostEqual(s.start, truth, delta=part_a.FRAME_SECONDS)
            self.assertGreaterEqual(s.score, al.min_score)


if __name__ == "__main__":
    unittest.main()
