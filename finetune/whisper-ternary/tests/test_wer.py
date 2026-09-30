"""Edit counts, corpus aggregation and Whisper normalization used for every reported WER."""
from __future__ import annotations

import unittest

import wer


def record(ref: str, hyp: str) -> dict:
    return {"ref_norm": ref, "hyp_norm": hyp}


class EditCountTest(unittest.TestCase):
    def test_known_cases(self) -> None:
        words = "the quick brown fox".split()
        self.assertEqual(wer.edit_counts(words, words), (0, 0, 0))
        self.assertEqual(wer.edit_counts(words, []), (0, 4, 0))
        self.assertEqual(wer.edit_counts([], words), (0, 0, 4))
        self.assertEqual(wer.edit_counts([], []), (0, 0, 0))
        self.assertEqual(wer.edit_counts(words, "the quick red fox".split()), (1, 0, 0))
        self.assertEqual(wer.edit_counts(words, "the very quick brown fox".split()), (0, 0, 1))
        self.assertEqual(wer.edit_counts(words, "quick brown fox".split()), (0, 1, 0))

    def test_total_equals_levenshtein(self) -> None:
        cases = [("a b c d e f", "a x c e f g h"), ("one two three", "three two one"),
                 ("a a a a", "a"), ("a", "b b b b")]
        for ref, hyp in cases:
            ref, hyp = ref.split(), hyp.split()
            prev = list(range(len(hyp) + 1))
            for i, a in enumerate(ref, 1):
                row = [i]
                for j, b in enumerate(hyp, 1):
                    row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (a != b)))
                prev = row
            s, d, i = wer.edit_counts(ref, hyp)
            self.assertEqual(s + d + i, prev[-1])
            self.assertEqual(len(ref) - d + i, len(hyp))


class CorpusTest(unittest.TestCase):
    def test_aggregation_is_corpus_level(self) -> None:
        result = wer.corpus_wer([record("a b c d", "a b c d"), record("e f", "e x"),
                                 record("g h i j", "g h i j k")])
        self.assertEqual((result["substitutions"], result["deletions"], result["insertions"]),
                         (1, 0, 1))
        self.assertEqual((result["ref_words"], result["hyp_words"], result["utterances"]),
                         (10, 11, 3))
        self.assertAlmostEqual(result["wer"], 2 / 10)

    def test_wer_can_exceed_one(self) -> None:
        result = wer.corpus_wer([record("yes", "yes yes yes yes")])
        self.assertEqual(result["insertions"], 3)
        self.assertAlmostEqual(result["wer"], 3.0)

    def test_no_reference_words(self) -> None:
        with self.assertRaises(ValueError):
            wer.corpus_wer([record("", "noise")])


class NormalizerTest(unittest.TestCase):
    def test_whisper_normalizer(self) -> None:
        normalize = wer.Normalizer()
        a = normalize("Mr. Brown's 2nd colour, isn't it?")
        b = normalize("MISTER BROWN'S SECOND COLOR ISN'T IT")
        self.assertEqual(a, "mister brown is 2nd color is not it")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
