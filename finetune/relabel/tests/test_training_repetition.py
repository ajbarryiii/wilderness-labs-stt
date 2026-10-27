"""Contract tests: repetition rejection, rule "repetition" version 1 (training/repetition.py): the run length of an
immediately repeated token n-gram (n = 1..6, in repeats of the n-gram), the threshold (10, inclusive), the students'
tokenizer and the rule's identity. The label store's use of the rule is in test_training_data.py."""
from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from labeling import filters
from training import repetition as trep


def tokenizer_file() -> Path:
    if "file" not in _TOKENIZER:
        _TOKENIZER["dir"] = tempfile.TemporaryDirectory()
        _TOKENIZER["file"] = filters.b0_tokenizer_file(Path(_TOKENIZER["dir"].name))
    return _TOKENIZER["file"]


_TOKENIZER: dict = {}


class RunLength(unittest.TestCase):
    def test_empty_and_no_repetition(self):
        self.assertEqual(trep.longest_repeat([]), (0, 0))
        self.assertEqual(trep.longest_repeat([7]), (1, 1))
        self.assertEqual(trep.longest_repeat([1, 2, 3, 4, 5]), (1, 1))

    def test_single_token_runs(self):
        self.assertEqual(trep.longest_repeat([5, 9, 9, 9, 4]), (3, 1))
        self.assertEqual(trep.longest_repeat([1] + [8] * 9 + [2]), (9, 1))
        self.assertEqual(trep.longest_repeat([1] + [8] * 10 + [2]), (10, 1))
        self.assertEqual(trep.longest_repeat([8] * 486), (486, 1))

    def test_n_gram_runs_counted_in_repeats_of_the_n_gram(self):
        self.assertEqual(trep.longest_repeat([1, 2] * 10), (10, 2))
        self.assertEqual(trep.longest_repeat([1, 2] * 9 + [1]), (9, 2))          # a trailing partial n-gram
        self.assertEqual(trep.longest_repeat([0] + [3, 4, 5, 6, 7, 8] * 10 + [0]), (10, 6))
        self.assertEqual(trep.longest_repeat([3, 4, 5, 6, 7, 8, 9] * 10), (1, 1))    # n = 7 is not looked at
        self.assertEqual(trep.longest_repeat([3, 4, 5, 6, 7, 8, 9] * 10, n_max=7), (10, 7))

    def test_the_longest_run_wins_and_ties_go_to_the_smallest_n(self):
        self.assertEqual(trep.longest_repeat([1, 1, 1, 2, 3, 2, 3, 2, 3, 2, 3]), (4, 2))
        self.assertEqual(trep.longest_repeat([4] * 12), (12, 1))                    # also 6 repeats of [4, 4]
        self.assertEqual(trep.longest_repeat([1, 2, 1, 2, 5, 5]), (2, 1))

    def test_batch_equals_one_by_one(self):
        rng = random.Random(3)
        seqs = [[]]
        for _ in range(400):
            n = rng.randint(1, 4)
            unit = [rng.randint(0, 3) for _ in range(n)]
            seqs.append([rng.randint(0, 3) for _ in range(rng.randint(0, 6))] + unit * rng.randint(1, 14)
                        + [rng.randint(0, 3) for _ in range(rng.randint(0, 6))])
        repeats, ns = trep.longest_repeats(seqs)
        self.assertEqual([(int(r), int(n)) for r, n in zip(repeats, ns)], [trep.longest_repeat(s) for s in seqs])

    def test_threshold_is_inclusive(self):
        self.assertEqual((trep.THRESHOLD, trep.N_MAX, trep.VERSION, trep.NAME), (10, 6, 1, "repetition"))
        self.assertFalse(trep.rejects(9))
        self.assertTrue(trep.rejects(10))
        self.assertFalse(trep.rejects(0))


class Tokenized(unittest.TestCase):
    """The rule on label text, through the students' SentencePiece model."""

    @classmethod
    def setUpClass(cls):
        cls.spec = trep.Repetition(tokenizer_file())

    def test_the_students_tokenizer(self):
        # B0's SentencePiece model (from the pinned .nemo) is the one Kea-B's export pins
        self.assertEqual(self.spec.tokenizer_sha256, trep.TOKENIZER_SHA256)
        import score_domain
        score_domain.binary_on_path()
        import binary_test_paths
        self.assertEqual(binary_test_paths.PINNED_FILES["export/tokenizer/tokenizer.model"], trep.TOKENIZER_SHA256)

    def test_label_text_cases(self):
        r = self.spec.repeats
        self.assertEqual(r(""), (0, 0))
        self.assertEqual(r("no no no"), (3, 1))                                    # a genuine short repetition
        self.assertEqual(r("Okay. Wu" + "r" * 10 + "."), (9, 1))                  # "▁W ur" then 9 "r"
        self.assertEqual(r("Okay. Wu" + "r" * 11 + "."), (10, 1))
        self.assertEqual(r("so " + "uh, " * 10 + "right"), (10, 2))              # "▁uh ," ten times
        self.assertEqual(r("so " + "uh, " * 9 + "right"), (9, 2))
        self.assertEqual(r("I shouldn't be surprised. Wu" + "r" * 487), (486, 1))

    def test_rejected_labels(self):
        rejected = self.spec.evaluate(["a", "b", "c", "d", "e"], ["kept", "kept", "kept", "rejected", "kept"],
                                      ["no no no", "Wu" + "r" * 11, "uh, " * 10, "Wu" + "r" * 40, ""])
        self.assertEqual(rejected["rejected"], [["b", 10, 1], ["c", 10, 2]])     # d: not kept by T*
        self.assertEqual(rejected["kept"], 4)

    def test_identity_names_the_rule_and_the_rejected_set(self):
        a = self.spec.identity(["y", "x"])
        self.assertEqual({k: a[k] for k in ("name", "version", "n", "threshold", "tokenizer_sha256", "rejected")},
                         {"name": "repetition", "version": 1, "n": [1, 6], "threshold": 10,
                          "tokenizer_sha256": trep.TOKENIZER_SHA256, "rejected": 2})
        self.assertEqual(a, self.spec.identity(["x", "y"]))                        # the sorted set
        self.assertNotEqual(a["rejected_sha256"], self.spec.identity(["x"])["rejected_sha256"])
        other = trep.Repetition(tokenizer_file(), threshold=20)
        self.assertNotEqual(other.identity(["x", "y"]), a)
        self.assertEqual(other.identity(["x", "y"])["rejected_sha256"], a["rejected_sha256"])


if __name__ == "__main__":
    unittest.main()
