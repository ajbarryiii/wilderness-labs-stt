"""Term density contracts (labeling plan Part B "Score"): terms are normalized like transcripts and
matched as whole-word sequences, longest first, without overlap; density is per 1,000 normalized words.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_pool_*' -v
"""
from __future__ import annotations

import unittest

from pool import common, terms_match


class TermMatcherTest(unittest.TestCase):
    def test_normalized_whole_word_multiword_longest_first(self):
        m = terms_match.TermMatcher(["Blood Pressure", "blood", "tourniquet", "Tranexamic acid"])
        words = common.normalize("Her blood pressure dropped; apply a tourniquet. Bloodline tourniquets blood.").split()
        # "blood pressure" once (not also "blood"), "tourniquet" once (not "tourniquets"), "blood" once.
        self.assertEqual(m.count(words), 3)

    def test_no_overlap(self):
        m = terms_match.TermMatcher(["a b", "b c"])
        self.assertEqual(m.count("a b c".split()), 1)
        self.assertEqual(m.count("a b b c".split()), 2)

    def test_terms_normalized_like_text(self):
        m = terms_match.TermMatcher(["Dr."])  # the normalizer spells this "doctor"
        self.assertEqual(m.count(common.normalize("The doctor and Dr. Who").split()), 2)

    def test_density(self):
        self.assertEqual(terms_match.density(10, 1000), 10.0)
        self.assertEqual(terms_match.density(2, 200), 10.0)
        self.assertEqual(terms_match.density(3, 0), 0.0)

    def test_rejects_bad_lists(self):
        with self.assertRaises(ValueError):
            terms_match.TermMatcher([])
        with self.assertRaises(TypeError):
            terms_match.TermMatcher([("a", "b")])


if __name__ == "__main__":
    unittest.main()
