"""Contract tests: filters 1-4 of the labeling pass (labeling plan Part C "Filters")."""
from __future__ import annotations

import unittest

import paths  # finetune/parakeet-ternary
from labeling import filters


def words(n, tag="wor"):
    """n distinct alphabetic words (the normalizer splits letters from digits, so no digits)."""
    return " ".join(f"{tag}{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(n))


def covered(_text):
    return True


def check(label, duration=3.0, ref=None, kind=None, b0=None, coverage=covered):
    return filters.check(duration, label, ref, kind, b0 if b0 is not None else label, coverage)


class Filter1(unittest.TestCase):
    def test_duration_bounds_inclusive(self):
        self.assertEqual(check("hello there", duration=1.0)["disposition"], "kept")
        self.assertEqual(check("hello there", duration=30.0)["disposition"], "kept")
        self.assertIn("duration_out_of_range", check("hello", duration=0.99)["reasons"])
        self.assertIn("duration_out_of_range", check("hello", duration=30.01)["reasons"])

    def test_empty_label(self):
        r = check("   ", b0="")
        self.assertEqual(r["disposition"], "rejected")
        self.assertIn("empty_label", r["reasons"])


class Filter2(unittest.TestCase):
    def test_human_reference_wer(self):
        ref = "the patient was given two units of blood"
        self.assertEqual(check("The patient was given two units of blood.", ref=ref, kind="human")["disposition"], "kept")
        r = check("completely different words here now", ref=ref, kind="human")
        self.assertIn("human_wer_gt_0.5", r["reasons"])
        self.assertGreater(r["values"]["human_wer"], 0.5)

    def test_exactly_half_is_kept(self):
        r = check("a b c d", ref="a b x y", kind="human")
        self.assertAlmostEqual(r["values"]["human_wer"], 0.5)
        self.assertNotIn("human_wer_gt_0.5", r["reasons"])

    def test_pseudo_reference_is_not_a_human_transcript(self):
        r = check("totally unrelated", ref="the patient was given blood", kind="pseudo", b0="totally unrelated")
        self.assertNotIn("human_wer_gt_0.5", r["reasons"])
        self.assertNotIn("human_wer", r["values"])


class Filter3(unittest.TestCase):
    def test_words_per_second(self):
        self.assertIn("words_per_second_gt_6", check(words(19), duration=3.0)["reasons"])
        r = check(words(18), duration=3.0)
        self.assertEqual(r["values"]["words"], 18)
        self.assertNotIn("words_per_second_gt_6", r["reasons"])

    def test_trigram_loop_only_for_short_labels(self):
        loop = "go to the go to the go to the go to the"            # 12 words, "go to the" x4
        self.assertIn("ngram_loop", check(loop, duration=10.0)["reasons"])
        three = "go to the go to the go to the end"
        self.assertNotIn("ngram_loop", check(three, duration=10.0)["reasons"])
        long = loop + " " + words(20)                             # 32 words: the 3-gram rule does not apply
        self.assertNotIn("ngram_loop", check(long, duration=10.0)["reasons"])

    def test_same_word_five_times_in_a_row(self):
        self.assertIn("ngram_loop", check("yes yes yes yes yes", duration=5.0)["reasons"])
        self.assertNotIn("ngram_loop", check("yes yes yes yes no", duration=5.0)["reasons"])
        long = words(40) + " no no no no no"
        self.assertIn("ngram_loop", check(long, duration=20.0)["reasons"])

    def test_b0_disagreement_only_without_a_human_transcript(self):
        r = check("alpha beta gamma delta", ref=None, kind=None, b0="one two three four five")
        self.assertIn("b0_wer_gt_0.8", r["reasons"])
        r = check("alpha beta gamma delta", ref="alpha beta gamma delta", kind="human", b0="one two three four five")
        self.assertNotIn("b0_wer_gt_0.8", r["reasons"])
        self.assertEqual(r["disposition"], "kept")
        r = check("alpha beta gamma delta", ref="pseudo label", kind="pseudo", b0="alpha beta gamma delta")
        self.assertNotIn("b0_wer_gt_0.8", r["reasons"])

    def test_b0_empty_and_teacher_speaks(self):
        r = check("hello there", kind=None, b0="")
        self.assertIn("b0_wer_gt_0.8", r["reasons"])


class Filter4(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(filters.map_text("Don’t “stop” — now…"), 'Don\'t "stop" - now...')

    def test_coverage_uses_the_mapped_text(self):
        seen = []

        def cov(text):
            seen.append(text)
            return "’" not in text
        r = check("Don’t", coverage=cov)
        self.assertEqual(seen, ["Don't"])
        self.assertEqual(r["text"], "Don't")
        self.assertEqual(r["disposition"], "kept")
        r = check("abc", coverage=lambda t: False)
        self.assertIn("tokenizer_coverage", r["reasons"])

    @unittest.skipUnless(paths.MODEL_FILE.exists(), "B0 .nemo not available")
    def test_b0_tokenizer(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            cov = filters.Coverage(filters.b0_tokenizer_file(Path(tmp)))
            self.assertTrue(cov(filters.map_text("Don’t stop, it's 5 mg.")))
            self.assertFalse(cov("中文"))
            self.assertIn("中", cov.uncovered("a 中"))


class Disposition(unittest.TestCase):
    def test_kept_iff_no_reason_and_reasons_in_filter_order(self):
        r = check("yes yes yes yes yes", duration=0.5, ref=None, kind=None, b0="no", coverage=lambda t: False)
        self.assertEqual(r["disposition"], "rejected")
        order = [filters.REASONS.index(x) for x in r["reasons"]]
        self.assertEqual(order, sorted(order))
        self.assertEqual(set(r["reasons"]), {"duration_out_of_range", "words_per_second_gt_6", "ngram_loop",
                                             "b0_wer_gt_0.8", "tokenizer_coverage"})
        self.assertEqual({k for k, v in r["flags"].items() if v}, set(r["reasons"]))
        ok = check("The patient is stable.")
        self.assertEqual((ok["disposition"], ok["reasons"]), ("kept", []))
        self.assertEqual(ok["norm"], "the patient is stable")


if __name__ == "__main__":
    unittest.main()
