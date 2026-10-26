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


def check(label, duration=3.0, ref=None, kind=None, b0=None, coverage=covered, version=filters.VERSION):
    return filters.check(duration, label, ref, kind, b0 if b0 is not None else label, coverage, version=version)


def substituted(text, n):
    """text with its first n words replaced (n substitutions against text)."""
    w = text.split()
    return " ".join(["other"] * n + w[n:])


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
        # B0 disagrees on 5 of 19 words, so nothing confirms the rate (version 2; RateConfirmation)
        self.assertIn("words_per_second_gt_6", check(words(19), duration=3.0, b0=substituted(words(19), 5))["reasons"])
        r = check(words(18), duration=3.0, b0=substituted(words(18), 5))
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


class RateConfirmation(unittest.TestCase):
    """Filters version 2 (labeling plan Part C "Filters", amended 2026-10-25): the words-per-second rule does not fire
    when an independent transcript confirms the label, normalized WER(confirmer, T*) <= 0.2; the confirmer is the human
    transcript when there is one, else B0's label."""
    FAST = words(19)                                           # 19 words in 3 s: 6.3 words per second

    def test_version(self):
        self.assertIn(2, filters.VERSIONS)
        self.assertGreaterEqual(filters.VERSION, 2)          # later versions keep the confirmation (version 3)
        self.assertEqual(filters.CONFIRM_WER, 0.2)

    def test_fast_label_confirmed_by_the_human_transcript_is_kept(self):
        r = check(self.FAST, ref=substituted(self.FAST, 1), kind="human", b0="nothing alike")
        self.assertGreater(r["values"]["words_per_second"], 6)
        self.assertEqual((r["disposition"], r["reasons"]), ("kept", []))
        self.assertIs(r["values"]["rate_confirmed"], True)

    def test_confirmation_at_exactly_the_threshold(self):
        fast = words(10)                                       # 10 words in 1.5 s; 2 of 10 substituted: WER 0.2
        r = check(fast, duration=1.5, ref=substituted(fast, 2), kind="human")
        self.assertAlmostEqual(r["values"]["human_wer"], 0.2)
        self.assertEqual(r["disposition"], "kept")
        r = check(fast, duration=1.5, ref=substituted(fast, 3), kind="human")
        self.assertEqual(r["reasons"], ["words_per_second_gt_6"])

    def test_same_rate_unconfirmed_by_the_human_transcript_is_rejected(self):
        r = check(self.FAST, ref=substituted(self.FAST, 5), kind="human", b0=self.FAST)   # WER 0.26: passes filter 2
        self.assertLessEqual(r["values"]["human_wer"], 0.5)
        self.assertGreater(r["values"]["human_wer"], 0.2)
        self.assertEqual((r["disposition"], r["reasons"]), ("rejected", ["words_per_second_gt_6"]))
        self.assertIs(r["values"]["rate_confirmed"], False)

    def test_without_a_human_transcript_b0_confirms(self):
        for kind, ref in ((None, None), ("pseudo", "a pseudo label that disagrees entirely")):
            r = check(self.FAST, ref=ref, kind=kind, b0=substituted(self.FAST, 1))
            self.assertEqual((r["disposition"], r["reasons"]), ("kept", []), kind)
            self.assertIs(r["values"]["rate_confirmed"], True)
            r = check(self.FAST, ref=ref, kind=kind, b0=substituted(self.FAST, 5))       # WER 0.26: passes B0's 0.8
            self.assertEqual((r["disposition"], r["reasons"]), ("rejected", ["words_per_second_gt_6"]), kind)

    def test_a_pseudo_label_does_not_confirm(self):
        r = check(self.FAST, ref=self.FAST, kind="pseudo", b0=substituted(self.FAST, 5))
        self.assertEqual(r["reasons"], ["words_per_second_gt_6"])

    def test_b0_does_not_confirm_when_there_is_a_human_transcript(self):
        r = check(self.FAST, ref=substituted(self.FAST, 5), kind="human", b0=self.FAST)
        self.assertEqual(r["reasons"], ["words_per_second_gt_6"])

    def test_an_empty_b0_label_does_not_confirm(self):
        r = check(self.FAST, kind=None, b0="")
        self.assertIn("words_per_second_gt_6", r["reasons"])
        self.assertIn("b0_wer_gt_0.8", r["reasons"])

    def test_a_loop_is_still_rejected(self):
        loop = "yes " * 20                                     # confirmed rate, but the same word 20 times
        r = check(loop, ref=loop, kind="human")
        self.assertIs(r["values"]["rate_confirmed"], True)
        self.assertEqual((r["disposition"], r["reasons"]), ("rejected", ["ngram_loop"]))
        r = check(loop, ref=substituted(loop, 6), kind="human")   # unconfirmed: both rules
        self.assertEqual(r["reasons"], ["words_per_second_gt_6", "ngram_loop"])
        r = check(loop, kind=None, b0=loop)
        self.assertEqual(r["reasons"], ["ngram_loop"])

    def test_slow_labels_record_no_confirmation(self):
        self.assertNotIn("rate_confirmed", check(words(18), ref=words(18), kind="human")["values"])

    def test_version_1_has_no_waiver(self):
        r = check(self.FAST, ref=self.FAST, kind="human", version=1)
        self.assertEqual((r["disposition"], r["reasons"]), ("rejected", ["words_per_second_gt_6"]))
        self.assertNotIn("rate_confirmed", r["values"])
        with self.assertRaises(ValueError):
            check(self.FAST, version=4)


class Filter4(unittest.TestCase):
    def test_mapping(self):
        for version in (1, 2):
            self.assertEqual(filters.map_text("Don’t “stop” — now…", version=version), 'Don\'t "stop" - now...')
        self.assertEqual(filters.map_text("Don’t “stop” — now…", version=3), "Don't stop - now...")

    def test_map_text_needs_a_known_version(self):
        with self.assertRaises(TypeError):
            filters.map_text("hello")
        with self.assertRaises(ValueError):
            filters.map_text("hello", version=4)

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
            for version in filters.VERSIONS:
                self.assertTrue(cov(filters.map_text("Don’t stop, it's 5 mg.", version=version)))
            self.assertFalse(cov("中文"))
            self.assertIn("中", cov.uncovered("a 中"))


B0_LACKS = set('"#&()*+;<=>@[\\]_{|}~`')


def b0_like(text):
    """A stand-in for B0's tokenizer coverage: ASCII without the characters B0's vocabulary lacks."""
    return all(c.isascii() and c not in B0_LACKS for c in text)


class PunctuationMapping(unittest.TestCase):
    """Filters version 3 (labeling plan Part C "Filters", amended 2026-10-26): before the tokenizer-coverage rule, after
    NFKC and the curly quote and dash mapping, `" *` are removed, `( ) [ ] { } _` become a space, `;` becomes `,`,
    `& + = @ ° §` become words; then whitespace runs collapse, a space before `, . ? ! :` is removed, and the text is
    stripped. Filters 1-3 are unchanged."""
    EXAMPLES = {
        'She said, "I\'m bored."': "She said, I'm bored.",
        "72.28% (386,815) of Denver": "72.28% 386,815 of Denver",
        "U.S. News & World Report": "U.S. News and World Report",
        "Q&A": "Q and A",
        "C++ provides": "C plus plus provides",
        "between 49° and 51° N.": "between 49 degrees and 51 degrees N.",
        "White, 68.9%; non-Hispanic": "White, 68.9%, non-Hispanic",
        "the skipper of the _Salvador_, and": "the skipper of the Salvador, and",
        "f(x) dx": "f x dx",
        "“Hello”": "Hello",
        '"': "",
        "これは100円です。": "これは100円です。",
    }

    def test_version(self):
        self.assertEqual(filters.VERSIONS, (1, 2, 3))
        self.assertEqual(filters.VERSION, 3)

    def test_examples(self):
        for raw, want in self.EXAMPLES.items():
            self.assertEqual(filters.map_text(raw, version=3), want, raw)

    def test_each_mapped_character(self):
        m = lambda t: filters.map_text(t, version=3)
        self.assertEqual(m('a "b" *c*'), "a b c")
        for c in "()[]{}_":
            self.assertEqual(m(f"a{c}b"), "a b", c)
        self.assertEqual(m("a; b"), "a, b")
        self.assertEqual(m("a&b+c=d@e°f§g"), "a and b plus c equals d at e degrees f section g")
        self.assertEqual(m("  a \t ( b )  ,  c . d ? e ! f : g  "), "a b, c. d? e! f: g")
        self.assertEqual(m("a — b – c"), "a - b - c")                      # dashes: the version 1 mapping only
        self.assertEqual(m("＆"), "and")                                    # NFKC first

    def test_other_characters_are_unchanged(self):
        for t in ("a # b", "a < b > c", "a \\ b | c ~ d", "a `b` c", "café naïve", "これは100円です。",
                  "it's 5-6 mg/kg, 10% off: ok?"):
            self.assertEqual(filters.map_text(t, version=3), t, t)

    def test_versions_1_and_2_map_as_before(self):
        import unicodedata

        def before(text):
            text = unicodedata.normalize("NFKC", text)
            return "".join(filters.TEXT_MAP.get(c, c) for c in text).strip()
        for raw in list(self.EXAMPLES) + ["  a \t ( b )  ,  c ", "Don’t “stop” — now…", "", "   "]:
            for version in (1, 2):
                self.assertEqual(filters.map_text(raw, version=version), before(raw), (version, raw))

    def test_stored_text_and_coverage_use_the_version_3_mapping(self):
        seen = []

        def cov(text):
            seen.append(text)
            return b0_like(text)
        label = 'She said, "I\'m bored." (twice)'
        r = check(label, coverage=cov, version=3)
        self.assertEqual(seen, ["She said, I'm bored. twice"])
        self.assertEqual((r["text"], r["disposition"], r["reasons"]), ("She said, I'm bored. twice", "kept", []))
        self.assertEqual(r["raw"], label)
        r2 = check(label, coverage=b0_like, version=2)
        self.assertEqual((r2["text"], r2["reasons"]), (label, ["tokenizer_coverage"]))

    def test_characters_b0_lacks_still_fail(self):
        for label in ("item #5 is ready", "これは100円です。", "a < b", "the café"):
            r = check(label, coverage=b0_like, version=3)
            self.assertEqual(r["reasons"], ["tokenizer_coverage"], label)

    def test_a_label_that_maps_to_nothing_fails_coverage(self):
        r = check('"', coverage=lambda t: True, version=3)
        self.assertEqual(r["text"], "")
        self.assertIn("tokenizer_coverage", r["reasons"])
        self.assertNotIn("empty_label", r["reasons"])
        r = check("  ", coverage=lambda t: True, version=3, b0="")
        self.assertIn("empty_label", r["reasons"])
        self.assertNotIn("tokenizer_coverage", r["reasons"])

    def test_filters_1_to_3_are_those_of_version_2(self):
        fast = words(19)                                               # 6.3 words per second in 3 s
        cases = [
            ('She said, "I\'m bored."', 3.0, None, None, None),
            ("72.28% (386,815) of Denver", 3.0, "72.28% 386,815 of Denver", "human", None),
            ("U.S. News & World Report", 3.0, None, None, "us news and world report"),
            ("Q&A", 1.5, "q and a", "human", None),
            ("C++ provides", 2.0, None, None, "see plus plus provides"),
            ('"yes yes yes yes yes"', 5.0, None, None, None),
            (f'"{fast}"', 3.0, None, None, substituted(fast, 5)),
            (f"({fast})", 3.0, fast, "human", None),
            (f"[{fast}]", 3.0, None, None, substituted(fast, 1)),
            ('"', 3.0, None, None, None),
            ('"', 3.0, None, None, ""),
            ("", 3.0, None, None, ""),
            ("   ", 3.0, "something said", "human", None),
            (" “Hello” ", 0.5, None, None, None),
            ("“Hello”", 31.0, "hello", "human", None),
            ("これは100円です。", 3.0, None, None, "this is one hundred yen"),
            ("item #5 & <b>", 3.0, None, None, None),
            ("[music] _intro_ ; °", 40.0, None, None, "music intro degrees"),
            ("completely (different) words; here", 3.0, "the patient was given blood", "human", None),
            ("(hello) there", 3.0, None, None, ""),
            ("f(x) dx = 3 § 2", 3.0, "pseudo label", "pseudo", "f x dx equals three section two"),
            ("@home + away", 3.0, None, None, "at home plus away"),
            ("go (to) the go to the go [to] the go to the", 10.0, None, None, None),
        ]
        for label, duration, ref, kind, b0 in cases:
            v2 = check(label, duration=duration, ref=ref, kind=kind, b0=b0, coverage=b0_like, version=2)
            v3 = check(label, duration=duration, ref=ref, kind=kind, b0=b0, coverage=b0_like, version=3)
            other = lambda r: [x for x in r["reasons"] if x != "tokenizer_coverage"]
            self.assertEqual(other(v3), other(v2), label)
            self.assertEqual((v3["raw"], v3["norm"], v3["values"]), (v2["raw"], v2["norm"], v2["values"]), label)
            self.assertEqual({k: v for k, v in v3["flags"].items() if k != "tokenizer_coverage"},
                             {k: v for k, v in v2["flags"].items() if k != "tokenizer_coverage"}, label)
            self.assertEqual(v3["text"], filters.map_text(v3["raw"], version=3), label)
            self.assertEqual(v3["disposition"], "rejected" if v3["reasons"] else "kept", label)
        # the set exercises both rules: some labels fail filters 1-3, some only coverage under version 2
        self.assertTrue(any(check(c[0], duration=c[1], ref=c[2], kind=c[3], b0=c[4], coverage=b0_like,
                                  version=2)["reasons"] == ["tokenizer_coverage"] for c in cases))

    @unittest.skipUnless(paths.MODEL_FILE.exists(), "B0 .nemo not available")
    def test_b0_tokenizer_covers_the_mapped_examples(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            cov = filters.Coverage(filters.b0_tokenizer_file(Path(tmp)))
            for raw, text in self.EXAMPLES.items():
                if raw in ('"', "これは100円です。"):
                    continue
                self.assertTrue(cov(text), (raw, text, cov.uncovered(text)))
            for c in '"()[]{}_;&+=@*':                                    # what version 3 maps away
                self.assertFalse(cov(f"a {c} b"), c)
            self.assertFalse(cov("これは100円です。"))
            for c in "#<>\\|~`":
                self.assertFalse(cov(f"a {c} b"), c)


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
