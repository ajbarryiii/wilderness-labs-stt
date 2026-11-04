"""Contract tests (CPU): the output-style transforms (b0prime-v2 plan, "Output style: verbatim or clean", and "Code and
order of work", PR 1). training/style.py's four classes, each a deterministic text transform:

- E, ellipsis: a final ellipsis becomes "." (dropped after a sentence mark), any other is removed; "1...10" stays;
- F, fragments: a token ending in "-" (not a suspended hyphen) or a token of 1-3 letters (nothing else), not a listed
  short word, that is a strict prefix of the next word is removed;
- R, repetitions: an immediate repeat of a 1-3-word sequence keeps its last copy; never single letters other than
  "I" and "a", numbers or a copy made of one real double ("no no", "No, no, no, no");
- D, "you know" / "I mean" delimited by commas or the text's edges are removed.

Per class: the plan's positive and negative cases, idempotence, every token outside a changed span byte-identical,
numbers and letter runs never touched, no text emptied, the edits reproducing the output, and E against the pinned
normalizer's actual output. The short-word table is checked for its rule (1-3 lowercase letters, single letters only
"a" and "i") and its builder against a small SCOWL-shaped fixture.
"""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from training import style

ST = style.Style.load()


def T(text: str, cls: str) -> str:
    return ST.transform(text, cls)


# (input, class, expected output); the plan's list first ("Code and order of work", item 1), then further cases.
PLAN_CASES = [
    ("breathless or...", "E", "breathless or."),
    ("but... nah", "E", "but nah"),
    ("or…", "E", "or."),
    ("2.5… billion", "E", "2.5 billion"),
    ("1...10", "E", "1...10"),
    ("perso- and", "F", "perso- and"),
    ("perso- the", "F", "the"),
    ("w we", "F", "we"),
    ("a about", "F", "a about"),
    ("the the", "R", "the"),
    ("in a, in a", "R", "in a"),
    ("had had", "R", "had had"),
    ("F I T Z", "R", "F I T Z"),
    ("S A A B", "R", "S A A B"),
    ("U N D E R W O O D", "R", "U N D E R W O O D"),
    ("1 1", "R", "1 1"),
    ("I I think", "R", "I think"),
    ("so, you know, the", "D", "so the"),
    ("do you know", "D", "do you know"),
]

MORE_CASES = [
    # E: the end of the text, the sentence mark, closers, spacing, numbers, letter runs
    ("Really?...", "E", "Really?"),
    ("Really? ...", "E", "Really?"),
    ("Stop!…", "E", "Stop!"),
    ('He said "wait..."', "E", 'He said "wait."'),
    ("(and so on...)", "E", "(and so on.)"),
    ("or....", "E", "or."),
    ("or.…", "E", "or."),
    ("or ...", "E", "or ."),
    ("the end..", "E", "the end.."),
    ("Wait... what?", "E", "Wait what?"),
    ("but ... nah", "E", "but nah"),
    ("but ...nah", "E", "but nah"),
    ("... and then", "E", "and then"),
    ("...and then", "E", "and then"),
    ("a…b", "E", "a b"),
    ("wait…, no", "E", "wait, no"),
    ("breathless or…?", "E", "breathless or?"),
    ("word ...? yes", "E", "word? yes"),
    ("1…10 and 3...4", "E", "1…10 and 3...4"),
    ("10... 20", "E", "10 20"),
    ("F I T Z...", "E", "F I T Z..."),
    ("F I T Z... the", "E", "F I T Z... the"),
    ("no ellipsis here.", "E", "no ellipsis here."),
    # F: suspended hyphens, the short-word list, contractions, numbers, letter runs, case, chains
    ("perso- or", "F", "perso- or"),
    ("perso- to", "F", "perso- to"),
    ("perso- nor", "F", "perso- nor"),
    ("self- and peer-review", "F", "self- and peer-review"),
    ("we were- we are going", "F", "we we are going"),
    ("different th these", "F", "different these"),
    ("I was th- the one", "F", "I was the one"),
    ("Th- the cat", "F", "The cat"),
    ("It was. Th these are", "F", "It was. These are"),
    ("w, we went", "F", "w, we went"),
    ("Ada B. Bloomquist", "F", "Ada B. Bloomquist"),
    ("(th these)", "F", "(th these)"),
    ("an analysis", "F", "an analysis"),
    ("I'm impressed", "F", "I'm impressed"),
    ("I Ian", "F", "I Ian"),
    ("10- the", "F", "10- the"),
    ("twenty- the", "F", "twenty- the"),
    ("S A A B", "F", "S A A B"),
    ("A B C- the", "F", "A B C- the"),
    ("t th the", "F", "the"),
    ("th perso- these", "F", "these"),
    ("w", "F", "w"),
    ("the end-", "F", "the"),
    # R: longer sequences, triples, case, numbers, letters, keys, real doubles
    ("The the cat", "R", "The cat"),
    ("It was. The the end", "R", "It was. The end"),
    ("the the the cat", "R", "the cat"),
    ("I think I think so", "R", "I think so"),
    ("went to the, went to the store", "R", "went to the store"),
    ("we went to the, we went to the store", "R", "we went to the, we went to the store"),
    ("a a dog", "R", "a dog"),
    ("I I I think", "R", "I think"),
    ("one one", "R", "one one"),
    ("two oh oh five", "R", "two oh oh five"),
    ("twenty-one twenty-one", "R", "twenty-one twenty-one"),
    ("in 2020, in 2020", "R", "in 2020, in 2020"),
    ("B B", "R", "B B"),
    ("plan B plan B", "R", "plan B plan B"),
    ("C O O P E R", "R", "C O O P E R"),
    ("we're were", "R", "we're were"),
    ("that that", "R", "that that"),
    ("very very", "R", "very very"),
    ("really really", "R", "really really"),
    ("No, no.", "R", "No, no."),
    ("yeah yeah", "R", "yeah yeah"),
    ("bye bye", "R", "bye bye"),
    ("ha ha", "R", "ha ha"),
    ("so so", "R", "so so"),
    ("No, no, no, no, they, they left", "R", "No, no, no, no, they left"),
    ("yeah yeah yeah yeah yeah", "R", "yeah yeah yeah yeah yeah"),
    ("no no, no no", "R", "no no, no no"),
    ("I had, I had had", "R", "I had had"),
    ("the — the", "R", "the — the"),
    # D: the edges, case, consecutive markers, undelimited uses, letter runs
    ("You know, it's fine", "D", "It's fine"),
    ("I mean, it's fine", "D", "It's fine"),
    ("so, I mean, the", "D", "so the"),
    ("so, you know, I mean, the", "D", "so the"),
    ("You know, I mean, it's", "D", "It's"),
    ("it's good, you know.", "D", "it's good."),
    ("it's good, you know", "D", "it's good"),
    ("it's good, you know?", "D", "it's good?"),
    ("as you know", "D", "as you know"),
    ("if you know, then", "D", "if you know, then"),
    ("you know what", "D", "you know what"),
    ("so, you know the plan", "D", "so, you know the plan"),
    ("Well. You know, it's", "D", "Well. You know, it's"),
    ("in 2020, you know, the", "D", "in 2020 the"),
    ("A B C, you know, the", "D", "A B C, the"),
]

NO_EMPTY = [("I mean", "D"), ("you know,", "D"), ("You know.", "D"), ("perso-", "F"), ("I mean, you know", "D")]

CORPUS = sorted({c[0] for c in PLAN_CASES + MORE_CASES + NO_EMPTY} | {
    "Okay so, you know, the the th these perso- people were... w we went 1...10 times, I I think.",
    "Q U O R V A L period chart number one eight six two nine one",
    "F-I-T-Z, you know, F I T Z... in a, in a 2.5… billion deal",
    "  leading and trailing spaces, you know,   with  doubled   gaps the the  ",
    "So, you know, I mean, the the plan was w- what we... wanted…",
    "“You know, I think…” she said",
    "uh, um, the cat",
    "",
    "...",
})

NUMBER_WORDS = {"zero", "oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
                "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
                "billion", "trillion"}


def core(token: str) -> str:
    return re.match(r"^\W*(.*?)\W*$", token, re.S).group(1)


def protected(text: str) -> tuple[list[int], list[int]]:
    """The plan's definitions, written independently of style.py: indices of the tokens of letter runs (3+ consecutive
    single-letter tokens, with or without dots, unless only "I") and of number tokens (a digit, or a number word,
    which the pinned normalizer reads as a digit)."""
    toks = text.split()
    single = [len(core(t)) == 1 and core(t).isalpha() for t in toks]
    runs, i = [], 0
    while i < len(toks):
        j = i
        while j < len(toks) and single[j]:
            j += 1
        if j - i >= 3 and any(core(t).lower() != "i" for t in toks[i:j]):
            runs += range(i, j)
        i = max(j, i + 1)
    numbers = [k for k, t in enumerate(toks) if re.search(r"\d", core(t))
               or (core(t) and all(p in NUMBER_WORDS for p in core(t).lower().split("-")))]
    return runs, numbers


def uncapitalized(before: str, after: str) -> str:
    """after with one uppercased letter put back as before has it (at the first position where they differ)."""
    for k, (x, y) in enumerate(zip(before, after)):
        if x != y:
            return after[:k] + x + after[k + 1:] if x.islower() and y == x.upper() else after
    return after


def is_subsequence(needles: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(any(n == h for h in it) for n in needles)


class PlanCases(unittest.TestCase):
    def test_plan_cases(self):
        for text, cls, want in PLAN_CASES:
            with self.subTest(text=text, cls=cls):
                self.assertEqual(T(text, cls), want)

    def test_more_cases(self):
        for text, cls, want in MORE_CASES:
            with self.subTest(text=text, cls=cls):
                self.assertEqual(T(text, cls), want)

    def test_letter_runs_and_numbers_untouched_by_every_class(self):
        for text in ("F I T Z", "S A A B", "U N D E R W O O D", "C O O P E R", "F. I. T. Z.", "1 1", "one one",
                     "2020 2020", "oh oh"):
            for cls in style.CLASSES:
                with self.subTest(text=text, cls=cls):
                    self.assertEqual(T(text, cls), text)

    def test_classes_are_the_plans(self):
        self.assertEqual(style.CLASSES, ("E", "F", "R", "D"))
        with self.assertRaises(ValueError):
            T("the the", "X")


class Properties(unittest.TestCase):
    def test_idempotent(self):
        for text in CORPUS:
            for cls in style.CLASSES:
                with self.subTest(text=text, cls=cls):
                    once = T(text, cls)
                    self.assertEqual(T(once, cls), once)

    def test_no_text_emptied(self):
        for text, cls in NO_EMPTY:
            with self.subTest(text=text, cls=cls):
                self.assertEqual(T(text, cls), text)
        for text in CORPUS:
            for cls in style.CLASSES:
                if re.search(r"\w", text):
                    with self.subTest(text=text, cls=cls):
                        self.assertRegex(T(text, cls), r"\w")

    def test_edits_reproduce_the_output(self):
        for text in CORPUS:
            for cls in style.CLASSES:
                with self.subTest(text=text, cls=cls):
                    out, edits = ST.edits(text, cls)
                    self.assertEqual(out, T(text, cls))
                    pos, rebuilt = 0, []
                    for e in edits:
                        self.assertEqual(e.cls, cls)
                        self.assertGreaterEqual(e.start, pos)
                        self.assertLessEqual(e.start, e.end)
                        self.assertEqual(text[e.start:e.end], e.before)
                        self.assertNotEqual(e.before, e.after)
                        rebuilt += [text[pos:e.start], e.after]
                        pos = e.end
                    rebuilt.append(text[pos:])
                    self.assertEqual("".join(rebuilt), out)
                    self.assertEqual(edits == [], out == text)

    def test_tokens_outside_edits_are_byte_identical(self):
        """Every input token that no edit touches is in the output byte for byte, at its place; F, R and D also keep it
        a whole token (E may close the space before punctuation: "word ...? yes" -> "word? yes")."""
        for text in CORPUS:
            spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]
            for cls in style.CLASSES:
                with self.subTest(text=text, cls=cls):
                    out, edits = ST.edits(text, cls)
                    outside = [(s, e, t) for s, e, t in spans if not any(x.start < e and s < x.end for x in edits)]
                    for s, e, t in outside:
                        shift = sum(len(x.after) - len(x.before) for x in edits if x.end <= s)
                        self.assertEqual(out[s + shift:e + shift], t)
                    if cls != "E":
                        self.assertTrue(is_subsequence([t for _, _, t in outside], out.split()), (outside, out))

    def test_edits_touch_only_their_class(self):
        """E changes only ellipsis characters and the spaces around them; F, R and D remove whole tokens and change a
        kept token only by uppercasing its first letter (the text's or sentence's new first word) or, for D, by
        dropping the comma before a marker (or putting the final marker's closing mark there)."""
        for text in CORPUS:
            for cls in style.CLASSES:
                out, edits = ST.edits(text, cls)
                for e in edits:
                    with self.subTest(text=text, cls=cls, edit=e):
                        if cls == "E":
                            self.assertRegex(e.before, r"^[\s.…]*[.…]{1}[\s.…]*$")
                            self.assertIn(e.after, ("", " ", "."))
                        elif e.after == "":
                            whole = [m for m in re.finditer(r"\S+", text) if e.start < m.end() and m.start() < e.end]
                            self.assertTrue(whole)
                            self.assertTrue(all(e.start <= m.start() and m.end() <= e.end for m in whole), e)
                        else:
                            b, a = e.before, uncapitalized(e.before, e.after)
                            self.assertEqual(len(b.split()), 1)
                            self.assertEqual(b, b.strip())
                            case = a == b
                            comma = cls == "D" and b.endswith(",") and (
                                a == b[:-1] or (a.startswith(b[:-1]) and re.fullmatch(r"\W+", a[len(b) - 1:]) is not None))
                            self.assertTrue(case or comma, e)

    def test_letter_runs_and_numbers_never_touched(self):
        """Every token of a letter run is byte-identical in every class; a number token's core (its digits or number
        word) is byte-identical in every class, the whole token in F and R."""
        for text in CORPUS:
            runs, numbers = protected(text)
            toks = text.split()
            for cls in style.CLASSES:
                out = T(text, cls)
                with self.subTest(text=text, cls=cls):
                    self.assertTrue(is_subsequence([toks[k] for k in runs], out.split()), out)
                    want = [toks[k] if cls in "FR" else core(toks[k]) for k in numbers]
                    have = out.split() if cls in "FR" else [core(t) for t in out.split()]
                    self.assertTrue(is_subsequence(want, have), out)


class NormalizerEffect(unittest.TestCase):
    """E against the pinned normalizer (evaluate.normalize): what the scorer sees before and after E."""

    def test_e_against_the_pinned_normalizer(self):
        import evaluate
        for text, before, after in (("breathless or...", "breathless or .", "breathless or"),
                                    ("or…", "or ..", "or"),
                                    ("2.5… billion", "2.5 . 1000000000", "2500000000"),
                                    ("but... nah", "but nah", "but nah")):
            with self.subTest(text=text):
                self.assertEqual(evaluate.normalize(text), before)
                self.assertEqual(evaluate.normalize(T(text, "E")), after)


class ShortWords(unittest.TestCase):
    def test_committed_table(self):
        words = style.read_short_words(style.SHORT_WORDS)
        self.assertEqual(words, sorted(set(words)))
        self.assertTrue(all(re.fullmatch(r"[a-z]{1,3}", w) for w in words))
        self.assertEqual([w for w in words if len(w) == 1], ["a", "i"])
        for w in ("a", "i", "an", "the", "we", "and", "or", "to", "nor", "one", "two", "six", "ten", "oh", "ago"):
            self.assertIn(w, words)
        for w in ("w", "th", "r", "f", "d", "wh", "b"):
            self.assertNotIn(w, words)
        self.assertEqual(ST.short_words_sha256, style.sha256_file(style.SHORT_WORDS))
        header = [line for line in style.SHORT_WORDS.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
        self.assertTrue(any(style.SCOWL_COPYRIGHT.name in line for line in header))
        notices = style.SCOWL_COPYRIGHT.read_text(encoding="utf-8")
        for notice in ("Copyright 2000-2018 by Kevin Atkinson", "WordNet 1.6 Copyright 1997 by Princeton University",
                       "Copyright 2000-2016 by Kevin Atkinson", "Copyright 1993, Geoff Kuenning",
                       "Copyright (c) J Ross Beresford 1993-1999"):
            self.assertIn(notice, notices)
        self.assertEqual(ST.identity(), {"classes": list(style.CLASSES), "short_words_sha256": ST.short_words_sha256})

    def test_builder_rule(self):
        """English and American word lists of SCOWL up to the given size, 1-3 lowercase ASCII letters, single letters
        only "a" and "i"; abbreviations, proper names and larger sizes are not read."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "english-words.10").write_text("a\nb\ni\nthe\nwe\nabout\nDon\n", encoding="latin-1")
            (d / "english-words.60").write_text("sh\nre\ncafé\n", encoding="latin-1")
            (d / "american-words.50").write_text("ax\n", encoding="latin-1")
            (d / "english-words.70").write_text("th\n", encoding="latin-1")
            (d / "english-abbreviations.10").write_text("cm\n", encoding="latin-1")
            (d / "english-proper-names.10").write_text("Al\n", encoding="latin-1")
            self.assertEqual(style.build_short_words(d, size=60), ["a", "ax", "i", "re", "sh", "the", "we"])
            text = style.short_words_table(style.build_short_words(d, size=60), "SCOWL test")
            with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
                f.write(text)
            self.assertEqual(style.read_short_words(Path(f.name)), ["a", "ax", "i", "re", "sh", "the", "we"])
            Path(f.name).unlink()

    def test_another_table_changes_f_only_through_the_list(self):
        st = style.Style(frozenset({"a", "i", "we", "th"}))
        self.assertEqual(st.transform("th these", "F"), "th these")
        self.assertEqual(st.transform("w we", "F"), "we")
        self.assertNotEqual(st.short_words_sha256, ST.short_words_sha256)


if __name__ == "__main__":
    unittest.main()
