"""Contract tests (CPU): cleaning version 3 (cleaning plan v7, Check 1's version 3 additions): spellings versus
acronyms (one case per class, trims, version 2's exclusions, the output form, never compacting what version 2 leaves,
idempotence, the lexicon artifact and its identity), dose units after a digit (every conversion and every case left as
written, "per" under both settings), the Schrödinger rows and their loader rule, the OpEx set, and version 2's
reproducibility (its texts and identity are today's, byte for byte). Store and runner contracts of version 3 are in
test_training_cleaning_store.py."""
from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

from training import cleaning as C

CUES = (C.TABLES_DIR / "spelling_cues.tsv").read_text(encoding="utf-8")
UNITS = (C.TABLES_DIR / "units.tsv").read_text(encoding="utf-8")


def tables3(forms="", variants="", dialect="", choices="", variants_v3="", choices_v3="", cues=CUES, units=UNITS,
            coverage=lambda text: True) -> C.Tables:
    return C.Tables({"forms.tsv": forms, "variants.tsv": variants, "dialect.tsv": dialect, "choices.tsv": choices,
                     "spelling_cues.tsv": cues, "units.tsv": units, "variants_v3.tsv": variants_v3,
                     "choices_v3.tsv": choices_v3}, version=3, coverage=coverage)


def tables2(forms="", variants="", dialect="", choices="") -> C.Tables:
    return C.Tables({"forms.tsv": forms, "variants.tsv": variants, "dialect.tsv": dialect, "choices.tsv": choices})


# acronyms of the test lexicon: (written compactly, distinct recordings)
LEX = {"CPU": (900, 300), "LCD": (400, 270), "TFIDF": (40, 25), "VHS": (60, 40), "DVD": (500, 300), "SCC": (30, 20),
       "NASA": (300, 200)}


def lexicon(counts=None, contract="c" * 64) -> C.Lexicon:
    return C.Lexicon.from_counts(LEX if counts is None else counts, contract_sha256=contract)


SPEC = {}


def spec3(**kw) -> C.Cleaning:
    key = json.dumps(kw, sort_keys=True, default=str)
    if key not in SPEC:
        lex = kw.pop("lex", None)
        per = kw.pop("units_per", True)
        SPEC[key] = C.Cleaning(tables3(**kw), version=3, lexicon=lexicon(lex), units_per=per)
    return SPEC[key]


def v3(text: str, **kw) -> str:
    return spec3(**kw).clean(text, "i", "B0")


def classes(text: str, **kw) -> list[tuple[str, str, str]]:
    """(run as written, class, evidence) of every run the rule judges in text."""
    return [(d.run, d.cls, d.evidence) for d in spec3(**kw).classify(text)]


class Classes(unittest.TestCase):
    """Cleaning plan v7 "Spellings versus acronyms": one case per class, first match wins."""

    def test_echo(self):
        t = "I advise my clients to cultivate a feeling of leisure, L E I S U R E, in the movements"
        self.assertEqual(v3(t), "I advise my clients to cultivate a feeling of leisure, L-E-I-S-U-R-E, in the movements")
        self.assertEqual(classes(t), [("L E I S U R E", "spelling", "echo")])
        # among the next 3 words; case-insensitive
        self.assertEqual(v3("she wrote K L I B A N O for klibano yesterday"),
                         "she wrote K-L-I-B-A-N-O for klibano yesterday")
        self.assertEqual(v3("she wrote K L I B A N O for the man Klibano"), "she wrote KLIBANO for the man Klibano")
        # among the previous 12 words, not the 13th
        words = "aa bb cc dd ee ff gg hh ii jj kk"
        self.assertEqual(v3(f"Klibano {words} K L I B A N O"), f"Klibano {words} K-L-I-B-A-N-O")
        self.assertEqual(v3(f"Klibano {words} ll K L I B A N O"), f"Klibano {words} ll KLIBANO")

    def test_echo_by_edit_distance(self):
        # distance 1 at 4+ letters, 2 at 7+
        self.assertEqual(v3("the doctor is Medaris, M E D A R S, today"), "the doctor is Medaris, M-E-D-A-R-S, today")
        self.assertEqual(classes("the doctor is Medaris, M E D A R S, today")[0][1:], ("spelling", "echo"))
        self.assertEqual(v3("we met Hernandez, H E R N A N D S, today"), "we met Hernandez, H-E-R-N-A-N-D-S, today")
        self.assertEqual(v3("we met Medaris, M E D R S, today"), "we met Medaris, MEDRS, today")   # 2 edits at 5
        self.assertEqual(v3("the cat, C A X, ran"), "the cat, CAX, ran")                             # 1 edit at 3

    def test_strong_cue_before_and_after(self):
        self.assertEqual(v3("It's spelt S A S Q U A T C H."), "It's spelt S-A-S-Q-U-A-T-C-H.")
        self.assertEqual(classes("It's spelt S A S Q U A T C H.")[0][1:], ("spelling", "cue"))
        self.assertEqual(v3("my last name is K L I B A N"), "my last name is K-L-I-B-A-N")
        # within the previous 6 words only
        self.assertEqual(v3("spelled one two three four five X Q Z V"), "spelled one two three four five X-Q-Z-V")
        self.assertEqual(v3("spelled one two three four five six X Q Z V"), "spelled one two three four five six XQZV")
        # an address right after the run
        self.assertEqual(v3("reach J S M I T H at gmail"), "reach J-S-M-I-T-H at gmail")
        self.assertEqual(v3("the plate Q T R W, as in Quebec"), "the plate Q-T-R-W, as in Quebec")
        self.assertEqual(classes("reach J S M I T H at gmail")[0][1:], ("spelling", "cue"))
        # weak cues are not used
        self.assertEqual(v3("a protocol called X Q Z V here"), "a protocol called XQZV here")

    def test_lexicon(self):
        # an acronym of the lexicon keeps version 2's output (here: a lowercase hyphenated run, left as written)
        self.assertEqual(classes("the t-f-i-d-f score"), [("t-f-i-d-f", "acronym", "lexicon")])
        self.assertEqual(v3("the t-f-i-d-f score"), "the t-f-i-d-f score")
        self.assertEqual(v3("the t-f-i-d-f score", lex={}), "the T-F-I-D-F score")          # not in it: hyphen
        self.assertEqual(v3("the T. F. I. D. F. score"), "the TFIDF score")
        # written compactly 5+ times and in 10+ recordings
        for counts in ({"TFIDF": (4, 25)}, {"TFIDF": (40, 9)}):
            self.assertEqual(classes("the t-f-i-d-f score", lex=counts)[0][1], "spelling", counts)
        self.assertEqual(classes("the t-f-i-d-f score", lex={"TFIDF": (5, 10)})[0][1], "acronym")

    def test_lexicon_split(self):
        self.assertEqual(classes("on v-h-s-d-v-d tapes"), [("v-h-s-d-v-d", "acronym", "split")])
        self.assertEqual(v3("on v-h-s-d-v-d tapes"), "on v-h-s-d-v-d tapes")
        self.assertEqual(v3("on V. H. S. D. V. D. tapes"), "on VHSDVD tapes")           # one compact token
        # each part 3-5 letters, written compactly 20+ times and in 10+ recordings
        for counts in ({"VHS": (19, 40), "DVD": (500, 300)}, {"VHS": (60, 9), "DVD": (500, 300)}):
            self.assertEqual(classes("on v-h-s-d-v-d tapes", lex=counts)[0][1], "spelling", counts)
        self.assertEqual(classes("on v-h-d-v-d tapes", lex={"VH": (900, 300), "DVD": (500, 300)})[0][1], "spelling")

    def test_hyphen(self):
        self.assertEqual(classes("the code X-Q-Z-V here"), [("X-Q-Z-V", "spelling", "hyphen")])
        self.assertEqual(v3("the code X-Q-Z-V here"), "the code X-Q-Z-V here")
        self.assertEqual(v3("the code x-q-z-v here"), "the code X-Q-Z-V here")

    def test_lowercase_without_evidence_is_left_as_written(self):
        self.assertEqual(classes("minus k v t l here"), [("k v t l", "other", "lowercase")])
        self.assertEqual(v3("minus k v t l here"), "minus k v t l here")
        self.assertEqual(v3("type c o n v e r t to convert"), "type C-O-N-V-E-R-T to convert")   # with evidence

    def test_three_capitals_or_fewer(self):
        self.assertEqual(classes("the Q X Z unit"), [("Q X Z", "acronym", "short")])
        self.assertEqual(v3("the Q X Z unit"), "the QXZ unit")
        self.assertEqual(v3("from D.H.S. today"), "from DHS today")

    def test_four_or_more_unevidenced_capitals_are_compacted(self):
        self.assertEqual(classes("at T. P. F. L. there is a lab"), [("T. P. F. L.", "ambiguous", "")])
        self.assertEqual(v3("at T. P. F. L. there is a lab"), "at TPFL there is a lab")
        self.assertEqual(v3("the A B I C O M P system"), "the ABICOMP system")

    def test_other_runs_are_left_as_written(self):
        for t, ev in (("files X A A, X A B, X A C here", "comma"), ("use the A B C D approach", "alphabet"),
                      ("use the A.B.C.D.E.F.G.H.I. approach", "alphabet"), ("the riff E G D E G D again", "notes")):
            self.assertEqual(v3(t), t, t)
            self.assertEqual(classes(t)[0][1:], ("other", ev), t)


class Trims(unittest.TestCase):
    """Class 1: a final "I" or "A" after a space or a comma, followed by a lowercase word, is split off."""

    def test_trims(self):
        self.assertEqual(v3("those of us in the U.S. I hope"), "those of us in the U.S. I hope")      # v2: USI
        self.assertEqual(v3("some kind of C. P. U. I guess"), "some kind of CPU I guess")             # v2: CPUI
        self.assertEqual(v3("A. B, I would"), "AB, I would")                                          # as version 2
        self.assertEqual(v3("Rossi, R O S S I said"), "Rossi, R-O-S-S I said")                       # the stated cost
        for t in ("those of us in the U.S. I hope", "some kind of C. P. U. I guess", "A. B, I would",
                  "Rossi, R O S S I said"):
            (d,) = spec3().classify(t)
            self.assertEqual(d.cls, "trim", t)
            self.assertIsNotNone(d.rest)
        (d,) = spec3().classify("Rossi, R O S S I said")
        self.assertEqual((d.rest.run, d.rest.cls, d.rest.evidence), ("R O S S", "spelling", "echo"))

    def test_non_trims(self):
        self.assertEqual(v3("S.L.I. was attached"), "SLI was attached")          # a dotted final "I." without a space
        self.assertEqual(classes("S.L.I. was attached")[0][1], "acronym")
        self.assertEqual(v3("the C P U I. Then"), "the CPUI. Then")                # no lowercase word follows
        self.assertEqual(v3("the X-Y-A said"), "the X-Y-A said")                  # a hyphen is no space or comma
        self.assertEqual(classes("the X-Y-A said")[0][1], "spelling")


class Exclusions(unittest.TestCase):
    """Version 2's pre-classes and exclusions keep their behaviour and take precedence."""

    def test_exclusions(self):
        for t in ("J. A. Mellon wrote", "by J. R. R. Tolkien", "…the A. P. I. We went", "I I think", "I, I, I think",
                  "the U.S. economy", "the T V is on", "grades A B and"):
            self.assertEqual(v3(t), t, t)
        self.assertEqual([d.cls for d in spec3().classify("by J. R. R. Tolkien and I I I think")],
                         ["initials", "i-repeat"])
        self.assertEqual(v3("two T. V. sets"), "two TV sets")                       # a pair: version 2's output
        self.assertEqual(classes("two T. V. sets")[0][1], "pair")
        # initials take precedence over a spelling cue
        self.assertEqual(v3("spelled F I T Z Gerald"), "spelled F I T Z Gerald")


class OutputForm(unittest.TestCase):
    def test_hyphenated_capitals_final_period_and_plural(self):
        self.assertEqual(v3("spelled F I T Z G E R A L D here"), "spelled F-I-T-Z-G-E-R-A-L-D here")
        self.assertEqual(v3("It's spelled s-e-n-n-a."), "It's spelled S-E-N-N-A.")
        self.assertEqual(v3("spelled F. I. T. Z. G. here"), "spelled F-I-T-Z-G here")
        # version 2's final-period rule: kept only as the label's last character
        self.assertEqual(v3("He'll write it like Butch B. U. T. C. H."), "He'll write it like Butch B-U-T-C-H.")
        self.assertEqual(v3("like Butch B. U. T. C. H. and"), "like Butch B-U-T-C-H and")
        # an undotted run's period is the sentence's
        self.assertEqual(v3("It's spelt S A S Q U A T C H. Then"), "It's spelt S-A-S-Q-U-A-T-C-H. Then")
        # an attached plural stays as written
        self.assertEqual(v3("Fitz, F I T Z's here"), "Fitz, F-I-T-Z's here")
        self.assertEqual(v3("Fitz, F I T Z s here"), "Fitz, F-I-T-Z s here")

    def test_edits(self):
        spec = spec3()
        text = "spelled F I T Z, and then the other one, the C. P. U. I guess, and 10 milligrams"
        out, edits = spec.edits(text, "i", "B0")
        self.assertEqual(out, "spelled F-I-T-Z, and then the other one, the CPU I guess, and 10 mg")
        self.assertEqual([(e.layer, e.table, e.group, e.original, e.replacement) for e in edits],
                         [(1, "acronyms", "spelling", "F I T Z", "F-I-T-Z"),
                          (1, "acronyms", "spelled-out", "C. P. U.", "CPU"),
                          (1, "units", "mg", " milligrams", " mg")])
        for e in edits:
            self.assertEqual((text[e.start:e.end], out[e.out_start:e.out_end]), (e.original, e.replacement))


CORPUS_PIECES = ["A", "P", "I", "G", "U", "T", "V", "J", "M", "s", "'s", "Mellon", "the", "U.S.", "html", "gonna",
                 "okay", "data", "set", "10", "milligrams", "mg", "per", "kilogram", "F", "Z", "R", "O", "S", "E", "-",
                 ",", "é", "5-10", "2.5", "maybe", "spelled", "Fitz", "hope", "guess", "said", "CPU", "C", "D", "H",
                 "x", "q", "k", "v", "Rossi", "Ross", "a", "i", "L", "B", "X", "Q", "mcg", "ml", "MG"]
CORPUS_SEPS = [" ", " ", ". ", ".", "", "-", ", ", "  "]


def random_texts(seed: int, n: int):
    rng = random.Random(seed)
    for _ in range(n):
        yield "".join(rng.choice(CORPUS_PIECES) + rng.choice(CORPUS_SEPS)
                      for _ in range(rng.randint(1, 12))).strip()


class Invariants(unittest.TestCase):
    def test_version_3_never_compacts_a_run_version_2_leaves_as_written(self):
        three = C.Cleaning(tables3(forms="html\tHTML\n"), version=3, lexicon=lexicon())
        two = C.Cleaning(tables2(forms="html\tHTML\n"), version=2)
        for t in random_texts(17, 4000):
            e3 = [e for e in three.edits(t, "i", "B0")[1] if (e.table, e.group) == ("acronyms", "spelled-out")]
            e2 = [e for e in two.edits(t, "i", "B0")[1] if e.table == "acronyms"]
            for e in e3:
                self.assertTrue(any(e.start < f.end and f.start < e.end for f in e2), (t, e))

    def test_layer1_is_idempotent_and_never_empties(self):
        spec = C.Cleaning(tables3(forms="html\tHTML\nmacos\tmacOS\n", variants="OK\tokay\n",
                                  dialect="gonna\tgoing to\n"), version=3, lexicon=lexicon())
        for t in random_texts(23, 6000):
            once = spec.clean(t, "i", "B0")
            self.assertEqual(spec.clean(once, "i", "B0"), once, t)
            self.assertEqual(bool(once), bool(t), t)
            out, edits = spec.edits(t, "i", "B0")
            pos, rebuilt = 0, []
            for e in edits:
                rebuilt += [t[pos:e.start], e.replacement]
                pos = e.end
            self.assertEqual("".join(rebuilt) + t[pos:], out, t)

    def test_both_label_sets_are_cleaned_identically(self):
        spec = spec3()
        for t in list(random_texts(29, 500)) + ["Rossi, R O S S I said 10 milligrams", "spelled s-e-n-n-a, 5 ml"]:
            self.assertEqual(spec.clean(t, "i", "T*"), spec.clean(t, "i", "B0"), t)

    def test_committed_tables_load_and_their_outputs_are_fixed_points(self):
        tabs = C.Tables.load(version=3)
        spec = C.Cleaning(tabs, version=3, lexicon=lexicon())
        for e in tabs.layer1.entries.values():
            self.assertEqual(spec.clean(e.output, "i", "B0"), e.output)
        for k, abbr in tabs.layer1.units.items():
            self.assertEqual(spec.clean(f"5 {abbr}", "i", "B0"), f"5 {abbr}")
        for f in C.TABLE_FILES + C.TABLE_FILES_V3:
            text = (C.TABLES_DIR / f).read_text(encoding="utf-8")
            self.assertNotIn("/mnt", text)
            self.assertNotIn("/home", text)


class Units(unittest.TestCase):
    """Cleaning plan v7 "Dose units after a digit" (layer 1, after the spelling rule, before the tables)."""

    def test_every_listed_conversion(self):
        for t, want in (("give 10 milligrams now", "give 10 mg now"), ("1 milligram", "1 mg"),
                        ("500 micrograms", "500 mcg"), ("1 microgram", "1 mcg"), ("5 µg", "5 mcg"),
                        ("5 μg", "5 mcg"), ("5 ug", "5 mcg"), ("5 mcg", "5 mcg"), ("10 milliliters", "10 mL"),
                        ("1 milliliter", "1 mL"), ("10 millilitres", "10 mL"), ("1 millilitre", "1 mL"),
                        ("70 kilograms", "70 kg"), ("1 kilogram", "1 kg"), ("20 milliequivalents", "20 mEq"),
                        ("1 milliequivalent", "1 mEq"), ("3 millimoles", "3 mmol"), ("1 millimole", "1 mmol"),
                        ("120 millimeters of mercury", "120 mmHg"), ("1 millimeter of mercury", "1 mmHg"),
                        ("120 millimetres of mercury", "120 mmHg"), ("1 millimetre of mercury", "1 mmHg")):
            self.assertEqual(v3(t), want, t)

    def test_numbers_ranges_adjectives_marks_casings_and_plurals(self):
        for t, want in (("2.5 milligrams", "2.5 mg"), ("0.5 micrograms", "0.5 mcg"), ("1,000 micrograms", "1,000 mcg"),
                        ("12,500.5 milligrams", "12,500.5 mg"), ("5-10 milligrams", "5-10 mg"),
                        ("5 to 10 milligrams", "5 to 10 mg"), ("a 10-milligram dose", "a 10 mg dose"),
                        ("a 10-mg dose", "a 10 mg dose"), ("take 10 milligrams.", "take 10 mg."),
                        ("10 milligrams, then", "10 mg, then"), ("(10 milligrams)", "(10 mg)"),
                        ("10 MG", "10 mg"), ("10 Mg", "10 mg"), ("5 ml", "5 mL"), ("5 ML", "5 mL"), ("70 Kg", "70 kg"),
                        ("120 mmhg", "120 mmHg"), ("120 MMHG", "120 mmHg"), ("20 meq", "20 mEq"), ("3 MMOL", "3 mmol"),
                        ("10 Milligrams", "10 mg"), ("10 mgs", "10 mg"), ("5 mls", "5 mL"), ("2 kgs", "2 kg"),
                        ("5 mcgs", "5 mcg"), ("5 meqs", "5 mEq"), ("2 mmols", "2 mmol")):
            self.assertEqual(v3(t), want, t)

    def test_left_as_written(self):
        for t in ("two milligrams a day", "a milligram", "milligrams of it", "10 units", "10 U", "10 IU", "5 cc",
                  "10 grams", "10 g", "50 percent", "the dose in milligrams", "vitamin B12 micrograms",
                  "COVID-19 milligrams", "milligrams per kilogram", "10 mg/kg", "ten mg"):
            self.assertEqual(v3(t), t, t)

    def test_per_under_both_settings(self):
        for t, on, off in (("5 milligrams per kilogram", "5 mg/kg", "5 mg per kilogram"),
                           ("5 mg per kg", "5 mg/kg", "5 mg per kg"),
                           ("0.1 micrograms per kilogram per minute", "0.1 mcg/kg per minute",
                            "0.1 mcg per kilogram per minute"),
                           ("5 milligrams per dose", "5 mg per dose", "5 mg per dose"),
                           ("5-10 milligrams per kilogram.", "5-10 mg/kg.", "5-10 mg per kilogram.")):
            self.assertEqual(v3(t), on, t)
            self.assertEqual(v3(t, units_per=False), off, t)
        self.assertNotEqual(spec3().identity(), spec3(units_per=False).identity())

    def test_idempotent_and_identical_for_both_label_sets(self):
        spec = spec3()
        for t in ("5 milligrams per kilogram", "a 10-milligram dose.", "5 µg and 2 ML", "1,000 MCGS"):
            once = spec.clean(t, "i", "T*")
            self.assertEqual(spec.clean(once, "i", "T*"), once, t)
            self.assertEqual(once, spec.clean(t, "i", "B0"), t)

    def test_no_table_entry_matches_a_unit_form(self):
        for kw in ({"forms": "ml\tML\n"}, {"variants": "mg\tmilligramme\n"}, {"choices": "mg\tMG2\n"},
                   {"dialect": "kilos\tkilograms\n"}):
            with self.assertRaises(ValueError, msg=kw):
                tables3(**kw)


class Schrodinger(unittest.TestCase):
    """Cleaning plan v7 "Schrödinger": variant rows of version 3, a non-ASCII canonical form B0's tokenizer covers."""

    def test_rows(self):
        spec = C.Cleaning(C.Tables.load(version=3), version=3, lexicon=lexicon())
        for t, want in (("the Schroedinger equation", "the Schrödinger equation"),
                        ("the Schrodinger equation", "the Schrödinger equation"),
                        ("Schroedinger's cat", "Schrödinger's cat"), ("schrodinger's cat", "Schrödinger's cat"),
                        ("the schrödinger equation", "the Schrödinger equation"),
                        ("Schroeder and Schröder", "Schroeder and Schröder")):
            self.assertEqual(spec.clean(t, "i", "B0"), want, t)
        # version 2 reads variants.tsv alone
        self.assertEqual(C.Cleaning(C.Tables.load(), version=2).clean("the Schroedinger equation", "i", "B0"),
                         "the Schroedinger equation")

    def test_the_loader_accepts_a_non_ascii_canonical_only_when_b0s_tokenizer_covers_it(self):
        row = "Schrödinger\tSchroedinger\n"
        self.assertEqual(tables3(variants_v3=row).layer1.apply("Schroedinger")[0], "Schrödinger")
        with self.assertRaises(ValueError):
            tables3(variants_v3=row, coverage=lambda text: False)
        with self.assertRaises(ValueError):                       # only the canonical side may be non-ASCII
            tables3(variants_v3="Schroedinger\tSchrödinger\n")
        with self.assertRaises(ValueError):
            tables3(forms="schrödinger\tSchrödinger\n")
        from labeling import filters
        with tempfile.TemporaryDirectory() as d:
            cov = filters.Coverage(filters.b0_tokenizer_file(Path(d)))
            tables3(variants_v3=row, coverage=cov)
            with self.assertRaises(ValueError):
                tables3(variants_v3="薛定谔\tXue\n", coverage=cov)


class OpEx(unittest.TestCase):
    def test_the_set_and_its_option_only_member(self):
        tabs = C.Tables.load(version=3)
        got = tabs.spans("we cut OPX, opx and Opx costs, OpEx and opex")
        self.assertEqual([s.original for s in got], ["OPX", "opx", "Opx"])
        self.assertEqual(got[0].set.options, ("OpEx", "OPX", C.CANT_TELL))
        self.assertEqual(got[0].set.option_only, frozenset({"OpEx"}))
        self.assertEqual(C.Tables.load().spans("we cut OPX costs"), [])                 # version 2: no such set
        text = "we cut Opx costs"
        sp = tabs.spans(text)
        art = Artifact(tabs, {("B0", "i"): [(0, sp[0].start, sp[0].end, sp[0].original, sp[0].set.name, "OpEx")]})
        self.assertEqual(C.Cleaning(tabs, art, version=3, lexicon=lexicon()).clean(text, "i", "B0"), "we cut OpEx costs")


class Artifact:
    def __init__(self, tabs: C.Tables, picks: dict, sha: str = "x" * 64, version: int = 3):
        self.manifest = {"version": version, "tables_sha256": tabs.sha256, "template_sha256": C.TEMPLATE_SHA256,
                         "threshold": C.THRESHOLD, "contract_sha256": "c" * 64, "file_sha256": sha}
        self.rows, self.dir, self.file_sha256 = picks, Path("mem"), sha


class LexiconArtifact(unittest.TestCase):
    def test_identity_carries_the_lexicon(self):
        tabs = tables3()
        a = C.Cleaning(tabs, version=3, lexicon=lexicon()).identity()
        self.assertEqual(a["version"], 3)
        self.assertEqual(a["lexicon_sha256"], lexicon().file_sha256)
        self.assertEqual(a["layer0"], C.LAYER0)
        b = C.Cleaning(tabs, version=3, lexicon=lexicon({"CPU": (900, 300)})).identity()
        self.assertNotEqual(a["lexicon_sha256"], b["lexicon_sha256"])
        self.assertNotIn("lexicon_sha256", C.Cleaning(tables2(), version=2).identity())

    def test_a_mismatched_or_missing_lexicon_is_refused(self):
        tabs = tables3()
        with self.assertRaises(C.ArtifactMismatch):
            C.Cleaning(tabs, version=3)                                   # version 3 reads the lexicon
        lex = lexicon()
        lex.manifest["thresholds"] = dict(lex.manifest["thresholds"], compact=4)
        with self.assertRaises(C.ArtifactMismatch):
            C.Cleaning(tabs, version=3, lexicon=lex)
        spec = C.Cleaning(tabs, version=3, lexicon=lexicon(contract="c" * 64))
        spec.bind_run("c" * 64)
        with self.assertRaises(C.ArtifactMismatch):
            spec.bind_run("d" * 64)
        with self.assertRaises(ValueError):                               # tables of another version
            C.Cleaning(tables2(), version=3, lexicon=lexicon())
        with self.assertRaises(ValueError):
            C.Cleaning(tabs, version=2)

    def test_the_lexicon_file_round_trips_and_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            lex = C.Lexicon.write(Path(d), LEX, contract_sha256="c" * 64, label_run="labels/x/run")
            back = C.Lexicon.load(Path(d))
            self.assertEqual(back.file_sha256, lex.file_sha256)
            self.assertEqual(back.manifest["contract_sha256"], "c" * 64)
            self.assertEqual(back.manifest["thresholds"], C.LEXICON)
            self.assertTrue(back.acronym("CPU") and not back.acronym("XYZ"))
            (Path(d) / C.LEXICON_FILE).write_bytes((Path(d) / C.LEXICON_FILE).read_bytes() + b"x")
            with self.assertRaises(C.ArtifactMismatch):
                C.Lexicon.load(Path(d))


class Versions(unittest.TestCase):
    def test_versions_and_directories(self):
        self.assertEqual(C.VERSIONS, (1, 2, 3))
        self.assertEqual(C.VERSION, 2)              # the default; version 3 (stages 4-5) is chosen explicitly
        from unittest import mock
        with mock.patch("labeling.layout.relabel_root", lambda: Path("/R")):
            self.assertEqual(C.production_dir("main-f3", 3), Path("/R/cleaning/main-f3/v3"))
            with mock.patch.object(C.Cleaning, "load", lambda **kw: kw):
                self.assertEqual(C.production("main-f3", 3)["lexicon_dir"], Path("/R/cleaning/main-f3/v3"))
                self.assertEqual(C.production("main-f3", 3)["choices_dir"], Path("/R/cleaning/main-f3/v3"))


V2_GOLDEN = {"tables_sha256": "87a960a6c7001a1b0be1a0b78681bfaa39f4d7e4fe126953b9e0db46321bee5a",
             "identity_sha256": "bb4540163f74ce308cf3b9412c36ceb9d945ac9825e7cef1d14ea7a1cd819e11",
             "outputs_sha256": "5ad00a3671cdcdcd01072d4088c3131195dc297d53042b49958bac79efc3c06b"}
GOLDEN_PIECES = ["A", "P", "I", "G", "U", "T", "V", "J", "M", "s", "'s", "Mellon", "the", "U.S.", "html", "gonna",
                 "okay", "data", "set", "10", "milligrams", "mg", "MG", "per", "kilogram", "Schroedinger",
                 "Schrodinger's", "OPX", "opx", "x-ray", "F", "Z", "R", "O", "S", "E", "-", ",", "é", "5-10", "2.5",
                 "1,000", "maybe", "cash", "ya", "p", "t", "d", "spelled", "Fitz", "hope", "guess", "said",
                 "micrograms", "mL", "ml", "kg", "millimeters", "of", "mercury", "CPU", "VHS", "DVD", "B", "C", "X-Y",
                 "s-e-n-n-a", "Rossi", "Ross", "ip", "Git", "a", "i"]
GOLDEN_SEPS = [" ", " ", ". ", ".", "", "-", ", ", "  "]


def golden_corpus(seed=20261030, n=3000):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        t = "".join(rng.choice(GOLDEN_PIECES) + rng.choice(GOLDEN_SEPS) for _ in range(rng.randint(1, 14))).strip()
        d = " ".join((w.capitalize() if rng.random() < 0.3 else w) + rng.choice(["", "", ",", ".", "?"])
                     for w in t.split())
        out.append((t, d))
    return out


class Version2Reproduced(unittest.TestCase):
    """With version 2 selected, the committed tables, the identity and the texts are today's (recorded with the code
    before version 3, 084fb2f), so stage 3's labels_sha256, its choices artifact and its resume are unchanged."""

    def test_tables_identity_and_texts_are_unchanged(self):
        tabs = C.Tables.load()
        self.assertEqual(tabs.sha256, V2_GOLDEN["tables_sha256"])              # the version 2 artifact's tables
        spec = C.Cleaning(tabs, version=2)
        self.assertEqual(hashlib.sha256(json.dumps(spec.identity(), sort_keys=True).encode()).hexdigest(),
                         V2_GOLDEN["identity_sha256"])
        h = hashlib.sha256()
        for t, d in golden_corpus():
            h.update(spec.clean(t, "i", "T*", d).encode() + b"\x00" + spec.clean(t, "i", "B0").encode() + b"\x01")
        self.assertEqual(h.hexdigest(), V2_GOLDEN["outputs_sha256"])
        self.assertNotEqual(C.Tables.load(version=3).sha256, tabs.sha256)


if __name__ == "__main__":
    unittest.main()
