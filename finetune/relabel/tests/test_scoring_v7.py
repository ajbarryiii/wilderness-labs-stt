"""Contract tests (CPU): the scoring changes of cleaning plan v7 ("Scoring changes"): term recall counts a dose unit
and its abbreviation after a number as one term (terms/equivalents.tsv; a contract change, AJ 2026-10-30), and the
style-neutral WER's acronym canonicalizer is symmetric (every run of single letters joined after normalization, on
both sides), with a new `unit` group in `all`."""
from __future__ import annotations

import unittest

import evaluate
import score_domain as sd
import style_metrics as sm
import terms

MEDICAL = None


def occ(ref: str, hyp: str, matcher=None, **kw):
    global MEDICAL
    if matcher is None:
        MEDICAL = MEDICAL or terms.matcher("medical")
        matcher = MEDICAL
    return [(s, e, ok) for s, e, ok in sd.term_occurrences(evaluate.normalize(ref), evaluate.normalize(hyp), matcher,
                                                           **kw)]


class UnitTerms(unittest.TestCase):
    def test_a_unit_and_its_abbreviation_after_a_number_are_one_term(self):
        for ref, hyp in (("500 milligrams", "500 mg"), ("500 mg", "500 milligrams"),
                         ("five hundred milligrams a day", "500 mg a day"),
                         ("give 2 milligrams", "give 2 mg"), ("250 micrograms", "250 mcg"), ("250 mcg", "250 micrograms"),
                         ("20 milliequivalents", "20 mEq"), ("20 meq", "20 milliequivalents"),
                         ("3 millimoles", "3 mmol"), ("3 mmol", "3 millimoles"),
                         ("5 milligrams per kilogram", "5 mg/kg"),
                         ("2 micrograms per kilogram", "2 mcg/kg")):
            found = occ(ref, hyp)
            self.assertEqual(len(found), 1, (ref, hyp, found))
            self.assertTrue(found[0][2], (ref, hyp))

    def test_not_after_a_number(self):
        found = occ("a few milligrams", "a few mg")
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0][2])
        self.assertFalse(occ("milligrams per kilogram", "mg/kg")[0][2])

    def test_only_at_the_aligned_position_and_only_listed_forms(self):
        self.assertFalse(occ("500 milligrams today", "500 grams today and mg")[0][2])
        self.assertFalse(occ("500 milligrams", "500 mcg")[0][2])
        self.assertEqual(occ("10 ml", "10 milliliters"), [])            # "ml" is no listed term: no occurrence
        self.assertFalse(occ("10 milliliters", "10 ml")[0][2])          # and no equivalence

    def test_the_old_rule_has_no_equivalences(self):
        self.assertFalse(occ("500 milligrams", "500 mg", variants=False)[0][2])
        self.assertTrue(occ("500 milligrams", "500 milligrams", variants=False)[0][2])

    def test_a_matcher_without_the_table(self):
        m = terms.Matcher(["milligrams", "mg"], evaluate.normalize, equivalents=())
        self.assertFalse(occ("500 milligrams", "500 mg", m)[0][2])
        m = terms.Matcher(["milligrams", "mg"], evaluate.normalize)
        self.assertTrue(occ("500 milligrams", "500 mg", m)[0][2])
        self.assertEqual(set(m.equivalents(("milligrams",))), {("mg",), ("milligram",), ("milligrams",)})
        self.assertEqual(m.equivalents(("aspirin",)), ())

    def test_the_committed_table_names_listed_medical_terms(self):
        listed = set(terms.matcher("medical").phrases)
        for group in terms.load_equivalents():
            for form in group:
                p = tuple(evaluate.normalize(form).split())
                self.assertEqual(" ".join(p), form, form)           # written as the normalizer writes it
                self.assertTrue(p in listed or form in ("mg kg", "mcg kg"), form)


def errors(ref: str, hyp: str, variant: str) -> int:
    return sm.utterance_errors(ref, hyp)[variant][0]


class AcronymCanonicalizer(unittest.TestCase):
    def test_the_plans_cases(self):
        self.assertEqual(errors("Halvorsen, H A L V O R S E N", "Haverson, HALVORSEN", "acr"), 1)
        self.assertEqual(errors("M-O-R-E-A-U", "M O R E A U", "acr"), 0)
        self.assertEqual(errors("U N D E R W O O D", "UNDERWOOD", "acr"), 0)
        self.assertEqual(errors("O B R I E N here", "OBRIEN here", "acr"), 0)
        self.assertEqual(errors("M U N R O", "Munro", "acr"), 0)
        e = sm.utterance_errors("I I think so", "I I think so")
        self.assertEqual(e["acr"], (0, 4))                             # a run of only "i" is not joined
        self.assertEqual(sm.utterance_errors("I I think so", "I think so")["acr"][0], 1)

    def test_symmetric(self):
        cases = [("Halvorsen, H A L V O R S E N", "Haverson, HALVORSEN"), ("the A. P. I. call", "the API call"),
                 ("call a p i now the cat", "call api now the bat"), ("U N D E R W O O D", "U N D E R W 0 0 D"),
                 ("F-I-T-Z-G-E-R-A-L-D", "FITZGERALS"), ("500 milligrams per kilogram", "500 mg/kg"),
                 ("we use the s q l table", "we use the sql tables")]
        for ref, hyp in cases:
            for v in sm.VARIANTS:
                self.assertEqual(sm.utterance_errors(ref, hyp)[v][0], sm.utterance_errors(hyp, ref)[v][0], (ref, v))

    def test_a_run_counts_as_one_word_and_a_wrong_letter_as_one_error(self):
        e = sm.utterance_errors("F I T Z G E R A L D", "F I T Z G E R A L T")
        self.assertEqual(e["acr"], (1, 1))
        self.assertEqual(e["std"], (1, 10))

    def test_reference_words_depend_on_the_reference_only(self):
        a = sm.utterance_errors("call a p i now", "call api now")["acr"]
        b = sm.utterance_errors("call a p i now", "call a p i now")["acr"]
        self.assertEqual((a[1], b[1]), (3, 3))


class UnitGroup(unittest.TestCase):
    def test_unit_forms_after_a_number_are_one_token(self):
        self.assertIn("unit", sm.VARIANTS)
        self.assertIn("unit", sm.VARIANTS["all"])
        for ref, hyp in (("500 milligrams", "500 mg"), ("5 milligrams per kilogram", "5 mg/kg"),
                         ("120 millimeters of mercury", "120 mmHg"), ("10 milliliters", "10 mL"),
                         ("0.5 micrograms", "0.5 µg"), ("20 milliequivalents", "20 mEq")):
            e = sm.utterance_errors(ref, hyp)
            self.assertGreater(e["std"][0], 0, ref)
            self.assertEqual((e["unit"][0], e["all"][0]), (0, 0), ref)
        self.assertGreater(errors("a few milligrams", "a few mg", "unit"), 0)    # no number: not joined
        self.assertEqual(errors("500 milligrams", "500 grams", "unit"), 1)

    def test_identical_texts_have_no_error_in_any_variant(self):
        for text in ("5 milligrams per kilogram, H A L V O R S E N, I I think", "the U.S. and C T X-ray 10 mL"):
            self.assertTrue(all(x[0] == 0 for x in sm.utterance_errors(text, text).values()), text)


if __name__ == "__main__":
    unittest.main()
