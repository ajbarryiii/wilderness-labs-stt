"""Contract tests: hygiene of generated sentences (synthetic plan "Text", "Terms").

- held-out terms (of any domain) never appear in kept text;
- sentences matching the medical or TCCC term lists are rejected;
- the 8-gram containment denylist (Part B's rule) is applied against held-out texts;
- exact and near duplicates are removed; spoken-form and length rules; the requested terms must be used;
- a run without eval-ext / eval-stem is provisional.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_synthetic_*' -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from synthetic import hygiene, layout, vocab
from synthetic_fixtures import heldout_sets, make_split, normalize, record

HELDOUT_UTT = ("so the next thing we want to look at is how the market reacted to the earnings call "
               "on the first day of trading")


class Hygiene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = make_split()
        cls.n = staticmethod(normalize())
        cls.med = vocab.medical_matcher(normalize())
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sets = heldout_sets(Path(cls.tmp.name), {"dev-x": [HELDOUT_UTT, "short one"]})
        cls.held = sorted(cls.split.heldout(), key=lambda e: e.term)
        cls.targeted = {d: sorted(cls.split.targeted(d), key=lambda e: e.term) for d in layout.DOMAINS}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_h(self, records, coverage=lambda t: True):
        return hygiene.run(records, self.split, self.med, self.n, coverage, self.sets,
                           {"provisional": False, "sets": []})

    def t(self, domain):
        return self.targeted[domain][0].term

    def test_heldout_terms_of_any_domain_never_appear_in_kept_text(self):
        self.assertTrue(self.held, "the fixture split needs held-out terms")
        recs = []
        for i, h in enumerate(self.held):
            for j, d in enumerate(layout.DOMAINS):        # the held-out term may be of another domain
                tt = self.t(d)
                recs.append(record(len(recs), d, f"We discussed the {tt} and the {h.term} in detail today.", [tt]))
        recs.append(record(len(recs), "chemistry", f"We discussed the {self.t('chemistry')} in detail today.",
                           [self.t("chemistry")]))
        kept, report, rows = self.run_h(recs)
        held = self.split.heldout_matcher()
        for k in kept:
            self.assertEqual(held.find(self.n(k["label"]).split()), [], k["label"])
        self.assertEqual(report["reasons"].get("heldout_term"), len(recs) - 1)
        self.assertEqual(len(kept), 1)

    def test_medical_and_tccc_sentences_are_rejected(self):
        tt = self.t("chemistry")
        recs = [record(0, "chemistry", f"The {tt} matters when a tourniquet is applied in the field.", [tt]),
                record(1, "chemistry", f"Think about the {tt} before the ketamine arrives at the lab.", [tt]),
                record(2, "chemistry", f"Think about the {tt} before the samples arrive at the lab.", [tt])]
        kept, report, rows = self.run_h(recs)
        self.assertEqual([r["reason"] for r in rows], ["medical", "medical", None])
        self.assertEqual([k["id"] for k in kept], [recs[2]["id"]])

    def test_the_8gram_containment_denylist_is_applied(self):
        tt = self.t("finance")
        recs = [record(0, "finance", f"Honestly {HELDOUT_UTT} and the {tt} moved too.", [tt]),
                record(1, "finance", f"The {tt} moved after the earnings call on the first day of trading.", [tt])]
        kept, report, rows = self.run_h(recs)
        self.assertEqual([r["reason"] for r in rows], ["denylist", None])
        self.assertEqual(report["denylist"]["denylisted"], 1)

    def test_exact_and_near_duplicates_are_removed_in_id_order(self):
        tt = self.t("software_engineering")
        base = f"The team replaced the {tt} in the old service because it kept failing under heavy load at night"
        recs = [record(0, "software_engineering", base + ".", [tt]),
                record(1, "software_engineering", base.upper() + "!", [tt]),                # same normalized text
                record(2, "software_engineering", base.replace("old", "legacy") + ".", [tt]),  # one word changed
                record(3, "software_engineering", f"Nobody on the platform group understands why the {tt} "
                                                  "was chosen for the payment flow.", [tt])]
        kept, report, rows = self.run_h(recs)
        self.assertEqual([r["reason"] for r in rows], [None, "duplicate", "near_duplicate", None])

    def test_spoken_form_length_target_and_tokenizer_rules(self):
        tt = self.t("mathematics")
        recs = [record(0, "mathematics", f"The {tt} is 42 here.", [tt]),
                record(1, "mathematics", f"The {tt} uses x = y + 2.", [tt]),
                record(2, "mathematics", f"{tt}", [tt]),
                record(3, "mathematics", " ".join(["very"] * 70) + f" {tt}.", [tt]),
                record(4, "mathematics", "We computed something entirely unrelated today.", [tt]),
                record(5, "mathematics", "", [tt]),
                record(6, "mathematics", f"\"The {tt} is twenty five percent larger than before.\"", [tt]),
                record(7, "mathematics", f"The {tt} looks fine to everyone here.", [tt])]
        kept, report, rows = self.run_h(recs, coverage=lambda text: "everyone" not in text)
        self.assertEqual([r["reason"] for r in rows],
                         ["form_symbols", "form_symbols", "form_length", "form_length", "no_target", "no_sentence",
                          None, "tokenizer"])
        self.assertFalse(kept[0]["label"].startswith('"'))

    def test_a_variant_of_a_requested_term_counts_as_using_it(self):
        e = next(e for e in self.split.entries if e.term == "rate constants")
        recs = [record(0, e.domain, "The rate constant doubles when the temperature rises by ten degrees.",
                       ["rate constants"])]
        kept, report, rows = self.run_h(recs)
        if e.side == "targeted":
            self.assertEqual(rows[0]["reason"], None)
        else:
            self.assertEqual(rows[0]["reason"], "heldout_term")

    def test_counts_per_domain(self):
        tt = self.t("chemistry")
        recs = [record(0, "chemistry", f"The {tt} was measured twice in the morning session.", [tt]),
                record(1, "chemistry", "", [tt])]
        kept, report, rows = self.run_h(recs)
        self.assertEqual(report["by_domain"]["chemistry"], {"generated": 2, "kept": 1, "reasons": {"no_sentence": 1}})
        self.assertEqual(report["by_domain"]["finance"]["generated"], 0)


class Provisional(unittest.TestCase):
    def sets_with(self, names):
        from unittest import mock
        from synthetic_fixtures import write_manifest
        with tempfile.TemporaryDirectory() as tmp:
            paths = [(n, write_manifest(Path(tmp) / f"{i}.jsonl", [{"text": "one two three"}])) for i, n in enumerate(names)]
            with mock.patch("pool.__main__.heldout_manifests", lambda extra: (paths, [])):
                return hygiene.heldout_sets()[1]

    def test_runs_without_eval_ext_or_eval_stem_are_provisional(self):
        base = ["librispeech_dev_clean", "test_ami", "manifests/med-dev-1", "manifests/tech-dev-2"]
        rec = self.sets_with(base + ["manifests/medical-b", "manifests/lectures-a"])
        self.assertTrue(rec["provisional"])
        self.assertEqual(rec["missing_required_parts"], ["stem-chem", "stem-math", "stem-cs"])
        rec = self.sets_with(base + [f"manifests/{p}" for p in hygiene.REQUIRED_HELDOUT_PARTS])
        self.assertFalse(rec["provisional"])
        self.assertEqual(len(rec["sets"]), len(base) + 5)


if __name__ == "__main__":
    unittest.main()
