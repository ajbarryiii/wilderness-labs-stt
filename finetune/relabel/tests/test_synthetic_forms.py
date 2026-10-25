"""Contract tests: label form normalization in text hygiene (synthetic plan "Text", hygiene: label forms).

- The label takes the conventional written form of listed terms (terms/synthetic/forms/<domain>.tsv): matched
  case-insensitively, on whole tokens, longest match first; Latin diacritics are folded to ASCII. Text outside the
  map is unchanged, and the normalization is idempotent.
- A form may change only letter case and a hyphen versus a space, so the scorer's normalized words never change;
  the held-out, medical and other hygiene rules run on the normalized label.
- The TTS input is unchanged: hygiene keeps the text before normalization as "spoken", and tts_text is the
  pronunciation layer's rendering of it.
- Substitution counts are recorded in the hygiene report.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_synthetic_*' -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from synthetic import forms, hygiene, layout, pronounce, tts, vocab
from synthetic_fixtures import heldout_sets, make_split, normalize, record

FILES = {
    "finance": "# comment\n\nebitda\tEBITDA\ngaap\tGAAP\nnon gaap\tnon-GAAP\nnon-gaap\tnon-GAAP\n"
               "high yield bond\thigh-yield bond\n",
    "software_engineering": "gpu\tGPU\ngpus\tGPUs\nmysql\tMySQL\nmoore's law\tMoore's law\n",
    "mathematics": "gaussian\tGaussian\nkaplan meier\tKaplan-Meier\n",
    "chemistry": "pseudo first order\tpseudo-first-order\n",
}


def fm(files=FILES) -> forms.Forms:
    return forms.Forms.parse(files)


class Apply(unittest.TestCase):
    def test_listed_forms_become_conventional_case_insensitively(self):
        f = fm()
        self.assertEqual(f.apply("Our ebitda and Ebitda and EBITDA rose.")[0],
                         "Our EBITDA and EBITDA and EBITDA rose.")
        self.assertEqual(f.apply("A gaussian fit and a Kaplan meier curve on the GPU.")[0],
                         "A Gaussian fit and a Kaplan-Meier curve on the GPU.")
        self.assertEqual(f.apply("as moore's law says")[0], "as Moore's law says")

    def test_whole_tokens_only(self):
        f = fm()
        unchanged = "The gaapish regaap rules, ebitdas, mysqldump, gpux, and Mooreslaw."
        self.assertEqual(f.apply(unchanged)[0], unchanged)
        self.assertEqual(f.apply("a gpu-based job, the gpu's fan, two gpus.")[0],
                         "a GPU-based job, the GPU's fan, two GPUs.")

    def test_longest_match_first(self):
        f = fm()
        self.assertEqual(f.apply("Under non gaap and non-gaap rules, not gaap.")[0],
                         "Under non-GAAP and non-GAAP rules, not GAAP.")
        self.assertEqual(f.apply("a pseudo first order reaction")[0], "a pseudo-first-order reaction")
        self.assertEqual(f.apply("a high yield bond")[0], "a high-yield bond")
        # a shorter form that is a prefix of a longer one at the same position: the longer wins
        g = fm(FILES | {"finance": FILES["finance"] + "gaap based\tGAAP-based\n"})
        self.assertEqual(g.apply("gaap based figures, gaap figures")[0], "GAAP-based figures, GAAP figures")
        # the substitutions are reported where they happened, longest first
        _, subs = f.apply("non gaap then gaap")
        self.assertEqual([(s["written"], s["form"]) for s in subs], [("non gaap", "non-GAAP"), ("gaap", "GAAP")])

    def test_text_outside_the_map_is_unchanged(self):
        f = fm()
        for text in ("Plain words only, with punctuation; and spacing!", "I think the p value is low?",
                     "The capex plan, the eps beat and the roe all changed.", ""):
            self.assertEqual(f.apply(text), (text, []))

    def test_idempotent(self):
        f = fm()
        for text in ("Our ebitda under non gaap and gaap; a gaussian on gpus, moore's law.",
                     "A Kaplan meier estimate, a pseudo first order step, a high yield bond."):
            once = f.apply(text)[0]
            self.assertNotEqual(once, text)
            self.assertEqual(f.apply(once), (once, []))
        committed = forms.Forms.load()
        for written, form in committed.pairs():
            self.assertEqual(committed.apply(form)[0], form)
            self.assertEqual(committed.apply(committed.apply(written)[0])[0], committed.apply(written)[0])

    def test_forms_never_change_the_scorers_words(self):
        n = normalize()
        for f in (fm(), forms.Forms.load()):
            for written, form in f.pairs():
                self.assertEqual(n(written), n(form), (written, form))
        text = "Our ebitda under non gaap, a gaussian and a Kaplan meier curve on gpus."
        self.assertEqual(n(fm().apply(text)[0]), n(text))

    def test_diacritics_fold_to_ascii(self):
        f = fm()
        self.assertEqual(f.apply("Schrödinger and L'Hôpital, naïve Gödel.")[0], "Schrodinger and L'Hopital, naive Godel.")

    def test_files_are_validated(self):
        for bad in ({"finance": "wifi\tWi-Fi\n"},               # changes the scorer's words
                    {"finance": "data set\tdataset\n"},
                    {"finance": "ebitda\tEBIT DA\n"},
                    {"finance": "schrodinger\tSchrödinger\n"},  # not ASCII
                    {"finance": "ebitda\n"},                     # no form
                    {"finance": "gaap\tGAAP\n", "mathematics": "GAAP\tGaap\n"}):   # one written form, two forms
            with self.assertRaises(ValueError, msg=bad):
                fm(bad)
        same = fm({"finance": "gaap\tGAAP\n", "mathematics": "GAAP\tGAAP\n"})
        self.assertEqual(same.apply("gaap")[0], "GAAP")
        self.assertTrue(forms.Forms.load().record()["files"])


HELDOUT_UTT = "so the next thing we want to look at is how the market reacted to the earnings call"


class InHygiene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = make_split()
        cls.n = staticmethod(normalize())
        cls.med = vocab.medical_matcher(normalize())
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sets = heldout_sets(Path(cls.tmp.name), {"dev-x": [HELDOUT_UTT]})
        cls.held = sorted(cls.split.heldout(), key=lambda e: e.term)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_h(self, records, f):
        return hygiene.run(records, self.split, self.med, self.n, lambda t: True, self.sets,
                           {"provisional": False, "sets": []}, forms=f)

    def test_labels_are_normalized_spoken_text_is_kept_and_counts_are_recorded(self):
        recs = [record(0, "finance", "Our ebitda rose under non gaap reporting this quarter.", ["EBITDA"]),
                record(1, "finance", "The working capital was strong again this quarter.", ["working capital"]),
                record(2, "finance", "Our gaap ebitda and the ebitda margin both grew.", ["EBITDA"])]
        kept, report, rows = self.run_h(recs, fm())
        by = {k["id"]: k for k in kept}
        self.assertEqual(by["smoke-000000"]["label"], "Our EBITDA rose under non-GAAP reporting this quarter.")
        self.assertEqual(by["smoke-000000"]["spoken"], "Our ebitda rose under non gaap reporting this quarter.")
        self.assertEqual(by["smoke-000001"]["label"], by["smoke-000001"]["spoken"])
        self.assertEqual(report["forms"]["sentences_changed"], 2)
        self.assertEqual(report["forms"]["kept_sentences_changed"], 2)
        self.assertEqual(report["forms"]["substitutions"], {"ebitda -> EBITDA": 3, "gaap -> GAAP": 1,
                                                            "non gaap -> non-GAAP": 1})
        self.assertEqual(report["forms"]["files"], fm().record()["files"])
        for r in rows:
            self.assertEqual(r["norm"], " ".join(self.n(r["label"]).split()))

    def test_heldout_and_exclusion_filters_run_on_the_normalized_label(self):
        # Forms never change the normalized words (validated at load), so the word rules give the same result on the
        # label and on the spoken text; the character rule (form_symbols) is the one that differs (diacritics test).
        h = self.held[0]
        written = h.term.lower()
        files = dict(FILES) | {"chemistry": FILES["chemistry"] + f"{written}\t{h.term.upper()}\n"}
        tgt = sorted(self.split.targeted("finance"), key=lambda e: e.term)[0].term
        recs = [record(0, "finance", f"We discussed the {tgt} and the {written} in detail today.", [tgt]),
                record(1, "finance", f"We discussed the {tgt} and ketamine synthesis today.", [tgt]),
                record(2, "finance", f"We discussed the {tgt} in ebitda terms today.", [tgt])]
        kept, report, rows = self.run_h(recs, fm(files))
        reasons = {r["id"]: r["reason"] for r in rows}
        self.assertEqual(reasons["smoke-000000"], "heldout_term")
        self.assertEqual(rows[0]["label"], f"We discussed the {tgt} and the {h.term.upper()} in detail today.")
        self.assertEqual(reasons["smoke-000001"], "medical")
        self.assertEqual([k["label"] for k in kept], [f"We discussed the {tgt} in EBITDA terms today."])
        self.assertEqual(report["forms"]["kept_sentences_changed"], 1)

    def test_diacritics_are_folded_instead_of_rejected(self):
        tgt = sorted(self.split.targeted("chemistry"), key=lambda e: e.term)[0].term
        recs = [record(0, "chemistry", f"Schrödinger would have liked the {tgt} in this lecture.", [tgt])]
        kept, report, _ = self.run_h(recs, fm())
        self.assertEqual(kept[0]["label"], f"Schrodinger would have liked the {tgt} in this lecture.")
        self.assertEqual(report["forms"]["folded_sentences"], 1)

    def test_the_tts_input_is_unchanged_by_label_forms(self):
        lex = pronounce.Lexicon.load()
        raw = ["Our ebitda rose under non gaap reporting this quarter.",
               "A gaussian fit and a Kaplan meier curve ran on two gpus.",
               "As moore's law slows, a pseudo first order model and a high yield bond appear."]
        recs = [record(i, layout.DOMAINS[i % 4], t, []) for i, t in enumerate(raw)]
        with_forms = hygiene.normalized_rows(recs, fm())
        without = hygiene.normalized_rows(recs, forms.Forms.parse({}))
        self.assertNotEqual([r["label"] for r in with_forms], [r["label"] for r in without])
        voices = {"syn-tts-a": ["p225"], "syn-tts-b": ["p1"]}
        plan_f, _ = pronounce.apply(tts.assign(with_forms, voices, seed=3), lex,
                                    spoken={r["id"]: r["spoken"] for r in with_forms})
        plan_0, _ = pronounce.apply(tts.assign(without, voices, seed=3), lex)
        self.assertEqual([r["tts_text"] for r in plan_f], [r["tts_text"] for r in plan_0])
        self.assertEqual([r["label"] for r in plan_f], [r["label"] for r in with_forms])


if __name__ == "__main__":
    unittest.main()
