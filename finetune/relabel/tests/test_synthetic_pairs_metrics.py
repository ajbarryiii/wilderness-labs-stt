"""Contract tests: the confusable-pair metrics (synthetic plan v5, "Metrics").

- the stable held-out set is exactly v2's held-out groups minus the moved ones;
- partner substitution and near forms behave as defined: "Schroeder equation" at a "Schrodinger equation" occurrence
  is a partner substitution, "EBIT" (or "adjusted EBIT") at an EBITDA occurrence is one, and "EBITA" at an EBITDA
  occurrence is a near form;
- pair recall is reported with and without the dev-informed seed;
- re-scoring an earlier model under v2's groups reproduces its recorded term recall exactly.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_synthetic_*' -v
"""
from __future__ import annotations

import unittest

from synthetic import layout, pair_metrics, vocab
from pairs_fixtures import group_of, make_v3
from synthetic_fixtures import normalize


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v2, cls.v3 = make_v3({("roe", "roi"): ("reject", "different ratios")})
        cls.n = staticmethod(normalize())

    def occ(self, ref, hyp):
        return pair_metrics.occurrences(self.n(ref), self.n(hyp), self.v3)

    def pair_id(self, a, b):
        return next(p["id"] for p in self.v3.data["pairs"] if set(p["terms"]) == {a, b})


class StableHeldout(Fixture):
    def test_the_stable_set_is_v2s_heldout_groups_minus_the_moved_ones(self):
        moved = {group_of(self.v2, p) for p in ("ebitda", "schrodinger equation", "opex")}
        self.assertEqual(pair_metrics.moved_groups(self.v2, self.v3), moved)
        stable = pair_metrics.stable_heldout_groups(self.v2, self.v3)
        self.assertEqual(stable, {e.group for e in self.v2.heldout()} - moved)
        self.assertEqual(stable, {e.group for e in self.v3.heldout()})
        self.assertIn(group_of(self.v2, "roe"), stable)


class Substitution(Fixture):
    def one(self, ref, hyp, phrase):
        occ = [o for o in self.occ(ref, hyp) if o["group"] == group_of(self.v3, phrase)]
        self.assertEqual(len(occ), 1, (ref, hyp))
        return occ[0]

    def test_schroeder_equation_at_a_schrodinger_equation_occurrence_is_a_partner_substitution(self):
        pid = self.pair_id("Schrodinger equation", "Cantor-Bernstein-Schroeder theorem")
        o = self.one("so the Schrödinger equation is linear", "so the Schroeder equation is linear",
                     "schrodinger equation")
        self.assertFalse(o["ok"])
        self.assertEqual(o["substituted"], {pid: True})
        self.assertFalse(o["near"])
        o = self.one("so the Schrödinger equation is linear", "so the Schrodinger equation is linear",
                     "schrodinger equation")
        self.assertTrue(o["ok"])
        self.assertEqual(o["substituted"], {pid: False})

    def test_ebit_at_an_ebitda_occurrence_is_a_partner_substitution(self):
        pid = self.pair_id("ebitda", "ebit")
        for hyp in ("we expect EBIT of ten million", "we expect adjusted EBIT of ten million"):
            o = self.one("we expect EBITDA of ten million", hyp, "ebitda")
            self.assertEqual(o["substituted"], {pid: True}, hyp)
            self.assertFalse(o["near"], hyp)
        o = self.one("we expect EBIT of ten million", "we expect EBITDA of ten million", "ebit")
        self.assertEqual(o["substituted"], {pid: True})                  # the other member, the other way

    def test_ebita_at_an_ebitda_occurrence_is_a_near_form(self):
        pid = self.pair_id("ebitda", "ebit")
        o = self.one("we expect EBITDA of ten million", "we expect EBITA of ten million", "ebitda")
        self.assertEqual(o["substituted"], {pid: False})
        self.assertTrue(o["near"])
        for hyp, near in (("we expect EBITDA of ten million", False),       # the listed form: reproduced
                          ("we expect editor of ten million", False),       # distance 3 > max(2, ceil(1.8))
                          ("we expect EBTA of ten million", True),
                          ("we expect of ten million", False)):
            o = self.one("we expect EBITDA of ten million", hyp, "ebitda")
            self.assertEqual(o["near"], near, hyp)
            self.assertEqual(o["substituted"], {pid: False}, hyp)

    def test_a_term_outside_every_pair_has_no_substitution(self):
        o = self.one("the yield curve inverted", "the yield curves inverted", "yield curve")
        self.assertEqual(o["substituted"], {})
        self.assertFalse(o["near"])


def records(rows, part):
    n = normalize()
    return [{"id": f"{part}:rec{k % 3}-{k:04d}", "ref_norm": n(r), "hyp_norm": n(h), "scored": True}
            for k, (r, h) in enumerate(rows)]


REFS = ["we expect EBITDA of ten million and the opex fell", "the roe of the bank rose while the alkane burned",
        "the Schrodinger equation and the Cantor-Bernstein-Schroeder theorem", "the gpu and the tpu were busy",
        "adjusted EBITDA and the free cash flow", "the ebit and the capex moved with the yield curve"]


class Analysis(Fixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        hyp_a = ["we expect EBITA of ten million and the op ex fell", "the roe of the bank rose while the alkene burned",
                 "the Schroeder equation and the Cantor-Bernstein-Schroeder theorem", "the gpu and the tpu were busy",
                 "adjusted EBIT and the free cash flow", "the ebit and the capex moved with the yield curve"]
        cls.parts = {"tech-dev-2": {"A": records(zip(REFS, hyp_a), "tech-dev-2"),
                                    "B": records(zip(REFS, REFS), "tech-dev-2")},
                     "stem-chem": {"A": records(zip(REFS[2:4], hyp_a[2:4]), "stem-chem"),
                                   "B": records(zip(REFS[2:4], REFS[2:4]), "stem-chem")}}
        cls.res = pair_metrics.analyze(cls.parts, cls.v2, cls.v3, {"A-B": ("A", "B")}, replicates=200)

    def test_rescoring_under_v2s_groups_reproduces_the_recorded_recall_exactly(self):
        for m in ("A", "B"):
            recorded = vocab.recall([r for p in self.parts.values() for r in p[m]], self.v2)
            for d in layout.DOMAINS:
                for side in vocab.SIDES:
                    g = self.res["groups"][f"v2/{d}/{side}"]
                    self.assertEqual((g["occurrences"], g["reproduced"][m]),
                                     (recorded[d][side]["occurrences"], recorded[d][side]["reproduced"]), (m, d, side))
            for side in vocab.SIDES:
                g = self.res["groups"][f"v2/{side}"]
                self.assertEqual((g["occurrences"], g["reproduced"][m]),
                                 (recorded["pooled"][side]["occurrences"], recorded["pooled"][side]["reproduced"]))

    def test_pair_recall_with_and_without_the_seed(self):
        with_seed, without = self.res["groups"]["pair-recall"], self.res["groups"]["pair-recall-no-seed"]
        self.assertEqual(with_seed["occurrences"], without["occurrences"] + 1)          # one opex occurrence
        self.assertEqual(with_seed["reproduced"]["B"], with_seed["occurrences"])
        self.assertEqual(without["reproduced"]["A"], 0)                                  # EBITA, adjusted EBIT, Schroeder
        self.assertIn("contrasts", with_seed)
        self.assertEqual(self.res["dev_informed_groups"], [group_of(self.v3, "opex")])

    def test_stable_heldout_and_targeted_recall_under_v3(self):
        st = self.res["groups"]["stable-heldout"]
        self.assertEqual(st["occurrences"], 2)                                          # roe, alkane
        self.assertEqual(st["reproduced"]["A"], 1)
        self.assertIn("targeted-v3", self.res["groups"])

    def test_partner_substitution_and_near_forms_per_pair_and_pooled(self):
        g = self.res["groups"]
        pid = self.pair_id("ebitda", "ebit")
        sub = g[f"partner-substitution/{pid}"]
        self.assertEqual(sub["reproduced"]["A"], 1)                                      # adjusted EBIT
        self.assertEqual(sub["reproduced"]["B"], 0)
        self.assertEqual(g[f"near-forms/{pid}"]["reproduced"]["A"], 1)                    # EBITA
        hh = self.pair_id("alkane", "alkene")
        self.assertEqual(g[f"partner-substitution/{hh}"]["reproduced"]["A"], 1)          # held-out pairs too
        self.assertEqual(g["partner-substitution"]["reproduced"]["A"], 4)                # + Schroeder twice, alkene
        self.assertEqual(g["partner-substitution"]["reproduced"]["B"], 0)


if __name__ == "__main__":
    unittest.main()
