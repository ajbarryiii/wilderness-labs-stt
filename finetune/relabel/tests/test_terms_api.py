"""Term-list contracts (labeling plan Part A critical-term recall, Part B selection): the stable API
terms.load / categories / Matcher / matcher, the committed files' format, the categories the plan names,
and the matching rule (normalized, greedy longest match, left to right, non-overlapping).

Run from finetune/relabel:  ./python -m unittest discover -s tests -p "test_terms_*" -v
"""
from __future__ import annotations

import unittest

import evaluate
import terms

# Everyday English words (and words with common non-domain senses) that must not be terms: as critical terms
# they measure nothing, and as Part B selectors they pull in general speech.
EVERYDAY = set("""
the of and to a in is it you that he was for on are with as i his they be at one have this from or had by
not word but what some we can out other were all there when up use your how said an each she which do their
time if will way about many then them write would like so these her long make thing see him two has look
more day could go come did number sound no most people my over know water than call first who may down side
been now find any new work part take get place made live where after back little only round man year came
show every good me give our under name very through just form sentence great think say help low line differ
turn cause much mean before move right boy old too same tell does set three want air well also play small end
put home read hand port large spell add even land here must big high such follow act why ask men change went
light kind off need house picture try us again animal point mother world near build self earth father head
stand own page should country found answer school grow study still learn plant cover food sun four between
state keep eye never last let thought city tree cross farm hard start might story saw far sea draw left late
run while press close night real life few north open seem together next white children begin got walk example
ease paper group always music those both mark often letter until mile river car feet care second book carry
took science eat room friend began idea fish mountain stop once base hear horse cut sure watch color face wood
main enough plain girl usual young ready above ever red list though feel talk bird soon body dog family direct
pose leave song measure door product black short numeral class wind question happen complete ship area half
rock order fire south problem piece told knew pass since top whole king space heard best hour better true
during hundred five remember step early hold west ground interest reach fast verb sing listen six table travel
less morning ten simple several vowel toward war lay against pattern slow center love person money serve
appear road map rain rule govern pull cold notice voice unit power town fine certain fly fall lead cry dark
machine note wait plan figure star box noun field rest correct able pound done beauty drive stood contain
front teach week final gave green oh quick develop ocean warm free minute strong special mind behind clear
tail produce fact street inch multiply nothing course stay wheel full force blue object decide surface deep
moon island foot system busy test record boat common gold possible plane stead dry wonder laugh thousand ago
ran check game shape equate hot miss brought heat snow tire bring yes distant fill east paint language among
march arm chest heart blood skin bone leg pain shock plasma calcium priority routine data model network cloud
market options bond guidance outlook equity colon period patch push drip stat tablet units grid pump optics
resistance thrust protein spectrum atmosphere evolution integral momentum friction excel react swift rust
deployment inference accuracy parameters baseline query storage chip container exploit tokens fatigue strain
benign burn rash depression litter blast pupils hoist routine expectant circulation orbit radius atrium
""".split())


class FilesTest(unittest.TestCase):
    def test_every_category_file_is_plain_one_term_per_line(self) -> None:
        for kind in terms.KINDS:
            for name in terms.categories(kind):
                path = terms.HERE / kind / f"{name}.txt"
                found = terms.read_file(path)
                self.assertGreater(len(found), 20, path)
                for term in found:
                    self.assertEqual(term, term.strip())
                    for bad in (",", ";", "\t", "|"):
                        self.assertNotIn(bad, term, f"{path}: {term!r} looks like several terms on one line")

    def test_plan_categories_exist(self) -> None:
        self.assertEqual(terms.categories("medical"),
                         ["drugs", "anatomy", "procedures", "conditions", "dosages_units", "tccc"])
        self.assertEqual(terms.categories("technical"), ["computing", "engineering", "science", "finance"])
        medical = set(terms.load("medical"))
        for term in ("tranexamic acid", "ketamine", "femur", "cricothyroidotomy", "milligrams", "tourniquet",
                     "combat gauze", "needle decompression", "tension pneumothorax"):
            self.assertIn(term, medical)
        technical = set(terms.load("technical"))
        for term in ("kubernetes", "torque", "photosynthesis", "ebitda"):
            self.assertIn(term, technical)


class LoadTest(unittest.TestCase):
    def test_load_is_deduplicated_and_ordered(self) -> None:
        for kind in terms.KINDS:
            loaded = terms.load(kind)
            self.assertEqual(len({t.casefold() for t in loaded}), len(loaded))
            expected = []
            for name in terms.categories(kind):
                expected += [t for t in terms.read_file(terms.HERE / kind / f"{name}.txt")
                             if t.casefold() not in {e.casefold() for e in expected}]
            self.assertEqual(loaded, expected)

    def test_only_some_categories(self) -> None:
        drugs = terms.load("medical", only=["drugs"])
        self.assertIn("ketamine", drugs)
        self.assertNotIn("femur", drugs)

    def test_unknown_kind_and_category_fail(self) -> None:
        with self.assertRaises(ValueError):
            terms.load("legal")
        with self.assertRaises(ValueError):
            terms.load("medical", only=["finance"])


class MatcherTest(unittest.TestCase):
    def test_greedy_longest_left_to_right_non_overlapping(self) -> None:
        m = terms.Matcher(["blood pressure", "pressure", "high blood", "x"], normalize=str.lower)
        found = m.find("high blood pressure pressure".split())
        self.assertEqual([(o.start, o.end) for o in found], [(0, 2), (2, 3), (3, 4)])
        found = m.find("low blood pressure".split())
        self.assertEqual([(o.start, o.end, o.phrase) for o in found], [(1, 3, ("blood", "pressure"))])
        self.assertEqual(m.count("pressure pressure".split()), 2)

    def test_terms_that_normalize_to_nothing_or_one_character_are_dropped(self) -> None:
        m = terms.Matcher(["um", "C", "aorta"], normalize=evaluate.normalize)
        self.assertEqual(sorted(m.dropped), ["C", "um"])
        self.assertEqual(m.find("c um".split()), [])

    def test_every_term_matches_its_own_normalized_form(self) -> None:
        for kind in terms.KINDS:
            m = terms.matcher(kind)
            for term in terms.load(kind):
                if term in m.dropped:
                    continue
                tokens = evaluate.normalize(term).split()
                found = m.find(tokens)
                self.assertEqual([(o.start, o.end) for o in found], [(0, len(tokens))], term)

    def test_normalizer_variants_match(self) -> None:
        m = terms.matcher("medical")
        for text in ("severe haemorrhage", "the oesophagus", "five hundred milligrams of TXA"):
            self.assertTrue(m.find(evaluate.normalize(text).split()), text)

    def test_no_term_is_an_everyday_word(self) -> None:
        for kind in terms.KINDS:
            m = terms.matcher(kind)
            singles = {p[0] for p in m.phrases if len(p) == 1}
            self.assertEqual(sorted(singles & EVERYDAY), [], kind)


if __name__ == "__main__":
    unittest.main()
