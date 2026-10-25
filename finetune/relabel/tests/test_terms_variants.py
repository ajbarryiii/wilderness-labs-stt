"""Contract tests: spelling variants of a listed term (labeling plan Part A "Critical-term recall", amended
2026-10-24; synthetic plan "Terms").

Rule: listed terms whose normalized words, joined without spaces, are equal ("dataset" / "data set", "e-commerce" /
"ecommerce") are variants of ONE term. Only listed forms are variants; nothing is split or joined automatically.
- An occurrence is any listed form in the reference, found greedily (longest first, non-overlapping), counted once.
- It is reproduced if the hypothesis has the same form under the old rule (every word matched, nothing inserted
  between them), or any listed form of the same term at the aligned position.
- The synthetic split puts a term's variants in one group (same side, same domain); the held-out filter rejects
  every variant of a held-out term.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_terms_variants.py' -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import evaluate
import score_domain as sd
import terms
from synthetic import hygiene, layout, vocab
from synthetic_fixtures import heldout_sets, normalize, record

LIST = ["dataset", "data set", "datasets", "data sets", "data", "e-commerce", "ecommerce", "rate constant",
        "backend", "back end", "blood pressure"]


def occ(ref: str, hyp: str, lst=LIST):
    m = terms.Matcher(lst, evaluate.normalize)
    return sd.term_occurrences(evaluate.normalize(ref), evaluate.normalize(hyp), m)


class Scorer(unittest.TestCase):
    def test_variants_score_as_one_term(self):
        for ref, hyp in (("we load the data set today", "we load the dataset today"),
                         ("we load the dataset today", "we load the data set today"),
                         ("our ecommerce sales", "our e-commerce sales"),
                         ("our e-commerce sales", "our e commerce sales"),
                         ("the back end is slow", "the backend is slow"),
                         ("two data sets here", "two datasets here")):
            found = occ(ref, hyp)
            self.assertEqual(len(found), 1, (ref, hyp))
            self.assertTrue(found[0][2], (ref, hyp))
        m = terms.Matcher(LIST, evaluate.normalize)
        self.assertEqual(set(m.variants(("data", "set"))), {("dataset",), ("data", "set")})
        self.assertEqual(m.variants(("rate", "constant")), (("rate", "constant"),))

    def test_no_double_counting(self):
        # one spoken occurrence is one occurrence, whichever form, and whatever the hypothesis says
        for ref in ("the data set", "the dataset", "the data set and the dataset"):
            for hyp in ("the dataset", "the data set", "the data set dataset data set"):
                found = occ(ref, hyp)
                self.assertEqual(len(found), ref.count("data"), (ref, hyp))
        # "data set" is one occurrence, not "data" plus something
        self.assertEqual([(s, e) for s, e, _ in occ("the data set is", "x")], [(1, 3)])
        res = sd.term_recall([{"ref_norm": "the data set", "hyp_norm": "the dataset"},
                              {"ref_norm": "a dataset", "hyp_norm": "a data set"}], terms.Matcher(LIST, evaluate.normalize))
        self.assertEqual((res["occurrences"], res["reproduced"]), (2, 2))

    def test_an_unlisted_joined_or_split_form_does_not_match(self):
        self.assertEqual(occ("the rateconstant doubled", "the rate constant doubled"), [])
        self.assertEqual(occ("high bloodpressure", "high blood pressure"), [])
        only_joined = ["dataset"]
        self.assertEqual(occ("the data set", "the dataset", only_joined), [])
        found = occ("the dataset", "the data set", only_joined)          # the split form is not listed
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0][2])
        self.assertEqual(occ("i on the line", "ion", ["ion"]), [])     # no joining of ordinary words

    def test_a_variant_counts_only_at_the_aligned_position(self):
        found = occ("we load the dataset today", "we load the files today and a data set")
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0][2])
        found = occ("we load the dataset today", "we load the data big set today")
        self.assertFalse(found[0][2])

    def test_the_old_rule_is_unchanged_without_variants(self):
        self.assertTrue(occ("the rate constant doubled", "a rate constant doubled")[0][2])
        self.assertFalse(occ("the rate constant doubled", "the rate big constant doubled")[0][2])
        self.assertFalse(occ("the rate constant doubled", "the great constant doubled")[0][2])

    def test_committed_lists_variant_groups_are_well_formed(self):
        for kind in terms.KINDS:
            m = terms.matcher(kind)
            for key, group in m.groups().items():
                self.assertGreater(len(group), 1)
                self.assertTrue(all("".join(p) == key for p in group))


SPLIT_LISTS = {"chemistry": ["enthalpy", "catalyst"],
               "mathematics": ["data set", "eigenvalue", "determinant"],
               "finance": ["e-commerce", "yield curve", "working capital"],
               "software_engineering": ["dataset", "ecommerce", "backend", "back end", "hash table", "deadlock"]}


def split(share=layout.TARGETED_SHARE, lists=SPLIT_LISTS) -> vocab.Split:
    n = normalize()
    return vocab.Split(vocab.build_split(lists, n, vocab.medical_matcher(n), share=share), n)


class SplitGroups(unittest.TestCase):
    def test_variants_share_group_side_and_domain(self):
        for share in (0.0, 0.3, 0.7, 1.0):
            s = split(share)
            by = s.by_phrase
            for a, b in ((("data", "set"), ("dataset",)), (("e", "commerce"), ("ecommerce",)),
                         (("backend",), ("back", "end"))):
                self.assertEqual((by[a].group, by[a].side, by[a].domain), (by[b].group, by[b].side, by[b].domain))
            # the domain is the first domain (layout.DOMAINS order) that lists any variant
            self.assertEqual(by[("dataset",)].domain, "mathematics")
            self.assertEqual(by[("ecommerce",)].domain, "finance")

    def test_the_heldout_filter_rejects_every_variant_of_a_heldout_term(self):
        s = split(0.0)                     # everything held out
        for text in ("we used the data set", "we used the dataset", "our e-commerce arm", "our ecommerce arm",
                     "our e commerce arm"):
            self.assertTrue(s.has_heldout(text), text)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        sets = heldout_sets(Path(tmp.name), {"dev-x": ["unrelated words only here"]})
        recs = [record(i, "software_engineering", t, ["deadlock"]) for i, t in
                enumerate(["The deadlock hit the data set again today.", "The deadlock hit the dataset again today.",
                           "The deadlock hit our ecommerce site today.", "The deadlock hit our e-commerce site today."])]
        _, _, rows = hygiene.run(recs, s, vocab.medical_matcher(normalize()), normalize(), lambda t: True, sets,
                                 {"provisional": False, "sets": []})
        self.assertTrue(all(r["flags"]["heldout_term"] for r in rows))
        recs = [record(9, "software_engineering", "We merged both data-sets again today.", []),
                record(10, "software_engineering", "We merged both tables again today.", [])]
        _, _, rows = hygiene.run(recs, s, vocab.medical_matcher(normalize()), normalize(), lambda t: True, sets,
                                 {"provisional": False, "sets": []})
        self.assertEqual([r["flags"]["heldout_term"] for r in rows], [True, False])   # an unlisted plural form

    def test_adding_a_variant_without_new_nesting_links_moves_no_group(self):
        base = {d: list(v) for d, v in SPLIT_LISTS.items()}
        more = {d: list(v) for d, v in SPLIT_LISTS.items()}
        base["software_engineering"] += ["data warehouse"]
        more["software_engineering"] += ["hashtable", "dead lock", "data-set", "data warehouse", "datawarehouse"]
        more["chemistry"] += ["cata lyst"]
        for share in (0.3, 0.5, 0.7):
            a, b = split(share, base), split(share, more)
            for e in a.entries:
                f = b.by_phrase[e.phrase]
                self.assertEqual((e.side, e.domain), (f.side, f.domain), (share, e.phrase))
            for new, old in ((("hashtable",), ("hash", "table")), (("dead", "lock"), ("deadlock",)),
                             (("cata", "lyst"), ("catalyst",)), (("datawarehouse",), ("data", "warehouse"))):
                self.assertEqual(b.by_phrase[new].side, a.by_phrase[old].side)

    def test_the_leak_guard_catches_unlisted_surface_forms_of_heldout_terms(self):
        s = split(0.0)                     # everything held out; the guard is not the scorer
        for text in ("two datasets", "the data-set", "our data sets", "a dataset", "the determinant is",
                     "both determinants", "the hash-table", "two hash tables", "a hashtable", "back-ends",
                     "an e-commerce site", "the e commerces", "dead locks"):
            words = normalize()(text).split()
            self.assertTrue(s.heldout_guard(words), text)
        for text in ("the data was set aside", "a hash of the table", "back to the end", "commerce in europe"):
            self.assertFalse(s.heldout_guard(normalize()(text).split()), text)
        self.assertEqual(split(1.0).heldout_guard(normalize()("two datasets").split()), False)
        singular_only = split(0.0, SPLIT_LISTS | {"chemistry": ["noble gas", "matrix", "data warehouse"]})
        for text in ("the noble gases", "a noble-gas", "two matrices", "the datawarehouses", "a data warehouse",
                     "the data-warehouse"):
            self.assertTrue(singular_only.has_heldout(text), text)
        self.assertFalse(singular_only.has_heldout("a noble cause and a gas leak"))
        long_term = split(0.0, SPLIT_LISTS | {"chemistry": ["catalyst", "standard enthalpy of formation change"]})
        self.assertTrue(long_term.heldout_guard(normalize()("the standard enthalpy of formation changes").split()))
        self.assertTrue(long_term.has_heldout("the standard-enthalpy of formation change"))

    def test_recall_attributes_every_variant_to_the_terms_domain_and_side(self):
        s = split(1.0)
        n = normalize()
        recs = [{"ref_norm": n("we load the data set"), "hyp_norm": n("we load the dataset")},
                {"ref_norm": n("we load the dataset"), "hyp_norm": n("we load the data set")}]
        r = vocab.recall(recs, s)
        self.assertEqual(r["mathematics"]["targeted"], {"occurrences": 2, "reproduced": 2, "recall": 1.0})
        self.assertEqual(r["software_engineering"]["targeted"]["occurrences"], 0)


class Provenance(unittest.TestCase):
    """A clip pool is usable for training only if its hygiene ran under the current split; reuse-text copies raw
    text only, so hygiene reruns under the current split, lists and forms."""

    def test_reuse_text_copies_raw_text_only(self):
        import synthetic.__main__ as cli
        from synthetic.textgen import read_jsonl, write_jsonl
        from synthetic_fixtures import roots
        with roots():
            src = layout.run_dir("pilot")
            write_jsonl(src / "prompts.jsonl", [{"id": "pilot-000000", "domain": "finance", "requested": []}])
            write_jsonl(src / "text" / "0000000.jsonl", [{"id": "pilot-000000", "raw": "A sentence."}])
            write_jsonl(src / "sentences.jsonl", [{"id": "pilot-000000", "label": "A sentence."}])
            (src / "hygiene.json").write_text("{}")
            saved = layout.require_mount
            layout.require_mount = lambda: None
            try:
                self.assertEqual(cli.main(["reuse-text", "--from", "pilot", "--run", "pilot2"]), 0)
            finally:
                layout.require_mount = saved
            dst = layout.run_dir("pilot2")
            self.assertEqual(read_jsonl(dst / "text" / "0000000.jsonl"), read_jsonl(src / "text" / "0000000.jsonl"))
            self.assertTrue((dst / "prompts.jsonl").exists())
            self.assertFalse((dst / "sentences.jsonl").exists())
            self.assertFalse((dst / "hygiene.json").exists())

    def test_a_pool_whose_hygiene_did_not_record_the_current_split_is_refused(self):
        import hashlib
        import json
        from synthetic import mixing, roundtrip
        from test_synthetic_sources import Finalize
        from synthetic_fixtures import make_split, roots
        sp = make_split()
        with roots() as t:
            d, _ = Finalize().world(t, sp)
            hyg = json.loads((d / "hygiene.json").read_text())
            hyg["heldout"]["provisional"] = False
            (d / "hygiene.json").write_text(json.dumps(hyg))                  # no split recorded (legacy)
            roundtrip.finalize(d, sp, listen=False)
            self.assertTrue(json.loads((d / "clips.meta.json").read_text())["split_stale"])
            with self.assertRaises(RuntimeError):
                mixing.ClipPool.load(d)
            hyg["split_sha256"] = "0" * 64                                    # another split
            (d / "hygiene.json").write_text(json.dumps(hyg))
            roundtrip.finalize(d, sp, listen=False)
            with self.assertRaises(RuntimeError):
                mixing.ClipPool.load(d)
            hyg["split_sha256"] = hashlib.sha256(layout.split_path().read_bytes()).hexdigest()
            (d / "hygiene.json").write_text(json.dumps(hyg))
            roundtrip.finalize(d, sp, listen=False)
            self.assertFalse(json.loads((d / "clips.meta.json").read_text())["split_stale"])
            self.assertTrue(mixing.ClipPool.load(d).clips)


class Rescore(unittest.TestCase):
    """term_rescore.py: old = the old lists and rule (must equal a stored term_recall), new = the current rule."""

    def test_old_reproduces_stored_values_and_new_counts_variants(self):
        import json
        import term_rescore as tr
        n = evaluate.normalize
        recs = [{"ref_norm": n("we load the data set"), "hyp_norm": n("we load the dataset"), "scored": True},
                {"ref_norm": n("the dataset grew"), "hyp_norm": n("the dataset grew"), "scored": True}]
        with tempfile.TemporaryDirectory() as tmp:
            old_lists = Path(tmp) / "old"
            for kind, files in {"medical": ["drugs"], "technical": ["computing"]}.items():
                (old_lists / kind).mkdir(parents=True)
                for f in terms.categories(kind):
                    (old_lists / kind / f"{f}.txt").write_text("dataset\n" if f == "computing" else "")
            old_m = terms.Matcher(terms.load("technical", base=old_lists), n)
            old = tr.recall(recs, old_m, variants=False)
            self.assertEqual((old["occurrences"], old["reproduced"]), (1, 1))     # "data set" was not listed
            new = tr.recall(recs, terms.Matcher(["dataset", "data set"], n), variants=True)
            self.assertEqual((new["occurrences"], new["reproduced"]), (2, 2))
            self.assertEqual(tr.set_entry({"a": new, "b": old})["pooled"], {"occurrences": 3, "reproduced": 3,
                                                                                "recall": 1.0})


if __name__ == "__main__":
    unittest.main()
