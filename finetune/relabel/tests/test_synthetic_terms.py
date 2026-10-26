"""Contract tests: stage 2c term lists and the frozen targeted / held-out split (synthetic plan "Terms").

- terms are normalized as the scorer normalizes references, and de-duplicated across domains;
- terms matching the medical / TCCC lists are excluded;
- variants (plural forms, nested phrases) form one group and are always on the same side;
- the split is a fixed hash of the group (70% targeted), reproducible, frozen, and refuses changed lists;
- term recall keeps score_domain's aligned-occurrence definition, overlapping phrases counted as it counts them;
- the committed lists obey the same rules.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_synthetic_*' -v
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import score_domain
import terms
from synthetic import layout, vocab
from synthetic_fixtures import LISTS, make_split, normalize, roots


class SplitRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = make_split()
        cls.by_term = {e.term: e for e in cls.split.entries}

    def test_terms_are_normalized_as_the_scorer_normalizes(self):
        n = normalize()
        for e in self.split.entries:
            self.assertEqual(e.phrase, tuple(n(e.term).split()))
        self.assertEqual(self.by_term["Bayes' theorem"].phrase, tuple(n("Bayes' theorem").split()))

    def test_deduplicated_across_domains_first_domain_wins(self):
        phrases = [e.phrase for e in self.split.entries]
        self.assertEqual(len(phrases), len(set(phrases)))
        self.assertEqual(self.by_term["stoichiometry"].domain, "chemistry")
        self.assertEqual([d["domain"] for d in self.split.data["duplicates"]], ["mathematics"])

    def test_medical_and_tccc_terms_are_excluded(self):
        excluded = {m["term"] for m in self.split.data["medical_excluded"]}
        self.assertEqual(excluded, {"ketamine synthesis", "tourniquet chemistry", "dependency injection"})
        med = vocab.medical_matcher(normalize())
        for e in self.split.entries:
            self.assertEqual(med.find(list(e.phrase)), [], e.term)

    def test_variants_share_a_group_and_a_side(self):
        groups = [("rate constant", "rate constants"), ("catalyst", "catalysts"), ("eigenvalue", "eigenvalues"),
                  ("Gibbs free energy", "free energy"), ("hash table", "hash tables")]
        for a, b in groups:
            self.assertEqual(self.by_term[a].group, self.by_term[b].group, (a, b))
            self.assertEqual(self.by_term[a].side, self.by_term[b].side, (a, b))
        self.assertNotEqual(self.by_term["enthalpy"].group, self.by_term["catalyst"].group)

    def test_acronym_plurals_share_a_group(self):
        lists = {"chemistry": [], "mathematics": [], "finance": ["etfs", "etf"],
                 "software_engineering": ["api", "apis", "rest apis", "gpu", "gpus", "status", "statu quo"]}
        s = make_split(lists)
        g = {e.term: e.group for e in s.entries}
        self.assertEqual(g["api"], g["apis"])
        self.assertEqual(g["apis"], g["rest apis"])
        self.assertEqual(g["gpu"], g["gpus"])
        self.assertEqual(g["etf"], g["etfs"])
        self.assertNotEqual(g["status"], g["gpu"])

    def test_nested_grouping_is_transitive_and_crosses_domains(self):
        lists = {"chemistry": ["energy density", "free energy"], "mathematics": ["free energy density"],
                 "finance": ["derivative"], "software_engineering": ["derivatives"]}
        s = make_split(lists)
        g = {e.term: e.group for e in s.entries}
        self.assertEqual(g["energy density"], g["free energy"])
        self.assertEqual(g["free energy"], g["free energy density"])
        self.assertEqual(g["derivative"], g["derivatives"])

    def test_side_is_the_fixed_hash_of_the_group(self):
        for e in self.split.entries:
            digest = hashlib.sha256(f"{layout.SPLIT_SALT}\n{e.group}".encode()).digest()
            expected = "targeted" if int.from_bytes(digest[:8], "big") < 0.70 * 2 ** 64 else "heldout"
            self.assertEqual(e.side, expected)

    def test_split_is_about_70_30_by_group(self):
        keys = [f"group {i}" for i in range(20000)]
        share = sum(vocab.side_of(k) == "targeted" for k in keys) / len(keys)
        self.assertAlmostEqual(share, 0.70, delta=0.015)
        self.assertNotEqual(vocab.side_of("x", salt="a"), None)

    def test_split_is_reproducible(self):
        again = make_split()
        self.assertEqual(again.data, self.split.data)

    def test_heldout_matcher_holds_only_heldout_terms(self):
        held = {e.phrase for e in self.split.heldout()}
        self.assertEqual(set(self.split.heldout_matcher().phrases), held)
        targeted = {e.phrase for d in layout.DOMAINS for e in self.split.targeted(d)}
        self.assertFalse(held & targeted)


class Freeze(unittest.TestCase):
    def test_freeze_writes_once_and_refuses_changed_lists(self):
        with roots() as t:
            d = t / "lists"
            d.mkdir()
            for dom, ts in LISTS.items():
                (d / f"{dom}.txt").write_text("\n".join(ts) + "\n")
            out = t / "split.json"
            first = vocab.freeze(out, d)
            self.assertEqual(vocab.freeze(out, d), first)                    # same inputs: unchanged
            loaded = vocab.Split.load(out, lists_dir=d)
            self.assertEqual(len(loaded.entries), len(first["entries"]))
            (d / "finance.txt").write_text("\n".join(LISTS["finance"] + ["amortization"]) + "\n")
            with self.assertRaises(RuntimeError):
                vocab.freeze(out, d)
            with self.assertRaises(RuntimeError):
                vocab.Split.load(out, lists_dir=d)
            # a plan decision replaces it: the old split is archived and the changes are recorded
            new = vocab.freeze(out, d, replace_reason="test")
            self.assertEqual(json.loads((t / "split.v2.json").read_text()), first)
            ch = json.loads((t / "split-changes.json").read_text())
            self.assertEqual([a["phrase"] for a in ch["added"]], ["amortization"])
            self.assertEqual(ch["reason"], "test")
            self.assertEqual(vocab.Split.load(out, lists_dir=d).data, new)


class Recall(unittest.TestCase):
    def test_recall_uses_the_aligned_occurrence_definition_per_domain_and_side(self):
        s = make_split()
        n = normalize()
        ref = "the gibbs free energy and the rate constant set the enthalpy of the yield curve"
        hyp = "the gibbs free energy and the rate of constant set the enthalpy of the yield"
        rec = [{"ref_norm": n(ref), "hyp_norm": n(hyp), "scored": True},
               {"ref_norm": n("enthalpy"), "hyp_norm": "", "scored": False}]
        out = vocab.recall(rec, s)
        found = score_domain.term_occurrences(n(ref), n(hyp), s.matcher())
        # Greedy longest match: "gibbs free energy" is one occurrence ("free energy" is not counted again).
        phrases = [" ".join(n(ref).split()[a:b]) for a, b, _ in found]
        self.assertEqual(phrases, ["gibbs free energy", "rate constant", "enthalpy", "yield curve"])
        total = sum(out[d][side]["occurrences"] for d in layout.DOMAINS for side in vocab.SIDES)
        self.assertEqual(total, 4)
        self.assertEqual([ok for _, _, ok in found], [True, False, True, False])   # insertion; deletion
        expected = {}
        for a, b, ok in found:
            e = s.by_phrase[tuple(n(ref).split()[a:b])]
            occ, rep = expected.get((e.domain, e.side), (0, 0))
            expected[(e.domain, e.side)] = (occ + 1, rep + int(ok))
        for d in layout.DOMAINS:
            for side in vocab.SIDES:
                occ, rep = expected.get((d, side), (0, 0))
                self.assertEqual((out[d][side]["occurrences"], out[d][side]["reproduced"]), (occ, rep), (d, side))
        self.assertEqual(out["pooled"]["targeted"]["reproduced"] + out["pooled"]["heldout"]["reproduced"], 2)
        self.assertEqual(out["pooled"]["targeted"]["occurrences"] + out["pooled"]["heldout"]["occurrences"], 4)


class Prompts(unittest.TestCase):
    def test_prompts_request_only_targeted_terms_balanced_and_seeded(self):
        from collections import Counter
        from synthetic import textgen
        s = make_split()
        ps = textgen.make_prompts(s, 400, seed=1, run="t")
        self.assertEqual(ps, textgen.make_prompts(s, 400, seed=1, run="t"))
        self.assertNotEqual(ps, textgen.make_prompts(s, 400, seed=2, run="t"))
        by_term = {e.term: e for e in s.entries}
        self.assertEqual(Counter(p["domain"] for p in ps), {d: 100 for d in layout.DOMAINS})
        for p in ps:
            self.assertTrue(1 <= len(p["requested"]) <= 3)
            es = [by_term[t] for t in p["requested"]]
            self.assertTrue(all(e.side == "targeted" and e.domain == p["domain"] for e in es), p)
            self.assertEqual(len({e.group for e in es}), len(es))
        for d in layout.DOMAINS:
            regs = Counter(p["register"] for p in ps if p["domain"] == d)
            self.assertLessEqual(max(regs.values()) - min(regs.values()), 1)
            used = Counter(t for p in ps if p["domain"] == d for t in p["requested"])
            self.assertEqual(set(used), {e.term for e in s.targeted(d)})
        self.assertIn('"' + ps[0]["requested"][0] + '"', textgen.prompt_text(ps[0]))


class CommittedLists(unittest.TestCase):
    def test_committed_lists_exist_for_every_domain_and_are_clean(self):
        n = normalize()
        med = vocab.medical_matcher(n)
        lists = vocab.read_lists()
        self.assertEqual(set(lists), set(layout.DOMAINS))
        seen = {}
        for dom, ts in lists.items():
            self.assertGreater(len(ts), 100, dom)
            for t in ts:
                self.assertRegex(t, r"^[A-Za-z][A-Za-z' -]*[A-Za-z]$")
                ph = tuple(n(t).split())
                self.assertNotIn(ph, seen, f"{t} in {dom} and {seen.get(ph)}")
                seen[ph] = dom
        s = vocab.Split(vocab.build_split(lists, n, med), n)
        self.assertEqual(s.data["duplicates"], [])
        for e in s.entries:
            self.assertEqual(med.find(list(e.phrase)), [])
        for d in layout.DOMAINS:
            self.assertTrue(s.targeted(d), d)


if __name__ == "__main__":
    unittest.main()
