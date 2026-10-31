"""Contract tests: the contrast top-up, run `pairs-c` (synthetic plan v5.2, "Contrast top-up, run pairs-c").

- a contrast prompt holds both members of its pair plus 0-2 other targeted terms of the pair's domain (both members'
  domains for a cross-domain pair), drawn seeded per prompt; never a held-out term; it still asks for both members as
  distinct things;
- the text-generation seed differs by run and round (and is unchanged for the runs made before v5.2);
- the target counts kept contrast clips over `pairs` and `pairs-c`: 40 per trained pair;
- duplicates are checked against `full` and `pairs` as well as within the run;
- `pairs-c` issues no single-term prompt.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_synthetic_*' -v
"""
from __future__ import annotations

import hashlib
import json
import math
import unittest
from collections import Counter

from synthetic import layout, pairs, textgen
from synthetic.textgen import read_jsonl, write_jsonl
from pairs_fixtures import clip_counts, group_of, make_v3, sha256
from synthetic_fixtures import roots

RUN = "pairs-c"


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v2, cls.v3 = make_v3({("roe", "roi"): ("reject", "different ratios")})
        cls.spec = pairs.Spec(cls.v3, clip_counts(cls.v2))
        cls.by_term = {e.term: e for e in cls.v3.entries}

    def pair(self, a, b):
        return next(p for p in self.spec.trained if set(p["terms"]) == {a, b})

    def prompts(self, kept=(), round=1):
        return pairs.round_prompts(self.spec, self.v3, pairs.State(self.spec, self.v3, kept), round, run=RUN)


class Prompts(Fixture):
    def test_contrast_prompts_hold_both_members_and_up_to_two_targeted_extras_of_the_pairs_domains(self):
        ps = self.prompts()
        self.assertEqual(ps, self.prompts())
        self.assertTrue(ps)
        extras = Counter()
        for p in ps:
            pair = self.spec.by_id[p["pair"]]
            self.assertEqual(p["requested"][:2], pair["terms"])
            es = [self.by_term[t] for t in p["requested"]]
            self.assertTrue(all(e.side == "targeted" for e in es), p)
            self.assertLessEqual(len(es), 4)
            self.assertEqual(len({e.group for e in es}), len(es), p)
            domains = set(pair["domains"])
            self.assertTrue(all(e.domain in domains for e in es[2:]), p)
            extras[len(es) - 2] += 1
            text = textgen.prompt_text(p)
            self.assertIn("as two different things", text)
            for t in p["requested"]:
                self.assertIn(f'"{t}"', text)
            self.assertTrue(p["id"].startswith(f"{RUN}-r1-"))
        self.assertEqual(set(extras), {0, 1, 2})                     # varied, drawn per prompt
        per_pair = Counter((p["pair"], tuple(p["requested"]), p["register"]) for p in ps)
        distinct = Counter(k[0] for k in per_pair)
        self.assertTrue(all(n > 3 for n in distinct.values()), distinct)   # more than terms x three registers

    def test_a_cross_domain_pair_draws_extras_from_both_domains(self):
        pair = self.pair("Schrodinger equation", "Cantor-Bernstein-Schroeder theorem")
        ps = [p for p in self.prompts() if p["pair"] == pair["id"]]
        seen = {self.by_term[t].domain for p in ps for t in p["requested"][2:]}
        self.assertEqual(seen, {"chemistry", "mathematics"})
        self.assertIn("the fields of chemistry and mathematics", textgen.prompt_text(ps[0]))

    def test_no_heldout_term_is_ever_requested(self):
        held = {e.term for e in self.v3.heldout()}
        self.assertTrue(held)
        for p in self.prompts():
            self.assertFalse(set(p["requested"]) & held, p)

    def test_pairs_c_issues_no_single_term_prompt(self):
        ps = self.prompts()
        self.assertEqual({p["kind"] for p in ps}, {"contrast"})
        self.assertTrue(any(n for n in pairs.shortfalls(self.spec, pairs.State(self.spec, self.v3))["groups"].values()
                            if n["shortfall"]))                     # groups below T(g) get no single-term prompt here

    def test_run_pairs_prompts_are_unchanged(self):
        ps = pairs.round_prompts(self.spec, self.v3, pairs.State(self.spec, self.v3), 1)
        self.assertEqual(ps, pairs.round_prompts(self.spec, self.v3, pairs.State(self.spec, self.v3), 1, run="pairs"))
        for p in ps:
            if p["kind"] == "contrast":
                self.assertEqual(p["requested"], self.spec.by_id[p["pair"]]["terms"])
        self.assertIn("single", {p["kind"] for p in ps})


class Target(Fixture):
    def clips(self, pair, n, prefix="pairs-r1"):
        a, b = pair["terms"]
        return [{"id": f"{prefix}-{k:06d}", "label": f"The {a} and the {b} differ.", "kind": "contrast",
                 "pair": pair["id"], "target": None} for k in range(n)]

    def test_the_target_counts_contrast_clips_over_pairs_and_pairs_c(self):
        ebit, gpu = self.pair("ebitda", "ebit"), self.pair("gpu", "tpu")
        kept = self.clips(ebit, 29) + self.clips(gpu, 40, "pairs-r2") + self.clips(ebit, 5, "pairs-c-r1")
        ps = self.prompts(kept, round=2)
        n = Counter(p["pair"] for p in ps)
        self.assertEqual(n[ebit["id"]], math.ceil(2.5 * (40 - 29 - 5)))
        self.assertNotIn(gpu["id"], n)
        for p in self.spec.trained:
            if p["id"] not in (ebit["id"], gpu["id"]):
                self.assertEqual(n[p["id"]], math.ceil(2.5 * 40))

    def test_selection_keeps_contrast_sentences_while_the_pair_is_below_40_over_both_runs(self):
        ebit = self.pair("ebitda", "ebit")
        state = pairs.State(self.spec, self.v3, self.clips(ebit, 34))
        prompts, sents = {}, []
        for k in range(10):
            pid = f"{RUN}-r1-{k:06d}"
            prompts[pid] = {"id": pid, "kind": "contrast", "pair": ebit["id"], "target": None,
                            "requested": ["ebitda", "ebit", "yield curve"]}
            sents.append({"id": pid, "label": f"The EBITDA and the EBIT moved with the yield curve in case {k}.",
                          "domain": "finance"})
        selected, decisions = pairs.select(sents, prompts, self.spec, self.v3, state)
        self.assertEqual(len(selected), 6)
        self.assertEqual(Counter(d["decision"] for d in decisions.values())["contrast_full"], 4)

    def test_state_before_a_pairs_c_round_reads_run_pairs_and_earlier_pairs_c_rounds(self):
        ebit = self.pair("ebitda", "ebit")
        with roots():
            v3_path = layout.split_version_path(3)
            v3_path.parent.mkdir(parents=True, exist_ok=True)
            v3_path.write_text(json.dumps(self.v3.data) + "\n")
            joined = layout.run_dir("pairs")
            write_jsonl(joined / "clips.jsonl", self.clips(ebit, 20))
            (joined / "clips.meta.json").write_text(json.dumps({"clips_sha256": sha256(joined / "clips.jsonl"),
                                                                "hygiene_split_sha256": sha256(v3_path),
                                                                "provisional": False}))
            self.assertEqual(pairs.state_before(self.spec, self.v3, 1, RUN).contrast[ebit["id"]], 20)
            d = pairs.round_dir(1, RUN)
            clips = self.clips(ebit, 7, "pairs-c-r1")
            write_jsonl(d / "clips.jsonl", clips)
            write_jsonl(d / "prompts.jsonl", [{"id": c["id"], "kind": "contrast", "pair": ebit["id"],
                                               "target": None} for c in clips])
            (d / "clips.meta.json").write_text(json.dumps({"clips_sha256": sha256(d / "clips.jsonl"),
                                                           "hygiene_split_sha256": sha256(v3_path)}))
            state = pairs.state_before(self.spec, self.v3, 2, RUN)
            self.assertEqual(state.contrast[ebit["id"]], 27)
            self.assertEqual(state.counts[group_of(self.v3, "ebit")], 214 + 27)


class Seed(unittest.TestCase):
    class Gen:
        batch = 4

        def __init__(self):
            self.seeds = []

        def load(self):
            pass

        def generate(self, users, seed):
            self.seeds.append(seed)
            return [("A sentence.", False) for _ in users]

        def info(self):
            return {"alias": "fake", "revision": "0", "batch": self.batch}

        def peak(self):
            return 0.0, 0.0

    def seeds(self, d, stream):
        write_jsonl(d / "prompts.jsonl", [{"id": f"x-{i}", "index": i, "domain": "finance", "register": "explanatory",
                                           "requested": ["ebit"]} for i in range(8)])
        g = self.Gen()
        textgen.generate_chunk(d, g, layout.SEED, 10, log=lambda m: None, stream=stream)
        self.assertEqual(sorted({r["seed"] for r in read_jsonl(d / "text" / "0000000.jsonl")}), sorted(set(g.seeds)))
        return g.seeds

    def test_the_seed_differs_by_run_and_round(self):
        with roots() as t:
            a = self.seeds(t / "a", pairs.text_stream("pairs-c-r1"))
            b = self.seeds(t / "b", pairs.text_stream("pairs-c-r2"))
            c = self.seeds(t / "c", pairs.text_stream("pairs-c-r1"))
            self.assertEqual(len(a), 2)
            self.assertFalse(set(a) & set(b))
            self.assertEqual(a, c)

    def test_runs_made_before_v52_keep_their_seeds(self):
        for run in ("full", "pilot", "pairs-r1", "pairs-r2"):
            self.assertIsNone(pairs.text_stream(run), run)
        self.assertEqual(pairs.text_stream("pairs-c-r3"), "pairs-c-r3")
        with roots() as t:
            got = self.seeds(t / "v1", None)
            want = [int.from_bytes(hashlib.sha256(f"{layout.SEED}|0|{b}".encode()).digest()[:4], "big") for b in (0, 4)]
            self.assertEqual(got, want)


class Duplicates(Fixture):
    def test_duplicates_are_checked_against_full_pairs_and_earlier_pairs_c_rounds(self):
        with roots():
            write_jsonl(layout.run_dir("full") / "sentences.jsonl", [{"id": "f", "norm": "full sentence"}])
            for k in (1, 2, 3):
                write_jsonl(pairs.round_dir(k) / "sentences.jsonl", [{"id": f"p{k}", "norm": f"pairs sentence {k}"}])
            write_jsonl(pairs.round_dir(1, RUN) / "sentences.jsonl", [{"id": "c1", "norm": "contrast sentence 1"}])
            want = ["full sentence", "pairs sentence 1", "pairs sentence 2", "pairs sentence 3"]
            self.assertEqual(pairs.prior_sentences(1, RUN), want)
            self.assertEqual(pairs.prior_sentences(2, RUN), want + ["contrast sentence 1"])
            self.assertEqual(pairs.prior_sentences(2), ["full sentence", "pairs sentence 1"])   # run pairs: as before

    def test_a_sentence_of_run_pairs_is_a_duplicate_in_pairs_c(self):
        import tempfile
        from pathlib import Path
        from synthetic import vocab
        from synthetic_fixtures import heldout_sets, normalize, record
        n = normalize()
        prior = [" ".join(n("The EBITDA and the EBIT differ by depreciation and amortization.").split())]
        recs = [record(0, "finance", "The EBITDA and the EBIT differ by depreciation and amortization.",
                       ["ebitda", "ebit"]),
                record(1, "finance", "Analysts compared the EBITDA with the EBIT before the merger was announced.",
                       ["ebitda", "ebit"])]
        with tempfile.TemporaryDirectory() as tmp:
            sets = heldout_sets(Path(tmp), {"dev-x": ["one two three four five six seven eight nine"]})
            kept, report, rows = pairs.hygiene(recs, self.v3, vocab.medical_matcher(n), n, lambda t: True, sets,
                                               {"provisional": False, "sets": []}, prior=prior)
        self.assertEqual([r["reason"] for r in rows], ["duplicate", None])


if __name__ == "__main__":
    unittest.main()
