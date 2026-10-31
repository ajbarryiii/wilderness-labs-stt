"""Contract tests: generation of run `pairs` (synthetic plan v5, "Generation (run pairs)").

- prompts request only terms targeted under v3, and contrast prompts request both members; a round issues 2.5 times
  the remaining need (40 contrast clips per trained pair; T(g) single-term clips per group);
- sides are read after the moves (plan v5.1): a held-out-held-out candidate whose two members other accepted pairs
  move gets contrast prompts; one with a member still held out gets none;
- v3's guard rejects every remaining held-out term in v1's covered forms;
- the medical and TCCC exclusion and the denylist still apply; duplicates are checked against `full`;
- balance: counts are per group over `full` and `pairs`, and a contrast clip counts for both members; no single-term
  clip is kept for a group already at or above T(g); existing and contrast clips are never dropped to meet T(g), so
  EBIT keeps 214 plus its contrast clips; shortfalls are reported.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_synthetic_*' -v
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from synthetic import layout, pairs, textgen, vocab
from synthetic.textgen import read_jsonl, write_jsonl
from pairs_fixtures import LISTS, clip_counts, group_of, make_v3, sha256
from synthetic_fixtures import heldout_sets, normalize, record, roots

HELDOUT_UTT = ("so the next thing we want to look at is how the market reacted to the earnings call "
               "on the first day of trading")
REJECT_ROE = {("roe", "roi"): ("reject", "different ratios")}


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v2, cls.v3 = make_v3(REJECT_ROE)
        cls.full = clip_counts(cls.v2)
        cls.spec = pairs.Spec(cls.v3, cls.full)
        cls.g = staticmethod(lambda phrase: group_of(cls.v3, phrase))

    def pair_id(self, a, b):
        return next(p["id"] for p in self.spec.trained if set(p["terms"]) == {a, b})


class Targets(Fixture):
    def test_targets_are_the_partners_full_clips_between_the_median_and_the_90th_percentile(self):
        t = self.spec.targets
        self.assertEqual((pairs.FLOOR, pairs.CAP), (139, 264))
        self.assertEqual(t[self.g("ebitda")], 214)              # EBIT's 214
        self.assertEqual(t[self.g("ebit")], 139)                # EBITDA has none: the floor
        self.assertEqual(t[self.g("schrodinger equation")], 183)
        self.assertEqual(t[self.g("tpu")], 264)                 # GPU's 300: the cap
        self.assertEqual(t[self.g("opex")], 264)                # the seed: CAPEX's 372, capped
        self.assertNotIn(self.g("roe"), t)                      # a rejected pair
        self.assertNotIn(self.g("alkane"), t)                   # held out under v3


class Prompts(Fixture):
    def test_prompts_request_only_v3_targeted_terms_and_contrast_prompts_both_members(self):
        state = pairs.State(self.spec, self.v3)
        ps = pairs.round_prompts(self.spec, self.v3, state, 1)
        self.assertEqual(ps, pairs.round_prompts(self.spec, self.v3, pairs.State(self.spec, self.v3), 1))
        by_term = {e.term: e for e in self.v3.entries}
        contrast = Counter()
        single = Counter()
        for p in ps:
            es = [by_term[t] for t in p["requested"]]
            self.assertTrue(all(e.side == "targeted" for e in es), p)
            self.assertEqual(len({e.group for e in es}), len(es), p)
            self.assertIn(p["register"], layout.REGISTERS)
            if p["kind"] == "contrast":
                pair = next(x for x in self.spec.trained if x["id"] == p["pair"])
                self.assertEqual(p["requested"], pair["terms"])
                self.assertIn("both", textgen.prompt_text(p))
                contrast[p["pair"]] += 1
            else:
                self.assertEqual(p["kind"], "single")
                self.assertEqual(es[0].group, p["target"])
                self.assertTrue(all(e.domain == es[0].domain for e in es), p)
                self.assertTrue(1 <= len(es) <= 3)
                single[p["target"]] += 1
        self.assertEqual(set(contrast), {p["id"] for p in self.spec.trained})
        self.assertEqual(set(contrast.values()), {math.ceil(2.5 * 40)})
        self.assertEqual(single, {self.g("ebitda"): math.ceil(2.5 * 214), self.g("schrodinger equation"):
                                  math.ceil(2.5 * 183), self.g("tpu"): math.ceil(2.5 * (264 - 120)),
                                  self.g("opex"): math.ceil(2.5 * 264)})
        requested = {t for p in ps for t in p["requested"]}
        self.assertFalse(requested & {"roe", "alkane", "alkene", "laws of thermodynamics"})
        self.assertEqual(len({p["id"] for p in ps}), len(ps))

    def test_a_later_round_issues_prompts_for_the_remaining_need_only(self):
        ebit = self.pair_id("ebitda", "ebit")
        kept = [{"id": f"r1-{k}", "label": "The EBITDA and the EBIT differ by depreciation.", "kind": "contrast",
                 "pair": ebit, "target": None} for k in range(30)]
        kept += [{"id": f"r1-s{k}", "label": "Adjusted EBITDA rose again.", "kind": "single", "pair": None,
                  "target": self.g("ebitda")} for k in range(100)]
        state = pairs.State(self.spec, self.v3, kept)
        ps = pairs.round_prompts(self.spec, self.v3, state, 2)
        self.assertEqual(sum(p["kind"] == "contrast" and p["pair"] == ebit for p in ps), math.ceil(2.5 * 10))
        self.assertEqual(sum(p.get("target") == self.g("ebitda") for p in ps), math.ceil(2.5 * (214 - 130)))
        self.assertTrue(all(p["id"].startswith("pairs-r2-") for p in ps))
        with self.assertRaises(ValueError):
            pairs.round_prompts(self.spec, self.v3, state, pairs.ROUNDS + 1)


class SidesAfterMoves(unittest.TestCase):
    def test_a_heldout_pair_whose_members_both_move_gets_contrast_prompts(self):
        # "alkyne" (targeted) pairs with both "alkane" and "alkene" (held out): both move, so the held-out-held-out
        # candidate alkane/alkene is a targeted-targeted pair under v3
        lists = LISTS | {"chemistry": LISTS["chemistry"] + ["alkyne"]}
        v2, v3 = make_v3({("roe", "roi"): ("reject", "different ratios")}, lists=lists)
        pair = next(p for p in v3.data["pairs"] if set(p["terms"]) == {"alkane", "alkene"})
        self.assertEqual(pair["sides_v2"], ["heldout", "heldout"])
        self.assertEqual((pair["side_type_v3"], pair["trained"]), ("targeted-targeted", True))
        spec = pairs.Spec(v3, clip_counts(v2))
        ps = pairs.round_prompts(spec, v3, pairs.State(spec, v3), 1)
        contrast = [p for p in ps if p["kind"] == "contrast" and p["pair"] == pair["id"]]
        self.assertEqual(len(contrast), math.ceil(2.5 * 40))
        self.assertTrue(all(p["requested"] == pair["terms"] for p in contrast))
        for g in pair["groups"]:
            self.assertIn(g, spec.targets)

    def test_a_pair_with_a_member_still_heldout_gets_nothing(self):
        # only "alkene" moves (through "alkyne", while alkane/alkyne is rejected): alkane/alkene is not trained
        lists = LISTS | {"chemistry": LISTS["chemistry"] + ["alkyne"]}
        v2, v3 = make_v3({("roe", "roi"): ("reject", "different ratios"),
                          ("alkane", "alkyne"): ("reject", "fixture")}, lists=lists)
        pair = next(p for p in v3.data["pairs"] if set(p["terms"]) == {"alkane", "alkene"})
        self.assertEqual((pair["side_type_v3"], pair["trained"]), ("heldout-targeted", False))
        spec = pairs.Spec(v3, clip_counts(v2))
        ps = pairs.round_prompts(spec, v3, pairs.State(spec, v3), 1)
        self.assertFalse([p for p in ps if p.get("pair") == pair["id"]])
        self.assertFalse([p for p in ps if "alkane" in p["requested"]])


class Guard(Fixture):
    def test_v3s_guard_rejects_every_remaining_heldout_term_in_v1s_covered_forms(self):
        held = self.v3.heldout()
        self.assertTrue(held)
        for e in held:
            joined = "".join(e.phrase)
            forms = [" ".join(e.phrase), joined, "-".join(e.phrase), " ".join(e.phrase) + "s", joined + "es"]
            for f in forms:
                self.assertTrue(self.v3.has_heldout(f"We looked at the {f} yesterday."), (e.term, f))
        for phrase in ("ebitda", "adjusted ebitda", "schrodinger equation", "opex"):      # moved: allowed now
            self.assertFalse(self.v3.has_heldout(f"We looked at the {phrase} yesterday."), phrase)
            self.assertTrue(self.v2.has_heldout(f"We looked at the {phrase} yesterday."), phrase)


class Hygiene(Fixture):
    def run_h(self, records, prior=()):
        with tempfile.TemporaryDirectory() as tmp:
            sets = heldout_sets(Path(tmp), {"dev-x": [HELDOUT_UTT]})
            n = normalize()
            return pairs.hygiene(records, self.v3, vocab.medical_matcher(n), n, lambda t: True, sets,
                                 {"provisional": False, "sets": []}, prior=prior)

    def test_medical_exclusion_denylist_and_heldout_guard_still_apply(self):
        recs = [record(0, "finance", "The EBITDA matters when a tourniquet is applied in the field.", ["ebitda"]),
                record(1, "finance", f"Honestly {HELDOUT_UTT} and the EBITDA moved too.", ["ebitda"]),
                record(2, "finance", "The roe of the bank and its EBITDA fell this year.", ["ebitda"]),
                record(3, "finance", "The EBITDA of the firm rose sharply this year.", ["ebitda"])]
        kept, report, rows = self.run_h(recs)
        self.assertEqual([r["reason"] for r in rows], ["medical", "denylist", "heldout_term", None])
        self.assertEqual(report["split_version"], 3)

    def test_duplicates_are_checked_against_full(self):
        prior = ["we talked about the ebit today", "the capex budget doubled after the merger was approved"]
        recs = [record(0, "finance", "We talked about the EBIT today.", ["ebit"]),
                record(1, "finance", "We talked about the EBIT today again.", ["ebit"]),
                record(2, "finance", "The opex and the capex budget both doubled after the merger closed in "
                                     "spring.", ["opex", "capex"]),
                record(3, "finance", "Nobody expected the EBIT to fall that much in one quarter.", ["ebit"])]
        kept, report, rows = self.run_h(recs, prior=prior)
        self.assertEqual([r["reason"] for r in rows], ["duplicate", "near_duplicate", None, None])
        self.assertEqual(report["prior_sentences"], 2)

    def test_the_prior_holds_fulls_kept_sentences_and_earlier_rounds(self):
        with roots():
            write_jsonl(layout.run_dir("full") / "sentences.jsonl",
                        [{"id": "full-1", "norm": "we talked about the ebit today"}])
            r1 = layout.run_dir("pairs-r1")
            write_jsonl(r1 / "sentences.jsonl", [{"id": "pairs-r1-000000", "norm": "the opex rose"}])
            self.assertEqual(pairs.prior_sentences(1), ["we talked about the ebit today"])
            self.assertEqual(pairs.prior_sentences(2), ["we talked about the ebit today", "the opex rose"])


def sentence(i, label, prompt):
    return {"id": f"pairs-r1-{i:06d}", "domain": "finance", "label": label, "requested": prompt["requested"]}


class Balance(Fixture):
    def world(self):
        ebit = self.pair_id("ebitda", "ebit")
        prompts, sents = {}, []

        def add(label, kind, pair=None, target=None, requested=("ebitda",)):
            p = {"id": f"pairs-r1-{len(sents):06d}", "kind": kind, "pair": pair, "target": target,
                 "requested": list(requested)}
            prompts[p["id"]] = p
            sents.append(sentence(len(sents), label, p))
        for k in range(50):
            add(f"The EBITDA and the EBIT parted ways in quarter {k}.", "contrast", ebit, requested=("ebitda", "ebit"))
        for k in range(5):
            add("Only the EBIT was discussed.", "contrast", ebit, requested=("ebitda", "ebit"))
        for k in range(200):
            add(f"Adjusted EBITDA rose again in region {k}.", "single", target=self.g("ebitda"))
        for k in range(3):
            add("The EBIT fell.", "single", target=self.g("ebit"), requested=("ebit",))
        for k in range(10):
            add("Opex and capex rose together.", "single", target=self.g("opex"), requested=("opex", "capex"))
        return prompts, sents, ebit

    def test_counts_are_per_group_over_full_and_pairs_and_contrast_counts_for_both(self):
        state = pairs.State(self.spec, self.v3)
        self.assertEqual(state.counts[self.g("ebit")], 214)
        clip = {"id": "x", "label": "The EBITDA and the EBIT differ.", "kind": "contrast",
                "pair": self.pair_id("ebitda", "ebit"), "target": None}
        state = pairs.State(self.spec, self.v3, [clip])
        self.assertEqual(state.counts[self.g("ebit")], 215)
        self.assertEqual(state.counts[self.g("ebitda")], 1)
        self.assertEqual(state.contrast[clip["pair"]], 1)
        single = {"id": "y", "label": "The opex and the capex rose.", "kind": "single", "pair": None,
                  "target": self.g("opex")}
        state = pairs.State(self.spec, self.v3, [single])
        self.assertEqual((state.counts[self.g("opex")], state.counts[self.g("capex")]), (1, 373))

    def test_no_single_term_clip_is_kept_at_or_above_the_target(self):
        prompts, sents, ebit = self.world()
        state = pairs.State(self.spec, self.v3)
        selected, decisions = pairs.select(sents, prompts, self.spec, self.v3, state)
        self.assertEqual(selected, pairs.select(sents, prompts, self.spec, self.v3, pairs.State(self.spec, self.v3))[0])
        kinds = Counter((prompts[s["id"]]["kind"], prompts[s["id"]]["target"]) for s in selected)
        self.assertEqual(kinds[("contrast", None)], 40)
        self.assertEqual(kinds[("single", self.g("ebitda"))], 214 - 40)
        self.assertEqual(kinds[("single", self.g("ebit"))], 0)
        self.assertEqual(kinds[("single", self.g("opex"))], 10)
        reasons = Counter(d["decision"] for d in decisions.values())
        self.assertEqual(reasons["contrast_full"], 10)
        self.assertEqual(reasons["missing_member"], 5)
        self.assertEqual(reasons["over_target"], 200 - 174 + 3)
        # walking the selection in its order, every single-term clip was below its target when kept
        counts = Counter(pairs.State(self.spec, self.v3).counts)
        for s in selected:
            p = prompts[s["id"]]
            if p["kind"] == "single":
                self.assertLess(counts[p["target"]], self.spec.targets[p["target"]])
            for g in pairs.groups_in(s["label"], self.v3):
                counts[g] += 1

    def test_existing_and_contrast_clips_are_never_dropped(self):
        prompts, sents, ebit = self.world()
        state = pairs.State(self.spec, self.v3)
        selected, _ = pairs.select(sents, prompts, self.spec, self.v3, state)
        kept = [s | {"kind": prompts[s["id"]]["kind"], "pair": prompts[s["id"]]["pair"],
                     "target": prompts[s["id"]]["target"]} for s in selected]
        after = pairs.State(self.spec, self.v3, kept)
        self.assertEqual(after.counts[self.g("ebit")], 214 + 40)          # above its target of 139
        self.assertEqual(after.counts[self.g("ebitda")], 214)
        self.assertEqual(after.contrast[ebit], 40)

    def test_shortfalls_are_reported(self):
        state = pairs.State(self.spec, self.v3)
        short = pairs.shortfalls(self.spec, state)
        self.assertEqual(short["groups"][self.g("ebitda")], {"target": 214, "count": 0, "shortfall": 214,
                                                             "dev_informed": False})
        self.assertTrue(short["groups"][self.g("opex")]["dev_informed"])
        self.assertEqual(short["groups"][self.g("ebit")]["shortfall"], 0)
        self.assertEqual(short["pairs"][self.pair_id("gpu", "tpu")], {"target": 40, "count": 0, "shortfall": 40})


class FinalizeRun(Fixture):
    def test_the_run_joins_its_rounds_and_reports_shortfalls(self):
        with roots():
            v3_path = layout.split_version_path(3)
            v3_path.parent.mkdir(parents=True, exist_ok=True)
            v3_path.write_text(json.dumps(self.v3.data) + "\n")
            ebit = self.pair_id("ebitda", "ebit")
            for k, n in ((1, 30), (2, 5)):
                d = layout.run_dir(f"pairs-r{k}")
                clips = [{"id": f"pairs-r{k}-{i:06d}", "label": "The EBITDA and the EBIT differ.",
                          "audio_filepath": "/dev/null", "duration": 2.0, "sha256": "0", "domain": "finance"}
                         for i in range(n)]
                write_jsonl(d / "clips.jsonl", clips)
                write_jsonl(d / "prompts.jsonl", [{"id": c["id"], "kind": "contrast", "pair": ebit, "target": None,
                                                   "requested": ["ebitda", "ebit"]} for c in clips])
                (d / "clips.meta.json").write_text(json.dumps({
                    "clips_sha256": sha256(d / "clips.jsonl"), "provisional": False,
                    "hygiene_split_sha256": sha256(v3_path), "split_sha256": sha256(v3_path)}))
            rep = pairs.finalize_run(self.spec, self.v3)
            out = read_jsonl(layout.run_dir("pairs") / "clips.jsonl")
            self.assertEqual(len(out), 35)
            self.assertEqual({c["kind"] for c in out}, {"contrast"})
            self.assertEqual(rep["pairs"][ebit]["shortfall"], 5)
            self.assertEqual(rep["groups"][self.g("ebitda")]["shortfall"], 214 - 35)
            meta = json.loads((layout.run_dir("pairs") / "clips.meta.json").read_text())
            self.assertEqual(meta["hygiene_split_sha256"], sha256(v3_path))
            self.assertFalse(meta["provisional"])
            from synthetic import mixing
            self.assertEqual(len(mixing.ClipPool.load("pairs", split_path=v3_path).clips), 35)


if __name__ == "__main__":
    unittest.main()
