"""Contract tests: confusable term pairs (synthetic plan v5, "Confusable pairs"): the detector, the curation file
terms/synthetic/pairs.tsv, split v3, and earlier clip pools under split v3.

- the detector is deterministic and reads only its listed inputs: an evaluation or hypothesis path is refused;
- fixtures that pair: EBIT/EBITDA (prefix), Schroeder/Schrodinger (shared prefix after folding), ROE/ROI and GPU/TPU
  (on their letter names); fixtures that don't: GPU/Gauss, thermodynamic/thermodynamics (an inflection), a token
  with 300 or more label occurrences, two tokens of the same group;
- the curation file is checked against the candidate file it was curated from, and the always-accepted pairs
  (EBITDA/EBIT, Schrodinger equation/Cantor-Bernstein-Schroeder theorem) must be accepted;
- split v3: only curated moves and the seed (OPEX) change side, and groups move whole; held-out(v3) is a subset of
  held-out(v2), and split.v2.json is unchanged; every move is listed with its reason;
- provenance: full's clips pass v3's guard, and a pool with a failing clip is refused.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_synthetic_*' -v
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from synthetic import confusable, layout, mixing, pairs, vocab
from synthetic.textgen import write_jsonl
from pairs_fixtures import (clip_counts, curation_text, find, group_of, lexicon, make_split, sha256, write_world)
from synthetic_fixtures import roots


def detect(split=None, words=None, clips=None):
    split = split or make_split()
    return confusable.detect(split, lexicon(), words or {}, clips if clips is not None else clip_counts(split))


class DetectorPairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split = make_split()
        cls.out = detect(cls.split)

    def pair(self, a, b):
        c = find(self.out, a, b)
        self.assertIsNotNone(c, f"{a} / {b} should be a candidate")
        return c

    def test_ebit_and_ebitda_pair_by_prefix(self):
        c = self.pair("ebitda", "ebit")
        self.assertEqual(c["rule"], "prefix")
        self.assertEqual(c["side_type"], "heldout-targeted")
        self.assertIn(["ebitda", "ebit"], [sorted(h["tokens"], reverse=True) for h in c["hits"]])

    def test_schroeder_and_schrodinger_share_a_prefix_after_folding(self):
        c = self.pair("Schrodinger equation", "Cantor-Bernstein-Schroeder theorem")
        self.assertEqual(c["rule"], "shared-prefix")
        hit = next(h for h in c["hits"] if set(h["tokens"]) == {"schrodinger", "schroeder"})
        self.assertEqual(set(h for h in hit["keys"]), {"shrodinger", "shroder"})

    def test_acronyms_pair_on_their_letter_names(self):
        c = self.pair("roe", "roi")
        hit = c["hits"][0]
        self.assertEqual(sorted(hit["spoken"]), ["arohee", "aroheye"])      # "ar oh ee", "ar oh eye"
        self.assertEqual(c["rule"], "prefix")
        c = self.pair("gpu", "tpu")
        self.assertEqual(sorted(c["hits"][0]["spoken"]), ["jeepeeyou", "teepeeyou"])
        self.assertEqual(c["rule"], "edit")
        self.assertEqual(c["side_type"], "targeted-targeted")

    def test_the_candidate_records_what_the_curator_receives(self):
        c = self.pair("ebitda", "ebit")
        by_term = {m["term"]: m for m in c["members"]}
        self.assertEqual(by_term["ebit"]["full_clips"], 214)
        self.assertEqual(by_term["ebitda"]["full_clips"], 0)
        self.assertEqual(set(by_term["ebitda"]["terms"]), {"ebitda", "adjusted ebitda"})
        self.assertEqual({m["side"] for m in c["members"]}, {"heldout", "targeted"})
        self.assertEqual({m["domain"] for m in c["members"]}, {"finance"})
        for h in c["hits"]:
            self.assertEqual(set(h), {"tokens", "spoken", "keys", "rule", "label_counts", "terms"})
        self.assertTrue(c["forced"])
        self.assertFalse(self.pair("roe", "roi")["forced"])

    def test_heldout_heldout_pairs_are_listed(self):
        self.assertEqual(self.pair("alkane", "alkene")["side_type"], "heldout-heldout")


class DetectorExclusions(unittest.TestCase):
    def test_gpu_and_gauss_do_not_pair(self):
        self.assertIsNone(find(detect(), "gpu", "Gauss law"))

    def test_an_inflection_does_not_pair(self):
        out = detect()
        self.assertIsNone(find(out, "thermodynamic equilibrium", "laws of thermodynamics"))
        self.assertGreaterEqual(out["summary"]["excluded"]["inflection"], 1)

    def test_a_token_with_300_or_more_label_occurrences_does_not_pair(self):
        self.assertIsNone(find(detect(words={"roi": 300}), "roe", "roi"))
        self.assertIsNotNone(find(detect(words={"roi": 299}), "roe", "roi"))
        out = detect(words={"roe": 1000})
        self.assertIsNone(find(out, "roe", "roi"))
        self.assertGreaterEqual(out["summary"]["excluded"]["common"], 1)

    def test_two_tokens_of_the_same_group_do_not_pair(self):
        s = make_split()
        self.assertEqual(group_of(s, "data set"), group_of(s, "dataset"))     # "data" / "dataset": a prefix
        for c in detect(s)["candidates"]:
            self.assertNotEqual(c["members"][0]["group"], c["members"][1]["group"])
            for h in c["hits"]:
                self.assertNotEqual(sorted(h["tokens"]), ["data", "dataset"])

    def test_opex_and_capex_are_not_found(self):
        self.assertIsNone(find(detect(), "opex", "capex"))                   # the plan's dev-informed seed


class ForcedPairs(unittest.TestCase):
    def test_an_always_accepted_pair_the_detector_excludes_is_added(self):
        out = detect(words={"ebit": 5000})
        c = find(out, "ebitda", "ebit")
        self.assertIsNotNone(c)
        self.assertTrue(c["forced"])
        self.assertEqual(c["rule"], "forced")
        self.assertEqual(out["summary"]["forced"]["ebitda / ebit"], "added")
        self.assertEqual(detect()["summary"]["forced"]["ebitda / ebit"], "detected")


class DetectorInputs(unittest.TestCase):
    def test_the_detector_is_deterministic(self):
        s = make_split()
        self.assertEqual(json.dumps(detect(s)), json.dumps(detect(make_split())))
        with roots():
            write_world(s, words={"enthalpy": 12})
            first = confusable.run()
            files = {p.name: p.read_bytes() for p in confusable.out_dir().iterdir()}
            second = confusable.run()
            self.assertEqual(first, second)
            self.assertEqual(files, {p.name: p.read_bytes() for p in confusable.out_dir().iterdir()})
            self.assertEqual(set(files) - {"words.json"}, {"candidates.json", "candidates.tsv",
                                                           "pairs.template.tsv"})

    def test_evaluation_and_hypothesis_paths_are_refused(self):
        with roots():
            r = layout.relabel_root()
            bad = [r / "devsets" / "manifests" / "tech-dev-2.jsonl", r / "devsets" / "scores" / "B0" / "tech-dev-2.json",
                   r / "devsets" / "scores" / "eval-stem" / "B0" / "stem-chem.json",
                   r / "decoder-only" / "B-T" / "lr1e-4" / "domain" / "tech-dev-2.json",
                   r / "b0prime" / "runs" / "b100" / "domain" / "stem-math.json",
                   r / "tmp" / "b0prime-regress" / "q2" / "occurrences.json"]
            write_world(make_split())
            for p in bad:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"format": 1, "counts": {}, "records": []}))
                for kind in confusable.INPUTS:
                    with self.assertRaises(confusable.InputRefused, msg=f"{kind}: {p}"):
                        confusable.require_input(p, kind)
            with self.assertRaises(confusable.InputRefused):
                confusable.run(words_path=bad[1])
            with self.assertRaises(confusable.InputRefused):
                confusable.run(split_path=bad[0])
            self.assertFalse(confusable.out_dir().joinpath("candidates.json").exists())

    def test_only_the_listed_inputs_are_accepted(self):
        with roots():
            paths = write_world(make_split())
            for kind, p in paths.items():
                self.assertEqual(confusable.require_input(p, kind), p.resolve())
            with self.assertRaises(confusable.InputRefused):
                confusable.require_input(paths["words"], "split")
            with self.assertRaises(confusable.InputRefused):
                confusable.require_input(layout.run_dir("pilot") / "clips.jsonl", "clips")

    def test_full_clip_counts_come_from_the_runs_clips(self):
        s = make_split()
        with roots():
            paths = write_world(s)
            out = confusable.run()
            c = find(out, "ebitda", "ebit")
            self.assertEqual({m["term"]: m["full_clips"] for m in c["members"]}, {"ebitda": 0, "ebit": 214})
            self.assertEqual(out["inputs"]["full"]["sha256"], sha256(paths["clips"]))
            self.assertEqual(out["inputs"]["split"]["sha256"], sha256(paths["split"]))
            self.assertNotIn("evaluation", json.dumps(out["inputs"]))

    def test_word_counts_are_whole_folded_tokens(self):
        c = confusable.word_counts(["Schrödinger's equation, EBITDA-adjusted", "the EBIT and ebit", "B2B data-set"])
        self.assertEqual(c["schrodinger"], 1)
        self.assertEqual(c["ebit"], 2)
        self.assertEqual(c["ebitda"], 1)
        self.assertEqual(c["set"], 1)
        self.assertNotIn("b2b", c)
        self.assertNotIn("b", c)



class CurationFile(unittest.TestCase):
    def setUp(self):
        self.split = make_split()
        self.cands = detect(self.split)
        self.sha = "ab" * 32

    def parse(self, text):
        return pairs.Curation.parse(text, self.cands, self.sha, self.split)

    def test_a_complete_curation_parses(self):
        c = self.parse(curation_text(self.cands, self.sha, {("roe", "roi"): ("reject", "different ratios, and "
                                                                                       "spelled they differ")}))
        self.assertEqual(len(c.rows), len(self.cands["candidates"]))
        self.assertNotIn("roe", {t for r in c.accepted for t in r["terms"]})
        self.assertIn(["ebitda", "ebit"], [r["terms"] for r in c.accepted])

    def test_malformed_files_are_refused(self):
        lines = curation_text(self.cands, self.sha).splitlines()
        rows = [i for i, l in enumerate(lines) if not l.startswith("#")]
        first = lines[rows[0]].split("\t")

        def edit(field: int, value: str | None) -> list[str]:
            f = list(first)
            if value is None:
                del f[field]
            else:
                f[field] = value
            return lines[:rows[0]] + ["\t".join(f)] + lines[rows[0] + 1:]
        bad = {
            "a missing row": lines[:rows[-1]] + lines[rows[-1] + 1:],
            "a duplicate row": lines + [lines[rows[0]]],
            "a duplicate row, members swapped": lines + ["\t".join([first[1], first[0]] + first[2:])],
            "an unknown term": edit(0, "no such term"),
            "a pair that is not a candidate": lines + ["enthalpy\ttitration\tedit\taccept\tno"],
            "a different rule": edit(2, "same-sound" if first[2] != "same-sound" else "edit"),
            "an unknown decision": edit(3, "maybe"),
            "an empty reason": edit(4, " "),
            "four fields": edit(4, None),
            "another candidate file": [l.replace(self.sha, "cd" * 32) for l in lines],
            "no candidate file": [l for l in lines if "candidates_sha256" not in l],
        }
        self.parse("\n".join(lines) + "\n")
        for name, ls in bad.items():
            with self.assertRaises(ValueError, msg=name):
                self.parse("\n".join(ls) + "\n")

    def test_the_always_accepted_pairs_must_be_accepted(self):
        for a, b in (("ebitda", "ebit"), ("Schrodinger equation", "Cantor-Bernstein-Schroeder theorem")):
            with self.assertRaises(ValueError, msg=(a, b)):
                self.parse(curation_text(self.cands, self.sha, {(a, b): ("reject", "no")}))

    def test_the_committed_file_is_absent_or_valid_for_its_candidate_file(self):
        if not pairs.PAIRS_FILE.exists():
            self.skipTest("terms/synthetic/pairs.tsv is not curated yet")
        cur = pairs.Curation.load()
        self.assertTrue(cur.accepted)


class SplitV3(unittest.TestCase):
    def world(self, decisions=None):
        s = make_split()
        paths = write_world(s)
        out = confusable.run()
        cand = confusable.out_dir() / "candidates.json"
        text = curation_text(out, sha256(cand), decisions)
        pfile = layout.relabel_root() / "pairs.tsv"
        pfile.write_text(text)
        return s, paths, pfile, cand

    def test_only_curated_moves_and_the_seed_change_side_and_groups_move_whole(self):
        with roots():
            s, paths, pfile, cand = self.world({("roe", "roi"): ("reject", "different ratios")})
            v3 = pairs.freeze_v3(pfile, cand)
            old = {e.phrase: e for e in s.entries}
            new = {tuple(e["phrase"].split()): e for e in v3["entries"]}
            self.assertEqual(set(old), set(new))
            moved = {group_of(s, p) for p in ("ebitda", "schrodinger equation", "opex")}
            for ph, e in old.items():
                want = "targeted" if e.group in moved else e.side
                self.assertEqual(new[ph]["side"], want, ph)
                self.assertEqual((new[ph]["group"], new[ph]["domain"], new[ph]["term"]), (e.group, e.domain, e.term))
            self.assertEqual(new[("adjusted", "ebitda")]["side"], "targeted")             # the whole group
            for p in ("roe", "alkane", "alkene", "laws of thermodynamics"):                # rejected, held-out pairs
                self.assertEqual(new[tuple(p.split())]["side"], "heldout", p)
            self.assertEqual(v3["version"], 3)

    def test_heldout_v3_is_a_subset_of_v2_and_split_v2_is_unchanged(self):
        with roots():
            s, paths, pfile, cand = self.world()
            v2_bytes = paths["split"].read_bytes()
            v3 = pairs.freeze_v3(pfile, cand)
            v2_file = layout.split_version_path(2)
            self.assertEqual(v2_file.read_bytes(), v2_bytes)
            self.assertEqual(paths["split"].read_bytes(), v2_bytes)
            held2 = {e.group for e in s.heldout()}
            held3 = {e["group"] for e in v3["entries"] if e["side"] == "heldout"}
            self.assertTrue(held3 < held2)
            self.assertEqual(v3["base"]["sha256"], hashlib.sha256(v2_bytes).hexdigest())
            self.assertEqual(pairs.freeze_v3(pfile, cand), v3)       # frozen: same inputs
            self.assertEqual(v2_file.read_bytes(), v2_bytes)
            pfile.write_text(curation_text(json.loads(cand.read_text()), sha256(cand),
                                           {("roe", "roi"): ("reject", "different ratios")}))
            with self.assertRaises(RuntimeError):                                          # changed curation
                pairs.freeze_v3(pfile, cand)
            v3_loaded = vocab.Split.load(layout.split_version_path(3))
            self.assertEqual(len(v3_loaded.entries), len(s.entries))

    def test_every_move_is_listed_with_its_reason(self):
        with roots():
            s, paths, pfile, cand = self.world({("roe", "roi"): ("reject", "different ratios")})
            v3 = pairs.freeze_v3(pfile, cand)
            ch = json.loads((layout.terms_dir() / "split-changes-v3.json").read_text())
            moved = {m["group"]: m for m in ch["moved"]}
            self.assertEqual(set(moved), {group_of(s, p) for p in ("ebitda", "schrodinger equation", "opex")})
            ebitda = moved[group_of(s, "ebitda")]
            self.assertEqual(ebitda["reasons"][0]["kind"], "pair")
            self.assertEqual(ebitda["reasons"][0]["terms"], ["ebitda", "ebit"])
            self.assertTrue(ebitda["reasons"][0]["reason"])
            opex = moved[group_of(s, "opex")]
            self.assertEqual([r["kind"] for r in opex["reasons"]], ["seed"])
            self.assertTrue(opex["dev_informed"])
            self.assertFalse(ebitda["dev_informed"])
            changed = {tuple(c["phrase"].split()) for c in ch["changed"]}
            self.assertEqual(changed, {("ebitda",), ("adjusted", "ebitda"), ("schrodinger", "equation"), ("opex",)})
            self.assertEqual(v3["dev_informed_groups"], [group_of(s, "opex")])

    def test_v3_lists_every_accepted_pair_for_generation_and_metrics(self):
        with roots():
            s, paths, pfile, cand = self.world({("roe", "roi"): ("reject", "different ratios")})
            v3 = pairs.freeze_v3(pfile, cand)
            by_terms = {tuple(p["terms"]): p for p in v3["pairs"]}
            self.assertNotIn(("roe", "roi"), by_terms)
            self.assertTrue(by_terms[("ebitda", "ebit")]["trained"])
            self.assertTrue(by_terms[("opex", "capex")]["seed"])
            self.assertTrue(by_terms[("opex", "capex")]["trained"])
            self.assertFalse(by_terms[("alkane", "alkene")]["trained"])                   # held-out-held-out
            self.assertEqual(by_terms[("opex", "capex")]["hits"][0]["tokens"], ["opex", "capex"])


class Provenance(unittest.TestCase):
    def pool(self, d: Path, split_file: Path, labels: list[str]):
        rows = [{"id": f"c{i:03d}", "label": t, "audio_filepath": "/dev/null", "duration": 1.0, "sha256": "0"}
                for i, t in enumerate(labels)]
        write_jsonl(d / "clips.jsonl", rows)
        meta = {"clips": len(rows), "clips_sha256": sha256(d / "clips.jsonl"), "provisional": False,
                "hygiene_split_sha256": sha256(split_file), "split_sha256": sha256(split_file)}
        (d / "clips.meta.json").write_text(json.dumps(meta))

    def v3(self):
        s = make_split()
        write_world(s)
        out = confusable.run()
        cand = confusable.out_dir() / "candidates.json"
        pfile = layout.relabel_root() / "pairs.tsv"
        pfile.write_text(curation_text(out, sha256(cand)))
        pairs.freeze_v3(pfile, cand)
        return s

    def test_full_clips_pass_v3s_guard_and_a_failing_clip_refuses_the_pool(self):
        with roots() as t:
            self.v3()
            v2, v3 = layout.split_version_path(2), layout.split_version_path(3)
            d = t / "full"
            clean = ["We compared the ebit of both firms.", "The Cantor-Bernstein-Schroeder theorem is neat."]
            self.pool(d, v2, clean)
            self.assertEqual(len(mixing.ClipPool.load(d, split_path=v3).clips), 2)
            self.assertEqual(len(mixing.ClipPool.load(d).clips), 2)                     # its own split: as before
            self.pool(d, v2, clean + ["The alkane burned slowly."])                       # alkane stays held out
            with self.assertRaises(RuntimeError):
                mixing.ClipPool.load(d, split_path=v3)

    def test_a_pool_of_another_split_is_refused(self):
        with roots() as t:
            self.v3()
            other = t / "other-split.json"
            other.write_text("{}")
            d = t / "pool"
            self.pool(d, other, ["We compared the ebit of both firms."])
            with self.assertRaises(RuntimeError):
                mixing.ClipPool.load(d, split_path=layout.split_version_path(3))

    def test_full_is_checked_and_recorded(self):
        with roots():
            self.v3()
            rec = pairs.check_full()
            self.assertTrue(rec["pass"])
            self.assertEqual(rec["clips"], sum(clip_counts(make_split()).values()))
            self.assertEqual(rec["split_sha256"], sha256(layout.split_version_path(3)))
            self.assertTrue((layout.base() / "v5" / "provenance-full.json").exists())


if __name__ == "__main__":
    unittest.main()
