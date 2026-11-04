"""Contract tests (CPU): Step 1's measurement (b0prime-v2 plan, "Output style", "Measurement" and "Rule"):
b0v2_step1.py.

- Delta_c(s) = WER(T_c applied to the saved hypotheses) - WER(the saved hypotheses), in points, with the pinned scorer
  (evaluate.normalize and edit_counts against the stored normalized reference); a hand-computed fixture over the ten
  decision sets, the guard (a hypothesis that would be emptied is scored unchanged), stored records that the pinned
  scorer does not reproduce refused;
- the decision sets (H4, YODAS dev, the five domain parts), equally weighted; the reported sets; the convention groups;
- clusters: H4 as training/devsplit.py, YODAS and the domain parts by recording, med-dev-2 by note (its conditions
  together);
- the bootstrap: training/bootstrap.paired over the ten sets, the ten-set mean recomputed per replicate (checked
  against an independent derivation of the documented draws);
- the rule: adopted only if the mean is below 0 and the upper bound of its 90% interval is below 0; H4's opposite sign
  flagged;
- the inputs: B0''s and Kea-B''s saved dev records only (a test-set name is refused), the files stage 4's report reads.
"""
from __future__ import annotations

import random
import unittest

import numpy as np

import b0v2_step1 as s1
from training import style

ST = style.Style.load()


def record(i: str, ref: str, hyp: str) -> dict:
    import evaluate
    r = evaluate.score(ref, hyp)
    return {"id": i, "ref": ref, "hyp": hyp, **r}


IDS = {
    "librispeech_dev_clean": lambda m, k: f"{100 + m}-1-{k:04d}",
    "librispeech_dev_other": lambda m, k: f"{200 + m}-1-{k:04d}",
    "ami_dev": lambda m, k: f"ami:AMI_ZZ90{m:02d}q_H00_ABC123_{k:07d}_{k + 1:07d}",
    "voxpopuli_dev": lambda m, k: f"voxpopuli:1999010{m}-0900-PLENARY-{k}",
    "yodas_dev": lambda m, k: f"yodas:en000_{m:08d}_ABCDEFGH-_{m}_{k}_00_1_00",
    "med-dev-1": lambda m, k: f"med-dev-1:rec{m}_a-{k:04d}",
    "med-dev-2": lambda m, k: f"med-dev-2:n9{m:02d}-{('c1', 'c2', 'c_low')[k % 3]}-{k:04d}",
    "med-dev-3": lambda m, k: f"med-dev-3:{m:032x}-{k:04d}",
    "tech-dev-1": lambda m, k: f"tech-dev-1:rec-{m}-{k:04d}",
    "tech-dev-2": lambda m, k: f"tech-dev-2:{9000000 + m}-{k:04d}",
}


def hand_fixture() -> dict[str, list[dict]]:
    """One or two utterances per decision set; Delta per set and class computed by hand below."""
    plain = {name: [record(IDS[name](1, 0), "hello world", "hello world")] for name in s1.DECISION_SETS}
    plain["librispeech_dev_clean"] = [record(IDS["librispeech_dev_clean"](1, 0), "breathless or", "breathless or..."),
                                      record(IDS["librispeech_dev_clean"](2, 0), "the cat sat", "the the cat sat")]
    plain["ami_dev"] = [record(IDS["ami_dev"](1, 0), "so the plan", "so, you know, the plan")]
    plain["med-dev-1"] = [record(IDS["med-dev-1"](1, 0), "so you know the plan", "so, you know, the plan")]
    plain["tech-dev-2"] = [record(IDS["tech-dev-2"](1, 0), "we will", "w we will")]
    plain["voxpopuli_dev"] = [record(IDS["voxpopuli_dev"](1, 0), "you know", "you know,")]
    return plain


def random_fixture(seed: int = 0) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    words = "the plan was to go there and see what we can do about it".split()
    artifacts = [lambda h: h + "...", lambda h: "the " + h, lambda h: "so, you know, " + h, lambda h: "w " + h,
                 lambda h: h, lambda h: h, lambda h: h.replace("the", "th the", 1), lambda h: h + ", I mean"]
    out = {}
    for name in s1.DECISION_SETS:
        recs = []
        for m in range(1, 6 + rng.randrange(4)):
            for k in range(2 + rng.randrange(3)):
                ref = " ".join(rng.choice(words) for _ in range(3 + rng.randrange(6)))
                hyp = rng.choice(artifacts)(ref)
                recs.append(record(IDS[name](m, k), ref, hyp))
        out[name] = recs
    return out


class HandComputed(unittest.TestCase):
    def test_delta_per_set_and_mean(self):
        res = s1.measure(hand_fixture(), ST, replicates=200)
        want = {c: {name: 0.0 for name in s1.DECISION_SETS} for c in style.CLASSES}
        # librispeech_dev_clean: 5 reference words, 2 base errors ("." inserted; "the" repeated)
        want["E"]["librispeech_dev_clean"] = 100 * (1 - 2) / 5
        want["R"]["librispeech_dev_clean"] = 100 * (1 - 2) / 5
        # ami_dev: "you know" inserted twice (2 errors over 3 words); D removes it
        want["D"]["ami_dev"] = 100 * (0 - 2) / 3
        # med-dev-1: the reference keeps "you know"; D deletes two reference words (5 words)
        want["D"]["med-dev-1"] = 100 * (2 - 0) / 5
        # tech-dev-2: "w" inserted (2 words); F removes it
        want["F"]["tech-dev-2"] = 100 * (0 - 1) / 2
        # voxpopuli_dev: "you know," alone is never emptied, so D leaves it: 0
        for c in style.CLASSES:
            for name in s1.DECISION_SETS:
                with self.subTest(cls=c, set=name):
                    self.assertAlmostEqual(res["classes"][c]["sets"][name]["point"], want[c][name], places=9)
            with self.subTest(cls=c, set="mean"):
                self.assertAlmostEqual(res["classes"][c]["mean"]["point"], sum(want[c].values()) / 10, places=9)
        self.assertAlmostEqual(res["classes"]["E"]["mean"]["point"], -2.0, places=9)
        self.assertAlmostEqual(res["classes"]["F"]["mean"]["point"], -5.0, places=9)
        self.assertAlmostEqual(res["classes"]["D"]["mean"]["point"], (-200 / 3 + 40) / 10, places=9)
        self.assertAlmostEqual(res["wer"]["librispeech_dev_clean"]["base"], 40.0, places=9)
        self.assertEqual(res["changed"]["D"]["voxpopuli_dev"]["hypotheses"], 0)
        self.assertEqual(res["changed"]["E"]["librispeech_dev_clean"]["hypotheses"], 1)

    def test_rescore_uses_the_pinned_scorer(self):
        import evaluate
        recs = [record("1-1-0001", "the cat sat", "The the cat sat..."), record("1-1-0002", "a dog", "a dog")]
        out = s1.rescore(recs, "R", ST)
        self.assertEqual([r["id"] for r in out], ["1-1-0001", "1-1-0002"])
        self.assertEqual(out[0]["hyp"], "The cat sat...")
        self.assertEqual(out[0]["hyp_norm"], evaluate.normalize("The cat sat..."))
        self.assertEqual((out[0]["S"], out[0]["D"], out[0]["I"]),
                         evaluate.edit_counts(out[0]["ref_norm"].split(), out[0]["hyp_norm"].split()))
        self.assertTrue(out[0]["changed"])
        self.assertEqual({k: out[1][k] for k in recs[1]}, recs[1])
        self.assertFalse(out[1]["changed"])

    def test_unreproducible_records_refused(self):
        recs = hand_fixture()
        bad = dict(recs["ami_dev"][0], hyp_norm="something else")
        with self.assertRaisesRegex(ValueError, "pinned scorer"):
            s1.measure({**recs, "ami_dev": [bad]}, ST, replicates=10)
        bad = dict(recs["ami_dev"][0], I=0)
        with self.assertRaisesRegex(ValueError, "pinned scorer"):
            s1.measure({**recs, "ami_dev": [bad]}, ST, replicates=10)

    def test_unscored_records_are_left_out(self):
        recs = hand_fixture()
        extra = dict(record(IDS["ami_dev"](2, 0), "", "you know..."), scored=False)
        res = s1.measure({**recs, "ami_dev": recs["ami_dev"] + [extra]}, ST, replicates=10)
        self.assertAlmostEqual(res["classes"]["D"]["sets"]["ami_dev"]["point"], -200 / 3, places=9)
        self.assertEqual(res["utterances"]["ami_dev"], 1)

    def test_every_decision_set_required(self):
        recs = hand_fixture()
        del recs["med-dev-3"]
        with self.assertRaisesRegex(ValueError, "med-dev-3"):
            s1.measure(recs, ST, replicates=10)


class Sets(unittest.TestCase):
    def test_decision_and_reported_sets(self):
        self.assertEqual(s1.H4_SETS, ("librispeech_dev_clean", "librispeech_dev_other", "ami_dev", "voxpopuli_dev"))
        self.assertEqual(s1.DECISION_SETS, s1.H4_SETS + ("yodas_dev", "med-dev-1", "med-dev-2", "med-dev-3",
                                                         "tech-dev-1", "tech-dev-2"))
        self.assertEqual(s1.REPORTED_SETS, ("stem-chem", "stem-cs", "stem-math", "cv27-en-dev", "ami-sdm-dev"))
        members = sorted(n for g in s1.GROUPS.values() for n in g)
        self.assertEqual(members, sorted(s1.DECISION_SETS))
        self.assertEqual(s1.GROUPS["verbatim"], ("ami_dev", "med-dev-1", "tech-dev-2"))
        self.assertEqual(s1.GROUPS["scripted"], ("med-dev-2",))

    def test_clusters(self):
        from training import devsplit
        import synthetic_report
        for name in s1.H4_SETS:
            fn = s1.cluster_fn(name)
            for m, k in ((1, 0), (2, 3)):
                i = IDS[name](m, k)
                self.assertEqual(fn(i), devsplit.cluster_of(name, i))
        for name in ("med-dev-1", "med-dev-3", "tech-dev-1", "tech-dev-2", "stem-chem"):
            fn = s1.cluster_fn(name)
            i = f"{name}:rec-a_1-0003" if name == "stem-chem" else IDS[name](3, 7)
            self.assertEqual(fn(i), synthetic_report.recording_of(i))
        note = s1.cluster_fn("med-dev-2")
        self.assertEqual(note("med-dev-2:n907-c1-0000"), note("med-dev-2:n907-c2-0003"))
        self.assertEqual(note("med-dev-2:n907-c_low-0001"), note("med-dev-2:n907-c3-0002"))
        self.assertNotEqual(note("med-dev-2:n907-c1-0000"), note("med-dev-2:n908-c1-0000"))
        yod = s1.cluster_fn("yodas_dev")
        self.assertEqual(yod("yodas:en000_00000042_abcd_efghij_294_94_2_54"),
                         yod("yodas:en000_00000042_abcd_efghij_2956_82_7_28"))
        self.assertNotEqual(yod("yodas:en000_00000042_abcd_efghij_294_94_2_54"),
                            yod("yodas:en000_00000043_abcd_efghiX_294_94_2_54"))
        sdm = s1.cluster_fn("ami-sdm-dev")
        self.assertEqual(sdm("ami-sdm:AMI_ZZ9001q_sdm_ABC123_0000100_0000200"),
                         devsplit.cluster_of("ami_dev", "ami:AMI_ZZ9001q_H00_ABC123_0000100_0000200"))
        self.assertIsNone(s1.cluster_fn("cv27-en-dev"))
        for name, bad in (("yodas_dev", "yodas:nope"), ("med-dev-2", "med-dev-2:x-0001"), ("ami-sdm-dev", "ami:x")):
            with self.assertRaises(ValueError):
                s1.cluster_fn(name)(bad)

    def test_outputs_only_under_step1_or_tmp(self):
        from pathlib import Path
        from labeling import layout
        root = layout.relabel_root().resolve()
        self.assertEqual(s1.output_dir(None), root / "b0prime-v2" / "step1")
        for ok in (root / "b0prime-v2" / "step1", root / "b0prime-v2" / "step1" / "smoke", root / "tmp" / "s1"):
            self.assertEqual(s1.output_dir(ok), ok)
        for bad in (Path("."), Path(__file__).parent, root, root / "b0prime-v2", root / "b0prime" / "runs" / "b100",
                    root / "b0prime-v2" / "step1" / ".." / "x", root / "tmpx"):
            with self.assertRaises(ValueError):
                s1.output_dir(bad)

    def test_inputs_are_dev_records_only(self):
        import keab_report
        dirs = keab_report.score_dirs()
        for name in s1.DECISION_SETS + s1.REPORTED_SETS:
            self.assertIn(s1.score_path("B0p", name).parent, dirs["B0p"]["fp32"])
            self.assertEqual(s1.score_path("Kea-B'", name).parent, dirs["Kea-B'"]["x1"])
            self.assertEqual(s1.score_path("B0p", name).name, f"{name}.json")
        self.assertEqual(s1.score_path("B0p", "ami_dev").parent.name, "eval-dev")
        self.assertEqual(s1.score_path("B0p", "med-dev-2").parent.name, "domain")
        self.assertEqual(s1.score_path("B0p", "ami-sdm-dev").parent, dirs["B0p"]["fp32"][-1])
        for name in ("librispeech_test_clean", "ami_test", "spgispeech_test", "med-test-1", "x"):
            for model in ("B0p", "Kea-B'"):
                with self.assertRaises(ValueError):
                    s1.score_path(model, name)
        with self.assertRaises(ValueError):
            s1.score_path("B0", "ami_dev")


class Bootstrap(unittest.TestCase):
    def test_the_ten_set_mean_is_recomputed_per_replicate(self):
        """bootstrap.paired's documented draws, derived independently: per decision set (sorted order k), clusters
        drawn with replacement from SeedSequence([seed, k]); per replicate, each set's Delta, then their mean."""
        recs = random_fixture()
        reps, seed = 400, 1234
        res = s1.measure(recs, ST, replicates=reps, seed=seed)
        for c in style.CLASSES:
            per_set = []
            for k, name in enumerate(sorted(s1.DECISION_SETS)):
                fn = s1.cluster_fn(name)
                new = s1.rescore(recs[name], c, ST)
                clusters = sorted({fn(r["id"]) for r in recs[name]})
                ix = {x: j for j, x in enumerate(clusters)}
                words, eb, ec = (np.zeros(len(clusters)) for _ in range(3))
                for r, t in zip(recs[name], new):
                    j = ix[fn(r["id"])]
                    words[j] += len(r["ref_norm"].split())
                    eb[j] += r["S"] + r["D"] + r["I"]
                    ec[j] += t["S"] + t["D"] + t["I"]
                rng = np.random.default_rng(np.random.SeedSequence([seed, k]))
                counts = np.stack([np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters))
                                   for _ in range(reps)]).astype(float)
                d = 100 * (counts @ ec) / (counts @ words) - 100 * (counts @ eb) / (counts @ words)
                per_set.append(d)
                lo, hi = np.percentile(d, [5, 95])
                self.assertAlmostEqual(res["classes"][c]["sets"][name]["ci"][0], lo, places=9)
                self.assertAlmostEqual(res["classes"][c]["sets"][name]["ci"][1], hi, places=9)
            lo, hi = np.percentile(np.mean(per_set, axis=0), [5, 95])
            with self.subTest(cls=c):
                self.assertAlmostEqual(res["classes"][c]["mean"]["ci"][0], lo, places=9)
                self.assertAlmostEqual(res["classes"][c]["mean"]["ci"][1], hi, places=9)
                naive = np.mean([res["classes"][c]["sets"][n]["ci"][0] for n in s1.DECISION_SETS])
                self.assertNotAlmostEqual(naive, lo, places=3)
            h4 = [per_set[sorted(s1.DECISION_SETS).index(n)] for n in s1.H4_SETS]
            lo, hi = np.percentile(np.mean(h4, axis=0), [5, 95])
            self.assertAlmostEqual(res["classes"][c]["h4"]["ci"][0], lo, places=9)
            self.assertAlmostEqual(res["classes"][c]["h4"]["ci"][1], hi, places=9)
            g = [per_set[sorted(s1.DECISION_SETS).index(n)] for n in s1.GROUPS["verbatim"]]
            lo, hi = np.percentile(np.mean(g, axis=0), [5, 95])
            self.assertAlmostEqual(res["classes"][c]["groups"]["verbatim"]["ci"][0], lo, places=9)
        self.assertEqual(res["replicates"], reps)
        self.assertEqual(res["seed"], seed)

    def test_defaults_are_the_plans(self):
        self.assertEqual(s1.REPLICATES, 10_000)
        self.assertEqual(s1.CI, 0.90)


class Rule(unittest.TestCase):
    def test_adopted_only_below_zero_with_the_upper_bound_below_zero(self):
        h4 = {"point": -0.1, "ci": [-0.2, -0.05]}
        for mean, adopted in (({"point": -0.10, "ci": [-0.20, -0.01]}, True),
                              ({"point": -0.10, "ci": [-0.20, 0.00]}, False),
                              ({"point": -0.10, "ci": [-0.20, 0.05]}, False),
                              ({"point": 0.00, "ci": [-0.10, -0.01]}, False),
                              ({"point": 0.10, "ci": [0.05, 0.20]}, False)):
            with self.subTest(mean=mean):
                self.assertEqual(s1.decide(mean, h4)["adopted"], adopted)

    def test_h4_opposite_sign_flag(self):
        mean = {"point": -0.10, "ci": [-0.20, -0.01]}
        self.assertTrue(s1.decide(mean, {"point": 0.20, "ci": [0.05, 0.30]})["h4_opposite"])
        self.assertFalse(s1.decide(mean, {"point": 0.20, "ci": [-0.05, 0.30]})["h4_opposite"])
        self.assertFalse(s1.decide(mean, {"point": -0.20, "ci": [-0.30, -0.05]})["h4_opposite"])
        self.assertTrue(s1.decide({"point": 0.10, "ci": [0.01, 0.20]}, {"point": -0.2, "ci": [-0.3, -0.1]})["h4_opposite"])

    def test_measure_applies_the_rule(self):
        res = s1.measure(hand_fixture(), ST, replicates=50)
        for c in style.CLASSES:
            e = res["classes"][c]
            self.assertEqual(e["decision"], s1.decide(e["mean"], e["h4"]))


if __name__ == "__main__":
    unittest.main()
