"""Scoring contracts (plans/build.md "Scoring", "Code and tests" 10), on stubs with known answers:

- per condition, the equal-weight mean of the five set WERs and the four-set human-reference mean
  (YODAS excluded, labeled pseudo-reference agreement); the noisy mean; degradation against the
  model's own clean row; the gap to B0;
- the paired cluster bootstrap: identical models give a zero-width interval, a constant per-word
  shift gives exactly that shift (pairing across models and conditions preserved), and clusters
  are the resampling unit;
- the scoring entry point takes the GPU lock, pins the decoder mode before every decode and checks
  it after, refuses another mode, refuses manifests that do not match the build, and writes the
  summary last. No model or GPU is used.

Review round 2: the entry point runs on a real (synthetic) builder output and validates the
build identity, manifests, expected audio paths and audio SHA-256 (findings 1 and 3);
comparisons require identical build identity, condition sets, per-ID references and exclusion
masks (finding 3); the report breaks each condition's WER difference down by noise source
group (finding 9).
"""
from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from pathlib import Path

import noisy_fixtures as fx
import numpy as np

import noisy_score

SETS = ["librispeech_dev_clean", "librispeech_dev_other", "ami_dev", "voxpopuli_dev", "yodas_dev"]


def rec(uid: str, errors: int, words: int, scored: bool = True) -> dict:
    """A record as evaluate.decode returns it (errors all substitutions)."""
    return {"id": uid, "ref": " ".join(["w"] * words), "hyp": "", "ref_norm": " ".join(["w"] * words),
            "hyp_norm": "", "S": errors, "D": 0, "I": 0, "duration": 1.0, "scored": scored}


def results_for(per_set_errors: dict[str, list[tuple[int, int]]], prefix: str = "u") -> dict[str, list[dict]]:
    return {s: [rec(f"{s}:{prefix}{k}", e, w) for k, (e, w) in enumerate(items)]
            for s, items in per_set_errors.items()}


class AggregationTest(unittest.TestCase):
    def test_equal_weight_five_set_mean_and_four_set_human_mean(self):
        # Set WERs 0.1 .. 0.5 with very different word counts: the mean is not the pooled WER.
        words = {"librispeech_dev_clean": 1000, "librispeech_dev_other": 10, "ami_dev": 50,
                 "voxpopuli_dev": 400, "yodas_dev": 20}
        wers = dict(zip(SETS, (0.1, 0.2, 0.3, 0.4, 0.5), strict=True))
        per_set = {s: [rec(f"{s}:a", round(wers[s] * words[s]), words[s])] for s in SETS}
        per_set["ami_dev"].append(rec("ami_dev:excluded", 7, 3, scored=False))  # excluded from the WER
        summary = noisy_score.summarize({"clean": per_set})
        row = summary["conditions"]["clean"]
        for s in SETS:
            self.assertAlmostEqual(row["sets"][s]["wer"], wers[s])
        self.assertAlmostEqual(row["mean_wer"], 0.3)
        self.assertAlmostEqual(row["human_mean_wer"], 0.25)
        self.assertEqual(row["sets"]["yodas_dev"]["label"], "pseudo-reference agreement")
        self.assertEqual(row["sets"]["ami_dev"]["excluded_empty_reference"], 1)

    def test_noisy_mean_and_degradation(self):
        clean = results_for({s: [(1, 10)] for s in SETS})
        a = results_for({s: [(2, 10)] for s in SETS})
        b = results_for({s: [(4, 10)] for s in SETS})
        summary = noisy_score.summarize({"clean": clean, "eng10": a, "eng0": b})
        self.assertAlmostEqual(summary["noisy_mean_wer"], 0.3)
        self.assertEqual(summary["noisy_conditions"], ["eng10", "eng0"])
        self.assertAlmostEqual(summary["conditions"]["eng10"]["degradation"], 0.1)
        self.assertAlmostEqual(summary["conditions"]["eng0"]["degradation"], 0.3)
        self.assertAlmostEqual(summary["noisy_degradation"], 0.2)


IDENTITY = {"lock_sha256": "a" * 64, "manifest_sha256": "b" * 64, "conditions": ["clean", "eng0"]}


class BootstrapTest(unittest.TestCase):
    clusters = {}

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        cls.base = {}
        for s in SETS:
            items = []
            for _ in range(12):
                words = int(rng.integers(5, 30))
                items.append((int(rng.integers(0, words)), words))
            cls.base[s] = items
            for k in range(12):
                cls.clusters[(s, f"{s}:u{k}")] = f"c{k // 3}"  # clusters of three utterances

    def models(self, shift_words: float = 0.0):
        """B0 and a model whose errors are B0's plus shift_words x reference words per utterance."""
        b0 = {"clean": results_for(self.base), "eng0": results_for(
            {s: [(e + w // 2, w) for e, w in items] for s, items in self.base.items()})}
        other = {c: {s: [rec(r["id"], r["S"] + round(shift_words * len(r["ref_norm"].split())),
                             len(r["ref_norm"].split())) for r in rows] for s, rows in sets.items()}
                 for c, sets in b0.items()}
        return {"B0": b0, "X": other}

    def compare(self, models, clusters=None, builds=None, **kw):
        builds = builds or {m: IDENTITY for m in models}
        return noisy_score.compare(models, self.clusters if clusters is None else clusters, reference="B0",
                                   resamples=kw.pop("resamples", 500), seed=3, builds=builds, **kw)

    def test_identical_models_give_zero_width_intervals(self):
        out = self.compare(self.models(0.0))
        for cond in ("clean", "eng0"):
            gap = out["models"]["X"]["gap_to_reference"][cond]
            self.assertEqual((gap["point"], gap["low"], gap["high"]), (0.0, 0.0, 0.0))
        self.assertEqual(out["resamples"], 500)

    def test_constant_shift_is_recovered_exactly(self):
        out = self.compare(self.models(1.0))
        for cond in ("clean", "eng0"):
            gap = out["models"]["X"]["gap_to_reference"][cond]
            for key in ("point", "low", "high"):
                self.assertAlmostEqual(gap[key], 1.0, places=12)
        noisy = out["models"]["X"]["noisy_mean_gap"]
        self.assertAlmostEqual(noisy["high"] - noisy["low"], 0.0, places=12)
        # B0's own degradation is a real quantity with a nonzero interval; X's equals it exactly.
        d_b0, d_x = (out["models"][m]["degradation"]["eng0"] for m in ("B0", "X"))
        self.assertGreater(d_b0["high"] - d_b0["low"], 0.0)
        self.assertAlmostEqual(d_x["low"], d_b0["low"], places=12)
        self.assertAlmostEqual(d_x["high"], d_b0["high"], places=12)

    def test_clusters_are_the_resampling_unit(self):
        # One set, two clusters: a (two utterances, all wrong) and b (one utterance, all right).
        # Cluster resampling can only produce WER 0, 1/2 or 1; utterance resampling could not.
        per_set = {"librispeech_dev_clean": [rec("x:1", 5, 5), rec("x:2", 5, 5), rec("x:3", 0, 10)]}
        clusters = {("librispeech_dev_clean", "x:1"): "a", ("librispeech_dev_clean", "x:2"): "a",
                    ("librispeech_dev_clean", "x:3"): "b"}
        draws = noisy_score.bootstrap_set_wers({"M": {"clean": per_set}}, clusters, resamples=400, seed=5)
        values = set(np.round(draws["M"]["clean"]["librispeech_dev_clean"], 12))
        self.assertTrue(values <= {0.0, 0.5, 1.0}, values)
        self.assertEqual(len(values), 3)

    def test_missing_cluster_refused(self):
        with self.assertRaises(KeyError):
            self.compare(self.models(0.0), clusters={}, resamples=10)

    def test_reference_mismatch_with_equal_word_counts_refused(self):
        models = self.models(0.0)
        r = models["X"]["eng0"]["ami_dev"][4]
        r["ref_norm"] = " ".join(["v"] * len(r["ref_norm"].split()))  # same word count, other words
        with self.assertRaisesRegex(ValueError, "reference"):
            self.compare(models, resamples=10)

    def test_exclusion_mask_mismatch_refused(self):
        models = self.models(0.0)
        models["X"]["clean"]["yodas_dev"][2]["scored"] = False
        with self.assertRaisesRegex(ValueError, "exclusion"):
            self.compare(models, resamples=10)

    def test_build_identity_mismatch_refused(self):
        models = self.models(0.0)
        for change in ({"lock_sha256": "c" * 64}, {"manifest_sha256": "d" * 64},
                       {"conditions": ["clean", "eng0", "radio"]}):
            builds = {"B0": IDENTITY, "X": {**IDENTITY, **change}}
            with self.assertRaisesRegex(ValueError, "build"):
                self.compare(models, builds=builds, resamples=10)

    def test_condition_set_mismatch_refused(self):
        models = self.models(0.0)
        del models["X"]["eng0"]
        with self.assertRaisesRegex(ValueError, "condition"):
            self.compare(models, resamples=10)

    def test_breakdown_by_noise_source_group(self):
        models = self.models(0.0)
        groups = {}
        for s in SETS:
            for k in range(12):
                groups[("eng0", s, f"{s}:u{k}")] = {"noise": "fsd50k:Car" if k < 6 else "fsd50k:Truck"}
        for rows in models["X"]["eng0"].values():  # X: one extra error per Car utterance
            for r in rows:
                if int(r["id"].rsplit("u", 1)[1]) < 6:
                    r["S"] += 1
        out = self.compare(models, groups=groups, resamples=10)
        car = out["by_group"]["eng0"]["noise"]["fsd50k:Car"]
        truck = out["by_group"]["eng0"]["noise"]["fsd50k:Truck"]
        self.assertEqual(car["utterances"], 30)
        car_words = sum(len(r["ref_norm"].split()) for rows in models["B0"]["eng0"].values()
                        for r in rows if int(r["id"].rsplit("u", 1)[1]) < 6)
        self.assertAlmostEqual(car["gap_to_reference"]["X"], 30 / car_words, places=12)
        self.assertEqual(truck["gap_to_reference"]["X"], 0.0)
        self.assertIn("conditional on the frozen noise realization", out["interval_scope"])


class ScoreEntryPointTest(unittest.TestCase):
    """On a real builder output (fixtures' tiny_build: 17 utterances x 16 conditions)."""

    @classmethod
    def setUpClass(cls):
        cls.base = fx.fresh_dir("score-build")
        cls.source, _ = fx.tiny_build(cls.base)
        cls.conditions = json.loads((cls.source / "build.json").read_text())["conditions"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base, ignore_errors=True)

    def setUp(self):
        self.root = fx.fresh_dir("score")
        self.build = self.root / "build"
        shutil.copytree(self.source, self.build)
        for path in (self.build / "manifests").rglob("*.jsonl"):  # absolute paths: point at the copy
            path.write_text(path.read_text().replace(str(self.source.resolve()), str(self.build.resolve())))
        self.events = []

    def hooks(self, mode_after: str = "no_while_loops"):
        events = self.events

        class Model:
            mode = None

        @contextlib.contextmanager
        def lock(label):
            events.append(("lock", label))
            yield
            events.append(("unlock", label))

        def load(spec, device):
            events.append(("load", spec.kind))
            return Model()

        def pin(model, role):
            model.mode = "no_while_loops"
            events.append(("pin", role))
            return model.mode

        def require(model, role):
            events.append(("require", role))
            if mode_after != "no_while_loops":
                raise noisy_score.DecoderModeError(f"{role}: {mode_after}")
            return model.mode

        def decode(model, records, batch_size, device):
            assert model.mode == "no_while_loops"
            events.append(("decode", len(records)))
            return [rec(r["id"], 1, 4) for r in records]

        return noisy_score.Hooks(gpu_lock=lock, load_model=load, pin_decoder=pin, require_decoder=require,
                                 decode=decode, deterministic=lambda: {})

    def test_scores_every_condition_under_the_lock_with_pinned_decoder(self):
        spec = noisy_score.ModelSpec(kind="binary", path=self.root / "export", name="arm-A")
        summary = noisy_score.score_model(spec, self.build, self.root / "scores", hooks=self.hooks(), device="cpu")
        kinds = [e[0] for e in self.events]
        self.assertEqual(kinds[0], "lock")
        self.assertEqual(kinds[-1], "unlock")
        self.assertEqual(kinds.count("decode"), len(self.conditions) * len(SETS))
        for i, kind in enumerate(kinds):
            if kind == "decode":
                self.assertEqual(kinds[i - 1], "pin")
                self.assertEqual(kinds[i + 1], "require")
        out = self.root / "scores" / "arm-A"
        for cond in self.conditions:
            self.assertTrue((out / f"{cond}.json").exists())
        self.assertEqual(json.loads((out / "summary.json").read_text())["decoder_graphs_mode"], "no_while_loops")
        self.assertEqual(list(summary["conditions"]), self.conditions)
        self.assertEqual(summary["build"]["conditions"], self.conditions)
        self.assertAlmostEqual(summary["conditions"]["eng10"]["mean_wer"], 0.25)  # the stub: 1 error in 4 words
        self.assertEqual(summary["model"]["kind"], "binary")

    def test_other_decoder_mode_refused_and_no_summary(self):
        spec = noisy_score.ModelSpec(kind="b0", path=None, name="B0")
        with self.assertRaises(noisy_score.DecoderModeError):
            noisy_score.score_model(spec, self.build, self.root / "scores", hooks=self.hooks("full_graph"), device="cpu")
        self.assertFalse((self.root / "scores" / "B0" / "summary.json").exists())

    def test_manifest_not_matching_the_build_refused(self):
        path = self.build / "manifests" / "radio" / "ami_dev.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
        spec = noisy_score.ModelSpec(kind="b0", path=None, name="B0")
        with self.assertRaises(ValueError):
            noisy_score.score_model(spec, self.build, self.root / "scores", hooks=self.hooks(), device="cpu")

    def test_limit_writes_a_smoke_directory(self):
        spec = noisy_score.ModelSpec(kind="ternary", path=self.root / "export", name="M1")
        noisy_score.score_model(spec, self.build, self.root / "scores", hooks=self.hooks(), device="cpu", limit=2)
        self.assertTrue((self.root / "scores" / "smoke-M1-limit2" / "summary.json").exists())
        self.assertFalse((self.root / "scores" / "M1").exists())

    def test_model_spec_validation(self):
        with self.assertRaises(ValueError):
            noisy_score.ModelSpec(kind="b0", path=Path("/x"), name="B0")
        with self.assertRaises(ValueError):
            noisy_score.ModelSpec(kind="binary", path=None, name="arm")
        with self.assertRaises(ValueError):
            noisy_score.ModelSpec(kind="fp16", path=None, name="x")

    def test_default_hooks_use_the_binary_pilot_pinning(self):
        hooks = noisy_score.default_hooks()
        self.assertEqual(noisy_score.DECODER_GRAPHS_MODE, "no_while_loops")
        self.assertEqual(hooks.pin_decoder.__name__, "pin_decoder_graphs_mode")
        self.assertEqual(hooks.require_decoder.__name__, "require_decoder_graphs_mode")



class SampleProvenanceTest(unittest.TestCase):
    """Review round 3, finding 4: a sample (or otherwise partial) build is scored only into a
    marked smoke directory, carries its provenance into the summary, and is never compared with
    a full build; production names are only for a complete full build."""

    @classmethod
    def setUpClass(cls):
        cls.root = fx.fresh_dir("score-sample")
        cls.full, _ = fx.tiny_build(cls.root, out="full")
        cls.sample, _ = fx.tiny_build(cls.root, out="sample", sample=10)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def hooks(self):
        def decode(model, records, batch_size, device):
            return [rec(r["id"], 1, 4) for r in records]

        return noisy_score.Hooks(gpu_lock=lambda label: contextlib.nullcontext(), load_model=lambda spec, d: object(),
                                 pin_decoder=lambda m, role: "no_while_loops",
                                 require_decoder=lambda m, role: "no_while_loops", decode=decode,
                                 deterministic=lambda: {})

    def score(self, build, name="B0", limit=None):
        spec = noisy_score.ModelSpec(kind="b0", path=None, name=name)
        return noisy_score.score_model(spec, build, self.root / "scores", hooks=self.hooks(), device="cpu",
                                       limit=limit, overwrite=True)

    def test_identity_carries_sample_provenance(self):
        sample = noisy_score.build_identity(self.sample)
        self.assertEqual(sample["sample"], 10)
        self.assertFalse(sample["full"])
        for key in ("skipped_conditions", "pool_missing_sources", "utterances"):
            self.assertIn(key, sample)
        self.assertTrue(noisy_score.build_identity(self.full)["full"])

    def edited(self, source: Path, name: str, edit) -> Path:
        out = self.root / name
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(source, out)
        info = json.loads((out / "build.json").read_text())
        edit(info)
        (out / "build.json").write_text(json.dumps(info))
        return out

    def test_completeness_comes_from_the_lock_not_the_summary(self):
        # Review round 5, finding 5: missing or contradictory summary fields are refused.
        cases = {
            "sample_fields_removed": (self.sample, lambda i: [i.pop(k) for k in
                                                              ("sample", "skipped_conditions", "pool_missing_sources")]),
            "sample_claimed_full": (self.sample, lambda i: i.update(sample=None)),
            "full_claimed_sample": (self.full, lambda i: i.update(sample=5)),
            "skipped_contradicted": (self.full, lambda i: i.update(skipped_conditions=["rev"])),
            "missing_contradicted": (self.full, lambda i: i.update(pool_missing_sources=["musan"])),
            "utterances_contradicted": (self.full, lambda i: i.update(utterances=3)),
            "conditions_reordered": (self.full, lambda i: i.update(conditions=list(reversed(i["conditions"])))),
        }
        for name, (source, edit) in cases.items():
            build = self.edited(source, name, edit)
            with self.assertRaisesRegex(ValueError, "lock|disagrees|missing", msg=name):
                noisy_score.build_identity(build)

    def test_full_requires_full_dev_coverage(self):
        identity = noisy_score.build_identity(self.full)
        self.assertTrue(identity["full"])
        self.assertEqual(identity["utterances"], sum(len(v) for v in fx.DEV_IDS.values()))
        self.assertEqual(identity["dev_sets"], list(fx.DEV_IDS))
        sample = noisy_score.build_identity(self.sample)
        self.assertFalse(sample["full"])
        self.assertEqual(sample["sample"], 10)

    def test_sample_build_scores_into_a_smoke_directory(self):
        summary = self.score(self.sample)
        self.assertTrue((self.root / "scores" / "smoke-B0-sample10" / "summary.json").exists())
        self.assertFalse((self.root / "scores" / "B0").exists())
        self.assertTrue(summary["smoke"])
        self.assertEqual(summary["build"]["sample"], 10)
        self.assertIsNone(summary["limit"])

    def test_full_build_takes_the_production_name(self):
        summary = self.score(self.full, name="B0full")
        self.assertTrue((self.root / "scores" / "B0full" / "summary.json").exists())
        self.assertFalse(summary["smoke"])

    def test_comparison_refuses_sample_with_full_and_limited_with_unlimited(self):
        full = self.score(self.full, name="Afull")
        sample = self.score(self.sample, name="Asample")
        limited = self.score(self.full, name="Alimited", limit=1)
        models = {"B0": noisy_score.load_scores(full["out_name"], self.root / "scores"),
                  "X": noisy_score.load_scores(sample["out_name"], self.root / "scores")}
        with self.assertRaisesRegex(ValueError, "build"):
            noisy_score.compare(models, noisy_score.build_clusters(self.full), resamples=10,
                                builds={"B0": noisy_score.load_build_identity(full["out_name"], self.root / "scores"),
                                        "X": noisy_score.load_build_identity(sample["out_name"], self.root / "scores")})
        with self.assertRaisesRegex(ValueError, "build"):
            noisy_score.compare(models, noisy_score.build_clusters(self.full), resamples=10,
                                builds={"B0": noisy_score.load_build_identity(full["out_name"], self.root / "scores"),
                                        "X": noisy_score.load_build_identity(limited["out_name"], self.root / "scores")})


if __name__ == "__main__":
    unittest.main()
