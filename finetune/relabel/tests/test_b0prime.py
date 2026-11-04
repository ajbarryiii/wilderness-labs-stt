"""Contract tests (CPU): stage 3 (b0prime plan "Code and review").

- the runs: spans in pools, seeds, the stop at the end of pool 781, B0' fixed as b100;
- the WSD schedule's learning rate at every phase boundary (pilot, trunk, branches), the trainer's linear schedule;
- the preregistered rules: learning-rate selection, the WSD sanity rule, the B0 re-score check and the gate;
- the labels: the label lookup with rejected ids dropped, the cleaning version and a read-time transform in
  labels_sha256, extra clip sources from their own provider (version 2 plans);
- the trainer seam on the fixture plan (CPU loss stand-in): A1 with every parameter trainable, the learning rate per
  planned batch, branch points at exactly B(p) and kept once written, a branch starting at the trunk's peak and
  reaching 0 at its last batch, branch and trunk independence, resume within the trunk and within a branch giving the
  same learning rates and batches, refusals of another configuration;
- the chunk order and readiness, the frozen selection and rule, the sources and resume keys.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

import b0prime_fixtures as fx
import train  # finetune/parakeet-ternary
import train_b0prime as tb
from decoder_fixtures import tiny_model
from training import batchplan as bp
from training import label_index, label_sources, stage3
from training import schedule as wsd
from training import seam as tseam


class FakePlan:
    """Pool starts of a plan whose pools hold 100 + (p % 7) planned batches (irregular, to catch off-by-one)."""

    def __init__(self, n_pools: int = stage3.PLAN_POOLS):
        self.n_pools = n_pools
        self.starts = [0]
        for p in range(n_pools):
            self.starts.append(self.starts[-1] + 100 + p % 7)
        self.n_batches = self.starts[-1]

    def pool_start(self, p: int) -> int:
        return self.starts[p]


class Runs(unittest.TestCase):
    def test_the_nine_runs_and_their_spans(self):
        R = stage3.RUNS
        self.assertEqual(list(R), ["pilot-lr3e-5", "pilot-lr1e-4", "pilot-lr3e-4", "sanity-wsd", "sanity-linear",
                                   "trunk", "b25", "b50", "b100"])
        self.assertEqual([R[n].lr for n in stage3.PILOT], [3e-5, 1e-4, 3e-4])
        for n in stage3.PILOT + stage3.SANITY:
            self.assertEqual((R[n].stop_pool, R[n].length_pools), (78, 78))
        for n in stage3.PILOT + ("sanity-wsd",):
            self.assertEqual((R[n].schedule, R[n].cooldown), ("wsd", (66, 78)))
        self.assertEqual((R["sanity-linear"].schedule, R["sanity-linear"].cooldown), ("linear", None))
        self.assertEqual((R["trunk"].stop_pool, R["trunk"].length_pools, R["trunk"].cooldown), (664, 664, None))
        self.assertEqual(R["trunk"].branch_points, (166, 332, 664))
        for name, (start, end) in (("b25", (166, 195)), ("b50", (332, 391)), ("b100", (664, 782))):
            self.assertEqual((R[name].branch_pool, R[name].cooldown, R[name].stop_pool, R[name].length_pools),
                             (start, (start, end), end, 664))
        self.assertEqual(stage3.B0PRIME, "b100")
        self.assertFalse(R["trunk"].scored)
        self.assertTrue(all(R[n].scored for n in R if n != "trunk"))

    def test_seeds(self):
        R = stage3.RUNS
        self.assertEqual({R[n].seed for n in stage3.PILOT + stage3.MAIN}, {stage3.SEED})
        self.assertEqual({R[n].seed for n in stage3.SANITY}, {stage3.SANITY_SEED})
        self.assertNotEqual(stage3.SANITY_SEED, stage3.SEED)
        self.assertNotEqual(stage3.SANITY_SEED, bp.SEED)

    def test_every_run_stops_at_or_before_the_end_of_pool_781(self):
        plan = FakePlan()
        stops = {n: stage3.stop_batch(r, plan) for n, r in stage3.RUNS.items()}
        self.assertEqual(stops["b100"], plan.n_batches)                  # B(782): the end of pool 781
        self.assertEqual(stops["trunk"], plan.pool_start(664))
        self.assertEqual((stops["b25"], stops["b50"]), (plan.pool_start(195), plan.pool_start(391)))
        self.assertTrue(all(s <= plan.n_batches for s in stops.values()))
        with self.assertRaises(ValueError):
            stage3.stop_batch(stage3.Run("x", "branch", 783, 664, 1, "wsd", (664, 783), branch_pool=664), plan)
        with self.assertRaises(ValueError):                              # a plan shorter than the run
            stage3.stop_batch(stage3.RUNS["b100"], FakePlan(781))

    def test_order_and_through(self):
        self.assertEqual(tb.ORDER[0], ("score", "B0"))
        self.assertEqual([n for _, n in tb.ORDER[1:]], list(stage3.RUNS))
        self.assertEqual(tb.ORDER[:tb.THROUGH["pilot"]][-1], ("train", "sanity-linear"))
        self.assertEqual(tb.ORDER[:tb.THROUGH["main"]], tb.ORDER)


class Schedules(unittest.TestCase):
    plan = FakePlan()
    B = staticmethod(plan.pool_start)

    def test_pilot_wsd_boundaries(self):
        B, peak = self.B, 1e-4
        s = stage3.schedule_of(stage3.RUNS["pilot-lr1e-4"], self.plan, peak)
        w = round(0.02 * B(78))
        self.assertEqual((s.warmup, s.cooldown_start, s.stop_batch), (w, B(66), B(78)))
        self.assertEqual(s.lr(0), 0.0)
        self.assertLess(s.lr(w - 1), peak)
        self.assertEqual(s.lr(w), peak)
        self.assertEqual(s.lr(B(66) - 1), peak)                  # stable through pool 65
        self.assertEqual(s.lr(B(66)), peak)                      # the cooldown starts from the peak at pool 66
        self.assertLess(s.lr(B(66) + 1), peak)
        self.assertEqual(s.lr(B(78) - 1), 0.0)                   # 0 at the last planned batch of pool 77
        with self.assertRaises(ValueError):
            s.lr(B(78))
        with self.assertRaises(ValueError):                      # a pilot run has its own peak
            stage3.schedule_of(stage3.RUNS["pilot-lr1e-4"], self.plan, 3e-4)

    def test_linear_is_the_trainers_schedule_on_the_planned_batch_index(self):
        B, peak = self.B, 3e-4
        lin = stage3.schedule_of(stage3.RUNS["sanity-linear"], self.plan, peak)
        ws = stage3.schedule_of(stage3.RUNS["sanity-wsd"], self.plan, peak)
        self.assertEqual((lin.warmup, lin.stop_batch), (ws.warmup, ws.stop_batch))
        for k in list(range(0, 400)) + list(range(B(78) - 300, B(78))):
            self.assertEqual(lin.lr(k), peak * train.lr_factor(k, lin.warmup, B(78)))
        self.assertEqual(lin.lr(lin.warmup - 1), peak)
        self.assertAlmostEqual(lin.lr(B(78) - 1), peak / (B(78) - lin.warmup))
        steps = [lin.lr(k) - lin.lr(k + 1) for k in range(lin.warmup, B(78) - 1)]
        self.assertAlmostEqual(max(steps), min(steps))
        with self.assertRaises(ValueError):
            lin.lr(B(78))
        self.assertEqual(stage3.schedule_dict(lin)["kind"], "linear")

    def test_trunk_boundaries(self):
        B, peak = self.B, 1e-4
        s = stage3.schedule_of(stage3.RUNS["trunk"], self.plan, peak)
        w = round(0.02 * B(664))
        self.assertEqual((s.warmup, s.cooldown_start, s.stop_batch), (w, None, None))
        self.assertEqual(s.lr(0), 0.0)
        self.assertLess(s.lr(w - 1), peak)
        for k in (w, B(166) - 1, B(166), B(332), B(664) - 1, B(664), B(782) - 1):
            self.assertEqual(s.lr(k), peak)

    def test_branches_start_at_the_trunks_peak_and_reach_zero_at_their_last_batch(self):
        B, peak = self.B, 1e-4
        trunk = stage3.schedule_of(stage3.RUNS["trunk"], self.plan, peak)
        for name, (p, q) in stage3.BRANCHES.items():
            s = stage3.schedule_of(stage3.RUNS[name], self.plan, peak)
            self.assertEqual((s.warmup, s.peak), (trunk.warmup, trunk.peak))
            self.assertEqual((s.cooldown_start, s.last_batch, s.stop_batch), (B(p), B(q) - 1, B(q)), name)
            self.assertEqual(s.lr(B(p)), trunk.lr(B(p)))
            self.assertEqual(s.lr(B(p)), peak)
            self.assertEqual(s.lr(B(p) - 1), trunk.lr(B(p) - 1))         # the trunk's, before the branch point
            self.assertEqual(s.lr(B(q) - 1), 0.0)
            with self.assertRaises(ValueError):
                s.lr(B(q))
            values = [s.lr(k) for k in range(B(p), B(q))]
            diffs = [a - b for a, b in zip(values, values[1:])]
            self.assertAlmostEqual(max(diffs), min(diffs))
            self.assertAlmostEqual(diffs[0], peak / (B(q) - B(p) - 1))
            self.assertEqual(stage3.branch_points(stage3.RUNS["trunk"], self.plan)[p], B(p))

    def test_on_the_real_plan(self):
        from labeling import config as lconfig
        from labeling.__main__ import pinned
        try:
            d = bp.production_dir(lconfig.config_id(pinned()), stage3.PLAN_POOLS)
        except Exception as error:   # noqa: BLE001 - no production roots in this environment
            self.skipTest(f"no production plan: {error}")
        if not (d / "plan.json").exists():
            self.skipTest("the stage 3 plan is not built")
        plan = bp.Plan.load(d)
        self.assertEqual((plan.n_pools, plan.consumed_batches), (782, 100_000))
        self.assertAlmostEqual(stage3.schedule_of(stage3.RUNS["trunk"], plan, 1e-4).warmup, 1_410, delta=60)
        self.assertAlmostEqual(stage3.schedule_of(stage3.RUNS["pilot-lr1e-4"], plan, 1e-4).warmup, 165, delta=10)
        self.assertEqual(stage3.stop_batch(stage3.RUNS["b100"], plan), plan.n_batches)


class Rules(unittest.TestCase):
    def test_learning_rate_selection(self):
        sel = stage3.select_lr({3e-5: 6.00, 1e-4: 5.80, 3e-4: 5.84})
        self.assertEqual(sel["selected"], 1e-4)
        self.assertEqual(stage3.select_lr({3e-5: 5.85, 1e-4: 5.80, 3e-4: 5.81})["selected"], 3e-5)   # near tie
        self.assertEqual(stage3.select_lr({3e-5: 5.86, 1e-4: 5.80, 3e-4: 5.81})["selected"], 1e-4)
        self.assertEqual(stage3.select_lr({3e-5: 5.9, 1e-4: 5.84, 3e-4: 5.79})["selected"], 1e-4)
        with self.assertRaises(ValueError):
            stage3.select_lr({3e-5: 5.0, 1e-4: 5.1})

    def test_wsd_rule(self):
        self.assertFalse(stage3.wsd_rule(0.05, [-0.10, 0.20])["stop"])
        self.assertFalse(stage3.wsd_rule(-0.30, [-0.50, -0.10])["stop"])
        self.assertFalse(stage3.wsd_rule(0.02, [0.0, 0.05])["stop"])        # the lower bound is not above 0
        self.assertTrue(stage3.wsd_rule(0.051, [-0.1, 0.2])["stop"])
        self.assertTrue(stage3.wsd_rule(0.03, [0.001, 0.06])["stop"])
        self.assertTrue(stage3.wsd_rule(0.03, [0.001, 0.06])["reasons"])
        self.assertTrue(stage3.wsd_rule(0.0, [-0.1, 0.1])["use_wsd"])

    def test_rescore_check(self):
        s1 = {"a": 2.00, "b": 5.00}
        self.assertTrue(stage3.rescore_check({"a": 2.05, "b": 4.95}, s1)["pass"])
        self.assertFalse(stage3.rescore_check({"a": 2.06, "b": 5.00}, s1)["pass"])
        with self.assertRaises(ValueError):
            stage3.rescore_check({"a": 2.0}, s1)

    def test_gate(self):
        ok = {"pass": True}
        sets = {"x": 0.1, "y": -0.5}
        self.assertTrue(stage3.gate({"point": 0.20, "ci": [0.01, 0.4]}, sets, ok)["pass"])
        self.assertFalse(stage3.gate({"point": 0.19, "ci": [0.01, 0.4]}, sets, ok)["pass"])
        self.assertFalse(stage3.gate({"point": 0.30, "ci": [0.0, 0.6]}, sets, ok)["pass"])
        self.assertTrue(stage3.gate({"point": 0.30, "ci": [0.1, 0.6]}, {"x": 0.25}, ok)["pass"])
        self.assertFalse(stage3.gate({"point": 0.30, "ci": [0.1, 0.6]}, {"x": 0.26}, ok)["pass"])
        self.assertFalse(stage3.gate({"point": 0.30, "ci": [0.1, 0.6]}, sets, {"pass": False})["pass"])
        g = stage3.gate({"point": 0.30, "ci": [0.1, 0.6]}, sets, ok, {"dev-medical": 0.51, "dev-technical": 0.5})
        self.assertEqual(g["domain_regressions_over_flag"], ["dev-medical"])
        self.assertTrue(g["pass"])                                          # reported, no threshold

    def test_branch_flags(self):
        self.assertEqual(stage3.branch_flags({"b25": 6.0, "b50": 5.79, "b100": 5.9}), ["b50"])
        self.assertEqual(stage3.branch_flags({"b25": 5.8, "b50": 5.85, "b100": 5.9}), [])


class World(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = fx.World()
        cls.h = fx.Harness(cls.world)

    @classmethod
    def tearDownClass(cls):
        cls.world.close()


class Labels(World):
    def test_t_star_for_kept_ids_none_for_rejected_and_repetitions(self):
        lab, w = self.world.labels, self.world.w
        store = lab.real.store
        self.assertEqual(lab.label_set, "T*")
        for i, rec in w.records.items():
            if i == w.hallucinated:
                self.assertIsNone(lab.text(i))                        # repetition v1
            elif rec["disposition"] != "kept":
                self.assertIsNone(lab.text(i))
            else:
                self.assertEqual(lab.text(i), rec["text"])
        self.assertTrue(w.reject)
        self.assertTrue(all(lab.text(i) is None for i in w.reject))
        self.assertIsNotNone(lab.text(w.near_miss))
        self.assertEqual(lab.identity["sha256"], store.identity["sha256"])   # no transform, no extra source
        self.assertIsNotNone(lab.identity["repetition"])
        with self.assertRaises(KeyError):
            lab.text("no-such-id")

    def test_cleaning_version_is_a_recorded_parameter_in_labels_sha256(self):
        from training import cleaning
        with mock.patch.object(cleaning, "production", lambda run, version: cleaning.Cleaning.load(version=version)):
            v1 = self.world.open_labels(cleaning_version=1)
            v2 = self.world.open_labels(cleaning_version=2)
            self.assertEqual(label_sources.cleaning_spec("fixture", 1).identity()["version"], 1)
            self.assertEqual(label_sources.cleaning_spec("fixture", 2).identity()["version"], 2)
        v0 = self.world.labels
        self.assertEqual((v1.identity["cleaning"]["version"], v2.identity["cleaning"]["version"]), (1, 2))
        self.assertIn("layer0", v2.identity["cleaning"])
        self.assertNotIn("cleaning", v0.identity)
        self.assertEqual(len({v0.identity["sha256"], v1.identity["sha256"], v2.identity["sha256"]}), 3)
        self.assertIsNone(label_sources.cleaning_spec("fixture", 0))
        with self.assertRaises(ValueError):
            label_sources.cleaning_spec("fixture", 3)                  # no such version in this checkout
        args = self.h.args(fx.RUNS["pw"], self.h.data(fx.RUNS["pw"], labels=v2), cleaning_version=2)
        self.assertEqual((args.cleaning_version, args.labels_sha256), (2, v2.identity["sha256"]))
        self.assertEqual(args.cleaning, v2.identity["cleaning"])
        self.assertIn("cleaning_version", tb.RESUME_EXTRA)
        a = tb.train_parser().parse_args(["--run", "trunk", "--labels-run", "R"])
        self.assertEqual(a.cleaning_version, 2)                                # b0prime plan v4: version 2
        self.assertEqual(tb.train_parser().parse_args(["--run", "trunk", "--labels-run", "R", "--cleaning-version",
                                                       "1"]).cleaning_version, 1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tb.train_parser().parse_args(["--run", "trunk", "--labels-run", "R", "--cleaning-version", "3"])
        seen = {}
        with mock.patch.object(tb, "source_stores", lambda: []), \
                mock.patch.object(tb, "pinned_plan", lambda stores: (None, "pool", "cid", self.world.plan, [])), \
                mock.patch.object(tb, "reject_list", lambda: "reject"), \
                mock.patch.object(tb, "reading_gate", lambda plan: seen.update(gate=plan)), \
                mock.patch.object(tb, "open_labels",
                                  lambda run, pool, through, cv, extra, reject: seen.update(cv=cv, reject=reject) or v2), \
                mock.patch.object(tb.treplay, "PlannedReplay", lambda *x, **k: self.world.replay):
            tb.offline_data(fx.RUNS["pw"], "R", fx.PEAK, 2)
        self.assertEqual(seen, {"cv": 2, "reject": "reject", "gate": self.world.plan})     # every run passes the gate

    def test_a_transform_applies_to_kept_labels_and_enters_the_identity(self):
        class Upper:
            identity = {"name": "upper", "version": 1}

            def __call__(self, text, item_id):
                return text.upper()

        lab = self.world.open_labels(transform=Upper())
        base = self.world.labels
        kept = [i for i in self.world.w.records if base.text(i) is not None]
        self.assertTrue(all(lab.text(i) == base.text(i).upper() for i in kept))
        self.assertTrue(all(lab.text(i) is None for i in self.world.w.records if base.text(i) is None))
        self.assertEqual(lab.identity["transform"], Upper.identity)
        self.assertNotEqual(lab.identity["sha256"], base.identity["sha256"])
        self.assertEqual(lab.identity["store_sha256"], base.identity["sha256"])

        class Emptying(Upper):
            identity = {"name": "empty"}

            def __call__(self, text, item_id):
                return ""
        bad = self.world.open_labels(transform=Emptying())
        with self.assertRaises(RuntimeError):
            bad.text(kept[0])

    def test_the_read_time_reject_list(self):
        """b0prime plan v4: the ted-test-text containment rule's main-pool rejects (pool/tedreject.py's artifact:
        rejected_ids.txt and reject.json), a hashed id list in the identity."""
        import tempfile
        base = self.world.labels
        kept = [i for i in self.world.w.records if base.text(i) is not None]
        with tempfile.TemporaryDirectory() as d:
            e = label_sources.RejectList.load(fx.write_reject(Path(d) / "empty", []))
            t = label_sources.RejectList.load(fx.write_reject(Path(d) / "two", kept[:2]))
            # a damaged or partial artifact is refused
            for name, kw in (("sha", {"ids_sha256": "0" * 64}), ("partial", {"complete": False}),
                             ("count", {"count": 3}), ("name", {"name": "another-list"})):
                with self.assertRaises(RuntimeError):
                    label_sources.RejectList.load(fx.write_reject(Path(d) / name, kept[:2], **kw))
            with self.assertRaises(RuntimeError):                          # an id listed twice
                label_sources.RejectList.load(fx.write_reject(Path(d) / "twice", [kept[0], kept[0]]))
            le = self.world.open_labels(reject=e)
            lt = self.world.open_labels(reject=t)
        self.assertTrue(all(le.text(i) == base.text(i) for i in self.world.w.records))     # an empty list works
        self.assertEqual([lt.text(i) for i in kept[:2]], [None, None])
        self.assertTrue(all(lt.text(i) == base.text(i) for i in kept[2:]))
        self.assertEqual((e.identity["rejected_ids"], t.identity["rejected_ids"]), (0, 2))
        self.assertEqual(e.identity["name"], "talk-text-reject")
        self.assertEqual(len({base.identity["sha256"], le.identity["sha256"], lt.identity["sha256"]}), 3)
        self.assertEqual(lt.identity["reject"], t.identity)
        self.assertEqual(lt.real.reject_counts(), {"listed": 2, "in_store": 2, "kept_in_store": 2})
        with self.assertRaises(KeyError):
            lt.text("no-such-id")
        with mock.patch.object(tb.llayout, "relabel_root", lambda: Path("/nonexistent-root")), \
                self.assertRaises(tb.NotReady):
            tb.reject_list()
        self.assertEqual(tb.REJECT_LIST, "pool/ted-reject")

    def test_a_layer_1_source_reads_its_labels_through_layer_1_alone(self):
        """b0prime plan v4 "Labels": the read-speech source's sentences go through cleaning layer 1 only (no layer 2
        choices, no layer 0), at the run's cleaning version; the provider keeps the stored label for the replay's
        check and the transform's identity."""
        from training import cleaning
        raw = {"a": "I'm gonna check the html page.", "b": "Plain text here."}
        clips = types.SimpleNamespace(identity="cv", clips=[{"id": k, "label": v} for k, v in raw.items()])
        l1 = label_sources.Layer1(2, "cv27-en")
        p = label_sources.ClipLabels(clips, "cv27-en:", transform=l1)
        spec = cleaning.Cleaning.load(version=2)
        self.assertIsNone(spec.choices)
        for k, v in raw.items():
            self.assertEqual(p.text(f"cv27-en:{k}"), spec.clean(v, k, "cv27-en"))
            self.assertEqual(p.raw(f"cv27-en:{k}"), v)
        self.assertNotEqual(p.text("cv27-en:a"), raw["a"])               # layer 1 changed it
        self.assertEqual(p.text("cv27-en:b"), raw["b"])
        self.assertEqual(l1.identity["layers"], [1])
        self.assertEqual(l1.identity["version"], 2)
        self.assertEqual(p.identity["transform"], l1.identity)
        plain = label_sources.ClipLabels(clips, "cv27-en:")
        self.assertNotIn("transform", plain.identity)
        self.assertEqual(plain.text("cv27-en:a"), raw["a"])
        self.assertNotEqual(label_sources.Labels(self.world.labels.real, {"cv27-en:": p}).identity["sha256"],
                            label_sources.Labels(self.world.labels.real, {"cv27-en:": plain}).identity["sha256"])
        self.assertEqual(label_sources.Layer1(1, "cv27-en").identity["version"], 1)
        self.assertIsNone(label_sources.layer1(0, "cv27-en"))                 # version 0: no cleaning at all
        # the production sources: the synthetic store as stored, the read-speech store through layer 1
        self.assertEqual([s[-1] for s in tb.EXTRA_SOURCES], ["as stored", "layer 1"])
        self.assertEqual(tb.EXTRA_SOURCES[1][2], "corpora/cv27-en/store")

    def test_extra_sources_come_from_their_provider(self):
        clips = types.SimpleNamespace(identity="pool-x", clips=[{"id": "c1", "label": "Synthetic one."},
                                                                  {"id": "c2", "label": "Synthetic two."}])
        lab = self.world.open_labels(extra={"syn:": label_sources.ClipLabels(clips)})
        self.assertEqual(lab.text("syn:c2"), "Synthetic two.")
        some = next(i for i in self.world.w.records if self.world.labels.text(i) is not None)
        self.assertEqual(lab.text(some), self.world.labels.text(some))
        self.assertIn("syn:", lab.identity["extra"])
        self.assertNotEqual(lab.identity["sha256"], self.world.labels.identity["sha256"])
        with self.assertRaises(ValueError):
            label_sources.Labels(lab.real, {"syn": label_sources.ClipLabels(clips)})


class LabelIndex(World):
    """b0prime plan v4.1: the label store is written once per label identity (training/label_index.py) and every
    later run start opens it memory-mapped; an index of any other identity is refused."""

    def fresh(self):
        from training import labels as tlabels
        return tlabels.LabelStore.open(self.world.w.labels_dir, self.world.w.pool, label_set="T*",
                                       through_batch=self.world.plan.consumed_batches, cleaning=None,
                                       repetition=fx.repetition_spec(self.world.root / "rep.json"))

    def key(self, cleaning_version=0, reject=None, plan=None, repetition=None):
        return tb.label_key("fixture", self.world.w.labels_dir, self.world.w.pool, plan or self.world.plan,
                            label_sources.cleaning_spec("fixture", cleaning_version) if cleaning_version else None,
                            repetition or fx.repetition_spec(self.world.root / "rep.json"), reject)

    def test_an_index_answers_as_the_store_it_was_written_from(self):
        store = self.world.labels.real.store                          # the fixture opens through the index
        self.assertIsInstance(store, label_index.IndexedLabelStore)
        fresh = self.fresh()
        self.assertEqual(store.identity, json.loads(json.dumps(fresh.identity)))
        self.assertEqual(len(store), len(fresh))
        for i in fresh._texts:
            self.assertIn(i, store)
            self.assertEqual(store.text(i), fresh.text(i))
            self.assertEqual(store.kept(i), fresh.kept(i))
        self.assertEqual(store.repetition_rejected, json.loads(json.dumps(fresh.repetition_rejected)))
        self.assertNotIn("no-such-id", store)
        with self.assertRaises(KeyError):
            store.text("no-such-id")
        # a later open reads the index, not the label shards
        with mock.patch.object(tb.tlabels.LabelStore, "open", side_effect=AssertionError("label shards re-read")):
            again = self.world.open_labels()
        self.assertEqual(again.identity, self.world.labels.identity)
        self.assertEqual(again.real.store.dir, store.dir)
        self.assertTrue(store.dir.is_relative_to(tb.root() / "label-index"))

    def test_every_value_round_trips(self):
        import tempfile
        from training import labels as tlabels
        texts = {"a": "x", "b": None, "c": "", "d": "naïve ü — “quoted”", "e": "two  spaces\tand tab"}
        store = tlabels.LabelStore(texts, {"label_set": "T*", "sha256": "s"}, {"b": [40, 2]})
        key, sha = label_index.key_of({"what": "fixture"})
        with tempfile.TemporaryDirectory() as d:
            got = label_index.load(label_index.write(Path(d) / "i", store, key, sha), key, sha)
            self.assertEqual({i: got.text(i) for i in texts}, texts)
            self.assertEqual((got.kept("b"), got.kept("c")), (False, True))
            self.assertEqual(got.repetition_rejected, {"b": [40, 2]})
            empty = tlabels.LabelStore({"z": None}, {"label_set": "T*", "sha256": "t"})
            self.assertIsNone(label_index.load(label_index.write(Path(d) / "j", empty, key, sha), key, sha).text("z"))

    def test_an_index_of_another_identity_is_refused(self):
        import tempfile

        from training import cleaning
        key, sha = self.key()
        directory = self.world.labels.real.store.dir
        self.assertEqual(directory, label_index.index_dir(tb.root() / "label-index", sha))
        other_plan = bp.build(self.world.w.pool, self.world.root / "plans" / "other-seed", fx.N_POOLS,
                              pool_batches=fx.P, budget=fx.BUDGET, seed=fx.SEED + 1)
        with tempfile.TemporaryDirectory() as d:
            reject = label_sources.RejectList.load(fx.write_reject(Path(d) / "r", []))
            with mock.patch.object(cleaning, "production",
                                   lambda run, version: cleaning.Cleaning.load(version=version)):
                variants = {"cleaning version": self.key(cleaning_version=2), "reject list": self.key(reject=reject),
                            "repetition rule": self.key(repetition=types.SimpleNamespace(
                                bind_run=lambda c: None, rule=lambda: {"name": "repetition", "threshold": 99})),
                            "plan": self.key(plan=other_plan)}
        for what, (k2, s2) in variants.items():
            self.assertNotEqual(s2, sha, what)
            with self.assertRaises(label_index.IndexMismatch, msg=what):
                label_index.load(directory, k2, s2)
            self.assertNotEqual(label_index.index_dir(tb.root() / "label-index", s2), directory, what)
        # a damaged index is refused too
        with tempfile.TemporaryDirectory() as d:
            import shutil
            copy = Path(d) / "copy"
            shutil.copytree(directory, copy)
            label_index.load(copy, key, sha)
            data = bytearray((copy / "texts.bin").read_bytes())
            data[0] ^= 1
            (copy / "texts.bin").write_bytes(bytes(data))
            with self.assertRaises(label_index.IndexMismatch):
                label_index.load(copy, key, sha)

    def test_a_version_2_plan_shares_the_index_of_its_real_plan(self):
        from test_synthetic_mixing import make_pool
        clips = make_pool(self.world.root / "syn-index", n=20)
        plan2 = bp.build(self.world.w.pool, self.world.root / "plans" / "syn-index", fx.N_POOLS, clips=clips,
                         pool_batches=fx.P, budget=fx.BUDGET, seed=fx.SEED, syn_seed=11, share=0.1)
        self.assertEqual(tb.real_plan_sha256(plan2), self.world.plan.sha256)
        self.assertEqual(self.key(plan=plan2), self.key())


class PlanExtremes(World):
    """b0prime plan v4.1 "Memory probe": the real plan's 10 most extreme batches by rows x longest clip (the plan's
    key: a synthetic row's longest augmented length), the fused seam's padded lattice (rows x longest encoder frames x
    (longest label in tokens + 1)) and rows x longest label, over the rows that keep a label."""

    @staticmethod
    def keys(plan, t):
        """The plan's cut keys of training position t's rows (samples.u32 indexed by entry index, in cut order)."""
        b = int(plan.shuffle[t])
        return [int(plan.samples[e]) for e in plan.order[int(plan.offsets[b]):int(plan.offsets[b + 1])]]

    def brute(self, plan, replay, labels, tokens):
        out = {}
        for t in range(plan.n_batches):
            ids = replay.batch_ids(t)
            rows = [(k, i) for k, i in zip(self.keys(plan, t), ids) if labels.text(i) is not None]
            if not rows:
                continue
            longest = max(k for k, _ in rows)
            label = max(tokens([labels.text(i)])[0] for _, i in rows)
            out[t] = {"rows_x_longest_clip": len(rows) * longest,
                      "padded_lattice": len(rows) * tb.encoder_frames(longest) * (label + 1),
                      "rows_x_longest_label": len(rows) * label}
        return out

    def check(self, plan, replay, labels, k):
        tokens = lambda texts: [len(x.split()) for x in texts]                    # noqa: E731
        got = tb.plan_extremes(plan, replay, labels, tokens, k=k)
        want = self.brute(plan, replay, labels, tokens)
        for metric in tb.EXTREME_METRICS:
            order = sorted(want, key=lambda t: (-want[t][metric], t))[:k]
            self.assertEqual(got["top"][metric], order, metric)
            for t in order:
                self.assertEqual(got["values"][t][metric], want[t][metric])
        self.assertEqual(got["batches"], sorted({t for m in tb.EXTREME_METRICS for t in got["top"][m]}))
        self.assertIn("stress coverage", got["note"])
        return got

    def test_the_rankings(self):
        self.check(self.world.plan, self.world.replay, self.world.labels, 3)

    def test_a_synthetic_row_counts_at_its_longest_augmented_length(self):
        from test_synthetic_mixing import make_pool
        from training import replay as treplay
        clips = make_pool(self.world.root / "syn-extremes", n=40)
        plan2 = bp.build(self.world.w.pool, self.world.root / "plans" / "syn-extremes", fx.N_POOLS, clips=clips,
                         pool_batches=fx.P, budget=fx.BUDGET, seed=fx.SEED, syn_seed=11, share=0.3)
        replay = treplay.PlannedReplay(plan2, self.world.w.pool, audio="materialized", store=self.world.store,
                                       synthetic=clips)
        self.addCleanup(replay.close)
        lab = self.world.open_labels(extra={"syn:": label_sources.ClipLabels(clips)}, plan=plan2)
        syn = [(k, i) for t in range(plan2.n_batches)
               for k, i in zip(self.keys(plan2, t), replay.batch_ids(t)) if i.startswith("syn:")]
        by_id = {f"syn:{c['id']}": c for c in clips.clips}
        self.assertTrue(syn)
        self.assertTrue(all(k == math.ceil(round(by_id[i]["duration"] * 16000) * 10 / 9) for k, i in syn))
        self.check(plan2, replay, lab, 4)

    def test_encoder_frames_is_nemos(self):
        """NeMo's own length functions with Parakeet TDT 0.6B v2's preprocessor (n_fft 512, hop 160, centered STFT)
        and its 8x dw-striding subsampling (three stride-2 convolutions, kernel 3, padding 1 on each side)."""
        import torch
        from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures
        from nemo.collections.asr.parts.submodules.subsampling import calc_length
        pre = types.SimpleNamespace(stft_pad_amount=None, n_fft=512, hop_length=160)
        for samples in (160, 16000, 16001, 20480, 123457, 480000):
            mel = FilterbankFeatures.get_seq_len(pre, torch.tensor([samples]))
            nemo = calc_length(mel.float(), all_paddings=2, kernel_size=3, stride=2, ceil_mode=False, repeat_num=3)
            self.assertEqual(tb.encoder_frames(samples), int(nemo.item()), samples)
        self.assertEqual((tb.encoder_frames(480000), tb.encoder_frames(20480)), (375, 16))


class Serving(World):
    def test_the_batcher_serves_the_planned_batches_with_their_labels(self):
        plan, w, lab = self.world.plan, self.world.w, self.world.labels
        seq = [r["id"] for r in w.journal]
        batcher = label_sources.Batcher(self.world.replay, lab, start=0, stop=plan.n_batches)
        try:
            for t in range(plan.n_batches):
                b = batcher.next(timeout=60)
                want = [seq[e] for e in plan.batch(t).tolist()]
                self.assertEqual((b["index"], b["ids"]), (t, want))
                self.assertEqual(b["labels"], [lab.text(i) for i in want])
        finally:
            batcher.close()

    def test_rejected_rows_are_dropped_after_the_cut_and_a_fully_rejected_batch_takes_no_step(self):
        data = self.h.data(fx.RUNS["pw"])
        mixed = [t for t in range(data.stop) if any(x is None for x in map(self.world.labels.text,
                                                                            self.world.replay.batch_ids(t)))]
        self.assertTrue(mixed)
        ctx = types.SimpleNamespace(state=types.SimpleNamespace(batches_read=0, acc={"dropped_rows": 0},
                                                                drop_counts={}, seconds_by_source={}),
                                    optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0),
                                    stream=data, device="cpu", student=tiny_model())
        batcher = data.batcher({"batch": mixed[0]})
        try:
            b = batcher.next(timeout=60)
        finally:
            batcher.close()
        ctx.state.batches_read = mixed[0] + 1
        out = tseam.label_batch(ctx, b)
        kept = [t is not None for t in b["labels"]]
        if any(kept):
            self.assertEqual(out["rows"], sum(kept))
            self.assertEqual(out["audio"].shape[0], sum(kept))
        else:
            self.assertIsNone(out)
        self.assertEqual(ctx.state.drop_counts["rejected"], kept.count(False))
        self.assertEqual(data.plan["steps"], data.stop - len(data.plan["empty_batches"]))

    def test_a_version_2_plan_serves_extra_rows_with_their_providers_labels(self):
        from test_synthetic_mixing import make_pool
        from training import replay as treplay
        clips = make_pool(self.world.root / "syn", n=40)
        plan2 = bp.build(self.world.w.pool, self.world.root / "plans" / "syn", fx.N_POOLS, clips=clips,
                         pool_batches=fx.P, budget=fx.BUDGET, seed=fx.SEED, syn_seed=11, share=0.1)
        replay = treplay.PlannedReplay(plan2, self.world.w.pool, audio="materialized", store=self.world.store,
                                       synthetic=clips)
        class Shout:
            identity = {"name": "shout"}

            def __call__(self, text, item_id):
                return text.upper()
        lab = self.world.open_labels(extra={"syn:": label_sources.ClipLabels(clips, transform=Shout())})
        data = tb.data_for(fx.RUNS["pw"], plan2, replay, lab, fx.PEAK)
        self.assertGreater(data.plan["synthetic_rows"], 0)
        by_id = {f"syn:{c['id']}": c["label"].upper() for c in clips.clips}      # the provider's text, transformed
        batcher = data.batcher(None)
        seen = 0
        try:
            for t in range(data.stop):
                b = batcher.next(timeout=60)
                for i, text in zip(b["ids"], b["labels"]):
                    if i.startswith("syn:"):
                        self.assertEqual(text, by_id[i])
                        seen += 1
                    else:
                        self.assertEqual(text, self.world.labels.text(i))
        finally:
            batcher.close()
            replay.close()
        self.assertEqual(seen, data.plan["synthetic_rows"])
        wrong = label_sources.Labels(lab.real, {"syn:": types.SimpleNamespace(
            text=lambda i: "another label", raw=lambda i: "another label", identity={"kind": "wrong"})})
        bad = label_sources.Batcher(replay, wrong, start=0, stop=data.stop)
        try:
            with self.assertRaises(RuntimeError):
                for _ in range(data.stop):
                    bad.next(timeout=60)
        finally:
            bad.close()


class Model(unittest.TestCase):
    def test_a1_trains_every_parameter_with_pred_joint_dropout_zero(self):
        m = tiny_model()
        info = tb.configure(m)
        self.assertEqual(info["frozen"], 0)
        self.assertEqual(info["trainable"], sum(p.numel() for p in m.parameters()))
        self.assertTrue(all(p.requires_grad for p in m.parameters()))
        self.assertEqual(info["verified"]["recipe_check"]["pred_joint_dropout"], [0.0])
        self.assertTrue(all(mod.training for mod in m.modules()))
        self.assertIsNotNone(m.spec_augmentation)
        opt, sched = tseam.make_optimizer(m, 1e-4, 10)
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        self.assertEqual(in_opt, {id(p) for p in m.parameters()})
        self.assertEqual((opt.defaults["betas"], opt.param_groups[0]["weight_decay"], opt.param_groups[1]["weight_decay"]),
                         ((0.9, 0.98), 0.01, 0.0))
        self.assertEqual(tb.recipe_of(types.SimpleNamespace(recipe=tb.RECIPE_NAME)),
                         {"encoder_matching": False, "train_pred_joint": True, "quantized": False})

    def test_verify_refuses_a_frozen_parameter_or_eval_mode(self):
        m = tiny_model()
        tb.configure(m)
        m.encoder.layers[0].requires_grad_(False)
        with self.assertRaises(RuntimeError):
            tb.verify(m)
        m = tiny_model()
        tb.configure(m)
        m.decoder.eval()
        with self.assertRaises(RuntimeError):
            tb.verify(m)


class Loop(World):
    """train.train_loop through the stage 3 seam on the fixture plan, CPU (fx.FakeLosses)."""

    def setUp(self):
        import shutil
        shutil.rmtree(tb.root(), ignore_errors=True)

    def B(self, p):
        return self.world.plan.pool_start(p)

    def trunk(self, stop_after=None):
        ctx = self.h.context("trunk")
        seen = self.h.loop(ctx, fx.FakeLosses(), stop_after=stop_after)
        return ctx, seen

    def finish_trunk(self, ctx):
        with tb.seam():
            return tb.finish(ctx, "config-sha")

    def test_the_trunk_writes_its_branch_points_at_exactly_b_p(self):
        ctx, seen = self.trunk()
        self.assertEqual(ctx.state.step, ctx.args.max_steps)
        self.assertEqual(ctx.state.batches_read, self.B(3))
        self.assertEqual({s["index"] for s in seen} | set(ctx.stream.plan["empty_batches"]), set(range(self.B(3))))
        self.assertTrue(all(s["lrs"] == {ctx.stream.schedule.lr(s["index"])} for s in seen))
        m1 = tb.read_branch_point(ctx.run, 1)
        self.assertEqual((m1["planned_batch"], m1["position"], m1["batches_read"]), (self.B(1), {"batch": self.B(1)},
                                                                                     self.B(1)))
        self.assertEqual(m1["step"], sum(1 for s in seen if s["index"] < self.B(1)))
        payload = torch.load(tb.branch_point_files(ctx.run, 1)[0], map_location="cpu", weights_only=False)
        self.assertEqual((payload["state"]["position"], payload["state"]["batches_read"], payload["step"]),
                         ({"batch": self.B(1)}, self.B(1), m1["step"]))
        self.assertEqual(set(payload["student"]), set(ctx.student.state_dict()))
        self.assertEqual(len(payload["optimizer"]["state"]), sum(1 for _ in ctx.student.parameters()))
        self.assertIn("rng", payload)
        with self.assertRaises(tb.NotReady):
            tb.read_branch_point(ctx.run, 3)                      # the last one is the trunk's final checkpoint
        result = self.finish_trunk(ctx)
        m3 = tb.read_branch_point(ctx.run, 3)
        self.assertEqual((m3["planned_batch"], m3["position"], m3["step"]), (self.B(3), {"batch": self.B(3)},
                                                                             ctx.args.max_steps))
        self.assertEqual(m3["sha256"], tb._sha256_file(train.ckpt_paths(ctx.run)[0]))
        self.assertTrue(math.isnan(result["dev_mean_wer"]))
        self.assertFalse(result["b0prime"]["scored"])

    def test_a_branch_starts_at_the_trunks_peak_and_cools_down_to_zero_independently(self):
        ctx, _ = self.trunk()
        self.finish_trunk(ctx)
        before = fx.tree_hashes(tb.run_dir("trunk"))
        for name, p, q in (("ba", 1, 2), ("bb", 3, 4)):
            b = self.h.context(name)
            m = tb.read_branch_point(tb.run_dir("trunk"), p)
            self.assertEqual(b.args.branch_of["sha256"], m["sha256"])
            self.assertEqual((b.state.step, b.state.position), (m["step"], {"batch": self.B(p)}))
            seen = self.h.loop(b, fx.FakeLosses())
            self.assertEqual(b.state.step, b.args.max_steps)
            self.assertEqual(b.state.batches_read, self.B(q))
            self.assertEqual([s["index"] for s in seen],
                             [k for k in range(self.B(p), self.B(q)) if k not in b.stream.plan["empty_batches"]])
            self.assertEqual(seen[0]["lr"], fx.PEAK)
            if seen[-1]["index"] == self.B(q) - 1:
                self.assertEqual(seen[-1]["lr"], 0.0)
            self.assertTrue(all(s["lrs"] == {b.stream.schedule.lr(s["index"])} for s in seen))
            self.assertEqual(fx.tree_hashes(tb.run_dir("trunk")), before)   # the trunk is never written by a branch
            self.assertTrue((tb.run_dir(name) / "ckpt" / "latest.pt").exists())

    def test_the_branch_continues_the_trunks_state(self):
        ctx, _ = self.trunk()
        self.finish_trunk(ctx)
        payload = torch.load(tb.branch_point_files(ctx.run, 1)[0], map_location="cpu", weights_only=False)
        b = self.h.context("ba")
        for k, v in b.student.state_dict().items():
            self.assertTrue(torch.equal(v, payload["student"][k]), k)
        self.assertEqual(train.rng_state(b.student)["python"], payload["rng"]["python"])
        opt = b.optimizer.state_dict()
        self.assertEqual(len(opt["state"]), len(payload["optimizer"]["state"]))
        self.assertTrue(torch.equal(opt["state"][0]["exp_avg"], payload["optimizer"]["state"][0]["exp_avg"]))
        self.assertEqual((b.state.session, b.state.metrics_offset), (0, 0))

    def run_resumed(self, name, stops, build=None):
        """A run interrupted after each step count in `stops` (a checkpoint, then a new context restored from it)."""
        build = build or (lambda: self.h.context(name))
        ctx, seen = build(), []
        for stop in stops:
            seen += self.h.loop(ctx, fx.FakeLosses(), stop_after=stop)
            ctx = self.h.restore(build())
        seen += self.h.loop(ctx, fx.FakeLosses())
        return ctx, seen

    def test_resume_within_the_trunk_and_within_a_branch(self):
        full_trunk, full_seen = self.trunk()
        self.finish_trunk(full_trunk)
        full_b = self.h.context("ba")
        full_bseen = self.h.loop(full_b, fx.FakeLosses())
        reference = {"trunk": ([(s["index"], s["lr"]) for s in full_seen], full_trunk.student.state_dict()),
                     "ba": ([(s["index"], s["lr"]) for s in full_bseen], full_b.student.state_dict())}
        import shutil
        shutil.rmtree(tb.root())
        ctx, seen = self.run_resumed("trunk", [1, 3])
        self.finish_trunk(ctx)
        self.assertEqual([(s["index"], s["lr"]) for s in seen], reference["trunk"][0])
        for k, v in ctx.student.state_dict().items():
            self.assertTrue(torch.equal(v, reference["trunk"][1][k]), k)
        start = tb.read_branch_point(ctx.run, 1)["step"]
        b, bseen = self.run_resumed("ba", [start + 1])
        self.assertEqual([(s["index"], s["lr"]) for s in bseen], reference["ba"][0])
        for k, v in b.student.state_dict().items():
            self.assertTrue(torch.equal(v, reference["ba"][1][k]), k)

    def test_a_written_branch_point_is_kept_when_the_trunk_reaches_it_again(self):
        ctx = self.h.context("trunk")
        before_b1 = sum(1 for k in range(self.B(1)) if k not in ctx.stream.plan["empty_batches"])
        self.h.loop(ctx, fx.FakeLosses(), stop_after=before_b1 - 1)        # checkpoint A, before pool 1
        ctx = self.h.restore(self.h.context("trunk"))
        self.h.loop(ctx, fx.FakeLosses(), stop_after=before_b1 + 2)        # checkpoint B, after it
        pt, marker = tb.branch_point_files(ctx.run, 1)
        written = (marker.read_text(), pt.stat().st_mtime_ns, pt.stat().st_ino)
        latest, previous, _ = train.ckpt_paths(ctx.run)
        latest.unlink()                                                   # back to checkpoint A
        ctx = self.h.restore(self.h.context("trunk"))
        self.assertEqual(ctx.state.position, {"batch": torch.load(previous, map_location="cpu", weights_only=False)
                                              ["state"]["position"]["batch"]})
        self.h.loop(ctx, fx.FakeLosses())
        self.assertEqual((marker.read_text(), pt.stat().st_mtime_ns, pt.stat().st_ino), written)

    def test_another_configuration_is_refused_on_resume(self):
        ctx = self.h.context("trunk")
        self.h.loop(ctx, fx.FakeLosses(), stop_after=2)
        self.assertEqual(ctx.args.cleaning_version, 0)
        for change in ({"plan_sha256": "other"}, {"labels_sha256": "other"}, {"cleaning_version": 1},
                       {"schedule": {"kind": "wsd", "peak": 1.0}}, {"seed": 1}, {"branch_of": {"sha256": "x"}}):
            other = self.h.context("trunk", **change)
            out = io.StringIO()
            with tb.seam(), contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as stop:
                train.load_resume_payload(other.run, other.args)
            self.assertEqual(stop.exception.code, train.EXIT_RESUME_MISMATCH, change)
        same = self.h.context("trunk", budget_minutes=55.0, deadline=123.0)
        with tb.seam():
            self.assertEqual(train.load_resume_payload(same.run, same.args)["step"], 2)

    def test_a_branch_refuses_a_trunk_of_other_labels_or_peak(self):
        ctx, _ = self.trunk()
        self.finish_trunk(ctx)
        data = self.h.data(fx.RUNS["ba"], peak=1e-3)
        with self.assertRaises(RuntimeError):
            self.h.args(fx.RUNS["ba"], data, peak=1e-3)
        class Same:
            identity = {"name": "identity transform"}

            def __call__(self, text, item_id):
                return text

        data = self.h.data(fx.RUNS["ba"], labels=self.world.open_labels(transform=Same()))
        with self.assertRaises(RuntimeError):
            self.h.args(fx.RUNS["ba"], data)

    def test_a_selected_run_trains_on_the_pilots_identity(self):
        run, data = fx.RUNS["pw"], self.h.data(fx.RUNS["pw"])
        a = types.SimpleNamespace(labels_run="fixture", ckpt_minutes=30.0, overwrite=False, budget_minutes=None,
                                  deadline=None, cleaning_version=0)
        ok = tb.identity_of(vars(self.h.args(run, data)))
        tb.namespace(a, run, data, fx.PEAK, {"selected": fx.PEAK, "identity": ok}, False, "b0")
        with self.assertRaises(RuntimeError):
            tb.namespace(a, run, data, fx.PEAK, {"selected": fx.PEAK, "identity": {**ok, "labels_sha256": "x"}},
                         False, "b0")

    def test_a_branch_waits_for_its_branch_point(self):
        data = self.h.data(fx.RUNS["ba"])
        with self.assertRaises(tb.NotReady):
            self.h.args(fx.RUNS["ba"], data)

    def test_the_trunk_alone_runs_dev_subset_evaluations(self):
        ctx, _ = self.trunk()
        self.assertTrue(list((ctx.run / "dev-subset").rglob("step-*.json")))
        self.assertTrue(ctx.state.final_subset)
        self.finish_trunk(ctx)
        b = self.h.context("ba")
        self.assertIsNone(b.evaluator)
        self.assertEqual(b.args.eval_every, b.args.max_steps)
        self.assertEqual(b.state.final_subset, {})


class Freezing(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [mock.patch.object(tb, "root", lambda: Path(self.tmp.name)),
                        mock.patch.object(tb.llayout, "require_mount", lambda: None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def scored_run(self, name, wers: dict[str, float], records=None):
        d = tb.run_dir(name)
        for n in tb.DEV_SETS:
            tb.write_json(d / "eval-dev" / f"{n}.json", {"wer": {"wer": wers.get(n, 5.0) / 100},
                                                        "records": (records or {}).get(n, [])})
        tb.write_json(d / "summary.json", {"run": name})

    def test_selection_waits_for_the_three_pilot_runs_and_is_frozen(self):
        with self.assertRaises(tb.NotReady):
            tb.frozen_selection()
        for name, x in zip(stage3.PILOT, (6.0, 5.8, 5.84)):
            self.scored_run(name, {n: x for n in tb.H4})
        sel = tb.frozen_selection()
        self.assertEqual(sel["selected"], 1e-4)
        self.assertEqual(tb.peak_of(stage3.RUNS["sanity-wsd"])[0], 1e-4)
        self.assertEqual(tb.peak_of(stage3.RUNS["pilot-lr3e-4"]), (3e-4, None))
        self.scored_run("pilot-lr3e-5", {n: 5.0 for n in tb.H4})            # a changed run: refused
        with self.assertRaises(RuntimeError):
            tb.frozen_selection()

    def test_the_pilots_must_share_one_label_and_plan_identity(self):
        for name, x in zip(stage3.PILOT, (6.0, 5.8, 5.84)):
            self.scored_run(name, {n: x for n in tb.H4})
            tb.write_json(tb.run_dir(name) / "config.json",
                          {"args": {"labels_run": "R", "cleaning_version": 2, "labels_sha256": "L", "plan_sha256": "P"}})
        self.assertEqual(tb.frozen_selection()["identity"]["labels_sha256"], "L")
        tb.selection_path().unlink()
        tb.write_json(tb.run_dir(stage3.PILOT[1]) / "config.json",
                      {"args": {"labels_run": "R", "cleaning_version": 1, "labels_sha256": "L1", "plan_sha256": "P"}})
        with self.assertRaises(RuntimeError):
            tb.frozen_selection()

    def records(self, errors: int) -> dict[str, list]:
        out = {}
        for n in tb.H4:
            ids = {"librispeech_dev_clean": [f"{s}-1-{u}" for s in range(1, 9) for u in range(3)],
                   "librispeech_dev_other": [f"{s}-2-{u}" for s in range(1, 9) for u in range(3)],
                   "ami_dev": [f"ami:AMI_M{s}_H0{u}_x" for s in range(1, 9) for u in range(3)],
                   "voxpopuli_dev": [f"voxpopuli:2009010{s}-0900-PLENARY-{u}" for s in range(1, 9)
                                     for u in range(3)]}[n]
            out[n] = [{"id": i, "ref_norm": "a b c d e f g h i j", "S": errors if k % 2 else 0, "D": 0, "I": 0,
                       "scored": True} for k, i in enumerate(ids)]
        return out

    def test_the_wsd_rule_is_computed_once_and_can_stop_the_main_run(self):
        for name, x in zip(stage3.PILOT, (6.0, 5.8, 5.84)):
            self.scored_run(name, {n: x for n in tb.H4})
        with self.assertRaises(tb.NotReady):
            tb.peak_of(stage3.RUNS["trunk"])
        self.scored_run("sanity-wsd", {}, self.records(3))
        self.scored_run("sanity-linear", {}, self.records(1))
        with self.assertRaises(tb.Stopped):
            tb.peak_of(stage3.RUNS["trunk"])
        rule = json.loads(tb.wsd_rule_path().read_text())
        self.assertTrue(rule["stop"])
        self.assertGreater(rule["d"], 0.05)
        tb.wsd_rule_path().unlink()
        self.scored_run("sanity-wsd", {}, self.records(1))
        peak, record = tb.peak_of(stage3.RUNS["b25"])
        self.assertEqual((peak, record["wsd_rule"]["use_wsd"]), (1e-4, True))
        self.assertEqual(record["wsd_rule"]["d"], 0.0)


class Chunk(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [mock.patch.object(tb, "root", lambda: Path(self.tmp.name)),
                        mock.patch.object(tb.llayout, "require_mount", lambda: None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def finish(self, item):
        kind, name = item
        if kind == "score":
            tb.write_json(tb.b0_dir() / "scores.json", {"complete": True})
            tb.write_json(tb.b0_dir() / "crosscheck.json", {"pass": True})
        else:
            tb.write_json(tb.run_dir(name) / "summary.json", {})

    def test_items_in_order_up_to_through(self):
        calls = []

        def runner(cmd, cwd=None):
            calls.append(cmd)
            self.finish(tb.ORDER[len(calls) - 1])
            return types.SimpleNamespace(returncode=0)

        self.assertEqual(tb.chunk("RUN", 600.0, "pilot", runner=runner), 0)
        self.assertEqual(len(calls), tb.THROUGH["pilot"])
        self.assertEqual(calls[0][2:3], ["score-b0"])
        self.assertEqual(calls[1][2:7], ["train", "--run", "pilot-lr3e-5", "--labels-run", "RUN"])
        self.assertEqual(calls[-1][4], "sanity-linear")
        self.assertEqual(tb.chunk("RUN", 600.0, "main", runner=runner), 0)
        self.assertEqual([c[4] for c in calls[tb.THROUGH["pilot"]:]], ["trunk", "b25", "b50", "b100"])
        self.assertTrue(all(c[-2:] == ["--cleaning-version", "2"] for c in calls[1:]))

    def test_the_cleaning_version_reaches_every_train_item(self):
        calls = []
        self.finish(tb.ORDER[0])
        tb.chunk("RUN", 600.0, "pilot", cleaning_version=0,
                 runner=lambda c, cwd=None: calls.append(c) or types.SimpleNamespace(returncode=0))
        self.assertEqual(calls[0][-2:], ["--cleaning-version", "0"])

    def test_a_failing_or_stopped_item_fails_the_chunk(self):
        self.assertEqual(tb.chunk("RUN", 600.0, "main",
                                  runner=lambda c, cwd=None: types.SimpleNamespace(returncode=tb.EXIT_STOPPED)),
                         tb.EXIT_STOPPED)

    def test_an_unfinished_item_ends_the_chunk(self):
        calls = []
        tb.chunk("RUN", 600.0, "main", runner=lambda c, cwd=None: calls.append(c) or types.SimpleNamespace(returncode=0))
        self.assertEqual(len(calls), 1)

    def test_finished_runs_of_other_labels_are_refused(self):
        """A finished run of another labels run or cleaning version is never skipped as done: the chunk runs nothing
        and run_train refuses (EXIT_RESUME_MISMATCH)."""
        self.finish(tb.ORDER[0])
        self.finish(tb.ORDER[1])
        tb.write_json(tb.run_dir(tb.ORDER[1][1]) / "config.json", {"args": {"labels_run": "RUN", "cleaning_version": 1}})
        calls = []
        rc = tb.chunk("RUN", 600.0, "pilot", runner=lambda c, cwd=None: calls.append(c) or types.SimpleNamespace(returncode=0))
        self.assertEqual((rc, calls), (train.EXIT_RESUME_MISMATCH, []))
        rc = tb.chunk("RUN", 600.0, "pilot", cleaning_version=1,
                      runner=lambda c, cwd=None: calls.append(c) or types.SimpleNamespace(returncode=0))
        self.assertEqual(rc, 0)
        self.assertEqual(tb.run_train(["--run", tb.ORDER[1][1], "--labels-run", "RUN"]), train.EXIT_RESUME_MISMATCH)
        self.assertEqual(tb.run_train(["--run", tb.ORDER[1][1], "--labels-run", "RUN", "--cleaning-version", "1"]), 0)
        self.assertEqual(tb.run_train(["--run", tb.ORDER[1][1], "--labels-run", "OTHER", "--cleaning-version", "1"]),
                         train.EXIT_RESUME_MISMATCH)

    def test_b0_counts_as_done_only_with_a_passing_check(self):
        tb.write_json(tb.b0_dir() / "scores.json", {"complete": True})
        tb.write_json(tb.b0_dir() / "crosscheck.json", {"pass": False})
        self.assertFalse(tb.item_done(("score", "B0")))

    def test_no_item_without_budget(self):
        calls = []
        tb.chunk("RUN", tb.MIN_CHUNK_MINUTES - 1, "main", runner=lambda c, cwd=None: calls.append(c))
        self.assertEqual(calls, [])


class SeamAndSources(unittest.TestCase):
    def test_sources_exist_and_resume_keys(self):
        base = Path(tb.__file__).resolve().parent.parent
        for name in tb.SOURCES:
            self.assertTrue((base / name).exists(), name)
        for key in ("labels_sha256", "cleaning_version", "label_transform", "extra_sources", "plan_sha256",
                    "replay_sha256", "stop_batch", "schedule", "spec", "branch_of", "lr_selection", "base_sha256"):
            self.assertIn(key, tb.RESUME_EXTRA)

    def test_the_seam_installs_and_restores(self):
        import paths
        saved = {n: getattr(train, n) for n in ("parse_args", "build_context", "finish", "run_eval", "train_step",
                                                "label_batch", "make_optimizer", "Batcher", "RESUME_KEYS")}
        runs = paths.RUNS
        with tb.seam():
            self.assertIs(train.parse_args, tb.parse_args)
            self.assertIs(train.train_step, tseam.guarded_train_step)
            self.assertIs(train.label_batch, tseam.label_batch)
            self.assertEqual(paths.RUNS, tb.root() / "runs")
            self.assertTrue(set(tb.RESUME_EXTRA) <= set(train.RESUME_KEYS))
            self.assertIn("branch-points", train.OUTPUTS)
        self.assertEqual({n: getattr(train, n) for n in saved}, saved)
        self.assertEqual(paths.RUNS, runs)

    def test_a_read_speech_clip_source_must_be_complete(self):
        import hashlib
        import tempfile

        from synthetic import mixing
        with tempfile.TemporaryDirectory() as d:
            data = json.dumps({"id": "a", "audio_filepath": "/x.flac", "duration": 1.0, "samples": 16000,
                               "sha256": "0", "label": "Hi."}) + "\n"
            Path(d, "clips.jsonl").write_text(data)
            sha = hashlib.sha256(data.encode()).hexdigest()
            Path(d, "clips.meta.json").write_text(json.dumps({"clips_sha256": sha, "provisional": False,
                                                              "complete": True}))
            self.assertEqual([c["id"] for c in mixing.ClipPool.load(Path(d), synthetic_hygiene=False).clips], ["a"])
            for meta in ({"clips_sha256": sha, "provisional": False},
                         {"clips_sha256": sha, "provisional": False, "complete": False}):
                Path(d, "clips.meta.json").write_text(json.dumps(meta))
                with self.assertRaises(RuntimeError):
                    mixing.ClipPool.load(Path(d), synthetic_hygiene=False)

    def test_every_run_mode_trains_through_the_fused_loss_seam(self):
        """b0prime plan v4.1: the fused joint+TDT loss seam (finetune/tdt-fused) in every B0' run, installed after
        stage 3's own TRAINING_SOURCES (so it extends them); its settings() in the identity and the resume key."""
        import tdt_fused
        original, sources = train.compute_losses, train.TRAINING_SOURCES
        with tb.seam():
            self.assertIs(train.compute_losses, tdt_fused.compute_losses)
            self.assertEqual(train.TRAINING_SOURCES, tuple(sources) + tb.SOURCES + (tdt_fused.SOURCE,))
        self.assertIs(train.compute_losses, original)
        self.assertEqual(train.TRAINING_SOURCES, sources)
        with tb.loss_seam():                                       # the memory probe's
            self.assertIs(train.compute_losses, tdt_fused.compute_losses)
        self.assertIs(train.compute_losses, original)
        self.assertIn("loss_seam", tb.RESUME_EXTRA)
        self.assertIn("loss_seam", tb.IDENTITY_KEYS)
        world = fx.World()
        self.addCleanup(world.close)
        h = fx.Harness(world)
        args = h.args(fx.RUNS["pw"], h.data(fx.RUNS["pw"]))
        self.assertEqual(args.loss_seam, tdt_fused.settings())
        self.assertEqual(tb.identity_of(vars(args))["loss_seam"], tdt_fused.settings())

    def test_the_budget_stop_and_the_lock_timeout_end_the_run_with_success(self):
        import tempfile

        import score_teacher
        argv = ["--run", "trunk", "--labels-run", "R"]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.object(tb.llayout, "require_mount", lambda: None), \
                mock.patch.object(tb, "root", lambda: Path(tmp.name)):
            with mock.patch.object(train, "main", side_effect=score_teacher.LockBusy("held")):
                self.assertEqual(tb.run_train(argv), 0)
            with mock.patch.object(train, "main", side_effect=tb.Stopped("rule")):
                self.assertEqual(tb.run_train(argv), tb.EXIT_STOPPED)
            with mock.patch.object(train, "main", side_effect=tb.NotReady("wait")):
                self.assertEqual(tb.run_train(argv), tb.EXIT_NOT_READY)

            def stopped(reason):
                def main(_argv):
                    tb._ACTIVE["ctx"] = types.SimpleNamespace(stop_reason=reason,
                                                              state=types.SimpleNamespace(step=1, batches_read=1))
                    raise SystemExit(train.EXIT_INTERRUPTED)
                return main
            with mock.patch.object(train, "main", stopped("budget")):
                self.assertEqual(tb.run_train(argv), 0)
            with mock.patch.object(train, "main", stopped("SIGTERM")), self.assertRaises(SystemExit):
                tb.run_train(argv)


class Gates(unittest.TestCase):
    """Gates enforced in code (b0prime plan v4.1, architect 2026-10-29): every run needs the replay throughput probe
    of the version 1 plan and the read probe of the version 2 plan to have met the rate rule on the last pools, and
    the version 2 plan's extremes to pass in the memory probe's chosen mode within its lease; the WSD rule's stop is
    the stage's status."""

    def setUp(self):
        import tempfile

        from labeling_fixtures import relabel_root
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = relabel_root(Path(self.tmp.name))
        env.__enter__()
        self.addCleanup(env.__exit__, None, None, None)
        self.v1 = types.SimpleNamespace(sha256="v1sha", n_pools=782, meta={"plan_id": "v1-plan"})
        self.v2 = types.SimpleNamespace(sha256="v2sha", n_pools=782, meta={"plan_id": "v2-plan"}, pool_batches=128)
        patch = mock.patch.object(tb, "v1_plan", lambda: (None, "pool", "cid", self.v1))
        patch.start()
        self.addCleanup(patch.stop)

    @staticmethod
    def probe_body(plan, meets=True, last=782):
        return {"plan": {"plan_sha256": plan.sha256}, "materialized_meets_required": meets,
                "regions": [{"reader": "materialized", "pools": [0, 3]},
                            {"reader": "materialized", "pools": [last - 3, last]}]}

    def test_the_reading_gate(self):
        with self.assertRaises(tb.NotReady):
            tb.reading_gate(self.v2)
        tb.write_json(tb.replay_probe_path(self.v1), self.probe_body(self.v1))
        with self.assertRaises(tb.NotReady):                                     # the version 2 read probe missing
            tb.reading_gate(self.v2)
        for body in (self.probe_body(self.v2, meets=False), self.probe_body(self.v1),
                     self.probe_body(self.v2, last=100)):
            tb.write_json(tb.read_probe_path(), body)
            with self.assertRaises(RuntimeError):
                tb.reading_gate(self.v2)
        tb.write_json(tb.read_probe_path(), self.probe_body(self.v2))
        tb.reading_gate(self.v2)
        tb.write_json(tb.replay_probe_path(self.v1), self.probe_body(self.v1, meets=False))
        with self.assertRaises(RuntimeError):
            tb.reading_gate(self.v2)

    def test_the_version_2_memory_check(self):
        decision = {"fits": True, "chosen": "ckpt", "grad_checkpointing": True, "lease_vram_gb": 20}
        with self.assertRaises(tb.NotReady):
            tb.grad_checkpointing()
        tb.write_json(tb.probe_path(), {"decision": decision})
        with self.assertRaises(tb.NotReady):
            tb.grad_checkpointing()
        ok = {"status": {"worst": "ok", "timed": "ok"}, "peak_process_gib_estimate": 17.5}
        self.assertTrue(tb.probe_v2_check(decision, ok)["pass"])
        self.assertFalse(tb.probe_v2_check(decision, ok | {"peak_process_gib_estimate": 18.5})["pass"])  # 21 > 20
        self.assertFalse(tb.probe_v2_check(decision, ok | {"status": {"worst": "oom at batches [3]",
                                                                      "timed": "ok"}})["pass"])
        sha = tb._sha256_file(tb.probe_path())
        for body, error in (({"pass": True, "mode": "ckpt", "probe_sha256": "other"}, RuntimeError),
                            ({"pass": True, "mode": "no_ckpt", "probe_sha256": sha}, RuntimeError),
                            ({"pass": False, "mode": "ckpt", "probe_sha256": sha}, RuntimeError)):
            tb.write_json(tb.probe_v2_path(), body)
            with self.assertRaises(error):
                tb.grad_checkpointing()
        tb.write_json(tb.probe_v2_path(), {"pass": True, "mode": "ckpt", "probe_sha256": sha, "plan_sha256": "p",
                                           "labels_sha256": "l"})
        self.assertTrue(tb.grad_checkpointing())
        self.assertTrue(tb.grad_checkpointing("p", "l"))
        for plan, labels in (("another plan", "l"), ("p", "other labels")):      # a later plan or label change
            with self.assertRaises(RuntimeError):
                tb.grad_checkpointing(plan, labels)
        tb.write_json(tb.probe_path(), {"decision": decision | {"fits": False}})
        with self.assertRaises(RuntimeError):
            tb.grad_checkpointing()

    def test_the_read_probe_reads_both_stores_and_fails_below_the_rate(self):
        from training import replay_probe
        seen = {}

        def run(plan, pool, store, regions, readers, max_seconds, sources=None):
            seen.update(regions=regions, readers=readers, sources=sources)
            return {"plan": {"plan_sha256": plan.sha256}, "materialized_meets_required": seen.get("meets", True),
                    "regions": []}
        with mock.patch.object(tb, "source_stores", lambda: ["stores"]), \
                mock.patch.object(tb, "pinned_plan", lambda stores: (None, "pool", "cid", self.v2, ["srcs"])), \
                mock.patch.object(tb, "replay_sources", lambda stores, sources: ["extra sources"]), \
                mock.patch.object(tb.llayout, "require_mount", lambda: None), \
                mock.patch.object(replay_probe, "run", run):
            self.assertEqual(tb.read_probe_job(10.0), 0)
            self.assertEqual(seen["sources"], ["extra sources"])
            self.assertEqual(seen["readers"], ["materialized"])
            self.assertEqual(seen["regions"], [(0, 3), (390, 393), (779, 782)])
            self.assertTrue(json.loads(tb.read_probe_path().read_text())["pass"])
            seen["meets"] = False
            self.assertEqual(tb.read_probe_job(10.0), 1)

    def test_a_wsd_stop_is_the_stages_status(self):
        tb.write_json(tb.wsd_rule_path(), {"stop": True, "reasons": ["d = +0.08 > 0.05"]})
        self.assertIn("STOPPED", tb.status()["stage 3"])
        tb.write_json(tb.wsd_rule_path(), {"stop": False, "reasons": []})
        self.assertNotIn("stage 3", tb.status())


class ProbePhases(unittest.TestCase):
    """The memory probe's phases: every worst-case batch is tried (an out-of-memory error is recorded and the next
    worst-case batch runs), the timed phase ends at its first one, and the timed phase runs whatever the worst case
    did, so the planned batches' step time is measured; a mode passes only when both phases pass."""

    class OOM(Exception):
        pass

    def run_phases(self, fail: dict):
        seen, freed, brackets = [], [], []

        def step(kind, j, batch):
            if j in fail.get(kind, ()):
                raise self.OOM()
            seen.append((kind, j))
        status = tb.probe_phases([("worst", iter(range(4))), ("timed", iter(range(4)))], step,
                                 lambda k, j: freed.append((k, j)), lambda k, s: brackets.append((k, s)),
                                 oom=self.OOM)
        return status, seen, freed, brackets

    def test_every_worst_case_batch_is_tried_and_the_timed_phase_still_runs(self):
        status, seen, freed, brackets = self.run_phases({"worst": (1, 3)})
        self.assertEqual(status, {"worst": "oom at batches [1, 3]", "timed": "ok"})
        self.assertEqual(seen, [("worst", 0), ("worst", 2)] + [("timed", j) for j in range(4)])
        self.assertEqual(freed, [("worst", 1), ("worst", 3)])
        self.assertEqual(brackets, [("worst", True), ("worst", False), ("timed", True), ("timed", False)])
        status, seen, _, _ = self.run_phases({"timed": (2, 3)})
        self.assertEqual(status, {"worst": "ok", "timed": "oom at batch 2"})
        self.assertEqual(seen[-1], ("timed", 1))
        self.assertEqual(self.run_phases({})[0], {"worst": "ok", "timed": "ok"})
        self.assertEqual(tb.probe_fits({"worst": "ok", "timed": "ok"}), True)
        self.assertEqual(tb.probe_fits({"worst": "oom at batches [1]", "timed": "ok"}), False)
        self.assertEqual(tb.probe_fits({"worst": "ok", "timed": "oom at batch 2"}), False)

    def test_other_errors_propagate(self):
        def step(kind, j, batch):
            raise ValueError("bug")
        with self.assertRaises(ValueError):
            tb.probe_phases([("worst", [1])], step, lambda k, j: None, lambda k, s: None, oom=self.OOM)


if __name__ == "__main__":
    unittest.main()
