"""Contract tests (CPU): stage 2c's synthetic arms (synthetic plan "Arms", "Mixing", "Augmentation", "Resources",
"Results"; train_synthetic.py and synthetic_report.py).

- the four arms B-TS, B-TSn, T-TS, T-TSn: each student's T* labels at the learning rate selected for its 2b T* arm,
  which must exist (refused until then), is frozen at first use and is recorded with its reason;
- a 2c arm sees exactly the 2b T* arm's real utterances and labels, pool by pool in the same pool order, plus the
  synthetic draw of the version 2 plan; TS and TSn get identical clips in identical batches; augmentation never
  changes a label and is off in TSn;
- the 2b arms are unchanged: their seam, resume keys, sources and ORDER, and every train.py attribute after a 2c
  seam;
- the 2c chunk path: refused until the 2b runs, the selection, the finalized clip pool and the version 2 plan exist;
  then ORDER (the T* runs' eval-stem scores, then the four runs), with no learning-rate argument;
- the mixed-batch probe's batches (most trained rows, most synthetic rows), the complete mixed batch as training
  assembles it, the padded worst case of augmentation, and the synthetic read timing under the replay throughput
  probe's rule (a fresh reader per region);
- the 2c results: recordings as bootstrap clusters, term recall per (domain, side) as vocab.recall counts it, the
  paired recall bootstrap, the underpowered-domain exclusion and the preregistered claims.

Run from finetune/relabel:  ./python -m unittest tests.test_synthetic_arms -v   (or discover -s tests)
"""
from __future__ import annotations

import collections
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import train  # finetune/parakeet-ternary
import paths  # finetune/parakeet-ternary
import train_decoder as td
import train_synthetic as ts
import synthetic_report as srep
from labeling_fixtures import relabel_root
from synthetic import augment, mixing
from training import batchplan as bp
from training import labels as tlabels
from training import poolaudio, schedule
from training import replay as treplay

P, N_POOLS, BUDGET, SEED = 4, 4, 160_000, 7         # as tests/test_training_batchplan.py
SYN_SEED, SHARE = 11, 0.1


class Arms(unittest.TestCase):
    def test_four_synthetic_arms_on_t_star_labels(self):
        self.assertEqual(ts.ARMS, {"B-TS": ("B", "TS"), "B-TSn": ("B", "TSn"), "T-TS": ("T", "TS"),
                                   "T-TSn": ("T", "TSn")})
        self.assertEqual(ts.BASE, {"B-TS": "B-T", "B-TSn": "B-T", "T-TS": "T-T", "T-TSn": "T-T"})
        for arm, base in ts.BASE.items():
            self.assertEqual(td.ARMS[base], (ts.ARMS[arm][0], "T*"))     # the same student, T* labels
        self.assertEqual(ts.SYNTHETIC_RUN, "full")
        self.assertEqual(ts.ORDER, [("stem", "B-T"), ("stem", "T-T"), ("train", "B-TS"), ("train", "B-TSn"),
                                    ("train", "T-TS"), ("train", "T-TSn")])

    def test_no_learning_rate_argument(self):
        a = ts.train_parser().parse_args(["--arm", "B-TS", "--labels-run", "R"])
        self.assertFalse(hasattr(a, "lr"))
        with self.assertRaises(SystemExit):
            ts.train_parser().parse_args(["--arm", "B-TS", "--labels-run", "R", "--lr", "1e-5"])
        with self.assertRaises(SystemExit):
            ts.train_parser().parse_args(["--arm", "B-T", "--labels-run", "R"])

    def test_the_2b_arms_and_their_parser_are_unchanged(self):
        self.assertEqual(td.ARMS, {"B-T": ("B", "T*"), "B-0": ("B", "B0"), "T-T": ("T", "T*"), "T-0": ("T", "B0")})
        self.assertEqual(td.ORDER, [("original", "B"), ("original", "T")] + [(a, lr) for a in td.ARMS for lr in td.LRS])
        # Contract change (2026-10-27, repetition rejection): the 2b resume key gains "repetition", the rule's
        # identity (also inside labels_sha256), so a run made without the rule is not resumed under it.
        self.assertEqual(td.RESUME_EXTRA, ("student", "label_set", "labels_run", "labels_sha256", "cleaning",
                                           "repetition", "plan_sha256", "replay_sha256", "stop_batch", "schedule",
                                           "export_sha256", "init_sha256"))
        self.assertNotIn("relabel/train_synthetic.py", td.SOURCES)
        for arm in ts.ARMS:                                # 2b's train command refuses the 2c arms
            with self.assertRaises(SystemExit):
                td.train_parser().parse_args(["--arm", arm, "--lr", "1e-5", "--labels-run", "R"])


class Seams(unittest.TestCase):
    NAMES = ("parse_args", "build_context", "finish", "recipe_of", "set_train_modes", "run_eval", "RESUME_KEYS",
             "RESUME_ALLOWED_CHANGES", "TRAINING_SOURCES", "OUTPUTS", "Batcher", "label_batch", "make_optimizer",
             "train_step")

    def snapshot(self):
        return {n: getattr(train, n) for n in self.NAMES} | {"RUNS": paths.RUNS, "gpu_lock": paths.gpu_lock}

    def test_the_2c_seam_installs_its_own_parse_build_finish_and_restores_everything(self):
        before = self.snapshot()
        with ts.seam("B-TS"):
            self.assertIs(train.parse_args, ts.parse_args)
            self.assertIs(train.build_context, ts.build_context)
            self.assertIs(train.finish, ts.finish)
            self.assertIs(train.set_train_modes, td.set_train_modes)  # the frozen-encoder recipe of 2b
            for k in td.RESUME_EXTRA + ts.RESUME_EXTRA:
                self.assertIn(k, train.RESUME_KEYS)
            for s in td.SOURCES + ts.EXTRA_SOURCES:
                self.assertIn(s, train.TRAINING_SOURCES)
            self.assertEqual(paths.RUNS, td.root() / "B-TS")
        self.assertEqual(self.snapshot(), before)

    def test_the_2b_seam_gets_no_2c_field(self):
        with td.seam("B-T"):
            self.assertIs(train.parse_args, td.parse_args)
            self.assertIs(train.finish, td.finish)
            for k in ts.RESUME_EXTRA:
                self.assertNotIn(k, train.RESUME_KEYS)
            for s in ts.EXTRA_SOURCES:
                self.assertNotIn(s, train.TRAINING_SOURCES)
        for s in ts.EXTRA_SOURCES:
            self.assertTrue((Path(td.__file__).resolve().parent.parent / s).exists(), s)

    def test_transform_for_augments_ts_only(self):
        pool = augment.Pool(["n.flac"], ["r.flac"], identity="fixture")
        t = mixing.transform_for("TS", seed=5, augment_pool=pool)
        self.assertIsInstance(t, augment.Augmenter)
        self.assertEqual(t.identity, {"seed": 5, "p_apply": augment.P_APPLY, "ops": augment.OPS, "pool": "fixture"})
        self.assertIsNone(mixing.transform_for("TSn", seed=5, augment_pool=pool))
        with self.assertRaises(ValueError):
            mixing.transform_for("T*")


class Selection(unittest.TestCase):
    """The 2c learning rate is the 2b T* arm's selection (decoder_report.select on the selection part), frozen at
    first use; refused while a 2b run is pending; a later different selection is refused."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.h4 = {"lr1e-5": 7.0, "lr3e-5": 6.5, "lr1e-4": 6.8}
        split = types.SimpleNamespace(sha256="split-sha")
        self.patches = [mock.patch.object(td, "root", lambda: self.root),
                        mock.patch.object(td, "part_wer", lambda out, s, part: {"h4": self.h4[out.name]}),
                        mock.patch.object(ts.devsplit, "load", lambda *a, **k: split),
                        mock.patch.object(ts.devsplit, "dev_manifests", lambda: {}),
                        mock.patch.object(td.relabel_paths, "require_mount", lambda: None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def finish_runs(self, arm="B-T", lrs=td.LRS):
        for lr in lrs:
            td.write_json(td.run_dir(arm, lr) / "summary.json", {"lr": lr})

    def test_refused_until_all_three_runs_are_finished(self):
        self.finish_runs(lrs=td.LRS[:2])
        with self.assertRaises(ts.NotReady):
            ts.frozen_selection("B-T")
        self.assertIsNone(ts.selected_lr("B-T"))
        self.assertFalse(ts.selection_path("B-T").exists())

    def test_frozen_at_first_use_with_its_reason(self):
        self.finish_runs()
        sel = ts.frozen_selection("B-T")
        self.assertEqual((sel["selected"], sel["selected_name"]), (3e-5, "lr3e-5"))
        self.assertEqual(sel["selection_h4"], self.h4)
        self.assertIn("lowest selection-part H4", sel["rule"])
        self.assertEqual(sel["split_sha256"], "split-sha")
        self.assertEqual(set(sel["summaries_sha256"]), set(self.h4))
        self.assertTrue(ts.selection_path("B-T").exists())
        self.assertEqual(ts.selected_lr("B-T"), 3e-5)
        self.assertEqual(ts.frozen_selection("B-T")["selected"], 3e-5)
        self.h4["lr1e-4"] = 6.0                               # a different selection later: refused
        with self.assertRaises(RuntimeError):
            ts.frozen_selection("B-T")

    def test_a_changed_selected_run_is_refused(self):
        self.finish_runs()
        ts.frozen_selection("B-T")
        td.write_json(td.run_dir("B-T", 3e-5) / "summary.json", {"lr": 3e-5, "rerun": True})
        with self.assertRaises(RuntimeError):
            ts.frozen_selection("B-T")

    def test_a_2c_run_does_not_start_without_the_selection(self):
        def never(_argv):
            raise AssertionError("train.main must not run without the 2b selection")
        with mock.patch.object(train, "main", never):
            self.assertEqual(ts.run_train(["--arm", "T-TSn", "--labels-run", "R"]), ts.EXIT_NOT_READY)


class Args(unittest.TestCase):
    """parse_args: the selected learning rate, its record, and the 2b T* run's labels, cleaning and plan."""

    def setUp(self):
        self.base = {"labels_run": "R", "labels_sha256": "L", "cleaning": {"version": 1},
                     "repetition": {"name": "repetition", "version": 1}, "plan_sha256": "P1"}
        self.sel = {"arm": "T-T", "selected": 3e-5, "selected_name": "lr3e-5", "rule": "lowest selection-part H4",
                    "selection_h4": {"lr1e-5": 7.0, "lr3e-5": 6.5, "lr1e-4": 6.8}, "split_sha256": "s",
                    "summaries_sha256": {"lr1e-5": "a", "lr3e-5": "b", "lr1e-4": "c"}}
        self.calls = []
        self.patches = [mock.patch.object(ts.devsplit, "load", lambda *a, **k: types.SimpleNamespace(sha256="s")),
                        mock.patch.object(ts.devsplit, "dev_manifests", lambda: {}),
                        mock.patch.object(td, "ensure_student", lambda s: {"export_sha256": "e", "checkpoint_sha256": "c"}),
                        mock.patch.object(ts, "frozen_selection", lambda t_arm: dict(self.sel, arm=t_arm)),
                        mock.patch.object(ts, "base_args", lambda t_arm, lr: dict(self.base)),
                        mock.patch.object(ts, "offline_data", self.offline_data)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        td._ACTIVE.update(data=None, ctx=None)

    def offline_data(self, arm, labels_run, lr, cleaning=True, repetition=True):
        self.calls.append((arm, labels_run, lr, cleaning, repetition))
        plan = types.SimpleNamespace(meta={"base_plan_sha256": "P1"}, version=bp.SYNTHETIC_VERSION)
        return types.SimpleNamespace(
            labels=types.SimpleNamespace(identity={"sha256": "L", "cleaning": self.base["cleaning"],
                                                   "repetition": self.base["repetition"]}, label_set="T*"),
            replay=types.SimpleNamespace(identity={"plan_id": "x-syn", "plan_sha256": "P2", "sha256": "RS",
                                                   "synthetic": {"clip_pool": "CP", "transform": {"seed": 1}}},
                                         consumed_batches=10, plan=plan),
            plan={"steps": 7, "empty_batches": [3], "synthetic_rows": 5}, stop=9,
            schedule=types.SimpleNamespace(to_dict=lambda: {"peak": lr}))

    def test_the_selected_lr_and_its_record(self):
        a = ts.parse_args(["--arm", "T-TS", "--labels-run", "R"])
        self.assertEqual(self.calls, [("T-TS", "R", 3e-5, True, True)])
        self.assertEqual((a.arm, a.lr, a.run_name, a.student, a.label_set), ("T-TS", 3e-5, "lr3e-5", "T", "T*"))
        self.assertEqual((a.synthetic_arm, a.base_arm, a.synthetic_run), ("TS", "T-T", "full"))
        self.assertEqual(a.lr_selection["selected"], 3e-5)
        self.assertEqual(a.lr_selection["selection_h4"], self.sel["selection_h4"])
        self.assertEqual(a.lr_selection["rule"], self.sel["rule"])
        self.assertEqual(a.lr_selection["base_run"], "T-T/lr3e-5")
        self.assertEqual((a.labels_sha256, a.plan_sha256, a.base_plan_sha256, a.clip_pool, a.augment),
                         ("L", "P2", "P1", "CP", {"seed": 1}))
        self.assertEqual(a.max_steps, 7)
        self.assertEqual(td._ACTIVE["data"].plan["steps"], 7)
        for k in ts.RESUME_EXTRA:
            self.assertTrue(hasattr(a, k), k)

    def test_cleaning_follows_the_t_star_run(self):
        self.base["cleaning"] = None
        ts.parse_args(["--arm", "B-TSn", "--labels-run", "R"])
        self.assertEqual(self.calls[-1], ("B-TSn", "R", 3e-5, False, True))

    def test_repetition_rejection_follows_the_t_star_run(self):
        a = ts.parse_args(["--arm", "B-TS", "--labels-run", "R"])
        self.assertEqual(a.repetition, self.base["repetition"])
        self.base["repetition"] = None
        a = ts.parse_args(["--arm", "B-TSn", "--labels-run", "R"])
        self.assertEqual(self.calls[-1], ("B-TSn", "R", 3e-5, True, False))
        self.assertIsNone(a.repetition)

    def test_refusals(self):
        with self.assertRaises(RuntimeError):                # another label run than the T* arm's
            ts.parse_args(["--arm", "B-TS", "--labels-run", "OTHER"])
        self.base["labels_sha256"] = "L2"                    # other labels than the T* arm's
        with self.assertRaises(RuntimeError):
            ts.parse_args(["--arm", "B-TS", "--labels-run", "R"])
        self.base.update(labels_sha256="L", plan_sha256="P0")   # the version 2 plan is not on the T* arm's plan
        with self.assertRaises(RuntimeError):
            ts.parse_args(["--arm", "B-TS", "--labels-run", "R"])


class World(unittest.TestCase):
    """LabeledWorld + the version 1 plan (the 2b arms') + materialized pool audio + a clip pool + the version 2
    plan, as tests/test_training_batchplan.py builds them."""

    @classmethod
    def setUpClass(cls):
        from test_synthetic_mixing import make_aug_pool, make_pool
        from training_fixtures import LabeledWorld
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.w = LabeledWorld(cls.root)
        cls.plan1 = bp.build(cls.w.pool, cls.root / "plans" / "v1", N_POOLS, pool_batches=P, budget=BUDGET, seed=SEED)
        cls.store = cls.root / "pool-audio-store"
        poolaudio.materialize(cls.plan1, cls.w.pool, cls.store)
        cls.clips = make_pool(cls.root / "syn", n=40)
        cls.aug_pool = make_aug_pool(cls.root / "aug")
        cls.plan2 = bp.build(cls.w.pool, cls.root / "plans" / "v2", N_POOLS, clips=cls.clips, pool_batches=P,
                             budget=BUDGET, seed=SEED, syn_seed=SYN_SEED, share=SHARE)
        cls.labels = tlabels.LabelStore.open(cls.w.labels_dir, cls.w.pool, label_set="T*",
                                             through_batch=cls.plan1.consumed_batches)
        cls.sched = schedule.pool_branch(schedule.pool_trunk(1e-4, cls.plan2, N_POOLS), cls.plan2, 2, N_POOLS)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def data(self, kind, **kw):
        args = dict(plan=self.plan2, base_plan=self.plan1, pool_dir=self.w.pool, store=self.store, labels=self.labels,
                    clips=self.clips, sched=self.sched, augment_pool=self.aug_pool, seed=3)
        args.update(kw)
        return ts.synthetic_data(kind, **args)

    def batches(self, data):
        b = data.batcher(None)
        try:
            return [b.next(timeout=60) for _ in range(data.stop)]
        finally:
            b.close()


class Data(World):
    def test_real_utterances_and_labels_are_the_t_star_arms_pool_by_pool(self):
        d = self.data("TS")
        base = treplay.PlannedReplay(self.plan1, self.w.pool, audio="materialized", store=self.store)
        self.assertIs(d.labels, self.labels)
        for p in range(N_POOLS):
            want = collections.Counter((i, self.labels.text(i)) for t in range(self.plan1.pool_start(p),
                                                                                 self.plan1.pool_start(p + 1))
                                       for i in base.batch_ids(t))
            got = collections.Counter()
            for t in range(self.plan2.pool_start(p), self.plan2.pool_start(p + 1)):
                for i, syn in zip(d.replay.batch_ids(t), d.replay.synthetic_mask(t)):
                    if not syn:
                        got[(i, self.labels.text(i))] += 1
            self.assertEqual(got, want, f"pool {p}")
        pools = [d.replay.pool_of(t) for t in range(d.stop)]
        self.assertEqual(pools, sorted(pools))                # pools in the same order, each whole
        self.assertEqual(d.stop, self.plan2.n_batches)
        self.assertEqual(d.plan["occurrences"], len(self.w.journal))
        self.assertEqual(d.plan["synthetic_rows"], self.plan2.meta["n_synthetic"])
        self.assertGreater(d.plan["synthetic_rows"], 0)

    def test_served_batches_hold_the_real_rows_with_their_labels_plus_the_draw(self):
        d = self.data("TSn")
        labels = {f"syn:{c['id']}": c["label"] for c in self.clips.clips}
        real = collections.Counter()
        for got in self.batches(d):
            for i, src, text in zip(got["ids"], got["sources"], got["labels"]):
                if src == "synthetic":
                    self.assertEqual(text, labels[i])
                else:
                    self.assertEqual(text, self.labels.text(i))
                    real[i] += 1
        self.assertEqual(real, collections.Counter(r["id"] for r in self.w.journal))

    def test_ts_and_tsn_get_identical_clips_in_identical_batches_and_only_ts_is_augmented(self):
        a, n = self.data("TS"), self.data("TSn")
        self.assertEqual(a.replay.batches, n.replay.batches)
        self.assertIsNone(n.replay.synthetic_transform)
        self.assertIsNone(n.replay.identity["synthetic"]["transform"])
        self.assertEqual(a.replay.identity["synthetic"]["transform"], a.replay.synthetic_transform.identity)
        self.assertNotEqual(a.state_dict()["config_hash"], n.state_dict()["config_hash"])
        changed = 0
        for x, y in zip(self.batches(a), self.batches(n)):
            self.assertEqual((x["index"], x["ids"], x["labels"], x["sources"]), (y["index"], y["ids"], y["labels"],
                                                                                   y["sources"]))
            for row, (i, src) in enumerate(zip(y["ids"], y["sources"])):
                ny, nx = int(y["audio_len"][row]), int(x["audio_len"][row])
                if src == "synthetic":
                    stored = torch.from_numpy(self.clips.read(i[4:]))
                    self.assertEqual(ny, len(stored))                 # TSn: the stored audio
                    self.assertTrue(torch.equal(y["audio"][row, :ny], stored))
                    changed += nx != ny or not torch.equal(x["audio"][row, :nx], stored)
                else:
                    self.assertEqual(nx, ny)
                    self.assertTrue(torch.equal(x["audio"][row, :nx], y["audio"][row, :ny]))
        self.assertGreater(changed, 0)

    def test_refusals(self):
        with self.assertRaises(RuntimeError):                # a version 1 plan
            self.data("TS", plan=self.plan1)
        other = bp.build(self.w.pool, self.root / "plans" / "v1-other", N_POOLS, pool_batches=P, budget=BUDGET,
                         seed=SEED + 1)
        with self.assertRaises(RuntimeError):                # not the base plan of this version 2 plan
            self.data("TS", base_plan=other)
        b0 = tlabels.LabelStore.open(self.w.labels_dir, self.w.pool, label_set="B0",
                                     through_batch=self.plan1.consumed_batches)
        with self.assertRaises(RuntimeError):                # B0 labels
            self.data("TS", labels=b0)
        from test_synthetic_mixing import make_pool
        with self.assertRaises(RuntimeError):                # another clip pool
            self.data("TSn", clips=make_pool(self.root / "syn-other", n=41))
        with self.assertRaises(ValueError):
            self.data("T*")


class Probe(World):
    def test_probe_batches_are_the_most_rows_and_most_synthetic_rows(self):
        d = self.data("TS")
        r = d.replay
        trained = [sum(self.labels.text(i) is not None for i, m in zip(r.batch_ids(t), r.synthetic_mask(t)) if not m)
                   + sum(r.synthetic_mask(t)) for t in range(d.stop)]
        syn = [sum(r.synthetic_mask(t)) for t in range(d.stop)]
        picks = ts.probe_batches(r, self.labels, sample=3, seed=1)
        self.assertEqual(picks["most_rows"], int(np.argmax(trained)))
        self.assertEqual(trained[picks["most_rows"]], max(trained))
        self.assertEqual(syn[picks["most_synthetic"]], max(syn))
        self.assertEqual(len(picks["sample"]), 3)
        self.assertEqual(len(set(picks["sample"])), 3)
        self.assertTrue(all(trained[t] > 0 for t in picks["sample"]))
        self.assertEqual(picks, ts.probe_batches(r, self.labels, sample=3, seed=1))

    def test_the_mixed_batch_is_the_one_training_assembles(self):
        d = self.data("TS")
        served = self.batches(d)
        for t in (0, d.stop - 1, ts.probe_batches(d.replay, self.labels, 1, 1)["most_synthetic"]):
            got = ts.mixed_batch(d.replay, self.labels, t)
            self.assertEqual((got["ids"], got["labels"]), (served[t]["ids"], served[t]["labels"]))
            self.assertTrue(torch.equal(got["audio"], served[t]["audio"]))

    def test_the_bound_puts_every_synthetic_row_at_its_planned_length(self):
        d = self.data("TSn")
        t = ts.probe_batches(d.replay, self.labels, 1, 1)["most_synthetic"]
        got = ts.mixed_batch(d.replay, self.labels, t, transform=ts.BoundTransform())
        b = int(self.plan2.shuffle[t])
        keys = self.plan2.samples[self.plan2.order[int(self.plan2.offsets[b]):int(self.plan2.offsets[b + 1])]]
        self.assertEqual(got["ids"], d.replay.batch_ids(t))
        bounded = 0
        for row, (i, src) in enumerate(zip(got["ids"], got["sources"])):
            if src == "synthetic":
                self.assertEqual(int(got["audio_len"][row]), int(keys[row]))
                self.assertEqual(got["labels"][row], self.clips.by_id[i[4:]]["label"])
                bounded += 1
        self.assertGreater(bounded, 0)
        self.assertLessEqual(got["audio"].shape[0] * got["audio"].shape[1], BUDGET)
        self.assertIsNone(d.replay.synthetic_transform)       # the replay's own transform is restored

    def test_synthetic_reads_are_timed_per_region_with_a_fresh_reader(self):
        with mock.patch.object(ts.treplay, "PlannedReplay", wraps=treplay.PlannedReplay) as opened:
            out = ts.read_probe_run(self.plan2, self.w.pool, self.store, self.clips, [(0, 2), (2, 4)], ["TS", "TSn"],
                                    60.0, augment_pool=self.aug_pool)
        self.assertEqual(opened.call_count, 4)                # a fresh reader per region
        regions = out["regions"]
        self.assertEqual([(x["kind"], x["pools"]) for x in regions],
                         [("TS", [0, 2]), ("TS", [2, 4]), ("TSn", [0, 2]), ("TSn", [2, 4])])
        for x in regions:
            lo, hi = (self.plan2.pool_start(p) for p in x["pools"])
            want = sum(int(e) >= bp.SYN_FLAG for t in range(lo, hi) for e in self.plan2.batch(t))
            self.assertEqual(x["planned_batches"], x["of"])
            self.assertEqual(x["synthetic"]["clips"], want)
            self.assertGreater(x["synthetic"]["read_seconds"], 0)
            self.assertEqual(x["synthetic"]["transform_seconds"] > 0, x["kind"] == "TS")
            self.assertIn("meets_required", x)
        self.assertIn("required_batches_per_s", out)


class Chunk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.lr = {"B-T": 3e-5, "T-T": 1e-4}
        self.patches = [mock.patch.object(td, "root", lambda: self.root),
                        mock.patch.object(ts.devsplit, "load", lambda *a, **k: None),
                        mock.patch.object(ts.devsplit, "dev_manifests", lambda: {}),
                        mock.patch.object(td.relabel_paths, "require_mount", lambda: None),
                        mock.patch.object(ts, "selected_lr", lambda t_arm: self.lr.get(t_arm)),
                        mock.patch.object(ts, "readiness", lambda: [])]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def finish_item(self, item):
        kind, name = item
        if kind == "stem":
            td.write_json(ts.stem_dir(name, self.lr[name]) / "summary.json", {"complete": True})
        else:
            td.write_json(td.run_dir(name, self.lr[ts.BASE[name]]) / "summary.json", {})

    def test_runs_unfinished_items_in_order_without_a_learning_rate(self):
        calls = []

        def runner(cmd, cwd=None):
            calls.append(cmd)
            self.finish_item(ts.ORDER[len(calls)])            # ORDER[0] is already done
            return types.SimpleNamespace(returncode=0)

        self.finish_item(ts.ORDER[0])
        self.assertEqual(ts.chunk("RUN", 110.0, runner=runner), 0)
        self.assertEqual(len(calls), len(ts.ORDER) - 1)
        self.assertEqual(calls[0][2:5], ["score-stem", "--arm", "T-T"])
        self.assertEqual(calls[1][2:7], ["train", "--arm", "B-TS", "--labels-run", "RUN"])
        self.assertEqual([c[4] for c in calls[1:]], ["B-TS", "B-TSn", "T-TS", "T-TSn"])
        self.assertTrue(all("--lr" not in c for c in calls))
        self.assertTrue(all(ts.item_done(i) for i in ts.ORDER))
        self.assertEqual(td.run_dir("T-TS", 1e-4), self.root / "T-TS" / "lr1e-4")   # the T* arm's lr

    def test_a_failed_or_unfinished_item_stops_the_chunk(self):
        self.assertEqual(ts.chunk("RUN", 110.0, runner=lambda cmd, cwd=None: types.SimpleNamespace(returncode=7)), 7)
        calls = []
        ts.chunk("RUN", 110.0, runner=lambda cmd, cwd=None: calls.append(cmd) or types.SimpleNamespace(returncode=0))
        self.assertEqual(len(calls), 1)

    def test_nothing_runs_before_the_2b_runs_selection_clip_pool_and_plan(self):
        calls = []
        with mock.patch.object(ts, "readiness", lambda: ["the 2b runs are not finished"]):
            rc = ts.chunk("RUN", 110.0, runner=lambda cmd, cwd=None: calls.append(cmd))
        self.assertEqual((rc, calls), (ts.EXIT_NOT_READY, []))

    def test_readiness_names_every_missing_input(self):
        ready = {"2b": True, "selection": True, "clips": True, "plan": True, "augment": True}

        def fail(name, exc=RuntimeError):
            def f(*a, **k):
                if not ready[name]:
                    raise exc(name)
                return types.SimpleNamespace(identity="cp")
            return f
        patches = [mock.patch.object(td, "item_done", lambda item: ready["2b"]),
                   mock.patch.object(ts, "frozen_selection", fail("selection", ts.NotReady)),
                   mock.patch.object(ts.mixing.ClipPool, "load", fail("clips")),
                   mock.patch.object(ts, "plan_dirs", lambda clips: (self.root / "v1", self.root / "v2")),
                   mock.patch.object(ts.augment.Pool, "load", fail("augment"))]
        self.patches[-1].stop()                              # the real readiness
        for p in patches:
            p.start()
        try:
            for d in ("v1", "v2"):
                (self.root / d).mkdir()
                (self.root / d / "plan.json").write_text("{}")
            self.assertEqual(ts.readiness(), [])
            for name in ready:
                ready[name] = False
                if name == "plan":
                    (self.root / "v2" / "plan.json").unlink()
                problems = ts.readiness()
                self.assertTrue(problems, name)
                ready[name] = True
                if name == "plan":
                    (self.root / "v2" / "plan.json").write_text("{}")
        finally:
            for p in patches:
                p.stop()
            self.patches[-1].start()


class Recordings(unittest.TestCase):
    def test_a_recording_is_the_segment_id_without_its_index(self):
        self.assertEqual(srep.recording_of("tech-dev-2:4320211-0001"), "tech-dev-2:4320211")
        self.assertEqual(srep.recording_of("tech-dev-1:dev-0-0000"), "tech-dev-1:dev-0")
        self.assertEqual(srep.recording_of("stem-chem:lecture_05_300k-0003-01"),   # a split utterance
                         "stem-chem:lecture_05_300k")
        self.assertEqual(srep.recording_of("tech-dev-2:talk-2019-0003"), "tech-dev-2:talk-2019")
        for bad in ("no-part-0001", "stem-cs:lecture", "stem-cs:lecture-12"):
            with self.assertRaises(ValueError):
                srep.recording_of(bad)


class Recall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from synthetic_fixtures import make_split
        cls.split = make_split()
        cls.texts = ["the eigenvalue and the eigenvector of a determinant", "a race condition and a deadlock",
                     "free cash flow and working capital in basis points", "enthalpy of the catalyst",
                     "the Markov chain and the Fourier transform", "a hash table and binary search",
                     "nothing technical here at all", "the Sharpe ratio of a hedge fund"]

    def records(self, part, drop_words=(), rec_count=4):
        out = []
        for k, text in enumerate(self.texts):
            ref = self.split.normalize(text)
            hyp = " ".join(w for w in ref.split() if w not in drop_words)
            out.append({"id": f"{part}:rec{k % rec_count}-{k:04d}", "ref_norm": ref, "hyp_norm": hyp, "S": 0, "D": 0,
                        "I": 0, "scored": True})
        return out

    def tables(self, models):
        parts = {"stem-math": 4, "tech-dev-2": 3}
        return {p: srep.RecallTable.from_records({m: self.records(p, drop, n) for m, drop in models.items()},
                                                 self.split) for p, n in parts.items()}

    def test_counts_are_vocab_recalls(self):
        from synthetic import vocab
        drop = ("eigenvector", "deadlock", "flow", "fund")
        t = self.tables({"A": drop})
        for side in ("targeted", "heldout"):
            for domain in srep.DOMAINS:
                occ = sum(int(x.occurrences[(domain, side)].sum()) for x in t.values())
                rep = sum(int(x.reproduced["A"][(domain, side)].sum()) for x in t.values())
                want = vocab.recall(self.records("stem-math", drop, 4) + self.records("tech-dev-2", drop, 3),
                                    self.split)[domain][side]
                self.assertEqual((occ, rep), (want["occurrences"], want["reproduced"]), (domain, side))
        self.assertEqual(t["stem-math"].clusters, [f"stem-math:rec{k}" for k in range(4)])

    def test_paired_bootstrap_of_recall(self):
        groups = srep.groups(underpowered=[])
        same = srep.recall_bootstrap(self.tables({"A": (), "TS": ()}), {"TS-A": ("TS", "A")}, groups, replicates=200)
        for g, e in same["groups"].items():
            if e["occurrences"]:
                self.assertEqual((e["contrasts"]["TS-A"]["point"], e["contrasts"]["TS-A"]["ci"]), (0.0, [0.0, 0.0]), g)
        words = {w for text in self.texts for w in self.split.normalize(text).split()}
        res = srep.recall_bootstrap(self.tables({"A": tuple(words), "TS": ()}), {"TS-A": ("TS", "A")}, groups,
                                    replicates=200)
        t = res["groups"]["targeted"]
        self.assertEqual((t["recall"]["TS"], t["recall"]["A"]), (100.0, 0.0))
        self.assertEqual(t["contrasts"]["TS-A"]["point"], 100.0)
        again = srep.recall_bootstrap(self.tables({"A": tuple(words), "TS": ()}), {"TS-A": ("TS", "A")}, groups,
                                      replicates=200)
        self.assertEqual(again, res)
        self.assertEqual(res["clusters"], {"stem-math": 4, "tech-dev-2": 3})

    def test_models_must_cover_the_same_references(self):
        a, b = self.records("stem-cs"), self.records("stem-cs")
        b[0] = dict(b[0], ref_norm="something else")
        with self.assertRaises(ValueError):
            srep.RecallTable.from_records({"A": a, "TS": b}, self.split)
        with self.assertRaises(ValueError):
            srep.RecallTable.from_records({"A": a, "TS": b[1:]}, self.split)

    def test_underpowered_domains_stay_out_of_the_pooled_claim(self):
        g = srep.groups(underpowered=["chemistry"])
        self.assertEqual(g["targeted"], {(d, "targeted") for d in srep.DOMAINS if d != "chemistry"})
        self.assertEqual(g["heldout"], {(d, "heldout") for d in srep.DOMAINS if d != "chemistry"})
        self.assertEqual(g["chemistry/targeted"], {("chemistry", "targeted")})
        self.assertEqual(g["all-domains/targeted"], {(d, "targeted") for d in srep.DOMAINS})


class Claims(unittest.TestCase):
    def res(self, ts_a, ts_a_ci, ts_tsn, ts_tsn_ci, h4_ts_a):
        recall = {"groups": {"targeted": {"contrasts": {"TS-A": {"point": ts_a, "ci": ts_a_ci},
                                                        "TS-TSn": {"point": ts_tsn, "ci": ts_tsn_ci},
                                                        "TSn-A": {"point": 0.0, "ci": [-1.0, 1.0]}}}}}
        h4 = {"mean": {"TS-A": {"point": h4_ts_a, "ci": [h4_ts_a - 1, h4_ts_a + 1]},
                       "TSn-A": {"point": 0.2, "ci": [0.05, 0.3]}, "TS-TSn": {"point": 0.0, "ci": [-1, 1]}}}
        return srep.claims(h4, recall)

    def test_claim_rules(self):
        c = self.res(2.0, [0.5, 3.0], 1.0, [0.1, 2.0], 0.05)
        self.assertTrue(c["synthetic_helps_targeted"])
        self.assertTrue(c["augmentation_matters"])
        self.assertTrue(c["tsn_worse_than_t_star_on_h4"])
        self.assertTrue(self.res(2.0, [0.5, 3.0], 1.0, [0.1, 2.0], 0.10)["synthetic_helps_targeted"])
        self.assertFalse(self.res(2.0, [0.5, 3.0], 1.0, [0.1, 2.0], 0.11)["synthetic_helps_targeted"])  # H4 worse
        self.assertFalse(self.res(2.0, [-0.1, 3.0], 1.0, [0.1, 2.0], 0.0)["synthetic_helps_targeted"])  # interval
        self.assertFalse(self.res(-1.0, [-2.0, -0.5], 1.0, [0.1, 2.0], 0.0)["synthetic_helps_targeted"])
        self.assertFalse(self.res(2.0, [0.5, 3.0], 1.0, [-0.1, 2.0], 0.0)["augmentation_matters"])
        self.assertFalse(self.res(2.0, [0.5, 3.0], -1.0, [-2.0, -0.1], 0.0)["augmentation_matters"])


class Report(unittest.TestCase):
    """The 2c results of one student from scored runs (fake records in a temporary decoder-only root): every
    preregistered quantity is there, with A = the selected T* run and TS, TSn at its learning rate."""

    H4_IDS = {"librispeech_dev_clean": [f"{s}-1-{k:04d}" for s in (11, 12, 13) for k in range(3)],
              "librispeech_dev_other": [f"{s}-2-{k:04d}" for s in (21, 22) for k in range(3)],
              "ami_dev": [f"ami:AMI_ES20{m}a_H00_{k}" for m in (11, 12, 13) for k in range(3)],
              "voxpopuli_dev": [f"voxpopuli:2020010{d}-0900-PLENARY-{k}" for d in (1, 2) for k in range(3)]}

    @classmethod
    def setUpClass(cls):
        from synthetic_fixtures import make_split
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.split = make_split()
        texts = ["the eigenvalue and the eigenvector of a determinant", "a race condition and a deadlock",
                 "free cash flow and working capital in basis points", "enthalpy of the catalyst",
                 "the Markov chain and the Fourier transform", "a hash table and binary search"]
        terms_words = {w for t in texts for w in cls.split.normalize(t).split()} - {"the", "a", "of", "and", "in"}
        drops = {"A": set(list(sorted(terms_words))[::2]), "TS": set(), "TSn": set(list(sorted(terms_words))[::3])}
        lr = 3e-5
        with mock.patch.object(td, "root", lambda: cls.root):
            dirs = {"A": td.run_dir("B-T", lr), "TS": td.run_dir("B-TS", lr), "TSn": td.run_dir("B-TSn", lr)}
            stems = {"A": ts.stem_dir("B-T", lr), "TS": ts.stem_dir("B-TS", lr), "TSn": ts.stem_dir("B-TSn", lr)}
        for m, d in dirs.items():
            for i, (name, ids) in enumerate(cls.H4_IDS.items()):
                recs = [{"id": x, "ref_norm": "one two three four", "hyp_norm": "one two three four", "S": int(k % 3 == 0
                         and m != "TS"), "D": 0, "I": 0, "scored": True} for k, x in enumerate(ids)]
                td.write_json(d / "eval-dev" / f"{name}.json", {"records": recs, "wer": {"wer": 0.05}})
            for name, parts in srep.TERM_SETS.items():
                for part in parts:
                    recs = []
                    for k, text in enumerate(texts):
                        ref = cls.split.normalize(text)
                        recs.append({"id": f"{part}:rec{k % 3}-{k:04d}", "ref_norm": ref, "S": 0, "D": 0, "I": 0,
                                     "hyp_norm": " ".join(w for w in ref.split() if w not in drops[m]), "scored": True})
                    out = stems[m] if name == "eval-stem" else d / "domain"
                    td.write_json(out / f"{part}.json", {"records": recs})
            td.write_json(stems[m] / "summary.json", {"complete": True})
            td.write_json(d / "proxy" / "summary.json", {"pooled": {"tokens_per_second": 3.0}})
            td.write_json(d / "summary.json", {"steps": 5, "batches_read": 6, "seconds_by_source": {"synthetic": 36.0},
                                               "synthetic": {"synthetic_rows": 4}})
        ids = {n: set(v) for n, v in cls.H4_IDS.items()}
        fake_split = types.SimpleNamespace(sha256="s", ids=lambda name, part: ids[name])
        cov = {"underpowered": ["chemistry"], "complete": True, "totals": None, "path": "c", "min_targeted": 100}
        cls.h4 = {"lr1e-5": 7.0, "lr3e-5": 6.5, "lr1e-4": 6.8}       # selection-part H4 of the three B-T runs
        patches = [mock.patch.object(td, "root", lambda: cls.root),
                   mock.patch.object(td, "part_wer", lambda out, split, part: {"h4": cls.h4[out.name]}),
                   mock.patch.object(td.relabel_paths, "require_mount", lambda: None),
                   mock.patch.object(srep.devsplit, "load", lambda *a, **k: fake_split),
                   mock.patch.object(srep.devsplit, "dev_manifests", lambda: {}),
                   mock.patch.object(srep.vocab.Split, "load", lambda *a, **k: cls.split),
                   mock.patch.object(srep, "coverage_info", lambda split: cov),
                   mock.patch.object(srep, "check_recordings", lambda part: 3),
                   mock.patch.object(srep.layout, "split_path", lambda: srep.layout.relabel_root() / "split.json")]
        cls.patches = patches
        for p in patches:
            p.start()
        for other in (1e-5, 1e-4):                         # the other two B-T runs; then freeze as training does
            td.write_json(td.run_dir("B-T", other) / "summary.json", {"steps": 5})
        ts.frozen_selection("B-T")
        cls.result = srep.analyze_all()
        cls.text = srep.markdown(cls.result)

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches:
            p.stop()
        cls.tmp.cleanup()

    def test_a_rerun_selected_t_star_run_is_refused(self):
        """Regression (review r1): a selected T* run rerun after the selection was frozen (its summary changed)."""
        path = td.run_dir("B-T", 3e-5) / "summary.json"
        saved = path.read_bytes()
        try:
            td.write_json(path, {"steps": 5, "rerun": True})
            with self.assertRaises(RuntimeError):
                srep.analyze_all()
        finally:
            path.write_bytes(saved)
        self.assertEqual(srep.analyze_all()["students"]["B"]["status"], "complete")

    def test_a_selection_that_moved_is_refused(self):
        """Regression (review r1): the three runs now select another learning rate than the frozen one."""
        try:
            self.h4["lr1e-4"] = 6.0
            with self.assertRaises(RuntimeError):
                srep.analyze_all()
        finally:
            self.h4["lr1e-4"] = 6.8

    def test_the_2c_section_uses_the_2b_sections_learning_rate(self):
        """Regression (review r1): main refuses, writing nothing, when the 2b section selects another rate."""
        out = self.root / "RESULTS.md"
        for lr, refused in ((1e-4, True), (3e-5, False)):
            r2b = {"selection": {"B-T": {"selected": lr}, "T-T": {"selected": None}}}
            with mock.patch.object(srep.drep, "analyze_all", lambda: r2b), \
                    mock.patch.object(srep.drep, "markdown", lambda r: "2b\n"):
                if refused:
                    with self.assertRaises(RuntimeError):
                        srep.main(out)
                    self.assertFalse(out.exists())
                    self.assertFalse((td.root() / "results-synthetic.json").exists())
                else:
                    self.assertEqual(srep.main(out), 0)
                    self.assertTrue(out.read_text().startswith("2b\n"))

    def test_one_student_complete_the_other_waiting_for_its_selection(self):
        b, t = self.result["students"]["B"], self.result["students"]["T"]
        self.assertEqual(b["status"], "complete")
        self.assertEqual(b["dirs"], {"A": "B-T/lr3e-5", "TS": "B-TS/lr3e-5", "TSn": "B-TSn/lr3e-5"})
        self.assertEqual(b["lr_selection"]["selected"], 3e-5)
        self.assertIn("not selected", t["status"])

    def test_the_preregistered_quantities(self):
        b = self.result["students"]["B"]
        g = b["recall"]["groups"]
        self.assertNotIn(("chemistry", "targeted"), {tuple(k.split("/")) for k in g["targeted"]["keys"]})
        self.assertIn("chemistry/targeted", g["all-domains/targeted"]["keys"])
        for name in ("targeted", "heldout", "mathematics/targeted", "mathematics/heldout"):
            self.assertIn(name, g)
        self.assertEqual(g["targeted"]["recall"]["TS"], 100.0)
        self.assertGreater(g["targeted"]["contrasts"]["TS-A"]["point"], 0)
        self.assertEqual(set(b["h4"]["mean"]), set(srep.CONTRASTS))
        self.assertGreater(b["h4"]["mean"]["TSn-A"]["point"], -1e-9)
        self.assertEqual(set(b["domain_wer"]), {"parts", "eval-stem", "dev-technical"})
        self.assertEqual(set(b["recall"]["clusters"]), {p for ps in srep.TERM_SETS.values() for p in ps})
        self.assertEqual(set(b["claims"]), {"synthetic_helps_targeted", "augmentation_matters",
                                            "tsn_worse_than_t_star_on_h4", "h4_ts_minus_a_points", "rules"})
        self.assertEqual(b["proxy"]["TS"], {"tokens_per_second": 3.0})
        for words in ("Claim 1", "Claim 2", "TSn worse than A on H4", "eval-stem (mean)", "Decoder-step proxy",
                      "Learning rate lr3e-5"):
            self.assertIn(words, self.text)


if __name__ == "__main__":
    unittest.main()
