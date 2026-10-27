"""Contract tests (CPU): stage 2b's arms, frozen-encoder modes, trainable set, decoder-step proxy, chunk order,
learning-rate selection and claims (decoder-only plan)."""
from __future__ import annotations

import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

import train  # finetune/parakeet-ternary
import train_decoder as td
import decoder_report as rep
from decoder_fixtures import tiny_model
from training import decoder_steps, schedule
from training import seam as tseam


class Arms(unittest.TestCase):
    def test_four_arms_three_learning_rates_twelve_runs(self):
        self.assertEqual(td.ARMS, {"B-T": ("B", "T*"), "B-0": ("B", "B0"), "T-T": ("T", "T*"), "T-0": ("T", "B0")})
        self.assertEqual(td.LRS, (1e-5, 3e-5, 1e-4))
        runs = [i for i in td.ORDER if i[0] != "original"]
        self.assertEqual(len(runs), 12)
        self.assertEqual(len(set(runs)), 12)
        self.assertEqual([i for i in td.ORDER if i[0] == "original"], [("original", "B"), ("original", "T")])
        self.assertEqual(td.run_dir("B-T", 3e-5), td.root() / "B-T" / "lr3e-5")
        self.assertEqual(td.STUDENTS["T"]["checkpoint_sha256"][:8], "003b1289")
        self.assertIsNone(td.STUDENTS["B"]["checkpoint_sha256"])     # recorded at first use

    def test_schedule_and_stop(self):
        # Contract change (2026-10-24, decoder-only plan "Data", "Schedule"): the span is the batch plan's pools
        # 0-155 and the cooldown pools 133-155, on the planned-batch index (was 20,000 consumed batches, 17,000-19,999).
        class Plan:                                   # pool starts of a plan with 100 planned batches per pool
            n_pools = 156

            def pool_start(self, p):
                return 100 * p
        s = schedule.decoder_only(1e-4, Plan())
        self.assertEqual(td.PLAN_POOLS, 156)
        self.assertEqual((s.warmup, s.cooldown_start, s.last_batch, s.stop_batch), (312, 13_300, 15_599, 15_600))


class Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.info = td.configure(cls.model)

    def test_only_prediction_and_joint_parameters_train(self):
        m = self.model
        self.assertTrue(all(not p.requires_grad for p in m.encoder.parameters()))
        for name, p in m.named_parameters():
            self.assertEqual(p.requires_grad, name.startswith(("decoder.", "joint.")), name)
        opt, sched = tseam.make_optimizer(m, 1e-4, 10)
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        want = {id(p) for n, p in m.named_parameters() if n.startswith(("decoder.", "joint."))}
        self.assertEqual(in_opt, want)
        self.assertEqual(opt.defaults["betas"], (0.9, 0.98))
        self.assertTrue(all(p.dtype == torch.float32 for g in opt.param_groups for p in g["params"]))

    def test_frozen_encoder_stays_in_eval_mode_after_every_set_train_modes(self):
        m = self.model
        m.eval()
        with td.seam("B-T"):
            train.set_train_modes(m, train.recipe_of(types.SimpleNamespace(recipe="decoder-only")))
        self.assertTrue(m.training)
        self.assertTrue(m.decoder.training and m.joint.training)
        self.assertTrue(m.spec_augmentation.training)              # SpecAugment stays on
        self.assertEqual(sum(x.training for x in m.encoder.modules()), 0)
        # without the seam, train.py's own set_train_modes leaves the encoder training: the seam is what freezes it
        train.set_train_modes(m, td.RECIPE)
        self.assertGreater(sum(x.training for x in m.encoder.modules()), 0)
        td.set_train_modes(m, td.RECIPE)
        self.assertEqual(sum(x.training for x in m.encoder.modules()), 0)

    def test_prediction_and_joint_dropout_zero(self):
        self.assertEqual(self.info["pred_joint_dropout"], [0.0])
        self.assertGreater(self.info["pred_joint_dropout_modules_zeroed"], 0)

    def test_verify_refuses_a_trainable_encoder_or_training_mode(self):
        m = tiny_model(1)
        td.configure(m)
        m.encoder.layers[0].requires_grad_(True)
        with self.assertRaises(RuntimeError):
            td.verify(m)
        m = tiny_model(2)
        td.configure(m)
        m.encoder.train()
        with self.assertRaises(RuntimeError):
            td.verify(m)

    def test_trained_state_round_trip_and_key_check(self):
        a, b = tiny_model(3), tiny_model(4)
        state = {k: v.clone() for k, v in td.trained_state(a).items()}
        self.assertTrue(all(k.startswith(("decoder.", "joint.")) for k in state))
        td.load_trained(b, state)
        for k, v in td.trained_state(b).items():
            self.assertTrue(torch.equal(v, state[k]))
        state.pop(next(iter(state)))
        with self.assertRaises(ValueError):
            td.load_trained(b, state)

    def test_encoder_hash_sees_any_change(self):
        m = tiny_model(5)
        h = td.encoder_sha256(m)
        self.assertEqual(h, td.encoder_sha256(m))
        with torch.no_grad():
            next(m.encoder.parameters()).view(-1)[0] += 1e-6
        self.assertNotEqual(h, td.encoder_sha256(m))


class DecoderSteps(unittest.TestCase):
    """The proxy's counts against NeMo itself: tokens against its greedy hypotheses, iterations and durations
    against its all-step record (preserve_alignments) on the same encoder output."""

    def test_counts_match_nemo(self):
        from nemo.collections.asr.parts.submodules.transducer_decoding.tdt_label_looping import (
            GreedyBatchedTDTLabelLoopingComputer)
        checked = tokens_seen = 0
        for seed in range(4):
            m = tiny_model(10 + seed).eval()
            with torch.no_grad():
                m.joint.joint_net[-1].bias[-6:-5] += float(seed - 1)    # vary the blank tendency across models
                g = torch.Generator().manual_seed(seed)
                audio = torch.randn(5, 16000 * 3, generator=g) * 0.1
                lengths = torch.tensor([48000, 40000, 30000, 20000, 9000])
                enc, enc_len = m.forward(input_signal=audio, input_signal_length=lengths)
                ours = decoder_steps.greedy_steps(m, enc, enc_len)
                hyps = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len,
                                                                  return_hypotheses=True)
                comp = m.decoding.decoding.decoding_computer
                ref = GreedyBatchedTDTLabelLoopingComputer(comp.decoder, comp.joint, comp._blank_index,
                                                           comp.durations.tolist(), comp.max_symbols,
                                                           preserve_alignments=True, include_duration=True,
                                                           allow_cuda_graphs=False)
                all_steps, _ = ref.torch_impl(enc.transpose(1, 2), enc_len)
            for i, h in enumerate(hyps):
                self.assertEqual(ours[i]["tokens"], [int(t) for t in torch.as_tensor(h.y_sequence).tolist()])
                n = int(all_steps.current_lengths[i])
                self.assertEqual(ours[i]["iterations"], n)
                self.assertEqual(ours[i]["durations"], int(all_steps.token_durations[i, :n].sum()))
                self.assertGreaterEqual(ours[i]["iterations"], len(ours[i]["tokens"]))
                checked += 1
                tokens_seen += len(ours[i]["tokens"])
        self.assertEqual(checked, 20)
        self.assertGreater(tokens_seen, 0)


class Chunk(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [mock.patch.object(td, "root", lambda: self.root),
                        mock.patch.object(td.devsplit, "load", lambda *a, **k: None),
                        mock.patch.object(td.relabel_paths, "require_mount", lambda: None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def finish_item(self, item):
        kind, value = item
        if kind == "original":
            td.write_json(td.original_dir(value) / "scores.json", {"complete": True})
        else:
            td.write_json(td.run_dir(kind, value) / "summary.json", {})

    def offline_data_calls(self, calls) -> dict:
        """offline_data(*args, **kw) for each (arm, args, kw) in calls, up to the label store; the store's
        cleaning and repetition arguments per call."""
        seen = {}

        def store_open(run_dir, pool, *, label_set, through_batch, cleaning=None, repetition=None):
            seen[len(seen)] = {"label_set": label_set, "cleaning": cleaning, "repetition": repetition}
            raise RuntimeError("stop")
        plan = types.SimpleNamespace(meta={"pinned_hash": "h"}, budget=td.BATCH_SECONDS * 16000, consumed_batches=5)
        patches = [mock.patch("labeling.__main__.pinned", lambda: {"batch": {"batch_seconds": td.BATCH_SECONDS},
                                                                   "pinned_hash": "h"}),
                   mock.patch("labeling.__main__.c1_dir", lambda: Path("pool")),
                   mock.patch("labeling.config.config_id", lambda pin: "cid"),
                   mock.patch("training.batchplan.Plan.load", lambda d: plan),
                   mock.patch("training.batchplan.production_dir", lambda cid, pools: Path("plan")),
                   mock.patch.object(td.tlabels.LabelStore, "open", store_open),
                   mock.patch.object(td.tcleaning, "production", lambda run: f"spec:{run}"),
                   mock.patch.object(td.trep, "production", lambda run: f"rule:{run}")]
        for p in patches:
            p.start()
        try:
            for arm, args, kw in calls:
                with self.assertRaises(RuntimeError):
                    td.offline_data(arm, *args, **kw)
        finally:
            for p in patches:
                p.stop()
        return seen

    def test_runs_unfinished_items_in_order_and_stops_on_an_unfinished_run(self):
        calls = []

        def runner(cmd, cwd=None):
            calls.append(cmd)
            item = td.ORDER[len(calls) - 1 + 1]                 # ORDER[0] is already done
            if len(calls) < 3:
                self.finish_item(item)
            return types.SimpleNamespace(returncode=0)

        self.finish_item(td.ORDER[0])
        self.assertEqual(td.chunk("RUN", 110.0, runner=runner), 0)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][2:5], ["score-original", "--student", "T"])
        self.assertEqual(calls[1][2:9], ["train", "--arm", "B-T", "--lr", "1e-05", "--labels-run", "RUN"])
        self.assertEqual(calls[2][4], "B-T")
        self.assertEqual(float(calls[2][6]), 3e-5)

    def test_a_failed_item_fails_the_chunk(self):
        self.assertEqual(td.chunk("RUN", 110.0, runner=lambda cmd, cwd=None: types.SimpleNamespace(returncode=7)), 7)

    def test_no_item_starts_without_enough_budget(self):
        calls = []
        td.chunk("RUN", td.MIN_CHUNK_MINUTES - 1, runner=lambda cmd, cwd=None: calls.append(cmd))
        self.assertEqual(calls, [])


class LabelCleaning(Chunk):
    """Cleaning plan "Where it applies": every arm (the B0-label controls included) reads cleaning v1 by default;
    --no-cleaning reads the uncleaned labels; the module and tables are training sources; the cleaning block is a
    resume key."""

    def test_on_by_default_for_every_arm_with_an_explicit_off(self):
        self.assertFalse(td.train_parser().parse_args(["--arm", "B-0", "--lr", "1e-5", "--labels-run", "R"])
                         .no_cleaning)
        self.assertTrue(td.train_parser().parse_args(["--arm", "T-0", "--lr", "1e-5", "--labels-run", "R",
                                                      "--no-cleaning"]).no_cleaning)
        with mock.patch.object(td.tcleaning, "production", lambda run: ("spec", run)):
            self.assertEqual(td.label_cleaning("R"), ("spec", "R"))
            self.assertIsNone(td.label_cleaning("R", enabled=False))

    def test_offline_data_reads_the_store_through_the_spec(self):
        for arm, (_, label_set) in td.ARMS.items():
            seen = self.offline_data_calls([(arm, ("R", 1e-5), {}), (arm, ("R", 1e-5), {"cleaning": False})])
            self.assertEqual(seen[0]["label_set"], label_set)
            self.assertEqual(seen[0]["cleaning"], "spec:R", arm)
            self.assertIsNone(seen[1]["cleaning"], arm)

    def test_chunk_passes_no_cleaning_to_every_train_item(self):
        calls = []
        for item in td.ORDER[:2]:
            self.finish_item(item)
        td.chunk("RUN", 110.0, runner=lambda cmd, cwd=None: calls.append(cmd) or types.SimpleNamespace(returncode=0),
                 cleaning=False)
        self.assertEqual(calls[0][-1], "--no-cleaning")
        calls.clear()
        td.chunk("RUN", 110.0, runner=lambda cmd, cwd=None: calls.append(cmd) or types.SimpleNamespace(returncode=0))
        self.assertNotIn("--no-cleaning", calls[0])

    def test_sources_and_resume_key(self):
        for name in ("relabel/training/cleaning.py", "relabel/terms/cleaning/forms.tsv",
                     "relabel/terms/cleaning/variants.tsv", "relabel/terms/cleaning/dialect.tsv",
                     "relabel/terms/cleaning/choices.tsv"):
            self.assertIn(name, td.SOURCES)
            self.assertTrue((Path(td.__file__).resolve().parent.parent / name).exists(), name)
        self.assertIn("cleaning", td.RESUME_EXTRA)
        self.assertIn("labels_sha256", td.RESUME_EXTRA)


class RepetitionRule(Chunk):
    """Repetition rejection (training/repetition.py, rule "repetition" v1): on by default for every arm, the B0-label
    controls included (they drop the ids the T* label rejects); --no-repetition-reject turns it off; its identity
    is part of labels_sha256 and its own resume key."""

    def test_on_by_default_for_every_arm_with_an_explicit_off(self):
        argv = ["--arm", "B-0", "--lr", "1e-5", "--labels-run", "R"]
        self.assertFalse(td.train_parser().parse_args(argv).no_repetition_reject)
        self.assertTrue(td.train_parser().parse_args(argv + ["--no-repetition-reject"]).no_repetition_reject)
        with mock.patch.object(td.trep, "production", lambda run: ("rule", run)):
            self.assertEqual(td.label_repetition("R"), ("rule", "R"))
            self.assertIsNone(td.label_repetition("R", enabled=False))

    def test_offline_data_passes_the_rule_to_every_arms_store(self):
        for arm, (_, label_set) in td.ARMS.items():
            seen = self.offline_data_calls([(arm, ("R", 1e-5), {}), (arm, ("R", 1e-5), {"repetition": False}),
                                            (arm, ("R", 1e-5), {"cleaning": False})])
            self.assertEqual(seen[0]["repetition"], "rule:R", arm)
            self.assertIsNone(seen[1]["repetition"], arm)
            self.assertEqual(seen[1]["cleaning"], "spec:R", arm)               # independent switches
            self.assertEqual(seen[2]["repetition"], "rule:R", arm)

    def test_chunk_passes_no_repetition_reject_to_every_train_item(self):
        calls = []
        for item in td.ORDER[:2]:
            self.finish_item(item)
        run = lambda cmd, cwd=None: calls.append(cmd) or types.SimpleNamespace(returncode=0)   # noqa: E731
        td.chunk("RUN", 110.0, runner=run, repetition=False)
        self.assertIn("--no-repetition-reject", calls[0])
        self.assertNotIn("--no-cleaning", calls[0])
        calls.clear()
        td.chunk("RUN", 110.0, runner=run)
        self.assertNotIn("--no-repetition-reject", calls[0])
        seen = []
        with mock.patch.object(td, "chunk", lambda run, minutes, cleaning=True, repetition=True:
                               seen.append((cleaning, repetition)) or 0):
            td.main(["chunk", "--labels-run", "R", "--no-repetition-reject"])
            td.main(["chunk", "--labels-run", "R"])
        self.assertEqual(seen, [(True, False), (True, True)])

    def finish_with_rule(self, item, rule):
        self.finish_item(item)
        td.write_json(td.run_dir(*item) / "config.json", {"args": {"repetition": rule}})

    def test_chunk_refuses_a_finished_run_of_another_rule(self):
        """A finished run is not resumed, so the resume check never sees it: the chunk checks its recorded rule."""
        rule = {"name": "repetition", "version": 1, "n": [1, 6], "threshold": 32, "rejected": 26}
        calls = []
        run = lambda cmd, cwd=None: calls.append(cmd) or types.SimpleNamespace(returncode=0)   # noqa: E731
        for item in td.ORDER[:2]:
            self.finish_item(item)
        self.finish_with_rule(td.ORDER[2], None)                       # B-T lr1e-5, made without the rule
        self.assertEqual(td.chunk("RUN", 110.0, runner=run), train.EXIT_RESUME_MISMATCH)
        self.assertEqual(calls, [])
        self.assertEqual(td.chunk("RUN", 110.0, runner=run, repetition=False), 0)
        self.assertGreater(len(calls), 0)
        calls.clear()
        self.finish_with_rule(td.ORDER[2], rule)
        self.assertEqual(td.chunk("RUN", 110.0, runner=run), 0)
        self.assertGreater(len(calls), 0)
        self.assertEqual(td.chunk("RUN", 110.0, runner=run, repetition=False), train.EXIT_RESUME_MISMATCH)
        self.finish_with_rule(td.ORDER[2], rule | {"version": 0})
        self.assertEqual(td.chunk("RUN", 110.0, runner=run), train.EXIT_RESUME_MISMATCH)

    def test_the_report_refuses_runs_of_different_rules(self):
        rule = {"name": "repetition", "version": 1, "n": [1, 6], "threshold": 32, "rejected": 26}
        self.finish_with_rule(("B-T", 1e-5), rule)
        self.finish_with_rule(("B-0", 1e-5), rule)
        self.assertEqual(rep.label_rule(), rule)
        self.finish_with_rule(("B-0", 1e-5), None)
        with self.assertRaises(RuntimeError):
            rep.label_rule()

    def test_sources_and_resume_key(self):
        self.assertIn("relabel/training/repetition.py", td.SOURCES)
        self.assertTrue((Path(td.__file__).resolve().parent.parent / "relabel/training/repetition.py").exists())
        self.assertIn("repetition", td.RESUME_EXTRA)

    def test_parse_args_records_the_rule_and_the_labels_hash(self):
        block = {"name": "repetition", "version": 1, "rejected": 26}
        seen = []

        def offline_data(arm, labels_run, lr, cleaning=True, repetition=True):
            seen.append(repetition)
            return types.SimpleNamespace(
                labels=types.SimpleNamespace(identity={"sha256": "L", "cleaning": None, "repetition": block},
                                             label_set="B0"),
                replay=types.SimpleNamespace(identity={"plan_id": "p", "plan_sha256": "P", "sha256": "RS"},
                                             consumed_batches=10),
                plan={"steps": 7, "empty_batches": [3]}, stop=9,
                schedule=types.SimpleNamespace(to_dict=lambda: {"peak": lr}))
        with mock.patch.object(td, "offline_data", offline_data), \
                mock.patch.object(td.devsplit, "dev_manifests", lambda: {}), \
                mock.patch.object(td.devsplit, "load", lambda *a, **k: types.SimpleNamespace(sha256="s")), \
                mock.patch.object(td, "ensure_student", lambda s: {"export_sha256": "e", "checkpoint_sha256": "c"}):
            a = td.parse_args(["--arm", "B-0", "--lr", "1e-5", "--labels-run", "R"])
            self.assertEqual((a.repetition, a.labels_sha256), (block, "L"))
            td.parse_args(["--arm", "B-0", "--lr", "1e-5", "--labels-run", "R", "--no-repetition-reject"])
        td._ACTIVE.update(data=None, ctx=None)
        self.assertEqual(seen, [True, False])


class ResumeRefusal(unittest.TestCase):
    """A run directory whose checkpoint was written under another label identity (labels_sha256, the repetition
    rule) is not resumed: train.load_resume_payload exits EXIT_RESUME_MISMATCH (78) naming the differences, and
    run_train passes that exit on (a chunk then fails with it)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name) / "B-T" / "lr3e-5"

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, **kw):
        import argparse
        a = argparse.Namespace(**{k: None for k in train.RESUME_KEYS + td.RESUME_EXTRA})
        a.__dict__.update(arm="B-T", recipe="decoder-only", lr=3e-5, max_steps=16_000, label_set="T*",
                          labels_run="main-f3", labels_sha256="old", repetition=None, ckpt_minutes=30.0,
                          budget_minutes=110.0, deadline=None)
        a.__dict__.update(kw)
        return a

    def test_another_label_identity_is_refused(self):
        import contextlib
        import io
        old = self.args()
        with td.seam("B-T"):
            train.save_checkpoint(self.run, {"step": 2839, "args": vars(old), "resume_key": train.resume_key(old)})
            self.assertEqual(train.load_resume_payload(self.run, self.args(budget_minutes=50.0))["step"], 2839)
            new = self.args(labels_sha256="new", repetition={"name": "repetition", "version": 1, "rejected": 26},
                            max_steps=15_998)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as stop:
                train.load_resume_payload(self.run, new)
        self.assertEqual(stop.exception.code, train.EXIT_RESUME_MISMATCH)
        self.assertIn("refusing to resume", out.getvalue())
        self.assertIn("--labels-sha256 'old' -> 'new'", out.getvalue())
        self.assertIn("--repetition None ->", out.getvalue())

    def test_run_train_passes_the_refusal_on(self):
        argv = ["--arm", "B-T", "--lr", "3e-5", "--labels-run", "R"]
        with mock.patch.object(td, "root", lambda: Path(self.tmp.name)), \
                mock.patch.object(td.relabel_paths, "require_mount", lambda: None), \
                mock.patch.object(train, "main", side_effect=SystemExit(train.EXIT_RESUME_MISMATCH)), \
                self.assertRaises(SystemExit) as stop:
            td.run_train(argv)
        self.assertEqual(stop.exception.code, train.EXIT_RESUME_MISMATCH)


PINNED_TRAIN_STEP = train.train_step          # finetune/parakeet-ternary/train.py's, before any seam


class FakeLosses:
    """A stand-in for train.compute_losses (the TDT loss is CUDA-only): scale x the sum of squares of the trainable
    parameters. bad = {call number: "nan-grad" (finite loss, NaN gradient) | "inf-loss"}."""

    def __init__(self, scale: float = 1.0, bad: dict | None = None):
        self.scale, self.bad, self.calls = scale, bad or {}, 0

    def __call__(self, student, audio, audio_len, tokens, token_len, recipe, teacher_enc=None, teacher_len=None):
        self.calls += 1
        params = [p for p in student.parameters() if p.requires_grad]
        loss = self.scale * sum(p.float().pow(2).sum() for p in params)
        kind = self.bad.get(self.calls)
        if kind == "nan-grad":
            loss = loss + (params[0].sum() * 0.0).abs().sqrt()             # value 0, gradient NaN
        elif kind == "inf-loss":
            loss = loss + float("inf")
        return loss, loss, None


class NonfiniteGuard(unittest.TestCase):
    """training/seam.py guarded_train_step, train_step of the decoder-only and synthetic arms: a finite step is the
    pinned train_step's, bit for bit; a non-finite loss or gradient norm skips the optimizer step (gradients
    zeroed, parameters and AdamW state unchanged), still steps the scheduler, and is counted in the run's state
    (checkpointed); the fourth one in a run stops it."""

    LABELED = {"audio": None, "audio_len": None, "tokens": None, "token_len": None, "teacher_enc": None,
               "teacher_len": None, "seconds": 2.0, "rows": 1}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def ctx(self, name="run", seed=0):
        model = tiny_model(21)
        td.configure(model)
        args = types.SimpleNamespace(**{k: None for k in train.RESUME_KEYS + td.RESUME_EXTRA})
        args.__dict__.update(arm="B-T", recipe="decoder-only", lr=1e-3, max_steps=100, seed=seed, label_set="T*",
                             ckpt_minutes=30.0)
        run = Path(self.tmp.name) / name
        run.mkdir(parents=True, exist_ok=True)
        train.seed_everything(seed, model)
        with td.seam("B-T"):
            c = td.DecoderContext(args, run, model, None, "cpu", td.encoder_sha256(model))
        for g in c.optimizer.param_groups:
            g["lr"] = 1e-3
        return c

    @staticmethod
    def trainable(c):
        return [p for g in c.optimizer.param_groups for p in g["params"]]

    def step(self, fn, c, losses):
        with mock.patch.object(train, "compute_losses", losses):
            return fn(c, dict(self.LABELED), self.trainable(c))

    def snapshot(self, c):
        return {k: v.detach().clone() for k, v in td.trained_state(c.student).items()}

    def assertSameState(self, a, b):
        for k, v in a.items():
            self.assertTrue(torch.equal(v, b[k]), k)

    def test_the_seams_install_it(self):
        import train_synthetic as ts
        for seam in (td.seam("B-T"), ts.seam("B-TS")):
            with seam:
                self.assertIs(train.train_step, tseam.guarded_train_step)
            self.assertIs(train.train_step, PINNED_TRAIN_STEP)
        self.assertEqual(tseam.MAX_NONFINITE_STEPS, 3)

    def test_finite_steps_are_the_pinned_train_step(self):
        for scale in (10.0, 1e-6):                     # gradient norm far above and below the clip at 1.0
            a, b = self.ctx("a"), self.ctx("b")
            self.assertSameState(self.snapshot(a), self.snapshot(b))
            for _ in range(3):
                va = self.step(PINNED_TRAIN_STEP, a, FakeLosses(scale))
                vb = self.step(tseam.guarded_train_step, b, FakeLosses(scale))
                self.assertEqual(va, vb)
                self.assertSameState(self.snapshot(a), self.snapshot(b))
            self.assertTrue(va["norm"] > 1.0 if scale > 1 else va["norm"] < 1.0, va)
            sa, sb = a.optimizer.state_dict(), b.optimizer.state_dict()
            for k in sa["state"]:
                for name, v in sa["state"][k].items():
                    self.assertTrue(torch.equal(v, sb["state"][k][name]), (k, name))
            self.assertEqual(a.scheduler.state_dict(), b.scheduler.state_dict())
            self.assertEqual(b.state.drop_counts.get(tseam.NONFINITE_KEY), 0)

    def test_a_non_finite_step_changes_nothing_but_the_scheduler_and_the_count(self):
        for kind in ("nan-grad", "inf-loss"):
            c = self.ctx(kind)
            self.step(tseam.guarded_train_step, c, FakeLosses(1.0))          # AdamW has state
            before, opt = self.snapshot(c), c.optimizer.state_dict()
            steps = c.scheduler.steps
            values = self.step(tseam.guarded_train_step, c, FakeLosses(1.0, {1: kind}))
            self.assertSameState(before, self.snapshot(c))
            self.assertTrue(all(p.grad is None for p in self.trainable(c)))
            after = c.optimizer.state_dict()
            for k in opt["state"]:
                for name, v in opt["state"][k].items():
                    self.assertTrue(torch.equal(v, after["state"][k][name]), (k, name))
            self.assertEqual(c.scheduler.steps, steps + 1)
            self.assertEqual(c.state.drop_counts[tseam.NONFINITE_KEY], 1)
            self.assertTrue(all(math.isfinite(v) for v in values.values()), values)
            # the next finite step trains again
            self.step(tseam.guarded_train_step, c, FakeLosses(1.0))
            self.assertFalse(torch.equal(next(iter(before.values())), next(iter(self.snapshot(c).values()))))

    def test_the_fourth_skip_stops_the_run(self):
        c = self.ctx()
        for _ in range(3):
            self.step(tseam.guarded_train_step, c, FakeLosses(1.0, {1: "nan-grad"}))
        before = self.snapshot(c)
        with self.assertRaises(tseam.NonfiniteSteps):
            self.step(tseam.guarded_train_step, c, FakeLosses(1.0, {1: "nan-grad"}))
        self.assertSameState(before, self.snapshot(c))
        self.assertEqual(c.state.drop_counts[tseam.NONFINITE_KEY], 3)

    def test_the_count_is_checkpointed_and_resumed(self):
        a = self.ctx("ckpt")
        self.step(tseam.guarded_train_step, a, FakeLosses(1.0))
        self.step(tseam.guarded_train_step, a, FakeLosses(1.0, {1: "inf-loss"}))
        a.state.step = 2
        with td.seam("B-T"):
            train.save_checkpoint(a.run, a.checkpoint_payload())
        payload = train.load_checkpoint(a.run)
        self.assertEqual(payload["state"]["drop_counts"][tseam.NONFINITE_KEY], 1)
        b = self.ctx("ckpt")
        with td.seam("B-T"):
            b.restore(payload)
        self.assertEqual(b.state.drop_counts[tseam.NONFINITE_KEY], 1)
        self.assertEqual(b.scheduler.steps, 2)
        for _ in range(2):
            self.step(tseam.guarded_train_step, b, FakeLosses(1.0, {1: "nan-grad"}))
        with self.assertRaises(tseam.NonfiniteSteps):
            self.step(tseam.guarded_train_step, b, FakeLosses(1.0, {1: "nan-grad"}))


class SkippedStepsReported(unittest.TestCase):
    """The non-finite steps a run skipped (summary decoder_only.nonfinite_skipped_steps) are in both reports' run
    tables (0 expected)."""

    def test_run_tables(self):
        import synthetic_report as srep
        import train_synthetic as ts
        with tempfile.TemporaryDirectory() as d, mock.patch.object(td, "root", lambda: Path(d)):
            for arm, lr in (("B-T", 1e-5), ("B-TS", 3e-5)):
                td.write_json(td.run_dir(arm, lr) / "summary.json",
                              {"steps": 5, "dev_mean_wer": 0.1, "decoder_only": {"nonfinite_skipped_steps": 0}})
            td.write_json(ts.selection_path("B-T"), {"selected": 3e-5})
            row = next(r for r in rep.run_table() if (r["arm"], r["lr"]) == ("B-T", "lr1e-5"))
            self.assertEqual(row["nonfinite_skipped_steps"], 0)
            row = next(r for r in srep.run_table() if r["arm"] == "B-TS")
            self.assertEqual(row["nonfinite_skipped_steps"], 0)


class Selection(unittest.TestCase):
    def test_lowest_selection_h4_with_near_ties_to_the_lower_learning_rate(self):
        def run(values):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                with mock.patch.object(td, "root", lambda: root), \
                        mock.patch.object(td, "part_wer", lambda out, split, part: {"h4": values[out.name]}):
                    for lr in td.LRS:
                        td.write_json(td.run_dir("B-T", lr) / "summary.json", {})
                    return rep.select("B-T", None)["selected"]
        self.assertEqual(run({"lr1e-5": 7.0, "lr3e-5": 6.5, "lr1e-4": 6.8}), 3e-5)
        self.assertEqual(run({"lr1e-5": 6.54, "lr3e-5": 6.5, "lr1e-4": 6.3}), 1e-4)
        self.assertEqual(run({"lr1e-5": 6.54, "lr3e-5": 6.5, "lr1e-4": 6.49}), 1e-5)   # within 0.05 of the best
        self.assertEqual(run({"lr1e-5": 6.57, "lr3e-5": 6.52, "lr1e-4": 6.51}), 3e-5)

    def test_selection_waits_for_all_three_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with mock.patch.object(td, "root", lambda: root):
                td.write_json(td.run_dir("T-0", 1e-5) / "summary.json", {})
                r = rep.select("T-0", None)
        self.assertIsNone(r["selected"])
        self.assertEqual(r["pending"], ["lr3e-5", "lr1e-4"])


class Claims(unittest.TestCase):
    def res(self, L, Lci, total, tci, set_totals):
        sets = {f"s{i}": {"total": {"point": t, "ci": [t - 1, t + 1]}} for i, t in enumerate(set_totals)}
        return {"mean_sets": list(sets), "sets": sets,
                "mean": {"L": {"point": L, "ci": Lci}, "total": {"point": total, "ci": tci}}}

    def test_claim_rules(self):
        c = rep.claims(self.res(0.2, [0.05, 0.3], 0.15, [0.02, 0.3], [0.1, 0.2, -0.2, 0.3]))
        self.assertTrue(c["labels_help_through_decoder"])
        self.assertTrue(c["decoder_only_candidate"])
        self.assertFalse(rep.claims(self.res(0.2, [-0.01, 0.3], 0.15, [0.02, 0.3], [0.1]))["labels_help_through_decoder"])
        self.assertFalse(rep.claims(self.res(0.2, [0.1, 0.3], 0.09, [0.02, 0.3], [0.1]))["decoder_only_candidate"])
        self.assertFalse(rep.claims(self.res(0.2, [0.1, 0.3], 0.15, [-0.01, 0.3], [0.1]))["decoder_only_candidate"])
        self.assertFalse(rep.claims(self.res(0.2, [0.1, 0.3], 0.15, [0.02, 0.3], [0.5, -0.26]))["decoder_only_candidate"])
        self.assertTrue(rep.claims(self.res(0.2, [0.1, 0.3], 0.15, [0.02, 0.3], [0.5, -0.25]))["decoder_only_candidate"])


if __name__ == "__main__":
    unittest.main()


class ContextAndCheckpoint(unittest.TestCase):
    """DecoderContext built as build_context builds it (inside the seam), on the CPU: the optimizer, the
    prediction/joint-only checkpoint, its restore, and the refusal of another encoder."""

    @classmethod
    def setUpClass(cls):
        from labeling_fixtures import relabel_root
        from training import labels as tlabels
        from training import replay as treplay
        from training_fixtures import LabeledWorld
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        w = LabeledWorld(cls.root)
        replay = treplay.JournalReplay(w.pool, 10)
        labels = tlabels.LabelStore.open(w.labels_dir, w.pool, label_set="T*", through_batch=10)
        cls.data = tseam.OfflineData(replay, labels, schedule.branch(schedule.trunk(1e-3, 100), 6, 4), 10)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def build(self, name, seed=0):
        model = tiny_model(21)
        td.configure(model)
        args = types.SimpleNamespace(**{k: None for k in train.RESUME_KEYS + td.RESUME_EXTRA})
        args.__dict__.update(arm="B-T", recipe="decoder-only", lr=1e-3, max_steps=self.data.plan["steps"], seed=seed,
                             label_set="T*", ckpt_minutes=30.0)
        run = self.root / name
        run.mkdir(parents=True, exist_ok=True)
        with td.seam("B-T"):
            return td.DecoderContext(args, run, model, self.data, "cpu", td.encoder_sha256(model))

    def test_context_uses_the_decoder_only_recipe_and_optimizer(self):
        c = self.build("ctx")
        self.assertEqual(c.recipe, td.RECIPE)
        self.assertIsNone(c.ramp)
        self.assertIsInstance(c.scheduler, tseam.ConstantScheduler)
        in_opt = {id(p) for g in c.optimizer.param_groups for p in g["params"]}
        self.assertEqual(in_opt, {id(p) for n, p in c.student.named_parameters() if n.startswith(("decoder.", "joint."))})

    def test_checkpoint_holds_prediction_and_joint_only_and_restores(self):
        a = self.build("ckpt")
        with torch.no_grad():
            for v in td.trained_state(a.student).values():
                if v.is_floating_point():
                    v.add_(0.5)
        a.state.step, a.state.batches_read, a.state.position = 3, 4, {"batch": 4}
        with td.seam("B-T"):
            train.save_checkpoint(a.run, a.checkpoint_payload())
        payload = train.load_checkpoint(a.run)
        self.assertTrue(payload["student"])
        self.assertTrue(all(k.startswith(("decoder.", "joint.")) for k in payload["student"]))
        self.assertEqual(payload["encoder_sha256"], a.encoder_sha256)
        b = self.build("ckpt")
        train.seed_everything(999, b.student)                 # restore must bring back the saved RNG
        with td.seam("B-T"):
            b.restore(payload)
        self.assertEqual(train.rng_state(b.student)["python"], payload["rng"]["python"])
        for k, v in td.trained_state(a.student).items():
            self.assertTrue(torch.equal(v, td.trained_state(b.student)[k]), k)
        self.assertEqual((b.state.step, b.state.position), (3, {"batch": 4}))
        payload["encoder_sha256"] = "0" * 64
        with td.seam("B-T"), self.assertRaises(RuntimeError):
            self.build("ckpt").restore(payload)

    def test_train_loop_skips_a_non_finite_step_and_finishes_on_the_plan(self):
        """train.train_loop through the seam (CPU; the loss is FakeLosses): a NaN gradient at the third step is
        skipped, the run still ends at the last planned batch with a step, the learning rate of every step is its
        planned batch's, and the metrics stay finite and leave the skipped step out."""
        c = self.build("loop")
        c.args.eval_every = c.args.max_steps
        lrs, label = [], tseam.label_batch

        def remember(ctx, batch):
            out = label(ctx, batch)
            if out is not None:
                lrs.append((int(batch["index"]), ctx.optimizer.param_groups[0]["lr"], out["seconds"]))
            return out
        with td.seam("B-T"), mock.patch.object(train, "compute_losses", FakeLosses(1.0, {3: "nan-grad"})), \
                mock.patch.object(train, "label_batch", remember):
            train.train_loop(c)
        self.assertEqual(c.state.step, c.args.max_steps)
        self.assertEqual(c.state.batches_read, self.data.stop)
        self.assertEqual(c.scheduler.steps, c.args.max_steps)
        self.assertEqual(c.state.drop_counts[tseam.NONFINITE_KEY], 1)
        self.assertEqual([lr for _, lr, _ in lrs], [self.data.schedule.lr(b) for b, _, _ in lrs])
        self.assertAlmostEqual(c.state.audio_seconds_seen, sum(s for _, _, s in lrs) - lrs[2][2], places=6)
        metrics = [json.loads(line) for line in (c.run / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual(metrics[-1]["drop_counts"][tseam.NONFINITE_KEY], 1)
        for m in metrics:
            self.assertTrue(all(math.isfinite(m[k]) for k in ("loss", "grad_norm", "audio_seconds_per_step")), m)


class Deadlines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lock = Path(self.tmp.name) / "gpu.lock"
        self.patch = mock.patch.object(td.relabel_paths, "GPU_LOCK", self.lock)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_lock_wait_is_bounded_by_the_deadline(self):
        import fcntl
        import time
        import score_teacher
        with open(self.lock, "a+") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            t0 = time.monotonic()
            with self.assertRaises(score_teacher.LockBusy):
                with td.bounded_lock(time.time() + 1.0)("x"):
                    pass
            self.assertLess(time.monotonic() - t0, 10)
        with td.bounded_lock(time.time() + 5.0)("x"):          # free: acquired
            pass

    def test_lock_timeout_and_budget_stop_end_the_run_with_success(self):
        import score_teacher
        argv = ["--arm", "B-T", "--lr", "1e-5", "--labels-run", "R", "--deadline", "1"]
        with mock.patch.object(td, "root", lambda: Path(self.tmp.name)), \
                mock.patch.object(td.relabel_paths, "require_mount", lambda: None):
            with mock.patch.object(train, "main", side_effect=score_teacher.LockBusy("held")):
                self.assertEqual(td.run_train(argv), 0)

            def stopped(reason):
                def main(_argv):
                    td._ACTIVE["ctx"] = types.SimpleNamespace(stop_reason=reason,
                                                              state=types.SimpleNamespace(step=1, batches_read=1))
                    raise SystemExit(train.EXIT_INTERRUPTED)
                return main

            with mock.patch.object(train, "main", stopped("budget")):
                self.assertEqual(td.run_train(argv), 0)
            with mock.patch.object(train, "main", stopped("SIGTERM")), self.assertRaises(SystemExit):
                td.run_train(argv)

    def test_score_original_gives_up_at_the_deadline(self):
        import score_teacher

        def busy(label, wait):
            raise score_teacher.LockBusy(f"waited {wait}")

        with mock.patch.object(score_teacher, "gpu_lock", busy), \
                mock.patch.object(td.relabel_paths, "require_mount", lambda: None), \
                mock.patch.object(td.devsplit, "load", lambda *a, **k: None), \
                mock.patch.object(td, "ensure_student", lambda s: {}):
            self.assertEqual(td.score_original("B", 3600.0, deadline=1.0), 0)
            with self.assertRaises(score_teacher.LockBusy):
                td.score_original("B", 0.0)
