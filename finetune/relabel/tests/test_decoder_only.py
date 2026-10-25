"""Contract tests (CPU): stage 2b's arms, frozen-encoder modes, trainable set, decoder-step proxy, chunk order,
learning-rate selection and claims (decoder-only plan)."""
from __future__ import annotations

import json
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
