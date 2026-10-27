"""Contract tests (CUDA: the TDT loss is CUDA-only): stage 2b training through the pinned train.train_loop with
the decoder-only seam, on real C1 shards and C2 label records (training_fixtures) and a tiny model.

- the frozen encoder receives no gradient and is bit-identical after training; it is in eval mode at every step;
- only prediction and joint parameters are in the optimizer, and they change;
- the learning rate of every step is the WSD value of its consumed-batch index; a fully rejected batch is
  consumed without a step;
- the B0-label control and the T* arm train on the same utterances, with different texts;
- an interrupted and resumed run equals an uninterrupted one; checkpoints hold prediction and joint state only, and
  a checkpoint of another encoder is refused;
- the seam's train_step (the non-finite guard) trains bit for bit as the pinned train_step on finite steps.
"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import train  # finetune/parakeet-ternary
import train_decoder as td
from decoder_fixtures import tiny_model
from labeling_fixtures import relabel_root
from training import labels as tlabels
from training import replay as treplay
from training import schedule
from training import seam as tseam
from training_fixtures import REJECTED_BATCH, LabeledWorld

HAS_CUDA = torch.cuda.is_available()
STOP = 10
SCHEDULE = schedule.branch(schedule.trunk(2e-3, 100), start=6, cooldown=4)     # warmup 2, cooldown 6-9


@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.w = LabeledWorld(cls.root)
        cls.replay = treplay.JournalReplay(cls.w.pool, STOP)
        cls.labels = {s: tlabels.LabelStore.open(cls.w.labels_dir, cls.w.pool, label_set=s, through_batch=STOP)
                      for s in ("T*", "B0")}
        cls.init = tiny_model(7).state_dict()
        train.deterministic_mode()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def args(self, data, label_set="T*") -> argparse.Namespace:
        steps = data.plan["steps"]
        return argparse.Namespace(
            arm="B-T" if label_set == "T*" else "B-0", recipe="decoder-only", lr=SCHEDULE.peak, run_name="t",
            max_steps=steps, train="fixture", stream="c1-journal-replay", batch_seconds=10.0, grad_checkpointing=False,
            eval_every=steps, select="final", dev_limit=None, eval_batch_size=8, ckpt_minutes=30.0, seed=11,
            probe_steps=None, no_powerlog=True, overwrite=False, smoke=True, student="B", label_set=label_set,
            labels_run="fixture", labels_sha256=data.labels.identity["sha256"],
            cleaning=data.labels.identity.get("cleaning"), repetition=data.labels.identity.get("repetition"),
            plan_sha256=None, replay_sha256=data.replay.identity["sha256"], stop_batch=STOP, schedule=SCHEDULE.to_dict(),
            export_sha256="x", init_sha256="y", split_sha256="z", budget_minutes=None)

    def build(self, run: Path, label_set="T*", seed=None):
        data = tseam.OfflineData(self.replay, self.labels[label_set], SCHEDULE, STOP)
        args = self.args(data, label_set)
        model = tiny_model(7)
        model.load_state_dict(self.init)
        td.configure(model)
        train.seed_everything(args.seed if seed is None else seed, model)
        sha = td.encoder_sha256(model)
        model.cuda()
        run.mkdir(parents=True, exist_ok=True)
        with td.seam("B-T"):                       # the decoder-only recipe and the offline optimizer
            return td.DecoderContext(args, run, model, data, "cuda", sha)

    def run_loop(self, ctx, step_fn=None, **kw) -> list[dict]:
        """train.train_loop, recording at every optimizer step its batch index, lr, encoder modes, kept ids and
        texts. Each step is the seam's train_step (the guard), or step_fn."""
        seen = []
        with td.seam("B-T"):
            original = step_fn or train.train_step

        def step(c, labeled, trainable):
            last = c._last_batch
            seen.append({"index": last["index"], "lr": c.optimizer.param_groups[0]["lr"],
                         "lrs": {g["lr"] for g in c.optimizer.param_groups},
                         "encoder_training": sum(m.training for m in c.student.encoder.modules()),
                         "model_training": c.student.training, "spec_training": c.student.spec_augmentation.training,
                         "kept": [i for i, t in zip(last["ids"], last["labels"]) if t is not None],
                         "texts": [t for t in last["labels"] if t is not None]})
            return original(c, labeled, trainable)

        label = tseam.label_batch

        def remember(c, batch):
            c._last_batch = batch
            return label(c, batch)

        with td.seam("B-T"), mock.patch.object(train, "train_step", step), \
                mock.patch.object(train, "label_batch", remember):
            train.train_loop(ctx, **kw)
        return seen

    def test_frozen_encoder_only_decoder_trains(self):
        ctx = self.build(self.root / "runs" / "frozen")
        enc0 = {k: v.detach().clone() for k, v in ctx.student.encoder.state_dict().items()}
        dec0 = {k: v.detach().clone() for k, v in td.trained_state(ctx.student).items()}
        in_opt = {id(p) for g in ctx.optimizer.param_groups for p in g["params"]}
        self.assertEqual(in_opt, {id(p) for n, p in ctx.student.named_parameters() if n.startswith(("decoder.", "joint."))})
        seen = self.run_loop(ctx)
        self.assertEqual(ctx.state.step, ctx.args.max_steps)
        for k, v in ctx.student.encoder.state_dict().items():
            self.assertTrue(torch.equal(v, enc0[k]), f"encoder {k} changed")
        self.assertEqual(td.encoder_sha256(ctx.student), ctx.encoder_sha256)
        self.assertTrue(all(p.grad is None for p in ctx.student.encoder.parameters()))
        changed = [k for k, v in td.trained_state(ctx.student).items() if not torch.equal(v, dec0[k])]
        self.assertGreater(len(changed), 0)
        self.assertTrue(all(s["encoder_training"] == 0 and s["model_training"] and s["spec_training"] for s in seen))

    def test_learning_rate_per_consumed_batch_and_rejected_batch(self):
        ctx = self.build(self.root / "runs" / "lr")
        seen = self.run_loop(ctx)
        indices = [s["index"] for s in seen]
        self.assertEqual(indices, [b for b in range(STOP) if b != REJECTED_BATCH])
        for s in seen:
            self.assertEqual(s["lrs"], {SCHEDULE.lr(s["index"])})
        self.assertEqual(seen[0]["lr"], 0.0)
        self.assertEqual(seen[-1]["lr"], 0.0)                      # batch 9 = the cooldown's last batch
        self.assertEqual(ctx.state.batches_read, STOP)
        self.assertEqual(ctx.state.skipped_batches, 1)
        self.assertEqual(ctx.state.position, {"batch": STOP})

    def test_control_and_t_star_arm_train_on_the_same_utterances(self):
        a = self.run_loop(self.build(self.root / "runs" / "same-t", "T*"))
        b = self.run_loop(self.build(self.root / "runs" / "same-0", "B0"))
        self.assertEqual([s["index"] for s in a], [s["index"] for s in b])
        self.assertEqual([s["kept"] for s in a], [s["kept"] for s in b])
        self.assertNotEqual([s["texts"] for s in a], [s["texts"] for s in b])
        for s in b:
            self.assertTrue(all(t.startswith("b0 ") for t in s["texts"]))

    def test_the_guard_trains_as_the_pinned_step(self):
        pinned = train.train_step
        self.assertIsNot(pinned, tseam.guarded_train_step)
        a = self.build(self.root / "runs" / "pinned")
        b = self.build(self.root / "runs" / "guarded")
        self.run_loop(a, step_fn=pinned)
        self.run_loop(b)
        self.assertEqual(b.state.drop_counts[tseam.NONFINITE_KEY], 0)
        for k, v in td.trained_state(a.student).items():
            self.assertTrue(torch.equal(v, td.trained_state(b.student)[k]), k)
        keys = ("step", "loss", "tdt", "grad_norm", "lr", "audio_seconds_seen", "rows_per_step")
        metrics = [[{k: m[k] for k in keys} for m in map(json.loads, (c.run / "metrics.jsonl").read_text().splitlines())]
                   for c in (a, b)]
        self.assertEqual(metrics[0], metrics[1])
        self.assertGreater(len(metrics[0]), 0)

    def test_resume_equals_an_uninterrupted_run(self):
        full = self.build(self.root / "runs" / "full")
        ids_full = [s["index"] for s in self.run_loop(full)]
        run = self.root / "runs" / "resumed"
        first = self.build(run)
        ids = [s["index"] for s in self.run_loop(first, stop_after=4)]
        payload = train.load_checkpoint(run)
        self.assertTrue(all(k.startswith(("decoder.", "joint.")) for k in payload["student"]))
        self.assertEqual(payload["encoder_sha256"], first.encoder_sha256)
        second = self.build(run, seed=999)
        with td.seam("B-T"):                     # the run's resume key includes the seam's extra fields
            second.restore(payload)
        self.assertEqual(sum(m.training for m in second.student.encoder.modules()), 0)
        ids += [s["index"] for s in self.run_loop(second)]
        self.assertEqual(ids, ids_full)
        for k, v in td.trained_state(full.student).items():
            self.assertTrue(torch.equal(v, td.trained_state(second.student)[k]), k)
        for k, v in full.student.encoder.state_dict().items():
            self.assertTrue(torch.equal(v, second.student.encoder.state_dict()[k]))
        bad = copy.copy(payload)
        bad["encoder_sha256"] = "0" * 64
        with td.seam("B-T"), self.assertRaises(RuntimeError):
            self.build(run).restore(bad)


if __name__ == "__main__":
    unittest.main()
