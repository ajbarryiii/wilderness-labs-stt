"""Contract tests (CUDA: the TDT loss is CUDA-only): the stage 4 trainer through the pinned train.py with the stage 4
seam and the fused joint+TDT loss, on the stage 4 fixture world (keab_fixtures) and tiny NeMo models.

- a mixed batch with a speed-perturbed synthetic row, a real row and an AMI pair (teacher input: the IHM recording)
  runs through train.compute_losses with encoder matching: equal teacher and student frame counts per row (the
  pinned length check), a finite loss with a non-zero matching term, gradients in the encoder only (the prediction and
  joint networks and the teacher unchanged), the activation quantizer on every binary module's input;
- train.main of a pilot-like run through the seam: every planned batch's learning rate, max_steps optimizer steps,
  the teacher's input rows counted, model.pt and model.json written, the run scored (stubbed) and summarized; the
  8-bit form unchanged by training (the frozen parts stay fp32(W8) of their load-time codes);
- a smoke-like run writes its 8-bit bundle and scoring view (training/keab_export.py) from the CUDA model.
"""
from __future__ import annotations

import contextlib
import json
import shutil
import unittest
from unittest import mock

import torch

import keab_fixtures as kf
import train  # finetune/parakeet-ternary
import train_keab as tk
from training import actquant, label_sources
from training import eightbit as eb
from training import seam as tseam

HAS_CUDA = torch.cuda.is_available()


@contextlib.contextmanager
def nolock(label):
    yield                                                                       # the test job holds the GPU lock


def mixed_batch(world, labels):
    """A planned batch with a real row, a synthetic row whose TS augmentation changed its length, and a kept AMI pair."""
    r3 = world.replay3
    best = None
    bt = label_sources.Batcher(r3, labels, start=0, stop=r3.through_batch)
    try:
        for _ in range(r3.through_batch):
            b = bt.next(timeout=60)
            t = int(b["index"])
            syn = [row for row, s in enumerate(b["sources"]) if s == "synthetic"
                   and int(b["audio_len"][row]) != r3._syn[(t, row)][2]]
            pair = [row for row in kf.pair_rows(b) if b["labels"][row] is not None]
            real = [row for row, s in enumerate(b["sources"]) if s not in ("synthetic", "cv27-en", "ami-sdm")
                    and b["labels"][row] is not None]
            if syn and pair and real:
                return b
            if pair and real and best is None:
                best = b
    finally:
        bt.close()
    return best


@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class OnCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = kf.World4()
        cls.h = kf.Harness4(cls.world, "cuda")
        train.deterministic_mode()

    @classmethod
    def tearDownClass(cls):
        cls.world.close()

    def setUp(self):
        shutil.rmtree(tk.root() / "runs", ignore_errors=True)

    def test_a_mixed_batch_trains_with_equal_teacher_and_student_frames(self):
        ctx = self.h.context("pm", "S2")
        b = mixed_batch(self.world, self.world.arm_labels["main"])
        self.assertIsNotNone(b, "no planned batch with a real row and an AMI pair")
        ctx.state.batches_read = int(b["index"]) + 1
        trainable = [p for g in ctx.optimizer.param_groups for p in g["params"]]
        frozen = {k: v.detach().clone() for k, v in ctx.student.state_dict().items()
                  if k.startswith(("decoder.", "joint."))}
        teacher = {k: v.detach().clone() for k, v in ctx.teacher.state_dict().items()}
        hooks = {n: h[0] for n, h in actquant.hooks(ctx.student).items()}
        calls = {n: h.calls for n, h in hooks.items()}
        with tk.seam():
            train.set_train_modes(ctx.student, tk.RECIPE)
            labeled = tk.label_batch(ctx, b)
            self.assertGreater(ctx.state.drop_counts[tk.TEACHER_ROWS_KEY], 0)
            values = tseam.guarded_train_step(ctx, labeled, trainable)       # the pinned length check inside
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in values.values()), values)
        self.assertGreater(values["enc"], 0.0)
        self.assertTrue(all(hooks[n].calls > calls[n] for n in hooks))
        with torch.no_grad():
            feats, fl = ctx.student.preprocessor(input_signal=labeled["audio"], length=labeled["audio_len"])
            _, enc_len = ctx.student.encoder(audio_signal=feats, length=fl)
        self.assertTrue(torch.equal(enc_len, labeled["teacher_len"]))       # equal frames per row
        after = ctx.student.state_dict()
        self.assertTrue(all(torch.equal(after[k], v) for k, v in frozen.items()))
        self.assertTrue(all(torch.equal(ctx.teacher.state_dict()[k], v) for k, v in teacher.items()))
        eb.keab_form(ctx.student)

    def test_train_main_runs_a_pilot_through_the_seam(self):
        run = kf.RUNS["pm"]
        data = self.h.data(run)
        args = self.h.args(run, data, "S1")
        scored = []

        def fake_score(net, out, source, device, label, groups, limit=None):
            scored.append({"label": label, "training": net.training, "groups": groups, "sha256": source["sha256"]})
            return {"dev_wer": {n: 0.05 for n in tk.DEV_SETS}, "dev_utterances": {n: 1 for n in tk.DEV_SETS},
                    "h4": 5.0, "sets": list(tk.DEV_SETS)}
        lrs = []
        original = tk.label_batch

        def remember(c, batch):
            out = original(c, batch)
            if out is not None:
                lrs.append((int(batch["index"]), c.optimizer.param_groups[0]["lr"]))
            return out
        patches = self.h.patches() + [mock.patch.object(tk, "parse_args", lambda argv=None: args),
                                      mock.patch.object(tk, "score_torch", fake_score),
                                      mock.patch.object(tk, "label_batch", remember)]
        with tk.seam(), kf.contextlib_all(patches + [mock.patch.object(tk.paths, "gpu_lock", nolock)]):
            train.parse_args = tk.parse_args                                   # the seam installed the original
            train.label_batch = tk.label_batch
            train.main(["--run", "pm", "--labels-run", "fixture"])
        d = tk.run_dir("pm")
        summary = json.loads((d / "summary.json").read_text())
        self.assertEqual(summary["steps"], args.max_steps)
        self.assertEqual(len(lrs), args.max_steps)
        self.assertEqual([lr for _, lr in lrs], [data.schedule.lr(k) for k, _ in lrs])
        self.assertGreater(summary["keab"]["teacher_audio_rows"], 0)
        self.assertEqual(summary["keab"]["nonfinite_skipped_steps"], 0)
        self.assertTrue((d / "model.pt").exists())
        manifest = json.loads((d / "model.json").read_text())
        self.assertEqual((manifest["arm"], manifest["scope"]), ("main", "S1"))
        self.assertNotIn("export", manifest)                                    # a pilot is not exported
        self.assertEqual(json.loads((d / "scores.json").read_text())["h4"], 5.0)
        self.assertEqual(len(scored), 1)
        self.assertFalse(scored[0]["training"])
        cfg = json.loads((d / "config.json").read_text())
        self.assertEqual(cfg["args"]["scope"], "S1")
        self.assertEqual(cfg["args"]["keab_arm"], "main")
        self.assertEqual(cfg["recipe"], tk.RECIPE)
        payload = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        student, _ = kf.student_factory()(tk.decoder_state(kf.teacher("main")), "S1", True)
        student.load_state_dict(payload["student"], strict=True)
        eb.keab_form(student)                                                   # still fp32(W8) of the same codes

    def test_a_smoke_writes_its_bundle_and_scoring_view(self):
        ctx = self.h.context("pc", "S2", smoke=True)
        with tk.seam(), kf.contextlib_all(self.h.patches()):
            ctx.state.step = ctx.args.max_steps
            manifest = tk.save_model(ctx)
        d = tk.run_dir("pc")
        self.assertIn("export", manifest)
        self.assertTrue((d / "export" / "bundle" / "manifest.json").exists())
        self.assertTrue((d / "export" / "view" / "export.safetensors").exists())
        bundle = json.loads((d / "export" / "bundle" / "manifest.json").read_text())
        self.assertEqual(bundle["scope"], "S2")


if __name__ == "__main__":
    unittest.main()
