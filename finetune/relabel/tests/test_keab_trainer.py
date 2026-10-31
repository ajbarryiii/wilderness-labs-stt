"""Contract tests (CPU): the stage 4 trainer, train_keab.py (keab-prime plan "Training", "Step 1", "Step 2", "Code and
review"), on the stage 4 fixture world (keab_fixtures: real C1 shards and C2 labels, batch plan version 3 with a
synthetic union under the TS augmentation, a once source and AMI pairs, tiny NeMo models).

- runs, spans, seeds and schedules: six pilots (two arms x 2e-5, 6e-5, 2e-4) over pools 0-77 with a cooldown over
  66-77, two trunks through pool 663 with branch points at 332 and 664, b100 for both arms and b50 for the main arm;
  the arms' twins have one spec; one seed;
- the rules: the learning rate with the lowest mean H4 over the arms, near-ties within 0.05 to the lower rate; the
  gate (delta >= 0.10 with the interval excluding 0, no H4 set more than 0.25 worse, domain flags above 0.5); the
  MB1 check (0.02 per set);
- the teacher package: the main arm's student gets B0''s prediction and joint weights, bit-identical before 8-bit
  quantization, the control B0's; in the scope's 8-bit form after (the frozen parts fp32(W8) once); the teacher is
  frozen, in eval mode and never in the optimizer; the whole encoder trainable and nothing else; the activation
  quantizer on every binary module;
- encoder matching: the teacher encodes a real row's own audio, a synthetic row's TS-augmented waveform (a speed-
  perturbed row changes its length) and an AMI pair's IHM recording; unequal encoder frames are refused; a mixed
  batch with all three trains with equal teacher and student frame counts per row; teacher_encode is P2's teacher path;
- the label identity: cleaning version 3 in labels_sha256 for both arms, a checkpoint of another cleaning version
  (or scope, arm, teacher) refused on resume;
- arms matched: the same plan hash, seed, sources and settings, except the teacher package;
- batch plan version 3 production sources (the synthetic union of full, pairs and pairs-c, Common Voice once, AMI SDM
  seven times), the provisional union, step 0's scope, the probe's extremes with the AMI teacher-side metric, the
  chunk's order and commands, the seam and the training sources.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import keab_fixtures as kf
import train  # finetune/parakeet-ternary
import train_keab as tk
from decoder_fixtures import tiny_model
from test_b0prime import FakePlan
from training import actquant, label_sources, stage4
from training import eightbit as eb

import binary_quant  # noqa: E402


class Runs(unittest.TestCase):
    def test_the_eleven_runs_and_their_spans(self):
        R = stage4.RUNS
        self.assertEqual(stage4.PILOT, ("pilot-main-lr2e-5", "pilot-control-lr2e-5", "pilot-main-lr6e-5",
                                        "pilot-control-lr6e-5", "pilot-main-lr2e-4", "pilot-control-lr2e-4"))
        self.assertEqual(len(R), 11)
        for n in stage4.PILOT:
            r = R[n]
            self.assertEqual((r.kind, r.stop_pool, r.length_pools, r.cooldown), ("pilot", 78, 78, (66, 78)))
        self.assertEqual({R[n].lr for n in stage4.PILOT}, {2e-5, 6e-5, 2e-4})
        for arm in stage4.ARMS:
            t = R[f"trunk-{arm}"]
            self.assertEqual((t.stop_pool, t.length_pools, t.branch_points, t.lr), (664, 664, (332, 664), None))
            b = R[f"b100-{arm}"]
            self.assertEqual((b.branch_pool, b.cooldown, b.stop_pool, b.trunk), (664, (664, 782), 782, f"trunk-{arm}"))
        b50 = R["b50-main"]
        self.assertEqual((b50.branch_pool, b50.cooldown, b50.stop_pool), (332, (332, 391), 391))
        self.assertNotIn("b50-control", R)
        self.assertEqual({r.seed for r in R.values()}, {stage4.SEED})
        self.assertEqual(stage4.KEAB_PRIME, "b100-main")
        for n, r in R.items():
            twin = stage4.twin(n)
            if twin is not None:
                self.assertEqual(R[twin].spec(), r.spec())
                self.assertNotEqual(R[twin].arm, r.arm)
        self.assertIsNone(stage4.twin("b50-main"))
        self.assertEqual({R[n].label_set for n in R}, {"T*", "B0"})
        self.assertEqual({(R[n].arm, R[n].teacher) for n in R}, {("main", "B0p"), ("control", "B0")})
        # 2,091 pools: six 78-pool pilots, two 664-pool trunks, two 118-pool cooldowns and one 59-pool cooldown
        pools = 6 * 78 + 2 * 664 + 2 * 118 + 59
        self.assertEqual(pools, 2091)
        est = tk.gpu_hours(FakePlan(782), 0.6)
        self.assertEqual(est["pools"], 2091)

    def test_schedules(self):
        plan = FakePlan(782)
        B = plan.pool_start
        pilot = stage4.schedule_of(stage4.RUNS["pilot-main-lr6e-5"], plan, 6e-5)
        w = round(0.02 * B(78))
        self.assertEqual(pilot.warmup, w)
        self.assertAlmostEqual(pilot.lr(w), 6e-5)
        self.assertAlmostEqual(pilot.lr(B(66) - 1), 6e-5)
        self.assertAlmostEqual(pilot.lr(B(66)), 6e-5)
        self.assertEqual(pilot.lr(B(78) - 1), 0.0)
        self.assertEqual(stage4.stop_batch(stage4.RUNS["pilot-main-lr6e-5"], plan), B(78))
        trunk = stage4.schedule_of(stage4.RUNS["trunk-main"], plan, 6e-5)
        self.assertEqual(trunk.warmup, round(0.02 * B(664)))
        self.assertAlmostEqual(trunk.lr(B(663)), 6e-5)
        for name, (start, end) in (("b100-control", (664, 782)), ("b50-main", (332, 391))):
            b = stage4.schedule_of(stage4.RUNS[name], plan, 6e-5)
            self.assertAlmostEqual(b.lr(B(start)), 6e-5)
            self.assertEqual(b.lr(B(end) - 1), 0.0)
            self.assertEqual(stage4.stop_batch(stage4.RUNS[name], plan), B(end))
        self.assertEqual(stage4.branch_points(stage4.RUNS["trunk-control"], plan), {332: B(332), 664: B(664)})


class Rules(unittest.TestCase):
    def test_the_learning_rate_selection_is_on_the_mean_of_the_arms(self):
        h4 = {"pilot-main-lr2e-5": 6.30, "pilot-control-lr2e-5": 6.50, "pilot-main-lr6e-5": 6.10,
              "pilot-control-lr6e-5": 6.50, "pilot-main-lr2e-4": 6.00, "pilot-control-lr2e-4": 6.70}
        sel = stage4.select_lr(h4)                                         # means 6.40, 6.30, 6.35
        self.assertEqual(sel["selected"], 6e-5)                            # not 2e-4, whose main arm is best
        self.assertAlmostEqual(sel["mean_h4"]["lr6e-5"], 6.30)
        tie = dict(h4, **{"pilot-main-lr2e-5": 6.16})                      # means 6.33 vs 6.30: within 0.05
        self.assertEqual(stage4.select_lr(tie)["selected"], 2e-5)
        far = dict(h4, **{"pilot-main-lr2e-5": 6.21})                      # 6.355 vs 6.30: 0.055
        self.assertEqual(stage4.select_lr(far)["selected"], 6e-5)
        with self.assertRaises(ValueError):
            stage4.select_lr({k: v for k, v in h4.items() if "control" not in k})

    def test_the_gate(self):
        ok = stage4.gate({"point": 0.12, "ci": [0.02, 0.2]}, {"ami_dev": -0.3, "voxpopuli_dev": 0.25})
        self.assertTrue(ok["pass"])
        self.assertFalse(stage4.gate({"point": 0.09, "ci": [0.02, 0.2]}, {"a": 0.0})["pass"])
        self.assertFalse(stage4.gate({"point": 0.2, "ci": [0.0, 0.3]}, {"a": 0.0})["pass"])      # touches 0
        self.assertFalse(stage4.gate({"point": 0.2, "ci": [0.1, 0.3]}, {"a": 0.26})["pass"])
        flags = stage4.gate({"point": 0.2, "ci": [0.1, 0.3]}, {"a": 0.0}, {"dev-medical": 0.6, "dev-technical": 0.5})
        self.assertEqual(flags["domain_regressions_over_flag"], ["dev-medical"])
        self.assertTrue(flags["pass"])                                                      # a flag, not a failure

    def test_the_mb1_check(self):
        rec = {"a": 6.00, "b": 2.00}
        self.assertTrue(stage4.mb1_check({"a": 6.02, "b": 1.98}, rec)["pass"])
        self.assertFalse(stage4.mb1_check({"a": 6.03, "b": 2.0}, rec)["pass"])

    def test_the_export_check_bound(self):
        self.assertAlmostEqual(tk.fidelity_bound(0.05), 0.10)
        self.assertAlmostEqual(tk.fidelity_bound(0.004), 0.014)


class World(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = kf.World4()
        cls.h = kf.Harness4(cls.world)

    @classmethod
    def tearDownClass(cls):
        cls.world.close()


class TeacherPackage(World):
    def build(self, arm: str, scope: str):
        teacher = kf.teacher(arm)
        student, info = kf.student_factory()(tk.decoder_state(teacher), scope, False)
        return teacher, student, info

    def test_each_arm_gets_its_teachers_decoder_bit_for_bit_then_its_8bit_form(self):
        for arm in stage4.ARMS:
            teacher, student, info = self.build(arm, "none")
            want = tk.decoder_state(teacher)
            got = tk.decoder_state(student)
            self.assertEqual(set(got), set(want))
            self.assertTrue(all(torch.equal(got[k], want[k]) for k in want), arm)        # scope none: unchanged
            self.assertEqual(info["teacher_decoder_sha256"], tk.state_sha256(want))
            other = tk.decoder_state(kf.teacher("control" if arm == "main" else "main"))
            self.assertFalse(all(torch.equal(got[k], other[k]) for k in other))
            for scope in ("S1", "S2"):
                _, s8, info8 = self.build(arm, scope)
                self.assertEqual(info8["teacher_decoder_sha256"], tk.state_sha256(want))    # before 8 bits
                frozen = eb.frozen_int8(s8)
                self.assertEqual(sorted(frozen), sorted(eb.keys_of([p for p in eb.KEAB_SCOPES[scope]
                                                                    if p in eb.FROZEN_PARTS])))
                for key, (q, sc) in frozen.items():
                    wq, ws = eb.int8_quantize(want[key])                                  # the teacher's weight's codes
                    self.assertTrue(torch.equal(q, wq) and torch.equal(sc, ws))
                    self.assertTrue(torch.equal(s8.get_parameter(key).detach(), eb.w8(wq, ws).float()))
                rest = [k for k in want if k not in frozen]
                self.assertTrue(all(torch.equal(tk.decoder_state(s8)[k], want[k]) for k in rest))
                self.assertEqual(info8["form"]["scope"], scope)
                with self.assertRaises(ValueError):
                    eb.quantize_frozen(s8, ["prediction_network"])                         # quantized once

    def test_the_student_trains_its_whole_encoder_and_nothing_else(self):
        _, student, info = self.build("main", "S2")
        v = info["verified"]
        self.assertEqual(v["trainable"], v["encoder_trainable"])
        self.assertEqual(v["trainable"], sum(p.numel() for p in student.encoder.parameters()))
        self.assertEqual((v["decoder_trainable"], v["joint_trainable"], v["encoder_frozen"]), (0, 0, 0))
        self.assertTrue(student.encoder.pre_encode.out.weight.requires_grad)               # its latent weight (W8 STE)
        self.assertEqual(v["actquant"]["modules"], len(binary_quant.quantized_module_names(student)))
        self.assertTrue(all(student.get_submodule(n).fraction == 1.0 for n in binary_quant.quantized_module_names(student)))
        self.assertIsNotNone(student.spec_augmentation)
        student.decoder.prediction.embed.weight.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            tk.verify(student, "S2", None)
        _, s1, _ = self.build("control", "S1")
        with self.assertRaises(RuntimeError):
            tk.verify(s1, "S2", None)                                                       # another scope's form

    def test_the_student_starts_from_the_init_state(self):
        _, student, _ = self.build("main", "none")
        init = kf.init_state()
        enc = {k: v for k, v in student.state_dict().items() if k.startswith("encoder.")}
        self.assertTrue(all(torch.equal(enc[k], init[k]) for k in enc))

    def test_the_teacher_is_frozen_in_eval_mode_and_never_in_the_optimizer(self):
        ctx = self.h.context("pm", "S1")
        teacher_params = {id(p) for p in ctx.teacher.parameters()}
        opt_params = {id(p) for g in ctx.optimizer.param_groups for p in g["params"]}
        self.assertFalse(teacher_params & opt_params)
        self.assertFalse(any(p.requires_grad for p in ctx.teacher.parameters()))
        self.assertFalse(ctx.teacher.training)
        self.assertEqual(opt_params, {id(p) for p in ctx.student.parameters() if p.requires_grad})
        self.assertEqual(ctx.ramp, 0)
        self.assertEqual(ctx.fraction(), 1.0)
        self.assertEqual(ctx.args.teacher, kf.TEACHER_RECORDS["main"])
        self.assertEqual(ctx.info["keab"]["init"]["model"], "fixture")


class EncoderMatching(World):
    def batch_with_all_three(self):
        """The first planned batch of the version 3 plan with a real, a synthetic and an AMI row."""
        bt = label_sources.Batcher(self.world.replay3, self.world.arm_labels["main"], start=0,
                                   stop=self.world.replay3.through_batch)
        try:
            for _ in range(self.world.replay3.through_batch):
                b = bt.next(timeout=60)
                kinds = set(b["sources"])
                kept_pair = [r for r in kf.pair_rows(b) if b["labels"][r] is not None]
                if "synthetic" in kinds and kept_pair and any(not s.startswith(("synthetic", "cv27", "ami"))
                                                              for s in kinds):
                    return b
        finally:
            bt.close()
        self.skipTest("no planned batch with all three kinds of row")

    def capture_teacher(self, batch):
        seen, original = {}, tk.teacher_encode

        def encode(teacher, audio, lengths):
            seen.update(audio=audio.clone(), lengths=lengths.clone())
            return original(teacher, audio, lengths)
        ctx = types.SimpleNamespace(
            state=types.SimpleNamespace(batches_read=int(batch["index"]) + 1, acc={"dropped_rows": 0},
                                        drop_counts={}, seconds_by_source={}),
            stream=types.SimpleNamespace(schedule=types.SimpleNamespace(lr=lambda k: 1e-4)),
            optimizer=types.SimpleNamespace(param_groups=[{"lr": 0}]), device="cpu", student=tiny_model(1),
            teacher=kf.teacher("main"))
        with mock.patch.object(tk, "teacher_encode", encode):
            out = tk.label_batch(ctx, batch)
        return out, seen, ctx

    def test_the_teacher_encodes_real_audio_the_augmented_waveform_and_the_ihm_recording(self):
        b = self.batch_with_all_three()
        out, seen, ctx = self.capture_teacher(b)
        rows = [r for r, t in enumerate(b["labels"]) if t is not None]
        self.assertEqual(seen["audio"].shape, out["audio"].shape)              # the student's padded length
        ihm = {"ami-sdm:" + sid: kf.f32(h) for sid, _, _, h in self.world.pair_specs}
        sdm = {"ami-sdm:" + sid: kf.f32(s) for sid, _, s, _ in self.world.pair_specs}
        for j, r in enumerate(rows):
            i, n = b["ids"][r], int(b["audio_len"][r])
            got = seen["audio"][j]
            if b["sources"][r] == "ami-sdm":
                np.testing.assert_array_equal(got[:len(ihm[i])].numpy(), ihm[i])          # IHM, not the student's SDM
                np.testing.assert_array_equal(out["audio"][j, :n].numpy(), sdm[i])
                self.assertFalse(np.array_equal(ihm[i], sdm[i]))
                self.assertEqual(int(seen["lengths"][j]), len(ihm[i]))
            else:                                                                  # real, synthetic: the same waveform
                self.assertTrue(torch.equal(got, out["audio"][j]))
                self.assertEqual(int(seen["lengths"][j]), n)
            self.assertTrue(torch.equal(got[int(seen["lengths"][j]):], torch.zeros_like(got[int(seen["lengths"][j]):])))
        self.assertEqual(ctx.state.drop_counts[tk.TEACHER_ROWS_KEY], sum(b["sources"][r] == "ami-sdm" for r in rows))
        self.assertTrue(torch.equal(out["teacher_len"], tk.teacher_encode(ctx.teacher, out["audio"], out["audio_len"])[1]))

    def test_a_speed_perturbed_synthetic_row_is_served_and_encoded_at_its_new_length(self):
        r3 = self.world.replay3
        found = None
        for t in range(r3.through_batch):
            for row, item in enumerate(r3.read(t)):
                if item["source"] == "synthetic":
                    stored = r3._syn[(t, row)][2]
                    if len(item["audio"]) != stored:
                        found = (t, row, item, stored)
                        break
            if found:
                break
        self.assertIsNotNone(found, "the TS augmentation never changed a clip's length")
        t, row, item, stored = found
        self.assertLessEqual(len(item["audio"]), r3._syn[(t, row)][3])       # within the plan's key
        bt = label_sources.Batcher(r3, self.world.arm_labels["main"], start=t, stop=t + 1)
        try:
            b = bt.next(timeout=60)
        finally:
            bt.close()
        self.assertEqual(int(b["audio_len"][row]), len(item["audio"]))
        out, seen, _ = self.capture_teacher(b)
        j = [r for r, x in enumerate(b["labels"]) if x is not None].index(row)
        self.assertEqual(int(seen["lengths"][j]), len(item["audio"]))
        self.assertTrue(torch.equal(seen["audio"][j], out["audio"][j]))

    def test_unequal_encoder_frames_are_refused(self):
        b = dict(self.batch_with_all_three())
        r = [x for x in kf.pair_rows(b) if b["labels"][x] is not None][0]
        n = int(b["audio_len"][r])
        b["teacher_audio"] = {**b["teacher_audio"], r: np.zeros(max(1, n - 4000), np.float32)}
        with self.assertRaises(RuntimeError):
            self.capture_teacher(b)
        b["teacher_audio"] = {**b["teacher_audio"], r: np.zeros(b["audio"].shape[1] + 1, np.float32)}
        with self.assertRaises(RuntimeError):
            self.capture_teacher(b)

    def test_teacher_encode_is_p2s_teacher_path(self):
        import teacher as pt
        t = kf.teacher("control")
        audio = torch.randn(2, 20000)
        lens = torch.tensor([20000, 15000])
        enc, enc_len = tk.teacher_encode(t, audio, lens)
        _, _, enc2, len2 = pt.teacher_label_batch(t, audio, lens, [None, None], "bf16")
        self.assertTrue(torch.equal(enc, enc2) and torch.equal(enc_len, len2))
        self.assertEqual(enc.dtype, torch.float32)
        self.assertFalse(t.training)

    def test_a_mixed_batch_trains_with_equal_teacher_and_student_frames(self):
        """The loop over the plan's pilot (CPU: the encoder-matching term only, the TDT loss is CUDA-only): every
        step's teacher lengths equal the student's encoder lengths (the pinned check), AMI pairs included."""
        ctx = self.h.context("pm", "S1")
        losses = kf.EncoderLosses()
        seen = self.h.loop(ctx, compute_losses=losses)
        self.assertEqual(len(seen), ctx.args.max_steps)
        self.assertEqual(losses.calls, ctx.args.max_steps)
        self.assertGreater(ctx.state.drop_counts[tk.TEACHER_ROWS_KEY], 0)
        schedule = ctx.stream.schedule
        self.assertEqual([s["lr"] for s in seen], [schedule.lr(s["index"]) for s in seen])


class LabelIdentity(World):
    def cleaning_v3(self):
        from test_training_cleaning_v3 import LEX
        from training import cleaning
        from labeling import c2
        run = c2.load_run(self.world.w.labels_dir)
        import hashlib
        contract = hashlib.sha256(json.dumps(c2.contract(run), sort_keys=True).encode()).hexdigest()
        d = cleaning.production_dir("fixture", 3)                     # where the label run's lexicon lives
        if not (d / cleaning.LEXICON_MANIFEST).exists():
            cleaning.Lexicon.write(d, LEX, contract_sha256=contract, label_run="labels/fixture")
        return lambda run_, version: cleaning.Cleaning.load(version=version, lexicon_dir=d if version >= 3 else None)

    def test_cleaning_version_3_is_in_both_arms_labels_sha256(self):
        from training import cleaning
        with mock.patch.object(cleaning, "production", self.cleaning_v3()):
            v3 = {arm: self.world.open_arm(arm, 3) for arm in stage4.ARMS}
        v0 = self.world.arm_labels
        for arm in stage4.ARMS:
            self.assertEqual(v3[arm].identity["cleaning"]["version"], 3)
            self.assertIn("lexicon_sha256", v3[arm].identity["cleaning"])
            self.assertNotEqual(v3[arm].identity["sha256"], v0[arm].identity["sha256"])
        self.assertEqual(v3["main"].identity["cleaning"], v3["control"].identity["cleaning"])
        self.assertNotEqual(v3["main"].identity["sha256"], v3["control"].identity["sha256"])
        self.assertEqual(v3["control"].label_set, "B0")
        with self.assertRaises(ValueError):
            label_sources.cleaning_spec("fixture", 2, tk.CLEANING_VERSIONS)              # stage 4 reads version 3
        self.assertEqual(tk.CLEANING_VERSION, 3)
        self.assertNotIn(3, label_sources.CLEANING_VERSIONS)                           # stage 3's stay 0-2

    def test_a_checkpoint_of_another_cleaning_version_scope_arm_or_teacher_is_refused(self):
        run = kf.RUNS["pm"]
        args = self.h.args(run, self.h.data(run))
        with tk.seam():
            keys = set(train.RESUME_KEYS)
            self.assertTrue({"cleaning_version", "cleaning", "labels_sha256", "scope", "keab_arm", "teacher", "init",
                             "actquant", "eightbit", "plan_sha256", "extra_sources"} <= keys)
            recorded = vars(args)                                       # a checkpoint's "args"
            for key, value in (("cleaning_version", 2), ("scope", "S2"), ("keab_arm", "control"),
                               ("teacher", kf.TEACHER_RECORDS["control"]), ("labels_sha256", "x" * 64)):
                other = argparse.Namespace(**{**vars(args), key: value})
                self.assertTrue(train.resume_mismatches(recorded, other), key)
            self.assertFalse(train.resume_mismatches(recorded, args))


class ArmsMatched(World):
    def test_the_arms_differ_only_in_the_teacher_package(self):
        main = self.h.args(kf.RUNS["pm"], self.h.data(kf.RUNS["pm"]))
        ctrl = self.h.args(kf.RUNS["pc"], self.h.data(kf.RUNS["pc"]))
        self.assertEqual(tk.matched_identity(vars(main)), tk.matched_identity(vars(ctrl)))
        for key in ("plan_sha256", "replay_sha256", "seed", "extra_sources", "scope", "loss_seam", "actquant",
                    "schedule", "cleaning", "init"):
            self.assertEqual(json.dumps(vars(main)[key], sort_keys=True, default=str),
                             json.dumps(vars(ctrl)[key], sort_keys=True, default=str), key)
        differ = {k for k in vars(main) if json.dumps(vars(main)[k], sort_keys=True, default=str)
                  != json.dumps(vars(ctrl)[k], sort_keys=True, default=str)}
        # the run's name and spec (which names its arm), and the teacher package; the same steps (T*'s mask)
        self.assertEqual(differ, {"run_name", "spec", "keab_arm", "label_set", "labels_sha256", "teacher"})
        self.assertNotEqual(tk.arm_identity(vars(main)), tk.arm_identity(vars(ctrl)))

    def test_shared_identity_refuses_runs_of_other_settings(self):
        a = {**vars(self.h.args(kf.RUNS["pm"], self.h.data(kf.RUNS["pm"])))}
        b = {**vars(self.h.args(kf.RUNS["pc"], self.h.data(kf.RUNS["pc"])))}
        recs = {"pm": a, "pc": b}
        with mock.patch.object(tk, "recorded_args", lambda n: recs[n]), mock.patch.object(tk, "RUNS", kf.RUNS):
            ident = tk.shared_identity(["pm", "pc"])
            self.assertEqual(set(ident["arms"]), {"main", "control"})
            recs["pc"] = {**b, "scope": "S2"}
            with self.assertRaises(RuntimeError):
                tk.shared_identity(["pm", "pc"])


class Sources(unittest.TestCase):
    def test_the_production_sources(self):
        cfg = {c[0]: c for c in tk.EXTRA_SOURCES}
        self.assertEqual(list(cfg), ["synthetic", "cv27-en", "ami-sdm"])
        self.assertEqual(cfg["synthetic"][2].split()[2:], ["full", "pairs", "pairs-c"])
        self.assertEqual((cfg["synthetic"][3], cfg["synthetic"][4], cfg["synthetic"][6]), ("about:2", True, "as stored"))
        self.assertEqual((cfg["cv27-en"][3], cfg["cv27-en"][4], cfg["cv27-en"][6]), ("once", False, "layer 1"))
        self.assertEqual((cfg["ami-sdm"][3], cfg["ami-sdm"][4], cfg["ami-sdm"][6]), ("times:7", False, "pair"))
        self.assertEqual(cfg["synthetic"][5], 20261025)                          # stage 2c's TS seed
        self.assertEqual(cfg["cv27-en"][5], 20261029)                            # stage 3's

    def test_the_synthetic_union_needs_every_run_unless_provisional(self):
        from synthetic import mixing
        from training import ami_store
        loaded = []

        def load(run, **kw):
            if run == "pairs-c":
                raise FileNotFoundError("no clips.meta.json")
            loaded.append(run)
            return mixing.ClipPool([{"id": f"{run}-0", "duration": 1.0, "label": "x"}], identity=run)
        with mock.patch.object(mixing.ClipPool, "load", staticmethod(load)), \
                mock.patch.object(ami_store.PairStore, "load", staticmethod(lambda d: types.SimpleNamespace(
                    clips=[], identity="pairs"))), \
                mock.patch.object(tk, "check_ami_draws", lambda store: {}):
            with self.assertRaises(tk.NotReady):
                tk.source_stores()
            stores = tk.source_stores(provisional=True)
        self.assertTrue(tk.is_provisional(stores))
        self.assertEqual([p[0] for p in stores[0][1].parts], ["full", "pairs"])

    def test_the_ami_draws_agree_with_the_store(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(tk.llayout, "relabel_root", lambda: Path(d)):
            store = types.SimpleNamespace(identity="abc")
            with self.assertRaises(tk.NotReady):
                tk.check_ami_draws(store)
            p = Path(d) / tk.AMI_DRAWS_FILE
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps({"draws": 7, "store_clips_sha256": "abc"}))
            self.assertEqual(tk.check_ami_draws(store)["draws"], 7)
            p.write_text(json.dumps({"draws": 8, "store_clips_sha256": "abc"}))
            with self.assertRaises(RuntimeError):
                tk.check_ami_draws(store)

    def test_step0s_scope(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(tk, "step0_report", lambda: Path(d) / "r.json"):
            with self.assertRaises(tk.NotReady):
                tk.step0_scope()
            (Path(d) / "r.json").write_text(json.dumps({"decision": False, "limit": 50, "chosen_scope": None}))
            with self.assertRaises(tk.NotReady):
                tk.step0_scope()
            (Path(d) / "r.json").write_text(json.dumps({"decision": True, "limit": None, "chosen_scope": "S1"}))
            self.assertEqual(tk.step0_scope(), "S1")
        rec = tk.scope_record("S2")
        self.assertEqual((rec["pre_encode_out"], rec["binary_scales"]), ("w8_ste", "fp32(fp16(alpha))"))
        self.assertEqual(tk.scope_record("none")["parts"], [])


class Extremes(unittest.TestCase):
    def test_the_ami_teacher_side_metric(self):
        plan = types.SimpleNamespace(n_batches=3, shuffle=np.arange(3), order=np.arange(6),
                                     offsets=np.array([0, 2, 4, 6]), samples=np.array([100, 200, 300, 900, 50, 60]))
        ids = [["r1", "ami-sdm:a"], ["r2", "r3"], ["ami-sdm:b", "r4"]]
        replay = types.SimpleNamespace(batch_ids=lambda t: ids[t])
        labels = types.SimpleNamespace(text=lambda i: None if i == "r4" else "w " * (len(i)))
        res = tk.plan_extremes(plan, replay, labels, lambda texts: [len(t.split()) for t in texts],
                               {"ami-sdm:a": 250, "ami-sdm:b": 70}, k=2)
        self.assertEqual(res["top"]["ami_teacher_audio"], [0, 2])               # batches with a kept pair only
        self.assertEqual(res["values"][0]["ami_teacher_audio"], 2 * 250)
        self.assertEqual(res["values"][2]["ami_teacher_audio"], 1 * 70)        # r4 is rejected
        self.assertEqual(res["ami_batches"], 2)


class Chunk(unittest.TestCase):
    def test_order_through_and_commands(self):
        trains = [i[1] for i in tk.ORDER if i[0] == "train"]
        self.assertEqual(trains, list(stage4.PILOT) + ["trunk-main", "trunk-control", "b50-main", "b100-main",
                                                        "b100-control"])
        self.assertEqual(tk.THROUGH["pilot"], 6)
        self.assertEqual(tk.ORDER[:tk.THROUGH["main"]], [("train", n) for n in trains])
        cmd = tk.item_command(("score", "b100-main", "x1"), "R", 30.0, 1e9)
        self.assertEqual(cmd[:3], ["env", "-u", "CUBLAS_WORKSPACE_CONFIG"])
        self.assertTrue(cmd[3].endswith("python-gpu1bit"))
        self.assertTrue(tk.item_command(("score", "B0", "fp32"), "R", 30.0, 1e9)[0].endswith("/python"))
        train_cmd = tk.item_command(("train", "pilot-main-lr2e-5"), "R", 30.0, 1e9)
        self.assertIn("--labels-run", train_cmd)
        self.assertNotIn("--scope", train_cmd)                              # production runs read step 0's scope

    def test_the_chunk_runs_unfinished_items_until_one_fails(self):
        calls = []
        done = set()

        def runner(cmd, cwd=None):
            name = cmd[cmd.index("--run") + 1]
            calls.append(name)
            done.add(name)
            return types.SimpleNamespace(returncode=0 if len(calls) < 3 else 5)
        with mock.patch.object(tk, "item_done", lambda item: item[1] in done), \
                mock.patch.object(tk, "run_mismatch", lambda n, r: None), \
                mock.patch.object(tk.llayout, "require_mount", lambda: None):
            rc = tk.chunk("R", 120.0, "pilot", runner=runner)
        self.assertEqual(rc, 5)
        self.assertEqual(calls, list(stage4.PILOT[:3]))


class SeamAndSources(unittest.TestCase):
    def test_sources_exist_and_the_seam_installs_and_restores(self):
        finetune = Path(tk.__file__).resolve().parent.parent
        missing = [s for s in tk.SOURCES if not (finetune / s).exists()]
        self.assertEqual(missing, [])
        for f in ("relabel/terms/cleaning/variants_v3.tsv", "relabel/terms/cleaning/choices_v3.tsv",
                  "relabel/terms/cleaning/spelling_cues.tsv", "relabel/training/actquant.py"):
            self.assertIn(f, tk.SOURCES)
        before = (train.parse_args, train.build_context, train.label_batch, train.train_step, train.compute_losses,
                  train.RESUME_KEYS, train.TRAINING_SOURCES)
        with tk.seam():
            self.assertIs(train.label_batch, tk.label_batch)
            self.assertIs(train.train_step, tk.tseam.guarded_train_step)
            self.assertIs(train.parse_args, tk.parse_args)
            self.assertIn("tdt-fused/tdt_fused.py", train.TRAINING_SOURCES)
            self.assertTrue(set(tk.RESUME_EXTRA) <= set(train.RESUME_KEYS))
            self.assertNotEqual(train.compute_losses, before[4])               # the fused loss seam
        self.assertEqual(before, (train.parse_args, train.build_context, train.label_batch, train.train_step,
                                  train.compute_losses, train.RESUME_KEYS, train.TRAINING_SOURCES))

    def test_smoke_runs_never_write_production_paths(self):
        with mock.patch.dict(tk._ACTIVE, {"smoke": "s1"}):
            self.assertTrue(tk.run_dir("pilot-main-lr2e-5").is_relative_to(tk.smoke_root()))
            self.assertFalse(tk.run_dir("pilot-main-lr2e-5").is_relative_to(tk.root() / "runs"))
        self.assertEqual(tk.run_dir("x").parent, tk.root() / "runs")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            with mock.patch.object(tk, "step0_scope", lambda: "S1"):
                tk.parse_args(["--run", "pilot-main-lr2e-5", "--labels-run", "R", "--scope", "S2"])


if __name__ == "__main__":
    unittest.main()
