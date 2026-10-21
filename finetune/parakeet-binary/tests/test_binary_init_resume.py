"""Contract tests 8-9 (plans/pilot.md "Contract tests"): M1 initialization and resume.

The pinned M1 checkpoint (7.3 GB) is replaced by a train.py-layout checkpoint of a tiny NeMo
model, with binary_paths' pinned path and SHA-256 pointed at it. Resume runs train.py's own
checkpoint code (Context.save, load_resume_payload, Context.restore) and the seam's build_context
on the CPU, with real optimizer updates on a stand-in loss (the TDT loss is CUDA-only, so
train.main and train_loop cannot run here; the real crash-and-resume path was exercised on the
GPU by the smoke runs, plans/pilot.md "Review round 1").
"""
from __future__ import annotations

import copy
import os
import shutil
import time
import unittest
from unittest import mock

import binary_fixtures as fx
import torch

import binary_paths
import binary_quant
import binary_train
import quant
import train

TRAIN_ARGS = ["--arm", "P2", "--run-name", "resume-test", "--max-steps", "8", "--stream",
              "local-librispeech", "--batch-seconds", "8", "--eval-every", "4", "--no-powerlog"]


def states_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def snapshot(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ----------------------------------------------------------------------------- contract 8

class InitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = fx.TEST_ROOT / "init"
        shutil.rmtree(cls.root, ignore_errors=True)
        cls.model = fx.tiny_model(seed=21)
        cls.m1_state = fx.perturbed_state(cls.model, seed=22)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self) -> None:
        cache = mock.patch.object(binary_paths, "CACHES", fx.TEST_ROOT / "sha256-cache")
        cache.start()
        self.addCleanup(cache.stop)

    def load(self, path, digest, **kw):
        state, info = binary_train.read_init_state(path, expected_sha256=digest, **kw)
        return state, info

    def test_loads_every_student_tensor(self) -> None:
        path = self.root / "good.pt"
        digest = fx.write_checkpoint(path, self.m1_state)
        student = fx.tiny_model(seed=23)
        state, info = self.load(path, digest)
        self.assertEqual((info["sha256"], info["weight_fraction"], info["step"]),
                         (digest, 1.0, binary_paths.M1_STEP))
        loaded = binary_train.apply_init_state(student, state)
        self.assertEqual(loaded, len(self.m1_state))
        self.assertTrue(states_equal(snapshot(student), self.m1_state))  # floats and buffers alike

    def test_refuses_a_different_sha256(self) -> None:
        path = self.root / "hash.pt"
        digest = fx.write_checkpoint(path, self.m1_state)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.load(path, "0" * 64)
        # The hash cache never vouches for changed contents: rewrite the file (new mtime).
        with fx.pinned_init(path, digest):
            self.assertEqual(binary_paths.cached_sha256(path), digest)
            time.sleep(0.01)
            other = dict(self.m1_state)
            key = next(k for k, v in other.items() if v.is_floating_point())
            other[key] = other[key] + 1.0
            fx.write_checkpoint(path, other)
            os.utime(path)
            self.assertNotEqual(binary_paths.cached_sha256(path), digest)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                self.load(path, digest)

    def test_refuses_missing_or_extra_keys(self) -> None:
        missing = dict(self.m1_state)
        missing.pop(next(iter(missing)))
        extra = {**self.m1_state, "encoder.extra_tensor": torch.zeros(3)}
        for name, state in (("missing", missing), ("extra", extra)):
            with self.subTest(name):
                path = self.root / f"{name}.pt"
                digest = fx.write_checkpoint(path, state)
                student = fx.tiny_model(seed=24)
                before = snapshot(student)
                with self.assertRaisesRegex(ValueError, name):
                    loaded, _ = self.load(path, digest)
                    binary_train.apply_init_state(student, loaded)
                self.assertTrue(states_equal(snapshot(student), before), "student changed by a refused load")

    def test_refuses_fraction_other_than_1(self) -> None:
        for fraction in (0.5, 0.0, None):
            with self.subTest(fraction=fraction):
                path = self.root / f"fraction-{fraction}.pt"
                digest = fx.write_checkpoint(path, self.m1_state, weight_fraction=fraction)
                with self.assertRaisesRegex(ValueError, "fraction"):
                    self.load(path, digest)

    def test_refuses_incomplete_or_other_step(self) -> None:
        path = self.root / "incomplete.pt"
        digest = fx.write_checkpoint(path, self.m1_state, complete=False)
        with self.assertRaises(ValueError):
            self.load(path, digest)
        path = self.root / "step.pt"
        digest = fx.write_checkpoint(path, self.m1_state, step=1000)
        with self.assertRaisesRegex(ValueError, "step"):
            self.load(path, digest)

    def test_pinned_constants(self) -> None:
        self.assertEqual(binary_paths.M1_SHA256,
                         "003b1289b19114b58327bebb61af598412bdaca408db47a6949f7951d62d212a")
        self.assertEqual(str(binary_paths.M1_CHECKPOINT), "/mnt/hd/wilderness-labs-stt/parakeet-ternary/"
                         "runs/main-M1-P2-lr5e-4/ckpt/latest.pt")
        self.assertEqual(binary_paths.M1_STEP, 250_000)


# ----------------------------------------------------------------------------- contract 9

class ResumeTest(unittest.TestCase):
    """An m1 arm checkpoints after its first update, is killed, and the same command resumes it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = fx.TEST_ROOT / "resume"
        shutil.rmtree(cls.root, ignore_errors=True)
        cls.fp32_state = snapshot(fx.tiny_model(seed=31))  # the "original" every start loads first
        cls.m1_state = fx.perturbed_state(fx.tiny_model(seed=31), seed=32)
        cls.m1_path = cls.root / "m1.pt"
        cls.m1_sha = fx.write_checkpoint(cls.m1_path, cls.m1_state)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self) -> None:
        self.stack = [fx.pinned_init(self.m1_path, self.m1_sha), fx.tiny_module_count(),
                      mock.patch.object(train, "prepare_tdt_loss", return_value="stub")]
        for cm in self.stack:
            cm.__enter__()

    def tearDown(self) -> None:
        for cm in reversed(self.stack):
            cm.__exit__(None, None, None)

    def fresh_student(self):
        student = fx.tiny_model(seed=33)
        student.load_state_dict(self.fp32_state)
        return student

    def args(self, init: str, quantizer: str, lr: str = "5e-4"):
        return binary_train.parse_args(["--init", init, "--quantizer", quantizer, "--lr", lr, *TRAIN_ARGS])

    def context(self, args, run, resuming: bool):
        """build_context's student and Context steps for the tiny model on the CPU."""
        student = self.fresh_student()
        quantizer, init = binary_train.prepare_student(args, student, resuming)
        recipe = train.recipe_of(args)
        train.configure_trainable(student, recipe)
        ctx = binary_train.make_context(args, run, student, None, None, None, quantizer, "cpu",
                                        label_fn=lambda *a, **k: None)
        return ctx, init

    @staticmethod
    def update(ctx, audio: torch.Tensor, lengths: torch.Tensor) -> None:
        """One real optimizer update in train_loop's order: the ramp fraction of the next 1-based
        update, train modes, forward (dither and SpecAugment draw from the RNGs), backward,
        clipping, AdamW step, schedule step. The loss stands in for the TDT loss (CUDA-only)."""
        k = ctx.state.step + 1
        ctx.quant.set_weight_fraction(ctx.student, train.ramp_weight_fraction(k, ctx.ramp))
        train.set_train_modes(ctx.student, ctx.recipe)
        # A copy: NeMo's dither adds its noise to the input tensor in place.
        enc, _ = ctx.student.forward(input_signal=audio.clone(), input_signal_length=lengths)
        ctx.optimizer.zero_grad(set_to_none=True)
        enc.pow(2).mean().backward()
        torch.nn.utils.clip_grad_norm_([p for g in ctx.optimizer.param_groups for p in g["params"]], 1.0)
        ctx.optimizer.step()
        ctx.scheduler.step()
        ctx.state.step = k

    def test_resume_restores_the_arm_checkpoint_not_m1_or_fp32(self) -> None:
        """Through the seam's build_context (train.main's order: load_resume_payload, then
        build_context, then Context.restore): a real update and checkpoint, a "kill", the same
        arguments again; the resume is detected from the run directory, M1 is not read, and the
        next update equals the uninterrupted run's bit for bit."""
        run = self.root / "run-B"
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir(parents=True)
        generator = torch.Generator().manual_seed(36)
        audio, lengths = 0.1 * torch.randn(2, 24000, generator=generator), torch.tensor([24000, 17000])
        with fx.stubbed_build(self.fp32_state, self.m1_path, self.m1_sha), binary_train.seam("binary"):
            args = self.args("m1", "binary")
            self.assertEqual((args.init, args.quantizer, args.ramp_kind), ("m1", "binary", "ternary"))
            self.assertEqual(args.init_sha256, self.m1_sha)
            key = train.resume_key(args)
            for name in ("init", "init_sha256", "quantizer", "ramp_kind", "quantizer_sha256"):
                self.assertIn(name, key)

            # Fresh start: no checkpoint, the student is M1's.
            self.assertIsNone(train.load_resume_payload(run, args))
            first, info = train.build_context(args, run, device="cpu")
            self.assertTrue(info["binary"]["init"]["loaded"])
            self.assertTrue(states_equal(snapshot(first.student), self.m1_state))
            self.assertIs(first.quant, binary_quant)
            start_signs = fx.sha256_file(run / binary_train.START_SIGNS)
            # One real update, then the first checkpoint; then the process dies.
            self.update(first, audio, lengths)
            first.save()
            trained = snapshot(first.student)
            optimizer = copy.deepcopy(first.optimizer.state_dict())
            self.assertTrue(optimizer["state"])

            # The same arguments again: train.main loads the checkpoint, build_context sees it.
            payload = train.load_resume_payload(run, args)
            self.assertEqual(payload["step"], 1)
            with mock.patch.object(binary_train, "read_init_state",
                                   side_effect=AssertionError("M1 read on resume")):
                second, info = train.build_context(args, run, device="cpu")
            self.assertFalse(info["binary"]["init"]["loaded"])
            self.assertNotIn("start_signs", info["binary"]["init"])
            self.assertEqual(fx.sha256_file(run / binary_train.START_SIGNS), start_signs)
            self.assertTrue(states_equal(snapshot(second.student), self.fp32_state))  # before restore
            second.restore(payload)
            self.assertEqual(second.state.step, 1)
            restored = snapshot(second.student)
            self.assertTrue(states_equal(restored, trained))
            self.assertFalse(states_equal(restored, self.m1_state))
            self.assertFalse(states_equal(restored, self.fp32_state))
            restored_optimizer = second.optimizer.state_dict()
            for i, state in optimizer["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(torch.as_tensor(value), torch.as_tensor(restored_optimizer["state"][i][name])))
            self.assertEqual(second.scheduler.state_dict(), first.scheduler.state_dict())
            self.assertEqual(binary_quant.weight_fraction(second.student),
                             train.ramp_weight_fraction(1, train.ramp_steps(args.max_steps)))
            self.assertEqual(second.timing["sessions"], 2)

            # Continuation: the resumed run's next update equals the uninterrupted run's.
            self.update(second, audio, lengths)
            train.set_rng_state(first.student, payload["rng"])
            self.update(first, audio, lengths)
            self.assertTrue(states_equal(snapshot(second.student), snapshot(first.student)))
            self.assertEqual(binary_quant.weight_fraction(second.student),
                             train.ramp_weight_fraction(2, train.ramp_steps(args.max_steps)))

            # Another init, quantizer, ramp or quantizer source is refused (train.py exit 78).
            others = {"init fp32": self.args("fp32", "binary"), "control C": self.args("m1", "ternary")}
            for field, value in (("init_sha256", "0" * 64), ("quantizer_sha256", "1" * 64),
                                 ("ramp_kind", "fp32"), ("quantizer", "ternary"), ("init", "fp32")):
                other = copy.deepcopy(args)
                setattr(other, field, value)
                others[field] = other
            for label, other in others.items():
                with self.subTest(label):
                    self.assertNotEqual(train.resume_key(other), payload["resume_key"])
                    with self.assertRaises(SystemExit) as caught:
                        train.load_resume_payload(run, other)
                    self.assertEqual(caught.exception.code, train.EXIT_RESUME_MISMATCH)

    def test_training_wall_time_is_persisted_across_a_resume(self) -> None:
        """Review round 1, finding 7: the training wall time up to each checkpoint, summed over
        sessions, travels in the checkpoint; time after the last checkpoint of a killed session
        is not counted."""
        run = self.root / "run-timing"
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir(parents=True)
        with fx.stubbed_build(self.fp32_state, self.m1_path, self.m1_sha), binary_train.seam("binary"), \
                mock.patch.object(binary_train, "clock") as clock:
            args = self.args("m1", "binary")
            clock.return_value = 100.0
            first, _ = train.build_context(args, run, device="cpu")
            first.state.step = 3
            clock.return_value = 160.0
            first.save()
            self.assertEqual(first.timing, {"train_wall_s": 60.0, "sessions": 1, "step": 3})
            clock.return_value = 170.0  # killed 10 s after the checkpoint
            payload = train.load_resume_payload(run, args)
            self.assertEqual(payload["binary_timing"]["train_wall_s"], 60.0)
            clock.return_value = 1000.0
            second, _ = train.build_context(args, run, device="cpu")
            second.restore(payload)
            self.assertEqual(second.timing, {"train_wall_s": 60.0, "sessions": 2, "step": 3})
            second.state.step = 5
            clock.return_value = 1030.0
            second.save()
            self.assertEqual(second.timing, {"train_wall_s": 90.0, "sessions": 2, "step": 5})
            self.assertEqual(train.load_checkpoint(run)["binary_timing"], second.timing)

    def test_control_c_first_update_is_m1_ternary_forward(self) -> None:
        """Arm C: the ternary quantizer unchanged, f = 1 from the first update, so its first
        forward is M1's ternary forward bit for bit (not lerp(W, T, 1/ramp))."""
        run = self.root / "run-C"
        shutil.rmtree(run, ignore_errors=True)
        run.mkdir(parents=True)
        with binary_train.seam("ternary"):
            args = self.args("m1", "ternary", lr="2e-4")
            self.assertEqual(args.ramp_kind, "none")
            ctx, init = self.context(args, run, resuming=False)
            self.assertIs(ctx.quant, quant)
            self.assertEqual(train.ramp_weight_fraction(1, ctx.ramp), 1.0)
            self.assertEqual(ctx.fraction(), 1.0)
            ctx.quant.set_weight_fraction(ctx.student, train.ramp_weight_fraction(1, ctx.ramp))
            reference = fx.tiny_model(seed=34)
            reference.load_state_dict(self.m1_state)
            quant.quantize_parakeet(reference)
            audio = 0.1 * torch.randn(2, 24000, generator=torch.Generator().manual_seed(35))
            lengths = torch.tensor([24000, 16000])
            ctx.student.eval()
            with torch.no_grad():
                got = ctx.student.forward(input_signal=audio, input_signal_length=lengths)[0]
                expected = reference.forward(input_signal=audio, input_signal_length=lengths)[0]
            self.assertTrue(torch.equal(got, expected))

    def test_ramp_policy_per_arm(self) -> None:
        with binary_train.seam("binary"):
            for init, ramp in (("fp32", "fp32"), ("m1", "ternary")):
                args = self.args(init, "binary")
                self.assertEqual(args.ramp_kind, ramp)
                run = self.root / f"policy-{init}"
                run.mkdir(parents=True, exist_ok=True)
                ctx, _ = self.context(args, run, resuming=False)
                self.assertEqual(ctx.ramp, train.ramp_steps(args.max_steps))
                self.assertEqual(binary_quant.quantized_module_names(ctx.student),
                                 sorted(n for n, _ in ctx.student.named_modules()
                                        if quant.QUANTIZED_PATTERN.fullmatch(n)))
        with self.assertRaises(SystemExit):  # (fp32, ternary) is the ternary pilot itself
            binary_train.parse_args(["--init", "fp32", "--quantizer", "ternary", "--lr", "5e-4", *TRAIN_ARGS])


if __name__ == "__main__":
    unittest.main()
