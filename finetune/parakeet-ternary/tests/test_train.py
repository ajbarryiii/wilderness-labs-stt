"""Training-loop tests: schedules, bucketing, stream position, losses, checkpoints, resume.

CPU tests cover the pure logic and the Batcher (on a synthetic in-memory resumable
stream). GPU tests (skipped without CUDA; the NeMo TDT loss is CUDA-only) build a tiny
EncDecRNNTBPEModel from the pinned checkpoint's own config and tokenizer (2 conformer
layers, d_model 64), quantize it with the real quant.py (its 264-module count check
patched to the tiny model's 22), label batches online with the real
teacher.teacher_label_batch (a tiny teacher), and run the real train_loop.

Resume semantics under test: after interrupting at step k and restoring into a fresh
model with scrambled RNGs, the step counter, optimizer, scheduler and RNG states equal
the checkpointed ones, the stream continues from the saved position (the batch ids
consumed after resume are exactly the uninterrupted run's), and because the synthetic
stream is deterministic the final parameters are bit-identical as well.

Run: finetune/parakeet-ternary/python -m unittest discover -s finetune/parakeet-ternary/tests -v
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import paths
import train

HAS_CUDA = torch.cuda.is_available()
TEST_ROOT = paths.ARTIFACTS / "tmp" / "test-train"
WORDS = ("the quick brown fox jumps over a lazy dog while seven pilots check their "
         "tourniquet airway breathing circulation and hypothermia lists").split()


def make_args(**overrides) -> argparse.Namespace:
    argv = ["--arm", overrides.pop("arm", "P2"), "--lr", str(overrides.pop("lr", 1e-3)),
            "--run-name", "test", "--max-steps", str(overrides.pop("max_steps", 6)),
            "--stream", "local-librispeech", "--batch-seconds",
            str(overrides.pop("batch_seconds", 8.0)), "--eval-every",
            str(overrides.pop("eval_every", 2)), "--no-powerlog"]
    args = train.parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class SyntheticStream:
    """Deterministic, resumable, infinite stream of noise utterances (the stream contract).

    Item i of epoch e is a pure function of (seed, i); the order is a seeded per-epoch
    permutation. fail_after raises ConnectionError once that many items were yielded by
    this object (a network failure after the stream's own retries).
    """

    def __init__(self, n: int = 40, seed: int = 0, fail_after: int | None = None,
                 durations: tuple[float, float] = (1.0, 3.5), human_text: bool = False) -> None:
        self.n, self.seed, self.fail_after, self.human = n, seed, fail_after, human_text
        self.low, self.high = durations
        self.epoch, self.position, self.yielded = 0, 0, 0

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "position": self.position, "seed": self.seed}

    def load_state_dict(self, state: dict) -> None:
        assert state["seed"] == self.seed
        self.epoch, self.position = state["epoch"], state["position"]

    def item(self, i: int) -> dict:
        rng = np.random.default_rng([self.seed, i])
        seconds = float(rng.uniform(self.low, self.high))
        audio = (rng.standard_normal(int(seconds * 16000)) * 0.1).astype(np.float32)
        text = " ".join(rng.choice(WORDS, 4)) if self.human else None
        return {"audio": audio, "duration": len(audio) / 16000, "text": text,
                "id": f"syn:{i}", "source": "alpha" if i % 3 else "beta"}

    def __iter__(self):
        while True:
            order = np.random.default_rng([self.seed, 99, self.epoch]).permutation(self.n)
            while self.position < self.n:
                if self.fail_after is not None and self.yielded >= self.fail_after:
                    raise ConnectionError("synthetic network failure")
                i = int(order[self.position])
                self.position += 1
                self.yielded += 1
                yield self.item(i)
            self.epoch, self.position = self.epoch + 1, 0


def drain(batcher: train.Batcher, k: int) -> list[dict]:
    out = [batcher.next(timeout=60) for _ in range(k)]
    batcher.close()
    return out


# ----------------------------------------------------------------------------- CPU tests

class ScheduleTest(unittest.TestCase):
    def test_lr_warmup_and_linear_decay(self) -> None:
        warm = train.warmup_steps(1000)
        self.assertEqual(warm, 20)  # 2%
        f = [train.lr_factor(s, warm, 1000) for s in range(1000)]
        self.assertAlmostEqual(f[0], 1 / 20)
        self.assertEqual(f[19], 1.0)  # peak on update 20
        self.assertAlmostEqual(f[999], 1 / 980)
        self.assertEqual(train.lr_factor(1000, warm, 1000), 0.0)
        self.assertTrue(all(b <= a for a, b in zip(f[19:], f[20:])))

    def test_ramp(self) -> None:
        ramp = train.ramp_steps(1000)
        self.assertEqual(ramp, 250)  # 25%
        self.assertEqual(train.ramp_weight_fraction(1, ramp), 1 / 250)
        self.assertEqual(train.ramp_weight_fraction(250, ramp), 1.0)
        self.assertEqual(train.ramp_weight_fraction(999, ramp), 1.0)
        with self.assertRaises(ValueError):
            train.ramp_weight_fraction(0, ramp)
        self.assertTrue(train.selectable(1.0) and train.selectable(None))
        self.assertFalse(train.selectable(0.999))


class ArgsTest(unittest.TestCase):
    def test_recipes(self) -> None:
        self.assertEqual(make_args(arm="P1").recipe, "P1")
        self.assertEqual(train.recipe_of(make_args(arm="P3")),
                         {"encoder_matching": True, "train_pred_joint": True, "quantized": True})
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            make_args(arm="M1")  # M1 needs --recipe
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            train.parse_args(["--arm", "P1", "--recipe", "P2", "--lr", "1", "--run-name", "x",
                              "--max-steps", "5"])
        args = train.parse_args(["--arm", "A1", "--recipe", "P2", "--train", "pilot", "--lr",
                                 "1e-4", "--run-name", "x", "--max-steps", "10"])
        self.assertFalse(train.recipe_of(args)["quantized"])
        smoke = train.parse_args(["--arm", "P1", "--lr", "1e-4", "--run-name", "x", "--smoke"])
        self.assertEqual((smoke.max_steps, smoke.eval_every, smoke.dev_limit), (20, 10, 16))
        self.assertEqual(train.parse_args(["--arm", "M1", "--recipe", "P1", "--lr", "1", "--run-name",
                                           "x", "--max-steps", "100"]).train, "main")
        self.assertEqual(make_args().stream, "local-librispeech")
        self.assertEqual(train.parse_args(["--arm", "P1", "--lr", "1", "--run-name", "x",
                                           "--max-steps", "5"]).stream, "hub")


class BucketingTest(unittest.TestCase):
    def test_cut_pool(self) -> None:
        durations = [5.0, 1.0, 9.0, 2.0, 2.5, 50.0, 3.0]
        batches = train.cut_pool(durations, 10.0)
        flat = sorted(i for b in batches for i in b)
        self.assertEqual(flat, list(range(len(durations))))
        for b in batches:
            d = [durations[i] for i in b]
            self.assertTrue(len(d) * max(d) <= 10.0 or len(d) == 1)
            self.assertEqual(d, sorted(d))
        self.assertIn([5], batches)  # the 50 s utterance alone

    def test_batches_deterministic_and_bounded(self) -> None:
        a = drain(train.Batcher(SyntheticStream(), 8.0, seed=1), 20)
        b = drain(train.Batcher(SyntheticStream(), 8.0, seed=1), 20)
        self.assertEqual([x["ids"] for x in a], [x["ids"] for x in b])
        self.assertNotEqual([x["ids"] for x in a],
                            [x["ids"] for x in drain(train.Batcher(SyntheticStream(), 8.0, seed=2), 20)])
        for x in a:
            longest = int(x["audio_len"].max())
            self.assertTrue(len(x["ids"]) * longest / 16000 <= 8.0 or len(x["ids"]) == 1)
            self.assertEqual(tuple(x["audio"].shape), (len(x["ids"]), longest))
            for row, n in enumerate(x["audio_len"].tolist()):
                self.assertTrue(torch.all(x["audio"][row, n:] == 0))

    def test_resume_from_any_position_continues_exactly(self) -> None:
        reference = drain(train.Batcher(SyntheticStream(), 8.0, seed=1), 30)
        for k in (1, 5, 7, 8, 13):  # inside pools and at pool boundaries
            position = reference[k - 1]["position_after"]
            resumed = drain(train.Batcher(SyntheticStream(), 8.0, seed=1,
                                          position=copy.deepcopy(position)), 30 - k)
            self.assertEqual([x["ids"] for x in resumed], [x["ids"] for x in reference[k:]], k)
            self.assertTrue(all(torch.equal(x["audio"], y["audio"])
                                for x, y in zip(resumed, reference[k:])))

    def test_prefilter_drops_out_of_range(self) -> None:
        stream = SyntheticStream(durations=(0.2, 40.0))
        batcher = train.Batcher(stream, 60.0, seed=0)
        batches = drain(batcher, 5)
        for x in batches:
            seconds = (x["audio_len"].float() / 16000).tolist()
            self.assertTrue(all(1.0 <= s <= 30.0 for s in seconds))
        self.assertGreater(sum(batcher.prefiltered.values()), 0)

    def test_stream_failure_surfaces_after_queued_batches(self) -> None:
        batcher = train.Batcher(SyntheticStream(fail_after=70), 8.0, seed=0)
        got = 0
        with self.assertRaises(train.StreamFailure):
            for _ in range(1000):
                batcher.next(timeout=60)
                got += 1
        batcher.close()
        self.assertGreater(got, 0)

    def test_local_librispeech_stream_resumes(self) -> None:
        stream = train.LocalLibriSpeechStream(seed=3, limit=50)
        it = iter(stream)
        first = [next(it)["id"] for _ in range(5)]
        state = stream.state_dict()
        rest = [next(it)["id"] for _ in range(5)]
        again = train.LocalLibriSpeechStream(seed=3, limit=50)
        again.load_state_dict(state)
        self.assertEqual([x["id"] for _, x in zip(range(5), iter(again))], rest)
        self.assertEqual(len(set(first + rest)), 10)
        concat = train.LocalLibriSpeechStream(seed=3, concat_seconds=29.5, limit=60)
        longs = [x for _, x in zip(range(5), iter(concat))]
        self.assertTrue(all(x["duration"] <= 29.5 for x in longs))
        self.assertTrue(any("+" in x["id"] and x["duration"] > 15.0 for x in longs))


class EncoderMatchingTest(unittest.TestCase):
    def test_masked_normalized_mse(self) -> None:
        g = torch.Generator().manual_seed(0)
        s = torch.randn(3, 8, 10, generator=g)
        t = torch.randn(3, 8, 10, generator=g) * 2 + 1
        lengths = torch.tensor([10, 4, 7])
        loss = train.encoder_matching_loss(s, t, lengths)
        sv = torch.cat([s[b, :, :n].T for b, n in enumerate(lengths.tolist())])
        tv = torch.cat([t[b, :, :n].T for b, n in enumerate(lengths.tolist())])
        expected = ((sv - tv) ** 2).mean() / tv.var(unbiased=False)
        torch.testing.assert_close(loss, expected)
        s2, t2 = s.clone(), t.clone()
        s2[1, :, 4:] = 1e6  # padding frames are ignored
        t2[2, :, 7:] = -1e6
        self.assertEqual(train.encoder_matching_loss(s2, t2, lengths).item(), loss.item())
        self.assertEqual(train.encoder_matching_loss(t, t, lengths).item(), 0.0)

    def test_gradient_only_to_student(self) -> None:
        s = torch.randn(2, 4, 5, requires_grad=True)
        t = torch.randn(2, 4, 5, requires_grad=True)
        train.encoder_matching_loss(s, t, torch.tensor([5, 3])).backward()
        self.assertIsNotNone(s.grad)
        self.assertIsNone(t.grad)
        self.assertTrue(torch.all(s.grad[1, :, 3:] == 0))


class ParamGroupTest(unittest.TestCase):
    def test_decay_on_matrices_only(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3),
                                    torch.nn.Conv1d(3, 3, 3, groups=3))
        model[1].requires_grad_(False)
        decay, no_decay = train.param_groups(model)
        self.assertEqual([p.shape for p in decay["params"]], [(3, 4), (3, 1, 3)])
        self.assertEqual([p.shape for p in no_decay["params"]], [(3,), (3,)])
        self.assertEqual((decay["weight_decay"], no_decay["weight_decay"]), (0.01, 0.0))


class CheckpointFileTest(unittest.TestCase):
    def setUp(self) -> None:
        paths.require_mount()
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.run = Path(tempfile.mkdtemp(dir=TEST_ROOT))

    def tearDown(self) -> None:
        shutil.rmtree(self.run, ignore_errors=True)

    def test_rotation_and_fallback(self) -> None:
        self.assertIsNone(train.load_checkpoint(self.run))
        train.save_checkpoint(self.run, {"step": 1})
        train.save_checkpoint(self.run, {"step": 2})
        latest, previous, tmp = train.ckpt_paths(self.run)
        self.assertEqual(train.load_checkpoint(self.run)["step"], 2)
        self.assertEqual(torch.load(previous, weights_only=False)["step"], 1)
        latest.write_bytes(latest.read_bytes()[:50])  # torn latest
        tmp.write_bytes(b"partial")
        self.assertEqual(train.load_checkpoint(self.run)["step"], 1)
        self.assertFalse(tmp.exists())
        self.assertEqual(len(list(latest.parent.glob("latest.pt.unreadable-*"))), 1)  # quarantined
        torch.save({"step": 9}, latest)  # no completion marker
        self.assertEqual(train.load_checkpoint(self.run)["step"], 1)

    def test_unreadable_checkpoints_refuse_to_restart(self) -> None:
        latest, previous, _ = train.ckpt_paths(self.run)
        latest.parent.mkdir(parents=True)
        latest.write_bytes(b"torn")
        previous.write_bytes(b"also torn")
        with self.assertRaises(train.CheckpointUnreadable):
            train.load_checkpoint(self.run)
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as exit_:
            train.check_checkpoints_readable(self.run)
        self.assertEqual(exit_.exception.code, train.EXIT_UNREADABLE_CHECKPOINT)
        self.assertTrue(latest.exists() and previous.exists())  # nothing discarded
        train.check_checkpoints_readable(self.run / "elsewhere")  # no files: fresh start is fine

    def test_prune_evidence_after_resume(self) -> None:
        run = self.run
        lines = [{"step": s, "loss": 1.0, "session": 0} for s in range(20, 101, 20)]
        lines.insert(3, {"step": 60, "eval": "dev-subset", "session": 0})
        text = "".join(json.dumps(r) + "\n" for r in lines) + '{"step": 120, "lo'  # torn
        (run / "metrics.jsonl").write_text(text)
        for name in ("a", "b"):
            for step in (50, 60, 70):
                train.write_json(run / "dev-subset" / name / f"step-{step:06d}.json", {"step": step})
        (run / "dev-subset" / "a" / "step-000080.json.tmp").write_text("x")
        for name in ("best-000040.pt", "best-000080.pt", "best-000090.pt.tmp"):
            (run / name).write_text("x")
        size = train.prune_evidence(run, 60, "best-000040.pt")
        kept = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([(r["step"], r.get("eval")) for r in kept],
                         [(20, None), (40, None), (60, None), (60, "dev-subset")])
        self.assertEqual(size, (run / "metrics.jsonl").stat().st_size)
        self.assertEqual(sorted(p.name for p in (run / "dev-subset").rglob("*")
                                if p.is_file()), ["step-000050.json"] * 2 + ["step-000060.json"] * 2)
        self.assertEqual(sorted(p.name for p in run.glob("best-*")), ["best-000040.pt"])
        self.assertFalse((run / "metrics.jsonl.tmp").exists())

    def test_torn_log_lines(self) -> None:
        path = self.run / "resumes.jsonl"
        path.write_text('{"step": 1}\n{"step": 2}\n{"step": 3, "trai')
        records, torn = train.read_log_jsonl(path)
        self.assertEqual(([r["step"] for r in records], torn), ([1, 2], True))
        train.record_resume(self.run, {"step": 4})  # drops the torn line, then appends
        self.assertEqual(path.read_text(), '{"step": 1}\n{"step": 2}\n{"step": 4}\n')
        path.write_text('{"step": 1}\n{"st\n{"step": 2}\n')  # damage in the middle
        with self.assertRaises(ValueError):
            train.read_log_jsonl(path)
        metrics = self.run / "metrics.jsonl"
        metrics.write_text('{"step": 20}\n{"step": 40, "lo')
        size = train.append_jsonl(metrics, {"step": 40})  # repairs before appending
        self.assertEqual(metrics.read_text(), '{"step": 20}\n{"step": 40}\n')
        self.assertEqual(size, metrics.stat().st_size)
        self.assertEqual(train.read_log_jsonl(self.run / "absent.jsonl"), ([], False))
        # a resume prunes a torn resumes.jsonl too
        path.write_text('{"step": 1}\n{"step": 2')
        train.prune_evidence(self.run, 100, None)
        self.assertEqual(path.read_text(), '{"step": 1}\n')

    def test_resume_argument_check(self) -> None:
        args = make_args(arm="P2", max_steps=6)
        self.assertEqual(train.resume_mismatches(vars(args), args), [])
        allowed = make_args(arm="P2", max_steps=6, ckpt_minutes=5.0, eval_batch_size=8)
        self.assertEqual(train.resume_mismatches(vars(args), allowed), [])
        for change in ({"lr": 5e-4}, {"grad_checkpointing": True}, {"batch_seconds": 300.0},
                       {"select": "best"}, {"eval_every": 3}, {"stream": "hub"}):
            self.assertTrue(train.resume_mismatches(vars(args), make_args(
                arm="P2", max_steps=6, **change)), change)
        train.save_checkpoint(self.run, {"step": 3, "args": vars(args),
                                         "resume_key": train.resume_key(args)})
        self.assertEqual(train.load_resume_payload(self.run, allowed)["step"], 3)
        with self.assertRaises(SystemExit) as exit_:
            train.load_resume_payload(self.run, make_args(arm="P2", max_steps=6, lr=5e-4))
        self.assertEqual(exit_.exception.code, train.EXIT_RESUME_MISMATCH)
        self.assertIsNone(train.load_resume_payload(self.run / "fresh", args))

    def test_recipe_flags_by_value(self) -> None:
        for arm, flags in train.ARM_FLAGS.items():
            self.assertEqual(train.recipe_of(make_args(arm=arm)), flags)
        self.assertEqual(train.ARM_FLAGS["P1"], {"encoder_matching": False,
                                                 "train_pred_joint": False, "quantized": True})
        m1 = train.parse_args(["--arm", "M1", "--recipe", "P3", "--lr", "1", "--run-name", "x",
                               "--max-steps", "5"])
        self.assertEqual(train.recipe_of(m1), {**train.RECIPES["P3"], "quantized": True})
        a1 = train.parse_args(["--arm", "A1", "--recipe", "P1", "--train", "pilot", "--lr", "1",
                               "--run-name", "x", "--max-steps", "5"])
        self.assertFalse(train.recipe_of(a1)["quantized"])
        with mock.patch.dict(train.RECIPES, {"P1": {"encoder_matching": True,
                                                    "train_pred_joint": False}}), \
                self.assertRaises(ValueError):
            train.recipe_of(make_args(arm="P1"))

    def test_drop_page_cache(self) -> None:
        path = self.run / "big.bin"
        path.write_bytes(b"x" * (1 << 20))
        train.drop_page_cache(path)
        train.drop_page_cache(self.run / "missing")  # best effort, never raises
        self.assertEqual(path.stat().st_size, 1 << 20)

    def test_atomic_json(self) -> None:
        train.write_json(self.run / "a" / "x.json", {"k": 1})
        self.assertEqual(json.loads((self.run / "a" / "x.json").read_text()), {"k": 1})
        self.assertEqual(list((self.run / "a").iterdir()), [self.run / "a" / "x.json"])


class ModuleGuardTest(unittest.TestCase):
    def test_refuses_whisper_modules(self) -> None:
        with self.assertRaises(ImportError):
            train.local_module("powerlog")  # exists only in finetune/whisper-ternary
        self.assertEqual(Path(train.local_module("quant").__file__).parent, paths.HERE)

    def test_powerlog_launcher_targets_parakeet_power_dir(self) -> None:
        self.assertIn("powerlog.POWER_DIR = paths.POWER", train.POWERLOG_LAUNCH)
        import powerlog  # the Whisper module, imported with this experiment's paths
        self.assertEqual(powerlog.paths.ARTIFACTS, paths.ARTIFACTS)

    def test_numba_cuda_environment(self) -> None:
        import os
        self.assertEqual(os.environ["NUMBA_FORCE_CUDA_CC"], "9.0")
        self.assertTrue(Path(os.environ["CUDA_HOME"], "nvvm", "lib64").exists())


class SweepTest(unittest.TestCase):
    """sweep.py: retries, validation, selection and the stop rule (no GPU, no training)."""

    def setUp(self) -> None:
        import sweep
        self.sweep = sweep
        self.plan = sweep.Plan(steps=100, batch_seconds=600.0, grad_checkpointing=True)
        self.sources = {name: f"h-{name}" for name in sweep.TRAINING_SOURCES}
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(dir=TEST_ROOT))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def summary(self, recipe="P2", lr="5e-4", wer=0.1, **kw) -> dict:
        s = {"run_name": self.sweep.run_name(recipe, lr), "arm": recipe, "recipe": recipe,
             "lr": float(lr), "max_steps": 100, "steps": 100, "select": "final",
             "selected_step": 100, "eval_every": 10, "grad_checkpointing": True,
             "source_hashes": dict(self.sources), "quantized_modules_sha256": "q",
             "dev_subset_mean_wer": wer, "dev_mean_wer": wer,
             "dev_wer": {n: wer for n in paths.DEV_SETS}, "seed": paths.SEED,
             "batch_seconds": 600.0, "train": "pilot",
             "stream": {"kind": "hub", "phase": "pilot", "seed": paths.SEED, "config_hash": "abc",
                        "stream_py_sha256": "s", "teacher_py_sha256": "t"},
             "quantized": True, "encoder_matching": True, "train_pred_joint": False,
             "ramp_steps": 25, "warmup_steps": 2, "audio_seconds_seen": 1.0, "smoke": False,
             "dev_limit": None, "started_utc": "x", "finished_utc": "y"}
        s.update(kw)
        return s

    def test_retries_stream_failures_with_backoff(self) -> None:
        codes, sleeps = [75, 75, 0], []
        runner = lambda cmd: argparse.Namespace(returncode=codes.pop(0))  # noqa: E731
        self.sweep.run_train(["x"], False, sleep=sleeps.append, runner=runner)
        self.assertEqual(sleeps, [60.0, 120.0])
        with self.assertRaises(RuntimeError):  # interrupted: never auto-restarted
            self.sweep.run_train(["x"], False, sleep=sleeps.append,
                                 runner=lambda cmd: argparse.Namespace(returncode=76))
        with self.assertRaises(RuntimeError):  # a crash is not retried
            self.sweep.run_train(["x"], False, sleep=sleeps.append,
                                 runner=lambda cmd: argparse.Namespace(returncode=1))
        with mock.patch.object(self.sweep, "MAX_STREAM_RETRIES", 2), self.assertRaises(RuntimeError):
            self.sweep.run_train(["x"], False, sleep=lambda s: None,
                                 runner=lambda cmd: argparse.Namespace(returncode=75))

    def expected(self, **reference) -> object:
        return self.sweep.Expected(self.plan, name="pilot-P2-lr5e-4", arm="P2", recipe="P2",
                                   lr="5e-4", steps=100, train="pilot", reference=reference)

    def config(self, **args_change) -> dict:
        args = {**self.expected().args(), **args_change}
        return {"args": args, "training_source_hashes": dict(self.sources),
                "recipe": {"encoder_matching": True, "train_pred_joint": False, "quantized": True},
                "trainable": {"trainable_modules": ["encoder"]},
                "stream": self.summary()["stream"], "quantized_modules_sha256": "q"}

    def test_validate_summary(self) -> None:
        validate = self.sweep.validate_summary
        self.assertEqual(validate(self.summary(), self.expected(), self.sources), [])
        for change in ({"steps": 90}, {"lr": 1e-3}, {"smoke": True}, {"batch_seconds": 300.0},
                       {"dev_limit": 16}, {"seed": 1}, {"dev_mean_wer": "0.1"},
                       {"grad_checkpointing": False}, {"eval_every": 20}, {"select": "best"},
                       {"selected_step": 60}, {"recipe": "P3"}):
            self.assertTrue(validate(self.summary(**change), self.expected(), self.sources), change)
        changed = dict(self.sources, **{"parakeet-ternary/train.py": "new"})
        self.assertTrue(any("train.py changed" in p
                            for p in validate(self.summary(), self.expected(), changed)))
        for flags in ({"encoder_matching": False}, {"train_pred_joint": True},
                      {"quantized": False}):  # P2 must be matching, frozen, quantized
            self.assertTrue(any("recipe flag" in p for p in validate(
                self.summary(**flags), self.expected(), self.sources)), flags)
        other = dict(self.summary()["stream"], config_hash="zzz")
        self.assertTrue(validate(self.summary(), self.expected(stream=other), self.sources))
        self.assertTrue(validate(self.summary(), self.expected(quantized_modules_sha256="r"),
                                 self.sources))

    def test_validate_config_of_started_runs(self) -> None:
        run = self.tmp / "run"
        run.mkdir()
        (run / "config.json").write_text(json.dumps(self.config()))
        self.assertEqual(self.sweep.validate_config(run, self.expected(), self.sources), [])
        for change in ({"eval_every": 5}, {"select": "best"}, {"grad_checkpointing": False},
                       {"lr": 1e-3}, {"max_steps": 50}, {"stream": "local-librispeech"}):
            (run / "config.json").write_text(json.dumps(self.config(**change)))
            self.assertTrue(self.sweep.validate_config(run, self.expected(), self.sources), change)
        (run / "config.json").write_text(json.dumps(self.config()))
        good = {"step": 5, "training_source_hashes": dict(self.sources),
                "args": self.expected().args()}
        (run / "resumes.jsonl").write_text(json.dumps(good) + "\n" + '{"step": 9, "tra')  # torn
        self.assertEqual(self.sweep.validate_config(run, self.expected(), self.sources), [])
        (run / "resumes.jsonl").write_text('{"step": 3, "tr\n' + json.dumps(good) + "\n")
        self.assertTrue(any("damaged" in p for p in
                            self.sweep.validate_config(run, self.expected(), self.sources)))
        edited = dict(good, training_source_hashes=dict(
            self.sources, **{"whisper-ternary/wer.py": "edited"}))
        (run / "resumes.jsonl").write_text(json.dumps(edited) + "\n")
        problems = self.sweep.validate_config(run, self.expected(), self.sources)
        self.assertTrue(any("resume 1: whisper-ternary/wer.py changed" in p for p in problems))
        (run / "resumes.jsonl").write_text(json.dumps(dict(good, args=dict(
            good["args"], lr=1e-3))) + "\n")
        self.assertTrue(any("resume 1 args lr" in p for p in
                            self.sweep.validate_config(run, self.expected(), self.sources)))
        (run / "resumes.jsonl").unlink()
        for bad in ({"recipe": {"encoder_matching": False, "train_pred_joint": False,
                                "quantized": True}},
                    {"trainable": {"trainable_modules": ["encoder", "decoder", "joint"]}}):
            (run / "config.json").write_text(json.dumps({**self.config(), **bad}))
            self.assertTrue(self.sweep.validate_config(run, self.expected(), self.sources), bad)
        (run / "config.json").write_text(json.dumps(self.config()))
        # ensure_run refuses to relaunch a started run made under other code/settings
        started = self.tmp / "pilot-P2-lr5e-4"
        started.mkdir()
        (started / "config.json").write_text(json.dumps(self.config(eval_every=5)))
        with mock.patch.object(self.sweep.paths, "RUNS", self.tmp), \
                mock.patch.object(self.sweep, "current_sources", return_value=self.sources), \
                mock.patch.object(self.sweep, "run_train") as launch, \
                self.assertRaises(RuntimeError):
            self.sweep.ensure_run(self.plan, "P2", "5e-4", False)
        launch.assert_not_called()

    def test_pilot_refuses_to_start_without_valid_b0(self) -> None:
        with mock.patch.object(self.sweep, "B0_DEV_SUMMARY", self.tmp / "missing.json"), \
                mock.patch.object(self.sweep, "ensure_run") as ensure, \
                self.assertRaises(SystemExit):
            self.sweep.phase_pilot(self.plan, False)
        ensure.assert_not_called()
        bad = self.tmp / "partial.json"
        sets = {n: {"wer": 5.0, "limit": None} for n in paths.DEV_SETS}
        sets[paths.DEV_SETS[0]]["limit"] = 100
        bad.write_text(json.dumps({"source": {"kind": "pretrained"}, "sets": sets}))
        with mock.patch.object(self.sweep, "B0_DEV_SUMMARY", bad), \
                mock.patch.object(self.sweep, "ensure_run") as ensure, \
                self.assertRaises(SystemExit):
            self.sweep.phase_pilot(self.plan, False)
        ensure.assert_not_called()
        sets[paths.DEV_SETS[0]]["limit"] = None
        bad.write_text(json.dumps({"source": {"kind": "export"}, "sets": sets}))
        with mock.patch.object(self.sweep, "B0_DEV_SUMMARY", bad), self.assertRaises(SystemExit):
            self.sweep.phase_pilot(self.plan, False)

    def test_commands(self) -> None:
        cmd = self.sweep.train_command(self.plan, "P3", "1e-3", "pilot-P3-lr1e-3")
        self.assertEqual(cmd[cmd.index("--max-steps") + 1], "100")
        self.assertIn("--grad-checkpointing", cmd)
        self.assertEqual(cmd[cmd.index("--select") + 1], "final")
        self.assertEqual(cmd[cmd.index("--eval-every") + 1], "10")
        self.assertNotIn("--overwrite", cmd)
        main = self.sweep.train_command(self.plan, "M1", "5e-4", "main-M1-P2-lr5e-4", "main",
                                        "P2", 20000)
        self.assertEqual(main[main.index("--recipe") + 1], "P2")
        self.assertEqual(self.sweep.run_name("M1", "5e-4", "main", "P2"), "main-M1-P2-lr5e-4")

    def test_source_list_covers_reused_modules(self) -> None:
        self.assertEqual(self.sweep.TRAINING_SOURCES, train.TRAINING_SOURCES)
        for name in ("parakeet-ternary/data.py", "parakeet-ternary/paths.py",
                     "whisper-ternary/quant.py", "whisper-ternary/wer.py"):
            self.assertIn(name, self.sweep.TRAINING_SOURCES)
        self.assertNotIn("whisper-ternary/powerlog.py", self.sweep.TRAINING_SOURCES)
        self.assertEqual(self.sweep.current_sources(), train.training_source_hashes())
        flags = {arm: self.sweep.Expected(self.plan, name="x", arm=arm, recipe=arm, lr="1",
                                          steps=1, train="pilot").flags() for arm in ("P1", "P2", "P3")}
        self.assertEqual(flags, train.ARM_FLAGS)
        a1 = self.sweep.Expected(self.plan, name="x", arm="A1", recipe="P2", lr="1", steps=1,
                                 train="pilot").flags()
        self.assertFalse(a1["quantized"])

    def test_heavy_unit_detection(self) -> None:
        inside = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/parakeet-pilot.service\n"
        outside = "0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-t3.scope\n"
        self.assertEqual(self.sweep.heavy_unit(inside), "parakeet-pilot.service")
        self.assertIsNone(self.sweep.heavy_unit(outside))
        with mock.patch.object(self.sweep, "heavy_unit", return_value=None), \
                self.assertRaises(SystemExit):
            self.sweep.main(["--phase", "pilot", "--dry-run"])

    def test_design_must_record_main_steps(self) -> None:
        text = "M1: the selected recipe, 41,250 steps."
        self.assertTrue(self.sweep.design_records_steps(41250, text))
        self.assertFalse(self.sweep.design_records_steps(4125, text))
        self.assertFalse(self.sweep.design_records_steps(41250, "41250 steps"))  # no M1

    def test_b0_and_selection_and_stop_rule(self) -> None:
        b0 = self.tmp / "b0.json"
        sets = {n: {"wer": 5.0, "limit": None} for n in paths.DEV_SETS}
        b0.write_text(json.dumps({"source": {"kind": "pretrained"}, "sets": sets}))
        with mock.patch.object(self.sweep, "B0_DEV_SUMMARY", b0):
            self.assertAlmostEqual(self.sweep.b0_dev_mean()["dev_mean_wer"], 0.05)
        runs = {("P1", "5e-4"): 0.12, ("P2", "5e-4"): 0.10, ("P3", "5e-4"): 0.11,
                ("P2", "2e-4"): 0.13, ("P2", "1e-3"): 0.09}
        launched = []

        def fake_ensure(plan, arm, lr, dry, **kw):
            launched.append((arm, lr))
            return self.summary(arm, lr, runs.get((arm, lr), 0.2))

        out = self.tmp / "selection.json"
        with mock.patch.object(self.sweep, "ensure_run", fake_ensure), \
                mock.patch.object(self.sweep, "B0_DEV_SUMMARY", b0), \
                mock.patch.object(self.sweep, "SELECTION", out):
            selection = self.sweep.phase_pilot(self.plan, False)
            self.assertEqual(launched, [("P1", "5e-4"), ("P2", "5e-4"), ("P3", "5e-4"),
                                        ("P2", "2e-4"), ("P2", "1e-3")])
            self.assertEqual((selection["recipe"], selection["lr"]), ("P2", "1e-3"))
            self.assertFalse(selection["stop_rule"]["every_pilot_run_above"])
            self.assertEqual(json.loads(out.read_text())["run"], "pilot-P2-lr1e-3")
            runs.update({k: 0.2 for k in runs})  # all above 3 x 5%
            with self.assertRaises(SystemExit):
                self.sweep.phase_pilot(self.plan, False)
            self.assertTrue(json.loads(out.read_text())["stop_rule"]["every_pilot_run_above"])


# ----------------------------------------------------------------------------- GPU fixtures

def tiny_base_dir() -> Path:
    """Tokenizer files and config from the pinned .nemo, extracted once."""
    out = TEST_ROOT / "tiny-base"
    if not (out / "done").exists():
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(paths.MODEL_FILE, "r:*") as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if member.isfile() and not name.endswith(".ckpt"):
                    (out / name).write_bytes(tar.extractfile(member).read())
        (out / "done").write_text("ok")
    return out


def tiny_model():
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import OmegaConf, open_dict
    base = tiny_base_dir()
    cfg = OmegaConf.load(base / "model_config.yaml")
    refs = {k: str(cfg.tokenizer[k]) for k in ("model_path", "vocab_path", "spe_tokenizer_vocab")}
    with open_dict(cfg):
        cfg.tokenizer.dir = str(base)
        for key, ref in refs.items():
            cfg.tokenizer[key] = str(base / ref.removeprefix("nemo:"))
        for ds in ("train_ds", "validation_ds", "test_ds"):
            cfg.pop(ds, None)
        cfg.encoder.update(n_layers=2, d_model=64, n_heads=2, subsampling_conv_channels=16)
        cfg.model_defaults.update(enc_hidden=64, pred_hidden=32, joint_hidden=32)
        cfg.decoder.prednet.update(pred_hidden=32, pred_rnn_layers=2)
        cfg.joint.jointnet.update(joint_hidden=32, encoder_hidden=64, pred_hidden=32)
        cfg.decoding.greedy.use_cuda_graph_decoder = False
    model = EncDecRNNTBPEModel(cfg=cfg)
    with open_dict(model.cfg):  # export.py looks the tokenizer up in the .nemo by these names
        for key, ref in refs.items():
            model.cfg.tokenizer[key] = ref
    return model.float()


class StubEvaluator:
    """Dev evaluation stand-in that runs the model and consumes every RNG (to prove that
    evaluation is isolated from the training trajectory). WER is a deterministic function
    of the weights, so best-checkpoint selection is exercised."""

    def __init__(self) -> None:
        import evaluate
        self.evaluate = evaluate
        root = TEST_ROOT / "stub-dev"
        root.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        records = []
        stream = SyntheticStream(seed=77)
        for i in range(4):
            item = stream.item(i)
            path = root / f"{i}.wav"
            if not path.exists():
                sf.write(path, item["audio"], 16000, subtype="FLOAT")
            records.append({"audio_filepath": str(path), "duration": item["duration"],
                            "text": "x", "id": f"dev:{i}"})
        self.subsets = {"dev_a": records[:2], "dev_b": records[2:]}

    def subset(self, model) -> dict:
        random.random(), np.random.rand(), torch.rand(3), torch.rand(3, device="cuda")
        audio, lengths = self.evaluate.audio_batch(self.subsets["dev_a"] + self.subsets["dev_b"])
        enc, _ = model.forward(input_signal=audio.cuda(), input_signal_length=lengths.cuda())
        value = float(enc.float().abs().mean().item())
        return {name: {"wer": value * (j + 1), "utterances": len(r), "records": []}
                for j, (name, r) in enumerate(self.subsets.items())}

    def full(self, model) -> dict:
        return self.subset(model)


@unittest.skipUnless(HAS_CUDA, "TDT loss is CUDA-only")
class GpuTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        paths.require_mount()
        train.deterministic_mode()
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        cls.root = Path(tempfile.mkdtemp(dir=TEST_ROOT))
        torch.manual_seed(123)
        model = tiny_model()
        cls.init_state = copy.deepcopy(model.state_dict())
        cls.quant = train.local_module("quant")
        cls.teacher_mod = train.local_module("teacher")
        cls.n_quant = 2 * len(cls.quant.PER_LAYER)
        cls.teacher = train.load_teacher(cls.fresh_model_cls()).cuda()

    @classmethod
    def fresh_model_cls(cls):
        model = tiny_model()
        model.load_state_dict(cls.init_state)
        return model

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def build(self, args, run: Path, scramble: int | None = None, stream=None,
              label_fn=None, evaluator=None) -> train.Context:
        """main()'s build_context for the tiny model (same order of seeding and setup)."""
        recipe = train.recipe_of(args)
        student = self.fresh_model_cls()
        train.seed_everything(args.seed if scramble is None else scramble, student)
        quant = None
        if recipe["quantized"]:
            quant = self.quant
            with mock.patch.object(quant, "EXPECTED_MODULES", self.n_quant):
                quant.quantize_parakeet(student)
        train.configure_trainable(student, recipe)
        student.cuda()
        if args.grad_checkpointing:
            train.enable_grad_checkpointing(student.encoder)
        run.mkdir(parents=True, exist_ok=True)
        return train.Context(args, run, student, self.teacher, stream or SyntheticStream(),
                             evaluator or StubEvaluator(), quant, "cuda", label_fn=label_fn)

    def snapshot(self, ctx: train.Context) -> dict:
        return {"model": {k: v.detach().cpu().clone() for k, v in ctx.student.state_dict().items()},
                "optim": copy.deepcopy(ctx.optimizer.state_dict()),
                "sched": copy.deepcopy(ctx.scheduler.state_dict()),
                "metrics": [{k: v for k, v in json.loads(line).items()
                             if not k.endswith("_s") and k not in (
                                 "session", "step_time_s", "audio_seconds_per_second", "utc",
                                 "data_wait_s_per_step", "label_s_per_step",
                                 "gpu_max_allocated_gib", "gpu_max_reserved_gib")}
                            for line in (ctx.run / "metrics.jsonl").read_text().splitlines()],
                "state": {k: v for k, v in train.asdict(ctx.state).items()},
                "rng": train.rng_state(ctx.student)}

    def assert_identical(self, a: dict, b: dict) -> None:
        self.assertEqual(a["model"].keys(), b["model"].keys())
        for key in a["model"]:
            self.assertTrue(torch.equal(a["model"][key], b["model"][key]), f"parameter {key} differs")
        sa, sb = a["optim"]["state"], b["optim"]["state"]
        self.assertEqual(sa.keys(), sb.keys())
        for i in sa:
            for name in sa[i]:
                self.assertTrue(torch.equal(sa[i][name], sb[i][name]), f"optimizer {i}.{name} differs")
        self.assertEqual(a["sched"], b["sched"])
        self.assertEqual(a["metrics"], b["metrics"])
        for key in ("step", "audio_seconds_seen", "best_wer", "best_step", "best_file", "evals",
                    "position", "batches_read", "seconds_by_source", "drop_counts"):
            self.assertEqual(a["state"][key], b["state"][key], key)
        self.assertTrue(torch.equal(a["rng"]["cuda"][0], b["rng"]["cuda"][0]))
        self.assertTrue(torch.equal(a["rng"]["torch"], b["rng"]["torch"]))
        self.assertEqual(a["rng"]["python"], b["rng"]["python"])

    def run_recorded(self, ctx: train.Context, **kw) -> list:
        """train_loop, recording the ids of every batch the loop consumed."""
        seen, original = [], train.Batcher.next

        def recording(batcher, timeout=None):
            batch = original(batcher, timeout)
            seen.append(batch["ids"])
            return batch

        with mock.patch.object(train.Batcher, "next", recording):
            train.train_loop(ctx, **kw)
        return seen

    def run_full(self, args, name: str) -> tuple[dict, list]:
        ctx = self.build(args, self.root / name)
        ids = self.run_recorded(ctx)
        return self.snapshot(ctx), ids

    def run_resumed(self, args, name: str, split: int) -> tuple[dict, list, dict]:
        run = self.root / name
        first = self.build(args, run)
        ids = self.run_recorded(first, stop_after=split)
        self.assertEqual(first.state.step, split)
        saved = self.snapshot(first)
        del first
        torch.cuda.empty_cache()
        second = self.build(args, run, scramble=999)  # different seed: restore must fix it
        payload = train.load_checkpoint(run)
        self.assertEqual(payload["step"], split)
        second.restore(payload)
        restored = self.snapshot(second)
        ids += self.run_recorded(second)
        return self.snapshot(second), ids, {"saved": saved, "restored": restored}

    # --------------------------------------------------------------------- tests

    def test_patched_tdt_kernel_matches_pytorch_reference(self) -> None:
        """The numba TDT loss (with prepare_tdt_loss's rebuilt grad kernel) against NeMo's
        pure-PyTorch TDT implementation: loss and gradient. omega = 0 (the reference has no
        sampled-RNNT term); sigma and durations as in the model config."""
        from nemo.collections.asr.losses.rnnt import RNNTLoss
        train.prepare_tdt_loss()
        kwargs = {"durations": [0, 1, 2, 3, 4], "sigma": 0.02}
        fast = RNNTLoss(num_classes=8, loss_name="tdt", reduction="sum",
                        loss_kwargs={**kwargs, "omega": 0.0, "clamp": -1.0})
        ref = RNNTLoss(num_classes=8, loss_name="tdt_pytorch", reduction="sum",
                       loss_kwargs=dict(kwargs))
        g = torch.Generator().manual_seed(0)
        acts = torch.randn(2, 6, 4, 8 + 1 + 5, generator=g).cuda()
        labels = torch.randint(0, 8, (2, 3), generator=g).cuda()
        act_lens, label_lens = torch.tensor([6, 4]).cuda(), torch.tensor([3, 2]).cuda()
        results = []
        for loss_fn in (fast, ref):
            x = acts.clone().requires_grad_(True)
            value = loss_fn(log_probs=x, targets=labels, input_lengths=act_lens,
                            target_lengths=label_lens)
            value.backward()
            results.append((value.detach(), x.grad))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(results[0][1], results[1][1], rtol=1e-4, atol=1e-5)

    def test_omega_rnnt_path_gradient(self) -> None:
        """omega = 1 forces the sampled standard RNN-T branch (compute_grad_kernel, also
        rebuilt by prepare_tdt_loss): its gradient must match central finite differences."""
        from nemo.collections.asr.losses.rnnt import RNNTLoss
        train.prepare_tdt_loss()
        loss_fn = RNNTLoss(num_classes=8, loss_name="tdt", reduction="sum",
                           loss_kwargs={"durations": [0, 1, 2, 3, 4], "sigma": 0.0,
                                        "omega": 1.0, "clamp": -1.0})
        g = torch.Generator().manual_seed(1)
        acts = torch.randn(1, 5, 3, 8 + 1 + 5, generator=g).cuda()
        labels = torch.randint(0, 8, (1, 2), generator=g).cuda()
        lens = (torch.tensor([5]).cuda(), torch.tensor([2]).cuda())

        def value(x):
            return loss_fn(log_probs=x, targets=labels, input_lengths=lens[0],
                           target_lengths=lens[1])

        x = acts.clone().requires_grad_(True)
        value(x).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        flat = x.grad.flatten()
        for index in torch.randperm(flat.numel(), generator=g)[:12].tolist():
            plus, minus = acts.clone().flatten(), acts.clone().flatten()
            plus[index] += 1e-2
            minus[index] -= 1e-2
            numeric = (value(plus.view_as(acts)) - value(minus.view_as(acts))).item() / 2e-2
            self.assertAlmostEqual(flat[index].item(), numeric, delta=2e-3 + 2e-2 * abs(numeric))

    def test_loss_reduction_survives_an_exception_inside_the_fused_loss(self) -> None:
        """An OOM inside NeMo's fused joint loss (caught by the probe) used to leave
        loss.reduction = None, so later TDT losses were per-utterance vectors and backward
        failed ("grad can be implicitly created only for scalar outputs")."""
        args = make_args(arm="P2")
        ctx = self.build(args, self.root / "reduction")
        batch = drain(train.Batcher(SyntheticStream(), 8.0, seed=0), 1)[0]
        labeled = train.label_batch(ctx, batch)
        configured = ctx.student.loss.reduction
        inner = ctx.student.loss._loss
        with mock.patch.object(inner, "forward",
                               side_effect=torch.OutOfMemoryError("simulated")):
            with self.assertRaises(torch.OutOfMemoryError):
                train.compute_losses(ctx.student, labeled["audio"], labeled["audio_len"],
                                     labeled["tokens"], labeled["token_len"], ctx.recipe,
                                     labeled["teacher_enc"], labeled["teacher_len"])
        self.assertIsNone(ctx.student.loss.reduction)  # the NeMo state the bug left behind
        loss, tdt, _ = train.compute_losses(ctx.student, labeled["audio"], labeled["audio_len"],
                                            labeled["tokens"], labeled["token_len"], ctx.recipe,
                                            labeled["teacher_enc"], labeled["teacher_len"])
        self.assertEqual(ctx.student.loss.reduction, configured)
        self.assertEqual(tdt.dim(), 0)
        loss.backward()

    def test_trainable_sets(self) -> None:
        counts = {}
        for arm in ("P1", "P2", "P3"):
            model = self.fresh_model_cls()
            counts[arm] = train.configure_trainable(model, train.recipe_of(make_args(arm=arm)))
            frozen = not train.RECIPES[arm]["train_pred_joint"]
            self.assertEqual(model.decoder.training, not frozen)
            self.assertEqual(model.joint.training, not frozen)
            self.assertTrue(model.encoder.training)
        self.assertEqual(counts["P1"], counts["P2"])
        self.assertEqual(counts["P1"]["trainable_modules"], ["encoder"])
        self.assertEqual(counts["P3"]["trainable_modules"], ["encoder", "decoder", "joint"])
        self.assertEqual(counts["P3"]["trainable"],
                         counts["P1"]["trainable"] + counts["P3"]["decoder_trainable"]
                         + counts["P3"]["joint_trainable"])

    def test_online_labels_and_augmentation(self) -> None:
        """Targets are the teacher's greedy texts; the teacher encodes clean features and the
        student SpecAugmented ones; encoder matching reuses the teacher's encoder output."""
        args = make_args(arm="P2")
        ctx = self.build(args, self.root / "augment")
        seen = {}
        hooks = [ctx.student.encoder.register_forward_pre_hook(
                     lambda m, a, kw: seen.__setitem__("student", kw["audio_signal"].detach().clone()),
                     with_kwargs=True),
                 self.teacher.encoder.register_forward_pre_hook(
                     lambda m, a, kw: seen.__setitem__("teacher", kw["audio_signal"].detach().clone()),
                     with_kwargs=True)]
        calls = []
        real = self.teacher.encoder.forward
        try:
            batch = drain(train.Batcher(SyntheticStream(), 8.0, seed=0), 1)[0]
            with mock.patch.object(self.teacher.encoder, "forward",
                                   side_effect=lambda *a, **k: calls.append(1) or real(*a, **k)):
                labeled = train.label_batch(ctx, batch)
                self.assertEqual(len(calls), 1)
                ctx.student.train()
                loss, tdt, enc = train.compute_losses(
                    ctx.student, labeled["audio"], labeled["audio_len"], labeled["tokens"],
                    labeled["token_len"], ctx.recipe, labeled["teacher_enc"], labeled["teacher_len"])
                self.assertEqual(len(calls), 1, "no second teacher forward")
        finally:
            for h in hooks:
                h.remove()
        texts, keep, _, _ = self.teacher_mod.teacher_label_batch(
            self.teacher, batch["audio"].cuda(), batch["audio_len"].cuda(), batch["texts"])
        expected = [ctx.student.tokenizer.text_to_ids(t) for t, k in zip(texts, keep.tolist()) if k]
        got = [labeled["tokens"][i, :n].tolist() for i, n in enumerate(labeled["token_len"].tolist())]
        self.assertEqual(got, expected)
        # Compare the longest kept row over its valid frames (padding frames are zero in both).
        kept = torch.nonzero(keep.cpu()).flatten()
        lengths = batch["audio_len"][kept]
        r = int(torch.argmax(lengths))
        frames = int(lengths[r]) // 160 - 1
        s, t = seen["student"][r, :, :frames], seen["teacher"][kept[r], :, :frames]
        masked = s == 0.0
        self.assertTrue(masked.any(), "SpecAugment did not mask the student input")
        self.assertFalse((t == 0.0).any(), "the teacher input was masked")
        torch.testing.assert_close(s[~masked], t[~masked], atol=1e-2, rtol=0)  # dither only
        self.assertGreater(enc.item(), 0.0)
        torch.testing.assert_close(loss, tdt + enc)

    def test_dropped_rows_and_skipped_batches(self) -> None:
        """Rows with keep False leave the batch; all-dropped batches advance the stream
        position without an optimizer step."""
        real = self.teacher_mod.teacher_label_batch
        calls = {"n": 0}

        def label(teacher, audio, lengths, human, precision, counts=None):
            texts, keep, enc, enc_len = real(teacher, audio, lengths, human, precision,
                                             counts=counts)
            calls["n"] += 1
            keep = keep.clone()
            if calls["n"] % 3 == 0:
                keep[:] = False
            else:
                keep[0] = False if len(keep) > 1 else keep[0]
            return texts, keep, enc, enc_len

        args = make_args(arm="P2", max_steps=4, eval_every=100)
        ctx = self.build(args, self.root / "drops", label_fn=label)
        train.train_loop(ctx)
        self.assertEqual(ctx.state.step, 4)
        self.assertGreaterEqual(ctx.state.skipped_batches, 1)
        self.assertEqual(ctx.state.batches_read, 4 + ctx.state.skipped_batches)
        self.assertLess(ctx.state.audio_seconds_seen, ctx.state.audio_seconds_read)

    def test_grad_checkpointing_is_exact(self) -> None:
        """Same updates with and without checkpointing; BatchNorm stats updated once."""
        results = []
        for flag in (False, True):
            args = make_args(arm="P2", grad_checkpointing=flag, max_steps=2, eval_every=100)
            ctx = self.build(args, self.root / f"ckpt-{flag}")
            train.train_loop(ctx)
            results.append({k: v.detach().cpu().clone() for k, v in ctx.student.state_dict().items()})
        for key in results[0]:
            self.assertTrue(torch.equal(results[0][key], results[1][key]), key)
        tracked = [k for k in results[1] if k.endswith("num_batches_tracked")]
        self.assertTrue(tracked)
        self.assertTrue(all(int(results[1][k]) == 2 for k in tracked))

    def test_resume_restores_state_and_stream_position(self) -> None:
        """P2 (ramp, grad checkpointing, evaluation before and after the split) and P3
        (trainable LSTM prediction network with dropout)."""
        for arm, steps, split, ckpt in (("P2", 6, 3, True), ("P3", 5, 2, False)):
            args = make_args(arm=arm, max_steps=steps, eval_every=2, grad_checkpointing=ckpt)
            full, full_ids = self.run_full(args, f"{arm}-full")
            again, _ = self.run_full(args, f"{arm}-again")
            self.assert_identical(full, again)  # deterministic given the stream
            resumed, resumed_ids, mid = self.run_resumed(args, f"{arm}-resumed", split)
            # Restored exactly what was saved: step, optimizer, scheduler, RNGs, position.
            self.assertEqual(mid["restored"]["state"]["step"], split)
            self.assertEqual(mid["restored"]["state"]["position"], mid["saved"]["state"]["position"])
            self.assertEqual(mid["restored"]["sched"], mid["saved"]["sched"])
            for i, s in mid["saved"]["optim"]["state"].items():
                for name, value in s.items():
                    self.assertTrue(torch.equal(value, mid["restored"]["optim"]["state"][i][name]))
            self.assertTrue(torch.equal(mid["restored"]["rng"]["cuda"][0], mid["saved"]["rng"]["cuda"][0]))
            self.assertEqual(mid["restored"]["rng"]["python"], mid["saved"]["rng"]["python"])
            sessions = [json.loads(x)["session"] for x in
                        (self.root / f"{arm}-resumed" / "metrics.jsonl").read_text().splitlines()]
            steps = [json.loads(x)["step"] for x in
                     (self.root / f"{arm}-resumed" / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(sessions, [0 if s <= split else 1 for s in steps])
            # The stream continued from the saved position: the same batches, in order.
            self.assertEqual(resumed_ids, full_ids)
            # And with a deterministic stream the whole run is bit-identical.
            self.assert_identical(full, resumed)
            fractions = [m["weight_fraction"] for m in full["metrics"] if "loss" in m]
            self.assertEqual(fractions[-1], 1.0)

    @mock.patch.object(train, "POOL_BATCHES", 2)  # small pools: the failure lands mid-run
    def test_stream_failure_checkpoints_and_resumes(self) -> None:
        args = make_args(arm="P1", max_steps=6, eval_every=100)
        full, full_ids = self.run_full(args, "nf-full")
        run = self.root / "nf"
        ctx = self.build(args, run, stream=SyntheticStream(fail_after=12))
        seen = []
        with self.assertRaises(train.StreamFailure):
            seen += self.run_recorded(ctx)
        failed_at = ctx.state.step
        self.assertTrue(0 < failed_at < 6)
        payload = train.load_checkpoint(run)
        self.assertEqual(payload["step"], failed_at)
        second = self.build(args, run, scramble=5)
        second.restore(payload)
        ids = self.run_recorded(second)
        self.assertEqual(second.state.step, 6)
        # Every batch after the failure point is the uninterrupted run's continuation.
        self.assertEqual(ids, full_ids[len(full_ids) - len(ids):])
        self.assert_identical(full, self.snapshot(second))

    def test_resume_refuses_other_configuration(self) -> None:
        args = make_args(arm="P1", max_steps=4, eval_every=2)
        run = self.root / "mismatch"
        ctx = self.build(args, run)
        train.train_loop(ctx, stop_after=1)
        other = self.build(make_args(arm="P1", max_steps=4, eval_every=2, lr=5e-4), run)
        with self.assertRaises(RuntimeError):
            other.restore(train.load_checkpoint(run))

    def test_best_selection_only_at_full_fraction(self) -> None:
        args = make_args(arm="P1", max_steps=8, eval_every=1, select="best")
        ctx = self.build(args, self.root / "best")
        train.train_loop(ctx)
        ramp = train.ramp_steps(8)  # 2
        lines = [json.loads(x) for x in (ctx.run / "metrics.jsonl").read_text().splitlines()]
        evals = [m for m in lines if m.get("eval")]
        self.assertEqual(len(evals), 8)
        self.assertTrue(all(not m["selectable"] for m in evals if m["step"] < ramp))
        self.assertGreaterEqual(ctx.state.best_step, ramp)
        self.assertEqual(sorted(p.name for p in ctx.run.glob("best-*.pt")), [ctx.state.best_file])

    def test_select_final_is_default_and_writes_no_best_files(self) -> None:
        args = make_args(arm="P1", max_steps=4, eval_every=1)
        self.assertEqual(args.select, "final")
        ctx = self.build(args, self.root / "final")
        train.train_loop(ctx)
        self.assertEqual(list(ctx.run.glob("best-*")), [])
        final_weights = {k: v.detach().clone() for k, v in ctx.student.state_dict().items()}
        selection = train.selected_model(ctx)
        self.assertEqual((selection["select"], selection["selected_step"]), ("final", 4))
        self.assertEqual(selection["selected_dev_subset_wer"], ctx.state.final_subset)
        self.assertIn("best_interim_subset_step", selection)  # information only
        for key, value in ctx.student.state_dict().items():
            self.assertTrue(torch.equal(value, final_weights[key]), key)

    def test_best_files_survive_a_crash_between_eval_and_checkpoint(self) -> None:
        """--select best: an eval writes a new best file after the last durable checkpoint and
        the process dies before the next one. The file that checkpoint references must still
        exist, resume must drop only the newer file, and finishing must work."""

        class Improving(StubEvaluator):
            calls = 0

            def subset(self, model):
                out = super().subset(model)
                Improving.calls += 1
                return {n: dict(r, wer=1.0 / Improving.calls) for n, r in out.items()}

        class Crash(Exception):
            pass

        args = make_args(arm="P1", max_steps=6, eval_every=2, select="best")
        run = self.root / "best-crash"
        first = self.build(args, run, evaluator=Improving())
        train.train_loop(first, stop_after=2)  # eval at 2 -> best-000002, checkpoint at 2
        self.assertEqual(first.state.best_file, "best-000002.pt")
        real_eval = train.run_eval

        def eval_then_crash(ctx):
            real_eval(ctx)
            if ctx.state.step == 4:
                raise Crash()

        with mock.patch.object(train, "run_eval", eval_then_crash), self.assertRaises(Crash):
            train.train_loop(first)
        self.assertTrue((run / "best-000002.pt").exists(), "referenced best file was deleted")
        self.assertTrue((run / "best-000004.pt").exists())
        del first
        second = self.build(args, run, scramble=3, evaluator=Improving())
        second.restore(train.load_checkpoint(run))
        self.assertEqual(second.state.best_file, "best-000002.pt")
        self.assertEqual(sorted(p.name for p in run.glob("best-*")), ["best-000002.pt"])
        train.train_loop(second)
        # After the durable final checkpoint only latest's and previous's references remain.
        self.assertEqual(sorted(p.name for p in run.glob("best-*")),
                         ["best-000002.pt", "best-000006.pt"])
        second.teacher = None
        with mock.patch.object(self.quant, "EXPECTED_MODULES", self.n_quant):
            result = train.finish(second, "0" * 64)
        self.assertEqual((result["select"], result["selected_step"]), ("best", 6))
        self.assertTrue(result["reconstruction"]["codes_exact"])

    def test_verify_recipe_checks_the_model(self) -> None:
        for arm in ("P1", "P2", "P3"):
            recipe = train.recipe_of(make_args(arm=arm))
            model = self.fresh_model_cls()
            with mock.patch.object(self.quant, "EXPECTED_MODULES", self.n_quant):
                self.quant.quantize_parakeet(model)
            train.configure_trainable(model, recipe)
            found = train.verify_recipe(model, recipe, self.quant)
            self.assertEqual(found["ternary_modules"], self.n_quant)
            self.assertEqual(found["pred_joint_dropout"], [0.0])
        p1 = train.recipe_of(make_args(arm="P1"))
        model.decoder.requires_grad_(True)  # P3-configured model checked as P1
        with self.assertRaises(RuntimeError):
            train.verify_recipe(model, p1, self.quant)
        plain = self.fresh_model_cls()
        train.configure_trainable(plain, p1)
        with self.assertRaises(RuntimeError):  # quantized recipe without ternary modules
            train.verify_recipe(plain, p1, self.quant)

    def test_pred_joint_dropout_disabled_in_every_arm(self) -> None:
        """DESIGN.md lets arms differ only in the loss and trainability: P3 trains the
        prediction and joint networks in train mode but with dropout 0."""
        for arm in ("P1", "P3"):
            model = self.fresh_model_cls()
            train.configure_trainable(model, train.recipe_of(make_args(arm=arm)))
            model.cuda()
            for part in (model.decoder, model.joint):
                for m in part.modules():
                    if isinstance(m, torch.nn.Dropout):
                        self.assertEqual(m.p, 0.0)
                    if isinstance(m, torch.nn.RNNBase):
                        self.assertEqual(m.dropout, 0.0)
            self.assertTrue(any(isinstance(m, torch.nn.Dropout) and m.p > 0
                                for m in model.encoder.modules()), "encoder dropout kept")
            self.assertEqual(model.decoder.training, arm == "P3")
            tokens = torch.randint(0, 1000, (3, 7), device="cuda")
            lengths = torch.tensor([7, 5, 3], device="cuda")
            enc = torch.randn(3, 64, 11, device="cuda")
            outs = []
            for _ in range(2):
                dec, _, _ = model.decoder(targets=tokens, target_length=lengths)
                outs.append((dec, model.joint.joint(enc.transpose(1, 2), dec.transpose(1, 2))))
            self.assertTrue(torch.equal(outs[0][0], outs[1][0]), f"{arm} decoder not deterministic")
            self.assertTrue(torch.equal(outs[0][1], outs[1][1]), f"{arm} joint not deterministic")

    def test_finish_exports_and_rebuilds(self) -> None:
        args = make_args(arm="P1", max_steps=4, eval_every=4, run_name="tiny-finish")
        run = self.root / "finish"
        ctx = self.build(args, run)
        train.train_loop(ctx)
        ctx.teacher = None
        with mock.patch.object(self.quant, "EXPECTED_MODULES", self.n_quant):
            result = train.finish(ctx, "0" * 64)
        self.assertEqual((result["select"], result["selected_step"]), ("final", 4))
        self.assertTrue(result["reconstruction"]["codes_exact"])
        self.assertTrue(result["reconstruction"]["scales_exact"])
        self.assertEqual(result["scored_on"], "model rebuilt from export")
        self.assertTrue((run / "export" / "manifest.json").exists())
        self.assertEqual(set(result["dev_wer"]), {"dev_a", "dev_b"})
        self.assertTrue(math.isfinite(result["dev_mean_wer"]))


if __name__ == "__main__":
    unittest.main()
