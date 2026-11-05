"""mutant_features contracts (mutant plan "Arms", "Features are per clip", "Code"): each arm's feature path, per-clip
features (batched equals alone; a batch-dependent encoder fails the check), Qwen's output length against
_get_feat_extract_output_lengths and its 13 Hz rate on real audio, frozen encoders (no gradient, unchanged weights),
D-T's vectors are the input of probe 1's BPE head, and the feature store (round trip, identity, resume).

CPU only: transformers' real Qwen3ASREncoder at a tiny size with the real Qwen3ASRFeatureExtractor, probe 1's tiny
stand-in T* encoder, a tiny Parakeet-TDT (tests/decoder_fixtures.py) for D-B, and real dev audio.

Run from finetune/relabel:  ./python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import encoder_probe_ctc as ep  # noqa: E402
import evaluate  # noqa: E402
import mutant_data as md  # noqa: E402
import mutant_features as mf  # noqa: E402
import paths  # noqa: E402
import relabel_paths  # noqa: E402
import test_encoder_probe_ctc as probe_tests  # noqa: E402
import test_mutant_data as data_tests  # noqa: E402

TMP = relabel_paths.ARTIFACTS / "tmp" / "tests" / "mutant-features"


def dev_clips(n: int, set_name: str = "librispeech_dev_clean", step: int = 37) -> tuple[list[np.ndarray], list[float]]:
    records = evaluate.read_manifest(paths.MANIFESTS / f"{set_name}.jsonl")[::step][:n]
    return [evaluate.load_audio(r["audio_filepath"]) for r in records], [r["duration"] for r in records]


def mixed_clips() -> tuple[list[np.ndarray], list[float]]:
    a, da = dev_clips(4, "librispeech_dev_clean")
    b, db = dev_clips(4, "ami_dev", step=211)
    rng = np.random.default_rng(0)
    short = [rng.standard_normal(n).astype(np.float32) * 0.05 for n in (200, 4000, 9000)]  # < 0.5 s, < 1 s
    return a + b + short, da + db + [len(c) / 16000 for c in short]


def tiny_qwen(seed: int = 0) -> mf.QwenArm:
    from transformers import Qwen3ASRFeatureExtractor
    from transformers.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASREncoderConfig
    from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder
    cfg = Qwen3ASREncoderConfig(d_model=32, encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=64,
                                downsample_hidden_size=8, num_mel_bins=128, n_window=50, n_window_infer=800,
                                max_position_embeddings=13, output_dim=16)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(seed)
    tower = Qwen3ASREncoder(cfg).float().eval()
    return mf.QwenArm(Qwen3ASRFeatureExtractor(), tower, torch.device("cpu"), dtype=torch.float32)


def tiny_tstar() -> mf.TStarArm:
    enc = probe_tests.TinyEncoder(dim=8, vocab=7).eval()
    out_llm = nn.Linear(8, 5)
    probe = ep.Probe(probe_tests.StandInProcessor(), "prompt", probe_tests.StandInTokenizer(), enc, out_llm,
                     torch.device("cpu"))
    return mf.TStarArm(probe)


class BatchMixing(mf.ArmEncoder):
    """A negative control: every clip's features are shifted by the batch's mean energy."""
    arm = "mixing"

    def __init__(self):
        self.preparer = mf.Preparer()
        self.lin = nn.Linear(1, 4)

    def batches(self, durations):
        return [list(range(len(durations)))]

    def modules(self):
        return [self.lin]

    def encode_prepared(self, items):
        mean = float(np.mean([np.abs(c).mean() for c in items]))
        with torch.no_grad():
            return [self.lin(torch.from_numpy(c[: len(c) // 1280 * 1280].reshape(-1, 1280).mean(1, keepdims=True)))
                    .add(mean).to(mf.STORE_DTYPE) for c in items]

    def identity(self):
        return {"arm": self.arm}


class TestQwen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm = tiny_qwen()
        cls.clips, cls.durations = mixed_clips()

    def test_output_length_is_the_feature_extract_formula(self):
        from transformers.models.qwen3_asr.modeling_qwen3_asr import _get_feat_extract_output_lengths
        feats = mf.encode_clips(self.arm, self.clips, self.durations)
        for clip, f in zip(self.clips, feats):
            mel, mask = self.arm.preparer(evaluate.pad_short(clip)[0])
            expect = int(_get_feat_extract_output_lengths(torch.tensor(int(mask.sum()))))
            self.assertEqual(f.shape, (expect, 32))
            self.assertEqual(f.dtype, mf.STORE_DTYPE)

    def test_rate_is_13_hz_on_real_audio(self):
        clips, durations = dev_clips(12, "librispeech_dev_clean", step=97)
        out = mf.qwen_rate_check(self.arm, clips, durations)
        self.assertTrue(out["lengths_match_formula"])
        self.assertAlmostEqual(out["frames_per_second"], 13.0, delta=0.15)
        self.assertTrue(out["passed"])

    def test_batched_equals_alone(self):
        out = mf.per_clip_check(self.arm, self.clips, self.durations)
        self.assertTrue(out["passed"], out)
        self.assertGreater(max(out["batch_sizes"]), 1, "the check must batch several clips")
        self.assertEqual(out["clips"], len(self.clips))

    def test_mel_is_computed_per_clip(self):
        """The snapshot's processor pads a batch to its longest clip; the arm computes each clip's mel alone, so a
        clip's mel does not depend on what else is batched."""
        a, _ = self.arm.preparer(evaluate.pad_short(self.clips[0])[0])
        b, _ = self.arm.preparer(evaluate.pad_short(self.clips[0])[0])
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.shape[0], 128)
        self.assertEqual(a.shape[1] % 100, 0)


class RawArm(mf.ArmEncoder):
    """A stand-in with an FP32 path: deterministic features of each clip, plus controlled differences: `iso` added to a
    batched FP32 pass, `batched` to a batched bf16 pass, `rounding` to every bf16 pass (its error against FP32)."""
    arm = "raw"

    def __init__(self, iso=0.0, batched=0.0, rounding=0.01):
        self.preparer = mf.Preparer()
        self.iso, self.batched_err, self.rounding = iso, batched, rounding
        self.one_clip = False
        self.lin = nn.Linear(1, 4)

    def batches(self, durations):
        return [[i] for i in range(len(durations))] if self.one_clip else [list(range(len(durations)))]

    def modules(self):
        return [self.lin]

    def encode_prepared(self, items):
        return [f.to(mf.STORE_DTYPE) for f in self.raw(items, torch.bfloat16)]

    def raw(self, items, dtype):
        out = []
        for k, c in enumerate(items):
            frames = torch.from_numpy(c[: len(c) // 1280 * 1280].reshape(-1, 1280).mean(1, keepdims=True)).float()
            f = torch.cat([frames * (j + 1) for j in range(4)], dim=1)
            sign = torch.ones_like(f)
            sign[:, ::2] = -1
            if dtype == torch.bfloat16:
                f = f + self.rounding * sign
                if len(items) > 1:
                    f = f + self.batched_err * sign.roll(1, dims=1)
            elif len(items) > 1:
                f = f + self.iso
            out.append(f)
        return out

    def identity(self):
        return {"arm": self.arm, "one_clip": self.one_clip}


class TestNoiseFloorRule(unittest.TestCase):
    """v3.2 "What 'within bf16 rounding' means": (1) FP32 batched vs alone max-abs <= 1e-3 with TF32 off (recorded);
    (2) per clip, bf16 batched-vs-alone mean-abs and max-abs <= 1.1 x that clip's bf16-vs-FP32 values."""

    def setUp(self):
        self.clips, self.durations = mixed_clips()

    def test_within_the_noise_floor_passes(self):
        out = mf.per_clip_check(RawArm(iso=5e-4, batched=0.0105, rounding=0.01), self.clips, self.durations)
        self.assertFalse(out["equal"])
        self.assertTrue(out["passed"], out)
        nf = out["noise_floor"]
        self.assertTrue(nf["isolation"]["passed"])
        self.assertTrue(nf["bf16"]["passed"])
        self.assertEqual(nf["tf32"], {"matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                                      "float32_matmul_precision": "highest"})
        self.assertEqual(len(nf["per_clip"]), len(self.clips))

    def test_above_the_noise_floor_fails(self):
        out = mf.per_clip_check(RawArm(iso=0.0, batched=0.0115, rounding=0.01), self.clips, self.durations)
        self.assertFalse(out["passed"])
        self.assertEqual(out["failed_part"], "noise_floor")

    def test_fp32_isolation_failure_and_the_one_clip_fallback(self):
        arm = RawArm(iso=2e-3, batched=0.005, rounding=0.01)    # bf16 within the floor, FP32 not isolated
        out = mf.per_clip_check(arm, self.clips, self.durations)
        self.assertFalse(out["passed"])
        self.assertEqual(out["failed_part"], "isolation")
        checks = mf.checked(arm, self.clips, self.durations)
        self.assertTrue(arm.one_clip)
        self.assertTrue(checks["passed"])
        self.assertEqual(checks["first"]["per_clip"]["failed_part"], "isolation")
        self.assertTrue(checks["per_clip"]["equal"])

    def test_noise_floor_failure_has_no_fallback(self):
        arm = RawArm(iso=0.0, batched=0.05, rounding=0.01)
        checks = mf.checked(arm, self.clips, self.durations)
        self.assertFalse(checks["passed"])
        self.assertFalse(arm.one_clip)

    def test_no_smoke_override(self):
        with self.assertRaises(SystemExit):
            mf.main(["extract", "--arm", "D-Q", "--smoke-shards", "40", "--smoke-ignore-failed-checks"])


class TestCleanStop(unittest.TestCase):
    """SIGTERM during a GPU command ends it through a normal exit (Stopped -> STOPPED_EXIT), never by the signal."""

    def test_sigterm_is_a_clean_stop_and_handlers_are_restored(self):
        import signal
        before = signal.getsignal(signal.SIGTERM)

        def work():
            os.kill(os.getpid(), signal.SIGTERM)
            for _ in range(1000):
                pass
            raise AssertionError("not stopped")
        with mock.patch("score_teacher.gpu_lock") as lock:
            lock.return_value.__enter__.return_value = None
            self.assertEqual(mf.locked("test", 1.0, work), mf.STOPPED_EXIT)
        self.assertIs(signal.getsignal(signal.SIGTERM), before)

    def test_a_broad_except_does_not_swallow_the_stop(self):
        import signal

        def work():
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                for _ in range(1000):
                    pass
            except Exception:                      # as library load paths do
                pass
            raise AssertionError("the stop was swallowed")
        with mock.patch("score_teacher.gpu_lock") as lock:
            lock.return_value.__enter__.return_value = None
            self.assertEqual(mf.locked("test", 1.0, work), mf.STOPPED_EXIT)
        self.assertFalse(issubclass(mf.Stopped, Exception))


CHILD = r"""
import contextlib, sys, time
from pathlib import Path
from unittest import mock
import torch
import evaluate, mutant_data as md, mutant_features as mf, score_teacher

root = Path(sys.argv[1])
lay = md.Layout(root / "w", root / "features", root / "decoder", None)


class SlowPreparer(mf.Preparer):
    def __call__(self, clip):
        time.sleep(0.02)
        return clip


class SlowArm(mf.ArmEncoder):
    arm = "slow"

    def __init__(self):
        self.preparer, self.lin = SlowPreparer(), torch.nn.Linear(1, 4)

    def batches(self, durations):
        return [[i] for i in range(len(durations))]

    def modules(self):
        return [self.lin]

    def identity(self):
        return {"arm": "slow"}

    def encode_prepared(self, items):
        # the stop lands inside the encoder's context managers, whose unwinding takes a while (SlowUnwind), so the
        # dying workers' SIGCHLD can raise inside it, as with the real inference contexts
        with torch.no_grad(), evaluate.strict_fp32(torch.device("cpu")), SlowUnwind():
            time.sleep(0.3)
            return [torch.zeros(3, 4, dtype=mf.STORE_DTYPE) for _ in items]


class SlowUnwind:
    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        if kind is not None:
            time.sleep(0.5)
        return False


def synced():
    print("SYNCED", flush=True)


mf.sync_cuda = synced
score_teacher.gpu_lock = lambda label, wait: contextlib.nullcontext()


def work():
    print("STARTED", flush=True)
    mf.extract(lay, SlowArm(), workers=6)
    return 0


with mock.patch.object(md.relabel_paths, "require_mount"):
    rc = mf.locked("process-group stop test", 1.0, work)
print("RC", rc, flush=True)
sys.exit(rc)
"""


class TestProcessGroupStop(unittest.TestCase):
    """A job stop signals the whole process group: extraction with six real DataLoader workers still ends through
    locked()'s stop path (CUDA synchronized, exit STOPPED_EXIT), leaves no process behind, and a later run completes."""

    def test_sigterm_to_the_process_group_during_extraction(self):
        import signal
        import subprocess
        import time as _time
        world = data_tests.TestSelectAndAudio
        world.setUpClass()
        root = TMP / "pg"
        shutil.rmtree(root, ignore_errors=True)
        lay = md.Layout(root / "w", root / "features", root / "decoder", None)
        shutil.copytree(world.lay.root, lay.root)
        with mock.patch.object(md.relabel_paths, "require_mount"):
            md.build_audio(lay)
        (root / "child.py").write_text(CHILD)
        here = Path(__file__).resolve().parent.parent
        for attempt in range(3):
            shutil.rmtree(lay.features, ignore_errors=True)
            p = subprocess.Popen([str(here / "python"), str(root / "child.py"), str(root)], cwd=str(here),
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            lines = []
            for line in p.stdout:
                lines.append(line)
                if line.startswith("STARTED"):
                    break
            _time.sleep(1.5)
            os.killpg(p.pid, signal.SIGTERM)
            out, _ = p.communicate(timeout=120)
            out = "".join(lines) + out
            self.assertEqual(p.returncode, mf.STOPPED_EXIT, out[-3000:])
            self.assertIn("SYNCED", out)
            self.assertIn("RC 76", out)
            deadline = _time.time() + 20
            while _time.time() < deadline:
                try:
                    os.killpg(p.pid, 0)
                except ProcessLookupError:
                    break
                _time.sleep(0.2)
            else:
                self.fail("processes of the stopped group are still running")
        arm_ids = data_tests.TestSelectAndAudio.data
        data = md.Data.load(lay)
        with mock.patch.object(md.relabel_paths, "require_mount"):
            stale = [f for kind in ("pools", "heldout") for f in (lay.features / "slow" / kind).glob("*.tmp*")]
            exec_ns = {}
            exec(compile(CHILD.split("def synced")[0], "child", "exec"), exec_ns)
            mf.extract(lay, exec_ns["SlowArm"](), workers=0)
        self.assertFalse([f for kind in ("pools", "heldout") for f in (lay.features / "slow" / kind).glob("*.tmp*")])
        for q in range(data.plan.n_pools):
            ids = data.pool_ids(q)
            for kind, split in (("pools", "train"), ("heldout", "heldout")):
                want = [i for i in ids if data.clips[i]["split"] == split]
                self.assertTrue(mf.committed(mf.store_paths(lay.features / "slow", kind, q), want, {
                    "version": mf.FEATURE_VERSION, "arm": "slow", "encoder": {"arm": "slow"}, "store_dtype": "bfloat16",
                    "pad_short_seconds": evaluate.MIN_DECODE_SECONDS, "plan_sha256": data.plan.sha256}))
        del arm_ids, stale


class TestSyncOnEveryExit(unittest.TestCase):
    """locked() synchronizes CUDA on success and on failure too, inside the GPU lock."""

    def test_sync_on_success_failure_and_stop(self):
        events = []

        class Lock:
            def __enter__(self):
                events.append("lock")

            def __exit__(self, *exc):
                events.append("unlock")
                return False
        with mock.patch("score_teacher.gpu_lock", return_value=Lock()), \
                mock.patch.object(mf, "sync_cuda", side_effect=lambda: events.append("sync")):
            self.assertEqual(mf.locked("t", 1.0, lambda: 7), 7)
            self.assertEqual(events, ["lock", "sync", "unlock"])
            events.clear()

            def boom():
                raise ValueError("failed run")
            with self.assertRaises(ValueError):
                mf.locked("t", 1.0, boom)
            self.assertEqual(events, ["lock", "sync", "unlock"])

    def test_a_broken_context_does_not_hide_the_error(self):
        with mock.patch("score_teacher.gpu_lock") as lock, \
                mock.patch.object(mf, "sync_cuda", side_effect=RuntimeError("context faulted")):
            lock.return_value.__enter__.return_value = None

            def boom():
                raise ValueError("the original error")
            with self.assertRaises(ValueError):
                mf.locked("t", 1.0, boom)


class TestStaleTemporaryFiles(unittest.TestCase):
    def test_only_files_of_dead_writers_are_removed(self):
        d = TMP / "stale"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        dead = next(p for p in range(4_190_000, 4_000_000, -1) if not mf._alive(p))
        files = {"00003.bin": "committed", "00003.json": "marker", f"00004.bin.tmp{dead}": "stale",
                 f"00004.json.tmp{dead}": "stale", f"00005.bin.tmp{os.getpid()}": "own",
                 f"00006.bin.tmp{os.getppid()}": "alive"}
        for name in files:
            (d / name).write_text(name)
        removed = mf.remove_stale_tmp(d)
        self.assertEqual(sorted(removed), sorted(n for n, k in files.items() if k == "stale"))
        self.assertEqual(sorted(p.name for p in d.iterdir()), sorted(n for n, k in files.items() if k != "stale"))
        self.assertEqual(mf.remove_stale_tmp(TMP / "missing-dir"), [])


class TestPerClipCheck(unittest.TestCase):
    def test_batch_dependent_encoder_fails(self):
        clips, durations = mixed_clips()
        out = mf.per_clip_check(BatchMixing(), clips[:6], durations[:6])
        self.assertFalse(out["passed"])
        self.assertGreater(out["max_abs_diff"], 0)

    def test_tstar_one_clip_per_pass_is_equal(self):
        clips, durations = mixed_clips()
        out = mf.per_clip_check(tiny_tstar(), clips, durations)
        self.assertTrue(out["passed"], out)
        self.assertTrue(out["equal"])


class TestTStar(unittest.TestCase):
    def test_vectors_are_probe1_bpe_head_inputs(self):
        arm = tiny_tstar()
        clips, durations = dev_clips(5, "ami_dev", step=301)
        feats = mf.encode_clips(arm, clips, durations)
        for clip, f in zip(clips, feats):
            clip = evaluate.pad_short(clip)[0]
            out = arm.probe.decode(clip)
            self.assertEqual(f.shape[0], out["pooled_frames"])
            frames = torch.softmax(arm.probe.out_llm(f.to(arm.probe.out_llm.weight.dtype)).float(), -1).argmax(-1)
            self.assertEqual(frames.tolist(), out["bpe_frames"])
            self.assertEqual(f.dtype, mf.STORE_DTYPE)

    def test_rate_is_12_5_hz(self):
        arm = tiny_tstar()
        clips, durations = dev_clips(6, "librispeech_dev_clean", step=113)
        feats = mf.encode_clips(arm, clips, durations)
        rate = sum(f.shape[0] for f in feats) / sum(durations)
        self.assertAlmostEqual(rate, 12.5, delta=0.3)


class TestFrozen(unittest.TestCase):
    def check_frozen(self, arm, clips, durations):
        before = mf.frozen_state(arm)
        for m in arm.modules():
            self.assertFalse(any(p.requires_grad for p in m.parameters()))
        feats = mf.encode_clips(arm, clips, durations)
        self.assertEqual(mf.frozen_state(arm), before)
        for m in arm.modules():
            self.assertTrue(all(p.grad is None for p in m.parameters()))
            self.assertFalse(m.training)
        for f in feats:
            self.assertFalse(f.requires_grad)
            self.assertIsNone(f.grad_fn)
            self.assertFalse(f.is_inference())
        return feats

    def test_every_arm_encoder_is_frozen(self):
        import decoder_fixtures
        clips, durations = dev_clips(4, "voxpopuli_dev", step=53)
        b0 = mf.B0Arm(decoder_fixtures.tiny_model(), torch.device("cpu"))
        feats = self.check_frozen(b0, clips, durations)
        self.assertTrue(all(f.shape[1] == 64 for f in feats))
        self.check_frozen(tiny_tstar(), clips, durations)
        self.check_frozen(tiny_qwen(), clips, durations)

    def test_b0_arm_rate_and_batching(self):
        import decoder_fixtures
        clips, durations = dev_clips(5, "librispeech_dev_clean", step=71)
        arm = mf.B0Arm(decoder_fixtures.tiny_model(), torch.device("cpu"))
        feats = mf.encode_clips(arm, clips, durations)
        rate = sum(f.shape[0] for f in feats) / sum(durations)
        self.assertAlmostEqual(rate, 12.5, delta=0.3)
        self.assertEqual(arm.batches(durations), evaluate.make_batches(
            [{"duration": d, "id": f"{i:09d}"} for i, d in enumerate(durations)], 64, 1200.0))


class TestStore(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(TMP, ignore_errors=True)

    def test_round_trip_and_identity(self):
        rng = torch.Generator().manual_seed(0)
        feats = [torch.randn(n, 6, generator=rng).to(mf.STORE_DTYPE) for n in (3, 0, 7, 1)]
        ids = ["a", "b", "c", "d"]
        p = mf.store_paths(TMP, "pools", 3)
        mf.write_features(p, ids, feats, {"arm": "x"})
        self.assertTrue(mf.committed(p, ids, {"arm": "x"}))
        self.assertFalse(mf.committed(p, ids, {"arm": "y"}))
        self.assertFalse(mf.committed(p, ids[:3], {"arm": "x"}))
        store = mf.FeatureFile(p)
        for i, f in zip(ids, feats):
            self.assertTrue(torch.equal(store.get(i), f))
        with open(p["bin"], "r+b") as h:
            h.write(b"\x00\x01")
        with self.assertRaises(mf.FeatureError):
            mf.FeatureFile(p, verify=True)

    def test_extract_pools_matches_encode_and_resumes(self):
        world = data_tests.TestSelectAndAudio
        world.setUpClass()
        lay = md.Layout(TMP / "w", TMP / "features", TMP / "decoder", None)
        shutil.copytree(world.lay.root, lay.root)
        data = md.Data.load(lay)
        arm = tiny_qwen()
        with mock.patch.object(md.relabel_paths, "require_mount"):
            md.build_audio(lay)
            mf.extract(lay, arm, workers=0)
            for p in range(data.plan.n_pools):
                audio = md.PoolAudio(lay, data, p)
                ids = data.pool_ids(p)
                want = mf.encode_clips(arm, [audio.clip(i) for i in ids], [data.clips[i]["duration"] for i in ids])
                train = mf.FeatureFile(mf.store_paths(lay.features / arm.arm, "pools", p))
                held = mf.FeatureFile(mf.store_paths(lay.features / arm.arm, "heldout", p))
                for i, w in zip(ids, want):
                    got = (held if data.clips[i]["split"] == "heldout" else train).get(i)
                    self.assertTrue(torch.equal(got, w))
                self.assertEqual(set(train.ids) | set(held.ids), set(ids))
                self.assertFalse(set(train.ids) & set(held.ids))
            with mock.patch.object(mf, "write_features", wraps=mf.write_features) as spy:
                mf.extract(lay, arm, workers=0)
            self.assertEqual(spy.call_count, 0)
        import mutant_train as mt
        items = mt.heldout_items(lay, arm.arm, data, mf.store_identity(arm, data))
        self.assertEqual(sorted(it["id"] for it in items), data.heldout_ids())
        self.assertTrue(all(it["text"] == data.clips[it["id"]]["reference"] for it in items))
        stats = json.loads((lay.features / arm.arm / "token-frames.json").read_text())
        for src, st in stats.items():
            rows = [c for c in data.clips.values() if c["source"] == src]
            frames = {i: f.shape[0] for p in range(data.plan.n_pools) for kind in ("pools", "heldout")
                      for i, f in ((i, mf.FeatureFile(mf.store_paths(lay.features / arm.arm, kind, p)).get(i))
                                   for i in mf.FeatureFile(mf.store_paths(lay.features / arm.arm, kind, p)).ids)}
            self.assertAlmostEqual(st["share_tokens_over_frames"],
                                   sum(len(c["tokens"]) > frames[c["id"]] for c in rows) / len(rows))


if __name__ == "__main__":
    unittest.main()
