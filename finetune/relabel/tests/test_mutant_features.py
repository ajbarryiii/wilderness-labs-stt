"""mutant_features contracts (mutant plan "Arms", "Features are per clip", "Code"): each arm's feature path, per-clip
features (batched equals alone; a batch-dependent encoder fails the check), Qwen's output length against
_get_feat_extract_output_lengths and its 13 Hz rate on real audio, frozen encoders (no gradient, unchanged weights),
D-T's vectors are the input of probe 1's BPE head, and the feature store (round trip, identity, resume).

CPU only: transformers' real Qwen3ASREncoder at a tiny size with the real Qwen3ASRFeatureExtractor, probe 1's tiny
stand-in T* encoder, a tiny Parakeet-TDT (tests/decoder_fixtures.py) for D-B, and real dev audio.

Run from finetune/relabel:  ./python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import shutil
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


if __name__ == "__main__":
    unittest.main()
