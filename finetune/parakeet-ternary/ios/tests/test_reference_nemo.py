"""reference.py against NeMo 3.0 on CPU in FP32: DESIGN.md gate 1 (NixOS, NeMo env, through ../heavy; ~7 GB).

FullDepthReal: the pinned v2 .nemo restored by NeMo, and the reference's own read of it (models.b0).
On three development clips, golden.run_clip
records NeMo's features, subsampling output, every layer, encoder output and its own greedy
decode (GreedyBatchedTDTInfer, every step) with the per-step LSTM state and logits, and
golden.compare gates the reference against them (rel <= 1e-5, abs <= 1e-4, identical decisions,
finite, input sensitivity). The reference's greedy tokens must also equal NeMo's own
model.decoding output.
ReducedDepth: NeMo instantiated from the pinned config with 2 encoder layers and the seeded random
model seed 2 (ternary modules handed to the reference as int8 codes + FP32 scales, to NeMo
dequantized): the same per-clip gate, plus a zero-padded batch of the three clips and seeded noise (masking,
and the max-symbols rule, which the noise triggers in this model),
teacher-forced prediction and joint outputs, and the frame-looping GreedyTDTInfer (strategy
"greedy", not this config's) for information.

  ../heavy ios-wp1-nemo --mem-max 12G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 <repo>/finetune/parakeet-ternary/python <repo>/finetune/parakeet-ternary/ios/tests/test_reference_nemo.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (puts ios/ and its parent on sys.path)
import golden  # noqa: E402
import models  # noqa: E402
import randomweights as rw  # noqa: E402
import reference  # noqa: E402

REDUCED_LAYERS = 2
REDUCED_SEED = 2


def valid_frames(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """[B, D, T] -> [sum(lengths), D] of the valid frames."""
    return torch.cat([x[b, :, :int(n)].T for b, n in enumerate(lengths)])


class GateMixin:
    def gate_clips(self, label: str) -> None:
        for record, audio in zip(self.records, self.clips):
            golden_outputs = golden.run_clip(self.nemo, audio)
            metrics, failures = golden.compare(self.ref, golden_outputs)
            common.report(label, clip=record["id"], failures=failures, **metrics)
            self.assertEqual(failures, [], record["id"])


class FullDepthReal(GateMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import paths
        from nemo.collections.asr.models import EncDecRNNTBPEModel

        torch.set_grad_enabled(False)
        stats = rw.load_stats()
        cls.model = EncDecRNNTBPEModel.restore_from(str(paths.MODEL_FILE), map_location="cpu").float().eval()
        cls.nemo = golden.NemoModules.wrap(cls.model, stats["model_config"])
        cls.ref = models.b0()  # the reference reads the .nemo itself, not NeMo's state_dict
        cls.ref_load = cls.ref.provenance["load"]
        cls.records = common.dev_clips()
        cls.clips = [common.load_audio(r["audio_filepath"]) for r in cls.records]
        common.report("real_load", **cls.ref_load)

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.model, cls.nemo, cls.ref

    def test_gate(self) -> None:
        self.gate_clips("real_full_depth_gate")

    def test_tokens_equal_nemo_decoding(self) -> None:
        model = self.model
        for record, clip in zip(self.records, self.clips):
            audio, lengths = common.batch([clip])
            feats, flen = model.preprocessor(input_signal=audio, length=lengths)
            enc_n, elen = model.encoder(audio_signal=feats, length=flen)
            hyps = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc_n, encoded_lengths=elen,
                                                                  return_hypotheses=True)
            hyp = (hyps[0] if isinstance(hyps, tuple) else hyps)[0]
            nemo_tokens = golden.as_list(hyp.y_sequence)
            enc_r, elen_r = self.ref(audio, lengths)  # the reference end to end
            trace = reference.greedy_decode(self.ref, enc_r, elen_r)[0]
            common.report("real_model_decoding", clip=record["id"], tokens=len(nemo_tokens),
                          tokens_equal=trace.tokens == nemo_tokens, text=model.tokenizer.ids_to_text(trace.tokens),
                          nemo_text=hyp.text)
            self.assertEqual(trace.tokens, nemo_tokens)


class ReducedDepth(GateMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_grad_enabled(False)
        stats = rw.load_stats()
        tensors, digest = golden.tensor_source(stats, REDUCED_SEED, REDUCED_LAYERS)
        cls.ref = reference.build(reference.Config.from_model_config(stats["model_config"], REDUCED_LAYERS))
        ref_load = reference.load_weights(cls.ref, tensors)
        cls.nemo = golden.NemoModules.from_config(stats["model_config"], REDUCED_LAYERS)
        nemo_load = reference.load_weights(cls.nemo, tensors)
        cls.records = common.dev_clips()
        cls.clips = [common.load_audio(r["audio_filepath"]) for r in cls.records]
        common.report("reduced_load", digest=digest, reference=ref_load, nemo=nemo_load)
        expected = REDUCED_LAYERS * len(reference.TERNARY_SUFFIXES)
        assert ref_load["ternary_modules"] == expected and nemo_load["ternary_modules"] == expected

    def test_gate(self) -> None:
        self.gate_clips("reduced_gate")

    def test_gate_max_symbols(self) -> None:
        """2 s of seeded noise drives this surrogate into the max-symbols rule; NeMo's captured decode
        (golden.run_clip) must show forced advances and the reference must pass the gate on it."""
        noise = (0.1 * torch.randn(32000, generator=torch.Generator().manual_seed(0))).numpy()
        golden_outputs = golden.run_clip(self.nemo, noise)
        metrics, failures = golden.compare(self.ref, golden_outputs)
        common.report("reduced_gate_max_symbols", failures=failures, **metrics)
        self.assertGreater(int(golden_outputs["trace_forced_advance"].sum()), 0)
        # 1% noise on white noise barely changes its spectrum, so the sensitivity criterion (meant for
        # speech; checked on the dev clips) is not applied to this decoding-rule input.
        self.assertEqual([f for f in failures if not f.startswith("sensitivity")], [])

    def test_padded_batch(self) -> None:
        """The three clips plus 2 s of seeded white noise (which drives this random model into the
        max-symbols rule) as one zero-padded batch."""
        noise = (0.1 * torch.randn(32000, generator=torch.Generator().manual_seed(0))).numpy()
        audio, lengths = common.batch(self.clips + [noise])
        feats_n, flen = self.nemo.preprocessor(input_signal=audio, length=lengths)
        feats_r, flen_r = self.ref.preprocessor(audio, lengths)
        enc_n, elen = self.nemo.encoder(audio_signal=feats_n, length=flen)
        enc_r, elen_r = self.ref.encoder(feats_n, flen)
        self.assertTrue(torch.equal(flen, flen_r) and torch.equal(elen, elen_r))
        feat_err = common.errors(feats_r, feats_n)
        enc_err = common.errors(valid_frames(enc_r, elen), valid_frames(enc_n, elen))
        hyps = golden.greedy_infer(self.nemo)(encoder_output=enc_n, encoded_lengths=elen)[0]
        hyps_frame = golden.greedy_infer(self.nemo, frame_looping=True)(encoder_output=enc_n, encoded_lengths=elen)[0]
        traces = reference.greedy_decode(self.ref, enc_r, elen, record=True)
        decode = []
        for b, (hyp, (trace, steps)) in enumerate(zip(hyps, traces)):
            labels, log_probs = golden.hyp_steps(hyp)
            same_steps = trace.token == labels
            decode.append({
                "steps": len(trace), "tokens": len(trace.tokens), "forced_advances": sum(trace.forced_advance),
                "labels_equal": same_steps, "tokens_equal": trace.tokens == golden.as_list(hyp.y_sequence),
                "timestamps_equal": trace.timestamps == golden.as_list(hyp.timestamp),
                "durations_equal": trace.token_durations == golden.as_list(hyp.token_duration),
                "log_probs": common.errors(steps.logits.log_softmax(-1), log_probs) if same_steps else None,
                "frame_looping_tokens_equal": golden.as_list(hyps_frame[b].y_sequence) == trace.tokens})
        common.report("reduced_padded_batch", features=feat_err, encoder_valid_frames=enc_err,
                      encoder_frames=elen.tolist(), decode=decode)
        for errs in (feat_err, enc_err):
            self.assertLessEqual(errs[0], common.REL_CEILING)
            self.assertLessEqual(errs[1], common.ABS_CEILING)
        self.assertGreater(sum(d["forced_advances"] for d in decode), 0, "max-symbols rule not exercised")
        for d in decode:
            self.assertTrue(d["labels_equal"] and d["tokens_equal"] and d["timestamps_equal"] and d["durations_equal"])
            self.assertLessEqual(d["log_probs"][0], common.REL_CEILING)
            self.assertLessEqual(d["log_probs"][1], common.ABS_CEILING)

    def test_prediction_and_joint(self) -> None:
        audio, lengths = common.batch(self.clips)
        feats, flen = self.nemo.preprocessor(input_signal=audio, length=lengths)
        enc, elen = self.nemo.encoder(audio_signal=feats, length=flen)
        targets = torch.randint(0, self.ref.cfg.vocab_size, (len(self.clips), 12),
                                generator=torch.Generator().manual_seed(0))
        g_n, _, _ = self.nemo.decoder(targets=targets, target_length=torch.tensor([12, 7, 9]))  # [B, H, U + 1]
        g_r = self.ref.decoder(targets)
        joint_n = self.nemo.joint.joint(enc.transpose(1, 2), g_n.transpose(1, 2))
        joint_r = self.ref.joint(enc.transpose(1, 2), g_r, log_softmax=True)
        valid = (torch.arange(enc.shape[2])[None, :] < elen[:, None])
        pred_err = common.errors(g_r, g_n.transpose(1, 2))
        joint_err = common.errors(joint_r[valid], joint_n[valid])
        common.report("reduced_prediction_joint", prediction=pred_err, joint_log_probs=joint_err)
        for errs in (pred_err, joint_err):
            self.assertLessEqual(errs[0], common.REL_CEILING)
            self.assertLessEqual(errs[1], common.ABS_CEILING)


def tearDownModule() -> None:
    common.report("test_reference_nemo_peak", peak_rss_mb=common.peak_rss_mb())


if __name__ == "__main__":
    unittest.main()
