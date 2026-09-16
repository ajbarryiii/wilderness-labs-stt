"""Stock CTranslate2 Whisper fixed-work control, without a source patch.

CTranslate2 v4.8.2 source contracts used here:
  src/models/whisper.cc: task prefix is forward_prompt'ed; text is start_tokens.
  src/decoding.cc: GreedySearch consumes hard text prefixes one token per step.
  include/ctranslate2/decoding.h: return_prefix defaults to true.
The returned prefix is checked on every run, so replay cannot silently fall back
to unconstrained model-generated token lengths.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


class CTranslate2Control:
    def __init__(self, model_path, compute_type="float16", flash_attention=False,
                 device_index=0, sot_id=50257, no_timestamps_id=50362):
        import ctranslate2
        self.ct2 = ctranslate2
        self.model_path = str(Path(model_path).resolve())
        self.requested_compute_type = compute_type
        self.flash_attention = flash_attention
        self.device_index = device_index
        self.sot_id, self.no_timestamps_id = sot_id, no_timestamps_id
        self.model = ctranslate2.models.Whisper(
            self.model_path, device="cuda", device_index=device_index,
            compute_type=compute_type, flash_attention=flash_attention,
            inter_threads=1, intra_threads=1)
        if self.model.compute_type != compute_type:
            raise RuntimeError(f"CTranslate2 precision fallback: requested {compute_type}, got {self.model.compute_type}")
        self.replay_verified = False

    def features(self, mel):
        if isinstance(mel, self.ct2.StorageView):
            return mel
        if hasattr(mel, "detach"):
            if mel.device.type != "cpu":
                raise ValueError("Pass CPU mel to CT2; measured transfer is performed by the runtime")
            mel = mel.detach().numpy()
        mel = np.asarray(mel, dtype=np.float32, order="C")
        if mel.shape != (1, 80, 3000):
            raise ValueError(f"Expected batch-one 30-second mel [1,80,3000], got {mel.shape}")
        return self.ct2.StorageView.from_array(mel)

    def encode(self, mel):
        return self.model.encode(self.features(mel), to_cpu=False)

    def decode(self, encoded, replay_tokens):
        tokens = [int(t) for t in replay_tokens]
        if not tokens or len(tokens) + 1 > 448:
            raise ValueError("Require 1..447 forced output tokens")
        # Ordinary English text IDs keep get_prompt_length exactly two; special
        # IDs could silently expand task prefill or trigger timestamp processing.
        if any(t < 0 or t >= 50256 for t in tokens):
            raise ValueError("Forced text tokens must be ordinary English IDs in [0,50256)")
        result = self.model.generate(
            encoded, [[self.sot_id, self.no_timestamps_id] + tokens],
            beam_size=1, patience=1, num_hypotheses=1,
            max_length=2 * len(tokens), return_scores=False,
            return_logits_vocab=False, return_no_speech_prob=False,
            suppress_blank=False, suppress_tokens=[], sampling_topk=1,
            sampling_temperature=1, repetition_penalty=1,
            no_repeat_ngram_size=0)
        actual = result[0].sequences_ids[0]
        if list(actual) != tokens:
            raise RuntimeError(
                f"CTranslate2 forced replay contract failed: expected {len(tokens)} "
                f"fixed tokens, got {len(actual)} tokens (version {self.ct2.__version__})")
        self.replay_verified = True
        return result

    def run(self, mel, replay_tokens):
        # A single generate call includes encoding and avoids the intermediate
        # synchronization imposed by the diagnostic encode() API.
        return self.decode(self.features(mel), replay_tokens)

    def metadata(self):
        return {"engine": "ctranslate2", "version": self.ct2.__version__,
                "model_path": self.model_path, "compute_type": self.model.compute_type,
                "requested_compute_type": self.requested_compute_type,
                "flash_attention": self.flash_attention, "device_index": self.device_index,
                "replay_verified": self.replay_verified,
                "replay": "one SOT prefill without logits; N sequential decode steps beginning at no_timestamps; N forced text outputs",
                "source_contract": "https://github.com/OpenNMT/CTranslate2/blob/v4.8.2/src/decoding.cc",
                "flash_attention_default_reason": "Installed CT2 4.8.2 wheel rejects Flash Attention 2 on this RTX 5090; explicit opt-in remains available",
                "same_engine_as_candidates": False}

    def unload(self):
        self.model.unload_model()
