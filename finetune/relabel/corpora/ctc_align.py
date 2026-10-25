"""A permissively licensed forced aligner for TRAINING corpora (architect decision 2026-10-21).

Part A's MMS aligner (devsets/align.py) has non-commercial weights, so it is used for the eval-only domain dev
sets only; training corpora (med-osce) are cut with this aligner instead. It reuses Part A's block algorithm
and confidence/drop rule unchanged (devsets.align.BlockAligner: blocks, wildcards, retries, the well-aligned
prefix, the per-unit score exp(mean over the unit's character frames of log p(char) - max log p(non-wildcard
label)), and MIN_SCORE 0.35), with a different acoustic model of the same kind:

- the registry's `train-aligner` entry: an Apache-2.0 wav2vec2 character CTC model (English letters and the
  apostrophe, 20 ms frames, the same frame geometry as Part A's: hop 320 samples, receptive field 400), loaded
  from its pinned revision with its weights' SHA-256 checked;
- emissions are computed exactly like Part A's (20 s chunks with 2 s of context on each side, central frames
  kept), each chunk normalized to zero mean and unit variance as the model's feature extractor does;
- the model's word-delimiter label is folded into the blank (Part A's alphabet has no delimiter), and a
  wildcard column (log-probability 0 in every frame) is appended;
- a unit's targets are the indices of its alignment characters (devsets.transcript.alignment_chars);
- the model runs on the CPU or the GPU (`device`); the audio is normalized on the CPU either way and the
  log-probabilities come back to the CPU as float32, and on the GPU TF32 is disabled, so the two devices
  differ only by floating-point reduction order. The device is recorded in the identity ("device").

Why this model: a BPE CTC model (CC-BY-4.0 FastConformer, 80 ms frames) was tried first and failed Part A's
protocol: its very peaky outputs let a path cross unmatched speech on blank frames, so neighbouring words
were placed up to 20 s late on matching sub-word tokens while still scoring high. Validation numbers for the
chosen model, MMS and that model under the same protocol are in corpora/validate.py's docstring.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from devsets import align as part_a

MIN_SCORE = part_a.MIN_SCORE   # Part A's threshold, re-validated for this model (corpora/validate.py)
DELIMITER = "|"
BLANK = "<pad>"


class CTCAligner(part_a.BlockAligner):
    """Part A's block aligner over a character CTC model at Part A's 20 ms frames (module docstring).

    forward(chunk [1, n] float32) -> log-probabilities [1, frame_count(n), classes]; `vocab` maps the model's
    labels to class indices."""

    def __init__(self, forward: Callable, vocab: Mapping[str, int], min_score: float = MIN_SCORE,
                 identity: dict | None = None) -> None:
        self.classes = max(vocab.values()) + 1
        self.delimiter = vocab.get(DELIMITER)
        chars = {c.lower(): i for c, i in vocab.items() if len(c) == 1 and (c.isalpha() or c == "'")}
        super().__init__(chars, star=self.classes, blank=vocab[BLANK], min_score=min_score)
        self.forward = forward
        self.identity = identity or {}

    def targets(self, chars: str | None) -> list[int]:
        if chars is None:
            return [self.star]
        missing = [c for c in chars if c not in self.dictionary]
        if missing or not chars:
            raise ValueError(f"unit {chars!r}: characters {missing} are not in the aligner's alphabet")
        return [self.dictionary[c] for c in chars]

    def emission(self, audio: np.ndarray):
        """Log-probabilities [frame_count, classes + wildcard] (CPU float32) of the whole recording, chunked."""
        import torch
        samples = len(audio)
        total = part_a.frame_count(samples)
        if total == 0:
            raise ValueError(f"audio of {samples} samples is too short to align")
        wave = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        out = torch.empty(total, self.classes + 1, dtype=torch.float32)
        with torch.inference_mode():
            for c0 in range(0, total, part_a.CHUNK_FRAMES):
                c1 = min(total, c0 + part_a.CHUNK_FRAMES)
                f0, f1 = max(0, c0 - part_a.CONTEXT_FRAMES), min(total, c1 + part_a.CONTEXT_FRAMES)
                last = samples if f1 == total else (f1 - 1) * part_a.HOP + part_a.RECEPTIVE_FIELD
                lp = self.forward(wave[f0 * part_a.HOP:last].unsqueeze(0))[0].float()
                if lp.shape[0] != f1 - f0 or lp.shape[1] != self.classes:
                    raise RuntimeError(f"emission {tuple(lp.shape)} != ({f1 - f0}, {self.classes})")
                out[c0:c1, :self.classes] = lp[c0 - f0:c1 - f0]
        out = out.clone()
        if self.delimiter is not None:   # word boundaries count as blank; the delimiter is never a target
            out[:, self.blank] = torch.logaddexp(out[:, self.blank], out[:, self.delimiter])
            out[:, self.delimiter] = -1e4
        out[:, self.star] = 0.0
        return out

    def align(self, audio: np.ndarray, units: list[str | None]):
        if not units:
            return []
        return self.align_emission(self.emission(audio), units)


def make_forward(model, normalize: bool, device: str = "cpu") -> Callable:
    """forward(chunk [1, n] CPU float32) -> CPU float32 log-probabilities [1, frames, classes]: the chunk is
    normalized on the CPU (the feature extractor's zero mean, unit variance) and the model runs on `device`."""
    import torch

    def forward(x):
        if normalize:
            x = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)
        return model(x.to(device)).logits.float().log_softmax(-1).cpu()
    return forward


def load(reg: dict, threads: int | None = None, min_score: float = MIN_SCORE, device: str = "cpu") -> CTCAligner:
    """The registry's pinned model on `device` (downloaded to the relabel HF cache once)."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
    if threads:
        torch.set_num_threads(threads)
    if device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError(f"aligner device {device!r} requested but CUDA is not available")
        torch.backends.cuda.matmul.allow_tf32 = False   # full float32, like the CPU
        torch.backends.cudnn.allow_tf32 = False
    for name, sha in reg["sha256"].items():
        path = Path(hf_hub_download(reg["hf_id"], name, revision=reg["revision"]))
        with open(path, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != sha:
            raise RuntimeError(f"aligner file {name} has SHA-256 {digest}, not the pinned {sha}")
    model = Wav2Vec2ForCTC.from_pretrained(reg["hf_id"], revision=reg["revision"]).eval().to(device)
    vocab = Wav2Vec2CTCTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"]).get_vocab()
    normalize = bool(Wav2Vec2FeatureExtractor.from_pretrained(reg["hf_id"], revision=reg["revision"]).do_normalize)
    if model.config.vocab_size != max(vocab.values()) + 1:
        raise RuntimeError("aligner model: vocabulary size mismatch")

    identity = {"aligner": "corpora.ctc_align.CTCAligner (Part A block algorithm)", "model": reg["hf_id"],
                "revision": reg["revision"], "license": reg["license"], "device": device, "min_score": min_score}
    return CTCAligner(make_forward(model, normalize, device), vocab, min_score, identity)
