"""Mutant (encoder probe 2): each arm's frozen-encoder features, their checks, and the precomputed feature store
(mutant plan "Arms", "Features are per clip", "Encoder features").

    ./python mutant_features.py extract --arm D-B|D-T|D-Q [--smoke-shards N] [--pools A-B] [--workers W]
                                        [--budget-minutes M] [--lock-wait S]        precompute (GPU)
    ./python mutant_features.py checks --arm D-T|D-Q [--smoke-shards N]             per-clip (and Qwen rate) checks (GPU)

Every arm maps a 16 kHz clip (zero-padded to evaluate.MIN_DECODE_SECONDS as every record is: evaluate.pad_short) to
1024-d vectors, stored in bfloat16 (STORE_DTYPE); training, the held-out metric and the dev and domain scores all
read them through encode_clips, the one code path:

- D-B: B0 (Parakeet-TDT-0.6B-v2, evaluate.load_pretrained, FP32) as its records were scored: the eval-mode
  preprocessor (dither 0, pad_to 0) and the encoder under evaluate.strict_fp32, in evaluate.make_batches batches
  (64 clips, at most 1,200 s); each clip's encoder output up to its length, 12.5 Hz, rounded to bfloat16.
- D-T: T* (granite-speech-5.0-2b at probe 1's revision, loaded by encoder_probe_ctc.load_probe, bfloat16), one clip
  per pass: probe 1's features and encode (the middle layer's character-head logits as the encoder computed them),
  posterior_weighted_pool of the final layer over windows of 4 with weights 1 - P(blank), the first pooled_length
  vectors, cast to the BPE head's dtype (bfloat16): the BPE head's input, 12.5 Hz.
- D-Q: the audio encoder of Qwen3-ASR-1.7B (Qwen/Qwen3-ASR-1.7B-hf at the diagnostic's revision, loaded as the
  diagnostic loaded the model: AutoModelForMultimodalLM, bfloat16; only model.audio_tower is kept and run). Each
  clip's 128-bin log-mel comes from the snapshot's feature extractor on that clip alone, with the processor's audio
  settings (padding True, truncation False, attention mask, n_window 50: the clip is its own longest, so the STFT's
  end padding never depends on other clips; clips under 0.5 s are zero-padded by the extractor as the diagnostic's
  were), computed on the CPU as the diagnostic's processor did. Clips are batched (evaluate.make_batches, at most
  QWEN_MAX_BATCH_SECONDS of audio: the convolutions run every 1 s chunk of a batch at once) by
  zero-padding the mel axis and the mask; the encoder packs each clip into its own attention windows (cu_seqlens),
  and its output after ln_post (before the multimodal projector) is split per clip by
  _get_feat_extract_output_lengths of the clip's mask: 13 frames per second of audio.

Checks (plan "Features are per clip", "What 'within bf16 rounding' means" v3.2, "About D-Q's encoder"):
per_clip_check encodes 50 mixed-length dev clips in the production batching and each clip alone. Bitwise equal
passes (D-T, one clip per pass). Otherwise the arm passes only if, on the same clips, (1) FP32 isolation: with TF32
off for matmuls and convolutions (evaluate.strict_fp32; the settings are read back and recorded), batched and
clip-alone FP32 outputs differ by at most FP32_ISOLATION_MAX = 1e-3 max-abs; and (2) noise floor: per clip, the bf16
batched-vs-alone mean-abs and max-abs differences are at most NOISE_MARGIN = 1.1 x that clip's bf16-alone-vs-FP32-alone
values. checked(): if (1) fails, the arm is switched to one clip per pass (one_clip) and checked again; if only (2)
fails, nothing is extracted (exit 5) and the architect is told. qwen_rate_check compares D-Q's output frames with
_get_feat_extract_output_lengths and with the audio's duration (13 Hz). D-B has no per-clip check (it is batched as
B0's records were). After an arm's extraction, token_frame_stats writes per source the tokens per second and the
share of clips with more tokens than the arm's stored frames (reported, not a filter).

Store: <features>/<arm>/{pools,heldout}/<ppppp>.bin (the pool's clips' vectors, bfloat16, concatenated in C1 order;
training clips in pools/, held-out clips in heldout/) with <ppppp>.json written last (ids, frame offsets and counts,
dim, the .bin's SHA-256, the arm's identity and the plan's SHA-256). A committed pool of the same identity is skipped,
so extraction resumes per pool. <features> is $RELABEL_ROOT/encoder-probe/features in production and under
relabel_paths.SMOKE with --smoke-shards.

GPU discipline as encoder_probe_ctc.py: the parakeet GPU lock is taken before anything touches CUDA (exit 3 if busy);
the job's VRAM lease (AGENT_JOBS_VRAM_FRACTION) is applied on the current device index.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import evaluate  # finetune/parakeet-ternary/evaluate.py
import mutant_data as md
import relabel_paths

ARMS = ("D-B", "D-T", "D-Q")
DIM = 1024
STORE_DTYPE = torch.bfloat16
FRAME_RATE = {"D-B": 12.5, "D-T": 12.5, "D-Q": 13.0}
FEATURE_VERSION = 1
BATCH_SIZE, MAX_BATCH_SECONDS = 64, 1200.0       # evaluate.make_batches, as the B0 records' scoring
QWEN_MAX_BATCH_SECONDS = 300.0                   # D-Q: its convolutions run every 1 s chunk of a batch at once
FP32_ISOLATION_MAX = 1e-3
NOISE_MARGIN = 1.1
CHECK_CLIPS = 50
QWEN_HF_ID = "Qwen/Qwen3-ASR-1.7B-hf"
QWEN_REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
QWEN_RATE = 13.0
QWEN_RATE_TOLERANCE = 0.15
QWEN_AUDIO_KWARGS = {"sampling_rate": 16000, "padding": True, "truncation": False, "return_attention_mask": True,
                     "n_window": 50, "return_tensors": "pt"}
CHECKS_FAILED_EXIT = 5
STOPPED_EXIT = 76


class Stopped(BaseException):
    """SIGTERM or SIGINT, raised in the main thread so the process exits normally (CUDA synchronized, never killed
    mid-kernel: under MPS a client killed by a signal can fault the other clients on the GPU). A BaseException, as
    KeyboardInterrupt is, so no `except Exception` on the way (ours, NeMo's, transformers') swallows it."""


def _raise_stopped(signum, frame):
    raise Stopped(f"signal {signum}")


@contextlib.contextmanager
def clean_stop():
    """SIGTERM and SIGINT raise Stopped for the duration; on Stopped, CUDA is synchronized and STOPPED_EXIT returned by
    the caller (an extraction pool in progress stays uncommitted and is redone by the next run)."""
    old = {s: signal.signal(s, _raise_stopped) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for s, h in old.items():
            signal.signal(s, h)


class FeatureError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- arms

class Preparer:
    """Per-clip CPU work before encoding (picklable, so DataLoader workers run it). Default: the clip itself."""

    def __call__(self, clip: np.ndarray):
        return np.ascontiguousarray(clip, dtype=np.float32)

    def identity(self) -> dict:
        return {"kind": "waveform"}


class ArmEncoder:
    arm: str
    preparer: Preparer

    def batches(self, durations: list[float]) -> list[list[int]]:
        return make_batches(durations)

    def encode_prepared(self, items: list) -> list[torch.Tensor]:
        raise NotImplementedError

    def modules(self) -> list[nn.Module]:
        raise NotImplementedError

    def identity(self) -> dict:
        raise NotImplementedError


def make_batches(durations: list[float], max_seconds: float = MAX_BATCH_SECONDS) -> list[list[int]]:
    """evaluate.make_batches (longest first, 64 clips, at most max_seconds) over clips given by duration."""
    records = [{"duration": float(d), "id": f"{i:09d}"} for i, d in enumerate(durations)]
    return evaluate.make_batches(records, BATCH_SIZE, max_seconds)


def _freeze(module: nn.Module) -> nn.Module:
    module.requires_grad_(False)
    return module.eval()


class B0Arm(ArmEncoder):
    """D-B: B0's eval-mode preprocessor and encoder in strict FP32."""
    arm = "D-B"

    def __init__(self, model, device: torch.device, source: dict | None = None):
        self.model, self.device, self.source = _freeze(model).to(device), torch.device(device), source or {}
        self.preparer = Preparer()

    def modules(self) -> list[nn.Module]:
        return [self.model.preprocessor, self.model.encoder]

    def encode_prepared(self, items):
        lengths = torch.tensor([len(c) for c in items], dtype=torch.long)
        audio = torch.zeros(len(items), int(lengths.max()), dtype=torch.float32)
        for row, c in enumerate(items):
            audio[row, :len(c)] = torch.from_numpy(c)
        with evaluate.strict_fp32(self.device), evaluate.inference_settings(self.model):
            feats, feat_len = self.model.preprocessor(input_signal=audio.to(self.device),
                                                      length=lengths.to(self.device))
            enc, enc_len = self.model.encoder(audio_signal=feats, length=feat_len)
            return [enc[i, :, :int(enc_len[i])].T.to("cpu", STORE_DTYPE) for i in range(len(items))]

    def identity(self) -> dict:
        return {"arm": self.arm, "model": self.source.get("sha256"), "precision": "float32 (evaluate.strict_fp32), "
                "eval-mode preprocessor (dither 0, pad_to 0)", "batching": f"evaluate.make_batches({BATCH_SIZE}, "
                f"{MAX_BATCH_SECONDS})", "output": "encoder output up to its length"}


class TStarArm(ArmEncoder):
    """D-T: probe 1's T* encoder pass and posterior-weighted pooling, one clip per pass."""
    arm = "D-T"

    def __init__(self, probe, info: dict | None = None):
        self.probe, self.info = probe, info or {}
        _freeze(probe.encoder)
        _freeze(probe.out_llm)
        self.preparer = Preparer()

    def batches(self, durations):
        return [[i] for i in range(len(durations))]

    def modules(self) -> list[nn.Module]:
        return [self.probe.encoder]

    def encode_prepared(self, items):
        import encoder_probe_ctc as ep
        out = []
        with torch.no_grad():
            for clip in items:
                final, mid = ep.encode(self.probe.encoder, self.probe.features(clip))
                n = ep.pooled_length(len(clip), final.shape[1])
                pooled = ep.posterior_weighted_pool(final, ep.importance(mid), ep.WINDOW)[0, :n]
                out.append(pooled.to(self.probe.out_llm.weight.dtype).to("cpu", STORE_DTYPE))
        return out

    def identity(self) -> dict:
        import encoder_probe_ctc as ep
        return {"arm": self.arm, "hf_id": ep.HF_ID, "revision": ep.REVISION, "conventions_sha256":
                ep.conventions_sha256(), "precision": ep.PRECISION, "batching": "one clip per pass",
                "output": "posterior_weighted_pool(final layer, 1 - P_mid(blank), 4)[:pooled_length], in the BPE "
                          "head's dtype", "load": {k: self.info.get(k) for k in ("revision", "encoder_dtype")}}


class QwenPreparer(Preparer):
    """D-Q's per-clip log-mel: the snapshot's feature extractor on the clip alone (CPU)."""

    def __init__(self, feature_extractor):
        self.fe = feature_extractor

    def __call__(self, clip):
        out = self.fe(np.ascontiguousarray(clip, dtype=np.float32), **QWEN_AUDIO_KWARGS)
        return out["input_features"][0].numpy(), out["attention_mask"][0].numpy().astype(np.int64)

    def identity(self) -> dict:
        return {"kind": "log-mel per clip", "extractor": type(self.fe).__name__,
                "settings": {k: v for k, v in QWEN_AUDIO_KWARGS.items() if k != "return_tensors"},
                "config": {k: getattr(self.fe, k, None) for k in ("feature_size", "hop_length", "n_fft", "min_length",
                                                                   "n_window", "dither", "chunk_length")}}


class QwenArm(ArmEncoder):
    """D-Q: Qwen3-ASR's audio tower on per-clip mels, batched by zero-padding the mel axis and the mask."""
    arm = "D-Q"

    def __init__(self, feature_extractor, tower: nn.Module, device: torch.device, dtype=torch.bfloat16,
                 source: dict | None = None):
        self.tower, self.device, self.dtype = _freeze(tower).to(device), torch.device(device), dtype
        self.preparer = QwenPreparer(feature_extractor)
        self.source = source or {}
        self.one_clip = False            # set by checked() after an FP32 isolation failure

    def modules(self) -> list[nn.Module]:
        return [self.tower]

    def batches(self, durations):
        if self.one_clip:
            return [[i] for i in range(len(durations))]
        return make_batches(durations, QWEN_MAX_BATCH_SECONDS)

    def encode_prepared(self, items):
        return [h.to("cpu", STORE_DTYPE) for h in self.raw(items, self.dtype)]

    def raw(self, items, dtype) -> list[torch.Tensor]:
        """The tower's per-clip outputs computed in `dtype` (the tower cast for the call), before storage rounding."""
        from transformers.models.qwen3_asr.modeling_qwen3_asr import _get_feat_extract_output_lengths
        width = max(m.shape[1] for m, _ in items)
        mel = torch.zeros(len(items), items[0][0].shape[0], width, dtype=torch.float32)
        mask = torch.zeros(len(items), width, dtype=torch.long)
        for row, (m, k) in enumerate(items):
            mel[row, :, :m.shape[1]] = torch.from_numpy(m)
            mask[row, :len(k)] = torch.from_numpy(k)
        lengths = _get_feat_extract_output_lengths(mask.sum(-1)).tolist()
        own = next(self.tower.parameters()).dtype
        try:
            if dtype != own:
                self.tower.to(dtype)
            with torch.no_grad(), evaluate.strict_fp32(self.device):
                hidden = self.tower(input_features=mel.to(self.device, dtype),
                                    input_features_mask=mask.to(self.device)).last_hidden_state
        finally:
            if dtype != own:
                self.tower.to(own)
        if hidden.shape[0] != sum(lengths):
            raise FeatureError(f"Qwen encoder returned {hidden.shape[0]} frames, the lengths say {sum(lengths)}")
        return [h.float().cpu() for h in torch.split(hidden, lengths)]

    def identity(self) -> dict:
        return {"arm": self.arm, "hf_id": QWEN_HF_ID, "revision": QWEN_REVISION, "precision": str(self.dtype),
                "attn_implementation": getattr(getattr(self.tower, "config", None), "_attn_implementation", None),
                "mel": self.preparer.identity(), "batching": "one clip per pass" if self.one_clip else
                f"evaluate.make_batches({BATCH_SIZE}, {QWEN_MAX_BATCH_SECONDS}), mel axis and mask zero-padded", "output": "audio tower after ln_post, "
                "before the projector, split by _get_feat_extract_output_lengths", "source": self.source}


def encode_clips(enc: ArmEncoder, clips: list[np.ndarray], durations: list[float] | None = None
                 ) -> list[torch.Tensor]:
    """The arm's features of each clip (padded as every record is: evaluate.pad_short), in input order."""
    prepared = [enc.preparer(evaluate.pad_short(np.asarray(c, dtype=np.float32))[0]) for c in clips]
    return encode_prepared_in_batches(enc, prepared, durations or [len(c) / 16000 for c in clips])


def encode_prepared_in_batches(enc: ArmEncoder, prepared: list, durations: list[float]) -> list[torch.Tensor]:
    out: list[torch.Tensor | None] = [None] * len(prepared)
    for batch in enc.batches(durations):
        for i, f in zip(batch, enc.encode_prepared([prepared[i] for i in batch])):
            out[i] = f
    if any(f is None for f in out):
        raise FeatureError("the arm's batches do not cover every clip")
    return out


def frozen_state(enc: ArmEncoder) -> str:
    """SHA-256 of every parameter and buffer of the arm's encoder modules (unchanged by encoding)."""
    digest = hashlib.sha256()
    for k, m in enumerate(enc.modules()):
        for name, t in sorted(m.state_dict().items()):
            t = t.detach().cpu().contiguous()
            digest.update(f"{k}|{name}|{t.dtype}|{tuple(t.shape)}\n".encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return digest.hexdigest()


# ----------------------------------------------------------------------------- loading

def load_arm(arm: str, device: torch.device) -> ArmEncoder:
    if arm == "D-B":
        model = evaluate.load_pretrained("cpu")
        source = evaluate.describe_source("pretrained", None, "pretrained Parakeet-TDT-0.6B-v2, FP32, no training")
        return B0Arm(model, device, source)
    if arm == "D-T":
        import encoder_probe_ctc as ep
        probe, info = ep.load_probe(device)
        return TStarArm(probe, info)
    if arm == "D-Q":
        return load_qwen(device)
    raise ValueError(f"unknown arm {arm!r}")


def load_qwen(device: torch.device) -> QwenArm:
    """The diagnostic's Qwen3-ASR (AutoModelForMultimodalLM, bfloat16, pinned revision), loaded on the CPU; only its
    audio tower is kept and moved to the device."""
    from transformers import AutoFeatureExtractor, AutoModelForMultimodalLM
    model = AutoModelForMultimodalLM.from_pretrained(QWEN_HF_ID, revision=QWEN_REVISION, dtype=torch.bfloat16)
    tower = model.model.audio_tower
    cfg = tower.config
    if (cfg.d_model, cfg.num_mel_bins, cfg.encoder_layers, cfg.n_window, cfg.n_window_infer) != (1024, 128, 24, 50, 800):
        raise FeatureError(f"unexpected Qwen audio encoder config: {cfg}")
    model.model.audio_tower = None
    del model
    fe = AutoFeatureExtractor.from_pretrained(QWEN_HF_ID, revision=QWEN_REVISION)
    return QwenArm(fe, tower, device, torch.bfloat16, {"loaded_as": "AutoModelForMultimodalLM.from_pretrained("
                                                       "dtype=bfloat16).model.audio_tower"})


# ----------------------------------------------------------------------------- checks

def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape:
        return {"same_shape": False, "equal": False, "max_abs_diff": None, "rms": None, "ratio": None}
    x, y = a.float(), b.float()
    diff = float((x - y).abs().max()) if x.numel() else 0.0
    rms = float(x.pow(2).mean().sqrt()) if x.numel() else 0.0
    return {"same_shape": True, "equal": bool(torch.equal(a, b)), "max_abs_diff": diff, "rms": rms,
            "ratio": diff / rms if rms > 0 else (0.0 if diff == 0 else float("inf")),
            "differing_fraction": float((x != y).float().mean()) if x.numel() else 0.0}


def tf32_state() -> dict:
    return {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def _diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a.float() - b.float()).abs()
    return (float(d.mean()), float(d.max())) if d.numel() else (0.0, 0.0)


def noise_floor(enc: ArmEncoder, clips: list[np.ndarray], durations: list[float]) -> dict:
    """Parts (1) and (2) of the v3.2 rule from the arm's raw(items, dtype) outputs (before storage rounding)."""
    prepared = [enc.preparer(evaluate.pad_short(np.asarray(c, dtype=np.float32))[0]) for c in clips]
    device = getattr(enc, "device", torch.device("cpu"))

    def outputs(dtype) -> tuple[list, list]:
        batched: list = [None] * len(prepared)
        for b in enc.batches(durations):
            for i, f in zip(b, enc.raw([prepared[i] for i in b], dtype)):
                batched[i] = f
        return batched, [enc.raw([p], dtype)[0] for p in prepared]

    with evaluate.strict_fp32(device):                # TF32 off for matmuls and convolutions
        tf32 = tf32_state()
        f32_b, f32_a = outputs(torch.float32)
    b16_b, b16_a = outputs(torch.bfloat16)
    rows = []
    for k in range(len(prepared)):
        iso = _diff(f32_b[k], f32_a[k])[1]
        bm, bx = _diff(b16_b[k], b16_a[k])
        rm, rx = _diff(b16_a[k], f32_a[k])
        rows.append({"fp32_batched_vs_alone_max": iso, "bf16_batched_vs_alone_mean": bm,
                     "bf16_batched_vs_alone_max": bx, "bf16_vs_fp32_alone_mean": rm, "bf16_vs_fp32_alone_max": rx,
                     "within_floor": bm <= NOISE_MARGIN * rm and bx <= NOISE_MARGIN * rx})
    iso_max = max(r["fp32_batched_vs_alone_max"] for r in rows)
    return {"tf32": tf32,
            "isolation": {"max_abs": iso_max, "limit": FP32_ISOLATION_MAX, "passed": iso_max <= FP32_ISOLATION_MAX},
            "bf16": {"margin": NOISE_MARGIN, "clips_over": sum(not r["within_floor"] for r in rows),
                     "max_ratio_mean": max(r["bf16_batched_vs_alone_mean"] / r["bf16_vs_fp32_alone_mean"]
                                           if r["bf16_vs_fp32_alone_mean"] else float("inf") for r in rows),
                     "max_ratio_max": max(r["bf16_batched_vs_alone_max"] / r["bf16_vs_fp32_alone_max"]
                                          if r["bf16_vs_fp32_alone_max"] else float("inf") for r in rows),
                     "passed": all(r["within_floor"] for r in rows)},
            "per_clip": rows}


def per_clip_check(enc: ArmEncoder, clips: list[np.ndarray], durations: list[float]) -> dict:
    """Production-batched features against each clip encoded alone, by the v3.2 rule (module docstring)."""
    batched = encode_clips(enc, clips, durations)
    rows = [compare(b, encode_clips(enc, [c], [d])[0]) for b, c, d in zip(batched, clips, durations)]
    same = all(r["same_shape"] for r in rows)
    equal = same and all(r["equal"] for r in rows)
    out = {"arm": enc.arm, "clips": len(clips), "batch_sizes": [len(b) for b in enc.batches(durations)],
           "same_shapes": same, "equal": equal,
           "max_abs_diff": max(r["max_abs_diff"] for r in rows) if same else None,
           "differing_fraction_max": max(r["differing_fraction"] for r in rows) if same else None,
           "rule": "bitwise equal; otherwise FP32 isolation (max-abs <= 1e-3, TF32 off) and the bf16 noise floor "
                   "(per clip mean-abs and max-abs <= 1.1 x bf16-vs-FP32)"}
    if equal:
        return out | {"passed": True, "failed_part": None}
    if not same:
        return out | {"passed": False, "failed_part": "shapes"}
    if not hasattr(enc, "raw"):
        return out | {"passed": False, "failed_part": "not bitwise equal and no FP32 path"}
    nf = noise_floor(enc, clips, durations)
    failed = None if nf["isolation"]["passed"] and nf["bf16"]["passed"] else \
        ("isolation" if not nf["isolation"]["passed"] else "noise_floor")
    return out | {"noise_floor": nf, "passed": failed is None, "failed_part": failed}


def checked(enc: ArmEncoder, clips: list[np.ndarray], durations: list[float]) -> dict:
    """run_checks' clips through the plan's checks; after an FP32 isolation failure the arm is switched to one clip
    per pass and checked again (the first result is kept under "first")."""
    out = check_set(enc, clips, durations)
    if not out["passed"] and (out.get("per_clip") or {}).get("failed_part") == "isolation" and \
            hasattr(enc, "one_clip") and not enc.one_clip:
        enc.one_clip = True
        out = check_set(enc, clips, durations) | {"first": out, "fallback": "one clip per pass after the FP32 "
                                                                            "isolation failure"}
    return out


def check_set(enc: ArmEncoder, clips: list[np.ndarray], durations: list[float]) -> dict:
    out = {"arm": enc.arm, "per_clip": per_clip_check(enc, clips, durations)}
    if enc.arm == "D-Q":
        out["rate"] = qwen_rate_check(enc, clips, durations)
    out["passed"] = all(v["passed"] for k, v in out.items() if isinstance(v, dict) and "passed" in v)
    return out


def qwen_rate_check(enc: QwenArm, clips: list[np.ndarray], durations: list[float]) -> dict:
    """D-Q's output frames against _get_feat_extract_output_lengths and against the clips' durations."""
    from transformers.models.qwen3_asr.modeling_qwen3_asr import _get_feat_extract_output_lengths
    feats = encode_clips(enc, clips, durations)
    expect = []
    for c in clips:
        _, mask = enc.preparer(evaluate.pad_short(np.asarray(c, dtype=np.float32))[0])
        expect.append(int(_get_feat_extract_output_lengths(torch.tensor(int(mask.sum())))))
    frames = [int(f.shape[0]) for f in feats]
    seconds = float(sum(durations))
    rate = sum(frames) / seconds
    per_clip = [f / d for f, d in zip(frames, durations)]
    return {"clips": len(clips), "audio_seconds": seconds, "frames": sum(frames), "frames_per_second": rate,
            "per_clip_rate_min": min(per_clip), "per_clip_rate_max": max(per_clip),
            "lengths_match_formula": frames == expect, "expected_rate": QWEN_RATE,
            "passed": frames == expect and abs(rate - QWEN_RATE) <= QWEN_RATE_TOLERANCE}


def check_clips(n: int = CHECK_CLIPS) -> list[dict]:
    """50 dev clips of mixed lengths: per H4 and YODAS dev set, its shortest, its longest and clips spread evenly over
    the duration-sorted manifest (dev audio only)."""
    out = []
    per = n // len(relabel_paths.SETS)
    for name in relabel_paths.SETS:
        records = sorted(evaluate.read_manifest(evaluate.resolve_manifest(name)[1]), key=lambda r: (r["duration"], r["id"]))
        idx = sorted({0, len(records) - 1} | {round(j * (len(records) - 1) / (per - 1)) for j in range(per)})
        out += [records[i] for i in idx][:per]
    return out


# ----------------------------------------------------------------------------- store

def store_paths(arm_dir: Path, kind: str, p: int) -> dict[str, Path]:
    if kind not in ("pools", "heldout"):
        raise ValueError(kind)
    base = Path(arm_dir) / kind
    return {"bin": base / f"{p:05d}.bin", "meta": base / f"{p:05d}.json"}


def write_features(paths: dict[str, Path], ids: list[str], feats: list[torch.Tensor], identity: dict) -> dict:
    if len(ids) != len(feats):
        raise FeatureError("ids and features differ in length")
    dims = {int(f.shape[1]) for f in feats if f.dim() == 2}
    if len(dims) > 1 or any(f.dtype != STORE_DTYPE or f.dim() != 2 for f in feats):
        raise FeatureError("features must be [frames, dim] bfloat16 with one dim")
    dim = dims.pop() if dims else 0
    paths["bin"].parent.mkdir(parents=True, exist_ok=True)
    tmp = paths["bin"].with_name(paths["bin"].name + f".tmp{os.getpid()}")
    digest, offsets, frames, offset = hashlib.sha256(), [], [], 0
    with open(tmp, "wb") as out:
        for f in feats:
            raw = f.contiguous().view(torch.int16).numpy().astype("<i2").tobytes()
            out.write(raw)
            digest.update(raw)
            offsets.append(offset)
            frames.append(int(f.shape[0]))
            offset += int(f.shape[0])
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, paths["bin"])
    meta = {"ids": list(ids), "offsets": offsets, "frames": frames, "dim": dim, "dtype": "bfloat16",
            "total_frames": offset, "bin_sha256": digest.hexdigest(), "identity": identity}
    md.write_json(paths["meta"], meta)
    return meta


def committed(paths: dict[str, Path], ids: list[str], identity: dict) -> bool:
    if not paths["meta"].exists() or not paths["bin"].exists():
        return False
    meta = json.loads(paths["meta"].read_text())
    return meta.get("identity") == identity and meta.get("ids") == list(ids)


class FeatureFile:
    """One committed feature file, read whole into memory (bfloat16 [total_frames, dim])."""

    def __init__(self, paths: dict[str, Path], verify: bool = False):
        self.meta = json.loads(paths["meta"].read_text())
        raw = Path(paths["bin"]).read_bytes()
        if verify and hashlib.sha256(raw).hexdigest() != self.meta["bin_sha256"]:
            raise FeatureError(f"{paths['bin']}: content differs from its marker")
        expect = self.meta["total_frames"] * self.meta["dim"] * 2
        if len(raw) != expect:
            raise FeatureError(f"{paths['bin']}: {len(raw)} bytes, the marker says {expect}")
        data = np.frombuffer(raw, dtype="<i2").copy()
        self.data = torch.from_numpy(data).view(STORE_DTYPE).reshape(self.meta["total_frames"], self.meta["dim"]) \
            if self.meta["dim"] else torch.zeros(0, 0, dtype=STORE_DTYPE)
        self.ids = self.meta["ids"]
        self.index = {i: (o, n) for i, o, n in zip(self.ids, self.meta["offsets"], self.meta["frames"])}
        _drop_cache(paths["bin"])

    def get(self, item_id: str) -> torch.Tensor:
        o, n = self.index[item_id]
        return self.data[o:o + n] if self.meta["dim"] else torch.zeros(n, 0, dtype=STORE_DTYPE)


def _drop_cache(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


def store_identity(enc: ArmEncoder, data: md.Data) -> dict:
    return {"version": FEATURE_VERSION, "arm": enc.arm, "encoder": enc.identity(), "store_dtype": "bfloat16",
            "pad_short_seconds": evaluate.MIN_DECODE_SECONDS, "plan_sha256": data.plan.sha256}


def _as_is(item):
    """The DataLoader's collate: items unchanged (no conversion of numpy arrays)."""
    return item


class _PoolSet(torch.utils.data.Dataset):
    """Batch k of a pool: (indices, prepared clips), audio decoded and checked against C1's SHA-256."""

    def __init__(self, path: Path, entries: list[dict], batches: list[list[int]], preparer: Preparer):
        self.path, self.entries, self.batches, self.preparer = Path(path), entries, batches, preparer

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, k):
        out = []
        with open(self.path, "rb") as f:
            for i in self.batches[k]:
                e = self.entries[i]
                clip = md.decode_checked(e["id"], os.pread(f.fileno(), e["length"], e["offset"]), e)
                out.append(self.preparer(evaluate.pad_short(clip)[0]))
        return self.batches[k], out


def extract_pool(lay: md.Layout, enc: ArmEncoder, data: md.Data, p: int, workers: int) -> dict:
    audio = md.PoolAudio(lay, data, p)
    ids = data.pool_ids(p)
    entries = [audio.entries[i] for i in ids]
    durations = [data.clips[i]["duration"] for i in ids]
    batches = enc.batches(durations)
    loader = torch.utils.data.DataLoader(_PoolSet(audio.path, entries, batches, enc.preparer), batch_size=None,
                                         shuffle=False, num_workers=workers, persistent_workers=False,
                                         collate_fn=_as_is)
    feats: list[torch.Tensor | None] = [None] * len(ids)
    t_enc, t0 = 0.0, time.perf_counter()
    for idx, items in loader:
        s = time.perf_counter()
        for i, f in zip(idx, enc.encode_prepared(items)):
            feats[i] = f
        t_enc += time.perf_counter() - s
    if any(f is None for f in feats):
        raise FeatureError(f"pool {p}: some clips were not encoded")
    identity = store_identity(enc, data)
    arm_dir = lay.features / enc.arm
    for kind, split in (("pools", "train"), ("heldout", "heldout")):
        sel = [k for k, i in enumerate(ids) if data.clips[i]["split"] == split]
        write_features(store_paths(arm_dir, kind, p), [ids[k] for k in sel], [feats[k] for k in sel], identity)
    hours = sum(durations) / 3600
    return {"pool": p, "clips": len(ids), "audio_hours": hours, "encode_s": t_enc, "wall_s": time.perf_counter() - t0,
            "frames": sum(int(f.shape[0]) for f in feats)}


def token_frame_stats(lay: md.Layout, arm: str, data: md.Data) -> dict:
    """Per source, over every selected clip: tokens per second and the share of clips with more tokens than the arm's
    stored frames (reported, not a filter)."""
    frames = {}
    for p in range(data.plan.n_pools):
        for kind in ("pools", "heldout"):
            meta = json.loads(store_paths(lay.features / arm, kind, p)["meta"].read_text())
            frames.update(zip(meta["ids"], meta["frames"]))
    out = {}
    for s in md.SOURCES:
        rows = [c for c in data.clips.values() if c["source"] == s]
        if rows:
            tokens = sum(len(c["tokens"]) for c in rows)
            out[s] = {"clips": len(rows), "tokens_per_second": tokens / sum(c["duration"] for c in rows),
                      "frames_per_second": sum(frames[c["id"]] for c in rows) / sum(c["duration"] for c in rows),
                      "share_tokens_over_frames": sum(len(c["tokens"]) > frames[c["id"]] for c in rows) / len(rows)}
    return out


_TMP = re.compile(r"\.tmp(\d+)$")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_stale_tmp(directory: Path) -> list[str]:
    """Delete the *.tmp<pid> files that earlier interrupted writers left in `directory` (their pid no longer runs);
    a running writer's file and every committed file are kept."""
    removed = []
    if Path(directory).is_dir():
        for f in sorted(Path(directory).iterdir()):
            m = _TMP.search(f.name)
            if m and int(m.group(1)) != os.getpid() and not _alive(int(m.group(1))):
                f.unlink(missing_ok=True)
                removed.append(f.name)
    return removed


def extract(lay: md.Layout, enc: ArmEncoder, pools: list[int] | None = None, workers: int = 6,
            budget_minutes: float | None = None) -> list[dict]:
    """Precompute (and commit per pool) the arm's features of every pool's clips; committed pools are skipped."""
    relabel_paths.require_mount()
    data = md.Data.load(lay)
    identity = store_identity(enc, data)
    arm_dir = lay.features / enc.arm
    for kind in ("pools", "heldout"):
        stale = remove_stale_tmp(arm_dir / kind)
        if stale:
            print(f"extract {enc.arm}: removed stale temporary files of interrupted runs in {kind}/: {stale}",
                  flush=True)
    done, t0 = [], time.time()
    for p in (range(data.plan.n_pools) if pools is None else pools):
        ids = data.pool_ids(p)
        want = {kind: [i for i in ids if data.clips[i]["split"] == split]
                for kind, split in (("pools", "train"), ("heldout", "heldout"))}
        if all(committed(store_paths(arm_dir, k, p), want[k], identity) for k in want):
            continue
        if budget_minutes is not None and time.time() - t0 > budget_minutes * 60:
            print(f"extract {enc.arm}: budget reached before pool {p}; the next run resumes", flush=True)
            break
        r = extract_pool(lay, enc, data, p, workers)
        done.append(r)
        print(f"extract {enc.arm}: pool {p} committed: {r['clips']} clips, {r['audio_hours']:.2f} h, encode "
              f"{r['encode_s']:.0f} s, wall {r['wall_s']:.0f} s ({r['audio_hours'] / max(r['wall_s'], 1e-9) * 3600:.0f}"
              " audio h per wall h)", flush=True)
        md.write_json(arm_dir / "extract-log" / f"{p:05d}.json", r)
    if all(committed(store_paths(arm_dir, k, p), [i for i in data.pool_ids(p) if data.clips[i]["split"] == split],
                     identity) for p in range(data.plan.n_pools) for k, split in (("pools", "train"),
                                                                                 ("heldout", "heldout"))):
        md.write_json(arm_dir / "token-frames.json", token_frame_stats(lay, enc.arm, data))
    return done


# ----------------------------------------------------------------------------- GPU entry points

def apply_lease(device: torch.device) -> None:
    fraction = os.environ.get("AGENT_JOBS_VRAM_FRACTION")
    if fraction and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(fraction), torch.cuda.current_device())


def peak_memory(device: torch.device) -> dict:
    cuda = device.type == "cuda"
    return {"gpu_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None,
            "gpu_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30 if cuda else None,
            "host_max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}


def run_checks(enc: ArmEncoder) -> dict:
    """The plan's checks for the arm on check_clips(): none for D-B (batched as B0's records were), checked() for D-T and
    D-Q."""
    if enc.arm == "D-B":
        return {"arm": enc.arm, "passed": True, "note": "no per-clip check: D-B is batched as B0's records were"}
    records = check_clips()
    clips = [evaluate.load_audio(r["audio_filepath"]) for r in records]
    durations = [r["duration"] for r in records]
    return {"clip_ids": [r["id"] for r in records]} | checked(enc, clips, durations)


def checks_path(lay: md.Layout, arm: str) -> Path:
    return lay.features / arm / "checks.json"


def locked(label: str, wait: float, fn):
    import score_teacher
    try:
        with clean_stop(), score_teacher.gpu_lock(label, wait):
            return fn()
    except score_teacher.LockBusy as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return score_teacher.LOCK_BUSY_EXIT
    except Stopped as exc:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        print(f"stopped ({exc}); CUDA synchronized; the next run resumes", file=sys.stderr)
        return STOPPED_EXIT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("extract", "checks"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--smoke-shards", type=int)
    parser.add_argument("--pools", help="A-B (inclusive) or A")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--budget-minutes", type=float)
    parser.add_argument("--lock-wait", type=float, default=3600.0)
    args = parser.parse_args(argv)
    lay = md.layout(args.smoke_shards)
    relabel_paths.require_mount()
    device = torch.device("cuda")

    def run() -> int:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required")
        apply_lease(device)
        t0 = time.perf_counter()
        enc = load_arm(args.arm, device)
        load_s = time.perf_counter() - t0
        path = checks_path(lay, args.arm)
        old = json.loads(path.read_text()) if path.exists() else None
        if old and old.get("passed") and old.get("fallback") and hasattr(enc, "one_clip"):
            enc.one_clip = True                       # the check passed after the one-clip fallback
        if old and old.get("passed") and old.get("encoder") == enc.identity():
            checks = old
        else:
            checks = run_checks(enc) | {"encoder": enc.identity(), "load_s": load_s,
                                        "memory": peak_memory(device)}
            md.write_json(path, checks)
        print(json.dumps({k: v for k, v in checks.items() if k not in ("clip_ids", "encoder")}, default=str)[:3000],
              flush=True)
        if not checks["passed"]:
            print("feature checks failed: nothing extracted; report to the architect", file=sys.stderr)
            return CHECKS_FAILED_EXIT
        if args.cmd == "checks":
            return 0
        pools = None
        if args.pools:
            a, _, b = args.pools.partition("-")
            pools = list(range(int(a), int(b or a) + 1))
        extract(lay, enc, pools, args.workers, args.budget_minutes)
        md.write_json(lay.features / args.arm / f"extract-memory-{os.getpid()}.json", peak_memory(device))
        return 0

    return locked(f"relabel mutant_features {args.cmd} {args.arm}", args.lock_wait, run)


if __name__ == "__main__":
    sys.exit(main())
