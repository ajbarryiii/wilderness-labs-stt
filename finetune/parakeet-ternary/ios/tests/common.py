"""Shared helpers of the ios/ tests: import paths, artifact directory, development clips, error metrics."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

IOS = Path(__file__).resolve().parents[1]
PARENT = IOS.parent
for path in (str(IOS), str(PARENT)):
    if path not in sys.path:
        sys.path.insert(0, path)

MAC_ARTIFACTS = Path("/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios")
NIXOS_ARTIFACTS = Path("/mnt/hd/wilderness-labs-stt/parakeet-ios")
GOLDEN_SEED = 0
CLIP_SET = "librispeech_dev_clean"
CLIP_COUNT = 3
CLIP_SECONDS = (2.0, 6.0)
# dev_clips() on the NixOS manifests; recorded so golden checks on any machine know what to expect
EXPECTED_CLIP_IDS = ("librispeech:1272-128104-0000", "librispeech:1272-128104-0001", "librispeech:1272-128104-0006")


def artifacts_dir() -> Path:
    """The machine's artifact directory (artifacts.root(): Mac outside any repository; NixOS on the mounted disk)."""
    import artifacts

    return artifacts.root()


def dev_clips() -> list[dict]:
    """The first CLIP_COUNT LibriSpeech dev-clean utterances by id of CLIP_SECONDS duration (NixOS manifests)."""
    import paths

    with open(paths.MANIFESTS / f"{CLIP_SET}.jsonl") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    lo, hi = CLIP_SECONDS
    chosen = sorted((r for r in records if lo <= r["duration"] <= hi), key=lambda r: r["id"])[:CLIP_COUNT]
    if tuple(r["id"] for r in chosen) != EXPECTED_CLIP_IDS:
        raise RuntimeError(f"development clip selection changed: {[r['id'] for r in chosen]}")
    return chosen


def load_audio(path: str | Path) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 16000 or audio.shape[1] != 1:
        raise ValueError(f"{path}: expected 16 kHz mono")
    return np.ascontiguousarray(audio[:, 0])


def batch(clips: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-padded [B, N] float32 and lengths [B]."""
    lengths = torch.tensor([len(c) for c in clips], dtype=torch.long)
    audio = torch.zeros(len(clips), int(lengths.max()))
    for i, c in enumerate(clips):
        audio[i, :len(c)] = torch.from_numpy(c)
    return audio, lengths


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


REL_CEILING = 1e-5   # DESIGN.md gate 1 (reference vs NeMo, FP32)
ABS_CEILING = 1e-4
TAU_FP32 = 1e-6


def errors(a, r, tau: float = TAU_FP32) -> tuple[float, float]:
    """DESIGN.md error definitions: rel = ||a - r||_2 / max(||r||_2, tau sqrt(n)), abs = max|a - r| / max(RMS(r), tau)."""
    a = torch.as_tensor(np.asarray(a) if not torch.is_tensor(a) else a).to(torch.float64)
    r = torch.as_tensor(np.asarray(r) if not torch.is_tensor(r) else r).to(torch.float64)
    if a.shape != r.shape:
        raise ValueError(f"shape {tuple(a.shape)} != {tuple(r.shape)}")
    n = r.numel()
    if n == 0:
        return 0.0, 0.0
    diff = a - r
    rel = float(diff.norm() / max(float(r.norm()), tau * n ** 0.5))
    rms = float(r.pow(2).mean().sqrt())
    return rel, float(diff.abs().max() / max(rms, tau))


def finite(*tensors) -> bool:
    return all(bool(torch.isfinite(torch.as_tensor(t)).all()) for t in tensors)


def report(name: str, **values) -> None:
    """One machine-readable result line."""
    print(f"RESULT {name} " + json.dumps(values, default=float), flush=True)


def peak_rss_mb() -> float:
    import randomweights

    return round(randomweights.peak_rss_mb(), 1)
