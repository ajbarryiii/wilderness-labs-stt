"""Hash-verified loading of the pinned Whisper tiny.en checkpoint, and source provenance.

Every load first checks the files listed in the checkpoint's lock.json against
their SHA-256 and the pinned revision. Files not listed in the lock (the local
download cache) are not checked. Loading is offline only.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lock(model_dir: Path = paths.MODEL_DIR, files: Iterable[str] | None = None) -> dict:
    """Raise unless the revision and every locked file (or the given subset) match lock.json."""
    lock = json.loads((model_dir / "lock.json").read_text())
    if lock.get("revision") != paths.MODEL_REVISION:
        raise ValueError(f"{model_dir}: lock revision {lock.get('revision')} != {paths.MODEL_REVISION}")
    for name in lock["files"] if files is None else files:
        if sha256(model_dir / name) != lock["files"][name]:
            raise ValueError(f"checksum mismatch for {model_dir / name}")
    return lock


def load_processor() -> WhisperProcessor:
    verify_lock()
    return WhisperProcessor.from_pretrained(paths.MODEL_DIR, local_files_only=True)


def load_model(device: str | torch.device = "cpu",
               dtype: torch.dtype = torch.float32) -> WhisperForConditionalGeneration:
    verify_lock()
    model = WhisperForConditionalGeneration.from_pretrained(
        paths.MODEL_DIR, local_files_only=True, use_safetensors=True, dtype=dtype)
    return model.to(device)


def base_provenance() -> dict[str, str]:
    """Identity of the pinned base checkpoint, recorded in export manifests and eval JSON."""
    lock = paths.MODEL_DIR / "lock.json"
    return {"base_model": json.loads(lock.read_text())["model"],
            "base_revision": paths.MODEL_REVISION, "base_lock_sha256": sha256(lock)}


def source_hashes() -> dict[str, str]:
    """SHA-256 of every experiment source file (*.py next to this module) for run provenance."""
    return {path.name: sha256(path) for path in sorted(paths.HERE.glob("*.py"))}
