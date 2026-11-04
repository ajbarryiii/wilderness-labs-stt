"""M1 (Kea-T), the stage 5 student's initialization and reference (keat-prime plan "Inputs").

- The latent checkpoint: $PARAKEET_TERNARY_ROOT/runs/main-M1-P2-lr5e-4/ckpt/latest.pt, SHA-256 M1_SHA256
  (parakeet-binary/binary_paths.py's pin, 003b1289...): finetune/parakeet-ternary/train.py's format 2, complete, at
  step 250,000 and weight fraction 1, holding the latent FP32 student and an AdamW state that ended at learning rate
  0. Its args have no quantizer key (train.py is ternary-native), so stage 4's MB1 check (quantizer == "binary") does
  not apply. The init check (load_m1_state) is instead: the SHA-256 before unpickling, then complete, the format,
  the step, the fraction, args arm "M1" with recipe "P2", and no quantizer. MB1's checkpoint fails it (its SHA-256,
  and its args name the binary quantizer).
- The export: .../main-M1-P2-lr5e-4/export/, parakeet-ternary-v1, export.safetensors SHA-256 M1_EXPORT_SHA256
  (a287e977...), quantized on CUDA.
Large-file hashes are cached under $RELABEL_ROOT/keat-prime/caches/sha256 by the file's identity (path, size, mtime,
inode), so the checkpoint is read once; the cache only avoids re-reading 7 GB, every caller compares with the pin.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import relabel_paths
from training import eightbit as _eb  # noqa: F401  puts finetune/parakeet-binary on sys.path
from training import keat_export as tke

import binary_paths  # noqa: E402  finetune/parakeet-binary/binary_paths.py (M1_SHA256)

M1_RUN = "main-M1-P2-lr5e-4"
M1_SHA256 = binary_paths.M1_SHA256
M1_STEP = 250_000
M1_ARM, M1_RECIPE = "M1", "P2"
M1_EXPORT_SHA256 = "a287e97719c451b785be2cd01ecc861fcaa010ebaa4ff2841783ec78fcd61503"
CKPT_FORMAT = 2     # finetune/parakeet-ternary/train.py CKPT_FORMAT (checked against train.py at load)


def m1_run() -> Path:
    return relabel_paths.PARAKEET_TERNARY / "runs" / M1_RUN


def m1_checkpoint() -> Path:
    return m1_run() / "ckpt" / "latest.pt"


def m1_export() -> Path:
    return m1_run() / "export"


def sha_cache_dir() -> Path:
    return tke.keat_root() / "caches" / "sha256"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
    return digest.hexdigest()


def cached_sha256(path: Path) -> str:
    """SHA-256 of a large file, cached by the file's identity (module docstring)."""
    path = Path(path).resolve()
    st = path.stat()
    identity = {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "inode": st.st_ino}
    entry = sha_cache_dir() / (hashlib.sha256(str(path).encode()).hexdigest()[:32] + ".json")
    if entry.exists():
        cached = json.loads(entry.read_text())
        if {k: cached.get(k) for k in identity} == identity:
            return cached["sha256"]
    digest = _sha256_file(path)
    if path.stat().st_mtime_ns == st.st_mtime_ns:
        entry.parent.mkdir(parents=True, exist_ok=True)
        tmp = entry.with_name(entry.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({**identity, "sha256": digest}) + "\n")
        os.replace(tmp, entry)
    return digest


def load_m1_state(path: Path | None = None, *, expected_sha256: str = M1_SHA256) -> tuple[dict, dict]:
    """(student state_dict, facts) of M1's latent training checkpoint after the init check (module docstring);
    RuntimeError otherwise, a SHA-256 mismatch before anything is unpickled. Memory-mapped: the optimizer state is
    never read."""
    import torch
    import train  # finetune/parakeet-ternary/train.py
    if train.CKPT_FORMAT != CKPT_FORMAT:
        raise RuntimeError(f"train.py's checkpoint format is {train.CKPT_FORMAT}, not M1's {CKPT_FORMAT}")
    path = Path(path or m1_checkpoint())
    digest = cached_sha256(path)
    if digest != expected_sha256:
        raise RuntimeError(f"{path}: SHA-256 {digest} is not M1's {expected_sha256}; refusing to unpickle it")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    args = payload.get("args") or {}
    facts = {"path": str(path), "sha256": digest, "step": payload.get("step"), "format": payload.get("format"),
             "complete": payload.get("complete"), "weight_fraction": payload.get("weight_fraction"),
             "arm": args.get("arm"), "recipe": args.get("recipe"), "quantizer": args.get("quantizer"),
             "run_name": args.get("run_name")}
    ok = (facts["complete"] is True and facts["format"] == CKPT_FORMAT and facts["step"] == M1_STEP
          and facts["weight_fraction"] == 1.0 and facts["arm"] == M1_ARM and facts["recipe"] == M1_RECIPE
          and "quantizer" not in args and isinstance(payload.get("student"), dict))
    if not ok:
        raise RuntimeError(f"{path} is not M1's final ternary checkpoint: {facts}")
    return payload["student"], facts


def init_record() -> dict:
    """The student's initialization as recorded with every stage 5 output."""
    return {"model": "M1", "path": str(m1_checkpoint()), "sha256": M1_SHA256, "step": M1_STEP, "format": CKPT_FORMAT,
            "weight_fraction": 1.0, "arm": M1_ARM, "recipe": M1_RECIPE, "quantizer": "ternary (train.py native)"}
