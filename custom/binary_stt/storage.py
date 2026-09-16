"""Fail-closed artifact placement and atomic persistence on the data disk."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

ROOT = Path("/mnt/hd/wilderness-labs-stt/binary-stt")
DISK = Path("/mnt/hd")


def ensure_artifact_path(path=ROOT, *, create_parent=False):
    if not os.path.ismount(DISK):
        raise RuntimeError("/mnt/hd is not mounted; refusing artifact writes")
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(ROOT) or not ROOT.resolve().is_relative_to(DISK):
        raise ValueError(f"Artifact path must remain under {ROOT}: {path}")
    # Both the lexical root and any symlink targets must remain on the mount.
    if not path.is_relative_to(DISK.resolve()):
        raise ValueError("Artifact path escaped mounted data disk")
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def storage():
    root = ensure_artifact_path()
    root.mkdir(parents=True, exist_ok=True)
    return root


def configure_environment():
    storage()
    locations = {
        "HF_HOME": "cache/huggingface", "HF_HUB_CACHE": "cache/huggingface/hub",
        "HF_DATASETS_CACHE": "cache/datasets", "XDG_CACHE_HOME": "cache",
        "TORCH_HOME": "cache/torch", "TRITON_CACHE_DIR": "cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "cache/inductor", "TORCH_EXTENSIONS_DIR": "cache/extensions",
        "CUDA_CACHE_PATH": "cache/cuda", "TMPDIR": "tmp", "UV_CACHE_DIR": "cache/uv",
        "NUMBA_CACHE_DIR": "cache/numba", "HF_XET_CACHE": "cache/xet",
    }
    for key, suffix in locations.items():
        path = ensure_artifact_path(ROOT / suffix)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    tempfile.tempdir = str(ROOT / "tmp")
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", WANDB_MODE="disabled",
                      TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, obj):
    path = ensure_artifact_path(path, create_parent=True)
    data = json.dumps(obj, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        ensure_artifact_path(path)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_json(path, obj):
    path = ensure_artifact_path(path, create_parent=True)
    with path.open("a") as handle:
        handle.write(json.dumps(obj, allow_nan=False, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_torch_save(path, obj):
    import torch
    path = ensure_artifact_path(path, create_parent=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(obj, handle)
            handle.flush()
            os.fsync(handle.fileno())
        ensure_artifact_path(path)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_free_space(minimum_gib=20):
    root = storage()
    free = shutil.disk_usage(root).free
    if free < minimum_gib * 2**30:
        raise RuntimeError(f"Insufficient data-disk space: {free / 2**30:.1f} GiB free, {minimum_gib} required")
    return free


def heartbeat(run_dir, stage, **extra):
    write_json(Path(run_dir) / "heartbeat.json", {"time": time.time(), "stage": stage, "pid": os.getpid(), **extra})


@contextlib.contextmanager
def gpu_lock(device="cuda"):
    """Cooperate with existing experiments; never change/kill foreign GPU work."""
    if not str(device).startswith("cuda"):
        yield
        return
    storage()
    # Existing experiments use this shared lock. Its parent is on the verified disk.
    lock_path = DISK / "wilderness-labs-stt/stt-distillation/active.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another training experiment holds the shared GPU lock") from exc
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=10)
            foreign = [line.strip() for line in output.splitlines() if line.strip() and line.strip() != str(os.getpid())]
            if foreign:
                raise RuntimeError(f"GPU already has compute processes: {', '.join(foreign)}")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
