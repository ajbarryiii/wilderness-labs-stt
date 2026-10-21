"""Locations, import paths and environment facts for gpu1bit (plans/gpu-kernel.md A7, A9, A10).

Importing this module puts finetune/parakeet-binary and finetune/parakeet-ternary on sys.path (so
`binary_export`, `export`, `paths` and M1's `inference` package resolve) without changing either
directory. Use gpu1bit/shared/python for every GPU run.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import sys
import time
from pathlib import Path

SHARED = Path(__file__).resolve().parent
GPU1BIT = SHARED.parent
BINARY_DIR = GPU1BIT.parent
REPO = BINARY_DIR.parents[1]
TERNARY_DIR = REPO / "finetune" / "parakeet-ternary"
for _p in (TERNARY_DIR, BINARY_DIR):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

MOUNT = Path("/mnt/hd")
ARTIFACTS_ROOT = Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit")
ARTIFACTS = ARTIFACTS_ROOT / "H"
MB1_EXPORT = Path("/mnt/hd/wilderness-labs-stt/parakeet-binary/runs/main-binB-m1-lr2e-4/export")
MB1_SHA256 = "9873d2f1a53127c673d0201db98cd3ba859ba432f334fef55e9e24fd057b9651"
M1_EXPORT = Path("/mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/export")
DERIVED_EXPORT = ARTIFACTS / "derived" / "MB1-ternary-layout"
LOCK_HOLD_LIMIT_S = 600.0
POWER_LIMIT_W = 400.0


def require_mount() -> None:
    if not os.path.ismount(MOUNT):
        raise RuntimeError("/mnt/hd is not mounted; refusing artifact writes.")


def artifacts(*parts: str) -> Path:
    """ARTIFACTS/<parts>, created after the mount check."""
    require_mount()
    path = ARTIFACTS.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def wrapper() -> dict:
    """Which interpreter wrapper launched this process, judged from the environment it exports."""
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    nvrtc = "inference-efficiency/deps/nvidia/cuda_nvrtc/lib" in ld
    name = os.environ.get("GPU1BIT_WRAPPER")
    if name is None:
        triton_cache = os.environ.get("TRITON_CACHE_DIR", "")
        name = ("parakeet-ternary/python" if "parakeet-ternary" in triton_cache else
                "parakeet-binary/python" if "parakeet-binary" in triton_cache else "unknown")
    return {"wrapper": name, "nvrtc_on_library_path": nvrtc, "executable": sys.executable,
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR")}


def require_shared_wrapper() -> None:
    if os.environ.get("GPU1BIT_WRAPPER") != "gpu1bit/shared/python":
        raise RuntimeError("timed runs must use gpu1bit/shared/python (plans/gpu-kernel.md A7)")


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_hashes(*directories: Path) -> dict[str, str]:
    out = {}
    for d in directories:
        for p in sorted(Path(d).rglob("*.py")):
            if "__pycache__" not in p.parts:
                out[str(p.relative_to(REPO))] = sha256_file(p)
    return out


@contextlib.contextmanager
def gpu_lock(label: str):
    """paths.gpu_lock for this whole block; the holder must stay <= LOCK_HOLD_LIMIT_S (A9).

    Yields a dict with 'acquired' (monotonic time) and 'waited_s'. Call `check_hold(state)` at safe
    points to stop before the limit. Never call helpers that take the lock themselves inside.
    """
    import paths
    if getattr(gpu_lock, "_held", False):
        raise RuntimeError("gpu_lock is not reentrant: one lock owner per process (A9)")
    requested = time.monotonic()
    with paths.gpu_lock(label):
        gpu_lock._held = True
        state = {"label": label, "acquired": time.monotonic(), "waited_s": time.monotonic() - requested,
                 "pid": os.getpid()}
        try:
            yield state
        finally:
            gpu_lock._held = False


def check_hold(state: dict, margin_s: float = 30.0) -> float:
    held = time.monotonic() - state["acquired"]
    if held > LOCK_HOLD_LIMIT_S - margin_s:
        raise RuntimeError(f"GPU lock held {held:.0f} s; stopping before the {LOCK_HOLD_LIMIT_S:.0f} s limit")
    return held
