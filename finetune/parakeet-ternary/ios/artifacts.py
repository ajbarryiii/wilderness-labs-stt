"""Where generated artifacts may be written (AGENTS.md: weights and data on /mnt/hd, never in the repo).

root(): Linux /mnt/hd/wilderness-labs-stt/parakeet-ios (the mount is required), macOS
/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios. check(path) resolves a path and refuses
it unless it lies under the machine's allowed area (Linux: /mnt/hd/wilderness-labs-stt on the
mounted disk; macOS: the artifacts directory) and outside this repository. Every write entry point
calls check(). Standard library only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LINUX_AREA = Path("/mnt/hd/wilderness-labs-stt")
LINUX_ROOT = LINUX_AREA / "parakeet-ios"
MAC_ROOT = Path("/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios")


def _area() -> Path:
    if sys.platform == "darwin":
        return MAC_ROOT
    if not os.path.ismount("/mnt/hd"):
        raise RuntimeError("/mnt/hd is not mounted; refusing artifact writes.")
    return LINUX_AREA


def check(path: str | Path) -> Path:
    """path resolved, if it is an allowed artifact location; else RuntimeError."""
    area = _area().resolve()
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(area):
        raise RuntimeError(f"refusing to write {resolved}: artifacts must live under {area}")
    if resolved.is_relative_to(REPO.resolve()):
        raise RuntimeError(f"refusing to write {resolved}: inside the repository")
    return resolved


def root() -> Path:
    """The machine's artifact root, created."""
    path = check(MAC_ROOT if sys.platform == "darwin" else LINUX_ROOT)
    path.mkdir(parents=True, exist_ok=True)
    return path
