"""Fixed locations for the ternary Whisper experiment. See DESIGN.md."""
from __future__ import annotations

import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Revision-pinned, hash-locked pretrained checkpoint (see finetune/stt/README.md).
MODEL_DIR = REPO / "finetune" / "stt" / "models" / "whisper-tiny.en"
MODEL_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"

ARTIFACTS = Path("/mnt/hd/wilderness-labs-stt/whisper-ternary")
DATA = ARTIFACTS / "data"
RUNS = ARTIFACTS / "runs"
MANIFESTS = DATA / "manifests"

TRAIN_CLEAN_100 = Path(
    "/mnt/hd/wilderness-labs-stt/stt-distillation/datasets/libri/LibriSpeech/train-clean-100")

SPLITS: dict[str, Path] = {
    "train-clean-100": TRAIN_CLEAN_100,
    "train-clean-360": DATA / "LibriSpeech" / "train-clean-360",
    "dev-clean": DATA / "LibriSpeech" / "dev-clean",
    "test-clean": DATA / "LibriSpeech" / "test-clean",
    "test-other": DATA / "LibriSpeech" / "test-other",
}

# Published LibriSpeech utterance counts; manifests must match exactly.
EXPECTED_UTTERANCES: dict[str, int] = {
    "train-clean-100": 28539,
    "train-clean-360": 104014,
    "dev-clean": 2703,
    "test-clean": 2620,
    "test-other": 2939,
}

SAMPLE_RATE = 16000
SEED = 20260929


def require_mount() -> None:
    """Refuse to write artifacts unless the data disk is mounted."""
    if not os.path.ismount("/mnt/hd"):
        raise RuntimeError("/mnt/hd is not mounted; refusing artifact writes.")


_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def run_dir(name: str) -> Path:
    """Create and return RUNS/name; the name may not contain path separators."""
    if not _RUN_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"run name must match {_RUN_NAME.pattern!r}: {name!r}")
    require_mount()
    path = (RUNS / name).resolve()
    if path.parent != RUNS.resolve():
        raise ValueError(f"run directory escapes {RUNS}: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
