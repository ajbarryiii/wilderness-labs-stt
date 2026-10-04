"""Fixed locations and constants for the ternary Parakeet experiment. See DESIGN.md."""
from __future__ import annotations

import contextlib
import fcntl
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

ARTIFACTS = Path("/mnt/hd/wilderness-labs-stt/parakeet-ternary")
MODEL_DIR = ARTIFACTS / "models" / "parakeet-tdt-0.6b-v2"
MODEL_FILE = MODEL_DIR / "parakeet-tdt-0.6b-v2.nemo"
MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
MODEL_REVISION = "ae9ad07059c7c739ffaf932226a8fe64ae2620b0"

DATA = ARTIFACTS / "data"            # raw downloads (parquet, tarballs)
AUDIO = ARTIFACTS / "audio"          # extracted 16 kHz mono FLAC, one file per utterance
MANIFESTS = ARTIFACTS / "manifests"  # NeMo-style JSONL manifests
LABELS = ARTIFACTS / "labels"        # teacher-labeled manifests
RUNS = ARTIFACTS / "runs"
EVAL = ARTIFACTS / "eval"
POWER = ARTIFACTS / "power"
GPU_LOCK = ARTIFACTS / "gpu.lock"

ESB_REPO = "hf-audio/esb-datasets-test-only-sorted"
ESB_REVISION = "b6bdcd0beb"
# Test sets scored for the report (original, not _cleaned, configs). TED-LIUM is not in the bundle.
TEST_SETS = {
    "librispeech_clean": ("librispeech", "test.clean"),
    "librispeech_other": ("librispeech", "test.other"),
    "ami": ("ami", "test"),
    "earnings22": ("earnings22", "test"),
    "gigaspeech": ("gigaspeech", "test"),
    "spgispeech": ("spgispeech", "test"),
    "voxpopuli": ("voxpopuli", "test"),
    "common_voice": ("common_voice", "test"),
}
# NVIDIA's published Open ASR Leaderboard WERs for this revision (model card .eval_results).
PUBLISHED_WER = {"librispeech_clean": 1.69, "librispeech_other": 3.19, "ami": 11.16,
                 "earnings22": 11.15, "gigaspeech": 9.74, "spgispeech": 2.17, "voxpopuli": 5.95,
                 "tedlium": 3.38}
MEAN_SETS = ["librispeech_clean", "librispeech_other", "ami", "earnings22", "gigaspeech",
             "spgispeech", "voxpopuli"]  # DESIGN.md primary mean (TED-LIUM unavailable)
DEV_SETS = ["librispeech_dev_clean", "librispeech_dev_other", "ami_dev", "voxpopuli_dev", "yodas_dev"]

SAMPLE_RATE = 16000
SEED = 20260930
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def require_mount() -> None:
    if not os.path.ismount("/mnt/hd"):
        raise RuntimeError("/mnt/hd is not mounted; refusing artifact writes.")


def run_dir(name: str) -> Path:
    """RUNS/name, created; the name may not contain path separators."""
    if not _NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"invalid run name {name!r}")
    require_mount()
    path = RUNS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def gpu_lock(label: str):
    """Exclusive lock held for the duration of any GPU job longer than a quick smoke test.

    Blocks until the GPU is free, so concurrent jobs queue instead of competing
    for memory and power.
    """
    require_mount()
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(GPU_LOCK, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0); handle.truncate(); handle.write(f"{os.getpid()} {label}\n"); handle.flush()
        try:
            yield
        finally:
            handle.seek(0); handle.truncate(); handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)
