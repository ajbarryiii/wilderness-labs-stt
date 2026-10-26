"""Locations and constants of the teacher diagnostic. See diagnostic plan.

Roots come from the environment (./python fills it from private/local.env; see local.env.example):
DATA_MOUNT, RELABEL_ROOT, PARAKEET_TERNARY_ROOT, PARAKEET_BINARY_ROOT. Everything this experiment writes
lives under ARTIFACTS = $RELABEL_ROOT:

    diagnostic/<teacher>/<set>.json, summary.json   production teacher records (score_teacher.py)
    diagnostic/report.json                          the Step 2 report (analyze.py)
    smoke/<teacher>-limit<N>[-cpu]/                 --limit smoke runs, never the production path
    deps/<teacher>/                                 extra packages for one teacher (./python RELABEL_DEPS)
    models/                                         assembled local model directories (canary-qwen)
    caches/, tmp/                                   Hugging Face and interpreter caches (relabel_boot.py)

The parakeet-ternary manifests, audio and B0/M1 records and the parakeet-binary MB1 records are read only.
"""
from __future__ import annotations

import os
from pathlib import Path

import paths  # finetune/parakeet-ternary/paths.py


def _root(key: str) -> Path:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"{key} is not set (environment or finetune/relabel/private/local.env; "
                           "see finetune/relabel/local.env.example)")
    return Path(value)


HERE = Path(__file__).resolve().parent
PLAN = "diagnostic plan"  # private design notes, not in the repository

DATA_MOUNT = _root("DATA_MOUNT")
ARTIFACTS = _root("RELABEL_ROOT")
PARAKEET_TERNARY = _root("PARAKEET_TERNARY_ROOT")
PARAKEET_BINARY = _root("PARAKEET_BINARY_ROOT")
DIAGNOSTIC = ARTIFACTS / "diagnostic"
SMOKE = ARTIFACTS / "smoke"
REPORT = DIAGNOSTIC / "report.json"
MODELS = ARTIFACTS / "models"

# The parakeet GPU lock, shared with every parakeet-ternary/-binary GPU job and the x2 benchmark. The
# manifests are read through parakeet-ternary evaluate.py, so both must be that project's.
GPU_LOCK = PARAKEET_TERNARY / "gpu.lock"
if Path(paths.ARTIFACTS) != PARAKEET_TERNARY:
    raise RuntimeError("PARAKEET_TERNARY_ROOT differs from finetune/parakeet-ternary/paths.py ARTIFACTS")

SETS = list(paths.DEV_SETS)
H4 = ["librispeech_dev_clean", "librispeech_dev_other", "ami_dev", "voxpopuli_dev"]  # decision mean
YODAS = "yodas_dev"            # pseudo-labelled reference: reported on its own, never in a decision
AMI = "ami_dev"                # the only conversational set: reported next to H4, no gate
assert set(H4) | {YODAS} == set(SETS) and YODAS not in H4

# Existing per-utterance records (diagnostic plan "Inputs").
STUDENT_RECORDS = {
    "B0": PARAKEET_TERNARY / "eval" / "B0-dev",
    "M1": PARAKEET_TERNARY / "runs" / "main-M1-P2-lr5e-4" / "eval-dev",
    "MB1": PARAKEET_BINARY / "eval" / "MB1-dev-check",
}

REAL_TEACHERS = ["qwen3-asr-1.7b", "granite-speech-5.0-2b", "canary-qwen-2.5b"]  # vote order
VOTE = "vote"
TEACHERS = REAL_TEACHERS + [VOTE]


def require_mount() -> None:
    if not os.path.ismount(DATA_MOUNT):
        raise RuntimeError(f"{DATA_MOUNT} is not mounted; refusing artifact writes.")
    if not ARTIFACTS.is_relative_to(DATA_MOUNT):
        raise RuntimeError("RELABEL_ROOT is not on DATA_MOUNT")


def output_dir(teacher: str, limit: int | None) -> Path:
    """Where score_teacher writes: DIAGNOSTIC/<teacher> for a full run, SMOKE/<teacher>-limit<N> otherwise."""
    if teacher not in REAL_TEACHERS:
        raise ValueError(f"unknown teacher {teacher!r}")
    if limit is None:
        return DIAGNOSTIC / teacher
    if limit <= 0:
        raise ValueError("--limit must be positive")
    out = SMOKE / f"{teacher}-limit{limit}"
    assert not out.is_relative_to(DIAGNOSTIC)
    return out
