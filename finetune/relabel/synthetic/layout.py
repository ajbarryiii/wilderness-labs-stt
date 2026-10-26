"""Roots, plan constants and artifact locations of stage 2c (synthetic plan).

Roots come from the environment (./python fills it from private/local.env). Everything stage 2c writes is under
$RELABEL_ROOT:

    synthetic/terms/split.json              the frozen term split (targeted / held out, by group)
    synthetic/terms/raw/<alias>/             knowledge-base query results used to write the term lists
    synthetic/engines/<alias>/               TTS model files (downloads, verified)
    synthetic/sources/<alias>/               noise and RIR downloads (download/) and extracted files
    synthetic/noise/pool.json                the TRAIN noise/RIR pool (augmentation), with audio/ (16 kHz clips)
    synthetic/runs/<run>/                    one generation run ("pilot", "pilot2", "full"):
        prompts.jsonl                        the seeded prompts
        text/<chunk>.jsonl                   raw generator outputs, one file per chunk job
        sentences.jsonl, hygiene.json        sentences after hygiene, and the hygiene report
        tts_plan.jsonl, pronunciation.json   engine / voice / tts_text per clip, and the lexicon used
        tts/<chunk>.jsonl, audio/<engine>/   rendered clips (16 kHz FLAC) and their records
        roundtrip/<chunk>.jsonl              T*'s transcripts and the filter decision per clip
        clips.jsonl                          the kept clips: the clip source of stages 2b/2c
        report.json, listening/              filter report and listening sample
    smoke/synthetic/<name>/                  smoke runs; never a production path

The private registry ($PRIVATE_REGISTRY/sources.json) maps aliases to sources (read by registry()).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# --- plan constants (synthetic plan) -----------------------------------------------------------------------------
DOMAINS = ("chemistry", "mathematics", "finance", "software_engineering")
TARGETED_SHARE = 0.70            # by group, fixed hash
SPLIT_SALT = "relabel-2c-term-split-v1"
REGISTERS = ("conversational", "explanatory", "dictated notes")
MIN_WORDS, MAX_WORDS = 3, 60     # sentence length
MIN_SECONDS, MAX_SECONDS = 1.0, 30.0
ROUNDTRIP_MAX_WER = 0.25         # normalized WER(text, T*) <= 0.25 keeps a clip
SYNTHETIC_SHARE = 0.10           # synthetic audio per batch, relative to the batch's real audio
SAMPLE_RATE = 16000
LISTEN_KEPT, LISTEN_REJECTED = 50, 20   # per domain
FULL_SENTENCES = 135_000
PILOT_HOURS = 5.0
SEED = 20261025
RUNS = ("pilot", "pilot2", "full")   # pilot2: the pilot's sentences re-rendered with the pronunciation layer

TEXTGEN = "syn-textgen"
TTS_ENGINES = ("syn-tts-a", "syn-tts-b")
NOISE_SOURCES = ("syn-noise-a", "syn-noise-b")
RIR_SOURCES = ("syn-rir-a",)
TERM_SOURCES = ("syn-terms-kb",)


def root(key: str) -> Path:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"{key} is not set (environment or finetune/relabel/private/local.env; "
                           "see finetune/relabel/local.env.example)")
    return Path(value)


def require_mount() -> None:
    mount = root("DATA_MOUNT")
    if not os.path.ismount(mount):
        raise RuntimeError(f"{mount} is not mounted; refusing artifact writes.")
    if not root("RELABEL_ROOT").is_relative_to(mount):
        raise RuntimeError("RELABEL_ROOT is not on DATA_MOUNT")


def relabel_root() -> Path:
    return root("RELABEL_ROOT")


def base() -> Path:
    return relabel_root() / "synthetic"


def terms_dir() -> Path:
    return base() / "terms"


def split_path() -> Path:
    return terms_dir() / "split.json"


def engine_dir(alias: str) -> Path:
    return base() / "engines" / alias


def source_dir(alias: str) -> Path:
    return base() / "sources" / alias


def noise_dir() -> Path:
    return base() / "noise"


def run_dir(run: str) -> Path:
    if run not in RUNS:
        raise ValueError(f"unknown production run {run!r} ({', '.join(RUNS)})")
    return base() / "runs" / run


def smoke_dir(name: str) -> Path:
    return relabel_root() / "smoke" / "synthetic" / name


def is_production(path: Path) -> bool:
    path = Path(path).resolve()
    return path.is_relative_to(base().resolve())


def deps_dir(name: str) -> Path:
    return relabel_root() / "deps" / name


def registry() -> dict:
    path = root("PRIVATE_REGISTRY") / "sources.json"
    if not path.exists():
        raise RuntimeError(f"{path} is missing (the private source registry)")
    return json.loads(path.read_text())


def source(alias: str) -> dict:
    entries = registry()
    if alias not in entries:
        raise RuntimeError(f"alias {alias!r} is not in the private source registry")
    return entries[alias]
