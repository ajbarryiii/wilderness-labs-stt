"""Roots, plan constants and artifact locations of stage 2c (synthetic plan).

Roots come from the environment (./python fills it from private/local.env). Everything stage 2c writes is under
$RELABEL_ROOT:

    synthetic/terms/split.json              the frozen term split (targeted / held out, by group; version 2)
    synthetic/terms/split.v<N>.json         split versions of record: split.v2.json (a copy of split.json, stages 2c
                                             and 3) and split.v3.json (synthetic plan v5: confusable pairs; pairs.py)
    synthetic/terms/raw/<alias>/             knowledge-base query results used to write the term lists
    synthetic/engines/<alias>/               TTS model files (downloads, verified)
    synthetic/sources/<alias>/               noise and RIR downloads (download/) and extracted files
    synthetic/noise/pool.json                the TRAIN noise/RIR pool (augmentation), with audio/ (16 kHz clips)
    synthetic/v5/                            confusable pairs (plan v5): label word counts, the detector's candidates
    synthetic/runs/<run>/                    one generation run ("pilot", "pilot2", "full"; "pairs" and its contrast
                                             top-up "pairs-c" are generated in rounds <run>-r1..r3, in
                                             runs/<run>/r1..r3, and joined in runs/<run>/):
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
PAIRS_RUN = "pairs"                  # synthetic plan v5: confusable pairs, under split v3
CONTRAST_RUN = "pairs-c"             # plan v5.2: the contrast top-up of run pairs
PAIRS_RUNS = (PAIRS_RUN, CONTRAST_RUN)
PAIRS_ROUNDS = ("pairs-r1", "pairs-r2", "pairs-r3")
CONTRAST_ROUNDS = ("pairs-c-r1", "pairs-c-r2", "pairs-c-r3")
ROUND_RUNS = PAIRS_ROUNDS + CONTRAST_ROUNDS
RUNS = ("pilot", "pilot2", "full") + PAIRS_RUNS + ROUND_RUNS   # pilot2: the pilot's sentences, pronunciation layer

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


def split_version_path(version: int) -> Path:
    return terms_dir() / f"split.v{int(version)}.json"


def run_split_path(run: str) -> Path:
    """The split a run's hygiene runs under: split v3 for runs pairs and pairs-c and their rounds, else split.json."""
    return split_version_path(3) if run in PAIRS_RUNS or run in ROUND_RUNS else split_path()


def v5_dir() -> Path:
    return base() / "v5"


def engine_dir(alias: str) -> Path:
    return base() / "engines" / alias


def source_dir(alias: str) -> Path:
    return base() / "sources" / alias


def noise_dir() -> Path:
    return base() / "noise"


def run_dir(run: str) -> Path:
    if run not in RUNS:
        raise ValueError(f"unknown production run {run!r} ({', '.join(RUNS)})")
    if run in ROUND_RUNS:
        joined, rk = run.rsplit("-", 1)
        return base() / "runs" / joined / rk
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
