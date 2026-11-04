"""Roots, plan constants and artifact locations of the read-speech extra source (b0prime plan "Extra sources").

Inputs are named by registry alias only. The private registry ($PRIVATE_REGISTRY/sources.json) gives, under each
alias's "local" object, paths relative to $RELABEL_ROOT:
    cv27-en        local.raw_dir        the release's language directory (the six split tsvs, clip_durations.tsv,
                                        clips/*.mp3); untrusted data, read only, never executed
                   local.esb_test_set   the parakeet-ternary test set of the same corpus (its parquet files under
                                        $PARAKEET_TERNARY_ROOT/data/esb/<name>/; columns id, text, audio.path)
    ted-test-text  local.text_parquet   the held-out talk text (the `text` column only)

Everything this source writes is under $RELABEL_ROOT/corpora/cv27-en/ (out_dir()):
    heldout/boilerplate.json, .u64        the global denylist's boilerplate 8-grams, reproduced (heldout.py)
    selection/selection.parquet, .json    every validated clip with its drop reason (None = selected), counts
    dev/sample.parquet                     the dev sample (dev.py); dev/audio/, dev/cv27-en-dev.jsonl + .meta.json
    store/decode.json                      the decode plan: selection hash, chunk size, chunk ids
    store/audio/<chunk>/<stem>.flac        16 kHz mono PCM_16 FLAC per clip
    store/chunks/<chunk>.jsonl             decode records of one chunk
    confirm/run.json, confirm/chunks/<chunk>.jsonl   the B0 check (settings; B0 transcript and WER per clip)
    store/clips.jsonl, clips.meta.json     the clip source (finalize): the synthetic store's format plus samples
                                           and PCM SHA-256
    confirm/b0.parquet                     B0's transcript of every checked clip
    reports/report.json                    every count
Smoke and probe runs write under $RELABEL_ROOT/smoke/cv27-en/<name>/ only.
"""
from __future__ import annotations

import os
from pathlib import Path

ALIAS = "cv27-en"
TALK_TEXT_ALIAS = "ted-test-text"
SAMPLE_RATE = 16000

# --- plan constants (b0prime plan "Extra sources") ---------------------------------------------------------------
TSVS = ("validated.tsv", "other.tsv", "invalidated.tsv", "train.tsv", "dev.tsv", "test.tsv")
HELD_SPLITS = ("test.tsv", "dev.tsv")      # the release's own test split and its dev split (reserved)
TRAIN_TSV = "validated.tsv"
DURATIONS_TSV = "clip_durations.tsv"
NGRAM = 8
SPEAKER_CAP = 1000                 # clips per speaker
SENTENCE_CAP = 3                   # recordings per normalized sentence
CAP_SEED = 20261029                # the caps' seeded order (numpy default_rng permutation of the clips sorted by id)
MIN_MS, MAX_MS = 1000, 30000       # 1-30 s inclusive, on clip_durations.tsv; then on the decoded samples
MIN_SAMPLES, MAX_SAMPLES = MIN_MS * SAMPLE_RATE // 1000, MAX_MS * SAMPLE_RATE // 1000
FILTERS_VERSION = 3                # labeling/filters.py: punctuation mapping and tokenizer coverage
B0_MAX_WER = 0.5                   # a clip is dropped if WER(sentence, B0) > 0.5 (evaluation normalizer)
DEV_SIZE, DEV_PER_SPEAKER, DEV_SEED = 2000, 2, 27
CHUNK_CLIPS = 2048                 # clips per decode / confirm chunk (in id order)
PROBE_CLIPS, PROBE_SEED = 2000, 20261030
DEV_SET = f"{ALIAS}-dev"


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


def out_dir() -> Path:
    return relabel_root() / "corpora" / ALIAS


def smoke_dir(name: str) -> Path:
    return relabel_root() / "smoke" / ALIAS / name


def is_production(path: Path) -> bool:
    return Path(path).resolve().is_relative_to(out_dir().resolve())


def registry_local(alias: str) -> dict:
    from corpora import registry
    entry = registry.source(alias)
    local = entry.get("local")
    if not isinstance(local, dict):
        raise RuntimeError(f"registry alias {alias!r} has no `local` object (input paths)")
    return local


def raw_dir() -> Path:
    return relabel_root() / registry_local(ALIAS)["raw_dir"]


def esb_test_set() -> str:
    return str(registry_local(ALIAS)["esb_test_set"])


def talk_text() -> Path:
    return relabel_root() / registry_local(TALK_TEXT_ALIAS)["text_parquet"]


def tokenizer_cache() -> Path:
    return relabel_root() / "caches" / "b0-tokenizer"


class Paths:
    """Artifact paths under one base directory (out_dir() in production, smoke_dir(name) for smokes)."""

    def __init__(self, base: Path):
        self.base = Path(base)

    @property
    def boilerplate(self) -> Path:
        return self.base / "heldout" / "boilerplate.json"

    @property
    def selection(self) -> Path:
        return self.base / "selection" / "selection.parquet"

    @property
    def selection_meta(self) -> Path:
        return self.base / "selection" / "selection.json"

    @property
    def dev_sample(self) -> Path:
        return self.base / "dev" / "sample.parquet"

    @property
    def dev_audio(self) -> Path:
        return self.base / "dev" / "audio"

    @property
    def dev_manifest(self) -> Path:
        return self.base / "dev" / f"{DEV_SET}.jsonl"

    @property
    def decode_plan(self) -> Path:
        return self.base / "store" / "decode.json"

    def audio(self, chunk: int, stem: str) -> Path:
        return self.base / "store" / "audio" / f"{chunk:05d}" / f"{stem}.flac"

    def decode_chunk(self, chunk: int) -> Path:
        return self.base / "store" / "chunks" / f"{chunk:05d}.jsonl"

    @property
    def confirm_run(self) -> Path:
        return self.base / "confirm" / "run.json"

    def confirm_chunk(self, chunk: int) -> Path:
        return self.base / "confirm" / "chunks" / f"{chunk:05d}.jsonl"

    @property
    def clips(self) -> Path:
        return self.base / "store" / "clips.jsonl"

    @property
    def clips_meta(self) -> Path:
        return self.base / "store" / "clips.meta.json"

    @property
    def b0_transcripts(self) -> Path:
        return self.base / "confirm" / "b0.parquet"

    @property
    def report(self) -> Path:
        return self.base / "reports" / "report.json"
