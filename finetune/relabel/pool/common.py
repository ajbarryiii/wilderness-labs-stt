"""Roots, constants, source specs and the text helpers shared by the pool modules (labeling plan Part B).

Text conventions (preregistered in the plan):
- Every transcript is normalized with the evaluation normalizer (finetune/parakeet-ternary/evaluate.py
  `normalize`, the Open ASR Leaderboard normalizer) and split on whitespace into words.
- A recording's transcript is its rows' normalized words concatenated in recording order (YODAS: offset in
  the source audio; peoples_speech: segment index), so n-grams that cross row boundaries count.
- Word 8-grams are hashed to 64 bits (blake2b per word, then an order-sensitive splitmix64 chain over the
  8 words); with ~1e9 training and ~1e7 held-out 8-grams the expected number of false matches is < 1e-3.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------
# Plan constants (labeling plan Part B)
# ------------------------------------------------------------------------------------------------
MED_THRESHOLD = 10.0         # medical terms per 1,000 words: yodas-med / ps-med (inclusive)
TECH_THRESHOLD = 10.0        # technical terms per 1,000 words: yodas-tech (inclusive), and not medical
MIN_WORDS = 200              # recordings with fewer normalized words are excluded
NGRAM = 8                    # containment n-gram order (words)
BOILERPLATE_MAX = 100        # an 8-gram in MORE than this many training recordings is boilerplate
CONTAIN_FRACTION = 0.5       # contained if >= this share of the held-out side's non-boilerplate 8-grams occur
SAMPLE_RECORDINGS = 50       # user sample per subset
SAMPLE_WORDS = 300           # first words shown per sampled recording
SAMPLE_SEED = 20261021
MIN_SECONDS, MAX_SECONDS = 1.0, 30.0   # the stream's utterance filter, for the kept-hours estimate

SUBSETS = {  # subset -> (source, kind)
    "yodas-med": ("yodas", "medical"),
    "ps-med": ("peoples_speech", "medical"),
    "yodas-tech": ("yodas", "technical"),
}


def now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"{now()} [pool pid {os.getpid()}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------
# Roots
# ------------------------------------------------------------------------------------------------

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


def pool_dir() -> Path:
    return root("RELABEL_ROOT") / "pool"


def probe_dir() -> Path:
    """Timing probes write here, never into pool_dir()."""
    return root("RELABEL_ROOT") / "smoke" / "pool-probe"


def corpora_dir() -> Path:
    return root("RELABEL_ROOT") / "corpora"


def devsets_dir() -> Path:
    return root("RELABEL_ROOT") / "devsets"


def ternary_manifests() -> Path:
    return root("PARAKEET_TERNARY_ROOT") / "manifests"


# ------------------------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------------------------

def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: object) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode())


def atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    atomic_write_bytes(path, "".join(line + "\n" for line in lines).encode())


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ------------------------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------------------------

_PS_ID = re.compile(r"(.+)_(\d+)\.flac")


def ps_recording(row_id: str) -> tuple[str, float]:
    """peoples_speech row id '<item>_SLASH_<file>_DOT_<ext>_<NNNNN>.flac' -> (source audio file, segment index)."""
    m = _PS_ID.fullmatch(row_id or "")
    if not m:
        raise ValueError(f"peoples_speech id without a segment suffix: {row_id!r}")
    return m.group(1), float(int(m.group(2)))


@dataclasses.dataclass(frozen=True)
class TextSource:
    """How to read one streamed source's transcripts (column projection) and group them into recordings."""
    name: str
    repo: str
    revision: str
    columns: tuple[str, ...]                   # the only parquet columns fetched
    text_col: str
    duration_col: str
    duration_scale: float                      # to seconds
    recording: Callable[[dict], tuple[str, float]]   # row -> (recording id, order within the recording)


def _yodas_recording(row: dict) -> tuple[str, float]:
    rec = row["original_audio_id"]
    if not rec:
        raise ValueError("yodas row without original_audio_id")
    return str(rec), float(row["original_audio_offset"] or 0.0)


def _ps_row_recording(row: dict) -> tuple[str, float]:
    return ps_recording(row["id"])


def text_sources() -> dict[str, TextSource]:
    """The two streamed sources at the revisions pinned by finetune/parakeet-ternary/data.py."""
    import data  # finetune/parakeet-ternary
    y, p = data.HF_SOURCES["yodas"], data.HF_SOURCES["peoples_speech"]
    return {
        "yodas": TextSource("yodas", y["repo"], y["revision"],
                            ("duration", "text", "original_audio_id", "original_audio_offset"),
                            "text", "duration", 1.0, _yodas_recording),
        "peoples_speech": TextSource("peoples_speech", p["repo"], p["revision"], ("id", "duration_ms", "text"),
                                     "text", "duration_ms", 1e-3, _ps_row_recording),
    }


def training_files(source: str) -> list[dict]:
    """[{path, size, sha256}] of the source's training parquet files, exactly the stream's file list
    (yodas: every English asr_only file except the 16 reserved development shards)."""
    import data
    import stream
    if source == "yodas":
        files, reserved = data.yodas_reserved()
        keep = set(stream.yodas_training_files([f["path"] for f in files], reserved))
        out = [f for f in files if f["path"] in keep]
    elif source == "peoples_speech":
        out = list(data.list_hf_files("peoples_speech", "train"))
    else:
        raise ValueError(f"unknown source {source!r}")
    return sorted(out, key=lambda f: f["path"])


def stream_files_sha256(source: str, files: Sequence[dict]) -> str:
    """The stream's files_sha256 (finetune/parakeet-ternary/stream.py file_list_sha256 over hf:// URLs)."""
    import stream
    spec = text_sources()[source]
    return stream.file_list_sha256([stream._hf_url(spec.repo, spec.revision, f["path"]) for f in files])


def file_key(path: str) -> str:
    """Cache key of one parquet path (yodas data/en000/asr_only/00000002.parquet -> en000-00000002)."""
    parts = Path(path).with_suffix("").parts
    if parts[0] == "data":
        return f"{parts[1]}-{parts[-1]}"
    return parts[-1]


# ------------------------------------------------------------------------------------------------
# Text
# ------------------------------------------------------------------------------------------------

def normalize(text: str | None) -> str:
    """The evaluation normalizer (Open ASR Leaderboard EnglishTextNormalizer, pinned spelling map)."""
    import evaluate
    return evaluate.normalize(text or "").strip()


_SEED = np.uint64(0x6A09E667F3BCC908)
_M1, _M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _mix(z: np.ndarray) -> np.ndarray:
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


class WordHasher:
    """Deterministic 64-bit word hashes (process-independent), cached."""

    def __init__(self, max_cache: int = 5_000_000) -> None:
        self._cache: dict[str, int] = {}
        self._max = max_cache

    def __call__(self, words: Sequence[str]) -> np.ndarray:
        cache, out = self._cache, np.empty(len(words), dtype=np.uint64)
        for i, w in enumerate(words):
            h = cache.get(w)
            if h is None:
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
                if len(cache) < self._max:
                    cache[w] = h
            out[i] = h
        return out


def ngram_hashes(word_hashes: np.ndarray, n: int = NGRAM) -> np.ndarray:
    """Hash of every contiguous n-word window (len(words) - n + 1 values; empty if fewer than n words)."""
    count = len(word_hashes) - n + 1
    if count <= 0:
        return np.empty(0, dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = np.full(count, _SEED, dtype=np.uint64)
        for k in range(n):
            h = _mix(h ^ word_hashes[k:k + count])
    return h
