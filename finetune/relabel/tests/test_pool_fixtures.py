"""Shared fixture builders for the test_pool_* and test_corpora_* contract tests (no tests here).

Synthetic words are lowercase consonant-vowel tokens the evaluation normalizer leaves unchanged, so test
texts and their normalized forms coincide.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_C, _V = "bdfgklmnprstvz", "aeiou"


def vocab(n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out: set[str] = set()
    while len(out) < n:
        out.add("".join(rng.choice(_C) + rng.choice(_V) for _ in range(rng.randint(2, 3))) + rng.choice("xq"))
    return sorted(out)


class Words:
    """Seeded filler words that never collide with explicitly constructed phrases (suffix 'x'/'q' + 'w')."""

    def __init__(self, seed: int = 0, size: int = 4000):
        self.rng = random.Random(seed)
        self.pool = [w + "w" for w in vocab(size, seed)]

    def take(self, n: int) -> list[str]:
        return [self.rng.choice(self.pool) for _ in range(n)]


def phrase(n: int, tag: str) -> list[str]:
    """n distinct words unique to `tag` (no filler word ends like these)."""
    return [f"{tag}{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(n)]


def write_cache(cache_dir: Path, source: str, path: str, rows: list[dict]) -> Path:
    """A text-cache shard as pool.fetch writes it: rows {recording, order, duration, row_group, text}."""
    from pool import common, fetch
    cols = {c: [] for c in fetch.CACHE_COLUMNS}
    for i, r in enumerate(rows):
        cols["row"].append(i)
        cols["row_group"].append(r.get("row_group", 0))
        cols["recording"].append(r["recording"])
        cols["order"].append(float(r.get("order", i)))
        cols["duration"].append(float(r.get("duration", 5.0)))
        cols["text"].append(r["text"])
        cols["norm"].append(common.normalize(r["text"]))
    schema = pa.schema([("row", pa.int32()), ("row_group", pa.int32()), ("recording", pa.string()),
                        ("order", pa.float64()), ("duration", pa.float32()), ("text", pa.string()),
                        ("norm", pa.string())])
    out, done = fetch.cache_paths(cache_dir, {"path": path})
    cache_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pydict(cols, schema=schema), buf)
    out.write_bytes(buf.getvalue())
    done.write_text(json.dumps({"source": source, "path": path, "rows": len(rows), "size": 1, "hours": 0,
                                "bytes_fetched": 1}))
    return out


def write_listing(cache_dir: Path, source: str, paths: list[str]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "files.json").write_text(json.dumps(
        {"source": source, "files": len(paths), "listing": [{"path": p, "size": 1, "sha256": None} for p in paths]}))


def write_manifest(path: Path, rows: list[dict]) -> Path:
    """A held-out manifest in the per-utterance schema (text is what matters)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for r in rows:
            handle.write(json.dumps({"audio_filepath": "/nonexistent.flac", "duration": 1.0,
                                     "source": r.get("source", "x"), **r}) + "\n")
    return path


def source_parquet(path: Path, kind: str, recordings: list[tuple[str, list[str]]], rows_per_group: int = 3,
                   audio_bytes: int = 20000, seed: int = 0) -> list[dict]:
    """A source-like parquet file with a large `audio` struct column; returns the rows written.

    kind 'yodas': utt_id, audio, duration, lang, task, text, translation_en, original_audio_id,
    original_audio_offset (the real column order). kind 'peoples_speech': id, audio, duration_ms, text.
    """
    rng = random.Random(seed)
    rows = []
    for rec, texts in recordings:
        for k, text in enumerate(texts):
            dur = round(rng.uniform(0.5, 35.0), 2)
            audio = {"bytes": rng.randbytes(audio_bytes), "path": None}
            if kind == "yodas":
                rows.append({"utt_id": f"en000_00000001_{rec}_{k}", "audio": audio, "duration": dur, "lang": "en",
                             "task": "asr", "text": text, "translation_en": None, "original_audio_id": rec,
                             "original_audio_offset": float(10 * k)})
            else:
                rows.append({"id": f"{rec}_{k:05d}.flac", "audio": audio, "duration_ms": int(dur * 1000),
                             "text": text})
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, row_group_size=rows_per_group)
    return rows
