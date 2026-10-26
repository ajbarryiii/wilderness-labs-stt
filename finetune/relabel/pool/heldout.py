"""Held-out utterances for the denylist: dev, domain-dev and test references, read as TEXT ONLY.

Every manifest row's `text` is normalized with the evaluation normalizer; an utterance with at least 8 words
contributes its distinct word 8-grams. Only the `id`, `text` and (domain dev) `part` fields are read.

Identity rule: a held-out utterance that publishes the id of a training route's recording. Two ways:
- by set name (IDENTITY): yodas_dev ids embed the YODAS original_audio_id
  ('<shard>_<file>_<recording>_<4 numeric fields>'); the ESB test sets carry no id in a training namespace;
- by row: an optional `identity` field {"<training source>": "<recording id>" | [ids]} written by the
  builder of a held-out set whose source publishes such ids (e.g. a YouTube-derived set naming the video ids
  that YODAS recordings use as original_audio_id).

Set names: a path given alone is named by its file name without `.jsonl` (the parakeet-ternary manifests,
e.g. `yodas_dev`, `test_gigaspeech`); discovered manifests come with their own names (pool/domain.py).
"""
from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pool import common


def yodas_dev_recording(row_id: str) -> str:
    native = row_id.split(":", 1)[1]
    parts = native.split("_")
    if len(parts) < 7 or not all(p.isdigit() for p in parts[-4:]):
        raise ValueError(f"unexpected yodas_dev id {row_id!r}")
    return "_".join(parts[2:-4])


IDENTITY = {"yodas_dev": ("yodas", yodas_dev_recording)}   # held-out set -> (training source, id -> recording)


@dataclasses.dataclass
class HeldOutSet:
    name: str
    path: str
    sha256: str
    ids: list[str]
    words: list[int]
    grams: list[np.ndarray]          # distinct 8-gram hashes per utterance (empty if < 8 words)
    identity: set[tuple[str, str]]
    parts: set[str]


def _row_identity(row: dict) -> set[tuple[str, str]]:
    out = set()
    for src, ids in (row.get("identity") or {}).items():
        for rec in ([ids] if isinstance(ids, str) else ids):
            if str(rec).strip():
                out.add((str(src), str(rec)))
    return out


def load_set(path: Path, hasher: common.WordHasher | None = None, name: str | None = None) -> HeldOutSet:
    path = Path(path)
    hasher = hasher or common.WordHasher()
    if name is None:
        name = path.name[:-len(".jsonl")] if path.name.endswith(".jsonl") else path.stem
    ident = IDENTITY.get(name)
    ids, words, grams, identity, parts = [], [], [], set(), set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ws = common.normalize(row.get("text")).split()
            ids.append(str(row["id"]))
            words.append(len(ws))
            grams.append(np.unique(common.ngram_hashes(hasher(ws))) if len(ws) >= common.NGRAM
                         else np.empty(0, dtype=np.uint64))
            if ident:
                identity.add((ident[0], ident[1](str(row["id"]))))
            identity |= _row_identity(row)
            if row.get("part"):
                parts.add(str(row["part"]))
    return HeldOutSet(name, str(path), common.sha256_file(path), ids, words, grams, identity, parts)


def load_sets(paths: Sequence[Path | tuple[str, Path]]) -> list[HeldOutSet]:
    """Held-out sets from paths or (name, path) pairs; names must be unique."""
    hasher = common.WordHasher()
    sets = [load_set(p[1], hasher, p[0]) if isinstance(p, tuple) else load_set(p, hasher) for p in paths]
    names = [s.name for s in sets]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate held-out set names: {sorted(names)}")
    return sets


class Index:
    """All held-out 8-grams: sorted unique hashes, and for each hash the utterances containing it (CSR)."""

    def __init__(self, sets: Sequence[HeldOutSet]):
        self.sets = list(sets)
        self.utt_set, self.utt_id, self.utt_words, grams = [], [], [], []
        for k, s in enumerate(self.sets):
            for uid, n, g in zip(s.ids, s.words, s.grams):
                self.utt_set.append(k)
                self.utt_id.append(uid)
                self.utt_words.append(n)
                grams.append(g)
        self.utt_set = np.array(self.utt_set, dtype=np.int32)
        self.utt_words = np.array(self.utt_words, dtype=np.int64)
        lens = np.array([len(g) for g in grams], dtype=np.int64)
        flat = np.concatenate(grams) if grams else np.empty(0, dtype=np.uint64)
        owner = np.repeat(np.arange(len(grams), dtype=np.int64), lens)
        self.hashes, inverse = np.unique(flat, return_inverse=True)
        self.utt_grams = np.split(inverse.astype(np.int64), np.cumsum(lens)[:-1]) if len(grams) else []
        order = np.argsort(inverse, kind="stable")
        self.csr_utts = owner[order]
        counts = np.bincount(inverse, minlength=len(self.hashes))
        self.csr_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    def __len__(self) -> int:
        return len(self.utt_id)
