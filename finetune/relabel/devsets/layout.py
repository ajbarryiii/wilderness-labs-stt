"""Locations of the domain dev sets under $RELABEL_ROOT (roots: environment or finetune/relabel/private/local.env).

    devsets/raw/<alias>/              verified downloads that are not in the Hugging Face cache, and
                                      fetch.json (file list, SHA-256, revision) for every alias
    devsets/work/<part>/<rec>.json    per-recording segmentation cache (resume)
    devsets/audio/<part>/<rec>/       segment FLAC files (16 kHz mono)
    devsets/manifests/<part>.jsonl    manifest, and <part>.meta.json
    devsets/scores/<model>/           score_domain.py output

A smoke build (--smoke N) writes everything except raw/ under $RELABEL_ROOT/smoke/devsets-build-<N>/ and
reads raw/ only; production paths are never written by a smoke run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def root(key: str) -> Path:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"{key} is not set (environment or finetune/relabel/private/local.env; "
                           "see finetune/relabel/local.env.example)")
    return Path(value)


def require_mount() -> None:
    mount, relabel = root("DATA_MOUNT"), root("RELABEL_ROOT")
    if not os.path.ismount(mount):
        raise RuntimeError(f"{mount} is not mounted; refusing artifact writes.")
    if not relabel.is_relative_to(mount):
        raise RuntimeError("RELABEL_ROOT is not on DATA_MOUNT")


def production_root() -> Path:
    return root("RELABEL_ROOT") / "devsets"


def smoke_root(limit: int) -> Path:
    if limit <= 0:
        raise ValueError("--smoke must be positive")
    return root("RELABEL_ROOT") / "smoke" / f"devsets-build-{int(limit)}"


@dataclass(frozen=True)
class Layout:
    base: Path

    @classmethod
    def production(cls) -> "Layout":
        return cls(production_root())

    @classmethod
    def smoke(cls, limit: int) -> "Layout":
        return cls(smoke_root(limit))

    @property
    def is_production(self) -> bool:
        return self.base == production_root()

    def raw(self, alias: str) -> Path:
        return production_root() / "raw" / alias  # shared, read-only for smoke builds

    def fetch_record(self, alias: str) -> Path:
        return self.raw(alias) / "fetch.json"

    def work(self, part: str) -> Path:
        return self.base / "work" / part

    def audio(self, part: str) -> Path:
        return self.base / "audio" / part

    def manifests(self) -> Path:
        return self.base / "manifests"

    def manifest(self, part: str) -> Path:
        return self.manifests() / f"{part}.jsonl"

    def meta(self, part: str) -> Path:
        return self.manifests() / f"{part}.meta.json"


def scores_root() -> Path:
    return production_root() / "scores"


def smoke_scores_root() -> Path:
    return root("RELABEL_ROOT") / "smoke" / "devsets-scores"
