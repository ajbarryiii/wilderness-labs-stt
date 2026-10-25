"""Held-out manifests built in this program (Part A's domain dev sets, the later `eval-ext`): discovered, not listed.

Every `*.jsonl` file anywhere under $RELABEL_ROOT/devsets/ whose rows are per-utterance manifest rows (keys
`id`, `text`, `audio_filepath`) is a held-out manifest for the denylist; other JSONL files there (records,
logs) are skipped and reported. Set names are the paths relative to that directory without `.jsonl`
(e.g. `dev-medical`, `eval-ext/<part>`), so names never collide across subdirectories.
"""
from __future__ import annotations

import json
from pathlib import Path

MANIFEST_KEYS = {"id", "text", "audio_filepath"}


def is_manifest(path: Path) -> bool:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    return False
                return isinstance(row, dict) and MANIFEST_KEYS <= set(row)
    return False


def manifests(devsets_dir: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """([(set name, path)], skipped paths) under devsets_dir, sorted by name."""
    if not devsets_dir.is_dir():
        return [], []
    found, skipped = [], []
    for p in sorted(devsets_dir.rglob("*.jsonl")):
        if not p.is_file():
            continue
        if is_manifest(p):
            found.append((p.relative_to(devsets_dir).as_posix()[:-len(".jsonl")], p))
        else:
            skipped.append(str(p))
    return found, skipped
