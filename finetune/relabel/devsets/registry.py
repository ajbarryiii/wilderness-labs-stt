"""The private source registry: $PRIVATE_REGISTRY/sources.json, alias -> source entry (read at run time).

Entry keys used here (other keys, e.g. license and acquisition, are recorded but not interpreted):

    origin       "hf" (a Hugging Face dataset repo), "github" (a GitHub repo, LFS files via the media host) or
                 "ia" (a web archive's public items; "files" pins each file: item, name, size, md5, sha1)
    repo         repository id
    revision     the pinned commit
    files        glob patterns of the repo files to fetch (repo-relative); for "ia", the pinned file list
    layout       sources.LAYOUTS
    license, acquisition   recorded in artifact metadata

Layout keys: audio_dir, transcript_dir, transcript_suffix (textgrid-channels); id_column, audio_column,
text_column, id_with_split (Parquet layouts); boundary_words (dictated sentence ends); exclude_test_set;
timing_from (acoustic variants reuse a reference rendering's cuts; build.py).
"""
from __future__ import annotations

import json
from pathlib import Path

from devsets import layout

REQUIRED = ("origin", "repo", "revision", "files", "layout", "license", "acquisition")
LAYOUT_KEYS = {
    "textgrid-channels": ("audio_dir", "transcript_dir"),
    "parquet-clips": ("id_column", "audio_column", "text_column"),
    "parquet-longform": ("id_column", "audio_column", "text_column"),
    "srt-longform": ("media_suffix",),
}


def path() -> Path:
    return layout.root("PRIVATE_REGISTRY") / "sources.json"


def load() -> dict:
    p = path()
    if not p.exists():
        raise RuntimeError(f"{p} is missing (the private source registry)")
    return json.loads(p.read_text())


def source(alias: str, registry: dict | None = None) -> dict:
    entries = load() if registry is None else registry
    if alias not in entries:
        raise KeyError(f"alias {alias!r} is not in the private registry")
    entry = entries[alias]
    missing = [k for k in REQUIRED if k not in entry]
    if entry.get("layout") in LAYOUT_KEYS:
        missing += [k for k in LAYOUT_KEYS[entry["layout"]] if k not in entry]
    else:
        missing.append(f"layout (one of {sorted(LAYOUT_KEYS)})")
    if entry.get("origin") not in ("hf", "github", "ia"):
        missing.append("origin (hf, github or ia)")
    elif entry["origin"] == "ia" and not all(
            isinstance(f, dict) and {"item", "name", "size", "md5", "sha1"} <= set(f) for f in entry.get("files", [])):
        missing.append("files (ia: item, name, size, md5, sha1 per file)")
    if missing:
        raise ValueError(f"registry entry {alias!r} lacks {missing}")
    return entry
