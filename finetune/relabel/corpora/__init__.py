"""Stage 2 Part B new corpora (labeling plan "New corpora"), by alias: `med-osce` and `syn-dialog`.

Real names, URLs, revisions and licenses are in the private registry ($PRIVATE_REGISTRY/sources.json); code
reads them at runtime. Per corpus, $RELABEL_ROOT/corpora/<alias>/status.json records the outcome:

    {"status": "built", "manifest": ".../manifest.jsonl", "sha256": ..., ...}   segmented, frozen manifest
    {"status": "dropped", "reason": ...}                                         not used (share returns to the mix)

    ./python -m corpora syn-dialog-check     the voice-provenance check (drops the corpus unless it passes)
    ./python -m corpora med-osce-fetch       download once, verify, extract
    ./python -m corpora med-osce-segment     segment into 1-30 s utterances, freeze ids and the manifest
    ./python -m corpora status
"""
from __future__ import annotations

import json
from pathlib import Path

ALIASES = ("med-osce", "syn-dialog")


def corpus_dir(alias: str) -> Path:
    from pool import common
    if alias not in ALIASES:
        raise ValueError(f"unknown corpus alias {alias!r}")
    return common.corpora_dir() / alias


def status(alias: str) -> dict:
    path = corpus_dir(alias) / "status.json"
    return json.loads(path.read_text()) if path.exists() else {}
