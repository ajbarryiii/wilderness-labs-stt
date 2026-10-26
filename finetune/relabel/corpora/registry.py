"""The private source registry: $PRIVATE_REGISTRY/sources.json maps alias -> {url or hf_id, revision, license,
acquisition, ...}. Committed code names sources only by alias; everything else is read from here."""
from __future__ import annotations

import json

from pool import common


def source(alias: str) -> dict:
    path = common.root("PRIVATE_REGISTRY") / "sources.json"
    if not path.exists():
        raise RuntimeError(f"{path} is missing (the private source registry)")
    data = json.loads(path.read_text())
    if alias not in data:
        raise RuntimeError(f"alias {alias!r} is not in the private source registry")
    return data[alias]
