"""Discovery of BinaryLayer implementations (plans/gpu-kernel.md "Team and ownership").

Each agent adds `gpu1bit/<its dir>/layer.py` (never editing H's files) exposing either
    IMPLEMENTATION = SomeBinaryLayerClass                      # registered as "<dir>"
or
    IMPLEMENTATIONS = {"variant": SomeClass, ...}              # registered as "<dir>:<variant>"
H's known-good FP64 reference is registered as "h_reference". Modules are imported lazily, by name,
under the package path gpu1bit.<dir>.layer, so a broken candidate does not break the others.
Environment GPU1BIT_IMPLEMENTATIONS="a,b" restricts discovery (names or dir prefixes).
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

from gpu1bit.shared import env

SKIP = {"shared", "tests", "__pycache__"}


def directories() -> list[str]:
    return sorted(p.name for p in env.GPU1BIT.iterdir()
                  if p.is_dir() and p.name not in SKIP and (p / "layer.py").is_file())


def discover(include_reference: bool = True) -> dict[str, object]:
    """{name: class or ImportError}; import failures are kept so tests report them per candidate."""
    found: dict[str, object] = {}
    if include_reference:
        from gpu1bit.shared.reference_layer import IMPLEMENTATION
        found["h_reference"] = IMPLEMENTATION
    for d in directories():
        try:
            module = importlib.import_module(f"gpu1bit.{d}.layer")
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            found[d] = ImportError(f"gpu1bit/{d}/layer.py failed to import: {exc!r}")
            continue
        if hasattr(module, "IMPLEMENTATIONS"):
            for variant, cls in dict(module.IMPLEMENTATIONS).items():
                found[f"{d}:{variant}"] = cls
        elif hasattr(module, "IMPLEMENTATION"):
            found[d] = module.IMPLEMENTATION
        else:
            found[d] = ImportError(f"gpu1bit/{d}/layer.py exposes neither IMPLEMENTATION nor IMPLEMENTATIONS")
    wanted = os.environ.get("GPU1BIT_IMPLEMENTATIONS")
    if wanted:
        keys = [w.strip() for w in wanted.split(",") if w.strip()]
        found = {n: c for n, c in found.items() if any(n == w or n.split(":")[0] == w for w in keys)}
    return found


def get(name: str):
    found = discover()
    if name not in found:
        raise KeyError(f"unknown implementation {name!r}; known: {sorted(found)}")
    cls = found[name]
    if isinstance(cls, Exception):
        raise cls
    return cls


def source_path(name: str) -> Path:
    return env.SHARED / "reference_layer.py" if name == "h_reference" else env.GPU1BIT / name.split(":")[0] / "layer.py"
