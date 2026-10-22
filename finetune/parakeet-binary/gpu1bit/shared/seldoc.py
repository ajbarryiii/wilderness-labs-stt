"""Selection documents (plans/x2-default.md Y1, Z2).

`shared/selected.json` is the deployment POINTER. Until promotion it is the x3 document; promotion makes
it a byte-identical copy of the candidate. `shared/selected-x2.json` is the frozen CANDIDATE: the
B-hybrid-x2 entry, B-hybrid-x3 as the override, and the M1 x2 default. The gates, the benchmark and test
scoring use the candidate explicitly.

- The document in use is GPU1BIT_SELECTION when set (inherited by workers), else the pointer.
- Every arm identity (fulldev) and gate fingerprint (fingerprint.make) hashes the CONTENT of the document
  in use (`load(...)["sha256"]`), never the pointer's path. Promotion therefore leaves the qualified
  identities unchanged, and an unqualified caller after promotion has them.
- A document may carry "parts": {"2": entry, "3": entry}. An entry has "kernel", "imports" and "hybrid"
  ({provider, pipeline, imports}). A document without "parts" (the x3 document) is its own part-3 entry.
  "default_parts" (MB1) and "m1_default_parts" (M1) default to 3.
- The pointer is accepted only byte-identical to the x3 document pinned at a freeze, or to the
  candidate (`pointer_state`); any third state is refused.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from gpu1bit.shared import env

SELECTED = env.SHARED / "selected.json"          # the deployment pointer
CANDIDATE = env.SHARED / "selected-x2.json"      # the frozen x2 candidate
PARTS = (2, 3)


def current_path() -> Path:
    p = os.environ.get("GPU1BIT_SELECTION")
    return Path(p) if p else SELECTED


def load(path: Path | str | None = None) -> dict:
    """{"path", "sha256" (of the bytes), "doc"} of a selection document (default: the one in use)."""
    p = Path(path) if path is not None else current_path()
    data = p.read_bytes()
    doc = json.loads(data)
    if doc.get("runtime") not in ("hybrid", "provider") or not doc.get("kernel"):
        raise ValueError(f"{p}: needs 'kernel' and 'runtime' in ('hybrid', 'provider')")
    return {"path": str(p), "sha256": hashlib.sha256(data).hexdigest(), "doc": doc}


def check_parts(parts) -> int:
    if not isinstance(parts, int) or isinstance(parts, bool) or parts not in PARTS:
        raise ValueError(f"part count must be one of {PARTS}, got {parts!r}")
    return parts


def use_candidate() -> Path:
    """Select the candidate document for this process and its children (GPU1BIT_SELECTION): every x2 entry
    point (gates, preflight and budget, benchmark session, test scoring) is defined under it, also while the
    live pointer is still the x3 document (Codex x2 c4)."""
    os.environ["GPU1BIT_SELECTION"] = str(CANDIDATE)
    return CANDIDATE


def part_count(kernel: str) -> int:
    """A registry name's own activation part count: "<kernel>:x<N>[...]" is N (the part-count layer variants,
    gpu1bit.actprec.parts), anything else (int8_mma, int8_mma:b01, ...) 3."""
    import re
    m = re.search(r":x(\d)(?::|$)", kernel)
    return int(m.group(1)) if m else 3


def x3_selection_record(doc: dict) -> dict:
    """The historical x3 selection record ({session, variant, ...}): the x3 document's own "selection", or the
    candidate's preserved "selection.x3_selection" (Codex x2 c1, finding 8)."""
    sel = doc.get("selection") or {}
    rec = sel.get("x3_selection") if isinstance(sel.get("x3_selection"), dict) else sel
    if not rec.get("session"):
        raise ValueError("the selection document carries no x3 selection record (session)")
    return rec


def default_parts(doc: dict) -> int:
    return check_parts(doc.get("default_parts", 3))


def m1_default_parts(doc: dict) -> int:
    return check_parts(doc.get("m1_default_parts", 3))


def entry(doc: dict, parts: int) -> dict:
    """{kernel, imports, hybrid} for a part count."""
    check_parts(parts)
    if "parts" in doc:
        e = doc["parts"].get(str(parts))
        if e is None:
            raise ValueError(f"the selection document has no part-{parts} entry")
        return {"kernel": e["kernel"], "imports": list(e.get("imports", [])), "hybrid": dict(e.get("hybrid", {}))}
    if parts != 3:
        raise ValueError(f"the x3 selection document has no part-{parts} entry")
    return {"kernel": doc["kernel"], "imports": list(doc.get("imports", [])), "hybrid": dict(doc.get("hybrid", {}))}


def pointer_state(pinned_x3_sha256: str, selected: Path | None = None, candidate: Path | None = None) -> str:
    """'x3' (byte-identical to the pinned x3 document) or 'promoted' (byte-identical to the candidate);
    SystemExit for any third state (Z2)."""
    sel = Path(selected or SELECTED)
    cand = Path(candidate or CANDIDATE)
    digest = env.sha256_file(sel)
    if digest == pinned_x3_sha256:
        return "x3"
    if cand.exists() and sel.read_bytes() == cand.read_bytes():
        return "promoted"
    raise SystemExit(f"{sel} is in a third state (sha256 {digest[:12]}): neither the pinned x3 document "
                     f"({pinned_x3_sha256[:12]}) nor a byte-identical copy of {cand.name}")


def promote(selected: Path | None = None, candidate: Path | None = None) -> str:
    """Replace the pointer with a byte-identical copy of the candidate (one atomic replace)."""
    sel = Path(selected or SELECTED)
    cand = Path(candidate or CANDIDATE)
    tmp = sel.with_name(f".{sel.name}.promote{os.getpid()}")
    tmp.write_bytes(cand.read_bytes())
    os.replace(tmp, sel)
    return env.sha256_file(sel)
