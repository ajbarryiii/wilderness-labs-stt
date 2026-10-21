"""Fingerprints that bind records to the exact code, data and environment they came from
(Codex code review c1, finding 2: gate results must not survive a source, export, environment or
workload change).

`make(kind, **extra)` returns a dict with a `hash` over:
  sources    SHA-256 of every file a gpu1bit run can import or read: H's shared/ (code + data),
             gpu1bit/runtime.py, EVERY file of every candidate directory (the registry imports all of
             them; B's patch modules live there too), M1's inference/*.py, the parakeet-ternary and
             parakeet-binary modules, hf/load_ternary.py, whisper-ternary wer.py and quant.py, the
             interpreter wrappers, plans/gpu-kernel.md (the frozen rules; A15 since Codex GPU c3) and
             plans/x2-default.md (the x2 gates, rosters and Z4 pins);
  exports    MB1's and the derived 2-bit export's COMPLETE export directories (Codex GPU code review
             c2): every file hashed (weights, manifest.json with the model configuration the loaders
             build NeMo from, tokenizer files), plus the manifest's weights digest and a configuration
             hash for readability; the same definition as fulldev.export_identity. The loaders still
             verify every file against the manifest;
  workloads  workloads.json's file SHA-256 and frozen content hash;
  selection  the SHA-256 of the CONTENT of the selection document in use (shared/seldoc.py: GPU1BIT_SELECTION
             or the pointer shared/selected.json, which itself is not a source), so promotion
             (a byte-identical copy) leaves fingerprints unchanged (plans/x2-default.md Y1, Z2);
  versions   torch, triton, nemo_toolkit, transformers;
  extra      caller fields (e.g. the implementation, the reference, the patch modules).
The GPU's identity (device name, driver) is recorded per record by GPU processes (`gpu_identity`).
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
from pathlib import Path

from gpu1bit.shared import env

CANDIDATE_DIRS = ("int8_mma", "triton_bf16", "bitserial")


def source_files() -> list[Path]:
    t, b = env.TERNARY_DIR, env.BINARY_DIR
    # shared/selected.json is the deployment POINTER (plans/x2-default.md Y1, Z2): never a source; the
    # CONTENT of the selection document in use is hashed instead (make()["selection"]).
    files = [p for p in env.SHARED.iterdir() if p.is_file() and p.suffix in (".py", ".json", "") and p.name != "__pycache__"
             and p.name != "selected.json"]
    files.append(env.GPU1BIT / "runtime.py")
    # the x2-default code (plans/x2-default.md): loaders, the actprec patch, the x2 package, the orchestrators
    files += [env.GPU1BIT / f for f in ("runtime_m1.py", "actprec_report.py", "rerun_timing.py")]
    for d in ("actprec", "x2"):
        files += [p for p in (env.GPU1BIT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for d in CANDIDATE_DIRS:
        files += [p for p in (env.GPU1BIT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += sorted((t / "inference").glob("*.py"))
    files += [t / f for f in ("evaluate.py", "export.py", "quant.py", "paths.py", "python")] + [t / "hf" / "load_ternary.py"]
    files += sorted(b.glob("binary_*.py")) + [b / "python"]
    files += [env.REPO / "finetune" / "whisper-ternary" / f for f in ("wer.py", "quant.py")]
    files.append(b / "plans" / "gpu-kernel.md")   # the frozen rules, incl. A15's winner rule (Codex GPU c3)
    files.append(b / "plans" / "x2-default.md")   # the x2 gates, rosters, bounds and the Z4 row-file pins
    return sorted({p for p in files if p.exists()})


def sources() -> dict[str, str]:
    return {str(p.relative_to(env.REPO)): env.sha256_file(p) for p in source_files()}


def exports() -> dict[str, dict]:
    from gpu1bit.shared.fulldev import export_identity
    out = {}
    for name, d in (("MB1", env.MB1_EXPORT), ("derived", env.DERIVED_EXPORT)):
        doc = export_identity(Path(d))
        out[name] = {k: doc[k] for k in ("sha256", "files", "config_sha256")}
    return out


def workloads() -> dict[str, str]:
    path = env.SHARED / "workloads.json"
    return {"file_sha256": env.sha256_file(path), "content_sha256": json.loads(path.read_text())["content_sha256"]}


def versions() -> dict[str, str]:
    return {k: md.version(k) for k in ("torch", "triton", "nemo_toolkit", "transformers")}


def make(kind: str, **extra) -> dict:
    from gpu1bit.shared import seldoc
    doc = {"kind": kind, "sources": sources(), "exports": exports(), "workloads": workloads(),
           "versions": versions(), "selection": {"sha256": seldoc.load()["sha256"]}, "extra": extra}
    doc["hash"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    return doc


def diff(a: dict, b: dict) -> list[str]:
    out = []
    for key in sorted(set(a) | set(b)):
        if key == "hash":
            continue
        va, vb = a.get(key), b.get(key)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += [f"{key}.{k}" for k in sorted(set(va) | set(vb)) if va.get(k) != vb.get(k)]
        elif va != vb:
            out.append(key)
    return out


def freeze(path: Path, doc: dict) -> dict:
    """Write `doc` at `path` once; an existing file must have the same hash (else SystemExit)."""
    env.require_mount()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text())
        if old["hash"] != doc["hash"]:
            raise SystemExit(f"{path}: fingerprint changed since it was frozen: {diff(old, doc)[:12]}")
        return old
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def gpu_identity() -> dict:
    """Device name and driver of GPU 0 (call from a GPU process holding the lock)."""
    import torch
    from inference.benchmark import energy_meter
    meter = energy_meter()
    try:
        meta = meter.metadata()
    finally:
        meter.close()
    return {"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "driver": meta.get("driver_version"), "uuid": meta.get("uuid")}
