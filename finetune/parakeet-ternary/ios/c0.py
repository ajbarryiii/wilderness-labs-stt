"""C0 artifact: pin FluidInference/parakeet-tdt-0.6b-v2-coreml, record every file's SHA-256 (c0.json), download.

DESIGN.md "Arms" (C0, the product baseline): the unmodified published Core ML export at a pinned revision.
Only the files FluidAudio 0.7.8 uses for v2 are pinned and downloaded:
- ModelNames.ASR (Sources/FluidAudio/ModelNames.swift at tag v0.7.8): Preprocessor.mlmodelc, Encoder.mlmodelc,
  Decoder.mlmodelc, JointDecision.mlmodelc and the vocabulary parakeet_vocab.json (Repo.parakeetV2 =
  FluidInference/parakeet-tdt-0.6b-v2-coreml);
- DownloadUtils.downloadRepo (same tag) also fetches every root *.json / *.txt file: config.json.
The repository's other models (Melspectogram, Melspectrogram_v2, ParakeetEncoder*, ParakeetDecoder,
RNNTJoint) are not used by FluidAudio 0.7.8 and are not downloaded.

Standard library only (runs on NixOS and the Mac).
  python c0.py pin                 # NixOS: HF API at REVISION -> c0.json (LFS SHA-256 from the API; the small
                                   # non-LFS files are fetched, hashed and checked against their git blob ids)
  python c0.py download --out DIR  # Mac (through macguard): DIR/<file>, every SHA-256 checked against c0.json
  python c0.py verify --dir DIR    # re-hash a download against c0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
C0_JSON = HERE / "c0.json"
REPO = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
REVISION = "ee09c569f73759e6d44c9bd16766f477b2b36d39"
MODELS = ("Preprocessor", "Encoder", "Decoder", "JointDecision")
ROOT_FILES = ("parakeet_vocab.json", "config.json")
FLUIDAUDIO = {"version": "0.7.8", "tag": "v0.7.8", "commit": "8136bd0642e7c5ce1f6f5b2931890266aeecb08c",
              "model_names": "https://github.com/FluidInference/FluidAudio/blob/8136bd0642e7c5ce1f6f5b2931890266aeecb08c/"
                             "Sources/FluidAudio/ModelNames.swift",
              "download_utils": "https://github.com/FluidInference/FluidAudio/blob/8136bd0642e7c5ce1f6f5b2931890266aeecb08c/"
                                "Sources/FluidAudio/DownloadUtils.swift",
              "compute_units": {"Preprocessor": "cpuOnly (AsrModels.createModelSpecs)",
                                "Encoder": "configuration (default cpuAndNeuralEngine)",
                                "Decoder": "configuration (default cpuAndNeuralEngine)",
                                "JointDecision": "configuration (default cpuAndNeuralEngine)"},
              "allowLowPrecisionAccumulationOnGPU": True}
UA = {"User-Agent": "wilderness-labs-stt/parakeet-ios c0.py"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def resolve_url(path: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{path}"


def wanted(path: str) -> bool:
    return path in ROOT_FILES or any(path.startswith(m + ".mlmodelc/") for m in MODELS)


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def schema(entries: list[dict]) -> list[dict]:
    keys = ("name", "dataType", "shape", "hasShapeFlexibility", "shapeFlexibility", "shapeRange", "formattedType")
    return [{k: e[k] for k in keys if k in e} for e in entries]


def cmd_pin(args) -> None:
    info = json.loads(get(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}"))
    tree = json.loads(get(f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true"))
    files, models, skipped = [], {}, set()
    for entry in sorted(tree, key=lambda e: e["path"]):
        if entry["type"] != "file":
            continue
        path = entry["path"]
        if not wanted(path):
            if path.endswith(".bin") or "/" in path:
                skipped.add(path.split("/")[0])
            continue
        lfs = entry.get("lfs")
        record = {"path": path, "bytes": entry["size"], "lfs": bool(lfs), "git_oid": entry["oid"]}
        if lfs:
            record["sha256"] = lfs["oid"]
        else:
            data = get(resolve_url(path))
            if len(data) != entry["size"] or git_blob_sha1(data) != entry["oid"]:
                raise ValueError(f"{path}: size or git blob id mismatch")
            record["sha256"] = hashlib.sha256(data).hexdigest()
            if path.endswith("/metadata.json"):
                meta = json.loads(data)[0]
                models[path.split(".mlmodelc")[0]] = {
                    "shortDescription": meta.get("shortDescription"), "storagePrecision": meta.get("storagePrecision"),
                    "computePrecision": meta.get("computePrecision"), "specificationVersion": meta.get("specificationVersion"),
                    "modelType": meta.get("modelType", {}).get("name"), "availability": meta.get("availability"),
                    "userDefinedMetadata": meta.get("userDefinedMetadata"), "inputs": schema(meta.get("inputSchema", [])),
                    "outputs": schema(meta.get("outputSchema", [])),
                    "mlProgramOperationTypeHistogram": meta.get("mlProgramOperationTypeHistogram")}
        files.append(record)
    missing = [m for m in MODELS if m not in models]
    if missing:
        raise ValueError(f"no metadata.json for {missing}")
    doc = {
        "schema": 1, "label": "C0",
        "description": "FluidInference's published Parakeet TDT 0.6B v2 Core ML export, the files FluidAudio 0.7.8 "
                       "loads for v2 (see c0.py)",
        "repo": REPO, "revision": REVISION, "revision_last_modified": info.get("lastModified"),
        "pinned_on": time.strftime("%Y-%m-%d"),
        "url": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "fluidaudio": FLUIDAUDIO,
        "not_downloaded": sorted(skipped),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
        "models": models,
    }
    C0_JSON.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"files": len(files), "total_bytes": doc["total_bytes"], "not_downloaded": doc["not_downloaded"]}))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def cmd_download(args) -> None:
    doc = json.loads(C0_JSON.read_text())
    import artifacts

    out = artifacts.check(args.out)  # model files: only under the machine's artifact area, never in Git
    t0, fetched = time.time(), 0
    for f in doc["files"]:
        dest = out / f["path"]
        if dest.exists() and dest.stat().st_size == f["bytes"] and sha256_file(dest) == f["sha256"]:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        h = hashlib.sha256()
        with urllib.request.urlopen(urllib.request.Request(resolve_url(f["path"]), headers=UA), timeout=120) as r, \
                open(tmp, "wb") as handle:
            while chunk := r.read(1 << 20):
                h.update(chunk); handle.write(chunk)
        if tmp.stat().st_size != f["bytes"] or h.hexdigest() != f["sha256"]:
            tmp.unlink()
            raise ValueError(f"{f['path']}: downloaded size or SHA-256 differs from c0.json")
        os.replace(tmp, dest)
        fetched += f["bytes"]
        print(f"fetched {f['path']} ({f['bytes']} bytes)", flush=True)
    cmd_verify(argparse.Namespace(dir=str(out)), extra={"fetched_bytes": fetched, "seconds": round(time.time() - t0, 1)})


def cmd_verify(args, extra: dict | None = None) -> None:
    doc = json.loads(C0_JSON.read_text())
    root = Path(args.dir)
    bad = [f["path"] for f in doc["files"] if not (root / f["path"]).exists() or sha256_file(root / f["path"]) != f["sha256"]]
    result = {"platform": sys.platform, "dir": str(root), "files": len(doc["files"]), "all_sha256_match": not bad,
              "mismatches": bad, **(extra or {})}
    print(json.dumps(result))
    if bad:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pin").set_defaults(func=cmd_pin)
    p = sub.add_parser("download"); p.add_argument("--out", required=True); p.set_defaults(func=cmd_download)
    p = sub.add_parser("verify"); p.add_argument("--dir", required=True); p.set_defaults(func=cmd_verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
