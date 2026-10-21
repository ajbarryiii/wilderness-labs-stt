"""Derived 2-bit export of MB1 in M1's "parakeet-ternary-v1" layout (plans/benchmark.md stage 1;
plans/gpu-kernel.md A1: the baselines R-compact / R-expanded / R-hybrid / R-bf16x3 run MB1 this way).

For every quantized module: binary_export.unpack_codes (bit 1 = +1, bit 0 = -1) then the ternary
export.pack_codes (+1 -> field 01, -1 -> field 10; no 00 fields). Scales and every other tensor are
copied byte for byte, the tokenizer files likewise. The manifest is MB1's with format, file hashes,
code histogram (zero 0-codes) and byte counts updated, plus `derived_from` (MB1 manifest and file
hashes, this script's hash). CPU only. Output (A10):
/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/H/derived/MB1-ternary-layout/

    CUDA_VISIBLE_DEVICES= gpu1bit/shared/python -m gpu1bit.shared.derive
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from gpu1bit.shared import env

TERNARY_FORMAT = "parakeet-ternary-v1"


def derive_codes(binary_codes: torch.Tensor, in_features: int) -> torch.Tensor:
    """MB1 uint8 [N, ceil(K/8)] -> ternary-layout uint8 [N, ceil(K/4)] with the same +-1 weights."""
    import binary_export
    import export as ternary_export
    return ternary_export.pack_codes(binary_export.unpack_codes(binary_codes, in_features))


def main(out_dir: Path = env.DERIVED_EXPORT) -> dict:
    import binary_export
    import export as ternary_export
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    env.require_mount()
    source = env.MB1_EXPORT
    manifest = json.loads((source / binary_export.MANIFEST).read_text())
    if manifest["format"] != binary_export.FORMAT or manifest["sha256"] != env.MB1_SHA256:
        raise ValueError("unexpected MB1 export")
    for rel, digest in manifest["files"].items():
        if env.sha256_file(source / rel) != digest:
            raise ValueError(f"{source / rel} does not match its manifest SHA-256")
    tensors = load_file(source / manifest["file"])
    with safe_open(source / manifest["file"], framework="pt") as f:
        source_metadata = f.metadata()
    out: dict[str, torch.Tensor] = {}
    histogram = torch.zeros(3, dtype=torch.long)
    per_module = {}
    code_bytes = 0
    for name, layer in manifest["quantized_layers"].items():
        n, k = layer["shape"]
        derived = derive_codes(tensors[f"{name}.codes"], k)
        signs = ternary_export.unpack_codes(derived, k)
        counts = torch.bincount(signs.flatten().long() + 1, minlength=3)
        histogram += counts
        per_module[name] = {"minus_one": counts[0].item() / signs.numel(), "zero": counts[1].item() / signs.numel(),
                            "plus_one": counts[2].item() / signs.numel()}
        out[f"{name}.codes"] = derived
        code_bytes += derived.nbytes
    for key, value in tensors.items():
        if key.endswith(".codes") and key[:-len(".codes")] in manifest["quantized_layers"]:
            continue
        out[key] = value  # byte-identical: same dtype, shape and contents
    if histogram[1].item() != 0:
        raise AssertionError("derived codes contain zeros")
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + f".tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / ternary_export.TOKENIZER_DIR).mkdir(parents=True)
    path = tmp / ternary_export.FILE
    save_file(out, path, metadata={"format": TERNARY_FORMAT})
    os.chmod(path, 0o644)
    files = {ternary_export.FILE: env.sha256_file(path)}
    for rel in manifest["files"]:
        if rel != manifest["file"]:
            shutil.copyfile(source / rel, tmp / rel)
            files[rel] = env.sha256_file(tmp / rel)
            if files[rel] != manifest["files"][rel]:
                raise AssertionError(f"{rel} copy differs")
    total = histogram.sum().item()
    derived_manifest = dict(manifest)
    derived_manifest.update(
        format=TERNARY_FORMAT, file=ternary_export.FILE, sha256=files[ternary_export.FILE], files=files,
        bit_layout="ternary-v1 2-bit fields, 4 per byte, code j at bits 2*(j%4) of byte j//4: 01 = +1, 10 = -1 (00 unused)",
        code_histogram={"minus_one": histogram[0].item() / total, "zero": histogram[1].item() / total,
                        "plus_one": histogram[2].item() / total, "counts": histogram.tolist(), "per_module": per_module},
        bytes={**manifest["bytes"], "packed_code_bytes": code_bytes, "file_bytes": path.stat().st_size},
        derived_from={"format": manifest["format"], "export": str(source), "manifest_sha256": env.sha256_file(source / binary_export.MANIFEST),
                      "files": manifest["files"], "safetensors_metadata": source_metadata,
                      "script": "finetune/parakeet-binary/gpu1bit/shared/derive.py",
                      "script_sha256": env.sha256_file(Path(__file__)),
                      "rule": "binary_export.unpack_codes -> export.pack_codes; every other tensor and file byte-identical"})
    (tmp / ternary_export.MANIFEST).write_text(json.dumps(derived_manifest, indent=2) + "\n")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, out_dir)
    print(json.dumps({"out": str(out_dir), "sha256": files[ternary_export.FILE], "packed_code_bytes": code_bytes,
                      "histogram": histogram.tolist()}))
    return derived_manifest


if __name__ == "__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit('run with CUDA_VISIBLE_DEVICES="" (CPU only)')
    main()
