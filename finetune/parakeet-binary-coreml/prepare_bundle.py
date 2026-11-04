#!/usr/bin/env python3
"""Build a LocalFlow model bundle of the 1-bit kea-b (MB1) model with one 1-bit encoder arm.

Derived from LocalFlow's scripts/prepare-parakeet.py (4/8/15/30-second multifunction encoder, folded
rel-pos tables extended to 30 s, FP16 plain layout, native FP32 front end and decoder/joint blobs), but it
reads the parakeet-binary-v1 export natively (mil/binary_export.py; no 2-bit re-encoding) and encodes
the 240 arm-encoded projections with a 1-bit arm (mil/encodings.py BINARY_ARMS).

  macguard --rss-cap 6G --timeout 1800 --threads 4 -- ios/pyenv/.venv/bin/python \
      finetune/parakeet-binary-coreml/prepare_bundle.py --arm B1L \
      --output /Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/kea-b1/B1L/bundle

Writes OUTPUT/{Encoder.mlmodelc, frontend.*, decoder_joint.*, vocabulary.json, bundle.json}; the conversion
(.mlpackage, build manifest) goes to OUTPUT/../conversion and a build record to OUTPUT/../build-record.json.
All outputs must be in the artifact area. Refuses any export other than MB1 (SHA-256 pinned).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import localflow  # noqa: F401  (sys.path for ios/)

MB1_EXPORT_SHA256 = "9873d2f1a53127c673d0201db98cd3ba859ba432f334fef55e9e24fd057b9651"
# Kea-B' (relabel stage 4 run b100-main): the parakeet-binary-v1 scoring view of its keab-8bit-v1 bundle
# (c00230f8...); the 8-bit parts are stored there as their FP16 forward weights W8.
KEAB_PRIME_EXPORT_SHA256 = "ec05526ed7aa7c742ee970c09724b1b8ccfaef6eb10f2ff1a11b031e9dfc4995"
EXPORTS = {MB1_EXPORT_SHA256: ("kea-b", 250000), KEAB_PRIME_EXPORT_SHA256: ("kea-b-prime", 250000 + 97137)}
MAC_TOKENIZER = Path("/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/freeflow-main/export/tokenizer")
BASE_FILES = ("frontend.json", "frontend.f32bin", "decoder_joint.json", "decoder_joint.f32bin", "vocabulary.json")
VOCAB_SIZE = 1024


def check_export(sha256: str) -> tuple[str, int]:
    """(base_model, training_step) of a pinned 1-bit export: MB1 or Kea-B'."""
    if sha256 not in EXPORTS:
        raise ValueError("This bundle requires a pinned 1-bit export (MB1 main-binB-m1-lr2e-4 step 250000, "
                         f"or Kea-B' b100-main); refusing export {sha256}")
    return EXPORTS[sha256]


def vocabulary(tokenizer_dir: Path) -> dict[int, str]:
    """The SentencePiece pieces by id (NeMo's vocab.txt is the legacy WordPiece view and omits <unk>)."""
    pieces = [line.split("\t")[0] for line in (Path(tokenizer_dir) / "tokenizer.vocab").read_text().splitlines()]
    if len(pieces) != VOCAB_SIZE:
        raise ValueError(f"Expected {VOCAB_SIZE} SentencePiece vocabulary entries, got {len(pieces)}")
    return dict(enumerate(pieces))


def bundle_files(output: Path) -> list[str]:
    """Bundle-relative paths of every file LocalFlow loads: the base files and all of Encoder.mlmodelc."""
    output = Path(output)
    missing = [name for name in BASE_FILES if not (output / name).is_file()]
    encoder = output / "Encoder.mlmodelc"
    if missing or not (encoder / "model.mil").is_file() or not (encoder / "weights" / "weight.bin").is_file():
        raise ValueError(f"incomplete bundle in {output}: missing {missing or 'Encoder.mlmodelc files'}")
    return list(BASE_FILES) + [str(p.relative_to(output)) for p in sorted(encoder.rglob("*")) if p.is_file()]


def _sha256(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def bundle_arms() -> tuple[str, ...]:
    """The 1-bit arms plus the corrected 0/1 arm; the timing-only arms (wrong outputs) never make a bundle."""
    from mil import encodings

    return encodings.BINARY_ARMS + encodings.CORRECTED_ARMS


def bundle_manifest(output: Path, arm: str, export_sha256: str, files: list[str],
                    buckets: tuple[int, ...] = localflow.BUCKETS) -> dict:
    if arm not in bundle_arms():
        raise ValueError(f"{arm} is not a bundle arm ({', '.join(bundle_arms())})")
    base_model, training_step = check_export(export_sha256)
    return {
        "model": "localflow", "display_name": "LocalFlow", "base_model": base_model, "training_step": training_step,
        "export_sha256": export_sha256, "encoder": arm, "layout": "plain",
        "compute_units": "cpuAndNeuralEngine", "decoder": "native-fp32",
        "buckets_seconds": sorted(buckets), "sample_rate": 16000,
        "files": {name: _sha256(Path(output) / name) for name in files},
    }


def build(arm: str, output: Path, export: Path | None, tokenizer: Path,
          buckets: tuple[int, ...] = localflow.BUCKETS) -> dict:
    import artifacts
    import native
    from mil import build as build_mod, weights
    from mil.binary_export import BinaryExportSource

    localflow.apply_buckets(buckets)
    export = Path(export or weights.default_path("mb1")).resolve()
    output = artifacts.check(output)
    work = artifacts.check(output.parent)
    t0 = time.time()
    source = BinaryExportSource(export)  # verifies the export file's SHA-256
    export_sha = source.manifest["sha256"]
    base_model, _ = check_export(export_sha)
    tokenizer_hashes = source.verify_tokenizer(tokenizer)
    # this process only: the "mb1" source reads --export; native blobs read the same file's stored tensors
    original_path = weights.default_path
    weights.default_path = lambda name: export if name == "mb1" else original_path(name)
    prov = {"model": "mb1", "export_dir": str(export), "export_sha256": export_sha}
    native.source_tensors = lambda model, keys: ({k: source.raw(k) for k in keys}, prov)
    output.mkdir(parents=True, exist_ok=True)
    native.cmd_frontend(SimpleNamespace(model="mb1", out=str(output)))
    native.cmd_weights(SimpleNamespace(model="mb1", out=str(output)))
    t_native = time.time()
    manifest = build_mod.build_encoder("mb1", arm, "multi", work / "conversion", plan=False, tag="-localflow",
                                       precision="fp16", layout="plain")
    encoder_dir = output / "Encoder.mlmodelc"
    if encoder_dir.exists():
        shutil.rmtree(encoder_dir)
    shutil.copytree(manifest["paths"]["mlmodelc"], encoder_dir)
    (output / "vocabulary.json").write_text(json.dumps(vocabulary(tokenizer), ensure_ascii=False))
    for stem in ("frontend", "decoder_joint"):
        path = output / f"{stem}.json"
        data = json.loads(path.read_text())
        data["provenance"] = {"model": "localflow", "base_model": base_model, "export_sha256": export_sha}
        path.write_text(json.dumps(data, indent=2) + "\n")
    files = bundle_files(output)
    bundle = bundle_manifest(output, arm, export_sha, files, buckets)
    (output / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    weight_bin = encoder_dir / "weights" / "weight.bin"
    record = {
        "arm": arm, "buckets_seconds": sorted(buckets), "bundle": str(output), "bundle_json_sha256": _sha256(output / "bundle.json"),
        "export": {"dir": str(export), "sha256": export_sha, "format": source.FORMAT},
        "tokenizer": {"dir": str(tokenizer), "verified_against_export_manifest": tokenizer_hashes},
        "chain": manifest["encoding"]["chain"],
        "constexpr_inputs_by_module": manifest["encoding"]["constexpr_inputs_by_module"],
        "encoded_bytes_total": manifest["encoding"]["encoded_bytes_total"],
        "encoded_bytes_by_const": manifest["encoding"]["encoded_bytes_by_const"],
        "program_constexpr_literals": {fn: d["constexpr_inputs"] for fn, d in manifest["program"].items()},
        "matmul_followed_by_mul": {fn: d["matmul_followed_by_mul"] for fn, d in manifest["program"].items()},
        "weight_bin_bytes": weight_bin.stat().st_size,
        "encoder_mlmodelc_bytes": sum((output / f).stat().st_size for f in files if f.startswith("Encoder.mlmodelc/")),
        "bundle_bytes": sum((output / f).stat().st_size for f in files),
        "mlpackage_bytes": manifest["sizes"]["mlpackage_bytes"],
        "timing_s": {**manifest["timing_s"], "native_blobs": round(t_native - t0, 1),
                     "compile": manifest["compile"]["compile_s"], "total": round(time.time() - t0, 1)},
        "peak_rss_mb": {"after_save": manifest["peak_rss_mb_after_save"],
                        "after_compile": manifest.get("peak_rss_mb_after_compile")},
        "pass_pipeline": manifest["pass_pipeline"]["name"], "opset": manifest["opset"],
        "environment": manifest["environment"],
        "conversion_manifest": str(Path(manifest["paths"]["mlpackage"]).parent / "manifest-multi-localflow-24L.json"),
    }
    (work / "build-record.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main() -> None:
    from mil import encodings

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", required=True, choices=bundle_arms())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export", type=Path, help="parakeet-binary-v1 export dir (default: weights.default_path('mb1'))")
    parser.add_argument("--tokenizer", type=Path, default=MAC_TOKENIZER)
    parser.add_argument("--buckets", default=",".join(map(str, localflow.BUCKETS)),
                        help="encoder functions in seconds (default 15, the one LocalFlow loads; 4,8,15,30 for all)")
    args = parser.parse_args()
    record = build(args.arm, args.output, args.export, args.tokenizer,
                   tuple(int(s) for s in args.buckets.split(",")))
    print(json.dumps({k: record[k] for k in ("arm", "bundle_json_sha256", "weight_bin_bytes", "bundle_bytes",
                                            "timing_s", "peak_rss_mb")}))


if __name__ == "__main__":
    main()
