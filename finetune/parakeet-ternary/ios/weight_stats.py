"""Weight statistics of a pilot ternary export for seeded random benchmark models. See DESIGN.md "Benchmark weights".

Reads a "parakeet-ternary-v1" export (../export.py: packed codes, FP32 row scales, FP16 other
tensors, FP32 window and filterbank), checks every file against its manifest SHA-256, and writes
weight_stats.json (no raw weights):
- provenance: export directory and run, export and manifest SHA-256, format, base model, the
  pinned .nemo's SHA-256 (lock.json) and the SHA-256 of its model_config.yaml;
- model_config: preprocessor, encoder, decoder, joint (without the vocabulary), decoding and
  model_defaults sections of the pinned .nemo's model_config.yaml (reference.Config input),
  checked equal to the export's stored config;
- ternary: per quantized module its kind, [out, in] shape, code counts and the per-row scale
  mean/std/min/max plus 257 quantiles (probabilities k/256, numpy linear interpolation);
- float: per other floating tensor its shape, stored dtype, mean/std/min/max, 129 value
  quantiles, all-zero rows (2-D tensors), whether every value is positive and the count of exact
  zeros (the P2 export has one BatchNorm running_var entry stored as 0: a tiny positive variance
  underflowed in FP16; the generator draws strictly positive variances regardless);
- constants: the window and filterbank exactly as the export (NVIDIA's checkpoint) stores them
  (base64 float32; the filterbank as nonzero indices + values), their SHA-256, and for information
  the SHA-256 of reference.py's numpy computation and its difference from the stored values
  (at most one float32 ulp in a few entries; a portable numpy computation is not bit-identical
  across x86-64 and arm64, so the generator writes the stored values, not computed ones);
- integer: integer buffers and their (single) value.
Quantiles are base64 little-endian float32 (randomweights.encode_f32).

Runs in the NeMo env without importing NeMo (one module unpacked at a time, < 1 GB):
  ../python weight_stats.py [--export DIR] [--out weight_stats.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import paths  # noqa: E402
import randomweights as rw  # noqa: E402
import reference  # noqa: E402

DEFAULT_EXPORT = paths.RUNS / "pilot-P2-lr5e-4" / "export"
CONFIG_SECTIONS = ("preprocessor", "encoder", "decoder", "joint", "decoding", "model_defaults")


def _summary(values: np.ndarray, count: int) -> dict:
    x = values.astype(np.float64).ravel()
    q = np.quantile(x, rw.quantile_grid(count), method="linear").astype(np.float32)
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
            "n_quantiles": count, "quantiles": rw.encode_f32(q)}


def nemo_model_config() -> tuple[dict, str]:
    """The selected sections of the pinned .nemo's model_config.yaml and that member's SHA-256."""
    import yaml

    with tarfile.open(paths.MODEL_FILE, "r:*") as tar:
        member = next(m for m in tar.getmembers() if Path(m.name).name == "model_config.yaml")
        data = tar.extractfile(member).read()
    cfg = yaml.safe_load(data)
    out = {k: cfg[k] for k in CONFIG_SECTIONS}
    out["joint"] = {k: v for k, v in out["joint"].items() if k != "vocabulary"}
    return out, rw.sha256_bytes(data)


def collect(export_dir: Path) -> dict:
    import torch
    from safetensors import safe_open

    import export

    export_dir = Path(export_dir).resolve()
    manifest_bytes = (export_dir / export.MANIFEST).read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["format"] != export.FORMAT:
        raise ValueError(f"unknown export format {manifest['format']!r}")
    for rel, digest in manifest["files"].items():
        if export.sha256_file(export_dir / rel) != digest:
            raise ValueError(f"{export_dir / rel} does not match the manifest SHA-256")
    model_config, config_sha = nemo_model_config()
    for section in CONFIG_SECTIONS:
        stored = manifest["config"][section]
        if section == "joint":
            stored = {k: v for k, v in stored.items() if k != "vocabulary"}
        if stored != model_config[section]:
            raise ValueError(f"export config section {section} differs from the pinned .nemo's model_config.yaml")
    cfg = reference.Config.from_model_config(model_config)
    lock = json.loads((paths.MODEL_DIR / "lock.json").read_text())
    stats = {
        "format": rw.STATS_FORMAT,
        "provenance": {
            "export_dir": str(export_dir), "run": export_dir.parent.name, "export_format": manifest["format"],
            "export_file": manifest["file"], "export_sha256": manifest["sha256"],
            "manifest_sha256": rw.sha256_bytes(manifest_bytes), "base_model": manifest["base_model"],
            "nemo_file_sha256": lock["files"][paths.MODEL_FILE.name], "model_config_yaml_sha256": config_sha,
            "quantiles": "base64 little-endian float32 at probabilities k/(n-1), numpy.quantile linear on float64",
        },
        "model_config": model_config,
        "ternary": {}, "float": {}, "constants": {}, "integer": {},
    }
    with safe_open(str(export_dir / manifest["file"]), framework="pt") as tensors:
        names = set(tensors.keys())
        for name, layer in sorted(manifest["quantized_layers"].items()):
            out_features, in_features = layer["shape"]
            codes = export.unpack_codes(tensors.get_tensor(f"{name}.codes"), in_features)
            scale = tensors.get_tensor(f"{name}.scale")
            names -= {f"{name}.codes", f"{name}.scale"}
            if layer["bias"] or scale.dtype != torch.float32 or not bool((scale > 0).all()):
                raise ValueError(f"{name}: expected no bias and positive FP32 scales")
            counts = torch.bincount(codes.flatten().long() + 1, minlength=3).tolist()
            stats["ternary"][name] = {
                "kind": layer["kind"], "shape": [out_features, in_features],
                "codes": dict(zip(("minus_one", "zero", "plus_one"), counts)),
                "scale": _summary(scale.numpy(), rw.SCALE_QUANTILES)}
        for name in sorted(names):
            value = tensors.get_tensor(name)
            if not value.is_floating_point():
                if value.numel() != 1:
                    raise ValueError(f"{name}: only scalar integer buffers are expected")
                stats["integer"][name] = {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
                                          "value": int(value)}
            elif name in export.FP32_BUFFERS:
                stored = value.numpy().astype(np.float32)
                computed = (reference.hann_window(cfg.win_length) if name.endswith("window")
                            else reference.mel_filterbank(cfg.sample_rate, cfg.n_fft, cfg.features)[None])
                if computed.shape != stored.shape:
                    raise ValueError(f"{name}: computed shape {computed.shape} != stored {stored.shape}")
                flat = stored.ravel()
                nonzero = np.flatnonzero(flat.view(np.uint32))  # by bit pattern: keeps any -0.0
                entry = {"shape": list(stored.shape), "sha256": rw.sha256_bytes(stored.astype("<f4").tobytes()),
                         "computed_sha256": rw.sha256_bytes(computed.astype("<f4").tobytes()),
                         "max_abs_diff_vs_computed": float(np.abs(computed - stored).max()),
                         "entries_differing_from_computed": int((computed != stored).sum())}
                if len(nonzero) < flat.size // 4:  # the filterbank is sparse
                    entry.update(nonzero_index=nonzero.tolist(), values=rw.encode_f32(flat[nonzero]))
                else:
                    entry["values"] = rw.encode_f32(flat)
                stats["constants"][name] = entry
            else:
                array = value.float().numpy()
                entry = {"shape": list(value.shape), "export_dtype": str(value.dtype).removeprefix("torch."),
                         **_summary(array, rw.VALUE_QUANTILES), "positive": bool((array > 0).all()),
                         "zero_values": int((array == 0).sum())}
                if array.ndim == 2:
                    zero_rows = np.flatnonzero(~array.any(axis=1)).tolist()
                    if zero_rows:
                        entry["zero_rows"] = zero_rows
                stats["float"][name] = entry
    var = [n for n in stats["float"] if n.endswith("running_var")]
    if not var or not all(stats["float"][n]["min"] >= 0 for n in var):  # FP16 storage can round tiny variances to 0
        raise ValueError("BatchNorm running_var must be present and non-negative")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--out", type=Path, default=rw.STATS_FILE)
    args = parser.parse_args()
    stats = collect(args.export)
    text = json.dumps(stats, indent=1, sort_keys=True) + "\n"
    args.out.write_text(text)
    hist = np.array([[e["codes"][k] for k in ("minus_one", "zero", "plus_one")] for e in stats["ternary"].values()])
    print(json.dumps({"out": str(args.out), "bytes": len(text.encode()), "ternary_modules": len(stats["ternary"]),
                      "float_tensors": len(stats["float"]), "code_fractions": (hist.sum(0) / hist.sum()).round(4).tolist(),
                      "constants": stats["constants"]}, indent=1))


if __name__ == "__main__":
    main()
