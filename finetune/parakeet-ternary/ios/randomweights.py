"""Seeded random Parakeet-TDT 0.6B v2 models with the pilot export's ternary structure. See DESIGN.md "Benchmark weights".

Input: weight_stats.json (weight_stats.py). Output directory:
- model.safetensors: every tensor of NeMo's state_dict under preprocessor/encoder/decoder/joint.
  Ternary modules as "<module>.codes" (int8 [out, in], pointwise convolutions as their [out, in]
  matrix) plus "<module>.scale" (FP32 [out]); every other floating tensor FP32 (no FP16 rounding
  here; that belongs to later conversion steps); integer buffers int64. Header metadata: format,
  seed, layers, SHA-256 of weight_stats.json.
- manifest.json: per-tensor dtype, shape and SHA-256 of its raw little-endian bytes, a digest over
  those lines (equal digests <=> equal tensors), the file's SHA-256, seed and stats hash.

Each tensor is generated on its own, independent of every other tensor and of --layers:
- RNG: numpy Generator(PCG64(SeedSequence([seed, stream, *SHA-256(name) as 8 uint32]))), stream 0
  for codes and values, 1 for scales. Only Generator.random() doubles are drawn, and they are
  mapped with comparisons and single exactly-rounded IEEE operations (one numpy ufunc each: no
  FMA contraction, no transcendental functions), so Linux x86-64 and macOS arm64 produce identical
  bits; the stored constants are checked against their SHA-256.
- Ternary codes i.i.d. from the module's code histogram: u < P(-1) -> -1, u < P(-1) + P(0) -> 0,
  else +1. Per-row scales i.i.d. by inverse-CDF interpolation of the module's 257 scale quantiles.
- Other floating tensors i.i.d. from their 129 value quantiles; rows that are exactly zero in the
  export stay zero (the blank embedding row, NeMo's padding_idx); BatchNorm running_var is
  checked to be strictly positive (its quantiles are >= 0 and a draw lands exactly on 0 only for
  u == 0). The window and mel filterbank are not drawn: they are the export's stored values
  (weight_stats.json; reference.py's numpy computation agrees to one float32 ulp but is not
  bit-identical across x86-64 and arm64); integer buffers (num_batches_tracked) are copied.
Memory: one tensor at a time (largest 4096 x 1024: 32 MB of doubles), written as it is generated.

CLI: python randomweights.py --stats weight_stats.json --seed 0 --out DIR [--layers N] [--manifest-only]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import resource
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

HERE = Path(__file__).resolve().parent
STATS_FILE = HERE / "weight_stats.json"
STATS_FORMAT = "parakeet-ios-weight-stats-v1"
MODEL_FORMAT = "parakeet-ios-random-v1"
MODEL_FILE = "model.safetensors"
MANIFEST_FILE = "manifest.json"
SCALE_QUANTILES = 257
VALUE_QUANTILES = 129
_ST_DTYPES = {np.dtype(np.int8): "I8", np.dtype(np.float32): "F32", np.dtype(np.int64): "I64"}
_ARRAY_DTYPES = {"int8": np.int8, "float32": np.float32, "int64": np.int64}


def encode_f32(values: np.ndarray) -> str:
    """Base64 of little-endian float32 values (exact and compact in JSON)."""
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")


def decode_f32(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype="<f4").astype(np.float32)


def quantile_grid(count: int) -> np.ndarray:
    return np.arange(count, dtype=np.float64) / (count - 1)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_stats(path: Path = STATS_FILE) -> dict:
    stats = json.loads(Path(path).read_text())
    if stats.get("format") != STATS_FORMAT:
        raise ValueError(f"{path}: unknown stats format {stats.get('format')!r}")
    stats["_sha256"] = sha256_file(Path(path))
    return stats


def _layer(name: str) -> int | None:
    parts = name.split(".")
    return int(parts[2]) if parts[:2] == ["encoder", "layers"] else None


def _keep(name: str, layers: int | None) -> bool:
    layer = _layer(name)
    return layers is None or layer is None or layer < layers


def tensor_specs(stats: dict, layers: int | None = None) -> list[tuple[str, str, tuple[int, ...]]]:
    """(name, dtype, shape) of every output tensor, sorted by name (the file order)."""
    specs = []
    for name, entry in stats["ternary"].items():
        out_features, in_features = entry["shape"]
        specs += [(f"{name}.codes", "int8", (out_features, in_features)), (f"{name}.scale", "float32", (out_features,))]
    specs += [(name, "float32", tuple(e["shape"])) for name, e in stats["float"].items()]
    specs += [(name, "float32", tuple(e["shape"])) for name, e in stats["constants"].items()]
    specs += [(name, e["dtype"], tuple(e["shape"])) for name, e in stats["integer"].items()]
    return sorted(s for s in specs if _keep(s[0], layers))


def _rng(seed: int, stream: int, name: str) -> np.random.Generator:
    words = np.frombuffer(hashlib.sha256(name.encode()).digest(), dtype="<u4")
    entropy = [int(seed), int(stream), *(int(w) for w in words)]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def inverse_cdf(u: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Piecewise-linear quantile function at u in [0, 1): float64, one ufunc per IEEE operation."""
    q = quantiles.astype(np.float64)
    pos = np.multiply(u, float(len(q) - 1))
    j = np.floor(pos)
    frac = np.subtract(pos, j)
    j = j.astype(np.int64)
    lo = q[j]
    step = np.subtract(q[j + 1], lo)
    return np.add(lo, np.multiply(frac, step))


def ternary_codes(rng: np.random.Generator, shape: tuple[int, int], counts: dict) -> np.ndarray:
    total = counts["minus_one"] + counts["zero"] + counts["plus_one"]
    first = counts["minus_one"] / total
    second = (counts["minus_one"] + counts["zero"]) / total
    u = rng.random(shape)
    return ((u >= first).astype(np.int8) + (u >= second).astype(np.int8) - np.int8(1)).astype(np.int8)


def _constant(name: str, stats: dict) -> np.ndarray:
    """The window or filterbank as weight_stats.json stores it (NVIDIA's buffer), checked by SHA-256."""
    entry = stats["constants"][name]
    values = decode_f32(entry["values"])
    if "nonzero_index" in entry:
        flat = np.zeros(int(np.prod(entry["shape"])), dtype=np.float32)
        flat[np.asarray(entry["nonzero_index"], dtype=np.int64)] = values
        values = flat
    value = values.reshape(entry["shape"])
    if sha256_bytes(value.astype("<f4").tobytes()) != entry["sha256"]:
        raise RuntimeError(f"{name}: decoded constant does not match its SHA-256")
    return value


def generate(stats: dict, seed: int, layers: int | None = None) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, array) in file order, one tensor at a time."""
    for name, dtype, shape in tensor_specs(stats, layers):
        if name.endswith((".codes", ".scale")) and name.rsplit(".", 1)[0] in stats["ternary"]:
            module, kind = name.rsplit(".", 1)
            entry = stats["ternary"][module]
            if kind == "codes":
                value = ternary_codes(_rng(seed, 0, module), shape, entry["codes"])
            else:
                u = _rng(seed, 1, module).random(shape[0])
                value = inverse_cdf(u, decode_f32(entry["scale"]["quantiles"])).astype(np.float32)
                if not (value > 0).all():
                    raise RuntimeError(f"{name}: non-positive scale")
        elif name in stats["float"]:
            entry = stats["float"][name]
            u = _rng(seed, 0, name).random(int(np.prod(shape, dtype=np.int64)))
            value = inverse_cdf(u, decode_f32(entry["quantiles"])).astype(np.float32).reshape(shape)
            for row in entry.get("zero_rows", []):
                value[row] = 0.0
            if (entry.get("positive") or name.endswith("running_var")) and not (value > 0).all():
                raise RuntimeError(f"{name}: expected positive values")
        elif name in stats["constants"]:
            value = _constant(name, stats)
        else:
            value = np.full(shape, stats["integer"][name]["value"], dtype=_ARRAY_DTYPES[dtype])
        assert value.dtype == _ARRAY_DTYPES[dtype] and value.shape == shape, name
        yield name, value if value.flags.c_contiguous else value.copy(order="C")


def _le_bytes(value: np.ndarray) -> bytes:
    return value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes()


def _manifest(stats: dict, seed: int, layers: int | None, tensors: dict[str, dict]) -> dict:
    lines = "".join(f"{n} {t['dtype']} {t['shape']} {t['sha256']}\n" for n, t in sorted(tensors.items()))
    return {"format": MODEL_FORMAT, "seed": seed, "layers": layers, "weight_stats_sha256": stats["_sha256"],
            "digest": sha256_bytes(lines.encode()), "tensors": tensors}


def manifest_only(stats: dict, seed: int, layers: int | None = None) -> dict:
    """The manifest (per-tensor hashes and digest) without writing anything."""
    tensors = {}
    for name, value in generate(stats, seed, layers):
        tensors[name] = {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": sha256_bytes(_le_bytes(value))}
    return _manifest(stats, seed, layers, tensors)


def write(stats: dict, seed: int, out_dir: Path, layers: int | None = None) -> dict:
    """Stream the model to out_dir/model.safetensors (temporary name, then renamed) and write manifest.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = tensor_specs(stats, layers)
    header, offset = {}, 0
    for name, dtype, shape in specs:
        size = int(np.prod(shape, dtype=np.int64)) * np.dtype(_ARRAY_DTYPES[dtype]).itemsize
        header[name] = {"dtype": _ST_DTYPES[np.dtype(_ARRAY_DTYPES[dtype])], "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    header["__metadata__"] = {"format": MODEL_FORMAT, "seed": str(seed), "layers": str(layers),
                              "weight_stats_sha256": stats["_sha256"]}
    text = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    tmp = out_dir / f"{MODEL_FILE}.tmp{os.getpid()}"
    tensors = {}
    with open(tmp, "wb") as handle:
        handle.write(len(text).to_bytes(8, "little"))
        handle.write(text)
        for (name, dtype, shape), (gen_name, value) in zip(specs, generate(stats, seed, layers)):
            assert name == gen_name
            data = _le_bytes(value)
            begin, end = header[name]["data_offsets"]
            if len(data) != end - begin:
                raise RuntimeError(f"{name}: {len(data)} bytes, header says {end - begin}")
            handle.write(data)
            tensors[name] = {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": sha256_bytes(data)}
    os.replace(tmp, out_dir / MODEL_FILE)
    manifest = _manifest(stats, seed, layers, tensors)
    manifest["file_sha256"] = sha256_file(out_dir / MODEL_FILE)
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (ru_maxrss is KB on Linux, bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stats", type=Path, default=STATS_FILE)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output directory (manifest only: manifest path)")
    parser.add_argument("--layers", type=int, default=None, help="only encoder layers < LAYERS (reduced depth)")
    parser.add_argument("--manifest-only", action="store_true", help="hash the tensors without writing the model")
    args = parser.parse_args()
    stats = load_stats(args.stats)
    if args.manifest_only:
        manifest = manifest_only(stats, args.seed, args.layers)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    else:
        manifest = write(stats, args.seed, args.out, args.layers)
    print(json.dumps({"seed": args.seed, "layers": args.layers, "tensors": len(manifest["tensors"]),
                      "digest": manifest["digest"], "file_sha256": manifest.get("file_sha256"),
                      "peak_rss_mb": round(peak_rss_mb(), 1)}))


if __name__ == "__main__":
    main()
