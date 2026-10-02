"""The benchmark models by name, as FP32 reference models. See DESIGN.md "Surrogate models, clips and traces".

- "b0": NVIDIA's pinned parakeet-tdt-0.6b-v2 (.nemo), dense FP32. Its model_weights.ckpt is
  extracted once to a cache file next to the artifacts and memory-mapped, so loading holds one
  dense copy plus file-backed pages.
- "mp2": M_P2, the primary benchmark model: the pilot P2 export (runs/pilot-P2-lr5e-4/export),
  read as packed codes + FP32 scales (reference.ExportSource), dequantized in place.
- "seed0", "seed1", "seed2": random surrogates from weight_stats.json (randomweights.py),
  generated in memory, or read from a written model.safetensors if a path is given.

Each function returns an eval-mode reference.ParakeetReference with a `provenance` dict
(file hashes or the surrogate's manifest digest). Default locations: NixOS
/mnt/hd/wilderness-labs-stt/..., Mac /Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/models/.
Full-size models need about 2.5 GB (b0 and mp2: 3-4 GB peak); run them in a memory-capped job.
"""
from __future__ import annotations

import json
import os
import sys
import tarfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import randomweights as rw  # noqa: E402
import reference  # noqa: E402

NAMES = ("b0", "mp2", "seed0", "seed1", "seed2")
_MAC = Path("/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios")
_LINUX = Path("/mnt/hd/wilderness-labs-stt")


def _defaults() -> dict[str, Path]:
    if sys.platform == "darwin":
        return {"nemo": _MAC / "models" / "parakeet-tdt-0.6b-v2.nemo", "mp2": _MAC / "models" / "pilot-P2-lr5e-4-export",
                "cache": _MAC / "cache"}
    if not os.path.ismount("/mnt/hd"):
        raise RuntimeError("/mnt/hd is not mounted")
    art = _LINUX / "parakeet-ternary"
    return {"nemo": art / "models" / "parakeet-tdt-0.6b-v2" / "parakeet-tdt-0.6b-v2.nemo",
            "mp2": art / "runs" / "pilot-P2-lr5e-4" / "export", "cache": _LINUX / "parakeet-ios" / "cache"}


def _config() -> reference.Config:
    return reference.Config.from_model_config(rw.load_stats()["model_config"])


def b0(nemo_file: str | Path | None = None) -> reference.ParakeetReference:
    """NVIDIA's original FP32 weights from the pinned .nemo."""
    defaults = _defaults()
    nemo_file = Path(nemo_file or defaults["nemo"])
    cache = defaults["cache"] / "b0" / "model_weights.ckpt"
    if not cache.exists() or cache.stat().st_mtime < nemo_file.stat().st_mtime:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + f".tmp{os.getpid()}")
        with tarfile.open(nemo_file, "r:*") as tar:
            member = next(m for m in tar.getmembers() if Path(m.name).name == "model_weights.ckpt")
            with tar.extractfile(member) as src, open(tmp, "wb") as dst:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)
        os.replace(tmp, cache)
    state = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    model = reference.build(_config())
    report = reference.load_weights(model, state)
    del state
    model.provenance = {"name": "b0", "nemo_file": str(nemo_file), "load": report}
    return model


def mp2(export_dir: str | Path | None = None) -> reference.ParakeetReference:
    """M_P2: the pilot P2 export, codes + FP32 scales dequantized in place (export file SHA-256 verified)."""
    source = reference.ExportSource(export_dir or _defaults()["mp2"])
    model = reference.build(reference.Config.from_model_config(source.model_config))
    report = reference.load_weights(model, source)
    model.provenance = {"name": "mp2", "export_dir": str(source.dir), "export_sha256": source.manifest["sha256"],
                        "load": report}
    return model


def surrogate(seed: int, path: str | Path | None = None) -> reference.ParakeetReference:
    """Random surrogate `seed`: generated in memory from weight_stats.json, or read from a written model.safetensors."""
    stats = rw.load_stats()
    model = reference.build(reference.Config.from_model_config(stats["model_config"]))
    if path is not None:
        manifest = json.loads((Path(path).parent / rw.MANIFEST_FILE).read_text())
        if manifest["seed"] != seed or manifest["weight_stats_sha256"] != stats["_sha256"]:
            raise ValueError(f"{path} is not surrogate seed {seed} of this weight_stats.json")
        report, digest = reference.load_weights(model, path), manifest["digest"]
    else:
        tensors, hashes = {}, {}
        for name, value in rw.generate(stats, seed):
            hashes[name] = {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": rw.sha256_bytes(value.tobytes())}
            tensors[name] = torch.from_numpy(value)
        report, digest = reference.load_weights(model, tensors), rw._manifest(stats, seed, None, hashes)["digest"]
    model.provenance = {"name": f"seed{seed}", "seed": seed, "digest": digest,
                        "weight_stats_sha256": stats["_sha256"], "load": report}
    return model


def seed0(path: str | Path | None = None) -> reference.ParakeetReference:
    return surrogate(0, path)


def seed1(path: str | Path | None = None) -> reference.ParakeetReference:
    return surrogate(1, path)


def seed2(path: str | Path | None = None) -> reference.ParakeetReference:
    return surrogate(2, path)


def load(name: str, path: str | Path | None = None) -> reference.ParakeetReference:
    """The reference model for a benchmark model name in NAMES."""
    if name not in NAMES:
        raise KeyError(f"unknown model {name!r}; known: {NAMES}")
    return globals()[name](path)
