"""Shared helpers of the parakeet-binary-coreml tests: import paths and invented synthetic export fixtures.

Every fixture is invented data (random codes, scales and floats from a fixed seed); no model weights,
audio or transcripts. Run on the Mac with the ios pyenv (coremltools 9.0, numpy, safetensors, torch):
  ios/pyenv/.venv/bin/python -m unittest discover -s finetune/parakeet-binary-coreml/tests -v
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]          # finetune/parakeet-binary-coreml
IOS = HERE.parent / "parakeet-ternary" / "ios"
for _p in (str(IOS), str(IOS.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pack_binary_reference(codes: np.ndarray) -> np.ndarray:
    """The parakeet-binary-v1 layout written out bit by bit (independent of the reader's numpy unpacking):
    code j of a row is bit j % 8 of byte j // 8, LSB first; bit 1 = +1, bit 0 = -1; tail bits 0."""
    out, inp = codes.shape
    packed = np.zeros((out, math.ceil(inp / 8)), dtype=np.uint8)
    for r in range(out):
        for j in range(inp):
            if codes[r, j] == 1:
                packed[r, j // 8] |= np.uint8(1 << (j % 8))
    return packed


def pack_ternary_reference(codes: np.ndarray) -> np.ndarray:
    """parakeet-ternary-v1 (export.py) 2-bit layout: code j at bits 2 (j % 4) of byte j // 4; 01 = +1, 10 = -1."""
    out, inp = codes.shape
    packed = np.zeros((out, math.ceil(inp / 4)), dtype=np.uint8)
    for r in range(out):
        for j in range(inp):
            field = {0: 0, 1: 1, -1: 2}[int(codes[r, j])]
            packed[r, j // 4] |= np.uint8(field << (2 * (j % 4)))
    return packed


MODULES = {  # invented tiny modules: name -> (out, in); in not a multiple of 8 to exercise the tail bits
    "encoder.layers.0.feed_forward1.linear1": (16, 12),
    "encoder.layers.0.conv.pointwise_conv1": (8, 9),
}
FLOATS = {  # invented floating tensors: name -> (shape, stored dtype)
    "encoder.layers.0.norm_out.weight": ((12,), np.float16),
    "decoder.prediction.embed.weight": ((5, 4), np.float16),
    "preprocessor.featurizer.window": ((7,), np.float32),
}


def synthetic_codes(rng: np.random.Generator, out: int, inp: int) -> np.ndarray:
    return np.where(rng.random((out, inp)) < 0.5, -1, 1).astype(np.int8)


def write_binary_export(directory: Path, seed: int = 0, tokenizer: dict[str, bytes] | None = None,
                        corrupt=None) -> dict:
    """A tiny invented parakeet-binary-v1 export in `directory`. Returns {"codes", "scales", "floats", "sha256"}.
    corrupt(tensors) may alter the tensors before writing (for negative tests); the manifest is computed after."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    tensors, codes, scales, floats = {}, {}, {}, {}
    layers = {}
    for name, (out, inp) in MODULES.items():
        c = synthetic_codes(rng, out, inp)
        s = (rng.random(out) * 0.05 + 0.002).astype(np.float32)
        codes[name], scales[name] = c, s
        tensors[f"{name}.codes"] = pack_binary_reference(c)
        tensors[f"{name}.scale"] = s
        layers[name] = {"kind": "linear", "shape": [out, inp], "bias": False}
    for name, (shape, dtype) in FLOATS.items():
        v = rng.standard_normal(shape).astype(dtype)
        floats[name] = v
        tensors[name] = v
    if corrupt:
        corrupt(tensors)
    path = directory / "export.safetensors"
    save_file(tensors, str(path))
    digest = sha256_bytes(path.read_bytes())
    files = {"export.safetensors": digest}
    for rel, data in (tokenizer or {}).items():
        files[rel] = sha256_bytes(data)
    manifest = {"format": "parakeet-binary-v1", "file": "export.safetensors", "sha256": digest, "files": files,
                "bit_layout": "uint8 [out, ceil(in/8)]; code j of a row is bit j % 8 of byte j // 8 (least "
                              "significant bit first); bit 1 = +1, bit 0 = -1; unused tail bits 0",
                "quantized_layers": layers, "config": {"invented": True}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return {"codes": codes, "scales": scales, "floats": floats, "sha256": digest}


def write_ternary_export(directory: Path, codes: dict, scales: dict, floats: dict) -> str:
    """The same invented model as a parakeet-ternary-v1 export (2-bit codes), for the equivalence contract."""
    from safetensors.numpy import save_file

    directory.mkdir(parents=True, exist_ok=True)
    tensors, layers = {}, {}
    for name, c in codes.items():
        tensors[f"{name}.codes"] = pack_ternary_reference(c)
        tensors[f"{name}.scale"] = scales[name]
        layers[name] = {"kind": "linear", "shape": list(c.shape), "bias": False}
    tensors.update(floats)
    path = directory / "export.safetensors"
    save_file(tensors, str(path))
    digest = sha256_bytes(path.read_bytes())
    (directory / "manifest.json").write_text(json.dumps({"format": "parakeet-ternary-v1", "file": "export.safetensors",
                                                         "sha256": digest, "quantized_layers": layers,
                                                         "config": {"invented": True}}))
    return digest
