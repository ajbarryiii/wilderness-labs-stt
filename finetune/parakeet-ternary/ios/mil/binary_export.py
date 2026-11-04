"""Native reader of "parakeet-binary-v1" exports (1-bit encoder projections, e.g. MB1 / kea-b).

Layout (the export manifest's "bit_layout"): "<module>.codes" uint8 [out, ceil(in / 8)], code j of a row is
bit j % 8 of byte j // 8 (least significant bit first), bit 1 = +1, bit 0 = -1, unused tail bits 0;
"<module>.scale" FP32 [out]; floating tensors FP16 (feature constants FP32).

BinaryExportSource(dir) verifies the format and the export file's SHA-256 (manifest "sha256" and, when
present, "files"["export.safetensors"]) and reads one tensor at a time with numpy (no torch):
- source["<module>.codes"] / binary(module): int8 [out, in] in {-1, +1}; source["<module>.scale"]: FP32;
- other floating tensors widened to FP32 (exact); raw(key) keeps the stored dtype (native blobs);
- verify_tokenizer(dir) checks tokenizer files against the manifest's "files" hashes.
weights.Source("mb1") wraps it with the (codes, scale) interface the encoder build uses, so builds need no
2-bit re-encoding.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

FORMAT = "parakeet-binary-v1"


def unpack_binary_codes(packed: np.ndarray, in_features: int) -> np.ndarray:
    """uint8 [out, ceil(in / 8)] -> int8 [out, in] in {-1, +1}; ValueError on a wrong dtype/width or set tail bits."""
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] != math.ceil(in_features / 8):
        raise ValueError(f"expected uint8 [out, {math.ceil(in_features / 8)}], got {packed.dtype} {packed.shape}")
    bits = np.unpackbits(packed, axis=1, bitorder="little")
    if bits[:, in_features:].any():
        raise ValueError("nonzero padding bits after the last code of a row")
    return (bits[:, :in_features].astype(np.int8) * 2 - 1).astype(np.int8)


def _sha256(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class BinaryExportSource(Mapping):
    """A parakeet-binary-v1 export directory as a lazy tensor mapping (NeMo state_dict names)."""

    FORMAT = FORMAT

    def __init__(self, export_dir: str | Path, verify: bool = True) -> None:
        from safetensors import safe_open

        self.dir = Path(export_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"unknown export format {self.manifest.get('format')!r}, expected {FORMAT}")
        path = self.dir / self.manifest["file"]
        listed = self.manifest.get("files", {}).get(self.manifest["file"])
        if listed is not None and listed != self.manifest["sha256"]:
            raise ValueError("manifest 'sha256' and 'files' disagree on the export file")
        if verify and _sha256(path) != self.manifest["sha256"]:
            raise ValueError(f"{path} does not match its manifest SHA-256")
        self._file = safe_open(str(path), framework="numpy")
        self._layers = self.manifest["quantized_layers"]
        if any(layer.get("bias") for layer in self._layers.values()):
            raise ValueError("binary module biases are not implemented")
        self._keys = set(self._file.keys())

    @property
    def model_config(self) -> dict:
        return self.manifest["config"]

    @property
    def modules(self) -> list[str]:
        return sorted(self._layers)

    def raw(self, key: str) -> np.ndarray:
        """The tensor as stored (packed codes stay packed, FP16 stays FP16)."""
        return self._file.get_tensor(key)

    def binary(self, module: str) -> tuple[np.ndarray, np.ndarray]:
        """(codes int8 [out, in] in {-1, +1}, scale FP32 [out]) of one quantized module."""
        out, inp = self._layers[module]["shape"]
        codes = unpack_binary_codes(self.raw(f"{module}.codes"), inp)
        scale = self.raw(f"{module}.scale")
        if codes.shape != (out, inp) or scale.dtype != np.float32 or scale.shape != (out,):
            raise ValueError(f"{module}: codes {codes.shape}, scale {scale.dtype} {scale.shape}, manifest {[out, inp]}")
        return codes, scale

    def __getitem__(self, key: str) -> np.ndarray:
        if key.endswith(".codes") and key[: -len(".codes")] in self._layers:
            return self.binary(key[: -len(".codes")])[0]
        value = self.raw(key)
        return value.astype(np.float32) if np.issubdtype(value.dtype, np.floating) else value

    def __iter__(self):
        return iter(sorted(self._keys))

    def __len__(self) -> int:
        return len(self._keys)

    def verify_tokenizer(self, tokenizer_dir: str | Path) -> dict[str, str]:
        """Check every tokenizer/<name> file the manifest lists against tokenizer_dir/<name>; returns rel -> sha."""
        tokenizer_dir = Path(tokenizer_dir)
        listed = {rel: sha for rel, sha in self.manifest.get("files", {}).items() if rel.startswith("tokenizer/")}
        if not listed:
            raise ValueError("the export manifest lists no tokenizer files")
        for rel, sha in listed.items():
            path = tokenizer_dir / Path(rel).name
            if not path.is_file() or _sha256(path) != sha:
                raise ValueError(f"{path} does not match the export manifest's {rel} SHA-256")
        return dict(listed)
