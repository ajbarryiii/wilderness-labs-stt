"""Tensor sources for the MIL builders: benchmark models as codes + FP32 scales, and C0's own tensors for G0.

Source(name) reads one tensor at a time (numpy), with NeMo state_dict names:
- "mp2": the pilot P2 export (reference.ExportSource; packed 2-bit codes unpacked to int8, FP32 row
  scales, FP16 floating tensors returned as FP32, exactly). The export file's SHA-256 is verified.
- "seed0".."seed2": a written surrogate model.safetensors (int8 codes, FP32 scales and floats), verified
  against its manifest (models._verify_surrogate_file).
ternary(module) -> (codes int8 [out, in], scale float32 [out]); floating(key) -> float32 array.

C0Tensors(path) parses C0's Encoder.mlmodelc (model.mil text + weights/weight.bin, g0probe's parser) into
per-role entries for G0: ("lut", packed uint8 indices, fp16[64] LUT, shape) for the 294 palettized
tensors (iOS16 constexpr_lut_to_dense layout, used verbatim) and ("dense", fp16 array) for the 320 dense
constants. Roles use our names: "pre_encode.conv.0.weight", "layers.3.self_attn.linear_q.weight",
"layers.3.pos_table" ([1, 8, 128, 375], C0's folded linear_pos for 188 frames), "layers.3.depthwise.weight"
/ ".bias" (BatchNorm folded by C0), "layers.3.norm_conv.weight", ... (see C0Tensors.ROLE_RE).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import IOS  # noqa: F401  (sys.path setup)

MAC_ART = Path("/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios")
LINUX_ART = Path("/mnt/hd/wilderness-labs-stt")
N_LAYERS = 24
TERNARY_SUFFIXES = ("feed_forward1.linear1", "feed_forward1.linear2", "self_attn.linear_q", "self_attn.linear_k",
                    "self_attn.linear_v", "self_attn.linear_out", "self_attn.linear_pos", "conv.pointwise_conv1",
                    "conv.pointwise_conv2", "feed_forward2.linear1", "feed_forward2.linear2")


def default_path(name: str) -> Path:
    mac = sys.platform == "darwin"
    if name == "mp2":
        return (MAC_ART / "models" / "pilot-P2-lr5e-4-export" if mac
                else LINUX_ART / "parakeet-ternary" / "runs" / "pilot-P2-lr5e-4" / "export")
    if name.startswith("seed"):
        return (MAC_ART if mac else LINUX_ART / "parakeet-ios") / "random" / name / "model.safetensors"
    if name == "c0":
        return (MAC_ART if mac else LINUX_ART / "parakeet-ios") / "c0" / "Encoder.mlmodelc"
    raise KeyError(name)


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class Source:
    """One benchmark model's tensors (NeMo names), read lazily."""

    def __init__(self, name: str, path: str | Path | None = None) -> None:
        import reference

        self.name = name
        path = Path(path or default_path(name))
        if name == "mp2":
            src = reference.ExportSource(path)
            self._get = lambda k: src[k].numpy()
            self._keys = set(src)
            self.provenance = {"name": name, "export_dir": str(path), "export_sha256": src.manifest["sha256"]}
        elif name.startswith("seed"):
            import models
            from safetensors import safe_open

            manifest = json.loads((path.parent / "manifest.json").read_text())
            if f"seed{manifest['seed']}" != name:
                raise ValueError(f"{path} is surrogate seed {manifest['seed']}, not {name}")
            models._verify_surrogate_file(path, manifest)
            handle = safe_open(str(path), framework="numpy")
            self._get = handle.get_tensor
            self._keys = set(handle.keys())
            self.provenance = {"name": name, "file": str(path), "digest": manifest["digest"],
                               "file_sha256": manifest["file_sha256"]}
        else:
            raise KeyError(f"unknown model {name!r}")

    def floating(self, key: str) -> np.ndarray:
        value = self._get(key)
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{key} is {value.dtype}, expected floating")
        return np.ascontiguousarray(value, dtype=np.float32)

    def ternary(self, module: str) -> tuple[np.ndarray, np.ndarray]:
        codes, scale = self._get(f"{module}.codes"), self._get(f"{module}.scale")
        if codes.dtype != np.int8 or codes.ndim != 2 or scale.dtype != np.float32 or scale.shape != (codes.shape[0],):
            raise ValueError(f"{module}: codes {codes.dtype} {codes.shape}, scale {scale.dtype} {scale.shape}")
        if codes.min() < -1 or codes.max() > 1 or not (scale > 0).all():
            raise ValueError(f"{module}: codes outside {{-1, 0, 1}} or non-positive scales")
        return np.ascontiguousarray(codes), np.ascontiguousarray(scale)

    def has(self, key: str) -> bool:
        return key in self._keys


def fp16_scale(scale: np.ndarray) -> np.ndarray:
    """FP32 row scales rounded to FP16 (round to nearest even); refuses overflow, underflow to 0 or subnormals."""
    s16 = scale.astype(np.float16)
    if not np.isfinite(s16).all() or (s16 <= 0).any() or (np.abs(s16) < np.finfo(np.float16).tiny).any():
        raise ValueError("a row scale is not a normal positive FP16 number")
    return s16


def effective_fp16(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """codes x FP16(s) as FP16 [out, in] (exact: +-FP16(s) or 0)."""
    s16 = fp16_scale(scale)
    out = codes.astype(np.float16) * s16[:, None]
    return out


# --- C0 (G0) ------------------------------------------------------------------------------------------

class C0Tensors:
    """C0's Encoder.mlmodelc tensors by role (see module docstring); parsing as g0probe.py."""

    def __init__(self, model: str | Path | None = None) -> None:
        import g0probe

        self.model = Path(model or default_path("c0"))
        mil = (self.model / "model.mil").read_text()
        self.blobs = g0probe.Blobs(self.model / "weights" / "weight.bin")
        self.entries: dict[str, tuple] = {}
        pos_tables, dw_weights, dw_biases = [], [], []
        for (dt, oshape, name, idt, ilen, ioff, ldt, lshape, loff, shape) in (m.groups() for m in g0probe.LUT_OP.finditer(mil)):
            packed, imeta = self.blobs.raw(int(ioff))
            lut_raw, lmeta = self.blobs.raw(int(loff))
            if dt != "fp16" or ldt != "fp16" or idt != "uint8" or imeta["mil_dtype"] != "uint8" or lmeta["mil_dtype"] != "fp16":
                raise ValueError(f"{name}: unexpected dtypes")
            entry = ("lut", np.array(packed, dtype=np.uint8), np.frombuffer(lut_raw.tobytes(), dtype=np.float16).copy(),
                     np.array(g0probe.dims(shape), dtype=np.uint32), {"index_offset": int(ioff), "lut_offset": int(loff)})
            role = self._role(name)
            if role == "pos_table":
                pos_tables.append(entry)
            elif role == "dw_weight":
                dw_weights.append(entry)
            else:
                self.entries[role] = entry
        dense_re = re.compile(r"tensor<(\w+), \[([\d, ]*)\]> (\S+) = const\(\)\[name = " + g0probe.NAME
                              + r", val = tensor<\w+, \[[\d, ]*\]>\(" + g0probe.BLOB)
        for dt, shape, name, off in (m.groups() for m in dense_re.finditer(mil)):
            raw, meta = self.blobs.raw(int(off))
            if dt != "fp16" or meta["mil_dtype"] != "fp16":
                raise ValueError(f"{name}: unexpected dtype {dt}/{meta['mil_dtype']}")
            value = np.frombuffer(raw.tobytes(), dtype=np.float16).reshape(g0probe.dims(shape)).copy()
            role = self._role(name)
            if role == "dw_bias":
                dw_biases.append(("dense", value, {"offset": int(off)}))
            elif role == "zero_bias":
                if value.any():
                    raise ValueError(f"{name}: expected an all-zero bias")
            else:
                self.entries[role] = ("dense", value, {"offset": int(off)})
        if not (len(pos_tables) == len(dw_weights) == len(dw_biases) == N_LAYERS):
            raise ValueError(f"expected {N_LAYERS} position tables / depthwise weights and biases, got "
                             f"{len(pos_tables)} / {len(dw_weights)} / {len(dw_biases)}")
        for i in range(N_LAYERS):  # MIL order = layer order
            self.entries[f"layers.{i}.pos_table"] = pos_tables[i]
            self.entries[f"layers.{i}.depthwise.weight"] = dw_weights[i]
            self.entries[f"layers.{i}.depthwise.bias"] = dw_biases[i]
        self.provenance = {"name": "c0-encoder", "model": str(self.model),
                           "weight_bin_sha256": sha256_file(self.model / "weights" / "weight.bin"),
                           "model_mil_sha256": sha256_file(self.model / "model.mil"),
                           "lut_tensors": sum(1 for e in self.entries.values() if e[0] == "lut"),
                           "dense_tensors": sum(1 for e in self.entries.values() if e[0] == "dense")}

    @staticmethod
    def _role(name: str) -> str:
        if re.fullmatch(r"op_\d+_to_fp16_palettized", name):
            return "pos_table"
        if re.fullmatch(r"const_\d+_to_fp16_palettized", name):
            return "dw_weight"
        if re.fullmatch(r"const_\d+_to_fp16", name):
            return "dw_bias"
        if re.fullmatch(r"linear_\d+_bias_\d+_to_fp16", name):
            return "zero_bias"
        m = re.fullmatch(r"module_(.+?)(_to_fp16_palettized|_to_fp16)", name)
        if not m:
            raise ValueError(f"unknown C0 tensor {name}")
        key = m.group(1)
        key = re.sub(r"^pre_encode_conv_(\d+)_", r"pre_encode.conv.\1.", key)
        key = key.replace("pre_encode_out_", "pre_encode.out.")
        key = re.sub(r"^layers_(\d+)_", r"layers.\1.", key)
        for part in ("feed_forward1", "feed_forward2", "self_attn", "norm_feed_forward1", "norm_feed_forward2",
                     "norm_self_att", "norm_conv", "norm_out"):
            key = key.replace(part + "_", part + ".")
        key = key.replace("conv_pointwise_conv", "conv.pointwise_conv")
        key = re.sub(r"\.(linear\d|linear_[a-z]+|pointwise_conv\d)_weight$", r".\1.weight", key)
        key = re.sub(r"_(weight|bias)$", r".\1", key)
        return key

    def __getitem__(self, role: str) -> tuple:
        return self.entries[role]
