"""G0 feasibility probe: can C0's compressed encoder tensors be taken out of Encoder.mlmodelc? (report only)

DESIGN.md "Arms" G0: our plain graph with C0's own 6-bit LUTs and indices, read from C0's compiled weight
file. This probe parses Encoder.mlmodelc/model.mil (text MIL) for every constexpr_lut_to_dense op, reads the
referenced blobs from weights/weight.bin with an independent parser of the MIL blob storage format
(coremltools mlmodel/src/MILBlob/Blob/StorageFormat.hpp: a 64-byte storage_header {uint32 count, uint32
version, 7 x uint64 reserved}; per blob a 64-byte blob_metadata {uint32 sentinel 0xDEADBEEF, uint32
mil_dtype, uint64 sizeInBytes, uint64 offset, uint64 padding_size_in_bits, 4 x uint64 reserved} at the
offset the MIL names, data at metadata.offset), unpacks the indices (iOS16 constexpr_lut_to_dense: uint8
bytes holding nbits = log2(len(lut)) bit fields, least significant bit first) and checks both against
coremltools 9.0's own reader (libmilstoragepython._BlobStorageReader) and decompression
(iOS16 constexpr_lut_to_dense.decompress). It records, per tensor, the shapes, offsets and fingerprints
(values at 64 seeded flat positions) that `compare` checks against the B0 weights on NixOS.

  Mac:   macguard --rss-cap 4G --timeout 900 -- ios/pyenv/.venv/bin/python ios/g0probe.py extract \
             --model <artifacts>/c0/Encoder.mlmodelc --out <artifacts>/results/g0probe.json
  NixOS: ../heavy ios-wp2-g0 --mem-max 4G --runtime 10min --wait -- env CUDA_VISIBLE_DEVICES= \
             <repo>/finetune/parakeet-ternary/python <repo>/finetune/parakeet-ternary/ios/g0probe.py compare \
             --probe <copy of g0probe.json>
The probe's JSON holds weight excerpts (fingerprints); it stays in the artifact directories, not in Git.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

SENTINEL = 0xDEADBEEF
MIL_DTYPES = {1: "fp16", 2: "fp32", 3: "uint8", 4: "int8", 5: "bf16", 6: "int16", 7: "uint16", 8: "int4", 9: "uint1",
              10: "uint2", 11: "uint4", 12: "uint3", 13: "uint6", 14: "int32", 15: "uint32"}
BLOB = r'BLOBFILE\(path = tensor<string, \[\]>\("@model_path/weights/weight.bin"\), offset = tensor<uint64, \[\]>\((\d+)\)\)'
NAME = r'tensor<string, \[\]>\("[^"]*"\)'
LUT_OP = re.compile(
    r"tensor<(\w+), \[([\d, ]*)\]> (\S+) = constexpr_lut_to_dense\(\)\[indices = tensor<(\w+), \[(\d+)\]>\(" + BLOB +
    r"\), lut = tensor<(\w+), \[([\d, ]*)\]>\(" + BLOB + r"\), name = " + NAME + r", shape = tensor<uint32, \[\d+\]>\(\[([\d, ]*)\]\)\]")
DENSE_CONST = re.compile(r"tensor<(\w+), \[([\d, ]*)\]> (\S+) = const\(\)\[name = " + NAME + r", val = tensor<\w+, \[[\d, ]*\]>\(" + BLOB)
FINGERPRINT = 64
FIRST_TENSORS = 6  # tensors whose full decompression is cross-checked against coremltools


def dims(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


class Blobs:
    """Independent reader of a MIL weight.bin."""

    def __init__(self, path: Path) -> None:
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.count, self.version = struct.unpack_from("<II", self.data, 0)

    def meta(self, offset: int) -> dict:
        sentinel, dtype, size, data_offset, padding = struct.unpack_from("<IIQQQ", self.data, offset)
        if sentinel != SENTINEL:
            raise ValueError(f"no blob sentinel at {offset}")
        return {"mil_dtype": MIL_DTYPES.get(dtype, dtype), "bytes": size, "data_offset": data_offset,
                "padding_bits": padding}

    def raw(self, offset: int) -> tuple[np.ndarray, dict]:
        m = self.meta(offset)
        return self.data[m["data_offset"]:m["data_offset"] + m["bytes"]], m


def unpack_lsb(packed: np.ndarray, nbits: int, count: int) -> np.ndarray:
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")[:count * nbits]
    return (bits.reshape(count, nbits).astype(np.uint32) << np.arange(nbits, dtype=np.uint32)).sum(axis=1)


def fingerprint_positions(size: int) -> np.ndarray:
    return np.random.default_rng(0).integers(0, size, FINGERPRINT)


def cmd_extract(args) -> None:
    from coremltools.converters.mil.mil.ops.defs.iOS16.constexpr_ops import constexpr_lut_to_dense
    from coremltools.libmilstoragepython import _BlobStorageReader
    import coremltools

    t0 = time.time()
    model = Path(args.model)
    mil = (model / "model.mil").read_text()
    blobs = Blobs(model / "weights" / "weight.bin")
    reader = _BlobStorageReader(str(model / "weights" / "weight.bin"))
    ops = [m.groups() for m in LUT_OP.finditer(mil)]
    n_text = mil.count("constexpr_lut_to_dense(")
    dense = [m.groups() for m in DENSE_CONST.finditer(mil)]
    tensors, checked, mismatches = [], 0, []
    index_bytes = lut_bytes = 0
    for k, (dt, oshape, name, idt, ilen, ioff, ldt, lshape, loff, shape) in enumerate(ops):
        packed, imeta = blobs.raw(int(ioff))
        lut_raw, lmeta = blobs.raw(int(loff))
        lut = np.frombuffer(lut_raw.tobytes(), dtype=np.float16)
        nbits = int(np.log2(lut.size))
        size = int(np.prod(dims(shape)))
        idx = unpack_lsb(packed, nbits, size)
        values = lut[idx].reshape(dims(shape))
        index_bytes += imeta["bytes"]; lut_bytes += lmeta["bytes"]
        ok_reader = (np.array_equal(np.asarray(reader.read_uint8_data(int(ioff))), np.asarray(packed))
                     and np.array_equal(np.asarray(reader.read_fp16_data(int(loff))).view(np.uint16), lut.view(np.uint16)))
        if not ok_reader:
            mismatches.append(name)
        if k < FIRST_TENSORS or name.endswith("feed_forward1_linear1_weight_to_fp16_palettized"):
            ref = constexpr_lut_to_dense.decompress(lut, np.asarray(packed, dtype=np.uint8), np.array(dims(shape), np.uint32))
            checked += 1
            if not np.array_equal(np.asarray(ref).view(np.uint16), values.view(np.uint16)):
                mismatches.append(name + " (decompress)")
        pos = fingerprint_positions(size)
        tensors.append({
            "name": name, "shape": dims(shape), "dtype": dt, "indices": {"dtype": idt, "bytes": imeta["bytes"],
            "offset": int(ioff), "data_offset": imeta["data_offset"], "mil_dtype": imeta["mil_dtype"]},
            "lut": {"entries": int(lut.size), "nbits": nbits, "offset": int(loff), "data_offset": lmeta["data_offset"],
                    "mil_dtype": lmeta["mil_dtype"]},
            "codes_used": int(np.unique(idx).size), "rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
            "fingerprint_positions": pos.tolist(), "fingerprint": values.reshape(-1)[pos].astype(float).tolist()})
    groups: dict[str, int] = {}
    for t in tensors:
        key = re.sub(r"layers_\d+_", "layers_N_", t["name"])
        key = re.sub(r"^(op|const)_\d+_", r"\1_K_", key)
        groups[key] = groups.get(key, 0) + 1
    result = {
        "model": str(model), "coremltools": coremltools.__version__,
        "weight_bin": {"bytes": int(blobs.data.size), "header_count": blobs.count, "header_version": blobs.version},
        "lut_ops_in_text": n_text, "lut_ops_parsed": len(ops), "dense_blob_consts": len(dense),
        "lut_bytes": lut_bytes, "index_bytes": index_bytes,
        "lut_entries": sorted({t["lut"]["entries"] for t in tensors}),
        "index_storage": sorted({(t["indices"]["dtype"], t["indices"]["mil_dtype"]) for t in tensors}),
        "groups": groups, "decompress_cross_checked": checked, "mismatches": mismatches,
        "example_mil_line": next(line.strip() for line in mil.splitlines() if "constexpr_lut_to_dense" in line)[:600],
        "seconds": round(time.time() - t0, 1), "tensors": tensors,
    }
    Path(args.out).write_text(json.dumps(result) + "\n")
    summary = {k: v for k, v in result.items() if k != "tensors"}
    print(json.dumps(summary, indent=1))


# --- NixOS side: fingerprints vs the B0 weights -------------------------------------------------------------

def b0_tensor(state: dict, name: str, layer_of_pos: dict[str, int]) -> np.ndarray | None:
    """The B0 tensor (FP32, C0's layout) that C0's palettized tensor `name` encodes, or None if unknown."""
    import torch

    m = re.fullmatch(r"module_(.+)_weight_to_fp16_palettized", name)
    if m:
        key = m.group(1)
        key = re.sub(r"^layers_(\d+)_", r"layers.\1.", key)
        key = key.replace("pre_encode_conv_", "pre_encode.conv.").replace("pre_encode_out", "pre_encode.out")
        for part in ("feed_forward1_", "feed_forward2_", "self_attn_", "conv_pointwise_"):
            key = key.replace(part, part[:-1] + ".") if part != "conv_pointwise_" else key.replace(part, "conv.pointwise_")
        w = state[f"encoder.{key}.weight"].float()
        return w.numpy()
    if name in layer_of_pos:  # folded linear_pos(pos_emb) for the 188-frame window: [1, 8, 128, 375]
        import reference

        layer = layer_of_pos[name]
        w = state[f"encoder.layers.{layer}.self_attn.linear_pos.weight"].float()
        pos = reference.rel_positional_embedding(188, 1024)  # [1, 375, 1024]
        p = (pos @ w.T).view(1, 375, 8, 128).permute(0, 2, 3, 1)
        return p.numpy()
    return None


def cmd_compare(args) -> None:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    probe = json.loads(Path(args.probe).read_text())
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True, mmap=True)
    pos_tables = [t["name"] for t in probe["tensors"] if t["shape"] == [1, 8, 128, 375]]
    layer_of_pos = {name: i for i, name in enumerate(pos_tables)}  # MIL order = layer order
    depthwise = [t["name"] for t in probe["tensors"] if t["shape"] == [1024, 1, 9]]
    rows, unmatched = [], []
    for t in probe["tensors"]:
        ref = b0_tensor(state, t["name"], layer_of_pos)
        if ref is None and t["name"] in depthwise:  # depthwise conv with BatchNorm folded in
            layer = depthwise.index(t["name"])
            pre = f"encoder.layers.{layer}.conv."
            w = state[pre + "depthwise_conv.weight"].float()
            g = state[pre + "batch_norm.weight"].float() / torch.sqrt(state[pre + "batch_norm.running_var"].float() + 1e-5)
            ref = (w * g[:, None, None]).numpy()
        if ref is None or list(ref.shape) != t["shape"]:
            unmatched.append(t["name"])
            continue
        r = ref.reshape(-1)[np.asarray(t["fingerprint_positions"])].astype(np.float64)
        c = np.asarray(t["fingerprint"], dtype=np.float64)
        rows.append({"name": t["name"], "rel_err": float(np.linalg.norm(c - r) / max(np.linalg.norm(r), 1e-12)),
                     "rms_ratio": t["rms"] / float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))})
    errs = np.array([r["rel_err"] for r in rows])
    summary = {"tensors": len(probe["tensors"]), "matched": len(rows), "unmatched": unmatched,
               "fingerprint_rel_err": {"median": float(np.median(errs)), "max": float(errs.max()),
                                       "p90": float(np.quantile(errs, 0.9))},
               "worst": sorted(rows, key=lambda r: -r["rel_err"])[:5],
               "rms_ratio_range": [float(min(r["rms_ratio"] for r in rows)), float(max(r["rms_ratio"] for r in rows))]}
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract"); p.add_argument("--model", required=True); p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_extract)
    p = sub.add_parser("compare"); p.add_argument("--probe", required=True)
    p.add_argument("--ckpt", default="/mnt/hd/wilderness-labs-stt/parakeet-ios/cache/b0/model_weights.ckpt")
    p.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
