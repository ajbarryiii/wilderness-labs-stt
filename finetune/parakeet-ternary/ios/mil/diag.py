"""Diagnostic: the encoder graph in FP32 and FP16 against the FP32 reference at reduced depth (not a gate).

  cd ios && python -m mil.diag depth [--layers 1,2,4,8,12] [--ids ...] [--arm C4]

For each depth N, builds the reference truncated to N layers (M_P2, FP16-rounded row scales, as gates 4-5)
and runs it on the clips (refcache features), frees it, then builds our plain graph with N layers twice,
compiled for the Mac:
- "F32": the same graph authored in FP32 with dense FP32 weights codes x FP16(s), compute precision
  FLOAT32, CPU_ONLY. It isolates graph semantics (masking, rel-pos, folding) from FP16 arithmetic: it
  should match the reference to FP32 rounding.
- the arm (default C4, exact weights) in FP16 with CPU_ONLY and CPU_AND_NE.
Reports rel/abs (DESIGN.md definitions) of the valid frames per clip and depth, the per-layer growth of
the FP16 error, and the activation range of the reference (max |x| of the residual stream after each
layer's FF1 and of the FF linear2 outputs) -> ios/results/diag/fp16_depth.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mil"

import numpy as np

from . import IOS, refcache  # noqa: E402

DIAG = IOS / "results" / "diag"
DEFAULT_IDS = "n02-2428-83699-0004,n15-1673-143396-0001,b04-N32160,silence-3s"


def reference_outputs(n_layers: int, ids: list[str]) -> tuple[dict, dict]:
    import torch

    import reference

    from .weights import Source

    torch.set_grad_enabled(False)
    source = Source("mp2")
    export = reference.ExportSource(source.provenance["export_dir"])
    model = reference.build(reference.Config.from_model_config(export.model_config, num_layers=n_layers))
    reference.load_weights(model, export)
    refcache.set_scales(model, source, "fp16s")
    ranges: dict[str, float] = {}
    hooks = []
    for i, layer in enumerate(model.encoder.layers):
        for name in ("feed_forward1.linear2", "feed_forward2.linear2", "norm_out"):
            def hook(mod, inp, out, key=f"layers.{i}.{name}"):
                ranges[key] = max(ranges.get(key, 0.0), float(out.abs().max()))
            hooks.append(layer.get_submodule(name).register_forward_hook(hook))
    outs = {}
    for cid in ids:
        ref = refcache.load_clip("mp2", cid)
        enc, length = model.encoder(torch.from_numpy(ref["features"])[None], torch.tensor([int(ref["mel_length"])]))
        outs[cid] = enc[0, :, :int(length[0])].numpy()
    for h in hooks:
        h.remove()
    del model
    return outs, ranges


def depth(args) -> None:
    import coremltools as ct

    from .build import build_encoder

    import artifacts

    ids = args.ids.split(",")
    out_root = artifacts.check(artifacts.root() / "diag")
    doc = {"description": __doc__.split("\n\n")[1], "clips": ids, "arm": args.arm, "depths": {}}
    path = DIAG / "fp16_depth.json"
    DIAG.mkdir(parents=True, exist_ok=True)
    for n in (int(x) for x in args.layers.split(",")):
        t0 = time.time()
        refs, ranges = reference_outputs(n, ids)
        entry = {"reference_max_abs": {k: round(v, 2) for k, v in ranges.items() if k.startswith(f"layers.{n - 1}.")
                                       or k.startswith("layers.0.")}, "runs": {}}
        for arm, units in (("F32", ("cpuOnly",)), (args.arm, ("cpuOnly", "cpuAndNeuralEngine"))):
            m = build_encoder("mp2", arm, "fixed", out_root, layers=n, plan=False, keep_package=False, tag=f"-{n}L")
            for u in units:
                cu = {"cpuOnly": ct.ComputeUnit.CPU_ONLY, "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE}[u]
                model = ct.models.CompiledMLModel(m["paths"]["mlmodelc"], compute_units=cu)
                rows = {}
                for cid in ids:
                    ref = refcache.load_clip("mp2", cid)
                    mel = np.zeros((1, 128, 1501), dtype=np.float32)
                    mel[0, :, :ref["features"].shape[1]] = ref["features"]
                    pred = model.predict({"mel": mel, "mel_length": np.array([ref["mel_length"]], dtype=np.int32)})
                    e = refs[cid].shape[1]
                    rel, ab = refcache.errors(np.asarray(pred["encoder"])[0, :, :e], refs[cid], 1e-3 if arm != "F32" else 1e-6)
                    rows[cid] = {"rel": rel, "abs": ab}
                del model
                entry["runs"][f"{arm}/{u}"] = rows
                print(n, arm, u, {k: (round(v["rel"], 6), round(v["abs"], 4)) for k, v in rows.items()}, flush=True)
            shutil.rmtree(Path(m["paths"]["mlmodelc"]), ignore_errors=True)
        entry["seconds"] = round(time.time() - t0, 1)
        doc["depths"][str(n)] = entry
        path.write_text(json.dumps(doc, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("depth")
    p.add_argument("--layers", default="1,2,4,8,12")
    p.add_argument("--ids", default=DEFAULT_IDS)
    p.add_argument("--arm", default="C4")
    args = parser.parse_args()
    depth(args)


if __name__ == "__main__":
    main()
