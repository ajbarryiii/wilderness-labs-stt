"""Build, convert, compile and describe the Core ML models of one arm (DESIGN.md S1; see README "WP3").

  cd ios && python -m mil.build encoder --model mp2 --arm C4 --variant fixed [--layers N] [--out-dir DIR]
  cd ios && python -m mil.build decoder --model mp2 [--out-dir DIR]
  cd ios && python -m mil.build plan --path X.mlmodelc [--units cpuAndNeuralEngine]

Encoder variants: fixed (single function, mel [1, 128, 1501]), multi (functions b2, b4, b8, b15 sharing
weights, default b15), enum (one function, EnumeratedShapes over the four mel lengths, default 1501). Arms:
encodings.ENCODER_ARMS and G0 (C0's tensors, iOS17 like C0, fixed only). Output (artifact area only):
DIR/<model>/<arm>/<variant>.mlpackage, .mlmodelc (on macOS), manifest-<variant>.json; a copy of the
manifest goes to ios/results/builds/<model>-<arm>-<variant>.json (text, no weights).

Conversion: the MIL program is built in FP16 (fp32/int32 I/O), converted with coremltools' default pass
pipeline (the pass list is recorded; C7 uses the same list unless probes.py found a folding pass) and
compute_precision FLOAT16. Multifunction models run the frontend + main + backend pipelines per function,
then coremltools' cross-function constant deduplication (weight_id sharing, as save_multifunction does),
then export without further passes.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mil"

from . import IOS, encoder as enc_mod, encodings  # noqa: E402

RESULTS = IOS / "results" / "builds"
VARIANTS = ("fixed", "multi", "enum")
C7_EXCLUDED_PASSES_FILE = IOS / "results" / "c7_excluded_passes.json"
DECODER_OPSET = "iOS26"


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


def dir_bytes(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def target(opset: str):
    import coremltools as ct

    return {"iOS17": ct.target.iOS17, "iOS18": ct.target.iOS18, "iOS26": ct.target.iOS26}[opset]


def pipeline_for(arm: str):
    """The pinned default pipeline; for C7 minus the passes probes.py found absorbing the output multiply."""
    import coremltools as ct

    pipe = ct.PassPipeline.DEFAULT
    excluded = []
    if arm == "C7" and C7_EXCLUDED_PASSES_FILE.exists():
        excluded = json.loads(C7_EXCLUDED_PASSES_FILE.read_text())["excluded_passes"]
        pipe.remove_passes(set(excluded))
    return pipe, excluded


def make_program(P, variant: str):
    """(program, inputs for ct.convert or None) of an encoder provider in a length variant."""
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil.mil import Program

    opset = target(P.opset)
    prog = Program()
    if variant == "fixed":
        prog.add_function("main", enc_mod.function(P, enc_mod.BUCKETS[15], opset))
        return prog, None
    if variant == "multi":
        for b, frames in enc_mod.BUCKETS.items():
            prog.add_function(f"b{b}", enc_mod.function(P, frames, opset))
        prog.default_function_name = "b15"
        prog.export_as_multifunction = True
        return prog, None
    if variant == "enum":
        prog.add_function("main", enc_mod.function(P, None, opset))
        shapes = [(1, enc_mod.MEL, f) for f in enc_mod.BUCKETS.values()]
        inputs = [ct.TensorType(name="mel", shape=ct.EnumeratedShapes(shapes=shapes, default=shapes[-1]), dtype=np.float32),
                  ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32)]
        return prog, inputs
    raise KeyError(variant)


def convert(prog, opset: str, pipeline, inputs=None, precision: str = "fp16"):
    """Converted MLModel (not loaded) and the final pymil program (precision "fp32" only for the F32 diagnostic)."""
    import coremltools as ct

    tgt = target(opset)
    cp = ct.precision.FLOAT32 if precision == "fp32" else ct.precision.FLOAT16
    if not prog.export_as_multifunction:
        model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=tgt,
                           compute_precision=cp, pass_pipeline=pipeline, inputs=inputs,
                           compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
        return model, model._mil_program
    from coremltools.converters.mil.converter import _mil_convert
    from coremltools.converters.mil.converter import ConverterRegistry
    from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline, PassPipelineManager
    from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
    from coremltools.models import MLModel

    default_name = prog.default_function_name
    prog = ct.convert(prog, convert_to="milinternal", minimum_deployment_target=tgt,
                      compute_precision=cp, pass_pipeline=pipeline)
    PassPipelineManager.apply_pipeline(prog, PassPipeline.get_pipeline("backend_mlprogram"))
    PASS_REGISTRY["common::const_deduplication"]._deduplicate_const_across_functions(prog)
    prog.default_function_name = default_name
    prog.export_as_multifunction = True
    prog.skip_all_passes = True
    spec_version = {"iOS17": 8, "iOS18": 9, "iOS26": 10}[opset]
    model = _mil_convert(prog, convert_from="milinternal", convert_to="mlprogram", registry=ConverterRegistry,
                         modelClass=MLModel, compute_units=ct.ComputeUnit.CPU_ONLY,
                         specification_version=spec_version, skip_model_load=True)
    return model, prog


def describe_program(prog) -> dict:
    """Per function: op histogram, constexpr input literals (dtype/shape, grouped), post-matmul multiplies."""
    from coremltools.converters.mil.mil import types

    out = {}
    for fname, func in prog.functions.items():
        hist = Counter(op.op_type for op in func.operations)
        constexpr: dict[str, Counter] = {}
        shared_weight_ids = sum(1 for op in func.operations if op.op_type == "const" and op.weight_id is not None)
        mm_then_mul = 0
        for op in func.operations:
            if op.op_type.startswith("constexpr_"):
                sig = ", ".join(f"{k}: {types.builtin_to_string(v.dtype)} {list(v.shape)}" for k, v in op.inputs.items()
                                if hasattr(v, "dtype") and k not in ("name",))
                constexpr.setdefault(op.op_type, Counter())[sig] += 1
            if op.op_type in ("linear", "conv", "matmul") and op.outputs[0].child_ops:
                child = op.outputs[0].child_ops
                if any(c.op_type == "mul" for c in child):
                    mm_then_mul += 1
        out[fname] = {"ops": sum(hist.values()), "op_histogram": dict(sorted(hist.items())),
                      "constexpr_inputs": {k: dict(v) for k, v in constexpr.items()},
                      "matmul_followed_by_mul": mm_then_mul, "consts_with_shared_weight_id": shared_weight_ids,
                      "inputs": {k: f"{types.builtin_to_string(v.dtype)} {list(v.shape)}" for k, v in func.inputs.items()},
                      "outputs": {v.name: f"{types.builtin_to_string(v.dtype)} {list(v.shape)}" for v in func.outputs}}
    return out


def compile_model(package: Path, dest: Path) -> dict:
    """mlpackage -> mlmodelc (macOS): wall time and size."""
    import coremltools as ct

    if dest.exists():
        shutil.rmtree(dest)
    t0 = time.time()
    compiled = ct.utils.compile_model(str(package), str(dest))
    return {"compile_s": round(time.time() - t0, 2), "mlmodelc": str(compiled), "mlmodelc_bytes": dir_bytes(Path(compiled))}


def computeplan_tool() -> Path:
    """The standalone Swift MLComputePlan tool (mil/computeplan.swift), built on first use into <artifacts>/bin."""
    import artifacts

    src = Path(__file__).resolve().parent / "computeplan.swift"
    exe = artifacts.root() / "bin" / "computeplan"
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        exe.parent.mkdir(parents=True, exist_ok=True)
        import subprocess

        subprocess.run(["xcrun", "swiftc", "-O", "-parse-as-library", "-o", str(exe), str(src)], check=True)
    return exe


def compute_plan(path: Path, units: str = "cpuAndNeuralEngine", functions: list[str] | None = None) -> dict:
    """MLComputePlan per-op device usage of every function of a compiled model, summarized (gate 6 record).

    macOS: the Swift tool, one function at a time (MLModelConfiguration.functionName); coremltools' Python
    MLComputePlan reports usage for the default function only."""
    import subprocess

    exe = computeplan_tool()
    out = {"compute_units": units, "tool": "mil/computeplan.swift (MLComputePlan, functionName set)", "functions": {}}
    for fn in functions or [None]:
        cmd = [str(exe), "--model", str(path), "--units", units] + (["--function", fn] if fn else [])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            out["functions"][fn or "main"] = {"error": proc.stderr[-1000:]}
            continue
        res = json.loads(proc.stdout)
        out["functions"][res.pop("function")] = res
        # Core ML caches a device-specialized copy per load under the tool's name (GBs per model); only this
        # tool writes there, so the cache is emptied after every plan (keeps the shared Mac's disk > 60 GB)
        cache = Path.home() / "Library" / "Caches" / "computeplan" / "com.apple.e5rt.e5bundlecache"
        if cache.is_dir():
            for entry in cache.glob("*/*"):
                shutil.rmtree(entry, ignore_errors=True)
    return out


def function_names(variant: str) -> list[str]:
    return ["b2", "b4", "b8", "b15"] if variant == "multi" else ["main"]


def load_times(path: Path, functions: list[str | None], units: str = "cpuAndNeuralEngine") -> dict:
    """Wall time of CompiledMLModel loads per function, first then second (informational; the Mac is shared)."""
    import coremltools as ct

    cu = {"cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE, "cpuOnly": ct.ComputeUnit.CPU_ONLY}[units]
    out = {}
    for fn in functions:
        times = []
        for _ in range(2):
            t0 = time.time()
            m = ct.models.CompiledMLModel(str(path), compute_units=cu, function_name=fn)
            times.append(round(time.time() - t0, 3))
            del m
        out[fn or "main"] = {"first_s": times[0], "second_s": times[1]}
    return out


def environment() -> dict:
    import coremltools as ct
    import numpy as np

    return {"coremltools": ct.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine()}


def drop_temp_package(mlmodel) -> None:
    """Delete the temporary .mlpackage coremltools keeps for an unloaded converted model (user temp dir)."""
    import tempfile

    path = getattr(mlmodel, "package_path", None)
    if path and Path(path).exists() and Path(path).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
        shutil.rmtree(path, ignore_errors=True)


def save_manifest(manifest: dict, out_dir: Path, model: str, arm: str, variant: str) -> None:
    text = json.dumps(manifest, indent=1, default=str) + "\n"
    (out_dir / f"manifest-{variant}.json").write_text(text)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if len(text) > 400_000:
        raise ValueError("manifest unexpectedly large")
    (RESULTS / f"{model}-{arm}-{variant}.json").write_text(text)


def build_encoder(model: str, arm: str, variant: str, out_root: Path, layers: int | None = None,
                  compile_: bool = True, plan: bool = True, keep_package: bool = True, tag: str = "") -> dict:
    import artifacts
    from .weights import C0Tensors, N_LAYERS, Source

    n_layers = layers or N_LAYERS
    out_dir = artifacts.check(out_root / model / arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    act_scales = None
    if arm == "G0":
        if variant != "fixed":
            raise ValueError("G0 is C0's fixed 15 s window only")
        P = enc_mod.G0Provider(C0Tensors(), n_layers)
        provenance = P.c0.provenance
    else:
        source = Source(model)
        provenance = source.provenance
        if arm == "C5":
            calib = json.loads(calibration_file(model).read_text())
            act_scales = {k: v["scale"] for k, v in calib["sites"].items()}
        P = enc_mod.ArmProvider(source, arm, n_layers, act_scales)
    pipe, excluded = pipeline_for(arm)
    prog, inputs = make_program(P, variant)
    t_built = time.time()
    mlmodel, final = convert(prog, P.opset, pipe, inputs, precision=P.dt)
    t_conv = time.time()
    name = f"{variant}{tag}"
    package = out_dir / f"{name}.mlpackage"
    if package.exists():
        shutil.rmtree(package)
    mlmodel.save(str(package))
    drop_temp_package(mlmodel)
    t_saved = time.time()
    weight_bin = package / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
    manifest = {
        "schema": 1, "model": model, "arm": arm, "variant": variant, "layers": n_layers,
        "exact": arm in encodings.EXACT, "exploratory": arm in ("C5",),
        "provenance": provenance, "opset": P.opset, "minimum_deployment_target": P.opset,
        "compute_precision": "FLOAT16 (graph authored in FP16; fp32/int32 I/O)",
        "functions": ({"b%d" % b: {"mel": [1, 128, f], "encoder_frames": enc_mod.encoder_frames(f)}
                       for b, f in enc_mod.BUCKETS.items()} if variant == "multi" else
                      {"main": {"mel": [1, 128, 1501] if variant == "fixed" else
                                {"enumerated": [[1, 128, f] for f in enc_mod.BUCKETS.values()], "default": [1, 128, 1501]},
                                "encoder_frames": 188 if variant == "fixed" else
                                [enc_mod.encoder_frames(f) for f in enc_mod.BUCKETS.values()]}}),
        "default_function": "b15" if variant == "multi" else "main",
        "encoding": P.manifest(),
        "pass_pipeline": {"name": "DEFAULT" + (" minus excluded" if excluded else ""), "excluded": excluded,
                          "passes": list(pipe.passes),
                          "multifunction": ("frontend_milinternal + main + backend_mlprogram per function, then "
                                            "const_deduplication._deduplicate_const_across_functions, export with "
                                            "skip_all_passes") if variant == "multi" else None},
        "program": describe_program(final),
        "sizes": {"mlpackage_bytes": dir_bytes(package), "weight_bin_bytes": weight_bin.stat().st_size if weight_bin.exists() else None},
        "timing_s": {"build_graph": round(t_built - t0, 1), "convert": round(t_conv - t_built, 1),
                     "save": round(t_saved - t_conv, 1)},
        "peak_rss_mb_after_save": peak_rss_mb(),
        "environment": environment(),
        "paths": {"mlpackage": str(package)},
    }
    del mlmodel, final, prog, P
    if compile_ and sys.platform == "darwin":
        manifest["compile"] = compile_model(package, out_dir / f"{name}.mlmodelc")
        manifest["paths"]["mlmodelc"] = manifest["compile"]["mlmodelc"]
        if plan:
            manifest["compute_plan"] = {u: compute_plan(Path(manifest["compile"]["mlmodelc"]), u, function_names(variant))
                                        for u in ("cpuAndNeuralEngine", "cpuOnly")}
        manifest["peak_rss_mb_after_compile"] = peak_rss_mb()
    if not keep_package:
        shutil.rmtree(package)
        manifest["paths"]["mlpackage"] = None
    if not tag and n_layers == N_LAYERS:
        save_manifest(manifest, out_dir, model, arm, variant)
    else:
        (out_dir / f"manifest-{name}-{n_layers}L.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    return manifest


def calibration_file(model: str) -> Path:
    return IOS / "results" / "calibration" / f"{model}-c5-activations.json"


def build_decoder(model: str, out_root: Path, plan: bool = True) -> dict:
    import artifacts
    from . import decoder as dec_mod
    from .weights import Source

    out_dir = artifacts.check(out_root / model / "decoder")
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Source(model)
    W = dec_mod.weights(source)
    manifest = {"schema": 1, "model": model, "provenance": source.provenance, "opset": DECODER_OPSET,
                "compute_precision": "FLOAT16 (graph authored in FP16; fp32/int32 I/O)", "models": {},
                "environment": environment()}
    pipe, _ = pipeline_for("decoder")
    for name, builder in dec_mod.BUILDERS.items():
        t0 = time.time()
        prog = builder(W)
        mlmodel, final = convert(prog, DECODER_OPSET, pipe)
        package = out_dir / f"{name}.mlpackage"
        if package.exists():
            shutil.rmtree(package)
        mlmodel.save(str(package))
        drop_temp_package(mlmodel)
        entry = {"program": describe_program(final), "convert_s": round(time.time() - t0, 1),
                 "mlpackage_bytes": dir_bytes(package), "paths": {"mlpackage": str(package)}}
        if sys.platform == "darwin":
            entry["compile"] = compile_model(package, out_dir / f"{name}.mlmodelc")
            entry["paths"]["mlmodelc"] = entry["compile"]["mlmodelc"]
            if plan:
                entry["compute_plan"] = {u: compute_plan(Path(entry["compile"]["mlmodelc"]), u)
                                         for u in ("cpuAndNeuralEngine", "cpuOnly")}
        manifest["models"][name] = entry
    manifest["pass_pipeline"] = list(pipe.passes)
    manifest["peak_rss_mb"] = peak_rss_mb()
    text = json.dumps(manifest, indent=1, default=str) + "\n"
    (out_dir / "manifest.json").write_text(text)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{model}-decoder.json").write_text(text)
    return manifest


def replan(model: str, arm: str, variant: str, only_units: str | None = None, only_function: str | None = None) -> None:
    """Recompute compute plans and merge them into the manifest and its results copy. By default both unit
    settings and every function; --units/--function restrict it to one, so each guarded job compiles at most
    one ANE program (smaller system-memory and swap footprint on the shared Mac)."""
    base = default_out() / model / arm
    units = (only_units,) if only_units else ("cpuAndNeuralEngine", "cpuOnly")
    if arm == "decoder":
        path = base / "manifest.json"
        m = json.loads(path.read_text())
        for name, entry in m["models"].items():
            entry["compute_plan"] = {u: compute_plan(base / f"{name}.mlmodelc", u) for u in units}
        text = json.dumps(m, indent=1, default=str) + "\n"
        path.write_text(text)
        (RESULTS / f"{model}-decoder.json").write_text(text)
        return
    path = base / f"manifest-{variant}.json"
    m = json.loads(path.read_text())
    fns = [only_function] if only_function else function_names(variant)
    plans = m.setdefault("compute_plan", {})
    for u in units:
        new = compute_plan(base / f"{variant}.mlmodelc", u, fns)
        if u in plans and "functions" in plans[u] and only_function:
            plans[u]["functions"].update(new["functions"])
        else:
            plans[u] = new
    save_manifest(m, base, model, arm, variant)
    print(json.dumps({u: {fn: v.get("ops_with_usage_by_preferred_device") for fn, v in p["functions"].items()}
                      for u, p in m["compute_plan"].items()}))


def default_out() -> Path:
    import artifacts

    return artifacts.root() / "arms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("encoder")
    p.add_argument("--model", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--layers", type=int)
    p.add_argument("--out-dir")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-plan", action="store_true")
    p.add_argument("--drop-package", action="store_true", help="delete the .mlpackage after compiling (disk)")
    p = sub.add_parser("decoder")
    p.add_argument("--model", required=True)
    p.add_argument("--out-dir")
    p = sub.add_parser("plan")
    p.add_argument("--path", required=True)
    p.add_argument("--units", default="cpuAndNeuralEngine")
    p.add_argument("--functions", default="")
    p = sub.add_parser("replan", help="recompute the compute plans of a built encoder/decoder and update its manifest")
    p.add_argument("--model", required=True)
    p.add_argument("--arm", required=True, help="encoder arm or 'decoder'")
    p.add_argument("--variant", default="fixed")
    p.add_argument("--units", choices=("cpuAndNeuralEngine", "cpuOnly"))
    p.add_argument("--function")
    p = sub.add_parser("loadtimes")
    p.add_argument("--path", required=True)
    p.add_argument("--functions", default="")
    p.add_argument("--units", default="cpuAndNeuralEngine")
    args = parser.parse_args()
    if args.cmd == "encoder":
        m = build_encoder(args.model, args.arm, args.variant, Path(args.out_dir) if args.out_dir else default_out(),
                          args.layers, compile_=not args.no_compile, plan=not args.no_plan,
                          keep_package=not args.drop_package)
        print(json.dumps({k: m[k] for k in ("model", "arm", "variant", "sizes", "timing_s", "peak_rss_mb_after_save")}
                         | {"compile": m.get("compile"), "peak_rss_mb_after_compile": m.get("peak_rss_mb_after_compile")}))
    elif args.cmd == "decoder":
        m = build_decoder(args.model, Path(args.out_dir) if args.out_dir else default_out())
        print(json.dumps({k: {"mlpackage_bytes": v["mlpackage_bytes"], "compile": v.get("compile")}
                          for k, v in m["models"].items()} | {"peak_rss_mb": m["peak_rss_mb"]}))
    elif args.cmd == "plan":
        fns = [f for f in args.functions.split(",") if f] or None
        print(json.dumps(compute_plan(Path(args.path), args.units, fns), indent=1))
    elif args.cmd == "replan":
        replan(args.model, args.arm, args.variant, args.units, args.function)
    elif args.cmd == "loadtimes":
        fns = [f or None for f in args.functions.split(",")] if args.functions else [None]
        print(json.dumps(load_times(Path(args.path), fns, args.units)))


if __name__ == "__main__":
    main()
