"""One-layer probes before any full build (DESIGN.md "Implementation decisions", gates 2, "C7 folding risk", "C7/C8 numerics").

  cd ios && python -m mil.probes memory [--arms C1,C4,...] [--variants fixed,multi]   # subprocess per build
  cd ios && python -m mil.probes gate2 --package P.mlpackage --model mp2 --arm C4 [--out JSON]
  cd ios && python -m mil.probes folding
  cd ios && python -m mil.probes stress [--arms C7,C8,C1,C4]
  cd ios && python -m mil.probes summary                                           # results/probes/summary.json

memory: for every encoding, builds the encoder with 1 and 2 layers of M_P2 (fixed and multifunction), each
in its own subprocess (mil.build, compile included on macOS), records the subprocess's peak RSS, build /
convert / save / compile times and sizes, runs gate 2 on the 1- and 2-layer fixed packages, and
extrapolates linearly to 24 layers: v24 = v1 + 23 (v2 - v1). A full build is allowed only if 2 x the
projected peak fits the job's RSS cap.

gate2: loads the saved package's program (coremltools milproto loader) and rebuilds every ternary module's
effective matrix from its constexpr chain with coremltools' own decompression (materialized_val_inference /
the ops' decompress): C3/C4/C5 decompressed weight; C7 diag(s) . C with s the post-matmul multiplier; C8
diag(s) . (P - N); C6s/C6d decompressed palettes; C1 the dense const. Each must equal codes x FP16(s) bit
for bit (FP16 bit patterns).

folding: a linear and a 1x1 conv with a C7 weight (constexpr LUT {-1,0,+1,0}) followed by the per-row
multiply, plus the same with a dense const weight as a control, converted with the pinned DEFAULT
pipeline; the final program is inspected for the multiply, and the pipeline is replayed pass by pass to
name any pass that removes it. On macOS the compiled model.mil is inspected too. Any absorbing pass is
written to results/c7_excluded_passes.json (used by build.pipeline_for for C7 only).

stress: C7 and C8 (C1 and C4 for comparison) on real M_P2 modules (layer 0 FF1 linear1 and linear2,
layer 23 FF2 linear2) with rows 0-3 replaced by adversarial rows (all +1, all -1, alternating runs of
256, half +1 / half -1), inputs = the clip's activation at that site x 1, 8, 64 and |activation| x 64,
run with CPU_ONLY and CPU_AND_NE. Outputs expose the intermediates (C7: matmul before the scale; C8: P, N,
P - N). Records maxima, finiteness and the error against float64 numpy. Any inf/NaN fails the arm.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "mil"

import numpy as np

from . import IOS, encodings  # noqa: E402

PROBES = IOS / "results" / "probes"
PROBE_ARMS = ("C1", "C3", "C4", "C7", "C8", "C6s2", "C6s4", "C6s8", "C6d4", "C6d8", "C5")
STRESS_MODULES = (("layers.0.feed_forward1.linear1", "ff1_in"), ("layers.0.feed_forward1.linear2", "ff1_mid"),
                  ("layers.23.feed_forward2.linear2", "ff2_mid"))


def art_root() -> Path:
    import artifacts

    return artifacts.root()


# --- gate 2 ---------------------------------------------------------------------------------------------

def load_program(package: Path):
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load as milproto_load

    spec = ct.utils.load_spec(str(package))
    return milproto_load(spec, spec.specificationVersion, str(Path(package) / "Data" / "com.apple.CoreML" / "weights"))


def _dense(var) -> np.ndarray:
    op = var.op
    if op.op_type == "const":
        return np.asarray(op.val.val if hasattr(op.val, "val") else var.val)
    val = op.materialized_val_inference()
    if isinstance(val, tuple):
        val = val[-1]
    return np.asarray(val)


def _chain(var) -> list[str]:
    out, op = [], var.op
    while op is not None and (op.op_type.startswith("constexpr_") or op.op_type == "const"):
        out.append(op.op_type)
        nxt = [v for k, v in op.inputs.items() if hasattr(v, "op") and v.op is not None
               and v.op.op_type.startswith("constexpr_")]
        op = nxt[0].op if nxt else None
    return out[::-1]




def gate2(package: Path, model: str, arm: str) -> dict:
    """Effective matrices of every ternary module in the package vs codes x FP16(s), bit for bit."""
    from .weights import TERNARY_SUFFIXES, Source

    suffixes = {s.replace(".", "_"): s for s in TERNARY_SUFFIXES}
    source = Source(model)
    prog = load_program(package)
    fname = "main" if "main" in prog.functions else next(iter(prog.functions))
    func = prog.functions[fname]
    rows, failures, chains = [], [], {}
    for op in func.operations:
        m = re.fullmatch(r"l(\d+)_(.+)_mm", op.name)
        if op.op_type not in ("linear", "conv") or not m:
            continue
        layer, suffix = int(m.group(1)), suffixes[m.group(2)]
        key = f"encoder.layers.{layer}.{suffix}"
        w = _dense(op.weight).astype(np.float16)
        w = w.reshape(w.shape[0], -1)
        chain = _chain(op.weight)
        fam = encodings.family(arm)
        children = op.outputs[0].child_ops
        post = None
        if fam == "C7":
            mul = next(c for c in children if c.op_type == "mul")
            s = (mul.y if mul.x is op.outputs[0] else mul.x).val.reshape(-1).astype(np.float16)
            eff = (w.astype(np.float32) * s.astype(np.float32)[:, None]).astype(np.float16)
            post = "mul"
        elif fam == "C8":
            split = next(c for c in children if c.op_type == "split")
            sub = split.outputs[0].child_ops[0]
            mul = sub.outputs[0].child_ops[0]
            if sub.op_type != "sub" or mul.op_type != "mul":
                raise ValueError(f"{op.name}: C8 tail is {sub.op_type} -> {mul.op_type}")
            s = (mul.y if mul.x is sub.outputs[0] else mul.x).val.reshape(-1).astype(np.float16)
            n = w.shape[0] // 2
            eff = ((w[:n].astype(np.float32) - w[n:].astype(np.float32)) * s.astype(np.float32)[:, None]).astype(np.float16)
            post = "split-sub-mul"
        else:
            eff = w
        codes, scale = source.ternary(key)
        ok = encodings.bits_equal(eff, encodings.reference_matrix(codes, scale))
        chains[" -> ".join(chain) + (f" -> {op.op_type} -> {post}" if post else f" -> {op.op_type}")] = \
            chains.get(" -> ".join(chain) + (f" -> {op.op_type} -> {post}" if post else f" -> {op.op_type}"), 0) + 1
        rows.append(key)
        if not ok:
            failures.append(key)
    return {"package": str(package), "model": model, "arm": arm, "function": fname, "modules_checked": len(rows),
            "bit_exact": not failures and len(rows) > 0, "failures": failures, "chains": chains,
            "method": "coremltools materialized_val_inference / decompress of the saved program's constexpr ops; "
                      "C7/C8 effective matrices with the post-matmul FP16 multiplier; compared as FP16 bit patterns"}


# --- memory / time probes -------------------------------------------------------------------------------

def _run_build(arm: str, variant: str, layers: int, out_dir: Path) -> dict:
    cmd = [sys.executable, "-m", "mil.build", "encoder", "--model", "mp2", "--arm", arm, "--variant", variant,
           "--layers", str(layers), "--out-dir", str(out_dir), "--no-plan"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=IOS, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        return {"error": proc.stderr[-2000:], "returncode": proc.returncode}
    last = [l for l in proc.stdout.splitlines() if l.startswith("{")][-1]
    res = json.loads(last)
    res["wall_s"] = round(wall, 1)
    return res


def memory(args) -> None:
    out_dir = art_root() / "probes" / "arms"
    PROBES.mkdir(parents=True, exist_ok=True)
    path = PROBES / "memory.json"
    doc = json.loads(path.read_text()) if path.exists() else {"runs": {}}
    arms = args.arms.split(",") if args.arms else PROBE_ARMS
    for arm in arms:
        for variant in args.variants.split(","):
            for layers in (1, 2):
                key = f"{arm}/{variant}/{layers}L"
                if key in doc["runs"] and not args.force:
                    continue
                res = _run_build(arm, variant, layers, out_dir)
                if variant == "fixed" and "error" not in res:
                    pkg = out_dir / "mp2" / arm / "fixed.mlpackage"
                    res["gate2"] = {k: v for k, v in gate2_subprocess(pkg, arm).items() if k != "package"}
                doc["runs"][key] = res
                path.write_text(json.dumps(doc, indent=1) + "\n")
                print(key, json.dumps(res)[:400], flush=True)
            shutil.rmtree(out_dir / "mp2" / arm, ignore_errors=True)


def gate2_subprocess(package: Path, arm: str) -> dict:
    proc = subprocess.run([sys.executable, "-m", "mil.probes", "gate2", "--package", str(package), "--model", "mp2",
                           "--arm", arm], cwd=IOS, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"error": proc.stderr[-1500:]}
    return json.loads([l for l in proc.stdout.splitlines() if l.startswith("{")][-1])


def extrapolate(doc: dict, cap_gb: float = 6.0) -> dict:
    out = {}
    for key, r1 in doc["runs"].items():
        if not key.endswith("/1L"):
            continue
        base = key[:-3]
        r2 = doc["runs"].get(base + "/2L")
        if not r2 or "error" in r1 or "error" in r2:
            out[base] = {"error": r1.get("error") or (r2 or {}).get("error") or "missing 2-layer run"}
            continue

        def lin(f):
            a, b = f(r1), f(r2)
            return None if a is None or b is None else round(a + 23 * (b - a), 1)

        peak = lambda r: r.get("peak_rss_mb_after_compile") or r.get("peak_rss_mb_after_save")
        proj_peak = lin(peak)
        entry = {"peak_rss_mb": {"1L": peak(r1), "2L": peak(r2), "24L_projected": proj_peak},
                 "convert_s": {"1L": r1["timing_s"]["convert"], "2L": r2["timing_s"]["convert"],
                               "24L_projected": lin(lambda r: r["timing_s"]["convert"])},
                 "wall_s": {"1L": r1["wall_s"], "2L": r2["wall_s"], "24L_projected": lin(lambda r: r["wall_s"])},
                 "mlpackage_mb": {"1L": round(r1["sizes"]["mlpackage_bytes"] / 2 ** 20, 2),
                                  "2L": round(r2["sizes"]["mlpackage_bytes"] / 2 ** 20, 2),
                                  "24L_projected": lin(lambda r: r["sizes"]["mlpackage_bytes"] / 2 ** 20)}}
        if r1.get("compile"):
            entry["compile_s"] = {"1L": r1["compile"]["compile_s"], "2L": r2["compile"]["compile_s"],
                                  "24L_projected": lin(lambda r: r["compile"]["compile_s"])}
            entry["mlmodelc_mb"] = {"1L": round(r1["compile"]["mlmodelc_bytes"] / 2 ** 20, 2),
                                    "24L_projected": lin(lambda r: r["compile"]["mlmodelc_bytes"] / 2 ** 20)}
        entry["full_build_allowed_at_cap_gb"] = {"cap": cap_gb, "allowed": proj_peak is not None
                                                 and 2 * proj_peak <= cap_gb * 1024}
        for g in ("gate2",):
            if g in r1:
                entry["gate2_1L"] = {k: r1[g].get(k) for k in ("bit_exact", "modules_checked", "failures", "chains", "error")}
            if g in r2:
                entry["gate2_2L"] = {k: r2[g].get(k) for k in ("bit_exact", "modules_checked", "failures", "error")}
        out[base] = entry
    return out


# --- C7 folding probe -----------------------------------------------------------------------------------

def _folding_programs(rng):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    codes = rng.integers(-1, 2, size=(256, 256)).astype(np.int8)
    scale = (rng.random(256) * 0.05 + 0.01).astype(np.float32)
    progs = {}
    for kind in ("linear", "conv"):
        for weight in ("lut", "dense"):
            rank = 2 if kind == "linear" else 3
            enc = encodings.encode("C7", codes, scale, rank)
            shape = (1, 8, 256) if kind == "linear" else (1, 256, 8)

            @mb.program(input_specs=[mb.TensorSpec(shape=shape, dtype=types.fp32)], opset_version=ct.target.iOS26)
            def prog(x):
                x16 = mb.cast(x=x, dtype="fp16")
                if weight == "lut":
                    w = encodings.weight_var(enc, "w")
                else:
                    w = mb.const(val=encodings.reference_matrix(codes, np.ones_like(scale)).reshape(enc.consts["indices"].shape),
                                 name="w")
                y = mb.linear(x=x16, weight=w, name="mm") if kind == "linear" else mb.conv(x=x16, weight=w, name="mm")
                s = enc.post_scale if kind == "linear" else enc.post_scale.reshape(1, -1, 1)
                y = mb.mul(x=y, y=s, name="row_scale")
                return mb.cast(x=y, dtype="fp32", name="out")

            progs[f"{kind}-{weight}"] = prog
    return progs


def _has_post_mul(prog) -> bool:
    for f in prog.functions.values():
        for op in f.operations:
            if op.op_type in ("linear", "conv") and any(c.op_type == "mul" for c in op.outputs[0].child_ops):
                return True
    return False


def folding(args) -> None:
    import copy

    import coremltools as ct
    from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline, PassPipelineManager

    from .build import compile_model, environment

    out_dir = art_root() / "probes" / "folding"
    out_dir.mkdir(parents=True, exist_ok=True)
    text_dir = PROBES / "c7_folding"
    text_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    pipeline = ct.PassPipeline.DEFAULT
    results = {}
    absorbing: set[str] = set()
    for name, prog in _folding_programs(rng).items():
        # pass-by-pass replay on a copy: which pass (if any) removes the multiply
        trial = copy.deepcopy(prog)
        removed_by = None
        for pipe in (PassPipeline.get_pipeline("frontend_milinternal"), pipeline,
                     PassPipeline.get_pipeline("backend_mlprogram")):
            for p in pipe.passes:
                single = PassPipeline(pass_names=[p], pipeline_name=p)
                for opt_pass, opts in getattr(pipe, "_pass_options", {}).items():
                    if opt_pass == p:
                        single._pass_options[p] = opts
                PassPipelineManager.apply_pipeline(trial, single)
                if removed_by is None and not _has_post_mul(trial):
                    removed_by = p
        model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS26,
                           compute_precision=ct.precision.FLOAT16, pass_pipeline=pipeline,
                           compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
        final = model._mil_program
        (text_dir / f"{name}.mil.txt").write_text(str(final))
        entry = {"post_mul_survives_conversion": _has_post_mul(final), "removed_by_pass": removed_by,
                 "final_ops": [op.op_type for op in final.functions["main"].operations if op.op_type != "const"]}
        pkg = out_dir / f"{name}.mlpackage"
        if pkg.exists():
            shutil.rmtree(pkg)
        model.save(str(pkg))
        from .build import drop_temp_package
        drop_temp_package(model)
        if sys.platform == "darwin":
            comp = compile_model(pkg, out_dir / f"{name}.mlmodelc")
            mil_text = (Path(comp["mlmodelc"]) / "model.mil").read_text()
            (text_dir / f"{name}.compiled.model.mil").write_text(mil_text)
            body = mil_text[mil_text.index("func main"):]
            entry["compiled_model_mil_ops"] = re.findall(r"= (\w+)\(", body)
            entry["compiled_has_mul_after_matmul"] = bool(re.search(r"(linear|conv)\(", body)) and " = mul(" in body
        if name.endswith("-lut") and removed_by:
            absorbing.add(removed_by)
        results[name] = entry
    doc = {"probe": "C7 folding", "pipeline": "ct.PassPipeline.DEFAULT (pinned list below) + frontend_milinternal + "
           "backend_mlprogram", "passes": list(pipeline.passes), "target": "iOS26", "cases": results,
           "absorbing_passes_for_lut_weights": sorted(absorbing), "environment": environment(),
           "note": "folding inside the device compiler (Core ML / ANE compiler) cannot be observed here; "
                   "unresolved by design (DESIGN.md)"}
    (PROBES / "c7_folding.json").write_text(json.dumps(doc, indent=1) + "\n")
    (IOS / "results" / "c7_excluded_passes.json").write_text(json.dumps(
        {"excluded_passes": sorted(absorbing), "source": "probes.py folding", "date": time.strftime("%Y-%m-%d")}, indent=1) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "final_ops"} for k, v in results.items()}, indent=1))


# --- C7 / C8 stress -------------------------------------------------------------------------------------

def adversarial(codes: np.ndarray) -> np.ndarray:
    c = codes.copy()
    n = c.shape[1]
    c[0] = 1
    c[1] = -1
    c[2] = np.where((np.arange(n) // 256) % 2 == 0, 1, -1)
    c[3] = np.where(np.arange(n) < n // 2, 1, -1)
    return c


def _stress_program(arm: str, codes, scale, t: int):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    enc = encodings.encode(arm, codes, scale, 2)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, t, codes.shape[1]), dtype=types.fp32)], opset_version=ct.target.iOS26)
    def prog(x):
        x16 = mb.cast(x=x, dtype="fp16")
        w = encodings.weight_var(enc, "w")
        y = mb.linear(x=x16, weight=w, name="mm")
        outs = []
        if arm == "C7":
            outs.append(mb.cast(x=y, dtype="fp32", name="raw"))
            y = mb.mul(x=y, y=enc.post_scale, name="scaled")
        elif arm == "C8":
            p, n = mb.split(x=y, num_splits=2, axis=-1, name="pn")
            d = mb.sub(x=p, y=n, name="diff")
            outs += [mb.cast(x=p, dtype="fp32", name="P"), mb.cast(x=n, dtype="fp32", name="N"),
                     mb.cast(x=d, dtype="fp32", name="raw")]
            y = mb.mul(x=d, y=enc.post_scale, name="scaled")
        outs.append(mb.cast(x=y, dtype="fp32", name="out"))
        return tuple(outs)

    return prog


def stress(args) -> None:
    import coremltools as ct

    from . import refcache
    from .build import compile_model, compute_plan, environment
    from .weights import Source, fp16_scale

    source = Source("mp2")
    probe_inputs = np.load(refcache.default_out("mp2") / "probe_inputs.npz")
    out_dir = art_root() / "probes" / "stress"
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    units = {"cpuOnly": ct.ComputeUnit.CPU_ONLY, "cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE}
    results = {}
    for module, site in STRESS_MODULES:
        codes, scale = source.ternary(f"encoder.{module}")
        codes = adversarial(codes)
        layer = module.split(".")[1]
        x0 = probe_inputs[f"layers_{layer}_{site}"].astype(np.float32)  # [T, in]
        w64 = codes.astype(np.float64) * fp16_scale(scale).astype(np.float64)[:, None]
        inputs = {"x1": x0, "x8": 8 * x0, "x64": 64 * x0, "abs_x64": 64 * np.abs(x0)}
        for arm in arms:
            prog = _stress_program(arm, codes, scale, x0.shape[0])
            model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS26,
                               compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY,
                               skip_model_load=True)
            pkg = out_dir / f"{module}-{arm}.mlpackage"
            if pkg.exists():
                shutil.rmtree(pkg)
            model.save(str(pkg))
            from .build import drop_temp_package
            drop_temp_package(model)
            comp = compile_model(pkg, out_dir / f"{module}-{arm}.mlmodelc")
            plan = compute_plan(Path(comp["mlmodelc"]), "cpuAndNeuralEngine")["functions"]["main"]
            for uname, cu in units.items():
                m = ct.models.CompiledMLModel(comp["mlmodelc"], compute_units=cu)
                for iname, x in inputs.items():
                    pred = m.predict({"x": x[None]})
                    ref = (x.astype(np.float64) @ w64.T)
                    row = {"finite": all(bool(np.isfinite(v).all()) for v in pred.values()),
                           "max_abs": {k: float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else None
                                       for k, v in pred.items()},
                           "nonfinite_counts": {k: int((~np.isfinite(v)).sum()) for k, v in pred.items()},
                           "reference_max_abs_out": float(np.abs(ref).max()),
                           "reference_max_abs_raw_rows0_3": float(np.abs(x.astype(np.float64) @ codes[:4].T.astype(np.float64)).max())}
                    if np.isfinite(pred["out"]).all():
                        row["rel_err_out"], row["abs_err_out"] = refcache.errors(pred["out"][0], ref, 1e-3)
                    results.setdefault(module, {}).setdefault(arm, {}).setdefault(uname, {})[iname] = row
                del m
            results[module][arm]["compute_plan_cpuAndNeuralEngine"] = plan["ops_with_usage_by_preferred_device"]
            print(module, arm, json.dumps({u: {i: (r["finite"], {k: round(v, 1) if v else v for k, v in r["max_abs"].items()})
                                               for i, r in d.items()} for u, d in results[module][arm].items()
                                           if u in units}), flush=True)
    verdict = {}
    for arm in arms:
        bad = [(mod, u, i) for mod in results for u in units for i, r in results[mod][arm][u].items() if not r["finite"]]
        verdict[arm] = {"any_nonfinite": bool(bad), "nonfinite_cases": bad}
    doc = {"probe": "C7/C8 stress (DESIGN.md 'C7/C8 numerics')", "modules": [m for m, _ in STRESS_MODULES],
           "adversarial_rows": {"0": "all +1", "1": "all -1", "2": "alternating runs of 256 (+1, -1, ...)",
                                "3": "first half +1, second half -1"},
           "inputs": {"x1": "layer activation of the probe clip (1x clip RMS)", "x8": "x 8", "x64": "x 64",
                      "abs_x64": "|activation| x 64 (all-positive: maximizes the all +1 row)"},
           "probe_clip": json.loads((refcache.default_out("mp2") / "probe_inputs.json").read_text())["clip"],
           "outputs": "C7: raw (matmul before the row scale), out; C8: P, N, raw = P - N, out; C1/C4: out. "
                      "Exposing intermediates as outputs may change the compiled graph.",
           "accumulation": "backend-defined (not exposed by Core ML); recorded as such",
           "verdict": verdict, "results": results, "environment": environment()}
    PROBES.mkdir(parents=True, exist_ok=True)
    (PROBES / "stress.json").write_text(json.dumps(doc, indent=1) + "\n")
    shutil.rmtree(out_dir, ignore_errors=True)
    print(json.dumps(verdict, indent=1))


def summary(args) -> None:
    mem = json.loads((PROBES / "memory.json").read_text())
    doc = {"memory_time_size": extrapolate(mem, args.cap_gb)}
    for name in ("c7_folding", "stress"):
        p = PROBES / f"{name}.json"
        if p.exists():
            d = json.loads(p.read_text())
            doc[name] = ({k: d[k] for k in ("absorbing_passes_for_lut_weights", "cases")} if name == "c7_folding"
                         else d["verdict"])
    (PROBES / "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["memory_time_size"], indent=1)[:6000])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("memory")
    p.add_argument("--arms")
    p.add_argument("--variants", default="fixed,multi")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("gate2")
    p.add_argument("--package", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--out")
    sub.add_parser("folding")
    p = sub.add_parser("stress")
    p.add_argument("--arms", default="C7,C8,C1,C4")
    p = sub.add_parser("summary")
    p.add_argument("--cap-gb", type=float, default=6.0)
    args = parser.parse_args()
    if args.cmd == "memory":
        memory(args)
    elif args.cmd == "gate2":
        res = gate2(Path(args.package), args.model, args.arm)
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        res["peak_rss_mb"] = round(r / 2 ** 20 if sys.platform == "darwin" else r / 1024)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res))
    elif args.cmd == "folding":
        folding(args)
    elif args.cmd == "stress":
        stress(args)
    elif args.cmd == "summary":
        summary(args)


if __name__ == "__main__":
    main()
