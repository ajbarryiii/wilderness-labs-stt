#!/usr/bin/env python3
"""Screening of the 1-bit encoder arms on MB1 (kea-b): gate 2, placement, probe timings, B1P headroom.

Run on the Mac through macguard with the dedicated interpreter alias (Core ML caches its device-specialized
copies per process name; this alias's cache, ~/Library/Caches/keab1py, is the only one purged here):

  PY=/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/kea-b1/pyenv/bin/keab1py
  $PY screen.py gate2-modules [--arms B1L,...]            # all 264 projections, per-module decompression
  $PY screen.py probe --arms B1L,B1S --layers 4           # N-layer multifunction build: sizes, gate 2, plans, timings
  $PY screen.py gate2-package --arm B1L --package P.mlpackage   # every function of a built package
  $PY screen.py plan --arm B1L --path X.mlmodelc [--units cpuAndNeuralEngine,cpuAndGPU,cpuOnly]
  $PY screen.py loadtimes --arm B1L --path X.mlmodelc [--units cpuAndNeuralEngine]
  $PY screen.py activations                               # B1P headroom on public LibriSpeech clips (FP32 reference)
  $PY screen.py stress [--arms B1S,B1P,B1L] [--units ...] # Core ML overflow stress (needs `activations` first)

Text results go to finetune/parakeet-binary-coreml/results/; models, probe builds and captured activations
stay in <artifacts>/kea-b1/. Timings here are informational screening on a shared Mac (Python predict calls,
synthetic mel input), not the benchmark.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import localflow  # noqa: F401  (sys.path)

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
UNITS = ("cpuAndNeuralEngine", "cpuAndGPU", "cpuOnly")
FP16_MAX = 65504.0


def art() -> Path:
    import artifacts

    path = artifacts.check(artifacts.root() / "kea-b1")
    path.mkdir(parents=True, exist_ok=True)
    return path


def own_cache() -> Path | None:
    """This process's Core ML cache, only when running as the dedicated keab1py alias (never a shared cache)."""
    name = Path(sys.executable).name
    return Path.home() / "Library" / "Caches" / name if name == "keab1py" else None


def purge_cache() -> None:
    cache = own_cache()
    if cache is not None and cache.is_dir():
        shutil.rmtree(cache, ignore_errors=True)


def dir_mb(path: Path | None) -> float | None:
    if path is None or not path.is_dir():
        return None
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6, 1)


def write(rel: str, doc: dict) -> Path:
    path = RESULTS / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, default=str) + "\n"
    if len(text) > 400_000:
        raise ValueError(f"{rel}: result unexpectedly large")
    path.write_text(text)
    return path


def modules() -> list[tuple[str, str]]:
    from mil.weights import N_LAYERS, TERNARY_SUFFIXES

    return [(f"encoder.layers.{i}.{s}", s) for i in range(N_LAYERS) for s in TERNARY_SUFFIXES]


REFERENCE_ARMS = ("C6s8",)  # kea-b's shipped chain on the same +-1 codes (the 372.7 MB baseline), for comparison


def arms_arg(text: str | None, reference: bool = False) -> list[str]:
    from mil import encodings

    arms = text.split(",") if text else list(encodings.BINARY_ARMS)
    allowed = encodings.BINARY_ARMS + (REFERENCE_ARMS if reference else ())
    bad = [a for a in arms if a not in allowed]
    if bad:
        raise SystemExit(f"not 1-bit arms: {bad}")
    return arms


# --- gate 2 ------------------------------------------------------------------------------------------------

def decompress(enc) -> np.ndarray:
    """The arm's effective FP16 matrix [out, in] through coremltools' own constexpr decompression (iOS26 ops),
    times the post-matmul FP16 row scale for B1P."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    from mil import encodings, probes

    held = {}

    @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)], opset_version=ct.target.iOS26)
    def prog(x):
        held["w"] = encodings.weight_var(enc, "w")
        return x

    w = probes._dense(held["w"]).astype(np.float16).reshape(enc.out, -1)
    if enc.post_scale is not None:
        w = (w.astype(np.float32) * enc.post_scale.astype(np.float32)[:, None]).astype(np.float16)
    return w


def cmd_gate2_modules(args) -> None:
    from mil import encodings
    from mil.encoder import CONV_MODULES
    from mil.weights import Source

    arms = arms_arg(args.arms)
    source = Source("mb1")
    t0 = time.time()
    failures = {a: [] for a in arms}
    checked = {a: 0 for a in arms}
    weights = 0
    for key, suffix in modules():
        codes, scale = source.ternary(key)
        weights += codes.size
        ref = encodings.reference_matrix(codes, scale)
        rank = 3 if suffix in CONV_MODULES else 2
        for arm in arms:
            ok = encodings.bits_equal(decompress(encodings.encode(arm, codes, scale, rank)), ref)
            checked[arm] += 1
            if not ok:
                failures[arm].append(key)
    doc = {"check": "gate 2, per module: coremltools constexpr decompression (materialized_val_inference) of each "
                    "arm's chain for every 1-bit projection of MB1 (incl. the 24 linear_pos, which the encoder "
                    "folds into position tables) vs codes x FP16(s), FP16 bit patterns",
           "model": source.provenance, "modules": len(modules()), "weights": weights,
           "arms": {a: {"modules_checked": checked[a], "bit_exact": not failures[a] and checked[a] == len(modules()),
                        "failures": failures[a]} for a in arms},
           "seconds": round(time.time() - t0, 1)}
    write("gate2-modules.json", doc)
    print(json.dumps({a: v["bit_exact"] for a, v in doc["arms"].items()}))
    if any(failures.values()):
        sys.exit(10)


def gate2_package(package: Path, arm: str) -> dict:
    """Every function of a saved package: each l<i>_<suffix>_mm weight decompressed vs codes x FP16(s)."""
    import re

    from mil import encodings, probes
    from mil.weights import TERNARY_SUFFIXES, Source

    suffixes = {s.replace(".", "_"): s for s in TERNARY_SUFFIXES}
    source = Source("mb1")
    prog = probes.load_program(package)
    out = {"package": str(package), "arm": arm, "functions": {}, "chains": {}}
    refs: dict[str, np.ndarray] = {}
    for fname, func in prog.functions.items():
        checked, failures = 0, []
        for op in func.operations:
            m = re.fullmatch(r"l(\d+)_(.+)_mm", op.name)
            if op.op_type not in ("linear", "conv") or not m:
                continue
            key = f"encoder.layers.{int(m.group(1))}.{suffixes[m.group(2)]}"
            w = probes._dense(op.weight).astype(np.float16)
            w = w.reshape(w.shape[0], -1)
            post = ""
            if encodings.family(arm) == "B1P":
                mul = next(c for c in op.outputs[0].child_ops if c.op_type == "mul")
                s = (mul.y if mul.x is op.outputs[0] else mul.x).val.reshape(-1).astype(np.float16)
                w = (w.astype(np.float32) * s.astype(np.float32)[:, None]).astype(np.float16)
                post = " -> mul"
            if key not in refs:
                codes, scale = source.ternary(key)
                refs[key] = encodings.reference_matrix(codes, scale)
            chain = " -> ".join(probes._chain(op.weight)) + f" -> {op.op_type}{post}"
            out["chains"][chain] = out["chains"].get(chain, 0) + 1
            checked += 1
            if not encodings.bits_equal(w, refs[key]):
                failures.append(key)
        out["functions"][fname] = {"modules_checked": checked, "failures": failures}
    out["bit_exact"] = all(not f["failures"] and f["modules_checked"] > 0 for f in out["functions"].values())
    return out


def cmd_gate2_package(args) -> None:
    doc = gate2_package(Path(args.package), args.arm)
    write(args.out or f"gate2-package-{args.arm}.json", doc)
    print(json.dumps({"bit_exact": doc["bit_exact"], "functions": {k: v["modules_checked"] for k, v in doc["functions"].items()}}))
    if not doc["bit_exact"]:
        sys.exit(10)


# --- placement and timing -------------------------------------------------------------------------------

MATMUL_OPS = ("linear", "conv", "matmul")


def plan_summary(plan: dict) -> dict:
    out = {}
    for fn, res in plan["functions"].items():
        if "error" in res:
            out[fn] = {"error": res["error"][-300:]}
            continue
        by_dev = res.get("operators_by_preferred_device", {})
        out[fn] = {"ops_by_device": res.get("ops_with_usage_by_preferred_device", {}),
                   "cost_share": {k: round(v, 4) for k, v in res.get("estimated_cost_share_by_preferred_device", {}).items()},
                   "matmul_ops_by_device": {d: sum(n for op, n in by.items() if op.split(".")[-1] in MATMUL_OPS)
                                            for d, by in by_dev.items()},
                   "cpu_operators": by_dev.get("cpu", {}), "gpu_operators": by_dev.get("gpu", {}),
                   "plan_load_s": round(res.get("plan_load_s", 0), 1)}
    return out


def plans(path: Path, units) -> dict:
    from mil import build

    out = {}
    for u in units:
        t0 = time.time()
        out[u] = plan_summary(build.compute_plan(path, u, localflow.FUNCTIONS))
        out[u]["_wall_s"] = round(time.time() - t0, 1)
    return out


def cu(name: str):
    import coremltools as ct

    return {"cpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE, "cpuOnly": ct.ComputeUnit.CPU_ONLY,
            "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU}[name]


def mel_input(fn: str, seed: int = 0) -> dict:
    from mil import encoder

    frames = encoder.BUCKETS[int(fn[1:])]
    rng = np.random.default_rng(seed)
    return {"mel": rng.standard_normal((1, encoder.MEL, frames)).astype(np.float32),
            "mel_length": np.array([frames], dtype=np.int32)}


def timings(path: Path, units, calls: int, warmup: int = 3, functions=None) -> dict:
    import coremltools as ct

    out = {}
    for u in units:
        for fn in functions or localflow.FUNCTIONS:
            purge_cache()
            t0 = time.time()
            model = ct.models.CompiledMLModel(str(path), compute_units=cu(u), function_name=fn)
            load_s = time.time() - t0
            x = mel_input(fn)
            for _ in range(warmup):
                y = model.predict(x)
            ms = []
            for _ in range(calls):
                t = time.perf_counter()
                y = model.predict(x)
                ms.append((time.perf_counter() - t) * 1e3)
            out.setdefault(u, {})[fn] = {"load_s": round(load_s, 2), "median_ms": round(statistics.median(ms), 2),
                                         "min_ms": round(min(ms), 2), "calls": calls,
                                         "output_finite": bool(np.isfinite(y["encoder"]).all())}
            del model
            purge_cache()
            print(u, fn, out[u][fn], flush=True)
    return out


def cmd_probe(args) -> None:
    from mil import build

    localflow.apply_buckets(localflow.ALL_BUCKETS)
    units = [u for u in args.units.split(",") if u]
    for arm in arms_arg(args.arms, reference=True):
        t0 = time.time()
        m = build.build_encoder("mb1", arm, "multi", art() / "probes", layers=args.layers, compile_=True, plan=False,
                                tag="-probe")
        mlmodelc = Path(m["paths"]["mlmodelc"])
        doc = {"arm": arm, "layers": args.layers, "functions": localflow.FUNCTIONS,
               "chain": m["encoding"]["chain"], "encoded_bytes_total": m["encoding"]["encoded_bytes_total"],
               "constexpr_literals": {fn: d["constexpr_inputs"] for fn, d in m["program"].items()},
               "matmul_followed_by_mul": {fn: d["matmul_followed_by_mul"] for fn, d in m["program"].items()},
               "sizes": m["sizes"], "mlmodelc_bytes": m["compile"]["mlmodelc_bytes"], "timing_s": m["timing_s"],
               "compile_s": m["compile"]["compile_s"], "peak_rss_mb": m.get("peak_rss_mb_after_compile")}
        g2 = gate2_package(Path(m["paths"]["mlpackage"]), arm)
        doc["gate2"] = {"bit_exact": g2["bit_exact"], "chains": g2["chains"],
                        "modules_checked": {k: v["modules_checked"] for k, v in g2["functions"].items()}}
        doc["compute_plans"] = plans(mlmodelc, args.plan_units.split(",")) if args.plan_units else {}
        doc["timings"] = timings(mlmodelc, units, args.calls) if units else {}
        doc["wall_s"] = round(time.time() - t0, 1)
        write(f"probes/{arm}-{args.layers}L.json", doc)
        if not args.keep:
            shutil.rmtree(Path(m["paths"]["mlpackage"]).parent, ignore_errors=True)
        print(json.dumps({"arm": arm, "gate2": doc["gate2"]["bit_exact"],
                          "median_ms": {u: {fn: v["median_ms"] for fn, v in t.items()} for u, t in doc["timings"].items()}}),
              flush=True)


def probe_model(arm: str, layers: int) -> Path:
    return art() / "probes" / "mb1" / arm / "multi-probe.mlmodelc"


def cmd_bench(args) -> None:
    """Round-robin probe timings of kept probe builds (`probe --keep`): every round visits every arm (order
    rotated per round) and function; per (arm, function) the median over rounds of the per-round medians."""
    import coremltools as ct

    localflow.apply_buckets(localflow.ALL_BUCKETS)
    arms = arms_arg(args.arms, reference=True)
    u = args.unit
    rounds: dict = {a: {fn: [] for fn in localflow.FUNCTIONS} for a in arms}
    loads: dict = {a: {} for a in arms}
    for r in range(args.rounds):
        order = arms[r % len(arms):] + arms[:r % len(arms)]
        for arm in order:
            for fn in localflow.FUNCTIONS:
                t0 = time.time()
                model = ct.models.CompiledMLModel(str(probe_model(arm, args.layers)), compute_units=cu(u), function_name=fn)
                loads[arm].setdefault(fn, []).append(round(time.time() - t0, 2))
                x = mel_input(fn)
                for _ in range(3):
                    model.predict(x)
                ms = []
                for _ in range(args.calls):
                    t = time.perf_counter()
                    model.predict(x)
                    ms.append((time.perf_counter() - t) * 1e3)
                rounds[arm][fn].append(round(statistics.median(ms), 3))
                del model
        print("round", r, {a: {fn: v[-1] for fn, v in rounds[a].items()} for a in arms}, flush=True)
    purge_cache()
    doc = {"unit": u, "layers": args.layers, "rounds": args.rounds, "calls_per_round": args.calls,
           "note": "informational screening on the shared Mac: Python CompiledMLModel.predict, synthetic mel, "
                   f"{args.layers}-layer MB1 encoders (LocalFlow buckets), round-robin over arms",
           "median_ms": {a: {fn: round(statistics.median(v), 2) for fn, v in rounds[a].items()} for a in arms},
           "per_round_ms": rounds, "load_s": loads}
    write(args.out or f"probes/bench-{args.layers}L-{u}.json", doc)
    print(json.dumps(doc["median_ms"], indent=1))


def cmd_plan(args) -> None:
    localflow.apply_buckets(localflow.ALL_BUCKETS)
    doc = {"arm": args.arm, "path": args.path, "compute_plans": plans(Path(args.path), args.units.split(","))}
    write(args.out or f"plans/{args.arm}.json", doc)
    print(json.dumps({u: {fn: (v.get("ops_by_device"), v.get("cost_share")) for fn, v in p.items() if fn != "_wall_s"}
                      for u, p in doc["compute_plans"].items()}))


def cmd_loadtimes(args) -> None:
    """First and second load per function (own cache emptied before the first), the cache size after the first
    load, and a short timing check (informational)."""
    import coremltools as ct

    localflow.apply_buckets(localflow.ALL_BUCKETS)
    path = Path(args.path)
    out = {}
    for u in args.units.split(","):
        for fn in (args.functions.split(",") if args.functions else localflow.FUNCTIONS):
            purge_cache()
            t0 = time.time()
            model = ct.models.CompiledMLModel(str(path), compute_units=cu(u), function_name=fn)
            first = time.time() - t0
            cache_mb = dir_mb(own_cache())
            del model
            t0 = time.time()
            model = ct.models.CompiledMLModel(str(path), compute_units=cu(u), function_name=fn)
            second = time.time() - t0
            x = mel_input(fn)
            for _ in range(2):
                model.predict(x)
            ms = []
            for _ in range(args.calls):
                t = time.perf_counter()
                y = model.predict(x)
                ms.append((time.perf_counter() - t) * 1e3)
            out.setdefault(u, {})[fn] = {"first_load_s": round(first, 1), "second_load_s": round(second, 2),
                                         "cache_mb_after_first_load": cache_mb,
                                         "median_ms": round(statistics.median(ms), 2), "calls": args.calls,
                                         "output_finite": bool(np.isfinite(y["encoder"]).all())}
            del model
            purge_cache()
            print(u, fn, out[u][fn], flush=True)
            time.sleep(args.pause)
    write(args.out or f"loadtimes/{args.arm}.json", {"arm": args.arm, "path": str(path), "loads": out,
                                                      "note": "informational: shared Mac; synthetic mel input; Python predict"})


# --- B1P headroom: activations on public clips and Core ML stress ------------------------------------------

def reference_mb1():
    import reference
    from mil.binary_export import BinaryExportSource
    from mil.weights import default_path

    source = BinaryExportSource(default_path("mb1"))
    model = reference.build(reference.Config.from_model_config(source.model_config))
    report = reference.load_weights(model, source)
    model.eval()
    return model, source, report


def cmd_activations(args) -> None:
    """FP32 reference of MB1 on public LibriSpeech dev-clean clips: per projection, max |x| of its input, the
    pre-scale B1P matmul max |x . c| (c = +-1 codes; what B1P's FP16 matmul must hold before the row scale) and
    the true output max |x . (c s)|. Captures layer 0 / 23 inputs of the longest clip for `stress`."""
    import torch

    import clips as clipmod
    from mil import refcache
    from mil.weights import TERNARY_SUFFIXES

    torch.set_grad_enabled(False)
    t0 = time.time()
    model, source, _ = reference_mb1()
    manifest = clipmod.load_manifest()
    naturals = [c for c in manifest["clips"] if c["kind"] == "natural"]
    chosen = []
    for b in (2, 4, 8, 15):
        chosen += sorted((c for c in naturals if c["bucket"] == b), key=lambda c: c["id"])[:args.per_bucket]
    probe = max((c for c in naturals if c["bucket"] == 15), key=lambda c: (c["length"], c["id"]))
    if probe not in chosen:
        chosen.append(probe)
    pcm_dir = refcache.default_pcm()
    codes = {}
    stats: dict[str, dict] = {}
    captured = {}
    current = {"probe": False}
    hooks = []
    sites = [s for s in TERNARY_SUFFIXES if s != "self_attn.linear_pos"]
    for i, layer in enumerate(model.encoder.layers):
        for suffix in sites:
            key = f"encoder.layers.{i}.{suffix}"
            c, s = source.binary(key)
            codes[key] = (c.astype(np.float64), s.astype(np.float16).astype(np.float64))

            def hook(mod, inputs, key=key, suffix=suffix, i=i):
                x = inputs[0][0]
                x = (x.transpose(0, 1) if suffix.startswith("conv.") else x).double().numpy()  # [T, in]
                c, s = codes[key]
                pre = x @ c.T
                st = stats.setdefault(key, {"max_abs_x": 0.0, "max_abs_pre_scale": 0.0, "max_abs_out": 0.0,
                                            "max_l1_x": 0.0})
                st["max_abs_x"] = max(st["max_abs_x"], float(np.abs(x).max()))
                st["max_l1_x"] = max(st["max_l1_x"], float(np.abs(x).sum(axis=1).max()))
                st["max_abs_pre_scale"] = max(st["max_abs_pre_scale"], float(np.abs(pre).max()))
                st["max_abs_out"] = max(st["max_abs_out"], float(np.abs(pre * s[None, :]).max()))
                if current["probe"] and i in (0, len(model.encoder.layers) - 1):
                    captured[key.replace(".", "_")] = x.astype(np.float32)
            hooks.append(layer.get_submodule(suffix).register_forward_pre_hook(hook))
    for clip in chosen:
        current["probe"] = clip is probe
        pcm = clipmod.read_pcm(pcm_dir, clip)
        if clipmod.pcm_sha256(pcm) != clip["sha256"]:
            raise ValueError(f"{clip['id']}: PCM SHA-256 differs from clips.json")
        feats, flen = model.preprocessor(torch.from_numpy(pcm.astype(np.float32))[None], torch.tensor([len(pcm)]))
        model.encoder(feats, flen)
    for h in hooks:
        h.remove()
    out_dir = art() / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "probe_inputs.npz", **captured)
    worst_pre = max(stats.items(), key=lambda kv: kv[1]["max_abs_pre_scale"])
    worst_out = max(stats.items(), key=lambda kv: kv[1]["max_abs_out"])
    ratio = {k: v["max_abs_pre_scale"] / max(v["max_abs_out"], 1e-30) for k, v in stats.items()}

    def over(field, factor):
        return sorted(k for k, v in stats.items() if v[field] * factor > FP16_MAX)

    doc = {"check": "B1P FP16 headroom: FP32 reference of MB1 on public LibriSpeech dev-clean clips; per arm-encoded "
                    "projection, max |x . c| before the row scale (B1P's FP16 matmul output) vs max |x . (c s)| "
                    "(the scaled output every pre-scaled arm produces); FP16 max 65504",
           "clips": [c["id"] for c in chosen], "probe_clip": probe["id"], "modules": len(stats),
           "worst_pre_scale": {"module": worst_pre[0], **worst_pre[1]},
           "worst_scaled_output": {"module": worst_out[0], **worst_out[1]},
           "pre_scale_over_output_ratio": {"min": min(ratio.values()), "median": statistics.median(ratio.values()),
                                           "max": max(ratio.values())},
           "modules_overflowing_fp16": {f"{field}_x{f}": len(over(field, f)) for field in ("max_abs_pre_scale", "max_abs_out")
                                        for f in (1, 8, 64)},
           "modules_overflowing_fp16_list_pre_scale_x8": over("max_abs_pre_scale", 8)[:40],
           "per_module": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in sorted(stats.items())},
           "seconds": round(time.time() - t0, 1)}
    write("b1p-activations.json", doc)
    print(json.dumps({k: doc[k] for k in ("worst_pre_scale", "worst_scaled_output", "pre_scale_over_output_ratio",
                                          "modules_overflowing_fp16")}, indent=1))


STRESS_MODULES = ("encoder.layers.0.feed_forward1.linear1", "encoder.layers.0.feed_forward1.linear2",
                  "encoder.layers.23.feed_forward2.linear2")


def _finite_json(value):
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_json(v) for v in value]
    return value


def cmd_stress(args) -> None:
    """WP3's stress probe on MB1: one matmul per arm with adversarial +-1 rows (all +1, all -1, runs of 256,
    halves), inputs = the probe clip's activation x1, x8, x64 and |x| x64, per compute unit. Rule (as WP3's,
    with B1S as the C4 analogue): an arm fails if it is non-finite where B1S stays finite, or at x1."""
    import coremltools as ct

    from mil import build, probes
    from mil.weights import Source, fp16_scale

    source = Source("mb1")
    captured = np.load(art() / "activations" / "probe_inputs.npz")
    arms = arms_arg(args.arms)
    if "B1S" not in arms:
        arms = ["B1S"] + arms
    units = args.units.split(",")
    work = art() / "stress"
    work.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for module in STRESS_MODULES:
        codes, scale = source.ternary(module)
        codes = probes.adversarial(codes)
        x0 = captured[module.replace(".", "_")]
        w64 = codes.astype(np.float64) * fp16_scale(scale).astype(np.float64)[:, None]
        inputs = {"x1": x0, "x8": 8 * x0, "x64": 64 * x0, "abs_x64": 64 * np.abs(x0)}
        for arm in arms:
            prog = probes._stress_program(arm, codes, scale, x0.shape[0])
            model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS26,
                               compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_ONLY,
                               skip_model_load=True)
            pkg = work / f"{module}-{arm}.mlpackage"
            if pkg.exists():
                shutil.rmtree(pkg)
            model.save(str(pkg))
            build.drop_temp_package(model)
            comp = build.compile_model(pkg, work / f"{module}-{arm}.mlmodelc")
            for u in units:
                m = ct.models.CompiledMLModel(comp["mlmodelc"], compute_units=cu(u))
                for name, x in inputs.items():
                    pred = m.predict({"x": x[None]})
                    ref = x.astype(np.float64) @ w64.T
                    row = {"finite": all(bool(np.isfinite(v).all()) for v in pred.values()),
                           "max_abs": {k: (float(np.abs(v[np.isfinite(v)]).max()) if np.isfinite(v).any() else None)
                                       for k, v in pred.items()},
                           "nonfinite": {k: int((~np.isfinite(v)).sum()) for k, v in pred.items()},
                           "reference_max_abs_out": float(np.abs(ref).max()),
                           "reference_max_abs_pre_scale": float(np.abs(x.astype(np.float64) @ codes.T.astype(np.float64)).max())}
                    results.setdefault(module, {}).setdefault(arm, {}).setdefault(u, {})[name] = row
                del m
                purge_cache()
            print(module, arm, {u: {i: r["finite"] for i, r in d.items()} for u, d in results[module][arm].items()}, flush=True)
    verdict = {}
    for arm in arms:
        for u in units:
            fails, x1 = [], []
            for module in results:
                for level, r in results[module][arm][u].items():
                    ref = results[module]["B1S"][u][level]
                    if not r["finite"] and ref["finite"]:
                        fails.append(f"{module} {level}")
                    if level == "x1" and not r["finite"]:
                        x1.append(module)
            verdict.setdefault(arm, {})[u] = {"pass": not fails and not x1, "nonfinite_where_B1S_finite": fails,
                                             "nonfinite_at_x1": x1,
                                             "nonfinite_cases": sum(not r["finite"] for mod in results
                                                                    for r in results[mod][arm][u].values())}
    shutil.rmtree(work, ignore_errors=True)
    doc = {"check": cmd_stress.__doc__, "units": units, "modules": list(STRESS_MODULES), "verdict": verdict,
           "results": _finite_json(results)}
    write("b1p-stress.json", doc)
    print(json.dumps(verdict, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gate2-modules")
    p.add_argument("--arms")
    p.set_defaults(func=cmd_gate2_modules)
    p = sub.add_parser("probe")
    p.add_argument("--arms")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--units", default="cpuAndNeuralEngine", help="timing units ('' for none)")
    p.add_argument("--plan-units", default=",".join(UNITS), help="compute plan units ('' for none)")
    p.add_argument("--calls", type=int, default=10)
    p.add_argument("--keep", action="store_true")
    p.set_defaults(func=cmd_probe)
    p = sub.add_parser("bench")
    p.add_argument("--arms")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--unit", default="cpuAndNeuralEngine")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--calls", type=int, default=10)
    p.add_argument("--out")
    p.set_defaults(func=cmd_bench)
    p = sub.add_parser("gate2-package")
    p.add_argument("--arm", required=True)
    p.add_argument("--package", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_gate2_package)
    p = sub.add_parser("plan")
    p.add_argument("--arm", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--units", default=",".join(UNITS))
    p.add_argument("--out")
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("loadtimes")
    p.add_argument("--arm", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--units", default="cpuAndNeuralEngine")
    p.add_argument("--functions", default="")
    p.add_argument("--calls", type=int, default=5)
    p.add_argument("--pause", type=float, default=20.0)
    p.add_argument("--out")
    p.set_defaults(func=cmd_loadtimes)
    p = sub.add_parser("activations")
    p.add_argument("--per-bucket", type=int, default=2)
    p.set_defaults(func=cmd_activations)
    p = sub.add_parser("stress")
    p.add_argument("--arms", default="B1S,B1P,B1L")
    p.add_argument("--units", default=",".join(UNITS))
    p.set_defaults(func=cmd_stress)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
