"""gpu1bit benchmark harness (plans/gpu-kernel.md A6, A9, A13, A14), M1's recorded methodology by import.

PROTOCOLS
P-kernel  per op as the pipeline calls it, warm CUDA-graph replay: M1's micro method,
          median over 3 windows of triton.testing.do_bench_cudagraph(fn, rep=25) (cuda_experiments.py
          micro, ptx_cuda.py), in us. Plus L2-COLD (A14): graphs [flush] and [flush, op], each
          replayed separately with CUDA events around replay; cold = median(flush+op) - median(flush);
          the flush writes 2 x 128 MiB (> the 5090's L2) so weights AND activations come from DRAM.
          Ops (per Conformer block, real MB1 layer-0 weights, real captured dev inputs at the dev
          clips' M = 38 / 126 / 368, workloads.json):
            qkv  linear 3072x1024, norm fused (norm_self_att)      ff1 linear 4096x1024, norm + SiLU out
            out  linear 1024x1024                                   ff2 linear 1024x4096
            pw1  conv 2048x1024, norm fused, transposed input       pw2 conv 1024x1024, SiLU on input
            pos  linear 1024x1024 at M = 2T-1 (outside M1's graph; reported, not in totals)
          The fixture input is the layer's post-norm module input; ops with a fused norm normalise it
          again (same work, values stay in range). Variants and what each op includes:
            r_compact   inference.fused_kernels.int8_matmul (R-compact): INT8x3 quantize (+norm/+SiLU
                        in), 2-bit tile decode INT8 dot, split-K finish (+SiLU out)
            r_expanded  inference.encoder4.ExpandedProjection("triton_col_tuned") (R-expanded): the
                        same with the INT8 column-major cache
            r_hybrid    inference.ptx_cuda.PtxCudaProjection(tuning="native", weight_bits=8,
                        row_warps="adaptive") (R-hybrid's projection, not its deferred row epilogue)
            r_bf16x3    F.layer_norm / F.silu (unfused) + inference.kernels.matmul(mode="bf16x3",
                        activation fused) (R-bf16x3, kernel level)
            <candidate> gpu1bit.shared.provider.BinaryProjection over the candidate's layer: its
                        `project` op if it has one, else composed (flagged)
P-paired  (CUDA_IMPROVEMENTS.md) dev_24: one warmed shape at a time, caches cleared between
          utterances, three balanced 1 s windows per variant per utterance alternating order, NVML
          energy over exactly the timed replays (EnergyMeter.measure synchronises start and stop),
          median window, paired changes with counts; encoder scope (GPU features -> encoder output,
          raw graph replay, inference.benchmark.capture) and full scope (CPU waveform -> CPU text).
P-clips   (EXPERIMENTS4.md / compare_v2.py) fresh process per variant, dev clips 3 / 10 / 30 s in that
          order, accumulating pipeline graph cache, 2 s warm-up then three 5 s windows, NVML energy.
P-memory  the P-clips process samples its own NVML process VRAM every 10 ms from before loading,
          including load and first capture, as compare_v2.py does.
Workloads: dev only (A13), except `--workloads historical-test --dev-run RUN` for P-paired and P-clips:
          M1's historical test workloads (gpu1bit/x2/historical.py), refused until the x2 benchmark run
          RUN has written its dev results. P-kernel `--run RUN` writes bench/<RUN>/kernel.json; a
          part-count layer variant ("<kernel>:x<N>") is timed at N activation parts (part_count).
Every run: one process, one gpu_lock (A9), started as `timeout 600 gpu1bit/shared/python ...`;
power limit asserted == 400 W and no foreign compute process (EnergyMeter guard) before timing;
environment facts recorded and compared across variants (A7); outputs under
/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/<agent>/bench/.

    timeout 600 gpu1bit/shared/python -m gpu1bit.shared.bench kernel --agent A --impl triton_bf16 [--baselines] [--M 38 126 368] [--ops qkv ff1 ...]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from gpu1bit.shared import api, env

OPS = {  # name: (kind, N, K, fixture layer providing x, norm module, activation, input_activation)
    "qkv": ("linear", 3072, 1024, "encoder.layers.0.self_attn.linear_q", "encoder.layers.0.norm_self_att", "", ""),
    "out": ("linear", 1024, 1024, "encoder.layers.0.self_attn.linear_out", None, "", ""),
    "ff1": ("linear", 4096, 1024, "encoder.layers.0.feed_forward1.linear1", "encoder.layers.0.norm_feed_forward1", "silu", ""),
    "ff2": ("linear", 1024, 4096, "encoder.layers.0.feed_forward1.linear2", None, "", ""),
    "pw1": ("conv", 2048, 1024, "encoder.layers.0.conv.pointwise_conv1", "encoder.layers.0.norm_conv", "", ""),
    "pw2": ("conv", 1024, 1024, "encoder.layers.0.conv.pointwise_conv2", None, "", "silu"),
    "pos": ("linear", 1024, 1024, "encoder.layers.0.self_attn.linear_pos", None, "", ""),
}
PER_BLOCK = {"qkv": 1, "out": 1, "ff1": 2, "ff2": 2, "pw1": 1, "pw2": 1}   # calls per Conformer block in the graph
BASELINES = ("r_compact", "r_expanded", "r_hybrid", "r_bf16x3")
FLUSH_BYTES = 2 * 128 * 1024 * 1024


# --- environment (A7, A9) ------------------------------------------------------------------------

def environment() -> dict:
    import triton
    facts = {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
             "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
             "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
             "float32_matmul_precision": torch.get_float32_matmul_precision(), "wrapper": env.wrapper()}
    return facts


def meter_and_checks():
    """M1's EnergyMeter (inference.benchmark.energy_meter); asserts 400 W exactly, no foreign PIDs."""
    from inference.benchmark import energy_meter
    meter = energy_meter()
    # A previous lock holder can still be tearing down its CUDA context just after releasing the
    # lock at exit; wait up to 20 s for foreign compute PIDs to vanish, then refuse (A9).
    deadline = time.monotonic() + 20
    while meter.foreign_processes() and time.monotonic() < deadline:
        time.sleep(0.5)
    meter._guard()
    limit = meter._nvml.nvmlDeviceGetPowerManagementLimit(meter._handle) / 1000
    if limit != env.POWER_LIMIT_W:
        meter.close()
        raise RuntimeError(f"GPU power limit is {limit} W, expected exactly {env.POWER_LIMIT_W} W (never changed here)")
    return meter


def check_power(meter) -> float:
    meter._guard()
    limit = meter._nvml.nvmlDeviceGetPowerManagementLimit(meter._handle) / 1000
    if limit != env.POWER_LIMIT_W:
        raise RuntimeError(f"GPU power limit changed to {limit} W")
    return limit


# --- P-kernel ------------------------------------------------------------------------------------

def _norm_module(name: str | None):
    if name is None:
        return None
    from safetensors import safe_open
    from gpu1bit.shared import fixtures
    with safe_open(env.MB1_EXPORT / fixtures.mb1_manifest()["file"], framework="pt") as f:
        w, b = f.get_tensor(f"{name}.weight").float(), f.get_tensor(f"{name}.bias").float()
    norm = torch.nn.LayerNorm(w.numel(), eps=1e-5).cuda().eval()
    with torch.no_grad():
        norm.weight.copy_(w)
        norm.bias.copy_(b)
    for p in norm.parameters():
        p.requires_grad_(False)
    return norm


def op_weights(op: str) -> dict:
    """MB1 layer-0 weights of an op (Q/K/V concatenated for qkv)."""
    from gpu1bit.shared import fixtures
    if op == "qkv":
        q = fixtures.mb1_layers([f"encoder.layers.0.self_attn.linear_{p}" for p in "qkv"])
        return {"codes": torch.cat([v["codes"] for v in q.values()]), "scale": torch.cat([v["scale"] for v in q.values()]),
                "bias": None, "K": 1024, "N": 3072}
    w = fixtures.mb1_layer(OPS[op][3])
    return {k: w[k] for k in ("codes", "scale", "bias", "K", "N")}


def op_input(op: str, utterance: int) -> torch.Tensor:
    from gpu1bit.shared import fixtures
    return fixtures.utterance_inputs(utterance)[OPS[op][3]]


def op_reference(op, x, w, norm):
    """FP64 composite: norm -> input SiLU -> alpha W z + b -> SiLU (the provider op's semantics)."""
    kind, _, _, _, _, act, in_act = OPS[op]
    z = (x.transpose(1, 2) if kind == "conv" else x).double()
    if norm is not None:
        z = F.layer_norm(z, norm.normalized_shape, norm.weight.double(), norm.bias.double(), norm.eps)
    if in_act == "silu":
        z = F.silu(z)
    y = api.reference_linear(z, w["codes"], w["scale"], w["bias"], w["K"])
    if act == "silu":
        y = F.silu(y)
    return y.transpose(1, 2) if kind == "conv" else y


def baseline_op(variant: str, op: str, w: dict, norm):
    """A callable x -> y running baseline `variant`'s op on the derived 2-bit weights."""
    from gpu1bit.shared import m1
    kind, n, k, _, _, act, in_act = OPS[op]
    conv = kind == "conv"
    module = m1.derived_packed_linear(w["codes"], w["scale"], w["bias"], k, conv=conv)
    mod = types.SimpleNamespace(packed_t=module.packed_t, scale=module.scale, bias=module.bias, out_features=n) \
        if op == "qkv" else module
    if variant == "r_compact":
        from inference.fused_kernels import int8_matmul
        return lambda x: int8_matmul(x, mod, conv, 3, act, norm, None, in_act), module
    if variant == "r_expanded":
        from inference.encoder4 import ExpandedProjection
        p = ExpandedProjection("triton_col_tuned")
        return lambda x: p(x, mod, conv, 3, act, norm, None, in_act), (module, p)
    if variant == "r_hybrid":
        from inference.ptx_cuda import PtxCudaProjection
        p = PtxCudaProjection(None, tuning="native", weight_bits=8, row_warps="adaptive")
        return lambda x: p(x, mod, conv, 3, act, norm, None, in_act), (module, p)
    if variant == "r_bf16x3":
        from inference.kernels import matmul

        def run(x):
            if norm is None and not in_act:
                return matmul(x, module.packed_t, module.scale, module.bias, k, conv=conv, mode="bf16x3", activation=act)
            z = x.transpose(1, 2) if conv else x
            if norm is not None:
                z = F.layer_norm(z, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
            if in_act:
                z = F.silu(z)
            y = matmul(z, module.packed_t, module.scale, module.bias, k, mode="bf16x3", activation=act)
            return y.transpose(1, 2) if conv else y
        return run, module
    raise ValueError(variant)


def part_count(name: str) -> int:
    """The activation part count of a registry name (shared/seldoc.py part_count)."""
    from gpu1bit.shared import seldoc
    return seldoc.part_count(name)


def candidate_op(impl, op: str, w: dict, norm, components: int = 3):
    """The candidate's op as the pipeline calls it, at `components` activation parts (a part-count layer
    variant falls back to its x3 path when called with 3; plans/x2-default.md, P-kernel of the x2 benchmark)."""
    from gpu1bit.shared.provider import BinaryProjection, Table, _handle
    kind, n, k, _, _, act, in_act = OPS[op]
    layer = impl.prepare(w["codes"], w["scale"], w["bias"], k, "cuda")
    table = Table()
    table.add(op, layer, kind, n, k)
    prov = BinaryProjection(table)
    mod = types.SimpleNamespace(packed_t=_handle(0, n))
    fused = hasattr(layer, "project")
    return (lambda x: prov(x, mod, kind == "conv", components, act, norm, None, in_act)), (layer, prov), fused


def _cold_us(fn, x, flush, repeats: int = 200) -> dict:
    """median(event time of [flush, op] replay) - median([flush] replay), in us."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(x)
        flush.zero_()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g_flush, g_both = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_flush):
        flush.zero_()
    with torch.cuda.graph(g_both):
        flush.zero_()
        fn(x)
    times = {"flush": [], "both": []}
    for i in range(repeats):
        for name, g in (("flush", g_flush), ("both", g_both)) if i % 2 == 0 else (("both", g_both), ("flush", g_flush)):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            g.replay()
            end.record()
            end.synchronize()
            times[name].append(start.elapsed_time(end) * 1000)
    del g_flush, g_both
    return {"cold_us": statistics.median(times["both"]) - statistics.median(times["flush"]),
            "flush_us": statistics.median(times["flush"]), "flush_plus_op_us": statistics.median(times["both"]),
            "repeats": repeats}


def p_kernel(args) -> Path:
    import triton.testing
    from gpu1bit.shared import registry, workloads
    env.require_shared_wrapper()
    clips = workloads.load()["dev_clips"]
    fixture_of_m = {}
    from gpu1bit.shared import fixtures
    for i, u in enumerate(fixtures.index()["utterances"]):
        fixture_of_m.setdefault(u["M"], i)
    variants = (list(BASELINES) if args.baselines else []) + list(args.impl or [])
    if any(":x" in v for v in args.impl or []):   # part-count layer variants: B's hash-pinned files first (F5)
        from gpu1bit.actprec import parts as AP
        AP.load_parts()
    out_dir = env.ARTIFACTS_ROOT / args.agent / "bench"
    env.require_mount()
    if getattr(args, "run", None):                 # run-scoped (a session step): bench/<run>/kernel.json
        out_dir = out_dir / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / ("kernel.json" if getattr(args, "run", None) else f"{time.strftime('%Y%m%dT%H%M%S')}-kernel.json")
    report = {"protocol": "P-kernel", "doc": __doc__.split("P-paired")[0], "args": vars(args), "environment": None,
              "dev_clips": [{k: c[k] for k in ("id", "duration", "M")} for c in clips], "rows": []}
    with env.gpu_lock(f"{args.agent}:P-kernel") as lock, torch.inference_mode():
        torch.backends.cuda.matmul.allow_tf32 = False
        report["environment"] = environment()
        meter = meter_and_checks()
        report["gpu"] = meter.metadata()
        report["lock"] = {k: lock[k] for k in ("label", "waited_s", "pid")}
        flush = torch.empty(FLUSH_BYTES // 4, device="cuda")
        for op in args.ops:
            kind, n, k, _, norm_name, act, in_act = OPS[op]
            w = op_weights(op)
            norm = _norm_module(norm_name)
            makers = {}
            for v in variants:
                if v in BASELINES:
                    makers[v] = (lambda v=v: baseline_op(v, op, w, norm) + (None,))
                else:
                    makers[v] = (lambda v=v: candidate_op(registry.get(v), op, w, norm, components=part_count(v)))
            for m in args.M:
                env.check_hold(lock)
                check_power(meter)
                utterance = fixture_of_m[m]
                x = op_input(op, utterance)
                ref = op_reference(op, x, w, norm)
                row = {"op": op, "kind": kind, "N": n, "K": k, "M": x.numel() // k, "clip_M": m,
                       "fixture": fixtures.index()["utterances"][utterance]["id"], "x_stride": list(x.stride()),
                       "norm": norm_name, "activation": act, "input_activation": in_act, "variants": {}}
                for v, make in makers.items():
                    fn, keep, fused = make()
                    y = fn(x)
                    torch.cuda.synchronize()
                    rel = api.relative_error(y, ref)
                    warm = [triton.testing.do_bench_cudagraph(lambda: fn(x), rep=25) * 1000 for _ in range(3)]
                    cold = _cold_us(fn, x, flush)
                    row["variants"][v] = {"warm_us": statistics.median(warm), "warm_windows_us": warm, **cold,
                                          "rel_frobenius_vs_fp64": rel,
                                          "fused_project_op": fused if v not in BASELINES else True}
                    del fn, keep
                    torch.cuda.empty_cache()
                report["rows"].append(row)
                print(json.dumps({"op": op, "M": row["M"], **{v: round(r["warm_us"], 2) for v, r in row["variants"].items()},
                                  "cold": {v: round(r["cold_us"], 2) for v, r in row["variants"].items()}}), flush=True)
                out.write_text(json.dumps(report, indent=1) + "\n")
        report["lock"]["held_s"] = env.check_hold(lock, margin_s=0)
        report["power_limit_w_end"] = check_power(meter)
        meter.close()
    report["totals_per_block_us"] = _block_totals(report["rows"], variants)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print("RESULTS", out)
    return out


def _block_totals(rows, variants) -> dict:
    """Per clip M and variant: sum over one Conformer block's in-graph projection calls (PER_BLOCK)."""
    totals = {}
    for m in sorted({r["clip_M"] for r in rows}):
        sel = [r for r in rows if r["clip_M"] == m and r["op"] in PER_BLOCK]
        if {r["op"] for r in sel} != set(PER_BLOCK):
            continue
        totals[str(m)] = {v: {"warm_us": sum(PER_BLOCK[r["op"]] * r["variants"][v]["warm_us"] for r in sel),
                              "cold_us": sum(PER_BLOCK[r["op"]] * r["variants"][v]["cold_us"] for r in sel)}
                          for v in variants if all(v in r["variants"] for r in sel)}
    return totals


# --- model variants (A1, A2, A11) -----------------------------------------------------------------

@contextlib.contextmanager
def variant_model(name: str, *, max_graphs: int = 4):
    """A FRESH model running variant `name` for the block: R-compact, R-expanded, R-hybrid
    (MB1 derived 2-bit on M1's runtimes) or a registry candidate through BinaryProjection
    (controlled substitution: M1's enable_optimizations(expanded) stack, provider swapped)."""
    import evaluate as ev
    from gpu1bit.shared import m1, provider as P, registry
    if name.startswith("R-"):
        model = m1.load_derived()
        with ev.inference_settings(model):
            model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
            with m1.baseline(name, model, max_graphs=max_graphs) as extra:
                yield model, {"kind": "baseline", "stack": name, "projection": extra}
    else:
        model, table = P.load_binary_model(env.MB1_EXPORT, registry.get(name))
        with ev.inference_settings(model):
            model.decoding.decoding.decoding_computer.force_cuda_graphs_mode("full_graph")
            prov = P.BinaryProjection(table, P.module_names(model))
            P.enable_binary_optimizations(model, prov, max_graphs=max_graphs)
            try:
                yield model, {"kind": "candidate", "stack": "enable_optimizations(expanded) + BinaryProjection",
                              "projection": prov, "weight_bytes": table.weight_bytes(),
                              "source_sha256": env.sha256_file(registry.source_path(name))}
            finally:
                from inference.optimized import disable_optimizations
                disable_optimizations(model)


def variant_facts(model) -> dict:
    """A7: per-variant settings that must agree across variants."""
    comp = model.decoding.decoding.decoding_computer
    feat = model.preprocessor.featurizer
    return {"cuda_graphs_mode": str(getattr(comp, "cuda_graphs_mode", None)),
            "decoder_fused": type(model.decoding.decoding.decoding_computer).__name__,
            "pipeline_graphs": type(model.forward).__name__, "dither": float(feat.dither), "pad_to": int(feat.pad_to),
            **{k: v for k, v in environment().items() if k != "wrapper"}}


def recognize(model, audio, length):
    out = model(input_signal=audio.cuda(), input_signal_length=length.cuda())
    result = model.decoding.rnnt_decoder_predictions_tensor(out[0], out[1], return_hypotheses=False)
    if isinstance(result, tuple):
        result = result[0]
    return result[0].text if hasattr(result[0], "text") else str(result[0])


def decoded_state(model) -> dict:
    bh = model.decoding.decoding.decoding_computer.state.batched_hyps
    n = int(bh.current_lengths[0])
    return {"tokens": bh.transcript[0, :n].tolist(), "timestamps": bh.timestamps[0, :n].tolist()}


# --- P-paired ------------------------------------------------------------------------------------

WORKLOAD_SETS = ("dev", "historical-test")


def workload_records(args, key: str) -> list[dict]:
    """The workloads of a P-paired ("dev_24") or P-clips ("dev_clips") process: the frozen dev workloads
    (A13), or, with --workloads historical-test, M1's historical test workloads (gpu1bit.x2.historical:
    the 24 multi-corpus test utterances / the LibriSpeech test-clean clips), which are refused until the
    named x2 benchmark run's dev results are written."""
    if getattr(args, "workloads", "dev") == "dev":
        from gpu1bit.shared import workloads
        return workloads.load()[key]
    from gpu1bit.x2 import historical
    if not getattr(args, "dev_run", None):
        raise SystemExit("--workloads historical-test needs --dev-run (A13)")
    return historical.records(args.dev_run)["utterances" if key == "dev_24" else "clips"]


def p_paired(args) -> Path:
    """One chunk of dev_24 utterances, all variants resident, balanced windows (see module doc)."""
    import evaluate as ev
    from nemo.utils import logging
    from inference.benchmark import capture
    env.require_shared_wrapper()
    logging.setLevel(logging.ERROR)
    records = workload_records(args, "dev_24")
    a, b = (int(v) for v in args.utterances.split("-"))
    chosen = list(range(a, b + 1))
    names = args.variants
    out_dir = env.ARTIFACTS_ROOT / args.agent / "bench" / args.run
    env.require_mount()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"paired-{args.scope}-{a:02d}-{b:02d}.json"
    report = {"protocol": "P-paired", "scope": args.scope, "args": vars(args), "variants": names, "rows": []}
    with env.gpu_lock(f"{args.agent}:P-paired {args.scope} {a}-{b}") as lock, torch.inference_mode(), \
            contextlib.ExitStack() as stack:
        import evaluate as ev2
        stack.enter_context(ev2.strict_fp32("cuda"))
        report["lock"] = {k: lock[k] for k in ("label", "waited_s", "pid")}
        meter = stack.enter_context(contextlib.closing(meter_and_checks()))
        report["gpu"] = meter.metadata()
        models, info = {}, {}
        for name in names:
            models[name], info[name] = stack.enter_context(variant_model(name, max_graphs=4 if args.scope == "full" else 0))
        report["variant_facts"] = {n: variant_facts(m) for n, m in models.items()}
        first = report["variant_facts"][names[0]]
        for n, facts in report["variant_facts"].items():
            diff = {k for k in facts if k not in ("pipeline_graphs",) and facts[k] != first[k]}
            if diff:
                raise RuntimeError(f"A7: {n} differs from {names[0]} in {sorted(diff)}")
        report["variants_info"] = {n: {k: v for k, v in i.items() if k != "projection"} for n, i in info.items()}
        for index in chosen:
            rec = records[index]
            env.check_hold(lock)
            audio, length = ev.audio_batch([rec])
            windows = {n: [] for n in names}
            checks = {}
            if args.scope == "encoder":
                features, flen = models[names[0]].preprocessor(input_signal=audio.cuda(), length=length.cuda())
                graphs, outputs = {}, {}
                for n in names:
                    g, o = capture(lambda m=models[n]: m.encoder(audio_signal=features, length=flen))
                    graphs[n], outputs[n] = g, tuple(t.clone() for t in o)
                ref = outputs[names[0]]
                t = int(ref[1][0])
                for n in names:
                    e = outputs[n][0][..., :t]
                    checks[n] = {"lengths_equal": bool(torch.equal(outputs[n][1], ref[1])), "finite": bool(torch.isfinite(outputs[n][0]).all()),
                                 "rel_frobenius_valid": ((e - ref[0][..., :t]).norm() / ref[0][..., :t].norm()).item(),
                                 "max_abs_valid": (e - ref[0][..., :t]).abs().max().item(),
                                 "bitwise": all(torch.equal(x, y) for x, y in zip(outputs[n], ref))}
                fns = {n: graphs[n].replay for n in names}
            else:
                for m in models.values():
                    m.forward.entries.clear()
                for n in names:
                    text = recognize(models[n], audio, length)
                    checks[n] = {"text": text, **decoded_state(models[n])}
                ref = checks[names[0]]
                for n in names:
                    checks[n]["identical_to_" + names[0]] = checks[n] == ref or all(checks[n][k] == ref[k] for k in ("text", "tokens", "timestamps"))
                fns = {n: (lambda m=models[n]: recognize(m, audio, length)) for n in names}
            for repeat in range(args.repeats):
                order = names if (repeat + index) % 2 == 0 else names[::-1]
                for n in order:
                    check_power(meter)
                    started = time.perf_counter()
                    while time.perf_counter() - started < 0.3:
                        fns[n]()
                        torch.cuda.synchronize()
                    w = meter.measure(fns[n], min_seconds=args.seconds, synchronize=torch.cuda.synchronize)
                    w["median_ms"] = statistics.median(w["iteration_seconds"]) * 1000
                    w.pop("samples", None)
                    w.pop("iteration_seconds", None)
                    windows[n].append(w)
            summary = {n: {k: statistics.median(w[k] for w in ws) for k in ("median_ms", "joules_per_iteration", "average_watts")}
                       for n, ws in windows.items()}
            row = {"index": index, "id": rec["id"], "corpus": rec["corpus"], "duration": rec["duration"], "M": rec["M"],
                   "target_seconds": rec["target_seconds"], "checks": checks, "summary": summary, "windows": windows}
            report["rows"].append(row)
            print(json.dumps({"index": index, "id": rec["id"], "M": rec["M"],
                              **{n: [round(s["median_ms"], 4), round(s["joules_per_iteration"], 4)] for n, s in summary.items()}}), flush=True)
            out.write_text(json.dumps(report, indent=1) + "\n")
            if args.scope == "encoder":
                del graphs, outputs
        report["lock"]["held_s"] = env.check_hold(lock, margin_s=0)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print("RESULTS", out)
    return out


def paired_merge(args) -> Path:
    """Merge chunk files of one run and scope; paired changes vs --reference with counts."""
    run = env.ARTIFACTS_ROOT / args.agent / "bench" / args.run
    rows, variants = [], None
    for f in sorted(run.glob(f"paired-{args.scope}-*.json")):
        doc = json.loads(f.read_text())
        variants = variants or doc["variants"]
        if doc["variants"] != variants:
            raise ValueError(f"{f}: different variants")
        rows += doc["rows"]
    rows.sort(key=lambda r: r["index"])
    if len({r["index"] for r in rows}) != len(rows):
        raise ValueError("duplicate utterances across chunks")
    ref = args.reference
    result = {"protocol": "P-paired", "scope": args.scope, "utterances": len(rows), "reference": ref, "changes": {}}
    for n in variants:
        if n == ref:
            continue
        per = {}
        for key in ("median_ms", "joules_per_iteration", "average_watts"):
            ch = [(r["summary"][n][key] - r["summary"][ref][key]) / r["summary"][ref][key] * 100 for r in rows]
            per[key] = {"median_change_pct": statistics.median(ch), "lower_count": sum(c < 0 for c in ch), "n": len(ch),
                        "by_target": {str(t): {"median_change_pct": statistics.median([c for c, r in zip(ch, rows) if r["target_seconds"] == t]),
                                               "lower_count": sum(c < 0 for c, r in zip(ch, rows) if r["target_seconds"] == t)}
                                      for t in sorted({r["target_seconds"] for r in rows})}}
        result["changes"][n] = per
    result["absolute"] = {n: {k: statistics.median(r["summary"][n][k] for r in rows)
                              for k in ("median_ms", "joules_per_iteration", "average_watts")} for n in variants}
    if args.scope == "full":
        result["identical_transcripts"] = {n: sum(r["checks"][n]["identical_to_" + variants[0]] for r in rows) for n in variants}
    else:
        result["encoder_checks"] = {n: {"max_rel_frobenius_valid": max(r["checks"][n]["rel_frobenius_valid"] for r in rows),
                                        "all_lengths_equal": all(r["checks"][n]["lengths_equal"] for r in rows),
                                        "all_finite": all(r["checks"][n]["finite"] for r in rows)} for n in variants}
    out = run / f"paired-{args.scope}-merged.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return out


# --- P-clips and P-memory ------------------------------------------------------------------------

def p_clips(args) -> Path:
    """Fresh process, one variant: dev clips 3/10/30 s in order, accumulating graph cache, three 5 s
    windows after 2 s warm-up each (compare_v2.py's loop), NVML process VRAM every 10 ms from start."""
    import os
    import threading
    import evaluate as ev
    import soundfile as sf
    from nemo.utils import logging
    env.require_shared_wrapper()
    logging.setLevel(logging.ERROR)
    clips = workload_records(args, "dev_clips")
    out_dir = env.ARTIFACTS_ROOT / args.agent / "bench" / args.run
    env.require_mount()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"clips-{args.variant.replace(':', '_')}.json"
    report = {"protocol": "P-clips+P-memory", "variant": args.variant, "args": vars(args), "workloads": []}
    with env.gpu_lock(f"{args.agent}:P-clips {args.variant}") as lock, contextlib.ExitStack() as stack:
        meter = stack.enter_context(contextlib.closing(meter_and_checks()))
        report["gpu"] = meter.metadata()
        report["lock"] = {k: lock[k] for k in ("label", "waited_s", "pid")}
        samples, errors = [], []

        def memory():
            ps = meter._nvml.nvmlDeviceGetComputeRunningProcesses(meter._handle)
            return max([int(q.usedGpuMemory) for q in ps if q.pid == os.getpid()] or [0])
        stop = threading.Event()

        def sample():
            while not stop.wait(.01):
                try:
                    samples.append((time.perf_counter(), memory()))
                except Exception as exc:  # noqa: BLE001
                    errors.append(str(exc))
                    return
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        stack.callback(lambda: (stop.set(), thread.join()))
        stack.enter_context(torch.inference_mode())
        stack.enter_context(ev.strict_fp32("cuda"))
        started = time.perf_counter()
        model, info = stack.enter_context(variant_model(args.variant, max_graphs=4))
        # the variant's own metadata (e.g. an activation-precision arm's recognition identity and run), as
        # P-paired records it in variants_info (review c2 of the activation-precision code)
        report["variant_info"] = json.loads(json.dumps({k: v for k, v in info.items() if k != "projection"}, default=str))
        report["variant_facts"] = variant_facts(model)
        report["load_seconds"] = time.perf_counter() - started

        def snapshot():
            torch.cuda.synchronize()
            return {"process_bytes": memory(), "torch_allocated": torch.cuda.memory_allocated(),
                    "torch_reserved": torch.cuda.memory_reserved(), "torch_peak_allocated": torch.cuda.max_memory_allocated(),
                    "torch_peak_reserved": torch.cuda.max_memory_reserved()}
        report["loaded"] = snapshot()
        for rec in clips:
            env.check_hold(lock)
            audio_np, rate = sf.read(rec["audio_filepath"], dtype="float32")
            assert rate == 16000 and audio_np.ndim == 1 and len(audio_np) == rec["samples"]
            audio = torch.from_numpy(audio_np).unsqueeze(0)
            length = torch.tensor([len(audio_np)])
            torch.cuda.reset_peak_memory_stats()
            begin = len(samples)
            first = time.perf_counter()
            text = recognize(model, audio, length)
            torch.cuda.synchronize()
            row = {"record": {k: rec[k] for k in ("id", "duration", "samples", "M")}, "hypothesis": text,
                   **decoded_state(model), "first_call_seconds": time.perf_counter() - first, "after_first": snapshot(),
                   "windows": []}
            for _ in range(args.repeats):
                check_power(meter)
                warm = time.perf_counter()
                while time.perf_counter() - warm < 2:
                    recognize(model, audio, length)
                    torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start_samples = len(samples)
                w = meter.measure(lambda: recognize(model, audio, length), min_seconds=args.seconds,
                                  synchronize=torch.cuda.synchronize)
                times = sorted(w["iteration_seconds"])
                w["median_ms"] = statistics.median(times) * 1000
                w["p95_ms"] = times[min(len(times) - 1, int(len(times) * .95))] * 1000
                w["memory"] = snapshot()
                w["process_sampled_peak_bytes"] = max([v for _, v in samples[start_samples:]] + [memory()])
                w.pop("samples", None)
                w.pop("iteration_seconds", None)
                row["windows"].append(w)
            row["summary"] = {k: statistics.median(w[k] for w in row["windows"])
                              for k in ("median_ms", "p95_ms", "joules_per_iteration", "average_watts")}
            row["process_sampled_peak_including_first_bytes"] = max([v for _, v in samples[begin:]] + [memory()])
            report["workloads"].append(row)
            out.write_text(json.dumps(report, indent=1) + "\n")
            print(args.variant, rec["id"], rec["duration"], row["summary"], flush=True)
        stop.set()
        thread.join()
        if errors:
            raise RuntimeError(errors)
        report["performance_phase_process_peak_bytes"] = max(v for _, v in samples)
        report["memory_samples_10ms"] = samples
        report["lock"]["held_s"] = env.check_hold(lock, margin_s=0)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print("RESULTS", out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="protocol", required=True)
    k = sub.add_parser("kernel", help="P-kernel")
    k.add_argument("--agent", required=True, help="artifact directory and lock label prefix (A10)")
    k.add_argument("--impl", nargs="*", default=[], help="registry names of candidates")
    k.add_argument("--baselines", action="store_true", help="also time r_compact r_expanded r_hybrid r_bf16x3")
    k.add_argument("--M", type=int, nargs="+", default=[38, 126, 368], help="dev clip M values (workloads.json)")
    k.add_argument("--ops", nargs="+", default=list(OPS), choices=list(OPS))
    k.add_argument("--run", default=None, help="session run: write bench/<run>/kernel.json (default: a timestamped file)")
    pp = sub.add_parser("paired", help="P-paired, one chunk of dev_24 per process")
    pp.add_argument("--agent", required=True)
    pp.add_argument("--run", required=True, help="run name grouping the chunks")
    pp.add_argument("--scope", choices=["encoder", "full"], required=True)
    pp.add_argument("--variants", nargs="+", required=True, help="first is the reference, e.g. R-expanded int8_mma")
    pp.add_argument("--utterances", default="0-5", help="inclusive dev_24 index range a-b (keep each chunk < 10 min)")
    pp.add_argument("--seconds", type=float, default=1.0)
    pp.add_argument("--repeats", type=int, default=3)
    pp.add_argument("--workloads", choices=WORKLOAD_SETS, default="dev")
    pp.add_argument("--dev-run", default=None, help="historical-test: the x2 benchmark run whose dev results are written (A13)")
    pm = sub.add_parser("paired-merge", help="merge P-paired chunks (CPU)")
    pm.add_argument("--agent", required=True)
    pm.add_argument("--run", required=True)
    pm.add_argument("--scope", choices=["encoder", "full"], required=True)
    pm.add_argument("--reference", default="R-expanded")
    pc = sub.add_parser("clips", help="P-clips + P-memory, one variant per fresh process")
    pc.add_argument("--agent", required=True)
    pc.add_argument("--run", required=True)
    pc.add_argument("--variant", required=True, help="R-compact | R-expanded | R-hybrid | registry name")
    pc.add_argument("--seconds", type=float, default=5.0)
    pc.add_argument("--repeats", type=int, default=3)
    pc.add_argument("--workloads", choices=WORKLOAD_SETS, default="dev")
    pc.add_argument("--dev-run", default=None, help="historical-test: the x2 benchmark run whose dev results are written (A13)")
    args = p.parse_args()
    {"kernel": p_kernel, "paired": p_paired, "paired-merge": paired_merge, "clips": p_clips}[args.protocol](args)


if __name__ == "__main__":
    main()
