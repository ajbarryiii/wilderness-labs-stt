"""Per-phase energy breakdown with power / SM-clock / memory-clock traces for a pair of variants
(coordinator 2026-10-20: B-hybrid vs B-hybrid-b01, whose full-scope energy gain of about -3.5% sat
beside a neutral encoder-only result). Dev_24 only; both variants resident in one process (P-paired).

Per utterance and variant, four phases, each in its own windows (variant order alternating per window
and per utterance, 0.3 s warm-up before each window, three 1 s windows; M1's EnergyMeter: NVML board
energy over exactly the synchronized iterations, telemetry sampled every 20 ms and KEPT as traces):
  full     bench.recognize: CPU waveform -> H2D -> pipeline graph (frontend + encoder) -> fused decoder -> text
  forward  model.forward(input_signal, length): H2D + the pipeline graph (frontend + encoder), no decoding
  encoder  the encoder alone: inference.benchmark.capture graph replay on fixed GPU features
  decode   rnnt_decoder_predictions_tensor on that variant's own fixed encoder output
Derived per utterance (medians of windows): frontend_transfer = forward - encoder; decode as measured;
host_and_idle = full - forward - decode (time and energy). Each phase is reported as the paired change
of the second variant against the first, with counts, plus power and clock statistics and traces.

    timeout 600 gpu1bit/shared/python -m gpu1bit.shared.dispatch --patches ... phases --agent H \\
        --run R --variants B-hybrid B-hybrid-b01 --utterances 0-3
Output: ARTIFACTS_ROOT/<agent>/bench/<run>/phases-<a>-<b>.json; `summarize(run, pair)` merges them.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import statistics
import sys
import time

import torch

from gpu1bit.shared import env

PHASES = ("full", "forward", "encoder", "decode")
TRACE_INTERVAL_S = 0.02


def _meter():
    """M1's custom/inference-efficiency EnergyMeter with a 20 ms telemetry interval (traces)."""
    import paths
    try:
        import pynvml  # noqa: F401
    except ImportError:
        sys.path.append("/mnt/hd/wilderness-labs-stt/inference-efficiency/deps")
    spec = importlib.util.spec_from_file_location("stt_energy", paths.REPO / "custom/inference-efficiency/energy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meter = module.EnergyMeter(synchronize=torch.cuda.synchronize, sample_interval=TRACE_INTERVAL_S)
    deadline = time.monotonic() + 20
    while meter.foreign_processes() and time.monotonic() < deadline:
        time.sleep(0.5)
    meter._guard()
    limit = meter._nvml.nvmlDeviceGetPowerManagementLimit(meter._handle) / 1000
    if limit != env.POWER_LIMIT_W:
        raise RuntimeError(f"power limit {limit} W != {env.POWER_LIMIT_W} W")
    return meter


def _trace(window: dict) -> dict:
    s = window.pop("samples")
    return {"elapsed_s": [x["elapsed_s"] for x in s], "power_w": [x["power_w"] for x in s],
            "sm_clock_mhz": [x["sm_clock_mhz"] for x in s], "memory_clock_mhz": [x["memory_clock_mhz"] for x in s],
            "temperature_c": [x["temperature_c"] for x in s]}


def measure(args) -> None:
    import evaluate as ev
    from nemo.utils import logging
    from inference.benchmark import capture
    from gpu1bit.shared import bench, workloads
    env.require_shared_wrapper()
    logging.setLevel(logging.ERROR)
    records = workloads.load()["dev_24"]
    a, b = (int(v) for v in args.utterances.split("-"))
    names = list(args.variants)
    out_dir = env.ARTIFACTS_ROOT / args.agent / "bench" / args.run
    env.require_mount()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"phases-{a:02d}-{b:02d}.json"
    report = {"protocol": "phases", "phases": PHASES, "variants": names, "args": vars(args), "rows": []}
    with env.gpu_lock(f"{args.agent}:phases {' '.join(names)} {a}-{b}") as lock, torch.inference_mode(), \
            contextlib.ExitStack() as stack:
        stack.enter_context(ev.strict_fp32("cuda"))
        meter = stack.enter_context(contextlib.closing(_meter()))
        report["gpu"] = meter.metadata()
        report["lock"] = {k: lock[k] for k in ("label", "waited_s", "pid")}
        models = {n: stack.enter_context(bench.variant_model(n, max_graphs=4))[0] for n in names}
        report["variant_facts"] = {n: bench.variant_facts(m) for n, m in models.items()}
        for index in range(a, b + 1):
            rec = records[index]
            env.check_hold(lock)
            audio, length = ev.audio_batch([rec])
            feats, flen = models[names[0]].preprocessor(input_signal=audio.cuda(), length=length.cuda())
            fns, checks = {}, {}
            for n, m in models.items():
                m.forward.entries.clear()
                text = bench.recognize(m, audio, length)                       # captures the pipeline graph
                g, o = capture(lambda m=m: m.encoder(audio_signal=feats, length=flen))
                enc, enc_len = (t.clone() for t in o)
                fns[n] = {"full": (lambda m=m: bench.recognize(m, audio, length)),
                          "forward": (lambda m=m: m.forward(input_signal=audio.cuda(), input_signal_length=length.cuda())),
                          "encoder": g.replay,
                          "decode": (lambda m=m, e=enc, l=enc_len: m.decoding.rnnt_decoder_predictions_tensor(e, l, return_hypotheses=False))}
                checks[n] = {"text": text, "graph": g}
            windows = {n: {p: [] for p in PHASES} for n in names}
            traces = {n: {p: [] for p in PHASES} for n in names}
            for repeat in range(args.repeats):
                order = names if (repeat + index) % 2 == 0 else names[::-1]
                for phase in PHASES:
                    for n in order:
                        bench.check_power(meter)
                        started = time.perf_counter()
                        while time.perf_counter() - started < 0.3:
                            fns[n][phase]()
                            torch.cuda.synchronize()
                        w = meter.measure(fns[n][phase], min_seconds=args.seconds, synchronize=torch.cuda.synchronize)
                        w["median_ms"] = statistics.median(w.pop("iteration_seconds")) * 1000
                        traces[n][phase].append(_trace(w))
                        windows[n][phase].append(w)
            summary = {n: {p: {k: statistics.median(w[k] for w in windows[n][p])
                               for k in ("median_ms", "joules_per_iteration", "average_watts")} for p in PHASES} for n in names}
            for n in names:
                ph = summary[n]
                ph["frontend_transfer"] = {k: ph["forward"][k] - ph["encoder"][k] for k in ("median_ms", "joules_per_iteration")}
                ph["host_and_idle"] = {k: ph["full"][k] - ph["forward"][k] - ph["decode"][k] for k in ("median_ms", "joules_per_iteration")}
                ph["telemetry"] = {p: {"sm_clock_mhz_mean": statistics.mean(w["telemetry_averages"]["sm_clock_mhz"] for w in windows[n][p]),
                                       "memory_clock_mhz_mean": statistics.mean(w["telemetry_averages"]["memory_clock_mhz"] for w in windows[n][p]),
                                       "temperature_c_mean": statistics.mean(w["telemetry_averages"]["temperature_c"] for w in windows[n][p])}
                                   for p in PHASES}
            row = {"index": index, "id": rec["id"], "corpus": rec["corpus"], "duration": rec["duration"], "M": rec["M"],
                   "target_seconds": rec["target_seconds"], "texts_identical": len({c["text"] for c in checks.values()}) == 1,
                   "summary": summary, "windows": windows, "traces": traces}
            report["rows"].append(row)
            out.write_text(json.dumps(report) + "\n")
            print(json.dumps({"index": index, "id": rec["id"], **{n: {p: round(summary[n][p]["joules_per_iteration"], 4)
                                                                      for p in PHASES} for n in names}}), flush=True)
            for c in checks.values():
                del c["graph"]
        report["lock"]["held_s"] = env.check_hold(lock, margin_s=0)
    out.write_text(json.dumps(report) + "\n")
    print("RESULTS", out)


def summarize(run: str, pair: list[str], agent: str = "H") -> dict:
    """Merged per-phase paired changes (second vs first variant) over all chunks of a run."""
    rows = []
    for f in sorted((env.ARTIFACTS_ROOT / agent / "bench" / run).glob("phases-*.json")):
        doc = json.loads(f.read_text())
        if doc["variants"] != list(pair):
            raise ValueError(f"{f}: variants {doc['variants']}")
        rows += doc["rows"]
    rows.sort(key=lambda r: r["index"])
    a, b = pair
    out = {"pair": pair, "utterances": len(rows), "texts_identical": sum(r["texts_identical"] for r in rows), "phases": {}}
    for phase in PHASES + ("frontend_transfer", "host_and_idle"):
        per = {}
        for key in ("median_ms", "joules_per_iteration"):
            vals_a = [r["summary"][a][phase][key] for r in rows]
            vals_b = [r["summary"][b][phase][key] for r in rows]
            diffs = [y - x for x, y in zip(vals_a, vals_b)]
            per[key] = {"median_a": statistics.median(vals_a), "median_b": statistics.median(vals_b),
                        "median_difference_b_minus_a": statistics.median(diffs), "b_lower": sum(d < 0 for d in diffs)}
            if phase in PHASES:
                ch = [(y - x) / x * 100 for x, y in zip(vals_a, vals_b)]
                per[key]["median_change_pct"] = statistics.median(ch)
        if phase in PHASES:
            per["average_watts"] = {"median_a": statistics.median(r["summary"][a][phase]["average_watts"] for r in rows),
                                    "median_b": statistics.median(r["summary"][b][phase]["average_watts"] for r in rows)}
            for k in ("sm_clock_mhz_mean", "memory_clock_mhz_mean", "temperature_c_mean"):
                per[k] = {"median_a": statistics.median(r["summary"][a]["telemetry"][phase][k] for r in rows),
                          "median_b": statistics.median(r["summary"][b]["telemetry"][phase][k] for r in rows)}
        out["phases"][phase] = per
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--variants", nargs=2, required=True)
    p.add_argument("--utterances", default="0-3")
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--repeats", type=int, default=3)
    measure(p.parse_args())


if __name__ == "__main__":
    main()
