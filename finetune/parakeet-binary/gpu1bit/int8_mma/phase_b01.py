"""b01 confirmation, step 2: energy and time by pipeline phase, B-hybrid vs B-hybrid-b01 (dev_24 only).

Both variants live in one process (as in H's p_paired) and see the same audio. Per utterance and
variant, H's EnergyMeter (inference.benchmark.energy_meter: NVML board energy counter over exactly the
synchronized iterations, power / clock / temperature telemetry) measures four phases, each in its own
windows, variant order alternating per window and per utterance:
  full     bench.recognize: CPU waveform -> H2D -> pipeline graph (frontend + encoder) -> fused decoder -> text
  forward  model(input_signal, length): H2D + the pipeline graph (frontend + encoder), no decoding
  encoder  the encoder alone, a CUDA graph replay on fixed features (P-paired encoder scope)
  decode   rnnt_decoder_predictions_tensor on that variant's fixed encoder output (bitwise equal across the
           two variants, so the decoder's work is identical)
Derived: frontend = forward - encoder; host_and_idle = full - forward - decode (time and energy). A saving
that b01 causes must appear in `encoder` (or `forward`); a difference in `decode` or `host_and_idle` is
outside the kernel. Each process: one gpu_lock (gpulock.locked), <= 600 s, 400 W asserted.

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.phase_b01 --utterances 0-5 --run b01-phases
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time

import torch

from gpu1bit.int8_mma import b01_run, gpulock  # noqa: F401  (b01_run registers B-hybrid-b01)
from gpu1bit.shared import bench, env

VARIANTS = ("B-hybrid", "B-hybrid-b01")
PHASES = ("full", "forward", "encoder", "decode")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--utterances", default="0-5")
    p.add_argument("--run", default="b01-phases")
    p.add_argument("--seconds", type=float, default=0.6)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    env.require_shared_wrapper()
    a, b = (int(v) for v in args.utterances.split("-"))
    out_dir = env.ARTIFACTS_ROOT / "B" / "bench" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"phases-{a:02d}-{b:02d}.json"
    with gpulock.locked(f"B:b01 phases {a}-{b}") as lock, torch.inference_mode(), contextlib.ExitStack() as stack:
        import evaluate as ev
        from nemo.utils import logging
        from inference.benchmark import capture
        from gpu1bit.shared import workloads
        logging.setLevel(logging.ERROR)
        stack.enter_context(ev.strict_fp32("cuda"))
        meter = stack.enter_context(contextlib.closing(bench.meter_and_checks()))
        report = {"protocol": "b01 phase breakdown", "args": vars(args), "variants": VARIANTS, "gpu": meter.metadata(),
                  "lock": {k: lock[k] for k in ("label", "waited_s", "pid")}, "rows": []}
        models = {v: stack.enter_context(bench.variant_model(v, max_graphs=4))[0] for v in VARIANTS}
        records = workloads.load()["dev_24"]
        for index in range(a, b + 1):
            env.check_hold(lock, margin_s=60)
            rec = records[index]
            audio, length = ev.audio_batch([rec])
            feats, flen = models[VARIANTS[0]].preprocessor(input_signal=audio.cuda(), length=length.cuda())
            graphs, enc = {}, {}
            for v, m in models.items():
                m.forward.entries.clear()
                g, o = capture(lambda m=m: m.encoder(audio_signal=feats, length=flen))
                graphs[v] = g
                enc[v] = (o[0].clone(), o[1].clone())
            same = torch.equal(enc[VARIANTS[0]][0], enc[VARIANTS[1]][0])
            fns = {}
            for v, m in models.items():
                fns[v] = {"full": (lambda m=m: bench.recognize(m, audio, length)),
                          "forward": (lambda m=m: m(input_signal=audio.cuda(), input_signal_length=length.cuda())),
                          "encoder": graphs[v].replay,
                          "decode": (lambda m=m, e=enc[v]: m.decoding.rnnt_decoder_predictions_tensor(e[0], e[1], return_hypotheses=False))}
            windows = {v: {ph: [] for ph in PHASES} for v in VARIANTS}
            for repeat in range(args.repeats):
                for ph in PHASES:
                    order = VARIANTS if (repeat + index + PHASES.index(ph)) % 2 == 0 else VARIANTS[::-1]
                    for v in order:
                        bench.check_power(meter)
                        warm = time.perf_counter()
                        while time.perf_counter() - warm < 0.2:
                            fns[v][ph]()
                            torch.cuda.synchronize()
                        w = meter.measure(fns[v][ph], min_seconds=args.seconds, synchronize=torch.cuda.synchronize)
                        w["median_ms"] = statistics.median(w["iteration_seconds"]) * 1000
                        w.pop("samples", None)
                        w.pop("iteration_seconds", None)
                        w.pop("telemetry_samples", None)
                        windows[v][ph].append(w)
            summary = {}
            for v in VARIANTS:
                s = {ph: {k: statistics.median(w[k] for w in windows[v][ph]) for k in ("median_ms", "joules_per_iteration", "average_watts")}
                     for ph in PHASES}
                for ph in PHASES:
                    tel = [w.get("telemetry_averages", {}) for w in windows[v][ph]]
                    s[ph]["sm_clock_mhz"] = statistics.median(t.get("sm_clock_mhz") or 0 for t in tel)
                    s[ph]["memory_clock_mhz"] = statistics.median(t.get("memory_clock_mhz") or 0 for t in tel)
                    s[ph]["temperature_c"] = statistics.median(t.get("temperature_c") or 0 for t in tel)
                s["frontend"] = {k: s["forward"][k] - s["encoder"][k] for k in ("median_ms", "joules_per_iteration")}
                s["host_and_idle"] = {k: s["full"][k] - s["forward"][k] - s["decode"][k] for k in ("median_ms", "joules_per_iteration")}
                summary[v] = s
            row = {"index": index, "id": rec["id"], "corpus": rec["corpus"], "duration": rec["duration"], "M": rec["M"],
                   "target_seconds": rec["target_seconds"], "encoder_outputs_bitwise_equal": same, "summary": summary,
                   "windows": windows}
            report["rows"].append(row)
            out.write_text(json.dumps(report, indent=1) + "\n")
            print(json.dumps({"index": index, "M": rec["M"], "bitwise": same,
                              **{v: {ph: round(summary[v][ph]["joules_per_iteration"] * 1000, 3) for ph in
                                     PHASES + ("frontend", "host_and_idle")} for v in VARIANTS}}), flush=True)
            del graphs
        report["lock"]["held_s"] = env.check_hold(lock, margin_s=0)
        out.write_text(json.dumps(report, indent=1) + "\n")
        print("RESULTS", out)


if __name__ == "__main__":
    main()
