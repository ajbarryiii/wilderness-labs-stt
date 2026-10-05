"""Reproducible SM120 tuning, numerical/ASR validation, and GPU energy measurement.

All results are written to a new directory on /mnt/hd. Uses the existing
inference-efficiency NVML meter (including its foreign-compute-process guard).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

import paths
from export import load_export, unpack_codes
from .kernels import matmul
from .runtime import PackedLinear, load_packed

DEFAULT_EXPORT = paths.RUNS / "main-M1-P2-lr5e-4" / "export"


def save(directory, name, data):
    paths.require_mount()
    (directory / name).write_text(json.dumps(data, indent=2) + "\n")


def energy_meter():
    try:
        import pynvml  # noqa: F401
    except ImportError:
        # Reuse the locally installed, pinned project dependency on this machine.
        sys.path.append("/mnt/hd/wilderness-labs-stt/inference-efficiency/deps")
    spec = importlib.util.spec_from_file_location("stt_energy", paths.REPO / "custom/inference-efficiency/energy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnergyMeter(synchronize=torch.cuda.synchronize)


def metadata(args):
    here = Path(__file__).resolve().parent
    manifest = json.loads((args.export / "manifest.json").read_text())
    return {"torch": torch.__version__, "triton": triton.__version__, "gpu": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(), "export": str(args.export),
            "export_sha256": manifest["sha256"], "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "runtime_wrapper_sha256": hashlib.sha256((paths.HERE / "python").read_bytes()).hexdigest(),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in here.glob("*.py")}}


def micro(args, directory, *, tune=False):
    from safetensors import safe_open

    manifest = json.loads((args.export / "manifest.json").read_text())
    seen, rows = set(), []
    with safe_open(args.export / manifest["file"], framework="pt") as f:
        for name, layer in manifest["quantized_layers"].items():
            n, k = layer["shape"]
            if (n, k) in seen:
                continue
            seen.add((n, k))
            packed = f.get_tensor(f"{name}.codes")
            scale = f.get_tensor(f"{name}.scale")
            module = PackedLinear(packed, scale, k).cuda()
            dense = (unpack_codes(packed, k).float() * scale[:, None]).cuda()
            for m in args.rows:
                x = torch.randn(m, k, device="cuda")
                reference = F.linear(x, dense)
                dense_us = triton.testing.do_bench_cudagraph(lambda: F.linear(x, dense), rep=60) * 1000
                configs = [None]
                if tune and m > 4:
                    configs += [(bm, bn, bk, split, 4, 2)
                                for bm, bn, bk in [(32, 64, 32), (64, 32, 32), (64, 64, 32), (64, 32, 64)]
                                for split in (1, 2, 4, 8)]
                candidates = []
                for config in configs:
                    fn = lambda: matmul(x, module.packed_t, module.scale, None, k, config=config)
                    y = fn()
                    # Model row scales are larger than the unit-test scales.
                    # FP32 cuBLAS and Tensor Core reduction orders differ.
                    torch.testing.assert_close(y, reference, atol=1e-4, rtol=2e-5)
                    us = [triton.testing.do_bench_cudagraph(fn, rep=30) * 1000 for _ in range(3)]
                    candidates.append({"config": config, "us": us, "median_us": statistics.median(us),
                                       "max_abs_error": float((y - reference).abs().max())})
                row = {"layer": name, "m": m, "n": n, "k": k, "dense_us": dense_us,
                       "candidates": sorted(candidates, key=lambda r: r["median_us"])}
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "candidates"}),
                      "best", row["candidates"][0], flush=True)
                save(directory, "micro.json", {"scope": "hot-weight CUDA graphs; excludes packing/compilation; not model energy", "rows": rows})


def records_for_sets(count):
    records = []
    for name in paths.TEST_SETS:
        source = [json.loads(line) for line in (paths.MANIFESTS / f"test_{name}.jsonl").read_text().splitlines()]
        # Evenly spaced across the entire manifest, deterministic, without duplicates.
        indices = sorted({i * (len(source) - 1) // max(1, min(count, len(source)) - 1)
                          for i in range(min(count, len(source)))})
        records.extend({**source[i], "source": name} for i in indices)
    return records


def quality(models, args, directory):
    import evaluate as ev

    records = records_for_sets(args.utterances_per_set)
    transcripts, stats = {}, {}
    for name, model in models.items():
        stats[name] = {}
        transcripts[name] = ev.transcribe_records(model, records, args.batch_size, "cuda", workers=0, stats=stats[name])
        print("quality", name, stats[name], flush=True)
    rows, per_set = [], {}
    for i, record in enumerate(records):
        hyps = {name: texts[i] for name, texts in transcripts.items()}
        row = {**record, "hypotheses": hyps, "scores": {name: ev.score(record["text"], text) for name, text in hyps.items()}}
        rows.append(row)
        s = per_set.setdefault(record["source"], {name: {"words": 0, "edits": 0} for name in models})
        for name, score in row["scores"].items():
            if score["scored"]:
                s[name]["words"] += len(score["ref_norm"].split())
                s[name]["edits"] += score["S"] + score["D"] + score["I"]
    for s in per_set.values():
        for result in s.values():
            result["wer_percent"] = 100 * result["edits"] / max(1, result["words"])
    report = {"utterances": len(records), "audio_seconds": sum(r["duration"] for r in records),
              "exact_transcript_matches": sum(a == b for a, b in zip(transcripts["dense_fp32"], transcripts["packed"])),
              "normalized_transcript_matches": sum(ev.normalize(a) == ev.normalize(b) for a, b in zip(transcripts["dense_fp32"], transcripts["packed"])),
              "per_set": per_set, "stats": stats, "records": rows,
              "scope": "deterministic subset validation, not a full-corpus WER reevaluation"}
    save(directory, "quality.json", report)
    print("QUALITY", {k: v for k, v in report.items() if k not in ("records", "stats")}, flush=True)


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            reference = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    # Capture records work without executing it. Never score uninitialized
    # capture output; replay and check the complete encoder against eager.
    graph.replay()
    torch.cuda.synchronize()
    for actual, expected in zip(output, reference):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    return graph, output


def measure_pair(functions, args, meter, directory, label):
    windows = []
    for repeat in range(args.repeats):
        names = list(functions)
        if repeat % 2:
            names.reverse()
        for name in names:
            # Settle clocks/power before every window, after changing workload.
            start = time.perf_counter()
            while time.perf_counter() - start < args.warmup_seconds:
                functions[name]()
                torch.cuda.synchronize()
            result = meter.measure(functions[name], min_seconds=args.seconds)
            latency = sorted(result["iteration_seconds"])
            result.update(variant=name, repeat=repeat, workload=label,
                          median_ms=statistics.median(latency) * 1000,
                          p95_ms=latency[min(len(latency) - 1, int(len(latency) * .95))] * 1000)
            windows.append(result)
            save(directory, f"{label}.json", windows)
            print(label, name, {k: result[k] for k in ("median_ms", "p95_ms", "joules_per_iteration", "average_watts")}, flush=True)
    return windows


def model_benchmark(args, directory):
    import evaluate as ev
    from nemo.utils import logging

    logging.setLevel(logging.ERROR)
    models = {"dense_fp32": load_export(args.export, "cuda"), "packed": load_packed(args.export)}
    save(directory, "storage.json", {name: {"parameter_bytes": sum(p.nbytes for p in model.parameters()),
                                          "buffer_bytes": sum(b.nbytes for b in model.buffers())}
                                     for name, model in models.items()})
    if args.utterances_per_set:
        quality(models, args, directory)
    records = [json.loads(l) for l in (paths.MANIFESTS / "test_librispeech_clean.jsonl").read_text().splitlines()]
    all_windows, errors = [], []
    with contextlib.ExitStack() as stack:
        for model in models.values():
            stack.enter_context(ev.inference_settings(model))
        meter = stack.enter_context(energy_meter())
        save(directory, "gpu.json", meter.metadata())
        for seconds in args.audio_seconds:
            record = min(records, key=lambda r: abs(r["duration"] - seconds))
            audio, lengths = ev.audio_batch([record])
            audio, lengths = audio.cuda(), lengths.cuda()
            # Measure encoder only, with a shared, identical frontend result.
            features, feature_lengths = models["dense_fp32"].preprocessor(input_signal=audio, length=lengths)
            graphs, outputs = {}, {}
            for name, model in models.items():
                graph, output = capture(lambda model=model: model.encoder(audio_signal=features, length=feature_lengths))
                graphs[name], outputs[name] = graph, output
            a, b = outputs["dense_fp32"][0], outputs["packed"][0]
            errors.append({"audio": record, "max_abs_encoder_error": float((a-b).abs().max()),
                           "relative_l2_encoder_error": float((a-b).norm() / a.norm()),
                           "shape": list(a.shape)})
            save(directory, "encoder-errors.json", errors)
            if not args.seconds:
                del graphs, outputs, a, b
                continue
            label = f"encoder-{seconds:g}s"
            windows = measure_pair({n: g.replay for n, g in graphs.items()}, args, meter, directory, label)
            all_windows.extend(windows)
            if seconds == args.audio_seconds[len(args.audio_seconds)//2]:
                # Full eager preprocessing + encoder + greedy TDT + text, input already on GPU.
                def decode(model):
                    enc, enc_len = model(input_signal=audio, input_signal_length=lengths)
                    return model.decoding.rnnt_decoder_predictions_tensor(
                        encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=False)
                functions = {name: (lambda model=model: decode(model)) for name, model in models.items()}
                all_windows.extend(measure_pair(functions, args, meter, directory, f"transcribe-{seconds:g}s"))
                from .graphs import enable_encoder_graphs, disable_encoder_graphs
                eager_outputs = {name: functions[name]() for name in models}
                for model in models.values():
                    enable_encoder_graphs(model)
                for name in models:
                    graphed = functions[name]()
                    def texts(output):
                        if isinstance(output, tuple):
                            output = output[0]
                        return [h.text if hasattr(h, "text") else str(h) for h in output]
                    if texts(graphed) != texts(eager_outputs[name]):
                        raise AssertionError(f"{name}: graph cache changed transcription")
                all_windows.extend(measure_pair(functions, args, meter, directory, f"transcribe-graphs-{seconds:g}s"))
                for model in models.values():
                    disable_encoder_graphs(model)
            del graphs, outputs, a, b
    summary = []
    for label in dict.fromkeys(w["workload"] for w in all_windows):
        values = {}
        for name in models:
            window = [w for w in all_windows if w["workload"] == label and w["variant"] == name]
            values[name] = {key: statistics.median(w[key] for w in window)
                            for key in ("median_ms", "p95_ms", "joules_per_iteration", "average_watts")}
        summary.append({"workload": label, **values,
                        "speedup": values["dense_fp32"]["median_ms"] / values["packed"]["median_ms"],
                        "energy_reduction_fraction": 1 - values["packed"]["joules_per_iteration"] / values["dense_fp32"]["joules_per_iteration"]})
    save(directory, "summary.json", {"scope": "GPU board energy; desktop remains active; host energy and audio disk I/O excluded; stock strict-FP32 same-weight NeMo control", "workloads": summary})
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("micro", "tune", "model"))
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 64, 128, 384, 1024])
    parser.add_argument("--utterances-per-set", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--warmup-seconds", type=float, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--audio-seconds", type=float, nargs="+", default=[3, 10, 30])
    args = parser.parse_args()
    if min(args.rows) < 1 or args.seconds < 0 or args.utterances_per_set < 0 or args.repeats < 1:
        parser.error("invalid workload size")
    paths.require_mount()
    directory = paths.ARTIFACTS / "kernel-runs" / f"{time.strftime('%Y%m%dT%H%M%S')}-{args.command}"
    directory.mkdir(parents=True, exist_ok=False)
    print("RESULTS", directory, flush=True)
    torch.manual_seed(417)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with paths.gpu_lock(f"packed kernels {args.command}"), torch.inference_mode():
        with energy_meter() as meter:
            meter._guard()
        save(directory, "metadata.json", metadata(args))
        if args.command == "model":
            model_benchmark(args, directory)
        else:
            micro(args, directory, tune=args.command == "tune")


if __name__ == "__main__":
    main()
