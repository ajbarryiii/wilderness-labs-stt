"""Reversible inference experiments. Nothing here is enabled by the loader.

Run the screening harness with ``python -m inference.experiments``. Models,
inputs, compiler caches and raw measurements remain on the data disk.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
import types

import torch

import paths
from .benchmark import DEFAULT_EXPORT, capture, energy_meter, metadata, save
from .kernels import matmul
from .runtime import PackedLinear, load_packed


from .transforms import encoder_transform as experiment


VARIANTS = {
    "baseline": {},
    "bf16x2": {"mode": "bf16x2"},
    "fp16x2": {"mode": "fp16x2"},
    "bf16_single": {"mode": "bf16"},
    "fp16_single": {"mode": "fp16_rounded"},
    "tf32_single": {"mode": "tf32"},
    "tf32x3": {"mode": "tf32x3"},
    "qkv": {"qkv": True},
    "silu": {"silu": True},
    "position": {"position": True},
    "sdpa": {"sdpa": True},
    "fold_bn": {"fold_bn": True},
    "fusions": {"qkv": True, "silu": True, "position": True, "fold_bn": True},
    "fusions_fp16x2": {"qkv": True, "silu": True, "position": True, "fold_bn": True, "mode": "fp16x2"},
    "fusions_bf16x2": {"qkv": True, "silu": True, "position": True, "fold_bn": True, "mode": "bf16x2"},
    "int8": {"mode": "int8"},
    "int8x2": {"mode": "int8x2"},
    "int8x3": {"mode": "int8x3"},
    "atomic": {"atomic": True},
    "norm": {"norm": True},
    "attention": {"attention": True},
    "all_fusions": {"qkv": True, "silu": True, "position": True, "fold_bn": True, "norm": True, "attention": True},
    "all_fp16x2": {"qkv": True, "silu": True, "position": True, "fold_bn": True, "norm": True, "attention": True, "mode": "fp16x2"},
    "all_bf16x2": {"qkv": True, "silu": True, "position": True, "fold_bn": True, "norm": True, "attention": True, "mode": "bf16x2"},
    "all_int8x2": {"qkv": True, "silu": True, "position": True, "fold_bn": True, "norm": True, "attention": True, "mode": "int8x2"},
    "all_int8x3": {"qkv": True, "silu": True, "position": True, "fold_bn": True, "norm": True, "attention": True, "mode": "int8x3"},
    "norm_int8x3": {"qkv": True, "silu": True, "position": True, "fold_bn": True, "norm": True, "attention": True, "mode": "int8x3", "norm_quant": True},
    "norm_int8x3_pos": {"qkv": True, "silu": True, "position": True, "fold_bn": True, "norm": True, "attention": True, "mode": "int8x3", "norm_quant": True, "position_dot": True},
    "untuned_int8x3": {"qkv": True, "silu": True, "position": True, "fold_bn": True, "norm": True, "attention": True, "mode": "int8x3", "norm_quant": True, "config": (32,64,64,4,4,3)},
}


def latency(graph, repeats=3):
    times = []
    for _ in range(repeats):
        for _ in range(10):
            graph.replay()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(100):
            graph.replay()
        b.record(); b.synchronize()
        times.append(a.elapsed_time(b) / 100)
    return times


def main():
    import evaluate as ev
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=__import__("pathlib").Path, default=DEFAULT_EXPORT)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--audio-seconds", nargs="+", type=float, default=[3, 10, 30])
    p.add_argument("--energy-seconds", type=float, default=0)
    args = p.parse_args()
    paths.require_mount()
    out = paths.ARTIFACTS / "kernel-runs" / f"{time.strftime('%Y%m%dT%H%M%S')}-experiments"
    out.mkdir(); print("RESULTS", out, flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    with paths.gpu_lock("optimization experiments"), torch.inference_mode(), energy_meter() as meter:
        meter._guard()
        save(out, "metadata.json", metadata(args))
        save(out, "gpu.json", meter.metadata())
        model = load_packed(args.export)
        records = [json.loads(s) for s in (paths.MANIFESTS / "test_librispeech_clean.jsonl").read_text().splitlines()]
        with ev.inference_settings(model):
            for seconds in args.audio_seconds:
                rec = min(records, key=lambda r: abs(r["duration"] - seconds))
                audio, lengths = ev.audio_batch([rec]); audio, lengths = audio.cuda(), lengths.cuda()
                f, fl = model.preprocessor(input_signal=audio, length=lengths)
                reference = model.encoder(audio_signal=f, length=fl)[0].clone()
                for name in args.variants:
                    try:
                        with experiment(model, **VARIANTS[name]):
                            graph, output = capture(lambda: model.encoder(audio_signal=f, length=fl))
                            y = output[0]
                            times = latency(graph)
                            row = dict(variant=name, audio_seconds=rec["duration"], gpu_ms=times,
                                       median_ms=statistics.median(times), max_abs_error=float((reference-y).abs().max()),
                                       relative_l2=float((reference-y).norm()/reference.norm()))
                            if args.energy_seconds:
                                result = meter.measure(graph.replay, min_seconds=args.energy_seconds)
                                save(out, f"energy-{seconds:g}-{name}.json", result)
                                row.update(joules=result["joules_per_iteration"], watts=result["average_watts"])
                            del graph, output, y
                    except Exception as exc:
                        row = dict(variant=name, audio_seconds=rec["duration"], error=repr(exc))
                    rows.append(row); save(out, "screen.json", rows)
                    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
