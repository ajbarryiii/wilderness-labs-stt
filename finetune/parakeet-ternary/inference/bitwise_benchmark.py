"""Bounded LUT/SIMT comparison on actual Parakeet weights. Never changes dispatch."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import time

import torch
import triton

import paths
from export import unpack_codes
from .runtime import PackedLinear
from .bitwise import linear
from .benchmark import DEFAULT_EXPORT, capture, energy_meter, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    paths.require_mount()
    directory = paths.ARTIFACTS / "kernel-runs" / f"{time.strftime('%Y%m%dT%H%M%S')}-bitwise"
    directory.mkdir(parents=True, exist_ok=False)
    print("RESULTS", directory, flush=True)
    manifest = json.loads((args.export / "manifest.json").read_text())
    from safetensors import safe_open

    shapes = [("conv.pointwise_conv2", 64), ("feed_forward1.linear1", 128), ("feed_forward1.linear2", 1)]
    rows = []
    with paths.gpu_lock("packed FP32 LUT/SIMT experiment"), torch.inference_mode(), energy_meter() as meter:
        meter._guard()
        torch.manual_seed(419)
        torch.backends.cuda.matmul.allow_tf32 = False
        save(directory, "metadata.json", {"gpu": meter.metadata(), "torch": torch.__version__,
             "triton": triton.__version__, "export_sha256": manifest["sha256"],
             "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).parent.glob("*.py")},
             "scope": "hot-weight microbenchmarks, finite FP32 activations, no activation quantization; not model energy"})
        with safe_open(args.export / manifest["file"], framework="pt") as f:
            for suffix, m in shapes:
                name = "encoder.layers.0." + suffix
                n, k = manifest["quantized_layers"][name]["shape"]
                p, scale = f.get_tensor(name + ".codes"), f.get_tensor(name + ".scale")
                layer = PackedLinear(p, scale, k).cuda()
                w = (unpack_codes(p, k).float() * scale[:, None]).cuda()
                x = torch.randn(m, k, device="cuda")
                reference = torch.nn.functional.linear(x, w)
                candidates = []
                for bm, bn, bg, split in itertools.product((1, 4, 8, 16), (16, 32), (16, 32), (1, 4)):
                    if m == 1 and bm != 1:
                        continue
                    config = (bm, bn, bg, split, 4)
                    try:
                        fn = lambda: linear(x, layer, config=config)
                        y = fn()
                        torch.testing.assert_close(y, reference, atol=1e-4, rtol=2e-5)
                        us = triton.testing.do_bench_cudagraph(fn, rep=20) * 1000
                        candidates.append({"config": config, "us": us, "max_abs_error": float((y-reference).abs().max())})
                    except triton.runtime.errors.OutOfResources:
                        continue
                winner = min(candidates, key=lambda r: r["us"])
                variants = {
                    "packed_dispatch": lambda: layer(x),
                    "fp32_lookup": lambda: linear(x, layer, config=winner["config"]),
                    "fp32_simt": lambda: linear(x, layer, variant="simt"),
                }
                row = {"layer": name, "m": m, "n": n, "k": k, "candidates": candidates, "winner": winner,
                       "latency_us": {}}
                for label, fn in variants.items():
                    torch.testing.assert_close(fn(), reference, atol=1e-4, rtol=2e-5)
                    row["latency_us"][label] = [triton.testing.do_bench_cudagraph(fn, rep=50)*1000 for _ in range(3)]
                print("TIMING", name, m, "winner", winner, "median_us",
                      {label: statistics.median(v) for label, v in row["latency_us"].items()}, flush=True)
                # One representative encoder projection; graph batches amortize host launch costs.
                if m == 128 and args.seconds:
                    graphs, outputs = {}, {}
                    for label, fn in variants.items():
                        def work(fn=fn):
                            for _ in range(32):
                                y = fn()
                            return (y,)
                        graphs[label], outputs[label] = capture(work)
                    windows = []
                    for repeat in range(args.repeats):
                        order = list(variants)
                        if repeat % 2:
                            order.reverse()
                        for label in order:
                            warm = time.perf_counter()
                            while time.perf_counter() - warm < 1:
                                graphs[label].replay()
                                torch.cuda.synchronize()
                            window = meter.measure(graphs[label].replay, min_seconds=args.seconds)
                            window.update(variant=label, repeat=repeat, calls_per_iteration=32,
                                          joules_per_call=window["joules_per_iteration"]/32)
                            windows.append(window)
                            print("ENERGY", label, window["joules_per_call"], window["average_watts"], flush=True)
                            save(directory, "energy.json", windows)
                    row["energy_summary"] = {
                        label: {"joules_per_call": statistics.median(w["joules_per_call"] for w in windows if w["variant"] == label),
                                "watts": statistics.median(w["average_watts"] for w in windows if w["variant"] == label)}
                        for label in variants}
                    del graphs, outputs
                rows.append(row)
                save(directory, "comparison.json", rows)


if __name__ == "__main__":
    main()
