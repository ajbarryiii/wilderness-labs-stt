"""Kernel-level breakdown (torch.profiler / CUPTI) of one provider call per variant: which kernels run
and how long each takes on the GPU. Eager calls after warm-up; durations are GPU kernel times.

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.profile_ops --call out --m 38
"""
from __future__ import annotations

import argparse
import collections
import json

import torch

from gpu1bit.shared import env
from gpu1bit.int8_mma import bench, gpulock
from gpu1bit.int8_mma.kernels import DotConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calls", nargs="+", default=["out"])
    p.add_argument("--m", type=int, nargs="+", default=[38])
    p.add_argument("--variants", nargs="+", default=list(bench.VARIANTS))
    p.add_argument("--config", default="", help="B dot config wm,wn,rm,rn,stages,cluster")
    p.add_argument("--reps", type=int, default=50)
    args = p.parse_args()
    cfg = DotConfig(*map(int, args.config.split(","))) if args.config else None
    with gpulock.locked("B:kernel breakdown"), torch.inference_mode():
        tensors = bench.mb1_tensors()
        for call in args.calls:
            codes, scale, norm = bench.block_weights(tensors, call)
            for m in args.m:
                x = bench.make_input(call, m)
                for v in args.variants:
                    var = bench.Variant(v, call, codes, scale, norm, b_config=cfg if v == "B" else None)
                    for _ in range(5):
                        var(x, var.base)
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for _ in range(args.reps):
                            var(x, var.base)
                        torch.cuda.synchronize()
                    totals = collections.defaultdict(lambda: [0.0, 0])
                    for e in prof.events():
                        if e.device_type == torch.autograd.DeviceType.CUDA:
                            totals[e.name][0] += e.device_time / args.reps
                            totals[e.name][1] += 1
                    rows = sorted(((n[:60], round(t, 2), c // args.reps) for n, (t, c) in totals.items()), key=lambda r: -r[1])
                    print(json.dumps({"call": call, "m": m, "variant": v, "total_us": round(sum(r[1] for r in rows), 2),
                                      "kernels": rows}), flush=True)


if __name__ == "__main__":
    main()
