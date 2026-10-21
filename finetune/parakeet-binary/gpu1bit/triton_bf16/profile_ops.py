"""Per-kernel GPU durations (CUPTI via torch.profiler) of one op: where a small-M op's time goes.

Runs each variant eagerly REPEAT times after warm-up and reports the median duration of every kernel
it launches (GPU time only; launch gaps excluded). Compare with bench.py's graph-replay latency.

    timeout 600 gpu1bit/shared/python -m gpu1bit.triton_bf16.profile_ops --shape out --m 38 \
        --cands fp16x2p@16,32,64,8,2,3,serial
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics

import torch
from safetensors import safe_open

from gpu1bit.triton_bf16.gpulock import locked
from gpu1bit.shared import env
from gpu1bit.triton_bf16.bench import SHAPES, baselines, candidates, load_weights, make_input

REPEAT = 50


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", default="out")
    p.add_argument("--m", type=int, default=38)
    p.add_argument("--cands", nargs="+", default=[])
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--ops", action="store_true", help="A's layers via project() with provider semantics")
    p.add_argument("--ms", type=int, nargs="+", default=None)
    args = p.parse_args()
    label, n, k, kind, names = next(s for s in SHAPES if s[0] == args.shape)
    with locked("A:profile"), torch.inference_mode():
        with safe_open(str(env.MB1_EXPORT / "export.safetensors"), framework="pt") as f:
            codes, scale = load_weights(f, names)
        fns = {}
        if args.baselines:
            base, keep = baselines(codes, scale, k, kind)
            fns.update(base)
        for m in (args.ms or [args.m]):
            run(label, n, k, kind, codes, scale, fns if m == args.m else {}, args, m)


def run(label, n, k, kind, codes, scale, base_fns, args, m):
    import types
    from gpu1bit.triton_bf16.layer import TritonBF16Layer
    from gpu1bit.triton_bf16.tune import SEMANTICS
    fns = dict(base_fns)
    x = make_input(label, kind, m, k, seed=1)
    if args.ops:
        use_norm, act, in_act = SEMANTICS[label]
        norm = types.SimpleNamespace(normalized_shape=(k,), eps=1e-5, weight=torch.rand(k, device="cuda") + .5,
                                     bias=torch.randn(k, device="cuda") * .1) if use_norm else None
        for spec in args.cands:
            layer = TritonBF16Layer.prepare(codes, scale, None, k, "cuda", mode=spec)
            fns[f"A-{spec}-op"] = (lambda layer=layer: lambda x: layer.project(
                x, conv=kind == "conv", activation=act, norm=norm, input_activation=in_act))()
    else:
        fns.update(candidates(codes, scale, k, kind, args.cands, m))
    if True:
        report = {}
        for name, fn in fns.items():
            for _ in range(5):
                fn(x)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(REPEAT):
                    fn(x)
                torch.cuda.synchronize()
            per = collections.defaultdict(list)
            for e in prof.events():
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    per[e.name[:60]].append(e.device_time)
            kernels = {kname: round(statistics.median(v), 3) for kname, v in per.items()}
            report[name] = {"kernels_us": kernels, "sum_us": round(sum(kernels.values()), 3)}
            print(label, m, name, json.dumps(report[name]), flush=True)


if __name__ == "__main__":
    main()
