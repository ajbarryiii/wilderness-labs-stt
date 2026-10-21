"""Agent A screening benchmark (phase-1 prototyping; H's shared/bench.py P-kernel is the recorded protocol).

Per provider shape (N, K, kind) of shapes.json's M1 runtime calls and M in {38, 126, 376}: warm
CUDA-graph replay latency of one op (a graph of REPS back-to-back calls, replayed; median of
WINDOWS windows), real MB1 weights (encoder layer 0; QKV = cat(q, k, v) as M1's fused [3072, 1024]),
synthetic heavy-tailed activations with the pipeline's input strides. Weights and activations are
L2-resident ("hot", the M1 micro convention). Baselines are M1's kernels on MB1's derived 2-bit layout,
called as M1's provider calls them (INT8x3 includes its quantization kernel):
  R-bf16x3  inference.kernels.matmul(mode="bf16x3")
  R-compact inference.fused_kernels.int8_matmul(components=3)
  R-expanded inference.encoder4.ExpandedProjection("triton_col_tuned")
  R-hybrid  inference.ptx_cuda.PtxCudaProjection(tuning="native") (standalone op: no row fusion)

    timeout 600 gpu1bit/shared/python -m gpu1bit.triton_bf16.bench [--ms 38 126 376] [--cands ...]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from safetensors import safe_open

from gpu1bit.triton_bf16.gpulock import locked
from gpu1bit.shared import api, env

SHAPES = [  # (label, N, K, kind, module names in layer 0)
    ("qkv", 3072, 1024, "linear", ["self_attn.linear_q", "self_attn.linear_k", "self_attn.linear_v"]),
    ("out", 1024, 1024, "linear", ["self_attn.linear_out"]),
    ("ff_up", 4096, 1024, "linear", ["feed_forward1.linear1"]),
    ("ff_down", 1024, 4096, "linear", ["feed_forward1.linear2"]),
    ("pw1", 2048, 1024, "conv", ["conv.pointwise_conv1"]),
    ("pw2", 1024, 1024, "conv", ["conv.pointwise_conv2"]),
]


def load_weights(f, names):
    codes = torch.cat([f.get_tensor(f"encoder.layers.0.{n}.codes") for n in names])
    scale = torch.cat([f.get_tensor(f"encoder.layers.0.{n}.scale") for n in names])
    return codes, scale


def make_input(label, kind, m, k, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g) * torch.exp(0.5 * torch.randn(m, k, generator=g))
    x = x.cuda()
    if kind == "linear":
        return x.unsqueeze(0)                          # [1, T, K] contiguous
    if label == "pw2":
        return x.t().contiguous().unsqueeze(0)         # [1, K, T] contiguous (depthwise-conv output)
    return x.unsqueeze(0).transpose(1, 2)              # [1, K, T] view of [1, T, K] (NeMo pw1)


def time_graph(fn, reps=20, windows=7, inner=10):
    fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            for _ in range(reps):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        graph.replay()
    out = []
    for _ in range(windows):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner):
            graph.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000 / (inner * reps))
    del graph
    return statistics.median(out), out


def baselines(codes, scale, k, kind):
    import export as ternary_export
    from inference.runtime import PackedLinear, PackedPointwiseConv1d
    from inference import kernels as m1k
    from inference.fused_kernels import int8_matmul
    from inference.encoder4 import ExpandedProjection
    signs = api.unpack_signs(codes, k)
    cls = PackedPointwiseConv1d if kind == "conv" else PackedLinear
    mod = cls(ternary_export.pack_codes(signs), scale.clone(), k).cuda()
    conv = kind == "conv"
    expanded = ExpandedProjection("triton_col_tuned")
    out = {
        "R-bf16x3": lambda x: m1k.matmul(x, mod.packed_t, mod.scale, None, k, conv=conv, mode="bf16x3"),
        "R-compact": lambda x: int8_matmul(x, mod, conv, 3),
        "R-expanded": lambda x: expanded(x, mod, conv, 3),
    }
    try:
        from inference.ptx_cuda import PtxCudaProjection
        hybrid = PtxCudaProjection(tuning="native", row_warps="adaptive")
        out["R-hybrid"] = lambda x: hybrid(x, mod, conv, 3)
    except Exception as exc:  # recorded, not hidden
        print("R-hybrid unavailable:", repr(exc))
    return out, (mod, expanded)


def candidates(codes, scale, k, kind, names, m):
    from gpu1bit.triton_bf16.layer import TritonBF16Layer
    out = {}
    for spec in names:
        mode, _, cfg = spec.partition("@")
        configs = {m: tuple(int(v) if v.isdigit() else v for v in cfg.split(","))} if cfg else None
        layer = TritonBF16Layer.prepare(codes, scale, None, k, "cuda", mode=mode, configs=configs)
        out[f"A-{spec}"] = (lambda x, layer=layer: layer.conv(x)) if kind == "conv" else (lambda x, layer=layer: layer.linear(x))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ms", type=int, nargs="+", default=[38, 126, 376])
    p.add_argument("--shapes", nargs="+", default=[s[0] for s in SHAPES])
    p.add_argument("--cands", nargs="+", default=["bf16x3", "bf16x3t", "fp16x2"])
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--tag", default="screen")
    args = p.parse_args()
    env.require_mount()
    outdir = env.ARTIFACTS_ROOT / "A" / f"{time.strftime('%Y%m%dT%H%M%S')}-{args.tag}"
    outdir.mkdir(parents=True)
    rows = []
    torch.backends.cuda.matmul.allow_tf32 = False
    with locked("A:bench") as lock, torch.inference_mode():
        meta = {"wrapper": env.wrapper(), "lock": lock, "args": vars(args),
                "power_limit_w": torch.cuda.get_device_properties(0).name,
                "sources": env.source_hashes(env.GPU1BIT / "triton_bf16")}
        with safe_open(str(env.MB1_EXPORT / "export.safetensors"), framework="pt") as f:
            for label, n, k, kind, names in SHAPES:
                if label not in args.shapes:
                    continue
                codes, scale = load_weights(f, names)
                base, keep = ({}, None) if args.no_baselines else baselines(codes, scale, k, kind)
                for m in args.ms:
                    env.check_hold(lock)
                    x = make_input(label, kind, m, k, seed=m + n)
                    ref = api.reference(kind, x, codes, scale, None, k)
                    fns = dict(base)
                    fns.update(candidates(codes, scale, k, kind, args.cands, m))
                    for name, fn in fns.items():
                        try:
                            y = fn(x)
                            err = api.relative_error(y, ref)
                            us, windows = time_graph(lambda: fn(x))
                            row = dict(shape=label, n=n, k=k, kind=kind, m=m, variant=name, us=round(us, 3),
                                       windows_us=[round(w, 3) for w in windows], rel_err=err)
                        except Exception as exc:
                            row = dict(shape=label, n=n, k=k, kind=kind, m=m, variant=name, error=repr(exc)[:300])
                        rows.append(row)
                        print(json.dumps({kk: v for kk, v in row.items() if kk != "windows_us"}), flush=True)
                del base, keep
        meta["held_s"] = env.check_hold(lock, margin_s=0)
        (outdir / "screen.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
        print("RESULTS", outdir)


if __name__ == "__main__":
    main()
