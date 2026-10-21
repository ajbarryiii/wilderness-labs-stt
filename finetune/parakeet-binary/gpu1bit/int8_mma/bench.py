"""Agent B's per-call screening benchmark (P-kernel style, plans/gpu-kernel.md A6), until H's bench.

Each op is timed exactly as M1's transforms call the projection provider (activation quantization
with fused LayerNorm / input SiLU, matmul, epilogue with optional SiLU), on MB1 block-0 weights
(R-* variants on the derived 2-bit layout, H's gpu1bit.shared.m1.derived_packed_linear):
  B           Int8MmaLayer.project (this directory)
  R-expanded  inference.encoder4.ExpandedProjection("triton_col_tuned")
  R-hybrid    inference.ptx_cuda.PtxCudaProjection(tuning="native", row_warps="adaptive"), the
              ptx_row_best projection called directly, i.e. WITHOUT its pipeline-level deferred
              epilogue / prequantized inputs (row_fusion.py): an upper bound on its per-call cost
  R-compact   inference.fused_kernels.int8_matmul (2-bit packed weights, INT8x3)
Modes: hot = one weight copy, 64 back-to-back calls per graph (weights L2-resident);
       cold = distinct weight copies totalling >= 3x the 96 MiB L2 per variant, one call each per
       graph (each call streams its weights from DRAM, as in the real 604 MB INT8 encoder).
Per window: NVML energy and host-timed graph replays (synchronized; graphs >= ~1 ms so the replay
overhead is amortized); per-call latency = median replay time / calls per graph. Windows alternate
variant order. One process, one gpu_lock, <= 600 s (A9); the 400 W cap and foreign PIDs are checked.

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.bench --m 38 --calls qkv out
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import types
from pathlib import Path

import torch

from gpu1bit.int8_mma import gpulock
from gpu1bit.shared import env

CALLS = {
    "qkv": dict(n=3072, k=1024, conv=False, act="", norm="norm_self_att", inact="",
                modules=["self_attn.linear_q", "self_attn.linear_k", "self_attn.linear_v"]),
    "out": dict(n=1024, k=1024, conv=False, act="", norm=None, inact="", modules=["self_attn.linear_out"]),
    "ff1": dict(n=4096, k=1024, conv=False, act="silu", norm="norm_feed_forward1", inact="",
                modules=["feed_forward1.linear1"]),
    "ff2": dict(n=1024, k=4096, conv=False, act="", norm=None, inact="", modules=["feed_forward1.linear2"]),
    "pw1": dict(n=2048, k=1024, conv=True, act="", norm="norm_conv", inact="", modules=["conv.pointwise_conv1"]),
    "pw2": dict(n=1024, k=1024, conv=True, act="", norm=None, inact="silu", modules=["conv.pointwise_conv2"]),
}
VARIANTS = ("B", "R-expanded", "R-hybrid", "R-compact")
L2_BYTES = 96 * 2**20
ARTIFACTS = env.ARTIFACTS_ROOT / "B"


def mb1_tensors():
    from safetensors.torch import load_file
    return load_file(env.MB1_EXPORT / "export.safetensors")


def block_weights(tensors, call):
    spec = CALLS[call]
    names = [f"encoder.layers.0.{m}" for m in spec["modules"]]
    codes = torch.cat([tensors[f"{n}.codes"] for n in names])
    scale = torch.cat([tensors[f"{n}.scale"] for n in names]).float()
    norm = None
    if spec["norm"]:
        norm = torch.nn.LayerNorm(spec["k"]).cuda()
        with torch.no_grad():
            norm.weight.copy_(tensors[f"encoder.layers.0.{spec['norm']}.weight"].float())
            norm.bias.copy_(tensors[f"encoder.layers.0.{spec['norm']}.bias"].float())
    return codes, scale, norm


class Variant:
    """Weight copies and a callable op for one variant / call."""

    def __init__(self, name, call, codes, scale, norm, b_config=None):
        spec = CALLS[call]
        self.name, self.spec, self.norm, self.b_config = name, spec, norm, b_config
        n, k = spec["n"], spec["k"]
        if name == "B":
            from gpu1bit.int8_mma.layer import Int8MmaLayer
            self.base = Int8MmaLayer.prepare(codes, scale, None, k, "cuda")
            self.copy_bytes = self.base.words.numel() * 4
        else:
            from gpu1bit.shared.m1 import derived_packed_linear
            mod = derived_packed_linear(codes, scale, None, k)
            self.base = types.SimpleNamespace(packed_t=mod.packed_t, scale=mod.scale, bias=None, out_features=n)
            if name == "R-expanded":
                from inference.encoder4 import ExpandedProjection
                self.provider = ExpandedProjection("triton_col_tuned")
                self.copy_bytes = n * k + mod.packed_t.numel() * 4
            elif name == "R-hybrid":
                from inference.ptx_cuda import PtxCudaProjection
                self.provider = PtxCudaProjection(tuning="native", row_warps="adaptive")
                self.copy_bytes = n * k + mod.packed_t.numel() * 4
            elif name == "R-compact":
                from inference.fused_kernels import int8_matmul
                self.provider = int8_matmul
                self.copy_bytes = mod.packed_t.numel() * 4
            else:
                raise ValueError(name)
        self.copies = [self.base]

    def make_copies(self, count):
        b = self.base
        while len(self.copies) < count:
            if self.name == "B":
                from gpu1bit.int8_mma.layer import Int8MmaLayer
                self.copies.append(Int8MmaLayer(b.out_features, b.in_features, b.device, b.words.clone(), None,
                                                b.scale, None))
            else:
                self.copies.append(types.SimpleNamespace(packed_t=b.packed_t.clone(), scale=b.scale, bias=None,
                                                         out_features=b.out_features))

    def __call__(self, x, copy):
        s = self.spec
        if self.name == "B":
            if self.b_config is None:
                return copy.project(x, conv=s["conv"], components=3, activation=s["act"], norm=self.norm,
                                    input_activation=s["inact"])
            from gpu1bit.int8_mma.layer import m1_quantize
            xx = x.transpose(1, 2) if s["conv"] else x
            b, t, k = xx.shape
            q, qs = m1_quantize(xx, self.norm, s["inact"])
            y = torch.empty(b, t, copy.out_features, device="cuda")
            copy.dot(q, qs, k, b * t, y.view(b * t, -1), silu=s["act"] == "silu", config=self.b_config)
            return y
        return self.provider(x, copy, s["conv"], 3, s["act"], self.norm, None, s["inact"])


def make_input(call, m):
    s = CALLS[call]
    if s["conv"] and s["inact"]:
        return torch.randn(1, s["k"], m, device="cuda")       # pointwise_conv2: contiguous [B, C, T]
    x = torch.randn(1, m, s["k"], device="cuda")
    return x.transpose(1, 2) if s["conv"] else x               # pointwise_conv1: view of [B, T, K]


def capture(variant, x, mode):
    if mode == "hot":
        sequence = [variant.copies[0]] * 64
    else:
        count = min(4096, max(8, -(-3 * L2_BYTES // variant.copy_bytes)))
        variant.make_copies(count)
        sequence = variant.copies[:count]
    for c in (sequence[:1] if mode == "hot" else sequence):  # warm: compile kernels, M1's INT8 expansion
        variant(x, c)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in sequence:  # outputs are dropped: the graph pool reuses their memory call to call
            variant(x, c)
    graph.replay()
    torch.cuda.synchronize()
    return graph, len(sequence)


def measure(meter, graph, calls, seconds):
    started = time.perf_counter()
    while time.perf_counter() - started < 0.05:
        graph.replay()
    torch.cuda.synchronize()
    r = meter.measure(graph.replay, min_seconds=seconds)
    n_iter = len(r["iteration_seconds"])
    return {"us_per_call": 1e6 * statistics.median(r["iteration_seconds"]) / calls,
            "uj_per_call": 1e6 * r["energy_joules"] / (n_iter * calls),
            "watts": r["energy_joules"] / r["elapsed_seconds"], "iterations": n_iter}


def summarize(windows):
    return {key: statistics.median(w[key] for w in windows) for key in ("us_per_call", "uj_per_call", "watts")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, nargs="+", default=[38, 126, 368])
    p.add_argument("--calls", nargs="+", default=list(CALLS), choices=list(CALLS))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--modes", nargs="+", default=["hot", "cold"], choices=["hot", "cold"])
    p.add_argument("--windows", type=int, default=5)
    p.add_argument("--seconds", type=float, default=0.15)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    env.require_shared_wrapper()
    env.require_mount()
    out = ARTIFACTS / "bench" / (time.strftime("%Y%m%dT%H%M%S") + (f"-{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=False)
    from inference.benchmark import energy_meter
    from inference.sprint4 import require_power_cap
    results = []
    with gpulock.locked("B:per-call screening bench") as lock, torch.inference_mode(), energy_meter() as meter:
        meter._guard()
        limit = require_power_cap(meter)
        if limit != 400:
            raise RuntimeError(f"power limit {limit} W, expected exactly 400 W (A9)")
        meta = {"wrapper": env.wrapper(), "gpu": meter.metadata(), "args": vars(args),
                "lock": {k: v for k, v in lock.items() if k != "acquired"}, "torch": torch.__version__,
                "source_sha256": env.source_hashes(Path(__file__).resolve().parent)}
        (out / "metadata.json").write_text(json.dumps(meta, indent=1))
        tensors = mb1_tensors()
        torch.manual_seed(20261020)
        for call in args.calls:
            codes, scale, norm = block_weights(tensors, call)
            variants = {v: Variant(v, call, codes, scale, norm) for v in args.variants}
            for m in args.m:
                x = make_input(call, m)
                for mode in args.modes:
                    env.check_hold(lock, margin_s=60)
                    graphs = {v: capture(variants[v], x, mode) for v in args.variants}
                    windows = {v: [] for v in args.variants}
                    for w in range(args.windows):
                        for v in (args.variants if w % 2 == 0 else list(reversed(args.variants))):
                            require_power_cap(meter)
                            windows[v].append(measure(meter, graphs[v][0], graphs[v][1], args.seconds))
                    row = {"call": call, "m": m, "mode": mode,
                           **{k: CALLS[call][k] for k in ("n", "k", "conv", "act", "norm", "inact")},
                           "calls_per_graph": {v: graphs[v][1] for v in args.variants},
                           "copy_bytes": {v: variants[v].copy_bytes for v in args.variants},
                           "median": {v: summarize(windows[v]) for v in args.variants}, "windows": windows}
                    results.append(row)
                    (out / "results.json").write_text(json.dumps(results, indent=1))
                    print(json.dumps({"call": call, "m": m, "mode": mode,
                                      **{v: "%.2f us %.1f uJ %.0f W" % tuple(row["median"][v][k] for k in
                                                                             ("us_per_call", "uj_per_call", "watts"))
                                         for v in args.variants}}), flush=True)
                    del graphs
                    torch.cuda.synchronize()
                    for v in variants.values():
                        del v.copies[1:]
                    torch.cuda.empty_cache()
        print("RESULTS", out)


if __name__ == "__main__":
    main()
