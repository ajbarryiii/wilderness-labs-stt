"""Agent A offline tile tuning on this 5090 (no inference-time autotuning; results feed configs.py).

For each provider shape and M: candidate (BM, BN, BK, SPLIT, warps, stages, REDUCE) are compiled in
parallel threads (Triton warmup: no launch), screened with short CUDA-graph timings, and the top
FINALISTS re-timed with more windows. Every timed config's output is checked against the FP64
reference first (rel err < 1e-6); failures and compile rejects are recorded. Real MB1 weights
(encoder layer 0), synthetic heavy-tailed activations with the pipeline's strides. Resumable:
entries already in --out are skipped. One process = one lock hold (stops before 10 minutes).

    timeout 600 gpu1bit/shared/python -m gpu1bit.triton_bf16.tune --modes fp16x2p --out <json>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import statistics
import time
import types
from pathlib import Path

import torch
import triton
from safetensors import safe_open

from gpu1bit.triton_bf16.gpulock import locked
from gpu1bit.shared import api, env
from gpu1bit.triton_bf16 import kernels
from gpu1bit.triton_bf16.bench import SHAPES, load_weights, make_input, time_graph
from gpu1bit.triton_bf16.layer import TritonBF16Layer

FINALISTS = 5


def stage1(m, n, k, mode):
    out = []
    bms = [b for b in (16, 32, 64, 128) if b <= max(16, triton.next_power_of_2(m))]
    for bm, bn, bk, split in itertools.product(bms, (32, 64, 128, 256), (64, 128, 256), (1, 2, 3, 4, 6, 8, 12, 16)):
        if bn > n or split > triton.cdiv(k, bk):
            continue
        ctas = triton.cdiv(m, bm) * triton.cdiv(n, bn) * split
        if not 60 <= ctas <= 1400:
            continue
        if bm * bn > 128 * 128 or bk * bn > 128 * 128:
            continue
        out.append((bm, bn, bk, split, 4, 3, "workspace", 1, 1))
    return out


def stage2(best, m, n, k):
    out = []
    for cfg in best:
        bm, bn, bk, split, _, _, reduce, groups, ew = cfg
        for warps, stages in itertools.product((2, 4, 8), (2, 3, 4)):
            if warps * 32 * 4 > bm * bn:   # fewer than 4 output elements per thread: skip
                continue
            out.append((bm, bn, bk, split, warps, stages, reduce, groups, ew))
        if bk >= 128:
            out.append((bm, bn, bk // 2, split, 4, 3, reduce, groups, ew))
        out.append((bm, bn, bk, split, 4, 3, "serial" if reduce == "workspace" else "workspace", groups, ew))
        out.append((bm, bn, bk, split, 4, 3, reduce, groups, 1 - ew))
    return list(dict.fromkeys(out))


# provider semantics per op (gpu1bit/shared/provider.py; H's P-kernel ops): (norm, activation, input_activation)
SEMANTICS = {"qkv": (True, "", ""), "out": (False, "", ""), "ff_up": (True, "silu", ""), "ff_down": (False, "", ""),
             "pw1": (True, "", ""), "pw2": (False, "", "silu")}


def op_reference(x, codes, scale, k, conv, norm, act, in_act):
    xx = (x.transpose(1, 2) if conv else x).double()
    if norm is not None:
        xx = torch.nn.functional.layer_norm(xx, (k,), norm.weight.double(), norm.bias.double(), norm.eps)
    if in_act == "silu":
        xx = torch.nn.functional.silu(xx)
    y = api.reference_linear(xx, codes, scale, None, k)
    if act == "silu":
        y = torch.nn.functional.silu(y)
    return y.transpose(1, 2) if conv else y


def evaluate(configs, layer, kind, x, ref, m, fast, sem=(False, "", ""), norm=None):
    conv = kind == "conv"
    _, act, in_act = sem
    norm_t = None if norm is None else (norm.weight, norm.bias, norm.eps)

    def warm(cfg):
        torch.cuda.set_device(0)
        x3 = x.transpose(1, 2) if conv else x
        t, n = x3.shape[1], layer.out_features
        y = torch.empty(1, device="cuda")
        try:
            kernels.bgemm(x3, layer.bits_t, layer.scale, None, layer.counters, n, layer.in_features, y,
                          (t * n, n, 1), mode=layer.mode, config=cfg, warmup=True, activation=act,
                          input_activation=in_act, norm=norm_t)
            return None
        except Exception as exc:
            return repr(exc)[:200]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        compile_errors = dict(zip(configs, pool.map(warm, configs)))
    rows = []
    for cfg in configs:
        if compile_errors[cfg]:
            rows.append({"config": cfg, "error": compile_errors[cfg]})
            continue
        layer.configs = {m: cfg}
        fn = lambda: layer.project(x, conv=conv, components=3, activation=act, norm=norm, input_activation=in_act)
        try:
            err = api.relative_error(fn(), ref)
            if not err < 1e-6:
                rows.append({"config": cfg, "error": f"rel err {err}"})
                continue
            if fast:
                us, _ = time_graph(fn, reps=10, windows=3, inner=4)
            else:
                us = statistics.median(time_graph(fn, reps=20, windows=7, inner=10)[0] for _ in range(3))
            rows.append({"config": cfg, "us": us, "rel_err": err})
        except Exception as exc:
            rows.append({"config": cfg, "error": repr(exc)[:200]})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modes", nargs="+", default=["fp16x2p"])
    p.add_argument("--ms", type=int, nargs="+", default=[38, 126, 376])
    p.add_argument("--shapes", nargs="+", default=[s[0] for s in SHAPES])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budget-s", type=float, default=540)
    p.add_argument("--ops", action="store_true", help="time layer.project with each op's provider semantics")
    args = p.parse_args()
    env.require_mount()
    done = json.loads(args.out.read_text()) if args.out.exists() else []
    keys = {(r["shape"], r["m"], r["mode"]) for r in done}
    torch.backends.cuda.matmul.allow_tf32 = False
    with locked("A:tune") as lock, torch.inference_mode():
        with safe_open(str(env.MB1_EXPORT / "export.safetensors"), framework="pt") as f:
            for (label, n, k, kind, names), mode, m in itertools.product(SHAPES, args.modes, args.ms):
                if label not in args.shapes or (label, m, mode) in keys:
                    continue
                if time.monotonic() - lock["acquired"] > args.budget_s:
                    print("budget reached; rerun to resume", flush=True)
                    break
                codes, scale = load_weights(f, names)
                layer = TritonBF16Layer.prepare(codes, scale, None, k, "cuda", mode=mode)
                x = make_input(label, kind, m, k, seed=m + n)
                sem = SEMANTICS[label] if args.ops else (False, "", "")
                norm = None
                if sem[0]:   # stand-in LayerNorm (timing does not depend on its values)
                    norm = types.SimpleNamespace(normalized_shape=(k,), eps=1e-5,
                                                 weight=torch.rand(k, device="cuda") + 0.5,
                                                 bias=torch.randn(k, device="cuda") * 0.1)
                ref = op_reference(x, codes, scale, k, kind == "conv", norm, sem[1], sem[2])
                t0 = time.monotonic()
                ev = lambda cfgs, fast: evaluate(cfgs, layer, kind, x, ref, m, fast, sem, norm)
                first = ev(stage1(m, n, k, mode), True)
                good = sorted((r for r in first if "us" in r), key=lambda r: r["us"])
                second = ev(stage2([r["config"] for r in good[:3]], m, n, k), True)
                pool = sorted((r for r in first + second if "us" in r), key=lambda r: r["us"])
                finals = ev(list(dict.fromkeys(r["config"] for r in pool[:FINALISTS])), False)
                finals = sorted((r for r in finals if "us" in r), key=lambda r: r["us"])
                row = {"shape": label, "n": n, "k": k, "kind": kind, "m": m, "mode": mode, "semantics": sem,
                       "best": finals[0], "finalists": finals, "screened": len(first) + len(second),
                       "rejected": [r for r in first + second if "error" in r][:20],
                       "seconds": round(time.monotonic() - t0, 1)}
                done.append(row)
                args.out.write_text(json.dumps(done, indent=1))
                print(json.dumps({kk: row[kk] for kk in ("shape", "m", "mode", "best", "screened", "seconds")}),
                      flush=True)
        print("held_s", round(env.check_hold(lock, margin_s=0), 1), "wrapper", env.wrapper()["wrapper"])


if __name__ == "__main__":
    main()
