"""Tile sweep for the exact deferred split-K (XCHG 5 + row_exact._row_epilogue_exact), B-hybrid's N = 1024 calls.

Per (K, M): the op pair the pipeline runs for one deferred projection followed by its row epilogue
(LayerNorm + INT8x3 quantization of the next input, factor 1, no bias, as MB1's layers), timed with
tune.time_graph (CUDA-graph replays of 100 pairs, median of 7 windows, hot weights):
  exact:   b1_dot with configs.choose (exact in-kernel finish) + M1's _row_epilogue (SPLIT = 1)
  frozen:  configs.choose_deferred (XCHG 4 FP32 slices) + M1's _row_epilogue (SPLIT = C)  [information]
  each XCHG 5 candidate (tune.deferred_candidates' tiles, pf dropped): dot_slices + _row_epilogue_exact,
           checked BITWISE against `exact` (y, fused q, qs) before timing.
One process, one gpu_lock; results under B/tune/<stamp>-exact-k<K>-m<M>/exact.json.

    CUDA_VISIBLE_DEVICES="" gpu1bit/shared/python -m gpu1bit.int8_mma.tune_exact --precompile --k 1024 4096 --m 38 126 256 368
    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.tune_exact --k 4096 --m 126
"""
from __future__ import annotations

import argparse
import json
import time
import types

import torch

from gpu1bit.int8_mma import configs, gpulock, hybrid_exact, kernels_exact, row_exact, tune
from gpu1bit.int8_mma.layer import m1_quantize
from gpu1bit.shared import env

N = 1024


def candidates(m: int, k: int) -> list[kernels_exact.ExactDeferredConfig]:
    return list(dict.fromkeys(kernels_exact.ExactDeferredConfig.of(c) for c in tune.deferred_candidates(m, N, k)))


def _compile(job):
    from gpu1bit.int8_mma import cuda
    values, b01 = job
    from gpu1bit.int8_mma.kernels import DotConfig
    cfg = DotConfig(*values)
    try:
        source, name = kernels_exact.dot_source(cfg, 0, b01)
        cuda.compile_cubin(source, name)
        return None
    except Exception as exc:  # noqa: BLE001 - reported
        return f"{cfg}: {exc!r}"[:300]


def precompile(args):
    import multiprocessing as mp
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit('precompile with CUDA_VISIBLE_DEVICES="" (CPU only, no GPU lock)')
    jobs = set()
    for k in args.k:
        for m in args.m:
            for c in candidates(m, k):
                jobs.add((tuple(getattr(c, f) for f in ("wm", "wn", "rm", "rn", "stages", "cluster", "xchg", "wk", "pf")), False))
    for c in hybrid_exact.TUNED_EXACT_DEFERRED.values():
        if c is not None:
            for b01 in (False, True):
                jobs.add((tuple(getattr(c, f) for f in ("wm", "wn", "rm", "rn", "stages", "cluster", "xchg", "wk", "pf")), b01))
    print(f"compiling {len(jobs)} kernels", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        failures = [f for f in pool.imap_unordered(_compile, sorted(jobs), chunksize=4) if f]
    print(f"done; {len(failures)} failures", *failures[:20], sep="\n")


def sweep(args, lock, out):
    from inference.row_fusion import DeferredProjection
    results = []
    for k in args.k:
        for m in args.m:
            case = tune.Case(N, k, m)
            layer = case.layer
            g = torch.Generator().manual_seed(m + k)
            norm = torch.nn.LayerNorm(N)
            with torch.no_grad():
                norm.weight.copy_(1 + 0.1 * torch.randn(N, generator=g))
                norm.bias.copy_(0.1 * torch.randn(N, generator=g))
            norm = norm.cuda()
            residual = torch.randn(1, m, N, device="cuda")
            x = torch.randn(1, m, k, device="cuda")
            q, qs = m1_quantize(x)
            provider = types.SimpleNamespace(row_warps="adaptive", prequantized={},
                                             null=torch.empty(0, device="cuda", dtype=torch.uint8))

            def finish(value):
                provider.prequantized.clear()
                y = row_exact._ORIGINAL(provider, value, residual, 1.0, norm) if not isinstance(
                    value, row_exact.ExactDeferredProjection) else row_exact.finish_residual_exact(provider, value, residual, 1.0, norm)
                q2, qs2 = provider.prequantized.popitem()[1]
                return y, q2, qs2

            exact_cfg = configs.choose(m, N, k)
            yv = torch.empty(1, m, N, device="cuda")

            def exact_pair():
                layer.dot(q, qs, k, m, yv.view(m, N), config=exact_cfg)
                return finish(DeferredProjection(yv, layer.scale, None, 1, (1, m, N)))
            ref = [t.clone() for t in exact_pair()]
            exact_us = tune.time_graph([exact_pair] * 100)
            frozen_cfg = configs.choose_deferred(m, N, k)
            frozen_us = None
            if frozen_cfg is not None:
                part = torch.empty(frozen_cfg.cluster, m, N, device="cuda")

                def frozen_pair():
                    layer.dot(q, qs, k, m, part[0], config=frozen_cfg)
                    return finish(DeferredProjection(part, layer.scale, None, frozen_cfg.cluster, (1, m, N)))
                frozen_pair()
                frozen_us = tune.time_graph([frozen_pair] * 100)
            rows = []
            for cfg in candidates(m, k):
                env.check_hold(lock, margin_s=45)
                slices = torch.empty(cfg.cluster, 2, m, N, device="cuda", dtype=torch.int32)

                def pair(cfg=cfg, slices=slices):
                    hybrid_exact.dot_slices(layer, q, qs, k, m, slices, cfg)
                    return finish(row_exact.ExactDeferredProjection(slices, layer.scale, None, cfg.cluster, (1, m, N), qs, 0))
                got = pair()
                torch.cuda.synchronize()
                if not all(torch.equal(a, b) for a, b in zip(ref, got)):
                    raise AssertionError(f"{cfg.label()} is not bitwise the exact path")
                rows.append({"config": cfg.label(), "cfg": [cfg.wm, cfg.wn, cfg.rm, cfg.rn, cfg.stages, cfg.cluster, cfg.xchg, cfg.wk, cfg.pf],
                             "us": tune.time_graph([pair] * 100)})
            rows.sort(key=lambda r: r["us"])
            summary = {"n": N, "k": k, "m": m, "exact_config": exact_cfg.label(), "exact_plus_epilogue_us": exact_us,
                       "frozen_deferred_config": frozen_cfg.label() if frozen_cfg else None, "frozen_deferred_us": frozen_us,
                       "candidates": len(rows), "best": rows[0] if rows else None, "bitwise_checked": len(rows)}
            results.append({**summary, "rows": rows})
            (out / "exact.json").write_text(json.dumps(results, indent=1) + "\n")
            print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, nargs="+", default=[1024, 4096])
    p.add_argument("--m", type=int, nargs="+", default=[38, 126, 256, 368])
    p.add_argument("--precompile", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    if args.precompile:
        return precompile(args)
    env.require_shared_wrapper()
    out = tune.ARTIFACTS / (time.strftime("%Y%m%dT%H%M%S") + f"-exact-k{'-'.join(map(str, args.k))}-m{'-'.join(map(str, args.m))}")
    out.mkdir(parents=True, exist_ok=False)
    with gpulock.locked(f"B:exact deferred sweep k{args.k} m{args.m}") as lock, torch.inference_mode():
        sweep(args, lock, out)
        print("RESULTS", out)


if __name__ == "__main__":
    main()
