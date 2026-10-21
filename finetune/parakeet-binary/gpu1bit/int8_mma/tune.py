"""Tile/split sweep for b1_dot on prequantized INT8x3 activations (dot + epilogue only).

For each (N, K, M): every candidate config is compiled, checked BITWISE against the first candidate
(all K splits are exact INT32, so every config must give identical output), then timed with CUDA
events over CUDA-graph replays: hot (one weight copy, 200 launches per graph) and cold (distinct
weight copies totalling >= 3x L2, one launch each). Median of windows. Also times M1's dot ops on
the same q/qs for reference: R-expanded `_int8_dot` (+ `_finish` when split > 1) with
expanded_configuration tiles and R-hybrid's native ptx_row_best kernel (PtxCudaProjection, quantize
excluded by timing only its dot via the prequantized cache). One process, one gpu_lock (A9).

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.tune --shapes 1024x1024 --m 38
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time

import torch
import triton

from gpu1bit.int8_mma import configs, cuda, gpulock, kernels
from gpu1bit.int8_mma.kernels import DotConfig
from gpu1bit.int8_mma.layer import Int8MmaLayer, m1_quantize
from gpu1bit.shared import api, env

L2_BYTES = 96 * 2**20
ARTIFACTS = env.ARTIFACTS_ROOT / "B" / "tune"


def candidates(m, n, k, wide=False, exchanges=(1, 3)):
    out = []
    mt = -(-m // 16)
    for wm, wn, wk, rm, rn, stages, cluster in itertools.product(
            (1, 2), (2, 4, 8), (1, 2), (1, 2, 3, 4), (1, 2, 3, 4), (2, 3, 4), (1, 2, 3, 4, 6, 8)):
        for xchg in ((0,) if cluster == 1 else exchanges):
            cfg = DotConfig(wm, wn, rm, rn, stages, cluster, xchg, wk)
            try:
                cfg.validate()
            except ValueError:
                continue
            kb = -(-k // 128)
            per_rank = -(-kb // cluster)
            if cluster > kb or per_rank < wk or (stages > 2 and -(-per_rank // wk) < 2):
                continue
            if not 64 <= cfg.threads <= 512 or 3 * rm * rn * 4 > 128:
                continue
            if wm == 2 and m <= 64:
                continue
            if cfg.bmr > max(16, -(-mt // 1) * 16) and cfg.bmr > 2 * mt * 16:  # do not tile far beyond M
                continue
            if cfg.bmr > 16 * mt and wm * rm > 1:
                continue
            gx, gy, gz = cfg.grid(m, n)
            ctas = gx * gy * gz
            if not wide and not (96 <= ctas <= 360):
                continue
            out.append(cfg)
    return out


class Case:
    def __init__(self, n, k, m, seed=0):
        g = torch.Generator().manual_seed(seed)
        signs = torch.randint(0, 2, (n, k), generator=g, dtype=torch.int8) * 2 - 1
        self.codes = api.pack_signs(signs)
        self.scale = torch.rand(n, generator=g) * 0.05 + 1e-3
        self.layer = Int8MmaLayer.prepare(self.codes, self.scale, None, k, "cuda")
        self.signs = signs
        x = torch.randn(1, m, k, device="cuda")
        self.q, self.qs = m1_quantize(x)
        self.n, self.k, self.m = n, k, m
        self.copies = [self.layer]

    def copies_for(self, count):
        b = self.layer
        while len(self.copies) < count:
            self.copies.append(Int8MmaLayer(b.out_features, b.in_features, b.device, b.words.clone(), None, b.scale, None))
        return self.copies[:count]


def time_graph(fn_list, windows=7, launches_per_window=None):
    """fn_list: callables enqueuing one op each; returns median us per op over graph replays."""
    for f in fn_list[:4]:
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in fn_list:
            f()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(windows):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        graph.replay()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (2 * len(fn_list)))
    del graph
    return statistics.median(times)


def b_op(case, layer, cfg, y, silu=False):
    return lambda: layer.dot(case.q, case.qs, case.k, case.m, y, silu=silu, config=cfg)


def m1_ops(case, silu=False):
    """R-expanded's _int8_dot(+_finish) with its own tiles on an expanded INT8 copy, prequantized."""
    from inference.encoder4 import expanded_configuration
    from inference.fused_kernels import _int8_dot
    from inference.kernels import _finish
    m, n, k = case.m, case.n, case.k
    w = case.signs.cuda().contiguous()
    bm, bn, bk, split, warps, stages = expanded_configuration(m, n, k)
    y = torch.empty(m, n, device="cuda")
    partial = torch.empty(split, m, n, device="cuda") if split > 1 else y

    def op(w=w):
        _int8_dot[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
            case.q, case.qs, w, case.layer.scale, None, y, partial, m, n, k, False, bm, bn, bk, split, 3,
            ACTIVATION="silu" if silu else "", DENSE=True, COLUMN=True, num_warps=warps, num_stages=stages)
        if split > 1:
            _finish[(triton.cdiv(m * n, 256),)](partial, case.layer.scale, None, y, m, n, m, m * n, n, 1, False,
                                                split, 256, ACTIVATION="silu" if silu else "")
    return op, w, (bm, bn, bk, split, warps, stages)


def deferred_candidates(m, n, k):
    """xchg 4 (deferred FP32 slices for R-hybrid's row epilogue): no cluster, so no co-scheduling limit."""
    out = []
    for c in candidates(m, n, k, exchanges=(3,)):
        if c.cluster > 1:
            for pf in (0, 1):
                out.append(DotConfig(c.wm, c.wn, c.rm, c.rn, c.stages, c.cluster, 4, c.wk, pf))
    return list(dict.fromkeys(out))


def row_epilogue_op(case, partial, split, residual, y, q, qs, norm):
    """inference.row_fusion._row_epilogue as finish_residual(norm=next LayerNorm) launches it."""
    import triton as _triton
    from inference.row_fusion import _row_epilogue
    m, n = case.m, case.n
    warps = 8 if m <= 192 else 4

    def op():
        _row_epilogue[(m,)](partial, case.layer.scale, None, residual, y, q, qs, norm.weight, norm.bias, None, None,
                            m, n, split, 1.0, False, True, False, norm.eps, 0.0, _triton.next_power_of_2(n),
                            num_warps=warps, enable_fp_fusion=False)
    return op


def deferred_sweep(args, out, lock):
    """dot + row epilogue, per (N=1024 shape, M): deferred xchg-4 configs vs the exact choose() config."""
    results = []
    for shape, m in itertools.product(args.shapes, args.m):
        n, k = map(int, shape.split("x"))
        if n != 1024:
            continue
        case = Case(n, k, m)
        norm = torch.nn.LayerNorm(n).cuda()
        residual = torch.randn(m, n, device="cuda")
        y = torch.empty(m, n, device="cuda")
        q = torch.empty(3, m, n, device="cuda", dtype=torch.int8)
        qs = torch.empty(3, m, device="cuda")
        exact_cfg = configs.choose(m, n, k)
        out1 = torch.empty(m, n, device="cuda")
        ops = [b_op(case, case.layer, exact_cfg, out1), row_epilogue_op(case, out1, 1, residual, y, q, qs, norm)]
        def pair(ops):
            return lambda: [o() for o in ops]
        base = time_graph([pair(ops)] * 100)
        ref_y = None
        rows = []
        for cfg in deferred_candidates(m, n, k):
            env.check_hold(lock, margin_s=45)
            partial = torch.empty(cfg.cluster, m, n, device="cuda")
            dot = b_op(case, case.layer, cfg, partial[0])
            epi = row_epilogue_op(case, partial, cfg.cluster, residual, y, q, qs, norm)
            dot(); epi(); torch.cuda.synchronize()
            rows.append({"config": cfg.label(), "cfg": list(vars(cfg).values()), "hot_us": time_graph([pair([dot, epi])] * 100)})
        rows.sort(key=lambda r: r["hot_us"])
        summary = {"n": n, "k": k, "m": m, "exact_plus_epilogue_us": base, "exact_config": exact_cfg.label(),
                   "best_deferred": rows[0] if rows else None, "candidates": len(rows)}
        results.append({**summary, "rows": rows})
        (out / "deferred.json").write_text(json.dumps(results, indent=1))
        print(json.dumps(summary), flush=True)


def _compile_one(job):
    cfg_values, silu = job
    cfg = DotConfig(*cfg_values)
    source, name = kernels.dot_source(cfg, silu)
    try:
        cuda.compile_cubin(source, name)
        return None
    except Exception as exc:  # noqa: BLE001 - reported
        return f"{cfg.label()}: {exc!r}"[:300]


def precompile(args):
    """CPU only (NVRTC needs no GPU; no lock): fill the cubin cache for every candidate."""
    import multiprocessing as mp
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit('precompile with CUDA_VISIBLE_DEVICES="" (CPU only, no GPU lock)')
    jobs = set()
    for shape, m in itertools.product(args.shapes, args.m):
        n, k = map(int, shape.split("x"))
        cands = ([DotConfig(*map(int, c.split(","))) for c in args.configs] if args.configs
                 else candidates(m, n, k, args.wide))
        if args.deferred:
            cands = deferred_candidates(m, n, k) if n == 1024 else []
        for cfg in cands:
            jobs.add((tuple(vars(cfg).values()), False))
            if args.silu:
                jobs.add((tuple(vars(cfg).values()), True))
    print(f"compiling {len(jobs)} kernels", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        failures = [f for f in pool.imap_unordered(_compile_one, sorted(jobs), chunksize=4) if f]
    print(f"done; {len(failures)} failures", *failures[:20], sep="\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shapes", nargs="+", default=["1024x1024", "3072x1024", "4096x1024", "2048x1024", "1024x4096"])
    p.add_argument("--m", type=int, nargs="+", default=[38, 126, 368])
    p.add_argument("--modes", nargs="+", default=["hot", "cold"])
    p.add_argument("--configs", nargs="*", default=None, help="explicit wm,wn,rm,rn,stages,cluster[,xchg] list")
    p.add_argument("--top", type=int, default=8, help="cold-time only the best TOP hot configs")
    p.add_argument("--wide", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--precompile", action="store_true", help="CPU-only NVRTC pass into the cubin cache")
    p.add_argument("--deferred", action="store_true", help="time xchg-4 configs + R-hybrid's row epilogue")
    p.add_argument("--silu", action="store_true", help="precompile the SiLU-epilogue variants too")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    if args.precompile:
        return precompile(args)
    env.require_shared_wrapper()
    out = ARTIFACTS / (time.strftime("%Y%m%dT%H%M%S") + (f"-{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=False)
    results = []
    with gpulock.locked("B:b1_dot tile sweep") as lock, torch.inference_mode():
        if args.deferred:
            deferred_sweep(args, out, lock)
            print("RESULTS", out)
            return
        for shape, m in itertools.product(args.shapes, args.m):
            n, k = map(int, shape.split("x"))
            case = Case(n, k, m)
            cands = ([DotConfig(*map(int, c.split(","))) for c in args.configs] if args.configs
                     else candidates(m, n, k, args.wide))
            ref = None
            rows = []
            for cfg in cands:
                env.check_hold(lock, margin_s=45)
                y = torch.full((m, n), float("nan"), device="cuda")
                try:
                    b_op(case, case.layer, cfg, y)()
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001 - recorded
                    rows.append({"config": cfg.label(), "error": repr(exc)[:300]})
                    continue
                exact = cfg.exact
                if exact and ref is None:
                    ref = y.clone()
                elif exact and not torch.equal(y, ref):
                    raise AssertionError(f"{cfg.label()} differs from the first exact config on {shape} m={m}")
                if not exact:
                    y2 = torch.full_like(y, float("nan"))
                    b_op(case, case.layer, cfg, y2)()
                    if not torch.equal(y, y2):
                        raise AssertionError(f"{cfg.label()} is not deterministic")
                    if ref is not None and ((y - ref).norm() / ref.norm()).item() > 1e-6:
                        raise AssertionError(f"{cfg.label()} FP32-split result too far from the exact one")
                hot = time_graph([b_op(case, case.layer, cfg, y)] * 200)
                rows.append({"config": cfg.label(), "cfg": list(vars(cfg).values()), "hot_us": hot})
            ok = sorted((r for r in rows if "hot_us" in r), key=lambda r: r["hot_us"])
            if "cold" in args.modes:
                count = max(8, -(-3 * L2_BYTES // case.layer.words.numel() // 4))
                copies = case.copies_for(count)
                for r in ok[:args.top]:
                    cfg = DotConfig(*r["cfg"])
                    y = torch.empty(m, n, device="cuda")
                    r["cold_us"] = time_graph([b_op(case, c, cfg, y) for c in copies])
            op, w, m1cfg = m1_ops(case)
            m1_hot = time_graph([op] * 200)
            m1_cold = None
            if "cold" in args.modes:
                count = max(8, -(-3 * L2_BYTES // (n * k)))
                ws = [w] + [w.clone() for _ in range(count - 1)]
                m1_cold = time_graph([m1_ops_for(case, wi, m1cfg) for wi in ws])
            best = ok[0] if ok else None
            best_cold = min((r for r in ok if "cold_us" in r), key=lambda r: r["cold_us"], default=None)
            summary = {"n": n, "k": k, "m": m, "candidates": len(cands), "best_hot": best,
                       "best_cold": best_cold, "m1_expanded_dot": {"config": m1cfg, "hot_us": m1_hot, "cold_us": m1_cold},
                       "heuristic": configs.heuristic(m, n, k).label()}
            results.append({**summary, "rows": rows})
            (out / "results.json").write_text(json.dumps(results, indent=1))
            print(json.dumps(summary), flush=True)
            case.copies = case.copies[:1]
            torch.cuda.empty_cache()
        print("RESULTS", out)


def m1_ops_for(case, w, m1cfg):
    from inference.fused_kernels import _int8_dot
    from inference.kernels import _finish
    m, n, k = case.m, case.n, case.k
    bm, bn, bk, split, warps, stages = m1cfg
    y = torch.empty(m, n, device="cuda")
    partial = torch.empty(split, m, n, device="cuda") if split > 1 else y

    def op():
        _int8_dot[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
            case.q, case.qs, w, case.layer.scale, None, y, partial, m, n, k, False, bm, bn, bk, split, 3,
            DENSE=True, COLUMN=True, num_warps=warps, num_stages=stages)
        if split > 1:
            _finish[(triton.cdiv(m * n, 256),)](partial, case.layer.scale, None, y, m, n, m, m * n, n, 1, False,
                                                split, 256)
    return op


if __name__ == "__main__":
    main()
