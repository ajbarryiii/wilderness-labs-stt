"""compute-sanitizer for the exact deferred path (kernels_exact XCHG 5 + row_exact._row_epilogue_exact).

The child (one gpu_lock) runs, for int8_mma and int8_mma:b01 layers, the tuned XCHG 5 configs and
test_exact.EXTRA (WK groups, slice counts 2-8, 2-warp-row M tiles) on N = 1024 / K = 1024, 4096 and the
tail shape N = 1001, K = 1088, M = 1, 17, 130, 300 (8 and 4 epilogue warps), every row-epilogue flag the
pipeline uses (factor 0.5 / 1, LayerNorm + quantization, out-norm), plus a CUDA-graph capture and
replay. Same tool command line as sanitize_b.py. Run memcheck twice: with PYTORCH_NO_CUDA_MEMORY_CACHING=1
(every tensor its own cudaMalloc, so an out-of-bounds access is reported; --control demonstrates it)
and without (adds the graph capture, which needs the caching allocator). Logs: /mnt/hd/.../gpu1bit/B/sanitizer/.

    gpu1bit/shared/python -m gpu1bit.int8_mma.sanitize_exact --tool memcheck|racecheck
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from gpu1bit.shared import env, sanitize


def control() -> None:
    """Positive control: the XCHG 5 kernel writing into a slice buffer one row short (must be reported)."""
    import torch
    from gpu1bit.int8_mma import cuda, gpulock, kernels_exact
    from gpu1bit.int8_mma.layer import Int8MmaLayer, m1_quantize
    from gpu1bit.int8_mma.test_exact import EXTRA, weights
    with gpulock.locked("B:sanitizer exact deferred control"), torch.inference_mode():
        n, k, m = 1024, 1024, 17
        codes, scale, _ = weights(n, k, 3)
        layer = Int8MmaLayer.prepare(codes, scale, None, k, "cuda")
        q, qs = m1_quantize(torch.randn(1, m, k, device="cuda"))
        cfg = kernels_exact.ExactDeferredConfig.of(EXTRA[0])
        short = torch.empty(cfg.cluster * 2 * m * n - n, device="cuda", dtype=torch.int32)
        source, name = kernels_exact.dot_source(cfg)
        cuda.load(source, name, kernels_exact.shared_bytes(cfg)).launch(
            cfg.grid(m, n), (cfg.threads, 1, 1),
            [cuda.ptr(q), cuda.i64(q.stride(0)), cuda.i32(k), cuda.i32(k), cuda.ptr(qs), cuda.i32(qs.stride(0)),
             cuda.ptr(layer.words), cuda.i32(layer.words.shape[0]), cuda.i32(layer.words.shape[1]),
             cuda.ptr(layer.scale), cuda.ptr(None), cuda.ptr(short), cuda.i32(n), cuda.i32(m), cuda.i32(n), cuda.ptr(None)])
        torch.cuda.synchronize()
        print("SANITIZE-CHILD-DONE control", flush=True)


def child() -> None:
    import types
    import torch
    from gpu1bit.int8_mma import gpulock, hybrid_exact, kernels_exact, row_exact
    from gpu1bit.int8_mma.layer import Int8MmaLayer, m1_quantize
    from gpu1bit.int8_mma.layer_b01 import Int8MmaB01Layer
    from gpu1bit.int8_mma.test_exact import EXTRA, norm_module, tuned_configs, weights
    with gpulock.locked("B:sanitizer exact deferred"), torch.inference_mode():
        launches = 0
        for (n, k) in ((1024, 1024), (1024, 4096), (1001, 1088)):
            codes, scale, _ = weights(n, k, n + k)
            for cls in (Int8MmaLayer, Int8MmaB01Layer):
                layer = cls.prepare(codes, scale, None, k, "cuda")
                for m in (1, 17, 130, 300):
                    x = torch.randn(1, m, k, device="cuda")
                    residual = torch.randn(1, m, n, device="cuda")
                    norm, out_norm = norm_module(n, 1), norm_module(n, 2)
                    q, qs = m1_quantize(x)
                    for i, cfg in enumerate(tuned_configs() + EXTRA):
                        cfg = kernels_exact.ExactDeferredConfig.of(cfg)
                        if cfg.cluster > -(-k // 128):
                            continue
                        slices = torch.empty(cfg.cluster, 2, m, n, device="cuda", dtype=torch.int32)
                        hybrid_exact.dot_slices(layer, q, qs, k, m, slices, cfg)
                        p = types.SimpleNamespace(row_warps="adaptive", prequantized={},
                                                  null=torch.empty(0, device="cuda", dtype=torch.uint8))
                        flags = [(0.5, norm, None), (1.0, norm, None), (0.5, norm, out_norm), (0.5, None, out_norm)][i % 4]
                        row_exact.finish_residual_exact(p, row_exact.ExactDeferredProjection(
                            slices, layer.scale, None, cfg.cluster, (1, m, n), qs, 0), residual, flags[0], flags[1], flags[2])
                        launches += 2
        import os
        if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1":  # exact bounds; graph capture needs the pool
            print(f"SANITIZE-CHILD-DONE launches={launches} (no caching allocator: every tensor its own cudaMalloc)", flush=True)
            return
        # CUDA graph capture + replay of the provider path
        n, k, m = 1024, 4096, 126
        codes, scale, _ = weights(n, k, 7)
        layer = Int8MmaLayer.prepare(codes, scale, None, k, "cuda")
        x = torch.randn(1, m, k, device="cuda")
        y = hybrid_exact.deferred_project(layer, x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            yg = hybrid_exact.deferred_project(layer, x)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(y, yg):
            raise AssertionError("graph replay differs")
        print(f"SANITIZE-CHILD-DONE launches={launches + 4}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", choices=["memcheck", "racecheck", "synccheck", "initcheck"], default="memcheck")
    p.add_argument("--child", action="store_true")
    p.add_argument("--control", action="store_true", help="positive control (an out-of-bounds slice write)")
    args = p.parse_args()
    if args.child:
        return control() if args.control else child()
    out = env.ARTIFACTS_ROOT / "B" / "sanitizer"
    env.require_mount()
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{time.strftime('%Y%m%dT%H%M%S')}-{args.tool}-exact-deferred{'-control' if args.control else ''}.log"
    cmd = ["timeout", "600", str(sanitize.SANITIZER), "--injection-path", str(sanitize.SANITIZER.parent),
           "--launch-timeout", "0", "--tool", args.tool, "--report-api-errors", "no", "--target-processes", "all",
           sys.executable, "-m", "gpu1bit.int8_mma.sanitize_exact", "--child", "--tool", args.tool] + (["--control"] if args.control else [])
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    text = log.read_text()
    summary = [line for line in text.splitlines()
               if "ERROR SUMMARY" in line or "RACECHECK SUMMARY" in line or "SANITIZE-CHILD-DONE" in line]
    print("\n".join(summary) or text[-2000:])
    print(f"exit {rc}; log {log}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
