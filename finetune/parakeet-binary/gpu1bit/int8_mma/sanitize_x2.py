"""compute-sanitizer for B-hybrid-x2's exact deferred path (kernels_exact_x2 XCHG 5 + row_exact_x2).

Child (one gpu_lock): for int8_mma:x2 layers on N x K in 1024 x 1024, 1024 x 4096, 1001 x 1088 and
M = 1, 17, 130, 300 (8 and 4 epilogue warps), every tuned XCHG 5 config and test_exact.EXTRA, the
2-part slices and their exact finish with the row-epilogue flags the pipeline uses (factor 0.5 / 1,
LayerNorm + quantization, out-norm, bias); the in-kernel x2 route of the provider; with the caching
allocator (default run) also a CUDA-graph capture and replay of the provider's deferred call.
Modes, as sanitize_exact.py:
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 (`--exact-bounds`): every tensor its own cudaMalloc, so any
  out-of-bounds access is reported; `--control` demonstrates it with a slice buffer one row short;
  default: the pooled allocator, graph capture included.

    gpu1bit/shared/python -m gpu1bit.int8_mma.sanitize_x2 --tool memcheck|racecheck [--exact-bounds] [--control]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from gpu1bit.shared import env, sanitize


def control() -> None:
    import torch
    from gpu1bit.int8_mma import cuda, gpulock, hybrid_x2, kernels_exact, kernels_exact_x2
    from gpu1bit.int8_mma.test_exact import EXTRA, weights
    with gpulock.locked("B:sanitizer x2 control"), torch.inference_mode():
        n, k, m = 1024, 1024, 17
        codes, scale, _ = weights(n, k, 3)
        layer = hybrid_x2.x2_layer_class().prepare(codes, scale, None, k, "cuda")
        q, qs = hybrid_x2.quantize_x2(torch.randn(1, m, k, device="cuda"))
        cfg = kernels_exact.ExactDeferredConfig.of(EXTRA[0])
        short = torch.empty(cfg.cluster * 2 * m * n - n, device="cuda", dtype=torch.int32)
        source, name = kernels_exact_x2.dot_source(cfg, 0, 2)
        cuda.load(source, name, kernels_exact_x2.shared_bytes(cfg, 2)).launch(
            cfg.grid(m, n), (cfg.threads, 1, 1),
            [cuda.ptr(q), cuda.i64(q.stride(0)), cuda.i32(k), cuda.i32(k), cuda.ptr(qs), cuda.i32(qs.stride(0)),
             cuda.ptr(layer.words), cuda.i32(layer.words.shape[0]), cuda.i32(layer.words.shape[1]),
             cuda.ptr(layer.scale), cuda.ptr(None), cuda.ptr(short), cuda.i32(n), cuda.i32(m), cuda.i32(n), cuda.ptr(None)])
        torch.cuda.synchronize()
        print("SANITIZE-CHILD-DONE control", flush=True)


def child() -> None:
    import types
    import torch
    from gpu1bit.int8_mma import gpulock, hybrid_exact, hybrid_x2, kernels_exact, row_exact_x2
    from gpu1bit.int8_mma.test_exact import EXTRA, norm_module, tuned_configs, weights
    from gpu1bit.int8_mma.test_exact_x2 import call, one_layer_provider
    with gpulock.locked("B:sanitizer x2 exact deferred"), torch.inference_mode():
        row_exact_x2.install()
        launches = 0
        for (n, k) in ((1024, 1024), (1024, 4096), (1001, 1088)):
            for bias in (False, True):
                codes, scale, b = weights(n, k, n + k + bias, bias=bias)
                layer = hybrid_x2.x2_layer_class().prepare(codes, scale, b, k, "cuda")
                for m in (1, 17, 130, 300):
                    x = torch.randn(1, m, k, device="cuda")
                    residual = torch.randn(1, m, n, device="cuda")
                    norm, out_norm = norm_module(n, 1), norm_module(n, 2)
                    q, qs = hybrid_x2.quantize_x2(x)
                    for i, cfg in enumerate(tuned_configs() + EXTRA):
                        cfg = kernels_exact.ExactDeferredConfig.of(cfg)
                        if cfg.cluster > -(-k // 128):
                            continue
                        slices = torch.empty(cfg.cluster, 2, m, n, device="cuda", dtype=torch.int32)
                        hybrid_x2.dot_slices_x2(layer, q, qs, k, m, slices, cfg, i % 2)
                        p = types.SimpleNamespace(row_warps="adaptive", prequantized={},
                                                  null=torch.empty(0, device="cuda", dtype=torch.uint8))
                        factor, nrm, onrm = [(0.5, norm, None), (1.0, norm, None), (0.5, norm, out_norm), (0.5, None, out_norm)][i % 4]
                        row_exact_x2.finish_residual_exact_x2(p, row_exact_x2.ExactDeferredProjectionX2(
                            slices, layer.scale, layer.bias, cfg.cluster, (1, m, n), qs, i % 2, 2), residual, factor, nrm, onrm)
                        launches += 2
                if n == 1024:
                    prov, handle = one_layer_provider(layer)
                    for m in (17, 38, 126, 300):
                        x = torch.randn(1, m, k, device="cuda")
                        for split in (True, False):
                            prov.split_deferred = split
                            call(prov, handle, x, defer=True)
                            call(prov, handle, x, defer=False, activation="silu")
                            launches += 4
                    prov.split_deferred = True
        if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1":  # exact bounds; graph capture needs the pool
            torch.cuda.synchronize()
            print(f"SANITIZE-CHILD-DONE launches={launches} (no caching allocator)", flush=True)
            return
        n, k, m = 1024, 4096, 126
        codes, scale, _ = weights(n, k, 7)
        layer = hybrid_x2.x2_layer_class().prepare(codes, scale, None, k, "cuda")
        x = torch.randn(1, m, k, device="cuda")
        y = hybrid_x2.deferred_project_x2(layer, x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            yg = hybrid_x2.deferred_project_x2(layer, x)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(y, yg):
            raise AssertionError("graph replay differs")
        if hybrid_exact.choose_exact_deferred(m, n, k) is None:
            raise AssertionError("the captured call did not take the XCHG 5 route")
        print(f"SANITIZE-CHILD-DONE launches={launches + 4} (pooled allocator, graph capture)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", choices=["memcheck", "racecheck", "synccheck", "initcheck"], default="memcheck")
    p.add_argument("--exact-bounds", action="store_true", help="PYTORCH_NO_CUDA_MEMORY_CACHING=1 in the child")
    p.add_argument("--control", action="store_true", help="positive control (an out-of-bounds slice write)")
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    if args.child:
        return control() if args.control else child()
    out = env.ARTIFACTS_ROOT / "B" / "sanitizer"
    env.require_mount()
    out.mkdir(parents=True, exist_ok=True)
    tag = ("-exactbounds" if args.exact_bounds else "-pooled") + ("-control" if args.control else "")
    log = out / f"{time.strftime('%Y%m%dT%H%M%S')}-{args.tool}-x2{tag}.log"
    cmd = ["timeout", "600", str(sanitize.SANITIZER), "--injection-path", str(sanitize.SANITIZER.parent),
           "--launch-timeout", "0", "--tool", args.tool, "--report-api-errors", "no", "--target-processes", "all",
           sys.executable, "-m", "gpu1bit.int8_mma.sanitize_x2", "--child", "--tool", args.tool] + \
        (["--control"] if args.control else [])
    child_env = {**os.environ, **({"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"} if args.exact_bounds else {})}
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=child_env).returncode
    text = log.read_text()
    summary = [line for line in text.splitlines()
               if "ERROR SUMMARY" in line or "RACECHECK SUMMARY" in line or "SANITIZE-CHILD-DONE" in line]
    print("\n".join(summary) or text[-2000:])
    print(f"exit {rc}; log {log}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
