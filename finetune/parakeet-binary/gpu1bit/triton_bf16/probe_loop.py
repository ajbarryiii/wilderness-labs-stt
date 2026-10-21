"""Per-K-iteration latency of small-tile Triton GEMM main loops (agent A, screening).

M=38 rows, N=1024, BM=16, BN=32, SPLIT=1, 4 warps: each CTA walks all of K. Kernel GPU time (CUPTI)
for K = 1024 and 4096 gives the per-iteration cost (slope) and the fixed cost (intercept) of:
  dense16  FP16 A x dense FP16 B (one dot): Triton's floor for this tile shape
  dense16x2 two FP16 dots sharing B (the FP16x2 MMA count, no decode)
  int8x3   three INT8 dots sharing an INT8 B (M1's arithmetic, prequantized)
  bits16x2 FP16 planes x 1-bit decoded B (our _pgemm path)
  f32split FP32 A split in-kernel to FP16x2 (tile scale) x 1-bit B (our _bgemm fp16x2)
"""
import collections
import json
import statistics
import sys

import torch
import triton
import triton.language as tl

from gpu1bit.triton_bf16.gpulock import locked
from gpu1bit.shared import env
from gpu1bit.triton_bf16 import kernels as K


@triton.jit
def _dense(A, B, Y, M, N: tl.constexpr, KD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           PARTS: tl.constexpr, INT8: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32 if INT8 else tl.float32)
    acc2 = tl.zeros((BM, BN), tl.int32 if INT8 else tl.float32)
    acc3 = tl.zeros((BM, BN), tl.int32 if INT8 else tl.float32)
    for t in range(KD // BK):
        k = t * BK + kk
        b = tl.load(B + k[:, None] * N + n[None, :])
        a = tl.load(A + m[:, None] * KD + k[None, :], m[:, None] < M, other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32 if INT8 else tl.float32)
        if PARTS > 1:
            a2 = tl.load(A + M * KD + m[:, None] * KD + k[None, :], m[:, None] < M, other=0)
            acc2 = tl.dot(a2, b, acc2, out_dtype=tl.int32 if INT8 else tl.float32)
        if PARTS > 2:
            a3 = tl.load(A + 2 * M * KD + m[:, None] * KD + k[None, :], m[:, None] < M, other=0)
            acc3 = tl.dot(a3, b, acc3, out_dtype=tl.int32 if INT8 else tl.float32)
    out = acc.to(tl.float32) + acc2.to(tl.float32) + acc3.to(tl.float32)
    tl.store(Y + m[:, None] * N + n[None, :], out, m[:, None] < M)


def kernel_us(fn, repeat=40):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
    per = collections.defaultdict(list)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            per[e.name[:40]].append(e.device_time)
    return {k: statistics.median(v) for k, v in per.items()}


def main():
    m, n, bm, bn, bk = 38, 1024, 16, 32, 64
    out = {}
    with locked("A:probe-loop"), torch.inference_mode():
        y = torch.empty(m, n, device="cuda")
        for kd in (1024, 4096):
            grid = (triton.cdiv(m, bm), n // bn)
            a16 = torch.randn(3, m, kd, device="cuda").half()
            b16 = torch.randn(kd, n, device="cuda").half()
            a8 = torch.randint(-127, 128, (3, m, kd), device="cuda", dtype=torch.int8)
            b8 = torch.randint(-1, 2, (kd, n), device="cuda", dtype=torch.int8)
            for stages in (2, 3, 4):
                for name, args in [("dense16", (a16, b16, 1, False)), ("dense16x2", (a16, b16, 2, False)),
                                   ("int8x3", (a8, b8, 3, True))]:
                    a, b, parts, i8 = args
                    bkk = 128 if i8 else bk
                    t = kernel_us(lambda: _dense[grid](a, b, y, m, n, kd, bm, bn, bkk, parts, i8, num_warps=4,
                                                       num_stages=stages))
                    out[f"{name}_K{kd}_s{stages}"] = round(sum(t.values()), 3)
            codes = torch.randint(0, 256, (n, kd // 8), dtype=torch.uint8)
            from gpu1bit.triton_bf16.layer import TritonBF16Layer
            x = torch.randn(1, m, kd, device="cuda")
            for mode in ("fp16x2p", "fp16x2", "bf16x3"):
                for stages in (2, 3, 4):
                    layer = TritonBF16Layer.prepare(codes, torch.rand(n), None, kd, "cuda", mode=mode,
                                                    configs={m: (bm, bn, bk, 1, 4, stages, "serial")})
                    t = kernel_us(lambda: layer.linear(x))
                    out[f"{mode}_K{kd}_s{stages}"] = {k: round(v, 3) for k, v in t.items()}
        for key, v in out.items():
            print(key, json.dumps(v))
        iters = (4096 - 1024) // bk
        print("per-iteration (us), slope over", iters, "extra BK=64 iterations:")
        for key in out:
            if "_K1024_" in key:
                other = out[key.replace("_K1024_", "_K4096_")]
                a, b = out[key], other
                sa = a if isinstance(a, float) else max(a.values())
                sb = b if isinstance(b, float) else max(b.values())
                div = iters if not key.startswith("int8") else (4096 - 1024) // 128
                print(f"  {key.replace('_K1024', ''):18s} fixed~{sa - (sb - sa) / 3:6.2f}  per-iter {(sb - sa) / div * 1000:7.1f} ns")

if __name__ == "__main__":
    main()
