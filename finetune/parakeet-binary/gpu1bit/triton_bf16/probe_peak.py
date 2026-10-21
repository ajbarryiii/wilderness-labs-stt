"""Roofline probe: dense tensor-core and memory peaks on this GPU (agent A, screening only).

Large square GEMMs through cuBLAS / CUTLASS (torch.mm, torch._int_mm, torch._scaled_mm) and a
Triton FP16 GEMM with FP32 vs FP16 accumulation, plus a device-to-device copy for DRAM bandwidth.
Run with finetune/parakeet-binary/python; holds the GPU lock for its whole life.
"""
import json
import sys

import torch
import triton
import triton.language as tl

from gpu1bit.triton_bf16.gpulock import locked


@triton.jit
def _gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr, ACC16: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    m = pm * BM + tl.arange(0, BM); n = pn * BN + tl.arange(0, BN); k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float16 if ACC16 else tl.float32)
    for s in range(0, K, BK):
        a = tl.load(A + m[:, None] * K + s + k[None, :])
        b = tl.load(B + (s + k[:, None]) * N + n[None, :])
        acc = tl.dot(a, b, acc, out_dtype=tl.float16 if ACC16 else tl.float32)
    tl.store(C + m[:, None] * N + n[None, :], acc.to(tl.float32))


def bench(fn):
    return triton.testing.do_bench(fn, warmup=50, rep=300)  # ms


def main():
    out = {}
    with locked("A:peak-probe"), torch.inference_mode():
        s = 8192
        a16 = torch.randn(s, s, device="cuda", dtype=torch.float16)
        b16 = torch.randn(s, s, device="cuda", dtype=torch.float16)
        flop = 2 * s ** 3
        out["fp16_mm_tflops"] = flop / bench(lambda: torch.mm(a16, b16)) / 1e9
        abf, bbf = a16.bfloat16(), b16.bfloat16()
        out["bf16_mm_tflops"] = flop / bench(lambda: torch.mm(abf, bbf)) / 1e9
        a8 = torch.randint(-127, 128, (s, s), device="cuda", dtype=torch.int8)
        b8 = torch.randint(-127, 128, (s, s), device="cuda", dtype=torch.int8).t().contiguous().t()
        out["int8_int_mm_tops"] = flop / bench(lambda: torch._int_mm(a8, b8)) / 1e9
        try:
            af8 = a16.to(torch.float8_e4m3fn); bf8 = b16.to(torch.float8_e4m3fn).t().contiguous().t()
            one = torch.ones((), device="cuda")
            out["fp8_scaled_mm_tflops"] = flop / bench(lambda: torch._scaled_mm(af8, bf8, scale_a=one, scale_b=one,
                                                                                  out_dtype=torch.float32)) / 1e9
        except Exception as exc:  # report, do not hide
            out["fp8_scaled_mm_error"] = repr(exc)
        c = torch.empty(s, s, device="cuda", dtype=torch.float32)
        for acc16 in (False, True):
            for bm, bn, bk, w, st in [(128, 128, 32, 4, 3), (128, 128, 64, 8, 3), (128, 256, 32, 8, 3)]:
                key = f"triton_fp16_acc{'16' if acc16 else '32'}_{bm}x{bn}x{bk}_w{w}s{st}"
                try:
                    fn = lambda: _gemm[(s // bm, s // bn)](a16, b16, c, s, s, s, bm, bn, bk, acc16,
                                                           num_warps=w, num_stages=st)
                    out[key] = flop / bench(fn) / 1e9
                except Exception as exc:
                    out[key] = repr(exc)[:200]
        src = torch.empty(1 << 28, device="cuda", dtype=torch.uint8); dst = torch.empty_like(src)
        out["d2d_copy_GBps"] = 2 * src.numel() / bench(lambda: dst.copy_(src)) / 1e6
        small = torch.empty(32 << 20, device="cuda", dtype=torch.uint8); small2 = torch.empty_like(small)
        out["l2_copy_32MB_GBps"] = 2 * small.numel() / bench(lambda: small2.copy_(small)) / 1e6
        p = torch.cuda.get_device_properties(0)
        out["sm_count"] = p.multi_processor_count
        out["l2_bytes"] = p.L2_cache_size
        out["smem_per_block_optin"] = p.shared_memory_per_block_optin
        json.dump(out, sys.stdout, indent=1)
        print()


if __name__ == "__main__":
    main()
