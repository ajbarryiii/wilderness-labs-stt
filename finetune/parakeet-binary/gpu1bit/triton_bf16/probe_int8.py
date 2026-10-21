"""Roofline probe part 2: Triton INT8 vs BF16 GEMM peak at the 400 W cap (agent A, screening)."""
import json, sys
import torch, triton, triton.language as tl
from gpu1bit.triton_bf16.gpulock import locked

@triton.jit
def _gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr, INT: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    m = pm * BM + tl.arange(0, BM); n = pn * BN + tl.arange(0, BN); k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32 if INT else tl.float32)
    for s in range(0, K, BK):
        a = tl.load(A + m[:, None] * K + s + k[None, :])
        b = tl.load(B + (s + k[:, None]) * N + n[None, :])
        acc = tl.dot(a, b, acc, out_dtype=tl.int32 if INT else tl.float32)
    tl.store(C + m[:, None] * N + n[None, :], acc.to(tl.float32))

def main():
    out = {}
    with locked("A:peak-probe-int8"), torch.inference_mode():
        for s in (4096, 8192):
            flop = 2 * s ** 3
            c = torch.empty(s, s, device="cuda")
            for name, dt in (("int8", torch.int8), ("bf16", torch.bfloat16)):
                a = (torch.randint(-127, 128, (s, s), device="cuda")).to(dt)
                b = (torch.randint(-127, 128, (s, s), device="cuda")).to(dt)
                for bm, bn, bk, w, st in [(128, 128, 64, 4, 3), (128, 128, 128, 8, 3), (128, 256, 64, 8, 3), (128,128,32,4,4)]:
                    key = f"{s}_{name}_{bm}x{bn}x{bk}_w{w}s{st}"
                    try:
                        fn = lambda: _gemm[(s // bm, s // bn)](a, b, c, s, s, s, bm, bn, bk, name == "int8", num_warps=w, num_stages=st)
                        out[key] = round(flop / triton.testing.do_bench(fn, warmup=50, rep=300) / 1e9, 1)
                    except Exception as exc:
                        out[key] = repr(exc)[:120]
        json.dump(out, sys.stdout, indent=1); print()

if __name__ == "__main__":
    main()
