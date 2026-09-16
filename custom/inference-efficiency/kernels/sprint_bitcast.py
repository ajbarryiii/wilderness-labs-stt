"""Isolated exact-code FP16 construction for the one-hour kernel sprint.

The GEMM algorithm, tile shape, activation precision, accumulation and scaling
match packed._gemm. Only construction of the exact FP16 {-1,0,+1} tile changes.
Importing this module changes no production dispatcher. GPU checks are invoked
explicitly by the parent experiment, which serializes GPU access.
"""
from pathlib import Path
import json
import math

import torch
import triton
import triton.language as tl

from . import packed


def variants():
    return ["gemm_bitcast", "gemm_bitmask", "gemm_int16"]


_STYLES = {"gemm_bitcast": 0, "gemm_bitmask": 1, "gemm_int16": 2}


@triton.jit
def _gemm_bitcast(X, WT, S, Bias, Y, M: tl.constexpr, N: tl.constexpr,
                 K: tl.constexpr, KW: tl.constexpr, BITS: tl.constexpr,
                 HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr, GROUP_M: tl.constexpr, STYLE: tl.constexpr = 0):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = pid // (GROUP_M * num_n)
    first_m = group * GROUP_M
    size_m = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % (GROUP_M * num_n)) % size_m
    pid_n = (pid % (GROUP_M * num_n)) // size_m
    m = pid_m * BM + tl.arange(0, BM)
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    word_k = tl.arange(0, BK // (32 // BITS))
    lane = tl.arange(0, 32 // BITS)
    scale = tl.load(S + n, n < N, other=0).to(tl.float16).to(tl.float32)
    accum = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(X + m[:, None] * K + kk[None, :],
                    (m[:, None] < M) & (kk[None, :] < K), other=0)
        wkk = start * (BK // (32 // BITS)) + word_k
        words = tl.load(WT + wkk[:, None] * N + n[None, :],
                        (wkk[:, None] < KW) & (n[None, :] < N), other=0)
        encoded = ((words[:, None, :].to(tl.uint32) >>
                    (lane[None, :, None] * BITS)) & ((1 << BITS) - 1)).reshape((BK, BN))
        if STYLE == 2:
            coefficients = (encoded.to(tl.int32) * 2 - 1 if BITS == 1
                            else encoded.to(tl.int32) - 1)
            b = coefficients.to(tl.int16).to(tl.float16)
        else:
            if BITS == 1:
                half_bits = ((1 - encoded) << 15) | 0x3c00
            elif STYLE == 0:
                half_bits = tl.where(encoded == 1, 0,
                                     ((encoded == 0).to(tl.uint32) << 15) | 0x3c00)
            else:
                sign = (~encoded << 14) & 0x8000
                nonzero = (encoded & 1) - 1
                half_bits = (0x3c00 | sign) & nonzero
            b = half_bits.to(tl.uint16).to(tl.float16, bitcast=True)
        accum = tl.dot(a, b, accum)
    accum = accum * scale[None, :]
    if HAS_BIAS:
        accum += tl.load(Bias + n, n < N, other=0)[None, :].to(tl.float32)
    tl.store(Y + m[:, None] * N + n[None, :], accum.to(tl.float16),
             (m[:, None] < M) & (n[None, :] < N))


class _ConfiguredGemm:
    def __init__(self, style):
        self.style = style

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return _gemm_bitcast[grid](*args, STYLE=self.style, **kwargs)
        return launch


def install(name, artifact_dir=None):
    if name not in variants():
        raise ValueError(f"Unknown bitcast candidate {name!r}")
    original = packed._gemm
    packed._gemm = _ConfiguredGemm(_STYLES[name])

    def restore():
        packed._gemm = original

    return restore


def compile_only(artifact_dir):
    """Compile representative SM120 GEMMs with an explicit target; no GPU use."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    folder = Path(artifact_dir)
    folder.mkdir(parents=True, exist_ok=True)
    results = {}
    signature = {"X": "*fp16", "WT": "*i32", "S": "*fp32",
                 "Bias": "*fp16", "Y": "*fp16"}
    for bits in (1, 2):
        constants = dict(M=1500, N=1024, K=1024, KW=1024 // (32 // bits),
                         BITS=bits, HAS_BIAS=True, BM=64, BN=64, BK=64, GROUP_M=8)
        for label, function in (("baseline", packed._gemm), *((v, _gemm_bitcast) for v in variants())):
            config = constants if label == "baseline" else {**constants, "STYLE": _STYLES[label]}
            attrs = {(i,): [["tt.divisibility", 16]] for i in range(5)}
            kernel = triton.compile(ASTSource(function, signature, constexprs=config, attrs=attrs),
                                    target=GPUTarget("cuda", 120, 32),
                                    options={"num_warps": 4, "num_stages": 3})
            name = f"{label}_b{bits}"
            for extension in ("ptx", "cubin", "ttgir"):
                value = kernel.asm[extension]
                path = folder / f"{name}.{extension}"
                path.write_bytes(value) if isinstance(value, bytes) else path.write_text(value)
            ptx = kernel.asm["ptx"]
            results[name] = {
                "shared_bytes": kernel.metadata.shared,
                "integer_to_fp32": ptx.count("cvt.rn.f32.s32"),
                "fp32_to_fp16": ptx.count("cvt.rn.f16.f32"),
                "mma_instructions": ptx.count("mma.sync"),
            }
    (folder / "bitcast-compilation.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def check(name="gemm_bitcast", artifact_dir=None):
    """Compare original GEMM, independent dense arithmetic, tails and replay."""
    if name not in variants():
        raise ValueError(f"Unknown bitcast candidate {name!r}")
    original = packed._gemm
    if original is _gemm_bitcast or isinstance(original, _ConfiguredGemm):
        raise RuntimeError("Run bitcast correctness before installing the candidate")
    torch.manual_seed(542087)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    count, worst = 0, 0.0
    shapes = ((1500, 1024, 1024), (1500, 3072, 1024),
              (1500, 4096, 1024), (1500, 1024, 4096),
              (1, 51864, 1024), (5, 19, 33),
              (3000, 1024, 240), (1500, 1024, 3072))
    for bits in (1, 2):
        for m, n, k in shapes:
            codes = (torch.randint(0, 2, (n, k), device="cuda", dtype=torch.int8) * 2 - 1
                     if bits == 1 else torch.randint(-1, 2, (n, k), device="cuda", dtype=torch.int8))
            scales = torch.linspace(-1.3, 1.3, n, device="cuda") / math.sqrt(k)
            p = packed.PackedWeight.from_codes(codes, bits, scales)
            x = torch.randn(m, k, device="cuda", dtype=torch.float16) * .2
            bias = torch.linspace(-.125, .125, n, device="cuda", dtype=torch.float16)
            reference = torch.empty((m, n), device="cuda", dtype=torch.float16)
            guarded = torch.full((m * n + 16,), 321., device="cuda", dtype=torch.float16)
            out = guarded[8:-8].view(m, n)
            bm, bn, bk = 16 if m <= 32 else 64, 64, 64
            grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
            args = (x, p.words_t, p.scales, bias)
            constants = (m, n, k, p.words.shape[1], bits, True, bm, bn, bk, 8)
            original[grid](*args, reference, *constants, num_warps=4, num_stages=3)
            _gemm_bitcast[grid](*args, out, *constants, STYLE=_STYLES[name], num_warps=4, num_stages=3)
            torch.testing.assert_close(out, reference, atol=0, rtol=0)
            if m == 5:
                expected = x.cpu().float() @ p.dense().cpu().float().T + bias.cpu().float()
                expected = expected.half()
                torch.testing.assert_close(out.cpu(), expected, atol=.003, rtol=.003)
                worst = max(worst, float((out.cpu().float() - expected.float()).abs().max()))
            if not bool((guarded[:8] == 321).all() & (guarded[-8:] == 321).all()):
                raise AssertionError("GEMM output canary overwritten")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                _gemm_bitcast[grid](*args, out, *constants, STYLE=_STYLES[name], num_warps=4, num_stages=3)
            stream.synchronize()
            original_x = x.clone()
            x.zero_()
            graph.replay()
            torch.testing.assert_close(out, bias.expand_as(out), atol=0, rtol=0)
            x.copy_(original_x)
            graph.replay()
            torch.testing.assert_close(out, reference, atol=0, rtol=0)
            count += 1
            del graph, p, x, bias, reference, guarded, out, original_x, codes, scales
    torch.cuda.synchronize()
    return {"candidate": name, "cases": count, "exact_baseline_match": True,
            "small_shape_dense_max_error": worst}
