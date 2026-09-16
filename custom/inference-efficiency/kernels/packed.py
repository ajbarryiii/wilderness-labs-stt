"""Direct packed W1/W2, FP16-activation inference for NVIDIA GPUs.

Decode runs as a vectorized SIMT reduction for small/medium output widths and
as a skinny tiled Tensor Core matmul for large ones (vocabulary projection);
encoder/prefill uses tiled Tensor Core matmul over a pre-transposed packed
layout. All paths unpack only the active tile and accumulate in FP32. Packing
and the explicit ``dense`` reference method are preparation operations, never
inference fallbacks. These kernels are original implementations; see
RESEARCH.md for related projects. They require PyTorch and Triton; when the
NVRTC wheel is available, decode uses the warp-per-row CUDA GEMV in
``cuda_gemv.py`` (compiled at load time, no nvcc toolchain needed).
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from . import cuda_gemv


@triton.jit
def _pack(Codes, Words, N: tl.constexpr, K: tl.constexpr,
          KW: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = idx // KW
    col = (idx % KW) * (32 // BITS)
    word = tl.full((BLOCK,), 0, tl.uint32)
    for bit in tl.static_range(32 // BITS):
        code = tl.load(Codes + row * K + col + bit,
                       (row < N) & (col + bit < K), other=0).to(tl.int32)
        if BITS == 1:
            encoded = (code > 0).to(tl.uint32)
        else:
            encoded = (code + 1).to(tl.uint32)
        word = word | (encoded << (bit * BITS))
    tl.store(Words + idx, word.to(tl.int32), idx < N * KW)


@triton.jit
def _gemv(X, W, S, Bias, Y, N: tl.constexpr, K: tl.constexpr,
          KW: tl.constexpr, BITS: tl.constexpr, HAS_BIAS: tl.constexpr,
          BN: tl.constexpr, BKC: tl.constexpr):
    # One packed-word load broadcasts over all of its constituent values.
    # K is consumed in chunks of BKC words so register pressure stays flat as K
    # grows; the whole-row variant collapsed to 25% occupancy at K=4096.
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    m = tl.program_id(1)
    kw = tl.arange(0, BKC)
    lane = tl.arange(0, 32 // BITS)
    accum = tl.zeros((BN,), tl.float32)
    for start in range(tl.cdiv(KW, BKC)):
        wkk = start * BKC + kw
        words = tl.load(W + n[:, None] * KW + wkk[None, :],
                        (n[:, None] < N) & (wkk[None, :] < KW), other=0)
        encoded = (words[:, :, None].to(tl.uint32) >>
                   (lane[None, None, :] * BITS)) & ((1 << BITS) - 1)
        if BITS == 1:
            codes = encoded.to(tl.float32) * 2.0 - 1.0
        else:
            codes = encoded.to(tl.float32) - 1.0
        k = wkk[:, None] * (32 // BITS) + lane[None, :]
        x = tl.load(X + m * K + k, k < K, other=0).to(tl.float32)
        accum += tl.sum(tl.sum(codes * x[None, :, :], axis=2), axis=1)
    # Scales are rounded to FP16 just like the dense FP16 reference weights.
    scale = tl.load(S + n, n < N, other=0).to(tl.float16).to(tl.float32)
    value = accum * scale
    if HAS_BIAS:
        value += tl.load(Bias + n, n < N, other=0).to(tl.float32)
    tl.store(Y + m * N + n, value.to(tl.float16), n < N)


@triton.jit
def _gemm(X, WT, S, Bias, Y, M: tl.constexpr, N: tl.constexpr,
          K: tl.constexpr, KW: tl.constexpr, BITS: tl.constexpr,
          HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr, GROUP_M: tl.constexpr):
    # Group neighboring M tiles to retain packed B tiles in L2.
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
    # Per-row scales are applied once to the FP32 accumulator, not per K tile.
    scale = tl.load(S + n, n < N, other=0).to(tl.float16).to(tl.float32)
    accum = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(X + m[:, None] * K + kk[None, :],
                    (m[:, None] < M) & (kk[None, :] < K), other=0)
        # WT is [KW, N], so the B tile decodes directly into [BK, BN] MMA
        # orientation; no register transpose through shared memory.
        wkk = start * (BK // (32 // BITS)) + word_k
        words = tl.load(WT + wkk[:, None] * N + n[None, :],
                        (wkk[:, None] < KW) & (n[None, :] < N), other=0)
        encoded = ((words[:, None, :].to(tl.uint32) >>
                    (lane[None, :, None] * BITS)) & ((1 << BITS) - 1)).reshape((BK, BN))
        if BITS == 1:
            b = (encoded.to(tl.float32) * 2.0 - 1.0).to(tl.float16)
        else:
            b = (encoded.to(tl.float32) - 1.0).to(tl.float16)
        accum = tl.dot(a, b, accum)
    accum = accum * scale[None, :]
    if HAS_BIAS:
        accum += tl.load(Bias + n, n < N, other=0)[None, :].to(tl.float32)
    tl.store(Y + m[:, None] * N + n[None, :], accum.to(tl.float16),
             (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _embedding(IDs, W, S, Y, COUNT: tl.constexpr, N: tl.constexpr,
               K: tl.constexpr, KW: tl.constexpr, BITS: tl.constexpr,
               BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = off < COUNT * K
    token = tl.load(IDs + off // K, valid, other=0)
    in_bounds = valid & (token >= 0) & (token < N)
    k = off % K
    word = tl.load(W + token * KW + k // (32 // BITS), in_bounds, other=0)
    code = (word.to(tl.uint32) >> ((k % (32 // BITS)) * BITS)) & ((1 << BITS) - 1)
    if BITS == 1:
        value = code.to(tl.float32) * 2.0 - 1.0
    else:
        value = code.to(tl.float32) - 1.0
    scale = tl.load(S + token, in_bounds, other=0).to(tl.float16).to(tl.float32)
    value = tl.where(in_bounds, value * scale, float("nan"))
    tl.store(Y + off, value.to(tl.float16), valid)


@dataclass(frozen=True)
class PackedWeight:
    """Row-major logical [N,K] weights packed into int32 words, both orientations.

    ``words`` is [N,ceil(K/VPW)] for row-parallel access (GEMV, embedding);
    ``words_t`` is its [ceil(K/VPW),N] transpose so the GEMM decodes tiles
    directly into [BK,BN] MMA orientation. Ternary uses two bits (-1=0, 0=1,
    +1=2); binary uses one (-1=0, +1=1). ``scales`` are FP32 per output row and
    rounded to FP16 on access. ``linear`` uses no dense fallback or transient
    full-weight unpacking. Input contiguous copies and optional output
    allocation are included in the caller's timing.
    """

    words: torch.Tensor
    scales: torch.Tensor
    n: int
    k: int
    bits: int
    words_t: torch.Tensor

    def __post_init__(self):
        if self.bits not in (1, 2) or min(self.n, self.k) < 1:
            raise ValueError("bits must be 1 or 2 and dimensions positive")
        kw = triton.cdiv(self.k, 32 // self.bits)
        if (self.words.shape != (self.n, kw)
                or self.words.dtype != torch.int32 or not self.words.is_contiguous()):
            raise ValueError("words must be contiguous int32 [N,ceil(K/values_per_word)]")
        if (self.words_t.shape != (kw, self.n)
                or self.words_t.dtype != torch.int32 or not self.words_t.is_contiguous()
                or self.words_t.device != self.words.device):
            raise ValueError("words_t must be contiguous int32 [ceil(K/values_per_word),N]")
        if (self.scales.shape != (self.n,) or self.scales.dtype != torch.float32
                or not self.scales.is_contiguous() or self.scales.device != self.words.device):
            raise ValueError("scales must be contiguous FP32 [N] on the same device")

    @classmethod
    def from_codes(cls, codes: torch.Tensor, bits: int = 2,
                   scale: float | torch.Tensor = 1.0) -> "PackedWeight":
        if codes.ndim != 2 or min(codes.shape) < 1 or codes.dtype != torch.int8:
            raise ValueError("codes must be a nonempty int8 matrix [N,K]")
        if bits not in (1, 2):
            raise ValueError("bits must be 1 or 2")
        valid = (codes == -1) | (codes == 1)
        if bits == 2:
            valid |= codes == 0
        if not bool(valid.all()):
            raise ValueError("codes contain values outside the selected alphabet")
        n, k = codes.shape
        scales = torch.as_tensor(scale, dtype=torch.float32, device=codes.device)
        if scales.ndim == 0:
            scales = scales.expand(n)
        scales = scales.contiguous()
        if scales.shape != (n,) or not bool(torch.isfinite(scales).all()):
            raise ValueError("scale must be finite scalar or [N]")
        vpw = 32 // bits
        kw = triton.cdiv(k, vpw)
        words = torch.zeros((n, kw), dtype=torch.int32, device=codes.device)
        if codes.is_cuda:
            _pack[(triton.cdiv(n * kw, 128),)](
                codes.contiguous(), words, n, k, kw, bits, 128)
        else:
            for lane in range(vpw):
                part = codes[:, lane::vpw].to(torch.int32)
                encoded = (part > 0).to(torch.int32) if bits == 1 else part + 1
                words[:, :part.shape[1]] |= encoded << (lane * bits)
        return cls(words, scales, n, k, bits, words.t().contiguous())

    @property
    def storage_bytes(self) -> int:
        return self.words.numel() * 4 + self.words_t.numel() * 4 + self.scales.numel() * 4

    def to(self, device: str | torch.device) -> "PackedWeight":
        return PackedWeight(self.words.to(device), self.scales.to(device),
                            self.n, self.k, self.bits, self.words_t.to(device))

    def dense(self) -> torch.Tensor:
        """Expand for correctness/reference preparation only, never inference."""
        k = torch.arange(self.k, device=self.words.device)
        encoded = (self.words[:, k // (32 // self.bits)] >>
                   ((k % (32 // self.bits)) * self.bits)) & ((1 << self.bits) - 1)
        codes = encoded * 2 - 1 if self.bits == 1 else encoded - 1
        return (codes.to(torch.float16) * self.scales.to(torch.float16)[:, None]).contiguous()

    def linear(self, x: torch.Tensor, bias: torch.Tensor | None = None,
               out: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.k:
            raise ValueError(f"expected input [...,{self.k}]")
        if x.dtype != torch.float16 or not x.is_cuda or x.device != self.words.device:
            raise ValueError("linear requires FP16 CUDA input on the weight device")
        if bias is not None and (bias.shape != (self.n,) or not bias.is_contiguous()
                                 or bias.device != x.device or bias.dtype != x.dtype):
            raise ValueError("bias must be contiguous FP16 [N] on the input device")
        shape = (*x.shape[:-1], self.n)
        if out is None:
            out = torch.empty(shape, dtype=x.dtype, device=x.device)
        elif (out.shape != shape or out.dtype != x.dtype or out.device != x.device
              or not out.is_contiguous()):
            raise ValueError("out has incorrect shape, dtype, device, or layout")
        m = math.prod(x.shape[:-1])
        if not m:
            return out
        x = x.contiguous()
        bias_arg = bias if bias is not None else out
        kw = self.words.shape[1]
        if m <= 4:
            ext = cuda_gemv.try_load()
            # Measured on the 5090 (see kernels/README.md): the NVRTC row-
            # per-lane/warp-per-row GEMV wins everywhere except 2-bit large-N
            # decode, where the skinny Tensor Core GEMM stays ahead (16.7us
            # vs 18.9us at n=51864) because HMMA needs fewer issue slots.
            if ext is not None and not (self.bits == 2 and self.n >= 8192):
                cuda_gemv.gemv(ext, x, self.words, self.scales, bias, out,
                               self.n, self.k, kw, self.bits, m)
            elif self.n >= 8192:
                # Large-N decode (vocabulary projection) as a skinny Tensor
                # Core GEMM: measured 2.4x the SIMT GEMV on the 5090.
                _gemm[(triton.cdiv(self.n, 64),)](
                    x, self.words_t, self.scales, bias_arg, out, m, self.n,
                    self.k, kw, self.bits, bias is not None, 16, 64, 64, 8,
                    num_warps=4, num_stages=3)
            else:
                bn, bkc, warps = _gemv_config(self.n, self.k, self.bits)
                _gemv[(triton.cdiv(self.n, bn), m)](
                    x, self.words, self.scales, bias_arg, out,
                    self.n, self.k, kw, self.bits, bias is not None,
                    bn, bkc, num_warps=warps)
        else:
            # Fixed, bounded launch choices. No autotuning inside measurement.
            if m <= 32:
                bm, bn, bk, warps, stages = 16, 64, 64, 4, 3
            else:
                bm, bn, bk, warps, stages = 64, 64, 64, 4, 3
            _gemm[(triton.cdiv(m, bm) * triton.cdiv(self.n, bn),)](
                x, self.words_t, self.scales, bias_arg, out, m, self.n,
                self.k, kw, self.bits, bias is not None, bm, bn, bk, 8,
                num_warps=warps, num_stages=stages)
        return out

    def embedding(self, ids: torch.Tensor,
                  out: torch.Tensor | None = None) -> torch.Tensor:
        if (ids.dtype not in (torch.int32, torch.int64) or not ids.is_cuda
                or ids.device != self.words.device):
            raise ValueError("embedding requires integer CUDA ids on the weight device")
        shape = (*ids.shape, self.k)
        if out is None:
            out = torch.empty(shape, dtype=torch.float16, device=ids.device)
        elif (out.shape != shape or out.dtype != torch.float16 or out.device != ids.device
              or not out.is_contiguous()):
            raise ValueError("out has incorrect shape, dtype, device, or layout")
        if ids.numel():
            _embedding[(triton.cdiv(ids.numel() * self.k, 256),)](
                ids.contiguous(), self.words, self.scales, out, ids.numel(), self.n,
                self.k, self.words.shape[1], self.bits, 256)
        return out

    def conv1d(self, x: torch.Tensor, kernel_size: int,
               bias: torch.Tensor | None = None, stride: int = 1,
               padding: int = 0) -> torch.Tensor:
        """Packed convolution via im2col + linear, including im2col overhead.

        x is [batch, channels, time]; flattened weights follow PyTorch Conv1d's
        [out_channels, in_channels, kernel_size] order. This explicit conversion
        is a documented first-pass limitation, not a dense-weight fallback.
        """
        if (x.ndim != 3 or kernel_size < 1 or stride < 1 or padding < 0
                or x.shape[1] * kernel_size != self.k):
            raise ValueError("invalid conv1d shape or arguments")
        columns = F.unfold(x.unsqueeze(2), (1, kernel_size),
                           padding=(0, padding), stride=(1, stride))
        return self.linear(columns.transpose(1, 2), bias).transpose(1, 2).contiguous()


def _gemv_config(n: int, k: int, bits: int) -> tuple[int, int, int]:
    """Fixed (BN, BKC, num_warps) from the revision-2 diagnostic sweep.

    Small-N with large K keeps a single whole-K pass: maximizing loads in
    flight per warp beat K-chunking by ~30% at (1024, 4096). Chunked loads
    win elsewhere; large-N decode routes to the GEMM instead (see linear).
    """
    if k >= 2048:
        return (4, triton.next_power_of_2(triton.cdiv(k, 32 // bits)), 4)
    if n <= 2048:
        return (2, 16, 4)
    return (4, 16, 1)
