# Packed inference kernels

`PackedWeight` in `packed.py` stores one-bit binary or two-bit ternary weights
as packed `int32` words in both row-major `[N,Kword]` and transposed
`[Kword,N]` orientations, with FP32 row scales. Activations and outputs are
FP16; arithmetic accumulates in FP32. Scales round to FP16 on access, matching
the dense FP16 reference weights.

- **Decode (1–4 input rows):** when NVRTC is available, `cuda_gemv.py`
  supplies the decode path: warp-per-row CUDA GEMV (lane-owned packed words,
  shuffle broadcast, direct coalesced activation reads) for most shapes, a
  row-per-lane variant (32 rows/warp, shared-memory weight tile) for 1-bit
  N ≥ 8192, and a scalar ragged-K fallback. 2-bit N ≥ 8192 stays on the
  skinny tiled Tensor Core GEMM, which measured faster there. Without NVRTC,
  decode falls back to the Triton SIMT GEMV / skinny GEMM pair.
- **Encoder/prefill:** fixed tiled Tensor Core GEMM over the transposed
  layout, so B tiles decode directly into `[BK,BN]` MMA orientation with no
  register transpose; per-row scales apply once to the FP32 accumulator
  instead of per K tile. Grouped tile scheduling, pipelined loads, and
  unpacking only the active tile. Compiled SM120 PTX was checked for
  `mma.sync` FP16-input/FP32-accumulator instructions.
- **Embedding:** gather and unpack only requested rows, including row scales.
- **Convolution:** `unfold` plus packed linear. Conversion and output-layout
  copies are part of inference; this is a documented initial implementation
  limitation. Biases and other small vectors are handled by the model runtime.

There is no whole-weight expansion during inference, CPU fallback, activation
quantization, or run-time autotuning. `dense()` exists only for preparation and
numeric checks. The optional `out` argument reuses linear/embedding buffers;
noncontiguous linear inputs incur an explicit, measured contiguous copy.

From `inference-efficiency/`:

```sh
./python -m kernels.verify --full-shapes
./python -m kernels.verify_cuda
./python -m kernels.benchmark
```

Verification covers both alphabets, CPU/GPU packing, ragged dimensions,
Whisper-medium matrix sizes, 1500-frame encoder input, 51864-token projection,
embedding, convolution, bias, and output reuse. The 84 linear comparisons pass
with `atol=0.003, rtol=0.003` against the identical FP16 dense values.
`verify_cuda` explicitly compiles and executes every CUDA variant, including
the 2-bit row-per-lane variant that automatic dispatch bypasses. It checks
FP16 scale rounding, cancellation, ragged dimensions, batches, output bounds,
non-default streams, and graph replay with changed inputs. CUDA compilation
errors fail this suite; automatic inference fallback reports the error once.
Set `EFFICIENCY_REQUIRE_CUDA_GEMV=1` for CUDA comparison runs to reject
compilation failures instead of falling back. Model metadata records the decode
backend; large-N ternary decode intentionally retains the skinny Triton GEMM.

The benchmark reports short CUDA-graph timings against cuBLAS FP16. It removes
Python launch overhead and excludes packing/compilation; repeatedly accessing
one weight can keep it in L2. These are kernel diagnostics, not model energy
measurements or investment-gate evidence.

The bounded profiling revision changed GEMM from repeated packed addresses in
a `[K,N]` tile to unique `[N,Kword]` loads broadcast into registers, and enlarged
the M tile from 32 to 64. It removed a major redundant-data-movement bottleneck.
On this shared 5090, revised encoder diagnostics were near the FP16 baseline
for two representative shapes, and vocabulary GEMV was faster; the raw initial
and revised diagnostics are stored under the experiment artifact directory.
Timing variation and concurrent GPU users prevent treating these as final
performance results.

A second revision (2026-09-11, same shared GPU) reworked decode and the GEMM
data path. Against revision 1 in identical conditions (CUDA-graph medians,
interleaved round-robin): the vocabulary projection runs 2.4–2.7× faster as a
skinny GEMM (now 8.2–8.5× faster than cuBLAS FP16, vs 3.1–3.3× in revision 1);
the prefill GEMM gains 3–10% from the transposed layout and hoisted scale;
small-square GEMV gains ~5%. The `(1024,4096)` decode GEMV is unchanged —
K-chunking measured ~30% slower than the whole-K pass there, so that shape
keeps the revision-1 structure. GEMV configs are fixed per shape class in
`_gemv_config`; no runtime autotuning. Raw diagnostics are stored as
`kernel-diagnostics-revision2.json` in the experiment artifact directory.
A warp-per-row CUDA GEMV was the next identified lever for the small/medium-N
decode shapes (SIMT Triton topped out near 0.3–0.5 TB/s there); revision 3
below implements that path.

A third revision (2026-09-11, same shared GPU) adds `cuda_gemv.py`: an
optional NVRTC-compiled CUDA GEMV family used for most decode (M ≤ 4)
projections when the `nvidia-cuda-nvrtc-cu12` wheel is present (no nvcc or
CUDA headers needed; Triton paths remain the fallback). Three kernel
structures share one source: warp-per-row with lane-owned packed words,
shuffle broadcast, and direct coalesced `__ldg` activation reads (templated
on words-per-lane, K % 32 == 0); row-per-lane for 1-bit N ≥ 8192 (32 rows
per warp, padded shared-memory weight tile, initially an x-sum epilogue); and a scalar
ragged-K general fallback. Two design lessons from ablation on this
contended GPU: per-block activation staging through shared memory costs
more than it saves when L2 is thrashed, and under issue-slot contention the
2-bit large-N decode stays faster on the skinny Tensor Core GEMM (HMMA
needs fewer issue slots), so that one shape keeps the Triton path.
Measured end-to-end through `PackedWeight.linear` (M=1, CUDA-graph
medians, interleaved): vs cuBLAS FP16 — (1024,1024) 1.33×/1.05×,
(4096,1024) 2.59×/2.47×, (1024,4096) 2.36×/1.47×, (51864,1024) 8.7×/8.4×
for 1-bit/2-bit (vocabulary dense numbers inflated by contention;
uncontended dense is ~54us, i.e. ~3.2×). Against revision 2 Triton decode:
1.4–2.9× on every non-vocabulary shape and 2.7×/2.4× on the vocabulary
projection. Compilation adds ~3s once per process. Raw diagnostics:
`kernel-diagnostics-revision3-cuda-gemv.json`. Remaining identified lever:
split-K with an FP32 atomic workspace for the 2-bit (1024,4096) shape
(only 6 warps/SM of parallelism there); deferred — it needs a
graph-capture-safe persistent workspace.

The CUDA correctness review fixed two numerical errors: CUDA epilogues now
round scales to FP16 consistently with the dense/Triton paths, and row-per-lane
accumulation decodes signed coefficients directly. Its former x-sum identity
could return about -0.99 for an all-zero ternary row with finite FP16 inputs;
removing it also removes the order-dependent shared-memory atomic sum. Shared
weight rows now always use an odd stride to avoid bank conflicts at odd KW.
The NVRTC loader also handles PyTorch caching an unset CUDA_HOME before setup.
The revision-3 timings above predate these fixes; performance must be measured
again with exclusive GPU access before drawing efficiency conclusions.

See [CUDA follow-up investigation](CUDA_FOLLOWUP.md) for compiled-code evidence
and isolated accumulator, cooperative-warp, and transposed-layout prototypes.
