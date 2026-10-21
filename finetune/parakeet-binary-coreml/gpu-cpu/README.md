# kea-b (MB1) encoder on the M1 Pro GPU and CPU: can a non-ANE backend beat the ANE?

Question: can a Metal GPU or CPU encoder for MB1 (24-layer FastConformer, 604M weights in {-1, +1} with one FP32
scale per row) beat the ANE encoder in any LocalFlow bucket (4 / 8 / 15 / 30 s = 51 / 101 / 188 / 376 encoder
frames), while using less memory than the ternary model (Core ML weight.bin 345.4 MB)?

**Answer: no, not in any bucket.**
- **Exact 1-bit GPU encoder (MLX + custom Metal kernel):** 32.4 / 50.5 / 83.6 / 161.8 ms. The ANE 1-bit arm B1Q
  takes 17.8 / 21.8 / 44.2 / 124.1 ms, and the ternary kea-t 14.8 / 19.0 / 39.3 / 127.9 ms. The GPU encoder is
  1.3 to 2.3× slower than the 1-bit ANE arm. It does use less memory: 153 MB of weights.
- **Best-case GPU bound:** projections at the GPU's measured matrix-unit ceiling plus MLX's measured cost for
  everything else gives 19.4 / 40.6 / 74.5 / 133.3 ms, still slower in every bucket.
- **CPU:** even at 100% of the NEON int8 dot-product peak on all 8 performance cores, the CPU loses every bucket
  by 1.1 to 2× (with one activation component) or 2.3 to 4× (with LocalFlow's two).
- **Recommendation:** keep the ANE for every bucket. Do not port a Metal encoder. An ANE + GPU split is not
  supported by these numbers.

Everything below is informational: a shared 16 GB M1 Pro, macOS 27, synthetic inputs, GPU timings from Metal
command-buffer timestamps or wall-clock MLX calls, and timing jobs run under jobs-mac `quiet_host` + macguard.

## 0. The bar (measured by the bench agent, `kb1bench`, run final-ane-20261021, same Mac)

Encoder only, warm `MLModel.prediction`, typical ms. These replace the briefing's pilot numbers (14.0 / 15.4 /
19.6 / 40.2 at 2 / 4 / 8 / 15 s), which did not include 30 s.

| ANE variant | 4 s | 8 s | 15 s | 30 s | weight.bin | loaded phys+neural |
| --- | --- | --- | --- | --- | --- | --- |
| kea-t (ternary C6s8, current) | 14.8 | 19.0 | 39.3 | 127.9 | 345.4 MB | 1600 MB |
| B1Q (1-bit ANE encoding) | 17.8 | 21.8 | 44.2 | 124.1 | 153.2 MB | 738 MB |
| B1S (1-bit ANE encoding) | 17.9 | 22.0 | 44.5 | 124.3 | 153.2 MB | 736 MB |

Source: `/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/kea-b1/bench/final-ane-20261021/summary.md`.
At 30 s the ANE is out of line: 3.2× the 15 s time for 2× the frames.

## 1. Roofline (written before measuring)

**Work per frame.** The projections cost 24.12M MACs per layer per frame:
- feed-forward: 2 × 2 × 4096 × 1024;
- attention q / k / v / out: 4 × 1024²;
- pointwise convs: 2048 × 1024 and 1024².

Over 24 layers that is 578.9M MACs, or 1.158 GFLOP per frame. Attention (QKᵀ, AV, Q × pos) adds about
4 T × 1024 × 24 MACs per frame: 0.9% at 51 frames, 6.4% at 376.

**GPU throughput.** The M1 Pro GPU peaks at 16 cores × 128 FP32 lanes × 2 × 1.296 GHz = 5.31 TFLOPS.
- On G13, FP16 runs at the same ALU rate as FP32.
- `simdgroup_matrix` runs on those same ALUs. There are no tensor cores, no int8 dot product and no binary MMA.
- So 1-bit weights do not reduce the arithmetic. Each weight still costs one FMA per frame, or one threadgroup
  lookup plus one add per 4 weights with tables.

**Measured ceilings** (`b1bench peak`):
- `simdgroup_multiply_accumulate`: 4.32 TFLOPS with FP32 fragments, 4.66 with FP16 fragments (81–88% of peak).
- My scalar-FMA probe reached only 1.90. It is latency-bound, so ignore it.

**Bandwidth is not the limit.**
- The 1-bit codes are 75.5 MB per pass, about 0.38 ms at roughly 200 GB/s.
- Even at 51 frames the intensity is 816 FLOP per weight byte, about 30× the ridge point.
- So the encoder is ALU-bound in every bucket.

| Bucket | Frames | Projection GFLOP | At 5.31 TFLOPS | At measured MMA ceiling 4.32 | ANE kea-t / B1Q |
| --- | --- | --- | --- | --- | --- |
| 4 s | 51 | 59.0 | 11.1 ms | 13.7 ms | 14.8 / 17.8 |
| 8 s | 101 | 116.9 | 22.0 ms | 27.1 ms | 19.0 / 21.8 |
| 15 s | 188 | 217.7 | 41.0 ms | 50.4 ms | 39.3 / 44.2 |
| 30 s | 376 | 435.3 | 82.0 ms | 100.8 ms | 127.9 / 124.1 |

**What the roofline already says, before measuring:**
- **8 s is out of reach** even at theoretical peak, before any non-projection work: 22.0 ms at peak against
  19.0 / 21.8 ms for the ANE.
- **15 s is out of reach** at the measured ceiling (50.4 ms against 39.3 / 44.2 ms). At theoretical peak it beats
  only the 1-bit ANE arm, by 3.2 ms (41.0 against 44.2 ms), and that margin is used up by the non-projection
  work measured in section 2.
- **At 4 s**, the projections plus the rest of the encoder (about 1,000 other dispatches) would have to fit in
  14.8 to 17.8 ms. That leaves 1–4 ms for everything else at the measured ceiling.
- **At 30 s** there is room on paper, because the ANE is slow there. The Metal microbenchmark had to settle these
  two buckets.

## 2. MLX

**1-bit support.** There is none in MLX 0.32.3, the Mac pyenv version.
- `mx.quantize` refuses `bits=1`. The supported bits are 2, 3, 4, 5, 6 and 8, and group sizes are 32, 64 and 128.
- `quantized_matmul`, `gather_qmm` and `dequantize` with hand-built `bits=1` arrays fail with
  `Unable to load kernel affine_qmv_float16_t_gs_*_b_1`.
- So the smallest native format is 2-bit.

**Exact 2-bit mapping.**
- Codes: q = 2·bit ∈ {0, 2}. Scales are FP16(s) and biases are −FP16(s), repeated over every group.
- `mx.dequantize` then gives ±FP16(s) exactly.
- `mlx_encoder.py exact` checks this on all 264 modules (240 projections + 24 `linear_pos`), FP16 and FP32: dequantized == codes × FP16(s)
  bit for bit, plus an independent unpack of the packed words and of the 1-bit words: 264/264 pass
  (`results/mlx_exact_g64.json`).

**Exact 1-bit in MLX.** The custom Metal kernel from section 3 (group-of-4 LUT, plus the MMA variant) is plugged
in through `mx.fast.metal_kernel` (`kernels/b1_mlx.py`, encoding `b1`). Weights are the export bytes read as
uint32 words, plus FP16(s) per row.
- `mlx_encoder.py b1check` tests 600 real-module cases: 30 MB1 modules (layers 0 / 11 / 23 × 10 projections),
  M = 7 / 51 / 101 / 188 / 376, 4 kernel tiles.
- The inputs are integer activations in [−3, 3], which make every sum exact in FP32.
- The FP16 output equals FP16(FP32(sum) × FP32(FP16(s))) bit for bit in all 600 cases.
- Whole-encoder agreement against the same graph in FP32 (synthetic mel), relative Frobenius error:

  | | 4 s | 15 s |
  | --- | --- | --- |
  | b1 FP16 | 9.3e-4 | 1.10e-3 |
  | MLX's own 2-bit FP16 path | 1.11e-3 | 1.13e-3 |

  That is, the 1-bit path is as accurate as MLX's built-in 2-bit path.

**Graph.** It mirrors WP6b's `mlxarm/encoder.py`, which was gated exact against the reference for M_P2 (same
FastConformer architecture). Buckets are LocalFlow's; `linear_pos` is folded into per-bucket position tables.

**Timing** (`mlx_encoder.py time`, quiet host):
- Each call is `mx.eval` + `mx.synchronize` inside the timed region.
- 5 warm-ups, then 30 timed calls per bucket, in 2 counterbalanced rounds (n = 60).
- Synthetic Gaussian mel input over the full bucket length.

Median ms per bucket (p10–p90 in brackets):

| Encoding | Projection weights | 4 s | 8 s | 15 s | 30 s |
| --- | --- | --- | --- | --- | --- |
| **b1** (1-bit, custom LUT kernel) | 73.2 MB | **32.4** [31.6–33.9] | **50.5** [49.6–58.4] | **83.6** [82.8–87.1] | **161.8** [159.3–173.3] |
| q2g64 (MLX 2-bit, as WP6b) | 180.9 MB | 35.5 [34.9–37.5] | 60.7 [58.4–65.8] | 84.7 [83.5–91.2] | 159.7 [157.7–171.7] |
| q2g128 (MLX 2-bit) | 162.8 MB | 36.3 [34.9–41.8] | 58.5 [57.9–60.6] | 83.9 [82.8–92.4] | 169.5 [158.3–185.0] |
| dense FP16 (diagnostic, not a candidate) | 1157.6 MB | 34.5 [33.7–36.6] | 57.5 [56.8–63.2] | 84.5 [83.4–90.2] | 173.9 [162.5–185.8] |
| ANE B1Q / kea-t (section 0) | 153.2 / 345.4 MB weight.bin | 17.8 / 14.8 | 21.8 / 19.0 | 44.2 / 39.3 | 124.1 / 127.9 |

**Reading the timing table.**
- **Consistent with WP6b.** MB1 at 2-bit matches WP6b's M_P2 2-bit timings at the same frame counts
  (34.1 / 57.1 / 82.1 ms at 51 / 101 / 188 frames).
- **Weight format barely matters on the GPU.** 1-bit, 2-bit and dense FP16 land within about 10% of each other
  in every bucket.
- **Speed against the 1-bit ANE arm.** The best GPU encoding (b1) is 1.8× / 2.3× / 1.9× / 1.3× slower than B1Q.

**Memory (b1 encoding).**
- **Weights: 153.1 MB**, against 345.4 MB for the ternary weight.bin:
  - 73.2 MB projections;
  - 9.8 MB other FP16;
  - 70.2 MB of position tables for all 4 buckets (36.9 MB if only the 30 s table is kept and sliced).
- **Process:** peak MLX allocation 719 MB, max RSS 478 MB. This includes 4 compiled bucket graphs and the 30 s
  attention activations.
- **Load:** 0.26 s from the export.
- **For comparison:** q2g64 weights 260.8 MB (peak 827 MB), dense 1238 MB (peak 1934 MB). The memory condition is
  met, but the latency condition is not.

**Profile** (`mlx_encoder.py profile`). "Projections" is the 240 projection matmuls alone, in one compiled graph
on fixed inputs; "rest" is the full encoder minus that. Values are ms (projection TFLOPS in brackets).

| | 4 s | 8 s | 15 s | 30 s |
| --- | --- | --- | --- | --- |
| b1 projections | 21.8 (2.71 TFLOPS) | 34.6 (3.38) | 65.8 (3.31) | 124.4 (3.50) |
| b1 rest | 9.9 | 18.3 | 24.1 | 37.3 |
| q2g64 projections | 29.6 (1.99) | 46.8 (2.50) | 68.9 (3.16) | 127.1 (3.43) |
| q2g64 rest | 5.7 | 13.5 | 25.0 | 32.5 |

## 3. Custom Metal kernel microbenchmark

`metal/b1gemm.metal` holds three 1-bit-weight × FP16-activation GEMM families, all with FP32 accumulation.
`metal/main.swift` is the standalone Swift harness. It compiles the MSL at run time and times with GPU
timestamps: 20 dispatches per command buffer, median of 7.
- **mma:** `simdgroup_matrix` 8×8 FP32 MMA. The weight fragments are decoded from the packed bits straight into
  each lane's two fragment elements (±1.0f by XOR on the sign bit), and X tiles go through threadgroup memory.
  10 tile shapes.
- **simt:** FMA micro-tiles (TM × TN per thread), with each sign decoded once per (n, k) and reused over TM.
  5 tile shapes.
- **lut:** T-MAC-style tables. Per frame and per group of 4 inputs, the 16 signed sums are built in threadgroup
  memory as FP32 (64 B per table, so different entries hit different banks and equal entries broadcast). Each
  output then costs one table read plus one add per 4 inputs. 4 tile shapes.

**Exactness** (`b1bench check`): all 19 kernels × 9 shapes (M from 7 to 376, K = 1024 / 4096, N = 1024 / 2048 /
4096), with FP32 and FP16 outputs, are bit-identical to a CPU reference on integer activations (exact sums):
171 / 171 cases, 0 mismatches (`results/metal_check.json`).

**Per-shape results.** Best tile per family. The "TFLOPS" columns are FMA-equivalent: 2MNK / t.

| N × K | M | LUT | µs | TFLOPS | MMA | µs | TFLOPS | SIMT | µs | TFLOPS | MPS dense FP16 µs | TFLOPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 × 1024 | 51 | lut_8_1_4 | 131 | 3.26 | mma_4_2_2_2 | 737 | 0.58 | simt_8_4_2_2 | 320 | 1.34 | 146 | 2.93 |
| 4096 × 1024 | 101 | lut_8_2_4 | 263 | 3.22 | mma_4_2_2_2 | 1595 | 0.53 | simt_8_4_1_2 | 744 | 1.14 | 284 | 2.98 |
| 4096 × 1024 | 188 | lut_8_1_4 | 411 | 3.84 | mma_4_2_2_2 | 2347 | 0.67 | simt_8_4_1_2 | 1018 | 1.55 | 425 | 3.71 |
| 4096 × 1024 | 376 | lut_16_1_4 | 894 | 3.53 | mma_4_2_2_2 | 4470 | 0.71 | simt_8_4_1_2 | 1955 | 1.61 | 874 | 3.61 |
| 1024 × 4096 | 51 | lut_8_1_4 | 182 | 2.34 | mma_2_4_1_2 | 981 | 0.44 | simt_4_8_2_2 | 609 | 0.70 | 305 | 1.40 |
| 1024 × 4096 | 101 | lut_8_1_4 | 259 | 3.27 | mma_2_4_2_2 | 1526 | 0.56 | simt_4_8_2_2 | 836 | 1.01 | 355 | 2.39 |
| 1024 × 4096 | 188 | lut_8_1_4 | 576 | 2.74 | mma_4_2_2_2 | 2525 | 0.62 | simt_8_4_2_2 | 1141 | 1.38 | 447 | 3.53 |
| 1024 × 4096 | 376 | lut_8_1_4 | 833 | 3.79 | mma_2_4_2_2 | 5474 | 0.58 | simt_8_4_1_2 | 2220 | 1.42 | 914 | 3.45 |
| 1024 × 1024 | 51 | lut_8_1_4 | 47 | 2.26 | mma_2_4_1_2 | 251 | 0.43 | simt_4_8_2_2 | 159 | 0.67 | 80 | 1.34 |
| 1024 × 1024 | 101 | lut_8_1_4 | 70 | 3.03 | mma_4_2_2_2 | 354 | 0.60 | simt_4_8_2_2 | 222 | 0.96 | 94 | 2.26 |
| 1024 × 1024 | 188 | lut_8_1_4 | 123 | 3.22 | mma_4_2_2_2 | 524 | 0.75 | simt_8_4_2_2 | 292 | 1.35 | 121 | 3.27 |
| 1024 × 1024 | 376 | lut_8_1_4 | 213 | 3.70 | mma_4_2_2_2 | 1152 | 0.68 | simt_8_4_1_2 | 560 | 1.41 | 227 | 3.47 |
| 2048 × 1024 | 51 | lut_8_1_4 | 70 | 3.07 | mma_4_2_2_2 | 359 | 0.60 | simt_4_8_2_2 | 224 | 0.95 | 96 | 2.23 |
| 2048 × 1024 | 101 | lut_8_1_4 | 126 | 3.36 | mma_4_2_2_2 | 661 | 0.64 | simt_8_4_2_2 | 318 | 1.33 | 147 | 2.89 |
| 2048 × 1024 | 188 | lut_8_1_4 | 214 | 3.69 | mma_4_2_2_2 | 1034 | 0.76 | simt_8_4_1_2 | 562 | 1.40 | 229 | 3.45 |
| 2048 × 1024 | 376 | lut_8_1_4 | 423 | 3.73 | mma_4_2_2_2 | 2189 | 0.72 | simt_8_4_1_2 | 1022 | 1.54 | 420 | 3.76 |

**What the per-shape results show.**
- **The 1-bit LUT kernel is the best GPU kernel here.** It is faster than Apple's dense FP16 MPS GEMM on 12 of
  16 shapes, by up to 1.7× at 51 frames. Three of the other four are within 2.3%; the outlier is 1024 × 4096 at
  M = 188 (576 vs 447 µs, 29% slower). It reads 1/16 of the weight bytes, and it is the kernel the MLX `b1` encoding uses.
- **The MMA kernel is far below the ceiling.** It reaches 0.4–0.8 TFLOPS, against 4.3 for the bare MMA
  instruction. Per-lane fragment decoding, scalar threadgroup reads and two barriers per 32 inputs dominate it.
  It was not tuned further, because the bound below shows that even a perfect MMA kernel cannot win.

**Full-encoder projections** (`b1bench encoder`): the 240 projection GEMMs of one encoder pass, with distinct
weights per layer (75.5 MB), in one command buffer at the best tile per shape. GPU ms:

| | 4 s | 8 s | 15 s | 30 s |
| --- | --- | --- | --- | --- |
| LUT | 22.9 | 38.2 | 62.9 | 118.1 |
| SIMT | 68.7 | 102.1 | 154.5 | 290.4 |
| MMA | 118.7 | 200.0 | 310.8 | 652.1 |
| Ideal MMA at the measured 4.32 TFLOPS ceiling | 13.7 | 27.1 | 50.4 | 100.8 |

**Extrapolation and kill criterion.** The extrapolated time is the projections plus the smallest "rest" MLX
measured (5.7 / 13.5 / 24.1 / 32.5 ms):

| | 4 s | 8 s | 15 s | 30 s |
| --- | --- | --- | --- | --- |
| Best measured kernel (LUT) | 28.6 | 51.7 | 87.0 | 150.6 |
| Measured end to end (MLX b1) | 32.4 | 50.5 | 83.6 | 161.8 |
| Ideal MMA kernel at the ceiling | 19.4 | 40.6 | 74.5 | 133.3 |
| ANE B1Q / kea-t | 17.8 / 14.8 | 21.8 / 19.0 | 44.2 / 39.3 | 124.1 / 127.9 |

**Conclusion of the kill criterion.**
- **Every bucket is slower than the ANE,** even for the ideal kernel.
- **4 s:** an ideal projection kernel would leave only 1.1–4.1 ms for the non-projection work, which MLX measures
  at 5.7–9.9 ms.
- **30 s is the only near miss.** Winning there would need all four of:
  - near-peak MMA projections (100.8 ms);
  - a fused attention / conv / norm path costing under about 23 ms, where MLX takes 32–37 ms (the attention
    arithmetic alone is about 6.4 ms at the ceiling);
  - a full hand-written Metal encoder;
  - a projection kernel 1.2× faster than anything measured here.

  Even then the margin over the ANE would be a few percent. That is not a clear win, so a full Metal port was not
  started.
- **The anomalous 30 s ANE time** (124–128 ms, versus 39–44 at 15 s) is probably the cheaper thing to
  investigate on the ANE side.

**ANE + GPU split** (for example, running only the 30 s bucket on the GPU): not supported. The GPU is slower in
every bucket, including 30 s (161.8 vs 124.1 ms measured; at best about 133 ms).

## 4. CPU

**The cpu1bit team's results** (`../../../finetune/parakeet-binary/cpu1bit/NOTES.md` in the upstream checkout,
read-only; plan `plans/cpu-benchmark-mac.md`):
- **No timed P-core numbers published yet.** As of their 12:26 note their quiet-window stages (validate,
  screening) had not run.
- **What exists on the Mac** (`/Users/ajbarry/wilderness-labs-stt-artifacts/cpu1bit/results/`):
  - exactness records;
  - plumbing and validate-plumbing replays under the background E-core clamp: 7–15 s per replay of a 16 s clip,
    which they label not comparable.
- **Dev numbers they did publish** (E-cores, background QoS):
  - **B-NEON (SDOT):** 47–52 GMAC/s per E-core thread at 2.06 GHz, about 90% of its kernel loop. The inner loop is
    modelled at 94% of SDOT peak on Firestorm.
  - **T-LUT (TBL):** parity with an SDOT microkernel on L1 data.
  - **P bit-serial (EOR + CNT + UDOT):** at least 3× SDOT's cycles per MAC, on E-cores and in the Firestorm uop
    model. Their P-core floor for a 16 s clip (T = 200, i8x2, 231.5 G MACs) is at least 0.45 s.
- **So no 1-bit CPU trick beats plain SDOT on M1.**

**SDOT ceiling from their model.**
- Firestorm issues 4 SDOTs per cycle at 16 int8 MACs each, so 8 P-cores × 64 × about 3.0 GHz = 1.54 T int8 MAC/s.
- LocalFlow's CPU format quantizes activations into C = 2 (or 3) int8 components, which doubles (or triples) the
  MACs.
- Projections alone at 100% of that peak:

| | 4 s | 8 s | 15 s | 30 s |
| --- | --- | --- | --- | --- |
| SDOT floor, C = 2 (LocalFlow) | 38.4 ms | 76.1 | 141.7 | 283.4 |
| SDOT floor, C = 1 (one int8 component, below LocalFlow's precision) | 19.2 | 38.1 | 70.9 | 141.7 |
| ANE B1Q / kea-t | 17.8 / 14.8 | 21.8 / 19.0 | 44.2 / 39.3 | 124.1 / 127.9 |

  At C = 2 the CPU is 2.3–4.0× slower than the faster ANE variant in each bucket. Even the unrealistic C = 1
  floor loses every bucket, by 1.1–2.0×.

**Measured CPU data points** (mine, guarded, quiet host):
- **AMX through Accelerate** (`cpu/amxbench.swift`): `cblas_sgemm` FP32 on dense ±1 weights, one layer's weights
  reused × 24. This is optimistic, since it ignores expansion and memory. The 240 projections take
  136.4 / 173.0 / 218.7 / 335.5 ms (0.43 / 0.68 / 1.00 / 1.30 TFLOPS) (`results/cpu_amx_sgemm.json`).
- **Core ML CPU, ternary** (WP7, `../../parakeet-ternary/ios/README.md`): C3 / C4 on `cpuAndGPU`, which WP6a's
  compute plans place almost entirely on the CPU, took 40.1–40.3 / 48.8–49.2 / 81.2–82.5 / 122.9 ms at
  2 / 4 / 8 / 15 s, against 14.3 / 15.6 / 19.7 / 40.1 ms for the C6s8 ANE.
- **Core ML `cpuOnly` on the 1-bit arm B1Q:** not measurable under the 4 GB guard.
  - The run used the bench agent's `kb1bench.py`, unmodified, with output in
    `$A/kea-b1/gpu-cpu/bench-cpu-B1Q/` (job `kea-b1-gpucpu-coreml-cpu`).
  - The first load at a new path was killed by macguard after 21 s at 4.26 GB RSS (cap 4 GB): Core ML expands
    the 1-bit constexpr weights for the CPU.
  - That alone breaks the memory condition, which requires less than the ternary model's 345 MB of weights /
    1.6 GB loaded.
  - Measuring its latency needs a higher `--rss-cap`, which needs the user's approval. It was not attempted.

**Conclusion: a CPU encoder cannot beat the ANE in any bucket.** Even the theoretical SDOT floor with a single
activation component loses each one.

## Files

| File | What it is |
| --- | --- |
| `mlx_encoder.py` | MB1 encoder in MLX (self-contained: numpy + mlx). Encodings `q2g{32,64,128}`, `b1`, `dense`. Commands `exact`, `b1check`, `time`, `profile` |
| `kernels/b1_mlx.py` | The LUT and MMA 1-bit kernels as `mx.fast.metal_kernel`; tile table loaded from `results/metal_bench.json` |
| `metal/b1gemm.metal` | MSL: `b1_mma`, `b1_simt`, `b1_lut` templates and instantiations |
| `metal/main.swift` | `b1bench`: `check`, `peak`, `bench`, `encoder` |
| `cpu/amxbench.swift` | Accelerate `cblas_sgemm` projection ceiling |
| `run_guarded.sh` | Runs a command under macguard and retries while it refuses (lock held or low memory) |
| `timing_metal.sh`, `timing_mlx.sh`, `timing_all.sh` | The timing sequences |
| `results/*.json` | Raw results: `mlx_exact_g64`, `mlx_b1check`, `mlx_time_{b1,q2g64,q2g128,dense}`, `mlx_profile_{b1,q2g64}`, `metal_check`, `metal_peak`, `metal_bench`, `metal_encoder`, `cpu_amx_sgemm` |

Model files and binaries stay outside git, in `/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/kea-b1/gpu-cpu/`:
`export/export.safetensors` (copy of the MB1 export, SHA-256 9873d2f1…), `bin/b1bench`, `bin/amxbench`.

## Reproduce (Mac, from `finetune/parakeet-binary-coreml/gpu-cpu/` of the twin worktree)

```sh
A=/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/kea-b1/gpu-cpu
PY=/Users/ajbarry/workspace/github.com/wilderness-labs-stt/finetune/parakeet-ternary/ios/pyenv/.venv/bin/python
SDK="$(xcrun --sdk macosx --show-sdk-path)"
xcrun swiftc -O -sdk "$SDK" metal/main.swift -o $A/bin/b1bench
xcrun swiftc -O -sdk "$SDK" cpu/amxbench.swift -o $A/bin/amxbench
# correctness (no quiet host needed)
./run_guarded.sh 4G 900 -- $PY mlx_encoder.py exact --export $A/export/export.safetensors --out results/mlx_exact_g64.json
./run_guarded.sh 4G 600 -- $A/bin/b1bench check metal/b1gemm.metal results/metal_check.json
./run_guarded.sh 4G 1200 -- $PY mlx_encoder.py b1check --export $A/export/export.safetensors --out results/mlx_b1check.json
# timing: as jobs-mac jobs with priority normal, full_machine, quiet_host
SKIP_BENCH=0 ./run_guarded.sh 4G 3000 -- ./timing_all.sh      # peak, bench, encoder sequence, MLX time + profile
MG_EXTRA="--threads 8" ./run_guarded.sh 4G 900 -- $A/bin/amxbench results/cpu_amx_sgemm.json 7
```

Set `PYTHONDONTWRITEBYTECODE=1` for the Python runs. Results are synced back with
`~/.cache/kea-b1/pull-from-mac.sh`.
