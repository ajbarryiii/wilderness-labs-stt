# CUDA encoder experiments and decoder token caching

A newer opt-in hybrid CUDA/Triton ternary runtime now beats our Expanded Triton runtime on the RTX 5090 at 400 W. Median warm batch-one full-transcription latency is 3.43% lower and GPU energy per transcription is 3.01% lower. Text, token IDs and timestamps matched on 2,048 sampled utterances; see [CUDA_IMPROVEMENTS.md](CUDA_IMPROVEMENTS.md). The rest of this document records earlier, independent experiments. Hand-written CUDA was slower than Triton, with 23.02% higher encoder latency and 24.92% higher energy. Direct cuBLAS was also slower, with 7.48% higher latency and 5.33% higher energy. CUDA regenerated from Triton's PTX only matched Triton. A separate ablation of the existing decoder token cache measured 7.29% lower decoder latency and 10.10% lower decoder GPU energy, using 10.5 MB of storage. Both runtimes in the new comparison already include that cache, so none of the new gains come from it.

Measurements below use **24 distinct utterances from eight corpora**, with
three timing/energy windows per variant and utterance. The earlier performance
screens used one clip per 3/10/30-second duration. Their separate 2,048-utterance
recognition validation did not make those timing screens a corpus benchmark.

## Encoder results

Changes are medians of paired per-utterance percentage changes versus the
current Expanded Triton encoder. Positive values mean more time or energy.

| Matmul implementation | Encoder latency change | Encoder GPU energy change | Latency wins |
|---|---:|---:|---:|
| Hand-written CUDA, best candidate per shape band | +23.02% | +24.92% | 0/24 |
| Direct cuBLAS INT8 GEMM | +7.48% | +5.33% | 0/24 |

Hand-written CUDA increased latency by 16.49–27.45% and energy by
20.61–38.83%. cuBLAS increased latency by 4.66–10.47%; its energy change ranged
from −2.02% to +15.88%, with six small energy wins and eighteen losses.

The following are medians within each duration target, eight utterances per
target. Each cell contains **encoder milliseconds / GPU joules per call**.
These exclude the frontend and decoder.

| Audio target | Triton | Hand-written CUDA | cuBLAS |
|---|---:|---:|---:|
| 3 s | 2.032 / 0.617 | 2.369 / 0.842 | 2.156 / 0.610 |
| 10 s | 2.970 / 1.079 | 3.661 / 1.320 | 3.280 / 1.199 |
| 30 s | 7.370 / 2.659 | 9.270 / 3.344 | 7.916 / 2.834 |

The CUDA implementations use explicit INT8 Tensor Core `mma.sync` instructions,
INT32 accumulation, and the same three residual activation components and
FP32 reconstruction as Triton. Eighteen hand-written configurations were timed,
covering direct loads, padded shared-memory staging, `cp.async`, double-buffered
pipelines, output tile sizes and split-K. The code defines one additional
shared-memory configuration covered by correctness tests but not this timing
sweep. Direct cuBLAS stacks the three activation components into one GEMM and
uses a separate reconstruction epilogue.

The broader microbenchmark sweep covered fifteen representative encoder shapes
(three row counts and five projection shapes). Hand-written CUDA did not beat
Triton on any tested shape. cuBLAS won one shape by 2.15%—24.99 µs versus
25.54 µs for the longest down-projection—but that small microbenchmark win did
not improve the full encoder. Microbenchmarks exclude activation quantization
and are not energy measurements.

A fresh 10-second encoder profile put the existing dot products and their
split-K finish kernels at 52.3% of traced encoder kernel time, confirming a
substantial target for optimization. Triton already fuses residual products
and reconstruction. Staging overhead and separate intermediate buffers are
possible explanations for the CUDA regressions; these runs do not establish
a detailed hardware bottleneck diagnosis or rule out a better CUDA design.

## CUDA source from Triton's generated PTX

A separate compiler experiment now generates CUDA functions from the current
Triton encoder's PTX. Each CUDA function encloses the original instruction body
in inline PTX assembly, adapting parameter loads and shared-memory addresses.
Original register names are changed to prevent collisions with NVRTC's operand
registers. This preserves Triton's thread mapping, memory pipeline and Tensor
Core operations as a starting point for further changes. It is **CUDA with
inline PTX**, not recovered high-level C++.

All **15 representative encoder shapes** compiled with NVRTC 12.9 for SM120a
and produced **bit-for-bit matching numerical outputs** against Triton. They
also passed the dense FP32 reference check. Compute Sanitizer memcheck repeated
all 15 shapes, including CUDA-graph replay after changing input contents, with
**zero memory-access errors**; API-error reporting was disabled.

Microbenchmark latency was near parity: the median paired change across shapes
was −1.01%, ranging from −3.79% to +0.24%. These short tests use synthetic,
resident, prequantized matrices and the same split-K finish kernel. They do
not measure energy or full-model performance on spoken utterances. The original
and regenerated binary containers differ; identical native instruction
sequences have not been established. Numerical equality does not require
identical compiler output.

The [generator](ptx_cuda.py) provides a concrete CUDA starting point without
selecting it in the production runtime. Generated `.cu`, original PTX and both
binaries remain on `/mnt/hd`; the
[evidence JSON](../results/ptx-cuda-rtx5090.json) records their paths and hashes.
Run `finetune/parakeet-ternary/python -m inference.ptx_cuda` to regenerate the
15-shape proof, or add `--check-only` for correctness and changed-input graph
checks without timing. The GPU cap remains 400 W.

The reconstructed CUDA adapter has now also been compared against the current
Expanded Triton **full encoder** on the same **24 utterances from eight
corpora**. The result is practical parity: median paired encoder latency was
**0.085% lower**, and GPU energy per call was **0.188% higher**. Latency changes
ranged from −0.339% to +0.164%, with fifteen wins and nine losses; energy changes
ranged from −3.042% to +3.217%, with ten wins and fourteen losses. Typical
within-variant ranges across the three windows were 0.113% for latency and
2.649% for energy, larger than the aggregate differences. These measurements
do not establish a material speed or energy gain.

| Audio target | Triton encoder ms / J | CUDA from PTX encoder ms / J |
|---|---:|---:|
| 3 s | 2.030 / 0.616 | 2.029 / 0.624 |
| 10 s | 2.967 / 1.078 | 2.964 / 1.075 |
| 30 s | 7.343 / 2.648 | 7.350 / 2.654 |

Each target has eight utterances; table cells are medians within targets.
**All 24 encoder tensors and lengths matched bit for bit.** The adapter
generated 65 shape/configuration/activation specializations. Compilation and
capture were outside the measured windows. Variant order alternated by both
utterance and window so each variant ran first equally often. The rest of the
encoder, activation quantization and split-K finish kernel were identical.
Both models remained resident, so this is not a deployment VRAM comparison.

The two adapter regression tests passed: omitted/present biases, partial output
tiles, zero/positive/negative rows, normalization, SiLU, transposed convolution
inputs and changed-input graph replay after 128 MB of allocator churn.
Compute Sanitizer memcheck also repeated both adapter tests with zero
memory-access errors; API-error reporting was disabled. See
the [full-encoder evidence](../results/ptx-cuda-encoder-rtx5090.json) for all
utterance/window results, source hashes and local raw-artifact paths. Reproduce
with `finetune/parakeet-ternary/python -m inference.cuda_experiments encoder
--candidates ptx_cuda --seconds 1 --repeats 3` (one shell command), or run
`finetune/parakeet-ternary/python -m unittest inference.test_ptx_cuda -v` for
the adapter checks. The GPU cap stays at 400 W and Triton remains the production
default. This CUDA path supplies a tested starting point for further tuning.

## Decoder cache results

The table precomputes `embedding @ first_layer_input_weight.T` once, replacing
the first LSTM layer's token-input multiplication with a lookup. Its FP32 shape
is 1,025 × 2,560: **10,496,000 bytes** (10.50 MB / 10.01 MiB). That removes one
of the two-layer predictor's four equally sized dense multiplications per
token; it does not remove the history-dependent recurrent work.

| Audio target | Uncached decoder ms / J | Cached decoder ms / J | Paired latency reduction | Paired energy reduction |
|---|---:|---:|---:|---:|
| 3 s | 0.615 / 0.0924 | 0.579 / 0.0837 | 6.05% | 8.43% |
| 10 s | 1.519 / 0.2759 | 1.398 / 0.2450 | 7.67% | 11.53% |
| 30 s | 3.037 / 0.5640 | 2.763 / 0.4984 | 8.47% | 9.72% |

The cache improved decoder latency and energy on all 24 utterances. Paired
reductions ranged from 2.73–9.35% for latency and 2.60–15.39% for energy.
Text, token IDs and timestamps matched on all 24. These are **decoder-only**
gains; they cannot be applied again to the already cached published runtime.

Caching the complete prediction by last token ID would be incorrect: different
prefixes can produce different hidden and cell states for the same token.
An identical-prefix cache could reuse predictor states and outputs, although
the joint network would still need the current audio features. Its additional
benefit depends on real prefix hit rates and lookup overhead, which were not
measured here. Repeated benchmark replays must not be used as an artificial
prefix hit-rate estimate.

A full conditional-graph decoder profiling attempt recorded only two LSTM
kernel calls for a 56-token transcript. It does not account for the repeated
conditional execution, so we retain the trace for diagnosis and do **not**
derive a predictor latency share or prefix-cache speedup ceiling from it.

## Method and validation

The selection takes a distinct positive-duration utterance nearest each
3/10/30-second target from AMI, Common Voice, Earnings22, GigaSpeech,
LibriSpeech clean/other, SPGISpeech and VoxPopuli. Total audio is 318.95 seconds;
actual durations range from 3.00 to 30.26 seconds. Selection uses duration,
not transcript content. This is a broader but still modest warm batch-one
sample, not a full-corpus or cold-start benchmark.

Each variant has three windows of at least one second, with a 0.3-second warmup
before each window. Latency summaries take the median iteration latency per
window, then the median across windows. Energy uses the NVML GPU-board total
energy counter divided by completed calls; host energy is excluded and idle
energy is not subtracted. The GPU lock, foreign-process guard and 400 W cap
checks apply throughout. Encoder variant order reverses on the middle window;
decoder cache-on/off order alternates by utterance.

Encoder measurements replay captured graphs from identical GPU features to
GPU encoder outputs. Decoder measurements use identical saved encoder outputs
and include decoding to CPU text. Compilation, initial capture, table creation
and encoder execution are outside the decoder measurements. Three encoder
models remain resident to preserve graph argument lifetimes, so their process
VRAM readings are **not** a deployment VRAM comparison. The table's bytes are
storage size, not a measured process-memory delta.

All 24 encoder comparisons passed `atol=3e-6, rtol=3e-5`; maximum absolute
errors were 7.16e-7 for hand-written CUDA and 6.11e-7 for cuBLAS. Three GPU
regression tests passed, covering all 20 defined configurations, partial tiles,
zero/positive/negative weight rows, SiLU, long reductions and graph replay
after allocator churn. Compute Sanitizer 12.9.79 memcheck reran those tests
with **zero memory-access errors**. API-error reporting was disabled; this is
not an audit of every CUDA API call. The full recognition suite was not rerun
for these rejected encoder candidates, and the production runtime is unchanged.

The measurements used PyTorch 2.11.0+cu128, Triton 3.6.0, NVRTC 12.9, driver
595.71.05 and the existing `main-M1-P2-lr5e-4/export` weights. The compact
[evidence JSON](../results/cuda-rtx5090.json) includes per-utterance and per-window
results, source/model hashes, microbenchmarks and hashed local raw-artifact
paths. Raw traces, source snapshots and logs remain on `/mnt/hd`.

## Run the experiments

From the repository root, with the project data disk mounted and the GPU
already capped at 400 W:

```sh
P=finetune/parakeet-ternary
$P/python -m inference.cuda_experiments encoder --candidates auto cublas --seconds 1 --repeats 3
$P/python -m inference.cuda_experiments decoder-cache --seconds 1 --repeats 3
$P/python -m inference.cuda_experiments micro
$P/python -m unittest inference.test_cuda_matmul -v
```

The micro command sweeps all currently defined configurations. Exact arguments
and source snapshots for the recorded runs are retained with their raw results.
The CUDA adapter is experiment-only and is not selected by the inference CLI.
