# RTX 5090 packed ternary kernel measurements — 2026-10-05

These are the first-pass measurements. The subsequent optimization pass and
its experiment ledger are in [EXPERIMENTS.md](EXPERIMENTS.md).

The packed implementation reduces encoder latency and GPU energy while retaining
all 264 layers as 2-bit weights. For full transcription, enable the optional
encoder graph cache: the measured 10-second workload is **1.48× faster and uses
36.8% less GPU energy** than the same FP32 model with the same graph optimization.
All **512/512** sampled transcripts match the FP32 reference exactly.

## Measured performance

RTX 5090, SM120, driver 595.71.05, PyTorch 2.11.0+cu128, Triton 3.6.0. The existing
400 W board power limit was retained; no clocks or power settings were changed.
Both variants use the M1 trained export, identical non-ternary layers, and strict
FP32 cuBLAS/cuDNN settings. Packed matmuls use compensated BF16 Tensor Core
arithmetic with FP32 inputs, outputs, scales, and accumulation.

Numbers are medians across three alternating 10-second windows per variant,
each following two seconds of warm-up. Joules are NVML cumulative GPU board
energy divided by completed inferences. Lower watts are not required for lower
energy: the 10-second packed encoder draws more power but finishes much sooner.

| Workload | Dense latency | Packed latency | Speedup | Dense J/inference | Packed J/inference | Energy reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Encoder, 3.00 s audio | 6.753 ms | 4.076 ms | 1.66× | 2.381 | 1.314 | 44.8% |
| Encoder, 10.00 s audio | 12.586 ms | 6.344 ms | 1.98× | 4.580 | 2.456 | 46.4% |
| Encoder, 30.04 s audio | 22.555 ms | 14.351 ms | 1.57× | 8.695 | 5.537 | 36.3% |
| Full transcription, 10 s, encoder graphs on both | 19.384 ms | 13.057 ms | 1.48× | 5.779 | 3.653 | 36.8% |
| Full transcription, 10 s, eager | 19.816 ms | 21.423 ms | 0.925× | 5.841 | 4.655 | 20.3% |

**Eager full transcription remains 8.1% slower.** Python and driver launch
overhead offsets the faster kernels. The graph cache is important for latency
gains on repeated shapes; it is opt-in and warms/captures each cached shape once.
It does not pad or bucket audio. Shapes beyond its configured capacity run eagerly.

For full transcription with encoder graphs, p95 latency is 19.966 ms dense vs
13.473 ms packed; average board power is 296.6 W vs 278.5 W. The conservative
comparison of the highest-energy packed window with the lowest-energy dense
window still saves 36.6%. Encoder energy savings under that comparison are
44.0%, 45.7%, and 36.1% for the three durations.

Encoder timings replay a graph with mel features already on the GPU. Full
transcription includes preprocessing, encoding, greedy TDT decoding and text
construction, with audio already on the GPU. Loading, compilation, capture,
audio disk I/O and CPU energy are excluded. The desktop remains active; the
meter rejects other compute processes. These are measurements of three fixed
utterances, not a latency distribution across the speech corpus or an embedded
device power estimate. The control is the existing strict-FP32 NeMo runtime,
not TensorRT or a lower-precision dense deployment.

## Accuracy and storage

The paired subset contains 64 evenly spaced utterances from each of LibriSpeech
clean/other, AMI, Earnings22, GigaSpeech, SPGISpeech, VoxPopuli and Common Voice:
512 utterances, 56.67 minutes, 8,782 reference words. All transcripts, including
punctuation and case, are identical. Each dataset's WER and edit counts are
therefore unchanged. This is subset validation, not a full test-set rerun.

Across the three timed clips, maximum absolute encoder difference is
`3.825e-7`, and maximum relative L2 difference is `4.291e-7`. Each captured
encoder graph was replayed and checked against its own eager implementation
before measurement. Full transcription with the encoder cache also matched
eager transcription for each variant.

All 24 final microbenchmark cases (the four model matrix shapes at 1, 16, 64,
128, 384 and 1,024 input rows) pass numerical comparison and are 1.30–3.97×
faster than dense FP32 in the hot-weight graph benchmark. The formerly slow
single-row `(N=1024, K=4096)` projection now takes 3.14 μs versus 6.52 μs dense.
These cache-friendly matrix timings are separate from the model measurements.

Eight kernel/runtime tests pass, covering both precision modes, ragged shapes,
zero rows, cancellation, wide activation ranges, strided convolutions, biases,
empty inputs, output ownership, stream behavior, graph replay with new inputs,
and validation failures. An additional trained-model integration test passes
after changing both features and lengths in a cached graph, and checks that
later calls do not mutate earlier outputs.

Persistent model parameter/buffer storage falls from **2,512,589,592 bytes to
249,434,904 bytes (90.1% less)**. This counts the FP32 remaining layers and NeMo
buffers as well as packed weights. It excludes activation workspaces, allocator
reserves, and graph pools, so it is not a peak-VRAM claim. The packed runtime
never transfers a dense ternary weight matrix to the GPU.

## Iteration record

1. The first direct packed TF32x3 tile kernel was correct but underutilized the
   GPU: the 64-row square projection took about 23 μs versus about 8 μs dense.
2. Transposed int32 decode words, split-K parallelism, and fixed shape-dependent
   tiles addressed bandwidth/layout and occupancy. A separate SIMT path handles
   1–4 rows; its long-K reduction also needed splitting.
3. Three BF16 components per FP32 activation improved Tensor Core throughput.
   Accumulating residual products separately was necessary: repeatedly adding
   tiny residuals directly into the high sum lost accuracy. The retained version
   combines high and residual sums before applying the FP32 row scale.
4. Pre-expanding activation components to GPU buffers did not justify the extra
   traffic and launch overhead in the tested shapes, so that prototype was removed.
   Components are formed in registers in the final kernel.
5. A bounded sweep over tile sizes and split counts selected static dispatch
   settings. Full-model measurements exposed eager launch overhead; the optional
   encoder graph cache then produced the full-transcription gain above.

Compiled PTX was inspected: `.target sm_120a` and
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` confirm the intended target
and Tensor Core arithmetic. The kernel is Triton code compiled for the GPU;
there is no Python arithmetic fallback in packed inference.

## Evidence and reproduction

Run commands are in [README.md](README.md). Model integration test:

```sh
PARAKEET_KERNEL_EXPORT=/mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/export \
  finetune/parakeet-ternary/python -m unittest inference.test_model_graphs -v
```

Authoritative final model measurements, raw telemetry, transcript pairs,
source hashes and GPU metadata:

[`20261005T102259-model`](/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T102259-model/summary.json).
The export SHA-256 is
`a287e97719c451b785be2cd01ecc861fcaa010ebaa4ff2841783ec78fcd61503`.

The bounded tile sweep is
[`20261005T100654-tune`](/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T100654-tune/micro.json).
The final matrix benchmark is
[`20261005T103022-micro`](/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T103022-micro/micro.json).
The first-pass `kernels.py`, `runtime.py`, `graphs.py` and `benchmark.py` SHA-256
hashes matched the source hashes recorded in this model run at measurement
time. Active files now include the subsequent optimization pass.
Earlier exploratory results are not the final measurements. In particular, an
early graph diagnostic compared capture buffers before their first replay; its
zero/NaN encoder errors were invalid. The final benchmark explicitly replays
and verifies outputs before scoring or measuring them.

All datasets, weights, generated compiler artifacts, and raw measurement files
remain under `/mnt/hd/wilderness-labs-stt/`.

## Bitwise / lookup follow-up

In response to the question about avoiding BF16, two additional paths were
tested on actual Parakeet weights, with FP32 activations and outputs:

- `bitwise.py` implements a tile-local FP32 lookup table: compute the 16 subset
  sums of four activations, index them with the packed positive and negative
  weight bits, subtract, and reduce. It never converts to BF16 or uses MMA.
- The existing packed SIMT reduction was extended to encoder-sized row counts
  for comparison. It also uses no BF16. Production dispatch already selects
  this path for 1–4 rows.

The lookup idea is related to [T-MAC](https://github.com/microsoft/T-MAC), whose
low-bit weight lookup approach can retain floating-point activations. The
implementation here is a separate Triton GPU prototype. A bounded 72-config
sweep selected lookup tile sizes; no activation quantization was introduced.
Inspected lookup PTX has shuffle instructions, no BF16, and no `mma.sync`.

| M, N, K | Production packed | FP32 lookup | FP32 SIMT |
| --- | ---: | ---: | ---: |
| 64, 1024, 1024 | 5.27 μs | 13.41 μs | 23.98 μs |
| 128, 4096, 1024 | 23.12 μs | 95.14 μs | 182.97 μs |
| 1, 1024, 4096 | 3.27 μs | 5.04 μs | 3.27 μs |

For `(128,4096,1024)`, three alternating five-second energy windows, with
32 calls per graph replay to amortize host launches, measured:

| Path | GPU mJ/call | Average board watts |
| --- | ---: | ---: |
| Production packed Tensor Core | 9.12 | 385.3 |
| FP32 lookup | 37.17 | 385.4 |
| FP32 SIMT | 71.53 | 385.8 |

The tested lookup path costs about **4.1× the latency and GPU energy** of the
packed Tensor Core path on that encoder projection. It is retained as an
explicit experiment, not installed in normal model dispatch. These are
cache-resident kernel diagnostics, not a full-model LUT deployment measurement
or a proof that every possible lookup implementation will lose.

The repository also contains earlier Whisper W2A1/W2A2 popcount experiments
under `custom/inference-efficiency/`. Those changed activations to binary or
ternary values and did not validate recognition quality. Parakeet's trained
model has floating-point activations; a practical packed-activation/popcount
path is a separate quantization and accuracy experiment. It has **not** been
validated for this Parakeet model.

Two additional LUT/SIMT numerical and graph-replay tests pass. Reproduce with:

```sh
finetune/parakeet-ternary/python -m unittest inference.test_bitwise -v
finetune/parakeet-ternary/python -m inference.bitwise_benchmark
```

Raw candidates, timings, source hashes and energy windows:
[`20261005T104228-bitwise`](/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T104228-bitwise/comparison.json).

## Standalone GPU memory

A fresh process loaded only the packed model and transcribed one clip at a time
on the RTX 5090. Persistent weights and buffers total **249,434,904 bytes**
(249.4 MB / 237.9 MiB), compared with **2,512,589,592 bytes** for the dense FP32
model: a 90.1% reduction. This tensor count excludes runtime allocations.

After warming each workload with three additional transcriptions:

| Execution | Audio seconds | Peak live PyTorch allocation | PyTorch reserved memory | Observed process GPU memory |
| --- | ---: | ---: | ---: | ---: |
| Eager | 3.00 | 301 MB | 386 MB | 1.13 GB |
| Eager | 10.00 | 350 MB | 463 MB | 1.20 GB |
| Eager | 30.04 | 502 MB | 629 MB | 1.37 GB |
| Encoder graphs | 3.00 | 312 MB | 459 MB | 1.21 GB |
| Encoder graphs | 10.00 | 328 MB | 596 MB | 1.35 GB |
| Encoder graphs | 30.04 | 353 MB | 935 MB | 1.69 GB |

MB and GB in this table are decimal. The process column is an NVML snapshot
after synchronization, including CUDA context/libraries and allocator pools;
it excludes other processes and is not a sampled peak. The PyTorch peak is
the maximum live allocation during the three warmed calls. Graph capture can
temporarily allocate more: the 30-second first call peaked at 534 MB live.
Immediately after model loading and releasing unused allocator blocks, the
process occupied 914 MB.

Workloads ran in the listed order, releasing unused allocator blocks between
them. Graph entries remain cached: the 10-second graph row includes the
3-second entry, and the 30-second row includes all three shapes. These figures
describe this batch-one run, not an upper bound for larger batches, longer
audio, or other cache settings. The earlier paired performance benchmark kept
both model variants loaded and must not be used as a standalone footprint.

Raw byte counts, including first-call peaks:
[`20261005T105547-memory.json`](/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T105547-memory.json).

A separate minimal CUDA Driver API process, without importing PyTorch or NeMo,
measured 522,190,848 bytes of NVML process memory after retaining the primary
context. Allocating and zeroing 249,434,904 bytes (the current model tensor size)
increased the process reading to 771,751,936 bytes. This is a raw allocation
baseline on this driver/GPU, not an inference measurement or an irreducible
CUDA context minimum. Execution additionally needs kernel code, activations,
and workspaces. Roughly 0.9–1.0 GB is therefore a plausible optimization target
for short, batch-one inference in a lean runtime; it remains unvalidated.

Raw minimal-context probe:
[`20261005T110257-cuda-memory-floor.json`](/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T110257-cuda-memory-floor.json).
