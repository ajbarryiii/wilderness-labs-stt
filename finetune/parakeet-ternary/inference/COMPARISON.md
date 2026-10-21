# Parakeet v2: isolated VRAM, latency, energy, and recognition comparison

These are the **second-pass** measurements. The subsequent decoder optimization
and graph-memory correction are documented in [EXPERIMENTS3.md](EXPERIMENTS3.md).
See [COMPONENTS.md](COMPONENTS.md) for the newer isolated component comparison.
The historical numbers below have not been relabeled as measurements of the
current optimized runtime.

Measured on this RTX 5090 on 2026-10-05. The optimized ternary path retains a
large VRAM advantage over dense v2. Its speed advantage becomes modest when
compared with a BF16 dense encoder using the same frontend/encoder graph wrapper;
the two are effectively tied on the tested 30-second clip. Optimization slightly
increases VRAM relative to the previous packed path.

The external implementations are NVIDIA's
[NeMo Parakeet-TDT-0.6B-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
and [onnx-asr](https://github.com/istupakov/onnx-asr) using its
[published v2 ONNX export](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx).
The graph-enhanced BF16 row is a local optimization of the original NVIDIA
checkpoint, included to provide a stronger dense baseline. TensorRT, TensorRT-RTX,
Sherpa, and Apple runtimes were not benchmarked; this is not a claim against every
available implementation.

## Measured results

### 3-second audio

| Variant | Latency (ms) | GPU joules/call | Process VRAM (GB) |
|---|---:|---:|---:|
| Previous packed | 6.57 | 1.72 | 1.158 |
| Optimized packed | 4.46 | 1.16 | 1.208 |
| NeMo FP32 | 10.37 | 2.98 | 3.408 |
| NeMo BF16 autocast | 12.31 | 2.52 | 3.408 |
| NeMo BF16 + graphs | 5.41 | 1.36 | 2.196 |
| ONNX CUDA | 7.86 | 1.97 | 4.127 |

### 10-second audio

| Variant | Latency (ms) | GPU joules/call | Process VRAM (GB) |
|---|---:|---:|---:|
| Previous packed | 12.49 | 3.56 | 1.300 |
| Optimized packed | 9.01 | 2.51 | 1.317 |
| NeMo FP32 | 18.74 | 5.75 | 3.485 |
| NeMo BF16 autocast | 15.92 | 3.69 | 3.431 |
| NeMo BF16 + graphs | 9.46 | 2.58 | 2.227 |
| ONNX CUDA | 15.77 | 3.81 | 4.136 |

### 30-second audio

| Variant | Latency (ms) | GPU joules/call | Process VRAM (GB) |
|---|---:|---:|---:|
| Previous packed | 28.61 | 9.10 | 1.692 |
| Optimized packed | 21.61 | 6.48 | 1.797 |
| NeMo FP32 | 36.30 | 13.17 | 3.674 |
| NeMo BF16 autocast | 25.05 | 7.41 | 3.601 |
| NeMo BF16 + graphs | 21.53 | 6.35 | 2.527 |
| ONNX CUDA | 36.02 | 9.08 | 4.186 |

At 10 seconds, optimized packed uses **40.9% less process VRAM** than
BF16 + graphs, **61.6% less** than BF16 autocast, and **68.2% less** than
ONNX CUDA. Its latency advantage over BF16 + graphs is only about 4.7%, and its
measured energy advantage about 2.7%; at 30 seconds the dense BF16 graph path is
slightly faster and lower-energy. These small margins warrant treating the two
as competitive rather than claiming a universal ternary speed/energy win.

Relative to previous packed, optimization adds 50 MB at 3 seconds, 17 MB at
10 seconds, and 105 MB at 30 seconds. Warm process memory therefore rises
4.3%, 1.3%, and 6.2%, respectively, while transcription becomes faster.

### What occupies VRAM

| Variant | Loaded live Torch GB | Loaded process GB | 10s live peak GB | 10s reserved GB |
|---|---:|---:|---:|---:|
| Previous packed | 0.251 | 0.935 | 0.309 | 0.547 |
| Optimized packed | 0.272 | 0.960 | 0.365 | 0.566 |
| NeMo FP32 | 2.515 | 3.198 | 2.622 | 2.745 |
| NeMo BF16 autocast | 2.515 | 3.198 | 2.588 | 2.691 |
| NeMo BF16 + graphs | 1.276 | 1.959 | 1.337 | 1.476 |
| ONNX CUDA | — | 4.066 | — | — |

The optimized model has about **272 MB of live CUDA allocations immediately
after loading**, compared with **1.276 GB** for the dense BF16 encoder variant.
Registered parameters/buffers alone report about 250 MB for optimized packed;
that misses closure-held fused copies and other CUDA allocations. Use allocator
and process measurements when budgeting memory.

At 30 seconds optimized packed peaks at **488 MB live in PyTorch**, reserves
**1.042 GB**, and occupies **1.797 GB for the process**. Reserved includes live
allocations; these columns must not be added together. Graph pools and retained
workspaces explain much of the difference. The earlier minimal CUDA-context
measurement of approximately 522 MB is a driver/context baseline for this
machine, not an additional model tensor allocation or a universal allocator tax.

First-use memory is retained separately in each workload's JSON. Sampled peaks
over load and all three performance workloads are approximately 1.69 GB previous
packed, 1.80 GB optimized, 3.68 GB FP32, 3.60 GB BF16 autocast, 2.53 GB BF16 +
graphs, and 4.19 GB ONNX. The first ONNX recognition took 29.7 seconds; warm
numbers exclude that initialization. Compiler caches already existed for the
ternary kernels, so first-call times here are not clean-install cold-start tests.

## Recognition check

Same deterministic 256 clips: 32 from each of the eight existing test manifests,
1,840.37 seconds of audio and 4,721 scored reference words. WER is pooled by word
count with the repository's existing normalization, not averaged across datasets.
This is a small subset check, not full-corpus WER or a statistical equivalence test.

| Variant | Word edits | WER | Exact transcripts vs NVIDIA FP32 |
|---|---:|---:|---:|
| Previous packed | 263 | 5.57% | 138/256 |
| Optimized packed | 263 | 5.57% | 138/256 |
| NeMo FP32 | 242 | 5.13% | 256/256 |
| NeMo BF16 autocast | 239 | 5.06% | 250/256 |
| NeMo BF16 + graphs | 240 | 5.08% | 251/256 |
| ONNX CUDA | 245 | 5.19% | 242/256 |

**Previous and optimized packed match each other on all 256 transcripts**,
including punctuation, in this batch-one run. The earlier 2,048-clip validation
also compared those same ternary weights across execution paths.

The original NVIDIA model and the fine-tuned ternary export have different
weights. The ternary checkpoint shows 21 more word errors than NVIDIA FP32 on
this subset (0.445 percentage points higher WER). Its speed and memory savings
must be considered alongside that observed quality difference. Equal transcripts
between the two packed runtimes demonstrate preservation of the ternary
checkpoint's behavior, not equivalence to original NVIDIA v2.

## Method

- One fresh process per variant; only that model resides on the GPU. Existing
  exclusive GPU lock and foreign-compute-process checks are retained.
- Batch one, identical real LibriSpeech-clean clips of 3.00, 10.00, and 30.04 seconds.
  Input is a cached CPU FP32 waveform; preprocessing, CPU/GPU transfers, encoder,
  greedy TDT decoding, and CPU text output are included. File I/O is excluded.
- Shapes are visited in that order and retained by graph caches (maximum four).
  Consequently the 10-second row includes two shapes and the 30-second row three.
  Memory results are specific to this cache policy and workload.
- Three five-second measurement windows per shape, each preceded by two seconds
  of warmup. Tables use the median of the three window medians / joules per call.
  Model load, first-use compilation, and capture are excluded from hot timings.
  Processes run sequentially, not in alternating paired order. Small differences
  should not be interpreted as a robust performance or energy lead.
- GPU-board joules use NVML's cumulative energy counter. Host energy is not
  included, and idle/display energy is not subtracted. Existing 400 W limit and
  desktop remain unchanged; no power or clock settings were changed.
- Process VRAM is sampled with NVML every 10 ms, excluding other processes and
  desktop allocations. These are sampled peaks, not guaranteed instantaneous
  peaks. PyTorch allocator peaks are recorded separately. All GB are decimal.
- CUDA context/library memory, graph pools, retained workspace, and unused
  allocator reservations all count toward process VRAM. Tensor bytes alone do
  not describe deployment memory. ONNX allocations are not visible to PyTorch.
- The main memory figures cover performance, before the separate quality pass
  encounters additional shapes. Complete raw samples also retain that later pass.

## Implementations and precision

| Variant | Implementation |
|---|---|
| Previous packed | Same ternary export; BF16x3 packed kernels, encoder graphs, old `no_while_loops` decoder graph mode. |
| Optimized packed | Same ternary export; residual INT8x3 kernels and fusions, frontend/encoder graphs, full conditional decoder graphs. |
| NeMo FP32 | Original NVIDIA checkpoint, strict FP32 eager frontend/encoder, full conditional decoder graphs. TF32 disabled. |
| NeMo BF16 autocast | Original NVIDIA checkpoint still stored in FP32, BF16 encoder autocast, FP32 decoder, full conditional decoder graphs. |
| NeMo BF16 + graphs | Original encoder weights and buffers explicitly cast to BF16, FP32 frontend/decoder; local frontend/encoder graph wrapper. |
| ONNX CUDA | Published FP32 v2 export, onnx-asr 0.12.0, ONNX Runtime 1.24.4 CUDA EP with default allocator/provider options. |

The ONNX encoder and decoder sessions both verified CUDA as their first active
provider. ORT also assigns some operations to its normal CPU fallback. The
benchmark does not claim every ONNX node executes on the GPU. Provider options
are saved in the worker JSON, including the default TF32 policy. The
[CUDA 12-compatible ORT build](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
was selected to match the installed CUDA 12.8 libraries; the initially downloaded
ORT 1.30 wheel targets CUDA 13 and was replaced before benchmarking.

PyTorch is 2.11.0+cu128, NeMo 3.0.0, Triton 3.6.0. Driver 595.71.05, SM120.
ONNX model revision: `0bbb45a3365852604aef28b538a8f066f4ccaa85`.
Original NVIDIA revision: `ae9ad07059c7c739ffaf932226a8fe64ae2620b0`.
Ternary export SHA256:
`a287e97719c451b785be2cd01ecc861fcaa010ebaa4ff2841783ec78fcd61503`.

## Reproduction and raw evidence

The worker and aggregator are [compare_v2.py](compare_v2.py) and
[compare_v2_report.py](compare_v2_report.py). From the repository root:

```sh
P=finetune/parakeet-ternary
OUT=/mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/new-run
for variant in packed_previous packed_optimized nemo_fp32 nemo_bf16 nemo_bf16_graphs onnx_cuda; do
  "$P/python" -m inference.compare_v2 --variant "$variant" --out "$OUT" --quality-per-set 32
done
"$P/python" -m inference.compare_v2_report "$OUT"
```

This machine has isolated comparison dependencies under
`/mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/deps`:
`onnx-asr==0.12.0`, `onnxruntime-gpu==1.24.4`. Other dependencies come from the
existing pinned project wrapper. These packages did not replace the training
runtime. The pinned ONNX files are under `comparison/onnx-v2`, with sizes and
SHA256 hashes in `comparison/onnx-files.json`. A different machine must provision
these packages and download that revision to the same configured data-disk
locations before running the worker. The mount check is mandatory.

Raw data: `/mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/run1/`.
Each worker saves source hashes, model provenance, versions, GPU metadata, raw
iteration timings, power/energy telemetry, memory samples, input records, and
quality hypotheses/scores. Worker logs and dependency-install logs are in the
parent `comparison/` directory. No weights or datasets are stored in this repo.

Earlier [EXPERIMENTS.md](EXPERIMENTS.md) performance comparisons used GPU-resident
inputs and multiple models resident simultaneously. Their latency boundary and
memory layout differ from this isolated comparison; use these new measurements
for cross-runtime process VRAM comparisons.
