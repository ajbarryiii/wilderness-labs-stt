# Isolating encoder and decoder costs

This component benchmark complements [the earlier end-to-end comparison](COMPARISON.md).
It separates preprocessing, the acoustic encoder, and the TDT prediction
network/joint/decoding loop. Component measurements use cached inputs and
independent warm loops; their latencies and energies must not be summed and
presented as an end-to-end measurement.

The later optional expanded-INT8 encoder is documented separately in
[EXPERIMENTS4.md](EXPERIMENTS4.md), including its encoder-only screens and
fresh-process end-to-end memory/energy tradeoff. The tables here retain the
original compact third-pass configuration and timing boundaries.

## Measured components

Measured on the RTX 5090 on 2026-10-05, batch one, with the existing 400 W power
limit. The same three LibriSpeech clips used previously are 3.00, 10.00 and
30.04 seconds long. These are warm latency tests, not throughput or cold-start
measurements. Small differences of a few percent should not be treated as
universal wins; there are only three measurement windows and three audio clips.

### Encoder only

| Implementation | 3s ms | 3s J | 10s ms | 10s J | 30s ms | 30s J |
|---|---:|---:|---:|---:|---:|---:|
| Same weights, dense FP32 + graphs | 6.778 | 2.359 | 12.569 | 4.511 | 22.445 | 8.583 |
| Previous packed + graphs | 4.092 | 1.311 | 6.345 | 2.464 | 14.312 | 5.531 |
| Second-pass packed + graphs | 2.601 | 0.837 | 3.867 | 1.501 | 9.165 | 3.531 |
| Third-pass packed + graphs | 2.470 | 0.823 | 3.771 | 1.450 | 9.173 | 3.536 |
| NVIDIA FP32 eager | 7.502 | 2.537 | 13.032 | 4.687 | 23.224 | 8.866 |
| NVIDIA BF16 autocast eager | 9.488 | 2.090 | 9.842 | 2.599 | 10.869 | 4.187 |
| NVIDIA BF16 + encoder graphs | 3.561 | 1.041 | 4.185 | 1.578 | 8.323 | 3.214 |
| NVIDIA BF16 + graphs + fused decoder | 3.562 | 1.045 | 4.179 | 1.579 | 8.341 | 3.204 |
| ONNX CUDA (GPU I/O) | 4.907 | 1.510 | 7.079 | 2.352 | 11.732 | 4.558 |

ONNX native host-array encoder latencies are 4.955, 7.179 and 11.892 ms;
its GPU I/O-binding diagnostic is shown above to match the GPU-resident encoder
boundary more closely. I/O binding still uses the published ONNX graph and CUDA
provider, including its normal CPU fallback operations.

For identical ternary weights at 10 seconds, the packed-kernel replacement changes the encoder
from **12.569 ms / 4.511 J to 6.345 ms / 2.464 J**: 1.98x faster and 45.4% less
GPU energy. Second-pass encoder fusions reach **3.867 ms / 1.501 J**: 3.25x
faster and 66.7% less energy than the same-weight dense FP32 control. These are
encoder results, not end-to-end gains.
The packed kernels also change how matrix products are executed (BF16 residual
decomposition on tensor cores versus dense FP32), so this does not establish a
speedup from storage compression by itself.

The new positional-score fusion changes second-pass encoder latency by about
5.1% at 3 seconds and 2.5% at 10 seconds; the 30-second result is effectively
unchanged. The original NVIDIA BF16 graph encoder is competitive: third-pass
packed is faster on the short and medium clips, while NVIDIA BF16 is faster
and uses less encoder energy on the 30-second clip. The checkpoints differ.

### Decoder, joint network, and token loop

| Implementation | 3s ms | 3s J | 10s ms | 10s J | 30s ms | 30s J |
|---|---:|---:|---:|---:|---:|---:|
| Same weights, dense FP32 + graphs | 1.755 | 0.323 | 4.860 | 0.916 | 12.768 | 2.438 |
| Previous packed + graphs | 1.768 | 0.325 | 4.723 | 0.874 | 12.478 | 2.407 |
| Second-pass packed + graphs | 1.768 | 0.327 | 4.801 | 0.903 | 12.171 | 2.290 |
| Third-pass packed + graphs | 0.621 | 0.107 | 1.444 | 0.292 | 3.317 | 0.713 |
| NVIDIA FP32 eager | 1.775 | 0.339 | 4.901 | 0.977 | 13.261 | 2.625 |
| NVIDIA BF16 autocast eager | 1.769 | 0.337 | 4.879 | 0.949 | 12.916 | 2.482 |
| NVIDIA BF16 + encoder graphs | 1.755 | 0.337 | 4.894 | 0.967 | 12.714 | 2.484 |
| NVIDIA BF16 + graphs + fused decoder | 0.628 | 0.115 | 1.486 | 0.310 | 3.536 | 0.783 |
| ONNX CUDA | 2.391 | 0.418 | 7.679 | 1.354 | 22.095 | 3.835 |

The four same-weight rows decode **identical stored encoder tensors**, so
decoder differences cannot be attributed to packing or changed encoder outputs.
The first three remain near 4.8 ms at 10 seconds. The new third-pass decoder
reaches **1.444 ms / 0.292 J**, versus **4.801 ms / 0.903 J** in the second pass:
3.32x faster and 67.6% less GPU energy. At 30 seconds it reaches 3.317 ms /
0.713 J versus 12.171 ms / 2.290 J: 3.67x faster and 68.9% less energy.

Applying the same decoder fusions to the original NVIDIA BF16 graph model also
works: **4.894 to 1.486 ms** and **0.967 to 0.310 J** at 10 seconds. Those two
NVIDIA rows decode identical stored NVIDIA encoder tensors. This gain is a
general decoder implementation improvement and is not caused by ternarization.
ONNX native decoder rows include host control and per-token copies, unlike the
GPU-resident NeMo entry boundary.

All three packed paths matched the dense control transcripts on all three
clips. Their encoder lengths matched, and maximum absolute encoder-output
error was below 4.4e-7. The NVIDIA fused decoder matched its unfused reference
on all three clips. This component check is not a new corpus-wide quality
evaluation; the broader validation belongs to the separate optimization run.

### Preprocessing

NeMo eager preprocessing measured 0.326–0.336 ms across these clips and
implementations, with approximately 0.040, 0.045 and 0.060 GPU J for the
3-, 10- and 30-second inputs. ONNX native preprocessing measured
0.224 / 0.512 / 1.330 ms and 0.033 / 0.068 / 0.174 GPU J. The production
optimized path captures preprocessing together with the encoder; this isolated
eager frontend measurement therefore does not predict its standalone contribution
to the end-to-end graph replay.

## Model scope

The reconstructed ternary checkpoint has 617,825,926 parameters:

| Component | Parameters | Share |
|---|---:|---:|
| Encoder | 608,879,616 | 98.55% |
| Prediction network | 7,219,840 | 1.17% |
| Joint network | 1,726,470 | 0.28% |

All 264 ternary modules are inside the encoder: 603,979,776 weight elements,
99.20% of encoder parameters and 97.76% of total parameters. Encoder
normalization, biases, depthwise convolutions, and subsampling are floating
point, as are the prediction and joint networks. Small parameter count does
not imply small latency: the decoder repeatedly evaluates small operations
and maintains sequential state while generating tokens.

## Controls and timing boundaries

The three attribution controls load the **same fine-tuned ternary export**:

- `dense_ternary_graphs`: reconstruct ternary weights as FP32 tensors and run
  ordinary NeMo dense operators.
- `packed_previous_graphs`: the packed BF16x3 matrix kernels from the first pass.
- `packed_second_pass_graphs`: residual INT8x3 kernels, fused normalization,
  QKV, SiLU and attention, folded batch normalization, and captured positional
  projections from the second pass.

All three use the same encoder CUDA graph wrapper, strict FP32 input/output,
disabled TF32, and NeMo's full conditional decoder graphs. This removes the
earlier previous-packed decoder fallback as a confound. The frozen second-pass
row does not enable newer predictor, joint, decoder-control, or positional-score
experiments. Comparing dense to previous packed isolates the packed matrix
implementation; comparing previous to second-pass packed measures additional
encoder fusions and arithmetic changes, not ternarization alone.

An additional `packed_third_pass_graphs` row enables the newly validated fused
positional-score operation and FP32 predictor/joint/control kernels. It uses the
same encoder graph boundary and the same dense encoded decoder input. This row
separates the new decoder improvement from the encoder's ternary representation;
decoder weights remain floating point.

An initial previous-packed run passed the three-second workload but failed with
an illegal memory access on the next decoder invocation. This worker had no new
decoder arithmetic or control fusions. Investigation identified allocation
lifetimes in NeMo's manually captured conditional graph child streams under
PyTorch 2.11. All final NeMo component configurations use the same local
[`decoder_graph_pool`](graph_memory.py) fix, which retains those child-capture
allocations for the graph lifetime. The corrected measurements therefore include
this shared memory-lifetime fix; they are not unmodified NeMo installations.
Failed-run logs and partial measurements are preserved separately from final
measurements.
The first fix also exposed a pool-destruction assertion when replacing a decoder
graph inside another pool recording. The corrected helper retains the old pool
until compilation exits its recording scope, then releases it safely; storage
remains bounded between compilations.

The dense control saves its encoded output. All three packed paths use those
**identical encoded tensors for the decoder-only measurement**. Their own
encoder outputs are also decoded once and checked against the dense reference;
numerical encoder differences and transcript equality are retained in JSON.

For NeMo, preprocessing measures GPU waveform to GPU features. Encoder timing
starts with GPU features and ends with GPU encoded tensors. Graph wrappers
include their static-input copies and owned-output clones. Decoder timing starts
with GPU FP32 encoded tensors and ends with CPU text; it includes the prediction
network, joint network, search bookkeeping, transfer and detokenization. BF16
conversion is outside the encoder/decoder timing boundary.

Public NVIDIA FP32 and BF16-autocast encoder rows use eager execution; NVIDIA
BF16-plus-graphs stores encoder tensors in BF16 and uses the same local encoder
graph wrapper. These rows have different weights from the fine-tuned ternary
export and are implementation comparisons, not controlled quantization effects.
Quality differences are documented in the earlier comparison.

ONNX-ASR's native API exchanges host NumPy arrays, including between its encoder
and decoder. Its native component rows therefore include transfers. A separate
encoder diagnostic uses ONNX Runtime I/O binding with GPU inputs and outputs.
The native ONNX decoder timing includes its host-controlled token loop and
per-token transfers; it is not a GPU-resident counterpart to NeMo's decoder.

## Reproduction

From the repository root:

```sh
finetune/parakeet-ternary/python -m inference.compare_v2_components \
  --variant all \
  --out /mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/new-components \
  --seconds 2 --repeats 3
```

The serial driver starts a fresh process per implementation, waits for its exit,
and stops on failure. It runs the dense same-weight control first to generate
decoder reference tensors. Individual workers accept `--variant` and packed
workers accept `--decoder-reference` pointing at the dense run's `decoder_inputs`
directory. The NVIDIA fused-decoder row uses `nvidia_decoder_inputs`, written by
the ordinary NVIDIA BF16-graph row. Downloaded models and comparison dependencies are the pinned versions
described in [COMPARISON.md](COMPARISON.md).

Each component has three two-second measurement windows, each preceded by a
one-second warmup. Component order reverses on alternate rounds. Reported values
are the median of the three window medians (latency) or per-call energies. Raw
iteration times, power/energy telemetry, model/source hashes, input records,
transcripts, and allocator/process-memory snapshots are retained in worker JSON.
GPU board energy includes idle/background board consumption but excludes CPU
energy; it is neither whole-system energy nor an additive per-layer allocation.
Memory snapshots here describe the component harness and do not replace the
isolated end-to-end deployment VRAM results.

The measured environment is PyTorch 2.11.0+cu128, NeMo 3.0.0, ONNX Runtime GPU
1.24.4, onnx-asr 0.12.0, and driver 595.71.05 (SM120). Original NVIDIA model
revision: `ae9ad07059c7c739ffaf932226a8fe64ae2620b0`. Published ONNX export
revision: `0bbb45a3365852604aef28b538a8f066f4ccaa85`. Ternary export SHA256:
`a287e97719c451b785be2cd01ecc861fcaa010ebaa4ff2841783ec78fcd61503`.
Each worker records its runtime-wrapper and inference-source SHA256 hashes,
exact input manifest records, GPU metadata and provider options. The experiment
uses the installed model revisions; no checkpoint was modified.

Raw evidence: `/mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/components1/`.
There are nine successful final worker JSON files. Files prefixed `failed-`
retain the two unsuccessful allocator/capture attempts;
`original-pool-dense_ternary_graphs.json` retains the initial dense probe and is
not included in the final tables.
