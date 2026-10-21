# Fourth pass: faster ternary encoder with more resident storage

The trained weights and decoder are unchanged. The new optional
`--optimized --encoder-storage expanded` path decodes ternary codes once into
column-major INT8 weights, tunes the integer matrix tiles for this RTX 5090,
and fuses Conformer gate/mask/padding, activation/quantization, and residual
operations. It retains the original two-bit weights for restoration and adds
603,979,776 bytes of cached weights after warmup. The default remains the
previous compact runtime.

The configured GPU power limit was **400 W** throughout. Benchmark and quality
workers check the NVML power limit and refuse to run above 400 W. No power or
clock setting was changed. Joules below measure GPU board energy, not total
system energy or outlet draw.

## Weight audit and binary opportunities

An exact CPU audit decoded all 603,979,776 ternary weights in 264 encoder
modules. Zeros account for 34.1695%, positive values for 32.9026%, and negative
values for 32.9279%. There are **no one-sided layers and no zero-free rows**.
Only two rows have nonzero values of a single sign, in the positional
projections of encoder layers 6 and 12. Those projections are already cached
outside warm graph execution. Seven one-sided columns occur in the second
feed-forward down projections of layers 3 and 4.

A sign-only binary model would erase meaningful zeros. A sign-plus-presence
representation still needs two bits. Only 42.3% of aligned groups of four have
at least two zeros, so the current weights do not satisfy global 2:4 structured
sparsity. This was an eligibility audit; no sparse tensor-core kernel or
pruning experiment was implemented. The audit also counts homogeneous and
one-sided blocks of 4, 16, 64 and 128 in every matrix.

Raw audit: `/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T142554-weight-audit/audit.json`.

## Experiments actually run

Screen timings are encoder-only, batch one, GPU features to GPU encoder output,
with preprocessing and decoder excluded. They use one real clip per duration,
one three-second measurement window after warmup. These are candidate screens,
not isolated process VRAM measurements or a workload-wide performance guarantee.

| Experiment | Observed result |
|---|---|
| Per-kernel and per-layer profiling | The packed integer dot kernels cost about 1.915 ms of the 3.73 ms 10-second encoder. The 24 Conformer blocks have similar costs; there is no unusually expensive block to target alone. |
| Exact row-major INT8 expansion, existing Triton dot | About 3.55 ms at 10 seconds. Some decoding work removed, modest gain. |
| Row-major `torch._int_mm` with three residual inputs stacked | About 10.76 ms at 10 seconds; rejected. |
| Column-major `torch._int_mm` | About 3.41 ms at 10 seconds and 8.17 ms at 30 seconds. Profiling shows PyTorch CUTLASS SM80 kernels; this is not a cuBLAS result. |
| Column-major INT8 with the fused Triton dot | About 3.19 ms at 10 seconds. Bit-identical encoder outputs in this screen before tile changes. |
| FP16x2 and BF16x3 activation residuals with expanded float weights | Slower than the integer winner, with about 1.208 GB of expanded weights. FP16x2 took 10.93 ms at 30 seconds; BF16x3 took 12.75 ms. |
| FP8x4 activation residuals and exact FP8 ternary weights | About 4.26 ms at 10 seconds and 11.30 ms at 30 seconds; rejected. An initial INT8-to-FP8 expansion cast did not compile; routing through FP32 fixed it. |
| FP8x3 activation residuals | Failed the short-clip numerical gate: one element differed by 3.308e-6. No successful timing result claimed. |
| Custom stacked INT8 residual GEMM, four tile choices | About 3.52–4.97 ms at 10 seconds; slower than the fused column-major kernel. |
| Selective expansion: down, up, QKV, gate, or other projections | All helped, but full expansion was fastest. Down-only used 201 MB extra and took 3.56 ms; QKV-only used 75.5 MB and took 3.63 ms. |
| Column-major tile/split-K sweep | Swept up to 25 configurations across five matrix shapes and three frame counts, with quantization and epilogue included. Chosen tiles improved the full 30-second encoder from 8.42 to 7.63 ms. |
| Fuse GLU, padding mask and depthwise-convolution padding; fuse SiLU into quantization | At 10 seconds, 3.14 to 3.02 ms after tile tuning. |
| Fuse first residual scaling/add and final residual/normalization | At 10 seconds, 3.14 to 3.08 ms alone, or 2.96 ms with convolution fusions. |

Final combined encoder screen:

| Audio | Previous encoder ms | New encoder ms | Previous J | New J |
|---|---:|---:|---:|---:|
| 3 s | 2.447 | 2.026 | 0.783 | 0.645 |
| 10 s | 3.737 | 2.956 | 1.441 | 1.141 |
| 30.04 s | 9.103 | 7.343 | 3.515 | 2.822 |

All three passed the encoder comparison with `atol=3e-6, rtol=3e-5`.
The maximum absolute difference was 3.576e-7. Expanded weights are exact;
changed split-K and fused arithmetic can change floating-point rounding.
The screen's per-variant live allocation counters include prior captures and
must not be presented as fresh-process deployment memory.

Raw screen: `/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T144825-sprint4/`.
Tile sweep: `kernel-runs/20261005T144327-expanded-tiles/` under the same artifact root.
All initial, rejected and failed screen logs are retained as `sprint4*.log`.

## Fresh-process end-to-end measurements

Batch one, CPU FP32 waveform to CPU text, including preprocessing and transfers.
Each implementation runs alone in a fresh process. Each clip has three
five-second measurement windows with two seconds of warmup per window; results
are medians across windows. Process VRAM is the maximum sampled during warm
windows. The four-entry pipeline graph cache accumulates shapes in the order
3, 10, 30.04 seconds. Compilation, loading, first capture and file I/O are
excluded. MB and GB are decimal; GPU energy excludes host-system energy.

| Audio | Previous ms | Expanded ms | Previous J | Expanded J | Previous process GB | Expanded process GB |
|---|---:|---:|---:|---:|---:|---:|
| 3 s | 3.159 | 2.734 | 0.912 | 0.765 | 1.122 | 1.741 |
| 10 s | 5.269 | 4.462 | 1.858 | 1.572 | 1.271 | 1.887 |
| 30.04 s | 12.210 | 10.564 | 4.740 | 3.989 | 1.617 | 2.233 |

This is **13.5–15.3% lower end-to-end latency** and **15.4–16.1% less GPU
energy** on these clips, for about **617–619 MB more process VRAM**. The
decoder is unchanged in this pass; encoder-only improvements are larger.
The measured warm average power is about 279–376 W for the expanded path,
with the configured limit remaining 400 W. A lower joule count does not imply
a lower configured cap.

At 10 seconds, peak live Torch allocations increase from 347.9 to 962.3 MB,
and reservation from 522.2 to 1,142.9 MB. These overlap with process VRAM and
must not be added to it. The exact expanded weight cache accounts for 604 MB;
changed graph temporaries and allocator reservations account for the rest.

These are sequential runs of one real clip per duration. They measure warmed
repeated shapes, not cold-start performance or all possible speech workloads.
Raw JSON, source hashes, export hash, memory samples and energy telemetry:
`/mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/run4/`.

### Original NVIDIA v2 with the same decoder optimizations

A third fresh process ran the original NVIDIA checkpoint with its dense BF16
encoder, the same pipeline graphs, graph-memory fix and FP32 decoder fusions.
It uses the same clips, three five-second windows, and 400 W configured limit.

| Audio | Expanded ternary ms | NVIDIA BF16 ms | Expanded J | NVIDIA J | Expanded process GB | NVIDIA process GB |
|---|---:|---:|---:|---:|---:|---:|
| 3 s | 2.734 | 4.254 | 0.765 | 1.154 | 1.741 | 2.110 |
| 10 s | 4.462 | 5.810 | 1.572 | 1.895 | 1.887 | 2.183 |
| 30.04 s | 10.564 | 11.558 | 3.989 | 4.482 | 2.233 | 2.349 |

At 10 seconds, expanded ternary uses 23.2% less time, 17.1% less GPU energy,
and 13.5% less process VRAM than this dense reference. It wins all three tested
durations, including the long clip where the compact third-pass runtime lost.
This does not establish a universal lead. The NVIDIA and ternary checkpoints
have different learned weights and recognition quality; see the earlier quality
comparison. The shared decoder optimizations are local additions, not stock
NeMo defaults. Public NeMo and ONNX implementation comparisons remain in
[COMPARISON.md](COMPARISON.md) and [COMPONENTS.md](COMPONENTS.md).

## Recognition validation

The integrated public option passed a 2,048-utterance comparison against the
third-pass compact runtime: 256 records from each of the eight existing test
manifests, totaling 14,156.25 seconds of audio. Encoder batches contain up to
eight utterances; decoding is compared at batch one on each encoder's output.

- Text, token IDs and token timestamps agree on **2,048/2,048** utterances.
- Both paths have 2,447 word edits over 36,699 scored words, with identical
  counts within every test set (6.668% pooled WER).
- Maximum encoder absolute difference: `1.3411e-6`; maximum relative L2
  difference: `5.8236e-7`. All comparisons pass `atol=3e-6, rtol=3e-5`.
- Maximum accumulated decoder score difference: `0.001465`.

This is observed preservation of the existing ternary checkpoint on a subset,
not proof of universal equivalence or original NVIDIA checkpoint quality.
The different checkpoints' quality comparison is in [EXPERIMENTS3.md](EXPERIMENTS3.md).

Raw validation: `/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T145045-quality4/quality.json`.

All **12 regression tests** passed. New checks cover exact expanded weights
with all-zero, positive and negative rows; odd matrix dimensions; convolution
input layouts; gate/mask/padding with unequal lengths; fused residual
normalization; silence/noise; graph-cache limits and reuse after allocator
churn; and restoration of forward methods, state-dict keys and convolution
weights on disable. Existing encoder and decoder regression checks also passed.
Log: `/mnt/hd/wilderness-labs-stt/parakeet-ternary/tests4-final.log`.

The separate fresh-process, batch-one CPU-waveform-to-text check also matches
**256/256 transcripts** (32 records per test set): 263 word edits over 4,721
scored words in each path. This exercises the same public API and runtime used
for the final performance measurements. Its raw records are in the two packed
worker JSON files under `comparison/run4/`.
The `--optimized --encoder-storage expanded` CLI smoke test also passed and
matched the 10-second benchmark transcript (`cli4.log` under the artifact root).

## Reproduction

```sh
P=finetune/parakeet-ternary
export PARAKEET_KERNEL_EXPORT=/mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/export
$P/python -m unittest inference.test_encoder4 inference.test_experiments inference.test_decoder
$P/python -m inference.quality4 --utterances-per-set 256 --batch-size 8
$P/python -m inference.sprint4 --variants baseline triton_col_tuned+conv+residual --durations 3 10 30
$P/python -m inference.compare_v2 --variant packed_optimized --out /mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/run4 --quality-per-set 32
$P/python -m inference.compare_v2 --variant packed_expanded --out /mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/run4 --quality-per-set 32
$P/python -m inference.compare_v2 --variant nemo_bf16_graphs_fused_decoder --out /mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/run4
$P/python -m inference --export "$PARAKEET_KERNEL_EXPORT" --optimized --encoder-storage expanded clip.wav
```

Run GPU jobs serially. Use `enable_optimizations(model, encoder_storage="expanded")`
from Python, and `disable_optimizations(model)` to release graphs and restore
the original modules before moving, editing, exporting or training the model.
The cache fills on warmup and lives as long as the optimized model. All weights,
compiler caches and raw measurements remain on the mounted data disk.
