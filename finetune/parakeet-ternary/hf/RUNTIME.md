# Kea-T runtime results (custom CUDA runtime on GitHub)

Detail for the [Kea-T model card](README.md).

The [custom CUDA runtime](https://github.com/ajbarryiii/wilderness-labs-stt/tree/main/finetune/parakeet-ternary/inference)
runs `export.safetensors` from its packed codes, without rebuilding dense floating-point
encoder weights on the GPU.
The runtime is MIT-licensed code in the GitHub repository and is not part of this
download. `load_ternary.py` here still rebuilds a dense FP32 NeMo model (see
[Use](README.md#use)).

## Results

Warm batch-one transcription of one LibriSpeech test-clean clip per length on one RTX 5090 at
a 400 W power limit:

| Clip | Configuration | Latency (ms) | Process VRAM (GB) | GPU energy (J) | Avg GPU power (W) |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 s | Ternary expanded — optimized by Wilderness Labs | 2.73 | 1.74 | 0.76 | 279 |
| 3 s | Ternary compact — optimized by Wilderness Labs | 3.16 | 1.12 | 0.91 | 288 |
| 3 s | Original BF16 — optimized by Wilderness Labs | 4.25 | 2.11 | 1.15 | 271 |
| 3 s | ONNX ASR / ORT CUDA — off the shelf | 7.87 | 4.13 | 1.97 | 249 |
| 10 s | Ternary expanded — optimized by Wilderness Labs | 4.46 | 1.89 | 1.57 | 351 |
| 10 s | Ternary compact — optimized by Wilderness Labs | 5.27 | 1.27 | 1.86 | 352 |
| 10 s | Original BF16 — optimized by Wilderness Labs | 5.81 | 2.18 | 1.90 | 326 |
| 10 s | ONNX ASR / ORT CUDA — off the shelf | 15.74 | 4.14 | 3.75 | 237 |
| 30.04 s | Ternary expanded — optimized by Wilderness Labs | 10.56 | 2.23 | 3.99 | 376 |
| 30.04 s | Ternary compact — optimized by Wilderness Labs | 12.21 | 1.62 | 4.74 | 386 |
| 30.04 s | Original BF16 — optimized by Wilderness Labs | 11.56 | 2.35 | 4.48 | 386 |
| 30.04 s | ONNX ASR / ORT CUDA — off the shelf | 36.38 | 4.19 | 9.00 | 248 |

## Configurations

- **Ternary expanded** (`--optimized --encoder-storage expanded`) keeps the packed
  codes and scales plus a cached exact INT8 copy of the codes (603,979,776 bytes,
  about 604 MB). Against our optimized BF16 build of the original at 10 s, it has
  23.2% lower latency, 13.5% lower process VRAM and 17.1% less GPU energy. It is
  also lower on all three at the 3 s and 30.04 s clips.
- **Ternary compact** (`--optimized`) keeps the ternary matrices as packed 2-bit
  codes with per-row FP32 scales and unpacks them inside the kernels. It used the
  least process VRAM of the four configurations at every length tested: 41.8% less
  than BF16 at 10 s, with 9.3% lower latency. Its 10 s energy is within 2% of BF16,
  and at 30.04 s its latency and energy are higher than BF16's.
- **Original BF16 — optimized by Wilderness Labs** is NVIDIA's original checkpoint
  with a BF16 encoder, plus our own CUDA graphs, fused FP32 decoder and graph
  allocation fix. It is not stock NeMo. The two checkpoints produce different
  transcripts and therefore different decoder work, so this row compares complete
  runtimes, not the effect of ternarization alone.
- **ONNX ASR / ORT CUDA — off the shelf** is the unmodified
  [onnx-asr](https://github.com/istupakov/onnx-asr) 0.12.0 package with
  onnxruntime-gpu 1.24.4. It runs the published FP32 export
  [istupakov/parakeet-tdt-0.6b-v2-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx)
  at a pinned revision, with the CUDA execution provider and default options. It is
  one common way to run this model without NeMo, and its weights are FP32 rather
  than BF16. At 10 s, our BF16 build also has 63.1% lower latency than this row; the
  two configurations differ in precision and runtime. This comparison is not a survey
  of optimized engines.

Both ternary modes use the weights in this repository and keep the non-ternary
tensors in floating point. Both run the ternary matrices on INT8 Tensor Cores, with
each activation split into three residual INT8 components, int32 accumulation and
FP32 outputs, and both run a fused FP32 prediction/joint decoder, with NeMo's greedy
TDT loop in conditional CUDA graphs.

## Method

- Latency covers a CPU waveform in to CPU text out, including feature extraction and
  host-device transfers, and excludes file I/O, loading, compilation and first graph
  capture.
- Each configuration ran in a fresh process, with one clip per length. Values are
  medians of three 5 s windows.
- Process VRAM is the warm process's sampled peak, in decimal GB. One process runs the
  lengths in order (3, 10, 30.04 s) and its graph cache holds up to four shapes, so process
  VRAM at the longer lengths includes graphs already captured for the shorter ones.
- GPU energy is NVML's board counter and excludes host power. At 10 s, compared with our BF16
  build, the ternary rows draw as much power or more (expanded 351 W, compact 352 W, BF16
  326 W) and use less energy because they finish sooner.
- Power: the configured 400 W limit (board default 600 W) was not changed for any row.

## Agreement between the two modes

Compact and expanded modes produce identical text, tokens and timestamps on 2,048
utterances from eight test sets. That checks agreement between the two modes, not
accuracy against the original. Recognition accuracy for this model is the
full-corpus WER in [EVALUATION.md](EVALUATION.md).

## Requirements

The runtime needs the GitHub
repository's pinned environment (NeMo 3.0, PyTorch 2.11 with CUDA 12.8, Triton 3.6,
NVRTC 12.9); it is not a pip package. The
[runtime guide](https://github.com/ajbarryiii/wilderness-labs-stt/blob/main/finetune/parakeet-ternary/inference/README.md)
has usage and reproduction steps, and
[rtx5090.json](https://github.com/ajbarryiii/wilderness-labs-stt/blob/main/finetune/parakeet-ternary/inference/results/rtx5090.json)
has the evidence, including p95 latency, every window and the GPU clocks.

## Limitations

The runtime results apply only to the custom CUDA runtime on GitHub. They were
measured on one RTX 5090 at a 400 W limit, warm, at batch one, with one clip per
length. Other GPUs, batch sizes, cold starts and host energy were not measured.
