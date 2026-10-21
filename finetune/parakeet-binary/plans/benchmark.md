# Plan: speed and energy benchmark of the binary model MB1

**Status: 2026-10-20: draft; not reviewed. Runs after MB1 test scoring
(plans/test-scoring.md), one GPU job at a time.**

## Question

How fast and how energy-efficient is MB1 on the RTX 5090, measured the way M1
was (finetune/parakeet-ternary/inference: README.md, COMPARISON.md,
EXPERIMENTS4.md, CUDA_IMPROVEMENTS.md), against M1 and the dense NVIDIA
baselines in the same session?

## What to expect, stated before measuring

MB1's export stores 1 bit per weight (bit 1 = +1, bit 0 = -1, per-row FP32
scale alpha). Mapping to {0, 1} for inference is exact and cheap: with
b in {0, 1} and w = 2b - 1,

    y_j = alpha_j * sum_i (2 b_ij - 1) x_i = alpha_j * (2 * sum_i b_ij x_i - sum_i x_i)

so a {0, 1} kernel needs one extra row sum of x per input vector, shared by all
output channels. M1's GPU runtimes do not exploit ternary zeros: the compact path
decodes 2-bit fields to +-1/0 and multiplies in BF16x3, and the fastest paths
(expanded INT8 weights, hybrid CUDA/Triton `ptx_row_best`) cache codes as INT8.
Binary codes run through the same arithmetic with identical shapes, so **GPU
latency and energy are expected to match M1's within noise**; the binary gains
are storage (105 MB vs 181 MB on disk; 75.5 MB vs 151 MB of packed codes) and,
only with a native 1-bit compact kernel, halved weight traffic on the compact
path. Activations stay floating point, so XNOR-popcount does not apply on GPU.

## Stage 1 (this plan): MB1 on M1's runtime, no kernel changes

1. **Derived export** (`binary_bench_convert.py`, new): read MB1's export
   (pinned SHA-256 `9873d2f1...`), unpack each module's codes with
   `binary_export.unpack_codes`, repack them with the ternary
   `export.pack_codes` (+1 -> field 01, -1 -> field 10), copy every other tensor
   byte for byte, and write a `parakeet-ternary-v1`-layout directory to
   `/mnt/hd/wilderness-labs-stt/parakeet-binary/derived/MB1-ternary-layout/`
   whose manifest records `derived_from` (MB1 manifest and weight hashes) and the
   conversion script's hash. Contract: for every module, ternary `unpack_codes`
   of the derived codes equals MB1's codes exactly, scales are bit-identical, the
   code histogram has zero 0-codes, and every non-code tensor is byte-identical.
2. **Correctness on the runtime**: the derived export through `load_packed` with
   `--optimized --encoder-storage expanded` and the hybrid `ptx_row_best` path
   transcribes the 24-utterance multi-corpus set and a quality sample from dev
   (not test), compared with MB1's FP32 NeMo reconstruction
   (`binary_export.load_export`): transcript and token match rate, WER of each.
   M1's runtime matched its own control on all 2,048 sampled transcripts; the
   same check is reported for MB1, with mismatches listed, not hidden.
3. **Benchmarks** (wrapper `binary_bench.py` that sets `compare_v2.EXPORT` and
   the CUDA experiment's export to the derived directory; parakeet-ternary files
   unchanged): the same variants, clips (3, 10, 30.04 s LibriSpeech; 24-utterance
   multi-corpus set), windows, warmup, 400 W cap check, NVML energy and process
   VRAM sampling as M1's recorded runs. M1 is re-measured in the same session as a
   control, in order M1, MB1, MB1, M1 per variant (fresh process each), so drift
   between October 5 and today is not mistaken for a binary effect. Dense NVIDIA
   BF16 + graphs and FP32 are re-run once each as reference rows.
4. **Report** (`results/BENCHMARK.md`): latency, GPU joules per transcription,
   average power and process VRAM per variant and duration, MB1 vs M1 paired
   changes with counts, disk and weight sizes, and dev WER beside each model.

Outputs go to `/mnt/hd/wilderness-labs-stt/parakeet-binary/benchmark/<run>/`.

## Proposed later stages (not approved; need the user's decision)

- **Stage 2, native 1-bit GPU compact kernel**: decode 32 weights per 32-bit word
  with the {0, 1} identity above; measures whether halved weight bytes speed up
  the compact (non-expanded) path. Kernel development with contract tests
  against a dense reference; a few days.
- **Stage 3, CPU**: the bit-plane CPU benchmark (`cpu-bitplanes/`, Ryzen
  9950X3D) with one plane instead of two (`popcnt(a XOR n)`). On M1 every
  bit-plane variant lost to INT8 VNNI; a single plane removes about half that
  work. Latency only unless RAPL access is granted.
- **Stage 4, Apple**: a uint1 + LUT {-1, +1} Core ML arm in `ios/`, run on the
  Mac over SSH (the iOS work so far measured mp2 on the M1 Pro Mac; no iPhone
  run has happened yet).

## Review

Codex gpt-6-astra xhigh reviews this plan and the Stage 1 code before any
benchmark run, iterating until CLEAN.
