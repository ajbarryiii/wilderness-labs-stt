# Parakeet-TDT 0.6B v2 on iPhone 15 Pro: inference pipeline and benchmarks

Status: **under review, S0 approved by the user, nothing run** (2026-10-02).

## Goal and scope

Build the lowest-latency, lowest-energy inference pipeline for the Parakeet-TDT
0.6B v2 architecture (English; the ternary QAT model from `../DESIGN.md`) on an
iPhone 15 Pro (A17 Pro, 8 GB, iOS 26), and the benchmarks that measure it on the
device. The workload is one push-to-talk utterance of 2-15 s.

- The architecture is fixed; the weights are not. Benchmarks use seeded random
  weights with the exact architecture and the ternary structure of the QAT model,
  so they can run before training finishes. They measure execution, not
  recognition. The trained weights enter only in stage S4.
- The deployment target may be iOS 26 or later; the pilot controls hardware and
  software.
- Out of scope here: wiring into tccc-bot (later step), recognition accuracy of
  any random model, streaming decoding, the accuracy cost of activation
  quantization (W8A8 is benchmarked for speed only), and Core AI / iOS 27.

## Workload

Encoder: 2-D convolutional subsampling (8x, 80 ms frames), 24 FastConformer
layers (d_model 1024, 8 heads, rel-pos attention, FF 4096, conv kernel 9).
264 ternary modules (FF, attention incl. `linear_pos`, pointwise convs), about
600M weights, one FP scale per output row; codes are about 33% each of -1/0/+1
(pilot P3 export). About 15M parameters stay floating point (subsampling,
depthwise convs, norms). Decoder: 2-layer LSTM prediction network (640), joint
network (640 -> 1024 tokens + 5 durations).

| Utterance | Mel frames | Encoder frames T | Encoder GFLOP |
| ---: | ---: | ---: | ---: |
| 2 s | 201 | 25 | 30 |
| 4 s | 401 | 50 | 60 |
| 8 s | 801 | 100 | 121 |
| 15 s (current fixed window) | 1501 | 188 | 227 |

Weight bytes read per encoder call: FP16 1.21 GB, int8 0.60 GB, 6-bit 0.45 GB,
2-bit 0.15 GB.

**Estimate, not a measurement.** The A17 Pro has about 51 GB/s of LPDDR5
bandwidth. Streaming FP16 weights then takes at least 24 ms per call. At a
nominal 17 TFLOPS FP16 on the ANE (half of the 35 TOPS int8 figure), compute for
a 4 s utterance takes about 4 ms. By this estimate the dense encoder is
memory-bound on the phone at every length we care about. The shorter the input,
the larger the share of time spent moving weights, and so the more compressed
weights should help.

FluidAudio measured the opposite behaviour on M-series Macs, which have much
more bandwidth than the phone. Which regime the A17 Pro is in is the central
open question; the benchmarks decide it.

## Prior evidence

Sources are in the 2026-10-02 research notes (session summary); the key ones are listed here.

- FluidAudio/mobius `parakeet-redux` sweep (M-series Mac, one 15 s window):

  | Encoding of a ternary encoder | ANE latency | First ANE load |
  | --- | ---: | ---: |
  | FP16 | 20.8 ms | 8.7 s |
  | 2-bit LUT with one scale per row (our exact format) | 70.4 ms | 45 s |
  | int8 per-channel (also exact for us) | 27.5 ms | ~14 s |

  The ANE cost grows with the number of palettes. The same 2-bit weights ran on
  the GPU in 21 ms with a 0.6 s load.
- FluidAudio Phonon-2 used a sparse mask plus one FP16 palette per 8 output rows
  (`constexpr_lut_to_sparse`). It ran in 18.6 ms on the ANE, faster than FP16.
  Our zeros are about 33%, not 51%, and our scales are per row, so grouped scales
  would need a retraining variant.
- Core ML decompresses compressed weights just in time from iOS 17 on, and Apple
  reports latency gains for memory-bound models on the ANE. W8A8 uses a faster
  int8 path on A17 Pro and M4. The M1 Pro in our Mac does not have that path, so
  W8A8 can only be judged on the phone.
- The ANE runs fixed and enumerated shapes. Flexible (RangeDim) shapes fall back
  to CPU or GPU. Multifunction models (iOS 18+) deduplicate shared weights.
- With `cpuAndNeuralEngine`, Core ML still runs the TDT decoder and joint 100% on
  the CPU. Per-call overhead is 0.1-0.5 ms. On a 7.8 s clip the decode loop
  (89 calls) costs about as much as the encoder.
- iOS does not allow GPU work while the app is in the background. CPU and ANE
  work is allowed.
- Repository precedent: `custom/cpu-inference` and `custom/inference-efficiency`.
  Both used seeded random models, forced decoder replay, dense twins and
  bit-exact parity, and the same rules apply here.

## Pipeline stages and arms

Every arm is compared against **C0, the status quo**: FluidInference's
published v2 Core ML conversion. It uses a fixed 15 s window, an encoder
palettized with 6-bit k-means, FP16 compute, and decoder and joint as separate
per-step Core ML calls. See "Implementation decisions".

**A. Front end.** Log-mel features via the Core ML preprocessor (C0) versus
Accelerate/vDSP on the CPU.

**B. Encoder length.** A fixed 15 s window (C0) versus a multifunction model with
length buckets (initial set 2/4/8/15 s, padded up to the next bucket) that share
weights. Things to check: weight residency when several functions are loaded,
the per-bucket copy of the folded `linear_pos` table (~18 MB at 15 s), and that
outputs match the 15 s window up to the masked tail.

**C. Encoder weight format on the ANE.** All formats use FP16 activations except
C5:

- C1: FP16 dense
- C2: 6-bit k-means. This is C0's weight format in our graph, so C2 vs C0 isolates graph differences.
- C3: int8 per-channel, exact for ternary × row scale
- C4: 2-bit LUT {-1,0,1} with one scale per row (`enable_per_channel_scale`), exact
- C5: W8A8 (int8 activations, calibrated; speed only)
- C6: sparse mask plus one palette per group of g rows (g = 8, 16). This needs
  grouped-scale weights; a retraining variant is built only if C6 wins on device.
- C7: per-row scale moved out of the weights. The weight is one per-tensor 2-bit
  palette {-1, 0, +1}, so each tensor has a single LUT and no per-channel scale.
  The scale is applied as an explicit per-channel multiply on the matmul output:
  y = s ⊙ (C x). This needs no retraining. Risk: graph passes may fold the
  multiply back into the constant weight. Check the compiled MIL for that.
- C8: two binary planes. P = 1[w = +1] and N = 1[w = -1], each a per-tensor
  1-bit palette {0, 1}, stacked into one [2·out, in] matmul, then split:
  y = s ⊙ (P x − N x). This is the same arithmetic as C7 with twice the MACs.
  It is included in case the ANE decodes 1-bit LUTs more cheaply than 2-bit ones.

**D. Encoder graph layout.** Our MIL graph comes in two layouts. The first is
plain: it mirrors NeMo's ops, as C0's traced graph does. The second is laid out
for the ANE: channels-first `(B, C, 1, T)`, linears as 1x1 conv2d, attention
split per head, no rank-changing reshapes in the hot path. Arms C1-C8 use the
plain layout. The ANE layout is crossed with the best two formats. Both layouts
must pass the same parity gates against the FP32 reference.

**E. Encoder on the GPU** (foreground only): Core ML GPU with 2-bit weights, and
MLX Swift with 2-bit affine quantization (exact for ternary × row scale with
q ∈ {0,1,2}, scale s, bias -s). A custom ternary Metal kernel is written only if
the GPU arm beats the best ANE arm on device energy. Such a kernel (and G) would
store two bit planes per weight, nonzero mask Z and sign S, with w = Z·(1 − 2S).
With FP16 activations each weight then costs a sign-bit XOR, an AND with the
mask, and one accumulate: y = s · Σ((x ⊕ S·0x8000) ∧ Z), with no multiplies and a
single accumulator. The alternative planes P/N need two masks and two
accumulators. For LUT-style CPU kernels the P/N split has its own advantage: one
16-entry table of partial sums per group of 4 activations serves both planes,
y = s · (Σ LUT[P nibble] − Σ LUT[N nibble]), which is T-MAC's bit-plane
decomposition.

**F. Decode loop.** Per-step Core ML decoder and joint calls (C0) versus a native
CPU loop (Accelerate/BNNS or NEON, FP16/FP32). The native loop projects the
encoder side of the joint for all T frames in one batched call before decoding,
runs the prediction network only after a non-blank token, and fuses the
prediction and joint steps.

**G. Ternary CPU encoder kernel** (NEON dotprod/i8mm, T-MAC-style LUT). Deferred.
The encoder is a GEMM over 25-188 rows, where published CPU LUT kernels lose
their advantage. Revisit only if every ANE arm turns out to be compute-bound.

## Benchmark weights and workload

- **Random models.** A seeded generator draws ternary codes with the pilot export's
  per-module code histogram and per-row scales from that export's scale
  distribution. Floating-point modules use NeMo's initialisation, with norms and
  biases drawn from the export's distribution. A grouped-scale variant is used
  for C6. Each random model has a dense FP32 twin.
- **Clips.** LibriSpeech dev-clean utterances bucketed by length (2/4/8/15 s, at
  least 16 per bucket), fixed and hashed. Real speech keeps the front end and
  padding realistic.
- **Forced decoding.** The real pinned v2 model (B0) is run once per clip. Its
  token and duration sequence is recorded, and every arm replays it. Every arm
  therefore makes the same number of prediction and joint steps, whatever its
  random weights would emit.

## Correctness gates (before any timing)

- **Format exactness.** For C3, C4 and the MLX format, the dequantized weights
  equal the ternary codes × the FP16-rounded scales exactly.
- **Arm vs FP32 twin.** On every clip, compare the encoder output's relative L2
  and max-abs error. Tolerances are set from the FP16 dense arm's own error
  against FP32 and recorded before the formats are compared. The replayed decode
  steps produce the same argmax token and duration as the twin on at least 99.9%
  of steps; the actual figure is reported.
- **Rewrite vs NeMo.** The ANE-layout rewrite (D) matches NeMo FP32 to 1e-4
  relative error, using random weights and B0 weights.

## Measurements

- **Latency:** per stage (front end, encoder, decode) and end to end, recorded
  with `os_signpost` and `mach_absolute_time`. Report p50 and p95 per bucket after
  warm-up.
- **Load and size:** cold load (first load including ANE compilation, with the
  Core ML cache cleared), warm load, first inference, peak `phys_footprint`, and
  size on disk.
- **Energy per utterance.** Workloads run in sustained windows (fixed number of
  utterances, fixed cadence) alternating with idle windows in identical device
  state. Energy per utterance = (P_run − P_idle) × window length / utterance
  count, reported as the median over windows.
  - Mac: from `macmon` (IOReport, no sudo) CPU/GPU/ANE/DRAM power.
  - iPhone: system power from the Xcode Power Profiler over wireless debugging.
    The phone must not be charging (charging makes system power read 0). The
    profiler has no ANE track, so total system power is used. Settings: fixed
    brightness, airplane mode, Low Power Mode off.
- **Thermal state:** recorded per window; windows that are not in nominal thermal
  state are flagged.

## Harness

`ParakeetBench`, a Swift package with three parts:

- a shared core: model loading, front end, decode loop, timing and energy-window
  scheduling;
- a macOS command-line target, run on the M1 Pro over SSH, with no phone needed;
- a minimal iOS app driven from the Mac with `xcrun devicectl` (install, launch
  with arguments, copy the result JSON back) and `xcrun xctrace record` (Power
  Profiler).

Model generation and conversion run in Python (coremltools 9, PyTorch, NeMo) on
the Mac, where models can be compiled and run. Results are committed; generated
models stay outside Git.

## Implementation decisions (S0)

- **C0 is the real artifact.** It is FluidInference's published
  `parakeet-tdt-0.6b-v2-coreml` at a pinned revision, unmodified, with real
  weights. Execution cost does not depend on weight values (beyond C2's k-means
  table, which is per-tensor either way), and forced replay equalises decoding.
  This makes C0 exactly what an app would ship today, with no conversion work.
- **Decode traces come from C0.** On each clip, C0 runs greedy TDT decoding with
  real weights in our harness. The recorded token and duration sequence is the
  replay trace for every arm. The trace only needs to be a realistic workload,
  so B0 FP32 is not needed for it.
- **Our arms are written directly as MIL programs** with coremltools' MIL
  builder, from numpy weights. Nothing is traced from PyTorch. This keeps peak
  memory near the size of the final weights rather than several FP32 copies,
  which matters on a shared 16 GB Mac. It also makes the graph form explicit:
  where C7's scale multiply sits, the layout for D, and the bucket shapes. Any
  graph pass that would fold C7's scale into its weight is disabled and checked
  in the saved MIL.
- **FP32 reference model.** A pure-PyTorch implementation of the v2 architecture
  (FastConformer with rel-pos attention, plus the TDT prediction and joint
  networks). It is checked against NeMo on this machine at reduced depth
  (2 layers, CPU, small enough not to count as a heavy job), with matching
  outputs to 1e-5 relative. It then serves as the FP32 twin on the Mac. The MIL
  arms are checked against it.
- **Random-weight statistics** come from the pilot P3 export and are committed
  as a small JSON file: per-module code histograms and the per-row scale
  quantiles. Weights are regenerated from a seed on the Mac.
- **Mac guard.** Every Mac job runs through `ios/macguard`, which:
  - refuses to start if the system free-memory percentage is below 40% or
    another guarded job holds the lock;
  - runs the job under `nice`;
  - kills the job's process tree if its total RSS exceeds a cap (default 6 GB);
  - logs load average and the top CPU users for every timed window.

  The Mac is shared with other work, so Mac numbers are used only to check
  function and prune arms, never for claims.
- **Artifacts on the Mac** live in `/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/`,
  outside any repository. LibriSpeech dev-clean is downloaded there from OpenSLR,
  and clips are selected by a committed manifest of IDs and SHA-256 hashes.
- **Python environment:** `ios/pyenv/` is a uv project with a committed
  `uv.lock`, using Python 3.12, coremltools 9.0, PyTorch (CPU), numpy and
  soundfile. MLX is added at stage S3 if E proceeds.
- **Git:** branch `parakeet-ios` contains only `finetune/parakeet-ternary/ios/`.
  The repository is public: no models, audio, credentials or tokens are
  committed, and the Mac pulls the branch.

## Stages

- **S0 Tooling (Mac only).** Set up the Python environment, the random-model
  generator, decode traces, the harness skeleton and the correctness gates.
- **S1 Mac sweep.** Run all arms on the M1 Pro. Use the results only to check
  function and prune arms. The M1 Pro's ANE is a different generation, so no
  claims are made from it.
- **S2 Device session 1** (~30 min, phone untethered, user present and
  approving). Run C0, C1, C3, C4, C5, C6 plus the best B/F combination, all four
  buckets, latency and energy.
- **S3 Kernel iteration** on whatever S2 shows (D rewrite, E GPU/MLX, C6 group
  size), then device session 2.
- **S4 Real weights.** Feed the M1 export through the chosen pipeline. Check that
  transcripts are identical to the PyTorch export on the dev sets on the Mac, then
  run a final device session.

## Decision rule

Use the S2/S3 device numbers. Among arms that pass the correctness gates, choose
the one with the lowest median energy per utterance, averaged equally over the
four buckets. Its p95 end-to-end latency must not exceed C0's in any bucket, and
its cold load must stay under 10 s after the first launch. Report every arm,
including the losers.

## Open questions

1. Resolved 2026-10-02: transcription runs only in the foreground with the screen
   on. The GPU arms (E) stay in. The display draws power in every arm equally, so
   it cancels in the run-minus-idle energy difference.
2. Resolved 2026-10-02: push branches to GitHub and pull them on the Mac. The
   Mac is shared with other work, so jobs there must stay light (see Mac guard).
3. Resolved 2026-10-02: installing macmon (Homebrew) and a reproducible Python
   environment is approved. sudo is not needed by any planned step.
