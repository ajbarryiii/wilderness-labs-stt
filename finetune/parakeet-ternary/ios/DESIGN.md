# Parakeet-TDT 0.6B v2 on iPhone 15 Pro: inference pipeline and benchmarks

Status: **revision 2, after Codex design review r1 (2026-10-02). S0 approved
by the user; nothing benchmarked yet.** Review findings and their resolution
are listed at the end.

## Goal and scope

Build the lowest-latency, lowest-energy inference pipeline for the Parakeet-TDT
0.6B v2 architecture (English; the ternary QAT model from `../DESIGN.md`) on an
iPhone 15 Pro (A17 Pro, 8 GB, iOS 26), and the benchmarks that measure it on the
device. The workload is one foreground push-to-talk utterance of 2-15 s.

- **Surrogate models.** The architecture is fixed; the weights are not. Most
  benchmarks use seeded random *surrogate* models. They have the exact
  architecture and the ternary structure of the selected recipe's export, so
  they can run before training finishes. Surrogate results describe execution
  only, not recognition. They are reported separately from trained-model
  results.
- **Trained weights enter at S5.** Finalists are re-measured with the trained
  weights, and their decoding is verified on the phone.
- **Deployment target:** iOS 26 or later; the pilot controls hardware and
  software.
- **Out of scope:**
  - wiring into tccc-bot (a later step);
  - recognition accuracy of surrogates;
  - streaming decoding;
  - background or screen-locked operation (foreground only, by the user's
    decision);
  - Core AI / iOS 27.
- **Exploratory arms** run and are reported but cannot win deployment. These
  are encodings that are not exact for the trained weights (C2 aside as the
  product baseline), and C5 (W8A8), whose accuracy needs its own QAT study.

## Workload

**Encoder.** Two-dimensional convolutional subsampling (8x, 80 ms frames), then
24 FastConformer layers: d_model 1024, 8 heads, rel-pos attention, FF 4096,
conv kernel 9.

- **Ternary modules:** 264 (FF, attention including `linear_pos`, pointwise
  convs), about 600M weights. Each output row has one FP32 scale; the export
  stores scales in FP32.
- **Code mix:** about 33% each of -1, 0, +1 (pilot exports).
- **Floating point:** about 15M parameters (subsampling, depthwise convs,
  norms).

**Decoder.** A 2-layer LSTM prediction network (640 wide) and a joint network
with 1,030 outputs: 1,024 tokens, blank, and 5 durations.

**Encoder size per utterance length.** The model gets fixed-shape tensors, so
some frames are only padding. Allocated frames are the encoder output frames
for the padded input shape; valid frames are those that hold the utterance.
Valid frames below are approximate (one per 80 ms). The GFLOP column is the
matmul estimate 2 × 600M × allocated frames. S0 recomputes every row from the
actual graph and records it.

| Bucket | Mel frames (allocated) | Encoder frames, allocated | Encoder frames, valid | Encoder GFLOP (est.) |
| ---: | ---: | ---: | ---: | ---: |
| 2 s | 201 | 26 | ≤ 25 | 31 |
| 4 s | 401 | 51 | ≤ 50 | 61 |
| 8 s | 801 | 101 | ≤ 100 | 121 |
| 15 s (current fixed window) | 1501 | 188 | ≤ 188 | 226 |

**Hypothesis, not a conclusion.** Encoded weight sizes are FP16 1.21 GB, int8
0.60 GB, 6-bit 0.45 GB, and 2-bit 0.15 GB. The A17 Pro has roughly 51 GB/s of
DRAM bandwidth. If weights were streamed from DRAM once per call in their
encoded form, the dense FP16 encoder would need at least 24 ms per call just to
move weights. That would make short utterances bandwidth-limited and favour
compressed weights.

That reasoning rests on unverified assumptions:
- the ANE's FP16 throughput (Apple publishes only an int8 TOPS figure);
- its achieved utilization;
- whether decompression happens during execution or before it (this is
  backend-dependent per Apple);
- activation traffic;
- the folded `linear_pos` constants.

FluidAudio reported compute-bound behaviour on M-series Macs. No gain is
attributed to bandwidth unless latency scales with encoded size across buckets
in the measured way, with operator placement recorded.

## Prior evidence (with limits)

Permalinks, artifact hashes and versions are committed in
`ios/references.md` (S0). Summary:

- **FluidAudio/mobius `parakeet-redux`** (M-series Mac, 15 s window): FP16
  ran at 20.8 ms on the ANE. A per-row 2-bit *diagnostic* encoding (not exact)
  ran at 70.4 ms with a 45 s first load. int8 per-channel ran at 27.5 ms. The
  GPU ran 2-bit weights at 21 ms.
- **FluidAudio Phonon-2:** a grouped sparse-palette encoding ran at 18.6 ms
  on the ANE. Its dense grouped-palette variant tied it, so the evidence
  favours grouped palettes, not sparsity as such.
- **coremltools 9:** `enable_per_channel_scale` with per-tensor granularity
  produces one shared LUT plus `constexpr_blockwise_shift_scale`; it does
  *not* produce one palette per row.
- **Apple:** compressed weights are decompressed just in time on the ANE
  (from iOS 17), with gains for memory-bound models. W8A8 uses faster int8
  compute on A17 Pro and M4. Our M1 Pro has neither.
- **Shapes:** fixed and enumerated shapes are compiled per shape. Flexible
  shapes can run on the ANE from iOS 17.4 with `reshapeFrequency =
  .infrequent`. No shape choice guarantees ANE placement, because
  `cpuAndNeuralEngine` allows CPU fallback. Placement is therefore recorded
  per arm and bucket with `MLComputePlan`.
- **Decoder placement:** FluidAudio reports the decoder and joint running on
  the CPU with 0.1-0.5 ms per call. This is to be confirmed with compute plans
  and our own timings.
- **Background:** iOS 26 can allow background GPU work through
  continued-processing tasks, with an entitlement. This experiment is
  foreground-only by scope.
- **Repository precedent:** `custom/cpu-inference` and
  `custom/inference-efficiency` used seeded random models, forced decoder
  replay, dense twins, and bit-exact parity.

## Arms

**C0, the product baseline.** FluidInference's published
`parakeet-tdt-0.6b-v2-coreml`, at a pinned revision and with its real weights:
fixed 15 s window, encoder with 6-bit k-means palettes, FP16 compute, decoder
and joint as separate per-step Core ML calls. It is what an app ships today.
It is **not** a graph-only control (see G0).

**G0, the graph control.** Our plain graph built from the real pinned v2 weights
(B0), with the same 6-bit k-means settings as mobius `quantize_coreml.py`, the
same shapes, precision, deployment target and runtime settings. G0 vs C0
isolates graph differences; the remaining difference is k-means
initialisation, which is reported.

**Encoder weight encodings (C).** All arms use FP16 activations except C5.
"Exact" means that the weights the arm decompresses equal codes × FP16(scale)
bit for bit. This is checked with coremltools' constexpr evaluation. Rounding
the FP32 export scales to FP16 is measured as its own error term (see gates).

| Arm | Weight constexpr chain | LUT / index | Scale applied | Exact |
| --- | --- | --- | --- | --- |
| C1 | dense FP16 const (scale folded into weight) | none | in weight | FP16 product rounding |
| C2 | `constexpr_lut_to_dense`, per-tensor 6-bit k-means | 1 LUT/tensor, 6-bit | in weight | no (product baseline format) |
| C3 | `constexpr_blockwise_shift_scale`, int8 data = codes, per-row scale | none, int8 | at decompression | yes |
| C4 | `constexpr_lut_to_dense` LUT {-1, 0, +1, 0} → `constexpr_blockwise_shift_scale` per-row | 1 LUT/tensor, 2-bit | at decompression | yes |
| C7 | `constexpr_lut_to_dense` LUT {-1, 0, +1, 0}, then explicit `mul` by per-row scale on the matmul output | 1 LUT/tensor, 2-bit | after matmul | yes |
| C8 | two planes P = [w = +1], N = [w = -1] stacked [2·out, in], `constexpr_lut_to_dense` LUT {0, 1}; output split, `sub`, `mul` | 1 LUT/tensor, 1-bit | after matmul | yes |
| C6s(g) | `constexpr_lut_to_sparse` + `constexpr_sparse_to_dense`; per-grouped-channel LUT over g output rows holds the 2g values {±s_r}; zeros in the mask | 1 LUT per g rows; g = 2, 4, 8 → 2, 3, 4-bit | in LUT | yes |
| C6d(g) | dense per-grouped-channel LUT {0, ±s_r}, 2g + 1 values | g = 4 → 4-bit, g = 8 → 6-bit | in LUT | yes |
| C5 | C3 weights plus int8 activation quantization (calibrated) | int8 | — | exploratory only |

Notes on the encodings:
- **C7/C8 numerics.** C7 and C8 change the numerics, not just the algebra.
  Delaying the scale makes the intermediate sums larger, and C8 subtracts two
  large sums. The FP16 ranges of those intermediates are recorded, and the
  gates apply.
- **C7 folding risk.** The C7 output multiply must survive conversion.
  Tiny probes (one linear and one 1×1 conv) check, with a pinned pass
  pipeline, whether `common::fuse_conv_scale` or another pass absorbs the
  multiply into the weight. The serialized MIL is inspected. Folding inside
  the device compiler cannot be observed; it is recorded as unresolved, and
  timings speak for themselves.

**D. Graph layout.**
- *Plain* mirrors NeMo's ops.
- *ANE* uses channels-first `(B, C, 1, T)`, linears as 1×1 conv2d, attention
  split per head, and no rank-changing reshapes in the hot path.

Every exact C arm is built in both layouts; Mac pruning only removes
correctness or resource failures (see Stages).

**B. Encoder length.**
- A fixed 15 s window (C0's contract).
- A multifunction model with buckets of 2, 4, 8 and 15 s, sharing weights.
- An enumerated-shape model with the same four shapes, as a control against
  multifunction.

The padding and masking contract is specified in S0 and gated (see gates).
Things to check:
- that weights stay resident once when several functions are loaded;
- the per-bucket copy of the folded `linear_pos` table.

**A. Front end.** Core ML preprocessor vs Accelerate/vDSP on the CPU. Both
implement the complete NeMo feature contract for this model's config:
- dither off at inference;
- pre-emphasis;
- STFT window and padding;
- mel filterbank;
- log guard;
- per-feature normalization over **valid** frames only;
- padding applied after normalization.

**E. Encoder on the GPU** (foreground): Core ML on the GPU (C4 and C7
encodings) and MLX Swift.
- **MLX encoding.** MLX weights are built explicitly as q = C + 1 ∈ {0, 1, 2},
  with scale s and bias -s repeated over every group of the row, at the
  supported group size. The round trip to codes × scale must be exact; MLX's
  own quantizer is not used.
- **Custom Metal kernel.** Built only if a GPU arm beats the best ANE arm on
  device energy.
  - Planes: nonzero mask Z and sign S, with w = Z·(1 − 2S).
  - Arithmetic per weight: reinterpret the FP16 activation's bits, XOR the
    sign with S << 15, then AND with the mask expanded to 0x0000 or 0xFFFF.
    Accumulate in FP32 and apply the row scale at the end.
  - Unpacking, reductions and the final scaling count toward its cost; it is
    not assumed free.
  - A P/N LUT variant (T-MAC style, one 16-entry partial-sum table per 4
    activations, shared by both planes) is the CPU-side alternative.

**F. Decode loop.** The C0 contract (per-step Core ML decoder and joint calls),
a fused decoder+joint Core ML call, and a native CPU loop (Accelerate/BNNS)
that:
- projects the encoder side of the joint for all frames in one batched call;
- caches the prediction state;
- fuses the prediction and joint steps.

Arms share the same *logical* work, from the replay trace (below). Physical
call counts differ by design and are reported.

**G. Ternary CPU encoder kernel** (NEON dotprod/i8mm or LUT): deferred.
Revisited only if the measured ANE and GPU placements make it plausible.

## Surrogate models, clips and traces

**Statistics.** Weight statistics come from the **selected recipe's** pilot
export (P2, lr 5e-4): per-module code histograms, per-row FP32 scale
quantiles, and per-tensor statistics for floating tensors. Normalization state
is kept valid (positive BatchNorm variances, plausible LayerNorm gains).

**Seeds.** Three surrogates (seeds 0, 1, 2) are generated deterministically on
any machine, and per-tensor SHA-256 manifests are compared across Linux and
macOS.

**Limits.** Marginal statistics leave out spatial structure, inter-row
correlations, and trained activation statistics. So surrogate rankings are
confirmed with the trained export at S5, and C5 calibration on surrogates is
for speed only.

**Clips.** LibriSpeech dev-clean utterances:
- 16 per bucket, plus boundary lengths placed just inside and outside each
  bucket edge and around subsampling-stride boundaries;
- a silence clip and an impulse clip;
- selected by a committed manifest of IDs and SHA-256 hashes.

**Replay trace.** The FP32 reference with the real B0 weights runs greedy TDT
decoding on every clip and records the complete trace. Each step records:
- the encoder frame index;
- the prediction-net input token;
- whether the step emitted blank or a token;
- the duration;
- the symbols-per-frame counter;
- whether the prediction state was updated.

Every arm replays this trace. Replay returns logits and LSTM states, so arms
are compared before any decision is overridden. Each arm also decodes freely,
reported separately.

## Correctness gates (before any timing)

Thresholds are fixed here, before any arm exists.

1. **Reference vs NeMo.** Two comparisons, both on CPU in FP32:
   - full depth with the real B0 weights;
   - full depth with surrogate seed 0, using NeMo golden outputs saved on Linux.

   Compared quantities: features; the subsampling output; every layer's
   output; the encoder output; LSTM states; joint logits; and greedy tokens and
   durations.

   Ceilings for all of these: relative L2 ≤ 1e-5, max-abs ≤ 1e-4 × the tensor's
   RMS, and identical greedy decisions.

   Required: every output finite. An input-sensitivity check must change the
   encoder output by at least 100× the parity error.
2. **Exact encodings.** The decompressed weights equal codes × FP16(scale) bit
   for bit. MLX: unpacked q − 1 == codes, with exact scale and bias.
3. **FP16-scale term.** The FP32 reference is run with FP16-rounded scales, and
   its encoder-output difference from the FP32-scale reference is reported.
   This is the price of FP16 scales, separate from the encoding.
4. **Arm vs FP32 reference** (same weights, FP16-rounded scales):
   - Encoder output: relative L2 ≤ 2e-2 and max-abs ≤ 0.25 × RMS on every
     clip and bucket.
   - Replayed joint logits: relative L2 ≤ 2e-2. Argmax agreement on at least
     99.5% of steps, with the distribution of logit margins reported so that
     a degenerate model cannot pass on agreement alone.
   - Every output finite; FP16 intermediate maxima recorded for C7 and C8.
5. **Buckets vs full window.** On the valid frames, every bucket matches the
   same arm's 15 s window output to within the gate-4 ceilings. This is
   checked on the boundary-length, silence and impulse clips. Front end A
   (vDSP) matches NeMo features to relative L2 ≤ 1e-5.
6. **Graph and placement record.** For each arm and bucket the record holds:
   the saved MIL (scale placement, constexpr chain); the `MLComputePlan`
   device usage per operation; and the fallbacks.

## Measurements

**Timing boundary.** End to end runs from a 16 kHz PCM buffer in memory to the
final token IDs on the CPU. It includes:
- the front end;
- all transfers;
- the encoder, with its output materialized: `MLMultiArray` read, or
  `mx.eval` plus synchronisation for MLX;
- the joint pre-projection;
- the decode loop.

Per-stage times use `os_signpost` and `mach_absolute_time`.

**Repetition and statistics** (fixed in advance):
- 3 warm-up calls per clip and arm.
- Timed calls: 10 per clip on the Mac, 5 on the phone.
- Per clip, the median is taken. Per bucket, the median of clip medians
  is reported.
- p95 uses the Harrell–Davis estimator over the per-call pool. Confidence
  intervals are 95% bootstrap intervals, clustered by clip and by session.
- Gross and incremental values are both published.

**Load.** Four separate quantities, per function and shape:
1. Package compile time (`.mlpackage` to `.mlmodelc`).
2. Uncached device specialization: the first load after a fresh install.
   Verified: a second fresh install is uncached again, and a relaunch is not.
3. Fresh-process cached load.
4. First inference.

Peak `phys_footprint` and size on disk are recorded too.

**Energy, phone.** The instrument is validated in the S2 pilot before any
comparison.
- *Candidate instruments:*
  - Power Profiler system power over wireless debugging, in % battery per
    hour (a proxy, not watts);
  - the battery gauge's instantaneous current and voltage, if they are
    readable over the network via the diagnostics relay. To be verified; if
    readable, it gives watts.
- *Verification:* the export path, sampling resolution and units are
  checked.
- *Preregistered metric:* the incremental battery fraction per utterance,
  converted to joules only if an absolute instrument is validated.
- *Protocol:*
  - a static black UI, fixed brightness, fixed radio settings;
  - battery between 40 and 90% and not charging;
  - randomized paired run/idle blocks, with idle blocks before and after;
  - a fixed settling time;
  - nominal thermal state only;
  - a sham workload with identical cadence;
  - one on-device Performance Trace check without the Mac attached.

**Energy, Mac.** Not measured for claims. macmon 0.8.2 on macOS 27 / M1 Pro
reports 0 W for the CPU while it is busy, and the Mac is shared. Only system
power is logged, for information.

## Harness

`ParakeetBench`, a Swift package with three parts:
- a shared core: model loading, front ends, decode loops, signposts, window
  scheduling;
- a macOS command-line target, run on the M1 Pro over SSH through `macguard`;
- a minimal iOS app.

The phone is driven from the Mac with `xcrun devicectl` (install, launch with
arguments, copy results back) and `xcrun xctrace record` (Power Profiler).

Model generation, conversion and gates run in Python (`ios/pyenv`) on the Mac,
one model or function at a time. Results and manifests are committed;
models, weights and audio stay outside Git.

## Implementation decisions (S0)

- **C0** is the unmodified published artifact (pinned revision, hashed).
- **Encoder arms are authored as MIL programs** from numpy weights with the
  coremltools MIL builder, layer by layer. Before full models, one-layer
  probes per encoding measure peak memory and conversion time, and the
  probes are extrapolated to full depth before any full build.
- **FP32 reference** (`ios/reference.py`): pure PyTorch, no NeMo, runs on
  both machines.
- **NeMo golden outputs** are made on Linux in a memory-capped CPU unit (the
  GPU belongs to the training run).
- **`ios/macguard`** wraps every Mac job:
  - start conditions: system free memory at least 40% and no other guarded
    job;
  - runs under `nice`, with thread caps;
  - kills the process tree when RSS goes over its cap or it runs past its
    timeout;
  - logs memory pressure and swap growth;
  - records load average and the top CPU users per window.

  RSS polling can miss short peaks and system compiler services, so
  conversions are sized from the probes with a 2× margin. A full FP32
  reference and a converted arm are never held in memory at the same time.
- **Artifacts on the Mac:** `/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/`.
- **Python environment:** `ios/pyenv/`, a uv project with `uv.lock`: Python
  3.12, coremltools 9.0, torch 2.7.0.
- **Git:** branch `parakeet-ios`, containing only
  `finetune/parakeet-ternary/ios/`. The repository is public, so nothing
  committed may contain models, audio, credentials or tokens.

## Stages

- **S0 Tooling (Mac and Linux).** Reference, golden outputs, surrogates,
  clips and traces, macguard, one-layer probes (including the C7 folding
  probe), the S0 recomputation of the workload table, and the harness
  skeleton with C0 running.
- **S1 Mac build and gates.** Every arm in C × D × B is built and gated. An
  arm is dropped on the Mac only when it fails a correctness gate or a
  resource limit (conversion over its memory cap, compile failure, or load
  over 10 min). Mac latencies are recorded for information only.
- **S2 Device pilot**, the first approved session, about 30 min. Goals:
  - validate the energy instrument and the cold-load procedure;
  - measure noise;
  - smoke-test every surviving arm at the 4 s and 15 s buckets.
- **S2b Preregistration.** Using the pilot's measured noise, commit the
  confirmation matrix: arms, buckets, repetitions and number of sessions,
  sized to the approved sessions. Every hypothesis the write-up claims gets
  coverage on the phone; untested combinations are labelled.
- **S3 Exploration.** Kernel and graph iteration (D, E, F, C6 group size) on
  the Mac and in exploration sessions on the device. Results are reported as
  exploration.
- **S4 Confirmation.** Finalists are frozen, then the preregistered matrix
  runs in fresh sessions. Selection uses only these data.
- **S5 Trained weights.** The M1 export goes through the finalists. Free
  decoding on the phone must match the PyTorch export's transcripts on the dev
  clips (the gate-4 numerics, plus a transcript agreement rate that is
  reported). The finalists are then measured again.

## Decision rule (applied to S4 data only)

1. **Eligible:** exact encodings that pass every gate. C2-format arms other
   than C0, C5, and any arm with an unresolved CPU fallback in the encoder
   are exploratory.
2. **Latency non-inferiority vs C0,** in every bucket: p50 end-to-end ≤ 1.05×
   C0, and p95 ≤ 1.10× C0. Fresh-process cached load ≤ 5 s.
3. **Energy improvement:** the incremental energy per utterance, averaged with
   equal weight over the four buckets, must improve on C0 by at least 10%,
   with the 95% clustered bootstrap CI of the improvement excluding zero.
   Per-bucket results are always reported.
4. **Selection:** among the arms that qualify, the lowest energy wins. If the
   top arms' CIs overlap, the lower cached load wins, then the smaller size on
   disk.
5. **No qualifying arm:** C0 is retained and the result is reported as
   inconclusive.
6. **Reporting:** the full Pareto plot (energy vs p95 latency, per bucket),
   including every losing arm.

## Publication rules

- Surrogate execution results are published separately from trained-model
  quality and performance.
- Each published result carries:
  - source permalinks;
  - artifact hashes;
  - toolchain, OS and device versions;
  - compression settings;
  - the raw observations.
- Exploration and confirmation are labelled.

## Open questions

All resolved, 2026-10-02:
1. **Background:** foreground only, screen on; the GPU arms stay in.
2. **Code transfer:** push branches to GitHub and pull them on the Mac. The Mac
   is shared with other work, so jobs there stay light.
3. **Installs:** macmon (Homebrew) and the uv environment are approved; no
   planned step needs sudo.

## Review r1 (Codex gpt-6-astra xhigh, 2026-10-02): resolution

The review is at `/mnt/hd/wilderness-labs-stt/parakeet-ios/reviews/design-r1.md`.

| # | Finding | Resolution |
| --- | --- | --- |
| 1 | The parity chain could certify a shared bug | Gate 1: full-depth NeMo goldens (real and surrogate weights, intermediates), fixed ceilings, sensitivity check |
| 2 | Eligibility contradicted the scope | Exploratory class; S5 decoding on the phone |
| 3 | C2 vs C0 did not isolate the graph | G0 control; C0 kept as the product baseline |
| 4 | Surrogate representativeness | P2 statistics, valid norms, 3 seeds, limits documented, S5 confirmation |
| 5 | Incomplete replay trace | Full trace, logits and states compared, free decoding separate, logical vs physical work |
| 6 | "Exact" vs FP32 scales | Gate 3 FP16-scale term; explicit C3 and MLX construction; exact round trips |
| 7 | C4 is already one LUT | Constexpr chains in the arm table |
| 8 | Exact grouped palettes are possible | C6s/C6d exact, no retraining |
| 9 | C7 folding | Probes, pinned passes, MIL inspection; device folding labelled unresolved |
| 10 | C7/C8 numerics; Metal formula | Intermediate ranges recorded; mask-expansion formula; costs counted |
| 11 | Front-end and bucket gates | Full feature contract; gate 5 with boundary, silence and impulse clips |
| 12 | Dimensions | Joint 1,030; allocated vs valid frames; S0 recomputation |
| 13 | Categorical placement claims | Claims qualified; MLComputePlan record; enumerated-shape control |
| 14 | Background GPU claim | Qualified; foreground by scope |
| 15 | Roofline | Relabelled as a hypothesis with its assumptions |
| 16 | Energy units | Instrument validation in S2; preregistered metric |
| 17 | Subtraction confounds | Paired randomized blocks, bracketing, sham, settling, thermal and battery rules |
| 18 | Timing boundaries and statistics | Boundary defined; repetitions and estimators fixed |
| 19 | Cold-load definitions | Four separate quantities |
| 20 | Pruning could drop the phone winner | Mac prunes only on failures; phone coverage rule |
| 21 | Noise-driven selection | Non-inferiority, threshold, CI, ties, fallback to C0, confirmation only |
| 22 | Mac guard bound | Probes first, one model at a time, swap and pressure monitoring, 2× margin |
| 23 | Phone session budget | S2 pilot; matrix sized from noise in S2b |
| 24 | Provenance | `references.md`; corrected readings of redux and Phonon-2 |
