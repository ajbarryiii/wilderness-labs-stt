# Plan: native 1-bit GPU kernels for MB1 (RTX 5090)

**Status: 2026-10-20: plan review round 1 NOT CLEAN (8 major, 2 minor); the
section "Amendments (plan review round 1)" at the end supersedes any earlier text
it conflicts with; round 2 NOT CLEAN (3 major), addressed in A11-A13; round 3 NOT CLEAN (1 major), addressed in A14; round 4 NOT CLEAN (1 major: no latency-only elimination), A14 amended; **round 5 CLEAN** (review/gpu-kernel/p5.md). Four implementation
agents started (user request 2026-10-20: several Opus agents in parallel,
collaborating, best approach wins). Stage 2 of plans/benchmark.md, brought
forward; stage 1 (MB1 on M1's runtime) is built here as the baseline.**

## Goal

Run MB1's 264 quantized encoder layers (linear and kernel-1 pointwise
convolutions, including NeMo's transposed `[B, K, T]` input views) on the RTX
5090 straight from **1-bit weights** (export layout `parakeet-binary-v1`: bit 1
= +1, bit 0 = -1, per-row FP32 scale), and measure, honestly, whether this is
faster or lower-energy than MB1 run on M1's existing runtimes (2-bit compact
BF16x3, expanded INT8, hybrid CUDA/Triton `ptx_row_best`). A negative result is
a valid result and is reported as such.

The identity every approach may use: with b in {0, 1}, w = 2b - 1,
`y_j = alpha_j * (2 * sum_i b_ij x_i - sum_i x_i) + bias_j`.

## Hard constraints (all agents)

- **Weight storage on the GPU stays at about 1 bit per weight**: at most 1.10x
  the 75.5 MB of MB1's packed codes for all 264 layers combined, plus FP32
  scales. Any re-layout (tiling, interleaving, bit transposition) at load time
  is allowed; expanding to INT8, FP16 or 2-bit is not (that is the baseline).
  Small per-row auxiliaries (e.g. row popcounts) are allowed within the 1.10x.
- **Numerics**: on real MB1 weights and real activations captured from MB1's
  encoder on dev audio (never test audio), each layer's relative error
  `||y - y_ref|| / ||y_ref||` against an FP64 reference must be no worse than
  1.5x the worse of M1's two production kernels (BF16x3 compact, expanded
  INT8x3) on the same weights (derived 2-bit layout) and inputs. End to end on
  the dev quality sample, transcripts and token IDs are compared with MB1 on
  M1's expanded runtime and with MB1's FP32 NeMo reconstruction; any mismatch is
  listed, not hidden.
- **Inference only**, CUDA-graph capturable, batch 1 (the M1 benchmark
  regime), also correct for batch > 1.
- **No edits** to `finetune/parakeet-ternary/` or `finetune/whisper-ternary/`,
  or to existing `finetune/parakeet-binary/` files. New code lives under
  `finetune/parakeet-binary/gpu1bit/`. Import M1's runtime (`inference.*`)
  rather than copying it; where a monkeypatch is needed, it lives in gpu1bit.
- **GPU discipline**: every process that touches the GPU holds
  `paths.gpu_lock("<agent>:<what>")` (parakeet-ternary paths.py) for its whole
  life, at most 10 minutes per hold; the 400 W cap is never changed; nothing
  else runs on the GPU concurrently. Timing numbers taken without the lock are
  invalid. Use the `python` wrapper that gives the needed libraries (the binary
  wrapper drops NVRTC; the ternary wrapper has it and the hybrid CUDA path needs
  it); record which.
- Artifacts (captured activations, benchmark JSON) go under
  `/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/`; verify the mount.
- Contract tests first (`gpu1bit/tests/`), never weakened. No commits.
- Codex (gpt-6-astra xhigh) reviews this plan now and the final code before the
  recorded benchmark run.

## Team and ownership (each agent writes only its own directory)

| Agent | Directory | Role |
| --- | --- | --- |
| H, harness | `gpu1bit/shared/`, `gpu1bit/tests/` | interface, FP64 reference, real-weight/activation fixtures, contract tests, derived 2-bit export (baseline), M1-kernel baselines, micro / encoder / end-to-end benchmark harness, later the integrated loader |
| A, Triton BF16 | `gpu1bit/triton_bf16/` | Triton: decode 1-bit words in registers to +-1 (or {0,1} with the row-sum identity), BF16x3 tensor-core MMA; the compact path's direct analogue |
| B, INT8 IMMA | `gpu1bit/int8_mma/` | activations as INT8 residual parts (as M1's expanded path), weights expanded from bits to INT8 only in shared memory / registers, INT8 tensor-core MMA; Triton or CUDA/PTX |
| C, bit-serial | `gpu1bit/bitserial/` | first establish whether SM120 runs `mma.sync` b1 (AND/XOR + popc) and at what throughput; if useful, activation bit-planes x weight bits on b1 tensor cores; else a small-M CUDA-core kernel (LOP3/popc or masked add), the GEMV regime of batch-1 decoding |

**Collaboration**: `gpu1bit/NOTES.md`, append-only, every entry headed
`## <agent> <time>`: interface questions, measured numbers (with shape, M,
lock-held), dead ends and why, ideas another approach could use. Read it before
each major step. Agents may borrow each other's ideas, not each other's files.
The coordinator (main session) relays questions and arbitrates.

## Interface (H owns `gpu1bit/shared/api.py`; A, B, C implement it)

```python
class BinaryLayer(Protocol):
    @classmethod
    def prepare(cls, codes: Tensor, scale: Tensor, bias: Tensor | None, in_features: int,
                device) -> "BinaryLayer": ...      # codes: MB1 uint8 [N, ceil(K/8)], LSB first
    def weight_bytes(self) -> int: ...             # everything resident on the GPU for this layer
    def linear(self, x: Tensor) -> Tensor: ...     # [..., K] FP32 (any strides) -> [..., N] FP32
    def conv(self, x: Tensor) -> Tensor: ...       # [B, K, T] FP32 (any strides) -> [B, N, T] FP32
```

Shapes: H extracts the 264 layers' (N, K, kind) and the M values of the 3, 10
and 30.04 s benchmark clips (batch 1) and publishes them in `shared/shapes.json`
early, with captured dev activations per distinct shape.

## Measurements (H's harness; same method as M1's recorded runs)

1. Per distinct shape and M: warm CUDA-graph replay latency (median of
   windows), against MB1 on M1's compact, expanded and `ptx_row_best` kernels.
2. Full encoder (all 264 layers with the rest of NeMo's encoder): latency, NVML
   GPU energy per call, average power, process VRAM, 3 / 10 / 30.04 s clips and
   the 24-utterance multi-corpus set, paired per utterance.
3. End to end (CPU waveform to text) with M1's optimized pipeline (graph cache,
   fused decoder) and the winning kernel swapped in for the encoder projections;
   fresh process per variant, order baseline, new, new, baseline.
4. Report `gpu1bit/RESULTS.md`: the above, weight bytes resident, numerics per
   layer, transcript agreement, what each approach tried and why it won or lost.

## Phases

1. (now) H: interface, shapes, fixtures, reference, contract tests, derived
   export and baselines. A, B, C: prototypes on the published shapes, own
   correctness checks until H's tests land, then H's.
2. Screening: each approach's best per-shape numbers in NOTES.md, measured with
   H's harness. The coordinator selects one approach, or a per-shape mix if the
   data show it, and stops dead ends.
3. Integration by H with the winner's author: `load_binary_packed(export_dir)`
   for MB1's export and the end-to-end pipeline; contract tests.
4. Codex code review until CLEAN; then the recorded benchmark run; report.

## Out of scope

Training changes, decoder or frontend kernels (M1's are reused unchanged),
binarized activations, power-limit changes, iPhone/Mac, CPU (next plan).

## Amendments (plan review round 1, 2026-10-20)

Codex: `/mnt/hd/wilderness-labs-stt/parakeet-binary/review/gpu-kernel/p1.md`.
These supersede the sections above where they conflict.

### A1. Baselines, named and pinned (finding 1)

MB1 (derived 2-bit export) on each of M1's variants, with its exact transform
stack recorded (module, function, arguments, source hashes):
- **R-compact**: `enable_optimizations(model)` default, i.e. `encoder_transform(mode="int8x3", ...)`
  (inference/optimized.py:43): compact 2-bit weights, INT8x3 arithmetic.
- **R-expanded**: `enable_optimizations(model, encoder_storage="expanded")`: INT8 weight cache, INT8x3.
- **R-hybrid**: a fresh `load_packed` model with `experimental_pipeline(model, "ptx_row_best")`
  applied once (it installs its own transforms and graphs; never stacked on
  `enable_optimizations`; as the separate branch in compare_v2.py:116).
- **R-bf16x3** (labelled separately, kernel-level only): `inference.kernels.matmul(mode="bf16x3")`.
A candidate is compared first with the baseline of the **same arithmetic class**
(INT8x3 candidates with R-compact/R-expanded/R-hybrid; a BF16x3 candidate with
R-bf16x3 and, as a whole runtime, with R-compact), so weight packing is not
conflated with arithmetic.

### A2. Integration seam: a projection provider (finding 2)

The unit of substitution is M1's **projection provider** protocol, the object
`encoder_transform`/`layer_fusions` call as
`provider(x, mod, conv, components, activation, norm, config, input_activation)`
(inference/encoder4.py `ExpandedProjection.__call__`; the deferred-epilogue
variant of `cuda_pipeline`/`row_fusion` for R-hybrid). H defines, before
screening ends, `shared/provider.py`: a `BinaryProjection` provider that serves
every call the transforms make (including fused QKV `[3072, 1024]`, SiLU and
norm-input fusion, `input_activation`, conv layout) from a candidate's 1-bit
layers, plus contract tests that, with a reference candidate, the transformed
model's call sequence is the same as R-expanded's (same fusions, no baseline
kernel executes; asserted by instrumentation). Two comparisons are kept apart:
- **controlled substitution**: identical transform stack and fusions, only the
  provider differs (candidate vs R-expanded, and R-hybrid's deferred variant if
  the winner supports it);
- **runtime vs runtime**: best candidate runtime vs best M1 runtime, labelled as
  such.
A/B/C implement the provider-facing operations their kernels support
(activation quantization + matmul + epilogue as one op); `BinaryLayer` remains
the unit-test interface.

### A3. Numerics (finding 3)

Per layer, on **captured inputs for all 264 layers** (dev audio only; H captures
each layer's actual input on a frozen set of dev utterances covering all five
dev corpora) and on synthetic cases (exact cancellation, silence x = 0 giving
y = bias exactly, large dynamic range, single nonzero input), with an FP64
reference:
- all outputs finite;
- Frobenius relative error `||y - y_ref||_F / max(||y_ref||_F, tiny)`;
- elementwise guard: `max_ij |y - y_ref|_ij / (alpha_j * ||x_i||_1 + tiny)`,
  the error relative to the worst-case accumulation magnitude (defined even
  where y_ref = 0);
- both no worse than 1.5x the worse of R-bf16x3 and the INT8x3 kernel of
  R-expanded on the same inputs, per layer; reported per layer.

### A4. Recognition gate (finding 4)

Frozen dev sample (H fixes the utterance IDs before any candidate timing, e.g.
512 utterances across the five dev corpora, plus the timed clips and the
24-utterance set; never test). For each candidate in the controlled
substitution: encoder output lengths equal; encoder outputs within the A3 guard;
text, token IDs and timestamps compared with MB1 on R-expanded. **Eligible to
win** only if (a) the timed clips and 24 utterances match exactly (M1's paired
benchmark asserts this, cuda_experiments.py:209), and (b) on the dev sample the
mismatch count is no more than the mismatch count between R-bf16x3 and
R-expanded on MB1 (the arithmetic-class noise floor), and the dev-sample WER is
within 0.02 points of R-expanded's. Otherwise it is reported as a quality
trade-off, not a winner.

### A5. Storage accounting (finding 5)

Budget: unique GPU storage (deduplicated by `untyped_storage().data_ptr()`)
attributable to the 264 layers' weights **across the whole transformed model**,
measured after preparation and after all warmups and graph captures, including
fused-QKV copies, retained original layouts, padding, closures and provider
caches; at most 1.10 x 75,497,472 bytes (the manifest's exact packed code bytes).
Retaining both per-matrix Q/K/V and a fused copy (+9,437,184 bytes) therefore
does not fit: the fused copy must replace the separate ones or be a view.
Scales, biases and activation workspaces are excluded from the budget but
reported. H's audit runs on every candidate and on the baselines.

### A6. Measurement protocols (finding 6)

Three separate protocols, each specified in `shared/bench.py` and recorded with
the results:
- **P-kernel**: per layer op as the pipeline calls it (activation quantization,
  matmul and epilogue included; stated per variant), warm CUDA-graph replay,
  median of windows.
- **P-paired** (CUDA_IMPROVEMENTS.md method): 24 utterances, caches cleared
  between utterances, one warmed shape at a time, three balanced one-second
  windows per variant per utterance alternating, median window, paired changes
  with counts; encoder scope and full scope; NVML energy over exactly the timed
  replays (synchronized start/stop).
- **P-clips** (EXPERIMENTS4.md method): LibriSpeech 3, 10, 30.04 s in that
  order, accumulating graph cache, three five-second windows, two-second warmup.
- **P-memory**: fresh process per variant, NVML process VRAM sampled every
  10 ms including load and first capture, as recorded for M1.

### A7. One environment (finding 7)

All timed variants run under one wrapper, `gpu1bit/shared/python` (H; the
ternary wrapper's environment, which has NVRTC, with the binary directory on
PYTHONPATH), and the harness asserts, per variant, the same effective decoder
CUDA-graph mode, decoder fusions, frontend settings, TF32/cuDNN flags and torch
/ Triton / driver versions; a mismatch aborts the run.

### A8. Contract-test matrix (finding 8)

Mandatory for every candidate before any timing counts: ragged K (incl. not a
multiple of 8/32), N and M (1, odd, > tile); bit order and nonzero-tail-padding
rejection; bias / no bias; batch > 1; every input stride pattern the transforms
produce (contiguous, transposed conv view, sliced); M = 0 (empty input);
CUDA-graph capture, replay with changed input contents equals eager; alternate
CUDA stream; output does not alias input or a reused buffer unless documented;
grad-enabled input rejected; `compute-sanitizer` memcheck/racecheck clean on one
small shape per kernel (if available on this machine); provider dispatch
coverage (every transform call path reaches the candidate; instrumented).

### A9. Locks (minor 1)

One lock owner per GPU worker process: each GPU run is a separate process
started as `timeout 600 <wrapper> ...` that takes `paths.gpu_lock` once at
start; harness code never calls functions that take the lock themselves while
holding it (e.g. `compare_v2.main`). Before timing, the harness asserts the power
limit equals 400 W exactly and that no foreign compute process is on the GPU
(nvidia-smi compute apps), as M1's runs did.

### A10. Shared files and handoff (minor 2)

NOTES.md entries are appended only with `gpu1bit/note <agent>` (flock
serialized). Artifacts per agent: `/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/<agent>/`.
Integration (phase 3) is an explicit handoff by the coordinator: the winner's
author freezes its directory, H integrates; before the recorded benchmark all
gpu1bit sources are snapshotted with SHA-256s into the results.

### A11. R-hybrid is a separate runtime (round 2, finding 1)

See A1: fresh model, `experimental_pipeline(model, "ptx_row_best")` once.

### A12. Recognition gate, fixed criteria (round 2, finding 2; replaces A4's criteria)

The reference is MB1 on R-expanded, same environment. Frozen now, before any
candidate result:
- Encoder output: finite everywhere; output lengths equal; on valid frames only
  (masked by length) the per-utterance relative Frobenius difference
  `||E - E_ref|| / ||E_ref||` is at most **1e-3**, and the maximum absolute
  difference is reported.
- Recognition on the dev sample: token-ID sequences differ on at most **0.5%**
  of utterances, and the dev-sample WER is within **0.02 points** of the
  reference's.
- Dev timing workloads (A13): text, token IDs and timestamps identical.
A candidate failing any of these is a quality trade-off, not a winner. R-bf16x3
stays kernel-level only; no BF16x3 recognition control is needed.

### A13. Dev-only workloads for screening and gates (round 2, finding 3)

H builds, from dev manifests only (asserting split = dev and the manifest
path), before any candidate timing: **dev clips** of about 3, 10 and 30 s from
LibriSpeech dev-clean (exact durations recorded), and a **dev 24-utterance set**,
8 per 3 / 10 / 30 s target across the five dev corpora, chosen by a fixed rule
(e.g. closest durations, ties by ID). All screening, P-kernel inputs, the gate
(A12) and selection use these and the dev sample only. M1's historical test
workloads (the LibriSpeech test clips and the 24 test utterances) are used only
after the candidate is frozen, for timing comparability with M1's recorded
numbers in the final report; no selection or tuning uses them, and the report
says so.

### A14. Selection is decided on integrated measurements (round 3)

Phase 2 per-shape screening is **provisional** and never rejects on speed: an
approach is stopped before integrated evaluation only if it fails its contract
tests, numerics (A3) or storage budget (A5). Every approach meeting those, and any
per-shape mixture proposed with data, goes through H's provider (A2) and is
measured on the dev workloads (A13) with P-paired at encoder and full scope and
with the A12 gate, before final selection. P-kernel also reports an L2-cold
variant (weights evicted between replays by streaming a buffer larger than L2,
or rotated across copies of the layer set) beside the warm one, since
single-matrix replays can keep weights in L2 (inference/README.md:187).

**Winner rule (frozen now):** among candidates that pass A12, the lowest median
paired change in **GPU energy per full transcription** on the dev 24-utterance
set (the project's objective is power efficiency); if two candidates differ by
less than 1% or neither is lower on at least 16 of 24 utterances, the lower
median full-transcription latency decides. The winner, its configuration and
source hashes are frozen before any historical test workload is run.

### A15. Winner among three or more candidates (Codex GPU code review c2/c3, 2026-10-21; retrospective)

**Status:** a retrospective amendment. A14's tie text is written for two candidates. The frozen GPU
implementation compared only the top two candidates by energy, and on sel-20261020b it picked
B-hybrid-b01. Codex GPU code review c2 found that a third candidate, B-hybrid, also ties with the
first and was never compared.

**Provenance:** this amendment was written after sel-20261020b's results were known. It adapts the
anchored tie set that the CPU plan already used (plans/cpu-benchmark.md review round 2; full text in
review/cpu-benchmark/p4.log, "Winner among improving candidates (deterministic)", 2026-10-20 08:43).
That earlier text shows the method existed; it is not a prior commitment for this GPU selection.

**Effect:** applied to sel-20261020b, it changes the winner from B-hybrid-b01 to B-hybrid.
`report.json` (top two, as recorded) and `report-corrected.json` (A15) are kept separate.

**Rule (frozen now, before any new measurement; implemented as `gpu1bit/shared/selection.py`
`winner_rule` and `a14_tie`):**

Paired comparisons are on the dev 24-utterance set at full-transcription scope. The change of a vs b
per utterance is (x_a - x_b) / x_b, and "median" is the median of these 24 changes.

1. **Anchor.** A* is the eligible candidate (A12, and everything selection requires) with the lowest
   (median energy change vs R-expanded, name).
2. **Tie set.** S is A* plus every other eligible candidate c whose head-to-head energy comparison
   with A* is an A14 tie. That is: |median change of A* vs c| < 1%, or neither A* nor c is lower on at
   least 16 of 24 utterances. S is anchored on A*, not closed under ties, because pairwise ties are
   nontransitive.
3. **If S = {A*},** A* wins by energy.
4. **Otherwise**, the member of S with the lowest median full-transcription latency change vs
   R-expanded wins. If several members are exactly equal on that, the one whose head-to-head latency
   median is below zero against every other equal member wins. If there is still no single winner,
   the name that sorts first wins.

With two candidates this is A14 as written (a tie is decided by latency).

**Freeze:** this plan file is part of every gpu1bit fingerprint (`shared/fingerprint.py`) and every
session freeze. Any later measurement therefore binds to this text, and any edit to it changes
every fingerprint.

