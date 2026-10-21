# Plan: fewer activation parts (INT8x2, INT8x1) for M1 and MB1 on the GPU

**Status: 2026-10-20: plan review round 1 NOT CLEAN (5 major, 2 minor); the
section "Amendments (review round 1)" supersedes conflicting text above; **round 2 CLEAN** (review/activation-precision/p2.md; its one minor fixed in R8). User request 2026-10-20:
"It doesn't hurt to try it. Let's bench on full transcripts both for M1 and
MB1 ... maybe also even try single INT8 with the 0,1 representation."**

## Question

Every runtime so far quantizes each activation row into three residual INT8
parts (INT8x3) and runs three tensor-core passes per projection. With two parts
(INT8x2) or one (INT8x1), how much recognition quality do M1 (ternary) and MB1
(binary) lose on full transcriptions, and how much latency and energy do they
save?

The per-layer bar of plans/gpu-kernel.md A3 (no worse than M1's own kernels)
would reject both by construction. Here they are judged on recognition, the
quantity that matters. Prior evidence: M1's 2026-10-05 screen
(`../parakeet-ternary/inference/EXPERIMENTS.md:121`) matched 512 of 512
transcripts with INT8x2, and 478 of 512 with single INT8, but on test
utterances and M1's weights only.

## Arms (dev only; no test audio anywhere)

| Model | Runtime | Parts |
| --- | --- | --- |
| M1 | M1's R-expanded stack (`encoder_transform` with mode int8 / int8x2 / int8x3, expanded INT8 weights) | x3 (reference), x2, x1 |
| MB1 | the winning 1-bit kernel in the provider stack (R-expanded's transforms, B's `int8_mma` layers) | x3 (reference), x2, x1 |

- **Same arithmetic, two weight sets.** The activation split is M1's own
  quantizer at each part count. Weights are M1's ternary INT8 cache versus
  MB1's 1-bit fragments. The x3 rows are the existing references.
- **{0,1} representation.** For MB1, ±1 and {0,1} operands give identical
  integers at every part count, so they cannot differ in quality. The {0,1}
  form (`int8_mma:b01`) is used if it wins its pending head-to-head with
  B-hybrid; otherwise ±1. With x1 the row-sum correction is one INT32 per row.
- **The hybrid runtime (`ptx_row_best`)** supports x3 only; it falls back for
  other part counts. Here all arms use the provider/expanded stack so the
  comparison is controlled. B-hybrid at x2 / x1 is out of scope unless B
  extends its deferred path, in which case it is reported separately.
- **Equal tuning effort.** M1's x2 / x1 kernels use untuned default tiles
  (`fused_kernels.py:149`), while MB1's would be tuned by agent B. To keep the
  comparison fair:
  - B tunes the tile configurations for both models with the same search space
    and budget;
  - M1's configurations are passed through `ExpandedProjection`'s `config`
    argument, with no edits to M1's files;
  - the tuning searches are logged and run on dev clips only.

## Measurements

1. **Recognition, full dev set.** All five dev sets, every utterance, through
   each arm's full runtime: waveform, then frontend, encoder and greedy TDT
   decoder, then text, at batch 1, the deployment regime.
   - Scoring uses frozen `evaluate.py`'s normalizer and corpus WER, called on
     the runtime's text.
   - Reported per set and as the mean.
   - Cross-check: each model's x3 arm must reproduce that model's recorded
     FP32-NeMo full-dev WER within 0.02 points. That shows the runtime path
     itself is faithful before part counts are compared. MB1 is checked
     against 6.312%, M1 against 6.086%.
   - Also recorded:
     - per-utterance token-ID mismatches against the same model's x3 arm;
     - empty-hypothesis counts;
     - encoder-output relative Frobenius on valid frames against x3, on the
       512-utterance dev sample.
2. **Latency and energy.**
   - P-paired on dev_24 at full and encoder scope: x2 and x1 each against x3
     of the same model, plus MB1-xN against M1-xN.
   - P-clips on the dev clips.
   - P-memory in fresh processes.
   - All with the same harness, the 400 W check, locks and balanced order as
     the selection session.
3. **Per-layer numerics (descriptive only).** The A3 error ratios for x2 and
   x1 on the 15 fixture utterances, so the size of the departure from M1-class
   accuracy is on record.

## Reading (frozen now)

This is a measurement, not a selection.

- **Recognition-equivalent.** For each model, an arm counts as equivalent to
  its x3 reference if all of these hold:
  - full-dev mean WER within **0.02 points**;
  - no dev set worse by more than **0.10 points**;
  - token mismatches on at most **0.5%** of dev utterances.
- **Otherwise it is a trade-off,** reported as WER cost (points, per set)
  against the energy saving (%).
- **No automatic switch.** Whether the deployment default changes from x3 is
  the user's decision, made after the report. If it changes, the new default
  gets its own one-time test scoring under a test-scoring plan like
  plans/test-scoring.md.

## Implementation

- **Agent B** (after the `b01` task) adds components 1 and 2 to `int8_mma` in
  new files. The frozen files stay byte-identical and the variants are
  registered by patch, as for `b01`.
  - B also tunes both models' configurations.
  - Contract tests: exact integer equality with M1's own `int8_matmul` at the
    same part count (bitwise, or one of M1's two rounding forms, as for x3),
    tails, split-K, CUDA graphs, the sanitizer, and the A5 audit.
- **Agent H** adds a full-dev transcription driver (`shared/fulldev.py`) that
  runs any registered runtime arm over the dev manifests at batch 1 with the
  graph cache. Its contract tests cover dev-split assertion, completeness,
  scorer parity with `evaluate.py`, and one GPU lock per process; it is
  resumable per set.
- **Review.** Codex reviews this plan and the code before the recorded runs,
  until CLEAN.

## Out of scope

Retraining for fewer parts, FP8 activations, test-set scoring of any new arm,
and the CPU (its benchmark already measures i8x2 and i8x3).

## Amendments (review round 1, 2026-10-20)

`/mnt/hd/wilderness-labs-stt/parakeet-binary/review/activation-precision/p1.md`.

**R1. One part count everywhere** (finding 1). A gpu1bit adapter
`PartCount(provider, n)` wraps the projection provider of the complete expanded
stack. That is M1's `ExpandedProjection` for M1 and H's `BinaryProjection` with
B's layers for MB1. The adapter overrides the `components` argument of
**every** projection call to n, including `layer_fusions4.py:72`'s hard-coded
`components=3` for the second pointwise convolution, and the stack's fusions
stay as they are.
- Instrumented contracts show that every call's effective part count is n, and
  that the fusion and dispatch coverage (call sequence, call sites) is identical
  to x3 on the eager path, during capture, on replay, and on the eager fallback
  after cache overflow.

**R2. Cross-check against exact pinned scores** (finding 2). Each model's
runtime-x3 arm is compared with its pinned FP32-NeMo full-dev WERs (MB1:
`runs/main-binB-m1-lr2e-4/summary.json`; M1: its main-run summary). It passes
only if the mean is within 0.02 points **and** every set is within 0.10
points, the same tolerances as plans/test-scoring.md.
- **On failure** (frozen now), in order:
  1. batch-1 FP32 NeMo decoding of the same model rebuilt from its export, to
     separate the runtime from batching;
  2. if that also misses, a historical-batch reproduction;
  3. a report to the user.
  No xN arm runs until the cause is known.
- **A passing check is a sanity check, not proof of fidelity.** Fidelity is
  reported separately as the token agreement of runtime-x3 with batch-1 FP32
  NeMo on the 512-utterance dev sample.

**R3. Bounded chunks with durable progress** (finding 3).
- Each utterance's record is appended durably to a per-arm JSONL log as it
  finishes.
- A worker process estimates its throughput from its first utterances and exits
  well inside the 10-minute lock hold: at most 8 minutes of work, then flush and
  exit.
- The next worker resumes from the first unfinished utterance ID.
- Contracts: a run killed mid-chunk and then resumed is complete, with each
  utterance exactly once; graph-cache overflow (eager shapes) is covered.

**R4. Immutable arm identity** (finding 4). Before any recorded run, H writes a
frozen identity per arm:
- the model export SHA-256;
- the stack (transform list and arguments) and the part count;
- the kernel registry name and the configuration tables' SHA-256;
- the SHA-256 of every imported experiment source, including the new b01 and
  part-count files outside the old freeze, the registry patches and
  `fulldev.py`;
- scorer dependencies (evaluate.py, normalizer, wer.py) and dev manifest
  hashes.

Every chunk and timing record carries its identity hash. Resume and report
refuse mismatches or mixed identities. Contracts test refusal after any change,
including to new files outside the old freeze.

**R5. Structural gates stay mandatory** (finding 5). Independent of the relaxed
numerics, an arm is **invalid**, not a trade-off, if any of these fail on any
utterance:
- encoder outputs are finite;
- encoder output lengths equal the x3 arm's;
- decoding completes without error.

Failure-injection tests (a NaN output, a wrong length, a decoder error) show
that each invalidates the arm.

**R6. No retuning** (finding 6; replaces "Equal tuning effort"). All arms use
their existing x3 tile configurations unchanged: M1's `expanded_configuration`
and B's frozen x3 tables. The measured change then isolates the part count. A
retuned xN row may be added later, labelled separately, under its own frozen
search.

**R7. Per-set token churn** (finding 7). "Recognition-equivalent" also requires
token mismatches on at most **1.0% of each dev set**, besides the pooled 0.5%.
Per-set mismatch rates are always reported.

**R8. Arithmetic oracle** (round 2 minor). B's xN contract compares against
M1's `int8_matmul` at **SPLIT = 1**, bitwise or within one of M1's two allowed
epilogue contraction forms, as the existing x3 tests do. Comparisons with M1's
deployed split-K configuration (floating partials) are reported separately and
are not a pass/fail oracle.
