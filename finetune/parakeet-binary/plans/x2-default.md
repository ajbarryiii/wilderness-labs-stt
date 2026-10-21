# Plan: INT8x2 activations become the deployment default

**Status: 2026-10-21: written; feasibility amendments F1-F6 (agent B); plan review round 1 NOT CLEAN (6 major, 2 minor), addressed in X1-X8; round 2 NOT CLEAN (3 major, 3 minor), addressed in Y1-Y6; round 3 NOT CLEAN (1 major, 2 minor), addressed in Z1-Z3, with H's harness notes Z4-Z6; **round 4 CLEAN** (review/x2-default/p4.md; its two minors fixed in Z7-Z8); later amendments supersede conflicting earlier text. Code review c1 NOT CLEAN (8 major), fixed; **code review c2 CLEAN** (review/x2-default/c2.md; its one minor fixed before the freeze).** User decision
2026-10-21, after the activation-precision report (plans/activation-precision.md,
run `ap-20261020r4`): "2x should be the default, given what we've seen here."

## Basis

On full dev at batch 1, x2 against x3 of the same model (`ap-20261020r4`):

| Model | Dev mean WER x3 | x2 | Energy x2 vs x3 (P-paired, full) |
| --- | ---: | ---: | ---: |
| M1 | 6.0856 | 6.0853 | −9.8% |
| MB1 | 6.3120 | 6.3124 | −11.6% |

Both x2 arms were recognition-equivalent under the frozen reading of that
plan. Those numbers come from the provider (R-expanded) stack; the deployed
MB1 runtime, B-hybrid (`hybrid_exact`), supports x3 only. This plan builds the
x2 hybrid, gates it, benchmarks the new default and scores it once on test.

## Scope

- **GPU, MB1:** new deployed arm **B-hybrid-x2**: B's exact deferred hybrid
  (`hybrid_exact.HybridProjectionExact` with `exact_hybrid_pipeline`) at two
  activation parts. Same INT32/int64 exact reduction, same tile tables
  (activation-precision R6: no retuning; a retuned row may follow later,
  labelled separately).
- **GPU, M1:** M1's files are frozen and its hybrid (`ptx_row_best`) supports
  x3 only. M1's x2 default is its R-expanded stack through `PartCount(·, 2)`,
  exactly the `ap-20261020r4` M1-x2 arm. No M1 hybrid-x2 is built; the report
  says so wherever M1-x2 is compared with B-hybrid-x2.
- **Kernel choice is not re-run.** B-hybrid was selected at x3 (sel-20261020b,
  anchored tie set). The x2 default keeps it. b01 at x2 is not built. The
  report states that the kernel ranking was established at x3 only.
- **CPU:** no change; plans/cpu-benchmark-mac.md already takes i8x2 as its
  primary condition.
- **Out of scope:** x1 (pending a noisy-dev check), retraining, FP8.

## Implementation

**Agent B** (owner of `gpu1bit/int8_mma/`), new files only; frozen files
byte-identical; registered by patch as for b01 and actprec:
- `components=2` on the exact deferred path (kernels, row path, pipeline).
- `PartCount` coverage (activation-precision R1) holds for the hybrid too:
  every projection call, including the hard-coded `components=3` of
  `layer_fusions4.py:72`, runs at 2, with call sequence and call sites
  identical to B-hybrid-x3 on the eager path, during capture, on replay and on
  eager fallback after cache overflow.
- Contract tests, written first:
  - arithmetic oracle (R8): M1's `int8_matmul` at SPLIT = 1, components = 2,
    bitwise or within one of M1's two allowed epilogue contraction forms;
  - per-projection parity with the provider-x2 arm (`int8_mma` under
    `PartCount(·, 2)`) under the same rule;
  - the deferred-path cancellation and boundary cases of `deferred_a3.py`
    (real and synthetic, complete coverage, per the GPU c3 eligibility fix) at
    x2: the deferred sum equals the non-deferred integer sum exactly;
  - tails, split-K, CUDA graphs, compute-sanitizer, A5 unique-storage audit
    (≤ 83,047,219 B).

**Agent H** (owner of `gpu1bit/shared/`):
- arm identity (activation-precision R4) for B-hybrid-x2;
- the dev gate, the test driver and the report below;
- updates to `selected.json`: a `default_parts: 2` field and the B-hybrid-x2
  arm, with B-hybrid-x3 kept as a named reference. The x3 selection record
  (report.json, report-corrected.json) is not edited.

## Gates (dev only, before any timing or test audio)

1. **Structural** (R5): finite encoder outputs, lengths equal to B-hybrid-x3's,
   decoding completes, on every dev utterance.
2. **Recognition,** full dev (22,418 utterances) at batch 1 through
   `shared/fulldev.py`, against the `ap-20261020r4` MB1-x2 provider arm
   (same arithmetic, different kernel), frozen now:
   - mean WER within **0.02 points**, every set within **0.10 points**;
   - token-ID mismatches on at most **0.5%** of utterances pooled and
     **1.0%** of each set;
   - encoder-output relative Frobenius on valid frames at most **1e-3** per
     utterance on the 512-utterance dev sample (A12).
   Exact integer parity predicts identical tokens; any mismatch is listed
   per utterance and explained in the report even when inside the bounds.
3. **On failure:** no timing and no test scoring for B-hybrid-x2; report to the
   user. Bounds are not loosened after seeing a result.

## Recorded GPU benchmark (replaces the pending x3-only recorded run)

After the gates pass and code review is CLEAN, H freezes the sources
(new `freeze-*.sha256`, every imported experiment file, including b01,
actprec, the x2 files and the patches) and runs one recorded session:

- **Arms:** B-hybrid-x2 (MB1 default), B-hybrid-x3 (previous MB1 selection),
  M1 R-expanded x2 (M1 default), R-hybrid x3 (M1's deployed runtime),
  R-expanded x3 (baseline).
- **Protocols:** P-paired (full and encoder scope, dev_24), P-clips (dev
  clips), P-memory (fresh processes), P-kernel, as in sel-20261020b; balanced
  order, 400 W check, locks; quiet-host agent-jobs run with the GPU exclusive
  (`--needs gpu --quiet-host --full-machine --priority normal`, authorized).
- **Then, after the freeze and the dev-workload results are written,** M1's
  historical test workloads (the LibriSpeech test clips and the 24 test
  utterances, A13) for comparability with M1's recorded numbers. They are
  reported separately, and nothing is selected or tuned on them.
- **Comparisons reported:**
  - each default against its own x3;
  - MB1-x2 against M1-x2;
  - MB1-x2 against R-expanded x3.

## One-time test scoring of the defaults

The x2 runtimes compute differently from FP32 NeMo, so the defaults are
scored once on test through their runtimes. The hygiene is that of
plans/test-scoring.md:

- **Arms:** MB1 on B-hybrid-x2, and M1 on R-expanded x2.
- **Corpus:** all 8 test sets, every utterance, batch 1, through a test mode
  of the fulldev driver. The mode is enabled only by a separate entry point
  with the output directory fixed in code. It inherits fulldev's resumable
  chunks (R3) and identity binding (R4).
- **Precheck** (read-only, must print ALL PASS):
  - the identity of both arms equals the frozen benchmark identity;
  - the scorer hashes are pinned (evaluate.py normalizer and WER, wer.py,
    normalizer.json, driver sources);
  - each test manifest's SHA-256 equals the one recorded in M1-test and
    b0-pretrained;
  - the normalizer reproduces M1-test's saved `ref_norm` and `hyp_norm`;
  - both gates above are recorded as passing for the same identity.
- **Attempt ledger** per arm (append-only: started / set_done with file hash /
  complete). Each arm is scored once, and any further attempt is refused.
  - A crash is recorded by hand with evidence, and the partial directory is
    moved aside.
  - After all sets reach set_done, only re-summarizing is allowed.
- **No `--limit` with test.** Smoke tests use dev only.
- **GPU:** accuracy only, so `--needs gpu:shared --vram 4G`. Run after the
  recorded benchmark so that its exclusive lease is not blocked.
- **Report:** `results/TEST.md` gains two rows, "MB1, runtime x2" and "M1,
  runtime x2", next to the FP32-NeMo rows.
  - Per set and seven-set mean, with Common Voice apart.
  - Each runtime row minus its own NeMo row.
  - Empty-hypothesis rates.
  - The label "runtime x2 − NeMo FP32 is the combined effect of INT8x2
    activations and the runtime's arithmetic; it is not an x2-vs-x3 test
    comparison". No x3 runtime arm is scored on test.
  - No decision depends on a test number.

## Write-up

- `README.md` gets a "Deployed runtime" line: the MB1 default B-hybrid-x2,
  with its test WER, energy and VRAM.
- `gpu1bit/RESULTS.md` gets the gate, the benchmark and the test-scoring
  sections.
- The x3 selection stays recorded as it was.

## Order

1. Plan review by Codex until CLEAN.
2. B and H implement, tests first. The c3 fixes of the GPU code review land
   first, because B-hybrid-x2 relies on the shared eligibility validator.
3. Code review by Codex of the whole gpu1bit tree, until CLEAN.
4. Dev gates (shared GPU lease).
5. Freeze, then the recorded benchmark (exclusive, quiet host).
6. Test scoring (shared lease).
7. Report.

## Feasibility amendments (agent B, 2026-10-21, before review round 1 returned)

- **F1. Parity oracle.** The x2 provider arm of `ap-20261020r4` is
  `int8_mma:x2` (B's hash-pinned `layer_parts`), not frozen `int8_mma`, which
  refuses x2. Per-projection parity is defined on identical projection inputs.
- **F2. Next-layer quantization.** M1's row epilogues (exact and fallback) are
  kept verbatim. They quantize the next layer's input into 3 parts, and the x2
  consumer reads parts 0 and 1. Because M1's quantizer is sequential, those
  equal M1's 2-part quantization. No 2-part row epilogue is built; it would be
  a speed-only change needing its own bitwise proof.
- **F3. Deferred exactness at x2.** The criterion is bitwise "deferred equals
  in-kernel x2" (X2 below). The x3-relative A3 limits do not apply, because x2
  fails them by design. `deferred_a3` gets a components parameter.
- **F4. Tuning.** `TUNED_EXACT_DEFERRED` (XCHG 5 versus in-kernel finish per
  shape) is kept as measured at x3 (R6); this is correct but may not be optimal
  at x2.
- **F5. Pinned part-count files.** The x2 modules live in `int8_mma/` and call
  `gpu1bit.actprec.parts.load_parts()` lazily as their first step, before any
  CUDA or model work, failing loudly on any hash mismatch.
  - Contract: importing the x3 modules (and importing the x2 module without
    using it) loads no `gpu1bit.actprec` module, checked in a fresh
    interpreter.
  - Contract: a tampered copy of a part-count file makes the x2 build raise,
    naming the file and both hashes, checked in a fresh process. The real
    files are verified unchanged afterwards.
- **F6. Token prediction.** The hybrid fuses the residual add, LayerNorm and
  quantization, while the provider stack runs them separately. The gate's
  expectation is therefore "near-identical tokens", not "identical tokens".
  At x3 the two gave a relative Frobenius difference of 9.8e-7 and 0 of 512
  token mismatches.

## Amendments (plan review round 1, 2026-10-21)

`/mnt/hd/wilderness-labs-stt/parakeet-binary/review/x2-default/p1.md`. These
supersede conflicting text above.

**X1. Hybrid-wide x2 coverage** (finding 1).
- **Mutable provider state.** `PartCount` forwards attribute writes, not only
  reads, to the inner provider, for every name except its own (`provider`,
  `n`, `record`). Hybrid contexts set `defer_epilogue` and `no_defer_weights`.
  - Contract: through `PartCount(HybridProjectionExact, 2)` the inner
    provider's state changes as on the unwrapped x3 provider, and each
    deferred call actually defers. The route each call takes (deferred XCHG 5,
    in-kernel finish, non-fused) is recorded and equals the x3 route table.
- **Parts actually consumed.** The 3-part buffers of F2 are allocated as
  before, and a poison contract checks that x2 consumers ignore the third part:
  - part 2 and its scale are overwritten with garbage (NaN scale,
    random codes) before each consumer;
  - outputs must be bitwise identical to the unpoisoned run, on both row
    paths (exact and fallback);
  - checked eagerly, under capture, on replay with changed input, and on the
    eager fallback after cache overflow.
- Every projection call's effective part count is 2 (R1), including
  `layer_fusions4.py:72`.

**X2. The fused floating-point epilogue is qualified directly** (finding 2).
Integer equality alone is not the contract. `test_exact.py` and
`test_exact_pipeline.py` are ported to x2:
- finished outputs and next-layer `q` / `qs` from the deferred path equal the
  same-stack non-deferred x2 path **bitwise**;
- this holds for every deployed flag combination and both deferred routes
  (XCHG 5 deferred, in-kernel finish), and for the M1 epilogue forms the x3
  tests cover.

Token agreement with the provider arm stays an empirical measurement (the
gate in X4).

**X3. Deployment defaults are executable** (finding 3).
- **MB1.** `gpu1bit.runtime.load_binary_packed(export)` and its CLI default to
  `selected.json` `default_parts: 2`. `parts=3` (CLI `--parts 3`) is the
  explicit override; any other value is refused.
  - The `Runtime` record carries `parts`, and its identity hash is the one
    fulldev computes for the benchmark arm.
  - Contract: a fresh process calling `load_binary_packed(export)` with no
    arguments gets B-hybrid-x2, with a runtime identity equal to the frozen
    B-hybrid-x2 arm identity. With `parts=3` it gets B-hybrid-x3 under its
    identity.
- **M1.** M1's files under `parakeet-ternary/` stay frozen; its
  `enable_optimizations` keeps x3 as M1's historical entry point. The x2
  default for M1 is a new loader, `gpu1bit.runtime_m1.load_ternary_packed(export)`,
  which builds M1's R-expanded stack under `PartCount(·, 2)`, with `parts=3`
  as the override. It has the same fresh-process identity contract.
  `parakeet-ternary/README.md` gets one line pointing to it.
- **Promotion.** `selected.json` (`default_parts: 2`, the B-hybrid-x2 entry,
  B-hybrid-x3 as the named override) is changed as part of the code under
  review, before the freeze. The gates and all measurements run on that
  frozen state, so the qualified identity is the deployed identity. If a gate
  fails, `selected.json` is reverted to x3 in a recorded commit, nothing is
  measured or scored, and the user is told. The x3 selection record
  (`report.json`, `report-corrected.json`) is not edited.

**X4. Fresh references under the final identity** (finding 4).
Historical rows are not reused as gate references.
- **Gate run.** After the freeze, one dev run (full dev, batch 1, `fulldev.py`)
  covers these arms under their final identities:

  | Arm | Model | Stack | Parts | Role |
  | --- | --- | --- | --- | --- |
  | g1 | MB1 | B-hybrid | 2 | candidate (MB1 default) |
  | g2 | MB1 | provider `int8_mma:x2` | 2 | reference for g1 |
  | g3 | MB1 | B-hybrid | 3 | R5 lengths, x3 context |
  | g4 | M1 | R-expanded | 2 | candidate (M1 default) |
  | g5 | M1 | R-expanded | 3 | reference for g4; R2 cross-check |

  The 512-utterance encoder comparison (relative Frobenius, A12) runs for
  g1 against g2 and for g4 against g5.
- **MB1 gate:** g1 against g2 with the bounds of "Gates" item 2.
- **M1 gate** (new):
  - g5 passes R2 against M1's pinned FP32-NeMo dev WERs (mean within 0.02,
    every set within 0.10);
  - g4 against g5 meets the recognition-equivalence reading of
    activation-precision (0.02 mean, 0.10 per set, at most 0.5% pooled and
    1.0% per-set token mismatches);
  - g4 is R5-valid.
- **Historical agreement, descriptive only.** Before the run, H pins in this
  plan the SHA-256 of `ap-20261020r4`'s per-arm row logs. g2 and g4 are
  then compared row by row with the M1-x2 and MB1-x2 rows of `ap-20261020r4`,
  read as data, verified against those pinned file hashes, and not
  revalidated against current sources. They are expected to be identical,
  since the code paths are unchanged. Any difference is listed and explained
  and does not gate.
- **Contracts:** refusal of a stale reference (identity frozen before the
  current sources), the wrong model's reference (M1 rows for an MB1 arm and
  vice versa), and a reference whose identity changed after its gate.

**X5. Explicit benchmark roster** (finding 5). H's existing `R-expanded` and
`R-hybrid` builders load MB1's derived export. The roster names model, export,
stack and parts for every arm, and the report uses these labels:

| Arm | Model (export) | Stack | Parts |
| --- | --- | --- | --- |
| b1 | MB1 (1-bit export) | B-hybrid | 2 |
| b2 | MB1 (1-bit export) | B-hybrid | 3 |
| b3 | MB1 (H's derived 2-bit layout) | R-expanded | 3 |
| b4 | M1 (M1 export `a287e977…`) | R-expanded | 2 |
| b5 | M1 (M1 export) | R-expanded | 3 |
| b6 | M1 (M1 export) | R-hybrid (`ptx_row_best`) | 3 |

Frozen comparison pairs:
- **Same stack, part count only:** b1−b2, b4−b5.
- **Change of deployed runtime:**
  - b1−b6, the MB1 default against M1's previously deployed runtime;
  - b4−b6, M1's new default against its previously deployed runtime;
  - b1−b4, both defaults;
  - b1−b3, the MB1 default against the selection baseline.
- **Reading.** If b4 is not lower in median paired energy than b6, the M1
  default is reported as slower than M1's deployed x3 hybrid, and the M1
  default goes back to the user. Nothing switches automatically.
- **Contract:** every arm's identity records the export SHA-256 and model
  name. The report refuses a label whose export doesn't match its roster row.

**X6. Resume within an attempt versus a failed attempt** (finding 6).
- **Same-attempt resume.** fulldev's chunked workers (R3) resume from the
  first unfinished utterance under the same attempt and identity. Completed
  rows and sets are kept. Resuming is not a new attempt.
- **Attempt lock.** An attempt lock (flock on the arm's attempt file) is held
  by the orchestrator for the attempt's life. Workers and recovery commands
  refuse to start without it, and a second orchestrator is refused.
- **Failed attempt.** An attempt fails only if resuming is impossible under
  the same identity, for example when a code fix is needed. It is recorded by
  hand with evidence, and its rows stay in place, read-only.
  - A new attempt then decodes everything again under its new identity.
  - Utterances the two attempts share are compared and reported.
  - This is allowed because no decision depends on a test number.
- **After every set is done,** only summary-only recovery is allowed (zero
  decoding).
- **Contracts:**
  - concurrent launch is refused;
  - a kill at each boundary (mid-utterance, between chunks, between sets,
    during the summary) resumes to a complete result, with each utterance
    exactly once;
  - a resume with a changed identity is refused;
  - a modified per-set file is refused through its hash;
  - summary-only recovery decodes nothing.

**X7. Memory and duration preflight** (finding 7). Before the freeze, H runs
each test arm (g1 and g4 builds) on synthetic and dev-derived inputs of the
longest test duration (105.7 s; dev clips concatenated, or padded audio).
- The peak reserved VRAM and the eager-fallback peak are recorded, and the
  lease is that peak plus max(25%, 1 GB).
- If any shape fails, scoring does not start.
- H also records the job duration budget (98,391 utterances at the measured
  batch-1 throughput by duration, plus chunk start-up costs) and sets the
  job timeout from it.

**X8. Label of the test deltas** (finding 8). The runtime − NeMo-FP32
difference is labelled as the combined difference of the deployment path:
- INT8x2 activations;
- the runtime's integer and fused arithmetic;
- batch 1 against the historical batch 64;
- the runtime's decoder execution (full CUDA-graph mode against NeMo's
  historical mode).

It is not an x2-versus-x3 test comparison.

## Amendments (plan review round 2, 2026-10-21)

`/mnt/hd/wilderness-labs-stt/parakeet-binary/review/x2-default/p2.md`. These
supersede X3's promotion rule, X4's roster and order, X5's reading, X6's
failed-attempt rule and X7's preflight wherever they conflict.

**Y1. Qualify first, then promote the same bytes** (finding 1).
- **Live default unchanged until qualification.** The live `selected.json`
  keeps x3 (`default_parts` absent or 3) until every gate has passed.
  - The candidate selection is a separate file, `shared/selected-x2.json`. It
    names the B-hybrid-x2 entry, B-hybrid-x3 as the override, and the M1 x2
    default.
  - Both loaders take an explicit `selection=` path. The gates, the benchmark
    and test scoring pass `selected-x2.json` explicitly.
- **Identity.** The identity hashes the **content** of the selection document
  in use, not its file name.
- **M1 loader.** `runtime_m1.load_ternary_packed` reads its default part count
  from the same selection document (`m1_default_parts`). Before promotion,
  unqualified M1 calls also get x3.
- **Promotion.** After the gates pass and the benchmark is written, promotion
  replaces `selected.json` with a byte-identical copy of `selected-x2.json`, in
  one recorded change.
  - Contract after promotion: a fresh-process unqualified call to each loader
    has an identity equal to the qualified arm's (b1 for MB1, b4 for M1).
  - Contract before promotion: a fresh-process unqualified call to each loader
    gets x3.
- **Failure.** If any gate fails, nothing is promoted, and both loaders stay
  at x3 for unqualified callers. `selected-x2.json` stays on disk, marked
  failed with its gate record, and the user is told.

**Y2. MB1 provider-x3 reference first** (finding 2).
- **Roster.** The gate run gains **g0: MB1, provider `int8_mma`, x3**, which is
  the activation-precision MB1-x3 arm under the current identity.
- **Order: references first, for each model.**
  - MB1: g0 passes R2 against MB1's pinned FP32-NeMo dev WERs (mean within
    0.02 points, every set within 0.10), then g2 runs, then g1 and g3.
  - M1: g5 passes R2, then g4 runs.
- **Contract.** `require_reference` refuses to launch g2 (or g4) when its
  reference arm is missing, incomplete, of another identity, of another model,
  or failed R2. No caller bypasses it.

**Y3. An unrecoverable test failure stops** (finding 3). This replaces X6's
"Failed attempt" item.
- **What happens.** An attempt that cannot resume under its identity is
  recorded with evidence, its rows are kept read-only, and scoring stops. The
  user is told.
- **Exceptional retry.** A new attempt is allowed only with all of these,
  verified by the precheck:
  - the user's approval;
  - a dev-only reproducer of the failure;
  - a fix reviewed CLEAN by Codex;
  - a new freeze, gates passing under the new identity, and a replacement
    benchmark (b1 to b6) bound to it.
- **Disclosure.** Every attempt is preserved and reported. Shared utterances
  are compared across attempts.
- **Contract.** A new attempt is refused without the replacement evidence, and
  refused when that evidence is bound to another identity.

**Y4. Descriptive reading of the defaults** (finding 4). This replaces X5's
"Reading". Every comparison pair reports median paired energy and median
latency separately, with no automatic consequence, and scoring continues once
qualification passes. The report states plainly where M1's new default (b4)
uses more energy, or has higher latency, than M1's previous deployed runtime
(b6).

**Y5. The attempt lock survives orchestrator death** (finding 5).
- **Lock ownership.** The attempt lock is a flock on an open file description
  that the orchestrator passes to every worker (`pass_fds`). The lock is held
  as long as any holder process is alive.
- **Recovery waits for the workers.** A new orchestrator or recovery command
  can take the lock only when no holder is alive. It also checks the recorded
  worker PIDs and start times, and refuses while any of them is still running.
- **Contract.** Kill the orchestrator while a worker is mid-chunk:
  - reacquisition is refused until the worker exits;
  - then the resume is complete, with each utterance exactly once.

**Y6. Qualify the worker's peak and time limits** (finding 6). This replaces
X7's memory measurement.
- **Memory.** For each test arm (g1 and g4 builds), one preflight process
  runs, in this order:
  - four distinct near-maximum shapes, which fill the 4-entry graph cache
    (durations 105.7 s, 100 s, 95 s and 90 s);
  - then a 105.7 s utterance on the eager fallback while the cache is full;
  - full decoding throughout.

  The peak reserved VRAM of that process sets the lease: peak plus
  max(25%, 1 GB).
- **Time.** The same preflight records the cold start-up plus capture time.
  The longest single-utterance time must leave the worker's 600-second limit
  (R3's 8-minute work budget) satisfiable with at least one utterance per
  chunk at 2× margin. Otherwise scoring does not start and the chunk budget is
  replanned under review.

## Amendments (plan review round 3 and harness notes, 2026-10-21)

`/mnt/hd/wilderness-labs-stt/parakeet-binary/review/x2-default/p3.md`, plus agent
H's notes on the harness. These supersede conflicting earlier text.

**Z1. Only a structurally valid utterance counts as done** (round 3, finding 1).
Test scoring only; the dev gates keep R5.
- **What counts as completed.** An utterance in the test entry point is
  completed only if all of these hold:
  - decoding returned without an exception;
  - the encoder output is finite;
  - the encoder length equals the frontend's expected length.
- **Failures.** A failed utterance writes a record (exception, location,
  finiteness, identity) to a separate per-arm failure log. It is not scored
  and not written as a row.
- **Same-identity recovery.** Each failed utterance is retried once, in a
  fresh worker under the same identity, which covers transient CUDA errors.
  A second failure of the same utterance is unrecoverable and goes to Y3: stop
  and tell the user.
- **No completion while failures remain.** `set_done`, `complete` and the
  report refuse while any unresolved failure exists in the attempt.
  Recovered failures stay in the log and are reported.
- **Contracts,** through the test entry point and the report: injected decoder
  exception, NaN encoder output and wrong encoder length; transient
  (recovered) and persistent (stop) cases; refusal of `set_done`, `complete`
  and the report.

**Z2. Frozen candidate selection and the deployment pointer** (round 3,
finding 2).
- **What the freeze snapshot holds.** It includes `shared/selected-x2.json`
  (the frozen candidate) and excludes the mutable pointer
  `shared/selected.json`.
- **The pointer is checked separately.** `verify_freeze` and the test precheck
  accept `selected.json` only in one of two states:
  - byte-identical to the pre-promotion x3 document, whose SHA-256 is pinned
    at freeze;
  - byte-identical to `selected-x2.json`.
  Anything else is refused.
- **Contracts:** benchmark revalidation, report regeneration and the test
  precheck succeed both before and after promotion without rewriting any
  recorded evidence, and are refused when `selected.json` is edited to any
  third state.
- **Identities and fingerprints.** All arm identities and gate fingerprints
  hash the content of the selection document in use (Y1). A contract checks
  that promotion leaves the b1 and b4 identities and the gate fingerprints
  unchanged.

**Z3. The preflight's eager step is really eager** (round 3, finding 3). This
replaces Y6's shape list.
- **Shapes.** Four distinct shapes of 104, 100, 96 and 92 s fill the 4-entry
  cache. Then the 105.7 s maximum, which is not cached, runs on the eager
  fallback while all four graphs are held.
- **Assertions:** the cache holds 4 entries before the fifth input, and the
  fifth input's recorded route is eager. Otherwise the preflight fails.

**Z4. Descriptive reference to `ap-20261020r4`** (H). It uses a "recorded
arm" binding that never gates. The binding does not recompute the identity
from current sources. Instead it checks that:
- the frozen `identity.json` hash equals the summary's and every row's
  identity;
- the arm is complete over the full subset and its x3 reference passed R2;
- the row files' SHA-256 equal the hashes pinned in this plan before the run;
- their ctime is no later than the arm's last chunk.

A cross-run comparison stores both runs and both identities. Contracts refuse
a modified row file, a mixed identity and an incomplete arm.

**Z5. Deferred-exactness coverage is specific to each criterion** (H).
- **At x3:** the existing requirement, including `discriminating_cases > 0`.
- **At x2:** complete coverage of the same expected real and synthetic case
  set as x3, cancellation and boundary cases included, and deferred equal to
  in-kernel x2 bitwise on every case. There is no `discriminating_cases`
  requirement.
- **Contract:** an x2 report missing any expected case, or with one inexact
  case, is refused.

**Z6. Runtime contracts at x2** (H).
- **Recognition.** For the x2 default, `test_runtime`'s recognition
  reference is the provider x2 arm (g2): text, token IDs and timestamps on
  the 24 timing workloads. Equality with x3 is reported as a hypothesis, not
  asserted.
- **Memory.** `test_runtime_fresh`'s VRAM reference for B-hybrid-x2 is the
  new benchmark's P-memory of b1.
- **Provider.** Inside `PartCount`, the hybrid x2 path uses B's new x2
  provider class, because `HybridProjectionExact` refuses x2. R1's coverage
  assertions apply to it.

## Amendments (plan review round 4 minors, 2026-10-21)

**Z7. Timestamp precision in Z4.** The driver records chunk `finished` times
truncated to whole seconds. The ctime cutoff is therefore
`finished + 1 s`, exclusive, and the pinned content hashes remain the
primary check. A contract test covers the existing case: M1-x2's last chunk
is recorded at 21:49:36, and `yodas_dev.jsonl` has ctime 21:49:36.79, which
is accepted.

**Z8. When the benchmark-relative runtime test runs.**
- **Before the freeze:** Z6's `test_runtime_fresh` comparison and refusal
  logic are tested on fixtures.
- **Against real data:** the assertion against b1's recorded P-memory runs
  after the recorded benchmark and before promotion and test scoring, on the
  unchanged frozen sources.
- **On failure:** if it fails, nothing is promoted or scored, and the user is
  told.

The order becomes:
1. gates;
2. freeze;
3. recorded benchmark;
4. benchmark-relative runtime tests;
5. promotion;
6. test scoring;
7. report.

## Pinned recorded-arm files (Z4; agent H, 2026-10-21, before the gate run)

SHA-256 of the `ap-20261020r4` files that the descriptive comparison (Z4, Z7) reads, relative to
`/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/H/fulldev/`: the per-set row logs, identity, summary and
chunks of M1-x2 and MB1-int8_mma-x2, and the identity and summary of their x3 references.
`gpu1bit/x2/recorded_arm.py` reads this block; it is not edited after the gate run starts.

```pinned-ap-20261020r4
c8571e77a5018255fbc171f241cb106bda9cbd53d2c0d435896e414d21b0ecd1  ap-20261020r4/M1-x2/librispeech_dev_clean.jsonl
ca6abb9aa3c36615745aa4b0f38f3407fc7bef85c896f3f590fe401876f9ee02  ap-20261020r4/M1-x2/librispeech_dev_other.jsonl
75ca63a329956041385f6de31f2e668ed1eb41f9b8a5cf197978e28e086bb75f  ap-20261020r4/M1-x2/ami_dev.jsonl
b2fd21a54b11840b59160254530d51e36f5a9e3e3623fb97750d1d6f8426e23a  ap-20261020r4/M1-x2/voxpopuli_dev.jsonl
8219df793a0b8936498be7ea98666c42e7766caf391c6c89cd38c787a56e1ebd  ap-20261020r4/M1-x2/yodas_dev.jsonl
c31c9740c6f3f6c0cdef927f7f12b9ea1212901270b84410174b67be92a0eaca  ap-20261020r4/M1-x2/identity.json
a3c33a23d5bd204efd940aa2fe4fc70752e968c6dc57540a8b3e04c779e779a1  ap-20261020r4/M1-x2/summary.json
1f69b844c9e38fb8f908281abc3e37b403e7c47cb843620d61010f7c4a1a27fc  ap-20261020r4/M1-x2/chunks.jsonl
ddbeea400fddde4a3db3b8db6e46eacd0c25c9dc168c747514da44cecb538a16  ap-20261020r4/MB1-int8_mma-x2/librispeech_dev_clean.jsonl
cd428690191991a3f7e6659db44c4fb25b6f026e6f31e6b299e3b7bbb7f488d1  ap-20261020r4/MB1-int8_mma-x2/librispeech_dev_other.jsonl
4c18604075649bb3d0f4a1eae2b94dc15ee8152c14aff38296c126f0ffb4007f  ap-20261020r4/MB1-int8_mma-x2/ami_dev.jsonl
5df4207e39a0acc1e780a57ef2ce5572cb202e6f2cfce5739f7d8a4e92e5f77a  ap-20261020r4/MB1-int8_mma-x2/voxpopuli_dev.jsonl
1565df2f4b1179940150006cac113f3e17b4557aaddb44ae4a9e5976fa764ae9  ap-20261020r4/MB1-int8_mma-x2/yodas_dev.jsonl
d74e98c974a1c49368f0d9b79044e9be3c70cb06869ea64648381bae6946165a  ap-20261020r4/MB1-int8_mma-x2/identity.json
4d9590b4a88d454a5724f4e483e7db1f25a28d9c443634d9463d5fd4a21c8906  ap-20261020r4/MB1-int8_mma-x2/summary.json
0a682b4e08a7f411b75c346093c43b32eb75801ad3ac2bbc895f0d6f1017a8f0  ap-20261020r4/MB1-int8_mma-x2/chunks.jsonl
6dc05673654cc40d61bf4cda6c19ca9778a05f111cb506a289772780a27f3777  ap-20261020r4/M1-x3/identity.json
9b8e799f472b18917a311115c71d3acdc4aa9f2a3697b4a02f21f98d7fb5baf2  ap-20261020r4/M1-x3/summary.json
d36b8eec10afde8f515e2160c60e6e2eda9cc3246c584a3048b23d779950c70b  ap-20261020r4/MB1-int8_mma-x3/identity.json
fb880237aeedf9c30da09976fa8d65afdf2af531fc2d54e49b23db5602b7dc2f  ap-20261020r4/MB1-int8_mma-x3/summary.json
```
