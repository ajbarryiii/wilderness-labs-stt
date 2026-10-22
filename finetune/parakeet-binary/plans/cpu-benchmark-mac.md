# Plan: native 1-bit CPU kernels for MB1 on Apple M1 Pro (NEON)

**Status:** **plan review p10 CLEAN** (2026-10-21,
review/cpu-benchmark-mac/p10.md; p8 and p9 findings resolved). **Code review c6
CLEAN** (2026-10-21, review/cpu-benchmark-mac/c6.md; c4 and c5 findings
resolved), on the dev copy published as cpu1bit/shared (smoke job
cpu1bit-H-smoke-170127). Consolidated
2026-10-21; supersedes the earlier text and its
amendments M1–M16, kept verbatim with the review trail in
[cpu-benchmark-mac-history.md](cpu-benchmark-mac-history.md). It replaces the
Ryzen plan (plans/cpu-benchmark.md) at the user's request. **User decisions:**
baseline and candidates are built and run on the Mac; CPU energy comes from
exactly `powermetrics --samplers cpu_power -i 100 -f plist`, which needs
non-interactive sudo for powermetrics (see the operator setup); timed runs
happen overnight as `job start
--host mac --quiet-host --full-machine --priority normal`.

**Review rule:** a major finding is either a code bug, or a plan defect that
could plausibly change a reported conclusion at effect sizes of 1% or more.
Stated, evidenced and checked assumptions are not findings.

## Question and expected outcome

On the M1 Pro's CPU, does a native 1-bit kernel run MB1's encoder projections
on less energy, or faster, than MB1 through a strong NEON port of LocalFlow's
2-bit method? Energy per replay is the primary metric (selection and
confirmation); latency is reported descriptively.

The expected answer is "no clear difference":
- the V and T agents' instruction models put both within about −6% to +5% of
  B-NEON;
- P's model has bit-serial losing by 2–3× (cpu1bit/NOTES.md).

Ties or descriptive-only verdicts are the likely outcome. The report's main
value is the measured numbers: energy and latency per arm, condition and clip,
weight bytes, and exactness.

## Scope

**Measured unit: the projection replay.** For one utterance, the encoder's 240
projection GEMMs in LocalFlow's order, on fixed captured FP32 inputs.
- Each call quantizes per frame into i8x2 or i8x3 residual parts, then runs the
  GEMM, then the FP32 epilogue, as LocalFlow does.
- **Schedule:** frozen in `cpu1bit/shared/schedule.json`, derived from the x86
  LocalFlow snapshot's `encoder.rs`.
  - Q, K and V are separate GEMMs that share one quantization (192
    quantizations in all).
  - linear_pos is cached after warmup and neither replayed nor timed.
  - Trace contracts assert the call count, shapes, reuse, capture layout and
    output order.
- **Claims:** conclusions are restricted to this replay. Removing the other
  operations changes cache state, synchronization and DVFS, and the captures
  follow the FP32 NeMo trajectory. No encoder-level or transcription-level
  saving is claimed.

**Hardware:** 8 P-cores in two clusters of 4, plus 2 E-cores. NEON with DotProd,
no I8MM. macOS has no thread affinity; QoS is the only lever.

## Arms

| Arm | Weights | Kernel |
| --- | --- | --- |
| R-MB1 (baseline) | MB1 via the exact derived 2-bit export | B-NEON |
| R-M1 | M1, 2-bit | B-NEON; reference for MB1 vs M1 |
| B1T-MB1 | MB1, 1-bit codes expanded into B-NEON's tile | descriptive: storage vs kernel |
| V | MB1, 1 bit | in-register expansion of bit groups, then `SDOT` |
| T | MB1, 1 bit | lookup tables (`TBL`), INT16 entries, INT32 accumulation, proven bounds |
| P | MB1, 1 bit | bit-serial `EOR`, `CNT`, accumulate (control) |

Any agent may propose a per-shape hybrid, backed by data, before the roster
freezes; it is then an arm like V, T and P.

**B-NEON.** Agent H writes it as a port of LocalFlow's method: per 64-row block,
the codes are expanded once into an L2-resident s8 tile, then multiplied with
`SDOT`, with NEON register blocking. Codex reviews it for being a strong
baseline, not a strawman.
- **Tuning budget:** 6 grid points on dev clips, as for every candidate. Every
  arm's tuning point is frozen before screening.
- **Before freezing:** the assembly is inspected for spills, and B-NEON is
  compared with a pre-expanded SDOT control (all weights stored as s8). A gap
  above 25% blocks the freeze until it is explained.

**Exactness (zero tolerance; required before an arm is timed).** Every arm's
integer dots and FP32 outputs must be bit-identical to an independent scalar
oracle that uses LocalFlow's quantization and epilogue order. The contract
tests cover ragged shapes (K not a multiple of 64, N not a multiple of the
tile, T = 0, T = 1, odd T); extremal activations and sign patterns; i8x2 and
i8x3 at 1, 4 and 8 threads; reuse with changed inputs and lengths; native codes
equal to derived codes; full-schedule trace and dispatch (no fallback); and
x86 golden vectors (ties, FMA-sensitive updates, zero, tiny and non-finite
frames: component bytes, scale bits and output bits). Real-capture exactness on
the timed clips and on a 16-utterance dev sample is recorded per arm.

**Storage and workspace (acceptance; plans/cpu-benchmark.md rule, adapted to
the replay).**
- **Weight bytes.** After loading and warmup, the replay process's counting
  allocator gives the live heap allocated since before loading. Everything in
  it that is not the harness's own buffers (counted exactly by the replayer),
  the kernel's declared workspace, or per-row values (scales, row sums; at most
  16 bytes per output row, reported) counts as weight bytes: packed codes,
  padding, expansions or index panels kept between calls, auxiliary tables,
  caches, and an export buffer not released.
- **Cap.** Arms that load the 1-bit export (V, T, P, B1T-MB1, hybrids): at most
  1.10 × 72,351,744 bytes, the packed 1-bit size of the 240 replayed matrices
  (the inherited 75,497,472 bytes counted all 264 projections; the 24 linear_pos
  matrices are not replayed). Arms on the 2-bit export and the pre-expanded
  control are reported, not capped.
- **Workspace.** Per-call tiles, tables and per-slot activation state, declared
  by the kernel: at most 32 MiB in total at 8 workers, for every arm; the peak is
  reported. Weight-derived data kept across calls is weight bytes, not workspace.
- **Preparation stays timed.** `prepare` (untimed, at load) only builds the
  stored form. Any per-call preparation (tables, planes, tile expansion) runs
  inside `gemm`, i.e. inside the timed replay, as does the shared quantization.
- **Check.** Kernels allocate only through the Rust global allocator (code
  review). The block process emits the audit after warmup and exits non-zero on
  a breach, which invalidates the block; an arm must pass the audit in an
  untimed run before its freeze.

## Workloads and captures

**Clips** are dev only, frozen and hashed in `cpu1bit/shared/clips.json`: 8
timed clips at targets of 16 s and 28 s, round-robin over the 5 dev corpora,
and 16 exactness utterances. No test clips go to the Mac.

**Captures** hold every quantization input, from the FP32 NeMo reconstructions
of M1 and MB1 on the NixOS GPU, transferred hash-checked. Only the timed
captures stay resident (about 3 GB per model); the exactness sample is
streamed, and transfer refuses if under 20 GiB would remain free.

## Measurement

**Process and threads.**
- Each block is a fresh process holding one arm.
- Workers: 8 (primary condition) or 4, at USER_INTERACTIVE. Each thread sets
  and reads back its QoS. Process and thread darwin-background state must be
  0, because a background process runs on the E-cores whatever the thread QoS;
  a mismatch aborts the block.
- Cluster residency is reported descriptively, and no P-core-only claim is
  made.
- Conditions: {8, 4 workers} × {i8x2, i8x3}; the primary is 8 workers, i8x2.

**Block protocol.**
1. Wait for a quiet host:
   - no user input for 10 min;
   - AC power and battery at 100%;
   - thermal nominal, Low Power Mode off;
   - load average < 1.5 and memory pressure normal;
   - no other agent job running.
2. Under the exclusive Mac lock and `caffeinate -i -s`:
   - spawn powermetrics, with s = CLOCK_UPTIME_RAW read just before the spawn;
   - start the 1 s state sampler;
   - run a 10 s idle window, which gives the idle reference; P-cluster busy
     share must be ≤ 0.10.
3. Run the arm process: load, then warmup (the longest clip twice, then every
   clip). Then per clip:
   - load the clip;
   - one untimed replay;
   - wait S + G, workers parked;
   - the timed window W = [t0, t1] of exactly N replays, with t0 and t1 read
     just before the first and after the last replay returns;
   - wait G + S′, workers parked.

   Window events are emitted only after the wait, so no runner activity is
   time-locked to W; the analysis rejects a window whose events arrive inside
   [t0 − G, t1 + G]. One clip's captures are resident at a time.
4. Read powermetrics until every window is bracketed (see A1), then stop it by
   closing the pipe.

**Constants:** G = 1.0 s, S = 1.2 s, S′ = 1.0 s, K = 2, L ≤ 130 ms per sample,
A_MAX = 600 s, U_MAX = 0.6 s, ρ = 50 ppm. G, S and S′ each exceed
U_MAX + 2ρ·A_MAX + the sample and transient margins.

**Fixed N and sizing.**
- N per condition and clip is the same for every arm:
  N = ⌈12 s / (0.95 · fastest arm's per-replay latency)⌉. Slower arms get
  longer windows.
- The latencies come from a sizing stage per condition: one block per arm, one
  replay per clip, never compared.
- **Preflight:** every arm's planned block must fit within A_MAX − 60 s. An
  infeasible condition stops before screening and is reported; fixing it is a
  plan change.
- N is frozen, with the sizing record's SHA-256, in the session file and in
  every stage fingerprint.
- Windows that later run shorter than 12 s are flagged, not invalidated.

**Host-state gates.** Checked each second from spawn to the end of the
capture (idle window, warmup, windows and the waits between), with bracketing
samples; any of these invalidates the block attempt, and the records name the
windows whose Π, extended by U + L, it touches:
- AC, battery or thermal transitions, Low Power Mode, user input;
- memory pressure above normal, or any increase in pageouts or swapouts;
- foreign CPU above 0.5 cores, after adding a 0.1-core tick margin;
- any probe failure.

Foreign CPU = host busy time (tick deltas) minus our own (`proc_pid_rusage` for
live processes, plus RUSAGE_CHILDREN for reaped ones minus what was already
attributed). Negative deltas, unreaped vanished processes and own time above
host time are errors.

### Energy

**Measurand.** Gross powermetrics-estimated CPU-domain energy over the padded
interval Π = [t0 − G, t1 + G], divided by N. It is neither process-isolated nor
whole-device energy.
- With e = E_W/N and g = E_guard/N, the paired change is
  (e_C + g_C − e_B − g_B)/(e_B + g_B). With equal guard energy (same protocol
  and same N), that is the unpadded change times e_B/(e_B + g), about 0.99.
- Arm-specific guard transients are work-induced and counted.
- No idle-corrected or net energy is computed (the Ryzen plan's E_net is
  dropped). Instead each window reports, as separate columns, the gross padded
  energy per replay, the guard energy per replay and the window duration, so a
  reader can apply any correction. Guard energy is the mean power of the
  samples lying wholly inside a guard ([t0 − G, t0] or [t1, t1 + G]) for every
  admissible offset, times 2G; it is descriptive and no statistic uses it.
- The edge and work checks below are threshold-based disturbance checks: finite
  padding neither captures every transient nor proves the absence of residual
  neighbouring energy. Replay completion relies on the pool's completion
  barriers.

**Mapping.** Sample k has elapsed el_k, cumulative elapsed c_k since spawn, and
delivery time d_k on our clock.
- Under A1–A3 below, its end is s + o + c_k(1 ± ρ), with one offset o in
  [−δ_A2, U].
- U = min_k(d_k − s − c_k(1 − ρ)), because a sample is not delivered before it
  ends.
- δ_A2 is the pre-spawn residual from the A2 test.
- U < 0 invalidates the block, and U > U_MAX invalidates the window.

**Samples summed.**
- P_out: the samples that can meet Π under some admissible offset.
- P_in: the samples inside Π under every admissible offset.
- X = P_out ∖ P_in: counted from the mapped intervals, and reported.

**Window validity.**
- Coverage: P_out lies in one run of parsed records, every sample is ≤ L, and
  the window ends before A_MAX.
- Edges: every sample of X, and the K outermost samples of P_in on each side,
  are idle-level: excess ≤ tol, with tol = max(5 mJ, the idle window's largest
  deviation + 3 mJ).
- Work present: at least one sample that lies inside W for every offset
  exceeds tol.

**Bounds.** With q = 1 mJ, the integer reporting granularity:
- E_upper = Σ_{P_out}(E + q);
- E_lower = Σ_{P_in}(E − q);
- E_est is prorated at the mid-offset.

Under A1–A3 they bracket the measurand. Their width is about 0.5–0.7% of a
window. They serve as a cheap deterministic sensitivity in the gate, not as a
guarantee beyond A1–A3.

**Recorded per window:** E_lower, E_est and E_upper per replay, guard energy
per replay, window duration, latency (duration/N), U, |X|, residency, foreign
CPU, and every invalidation reason.

### Assumptions A1–A3 and validation

The energy result is a paired statistical estimate under three assumptions.
Each is stated, backed by evidence, and checked in validate before any timed
stage.

**A1 (producer continuity).** powermetrics emits every sample, contiguously, and
each record's energy and elapsed_ns cover the same interval.
- **Evidence:** job `cpu1bit-H-plist-fields2`: 7,061 records over 2,213 s. The
  band of (timestamp − Σ elapsed_ns) was 1.001 s and 1.000 s wide, with no
  drift; no discontinuity was detected by this check.
- **Deterministic:** powermetrics writes to a blocking pipe, every
  NUL-terminated chunk must parse, and EOF or a producer exit invalidates the
  block. Only the chunk cut by our own stop is exempt.
- **Per block:** the band must be ≤ 1 s + 50 ms. This catches a skip only when
  sample lengths populate both band edges (ours are 106–119 ms). With exactly
  100 ms samples, a skip is undetectable.
- **Per window (flag only):** (d_last − d_first) is compared with Σ el.
- **Bracketing:** a window is invalid if the capture ends before the record
  whose earliest start exceeds t1 + G.

**A2 (post-spawn origin).** The first record starts at or after spawn.
- **Validate:** 20 trials, each with a ≥ 500 ms spin ending at the spawn call;
  the first record must be idle-level.
- This yields δ_A2 = tol / spin power.
- It catches a first interval starting 100 ms before spawn, which U ≥ 0 cannot.

**A3 (constant rate within a block).** elapsed_ns follows CLOCK_UPTIME_RAW to a
constant |ρ| ≤ 50 ppm.
- **Evidence:** powermetrics imports mach_absolute_time, mach_continuous_time
  and mach_timebase_info, i.e. the same kernel timebase. Its energy comes from
  IOReport deltas between the same samples.
- **Check** (validate's 10-minute capture, and every block): for every prefix
  k, |(d_k − d_1) − (c_k − c_1)| − J ≤ ρ(c_k − c_1). J is the delivery jitter:
  the range of d − c within 10 s stretches.
- A failure in validate stops all timing; in a block it invalidates the block.
  A cancelling ±100 ppm drift fails this check.

**Validate also records:**
- sample intervals, U and the first-sample offset;
- J, the idle reference and the P-cluster idle share;
- an energy-interval check: no excess may appear in samples that end, under
  every offset, before a known spin starts;
- one R-MB1 instrument block, with its cadence and bound widths.

**Validate passes if:** all 8 windows of the instrument block are valid, U ≤
U_MAX everywhere, the median relative bound width is ≤ 1%, and every test
passes. Otherwise nothing is timed until the constants are revisited, with
numbers, as a plan change. Validate is instrument-only; no statistic reads it.

**Invalidation and records.**
- An invalid window invalidates its block attempt, and the whole sequence
  attempt is re-measured, with the same replicate and sequence IDs and a new
  attempt number.
- Up to 4 attempts per pass and 2 passes; then the stage exits incomplete and
  analysis refuses it.
- Records are written once, atomically.
- stage.json fingerprints the configuration, binaries, exports, captures, N,
  the harness sources and the validation. A resume refuses any change.
- Arms are paired only within one complete, valid attempt.

## Design and selection

**Stages.**
- Tuning (6-point grid per arm), single-run command stages such as P's ISA
  microbenchmark, and B-NEON's gap check run first, in the primary condition;
  none is a comparison.
- Screening: R = 1 Williams replicate (k sequences for even k, 2k for odd)
  over the roster, frozen beforehand (no arm is added later), in every
  condition; only the primary selects, and no significance is claimed.
  - Statistic, per candidate against R-MB1: the median over the 8 clips of the
    per-clip mean paired % change in E_est per replay.
  - The candidate with the lowest (statistic, name) is confirmed if its
    statistic is negative. Otherwise the baseline wins, nothing is confirmed,
    and the best candidate is reported descriptively.
- Confirmation: R = 6 fresh Williams replicates of R-MB1 against that candidate
  (k = 2), in the primary condition; the screening data are not reused. R is
  fixed; there is no adaptive extension.
  - Interval: the frozen 95% percentile interval of a crossed bootstrap, i.e.
    the 2.5th and 97.5th percentiles of 10,000 draws, seed 20261020. Each draw
    resamples the replicate IDs and one shared clip vector with replacement,
    keeps the pairing, takes each sampled clip's mean over the sampled
    replicates, then the median over clips.
  - The gate interval uses candidate E_upper against baseline E_lower. It is
    the only inferential claim; with one confirmed comparison there is no
    multiplicity.

**Rejections: the +50% envelope.**
- Each rejected confirmation sequence attempt is imputed as an extra sequence of
  its replicate, with the candidate 50% worse than its baseline.
- An improvement is declared only if the imputed interval's upper bound is
  below 0 and there are at most 2 rejected attempts.
- This is a sensitivity analysis under the stated envelope, with no protection
  claimed beyond it.
- Acceptance is reported by arm, by position and against host state. Three
  conditions are flags only: block acceptance below 95%, Fisher p < 0.10 for
  arm or position, or a gap above 5 points.

**Simulation check.**
- The confirmation procedure, with retries, imputation and the cap, is
  simulated on null data resampled from the screening's variance components.
- Rejection rates r ∈ {0, r̂_U/4, r̂_U/2, 3r̂_U/4, r̂_U}, where r̂_U is the
  one-sided 95% Clopper–Pearson upper bound of the confirmation's
  sequence-attempt rejection rate. Rejection is either independent or
  adversarial (the attempts least favourable to the candidate are rejected).
- **Pass:** the maximum false-improvement rate over the grid and both models is
  ≤ 3.5%, and coverage at r = 0 is ≥ 93% (support for the directional gate,
  not two-sided coverage of imputed intervals). Otherwise the result is
  descriptive only. The report gives the whole grid.
- The error control holds for this generating model only.
- **Stress family (reported, not gating):** rare +45% outcomes with
  p ∈ {5%, 10%}, always rejected, mean change 0. About 28% of confirmations at
  p = 10% see no rejection at all.

**Verdict conditions.** The verdict text states:
- A1, A2 and A3;
- that rejections come from the disturbance gates;
- the +50% envelope;
- the simulation's generating model;
- the stress-family limit.

The non-primary conditions (4 workers; i8x3) are descriptive only, with no
interval or claim.

## Team, code and process

H owns the harness, captures, oracle, B-NEON, runner and analysis; V, T and P
own one approach each. The code is a Rust workspace in
`finetune/parakeet-binary/cpu1bit/` (one crate per approach plus `shared`),
pushed hash-checked to the Mac and built with the pinned stable toolchain
(1.99.1), `-C target-cpu=apple-m1` and `--locked`; frozen binaries come from one
staged tree. Builds and tests run at background priority under the shared Mac
lock, which timed blocks take exclusively. LocalFlow is read-only, nothing is
committed, and Codex reviews the code before any timed stage.

## Report

`cpu1bit/RESULTS.md`, per arm × condition × clip: gross padded energy per
replay (E_est, E_lower, E_upper), guard energy per replay and window duration
as separate columns (no idle-corrected energy), latency per replay, N; paired
changes, with an interval only for the confirmed
comparison and its stated conditions (the interval with and without
imputation, the imputed count, r̂_U, every rejected attempt with the gate that
fired, and the audit flags); the storage audit, workspace and exactness;
residency, acceptance and invalidation audits; band and rate evidence; B-NEON's
tuning log and gap to the control; what each approach tried and why it won or
lost; and the scope limits, stated plainly.

## Out of scope

The full-encoder NEON port, Core ML/ANE/GPU on the Mac, the iPhone, the Ryzen
run, training, and binarized activations.
