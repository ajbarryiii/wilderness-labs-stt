# History: plans/cpu-benchmark-mac.md up to amendment M16 (verbatim, superseded)

Superseded on 2026-10-21 by the consolidated plan [cpu-benchmark-mac.md](cpu-benchmark-mac.md). Kept verbatim for the review trail (plan reviews p1-p6, code reviews c1-c3 under /mnt/hd/wilderness-labs-stt/parakeet-binary/review/cpu-benchmark-mac/).

# Plan: native 1-bit CPU kernels for MB1 on Apple M1 Pro (NEON)

**Status: 2026-10-20: plan review round 1 NOT CLEAN (6 major, 3 minor); section
"Amendments (review round 1)" supersedes conflicting text; round 2 NOT CLEAN (1 major, 2 minor), addressed in M10–M12; **round 3 CLEAN** (review/cpu-benchmark-mac/p3.md). Team started 2026-10-21. 2026-10-21: draft amendment M13 after code reviews c1–c3; plan review p4 NOT CLEAN (5 major, 1 minor); amendment M14 (gross padded energy, spawn-based alignment bound) supersedes M13's bounds; plan review p5 NOT CLEAN (4 major, 3 minor); amendment M15 (coverage assumption A1, fixed replay count, +50% envelope, simulation grid); plan review p6 NOT CLEAN (4 major, 2 minor); draft amendment M16 (assumptions A2 and A3, stress family, sizing feasibility) supersedes conflicting M15 text, pending plan review p7. Replaces the Ryzen plan
(plans/cpu-benchmark.md, reviewed CLEAN, kept for reference) at the user's
request. User decisions:**
- Run on the Mac and build both the baseline and the candidates.
- CPU energy comes from exactly `powermetrics --samplers cpu_power -i 100 -f
  plist`, which needs non-interactive sudo for powermetrics (see the operator
  setup); verified 2026-10-20.
- Timed runs happen overnight with `--quiet-host --full-machine --priority
  normal` on the Mac's job supervisor.

All rules of plans/cpu-benchmark.md carry over unless changed below:
exactness, storage, dev-only workloads, screen-then-confirm selection, the
crossed bootstrap, the coverage check, team ownership and Codex review.

## What exists on the Mac (survey 2026-10-20)

- **Hardware:** M1 Pro, 8 P-cores in two clusters of 4 with 12 MB of L2 each,
  and 2 E-cores. NEON with DotProd (`SDOT`/`UDOT`); no I8MM, no BF16.
  16 GB RAM, about 44 GB free on disk.
- **LocalFlow has no aarch64 CPU path.** lf-cpu is AVX-512 only and does not
  compile on the Mac, and the Mac app runs Parakeet's encoder through Core ML.
  So **no CPU baseline exists on the Mac**. We write it.
- **Threads:** macOS has no affinity API. The only lever is QoS:
  USER_INTERACTIVE prefers P-cores, BACKGROUND confines a thread to the E-cores.
- **Energy:** per 100 ms sample, powermetrics reports `cpu_energy` (mJ,
  CPU clusters), `gpu_energy`, `ane_energy` and per-cluster DVFS residency.
  The IOReport readers that need no root report 0 for CPU energy.

## Scope: the projection workload, not full transcription

Porting LocalFlow's whole encoder (attention, convolutions, normalization,
frontend, decoder) to NEON is a separate, larger project. Here the measured
unit is the **projection workload**: for one utterance, the encoder's 216
projection calls in order. Per Conformer block that is ff1.linear1,
ff1.linear2, fused QKV, linear_pos, linear_out, pointwise_conv1,
pointwise_conv2, ff2.linear1 and ff2.linear2. Each call runs, as LocalFlow
does:
1. per-frame activation quantization into i8x2 / i8x3 residual parts;
2. the GEMM;
3. the FP32 epilogue.

Every arm processes the same captured real FP32 inputs.
- **Why this is the right unit.** The non-projection ops are identical across
  arms, so differences in encoder time and energy are the differences in this
  workload. On the Ryzen, projections were 70–76% of the encoder's time with
  VNNI.
- **Limit.** The report states this scope plainly and claims nothing about
  full-transcription latency or energy on the Mac.

**Inputs.**
- Source: H captures every projection's actual FP32 input on the NixOS GPU,
  using the FP32 NeMo reconstruction of each model, M1 and MB1, from its
  export.
- Workload: the frozen dev clips (dev manifests only).
- Size: FP32 at about 0.3 GB per clip, written to the Mac's artifacts root,
  `~/wilderness-labs-stt-artifacts/`, with SHA-256 checks
  (`ios/macpush.py`).
- Transfer: the captures are made once, before any timing. M1 arms use M1's
  activations and MB1 arms use MB1's.

## Arms

| Arm | Weights | Kernel |
| --- | --- | --- |
| **R-M1** | M1 ternary, 2-bit codes | **baseline B-NEON**: a faithful NEON port of LocalFlow's method. The 2-bit codes are expanded per 64-row block into an s8 tile resident in L2, then multiplied with `SDOT` (s8 × s8; no +128 offset is needed). |
| **R-MB1** | MB1 through the exact derived 2-bit export | B-NEON |
| V | MB1, 1 bit | weights stay 1-bit in memory; each 16-bit group of weight bits is expanded in registers to 16 bytes, 0x01 / 0xFF or 0x00 / 0x01 (the {0,1} form), with `vtst`/`vand`/`vbsl`, then `SDOT`. Exact integer correction as in the Ryzen plan; with {0,1}: d = 2·Σ b·q − Σ q. |
| T | MB1, 1 bit | T-MAC lookup tables. Per frame-part and group of g = 4 inputs: 16 INT16 partial sums, indexed by 4 weight bits with `TBL` (`vqtbl1q_u8` on split low/high bytes) or `vqtbl2q`. INT32 accumulation with proven intervals; no table requantization. |
| P | MB1, 1 bit | bit-serial control: popcount((a_r XOR n) AND valid) with `CNT` (`vcntq_u8`) and pairwise adds. |

**Fairness of a self-written baseline.**
- Agent H writes B-NEON, not the candidate agents. It ports LocalFlow's x86
  method structure for structure: block expansion into an L2 tile, then
  multiply-accumulate. It gets the same tuning budget as each candidate (tile
  shapes on dev clips) and is frozen before candidates are screened.
- Codex reviews it specifically for being a strong, not strawman, baseline.
- B-NEON also exposes a "1-bit tile" variant, which expands 1-bit codes into
  the same L2 tile and so isolates storage from kernel design. It is reported
  descriptively.

**Exactness.** Every arm's integer dot and FP32 output must be bit-identical
to an independent scalar oracle in Rust. The oracle uses LocalFlow's
quantization (`quant.rs`) and epilogue order, so M1 and MB1 outputs equal the
x86 LocalFlow semantics. The contract tests of plans/cpu-benchmark.md apply
unchanged:
- ragged shapes and extremal patterns;
- i8x2 and i8x3;
- 1, 4 and 8 threads;
- dispatch over all 216 calls;
- reuse with changed inputs;
- the native code unpacks to the derived codes exactly.

## Measurement on the Mac

**Process and threads.**
- Each block is a fresh process holding one arm.
- Threads are QoS USER_INTERACTIVE, with 8 workers (primary condition) or
  4 (secondary).
- Placement is verified, not assumed. powermetrics cluster residency during
  each window must show the E-cluster's busy share below a frozen limit
  (≤ 10% above its pre-window idle level) and both P-clusters active (for 8
  threads). A window that fails is invalid and rerun.

**Energy.** A long-running `sudo -n powermetrics --samplers cpu_power -i 100 -f
plist` per block, read from a pipe and stopped by closing it.
- **Primary:** the window's `cpu_energy` is the sum of the samples fully inside
  the window plus prorated boundary samples, using `elapsed_ns` and the sample
  timestamps.
- **Window length:** at least 6 s, so prorating error is ≤ 1 sample of 60.
- **Secondary:** idle-subtracted energy, from a 10 s pre-block idle window, as
  E_net. `combined_power` is recorded.
- **Kept:** raw plists.

**Quiet host.**
- **Jobs:** blocks run only as `job start --host mac --quiet-host --needs ...
  --full-machine --priority normal`, as the user authorized for overnight
  runs.
- **Before each block:**
  - no user input for 10 minutes (`ioreg` HIDIdleTime);
  - AC power and battery at 100%;
  - `pmset -g therm` nominal and Low Power Mode off;
  - load average below 1.5;
  - no other agent job running;
  - powermetrics shows the P-clusters idle.
- **During each window:** HIDIdleTime is rechecked, and a window with user
  input is dropped.
- **Lock:** one Mac-side lock file serializes builds and timed blocks, shared
  for builds and tests and exclusive for timed blocks.

**Design.** Williams replicates, screening R = 1, and confirmation of one
candidate with R = 6, with the crossed bootstrap and coverage check exactly as
in plans/cpu-benchmark.md. Primary condition: 8 P-threads, i8x2, CPU energy per
projection workload. Ties as specified there. The baseline for selection is
**R-MB1** (same weights); R-M1 is the drift-free reference for "MB1 vs M1 on
the Mac CPU".

**Workloads.** Dev only, frozen and hashed by H before any candidate timing:
- 8 timed dev clips, the 5 dev corpora at targets of about 16 s and 28 s,
  chosen by the round-robin rule;
- a 64-utterance exactness sample, all of whose projection inputs are captured.

There are no test clips anywhere on the Mac side.

## Team and code layout

**Agents.** Four Opus agents at xhigh reasoning, as for the GPU work:
- H (harness): captures and transfer, the Rust workspace, the oracle,
  B-NEON, the energy and QoS harness and the runner;
- V, T and P: one approach each.

**Code.**
- It lives in `finetune/parakeet-binary/cpu1bit/` on NixOS: a Rust workspace
  with one crate per approach plus `shared`.
- It is pushed to the Mac with hash-checked transfer and built there with
  `cargo --release` for `aarch64-apple-darwin`, on the installed stable
  toolchain, pinned and recorded. `-C target-cpu=apple-m1` is used for every
  arm alike.
- No LocalFlow files are copied or edited. `quant.rs` semantics are
  reimplemented and cross-checked against a reference dump from the x86 build.

**Ownership.** Each agent writes only its own crate and directory. Notes go
through a flock-serialized `cpu1bit/note`. Builds and correctness tests may
run on the Mac at any time at background priority under the shared lock. Only
timed blocks need the quiet overnight window. No commits.

## Report

`cpu1bit/RESULTS.md`, per arm × precision × thread count:
- latency and CPU energy per projection workload;
- paired changes, with a CI only for the confirmed comparison;
- weight bytes and workspace;
- exactness evidence;
- the placement and residency audit;
- B-NEON's tuning log;
- what each approach tried and why it won or lost;
- the scope limitation, stated plainly.

## Out of scope

The full-encoder NEON port, Core ML / ANE / GPU on the Mac, the iPhone, the
Ryzen run, training, and binarized activations.

## Amendments (review round 1, 2026-10-20)

`/mnt/hd/wilderness-labs-stt/parakeet-binary/review/cpu-benchmark-mac/p1.md`.

**M1. LocalFlow's schedule, not a 216-call replay.** H derives the projection
schedule from the measured x86 snapshot
(`lf-asr/src/encoder.rs`, around line 443) and freezes it in
`cpu1bit/shared/schedule.json` before any timing:
- `linear_pos` is cached after warmup, as LocalFlow does, and is not timed.
- Q, K and V are separate GEMMs sharing one activation quantization. There is
  no fused QKV.
- Quantization is reused exactly where LocalFlow reuses it.
- Shapes, call order and captured input layouts are recorded.

Trace contracts assert the call count, shapes, quantization reuse, capture
layouts and output ordering against the schedule. Any deliberate departure is
labelled.

**M2. Conclusions are restricted to the captured projection replay.** The
report says "projection replay on fixed captured inputs". Removing the
intervening operations changes cache state, activation traffic,
synchronization and DVFS. The captures follow the FP32 NeMo trajectory, not
LocalFlow's residual-INT8 encoder. So no encoder-level saving and no
encoder-output equivalence is claimed. An integrated encoder measurement needs
the full NEON encoder port, which is out of scope.

**M3. Energy is a bounded estimate.** It is reported as **"powermetrics-estimated
CPU-domain energy"**.
- **Before any timing,** H validates on the Mac and records:
  - timestamp resolution, actual sample intervals and jitter;
  - clock alignment between the sample timestamps and the benchmark's
    monotonic clock;
  - complete coverage of each window by samples (no gaps);
  - unit conversion.
- **Integration tests** cover jitter, gaps, malformed records and units.
- **Per-window uncertainty bound in joules:** the energy of the two boundary
  samples times the worst observed alignment error, plus a gap penalty. Any
  gap invalidates the window.
- **The confirmation verdict** of plans/cpu-benchmark.md must survive that
  bound. The improvement is declared only if the CI's upper bound stays below
  0 with every window's energy shifted to its least favourable bound.
- **Idle subtraction** is secondary. It does not isolate process energy, and
  the report says so.

**M4. Placement is stated, not over-claimed.**
- **Naming:** the conditions are **"8 / 4 USER_INTERACTIVE workers"**.
- **QoS check:** each worker sets its QoS and reads it back
  (`pthread_get_qos_class_np`); a mismatch aborts the block.
- **Residency:** cluster residency is reported descriptively, with the
  threshold defined exactly. Example: an E-cluster busy share more than 10
  percentage points above its pre-block idle mean flags the window, and
  flagged windows are reported and counted.
- **Claim:** no "P-core-only" claim is made, since macOS gives no thread-to-core
  evidence.

**M5. Storage and memory budget.**
- **Sizes:** H measures actual capture sizes and states a budget before
  transfer.
- **Resident on the Mac:** only the captures of the 8 timed clips for both
  models, at most 8 GB total.
- **Exactness sample:** reduced to 16 dev utterances. It is streamed in
  batches: transfer, verify the hash, run the exactness tests, then delete.
- **Transfer tool:** `cpu1bit/shared/push.py` reuses `macpush.py`'s
  hash-checked mechanism with the destination
  `~/wilderness-labs-stt-artifacts/cpu1bit/`. It refuses below
  20 GiB free.
- **Memory checks before each block:** memory pressure "normal", swap use not
  growing, and at most 2 GB of captures mapped per process.

**M6. Invalidation keeps the paired design.**
- **Records:** an invalidated window, whether from user input, a power or
  thermal transition, a sampling gap or a QoS mismatch, is recorded, never
  deleted.
- **Replacement:** its whole paired block-sequence (same replicate and
  sequence IDs) is re-measured once quiet conditions return, until the fixed
  number of valid replicates is reached.
- **Simulation:** the coverage simulation runs on the valid, complete design.

**M7. Baseline strength has an acceptance check.**
- **Freedom:** B-NEON may use NEON-specific register blocking beneath the
  expansion tile. Its tuning budget is frozen: the same grid size as each
  candidate.
- **Before freezing,** H:
  - inspects the assembly for spills;
  - compares B-NEON with a **pre-expanded SDOT control**: all weights stored as
    s8, i.e. no expansion cost, a ceiling for this arithmetic.
- **Reported:** B-NEON's gap to that control. A gap above 25% blocks the freeze
  until it is explained or fixed.

**M8. Quantizer golden vectors from x86.** H generates golden vectors with the
measured x86 LocalFlow build on NixOS, which needs AVX-512, so it runs on the
Ryzen. They cover:
- rounding ties;
- FMA-sensitive residual updates and epilogues;
- zero and tiny frames;
- non-finite rejection.

The NEON quantizer and every arm must match the component bytes, scale bits and
final output bits.

**M9. Whole-block state monitoring.**
- **Sleep assertion:** each block holds `caffeinate -i -s` for its lifetime.
- **Sampling:** AC state, battery, `pmset -g therm` and HIDIdleTime are sampled
  every second, and foreign CPU use is sampled with `ps`, the inherited audit
  ported to macOS.
- **Invalidation:** any power or thermal transition, user input, or foreign
  busy share above 0.5 cores invalidates the window under M6.

**M10. Conservative boundary-energy bounds** (round 2, major; replaces M3's
bound). Each powermetrics sample covers an interval [a_i, b_i] reconstructed
from its timestamp and `elapsed_ns`. ε is the validated clock-alignment
uncertainty, measured before timing; a window whose alignment check exceeds the
tolerance is invalid. For a window [t0, t1]:
- **E_lower** = the energy of samples whose intervals lie fully inside
  [t0 + ε, t1 − ε];
- **E_upper** = the energy of every sample whose interval intersects
  [t0 − ε, t1 + ε];
- **E_est** = the prorated estimate, reported as the point value.

All three are divided by the number of replays completed in the window.
Windows last at least **12 s**, so the bound width is about 2 samples in 120 or
more.
- **Confirmation gate:** an improvement is declared only if the CI's upper
  bound is below 0 when every candidate window uses E_upper and every baseline
  window uses E_lower.
- **Tests:**
  - concentrated activity at either boundary;
  - zero alignment error, where the bounds remain the full boundary samples;
  - gaps and jitter;
  - malformed records;
  - unit conversion.

**M11. Foreign-CPU estimator** (round 2, minor). Host busy time is computed from
cumulative CPU-tick deltas (`host_processor_info`), and our own processes'
busy time from cumulative CPU-time deltas (`proc_pid_rusage`), over the same 1 s
intervals:
- foreign busy cores = host busy − own busy;
- the snapshot of our process set is refreshed every interval;
- a probe failure invalidates the window under M6;
- `ps %cpu` is not used.

**M12. Memory monitoring throughout** (round 2, minor). Memory pressure
(`kern.memorystatus_vm_pressure_level`) and swap-out deltas (`vm_stat`
pageouts and swapouts) are sampled every second during warmup, idle and
timing. Pressure above normal, or any swap-out increase, invalidates the window
under M6.

## Amendment M13 (draft 2026-10-21; plan review p4 NOT CLEAN; M13.2–M13.8 superseded by M14)

Code reviews c1–c3 (`review/cpu-benchmark-mac/c1.md`–`c3.md`) showed that M10's ε
cannot be established defensibly: every calibration-burst estimator of the
absolute powermetrics alignment depended on a model of burst onsets (overshoot,
false detections), and its rejections made acceptance arm- and state-dependent
(c3 major 2). M13 removes the need for a precise alignment.

**M13. Guard-banded windows, alignment-free energy bounds, acceptance audit.**
Supersedes M3's alignment validation and per-window bound, M10's ε, E_lower,
E_upper and E_est, and the calibration bursts of code reviews c1/c2. M10's
window length (≥ 12 s), its per-replay normalization and its confirmation gate
(candidate E_upper against baseline E_lower) remain, with the bounds below.

**M13.1 Block protocol.** For each clip, in order: load the clip, one untimed
replay, park, wait S + G, the timed window W = [t0, t1] (back-to-back replays
for ≥ 12 s; t0 and t1 on CLOCK_UPTIME_RAW immediately before the first and
after the last timed replay), park, wait G + S′. Then the next clip.
- **Parked:** no replay or other process work runs; the pool workers park on
  their own after LocalFlow's spin (≤ 200,000 spin iterations, about 1 ms).
- **Window events are emitted after the post-window wait.** No stdout write and
  no parsing by the runner happens inside [t0 − G, t1 + G], so no runner
  activity is time-locked to window boundaries.
- **Frozen constants:** G = 1.0 s, S = 0.5 s (after the untimed replay),
  S′ = 0.3 s (≥ one sample beyond the padded region before the next clip
  load).

**M13.2 Samples considered (crude, causally safe alignment).** The provisional
timeline is used without any absolute correction: within a segment,
b_i = min_j(arrival_j − cum_j) + cum_i, which is late by e ≥ 0 because no sample
is delivered before it ends. For window W, C is the set of samples whose
provisional interval meets [t0 − G, t1 + G].
- C must lie in one segment, with no malformed or lost record.
- L is the longest `elapsed_ns` in C.

**M13.3 Idle reference and classification.** From the pre-block 10 s idle
window (M10, unchanged):
- p_ref = median per-sample power;
- tol = max(5 mJ, max over idle-window samples of |E_i − p_ref·el_i| + 3 mJ).

Each sample of C has an excess x_i = E_i − p_ref·el_i. It is idle-classified if
x_i ≤ tol, and active otherwise.

**M13.4 The activity span A and its validity.** A is the longest run of
consecutive active samples in C. The window is valid only if all of these
hold:
1. **Edges:** the first K and the last K samples of C are idle-classified
   (K = 3).
2. **Separation:** A is separated from every other active sample of C by at
   least one idle-classified sample. Other active runs are at most 2 samples
   long (background spikes).
3. **Coverage of the window duration:** |A| ≥ |W| − R, where |A| = Σ el_i over
   A, |W| = t1 − t0 and R = 200 ppm · |W| (rate difference between
   `elapsed_ns` and CLOCK_UPTIME_RAW).
4. **No long tail or adjacent activity:** D = |A| − |W| ≤ D_MAX, with
   D_MAX = 2 L + 0.4 s.
5. **Crude placement:** A's provisional start lies in
   [t0 − L − 5 ms, t0 + G/2]. Causality forbids an earlier start. The upper
   limit and condition 1 together exclude capturing any other activity span.
6. **Guard background sample:** I is the set of samples of C outside A and
   outside A's two neighbours. I must hold at least 2K samples.

All M9, M11 and M12 state checks over [t0 − G, t1 + G] (bracketing samples
included) must pass as before.

**M13.5 Quantity measured.** E_window = ∫_W P dt + ∫_{A∖W} (P − B) dt, where B
is the background (non-work) power. That is, the gross CPU-domain energy over
the timed window, plus the work-induced energy above background in the
activity span outside it (transient tails count as work).
- **Not included:** work-induced excess that stays below tol beyond A. This is
  a slow, sub-threshold tail such as thermal leakage. It is reported per window
  as the post-guard excess: the mean power of the post-window samples of I
  minus the mean power of the pre-window samples of I.

**M13.6 Bounds** (q = 1 mJ, the integer-mJ reporting granularity per sample;
P_lo and P_hi are min_I (E_i − q)/el_i and max_I (E_i + q)/el_i):
- **E_lower** = Σ_A (E_i − q) − (D + R)·P_hi − 2L·(P_hi − P_lo)
- **E_upper** = Σ_A (E_i + q) − max(D − R, 0)·P_lo
  + Σ over A's two neighbours N of max(0, E_N + q − el_N·P_lo)
- **E_est** = Σ_A E_i − D·P_mid, with P_mid = median_I E_i/el_i (point value)

All three are divided by the window's replay count, as in M10.

**Derivation.** These rest on three assumptions:
- **(a) Work detectability:** every whole sample inside W has x_i > tol. 8 or 4
  busy P-core workers exceed idle by ≥ 1 W, i.e. ≥ 100 mJ per sample, against a
  tol of about 5–30 mJ. Validate measures the minimum ratio of interior excess
  to tol, and requires it to be ≥ 10.
- **(b) Contiguity:** W is one interval of work.
- **(c) Background range:** background power B(t) on A∖W and on the part of A's
  neighbours outside W lies within [P_lo, P_hi], the range observed in the same
  window's guards.

By (a) and (b), no whole sample inside W is idle-classified. So W ⊂ A ∪ N_pre ∪
N_post, and W extends beyond A by δ ≤ L on each side at most.
- **If W ⊂ A:** E_window = Σ_A E − ∫_{A∖W} B with |A∖W| = D, which gives the
  bounds without the sliver terms.
- **If W extends by δ into a neighbour N:** |A∖W| = D + δ.
  - For E_window: the term ∫_{W∖A} P lies between δ·P_lo and E_N − (el_N − δ)·P_lo.
  - For the lower bound: the extra −δ·(P_hi − P_lo) is at most
    −L·(P_hi − P_lo) per side.
  - For the upper bound: −(D + δ)·P_lo + E_N − (el_N − δ)·P_lo =
    −D·P_lo + E_N − el_N·P_lo.
- **Other terms:** q bounds each sample's rounding, and R bounds the clock-rate
  difference in D.
- **Alignment:** none of the terms uses the alignment e. Misalignment can only
  move A towards an edge of C, which conditions 1, 4 and 5 then reject.
  Misalignment never changes a bound.
- **Expected width:** 2q·n_A + D·(P_hi − P_lo) + neighbour excess + small
  terms. That is about 0.3 J for a 12 s window at about 8 W, i.e. roughly
  0.3%; validate measures it.

**Residual attributions.** These are stated in the report:
- Background activity overlapping W or A is counted as window energy. This
  holds whether it falls inside the window or in its boundary samples; M9 and
  M11 bound it (foreign busy + tick margin ≤ 0.5 cores per 1 s interval).
- The state sampler's 1 Hz probes run in every window and guard alike.

**M13.7 Validation (replaces M3's alignment items).** Validate runs before any
timed stage. It is instrument-only, and no statistic reads it.
- It records:
  - sample intervals and jitter;
  - provisional output latency;
  - the unit check;
  - the idle-window p_ref and tol;
  - P-cluster idle share;
  - one instrument block of R-MB1 in the primary condition.
- Per window it records:
  - validity and reasons;
  - |A|, D and L;
  - P_lo, P_hi and the I size;
  - neighbour excess;
  - the minimum ratio of interior excess to tol;
  - the post-guard excess;
  - E_lower, E_est, E_upper and the relative bound width;
  - block cadence.
- **Pass:** all 8 windows valid; minimum interior excess/tol ≥ 10; median
  relative width (E_upper − E_lower)/E_est ≤ 1%; D ≤ D_MAX in every window.
  Otherwise the constants are revisited, with numbers, before any timing.
- **Tests (contract-first):**
  - Codex c3's scenario: 50 ms delivery delay, two 20 mJ pre-burst disturbances
    per calibration, 0.4 W background, true window energy 4.800 J. The bounds
    must contain the truth, and there are no calibration bursts.
  - Constant misalignment from 0 to G/2: bounds always contain the truth.
  - Misalignment beyond the guard: the window is invalid.
  - Edge contamination: work or a spike in the first or last K samples makes
    the window invalid.
  - Long and short tails: inside D_MAX the bounds contain the truth including
    the tail; beyond it the window is invalid.
  - Background spikes in the guards, in the neighbours and inside A∖W.
  - Sliver overlap into a neighbour.
  - Rate error, quantization and malformed or lost records.

**M13.8 Acceptance audit (c3 major 2).** The unit is a block attempt: valid if
every window of the block is valid.
- **Every stage reports** acceptance:
  - overall;
  - by arm;
  - by position in the sequence;
  - against host-state covariates: idle-window power, P-cluster idle share,
    load average, thermal state and time since the previous block.
- **Frozen rule for confirmation.** The confirmation verdict is inferential only
  if all of these hold on the confirmation stage's own attempts:
  - **(a)** overall block acceptance ≥ 95%;
  - **(b)** acceptance does not depend on arm: two-sided Fisher exact test on
    (valid, invalid) × arm with p ≥ 0.10, and an absolute difference
    ≤ 5 percentage points;
  - **(c)** the same holds for position.

  Otherwise the result is descriptive only, and no improvement is declared.
- **Coverage and false-improvement simulation** (plans/cpu-benchmark.md, M10
  gate). It now includes rejection and retry under worst-case selection.
  - The observed confirmation rejection rate r is the per-block-attempt
    rejection rate.
  - In each simulated stage, a sequence attempt is rejected with probability
    1 − (1 − r)^b, where b is the number of blocks per sequence. Rejection
    falls on the attempts with the largest candidate-minus-baseline paired
    change, i.e. the attempts most unfavourable to the candidate, which biases
    towards false improvement.
  - Rejected attempts are re-drawn, as M6 re-measures them.
  - The 93% coverage and 3.5% false-improvement thresholds apply to this
    procedure.
  - Codex c3's example (acceptance 55% versus 92%, true mean 0) fails rule (a).
    The worst-case simulation bounds the bias at any admitted acceptance.
- **Screening:** acceptance is audited descriptively. A screening stage with
  acceptance below 95%, or with an arm dependence under the same test, is
  reported as such.

## Amendment M14 (2026-10-21; plan review p5 NOT CLEAN; conflicting text superseded by M15)

Plan review p4 (`review/cpu-benchmark-mac/p4.md`) found 5 majors and 1 minor in
M13:
- Majors 1, 2 and 4 attack the inference of the work span A and the background
  subtracted beneath it. These needed a minimum power inside W, a background
  range under the work, and detectable interior samples.
- Major 3: no coarse alignment bound.
- Major 5: the rejection model.
- Minor: the c3 fixture.

M14 drops A, the background subtraction and the tail definition.

**M14. Gross padded energy, spawn-based alignment bound, missing-outcome
imputation.**
- **Keeps:** M13.1's block protocol (guards, settle waits, window events emitted
  after the post-window wait) and M13.3's idle reference.
- **Supersedes:** M13.2 and M13.4–M13.8. M10's window length, per-replay
  normalization and confirmation gate structure (candidate E_upper against
  baseline E_lower) remain, with the bounds below.

**M14.1 Measurand.** For a timed window W = [t0, t1], the padded interval is
Π = [t0 − G, t1 + G] on CLOCK_UPTIME_RAW, and the measurand is

  E_pad = ∫_Π P(t) dt,

the gross powermetrics-estimated CPU-domain energy over Π, divided by the
window's replay count.
- **No assumptions:** there is no activity span, no background subtraction, no
  tail definition, and no assumption about power inside W.
- **Contents:** E_pad contains the window's work, its transients up to t1 + G,
  and 2G of guard energy.
- **The guard energy** is about 2G × idle power, i.e. about 1% of a 12 s window
  at about 8 W, and is the same by design for every arm. Paired differences
  cancel it in expectation.
  - Divided by replays, it attenuates paired changes toward 0 by the factor
    E_window / (E_window + E_guard).
  - When a candidate draws more power per window but uses less energy per
    replay, it can enlarge an improvement by at most E_guard/E_window of the
    change. This is about 1% of the change, e.g. −20% reported as −20.2%.
  - The report states this.
- **An idle-corrected value** E_pad − 2G·p_ref is reported descriptively only. No
  gate uses it.

**M14.2 Model-free alignment bound** (p4 major 3). The block's powermetrics
process is spawned by the runner. s is CLOCK_UPTIME_RAW read immediately before
the spawn. Sample k has elapsed el_k and delivery (read) time d_k.
- **Assumption (A1):** samples are non-overlapping and contiguous. Each
  record's energy and elapsed_ns are differences between consecutive reads of
  cumulative counters (`is_delta = true` in every record).
- **Then end_k = s + o + c_k:**
  - c_k = Σ_{j≤k} el_j is the sum over the records received;
  - o ≥ 0 is the common start offset;
  - o is constant within a segment;
  - a lost or malformed record makes o larger for the segment after it.
- **Causality:**
  - end_k ≥ s + c_k: sampling cannot start before the spawn;
  - end_k ≤ d_k: a sample is not delivered before it ends.
- **Hence**, allowing a rate difference |δ| ≤ ρ = 200 ppm between elapsed_ns and
  CLOCK_UPTIME_RAW: end_k ∈ [s + o + c_k(1 − ρ), s + o + c_k(1 + ρ)], with
  o ∈ [0, U] and U = min_k (d_k − s − c_k(1 − ρ)) over the window's segment.
- **Checks:**
  - U < 0 (some d_k < s + c_k(1 − ρ)) contradicts (A1) or causality and
    invalidates every window of the block;
  - a window is valid only if U ≤ U_MAX.
- **Constants:** frozen U_MAX = 0.6 s, measured by validate. G ≥ U_MAX + K·L + 0.1 s,
  so G = 1.0 s with K = 2 and L ≤ 0.13 s.
- **Stale mappings are impossible by construction.** Codex's 13.5 s constant
  delivery delay gives U ≥ 13.5 s and is rejected.
- **Validate records:** the distribution of U, the first-sample offset, and the
  stability of d_k − s − c_k over a block. A growing trend would indicate unsampled
  time and is reported.

**M14.3 Samples summed.** With the mapping of M14.2 (end_k between
s + o + c_k(1 − ρ) and s + o + c_k(1 + ρ), start_k likewise with c_{k−1}) and
o ∈ [0, U]:
- P_out = samples whose interval can intersect Π for some o in [0, U];
- P_in = samples whose interval lies inside Π for every o in [0, U];
- X = P_out ∖ P_in are the edge-uncertain samples, at most ⌈U/L⌉ + 1 on each
  side.

P_out must lie in one segment, with no malformed or lost record, so that Π is
covered by contiguous samples.

**M14.4 Validity checks.** All are disturbance or coverage checks. None is an
assumption of the bounds.
1. **Alignment:** U ≤ U_MAX, and no causality contradiction (M14.2).
2. **Coverage:** M14.3.
3. **Edges:**
   - every sample of X and the K outermost samples of P_in at each end are
     idle-level against M13.3's pre-block reference (x_k ≤ tol);
   - this shows that no work, tail or foreign activity reaches the boundaries
     of Π under any admissible offset;
   - the settle waits keep the untimed replay and the next clip load out of
     every sample that P_out can contain: S ≥ U_MAX + L + 0.4 s, frozen at
     S = 1.2 s, and S′ ≥ U_MAX + L + 0.2 s, frozen at S′ = 1.0 s. These
     replace M13.1's S = 0.5 s and S′ = 0.3 s.
4. **Work present:** at least one sample of P_in lying inside W for every offset
   is active (x_k > tol).
   - This rejects a window without work, e.g. the c3 fixture with only 0.4 W
     background (p4 minor).
   - It imposes no detectability or minimum-power requirement.
5. **State:** M9, M11 and M12 state checks over Π extended by U + L, with
   bracketing samples, as before.

**M14.5 Bounds.** q = 1 mJ is the per-sample rounding of integer-mJ energies.
- **E_upper** = Σ_{P_out} (E_k + q). Π is inside the union of P_out's intervals
  for every admissible o, and energy is non-negative.
- **E_lower** = Σ_{P_in} (E_k − q). P_in's intervals lie inside Π for every
  admissible o.
- **E_est** is the prorated sum at o = U/2 (point value).

All three are divided by the replay count.
- **Width:** Σ_X (E_k + q) + 2q·|P_in|. X is idle-level by check 3, so the
  width is about 2(⌈U/L⌉ + 1) idle samples (about 40–50 mJ each) plus
  2 mJ × about 130 samples, i.e. about 0.5–0.7 J on a 12 s window at 8 W (about
  0.6%).
- **The confirmation gate** (M10) uses candidate E_upper against baseline
  E_lower. This is the deterministic least-favourable shift. Everything else is
  statistical and carried by the CI.

**M14.6 Validation** (instrument-only; replaces M13.7).
- **Recorded per window:**
  - sample intervals;
  - s, U and the first-sample offset;
  - |X|, P_in and P_out;
  - the edge excesses;
  - E_lower, E_est and E_upper, and the relative width;
  - the idle-corrected value;
  - block cadence.
- **Recorded per block:** p_ref, tol and the P-cluster idle share.
- **Pass:**
  - all 8 windows of the R-MB1 instrument block are valid;
  - U ≤ U_MAX in every window;
  - median relative width ≤ 1%.

  Otherwise U_MAX, G, S, S′ or K are revisited, with numbers, before any timing.

**M14.7 Rejection and selection** (p4 major 5).
- **Rejection gates** are disturbance and coverage checks only:
  - foreign CPU, thermal, power, user input, memory (M9, M11, M12);
  - QoS (M4);
  - powermetrics coverage and the alignment bound (M14.2–M14.4).

  None depends on an arm-specific model.
- **Codex's p4 example:** states with −1% in 98% of sequence attempts and +49%
  in 2%, with the latter rejected.
  - It is mainly heavy-tail sampling variability. With no rejection at all, 12
    confirmation sequences contain no +49% state with probability
    0.98¹² ≈ 78.5%, and the CI then reflects only the −1% state.
  - Six replicates cannot detect variation that rare, with or without
    rejection. The report states this limit.
- **Missing-outcome bound:** what rejection could add beyond that variability is
  handled by imputation.
  - Every rejected confirmation sequence attempt is imputed, whatever its
    number of failed blocks. Accounting is at sequence level, so dependent
    block failures are covered.
  - It enters as one additional sequence observation of its replicate at the
    worst case for the candidate. Per clip, the candidate's largest E_upper per
    replay over the valid windows of the confirmation stage is paired against
    the baseline's smallest E_lower per replay over the same windows.
  - The replicate's value per clip is the mean over its valid and imputed
    sequences.
  - The crossed bootstrap and the M10 verdict are computed with the
    imputation, and an improvement is declared only if it survives.
  - The imputation assumes that a rejected attempt's outcome lies within the
    observed range of the stage. The report states this assumption.
- **The coverage and false-improvement simulation** includes the same
  procedure.
  - Each simulated sequence attempt is rejected with probability r̂_U, the
    one-sided 95% Clopper–Pearson upper bound of the confirmation's
    sequence-attempt rejection rate. This accounts for rejection-rate
    uncertainty.
  - Rejected attempts are re-drawn (M6) and imputed as above, from the
    simulated stage's own extremes.
  - A simulated stage counts as declaring an improvement only under the gate
    below: the imputed interval's upper bound is below 0, and it has at most 2
    rejected attempts.
  - The 93% coverage and 3.5% false-improvement thresholds apply.
- **Gate** (coordinator decision 2026-10-21). The confirmation is inferential
  only if both of these hold:
  - the imputed analysis still shows the improvement, i.e. every rejected
    sequence attempt is imputed at the worst case and the M10 interval upper
    bound is below 0 under the M14.5 least-favourable bounds (candidate E_upper,
    baseline E_lower);
  - the stage has at most 2 rejected sequence attempts.

  Beyond 2 rejected attempts the result is descriptive. The worst-case
  imputation is the protection against selection: zero rejections are not
  required.
- **The acceptance audit is reporting only:** overall, by arm, by position, and
  against host-state covariates, at block and at sequence level.
  - Three conditions are reported as flags and do not gate: overall block
    acceptance below 95%, a two-sided Fisher exact p < 0.10 for arm or position
    dependence, and an arm or position difference above 5 points.
  - Every rejected attempt is listed with the disturbance or coverage gate that
    fired (M14.4, M9, M11, M12, M4).
- **Report per confirmation:**
  - the interval with imputation (the gate) and without it;
  - the number of imputed sequences;
  - r̂_U;
  - the rejected attempts and their reasons;
  - the audit flags.

**M14.8 Tests** (contract-first; shared/mac/tests/test_guard.py and
shared/tests/test_acceptance.py).
- **Bounds contain the gross padded energy:**
  - clean windows, start offsets from 0 to U_MAX, and delivery jitter;
  - p4 major 1's low-power start sliver;
  - p4 major 2's guard background above the background under the tail;
  - p4 major 4's 0.5 s sub-threshold start with a detectable tail;
  - guard spikes;
  - tails that end before the edges;
  - ±150 ppm rate error;
  - a lost record before the window's segment.
- **Invalid:**
  - U > U_MAX, including p4 major 3's 13.5 s delay;
  - a causality contradiction;
  - edge contamination under any admissible offset;
  - a tail reaching the edges;
  - the untimed replay or clip load overrunning the settle waits;
  - a malformed or lost record inside P_out;
  - the c3 background-only fixture.
- **Analysis:**
  - Fisher exact test against enumeration;
  - audit counting of every attempt at block and sequence level;
  - imputation of rejected sequence attempts at the worst case, which removes a
    marginal improvement and keeps a large one;
  - a large improvement with 1 or 2 rejected attempts stays inferential;
  - 3 rejected attempts make the result descriptive;
  - the audit flags are reported and do not change the verdict;
  - every rejected attempt is listed with its reasons;
  - the Clopper–Pearson bound;
  - the simulation with rejection equals the plain one at r = 0, and its false
    improvements do not rise with r.

## Amendment M15 (2026-10-21; plan review p6 NOT CLEAN; conflicting text superseded by M16)

Plan review p5 (`review/cpu-benchmark-mac/p5.md`) confirms that the M14.5
inequalities are conservative given four prerequisites: A1, a fresh post-spawn
baseline, the rate bound and complete coverage. M15 establishes or states those
prerequisites and scopes the claims. It supersedes conflicting M14 text.

**M15.1 Coverage** (p5 major 1).
- **Framing (deterministic).** powermetrics writes to a blocking pipe, and the
  reader keeps every byte, so the transport cannot drop records. Every
  NUL-terminated chunk must parse into a record of the enforced schema (M3,
  c1 minor), which now includes `timestamp`. The block is invalid if any of
  these occur:
  - a chunk fails to parse;
  - the stream ends (EOF) before the runner stops it;
  - powermetrics exits on its own.

  The only chunk exempt is the one unterminated chunk cut by the runner's own
  stop, which comes after every bracketing sample.
- **Assumption A1: producer continuity.** powermetrics emits every sample. Each
  record's interval starts where the previous one ended, and its energy and
  elapsed_ns cover that same interval. The plist carries no sequence number or
  cumulative counter, so A1 cannot be proven per sample.
  - **Evidence.** In job `cpu1bit-H-plist-fields2` (2026-10-21), two captures
    totalling 7,061 records over 2,213 s showed b = timestamp − Σ elapsed_ns
    (timestamp has 1 s resolution) inside a band exactly 1 s wide (1.001 s and
    1.000 s), with no drift: no discontinuity detected by this check. Its
    limit is stated in M16.4.
  - **Per-block check.** The band width over every block's records must be
    ≤ 1 s + τ_band, with τ_band = 50 ms; otherwise the block is invalid. This
    catches a skip once samples near both band edges occur on both sides of
    it: likely over a block, but not guaranteed within one window.
  - **Accumulated evidence.** Overnight stages add band evidence from every
    block. The report gives the total records and the widest band seen.
  - **Verdict.** The verdict is conditional on A1. Codex's p5 repro (one
    interior record lost while delivery latency drops by the same amount) is a
    producer-side skip and lies outside A1.
- **Per-window corroboration (flag, never gate).** For the window's samples,
  compare (d_last − d_first) with Σ_{k>first} el_k. A mismatch larger than the
  delivery jitter J measured in validate is flagged. A skip shows up here
  unless a latency change of the same size masks it.
- **Bracketing.** The runner keeps reading after the block process exits until,
  for every window, a record has arrived whose earliest admissible start
  exceeds t1 + G. A capture that ends earlier makes that window invalid.
- **Startup baseline and interval semantics** (validate; recorded, with
  numbers):
  - spawn time s, first-sample bound d_1 − s − el_1, and the distribution of U;
  - the delivery jitter J;
  - the rate between Σ el and the delivery span over a 10-minute capture;
  - an energy-interval falsification test. Twenty one-thread spins of known
    [on, off] are run; no sample whose mapped interval ends before `on` under
    every admissible offset may carry excess energy above tol. This shows,
    within U, that energy and elapsed_ns cover the same interval. A failure
    stops all timing.

**M15.2 Fixed replay count** (p5 major 2; replaces "back-to-back replays for
≥ 12 s").
- **N is fixed per condition and clip, and is the same for every arm.**
  N_{cond,clip} = ⌈12 s / (0.95 · min over roster arms of the per-replay
  latency)⌉. The latencies come from a sizing stage per condition: one block
  per arm, one replay per clip, recorded like validate and never compared,
  run before that condition's first stage.
- **Each window is exactly N replays.** Its duration varies by arm: the fastest
  arm's window is planned at ≥ 12 s, and slower arms' windows are longer.
- **No rule assumed a fixed duration** except M10's "≥ 12 s":
  - per-replay latency is still the window duration divided by N;
  - the M14.5 width depends on |X| and the sample count, not on duration;
  - state checks cover whatever span the window has.
- **Exact relationship.** Per replay, e = E_W/N and g = E_guard/N, where E_guard
  is the energy over the two guards, including any guard transient. The
  measurand is e + g, and the paired change is
  (e_C + g_C − e_B − g_B)/(e_B + g_B).
  - With equal guard energy (g_C = g_B = g, the same protocol term at the same
    N), this equals (e_C − e_B)/(e_B + g). That is the unpadded change
    attenuated by the factor e_B/(e_B + g), i.e. by g/(e_B + g), about 1%.
  - Guard transients that differ between arms (tails) make g_C ≠ g_B. That
    difference is work-induced and is counted.
- **Reported per arm and clip:** N, window durations, and the guard
  contribution (the energy of the samples lying wholly in the guards for
  every admissible offset, per second, times 2G).

**M15.3 Selection: a prespecified sensitivity envelope** (p5 major 3; replaces
M14.7's observed-range imputation).
- **Imputation.** Every rejected confirmation sequence attempt is imputed as one
  additional sequence observation of its replicate. Its paired change is
  +50% for every clip, i.e. the candidate 50% worse than its paired baseline.
  Codex's +45% case lies inside this envelope.
- **Gate (M14.7, otherwise unchanged).** The imputed M10 interval's upper bound
  must be below 0 under the M14.5 least-favourable bounds, and the stage may
  have at most 2 rejected sequence attempts.
- **The verdict text states** that it is conditional on rejections arising
  only from the external disturbance and coverage gates, and on the +50%
  envelope. No protection is claimed beyond the envelope.

**M15.4 Simulation** (p5 major 4; replaces M14.7's simulation and its
monotonicity test).
- **Grid.** The full stage-2 procedure, with retries, the +50% imputation and
  the 2-rejection cap, is simulated over rejection rates
  r ∈ {0, r̂_U/4, r̂_U/2, 3r̂_U/4, r̂_U}. r̂_U is the one-sided 95%
  Clopper–Pearson upper bound of the confirmation's sequence-attempt
  rejection rate.
- **Two rejection models:**
  - independent;
  - adversarial dependent: in each simulated stage the rejected attempts are
    those least favourable to the candidate (largest candidate-minus-baseline
    change).
- **Pass:** the maximum false-improvement rate over the grid and both models is
  ≤ 3.5%, and coverage of the interval at r = 0 is ≥ 93%.
  - Coverage is evaluated at r = 0 because the imputation deliberately biases
    the interval upward.
  - Otherwise the result is descriptive only.
- **The report gives** the whole grid.

**M15.5 Attribution wording** (p5 minor 1).
- **Threshold-based disturbance checks.** The idle-level edge checks and the
  work-present check (M14.4) are threshold-based disturbance checks:
  - they do not prove that no work, neighbouring residual energy or carryover
    is present: a 40 mW residual passes a 5 mJ / 100 ms tolerance;
  - a permitted foreign spike can satisfy work-present.
- **Finite padding** does not capture every transient.
- **Replay completion** relies on the completion barriers: ThreadPool::run
  returns only after every worker finishes, and t1 is stamped after the last
  replay returns.

**M15.6 Clock uncertainty** (p5 minor 2).
- **Mapping.** For sample k, with c_k the cumulative elapsed time since spawn,
  the mapping envelope spans U + 2ρc_k.
- **Constants:**
  - ρ = 50 ppm: validated in M15.1 by the measured rate plus its uncertainty,
    and invalidating timing if exceeded;
  - powermetrics is respawned for every block (as now);
  - a maximum block age A_MAX = 600 s from spawn to the last bracketing
    sample;
  - L ≤ 130 ms for every sample of P_out is a validity gate.
- **The inequalities become:**
  - G ≥ U_MAX + 2ρA_MAX + K·L + 0.05 s = 0.97 s, so G = 1.0 s;
  - S ≥ U_MAX + 2ρA_MAX + L + 0.3 s = 1.09 s, so S = 1.2 s;
  - S′ ≥ U_MAX + 2ρA_MAX + L + 0.1 s = 0.89 s, so S′ = 1.0 s.
- **A window ending after A_MAX** is invalid.
- **X is computed** from the actual mapped sample intervals, not as ⌈U/L⌉ + 1,
  and its count is reported per window.

**M15.7 Fixtures** (p5 minor 3). Startup offsets in the tests are
non-negative. The p4-major-2 fixture uses a 1 W idle background.

## Amendment M16 (draft 2026-10-21, pending plan review p7)

Plan review p6 (`review/cpu-benchmark-mac/p6.md`) resolved the guard
cancellation, the observed extrema (now a conditional sensitivity analysis),
the attribution wording and the fixtures. M16 adds two assumptions, scopes the
simulation claims, corrects the continuity wording and makes sizing
fail-closed. It supersedes conflicting M15 text.

**M16.1 Assumption A2: post-spawn origin** (p6 major 1). The first record's
interval begins at or after the spawn time s. That is, powermetrics takes a
fresh counter baseline after it starts. The mapping's o ≥ 0 relies on this.
- **Falsification test in validate.** Run 20 trials. In each, a one-thread spin
  of ≥ 500 ms ends immediately before s, and then powermetrics is spawned.
  - The first record must be idle-level: excess ≤ tol (M13.3). Any excess fails
    validate and stops all timing.
  - This bounds any pre-spawn coverage to δ_A2 = tol / (measured spin excess
    power). δ_A2 is reported, and o ∈ [−δ_A2, U] replaces o ∈ [0, U] in M14.3.
- **Per block.** el_1 ≤ d_1 − s must hold (already implied by U ≥ 0, M14.2).
- **Verdict.** The verdict is conditional on A2.
- **Contract.** A first interval starting 100 ms before spawn, with 200 ms
  delivery latency, passes U ≥ 0 and every window check, but must fail the
  pre-spawn test.

**M16.2 Assumption A3: constant rate within a block** (p6 major 2). Within a
block, elapsed_ns advances at the same rate as CLOCK_UPTIME_RAW, up to a
constant |ρ| ≤ 50 ppm.
- **Evidence.** /usr/bin/powermetrics imports mach_absolute_time,
  mach_continuous_time and mach_timebase_info (`nm -u -arch arm64e`,
  2026-10-21).
  - CLOCK_UPTIME_RAW is mach_absolute_time. mach_continuous_time differs only
    across sleep, which caffeinate prevents.
  - So elapsed_ns is very likely on the same kernel timebase, with ρ ≈ 0.
  - Energy comes from IOReportCreateSamplesDelta between the same samples.
- **Validation over all prefixes** (validate's 10-minute capture, and every
  block).
  - E_k = (d_k − d_1) − (c_k − c_1) is the accumulated error for every prefix
    k.
  - J is the delivery jitter, measured as the range of d − c within any 10 s
    stretch, where rate error contributes ≤ 0.5 ms.
  - Require max_k (|E_k| − J) ≤ ρ·(c_k − c_1).
  - In validate, a failure stops all timing. In a block, it invalidates the
    block.
- **Verdict.** The verdict is conditional on A3.
- **Contracts that must fail:**
  - a cancelling rate error of +100 ppm for 300 s, then −100 ppm for 300 s.
    The endpoint error is about 0, but the intermediate error is 30 ms against
    a 15 ms allowance.
  - a late-block window under a 100 ppm error.

**M16.3 Simulation scope** (p6 major 3).
- **Claims are restricted to the generating model.** The M15.4 error-control
  claims hold only for that model: null data resampled from the screening's
  variance components, with independent or adversarial rejection on that
  distribution.
- **Coverage at r = 0** supports the directional gate only. It is not two-sided
  coverage of imputed intervals.
- **Stress family** (a known limitation, reported, not a pass criterion).
  - Unconditional sequence outcomes are a mixture: an unfavourable change u
    with probability p, and a favourable change f = −p·u/(1 − p), so the mean
    is 0. They are combined with the screening noise.
  - Every unfavourable attempt is rejected, so rejection depends on the
    outcome.
  - Cases: p ∈ {5%, 10%}, with u = +45%. Zero-observed-rejection realizations
    are included. For p = 10% they occur with probability 0.9¹² ≈ 28%, and
    imputation then has nothing to act on.
  - The report gives the false-improvement rate of each case.
- **The verdict text quotes the limitation**: rare unfavourable outcomes that
  are always rejected can go unobserved, and the error control does not cover
  them.

**M16.4 Continuity wording** (p6 minor 1). The A1 evidence (M15.1) reads:
"two captures totalling 7,061 records over 2,213 s; no discontinuity detected
by this check (band widths 1.001 s and 1.000 s)".
- **Limit of the check.** With exactly 100 ms samples and whole-second
  timestamps, the complete band is 900 ms. One skipped record makes it 1,000 ms,
  which is still below 1 s + τ_band. So a skip is detectable only when sample
  lengths vary enough to populate both band edges.
- **Our captures** have lengths of 106–119 ms, which populate the edges, but
  this is not a guarantee.
- **A fixture documents** the undetectable case.

**M16.5 Sizing feasibility** (p6 minor 2).
- **Preflight.** After sizing, the runner computes per arm the planned block
  duration: idle window, warmup, and Σ_clips (N·latency + S + 2G + S′ + clip
  load + untimed replay), using the sizing latencies. It checks:
  - every block ≤ A_MAX − 60 s;
  - the fastest arm's planned windows ≥ 12 s, which holds by construction.
- **Infeasible conditions.** A condition with any arm failing the preflight
  stops before screening and is reported. It is not repaired ad hoc.
  - Example: a fastest replay of 100 ms gives N = 127. An arm at 600 ms then
    needs 609.6 s for its eight windows alone, which is infeasible.
  - Changing a clip set, G or A_MAX is a plan amendment.
- **Persistence.**
  - The sizing record is frozen in the session file: N per condition and clip,
    the latencies, and its SHA-256.
  - Every later stage of that condition (screen, confirm) must pass that N to
    the block process. N is part of the stage fingerprint (c1 finding 7). A
    stage whose N differs from the frozen sizing record refuses to start.
- **Guard band.** If a later stage's measured window is shorter than 12 s (the
  fastest replay ran faster than sizing assumed), the affected windows are
  flagged and reported. The stage continues unchanged.

**M16.6 Verdict conditions** (summary). An inferential improvement is
conditional on all of the following, each stated in the verdict text:
- A1: producer continuity;
- A2: post-spawn origin;
- A3: constant rate;
- the external-disturbance origin of rejections and the +50% envelope;
- the generating model of the simulation;
- the stress-family limitation.
