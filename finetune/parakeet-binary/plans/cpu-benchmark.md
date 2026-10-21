# Plan: native 1-bit CPU kernels and benchmark for MB1 (Ryzen 9 9950X3D)

**Status: 2026-10-20: plan review round 1 NOT CLEAN (9 major, 1 minor); this
revision addresses every finding ("Review round 1" at the end); round 2 NOT CLEAN (2 major, 3 minor), addressed ("Review round 2");
round 3 NOT CLEAN (2 major, 1 minor), addressed ("Review round 3"); round 4 NOT CLEAN (1 major: multiplicity), addressed by screen-then-confirm ("Review round 4"); **round 5 CLEAN** (review/cpu-benchmark/p5.md; its one minor fixed). Implementation and all timed runs start only after the GPU kernel work
(plans/gpu-kernel.md) has finished, because its processes load the CPU.**

## Question

How fast, and how energy-efficient (package energy, if the counter is
readable), is MB1's encoder on this CPU when it runs from 1-bit weights?
The comparison is with MB1 on the existing ternary CPU path, LocalFlow's INT8
VNNI kernel, which was measured for M1 on 2026-10-08. The answer may be that the
baseline wins; a native kernel is declared better only if it shows an
improvement under the rule below.

## What exists (M1, `../parakeet-ternary/cpu-bitplanes/`, run `20261008T121546`)

**LocalFlow.** A Rust program; the measured snapshot (revision 3bbbb0c) is in
the run directory. It runs Parakeet on the CPU.
- Activations are quantized per frame into residual INT8 parts,
  x ≈ s·(q0 + q1/254 + q2/254²) (`i8x2`, `i8x3`), stored as u = q + 128.
- **VNNI baseline:** the 2-bit codes are repacked into 64-row blocks. Each block
  is expanded once into an s8 tile in L2 with VBMI, and the inner loop is
  `vpdpbusd`.
  - The required dot is d = Σ w·(u − 128), corrected by 128·Σw.
  - The epilogue applies the residual components in descending order, then
    `(sum × weight_scale) × frame_scale`.

**Bit-plane variants.** These were exact bit-serial versions, run on 16 cores
with i8x2. VNNI took 49.7 ms per full transcription:

| Variant | vs VNNI |
| --- | ---: |
| masked | +55% |
| pn | +113% |
| swar | +221% |

No variant was faster on any utterance. In that run all backends were resident
in one process, which inflated the decoder by 13–16% (a cache confound).

**Energy** was not measured: RAPL `energy_uj` is root-only and
`perf_event_paranoid` = 2.

## Approaches (one agent each, competing)

All candidates keep LocalFlow's activation quantization and its epilogue,
including the floating-point operation order. They must produce exactly the
baseline's integer d for every (row, output, component) and therefore
bit-identical FP32 outputs.

Notation, over valid K only:
- b ∈ {0, 1} is the weight bit, w = 2b − 1;
- P = popcount(b) over the row, so Σw = 2P − K;
- U = Σ u over the frame-component.

| Agent | Approach and exact integer identity |
| --- | --- |
| V | **VNNI from 1-bit storage.** The weights stay 1-bit in memory and are expanded in registers just before `vpdpbusd`: `vpmovm2b` turns a 64-bit mask into bytes of 0 / −1. With z = Σ u·(−b) from `vpdpbusd`, the dot is **d = −2z − U − 128(2P − K)**. There is no s8 tile in L2: the codes take half the bytes of the 2-bit codes and an eighth of the tile's. |
| T | **Lookup tables (T-MAC style).** For each frame-component and each group of g ∈ {4, 8} consecutive inputs, precompute all 2^g signed partial sums Σ ±(u − 128), as INT16 entries: \|entry\| ≤ 8·128 = 1,024 for g = 8. A weight's g-bit index selects an entry; selected entries are accumulated in INT32. If a narrower partial accumulator is used, its interval must be proven (at most K/g entries per INT16 partial, flushed before \|sum\| can exceed 32,767). There is no table requantization. |
| P | **Bit-serial control.** The single-plane binary form of `masked`, using the negative mask n = ¬b and planes a_r of u: **C = Σ_r 2^r · popcount((a_r XOR n) AND valid)** and **d = C − 128·N₊ − 127·N₋**, where N₊ = P and N₋ = K − P. Plane layout is free (VPOPCNTDQ, `vpshufbitqmb`, GFNI). This is mainly a measured control for the expectation that bit-serial still loses: it saves only the p load and half the weight bytes compared with `masked`. |

Agent H owns the harness: the snapshot, the binary weight interface, the
contract tests, the energy meter and the runner. Any agent may propose a
per-shape hybrid, backed by data.

## Interface and integration (H, first)

1. **Weight interface.** A `BinaryMatrix` type (codes, scales, K, N; exact
   `parakeet-binary-v1` layout: LSB first, bit 1 = +1, nonzero tail bits
   rejected) and a binary `Projector` path in `lf-asr`.
   - The binary path replaces `PackedTernary` for all 264 modules, so a binary
     run never holds 2-bit codes.
   - Candidates implement one `BinaryGemm` trait (prepare once; then `gemm` with
     LocalFlow's slot / reuse semantics). The VNNI baseline runs MB1 through the
     unchanged ternary path from the derived 2-bit export
     (`gpu1bit/H/derived/MB1-ternary-layout`).
2. **Contract tests (Rust, before any candidate is timed):**
   - **Exact equality** of d and of FP32 outputs. The oracle is an independent
     scalar integer reference with LocalFlow's FP32 epilogue in its exact
     order, for every shape; the unchanged VNNI baseline is also compared
     directly on the shapes it supports (N a multiple of 64, K a multiple of 4,
     which covers every real model shape). Cases:
     - random, ragged and extremal shapes (K not a multiple of 64, N
       not a multiple of the tile, M = 0, M = 1, odd frame counts);
     - extremal activations (all u = 0, all 255, alternating) and sign patterns
       (all +1, all −1, alternating, a single −1);
     - both precisions, and thread counts 1, 4 and 16.
   - **Dispatch:** instrumented counters show that all 264 modules call the
     candidate and none fall back to VNNI.
   - **Reuse:** slot / reuse with changed inputs and changed lengths, so that
     stale tables or tiles are caught.
   - **Native equals derived:** unpacking the native binary codes gives the
     derived 2-bit codes exactly.
   - **Affinity:** each worker's actual affinity is read back
     (`sched_getaffinity`) and the run fails if pinning failed; the inherited
     pool ignores pinning failures silently.
   - **End to end:** bit-identical encoder output hash, tokens and text against
     MB1 on VNNI, on every timed dev clip and on a frozen dev sample of 128
     utterances.

## Storage and workspace (the GPU plan's rule A5, carried over)

- **Weight storage.** After loading and warmup, an allocation audit
  (instrumented allocator) counts the unique live bytes holding weight codes
  across the whole process. This includes the original export bytes, padding,
  auxiliary tables and caches. For the native candidates the limit is
  1.10 × 75,497,472 bytes.
  - The loader must release the export buffer once the matrices are built,
    otherwise it counts.
  - Scales and biases are excluded from the limit and reported.
- **Workspace.** Per-block tiles and per-frame tables are allowed: at most
  32 MiB in total at 16 threads. Peak workspace is reported per shape and per
  thread count.
- **Preparation time.** Any per-call preparation, such as building tables or
  expanding planes, is inside the timed scopes.

## Workloads (dev only for screening, tuning and selection)

Before any candidate timing, H freezes and hashes the workloads, from dev
manifests only, with the split asserted:
- **16 timed dev clips:** the 5 dev corpora × the clips closest to 16 s and
  28 s, plus 6 more at those targets chosen round-robin by a fixed rule.
- **The 128-utterance exactness sample.**

M1's historical 16 test clips are run only after the winner is frozen. They are
reported separately, labelled as a comparison with the multi-backend
measurement of 2026-10-08.

## Measurement

**Arms:**
- MB1 on VNNI (the baseline);
- M1 on VNNI (a control for drift since 2026-10-08);
- each candidate, and each proposed hybrid.

**Conditions:** each arm × {i8x2, i8x3} × {16 cores (CPUs 0–15, SMT siblings
idle), 4 cores (CPUs 0–3)}.

**Isolation.** Every measurement block is a fresh process that holds only that
arm's model and backend. The parent process drives the schedule.

**Blocks.** Within a block:
- warmup is identical for every arm: the longest clip twice, then one pass over
  all clips;
- then one window per clip;
- each window repeats that single clip for at least 3 s, with the energy
  counter and the timer read at the same boundaries.

**Schedule.** A replicate is one complete Williams design for the actual
number of arms k: k sequences if k is even, 2k if odd, each sequence one block
per arm. It is generated and recorded before measurement, and every block
carries its replicate, sequence and position IDs.
- Screening and the three secondary conditions use R = 1 over all arms and
  are descriptive only.
- Confirmation (see "Selection") uses R = 6 over two arms.
- R is fixed now; there is no adaptive extension.
- The window count per arm and clip follows from the design and is reported.
- H states the expected wall time before the run (about 60 s per block).

**Scopes:** full transcription, encoder, projections and decoder; the decoder
code is identical in every arm.

**Host activity.** Host activity is audited during the idle, warmup and timed
phases. A phase is invalid if foreign busy cores exceed 0.5, and invalid
phases are rerun, not dropped silently.

**Frequency.** The governor and EPP are recorded and not changed
(`amd-pstate-epp`, `powersave`, `balance_performance`, boost on).

**Energy.** `custom/cpu-inference/hardware.py` `CpuEnergyMeter` is reused: its
package-domain discovery, range-aware wraparound handling and read-failure
checks.
- **Access:** the counter is readable only if the user grants it, per boot:
  `sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj`.
- **Records:** raw counter values and timestamps are kept for every window.
- **Gross energy:** **E = package joules per transcription** is primary.
- **Net energy:** E_net = (ΔJ_window − P_idle·Δt_window) / transcriptions_in_window
  is reported separately. P_idle comes from
  a 10 s idle window inside the same lock, audited, before each block. E_net is
  not clamped, and invalid values are flagged.
- **Interpretation:** this is package energy under this workload. It is not
  process-isolated and not whole-device energy.
- **Without access** the benchmark is latency only, and the report says so.

**Locks.** The existing machine-wide CPU lock,
`/mnt/hd/wilderness-labs-stt/cpu-inference/active.lock`, is the only one.
- Timed blocks, including idle and warmup, hold it exclusively (`flock`
  LOCK_EX).
- Builds, unit tests and microbenchmarks hold it shared (LOCK_SH), so they never
  overlap a timed block.

## Selection (frozen now): screen, then confirm one candidate

The roster (V, T, P and any hybrids) is frozen before screening. No candidate
or hybrid is added afterwards.

**Stage 1, screening (descriptive).**
- Design: one replicate of the full design in the primary condition (16 cores,
  i8x2) over all arms.
- Statistic: for each candidate against the MB1-VNNI baseline, the median over
  the 16 clips of each clip's mean paired % change in gross package energy per
  full transcription, or in latency if energy is unavailable.
- Pick: the candidate with the lowest (statistic, name) is the **confirmation
  candidate**. No significance claim is made at this stage.
- Fallback: if no candidate has a negative screening median, the baseline wins,
  no confirmation is run, and the best native candidate is reported
  descriptively.

**Stage 2, confirmation (the only inferential claim).**
- Design: fresh measurements of just two arms, the baseline and the
  confirmation candidate, in the primary condition.
  - R = 6 complete Williams replicates for k = 2.
  - New processes; the screening data are not reused.
- Statistic: as in stage 1.
- Interval: a crossed bootstrap, 10k draws, seed 20261020 (frozen). Each draw
  resamples the 6 replicate IDs with replacement and one shared vector of 16
  clip IDs with replacement, keeps all sequence observations and the arm
  pairing, computes each sampled clip's mean across the sampled replicates,
  then takes the median (Owen & Eckles 2012).
- Coverage check: before confirmation, H simulates null data with shared
  replicate offsets, shared clip effects and noise, at magnitudes estimated from
  the screening run's spread. It checks two things for the whole stage-2
  procedure: the 95% interval's empirical coverage, and its false-improvement
  rate. If coverage is below 93% or the false-improvement rate is above 3.5%,
  the result is reported as descriptive only.
- **Improvement:** declared only if the interval's upper bound is below 0.
  Otherwise the baseline wins. A single confirmed comparison means there is no
  multiplicity.

The other conditions (4 cores; i8x3) are measured descriptively for all arms,
with R = 1 and no interval or claim.

## Rules

- **External repository.** LocalFlow is read-only. The M1 snapshot is copied to
  `/mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/snapshot/`. New code is
  added files plus patches, recorded with SHA-256s as `prepare.py` did.
- **Team.** As in plans/gpu-kernel.md: ownership by directory under
  `finetune/parakeet-binary/cpu1bit/`, notes through a flock-serialized `note`,
  contract tests first, no commits.
- **Review.** Codex reviews this plan, and the code before the recorded run,
  until CLEAN.

## Report

`cpu1bit/RESULTS.md`, per arm × precision × thread count:
- latency by scope, energy E and E_net, and paired changes; a CI only for the
  confirmed primary comparison (after its coverage gate), everything else
  labelled descriptive;
- weight bytes and peak workspace;
- the exactness evidence and the dispatch audit;
- the historical-clip comparison, labelled;
- what each approach tried and why it won or lost.

## Out of scope

Training, decoder kernels, binarized activations (W1A1), governor or power-limit
changes, and Apple devices.

## Review round 1 (2026-10-20)

Codex, `/mnt/hd/wilderness-labs-stt/parakeet-binary/review/cpu-benchmark/p1.md`:

1. **Cache confound:** each block is a fresh process holding only one arm's
   model and backend, rotated by the parent.
2. **FP32 bit-identity:** exact integer corrections are stated per approach, and
   LocalFlow's epilogue operation order is kept.
3. **LUT widths:** INT16 entries, INT32 accumulation, proven intervals, no
   requantization, extremal tests.
4. **Storage:** a whole-process allocation audit, the exact denominator, the
   export buffer released, a workspace bound, and preparation time inside the
   timed scopes.
5. **Energy granularity:** single-clip windows of at least 3 s give 16 paired
   per-clip observations.
6. **Energy validity:** `CpuEnergyMeter` is reused with raw counters kept; gross
   energy is primary and E_net separate and unclamped; the interpretation is
   limited to package energy.
7. **Selection:** primary statistic, CI rule and tie order are frozen, and the
   baseline can win.
8. **Integration contracts:** the interface comes first, followed by dispatch,
   reuse, native-equals-derived, extremal, affinity and frozen-dev tests.
9. **Locks:** the canonical `active.lock`, exclusive for timed blocks including
   idle and warmup, shared for builds and tests.
10. **Order balance:** a Williams design for the actual number of arms.

## Review round 2 (2026-10-20)

`review/cpu-benchmark/p2.md`; findings 1–6 and 8–10 resolved, and the integer
identities verified against the snapshot.

1. **(major) Ties.** The winner rule is now deterministic: a tie set anchored on
   A*, then the secondary condition, then latency, then name.
2. **(major) Block variation.** The CI is a two-stage bootstrap over replicates
   and clips, with R fixed at 3 for the primary condition.
3. **(minor) Ragged oracle.** An independent scalar integer oracle with the
   exact epilogue; VNNI is compared directly on the shapes it supports.
4. **(minor) E_net.** Normalization is defined per window.
5. **(minor) Balance.** Complete Williams designs, with the window count derived
   from the design.

## Review round 3 (2026-10-20)

`review/cpu-benchmark/p3.md`; findings 3–5 of round 2 resolved, and the
identities verified again.

1. **(major) Too few replicates for a 95% claim.** R = 6 for the primary
   condition, with a coverage check on simulated shared-offset nulls; below 93%
   the result is descriptive only. The other conditions are descriptive only.
2. **(major) Crossed, not nested.** The bootstrap draws replicate IDs and one
   shared clip-ID vector per draw, keeps the pairing, and takes clip means
   before the median.
3. **(minor) A* ties.** A* is chosen by (statistic, name), and the bootstrap
   seed is frozen.

## Review round 4 (2026-10-20)

`review/cpu-benchmark/p4.md`. Round-3 findings resolved.

1. (major) Individual intervals did not protect the selected-winner claim. Fix:
   the roster is frozen before screening, and one candidate is chosen
   descriptively. It is then confirmed against the baseline in fresh, independent
   two-arm measurements (R = 6). The coverage check also covers the
   false-improvement rate of that procedure.
