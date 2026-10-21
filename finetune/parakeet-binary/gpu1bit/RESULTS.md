# Native 1-bit GPU kernels for MB1 on the RTX 5090: results (DRAFT)

**Status 2026-10-21: DRAFT.**
- Done:
  - the fresh selection session sel-20261020b. Its winner is **B-hybrid** under the anchored tie-set
    implementation of A14 (Codex GPU code review c2, 2026-10-21); the earlier top-two implementation
    had picked B-hybrid-b01;
  - the integration: runtime.py deploys B-hybrid through selected.json;
  - activation precision: recognition and timing, with Codex actprec review c4 CLEAN;
  - **Codex GPU code review round 4 is CLEAN** (`review/gpu-kernel/c4.md`, no fresh findings), on the
    tree with the c3 fixes.
- **The x3-only recorded benchmark is superseded.** The user made INT8x2 the deployment default
  (2026-10-21). The recorded benchmark is now plans/x2-default.md's six-arm benchmark (b1 to b6), after
  its dev gates (g0 to g5) and its own Codex code review. M1's historical test workloads run within it.
- **x2 code review c1 NOT CLEAN** (`review/x2-default/c1.md`, 8 majors in H's files), fixed contract-first:
  - test scoring requires its qualification: per-attempt pins of a complete, accepted benchmark, the
    gates, the preflights and the Z6/Z8 runtime records;
  - Y3 retry evidence is opened record by record;
  - torn JSONL tails are repaired under the attempt lock; retries reach every failed utterance;
  - benchmark reports and the historical authorization use only accepted measurements;
  - the memory contract requires all three clips in every recording;
  - explicit-kernel and historical x3 callers stay x3 after promotion;
  - a promotion contract runs the whole suite in a promoted scratch copy.
- Every number below comes from dev workloads only (A13).
Plan: [plans/gpu-kernel.md](../plans/gpu-kernel.md) (amendments A1–A15 are final) and
[plans/x2-default.md](../plans/x2-default.md) (the INT8x2 default; CLEAN at plan review round 4).

## Summary (selection, dev workloads)

**Robust conclusion (fresh session sel-20261020b, corrected code, clean quiet-host timing).** Any of
the four 1-bit int8_mma variants saves energy per transcription, paired over dev_24:
- **vs M1's expanded runtime (R-expanded):** 8.3–11.8% less energy, 24/24 utterances for every
  variant.
- **vs M1's hybrid runtime (R-hybrid):** 5.0–5.7% less for int8_mma, B-hybrid and B-hybrid-b01, and 8.0%
  for int8_mma:b01. All are lower on 24/24.
- **Memory:** about 0.7 GB less process VRAM (1.50 GB vs 2.22–2.23 GB).
- **Quality:** transcripts are identical to R-expanded's.

The choice between the {0,1} (b01) and ±1 operand forms is second-order. See the b01 caveat below.

**Selection under A14 with plan amendment A15 (anchored tie set; a retrospective amendment, Codex GPU
code review c2/c3):**
- **Winner: B-hybrid.** It runs B's `int8_mma` 1-bit kernels (±1 operands) under R-hybrid's stack, with
  B's exact integer deferred split-K (`hybrid_exact`).
- **Energy vs R-expanded:** −8.52% per full transcription (24/24 utterances), at −7.49% latency, the
  fastest of the 1-bit variants.
- **How it was decided:**
  - int8_mma:b01 has the lowest energy vs R-expanded (−11.83%), so it is the anchor A*.
  - Two candidates tie with A* head to head under A14:
    - B-hybrid-b01: +0.63%, lower on 14/24 vs A*'s 10;
    - B-hybrid: A* −1.87%, but A* is lower on only 15/24 and B-hybrid on 9, so neither reaches 16.
  - int8_mma does not tie (A* −2.93%, lower on 20/24).
  - Within that tie set, A14's latency decides: B-hybrid −7.49%, B-hybrid-b01 −5.29%, int8_mma:b01
    −3.88% vs R-expanded.
- **Change from the earlier decision:** the top-two implementation compared only int8_mma:b01 with
  B-hybrid-b01 and picked B-hybrid-b01. B-hybrid, third by energy, also ties with int8_mma:b01 and is
  faster, but it was never compared. See "Winner rule" below.
- **The relations are nontransitive:** B-hybrid-b01 beats B-hybrid head to head by energy (−1.18%,
  lower on 18/24), yet both tie with A*. The frozen rule anchors on A* and does not use that pairing.
  B-hybrid is 2.3% faster than B-hybrid-b01 and uses 1.2% more energy.
- All six candidates pass A12 and A5, and all 1-bit variants peak at 1.50 GB process VRAM, vs 2.22 GB
  for R-expanded and R-hybrid.
- Details: "Fresh selection session sel-20261020b" below. The paragraph below is the superseded first
  session (sel-20261020, frozen FP32-deferred provider).

Superseded (sel-20261020): the winner was **B-hybrid**. It runs B's `int8_mma` 1-bit kernels under R-hybrid's stack: M1's
fastest runtime, with only the weight storage and the projection kernels replaced. MB1's weights
stay resident at 1 bit (75,497,472 B of codes). Activations use M1's INT8x3 residual arithmetic, and
the INT8 ±1 weight operands are expanded in registers. On the dev 24-utterance set,
B-hybrid compared with R-hybrid (runtime vs runtime) used **9.05% less GPU energy per full
transcription** (lower on 23/24 utterances) and **6.36% less time** (24/24). Compared with
R-expanded it was −11.82% energy and −9.38% latency. Process VRAM fell by 0.62–0.73 GB. Text, token
IDs and timestamps matched MB1 on R-expanded on 24/24 timing utterances and 512/512 dev-sample
utterances. The two float approaches (BF16x3 and FP32 SIMT from 1-bit weights) pass every quality
gate but use 50% and 82% more energy. That is a measured negative result.

## Codex code review round 1 (c1: NOT CLEAN) and the corrections

Review: `/mnt/hd/wilderness-labs-stt/parakeet-binary/review/gpu-kernel/c1.md`. A fresh selection
session on the corrected code is required; its results will replace the sel-20261020 tables below.

1. **Major: B-hybrid's deferred FP32 partials can break A3, and A3 never tested the deferred path.**
   - **Harness.** `gpu1bit/shared/deferred_a3.py` (contract `tests/test_deferred_a3.py`) qualifies the
     actual deferred providers. Each layer is called as R-hybrid's `row_fusions` calls it, and a
     returned DeferredProjection is finished by `inference.row_fusion.finish_residual` (looked up at call
     time, as in the pipeline) with a zero residual and factor 1, i.e. the exact value the pipeline adds
     to the residual stream.
   - **Real cases.** All 264 layers × 15 dev fixture utterances = 3,960 evaluations, 1,440 of them deferred.
   - **Synthetic cases.** 220 split-boundary cancellation cases, including Codex's counterexample verbatim.
     An M1-exact family puts +a and −a 1,024 apart, which is the same split part for both M1
     references; 37 cases have both M1 references exactly at 0 error, so they discriminate.

   | Provider | Real failures (worst ratio to the limit, Frob. / guard) | Synthetic failures | Discriminating failures |
   |---|---|---|---|
   | B-hybrid, frozen (FP32 deferred slices) | 0 / 3,960 (0.691 / 0.921) | 35 / 220 | **35 / 37**, incl. Codex's (rel. error 1.0) |
   | **B-hybrid, fixed** (B's `hybrid_exact.HybridProjectionExact`, int64 slices) | 0 / 3,960 (0.674 / 0.888) | **0 / 220** (max rel. 2.9e-8) | 0 / 37 |
   | **B-hybrid-b01, fixed** | 0 / 3,960 (0.674 / 0.888) | 0 / 220 | 0 / 37 |
   | R-hybrid (information only) | 0 / 3,960 (0.69 / 0.914) | 0 / 220 (relative to M1) | 0 / 37 |

   Caveat: A3 is relative to M1. In 183 of the 220 synthetic cases M1's own references also lose
   the small residual (rel. 1.0), so the limit is loose there.
2. **Major: gate records could be reused across freezes.**
   - Gate records are now run-scoped and carry a fingerprint (`shared/fingerprint.py`): every source a
     run can import, both exports, the workloads, the package versions, the implementation, the
     reference and the patch modules.
   - Records of another fingerprint are refused, a source change after the gate fails the verdict, and
     every record carries the GPU identity.
   - The verdict is always recomputed; `verdict.json` is output only.
3. **Minor: ranking against different baselines.** Corrected; see "Winner rule" below.
4. **Minor: A12 encoder criteria excluded the timing workloads.**
   - Timing rows now record the encoder metrics, and the gate enforces finite outputs, equal lengths and
     rel. Frobenius ≤ 1e-3 over the union of the 512-utterance sample and the 24 timing workloads.
   - Eligibility also requires the P-paired encoder checks on all 24 dev utterances.
5. **Minor: reclassified decoder bytes left the category totals.**
   - The entries and their bytes now move into `dense_bytes`, with storage counts updated and every
     category total conserved; `raw` keeps the unmodified audit.

The B-hybrid figures in this document marked "sel-20261020" are for the frozen FP32-deferred provider.

## Codex GPU code review round 2 (c2: NOT CLEAN) and the corrections

Review: `/mnt/hd/wilderness-labs-stt/parakeet-binary/review/gpu-kernel/c2.md`. Round-1 findings 1, 3, 4
and 5 are resolved. Finding 2 (fingerprints) was not completely resolved, and there are two new
findings.

1. **Major: the A14 implementation considered only the top two by energy.**
   - **The ambiguity.** A14's text defines a tie between two candidates. With three or more, the c1
     implementation compared only the first two by energy.
   - **A retrospective amendment (plan amendment A15).** It was written after sel-20261020b's results
     were known, and it changes the winner from B-hybrid-b01 to B-hybrid. The frozen GPU implementation
     was top-two. The method is the anchored tie set that the CPU plan already used
     (plans/cpu-benchmark.md review round 2: "a tie set anchored on A*, then the secondary condition,
     then latency, then name"; full text in `review/cpu-benchmark/p4.log`, 2026-10-20 08:43). That
     text shows the method existed. It is not a prior commitment for this GPU selection (Codex GPU
     c3). A15 adapts it to A14's tie definition:
     1. A* = the eligible candidate with the lowest (median full-scope energy change vs R-expanded,
        name).
     2. S = A* plus every eligible candidate whose head-to-head energy with A* is an A14 tie
        (|median| < 1%, or neither lower on at least 16 of 24).
     3. Within S, the lowest median full-transcription latency vs R-expanded wins, then the lower
        head-to-head latency, then the name.
   - **Code and tests.** Implemented in `shared/selection.py` (`winner_rule`, `a14_tie`; the docstring
     records the correction). The contract tests in `tests/test_selection_rules.py` cover three and
     more candidates:
     - a third-ranked tie member that is fastest;
     - nontransitive ties, where S is anchored on A* and not closed under ties;
     - a tie member that beats another head to head;
     - A* chosen by name;
     - latency tie-breaks, then the name.
     The earlier tests' "top_two" fields were replaced (a contract change, explained in the test). Their
     two-candidate outcomes are unchanged. sel-20261020 still amends to B-hybrid by energy.
   - **Result on sel-20261020b: the winner changes from B-hybrid-b01 to B-hybrid** (see the Summary).
     The decision is regenerated from the RECORDED clean timing, in
     `H/selection/sel-20261020b/report-corrected.json` (`selection regenerate`). report.json stays as
     recorded.
2. **Major: the export fingerprint covered only the weights digest.**
   - `shared/fingerprint.py` `exports()` now hashes both complete export directories (MB1's and the
     derived 2-bit export): every file, including manifest.json with the configuration the loaders
     build NeMo from, and the tokenizer files. This is the definition of fulldev's `export_identity`.
     The loaders still verify every file against the manifest.
   - Tests: changing only `config.preprocessor.normalize` (per_feature → all_features) changes the
     fingerprint, while the weights digest is unchanged. It also fails the verdict of gate records
     bound before the change, and restoring the file restores the fingerprint.
   - **Re-run under the new fingerprints** (agent-jobs `gpu1bit-selb-fp2-checks`, shared GPU lease,
     2026-10-21 04:06–04:15, check run `sel-20261020b-fp2`):
     - the A12 gates of all six candidates: all pass, 0/512 token mismatches, encoder max rel. Frobenius
       ≤ 1.1e-6 over 536 utterances, no problems;
     - the A5 audits of all eight variants, now carrying their fingerprint: identical bytes to the
       session's audits;
     - the deferred-A3 qualification, now with a fingerprint and run-scoped: fixed B-hybrid and
       B-hybrid-b01 0/3,960 real and 0/220 synthetic failures; the frozen FP32-deferred providers (B-hybrid
       and B-hybrid-b01) 35/220 synthetic failures each, 35 of the 37 discriminating cases, as before;
       R-hybrid (information) 0;
     - the deferred-A3 contract test passes.
     Eligibility in the corrected report uses only these re-run records, bound to the current
     fingerprints. Hybrid-stack candidates must also pass deferred A3.
   - **The recorded timing stays valid; checked by `selection.verify_timing_binding`** (contract tests
     `TimingBinding`):
     - the archived sources `frozen-sources-sel-20261020b.tar.gz` (sha256 d49fe9af…) equal the freeze
       manifest file for file: 166 files, none missing, extra or changed;
     - each of the 34 timing steps has an accepted sidecar attempt whose freeze check passed (166
       files, none mismatched, none added), with load1 ≤ 2;
     - each output's inode change time is no later than the end of its accepted attempt, so it is
       unchanged since acceptance. Each output names its protocol, run, roster variants and range;
     - the roster, the freeze manifest and every file and directory of both exports were last changed
       before the first accepted attempt started (inputs last changed 2026-10-20 13:32:19; first attempt
       17:48:41).
     So the timing measured exactly the frozen sources on the same export bytes that the re-run gates
     now fingerprint. M1's EnergyMeter (`custom/inference-efficiency/energy.py`) is outside the freeze;
     its last change was 2026-09-11 (sha256 40a2bbcf…).
   - **Source state.** After the corrected report, the only source change is `shared/selected.json`
     (the switch to B-hybrid), checked with `fingerprint.sources()` against the report's recorded
     fingerprint.
3. **Minor: the published reproduction commands failed, and the recorded-winner label was wrong.**
   Both are corrected: see "Reproduction" and "Recorded benchmark".

## Codex GPU code review round 3 (c3: NOT CLEAN) and the corrections

Review: `/mnt/hd/wilderness-labs-stt/parakeet-binary/review/gpu-kernel/c3.md`. The round-2 tie-set and
export-fingerprint findings are resolved. The review also confirmed that the clean medians recompute,
the 34 output ctimes fall inside their accepted attempts, and the b01 caveat is supported.

1. **Major: deferred-A3 eligibility could pass without qualification.**
   - **The bug.** `deferred_a3 qualify --no-real --no-synthetic` reported passes=True, and its
     fingerprint did not include the scope. The corrected-decision path accepted such a report, and the
     ordinary `report` path ignored deferred A3 altogether.
   - **The harness now only passes complete runs.** `shared/deferred_a3.py` sets passes=True only for
     the complete scope (`full_scope`: every fixture utterance, real and synthetic, part count 3) with
     zero failures. A partial run is marked partial and is informational only. Each report records its
     scope, the provider identity (registry kernel and factory) and a fingerprint that includes the
     scope.
   - **One validator, `selection.deferred_a3_check`**, used by both reporting paths through
     `evaluate_eligibility`. A report counts only if it has:
     - the candidate's provider and module, and the current provider identity;
     - not informational;
     - the full scope, with the full expected coverage: 264 × 15 = 3,960 real evaluations, 1,440 of
       them deferred, 0 deferral mismatches, all 220 synthetic cases with discriminating ones present;
     - a fingerprint bound to the current source state;
     - zero failures and passes=True.
   - **Problem classes.** Anything else is a provenance problem. Failures are a quality outcome: the
     candidate is ineligible.
   - **Shared eligibility.** `report` and `regenerate` now share `evaluate_eligibility`: the A12 verdict,
     a bound A5 audit, the P-paired encoder checks, and deferred A3 for hybrid-stack candidates. Provenance
     problems are kept separate from quality failures.
   - **Generality.** The validator is parametrized by candidate, provider module and part count (roster
     keys `deferred_module`, `deferred_components`; the part count is in the scope), for further hybrid
     arms such as a B-hybrid-x2.
   - **Tests.** `tests/test_eligibility.py`: partial runs never pass; the fingerprint includes the
     scope; the expected coverage; each of 16 refusals; failures are quality, not provenance; both paths
     use the shared validator; part-count generality.
2. **Minor: the pre-registration was overstated.** The rule is now plan amendment A15 in
   `plans/gpu-kernel.md`, described as retrospective with the exact algorithm.
   - **Freeze.** The plan file is in every gpu1bit fingerprint and every session freeze from now on, so
     any new measurement binds to A15's text. Plan sha256 at the freeze: `6d4975fc…`, recorded in
     `report-corrected.json` "rule".
   - **Reports.** `report.json` (top two, as recorded) and `report-corrected.json` (A15) stay separate.
3. **Minor: the reproduction reused a stale check run and could overwrite the accepted decision.**
   - **Check-run names follow the source state.** `selection checks --run <session>` runs the A12 gates,
     A5 audits and deferred-A3 qualifications under `checks_run_name`, which is
     `<session>-chk-<fingerprint hash>`. A source change gives a new name, so records of another source
     state are never reused. `regenerate` uses that name by default.
   - **Guarded writes.** `write_decision` writes a decision only without provenance problems. It keeps
     the previous file as `<stem>.previous-<time>.json`; otherwise it writes nothing and names the
     problems.
   - **Tests.** `DecisionWrites` (no write on provenance problems, the accepted file stays byte-identical;
     a source change gives a new check-run name).
   - **On the real data.** `regenerate --checks sel-20261020b-fp2`, the stale run, was refused because
     its gates' sources changed, and `report-corrected.json` stayed byte-identical.

**Checks re-run for the deployed source state** (the deployed selected.json, A15 in the plan):
- agent-jobs `gpu1bit-selb-checks-c3`, shared GPU lease of 4G (peak 3.6 GB), 2026-10-21 08:07–08:19,
  check run `sel-20261020b-chk-dbb832641f2d`.
- **Deferred A3:** the contract test passes. Fixed B-hybrid and B-hybrid-b01, full scope: 0 / 3,960
  real (1,440 deferred) and 0 / 220 synthetic failures, 37 discriminating cases with 0 failures.
- **A12 gates:** all six candidates pass, with 0/512 token mismatches.
- **A5 audits:** all eight are bound.

**The corrected decision, reproduced by the documented commands** (`selection checks`, then
`selection regenerate`):
- no provenance problems; all six candidates eligible;
- A* = int8_mma:b01;
- tie set {int8_mma:b01, B-hybrid-b01, B-hybrid};
- **winner B-hybrid**, by latency vs R-expanded.
`report-corrected.json` was rewritten with the same decision; the previous file is kept as
`report-corrected.previous-20261021T081941.json`.

## Fresh selection session sel-20261020b (corrected code; clean quiet-host timing)

**Setup.**
- Roster: R-expanded, R-hybrid, int8_mma, B-hybrid (fixed), int8_mma:b01, B-hybrid-b01 (fixed),
  triton_bf16:bf16x3, bitserial.
- Frozen 2026-10-20 13:32 (166 files,
  `/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/freeze-sel-20261020b.sha256`). The freeze was
  verified before every process.
- Reports: `.../H/selection/sel-20261020b/report.json` (`selection report`, as recorded, top-two
  implementation) and `report-corrected.json` (`selection regenerate`, A15 anchored tie set; checks of the
  current source state; current decision).

**How the timing was taken.**
- **First timing run: set aside as contaminated (host load ~50 from 13:35).**
  - Another session's agent-jobs runs (`spell-pilot-*`, CPU-heavy) kept the load average at 47–51 on 32
    cores.
  - Their short-lived workers also opened GPU contexts, so phases chunks failed the foreign-PID guard.
  - Its outputs are in `bench/contaminated/sel-20261020b-hostload50/` (with `LABEL.json`). Nothing below
    comes from it.
- **Gates and audits** ran during the busy period (numerics and storage, not timing). All of them are
  valid.
- **Timing rerun.** It ran 2026-10-20 17:47–18:51 as one agent-jobs run, `gpu1bit-timing-selb`:
  - Flags: exclusive `--needs gpu`, `--quiet-host`, `--full-machine`, `--priority normal`. The user
    granted full GPU control for this run, and that is recorded in every sidecar.
  - Command: `gpu1bit/rerun_timing.py --timing-only`, which runs the session's own frozen commands in the
    session's order.
  - Guards before every process: load1 < 2, no cpu-heavy holder, no GPU compute process, freeze verified.
    During each process, load was sampled every 1 s, and a chunk was to be dropped if load1 rose above 2.
  - **Result: 34 of 34 steps accepted on the first attempt.**
    - load1 max 1.58 (pre-checks max 1.35).
    - Foreign busy cores: max 1.37, median 0.06.
    - No cpu-heavy holder at any time, and no foreign GPU process.
    - Sidecars: `selection/sel-20261020b/host/<step>.json`.

**P-paired, dev_24, median of per-utterance changes vs R-expanded** (lower = count of 24 utterances):

| Variant | Full: ms | Full: J | Full energy vs R-exp. | Full latency vs R-exp. | Encoder energy vs R-exp. | Encoder latency vs R-exp. |
|---|---|---|---|---|---|---|
| R-expanded | 4.573 | 1.517 | | | | |
| R-hybrid | 4.376 | 1.473 | −3.14% (23) | −4.16% (24) | −4.57% (24) | −5.22% (24) |
| int8_mma | 4.371 | 1.386 | −8.30% (24) | −4.90% (24) | −9.01% (24) | −7.92% (24) |
| **B-hybrid** | 4.242 | 1.362 | −8.52% (24) | −7.49% (24) | −10.11% (24) | −11.26% (24) |
| int8_mma:b01 | 4.484 | 1.349 | −11.83% (24) | −3.88% (24) | −8.03% (24) | −7.05% (24) |
| B-hybrid-b01 | 4.366 | 1.326 | −9.17% (24) | −5.29% (24) | −9.85% (24) | −8.87% (24) |
| triton_bf16:bf16x3 | 6.204 | 2.301 | +49.91% (0) | +35.68% (0) | +59.91% (0) | +59.84% (0) |
| bitserial | 7.616 | 2.823 | +80.80% (0) | +67.23% (0) | +105.93% (0) | +107.21% (0) |

**Head-to-head comparisons (paired, full scope):**
- Against their stack baseline R-hybrid: B-hybrid −5.48% energy (24/24), −3.31% latency;
  B-hybrid-b01 −5.73% (24/24), −1.18%.
- B-hybrid-b01 vs B-hybrid: −1.18% energy (lower 18/24), +2.26% latency (lower 3/24).
- int8_mma:b01 vs int8_mma: −2.93% energy (20/24), +2.37% latency (0/24).
- Encoder scope, b01 vs ±1: energy −0.05% (hybrid) and +0.92% (provider), latency +2.7% and +2.9%.
  So {0,1} saves no encoder energy; its full-pipeline saving comes from elsewhere (see the phases).

**Winner rule (A14 with the anchored tie set; `report-corrected.json`).** All six candidates are
eligible (re-run gates and audits, deferred A3 for the hybrid stacks).
1. A* (lowest energy vs R-expanded): int8_mma:b01, −11.83%. Then B-hybrid-b01 −9.17%, B-hybrid −8.52%,
   int8_mma −8.30%, triton_bf16:bf16x3 +49.91%, bitserial +80.80%.
2. Head to head with A* (paired energy, A* vs candidate; utterances lower for A* / for the candidate):

   | Candidate | Median | A* lower | Candidate lower | A14 tie |
   |---|---|---|---|---|
   | B-hybrid-b01 | +0.63% | 10 | 14 | yes (< 1%) |
   | B-hybrid | −1.87% | 15 | 9 | yes (neither ≥ 16) |
   | int8_mma | −2.93% | 20 | 4 | no |
   | triton_bf16:bf16x3 | −40.28% | 24 | 0 | no |
   | bitserial | −51.50% | 24 | 0 | no |

3. Tie set S = {int8_mma:b01, B-hybrid-b01, B-hybrid}. Its latency vs R-expanded: B-hybrid −7.49%,
   B-hybrid-b01 −5.29%, int8_mma:b01 −3.88%.

**Winner: B-hybrid.**
- Inside S the head-to-head relations are nontransitive. B-hybrid-b01 vs B-hybrid is −1.18% energy
  (lower on 18/24): b01 beats ±1 directly, while both tie with A*.
- The rule is anchored on A* by design, as in the CPU plan, and decides within S by latency.
- As recorded on 2026-10-20 with the top-two implementation (`report.json`), the winner was
  B-hybrid-b01. That decision is superseded.

**Quality and storage.**
- **Quality:** full-scope transcripts of all 8 variants are identical to R-expanded's on 24/24. Encoder
  checks pass for all (max rel. Frobenius vs R-expanded 6.3e-7).
- **A12 gates:** all six candidates pass. 0 token mismatches on the 512-utterance sample. Encoder max
  rel. Frobenius ≤ 1.1e-6 over the 536-utterance union.
- **A5 audits:** every 1-bit variant has 75,497,472 B of weight codes, within the budget. R-expanded and
  R-hybrid hold 773,849,088 B. For the hybrid stacks, 11,562,815 B of decoder-side storage is
  reclassified as dense, with totals conserved.

**P-clips / P-memory** (fresh processes; forward-order values shown, reverse order agrees within 0.7%).
Dev clips of 3, 10 and 30 s:

| Variant | 3 s | 10 s | 30 s | Peak process VRAM |
|---|---|---|---|---|
| R-expanded | 2.74 ms / 0.745 J | 4.51 ms / 1.534 J | 10.52 ms / 3.795 J | 2.23 GB |
| R-hybrid | 2.58 ms / 0.711 J | 4.35 ms / 1.489 J | 10.39 ms / 3.757 J | 2.22 GB |
| **B-hybrid** | 2.49 ms / 0.685 J | 4.21 ms / 1.378 J | 9.99 ms / 3.551 J | 1.50 GB |
| B-hybrid-b01 | 2.54 ms / 0.677 J | 4.31 ms / 1.334 J | 10.02 ms / 3.535 J | 1.50 GB |
| int8_mma:b01 | 2.59 ms / 0.650 J | 4.42 ms / 1.346 J | 10.13 ms / 3.593 J | 1.50 GB |

**Phases: B-hybrid (a) vs B-hybrid-b01 (b).** Medians over dev_24. Power, SM clock and memory clock were
traced every 20 ms. Texts are identical on 24/24.

| Phase | ms a / b | J a / b | b lower (of 24) |
|---|---|---|---|
| full | 4.095 / 4.196 (+2.1%) | 0.9895 / 0.9514 (−1.5%) | 17 |
| forward (frontend + encoder) | 2.774 / 2.851 (+2.8%) | 0.7622 / 0.7747 (+0.2%) | 11 |
| encoder | 2.654 / 2.727 (+2.7%) | 0.7178 / 0.7334 (+0.3%) | 12 |
| decode | 1.369 / 1.365 (−0.5%) | 0.2061 / 0.2026 (−0.7%) | 16 |
| host_and_idle (full − forward − decode) | | 0.0183 / 0.0022 | 16 |

- **Clocks:** SM clocks are equal (2,909 vs 2,912 MHz over full; 2,862 vs 2,866 in the encoder). The
  memory clock is 13,801 MHz for both.
- **Power over the full transcription:** 238 W vs 223 W.
**b01 caveat: a small edge with an only partly explained mechanism.**
- **Size:** b01's edge over ±1 B-hybrid is small. Paired over full transcriptions it uses 1.2% less
  energy (lower on 18/24 utterances) and takes 2.3% more time.
- **Encoder phase:** energy is +0.3% at +2.7% time, so average power is about 2.4% lower. That fits lower
  tensor-core switching activity with 0x00/0x01 operand bytes than with 0xFF/0x01. The decoder phase
  is −0.7%.
- **Most of the full-scope difference sits in the residual**, host_and_idle = full − forward − decode
  (0.018 J vs 0.002 J per transcription, b01 lower on 16/24). By construction this residual holds
  everything in a full transcription that the separately measured forward and decode phases do not:
  - the gaps between the phases within one transcription;
  - host-device transfers outside forward;
  - synchronization;
  - Python work: decode set-up, hypothesis and text construction;
  - GPU idle power during those host gaps;
  - and, because each phase is timed in its own windows, the window-to-window differences in power
    state. Its time component is slightly negative (−0.03 ms), so it is a difference of separately
    measured quantities, not a clean phase.
- **The residual difference is not explained by anything b01 changes.** b01 changes only the weight
  operand encoding inside the encoder's projection kernels, and none of the residual's components
  involves those kernels.
- **The direction reproduced** in three independent sessions: b01's lower full-scope energy appeared in
  B's two runs and in this one.

Footnote (preliminary, contaminated, not a result): on the set-aside run, the full-scope energy vs
R-expanded was int8_mma:b01 −11.8%, B-hybrid-b01 −8.6%, B-hybrid −8.4%, int8_mma −8.4%, R-hybrid −3.4%,
triton_bf16:bf16x3 +50.9% and bitserial +81.7%. R-expanded's own full-scope latency was about 8% higher
than in sel-20261020, which is consistent with the host contention.

## What was compared

| Name | What it is | Weights resident |
|---|---|---|
| R-expanded | MB1 (derived 2-bit export) on `enable_optimizations(encoder_storage="expanded")`: INT8x3, INT8 weight cache | 2-bit + INT8 cache |
| R-hybrid | MB1 (derived) on `experimental_pipeline(model, "ptx_row_best")` (A11): Triton-PTX CUDA kernels, deferred row epilogues | 2-bit + INT8 cache |
| int8_mma (B) | B's 1-bit kernels via H's `BinaryProjection` in R-expanded's stack (controlled substitution, A2) | 1-bit |
| **B-hybrid (B)** | B's 1-bit kernels via B's `HybridProjection` in R-hybrid's stack (runtime vs runtime) | 1-bit |
| triton_bf16:bf16x3 (A) | Triton BF16x3 tensor-core GEMM decoding 1-bit words in registers, Kahan-compensated, in R-expanded's stack | 1-bit |
| bitserial (C) | FP32 SIMT FFMA from 1-bit weights expanded per CTA, Fast2Sum-compensated, cluster split-K, in R-expanded's stack | 1-bit |

R-bf16x3 (`inference.kernels.matmul(mode="bf16x3")`) is a kernel-level reference only (A1, A12).

## Method

- **Environment (A7).** `gpu1bit/shared/python` is the parakeet-ternary environment (torch 2.11.0+cu128,
  Triton 3.6.0, NVRTC 12.9) with parakeet-binary on PYTHONPATH. The harness asserted the same decoder
  CUDA-graph mode (`full_graph`), decoder fusions, frontend settings (dither 0, pad_to 0) and
  TF32/cuDNN flags (off) for every variant. A mismatch aborts the run.
- **GPU discipline (A9).** One `paths.gpu_lock` per process, held at most 114 s in this session.
  Each process ran as `timeout 600`. A power limit of exactly 400 W and no foreign compute process were
  asserted before every window. The limit was never changed.
- **Freeze.** All candidate and harness sources are pinned in
  `/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/freeze-20261020.sha256` (78 files). That list
  was verified before each of the 40 session processes and after the session.
- **Session.** `sel-20261020`, 10:55–11:35, driven by `gpu1bit/shared/selection.py session`.
  Raw data is under `/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/H/`: `bench/sel-20261020*`,
  `gate/<candidate>/`, `selection/sel-20261020/` (report.json, ledger.json, audit/, logs/).
- **Workloads (A13).** These were frozen in `gpu1bit/shared/workloads.json` before any timing:
  - dev clips: LibriSpeech dev-clean 3.005, 10.0 and 29.4 s (M = 38, 126, 368);
  - dev_24: 8 utterances per 3, 10 and 30 s target, round-robin over the five dev corpora;
  - dev sample: 512 utterances evenly spaced over the five dev manifests.

  No test audio was used.
- **P-paired (A6).** This is the CUDA_IMPROVEMENTS.md method:
  - all six variants are resident in each chunk process (4 utterances per chunk);
  - each variant gets three 1 s windows per utterance, in alternating forward/reverse order, each after a 0.3 s warm-up;
  - NVML board energy is counted over exactly the timed calls;
  - each utterance's median window is used, and the reported figure is the median of per-utterance paired changes, with counts;
  - encoder scope is GPU features → encoder output as a raw graph replay;
  - full scope is a CPU waveform → CPU text, with caches cleared between utterances.
- **P-clips / P-memory (A6).** This is the EXPERIMENTS4.md / compare_v2 method:
  - a fresh process per variant, run twice, once in forward and once in reverse order;
  - the dev clips run in order 3, 10, 29.4 s with an accumulating graph cache;
  - each clip gets a 2 s warm-up, then three 5 s windows;
  - NVML process VRAM is sampled every 10 ms from before the model loads.

### Review item 2: all variants resident in each P-paired process

This is M1's own paired method (`cuda_experiments.multi_encoder` / `multi_pipeline`). M1's
CUDA_IMPROVEMENTS.md says "Both models stay resident, so this process is not used for deployment
VRAM".

It is acceptable for **latency and energy**:
- each 1 s window runs one variant's work while every other model is idle;
- there are no concurrent kernels, and all variants share one CUDA context, clocks and thermal history;
- windows of all variants interleave per utterance in balanced order, so drift is shared, not attributed;
- idle allocations add no board power, because all GDDR7 is refreshed whether allocated or not;
- idle allocations don't occupy L2 during another variant's window, and the 0.3 s warm-up before each window displaces the previous variant's L2 lines.

It is **not** acceptable for VRAM. NVML process memory is the sum over all resident models, graph
pools and allocator reservations. Per-variant memory therefore comes only from P-memory, with one
fresh process per variant. The same text is in `gpu1bit/shared/selection.py`'s docstring.

## Selection results: P-paired on dev_24

Medians of per-utterance paired changes (count of 24 utterances lower), then the median per target
(3 / 10 / 30 s; 8 utterances each).

**Full transcription (CPU waveform → text):**

| Candidate vs its baseline | Latency | GPU energy | Avg power | Energy by target 3 / 10 / 30 s |
|---|---|---|---|---|
| **B-hybrid vs R-hybrid** | **−6.36% (24/24)** | **−9.05% (23/24)** | −2.58% | −9.67 / −9.32 / −7.04 |
| int8_mma vs R-expanded | −4.70% (24/24) | −8.37% (24/24) | −2.26% | −8.77 / −8.09 / −6.09 |
| triton_bf16:bf16x3 vs R-expanded | +36.02% (0/24) | +50.21% (0/24) | +11.77% | +47.70 / +50.95 / +51.22 |
| bitserial vs R-expanded | +67.01% (0/24) | +81.58% (0/24) | +10.63% | +80.56 / +84.62 / +76.88 |
| (R-hybrid vs R-expanded) | −3.26% (24/24) | −2.90% (24/24) | +0.68% | |
| (B-hybrid vs R-expanded) | −9.38% (24/24) | −11.82% (23/24) | −1.84% | |

**Encoder only (graph replay):**

| Candidate vs its baseline | Latency | GPU energy | Avg power |
|---|---|---|---|
| B-hybrid vs R-hybrid | −9.26% (24/24) | −9.87% (24/24) | +0.15% |
| int8_mma vs R-expanded | −7.92% (24/24) | −8.27% (24/24) | −0.21% |
| triton_bf16:bf16x3 vs R-expanded | +59.78% (0/24) | +58.81% (0/24) | +0.29% |
| bitserial vs R-expanded | +107.0% (0/24) | +104.9% (0/24) | +0.28% |
| (R-hybrid vs R-expanded) | −5.15% (24/24) | −4.47% (24/24) | +0.12% |

**Medians over the 24 utterances**, absolute (mixed 3–30 s; use the paired changes above for comparisons):

| Variant | Full ms | Full J | Full W | Encoder ms | Encoder J |
|---|---|---|---|---|---|
| R-expanded | 4.566 | 1.515 | 332.5 | 2.972 | 1.103 |
| R-hybrid | 4.409 | 1.481 | 336.9 | 2.820 | 1.038 |
| int8_mma | 4.366 | 1.408 | 322.8 | 2.736 | 1.014 |
| **B-hybrid** | **4.147** | **1.345** | 326.3 | **2.559** | **0.953** |
| triton_bf16:bf16x3 | 6.210 | 2.308 | 368.5 | 4.749 | 1.751 |
| bitserial | 7.599 | 2.783 | 366.8 | 6.163 | 2.270 |

**Checks.**
- Encoder lengths were equal and outputs finite for every variant on 24/24 utterances. The maximum relative Frobenius difference from R-expanded (valid frames) was 4.2e-7 (int8_mma), 4.6e-7 (B-hybrid), 5.1e-7 (triton_bf16), 6.3e-7 (bitserial) and 4.8e-7 (R-hybrid).
- In the full scope, text, token IDs and timestamps were identical to R-expanded on 24/24 for every variant. B-hybrid was also identical to R-hybrid on 24/24.

### Winner rule (A14), applied mechanically

**Implementation corrected after Codex code review round 1, finding 3. The rule text is unchanged.**

**As implemented in this session (superseded):**
1. Eligible: candidates that pass A12 and A5. All four do.
2. Ranking by median paired energy change in the full scope, each against its own stack baseline:
   B-hybrid −9.05% vs R-hybrid, int8_mma −8.37% vs R-expanded, triton_bf16 +50.21%, bitserial +81.58%.
3. The top two differ by 0.68 points, which is under 1%, so latency decided. B-hybrid was lower:
   4.147 vs 4.366 ms median, and −5.00% head to head (24/24).

Codex pointed out that comparing changes against different baselines can reverse a ranking regardless
of the candidates themselves.

**Corrected implementation** (`gpu1bit/shared/selection.py`: `winner_rule`; contract tests in `gpu1bit/tests/test_selection_rules.py`):
1. Rank the eligible candidates by median paired energy change against one common reference, R-expanded.
2. Apply "differ by less than 1%" and "lower on at least 16 of 24" to the paired candidate-vs-candidate
   differences of the top two.
3. If neither tie condition holds, the candidate that is lower in that paired energy comparison wins;
   otherwise the one lower in the paired latency comparison wins.

Applied to this session's data (`selection/sel-20261020/report-amended.json`):
- B-hybrid −11.82%, int8_mma −8.37% against R-expanded.
- Head to head, B-hybrid is −3.99% energy, lower on 23/24.
- Not a tie, so **B-hybrid wins by energy**; the stated latency tie-break was wrong.

The winner did not change. The coordinator confirmed B-hybrid on 2026-10-20, and
`gpu1bit/shared/selected.json` records it. Codex's finding 1, below, then required a fresh session.

## A12 recognition gate (reference: MB1 on R-expanded)

| Candidate | Encoder max rel. Frobenius (valid frames) | Encoder max abs | Token-ID mismatches, 512 dev sample | WER, ref = candidate | Timing workloads identical | Passes |
|---|---|---|---|---|---|---|
| int8_mma | 9.78e-7 | 9.39e-7 | 0 / 512 | 5.954% = 5.954% (8,952 words) | 24 / 24 | yes |
| **B-hybrid** | 1.05e-6 | 8.79e-7 | 0 / 512 | 5.954% = 5.954% | 24 / 24 | yes |
| triton_bf16:bf16x3 | 1.12e-6 | 1.37e-6 | 0 / 512 | 5.954% = 5.954% | 24 / 24 | yes |
| bitserial | 9.17e-7 | 1.18e-6 | 0 / 512 | 5.954% = 5.954% | 24 / 24 | yes |

The limits are a rel. Frobenius of at most 1e-3, at most 0.5% token mismatches (≤ 2 / 512), WER within
0.02 points, and the timing workloads identical (dev_clips ∪ dev_24 = 24 unique utterances).

## A5 storage audit (whole transformed model, after warm-up and capture of 3 pipeline graphs)

| Variant | Weight-code bytes | Within 83,047,219 B (1.10 × 75,497,472) | FP32 scale bytes |
|---|---|---|---|
| int8_mma | 75,497,472 | yes | 1,769,472 |
| **B-hybrid** | 75,497,472 (raw 87,060,287; see review item 1) | yes | 1,769,472 |
| triton_bf16:bf16x3 | 75,497,472 | yes | 1,769,472 |
| bitserial | 75,497,472 | yes | 1,769,472 |
| R-expanded | 773,849,088 = 2-bit 150,994,944 + fused-QKV 2-bit copies 18,874,368 + INT8 cache 603,979,776 | no (baseline) | 2,064,384 |
| R-hybrid | 773,849,088 (raw 785,411,903) | no (baseline) | 2,064,384 |

How the audit works (`gpu1bit/shared/audit.py`):
- it finds every CUDA tensor reachable from the model, the provider, patched functions, closures, default arguments, transform contexts and candidate module globals;
- it deduplicates storages by `untyped_storage().data_ptr()`;
- it sorts storages into dense / activation / weights;
- within weights, a storage whose contents equal a layer's FP32 scale or bias vector (or the Q/K/V concatenation) is reported as scale/bias, and everything else counts as code.

### Review item 1: hybrid-stack decoder-side state reclassified as dense

This affects R-hybrid and B-hybrid only.
- **Why the raw figure is wrong.** These stacks keep their transform contexts in suspended generator frames, not in `model._packed_optimization`. The audit's weight walk therefore first reaches the fused decoder's state through the functions that M1's `decoder_transform` / `control_transform` patch onto `decoder.*` and `joint.*` modules.
- **What that state is.** The precomputed first-layer token table, FP32 [1025, 2560] = 10,496,000 B. The label-looping computer's state: `encoder_output_projected` [1, T, 640], the `batched_hyps` transcript and timestamps, decoder states and `_packed_decoder_keepalive`.
- **Where the rule lives.** `selection.reclassify_decoder_side`. A storage classed as "code" whose first path starts with `decoder.`, `joint.` or `decoding` is moved to dense:
  - its entry leaves the weights list and is listed under `reclassified_decoder_side`;
  - its bytes leave `weight_code_bytes` and are added to `dense_bytes`, and the storage counts move with it, so every category total is conserved (corrected after Codex c1, finding 5);
  - `raw` keeps the unmodified audit (totals, counts and entries), and `raw_weight_code_bytes` the unadjusted figure.
- **Why it is justified.** These storages derive from the dense FP32 decoder/joint weights and decoder activations, not from the 264 quantized encoder layers. With R-expanded and provider stacks, `audit.audit` itself classifies the same objects as dense, via the decoder_transform / control_transform generators in `model._packed_optimization`.
- **Amounts.** The selection moved 11,562,815 B for both R-hybrid and B-hybrid, and 0 B elsewhere. The integration test moved 11,568,255 B; the decoder state buffers are sized by the last decoded utterance. After adjustment B-hybrid holds exactly MB1's code bytes.

## P-clips and P-memory (dev clips; fresh process per variant; forward / reverse order runs)

| Variant | 3.005 s ms / J / W | 10.0 s ms / J / W | 29.4 s ms / J / W | Process VRAM incl. load + first capture, GB (3 / 10 / 29.4 s) |
|---|---|---|---|---|
| R-expanded | 2.741 / 0.744 / 271 | 4.50 / 1.536 / 341 | 10.52 / 3.80 / 361 | 1.741 / 1.887 / 2.225 |
| R-hybrid | 2.578 / 0.713 / 276 | 4.34 / 1.484 / 342 | 10.39 / 3.73 / 360 | 1.739 / 1.883 / 2.223 |
| int8_mma | 2.528 / 0.678 / 268 | 4.30 / 1.401 / 326 | 10.10 / 3.59 / 355 | 1.120 / 1.158 / 1.497 |
| **B-hybrid** | **2.367 / 0.640 / 270** | **4.11 / 1.355 / 329** | **9.88 / 3.49 / 353** | **1.118 / 1.156 / 1.495** |
| triton_bf16:bf16x3 | 3.113 / 1.097 / 352 | 6.18 / 2.279 / 368 | 15.39 / 5.71 / 370 | 1.017 / 1.164 / 1.502 |
| bitserial | 3.950 / 1.347 / 341 | 7.58 / 2.793 / 368 | 18.23 / 6.73 / 369 | 1.015 / 1.162 / 1.502 |

The two runs per variant agree within about 0.4% (values are means of the two runs; per-run JSON is
in `bench/sel-20261020-clips{1,2}/`). All 12 runs produced identical text per clip.

B-hybrid compared with R-hybrid: −0.621 / −0.727 / −0.728 GB of process VRAM. This is the
604 MB INT8 cache plus 151 MB of 2-bit buffers, less what 1-bit codes and B's workspaces need.

## Per-layer numerics (A3; 264 layers × 15 frozen dev fixture utterances = 3,960 evaluations)

The limits are per layer and utterance: 1.5 × the worse of R-bf16x3 and R-expanded INT8x3 on the same
input, applied to both the relative Frobenius error and the elementwise guard. The limits' median is
2.0e-7 (rel. Frobenius).

| Candidate | Failures | Worst ratio to limit (Frobenius / guard) | Fix that made it pass |
|---|---|---|---|
| int8_mma | 0 | 0.674 / 0.888 | INT32-exact split-K (FP32 split partials failed: up to 1.16x) |
| triton_bf16:bf16x3 | 0 (731 before the fix) | 0.55 / 0.67 | Kahan/TwoSum running sum across K tiles |
| bitserial | 0 (39, then 616 with other configs, before the fix) | 0.68 / 0.90 | Fast2Sum-compensated 16-k partials |
| h_reference (FP64, H's known-good) | 0 | 0.354 / 0.355 | — |

Root cause (found independently by A and C): FP32 rounding of large per-K-tile partial sums. It shows
up on positional-embedding rows (linear_pos) and on post-LayerNorm inputs. M1 passes only because
its default split-K of 8 keeps each part's sum short.

## Screening (phase 2, provisional, P-kernel, warm / L2-cold µs per Conformer block, 8 in-graph ops)

These numbers come from different runs of `gpu1bit/shared/bench.py kernel`, and each run has its own
baseline rows. Compare within a column only roughly. P-kernel never decided selection (A14).

| M (dev clip) | int8_mma (B) | triton_bf16:bf16x3 (A) | bitserial (C) | R-expanded | R-hybrid | R-compact | R-bf16x3 |
|---|---|---|---|---|---|---|---|
| 38 | 47.6 / 68.4 | 72.1 / 91.1 | 107.7 / 130.5 | 52.3 / 94.0 | 49.7 / 96.8 | 65.9 / 92.0 | 86.7 / 123.3 |
| 126 | 69.1 / 95.1 | 148.5 / 171.2 | 205.3 / 223.0 | 77.7 / 116.4 | 77.9 / 117.2 | 101.7 / 134.2 | 155.4 / 184.0 |
| 368 | 140.7 / 174.8 | 364.0 / 380.5 | 498.5 / 474.3 | 160.1 / 189.0 | 155.7 / 193.8 | 224.4 / 240.0 | 342.0 / 362.2 |

Sources: B `B/bench/20261020T103100-kernel.json` (baselines from that run), A
`A/bench/20261020T093529-kernel.json`, C `C/bench/20261020T085024-kernel.json`.

## Approaches tried, and why each won or lost

- **B, INT8 IMMA from 1-bit weights (`int8_mma`): won, as B-hybrid.**
  - Weights are re-laid out at load time so each lane's 32-bit word holds its m16n8k32 B fragment as bit planes.
  - Bits are expanded to INT8 ±1 in registers (SHF + LOP3 + IMAD per 4 weights), reused across 3 activation parts and the row tiles.
  - Activations are M1's own INT8x3 (`_quantize`), so the arithmetic class equals R-expanded's.
  - Split-K is exact INT32, in one kernel: warp groups (WK) or a thread-block cluster with an L2 exchange. Scales are prefetched.
  - Why it wins: 75.5 MB of weights stream per forward instead of 604 MB, there is no separate finish kernel, and tiles are tuned per (N, K, M band).
  - In B-hybrid, R-hybrid's deferred row epilogues also fold the projection epilogue, residual, next LayerNorm and next quantization into one kernel.
  - The energy gain is mostly time: average power is about the same or lower.
  - Where it loses: single warm ops at M = 38 against R-hybrid's grouped Triton-PTX kernels (out, pw2; about 0.25 µs).
  - Dead ends: DSMEM split-K exchange (+5.3 µs); FP32 partial exchange (fails A3); fused per-slice activation quantization (fails the A3 guard in emulation).
- **A, Triton BF16x3 tensor cores from 1-bit weights (`triton_bf16:bf16x3`): lost by 50% on energy.**
  - Bits are decoded to exact ±1 BF16 in registers; FP32 activations are split into 3 BF16 parts (M1's scheme).
  - It uses the Kahan running sum, a LN/SiLU prep kernel or a fused path, and intra-op PDL.
  - It beats R-bf16x3 (its arithmetic class) at M = 38/126, and is the fastest variant L2-cold at M = 38 per block.
  - Why it loses to INT8x3: per MAC it needs 3 BF16 MMAs at 254 TFLOPS against INT8x3's 3 at 974 TOPS, a 3.8x tensor floor gap. Even at 100% of its floor, ff1 at M = 368 would be slower than R-hybrid. In the pipeline it also draws about 11% more power.
  - FP16x2 (2 MMAs, 22 of 24 bits) failed A3's synthetic single_nonzero case by 6%.
  - Dead ends: batched 3-D `tl.dot` split-K, counter-based split-K, Gluon register-direct operands, and the {0,1} identity with float accumulation (1–8x more error).
- **C, bit-serial / FP32 SIMT (`bitserial`): lost by 82% on energy.**
  - First established that `mma.sync ... .b1` on SM120 is emulated (8 IMMA + LOP3 per m16n8k256; 0.63x INT8 per K). That closes bit-plane tensor-core arithmetic.
  - The kernel then used FP32 FFMA with weights expanded per CTA, Fast2Sum, and a cluster/DSMEM split-K in one kernel.
  - Why it loses: FFMA peak is 57 T MAC/s against INT8x3's effective 162 T MAC/s. At M ≥ 126, even 100% of FFMA peak is slower than R-hybrid.
  - LOP3 + POPC with 24 activation planes would reach about 10.5 T MAC/s.
- **Common lessons.**
  - FP32 accumulation of large per-tile partials needs compensation to meet A3 on real activations.
  - The 1-bit win on this GPU comes from weight traffic, which is why it shows up L2-cold and in the full encoder rather than in single warm ops.

## Integration (phase 3 step 2): `gpu1bit/runtime.py`

`load_binary_packed(export_dir, *, runtime="hybrid"|"provider", kernel=<registry name>)`:
- validates the export: `parakeet-binary-v1` format (also in the safetensors metadata); every file's SHA-256 against the manifest; exactly the 264 layers; every layer through `api.validate_weights` (dtype, width, tail bits, finite scales);
- refuses 2-bit exports;
- builds NeMo from the manifest config and the export's own tokenizer. No .nemo and no derived export is touched, and strace proves it;
- prepares the layers with the kernel named in `gpu1bit/shared/selected.json`. Since 2026-10-21 (the
  corrected sel-20261020b decision) that is `int8_mma`. From 2026-10-21 00:05 to 04:15 it was
  `int8_mma:b01`, registered by the selected import `gpu1bit.int8_mma.b01_run`;
- applies the hybrid stack named in selected.json, or the R-expanded stack with `BinaryProjection`.
  Since the switch, the hybrid stack is B's fixed `hybrid_exact.HybridProjectionExact` with
  `exact_hybrid_pipeline` and the `row_exact` finish_residual routing (exact integer deferred split-K).
  It is imported, not copied. The frozen FP32-deferred `hybrid.py` is no longer the deployed path.

`transcribe(model, items)` is M1's `hf.load_ternary.transcribe` at batch 1 under `inference_mode`.
The CLI, `gpu1bit/shared/python -m gpu1bit.runtime --export DIR [--runtime] [--kernel] [--tokens] clip.wav`,
takes the GPU lock for its whole life and exits inside it.

**Swapping the kernel.** A candidate kernel replaces the selected one by changing `"kernel"` (and its
registering `"imports"`) in `selected.json`, after it wins under the same rules. `runtime.py` is not
edited.
- **Current deployment:** `selected.json` names kernel `int8_mma`, the hybrid_exact stack, and selection
  `sel-20261020b` / `B-hybrid` (`report-corrected.json`).
- **Earlier states:** B-hybrid-b01 (top-two decision) is kept in
  `H/selection/sel-20261020b/selected-before-c2-switch.json`, and the original frozen-hybrid
  `int8_mma` file in `selected-before-switch.json`.

**After the switch to B-hybrid (2026-10-21 04:16), all runtime contracts pass on the deployed stack**
(agent-jobs `gpu1bit-runtime-tests-c2`, shared lease, peak 3.6 GB; `gpu1bit-runtime-fresh-c2`,
exclusive). They passed the same way on B-hybrid-b01 on 2026-10-21 00:05:
- `test_runtime`, 15 tests, shared GPU lease. Text, token IDs and timestamps equal MB1 on R-expanded on
  all 24 timing workloads for both runtimes; refusals, graphs and the A5 audit pass.
- `test_finish_residual_routing`. The deployed spec equals B's exact spec: the router is installed, no
  stale reference to M1's original finish_residual exists, and exact deferred projections are used.
- `test_runtime_fresh`, 3 tests, exclusive lease because its children use the EnergyMeter guard.
  Fresh-process VRAM equals the sel-20261020b P-memory of the selected variants: now B-hybrid
  (hybrid) and int8_mma (provider), before that B-hybrid-b01 and int8_mma:b01. The strace check passes.

Integration contract tests (first run 2026-10-20, kernel int8_mma; all pass):
- `gpu1bit/tests/test_runtime.py`, 15 tests, in-process under one lock:
  - refusals: ternary-layout exports (M1's and the derived MB1 layout), a wrong format field, a hash mismatch (corrupted tokenizer files), tail bits (a crafted K = 1020 claim), a missing layer, an unknown kernel, and a kernel without `dot` for the hybrid stack;
  - for each runtime, text, token IDs and timestamps equal MB1 on R-expanded on all 24 dev timing workloads (1,560 tokens);
  - M1 weight kernels launched by the runtime: 0. The same instrumentation counts 840 `_int8_dot` / `ExpandedProjection` calls in R-expanded and 840 `PtxCudaProjection` calls in R-hybrid;
  - graphs for the three dev clip shapes, replayed in both orders and after allocator churn, are bitwise equal to the captured call, with identical decodes; eager matches within M1's capture tolerance;
  - the A5 audit is 75,497,472 B of codes for both runtimes.
- `gpu1bit/tests/test_runtime_fresh.py`, 3 tests. The parent has CUDA hidden, and each child is its own locked GPU process:
  - fresh-process VRAM equals the selection's P-memory to the byte: hybrid 1,117,782,016 / 1,155,530,752 / 1,495,269,376 B at 3 / 10 / 29.4 s, about 0.62–0.73 GB below R-expanded's;
  - the strace check described above (no .nemo, no M1 model directory, no M1 export, no derived export opened).

  Finding along the way: the test runner's `CUBLAS_WORKSPACE_CONFIG=:4096:8` (deterministic cuBLAS for
  in-process tests) costs +25.2 MB at load and +37.7 / +83.9 / +117.4 MB at the three clips. The children
  therefore run without it, as deployment and P-memory did.

## Quality and contract status of every candidate (frozen sources)

| | int8_mma / B-hybrid | triton_bf16:bf16x3 | bitserial |
|---|---|---|---|
| H contract suite (A8) | 32/32 | 19/19 (incl. meta) | 32/32 |
| A3 real numerics | 0 / 3,960 | 0 / 3,960 | 0 / 3,960 |
| compute-sanitizer memcheck / racecheck | 0 errors / 0 hazards (with a positive control) | 0 / 0, incl. LN/SiLU `project` paths | 0 / 0 |
| B-hybrid stack verification | 216/216 provider calls equal R-hybrid's; 588/588 patched attributes identical; M1 weight kernels 0 | — | — |

## Anomalies and caveats

- **dev_24 index 20** (yodas 29.94 s, M = 375): B-hybrid's three full-scope windows spread
  3.667–4.060 J, against R-hybrid's steady 3.865–3.898 J. This is the one utterance where B-hybrid
  was not lower (23/24). It does not affect the ranking.
- **Reclassification variation.** The amount moved by the item-1 rule depends on the size of the
  decoder state buffers at audit time (11,562,815 B in the session vs 11,568,255 B in the
  integration test).
- **Gate verdicts computed afterwards.** The session driver did not compute the gate verdicts. They
  were computed afterwards on the CPU from the session's gate records (`gate/<candidate>/verdict.json`).
- **Renamed analysis module.** `gpu1bit/shared/select.py` was renamed `selection.py` after the session,
  because it shadowed the stdlib `select` module when run from its own directory. Regenerating the
  report with the renamed module gives a byte-identical `report.json`.
- **New B files.** After the session, B added unfrozen files to `gpu1bit/int8_mma/` (`*_b01*`, the
  post-selection variant). All 78 frozen files still match. The selected kernel imports none of them.
- **R-bf16x3 at kernel level only.** It is a kernel-level reference, as A12 allows; there is no BF16x3
  recognition control.

## Activation precision: INT8x2 and INT8x1 recognition on full dev (plans/activation-precision.md)

Every utterance of the five dev sets (22,418) runs at batch 1 through each arm's full deployed runtime,
with its 4-entry pipeline graph cache. All six arms run through `gpu1bit/shared/fulldev.py` with the
hash-pinned patch `gpu1bit/actprec/parts.py`.

**Runs.**
- **Recorded run: `ap-20261020r4`** (2026-10-20 21:37–22:42, agent-jobs `gpu1bit-actprec-r4`, shared GPU
  lease of 4G, peak 3.97 GB). It runs under the identities after the fixes for Codex review c3
  (`review/activation-precision/c3.md`):
  - the recognition run is explicit: the timing orchestrator passes the roster's identity_run to every
    worker (`GPU1BIT_ACTPREC_RUN`), and the timing patch has no default run;
  - the orchestrator's guard gates that run's references before any reuse;
  - timing artifacts must name the run and each variant's reference arm and identity;
  - an x3 summary must name its own run to pass the reference gate.
  - **It is row-identical to `ap-20261020r3`** in the same fields, report entries and encoder checks
    as below.
- **Previous run `ap-20261020r3`** (20:22–21:20, agent-jobs `gpu1bit-actprec-r3`). It ran under the
  identities after the fixes for Codex review c2 (`review/activation-precision/c2.md`):
  - the R2 reference gate guards every recording entry point: the driver, direct `parts run` workers,
    the timed variants (`actprec/bench.py`) and the encoder checks;
  - P-clips/P-memory artifacts record the variant's recognition identity, run and reference identity;
  - timing resume runs the identity guard first and reuses a step only if it is bound to the same
    roster, protocol and identity map with a byte-identical, bound output;
  - a non-finite x3 encoder output on the 512 sample invalidates the reference and withholds
    classification;
  - fulldev prefetches the next utterance's audio in one background thread.
  - **It is row-identical to `ap-20261020r2`**: all 6 × 22,418 rows (tokens, text, encoder lengths
    and frames, timestamps, finiteness, decode errors), the report entries apart from the identities,
    and both encoder checks utterance by utterance.
  - **The prefetch gave no measurable speed-up.** Throughput is within run-to-run noise (M1-x3 63.1
    vs 62.4 utterances/s, MB1-int8_mma-x3 48.6 vs 48.0), so audio loading is not the bottleneck.
- **Previous run `ap-20261020r2`** (19:12–20:12, agent-jobs `gpu1bit-actprec-r2`). It ran under the
  identities after the fixes for Codex review c1 of the activation-precision code
  (`review/activation-precision/c1.md`):
  - fulldev's identity binds the complete export: every file, including manifest.json with its
    configuration and the tokenizer;
  - the report reads only identity-bound, complete artifacts, and comparisons carry both identities;
  - an x3 reference that is not complete, verified, R5-valid and cross-check-passed stops xN launch and
    classification (R2);
  - all six arms, x3 included, are built by `parts.build`. Their identities also hash
    `gpu1bit/actprec/` (including the timing patch `bench.py`) and `gpu1bit/rerun_timing.py`.
- **It reproduces the first run `ap-20261020` (15:09–16:50) exactly:**
  - all 6 × 22,418 rows have identical tokens, text, encoder lengths, timestamps and finiteness;
  - the report entries are identical except for the identities;
  - the 512-sample encoder checks are identical utterance by utterance.

**Arms.**
- Each model's stack is unchanged. Its own projection provider is wrapped in `PartCount(provider, n)`:
  M1's `ExpandedProjection('triton_col_tuned')`, or MB1's `BinaryProjection` over B's `int8_mma:xN` (±1).
- The x3 tile tables are used unchanged (R6).
- B's x2/x1 files are executed from B/wip/parts at the bytes of B's 14:55 manifest.

**R1 contracts.** `tests/test_partcount.py` passes: its CPU tests (identity, R2 gate) and 4 GPU path
tests, the latter rerun under the review-c1 code as agent-jobs `gpu1bit-partcount-paths`.
- Every projection call on the eager warm-up, capture, overflow and direct-eager paths reaches the
  provider with n parts.
- On each path, the call sequence equals the deployed x3 stack's (840 calls for a shape's
  warm-up + capture, 216 eager).
- Replay makes no call, and replays bitwise the direct eager output, which differs from x3's.

**R2.** The x3 arms reproduce the pinned FP32-NeMo dev WERs exactly (0.0 points, mean and every set).
Their tokens, text and encoder lengths are identical to the cross-check run's on all 22,418 utterances,
so `parts.build(n=3)` equals fulldev's original x3 build on all of dev.

| Arm | Dev mean WER | Δ mean vs x3 | Worst set Δ | Token mismatches vs x3, pooled (max set) | Encoder rel. Frob. vs x3, 512-sample, median / max | R5 | Reading |
|---|---|---|---|---|---|---|---|
| M1-x3 | 6.0856% | | | | | valid | reference |
| M1-x2 | 6.0853% | −0.0002 | +0.001 (ami) | 0.02% (0.06% voxpopuli) | 7.6e-5 / 2.3e-4 | valid | **recognition-equivalent** |
| M1-x1 | 6.1101% | +0.0245 | +0.072 (yodas) | 5.05% (7.15% yodas) | 3.8e-2 / 6.5e-2 | valid | trade-off |
| MB1-int8_mma-x3 | 6.3120% | | | | | valid | reference |
| MB1-int8_mma-x2 | 6.3124% | +0.0004 | +0.002 (ami) | 0.01% (0.02% ami) | 7.8e-5 / 2.5e-4 | valid | **recognition-equivalent** |
| MB1-int8_mma-x1 | 6.3333% | +0.0213 | +0.045 (ami) | 5.35% (8.35% yodas) | 3.7e-2 / 7.2e-2 | valid | trade-off |

- **Per-set WER deltas** (clean / other / ami / voxpopuli / yodas):
  - M1-x2: +0.000 / +0.000 / +0.001 / −0.002 / +0.000
  - M1-x1: +0.002 / +0.039 / +0.019 / −0.009 / +0.072
  - MB1-x2: +0.000 / +0.000 / +0.002 / +0.000 / +0.000
  - MB1-x1: +0.002 / +0.004 / +0.045 / +0.020 / +0.036
- **Token-mismatch rates per set** (clean / other / ami / voxpopuli / yodas):
  - M1-x1: 4.51 / 6.88 / 4.49 / 4.68 / 7.15%
  - MB1-x1: 5.40 / 7.61 / 4.39 / 5.31 / 8.35%
- **Empty hypotheses** (x3 → xN): M1 24 → 24 (x2) and 23 (x1); MB1 33 → 33 (x2) and 30 (x1).
- **R5 (structural).** Every arm is valid: all encoder outputs are finite, every decode succeeds, and every
  encoder length equals x3's on full dev and on the 512-sample check.
- **Why x1 is a trade-off.** Both x1 arms miss the 0.02-point mean limit, by a small margin. Both also
  churn about 5% of the token sequences, against limits of 0.5% pooled and 1.0% per set (R7). No set is
  worse by more than 0.10 points.
- **x2.** For both models, x2 is recognition-equivalent to x3 under every criterion. Its sample encoder
  error vs x3 (max 2.5e-4) is also inside the A12 bound of 1e-3, which does not apply here.
- **Descriptive A3 ratios** (264 layers × 15 fixture utterances, against the A3 limits; `H/actprec/a3-parts.json`):
  - x2: 0 / 3,960 within A3. Rel. Frobenius median 3.0e-5, max 3.0e-4; ratio to the limit median 154×, max 237× (guard: 150×, 295×).
  - x1: 0 / 3,960 within A3. Rel. Frobenius median 7.8e-3, max 6.4e-2; ratio median 3.9e4×, max 9.7e4× (guard: 3.8e4×, 1.6e5×).
  - B's int8_mma:xN and M1's own ExpandedProjection at xN, on the same derived weights, give the same
    errors. The error ratio B/M1 is 0.9992–1.0003 at x2 and 1.0000 at x1, so the departure is the part
    count, not B's kernels.
- **Code review:** the activation-precision code passed Codex review round 4 (CLEAN,
  `review/activation-precision/c4.md`) before the timing below was measured.

### Latency and energy (P-paired, P-clips, P-memory)

**How it was measured.**
- One agent-jobs run, `gpu1bit-timing-actprec` (2026-10-20 22:52–23:54): exclusive GPU, `--quiet-host`,
  `--full-machine`, `--priority normal`. Command: `gpu1bit/rerun_timing.py --roster-file
  H/actprec/timing-ap-20261020r4/roster.json`.
- Same harness, 400 W check and locks as the selection session.
- Every timed variant is bound to the recognition run `ap-20261020r4`: its identity and its x3 reference
  are checked before every process and recorded in every artifact.
- 24 of 24 steps accepted; 7 attempts were dropped because load1 rose above 2 mid-run (2.02–9.35). The
  load came from other sessions' tool processes outside agent-jobs (e.g. a `bfs` file scan of
  /mnt/hd); the queued agent jobs were held off by quiet-host. Accepted chunks peaked at load1 1.95.
- Summary: `H/actprec/timing-ap-20261020r4.json` (`actprec_report.py timing`). It reads only accepted
  artifacts that are still bound.
- **The timing runs every arm through the provider/expanded stack** (R-expanded's transforms), not the
  deployed hybrid runtime, so the comparison is controlled. B-hybrid at x2/x1 is out of scope; the plan
  leaves it out.

**Full transcription, dev_24, paired** (median of per-utterance changes; in parentheses, utterances
where the first arm is lower):

| Comparison | Energy | Latency | Average power |
|---|---|---|---|
| M1-x2 vs M1-x3 | −9.78% (24/24) | −2.97% (24/24) | −7.89% |
| M1-x1 vs M1-x3 | −21.07% (24/24) | −8.77% (24/24) | −14.42% |
| MB1-x2 vs MB1-x3 | −11.63% (24/24) | −6.67% (24/24) | −5.17% |
| MB1-x1 vs MB1-x3 | −25.13% (24/24) | −15.13% (24/24) | −11.95% |
| MB1-x3 vs M1-x3 | −8.33% (24/24) | −4.85% (24/24) | −1.64% |
| MB1-x2 vs M1-x2 | −8.66% (24/24) | −7.49% (24/24) | −0.64% |
| MB1-x1 vs M1-x1 | −9.96% (24/24) | −11.09% (24/24) | −0.43% |

Absolute medians, full scope (ms / J / W):

| Arm | ms | J | W |
|---|---|---|---|
| M1-x3 | 4.565 | 1.483 | 328 |
| M1-x2 | 4.433 | 1.312 | 297 |
| M1-x1 | 4.165 | 1.123 | 274 |
| MB1-x3 | 4.374 | 1.356 | 312 |
| MB1-x2 | 4.105 | 1.207 | 296 |
| MB1-x1 | 3.706 | 1.017 | 276 |

**Encoder only, paired:**

| Comparison | Energy | Latency | Average power |
|---|---|---|---|
| M1-x2 vs x3 | −9.60% | −7.24% | +0.1% |
| M1-x1 vs x3 | −20.84% | −16.25% | −5.6% |
| MB1-x2 vs x3 | −12.62% | −10.41% | −1.8% |
| MB1-x1 vs x3 | −26.22% | −24.73% | −1.5% |
| MB1-xN vs M1-xN, x3 / x2 / x1 | −7.74% / −11.88% / −13.65% | −7.81% / −10.89% / −17.68% | |

In every comparison above, at both scopes, the first arm's energy and latency were lower on 24/24
utterances. Encoder outputs were finite with equal lengths for every arm.

On dev_24, full transcripts were identical to x3's for x2 on 24/24 utterances (both models). For x1
they matched on 18/24 (M1) and 22/24 (MB1); the full-dev reading above is the recognition measure.

**P-clips / P-memory** (fresh process per arm; forward order shown, reverse within 1–4%, mostly <1%).
Dev clips of 3 / 10 / 30 s:

| Arm | 3 s | 10 s | 30 s | Peak process VRAM |
|---|---|---|---|---|
| M1-x3 | 2.75 ms / 0.735 J | 4.51 ms / 1.524 J | 10.53 ms / 3.792 J | 2.23 GB |
| M1-x2 | 2.68 ms / 0.665 J | 4.35 ms / 1.347 J | 9.91 ms / 3.502 J | 2.23 GB |
| M1-x1 | 2.53 ms / 0.574 J | 4.09 ms / 1.144 J | 9.29 ms / 3.146 J | 2.23 GB |
| MB1-x3 | 2.53 ms / 0.671 J | 4.30 ms / 1.388 J | 10.13 ms / 3.575 J | 1.50 GB |
| MB1-x2 | 2.36 ms / 0.585 J | 4.05 ms / 1.228 J | 9.33 ms / 3.231 J | 1.50 GB |
| MB1-x1 | 2.14 ms / 0.487 J | 3.65 ms / 1.036 J | 8.56 ms / 2.861 J | 1.50 GB |

**Reading (no automatic switch; the user decides).**
- **x2 is recognition-equivalent and saves energy** in both models: 9.8% per full transcription for M1
  and 11.6% for MB1, with lower latency.
- **x1 is a trade-off.** It saves 21–25% energy, at a dev-mean WER cost of +0.02 points and about 5%
  token churn.
- **MB1 stays 8–10% below M1** in energy at every part count.
- The x2/x1 timing uses the provider stack. The deployed winner B-hybrid has no x2 path; the plan
  leaves that out of scope.

Records: `/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/H/fulldev/ap-20261020r4/` (per-arm identity,
rows, chunks, summary; `compare/`) and `.../H/actprec/` (`report-ap-20261020r4.json`,
`encoder-check-ap-20261020r4-{M1,MB1}.json`, `a3-parts.json`) and `.../H/partcount/`. The earlier runs'
records (`ap-20261020`, `ap-20261020r2`, `ap-20261020r3`) are kept for the reproduction checks.

## Recorded benchmark (SUPERSEDED by plans/x2-default.md; the checklist below is kept for the record)

The x3-only recorded run below is not run. plans/x2-default.md replaces it with a recorded benchmark
of six arms:
- b1 MB1 B-hybrid x2, b2 MB1 B-hybrid x3, b3 MB1 (derived layout) R-expanded x3;
- b4 M1 R-expanded x2, b5 M1 R-expanded x3, b6 M1 R-hybrid x3.

It runs after the dev gates g0 to g5 and a Codex code review of the x2 code, with M1's historical test
workloads within it.

- [ ] Codex (gpt-6-astra xhigh) code review of gpu1bit/ (shared/, tests/, runtime.py, int8_mma/ as
      frozen), iterated until CLEAN, covering review items 1 and 2 above.
- [ ] Snapshot of all gpu1bit sources with SHA-256s into the results (A10).
- [ ] Recorded run of the winner, B-hybrid: `gpu1bit.runtime` (runtime hybrid, kernel `int8_mma`, the exact
      stack `hybrid_exact.HybridProjectionExact` + `exact_hybrid_pipeline` with the `row_exact` routing),
      as named in `shared/selected.json`. It runs under its own new freeze and source snapshot (A10),
      separate from the archived sel-20261020b sources:
  - [ ] P-paired encoder and full scope on dev_24 against R-hybrid and R-expanded. *Table: TBD.*
  - [ ] P-clips / P-memory on the dev clips, fresh processes, order baseline, new, new, baseline. *Table: TBD.*
  - [ ] **M1's historical test workloads, for comparability with M1's recorded numbers only**
        (LibriSpeech test-clean 3.0 / 10.0 / 30.04 s clips, compare_v2 method; the 24-utterance
        multi-corpus test set, cuda_experiments method). These are run after the freeze; no selection
        or tuning uses them. *Tables: TBD (MB1 on the winner vs MB1 on R-hybrid / R-expanded vs
        M1's recorded EXPERIMENTS4 / CUDA_IMPROVEMENTS numbers).*
- [ ] End-to-end transcript agreement on the dev quality sample against MB1 on R-expanded (done in
      A12: 512/512) **and against MB1's FP32 NeMo reconstruction** (`binary_export.load_export`),
      a hard constraint of the plan not yet run. *TBD; mismatches listed, not hidden.*
- [ ] Weight bytes resident, disk sizes (MB1 export 105,299,804 B vs M1 180,797,564 B) and the
      final summary. *TBD.*

## Reproduction

There are two separate source states:
- **The recorded selection session sel-20261020b** ran on 166 frozen files, archived as
  `H/selection/sel-20261020b/frozen-sources-sel-20261020b.tar.gz` (sha256 d49fe9af…) and listed in
  `/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/freeze-sel-20261020b.sha256`. The working tree
  has changed since then, so that session reproduces only from the archive.
- **The recorded benchmark** (still to run) gets its own new freeze file and source snapshot (A10).

**sel-20261020b from the archived sources.** Tested 2026-10-21: the commands below reproduce
`report.json` byte for byte (`cmp` identical).

```sh
H=/mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/H
REPO=/home/aj/workspace/github.com/wilderness-labs/wilderness-labs-stt
X=$H/repro/sel-20261020b-src                       # an empty directory on /mnt/hd
mkdir -p $X && tar xzf $H/selection/sel-20261020b/frozen-sources-sel-20261020b.tar.gz -C $X
(cd $X && sha256sum -c --quiet /mnt/hd/wilderness-labs-stt/parakeet-binary/gpu1bit/freeze-sel-20261020b.sha256)
mkdir -p $X/custom/autoresearch                    # environment, not part of the archive:
ln -s $REPO/custom/autoresearch/.training-runtime $X/custom/autoresearch/.training-runtime   # interpreter
ln -s $REPO/custom/inference-efficiency $X/custom/inference-efficiency                       # M1's EnergyMeter
cd $X/finetune/parakeet-binary && P=gpu1bit/shared/python
CUDA_VISIBLE_DEVICES= $P -m gpu1bit.shared.selection verify --run sel-20261020b        # 166 files, ok
cp -p $H/selection/sel-20261020b/report.json $H/repro/report.recorded.json             # `report` rewrites it
CUDA_VISIBLE_DEVICES= $P -m gpu1bit.shared.selection report --run sel-20261020b
cmp $H/selection/sel-20261020b/report.json $H/repro/report.recorded.json                # identical
```

**Re-measuring from the archive.** Make a new roster under a new run name. `roster` writes a new freeze
of the extracted tree. Then run the archived session driver as one quiet-host job, from `$X/finetune/parakeet-binary`:

```sh
CUDA_VISIBLE_DEVICES= $P -m gpu1bit.shared.selection roster --run sel-repro-1 \
    --variants R-expanded R-hybrid int8_mma B-hybrid int8_mma:b01 B-hybrid-b01 triton_bf16:bf16x3 bitserial \
    --patches gpu1bit.int8_mma.b01_run,gpu1bit.int8_mma.session_patch --phase-pair B-hybrid B-hybrid-b01 \
    --stack-baseline B-hybrid=R-hybrid B-hybrid-b01=R-hybrid
job start sel-repro-1 --needs gpu --quiet-host --cwd $X/finetune/parakeet-binary -- \
    env CUDA_VISIBLE_DEVICES= gpu1bit/shared/python -m gpu1bit.shared.selection session --run sel-repro-1
```

- The recorded timing was taken by `gpu1bit/rerun_timing.py` under `job --quiet-host`. That orchestrator
  is outside the archive. Its SHA-256 at the time is recorded in every sidecar, and it ran the session's
  own commands (`selection/sel-20261020b/logs/`).
- The archived session driver runs those same commands. It has no per-process load checks of its own,
  so the host must be kept quiet by `--quiet-host`.

**The corrected decision** comes from the current tree. It reads the recorded timing and the checks of
the current source state. These commands reproduced it on 2026-10-21:

```sh
cd $REPO/finetune/parakeet-binary
CUDA_VISIBLE_DEVICES= gpu1bit/shared/python -m gpu1bit.shared.selection checks-name --run sel-20261020b
    # -> sel-20261020b-chk-<hash of the current source state>; a source change gives a new name
job start selb-checks --needs gpu:shared --vram 4G --cwd $PWD -- \
    env CUDA_VISIBLE_DEVICES= gpu1bit/shared/python -m gpu1bit.shared.selection checks --run sel-20261020b
    # A12 gates, A5 audits, deferred A3 (full scope) for the roster, one locked process per step
CUDA_VISIBLE_DEVICES= gpu1bit/shared/python -m gpu1bit.shared.selection regenerate --run sel-20261020b
    # -> report-corrected.json (timing binding re-verified); refuses to write on any provenance problem
```

- Any change to a fingerprinted file between `checks` and `regenerate` makes `regenerate` refuse. Run
  `checks` again; it picks the new name.
- The fingerprinted files are shared/, runtime.py, the candidate directories, M1's modules and
  `plans/gpu-kernel.md`.

**The deployed runtime** (current tree):

```sh
cd $REPO/finetune/parakeet-binary
gpu1bit/shared/python -m gpu1bit.tests.run --label H:runtime gpu1bit.tests.test_runtime          # shared lease
gpu1bit/shared/python -m gpu1bit.tests.run --label H:routing gpu1bit.tests.test_finish_residual_routing
gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_runtime_fresh                 # exclusive lease
gpu1bit/shared/python -m gpu1bit.runtime --export /mnt/hd/wilderness-labs-stt/parakeet-binary/runs/main-binB-m1-lr2e-4/export clip.wav
```
(Run each as `job start ... --needs gpu:shared --vram 8G`, or `--needs gpu` for test_runtime_fresh,
whose children use the EnergyMeter guard.)


## To-do (not scheduled)

- Batched full-dev evaluation path (recognition runs only; speed benchmarks stay at batch 1): length-sorted batches, one-time equivalence check against batch 1 per runtime (row-identical, or the A12 gate), Codex review. Requested 2026-10-20; expected 2-4x faster evaluation arms.
