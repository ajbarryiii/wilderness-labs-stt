# WP7 experiment plan: deployed-pipeline gates and the informational sweep of the new arms

Revision 2 (after pre-run review round 1, `reviews/wp7-r1.md`). DESIGN.md revision 9 is authoritative. Nothing in
this plan runs until the review comes back clean. Tests (unit tests, macguard suite, pipegate self-test, the
`--verify-only` deployment check) are not experiments and may run.

## 0. Status of earlier outputs (decision (a))

All gate and sweep outputs made before the review are unreviewed and withdrawn:
- records and summaries: `results/eligibility/pipelines/quarantine/pre-review-20261003/` (no runner, report or
  eligibility check reads it; the runner reads only `results/eligibility/pipelines/<record name>.json`);
- NixOS raw-run pointers renamed `LATEST.quarantined-pre-review`; nothing reads `LATEST` any more (reports are
  manifest-driven, section 5);
- README WP7 says "pending rerun after a clean review".
Everything below is rerun from scratch with the reviewed commit.

## 1. Deploy the reviewed commit to the Mac

```
# NixOS: the reviewed commit is pushed to parakeet-ios
# Mac (repo root): GIT_SSH_COMMAND="ssh -o BatchMode=yes" git pull -q --ff-only
# Mac (finetune/parakeet-ternary):
ios/macguard --rss-cap 4G --timeout 1800 -- sh ios/build_reviewed.sh
```
`build_reviewed.sh` refuses a checkout with modified tracked files, builds release, runs the unit tests (7) and
writes `bench/.build/release/BUILD_INFO.json` (commit, executable SHA-256). Gate jobs count as "clean" only when
that file names their checkout's commit and the binary they run. Then the tests:
```
MACGUARD_DIR=$A sh ios/tests/macguard_tests.sh "$PWD/ios/macguard" /usr/bin/python3 $A/macguard-test-wp4   # Mac
(cd ios && ./macguard --rss-cap 2G --timeout 900 -- pyenv/.venv/bin/python tests/pipegate_selftest.py)      # Mac
```
The self-test mocks a perfect pipeline from WP3's reference and three faults (NaN in a decoder state, a missing
clip, 5% flipped decisions) and requires `evaluate` to pass / fail exactly as expected.

## 2. Regenerate the FP64-feature reference

```
(cd ios && ./macguard --rss-cap 6G --timeout 1800 -- pyenv/.venv/bin/python -m pipegate ref64)   # Mac, about 1 min
```
FP32 reference encoder with FP16-rounded scales (WP3 refcache `fp16s`) on `native.fp64_features` of every clip
(DESIGN.md rev. 9, gate 4, "Deployed-pipeline references"). Its index records the clips.json SHA-256, the front-end
manifest SHA-256, a hash of the reference code (`native.fp64_features`, `pipegate.cmd_ref64`, refcache's scale
setter and model loader, `models.py`, `reference.py`), the model source identity and the refcache provenance;
`evaluate` refuses the reference if any of them differs from what the gate loaded or from the current code.
The index is copied to `results/pipegates/ref64_index.json` in step 4.

## 3. Deployed-pipeline gates

**Pipelines.** Front end A (vDSP FP32, `native/mp2` constants) → encoder arm → decode loop: F2 (native FP32,
`native/mp2` decoder_joint), F0 (Core ML `Decoder` + `JointDecision`, `arms/mp2/decoder-fp32`), F1 (Core ML
`DecoderJoint`, same directory). Decoder models load with the encoder's compute units.

**Combinations.** Every revision-8 WP3 encoder record with `timing_allowed`, except G0 (control), MLX (no Swift
implementation) and the `-dec-fp16` records: 49 encoder combinations × {F2, F0, F1} = 147 pipelines.

| arm | ANE (cpuAndNeuralEngine) | CPU (cpuOnly) | GPU (cpuAndGPU) |
|---|---|---|---|
| C1 | fixed, multi | fixed, multi | – (no GPU record) |
| C3 | fixed, multi | fixed, multi | multi |
| C4 | enum, fixed, multi | enum, fixed, multi | multi |
| C6s2, C6s4 | enum, fixed, multi | enum, fixed, multi | – |
| C6s8 | fixed, multi | fixed, multi | multi |
| C6d4, C6d8 | fixed, multi | fixed, multi | – |
| C7 | enum, fixed, multi | – (stress fails) | – (stress fails) |
| C8 | fixed, multi | – | – |
| C3-ane, C4-ane, C6s8-ane | multi | – | – |

**Clips.** All 82 clips of `clips.json` (64 natural, 16 per bucket; boundary-length, silence, impulse), each in its
own bucket (fixed: the 15 s window).

**Command.** `CUDA_VISIBLE_DEVICES= ./python ios/pipegate.py run` (NixOS; all 49) then `pipegate.py table`.
Per combination:
1. withdraw any existing records/summary of the combination (a failed attempt leaves nothing behind);
2. restore the encoder package if archived; measure the disk need (package + Core ML cache estimate: 4 × package
   for dense C1, else 1 GB; + 1 GB);
3. one macguard job, `pipegate_job.sh` in a unique `<artifacts>/results/pipegate/<combo>/<run id>/`: fail-closed
   disk check (30 GB floor + need), Core ML cache purge (exit 6 if it fails), `build.json` (commit, modified tracked
   files, executable SHA-256, BUILD_INFO match), `parakeet-bench gate` (below), `python -m pipegate evaluate`,
   removal of the per-step diagnostics (their SHA-256s are in gate.jsonl / evaluation.json), purge;
4. retrieval of evaluation.json, gate.jsonl, job.log, cache.log, build.json into `.part`, atomic rename;
5. `pipegate.py record`: refuses an evaluation not made by this pipegate.py / revision 9 / a clean build of this
   checkout's commit; scores WER; writes the summary and the three revision-9 records;
6. archive arms the run restored once no later combination needs them.
`run` checks first that the Mac checkout is this commit with no modified tracked files, and that every requested
combination exists. Exit 0 = all 147 gated and passed; 10 = all gated, some failed; 1 = an orchestration or job
failure (that combination has no records).

**What `parakeet-bench gate` captures per clip and decode loop.** Free decoding: tokens, logical steps, and every
decoder output (finiteness counted). Forced replay of the clip's B0 trace: argmax token / duration per step, logical
steps / predictions, every decoder output (finiteness counted) and per-step diagnostics: F2 logits [1030], h, c;
F0 h, c and, as a diagnostic, JointLogits on the same joint inputs (raw logits behind JointDecision); F1 h_out,
c_out (DecoderJoint exposes no logits). Header: components of every pipeline (below), executable SHA-256.

**Pass criteria per pipeline** (DESIGN.md rev. 9, gate 4b; all must hold):
- coverage: all 82 clips; every trace replayed in full; all 64 natural clips free-decoded; clips.json unchanged;
  encoder files and diagnostics match their SHA-256; FP64-feature reference current; build clean;
- finite: encoder output and every captured decoder output, in free decoding and replay (Swift count = 0 and
  Python recount = 0);
- encoder: rel ≤ 0.1 (tau 1e-3) vs the FP64-feature reference on every clip; length = clips.json;
- decisions per head (token incl. blank; duration), pooled: decisive steps (FP32-feature reference raw-logit
  top-1 margin ≥ 1.0) ≥ 50%; agreement ≥ 99.5% on decisive and ≥ 99% on all steps;
- free decoding: ≥ 61/64 natural clips identical to the FP32-feature reference tokens; WER ≤ reference + 0.2 points;
- decoder precision fp32; the arm's WP3 record still revision 8 and timing-allowed.
Reported, not gated: head errors (token logits, duration logits for F2 and F0; h and c for all three), token-
probability difference vs the reference softmax, margin distributions (reference, pipeline's own where logits
exist, reference margins of disagreeing steps), encoder errors vs the FP32-feature reference (diagnostic).

**Record identity** (`components`, recomputed and compared by `parakeet-bench run` before timing): compute units;
model configuration (allowLowPrecisionAccumulationOnGPU) and label-loop constants (blank, durations, max symbols);
executable SHA-256; encoder package (per-file SHA-256 tree); front-end blob SHA-256 and manifest SHA-256; F2 blob
SHA-256 and manifest SHA-256, or F0/F1 Core ML model trees + decoder manifest SHA-256 and precision.

**Resource caps.** RSS 4G (ANE combinations), 6G (C1, CPU and GPU backends); timeout 5400 s; macguard's system-
memory (abort below 25% free) and swap (1 GB growth) aborts; lock refusals retried every 60 s up to 120 times; one
job at a time. Expected 1–9 min per combination (about 25 min for C6s8 on the GPU), about 5 h in total.

## 4. Publish the records and transfer them to the Mac

```
CUDA_VISIBLE_DEVICES= ./python ios/pipegate.py table
# copy the Mac's refcache/mp2-ref64/index.json to ios/results/pipegates/ref64_index.json
git add ios/results/eligibility/pipelines/*.json ios/results/eligibility/pipelines/table.txt ios/results/pipegates/
git commit; git push                                     # text only, checked with git diff --cached --stat
# Mac: git pull --ff-only (NO rebuild: the records bind the gates' executable)
CUDA_VISIBLE_DEVICES= ./python ios/wp5sweep.py verify --groups c
```
`verify` checks that the Mac checkout is the new commit and clean, that the Mac binary's SHA-256 equals the
`executable_sha256` in every sweep arm's records, and runs `parakeet-bench run --verify-only` with each arm's exact
sweep arguments (pipeline record, component hashes, WP3 record, C0 identity against c0.json); nothing is loaded or
timed.

## 5. Informational paired sweep of the new arms (WP7 item 4)

**Arms** (all front end A + F2, multifunction, mp2), in this order:
1. C4-ane, 2. C3-ane, 3. C6s8-ane (ANE layout, cpuAndNeuralEngine);
4. C4, 5. C3, 6. C6s8 on cpuAndGPU;
7. C6s8 plain on cpuAndNeuralEngine (within-session anchor; compare with WP5's C6s8 row).

**Design (finding 9).** The 64 natural clips are split into halves A and B: alternate clips of each bucket in
clips.json order (8 + 8 per bucket). Half A runs the arms in the order above, half B in reverse (7 → 1): 14 runs,
each arm once early and once late, every arm's 64 clips from both ends of the session.

**Protocol per run** (WP5's): mode free, 3 warm-up + 10 timed calls per clip and arm; C0 (pinned export, verified
against c0.json, on cpuAndNeuralEngine / preprocessor cpuOnly) and the arm interleaved clip by clip in one process,
C0 first on even clip indices. Before every block: 500 ms settle, then wait (≤ 120 s) while
ProcessInfo.thermalState is serious or critical; a block record holds the thermal state before/after and waits.
One macguard job per run (`sweep_job.sh`): fail-closed disk check, Core ML cache purge, run 1 = the paired sweep
("post-purge loads"), run 2 = a fresh process loading the arm alone with one call on `n15-2412-153947-0005`
("subsequent fresh-process load", arm-only phys_footprint), purge.

**Command.** `CUDA_VISIBLE_DEVICES= ./python ios/wp5sweep.py sweep --name wp7 --groups c --settle-ms 500`. It
refuses unless every arm has a revision-9 timing-allowed pipeline record whose executable is the Mac's binary and
the Mac checkout is this commit and clean. Manifest: `/mnt/hd/wilderness-labs-stt/parakeet-ios/results/wp5/sweeps/
wp7-<time>.json` (build, halves, order, every run's id/status/position), rewritten atomically after each run. Runs
in never-reused `<arm>/<run id>/` directories on both machines, retrieved completely into `.part` and renamed;
failed runs are not published. Exit 0 only if all 14 runs have status 0.

**Report.** `CUDA_VISIBLE_DEVICES= ./python ios/wp5sweep.py report --sweep wp7-<time> --tag wp7` reads only the
runs the manifest names and refuses unless the manifest is complete, every run is status 0 with all files, every
load record carries the manifest's executable, each run's clips equal its half, and the halves cover the 64 natural
clips exactly once. Per arm, armreport merges the two halves (identical arm, units, eligibility, protocol,
executable, C0 identity required; disjoint clips) and checks each half's pairing (session id, clips.json SHA-256,
clip ids, warm-ups/timed, mode) and C0 identity.

**Outputs.** `ios/results/wp7/<arm>.summary.json`, `<arm>.c0block.summary.json`, `sweep.json` (+ git-ignored
`sweep_table.md`, reproduced in the README): per arm and bucket the arm/C0 typical total-latency ratio (median over
clips of per-clip median ratios) with a 95% percentile bootstrap CI (2,000 clip resamples, pairs kept together,
seed 0), HD p95 ratio, encoder / preprocess / decode stage medians, post-purge and subsequent fresh-process load
times per half, arm-only footprint peak, WER and token identity to mp2's reference, physical and logical call
counts, thermal-state counts per block, the runs' positions in the session. No pass/fail: **informational, shared
Mac, no claims.**

**Resource caps.** RSS 4G (ANE arms), 6G (GPU arms); guard timeout 3600 s (ANE), 7200 s (GPU); same macguard aborts
and lock retries; disk as in section 3. Expected about 2–5 min per ANE/GPU run and 15–25 min per C6s8 GPU run
(loads), about 2 h in total.

**Remaining limitations (stated beside every comparison in the README).**
- The Mac is shared with the user's other work: background load varies within and between runs; pairing with C0
  per clip in one process and the counterbalanced order reduce but do not remove this.
- Order is counterbalanced (A forward, B reverse), not randomized; a drift that is not monotone over the session
  is not cancelled. One session, so no between-session variance.
- ProcessInfo.thermalState is coarse (nominal / fair / serious / critical); throttling below "serious" is not seen.
- Loads are post-purge and subsequent fresh-process loads; whether Core ML specialized or reused a cache is not
  established (no Instruments cache events). Placement is not traced (compute plans only, WP6a).
- GPU-backend compute plans place almost all of the compressed encoders on the CPU (WP6a); "GPU" names the compute
  units requested, not where the work ran.
- The anchor shows within-session agreement with WP5's C6s8 numbers only for that one arm.
