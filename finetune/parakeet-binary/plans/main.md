# Plan: binary-weight Parakeet main run (MB1)

**Status: 2026-10-17: implemented, smoke-tested (two SIGKILLs and exact resumes), 93 unit
tests pass. Codex review (gpt-6-astra xhigh, `/mnt/hd/wilderness-labs-stt/parakeet-binary/review/main/`):
round 1 NOT CLEAN (1 major, 3 minor, all fixed), round 2 CLEAN with no new findings. MB1
launched 2026-10-17 in unit `parakeet-binary-binary-main` (log `logs/binary-main.log`). Finished 2026-10-20 05:23 (68.6 h, 0 resumes): full-dev mean 6.312% (M1 +0.226, pilot B2 -0.710, B0 +0.737); export 105.3 MB; `runs/main-report.json`. Test scoring: plans/test-scoring.md.**

## Results (2026-10-20)

Finished 2026-10-20 05:23 in one session (68.6 h wall, 0 resumes, 33,246 audio
hours seen). Report: `/mnt/hd/wilderness-labs-stt/parakeet-binary/runs/main-report.json`.

Full-dev WER (%), final checkpoint (step 250,000), scored on the rebuilt export:

| Model | LS clean | LS other | AMI | VoxPopuli | YODAS | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 (FP32 original) | 1.59 | 2.94 | 12.12 | 5.75 | 5.47 | 5.58 |
| M1 (ternary) | 1.99 | 4.20 | 11.71 | 6.02 | 6.51 | 6.09 |
| **MB1 (binary)** | 2.16 | 4.69 | 11.95 | 6.06 | 6.70 | **6.31** |
| pilot B2 (binary, 12k steps) | 2.42 | 5.51 | 12.90 | 6.53 | 7.76 | 7.02 |

- **MB1 - M1 = +0.23** (net change: binary conversion plus 250k more steps on a
  new data order; not a causal binary cost). Largest on LS other (+0.49).
- **MB1 - pilot B2 = -0.71**: the 20x longer run recovered three quarters of
  the pilot's gap to M1; every set improved, most on YODAS (-1.06) and AMI (-0.95).
- **MB1 - B0 = +0.74**; on AMI MB1 is 0.17 better than B0 (M1 is 0.41 better).
- Size: export 105.3 MB (75.5 MB packed codes) vs M1 180.8 MB and 2,472 MB .nemo.
- 8.0% of latent signs changed from M1 (attention 11.3%, pointwise 7.9%, FF 7.1%);
  codes balanced 50.0 / 50.0.
- Stream waits totalled 7.2 h (10.5% of wall), concentrated in occasional stalls.

Dev-subset curve against M1 at the same step (same subset, same schedule shape):

| Step | 62.5k | 100k | 150k | 187.5k | 200k | 225k | 237.5k | 250k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | 8.00 | 7.67 | 7.17 | 6.85 | 6.69 | 6.57 | 6.47 | 6.31 |
| MB1 | 7.38 | 7.35 | 7.05 | 6.85 | 6.75 | 6.64 | 6.41 | 6.32 |

MB1 is lower through the ramp (it starts from a converged quantized model; M1's
ramp started from FP32), then tracks M1 within about 0.1 points through the decay.

**Test (2026-10-20, scored once, results/TEST.md):** seven-set mean MB1 7.05%,
M1 6.84%, B0 6.45% (MB1 − M1 +0.21; per set +0.07 to +0.45; Common Voice apart
+0.86). MB1 is below B0 on AMI (10.65 vs 11.15).

## Decision and inputs

The pilot (`plans/pilot.md`, finished 2026-10-17 00:56, `runs/pilot-selection.json`)
selected start M1 at learning rate 2e-4 (arm B2, full dev 7.02%; A 8.46%, B5 7.16%,
C 6.19%, M1 6.09%) and proposed a binary main run (B2 within 1.5 points of the
ternary pilot's 6.85%). On 2026-10-17 the user approved the main run and chose
**250,000 steps** (M1's step count) over 125,000 and 60,000.

## The run (everything fixed before launch)

| Item | Value |
| --- | --- |
| Run name | `main-binB-m1-lr2e-4` (binary RUNS) |
| Init | M1 student state, `ckpt/latest.pt`, SHA-256 `003b1289...d62d212a` (pinned, as in the pilot) |
| Quantizer, ramp | binary; ternary to binary (`--init m1 --quantizer binary`, ramp kind "ternary") |
| Recipe | train.py `--arm P2` (teacher transcripts plus encoder-output matching, prediction and joint frozen), unchanged |
| Peak learning rate | 2e-4 (pilot winner B2) |
| Steps | 250,000 |
| Schedule | warmup 2% (5,000 steps), weight ramp 25% (62,500 steps, f = 1 from step 62,500), linear decay to zero at 250,000; train.py's fractions, as in M1 and the pilot |
| Batches | 600 s duration-bucketed, gradient checkpointing, BF16 autocast with FP32 master weights |
| Stream | Hub stream `--train main`, 8 workers, **seed 20261017** (not M1's 20260930; see "Data order"); its configuration hash is computed offline before launch, pinned in `binary_paths`, and any other hash is refused |
| Dev-subset evaluation | every 12,500 steps (20 points), as M1 |
| Selection | final checkpoint only (step 250,000, f = 1); no early-stopping or best-checkpoint selection |
| Scoring | full dev (five sets) on the model rebuilt from its 1-bit export; exact-reconstruction check must pass |
| Checkpoints | every 30 minutes with exact resume; rerunning the same command resumes |

**Data order: a new seed (user decision 2026-10-17).** With M1's seed, worker
count and configuration the run would replay M1's exact item sequence, starting
from weights already fitted to it: a second pass over the same 33,248 exposure
hours in the same order, which invites memorization. Seed 20261017 instead:

- The mixture, filters, file lists, worker count and source shares are
  unchanged; only the per-worker file permutations, slot and mixing draws, batch
  shuffling, SpecAugment masks and other `--seed`-driven randomness change.
- YODAS (about 100,000 hours available, 22,040 seen by M1) is drawn in a new file
  order, so most YODAS audio MB1 hears is new to M1 (expected file overlap about
  a fifth, at random). People's Speech likewise. LibriSpeech, AMI and VoxPopuli
  are small and repeat regardless, now in a different order.
- Held-out data is unaffected: the dev sets and the reserved YODAS recordings
  excluded from training are fixed by stored files built from the project seed
  (`data.reserved_recordings()`, dev manifests), not by the training seed. A
  contract test asserts the exclusion list is identical under both seeds.
- Side effect to handle: `stream.experiment_config` rewrites the shared
  `parakeet-ternary` `DATA/stream/sources.json` (its `seed` field) on every
  open. MB1 must not leave M1's provenance record changed: the implementer
  records how (for example, restore or redirect the record in the binary seam)
  before writing the code.
- Cost: the stream at the new seed is untested on the GPU. The smoke run
  (procedure step 2) uses the new seed and checks delivery rate.
- Comparison to pilot B2 (project seed) now mixes in a seed effect, which is
  small next to a 20x longer run.

**Gate.** The G-B gate (M1 at f = 0 within 0.02 points of 6.086%) passed in the
pilot and is reused only if its sources still match (the existing
`require_gate`); otherwise it is rescored before training.

## Cost and resources

About 250,000 x 0.84 s = 58 GPU-hours of steps, plus 20 dev-subset evaluations,
checkpoint writes and the final export and full-dev scoring: about 66-70 hours
(M1 took 69.3 hours). One unit, memory-capped:

```sh
cd finetune/parakeet-binary
./heavy binary-main --mem-max 36G --runtime 100h -- /mnt/hd/wilderness-labs-stt/parakeet-binary/launch-main.sh
```

`launch-main.sh` refuses to start unless the GPU power limit is at most 400 W
(as `launch-pilot.sh`), then runs `./python binary_sweep.py --phase main`.
A crash, power loss or runtime stop resumes by rerunning the same command.
Disk: 5.2 TB free; one checkpoint is 6.9 GB.

**Pausing for other GPU work.** Noisy-dev scoring (`finetune/noisy-dev`) needs the
GPU. If the user wants it before MB1 ends, the unit is stopped (SIGTERM; the next
run resumes from the last checkpoint, losing at most 30 minutes) and restarted
afterwards with the same command. The GPU lock prevents overlap.

## Reported

- Full-dev mean and per set, against M1 (6.086%), B0 (5.575%) and pilot B2 (7.021%).
  - MB1 - M1 is the **net change** from converting M1 to binary and training
    it for another 250,000 steps (restarted schedule, mostly new YODAS audio).
    It is not a causal cost of binarization: the continuation and the new data
    could hide part of a binary penalty, or add a gain of their own. Pilot
    control C bounds the continuation effect only at pilot scale (12,000 steps,
    C - M1 +0.10 points, a tie). A matched 250,000-step ternary continuation
    from M1 would isolate it (another about 70 GPU-hours); not run, proposed
    only if the net change needs a causal reading (review round 1, finding 2).
  - MB1 - B2 is what the 20x longer run bought, including a seed effect.
- Dev-subset curve (20 points), code balance (overall and per family), fraction
  of latent signs changed from M1, export size, wall time, stream waits, resumes.
- No threshold decides anything here: the numbers are reported as they are.

**Test sets** are scored once, after MB1 finishes, under a separate plan
(`plans/test-scoring.md`, modeled on `parakeet-ternary/plans/test-scoring.md`)
that is reviewed before it runs. Not part of this plan.

## Code changes (smallest seam; parakeet-ternary and whisper-ternary stay byte-identical)

1. `binary_sweep.py --phase main`: requires `pilot-selection.json` to exist, have
   proposal "binary main run", start "m1" and winner "B2", and to have been
   produced from arm runs that still validate under current sources; refuses
   otherwise. Then the gate (`require_gate`), then one arm `MB1` = `--init m1
   --quantizer binary --arm P2 --lr 2e-4 --train main --max-steps 250000
   --seed 20261017 --run-name main-binB-m1-lr2e-4` with eval every 12,500 steps, validated like a
   pilot arm (`validate_arm`, train "main", seed 20261017, pinned new-seed hash), and
   finally writes `RUNS/main-report.json` with the reported numbers above.
   `--dry-run` prints the command without touching the GPU.
2. `binary_main.py` (the MB1 entry point; `binary_train.py` is frozen, see "Implementation
   notes (2026-10-17)", item 1): accept `--train main --seed 20261017` (8 workers, the pinned
   new-seed main hash; refuse an unpinned seed or hash), keeping every pilot check. The
   resume key already includes `train` and `max_steps`.
3. `/mnt/hd/wilderness-labs-stt/parakeet-binary/launch-main.sh` (outside the repo,
   like `launch-pilot.sh`).
4. Pilot code paths and the finished pilot's results must keep validating
   (rerunning `--phase pilot` must reuse all four arms and rewrite an identical
   decision).

## Contract tests (before the implementation)

1. `--phase main --dry-run` builds exactly the command in item 1 (init, quantizer,
   recipe, lr, train main, steps 250,000, run name, eval interval 12,500).
2. `--phase main` refuses when `pilot-selection.json` is missing, proposes a mixed
   pilot, names a different winner or start, or is stale (an arm no longer
   validates).
3. `--phase main` refuses to train if the gate did not pass.
4. `binary_main.py --train main --seed 20261017` opens 8 workers and refuses a
   stream whose configuration hash is not the pinned one, and refuses any
   unpinned seed; `--train pilot` behavior is unchanged.
5. The reserved-recording exclusion list and the dev manifests are identical
   under seeds 20260930 and 20261017; opening the MB1 stream leaves the ternary
   `DATA/stream/sources.json` byte-identical (or as recorded in the seam).
6. A finished MB1 run with a wrong step count, learning rate, init, quantizer,
   seed, stream phase or source hash fails validation and is not reused.
7. Resume: a run directory from a pilot arm, another seed or a different
   `max_steps` is never resumed as MB1 (exit 78).
8. Existing 62 tests still pass; provenance test still passes.

## Implementation notes (2026-10-17)

Recorded before the code (procedure step 1); not yet reviewed.

1. **No binary pilot source changes, so the seam moves to two new files.** Every
   finished pilot arm's `summary.json`, `config.json` and `resumes.jsonl` record the
   SHA-256 of the six `BINARY_SOURCES` (`binary_train.py`, `binary_quant.py`,
   `binary_export.py`, `binary_paths.py`, `python`, `heavy`), and the G-B gate's
   summary records them too. The sweep reuses a run or point only if those hashes
   match the files as they are now. Editing `binary_train.py` or `binary_paths.py`
   (code changes, items 1-2) would therefore make all four arms and the gate
   unreusable, which breaks item 4. Those six files stay byte-identical, like
   `parakeet-ternary/`. A contract test checks them against the pilot arms' records.
   Instead:
   - `binary_main_paths.py` (no torch, so the sweep can import it) holds the MB1
     constants and the stream pins in place of `binary_paths`. The pins are keyed by
     (stream phase, seed) and give the worker count and configuration hash:
     (pilot, 20260930) = 6 workers, `874bf7e4...` (taken from `binary_paths`, unchanged);
     (main, 20261017) = 8 workers,
     `bad909114fbe3e50505434635c49406f751f0673663e275b460ebdb82612badf`. M1's own
     (main, 20260930) order is deliberately not pinned, so it is refused.
   - `binary_main.py` is the MB1 training entry point. It takes the same arguments as
     `binary_train.py` and runs `binary_train.main`'s body unchanged inside
     `binary_train.seam()`, after it has replaced three `binary_train` attributes. They
     are restored on exit:
     `parse_args` refuses an unpinned (train, seed) pair before any model loads.
     `open_stream` refuses an unpinned pair and opens the Hub stream with the pinned
     worker count. It refuses a configuration hash other than the pin for every seed.
     (`binary_train.open_stream` checks the hash only at the project seed.) It also
     redirects the stream record (item 3). `BINARY_SOURCES` gains `binary_main.py`
     and `binary_main_paths.py`, so MB1's config, resumes and summary record them.
     The sweep checks them for MB1 only, and the pilot arms keep their own source set.
   - `binary_train.py` itself is unchanged. Run directly, it behaves exactly as in
     the pilot, including for `--train pilot`. MB1 and its smoke run go through
     `binary_main.py`. This deviates, for the reason above, from the draft's code-change
     item 2, which named `binary_train.py`. Item 2 was amended in review round 1.
2. **Hash computation (offline, no GPU).** `stream.experiment_config(seed)` with
   `stream.record_sources` replaced by a no-op, then `StreamConfig.digest(workers)` and
   the `TrainingStream(cfg, num_workers=workers).config_hash` of an unstarted stream.
   This reproduces the recorded pilot (6 workers, `874bf7e4...`) and M1 (8 workers,
   `519aeeb5...`) hashes exactly. With seed 20261017 and 8 workers it gives
   `bad90911...`. The exclusion list (`exclude_groups`, 1 hash for all three) is the
   same under both seeds.
3. **`sources.json`: redirect, not restore.** `stream.experiment_config` calls
   `stream.record_sources`, which writes `paths.DATA/stream/sources.json` (that is,
   `/mnt/hd/wilderness-labs-stt/parakeet-ternary/data/stream/sources.json`, SHA-256
   `d33d395a...cbdf8` on 2026-10-17, seed 20260930) and `yodas_train_files.json`
   (`2d0238e7...d8a7`). Both files are backed up with their hashes in
   `/mnt/hd/wilderness-labs-stt/parakeet-binary/backup-2026-10-17/`. Restoring the
   files after the stream opens would leave the ternary record changed whenever the
   process dies in between. So `binary_main.open_stream` replaces `stream.record_sources`
   while the stream is configured. The replacement calls the original with `paths.DATA`
   pointed at `ARTIFACTS/stream-records/seed-<seed>/`, so the same record (same
   code, same content) is written under the binary artifacts, at
   `.../stream-records/seed-20261017/stream/sources.json`. Nothing else reads
   `paths.DATA` during that call. Both are restored afterwards. As a guard, the
   two ternary files are hashed before and after opening, and a difference raises
   before training. The pilot (`binary_train.py`, unchanged) still rewrites the
   ternary record with the project seed, which gives the same bytes as now.
4. **Sweep (`binary_sweep.py`, orchestration, not a training source).** `--phase main`:
   references; the selection check; the gate (`require_gate`); MB1; `main-report.json`.
   - Selection check: `RUNS/pilot-selection.json` must exist, with proposal "binary
     main run", start "m1", winner and recommended start "B2". Each of the four pilot
     arms must still validate under current sources (`validate_arm`, the pilot
     protocol). `decide()` recomputed from those summaries and the current references
     must equal the file's `decision` exactly; otherwise the file is stale. Any
     failure refuses with exit status 1 and gives the reason. This also runs on
     `--dry-run`, which only reads.
   - MB1 command: `./python binary_main.py --init m1 --quantizer binary --arm P2 --lr
     2e-4 --run-name main-binB-m1-lr2e-4 --train main --max-steps 250000 --seed 20261017
     --batch-seconds 600 --eval-every 12500 --select final --stream hub
     --grad-checkpointing`.
   - Validation: `validate_arm` with train "main", 250,000 steps, eval every 12,500,
     seed 20261017, the pinned MB1 hash and the pilot's teacher and quantized module
     hashes as reference, and sources `sweep.TRAINING_SOURCES` + `BINARY_SOURCES` +
     the two new files. `sweep.validate_summary` compares the summary's seed with
     `paths.SEED`. For MB1 the binary sweep therefore checks the seed itself (summary
     and stream record), then runs `validate_summary` on a copy with the seed
     replaced. `sweep.validate_config` uses the expected arguments, which carry
     20261017.
   - `RUNS/main-report.json`: MB1's `arm_report` (per-set and mean full dev,
     dev-subset curve, code balance, sign changes from M1, export size, wall and
     training time, stream waits, resumes). Also MB1 - M1, MB1 - B0 and MB1 - pilot
     B2 (mean and per set), the references with their hashes, the gate, the
     selection file's hash, the protocol and the stream pin.
5. **Launch script** `/mnt/hd/wilderness-labs-stt/parakeet-binary/launch-main.sh`: a copy
   of `launch-pilot.sh` (same 400 W check) ending in `exec ./python binary_sweep.py
   --phase main`.
6. **Tests.** `tests/test_binary_main.py` had 23 tests at first: contract tests 1-8,
   plus a check of the `main-report.json` differences. That made the suite 85 tests,
   all on the CPU. Review round 1 added 8 more, so the suite is now 93. Contract 5
   opens the real MB1 and pilot streams (not iterated, so no workers start) with the record
   redirected to a test directory; should the redirect ever fail, its cleanup writes the
   ternary record's original bytes back. Contract 8 checks the six pilot binary sources
   against the four pilot arms' and G-B's recorded hashes.
7. **Commands** (from `finetune/parakeet-binary/`):
   - Tests: `./heavy binary-tests --mem-max 24G --runtime 30min --wait -- python -m unittest discover -s tests -v`
   - Smoke (step 2): `./heavy smoke-main --mem-max 36G --runtime 45min -- python binary_main.py
     --init m1 --quantizer binary --arm P2 --lr 2e-4 --train main --seed 20261017 --smoke
     --max-steps 200 --eval-every 100 --ckpt-minutes 1 --stream hub --grad-checkpointing
     --run-name smoke-main-binB`. These are the pilot smoke `smoke-binB`'s limits (200 steps,
     dev-subset evaluation every 100, 16 utterances per dev set, 1-minute checkpoints).
     `binary_main.py` replaces `binary_train.py` (item 1).
   - Dry run: `./python binary_sweep.py --phase main --dry-run --allow-unconfined`
   - Launch (step 4): `./heavy binary-main --mem-max 36G --runtime 100h -- /mnt/hd/wilderness-labs-stt/parakeet-binary/launch-main.sh`
   - Hash computation (item 2): `./python /mnt/hd/wilderness-labs-stt/parakeet-binary/tmp/hashcalc/calc.py`
8. **Never train an MB1-style run (`--train main`, or any seed other than the project's)
   with `binary_train.py`.** That file is frozen, so it would train without the hash check
   and rewrite the ternary `sources.json`. Use `binary_main.py`. The sweep only launches
   `binary_main.py`, and its validation refuses a run that lacks `binary_main.py` and
   `binary_main_paths.py` in its recorded sources or has another stream hash.

## Smoke runs (2026-10-17, step 2)

Plumbing only. 16 utterances per dev set, so none of these numbers is a result. Runs
are under `/mnt/hd/wilderness-labs-stt/parakeet-binary/runs/smoke-main-binB*`, logs in
`logs/smoke-main*.log`. GPU power limit 400 W.

- **Kill and resume (`smoke-main-binB`, command in note 7).** Session 1 wrote its first
  checkpoint at step 1, because the first step, including stream start-up, took longer
  than the 1-minute interval. It was then killed (`systemctl --user kill
  --signal=SIGKILL`, the whole unit, stream workers included). The same command resumed
  at step 1, wrote a checkpoint at step 2, and was killed the same way. A third run
  resumed at step 2 and finished: 200 steps, export, full dev. `resumes.jsonl` has both
  resumes, each with the MB1 stream hash in the restored stream state. Training wall
  time 779 s over 3 sessions.
- **Resume is exact.** An uninterrupted run with the same flags (`smoke-main-binB-ref`,
  489 s, 1 session) matches the killed run in every `metrics.jsonl` record (loss, TDT,
  encoder matching, gradient norm, learning rate, f, dev-subset WER per set). Full-dev
  WER, audio seconds per source, the latent-sign-change fraction and the export's SHA-256
  (`1a57184c...`) also match.
- **Stream.** Config records phase main, seed 20261017, 8 workers and hash
  `bad90911...`, as pinned. Steady state is 0.82-0.90 s/step with no stream wait
  (at most 1.4e-5 s/step) from step 21 on in the killed run and from step 41 in the
  reference, as in the pilot smoke runs (0.83-0.87 s/step). All waiting is at start-up.
  Killed run: 456 s in steps 1-20, made up of about 2.3 min to the first step in
  session 1, 4.5 min in session 2 and 1 min in session 3. Reference: 122 s in steps 1-40.
  The pilot's fresh `smoke2-binB` waited 51 s. Start-up at the new seed is slower by a
  minute or more and varies; at 250,000 steps that is negligible.
- **Numbers.** Loss 1.266 at step 100 and 1.112 at step 200. Dev-subset WER 15.31% and
  12.91%; full dev (limit 16) 12.04%. Reconstruction exact (codes and scales), and greedy
  hypotheses equal between the QAT model and the rebuilt export. Export 105.3 MB (75.5
  MB packed codes). Code balance 49.99% +1; latent signs changed 0.29% (attention 0.59%,
  feed-forward and pointwise 0.21%). Every decoder ran in mode `no_while_loops`. The
  summary's source hashes include `binary_main.py` and `binary_main_paths.py`.
- **Ternary record.** `parakeet-ternary/data/stream/sources.json` (`d33d395a...`) and
  `yodas_train_files.json` (`2d0238e7...`) were byte-identical after every session, the
  test runs and the dry runs. The MB1 record is at
  `stream-records/seed-20261017/stream/`. It differs from the ternary record only in
  `seed` (checked by contract test 5).
- **Dry runs.** `./python binary_sweep.py --phase main --dry-run --allow-unconfined`
  reported that the pilot selection matches its arms (proposal binary main run, start m1,
  winner B2), reused G-B (-0.002 points vs M1, PASS), and printed exactly the MB1 command in
  note 4. `--phase pilot --dry-run` still reuses G-B, Z-A, Z-B and all four arms. The
  pilot decision recomputed from them equals `pilot-selection.json` (checked by
  `require_selection`).
- The two smoke runs keep their 7.3 GB `ckpt/latest.pt`, as the pilot smoke runs do.

## Review round 1 (2026-10-17)

Codex (`gpt-6-astra`, xhigh): NOT CLEAN, 1 major and 3 minor
(`/mnt/hd/wilderness-labs-stt/parakeet-binary/review/main/r1.md`). Resolutions:

1. **major, source changes were not refused on every resume.** Fixed in `binary_main.py`.
   The seam now also replaces `train.load_resume_payload`. It runs train.py's own checks
   first: an unreadable checkpoint exits 77 and different arguments exit 78, so contract
   7 keeps exit 78. Then, before any model is built or any checkpoint restored, it
   compares `train.training_source_hashes()` under the seam with `config.json`'s
   `training_source_hashes`, the record written at the run's first start. That covers
   every hashed source, `binary_main.py` and `binary_main_paths.py` included. Any
   difference, or a checkpoint without that record, exits `EXIT_SOURCES_CHANGED` = 79
   and names the changed files. `sweep.run_train` retries only 75, so a source edited
   during a stream outage stops the sweep (`train.py exited 79`) instead of resuming.
   Nothing is deleted. Tests: `SourceProvenanceResumeTest` in
   `tests/test_binary_main.py`. They run `binary_main.main` in-process with CUDA
   reported available and a stub `build_context`, and a changed file is simulated
   through `train.sha256`. They cover:
   - unchanged sources resume;
   - a direct resume after `binary_main.py`, `binary_main_paths.py`, `binary_train.py`
     or `stream.py` changed is refused (79) without restoring, and the checkpoint is
     kept;
   - a checkpoint without `config.json` is refused;
   - through `sweep.run_train`, a first session ends with 75, `binary_main.py` changes,
     and the retry is refused with 79 and not retried (2 launches);
   - exit code 79 differs from 0 and 75-78.

   The contract 7 fixture now writes the `config.json` source record a started run has.
   Its expectations are unchanged.
2. **minor, MB1 - M1 is not a causal binary cost.** The coordinator rewrote the plan's
   "Reported" section. In code, `main_report`'s `readings["MB1_minus_M1"]` is now "net
   change after binary conversion and 250,000 further steps from M1 (not a causal binary
   cost; see plans/main.md 'Reported')". The test is `MainReportTest`, which asserts the
   string.
3. **minor, procedure step 2 named the unsafe entry point.** Fixed. Step 2 now gives the
   exact `binary_main.py` command that was run (note 7). Code-change item 2 and contract
   test 4 now name `binary_main.py`. Note 8 says never to train an MB1-style run with
   `binary_train.py`. The remaining mentions of `binary_train.py` describe the frozen
   pilot file.
4. **minor, the launcher accepted unknown power readings.** Fixed in `launch-main.sh`.
   nvidia-smi is `$NVIDIA_SMI`, defaulting to the same store path, so behaviour on the
   real system is unchanged. A failed query, or a first line that after removing
   whitespace is not a positive decimal number (empty, `N/A`, `[N/A]`, `abc`, `0`,
   `-5`, `400W`, `1e3`), exits 3 with a `launch-main:` message before the 400 W check.
   Test: `LaunchScriptTest`. It uses a stub nvidia-smi through `NVIDIA_SMI` and a stub
   `./python` in the working directory:
   - 400, 400.00, 350.5 and " 400.00 " launch `binary_sweep.py --phase main`;
   - the malformed values above, 0.00, 450.00, 400.01, no output and a failing query are
     refused;
   - the default path is the store path.

   Against the real GPU (read only, with a stub `./python`), the launcher read "400.00
   W" and started the stub.

After the fixes: 93 unit tests pass. `--phase pilot --dry-run` still reuses G-B, Z-A,
Z-B and the four arms. `--phase main --dry-run` accepts the selection and the gate and
prints the MB1 command. The ternary `sources.json` and `yodas_train_files.json` are
unchanged (`d33d395a...`, `2d0238e7...`). No file in `parakeet-ternary/` or
`whisper-ternary/` changed. The smoke runs predate the resume check. It does not
change training, so they were not repeated.

## Procedure

1. Implement against the contract tests (Opus subagent).
2. Smoke, from `finetune/parakeet-binary/` (implementation note 7): `./heavy smoke-main
   --mem-max 36G --runtime 45min -- python binary_main.py --init m1 --quantizer binary --arm
   P2 --lr 2e-4 --train main --seed 20261017 --smoke --max-steps 200 --eval-every 100
   --ckpt-minutes 1 --stream hub --grad-checkpointing --run-name smoke-main-binB`, including
   a SIGKILL after the first checkpoint and a resume (the same command again); then
   `./python binary_sweep.py --phase main --dry-run --allow-unconfined`.
3. Codex review (gpt-6-astra, reasoning effort xhigh, read-only, explicit file
   scope) of this plan and the code; fix or record every finding; repeat until no
   blocker or major remains.
4. Launch with the command above. Write the results here and in the README.

## Out of scope

Augmentation and noisy-dev training (the noisy-dev set exists now but has not
been scored; it is a separate step), a fresh-seed or larger-data run, the mixed
FF-binary variant, activation quantization, packed kernels, speed or energy
claims, test-set scoring (separate plan), publishing.
