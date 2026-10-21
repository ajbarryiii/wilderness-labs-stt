# Plan: final test-set scoring (Phase 3)

**Status: completed 2026-10-04.** Precheck passed; M1 and B1 scored once each;
results in `../results/TEST.md` and DESIGN.md "Main-run and test results". The
AMI empty-hypothesis sanity flag was investigated and closed (DESIGN.md).

Written 2026-10-04, before M1 finished. Implements DESIGN.md "Evaluation" and
"Phase 3". No code changes are planned; the scoring code is the frozen
`evaluate.py` (hash recorded in DESIGN.md item 10 and in M1's summary.json).

## What is scored

| Arm | Model | Test sets | Status |
| --- | --- | --- | --- |
| B0 | FP32 original, pinned revision | all 8 | done 2026-09-30 (`eval/b0-pretrained`), reused, not rerun |
| B1 | ternary post-training quantization, no training | all 8 | LS clean/other and AMI done on export `runs/b1-ptq/export` (SHA-256 b241b984...); score the other 5 on that same existing export (no re-quantization) |
| M1 | main run, final checkpoint (step 250,000, f = 1), rebuilt from its export | all 8 | to do |

Pilot arms are not scored on test: they were used only for selection on dev.
The optional FP32 control A1 is not run (DESIGN.md: only if 12 GPU-hours
remain after M1; decided separately).

## Preconditions (checked by `precheck_scoring.py`, which must print ALL PASS)

`./python precheck_scoring.py` (read-only, no GPU) writes `eval/M1-precheck.json`
and verifies:

1. Unit success from durable evidence: the user journal of
   `parakeet-main.service` shows it deactivated successfully and no "Failed
   with result" (units are collected after exit, so `systemctl show` is not
   used), and the sweep log's main phase ends with `done`. No `parakeet-*`
   unit is active.
2. Final-M1 identity: summary.json has run_name `main-M1-P2-lr5e-4`, arm M1,
   recipe P2, lr 5e-4, select final, selected_step = steps = max_steps =
   250,000, quantized, scored_on "model rebuilt from export"; and the export
   manifest's `extra` run_name, arm, recipe, lr, select, selected_step and
   config_sha256 equal the summary's. A stale or pilot export cannot pass.
3. Integrity: SHA-256 of export.safetensors equals the manifest; the export's
   reconstruction.json has codes and scales exact and equals the summary's
   reconstruction record.
4. Code: every source hash recorded in summary.json equals the current file,
   and evaluate.py equals the frozen snapshot.

The M1 test outputs must report the same export SHA-256 that the precheck
verified; `report_test.py` refuses otherwise.

## Commands (from `finetune/parakeet-ternary/`, each in its own capped unit, sequentially)

```sh
./heavy test-m1 --mem-max 24G --runtime 2h --wait -- \
  python evaluate.py --source export --path /mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/export \
  --sets test --out-dir /mnt/hd/wilderness-labs-stt/parakeet-ternary/eval/M1-test
./heavy test-b1 --mem-max 24G --runtime 2h --wait -- \
  python evaluate.py --source export --path /mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/b1-ptq/export \
  --sets earnings22 gigaspeech spgispeech voxpopuli common_voice \
  --out-dir /mnt/hd/wilderness-labs-stt/parakeet-ternary/eval/b1-ptq-rest
./python report_test.py   # builds results/TEST.md and results/test.json
```

Order: precheck, then M1, then B1, then the report. `heavy --wait` returns the
unit's exit status, which is recorded; a non-zero status means that arm's
output directory is discarded (renamed `*.failed-<time>`) and rerun from
scratch. Expected time: about 15-20 minutes each (B0 decoded all 192 test
hours in 15.4 minutes). Outputs are new directories; nothing existing is
overwritten.

**B1 aggregation.** B1's eight sets come from two complete runs on the same
export: `eval/b1-ptq` (LS clean, LS other, AMI; scored 2026-09-30) and
`eval/b1-ptq-rest` (the other five). `report_test.py` merges them only if
both outputs record the same export SHA-256 and it equals the current
`runs/b1-ptq/export` file, each set appears exactly once, and no output is a
`--limit` partial. This is two complete evaluations of disjoint sets on one
fixed model, not a combination of partial results.

## Protocol rules

- Each arm is scored on the test sets exactly once. No decision (checkpoint,
  recipe, learning rate, decoding setting) depends on a test number. If a run
  fails partway, it is rerun from scratch into a fresh directory and the
  failure is recorded; partial results are never combined.
- Decoding, normalizer, empty-reference rule and corpus WER follow the same
  procedure as B0. B0 (2026-09-30) was scored by an earlier revision of
  `evaluate.py` that predates the 30 ms padding rule and recorded no evaluator
  hash, so exact code identity with today's frozen evaluator cannot be shown
  from its artifacts. The padding rule cannot affect any test set (shortest
  test utterance 40 ms), and saved decoding/normalizer metadata agree; B0 is
  reused on that basis and the report states it.
- Report: per-set WER for B0 / B1 / M1, NVIDIA's published numbers, the
  seven-set mean (LS clean, LS other, AMI, Earnings-22, GigaSpeech, SPGISpeech,
  VoxPopuli; TED-LIUM unavailable, stated), Common Voice separately, absolute
  and relative gap M1 vs B0 per set, export size (MB) versus 2.47 GB.
- Secondary (labelled as such): runaway-output diagnostics as in the Whisper
  experiment are not planned; only if M1 shows insertion-dominated errors.

## Success criteria

None. This is a measurement, not a gate; the numbers are reported as they come.
Sanity checks, applied to M1 only, that would pause the write-up for
investigation (not a re-score): any set with M1 WER > 3x B0, or any set with
more than 1% empty M1 hypotheses. B1's 100% WER and empty outputs are the
expected PTQ result and are reported as is.

## Resource caps

24 GB memory per unit, 2 h runtime limit, GPU used only after the training
unit has exited, no network access needed.
