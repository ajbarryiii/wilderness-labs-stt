# Plan: binary-weight Parakeet pilot (start from FP32 or from ternary M1?)

**Status: 2026-10-17: complete. Results below ("Results"); selection record
`/mnt/hd/wilderness-labs-stt/parakeet-binary/runs/pilot-selection.json`. It led
to the main run MB1 (plans/main.md).**

## Question

The ternary QAT model M1 (`finetune/parakeet-ternary`, DESIGN.md "Main-run and
test results") costs 0.39 WER points on the seven-set test mean (6.84 vs 6.45)
with weights in {-1, 0, +1}. Binary weights in {-1, +1} halve the packed size
(about 75 MB of codes instead of 151 MB) and need one bitplane instead of two
in the CPU bitplane kernel. This pilot answers two questions:

1. **Starting point:** for binary QAT, is it better to start from the FP32
   original or from the ternary M1 latents?
2. **Cost:** at the same 12,000-step pilot budget, how much more WER does
   binary cost than ternary (ternary pilot P2 at 5e-4: 6.85% full dev)?

It does not decide whether to run a binary main run; the results feed that
decision, which the user makes.

## Results (2026-10-17)

Full-dev WER (%), final checkpoint (step 12,000, f = 1), scored on the model
rebuilt from its export. M1 and B0 are references; C is the ternary control.

| Model | LS clean | LS other | AMI | VoxPopuli | YODAS | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 (FP32 original) | 1.59 | 2.94 | 12.12 | 5.75 | 5.47 | 5.58 |
| M1 (ternary, start of B and C) | 1.99 | 4.20 | 11.71 | 6.02 | 6.51 | 6.09 |
| C (M1 + 12k ternary steps, 2e-4) | 1.96 | 4.35 | 11.86 | 6.12 | 6.64 | 6.19 |
| **B2** (M1, binary, 2e-4) | 2.42 | 5.51 | 12.90 | 6.53 | 7.76 | **7.02** |
| B5 (M1, binary, 5e-4) | 2.56 | 5.90 | 12.78 | 6.60 | 7.97 | 7.16 |
| A (FP32, binary, 5e-4) | 3.14 | 7.17 | 14.59 | 7.68 | 9.71 | 8.46 |

Zero-step points: Z-A (FP32 binarized) 100% on every set; Z-B (M1 binarized)
82.98%; gate G-B 6.084% vs M1 6.086%, passed.

- **Start point:** M1 wins; A - B2 = +1.44 points (tie threshold 0.15). M1's
  float parts and latents transfer; starting from FP32 does not catch up in
  12,000 steps.
- **Continuation control:** C - M1 = +0.10, a tie: re-warming M1 for 12k
  ternary steps neither helps nor hurts. B2 - C = +0.83 is therefore the binary
  cost at this budget (B2 - M1 +0.94); A - ternary pilot P2 = +1.61.
- **Learning rate:** 2e-4 beats 5e-4 from M1 by 0.14 points; 2e-4 also
  changed fewer latent signs (1.8% vs 3.9%).
- Binary loses most on LS other, YODAS and AMI; codes stay balanced (50.0% /
  50.0% per family). Export 105.3 MB (75.5 MB codes) vs 180.8 MB ternary.
- Rules: stop rule not triggered; B2 within 1.5 points of P2 (6.85%), so a binary
  main run from M1 at 2e-4 was proposed. The user approved 250,000 steps.

## Pre-pilot diagnostic (done 2026-10-15, CPU only)

`diagnostics/binary_diag.py` and `diagnostics/sign_agree.py`, outputs in
`diagnostics/result.json`. Over the 264 quantized matrices (604M weights),
with per-row s = mean|W|:

| Latents | ternary zeros | \|W\|/s < 0.1 | \|W\|/s < 0.25 | rel. error ternary | rel. error binary | \|\|B(W) - T(W)\|\| / \|\|T(W)\|\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 original | 32.4% | 6.8% | 16.7% | 0.533 | 0.620 | 0.686 |
| M1 (step 250,000) | 34.2% | 6.3% | 15.7% | 0.566 | 0.643 | 0.714 |

M1 vs FP32: 87.8% of latent signs agree (96.1% among weights that are nonzero
in FP32's ternary quantization); relative latent drift 0.46. FF, attention and
pointwise families behave alike (within about 1 point on every column).

Reading: ternary QAT did **not** collapse M1's latents onto a {-s, 0, +s}
lattice; their distribution is still close to the original's (near-Gaussian:
a Gaussian has 31% of mass below 0.5 * mean|W|). So the concern that M1 is a
bad starting point because of a dense cluster of near-zero latents with random
signs is not supported. What remains against M1 is functional: the network
relies on 34% of its forward weights being exactly 0, and binarizing them
moves the forward weights by 71% of their norm. What remains for M1 is that the
float parts (pre-encode, depthwise convolutions, norms, biases) have already
adapted to coarse weights over 250,000 steps. Neither argument settles the
question, hence the pilot.

## What stays exactly as in the ternary pilot

Everything in `parakeet-ternary/DESIGN.md` not listed under "What changes":
the quantized set (264 encoder modules; everything else float), recipe P2 (TDT
loss on the frozen FP32 teacher's transcripts plus encoder-output matching,
prediction and joint networks frozen), AdamW and its hyperparameters, BF16
autocast with FP32 master weights, 600 s duration-bucketed batches, gradient
checkpointing, SpecAugment on the student only, the pilot stream (`--train
pilot`, same seed, same data order), 12,000 optimizer steps, warmup 2% (240
steps), weight ramp 25% (3,000 steps), linear decay to zero, dev-subset
evaluation every 1,200 steps, selection on the final checkpoint (step 12,000,
f = 1), full-dev scoring on the model rebuilt from its export, checkpoint every
30 minutes with exact resume.

Arm A starts from the same weights and sees the same data as ternary pilot P2
at 5e-4, so A vs that run isolates ternary vs binary. Arm B re-reads the first
12,000 pilot batches, which M1 has already seen once; noted, not corrected.

## What changes

**Binary quantizer.** Per output row: scale α = max(mean|W|, 1e-8) (the
least-squares-optimal scale for sign codes, and the same statistic as the
ternary scale), codes c = +1 where W >= 0 and -1 where W < 0 (exact zero maps to
+1; `torch.sign` is not used because it returns 0). Forward weight exactly c * α
at f = 1. STE: identity gradient to the FP32 latent, no gradient to α.
Activations not quantized.

**Ramps (the only difference between arms besides initialization):**

- From FP32 (arm A): W_eff = lerp(W, B_ste(W), f), the ternary project's ramp
  with the binary quantizer.
- From M1 (arms B5, B2): W_eff = lerp(T_ste(W), B_ste(W), f), ternary to
  binary. At f = 0 this is bit-identical to M1's forward; identity gradient to W
  at every f.

**Initialization from M1.** Load `student` from
`/mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/ckpt/latest.pt`
(SHA-256 `003b1289b19114b58327bebb61af598412bdaca408db47a6949f7951d62d212a`,
step 250,000, f = 1); refuse any other hash. Load the full student state (latent
encoder weights and every float tensor), not just the quantized matrices.
Optimizer, scheduler and random states start fresh. Resume after a crash
restores the arm's own checkpoint and never re-initializes from M1 (the resume
key includes the init source and its hash).

**Export.** One bit per weight, eight codes per byte, least significant bit
first, bit 1 = +1, bit 0 = -1; one FP32 scale per row; pointwise convolutions
as [out, in]. Same safetensors layout, manifest, SHA-256, tokenizer and
reconstruction check as the ternary export (codes and scales must be exact;
logit difference recorded). Export refuses f < 1.

## Arms

| Arm | Init | Ramp | Peak LR | Run name |
| --- | --- | --- | ---: | --- |
| A | FP32 original | FP32 to binary | 5e-4 | `pilot-binA-fp32-lr5e-4` |
| B5 | M1 | ternary to binary | 5e-4 | `pilot-binB-m1-lr5e-4` |
| B2 | M1 | ternary to binary | 2e-4 | `pilot-binB-m1-lr2e-4` |
| C | M1 | none: stays ternary (f = 1 throughout) | 2e-4 | `pilot-terC-m1-lr2e-4` |

C is a control, added 2026-10-16, and not a candidate in the start-point
selection. It is M1 trained for the same extra 12,000 steps on the same data
with the same schedule as B2, but without binarization (the existing ternary
quantizer, unchanged). B2 - C isolates the cost of binarization from the
effect of more training and of re-warming M1. C - M1 (6.09%) shows whether a
re-warmed M1 keeps improving, which tests the hypothesis that M1 was still
compute-limited when its run ended.

B2 is included because M1 ended at learning rate 0 after a full decay, and
re-warming a converged model to 5e-4 may undo some of what makes it a good
starting point. 5e-4 was the best learning rate from FP32 for ternary (2e-4:
7.11, 1e-3: 7.04), so A is not swept.

**Zero-step points (full dev, no training, about 10 minutes each):**

- Z-A: FP32 original, binarized (binary PTQ). Expected near 100% like ternary
  PTQ (B1); recorded for completeness.
- Z-B: M1, binarized. Shows how much of M1 survives losing its zeros with no
  retraining.
- Gate G-B: arm B's model at f = 0 before any update must score M1's full-dev
  mean (6.09%) within 0.02 points; otherwise the init or ramp is wrong and
  nothing runs.

Cost: 4 arms at about 2.9 GPU-hours each plus scoring, about 12.5 GPU-hours,
run one at a time.

## Selection and reporting (fixed before any run)

Metric: unweighted full-dev mean WER over the five dev sets, final checkpoint
(step 12,000, f = 1), scored on the model rebuilt from its export.

- **Stop rule:** if every arm is above 16.7% (3x B0's 5.58%), stop and report;
  no follow-up without reassessment.
- **Starting point:** the better of B5 and B2 is "B". If |A - B| < 0.15 points,
  call it a tie and prefer A (standalone recipe, no dependence on the 69-hour
  M1 run). Otherwise the lower one wins.
- **Binary cost:** report A - 6.85 (ternary P2 pilot, same data and steps),
  B2 - C (same start, data, steps and learning rate; the cleanest measure) and
  B - 6.09 (M1, its starting point), with every per-set number.
- **Continuation control:** report C - 6.09. Read as "M1 still improving" only
  if C is below 6.09 by more than 0.15 points; within 0.15 is a tie.
- **Recommendation to the user (not a decision):** if the winner is within 1.5
  points of the ternary pilot (at or below 8.35%), propose a binary main run
  with the winning start. Otherwise propose a mixed pilot (feed-forward layers
  binary, 403M of the 604M quantized weights; attention and pointwise
  convolutions ternary) instead of a main run.
- Reported per arm: per-set and mean full dev, the dev-subset curve, code
  balance (+1 fraction overall and per family), the fraction of latent signs
  that changed during training (from the arm's start), export size, wall time,
  stream waits.

No test-set scoring in the pilot; test sets are touched only by a later main
run, once.

## Code layout and constraints

- New code lives in `finetune/parakeet-binary/`. Every file in
  `finetune/parakeet-ternary/` and `finetune/whisper-ternary/` stays
  byte-identical: their hashes are M1's recorded provenance (summary.json
  `source_hashes`), and a test asserts that.
- Reuse stream, teacher, data, evaluation and paths from parakeet-ternary by
  import, the way parakeet-ternary imports the Whisper quantizer. Where
  `train.py` hard-wires the ternary quantizer (`local_module("quant")`), the
  implementer proposes the smallest seam (for example a binary entry point that
  supplies its own quantizer module and init loader) and records it here
  before writing it.
- Artifacts: `/mnt/hd/wilderness-labs-stt/parakeet-binary/` (runs, eval, logs,
  tmp); read-only reuse of parakeet-ternary's model, data and caches. Verify
  `/mnt/hd` is mounted before writing.

### Implementation seam (recorded 2026-10-16, before the code)

No file in `parakeet-ternary/` or `whisper-ternary/` changes. The binary modules
have names that collide with nothing there (`binary_*.py`), so `import train`,
`quant`, `export`, `evaluate`, `stream`, `teacher` and `paths` keep resolving to
the ternary files, and `parakeet-binary/python` only prepends this directory to
the ternary interpreter's `PYTHONPATH`.

- `binary_quant.py`: the binary quantizer and both ramps. It imports
  `parakeet-ternary/quant.py` for `QUANTIZED_PATTERN`, the module count, the
  fraction validation and the ternary STE (`quant._effective_weight(W, 1.0)` is
  exactly the ternary module's forward weight), and exposes the same functions
  `train.py` calls on its quantizer (`quantized_module_names`,
  `set_weight_fraction`, `weight_fraction`, `code_histogram`,
  `parameter_accounting`).
- `binary_export.py`: 1-bit export and loader of its own; the tokenizer
  extraction and the reconstruction check are the ternary functions, the latter
  called with `export.quant` temporarily pointing at `binary_quant` (it reads
  codes as sign(W) and scales as the row max |W|, which is exact for binary).
- `binary_train.py --init {fp32,m1} --quantizer {binary,ternary} <train.py
  arguments>`: imports `train` and runs `train.main` unchanged after replacing a
  fixed list of module attributes (applied in one place, `seam()`, and restored
  on exit). The ramp follows from the pair: (fp32, binary) is "fp32" (arm A),
  (m1, binary) is "ternary" (B5, B2), (m1, ternary) is "none" (control C: the
  ternary quantizer unchanged, f = 1 from the first update); (fp32, ternary) is
  the ternary pilot itself and is refused. Every arm runs train.py's `--arm P2`.
  1. `train.parse_args`: adds `--init`, `--quantizer` and the derived, recorded
     arguments `ramp_kind`, `init_sha256`, `quantizer_sha256`;
  2. `train.RESUME_KEYS`: plus those five, so a checkpoint of another init,
     quantizer, ramp or quantizer source is refused (train.py's own exit 78);
  3. `train.TRAINING_SOURCES`: plus the `parakeet-binary` training sources, so
     config, resumes and summary record them and the sweep checks them;
  4. `train.build_context`: the student is the FP32 original, loaded with M1's
     student state on a fresh start of an `m1` arm (never on a resume: the
     checkpoint replaces every tensor), quantized with the arm's quantizer and
     ramp (C: `Context.ramp = 0`, so train.py sets f = 1 at every step); the
     Hub stream is opened with the worker count of the ternary pilot (see
     "Implementation notes"); the start's latent signs are stored for the
     sign-change report;
  5. `train.local_module`: `export` is the arm's export (binary, or the ternary
     export for C) wrapped so that the rebuilt model's decoder mode is pinned and
     an inexact reconstruction raises before scoring; for the binary arms `quant`
     is `binary_quant` (amended in review round 1, findings 1-2);
  6. `train.finish`: adds a `binary` block (init, quantizer, ramp, code balance
     per family, latent sign changes, decoder modes, training wall time) to the
     result, hence to `summary.json`;
  7. `paths.RUNS`, `paths.POWER`, `paths.ARTIFACTS` (the root the ternary
     export accepts, for C) and `train.POWERLOG_LAUNCH`: runs, exports and power
     logs under `/mnt/hd/wilderness-labs-stt/parakeet-binary/`. The GPU lock
     stays the ternary project's, so the two projects never share the GPU.
- `binary_eval.py --point {Z-A,Z-B,G-B}`: the zero-step points and the gate,
  scored with `train.DevEvaluator` exactly like an arm's final model.
- `binary_sweep.py`: the orchestration (G-B, Z-A, Z-B, then A, B5, B2, C),
  reusing `sweep.py`'s validation (`Plan`, `Expected`, `validate_run`,
  `run_train`) with the binary arguments and sources added.

## Contract tests (written before the implementation)

1. Binary quantizer: codes are only -1 and +1; W = 0 maps to +1; scale equals
   per-row mean|W| (clamped at 1e-8); rejects non-2-D input.
2. STE: forward value bit-identical to codes * scale; gradient to the latent is
   the identity at f = 1 and every f in between.
3. Ramp from FP32: f = 0 reproduces the original module's output bit for bit;
   f = 1 equals the binary forward.
4. Ramp from ternary: f = 0 equals the ternary module's forward bit for bit;
   f = 1 equals the binary forward.
5. Quantized set: exactly the same 264 module names as `parakeet-ternary`'s
   `QUANTIZED_PATTERN`; state_dict keys and shapes unchanged.
6. Pack/unpack round trip, including an input width that is not a multiple of
   8, and the bit convention (LSB first, 1 = +1).
7. Export: rebuilt model has exact codes and scales; export refuses f < 1;
   manifest SHA-256 matches the file.
8. M1 init: refuses a checkpoint with a different SHA-256, a missing or extra
   key, or f != 1; loads all student tensors.
9. Resume: an arm killed after its first checkpoint resumes from that
   checkpoint, not from M1 or FP32 (resume key covers init source, hash, ramp
   kind, quantizer).
10. Provenance: parakeet-ternary and whisper-ternary sources hash to the values
    in M1's summary.json.

## Procedure

1. Implement (Opus subagent) against the contract tests; all tests pass.
2. Smoke run on CPU or GPU with `--smoke` (a few hundred steps, small dev
   limit) for arm B, including a SIGKILL after the first checkpoint and a
   resume; plus the G-B gate on full dev.
3. Codex review (`gpt-6-astra`, reasoning effort xhigh, read-only, explicit
   file scope) of this plan and the code; fix or record each finding; repeat
   until no blocker or major remains.
4. Z-A and Z-B scoring.
5. Arms A, B5, B2, C in that order, each in its own capped unit, one at a time,
   for example:

   ```sh
   ./heavy binary-pilot --mem-max 36G --runtime 14h -- python <binary sweep entry> --phase pilot
   ```

   (exact command fixed in step 1; see "Implementation notes", item 1). A crash or power loss is resumed by rerunning
   the same command. Optional: cap GPU power (for example `nvidia-smi -pl 450`)
   given the 2026-09-29 breaker trip; the ternary pilot drew about 375 W.
6. Write results here and in a `finetune/parakeet-binary/README.md` results
   section; report to the user with the recommendation.

## Implementation notes (2026-10-16, step 1; not yet reviewed)

Code in `finetune/parakeet-binary/` (`binary_paths.py`, `binary_quant.py`,
`binary_export.py`, `binary_train.py`, `binary_eval.py`, `binary_sweep.py`,
`python`, `heavy`, `tests/`); seam as recorded above. All tests run on the CPU
(`./heavy binary-tests --mem-max 24G --runtime 30min --wait -- python -m
unittest discover -s tests -v`, from `finetune/parakeet-binary/`; the
interpreter has unittest, not pytest).

1. **Commands** (from `finetune/parakeet-binary/`). Each phase starts with the
   G-B gate (scored once, then reused) and stops if it did not pass; scored
   points and finished arms are reused only if their sources still match.
   - Step 2 gate: `./heavy binary-gate --mem-max 36G --runtime 1h -- python binary_sweep.py --phase gate`
   - Step 4: `./heavy binary-zero --mem-max 36G --runtime 2h -- python binary_sweep.py --phase zero`
   - Step 5: `./heavy binary-pilot --mem-max 36G --runtime 16h -- python binary_sweep.py --phase pilot`
     (four arms of about 2.9 h plus finishing; a runtime stop checkpoints and the
     same command resumes).
   - Dry run: `./python binary_sweep.py --phase pilot --dry-run --allow-unconfined`.
   - One arm by hand (e.g. the step-2 smoke): `python binary_train.py --init m1
     --quantizer binary --arm P2 --lr 5e-4 --run-name smoke-binB --smoke
     --stream hub --grad-checkpointing` inside a `./heavy` unit.
   The arms run one at a time inside the one capped unit of the sweep (its
   children inherit the cgroup), as in the ternary pilot and the example
   command above, not each in a unit of its own.
2. **Data order.** The plan says arm A "sees the same data as ternary pilot P2".
   That holds only with the pilot's 6 stream workers: `stream.py`'s
   `DEFAULT_WORKERS` became 8 for M1 (DESIGN.md "Main run"), and the worker count
   changes the item order. `binary_train.py` opens the pilot stream with 6
   workers and refuses to train unless its configuration hash is the one P2
   recorded (`874bf7e4...`; checked offline with the current `stream.py`). All
   four arms see that same data in the same order.
3. **"Arm B re-reads the first 12,000 pilot batches, which M1 has already
   seen"** is not accurate: M1 ran with 8 workers, so its item order differed
   from the pilot's. The B arms and C see data from the same sources and largely
   the same files as M1's first hours, so overlap is partial, not complete.
   Noted, not corrected, as before.
4. **Reference values are the recorded ones, not the rounded ones:** B0 full dev
   5.575% (stop threshold 3 x 5.575 = 16.72%), ternary pilot P2 6.850%, M1
   6.086% (gate and C - M1 compare against 6.086, not 6.09). The
   recommendation's limit is ternary pilot + 1.5 points = 8.350%.
5. **Stop rule** counts the candidate arms A, B5 and B2 only. C is a ternary
   control and would otherwise keep the rule from ever triggering.
6. **Boundaries:** start-point tie is |A - B| < 0.15 points (strict, as
   written); continuation "within 0.15" is inclusive; "M1 still improving"
   needs C - M1 < -0.15; above +0.15 the report says "worse than M1". If B5 and
   B2 are exactly equal, B is B5.
7. **G-B** is scored on the QAT model at f = 0 in FP32 (f < 1 is not
   exportable), while M1's 6.086% was scored on M1's rebuilt export (float
   tensors stored in FP16), so the two are close but not bit-identical; the 0.02
   tolerance covers that. The gate also requires the encoder output at f = 0 to
   equal the ternary M1 model's bit for bit on four dev utterances (that ternary
   model is C's start). The first training update of A and B is at f = 1/3000,
   as in the ternary pilot.
8. **C** uses `parakeet-ternary/quant.py` and `export.py` unchanged; `Context.ramp`
   is 0, so f = 1 from the first update and `summary.json` has `ramp_steps` 0.
   Its export is the ternary 2-bit format (written under the binary artifacts).
   Its resume key differs from B2's in `quantizer`, `ramp_kind` and
   `quantizer_sha256`.
9. **Gradient identity** between f = 0 and 1 holds up to the rounding of
   (1 - f) + f, exactly as in the ternary ramp; it is exact at f = 1.
10. **Reported per arm** (`runs/pilot-selection.json`): wall time is
    finished - started (including any downtime between sessions); training wall
    time is recorded by the seam in every checkpoint (review round 1, finding
    7); stream waits are summed from `metrics.jsonl`'s per-step accumulators,
    with the number of 20-step windows that waited more than 50 ms per step. Code balance has -1 / 0 / +1
    fractions (0 is always 0 for binary) overall and per family (feed_forward,
    self_attn, pointwise_conv); sign changes compare the latent signs (W >= 0)
    with those stored at the arm's start (`start-signs.safetensors`, written on
    the fresh start only, never on a resume).
11. **M1 loading:** the 7.3 GB checkpoint is hashed once and the result cached
    in `caches/sha256/` under its path, size, mtime, inode and device (a
    rewrite is hashed again); it is memory-mapped and only `student` is read.
12. **One write outside the binary artifacts:** `stream.make_training_stream`
    rewrites `parakeet-ternary/data/stream/sources.json` and
    `yodas_train_files.json` at every start (stream.py's own behaviour, the
    same content for the same configuration). The tests patch it out.

## Smoke runs (2026-10-16, step 2)

Plumbing only, `--smoke` with 16 utterances per dev set; none of these numbers
is a result. Runs under `/mnt/hd/wilderness-labs-stt/parakeet-binary/runs/smoke-*`.

- **Environment fix.** The first arm-B smoke run crashed with a CUDA illegal
  memory access inside NeMo's greedy TDT decoder, in the teacher's first
  labeling after the step-100 dev evaluation. It reproduced exactly on resume.
  Cause: `parakeet-ternary/python` gained NVRTC on the library path on
  2026-10-05, after M1, which enables NeMo's full CUDA-graph decoder. M1's log
  shows that decoder failing to compile and falling back to native PyTorch CUDA
  graphs throughout training. `parakeet-binary/python` therefore omits NVRTC, so
  the pilot decodes exactly as M1 and the ternary pilot did. With that change the
  same run resumed and finished.
- **Arm B path (`smoke-binB`, M1 init, binary, 200 steps, 1-minute
  checkpoints).** The M1 hash was verified and all 725 student tensors loaded.
  The loss rose with the ramp (0.65 at f = 0.4, 1.25 at f = 1) and then fell (1.04
  at step 200). After each of the two crashes the rerun resumed from the latest
  checkpoint (steps 73 and 77) and reproduced the step-100 loss (1.1267) and
  dev-subset WER (13.85%) exactly. The export is 105.3 MB (75.5 MB packed codes,
  1.8 MB scales, 27.8 MB FP16 float tensors), codes and scales exact, greedy
  hypotheses equal between the QAT model and the rebuilt export. Code balance
  50.0% +1; latent sign changes 0.66% of weights (attention 1.3%).
- **Arm A path (`smoke-binA`, FP32 init, 20 steps).** Completed through
  export; 98.6% WER, as expected for binary after a 5-step ramp.
- **Control C path (`smoke-terC`, M1 init, ternary, 20 steps).** Completed
  through the ternary export; 6.52% dev-subset WER (M1 level).
- **Gate path (`binary_eval.py --point G-B --limit 16`).** Runs end to end and
  correctly gives no verdict on a limited run. (Moved from `eval/G-B` to
  `eval/smoke-G-B-limit16` in review round 1, finding 4.)
- **After the review round 1 fixes (`smoke2-binB`, M1 init, binary, 120 steps,
  eval every 60).** Passed the post-evaluation teacher labeling that crashed
  before, and finished through export. Codes and scales were exact, and
  `train_wall_s` was recorded. `decoder_graphs` recorded `no_while_loops` for
  the teacher, the student, the rebuilt model, the dev subset and the full dev
  set. The "Full CUDA graph compilation failed" fallback no longer appears,
  because the mode is set explicitly.

## Review round 1 (2026-10-16)

Codex (`gpt-6-astra`, xhigh) review of the plan and code after the smoke runs:
NOT CLEAN, 2 major and 5 minor findings
(`/mnt/hd/wilderness-labs-stt/parakeet-binary/review/r1.md`). Resolutions:

1. **major, reconstruction failures did not invalidate a run.** Fixed. The seam's
   `train.local_module("export")` is now a `FinishingExport` around the arm's
   export (`binary_export`, or `parakeet-ternary/export.py` for C) whose
   `reconstruction_check` raises `ReconstructionError` unless `codes_exact` and
   `scales_exact` are both true; `train.finish` runs that check before it scores
   anything, so a failed check leaves no `eval-dev.json`. The zero-step points
   use the same wrapper. `binary_sweep.py` rejects an arm or a Z point whose
   record lacks an exact reconstruction. Negative tests: a tampered rebuilt
   model for B and C, and for Z-A; sweep validation of missing or false fields.
2. **major, NVRTC removal not enforced.** Fixed, three ways. (a) `python` drops
   every inherited `LD_LIBRARY_PATH` entry that holds `libnvrtc*`. (b) NeMo's
   mode lives in `model.decoding.decoding.decoding_computer.cuda_graphs_mode`
   (`GreedyBatchedLabelLoopingComputerBase.CudaGraphsMode`, label_looping_base.py).
   M1's log shows NeMo choosing `full_graph`, failing to compile it without
   NVRTC and falling back ("Falling back to native PyTorch CUDA graphs") to
   `no_while_loops` before its first decode. `binary_train` now sets
   `no_while_loops` explicitly with NeMo's `force_cuda_graphs_mode` on the
   teacher and student at build time, on every model the dev evaluator decodes,
   and on the model rebuilt from the export. That is the same partial-graph
   decoder M1 ran after its fallback, without the failed full-graph attempt, so
   decoding outputs do not change. It cannot be checked bit for bit on the CPU;
   the G-B gate and the smoke runs check it on the GPU. (c) The mode is verified
   before and after every teacher labeling and after every evaluation, and any
   other mode raises `DecoderModeError`. A decoder config that disallows graphs
   is refused. The modes are recorded in `config.json` and `summary.json`
   (`binary.decoder_graphs`) and in the Z/G summaries, and the sweep refuses runs
   that lack them or record another mode.
3. **minor, provenance excluded the wrapper.** Fixed: `parakeet-binary/python`
   and `heavy` are in `BINARY_SOURCES`, hence in `train.TRAINING_SOURCES` under
   the seam (config, every resume, summary) and in the sweep's checks; decoder
   modes recorded as in 2.
4. **minor, limited scoring occupied the production gate path.** Fixed:
   `binary_eval.py --limit N` writes `eval/smoke-<point>-limit<N>` (exports under
   `runs/smoke-<point>-limit<N>`). The existing limited G-B (16 utterances,
   2026-10-16 17:08 UTC, no verdict, ternary twin equal) was moved, not
   deleted, to `eval/smoke-G-B-limit16`; `eval/G-B` is free, so the documented
   gate, zero and pilot commands run the full gate (checked with `--dry-run`).
5. **minor, contract test 9 did not do a real kill and resume.** Accepted by the
   coordinator, because `train.main` and `train_loop` need CUDA and the real 600M
   model. Evidence for the real path: `runs/smoke-binB` crashed twice on the GPU
   (CUDA illegal memory access, finding 2) and was resumed by rerunning the same
   command from its checkpoints at steps 73 and 77 (`resumes.jsonl`). The resumed
   run reproduced the step-100 loss (1.1267) and dev-subset WER (13.85%) exactly.
   The test was still strengthened: it now goes through the seam's
   `build_context`, which detects the resume from the run directory, with a real
   AdamW update before the checkpoint. It restores model, optimizer, scheduler,
   RNG and ramp, and checks that the next update equals the uninterrupted run's
   bit for bit. (While writing it: NeMo's dither adds its noise to the input
   tensor in place, so the test passes a copy each step; train.py gets a fresh
   batch every step.)
6. **minor, data-repetition wording.** Fixed by the coordinator in
   "Augmentation and more data".
7. **minor, training time inferred from metric-step differences.** Fixed:
   `BinaryContext` records the training wall time, from building each session's
   context to its latest checkpoint and summed over sessions, in every checkpoint
   (`binary_timing`), restores it on resume, and reports it in
   `summary.json` (`binary.timing`). The report uses that time instead of
   `step_time_s`, whose timer restarts at evaluations, checkpoints and resumes.
   Stream waits stay derived from `metrics.jsonl`. That derivation is exact:
   `data_wait_s_per_step` is train.py's accumulator over the steps since the
   previous line, the accumulator is checkpointed, and lines after the
   checkpoint are pruned on resume. Tests cover a resume-shaped metrics file and
   save, kill and restore of the timing with a stubbed clock. Checkpoints written
   before this change (the smoke runs) have no timing record, so a resume of one
   reports `uncounted_before_step`.

## Augmentation and more data: deliberately not in this pilot (2026-10-16)

Raised by the user: more augmentation and more training data, for robustness
and because binary may converge less easily. Kept out of the pilot because:

- **Convergence looks compute-limited more than data-limited (corrected after
  review round 1).** The hours below are exposure, not unique audio: the
  sources cycle independently. M1's 33,248 exposure hours were YODAS 22,040,
  People's Speech 3,370, LibriSpeech 3,188, VoxPopuli 3,145 and AMI 1,504.
  LibriSpeech (960 h) was therefore seen about 3.3 times, and the smaller AMI
  and VoxPopuli training sets several times each. The 12,000-step pilot (about
  1,600 exposure hours) repeats little. Despite the repetition, M1's training
  losses were still falling at the end (TDT 0.435 at step 62,500, 0.379 at
  150,000, 0.315 at 240,000). Its dev-subset WER also kept improving, including
  on AMI and VoxPopuli dev, the most repeated domains (6.43%, 6.47%, 6.31% over
  the last three evaluations; the late gains are partly the learning-rate decay
  to zero). That points to underfitting rather than memorization, so more
  steps or a stronger distillation signal are the first convergence levers. It
  does not rule out that more or more diverse data would help, especially for a
  long binary main run, where the small sources would repeat even more.
- **Augmentation is a robustness lever, and the current dev sets cannot see it.**
  The five dev sets are mostly clean, so a robustness gain would not show up and
  a small clean-WER cost would. Measuring it needs a noisy evaluation set first.
- **Mixing factors would confound the start-point question.**

Proposed follow-up, after this pilot and only on the winning start: one
augmentation arm against the plain arm. The student gets noise, reverb, speed
and band-limited (radio) audio, while the teacher labels and matching target
stay on clean audio. It is judged on a fixed-seed noisy copy of the dev sets
(held-out noise sources, fixed SNRs) and on clean dev. Separately, the Common
Voice gap (M1 +4.06 points) points to accent and microphone diversity in the
mixture, which is a data-composition change rather than more hours.

## Out of scope

Activation quantization, packed binary kernels and speed or energy claims (the
CPU bitplane kernel is a separate follow-up), test-set scoring, the mixed
FF-binary variant (only proposed, see above), learnable or mean-centered
scales, and publishing.
