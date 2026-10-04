# Ternary-weight QAT of NVIDIA Parakeet-TDT-0.6B-v2: experimental design

Written 2026-09-30. Experimental design and architecture are fixed here;
implementation follows this document. Deviations require updating this file
first. Budget: about one week of the single RTX 5090.

## Question

How close to the original can Parakeet-TDT-0.6B-v2 stay on the Open ASR
Leaderboard test sets when almost every encoder weight is constrained to ternary
codes {-1, 0, +1} with one FP32 scale per output row, after quantization-aware
training (QAT) from the pretrained checkpoint?

This extends `finetune/whisper-ternary`, where the same quantizer took Whisper
tiny.en to 12.1% LibriSpeech test-clean WER (4.5% FP32 control) only once a
float-to-ternary ramp and distillation from the FP32 model were added.

## Base model

`nvidia/parakeet-tdt-0.6b-v2`, revision `ae9ad07059c7c739ffaf932226a8fe64ae2620b0`,
CC-BY-4.0, stored at
`/mnt/hd/wilderness-labs-stt/parakeet-ternary/models/parakeet-tdt-0.6b-v2/` with a
SHA-256 `lock.json`. NeMo `EncDecRNNTBPEModel`: FastConformer encoder (24 layers,
d_model 1024, 8 heads, rel-pos attention, 8x subsampling, conv kernel 9), LSTM
prediction network, joint network, TDT durations {0..4}, 1,024 BPE tokens.
617,825,926 parameters: encoder 608,879,616; prediction network 7,219,840;
joint 1,726,470. Loads and transcribes on the 5090 with the existing NeMo 3.0 /
torch 2.11 cu128 runtime (`custom/stt-distillation/python`). NeMo falls back
from CUDA-graph decoding on this GPU; decoding is correct but slower.

## Quantized set

Ternary, in every one of the 24 encoder layers:

- `feed_forward1.linear1`, `feed_forward1.linear2`, `feed_forward2.linear1`,
  `feed_forward2.linear2` (402.7M parameters in total)
- `self_attn.linear_q`, `linear_k`, `linear_v`, `linear_out`, `linear_pos`
  (125.8M)
- `conv.pointwise_conv1`, `conv.pointwise_conv2` (Conv1d with kernel 1, treated
  as a linear map on the [out, in] weight; about 75M)

Kept in floating point: the convolutional subsampling front end and its output
linear, depthwise convolutions, all norms and biases, the prediction network and
the joint network. These are about 15M parameters. The code computes and records
exact counts; the design commits to the module list, not to the rounded totals.

## Quantizer

Identical to `finetune/whisper-ternary` (see its DESIGN.md "Quantizer"): per
output row `s = max(mean|W|, 1e-8)`, `C = clamp(round(W / s), -1, 1)`,
`W_hat = C * s`, forward weight exactly `W_hat`, identity straight-through
gradient `W_hat + (W - W.detach())`, FP32 latent weights, recomputed every
forward, no gradient to the scale, activations not quantized. Progressive ramp
`lerp(W, W_hat_ste, f)` with `f` rising linearly from 0 to 1 over the first
25% of steps (the Whisper v2 setting); only checkpoints at `f = 1` can be
selected or exported.

## Supervision: teacher transcripts, not corpus labels

Parakeet emits punctuated, capitalized text, learned from Granary transcripts
that preserve both. Most public training corpora have lowercase or uppercase
text without punctuation; training on them would teach the student a different
output style and inflate its error against the teacher. The student is
therefore trained on the frozen FP32 teacher's own greedy transcripts
(sequence-level knowledge distillation). Human transcripts are used only to
filter bad segments and for evaluation.

Labels are produced online (revised 2026-09-30, because training audio is
streamed and not stored): every batch, the frozen teacher encodes the clean,
unaugmented audio and decodes it greedily with its own decoding configuration.
The same teacher encoder output feeds the encoder-matching loss, so no second
teacher forward is needed. Every arm, including P1 and the FP32 control, loads
the teacher for labeling.

Filtering, applied per batch: keep utterances of 1 to 30 seconds with a
nonempty teacher transcript; for sources with human transcripts, drop an
utterance if the normalized WER between teacher and human transcript exceeds
50% (misaligned or mislabeled audio). Counts dropped per source and reason are
logged.

## Training data

Training audio is streamed from the Hugging Face Hub during training and not
stored on the machine (user decision, 2026-09-30). LibriSpeech train-clean-100
and train-clean-360 may be read from the copies already on the data disk.
Development and test sets are kept locally because they are scored repeatedly.
Sources (pinned revisions recorded by the streaming code):

| Source | Why | Target share of exposure |
| --- | --- | ---: |
| YODAS-Granary English `asr_only` shards (`espnet/yodas-granary`) | The web-speech distribution Parakeet was trained on | 65% |
| LibriSpeech 960 h | Read speech; human transcripts for filtering | 10% |
| People's Speech `clean` subset | Diverse public recordings | 10% |
| VoxPopuli English train | Parliamentary speech, accents | 10% |
| AMI IHM train | Meetings, spontaneous speech | 5% |

The sources are interleaved with these sampling probabilities through a seeded
shuffle buffer. YODAS shards reserved for development are excluded from the
training file list. The pilot and the main run use the same stream; the pilot
simply runs fewer steps. Before the pilot, the stream's delivery rate on this
network must be measured against the GPU's consumption rate; if streaming cannot
keep the GPU fed, the plan is revised here before any run, not worked around
silently.

Known overlap risk: YODAS and GigaSpeech both draw on YouTube. The teacher was
also trained on YODAS, so any overlap applies equally to the reference.

## Distillation variants compared in the pilot

Every arm uses the TDT loss on teacher transcripts, the 25% ramp, and identical
data order, steps and schedule. They differ only in:

| Arm | Extra loss | Prediction and joint networks |
| --- | --- | --- |
| P1 | none (sequence-level distillation only) | frozen |
| P2 | encoder-output matching | frozen |
| P3 | encoder-output matching | trainable |

Encoder-output matching: `mean((E_s - E_t)^2) / var(E_t)` over valid frames of
the final encoder output, weight 1.0, where the teacher is the frozen FP32
model in BF16 autocast on the same (unaugmented) input. With frozen prediction
and joint networks, a student encoder that reproduces the teacher's encoder
output reproduces the teacher.

## Training recipe

AdamW, betas (0.9, 0.98), eps 1e-8, weight decay 0.01 on matrices only,
gradient clipping at 1.0, BF16 autocast with FP32 master weights, linear warmup
over the first 2% of steps then linear decay to zero. Batches are built by total
audio duration (duration bucketing), with the batch size and any gradient
checkpointing fixed in Phase 0 from measured memory. The model's configured
SpecAugment is kept for the student input only. Seed 20260930.

Runs must survive a hard power loss and network outages: checkpoint every 30
minutes with model, optimizer, scheduler, the stream's position state, random
states and ramp fraction, and resume from the latest complete checkpoint.
Because the data is streamed, resume continues from the saved stream position
but is not bit-identical to an uninterrupted run. Checkpoints are written to a
temporary name and renamed, so a torn write never replaces a good one. If the
stream fails after its own retries, the run checkpoints and exits with a
distinct code; the orchestrator waits and resumes it, and never restarts a run
from scratch silently.

## Evaluation

- **Test sets** (primary, reported once per arm at the end): the Open ASR
  Leaderboard bundle `hf-audio/esb-datasets-test-only-sorted`, revision
  `b6bdcd0beb`, original (not `_cleaned`) configs: LibriSpeech test-clean and
  test-other, AMI, Earnings-22, GigaSpeech, SPGISpeech, VoxPopuli, Common Voice.
  TED-LIUM is not in the bundle and is not evaluated; this is stated wherever the
  mean is reported. The primary summary is the unweighted mean WER over the
  seven sets NVIDIA reports for this model except TED-LIUM, plus every per-set
  number. Common Voice is reported separately.
- **Development sets** (selection only): LibriSpeech dev-clean and dev-other,
  AMI IHM validation, VoxPopuli English validation, and 2,000 utterances from
  reserved YODAS shards. In-training evaluation uses a fixed 400-utterance subset
  of each; recipe and learning-rate selection use the full sets. The selection
  metric is the unweighted mean WER over the five.
- **Scoring:** NeMo greedy TDT decoding with the model's configuration,
  identical for every arm; the Whisper English text normalizer on both sides,
  which is what the leaderboard uses; corpus-level WER; per-utterance records
  saved.
- **Reproduction gate:** before any training, the FP32 original must reproduce
  NVIDIA's published numbers (1.69, 3.19, 11.16, 11.15, 9.74, 2.17, 5.95) within
  0.2 absolute WER points per set, or the difference must be explained and
  documented. No QAT result is interpreted until this gate passes.
- Every ternary number is scored on the model rebuilt from its export.

## Arms and phases

**Phase 0: infrastructure and baselines (about 1.5 days).**
Environment, development and test data, the resumable training stream and its
measured delivery rate, online teacher labeling, evaluation harness, quantizer
and export for NeMo, resumable training loop. Baselines: B0 FP32 original on all test and development sets
(the reproduction gate); B1 ternary post-training quantization, no training.
Measure memory and throughput for student plus teacher at candidate batch sizes.

**Phase 1: pilot (about 1 day).** P1, P2 and P3 at learning rate 5e-4, each for
the same fixed step count sized to about 3 GPU-hours. Then the best of the
three at 2e-4 and 1e-3. Selection: lowest full development mean WER at the
final `f = 1` checkpoint. Stop rule: if every pilot arm is above 3x the FP32
development mean, stop and reassess before spending the main budget.

**Phase 2: main run (about 3 days).** M1: the selected recipe and learning rate
on the streamed mixture, run for a fixed step count computed from measured
throughput to fill 60 GPU-hours, recorded here before launch.

**Phase 3: evaluation, export and publication (about 1 day).** Export M1, score
it on every test set, write the results and model card, publish to Hugging Face
under the same CC-BY-4.0 terms with attribution.

**Optional A1 FP32 control**, only if at least 12 GPU-hours remain: the FP32
model fine-tuned on the same teacher transcripts for the same steps. It
separates the effect of the data mixture from the effect of quantization. If it
is not run, the FP32 original (B0) is the reference, and the report says so.

## Export

The Whisper experiment's 2-bit format: codes packed four per byte, one FP32
scale per row, pointwise convolutions stored as [out, in]. Every other tensor
in FP16. The loader rebuilds a NeMo model in FP32 with dequantized weights. A
reconstruction check requires exact codes and scales and records logit
agreement. The expected size is about 180 MB against 2.47 GB for the original.

## Artifacts, power and hygiene

Code lives in `finetune/parakeet-ternary/`. All data, labels, checkpoints,
exports and evaluation outputs go under
`/mnt/hd/wilderness-labs-stt/parakeet-ternary/`, and every wrapper refuses to run
without the mount. No weights enter Git. Every GPU job runs detached with the
`powerlog.py` flight recorder from the Whisper experiment. Codex (gpt-6-astra,
xhigh) reviews each code batch before it runs on the GPU and before anything is
published.

## Phase 0 results and recorded deviations (2026-09-30)

**Reproduction gate: passed.** FP32 original, greedy TDT, FP32 decoding, all
192 hours of test audio in 15.4 minutes:

| Set | Ours | Published | Difference |
| --- | ---: | ---: | ---: |
| LibriSpeech clean | 1.70 | 1.69 | +0.01 |
| LibriSpeech other | 3.19 | 3.19 | 0.00 |
| AMI | 11.15 | 11.16 | -0.01 |
| Earnings-22 | 11.24 | 11.15 | +0.09 |
| GigaSpeech | 9.78 | 9.74 | +0.04 |
| SPGISpeech | 2.14 | 2.17 | -0.03 |
| VoxPopuli | 5.94 | 5.95 | -0.01 |
| Mean of the seven | 6.45 | 6.44 | |
| Common Voice (separate) | 8.50 | n/a | |

**B1 post-training quantization:** 100.00% WER on LibriSpeech clean and other and
AMI (every hypothesis empty). Export 180.8 MB (packed codes 151.0, row scales
1.8, FP16 tensors 27.8 MB) against 2.47 GB.

Deviations from the text above, accepted:

1. **Normalizer and scoring:** transformers' Whisper `EnglishTextNormalizer`
   with `normalizer.json`, verified identical to the leaderboard's 2025
   normalizer on all 42,716 test references. As the leaderboard does,
   utterances whose normalized reference is empty are decoded but not scored
   (AMI 990, Earnings-22 4, GigaSpeech 33). Common Voice repeats 5,673 ids with
   different audio; repeats get a `#k` suffix and are scored separately.
2. **Precision:** all evaluation decodes in FP32 (the current leaderboard script
   uses BF16). Online teacher labels use a BF16 encoder and FP32 decoding; they
   match FP32 labels on 98.6-98.8% of utterances (0.01-0.02% normalized WER
   between them) at about twice the throughput (2,580 versus 1,450 audio hours
   per GPU-hour at 600 s batches, teacher alone). The 1-30 s duration filter is
   applied in the teacher's keep mask.
3. **Export:** the feature-extraction constants (window and mel filterbank,
   0.13 MB) stay FP32 and integer BatchNorm counters keep their type; all other
   non-ternary tensors are FP16 as specified.
4. **Code reuse:** the Whisper quantizer and WER code are imported by file path
   (module names collide); `pack_codes` and `unpack_codes` are copied verbatim and
   tested against the original.

## Infrastructure incident and fixes (2026-09-30 to 2026-10-01)

At 17:39 on 2026-09-30 a streaming benchmark (8 workers, about 7 GB each) ran
alongside a GPU probe inside the agent session's own process group, with no
swap; the global OOM killed desktop processes and the session. Findings and
fixes, each verified in isolation before the next:

1. **Isolation.** Every heavy job now runs through `./heavy`, a memory-capped
   systemd user unit (MemoryMax 36G by default, no swap, first OOM victim, no
   soft limit so an overrun is killed in its own unit instead of stalling),
   leaving at least 24 GB for the desktop and session. One heavy job at a time.
2. **Stream memory.** pyarrow's default parquet pre-buffering kept every column
   chunk read so far alive while a file was open, so worker memory grew with
   bytes read (about 1.4 GB per 586 MB shard; 14 open files per worker).
   `pre_buffer=False`, an 8 MB read buffer, no remote cache, and 6 open files
   per worker (2 YODAS, 1 per other source) bound a worker at about 2.1 GB:
   flat for 10 minutes at 4 workers (8.6 GB total), previously 32 GB.
3. **Stream throughput.** Delivery is latency-bound (0.1 CPU cores). 4 workers
   gave 493 audio-hours per hour, 6 workers 766, against about 580 consumed by
   training; the default is now 6 workers.
4. **CUDA compilation.** NeMo's TDT loss JIT-compiles numba CUDA kernels; numba
   0.67 with CUDA 12.9 NVVM segfaults when targeting sm_120 and cannot compile
   builtin min/max in device code. Training targets compute_90 PTX (driver JIT to
   sm_120) and rebuilds the two gradient kernels with the clamp lines rewritten.
   Checked against NeMo's pure-PyTorch references at three sizes: TDT loss
   within 3.4e-7 relative, gradients within 4.2e-5 of the largest gradient; the
   standard RNN-T kernel used on the omega branch: loss exact, gradient 1.1e-4,
   just over the 1e-4 bound set beforehand and consistent with FP32 lattice
   rounding growing with size. Accepted and recorded.
5. **Bounded training smoke (P2, 600 s batches, gradient checkpointing, Hub
   stream, 600 steps).** 0.83 s per step, 570-586 audio-seconds per second,
   stream wait 0.00 s per step; process memory steady at about 18.5 GB (page
   cache from 7.3 GB checkpoint writes brings the unit to 29-33 GB, which is
   reclaimable); GPU 375 W. A SIGKILL of the whole unit at step 280 and a rerun
   of the same command resumed at the step-223 checkpoint with the learning-rate
   schedule continuous. Export reconstruction exact (180.8 MB). Dev mean WER
   after 600 steps (80 audio hours): 12.87% full dev (plumbing check only).

6. **Data decisions confirmed (2026-10-01).** Source shares are shares of audio
   hours (per-item probabilities are derived from measured mean durations).
   YODAS has no human transcript, so the teacher-versus-human filter does not
   apply to it. Development sets are not duration-filtered, except yodas_dev,
   which applies the training 1-30 s filter because it measures in-distribution
   performance on the training source (corrected after the Codex review, which
   found the earlier sentence inaccurate). VoxPopuli's filter
   reference is `normalized_text`, the column the leaderboard's VoxPopuli test
   uses. The 1-30 s training cap is kept even though it drops about 13% of YODAS
   rows (up to about a third of a shard's hours, mostly 30-40 s segments): the
   smoke run already used 27-29 GB of the 32 GB GPU at 600 s batches, and
   attention memory grows quadratically with utterance length. Because the
   stream is effectively unlimited, the cost is a bias toward shorter YODAS
   segments, not a shortage of data.

7. **Pilot configuration fixed before launch (2026-10-01).** 600 s batches with
   gradient checkpointing (the worst case of 30 s utterances fits at 21.6 GB;
   1,200 s batches run out of memory on long-utterance batches for 2% more
   speed). 12,000 steps per pilot arm, about 2.9 GPU-hours and 1,600 audio hours
   each. The student encodes its own augmented features (dither and SpecAugment)
   and the teacher the clean audio; one student forward feeds both the TDT and
   the encoder-matching loss. If P2 and P3 lose to P1, the first follow-up is to
   exclude time-masked frames from the matching loss. M1 at 60 GPU-hours is
   about 254,000 steps and 29,000 audio hours, against about 100,000 hours in the
   YODAS training file list.

8. **Codex review of the Phase 0 code (2026-10-01), resolutions.** Pilot arms are
   scored and selected on their final checkpoint (step = max_steps, f = 1), as
   the Phases section specifies, not on the best interim dev-subset checkpoint.
   The prediction and joint networks run with dropout disabled in every arm, so
   P3 differs from P2 only in trainability. Training refuses to start on
   unreadable checkpoints instead of restarting from zero; best-weight files are
   never deleted while a durable checkpoint references them; the sweep validates
   every reused run's settings and code hashes, and validates the full-dev FP32
   baseline before the first pilot. Evaluation zero-pads utterances shorter than
   30 ms to 30 ms (one 20 ms ami_dev segment crashed NeMo's per-feature
   normalization; every test utterance is at least 40 ms, so no reported
   number changes).

9. **FP32 development baseline (B0-dev, 2026-10-01).** LibriSpeech dev-clean
   1.59, dev-other 2.94, AMI dev 12.12, VoxPopuli dev 5.75, YODAS dev 5.47;
   five-set mean 5.58% (one 20 ms AMI segment padded). The pilot stop threshold
   is therefore 3 x 5.58 = 16.7% development mean WER.

10. **Code frozen for the pilot (2026-10-01 06:09 UTC).** After two Codex
    re-checks (all findings resolved), the source hashes below are what every
    pilot run must match; sweep.py refuses to reuse or resume a run whose
    recorded hashes differ. Snapshot:
    `/mnt/hd/wilderness-labs-stt/parakeet-ternary/frozen-code-20261001T0609Z/`.
    Launch: `./heavy pilot --mem-max 36G --runtime 30h -- python sweep.py --phase pilot`.

```
3a1e17600b7db702e54b8792dae946dbc3c1eecd14de4e6871f5bde91ab1fc1f  data.py
b3136a80558df15441b915568aca7199fbff878a392a2eed272f85fea9120a20  evaluate.py
1cfeb049192d81b8cb993e95e603234f5fbe5dd7829af45693f5e28ae39d5065  export.py
8ee2ed9298d434907cda9f055ef67282fd42ef73eae49c2cf0111c250cb33fa2  paths.py
b60a6118b8771b479a4356d380de9722a2a5d833dba5d2431911a5e18aa7d66b  quant.py
ae1d93618c271731620f5f0cfd1d6beb545072c4366cb107b4f57a6f3a5f782e  stream.py
2589c07d52e0ef64b46bd2c5304617b128b2bfd21d46f76f3f1dd26b547ac890  sweep.py
1df0d1a8346923b86d1925cdaa224bfb61b58028c89410e4dccbb8f63f289460  teacher.py
9668f573489cd31d25ec4e402eb8eb8fa95e1e6dd094da5628316a1dfc5fe195  testsets.py
d44996b1b2c313ac8baf04228a1b8873ff0df12c99da9708459191ea7cf3c853  train.py
9daf904a0912b1ab644f8ef517d12366c5d6a6c67c7a991f56e52c748d3ef8bd  whisper-ternary/quant.py
f1ced52d0b5b0a738cc7e3e494bae3b634ee0f0edb5f521e2974a900f5e41ae4  whisper-ternary/wer.py
```

## Pilot results

**Recipe stage (2026-10-01, learning rate 5e-4, 12,000 steps each, about
1,600 audio hours, scored on each run's final exported checkpoint).** Full
development WER:

| Arm | LS dev-clean | LS dev-other | AMI | VoxPopuli | YODAS | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 FP32 original | 1.59 | 2.94 | 12.12 | 5.75 | 5.47 | 5.58 |
| P1 sequence distillation only | 2.35 | 5.30 | 13.00 | 6.82 | 7.78 | 7.052 |
| P2 + encoder matching | 2.25 | 4.97 | 12.91 | 6.59 | 7.53 | 6.850 |
| P3 + trainable prediction/joint | 2.23 | 5.00 | 12.98 | 6.51 | 7.56 | 6.854 |

Encoder matching (P2 versus P1) helps by 0.20 points and on all five sets. Making
the prediction and joint networks trainable (P3 versus P2) changes the mean by
0.004 points, which is noise; P2 is selected by the preregistered rule and is
also the simpler recipe. All arms are far below the 16.7% stop threshold. Each
arm spent about 10% of its 100-step windows waiting on the stream (no errors or
retries), costing wall time only.

**Learning-rate stage (P2, 12,000 steps each).** Full development mean WER:
2e-4 7.11%, 5e-4 6.85%, 1e-3 7.04%. Selected: P2 at 5e-4 (`runs/pilot-selection.json`);
the stop rule was not triggered.

## Main run (M1), fixed before launch (2026-10-01)

- Recipe P2 (TDT loss on teacher transcripts plus encoder-output matching;
  prediction and joint networks frozen), peak learning rate 5e-4.
- **M1 step count: 250,000 steps** (about 58 GPU-hours at the measured 0.84 s
  per step, plus dev-subset evaluation every 12,500 steps, 20 points in all, and
  about 80 minutes of checkpoint writes). Warmup 2% (5,000 steps), ramp 25%
  (62,500 steps), linear decay to zero; 600 s batches with gradient
  checkpointing. The pre-launch estimate written here, "about 140,000 audio
  hours of exposure", was wrong: 250,000 steps of at most 600 s cannot exceed
  41,667 hours. The run recorded 33,248 hours (about 480 s per step after
  filtering), of which 22,040 were YODAS, against about 100,000 YODAS hours
  available, so YODAS was not repeated to any meaningful degree (corrected
  2026-10-04).
- Scored and exported on the final checkpoint, as in the pilot.
- Stream workers raised from 6 to 8 (`DEFAULT_WORKERS`; one constant changed in
  `stream.py` after the freeze): 6 workers left the GPU waiting on data in about
  10% of 100-step windows; 8 workers deliver 1,001 audio-hours per hour at 17.4
  GB (measured 2026-10-01), against about 580 consumed. Sampling distribution
  and filters are unchanged; only the partitioning, and hence the item order,
  differs from the pilot.
- Launch: `./heavy main --mem-max 36G --runtime 100h -- python sweep.py --phase main --max-steps 250000`.

## Main-run and test results (2026-10-04)

M1 (P2 recipe, 5e-4, 250,000 steps, 33,248 audio hours seen, 69.3 hours of
training) finished 2026-10-04 13:30; final checkpoint full development mean WER 6.09% (B0 5.58%).
Test sets were scored once per `plans/test-scoring.md` (Codex-reviewed clean in
four rounds; precheck all pass; outputs in `results/TEST.md` and `test.json`):

| Arm | LS clean | LS other | AMI | Earnings-22 | GigaSpeech | SPGISpeech | VoxPopuli | Mean of 7 | Common Voice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA published | 1.69 | 3.19 | 11.16 | 11.15 | 9.74 | 2.17 | 5.95 | 6.44 | n/a |
| B0 FP32 original | 1.70 | 3.19 | 11.15 | 11.24 | 9.78 | 2.14 | 5.94 | 6.45 | 8.50 |
| B1 ternary PTQ | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| M1 ternary QAT | 2.05 | 4.20 | 10.42 | 11.72 | 10.35 | 2.94 | 6.19 | 6.84 | 12.56 |
| M1 - B0 (points) | +0.36 | +1.01 | -0.72 | +0.48 | +0.57 | +0.81 | +0.25 | +0.39 | +4.06 |

M1's export is 180.8 MB against 2,472 MB for the original `.nemo` file.

**Sanity flag investigated (2026-10-04).** `report_test.py` flagged 1.43% empty
M1 hypotheses on AMI against the plan's 1% threshold. The threshold was
miscalibrated for AMI, not a model defect: the FP32 original leaves 3.12% of AMI
hypotheses empty (M1 1.43%), and M1 has fewer or equal empty outputs than B0 on
every set checked (Earnings-22 0.07 vs 0.73%, GigaSpeech 0.13 vs 0.35%, Common
Voice 0.14 vs 0.15%, LibriSpeech clean 0 vs 0). The 51 AMI utterances empty only
for M1 are mostly one-word backchannels (median 0.39 s) and account for 78 of
M1's 9,334 AMI errors. The flag is closed; no number changes.

Reading: ternary weights cost 0.39 WER points on the seven-set mean (6.84
versus 6.45, 1.06x), between -0.72 (AMI, better than FP32) and +1.01 (LS other)
points per set. Common Voice, which is outside the leaderboard mean and the
training mixture, shows the largest gap (+4.06). The optional FP32 control A1
was not run, so the reference is the original model, as the design specifies.

## Out of scope

Activation quantization, packed-kernel inference, streaming decoding, the
prediction and joint networks in ternary, TED-LIUM, and any energy or speed
claim.
