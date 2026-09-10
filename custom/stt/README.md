# Transcription and gated-STT autoresearch

This track trains a real character-CTC speech recognizer on local transcript-bearing
audio and compares always-on recognition with both archived VAD candidates (008
and 010). The custom graph is a small causal convolutional ternary prototype.
It is an engineering step toward the planned 0.5–1B model, not that final model.
A separate local Whisper reference under `finetune/stt` provides immediately
usable transcription while custom training is developed. The [initial integration
report](results/20260909-integration/REPORT.md) records measured results and limitations:
pretrained always-on WER 11.8%, versus 92.2%/87.3% with the uncalibrated streaming
gates. The custom baseline remains `infeasible_best`; pipeline readiness is not
an accuracy acceptance claim.

## Prepare and qualify

Use the existing pinned Nix/CUDA environment and real audio cache documented in
`custom/autoresearch/runtime_data.md`. All preparation and qualification occur
outside the research timer:

```sh
custom/autoresearch/runtime-python custom/stt/prepare_gates.py
custom/autoresearch/runtime-python custom/stt/data.py prepare
custom/autoresearch/runtime-python custom/stt/tests.py
custom/stt/control preflight --train-seconds 300
custom/stt/control status
```

Gate preparation requires the original local VAD run (or `--run /path/to/run`).
Weights are deliberately excluded from Git; a fresh checkout must obtain the
original artifacts through local storage before qualification. All recognizer
weights, dependency wheels and audio caches are also ignored.

Preparation hashes audio and transcripts, preserves the established 18/5/5
train/calibration/development speaker split and noise-source separation, and
asserts CTC alignment feasibility including repeated adjacent characters. CTC
uses real transcript targets with `zero_infinity=False`. The fixed 80-bin
log-mel frontend is causal; the graph subsamples 2x, uses bias-free ternary
convolutions and non-affine normalization, and decodes 29 symbols greedily.
All learned weights export as ternary codes plus per-tensor scales. Inference
expands these codes to FP32; packed storage does not imply packed computation.

Qualification runs real CUDA optimizer work, independently reconstructs the
exported model without pickle or candidate training imports, evaluates decoded
speech, and exercises a bounded proposal. Blank or constant recognizers cannot qualify. The provisional quality limits
are WER <95% and CER <80%, **not usable field accuracy targets**. A diverse,
nonempty decoder with WER <150% and CER <95% may instead be retained as
`infeasible_best` to support research from an immature baseline. Readiness then
means the training/scoring machinery works, while `quality_qualified` stays false.
The pretrained reference is the working recognizer in that case. A
successful check does not start or schedule a research session.

## Run research explicitly

```sh
custom/stt/control start --authorize-one-hour
custom/stt/control status
custom/stt/control stop
```

The service has a hard 3,600-second systemd descendant-process limit, a 90-second
reporting reserve, 300-second training envelopes, 180-second evaluation limits,
and 120-second proposal limits. Equal-budget trials start from the same seed.
The controller snapshots and hashes graph, trainer, evaluator, gate exports,
recipe, cache lock and runtime contract. A read-only Codex proposal returns only
a bounded JSON recipe and hypothesis; the supervisor validates it and executes
training/evaluation independently. Scoring and data are never proposal-editable.
Remote proposal reasoning is separate from entirely local speech inference.
No experiment-specific remote model override is used.

WER improvement is the primary promotion criterion, with CER breaking exact
WER ties. This explicitly rewards fewer missed words, avoiding the old VAD
selection-rule blind spot. Gating comparisons are reported separately so a gate
that suppresses speech cannot hide behind reduced compute. Recoverable failed
trials do not end the search; three consecutive execution/proposal failures stop
it with an error record. Full logs, telemetry, snapshots, exports and results
live under ignored `runs/`. Candidate exports are not automatically merged.

## Transcribe and compare

```sh
custom/autoresearch/runtime-python custom/stt/transcribe.py \
  --candidate custom/stt/runs/RUN/trials/000 --audio /path/to/16khz.wav
custom/autoresearch/runtime-python custom/stt/evaluate.py \
  --candidate custom/stt/runs/RUN/trials/000 \
  --output custom/stt/runs/RUN/paired.json --limit 24 --repeats 3
```

Use `--limit 0` for all development speech/noise records. A nonzero limit selects
records spread deterministically across the source list, rather than only the
first speaker. The default research subset has 24 utterances, each evaluated
clean and mixed with real noise at 10dB SNR, plus 24 noise-only recordings.
Every speech scene has two seconds of noise before and after the utterance.
These are composed playback scenes, not fresh field recordings or frame-labeled
VAD validation. Dataset license/source records are copied into the cache manifest.
Development is reused during optimization; official final holdouts and Eka are
not downloaded, trained on, or consulted by this pipeline.

Both arms use the same fixed maximum 20-second ASR chunks; gated segments restart
recognition. The gate examines trailing one-second windows every 200ms and uses
500ms pre-roll and 800ms hangover. Decisions never see future samples, and
merged intervals prevent duplicate transcription of overlapping pre-roll.
The old full-clip thresholds are frozen for this first integration test. Their
behavior on short windows is unvalidated: poor results require calibration on
separate continuous development recordings, not tuning on the scoring set.

Reports include complete-scene WER/CER, clean/noisy breakdowns, noise-word
insertions, utterance audio omitted, ASR audio fraction, and total measured
processing time including the gate. Omitted utterance audio includes internal
silence and is not a frame-level missed-speech metric. Gating passes its provisional integration check only if clean/noisy WER and
CER each worsen by at most 0.5 percentage points and at least 99% of utterance
audio is retained. This is a conservative utterance-preservation proxy, not
frame speech recall. Repeated arms rotate execution order. CPU package joules are reported only when accessible hardware
counters exist; unavailable energy stays null. GPU telemetry during training
is not inference energy. Neither skipped audio nor faster processing establishes
phone power savings. Real-time microphone/capture, idle and wake costs, sustained
phone energy, medical terms/numbers/negation and untouched holdout accuracy remain
separate acceptance work.

Technical references: [PyTorch CTC loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html)
and the original [CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf).
