# Ternary-weight QAT of Whisper tiny.en

Measures how much LibriSpeech accuracy Whisper tiny.en loses when every attention
and feed-forward projection is constrained to ternary codes {-1, 0, +1} with one
FP32 scale per output row, after quantization-aware fine-tuning, against an
identically fine-tuned FP32 control. The arms, quantized set, quantizer, data,
training recipe, evaluation protocol and export format are fixed in
[DESIGN.md](DESIGN.md); the code implements that document and nothing more.
Ternary numbers are always measured on the model rebuilt from the exported file.
Nothing here measures speed or energy.

All Python runs through `finetune/whisper-ternary/python` (pinned runtime,
offline, refuses to run without `/mnt/hd`). Data, checkpoints, exports and
evaluation JSON live under `/mnt/hd/wilderness-labs-stt/whisper-ternary/`; no
weights enter Git. Commands below run from the repository root with:

```sh
W=finetune/whisper-ternary
R=/mnt/hd/wilderness-labs-stt/whisper-ternary/runs
```

## Tests

```sh
$W/python -m unittest discover -s $W/tests -v
```

## Smoke (minutes; not results)

```sh
$W/python $W/evaluate.py --source pretrained --split dev-clean --limit 64 --out $R/smoke/eval-pretrained-dev64.json
$W/python $W/train.py --arm ternary --lr 1e-4 --run-name smoke-ternary --max-steps 30 --eval-every 15 \
  --dev-subset 16 --batch-size 8 --workers 4 --limit-train 256 --overwrite
$W/python $W/train.py --arm fp32 --lr 3e-5 --run-name smoke-fp32 --max-steps 20 --eval-every 10 \
  --dev-subset 16 --batch-size 8 --workers 4 --limit-train 256 --overwrite
```

Revision 2 smoke (ramp over the first 12 of 24 steps, distillation on):

```sh
$W/python $W/train.py --arm ternary --lr 3e-4 --run-name v2smoke-ternary --max-steps 24 --warmup 4 \
  --eval-every 12 --dev-subset 8 --batch-size 4 --workers 2 --limit-train 64 \
  --quant-ramp-fraction 0.5 --distill-weight 0.5 --overwrite
```

## Experiment

`$W/python $W/sweep.py` runs every step below in order and skips finished ones.
The individual commands are:

```sh
# A0 zero-shot and A2/A3 post-training quantization (no training), dev-clean
$W/python $W/evaluate.py --source pretrained --split dev-clean --out $R/fp32-zeroshot/eval-dev-clean.json
$W/python $W/evaluate.py --source pretrained --ptq ternary --split dev-clean --out $R/ternary-ptq/eval-dev-clean.json
$W/python $W/evaluate.py --source pretrained --ptq ternary-embed --split dev-clean --out $R/ternary-embed-ptq/eval-dev-clean.json

# A1 learning-rate sweep (FP32 control)
for lr in 1e-5 3e-5 1e-4; do $W/python $W/train.py --arm fp32 --lr $lr --run-name fp32-lr$lr; done
# A2 learning-rate sweep (ternary projections)
for lr in 5e-5 1e-4 3e-4; do $W/python $W/train.py --arm ternary --lr $lr --run-name ternary-lr$lr; done
# Select per arm by the lowest dev_clean_wer in $R/<run>/summary.json, then A3 at A2's rate:
$W/python $W/train.py --arm ternary-embed --lr $A2_LR --run-name ternary-embed-lr$A2_LR

# Final evaluation, once per reported arm and test split (never used for selection)
for split in test-clean test-other; do
  $W/python $W/evaluate.py --source pretrained --split $split --out $R/final/A0-fp32-zeroshot-$split.json
  $W/python $W/evaluate.py --source hf-dir --path $R/fp32-lr$A1_LR/best-hf --split $split --out $R/final/A1-fp32-finetune-$split.json
  $W/python $W/evaluate.py --source export --path $R/ternary-ptq/export --split $split --out $R/final/A2-ternary-ptq-$split.json
  $W/python $W/evaluate.py --source export --path $R/ternary-lr$A2_LR/export --split $split --out $R/final/A2-ternary-$split.json
  $W/python $W/evaluate.py --source export --path $R/ternary-embed-lr$A2_LR/export --split $split --out $R/final/A3-ternary-embed-$split.json
done
```

Each training run writes `config.json`, `metrics.jsonl`, `best.pt`, `last.pt`, `dev-subset/step-NNNNN.json`,
`eval-dev-clean.json` and `summary.json`, plus `best-hf/` (FP32) or `export/`
with `manifest.json` and `reconstruction.json` (ternary). Each evaluation JSON
keeps per-utterance references, hypotheses and edit counts. Transformers prints
deprecation and duplicate-suppression warnings about `forced_decoder_ids` and
`suppress_tokens` during generation; the shipped generation config is still the
one applied, and reloaded models decode identically to the in-memory ones.

## Power logging

The training machine has tripped the room's breaker twice. Every non-dry
`sweep.py` run therefore starts the `powerlog.py` flight recorder as a detached
process and sends it SIGTERM when the sweep ends, fails or is interrupted. The
recorder also stops by itself if the sweep process disappears. If it cannot
start, the sweep logs a warning and carries on. `--no-powerlog` turns it off. It
is a safety record for the training box, not an energy measurement of the models.

Logs go to `$R/power/sweep-<protocol>-<UTC start>.jsonl`, with the recorder's
own output in `$R/power/sweep-<protocol>.log`. Each record is one JSON line,
flushed and fsynced, so the file survives a hard power cut:

- a `header`;
- one `sample` per second: GPU power, clocks, temperature and throttle flags;
  CPU package power, busy fraction, frequency, Tctl and load; wall power;
- a `warn` when wall power is above `--warn-wall-watts` (default 700 W), at most
  once every 30 s;
- an `error` once per distinct sensor failure;
- an `end` with the stop reason.

`train.py` writes `started_utc` to `config.json`, and `started_utc` and
`finished_utc` to `summary.json`, so runs can be lined up with the log.

```sh
$W/python $W/powerlog.py summary $R/power/sweep-v2-<UTC>.jsonl   # means, maxima, last 30 s
$W/python $W/powerlog.py status                                  # one live sample as JSON
$W/python $W/powerlog.py record --name manual --stop-file /tmp/stop-power   # standalone
```

After a trip, `summary` on the newest file shows `end: NONE` and the samples
from the last 30 s before the power went. A line torn by the power cut is
skipped.

`est_wall_w = (gpu_w + cpu_w + 90) / 0.90` (`--baseline-watts`,
`--psu-efficiency`) is an estimate. `cpu_w` is the RAPL CPU package power when
it can be read, otherwise the proxy `cpu_busy * 170 + 30` W.
`est_wall_w_basis` says which one was used. On this machine RAPL is root-only,
which has been the kernel default since the PLATYPUS side channel
(CVE-2020-8694). To make it readable until the next reboot:

```sh
sudo chmod 444 /sys/class/powercap/intel-rapl:0/energy_uj
```

To make it persistent on NixOS, add this to `configuration.nix`:

```nix
systemd.tmpfiles.rules = [ "z /sys/class/powercap/intel-rapl:0/energy_uj 0444 root root -" ];
```

After a reboot, check the mode with `ls -l`. If it is still `-r--------`, the
driver loaded after tmpfiles ran. A running recorder checks again every 60 s
and switches to RAPL without a restart. hwmon `amdgpu` `PPT` is the integrated
GPU's power, not the CPU socket's, so do not use it.

`--wall-command CMD` replaces the estimate with a measurement from a metering
smart plug. `CMD` runs through the shell at every sample, with a timeout of
max(2 s, interval), and must print one number in watts. A failure logs `null`
and one `error` record. When `wall_w` is present, warnings use it. `sweep.py`
does not pass this option. To use a meter during a sweep, run the sweep with
`--no-powerlog` and start the recorder yourself. For example, for a Tasmota
plug (not tested here):

```sh
$W/python $W/powerlog.py record --name sweep-v2 --wall-command \
  'curl -s --max-time 1 "http://plug.lan/cm?cmnd=Status%208" | sed -n "s/.*\"Power\":\([0-9.]*\).*/\1/p"'
```

## Revision 2 options

DESIGN.md "Revision 2" changes only the optimization path from the FP32
checkpoint to the ternary model. `train.py` has four options for it. All are off
by default, so a run without them reproduces the v1 recipe exactly (same loss,
same data order, same checkpoint rule; the log only gains the new keys).

| Flag | Default | Effect |
| --- | --- | --- |
| `--quant-ramp-fraction F` | `0` | Ternary arms only; ignored for `fp32`. `ramp_steps = round(F * max_steps)`. Before update k (1-based), every ternary module's weight fraction is set to `f = min(1, k / ramp_steps)`, so the forward weight is `lerp(W, W_hat, f)`; after the ramp f stays 1. Dev-subset evaluations score the model at its current f, but only evaluations at `f == 1` can become `best.pt`. The export and the final dev-clean score always use `f = 1`. |
| `--distill-weight W` | `0` | Loss `(1 - W) * CE + W * KD`. KD is `T^2 * KL(teacher || student)`, averaged over label positions. The teacher is a second, frozen FP32 copy of the pinned pretrained checkpoint (never quantized). It runs under `no_grad` with the same BF16 autocast, features and labels as the student. Applies to every arm, the FP32 control included. |
| `--distill-temperature T` | `1` | Softmax temperature of the KD term. |
| `--train-splits a,b` | `train-clean-100` | Concatenates the count-verified manifests in the given order, each sorted by id (`data.build_training_manifest`). Duplicates and any `dev-*` / `test-*` split are refused. The seeded per-epoch permutation runs over the whole concatenated list. |

The v2 recipe from DESIGN.md: 12,000 steps, 600 warmup steps, dev-subset
evaluation every 1,000 steps, a ramp over the first 25% of steps, and a 0.5
CE / 0.5 KL mix at temperature 1. It trains on train-clean-100 plus
train-clean-360 once that split's download and manifest verify, otherwise on
train-clean-100 alone. Run names take the `v2-` prefix. Control learning rates
are 1e-5 and 3e-5; the ternary rates are chosen after the v1 3e-4 result:

```sh
SPLITS=train-clean-100,train-clean-360   # or train-clean-100 if train-clean-360 is not verified
for lr in 1e-5 3e-5; do
  $W/python $W/train.py --arm fp32 --lr $lr --run-name v2-fp32-lr$lr --max-steps 12000 --warmup 600 \
    --eval-every 1000 --distill-weight 0.5 --distill-temperature 1.0 --train-splits $SPLITS
done
$W/python $W/train.py --arm ternary --lr $LR --run-name v2-ternary-lr$LR --max-steps 12000 --warmup 600 \
  --eval-every 1000 --quant-ramp-fraction 0.25 --distill-weight 0.5 --distill-temperature 1.0 \
  --train-splits $SPLITS
```

Records a run keeps for Revision 2:

- `metrics.jsonl`: training lines add `ce` and `kd` (window means; `kd` is null
  without a teacher) and `weight_fraction` (null for `fp32`). Evaluation lines
  add `weight_fraction` and `selectable`.
- `dev-subset/step-NNNNN.json`: carries `weight_fraction` and `selectable`.
  Evaluations taken during the ramp are kept for the record even though they
  cannot be selected.
- `config.json` and `summary.json`: record `train_splits`,
  `train_split_utterances`, `quant_ramp_fraction`, `ramp_steps` (null for
  `fp32`), `distill_weight`, `distill_temperature` and `teacher` (model
  directory, revision and lock hash; null without distillation).
  `summary.json` also records `best_weight_fraction`, and its
  `train_utterances` is the total across all splits.

## Results

Generated tables live in `results/` (`RESULTS.md` for protocol v1, `v2-RESULTS.md`
for Revision 2 once it completes), with the export manifests of the reported
ternary artifacts next to them. `sweep.py --protocol v1|v2` produces them.

**Protocol v1 (preregistered recipe, 4,000 steps, train-clean-100), 2026-09-30.**
Test-set WER with the Whisper English normalizer, each arm scored once:

| Arm | test-clean | test-other | Artifact |
| --- | ---: | ---: | ---: |
| A0 FP32 zero-shot | 5.66% | 14.54% | 151.0 MB FP32 |
| A1 FP32 fine-tuned control (lr 1e-5) | 5.41% | 12.95% | 151.0 MB FP32 |
| A2-PTQ ternary projections, no training | 100.00% | 100.00% | 46.8 MB |
| A2 ternary projections, QAT (lr 3e-4) | 50.46% | 71.86% | 46.8 MB |
| A3 ternary projections + embedding, QAT (lr 3e-4) | 76.31% | 87.58% | 12.2 MB |

Reading: the evaluation pipeline reproduces the published tiny.en zero-shot
number; the FP32 control gains from LibriSpeech adaptation; ternary weights
without training destroy the model; and quantization-aware fine-tuning with
cross-entropy alone recovers only partially within 4,000 steps. The two lower
ternary learning rates (5e-5, 1e-4) stalled in audio-independent language-model
loops; 3e-4 was still improving when the schedule ended. This is the negative
result that motivated Revision 2 in `DESIGN.md` (progressive quantization,
distillation from the FP32 teacher, 12,000 steps, more data).

**Protocol v2 (Revision 2: 25% quantization ramp, 0.5 distillation from the
FP32 teacher, 12,000 steps, train-clean-100 + train-clean-360 = 464 h),
2026-09-30.** Same evaluation, each arm scored once on test:

| Arm | test-clean | test-other | Artifact |
| --- | ---: | ---: | ---: |
| A0 FP32 zero-shot | 5.66% | 14.54% | 151.0 MB FP32 |
| A1 FP32 fine-tuned control (lr 3e-5) | 4.50% | 12.37% | 151.0 MB FP32 |
| A2-PTQ ternary projections, no training | 100.00% | 100.00% | 46.8 MB |
| A2 ternary projections, QAT (lr 1e-3) | 13.12% | 30.90% | 46.8 MB |
| A3 ternary projections + tied embedding, QAT (lr 1e-3) | 12.12% | 28.42% | 12.2 MB |

Reading: the revised optimization path takes the ternary model from 50% to
12-13% test-clean WER, 2.7x the matched FP32 control's error on clean speech
and 2.3x on noisy speech, in a file 12x smaller than the FP32 original. The
tied embedding ternarizes essentially for free (A3 is within noise of A2 on
every split), so the 12.2 MB artifact is the one to show. Learning rate
mattered most: 1e-3 beat 3e-4 by 1.4 points on dev-clean, and both v1 rates
below that never recovered. `results/v2-SECONDARY.md` shows that runaway
continuations after the end of speech account for under half a point on the
full test sets, so the remaining gap is word accuracy, not decoding failure.
`sweep.py --protocol v2` reproduces the whole table; the power flight recorder
logged a mean GPU draw of 387 W (at its 400 W cap 92% of the time) and an
estimated 585 W at the wall for the 97-minute sweep.

## Transcribe a file (demo)

`transcribe.py` loads an export (SHA-256 checked against its manifest),
rebuilds the model from the ternary codes and scales, and decodes exactly like
the evaluation: greedy, shipped generation config, 225 tokens per 30 s window,
FP32. Input must be 16 kHz; channels are averaged. The default export is the v2
ternary projections + embedding run (12.2 MB).

```sh
$W/python $W/transcribe.py clip.wav                 # ternary model
$W/python $W/transcribe.py clip.wav --compare-fp32  # plus the pretrained FP32 tiny.en
$W/python $W/transcribe.py clip.wav --export DIR --duration-cap --device cuda
ffmpeg -i in.m4a -ar 16000 -ac 1 clip.wav           # convert other formats first
```

Audio over 30 s is split into consecutive 30 s windows without overlap, so a
word cut at a boundary can be misrecognized. `--duration-cap` applies the
secondary ceil(4.5 x seconds) + 5 word cap, counted on the printed words rather
than the normalized ones `analysis.py` counts; it is off by default so output
matches the primary metric. The model runs on dequantized FP32 weights: this
shows accuracy and file size, not packed-kernel speed. The ternary model was
fine-tuned on lowercase, unpunctuated LibriSpeech text, so its output is
mostly lowercase with little punctuation, unlike the pretrained FP32 model.

## Error analysis (secondary)

`analysis.py` computes the post hoc analyses in DESIGN.md "Secondary analyses"
from saved evaluation JSON. It never rewrites that JSON, and the primary metric
stays the uncapped corpus WER, copied unchanged. An utterance is "runaway"
(correct transcript, then continued generation) if `I >= 20` or its hypothesis
has more than `1.5 * ref words + 10` words; each criterion is also counted
alone. All secondary WERs are recounted with `wer.edit_counts`: without runaway
utterances; with runaway hypotheses cut to `ref words + 5`; and duration-capped,
the deployable rule, with every hypothesis cut to `ceil(4.5 * audio s) + 5` words
(durations from the split manifest). It also prints how many references that cap
would cut, which must be 0. Usage:
`$W/python $W/analysis.py FILE... [--json OUT] [--markdown OUT.md]` and
`$W/python $W/analysis.py trajectory $R/<run>`.
