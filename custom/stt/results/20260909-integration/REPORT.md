# Initial transcription and gating integration — 2026-09-09

A local pretrained Whisper tiny.en reference transcribes real cached speech.
The custom 1,556,928-parameter ternary CTC prototype performs real CUDA training,
but its first five-minute run does not generalize to usable transcription quality.
This is an engineering prototype, not the planned 0.5–1B recognizer.

## Pretrained paired development comparison

24 unique development utterances are each tested clean and mixed with real noise
at 10dB SNR, with two seconds of noise before/after each utterance. Add 24
noise-only recordings: 72 composed scenes, each replayed three times per arm.
Repeated playback does not increase the number of independent examples.

| Arm | Complete-scene WER | Clean-scene WER | Noisy-scene WER | Audio sent to ASR |
|---|---:|---:|---:|---:|
| Always-on | 11.8% | 7.8% | 10.4% | 100% |
| VAD 008 | 92.2% | 80.4% | 104.0% | 10.4% |
| VAD 010 | 87.3% | 81.3% | 93.2% | 9.5% |

Always-on recognition inserted 46 words on noise-only recordings per playback;
both gates inserted zero there. This reduction does not compensate for the large
loss of speech. WER can exceed 100% when insertions are numerous; scores are not
clipped. Full-clip accuracy did not transfer to this one-second causal window
adapter. **Neither gate is acceptable for deployment.**

CPU package and iPhone energy were unavailable. Raw wall times are retained, but
the comparison overlapped CUDA training on the same host and is not an isolated
efficiency benchmark. Reduced audio processing is not measured power savings. A subsequent isolated
four-utterance check with the final evaluator reproduced the accuracy failure
(always-on 16.1%, gates 91.1%/92.1%) and rejected both gates under the explicit
utterance-preservation and WER/CER criteria. Energy was still unavailable.

## Custom training smoke check

Five-minute CUDA envelope; 56,767 AdamW updates over 898 training utterances.
CTC loss fell from 8.226 to 0.036. All 1,556,928 learned coefficients export as
ternary codes, using 389,232 packed bytes plus 40 scale bytes. Inference expands
these into FP32 weights; no packed-kernel efficiency is established.

On a four-utterance engineering subset, complete-scene WER was 124.7% and CER
77.8%. Gating removed most utterance audio and increased total CPU processing
time for this small custom model. Low training loss therefore does not establish
useful recognition. Preserve this result as an immature training baseline, not a
successful STT accuracy result. The final full-budget qualification used 24 speech utterances in two conditions
plus 24 noise-only recordings. Its always-on WER was 114.8% and CER 78.5%.
The controller correctly retained it as `infeasible_best`, not a quality-qualified
model. Pipeline readiness passed real CUDA training, independent safe-export
scoring and an actual bounded recipe proposal. The proposal reduced learning rate
from 0.001 to 0.0007. The proposed recipe also completed a separate 60-second CUDA training and
independent evaluation smoke check. That shortened trial is explicitly unranked,
was not compared as an equal-budget optimization gain, and did not promote a model.
No one-hour research service was started.

## Next experimental question

Calibrate or retrain the gate for continuous speech using separate calibration
recordings, then repeat the unchanged transcript comparison. In parallel, the
custom recipe search can improve decoded WER/CER from its diagnostic baseline.
Use the pretrained always-on model as the working transcription reference.

All audio is public-corpus material; source URLs, licenses, split policy and hashes
are in provenance.json and the prepared data manifest. No official final holdout
or medical/field benchmark was used. Learned weights stay local; this directory
contains evidence, source snapshots, metadata and hashes only.
