# Ternary-weight QAT of Whisper tiny.en: experimental design

Written 2026-09-29. Owner: experimental design and architecture are fixed here;
implementation follows this document. Deviations require updating this file first.

## Question

How much recognition accuracy does the pretrained Whisper tiny.en lose when every
attention and feed-forward projection weight is constrained to ternary codes
{-1, 0, +1} with one FP32 scale per output row, after quantization-aware
fine-tuning (QAT) from the pretrained checkpoint, compared with an identically
fine-tuned FP32 control?

This directly follows the conclusion in
`custom/binary_stt/VIBEVOICE_ASR_BITNET_REVIEW.md`: quantize a model that already
transcribes, rather than training a quantized model from random initialization.

## Arms

| Arm | Name | Weights | Training | Purpose |
| --- | --- | --- | --- | --- |
| A0 | `fp32-zeroshot` | Pretrained FP32 | None | Anchor to the published Whisper tiny.en LibriSpeech numbers |
| A1 | `fp32-finetune` | FP32 | Same data, steps, schedule as A2 | Control: separates LibriSpeech adaptation from quantization loss |
| A2-PTQ | `ternary-ptq` | Ternary projections, no training | None | Shows why QAT is needed |
| A2 | `ternary` | Ternary projections, FP everything else | QAT | Primary result |
| A3 | `ternary-embed` | A2 plus ternary tied token embedding / output projection | QAT | Secondary: where most of the byte reduction comes from at this model size |

The reported ternary result is the gap A2 minus A1 on test-clean and test-other,
alongside A0 and A2-PTQ for context. A3 is reported separately.

## Model and quantized set

Base checkpoint: `finetune/stt/models/whisper-tiny.en`, Hugging Face revision
`87c7102498dcde7456f24cfd30239ca606ed9063`, verified against its `lock.json`
hashes before every load. Config: d_model 384, 4 encoder and 4 decoder layers,
feed-forward 1536, vocabulary 51,864, tied embedding and output projection.

Quantized in A2 (Hugging Face module names):

- `model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj`
- `model.encoder.layers.{i}.fc1`, `.fc2`
- `model.decoder.layers.{i}.self_attn.{q,k,v,out}_proj`
- `model.decoder.layers.{i}.encoder_attn.{q,k,v,out}_proj`
- `model.decoder.layers.{i}.fc1`, `.fc2`

Additionally quantized in A3: `model.decoder.embed_tokens` and `proj_out`, which
share one latent parameter. The embedding lookup returns the dequantized row
(codes times that row's scale); the output projection uses the same dequantized
matrix. Scales are per vocabulary row.

Never quantized: `model.encoder.conv1`, `conv2`, `model.encoder.embed_positions`,
`model.decoder.embed_positions`, every layer norm, every bias. This matches the
VibeVoice-ASR-BitNet scope (projections only; embeddings, norms and biases in
higher precision), except that A3 deliberately tests the embedding.

Approximate parameter accounting for tiny.en (exact numbers are computed and
recorded by the code): projection weights ~16.5M, tied embedding ~19.9M,
convolutions ~0.5M, positions ~0.75M, norms and biases small. Total ~37.8M.
At this size the embedding dominates bytes, which is why A3 exists.

## Quantizer

For a weight matrix W with shape [out, in]:

```
s_i   = max(mean_j |W_ij|, 1e-8)             # one FP32 scale per output row
C_ij  = clamp(round(W_ij / s_i), -1, +1)     # ternary code, torch round-half-to-even
W_hat = C_ij * s_i                            # dequantized weight used in the forward pass
```

Training keeps FP32 latent W as the optimizer parameter and recomputes C and s on
every forward. The straight-through estimator is the identity with respect to
the latent weight, implemented as `W_hat + (W - W.detach())`: the forward
weight is bit-exactly W_hat (the second term is exactly zero), and the gradient
reaching W is the identity. No gradient flows to the scale. This is the BitNet
b1.58 absmean recipe with per-row rather than per-tensor scale. Per-row scales match the packed-kernel storage convention
already used in `custom/cpu-inference` and `custom/inference-efficiency`.

Quantization is active from the first training step; there is no ramp.
Activations are not quantized. Biases are FP32 and trainable.

## Data

Training: LibriSpeech train-clean-100, already extracted at
`/mnt/hd/wilderness-labs-stt/stt-distillation/datasets/libri/LibriSpeech/train-clean-100`
(28,539 utterances, ~100.6 h). Transcripts are lowercased; no other text edits.

Selection: LibriSpeech dev-clean (2,703 utterances). In-training checkpoint
selection uses a fixed deterministic 400-utterance subset (every k-th utterance
of the id-sorted list). The learning-rate choice per arm uses full dev-clean.

Reporting: LibriSpeech test-clean (2,620) and test-other (2,939), each scored
once per arm for the selected checkpoint only. No test-set value influences any
decision.

dev-clean, test-clean and test-other are downloaded from openslr.org resource 12
into `/mnt/hd/wilderness-labs-stt/whisper-ternary/data/`, with tarball SHA-256
and utterance counts recorded. All reads use the extracted FLAC directories.

Utterances longer than 30 s are truncated by the Whisper feature extractor. The
count of truncated utterances per split is recorded; the policy is identical
across arms.

## Training recipe

- Features: `WhisperFeatureExtractor` from the model directory, 80 log-mel bins,
  padded or truncated to 30 s, computed on the fly in data-loader workers.
- Labels: `WhisperTokenizer` from the model directory applied to the lowercased
  transcript, producing prefix tokens, text tokens and end-of-text. Following
  the standard Hugging Face collator, a leading decoder-start token is removed
  because the model prepends it when shifting labels. Padding uses -100.
- Loss: the model's built-in token cross-entropy.
- Optimizer: AdamW, betas (0.9, 0.98), eps 1e-6, weight decay 0.01, on FP32
  master parameters. Gradient clipping at global norm 1.0.
- Schedule: 200 linear warmup steps, then linear decay to zero at step 4,000.
  Indexing: update k (1-based) uses factor k/200 during warmup, so update 200
  is the first at peak, and (4000 - (k - 1)) / 3800 afterwards, so update 201
  is also at peak and update 4,000 uses 1/3800 of peak. The zero factor is
  reached after the final update. Identical for all arms.
- Batch: 32 utterances per step. 4,000 steps is about 4.5 epochs.
- Precision: BF16 autocast for matrix products; FP32 parameters, optimizer
  state and loss.
- Dropout 0 (the checkpoint default). No SpecAugment. Full fine-tuning; nothing
  frozen except the encoder's sinusoidal position table, which transformers
  loads with `requires_grad=False` because it is a constant in the original
  Whisper. It is counted among the residual parameters.
- Seed 20260929 for data order; all arms see the same order.
- Every 500 steps: greedy decode of the 400-utterance dev subset in FP32,
  normalized WER, keep the best checkpoint by that WER.

Learning-rate sweep, selected by full dev-clean WER of each run's best
checkpoint:

- A1: 1e-5, 3e-5, 1e-4
- A2: 5e-5, 1e-4, 3e-4
- A3: A2's selected learning rate only

Each run is expected to take well under an hour on the RTX 5090. The whole
sweep plus evaluation should fit in an afternoon.

## Evaluation

- Greedy decoding, `num_beams=1`, `do_sample=False`, `max_new_tokens=225`,
  the checkpoint's shipped generation config left unchanged (it carries the
  English-only prompt handling and token suppression), batch size 64, FP32 on
  GPU. The same decoding function scores the in-training dev subset and the
  final splits. Identical for all arms; exports store the shipped generation
  config so reconstructed models decode identically.
- Text normalization: the Whisper `EnglishTextNormalizer` initialized from the
  model directory's `normalizer.json`, applied to both hypothesis and reference.
- WER is corpus-level: total substitutions, deletions and insertions divided by
  total reference words over the split. Per-utterance hypotheses, references
  and edit counts are saved so any number can be recomputed.
- Every reported dev-clean and test number for a ternary arm is scored on the
  model reconstructed from the exported artifact, so it belongs to a deployable
  file. In-training checkpoint selection (the 400-utterance dev subset) scores
  the training graph in FP32 instead; its quantized layers already use exactly
  the exported codes and scales, so the only difference from the export is
  FP16 storage of the residual tensors. Per-utterance records of every
  in-training evaluation are kept so the selection can be audited.

## Export

One safetensors file per ternary run plus `manifest.json`:

- For each quantized layer: `codes` as uint8 [out, ceil(in / 4)], four 2-bit
  codes per byte (00 = 0, 01 = +1, 10 = -1, 11 unused); `scale` as FP32 [out];
  `bias` as FP32 [out] when present.
- Every non-quantized tensor stored in FP16.
- Manifest: checkpoint revision, arm, quantized layer names and shapes, training
  run id and config hash, byte accounting (packed code bytes, scale bytes,
  FP16 residual bytes, total file bytes) next to the FP32 and FP16 sizes of the
  original checkpoint, code histogram (fraction of -1 / 0 / +1), and the file's
  SHA-256.
- `load_export()` reconstructs a Hugging Face `WhisperForConditionalGeneration`
  in FP32. A check compares it with the training graph on a fixed batch: codes
  and scales must match exactly; logits may differ only by FP16 storage of the
  residual tensors (max absolute difference and argmax agreement are recorded).

## Artifacts and hygiene

All data, checkpoints, exports and run outputs live under
`/mnt/hd/wilderness-labs-stt/whisper-ternary/`. The Python wrapper refuses to
run if `/mnt/hd` is not mounted. Each run directory holds `config.json`,
`metrics.jsonl`, the best latent checkpoint, the export, the manifest and every
evaluation JSON. Source file hashes are recorded per run. No weights enter Git;
only code, this design, result tables and manifests without tensors.

## Reported outputs

1. Main table: A0, A1 (selected), A2-PTQ, A2 (selected), A3 on dev-clean,
   test-clean and test-other WER, with ternary parameter count, packed bytes
   and total artifact bytes.
2. Learning-rate sweep table on dev-clean for A1 and A2.
3. Code histogram and the export reconstruction check results.

## Expectations, stated before running

A2-PTQ is expected to be near-unusable. A2 after QAT is expected to land between
A1 and a few absolute WER points above A0 on test-clean, with a larger gap on
test-other. A3 is expected to cost additional accuracy. These are guesses and
carry no weight in analysis; the measured numbers do.

## Revision 2 (2026-09-29, written after the first sweep's ternary runs stalled)

Observed in the first sweep (recipe above, referred to as v1; full table in
`results/RESULTS.md`): ternary at 5e-5 and 1e-4 stalls at a training loss
near 5.4 (FP32 control: about 0.55) and decodes as audio-independent
language-model loops such as "the king of the king of ...", ending at 110.98%
and 95.34% dev-clean WER. Ternary at 3e-4 was still recovering when the
schedule ended (dev-subset WER 94.5 → 87.1 → 62.2 → 50.0% at steps 1,000 to
4,000; 50.29% full dev-clean, 50.46% test-clean). The v1 numbers are kept and
reported as the preregistered result. Revision 2 changes only the optimization
path from the FP32 checkpoint to the ternary model; the question, quantizer,
quantized set, evaluation and export are unchanged.

- **Progressive quantization.** The effective weight is
  `lerp(W, W_hat_ste, f)` with `f` rising linearly from 0 to 1 over the first
  25% of training steps, then held at 1. Export requires `f == 1`. The
  gradient to the latent weight remains the identity for every `f`.
- **Distillation from the frozen FP32 pretrained model.** Both teacher and
  student are teacher-forced on the ground-truth labels. Loss is
  `0.5 * CE(labels) + 0.5 * KL(teacher || student)` over label positions at
  temperature 1. The FP32 control uses the same loss so the comparison stays
  matched; the control has no quantization ramp because it has no quantizer.
- **Longer schedule.** 12,000 steps, 600 warmup steps, otherwise the same
  optimizer and linear decay; dev-subset evaluation every 1,000 steps.
- **More unique data if available.** train-clean-360 (104,014 utterances) is
  added to train-clean-100 when its download and manifest verify; otherwise
  train-clean-100 alone. The run's `config.json` records which.
- **Learning rates.** Control: 1e-5 and 3e-5. Ternary: 3e-4 and 1e-3. The v1
  grid of 5e-5 and 1e-4 was too low to move codes, and 3e-4 was budget-limited
  rather than stalled.
- v2 runs use the `v2-` run-name prefix and are reported in a separate table
  next to v1.

Outcome (2026-09-30, `results/v2-RESULTS.md`): control 4.50% / 12.37%
test-clean / test-other; ternary projections 13.12% / 30.90% (lr 1e-3
selected over 3e-4 on dev-clean); ternary projections plus embedding 12.12% /
28.42% in 12.2 MB. Both ternary arms chose their final or penultimate
checkpoint, so the schedule was not over-long. See "Secondary analyses" for
the runaway-continuation diagnosis.

## Secondary analyses (post hoc, always labelled as such)

Added 2026-09-30 after inspecting the v2 ternary runs. The primary metric stays
the uncapped corpus WER above; nothing here changes it or the selection rule.

Observation: later ternary checkpoints transcribe most utterances correctly and
then keep generating fluent text after the speech ends instead of emitting
end-of-text. At step 9,000 of the v2 3e-4 run, 6 of 400 dev-subset utterances
held 815 of 941 insertions; without them the subset WER was 10.7% against 21.7%
primary, and the without-runaway number fell monotonically across checkpoints
while the primary swung by up to 7 points depending on how many utterances ran
away. The FP32 arms show no such utterances.

`analysis.py` computes, from the saved per-utterance records only:

- **Runaway utterances:** insertions >= 20, or hypothesis words > 1.5 x
  reference words + 10. Counted separately and as a union. For weak models
  (v1 ternary) the insertion rule also catches long garbled output, so the
  count is read together with the worst-utterance listing.
- **WER excluding runaway utterances** and **WER with runaway hypotheses capped**
  to reference length + 5 words. Both use the reference to decide, so they are
  diagnostics of where the errors are, not deployable metrics.
- **Duration-capped WER (deployable rule):** every hypothesis, for every arm, is
  truncated to `ceil(4.5 x audio seconds) + 5` words before scoring. This uses
  only the audio duration, so a runtime could apply it. The constant was chosen
  from the reference speaking-rate distribution before any ternary number was
  looked at under it: over the 8,262 dev-clean, test-clean and test-other
  utterances the rule truncates 0 references and 0 FP32 hypotheses (maximum
  observed reference rate 5.5 words/s on a short utterance, covered by the +5).
  Because greedy decoding is deterministic, truncating the output is equivalent
  to having stopped decoding at that length.

Secondary tables are written next to the primary ones as
`results/<prefix>SECONDARY.md` and carry this label in their heading.

## Out of scope for this step

Activation quantization, knowledge distillation from the FP32 teacher,
train-clean-360 or larger data, running on the packed popcount kernels
(which need packed activations), streaming, and any energy measurement.
