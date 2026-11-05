# Kea-B 0.1: training

Back to the [model card](README.md). Results are in [EVALUATION.md](EVALUATION.md).

## Lineage

[{{base_id}}](https://huggingface.co/{{base_id}}) (NVIDIA, FP32) → [Kea-T](https://huggingface.co/{{kea_t}})
(ternary encoder) → [Kea-B 0.0](https://huggingface.co/{{kea_b0}}) (1-bit encoder) → Kea-B 0.1, which continues
Kea-B 0.0's training.

## Teacher package

- Labels: for the recorded corpora, transcripts by an open-weight speech model (see the tools below), with
  punctuation carried over from the original model's transcripts; for Common Voice, its read sentences; for the
  synthetic speech, the sentences it was synthesized from. The first two went through a fixed label-cleaning pass
  (casing, spelling variants, acronyms, spelled letters, dose units; for the teacher's transcripts also
  same-sound spellings chosen by an open-weight LLM).
- Encoder teacher: B0′, an FP32 fine-tune of the original model on the same labels (not released); Kea-B 0.1
  matched its encoder outputs.
- Prediction and joint networks: B0′'s, converted once to INT8 and kept frozen.

## Recipe

- Start: Kea-B 0.0's trained weights (264 binary encoder matrices, sign times a per-row mean |W| scale).
- {{steps}} steps of up to {{batch_seconds}} s of audio each, {{audio_hours}} hours of audio in total; peak
  learning rate {{lr}} (warm-up {{warmup}} steps, constant, then a {{cooldown}}-step linear cool-down); seed
  {{seed}}; one RTX 5090.
- Activation-aware training: the input of every binary matrix goes through the i8x1 kernels' per-frame INT8
  quantizer (straight-through gradient), so the trained model is the model the kernels run.
- 8 bits: the prediction network, the joint and the encoder's `pre_encode.out` in INT8 per output row
  (`pre_encode.out` trained through its INT8 weights); binary row scales rounded to FP16 in training and stored in
  FP16.
- The exported bundle is the trained model's forward weights exactly; the `compat/` view is derived from the
  bundle bit for bit.

## Data

Every source the run trained on, with the audio it saw (repeats included):

{{training_table}}

The synthetic technical speech (chemistry, mathematics, finance, software engineering) was generated for this
project:

{{synthetic_parts}}

Held-out data: development and test material was excluded from training (a global denylist of held-out
recordings and texts, and per-source checks).

## Tools

{{notice_tools}}

Encoder teacher: SHA-256 of B0′ `{{teacher_sha256}}`.
