# Kea-B 0.1: evaluation

Back to the [model card](README.md). Training is described in [TRAINING.md](TRAINING.md).

## Test sets

Word error rate (%) on every utterance of the public Open ASR Leaderboard test bundle
(`hf-audio/esb-datasets-test-only-sorted`), greedy decoding, references and outputs through the Whisper English
text normalizer. The mean is over the seven sets NVIDIA reports that are in the public bundle; Common Voice is
reported apart.

{{eval_table}}

- **Kea-B 0.1, FP32 rebuild (K-fp)**: the weights rebuilt in FP32 exactly as `load_kea_b.py` rebuilds them, FP32
  activations, batches of 64 utterances. This is the procedure Kea-B 0.0, Kea-T and the original were scored
  with, so it is the like-for-like row.
- **Kea-B 0.1, i8x1 kernels (K-x1)**: the deployed mode the model was trained for: our 1-bit GPU kernels with
  per-frame INT8 activations, one utterance at a time. Every set is within {{x1_max_set_diff}} points of K-fp.
- **Original, FP32 (our scoring)**: `{{base_id}}` scored with the same code. NVIDIA's published numbers are shown
  for reference.
- Each model was scored once on the test sets; nothing was selected on them.

Change against Kea-B 0.0:

{{diff_table}}

Common Voice is not like for like: Kea-B 0.1 trained on Common Voice audio (clips whose speaker or sentence
appears in a Common Voice test or dev split were removed), and Kea-B 0.0 and Kea-T did not.

Empty outputs (fraction of scored utterances):

{{empty_table}}

## Development sets and the training gate

Kea-B 0.1 was accepted by a preregistered gate on development sets (deployed mode, i8x1 kernels), against a
control trained the same way on the original model's labels and with the original model as teacher. On the
four-set development mean the control minus Kea-B 0.1 is {{gate_delta}} points (90% paired cluster-bootstrap
interval {{gate_ci}}):

{{gate_table}}

These development sets were also used for earlier decisions, so the gate is a development decision, not a
confirmatory result.

Checks of the scoring paths before test scoring: the deployed path reproduced the training run's recorded
development WER exactly (mean {{dev_x1_mean}}% against {{dev_x1_recorded}}%, token sequences {{dev_token_identity}}
identical), and the FP32 path reproduced Kea-B 0.0's recorded development WER exactly (mean {{dev_mb1_mean}}%).
Kea-B 0.1's FP32-path development WERs: {{dev_fp_sets}}.

## Known weaknesses

- Spelled-out letters.
- SPGISpeech is {{spgi_delta}} points worse than Kea-B 0.0 (verbatim style: fillers and false starts that the
  references clean up).
- EBITDA and EBIT.

## Method notes

- Decoding: preprocessor, encoder, then NeMo's greedy TDT decoder (durations 0-4, at most 10 symbols per frame),
  strict FP32 (autocast off, TF32 off), dither 0, on CUDA with NeMo's `no_while_loops` CUDA-graph decoder.
- Environment: NeMo {{nemo_version}}, torch {{torch_version}}, one RTX 5090.
- No intervals are reported for the test sets.
