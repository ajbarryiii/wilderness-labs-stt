# Relabeling the training pool with a stronger teacher

Work in progress. This directory relabels the student training pool with a stronger open-weight teacher
(T* = `ibm-granite/granite-speech-5.0-2b`, picked by a no-training diagnostic). The new labels are used to
fine-tune the published students (Kea-B, `../parakeet-binary`; Kea-T, `../parakeet-ternary`) and an FP32
Parakeet baseline (B0′). No model in this directory has been released yet.

## Stages

| Stage | Code | What it does |
| --- | --- | --- |
| 1, diagnostic | `score_teacher.py`, `analyze.py`, `teachers.py`, `launch.sh` | Scores candidate teachers on the dev sets and picks T* by a preregistered rule. |
| 2 Part A | `devsets/`, `score_domain.py`, `terms/` | Builds held-out medical and technical dev sets (evaluation only) and measures critical-term recall. |
| 2 Part B | `pool/`, `corpora/` | Builds text-only domain subsets of the streamed pool, new corpora and a global denylist of held-out ids. |
| 2 Part C | `labeling/` | C1 (CPU) replays the pinned training stream once into local FLAC shards plus a journal. C2 (GPU) labels the shards with T* (vLLM, batch-invariant) and B0, keeping per-token confidence. |
| 2a | `training/cleaning.py`, `training/cleaning_run.py`, `terms/cleaning/` | Label cleaning, read at training time: fixed casing, spelling-variant and dialect forms (layer 1; from cleaning version 2 also spelled-out acronyms, "A. P. I." to "API"; version 3, for stages 4-5, keeps the letters of a deliberate spelling as hyphenated capitals, "F-I-T-Z", compacts only acronyms, and abbreviates dose units after a digit, "10 mg"), and same-sound spellings chosen by an open-weight LLM from option-letter probabilities (layer 2, precomputed into a hashed artifact), with a net-harm check against human references. On by default for every arm. |
| 2b | `train_decoder.py`, `training/` | Decoder-only fine-tuning of Kea-B and Kea-T on the new labels, from an offline batch plan (`training/batchplan.py`). Every arm rejects the utterances whose T* label is a hallucinated repetition (a run of 32 or more repeated token n-grams, `training/repetition.py`; the B0-label controls drop the same ones), and a non-finite loss or gradient norm skips the optimizer step (at most 3 per run). |
| 2c | `synthetic/`, `terms/synthetic/`, `train_synthetic.py`, `synthetic_report.py` | Synthetic technical speech (chemistry, mathematics, finance, software engineering): open-weight text generation, then TTS and augmentation; four more decoder-only arms that mix the clips into the batch plan (augmented and unaugmented) at each student's selected stage 2b learning rate. |
| 3 | `train_b0prime.py`, `b0prime_report.py`, `style_metrics.py`, `training/stage3.py`, `training/label_sources.py`, `training/label_index.py` | B0′: the FP32 Parakeet v2 fully fine-tuned on the new labels, through the fused joint+TDT loss seam (`../tdt-fused`) (cleaning version 2, which also carries B0's punctuation into T*'s labels), with extra clip sources mixed into every pool of the batch plan (`training/batchplan.py` `compute_sources`: the stage 2c synthetic clips, and a read-speech store drawn exactly once). A learning-rate pilot and a WSD-against-linear sanity pair, then one constant-rate trunk saved at fixed branch points and three linear cooldown branches; the longest branch is B0′. Labels come from one provider per clip source, with the cleaning version, a read-time reject list and any transform in the run's label hash. The report adds style-neutral and formatted (case and punctuation) metrics. |
| 3, extra sources | `cv27/`, `pool/tedreject.py` | A CC0 read-speech corpus (`cv27-en`) prepared as an extra training source: every held-out speaker, clip and sentence removed (speakers joined by clip path across releases), the global denylist and a talk-text 8-gram rule, speaker and sentence caps, labels through filters version 3 (cleaning layer 1 is applied when read), a 16 kHz FLAC store in the synthetic store's format, and a check that B0's transcript matches the read sentence. Also the main pool's reject list for held-out talk text (`ted-test-text`), by the containment rule per source recording. |
| 4, step 0 | `keab_step0.py`, `training/eightbit.py`, `training/keab_export.py`, `training/deployed.py`, `training/keab_launch.sh`, `python-gpu1bit` | Which of Kea-B's floating-point parts (prediction network, joint, `pre_encode.out`, binary row scales) stage 4 trains and exports in 8 bits. INT8 per output channel (the weight computed with is the FP16 rounding of the dequantized value), checked on Kea-B in deployed mode (the real i8x1 kernels at one activation part, through `../parakeet-binary/gpu1bit`'s research loader) and on B0′ with only its decoder quantized, against a preregistered rule; palettization and a W8A8 `pre_encode.out` are measured for information. The export adapter writes an 8-bit bundle and derives from it the parakeet-binary-v1 scoring view that the runtime reads, bit-exact with the training forward weights. |
| 5, step 0 | `keat_step0.py`, `training/keat_eightbit.py`, `training/keat_export.py`, `training/keat_deployed.py`, `training/keat_m1.py`, `training/keat_launch.sh` | Stage 5 continues Kea-T (M1) instead of Kea-B. Step 0 is the same 8-bit check of the floating-point parts, on Kea-T in deployed mode. The x1 loader builds Kea-T's own deployed stack (`../parakeet-binary/gpu1bit`'s `runtime_m1.build_stack`) at one activation part on any export directory. Before anything else is scored, it must reproduce the recorded x1 dev WERs of Kea-T's export. Stage 4's INT8 format and scopes are reused. Under S2, each ternary module codes its weight from the FP32 row scale and computes with the FP16-rounded scale. The export adapter writes an 8-bit bundle and derives from it the parakeet-ternary-v1 scoring view; with nothing in 8 bits, that view is Kea-T's own export byte for byte. B0′'s rows are read from stage 4's step 0 report. |

Docstrings cite sections of the design plans ("labeling plan Part C", "synthetic plan"). The plans live in a
private planning space and are not part of this repository.

## Setup

- Copy `local.env.example` to `private/local.env` (gitignored) and fill in the data roots. `./python`
  reads it, refuses to run if the data mount is missing, and runs everything in the shared runtime.
- vLLM runs in its own environment (`labeling/vllm_env.sh`).
- Long jobs go through the `job` supervisor. See the `launch.sh` script of each stage.

## Batch plan cut engines

`training/batchplan.py` cuts each pool into batches with numpy by default. With `--cut bend` the cut comes
from the Bend 2 planner in `../bend-planner`, whose laws are proved, and numpy recomputes the same cut as a
check. The plan is refused, and nothing is written, unless the two cuts are byte-identical. A plan directory,
`plan.json` included, is identical whichever engine built it, so its identity does not depend on the engine.
The build prints which engine ran.

```
(cd ../bend-planner && nix develop ../.. -c bend planner.bend -o build/planner)   # once; build/ is gitignored
./python -m training.batchplan build --pools N --cut bend [--threads 8] [--planner PATH]
```

`--planner` defaults to `../bend-planner/build/planner`. The planner's inputs and outputs go to a temporary
directory under `$RELABEL_ROOT/tmp`, which is removed afterwards. The engine tests in
`tests/test_training_batchplan_engine.py` that need the real binary are skipped when it has not been built.

## Data policy

- Data, audio, labels and checkpoints are kept on the data drive, never in the repository.
- Sources whose licences are not permissive are evaluation only, never trained on.
- Code and docs refer to evaluation sets and corpora by alias (`med-dev-1`, `tech-dev-2`, `lectures-a`, …).
  A private registry, read at run time (`$PRIVATE_REGISTRY/sources.json`), maps each alias to its source,
  revision, licence and acquisition.
- Label text comes only from open-weight models.

## Tests

```
./python -m unittest discover -s tests
```

Most are CPU contract tests. The stage 2b loss test runs only when CUDA is available. The aligner device
test also needs `RELABEL_ALIGN_GPU_TEST=1`.
