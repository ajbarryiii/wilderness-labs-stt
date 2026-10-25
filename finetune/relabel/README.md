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
| 2b | `train_decoder.py`, `training/` | Decoder-only fine-tuning of Kea-B and Kea-T on the new labels, from an offline batch plan (`training/batchplan.py`). |
| 2c | `synthetic/`, `terms/synthetic/` | Synthetic technical speech (chemistry, mathematics, finance, software engineering): open-weight text generation, then TTS and augmentation. |
| 3 | `training/` (stage 3 entry point to come) | B0′: FP32 Parakeet fine-tuned on the new labels with a WSD schedule and cooldown branches. |

Docstrings cite sections of the design plans ("labeling plan Part C", "synthetic plan"). The plans live in a
private planning space and are not part of this repository.

## Setup

- Copy `local.env.example` to `private/local.env` (gitignored) and fill in the data roots. `./python`
  reads it, refuses to run if the data mount is missing, and runs everything in the shared runtime.
- vLLM runs in its own environment (`labeling/vllm_env.sh`).
- Long jobs go through the `job` supervisor. See the `launch.sh` script of each stage.

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
