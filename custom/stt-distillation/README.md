# STT distillation into ternary models

The revised September 10 four-hour pilot passed the small learning gates but
showed FP acoustic collapse and poor ternary generalization on broader speech;
see the [completed investigation](INVESTIGATION_BROAD_20260910.md).
The [training recovery and local LM plan](RECOVERY_AND_DECODING_PLAN.md) specifies
controlled repair experiments, a small pretrained CTC reference, and beam-search
comparisons with an existing compact language model.
The [eight-hour implementation and operating guide](RECOVERY_8H.md) describes
the recovery worker, resumable supervisor, local decoder, checks, and run commands.
The September 11 recovery run restored training with gate initialization but
retained severe overfitting. The [eight-hour augmentation pilot](AUGMENTATION_8H_PLAN.md)
compares masking, environmental noise and quieter competing speech, with a pinned
runtime and mandatory training/resume/evaluation preflight before launch.

The [unique-data/compute experiment](DATA_SCALING_PLAN.md) prepares nested
6.7 / 23.2 / 45.2 hour datasets, two paired seeds, and retained checkpoints
through 27,000 updates. `scaling.json` defines the budget; `scaling_data.py`,
`scaling_control.py`, `scaling_train.py`, and `scaling_analysis.py` prepare,
run and analyze it. Preparation and CPU validation do not start training.
Full-size GPU preflight and launch are separate explicit operations.

Status: the first eight-hour pilot completed. All four students collapsed to
empty or constant outputs; see [the results and diagnostic audit](RESULTS_20260910.md).
See also [the kernel profile and initial training-repair probes](TRAINING_REPAIR_20260910.md).
The earlier September 10 08:04 four-hour pilot stopped at its FP digit-learning
gate; see [the investigation](RESULTS_PILOT4_20260910.md) and [protocol](PILOT_4H.md).
The [onset repair experiments](ONSET_REPAIR.md) test sustained signal onset,
first-emission timing, and robustness to leading silence and recording gain.
The fresh FP recipe passed its stronger learning gate and all 192 transformed
digit cases; see [results and limits](RESULTS_ONSET_20260910.md).
The [pilot protocol](PILOT_8H.md) records the four-arm test and operating commands.

Distill a pretrained English recognizer into a custom student with approximately
matching parameter count. Establish full-precision and ternary quality controls,
then progressively reduce the ternary student's capacity to measure the accuracy,
latency, memory and energy tradeoff. Binary weights are a later ablation.

Read [the experiment plan](PLAN.md) for the matrix, stage gates, data policy,
implementation sequence and stopping rules.

This is a separate track from [the existing STT autoresearch](../stt/README.md).
It uses the precision and streaming design in [the custom model plan](../PLAN.md),
but deliberately permits compression below that plan's earlier 0.5–1B capacity
range. This experiment targets the lowest measured energy at acceptable quality.

Code, configuration, aggregate reports and artifact hashes belong here. All audio,
prepared data, teacher targets, caches, weights and run artifacts must live under
`/mnt/hd/wilderness-labs-stt/stt-distillation/`. Preparation must verify that
`/mnt/hd` is mounted before creating artifact directories or writing files.
