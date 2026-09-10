# Wilderness Labs: one-hour GPU autoresearch

This is an original adaptation of the baseline / experiment / evaluate / retain-or-reject workflow in [Karpathy's program.md](https://github.com/karpathy/autoresearch/blob/master/program.md). The experiment is bounded to one hour and uses a prepared ternary speech/no-speech classifier. The upstream language-model metric and indefinite loop do not apply.

## Readiness comes before the clock

Environment construction, dependency installation, audio downloads, label/split auditing, evaluator implementation, compilation warmup and baseline qualification happen during preparation, outside the timed research service. The launcher must refuse to start until readiness is verified against the exact files being snapshotted. A tools/authentication check alone is insufficient.

Readiness evidence must include:
- the pinned training interpreter and dependency versions; a CUDA forward/backward/optimizer step on the RTX 5090 with finite loss and changed learned parameters;
- real, locally available speech and non-speech audio with source/license/checksum records, sufficient examples of both classes, documented label meaning and separated train/calibration/development groups;
- a completed baseline training and independent evaluation through the same interfaces used by trials, with a usable checkpoint, audited ternary weights and measured discrimination: FNR < 0.95, FPR < 0.95, balanced error (FNR + FPR) / 2 < 0.49 and finite score range > 1e-8; clearly distinguish a qualified optimization baseline from one meeting the stricter deployment-proxy operating gates;
- evaluator tests rejecting all-speech, all-non-speech, constant scores, missing/invalid denominators and invalid precision; an actual agent proposal under the workspace sandbox followed by a 60-second CUDA training and independent evaluation smoke run;
- an immutable snapshot of the evaluator, model/inference contract, data manifest and split/features checksums, benchmark, dependency locks and environment commands.

Readiness is evidence, not a request to launch. Start only with user authorization. The one-hour timer begins after these gates pass and the isolated snapshot is ready. Do not repeat provisioning during the timed hour or replace real data with synthetic features to hide a failed prerequisite. If prepared assets become inaccessible or corrupted, record the exact failure; do not manufacture a replacement benchmark.

## Project requirements

Read AGENTS.md, custom/PLAN.md, the frozen benchmark and this program. This program replaces the earlier bootstrap-during-the-hour procedure for this experiment.

- English-only, entirely local deployed STT for hands-free access to a TCCC manual.
- The first inference stage is a small speech/no-speech classifier. Pre-roll, hysteresis, hangover and bounded state preserve speech while gating the more expensive recognizer.
- Final recognizer capacity remains approximately 0.5–1B parameters. This pilot optimizes a separate classifier with at most 50,000 learned parameters; it does not establish final recognizer quality or training feasibility.
- Every learned inference weight, including input/convolution/head weights, uses ternary or binary codes. Scaling metadata and activation/arithmetic precision must be explicit. No hidden floating-point learned biases or affine normalization. FP32 training master weights are permitted.
- First deployment baseline is browser WASM SIMD CPU. Early latency and power hardware is iPhone 15 Pro running foreground Safari; an ARM64 Chrome laptop is a development target. The RTX 5090 is for training.
- Preserve medical dataset roles: PriMock57 is an adaptation candidate; Eka English remains held-out final evaluation; the symptom dataset requires license resolution; IntelMedica TTS remains excluded by default. The classifier pilot does not tune on those held-out medical data.
- Do not change host settings, drivers or GPU power limits; upload audio; publish; push; send external messages; accept new gated dataset terms; or access SSH credentials.

## Ownership and allowed edits

The supervisor creates the isolated workspace and frozen manifests before the deadline begins. It owns Git initialization/snapshots, training/evaluation execution, metrics, ranking, checkpoints, the ledger and the systemd deadline. You are the proposal agent. Work only in the supplied workspace, with the configured Codex model and sandbox.

For each proposal, edit only custom/vad/train.py and write proposal.json at the workspace root:

    {"label": "short descriptive name", "hypothesis": "One specific change and its expected effect"}

Use journal.md for short scientific notes when useful. Do not edit the controller, evaluator, benchmark, model/inference contract, manifests, source audio, cached features, dependency files, fixed thresholds or time limits. Do not write metrics/results/checkpoints, import evaluation labels into training, or alter evaluator imports through extra modules or environment tricks. Do not invoke training, evaluation, controller freeze/trial, dependency installation or long commands yourself: the supervisor runs the fixed training and independent evaluator outside the proposal sandbox. Small source inspection and syntax checks are sufficient during a proposal turn.

Do not commit, create alternate Git directories, set GIT_DIR/GIT_WORK_TREE, weaken sandboxing or work around a denied write. Git state is supervisor-owned. No agents, detached jobs, setsid/nohup/systemd-run, cron or scheduled follow-ups from the proposal agent. A denied access is diagnostic information, not authority to bypass the boundary.

The evaluator must reconstruct inference from the fixed model contract and safe checkpoint tensors/configuration. It must not execute candidate-defined inference code or accept candidate-supplied accuracy/precision results. If a proposed architecture is outside that contract, record it for a future prepared experiment; do not modify the harness mid-run.

## Data and scientific contract

Only the prepared, checksum-verified real-audio benchmark is scored. Procedural vectors, tones and invented speech labels may be correctness fixtures outside the scored benchmark; they never substitute for a speech accuracy result. Augmentation of real speech/noise is allowed only under the fixed source-grounded labeling and split rules. Report such augmentation as augmentation, not new human recordings.

A transcript does not label every frame as speech. Use the label granularity in benchmark.json exactly. Clip-level human speech-presence data is a valid first classifier proxy but cannot establish continuous frame-VAD recall, onset/tail clipping, entirely missed utterances, false activations/hour or deployment gating quality. Leave unavailable metrics null and disclose the limitation.

Keep training, threshold calibration and development assignments fixed, with speaker/session/source-recording grouping as specified by the manifest. Do not use development labels for gradient updates or threshold selection. Only the frozen calibration procedure selects the operating threshold. The independent evaluator applies it unchanged to development audio. Preserve any final test boundary.

## GPU trial contract

The supervisor runs the unchanged baseline first, then proposal/evaluate cycles. Each training subprocess receives:

    --output DIRECTORY --train-seconds 300 --seed 20260909

It has a 300-second envelope including imports, setup and checkpoint writing. Use the ready CUDA environment, require CUDA availability, keep optimizer updates on the RTX 5090 and train on the real training split. A CPU fallback or a checkpoint produced without meaningful learning is an invalid trial. Use the fixed seed and equal training budget; initialize from the same prescribed starting state. No incumbent checkpoint warm-start unless the frozen benchmark explicitly specifies an equal starting checkpoint for every candidate. The supervisor may spend the remaining tail budget on a separate, unranked fresh-seed confirmation of the retained architecture/recipe. Such a shortened run must be labeled confirmation, record its actual budget and seed, and never replace the best candidate or enter equal-budget ranking.

Train repeatedly until the save margin, preserving at least 15 seconds for checkpoint output. Do not finish after a tiny fixed number of updates when useful training budget remains. Save the schema required by the fixed evaluator, training-step/loss history and device/elapsed-time metadata. Never invent telemetry. Follow any stricter minimum-step and training-duration checks in the frozen benchmark.

The evaluator runs separately with:

    evaluate --candidate DIRECTORY --output METRICS_JSON

The complete training-plus-evaluation trial is capped at 420 seconds. It independently validates checkpoint structure, finite tensors, effective inference weight codes, size and predictions; computes the fixed split metrics; and times the reference CPU workload. The controller verifies frozen hashes before and after trials and validates required metric fields. Trials with crashes, changed protected files, missing classes, malformed metrics or failed precision/learning/device gates are errors, not zero-error scores.

## Selection and efficiency claims

Use the frozen benchmark's thresholds and ranking. Provisional feasibility requires both FNR <= 0.01 and FPR <= 0.20. These are research operating gates, not deployment or field acceptance. Reaching 99% recall by declaring everything speech is a failed classifier. All-speech, all-non-speech and constant-score candidates must be rejected even if one marginal error rate looks good. Do not promote an infeasible candidate as a successful result.

Among feasible candidates minimize non-speech false positives with the predeclared minimum improvement, while preserving speech recall. Use the fixed latency tie-break rule only within the permitted accuracy regression margins. An infeasible candidate may be retained only as a discriminative diagnostic baseline: FNR < 0.95, FPR < 0.95, balanced error (FNR + FPR) / 2 < 0.49 and finite score range > 1e-8. Apply the controller's maximum normalized FNR/FPR violation ranking and always label the result infeasible_best. This retention enables measured optimization; it is not a successful operating point. Do not change the operating point or thresholds after viewing development results.

Report true/false positives and negatives, calibration threshold, label granularity, class sample counts, condition breakdown where supported, parameters, packed bytes, precision audit, training steps/time/device and peak VRAM. CPU p95 is a remote x86 reference. PyTorch fake quantization is training/reference arithmetic, not a packed browser kernel. GPU utilization and GPU energy are not iPhone power. iPhone 15 Pro latency/joules remain unmeasured until measured on that device. The hour is research wall time, not a promise of 100% GPU utilization: proposal reasoning and evaluation consume time too. Aim to keep proposal turns brief and deliver substantive CUDA trials.

## Proposal loop and deadline

The supervisor records absolute start/deadline times and enforces the 3,600-second systemd control-group deadline. Reserve 90 seconds for reporting. A complete next cycle must fit its bounded proposal, training/evaluation and reporting budgets. Each proposal turn has at most 120 seconds; target a clear change in under one minute.

1. Read the supplied remaining budget, baseline, ledger, best candidate and last failure summary. Inspect only the source needed to make a change.
2. Choose one grounded hypothesis within the fixed model contract, such as ternary optimization/scale handling, learning rate or schedule, batch strategy, or permitted regularization. Connect it to observed learning behavior or a measured error condition.
3. Modify train.py, write proposal.json, record a short note if needed, and end the proposal turn promptly. The supervisor then trains and evaluates it.
4. On the next turn, inspect the independent outcome. Start from the retained candidate supplied by the supervisor. Repair a recoverable training failure before making another speculative change. A rejected candidate is evidence, not a reason to stop.
5. Continue while another bounded cycle fits. Do not declare the experiment done merely because one baseline works or a first idea fails. If no useful new proposal is ready, explicitly state this and leave a valid baseline for the supervisor's recorded fallback/reproducibility work.
6. In report-only mode, write scientific synthesis into REPORT.md using actual ledger results. Do not initiate work outside the remaining budget. The supervisor alone may first run an unranked fresh-seed confirmation when at least 60 seconds of training plus its full evaluation and report reserves fit; this is robustness evidence at a different budget, not an equal-training comparison.

Do not create BLOCKED.md for ordinary rejected candidates, lack of a novel idea or the earlier missing-dependency condition that readiness already resolved. A concrete new external failure such as broken CUDA access, missing prepared files or unavailable CLI service must be reported plainly; the supervisor determines recovery or stopping. Never conceal repeated failures or consume the remaining hour in an unchanged failed command loop.

## Deliverables

The supervisor preserves every proposal, source snapshot, training/evaluation log, metrics, accepted checkpoint and rejected/error result. Its report is always available even if the proposal service becomes unavailable. Add concise synthesis covering the hypothesis, baseline and every valid result; best candidate and checkpoint; failed attempts; benchmark/proxy limitations; actual CUDA training work; and the next specific experiment.

If no feasible improvement is found, say so and retain the qualified baseline or best discriminative diagnostic candidate, explicitly preserving infeasible status where applicable. If no valid trial completes, say so prominently. Never infer WER, clinical readiness, final 0.5–1B performance or iPhone battery savings from this classifier experiment. Stop at the deadline; extending the hour requires a new user request.

If proposal generation repeatedly fails, the supervisor may continue the already retained recipe with distinct seeds for explicitly unranked reproducibility checks. These are logged as recovery confirmations, never promoted as improvements. A failed validated training/evaluation retry stops this fallback. The report preserves the original proposal-service failure.
