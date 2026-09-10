# One-hour ternary STT GPU autoresearch

This runner optimizes a prepared ternary speech/no-speech classifier on the RTX 5090. The final recognizer remains approximately 0.5–1B parameters. A classifier pilot does not establish final-scale STT quality or iPhone power savings.

The initial run, 20260909T213516Z, stopped after about 8 minutes 19 seconds without real GPU training. Its synthetic scores and retained all-speech candidate are invalid as speech-model evidence. The repaired protocol moves all provisioning before the timer, requires real-audio and CUDA baseline qualification, rejects degenerate classifiers, and makes the supervisor execute training/evaluation independently of the proposal agent.

## Before starting

Preparation must finish outside the one-hour service: install/pin dependencies, cache permitted real labeled audio, audit split provenance, warm up CUDA, complete baseline training/evaluation, test degeneracy rejection, and verify an actual sandbox proposal followed by a 60-second CUDA training/evaluation smoke run. The launch gate must verify this evidence against current file/data hashes. Merely finding Python, Codex and systemd does not establish readiness.

Dependency or data preparation can take longer than an hour. No research clock is started to hide that work. See the current readiness artifact and controller status for operational state; this README and experiment.json are configuration, not live run status. Preparation never schedules or implicitly starts a future session.

## Run on aj@nixos from the repository

```sh
custom/autoresearch/control check
# Minimum CUDA training qualification (outside the research timer):
custom/autoresearch/control preflight --train-seconds 60
# Use the complete trial envelope for full launch qualification:
custom/autoresearch/control preflight --train-seconds 300
custom/autoresearch/control status
# After readiness passes and the user authorizes this session:
custom/autoresearch/control start --authorize-one-hour
custom/autoresearch/control status
custom/autoresearch/control stop
```

Preflight runs a real CUDA baseline and independent evaluation without starting the one-hour service. A 60-second training preflight is the minimum supported check; the repair is being qualified with the full 300-second training envelope. Start refuses missing, stale or changed readiness evidence and rechecks current CUDA/data access. The first research minutes are reserved for verified training, not dependency/data bootstrap.

## Execution

The prepared data recipe uses real MiniLibriSpeech utterances as speech-presence positives and recorded noise sources from the OpenSLR RIRS_NOISES pointsource collection as negatives, with documented source/speaker separation. Exact included files, licenses and hashes belong to the data manifest. This is a clip-presence proxy; entire utterances are not mislabeled as all-speech frames.

Before the timed service, the supervisor copies the complete ready classifier, environment contract and immutable evaluator/data metadata into a separate workspace and records frozen hashes. It owns Git setup, so the proposal agent does not need to commit or bypass sandbox restrictions.

The supervisor runs a baseline followed by bounded iterations. A Codex proposal turn edits only custom/vad/train.py and writes proposal.json with label and hypothesis. It has a 120-second cap. The supervisor then runs CUDA training for up to 300 seconds and the independent evaluator within a 420-second total trial budget. It records keep/discard/infeasible/error outcomes and restores the accepted source for the next proposal. A rejected candidate does not end the session. When a full scored cycle no longer fits, the supervisor may use the tail budget for an unranked fresh-seed confirmation of the retained model/recipe if at least 60 seconds of training plus evaluation and reporting fit. It records the shortened budget and seed, marks the result confirmation, and never promotes it or compares it as an equal-budget trial.

The fixed evaluator reads safe checkpoint tensors/configuration using the frozen inference contract. It does not execute candidate inference code or trust candidate-written metrics. Feasible candidates require FNR <= 0.01 and FPR <= 0.20; constant and all-speech/all-non-speech outputs cannot win. A discriminative candidate missing those operating gates may be retained as infeasible_best only when FNR and FPR are each < 0.95, balanced error is < 0.49, and finite score range is > 1e-8. Such retention does not count as meeting the operating point. Real clip-presence data, if used, is labeled as a proxy rather than continuous frame-VAD validation.

The configured remote Codex model is used without an experiment-specific model override. The proposal service uses that CLI; deployed speech inference remains local. Audio is not uploaded. Setup network access belongs to preparation; training uses local cached data and fixed dependencies.

## Time and output

systemd RuntimeMaxSec=3600 and KillMode=control-group enforce the hard one-hour service limit for all child processes. The supervisor reserves 90 seconds for reporting and admits only cycles that fit. Proposal reasoning and evaluation count toward the hour, so the allocation is not 3,600 GPU-training seconds or guaranteed 100% utilization. GPU trials must nevertheless perform meaningful optimizer work, with steps, elapsed time, losses, device and memory recorded.

Runs are stored under custom/autoresearch/runs/RUN_ID/: state, supervisor/agent logs, the isolated workspace, readiness/frozen evidence, trial source/checkpoint/log snapshots, metrics, results.tsv and REPORT.md. Main-checkout changes are not automatically merged. Prior failed runs remain available for audit. Runtime/environment artifacts are separate from configuration and ignored by Git.

The final report separates measured real-audio accuracy, remote CPU timing and CUDA training telemetry from unavailable measurements. iPhone 15 Pro Safari/WASM latency and power remain unverified until tested on the phone.

## Files

- program.md: proposal-agent instructions and scientific boundaries.
- experiment.json: fixed session, proposal, trial and reporting budgets; readiness and metric requirements.
- control.py and control: explicit launch, readiness gate, isolated snapshot, deadline, independent training/evaluation, ledger and report.
- tests.py: controller regression tests, including invalid metrics, degenerate candidates and launch admission.
- provenance.json: original upstream adaptation provenance; inspect readiness/run artifacts for current runtime and experiment evidence.

If proposal turns repeatedly fail, the supervisor records the failure and may use the remaining authorized GPU budget for unranked fresh-seed repetitions of the retained recipe. These are reproducibility evidence, not claimed optimization gains. Failure of the validated fallback stops further retries.
