#!/usr/bin/env bash
# Jobs of stage 2b (decoder-only plan), run through the `job` supervisor from finetune/relabel.
#
#   training/launch.sh tests                      unit and contract tests that need no GPU (foreground)
#   training/launch.sh gpu-tests VRAM             the CUDA contract tests (tiny model), a short GPU job
#   training/launch.sh split                      freeze the dev split (CPU, foreground; once, before any run)
#   training/launch.sh equivalence S VRAM         encoder-equivalence check of student S (B or T), 20 dev clips
#   training/launch.sh probe S VRAM               100-step probe of student S (peak VRAM, step time)
#   training/launch.sh chain RUN VRAM N           N chained chunk jobs over ORDER (the two original students, then
#                                                 the 12 runs) for labels/<T*>/RUN; each chunk ~110 min
#   training/launch.sh status | report            progress; results.json and RESULTS-decoder-only.md (CPU)
#
# Stage 2c (synthetic plan; train_synthetic.py), after the 12 runs and the finalized "full" synthetic run, with the
# version 2 plan built (./python -m training.batchplan build --pools 156 --synthetic-run full):
#   training/launch.sh syn-probe S RUN VRAM       mixed-batch probe of student S on labels/<T*>/RUN (GPU)
#   training/launch.sh syn-read-probe             synthetic reads under the replay throughput probe's rule (CPU)
#   training/launch.sh syn-chain RUN VRAM N [JOB] N chained chunk jobs over train_synthetic.ORDER (the T* runs'
#                                                 eval-stem, then B-TS, B-TSn, T-TS, T-TSn); the first waits for JOB
#                                                 (e.g. the last stage 2b chunk) if given, else readiness is checked
#   training/launch.sh syn-status | syn-report    progress; results-synthetic.json and RESULTS-decoder-only.md
#
# GPU jobs: a shared lease (--needs gpu:shared --vram VRAM, VRAM in GB: the probe's peak plus 2 GB), 36G memory
# cap, low priority (the agent default). Each process takes the parakeet GPU lock ($PARAKEET_TERNARY_ROOT/gpu.lock)
# for its own duration only: train.py's paths.gpu_lock for training sessions, score_teacher.gpu_lock for scoring,
# equivalence and probe jobs, flock for the GPU tests. A chunk job runs its items as subprocesses, so exclusive
# jobs get the GPU between items and between chunks. Every training run is resumable from its latest checkpoint;
# a chunk that stops on its budget exits 0 and the next chunk continues. Nothing is downloaded (HF_HUB_OFFLINE=1).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
gpu=(--mem 36G --stall 30m --env HF_HUB_OFFLINE=1)

case "${1:-}" in
  tests)
    ./python -m unittest discover -s tests -p 'test_training_*.py' -v
    exec ./python -m unittest discover -s tests -p 'test_decoder_only.py' -v ;;
  gpu-tests)
    v="${2:?VRAM GB}"
    lock="$(./python -c 'import relabel_paths; print(relabel_paths.GPU_LOCK)')"
    exec job start relabel-decoder-gpu-tests --cwd "$here" --needs gpu:shared --vram "${v}G" "${gpu[@]}" \
      --timeout 40m -- flock "$lock" ./python -m unittest discover -s tests -p 'test_decoder_only_gpu.py' -v ;;
  split)
    exec ./python train_decoder.py split ;;
  equivalence|probe)
    s="${2:?student B|T}"; v="${3:?VRAM GB}"
    exec job start "relabel-decoder-$1-$s" --cwd "$here" --needs gpu:shared --vram "${v}G" "${gpu[@]}" \
      --timeout 1h -- ./python train_decoder.py "$1" --student "$s" ;;
  chain)
    run="${2:?labels run name}"; v="${3:?VRAM GB}"; n="${4:?number of chunks}"
    ./python -c 'import sys, train_decoder as td, training.devsplit as d; d.load(td.split_path(), d.dev_manifests())' \
      || { echo "the dev split is not frozen (training/launch.sh split)" >&2; exit 2; }
    prev=""
    for i in $(seq 1 "$n"); do
      after=(); [[ -n "$prev" ]] && after=(--after "$prev")
      job start "relabel-decoder-$run-$i" --cwd "$here" "${after[@]}" --needs gpu:shared --vram "${v}G" "${gpu[@]}" \
        --timeout 3h -- ./python train_decoder.py chunk --labels-run "$run" --budget-minutes 110
      prev="relabel-decoder-$run-$i"
    done ;;
  syn-probe)
    s="${2:?student B|T}"; run="${3:?labels run name}"; v="${4:?VRAM GB}"
    exec job start "relabel-synthetic-probe-$s" --cwd "$here" --needs gpu:shared --vram "${v}G" "${gpu[@]}" \
      --timeout 1h -- ./python train_synthetic.py probe --student "$s" --labels-run "$run" ;;
  syn-read-probe)
    exec job start relabel-synthetic-read-probe --cwd "$here" --needs cpu-heavy --mem 8G --timeout 1h -- \
      ./python train_synthetic.py read-probe ;;
  syn-chain)
    run="${2:?labels run name}"; v="${3:?VRAM GB}"; n="${4:?number of chunks}"; prev="${5:-}"
    if [[ -z "$prev" ]]; then
      ./python -c 'import sys, train_synthetic as ts; p = ts.readiness(); print(*p, sep="\n", file=sys.stderr); sys.exit(2 if p else 0)' \
        || { echo "stage 2c is not ready (above); pass the last stage 2b chunk job to wait for it" >&2; exit 2; }
    fi
    for i in $(seq 1 "$n"); do
      after=(); [[ -n "$prev" ]] && after=(--after "$prev")
      job start "relabel-synthetic-$run-$i" --cwd "$here" "${after[@]}" --needs gpu:shared --vram "${v}G" "${gpu[@]}" \
        --timeout 3h -- ./python train_synthetic.py chunk --labels-run "$run" --budget-minutes 110
      prev="relabel-synthetic-$run-$i"
    done ;;
  syn-status)
    exec ./python train_synthetic.py status ;;
  syn-report)
    exec ./python train_synthetic.py report "${@:2}" ;;
  status)
    exec ./python train_decoder.py status ;;
  report)
    exec ./python train_decoder.py report "${@:2}" ;;
  *)
    sed -n '2,27p' "$0" >&2; exit 2 ;;
esac
