#!/usr/bin/env bash
# Jobs of stage 4's trainer (keab-prime plan; train_keab.py), run through the `job` supervisor from finetune/relabel.
# VRAM and MEM are in GB; PROV is "--provisional" (the synthetic union before run pairs-c is final) or empty.
#
#   training/keab_train.sh tests                          the CPU contract tests (foreground, CUDA hidden)
#   training/keab_train.sh gpu-tests VRAM MEM             the CUDA contract tests (quantizer against the kernel, trainer)
#   training/keab_train.sh plan MEM [PROV]                batch plan version 3 of pools 0-781 (CPU job)
#   training/keab_train.sh read-probe MEM [PROV]          its read probe (CPU job)
#   training/keab_train.sh labels RUN MEM [PROV]          both arms' label indexes on it (CPU job)
#   training/keab_train.sh probe RUN SCOPE VRAM MEM [PROV] [AFTER]
#                                                         the memory probe, both modes (GPU)
#   training/keab_train.sh smoke ARM RUN SCOPE STEPS VRAM MEM [PROV] [AFTER]
#                                                         a smoke run under keab-prime/smoke/ (GPU)
#   training/keab_train.sh fidelity VRAM MEM SCOPE|SMOKE [RUN] [AFTER]
#                                                         the export check: MB1 at SCOPE first (the bound), then a smoke
#                                                         model (SMOKE RUN) against MB1 at the smoke's scope
#   training/keab_train.sh chain RUN VRAM MEM N THROUGH [AFTER]
#                                                         N chained chunk jobs over train_keab.ORDER through pilot, main
#                                                         or score (the architect launches it; never a smoke)
#   training/keab_train.sh status | report
#
# GPU jobs: a shared lease (--needs gpu:shared --vram VRAM), --priority normal, failures notified, nothing downloaded.
# Each process holds the parakeet GPU lock for its own duration (train.py's paths.gpu_lock, score_teacher.gpu_lock, flock
# for the GPU tests). Deployed-mode scoring (x1, x3) runs under ./python-gpu1bit with CUBLAS_WORKSPACE_CONFIG unset (the
# recorded x1 arm's environment); the chunk starts those items so itself.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
common=(--priority normal --notify failures --env HF_HUB_OFFLINE=1)
gpu() { echo --needs gpu:shared --vram "${1}G" --mem "${2}G"; }
after() { [[ -n "${1:-}" ]] && echo --after "$1" || true; }
lock() { ./python -c 'import relabel_paths; print(relabel_paths.GPU_LOCK)' 2>/dev/null | tail -1; }

case "${1:-}" in
  tests)
    export CUDA_VISIBLE_DEVICES=
    for p in test_keab_actquant.py test_keab_batchplan.py test_keab_trainer.py test_keab_report.py; do
      ./python -m unittest discover -s tests -p "$p" -v
    done ;;
  gpu-tests)
    exec job start relabel-keab-trainer-gpu-tests --cwd "$here" $(gpu "${2:?VRAM}" "${3:?MEM}") "${common[@]}" \
      --timeout 90m -- flock "$(lock)" bash -c \
      './python -m unittest discover -s tests -p test_keab_actquant_gpu.py -v && ./python -m unittest discover -s tests -p test_keab_trainer_gpu.py -v' ;;
  plan|read-probe)
    exec job start "relabel-keab-$1" --cwd "$here" --mem "${2:?MEM}G" "${common[@]}" --timeout 3h \
      -- ./python train_keab.py "$1" ${3:-} ;;
  labels)
    exec job start relabel-keab-labels --cwd "$here" --mem "${3:?MEM}G" "${common[@]}" --timeout 4h \
      -- ./python train_keab.py labels --labels-run "${2:?RUN}" ${4:-} ;;
  probe)
    exec job start relabel-keab-probe --cwd "$here" $(gpu "${4:?VRAM}" "${5:?MEM}") "${common[@]}" --stall 45m \
      --timeout 4h $(after "${7:-}") -- ./python train_keab.py probe --labels-run "${2:?RUN}" --scope "${3:?SCOPE}" ${6:-} ;;
  smoke)
    arm="${2:?ARM}"
    exec job start "relabel-keab-smoke-$arm" --cwd "$here" $(gpu "${6:?VRAM}" "${7:?MEM}") "${common[@]}" --stall 45m \
      --timeout 3h $(after "${9:-}") -- ./python train_keab.py smoke --arm "$arm" --labels-run "${3:?RUN}" \
      --scope "${4:?SCOPE}" --steps "${5:?STEPS}" --name "smoke-$arm" ${8:-} ;;
  fidelity)
    what="${4:?SCOPE or SMOKE}"
    if [[ "$what" == none || "$what" == S1 || "$what" == S2 ]]; then
      extra=(--scope "$what"); name="relabel-keab-fidelity-MB1-$what"
    else
      extra=(--smoke "$what" --run "${5:?RUN}"); name="relabel-keab-fidelity-$what"
    fi
    exec job start "$name" --cwd "$here" $(gpu "${2:?VRAM}" "${3:?MEM}") "${common[@]}" --stall 45m --timeout 2h \
      $(after "${6:-}") -- env -u CUBLAS_WORKSPACE_CONFIG ./python-gpu1bit train_keab.py fidelity "${extra[@]}" ;;
  chain)
    run="${2:?RUN}"; v="${3:?VRAM}"; m="${4:?MEM}"; n="${5:?N}"; through="${6:?THROUGH}"; prev="${7:-}"
    for i in $(seq 1 "$n"); do
      job start "relabel-keab-$through-$(printf %02d "$i")" --cwd "$here" $(gpu "$v" "$m") "${common[@]}" --stall 45m \
        --timeout 3h $(after "$prev") -- ./python train_keab.py chunk --labels-run "$run" --through "$through" \
        --budget-minutes 110 >/dev/null
      prev="relabel-keab-$through-$(printf %02d "$i")"; echo "$prev"
    done ;;
  status)
    exec ./python train_keab.py status ;;
  report)
    exec ./python train_keab.py report ;;
  *)
    sed -n '2,24p' "$0"; exit 2 ;;
esac
