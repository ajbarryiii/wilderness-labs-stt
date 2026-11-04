#!/usr/bin/env bash
# Jobs of stage 5 step 0 (keat-prime plan "Step 0", "Scoring and gate"; keat_step0.py), run through the `job`
# supervisor from finetune/relabel. VRAM and MEM are in GB; COMPUTE (environment, default 100) is the shared lease's
# compute share.
#
#   training/keat_launch.sh tests                         the stage 5 CPU contract tests (foreground, CUDA hidden)
#   training/keat_launch.sh gpu-tests VRAM MEM            the CUDA contract tests on M1's checkpoint and export, one
#                                                         job under the parakeet GPU lock: the export tests under
#                                                         ./python, the loader tests under ./python-gpu1bit
#   training/keat_launch.sh prepare VRAM MEM [OPTIONS]    M1's bundles and scoring views (exports/, GPU; options as
#                                                         keat_step0.py prepare, e.g. --variants, --no-gpu-lock)
#   training/keat_launch.sh m1-check VRAM MEM [LIMIT]     M1's export re-scored at x1 on full dev (or a smoke)
#   training/keat_launch.sh run VARIANT VRAM MEM [keat_step0.py run options, e.g. --sets dev --limit 20]
#   training/keat_launch.sh full VRAM MEM [AFTER] [LOCK_WAIT]
#                                                         the production scoring (keat_step0.py plan): the M1 check
#                                                         (a no-op if it already passed with this scorer), then one
#                                                         job per variant on the dev and domain sets, chained with
#                                                         --after (one GPU job at a time), then the report job; each
#                                                         waits up to LOCK_WAIT s (default 21600) for the GPU lock.
#                                                         Run it at COMPUTE=100: the recorded x1 arm ran on the whole
#                                                         GPU, and under MPS a smaller share changes the SMs seen
#   training/keat_launch.sh report [LIMIT]                the rule and tables (CPU, foreground)
#
# GPU jobs: shared lease with an explicit compute share, --priority normal, failures notified. Deployed-mode runs use
# ./python-gpu1bit with CUBLAS_WORKSPACE_CONFIG unset (the recorded x1 arm's environment); each process holds the
# parakeet GPU lock for its run (keat_step0.py). Nothing is downloaded (HF_HUB_OFFLINE=1).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
common=(--priority normal --notify failures --env HF_HUB_OFFLINE=1)
gpu() { echo --needs gpu:shared --compute "${COMPUTE:-100}" --vram "${1}G" --mem "${2}G"; }
x1=(env -u CUBLAS_WORKSPACE_CONFIG ./python-gpu1bit keat_step0.py)

case "${1:-}" in
  tests)
    export CUDA_VISIBLE_DEVICES=
    for p in test_keat_eightbit.py test_keat_export.py test_keat_deployed.py test_keat_step0.py; do
      ./python -m unittest discover -s tests -p "$p" -v
    done ;;
  gpu-tests)
    lock="$(./python -c 'import relabel_paths; print(relabel_paths.GPU_LOCK)')"
    script="set -euo pipefail
./python -m unittest discover -s tests -p test_keat_export_gpu.py -v
env -u CUBLAS_WORKSPACE_CONFIG ./python-gpu1bit -m unittest discover -s tests -p test_keat_deployed_gpu.py -v"
    exec job start relabel-keat-gpu-tests --cwd "$here" $(gpu "${2:?VRAM}" "${3:?MEM}") "${common[@]}" --timeout 60m \
      -- flock "$lock" bash -c "$script" ;;
  prepare)
    exec job start relabel-keat-step0-prepare --cwd "$here" $(gpu "${2:?VRAM}" "${3:?MEM}") "${common[@]}" \
      --timeout 90m -- ./python keat_step0.py prepare "${@:4}" ;;
  m1-check)
    extra=(); name=relabel-keat-step0-m1-check
    if [[ -n "${4:-}" ]]; then extra=(--limit "$4"); name="$name-limit$4"; fi
    exec job start "$name" --cwd "$here" $(gpu "${2:?VRAM}" "${3:?MEM}") "${common[@]}" --stall 60m \
      -- "${x1[@]}" m1-check "${extra[@]}" ;;
  run)
    variant="${2:?VARIANT}"; v="${3:?VRAM}"; m="${4:?MEM}"; shift 4
    exec job start "relabel-keat-step0-M1-$variant" --cwd "$here" $(gpu "$v" "$m") "${common[@]}" --stall 60m \
      -- "${x1[@]}" run --variant "$variant" "$@" ;;
  full)
    prev="${4:-}"; wait_s="${5:-21600}"
    after=(); [[ -n "$prev" ]] && after=(--after "$prev")
    job start relabel-keat-step0-m1-check --cwd "$here" $(gpu "${2:?VRAM}" "${3:?MEM}") "${common[@]}" --stall 60m \
      "${after[@]}" -- "${x1[@]}" m1-check --lock-wait "$wait_s" >/dev/null
    prev=relabel-keat-step0-m1-check; echo "$prev"
    while read -r variant groups; do
      after=(); [[ -n "$prev" ]] && after=(--after "$prev")
      # shellcheck disable=SC2086  # groups is a word list (dev domain)
      job start "relabel-keat-step0-M1-$variant" --cwd "$here" $(gpu "${2:?VRAM}" "${3:?MEM}") "${common[@]}" \
        --stall 60m "${after[@]}" -- "${x1[@]}" run --variant "$variant" --sets $groups --lock-wait "$wait_s" \
        >/dev/null
      prev="relabel-keat-step0-M1-$variant"; echo "$prev ($groups)"
    done < <(./python keat_step0.py plan | grep -E "^(none|S1-int8|S2-int8) ")
    exec job start relabel-keat-step0-report --cwd "$here" --mem 8G "${common[@]}" --after "$prev" \
      -- ./python keat_step0.py report ;;
  report)
    if [[ -n "${2:-}" ]]; then exec ./python keat_step0.py report --limit "$2"; fi
    exec ./python keat_step0.py report ;;
  *)
    sed -n '2,26p' "$0"; exit 2 ;;
esac
