#!/usr/bin/env bash
# Jobs of stage 4 step 0 (keab-prime plan "Step 0"; keab_step0.py), run through the `job` supervisor from
# finetune/relabel. VRAM and MEM are in GB.
#
#   training/keab_launch.sh tests                         the CPU contract tests (foreground, CUDA hidden)
#   training/keab_launch.sh gpu-tests VRAM MEM            the CUDA contract tests on MB1's checkpoint and export
#   training/keab_launch.sh prepare VRAM MEM [OPTIONS]    MB1's bundles and scoring views (exports/, GPU; options
#                                                         as keab_step0.py prepare, e.g. --variants, --no-gpu-lock)
#   training/keab_launch.sh mb1-check VRAM MEM [LIMIT]    MB1's export re-scored at x1 on full dev (or a smoke)
#   training/keab_launch.sh run MODEL VARIANT VRAM MEM [keab_step0.py run options, e.g. --sets dev --limit 20]
#   training/keab_launch.sh smoke LIMIT VRAM MEM [AFTER]  every model x variant on the first LIMIT utterances per
#                                                         set, one job (MB1 under ./python-gpu1bit, B0' under ./python),
#                                                         without the parakeet GPU lock (a quick smoke; the lease
#                                                         bounds its VRAM)
#   training/keab_launch.sh full VRAM_MB1 MEM_MB1 VRAM_B0P MEM_B0P [AFTER]
#                                                         the production scoring (keab_step0.py plan): one job per
#                                                         model x variant, the rule's variants on the dev and domain
#                                                         sets, palettization and W8A8 on the dev sets only, chained
#                                                         with --after (one GPU job at a time), then the report job
#   training/keab_launch.sh report [LIMIT]                the rule and tables (CPU, foreground)
#
# GPU jobs: shared lease, --priority normal, failures notified. MB1's deployed-mode runs use ./python-gpu1bit with
# CUBLAS_WORKSPACE_CONFIG unset (the recorded x1 arm's environment); each process holds the parakeet GPU lock for its
# run (keab_step0.py). Nothing is downloaded (HF_HUB_OFFLINE=1).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
common=(--priority normal --notify failures --env HF_HUB_OFFLINE=1)
gpu() { echo --needs gpu:shared --vram "${1}G" --mem "${2}G"; }
mb1=(env -u CUBLAS_WORKSPACE_CONFIG ./python-gpu1bit keab_step0.py)

case "${1:-}" in
  tests)
    export CUDA_VISIBLE_DEVICES=
    for p in test_keab_eightbit.py test_keab_export.py test_keab_step0.py test_keab_write_guard.py; do
      ./python -m unittest discover -s tests -p "$p" -v
    done ;;
  gpu-tests)
    lock="$(./python -c 'import relabel_paths; print(relabel_paths.GPU_LOCK)')"
    exec job start relabel-keab-gpu-tests --cwd "$here" $(gpu "${2:?VRAM}" "${3:?MEM}") "${common[@]}" --timeout 60m \
      -- flock "$lock" ./python -m unittest discover -s tests -p 'test_keab_export_gpu.py' -v ;;
  prepare)
    exec job start relabel-keab-step0-prepare --cwd "$here" $(gpu "${2:?VRAM}" "${3:?MEM}") "${common[@]}" \
      --timeout 90m -- ./python keab_step0.py prepare "${@:4}" ;;
  mb1-check)
    extra=(); name=relabel-keab-step0-mb1-check
    if [[ -n "${4:-}" ]]; then extra=(--limit "$4"); name="$name-limit$4"; fi
    exec job start "$name" --cwd "$here" $(gpu "${2:?VRAM}" "${3:?MEM}") "${common[@]}" --stall 60m \
      -- "${mb1[@]}" mb1-check "${extra[@]}" ;;
  run)
    model="${2:?MODEL}"; variant="${3:?VARIANT}"; v="${4:?VRAM}"; m="${5:?MEM}"; shift 5
    py=(./python keab_step0.py); [[ "$model" == MB1 ]] && py=("${mb1[@]}")
    exec job start "relabel-keab-step0-$model-$variant" --cwd "$here" $(gpu "$v" "$m") "${common[@]}" --stall 60m \
      -- "${py[@]}" run --model "$model" --variant "$variant" "$@" ;;
  smoke)
    n="${2:?LIMIT}"; after=(); [[ -n "${5:-}" ]] && after=(--after "$5")
    script="set -euo pipefail
for v in none S1-int8 S2-int8 S1-palette S2-palette S2-int8-w8a8; do
  env -u CUBLAS_WORKSPACE_CONFIG ./python-gpu1bit keab_step0.py run --model MB1 --variant \$v --limit $n --no-gpu-lock
done
for v in none S1-int8 S2-int8 S1-palette S2-palette; do
  ./python keab_step0.py run --model B0p --variant \$v --limit $n --no-gpu-lock
done
./python keab_step0.py report --limit $n"
    exec job start "relabel-keab-step0-smoke-$n" --cwd "$here" $(gpu "${3:?VRAM}" "${4:?MEM}") "${common[@]}" \
      --stall 60m "${after[@]}" -- bash -c "$script" ;;
  full)
    prev="${6:-}"
    while read -r model variant groups; do
      after=(); [[ -n "$prev" ]] && after=(--after "$prev")
      if [[ "$model" == MB1 ]]; then py=("${mb1[@]}"); v="${2:?VRAM_MB1}"; m="${3:?MEM_MB1}"
      else py=(./python keab_step0.py); v="${4:?VRAM_B0P}"; m="${5:?MEM_B0P}"; fi
      # shellcheck disable=SC2086  # groups is a word list (dev [domain])
      job start "relabel-keab-step0-$model-$variant" --cwd "$here" $(gpu "$v" "$m") "${common[@]}" --stall 60m \
        "${after[@]}" -- "${py[@]}" run --model "$model" --variant "$variant" --sets $groups >/dev/null
      prev="relabel-keab-step0-$model-$variant"; echo "$prev ($groups)"
    done < <(./python keab_step0.py plan | grep -E "^(MB1|B0p) ")
    exec job start relabel-keab-step0-report --cwd "$here" --mem 8G "${common[@]}" --after "$prev" \
      -- ./python keab_step0.py report ;;
  report)
    if [[ -n "${2:-}" ]]; then exec ./python keab_step0.py report --limit "$2"; fi
    exec ./python keab_step0.py report ;;
  *)
    sed -n '2,20p' "$0"; exit 2 ;;
esac
