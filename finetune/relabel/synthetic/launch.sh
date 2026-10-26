#!/usr/bin/env bash
# Jobs of stage 2c (synthetic plan), run through the `job` supervisor from finetune/relabel.
#
#   synthetic/launch.sh tests                       contract tests (foreground)
#   synthetic/launch.sh fetch ALIAS...              pinned downloads (network, no GPU)
#   synthetic/launch.sh noise-pool                  extract + prepare the TRAIN noise/RIR pool, disjointness check (CPU)
#   synthetic/launch.sh textgen RUN VRAM N          N chained generator chunk jobs (GPU, shared lease)
#   synthetic/launch.sh tts RUN ENGINE N [cpu]              N chained CPU rendering chunk jobs of one engine
#   synthetic/launch.sh tts RUN ENGINE N cuda VRAM MEM      the same on the GPU (shared lease; VRAM declared in GB,
#                                                   MEM = the probe's host peak + 25%)
#   synthetic/launch.sh tts-probe ENGINE VRAM MEM   GPU rendering probe (<= 15 min) in smoke/synthetic/tts-probe
#                                                   (prepare it first: ./python -m synthetic reuse-text --from pilot
#                                                   --smoke tts-probe, then hygiene and tts-plan --smoke tts-probe)
#   synthetic/launch.sh roundtrip RUN VRAM N [KIND] [SETTING]  N chained T* chunk jobs (GPU, shared lease)
#   synthetic/launch.sh after-text RUN              hygiene + tts-plan (CPU, after the textgen chunks)
#   synthetic/launch.sh finalize RUN                clips, report, listening sample (CPU)
#
# GPU jobs: a shared lease (--needs gpu:shared --vram VRAM, VRAM in GB), 36G memory cap, low priority, chunks of
# at most ~110 min; each job takes the parakeet GPU lock for its own duration only (the CLI holds it), so exclusive
# jobs get the GPU between chunks. Nothing is downloaded under the lock (HF_HUB_OFFLINE=1). RUN is pilot, pilot2 or
# full. TTS on the GPU: LEASE = ceil(VRAM) + 1 GB, as in labeling/launch.sh (the supervisor's lease fraction is over
# the GPU's nominal size, the engines' check over the smaller size torch reports; engines.vram_fraction).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
gpu=(--mem 36G --stall 30m --env HF_HUB_OFFLINE=1)
lease() { awk -v v="$1" 'BEGIN { c = int(v); if (c < v) c++; printf "%dG", c + 1 }'; }   # labeling/launch.sh's rule

chain() {   # chain NAME N -- command...: N jobs NAME-1..N, each after the previous
  local name="$1" n="$2"; shift 3
  local prev="" after
  for i in $(seq 1 "$n"); do
    after=(); [[ -n "$prev" ]] && after=(--after "$prev")
    job start "$name-$i" --cwd "$here" "${after[@]}" "${JOB_ARGS[@]}" -- "$@"
    prev="$name-$i"
  done
}

case "${1:-}" in
  tests)
    exec ./python -m unittest discover -s tests -p 'test_synthetic_*.py' -v ;;
  fetch)
    shift
    exec job start "syn-fetch-$(echo "$@" | tr ' ' '-')" --cwd "$here" --mem 8G --cpus 2 --timeout 8h --stall 30m -- \
      ./python -m synthetic fetch "$@" ;;
  noise-pool)
    exec job start syn-noise-pool --cwd "$here" --needs cpu-heavy --mem 24G --cpus 8 --timeout 6h --stall 40m -- \
      ./python -m synthetic noise-pool --workers 6 ;;
  textgen)
    run="${2:?run}"; v="${3:?VRAM GB}"; n="${4:?chunks}"
    JOB_ARGS=(--needs gpu:shared --vram "${v}G" "${gpu[@]}" --timeout 2h30m)
    chain "syn-textgen-$run" "$n" -- ./python -m synthetic textgen --run "$run" --declared-vram-gb "$v" \
      --budget-minutes 110 ;;
  tts)
    run="${2:?run}"; eng="${3:?engine}"; n="${4:?chunks}"; dev="${5:-cpu}"
    if [[ "$dev" == cuda ]]; then
      v="${6:?VRAM GB}"; mem="${7:?MEM (probe host peak + 25%)}"
      JOB_ARGS=(--needs gpu:shared --vram "$(lease "$v")" --mem "$mem" --cpus 4 --stall 30m --env HF_HUB_OFFLINE=1
                --timeout 2h30m --env "RELABEL_DEPS=$eng")
      chain "syn-tts-$run-$eng" "$n" -- ./python -m synthetic tts --run "$run" --engine "$eng" --device cuda \
        --declared-vram-gb "$v" --threads 2 --budget-minutes 110
    else
      JOB_ARGS=(--mem 12G --cpus 6 --stall 30m --timeout 2h30m --env "RELABEL_DEPS=$eng")
      chain "syn-tts-$run-$eng" "$n" -- ./python -m synthetic tts --run "$run" --engine "$eng" --device cpu \
        --threads 6 --budget-minutes 110
    fi ;;
  tts-probe)
    eng="${2:?engine}"; v="${3:?VRAM GB}"; mem="${4:?MEM}"
    exec job start "syn-tts-probe-$eng" --cwd "$here" --needs gpu:shared --vram "$(lease "$v")" --mem "$mem" --cpus 4 \
      --stall 10m --timeout 20m --expect 15m --env HF_HUB_OFFLINE=1 --env "RELABEL_DEPS=$eng" -- \
      ./python -m synthetic tts --smoke tts-probe --engine "$eng" --device cuda --declared-vram-gb "$v" --threads 2 \
      --budget-minutes 10 --lock-wait 300 ;;
  roundtrip)
    run="${2:?run}"; v="${3:?VRAM GB}"; n="${4:?chunks}"; kind="${5:-transformers}"; s="${6:-0}"
    JOB_ARGS=(--needs gpu:shared --vram "${v}G" "${gpu[@]}" --timeout 2h30m)
    if [[ "$kind" == vllm ]]; then
      chain "syn-roundtrip-$run" "$n" -- labeling/vllm_env.sh python -m synthetic roundtrip --run "$run" \
        --engine-kind vllm --setting "$s" --budget-minutes 110
    else
      chain "syn-roundtrip-$run" "$n" -- ./python -m synthetic roundtrip --run "$run" --engine-kind transformers \
        --setting "$s" --declared-vram-gb "$v" --budget-minutes 110
    fi ;;
  after-text)
    run="${2:?run}"
    exec job start "syn-after-text-$run" --cwd "$here" --mem 16G --cpus 4 --timeout 2h -- \
      bash -c "./python -m synthetic hygiene --run $run && ./python -m synthetic tts-plan --run $run" ;;
  finalize)
    run="${2:?run}"
    exec job start "syn-finalize-$run" --cwd "$here" --mem 16G --cpus 4 --timeout 2h -- \
      ./python -m synthetic finalize --run "$run" ;;
  *)
    sed -n '2,24p' "$0" >&2; exit 2 ;;
esac
