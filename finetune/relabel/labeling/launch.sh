#!/usr/bin/env bash
# Jobs of stage 2 Part C (labeling plan), run through the `job` supervisor from finetune/relabel.
#
#   labeling/launch.sh tests                     unit and contract tests (foreground)
#   labeling/launch.sh vllm-install              the isolated vLLM environment (network, no GPU)
#   labeling/launch.sh candidate                 the stream configuration without W (CPU)
#   labeling/launch.sh replay-probes             C1 for 15 min at W = 8, 12, 16, one after another (CPU)
#   labeling/launch.sh choose                    the worker-sizing rule over the probes (foreground)
#   labeling/launch.sh pin W                     pin the configuration at W (foreground)
#   labeling/launch.sh c1 N                      N chained C1 chunk jobs of at most ~110 min (CPU, no GPU lease)
#   labeling/launch.sh gpu-probe ENGINE I VRAM [MEM [AFTER]]  GPU probe of setting I (0-2) of ENGINE (vllm|transformers);
#                                                MEM: host memory cap (default 36G; use the measured peak + 25%),
#                                                AFTER: a job that must succeed first (chain the six probes)
#   labeling/launch.sh consistency ENGINE I VRAM the consistency gate at that setting (prepares the dev shards first)
#   labeling/launch.sh equivalence V T VRAM      the equivalence check at vLLM setting V, transformers setting T
#   labeling/launch.sh c2 RUN VRAM N             N chained C2 chunk jobs for labels/<T*>/RUN (after c2-init)
#   labeling/launch.sh gate RUN CONS_JSON [EQ_JSON]  the gate to stage 3 (CPU and network: independent replay;
#                                                re-hashes all C1 audio); EQ_JSON is required for a vLLM run
#   labeling/launch.sh smoke-vllm VRAM           vLLM on 10 stage 1 dev clips per set, compared with stage 1 (smoke)
#
# GPU jobs: a shared lease (--needs gpu:shared --vram LEASE), 36G memory cap, low priority; each job
# takes the parakeet GPU lock for its own duration only, so exclusive jobs get the GPU between chunks. Nothing is
# downloaded under the lock (HF_HUB_OFFLINE=1). C1 and the CPU checks take no GPU lease and no lock.
# LEASE = ceil(VRAM) + 1 GB: the supervisor's lease fraction is VRAM over the GPU's nominal size, while the
# engines check VRAM over the memory torch reports, which is smaller (31.3 of 32 GiB here), so a lease of
# exactly VRAM is refused (engines.vram_fraction).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
gpu=(--mem 36G --stall 30m --env HF_HUB_OFFLINE=1)
lease() { awk -v v="$1" 'BEGIN { c = int(v); if (c < v) c++; printf "%dG", c + 1 }'; }

case "${1:-}" in
  tests)
    exec ./python -m unittest discover -s tests -p 'test_labeling*.py' -v ;;
  vllm-install)
    exec job start relabel-vllm-install --cwd "$here" --mem 12G --cpus 4 --timeout 1h -- labeling/vllm_env.sh install ;;
  candidate)
    exec job start relabel-labeling-candidate --cwd "$here" --needs cpu-heavy --mem 16G --timeout 1h -- \
      ./python -m labeling candidate ;;
  replay-probes)
    prev=""
    for w in 8 12 16; do
      after=(); [[ -n "$prev" ]] && after=(--after "$prev")
      job start "relabel-replay-probe-w$w" --cwd "$here" "${after[@]}" --mem 36G --cpus $((w + 6)) --timeout 40m \
        --stall 10m -- ./python -m labeling replay-probe --workers "$w"
      prev="relabel-replay-probe-w$w"
    done ;;
  choose)
    exec ./python -m labeling replay-choose ;;
  pin)
    exec ./python -m labeling pin --workers "${2:?W}" ;;
  c1)
    n="${2:?number of chunks}"
    w="$(./python -c 'from labeling.__main__ import pinned; print(pinned()["workers"])')"
    [[ -n "$w" ]] || { echo "no pinned configuration" >&2; exit 2; }
    prev=""
    for i in $(seq 1 "$n"); do
      after=(); [[ -n "$prev" ]] && after=(--after "$prev")
      job start "relabel-c1-$i" --cwd "$here" "${after[@]}" --mem 36G --cpus $((w + 6)) --timeout 2h30m --stall 20m -- \
        ./python -m labeling replay --budget-minutes 110
      prev="relabel-c1-$i"
    done ;;
  gpu-probe)
    e="${2:?engine}"; i="${3:?setting}"; v="${4:?VRAM GB}"; m="${5:-36G}"
    after=(); [[ -n "${6:-}" ]] && after=(--after "$6")
    exec job start "relabel-gpu-probe-$e-$i" --cwd "$here" "${after[@]}" --needs gpu:shared --vram "$(lease "$v")" \
      --mem "$m" --stall 30m --env HF_HUB_OFFLINE=1 --timeout 1h -- \
      ./python -m labeling gpu-probe --engine "$e" --setting "$i" --declared-vram-gb "$v" ;;
  consistency)
    e="${2:?engine}"; i="${3:?setting}"; v="${4:?VRAM GB}"
    job start relabel-consistency-prepare --cwd "$here" --mem 16G --timeout 1h -- ./python -m labeling consistency-prepare
    exec job start "relabel-consistency-$e-$i" --cwd "$here" --after relabel-consistency-prepare --needs gpu:shared \
      --vram "$(lease "$v")" "${gpu[@]}" --timeout 2h -- ./python -m labeling consistency --engine "$e" --setting "$i" \
      --declared-vram-gb "$v" ;;
  equivalence)
    vs="${2:?vllm setting}"; ts="${3:?transformers setting}"; v="${4:?VRAM GB}"
    job start relabel-equivalence-prepare --cwd "$here" --mem 16G --timeout 2h -- ./python -m labeling equivalence-prepare
    exec job start "relabel-equivalence" --cwd "$here" --after relabel-equivalence-prepare --needs gpu:shared \
      --vram "$(lease "$v")" "${gpu[@]}" --timeout 3h -- ./python -m labeling equivalence --vllm-setting "$vs" \
      --tf-setting "$ts" --declared-vram-gb "$v" ;;
  c2)
    run="${2:?run name}"; v="${3:?VRAM GB}"; n="${4:?number of chunks}"
    rd="$(./python -c 'import sys; from labeling import layout; print(layout.labels_dir(layout.TEACHER, sys.argv[1]))' "$run")"
    [[ -f "$rd/run.json" ]] || { echo "no $rd/run.json (run: ./python -m labeling c2-init ...)" >&2; exit 2; }
    prev=""
    for i in $(seq 1 "$n"); do
      after=(); [[ -n "$prev" ]] && after=(--after "$prev")
      job start "relabel-c2-$run-$i" --cwd "$here" "${after[@]}" --needs gpu:shared --vram "$(lease "$v")" "${gpu[@]}" \
        --timeout 2h30m -- ./python -m labeling.c2 chunk --run-dir "$rd" --budget-minutes 110
      prev="relabel-c2-$run-$i"
    done ;;
  gate)
    run="${2:?run name}"; cons="${3:?consistency result json}"
    eq=(); [[ -n "${4:-}" ]] && eq=(--equivalence "$4")
    exec job start "relabel-gate-$run" --cwd "$here" --mem 36G --cpus 12 --timeout 6h -- \
      ./python -m labeling gate --run "$run" --consistency "$cons" "${eq[@]}" ;;
  smoke-vllm)
    v="${2:?VRAM GB}"
    d="$(./python -c 'from labeling import layout; print(layout.smoke_dir("consistency-limit10"))')"
    job start relabel-smoke-vllm-prepare --cwd "$here" --mem 8G --timeout 20m -- \
      ./python -m labeling consistency-prepare --limit 10 --dir "$d"
    exec job start relabel-smoke-vllm --cwd "$here" --after relabel-smoke-vllm-prepare --needs gpu:shared \
      --vram "$(lease "$v")" "${gpu[@]}" --timeout 25m -- ./python -m labeling consistency --engine vllm --setting 0 \
      --declared-vram-gb "$v" --group-shards 1 --dir "$d" ;;
  *)
    sed -n '2,20p' "$0" >&2; exit 2 ;;
esac
