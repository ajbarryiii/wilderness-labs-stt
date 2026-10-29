#!/usr/bin/env bash
# Jobs of the read-speech extra source (b0prime plan "Extra sources"), run through the `job` supervisor from
# finetune/relabel. Every job has a memory cap (MEM = the measured peak + 25%).
#
#   cv27/launch.sh tests                          contract tests (foreground)
#   cv27/launch.sh boilerplate                    reproduce the denylist's boilerplate 8-grams (CPU, cpu-heavy)
#   cv27/launch.sh select MEM                     the selection table and the dev sample (CPU, cpu-heavy)
#   cv27/launch.sh probe-decode MEM               decode the 2,000-clip probe into smoke/ (CPU, cpu-heavy)
#   cv27/launch.sh probe-b0 SECONDS VRAM MEM      time B0 on the probe at batch SECONDS (GPU, shared lease)
#   cv27/launch.sh decode N MEM                   N chained decode chunk jobs, ~110 min each (CPU: cpu-heavy and
#                                                 disk-io, since the store is ~0.1 TB of small files on the data disk)
#   cv27/launch.sh dev MEM                        the dev sample's FLAC and manifest (CPU)
#   cv27/launch.sh confirm SECONDS VRAM N MEM     N chained B0 chunk jobs (GPU, shared lease), after the decode chain
#   cv27/launch.sh finalize MEM                   the clip source and report.json (CPU)
#   cv27/launch.sh ted-reject MEM                 the main pool's talk-text reject list (CPU, cpu-heavy)
#
# GPU jobs: a shared lease (--needs gpu:shared --vram LEASE, LEASE = ceil(VRAM) + 2 GB, the plan's probe peak + 2 GB),
# each holding the parakeet GPU lock for its own duration only (the CLI takes it), HF_HUB_OFFLINE=1.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
lease() { awk -v v="$1" 'BEGIN { c = int(v); if (c < v) c++; printf "%dG", c + 2 }'; }

chain() {   # chain NAME N -- command...: N jobs NAME-1..N, each after the previous
  local name="$1" n="$2"; shift 3
  local prev="${CHAIN_AFTER:-}" after
  for i in $(seq 1 "$n"); do
    after=(); [[ -n "$prev" ]] && after=(--after "$prev")
    job start "$name-$i" --cwd "$here" "${after[@]}" "${JOB_ARGS[@]}" -- "$@"
    prev="$name-$i"
  done
}

case "${1:-}" in
  tests)
    exec ./python -m unittest discover -s tests -p 'test_cv27_*.py' -v ;;
  boilerplate)
    exec job start cv27-boilerplate --cwd "$here" --needs cpu-heavy --mem 9G --cpus 7 --timeout 50m --stall 15m -- \
      ./python -m cv27 boilerplate --workers 6 ;;
  select)
    exec job start cv27-select --cwd "$here" --needs cpu-heavy --mem "${2:?MEM}" --cpus 2 --timeout 2h --stall 30m -- \
      ./python -m cv27 select ;;
  probe-decode)
    exec job start cv27-probe-decode --cwd "$here" --needs cpu-heavy --mem "${2:?MEM}" --cpus 9 --timeout 30m \
      --stall 10m -- ./python -m cv27 probe-decode --workers 8 ;;
  probe-b0)
    s="${2:?batch seconds}"; v="${3:?VRAM GB}"; mem="${4:?MEM}"
    exec job start "cv27-probe-b0-${s%.*}" --cwd "$here" --needs gpu:shared --vram "$(lease "$v")" --mem "$mem" \
      --cpus 9 --timeout 40m --stall 15m --env HF_HUB_OFFLINE=1 -- \
      ./python -m cv27 probe-b0 --batch-seconds "$s" --declared-vram-gb "$v" ;;
  decode)
    n="${2:?chunks}"; mem="${3:?MEM}"
    JOB_ARGS=(--needs cpu-heavy,disk-io --mem "$mem" --cpus 9 --timeout 2h15m --stall 20m)
    chain cv27-decode "$n" -- ./python -m cv27 decode --workers 8 --budget-minutes 110 ;;
  dev)
    exec job start cv27-dev --cwd "$here" --mem "${2:?MEM}" --cpus 5 --timeout 30m --stall 10m -- \
      ./python -m cv27 dev --workers 4 ;;
  confirm)
    s="${2:?batch seconds}"; v="${3:?VRAM GB}"; n="${4:?chunks}"; mem="${5:?MEM}"
    JOB_ARGS=(--needs gpu:shared --vram "$(lease "$v")" --mem "$mem" --cpus 9 --timeout 2h30m --stall 30m
              --env HF_HUB_OFFLINE=1)
    chain cv27-confirm "$n" -- ./python -m cv27 confirm --batch-seconds "$s" --declared-vram-gb "$v" \
      --budget-minutes 110 ;;
  finalize)
    exec job start cv27-finalize --cwd "$here" --mem "${2:?MEM}" --cpus 2 --timeout 1h --stall 20m -- \
      ./python -m cv27 finalize ;;
  ted-reject)
    exec job start relabel-ted-reject --cwd "$here" --needs cpu-heavy --mem "${2:?MEM}" --cpus 7 --timeout 1h \
      --stall 20m -- ./python -m pool.tedreject build --workers 6 ;;
  *)
    sed -n '2,20p' "$0"; exit 2 ;;
esac
