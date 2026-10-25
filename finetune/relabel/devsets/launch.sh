#!/usr/bin/env bash
# Commands of the domain dev sets (labeling plan Part A), run through the `job` supervisor.
#
#   devsets/launch.sh tests             contract tests of this part (foreground, CPU, ~1 min)
#   devsets/launch.sh fetch             pinned, verified downloads (network, no GPU)
#   devsets/launch.sh smoke-build N     first N recordings per part into $RELABEL_ROOT/smoke/devsets-build-N (CPU)
#   devsets/launch.sh build             all parts into $RELABEL_ROOT/devsets (CPU, resumable; after fetch)
#   devsets/launch.sh smoke-score M     --limit 16 per part, smoke path (whole-GPU lease + parakeet GPU lock)
#   devsets/launch.sh score M           M in B0 M1 MB1: one GPU job, both sets (whole-GPU lease + GPU lock)
#   devsets/launch.sh evalext-fetch     eval-ext downloads (network; the archive one file at a time)
#   devsets/launch.sh evalext-build P   one eval-ext part (medical-b, lectures-a) per job (CPU, resumable)
#   devsets/launch.sh evalext-score M   eval-ext only, its own output dir, same ami_dev cross-check (GPU)
#   devsets/launch.sh evalstem-fetch    eval-stem downloads (the archive, one file at a time)
#   devsets/launch.sh evalstem-build P  one eval-stem part (stem-chem, stem-math, stem-cs) per job (CPU, resumable)
#   devsets/launch.sh evalstem-score M  eval-stem only, scores/eval-stem/M, same ami_dev cross-check (GPU)
#
# CPU jobs: 16 cores, 24G memory cap, low priority (agent default). GPU jobs: --needs gpu, 36G cap;
# score_domain.py also takes the parakeet GPU lock for its whole life and refuses to start if the lock is
# not free within --lock-wait. HF_HUB_OFFLINE=1 for build and scoring (nothing is downloaded after fetch).
# Roots: finetune/relabel/private/local.env, read by ../python.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
models=(B0 M1 MB1)

check_model() {
  local m="$1"
  for x in "${models[@]}"; do [[ "$x" == "$m" ]] && return 0; done
  echo "unknown model $m (expected one of ${models[*]})" >&2; exit 2
}

case "${1:-}" in
  tests)
    cd "$here" && ./python -m unittest discover -s tests -p "test_devsets_*" -v \
      && ./python -m unittest discover -s tests -p "test_terms_*" -v \
      && ./python -m unittest discover -s tests -p "test_score_domain_*" -v ;;
  fetch)
    exec job start relabel-devsets-fetch --cwd "$here" --mem 8G --cpus 2 --timeout 3h --stall 30m -- \
      ./python -m devsets.build fetch ;;
  smoke-build)
    n="${2:?number of recordings per part}"
    exec job start "relabel-devsets-smoke-build-$n" --cwd "$here" --mem 24G --cpus 16 --timeout 2h --stall 30m \
      --env HF_HUB_OFFLINE=1 -- ./python -m devsets.build build --smoke "$n" --threads 16 ;;
  build)
    exec job start relabel-devsets-build --cwd "$here" --mem 24G --cpus 16 --timeout 14h --stall 45m \
      --env HF_HUB_OFFLINE=1 -- ./python -m devsets.build build --threads 16 ;;
  smoke-score)
    m="${2:?model}"; check_model "$m"
    exec job start "relabel-devsets-smoke-score-$m" --cwd "$here" --needs gpu --mem 36G --timeout 30m --stall 15m \
      --queue-timeout 2h --env HF_HUB_OFFLINE=1 -- ./python score_domain.py --model "$m" --limit 16 --lock-wait 900 ;;
  score)
    m="${2:?model}"; check_model "$m"
    exec job start "relabel-devsets-score-$m" --cwd "$here" --needs gpu --mem 36G --timeout 1h --stall 20m \
      --env HF_HUB_OFFLINE=1 -- ./python score_domain.py --model "$m" --lock-wait 1800 ;;
  evalext-fetch)
    exec job start relabel-evalext-fetch --cwd "$here" --mem 8G --cpus 2 --timeout 4h --stall 30m -- \
      ./python -m devsets.build fetch --parts medical-b lectures-a ;;
  evalext-build)
    p="${2:?part (medical-b or lectures-a)}"
    [[ "$p" == medical-b || "$p" == lectures-a ]] || { echo "unknown eval-ext part $p" >&2; exit 2; }
    exec job start "relabel-evalext-build-$p" --cwd "$here" --mem 24G --cpus 16 --timeout 6h --stall 45m \
      --env HF_HUB_OFFLINE=1 -- ./python -m devsets.build build --parts "$p" --threads 16 ;;
  evalext-score)
    m="${2:?model}"; check_model "$m"
    exec job start "relabel-evalext-score-$m" --cwd "$here" --needs gpu --mem 36G --timeout 1h --stall 20m \
      --env HF_HUB_OFFLINE=1 -- ./python score_domain.py --model "$m" --sets eval-ext --lock-wait 1800 ;;
  evalstem-fetch)
    exec job start relabel-evalstem-fetch --cwd "$here" --mem 8G --cpus 2 --timeout 4h --stall 30m -- \
      ./python -m devsets.build fetch --parts stem-chem stem-math stem-cs ;;
  evalstem-build)
    p="${2:?part (stem-chem, stem-math or stem-cs)}"
    [[ "$p" == stem-chem || "$p" == stem-math || "$p" == stem-cs ]] || { echo "unknown eval-stem part $p" >&2; exit 2; }
    exec job start "relabel-evalstem-build-$p" --cwd "$here" --after relabel-evalstem-fetch --mem 24G --cpus 16 \
      --timeout 4h --stall 45m --env HF_HUB_OFFLINE=1 -- ./python -m devsets.build build --parts "$p" --threads 16 ;;
  evalstem-score)
    m="${2:?model}"; check_model "$m"
    exec job start "relabel-evalstem-score-$m" --cwd "$here" --needs gpu --mem 36G --timeout 1h --stall 20m \
      --env HF_HUB_OFFLINE=1 -- ./python score_domain.py --model "$m" --sets eval-stem --lock-wait 1800 ;;
  *)
    sed -n '2,20p' "$0" >&2; exit 2 ;;
esac
