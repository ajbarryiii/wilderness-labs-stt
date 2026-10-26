#!/usr/bin/env bash
# Commands of the teacher diagnostic (diagnostic plan), run through the `job` supervisor.
#
#   ./launch.sh tests              CPU unit and contract tests (in the foreground, ~20 s)
#   ./launch.sh prefetch           download the three pinned teachers (no GPU, no lock)
#   ./launch.sh smoke NAME         --limit 8 --shortest 2 per set, smoke path only (whole-GPU lease + lock)
#   ./launch.sh smoke-cpu NAME     the same on CPU, no GPU and no lock (adapter/schema check when the GPU is busy)
#   ./launch.sh full               Step 1: the three teachers one after another (whole-GPU lease + lock each)
#   ./launch.sh analyze            Step 2: report.json, CPU only, once the three full runs have ended
#
# GPU runs: `--needs gpu` (whole-GPU lease), 36G memory cap, low priority (agent default), 3 h runtime cap
# per teacher; score_teacher.py also takes the parakeet GPU lock for its whole life and refuses to start
# if the lock is not free within --lock-wait. HF_HUB_OFFLINE=1: nothing is downloaded under the lock
# (run `prefetch` first). canary-qwen-2.5b needs RELABEL_DEPS=canary-qwen-2.5b (peft, accelerate, psutil
# under $RELABEL_ROOT/deps/canary-qwen-2.5b). Roots: private/local.env, read by ./python (local.env.example).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
teachers=(qwen3-asr-1.7b granite-speech-5.0-2b canary-qwen-2.5b)

deps_env() {  # extra --env for a teacher
  if [[ "$1" == canary-qwen-2.5b ]]; then echo "--env RELABEL_DEPS=canary-qwen-2.5b"; fi
}

case "${1:-}" in
  tests)
    cd "$here" && exec ./python -m unittest discover -s tests -v ;;
  prefetch)
    exec job start relabel-prefetch --cwd "$here" --mem 8G --timeout 2h --stall 20m -- ./python prefetch.py ;;
  smoke)
    t="${2:?teacher name}"
    # shellcheck disable=SC2046
    exec job start "relabel-smoke-$t" --cwd "$here" --needs gpu --mem 36G --timeout 20m --stall 10m \
      --queue-timeout 30m --env HF_HUB_OFFLINE=1 $(deps_env "$t") -- \
      ./python score_teacher.py --teacher "$t" --limit 8 --shortest 2 --lock-wait 600 --min-available-gb 10 ;;
  smoke-cpu)
    t="${2:?teacher name}"
    # shellcheck disable=SC2046
    exec job start "relabel-smoke-cpu-$t" --cwd "$here" --needs cpu-heavy --mem 36G --cpus 8 --timeout 30m \
      --stall 15m --env HF_HUB_OFFLINE=1 --env OMP_NUM_THREADS=8 $(deps_env "$t") -- \
      ./python score_teacher.py --teacher "$t" --limit 8 --shortest 2 --device cpu --min-available-gb 10 ;;
  full)
    # Independent runs (no --after: one teacher's failure must not skip the others); the whole-GPU lease
    # and the GPU lock keep them one at a time.
    for t in "${teachers[@]}"; do
      # shellcheck disable=SC2046
      job start "relabel-score-$t" --cwd "$here" --needs gpu --mem 36G --timeout 3h --stall 20m \
        --env HF_HUB_OFFLINE=1 $(deps_env "$t") -- \
        ./python score_teacher.py --teacher "$t" --lock-wait 1800
    done ;;
  analyze)
    # After the three full runs have ended (`job wait relabel-score-<teacher>`); a teacher whose run did not
    # complete is reported as not scored by analyze.py.
    exec job start relabel-analyze --cwd "$here" --needs cpu-heavy --mem 16G --timeout 1h -- ./python analyze.py ;;
  *)
    sed -n '2,15p' "$0" >&2; exit 2 ;;
esac
