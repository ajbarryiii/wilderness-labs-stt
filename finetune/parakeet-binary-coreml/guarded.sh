#!/bin/sh
# Mac job wrapper: guarded.sh TIMEOUT_S CMD ARGS...
# Runs CMD from this directory under the upstream macguard (6 GB RSS cap unless RSS_CAP is set, 4 threads),
# with TMPDIR in the kea-b1 artifact area (coremltools' temporary packages) and the keab1py interpreter
# alias on PATH as $PY (Core ML caches its device-specialized copies per process name; only that cache is
# purged by screen.py). macguard refusals (exit 3: lock held by another guarded job, or < 40% free memory)
# are retried every 30 s for up to 2 h; other jobs are never stopped.
set -u
IOS=/Users/ajbarry/workspace/github.com/wilderness-labs-stt/finetune/parakeet-ternary/ios
K=/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/kea-b1
PY=$K/pyenv/bin/keab1py
export PY
TMPDIR=$K/tmp/job-$$
export TMPDIR
mkdir -p "$TMPDIR"
cd -- "$(dirname -- "$0")" || exit 2
timeout_s=$1
shift
tries=0
while :; do
  "$IOS/macguard" --rss-cap "${RSS_CAP:-6G}" --timeout "$timeout_s" --threads 4 -- "$@"
  status=$?
  [ "$status" -ne 3 ] && break
  tries=$((tries + 1))
  [ "$tries" -ge 240 ] && break
  echo "guarded.sh: macguard refused (try $tries); retrying in 30 s" >&2
  sleep 30
done
rm -rf "$TMPDIR"
exit "$status"
