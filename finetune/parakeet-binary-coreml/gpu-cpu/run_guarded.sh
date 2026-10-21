#!/bin/sh
# Run one command under macguard on the shared Mac, waiting while macguard refuses (exit 3: lock held by another
# guarded job, or free memory below 40%). Never touches anyone else's jobs.
# Usage: [MG_EXTRA="--threads 8"] run_guarded.sh RSS_CAP TIMEOUT_S -- CMD ARGS...
set -u
cap=$1; tmo=$2; shift 2
[ "$1" = "--" ] && shift
MG=/Users/ajbarry/workspace/github.com/wilderness-labs-stt/finetune/parakeet-ternary/ios/macguard
tries=0
while :; do
  # shellcheck disable=SC2086
  "$MG" --rss-cap "$cap" --timeout "$tmo" ${MG_EXTRA:-} -- "$@"
  st=$?
  [ "$st" -ne 3 ] && exit "$st"
  tries=$((tries + 1))
  [ "$tries" -ge 120 ] && { echo "run_guarded: still refused after $tries tries" >&2; exit 3; }
  echo "run_guarded: macguard refused (try $tries), waiting 30 s" >&2
  sleep 30
done
