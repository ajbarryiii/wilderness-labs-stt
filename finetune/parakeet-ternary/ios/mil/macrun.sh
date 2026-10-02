#!/bin/sh
# Run WP3 steps on the Mac, each as its own macguard job; a refusal (exit 3: lock held or < 40% free memory)
# is retried every 180 s, at most 40 times. Any other non-zero status stops the sequence.
#
#   cd ios && nohup mil/macrun.sh CAP TIMEOUT_S STEPFILE > LOG 2>&1 &
#
# STEPFILE: one command per line (run from ios/ with pyenv/.venv/bin/python); blank lines and # comments
# are skipped. Every line runs as: ./macguard --rss-cap CAP --timeout TIMEOUT_S -- sh -c "LINE".
set -u
cap=$1 timeout=$2 steps=$3
cd "$(dirname "$0")/.." || exit 2
n=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue;; esac
  n=$((n + 1))
  tries=0
  while :; do
    echo "=== step $n: $line ($(date '+%F %T'))"
    ./macguard --rss-cap "$cap" --timeout "$timeout" -- sh -c "$line" < /dev/null
    status=$?
    [ "$status" -ne 3 ] && break
    tries=$((tries + 1))
    [ "$tries" -ge 40 ] && { echo "=== step $n refused 40 times; stopping"; exit 3; }
    echo "=== step $n refused (exit 3); retrying in 180 s"
    sleep 180
  done
  echo "=== step $n exit $status ($(date '+%F %T'))"
  [ "$status" -eq 0 ] || { echo "=== stopping after step $n"; exit "$status"; }
done < "$steps"
echo "=== all $n steps done ($(date '+%F %T'))"
