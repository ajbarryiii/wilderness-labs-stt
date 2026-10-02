#!/bin/sh
# Run WP3 steps on the Mac, each as its own macguard job; a refusal (exit 3: lock held or < 40% free memory)
# is retried every 180 s, at most 40 times. Any other non-zero status stops the sequence.
#
#   cd ios && nohup mil/macrun.sh CAP TIMEOUT_S STEPFILE > LOG 2>&1 &
#
# STEPFILE: one command per line (run from ios/ with pyenv/.venv/bin/python); blank lines and # comments
# are skipped. Every line runs as: ./macguard --rss-cap CAP --timeout TIMEOUT_S -- sh -c "LINE".
#
# Disk: Core ML keeps device-specialized copies of every model a Python process loads in
# ~/Library/Caches/python/com.apple.e5rt.e5bundlecache (up to several GB per load; 28 GB after the first
# WP3 runs). After each step the entries created during it are deleted (guarded jobs are serialized by
# macguard's lock, so they are this step's); the run stops if the data volume has less than 60 GB free.
set -u
cap=$1 timeout=$2 steps=$3
cd "$(dirname "$0")/.." || exit 2
cache="$HOME/Library/Caches/python/com.apple.e5rt.e5bundlecache"
marker="${TMPDIR:-/tmp}/macrun-marker.$$"
n=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue;; esac
  n=$((n + 1))
  tries=0
  while :; do
    echo "=== step $n: $line ($(date '+%F %T'))"
    touch "$marker"
    ./macguard --rss-cap "$cap" --timeout "$timeout" -- sh -c "$line" < /dev/null
    status=$?
    [ -d "$cache" ] && find "$cache" -mindepth 2 -maxdepth 2 -newer "$marker" -exec rm -rf {} +
    [ "$status" -ne 3 ] && break
    tries=$((tries + 1))
    [ "$tries" -ge 40 ] && { echo "=== step $n refused 40 times; stopping"; exit 3; }
    echo "=== step $n refused (exit 3); retrying in 180 s"
    sleep 180
  done
  echo "=== step $n exit $status ($(date '+%F %T'))"
  [ "$status" -eq 0 ] || { echo "=== stopping after step $n"; exit "$status"; }
  free_gb=$(df -g "$HOME" | awk 'NR == 2 {print $4}')
  [ "$free_gb" -ge 60 ] || { echo "=== stopping after step $n: ${free_gb} GB free < 60"; exit 4; }
done < "$steps"
rm -f "$marker"
echo "=== all $n steps done ($(date '+%F %T'))"
