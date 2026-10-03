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
# macguard's lock, so they are this step's), as are coremltools' temporary *.mlpackage directories in the
# user temp dir; the run stops if the data volume has less than 30 GB free (floor lowered from 60 GB with
# the user's approval, 2026-10-02).
#
# A guard abort (exit 124) caused by system swap growth or low free memory -- i.e. by the whole shared Mac's
# state, not by the job's own RSS cap or timeout -- is retried at most twice, after the Mac has been calm
# (1-minute load < 6 and >= 50% free memory) for one check; every other abort stops the run. The guard's
# limits are never changed.
set -u
cap=$1 timeout=$2 steps=$3
cd "$(dirname "$0")/.." || exit 2
cache="$HOME/Library/Caches/python/com.apple.e5rt.e5bundlecache"
tmpd=$(getconf DARWIN_USER_TEMP_DIR 2>/dev/null || echo "${TMPDIR:-/tmp}")
floor=30
guardlog="${MACGUARD_DIR:-/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios}/logs/macguard.log"
calm() {  # wait (<= 2 h) until load1 < 6 and free memory >= 50%
  i=0
  while [ $i -lt 60 ]; do
    load=$(sysctl -n vm.loadavg | awk '{print int($2)}')
    free=$(memory_pressure -Q | awk -F': ' '/free percentage/ {gsub("%", "", $2); print $2}')
    [ "$load" -lt 6 ] && [ "$free" -ge 50 ] && return 0
    sleep 120; i=$((i + 1))
  done
  return 1
}
marker="${TMPDIR:-/tmp}/macrun-marker.$$"
n=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue;; esac
  n=$((n + 1))
  tries=0
  aborts=0
  while :; do
    echo "=== step $n: $line ($(date '+%F %T'))"
    touch "$marker"
    ./macguard --rss-cap "$cap" --timeout "$timeout" -- sh -c "$line" < /dev/null
    status=$?
    [ -d "$cache" ] && find "$cache" -mindepth 2 -maxdepth 2 -newer "$marker" -exec rm -rf {} +
    [ -d "$tmpd" ] && find "$tmpd" -maxdepth 2 -name '*.mlpackage' -newer "$marker" -prune -exec rm -rf {} +
    if [ "$status" -eq 124 ] && [ "$aborts" -lt 2 ] && tail -40 "$guardlog" | grep "end: status 124" | tail -1 | grep -q "swap grew\|free memory"; then
      aborts=$((aborts + 1))
      echo "=== step $n aborted by the guard on system memory (retry $aborts of 2 once the Mac is calm)"
      calm || { echo "=== step $n: the Mac did not calm down within 2 h; stopping"; exit 124; }
      continue
    fi
    [ "$status" -ne 3 ] && break
    tries=$((tries + 1))
    [ "$tries" -ge 40 ] && { echo "=== step $n refused 40 times; stopping"; exit 3; }
    echo "=== step $n refused (exit 3); retrying in 180 s"
    sleep 180
  done
  echo "=== step $n exit $status ($(date '+%F %T'))"
  [ "$status" -eq 0 ] || { echo "=== stopping after step $n"; exit "$status"; }
  free_gb=$(df -g "$HOME" | awk 'NR == 2 {print $4}')
  [ "$free_gb" -ge "$floor" ] || { echo "=== stopping after step $n: ${free_gb} GB free < $floor"; exit 4; }
done < "$steps"
rm -f "$marker"
echo "=== all $n steps done ($(date '+%F %T'))"
