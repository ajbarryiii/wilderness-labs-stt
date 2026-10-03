#!/bin/sh
# Run WP3 steps on the Mac, each as its own macguard job wrapping mil/job.sh.
#
#   cd ios && nohup mil/macrun.sh CAP TIMEOUT_S STEPFILE > LOG 2>&1 &
#
# STEPFILE: one shell line per step (run from ios/; `pyenv/.venv/bin/python` runs as the wp3py alias, see
# job.sh). Blank lines and # comments are skipped. A line may start with "@N " to declare N GB of expected
# transient disk use (default 8); job.sh refuses the step (exit 5) unless 30 GB + N GB are free before it
# starts. Each step runs as: ./macguard --rss-cap CAP --timeout TIMEOUT_S -- sh mil/job.sh N "LINE".
#
# All cleanup (per-job TMPDIR, the wp3py/computeplan Core ML caches) happens inside job.sh, i.e. while the
# guard's lock is held; this runner deletes nothing.
#
# Retries: a refusal by macguard (exit 3: lock held or < 40% free memory) is retried every 180 s, at most 40
# times. A guard abort (124) caused by system swap growth or low free memory -- the shared Mac's state, not
# the job's own RSS cap or timeout -- is retried at most twice, after the Mac has been calm (1-minute load < 6
# and >= 50% free memory). Exit 10 (a gate that ran and failed: mil/gates.py, mil/eligibility.py) is recorded
# and the run continues; every other non-zero status stops the run. The guard's limits are never changed.
set -u
cap=$1 timeout=$2 steps=$3
cd "$(dirname "$0")/.." || exit 2
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
n=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|'#'*) continue;; esac
  need=8
  case "$line" in @*) need=${line%% *}; need=${need#@}; line=${line#* };; esac
  n=$((n + 1))
  tries=0
  aborts=0
  while :; do
    echo "=== step $n: $line (need ${need} GB; $(date '+%F %T'))"
    ./macguard --rss-cap "$cap" --timeout "$timeout" -- sh mil/job.sh "$need" "$line" < /dev/null
    status=$?
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
  [ "$status" -eq 10 ] && { echo "=== step $n: gate FAILED (recorded; continuing)"; continue; }
  [ "$status" -eq 0 ] || { echo "=== stopping after step $n"; exit "$status"; }
done < "$steps"
echo "=== all $n steps done ($(date '+%F %T'))"
