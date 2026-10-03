#!/bin/sh
# One deployed-pipeline gate job on the Mac, run INSIDE macguard (WP7):
#
#   ios/macguard --rss-cap CAP --timeout S -- sh ios/pipegate_job.sh NEED_GB OUTDIR -- GATE ARGS...
#
# 1. Disk: refuses (exit 5) unless `df -k` parses and the data volume has 30 GB + NEED_GB free (fail closed:
#    an unreadable or non-numeric reading refuses too). NEED_GB is the caller's measured headroom for this arm
#    (package already on disk + its Core ML cache estimate).
# 2. Purges this binary's Core ML cache (~/Library/Caches/parakeet-bench) before and after; a failed purge exits 6.
# 3. `parakeet-bench gate GATE ARGS... --out OUTDIR`, then `python -m pipegate evaluate --gate-dir OUTDIR`
#    (exit 10 = a gate condition failed; the evaluation is written either way). Returns the first nonzero status.
set -u
FLOOR=30
need=$1 out=$2; shift 2
[ "$1" = "--" ] && shift
ios=$(cd -- "$(dirname -- "$0")" && pwd)
art=/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios
bench=$ios/bench/.build/release/parakeet-bench
cache="$HOME/Library/Caches/parakeet-bench"
free_kb=$(df -k "$art" 2>/dev/null | awk 'NR == 2 {print $4}')
case "$free_kb" in ''|*[!0-9]*) echo "pipegate_job.sh: refusing: cannot read free disk space" >&2; exit 5;; esac
if [ "$free_kb" -lt $(( (FLOOR + need) * 1024 * 1024 )) ]; then
  echo "pipegate_job.sh: refusing: $((free_kb / 1048576)) GB free < ${FLOOR} GB floor + ${need} GB" >&2
  exit 5
fi
purge() { rm -rf "$cache" && [ ! -e "$cache" ] || { echo "pipegate_job.sh: cache purge failed" >&2; exit 6; }; }
purge
mkdir -p "$out"
"$bench" gate "$@" --out "$out"
s1=$?
echo "cache after gate: $(du -sk "$cache" 2>/dev/null | cut -f1)KB" > "$out/cache.log"
s2=0
if [ $s1 -eq 0 ]; then
  (cd "$ios" && pyenv/.venv/bin/python -m pipegate evaluate --gate-dir "$out")
  s2=$?
fi
purge
[ $s1 -ne 0 ] && exit $s1
exit $s2
