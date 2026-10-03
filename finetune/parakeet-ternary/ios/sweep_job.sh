#!/bin/sh
# One WP5 timing job on the Mac, run INSIDE macguard (everything below happens while the guard's lock is held):
#
#   ios/macguard --rss-cap 4G --timeout 3600 -- sh ios/sweep_job.sh NEED_GB OUTDIR CACHED_CLIP -- RUN ARGS...
#
# 1. Disk: refuses (exit 5) unless the data volume has 30 GB (user-approved floor) + NEED_GB free.
# 2. Purges the Core ML device-specialization cache of this binary (~/Library/Caches/parakeet-bench; only
#    parakeet-bench writes there, and guarded jobs are serialized), so the first load below is uncached.
# 3. Run 1: `parakeet-bench run RUN ARGS... --out OUTDIR/arm.jsonl` (the caller passes --pair-c0 and --c0-out
#    for the interleaved C0 baseline; the models load uncached: "first load").
# 4. Run 2: a fresh process loads the arm alone from the now-populated cache and makes one call on CACHED_CLIP
#    ("fresh-process cached load", arm-only phys_footprint): OUTDIR/cached.jsonl.
# 5. Purges the cache again and returns run 1's status (run 2's if run 1 succeeded).
# Mac timings are informational (the Mac is shared).
set -u
FLOOR=30
need=$1 out=$2 cached_clip=$3; shift 3
[ "$1" = "--" ] && shift
ios=$(cd -- "$(dirname -- "$0")" && pwd)
art=/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios
bench=$ios/bench/.build/release/parakeet-bench
cache="$HOME/Library/Caches/parakeet-bench"
free_gb=$(df -g "$art" | awk 'NR == 2 {print $4}')
if [ "$free_gb" -lt $((FLOOR + need)) ]; then
  echo "sweep_job.sh: refusing: ${free_gb} GB free < ${FLOOR} GB floor + ${need} GB" >&2
  exit 5
fi
mkdir -p "$out"
rm -rf "$cache"
echo "cache before run 1: $(du -sk "$cache" 2>/dev/null | cut -f1)KB" > "$out/cache.log"
"$bench" run "$@" --out "$out/arm.jsonl"
s1=$?
echo "cache after run 1: $(du -sk "$cache" 2>/dev/null | cut -f1)KB" >> "$out/cache.log"
# run 2: drop the pairing and the clip selection, keep the arm, one warm-up call on one clip
args=""
skip=0
for a in "$@"; do
  if [ $skip -eq 1 ]; then skip=0; continue; fi
  case "$a" in --pair-c0|--c0-out|--ids|--kinds|--limit|--warmups|--timed|--diag-dir) skip=1; continue;; esac
  args="$args $(printf '%s' "$a" | sed "s/'/'\\\\''/g; s/^/'/; s/\$/'/")"
done
eval "\"\$bench\" run $args --ids \"\$cached_clip\" --warmups 1 --timed 0 --emit-warmups --out \"\$out/cached.jsonl\""
s2=$?
rm -rf "$cache"
[ $s1 -ne 0 ] && exit $s1
exit $s2
