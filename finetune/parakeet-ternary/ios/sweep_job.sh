#!/bin/sh
# One WP5 timing job on the Mac, run INSIDE macguard (everything below happens while the guard's lock is held):
#
#   ios/macguard --rss-cap 4G --timeout 3600 -- sh ios/sweep_job.sh NEED_GB OUTDIR CACHED_CLIP -- RUN ARGS...
#
# 1. Disk: refuses (exit 5) unless `df -k` parses and the data volume has 30 GB (user-approved floor) + NEED_GB
#    free (NEED_GB: the caller's measured headroom: package + Core ML cache estimate). Unreadable readings refuse.
# 2. Purges this binary's Core ML cache directory (~/Library/Caches/parakeet-bench; only parakeet-bench writes
#    there, and guarded jobs are serialized); a failed purge exits 6.
# 3. Run 1: `parakeet-bench run RUN ARGS... --out OUTDIR/arm.jsonl` (the caller passes --pair-c0 and --c0-out
#    for the interleaved C0 baseline). Its load times are "post-purge loads" (whether Core ML specialized the
#    model then is not established without the Instruments cache events, DESIGN.md "Load").
# 4. Run 2: a fresh process loads the arm alone and makes one call on CACHED_CLIP ("subsequent fresh-process
#    load", arm-only phys_footprint): OUTDIR/cached.jsonl.
# 5. Purges the cache again and returns run 1's status (run 2's if run 1 succeeded).
# Mac timings are informational (the Mac is shared).
set -u
FLOOR=30
need=$1 out=$2 cached_clip=$3; shift 3
[ "$1" = "--" ] && shift
ios=$(cd -- "$(dirname -- "$0")" && pwd)
art=/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios
bench=$art/reviewed/parakeet-bench   # installed by build_reviewed.sh
cache="$HOME/Library/Caches/parakeet-bench"
free_kb=$(df -k "$art" 2>/dev/null | awk 'NR == 2 {print $4}')
case "$free_kb" in ''|*[!0-9]*) echo "sweep_job.sh: refusing: cannot read free disk space" >&2; exit 5;; esac
if [ "$free_kb" -lt $(( (FLOOR + need) * 1024 * 1024 )) ]; then
  echo "sweep_job.sh: refusing: $((free_kb / 1048576)) GB free < ${FLOOR} GB floor + ${need} GB" >&2
  exit 5
fi
purge() { rm -rf "$cache" && [ ! -e "$cache" ] || { echo "sweep_job.sh: cache purge failed" >&2; exit 6; }; }
mkdir -p "$out"
purge
echo "cache before run 1: $(du -sk "$cache" 2>/dev/null | cut -f1)KB" > "$out/cache.log"
"$bench" run "$@" --out "$out/arm.jsonl"
s1=$?
echo "cache after run 1: $(du -sk "$cache" 2>/dev/null | cut -f1)KB" >> "$out/cache.log"
# run 2: drop the pairing and the clip selection, keep the arm, one warm-up call on one clip
args=""
skip=0
for a in "$@"; do
  if [ $skip -eq 1 ]; then skip=0; continue; fi
  case "$a" in --pair-c0|--c0-out|--c0-compute-units|--ids|--kinds|--limit|--warmups|--timed|--diag-dir) skip=1; continue;; esac
  args="$args $(printf '%s' "$a" | sed "s/'/'\\\\''/g; s/^/'/; s/\$/'/")"
done
eval "\"\$bench\" run $args --ids \"\$cached_clip\" --warmups 1 --timed 0 --emit-warmups --out \"\$out/cached.jsonl\""
s2=$?
purge
[ $s1 -ne 0 ] && exit $s1
exit $s2
