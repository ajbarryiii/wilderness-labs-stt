#!/bin/sh
# Build the reviewed parakeet-bench on the Mac, run INSIDE macguard (WP7 reviews r1 finding 7, r2 finding 1,
# r3 finding 4):
#
#   ios/macguard --rss-cap 4G --timeout 1800 -- sh ios/build_reviewed.sh
#
# 1. Removes the previous reviewed binary and stamp ($A/reviewed/) first, so a failure below can never leave an
#    older binary certified.
# 2. Disk preflight: 30 GB floor + 3 GB (ios/diskcheck.sh, fail closed).
# 3. Refuses unless the checkout has no modified tracked files (the job scripts run from it).
# 4. Isolated sources: `git archive` of HEAD's ios/bench tree into a fresh scratch directory
#    ($A/scratch/build-reviewed), so only committed, reviewed files are compiler inputs: untracked or ignored
#    .swift files in the checkout (which SwiftPM would pick up) never reach the build.
# 5. `swift build -c release`, then `swift test`, in that directory; each command's own exit status is checked
#    (logs kept in $A/reviewed/build.log, test.log).
# 6. Only after both succeed: installs the binary as $A/reviewed/parakeet-bench and writes
#    $A/reviewed/BUILD_INFO.json (commit, source tree hash, executable SHA-256). The scratch directory is removed
#    on every exit path this shell sees (trap); after a guard kill (SIGKILL) the next run removes it first.
# pipegate_job.sh and sweep_job.sh run only $A/reviewed/parakeet-bench.
# Exit: 0 stamped; 2 modified checkout / no source tree; 5 disk; 7 build failed; 8 tests failed; 9 install/stamp.
set -u
ios=$(cd -- "$(dirname -- "$0")" && pwd)
A=${BUILD_REVIEWED_ARTIFACTS:-/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios}
dest="$A/reviewed"
stamp="$dest/BUILD_INFO.json"
work="$A/scratch/build-reviewed"
rm -f "$stamp" "$dest/parakeet-bench" || exit 9
[ ! -e "$stamp" ] && [ ! -e "$dest/parakeet-bench" ] || exit 9
rm -rf "$work"
trap 'rm -rf "$work"' EXIT
trap 'exit 130' INT TERM HUP
sh "$ios/diskcheck.sh" 3 || exit 5
cd "$ios" || exit 2
dirty=$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')
[ "$dirty" = 0 ] || { echo "build_reviewed.sh: refusing: $dirty modified tracked files" >&2; exit 2; }
commit=$(git rev-parse HEAD) || exit 2
prefix=$(git rev-parse --show-prefix) || exit 2
tree=$(git rev-parse "$commit:${prefix}bench") || exit 2
mkdir -p "$work" "$dest" || exit 9
top=$(git rev-parse --show-toplevel) || exit 2
# from the top level: run in a subdirectory, git archive would restrict the tree to that subdirectory's path
git -C "$top" archive "$tree" | tar -x -C "$work" || exit 2
[ -f "$work/Package.swift" ] || { echo "build_reviewed.sh: archived tree has no Package.swift" >&2; exit 2; }
cd "$work" || exit 7
swift build -c release > "$dest/build.log" 2>&1
s=$?; tail -3 "$dest/build.log"
[ $s -eq 0 ] || { echo "build_reviewed.sh: swift build failed ($s)" >&2; exit 7; }
swift test > "$dest/test.log" 2>&1
s=$?; grep -E "Executed|error:|failed" "$dest/test.log" | tail -3
[ $s -eq 0 ] || { echo "build_reviewed.sh: swift test failed ($s)" >&2; exit 8; }
cp .build/release/parakeet-bench "$dest/parakeet-bench.part" && mv "$dest/parakeet-bench.part" "$dest/parakeet-bench" || exit 9
exe=$(shasum -a 256 "$dest/parakeet-bench" | cut -d' ' -f1)
case "$exe" in [0-9a-f]*) ;; *) exit 9;; esac
printf '{"commit": "%s", "source_tree": "%s", "executable_sha256": "%s", "built": "%s"}\n' \
  "$commit" "$tree" "$exe" "$(date '+%Y-%m-%d %H:%M:%S')" > "$stamp.part" && mv "$stamp.part" "$stamp" || exit 9
cat "$stamp"
