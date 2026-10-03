#!/bin/sh
# Build the reviewed parakeet-bench on the Mac, run INSIDE macguard (WP7 review r1 finding 7, r2 finding 1):
#
#   ios/macguard --rss-cap 4G --timeout 1800 -- sh ios/build_reviewed.sh
#
# 1. Removes any previous stamp (bench/.build/release/BUILD_INFO.json) first, so a failure below can never leave
#    an older binary certified.
# 2. Disk preflight: 30 GB floor + 3 GB (release and test builds; ios/diskcheck.sh, fail closed).
# 3. Refuses unless the checkout has no modified tracked files.
# 4. `swift build -c release`, then `swift test`; each command's own exit status is checked (no pipeline hides
#    it; full logs in bench/.build/build_reviewed.{build,test}.log).
# 5. Only after both succeed: writes the stamp (commit, executable SHA-256, time). pipegate_job.sh requires it to
#    describe the binary it runs and the checkout it runs in.
# Exit: 0 stamped; 2 modified checkout; 5 disk; 7 build failed; 8 tests failed; 9 stamp could not be written.
set -u
ios=$(cd -- "$(dirname -- "$0")" && pwd)
stamp="$ios/bench/.build/release/BUILD_INFO.json"
rm -f "$stamp" || exit 9
[ ! -e "$stamp" ] || exit 9
sh "$ios/diskcheck.sh" 3 || exit 5
cd "$ios" || exit 2
dirty=$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')
[ "$dirty" = 0 ] || { echo "build_reviewed.sh: refusing: $dirty modified tracked files" >&2; exit 2; }
commit=$(git rev-parse HEAD) || exit 2
cd bench || exit 7
mkdir -p .build
swift build -c release > .build/build_reviewed.build.log 2>&1
s=$?; tail -3 .build/build_reviewed.build.log
[ $s -eq 0 ] || { echo "build_reviewed.sh: swift build failed ($s)" >&2; exit 7; }
swift test > .build/build_reviewed.test.log 2>&1
s=$?; grep -E "Executed|error:|failed" .build/build_reviewed.test.log | tail -3
[ $s -eq 0 ] || { echo "build_reviewed.sh: swift test failed ($s)" >&2; exit 8; }
exe=$(shasum -a 256 .build/release/parakeet-bench | cut -d' ' -f1)
case "$exe" in [0-9a-f]*) ;; *) exit 9;; esac
printf '{"commit": "%s", "executable_sha256": "%s", "built": "%s"}\n' "$commit" "$exe" "$(date '+%Y-%m-%d %H:%M:%S')" \
  > "$stamp.part" && mv "$stamp.part" "$stamp" || exit 9
cat "$stamp"
