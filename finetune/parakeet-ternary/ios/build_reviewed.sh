#!/bin/sh
# Build the reviewed parakeet-bench on the Mac, run INSIDE macguard (WP7 review r1 finding 7):
#
#   ios/macguard --rss-cap 4G --timeout 1800 -- sh ios/build_reviewed.sh
#
# Refuses unless the checkout has no modified tracked files. Builds release, runs the unit tests, and writes
# bench/.build/release/BUILD_INFO.json (commit, executable SHA-256, time). pipegate_job.sh requires that file to
# describe the binary it runs and the checkout it runs in, so a gate never runs a binary built from other sources.
set -eu
ios=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$ios"
dirty=$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')
[ "$dirty" = 0 ] || { echo "build_reviewed.sh: refusing: $dirty modified tracked files" >&2; exit 2; }
commit=$(git rev-parse HEAD)
cd bench
swift build -c release 2>&1 | tail -3
swift test 2>&1 | grep -E "Executed|error|failed" | tail -3
exe=$(shasum -a 256 .build/release/parakeet-bench | cut -d' ' -f1)
printf '{"commit": "%s", "executable_sha256": "%s", "built": "%s"}\n' "$commit" "$exe" "$(date '+%Y-%m-%d %H:%M:%S')" \
  > .build/release/BUILD_INFO.json
cat .build/release/BUILD_INFO.json
