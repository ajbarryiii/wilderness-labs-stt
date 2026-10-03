#!/bin/sh
# Mocked tests of build_reviewed.sh and diskcheck.sh (WP7 reviews r2 findings 1, 6; r3 finding 4); runs anywhere.
# Fake `swift`, `shasum` and `df` on PATH; a throwaway git checkout holds copies of the two scripts and a bench tree.
#   sh ios/tests/build_reviewed_tests.sh
set -u
here=$(cd -- "$(dirname -- "$0")/.." && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
mkdir -p "$W/repo/ios/bench/Sources/BenchCore" "$W/bin" "$W/art"
cp "$here/build_reviewed.sh" "$here/diskcheck.sh" "$W/repo/ios/"
echo t > "$W/repo/ios/tracked.txt"
echo "// reviewed" > "$W/repo/ios/bench/Sources/BenchCore/Reviewed.swift"; echo "// package" > "$W/repo/ios/bench/Package.swift"
(cd "$W/repo" && git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -qm init)
cat > "$W/bin/swift" <<'X'
#!/bin/sh
case "$1" in build) find . -name '*.swift' | sort > "$SWIFT_INPUTS"; [ -n "${FAIL_BUILD:-}" ] && exit 42
  mkdir -p .build/release; echo bin > .build/release/parakeet-bench; echo built;;
test) [ -n "${FAIL_TEST:-}" ] && { echo "error: boom"; exit 42; }; echo "Executed 8 tests, with 0 failures";; esac
X
cat > "$W/bin/shasum" <<'X'
#!/bin/sh
echo "abcdef0123  $3"
X
cat > "$W/bin/df" <<'X'
#!/bin/sh
echo "Filesystem 1K-blocks Used Available"; echo "fake 1 1 ${FREE_KB:-104857600}"
X
chmod +x "$W/bin/"*
export PATH="$W/bin:$PATH" DISKCHECK_PATH="$W" BUILD_REVIEWED_ARTIFACTS="$W/art" SWIFT_INPUTS="$W/inputs.txt"
S="$W/art/reviewed/BUILD_INFO.json"
FAILS=0
check() { if [ "$2" = "$3" ]; then echo "PASS $1"; else echo "FAIL $1: expected '$2', got '$3'"; FAILS=$((FAILS+1)); fi; }
stamp() { [ -f "$S" ] && [ -f "$W/art/reviewed/parakeet-bench" ] && echo yes || echo no; }
run() { sh "$W/repo/ios/build_reviewed.sh" > /dev/null 2>&1; echo $?; }
check "1 clean build: stamped, scratch removed" "0 yes no" "$(run) $(stamp) $(test -e "$W/art/scratch/build-reviewed" && echo yes || echo no)"
check "2 build failure: exit 7, previous stamp and binary removed" "7 no" "$(FAIL_BUILD=1 run) $(stamp)"
run > /dev/null
check "3 test failure: exit 8, no stamp" "8 no" "$(FAIL_TEST=1 run) $(stamp)"
run > /dev/null; echo x >> "$W/repo/ios/tracked.txt"
check "4 modified checkout: exit 2, no stamp" "2 no" "$(run) $(stamp)"
(cd "$W/repo" && git checkout -q -- ios/tracked.txt); run > /dev/null
check "5 low disk (32 GB < 30 + 3): exit 5, no stamp" "5 no" "$(FREE_KB=33554432 run) $(stamp)"
check "6 unreadable df: exit 5" "5" "$(FREE_KB=abc run)"
check "7 diskcheck: 40 GB >= 30 + 9" "0" "$(FREE_KB=41943040 sh "$W/repo/ios/diskcheck.sh" 9 > /dev/null 2>&1; echo $?)"
# 8: untracked (and ignored) Swift files in the checkout are not compiler inputs: only the committed tree is built
echo "// unreviewed" > "$W/repo/ios/bench/Sources/BenchCore/Untracked.swift"
mkdir -p "$W/repo/ios/bench/.build"; echo "// ignored" > "$W/repo/ios/bench/Sources/BenchCore/Ignored.swift"
echo "Ignored.swift" > "$W/repo/ios/bench/Sources/BenchCore/.gitignore"
check "8 untracked/ignored .swift excluded from the build" "0 ./Package.swift ./Sources/BenchCore/Reviewed.swift" "$(run) $(tr '\n' ' ' < "$W/inputs.txt" | sed 's/ $//')"
echo "failed checks: $FAILS"
exit $FAILS
