#!/bin/sh
# Mocked tests of build_reviewed.sh and diskcheck.sh (WP7 review r2 findings 1 and 6); runs anywhere (Linux or Mac).
# Fake `swift`, `shasum` and `df` on PATH; a throwaway git checkout holds copies of the two scripts.
#   sh ios/tests/build_reviewed_tests.sh
set -u
here=$(cd -- "$(dirname -- "$0")/.." && pwd)
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
mkdir -p "$W/repo/ios/bench" "$W/bin"
cp "$here/build_reviewed.sh" "$here/diskcheck.sh" "$W/repo/ios/"; echo t > "$W/repo/ios/tracked.txt"
(cd "$W/repo" && git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -qm init)
cat > "$W/bin/swift" <<'X'
#!/bin/sh
case "$1" in build) [ -n "${FAIL_BUILD:-}" ] && exit 42; mkdir -p .build/release; echo bin > .build/release/parakeet-bench; echo built;;
test) [ -n "${FAIL_TEST:-}" ] && { echo "error: boom"; exit 42; }; echo "Executed 7 tests, with 0 failures";; esac
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
export PATH="$W/bin:$PATH" DISKCHECK_PATH="$W"
S="$W/repo/ios/bench/.build/release/BUILD_INFO.json"
FAILS=0
check() { if [ "$2" = "$3" ]; then echo "PASS $1"; else echo "FAIL $1: expected '$2', got '$3'"; FAILS=$((FAILS+1)); fi; }
stamp() { [ -f "$S" ] && echo yes || echo no; }
run() { sh "$W/repo/ios/build_reviewed.sh" > /dev/null 2>&1; echo $?; }
check "1 clean build: stamped" "0 yes" "$(run) $(stamp)"
check "2 build failure: exit 7, previous stamp removed" "7 no" "$(FAIL_BUILD=1 run) $(stamp)"
run > /dev/null
check "3 test failure: exit 8, no stamp" "8 no" "$(FAIL_TEST=1 run) $(stamp)"
run > /dev/null; echo x >> "$W/repo/ios/tracked.txt"
check "4 modified checkout: exit 2, no stamp" "2 no" "$(run) $(stamp)"
(cd "$W/repo" && git checkout -q -- ios/tracked.txt); run > /dev/null
check "5 low disk (32 GB < 30 + 3): exit 5, no stamp" "5 no" "$(FREE_KB=33554432 run) $(stamp)"
check "6 unreadable df: exit 5" "5" "$(FREE_KB=abc run)"
check "7 diskcheck: 40 GB >= 30 + 9" "0" "$(FREE_KB=41943040 sh "$W/repo/ios/diskcheck.sh" 9 > /dev/null 2>&1; echo $?)"
echo "failed checks: $FAILS"
exit $FAILS
