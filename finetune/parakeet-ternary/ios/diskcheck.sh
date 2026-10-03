#!/bin/sh
# Disk preflight for every Mac job that writes (WP7 review r2 finding 6):
#
#   sh ios/diskcheck.sh NEED_GB [PATH]
#
# Exits 0 only if `df -k` on PATH (default: the Mac artifact area) parses and shows at least the user-approved
# 30 GB floor + NEED_GB free; an unreadable or non-numeric reading refuses (exit 5, fail closed).
FLOOR=30
need=${1:?NEED_GB}
path=${2:-${DISKCHECK_PATH:-/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios}}
case "$need" in ''|*[!0-9]*) echo "diskcheck.sh: NEED_GB must be a whole number" >&2; exit 5;; esac
free_kb=$(df -k "$path" 2>/dev/null | awk 'NR == 2 {print $4}')
case "$free_kb" in ''|*[!0-9]*) echo "diskcheck.sh: refusing: cannot read free disk space of $path" >&2; exit 5;; esac
if [ "$free_kb" -lt $(( (FLOOR + need) * 1024 * 1024 )) ]; then
  echo "diskcheck.sh: refusing: $((free_kb / 1048576)) GB free < ${FLOOR} GB floor + ${need} GB" >&2
  exit 5
fi
echo "diskcheck.sh: $((free_kb / 1048576)) GB free >= ${FLOOR} + ${need} GB"
