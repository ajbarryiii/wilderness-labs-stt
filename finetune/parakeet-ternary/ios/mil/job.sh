#!/bin/sh
# One WP3 Mac job, run INSIDE macguard (so all of this happens while the guard's lock is held):
#
#   ./macguard --rss-cap 6G --timeout 5400 -- sh mil/job.sh NEED_GB "SHELL LINE" 
#
# 1. Disk: refuses (exit 5) unless the data volume has at least FLOOR (30 GB, user-approved) + NEED_GB free,
#    NEED_GB being the job's expected transient size (build output, Core ML device specializations, restores).
# 2. Temporary storage is isolated and owned:
#    - TMPDIR = <artifacts>/tmp/job-<pid> (coremltools' temporary .mlpackage directories, compiles), created
#      here and removed at the end;
#    - Python runs as `wp3py`, a symlink to the uv environment's python, so Core ML writes its
#      device-specialized model copies to ~/Library/Caches/wp3py, a directory only these jobs use; the Swift
#      plan tool likewise uses ~/Library/Caches/computeplan. Both are emptied before and after the job (jobs
#      are serialized by macguard's lock; nothing else writes there). Shared directories are never purged.
#    - WP3_COREML_CACHE tells the Python side which directory it may empty between model loads.
# 3. The line runs under sh from ios/, with `pyenv/.venv/bin/python` replaced by `pyenv/.venv/bin/wp3py`; its
#    exit status is returned unchanged (5 = refused for disk space).
set -u
FLOOR=30
need=$1; shift
here=$(cd -- "$(dirname -- "$0")/.." && pwd)
art=/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios
free_gb=$(df -g "$art" | awk 'NR == 2 {print $4}')
if [ "$free_gb" -lt $((FLOOR + need)) ]; then
  echo "job.sh: refusing: ${free_gb} GB free < ${FLOOR} GB floor + ${need} GB expected for: $1" >&2
  exit 5
fi
ln -sf python "$here/pyenv/.venv/bin/wp3py"
caches="$HOME/Library/Caches/wp3py $HOME/Library/Caches/computeplan"
clean() { for c in $caches; do rm -rf "$c"; done; rm -rf "$TMPDIR"; }
TMPDIR="$art/tmp/job-$$"
export TMPDIR
export WP3_COREML_CACHE="$HOME/Library/Caches/wp3py"
for c in $caches; do rm -rf "$c"; done
rm -rf "$art"/tmp/job-*  # leftovers of guard-aborted jobs (only these jobs write there)
mkdir -p "$TMPDIR"
line=$(printf '%s' "$1" | sed 's#pyenv/\.venv/bin/python #pyenv/.venv/bin/wp3py #g')
cd "$here" && sh -c "$line"
status=$?
clean
exit $status
