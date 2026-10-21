#!/usr/bin/env bash
# Freeze the roster's code for timed stages (plans/cpu-benchmark-mac.md "Code", "Selection": the roster
# is frozen before screening). Stages EVERY member that has a Cargo.toml from the live tree, pushes it to
# the Mac's ws/frozen, builds every replay binary (`cargo build --release --locked --bins`, pinned
# toolchain, -C target-cpu=apple-m1, low priority under the shared lock), and records the staged tree
# hash and the SHA-256 of every target/frozen/release/replay-* binary in
#   Mac:   results/frozen/FROZEN-<tree>.json      NixOS: /mnt/hd/.../cpu1bit/frozen/FROZEN-<tree>.json
#   cpu1bit/shared/freeze-build.sh
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
py="$(cd -- "$here/../.." && pwd)/python"
D=$("$here/mac-root")
# The pinned Cargo.lock must already cover every workspace member (coordinator 2026-10-21): staging
# all members must not change it, so the frozen `--locked` build uses exactly the committed lock.
chk=$(mktemp -d /mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/freeze-check.XXXXXX)
"$py" "$here/stage.py" "$chk/tree" >/dev/null
if ! cmp -s "$here/../Cargo.lock" "$chk/tree/Cargo.lock"; then
  echo "Cargo.lock does not cover every workspace member (diff below); H must update it before freezing" >&2
  diff "$here/../Cargo.lock" "$chk/tree/Cargo.lock" >&2 || true
  rm -rf "$chk"; exit 1
fi
rm -rf "$chk"
push=$("$py" "$here/push.py" code --agent frozen)
tree=$(printf '%s' "$push" | "$py" -c 'import json,sys; print(json.load(sys.stdin)["tree_sha256"])')
members=$(printf '%s' "$push" | "$py" -c 'import json,sys; print(",".join(json.load(sys.stdin)["members"]))')
echo "staged tree $tree members $members"
name=cpu1bit-freeze-${tree:0:8}
job start "$name" --host mac --cwd "$D/ws/frozen" --stall 30m --timeout 3h -- sh -c "
  set -e
  $D/ws/frozen/shared/mac/cargo build --release --locked --workspace --bins
  mkdir -p $D/results/frozen
  cd $D/target/frozen/release
  /usr/bin/python3 -I -c '
import glob, hashlib, json, os, sys
bins = sorted(p for p in glob.glob(\"replay-*\") + [\"oracle-digests\"] if os.path.isfile(p) and os.access(p, os.X_OK))
doc = {\"tree_sha256\": sys.argv[1], \"members\": sys.argv[2].split(\",\"), \"target\": os.getcwd(),
       \"binaries\": {b: hashlib.sha256(open(b, \"rb\").read()).hexdigest() for b in bins}}
open(\"$D/results/frozen/FROZEN-\" + sys.argv[1][:16] + \".json\", \"w\").write(json.dumps(doc, indent=1))
print(json.dumps(doc))
' $tree $members
" >/dev/null
job wait "$name" --host mac --timeout 170m
mkdir -p /mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/frozen
job tail "$name" --host mac -n 1 > "/mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/frozen/FROZEN-${tree:0:16}.json"
cat "/mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/frozen/FROZEN-${tree:0:16}.json"
