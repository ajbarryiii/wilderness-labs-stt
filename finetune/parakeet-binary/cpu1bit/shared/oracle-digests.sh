#!/usr/bin/env bash
# Oracle reference digests for every captured clip (cpu1bit exactness, plans/cpu-benchmark-mac.md).
set -euo pipefail
B=/mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit
bin=$B/nixos-target/release/oracle-digests
sha256sum "$bin"
for g in timed exactness; do
  $bin --export /mnt/hd/wilderness-labs-stt/parakeet-binary/runs/main-binB-m1-lr2e-4/export --format binary \
       --captures $B/captures/MB1/$g --out $B/oracle/MB1/$g --components 2,3 --threads 16
  $bin --export /mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/export --format ternary \
       --captures $B/captures/M1/$g --out $B/oracle/M1/$g --components 2,3 --threads 16
done
