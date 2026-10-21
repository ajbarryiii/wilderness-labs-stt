#!/bin/sh
# B-NEON real-capture exactness, timed clips, remaining arms (R-MB1 C=2,3 at 8 threads already exact:
# results/exactness/cpu1bit-H-exact-bneon-partial.log on NixOS). Binaries: target/H-smoke (current code).
set -eu
D=${CPU1BIT_ROOT:-$HOME/wilderness-labs-stt-artifacts/cpu1bit}
R=$D/target/H-smoke/release
mkdir -p $D/results/exactness
run() { /usr/bin/perl $D/ws/H-smoke/shared/mac/withlock shared $D/lock/active.lock -- "$@"; }
shasum -a 256 $R/replay-b-neon $R/replay-b-neon-1bit-tile $R/replay-b-neon-pre-expanded
run $R/replay-b-neon exactness --export $D/data/exports/M1 --captures $D/data/captures/M1/timed --reference $D/data/oracle/M1/timed --threads 8 --components 2,3 --out $D/results/exactness/b-neon-R-M1-timed.json
run $R/replay-b-neon-1bit-tile exactness --export $D/data/exports/MB1 --captures $D/data/captures/MB1/timed --reference $D/data/oracle/MB1/timed --threads 8 --components 2,3 --out $D/results/exactness/b-neon-1bit-tile-MB1-timed.json
run $R/replay-b-neon-pre-expanded exactness --export $D/data/exports/MB1-derived --captures $D/data/captures/MB1/timed --reference $D/data/oracle/MB1/timed --threads 8 --components 2,3 --out $D/results/exactness/b-neon-pre-expanded-MB1-timed.json
