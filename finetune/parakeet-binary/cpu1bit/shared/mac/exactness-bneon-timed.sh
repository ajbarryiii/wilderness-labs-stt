#!/bin/sh
# B-NEON real-capture exactness (timed clips), all four arms built in ws/H.
set -eu
D=${CPU1BIT_ROOT:-$HOME/wilderness-labs-stt-artifacts/cpu1bit}
R=$D/target/H/release
mkdir -p $D/results/exactness
run() { /usr/bin/perl $D/ws/H/shared/mac/withlock shared $D/lock/active.lock -- "$@"; }
run $R/replay-b-neon exactness --export $D/data/exports/MB1-derived --captures $D/data/captures/MB1/timed --reference $D/data/oracle/MB1/timed --threads 8,1 --components 2,3 --out $D/results/exactness/b-neon-R-MB1-timed.json
run $R/replay-b-neon exactness --export $D/data/exports/M1 --captures $D/data/captures/M1/timed --reference $D/data/oracle/M1/timed --threads 8 --components 2,3 --out $D/results/exactness/b-neon-R-M1-timed.json
run $R/replay-b-neon-1bit-tile exactness --export $D/data/exports/MB1 --captures $D/data/captures/MB1/timed --reference $D/data/oracle/MB1/timed --threads 8 --components 2,3 --out $D/results/exactness/b-neon-1bit-tile-MB1-timed.json
run $R/replay-b-neon-pre-expanded exactness --export $D/data/exports/MB1-derived --captures $D/data/captures/MB1/timed --reference $D/data/oracle/MB1/timed --threads 8 --components 2,3 --out $D/results/exactness/b-neon-pre-expanded-MB1-timed.json
