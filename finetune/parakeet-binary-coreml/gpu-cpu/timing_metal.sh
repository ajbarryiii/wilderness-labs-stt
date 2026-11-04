#!/bin/sh
# Metal microbenchmark timing run (run from gpu-cpu/ under run_guarded.sh, quiet host). Writes results/metal_*.json.
set -eu
BIN=/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/kea-b1/gpu-cpu/bin
if [ "${SKIP_BENCH:-0}" != 1 ]; then
  "$BIN/b1bench" peak metal/b1gemm.metal results/metal_peak.json
  "$BIN/b1bench" bench metal/b1gemm.metal results/metal_bench.json 7
fi
"$BIN/b1bench" encoder metal/b1gemm.metal results/metal_encoder.json results/metal_bench.json
