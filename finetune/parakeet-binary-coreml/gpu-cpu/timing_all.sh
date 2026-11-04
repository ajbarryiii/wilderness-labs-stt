#!/bin/sh
# Final timing sequence under one guarded quiet-host job: Metal encoder-sequence (bench already done unless
# SKIP_BENCH=0), then the MLX encoder timings and profiles. Run from gpu-cpu/.
set -eu
SKIP_BENCH=${SKIP_BENCH:-1} ./timing_metal.sh
./timing_mlx.sh
