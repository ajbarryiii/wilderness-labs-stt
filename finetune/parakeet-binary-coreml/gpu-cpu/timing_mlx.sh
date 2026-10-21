#!/bin/sh
# MLX encoder timing run (run from gpu-cpu/ under run_guarded.sh, quiet host). Writes results/mlx_*.json.
set -eu
PY=/Users/ajbarry/workspace/github.com/wilderness-labs-stt/finetune/parakeet-ternary/ios/pyenv/.venv/bin/python
EX=/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/kea-b1/gpu-cpu/export/export.safetensors
export PYTHONDONTWRITEBYTECODE=1
for enc in ${ENCODINGS:-q2g64 q2g128 b1 dense}; do
  "$PY" mlx_encoder.py time --export "$EX" --encoding "$enc" --out "results/mlx_time_$enc.json"
done
for enc in ${PROFILE_ENCODINGS:-q2g64 b1}; do
  "$PY" mlx_encoder.py profile --export "$EX" --encoding "$enc" --warmup 5 --runs 20 --out "results/mlx_profile_$enc.json"
done
