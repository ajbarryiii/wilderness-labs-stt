# Parakeet-TDT 0.6B v2 on iPhone: inference benchmark

Design: [DESIGN.md](DESIGN.md). Nothing here is a model or audio; generated weights, golden
outputs and logs live outside Git (NixOS `/mnt/hd/wilderness-labs-stt/parakeet-ios/`, Mac
`/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/`).

## S0 tooling: reference, surrogates, golden outputs, Mac guard

| File | What |
| --- | --- |
| `reference.py` | Pure-PyTorch FP32 reference of the v2 architecture (front end, FastConformer encoder, TDT prediction and joint networks, NeMo-equivalent greedy TDT with a complete step trace, forced replay returning logits and LSTM states). NeMo state_dict names; ternary modules load from int8 codes + FP32 row scales. |
| `weight_stats.py` → `weight_stats.json` | Statistics of the selected pilot export (P2, lr 5e-4): per-module code histograms, 257 row-scale quantiles, 129 value quantiles per floating tensor, the pinned model config, provenance. No weights. |
| `randomweights.py` | Seeded surrogate models from `weight_stats.json` (LayerNorm biases 0), bit-identical on Linux and macOS (per-tensor SHA-256 manifest), streamed to safetensors. |
| `models.py` | The benchmark models by name as reference models: `b0` (NVIDIA's .nemo), `mp2` (M_P2, the pilot P2 export, the primary model), `seed0`-`seed2` (surrogates). |
| `golden.py` | NeMo FP32 golden outputs of a full-depth benchmark model (`--model seed0` or `mp2`, NixOS) and the gate-1 comparison `compare()`. |
| `macguard` | Wrapper for every Mac job: start conditions, process-group RSS cap, timeout, memory and swap aborts, logging. |

Commands (NixOS from `finetune/parakeet-ternary/`, one memory-capped unit at a time; `R` is the
repository root; every unit runs CPU only):

```sh
# weight statistics (no NeMo, < 1 GB, may run directly)
CUDA_VISIBLE_DEVICES= ./python ios/weight_stats.py
# a surrogate (no NeMo, streams one tensor at a time, < 0.5 GB)
CUDA_VISIBLE_DEVICES= ./python ios/randomweights.py --seed 0 --out /mnt/hd/wilderness-labs-stt/parakeet-ios/random/seed0
# reference vs NeMo: real B0 weights at full depth, and a 2-layer surrogate (gate 1)
./heavy ios-wp1-nemo --mem-max 12G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/tests/test_reference_nemo.py -v
# NeMo golden outputs of surrogate seed 0 and of M_P2 (one unit each), then the reference against them (gate 1)
./heavy ios-wp1-golden --mem-max 12G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/golden.py --model seed0  # or mp2
./heavy ios-wp1-golden-check --mem-max 10G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/tests/test_reference_golden.py -v
# surrogate determinism, fidelity and a full-depth forward pass
./heavy ios-wp1-random --mem-max 8G --runtime 30min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/tests/test_randomweights.py -v
```

On the Mac (repository `finetune/parakeet-ternary/`), every job goes through `macguard`, which
logs to `<artifacts>/logs/macguard.log`:

```sh
ios/macguard --rss-cap 6G --timeout 1800 -- ios/pyenv/.venv/bin/python ios/tests/test_randomweights.py -v
```

Both machines write `<artifacts>/random/manifests/seed{0,1,2}.json`; equal per-tensor SHA-256s
(and so equal `digest`) mean the surrogates are bit-identical across machines.
