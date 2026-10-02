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
| `macguard` | Wrapper for every Mac job (Python, the Mac's `/usr/bin/python3`, stdlib only): start conditions, flock lock held through cleanup, own session and process group, RSS cap, timeout, memory and swap aborts, fail-closed probes, TERM→KILL until the group is verifiably empty, logging. Exit: the job's status; 124 if the guard aborted it; 130 interrupted; 3 refused; 2 usage; 125 internal. |
| `artifacts.py` | Allowed artifact locations; every writer refuses paths off `/mnt/hd` (Linux), outside the artifacts directory (Mac) or inside the repository. |

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
# NeMo golden outputs of B0, M_P2 and surrogate seed 0 (one unit each), then the reference against them (gate 1)
./heavy ios-wp1-golden --mem-max 12G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/golden.py --model seed0  # or mp2, b0
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

Both machines write `<artifacts>/random/manifests/seed{0,1,2}.json`; `tests/compare_manifests.py`
compares them. Verified: seeds 0-2, all 989 tensors identical between Linux x86-64 (NumPy 2.4) and
macOS 27 arm64 (NumPy 2.2.6), recorded in `results/wp1_manifest_crosscheck.json`. Other platforms
or NumPy versions are not covered by that check (NumPy's Generator stream is only guaranteed per
version); re-run the comparison after changing either.

## WP2: clips, replay traces, C0 baseline, harness

| File | What |
| --- | --- |
| `clips.py` → `clips.json` | 82 clips from LibriSpeech dev-clean, read from `dev-clean.tar.gz` (MD5 `42e2234b…` = OpenSLR's published value; SHA-256 recorded). 16 natural utterances per 2/4/8/15 s bucket (distinct speakers first), 16 boundary crops (bucket edges `N_e`, `N_e + 1`, and `N = 160 M` with `M = 8k - 1, 8k, 8k + 1`), a silence and an impulse clip. Per-clip SHA-256 of the float32 PCM. The frame formulas are in the docstring. |
| `traces.py` → `traces.json` | B0 (`models.b0`, the FP32 reference) decodes every clip. Records the complete greedy TDT trace: frame, prediction input, token, duration, emitted, pred_updated, symbols_at_frame, forced_advance, advance. Also the transcript and WER. 114 KB. |
| `c0.py` → `c0.json` | C0 pin: `FluidInference/parakeet-tdt-0.6b-v2-coreml` at `ee09c56`. Only the files FluidAudio 0.7.8 loads for v2: `Preprocessor`/`Encoder`/`Decoder`/`JointDecision` `.mlmodelc`, `parakeet_vocab.json`, `config.json`. SHA-256 and I/O schema of every file. |
| `references.md` | Pinned permalinks and verbatim quotes behind DESIGN.md "Prior evidence", plus the C0 artifact table. |
| `bench/` | `ParakeetBench` Swift package (tools 6.0; macOS 15+ / iOS 26+), with `BenchCore` and the `parakeet-bench` CLI. C0 pipeline: Preprocessor → Encoder → per-step Decoder/JointDecision calls replicating FluidAudio 0.7.8's `TdtDecoderV3` (cited in `C0Pipeline.swift`). Also free and replay modes, the warm-up/timed protocol, `mach_absolute_time` stage times, `os_signpost` intervals, and `MLComputePlan` per-op dumps. |
| `c0report.py` | Scores `parakeet-bench` JSON lines against `clips.json` and the B0 traces, with the parent experiment's Whisper-normalized WER. |
| `g0probe.py` | G0 feasibility: parses C0's `Encoder.mlmodelc` (MIL text and blob file), cross-checks with coremltools, and fingerprints the tensors against B0. |
| `results/c0_free_natural.summary.json`, `results/g0_probe.summary.json` | WP2 results. They hold no weights or audio. |

Commands (`A` is the machine's artifact directory, `R` the repository root):

```sh
# NixOS
CUDA_VISIBLE_DEVICES= ./python ios/clips.py select              # writes ios/clips.json
CUDA_VISIBLE_DEVICES= ./python ios/clips.py materialize --out $A/clips
./heavy ios-wp2-traces --mem-max 10G --runtime 40min --wait -- env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/traces.py
CUDA_VISIBLE_DEVICES= ./python ios/c0.py pin                    # writes ios/c0.json
# Mac (repository finetune/parakeet-ternary/)
ios/macguard --rss-cap 1G --timeout 3600 -- sh -c 'curl -fsSL -o $A/data/dev-clean.tar.gz https://www.openslr.org/resources/12/dev-clean.tar.gz \
    && ios/pyenv/.venv/bin/python ios/c0.py download --out $A/c0'
ios/macguard --rss-cap 1G --timeout 600 -- ios/pyenv/.venv/bin/python ios/clips.py materialize --out $A/clips
(cd ios/bench && ../macguard --rss-cap 4G --timeout 900 -- swift build -c release)
B=ios/bench/.build/release/parakeet-bench
ios/macguard --rss-cap 4G --timeout 1800 -- $B run --mode free --models $A/c0 --clips ios/clips.json --pcm $A/clips \
    --kinds natural --warmups 3 --timed 1 --out $A/results/c0-free-natural.jsonl   # --mode replay --traces ios/traces.json
ios/macguard --rss-cap 4G --timeout 900 -- $B plan --models $A/c0 --out $A/results/computeplan-c0
ios/macguard --rss-cap 4G --timeout 900 -- ios/pyenv/.venv/bin/python ios/g0probe.py extract \
    --model $A/c0/Encoder.mlmodelc --out $A/results/g0probe.json
# NixOS, on copies of the Mac outputs
CUDA_VISIBLE_DEVICES= ./python ios/c0report.py c0-free-natural.jsonl --plan summary.cpuAndNeuralEngine.json \
    --out ios/results/c0_free_natural.summary.json
./heavy ios-wp2-g0 --mem-max 8G --runtime 10min --wait -- env CUDA_VISIBLE_DEVICES= \
    $R/finetune/parakeet-ternary/python $R/finetune/parakeet-ternary/ios/g0probe.py compare --probe g0probe.json
```

Results (2026-10-02; Mac = M1 Pro, macOS 27.0, Swift 6.3.1; Mac timings are informational because the Mac is shared):

- **Clips.** The PCM of all 82 clips is bit-identical on NixOS and the Mac. Both machines report `all_sha256_match`, and the Mac's own OpenSLR download has the same MD5.
- **B0 traces.** 3,025 steps and 2,624 tokens. The max-symbols rule fires once (`b04-N32001`, a crop that ends mid-word and loops). Replaying the trace on the same model reproduces every decision. Whisper-normalized WER on the 64 natural clips is **1.77%** (17 errors / 959 words).
- **C0 free decoding**, 64 natural clips, `cpuAndNeuralEngine` (preprocessor `cpuOnly`):
  - WER **1.67%** vs LibriSpeech (16 / 959) and **0.21%** vs B0's transcripts. 53 of 64 token sequences equal B0's.
  - Median total latency per bucket, 2/4/8/15 s: 62.1 / 67.6 / 80.1 / 99.4 ms.
  - Encoder 41.3-42.7 ms in every bucket (fixed 15 s window).
  - Preprocessor 13.8-16.5 ms.
  - Decoder 0.51-0.54 ms per call, joint 0.136 ms per call.
  - Physical calls: 1 preprocessor + 1 encoder per utterance. Decoder 1,951 calls and joint 2,147 calls, against B0's 1,955 prediction-net runs and 2,159 logical steps.
- **Load.** The first Encoder load took 29.4 s (device compilation); cached, it takes 109 ms. `phys_footprint` stays at 28-62 MB, because the model memory is not attributed to the process. The macguard group RSS peak was 0.5 GB.
- **Compute plan** (`cpuAndNeuralEngine`):
  - Encoder: 1,379 of 1,385 placed ops prefer the ANE (99.95% of estimated cost). The 6 CPU ops are 4 `cast`, 1 `expand_dims` and 1 `less`, i.e. input and length handling.
  - Decoder (24 ops) and JointDecision (21 ops) prefer the CPU entirely, as does the preprocessor.
  - The plan is not proof of placement (DESIGN.md gate 6).
- **Replay** works. On a 3-clip smoke test, C0's own argmax matched the trace on 98/99 token and 90/99 duration decisions of a natural 15 s clip.

C0 deviates from NeMo/B0 in three places. These are properties of the published pipeline, reproduced here, not harness bugs:
1. **One extra valid mel frame.** C0's preprocessor reports `mel_length = N // 160 + 1`, where NeMo has `N // 160`. With FluidAudio's `ceil(N / 1280)`, 5 of the 64 natural clips decode one more encoder frame than B0.
2. **Final token dropped at the end.** FluidAudio emits a token only if `t + duration` is still inside the utterance. This drops a final token whose duration reaches the end: 2 of 3 clips lose final punctuation this way, and the third is a decision difference.
3. **Different joint model.** JointDecision is FP16 with an in-model argmax and per-step encoder projection, so durations differ more often than tokens.

G0 feasibility: **practical.**
- `model.mil` holds 294 `constexpr_lut_to_dense` ops. Each has `indices` = packed 6-bit uint8 blob and `lut` = fp16[64], one LUT per tensor, plus 320 dense fp16 consts. That accounts for all 908 blobs in `weight.bin`.
- The blob format is a 64-byte header (count 908, version 2), then per blob 64 bytes of metadata (sentinel `0xDEADBEEF`, dtype, size, data offset) and 64-aligned data.
- An independent parser agrees exactly with coremltools 9.0's `_BlobStorageReader` on all 294 tensors and with `constexpr_lut_to_dense.decompress` on the 30 tensors checked in full.
- All 294 tensors map onto B0 weights. Median fingerprint relative error is 2.8% (max 7.8%), and RMS ratios are 0.9986-1.0005. These are 6-bit k-means of B0.
- Two catches for G0's graph:
  - the 24 depthwise convolutions have BatchNorm folded in (with separate fp16 bias consts);
  - `linear_pos` is not stored. Each layer has a folded position table `[1, 8, 128, 375]` for the 188-frame window. Shorter buckets need the middle `2T - 1` columns, a slice that is exact because the projection has no bias.
