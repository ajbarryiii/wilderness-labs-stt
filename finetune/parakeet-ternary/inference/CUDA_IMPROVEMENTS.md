# Hybrid CUDA/Triton ternary runtime on RTX 5090

On our RTX 5090 at its configured **400 W power cap** (no power-limit or clock changes), the opt-in hybrid CUDA/Triton runtime `ptx_row_best` beat our Expanded Triton ternary runtime. Median warm batch-one full-transcription latency was **3.43% lower** and GPU energy per transcription was **3.01% lower**. Encoder latency was 5.26% lower and encoder energy was 4.56% lower. Process VRAM was essentially unchanged. The default runtime is unchanged.

## Results

The figures are medians of per-utterance paired changes over 24 real utterances. Counts show how many utterances were lower with the hybrid. Data: [cuda-iteration-rtx5090.json](../results/cuda-iteration-rtx5090.json).

| Scope | Latency | GPU energy | Average power |
|---|---|---|---|
| Encoder | -5.26%; 24/24 | -4.56%; 24/24 | +0.82% |
| Full transcription | -3.43%; 24/24 | -3.01%; 21/24 | +0.54% |

By duration target, with eight utterances each:

| Target | Actual audio | Encoder latency / energy | Full latency / energy |
|---|---|---|---|
| 3 s | 3.00–5.01 s | -8.31% / -5.41% | -6.07% / -4.01% |
| 10 s | 9.99–10.01 s | -5.26% / -4.82% | -3.43% / -3.35% |
| 30 s | 15.00–30.26 s | -1.86% / -1.82% | -0.90% / -1.90% |

Every per-target change was lower for 8/8 utterances except 30 s full-transcription energy. That one was lower for only 5/8, so it is not a clear gain. GPU energy is power multiplied by time. At 3 s, the encoder drew 3.07% more power but finished 8.31% sooner, so it used 5.41% less energy.

## Method

- The set has 24 utterances, eight per 3, 10 and 30 s target. They come from AMI, Common Voice, Earnings22, GigaSpeech, LibriSpeech clean and other, SPGISpeech and VoxPopuli, for 318.95 s of audio in total.
- Each variant ran three balanced one-second windows per utterance. Each paired change uses each variant's median window.
- Timings are warm batch-one CUDA-graph replays. Compilation, graph capture and audio file I/O are excluded. This is not a cold-start or full-corpus benchmark.
- Encoder scope runs from GPU features to encoder output. Full scope runs from a CPU float32 waveform to CPU text, including transfers, frontend and decoder.
- Energy is NVML GPU board energy. It includes desktop GPU activity and excludes host energy. The 400 W cap was checked during every experiment.
- Both variants include the existing decoder token cache, unchanged. None of these gains come from it.
- The primary paired benchmark clears each model's bounded cache between utterances and retains one warmed waveform shape at a time. Both models stay resident, so this process is not used for deployment VRAM.

## Process VRAM

Each variant ran once in a separate fresh process on the same LibriSpeech clips of 3, 10 and 30.04 s, in that order. The graph cache holds at most four entries. Only three shapes ran, so all three graphs and the earlier pools accumulated. Each value therefore includes earlier shapes, model loading and first-use capture (NVML, sampled every 10 ms). GB and MB are decimal (10^9 and 10^6 bytes).

| Audio | Expanded Triton | Hybrid | Change |
|---|---|---|---|
| 3 s | 1.741 GB | 1.739 GB | -2.10 MB (-0.12%) |
| 10 s | 1.887 GB | 1.883 GB | -4.19 MB (-0.22%) |
| 30.04 s | 2.233 GB | 2.231 GB | -2.10 MB (-0.09%) |

These differences come from single fresh-process comparisons. They are too small to show a memory gain.

## What changed

The hybrid changes three kernel groups.

- **Short mix/gate projections** ([ptx_cuda.py](ptx_cuda.py), [native_groups.py](native_groups.py)) use CUDA inline PTX derived from Triton's Tensor Core program. Two 128-thread programs share a 256-thread CTA and combine partial sums with an exact INT32 reduction before FP32 conversion.
- **Short-sequence QKV projections** use three pipeline stages.
- **Fused rows** ([row_fusion.py](row_fusion.py)) combine the projection epilogue, residual, next normalization and three-component residual INT8 quantization in one kernel. They use eight warps up to 192 rows and four above that.

## Numerics and validation

Trained ternary weights, scales and the three residual INT8 components are unchanged. The FP32 reduction order changes, so encoder tensors differ slightly. The maximum absolute difference was 2.4139881134033203e-6 and the maximum relative L2 difference was 1.82e-6.

Validation used a **sampled subset** of 2,048 utterances (14,156 s total, 0.13–33.30 s each). Recognition text, token IDs and timestamps matched on all 2,048. The maximum decoder score difference was 0.00195. This is not full-corpus or field validation.

Compute Sanitizer 12.9.79 memcheck found 0 errors and 5 tests passed, with API-error reporting disabled. Synccheck found 0 errors and 1 test passed.

## Experiments that did not help

Synthetic microbenchmarks covered 953 configuration/shape evaluations: 916 ran and 37 were compile or resource rejects. These approaches gave no gain:

- register caps (strong caps regressed)
- compressed 2-bit or 4-bit weights (none beat INT8 storage)
- exact INT4 decomposition (4–7x slower)
- batched Triton reductions (lowered to SIMT and slower)
- four native programs per CTA (27 of 32 rejected, no wins)

## Run it

The runtime is opt-in through `--experimental-cuda`. It requires Torch 2.11, Triton 3.6, NVRTC 12.9 and an SM120 GPU. From the repository root, these commands transcribe a clip, rerun the encoder and full-transcription comparisons, and rerun the recognition check:

```bash
finetune/parakeet-ternary/python -m inference --experimental-cuda --export /mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/export clip.wav
finetune/parakeet-ternary/python -m inference.cuda_experiments encoder --candidates ptx_row_best --seconds 1 --repeats 3
finetune/parakeet-ternary/python -m inference.cuda_experiments pipeline --candidates ptx_row_best --seconds 1 --repeats 3
finetune/parakeet-ternary/python -m inference.quality4 --reference expanded --candidate ptx_row_best --batch-size 1 --utterances-per-set 256 --compile-workers 8
```
