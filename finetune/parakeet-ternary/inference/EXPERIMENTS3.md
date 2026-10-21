# Third optimization pass: encoder and decoder measured separately

This pass keeps the trained ternary checkpoint unchanged and preserves NeMo's
greedy TDT decoding algorithm. The encoder remains two-bit packed; prediction
LSTM and joint weights remain FP32. The optional `--optimized` runtime now adds
FP32 decoder kernels and short-sequence positional-score fusion. The previous
arithmetic is available through
`enable_optimizations(model, decoder=False, position_dot=False)`.

See [COMPONENTS.md](COMPONENTS.md) for isolated frontend, encoder and decoder
comparisons, including dense reconstruction of the *same ternary weights* and
public NVIDIA / ONNX Parakeet v2 implementations. That control distinguishes
packed encoder execution from changes in checkpoint quality or decoding work.

## Implemented experiments

All rows below were actually implemented and exercised on this RTX 5090.
Screening timings are short, single-model measurements, not final deployment
latency or a universal speedup claim.

| Experiment | Outcome |
|---|---|
| Profile the second-pass runtime | At 10 seconds, frontend+encoder about 3.92 ms and decoder 4.78 ms. Decoder is material despite holding only 1.45% of model parameters including joint. |
| FP32 joint GEMV, fused add/ReLU/final projection | Reduced the 10-second full-path screen from 8.95 to 8.49 ms. No decoder quantization. |
| Custom two-layer FP32 LSTM step | About 8.18 ms full path; avoids generic recurrent-library work for one-token prediction. |
| Precompute embedding × first-layer input weights | About 8.06 ms with stock joint; table uses about 10.5 MB. Weights are unchanged, and matrix multiplication uses strict FP32. |
| Combined predictor and joint kernels | About 7.66–7.85 ms. Tested 1, 2, 4 and 8 hidden units per CTA; selected 1 on this workload. |
| Fuse token/duration argmax and time/mask bookkeeping | Combined with predictor/joint, about 6.26 ms at 10 seconds. Existing conditional graph loop and greedy policy retained. |
| Fuse token storage and recurrent-state selection | About 5.29 ms at 10 seconds and 12.05 ms at 30 seconds in the initial screen. Final results below include graph-allocation protection. |
| Fuse positional bias addition and relative-position product | Encoder screen 2.576 → 2.447 ms at 3 seconds and 3.801 → 3.725 ms at 10 seconds. Long-sequence result was tied/slightly worse (9.090 → 9.126 ms), so selected only for at most 192 encoder frames. TF32x3 products, FP32 inputs/outputs. |
| Alternating model instances / changing audio shapes | Exposed an illegal-address failure that single-model screens missed. It also reproduced in the unchanged NeMo decoder, so initial timing results were provisional. |
| Retain raw Triton arguments | Did not fix the failure by itself; it cannot retain temporaries internal to library operations. |
| Own conditional-body allocations with a private memory pool | Fixed the alternating-model failure. A second failure during graph replacement required deferring destruction of the previous pool until capture ended. Pool ownership remains bounded. |

The graph fix is in [graph_memory.py](graph_memory.py). PyTorch 2.11 associates
its graph allocation pool with the root capture ID; NeMo manually captures
conditional bodies on streams with other capture IDs. An enclosing thread-local
`torch.cuda.MemPool` retains those allocations until the decoder graph is
discarded. The root graph still owns its normal pool. This is local runtime
integration, not a newly trained model or a stock NeMo configuration switch.
Relevant upstream implementations:
[PyTorch 2.11 CUDAGraph](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/cuda/CUDAGraph.cpp)
and [allocator](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDACachingAllocator.cpp).

## Fresh-process end-to-end measurements

Batch one, CPU FP32 waveform through CPU text, including preprocessing and
transfers. Each implementation runs alone in a fresh process. Each clip has
three five-second measurement windows, with two seconds of warmup per window.
Numbers are medians across windows; process VRAM is the maximum sampled during
those warm windows. Shapes accumulate in order (3, 10, 30.04 seconds) in the
four-entry pipeline graph cache. Load, first capture, compilation and file I/O
are outside hot timings; GPU board energy excludes CPU energy.

| Audio | Second-pass ms | New ms | Second-pass J | New J | Second-pass process GB | New process GB |
|---|---:|---:|---:|---:|---:|---:|
| 3 s | 4.454 | 3.162 | 1.158 | 0.933 | 1.296 | 1.122 |
| 10 s | 9.029 | 5.274 | 2.552 | 1.892 | 1.361 | 1.271 |
| 30.04 s | 21.644 | 12.245 | 6.546 | 4.751 | 1.885 | 1.617 |

This is **29.0–43.4% lower latency** and **19.4–27.4% less GPU energy** for
these clips. The active board power can rise while total joules fall because
execution finishes sooner. Do not infer a reduction in peak watts.

Both rows include the same corrected conditional-graph allocation policy.
That correction changes memory ownership, so the second-pass VRAM here differs
from the historical pre-fix table in COMPARISON.md. New process memory is lower
despite the token-projection table: removing generic recurrent/joint work also
changes captured temporary and library workspace requirements.

Immediately after load/enable, live Torch allocations are 271.6 MB for the
second pass and 291.6 MB for the new path; process memory is 960.5 and 968.9 MB.
The new lookup table alone is about 10.5 MB; allocator/library allocations mean
that table size is not the full change in live or process memory. At 10 seconds,
the new runtime's live peak is 347.9 MB and reservation is 522.2 MB, while the
process occupies 1.271 GB. These are overlapping accounting views, not additive.

Runs are sequential rather than alternating paired windows. There is one real
LibriSpeech-clean clip per duration, so results depend on sequence length and
spoken token count. Small differences between implementations should not be
treated as a universal lead. All GB/MB here are decimal.

Raw worker JSON and telemetry:
`/mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/run3/`.

### Equally optimized original NVIDIA model

The decoder fusions also work on the original NVIDIA model. For fairness, these
fresh-process numbers use its BF16 encoder, the same pipeline graph wrapper,
the same graph-memory fix, and the same FP32 decoder fusions:

| Audio | Ternary ms | Dense BF16 ms | Ternary J | Dense BF16 J | Ternary process GB | Dense BF16 process GB |
|---|---:|---:|---:|---:|---:|---:|
| 3 s | 3.162 | 4.259 | 0.933 | 1.177 | 1.122 | 2.110 |
| 10 s | 5.274 | 5.806 | 1.892 | 1.921 | 1.271 | 2.183 |
| 30.04 s | 12.245 | 11.614 | 4.751 | 4.496 | 1.617 | 2.349 |

Ternary uses **41.8% less process VRAM** at 10 seconds, but the 1.5% energy
difference is too small to claim a robust energy lead. Dense BF16 wins latency
and energy on this 30-second clip. Short-clip results favor ternary more clearly.
Thus the new decoder speedup must not be attributed to ternarization, and the
ternary runtime does not beat equally optimized dense v2 at every length.

For comparison, NVIDIA BF16 with pipeline graphs but the original decoder
measured 9.471 ms / 2.636 J / 2.271 GB at 10 seconds in this same run. The new
decoder reduces that to 5.806 ms / 1.921 J / 2.183 GB. Both NVIDIA variants
match **256/256** transcripts and score 240 word edits / 4,721 reference words.
Both ternary variants match each other and score 263 / 4,721. Those checkpoints
have different weights and observed recognition quality; runtime preservation
is not proof that ternary equals the original model's accuracy.

## Recognition and numerical validation

The 2,048-utterance test covers 256 records from each existing test manifest:
LibriSpeech clean/other, AMI, earnings22, Common Voice, VoxPopuli, GigaSpeech,
and SPGISpeech. It contains 14,156.25 seconds of audio. Encoders use batch eight;
decoders are compared at batch one, alternating the old and new model instances.

- Text, token IDs and token timestamps match on **2,048/2,048** utterances.
- Word edits are identical within every test set.
  Both paths have 2,447 edits over 36,699 scored words (6.668% pooled WER).
- Maximum encoder absolute difference is `2.049e-6`; maximum relative L2
  difference is `6.423e-7`.
- Maximum accumulated decoder score difference is `0.001709`. Floating-point
  reduction order changes, so scores and activations are not bit-identical.

This establishes observed preservation of the ternary checkpoint on this
subset. It does not establish equivalence between that checkpoint and the
original NVIDIA model, nor does it guarantee every possible utterance agrees.
The existing original-v2 quality comparison remains in
[COMPARISON.md](COMPARISON.md).

Raw validation:
`/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T131508-quality3/quality.json`.
The later pool-destruction correction changes allocation lifetime only.

The integrated runtime also passed a 512-utterance batch-eight comparison
against the original packed BF16x3 path: **512/512 exact transcripts**, with
526 word edits over 8,782 scored words in both paths. The eight decoder/encoder
regression tests passed, including two alternating models, graph growth from
3-second to 30-second audio, repeated shape changes, allocation churn and
`empty_cache` between replays. Predictor checks cover batch sizes 1, 2, 8 and
the batch-nine fallback; other tests cover silence, noise, blank/zero-duration
transitions, argmax ties, state/timestamp preservation and restoration on disable.
Logs are under `/mnt/hd/wilderness-labs-stt/parakeet-ternary/final3/`.

A separate fresh-process batch-one, CPU-waveform-to-text comparison also matched
**256/256** transcripts between second and third passes (32 records per test
set). This includes the public optimization switch and pipeline graph cache.

The public `--optimized` CLI smoke test passed. Compute Sanitizer 12.9.79
memcheck also completed the model integration and positional-attention tests,
filtered to the eight new decoder/position kernels, with **zero memory errors**.
The initial tool run reported 20 `cuGetProcAddress_v2` invalid-value API lookup
diagnostics during CUDA-binding initialization, and no memory-access errors.
The memory-only rerun disabled API-error reporting; it is not a claim that the
entire CUDA API/library stack passed an unfiltered sanitizer audit. Both logs
are retained as `final3/memcheck.log` and `final3/memcheck-memory-only.log`.

## Reproduction

```sh
P=finetune/parakeet-ternary
export PARAKEET_KERNEL_EXPORT=/mnt/hd/wilderness-labs-stt/parakeet-ternary/runs/main-M1-P2-lr5e-4/export
$P/python -m unittest inference.test_decoder inference.test_experiments -v
$P/python -m inference.quality3 --position-dot --utterances-per-set 256
$P/python -m inference.sprint3 --variants baseline all_storage --durations 3 10 30
$P/python -m inference.compare_v2 --variant packed_second --out /mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/run3
$P/python -m inference.compare_v2 --variant packed_optimized --out /mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/run3
$P/python -m inference.compare_v2 --variant nemo_bf16_graphs_fused_decoder --out /mnt/hd/wilderness-labs-stt/parakeet-ternary/comparison/run3
```

Run GPU workers serially. Parallel development does not mean overlapping GPU
measurements. All model artifacts, raw results and compiler caches remain on
the mounted data disk. No power limit, clock, training weight or tokenizer was
changed. GPU joules exclude host-system energy.
