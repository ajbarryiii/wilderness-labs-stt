# Second optimization pass on RTX 5090

This ledger records the second pass. The current optional optimized runtime
also includes the later work in [EXPERIMENTS3.md](EXPERIMENTS3.md).

The control is the previously delivered 2-bit packed BF16x3 encoder, with
strict FP32 NeMo operations and an optional encoder graph cache. The export is
unchanged (`a287e97719c451b785be2cd01ecc861fcaa010ebaa4ff2841783ec78fcd61503`).
No training, weight requantization, model download, or system power/clock change
was performed. All raw artifacts are on `/mnt/hd/wilderness-labs-stt/`.

## Experiment ledger

These are actual implemented or measured experiments, including rejected paths.
Screening numbers below are encoder-only, batch one, 10-second audio. They are
not the final paired end-to-end measurements. A screen used three CUDA-event
timing blocks and a short NVML energy window, so the later alternating windows
are the stronger evidence for energy claims.

| Experiment | Outcome |
| --- | --- |
| Profile previous encoder | Packed matmul 59.2% of GPU time, split reduction 8.0%, layer norms 6.0%; attention also material. |
| BF16x2 activation decomposition | About 5.42 ms vs 6.34 ms; relative encoder error about 1.7e-6. |
| Scaled residual FP16x2 | About 5.38 ms; error about 2.2e-7. Requires bounded activation range, so not the selected general path. |
| Single BF16 | About 4.91 ms; error rises to about 1.2e-3. Kept as an explicit quality experiment. |
| Single FP16 | About 4.91 ms; error about 1.4e-4. Kept as an explicit quality experiment. |
| Single TF32 | About 5.32 ms; error about 1.1e-3. Inferior speed/accuracy tradeoff. |
| TF32x3 | About 9.08 ms; accurate but slower. |
| Concatenate packed Q/K/V projections | About 6.18 ms; saves launches and shares input work. Retained. |
| Fuse SiLU into projection reduction | About 6.22 ms; avoids intermediate activation work. Retained. |
| Hoist fixed positional projections out of graph replay | About 6.01 ms; identical encoder output in isolated test. Retained for offline graphs. |
| PyTorch SDPA | About 6.51 ms, slower than the control on this workload. Not selected. |
| Fold evaluation BatchNorm into depthwise convolution | About 6.28 ms; small gain. Initial biasless-convolution limitation was fixed. Retained. |
| Triton FP32 layer normalization | About 6.17 ms; retained. |
| Fused relative attention with TF32x3 | About 5.97 ms; avoids separate score, shift, mask, softmax, value-product work. Initial tile exceeded SM120 shared memory; smaller tile and one pipeline stage passed. Retained. |
| Wider tile/warp/stage/split sweep | 65 candidates for each of four projection shapes and two arithmetic modes at M=126 (520 candidates). Mostly confirmed prior settings; small gains on some FP16x2 shapes. No broad default retuning selected. |
| Atomic split-K accumulation | About 6.21 ms; small gain with nondeterministic summation. Deterministic reduction retained. |
| Dynamic per-token INT8 | About 4.40 ms; relative encoder error about 3.8e-2. Too much numerical drift for the selected path. |
| Two residual INT8 passes | About 4.73 ms; relative error falls to about 6.9e-5. Faster experimental option. |
| Three residual INT8 passes | About 5.03 ms; relative error about 3.4e-7. Selected arithmetic. |
| Three-pass INT8 plus all fusions | About 4.00 ms; about 1.54 J vs 2.43 J in its screen. |
| Fuse layer normalization into residual INT8 preparation | About 3.92 ms; saves another 96 normalization launches per encoder. Selected. |
| Dedicated residual-INT8 tile sweep | 252 additional candidates across four projection shapes and three row counts. Selected larger K tiles and fewer split reductions where beneficial. Full encoder screen improved another 4.1%, 2.6%, and 9.2% at 3/10/30 seconds. |
| Capture frontend and encoder together | Capture passed with changed audio and lengths; removes frontend launch overhead. Selected. |
| Diagnose decoder's partial-graph fallback | Forced full graph exposed missing NVRTC library discovery; original decoder performed about 120 host-controlled loop synchronizations per clip. |
| Expose existing NVRTC wheel through library path | Full conditional decoder graphs now work; no decoder algorithm change or dependency download. Selected. |
| Lower board power limit / clock controls | Device reports 400 W minimum and was already at 400 W. Passwordless privileged controls unavailable. No hardware settings changed; no clock-sweep benefit claimed. |

Earlier FP32 lookup-table and direct SIMT encoder experiments remain documented
in [MEASUREMENTS.md](MEASUREMENTS.md); they were slower and consumed more energy
than the packed Tensor Core path.

## Selected design

Weights remain two-bit packed. A token's floating-point input is approximated
as the sum of three separately scaled INT8 residual vectors. Each integer
Tensor Core product accumulates into INT32; conversion, row scales, bias and
outputs use FP32. This is a high-accuracy approximation of floating activations,
not ordinary single-pass INT8 activation quantization. It changes no trained
weights and introduces no calibration dataset.

Normalization can run in the same kernel that produces the three residual
vectors. The Q/K/V projections share one prepared input and packed projection;
SiLU runs in the projection's final reduction. Relative attention keeps FP32
inputs, softmax and outputs with TF32x3 products. Ordinary NeMo modules remain
responsible for preprocessing, depthwise convolution, decoding and tokenization.

Graph caches use exact shapes and copy current audio and lengths every call.
Fixed positional projections are computed during warmup and kept alive for each
captured shape. The decoder uses NeMo's existing conditional CUDA graph path.
The public optimization switch is reversible and inference-only; disable it
before changing weights/device, serializing state, or training.

The CUDA configuration was checked against NVIDIA's
[Blackwell tuning guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/).
SM120's per-block shared-memory limit matters: SM100 examples cannot simply be
copied onto this consumer GPU. Triton's
[matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
provides background on tile selection and memory reuse; the kernels here are
specific to the existing Parakeet export.

## Paired end-to-end measurements

The table below measures the final runtime, including integer tile tuning.

Three alternating 10-second windows per variant and workload, with two seconds
of warmup before each window, measured the following medians. Batch size is one.
The prior packed control explicitly uses its earlier partial-graph decoder;
the optimized variant uses the same trained weights and full conditional
decoder graphs. Input audio is already on GPU, and returned text is included.

| Audio | Prior packed latency | Optimized latency | Speedup | Prior GPU energy | Optimized GPU energy | Energy reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3.00 s | 6.504 ms | 4.410 ms | 1.47× | 1.716 J | 1.142 J | 33.4% |
| 10.00 s | 12.378 ms | 8.934 ms | 1.39× | 3.540 J | 2.545 J | 28.1% |
| 30.04 s | 29.944 ms | 21.483 ms | 1.39× | 9.419 J | 6.563 J | 30.3% |

Decoder-library discovery alone, with the previous encoder, measured 6.281,
11.848 and 27.045 ms respectively. Thus the measured gain is not solely a
decoder configuration fix. Before final tile tuning, a separate paired run of
the 10-second encoder alone measured 3.947 ms versus 6.351 ms (about 38% less
time and GPU energy). Final integer tile settings improved the encoder further
in the subsequent screen described above.

Median window p95 latencies for optimized transcription were 4.429, 9.025 and 21.730 ms.
Encoder outputs passed numerical checks, and all three variants returned the
same transcripts for these clips. Average watts can increase even while joules
per transcription decrease; energy per completed utterance is the objective.

Raw alternating windows, checks, source hashes, and summary:
[`20261005T120235-benchmark2`](/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T120235-benchmark2/summary.json).
The final runtime files and interpreter wrapper match this run's recorded
source hashes. The optimized CLI matched its control transcript and shut down
cleanly. The earlier, pre-tuning paired run is preserved at
[`20261005T113817-benchmark2`](/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T113817-benchmark2/summary.json);
its post-measurement cleanup issue was fixed and tested before the final run.
Cold compilation/loading, host
energy and audio disk I/O are excluded. The desktop remained active and all
three models were resident, so this run is not a standalone memory measurement.

## Raw screening artifacts

The 512-utterance recognition screen covered 3,400 seconds of audio and 8,782
reference words across eight test sets. The selected normalization+INT8x3
combination, both two-component floating paths, single FP16, and both tested
INT8x2 paths matched all 512 control transcripts. Single BF16 matched 510;
single INT8 matched 478. Their aggregate reference edit counts were 525 and
522 versus the control's 526: changed transcripts did not imply worse aggregate
WER in this subset, but neither establishes quality equivalence. INT8x3 was
selected for its much smaller numerical drift. The new full-graph decoder's
control transcripts also matched all 512 prior dense FP32 reference outputs.

These are subset checks, not a full-corpus WER guarantee.

The final public `enable_optimizations` API, including tuned integer tiles and
the bounded pipeline graph cache, then matched **2,048/2,048** control transcripts
exactly. This expanded run covered **14,156.25 seconds (3.93 hours)** of audio
and **36,699 reference words**. Every test set had identical error counts;
both variants totaled 2,447 word edits. This also exercised variable-length
batches beyond the four-entry graph cache, where execution falls back to eager.
The six runtime source hashes match the tested final code.

Expanded quality evidence:
[`20261005T115402-quality2`](/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/20261005T115402-quality2/quality.json).
All 16 GPU/unit/integration tests passed; the six relevant tests were rerun and
passed after the final tile and cleanup changes. They cover three-component
arithmetic, fused normalization and SiLU, all-masked attention rows, changing
graph inputs/lengths, revisiting cached shapes, output ownership, and restoring
the original model outside `inference_mode`.

All paths below are under
`/mnt/hd/wilderness-labs-stt/parakeet-ternary/kernel-runs/`:

- `20261005T110706-profile2`: encoder profile.
- `20261005T111000-experiments`: arithmetic, QKV, SiLU, positional, SDPA screens.
- `20261005T111125-experiments`: BatchNorm and initial combinations at 3/10/30 seconds.
- `20261005T111316-experiments`: first normalization/attention/INT8 attempts, including compile failures.
- `20261005T111414-tune2`: 520 tile candidates and finalist rechecks.
- `20261005T111750-experiments`: corrected fused attention and integer path.
- `20261005T112023-experiments`: atomics and residual integer arithmetic.
- `20261005T112211-experiments`: combined integer and floating-point candidates.
- `20261005T112417-pipeline-probe`: successful frontend graph, original partial decoder mode.
- `20261005T112525-experiments`: normalization plus residual preparation fusion.
- `20261005T112651-pipeline-probe`: full decoder graph after NVRTC discovery fix.
- `20261005T112823-quality2`: deterministic multi-domain recognition comparison.
- `20261005T114833-integer-tiles`: 252 residual-integer tile candidates.
- `20261005T115224-experiments`: full-encoder confirmation of integer tile changes.

An earlier `20261005T111841-pipeline-probe` was rejected by the foreign-process
guard while the preceding process was exiting. It contains no valid timing.
Screens were sequential GPU jobs. Final performance claims require the paired
benchmark; cold compilation/loading, host energy and audio disk I/O are excluded.
