# Kea-B runtime results (native 1-bit research runtime on GitHub)

Detail for the [Kea-B model card](README.md).

Our native 1-bit
[research runtime](https://github.com/ajbarryiii/wilderness-labs-stt/blob/main/finetune/parakeet-binary/gpu1bit/README.md)
runs `export.safetensors` from its packed bits, without rebuilding dense floating-point encoder
weights. Speed measured with our research runtime on an RTX 5090; the runtime code is published
for reference and is not packaged for general use. It is MIT-licensed code in the GitHub
repository and is not part of this download. `load_binary.py` here still rebuilds a dense FP32
NeMo model (see [Use](README.md#use)).

The runtime multiplies the 1-bit encoder matrices on INT8 Tensor Cores, with int32 accumulation
and FP32 outputs. Each activation enters those multiplies as two residual INT8 parts ("x2"); the
earlier version of the runtime used three ("x3"). The weights are the same in both. The
prediction and joint networks run in FP32 through a fused decoder, and NeMo's greedy TDT loop
runs as CUDA graphs.

## Results

Warm batch-one transcription of one LibriSpeech test-clean clip per length on one RTX 5090 at a
400 W power limit. The Kea-B rows were measured side by side in one session; the original's rows
come from an earlier session (see [Comparing across sessions](#comparing-across-sessions)).

| Clip | Configuration | Latency (ms) | Process VRAM (GB) | GPU energy (J) | Avg GPU power (W) |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 s | Original FP32 — NeMo | 10.37 | 3.41 | 2.98 | 283 |
| 3 s | Original BF16 — optimized by Wilderness Labs | 4.25 | 2.11 | 1.15 | 271 |
| 3 s | Kea-B, x3 activations | 2.46 | 1.01 | 0.67 | 273 |
| 3 s | **Kea-B, x2 activations (current)** | 2.29 | 1.01 | 0.58 | 252 |
| 10 s | Original FP32 — NeMo | 18.74 | 3.49 | 5.75 | 304 |
| 10 s | Original BF16 — optimized by Wilderness Labs | 5.81 | 2.18 | 1.90 | 326 |
| 10 s | Kea-B, x3 activations | 4.13 | 1.16 | 1.37 | 330 |
| 10 s | **Kea-B, x2 activations (current)** | 3.88 | 1.16 | 1.20 | 310 |
| 30.04 s | Original FP32 — NeMo | 36.30 | 3.67 | 13.17 | 362 |
| 30.04 s | Original BF16 — optimized by Wilderness Labs | 11.56 | 2.35 | 4.48 | 386 |
| 30.04 s | Kea-B, x3 activations | 10.05 | 1.51 | 3.58 | 356 |
| 30.04 s | **Kea-B, x2 activations (current)** | 9.29 | 1.51 | 3.24 | 348 |

At 10 s, against our BF16 build, Kea-B with x2 has 33.3% lower latency, 47.0% lower process VRAM
and uses about 37% less GPU energy. Its latency saving is roughly constant per transcription
(2.0, 1.9 and 2.3 ms at 3, 10 and 30.04 s), while the energy saving grows with clip length (about
0.58, 0.69 and 1.24 J).

- **Original FP32 — NeMo** is NVIDIA's checkpoint run by NeMo in strict FP32, with NeMo's full
  CUDA-graph decoder.
- **Original BF16 — optimized by Wilderness Labs** is NVIDIA's checkpoint with a BF16 encoder,
  plus our own CUDA graphs, fused FP32 decoder and graph allocation fix. It is not stock NeMo.
  The two checkpoints produce different transcripts and therefore different decoder work, so
  these rows compare complete runtimes.

## x2 against x3, side by side

The two Kea-B runtimes differ only in the number of activation parts. They were measured in the
same session on 24 development utterances (8 each with target lengths of 3, 10 and 30 s). Each
value is the median of the per-utterance changes; in parentheses, the utterances where x2 was
lower:

| Scope | GPU energy | Latency |
| --- | ---: | ---: |
| Full transcription | 11.4% less (24 of 24) | 6.8% lower (24 of 24) |
| Encoder only | 11.9% less (24 of 24) | 11.0% lower (24 of 24) |

x2 makes two INT8 passes per matrix instead of three, and the average power during full
transcription fell from 323 W to 304 W. The same comparisons on 24 test utterances agree within
0.6 points. Process VRAM is the same for x2 and x3.

## Recognition with x2

On the full development sets (22,418 utterances, batch one), the runtime's mean WER is 6.3122%
with x2 and 6.3120% with x3; FP32 decoding of this export gives 6.3120%. The test sets were not
re-scored with the runtime, so the card's test WERs are from FP32 decoding.

## Method

- **Timed path:** a CPU FP32 waveform in, CPU text out, at batch one. Timing includes feature
  extraction and host-device transfers, and excludes file I/O, loading, compilation and the
  first graph capture.
- **Runs:** each configuration ran in a fresh process, with the lengths in order. Each value is
  the median of three 5 s windows. Each Kea-B configuration ran in two processes, in opposite
  configuration orders, and the table shows their mean; the original's rows ran once.
- **Process VRAM** is the warm process's sampled peak, in decimal GB; for Kea-B, the two
  processes agreed exactly. One process runs the lengths in order and keeps up to four captured
  graph shapes, so process VRAM at the longer lengths includes graphs already captured for the
  shorter ones.
- **GPU energy** is NVML's board energy counter per transcription and excludes CPU and host
  energy.
- **Power:** the configured 400 W limit (board default 600 W) was not changed for any row.

## Comparing across sessions

The original's rows were measured with the same clips, protocol, GPU and driver in an earlier
session, not side by side with Kea-B. Kea-T's x3 runtimes, remeasured on the same clips against
their earlier records, show how well the numbers carry over: they agreed within 1% on latency but
only within 8% on energy, and the expanded runtime's process VRAM was identical. The latency and
memory comparisons are therefore solid, and the energy comparison is approximate.

## Requirements

The research runtime runs on SM120 GPUs (RTX 5090) only and was measured with torch 2.11.0+cu128,
triton 3.6.0, nemo_toolkit 3.0.0 and Python 3.13.
[RESULTS.md](https://github.com/ajbarryiii/wilderness-labs-stt/blob/main/finetune/parakeet-binary/gpu1bit/RESULTS.md)
has the evidence.

## Limitations

The runtime results apply only to the research runtime on GitHub. They were measured on one RTX 5090
at a 400 W limit, warm, at batch one, with one clip per length. Other GPUs, batch sizes, cold
starts and host energy were not measured.
