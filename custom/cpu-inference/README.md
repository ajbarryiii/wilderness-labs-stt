# CPU inference efficiency

Custom packed binary/ternary inference for this machine's **AMD Ryzen 9
9950X3D**, with native scalar POPCNT and AVX-512 VPOPCNTDQ implementations.
The benchmark compares identical quantized models against a dense FP32
reference, then measures pretrained CTranslate2 FP32 and INT8 as external
controls. All inference runs offline on the CPU.

See [measured results](RESULTS.md) for the hardware sweep, complete-model checks,
and original sustained native-kernel comparison. The follow-up
[CPU kernel optimization](OPTIMIZATION_RESULTS.md) compares the original
AVX-512 backend with the optimized implementation.

The host has 16 physical cores and two L3 domains: cores 0–7 share 96 MiB;
cores 8–15 share 32 MiB. The harness discovers these from sysfs, tests explicit
physical-core placement without SMT, and records frequency policy and thermal
telemetry. It leaves the existing power policy in place.

## Implementations

`native.py` compiles `kernels.cpp` through the installed C++ compiler into the
mounted data disk. No Python extension framework or GPU compiler is needed.

- **Scalar:** hardware POPCNT, row-major packed weight planes.
- **AVX-512:** eight interleaved output channels per vector, with VPOPCNTDQ
  accumulating independent dot products in each lane. Runtime ISA checks
  prevent unsupported execution.
- **Optimized AVX-512 (`avx512_opt`):** retains the original packed layout and
  arithmetic, reuses weight vectors across multiple input rows, vectorizes
  scaling and bias, and specializes single-row decoder execution. The original
  `avx512` backend remains available for direct before/after measurements.
  Small decoder projections run serially; `threads` is the maximum native
  worker count. Bulk projections reuse each weight vector across eight rows.
  This is the default for `PackedWeight` and `CPUReplayWhisper`.
- **Dense:** quantized FP32 activation/code matmul followed by the same separate
  FP32 scaling and bias operations. This is a numerical reference and a
  within-runtime baseline.

W1A1 has binary weights and activations. W2A1 has ternary weights and binary
activations. W2A2 has ternary weights and activations. Binary activation
quantization is `x >= 0 ? +1 : -1`; ternary uses +1 for `x >= 0.5`, -1 for
`x <= -0.5`, and zero otherwise. Weight packing is performed once; activation
packing, scaling, bias and output allocation are included in inference.
Ternary weight nonzero counts are precomputed for W2A1. W2A2 computes the
activation/weight nonzero intersection. Native objects retain compact weights
and reusable activation scratch; embeddings are decoded on lookup.

`runtime.py` reuses the existing Whisper-shaped graph with FP32 CPU attention,
normalization and residual arithmetic, tied token/output weights, fused QKV
projections, and growing static KV caches. Quantized convolution uses the same
explicit im2col operation in every custom arm. These are **seeded random
models**, matching the purpose of the previous GPU efficiency experiment.
Neither forced replay nor equality to the dense twin establishes recognition
quality. CPU FP32 arithmetic also prevents a claim of exact numerical
equivalence to the earlier GPU FP16 models.

## Run

Use the existing inference experiment's installed runtime and prepared files.
If these are missing, follow [its setup instructions](../inference-efficiency/README.md).
No new model downloads are required. From the repository root:

```sh
custom/cpu-inference/python custom/cpu-inference/benchmark.py inspect
custom/cpu-inference/python -m unittest discover -s custom/cpu-inference -p 'test_*.py'

# Projection shapes from the medium.en encoder, decoder and vocabulary head.
custom/cpu-inference/python custom/cpu-inference/benchmark.py micro

# Fresh-process whole-model tuning; first frozen clip, three measured replays.
custom/cpu-inference/python custom/cpu-inference/benchmark.py screen

# Sustained confirmation: three alternating windows, >=60 s and all 16 clips.
custom/cpu-inference/python custom/cpu-inference/benchmark.py run \
  --reference-backend scalar --placements largest_l3_physical:4 \
  --variants w1a1-scalar w1a1-avx512 w2a2-scalar w2a2-avx512

# Compare the optimized kernels with the original AVX-512 implementation.
custom/cpu-inference/python custom/cpu-inference/benchmark.py run \
  --reference-backend avx512 --placements largest_l3_physical:4 \
  --variants w1a1-avx512 w1a1-avx512_opt w2a2-avx512 w2a2-avx512_opt
```

This confirmation isolates AVX-512 against the same-model scalar POPCNT
baseline. Both paths are checked against dense arithmetic in kernel and graph
tests; the initial full-size screen also compares dense outputs. The sustained
report separately checks all 16 clips against its selected scalar reference.
Use `--reference-backend dense` (the default) and include each family's `-dense`
variant to repeat the complete dense comparison. Include `ct2-fp32 ct2-int8`
for external controls.

The optimization comparison validates all 16 clips against the original
AVX-512 backend in the same run. Reports name the selected reference explicitly;
an AVX-512 reference is not reported as a new full-workload dense comparison.

Use `--placements largest_l3_physical:1 largest_l3_physical:2
largest_l3_physical:4 largest_l3_physical:8 smallest_l3_physical:8
all_physical:16` to sweep physical thread counts and L3 domains. Use `--variants`
to select arms. `screen --clips 16 --cycles 1` checks and times the entire
workload without claiming sustained energy evidence. `run` refuses reduced
clip counts, fewer than three windows, or windows shorter than 60 seconds.

Every worker uses a new process, explicit affinity, one inter-op thread,
explicit OpenMP/MKL thread counts, and passive OpenMP waiting. Measurement
runs serialize through a project lock. The package energy counter includes
other host processes, so run with other compute workloads idle. The report
records estimated background CPU activity rather than assuming affinity
isolates package power.

## Measurement and artifacts

The workload is the same 16 frozen 30-second speech clips and 128 predetermined
decoder tokens used for the 5090. Every model processes the audio frontend,
complete encoder, decoder prefill, 128 sequential vocabulary projections and
greedy reductions. Construction, compilation and weight packing are excluded.
Each clip is validated before the measured window. CTranslate2's forced-prefix
contract is checked on each inference call.

Results include per-clip latency, median/p95, real-time factor, package joules
per clip/audio-second, average watts, idle-subtracted energy, peak process RSS,
raw energy samples, output hashes, model storage, compiler provenance, the
loaded native library's SHA-256, and
source/workload hashes. Screen and microbenchmark results are tuning evidence;
only a completed sustained run with numerical checks can qualify custom-model
energy results. CTranslate2 uses pretrained weights in a separate runtime and
has its own precision, so comparisons against it are throughput comparisons.
Package versions and CPU backend environment overrides are frozen for every
new run. To test CTranslate2's alternative backend, prefix a command with
`CT2_USE_MKL=1`; this override is recorded and verified in each worker.

Power measurement uses readable Linux powercap **package** cumulative energy,
with a read-only perf energy-event fallback. It handles counter wrap and
rejects resets, stalled counters, read failures and multiplexed perf data.
Core subdomains are not added to package energy. Missing access produces null
energy and watts, never an estimate from TDP or CPU utilization. This is CPU
package energy, excluding discrete GPU power, wall-plug losses and any memory
power outside the package.

On this NixOS host the package counter initially requires root. To permit
read-only measurement for the current boot:

```sh
sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
```

All outputs, compiled libraries, caches and temporary files live under
`/mnt/hd/wilderness-labs-stt/cpu-inference/`. Existing model/audio artifacts are
read from `/mnt/hd/wilderness-labs-stt/inference-efficiency/`. Both the wrapper
and artifact helpers refuse writes when `/mnt/hd` is unmounted. No weights,
prepared audio or compiled libraries belong in Git. Each run has `config.json`,
source snapshots, raw window JSON, `summary.json`, `status.json` and a report.

The measurement interfaces follow the [Linux powercap documentation](https://cdn.kernel.org/doc/html/latest/power/powercap/powercap.html).
CTranslate2 thread configuration follows its [parallelism documentation](https://opennmt.net/CTranslate2/parallel.html);
the installed 4.8.2 CPU runtime supports FP32 and INT8/FP32, which are checked
explicitly to reject precision fallback.

The external control's [v4.8.2 decoding loop](https://github.com/OpenNMT/CTranslate2/blob/v4.8.2/src/decoding.cc)
computes vocabulary logits and TopK before replacing each prediction with the
forced token. It does not prefill the text prefix in parallel. Prefix equality
checks scheduling, not recognition quality. This workload disables beam search,
timestamps, token suppression and free-running stopping. Its SOT prefill skips
the final output normalization that the custom runtime computes; neither
runtime projects the SOT prefill to vocabulary logits.
