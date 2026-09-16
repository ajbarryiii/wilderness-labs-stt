# Binary speech model training

This is an executable training pipeline for the [488M binary CTC architecture](../BINARY_STT_5090_ARCHITECTURE.md). It streams speech from Hugging Face, trains with FP32 latent parameters and BF16 matrix operations on the RTX 5090, freezes a validation panel, and stops with durable diagnostics and email alerts when specified failure conditions occur. It is an experimental W1A1 speech model; recognition quality has not yet been established.

The full model has **488,270,080 parameters**, including **482,344,960 binary projection weights** at full quantization. The frontend, normalization, residual streams, attention probabilities/aggregation, depthwise convolution, and CTC head retain floating-point computation. Training simulates the binary forward calculation with differentiable matrix operations. Packed XNOR/popcount inference and an exported incremental decoding runtime are separate work.

## Data and experiment sizes

Use these four sources in order. Exact revisions, configurations, licensing/provenance notes, excluded alternatives, overlap limitations, and links to publisher evidence are in [DATASETS.md](DATASETS.md).

| Stage | Sources | Purpose |
| --- | --- | --- |
| Pilot | LibriSpeech's 960 training hours and AMI's training split | Establish CTC learning, resume correctness, and quantization stability on human-transcribed speech. |
| Expansion | Add People's Speech `clean` | Broaden acoustic and speaking-style coverage; the full release's 30k+ hours must not be mistaken for the smaller `clean` subset. |
| Scale | Add YODAS-Granary `English` / `asr_only`, approximately 102.5k published hours | Supply a large pseudo-labeled corpus through streaming. The English source is approximately 11.3 TB; a complete local copy would exceed this machine's data disk. |

The earlier **100–200k unique-hour** recommendation remains an experiment target, not a demonstrated Chinchilla optimum for W1A1 ASR. This default mixture supplies roughly 100k-plus published hours, before filtering and overlap audits; it does not yet establish 200k unique usable hours. MLS English and the other Granary components are expansion routes requiring separate audio acquisition/preparation. Do not count multiple microphone views, repeated epochs, overlapping recordings, or augmented speech as additional unique content.

The presets separate model size from a bounded first-run budget:

| Preset | Encoder dimensions | Maximum steps | Maximum audio exposure | Maximum elapsed time | Quantization |
| --- | --- | ---: | ---: | ---: | --- |
| `smoke` | 2 layers, width 32, FFN 128 | 4 | 1 hour | 1 hour | Reaches W1A1 for the last two steps; generated fixture only. |
| `pilot` | 4 layers, width 256, FFN 1024 | 2,000 | 200 hours | 24 hours | Weight ramp steps 100–400; activation ramp 400–1,000. |
| `full` | 20 layers, width 1,024, FFN 4,096 | 100,000 | 10,000 hours | 720 hours | Weight ramp 2,000–10,000; activation ramp 10,000–20,000. |

The first limit reached ends the run. These budgets are starting experiments, not a claim that 10k exposure hours fully trains the 488M model. The full preset uses 16 heads of width 64, an 80-bin 16 kHz causal log-mel frontend, 8× subsampling, and a 2,048-class CTC head. Four encoded frames per attention chunk correspond to a 320 ms chunk span; this is not a measured microphone-to-text latency. The training attention mask retains 64 previous encoded frames.

Microbatches contain at most four utterances and nominally 40 seconds of audio; one longer-than-budget utterance can still form a singleton. The reader rejects segments outside 0.25–20 seconds without cropping their transcripts. Pilot/full gradient accumulation is 4/8 microbatches. Audio duration budgets do not bound padded computation exactly. Profile real duration distributions before increasing batches or estimating total training time.

## Run on this machine

Use commands from `custom/`. The supplied `binary_stt/python` wrapper uses the existing NixOS Python runtime and CUDA dependency installation. It explicitly relocates Hugging Face, Torch, compilation, temporary, and package caches to `/mnt/hd/wilderness-labs-stt/binary-stt/`. It refuses to run if `/mnt/hd` is not mounted. The pinned package versions are in [requirements.txt](requirements.txt); this wrapper is configured for this host, not a portable environment installer.

Inspect the architecture or run the bounded offline fixture:

```bash
./binary_stt/python -m binary_stt inspect --preset full
./binary_stt/python -m unittest discover -s binary_stt/tests -v
./binary_stt/python -m binary_stt smoke \
  --run-dir /mnt/hd/wilderness-labs-stt/binary-stt/runs/smoke-001
```

The smoke command writes tiny synthetic FLAC-in-Parquet fixtures, uses the real HF streaming API, prepares validation, trains two optimizer steps, then resumes through step four. It disables outbound notifications and checks plumbing rather than speech accuracy. Use a fresh run directory each time.

Probe real upstream audio before spending a training budget:

```bash
./binary_stt/python -m binary_stt probe-data --preset full \
  --run-dir /mnt/hd/wilderness-labs-stt/binary-stt/preflight/full-001 \
  --per-source 1
```

This reads one accepted recording from each configured source/split and stores a metadata receipt. A successful probe does not audit the remainder of a corpus. The default sources are ungated; an optional `HF_TOKEN` environment variable raises authenticated Hub rate limits. Credentials are not part of saved configuration.

Create a reviewable pilot configuration, edit it before preparation, and freeze the run:

```bash
./binary_stt/python -m binary_stt write-config --preset pilot \
  --output /mnt/hd/wilderness-labs-stt/binary-stt/configs/pilot-001.json
./binary_stt/python -m binary_stt prepare \
  --config /mnt/hd/wilderness-labs-stt/binary-stt/configs/pilot-001.json \
  --run-dir /mnt/hd/wilderness-labs-stt/binary-stt/runs/pilot-001
```

Preparation streams transcript columns to fit a SentencePiece BPE tokenizer with byte fallback, then downloads only the bounded held-out audio panel. It validates CTC feasibility, keeps separate official validation splits, and freezes source revisions, tokenizer hash, validation hash, code hashes, runtime versions, and configuration. Reuse a trained tokenizer through `tokenizer.path`, with its files under the data-disk project root. A token is never silently discarded because it is outside an ASCII alphabet.

After configuring and testing email below, launch the supervisor:

```bash
./binary_stt/python -m binary_stt \
  --env-file /mnt/hd/wilderness-labs-stt/binary-stt/secrets/email.env \
  train --run-dir /mnt/hd/wilderness-labs-stt/binary-stt/runs/pilot-001
```

Keep the supervisor alive in the terminal or an existing session manager. It acquires the shared project GPU lock and checks for foreign compute processes. It does not kill other GPU jobs or alter the GPU power limit. No long pilot/full training job is started by preparation or inspection.

## Email stop alerts

The recipient is **ajbarryiii@gmail.com**. Default pilot/full configurations require an email transport to be configured before the training worker starts. SMTP STARTTLS on port 587 and TLS from connection start on port 465 are supported. The worker and watchdog can both alert, so one fatal event may produce both a specific failure email and a supervisor exit email.

A private template has been created at `/mnt/hd/wilderness-labs-stt/binary-stt/secrets/email.env` with mode `0600`. For another installation, copy [email.env.example](email.env.example) to that location on the mounted data disk and make it readable only by its owner. Preserve an existing credential file.

Fill `BINARY_STT_SMTP_PASSWORD` locally. For the included Gmail example, use an app password; Google's [app-password setup](https://support.google.com/accounts/answer/185833) requires 2-Step Verification and availability depends on account settings. An existing relay can use its own host, sender and credentials. Do not place the password in a command-line argument or chat. The file parser accepts plain quoted `KEY=value` assignments and never executes shell code.

Test actual delivery:

```bash
./binary_stt/python -m binary_stt \
  --env-file /mnt/hd/wilderness-labs-stt/binary-stt/secrets/email.env \
  test-email --run-dir /mnt/hd/wilderness-labs-stt/binary-stt/preflight/email
```

The command exits nonzero on delivery failure and writes a local receipt. Training's preflight checks configuration; a successful test is how to verify that credentials and delivery work. No credential has been supplied and no real email delivery has been verified yet.

`curl` can also submit email over SMTP, but it needs the same sending service and authentication. See [curl's SMTP documentation](https://everything.curl.dev/usingcurl/smtp.html). The pipeline uses Python's SMTP client so alerts are available directly in exception handlers and the watchdog. An email transport failure never allows training to continue after a detected collapse: the run stops, and the delivery failure is recorded locally. A host-wide power/network outage can prevent immediate email; detecting that from outside the machine would require an external monitor.

## Training recipe and BitNet adaptations

Read [BITNET_TRAINING.md](BITNET_TRAINING.md) for the examined official repository revision, code/PDF references, and the distinction between transferable ideas and this model's experimental choices. Microsoft's current BitNet GPU code is ternary-weight/INT8 inference using `dp4a`; it is not a binary-activation popcount training pipeline.

Implemented training choices include identity straight-through estimators; deterministic `-1/+1` signs; learned positive output-row scales and activation thresholds; FP32 normalization/statistics, latent weights and AdamW state; BF16 CUDA matrix operations; activation checkpointing; configurable weight then activation quantization ramps; and a fully quantized final phase. FP32 parameters and optimizer state are retained at all stages. A partly blended checkpoint is not deployable as a fully binary model.

AdamW begins at a configurable peak learning rate of `2e-4`, betas `(0.9, 0.95)`, and matrix weight decay `0.01`; norms, thresholds, scales and biases are exempt. Warmup precedes cosine decay and a final cooldown with weight decay zero. Gradients are clipped at norm 1.0. These are pilot hyperparameters, not a proven optimum transferred from language modeling.

Training-only SpecAugment provides mild frequency/time masking after a configurable start step. It preserves padding, length and target alignment; validation is unaugmented. The source labels already include teacher-generated supervision for YODAS. Aligned teacher-logit distillation, speed perturbation, background-noise mixing, reverberation and domain-specific acoustic augmentation are not implemented in this baseline. Teacher-logit options fail explicitly instead of being silently ignored.

The zero-quantization reference branch follows this architecture's computation graph. In particular, its binary FFN sign is introduced through the activation ramp; the unquantized FFN pair does not substitute a conventional SiLU/SwiGLU activation. Compare quantization schedules against a deliberate floating-point speech baseline before drawing accuracy conclusions.

## Monitoring and recovery

Numerical failures stop immediately: nonfinite inputs, CTC losses, gradients, latent parameters, buffers, or Adam state. Latent states are checked separately because sign conversion can conceal a NaN behind a finite binary code. Sustained loss explosions and near-zero gradients use a warmup, fixed reference, and patience. WER/empty-output/blank-collapse guards arm only after learning has produced useful validation predictions; an early all-blank CTC model or a loss plateau alone does not trigger a collapse alarm. Quantization transitions have bounded grace periods, never a NaN exemption.

The independent supervisor watches heartbeats written at actual progress boundaries. Default limits are five minutes for a blocked stream read, fifteen minutes for other work, and thirty minutes for a validation/checkpoint operation. A stalled worker is terminated as its own process group, escalated to SIGKILL after thirty seconds if necessary. Abnormal exits and missing terminal status records generate alerts. There is no automatic restart.

`latest.pt` is the latest finite optimizer-boundary recovery snapshot. `last_known_good.pt` is promoted after a completed validation whose guards are clear; the initial finite baseline qualifies numerically and does **not** imply good ASR accuracy. On collapse, suspect state is written to diagnostics rather than replacing either recovery checkpoint. At 488M with FP32 Adam, checkpoints are multi-gigabyte files, and atomic replacement temporarily needs additional free disk space.

Saved state includes parameters, optimizer, quantization schedule position, health monitors, Python/NumPy/Torch/CUDA RNG, HF iterators, the custom raw shuffle buffer, pending batch audio, and unique-data accounting. The pipeline deliberately uses one data iterator in the worker process; adding ordinary multiprocessing prefetch would need a new checkpoint contract. Resume rejects altered configuration, code, tokenizer, runtime or validation. CPU tests check exact sample/state continuation; GPU floating-point execution is not claimed to be bitwise deterministic.

Source shards are deterministically shuffled from the first pass, with seeds derived from the master seed, source identity and logical epoch. Each source also has a 32-row local shuffle buffer; the combined raw-buffer byte cap defaults to 256 MiB. This is not a global uniform shuffle or a cap on decoded Arrow tables and other working memory. The implementation uses public `set_epoch()` behavior verified in `datasets==5.0.1` and refuses a different version until replay is revalidated. Training and tokenizer sampling both shuffle shard order.

Remote-audio tests exposed a shutdown hang in the installed Arrow/HF asynchronous Parquet scanner. A small per-stream adapter uses synchronous `ParquetFile.iter_batches(use_threads=False)` while retaining HF's streaming loader, shard selection and iterator checkpoints. All callers close their iterators before Python shuts down. This version-specific adapter rejects unsupported filters rather than silently changing their behavior; verify remote exit and sample replay before upgrading HF/Arrow.

Inspect or request a graceful stop:

```bash
./binary_stt/python -m binary_stt status \
  --run-dir /mnt/hd/wilderness-labs-stt/binary-stt/runs/pilot-001
./binary_stt/python -m binary_stt stop \
  --run-dir /mnt/hd/wilderness-labs-stt/binary-stt/runs/pilot-001
```

The worker completes its current optimizer boundary before saving a normal stop. Remove the run's `STOP` file before intentionally resuming. `train --resume` explicitly resumes `latest.pt` after a normal stop or interrupted process; `--stop-after N` means absolute optimizer step N. A run recorded by the worker as collapsed or failed is refused for automatic same-run recovery. Diagnose it and prepare a reviewed new experiment. An externally killed worker may leave `status.json` saying `running`; consult `supervisor_status.json` before requesting manual resume.

Metrics and diagnostics live alongside the run:

| File | Contents |
| --- | --- |
| `metrics.jsonl` | Train loss, pre-clipping gradient norm, learning rate, quantization phase, aggregate/per-source validation WER/CER and collapse indicators. |
| `status.json`, `supervisor_status.json`, `heartbeat.json` | Worker progress, process outcome, and last real progress stage. |
| `unique.sqlite3` | On-disk exact decoded-PCM hashes and first optimizer step; resume rolls back uncheckpointed exposure. |
| `alerts.jsonl`, `failure.json`, `worker.log` | Durable notification events/delivery failures, failure diagnostics and worker output. |
| `config.json`, `prepared.json`, `validation_manifest.json` | Frozen recipe and provenance. |

Unique seconds measure distinct exact decoded waveforms used in optimizer steps. They do not establish unique recording content across lossy re-encodings, overlapping segments, or alternate microphone views. Reader counters include decoded/prefetched examples; optimized per-source exposure counts only consumed training examples. Global acoustic deduplication and broader domain/test evaluation remain required before a final data-scaling claim.

## Verification scope

The checked-in tests cover binary algebra/STE gradients, causal features and chunk masking, padding invariance, actual CTC backward passes, real HF local-Parquet replay, tokenizer byte fallback, optimizer/ledger recovery, and injected collapse/watchdog/SMTP failures. SMTP tests use mocks; they do not prove inbox delivery.

Final verification passed **74 tests**, a supervised CUDA fixture with stop/resume, and a four-step CPU run on real streamed LibriSpeech audio with a deliberate stop after step two and successful resume. All involved workers exited cleanly. The combined receipt, including code hashes and runtime versions, is `/mnt/hd/wilderness-labs-stt/binary-stt/verification/verification.json`.

On this machine, all nine configured remote source/split probes yielded and decoded real audio. A separate full-model check completed three optimizer steps at floating-point, 50% blended and fully W1A1 settings with input shape `[4,80,1000]` and output `[4,125,2048]`. Peak allocated GPU memory was about 9.12 GiB and reserved memory about 9.58 GiB, with FP32 parameters/Adam state and BF16 autocast. These are short synthetic execution checks, not training-throughput, speech-accuracy or packed-inference benchmarks.

The raw receipts are under `/mnt/hd/wilderness-labs-stt/binary-stt/verification/`. Repeat the bounded GPU check with a fresh output file:

```bash
./binary_stt/python -m binary_stt.gpu_check \
  --output /mnt/hd/wilderness-labs-stt/binary-stt/verification/gpu-full-002.json
```

All tokenizer assets, cached data, frozen audio, checkpoints and verification artifacts remain on `/mnt/hd`. No model weights belong in GitHub.
