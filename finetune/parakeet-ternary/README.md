# Ternary-weight QAT of Parakeet-TDT-0.6B-v2

**Published:** [rajb3/parakeet-tdt-0.6b-v2-ternary](https://huggingface.co/rajb3/parakeet-tdt-0.6b-v2-ternary)
(CC-BY-4.0). Test results: [`results/TEST.md`](results/TEST.md). The upload folder is built
by `hf/stage.py` (52 required checks, including identical tensors and transcripts between the
standalone `hf/load_ternary.py` and this repository's loader) and was verified on a private
repository, including a download-and-transcribe round trip, before being made public.

Design: [DESIGN.md](DESIGN.md). Run every script with `./python` (pinned NeMo 3.0 /
torch 2.11 runtime; all caches and artifacts on `/mnt/hd/wilderness-labs-stt/parakeet-ternary/`).

## CUDA runtime

[`inference/`](inference/) runs this checkpoint on the GPU from its packed ternary
weights, using CUDA graphs and a fused decoder. The table shows warm batch-one
transcription of a 10 s clip on one RTX 5090 at a 400 W power limit:

| Configuration | Latency | Process VRAM | GPU energy |
| --- | ---: | ---: | ---: |
| Ternary expanded — optimized by Wilderness Labs | **4.46 ms** | 1.89 GB | **1.57 J** |
| Ternary compact — optimized by Wilderness Labs | 5.27 ms | **1.27 GB** | 1.86 J |
| Original BF16 — optimized by Wilderness Labs | 5.81 ms | 2.18 GB | 1.90 J |
| ONNX ASR / ORT CUDA — off the shelf | 15.74 ms | 4.14 GB | 3.75 J |

Both ternary modes use the same trained export:

- Compact (`--optimized`) keeps the ternary matrices as packed 2-bit codes with per-row
  FP32 scales, unpacked inside the kernels, and has the lowest VRAM.
- Expanded (`--optimized --encoder-storage expanded`) also caches an exact INT8 copy of
  the codes (about 604 MB) and has the lowest latency and energy.

Both run the ternary matrices on INT8 Tensor Cores and keep the non-ternary tensors in
floating point.

The two modes produce identical text, tokens and timestamps on 2,048 validation
utterances. The BF16 row is NVIDIA's original checkpoint with our own CUDA graphs and
fused FP32 decoder, not stock NeMo. The ONNX row is the unmodified onnx-asr package.

Results for 3, 10 and 30.04 s clips, the method, usage and tests are in
[`inference/README.md`](inference/README.md), with evidence in
[`inference/results/rtx5090.json`](inference/results/rtx5090.json). Accuracy for this
checkpoint is the full-corpus WER in [`results/TEST.md`](results/TEST.md).

## Model and evaluation

### Files

| File | Role |
| --- | --- |
| `quant.py` | Quantized set (`QUANTIZED_PATTERN`, 264 modules), `TernaryLinear` / `TernaryPointwiseConv1d`, `quantize_parakeet`, ramp (`set_weight_fraction`, `weight_fraction`), `code_histogram`, `parameter_accounting`. The quantizer math is imported unchanged from `finetune/whisper-ternary/quant.py` (by file path, as `whisper_ternary_quant`). |
| `export.py` | `export_model` (2-bit packed codes + FP32 row scales, FP16 elsewhere, NeMo config, tokenizer files, SHA-256 manifest), `load_export` (plain FP32 NeMo model, no `.nemo` needed), `reconstruction_check`. |
| `testsets.py` | Open ASR Leaderboard test sets to 16 kHz mono FLAC + `manifests/test_<set>.jsonl`. |
| `evaluate.py` | Shared decoding (`transcribe_records`, `decode`), normalizer, corpus WER, CLI, B1 PTQ (`--ptq`). |
| `teacher.py` | In-loop teacher labeling for streamed training batches (`teacher_label_batch`) and its throughput measurement. |

```bash
P=finetune/parakeet-ternary
$P/python -m unittest discover -s $P/tests -v                           # from the repository root; loads the model, so run it via ./heavy
$P/python $P/testsets.py                                                  # all test sets, ~15 min CPU
$P/python $P/evaluate.py --source pretrained --sets test --out-dir /mnt/hd/wilderness-labs-stt/parakeet-ternary/eval/b0-pretrained
$P/python $P/evaluate.py --source pretrained --ptq --sets librispeech_clean librispeech_other ami --out-dir .../eval/b1-ptq
$P/python $P/evaluate.py --source export --path .../runs/<run>/export --sets test --out-dir .../eval/<run>
$P/python $P/teacher.py                                                   # teacher labeling throughput
```

`evaluate.py` takes the GPU lock, writes `<set>.json` per set (source block with weight
SHA-256 and base revision, decoding config, WER counts, wall time, per-utterance records)
and `summary.json` (per-set WER, the mean over the seven DESIGN.md sets, and for the
pretrained model the published WER and the 0.2-point gate). `--sets dev` reads the data
preparation's `manifests/<name>.jsonl` for `paths.DEV_SETS`.

### Scoring

- Decoding: preprocessor -> encoder -> `model.decoding.rnnt_decoder_predictions_tensor`
  (greedy TDT, the model's own decoding config), i.e. the batched path of
  `model.transcribe()`, with its inference settings (eval, dither 0, pad_to 0), strict FP32
  (TF32 and autocast off). Called directly because `transcribe()` unfreezes all modules
  when it finishes. Checked identical to `transcribe()` outputs and invariant to batch size.
  Batches: descending duration, at most 64 utterances and 1,200 s of audio.
- Normalizer: transformers' Whisper `EnglishTextNormalizer` with Whisper's
  `normalizer.json` spelling map (SHA-256 pinned). Verified identical to the leaderboard's
  normalizer at commit `431dd91` (the 2025 version NVIDIA's numbers come from) on all
  42,716 references available at the time of the check (spelling map identical; the
  leaderboard added name/acronym/compound rules only in 2026-04).
- Utterances shorter than 0.03 s (`MIN_DECODE_SECONDS`) are zero-padded to 0.03 s before
  feature extraction, for every source and arm: NeMo's per-feature normalization fails on a
  one-frame input. Only one utterance is affected (ami_dev, 0.020 s); every test-set utterance
  is at least 0.040 s, so test results are unchanged. Counts are recorded as `padded_short`.
- As in the leaderboard, utterances whose normalized reference is empty (AMI: 990,
  Earnings-22: 4, GigaSpeech: 33) are decoded but not scored.
- Test-set audio: soundfile decode, channel mean, soxr HQ resampling to 16 kHz (as
  `datasets.Audio(sampling_rate=16000)` did), stored as 16-bit FLAC (the leaderboard's
  cache files are 16-bit WAV). Common Voice repeats 5,673 ids with different audio; repeats
  get a `#k` suffix and are scored on their own audio (the leaderboard's NeMo script
  would re-use the first clip's cached file for a repeated id).

### Test sets (hf-audio/esb-datasets-test-only-sorted @ b6bdcd0beb)

| Set | Utterances | Hours | Longest (s) |
| --- | ---: | ---: | ---: |
| LibriSpeech test-clean | 2,620 | 5.40 | 35.0 |
| LibriSpeech test-other | 2,939 | 5.34 | 34.5 |
| AMI | 12,643 | 8.68 | 26.2 |
| Earnings-22 | 2,741 | 5.43 | 36.4 |
| GigaSpeech | 19,931 | 35.36 | 22.0 |
| SPGISpeech | 39,341 | 100.00 | 15.0 |
| VoxPopuli | 1,842 | 4.93 | 69.2 (1 over 40 s) |
| Common Voice | 16,334 | 27.00 | 105.7 (5 over 40 s) |

### B0: FP32 original (reproduction gate)

FP32, greedy TDT, RTX 5090, 15.4 min of decoding for 192 h of audio.

| Set | Ours | Published | Diff | Gate (±0.2) |
| --- | ---: | ---: | ---: | --- |
| LibriSpeech test-clean | 1.70 | 1.69 | +0.01 | pass |
| LibriSpeech test-other | 3.19 | 3.19 | +0.00 | pass |
| AMI | 11.15 | 11.16 | -0.01 | pass |
| Earnings-22 | 11.24 | 11.15 | +0.09 | pass |
| GigaSpeech | 9.78 | 9.74 | +0.04 | pass |
| SPGISpeech | 2.14 | 2.17 | -0.03 | pass |
| VoxPopuli | 5.94 | 5.95 | -0.01 | pass |
| **Mean (7 sets, no TED-LIUM)** | **6.45** | **6.44** | +0.01 | |
| Common Voice (separate) | 8.50 | n/a | | |

### B0 on the development sets (selection metric)

FP32 original, full development sets (`eval/B0-dev`), 1.6 min of decoding for 28.5 h:

| Set | Utterances (scored) | Hours | WER |
| --- | ---: | ---: | ---: |
| LibriSpeech dev-clean | 2,703 | 5.39 | 1.59 |
| LibriSpeech dev-other | 2,864 | 5.12 | 2.94 |
| AMI dev | 13,098 (12,207) | 8.94 | 12.12 |
| VoxPopuli dev | 1,753 | 4.98 | 5.75 |
| YODAS dev | 2,000 | 4.09 | 5.47 |
| **Mean (5 sets)** | | | **5.58** |

One utterance padded (ami_dev, 0.020 s).

### B1: ternary PTQ, no training

All 264 modules ternary (603,979,776 of 617,825,926 parameters), codes -1/0/+1 =
33.8% / 32.4% / 33.8%. The rebuilt export emits empty transcripts for every utterance:
100.00% WER (all deletions) on LibriSpeech test-clean, test-other and AMI.

Export (`runs/b1-ptq/export`): 180.8 MB safetensors (2.47 GB FP32 original):
packed codes 151.0 MB, row scales 1.77 MB, FP16 float tensors 27.8 MB, FP32 feature
constants 0.13 MB, header 0.11 MB; tokenizer files 0.27 MB.
Reconstruction check (GPU, four fixed LibriSpeech test-clean utterances): codes and scales
exact; encoder output max abs diff 0.0013 (max |output| 2.06); joint output max abs diff
0.030 teacher-forced on the 197 reference tokens; greedy hypotheses equal (all empty).

### Teacher labeling throughput (`teacher.py`)

2,000 seeded LibriSpeech train-clean-100 utterances (7.0 h), unsorted batches capped by
total audio seconds, timing `teacher_label_batch` only (audio already decoded):

| Batch audio | BF16 encoder: audio h / GPU h | peak GB | FP32: audio h / GPU h | peak GB |
| ---: | ---: | ---: | ---: | ---: |
| 300 s (23 utts) | 2,188 | 3.9 | 1,337 | 5.5 |
| 600 s (47 utts) | 2,582 | 5.2 | 1,451 | 7.9 |
| 1,200 s (91 utts) | 2,827 | 8.0 | 1,369 | 13.1 |

BF16-encoder labels match FP32 labels exactly on 98.6-98.8% of utterances (normalized WER
between them 0.01-0.02%). FP32 teacher vs LibriSpeech human transcripts: 1.45% WER.

## Data

| File | Role |
| --- | --- |
| `data.py` | Development sets, stored locally: download, 16 kHz mono FLAC extraction, manifests, 400-utterance subsets, YODAS reserved shards; `load_manifest(name)` for other modules. |
| `stream.py` | Training data, **streamed from the Hugging Face Hub during training** (nothing stored): `make_training_stream(phase, seed, num_workers=8)`, resumable with `state_dict()` / `load_state_dict()`. |
| `tests/test_data.py`, `tests/test_stream.py` | Unit tests (no network): manifests, filters, subsets, reserved shards, resampling, stream mixture, resume, retries, memory bound. |

```bash
P=finetune/parakeet-ternary
$P/python -m unittest discover -s $P/tests -v -p 'test_data.py'
$P/python -m unittest discover -s $P/tests -v -p 'test_stream.py'   # ~35 s; writes a 205 MB temp file under ARTIFACTS/tmp
$P/python $P/data.py dev        # rebuild/verify development sets (resumable)
$P/python $P/data.py status
$P/python $P/stream.py sources  # list training files, write data/stream/sources.json
```

Anything that may use more than 2 GB of RAM, any streaming benchmark and any GPU job runs
through `./heavy NAME [--mem-max 16G] --runtime <bound> [--wait] -- python ...` (a
memory-capped systemd user unit, logs in `logs/NAME.log`), one heavy job at a time
(`systemctl --user list-units 'parakeet-*'`).

### Development sets (local)

Manifests: `manifests/<name>.jsonl`, one JSON object per line with exactly
`audio_filepath` (absolute), `duration` (s), `text` (reference), `id`
(`<source>:<corpus id>`, unique), `source`; sorted by id. `<name>.meta.json` holds the
repository, config, split, revision, license, text column, counts, hours, drops per
reason and the SHA-256 of the `.jsonl`. Dev sets are complete (no duration filter;
only undecodable/empty/all-zero audio or an empty reference would be dropped; none
were). `<name>_400` is every k-th utterance of the id-sorted set.

| Manifest | Source (revision) | Utterances | Hours | `_400` hours |
| --- | --- | ---: | ---: | ---: |
| `librispeech_dev_clean` | OpenSLR SLR12 dev-clean (reused from the Whisper experiment) | 2,703 | 5.39 | 0.80 |
| `librispeech_dev_other` | OpenSLR SLR12 dev-other (ELDA mirror; tarball SHA-256 matches torchaudio's) | 2,864 | 5.12 | 0.71 |
| `ami_dev` | `edinburghcstr/ami` ihm validation @ `46f28f2` | 13,098 | 8.94 | 0.28 |
| `voxpopuli_dev` | `facebook/voxpopuli` en validation @ `42f0187`, `normalized_text` (the ESB column) | 1,753 | 4.98 | 1.12 |
| `yodas_dev` | `espnet/yodas-granary` English asr_only @ `9699445`, reserved shards | 2,000 | 4.09 | 0.80 |

All sources were already 16 kHz; AMI and VoxPopuli are stored as float WAV in parquet
and were converted to 16-bit FLAC (clipped), YODAS (16-bit WAV) re-encoded, nothing
resampled. `yodas_dev` text is the Granary pseudo-label (faster-whisper-large-v3 plus
Qwen2.5 punctuation/case restoration), not a human transcript.

**Reserved YODAS shards.** 16 of the 18,496 English asr_only parquet files, chosen by
`np.random.default_rng([20260930, 1]).choice(18496, 16, replace=False)` over the sorted
paths, are recorded with their row counts and all 637 source recordings
(`original_audio_id`) in `manifests/yodas_reserved_shards.json`. `yodas_dev` reads only
the metadata columns of those shards plus 4 seeded row groups each (64 row groups, about
2 GB over HTTP range requests); it keeps 2,000 of the 5,516 candidate utterances of 1-30 s
with nonempty text (157 recordings) and stores only their audio. Training never lists the
reserved files and drops any row whose recording is among the 637.

### Training stream (Hub)

Sources at pinned revisions (file lists and their SHA-256 in `data/stream/sources.json`;
the YODAS list is also in `data/stream/yodas_train_files.json`):

| Source | Repository @ revision | Parquet files | Hour share | Per-item probability |
| --- | --- | ---: | ---: | ---: |
| YODAS-Granary English asr_only (minus 16 reserved) | `espnet/yodas-granary` @ `9699445` | 18,480 | 0.65 | 0.705 |
| LibriSpeech 960 h (`all/train.*`) | `openslr/librispeech_asr` @ `71cacbf` | 126 | 0.10 | 0.062 |
| People's Speech clean train | `MLCommons/peoples_speech` @ `f10597c` | 804 | 0.10 | 0.057 |
| VoxPopuli en train | `facebook/voxpopuli` @ `42f0187` | 30 | 0.10 | 0.075 |
| AMI IHM train | `edinburghcstr/ami` @ `46f28f2` | 42 | 0.05 | 0.101 |

DESIGN.md's shares are shares of exposure, so the per-item probability is share / mean
utterance duration (normalized). LibriSpeech is streamed too (one code path; the local
train-clean-100/360 copies are not used). Items: `{"audio": float32 16 kHz mono,
"duration", "text", "id", "source"}`; `text` is the human transcript, `None` for YODAS.
"pilot" and "main" are the same stream; the pilot just runs fewer steps.

- **Reading.** `huggingface_hub.HfFileSystem` + pyarrow with column projection (the layer
  `datasets` streaming uses), not `datasets.load_dataset`: in the installed `datasets`
  5.0.1 every Audio feature, even `decode=False`, imports torchcodec, which is not
  installed. Files are split across worker processes (file i to worker i mod N); each
  worker keeps 2 YODAS files and 1 file per other source open, draws the source with the
  mixture probabilities and a slot uniformly (seeded generators), and reads each file
  sequentially; file order is a seeded permutation per pass, and an exhausted source starts
  its next pass (AMI repeats; YODAS does not within this experiment). The parent takes
  items from workers in strict round robin.
- **Filters inline:** 1.0-30.0 s inclusive, decodable, nonempty, finite, not all zeros;
  counted per source and reason (`stream.counters()`). About 13% of YODAS rows are
  dropped (half over 30 s, half under 1 s) and about 36% of AMI rows (under 1 s).
- **Resume.** `state_dict()` (small, JSON-serializable) records per worker the generator
  states, each source's file cursor and each open file's absolute row, as of the last item
  the caller consumed. After `load_state_dict()` a rebuilt stream yields exactly the next
  items of the uninterrupted stream (tested in-process and with 2 workers); resuming re-reads
  at most one row group per open file. The state carries a hash of the configuration (file
  lists, shares and mean durations, seed, filters, open files per source, block size, worker
  count) and is refused with `ValueError` under any other configuration.
- **Network:** every read retries with backoff (5 s doubling to 5 min, 12 attempts, about
  30 minutes) and logs; a file that keeps failing raises `StreamError` naming the source; a
  dead worker restarts from its last consumed state (at most 5 times).

**Memory finding (2026-09-30 OOM).** pyarrow's default parquet `pre_buffer=True` keeps every
column chunk read so far alive while a file is open, so a worker's memory grew with the bytes
it had read (about 1.4 GB per 586 MB YODAS shard). With 14 open files per worker, an
8-worker benchmark reached about 7 GB per worker and, running beside a GPU probe without
swap, triggered a system-wide OOM. Fix: `ParquetFile(..., pre_buffer=False,
buffer_size=8 MB)`, `HfFileSystem.open(..., block_size=8 MB, cache_type="none")`, 6 open
files per worker. `tests/test_stream.py::MemoryRegressionTest` reads a 205 MB local parquet
end to end: RSS +10 MB with the fix, +92 MB with pre-buffering forced back on (bound 45 MB).

**Throughput** (CPU only, audio hours delivered per wall hour, after 60 s warm-up):

| Workers | Open files per worker | Measured | Audio h / h | Network MB/s | CPU cores | Total RSS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 14 (before the fix) | 241 s | 630 | 35.0 | 0.15 | 32.0 GB |
| 8 | 14 (before the fix) | killed by the OOM | | | | |
| 4 | 6 | 600 s | 493 | 16.9 | 0.09 | 8.6 GB (flat) |
| 6 (pilot) | 6 | 301 s | 766 | 26.8 | 0.15 | 13.0 GB (13.6 peak) |
| **8 (default since the main run)** | 6 | 300 s | **1,001** | 33.6 | | 16.8 GB (17.4 peak) |

Delivery is latency-bound, not CPU-bound. Training consumes about 580 audio hours per
hour (the P2 smoke run measured 570-586 audio-seconds per second with 0.00 s stream wait
per step at 6 workers), so 6 workers leave about 30% headroom and 4 would starve the GPU.
The pilot used 6 workers; the GPU still waited on data in about 10% of 100-step windows, so
the main run used 8 (about 7% of windows), and 8 is the default now. Zero retries in every run, and realized hour
shares within about 1 point of the targets (YODAS 0.658-0.664, AMI 0.044-0.052). Measured
mean kept durations: YODAS 7.76 s, LibriSpeech 12.39 s, People's Speech 14.39 s, VoxPopuli
10.21 s, AMI 3.90 s (the stream's per-item weights use the earlier estimates 7.05 / 12.30 /
13.33 / 10.23 / 3.79; changing them would invalidate stream checkpoints).

**Disk.** Dev audio 1.5 GB (`audio/{librispeech,ami,voxpopuli,yodas}`), dev raw parquet
2.4 GB (`data/hf`: AMI IHM validation and VoxPopuli en validation, kept for
re-extraction), dev-other tarball 0.3 GB. No training audio is stored.

## Training

`train.py` runs one (arm, learning rate) to completion in `RUNS/<run-name>`;
`sweep.py` runs the Phase 1 pilot and launches M1. Arms: P1 (TDT loss on teacher
transcripts, prediction and joint networks frozen), P2 (plus encoder-output matching),
P3 (P2 with trainable prediction and joint networks), M1 (`--recipe` = selected pilot
recipe), A1 (FP32 control, `--recipe` as well). Every arm loads the frozen teacher.

### Running (always inside a `./heavy` unit)

GPU and stream jobs run only through `./heavy` (memory-capped systemd user unit; log in
`/mnt/hd/wilderness-labs-stt/parakeet-ternary/logs/NAME.log`), one at a time; check
`systemctl --user list-units 'parakeet-*'` first. The sweep is launched as one unit; its
`train.py` children inherit the cgroup, and `sweep.py` refuses to start outside a
`parakeet-*` unit (`--allow-unconfined` overrides, for dry runs):

    ./heavy pilot --mem-max 36G -- python sweep.py --phase pilot
    ./heavy main  --mem-max 36G -- python sweep.py --phase main --max-steps N   # N recorded in DESIGN.md first
    ./heavy smoke --mem-max 36G --runtime 15min -- python train.py --arm P2 --lr 5e-4 \
        --run-name smoke --smoke --grad-checkpointing
    ./heavy probe --mem-max 36G --runtime 12min -- python train.py --arm P2 --lr 5e-4 \
        --run-name probe --stream local-librispeech --batch-seconds 600 --grad-checkpointing --probe-steps 30

Stop a run with `systemctl --user stop parakeet-NAME`: train.py checkpoints at the next
step boundary (a checkpoint takes 30-45 s, inside systemd's 90 s stop timeout) and exits
76. Rerunning the same command resumes. Never pass `--overwrite` unless a run is to be
discarded.

### Data and labels per step

The stream (`stream.make_training_stream(phase, seed)`, 8 download workers, spawn
processes) is read by a background producer thread that fills a bounded queue of
collated, pinned batches; the training loop only waits on the queue (measured stream
wait 0.00 s per step). Batches are duration-bucketed: pools of about 8 batches' worth
of audio are sorted by duration and cut so that count x longest utterance <=
`--batch-seconds`, then shuffled with a generator seeded by (seed, pool index);
utterances outside 1-30 s are dropped before bucketing. Each batch is labeled online by
`teacher.teacher_label_batch` (teacher encoder in BF16, greedy TDT decoding with the
model's config); rows it rejects (DESIGN.md filters) leave the batch, a batch with no
kept rows advances the stream without an optimizer step, and the teacher's encoder
output is reused for encoder matching (no second teacher forward). Drop counts and
kept seconds per source are in `metrics.jsonl` and `summary.json`.

**Augmented student, clean teacher (decision for review).** The student computes its
own log-mel features (preprocessor in training mode, i.e. the model's 1e-5 dither) and
applies the model's SpecAugment (2 frequency masks of width up to 27, 10 time masks of
up to 5% of the length); the teacher encodes the clean audio (no dither, no
SpecAugment). Both the TDT loss and the encoder-matching loss use that single student
forward on the augmented input. This is standard feature distillation and follows
DESIGN.md literally (SpecAugment for the student only; teacher on unaugmented input).
It costs nothing extra and asks the student to reproduce the clean representation from
masked input, a denoising target. The alternatives were a second, clean student forward
for the matching loss (about +60% student compute, and an unregularized matching term)
or turning augmentation off (contradicts the recipe). Known consequence: the matching
loss cannot reach zero on time-masked frames. If P2/P3 underperform P1, the first
variant to try is excluding time-masked frames from the matching loss.

Other choices: dropout in the prediction and joint networks (20% in the pinned config)
is set to 0 in every arm, so the arms differ only in the extra loss and in
trainability (DESIGN.md). Frozen (P1/P2) they run in eval mode, so the encoder is
trained through exactly the function the teacher decodes with; P3 trains them in train
mode (cuDNN's LSTM backward requires it) with dropout 0, a deterministic forward
(tested). The encoder keeps its configured dropout. `var(E_t)` in the matching loss is the population variance over
every element of every valid teacher frame in the batch. Weight decay applies to
parameters with dim >= 2 (including depthwise conv kernels). The ramp sets
f = min(1, k / round(0.25 x max_steps)) before update k; warmup is round(0.02 x
max_steps) updates, then linear decay to 0.

### Checkpoints and resume

`ckpt/latest.pt` (7.3 GB: latent FP32 weights incl. BatchNorm buffers, optimizer,
scheduler, Python/NumPy/torch CPU/CUDA RNG states, ramp step, metric accumulators,
best-checkpoint record, the stream position) is written every `--ckpt-minutes` (30), at
the last step, on SIGTERM/SIGINT (exit 76) and when the stream fails after its own
retries (exit 75; `sweep.py` waits 60 s doubling to 30 min and resumes, up to 20 times).
Writes go to `latest.pt.tmp`, are fsynced, and the old `latest.pt` becomes
`previous.pt` before the rename; loading falls back to `previous.pt` if `latest.pt` is
torn; a torn `latest.pt` is then renamed `latest.pt.unreadable-*` so the next save
cannot rotate it over the good `previous.pt`. If checkpoint files exist but none loads,
train.py refuses to start (exit 77, checked before any model is loaded) instead of
starting over; only `--overwrite` discards such a run. A resume must use the original
arguments: anything other than `--ckpt-minutes`, `--eval-batch-size` and `--no-powerlog`
differing from the arguments stored in the checkpoint makes train.py refuse (exit 78),
checked, like readability, before any model is loaded. After fsync each checkpoint, best checkpoint, export file and `best.nemo` is
dropped from the page cache (`posix_fadvise(DONTNEED)`), which is otherwise charged to
the unit's memory limit. A restart refuses a checkpoint written with different
arguments (arm, recipe, lr, steps, stream, batch seconds, seed, eval interval).

Stream position: each batch carries the stream's `state_dict()` from just before its
pool was read, so the position after a batch is (pool-start state, pool index, batches
consumed). Resume restores the stream to the pool start, re-reads that pool, rebuilds
the same batches and skips the consumed ones; no audio is stored in checkpoints. Because
the Hub stream resumes at the item level (stream.py), the resumed run sees the same
batches as an uninterrupted one. Bit-identical resume is not required by DESIGN.md; on a
deterministic stream it holds anyway (tested), because cuDNN/cuBLAS run in
deterministic mode, evaluation runs inside a save/restore of every RNG, and the cached
cuDNN LSTM dropout state is reseeded from the checkpointed CUDA generator each step.

Evidence files after a resume: `metrics.jsonl` is rewritten atomically to the lines with
`step` <= the checkpoint step (a torn last line is dropped), `dev-subset/<set>/step-*.json`
for later steps and stray `best-*.pt` files are deleted, so the replayed steps are
written once. Every metrics line carries `session` (0 for the first process, +1 per
resume); `resumes.jsonl` records each resume (step, position, arguments, source
hashes). Both logs are appended with a single `write()` of a complete line plus fsync. A
power cut during an append can still leave a torn last line: every reader (train.py,
sweep.py) ignores it with a warning, the next append or resume rewrites the file
atomically without it, and a damaged line anywhere but the end is an error. The unit log
in `logs/` is append-only and therefore shows the killed session's output too.

### Evaluation, selection, export

Every `--eval-every` steps (pilot: 1,200) the five 400-utterance dev subsets are decoded
with `evaluate.decode` (strict FP32, the current weight fraction); per-set records go to
`dev-subset/<set>/step-NNNNNN.json`.

Selection (`--select`, default and pilot policy `final`, DESIGN.md "Selection: lowest
full development mean WER at the final f = 1 checkpoint"): the scored and exported model
is the final one (step == max_steps, f == 1); no best weights are written, and the best
interim dev-subset step and WER are reported for information only
(`best_interim_subset_*` in summary.json). `--select best` instead keeps the f == 1
evaluation with the lowest dev-subset mean WER as `best-NNNNNN.pt`; every new best gets
a new name, nothing referenced by `ckpt/latest.pt` or `ckpt/previous.pt` is ever
deleted, unreferenced best files are collected only after a checkpoint is durable, and a
resume drops only best files newer than its checkpoint (tested with a crash between an
evaluation and the next checkpoint). The selected model is exported (`export.py`; f set
to 1), rebuilt with `export.load_export`, reconstruction-checked, and the rebuilt model
scores the full dev sets (`eval-dev.json`, `eval-dev/<set>.json`, both naming the
selected step); A1 is saved and restored as `best.nemo`. `summary.json` is written last, atomically, with type-checked
required keys. `sweep.py` reuses a finished run, or relaunches an unfinished one, only
if summary.json, config.json and every resumes.jsonl entry match the protocol: arm,
recipe, lr, steps, seed, batch seconds, gradient checkpointing, evaluation interval,
selection policy (final), stream kind, phase and configuration hash, the quantized
module set (hash shared by the compared runs), the recipe flag values (P1: no encoder
matching, prediction/joint frozen; P2: matching, frozen; P3: matching, trainable; A1:
not quantized; M1: its recipe), and the current SHA-256 of every source that affects
training or scoring: `parakeet-ternary/{train,quant,stream,teacher,evaluate,export,data,
paths}.py` and the reused `whisper-ternary/{quant,wer}.py` (`powerlog.py` is excluded: a
separate recorder process that cannot affect results). The hashes are recorded at the
start (config.json `training_source_hashes`), at every resume (resumes.jsonl) and at
finishing (summary.json `source_hashes`), and all must match. train.py itself checks
the recipe flags against these values and verifies the configured model (trainable
parameters, ternary modules present iff quantized, prediction/joint dropout 0) before
training. Any mismatch stops the sweep with the reason; nothing is reused, replaced or
restarted silently. Exits 77 and 78 from train.py also stop it.

`sweep.py --phase pilot`: P1, P2, P3 at 5e-4; the lowest full-dev mean WER picks the
recipe (each scored at its final checkpoint); that recipe at 2e-4 and 1e-3; the lowest
of its three runs is written to `RUNS/pilot-selection.json` with every candidate. Stop
rule: if every pilot run's dev mean is above 3x the FP32 original's
(`EVAL/B0-dev/summary.json`, evaluate.py percent values converted to fractions; B0 dev
mean 5.58%, so the threshold is 16.7%), the selection records it and the main phase
refuses. The B0 file is validated before the first pilot run (pretrained source, all
five dev sets, no limit); the pilot does not start without it.

### Environment fixes inside train.py

NeMo's TDT loss is a numba CUDA kernel and this runtime has no CUDA toolkit for numba.
train.py points numba at the Nix CUDA 12.9 libNVVM through a `CUDA_HOME` shim
(`caches/cuda-home/nvvm/{lib64,libdevice}` symlinks), targets compute_90 PTX
(`NUMBA_FORCE_CUDA_CC=9.0`; the driver JIT-compiles it for sm_120, while NVVM 12.9
segfaults on compute_120), and rebuilds NeMo's `compute_tdt_grad_kernel` and
`compute_grad_kernel` (the omega branch) with their two `min`/`max` clamp lines
rewritten as conditional expressions, because numba 0.67 cannot compile builtin
`min`/`max` in device code. The clamp branch is disabled in this config. Tests compare
the TDT loss and gradient with NeMo's pure-PyTorch reference and the omega branch with
finite differences. Gradient checkpointing wraps each conformer layer
(non-reentrant); arguments are passed positionally so the CUDA RNG is restored for the
recompute (identical dropout masks), and BatchNorm running statistics are restored after
the recompute, so checkpointing is bit-identical to no checkpointing (tested).

### Measured throughput and memory (Phase 0)

Probe: `train.py --probe-steps` (2 untimed worst-case updates on 30 s utterances, built by
concatenating LibriSpeech, then timed updates), stand-in stream = local LibriSpeech
train-clean-100, online teacher labeling included, quantized student at f = 0.5, 400 W
GPU power limit. Audio hours per GPU-hour = kept audio / (stream wait + labeling +
student update).

| Arm | Batch s | Grad ckpt | Worst case (30 s utts) | Typical peak alloc | s / step | Label s | Audio h / GPU-h |
| --- | ---: | :-: | --- | ---: | ---: | ---: | ---: |
| P1 | 300 | no | ok, 27.7 GB | 26.6 GB | 0.42 | 0.10 | 643 |
| P1 | 300 | yes | ok, 17.6 GB | 16.1 GB | 0.48 | 0.10 | 559 |
| P2 | 300 | no | ok, 27.7 GB | 26.6 GB | 0.42 | 0.10 | 647 |
| P2 | 300 | yes | ok, 17.6 GB | 16.1 GB | 0.48 | 0.10 | 562 |
| P2 | 600 | no | OOM | OOM | | | |
| **P2** | **600** | **yes** | **ok, 21.6 GB** | 19.9 GB | **0.87** | 0.17 | **621** |
| P2 | 1200 | no | OOM | OOM | | | |
| P2 | 1200 | yes | OOM | 27.3 GB | 1.70 | 0.32 | 635 |

Chosen: 600 s batches with gradient checkpointing (the fastest setting whose worst case
fits with headroom; 1200 s is 2% faster but fails on a batch of 30 s utterances). The
Hub-stream smoke run (P2, 600 steps, 6 stream workers) measured 0.83 s per step,
570-586 audio-seconds per second, stream wait
0.00 s per step, process RSS steady at about 18.5 GB, GPU memory 27-29 GB as seen by
nvidia-smi (23 GB peak allocated by PyTorch), 375 W. That is about 580 audio hours per
GPU-hour, against 766 audio hours per hour delivered by the stream: the GPU is the
bottleneck, with about 30% stream headroom.

Pilot sizing: `PILOT_STEPS = 12000` (sweep.py) = 2.77 h of steps at 0.83 s plus about
7 minutes of overhead (model loading, ten dev-subset evaluations at 4-8 s, about six
30-minute checkpoints at 30-45 s, export and full dev scoring about 75 s) = about 2.9
GPU-hours per arm, 1,600 audio hours per arm (479 kept audio-seconds per step).

## Known limitation

`evaluate.py --source checkpoint` expects a `model` key, but training checkpoints
(`runs/<run>/ckpt/latest.pt`) store the weights under `student`, so evaluating a raw
training checkpoint fails. Every reported number was scored on an export
(`--source export`), and `evaluate.py` is part of the frozen, hash-checked scoring code,
so this is documented rather than changed.
