# Open speech data for the binary CTC training pipeline

Checked 2026-09-13 against publisher dataset cards and live Hugging Face repository metadata. The four primary repositories below are ungated and publish Parquet with embedded audio. Configurations, columns, and commit IDs were verified through the Hub API; this research did **not** download their audio or demonstrate a complete training epoch. Run the pipeline's bounded data preflight before a long job because streaming access, audio decoding, and rate limits still need runtime verification.

Start with **LibriSpeech + AMI** to establish reliable training and quantization behavior. Add **People's Speech `clean`** and **YODAS-Granary English** after the pilot. The latter contributes approximately **102,500 published English hours**; its size makes streaming particularly useful. Published hours are not an audit of unique, usable recordings after our filters and overlap checks.

## Primary sources

| Corpus | Published scale and supervision | Useful coverage | License and access | Stream configuration |
| --- | --- | --- | --- | --- |
| [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) | 960 training hours, aligned audiobook text | Clean baseline and standard regression evaluation | CC-BY-4.0; ungated | `openslr/librispeech_asr`, `all`, three training splits below |
| [AMI](https://huggingface.co/datasets/edinburghcstr/ami) | Approximately 100 hours of meeting recordings overall; training split is smaller | Spontaneous conversation, non-native accents, near/far microphones | CC-BY-4.0; ungated; [publisher license statement](https://groups.inf.ed.ac.uk/ami/corpus/) | `edinburghcstr/ami`, `ihm` or `sdm`, `train` |
| [People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech) | 30,000+ hours across the whole release; `clean` is only a subset | Diverse public recordings with automatically aligned transcripts | `clean` selects CC-BY material; source-dependent BY versions appear in repository metadata. `_sa` configurations separately include ShareAlike material | `MLCommons/peoples_speech`, `clean`, `train` |
| [YODAS-Granary](https://huggingface.co/datasets/espnet/yodas-granary) | Approximately 102.5k English hours and 40.81M utterances; teacher-generated, filtered labels | Large, varied web speech | CC-BY-3.0; ungated | `espnet/yodas-granary`, **`English`**, **`asr_only`** |

The YODAS English hours come from the publisher's [language distribution chart](https://huggingface.co/datasets/espnet/yodas-granary/blob/969944574ea3f37890beaf67ea651e160cfaf043/chart.png), inspected directly. Its labels were produced with Whisper-large-v3 and punctuation/capitalization restoration; they remain pseudo-labels. English storage is reported as 11.3 TB, larger than our second drive. Stream the corpus and keep bounded caches; do not attempt a full local copy.

AMI `ihm` and `sdm` contain different microphone views of the same meetings. They provide acoustic augmentation, not twice as much unique conversational content. Use `ihm` initially, then compare or mix in `sdm` while retaining meeting-level separation.

People's Speech includes several CC-BY versions in its metadata (2.0, 2.5, 3.0, and 4.0). The `license` string in our source configuration is descriptive metadata, not a replacement for source terms or attribution records. The Parquet rows do not expose a per-recording license column. Keep the upstream provenance and applicable attribution records when collecting, modifying, or distributing data. Neither a dataset-card tag nor a model checkpoint establishes that these obligations have been discharged.

## Revisions and schema

| Repository | Pinned revision | Audio / transcript / segment ID | Useful grouping or duration fields |
| --- | --- | --- | --- |
| `openslr/librispeech_asr` | `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` | `audio` / `text` / `id` | `speaker_id`, `chapter_id`; obtain duration from decoded audio |
| `edinburghcstr/ami` | `46f28f2503e2ec48f8867a84eef356c70476beab` | `audio` / `text` / `audio_id` | `meeting_id`, `speaker_id`, `begin_time`, `end_time` |
| `MLCommons/peoples_speech` | `f10597c5d3d3a63f8b6827701297c3afdf178272` | `audio` / `text` / `id` | `duration_ms`; no independent speaker field |
| `espnet/yodas-granary` | `969944574ea3f37890beaf67ea651e160cfaf043` | `audio` / `text` / `utt_id` | `duration` in seconds, `original_audio_id`, `original_audio_offset`, `lang`, `task` |

LibriSpeech `all` exposes `train.clean.100`, `train.clean.360`, `train.other.500`, `validation.clean`, `validation.other`, `test.clean`, and `test.other`. The alternative `clean` configuration uses different split names (`train.100`, `train.360`); do not mix those naming conventions. AMI has `train`, `validation`, and `test`. People's Speech exposes validation/test through several configurations, but those are shared held-out sets, not additional independent evaluation corpora. These definitions are recorded in the respective pinned cards: [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr/blob/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/README.md), [AMI](https://huggingface.co/datasets/edinburghcstr/ami/blob/46f28f2503e2ec48f8867a84eef356c70476beab/README.md), [People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech/blob/f10597c5d3d3a63f8b6827701297c3afdf178272/README.md).

## Runnable source entries

The following JSON contains pipeline source objects. `weight` means relative **example-sampling probability**, not a fraction of unique hours or audio exposure. Source identity is the combination of repository, configuration, split, and revision; repository ID alone is insufficient.

Initial training mixture:

```json
[
  {
    "id": "openslr/librispeech_asr",
    "config": "all",
    "split": "train.clean.100",
    "revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    "text_column": "text",
    "audio_column": "audio",
    "id_column": "id",
    "speaker_column": "speaker_id",
    "weight": 0.15,
    "license": "cc-by-4.0"
  },
  {
    "id": "openslr/librispeech_asr",
    "config": "all",
    "split": "train.clean.360",
    "revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    "text_column": "text",
    "audio_column": "audio",
    "id_column": "id",
    "speaker_column": "speaker_id",
    "weight": 0.35,
    "license": "cc-by-4.0"
  },
  {
    "id": "openslr/librispeech_asr",
    "config": "all",
    "split": "train.other.500",
    "revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    "text_column": "text",
    "audio_column": "audio",
    "id_column": "id",
    "speaker_column": "speaker_id",
    "weight": 0.45,
    "license": "cc-by-4.0"
  },
  {
    "id": "edinburghcstr/ami",
    "config": "ihm",
    "split": "train",
    "revision": "46f28f2503e2ec48f8867a84eef356c70476beab",
    "text_column": "text",
    "audio_column": "audio",
    "id_column": "audio_id",
    "speaker_column": "speaker_id",
    "weight": 0.05,
    "license": "cc-by-4.0"
  }
]
```

For the expanded mixture, retain those four entries with weights **0.04, 0.08, 0.08, 0.05**, respectively, and append:

```json
[
  {
    "id": "MLCommons/peoples_speech",
    "config": "clean",
    "split": "train",
    "revision": "f10597c5d3d3a63f8b6827701297c3afdf178272",
    "text_column": "text",
    "audio_column": "audio",
    "id_column": "id",
    "speaker_column": null,
    "weight": 0.25,
    "license": "CC-BY (source-dependent version)"
  },
  {
    "id": "espnet/yodas-granary",
    "config": "English",
    "split": "asr_only",
    "revision": "969944574ea3f37890beaf67ea651e160cfaf043",
    "text_column": "text",
    "audio_column": "audio",
    "id_column": "utt_id",
    "speaker_column": "original_audio_id",
    "weight": 0.50,
    "license": "cc-by-3.0"
  }
]
```

Here Granary's `speaker_column` intentionally carries a **recording grouping key**, because the release does not supply true speaker IDs. One YouTube recording may contain multiple speakers, and a speaker may appear in several recordings. Do not report this field as an independently verified speaker identity.

Weights are starting experiments. Increase the new source share gradually, track exposure by corpus, and adjust against held-out WER and difficult-condition errors. These values do not establish a compute-optimal mixture for W1A1 speech recognition.

## Evaluation and exclusion

Keep official training, validation, and test splits separate. Use validation for scheduling, collapse detection, and checkpoint selection. Reserve test for infrequent reporting. A deterministic bounded validation panel can make frequent checks cheap; freeze its selected IDs and preprocessing with the run, and use a larger panel periodically.

```json
[
  {
    "id": "openslr/librispeech_asr",
    "config": "all",
    "split": "validation.clean",
    "revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    "text_column": "text",
    "audio_column": "audio",
    "id_column": "id",
    "speaker_column": "speaker_id",
    "weight": 1.0,
    "license": "cc-by-4.0"
  },
  {
    "id": "openslr/librispeech_asr",
    "config": "all",
    "split": "validation.other",
    "revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    "text_column": "text",
    "audio_column": "audio",
    "id_column": "id",
    "speaker_column": "speaker_id",
    "weight": 1.0,
    "license": "cc-by-4.0"
  },
  {
    "id": "edinburghcstr/ami",
    "config": "ihm",
    "split": "validation",
    "revision": "46f28f2503e2ec48f8867a84eef356c70476beab",
    "text_column": "text",
    "audio_column": "audio",
    "id_column": "audio_id",
    "speaker_column": "speaker_id",
    "weight": 1.0,
    "license": "cc-by-4.0"
  }
]
```

Granary has no official English validation/test split. If adding an in-source diagnostic set, partition by a stable hash of `original_audio_id`, and exclude the same recording groups from training **before** shuffle, augmentation, and tokenizer fitting. For new AMI partitions, use `meeting_id` rather than utterance ID; keep all microphone channels from a meeting together. People's Speech's `id` includes source path information, but its format needs an audited extraction rule before it can serve as a recording grouping key.

An official split protects against its publisher's intended within-corpus leakage. It does not prove separation from every other corpus. LibriSpeech, MLS, and Libri-Light draw from LibriVox; web speech collections can share source videos or reuploads. Build a common recording manifest, retain publisher IDs/URLs where available, and audit acoustic fingerprints across training and evaluation. Matching source IDs catches some overlap; it cannot establish global deduplication or unique-hour counts. Never infer that identical transcript text alone means identical audio.

For this application, add a separately recorded held-out panel covering terminology, numbers, units, negation, microphone placement, quiet speech, and operational acoustics. General public corpora do not establish performance on those conditions. Human-check important labels: pseudo-label errors can otherwise become both the training target and a misleading evaluation reference.

## Expansion sources with additional preparation

| Source | Why it is useful | Current limitation |
| --- | --- | --- |
| [MLS English](https://www.openslr.org/94/) | 44,659.74 published training hours; CC-BY-4.0 audiobook speech | The current official `facebook/multilingual_librispeech` Hub repository has seven non-English configurations and **no English configuration**, despite the general card describing English. Official downloads are 2.4 TB FLAC or 651 GB Opus; obtain and prepare local shards under `/mnt/hd` before using HF local-file streaming. |
| [Broader Granary](https://huggingface.co/datasets/nvidia/Granary) | The [paper](https://arxiv.org/html/2505.13404v2) reports approximately 275k English hours across all constituent corpora | `nvidia/Granary` contains manifests, not embedded audio. VoxPopuli, YouTube-Commons, and Libri-Light audio must be obtained separately with their source licenses. Do not count YODAS-Granary twice. |
| [Common Voice](https://commonvoice.mozilla.org/en/datasets) | Crowdsourced accents and recording diversity; inspect the chosen release's license | [Mozilla now distributes Common Voice exclusively through Mozilla Data Collective](https://community.mozilladatacollective.com/faq-can-i-get-the-common-voice-or-other-mdc-datasets-from-other-platforms-like-github-or-hugging-face/). An old `mozilla-foundation/common_voice_*` HF example is not a current official streaming solution. Obtain through MDC, then use local HF streaming. |
| [Unsupervised People's Speech](https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech) | A possible larger pool for future teacher labeling | This is unlabeled audio, not ready supervised CTC training data. Requires language filtering, segmentation, labels, per-source license handling, and overlap checks. |

The missing English configuration was confirmed from [current official MLS metadata](https://huggingface.co/datasets/facebook/multilingual_librispeech/blob/2e83e61823b4c47dcbcb1980bb88601274127609/README.md). It should not be silently replaced with an unverified mirror.

GigaSpeech and SPGISpeech are useful research corpora, but exclude them from the default deployment-oriented mixture: [GigaSpeech's access terms](https://huggingface.co/datasets/speechcolab/gigaspeech#terms-of-access) limit use to noncommercial research/education; [SPGISpeech's terms](https://huggingface.co/datasets/kensho/spgispeech) restrict use to academic research/internal use and impose additional limitations. GigaSpeech's discussion of possible commercial model licensing does not remove its explicit data-access terms. TED-LIUM also has restrictive CC-BY-NC-ND-3.0 terms in the [published loader](https://huggingface.co/datasets/distil-whisper/tedlium/blob/db6c0314dee6cc8e2770b34fda08f4a3bf1ebd59/tedlium.py); the official `LIUM/tedlium` API was inaccessible during this check. These are not interchangeable with the ungated CC-BY training sources above.

## Streaming and accounting requirements

The following are implementation requirements and research recommendations, not claims that source metadata automatically enforces them.

1. Verify `/mnt/hd` is mounted before imports or operations that could initialize caches. Put dataset/model caches, temporary audio, tokenizer assets, checkpoints, and exports under `/mnt/hd/wilderness-labs-stt/`. Streaming training needs network access; deployed recognition remains local.
2. Pass `streaming=True`, an explicit split, a pinned revision, and an explicit cache directory to `load_dataset`. Keep metadata and a modest shuffle buffer in memory. Decode/resample only the selected examples, and bucket a bounded number by duration to reduce padding. Parquet column selection and metadata filters can reduce unnecessary I/O; [HF supports these streaming options](https://huggingface.co/docs/datasets/stream).
3. Current HF audio access returns TorchCodec `AudioDecoder` objects, which require a compatible FFmpeg runtime. Either handle that interface or request `Audio(decode=False)` and decode embedded WAV/FLAC bytes explicitly with a tested audio library. Do not assume every release returns the old `{"array": ..., "sampling_rate": ...}` shape. See [HF audio-loading documentation](https://huggingface.co/docs/datasets/audio_load).
4. Reject nonfinite/empty audio, empty transcripts, invalid sampling metadata, and transcript/audio pairs that cannot align under CTC. Count rejected examples and reasons by source; excessive rejection is a pipeline fault or data-quality signal. Duration filtering must not crop audio while retaining an unchanged full transcript. Preserve digits and meaningful text during normalization; do not silently erase unsupported characters.
5. Track observed valid audio seconds, optimization exposure, examples, source passes, and rejection rates separately. A streaming epoch, repeated small corpus, or augmentation does not create new unique hours. A verified unique-hour total requires stable recording identity and overlap accounting.
6. Make corpus restart and stopping behavior explicit. A mixed stream ending when the smallest corpus finishes can truncate training; oversampling until every source finishes can repeat AMI many times. Use fixed optimization/audio budgets and report per-source exposure rather than relying on an ambiguous global epoch.
7. Checkpoint dataset iterator and sampler states, model/optimizer/scheduler state, quantization stage, RNG state, and source revisions together. HF documents that its normal `.shuffle()` buffers are **lost and refilled on resume**, including when used with `StatefulDataLoader`; save a custom buffer or replay deterministically if exact sample continuation is required. Do not claim exact resume merely because `state_dict()` exists. [HF checkpoint/resume behavior](https://huggingface.co/docs/datasets/stream#save-a-dataset-checkpoint-and-resume-iteration).
8. Treat network retries, decode failures, and source exhaustion as observable events. Use bounded retries and stop on sustained failure; silently spinning through bad data can look like a stalled or collapsed model.

Before broadening beyond roughly 100k unique hours, compare the gains from better filtering, additional human-supervised speech, domain recordings, and more diverse sources. The published Granary experiments show that filtering can preserve or improve recognition quality with fewer hours; that is useful motivation, not a validated data scaling law for our 488M W1A1 model. [Granary training experiments](https://arxiv.org/html/2505.13404v2#S4).
