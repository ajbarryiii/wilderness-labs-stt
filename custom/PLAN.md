# Custom English ternary speech-to-text model plan

Status: implementation in progress. The real-audio VAD experiment is archived; a local pretrained STT reference and a custom ternary CTC training/comparison pipeline are implemented. See `stt/README.md` and its results for current qualification evidence.
Written: 2026-09-09.
Repository: /home/aj/workspace/github.com/wilderness-labs/wilderness-labs-stt

## Implementation update (2026-09-09)

The first working transcription reference is local Whisper tiny.en under `finetune/stt`.
The custom engineering recognizer under `custom/stt` is a 1,556,928-parameter causal
convolutional CTC model with ternary learned weights. It is the convolution-only
pilot/control, not the planned Conformer or the final 0.5–1B recognizer. Its first
short training run overfits the small corpus; decoded development quality remains
poor. A low training loss is not evidence of a usable recognizer.

The paired playback evaluator compares always-on transcription against both
archived VAD candidates. Applying their whole-clip thresholds directly to trailing
one-second windows drops substantial speech. These gates remain experimental;
streaming calibration and speech-preservation validation are required before
using them in the deployment flow below. CPU timing and skipped audio are cost
proxies. No phone power improvement has been established.

Model weights remain local. Git stores code, recipes, results, provenance and
artifact hashes. The original starting-point observations below are historical.

## 1. Objective and inspected starting point

Build an English-only streaming recognizer whose deployed learned weights are at most ternary, selected for the lowest measured energy at an acceptable transcription quality and latency. Its intended role is hands-free access to a local TCCC manual/checklist. Transcription and navigation must work without a network connection. The first model-pipeline stage is a mandatory, always-listening speech/no-speech classifier (voice activity detection, VAD). It gates the heavier transcription frontend and recognizer so they can sleep during non-speech.

The final recognizer target is approximately **0.5–1 billion learned parameters**. The small models below are engineering prototypes, not substitutes for that final capacity target. The always-listening speech classifier remains small and separate; report its parameter count as well as the recognizer count.

The repository's AGENTS.md establishes local operation and power efficiency as central requirements and separates custom model development in custom/ from existing-model work in finetune/. This plan belongs to custom/. It does not establish medical procedures or validate clinical use.

Inspection on 2026-09-09 found:

| Item | Observed state |
| --- | --- |
| Git | main tracking origin/main; clean before this document; HEAD 292cbc9 |
| Repository | AGENTS.md, MIT LICENSE, generic Python .gitignore; custom/ and finetune/ empty |
| OS | NixOS 26.05; Nix 2.34.7 |
| Training GPU | NVIDIA GeForce RTX 5090; 32,607 MiB VRAM; compute capability 12.0 |
| GPU driver / configured power limit | 595.71.05 / 400 W; preserve existing settings |
| CPU | Ryzen 9 9950X3D, 16 cores / 32 threads |
| RAM | Approximately 60 GiB total; no swap |
| Disk | Approximately 666 GiB available on the repository filesystem |
| Tools | nix, git, curl, nvidia-smi available; python, python3, nvcc, uv absent from the inspected SSH PATH |

The GPU is the training platform. The first deployment target is foreground browser inference on CPU using WebAssembly SIMD: Chrome on an ARM64 laptop for development, then Safari on an iPhone 15 Pro for initial latency and power acceptance. The phone model is fixed; the development laptop, tested iOS/Safari version, battery budget, microphone and operating duty cycle remain to be recorded. The remote x86 CPU is a development reference, not the portable power-efficiency acceptance target. WebGPU ASR and native ARM64 execution are later comparison paths.

## 2. Precision contract

Use **scaled ternary weights**, not an ambiguous average-bit-width target:

- Every deployed learned matrix or convolution kernel is represented as W = alpha * Q, with Q in {-1, 0, +1}. This includes every learned layer in the speech/no-speech classifier as well as the ASR feature projection, attention projections, feed-forward layers, depthwise/pointwise convolutions and CTC head. A two-class output does not imply binary weights; classifier weights remain at most ternary.
- Start with one positive scale alpha per tensor. Derive it from the latent weights during training and freeze it at export. Scales are explicit higher-precision quantization metadata, included in size and arithmetic accounting. Do not introduce unreported full-precision residual matrices, embeddings, biases, adapters or normalization parameters.
- Use bias-free layers and normalization without learned affine parameters. Fixed DSP constants and positional functions are not learned weight tensors.
- Target INT8 inputs to learned linear/convolution operators and INT32 accumulation. Initially permit BF16/FP32 activations in the training/reference implementation. Keep normalization, attention softmax, nonlinear functions and residual rescaling at explicitly documented precision until validated lower-precision implementations exist.
- Attention's activation-by-activation products still require arithmetic. Ternary weights do not make the entire model multiplication-free.
- Training uses FP32 latent weights, gradients and optimizer state; quantized forward weights use a straight-through estimator (STE). Training memory is not the packed inference footprint.
- A literal requirement that activations, accumulators and scale metadata also have only three values would be a different research problem. This plan assumes the ternary constraint applies to learned weight codes.

Export initially packs four 2-bit codes per byte, reserving one code as invalid. This costs 2 bits/weight before padding and scales. log2(3) is approximately 1.585, but that is not the initial physical storage cost. Test five-trits-per-byte storage (1.6 bits/weight) only if decoding overhead is worthwhile. Binary weights are a later, stricter ablation, not the initial accuracy baseline.

## 3. What the evidence supports

These sources motivate experiments; their results do not establish this model's quality or energy savings:

- BitNet b1.58 demonstrates ternary weight training with higher-precision activations in language models. Its results cannot be transferred directly to small acoustic models. [BitNet b1.58](https://arxiv.org/abs/2402.17764)
- The directly relevant ASR study uses a ternary codebook for its "2-bit" case, but defaults to a 4-bit decoder and retains higher-precision convolutions in its strongest quality-preserving configurations. Its all-2-bit convolution experiment shows degradation. Strict ternary coverage, especially the acoustic layers and output head, therefore remains a central research risk. [Towards One-bit ASR, sections 4.1–4.4](https://arxiv.org/html/2505.21245v1)
- A newer ASR report also uses a mixed design: INT8 acoustic tokenizer plus a ternary autoregressive model. It is useful context, but does not satisfy this plan's strict learned-weight contract. [VibeVoice-ASR-BitNet](https://arxiv.org/abs/2607.21075)
- Conformer and QuartzNet provide useful acoustic architecture families. The causal, small, fully ternary adaptations below are our proposed experiments. [Conformer](https://arxiv.org/abs/2005.08100), [QuartzNet](https://arxiv.org/abs/1910.10261)
- T-MAC and bitnet.cpp provide low-bit kernel ideas and implementations for language-model workloads. Their support for our shapes, convolutions, caching and target device must be established independently. [T-MAC](https://github.com/microsoft/T-MAC), [BitNet runtime](https://github.com/microsoft/BitNet)

The first decision is whether a strict ternary acoustic model can meet quality requirements with a runtime that actually benefits from its representation. Do not spend the full training budget before investigating both.

## 4. Pipeline: speech/no-speech classifier, then streaming CTC

### Mandatory first stage: speech/no-speech classification

Deployment flow: **microphone/PCM capture -> minimal classifier DSP -> speech/no-speech classifier -> buffered speech -> ASR log-mel frontend -> causal encoder -> CTC decoder/endpointing**. Capture and the minimal features required by the classifier precede its decision; the full ASR frontend and encoder do not run continuously to supply classifier inputs.

The classifier runs locally at low cost throughout listening. Start with a compact causal temporal-convolution classifier, targeting at most 50,000 learned weights, ternary weights and INT8 operator inputs under the same precision contract as ASR. Proposed initial inputs are 16 coarse spectral bands from trailing 20 ms windows at a 10 ms hop, with bounded past context of about 200 ms and no future audio. These are profiling candidates, not established optimal settings. Count its feature extraction and state overhead when choosing between this feature-based design and simpler alternatives. Fixed DSP is allowed; no high-precision pretrained VAD is required at deployment.

Use a state machine rather than independently dropping every frame labeled non-speech:

- **Listening:** keep capture, classifier and a bounded raw-PCM ring buffer active; suspend the ASR frontend, encoder and decoder. Start with 300 ms of pre-roll. Avoid polling/waking the heavy runtime while idle.
- **Speech onset:** use a development-calibrated speech threshold with short persistence (initially 20–30 ms). Wake ASR and replay pre-roll plus queued audio in chronological order, once, preserving sample timestamps. Size the buffer to exceed measured onset-detection delay plus runtime wake latency and frontend history requirements; 300 ms is only a starting value.
- **Active speech:** feed contiguous audio, including brief pauses, to ASR. Use a lower continuation threshold than the onset threshold to reduce rapid on/off switching. Keep classifier and ASR state across short gaps; uncertain segments should favor retaining possible speech.
- **Speech end:** start with 200 ms of non-speech hangover, flush the segment and finalize decoding, then reset ASR state at the segment boundary and return to listening. Tune hangover jointly with endpointing so their combined delay meets the existing finalization budget; do not stack two independent long endpoint timers. Continuous speech keeps ASR active.

This is speech detection, not keyword or speaker identification: background/overlapping speech is still speech, and quiet speech and unvoiced consonants must remain eligible for transcription. Evaluate any later intended-speaker filtering separately.

Implement the classifier and gating state machine before full-pipeline training/evaluation. ASR-only execution remains a diagnostic ablation to quantify gating losses and savings; the planned deployment pipeline always starts with classification. Train the recognizer on complete labeled speech segments and evaluate the assembled pipeline on continuous recordings, so gate mistakes cannot disappear through speech-only test selection.

### ASR audio and text

- Input: 16 kHz mono PCM; fixed log-mel frontend with 80 bands, 25 ms trailing windows and 10 ms hop. Use causal framing without centered STFT padding or utterance-wide normalization. Compute any dataset normalization statistics from training data only.
- Initial subsampling: stack the latest four feature frames at a stride of four, then use a ternary 320-to-d projection. This avoids requiring a learned high-precision feature extractor and produces 25 encoder frames/second. Retain a 2x stacking/stride option for fast speech.
- Output baseline: 29 CTC symbols: a–z, apostrophe, space and blank. Train on a consistent spoken-form English transcript convention. Preserve distinctions involving numbers, negation, units and acronyms in both annotation and scoring; do not hide them behind aggressive normalization.
- Greedy CTC first: no autoregressive decoder, external language model or cloud service. The original CTC formulation motivates this alignment-free sequence objective. [CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- Assert CTC path feasibility for every example: output frames >= target symbols + adjacent identical-symbol pairs. Diagnose failures and change stride/tokenization; never silently zero infeasible losses or discard fast/long speech.
- Later compare a 256/512-piece English tokenizer trained only on training transcripts, with complete character fallback. Explore 8x subsampling only after measuring rare-term/acronym accuracy and alignment feasibility.

### Encoder configurations

Start with T7 for debugging and the first browser CPU deployment. T15 and T29 are optional intermediate prototypes for learning and kernel experiments. The final recognizer targets 0.5–1B parameters; passing a small-model iPhone test does not establish final-model feasibility.

| Candidate | Blocks / width / heads | FFN expansion | Approximate learned weights | Raw 2-bit weight payload |
| --- | --- | --- | --- | --- |
| T7 pilot | 8 / 192 / 4 | 4 | 6.9 million | 1.7 MB |
| T15 intermediate prototype | 10 / 256 / 4 | 4 | 15.2 million | 3.8 MB |
| T29 intermediate prototype | 12 / 320 / 4 | 4 | 28.5 million | 7.1 MB |

These are design estimates in decimal MB, not measured artifacts. They assume two feed-forward modules per block, four attention projections, a GLU convolution module, a kernel-15 depthwise convolution, and the small input/output projections. The implementation must calculate exact counts, padding, scales and runtime memory.

Each block uses two half-step residual FFNs, local causal attention, and a causal convolution module. Use fixed positional functions, bias-free projections, non-affine normalization and no batch statistics spanning future frames. All learned kernels remain ternary. Start attention with 64 previous encoder frames (2.56 seconds at 4x stride), plus the current frame; no acoustic right context. Convolution uses a bounded history buffer.

Process 80 ms chunks initially; compare 40/80/160 ms for energy versus token delay. Chunk batching adds scheduling latency even with zero right context. Maintain projection caches and convolution state rather than recomputing history. Audit the effective receptive field across all blocks: a per-layer 64-frame window does not imply a 2.56-second total receptive field.

The T15 INT8 K/V cache alone is roughly 2 * 10 * 64 * 256 bytes, or 0.33 MB; float caches, convolution history, scratch buffers and frontend state add to this. Avoid allocations that grow with recording duration. A chunked forward pass should match a full-sequence forward pass using the same causal/window masks.

### Final capacity target and scaling gates

| Recognizer scale | Role | Raw packed 2-bit weight payload | FP32 weights + gradients + two Adam moments at 16 bytes/parameter |
| --- | --- | --- | --- |
| About 0.5B parameters | Lower end of final target | 125 MB (about 119 MiB) | 8 GB (about 7.45 GiB) |
| About 1B parameters | Upper end of final target | 250 MB (about 238 MiB) | 16 GB (about 14.9 GiB) |

These arithmetic budgets exclude scales, padding, classifier weights, activations, caches, scratch buffers and framework overhead. A 1.6-bit packing experiment would reduce raw payloads to about 100/200 MB, but is not the default runtime format. Neither packed size nor parameter-state memory establishes real-time inference or training feasibility.

Choose depth, width and temporal operator mix after profiling representative final-scale blocks; do not simply widen the pilot without measuring it. Keep the strict ternary contract, English-only output and classifier-first pipeline at all scales. Record total and, if conditional computation is explored, active parameters separately; sparse execution is an experiment, not an assumed saving. The initial dense design should be measured explicitly.

Use small pilots to validate correctness and QAT, then a bounded intermediate scale if needed, before allocating a full 0.5B run. Profile both 0.5B and 1B shapes and short training runs to compare peak memory and throughput. Advance toward 1B when its quality/energy tradeoff and training budget justify it; do not interpret 1B as a mandatory minimum.

On iPhone 15 Pro, test final-sized packed artifact loading, peak memory, worker responsiveness and representative/full-stack execution early, using synthetic weights if trained weights are not yet available. Synthetic-weight runs establish capacity/runtime feasibility only, not accuracy. Measure cold start, buffer backlog and sustained active-speech latency as well as silence power. The speech gate saves work during non-speech but does not reduce the per-frame cost while speech is active.

CPU WASM SIMD remains the first runtime baseline. If 0.5–1B execution misses its budget, measure operator/frame-rate changes and optional WebGPU/native paths rather than assuming the tiny pilot's results transfer. Report an unmet browser target explicitly; do not silently shrink the final model below the requested range or relax weight precision to claim success.

### Bounded architecture control

Train one comparable causal time-channel separable convolution CTC model with the same frontend, text convention and precision contract. This tests whether attention/softmax/cache overhead is worth its accuracy gain.

If ternary depthwise kernels are a bottleneck, test fixed temporal filters plus ternary channel mixing, or adjusted width/receptive field. If necessary, increase ternary capacity. A mixed-precision model can diagnose sensitivity, but cannot be relabeled as meeting the strict ternary requirement.

Defer transducers, large decoder language models and multi-rate encoders until the simple candidates expose a measured limitation. Fast Conformer and Zipformer are later references for frame-rate and encoder-efficiency experiments. [Fast Conformer](https://arxiv.org/abs/2305.05084), [Zipformer](https://arxiv.org/abs/2310.11230)

## 5. Training on the single 5090

### Reproducible environment first

Create a root flake.nix/flake.lock and a custom Python package with pinned dependencies. Pin Emscripten and the browser harness JavaScript toolchain/dependency lockfile as well; record tested development Chrome/OS and target iOS/Safari versions. Add a WASM SIMD compile/load smoke test alongside GPU preflight. Keep training inside a Nix development environment; do not depend on packages in an interactive user shell. Prefer a compatible CUDA-enabled Nixpkgs PyTorch package. If using upstream wheels, explicitly package the required dynamic libraries and host driver access through the Nix environment.

Verify the chosen PyTorch/CUDA combination supports sm_120 and the installed driver. PyTorch introduced Blackwell support with its CUDA 12.8 builds; that is historical compatibility evidence, not an instruction to pin an old release. Select and lock a supported combination from the current official matrix, then run it on this machine. [PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/), [current support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md), [Nixpkgs CUDA documentation](https://nixos.org/manual/nixpkgs/stable/#cuda)

Preflight must execute CUDA matrix multiplication, convolution, attention, forward/backward through the ternary STE and FP32 CTC loss; verify finite gradients and checkpoint resume. Record actual library/driver versions and GPU properties. Use one process on one GPU. Add torch.compile or custom CUDA kernels only after eager execution works. Leave the observed 400 W limit unchanged during setup.

### Classifier training

Train the speech/no-speech classifier first on frame/segment speech-activity labels using binary cross-entropy with class weighting or sampling tuned on development data. Include real speech boundaries, pauses, quiet speech, short utterances and unvoiced sounds alongside silence, wind, breathing, clothing rub and machinery. Speech mixed with noise remains a positive example. Do not label every frame of a transcript-bearing recording as speech: use reviewed activity annotations, recording-aligned labels or checked alignments, and document uncertainty near boundaries. Preserve disjoint speakers, sessions and noise identities.

Train with ternary forward weights, then validate the deployed activation precision. Calibrate thresholds, persistence and hangover on development recordings at realistic speech duty cycles. Prefer low missed-speech rates subject to a measured energy budget over high frame accuracy on imbalanced silence-heavy data. Export and audit the classifier independently before integrating it with ASR.

### Quantization and optimization

1. Build the identical architecture in BF16 as a diagnostic quality baseline and training-only teacher. This baseline is not a deployment candidate under the weight constraint.
2. Train the custom student with ternary forward weights from initialization. Initial quantizer: alpha = max(mean(abs(W)), epsilon), Q = clamp(round(W / alpha), -1, 1), effective weight alpha * Q. Define and test rounding, all-zero tensors, STE gradient behavior and scale handling explicitly.
3. Start with weight quantization and BF16 activations. After alignment begins to learn, add symmetric INT8 activation fake quantization, calibrating only on training/development data. Compare gradual activation quantization against training with it from the start. This does not relax the weight constraint.
4. Proposed optimizer starting point: AdamW, peak learning rate 3e-4, weight decay 0.01, 5% warmup then cosine decay, gradient norm clipping at 1.0, dropout 0.1. Treat these as pilot settings; use a small 1e-4/3e-4/1e-3 learning-rate screen only if learning is unstable.
5. Track CTC blank rate, gradient norms, weight zero fraction per layer, quantizer saturation, scale drift, activation clipping and each development split's WER. Do not optimize sparsity before a kernel can exploit it.
6. If quality trails, test same-architecture distillation from a frozen teacher using aligned CTC posterior KL plus ground-truth CTC. Weight blank and nonblank contributions deliberately. Logits must share tokenization, frame rate and masks; do not align unrelated teacher/student frames by index. Generate targets in a separate pass or run the teacher sequentially to bound memory.
7. Optional existing teachers can provide training targets only after their licenses and provenance are recorded. Validate uncertain labels with ground truth. No teacher is needed at inference. Learned-scale or stochastic-precision experiments are secondary and must preserve explicit exported metadata and strict weight coverage.

### Memory, throughput and stop rules

- For the small pilots, begin with 5–15 second utterances in duration buckets and 60–120 audio seconds per microbatch, reducing until measured peak reserved VRAM stays below approximately 28 GiB. Accumulate toward an effective batch of about 240 audio seconds. At 0.5–1B, start memory profiling with one short utterance per microbatch, then increase duration/batch only from measured headroom; do not reuse the pilot batch budget. Profile padding and then include longer utterances.
- Compute CTC log probabilities/loss in FP32. Add activation checkpointing only when useful. Mixed precision and gradient accumulation do not imply 2-bit optimizer state.
- Budget roughly 16 bytes/parameter for FP32 weights, gradients and two Adam moments before temporary copies: about 240 MB for a 15-million-weight prototype, 8 GB at 0.5B and 16 GB at 1B. Activations, quantized/temporary weight copies, attention, CTC workspace and allocator overhead remain additional. Profile activation checkpointing, small microbatches and gradient accumulation on the single 5090; consider optimizer/offload changes only if required, recording their host-RAM and throughput costs. Keep a large teacher out of the simultaneous GPU working set. A separate full-size BF16 reference training run requires its own measured budget; small-scale comparison results do not prove final-scale quality parity.
- Save config, seed, dataset/normalizer hashes, model, optimizer, scheduler, random states and sampler position. Keep best and a bounded number of recent resumable checkpoints; store runs/data outside tracked source directories. FP32 weights plus two Adam moments alone occupy roughly 6–12 GB per 0.5–1B training checkpoint; budget retention and temporary save space independently of the 125–250 MB deployment artifact.
- After warmup, profile 200–500 steps across representative duration buckets. Record audio seconds processed per wall second R, peak allocated/reserved VRAM, host RAM, dataloader time and GPU utilization.
- Estimate training hours as dataset_hours * passes / R, then add measured validation/checkpoint overhead. Use this to set a run budget; do not predict training completion from GPU model alone.
- First gate: overfit a tiny 16–64 utterance set. Next use train-clean-100, initially budgeting ten passes and reassessing the learning curve rather than launching an unbounded sweep. Set a minimum warmup period, maximum steps and a development-based early-stop rule before each run.
- Repeat promising pilot comparisons with at least three seeds before attributing a small quality difference to quantization. Expand only surviving configurations.

## 6. Data curriculum and provenance

Keep raw datasets and derived caches outside Git and track versioned manifests containing source URL/version, license/access terms, checksums, sample rate, duration, transcript, split and speaker/session identity. The repository's MIT license does not replace dataset or pretrained-model licenses.

| Stage | Data | Purpose / conditions |
| --- | --- | --- |
| VAD pilot | Labeled continuous speech/non-speech recordings, including quiet speech and hard noise negatives | Classifier training, onset/offset labels and gating-state evaluation |
| Smoke test | 16–64 real English utterances | Alignment, quantizer, loss and exact overfit |
| Pilot | LibriSpeech train-clean-100 | Reproducible architecture/QAT comparisons |
| General-English foundation | LibriSpeech 960-hour training split | Broader acoustic learning after pilot gates |
| Public medical adaptation | PriMock57; optional cleaned Medical Speech, Transcription, and Intent | Human medical audio after provenance, split and quality audit; details below |
| Independent medical evaluation | Eka Medical ASR Evaluation, English subset | Freeze as held-out evaluation; exclude from training and tuning |
| Conditional synthetic augmentation | IntelMedica Medical TTS v2 | Listed for consideration; excluded from default training pending access/license compatibility |
| Domain coverage | Proposed initial 10–20 hours from 20–50 consenting speakers | Checklist navigation, terminology, corrections, abbreviations, numbers, units and distractors |
| Optional expansion | Bounded Common Voice English or MLS English subsets | Accents/devices/style diversity after provenance, overlap and storage review |

LibriSpeech is read English under CC BY 4.0. Retain official development and test roles; it is a baseline, not evidence of field robustness. [LibriSpeech](https://www.openslr.org/12/)

MUSAN provides augmentation audio under CC BY 4.0, and OpenSLR's room-response/noise database is Apache 2.0. Preserve attributions and keep noise recordings/RIR identities disjoint across training and evaluation. [MUSAN](https://www.openslr.org/17/), [RIR/noises](https://www.openslr.org/28/)

Common Voice releases have specific access/use conditions in addition to their license labels; pin and review the selected release before use. MLS English is large: the listed FLAC and Opus packages exceed a sensible full-download budget on this host. Use a bounded subset if needed and check LibriVox/book/speaker/audio overlap with LibriSpeech. [Common Voice English release](https://mozilladatacollective.com/datasets/cmqim2hn800ssnr07gvmpcnwu), [MLS](https://www.openslr.org/94/)

The 960-hour foundation and small public medical corpora remain initial data stages, not a claim that they are sufficient to train 0.5–1B parameters to the required quality. Before a full-scale run, inspect learning curves, overfitting and medical/general-English tradeoffs; budget permitted additional English audio or teacher supervision if needed. Retain all held-out boundaries and estimate GPU-hours from final-scale throughput.

### Public medical audio datasets

The following four datasets supplement general-English training and the planned TCCC recordings. Descriptions and license labels were inspected on 2026-09-09; recheck the exact release and accompanying terms when acquiring data. Public availability does not imply identical training or redistribution rights. This plan does not download datasets or accept gated access terms.

| Dataset | Available material | Planned role | Access/license and quality conditions |
| --- | --- | --- | --- |
| [PriMock57](https://github.com/babylonhealth/primock57) | 57 mock English primary-care consultations, clinician/simulated-patient audio and manual utterance-level transcripts | First public medical adaptation candidate; retain separate development/test consultations | [CC BY 4.0](https://github.com/babylonhealth/primock57/blob/main/LICENSE.md); preserve attribution. Audio uses Git LFS. Use spoken transcripts, not consultation-note summaries, as ASR targets. |
| [Eka Medical ASR Evaluation](https://huggingface.co/datasets/ekacare/eka-medical-asr-evaluation-dataset) | About 3,600 English recordings within a roughly 3,900-record English/Hindi collection; terms, narrated sentences, conversations and medical-entity annotations | Independent English terminology and accent benchmark | Dataset card specifies MIT. Select English explicitly. Indian accents and branded-drug vocabulary are useful coverage, but do not substitute for the intended TCCC vocabulary and recording conditions. |
| [Medical Speech, Transcription, and Intent](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent) | About 8.5 hours of human symptom-description recordings with transcripts | Optional supplementary adaptation after cleaning and license verification | Publisher lists the license as Other and flags incorrect labels and poor audio. Resolve original Figure Eight terms before training; do not infer permission from a mirror's license label. |
| [IntelMedica Medical TTS v2](https://huggingface.co/datasets/intelmedica/medical-tts-parquet-2-16khz) | 101,475 synthetic audio/text pairs, about 184 hours, 16 kHz mono and 19 English voices; approximately 92% drug-related | Conditional training-only vocabulary augmentation; not a real-speech acceptance benchmark | Access-gated and labeled CC BY-NC 4.0, with component terms. Exclude from the default training recipe unless intended use and release are compatible or appropriate permission is obtained. Include source-term provenance and pronunciation checks if used. |

Audit PriMock57 first: verify actual audio downloads rather than LFS pointer files, channel/utterance alignment, timestamps, duration and transcript consistency. Partition by consultation and speaker before extracting clips; account for clinicians recurring across consultations so the same voices do not leak between claimed speaker-independent splits. Document any remaining overlap rather than claiming independence. Reserve development/test material before fitting normalizers or selecting vocabulary.

Keep Eka's English evaluation audio, transcripts and medical-entity annotations out of model training, teacher-target generation, tokenizer/lexicon construction, contextual biasing and confidence/threshold tuning. Freeze the selected release and evaluate after the recipe is fixed; use separate medical development recordings for iteration. Report general WER plus entity/term errors, numbers, units and negation errors. Never pool medical benchmark scores into a single result that hides dataset-specific weaknesses.

For the symptom dataset, validate audio/transcript joins, clipping and mislabeled examples; record exclusions and corrections. Maintain valid existing held-out boundaries, audit speaker/text duplicates across all datasets, and record any new grouping needed to prevent leakage. Add it to training only after the license issue is resolved.

For any permitted synthetic data, audit difficult pronunciations and transcript/audio agreement before inclusion. Use a separately tracked, bounded training mixture and compare with a real-speech-only adaptation control. Tune mixture proportions on development data; do not let the large drug-heavy synthetic set overwhelm real speech or TCCC coverage. Report synthetic and human data hours separately. Synthetic samples and derivative prompts must not enter held-out real-speech acceptance sets.

The public medical datasets chiefly support ASR. Retain continuous speech/non-speech recordings and hard noise negatives for the mandatory classifier; transcript-bearing clips are not automatically all-speech frame labels. PriMock57 can support boundary work only after validating activity annotations. Continue collecting the proposed TCCC-specific recordings through representative microphones for deployment evaluation.

### Storage and real-domain collection

Reserve at least 150 GiB free for Nix builds, checkpoints and temporary files. Before downloading, estimate compressed data, extraction and caches together. Initially cap combined datasets and caches near 250 GiB; avoid duplicating all audio as expanded WAV and float features. Recheck free disk during runs.

Augment with speed/gain variation, SpecAugment, reverberation and noise; retain clean examples. Proposed noise conditions are clean and 20/10/5/0/-5 dB SNR. Add clipping, bandwidth limitation and dropouts where they match the intended microphone/radio path. Test actual wind, breathing, clothing rub, vehicles, overlapping speakers and protective equipment recordings as available.

Build real domain development/test material early, separate by speaker, session and device, and hold out wording templates where possible. Have a domain reviewer adjudicate transcript distinctions and important terminology. Synthetic voices can supplement training coverage but must not define the final real-speech test.

Exclude test audio, transcripts and their derivatives from training, vocabulary design, distillation and augmentation. Audit near-duplicate utterances across corpus boundaries. Test data is not a tuning set.

## 7. Evaluation and selection rules

### Quality and streaming

Report general-English WER and CER, with substitutions/deletions/insertions, on dev-clean/dev-other during tuning and frozen test-clean/test-other at the final decision. Use a versioned normalization contract and retain raw transcripts. Standard scoring can use NIST SCTK/SCLITE. [NIST speech tools](https://www.nist.gov/itl/tted/mltg/tools)

Domain evaluation reports PriMock57 held-out results, the frozen Eka English benchmark and the separately collected TCCC test set individually. Track which permitted public medical datasets contributed training hours and retain a general-English-only adaptation control. Domain evaluation additionally reports:

- Critical-term recall and false insertions.
- Negation deletions/substitutions and exact numeric-value/unit accuracy.
- Acronym/abbreviation accuracy, utterance exact match and navigation-command confusions.
- False transcript events/hour for silence, noise and unrelated speech.
- Accuracy versus abstention/coverage; calibrate thresholds on development data.
- Results by accent, microphone, noise level and speaking condition, including sample counts.

CTC confidence is not automatically calibrated correctness. Keep uncertainty visible to the manual/navigation layer. Test whether vocabulary biasing substitutes expected terms into unrelated speech. Any decoder biasing uses local data and is evaluated separately from raw greedy CTC.

Run real-time microphone/replay tests as well as accelerated batch-one inference. Measure first stable token delay, word emission delay, end-of-speech-to-final latency, transcript revision rate and endpoint truncation, with p50/p95. Include pauses, long continuous speech, chunk boundaries, resets and extended silence. Distinguish lookahead, chunk buffering, compute and endpointing delay. These are separate from throughput. [On-device user-perceived latency study](https://www.isca-archive.org/interspeech_2021/shangguan21_interspeech.html)

### Classifier and gating evaluation

Report frame precision/recall, speech false-negative rate, entirely missed utterances, false activations/hour, onset delay, onset/tail clipping duration and ASR-active fraction. Stratify by quiet speech, SNR, short words, unvoiced onsets, accents and microphone; unrelated speech is a speech-positive case, not a VAD false alarm. Also report continuous-recording WER and domain metrics against the same ASR run ungated, including fully missed utterances as deletions. Measure buffer overflow, duplicated/dropped samples, wake latency and state continuity in onset/pause/offset tests.

Use identical downstream settings and precision when isolating the classifier's energy effect. Record its always-on power, false-activation cost and pre-roll/hangover processing overhead. Reject a threshold setting that saves energy by silently dropping difficult speech. Fix acceptable missed-utterance and clipping limits before final evaluation; the existing ASR quality margin does not automatically authorize extra gating loss.

### Proposed research gates

These are initial experiment-selection criteria, not field acceptance guarantees:

| Gate | Proposed criterion |
| --- | --- |
| Correctness | Tiny-set overfit; finite gradients; no CTC-invalid examples hidden; chunk/cache correctness |
| Precision | Export audit covers every learned weight in both classifier and ASR and reports all scale/activation/operator precision |
| Speech gate | Classifier and buffered state machine run first; no dropped/duplicated samples in state tests; measure missed utterances, clipping and continuous-recording WER versus ungated ASR against pre-agreed limits |
| Pilot quality | Development WER within max(0.5 percentage points, 5% relative) of the same-architecture BF16 baseline on each general-English split; inspect domain regressions separately |
| Streaming | Zero intended acoustic right context; preliminary targets: p95 chunk compute below half the chunk interval and p95 finalization within 500 ms of annotated speech end on the selected device |
| Runtime | Native scalar versus WASM scalar/SIMD integer parity and documented floating-operation tolerances; equivalent decoded outputs on a fixed golden set |
| Browser first pass | Foreground Safari on the iPhone 15 Pro runs classifier plus T7 ASR on CPU in real time, without growing audio backlog, with the streaming targets above; offline reload works after explicit local asset preparation |
| Energy | Seek at least 20% lower total joules/audio-second than a quality-matched optimized INT8 control, with matched latency and reported uncertainty |
| Final capacity | Recognizer is approximately 0.5–1B learned parameters; report actual count and all runtime memory, rather than equating small-prototype success with final acceptance |
| Final selection | Within the target capacity range, meet domain quality/latency criteria fixed before final tests and choose the measured energy/quality Pareto frontier; report unmet constraints |

Use paired speaker/session-level bootstrap confidence intervals and multiple seeds for close comparisons. A nonsignificant difference is not proof of equivalence. The pilot quality margin only decides whether to keep researching; the requirement to preserve quality still needs final agreed noninferiority bounds and domain acceptance criteria.

If T7 fails, identify alignment/quantization/architecture causes before adding data. If strict ternary quality fails, try a better ternary architecture, capacity or training recipe. If packed kernels fail to save energy, improve the runtime or operator mix before scaling. If all candidates miss the final quality floor, report the unmet constraint rather than quietly retaining higher-precision weights.

## 8. Packed runtime and whole-pipeline energy

### First deployment target: browser CPU

| Target | Initial role | Acceptance scope |
| --- | --- | --- |
| Chrome on ARM64 laptop | Development, correctness and browser profiling | Use available hardware; record exact CPU, OS and browser versions |
| Safari on iPhone 15 Pro | Primary first-pass latency and energy target | Foreground, screen-on, real-time classifier plus T7 on CPU; record and freeze the tested iOS/Safari version and device settings for each comparison |
| WASM scalar fallback | Compatibility and numerical reference | Feature-detect SIMD; fallback performance is reported separately and is not presumed to pass real-time gates |
| WebGPU recognizer | Optional later acceleration comparison | Classifier stays on CPU; retain only if full-pipeline quality/latency/energy justify it |
| Native ARM64 runtime | Later browser-overhead comparison | Same model, audio, gate and comparable settings on the same device where practical |

Compile a C++ runtime to wasm32 with Emscripten SIMD, using portable scalar and WASM SIMD kernels rather than assuming native NEON/AVX implementations carry over unchanged. Begin with single-threaded inference off the UI thread; this still allows independent audio capture and inference workers. Benchmark additional ASR threads only when latency requires them and account for their energy cost. SIMD and shared-memory threading are distinct capabilities. Threaded builds/SharedArrayBuffer require cross-origin isolation with appropriate COOP/COEP headers; feature-detect and retain a non-shared-buffer path. [Emscripten SIMD](https://emscripten.org/docs/porting/simd.html), [Emscripten threads](https://emscripten.org/docs/porting/pthreads.html)

Keep the speech classifier, its minimal DSP and gating first in every deployment path. Use this browser division of work:

- AudioWorklet handles short, bounded capture/buffering work; do not run full ASR in the audio callback. Obtain microphone access through getUserMedia in a secure context after user permission/start interaction. Inspect the actual input sample rate and channel configuration, then perform stateful resampling to 16 kHz; do not assume a requested rate was honored. Record browser echo-cancellation/noise-suppression/gain settings for reproducibility. [AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet), [getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
- An inference worker initially owns the single-threaded WASM classifier, pre-roll state and gated ASR. Run bounded ASR work slices and service incoming classifier frames between them; large ASR calls must not create an unbounded classifier backlog. If profiling shows interference, move classification to its own worker and include the extra scheduling/transfers in power measurements.
- Transfer timestamped audio through a bounded, reusable buffer pool initially; optionally use a shared ring buffer when cross-origin isolation is enabled. Test overflow/backpressure and continuity. Keep the UI thread responsible for controls and transcript display.
- During non-speech, stop expensive ASR calls but keep the packed model resident initially, avoiding reload on every onset. Keep capture/classification alive throughout the foreground listening session; do not suspend the AudioContext that supplies their audio. Report retained memory and wake latency separately.

Package the app shell, WASM module, classifier and ASR weights locally, with versioned hashes and an explicit offline-ready check. For the browser prototype, provision/cache these assets ahead of offline use and verify reload with networking disabled; never rely on a CDN, cloud inference or the browser's built-in speech-recognition service at runtime. Missing/evicted assets must be detected and reported, rather than silently using a network recognizer. Model caching is preparation, not a promise that the browser will retain assets indefinitely.

First-pass support is explicitly a foreground Safari tab on the iPhone 15 Pro with the screen on. Run WASM SIMD loading, microphone/AudioWorklet capture, worker scheduling and offline reload preflight on that phone early; successful development-laptop execution alone does not meet acceptance. Test tab hiding, lock-screen transitions, microphone interruptions and resume; detect discontinuities and reset state safely. Background/locked-screen continuous operation is a separate later requirement, not assumed by this browser milestone.

ONNX Runtime Web may provide a reference implementation for correctness and ordinary CPU/GPU controls. A model whose tensors happen to contain ternary values is not evidence of packed ternary execution: the selected runtime must explicitly consume the packed representation. [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)

WebGPU is an optional ASR experiment after the CPU baseline. Detect adapter/features at runtime and record browser/device support. Keep VAD on WASM CPU and include GPU dispatch, transfers, initialization and idle overhead in comparisons. A GPU backend with expanded weights is a clearly labeled control; compliant packed GPU execution requires an explicit packed-weight kernel. Browser support does not by itself establish operator support or an energy advantage. [WebGPU availability](https://web.dev/blog/webgpu-supported-major-browsers)

### Final-scale browser memory

For the 0.5–1B target, budget 125–250 MB of packed weight payload plus download/cache representation, WASM linear memory, activation/caching/scratch storage and any GPU buffers. Audit duplicate copies during fetch, decoding and worker initialization; use bounded loading and reusable buffers rather than assuming payload size equals peak memory. Do not expand the full model into floats. Measure actual Safari behavior, cold-start time and offline asset retention on the selected iOS version; device RAM alone does not establish the browser's usable budget.

### Packed kernel experiments

Build a scalar packed reference plus WASM SIMD CPU kernels, keeping native ARM/x86 interfaces portable for later comparison. Prototype T-MAC-style lookup and sign/zero add-subtract kernels against an optimized INT8 baseline using the actual small streaming shapes:

- Input projection: 320-to-192/256.
- FFNs: 192/256-to-768/1024 and back.
- Attention and convolution pointwise projections, kernel-15 depthwise operations and 29-symbol CTC head.
- Batch one with 1/2/4 encoder frames per call; realistic cache residency and thread counts.
- Repeat the shape suite for proposed 0.5B/1B widths and depths before full-scale training; include classifier scheduling delay and full-stack memory traffic, not only isolated small-pilot operators.

Measure packing/load cost, lookup-table construction, input quantization, scale application and transfers, including scalar tail shapes. Do not decompress the entire model to floats at load time and call its execution a packed ternary runtime. Check INT32 accumulator bounds for each reduction and keep reserved packed codes invalid. Zero weights are not automatically free without a kernel/layout that benefits from them.

Separate training/reference execution on the 5090, native/WASM operator development, and browser end-to-end measurements on the iPhone 15 Pro Safari target. The remote CPU and development laptop establish development comparisons only. Compare WASM CPU versus optional WebGPU and later native ARM64 on the same portable hardware; the first acceptance target remains CPU browser execution.

Benchmark the complete PCM/microphone -> minimal classifier DSP -> mandatory speech/no-speech classifier -> buffered ASR frontend -> encoder -> decoder/endpointing path. Implement classification and gating from the beginning. During non-speech only capture, classifier features/inference and the ring buffer remain active; verify that expensive ASR execution is actually suspended. Compare against ungated ASR only as a controlled benchmark. For ternary-versus-INT8 recognizer comparisons, hold the classifier and its decisions fixed; report classifier precision comparisons separately.

Use the measured accounting model P_average = P_capture_and_idle + P_classifier_DSP_and_inference + f_ASR_active * delta_P_ASR + wake_events_per_second * E_wake, with components defined to avoid double counting. Active fraction includes false activations, buffered replay and hangover; it is not just the true speech fraction. Measure the break-even point across duty cycles, including continuous speech where gating may add overhead. Optimize whole-system energy subject to speech preservation rather than assuming the classifier always saves power.

On the iPhone 15 Pro, use a battery-rail or whole-system supply meter where available. Keep screen brightness, browser/UI activity, radio state, iOS Low Power Mode, battery state of charge, charging state and thermal conditions consistent; record iOS/Safari and runtime versions. If measuring at USB input, document charging and battery contribution; input power alone must not be equated with device consumption without accounting for them. Do not infer joules from browser timing or battery-percentage changes alone. If direct energy measurement is unavailable, report latency and other proxies as such and leave the energy gate unverified. Measure sustained foreground sessions (initially at least 15 minutes) to expose backlog and thermal effects. Report:

- Total joules per audio second, and joules/query including wakeup and tail.
- Listening-idle and active average watts; peak memory and serialized/runtime sizes.
- Representative 1%, 10% and 50% speech-duty scenarios plus continuous speech.
- Total energy and separately labeled idle-subtracted incremental energy.
- At least three repeated sustained runs, with temperature, thread count, clocks/governor, device state, runtime revision and audio workload fixed and recorded.

For continuous real-time replay, total J/audio-second equals average system watts. For accelerated replay it is a different workload; retain both results. Energy/query must include enough silence/tail to reflect use, not just the recognized words. Whole-system measurement follows the principle of edge benchmark power accounting; these measurements are not certified MLPerf results. [MLCommons edge methodology](https://mlcommons.org/benchmarks/inference-edge/)

Use NVML only as a development/training proxy: its board power excludes much of the host, and recent GPUs report power averaged over about one second. Prefer an available total-energy counter difference over a sufficiently long workload, or integrate timestamped power samples. Do not infer utterance energy from a handful of samples around a short kernel. Record training GPU-hours/energy separately from inference efficiency. [NVIDIA NVML documentation](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)

## 9. Work packages and deliverables

Suggested future layout, not files implemented by this planning task:

```text
flake.nix / flake.lock           reproducible shared development environment
custom/PLAN.md                  this plan
custom/pyproject.toml           pinned package/dependency definition
custom/configs/                 model, data, training and evaluation configurations
custom/src/                    speech classifier, gate/buffer state, frontend, encoder, quantization and CTC
custom/data/                   manifest builders and dataset attribution records
custom/runtime/                export format, C++ scalar/native reference and WASM SIMD kernels
custom/web/                    browser harness, AudioWorklet, worker, offline assets and locked dependencies
custom/tests/                  classifier/gate, export/WASM parity, browser audio/offline, CTC and golden tests
custom/benchmarks/              operator, end-to-end latency and energy harnesses
custom/reports/                 versioned results, selection decisions and model card
```

| Milestone | Work and concrete deliverable | Exit / dependency |
| --- | --- | --- |
| M0: contract + environment | Lock Nix/Python/CUDA, Emscripten and browser toolchain; GPU and WASM SIMD smoke checks; speech-activity/evaluation schemas; record development hardware and the iPhone 15 Pro iOS/Safari target | Training environment and browser build load; classifier-first, browser CPU and precision contracts fixed |
| M1a: speech gate first | Train/export classifier to WASM; AudioWorklet/worker capture, resampling, pre-roll, hysteresis, hangover and timestamp-preserving buffers | Classifier runs first in development Chrome and target Safari; state tests and continuous-recording recall/idle measurements pass |
| M1b: ASR feasibility | Scalar/WASM SIMD packed kernels and browser shape benchmarks; frontend, T7 BF16/ternary CTC, tiny overfit and bounded manifests; connect M1a | Learning and gated browser CPU execution work; iPhone 15 Pro Safari feasibility measured before full-scale runs |
| M2: 100-hour pilot | Compare precision/distillation and one convolution-only control; score continuous recordings through the classifier; audit PriMock57, freeze medical splits/Eka release and record conditional dataset eligibility | Select at most two candidates using gated quality/cost; medical manifests and exclusions recorded without tuning on Eka |
| M3: integrated runtime | Audit both exports; WASM parity, bounded buffers/caches, offline reload, interruption tests and gate/endpoint tuning | Foreground iPhone 15 Pro Safari CPU real-time acceptance, including no sustained backlog and latency targets |
| M4a: final-scale feasibility | Profile proposed 0.5B and 1B blocks/full stacks, 5090 microbatch memory/throughput, final-sized Safari assets and sustained active-speech execution | Measured training/data/checkpoint budget and browser feasibility; no inference from tiny-model results |
| M4b: foundation + domain at target scale | Train a roughly 0.5B candidate; explore toward 1B as justified. Start with 960 hours plus eligible PriMock57/TCCC data, assess additional English data needs; optional symptom/synthetic data only if eligible | Final-range recipe selected on development data; provenance and source hours recorded; Eka remains held out |
| M5: device validation | Repeat sustained iPhone 15 Pro Safari power/latency with the trained 0.5–1B model; separate PriMock57/Eka/TCCC held-out results; optional WebGPU/native controls | Final-scale Pareto report with medical-term errors, browser overhead and unmet deployment constraints |
| M6: further compression | Binary weights, smaller activation/cache precision, tokenizer/downsampling and measured kernel-supported pruning experiments | Preserve approximately 0.5–1B final capacity unless scope is explicitly revised; retain energy gains only within quality/latency limits |

Share datasets/manifests and scoring protocols with finetune/ so future pathway comparisons are fair. Do not require implementing or modifying that pathway to begin custom work.

## 10. Decisions still needed before deployment claims

1. Exact ARM64 development laptop, microphone, tested development Chrome/OS and iPhone 15 Pro iOS/Safari versions, and battery-energy budget. The initial power-test phone is fixed as iPhone 15 Pro. Browser WASM SIMD on CPU is the agreed first target; final field hardware remains open.
2. Expected speech/query duty cycle, acceptable response delay and classifier missed-utterance/clipping limits. Speech/no-speech classification is required as the first stage; any future explicit activation control is an additional layer.
3. Required domain vocabulary, real recording access and quality thresholds for consequential transcription distinctions.
4. Whether a requirement beyond scaled ternary learned weights also restricts activations or metadata.

None of these prevents M0–M2. Until they are resolved, architecture sizes and acceptance numbers above remain proposed research settings. The immediate next implementation step is M0, followed by M1a (speech classifier and gating) and M1b (ASR feasibility and integration). The one-hour autoresearch prompt and manual launcher live in custom/autoresearch/. The initial run stopped without real GPU training and produced no useful speech-model result; its artifacts are preserved. The repaired workflow provisions the CUDA environment and real labeled audio, then qualifies a discriminative baseline before starting the research timer. Readiness remains pending until those checks pass; consult the current readiness and run artifacts rather than treating this plan as live status.

## 11. Readiness-gated one-hour GPU autoresearch

See [the adapted agent prompt](autoresearch/program.md) and [launcher instructions](autoresearch/README.md). The failed initial run, 20260909T213516Z, stopped after about 8 minutes 19 seconds without real GPU training. Its procedural synthetic-feature scores and retained all-speech classifier provide no useful speech-model evidence. Preserve this run for audit; do not seed or compare new models against those scores.

Provision and pin the training environment, prepare permitted real speech/non-speech audio, verify CUDA forward/backward/optimizer execution on the RTX 5090, and run a full baseline training/evaluation before the timed service. `control check` verifies current dependencies and data; `control preflight --train-seconds 60` is the minimum training qualification, and `control preflight --train-seconds 300` exercises the full training envelope. Missing, stale or changed readiness evidence blocks `control start --authorize-one-hour`. No schedule or automatic future launch is configured. Preparation also exercises an actual sandbox proposal followed by 60-second CUDA training and independent evaluation. Preparation does not consume the one-hour research allocation.

The supervisor snapshots and freezes the complete ready evaluator/model contract, benchmark, manifests and environment before the timer. It owns Git state, GPU training, independent evaluation, the ledger and deadline. The Codex proposal agent edits only custom/vad/train.py and submits a hypothesis in proposal.json during a bounded 120-second turn; it does not commit, alter labels, install dependencies, run the evaluator or invent metrics. Candidate inference is reconstructed by a fixed evaluator from safe checkpoint tensors/configuration. The research service uses a 3,600-second wall-clock deadline, 300-second training and 420-second total-trial budgets, and a 90-second reporting reserve; systemd controls descendant processes. Reasoning and evaluation also consume time, so this does not promise 100% GPU utilization, particularly for a small classifier. A remaining tail budget may fund an explicitly unranked fresh-seed confirmation of the retained recipe, only when at least 60 seconds of training plus evaluation/report reserves fit. Record its shorter budget and keep it out of equal-budget rankings and best-candidate promotion.

The immediate benchmark uses real-audio clip-presence labels: MiniLibriSpeech utterances and recorded non-speech noise, with exact provenance, license records and separated speaker/source groups in the manifest. It is a classifier feasibility proxy, not frame-VAD or field validation. Procedural synthetic fixtures remain correctness tests only. Provisional candidate feasibility jointly requires FNR <= 0.01 and FPR <= 0.20. Reject constant/all-speech/all-non-speech predictions. Retain a candidate missing the operating gates only as infeasible_best when FNR < 0.95, FPR < 0.95, balanced error < 0.49 and finite score range > 1e-8; report that unmet operating point plainly.

Keep iterating on recoverable candidate failures within the verified environment; missing prerequisites are handled before the clock. Preserve training steps, losses, device/elapsed time, GPU telemetry and every independently evaluated result, including rejected and failed attempts. Report any new unrecoverable failure directly. Medical dataset roles, strict ternary coverage, the 0.5–1B final recognizer target and iPhone 15 Pro Safari power-measurement requirements remain in force. Remote GPU/CPU and clip-proxy results do not establish packed browser execution, continuous speech preservation or iPhone energy savings.
