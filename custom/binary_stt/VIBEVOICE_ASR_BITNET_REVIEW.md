VibeVoice-ASR-BitNet review, 2026-09-24, revised 2026-09-25

The release is most useful to us as a recipe precedent and as two exact
quantizer definitions, not as training code or hyperparameters. Its central
lesson is that Microsoft never trained a quantized speech model from random
initialization: a pretrained floating-point tokenizer was adapted to INT8, and
a pretrained floating-point language model was adapted to ternary weights.
Our runs so far quantize models that have not yet learned to transcribe. The
recommended order therefore remains: obtain a floating-point checkpoint that
produces useful held-out transcripts, then compare ternary-weight/float,
ternary-weight/INT8 and ternary-weight/ternary activation arms from it.

Inspected revisions (rechecked 2026-09-25; nothing changed since the first
review, and no training code or new weights appeared in any of them):

| Source | Revision | Last change |
| --- | --- | --- |
| [Hugging Face release](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet/tree/66e78021ab8f5f06133d1ab421ba4d348bda97c9) | `66e78021ab8f5f06133d1ab421ba4d348bda97c9` | 2026-07-24 |
| [VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp/tree/c4334009c88060f86cdbbd684b62662f710b6c20) | `c4334009c88060f86cdbbd684b62662f710b6c20` | 2026-09-15 |
| [llama.cpp fork used by VibeASR.cpp](https://github.com/XsquirrelC/llama.cpp/tree/a2fdadc20285df2dce90402fca9264a93a8eb32f) | `a2fdadc20285df2dce90402fca9264a93a8eb32f` | submodule pin |
| [VibeVoice](https://github.com/microsoft/VibeVoice/tree/1541f590c7099820f10ea012f48d2399282df69f) | `1541f590c7099820f10ea012f48d2399282df69f` | 2026-09-03 |
| [microsoft/BitNet](https://github.com/microsoft/BitNet/tree/0b341e582afbf9e1011f24744b554c96a3477eb5) | `0b341e582afbf9e1011f24744b554c96a3477eb5` | 2026-07-27 |
| [BitNet report](https://arxiv.org/abs/2607.21075) | arXiv 2607.21075 v2 | 2026-07-25 |
| [VibeVoice-ASR report](https://arxiv.org/abs/2601.18184) | arXiv 2601.18184 v2 | 2026-03-14 |
| [VibeVoice-ASR-Streaming report](https://arxiv.org/abs/2609.02812) | arXiv 2609.02812 v2 | 2026-09-10 |

## What the release contains

The three safetensors shards hold 1,177 FP32 tensors, 2.81 billion parameters:
552 acoustic-tokenizer, 276 semantic-tokenizer, 338 language-model, ten
connector tensors and a tied `lm_head.weight`. No scale, threshold, blending
coefficient or other quantizer state is stored. These are the latent
floating-point weights left by quantization-aware training; every quantization
parameter is recomputed at export. Two GGUF files are the deployable model:
a 703 MB INT8 VAE encoder and a 993 MB ternary decoder with 6-bit embeddings.

The [LM converter](https://github.com/microsoft/VibeASR.cpp/blob/c4334009c88060f86cdbbd684b62662f710b6c20/utils/convert_lm_to_gguf.py)
defines the weight quantizer exactly:

```
s   = 1 / max(mean(|W|), 1e-5)          # one scalar per weight tensor
W_q = clamp(round(W * s), -1, 1) / s
```

It is applied only to the `q/k/v/o/gate/up/down` projection weights. The
Qwen2 `q/k/v` biases and all norms are left unquantized; the tied embedding and output
head are exported as Q6_K. The [I2_S packer](https://github.com/microsoft/VibeASR.cpp/blob/c4334009c88060f86cdbbd684b62662f710b6c20/src/ggml-lm-mad.cpp)
stores codes 0/1/2 for -1/0/+1, treats |w| < 1e-6 as zero, and appends a
single FP32 scale after each tensor's codes. The deployed scale granularity is
therefore per tensor, which matches the one-alpha-per-tensor contract in
[PLAN.md](../PLAN.md) and differs from the per-row learned output scales in
[model.py](model.py).

The runtime activation quantizer is `quantize_row_i8_s` in the pinned fork's
[ggml-quants.c](https://github.com/XsquirrelC/llama.cpp/blob/a2fdadc20285df2dce90402fca9264a93a8eb32f/ggml/src/ggml-quants.c),
line 3503:

```
s   = 127 / max(max(|x|), 1e-5)         # one scalar per token vector
x_q = clamp(nearest_int(x * s), -128, 127)
```

The kernel also records each row's code sum for the unsigned unpack trick.
The VAE path is different: [vae.cpp](https://github.com/microsoft/VibeASR.cpp/blob/c4334009c88060f86cdbbd684b62662f710b6c20/src/vae.cpp)
quantizes the raw audio and intermediate activations with one 127/absmax
scale per whole tensor, and its INT8 graph uses ReLU where the FP graph uses
GELU. I did not locate the VAE weight quantizer in the inspected files.

The [technical report](https://arxiv.org/abs/2607.21075) exists in two
versions two days apart. Version 2 adds one training fact: decoder training
data was limited to segments under four minutes, unlike the 60-minute
long-form audio of the original VibeVoice-ASR. The TeX source contains no
commented-out details. What the report gives on training, gathered from the
text, Figure 3 and the two reports it defers to:

| Item | Source | Value |
| --- | --- | --- |
| Tokenizer QAT objective | Figure 3 axis label | "Distillation Loss (CE)": the INT8 encoder is distilled from its FP teacher, not trained on transcripts |
| Tokenizer QAT stages | Figure 3, semantic encoder | ReLU finetune steps 0 to about 1,100; alpha ramp about 1,100 to 2,200; alpha = 1 about 2,200 to 4,500 |
| Tokenizer loss path | Figure 3 | 2.8 at the GELU-to-ReLU swap, 1.0 by the end of stage 1, a bump to 1.15 when the ramp starts, 0.91 at the end; direct QAT flat at 3.3 for the same 4,500 steps |
| Decoder init and precision | Section 2.3.2 | Pretrained Qwen2.5-1.5B; INT8 tokenizer frozen; ternary weights and per-token INT8 activations; embedding and head 6-bit |
| Decoder ramp | Section 2.3.2 | Not stated; the blend is described only for the tokenizer |
| Decoder data and stages | Section 2.3.2 and base report | Speech-text pretraining on pseudo-labelled audio, then SFT; segments under four minutes |
| Learning rate, batch, optimizer, steps, GPUs | Whole report | Absent |

The [base VibeVoice-ASR report](https://arxiv.org/abs/2601.18184) supplies the
"two-stage procedure": pseudo-labels from Silero VAD segmentation to at most
30 s, Whisper-large-v3-turbo transcripts, WeSpeaker diarization, a second ASR
pass that discards recordings when over 30 percent of segments exceed 20
percent WER or speech is under 60 percent of the duration, a sequence-length
curriculum from 8,192 to 65,536 tokens, then SFT on MLC-SLM and Fisher
training splits, Muse music, about 6,000 hours of synthetic context-aware
audio and GPT-5-restored long-form transcripts, mixed 0.5:0.1:0.1:0.3. Its
language figure, read from the PDF's embedded text, puts English near 66.7
percent of pretraining data and the next language near 14.4 percent; the
total hours are not stated. Two sentences are commented out of its TeX
source and were never published: pretraining the 7B model took about six
days on 128 AMD MI300X GPUs, and SFT about three hours on 16. The
[streaming report](https://arxiv.org/abs/2609.02812), for the same 1.5B and
7B decoder family in floating point, is the only one of the three with an
optimizer appendix: AdamW with betas 0.9 and 0.95, weight decay 0.1,
gradient clipping 2.0, bfloat16, cosine schedule with peak learning rate
5e-5, sequences packed to 8,192 tokens, a multi-node stage over roughly
420,000 hours of English and Chinese, and a final 500-step stage on eight
GPUs at a global batch of 64 sequences with 35 warmup steps. Those numbers
describe floating-point adaptation of a pretrained decoder, not the BitNet
run, and the corpus is several orders of magnitude beyond ours. Reported
accuracy loss versus the 7B FP model is 1 to 4 absolute WER points, which
mixes the size change with the precision change.

The [VibeVoice LoRA example](https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/finetuning-asr/lora_finetune.py)
freezes the tokenizers and adds adapters to the FP language model; it contains
no fake quantizer. The BitNet repository contains inference, conversion and
kernel code only. There is no reproducible BitNet training loop in any of the
inspected sources.

## Our ternary-001 runs against that recipe

The [matched ternary experiment](TERNARY_EXPERIMENT.md) is worth rereading
with the release in mind. All three runs use the 488M model, peak LR 2e-5 and
500 updates. Values below come from each run's `metrics.jsonl` and
`acoustic-check.json` under `/mnt/hd/wilderness-labs-stt/binary-stt/runs/`.

| Run | Train loss, step 10 | Train loss, step 500 | Grad norm, steps 100 to 500 | Held-out at step 500 |
| --- | ---: | ---: | --- | --- |
| `full-lr-low-001` (FP control) | 9.51 | 6.28 | 5.6 to 3.3 | WER 99.4%, every clip decodes to "yeah" |
| `full-ternary-float-001` | 11.72 | 6.30 | 1.9 to 3.5 | WER 99.4%, every clip decodes to "yeah" |
| `full-ternary-both-001` | 20.24 | 6.39 | 32 to 521 | WER 100%, every clip blank |

The identical final metrics of the first two runs are not a sign that the
weight quantizer was inactive. Its configuration has weight quantization from
step 0, its step-10 loss and gradient norm differ from the control, and its
held-out loss differs at steps 100 through 400. Both models simply converged
to the same constant transcript, the frequent AMI token "yeah", which is a
unigram-prior collapse. So at this budget the floating-point recipe itself has
not learned; ternary weights neither helped nor measurably hurt; and ternary
activations added a rising gradient norm, which is the instability pattern
the report describes for direct quantization. None of these runs tests what
Microsoft tested, namely quantizing a model that already works.

There is also a structural difference from both of Microsoft's networks. Their
ternary decoder keeps its SiLU-gated feed-forward and their INT8 tokenizer keeps
ReLU. `BinaryFeedForward` in [model.py](model.py) has no hidden nonlinearity
unless activation quantization is on, so the FP control and the ternary-float
arm both ran with linear feed-forward blocks. That handicap is independent of
precision and should be removed before the next control run.

## What to take, in priority order

1. **Recipe order.** Get a floating-point checkpoint that transcribes held-out
   audio, then quantize from it. Microsoft's evidence is for adapting a
   pretrained model, and the 500-update budgets used so far are far below
   what CTC Conformers normally need to leave the blank plateau. Initializing
   from a pretrained encoder is the lower-risk route to that checkpoint; the
   from-scratch 0.5 to 1B target in [PLAN.md](../PLAN.md) is the harder one.
2. **Feed-forward nonlinearity in every mode.** Add SiLU, or ReLU if INT8
   kernels are the deployment target, between the up and down projections,
   and keep it across all precision arms.
3. **Per-tensor absmean weight quantizer without a learned scale.** The
   four-line converter formula above, as an alternative to the per-row
   learned `log_scale` and its custom straight-through estimator. It matches
   our precision contract and is demonstrated at 1.5B parameters.
4. **Per-token absmax INT8 activation quantizer as a separate control.** The
   `quantize_row_i8_s` formula above. Our `quantizer="ternary"` currently
   forces three-level activations; the missing arm is ternary weights with
   INT8 activations, which is also what the I2_S by I8_S kernels consume.
5. **Ramp only what fails directly, and distil the frontend.** The blend
   was applied to the INT8 tokenizer, where it was trained by distillation
   from the FP tokenizer over about 4,500 steps with roughly a quarter of the
   run on the ramp and half at full quantization. The decoder was trained
   ternary from a pretrained FP start with no reported ramp. Our schedule in
   [train.py](train.py) already separates weight and activation ramps, so try
   weight quantization without a ramp from a working checkpoint, ramp
   activations, and consider distilling a quantized frontend from a
   floating-point frontend rather than training it on CTC labels.
6. **Keep boundary layers at higher precision.** Embeddings and head at
   6-bit, norms and biases FP32 in the release. Our stem projection,
   normalization and CTC head already stay floating point; keep that.
7. **Two-stage data recipe.** Broad speech-text training followed by
   domain finetuning is the release's structure and fits our TCCC glossary
   stage. Supply the glossary independently of the reference transcript and
   measure false insertions as well as term recall.
8. **CPU teacher candidate.** VibeVoice-ASR-BitNet runs on CPU through
   VibeASR.cpp and could label project audio through the pattern in
   [teachers.py](../stt-distillation/teachers.py). Benchmark it against the
   existing teachers on manually transcribed recordings first. Its 151,936
   token Qwen2 vocabulary and autoregressive decoder rule out frame-level
   logit distillation into our CTC model; transcript supervision is fine.
9. **Optional reference statistics.** The safetensors shards are converged
   QAT latent weights. Downloading them, about 11.2 GB, and histogramming the
   ternary codes per projection would give a reference zero fraction for a
   healthy ternary model that [health.py](health.py) could compare against.
   Not done in this review.

Not transferable: training hyperparameters, which are unpublished; the
release's accuracy numbers, which do not cover our recordings; and the
kernels as drop-in code, since our current W1A1 popcount path uses different
operands. The I2_S by I8_S kernel family is still the closest existing
implementation of the PLAN.md contract on AVX2 and NEON, and any future
export must match its per-tensor weight scale, per-token activation scale,
rounding and clamp range. The LM converter swaps the checkpoint files in
place during conversion, so run it on a copy.

The [model card](https://huggingface.co/microsoft/VibeVoice-ASR-BitNet)
reports a 1.58 GB deployable model and LibriSpeech clean/other WER of
2.41%/6.27%. Its memory comparison uses the smaller FP16 model while its
accuracy table uses the 7B model as reference, so quantization loss is not
isolated. RTF figures are CPU runtime, not energy or microphone-to-text
latency, and the GitHub README reports different timings from the card, so
future benchmarks must pin revisions and workloads.

A bounded experiment should use one small conventional architecture, fixed
tokenization, identical data order and matched budgets:

1. Train or adapt a floating-point reference until it produces useful held-out
   transcripts. Freeze its evaluation panel and starting checkpoint.
2. Branch into floating-point, ternary-weight/float-activation and
   ternary-weight/INT8-activation arms, with the feed-forward nonlinearity,
   frontend and CTC head held fixed.
3. Compare direct and progressive quantization on the promising arm, with a
   meaningful final phase at alpha one, and evaluate the fully quantized
   computation rather than a blended checkpoint.
4. Test ternary or binary activations only after those controls learn, and
   repeat with paired seeds before scaling size or data.

Track held-out WER/CER, empty and constant transcripts, deletions and
insertions, domain-term and number recognition, and robustness to silence,
gain and noise. Select a deployable result on measured energy per audio
minute, latency and memory alongside accuracy.

Both reviews downloaded no model weights or datasets, ran no training or
benchmarks, and changed no executable code. Future artifacts must remain under
`/mnt/hd/wilderness-labs-stt/` with an explicit mount check.
