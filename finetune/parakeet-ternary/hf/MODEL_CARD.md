---
license: cc-by-4.0
language:
- en
library_name: nemo
pipeline_tag: automatic-speech-recognition
base_model: nvidia/parakeet-tdt-0.6b-v2
datasets:
- espnet/yodas-granary
- openslr/librispeech_asr
- MLCommons/peoples_speech
- facebook/voxpopuli
- edinburghcstr/ami
tags:
- automatic-speech-recognition
- speech
- audio
- NeMo
- parakeet
- TDT
- FastConformer
- quantization
- ternary
- bitnet
- quantization-aware-training
- knowledge-distillation
model-index:
- name: "{repo_name}"
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: LibriSpeech test-clean (ESB test bundle)
      type: hf-audio/esb-datasets-test-only-sorted
      config: librispeech
      split: test.clean
      revision: b6bdcd0beb
      args:
        language: en
    metrics:
    - type: wer
      value: 2.05
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: LibriSpeech test-other (ESB test bundle)
      type: hf-audio/esb-datasets-test-only-sorted
      config: librispeech
      split: test.other
      revision: b6bdcd0beb
      args:
        language: en
    metrics:
    - type: wer
      value: 4.20
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: AMI test (ESB test bundle)
      type: hf-audio/esb-datasets-test-only-sorted
      config: ami
      split: test
      revision: b6bdcd0beb
      args:
        language: en
    metrics:
    - type: wer
      value: 10.42
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Earnings-22 test (ESB test bundle)
      type: hf-audio/esb-datasets-test-only-sorted
      config: earnings22
      split: test
      revision: b6bdcd0beb
      args:
        language: en
    metrics:
    - type: wer
      value: 11.72
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: GigaSpeech test (ESB test bundle)
      type: hf-audio/esb-datasets-test-only-sorted
      config: gigaspeech
      split: test
      revision: b6bdcd0beb
      args:
        language: en
    metrics:
    - type: wer
      value: 10.35
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: SPGISpeech test (ESB test bundle)
      type: hf-audio/esb-datasets-test-only-sorted
      config: spgispeech
      split: test
      revision: b6bdcd0beb
      args:
        language: en
    metrics:
    - type: wer
      value: 2.94
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: VoxPopuli test (ESB test bundle)
      type: hf-audio/esb-datasets-test-only-sorted
      config: voxpopuli
      split: test
      revision: b6bdcd0beb
      args:
        language: en
    metrics:
    - type: wer
      value: 6.19
      name: Test WER
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: Common Voice test (ESB test bundle)
      type: hf-audio/esb-datasets-test-only-sorted
      config: common_voice
      split: test
      revision: b6bdcd0beb
      args:
        language: en
    metrics:
    - type: wer
      value: 12.56
      name: Test WER
---

# Parakeet-TDT-0.6B-v2 with ternary encoder weights {-1, 0, +1}

On one RTX 5090, the custom CUDA runtime in our GitHub repository transcribes a 10 s
clip from these weights in 4.46 ms, with 1.89 GB of process VRAM and 1.57 J of GPU
energy (expanded mode). That is 23.2% lower latency, 13.5% lower process VRAM and
17.1% less GPU energy than our own optimized BF16 build of the original. The compact
mode uses 1.27 GB, 41.8% less VRAM than that build. The weight file is **180.8 MB**,
against 2,472 MB for the original `.nemo` file. On the seven Open ASR Leaderboard
test sets we could score, mean WER rises from 6.45% to 6.84%.

Every large matrix in the 24-layer FastConformer encoder of
[nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
(the attention and feed-forward projections and the pointwise convolutions, 98% of
the model's parameters) is constrained to three values (-1, 0, +1) with one FP32
scale per output row. The accuracy loss is largely recovered by quantization-aware
training (QAT) with distillation from the original model.

## Runtime results (custom CUDA runtime on GitHub)

The [custom CUDA runtime](https://github.com/ajbarryiii/wilderness-labs-stt/tree/main/finetune/parakeet-ternary/inference)
runs `export.safetensors` from its packed codes, without rebuilding dense floating-point
encoder weights on the GPU.
The runtime is MIT-licensed code in the GitHub repository and is not part of this
download. `load_ternary.py` here still rebuilds a dense FP32 NeMo model (see Use).

The table shows warm batch-one transcription of a 10 s LibriSpeech clip on one
RTX 5090 at a 400 W power limit:

| Configuration | Latency | Process VRAM | GPU energy |
| --- | ---: | ---: | ---: |
| Ternary expanded — optimized by Wilderness Labs | **4.46 ms** | 1.89 GB | **1.57 J** |
| Ternary compact — optimized by Wilderness Labs | 5.27 ms | **1.27 GB** | 1.86 J |
| Original BF16 — optimized by Wilderness Labs | 5.81 ms | 2.18 GB | 1.90 J |
| ONNX ASR / ORT CUDA — off the shelf | 15.74 ms | 4.14 GB | 3.75 J |

- **Ternary expanded** (`--optimized --encoder-storage expanded`) keeps the packed
  codes and scales plus a cached exact INT8 copy of the codes (603,979,776 bytes,
  about 604 MB). Against our optimized BF16 build of the original at 10 s, it has
  23.2% lower latency, 13.5% lower process VRAM and 17.1% less GPU energy. It is
  also lower on all three at the 3 s and 30.04 s clips.
- **Ternary compact** (`--optimized`) keeps the ternary matrices as packed 2-bit
  codes with per-row FP32 scales and unpacks them inside the kernels. It used the
  least process VRAM of the four configurations at every length tested: 41.8% less
  than BF16 at 10 s, with 9.3% lower latency. Its 10 s energy is within 2% of BF16,
  and at 30.04 s its latency and energy are higher than BF16's.
- **Original BF16 — optimized by Wilderness Labs** is NVIDIA's original checkpoint
  with a BF16 encoder, plus our own CUDA graphs, fused FP32 decoder and graph
  allocation fix. It is not stock NeMo. The two checkpoints produce different
  transcripts and therefore different decoder work, so this row compares complete
  runtimes, not the effect of ternarization alone.
- **ONNX ASR / ORT CUDA — off the shelf** is the unmodified
  [onnx-asr](https://github.com/istupakov/onnx-asr) 0.12.0 package with
  onnxruntime-gpu 1.24.4. It runs the published FP32 export
  [istupakov/parakeet-tdt-0.6b-v2-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx)
  at a pinned revision, with the CUDA execution provider and default options. It is
  one common way to run this model without NeMo, and its weights are FP32 rather
  than BF16. Our BF16 build also has 63.1% lower latency than this row; the two
  configurations differ in precision and runtime. This comparison is not a survey
  of optimized engines.

Both ternary modes use the weights in this repository and keep the non-ternary
tensors in floating point. Both run the ternary matrices on INT8 Tensor Cores, with
each activation split into three residual INT8 components, int32 accumulation and
FP32 outputs, and both run a fused FP32 prediction/joint decoder, with NeMo's greedy
TDT loop in conditional CUDA graphs. The runtime needs the GitHub
repository's pinned environment (NeMo 3.0, PyTorch 2.11 with CUDA 12.8, Triton 3.6,
NVRTC 12.9); it is not a pip package. The
[runtime guide](https://github.com/ajbarryiii/wilderness-labs-stt/tree/main/finetune/parakeet-ternary/inference)
has results for 3, 10 and 30.04 s clips, plus usage and reproduction steps.

*Method and quality.*

- Latency covers a CPU waveform in to CPU text out, including feature extraction and
  host-device transfers, and excludes file I/O, loading, compilation and first graph
  capture.
- Each configuration ran in a fresh process, with one clip per length. Values are
  medians of three 5 s windows.
- Process VRAM is the warm process's sampled peak, in decimal GB.
- GPU energy is NVML's board counter and excludes host power. The ternary rows draw as
  much power or more and use less energy because they finish sooner.
- Compact and expanded modes produce identical text, tokens and timestamps on 2,048
  utterances from eight test sets. That checks agreement between the two modes, not
  accuracy against the original. Recognition accuracy for this model is the
  full-corpus WER below.

## Recognition accuracy

Test WER (%), greedy decoding:

| Model | LS clean | LS other | AMI | Earnings-22 | GigaSpeech | SPGISpeech | VoxPopuli | Mean of 7 | Common Voice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, NVIDIA published | 1.69 | 3.19 | 11.16 | 11.15 | 9.74 | 2.17 | 5.95 | 6.44 | n/a |
| Original, FP32, our scoring | 1.70 | 3.19 | 11.15 | 11.24 | 9.78 | 2.14 | 5.94 | 6.45 | 8.50 |
| Same ternary quantizer, no training | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| **This model** | 2.05 | 4.20 | 10.42 | 11.72 | 10.35 | 2.94 | 6.19 | 6.84 | 12.56 |
| Difference from our FP32 scoring (points) | +0.36 | +1.01 | -0.72 | +0.48 | +0.57 | +0.81 | +0.25 | +0.39 | +4.06 |
| Ratio to our FP32 scoring | 1.21x | 1.32x | 0.94x | 1.04x | 1.06x | 1.38x | 1.04x | 1.06x | 1.48x |

- **Test sets:** the Open ASR Leaderboard bundle
  [hf-audio/esb-datasets-test-only-sorted](https://huggingface.co/datasets/hf-audio/esb-datasets-test-only-sorted)
  (revision `b6bdcd0beb`, original, not `_cleaned`, configs), every utterance.
- **Mean of 7:** the unweighted mean over the eight sets NVIDIA reports for this
  model except TED-LIUM, which is not in the public bundle and was not evaluated.
  NVIDIA's published mean in that column is over the same seven sets (their
  TED-LIUM result, 3.38, is excluded).
- **Common Voice** is outside that mean and is reported separately. It shows the
  largest gap (+4.06 points); it is not part of the training mixture.
- **Scoring:** corpus-level WER with the Whisper English text normalizer on both
  sides (the leaderboard's normalizer); as on the leaderboard, utterances whose
  normalized reference is empty are decoded but not scored. Greedy TDT decoding
  with the model's own decoding configuration, in strict FP32 (no TF32, no
  autocast; the current leaderboard script uses BF16). Our scoring of the FP32
  original reproduces NVIDIA's published numbers within 0.09 points per set.
- **This model** was scored once, on the model rebuilt from this exact
  `export.safetensors` by the same procedure `load_ternary.py` implements.
  Differences and ratios are computed from unrounded WERs, so they can differ by
  0.01 from the difference of the rounded entries.

Code, design document, full results and per-utterance evaluation procedure:
[github.com/ajbarryiii/wilderness-labs-stt](https://github.com/ajbarryiii/wilderness-labs-stt/tree/main/finetune/parakeet-ternary).

## Use

Needs `nemo_toolkit[asr]` (with its `torch`), `safetensors` and `soundfile`.

```sh
pip install "nemo_toolkit[asr]"
hf download {repo_id} --local-dir parakeet-ternary
python parakeet-ternary/load_ternary.py clip.wav       # 16 kHz audio; prints the transcript
```

```python
import sys; sys.path.insert(0, "parakeet-ternary")
from load_ternary import load, transcribe
model = load("parakeet-ternary", device="cuda")          # or "cpu"
print(transcribe(model, ["clip.wav", "clip2.flac"]))     # paths or 1-D 16 kHz float arrays
```

`load()` checks the SHA-256 of `export.safetensors`, `tokenizer/tokenizer.model`,
`tokenizer/tokenizer.vocab` and `tokenizer/vocab.txt` against the hashes recorded
in `manifest.json` (the manifest itself is not hash-checked), unpacks the
ternary codes and rebuilds a standard NeMo `EncDecRNNTBPEModel` in FP32 with the
dequantized weights (strict `load_state_dict`). It is a plain NeMo model
afterwards. `transcribe()` decodes exactly as the evaluation did (greedy TDT,
FP32, dither 0). Output is punctuated and capitalized, like the original.

**Tested environment:** NeMo 3.0.0 and torch 2.11.0+cu128. The WERs above were
measured decoding on an NVIDIA RTX 5090 (CUDA, strict FP32). Loading on the CPU
was tested; decoding on the CPU was not. Other environments rebuild the same
weights, but WER was only measured as described here.

The rebuilt model runs dense FP32 matrices, so it needs the same memory and
compute as the original. With `load_ternary.py`, the small file is a storage and
download saving only. This Hugging Face repository ships no packed ternary
kernels. The runtime results above come from the separate custom CUDA runtime on
GitHub, which reads the same `export.safetensors`.

**Long audio:** each input is decoded in one pass with full attention, with no
chunking; memory grows roughly quadratically with length. The model was trained
on 1 to 30 s utterances and evaluated on test utterances up to about 106 s.
Longer inputs were not evaluated; split long recordings into segments of about
30 s or less.

## What is in the files

- `export.safetensors` (180,797,564 bytes):
  - 264 ternary weight matrices, 603,979,776 parameters: in each of the 24
    encoder layers, `feed_forward1.linear1`, `feed_forward1.linear2`,
    `feed_forward2.linear1`, `feed_forward2.linear2`, `self_attn.linear_q`,
    `linear_k`, `linear_v`, `linear_out`, `linear_pos`, and
    `conv.pointwise_conv1`, `conv.pointwise_conv2` (kernel-1 convolutions, stored
    as [out, in] matrices). Each is stored as 2-bit codes packed four per byte
    (150,994,944 bytes) plus one FP32 scale per output row (442,368 rows,
    1,769,472 bytes). None of these layers has a bias.
  - Every other learned tensor, 13,846,150 parameters, plus the BatchNorm
    running statistics, in FP16 (27,790,604 bytes): the convolutional subsampling
    front end and its output projection, depthwise convolutions, norms, biases,
    the LSTM prediction network and the joint network. The feature-extraction
    constants (`preprocessor.featurizer.window` and `.fb`, 133,184 bytes) stay
    FP32, and the BatchNorm `num_batches_tracked` counters keep their integer type.
  - Code distribution over all ternary weights: {code_histogram}.
- `manifest.json`: format, SHA-256 of `export.safetensors` and the three
  tokenizer files, the full NeMo model
  configuration, per-layer shapes, per-module code histograms, parameter and
  byte accounting, and training provenance (run name, step, source hashes).
- `reconstruction.json`: the export check against the trained QAT model:
  codes and scales exact for all 264 layers, identical greedy transcripts on four
  LibriSpeech dev-clean clips, and a maximum absolute encoder-output difference of
  about 5.5e-4 (exact value in the file; from FP16 storage of the float tensors).
- `tokenizer/`: the base model's SentencePiece tokenizer (1,024 tokens),
  unchanged.
- `load_ternary.py`: the standalone loader and a small transcription CLI (MIT
  License; the copyright and permission notice is at the top of the file).
- `LICENSE` (CC-BY-4.0 legal code) and `NOTICE` (attribution and changes).

## How it was trained

- **Start:** the pretrained FP32 checkpoint (revision
  `ae9ad07059c7c739ffaf932226a8fe64ae2620b0`), FP32 latent weights.
- **Quantizer:** per output row `s = max(mean|W|, 1e-8)`,
  `codes = clamp(round(W / s), -1, 1)`, forward weight exactly `codes * s`,
  recomputed every forward pass, with an identity straight-through gradient to the
  latent weights. Activations are not quantized. A linear ramp from float to fully
  ternary weights over the first 25% of steps (62,500), then fully ternary; only
  the fully ternary final checkpoint was exported.
- **Supervision:** the frozen original model (FP32 weights, encoder run in BF16
  autocast) produced greedy transcripts of every training batch online
  (sequence-level distillation; the student learns the original's punctuated,
  cased output), plus an encoder-output matching loss,
  `mean((E_student - E_teacher)^2) / var(E_teacher)` over valid frames, weight 1.0.
  The prediction and joint networks were frozen at their original weights. Human
  transcripts were used only to drop utterances whose teacher transcript disagreed
  with them by more than 50% normalized WER; utterances outside 1 to 30 s or with
  an empty teacher transcript were also dropped.
- **Recipe:** 250,000 steps of batches of up to 600 s of audio, AdamW (betas 0.9,
  0.98, weight decay 0.01 on matrices), peak learning rate 5e-4 with 2% linear
  warmup and linear decay to zero, gradient clipping 1.0, BF16 autocast with FP32
  master weights, SpecAugment on the student input only, seed 20260930. One
  RTX 5090, 69.3 hours of training (about 2.9 days).
- **Data:** 33,248 hours of audio exposure, streamed from the Hugging Face Hub
  and interleaved by audio-hour share: YODAS-Granary English (22,040 h, 66.3%),
  People's Speech `clean` (3,370 h, 10.1%), LibriSpeech 960 h training sets
  (3,188 h, 9.6%), VoxPopuli English train (3,145 h, 9.5%) and AMI IHM train
  (1,504 h, 4.5%). The per-source hours are rounded separately and sum to 33,247;
  the unrounded total is 33,247.6 h.
- **Pilot (12,000 steps per arm, scored on full development sets):** adding the
  encoder-output matching loss improved the five-set development mean WER from
  7.05% to 6.85% and helped on all five sets; additionally making the prediction
  and joint networks trainable changed it by 0.004 points (noise), so they stayed
  frozen. Of learning rates 2e-4, 5e-4 and 1e-3, 5e-4 was best. The final model's
  development mean WER is 6.09% (FP32 original 5.58%).

## Limitations

- English only, like the base model.
- `load_ternary.py` keeps activations in floating point and rebuilds dense FP32
  weights, so no speed, memory or energy benefit is claimed for that loader. The
  runtime results above apply only to the custom CUDA runtime on GitHub. They were
  measured on one RTX 5090 at a 400 W limit, warm, at batch one, with one clip per
  length. Other GPUs, batch sizes, cold starts and host energy were not measured.
- The gap to the original is largest on Common Voice (+4.06 points), which is
  outside the training mixture, followed by LibriSpeech test-other (+1.01) and
  SPGISpeech (+0.81). Expect larger degradation on domains far from
  the training data.
- AMI: 1.43% of this model's AMI test hypotheses are empty, fewer than the FP32
  original's 3.12%. The utterances empty only for this model are mostly one-word
  backchannels.
- Only greedy decoding was evaluated. Beam search and inputs longer than the test
  utterances were not evaluated. Timestamp accuracy was not evaluated either: the
  runtime check that compact and expanded modes emit identical timestamps shows
  that the two modes agree, not that the timestamps are correct. TED-LIUM was not
  evaluated.
- YODAS (training data) and GigaSpeech (a test set) both draw on YouTube, so some
  test audio may overlap the training audio. The original model was also trained
  on YODAS, so the same applies to its numbers.
- No FP32 model was fine-tuned on the same data as a control; the comparison is
  against the original model, so the effect of the training data and of
  quantization are not separated.

## License and attribution

The model weights and this model card are licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (see `LICENSE`), the
license of the base model. The model is a modified version of
[nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
by NVIDIA. Changes: the encoder's projection and pointwise-convolution weights
were retrained with ternary quantization-aware training and are stored as
ternary codes with per-row FP32 scales; all other tensors are stored in FP16
(feature-extraction constants FP32); the weights are distributed as safetensors
with a standalone loader instead of a `.nemo` file. See `NOTICE` for details.
The loader, `load_ternary.py`, is software under the MIT License (copyright and
permission notice at the top of the file).

Training audio (used with transcripts generated by the base model):

- YODAS-Granary English,
  [espnet/yodas-granary](https://huggingface.co/datasets/espnet/yodas-granary):
  CC-BY-3.0. References:
  [Yodas: Youtube-oriented dataset for audio and speech](https://arxiv.org/abs/2406.00899);
  [Granary: Speech Recognition and Translation Dataset in 25 European Languages](https://arxiv.org/pdf/2505.13404).
- LibriSpeech,
  [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr):
  CC-BY-4.0.
- People's Speech `clean`,
  [MLCommons/peoples_speech](https://huggingface.co/datasets/MLCommons/peoples_speech):
  the `clean` configuration selects CC-BY material; the CC-BY version (2.0, 2.5,
  3.0 or 4.0) depends on the source recording.
- VoxPopuli English,
  [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli): CC0-1.0.
- AMI IHM, [edinburghcstr/ami](https://huggingface.co/datasets/edinburghcstr/ami):
  CC-BY-4.0 ([publisher license statement](https://groups.inf.ed.ac.uk/ami/corpus/)).

The training and evaluation code
([repository](https://github.com/ajbarryiii/wilderness-labs-stt/tree/main/finetune/parakeet-ternary))
is MIT-licensed.
