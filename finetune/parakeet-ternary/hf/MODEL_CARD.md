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

# Kea-T: Parakeet-TDT-0.6B-v2 with ternary encoder weights {-1, 0, +1}

Kea-T is NVIDIA's English speech recognizer
[nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) with every
large matrix in its 24-layer FastConformer encoder (the attention and feed-forward projections
and the pointwise convolutions, 98% of the model's parameters) constrained to three values
(-1, 0, +1) with one FP32 scale per output row. The accuracy loss is largely recovered by
quantization-aware training (QAT) with distillation from the original model. Its 1-bit sibling
[Kea-B](https://huggingface.co/rajb3/kea-b) is 105.3 MB with a 7.05% test mean.

## At a glance

| | Kea-T | Original |
| --- | ---: | ---: |
| Weight file | **180.8 MB** | 2,472 MB |
| Test WER, mean of 7 sets | 6.84% | 6.45% |
| Latency, 10 s clip | **4.46 ms** (23.2% lower) | 5.81 ms |
| GPU energy, 10 s clip | **1.57 J** (17.1% less) | 1.90 J |
| Process VRAM, 10 s clip | **1.89 GB** (13.5% lower) | 2.18 GB |

The original's size is NVIDIA's `.nemo` file, and its WER is the FP32 model as we scored it.
The speed rows are warm batch-one transcription on one RTX 5090 at a 400 W power limit, Kea-T
in expanded mode against our own optimized BF16 build of the original (not stock NeMo). That
speed needs our
[custom CUDA runtime](https://github.com/ajbarryiii/wilderness-labs-stt/tree/main/finetune/parakeet-ternary/inference)
on GitHub, not `load_ternary.py`. The runtime's compact
mode uses 1.27 GB of process VRAM, 41.8% less than that build. All configurations, the 3 s and
30.04 s clips and the method are in [RUNTIME.md](RUNTIME.md).

## Results

Test WER (%), greedy decoding:

| Model | LS clean | LS other | AMI | Earnings-22 | GigaSpeech | SPGISpeech | VoxPopuli | Mean of 7 | Common Voice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, FP32 (our scoring) | 1.70 | 3.19 | 11.15 | 11.24 | 9.78 | 2.14 | 5.94 | 6.45 | 8.50 |
| **Kea-T (this model)** | 2.05 | 4.20 | 10.42 | 11.72 | 10.35 | 2.94 | 6.19 | 6.84 | 12.56 |

We scored both models with the same code on every utterance of the public Open ASR Leaderboard
test bundle. TED-LIUM isn't in that bundle, so the mean covers the other seven sets, and Common
Voice is shown on its own. Decoding is greedy in strict FP32, and both the references and the
outputs go through the Whisper English text normalizer. Our scoring of the FP32 original
reproduces NVIDIA's published numbers within 0.09 points per set. Kea-T was scored once, on the
model rebuilt from this exact `export.safetensors`. NVIDIA's published row, the difference and
ratio rows and the scoring details are in [EVALUATION.md](EVALUATION.md).

## How it was trained

- We started from NVIDIA's pretrained FP32 checkpoint. In each of the 264 ternary matrices, a
  weight W becomes `clamp(round(W / s), -1, 1)` times its row's scale s, the row's mean |W|.
  Over the first 25% of steps (62,500) the weights were blended from float to fully ternary;
  after that they stayed fully ternary.
- The frozen original model was the teacher. Kea-T learned the original's transcripts of each
  training batch and matched its encoder outputs, and the prediction and joint networks kept
  NVIDIA's weights.
- Training ran 250,000 steps of batches of up to 600 s of audio, with a peak learning rate of
  5e-4: 33,248 hours of audio from YODAS-Granary, LibriSpeech, People's Speech, VoxPopuli and
  AMI. It took 69.3 hours on one RTX 5090.
- The remaining learned weights are stored in FP16. We checked the export against the trained
  model: the codes, the scales and the greedy transcripts of four check clips were identical.

The full recipe, the data mix, the pilot and the file format are in [TRAINING.md](TRAINING.md).

## Use

Needs `nemo_toolkit[asr]` (with its `torch`), `safetensors` and `soundfile`.

```sh
pip install "nemo_toolkit[asr]"
hf download {repo_id} --local-dir kea-t
python kea-t/load_ternary.py clip.wav       # 16 kHz audio; prints the transcript
```

```python
import sys; sys.path.insert(0, "kea-t")
from load_ternary import load, transcribe
model = load("kea-t", device="cuda")                    # or "cpu"
print(transcribe(model, ["clip.wav", "clip2.flac"]))    # paths or 1-D 16 kHz float arrays
```

`load()` checks the weights and tokenizer files against the SHA-256 hashes in `manifest.json`,
unpacks the ternary codes and rebuilds a standard NeMo model in FP32; `transcribe()` decodes
exactly as the evaluation did (greedy TDT, FP32, dither 0). Output is punctuated and
capitalized, like the original. The rebuilt model needs the same memory and compute as the
original, so with `load_ternary.py` the small file is a storage and download saving only.

Each input is decoded in one pass with full attention, with no chunking. The model was trained
on 1 to 30 s utterances; split long recordings into segments of about 30 s or less.

## Files

- `export.safetensors` (180,797,564 bytes): the ternary codes packed four per byte, one FP32
  scale per output row, and every other learned tensor in FP16.
- `manifest.json` (hashes, configuration, accounting, provenance), `reconstruction.json` (the
  check of the export against the trained model) and `tokenizer/` (unchanged).
- `load_ternary.py`: the standalone loader and a small transcription CLI (MIT License).
- `LICENSE` (CC-BY-4.0 legal code) and `NOTICE` (attribution and changes).
- [RUNTIME.md](RUNTIME.md), [EVALUATION.md](EVALUATION.md) and [TRAINING.md](TRAINING.md).

## Limitations

- English only, like the base model.
- The gap to the original is largest on Common Voice (+4.06 points), which is outside the
  training mixture, followed by LibriSpeech test-other (+1.01) and SPGISpeech (+0.81). Expect
  larger degradation on domains far from the training data.
- The runtime results were measured on one RTX 5090 at a 400 W limit, warm, at batch one, with
  one clip per length.
- Only greedy decoding was evaluated. Tested with NeMo 3.0.0 and torch 2.11.0+cu128; loading on
  the CPU was tested, decoding on the CPU was not. More in
  [EVALUATION.md](EVALUATION.md#limitations).

## Used in

- [LocalFlow](https://github.com/ajbarryiii/LocalFlow), an offline macOS
  dictation app (beta; Apple Silicon, macOS 26 or later) forked from
  [FreeFlow](https://github.com/zachlatta/freeflow). It runs a Core ML
  conversion of these weights (ternary lookup tables, FP16 values) on the
  Mac's Neural Engine, and audio never leaves the device.

## License and attribution

The model weights, this model card and its detail documents (`RUNTIME.md`, `EVALUATION.md`,
`TRAINING.md`) are licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
(see `LICENSE`), the license of the base model. The model is a modified version of
[nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) by NVIDIA:
the encoder's projection and pointwise-convolution weights were retrained with ternary
quantization-aware training and are stored as ternary codes with per-row FP32 scales; the
remaining learned weights are stored in FP16; the weights are distributed as safetensors with a
standalone loader instead of a `.nemo` file. `NOTICE` lists the changes and
the training data, and [TRAINING.md](TRAINING.md#training-data-and-licenses) their licenses.
`load_ternary.py` is MIT-licensed (notice at the top of the file), and so is the
[training and evaluation code](https://github.com/ajbarryiii/wilderness-labs-stt/tree/main/finetune/parakeet-ternary).
