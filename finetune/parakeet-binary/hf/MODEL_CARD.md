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
- binary
- 1-bit
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
      value: 2.20
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
      value: 4.56
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
      value: 10.65
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
      value: 12.17
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
      value: 10.46
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
      value: 3.01
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
      value: 6.30
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
      value: 13.42
      name: Test WER
---

# Kea-B: Parakeet-TDT-0.6B-v2 with 1-bit encoder weights

Kea-B is NVIDIA's [Parakeet-TDT-0.6B-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
English speech recognizer with its large encoder matrices stored at one bit per weight: every
weight is +1 or -1, times one scale per row. The weight file is 105.3 MB, against 2,472 MB for
the original `.nemo` file, and the mean word error rate (WER) on seven Open ASR Leaderboard test
sets goes from 6.45% to 7.05%. It is the binary sibling of
[Kea-T](https://huggingface.co/rajb3/kea-t), which uses ternary weights (-1, 0, +1), takes
180.8 MB and scores 6.84%.

## Results

Test WER (%), greedy decoding:

| Model | LS clean | LS other | AMI | Earnings-22 | GigaSpeech | SPGISpeech | VoxPopuli | Mean of 7 | Common Voice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, FP32 (our scoring) | 1.70 | 3.19 | 11.15 | 11.24 | 9.78 | 2.14 | 5.94 | 6.45 | 8.50 |
| [Kea-T](https://huggingface.co/rajb3/kea-t) | 2.05 | 4.20 | 10.42 | 11.72 | 10.35 | 2.94 | 6.19 | 6.84 | 12.56 |
| **Kea-B (this model)** | 2.20 | 4.56 | 10.65 | 12.17 | 10.46 | 3.01 | 6.30 | 7.05 | 13.42 |

We scored all three models with the same code on every utterance of the public Open ASR
Leaderboard test bundle. TED-LIUM isn't in that bundle, so the mean covers the other seven sets,
and Common Voice is shown on its own. Decoding is greedy in FP32, and both the references and the
outputs go through the Whisper English text normalizer. Kea-B was scored once, on the model
rebuilt from this exact file. Per-set details, NVIDIA's published numbers and the rate of empty
outputs are in
[results/TEST.md](https://github.com/ajbarryiii/wilderness-labs-stt/blob/main/finetune/parakeet-binary/results/TEST.md).

## How it was trained

- We started from Kea-T rather than from the original model and kept its training recipe.
- In each of the 264 large encoder matrices (the attention and feed-forward projections and the
  pointwise convolutions), a weight W becomes sign(W) times the mean |W| of its row. Over the
  first 62,500 steps the weights were blended from Kea-T's ternary values to these binary ones;
  after that they stayed fully binary.
- The original model was the teacher. Kea-B learned the original's transcripts of each training
  batch and matched its encoder outputs, and the prediction and joint networks kept NVIDIA's
  weights.
- Training ran 250,000 steps of up to 600 s of audio each, with a peak learning rate of 2e-4. That
  is 33,246 hours of audio from YODAS, LibriSpeech, People's Speech, VoxPopuli and AMI, in a new
  order rather than Kea-T's. It took 68.6 hours on one RTX 5090.
- The remaining learned weights are stored in FP16. We checked the export against the trained
  model: the bits, the scales and the transcripts of four check clips were identical.

## Use

You need `nemo_toolkit[asr]` (which brings `torch`), `safetensors` and `soundfile`.

```sh
pip install "nemo_toolkit[asr]"
hf download {repo_id} --local-dir kea-b
python kea-b/load_binary.py clip.wav    # 16 kHz audio; prints the transcript
```

```python
import sys; sys.path.insert(0, "kea-b")
from load_binary import load, transcribe
model = load("kea-b", device="cuda")    # or "cpu"
print(transcribe(model, ["clip.wav", "clip2.flac"]))
```

`load()` checks the files against the SHA-256 hashes in `manifest.json`, unpacks the bits and
rebuilds an ordinary NeMo model. `transcribe()` decodes the way our evaluation did, and the
output is punctuated and capitalized like the original's.

The rebuilt model is a dense FP32 model, so it needs as much memory and compute as the original.
For now, the small file only saves download time and disk space. We are working on native 1-bit
kernels, but they are not part of this release.

## Files

- `export.safetensors` (105,299,804 bytes): the packed bits (eight weights per byte, least
  significant bit first, 1 = +1), one FP32 scale per row, and the remaining learned weights in FP16.
- `manifest.json`: file hashes, the NeMo config, layer shapes, byte accounting and provenance.
- `reconstruction.json`: the check of the export against the trained model.
- `tokenizer/`: the original SentencePiece tokenizer, unchanged.
- `load_binary.py`: the loader and a small command-line tool (MIT License).

## Limitations

- Common Voice has the largest gap: 13.42% against 8.50% for the original. It is not in the
  training data, so expect bigger losses on audio that is far from it.
- English only, like the base model.
- Each input is decoded in one pass with full attention. The model was trained on clips of
  1 to 30 s, so split long recordings into pieces of about 30 s.
- Tested with NeMo 3.0.0 and torch 2.11.0+cu128. The WERs were measured on an RTX 5090 in FP32
  with greedy decoding. Loading on the CPU was tested, but decoding on the CPU was not.

## License and attribution

The weights and this card are licensed
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (see `LICENSE`), like the base model.
Kea-B is a modified version of
[nvidia/parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) by NVIDIA. The
encoder's projection and pointwise-convolution weights were retrained as 1-bit weights with
per-row scales, the remaining learned weights are stored in FP16, and the model ships as
safetensors with a standalone loader instead of a `.nemo` file. `NOTICE` lists the changes and
the training data with their licenses. `load_binary.py` is MIT-licensed, and so is the
[training and evaluation code](https://github.com/ajbarryiii/wilderness-labs-stt/tree/main/finetune/parakeet-binary).
