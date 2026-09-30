---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: automatic-speech-recognition
base_model: openai/whisper-tiny.en
datasets:
- openslr/librispeech_asr
tags:
- whisper
- quantization
- ternary
- bitnet
- quantization-aware-training
model-index:
- name: "{repo_name}"
  results:
  - task:
      type: automatic-speech-recognition
    dataset:
      name: LibriSpeech test-clean
      type: openslr/librispeech_asr
      config: clean
      split: test
    metrics:
    - type: wer
      value: 12.12
  - task:
      type: automatic-speech-recognition
    dataset:
      name: LibriSpeech test-other
      type: openslr/librispeech_asr
      config: other
      split: test
    metrics:
    - type: wer
      value: 28.42
---

# Whisper tiny.en with ternary weights {-1, 0, +1}

Every attention and feed-forward projection **and** the tied token embedding /
output projection of [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en)
are constrained to three values with one FP32 scale per output row, recovered by
quantization-aware fine-tuning. The file is **12.2 MB**, against 151 MB for the
FP32 original.

| Model | LibriSpeech test-clean WER | test-other WER | File |
| --- | ---: | ---: | ---: |
| Whisper tiny.en, FP32, zero-shot | 5.66% | 14.54% | 151.0 MB |
| Whisper tiny.en, FP32, fine-tuned with the same recipe (control) | 4.50% | 12.37% | 151.0 MB |
| Same ternary quantizer, no training | 100% | 100% | — |
| **This model** | **12.12%** | **28.42%** | **12.2 MB** |

WER is corpus-level with the Whisper English text normalizer on both sides,
greedy decoding, scored once per arm on the model rebuilt from this exact file.

Code, design document, full results and negative results:
[github.com/ajbarryiii/wilderness-labs-stt](https://github.com/ajbarryiii/wilderness-labs-stt/tree/main/finetune/whisper-ternary).

## Use

Needs `torch`, `transformers`, `safetensors` and, for the command line, `soundfile`.

```sh
hf download {repo_id} --local-dir whisper-ternary
python whisper-ternary/load_ternary.py clip.wav      # 16 kHz audio
```

```python
import sys; sys.path.insert(0, "whisper-ternary")
from load_ternary import load, transcribe
model, processor = load("whisper-ternary")
print(transcribe(model, processor, audio_16khz_float32))
```

The model is rebuilt in FP32 from the ternary codes, so it reproduces the
accuracy above on any hardware. It does not run on packed ternary kernels; that
needs activation quantization too, which this model does not have.

## What is in the file

- `export.safetensors`: 65 ternary weight matrices (36.4M parameters; the
  output projection shares the token embedding's matrix), each as 2-bit codes
  packed four per byte plus an FP32 scale per row, with the quantized layers'
  biases in FP32. Every other tensor (convolutions, positions, norms, other
  biases; 1.3M parameters) is FP16.
- `manifest.json`: format, layer shapes, tie map, byte accounting, code
  histogram, SHA-256 of the weights, and training provenance.
- `load_ternary.py`: the standalone loader and a small transcription CLI.
- Tokenizer and feature-extractor files from the base model, unchanged.

Code distribution: {code_histogram}.

## How it was trained

Starting from the pretrained checkpoint, keep FP32 latent weights and quantize
each row to `clamp(round(W / mean|W|), -1, 1)` on every forward pass with an
identity straight-through gradient. Ramp linearly from float to fully ternary
weights over the first 25% of training, then hold. Loss is 0.5 token
cross-entropy plus 0.5 KL divergence to the frozen FP32 model. 12,000 steps at
batch 32 on LibriSpeech train-clean-100 + train-clean-360 (464 h), AdamW, peak
learning rate 1e-3 chosen on dev-clean, BF16 autocast, one RTX 5090, about 19
minutes. The same quantizer with cross-entropy alone and no ramp reached only
50% test-clean WER.

## Limitations

- English read speech (LibriSpeech) only; not evaluated on conversational,
  accented, noisy field or domain-specific audio. test-other shows the gap
  widening on harder speech.
- Fine-tuned on lowercase, unpunctuated transcripts, so output is mostly
  lowercase with little punctuation.
- Occasionally keeps generating plausible text after the speech ends instead
  of stopping. Setting those utterances aside lowers WER by 0.5 points on
  test-clean and 0.7 on test-other (secondary diagnostic; the table above
  includes them).
- Activations are floating point. No speed or energy benefit is claimed for
  this checkpoint.

## License

Apache-2.0 (see `LICENSE`), following the base model. This is a modified
version of openai/whisper-tiny.en; see `NOTICE` for attribution and the list of
changes. Trained on LibriSpeech (CC BY 4.0, Panayotov et al., 2015).
