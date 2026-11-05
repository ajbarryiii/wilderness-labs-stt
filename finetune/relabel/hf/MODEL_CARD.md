---
license: cc-by-4.0
language:
- en
library_name: nemo
pipeline_tag: automatic-speech-recognition
base_model: {{kea_b0}}
base_model_relation: finetune
datasets:
{{datasets_yaml}}
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
- int8
- quantization-aware-training
- knowledge-distillation
{{model_index}}
---

# Kea-B 0.1

Kea-B 0.1 is an English speech recognizer in {{bundle_mb}} MB. It is NVIDIA's
[Parakeet-TDT-0.6B-v2](https://huggingface.co/{{base_id}}) with almost every weight stored as a single bit (+1 or
−1, with one scale per row) and its decoder in 8-bit integers. The original is {{original_mb}} MB.

Across the seven English test sets NVIDIA reports, it averages a {{keab_mean}}% word error rate, against
{{nvidia_mean}}% for the original. [Kea-B 0.0](https://huggingface.co/{{kea_b0}}) scored {{kea_b0_mean}}% at
{{kea_b0_mb}} MB, so 0.1 is both smaller and more accurate. It even edges out our 2-bit
[Kea-T](https://huggingface.co/{{kea_t}}) ({{kea_t_mean}}%) at about half its size.

![Model size against mean WER: Kea-B 0.1 is smaller and more accurate than Kea-B 0.0 and Kea-T](assets/kea-b-0.1-size-vs-wer.png)

## Results

Test word error rate (%), greedy decoding, lower is better:

{{card_table}}

Kea-B 0.1 is scored the same way as Kea-B 0.0 and Kea-T: weights rebuilt in FP32 from this repository. On our
INT8 × 1-bit GPU kernels, which it was trained for, the mean is {{keab_x1_mean}}% and no set moves by more than
{{x1_max_set_diff}} points. Kea-B 0.1 has trained on Common Voice audio (test speakers and sentences removed), so its Common Voice
score isn't comparable with the others. [EVALUATION.md](EVALUATION.md) has every number and how it was measured.

## What changed since 0.0

- **Better labels.** It learned from a stronger teacher: a full-precision Parakeet fine-tuned on transcripts from
  IBM's Granite Speech, instead of the original model's own output.
- **Trained for the kernels.** During training its activations go through the same INT8 rounding our GPU kernels
  use, so the accuracy holds when it runs on them.
- **An 8-bit decoder.** The prediction and joint networks moved from FP16 to INT8.
- **Harder audio.** Far-field meeting recordings, Common Voice and technical speech joined the training mix.

![Where the bytes go: the same 75.5 MB of 1-bit encoder codes in both releases; 0.1 stores the rest in INT8 and FP16](assets/kea-b-0.1-bytes.png)

## Where it's worse

- Spelled-out letters ("F-I-T-Z") are often joined into one word or dropped.
- SPGISpeech is {{spgi_delta}} points worse than Kea-B 0.0. Kea-B 0.1 writes down fillers and false starts that
  SPGISpeech's references clean up, and trailing "…" that the scorer counts as a word.
- It mixes up EBITDA and EBIT more often than a matched control model did.

## Use it

```sh
pip install "nemo_toolkit[asr]"
hf download {{repo_id}} --local-dir kea-b-0.1
python kea-b-0.1/load_kea_b.py clip.wav    # 16 kHz audio; prints the transcript
```

```python
import sys; sys.path.insert(0, "kea-b-0.1")
from load_kea_b import load, transcribe
model = load("kea-b-0.1", device="cuda")    # or "cpu"
print(transcribe(model, ["clip.wav"]))
```

`load_kea_b.py` checks every file against its manifest and rebuilds an ordinary FP32 NeMo model. You get the
accuracy above, but not the memory or speed savings: those come from running the 1-bit weights directly, as our
GPU kernels do.

## Files

| Path | Size | What |
| --- | ---: | --- |
| `bundle.safetensors`, `manifest.json`, `tokenizer/` | {{bundle_file_bytes}} bytes | The model: 1-bit codes, INT8 decoder, FP16 for the rest. |
| `compat/` | {{view_file_bytes}} bytes | The same weights in Kea-B 0.0's format, for tools that read it. `load("kea-b-0.1/compat")` gives the identical model. |
| `load_kea_b.py` | | Loader and command-line tool (MIT License). Reads either format. |
| `EVALUATION.md`, `TRAINING.md` | | Every result, the recipe and the training data. |

## More

The training and evaluation code is at {{github}} (`finetune/relabel`). Earlier releases:
[Kea-B 0.0](https://huggingface.co/{{kea_b0}}) and [Kea-T](https://huggingface.co/{{kea_t}}).

The weights and documents are licensed [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), like the base
model they modify, NVIDIA's [{{base_id}}](https://huggingface.co/{{base_id}}). `NOTICE` lists the changes, the
training data and the tools, with their licences.
