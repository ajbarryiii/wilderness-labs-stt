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

# Kea-B 0.1: Parakeet-TDT-0.6B-v2 with 1-bit encoder weights and an 8-bit decoder

Draft card (the release text is written separately). Kea-B 0.1 stores NVIDIA's
[Parakeet-TDT-0.6B-v2](https://huggingface.co/{{base_id}}) encoder matrices at one bit per weight and its
decoder in 8 bits: {{bundle_mb}} MB of weights, against {{original_mb}} MB for the original `.nemo`. Its mean
word error rate on seven Open ASR Leaderboard test sets is {{keab_mean}}%, against {{nvidia_mean}}% published
by NVIDIA. It supersedes [Kea-B 0.0](https://huggingface.co/{{kea_b0}}) ({{kea_b0_mean}}%, {{kea_b0_mb}} MB)
and is also better than [Kea-T](https://huggingface.co/{{kea_t}}) ({{kea_t_mean}}%, {{kea_t_mb}} MB).

{{graphics}}

## Results

Test WER (%), greedy decoding:

{{card_table}}

Kea-B 0.1 is scored on the FP32 weights rebuilt from this repository (the procedure Kea-B 0.0 and Kea-T were
scored with). On our i8x1 GPU kernels, the deployed mode it was trained for, every set is within
{{x1_max_set_diff}} points of these numbers (mean {{keab_x1_mean}}%). Common Voice is shown apart: Kea-B 0.1
trained on Common Voice audio (test speakers and sentences removed), so its Common Voice number is not like for
like with the others. Details: [EVALUATION.md](EVALUATION.md).

## What changed since 0.0

- Better teacher labels and a fine-tuned FP32 teacher.
- Activation-aware training for the INT8 kernels.
- An 8-bit decoder (prediction and joint networks) and FP16 row scales.
- More data: far-field meetings, Common Voice, technical speech.

Details: [TRAINING.md](TRAINING.md).

## Known weaknesses

- Spelled-out letters.
- SPGISpeech is {{spgi_delta}} points worse than Kea-B 0.0 (verbatim style: fillers and false starts that the
  references clean up).
- EBITDA and EBIT.
- Common Voice is not like for like (see above).

## Use

```sh
pip install "nemo_toolkit[asr]"
hf download {{repo_id}} --local-dir kea-b-0.1
python kea-b-0.1/load_kea_b.py clip.wav    # 16 kHz audio; prints the transcript
```

```python
import sys; sys.path.insert(0, "kea-b-0.1")
from load_kea_b import load, transcribe
model = load("kea-b-0.1", device="cuda")    # or "cpu"; "kea-b-0.1/compat" loads the same weights
print(transcribe(model, ["clip.wav"]))
```

`load_kea_b.py` checks every file against `manifest.json` and rebuilds a dense FP32 NeMo model, so it saves
download size only, not memory or compute.

| Files | What |
| --- | --- |
| `bundle.safetensors`, `manifest.json`, `tokenizer/` | The 8-bit bundle ({{bundle_file_bytes}} bytes): 1-bit codes with FP16 row scales, INT8 decoder matrices with FP32 row scales, other weights in FP16. |
| `compat/` | The same weights in Kea-B 0.0's format ({{view_file_bytes}} bytes), for its loader and runtime. |
| `load_kea_b.py` | Loader and command-line tool (MIT License); reads either format. |
| `EVALUATION.md`, `TRAINING.md` | Results and method. |
| `LICENSE`, `NOTICE` | CC-BY-4.0; attribution, changes, data and licences. |

## Links

- [Kea-B 0.0](https://huggingface.co/{{kea_b0}}) and [Kea-T](https://huggingface.co/{{kea_t}})
- [Code]({{github}}/tree/main/finetune/relabel)

## License and attribution

The weights and documents are licensed [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (see `LICENSE`),
like the base model. Kea-B 0.1 is a modified version of [{{base_id}}](https://huggingface.co/{{base_id}}) by
NVIDIA; `NOTICE` lists the changes, the training data and the tools with their licences. `load_kea_b.py` is
MIT-licensed.
