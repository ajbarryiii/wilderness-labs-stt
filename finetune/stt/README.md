# Local pretrained STT reference

Whisper tiny.en is a working FP32 transcription reference for the paired gating
benchmark. This does not satisfy the custom track's all-ternary weight contract
and is not a finetuned model. Keep this reference separate from custom training
results; do not infer anything about the final 0.5–1B recognizer from its size.

Prepare the pinned Nix CUDA runtime described in `custom/autoresearch/runtime_data.md`,
then run `finetune/stt/setup` from the repository root. The installer adds hashed
Python wheels to an ignored local directory, preserving the existing Nix runtime.
Setup downloads a revision-pinned model and writes SHA256 hashes for every model
file. Transcription and evaluation require those local files and use offline mode.
No audio is uploaded.

```sh
custom/autoresearch/runtime-python custom/stt/pretrained.py transcribe \
  --audio /absolute/path/to/16khz.wav
custom/autoresearch/runtime-python custom/stt/evaluate.py \
  --backend pretrained --candidate finetune/stt/models/whisper-tiny.en \
  --output finetune/stt/runs/paired.json --limit 24 --repeats 3
```

The default device is one x86 CPU thread. Model loading is excluded from timing;
warm steady-state processing includes gate features, gate inference and actual
transcription. GPU operation is available via `--device cuda`, but CPU package
energy excludes GPU energy. Audio allocation/loading and microphone capture are
outside the measured processing region. Offline replay is not a phone battery test.

The model and license are documented in the [publisher's model card](https://huggingface.co/openai/whisper-tiny.en).
Generation is greedy and uses a fixed maximum of 256 new tokens per 20-second
chunk. Spelled numbers versus numerical formatting remain visible in literal WER;
this is intentionally not medical entity equivalence scoring.
