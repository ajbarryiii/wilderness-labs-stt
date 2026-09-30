# Results: protocol v2

Generated 2026-09-30 20:37 UTC. WER uses the Whisper English normalizer on both sides; see DESIGN.md. Training splits: train-clean-100, train-clean-360; 12000 steps.

| Arm | dev-clean WER | test-clean WER | test-other WER | Ternary params | Packed code bytes | Artifact bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0-fp32-zeroshot | 5.88% | 5.66% | 14.54% | — | — | — |
| A1-fp32-finetune | 4.69% | 4.50% | 12.37% | — | — | — |
| A2-ternary-ptq | 100.00% | 100.00% | 100.00% | 16515072 | 4.1 MB | 46.8 MB |
| A2-ternary | 11.79% | 13.12% | 30.90% | 16515072 | 4.1 MB | 46.8 MB |
| A3-ternary-embed | 11.40% | 12.12% | 28.42% | 36430848 | 9.1 MB | 12.2 MB |

## Learning-rate selection (full dev-clean WER)

| Arm | LR | dev-clean WER | best step | selected |
| --- | ---: | ---: | ---: | :-: |
| fp32 | 1e-5 | 4.78% | 11000 |  |
| fp32 | 3e-5 | 4.69% | 11000 | yes |
| ternary | 3e-4 | 13.17% | 12000 |  |
| ternary | 1e-3 | 11.79% | 11000 | yes |
