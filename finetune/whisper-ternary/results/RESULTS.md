# Results: protocol v1

Generated 2026-09-30 18:07 UTC. WER uses the Whisper English normalizer on both sides; see DESIGN.md. Training splits: train-clean-100; 4000 steps.

| Arm | dev-clean WER | test-clean WER | test-other WER | Ternary params | Packed code bytes | Artifact bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0-fp32-zeroshot | 5.88% | 5.66% | 14.54% | — | — | — |
| A1-fp32-finetune | 5.31% | 5.41% | 12.95% | — | — | — |
| A2-ternary-ptq | 100.00% | 100.00% | 100.00% | 16515072 | 4.1 MB | 46.8 MB |
| A2-ternary | 50.29% | 50.46% | 71.86% | 16515072 | 4.1 MB | 46.8 MB |
| A3-ternary-embed | 76.48% | 76.31% | 87.58% | 36430848 | 9.1 MB | 12.2 MB |

## Learning-rate selection (full dev-clean WER)

| Arm | LR | dev-clean WER | best step | selected |
| --- | ---: | ---: | ---: | :-: |
| fp32 | 1e-5 | 5.31% | 3500 | yes |
| fp32 | 3e-5 | 5.60% | 1500 |  |
| fp32 | 1e-4 | 6.38% | 3500 |  |
| ternary | 5e-5 | 110.98% | 1500 |  |
| ternary | 1e-4 | 95.34% | 4000 |  |
| ternary | 3e-4 | 50.29% | 4000 | yes |
