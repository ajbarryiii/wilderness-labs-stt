# Test-set WER (%), Open ASR Leaderboard sets: B0 / M1 / MB1

Mean over the seven sets NVIDIA reports except TED-LIUM (not in the public bundle); Common Voice reported separately. B0 = FP32 original, M1 = ternary QAT main run, MB1 = binary QAT main run (final checkpoint, step 250,000, scored once on the model rebuilt from its 1-bit export), all decoded by the frozen evaluate.py at batch 64 with the same manifests and normalizer.

**MB1 - M1 is the net change from converting M1 to binary and training 250,000 more steps on a new data order, not a causal binary cost (plans/main.md "Reported").**

Export sizes: MB1 105.3 MB, M1 180.8 MB, original .nemo 2472 MB.

Dev cross-check (launch gate): mean 6.3120% vs recorded 6.3120% (+0.0000 points, tolerance 0.02; per set within 0.10): PASS.

| Arm | librispeech_clean | librispeech_other | ami | earnings22 | gigaspeech | spgispeech | voxpopuli | mean | common_voice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA published | 1.69 | 3.19 | 11.16 | 11.15 | 9.74 | 2.17 | 5.95 | 6.44 | n/a |
| B0 | 1.70 | 3.19 | 11.15 | 11.24 | 9.78 | 2.14 | 5.94 | 6.45 | 8.50 |
| M1 | 2.05 | 4.20 | 10.42 | 11.72 | 10.35 | 2.94 | 6.19 | 6.84 | 12.56 |
| MB1 | 2.20 | 4.56 | 10.65 | 12.17 | 10.46 | 3.01 | 6.30 | 7.05 | 13.42 |
| MB1 - M1 (points) | +0.15 | +0.36 | +0.23 | +0.45 | +0.11 | +0.07 | +0.11 | +0.21 | +0.86 |
| MB1 - M1 (relative) | +7.2% | +8.6% | +2.2% | +3.9% | +1.1% | +2.4% | +1.7% | +3.1% | +6.8% |
| MB1 - B0 (points) | +0.51 | +1.37 | -0.49 | +0.93 | +0.68 | +0.88 | +0.36 | +0.60 | +4.92 |
| MB1 - B0 (relative) | +29.8% | +42.8% | -4.4% | +8.3% | +7.0% | +40.9% | +6.1% | +9.4% | +57.9% |

## Empty hypotheses (fraction of scored utterances)

| Arm | librispeech_clean | librispeech_other | ami | earnings22 | gigaspeech | spgispeech | voxpopuli | common_voice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 0.00% | 0.00% | 3.12% | 0.73% | 0.35% | 0.00% | 0.00% | 0.15% |
| M1 | 0.00% | 0.00% | 1.43% | 0.07% | 0.13% | 0.00% | 0.00% | 0.14% |
| MB1 | 0.00% | 0.00% | 1.31% | 0.15% | 0.16% | 0.00% | 0.00% | 0.10% |
