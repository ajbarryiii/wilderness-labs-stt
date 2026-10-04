# Test-set WER (%), Open ASR Leaderboard sets

Sanity flag `ami: 1.43% empty M1 hypotheses`: investigated and closed. The FP32 original leaves more AMI hypotheses empty (3.12%) than M1 (1.43%); AMI is full of sub-second backchannels, and M1 has fewer or equal empty outputs than B0 on every set.

Mean over the seven sets NVIDIA reports except TED-LIUM (not in the public bundle); Common Voice reported separately. B0 = FP32 original, B1 = ternary PTQ without training, M1 = ternary QAT main run (final checkpoint, scored on the rebuilt export). M1 export 180.8 MB versus 2472 MB for the original .nemo file.

B0 was scored on 2026-09-30 by an earlier revision of evaluate.py that predates the 30 ms padding rule and recorded no evaluator hash; the rule cannot affect any test set (shortest test utterance 40 ms) and the saved decoding and normalizer metadata match.

| Arm | librispeech_clean | librispeech_other | ami | earnings22 | gigaspeech | spgispeech | voxpopuli | mean | common_voice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA published | 1.69 | 3.19 | 11.16 | 11.15 | 9.74 | 2.17 | 5.95 | 6.44 | n/a |
| B0 | 1.70 | 3.19 | 11.15 | 11.24 | 9.78 | 2.14 | 5.94 | 6.45 | 8.50 |
| B1 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| M1 | 2.05 | 4.20 | 10.42 | 11.72 | 10.35 | 2.94 | 6.19 | 6.84 | 12.56 |
| M1 - B0 (points) | +0.36 | +1.01 | -0.72 | +0.48 | +0.57 | +0.81 | +0.25 | +0.39 | +4.06 |
| M1 / B0 (relative) | 1.21x | 1.32x | 0.94x | 1.04x | 1.06x | 1.38x | 1.04x | 1.06x | 1.48x |
