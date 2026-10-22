# Kea-T evaluation

Detail for the [Kea-T model card](README.md).

## Recognition accuracy

Test WER (%), greedy decoding:

| Model | LS clean | LS other | AMI | Earnings-22 | GigaSpeech | SPGISpeech | VoxPopuli | Mean of 7 | Common Voice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original, NVIDIA published | 1.69 | 3.19 | 11.16 | 11.15 | 9.74 | 2.17 | 5.95 | 6.44 | n/a |
| Original, FP32, our scoring | 1.70 | 3.19 | 11.15 | 11.24 | 9.78 | 2.14 | 5.94 | 6.45 | 8.50 |
| Same ternary quantizer, no training | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| **This model** | 2.05 | 4.20 | 10.42 | 11.72 | 10.35 | 2.94 | 6.19 | 6.84 | 12.56 |
| Difference from our FP32 scoring (points) | +0.36 | +1.01 | -0.72 | +0.48 | +0.57 | +0.81 | +0.25 | +0.39 | +4.06 |
| Ratio to our FP32 scoring | 1.21x | 1.32x | 0.94x | 1.04x | 1.06x | 1.38x | 1.04x | 1.06x | 1.48x |

## Test sets and scoring

- **Test sets:** the Open ASR Leaderboard bundle
  [hf-audio/esb-datasets-test-only-sorted](https://huggingface.co/datasets/hf-audio/esb-datasets-test-only-sorted)
  (revision `b6bdcd0beb`, original, not `_cleaned`, configs), every utterance.
- **Mean of 7:** the unweighted mean over the eight sets NVIDIA reports for this
  model except TED-LIUM, which is not in the public bundle and was not evaluated.
  NVIDIA's published mean in that column is over the same seven sets (their
  TED-LIUM result, 3.38, is excluded).
- **Common Voice** is outside that mean and is reported separately. It shows the
  largest gap (+4.06 points); it is not part of the training mixture.
- **Scoring:** corpus-level WER with the Whisper English text normalizer on both
  sides (the leaderboard's normalizer); as on the leaderboard, utterances whose
  normalized reference is empty are decoded but not scored. Greedy TDT decoding
  with the model's own decoding configuration, in strict FP32 (no TF32, no
  autocast; the current leaderboard script uses BF16). Our scoring of the FP32
  original reproduces NVIDIA's published numbers within 0.09 points per set.
- **This model** was scored once, on the model rebuilt from this exact
  `export.safetensors` by the same procedure `load_ternary.py` implements.
  Differences and ratios are computed from unrounded WERs, so they can differ by
  0.01 from the difference of the rounded entries.

Code, design document, full results and per-utterance evaluation procedure:
[github.com/ajbarryiii/wilderness-labs-stt](https://github.com/ajbarryiii/wilderness-labs-stt/tree/main/finetune/parakeet-ternary).
The table above is
[results/TEST.md](https://github.com/ajbarryiii/wilderness-labs-stt/blob/main/finetune/parakeet-ternary/results/TEST.md)
in that repository.

## Limitations

- The gap to the original is largest on Common Voice (+4.06 points), which is
  outside the training mixture, followed by LibriSpeech test-other (+1.01) and
  SPGISpeech (+0.81). Expect larger degradation on domains far from
  the training data.
- AMI: 1.43% of this model's AMI test hypotheses are empty, fewer than the FP32
  original's 3.12%. The utterances empty only for this model are mostly one-word
  backchannels.
- Only greedy decoding was evaluated. Beam search and inputs longer than the test
  utterances were not evaluated. Timestamp accuracy was not evaluated either: the
  runtime check that compact and expanded modes emit identical timestamps shows
  that the two modes agree, not that the timestamps are correct. TED-LIUM was not
  evaluated.
- **Long audio:** each input is decoded in one pass with full attention, with no
  chunking; memory grows roughly quadratically with length. The model was trained
  on 1 to 30 s utterances and evaluated on test utterances up to about 106 s.
  Longer inputs were not evaluated; split long recordings into segments of about
  30 s or less.
- YODAS (training data) and GigaSpeech (a test set) both draw on YouTube, so some
  test audio may overlap the training audio. The original model was also trained
  on YODAS, so the same applies to its numbers.
- No FP32 model was fine-tuned on the same data as a control; the comparison is
  against the original model, so the effect of the training data and of
  quantization are not separated.
