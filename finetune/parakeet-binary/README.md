# Binary-weight QAT of Parakeet-TDT-0.6B-v2

**Published:** on Hugging Face as [Kea-B](https://huggingface.co/rajb3/kea-b)
(CC-BY-4.0). The upload folder is built by `hf/stage.py` (61 required checks) and
was verified from a fresh download before being made public
([plans/release.md](plans/release.md)).

All 264 encoder projection matrices (604M weights) quantized to {-1, +1} with
one FP32 scale per row; activations stay floating point. Trained from the
ternary model M1 (`../parakeet-ternary`) by ramping its ternary weights to
binary. The binary model is called MB1.

| Model | Weights | Export | Dev mean WER | Test mean WER |
| --- | --- | ---: | ---: | ---: |
| B0, NVIDIA original | FP32 | 2,472 MB | 5.58% | 6.45% |
| M1 | ternary, 2 bits | 180.8 MB | 6.09% | 6.84% |
| **MB1** | **binary, 1 bit** | **105.3 MB** | **6.31%** | **7.05%** |

The full dev set is LibriSpeech dev-clean and dev-other, AMI, VoxPopuli and
YODAS, and the mean is unweighted. Test is the seven-set Open
ASR Leaderboard mean (TED-LIUM unavailable), scored once
([results/TEST.md](results/TEST.md), [plans/test-scoring.md](plans/test-scoring.md)).
MB1 − M1 is +0.21 points on test (+0.07 to +0.45 per set; Common Voice, reported
apart, +0.86). On AMI MB1 beats B0 by 0.49. Dev per-set results:
[plans/main.md](plans/main.md) "Results". Of the 105.3 MB, codes are 75.5 MB,
row scales 1.8 MB and the 14M non-quantized parameters (FP16) 27.8 MB.

## Recipe

- **Quantizer:** per row α = mean|W|; c = +1 if W ≥ 0, else −1; forward
  weight c·α; straight-through estimator to the FP32 latent.
- **Ramp:** W_eff = lerp(T(W), B(W), f), where T and B are the ternary and
  binary quantizers. f goes from 0 to 1 over the first 25% of steps. At f = 0
  the model is bit-identical to M1, which is checked as a gate (6.084% vs 6.086%).
- **Training:** M1's recipe P2, which uses teacher transcripts plus
  encoder-output matching against B0 with the prediction and joint networks
  frozen.
  - 250k steps of 600 s batches, peak learning rate 2e-4, linear decay.
  - Hub stream with a new data-order seed, so MB1 does not replay M1's sequence.
  - 68.6 GPU-hours on one RTX 5090, 33k audio hours.
- **Export:** 1 bit per weight, LSB first, bit 1 = +1; rebuilt into plain NeMo
  with exact codes and scales; scored on the rebuilt model.

## What the pilot showed ([plans/pilot.md](plans/pilot.md))

All pilot arms ran 12k steps; results are full-dev means.

- **Starting from M1 beats starting from FP32** by 1.44 points (7.02% vs
  8.46%). Binarizing either one with no retraining is useless: FP32 scores
  100% and M1 83%.
- **The binary cost was 0.83 points at the pilot budget** (B2 − C).
  - C, the control, continued M1 as ternary for the same steps and tied with M1 (+0.10).
  - Learning rate 2e-4 beat 5e-4.
- **Extending to 250k steps cut the binary model's gap to M1 from 0.94 to
  0.23 points.** Through the learning-rate decay, MB1's curve tracks M1's
  within about 0.1 points.

## Files

| File | Role |
| --- | --- |
| `binary_quant.py` | `BinaryLinear` / `BinaryPointwiseConv1d`, ramps, code balance |
| `binary_train.py` | M1 initialization (pinned hash), ternary-to-binary ramp, decoder-mode pinning, reconstruction check; wraps `../parakeet-ternary/train.py` |
| `binary_export.py` | 1-bit export (`parakeet-binary-v1`) and FP32 NeMo rebuild |
| `binary_eval.py`, `binary_sweep.py` | zero-step points, gate, pilot arms, selection, main report |
| `binary_main.py`, `binary_main_paths.py` | MB1 entry point (stream seed, provenance) |
| `binary_test.py`, `binary_report_test.py` | one-time test scoring with precheck, dev gate, attempt ledger |
| `hf/` | Hugging Face release: `stage.py` builds and checks the upload folder, `load_binary.py` is the standalone loader (MIT), `verify_download.py` checks a fresh download; model card, LICENSE, NOTICE |
| `gpu1bit/` | native 1-bit RTX 5090 research runtime for MB1 (INT8x2 activations by default) and the M1 x2 loader: the code behind the recorded measurements, published for reference, not packaged for use elsewhere ([gpu1bit/README.md](gpu1bit/README.md); results in [gpu1bit/RESULTS.md](gpu1bit/RESULTS.md); plans [gpu-kernel.md](plans/gpu-kernel.md), [x2-default.md](plans/x2-default.md)) |
| `diagnostics/` | pre-pilot latent analysis |

Run everything with `./python`, which uses the parakeet-ternary runtime with
this directory first on the path. Long GPU jobs go through `./heavy`, which runs
them as memory-capped systemd units, one at a time. Artifacts live under
`/mnt/hd/wilderness-labs-stt/parakeet-binary/`.

```sh
./python -m unittest discover -s tests
./heavy binary-main --mem-max 36G --runtime 100h -- /mnt/hd/wilderness-labs-stt/parakeet-binary/launch-main.sh   # MB1; rerun to resume
./python binary_test.py --precheck   # then --dev-check, --test; see plans/test-scoring.md
```
