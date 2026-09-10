# Real-audio ternary speech-presence baseline

This pilot learns whether a complete English audio clip contains speech. It prepares the mandatory speech/no-speech stage, but it has no frame activity labels and does not establish streaming onset/offset behavior, medical speech accuracy, or power savings on iPhone15Pro. The final ASR target remains0.5–1Bparameters.

The prepared benchmark contains2,903 real clips from MiniLibriSpeech train-clean-5 and the MUSAN point-source noise subset distributed in RIRS_NOISES. Full3–20second transcript-bearing utterances are speech-positive; their silent frames are never given speech labels. Noise segments follow the observed speech-duration distribution. Speakers and original noise recordings are separated across training/calibration/development. Source URLs, license statements and every audio SHA256 are retained in data_manifest.json. Individual noise recordings have not been independently reviewed for residual speech contamination; corpus differences can inflate this proxy's apparent accuracy.

The fixed frontend computes16 logmel bands from20ms windows at10ms hops and averages4frames into40ms bins. The candidate has bias-free causal ternary convolutions, non-affine channel RMS normalization, ReLU, valid-length whole-clip mean/max pooling and a ternary head. All14,176 learned coefficients deploy as {-1,0,+1} codes, with four derivedFP32scale values. Training master weights useFP32 and CUDA BF16autocast. Evaluation expands the codes intoFP32 reference convolutions; it does not measure a packed ternary execution kernel. weights.2bit is an audited packed artifact, not a runtime speed claim.

prepare.py is the fixed evaluator and never imports candidate train.py or unpickles its training checkpoint. It validates the JSON inference graph, every ternary code and packed bytes, checks the cache checksum, selects a threshold on calibration speech for99%recall, and applies it unchanged to development clips. Constant outputs and non-discriminative/always-speech classifiers are rejected before ranking. Provisional feasibility requires developmentFNR<=1% andFPR<=20%; these small-corpus research thresholds are not field acceptance criteria. CPU timing includes one second of16kHz mono PCM through the frontend and model on one remote x86 thread.

Use the frozen runtime from the repository root:

```sh
custom/autoresearch/runtime-python custom/vad/prepare.py validate-data
custom/autoresearch/runtime-python -m unittest discover -s custom/tests -p test_vad.py -v
custom/autoresearch/runtime-python custom/vad/train.py --output custom/vad/artifacts/example --train-seconds 300 --seed 20260909
custom/autoresearch/runtime-python custom/vad/prepare.py evaluate --candidate custom/vad/artifacts/example --output custom/vad/artifacts/example/metrics.json
```

The training budget covers process startup and export; updates end20seconds early. Missing CUDA, zero updates, unchanged master weights, nonfinite loss or invalid frozen data cause a failure. There is no synthetic or CPU training fallback. Training artifacts retain the final model, packed export, CUDA training metrics and master-weight/optimizer checkpoint. Controlled research trials are launched through custom/autoresearch/control rather than these manual examples.
