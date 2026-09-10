# Real-audio VAD autoresearch, 2026-09-09

Completed in 57m32s: 10 scored CUDA trials and one failed training attempt.
Retained trial 008 reduced development missed speech from 6/211 (2.84%) to
2/211 (0.95%); false positives stayed 0/212. It uses 14,176 ternary weights
(3,544 packed bytes plus 16 scale bytes). CPU p95 was 0.383 ms for the
one-second reference workload, using expanded FP32 weights.

Trial 010 scored 0/211 misses and 0/212 false positives, but the frozen
ranking rule did not reward further FNR improvement once feasible. Preserve
both 008 and 010 for pipeline comparisons; neither is independently validated.

This was a whole-clip development proxy, with no untouched final test,
continuous VAD validation, transcription measurement or phone power result.
The prior run 20260909T213516Z stopped without real GPU training; its synthetic
scores are invalid and must not be combined with these results.

This archive contains trial recipes, safe inference exports, metrics and artifact
hashes. Large optimizer states, runtime environments, cached audio and full GPU
telemetry remain in the ignored local runs directory. Absolute paths in historical
evidence describe the original host. Main-checkout train.py is not implicitly
replaced by a winning trial.
