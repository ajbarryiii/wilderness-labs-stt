# Supervised GPU experiment status

Status: completed. Updated: 2026-09-09T23:39:20.781420+00:00.

Real-audio clip classification is a feasibility proxy; frame VAD, iPhone energy and a 0.5–1B recognizer remain unvalidated. Remote CPU timing is a reference only.

| Trial | Status | FNR | FPR | CPU p95 ms | Seconds |
|---|---|---:|---:|---:|---:|
| 000 | infeasible_best | 0.02843601895734597 | 0.0 | 0.3832841 | 282.175 |
| 001 | discard | 0.08056872037914692 | 0.0 | 0.37970204999999996 | 282.144 |
| 002 | error |  |  |  | 3.035 |
| 003 | discard | 0.04265402843601896 | 0.0 | 0.38284694999999996 | 282.16 |
| 004 | infeasible_best | 0.023696682464454975 | 0.0 | 0.37885615 | 283.142 |
| 005 | discard | 0.04739336492890995 | 0.0 | 0.3814455 | 283.148 |
| 006 | discard | 0.02843601895734597 | 0.0 | 0.38144175 | 282.163 |
| 007 | discard | 0.04265402843601896 | 0.0 | 0.3766775 | 282.131 |
| 008 | keep | 0.009478672985781991 | 0.0 | 0.3825341 | 283.128 |
| 009 | discard | 0.023696682464454975 | 0.0 | 0.37907905 | 282.159 |
| 010 | discard | 0.0 | 0.0 | 0.3867934499999999 | 283.114 |

Retained candidate: 008.

See per-trial checkpoints, independent metrics, train/evaluation logs and gpu.csv for evidence.
