# WP7 experiment plan: deployed-pipeline gates and the informational sweep of the new arms

For review before running (standing rule, 2026-10-03). DESIGN.md revision 8 is authoritative; this plan applies it.

## Status at the time of writing (runs made before the review rule arrived)

The rule reached the implementing agent after these runs had already been made. They are listed so the review can
decide whether to keep or redo them; nothing further runs until the review comes back clean.
- Pipeline gates (section 1): run in full, 2026-10-03 07:44–12:50 (Mac time). Records and summaries are committed
  (`results/eligibility/pipelines/`, `results/pipegates/`; commits e3db033, 53d455e, cb01896, fbddd44).
  147/147 records are timing-allowed. Code versions: `pipegate.py` CODE_VERSION wp7-pipegate-2 (the first
  smoke on C4 multi used wp7-pipegate-1 and the FP32-feature reference; that run was superseded).
- Sweep (section 2), first pass, 2026-10-03 09:25–10:36: all 7 arms. **Invalid for footprint:** component hashing
  in the eligibility check left 270–390 MB in phys_footprint before any model loaded (fixed in b8eac6d). Raw
  records stay on NixOS under `/mnt/hd/wilderness-labs-stt/parakeet-ios/results/wp5/<arm>/<run id>/`; no
  summaries committed.
- Sweep second pass (fixed binary), 12:58–13:49: C4-ane, C3-ane, C6s8-ane, C4 gpu, C3 gpu completed; C6s8 gpu was
  stopped by SIGTERM through macguard (status 130) when the rule arrived; the anchor did not run. Not reported.
  The stopped job's Core ML cache was not purged; the next guarded job purges it first.

## 1. Deployed-pipeline gates (review blocker 1)

**What is gated.** Pipeline = front end A (vDSP, FP32, `native/mp2` constants) → encoder arm → decode loop:
- F2: native FP32 Accelerate loop (`native/mp2` decoder_joint weights);
- F0: Core ML `Decoder` + `JointDecision` from `arms/mp2/decoder-fp32` (FP32, rev. 8);
- F1: Core ML `DecoderJoint` from the same directory.
Decoder models load with the encoder's compute units.

**Combinations.** Every revision-8 encoder record with `timing_allowed` true, except G0 (control with C0's
weights) and MLX (no Swift implementation), and except the `-dec-fp16` records (FP16 decoder; rev. 8 failed them).
49 encoder combinations × {F2, F0, F1} = 147 pipelines:

| arm | ANE (cpuAndNeuralEngine) | CPU (cpuOnly) | GPU (cpuAndGPU) |
|---|---|---|---|
| C1 | fixed, multi | fixed, multi | – (no GPU record) |
| C3 | fixed, multi | fixed, multi | multi |
| C4 | enum, fixed, multi | enum, fixed, multi | multi |
| C6s2, C6s4 | enum, fixed, multi | enum, fixed, multi | – |
| C6s8 | fixed, multi | fixed, multi | multi |
| C6d4, C6d8 | fixed, multi | fixed, multi | – |
| C7 | enum, fixed, multi | – (stress fails) | – (stress fails) |
| C8 | fixed, multi | – | – |
| C3-ane, C4-ane, C6s8-ane | multi | – | – |

**Clips.** All 82 clips of `clips.json` (64 natural: 16 per bucket 2/4/8/15 s; boundary-length, silence, impulse).
Each clip runs in its own bucket (fixed: the 15 s window).

**Procedure per combination** (`pipegate.py run`, NixOS orchestrates; Mac work in one macguard job):
1. Restore the encoder package if archived (`mil/archive.py back`), measure its size on the Mac.
2. `macguard --rss-cap CAP --timeout 5400 -- sh pipegate_job.sh NEED_GB OUT -- <gate args>`, OUT =
   `<artifacts>/results/pipegate/<combo>/<run id>/` (unique). The job: fail-closed disk check (30 GB floor + NEED_GB),
   purge `~/Library/Caches/parakeet-bench` (exit 6 if the purge fails), then
   `parakeet-bench gate --eligibility mp2:ARM --encoder <pkg> --encoder-variant V --compute-units U
   --frontend-constants native/mp2 --native-weights native/mp2 --decoder-models arms/mp2/decoder-fp32
   --decodes f2,f0,f1 --clips clips.json --pcm clips --traces traces.json --out OUT`:
   per clip, front end A → encoder once → `enc/<id>.f32` (+ SHA-256) → per decode loop: free decoding (tokens) and
   forced replay of the clip's B0 trace with the argmax token / duration of every step and the logical step counts.
   Header: components (SHA-256 + configuration) of every pipeline, clip ids, clips.json SHA-256.
3. `python -m pipegate evaluate --gate-dir OUT` on the Mac (refcache + FP64-feature reference are there); purge.
4. Retrieval of evaluation.json, gate.jsonl, job.log, cache.log into `<run id>.part`, atomic rename, `LATEST`.
5. `pipegate.py record` on NixOS: WER (parent experiment scorer and tokenizer), summary
   `results/pipegates/<combo>-vdsp.json`, records `results/eligibility/pipelines/<combo>-vdsp-<decode>.json`.
6. Archive arms the run restored once no later combination needs them.

**Reference.** FP32 reference encoder with FP16-rounded scales (WP3 refcache `fp16s`, the weights every exact arm
encodes) run on the FP64 evaluation of the reference front end (`pipegate.py ref64`, computed once on the Mac,
`results/pipegates/ref64_index.json`). Rationale: DESIGN.md rev. 5 gate 5 uses the FP64 evaluation because the FP32
reference front end is rounding noise on the silence clip (features vs FP64 rel 1.6e5; exact features 0, which
front end A returns). refcache vs ref64 encoder outputs: rel 0.34 on silence, ≤ 2.6e-6 on every other clip.
Decision references (raw logits, margins) and free-decoding reference tokens are WP3's refcache (forced replay of
the B0 trace, which is input-independent; free tokens from FP32 features — identical except possibly on silence,
which has no tokens). **Review question:** accept FP64 features as "the reference features" for this gate, or
require the FP32-feature reference (then all 49 combinations fail on the silence clip only, rel 0.32–0.33)?

**Pass criteria** (per pipeline; all must hold, DESIGN.md rev. 8 gate 4b):
- coverage: all 82 clips present; every trace replayed in full (replay steps = trace steps); all 64 natural clips
  free-decoded; clips.json unchanged; enc files match their recorded SHA-256; references not stale;
- encoder: rel ≤ 0.1 (tau 1e-3) on every clip, all finite, encoder length = clips.json;
- decisions, per head (token incl. blank; duration), pooled over clips: decisive steps (reference raw-logit top-1
  margin ≥ 1.0) ≥ 50% of steps; agreement with the reference argmax ≥ 99.5% on decisive steps and ≥ 99% on all;
- free decoding: ≥ 61/64 natural clips token-identical to the reference; WER ≤ reference WER + 0.2 points;
- decoder precision fp32; the arm's WP3 record still timing-allowed and revision 8.
Record: `timing_allowed` = all pass; `selection_eligible` also needs the WP3 record's.

**Resource caps.** macguard RSS cap 4G for ANE combinations, 6G for C1 and for CPU and GPU backends (decompressed /
GPU-visible model memory counts in RSS: C3 multi on the CPU and C3/C4/C6s8 on the GPU exceeded 4.19 GB with the
decoder models); timeout 5400 s; macguard's system-memory (abort below 25% free) and swap (1 GB growth) aborts;
lock refusals retried every 60 s up to 120 times. Disk: 30 GB floor + measured need (package + cache estimate:
4 × package for C1, else 1 GB, + 1 GB). One job at a time on the Mac (macguard lock). Expected 1–9 min per
combination on ANE/CPU, about 25 min for C6s8 on the GPU; about 5 h in total including restores.

**Commands.**
```
P=finetune/parakeet-ternary
# Mac, once: ios/macguard --rss-cap 6G --timeout 1800 -- ios/pyenv/.venv/bin/python -m pipegate ref64   (from ios/)
CUDA_VISIBLE_DEVICES= $P/python $P/ios/pipegate.py run [--only COMBO,...]
CUDA_VISIBLE_DEVICES= $P/python $P/ios/pipegate.py table
```

## 2. Informational paired sweep of the new arms (WP7 item 4)

**Arms** (all front end A + F2, multifunction, mp2):
- ANE layout on cpuAndNeuralEngine: C4-ane, C3-ane, C6s8-ane;
- GPU backend (cpuAndGPU): C4, C3, C6s8;
- within-session anchor: C6s8 plain on cpuAndNeuralEngine (re-timed; compare with WP5's C6s8 row).
Each needs its pipeline record `<…>-vdsp-f2.json` (section 1); the runner refuses otherwise.

**Protocol** (WP5's): 64 natural clips (16 per bucket), mode free, 3 warm-up + 10 timed calls per clip and arm.
Paired in one process with C0 (`--pair-c0 c0 --c0-out … --c0-compute-units cpuAndNeuralEngine`: C0 always runs as
shipped, on the ANE, whatever the arm's backend), clip by clip; C0 first on even clip indices, the arm first on odd.
Per arm one macguard job (`sweep_job.sh`): fail-closed disk check, purge the binary's Core ML cache, run 1 = the
paired sweep (its loads are "post-purge loads"), run 2 = a fresh process loading the arm alone with one call on
`n15-2412-153947-0005` ("subsequent fresh-process load", arm-only phys_footprint), purge.
Order: C4-ane, C3-ane, C6s8-ane, C4 gpu, C3 gpu, C6s8 gpu, anchor. Unique run dirs on both machines, atomic publish,
`LATEST`; the report reads only `LATEST` runs with status 0.

**Outputs.** Raw: `/mnt/hd/wilderness-labs-stt/parakeet-ios/results/wp5/<arm>/<run id>/` (arm.jsonl, c0.jsonl,
cached.jsonl, cache.log, job.log, STATUS). Summaries: `ios/results/wp7/<arm>.summary.json`,
`<arm>.c0block.summary.json`, `sweep.json`, table `sweep_table.md` (git-ignored; reproduced in the README).
Per arm and bucket: arm/C0 typical total-latency ratio (median over clips of per-clip median ratios) with a 95%
percentile bootstrap CI (2,000 clip resamples, pairs kept together, seed 0); HD p95 ratio; encoder, preprocess,
decode stage medians; post-purge and subsequent fresh-process load times; arm-only footprint peak; WER and token
identity to mp2's reference; physical and logical call counts.

**Validity checks** (armreport, fail = not reported): baseline is C0 with the identical pairing block (per-process
session id), clips.json SHA-256, clip ids, warm-ups/timed and mode; complete calls for every clip; ≥ 3 warm-ups and
≥ 10 timed for "baseline-eligible". No pass/fail on latency: **informational, shared Mac, no claims.**

**Resource caps.** RSS cap 4G (ANE arms), 6G (GPU arms); guard timeout 3600 s (ANE), 7200 s (GPU: C6s8's GPU load
is about 310 s per function); same macguard memory/swap aborts and lock retries; disk as in section 1. Expected
4–11 min per ANE/GPU arm, about 30 min for C6s8 on the GPU, about 75 min in total.

**Commands.**
```
CUDA_VISIBLE_DEVICES= $P/python $P/ios/wp5sweep.py plan
CUDA_VISIBLE_DEVICES= $P/python $P/ios/wp5sweep.py run --only C4-ane-multi-vdsp-f2,C3-ane-multi-vdsp-f2,\
C6s8-ane-multi-vdsp-f2,C4-multi-gpu-vdsp-f2,C3-multi-gpu-vdsp-f2,C6s8-multi-gpu-vdsp-f2,C6s8-multi-vdsp-f2-anchor-wp7
CUDA_VISIBLE_DEVICES= $P/python $P/ios/wp5sweep.py report --groups c --tag wp7
```
