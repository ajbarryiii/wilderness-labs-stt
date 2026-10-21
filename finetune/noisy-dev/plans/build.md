# Plan: a fixed noisy copy of the development sets (robustness evaluation)

**Status: 2026-10-16: Codex review CLEAN in round 6 (plan + code; reviews in
`/mnt/hd/wilderness-labs-stt/noisy-dev/review/`), 160 unit tests pass. A first
real-noise sample (`samples/sample-50-r5`) passed. Ready to build once MUSAN and
BUT ReverbDB finish downloading; listening check before any scoring.**

## Purpose

The five development sets used for selection (LibriSpeech dev-clean and
dev-other, AMI IHM validation, VoxPopuli English validation, 2,000 reserved
YODAS utterances) are mostly clean speech. The planned augmentation follow-up
to the binary pilot (`finetune/parakeet-binary/plans/pilot.md`, "Augmentation
and more data") needs a measurement that can see a robustness gain. The
deployment target is a voice-operated TCCC manual used by Pararescue in the
field, so the conditions are gunfire and explosions, rotor, engine and vehicle
noise, wind, babble, crowd and shouting, reverberant rooms, and radio/intercom
channels.

This plan builds that measurement once, deterministically, before any
augmentation is trained. It is evaluation only and selects nothing in the
binary pilot. It also reserves the EVALUATION pool of noise and impulse
response sources. A later augmentation plan must draw its TRAIN pool from
sources disjoint from it, enforced by a check that fails on any intersection.

## Utterances

- **The full development sets**, not the 400-utterance monitoring subsets.
  The subsets are strided slices of sorted populations (`data.every_kth`) that
  truncate speakers and date ranges, and they hold about 36,000 reference
  words. Full-dev scoring takes about 80 seconds per model on the RTX 5090
  (G-B and Z-B in the binary pilot), so 16 conditions cost about 25 minutes
  per model.
- References, normalizer, scored denominators and exclusion semantics are
  exactly those of `parakeet-ternary` `evaluate.py` (Whisper English
  normalizer, corpus-level WER; rows whose normalized reference is empty are
  excluded as they are now). A noisy utterance keeps its clean reference.
- The clean input audio is frozen: the build lock records the SHA-256 of every
  input utterance's decoded PCM and of the dev manifests.

## Evaluation-pool sources (reserved whole, not only the files sampled)

Licence rule: only sources whose licence permits this use (CC0, CC BY, public
domain), verified against the primary source. Sources with NC terms or an
unverified licence are excluded. Exact licence versions and original
attribution metadata (Freesound uploader, sound ID, licence URL) are kept in
the manifest.

| Category | Source (reserved unit) | Licence (primary source) | Notes |
| --- | --- | --- | --- |
| Gunfire and explosions | FSD50K **eval split**: classes `Gunshot_and_gunfire`, `Explosion`; per-clip CC0 or CC BY only | CC0 / CC BY 4.0 per clip (FSD50K metadata) | FSD50K's eval split is uploader-disjoint from its dev split. Reserved: every eval-split clip and every eval-split uploader |
| Gunfire (field) | Zenodo 3997406, open-field gunshots, all four microphones | CC0 (Zenodo record) | Reserved whole |
| Engine, vehicle, aircraft | FSD50K eval: `Engine`, `Vehicle`, `Car`, `Truck`, `Aircraft`, `Fixed-wing_aircraft` (CC0 / CC BY clips) | as above | |
| Rotor / helicopter | **Freesound API, CC0 clips only, split by uploader**, if the user provides an API key. Otherwise rotor robustness is reported as **unmeasured**; the engine condition does not stand in for it | CC0 | Separate `rotor` conditions; never mixed into `eng` |
| Wind | FSD50K eval: `Wind` | as above | |
| Crowd, shouting | FSD50K eval: `Crowd`, `Chatter`, `Shout`, `Yell` | as above | May contain intelligible speech; reported separately from babble |
| Ambient environments | DEMAND, all 16 kHz environments | Zenodo lists both CC BY-SA 3.0 (description) and CC BY 4.0 (rights field); treated as **CC BY-SA 3.0**, the stricter | Reserved whole (all 16 channels). Any DNS-Challenge subset used for TRAIN must drop its DEMAND part |
| Babble talkers | MUSAN `speech/us-gov` (US government hearings) | Public domain (MUSAN paper; MUSAN CC BY 4.0) | Reserved whole. Not LibriVox, so no LibriSpeech speaker or text overlap; see "Overlap audit" |
| Reverberation | BUT ReverbDB RIR-only release (`BUT_ReverbDB_rel_19_06_RIR-Only.tgz`), loudspeaker-to-microphone RIRs, 9 rooms | CC BY 4.0 (BUT page) | Reserved by room. Never the LibriSpeech-retransmission set |
| Radio channel | Codec2 3200 bit/s, nixpkgs `codec2`, pinned store path | LGPL-2.1 (tool) | A channel surrogate, not validated against fielded radios |

Excluded:
- ESC-50, UrbanSound8K, WHAM!/WHAMR!, NOISEX-92: NC or unclear licences.
- The MIT IR Survey: licence stated only on a mirror. IR64-CAR: it measures an
  external loudspeaker through a closed car, not an in-cabin talker.
- AudioSet and anything YouTube-sourced (DNS AudioSet part, MAD): no rights in
  the audio, and overlap risk with YODAS.
- Sonniss: bans AI use. SESA: unverifiable per-clip licences. MUSAN LibriVox
  speech. RWCP: R&D use only.

Downloads go to `/mnt/hd/wilderness-labs-stt/noisy-dev/raw/<source>/`, each
in its own directory, and are untrusted data. Nothing in them is executed;
archives are extracted into a separate directory with path-traversal checks
and read only by path with `python -I`. Every download is verified against
the publisher's checksum (Zenodo's per-file MD5 from its API; OpenSLR's
published MD5 for MUSAN), and the build refuses a source whose verification
record is missing. `.done` markers are only resume hints. BUT ReverbDB
publishes no checksum. It is downloaded over HTTPS only (TLS-authenticated
transport), its exact size is pinned from the server's ETag (9,308,593,693
bytes, Last-Modified 2019-06-05), and its SHA-256 is pinned on first download.
That is trust on first use over authenticated transport, accepted in review
round 2 because no independent digest exists. A partial download fetched
earlier over plain HTTP was deleted and restarted over HTTPS.

## Conditions (each applied to every utterance of the five full dev sets)

| ID | Condition | Definition |
| --- | --- | --- |
| clean | none | reference row |
| gun0, gun10 | impulsive: gunfire, explosions | events at event-to-speech ratio 0 dB and +10 dB (see "Impulsive events") |
| eng10, eng0 | engine, vehicle, aircraft | added-noise ratio 10 and 0 dB, continuous |
| rotor10, rotor0 | helicopter (only if licensed rotor material exists) | 10 and 0 dB, continuous |
| wind10, wind0 | wind | 10 and 0 dB |
| bab10, bab0 | synthetic babble (MUSAN us-gov) | 10 and 0 dB |
| crowd10, crowd0 | crowd and shouting (FSD50K) | 10 and 0 dB |
| amb10, amb0 | DEMAND ambient | 10 and 0 dB |
| rev | reverberation | one BUT RIR per utterance |
| radio | radio channel | see "Radio channel" |
| field | compound | rev, then engine at 5 dB measured against the reverberated speech, then radio |

That is 18 conditions with rotor and 16 without. At about 25 audio hours of
dev audio per condition, about 400-450 hours of audio, about 50 GB as 16 kHz
24-bit FLAC (review round 2, finding 8).

**Added-noise ratio (continuous noise).** 10·log10 of the clean utterance's
RMS power over its whole length divided by the noise segment's RMS power over
the same length. This is a ratio of added signals, not a measured total
acoustic SNR. Utterance-level RMS is used instead of a speech-activity gate so
the definition needs no estimator. Its corpus dependence is reported, not
corrected: at a fixed ratio, an utterance with long pauses has a lower
utterance RMS than its active speech, so its noise is *quieter* relative to
the speech while it is active (AMI is the most affected). the build reports, per set, the active-frame fraction from a
-30 dB-relative energy gate as a diagnostic. Utterances shorter than 0.3 s or
with RMS below -70 dBFS keep their clean audio in noise conditions and are
flagged; their count is reported.

**Continuous noise segments.** A noise source shorter than the utterance is
looped with a 50 ms raised-cosine crossfade. The offset is drawn uniformly. If
a multi-source category is sampled, the source is drawn uniformly over the
sorted, reserved pool.

**Impulsive events (gun0, gun10).**
- Event windows: each FSD50K or Zenodo clip is cut into events at frames
  within 30 dB of the clip's peak frame (20 ms frames), merging gaps under
  100 ms. Events shorter than 30 ms or longer than 3 s are dropped. The event
  list is frozen in the build lock.
- Count: one event per started 3 s of utterance, at most 6, so the rate does
  not depend on length.
- Placement: event onsets are drawn uniformly over the utterance's active
  frames. Events never extend past the utterance end; an event that would is
  shifted earlier, and if it cannot fit it is truncated with a 10 ms fade.
- Level: each event's RMS over its own window is set to the clean utterance
  RMS plus the condition's event-to-speech ratio (0 or +10 dB), so loudness
  does not depend on utterance length or event count.
- Reported: events per utterance, overlap fraction with active speech frames,
  peak and window levels, and the resulting utterance-level added-noise ratio.

**Babble.** Five segments from five different MUSAN us-gov files (distinct
hearings), each with an active-frame fraction of at least 60%. Each segment is
normalized to equal RMS before summing, and the sum is then set to the
condition's ratio. The target is always the dev-set utterance. At 0 dB each
babble talker is about 7 dB below the target. Distinct files do not guarantee
distinct speakers; that is accepted and stated.

**Reverberation.**
- RIR channel: one microphone channel per utterance, chosen by seed.
- Onset: the first sample whose magnitude exceeds 10% of the RIR's maximum.
  The RIR is shifted so that sample is at index 0; anything earlier is dropped.
- Normalization: the full tail is kept. The convolution output is truncated
  to the input length plus 0.5 s, so final words keep their reverberant tail,
  and scaled to the input's RMS.
- Tests: an identity (delta) RIR and a delayed delta reproduce the input
  exactly. The energy of the last 200 ms of the input region is preserved
  within 1 dB for a synthetic short RIR.

**Radio channel.**
- Input: speech scaled to -20 dBFS RMS, band-pass 300-3400 Hz (8th-order
  Butterworth, zero-phase), resampled to 8 kHz.
- Codec2: padded with zeros to a whole number of frames plus four flush
  frames, encoded with Codec2 3200 and decoded. The codec delay is measured
  once by cross-correlation on a fixed test signal, recorded in the lock, and
  compensated; output is trimmed to the input length.
- Clipping: soft clipping `tanh(2x)/tanh(2)` at 8 kHz, then the same band-pass
  again, then resampling to 16 kHz, then rescaling to the input's original RMS.
  The bandwidth contract (energy above 3.6 kHz at least 40 dB below the
  passband) is tested on the final output.
- Tests: on synthetic input the last 100 ms is not truncated, and the delay
  compensation aligns within 1 ms.

**Mixture headroom.** After mixing, a mixture with a peak above 0.99 is scaled
down as a whole. The radio and field conditions' clipping is the only intended
nonlinearity.

## Determinism and the build lock

`build.lock.json` freezes everything an output depends on:
- dev manifests and input-audio PCM hashes;
- source verification records;
- the sorted, reserved pools with their per-file decoded-PCM SHA-256 and the
  event list;
- every processing parameter and the measured codec delay;
- library versions (numpy, scipy, soundfile, libsndfile, torch) and the
  Codec2 store path.

The RNG is numpy `Generator(PCG64)` seeded from SHA-256(utterance ID,
condition ID, build seed 20261016), so adding a condition (e.g. rotor) changes
no other condition's files. The manifest records, per output, the source files
and offsets, gains, measured ratios, RIR ID and channel, and the SHA-256 of the
written FLAC.

Tests: a rebuild reproduces byte-identical FLACs and manifest rows with 1 and
with 4 workers, and after an interrupted build resumed from its manifest.

## Leakage controls

- **Reserved units, not sampled files.** The blocklist
  (`eval_pool_blocklist.json`) lists:
  - whole datasets and campaigns (Zenodo 3997406, DEMAND, MUSAN us-gov,
    BUT ReverbDB by room);
  - for FSD50K, every eval-split sound ID and every eval-split uploader;
  - the decoded-PCM SHA-256 of every file in the reserved pool.
- **TRAIN check.** `check_train_pool(manifest)` fails if a candidate TRAIN
  source matches any reserved dataset, campaign, room, Freesound sound ID or
  uploader, or any reserved PCM hash. A unit test shows it fails on each kind.
  Re-encoded or edited derivatives that change the PCM are not caught. That is
  an accepted limitation, mitigated by blocking at the dataset and uploader
  level.
- **Scope of the TRAIN check (decided after review round 5).** This build's
  leakage deliverable is the blocklist: the reserved units, IDs, uploaders and
  PCM hashes, which is data. `check_train_pool` is a provisional helper. Four
  review rounds kept finding new provenance shapes it mishandled (wrapper
  collections, explicit origins, mixed subsets), and every one of them depends
  on how a future TRAIN manifest describes its sources, which no plan has
  fixed yet. The authoritative TRAIN-pool check is therefore specified,
  implemented and reviewed together with the augmentation plan, against that
  plan's actual TRAIN sources and manifest schema. Until then, no TRAIN pool
  may be built, and the helper must not be cited as evidence of disjointness.
  Its current behaviour stays tested (it fails closed on everything shown so
  far), but it is out of this plan's acceptance criteria.
- **Threat model.** The integrity checks defend against corrupted or malicious
  third-party downloads and against accidental changes to local artifacts:
  stale caches, partial builds, edited files, a sample mistaken for a full
  build. Deliberate tampering with local files by someone with write access to
  `/mnt/hd` is out of scope. Where a cheap check also covers it, it is kept.
- **Speech overlap: not audited (accepted, review round 2).** Babble uses US
  government hearings, a source independent by provenance of LibriVox, AMI,
  European Parliament (VoxPopuli) and the reserved YODAS shards. No audio or
  transcript audit is done. Hearings could appear in web corpora, including
  YouTube and hence YODAS. If a babble segment contained the same speech as a
  YODAS target, its words would be present in the interference. This is
  recorded as an unaudited risk. It is judged small (2,000 YODAS dev utterances
  against US hearing recordings) but is not excluded. YODAS babble rows are
  already labeled pseudo-reference. The five dev sets remain excluded from any
  future TRAIN pool, as they are now.

## Scoring (after the build, GPU)

- **Models:** B0 (FP32 original), M1 (ternary), and every binary pilot arm's
  final export. Each is scored on the model rebuilt from its export, with
  greedy TDT decoding pinned to M1's decoder mode (`no_while_loops`, as in
  `parakeet-binary`), under the GPU lock.
- **Per condition:** WER per dev set, and the equal-weight mean of the five
  set WERs, the existing selection metric's formula. YODAS references are
  machine-generated, so YODAS numbers are labeled "pseudo-reference agreement".
  A human-reference-only four-set mean is reported as a sensitivity result.
- **Aggregate:** the equal-weight mean over the noisy conditions ("noisy
  mean"), each model's degradation against its own clean row, and the gap to
  B0 per condition.
- **Uncertainty:** paired cluster bootstrap, 2,000 resamples, stratified by
  dev set. Clusters are speakers (LibriSpeech, VoxPopuli), meetings (AMI) and
  videos (YODAS), resampled jointly across all conditions and both models of
  each comparison, so the pairing is preserved. It reports 95% intervals for
  every model difference. No precision is promised in advance; differences
  whose interval includes zero are reported as unresolved.
- **Scope of the intervals (accepted, review round 2).** Each condition uses one
  frozen realization of the noise and RIR sources: fixed DEMAND environments,
  hearings, FSD50K clips and BUT rooms. Those are not resampled, so the
  intervals cover speech sampling only. Every reported difference is stated
  as conditional on this noise realization, not as a generalization across
  noise recordings. The report also lists, per condition, the WER difference
  broken down by noise source group (FSD50K class, DEMAND environment, BUT
  room) so a result driven by one source is visible.

## Running alongside the GPU pilot

The user asked for this to run concurrently with the binary pilot.

- **Network:** total download bandwidth is capped at 16 MiB/s aggregate (not
  per connection). A watchdog in `download.sh` reads the pilot's latest
  `metrics.jsonl` entry every 60 s. It suspends all transfers (SIGSTOP) while
  `data_wait_s_per_step` exceeds 0.05 s and resumes them below 0.02 s.
- **Memory:** download unit `--mem-max 1G`, build unit `--mem-max 6G`. With
  the pilot's measured peak of 25.7 GB that leaves about 28 GB for the desktop
  and T3. The caps sum to 43 GB, against the 36 GB the 24 GB-headroom rule
  allows. That risk is accepted: all units are first OOM victims
  (OOMScoreAdjust 500) under systemd-oomd, so an overrun kills a job, not the
  desktop.
- **CPU and I/O:** the build runs `nice -n 19` with 4 worker processes,
  `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`, at the units' IOWeight 50.
  The pilot's step time is checked after the build starts. If it rises by more
  than 5%, the build is reduced to 2 workers.

## Code and tests

Code lives in `finetune/noisy-dev/`; artifacts in
`/mnt/hd/wilderness-labs-stt/noisy-dev/`. The contract tests are written first:
1. Determinism: byte-identical FLAC and manifest across rebuilds, with 1 and
   4 workers, and after an interrupted resume.
2. Added-noise ratio within 0.1 dB of target (synthetic fixtures). Event
   window levels within 0.1 dB of target. Event count and placement follow
   the rules.
3. Licence filter: only CC0 or CC BY clips from the FSD50K eval split enter
   the pool. No NC or unverified source can enter any pool.
4. Download verification: a source with a missing or mismatching checksum is
   refused.
5. The blocklist covers every reserved unit. (The provisional `check_train_pool`
   helper's tests stay, but they are not acceptance criteria; see "Scope of the
   TRAIN check".)
6. Reverberation identity, delayed-identity and tail preservation (above).
7. Radio: final-output bandwidth contract, delay alignment within 1 ms, and no
   truncation of the last 100 ms.
8. Each condition has exactly the full dev sets' rows, IDs and clean
   references. Short or silent utterances are flagged and kept clean.
9. Adding the rotor conditions does not change any other condition's files.
10. Scoring aggregation: equal-weight five-set mean, the four-set
    sensitivity mean, and paired cluster-bootstrap intervals that preserve
    pairing (checked on stubs with known answers).

## Procedure

1. Downloads (running, rate-limited, eval-pool sources only).
2. Implementation against the contract tests (Opus subagent, in progress).
3. Codex review of this plan and the code, iterated until clean.
4. Build (CPU, concurrent with the pilot). Record sizes, ratio and event
   statistics, active-frame fractions, and the blocklist.
5. Listen check: two samples per condition, saved to a listening folder for
   the user. Nothing is scored until the user has had the chance to listen.
6. Score B0 and M1 when the GPU is free, then the binary arms.

## Review round 1 responses (2026-10-16)

Codex plan review round 1 (`/mnt/hd/wilderness-labs-stt/noisy-dev/review/plan-r1.md`):
1. Precision (major): full dev sets instead of subsets, plus paired cluster
   bootstrap intervals; no advance precision claim.
2. Blocklist (major): whole units reserved; uploader, room and PCM-hash
   blocking; a failing TRAIN check with tests.
3. Speech overlap (major): the audit is documented; the remaining overlap risk
   is stated as unresolved and low impact.
4. Licences (major): ESC-50 and the MIT IR Survey excluded. Rotor material
   must be CC0 Freesound, or rotor is reported as unmeasured. DEMAND is
   treated as BY-SA 3.0.
5. Levels (major): utterance-RMS added-noise ratio, labeled as such, with an
   active-frame diagnostic. Short and silent inputs handled. Radio input
   calibrated to -20 dBFS. The field ratio is measured against the
   reverberated speech.
6. Impulsive (major): event windows, a rate-based count, placement and
   truncation rules, per-event levels, and overlap reporting.
7. Babble (major): recipe specified; babble and crowd reported separately.
8. Reverb (major): IR64-CAR dropped; alignment, tail and normalization
   defined, with tests.
9. Radio (major): padding, flush, delay compensation, clipping placement and
   a final-output bandwidth test. Labeled as a surrogate.
10. Reproducibility (major): build lock, checksum verification, PCG64, and
    worker-count and resume tests.
11. Concurrency (major): aggregate bandwidth cap with a data-wait watchdog,
    nice and thread limits, memory caps; the cap-sum risk is accepted
    explicitly.
12. Subset bias (minor): full dev sets.
13. Aggregation (minor): equal-weight five-set mean, existing exclusions,
    YODAS labeled, four-set sensitivity mean.
14. Rotor (minor): separate conditions or explicitly unmeasured.

## Out of scope

Training with augmentation (a later plan), building the TRAIN pool, test sets
in noise, and redistributing the noisy audio.

## Implementation notes (2026-10-16, implementation subagent)

Ambiguities resolved in the code (`noisy_*.py`); each is a candidate for review.

- **Interpreter.** `./python` runs with `-P -s` instead of `python -I`: `-I` would also
  drop `PYTHONPATH`, which the build needs (parakeet-ternary, parakeet-binary). `-P` keeps
  the script and working directories off `sys.path`, and no process starts inside a data
  directory. As in `parakeet-binary/python`, there is no NVRTC. Threads default to 1.
- **Downloads.** `noisy_sources.py verify` does not read a raw file until its `.done`
  marker exists. The marker is only a precondition, never evidence. The publisher MD5s
  are pinned in `noisy_sources.SOURCES` (fetched 2026-10-16) and the build checks them.
  `require_verified` also refuses a file whose size or mtime changed after verification.
- **Short/silent rule.** It applies to every processed condition (rev, radio and field
  too), not only to the noise conditions.
- **Continuous noise.** The source is drawn uniformly over the sorted category pool. The
  offset is uniform over `[0, frames - n]`, or over `[0, frames)` into the looped stream
  when the clip is shorter than the utterance. The crossfade is an equal-gain raised
  cosine. A segment below -70 dBFS is redrawn, at most 10 times.
- **FSD50K.**
  - Labels are "smeared" up the ontology, so `Explosion` also admits `Fireworks`.
  - A clip whose labels fall in two categories is excluded.
  - A gun, engine or wind clip that carries a voice label (`Human_voice`, `Speech` and
    others) is excluded.
  - A continuous clip shorter than 0.5 s, or quieter than -70 dBFS, is excluded.
  - The per-clip licence must be CC0 1.0 or CC BY (any version). NC, Sampling+ and a
    missing licence are excluded.
  - All of this applies only to clips listed in `eval.csv`. Every eval-split sound ID,
    uploader and PCM hash is reserved.
- **DEMAND.** The pool is all 16 channels of the 17 environments that have a 16 kHz
  release (SCAFE has 48 kHz only): 272 files, drawn uniformly.
- **Events.**
  - The event is drawn uniformly over the frozen event list, with FSD50K and Zenodo
    events pooled.
  - The onset is uniform over the samples of the active frames: a uniformly chosen
    active frame, then a uniform offset inside it.
  - The level is set on the event's full window, before any truncation fade.
  - Rows record `event_to_speech_db` (the target) and `ratio_measured_db` (the resulting
    utterance-level added-noise ratio, speech over noise).
  - Per the plan, gun10 has events 10 dB *above* the speech, so it is the harder gun
    condition. That is the opposite direction to eng10.
- **Babble.** Talkers are five files drawn without replacement. A segment counts as
  active if at least 60 % of its frames pass the -30 dB gate relative to the segment's
  own loudest frame. Each file gets up to 20 offsets; after that, another file replaces
  it. Each talker is normalized to equal whole-segment RMS.
- **Reverberation.**
  - The RIR file is drawn uniformly over the sorted BUT pool, then a channel uniformly.
  - "Scaled to the input's RMS" means: the output's RMS over the input region
    `[0, n)` equals the input RMS. That way a delta RIR reproduces the input exactly.
  - rev and field outputs are 0.5 s longer than clean, and the manifest durations say so.
- **BUT ReverbDB layout** (not verified yet; the archive is still downloading). The
  index uses every `.wav` under a directory named `RIR`. The room is the path
  component that names one of the nine rooms, otherwise the first component.
- **Radio.**
  - "8th-order Butterworth" is `scipy.signal.butter(N=4, band-pass)`, which is order
    8, run zero-phase with `sosfiltfilt`.
  - The Codec2 delay is 156 samples at 8 kHz (19.5 ms). It is measured by
    cross-correlating 20 ms-Hann RMS envelopes on a fixed 12 s signal of aperiodic
    harmonic bursts (PCG64 seed 0). Codec2 is parametric (its phase is synthesized), so
    waveform alignment is not meaningful.
  - The residual alignment on seven other such signals is 0-16 samples at 16 kHz,
    that is within 1 ms but at the edge.
  - The final-output band ratio is -62 dB on white noise and -53 to -69 dB on 19 real
    dev utterances (contract: at most -40 dB).
- **Headroom.** The 0.99 peak rule is applied to every processed output, rev and radio
  included, so that no int16 clipping can occur. Counts are in `build.json`.
- **Clusters.**
  - LibriSpeech: the speaker.
  - AMI: the meeting (the second ID field).
  - YODAS: the video (the 11-character YouTube ID after the shard prefix).
  - VoxPopuli: `speaker_id` from the local validation parquet. Where it is `None`
    (some rows), the plenary session is used.
- **Clean condition.** Clean rows point at the original audio (no copy). The lock
  records every input's decoded-PCM SHA-256, and each worker re-checks it before
  rendering.
- **PCM hash.** SHA-256 of `b"<rate>,<channels>\n"` followed by the soundfile-decoded
  float32 little-endian samples (`noisy_sources.pcm_sha256`). A TRAIN check must hash
  candidates the same way.
- **Rotor plug-in "freesound".** It reads `raw/freesound-rotor/sounds.json`, with entries
  `{id, username, license, split, file, sha256}`, and admits only CC0 clips of
  `split == "eval"`. The downloader and the uploader split are not written: they need
  the user's API key and the uploader-split rule.
- **Size.** With synthetic noise, the measured output is about 69 MB per audio hour. For
  about 375 hours that is about 26 GB, not the 12 GB the plan estimates. Real noise
  compresses differently; there is ample disk.
- **Build memory.** The unit runs at `--mem-max 6G` (this plan). Measured peaks: 1.1-1.3
  GB with 4 workers.

## Review round 2 responses (code, 2026-10-16, implementation subagent)

Codex round 2 (`/mnt/hd/wilderness-labs-stt/noisy-dev/review/r2.md`), the code findings. Each
fix has failing tests written first. The suite is 114 tests (70 before this round), run on the
CPU inside `../parakeet-ternary/heavy` (6G, `nice -n 19`), all passing. Where this section and
"Implementation notes" disagree, this section supersedes.

1. **Blocker, build.json schema.**
   - `noisy_build.py:800` keeps the condition IDs in `conditions`; `noisy_build.py:806`
     puts the statistics under `statistics`.
   - `noisy_score.build_identity` (`noisy_score.py:373`) requires a list of IDs.
   - The scorer is now tested on real builder output: `test_noisy_build`
     `test_scorer_accepts_the_real_builder_output`, and `ScoreEntryPointTest` runs on a
     fixture `tiny_build`. It was also run on a real sample build.
2. **Input identity.**
   - `noisy_sources.check_pool_inputs` (`noisy_sources.py:724`) runs before every build
     and resume (`noisy_build.py:696`). It checks two things: each registry source's
     current verification record names exactly the downloads the pool was built from,
     and every pool audio file still decodes to its recorded PCM hash.
   - `prepare` (`noisy_sources.py:376`) names a copy by the original's SHA-256 and the
     preparation parameters, with a sidecar. A stale or tampered copy is redone.
   - `extract_archive` (`noisy_sources.py:754`) keys its directory by the verified archive
     SHA-256, the member selection, the budget and the extractor's code hash. On reuse,
     every member must keep its recorded size and SHA-256.
   - A rerun of a complete build re-verifies every output and redoes corrupted ones
     (`noisy_build.py:732`).
   - Tests: a noise file, an RIR file or a verified download changed after an
     interruption is refused; re-verification with other contents is refused; a
     corrupted completed output is redone byte-identically; the caches are bound.
3. **Scoring pairing.**
   - `condition_rows` (`noisy_score.py:392`) checks the lock SHA-256 against
     `build.lock.json` and the manifest SHA-256. Per row it checks IDs, references,
     durations and the expected `audio_filepath`, plus the SHA-256 of every audio file.
   - `compare` (`noisy_score.py:310`) requires identical build identity (lock, manifest,
     condition list) and identical condition sets. It also requires, per utterance,
     identical normalized references and exclusion flags (`noisy_score.py:243-245`).
   - Tests: clean paths substituted under radio, a tampered FLAC, a tampered lock, a
     reference mismatch with equal word counts, an exclusion-mask mismatch, a build-identity
     mismatch and a condition-set mismatch are all refused.
4. **BUT verification.**
   - BUT is now trust on first use (`noisy_sources.py:255`): an https URL only, the exact
     size, SHA-256 trusted on first use, and records carrying `transport: https` and the
     server's Last-Modified. A non-https record is refused by `verify` and
     `require_verified`. A known digest can be pinned in `SOURCES`.
   - The contract test changed accordingly: the old "first-seen ok" test is replaced by
     the trust-on-first-use tests.
   - **The pinned size is 9,308,593,693 bytes, not the 9,307,512,349 in "Evaluation-pool
     sources".** Both a HEAD request over HTTPS on 2026-10-16 (Content-Length) and the
     ETag `22ad5de1d-…` give 0x22ad5de1d = 9,308,593,693. The coordinator was told; that
     paragraph is theirs to correct.
5. **Extraction budgets.**
   - `noisy_extract.Budget` (`noisy_extract.py:41`) caps the member count (all members,
     whether selected or not), the expanded bytes per member and the total expanded bytes.
   - Each zlib call yields at most 1 MiB of output, and a member that inflates past its
     declared size is refused (`_Sink`, `noisy_extract.py:192`). Tar budgets count every
     member in the stream.
   - Per-source budgets are in `noisy_sources.py:136`.
   - Tests: a zip bomb, a zip member that lies about its size, too many members, too
     large a total, the tar equivalents, and a within-budget control.
6. **Leakage provenance.**
   - `check_train_pool` (`noisy_blocklist.py:128`) now requires `dataset` and
     `pcm_sha256` on every row. Freesound-derived rows also need a sound ID and an
     uploader, FSD50K rows a split (eval or test is a reserved split), MUSAN rows a subset
     (a parent of `speech/us-gov` is ambiguous), and BUT rows a room.
   - The FSD50K reservation now comes from `eval.csv` plus the clip info
     (`noisy_sources.py:440`). A missing expected audio file or clip-info entry fails.
   - Checked on the real metadata (publisher MD5 verified): 10,231 eval clips, all with
     uploaders (2,297 uploaders). After the licence and voice filters, about 184 gun, 726
     engine, 90 wind and 376 crowd clips remain admissible, before the length and level
     filters.
7. **Babble.** `talker_ok` (`noisy_build.py:364`) accepts a segment only if its active
   frames are at least -50 dBFS RMS, its 20 ms frame energies are modulated by at least
   10 dB (90th minus 10th percentile), and at least 60 % of its frames are active. Silence
   and a stationary noise floor fail. Tests: bad talkers are never chosen, too few speech
   files are refused, and the written babble matches an independent reconstruction from
   the recorded talkers, each normalized to equal RMS.
8. **Radio after serialization.**
   - Every output is now 24-bit FLAC (`noisy_build.py:135-141`), the flagged clean copies
     included (exact for the 16-bit inputs).
   - Measured at a -69 dBFS input: the band ratio is -34.7 dB at 16 bits (fails) and
     -62.0 dB at 24 bits.
   - Tests: the bandwidth contract is checked on the decoded files at 7 input levels from
     -69 to -3 dBFS, for noise and for speech, plus six non-frame-aligned lengths (output
     length, last 100 ms, level, reverb length).
   - **Size:** a real 20-utterance sample's radio audio grew from 2.1 MB to 4.7 MB (2.2x).
     The full build will be about 50 GB, against 12 GB in the plan and 26 GB at 16 bits.
9. **Report by noise-source group.**
   - Rows carry `groups` (`noisy_build.py:530`): `noise` is the FSD50K class (the most
     specific match), the DEMAND environment, the Zenodo microphone, MUSAN or the rotor
     uploader; impulsive rows use the most frequent event group. `rir` is the BUT room.
   - `compare(..., groups=)` adds `by_group`: pooled WER per model and the gap to the
     reference per group (`noisy_score.py:285`). The `interval_scope` field says the
     intervals are conditional on the frozen noise realization.
   - Test: a known-answer breakdown.

Not addressed in code, because they are not code findings or are handled elsewhere: the
`download.sh` watchdog (the coordinator's), the speech-overlap audit and the interval
scope wording (plan text, amended by the coordinator), and the pause-severity wording.

Still open: BUT archive layout (index rule unchanged and unverified), the rotor
downloader, a real full-source sample (no source is completely downloaded yet), GPU
scoring, the listening folder.

## Review round 3 responses (code, 2026-10-16, implementation subagent)

Codex round 3 (`/mnt/hd/wilderness-labs-stt/noisy-dev/review/r3.md`): six findings. Each has
failing tests written first; no existing expectation was weakened. The blocklist negatives
were tightened to assert exact reasons. The suite is 136 tests (114 before this round), run on
the CPU inside `../parakeet-ternary/heavy` (6G, `nice -n 19`, 616 MB peak), all passing.

1. **Metadata budgets.**
   - `Budget.max_metadata_bytes` (`noisy_extract.py:57`) bounds a zip central directory and
     each tar extended header.
   - Zip: the entry count and central-directory size, classic or ZIP64, are checked before
     the directory is read (`noisy_extract.py:152`), as is consistency with the archive size.
     Only the fixed-size end records are read before that.
   - Tar: an own parser (`parse_tar`, `noisy_extract.py:405`, over `_TarStream`, `:298`)
     replaces `tarfile`, whose stream parser reads PAX/GNU headers whole. Each 512-byte header
     is checksum-verified. An extended header (`x`/`g`/`L`/`K`) above the budget is refused
     before its data is read. Extended headers count as members, their total is capped
     (`METADATA_TOTAL_FACTOR`, `:60`), and the decompressed stream is bounded.
   - The FSD50K eval central-directory budget is 16 MiB (about 10k entries).
   - Tests: a 1 GiB declared central directory, ZIP64 with a huge size or count (the largest
     read request stays at the 64 KiB tail), a 64 KiB PAX header and a 5,000-character GNU
     long name under a 1 KiB budget, extended headers as members, PAX/GNU/USTAR long-name
     parsing, the header checksum, and ZIP64 parsing. Re-run on the real FSD50K metadata
     zips (publisher MD5 verified).
2. **`check_train_pool`.**
   - `resolve_origin` (`noisy_blocklist.py:114`) maps collection wrappers (`WRAPPERS`, `:63`:
     DNS-Challenge, RIRS_NOISES/SLR28, SLR26) through their subset to the canonical origin;
     DNS `DEMAND/...` resolves to DEMAND, which is reserved whole. A mixed-origin wrapper subset
     needs an explicit `origin`, and an unknown wrapper subset is ambiguous.
   - A dataset outside `CANONICAL_DATASETS` (`:56`) fails closed (`unknown_dataset`).
   - A MUSAN subset must lie in one `MUSAN_GROUPS` group (`:53`); `noise` and `speech` are
     ambiguous. A BUT room may not be empty or whitespace (`:167`).
   - Every negative test is an otherwise valid candidate and asserts the exact set of reasons.
3. **Resume certification.**
   - A completed manifest is reused only if its SHA-256 equals build.json's
     (`ManifestTampered`, `noisy_build.py:136`, `:763`). It is then left in place, not converted.
   - Journal rows (`_read_partial`, `:645`) must match `expected_rows` (`:633`) from the
     current locked inputs: identity, reference, source, cluster, seed and audio path, plus the
     audio SHA-256. Any other row is redone.
   - Tests: an edited reference, an edited cluster, and a row pointing at another valid file
     (with that file's hash) are all redone byte-identically; an edited completed manifest is
     refused. On real data, a rerun of the complete sample re-validated all 40 rows.
4. **Sample provenance.**
   - `build_identity` (`noisy_score.py:378`) carries `complete`, `sample`,
     `skipped_conditions`, `pool_missing_sources`, `utterances` and `full`.
   - `output_name` (`:436`) gives production names only to a full build without `--limit`.
     Otherwise the output is `smoke-<name>-sample<N>` or `-partial`, with `-limit<N>` appended
     for a limit. The summary records `smoke` and `out_name` (`:489`).
   - `compare` requires these provenance fields and the limit to be equal (`IDENTITY_KEYS`,
     `:285`) and marks smoke comparisons (`:327`).
   - Tests: a sample build scores to `smoke-B0-sample10`, a full build to its name, and
     comparisons mixing sample with full, or limited with unlimited, are refused. On real data,
     the 20-utterance sample resolves to `smoke-B0-sample20`.
5. **Independent reconstructions** (`test_noisy_build.ReconstructionTest`).
   - field: the test does its own RIR onset alignment, convolution, tail and normalization,
     its own looping, its own 5 dB gain against the reverberated speech (equal to the
     recorded gain to 1e-9), then radio and headroom, and matches the written FLAC.
   - Impulsive events: the test re-derives each event gain from the frozen event window and
     the clean RMS, applies its own truncation and fade, and matches the written FLAC.
   - The fixtures gained two sustained 2.5 s events so that truncation occurs.
6. **Interpreter.** `./python` sets `PYTHONPATH` from a fixed list of trusted absolute
   directories (`python:28`), checks that each exists, and never appends the inherited value.
   A test runs the wrapper with a hostile `PYTHONPATH` (an untrusted directory holding a fake
   `noisy_paths.py`, a relative entry and empty entries) from inside that directory.

Real data on 2026-10-16: none of the five sources is completely downloaded (DEMAND 13 of 17
done, FSD50K metadata only). The real sample build therefore still covers clean and radio only.

## Review round 4 responses (code, 2026-10-16, implementation subagent)

Codex round 4 (`/mnt/hd/wilderness-labs-stt/noisy-dev/review/r4.md`): three majors. Each has
failing tests written first; no existing expectation was weakened. The suite is 147 tests (136
before this round), run on the CPU inside `../parakeet-ternary/heavy` (6G, `nice -n 19`, 621 MB
peak), all passing; lint is clean.

1. **ZIP deflate stream** (`noisy_extract.py:276`, `:288`).
   - A deflated member must be one complete deflate stream that consumes exactly its declared
     compressed size.
   - Before each compressed chunk is read, a finished stream with bytes left is refused.
     After every zlib call, `unused_data` (and `unconsumed_tail` once at the end) are refused.
   - A stream without an end is refused as truncated. The final `flush()` that could hide one
     was removed.
   - Tests: one decoded byte plus 4 MiB of trailing data under a 16-byte budget is refused
     after exactly one 1 MiB compressed read; a truncated stream is refused; an exact stream
     is extracted.
2. **Explicit origins** (`resolve_chain`, `noisy_blocklist.py:139`).
   - An origin may only carry `ORIGIN_FIELDS` (`:121`): dataset, subset, split, room, sound ID
     and uploader, never a PCM hash.
   - An origin must agree with a known wrapper mapping, the `*` mappings included (DNS
     `DEMAND/...` with origin `slr26-simulated` conflicts). A non-wrapper candidate's origin
     must name the candidate's own dataset.
   - The candidate's own PCM hash is always checked (`:251`), and units, reserved splits,
     sound IDs and uploaders are checked at every link of the chain (`:259`).
   - Tests (otherwise valid candidates, exact reasons): a wrapper conflict, a `*` conflict, a
     plain conflict, an origin carrying a hash (ambiguous, plus the outer hash match), a valid
     origin over a reserved outer hash, a reserved outer uploader or sound ID hidden by an
     origin, an unknown origin field, and a valid explicit origin that passes.
3. **Unregistered sources** (`noisy_sources.py:640`, `:763`).
   - Pool admission and `check_pool_inputs` accept only registered sources (`SOURCES`, through
     `require_verified`) and registered plug-ins (`PLUGIN_SOURCES`, `:629`).
   - The rotor plug-in is checked by `verify_freesound_rotor` (`:611`): the SHA-256 of
     `sounds.json` and of every admitted download, recorded by the plug-in when it indexed them.
   - Any other source is refused, whatever verification it claims. `sources_checked` now
     lists every pooled source.
   - The fixtures build the rotor pool through the real plug-in, including an NC entry that
     never enters.
   - Tests: an unregistered self-attested source is refused at admission (contract 3) and by
     `check_pool_inputs` (contract 4); so are a registered source whose record is missing,
     changed plug-in metadata, a changed plug-in download, and a plug-in entry without
     verification files.

Real data: the FSD50K metadata zips were re-extracted with the new deflate checks, and a fresh
20-utterance sample (`samples/sample-20-r4`, clean and radio only, since no source has finished
downloading) passed the scorer's validation and resolves to `smoke-B0-sample20`.

## Review round 5 responses (code, 2026-10-16, implementation subagent)

Codex round 5 (`/mnt/hd/wilderness-labs-stt/noisy-dev/review/r5.md`). The suite is 160 tests
(147 before this round), run on the CPU inside `../parakeet-ternary/heavy` (6G, `nice -n 19`,
619 MB peak), all passing; lint is clean. The new tests were written first and failed before
the fixes.

1. **Origins** (provisional check, as rescoped by the coordinator).
   - `resolve_chain` (`noisy_blocklist.py:160`) refuses an explicit origin whenever the
     candidate's dataset is a known wrapper or MUSAN.
   - The two negatives from the review are tests: MUSAN `noise/free-sound` with origin
     `speech/librivox`, and DNS `slr28/pointsource_noises` with origin `rirs-noises/simulated_rirs`.
   - **Contract change:** the round-4 cases that resolved a wrapper through an explicit origin
     now fail closed (ambiguous). Reserved identities and hashes stated on the candidate are
     still reported alongside, and the tests were updated to these expectations.
2. **Verification records.**
   - `validate_record` (`noisy_sources.py:312`) requires `ok`, the source's own name, exactly
     the registry's files (none missing, none extra, no duplicates), complete entries (size,
     mtime, MD5, SHA-256) and the registry's checksums: the publisher MD5, or for trust on first
     use https transport, the pinned size and any pinned digest.
   - `require_verified` and `check_pool_inputs` take the registry explicitly (production
     `SOURCES` by default). The build has `BuildConfig.registry`, and the tests inject their own
     registry (`noisy_fixtures.registry`), so a test record never passes under production rules.
   - `verify_source` no longer crashes on a malformed previous record. For trust-on-first-use
     sources it refuses one.
   - Tests: no files, a missing file, an extra file, another MD5, another source name, a missing
     SHA-256, `ok: "yes"`, the bare `{"ok": true, "files": []}`, and a test record under a
     production name are all refused. An empty record matching an empty pool entry is refused.
     Under its own registry, a re-verified download now fails registry validation; a pool built
     from other downloads still raises `InputChanged`.
3. **Extraction caches.**
   - On reuse (`noisy_sources.py:883`), the directory must hold exactly the recorded members:
     an unexpected file, link or special file raises `InputChanged`.
   - `extract` returns the verified member list `[(dest, name, sha256)]`, and the indexers
     (`index_fsd50k` `:497`, `index_demand` `:581`, `index_musan` `:598`, `index_but` `:616`)
     read only that list, never a directory glob.
   - Every original, prepared or used in place, is checked against its member SHA-256 before
     use (`prepare`/`_prepared`, `:436`), and the gunshot recordings against their
     verification record.
   - Tests: an extra.wav and a symlink are refused, the indexer ignores an unlisted extra.wav,
     and prepared audio is refused for a wrong member hash.
4. **Compression.** TAR is accepted only as gzip, checked by magic (`noisy_extract.py:316`).
   XZ, bzip2, plain tar and anything else are refused whatever the file name, and the XZ and
   bzip2 decoders are no longer imported. ZIP methods stay limited to stored and deflate.
   Tests: an XZ header announcing a ~4 GiB dictionary, named `.tgz` (the test confirms that a
   64 MiB-limited decoder rejects it), is refused before any decoder is created, as are
   bzip2-in-`.tgz` and plain tar.
5. **Completeness** (`build_identity`, `noisy_score.py:382`).
   - Conditions, sample size, skipped conditions, missing pool sources and utterances are
     derived from the validated lock.
   - The manifest must cover exactly the locked conditions times dev sets, with exactly the
     locked utterances.
   - `full` additionally needs every base condition, the five dev sets, and each locked dev
     manifest unchanged (its SHA-256) and complete (row count).
   - `build.json` must contain every field in `SUMMARY_FIELDS` (`:378`) and agree with the
     derived values.
   - Tests: removed sample fields, a sample claiming full, a full build claiming a sample, and
     contradicted skipped conditions, missing sources, utterances or condition order are all
     refused; the full-coverage identity is checked.

**First real-noise sample.**
- **Sources:** FSD50K, DEMAND and the Zenodo gunshots finished downloading and verified
  against their publisher MD5s. MUSAN and BUT are still downloading.
- **Pool:**
  - Noise clips: engine 723, crowd 371, wind 90, DEMAND 272.
  - Impulsive: 188 clips (184 FSD50K, 4 Zenodo) with 624 events (511 FSD50K, 113 Zenodo).
  - FSD50K: all 10,231 eval clips reserved.
  - FSD50K exclusions: 273 licence, 207 multi-category, 110 voice label, 6 too short,
    2 silent.
- **Build** (`samples/sample-50-r5`, 50 utterances, 12 conditions, 600 rows; babble, rev and
  field skipped): 34 s with 4 workers, 0.93 GB process memory (the rest of the unit's peak is
  page cache).
  - Continuous ratios measured on the written files match their targets to 1e-4 dB.
  - gun0 and gun10 average 3.0 events per utterance with 81-84 % overlap with active speech;
    peaks are about 15.5 dB above their window level, and 35 of 50 gun10 mixtures needed
    headroom scaling.
  - Output is 140 MB for 1.12 audio hours, about 125 MB per hour, so the full build will be
    about 47 GB.
- **Checks:** a 1-worker rebuild is byte-identical (550 FLACs and the manifest). The scorer
  validates the build and routes it to `smoke-B0-sample50`.
- **For the listening check:** the Zenodo recordings yield 113 detected events where the
  readme lists 22 shots per microphone (88). The extra detections may be impacts, echoes or
  other loud sounds.

## Review rounds 7-8 (2026-10-16, focused, after the CLEAN round 6)

The first full build stopped in extraction because the real BUT ReverbDB archive
begins with a `./` directory entry, which the safe-name check refused. Fixes:
- `extract_tar` skips a directory entry that names the archive root.
- `extract_zip` skips a root entry with no data: stored, or deflate's 2-byte
  empty stream. It still counts toward the member budget.

Tests `test_tar_root_directory_entry_accepted` and
`test_zip_root_directory_entry_accepted` failed before the fixes; the suite now
has 162 tests, all OK.
- Round 7 found no tar regression, and flagged the same gap in ZIP.
- Round 8 flagged the deflated empty root, now covered. Its severity was P2.
  Before the fix such an archive was refused, which is fail-closed, and none of
  the 21 registered ZIP archives has a root entry.

## Rotor decision (2026-10-16, user)

No Freesound API key. Rotor and helicopter robustness is reported as
**unmeasured**. The rotor10/rotor0 conditions are not built, and engine noise
does not stand in for them. The rotor plug-in stays in the code, unused.
