"""Build the fixed noisy copy of the five full development sets. See plans/build.md.

    ./python noisy_build.py build [--workers 4] [--sample N --allow-missing] [--rotor pool/rotor.json]
    ./python noisy_build.py status [--out DIR]

(real data: inside `../parakeet-ternary/heavy noisy-build --mem-max 6G -- nice -n 19 $PWD/python ...`)

Inputs: the dev manifests of parakeet-ternary (MANIFESTS/<set>.jsonl, all rows), the pool
(noisy_sources: pool/pool.json, optionally a rotor plug-in's pool/rotor.json) and the pinned
Codec2. Every input is frozen in OUT/build.lock.json (dev manifest hashes and every input
utterance's decoded-PCM SHA-256, the whole pool with its per-file PCM hashes and event list, all
processing parameters, the measured codec delay, library versions, the code's SHA-256); a build
whose recomputed lock differs from an existing one is refused (LockMismatch).

Conditions (CONDITIONS; rotor10/rotor0 only with a rotor pool) are rendered per utterance with
numpy Generator(PCG64(int(SHA-256(utterance ID, condition ID, build seed)))), so an output depends
only on its utterance, its condition and the pool, never on the worker count, the order of work
or the other conditions. Short (< 0.3 s) or silent (< -70 dBFS RMS) utterances keep their clean
audio in every processed condition and are flagged. The clean condition's rows point at the
original audio (no copy).

Outputs (OUT = build/ for the full build, samples/<name>/ for --sample):

    audio/<condition>/<set>/<name>.flac        16 kHz mono 24-bit FLAC (review round 2: 16-bit quantization
                                               broke the radio bandwidth contract at low levels)
    manifests/<condition>/<set>.jsonl          NeMo-style rows (audio_filepath, duration, text, id, source, cluster)
    build_manifest.jsonl                       one canonical-JSON row per output, sorted (condition, set, id)
    build.lock.json, eval_pool_blocklist.json, build.json (statistics, timing; written last)

While a build runs, rows are appended to build_manifest.partial.jsonl after their FLACs are
written; a rerun resumes from it (rows whose FLAC SHA-256 no longer matches, and a torn last
line, are redone) and deletes it when the build is complete. A rerun of a complete build
re-verifies every output the same way and redoes corrupted ones (byte-identical by construction).
Review round 3: a completed manifest is used only if its SHA-256 equals the one build.json
recorded (ManifestTampered otherwise), and a journal row is reused only if its identity,
reference, source, cluster, seed and audio path equal those the current locked inputs give
(expected_rows) and its audio file still has its SHA-256; any other row is redone.

Before every build and resume, noisy_sources.check_pool_inputs re-checks the pool's verified
downloads and every pool audio file's decoded-PCM SHA-256 (InputChanged otherwise), so a source
changed after an interruption is never mixed into a build under the old lock.

build.json holds the condition IDs under "conditions" and the statistics under "statistics"
(review round 2: the scorer validates the former).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import datetime as dt
import functools
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

import noisy_blocklist
import noisy_dsp as dsp
import noisy_paths
import noisy_sources

SR = noisy_paths.SAMPLE_RATE
REVERB_TAIL = dsp.REVERB_TAIL
SHORT_SECONDS = 0.3
SILENT_DBFS = -70.0
MIN_SEGMENT_DBFS = -70.0   # a drawn noise segment quieter than this is redrawn (it would be amplified hiss)
SEGMENT_ATTEMPTS = 10
BABBLE_TALKERS = 5
BABBLE_MIN_ACTIVE = 0.6
BABBLE_MIN_ACTIVE_DBFS = -50.0   # absolute level of a talker segment over its active frames
BABBLE_MIN_MODULATION_DB = 10.0  # 90th minus 10th percentile of 20 ms frame energies: speech, not a stationary floor
BABBLE_ATTEMPTS = 20
FIELD_RATIO_DB = 5.0
LOCK_FORMAT = "noisy-dev-lock/1"
RNG = "numpy.random.Generator(PCG64(int(SHA-256(utterance ID, condition ID, build seed))))"
CODE_FILES = ("noisy_build.py", "noisy_dsp.py", "noisy_sources.py", "noisy_paths.py", "noisy_blocklist.py",
              "python")
CACHE_FRAMES = 500_000     # clips up to ~31 s are cached per worker (<= 64 x 4 MB); longer ones are read by offset
BUILD_PARAMETERS = {"short_seconds": SHORT_SECONDS, "silent_dbfs": SILENT_DBFS, "min_segment_dbfs": MIN_SEGMENT_DBFS,
                    "segment_attempts": SEGMENT_ATTEMPTS, "babble_talkers": BABBLE_TALKERS,
                    "babble_min_active": BABBLE_MIN_ACTIVE, "babble_attempts": BABBLE_ATTEMPTS,
                    "babble_min_active_dbfs": BABBLE_MIN_ACTIVE_DBFS,
                    "babble_min_modulation_db": BABBLE_MIN_MODULATION_DB,
                    "field_ratio_db": FIELD_RATIO_DB, "flac": "soundfile PCM_24 FLAC (int32 << 8 of round(x * 2^23)), libsndfile default level",
                    "draws": "source uniform over the sorted pool (Generator.integers); offset uniform; events "
                             "uniform over the sorted event list; onsets uniform over active-frame samples"}


@dataclasses.dataclass(frozen=True)
class Condition:
    id: str
    kind: str                     # clean, noise, events, babble, reverb, radio, field
    category: str | None = None   # pool category (noise, babble, field's noise) or "gun"
    ratio_db: float | None = None


CONDITIONS = (
    Condition("clean", "clean"),
    Condition("gun0", "events", "gun", 0.0), Condition("gun10", "events", "gun", 10.0),
    Condition("eng10", "noise", "engine", 10.0), Condition("eng0", "noise", "engine", 0.0),
    Condition("wind10", "noise", "wind", 10.0), Condition("wind0", "noise", "wind", 0.0),
    Condition("bab10", "babble", "babble", 10.0), Condition("bab0", "babble", "babble", 0.0),
    Condition("crowd10", "noise", "crowd", 10.0), Condition("crowd0", "noise", "crowd", 0.0),
    Condition("amb10", "noise", "demand", 10.0), Condition("amb0", "noise", "demand", 0.0),
    Condition("rev", "reverb"), Condition("radio", "radio"), Condition("field", "field", "engine", FIELD_RATIO_DB),
)
ROTOR_CONDITIONS = (Condition("rotor10", "noise", "rotor", 10.0), Condition("rotor0", "noise", "rotor", 0.0))
ROTOR_AFTER = "eng0"
UNMEASURED_ROTOR = "unmeasured (no licensed rotor material)"


class BuildError(RuntimeError):
    pass


class LockMismatch(BuildError):
    """The inputs, code or parameters differ from those recorded in an existing build lock."""


class BuildInterrupted(BuildError):
    """Raised by the stop_after hook (tests of the resume path)."""


class ManifestTampered(BuildError):
    """A completed build manifest differs from the SHA-256 its build.json recorded."""


InputChanged = noisy_sources.InputChanged  # a pool input differs from the one the pool was built from


def encode_pcm24(x: np.ndarray) -> np.ndarray:
    """round(x * 2^23) clipped to 24 bits, as int32 left-aligned for soundfile's PCM_24 writer."""
    q = np.clip(np.round(np.asarray(x, dtype=np.float64) * 8388608.0), -8388608, 8388607).astype(np.int32)
    return q << 8


def write_flac(path: Path, x: np.ndarray) -> None:
    """Atomically write 16 kHz mono 24-bit FLAC of a float signal (|x| <= 1)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}.flac")
    sf.write(str(tmp), encode_pcm24(x), SR, format="FLAC", subtype="PCM_24")
    os.replace(tmp, path)


def decoded(x: np.ndarray) -> np.ndarray:
    """What write_flac stores, as read back in float64."""
    return (encode_pcm24(x) >> 8).astype(np.float64) / 8388608.0


def conditions(rotor: bool) -> list[Condition]:
    out = []
    for c in CONDITIONS:
        out.append(c)
        if rotor and c.id == ROTOR_AFTER:
            out.extend(ROTOR_CONDITIONS)
    return out


def utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[noisy_build {dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


sha256_file = noisy_paths.sha256_file


# --------------------------------------------------------------------------------------------- utterances

@dataclasses.dataclass(frozen=True)
class Utterance:
    set: str
    id: str
    text: str
    audio_filepath: str
    duration: float
    source: str
    cluster: str
    name: str      # output file stem


def cluster_of(set_name: str, uid: str, voxpopuli_speakers: dict[str, str]) -> str:
    """The bootstrap cluster: LibriSpeech and VoxPopuli speakers, AMI meetings, YODAS videos. A
    VoxPopuli utterance without a speaker ID falls back to its plenary session."""
    native = uid.split(":", 1)[1]
    if set_name.startswith("librispeech"):
        return "speaker:" + native.split("-")[0]
    if set_name == "ami_dev":
        return "meeting:" + native.split("_")[1]
    if set_name == "yodas_dev":
        parts = native.split("_")
        return "video:" + native[len(parts[0]) + len(parts[1]) + 2:][:11]  # YouTube IDs: 11 characters
    if set_name == "voxpopuli_dev":
        speaker = voxpopuli_speakers.get(native)
        if speaker and speaker != "None":
            return f"speaker:{speaker}"
        return "session:" + native.split("_")[0].removesuffix("-en")
    raise ValueError(f"no cluster rule for {set_name}")


def load_voxpopuli_speakers() -> tuple[dict[str, str], dict]:
    """audio_id -> speaker_id from the local VoxPopuli validation parquet of parakeet-ternary."""
    import pyarrow.parquet as pq

    import paths
    files = sorted((paths.DATA / "hf" / "facebook__voxpopuli" / "en").glob("validation-*.parquet"))
    if not files:
        raise FileNotFoundError("VoxPopuli validation parquet not found (parakeet-ternary data/hf)")
    speakers = {}
    for f in files:
        table = pq.read_table(f, columns=["audio_id", "speaker_id"])
        speakers.update(zip(table.column("audio_id").to_pylist(), map(str, table.column("speaker_id").to_pylist()),
                            strict=True))
    return speakers, {"files": [str(f) for f in files], "sha256": [noisy_paths.sha256_file(f) for f in files]}


def _file_stem(uid: str) -> str:
    native = uid.split(":", 1)[1]
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", native).strip(".")[:160] or "utt"
    return f"{safe}-{hashlib.sha256(uid.encode()).hexdigest()[:8]}"


def load_utterances(manifests: Path, sets: list[str], voxpopuli_speakers: dict[str, str],
                    sample: int | None = None) -> list[Utterance]:
    """Every row of each full dev manifest (or, with sample, evenly spaced rows of each)."""
    out = []
    for name in sets:
        rows = [json.loads(line) for line in (Path(manifests) / f"{name}.jsonl").read_text().splitlines() if line.strip()]
        if [r["id"] for r in rows] != sorted(r["id"] for r in rows) or len({r["id"] for r in rows}) != len(rows):
            raise BuildError(f"{name}: manifest ids are not unique and sorted")
        if sample is not None:
            k = max(1, sample // len(sets))
            picks = sorted({int(round(i)) for i in np.linspace(0, len(rows) - 1, min(k, len(rows)))})
            rows = [rows[i] for i in picks]
        for r in rows:
            out.append(Utterance(set=name, id=r["id"], text=r["text"], audio_filepath=r["audio_filepath"],
                                 duration=float(r["duration"]), source=r["source"],
                                 cluster=cluster_of(name, r["id"], voxpopuli_speakers), name=_file_stem(r["id"])))
    return out


def read_clean(path: str) -> np.ndarray:
    pcm, sr = sf.read(path, dtype="int16", always_2d=True)
    if sr != SR or pcm.shape[1] != 1:
        raise BuildError(f"{path}: expected 16 kHz mono, got {sr} Hz x {pcm.shape[1]}")
    return dsp.from_int16(pcm[:, 0])


def flag_of(clean: np.ndarray) -> str | None:
    if len(clean) < SHORT_SECONDS * SR:
        return "short"
    if dsp.level_db(clean) < SILENT_DBFS:
        return "silent"
    return None


# --------------------------------------------------------------------------------------------- seeds

def utterance_seed(uid: str, condition: str, build_seed: int = noisy_paths.BUILD_SEED) -> int:
    return int.from_bytes(hashlib.sha256(f"{uid}\x1f{condition}\x1f{build_seed}".encode()).digest(), "big")


def rng_for(uid: str, condition: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(utterance_seed(uid, condition)))


# --------------------------------------------------------------------------------------------- pool access

class PoolView:
    """Read access to the pool's audio (16 kHz) inside one process."""

    def __init__(self, pool: dict) -> None:
        self.pool = pool
        self.clips = pool["clips"]
        self.events = pool["events"]
        self.rirs = pool["rirs"]
        self.event_clips = {c["key"]: c for c in pool["event_clips"]}

    @staticmethod
    @functools.lru_cache(maxsize=64)
    def _whole(path: str) -> np.ndarray:
        return noisy_sources.read_mono(Path(path))

    def read(self, clip: dict, start: int, frames: int) -> np.ndarray:
        if clip["frames"] <= CACHE_FRAMES:
            x = self._whole(clip["path"])[start:start + frames]
        else:
            x = noisy_sources.read_mono(Path(clip["path"]), start=start, frames=frames)
        if len(x) != frames:
            raise BuildError(f"{clip['key']}: read {len(x)} of {frames} frames at {start}")
        return x

    def read_all(self, clip: dict) -> np.ndarray:
        return self.read(clip, 0, clip["frames"])

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def rir_channel(path: str, channel: int) -> np.ndarray:
        x, _ = sf.read(path, dtype="float64", always_2d=True)
        return x[:, channel].copy()


def draw_segment(view: PoolView, clips: list[dict], n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """n samples of one clip drawn uniformly from the sorted pool, at a uniform offset; looped with
    the 50 ms crossfade if the clip is shorter. Near-silent segments are redrawn."""
    for attempt in range(1, SEGMENT_ATTEMPTS + 1):
        clip = clips[int(rng.integers(len(clips)))]
        if clip["frames"] >= n:
            offset = int(rng.integers(0, clip["frames"] - n + 1))
            seg, looped = view.read(clip, offset, n), False
        else:
            offset = int(rng.integers(0, clip["frames"]))
            seg, looped = dsp.loop(view.read_all(clip), n, offset), True
        if dsp.level_db(seg) >= MIN_SEGMENT_DBFS:
            return seg, {"key": clip["key"], "offset": offset, "length": n, "looped": looped, "attempt": attempt,
                         "group": clip.get("group") or clip["source"]}
    raise BuildError(f"no segment above {MIN_SEGMENT_DBFS} dBFS in {SEGMENT_ATTEMPTS} draws")


# --------------------------------------------------------------------------------------------- rendering

def render_noise(view, clean, cond: Condition, rng) -> tuple[np.ndarray, dict]:
    seg, info = draw_segment(view, view.clips[cond.category], len(clean), rng)
    g = dsp.noise_gain(clean, seg, cond.ratio_db)
    return clean + g * seg, {"noise": [info], "noise_gain": g}


def render_events(view, clean, cond: Condition, rng) -> tuple[np.ndarray, dict]:
    n = len(clean)
    mask = dsp.active_frames(clean)
    active = np.nonzero(mask)[0]
    target = dsp.level_db(clean) + cond.ratio_db
    noise = np.zeros(n)
    covered = np.zeros(n, dtype=bool)
    events = []
    for _ in range(dsp.event_count(n)):
        e = view.events[int(rng.integers(len(view.events)))]
        frame = int(active[int(rng.integers(len(active)))])
        onset = min(n - 1, frame * dsp.FRAME + int(rng.integers(0, dsp.FRAME)))
        window = view.read(view.event_clips[e["clip"]], e["start"], e["length"])
        gain = 10 ** (target / 20) / dsp.rms(window)
        scaled = window * gain
        start, kept, shifted, truncated = dsp.place_event(e["length"], n, onset)
        noise[start:start + kept] += dsp.fit_event(scaled, n) if truncated else scaled
        covered[start:start + kept] = True
        clip = view.event_clips[e["clip"]]
        events.append({"event": e["key"], "clip": e["clip"], "group": clip.get("group") or clip["source"],
                       "start": start, "length": kept, "onset_drawn": onset,
                       "shifted": shifted, "truncated": truncated, "gain": gain,
                       "window_level_db": dsp.level_db(scaled),
                       "peak_db": 20 * math.log10(float(np.max(np.abs(scaled))))})
    active_samples = np.repeat(mask, dsp.FRAME)[:n]
    overlap = float((covered & active_samples).sum() / covered.sum())
    return clean + noise, {"events": events, "event_active_overlap": overlap}


def talker_ok(seg: np.ndarray) -> tuple[bool, dict]:
    """A babble segment counts as a talker if at least 60 % of its frames pass the -30 dB gate,
    its active frames are at least -50 dBFS RMS, and its frame energies are modulated like speech
    (p90 - p10 >= 10 dB): silence and a stationary noise floor fail."""
    mask = dsp.active_frames(seg)
    fraction = float(np.mean(mask))
    if not mask.any():
        return False, {"active_fraction": 0.0}
    active = np.repeat(mask, dsp.FRAME)[: len(seg)]
    level = dsp.level_db(seg[active])
    energy = dsp.frame_energies(seg)
    with np.errstate(divide="ignore"):
        db = 10 * np.log10(energy[: len(seg) // dsp.FRAME] if len(seg) >= dsp.FRAME else energy)
    db = db[np.isfinite(db)]
    modulation = float(np.percentile(db, 90) - np.percentile(db, 10)) if len(db) else 0.0
    ok = fraction >= BABBLE_MIN_ACTIVE and level >= BABBLE_MIN_ACTIVE_DBFS and modulation >= BABBLE_MIN_MODULATION_DB
    return ok, {"active_fraction": fraction, "active_level_db": level, "modulation_db": modulation}


def render_babble(view, clean, cond: Condition, rng) -> tuple[np.ndarray, dict]:
    n = len(clean)
    files = view.clips["babble"]
    used: set[int] = set()
    talkers, total = [], np.zeros(n)
    while len(talkers) < BABBLE_TALKERS:
        left = [i for i in range(len(files)) if i not in used]
        if not left:
            raise BuildError(f"babble pool cannot supply {BABBLE_TALKERS} speech talkers (activity >= "
                             f"{BABBLE_MIN_ACTIVE:.0%}, >= {BABBLE_MIN_ACTIVE_DBFS} dBFS, modulation >= "
                             f"{BABBLE_MIN_MODULATION_DB} dB)")
        i = left[int(rng.integers(len(left)))]
        used.add(i)
        clip = files[i]
        for attempt in range(1, BABBLE_ATTEMPTS + 1):
            if clip["frames"] >= n:
                offset = int(rng.integers(0, clip["frames"] - n + 1))
                seg, looped = view.read(clip, offset, n), False
            else:
                offset = int(rng.integers(0, clip["frames"]))
                seg, looped = dsp.loop(view.read_all(clip), n, offset), True
            ok, stats = talker_ok(seg)
            if ok:
                total += seg / dsp.rms(seg)  # equal RMS per talker before summing
                talkers.append({"key": clip["key"], "offset": offset, "length": n, "looped": looped,
                                "attempt": attempt, "group": clip.get("group") or clip["source"], **stats})
                break
    g = dsp.noise_gain(clean, total, cond.ratio_db)
    return clean + g * total, {"babble": talkers, "noise_gain": g}


def render_reverb(view, clean, rng) -> tuple[np.ndarray, dict]:
    rir = view.rirs[int(rng.integers(len(view.rirs)))]
    channel = int(rng.integers(rir["channels"]))
    h = view.rir_channel(rir["path"], channel)
    aligned = dsp.align_rir(h)
    return dsp.reverberate(clean, aligned), {"rir": {"key": rir["key"], "room": rir["room"], "channel": channel,
                                                     "onset": len(h) - len(aligned), "length": len(aligned),
                                                     "group": rir.get("group") or f"{rir['source']}:{rir['room']}"}}


class Renderer:
    """Renders all conditions of one utterance; one per process (pool view, codec)."""

    def __init__(self, pool: dict, out: Path, codec_delay: int, conds: list[Condition], pcm_hashes: dict) -> None:
        self.view = PoolView(pool)
        self.out = Path(out)
        self.delay = codec_delay
        self.codec = dsp.Codec2()
        self.conds = conds
        self.pcm_hashes = pcm_hashes

    def audio_rel(self, cond: str, utt: Utterance) -> str:
        return f"audio/{cond}/{utt.set}/{utt.name}.flac"

    def write(self, rel: str, x: np.ndarray) -> str:
        path = self.out / rel
        write_flac(path, x)
        return sha256_file(path)

    def render(self, utt: Utterance, todo: list[str]) -> list[dict]:
        if noisy_sources.pcm_sha256(Path(utt.audio_filepath)) != self.pcm_hashes[(utt.set, utt.id)]:
            raise LockMismatch(f"{utt.id}: input audio differs from the lock")
        clean = read_clean(utt.audio_filepath)
        flag = flag_of(clean)
        rows = []
        for cond in self.conds:
            if cond.id in todo:
                rows.append(self.render_one(utt, clean, flag, cond))
        return rows

    def render_one(self, utt: Utterance, clean: np.ndarray, flag: str | None, cond: Condition) -> dict:
        seed = utterance_seed(utt.id, cond.id)
        row = {"condition": cond.id, "set": utt.set, "id": utt.id, "text": utt.text, "source": utt.source,
               "cluster": utt.cluster, "flagged": flag, "seed": f"{seed:064x}"}
        if cond.kind == "clean":
            level = dsp.level_db(clean)
            row.update(audio=utt.audio_filepath, sha256=sha256_file(Path(utt.audio_filepath)), samples=len(clean),
                       duration=len(clean) / SR, chain=[], active_fraction=dsp.active_fraction(clean),
                       level_db=level if math.isfinite(level) else None)
            return row
        rel = self.audio_rel(cond.id, utt)
        if flag:
            row.update(audio=rel, sha256=self.write(rel, clean), samples=len(clean),
                       duration=len(clean) / SR, chain=["kept clean (flagged)"], headroom_scale=1.0)
            return row
        rng = np.random.Generator(np.random.PCG64(seed))
        measure_written = True
        if cond.kind == "noise":
            y, info = render_noise(self.view, clean, cond, rng)
            chain = [f"add {cond.category} at {cond.ratio_db:g} dB added-noise ratio"]
        elif cond.kind == "events":
            y, info = render_events(self.view, clean, cond, rng)
            chain = [f"add impulsive events at {cond.ratio_db:g} dB event-to-speech ratio"]
        elif cond.kind == "babble":
            y, info = render_babble(self.view, clean, cond, rng)
            chain = [f"add {BABBLE_TALKERS}-talker babble at {cond.ratio_db:g} dB added-noise ratio"]
        elif cond.kind == "reverb":
            y, info = render_reverb(self.view, clean, rng)
            chain, measure_written = ["reverberate"], False
        elif cond.kind == "radio":
            y, info = dsp.radio(clean, self.codec, self.delay), {}
            chain, measure_written = ["radio (Codec2 3200 surrogate)"], False
        elif cond.kind == "field":
            rev, info = render_reverb(self.view, clean, rng)
            seg, noise = draw_segment(self.view, self.view.clips[cond.category], len(rev), rng)
            g = dsp.noise_gain(rev, seg, cond.ratio_db)
            info.update(noise=[noise], noise_gain=g, ratio_measured_db=dsp.added_noise_ratio_db(rev, g * seg),
                        ratio_reference="reverberated speech, whole length, before the radio channel")
            y = dsp.radio(rev + g * seg, self.codec, self.delay)
            chain, measure_written = ["reverberate", f"add {cond.category} at {cond.ratio_db:g} dB",
                                      "radio (Codec2 3200 surrogate)"], False
        else:
            raise BuildError(f"unknown condition kind {cond.kind!r}")
        y, scale = dsp.headroom(y)
        row.update(info)
        row["groups"] = row_groups(row)
        row.update(audio=rel, sha256=self.write(rel, y), samples=len(y), duration=len(y) / SR, chain=chain,
                   headroom_scale=scale)
        if cond.kind == "events":  # the target is per event (event over speech), not an added-noise ratio
            row["event_to_speech_db"] = cond.ratio_db
        elif cond.ratio_db is not None:
            row["ratio_target_db"] = cond.ratio_db  # speech over added noise
        if measure_written:  # from the written PCM: written - scaled clean speech = added noise
            speech = scale * clean
            noise = decoded(y) - speech
            row.update(speech_scale=scale, ratio_measured_db=dsp.added_noise_ratio_db(speech, noise),
                       ratio_reference="clean utterance, whole length, measured on the written FLAC")
        return row


_WORKER: dict = {}


def _worker_init(pool: dict, out: str, delay: int, conds: list[Condition], pcm_hashes: dict) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    _WORKER["renderer"] = Renderer(pool, Path(out), delay, conds, pcm_hashes)


def _worker_render(utt: Utterance, todo: list[str]) -> list[dict]:
    return _WORKER["renderer"].render(utt, todo)


def _worker_hash(path: str) -> str:
    return noisy_sources.pcm_sha256(Path(path))


def row_groups(row: dict) -> dict:
    """Noise-source groups of a row for the per-group report: "noise" (FSD50K class, DEMAND
    environment, Zenodo microphone, MUSAN, rotor uploader; for impulsive rows the most frequent
    event group, ties to the first in sorted order) and "rir" (BUT room)."""
    out = {}
    noise = [s["group"] for s in row.get("noise", [])] + [e["group"] for e in row.get("events", [])]
    noise += [t["group"] for t in row.get("babble", [])]
    if noise:
        counts = Counter(noise)
        out["noise"] = sorted(counts, key=lambda g: (-counts[g], g))[0]
    if "rir" in row:
        out["rir"] = row["rir"]["group"]
    return out


def referenced_sources(row: dict) -> list[str]:
    """Pool keys a manifest row's audio was made from."""
    keys = [s["key"] for s in row.get("noise", [])] + [e["clip"] for e in row.get("events", [])]
    keys += [t["key"] for t in row.get("babble", [])]
    if "rir" in row:
        keys.append(row["rir"]["key"])
    return keys


# --------------------------------------------------------------------------------------------- lock

def versions() -> dict:
    out = {"python": platform.python_version(), "numpy": np.__version__, "soundfile": sf.__version__,
           "libsndfile": sf.__libsndfile_version__}
    import scipy
    out["scipy"] = scipy.__version__
    try:
        out["torch"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        out["torch"] = None
    return out


def code_hashes() -> dict:
    return {name: sha256_file(noisy_paths.HERE / name) for name in CODE_FILES}


def needs_met(cond: Condition, pool: dict) -> bool:
    if cond.kind == "clean" or cond.kind == "radio":
        return True
    if cond.kind == "events":
        return bool(pool["events"])
    if cond.kind == "babble":
        return len(pool["clips"].get("babble", [])) >= BABBLE_TALKERS
    if cond.kind == "reverb":
        return bool(pool["rirs"])
    if cond.kind == "field":
        return bool(pool["rirs"]) and bool(pool["clips"].get(cond.category))
    return bool(pool["clips"].get(cond.category))


def make_lock(cfg, utts: list[Utterance], pool: dict, conds: list[Condition], skipped: list[str], delay: int,
              pcm_hashes: dict, clusters: dict) -> dict:
    dev_sets = {}
    for name in cfg.sets:
        manifest = Path(cfg.manifests) / f"{name}.jsonl"
        dev_sets[name] = {"manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
                          "pcm_sha256": {u.id: pcm_hashes[(u.set, u.id)] for u in utts if u.set == name}}
    return {"format": LOCK_FORMAT, "build_seed": noisy_paths.BUILD_SEED, "rng": RNG,
            "conditions": [dataclasses.asdict(c) for c in conds], "skipped_conditions": skipped,
            "rotor": cfg.rotor_pool is not None, "sample": cfg.sample,
            "parameters": {**dsp.PARAMETERS, **BUILD_PARAMETERS},
            "codec2": {"store_path": str(noisy_paths.CODEC2_STORE), "version": noisy_paths.CODEC2_VERSION,
                       "c2enc_sha256": sha256_file(noisy_paths.CODEC2_ENC),
                       "c2dec_sha256": sha256_file(noisy_paths.CODEC2_DEC), "mode": dsp.CODEC2_MODE,
                       "delay_samples_8k": delay, "delay_measurement": dsp.DELAY_SIGNAL},
            "dev_sets": dev_sets, "clusters": clusters, "pool": pool, "versions": versions(), "code": code_hashes()}


# --------------------------------------------------------------------------------------------- build

@dataclasses.dataclass
class BuildConfig:
    out: Path
    manifests: Path = noisy_paths.DEV_MANIFESTS
    pool: Path = noisy_paths.POOL / "pool.json"
    rotor_pool: Path | None = None             # a rotor plug-in's pool; None: rotor unmeasured
    workers: int = 4
    sets: tuple[str, ...] = tuple(noisy_paths.DEV_SETS)
    sample: int | None = None                  # sample builds only: N utterances spread over the sets
    allow_missing: bool = False                # sample builds only: skip conditions the pool cannot serve
    voxpopuli_speakers: dict | None = None     # None: read from the VoxPopuli parquet
    stop_after: int | None = None              # test hook: BuildInterrupted after N utterances
    raw: Path = noisy_paths.RAW                # downloads and their verification records, re-checked
    verify_dir: Path = noisy_paths.VERIFY      # before every build and resume
    registry: dict | None = None               # the source registry (None: noisy_sources.SOURCES); tests inject theirs


JOURNAL_FIELDS = ("condition", "set", "id", "text", "source", "cluster", "seed", "audio")


def expected_rows(utts: list[Utterance], conds: list[Condition]) -> dict[tuple, dict]:
    """(condition, set, id) -> the JOURNAL_FIELDS every row of this build must have."""
    out = {}
    for u in utts:
        for c in conds:
            audio = u.audio_filepath if c.kind == "clean" else f"audio/{c.id}/{u.set}/{u.name}.flac"
            out[(c.id, u.set, u.id)] = {"condition": c.id, "set": u.set, "id": u.id, "text": u.text,
                                        "source": u.source, "cluster": u.cluster,
                                        "seed": f"{utterance_seed(u.id, c.id):064x}", "audio": audio}
    return out


def _read_partial(path: Path, lock_sha: str, out: Path, expected: dict[tuple, dict]) -> dict[tuple, dict]:
    """Verified rows of an interrupted build, keyed (condition, set, id): rows whose JOURNAL_FIELDS
    match the current locked inputs and whose audio still has its SHA-256 (others are redone); a
    torn last line is dropped."""
    if not path.exists():
        return {}
    lines = path.read_text().split("\n")
    header = json.loads(lines[0])
    if header.get("lock_sha256") != lock_sha:
        raise LockMismatch(f"{path} belongs to another lock")
    done = {}
    body = [line for line in lines[1:] if line]
    for k, line in enumerate(body):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if k == len(body) - 1:
                break  # torn by an interruption
            raise
        key = (row.get("condition"), row.get("set"), row.get("id"))
        want = expected.get(key)
        if want is None or any(row.get(f) != want[f] for f in JOURNAL_FIELDS):
            continue  # not this build's row, or an edited identity, reference, cluster, seed or path
        audio = Path(row["audio"]) if row["condition"] == "clean" else out / row["audio"]
        if audio.is_file() and sha256_file(audio) == row.get("sha256"):
            done[key] = row
    return done


def _stats(rows: list[dict], conds: list[Condition], out: Path, utts: list[Utterance]) -> dict:
    per = {}
    for cond in conds:
        rs = [r for r in rows if r["condition"] == cond.id]
        entry = {"rows": len(rs), "flagged": sum(1 for r in rs if r["flagged"]),
                 "hours": sum(r["duration"] for r in rs) / 3600}
        if cond.id != "clean":
            entry["bytes"] = sum((out / r["audio"]).stat().st_size for r in rs)
        diffs = [r["ratio_measured_db"] - r["ratio_target_db"] for r in rs
                 if not r["flagged"] and "ratio_measured_db" in r and "ratio_target_db" in r]
        if diffs:
            entry["ratio_error_db"] = {"mean": float(np.mean(diffs)), "max_abs": float(np.max(np.abs(diffs))),
                                       "n": len(diffs)}
        measured = [r["ratio_measured_db"] for r in rs if not r["flagged"] and "ratio_measured_db" in r]
        if measured:
            entry["ratio_measured_db"] = {"mean": float(np.mean(measured)), "std": float(np.std(measured)),
                                          "min": float(np.min(measured)), "max": float(np.max(measured))}
        if cond.kind == "events":
            ev = [r for r in rs if not r["flagged"]]
            if ev:
                entry["events"] = {
                    "per_utterance_mean": float(np.mean([len(r["events"]) for r in ev])),
                    "active_overlap_mean": float(np.mean([r["event_active_overlap"] for r in ev])),
                    "peak_db_re_window_mean": float(np.mean([e["peak_db"] - e["window_level_db"]
                                                             for r in ev for e in r["events"]])),
                    "shifted": sum(e["shifted"] for r in ev for e in r["events"]),
                    "truncated": sum(e["truncated"] for r in ev for e in r["events"])}
        entry["headroom_scaled"] = sum(1 for r in rs if r.get("headroom_scale", 1.0) < 1.0)
        per[cond.id] = entry
    active = defaultdict(list)
    for r in rows:
        if r["condition"] == "clean":
            active[r["set"]].append(r["active_fraction"])
    flagged = Counter(r["flagged"] for r in rows if r["condition"] == "clean" and r["flagged"])
    return {"conditions": per, "flagged": dict(sorted(flagged.items())),
            "active_fraction_by_set": {s: {"mean": float(np.mean(v)), "p10": float(np.percentile(v, 10))}
                                       for s, v in active.items()},
            "utterances_by_set": dict(Counter(u.set for u in utts))}


def build(cfg: BuildConfig) -> dict:
    """Render every condition of every utterance into cfg.out (resumable); returns build.json."""
    t0 = time.time()
    noisy_paths.require_mount()
    if cfg.allow_missing and cfg.sample is None:
        raise BuildError("allow_missing is for sample builds only")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = noisy_sources.load_pool(cfg.pool, cfg.rotor_pool)
    if pool.get("missing") and cfg.sample is None:
        raise BuildError(f"the pool lacks sources {sorted(pool['missing'])}; only sample builds may skip them")
    inputs = noisy_sources.check_pool_inputs(pool, cfg.raw, cfg.verify_dir, max(1, int(cfg.workers)), cfg.registry)
    all_conds = conditions(cfg.rotor_pool is not None)
    conds = [c for c in all_conds if needs_met(c, pool)]
    skipped = [c.id for c in all_conds if c not in conds]
    if skipped and not cfg.allow_missing:
        raise BuildError(f"the pool cannot serve conditions {skipped}")
    if cfg.voxpopuli_speakers is None:
        speakers, cluster_source = load_voxpopuli_speakers()
    else:
        speakers, cluster_source = cfg.voxpopuli_speakers, {"files": [], "note": "speakers given by the caller"}
    clusters = {"voxpopuli_speakers": cluster_source,
                "rules": "librispeech speaker, ami meeting, yodas video, voxpopuli speaker (else session)"}
    utts = load_utterances(cfg.manifests, list(cfg.sets), speakers, cfg.sample)
    names = Counter((u.set, u.name) for u in utts)
    if max(names.values()) > 1:
        raise BuildError("output file name collision")
    codec = dsp.Codec2()
    delay = dsp.measure_codec_delay(codec)
    workers = max(1, int(cfg.workers))
    ctx = mp.get_context("spawn")
    if workers == 1:
        hashes = [noisy_sources.pcm_sha256(Path(u.audio_filepath)) for u in utts]
    else:
        with cf.ProcessPoolExecutor(workers, mp_context=ctx) as ex:
            hashes = list(ex.map(_worker_hash, [u.audio_filepath for u in utts], chunksize=64))
    pcm_hashes = {(u.set, u.id): h for u, h in zip(utts, hashes, strict=True)}
    lock = make_lock(cfg, utts, pool, conds, skipped, delay, pcm_hashes, clusters)
    lock_sha = noisy_paths.sha256_bytes(noisy_paths.canonical_json(lock).encode())
    lock_path = out / "build.lock.json"
    if lock_path.exists():
        if noisy_paths.canonical_json(json.loads(lock_path.read_text())) != noisy_paths.canonical_json(lock):
            raise LockMismatch(f"{lock_path} differs from this build's inputs, code or parameters")
    else:
        noisy_paths.write_json(lock_path, lock)
    final = out / "build_manifest.jsonl"
    partial = out / "build_manifest.partial.jsonl"
    if final.exists() and partial.exists():
        final.unlink()  # interrupted between writing the manifest and removing the journal: the journal rules
    elif final.exists():  # a rerun of a complete build: re-verify every output, redo corrupted ones
        info = out / "build.json"
        recorded = json.loads(info.read_text()).get("manifest_sha256") if info.exists() else None
        if recorded is None or sha256_file(final) != recorded:
            raise ManifestTampered(f"{final} differs from the manifest SHA-256 build.json recorded; refusing to "
                                   "reuse it (inspect it, then remove it to rebuild)")
        noisy_paths.write_text(partial, json.dumps({"lock_sha256": lock_sha}) + "\n" + final.read_text())
        final.unlink()
        info.unlink()
    done = _read_partial(partial, lock_sha, out, expected_rows(utts, conds))
    noisy_paths.write_text(partial, json.dumps({"lock_sha256": lock_sha}) + "\n"
                           + "".join(noisy_paths.canonical_json(r) + "\n" for r in done.values()))
    todo = []
    for u in utts:
        left = [c.id for c in conds if (c.id, u.set, u.id) not in done]
        if left:
            todo.append((u, left))
    log(f"{len(utts)} utterances x {len(conds)} conditions; {len(done)} rows resumed, {len(todo)} utterances to "
        f"render, {workers} workers, codec delay {delay} samples (8 kHz); skipped {skipped}")
    rows = dict(done)
    completed = 0
    with open(partial, "a", encoding="utf-8") as handle:
        def accept(new_rows: list[dict]) -> None:
            nonlocal completed
            for r in new_rows:
                rows[(r["condition"], r["set"], r["id"])] = r
                handle.write(noisy_paths.canonical_json(r) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed += 1
            if completed % 500 == 0:
                log(f"{completed}/{len(todo)} utterances ({time.time() - t0:.0f} s)")
            if cfg.stop_after is not None and completed >= cfg.stop_after:
                raise BuildInterrupted(f"stopped after {completed} utterances (stop_after)")

        if workers == 1:
            renderer = Renderer(pool, out, delay, conds, pcm_hashes)
            for u, left in todo:
                accept(renderer.render(u, left))
        else:
            ex = cf.ProcessPoolExecutor(workers, mp_context=ctx, initializer=_worker_init,
                                        initargs=(pool, str(out), delay, conds, pcm_hashes))
            try:
                pending, queue = set(), iter(todo)
                for item in queue:  # bounded number of tasks in flight
                    pending.add(ex.submit(_worker_render, *item))
                    if len(pending) >= 8 * workers:
                        finished, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                        for f in finished:
                            accept(f.result())
                for f in cf.as_completed(pending):
                    accept(f.result())
            finally:
                ex.shutdown(wait=True, cancel_futures=True)
    order = {c.id: k for k, c in enumerate(conds)}
    set_order = {s: k for k, s in enumerate(cfg.sets)}
    missing = [(c.id, u.set, u.id) for u in utts for c in conds if (c.id, u.set, u.id) not in rows]
    if missing or len(rows) != len(utts) * len(conds):
        raise BuildError(f"{len(missing)} outputs missing after the build")
    ordered = sorted(rows.values(), key=lambda r: (order[r["condition"]], set_order[r["set"]], r["id"]))
    for cond in conds:
        for name in cfg.sets:
            lines = []
            for r in ordered:
                if r["condition"] == cond.id and r["set"] == name:
                    path = r["audio"] if cond.id == "clean" else str((out / r["audio"]).resolve())
                    lines.append(json.dumps({"audio_filepath": path, "duration": r["duration"], "text": r["text"],
                                             "id": r["id"], "source": r["source"], "cluster": r["cluster"]},
                                            ensure_ascii=False))
            noisy_paths.write_text(out / "manifests" / cond.id / f"{name}.jsonl", "".join(x + "\n" for x in lines))
    noisy_paths.write_json(out / "eval_pool_blocklist.json", noisy_blocklist.make_blocklist(pool))
    noisy_paths.write_text(final, "".join(noisy_paths.canonical_json(r) + "\n" for r in ordered))
    summary = {"complete": True, "out": str(out), "lock_sha256": lock_sha, "manifest_sha256": sha256_file(final),
               "conditions": [c.id for c in conds], "skipped_conditions": skipped, "input_check": inputs,
               "rotor": ({"plugin": pool["reserved"]["rotor"]["plugin"], "clips": len(pool["clips"]["rotor"])}
                         if cfg.rotor_pool is not None else UNMEASURED_ROTOR),
               "pool_missing_sources": sorted(pool.get("missing", {})), "sample": cfg.sample,
               "utterances": len(utts), "rows": len(ordered), "rows_resumed": len(done), "workers": workers,
               "codec_delay_samples_8k": delay, "wall_s": time.time() - t0, "finished_utc": utc()}
    summary["statistics"] = _stats(ordered, conds, out, utts)
    summary["audio_bytes"] = sum(e.get("bytes", 0) for e in summary["statistics"]["conditions"].values())
    noisy_paths.write_json(out / "build.json", summary)  # last write
    partial.unlink()
    log(f"done: {len(ordered)} rows, {summary['audio_bytes'] / 2**30:.2f} GiB, {summary['wall_s']:.0f} s")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "status"])
    parser.add_argument("--out", type=Path, help="default: build/ (full) or samples/sample-N (--sample N)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample", type=int, help="N utterances spread over the five sets (never the production path)")
    parser.add_argument("--allow-missing", action="store_true", help="with --sample: skip conditions without sources")
    parser.add_argument("--pool", type=Path, default=noisy_paths.POOL / "pool.json")
    parser.add_argument("--rotor", type=Path, help="a rotor plug-in's pool (adds rotor10/rotor0)")
    args = parser.parse_args(argv)
    noisy_paths.require_mount()
    if args.allow_missing and args.sample is None:
        parser.error("--allow-missing needs --sample")
    out = args.out or (noisy_paths.SAMPLES / f"sample-{args.sample}" if args.sample else noisy_paths.BUILD)
    if args.sample is not None and Path(out).resolve() == noisy_paths.BUILD.resolve():
        parser.error("a sample build may not write the production path")
    if args.command == "status":
        path = Path(out) / "build.json"
        print(path.read_text() if path.exists() else f"{out}: not complete")
        return
    if args.workers > 4:
        parser.error("at most 4 workers while the GPU pilot runs (plans/build.md)")
    build(BuildConfig(out=Path(out), pool=args.pool, rotor_pool=args.rotor, workers=args.workers,
                      sample=args.sample, allow_missing=args.allow_missing))


if __name__ == "__main__":
    sys.exit(main())
