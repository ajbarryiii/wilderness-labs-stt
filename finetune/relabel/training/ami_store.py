"""The AMI distant-microphone (SDM) pair store, as training reads it (keab-prime plan "Data", "AMI distant microphone
(new)": "Storage", "Labels", "Teacher input"). The store is built by ami_sdm.py.

Store directory ($RELABEL_ROOT/corpora/ami-sdm/store): clips.jsonl, one JSON line per pair sorted by id, and
clips.meta.json (clips, hours of SDM audio, clips_sha256 = SHA-256 of clips.jsonl, provisional false, complete true,
and the build's provenance and counts): the synthetic and read-speech stores' format (synthetic.mixing.ClipPool),
with the pair's second recording beside the first. A clip (one pair) has:

    id                      the SDM row's audio_id ("AMI_<meeting>_sdm_<speaker>_<begin cs>_<end cs>"); a training
                            row's id is PREFIX + id
    audio_filepath, samples, duration, sha256, pcm_sha256
                            the SDM recording, the student's input: absolute path of its 16 kHz mono PCM_16 FLAC,
                            sample count, duration = samples / 16000 exactly (so training/batchplan.py clip_samples
                            reads whole samples, the plan's sort key), SHA-256 of the FLAC bytes, SHA-256 of the
                            little-endian int16 samples (labeling/shards.py pcm_sha256)
    ihm_id                  the IHM row's id in the pool, the label store and C1: "ami:" + its audio_id
    ihm_audio_filepath, ihm_samples, ihm_sha256, ihm_pcm_sha256
                            the IHM recording of the same segment, the teacher's input, in the same format
    frames                  encoder frames of both sides (equal by construction; a store with unequal ones is refused)
    meeting_id, speaker_id, begin_time, end_time    the pairing key

No label is stored. A pair's training text is its IHM id's stored label, read through the run's label store
(PairLabels: T*'s for the main arm, B0's under T*'s disposition mask for the control, through the run's cleaning,
repetition rejection and reject list), so a row whose IHM label is rejected at read time has no label.

PairStore (a synthetic.mixing.ClipPool: .clips, .by_id, .identity):
    read(clip_id)           the SDM samples as float32 (int16 / 32768), the student's input
    read_teacher(clip_id)   the IHM samples as float32, the teacher's input
    ihm_id(clip_id)         the IHM id (the label store key)
Each read checks the FLAC file's SHA-256, its sample count and its PCM SHA-256 against clips.jsonl (verify_audio=True,
the default); load checks clips.jsonl against the meta, the meta's completeness and every clip's frames.

encoder_frames(samples): Parakeet TDT 0.6B v2's encoder frames from NeMo's valid mel length (train_b0prime.py
encoder_frames, the stage 3 trainer's rule): samples // 160, then three stride-2 convolutions n -> (n - 1) // 2 + 1.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from labeling import shards
from synthetic import mixing

ALIAS = "ami-sdm"
PREFIX = "ami-sdm:"          # a training row's id: PREFIX + the clip's id
IHM_PREFIX = "ami:"          # the pool's AMI ids (labeling/pool_stream.py: "<source>:<audio_id>")
SAMPLE_RATE = 16000
FORMAT_VERSION = 1
SIDES = {"sdm": "", "ihm": "ihm_"}     # field prefix of each recording's audio fields
AUDIO_FIELDS = ("audio_filepath", "samples", "sha256", "pcm_sha256")
FIELDS = ("id", "audio_filepath", "duration", "samples", "sha256", "pcm_sha256", "frames", "ihm_id",
          "ihm_audio_filepath", "ihm_samples", "ihm_sha256", "ihm_pcm_sha256", "meeting_id", "speaker_id",
          "begin_time", "end_time")


def encoder_frames(samples: int) -> int:
    """Encoder frames of a clip of `samples` samples (module docstring)."""
    n = int(samples) // 160
    for _ in range(3):
        n = (n - 1) // 2 + 1
    return n


def check_clip(c: dict) -> None:
    """ValueError unless the record is a pair of the store's format with equal encoder frames."""
    missing = [k for k in FIELDS if k not in c]
    if missing:
        raise ValueError(f"{c.get('id')}: missing fields {missing}")
    if "label" in c or "text" in c:
        raise ValueError(f"{c['id']}: the pair store holds no label")
    if not str(c["ihm_id"]).startswith(IHM_PREFIX):
        raise ValueError(f"{c['id']}: ihm_id {c['ihm_id']!r} is not a pool AMI id")
    for side in SIDES.values():
        n = c[f"{side}samples"]
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"{c['id']}: bad {side}samples {n!r}")
        if not Path(c[f"{side}audio_filepath"]).is_absolute():
            raise ValueError(f"{c['id']}: {side}audio_filepath is not absolute")
    if c["duration"] != c["samples"] / SAMPLE_RATE:
        raise ValueError(f"{c['id']}: duration is not samples / {SAMPLE_RATE}")
    frames = encoder_frames(c["samples"])
    if frames != encoder_frames(c["ihm_samples"]) or c["frames"] != frames:
        raise ValueError(f"{c['id']}: encoder frames differ (SDM {frames}, IHM {encoder_frames(c['ihm_samples'])}, "
                         f"recorded {c['frames']})")


def write_store(directory: Path, clips: list[dict], meta: dict) -> dict:
    """clips.jsonl (FIELDS, sorted by id) and clips.meta.json; ValueError for a record check_clip refuses or a
    repeated id. Returns the meta."""
    from pool import common
    clips = sorted(clips, key=lambda c: c["id"])
    for c in clips:
        check_clip(c)
    for a, b in zip(clips, clips[1:]):
        if a["id"] == b["id"]:
            raise ValueError(f"clip {a['id']} twice")
    data = "".join(json.dumps({k: c[k] for k in FIELDS}, ensure_ascii=False) + "\n" for c in clips).encode()
    directory = Path(directory)
    common.atomic_write_bytes(directory / "clips.jsonl", data)
    body = {"alias": ALIAS, "format_version": FORMAT_VERSION, "clips": len(clips),
            "hours": sum(c["samples"] for c in clips) / SAMPLE_RATE / 3600,
            "ihm_hours": sum(c["ihm_samples"] for c in clips) / SAMPLE_RATE / 3600,
            "clips_sha256": hashlib.sha256(data).hexdigest(), "provisional": False, "complete": True,
            "sample_rate": SAMPLE_RATE, "audio": "16 kHz mono PCM_16 FLAC, one file per recording (SDM and IHM)",
            "label": "none stored: a pair's label is its IHM id's in the run's label store (training/ami_store.py "
                     "PairLabels)", **meta}
    common.atomic_write_json(directory / "clips.meta.json", body)
    return body


class PairStore(mixing.ClipPool):
    """The pair store as a synthetic.mixing.ClipPool (module docstring)."""

    @classmethod
    def load(cls, directory: Path, verify_audio: bool = True) -> "PairStore":
        directory = Path(directory)
        meta = json.loads((directory / "clips.meta.json").read_text())
        data = (directory / "clips.jsonl").read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != meta.get("clips_sha256"):
            raise RuntimeError(f"{directory}/clips.jsonl does not match clips.meta.json")
        if meta.get("complete") is not True or meta.get("provisional") is not False:
            raise RuntimeError(f"{directory}: the pair store is not complete (clips.meta.json)")
        if meta.get("format_version") != FORMAT_VERSION:
            raise RuntimeError(f"{directory}: pair store format {meta.get('format_version')}, not {FORMAT_VERSION}")
        clips = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        if len(clips) != meta.get("clips"):
            raise RuntimeError(f"{directory}: {len(clips)} clips, the meta says {meta.get('clips')}")
        ids = [c.get("id") for c in clips]
        if ids != sorted(set(ids)):
            raise RuntimeError(f"{directory}: clip ids are not sorted and distinct")
        for c in clips:
            try:
                check_clip(c)
            except ValueError as error:
                raise RuntimeError(f"{directory}: {error}") from error
        store = cls(clips, identity=sha, verify_audio=verify_audio)
        store.meta = meta
        return store

    def _read(self, clip_id: str, side: str) -> np.ndarray:
        c = self.by_id[clip_id]
        path = c[f"{side}audio_filepath"]
        raw = Path(path).read_bytes()
        if self.verify_audio and hashlib.sha256(raw).hexdigest() != c[f"{side}sha256"]:
            raise RuntimeError(f"{path}: SHA-256 differs from clips.jsonl")
        try:
            pcm = shards.decode_flac(raw)
        except Exception as error:
            raise RuntimeError(f"{path}: not a 16 kHz mono FLAC ({error})") from error
        if len(pcm) != c[f"{side}samples"]:
            raise RuntimeError(f"{path}: {len(pcm)} samples, clips.jsonl has {c[f'{side}samples']}")
        if self.verify_audio and shards.pcm_sha256(pcm) != c[f"{side}pcm_sha256"]:
            raise RuntimeError(f"{path}: PCM SHA-256 differs from clips.jsonl")
        return pcm.astype(np.float32) / 32768.0

    def read(self, clip_id: str) -> np.ndarray:
        """The SDM recording (the student's input)."""
        return self._read(clip_id, SIDES["sdm"])

    def read_teacher(self, clip_id: str) -> np.ndarray:
        """The IHM recording of the same segment (the teacher's input)."""
        return self._read(clip_id, SIDES["ihm"])

    def ihm_id(self, clip_id: str) -> str:
        return self.by_id[clip_id]["ihm_id"]


class PairLabels:
    """A pair's training text: its IHM id's text in `labels` (anything with text(id) -> str | None: a
    training.labels.LabelStore or label_sources.StoreLabels of the run's label set, cleaning, repetition rule and
    reject list). A row id is PREFIX + clip id; KeyError for another id. None when the IHM id has no label (T*'s
    mask), in either label set."""

    def __init__(self, store: PairStore, labels, prefix: str = PREFIX):
        self.store, self.labels, self.prefix = store, labels, prefix
        self._ihm = {f"{prefix}{c['id']}": c["ihm_id"] for c in store.clips}
        pairs = sorted(self._ihm.items())
        self.identity = {"kind": "ami pair store", "prefix": prefix, "clip_pool": str(store.identity),
                         "label_of": "the IHM id's label in the run's label store",
                         "ihm_ids_sha256": hashlib.sha256(json.dumps(pairs).encode()).hexdigest()}

    def ihm_id(self, item_id: str) -> str:
        return self._ihm[item_id]

    def text(self, item_id: str) -> str | None:
        return self.labels.text(self._ihm[item_id])
