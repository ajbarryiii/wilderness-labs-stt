"""The audio store and the clip source of the read-speech extra source (b0prime plan "Extra sources" "Audio").

Decoding (decode_bytes): the release's MP3 is decoded the way the parakeet-ternary test sets were (testsets.
decode_audio: soundfile to float64, channel mean, soxr "HQ" to 16 kHz), then quantized here to PCM16 as
clip(rint(x * 32768), -32768, 32767) (the count of clipped samples is recorded) and written losslessly as 16 kHz mono
PCM_16 FLAC. The FLAC is decoded again and must give exactly those samples. Per clip: `samples`, `pcm_sha256`
(SHA-256 of the little-endian int16 samples, as labeling/shards.py), `sha256` (of the FLAC file's bytes) and the
source MP3's SHA-256. A clip whose MP3 does not decode is `undecodable`; one whose decoded length is outside
1-30 s (16,000-480,000 samples) is `decoded_duration`; neither gets a FLAC file.

Chunks (decode.json): the selected clips sorted by id, CHUNK_CLIPS per chunk; decode.json records the selection
table's SHA-256 and each chunk's id-list SHA-256, and every later step refuses a store made from another selection.
A chunk is done when store/chunks/<k>.jsonl exists (written atomically after all its clips) with exactly its ids;
the decode job skips done chunks, so it resumes at a chunk boundary.

The clip source (write_source, after B0's confirmation): store/clips.jsonl, one JSON line per kept clip sorted by id,
in the synthetic store's format (synthetic/roundtrip.py finalize: "id", "audio_filepath" absolute, "duration",
"sha256" of the FLAC bytes, "label") plus "samples", "pcm_sha256", "sentence", "accents", "b0", "b0_wer", and
store/clips.meta.json (clips, hours, clips_sha256, provisional false, and the provenance). duration is samples /
16,000 exactly, so training/batchplan.py clip_samples reads whole samples. ClipPool (a synthetic.mixing.ClipPool)
loads it and reads a clip's audio as float32 int16 / 32768, checking the FLAC SHA-256, the sample count and the PCM
SHA-256.

The label is filters version 3's text and is NOT cleaned: the reader applies cleaning layer 1 alone, at its own
cleaning version, with label_layer1 (training/cleaning.py with no choices artifact, so layer 2 leaves every span as
written; layer 0, which transfers B0's punctuation into T* labels, never applies to this source's human text).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cv27 import layout
from synthetic import mixing

STATUSES = ("ok", "undecodable", "decoded_duration")


def pcm_sha256(pcm: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pcm, dtype="<i2").tobytes()).hexdigest()


def quantize(x: np.ndarray) -> tuple[np.ndarray, int]:
    """(PCM16 samples, number of clipped samples) of float audio in [-1, 1)."""
    y = np.rint(np.asarray(x, dtype=np.float64) * 32768.0)
    clipped = int(np.count_nonzero((y > 32767) | (y < -32768)))
    return np.clip(y, -32768, 32767).astype(np.int16), clipped


def encode_flac(pcm: np.ndarray) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.ascontiguousarray(pcm, dtype=np.int16), layout.SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def decode_flac(data: bytes) -> np.ndarray:
    import soundfile as sf
    pcm, rate = sf.read(io.BytesIO(data), dtype="int16", always_2d=True)
    if rate != layout.SAMPLE_RATE or pcm.shape[1] != 1:
        raise ValueError(f"expected 16 kHz mono FLAC, got {rate} Hz x {pcm.shape[1]}")
    return np.ascontiguousarray(pcm[:, 0])


def decode_bytes(data: bytes) -> tuple[np.ndarray, bytes, int, int, int]:
    """(PCM16 samples, FLAC bytes, source rate, source channels, clipped samples) of one MP3 (module docstring)."""
    import testsets  # finetune/parakeet-ternary
    x, rate, channels = testsets.decode_audio(data)
    pcm, clipped = quantize(x)
    flac = encode_flac(pcm)
    if not np.array_equal(decode_flac(flac), pcm):
        raise RuntimeError("FLAC round trip changed the samples")
    return pcm, flac, int(rate), int(channels), clipped


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "wb") as handle:
        handle.write(data)
    os.replace(tmp, path)


def decode_clip(task: tuple[str, str, str]) -> dict:
    """One clip: (clip id, source MP3 path, target FLAC path) -> its decode record (module docstring)."""
    cid, mp3, target = task
    rec: dict = {"id": cid}
    try:
        data = Path(mp3).read_bytes()
        rec["mp3_sha256"] = hashlib.sha256(data).hexdigest()
        pcm, flac, rate, channels, clipped = decode_bytes(data)
    except Exception as exc:   # a damaged source file is recorded, not fatal
        return rec | {"status": "undecodable", "error": f"{type(exc).__name__}: {exc}"[:300]}
    n = int(len(pcm))
    rec.update(samples=n, source_rate=rate, source_channels=channels, clipped_samples=clipped)
    if not layout.MIN_SAMPLES <= n <= layout.MAX_SAMPLES:
        return rec | {"status": "decoded_duration"}
    _atomic_write(Path(target), flac)
    return rec | {"status": "ok", "audio_filepath": str(target), "duration": n / layout.SAMPLE_RATE,
                  "sha256": hashlib.sha256(flac).hexdigest(), "pcm_sha256": pcm_sha256(pcm)}


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"


# --- chunks --------------------------------------------------------------------------------------------------------

def chunk_ids(ids: Sequence[str], chunk_clips: int = layout.CHUNK_CLIPS) -> list[list[str]]:
    return [list(ids[i:i + chunk_clips]) for i in range(0, len(ids), chunk_clips)]


def ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def plan(paths: layout.Paths, selection_sha256: str, ids: Sequence[str],
         chunk_clips: int = layout.CHUNK_CLIPS) -> dict:
    """Write decode.json for these ids, or verify the existing one is the same plan."""
    from pool import common
    chunks = chunk_ids(ids, chunk_clips)
    body = {"selection_sha256": selection_sha256, "chunk_clips": chunk_clips, "clips": len(ids),
            "chunks": [{"chunk": k, "clips": len(c), "first": c[0], "last": c[-1], "ids_sha256": ids_sha256(c)}
                       for k, c in enumerate(chunks)]}
    if paths.decode_plan.exists():
        old = json.loads(paths.decode_plan.read_text())
        if old != body:
            raise RuntimeError(f"{paths.decode_plan} was made from another selection or chunking; refusing to mix "
                               "stores (move the old store aside)")
        return old
    common.atomic_write_json(paths.decode_plan, body)
    return body


def read_chunk(path: Path, ids: Sequence[str]) -> list[dict] | None:
    """A done chunk's records (exactly `ids`, in order) or None if not done; raises on a mismatching file."""
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [r["id"] for r in rows] != list(ids):
        raise RuntimeError(f"{path}: its clip ids are not the chunk's")
    return rows


def decode_chunks(paths: layout.Paths, rows: Sequence[dict], raw_clips: Path, chunks: Sequence[int], workers: int,
                  deadline: float | None = None, log=print) -> dict:
    """Decode the given chunks (rows: the selected rows sorted by id, as planned); skips done chunks; stops starting
    new chunks after `deadline` (time.monotonic())."""
    import concurrent.futures as cf
    import multiprocessing as mp
    import time

    from cv27 import heldout
    from pool import common
    body = json.loads(paths.decode_plan.read_text())
    by_id = {r["id"]: r for r in rows}
    ids = [r["id"] for r in rows]
    groups = chunk_ids(ids, body["chunk_clips"])
    done = skipped = 0
    executor = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker) \
        if workers > 0 else None
    try:
        for k in chunks:
            if ids_sha256(groups[k]) != body["chunks"][k]["ids_sha256"]:
                raise RuntimeError(f"chunk {k}: ids differ from decode.json")
            if read_chunk(paths.decode_chunk(k), groups[k]) is not None:
                skipped += 1
                continue
            if deadline is not None and time.monotonic() > deadline:
                log(f"decode: budget reached before chunk {k}")
                break
            t0 = time.monotonic()
            tasks = [(cid, str(raw_clips / by_id[cid]["path"]), str(paths.audio(k, heldout.clip_stem(by_id[cid]["path"]))))
                     for cid in groups[k]]
            recs = list(executor.map(decode_clip, tasks, chunksize=8) if executor else map(decode_clip, tasks))
            common.atomic_write_lines(paths.decode_chunk(k), (json.dumps(r, sort_keys=True) for r in recs))
            done += 1
            bad = sum(r["status"] != "ok" for r in recs)
            log(f"decode: chunk {k} ({len(recs)} clips, {bad} not ok) in {time.monotonic() - t0:.1f} s")
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    return {"decoded_chunks": done, "already_done": skipped}


def chunk_records(paths: layout.Paths, rows: Sequence[dict]) -> tuple[list[list[dict] | None], dict]:
    """(records per chunk or None if not done, decode.json), checked against the selected rows."""
    body = json.loads(paths.decode_plan.read_text())
    groups = chunk_ids([r["id"] for r in rows], body["chunk_clips"])
    if len(groups) != len(body["chunks"]):
        raise RuntimeError("the selection does not match decode.json")
    return [read_chunk(paths.decode_chunk(k), g) for k, g in enumerate(groups)], body


# --- the clip source ------------------------------------------------------------------------------------------------

SOURCE_FIELDS = ("id", "audio_filepath", "duration", "samples", "sha256", "pcm_sha256", "label", "sentence",
                 "accents", "b0", "b0_wer")


def write_source(paths: layout.Paths, clips: list[dict], meta: dict) -> dict:
    """clips.jsonl (SOURCE_FIELDS, sorted by id) and clips.meta.json; returns the meta."""
    from pool import common
    clips = sorted(clips, key=lambda c: c["id"])
    data = "".join(json.dumps({k: c[k] for k in SOURCE_FIELDS}, ensure_ascii=False) + "\n" for c in clips).encode()
    common.atomic_write_bytes(paths.clips, data)
    meta = {"alias": layout.ALIAS, "clips": len(clips), "hours": sum(c["samples"] for c in clips) / layout.SAMPLE_RATE / 3600,
            "clips_sha256": hashlib.sha256(data).hexdigest(), "provisional": False, "complete": True,
            "sample_rate": layout.SAMPLE_RATE, "audio": "16 kHz mono PCM_16 FLAC, one file per clip", **meta}
    common.atomic_write_json(paths.clips_meta, meta)
    return meta


class ClipPool(mixing.ClipPool):
    """The clip source as a synthetic.mixing.ClipPool (clips, identity, read)."""

    @classmethod
    def load(cls, directory: Path | None = None, verify_audio: bool = True) -> "ClipPool":
        paths = layout.Paths(layout.out_dir() if directory is None else directory)
        meta = json.loads(paths.clips_meta.read_text())
        data = paths.clips.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != meta["clips_sha256"] or not meta.get("complete") or meta.get("provisional"):
            raise RuntimeError(f"{paths.clips} does not match {paths.clips_meta} (or the source is not complete)")
        clips = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        if len(clips) != meta["clips"]:
            raise RuntimeError(f"{paths.clips}: {len(clips)} clips, the meta says {meta['clips']}")
        return cls(clips, identity=sha, verify_audio=verify_audio)

    def read(self, clip_id: str) -> np.ndarray:
        c = self.by_id[clip_id]
        raw = Path(c["audio_filepath"]).read_bytes()
        if self.verify_audio and hashlib.sha256(raw).hexdigest() != c["sha256"]:
            raise RuntimeError(f"{c['audio_filepath']}: SHA-256 differs from clips.jsonl")
        pcm = decode_flac(raw)
        if len(pcm) != c["samples"] or (self.verify_audio and pcm_sha256(pcm) != c["pcm_sha256"]):
            raise RuntimeError(f"{c['audio_filepath']}: samples differ from clips.jsonl")
        return pcm.astype(np.float32) / 32768.0


def label_layer1(version: int):
    """(clean, identity): cleaning layer 1 alone at cleaning `version`, as a function (label, clip id) -> text.

    training/cleaning.py with no choices artifact: layer 2 leaves every span as written, so only layer 1 changes the
    text (cleaning plan "Read-time order"). Layer 0 needs a T* label and its B0 donor and never applies here."""
    import inspect

    from training import cleaning
    versions = tuple(getattr(cleaning, "VERSIONS", (cleaning.VERSION,)))
    if version not in versions:
        raise ValueError(f"cleaning version {version!r} is not one of {versions}")
    kwargs = {"version": version} if "version" in inspect.signature(cleaning.Cleaning.load).parameters else {}
    spec = cleaning.Cleaning.load(choices_dir=None, **kwargs)
    if getattr(spec, "choices", None) is not None:
        raise RuntimeError("layer 1 alone must not load a choices artifact")

    def clean(label: str, clip_id: str) -> str:
        return spec.clean(label, clip_id, layout.ALIAS)

    return clean, spec.identity() | {"layers": [1]}
