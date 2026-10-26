"""C1's shard format (labeling plan Part C "C1, pool replay"), shared by C1, C2 and the checks.

A shard k covers consumed batches [k * SHARD_BATCHES, (k + 1) * SHARD_BATCHES) and has three files in
<dir>/shards/:

    <k>.journal.parquet   every id occurrence in consumption order: batch, pos (in the batch), id, source,
                          duration (s), first (the id's first occurrence in the whole sequence)
    <k>.audio.parquet     one row per id first seen in this shard: id, source, duration, samples, reference
                          (nullable), reference_kind ("human" | "pseudo" | null), pcm_sha256 (of the int16
                          samples), flac (16 kHz mono PCM16 FLAC of the stream's clean, unaugmented samples)
    <k>.json              written last (the commit marker): batch range, counts, seconds, the two files'
                          SHA-256 and sizes, the stream position after the shard's last batch, counters

The FLAC column is uncompressed by parquet (FLAC is already compressed) and written in row groups of
AUDIO_ROW_GROUP clips, so a writer buffers at most that many encoded clips.

`materialize` writes manifest-based clips (dev sets, the equivalence set) in the same audio format, so C2's
code labels them exactly as it labels C1's shards.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

SR = 16000
AUDIO_ROW_GROUP = 64
AUDIO_COLUMNS = ("id", "source", "duration", "samples", "reference", "reference_kind", "pcm_sha256", "flac")
JOURNAL_COLUMNS = ("batch", "pos", "id", "source", "duration", "first")
REFERENCE_KINDS = (None, "human", "pseudo")


def shard_name(k: int) -> str:
    return f"{int(k):05d}"


def paths(directory: Path, k: int) -> dict[str, Path]:
    base = Path(directory) / "shards"
    name = shard_name(k)
    return {"audio": base / f"{name}.audio.parquet", "journal": base / f"{name}.journal.parquet",
            "meta": base / f"{name}.json"}


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pcm_sha256(pcm: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pcm, dtype="<i2").tobytes()).hexdigest()


def to_int16(audio: np.ndarray) -> np.ndarray:
    """The stream's float32 audio (int16 / 32768, exact) back to its int16 samples, losslessly."""
    x = np.asarray(audio, dtype=np.float64) * 32768.0
    pcm = np.rint(x)
    if not np.array_equal(pcm, x) or pcm.min(initial=0) < -32768 or pcm.max(initial=0) > 32767:
        raise ValueError("audio is not int16 / 32768 (not the stream's PCM16 samples)")
    return pcm.astype(np.int16)


def encode_flac(pcm: np.ndarray) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.ascontiguousarray(pcm, dtype=np.int16), SR, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def decode_flac(data: bytes) -> np.ndarray:
    """int16 samples of a 16 kHz mono FLAC clip."""
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(data), dtype="int16", always_2d=True)
    if sr != SR or x.shape[1] != 1:
        raise ValueError(f"expected 16 kHz mono, got {sr} Hz x {x.shape[1]}")
    return np.ascontiguousarray(x[:, 0])


def _schema():
    import pyarrow as pa
    return pa.schema([("id", pa.string()), ("source", pa.string()), ("duration", pa.float64()),
                      ("samples", pa.int64()), ("reference", pa.string()), ("reference_kind", pa.string()),
                      ("pcm_sha256", pa.string()), ("flac", pa.binary())])


def _journal_schema():
    import pyarrow as pa
    return pa.schema([("batch", pa.int32()), ("pos", pa.int32()), ("id", pa.string()), ("source", pa.string()),
                      ("duration", pa.float64()), ("first", pa.bool_())])


class AudioWriter:
    """Streams audio rows into a temporary parquet file; commit() renames it and returns (sha256, bytes)."""

    def __init__(self, path: Path):
        import pyarrow.parquet as pq
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(f"{self.path.name}.tmp{os.getpid()}")
        self.writer = pq.ParquetWriter(self.tmp, _schema(), compression={c: ("NONE" if c == "flac" else "ZSTD")
                                                                         for c in AUDIO_COLUMNS},
                                       use_dictionary=["source", "reference_kind"])
        self.rows: list[dict] = []
        self.count = 0
        self.seconds = 0.0

    def add(self, row: dict) -> None:
        if row["reference_kind"] not in REFERENCE_KINDS or (row["reference"] is None) != (row["reference_kind"] is None):
            raise ValueError(f"{row['id']}: bad reference / reference_kind")
        self.rows.append(row)
        self.count += 1
        self.seconds += row["duration"]
        if len(self.rows) >= AUDIO_ROW_GROUP:
            self.flush()

    def flush(self) -> None:
        import pyarrow as pa
        if self.rows:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=_schema()))
            self.rows = []

    def commit(self) -> tuple[str, int]:
        self.flush()
        self.writer.close()
        with open(self.tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(self.tmp, self.path)
        return sha256_file(self.path), self.path.stat().st_size

    def abort(self) -> None:
        try:
            self.writer.close()
        except Exception:
            pass
        self.tmp.unlink(missing_ok=True)


def write_journal(path: Path, rows: list[dict]) -> tuple[str, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows, schema=_journal_schema()), tmp, compression="zstd")
    with open(tmp, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return sha256_file(path), path.stat().st_size


def write_meta(path: Path, meta: dict) -> None:
    from pool import common
    common.atomic_write_json(path, meta)


def read_journal(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def read_audio(path: Path, columns: Iterable[str] | None = None) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=list(columns) if columns else None).to_pylist()


def iter_audio(path: Path, columns: Iterable[str] | None = None, batch_rows: int = AUDIO_ROW_GROUP) -> Iterator[dict]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=list(columns) if columns else None):
        yield from batch.to_pylist()


def committed(directory: Path) -> list[int]:
    """Shard indices with a commit marker, sorted."""
    base = Path(directory) / "shards"
    if not base.exists():
        return []
    return sorted(int(p.name.split(".")[0]) for p in base.glob("*.json") if p.name.split(".")[0].isdigit())


def load_meta(directory: Path, k: int) -> dict:
    return json.loads(paths(directory, k)["meta"].read_text())


def verify_shard(directory: Path, k: int, audio_hash: bool = True) -> dict:
    """The shard's meta after checking its files against it: SHA-256 of the journal, and of the audio file
    unless audio_hash is False (then only its size; hashing ~1 TB of audio is left to the gate)."""
    meta = load_meta(directory, k)
    p = paths(directory, k)
    for key in ("audio", "journal"):
        if key not in meta["files"] or (key == "audio" and meta.get("audio_deleted")):
            continue
        if key == "audio" and not audio_hash:
            if p[key].stat().st_size != meta["files"][key]["bytes"]:
                raise RuntimeError(f"shard {shard_name(k)}: audio size differs from its meta")
        elif sha256_file(p[key]) != meta["files"][key]["sha256"]:
            raise RuntimeError(f"shard {shard_name(k)}: {key} SHA-256 differs from its meta")
    return meta


def remove_uncommitted(directory: Path) -> list[str]:
    """Delete shard files without a commit marker and temporary files (an interrupted shard)."""
    base = Path(directory) / "shards"
    removed = []
    if not base.exists():
        return removed
    done = {shard_name(k) for k in committed(directory)}
    for p in base.iterdir():
        stem = p.name.split(".")[0]
        if ".tmp" in p.name or (stem.isdigit() and stem not in done):
            p.unlink()
            removed.append(p.name)
    return removed


def materialize(records: list[dict], directory: Path, shard_size: int, source_of=None) -> list[int]:
    """Write manifest records (audio_filepath, id, duration, text; optional source) as audio shards of
    shard_size clips (in the given order) with commit markers (kind "materialized", no journal). The
    reference is the manifest text, kind "human" (dev sets and corpora carry human transcripts)."""
    import soundfile as sf
    directory = Path(directory)
    ks = []
    for k, start in enumerate(range(0, len(records), shard_size)):
        p = paths(directory, k)
        if p["meta"].exists():
            ks.append(k)
            continue
        writer = AudioWriter(p["audio"])
        try:
            for r in records[start:start + shard_size]:
                info = sf.info(r["audio_filepath"])
                if info.samplerate != SR or info.channels != 1 or info.subtype != "PCM_16":
                    raise ValueError(f"{r['audio_filepath']}: expected 16 kHz mono PCM16")
                x, _ = sf.read(r["audio_filepath"], dtype="int16", always_2d=True)
                pcm = np.ascontiguousarray(x[:, 0])
                text = r.get("text")
                text = text if isinstance(text, str) and text.strip() else None
                writer.add({"id": r["id"], "source": source_of(r) if source_of else r.get("source", ""),
                            "duration": len(pcm) / SR, "samples": len(pcm), "reference": text,
                            "reference_kind": "human" if text else None, "pcm_sha256": pcm_sha256(pcm),
                            "flac": encode_flac(pcm)})
            sha, size = writer.commit()
        except BaseException:
            writer.abort()
            raise
        write_meta(p["meta"], {"shard": k, "kind": "materialized", "new_ids": writer.count,
                               "audio_seconds_new": writer.seconds,
                               "files": {"audio": {"name": p["audio"].name, "sha256": sha, "bytes": size}}})
        ks.append(k)
    return ks
