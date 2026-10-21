"""Development sets for the ternary Parakeet experiment (stored locally).

See DESIGN.md "Evaluation" (development sets). Training data is streamed from
the Hugging Face Hub during training (stream.py) and is not stored here.

Output contract (other modules code against it):

* Audio: 16 kHz mono 16-bit FLAC, one file per utterance, under
  paths.AUDIO/<source>/... . LibriSpeech dev-clean is reused in place from the
  Whisper experiment (already 16 kHz FLAC); the manifest points there.
* Manifests: paths.MANIFESTS/<name>.jsonl, one JSON object per line with exactly
  the keys audio_filepath (absolute), duration (seconds), text (reference
  transcript), id (unique, "<source>:<corpus id>"), source; sorted by id. Next to
  each, <name>.meta.json records provenance, counts, hours, the drop count per
  reason, and the SHA-256 of the .jsonl.
* Development sets are complete: only undecodable, empty, non-finite or all-zero
  audio and empty reference transcripts are dropped (no duration filter), except
  yodas_dev, which samples utterances of 1-30 s like the training filter.
* yodas_dev: 2,000 utterances from 16 English asr_only shards reserved by a
  seeded sample over the sorted shard list (paths.MANIFESTS/yodas_reserved_shards.json).
  Only 4 seeded row groups per reserved shard are fetched (HTTP range reads), and
  only the chosen utterances' audio is stored. stream.py never reads the reserved
  shards and drops any utterance whose source recording appears in them.
* <name>_400.jsonl: every k-th utterance of the id-sorted full set.

CLI (detached for long work; resumable, AMI/VoxPopuli shards have done markers):

    ./python data.py dev       # all development manifests and their 400-utterance subsets
    ./python data.py status
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

import paths

SR = paths.SAMPLE_RATE
MIN_SECONDS, MAX_SECONDS = 1.0, 30.0
MANIFEST_KEYS = ("audio_filepath", "duration", "text", "id", "source")
DEV_SUBSET = 400
YODAS_RESERVED_SHARDS = 16
YODAS_DEV_UTTERANCES = 2000
YODAS_DEV_ROW_GROUPS = 4  # seeded row groups read per reserved shard (~100 rows each)
WORKERS = 8
# Independent seeded RNG streams: np.random.default_rng([paths.SEED, stream]).
STREAMS = {"yodas_reserved": 1, "yodas_dev": 3}

HF_SOURCES = {
    "ami": {"repo": "edinburghcstr/ami", "revision": "46f28f2503e2ec48f8867a84eef356c70476beab",
            "config": "ihm", "prefix": "ihm/", "id_col": "audio_id", "text_col": "text",
            "group_col": "meeting_id", "license": "CC-BY-4.0",
            "text_kind": "human transcript (uppercase, no punctuation)"},
    "voxpopuli": {"repo": "facebook/voxpopuli", "revision": "42f01879c780b4a2e90ec0b4f616c2ece526e4f1",
                  "config": "en", "prefix": "en/", "id_col": "audio_id", "text_col": "normalized_text",
                  "group_col": None, "license": "CC0-1.0",
                  "text_kind": "human transcript, normalized_text column (lowercase, numbers "
                               "spelled out; the column the ESB VoxPopuli test set uses)"},
    "peoples_speech": {"repo": "MLCommons/peoples_speech",
                       "revision": "f10597c5d3d3a63f8b6827701297c3afdf178272", "config": "clean",
                       "prefix": "clean/", "id_col": "id", "text_col": "text", "group_col": None,
                       "license": "CC-BY (source-dependent 2.0/2.5/3.0/4.0; keep attribution records)",
                       "text_kind": "human captions force-aligned by MLCommons (lowercase)"},
    "yodas": {"repo": "espnet/yodas-granary", "revision": "969944574ea3f37890beaf67ea651e160cfaf043",
              "config": "English", "prefix": "data/en", "id_col": "utt_id", "text_col": "text",
              "group_col": "original_audio_id", "license": "CC-BY-3.0",
              "text_kind": "Granary pseudo-label (faster-whisper-large-v3 + Qwen2.5-7B punctuation "
                           "and capitalization restoration), not a human transcript"},
}

LS_URL = "https://openslr.elda.org/resources/12/{split}.tar.gz"  # ELDA mirror of OpenSLR SLR12
LS_LICENSE = "CC-BY-4.0 (OpenSLR SLR12, Panayotov et al. 2015)"
_WT = Path("/mnt/hd/wilderness-labs-stt/whisper-ternary/data/LibriSpeech")
LS_REUSED = {"dev-clean": _WT / "dev-clean"}
LS_EXPECTED = {"dev-clean": 2703, "dev-other": 2864}
# SHA-256 of the SLR12 tarball as listed in torchaudio's LibriSpeech loader. A mismatch
# is recorded, not fatal: gzip CRCs and the exact utterance count also verify the data.
LS_KNOWN_SHA256 = {"dev-other": "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365"}
LS_DEV = {"librispeech_dev_clean": "dev-clean", "librispeech_dev_other": "dev-other"}


def _records_dir() -> Path:
    return paths.DATA / "records"


def _hf_dir() -> Path:
    return paths.DATA / "hf"


def _log(msg: str) -> None:
    print(f"{dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%M:%SZ} {msg}", flush=True)


def _now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------------------------
# Files, manifests
# ---------------------------------------------------------------------------------------------

def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: object) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def manifest_path(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError(f"invalid manifest name {name!r}")
    return paths.MANIFESTS / f"{name}.jsonl"


def meta_path(name: str) -> Path:
    return paths.MANIFESTS / f"{name}.meta.json"


def _check_row(row: dict) -> dict:
    if set(row) != set(MANIFEST_KEYS):
        raise ValueError(f"manifest row keys {sorted(row)} != {sorted(MANIFEST_KEYS)}")
    if not isinstance(row["id"], str) or not row["id"]:
        raise ValueError(f"bad id {row['id']!r}")
    if not isinstance(row["audio_filepath"], str) or not os.path.isabs(row["audio_filepath"]):
        raise ValueError(f"{row['id']}: audio_filepath must be an absolute path")
    if not isinstance(row["duration"], float) or not math.isfinite(row["duration"]) or row["duration"] <= 0:
        raise ValueError(f"{row['id']}: bad duration {row['duration']!r}")
    if row["text"] is not None and not isinstance(row["text"], str):
        raise ValueError(f"{row['id']}: text must be a string or null")
    return {k: row[k] for k in MANIFEST_KEYS}


def write_manifest(name: str, rows: Iterable[dict], meta: dict | None = None) -> dict:
    """Write MANIFESTS/<name>.jsonl (validated, sorted by id) and its .meta.json; return the meta."""
    paths.require_mount()
    lines, seconds, sources = [], 0.0, set()
    for row in rows:  # (id, json line) pairs keep ~10M-row manifests within a few GB of RAM
        row = _check_row(row)
        seconds += row["duration"]
        sources.add(row["source"])
        lines.append((row["id"], json.dumps(row, ensure_ascii=False)))
    lines.sort(key=lambda pair: pair[0])
    for a, b in zip(lines, lines[1:]):
        if a[0] == b[0]:
            raise ValueError(f"{name}: duplicate id {a[0]!r}")
    path = manifest_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        for _, line in lines:
            handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    count = len(lines)
    del lines
    full = {"name": name, "path": str(path), "utterances": count, "audio_files": count,
            "hours": round(seconds / 3600, 4), "sources": sorted(sources),
            **(meta or {}), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "created": _now(), "keys": list(MANIFEST_KEYS)}
    atomic_write_json(meta_path(name), full)
    return full


def read_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_manifest(name: str) -> list[dict]:
    """Rows of MANIFESTS/<name>.jsonl, sorted by id (read-only helper for other modules)."""
    return read_manifest(manifest_path(name))


def manifest_ids(name: str) -> set[str]:
    with open(manifest_path(name), encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}


def load_meta(name: str) -> dict:
    return json.loads(meta_path(name).read_text())


# ---------------------------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------------------------

def duration_drop_reason(duration: float | None, lo: float = MIN_SECONDS,
                         hi: float = MAX_SECONDS) -> str | None:
    """None if lo <= duration <= hi (both inclusive), else the drop reason."""
    if duration is None or not math.isfinite(duration) or duration <= 0:
        return "empty_audio"
    if duration < lo:
        return "too_short"
    if duration > hi:
        return "too_long"
    return None


def every_kth(rows: Sequence[dict], n: int = DEV_SUBSET) -> list[dict]:
    """Deterministic subset: every k-th row of the id-sorted list, k = len // n, first n."""
    ordered = sorted(rows, key=lambda r: r["id"])
    if n <= 0:
        raise ValueError("subset size must be positive")
    if len(ordered) <= n:
        return ordered
    return ordered[::len(ordered) // n][:n]


def rng(stream: str, seed: int = paths.SEED) -> np.random.Generator:
    return np.random.default_rng([seed, STREAMS[stream]])


def reserve_shards(names: Sequence[str], n: int, seed: int = paths.SEED) -> list[int]:
    """Sorted indices into sorted(names) of n shards reserved for development."""
    count = len(sorted(set(names)))
    if count != len(names) or not 0 < n < count:
        raise ValueError("names must be unique and n in 1..len-1")
    return sorted(int(i) for i in rng("yodas_reserved", seed).choice(count, size=n, replace=False))


def filter_records(records: Iterable[dict], source: str, duration_filter: bool,
                   exclude_groups: set[str] | None = None) -> tuple[list[dict], dict[str, int]]:
    """Manifest rows from extraction records, plus the drop count per reason."""
    rows, drops, seen = [], collections.Counter(), set()
    for rec in records:
        reason = rec.get("drop")
        if reason is None and rec.get("audio_filepath") is None:
            reason = "no_audio_file"
        if reason is None and exclude_groups and rec.get("group") in exclude_groups:
            reason = "dev_recording_overlap"
        if reason is None and duration_filter:
            reason = duration_drop_reason(rec["duration"])
        if reason is None and rec["id"] in seen:
            reason = "duplicate_id"
        if reason:
            drops[reason] += 1
            continue
        seen.add(rec["id"])
        rows.append({"audio_filepath": rec["audio_filepath"], "duration": float(rec["duration"]),
                     "text": rec.get("text") or None, "id": rec["id"], "source": source})
    return rows, dict(sorted(drops.items()))


# ---------------------------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------------------------

def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """Band-limited resampling of a 1-D float signal (soxr VHQ; scipy polyphase fallback)."""
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out:
        return x
    try:
        import soxr
        return np.asarray(soxr.resample(x, sr_in, sr_out, quality="VHQ"), dtype=np.float32)
    except ImportError:
        from scipy.signal import resample_poly
        g = math.gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(x, dtype=np.float64) * 32768.0), -32768, 32767).astype(np.int16)


def convert(data: bytes) -> tuple[np.ndarray | None, str | None, bool]:
    """Decode encoded audio to 16 kHz mono int16.

    Returns (pcm, drop_reason, passthrough); passthrough means the input is already
    16 kHz mono 16-bit FLAC and may be stored byte for byte.
    """
    try:
        info = sf.info(io.BytesIO(data))
        dtype = "int16" if info.subtype == "PCM_16" else "float32"
        x, sr = sf.read(io.BytesIO(data), dtype=dtype, always_2d=True)
    except Exception:
        return None, "decode_error", False
    if x.shape[0] == 0:
        return None, "empty_audio", False
    if x.dtype != np.int16 and not np.isfinite(x).all():
        return None, "nonfinite_audio", False
    passthrough = info.format == "FLAC" and info.subtype == "PCM_16" and sr == SR and x.shape[1] == 1
    if x.shape[1] == 1 and sr == SR:
        pcm = x[:, 0] if x.dtype == np.int16 else to_int16(x[:, 0])
    else:
        mono = x.astype(np.float32).mean(axis=1)
        if x.dtype == np.int16:
            mono /= 32768.0
        pcm = to_int16(resample(mono, sr))
    if pcm.size == 0:
        return None, "empty_audio", False
    if not pcm.any():
        return None, "silent_audio", False
    return np.ascontiguousarray(pcm), None, passthrough


def write_flac(out: Path, pcm: np.ndarray, original: bytes | None = None) -> None:
    """Atomically write 16 kHz mono 16-bit FLAC (the original bytes when given)."""
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}")
    if original is not None:
        tmp.write_bytes(original)
    else:
        sf.write(tmp, pcm, SR, format="FLAC", subtype="PCM_16")
    os.replace(tmp, out)


def safe_name(native_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", native_id).strip(".")
    if not name:
        raise ValueError(f"cannot derive a file name from {native_id!r}")
    return name[:200]


# ---------------------------------------------------------------------------------------------
# Hugging Face parquet sources
# ---------------------------------------------------------------------------------------------

def list_hf_files(source: str, split: str) -> list[dict]:
    """[{path, size, sha256}] of the source's parquet shards for split, sorted; cached on disk."""
    spec = HF_SOURCES[source]
    cache = _hf_dir() / f"{source}.{split}.files.json"
    if cache.exists():
        return json.loads(cache.read_text())["files"]
    from huggingface_hub import HfApi
    api = HfApi()
    kw = {"repo_type": "dataset", "revision": spec["revision"]}
    if source == "yodas":
        subsets = [f.path for f in api.list_repo_tree(spec["repo"], path_in_repo="data", **kw)
                   if f.path.startswith(spec["prefix"])]
        entries = [f for sub in subsets
                   for f in api.list_repo_tree(spec["repo"], path_in_repo=f"{sub}/{split}", **kw)]
    else:
        entries = [f for f in api.list_repo_tree(spec["repo"], path_in_repo=spec["prefix"].rstrip("/"), **kw)
                   if f.path.split("/")[-1].startswith(f"{split}-")]
    files = sorted(({"path": f.path, "size": f.size, "sha256": f.lfs.sha256 if f.lfs else None}
                    for f in entries if f.path.endswith(".parquet")), key=lambda f: f["path"])
    if not files:
        raise RuntimeError(f"no parquet files for {source} {split}")
    atomic_write_json(cache, {"repo": spec["repo"], "revision": spec["revision"], "split": split,
                              "listed": _now(), "files": files})
    return files


def shard_key(path: str) -> str:
    parts = Path(path).with_suffix("").parts
    if parts[0] == "data":  # YODAS: data/en000/asr_only/00000002.parquet -> en000-00000002
        return f"{parts[1]}-{parts[-1]}"
    return parts[-1]


def shard_job(source: str, split: str, entry: dict) -> dict:
    """Description of one development shard's extraction."""
    key = shard_key(entry["path"])
    base = _records_dir() / source / split
    return {"source": source, "path": entry["path"], "size": entry["size"], "sha256": entry["sha256"],
            "key": key, "out_dir": str(paths.AUDIO / source / split / key),
            "records": str(base / f"{key}.jsonl"), "done": str(base / f"{key}.done.json")}


def download_hf(source: str, path: str, sha256: str | None, size: int | None,
                attempts: int = 6) -> Path:
    """Local copy of one repository file at the pinned revision, size- and SHA-256-verified."""
    from huggingface_hub import hf_hub_download
    spec = HF_SOURCES[source]
    local_dir = _hf_dir() / spec["repo"].replace("/", "__")
    for attempt in range(attempts):
        try:
            local = Path(hf_hub_download(spec["repo"], path, repo_type="dataset",
                                         revision=spec["revision"], local_dir=local_dir))
            if size is not None and local.stat().st_size != size:
                raise OSError(f"{path}: size {local.stat().st_size} != {size}")
            if sha256 is not None and sha256_file(local) != sha256:
                raise OSError(f"{path}: SHA-256 mismatch")
            return local
        except Exception as exc:  # network errors, truncated files
            (local_dir / path).unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
            wait = min(600, 15 * 2 ** attempt)
            _log(f"download {source} {path} failed ({exc!r}); retry {attempt + 1} in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(2)


def extract_shard(job: dict) -> dict:
    """Download one development parquet shard and write its utterances as FLAC.

    Every row gets a record (kept, or with its drop reason); every decodable row
    with a reference transcript is written. Idempotent via the done marker.
    """
    done = Path(job["done"])
    if done.exists():
        return json.loads(done.read_text())
    import pyarrow.parquet as pq
    t0 = time.time()
    source, spec = job["source"], HF_SOURCES[job["source"]]
    local = download_hf(source, job["path"], job["sha256"], job["size"])
    t_download = time.time() - t0
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [spec["id_col"], spec["text_col"], "audio"] + ([spec["group_col"]] if spec["group_col"] else [])
    records, drops, names = [], collections.Counter(), set()
    raw_seconds = kept_seconds = 0.0
    rows_total = 0
    for batch in pq.ParquetFile(local).iter_batches(batch_size=32, columns=cols, use_threads=False):
        for row in batch.to_pylist():
            rows_total += 1
            native = str(row[spec["id_col"]])
            uid = f"{source}:{native}"
            text = row[spec["text_col"]]
            text = text.strip() if isinstance(text, str) else ""
            group = row[spec["group_col"]] if spec["group_col"] else None
            rec = {"id": uid, "audio_filepath": None, "duration": None, "text": text or None,
                   "group": group, "drop": None}
            data = (row.get("audio") or {}).get("bytes")
            reason = None
            if not data:
                reason = "empty_audio"
            else:
                try:
                    rec["duration"] = sf.info(io.BytesIO(data)).duration
                    raw_seconds += rec["duration"]
                except Exception:
                    reason = "decode_error"
            if reason is None and not text:
                reason = "empty_text"
            if reason is None:
                pcm, reason, passthrough = convert(data)
            if reason is None:
                fname = safe_name(native) + ".flac"
                if fname in names:
                    raise ValueError(f"{job['path']}: file name collision for {native!r}")
                names.add(fname)
                rec["duration"] = pcm.size / SR
                out = out_dir / fname
                write_flac(out, pcm, data if passthrough else None)
                rec["audio_filepath"] = str(out)
                kept_seconds += rec["duration"]
            rec["drop"] = reason
            if reason:
                drops[reason] += 1
            records.append(rec)
    atomic_write_text(Path(job["records"]),
                      "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    summary = {"source": source, "path": job["path"], "key": job["key"], "bytes": job["size"],
               "sha256": job["sha256"], "rows": rows_total, "records": len(records),
               "kept": len(records) - sum(drops.values()), "raw_hours": raw_seconds / 3600,
               "kept_hours": kept_seconds / 3600, "drops": dict(sorted(drops.items())),
               "download_seconds": round(t_download, 1), "seconds": round(time.time() - t0, 1),
               "finished": _now()}
    atomic_write_json(done, summary)
    return summary


def read_records(job: dict) -> list[dict]:
    return read_manifest(Path(job["records"]))


def run_shards(jobs: Sequence[dict], workers: int, label: str) -> None:
    """Extract every job (each worker process downloads and encodes one shard); skips finished ones."""
    todo = [j for j in jobs if not Path(j["done"]).exists()]
    if not todo:
        return
    with cf.ProcessPoolExecutor(min(workers, len(todo)), mp_context=mp.get_context("spawn"),
                                initializer=_worker_init) as pool:
        for summary in pool.map(extract_shard, todo):
            _log(f"{label} {summary['key']}: {summary['kept']}/{summary['rows']} kept, "
                 f"{summary['kept_hours']:.2f} h, {summary['seconds']:.0f}s")


def hf_meta(source: str, split: str, jobs: Sequence[dict], extra: dict | None = None) -> dict:
    spec = HF_SOURCES[source]
    summaries = [json.loads(Path(j["done"]).read_text()) for j in jobs]
    return {"source": source, "repo": spec["repo"], "config": spec["config"], "split": split,
            "revision": spec["revision"], "license": spec["license"], "text_field": spec["text_col"],
            "text_kind": spec["text_kind"], "raw_files": len(jobs),
            "raw_bytes": sum(s["bytes"] or 0 for s in summaries),
            "raw_rows": sum(s["rows"] for s in summaries),
            "raw_hours": round(sum(s["raw_hours"] for s in summaries), 4),
            "raw_file_list": [j["path"] for j in jobs], **(extra or {})}


def hf_manifest(name: str, source: str, split: str, jobs: Sequence[dict], extra: dict | None = None) -> dict:
    records = (r for j in jobs for r in read_records(j))
    rows, drops = filter_records(records, source, duration_filter=False)
    meta = hf_meta(source, split, jobs, extra)
    meta.update(kind="dev", dropped=drops, filters=_filters(False))
    out = write_manifest(name, rows, meta)
    _log(f"manifest {name}: {out['utterances']} utterances, {out['hours']:.2f} h, dropped {drops}")
    return out


def _filters(train: bool) -> dict:
    if train:
        return {"duration_seconds": [MIN_SECONDS, MAX_SECONDS], "bounds": "inclusive",
                "audio": "decodable, nonempty, finite, not all zeros"}
    return {"duration_seconds": None, "audio": "decodable, nonempty, finite, not all zeros",
            "text": "nonempty reference transcript"}


# ---------------------------------------------------------------------------------------------
# LibriSpeech (OpenSLR SLR12 tarballs)
# ---------------------------------------------------------------------------------------------

def ls_dir(split: str) -> Path:
    return LS_REUSED.get(split) or paths.AUDIO / "librispeech" / "LibriSpeech" / split


def ensure_librispeech(split: str) -> Path:
    """Download (curl, resumable) and extract an SLR12 split that is not already on disk."""
    root = ls_dir(split)
    if split in LS_REUSED:
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root
    marker = root.with_name(root.name + ".done.json")
    if marker.exists():
        return root
    paths.require_mount()
    tar_dir = paths.DATA / "openslr"
    tar_dir.mkdir(parents=True, exist_ok=True)
    tarball, part = tar_dir / f"{split}.tar.gz", tar_dir / f"{split}.tar.gz.part"
    while part.exists() and not tarball.exists() and time.time() - part.stat().st_mtime < 120:
        _log(f"waiting for an in-progress download of {part}")  # started by another process
        time.sleep(60)
    if not tarball.exists():
        url = LS_URL.format(split=split)
        _log(f"downloading {url}")
        for attempt in range(6):
            if subprocess.run(["curl", "-fsSL", "--retry", "5", "-C", "-", "-o", str(part), url]).returncode == 0:
                break
            time.sleep(30)
        else:
            raise RuntimeError(f"download of {url} failed")
        os.replace(part, tarball)
    digest = sha256_file(tarball)
    _log(f"{tarball.name} sha256 {digest}")
    staging = root.parent.parent / f".extract-{split}"
    if staging.exists():
        subprocess.run(["rm", "-rf", str(staging)], check=True)
    staging.mkdir(parents=True)
    subprocess.run(["tar", "-xzf", str(tarball), "-C", str(staging)], check=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging / "LibriSpeech" / split, root)
    for extra in (staging / "LibriSpeech").iterdir():  # SPEAKERS.TXT etc.
        target = root.parent / extra.name
        if not target.exists():
            os.replace(extra, target)
    subprocess.run(["rm", "-rf", str(staging)], check=True)
    flacs = sum(1 for _ in root.glob("*/*/*.flac"))
    if flacs != LS_EXPECTED[split]:
        raise RuntimeError(f"{split}: {flacs} flac files, expected {LS_EXPECTED[split]}")
    atomic_write_json(marker, {"split": split, "url": LS_URL.format(split=split), "tarball": str(tarball),
                               "tarball_bytes": tarball.stat().st_size, "tarball_sha256": digest,
                               "known_sha256": LS_KNOWN_SHA256.get(split),
                               "sha256_matches_known": digest == LS_KNOWN_SHA256.get(split),
                               "flac_count": flacs, "extracted": _now()})
    return root


def _scan_chapter(chapter: str) -> list[dict]:
    out = []
    for trans in sorted(Path(chapter).glob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            if not line.strip():
                continue
            uid, text = line.split(" ", 1)
            flac = Path(chapter) / f"{uid}.flac"
            try:
                info = sf.info(str(flac))
            except Exception:
                out.append({"id": f"librispeech:{uid}", "audio_filepath": None, "duration": None,
                            "text": text.strip() or None, "group": None, "drop": "decode_error"})
                continue
            if info.samplerate != SR or info.channels != 1:
                raise ValueError(f"{flac}: {info.samplerate} Hz x{info.channels}, expected {SR} Hz mono")
            out.append({"id": f"librispeech:{uid}", "audio_filepath": str(flac.absolute()),
                        "duration": info.frames / info.samplerate, "text": text.strip() or None,
                        "group": uid.rsplit("-", 1)[0], "drop": None if info.frames else "empty_audio"})
    return out


def librispeech_records(split: str, workers: int = 8) -> list[dict]:
    """Records for one split (cached in DATA/records/librispeech/<split>.jsonl)."""
    path = _records_dir() / "librispeech" / f"{split}.jsonl"
    done = path.with_suffix(".done.json")
    if done.exists():
        return read_manifest(path)
    root = ensure_librispeech(split)
    chapters = sorted(str(c) for c in root.glob("*/*") if c.is_dir())
    with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        records = [r for part in pool.map(_scan_chapter, chapters, chunksize=16) for r in part]
    records.sort(key=lambda r: r["id"])
    if len(records) != LS_EXPECTED[split]:
        raise RuntimeError(f"{split}: {len(records)} transcripts, expected {LS_EXPECTED[split]}")
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    atomic_write_json(done, {"split": split, "root": str(root), "records": len(records),
                             "hours": sum(r["duration"] or 0 for r in records) / 3600, "finished": _now()})
    return records


def ls_meta(splits: Sequence[str], extra: dict | None = None) -> dict:
    provenance = {}
    for split in splits:
        marker = ls_dir(split).with_name(ls_dir(split).name + ".done.json")
        provenance[split] = {"directory": str(ls_dir(split)), "url": LS_URL.format(split=split),
                             **(json.loads(marker.read_text()) if marker.exists() else
                                {"note": "reused in place from an earlier experiment on this disk"})}
    return {"source": "librispeech", "repo": "OpenSLR SLR12", "config": None, "split": list(splits),
            "revision": "SLR12 tarballs (ELDA mirror)", "license": LS_LICENSE, "text_field": "trans.txt",
            "text_kind": "human transcript (uppercase, no punctuation)", "raw_files": len(splits),
            "splits": provenance, **(extra or {})}


# ---------------------------------------------------------------------------------------------
# Development sets
# ---------------------------------------------------------------------------------------------

def reserved_path() -> Path:
    return paths.MANIFESTS / "yodas_reserved_shards.json"


def yodas_reserved() -> tuple[list[dict], list[int]]:
    files = list_hf_files("yodas", "asr_only")
    return files, reserve_shards([f["path"] for f in files], YODAS_RESERVED_SHARDS)


def _remote_parquet(source: str, path: str):
    """(ParquetFile, handle) for a repository file at the pinned revision, read by HTTP range requests."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    spec = HF_SOURCES[source]
    handle = HfFileSystem().open(f"datasets/{spec['repo']}@{spec['revision']}/{path}", "rb",
                                 block_size=8 * 2**20)
    return pq.ParquetFile(handle), handle


def _retry(fn, what: str, attempts: int = 6):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            wait = min(600, 15 * 2 ** attempt)
            _log(f"{what} failed ({exc!r}); retry {attempt + 1} in {wait}s")
            time.sleep(wait)


def build_yodas_dev() -> dict:
    """yodas_dev: 2,000 utterances (1-30 s, nonempty text) from seeded row groups of the reserved shards.

    Reads the small metadata columns of every reserved shard (to record all of
    their recordings), then the audio of YODAS_DEV_ROW_GROUPS seeded row groups
    per shard; stores only the chosen utterances.
    """
    paths.require_mount()
    files, reserved = yodas_reserved()
    gen = rng("yodas_dev")
    candidates, recordings, plan = [], set(), {}
    meta_cols = ["utt_id", "duration", "text", "original_audio_id"]
    for index in reserved:
        path = files[index]["path"]

        def read_meta(path=path):
            pf, handle = _remote_parquet("yodas", path)
            with handle:
                sizes = [pf.metadata.row_group(g).num_rows for g in range(pf.metadata.num_row_groups)]
                return sizes, pf.read(columns=meta_cols).to_pylist()

        sizes, rows = _retry(read_meta, f"yodas metadata {path}")
        recordings.update(r["original_audio_id"] for r in rows)
        groups = sorted(int(g) for g in gen.choice(len(sizes), size=min(YODAS_DEV_ROW_GROUPS, len(sizes)),
                                                   replace=False))
        plan[path] = {"index": index, "rows": len(rows), "row_groups": len(sizes), "chosen_row_groups": groups}
        offsets = np.cumsum([0] + sizes)
        for g in groups:
            for r in rows[offsets[g]:offsets[g + 1]]:
                if MIN_SECONDS <= r["duration"] <= MAX_SECONDS and (r["text"] or "").strip():
                    candidates.append((f"yodas:{r['utt_id']}", path, g))
    candidates.sort()
    order = [candidates[i] for i in gen.permutation(len(candidates))]
    tentative = order[:YODAS_DEV_UTTERANCES + 200]  # margin for undecodable audio
    wanted: dict[tuple[str, int], set[str]] = collections.defaultdict(set)
    for uid, path, g in tentative:
        wanted[(path, g)].add(uid)
    good: dict[str, dict] = {}
    for (path, g), ids in sorted(wanted.items()):
        def read_group(path=path, g=g):
            pf, handle = _remote_parquet("yodas", path)
            with handle:
                return pf.read_row_group(g, columns=["utt_id", "audio", "text", "original_audio_id"]).to_pylist()

        out_dir = paths.AUDIO / "yodas" / "reserved" / shard_key(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        for row in _retry(read_group, f"yodas row group {path}#{g}"):
            uid = f"yodas:{row['utt_id']}"
            if uid not in ids:
                continue
            pcm, reason, _ = convert((row["audio"] or {}).get("bytes") or b"")
            if reason is None and duration_drop_reason(pcm.size / SR) is None:
                out = out_dir / (safe_name(row["utt_id"]) + ".flac")
                write_flac(out, pcm)
                good[uid] = {"id": uid, "audio_filepath": str(out), "duration": pcm.size / SR,
                             "text": row["text"].strip(), "group": row["original_audio_id"], "drop": None}
    chosen = [good[uid] for uid, _, _ in tentative if uid in good][:YODAS_DEV_UTTERANCES]
    if len(chosen) != YODAS_DEV_UTTERANCES:
        raise RuntimeError(f"yodas_dev: only {len(chosen)} usable utterances")
    rows, drops = filter_records(chosen, "yodas", duration_filter=True)
    spec = HF_SOURCES["yodas"]
    atomic_write_json(reserved_path(), {
        "repo": spec["repo"], "revision": spec["revision"], "config": "English", "split": "asr_only",
        "seed": paths.SEED, "rng_stream": STREAMS["yodas_reserved"],
        "method": f"np.random.default_rng([SEED, stream]).choice(n_shards, {YODAS_RESERVED_SHARDS}, "
                  "replace=False) over the sorted English asr_only parquet paths",
        "n_shards_total": len(files),
        "reserved": [{"path": path, **info} for path, info in plan.items()],
        "reserved_recordings": sorted(recordings),
        "note": "Never train on these shards; training also drops utterances whose "
                "original_audio_id is listed here."})
    meta = {"source": "yodas", "repo": spec["repo"], "config": spec["config"], "split": "asr_only",
            "revision": spec["revision"], "license": spec["license"], "text_field": spec["text_col"],
            "text_kind": spec["text_kind"], "kind": "dev", "dropped": drops, "filters": _filters(True),
            "reserved_shards": str(reserved_path()), "raw_files": len(plan),
            "row_groups_read": sum(len(v["chosen_row_groups"]) for v in plan.values()),
            "candidates": len(candidates), "recordings": len({r["group"] for r in chosen}),
            "selection": f"{YODAS_DEV_UTTERANCES} utterances: the first decodable ones in a seeded "
                         f"permutation (stream {STREAMS['yodas_dev']}) of the {len(candidates)} id-sorted "
                         f"1-30 s, nonempty-text rows in {YODAS_DEV_ROW_GROUPS} seeded row groups of each "
                         f"of the {len(plan)} reserved shards"}
    out = write_manifest("yodas_dev", rows, meta)
    _log(f"manifest yodas_dev: {out['utterances']} utterances, {out['hours']:.2f} h, "
         f"{meta['recordings']} recordings")
    return out


def reserved_recordings() -> set[str]:
    if not reserved_path().exists():
        raise RuntimeError("run `data.py dev` first: the YODAS reserved shards are not recorded")
    return set(json.loads(reserved_path().read_text())["reserved_recordings"])


def write_subset(name: str, n: int = DEV_SUBSET) -> dict:
    rows = load_manifest(name)
    subset = every_kth(rows, n)
    parent = load_meta(name)
    meta = {k: parent[k] for k in ("source", "repo", "config", "split", "revision", "license",
                                   "text_field", "text_kind") if k in parent}
    meta.update(kind="dev_subset", parent=name, parent_sha256=parent["sha256"],
                selection=f"every k-th row of the id-sorted parent, k = {len(rows) // n if len(rows) > n else 1}, "
                          f"first {n}")
    return write_manifest(f"{name}_{n}", subset, meta)


def cmd_dev(workers: int) -> None:
    for name, split in LS_DEV.items():
        rows, drops = filter_records(librispeech_records(split, workers), "librispeech", duration_filter=False)
        out = write_manifest(name, rows, {**ls_meta([split]), "kind": "dev", "dropped": drops,
                                          "filters": _filters(False)})
        _log(f"manifest {name}: {out['utterances']} utterances, {out['hours']:.2f} h, dropped {drops}")
    for name, source in (("ami_dev", "ami"), ("voxpopuli_dev", "voxpopuli")):
        jobs = [shard_job(source, "validation", e) for e in list_hf_files(source, "validation")]
        run_shards(jobs, workers, name)
        hf_manifest(name, source, "validation", jobs)
    build_yodas_dev()
    for name in paths.DEV_SETS:
        write_subset(name)
    atomic_write_json(paths.MANIFESTS / "dev.done.json",
                      {"finished": _now(), "manifests": {n: load_meta(n)["sha256"] for n in paths.DEV_SETS}})


def cmd_status() -> None:
    for name in [n for d in paths.DEV_SETS for n in (d, f"{d}_{DEV_SUBSET}")]:
        if not meta_path(name).exists():
            print(f"{name}: missing")
            continue
        m = load_meta(name)
        print(f"{m['name']}: {m['utterances']} utterances, {m['hours']:.2f} h, dropped {m.get('dropped', {})}")
    print(f"dev: {'done' if (paths.MANIFESTS / 'dev.done.json').exists() else 'not done'}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["dev", "status"])
    parser.add_argument("--workers", type=int, default=WORKERS, help="processes (max 8)")
    args = parser.parse_args(argv)
    paths.require_mount()
    for d in (paths.DATA, paths.AUDIO, paths.MANIFESTS):
        d.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        cmd_status()
        return
    _log(f"data.py {args.command} start (pid {os.getpid()})")
    cmd_dev(min(8, args.workers))
    _log(f"data.py {args.command} done")


if __name__ == "__main__":
    sys.exit(main())
