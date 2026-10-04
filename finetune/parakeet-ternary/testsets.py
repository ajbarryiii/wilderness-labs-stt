"""Extract the Open ASR Leaderboard test sets to 16 kHz mono FLAC plus NeMo-style manifests. See DESIGN.md "Evaluation".

Input: the parquet files of hf-audio/esb-datasets-test-only-sorted at paths.ESB_REVISION,
downloaded to paths.DATA/"esb" (one directory per config; columns audio{bytes,path},
dataset, text, id, audio_length_s; the text column is "text" in every config).

Audio is decoded the way the leaderboard's `datasets.Audio(sampling_rate=16000)` cast did
when NVIDIA scored this model (datasets 3.x): soundfile decode to float64, channel mean
(librosa.to_mono), soxr "HQ" resampling (librosa's default soxr_hq) when the source rate
is not 16 kHz; then written as 16-bit FLAC, which matches the leaderboard's
soundfile.write(..., "x.wav") default PCM_16 cache files. Earnings-22 has 22.05/24/44.1 kHz
and stereo files; Common Voice is 48 kHz MP3; the rest are 16 kHz mono (WAV float, WAV
PCM_16 or FLAC).

Every utterance is kept (no duration filter on test sets); utterances over 40 s are listed
in the .meta.json. Common Voice repeats some ids with different audio; repeated ids get a
"#<k>" suffix (k = occurrence index in parquet order, the first occurrence keeps the bare
id). The leaderboard's NeMo script caches audio by id, so for such repeats it scores the
first clip's audio again; we score each clip's own audio (Common Voice is reported
separately and is not in the DESIGN.md mean).

Output per set: paths.AUDIO/"test"/<set>/<file>.flac, paths.MANIFESTS/test_<set>.jsonl
({"audio_filepath","duration","text","id","source"}, sorted by id, source = set name) and
test_<set>.meta.json. Resumable: finished FLAC files are reused; each file and manifest is
written to a temporary name and renamed.

Usage: python testsets.py [--sets NAME ...] [--workers 4]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

import paths

ESB_DIR = paths.DATA / "esb"
DOWNLOAD_LOG = paths.DATA / "esb-download.log"
LONG_S = 40.0
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def manifest_path(name: str) -> Path:
    return paths.MANIFESTS / f"test_{name}.jsonl"


def read_manifest(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def wait_for_download(poll_s: float = 30.0) -> None:
    """Block until the ESB download log ends with exit=0; raise on a nonzero exit."""
    while True:
        lines = DOWNLOAD_LOG.read_text(errors="replace").strip().splitlines() if DOWNLOAD_LOG.exists() else []
        last = lines[-1].strip() if lines else ""
        if last == "exit=0":
            return
        if last.startswith("exit="):
            raise RuntimeError(f"ESB download failed: {last} (see {DOWNLOAD_LOG})")
        print(f"waiting for the ESB download ({DOWNLOAD_LOG})", flush=True)
        time.sleep(poll_s)


def parquet_files(name: str) -> list[Path]:
    config, split = paths.TEST_SETS[name]
    files = sorted((ESB_DIR / config).glob(f"{split}-*-of-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files for {config}/{split} under {ESB_DIR}")
    total = int(files[0].name.rsplit("-of-", 1)[1].split(".")[0])
    if len(files) != total:
        raise RuntimeError(f"{config}/{split}: found {len(files)} of {total} parquet files")
    return files


def decode_audio(data: bytes) -> tuple[np.ndarray, int, int]:
    """16 kHz mono float32 samples, source rate, source channels (see the module docstring)."""
    array, rate = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    channels = array.shape[1]
    array = array.mean(axis=1) if channels > 1 else array[:, 0]
    if rate != paths.SAMPLE_RATE:
        import soxr
        array = soxr.resample(array, rate, paths.SAMPLE_RATE, quality="HQ")
    return np.float32(array), rate, channels


def _extract(job: tuple[str, str, bytes | None]) -> dict:
    """Write one FLAC (unless already complete) and return its duration and source format."""
    uid, path, data = job
    target = Path(path)
    if data is None:  # already extracted in an earlier run
        info = sf.info(target)
        return {"id": uid, "frames": info.frames, "source_rate": None, "source_channels": None}
    array, rate, channels = decode_audio(data)
    tmp = target.with_name(target.name + f".tmp{os.getpid()}.flac")
    sf.write(tmp, array, paths.SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    os.replace(tmp, target)
    return {"id": uid, "frames": len(array), "source_rate": rate, "source_channels": channels}


def _rows(files: list[Path]):
    """(id, text, audio bytes) in parquet file and row order."""
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches(batch_size=32, columns=["id", "text", "audio"]):
            for uid, text, audio in zip(batch.column("id").to_pylist(), batch.column("text").to_pylist(),
                                        batch.column("audio").to_pylist()):
                yield uid, text, audio["bytes"]


def build(name: str, workers: int) -> dict:
    """Extract one test set and write its manifest and meta; return the meta."""
    paths.require_mount()
    files = parquet_files(name)
    audio_dir = paths.AUDIO / "test" / name
    audio_dir.mkdir(parents=True, exist_ok=True)
    paths.MANIFESTS.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    used_files: set[str] = set()
    records: dict[str, dict] = {}
    repeated: list[str] = []
    start = time.perf_counter()

    def jobs():
        for raw_id, text, data in _rows(files):
            k = seen.get(raw_id, 0)
            seen[raw_id] = k + 1
            uid = raw_id if k == 0 else f"{raw_id}#{k}"
            if k:
                repeated.append(uid)
            if uid in records:
                raise RuntimeError(f"{name}: id {uid!r} still not unique")
            stem = _UNSAFE.sub("_", uid.removesuffix(".wav"))[:200]
            while stem in used_files:
                stem += "_"
            used_files.add(stem)
            target = audio_dir / f"{stem}.flac"
            records[uid] = {"audio_filepath": str(target), "text": text, "id": uid, "source": name}
            yield uid, str(target), None if target.exists() else data

    source_rates: dict[str, int] = {}
    done = 0

    def collect(future) -> None:
        nonlocal done
        result = future.result()
        records[result["id"]]["duration"] = result["frames"] / paths.SAMPLE_RATE
        if result["source_rate"] is not None:
            key = f"{result['source_rate']}Hz/{result['source_channels']}ch"
            source_rates[key] = source_rates.get(key, 0) + 1
        done += 1
        if done % 5000 == 0:
            print(f"{name}: {done} utterances", flush=True)

    # Bounded submission: Executor.map would read every audio blob into memory up front.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = []
        for job in jobs():
            pending.append(pool.submit(_extract, job))
            if len(pending) >= 8 * workers:
                collect(pending.pop(0))
        for future in pending:
            collect(future)
    ordered = [{key: records[uid][key] for key in ("audio_filepath", "duration", "text", "id", "source")}
               for uid in sorted(records)]
    manifest = manifest_path(name)
    write_atomic(manifest, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
    durations = [r["duration"] for r in ordered]
    long_utts = sorted(((r["id"], round(r["duration"], 3)) for r in ordered if r["duration"] > LONG_S),
                       key=lambda x: -x[1])
    config, split = paths.TEST_SETS[name]
    meta = {
        "set": name, "repo": paths.ESB_REPO, "revision": paths.ESB_REVISION, "config": config, "split": split,
        "parquet_files": [{"name": f"{config}/{f.name}", "bytes": f.stat().st_size} for f in files],
        "utterances": len(ordered), "hours": sum(durations) / 3600,
        "min_duration_s": min(durations), "max_duration_s": max(durations),
        "over_40s": len(long_utts), "over_40s_ids": long_utts,
        "empty_text": sum(not (r["text"] or "").strip() for r in ordered),
        "repeated_source_ids": len(repeated), "repeated_ids_renamed": repeated[:50],
        "source_formats_extracted_this_run": source_rates,
        "audio": "16 kHz mono FLAC PCM_16; float64 decode, channel mean, soxr HQ resample",
        "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
        "wall_s": time.perf_counter() - start,
    }
    write_atomic(manifest.with_suffix(".meta.json"), json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", choices=sorted(paths.TEST_SETS), default=list(paths.TEST_SETS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    paths.require_mount()
    wait_for_download()
    for name in args.sets:
        meta = build(name, args.workers)
        print(f"{name}: {meta['utterances']} utterances, {meta['hours']:.2f} h, over 40 s {meta['over_40s']}, "
              f"repeated ids {meta['repeated_source_ids']}, {meta['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
