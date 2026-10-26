"""`med-osce`: fetched once, then segmented like Part A (labeling plan "New corpora").

Fetch: the registry's single archive is downloaded to $RELABEL_ROOT/corpora/med-osce/raw/ (resumable curl),
verified against the registry's size and MD5 (its SHA-256 is recorded), and only the audio and transcript
members named by the registry layout are extracted (flat, by recording id; no other path is written).

Transcripts have no timestamps. Each line is a speaker turn starting with a speaker label ("D:"/"P:", with
typing variants such as "D;", "P No.", "DL OK"), or a continuation of the previous turn. The reference of a
recording is its turns' text with the labels removed, in order. Recordings are mono (both speakers in one
channel), so the whole reference is segmented at once by Part A's segmenter (devsets.segment.segment) with
the permissively licensed training aligner (corpora/ctc_align.py; never Part A's non-commercial MMS model):
1-30 s segments with stable ids "<recording>-<k>", dropped segments counted per reason with their seconds.
Every record carries the aligner's identity; a record made by another aligner is never reused, and the frozen
manifest's meta and status record the single aligner that produced all of it. The device the aligner ran on
(identity "device", CPU or GPU) is not part of that comparison: the algorithm and weights are the same and
only floating-point reduction order differs, so a CPU run is resumed on the GPU. The meta and status name the
aligner without its device and count the recordings per device ("aligner_devices").

Output (frozen once written: a built corpus is never re-segmented):
    audio/<recording>/<segment id>.flac    16 kHz mono 16-bit
    records/<recording>.json               per-recording segments and drops (resume unit)
    manifest.jsonl                         audio_filepath, duration, text, id ("med-osce:<segment id>"), source,
                                           recording, start, end, untranscribed; sorted by id
    manifest.meta.json, status.json        counts, hours, drops, SHA-256s
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from pool import common

ALIAS = "med-osce"
_LABEL = re.compile(r"^[DPdp](?:\s*[:;\"]\s*|L\s+|\s+)")
_SAFE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("transcript is neither UTF-8 nor cp1252")


def reference_text(raw: str) -> tuple[str, dict]:
    """The transcript's words with speaker labels removed, turns in order; plus line counts."""
    parts, labelled, continuation = [], 0, 0
    for line in raw.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        m = _LABEL.match(line)
        if m:
            labelled += 1
            line = line[m.end():]
        else:
            continuation += 1
        if line:
            parts.append(line)
    return " ".join(parts), {"labelled_lines": labelled, "continuation_lines": continuation}


# ------------------------------------------------------------------------------------------------
# Fetch
# ------------------------------------------------------------------------------------------------

def _md5_sha256(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def download(url: str, dest: Path, attempts: int = 6) -> None:
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        if subprocess.run(["curl", "-fsSL", "--retry", "5", "-C", "-", "-o", str(part), url]).returncode == 0:
            part.replace(dest)
            return
        common.log(f"{ALIAS}: download attempt {attempt} failed")
        time.sleep(min(600, 30 * attempt))
    raise RuntimeError(f"{ALIAS}: download failed after {attempts} attempts")


def extract(archive: Path, layout: dict, out: Path) -> dict:
    """Extract the layout's audio and transcript members to out/audio/<id><ext> and out/transcripts/<id><ext>."""
    audio, text = {}, {}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            folder, _, name = info.filename.rpartition("/")
            for kind, d, ext, dest in (("audio", layout["audio_dir"], layout["audio_ext"], audio),
                                       ("transcripts", layout["transcript_dir"], layout["transcript_ext"], text)):
                if folder == d and name.endswith(ext):
                    stem = name[:-len(ext)]
                    if not _SAFE_STEM.fullmatch(stem) or stem in dest:
                        raise ValueError(f"{ALIAS}: unexpected member {info.filename!r}")
                    target = out / kind / f"{stem}{ext}"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_name(target.name + ".tmp")
                    with z.open(info) as src, open(tmp, "wb") as dst:
                        while block := src.read(1 << 20):
                            dst.write(block)
                    tmp.replace(target)
                    dest[stem] = str(target)
    return {"audio": audio, "transcripts": text}


def fetch(reg: dict, cdir: Path, downloader=download) -> dict:
    raw = cdir / "raw"
    marker = raw / "fetch.done.json"
    if marker.exists():
        return json.loads(marker.read_text())
    archive = raw / reg["file"]
    if not archive.exists():
        downloader(reg["download_url"], archive)
    size = archive.stat().st_size
    md5, sha = _md5_sha256(archive)
    if size != reg["size"] or md5 != reg["md5"]:
        archive.rename(archive.with_name(archive.name + ".bad"))
        raise RuntimeError(f"{ALIAS}: archive size {size} / md5 {md5} != registry {reg['size']} / {reg['md5']}")
    files = extract(archive, reg["layout"], raw / "extracted")
    paired = sorted(set(files["audio"]) & set(files["transcripts"]))
    summary = {"archive": str(archive), "bytes": size, "md5": md5, "sha256": sha,
               "audio_files": len(files["audio"]), "transcript_files": len(files["transcripts"]),
               "paired": len(paired), "audio_only": sorted(set(files["audio"]) - set(files["transcripts"])),
               "transcript_only": sorted(set(files["transcripts"]) - set(files["audio"])),
               "recordings": {r: {"audio": files["audio"][r], "transcript": files["transcripts"][r]} for r in paired},
               "fetched": common.now()}
    common.atomic_write_json(marker, summary)
    return summary


# ------------------------------------------------------------------------------------------------
# Segmentation
# ------------------------------------------------------------------------------------------------

def load_audio(path: Path) -> np.ndarray:
    import soundfile as sf
    from devsets import sources   # Part A: averaging to mono and resampling to 16 kHz
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return sources.to_16k_mono(audio, rate)


def segment_recording(rec: str, audio: np.ndarray, raw_transcript: str, aligner, audio_dir: Path) -> dict:
    """Segment one recording, write its segment FLACs, and return its record (rows, drops)."""
    import soundfile as sf
    import data   # finetune/parakeet-ternary: to_int16
    from devsets import segment as seg
    text, lines = reference_text(raw_transcript)
    result = seg.segment(audio, rec, text, aligner=aligner)
    rows = []
    rdir = audio_dir / seg.safe_id(rec)
    rdir.mkdir(parents=True, exist_ok=True)
    for s in result.segments:
        piece = audio[seg.samples(s.start):seg.samples(s.end)]
        out = rdir / f"{s.id}.flac"
        tmp = out.with_name(out.name + ".tmp.flac")
        sf.write(tmp, data.to_int16(piece), seg.SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        tmp.replace(out)
        rows.append({"audio_filepath": str(out), "duration": len(piece) / seg.SAMPLE_RATE, "text": s.text,
                     "id": f"{ALIAS}:{s.id}", "source": ALIAS, "recording": rec, "start": round(s.start, 4),
                     "end": round(s.end, 4), "untranscribed": bool(getattr(s, "untranscribed", False))})
    return {"recording": rec, "audio_seconds": len(audio) / seg.SAMPLE_RATE, "lines": lines, "rows": rows,
            "drops": dict(result.drops), "dropped_seconds": {k: round(v, 3) for k, v in result.dropped_seconds.items()},
            "aligner": getattr(aligner, "identity", None)}


def segment_all(cdir: Path, aligner, limit: int | None = None, batch: int | None = None) -> dict:
    """Segment every fetched recording (resumable per recording), then freeze the manifest. A corpus whose
    status is already "built" is returned unchanged. `batch` caps the recordings segmented in this call (the
    manifest is frozen only once every recording is done); aligner None only freezes existing records."""
    status_path = cdir / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("status") == "built":
        return json.loads(status_path.read_text())
    fetched = json.loads((cdir / "raw" / "fetch.done.json").read_text())
    recs = sorted(fetched["recordings"])[:limit] if limit else sorted(fetched["recordings"])
    records_dir = cdir / "records"
    ident = without_device(getattr(aligner, "identity", None))
    done_now = 0
    for n, rec in enumerate(recs, 1):
        marker = records_dir / f"{rec}.json"
        if marker.exists() and (aligner is None or without_device(json.loads(marker.read_text()).get("aligner")) == ident):
            continue
        if aligner is None:
            raise RuntimeError(f"{ALIAS}: {rec} is not segmented and no aligner was given")
        if batch is not None and done_now >= batch:
            return {"status": "partial", "recordings": len(recs),
                    "done": sum(1 for r in recs if (records_dir / f"{r}.json").exists())}
        done_now += 1
        t0 = time.time()
        paths = fetched["recordings"][rec]
        raw = decode_text(Path(paths["transcript"]).read_bytes())
        record = segment_recording(rec, load_audio(Path(paths["audio"])), raw, aligner, cdir / "audio")
        record["seconds"] = round(time.time() - t0, 1)
        common.atomic_write_json(marker, record)
        common.log(f"{ALIAS}: {n}/{len(recs)} {rec}: {len(record['rows'])} segments, drops {record['drops']}, "
                   f"{record['seconds']}s")
    if limit:
        return {"status": "partial", "recordings": len(recs)}
    return freeze(cdir, recs, fetched)


def without_device(identity: dict | None) -> dict | None:
    """An aligner identity without the device its emissions were computed on (module docstring)."""
    return None if identity is None else {k: v for k, v in identity.items() if k != "device"}


def freeze(cdir: Path, recs: list[str], fetched: dict) -> dict:
    rows, drops, dropped_seconds, lines = [], Counter(), Counter(), Counter()
    audio_seconds, aligners, devices = 0.0, set(), Counter()
    for rec in recs:
        r = json.loads((cdir / "records" / f"{rec}.json").read_text())
        aligners.add(json.dumps(without_device(r.get("aligner")), sort_keys=True))
        devices[(r.get("aligner") or {}).get("device") or "unknown"] += 1
        rows += r["rows"]
        drops.update(r["drops"])
        dropped_seconds.update(r["dropped_seconds"])
        lines.update(r["lines"])
        audio_seconds += r["audio_seconds"]
    if len(aligners) != 1:
        raise RuntimeError(f"{ALIAS}: records were made by {len(aligners)} different aligners")
    aligner = json.loads(aligners.pop())
    rows.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{ALIAS}: duplicate segment ids")
    manifest = cdir / "manifest.jsonl"
    common.atomic_write_lines(manifest, (json.dumps(r, ensure_ascii=False) for r in rows))
    meta = {"alias": ALIAS, "recordings": len(recs), "segments": len(rows),
            "hours": round(sum(r["duration"] for r in rows) / 3600, 4),
            "recording_hours": round(audio_seconds / 3600, 4), "drops": dict(sorted(drops.items())),
            "dropped_hours": {k: round(v / 3600, 4) for k, v in sorted(dropped_seconds.items())},
            "transcript_lines": dict(lines), "archive_sha256": fetched["sha256"], "aligner": aligner,
            "aligner_devices": dict(sorted(devices.items())),
            "recordings_without_pair": {"audio_only": fetched["audio_only"],
                                        "transcript_only": fetched["transcript_only"]},
            "manifest": str(manifest), "sha256": common.sha256_file(manifest), "created": common.now()}
    common.atomic_write_json(cdir / "manifest.meta.json", meta)
    status = {"status": "built", "manifest": str(manifest), "sha256": meta["sha256"], "segments": len(rows),
              "hours": meta["hours"], "aligner": aligner, "aligner_devices": meta["aligner_devices"],
              "created": common.now()}
    common.atomic_write_json(cdir / "status.json", status)
    return status
