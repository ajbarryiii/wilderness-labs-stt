"""Fetch and build the domain dev sets (labeling plan Part A). CPU and network only, no GPU.

    ./python -m devsets.build fetch [--parts PART ...]
    ./python -m devsets.build build [--parts PART ...] [--smoke N] [--threads T] [--device cpu|cuda] [--lock-wait S]

fetch: pinned, verified downloads per alias (fetch.py) -> layout.raw(alias)/fetch.json.
build: per part, its recordings (sources.py) are segmented (segment.py; the MMS aligner, loaded only if a
part needs it, runs on the CPU with T threads, or with --device cuda computes its emissions on the GPU while
the build holds the parakeet GPU lock, waiting at most S seconds for it: run that under a shared GPU lease);
each kept segment is written as 16 kHz mono 16-bit FLAC,
and the part's manifest (sorted by id) and meta are written last: <part>.jsonl, then <part>.meta.json
(the completion marker, removed when a part's build starts). Records: audio_filepath, duration (of the
written audio), text, id ("<part>:<segment id>"), source (= part) and part.

Acoustic variants (registry "timing_from": a pattern with groups base and variant over recording ids, a
template and a variant -> reference-variant map): a recording that is an acoustic variant of another
rendering (the same speech with noise, a codec or a channel applied) reuses the reference rendering's cuts
and texts, after checking that the transcript is identical and the sample count within
VARIANT_SAMPLE_TOLERANCE (otherwise it is aligned on its own). Its segments are then the reference's, cut from its own audio, with its own ids; the
reference's alignment drops apply to it too. The aligner is thus run on the clean rendering, where it is
reliable, and every variant is scored on the same segments (a noisy rendering's low alignment confidence
would otherwise drop its hardest segments). The meta counts these recordings and breaks counts down by
variant.

Resume: each recording's result is cached in work/<part>/<recording>.json with the segmentation config's
hash; a rerun reuses a cached recording whose hash matches and whose audio files all exist, and re-segments
every other one. The aligner's device is not part of that hash (same algorithm and weights; the devices
differ only by floating-point reduction order), so a CPU build is resumed on the GPU: each cache records the
device of the build that segmented it ("aligner_device"; caches without it predate the GPU option and were
built on the CPU) and the meta counts recordings per device ("aligner_devices"). --smoke N builds the first
N recordings of each part into the smoke layout (layout.Layout.smoke), never the production paths.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import soundfile as sf

import devsets
from devsets import align, fetch, layout, registry, segment, sources

SEGMENTER_VERSION = 2
# An acoustic variant may differ from its reference rendering by MP3 frame padding at the end: the dev split's
# MP3 decodes are 0, 15 or 31 samples longer than their reference, with segment start times equal (median
# 0.00 s over 74 segments aligned independently). Up to 10 ms of difference keeps the reference's cuts.
VARIANT_SAMPLE_TOLERANCE = 160


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def segmentation_config(entry: dict) -> dict:
    """Everything a segment's boundaries, text and acceptance depend on (hashed for resume)."""
    return {"version": SEGMENTER_VERSION, "layout": entry["layout"],
            "boundary_words": sorted(w.lower() for w in entry.get("boundary_words", ())),
            "min_seconds": segment.MIN_SECONDS, "max_seconds": segment.MAX_SECONDS,
            "target_seconds": segment.TARGET_SECONDS, "pad_seconds": segment.PAD_SECONDS,
            "max_word_seconds": segment.MAX_WORD_SECONDS, "min_score": align.MIN_SCORE,
            "aligner_sha256": align.CHECKPOINT_SHA256,
            "aligner": {k: getattr(align, k) for k in ("CHUNK_FRAMES", "CONTEXT_FRAMES", "BLOCK_SECONDS",
                                                        "OVERLAP_SECONDS", "WINDOW_FACTOR", "SLACK_SECONDS",
                                                        "RETRY_FACTORS", "MAX_WINDOW_SECONDS")},
            "timing_from": entry.get("timing_from"), "sample_rate": segment.SAMPLE_RATE, "flac_subtype": "PCM_16",
            **({"variant_sample_tolerance": VARIANT_SAMPLE_TOLERANCE} if entry.get("timing_from") else {}),
            **({"decoder": sources.ffmpeg_version()} if entry["layout"] == "srt-longform" else {})}


def config_hash(config: dict, fetch_sha256: str) -> str:
    blob = json.dumps({"config": config, "fetch": fetch_sha256}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _cached(path: Path, chash: str) -> dict | None:
    if not path.exists():
        return None
    try:
        cache = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if cache.get("config_hash") != chash:
        return None
    if any(not Path(s["audio_filepath"]).exists() for s in cache.get("segments", [])):
        return None
    return cache


def transcript_sha256(transcript) -> str:
    data = transcript if isinstance(transcript, str) else [[u.start, u.end, u.text] for u in transcript]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


def variant_of(recording_id: str, entry: dict) -> tuple[str, str] | None:
    """(base, variant) of a recording id under the registry's "timing_from" pattern, or None."""
    spec = entry.get("timing_from")
    match = re.fullmatch(spec["pattern"], recording_id) if spec else None
    return (match.group("base"), match.group("variant")) if match else None


def reference_of(recording_id: str, entry: dict) -> str | None:
    """The recording whose cuts this acoustic variant reuses (registry "timing_from"), or None."""
    found = variant_of(recording_id, entry)
    if found is None:
        return None
    base, variant = found
    target = entry["timing_from"]["reference"].get(variant)
    return entry["timing_from"]["template"].format(base=base, variant=target) if target else None


def build_recording(part: str, src: sources.Source, item: sources.Item, lay: layout.Layout, aligner,
                    chash: str, reference: dict | None = None) -> dict:
    """Segment one recording and write its audio. With `reference` (the cache of the rendering this one is an
    acoustic variant of), the reference's cuts and texts are reused if the transcript and the sample count
    are identical; otherwise the recording is aligned on its own."""
    rid = segment.safe_id(item.id)
    try:
        rec = src.load(item)
    except ValueError as error:
        if str(error) not in sources.RECORDING_DROPS:
            raise
        return {"config_hash": chash, "recording": item.id, "dropped": str(error), "segments": []}
    digest, samples = transcript_sha256(rec.transcript), len(rec.audio)
    timing_from = None
    if (reference is not None and not reference.get("dropped") and reference.get("transcript_sha256") == digest
            and abs(reference.get("samples", -10**9) - samples) <= VARIANT_SAMPLE_TOLERANCE):
        prefix = f"{part}:{segment.safe_id(reference['recording'])}"
        end = samples / segment.SAMPLE_RATE
        found = [segment.Segment(rid + s["id"][len(prefix):], s["start"], min(s["end"], end), s["text"], s["aligned"],
                                 s["untranscribed"]) for s in reference["segments"]]
        drops, dropped_seconds = reference["drops"], reference["dropped_seconds"]
        timing_from = reference["recording"]
    else:
        result = segment.segment(rec.audio, rec.id, rec.transcript, aligner=aligner,
                                 boundary_words=src.boundary_words)
        found, drops, dropped_seconds = result.segments, dict(result.drops), dict(result.dropped_seconds)
    out_dir = lay.audio(part) / rid
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    segs = []
    for s in found:
        clip = rec.audio[segment.samples(s.start):segment.samples(s.end)]
        path = out_dir / f"{s.id}.flac"
        sf.write(path, clip, segment.SAMPLE_RATE, subtype="PCM_16", format="FLAC")
        segs.append({"id": f"{part}:{s.id}", "audio_filepath": str(path), "duration": len(clip) / segment.SAMPLE_RATE,
                     "text": s.text, "start": s.start, "end": s.end, "aligned": s.aligned,
                     "untranscribed": s.untranscribed})
    return {"config_hash": chash, "recording": item.id, "audio_seconds": samples / segment.SAMPLE_RATE,
            "samples": samples, "transcript_sha256": digest, "timing_from": timing_from,
            "segments": segs, "drops": drops, "dropped_seconds": dropped_seconds}


def build_part(part: str, entry: dict, lay: layout.Layout, aligner, smoke: int | None) -> dict:
    fetch_path = lay.fetch_record(part)
    if not fetch_path.exists():
        raise SystemExit(f"{part}: not fetched ({fetch_path}); run `fetch` first")
    fetch_record = json.loads(fetch_path.read_text())
    if fetch_record.get("revision") != entry["revision"] or fetch_record.get("repo") != entry["repo"]:
        raise SystemExit(f"{part}: {fetch_path} is not the registry's pinned revision; fetch again")
    fetch_sha = hashlib.sha256(fetch_path.read_bytes()).hexdigest()
    lay.meta(part).unlink(missing_ok=True)  # the part is incomplete until its meta is written again
    config = segmentation_config(entry)
    chash = config_hash(config, fetch_sha)
    src = sources.Source(part, entry, [Path(f["local"]) for f in fetch_record["files"]])
    items, recording_drops = src.items()
    listed = len(items) + sum(recording_drops.values())
    by_id = {item.id: item for item in items}
    if smoke:
        items = items[:smoke]
    start = time.time()

    def ensure(item: sources.Item, n: int, allow_reference: bool = True) -> dict:
        cache_path = lay.work(part) / f"{segment.safe_id(item.id)}.json"
        cache = _cached(cache_path, chash)
        if cache is not None:
            return cache
        ref_id = reference_of(item.id, entry) if allow_reference else None
        reference = (ensure(by_id[ref_id], n, allow_reference=False)
                     if ref_id and ref_id != item.id and ref_id in by_id else None)
        cache = build_recording(part, src, item, lay, aligner, chash, reference)
        cache["aligner_device"] = getattr(aligner, "device", "cpu")
        write_json(cache_path, cache)
        print(f"[{part} {n}/{len(items)} {time.time() - start:.0f}s] {item.id}: {len(cache['segments'])} segments "
              f"{cache.get('dropped') or dict(cache.get('drops', {}))}"
              + (f" (cuts of {cache['timing_from']})" if cache.get("timing_from") else ""), flush=True)
        return cache

    records, drops, dropped_seconds, devices = [], Counter(), Counter(), Counter()
    untranscribed = aligned = audio_seconds = used = from_reference = 0
    variants: dict[str, dict] = {}
    for n, item in enumerate(items, 1):
        cache = ensure(item, n)
        devices[cache.get("aligner_device", "cpu")] += 1
        if cache.get("dropped"):
            recording_drops[cache["dropped"]] += 1
            continue
        used += 1
        from_reference += bool(cache.get("timing_from"))
        audio_seconds += cache["audio_seconds"]
        drops.update(cache["drops"])
        dropped_seconds.update(cache["dropped_seconds"])
        found = variant_of(item.id, entry)
        if found is not None:
            v = variants.setdefault(found[1], {"recordings": 0, "utterances": 0, "hours": 0.0, "segments_dropped": Counter()})
            v["recordings"] += 1
            v["utterances"] += len(cache["segments"])
            v["hours"] += sum(s["duration"] for s in cache["segments"]) / 3600
            v["segments_dropped"].update(cache["drops"])
        for s in cache["segments"]:
            records.append({"audio_filepath": s["audio_filepath"], "duration": s["duration"], "text": s["text"],
                            "id": s["id"], "source": part, "part": part})
            untranscribed += s["untranscribed"]
            aligned += s["aligned"]
    records.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{part}: duplicate segment ids")
    manifest = lay.manifest(part)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest.with_name(manifest.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as handle:
        for r in records:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, manifest)
    hours = sum(r["duration"] for r in records) / 3600
    meta = {"part": part, "set": devsets.set_of(part), "complete": True, "smoke": smoke,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": {"alias": part, "registry_entry": entry, "revision": entry["revision"],
                       "fetch_record": str(fetch_path), "fetch_record_sha256": fetch_sha},
            "segmentation": config, "config_hash": chash,
            "aligner": {k: v for k, v in (getattr(getattr(aligner, "_aligner", None), "identity", None) or
                        {"aligner": "torchaudio.pipelines.MMS_FA (with_star)", "sha256": align.CHECKPOINT_SHA256}).items() if k != "device"},
            "aligner_devices": dict(sorted(devices.items())),
            "recordings": {"listed": listed, "selected": len(items), "used": used,
                           "dropped": dict(recording_drops), "audio_hours": audio_seconds / 3600},
            "utterances": len(records), "hours": hours,
            "segments_dropped": {r: {"count": drops[r], "hours": dropped_seconds[r] / 3600}
                                 for r in segment.DROP_REASONS},
            "aligned_segments": aligned, "source_timestamp_segments": len(records) - aligned,
            "untranscribed_marker_segments": untranscribed, "recordings_with_reference_cuts": from_reference,
            "by_variant": {k: {**v, "segments_dropped": dict(v["segments_dropped"])} for k, v in sorted(variants.items())},
            "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    write_json(lay.meta(part), meta)
    print(f"{part}: {len(records)} segments, {hours:.2f} h; segment drops {dict(drops)}; "
          f"recording drops {dict(recording_drops)}", flush=True)
    return meta


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "build"))
    parser.add_argument("--parts", nargs="+", default=list(devsets.PARTS), choices=list(devsets.PARTS))
    parser.add_argument("--smoke", type=int, help="build only the first N recordings per part, into the smoke layout")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for the aligner and resampling")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="where the aligner's emissions are computed (cuda: holds the parakeet GPU lock)")
    parser.add_argument("--lock-wait", type=float, default=1800.0, help="seconds to wait for the GPU lock")
    args = parser.parse_args(argv)
    if args.smoke is not None and args.command != "build":
        parser.error("--smoke applies to build only")
    if args.smoke is not None and args.smoke <= 0:
        parser.error("--smoke must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    layout.require_mount()
    entries = registry.load()
    if args.command == "fetch":
        lay = layout.Layout.production()
        for part in args.parts:
            record = fetch.fetch(part, registry.source(part, entries), lay.raw(part))
            print(f"{part}: {len(record['files'])} files, "
                  f"{sum(f['bytes'] for f in record['files']) / 2**30:.2f} GiB", flush=True)
        return 0
    import torch

    torch.set_num_threads(args.threads)
    lay = layout.Layout.smoke(args.smoke) if args.smoke else layout.Layout.production()
    aligner = align.LazyAligner(args.threads, device=args.device)
    lock = contextlib.nullcontext()
    if args.device != "cpu":
        import score_teacher
        lock = score_teacher.gpu_lock(f"relabel devsets build {' '.join(args.parts)}", args.lock_wait)
    with lock:
        for part in args.parts:
            build_part(part, registry.source(part, entries), lay, aligner, args.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
