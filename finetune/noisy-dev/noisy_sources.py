"""Evaluation-pool sources: registry, download verification, extraction, indexing and the pool format.
See plans/build.md "Evaluation-pool sources" and "Determinism and the build lock".

Sources (SOURCES) carry their publisher checksums, pinned here from the primary source on
2026-10-16: Zenodo's per-file MD5 from https://zenodo.org/api/records/<id> (files[].checksum),
OpenSLR's https://www.openslr.org/resources/17/checksum.md5 for MUSAN. BUT ReverbDB publishes no
checksum: it is accepted only over HTTPS, with the exact size the server states (pinned), and its
SHA-256 is trusted on first use and enforced from then on (review round 2, finding 4).
EXCLUDED_SOURCES can never enter a pool. Downloads (raw/<source>/, written only by download.sh)
are untrusted:

    verify   raw files against the publisher checksums -> verify/<source>.json (ok records only)
    extract  archives of verified sources -> extracted/<source>/<archive>-<key>/ (noisy_extract
             checks and EXTRACTION_BUDGETS); <key> binds the directory to the verified archive
             SHA-256, the member selection, the budget and the extractor's code
    index    verified, extracted sources -> pool/pool.json, prepared 16 kHz clips in pool/audio/
             (prepare: named by the original's SHA-256 and the preparation parameters, with a
             sidecar; a stale or tampered copy is redone)

check_pool_inputs (run by the build before every build and resume) re-checks each pooled source's
current verification record against the one the pool was built from and every pool file's
decoded-PCM SHA-256 (InputChanged otherwise). Only registered sources (SOURCES) and registered
plug-ins (PLUGIN_SOURCES: the rotor plug-in, verified by its sounds.json and download hashes)
are admitted or accepted; a self-attested verification of any other source is refused (review
round 4, finding 3).

The pool (make_pool / write_pool / load_pool) is the build's only view of the sources. It holds,
sorted by key, the continuous-noise clips per category (engine, wind, crowd, demand, babble and,
from a rotor plug-in's separate rotor.json, rotor), the impulsive clips and their frozen event
list, the RIRs, the source verification records and the reserved units (for FSD50K: every
eval-split sound ID, uploader and decoded-PCM hash). make_pool refuses a clip whose source is
excluded or unverified, or whose licence is not CC0, CC BY, CC BY-SA 3.0 (DEMAND) or public domain.

FSD50K (index_fsd50k): only clips listed in the eval split's ground truth, with a per-clip licence
of CC0 or CC BY (any version), enter a category; a clip whose labels fall in two categories, or a
gun/engine/wind clip that also carries a voice label, is excluded (plans/build.md "Implementation
notes"). Every eval-split clip is reserved, used or not: sound IDs and uploaders come from the
complete metadata (eval.csv and the clip info), and a missing expected audio file or clip-info
entry fails the index. Each clip records its noise-source group (the most specific matching
FSD50K class), as do DEMAND (environment), Zenodo (microphone), MUSAN and BUT (room).

Rotor (ROTOR_PLUGINS): absent by default. The "freesound" plug-in indexes CC0 Freesound clips of
the eval uploader split from raw/freesound-rotor/sounds.json into pool/rotor.json; the build adds
the rotor10/rotor0 conditions only when given that file, and no other condition depends on it.

CLI (inside ../parakeet-ternary/heavy for real data):

    ./python noisy_sources.py verify [SOURCE ...]
    ./python noisy_sources.py extract [SOURCE ...]
    ./python noisy_sources.py index [--allow-missing]
    ./python noisy_sources.py rotor --plugin freesound
    ./python noisy_sources.py status
"""
from __future__ import annotations

import argparse
import csv
import concurrent.futures as cf
import dataclasses
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

import noisy_dsp
import noisy_extract
import noisy_paths

SR = noisy_paths.SAMPLE_RATE
POOL_FORMAT = "noisy-dev-pool/1"
ROTOR_FORMAT = "noisy-dev-rotor/1"
_ZENODO = "https://zenodo.org/api/records/{} (files[].checksum), fetched 2026-10-16"
DEMAND_ENVS = ("DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK", "NRIVER", "OHALLWAY", "OMEETING", "OOFFICE",
               "PCAFETER", "PRESTO", "PSTATION", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO")
_DEMAND_MD5 = {"DKITCHEN": "7ffbf52d7f4699f96927846103dc8788", "DLIVING": "46741384d9e434a0bd8b3ec1830b6052",
               "DWASHING": "7e5ee9437ce9409c5f9a779b6212a240", "NFIELD": "a740046c6f4e174e16f5d568aaec5024",
               "NPARK": "80f1385a34d7f1705758926b57f138ce", "NRIVER": "54264db61d3fe073fb81f2e40e0d19b5",
               "OHALLWAY": "fe918bbb0e63e73d09ba7f4843ef33f1", "OMEETING": "62f7cfe7fe6d30b7d8a215fe37c2dfd2",
               "OOFFICE": "7b61cc2d182d5a654cb9c3101ddd4041", "PCAFETER": "99927d148128254141a9417d051510bb",
               "PRESTO": "b98d2e6854eeebb397f29a8ad7457092", "PSTATION": "d7448009f6c2aeb6ba570375df1750a3",
               "SPSQUARE": "205d0e7b8fe74504a2f8d252fc414b9e", "STRAFFIC": "2efa87262f272bbf9ba578088e81939c",
               "TBUS": "706b11b0d8504f9f3b3f3211e91b3863", "TCAR": "4d930012796bd298932245a26189f973",
               "TMETRO": "95daf4df678e13b120e14211e6d89571"}

SOURCES = {
    "fsd50k": {
        "dataset": "FSD50K", "url": "https://zenodo.org/records/4060432",
        "licence": "per clip; only CC0-1.0 and CC BY clips of the eval split are admitted",
        "checksum": "md5", "checksum_source": _ZENODO.format(4060432),
        "files": {"FSD50K.eval_audio.z01": "3090670eaeecc013ca1ff84fe4442aeb",
                  "FSD50K.eval_audio.zip": "6fa47636c3a3ad5c7dfeba99f2637982",
                  "FSD50K.metadata.zip": "b9ea0c829a411c1d42adb9da539ed237",
                  "FSD50K.ground_truth.zip": "ca27382c195e37d2269c4c866dd73485"}},
    "gunshots-ipod": {
        "dataset": "zenodo-3997406", "url": "https://zenodo.org/records/3997406",
        "licence": "CC0-1.0 (Zenodo record and readme.txt)",
        "checksum": "md5", "checksum_source": _ZENODO.format(3997406),
        "files": {"readme.txt": "0c061fd38e0198cfc7f8527a52b257f7", "mic1raw.wav": "e2f4acad742278c973e9c190fee03b5f",
                  "mic2raw16b.wav": "184d888f457f331f592eeb4ee5fa12e2",
                  "mic3raw.wav": "db2ffa4fdf8b42b98c208670f4578b4b",
                  "mic4raw.wav": "571da84eecf397bf0356a24400769ced"}},
    "demand": {
        "dataset": "DEMAND", "url": "https://zenodo.org/records/1227121",
        "licence": "CC-BY-SA-3.0 (treated as; the Zenodo rights field says CC BY 4.0, the description BY-SA 3.0)",
        "checksum": "md5", "checksum_source": _ZENODO.format(1227121),
        "files": {f"{env}_16k.zip": md5 for env, md5 in _DEMAND_MD5.items()}},
    "musan": {
        "dataset": "MUSAN", "url": "https://www.openslr.org/17/",
        "licence": "public domain (MUSAN us-gov speech, MUSAN paper); MUSAN corpus CC BY 4.0",
        "checksum": "md5", "checksum_source": "https://www.openslr.org/resources/17/checksum.md5, fetched 2026-10-16",
        "files": {"musan.tar.gz": "0c472d4fc0c5141eca47ad1ffeb2a7df"}},
    "but-reverbdb": {
        "dataset": "BUT ReverbDB", "url": "https://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz",
        "page": "https://speech.fit.vut.cz/software/but-speech-fit-reverb-database",
        "licence": "CC-BY-4.0 (BUT page)", "checksum": "tofu-sha256",
        "checksum_source": "none published: SHA-256 trusted on first use over HTTPS; exact size and Last-Modified "
                           "from the server (HEAD over HTTPS 2026-10-16, ETag 22ad5de1d-58a92c6c861a7: "
                           "0x22ad5de1d = 9308593693 bytes)",
        "sizes": {"BUT_ReverbDB_rel_19_06_RIR-Only.tgz": 9308593693},
        "last_modified": "Wed, 05 Jun 2019 12:31:28 GMT",
        "sha256": {"BUT_ReverbDB_rel_19_06_RIR-Only.tgz": None},  # pin here once the first download is recorded
        "files": {"BUT_ReverbDB_rel_19_06_RIR-Only.tgz": None}},
}
GiB, MiB = 1 << 30, 1 << 20
# Extraction budgets per source and archive (review round 2, finding 5): generous but finite, from
# the archives' known contents (FSD50K eval: 10,231 clips of at most 30 s at 44.1 kHz, about 6.3 GB;
# DEMAND: 16 five-minute 16 kHz channels per environment; MUSAN: about 2,000 files, 11 GB packed;
# BUT RIR-only: 9.3 GB packed).
EXTRACTION_BUDGETS = {
    "fsd50k": {"ground_truth": noisy_extract.Budget(100, 64 * MiB, 256 * MiB),
               "metadata": noisy_extract.Budget(100, 256 * MiB, 512 * MiB),
               "eval_audio": noisy_extract.Budget(12_000, 64 * MiB, 12 * GiB, 16 * MiB)},  # CD: ~10k entries
    "demand": {f"{env}_16k": noisy_extract.Budget(40, 32 * MiB, 400 * MiB) for env in DEMAND_ENVS},
    "musan": {"musan": noisy_extract.Budget(5_000, 512 * MiB, 20 * GiB)},
    "but-reverbdb": {"rir-only": noisy_extract.Budget(200_000, 256 * MiB, 40 * GiB)},
}
# plans/build.md "Excluded": never in any pool, and listed as excluded families in the blocklist.
EXCLUDED_SOURCES = {
    "esc50": "CC BY-NC 3.0", "urbansound8k": "CC BY-NC 3.0", "wham": "CC BY-NC 4.0", "whamr": "CC BY-NC 4.0",
    "noisex-92": "no clear licence", "mit-ir-survey": "licence stated only on a mirror",
    "ir64-car": "external loudspeaker through a closed car, not an in-cabin talker",
    "audioset": "YouTube-sourced; no rights in the audio, overlap risk with YODAS",
    "dns-audioset": "YouTube-sourced (AudioSet part of the DNS Challenge)", "mad": "YouTube-sourced",
    "sonniss": "bans AI use", "sesa": "unverifiable per-clip licences",
    "musan-librivox": "LibriVox speech (LibriSpeech speakers and texts)", "rwcp": "R&D use only",
    "but-reverbdb-retransmitted": "BUT LibriSpeech-retransmission set",
}

FSD_CATEGORIES = {"gun": ("Gunshot_and_gunfire", "Explosion"),
                  "engine": ("Engine", "Vehicle", "Car", "Truck", "Aircraft", "Fixed-wing_aircraft"),
                  "wind": ("Wind",), "crowd": ("Crowd", "Chatter", "Shout", "Yell")}
VOICE_LABELS = frozenset({"Human_voice", "Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
                          "Child_speech_and_kid_speaking", "Conversation", "Speech_synthesizer", "Singing",
                          "Shout", "Yell", "Screaming", "Whispering", "Chatter", "Crowd", "Laughter",
                          "Human_group_actions", "Cheering", "Applause"})
NO_VOICE_CATEGORIES = ("gun", "engine", "wind")
NOISE_CATEGORIES = ("engine", "wind", "crowd", "demand", "babble", "rotor")
MIN_CONTINUOUS_S = 0.5    # shorter clips cannot be looped meaningfully with a 50 ms crossfade
MIN_CLIP_DBFS = -70.0     # quieter clips would be amplified noise floor
BUT_ROOMS = ("Hotel_SkalskyDvur_ConferenceRoom2", "Hotel_SkalskyDvur_Room112", "VUT_FIT_C236", "VUT_FIT_D105",
             "VUT_FIT_E112", "VUT_FIT_L207", "VUT_FIT_L212", "VUT_FIT_L227", "VUT_FIT_Q301")
_POOL_LICENCE = re.compile(r"^(CC0-1\.0|CC-BY-\d\.\d|CC-BY-SA-3\.0|public domain)$", re.IGNORECASE)


class VerificationError(RuntimeError):
    """A download is missing, does not match its publisher checksum, or changed after verification."""


class PoolError(ValueError):
    """A clip or source may not enter the evaluation pool."""


class InputChanged(RuntimeError):
    """A pool input (verified download, extracted member, prepared or pool audio file) is not the
    one the pool, lock or cache was made from."""


def utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[noisy_sources {dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------- licences

def licence_allowed(url: str | None) -> bool:
    """FSD50K per-clip licence URL: CC0 1.0 or CC BY (any version), nothing else."""
    if not url:
        return False
    u = re.sub(r"^https?://(www\.)?", "", url.strip().lower().rstrip("/"))
    return (u == "creativecommons.org/publicdomain/zero/1.0"
            or re.fullmatch(r"creativecommons\.org/licenses/by/\d\.\d", u) is not None)


def licence_spdx(url: str) -> str:
    u = re.sub(r"^https?://(www\.)?", "", url.strip().lower().rstrip("/"))
    if u == "creativecommons.org/publicdomain/zero/1.0":
        return "CC0-1.0"
    m = re.fullmatch(r"creativecommons\.org/licenses/by/(\d\.\d)", u)
    if not m:
        raise PoolError(f"not an admitted licence: {url!r}")
    return f"CC-BY-{m.group(1)}"


def licence_spdx_or_none(url: str | None) -> str | None:
    try:
        return licence_spdx(url or "")
    except PoolError:
        return None


def pool_licence_allowed(licence: str | None) -> bool:
    return bool(licence) and _POOL_LICENCE.fullmatch(licence.strip()) is not None


# --------------------------------------------------------------------------------------------- verification

def _hashes(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            md5.update(chunk)
            sha.update(chunk)
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)  # multi-GB reads
        except OSError:
            pass
    return md5.hexdigest(), sha.hexdigest()


def verify_source(name: str, raw: Path = noisy_paths.RAW, verify_dir: Path = noisy_paths.VERIFY,
                  registry: dict | None = None) -> dict:
    """Check every required file of a source against its publisher checksum; write and return the
    verification record. VerificationError on a missing file, a missing or mismatching checksum,
    or (trust-on-first-use sources) a non-https source, a size other than the pinned one or a file
    that differs from the SHA-256 of its first use. A file whose
    download.sh `<file>.done` marker is missing is still being written and is not read at all
    (the marker is a precondition only, never evidence of integrity)."""
    registry = SOURCES if registry is None else registry
    if name in EXCLUDED_SOURCES or name not in registry:
        raise VerificationError(f"{name!r} is not an evaluation-pool source")
    spec = registry[name]
    unfinished = [f for f in sorted(spec["files"]) if not (Path(raw) / name / f"{f}.done").exists()]
    if unfinished:
        raise VerificationError(f"{name}: download not finished (no .done marker) for {unfinished}")
    tofu = spec["checksum"] == "tofu-sha256"
    if tofu and not str(spec.get("url", "")).startswith("https://"):
        raise VerificationError(f"{name}: trust on first use requires an https source, not {spec.get('url')!r}")
    record_path = Path(verify_dir) / f"{name}.json"
    try:
        previous = json.loads(record_path.read_text()) if record_path.exists() else None
    except ValueError:
        previous = {"files": "unreadable"}
    prev_files = (previous or {}).get("files", [])
    well_formed = isinstance(prev_files, list) and all(
        isinstance(f, dict) and isinstance(f.get("name"), str) and re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256")))
        for f in prev_files)
    if tofu and previous is not None and not well_formed:
        raise VerificationError(f"{name}: the recorded first use is malformed; refusing to trust a new download")
    first_seen = {f["name"]: f["sha256"] for f in prev_files} if well_formed else {}
    if tofu and previous and any(f.get("transport") != "https" for f in prev_files):
        raise VerificationError(f"{name}: the recorded first use was not over https; delete it and re-download")
    files = []
    for fname, expected in sorted(spec["files"].items()):
        path = Path(raw) / name / fname
        if not path.is_file():
            raise VerificationError(f"{name}: {path} is missing")
        st = path.stat()
        md5, sha = _hashes(path)
        entry = {"name": fname, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "md5": md5, "sha256": sha}
        if spec["checksum"] == "md5":
            if not (isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{32}", expected)):
                raise VerificationError(f"{name}/{fname}: no publisher checksum")
            if md5 != expected:
                raise VerificationError(f"{name}/{fname}: MD5 {md5} != publisher {expected}")
            entry.update(expected_md5=expected, checksum_kind="publisher md5")
        elif tofu:
            size = (spec.get("sizes") or {}).get(fname)
            if size is None or st.st_size != size:
                raise VerificationError(f"{name}/{fname}: size {st.st_size} != pinned size {size}")
            pinned = (spec.get("sha256") or {}).get(fname)
            if pinned is not None and sha != pinned:
                raise VerificationError(f"{name}/{fname}: SHA-256 {sha} != pinned {pinned}")
            if fname in first_seen and first_seen[fname] != sha:
                raise VerificationError(f"{name}/{fname}: SHA-256 {sha} != first use {first_seen[fname]}")
            entry.update(checksum_kind="trust on first use: sha256 over https, pinned size", transport="https",
                         url=spec["url"], last_modified=spec.get("last_modified"), expected_size=size,
                         first_use_utc=(previous or {}).get("first_use_utc") or utc())
        else:
            raise VerificationError(f"{name}: unknown checksum kind {spec['checksum']!r}")
        files.append(entry)
    record = {"source": name, "dataset": spec["dataset"], "url": spec.get("url"), "licence": spec["licence"],
              "checksum_source": spec.get("checksum_source"), "ok": True, "verified_utc": utc(), "files": files}
    if tofu:
        record["first_use_utc"] = (previous or {}).get("first_use_utc") or record["verified_utc"]
    noisy_paths.write_json(record_path, record)
    return record


def validate_record(name: str, record, spec: dict) -> dict:
    """The record if it is a complete verification of the registered source (review round 5): ok,
    for this source, exactly the registry's files, each with size, mtime, MD5 and SHA-256, the
    registry's MD5 (publisher sources) or https transport, the pinned size and any pinned SHA-256
    (trust on first use). VerificationError otherwise."""
    if not isinstance(record, dict) or record.get("ok") is not True:
        raise VerificationError(f"{name}: verification record is not ok")
    if record.get("source") != name:
        raise VerificationError(f"{name}: the verification record is for {record.get('source')!r}")
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise VerificationError(f"{name}: verification record lists no files (incomplete)")
    names = [f.get("name") if isinstance(f, dict) else None for f in files]
    if sorted(map(str, names)) != sorted(spec["files"]) or len(set(names)) != len(names):
        raise VerificationError(f"{name}: verified files {names} do not match the registry's {sorted(spec['files'])}")
    for f in files:
        if not (isinstance(f.get("size"), int) and isinstance(f.get("mtime_ns"), int)
                and re.fullmatch(r"[0-9a-f]{32}", str(f.get("md5"))) and re.fullmatch(r"[0-9a-f]{64}", str(f.get("sha256")))):
            raise VerificationError(f"{name}/{f['name']}: incomplete file entry in the verification record")
        expected = spec["files"][f["name"]]
        if spec["checksum"] == "md5":
            if f["md5"] != expected or f.get("expected_md5") != expected:
                raise VerificationError(f"{name}/{f['name']}: recorded MD5 is not the registry's publisher checksum")
        elif spec["checksum"] == "tofu-sha256":
            if f.get("transport") != "https" or not str(f.get("checksum_kind", "")).startswith("trust on first use"):
                raise VerificationError(f"{name}/{f['name']}: trusted on first use without https transport")
            if f["size"] != (spec.get("sizes") or {}).get(f["name"]):
                raise VerificationError(f"{name}/{f['name']}: size differs from the registry's pinned size")
            pinned = (spec.get("sha256") or {}).get(f["name"])
            if pinned is not None and f["sha256"] != pinned:
                raise VerificationError(f"{name}/{f['name']}: SHA-256 differs from the registry's pinned digest")
        else:
            raise VerificationError(f"{name}: unknown checksum kind {spec['checksum']!r} in the registry")
    return record


def require_verified(name: str, verify_dir: Path = noisy_paths.VERIFY, raw: Path = noisy_paths.RAW,
                     registry: dict | None = None) -> dict:
    """The source's verification record, if it is a complete verification against the registry
    (validate_record; production SOURCES unless a registry is injected) and every file is still
    the one verified (size and mtime); VerificationError otherwise. The build refuses sources
    without one."""
    registry = SOURCES if registry is None else registry
    if name not in registry:
        raise VerificationError(f"{name}: not a source of the registry")
    path = Path(verify_dir) / f"{name}.json"
    if not path.exists():
        raise VerificationError(f"{name}: no verification record ({path}); run `noisy_sources.py verify {name}`")
    try:
        record = json.loads(path.read_text())
    except ValueError:
        raise VerificationError(f"{name}: unreadable verification record") from None
    validate_record(name, record, registry[name])
    for f in record["files"]:
        p = Path(raw) / name / f["name"]
        if not p.is_file():
            raise VerificationError(f"{name}: {p} is missing")
        st = p.stat()
        if (st.st_size, st.st_mtime_ns) != (f["size"], f["mtime_ns"]):
            raise VerificationError(f"{name}: {p} changed after verification")
    return record


def source_entry(record: dict, spec: dict) -> dict:
    """The pool's entry for a verified source: its identity and the verified files' hashes, which
    check_pool_inputs compares with the current verification record."""
    return {"dataset": spec["dataset"], "url": spec.get("url"), "licence": spec["licence"],
            "verification": {"ok": True, "checksum_source": record.get("checksum_source"),
                             "files": [{k: f[k] for k in ("name", "size", "md5", "sha256")} for f in record["files"]]}}


# --------------------------------------------------------------------------------------------- audio records

def pcm_sha256(path: Path) -> str:
    """SHA-256 of the decoded PCM: b"<rate>,<channels>\\n" then the samples as little-endian
    float32, interleaved (soundfile decoding; exact for 16- and 24-bit PCM). The TRAIN check of a
    later plan must hash candidates the same way (noisy_blocklist.pcm_sha256 is this function)."""
    digest = hashlib.sha256()
    with sf.SoundFile(str(path)) as f:
        digest.update(f"{f.samplerate},{f.channels}\n".encode())
        while True:
            block = f.read(1 << 18, dtype="float32", always_2d=True)
            if not len(block):
                break
            digest.update(np.ascontiguousarray(block, dtype="<f4").tobytes())
    return digest.hexdigest()


def read_mono(path: Path, start: int = 0, frames: int = -1) -> np.ndarray:
    """float64 samples of a file (channels averaged)."""
    x, _ = sf.read(str(path), start=start, frames=frames, dtype="float64", always_2d=True)
    return x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]


def _record(path: Path, **attrs) -> dict:
    path = Path(path)
    info = sf.info(str(path))
    level = noisy_dsp.level_db(read_mono(path))
    rec = {"path": str(path.resolve()), "frames": int(info.frames), "sample_rate": int(info.samplerate),
           "channels": int(info.channels), "sha256": noisy_paths.sha256_file(path), "pcm_sha256": pcm_sha256(path),
           "level_db": round(level, 4) if math.isfinite(level) else None}
    rec.update({k: v for k, v in attrs.items() if v is not None})
    return dict(sorted(rec.items()))


def clip_record(path: Path, *, key: str, source: str, category: str, original: str, **attrs) -> dict:
    """A pool clip: its audio file (16 kHz for the build), provenance and hashes."""
    return _record(path, key=key, source=source, category=category, original=original, **attrs)


def rir_record(path: Path, *, key: str, source: str, room: str, original: str, **attrs) -> dict:
    return _record(path, key=key, source=source, category="rir", room=room, original=original, **attrs)


PREPARE_PARAMETERS = {"sample_rate": SR, "resampler": "scipy.signal.resample_poly (default Kaiser window), float64",
                      "output": "float32 WAV, channels kept", "version": 2}


def prepare(original: Path, prepared_dir: Path, source: str, name: str, expected_sha256: str | None = None) -> Path:
    """16 kHz float32 WAV of an original file (resample_poly; channels kept). The copy is named by
    the original's SHA-256 and PREPARE_PARAMETERS and has a sidecar recording both and its own
    decoded-PCM SHA-256; it is reused only if all three still hold, else redone."""
    original = Path(original)
    original_sha = noisy_paths.sha256_file(original)
    if expected_sha256 is not None and original_sha != expected_sha256:
        raise InputChanged(f"{original}: SHA-256 differs from its verified member record")
    params = noisy_paths.canonical_json(PREPARE_PARAMETERS)
    key = noisy_paths.sha256_bytes(f"{original_sha}|{params}".encode())[:16]
    out = Path(prepared_dir) / source / f"{name}-{key}.wav"
    side = out.with_name(out.name + ".json")
    if out.exists() and side.exists():
        meta = json.loads(side.read_text())
        if (meta.get("original_sha256"), meta.get("parameters")) == (original_sha, PREPARE_PARAMETERS) \
                and pcm_sha256(out) == meta.get("prepared_pcm_sha256"):
            return out
    x, sr = sf.read(str(original), dtype="float64", always_2d=True)
    if sr != SR:
        g = math.gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g, axis=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}.wav")
    sf.write(str(tmp), x.astype(np.float32), SR, format="WAV", subtype="FLOAT")
    os.replace(tmp, out)
    noisy_paths.write_json(side, {"original": str(original), "original_sha256": original_sha,
                                  "parameters": PREPARE_PARAMETERS, "prepared_pcm_sha256": pcm_sha256(out)})
    return out


def _prepared(original: Path, prepared_dir: Path | None, source: str, name: str, mono: bool = True,
              expected_sha256: str | None = None) -> Path:
    """The original if it already is 16 kHz (mono when required), else its prepared copy; either
    way bound to the verified member's SHA-256 when given (InputChanged otherwise)."""
    if expected_sha256 is not None and noisy_paths.sha256_file(original) != expected_sha256:
        raise InputChanged(f"{original}: SHA-256 differs from its verified member record")
    info = sf.info(str(original))
    if mono and info.channels > 1:
        raise PoolError(f"{original}: {info.channels} channels where one is expected")
    if prepared_dir is None or info.samplerate == SR:
        return Path(original)
    return prepare(original, Path(prepared_dir), source, name, expected_sha256)


def _members(members) -> dict[str, tuple[Path, str | None]]:
    """name -> (path, verified SHA-256 or None) of a member list [(root, name[, sha256])]."""
    out = {}
    for m in members:
        root, name = Path(m[0]), m[1]
        if name in out:
            raise PoolError(f"member {name} listed twice")
        out[name] = (root / name, m[2] if len(m) > 2 else None)
    return out


# --------------------------------------------------------------------------------------------- indexers

# Most specific class first: a clip's group is the first of these among its (smeared) labels.
FSD_GROUP_ORDER = {"gun": ("Gunshot_and_gunfire", "Explosion"),
                   "engine": ("Fixed-wing_aircraft", "Aircraft", "Truck", "Car", "Engine", "Vehicle"),
                   "wind": ("Wind",), "crowd": ("Shout", "Yell", "Chatter", "Crowd")}


def fsd_group(category: str, labels: list[str]) -> str:
    return "fsd50k:" + next(c for c in FSD_GROUP_ORDER[category] if c in labels)


def index_fsd50k(members, prepared_dir: Path | None = None) -> dict:
    """Category clips, exclusions and the reserved eval split of an extracted FSD50K, read only
    from its verified member list [(root, name[, sha256])] (review round 5). The reservation is the complete metadata: every
    fname of eval.csv, its uploader from the clip info, the PCM hash of its audio. A missing
    clip-info entry or uploader raises PoolError, a missing expected audio file FileNotFoundError."""
    files = _members(members)

    def one(rel: str) -> Path:
        if rel not in files:
            raise FileNotFoundError(f"{rel} is not among the verified members")
        return files[rel][0]

    eval_csv = one("FSD50K.ground_truth/eval.csv")
    info = json.loads(one("FSD50K.metadata/eval_clips_info_FSD50K.json").read_text())
    audio = {Path(n).stem: v for n, v in files.items()
             if n.startswith("FSD50K.eval_audio/") and n.endswith(".wav") and n.count("/") == 1}
    with open(eval_csv, newline="") as handle:
        labels = {row["fname"]: [s for s in row["labels"].split(",") if s] for row in csv.DictReader(handle)}
    if not labels:
        raise PoolError(f"{eval_csv}: no eval clips")
    incomplete = sorted((f for f in labels if not (info.get(f) or {}).get("uploader")), key=int)
    if incomplete:
        raise PoolError(f"FSD50K clip info lacks {len(incomplete)} eval clips or their uploaders: {incomplete[:20]}")
    missing = [f for f in sorted(labels, key=int) if f not in audio or not audio[f][0].is_file()]
    if missing:
        raise FileNotFoundError(f"FSD50K eval audio missing for {len(missing)} expected clips: {missing[:20]}")
    clips: dict[str, list[dict]] = {cat: [] for cat in FSD_CATEGORIES}
    excluded, hashes = [], []
    for stem in sorted(audio, key=lambda f: (len(f), f)):
        if stem not in labels:
            excluded.append({"fname": stem, "reason": "not_eval_split"})  # never used, never reserved
    for fname in sorted(labels, key=int):
        path, member_sha = audio[fname]
        meta = info[fname]
        digest = pcm_sha256(path)
        hashes.append(digest)
        cats = [cat for cat, classes in FSD_CATEGORIES.items() if set(classes) & set(labels[fname])]
        if not cats:
            continue
        reason = None
        if len(cats) > 1:
            reason = "multi_category"
        elif not licence_allowed(meta.get("license")):
            reason = "licence"
        elif cats[0] in NO_VOICE_CATEGORIES and VOICE_LABELS & set(labels[fname]):
            reason = "voice_label"
        else:
            sf_info = sf.info(str(path))
            if cats[0] != "gun" and sf_info.frames / sf_info.samplerate < MIN_CONTINUOUS_S:
                reason = "too_short"
        if reason:
            excluded.append({"fname": fname, "reason": reason, "categories": cats})
            continue
        clip_audio = _prepared(path, prepared_dir, "fsd50k", fname, expected_sha256=member_sha)
        rec = clip_record(clip_audio, key=f"fsd50k:{fname}", source="fsd50k", category=cats[0],
                          original=f"FSD50K.eval_audio/{fname}.wav", freesound_id=int(fname),
                          uploader=meta["uploader"], licence=licence_spdx(meta["license"]),
                          licence_url=meta["license"], title=meta.get("title"), labels=labels[fname],
                          split="eval", original_pcm_sha256=digest, group=fsd_group(cats[0], labels[fname]))
        if cats[0] != "gun" and (rec["level_db"] is None or rec["level_db"] < MIN_CLIP_DBFS):
            excluded.append({"fname": fname, "reason": "silent", "categories": cats})
            continue
        clips[cats[0]].append(rec)
    return {"clips": clips, "excluded": excluded,
            "reserved": {"sound_ids": sorted(int(f) for f in labels),
                         "uploaders": sorted({info[f]["uploader"] for f in labels}),
                         "pcm_sha256": sorted(hashes), "source": "eval.csv and eval_clips_info_FSD50K.json"}}


def index_gunshots(raw_dir: Path, prepared_dir: Path | None, verified: dict[str, str] | None = None) -> list[dict]:
    """The four Zenodo recordings, read in place; with `verified` (name -> SHA-256 from the
    verification record) each is bound to its verified download."""
    out = []
    for fname in ("mic1raw.wav", "mic2raw16b.wav", "mic3raw.wav", "mic4raw.wav"):
        original = Path(raw_dir) / fname
        stem = fname.removesuffix(".wav")
        audio = _prepared(original, prepared_dir, "gunshots-ipod", stem,
                          expected_sha256=(verified or {}).get(fname) if verified is not None else None)
        out.append(clip_record(audio, key=f"gunshots-ipod:{stem}", source="gunshots-ipod", category="gun",
                               original=fname, licence="CC0-1.0", original_pcm_sha256=pcm_sha256(original),
                               group=f"zenodo-3997406:{stem}"))
    return out


def index_demand(members) -> list[dict]:
    """All 16 channels of every 16 kHz environment (the reserved unit and the sampled pool), from
    the verified member list."""
    files = _members(members)
    out = []
    for env in DEMAND_ENVS:
        for ch in range(1, 17):
            rel = f"{env}/ch{ch:02d}.wav"
            if rel not in files:
                raise FileNotFoundError(f"DEMAND {rel} is not among the verified members")
            path = _prepared(files[rel][0], None, "demand", rel, expected_sha256=files[rel][1])
            out.append(clip_record(path, key=f"demand:{env}/ch{ch:02d}", source="demand", category="demand",
                                   original=f"{env}/ch{ch:02d}.wav", licence="CC-BY-SA-3.0", environment=env,
                                   group=f"demand:{env}"))
    return out


def index_musan(members) -> list[dict]:
    """MUSAN us-gov speech, from the verified member list only."""
    files = _members(members)
    names = sorted(n for n in files if re.fullmatch(r"musan/speech/us-gov/[^/]+\.wav", n))
    if not names:
        raise FileNotFoundError("no MUSAN us-gov speech among the verified members")
    return [clip_record(_prepared(files[n][0], None, "musan", n, expected_sha256=files[n][1]),
                        key=f"musan:{Path(n).stem}", source="musan", category="babble", original=n,
                        licence="public domain", group="musan:us-gov") for n in names]


def but_room(rel: Path) -> str:
    for part in rel.parts:
        if part in BUT_ROOMS:
            return part
    return rel.parts[0]


def index_but(members, prepared_dir: Path | None) -> list[dict]:
    """Every RIR wav (a file under an 'RIR' directory) of the BUT ReverbDB RIR-only release, from
    the verified member list only; the room is the path component naming one of the nine rooms."""
    out = []
    for name, (path, member_sha) in sorted(_members(members).items()):
        rel = Path(name)
        if rel.suffix != ".wav" or "RIR" not in rel.parts[:-1]:
            continue
        key = "but-reverbdb:" + rel.with_suffix("").as_posix()
        audio = _prepared(path, prepared_dir, "but-reverbdb", hashlib.sha256(key.encode()).hexdigest()[:24],
                          mono=False, expected_sha256=member_sha)
        room = but_room(rel)
        out.append(rir_record(audio, key=key, source="but-reverbdb", room=room, original=rel.as_posix(),
                              licence="CC-BY-4.0", original_pcm_sha256=pcm_sha256(path),
                              group=f"but-reverbdb:{room}"))
    if not out:
        raise FileNotFoundError("no RIR files among the verified members")
    return out


def index_events(event_clips: Iterable[dict]) -> list[dict]:
    """The frozen event list: noisy_dsp.detect_events over each impulsive clip (16 kHz, mono)."""
    events = []
    for clip in sorted(event_clips, key=lambda c: c["key"]):
        if clip["sample_rate"] != SR:
            raise PoolError(f"{clip['key']}: events need 16 kHz audio")
        x = read_mono(Path(clip["path"]))
        for k, (start, length) in enumerate(noisy_dsp.detect_events(x)):
            window = x[start:start + length]
            level = noisy_dsp.level_db(window)
            if not math.isfinite(level):
                continue
            events.append({"key": f"{clip['key']}#{k:04d}", "clip": clip["key"], "start": start, "length": length,
                           "level_db": round(level, 4),
                           "peak_db": round(20 * math.log10(float(np.max(np.abs(window)))), 4)})
    return events


def index_freesound_rotor(raw_dir: Path, prepared_dir: Path | None) -> tuple[list[dict], dict]:
    """Rotor plug-in "freesound": raw/freesound-rotor/sounds.json lists downloaded Freesound clips
    as {id, username, license, split, file, sha256} (Freesound API metadata plus the uploader split
    and the file's SHA-256 at download). Only CC0 clips of the "eval" split enter."""
    meta_path = Path(raw_dir) / "sounds.json"
    sounds = json.loads(meta_path.read_text())
    clips = []
    for s in sorted(sounds, key=lambda s: int(s["id"])):
        if s.get("split") != "eval" or licence_spdx_or_none(s.get("license")) != "CC0-1.0":
            continue
        original = Path(raw_dir) / Path(s["file"]).name
        if noisy_paths.sha256_file(original) != s["sha256"]:
            raise VerificationError(f"freesound {s['id']}: SHA-256 differs from the download record")
        audio = _prepared(original, prepared_dir, "freesound-rotor", str(int(s["id"])))
        clips.append(clip_record(audio, key=f"freesound:{int(s['id'])}", source="freesound-rotor", category="rotor",
                                 original=original.name, freesound_id=int(s["id"]), uploader=s["username"],
                                 licence="CC0-1.0", licence_url=s["license"], split="eval",
                                 group=f"freesound-rotor:{s['username']}"))
    source = {"dataset": "freesound", "licence": "CC0-1.0", "url": "https://freesound.org/apiv2/",
              "verification": {"ok": True, "plugin": "freesound",
                               "method": "per-file SHA-256 recorded at download in sounds.json",
                               "sounds_json_sha256": noisy_paths.sha256_file(meta_path),
                               "files": [{"name": c["original"],
                                          "sha256": noisy_paths.sha256_file(Path(raw_dir) / c["original"])}
                                         for c in clips]}}
    return clips, source


def verify_freesound_rotor(entry: dict, raw: Path) -> None:
    """The rotor plug-in's verification: sounds.json and every admitted download are still the
    ones the plug-in indexed (SHA-256), as recorded in the pool's source entry."""
    v = entry.get("verification") or {}
    if v.get("plugin") != "freesound" or not v.get("sounds_json_sha256") or not v.get("files"):
        raise VerificationError("freesound-rotor: no plug-in verification record (sounds.json and file hashes)")
    root = Path(raw) / "freesound-rotor"
    meta = root / "sounds.json"
    if not meta.is_file() or noisy_paths.sha256_file(meta) != v["sounds_json_sha256"]:
        raise InputChanged(f"{meta} differs from the sounds.json the rotor pool was indexed from")
    for f in v["files"]:
        path = root / Path(f["name"]).name
        if not path.is_file() or noisy_paths.sha256_file(path) != f["sha256"]:
            raise InputChanged(f"freesound-rotor: {path} differs from the indexed download")


ROTOR_PLUGINS = {"freesound": index_freesound_rotor}
# Registered plug-in sources: pool source name -> its verification (review round 4, finding 3).
PLUGIN_SOURCES = {"freesound-rotor": verify_freesound_rotor}


# --------------------------------------------------------------------------------------------- pool

def _check_clip(clip: dict, sources: dict, noise: bool) -> None:
    source = clip.get("source")
    dataset = str(sources.get(source, {}).get("dataset", "")).lower()
    if source in EXCLUDED_SOURCES or dataset in EXCLUDED_SOURCES:
        raise PoolError(f"{clip.get('key')}: source {source!r} is excluded ({EXCLUDED_SOURCES.get(source)})")
    if source not in SOURCES and source not in PLUGIN_SOURCES:
        raise PoolError(f"{clip.get('key')}: {source!r} is not a registered source or plug-in; a self-attested "
                        "verification is not accepted")
    if source not in sources or (sources[source].get("verification") or {}).get("ok") is not True:
        raise PoolError(f"{clip.get('key')}: source {source!r} has no ok verification record")
    if not pool_licence_allowed(clip.get("licence")):
        raise PoolError(f"{clip.get('key')}: licence {clip.get('licence')!r} is not admitted")
    if clip["sample_rate"] != SR:
        raise PoolError(f"{clip['key']}: {clip['sample_rate']} Hz, the build reads 16 kHz")
    if noise and clip["channels"] != 1:
        raise PoolError(f"{clip['key']}: noise clips are mono")


def make_pool(*, clips: dict[str, list[dict]], event_clips: list[dict], events: list[dict], rirs: list[dict],
              sources: dict, reserved: dict, missing: dict | None = None) -> dict:
    """The validated, sorted pool object (format POOL_FORMAT)."""
    for cat, items in clips.items():
        if cat not in NOISE_CATEGORIES:
            raise PoolError(f"unknown noise category {cat!r}")
        for clip in items:
            _check_clip(clip, sources, noise=True)
            if clip["frames"] <= 2 * noisy_dsp.CROSSFADE:
                raise PoolError(f"{clip['key']}: too short to loop")
    for clip in event_clips:
        _check_clip(clip, sources, noise=True)
    for rir in rirs:
        _check_clip(rir, sources, noise=False)
    seen = set()
    for item in [c for items in clips.values() for c in items] + list(event_clips) + list(rirs):
        if item["key"] in seen:
            raise PoolError(f"duplicate key {item['key']}")
        seen.add(item["key"])
    by_key = {c["key"] for c in event_clips}
    for e in events:
        if e["clip"] not in by_key:
            raise PoolError(f"event {e['key']} of an unknown clip")
    return {"format": POOL_FORMAT,
            "sources": dict(sorted(sources.items())),
            "clips": {cat: sorted(items, key=lambda c: c["key"]) for cat, items in sorted(clips.items()) if items},
            "event_clips": sorted(event_clips, key=lambda c: c["key"]),
            "events": sorted(events, key=lambda e: e["key"]),
            "rirs": sorted(rirs, key=lambda r: r["key"]),
            "reserved": json.loads(noisy_paths.canonical_json(reserved)),
            "missing": dict(sorted((missing or {}).items()))}


def write_pool(path: Path, pool: dict) -> None:
    noisy_paths.write_text(path, json.dumps(pool, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def write_rotor_pool(path: Path, clips: list[dict], *, plugin: str, source: dict) -> dict:
    """pool/rotor.json of a rotor plug-in: CC0 clips only (plans/build.md: rotor must be CC0)."""
    if plugin not in ROTOR_PLUGINS:
        raise PoolError(f"unknown rotor plug-in {plugin!r}")
    sources = {c["source"]: source for c in clips}
    for clip in clips:
        _check_clip(clip, sources, noise=True)
        if clip["licence"] != "CC0-1.0" or clip["category"] != "rotor":
            raise PoolError(f"{clip['key']}: rotor clips must be CC0 and category rotor")
    obj = {"format": ROTOR_FORMAT, "plugin": plugin, "source": source,
           "clips": sorted(clips, key=lambda c: c["key"])}
    noisy_paths.write_text(path, json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return obj


def load_pool(path: Path, rotor_path: Path | None = None) -> dict:
    """The pool, re-validated; with rotor_path, the rotor plug-in's clips as category "rotor"."""
    pool = json.loads(Path(path).read_text())
    if pool.get("format") != POOL_FORMAT:
        raise PoolError(f"{path}: not a {POOL_FORMAT} pool")
    clips = {cat: list(items) for cat, items in pool["clips"].items()}
    sources = dict(pool["sources"])
    reserved = dict(pool["reserved"])
    if "rotor" in clips:
        raise PoolError("rotor clips come only from a rotor plug-in's pool")
    if rotor_path is not None:
        rotor = json.loads(Path(rotor_path).read_text())
        if rotor.get("format") != ROTOR_FORMAT:
            raise PoolError(f"{rotor_path}: not a {ROTOR_FORMAT} pool")
        clips["rotor"] = rotor["clips"]
        for clip in rotor["clips"]:
            sources[clip["source"]] = rotor["source"]
        reserved["rotor"] = {"plugin": rotor["plugin"],
                             "sound_ids": sorted({c["freesound_id"] for c in rotor["clips"] if "freesound_id" in c}),
                             "uploaders": sorted({c["uploader"] for c in rotor["clips"] if "uploader" in c})}
    return make_pool(clips=clips, event_clips=pool["event_clips"], events=pool["events"], rirs=pool["rirs"],
                     sources=sources, reserved=reserved, missing=pool.get("missing"))


def pool_files(pool: dict) -> list[dict]:
    """Every file of the reserved pool with its decoded-PCM SHA-256 (original and, if different,
    prepared), plus the FSD50K eval clips reserved without being sampled."""
    out = []
    for item in _pool_audio(pool):
        out.append({"key": item["key"], "source": item["source"], "pcm_sha256": item["pcm_sha256"]})
        if item.get("original_pcm_sha256") and item["original_pcm_sha256"] != item["pcm_sha256"]:
            out.append({"key": item["key"] + "@original", "source": item["source"],
                        "pcm_sha256": item["original_pcm_sha256"]})
    for k, digest in enumerate((pool["reserved"].get("fsd50k") or {}).get("pcm_sha256", [])):
        out.append({"key": f"fsd50k:reserved#{k}", "source": "fsd50k", "pcm_sha256": digest})
    return out


# --------------------------------------------------------------------------------------------- extraction

def _pool_audio(pool: dict) -> list[dict]:
    return [c for items in pool["clips"].values() for c in items] + pool["event_clips"] + pool["rirs"]


def _pcm_job(path: str) -> str:
    return pcm_sha256(Path(path))


def check_pool_inputs(pool: dict, raw: Path = noisy_paths.RAW, verify_dir: Path = noisy_paths.VERIFY,
                      workers: int = 1, registry: dict | None = None) -> dict:
    """Before every build and resume: each registry source of the pool still has an ok
    verification record (require_verified: files unchanged since) for exactly the downloads the
    pool was built from, and every pool audio file still decodes to its recorded PCM SHA-256.
    Raises VerificationError or InputChanged; returns a summary for the build record."""
    for name, entry in pool["sources"].items():
        if name in PLUGIN_SOURCES:
            PLUGIN_SOURCES[name](entry, raw)  # the plug-in's own verification
            continue
        if name not in (SOURCES if registry is None else registry):
            raise VerificationError(f"{name}: not a registered source or plug-in; refusing a self-attested "
                                    "verification")
        record = require_verified(name, verify_dir, raw, registry)
        current = sorted((f["name"], f["size"], f["sha256"]) for f in record["files"])
        pooled = sorted((f["name"], f["size"], f["sha256"]) for f in (entry.get("verification") or {}).get("files", []))
        if current != pooled:
            raise InputChanged(f"{name}: the verified downloads differ from those the pool was built from")
    items = _pool_audio(pool)
    paths = [item["path"] for item in items]
    if workers > 1 and len(paths) > 1:
        with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            digests = list(ex.map(_pcm_job, paths, chunksize=16))
    else:
        digests = [_pcm_job(p) for p in paths]
    changed = [item["key"] for item, digest in zip(items, digests, strict=True) if digest != item["pcm_sha256"]]
    if changed:
        raise InputChanged(f"{len(changed)} pool audio files changed since the pool was indexed: {changed[:10]}")
    return {"sources_checked": sorted(pool["sources"]), "files_checked": len(items)}


EXTRACTOR_SHA256 = noisy_paths.sha256_file(noisy_paths.HERE / "noisy_extract.py")


def extract_archive(source: str, archive_name: str, segments: list[Path], kind: str, select: tuple,
                    budget: noisy_extract.Budget, verification: dict, root: Path = noisy_paths.EXTRACTED) -> dict:
    """Extract (or reuse) one archive. The destination root/<source>/<archive>-<key> is keyed by
    the archive segments' verified SHA-256, the selection (select = (id, predicate)), the budget
    and the extractor's code; a reused extraction must still hold every member with its recorded
    size and SHA-256 (InputChanged otherwise)."""
    verified = {f["name"]: f["sha256"] for f in verification["files"]}
    names = [Path(p).name for p in segments]
    if any(n not in verified for n in names):
        raise VerificationError(f"{source}/{archive_name}: segments {names} not all verified")
    inputs = {"archives": [[n, verified[n]] for n in names], "select": select[0],
              "budget": dataclasses.asdict(budget), "extractor_sha256": EXTRACTOR_SHA256}
    key = noisy_paths.sha256_bytes(noisy_paths.canonical_json(inputs).encode())[:16]
    dest = Path(root) / source / f"{archive_name}-{key}"
    marker = dest.with_name(f"{dest.name}.done.json")
    if marker.exists():
        record = json.loads(marker.read_text())
        if record.get("inputs") != inputs:
            raise InputChanged(f"{marker}: recorded inputs differ")
        expected = {m["name"] for m in record["member_list"]}
        present = set()
        for base, dirs, files in os.walk(dest):  # no links followed
            for entry in dirs + files:
                path = Path(base) / entry
                rel = path.relative_to(dest).as_posix()
                if path.is_symlink() or (entry in files and not path.is_file()):
                    raise InputChanged(f"{source}/{archive_name}: unexpected link or special file {rel} in {dest}")
                if entry in files:
                    present.add(rel)
        if present != expected:
            raise InputChanged(f"{source}/{archive_name}: unexpected files {sorted(present - expected)[:10]} or "
                               f"missing members {sorted(expected - present)[:10]} in {dest}")
        for m in record["member_list"]:
            path = dest / m["name"]
            if path.stat().st_size != m["size"] or noisy_paths.sha256_file(path) != m["sha256"]:
                raise InputChanged(f"{source}/{archive_name}: extracted member {m['name']} changed; remove {dest}")
        return record
    if dest.exists():
        raise FileExistsError(f"{dest} exists without a done marker; remove it to re-extract")
    log(f"extract {source}/{archive_name} -> {dest.name}")
    members = (noisy_extract.extract_zip(segments, dest, select[1], budget) if kind == "zip"
               else noisy_extract.extract_tar(segments[0], dest, select[1], budget))
    for m in members:
        m["sha256"] = noisy_paths.sha256_file(dest / m["name"])
    record = {"source": source, "archive": names, "dest": str(dest), "key": key, "inputs": inputs,
              "members": len(members), "bytes": sum(m["size"] for m in members), "member_list": members,
              "extracted_utc": utc()}
    noisy_paths.write_json(marker, record)
    return record


def extract(source: str, raw: Path = noisy_paths.RAW, verify_dir: Path = noisy_paths.VERIFY,
            root: Path = noisy_paths.EXTRACTED) -> list[tuple[Path, str, str]]:
    """Extract a verified source's archives (selected members only); returns the verified member
    list [(destination, member name, SHA-256)], the only files the indexers read."""
    noisy_paths.require_mount()
    record = require_verified(source, verify_dir, raw)
    r = Path(raw) / source
    budgets = EXTRACTION_BUDGETS.get(source, {})

    def one(archive, segments, kind, select):
        rec = extract_archive(source, archive, segments, kind, select, budgets[archive], record, root)
        return [(Path(rec["dest"]), m["name"], m["sha256"]) for m in rec["member_list"]]

    if source == "fsd50k":
        return [m for ms in [one("ground_truth", [r / "FSD50K.ground_truth.zip"], "zip", ("csv", lambda n: n.endswith(".csv"))),
                one("metadata", [r / "FSD50K.metadata.zip"], "zip",
                    ("clip-and-class-info", lambda n: n.endswith(("eval_clips_info_FSD50K.json",
                                                                  "class_info_FSD50K.json")))),
                one("eval_audio", [r / "FSD50K.eval_audio.z01", r / "FSD50K.eval_audio.zip"], "zip",
                    ("eval-wav", lambda n: n.startswith("FSD50K.eval_audio/") and n.endswith(".wav")))] for m in ms]
    if source == "demand":
        return [m for env in DEMAND_ENVS for m in one(
            f"{env}_16k", [r / f"{env}_16k.zip"], "zip",
            (f"{env}-ch01-16", lambda n, env=env: re.fullmatch(rf"{env}/ch(0[1-9]|1[0-6])\.wav", n) is not None))]
    if source == "musan":
        return one("musan", [r / "musan.tar.gz"], "tar",
                   ("us-gov+readme", lambda n: n.startswith("musan/speech/us-gov/") or n == "musan/README"))
    if source == "but-reverbdb":
        return one("rir-only", [r / "BUT_ReverbDB_rel_19_06_RIR-Only.tgz"], "tar",
                   ("rir-wav", lambda n: n.endswith(".wav") and "RIR" in Path(n).parts[:-1]))
    if source == "gunshots-ipod":
        return []  # plain WAV files, read in place by path
    raise ValueError(f"unknown source {source!r}")


# --------------------------------------------------------------------------------------------- pool build

def build_pool(allow_missing: bool = False, out: Path = noisy_paths.POOL / "pool.json") -> dict:
    """Index every verified, extracted source into the pool. allow_missing (sample builds only)
    skips sources that are not downloaded or verified yet and records them as missing."""
    noisy_paths.require_mount()
    prepared = noisy_paths.POOL / "audio"
    clips: dict[str, list[dict]] = {}
    event_clips, rirs, sources, reserved, missing = [], [], {}, {}, {}
    for name, spec in SOURCES.items():
        try:
            verification = require_verified(name)
            members = extract(name)
        except (VerificationError, FileNotFoundError) as error:
            if not allow_missing:
                raise
            missing[name] = str(error)
            log(f"{name}: missing ({error})")
            continue
        sources[name] = source_entry(verification, spec)
        log(f"index {name}")
        if name == "fsd50k":
            index = index_fsd50k(members, prepared)
            event_clips += index["clips"].pop("gun")
            for cat, items in index["clips"].items():
                clips.setdefault(cat, []).extend(items)
            reserved["fsd50k"] = {**index["reserved"], "excluded": index["excluded"]}
        elif name == "gunshots-ipod":
            event_clips += index_gunshots(noisy_paths.RAW / name, prepared,
                                          {f["name"]: f["sha256"] for f in verification["files"]})
        elif name == "demand":
            clips["demand"] = index_demand(members)
        elif name == "musan":
            clips["babble"] = index_musan(members)
        elif name == "but-reverbdb":
            rirs += index_but(members, prepared)
    events = index_events(event_clips)
    pool = make_pool(clips=clips, event_clips=event_clips, events=events, rirs=rirs, sources=sources,
                     reserved=reserved, missing=missing)
    write_pool(out, pool)
    log(f"pool: {sum(len(v) for v in pool['clips'].values())} noise clips, {len(pool['event_clips'])} impulsive "
        f"clips with {len(pool['events'])} events, {len(pool['rirs'])} RIRs; missing {sorted(missing)}")
    return pool


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["verify", "extract", "index", "rotor", "status"])
    parser.add_argument("sources", nargs="*")
    parser.add_argument("--allow-missing", action="store_true", help="index: skip unverified sources (samples only)")
    parser.add_argument("--plugin", choices=sorted(ROTOR_PLUGINS))
    args = parser.parse_args(argv)
    noisy_paths.require_mount()
    names = args.sources or list(SOURCES)
    if args.command == "verify":
        failed = []
        for name in names:
            try:
                record = verify_source(name)
                log(f"{name}: ok ({len(record['files'])} files)")
            except VerificationError as error:
                failed.append(name)
                log(f"{name}: REFUSED: {error}")
        sys.exit(1 if failed else 0)
    if args.command == "extract":
        for name in names:
            extract(name)
    elif args.command == "index":
        build_pool(allow_missing=args.allow_missing)
    elif args.command == "rotor":
        if not args.plugin:
            parser.error("--plugin is required")
        clips, source = ROTOR_PLUGINS[args.plugin](noisy_paths.RAW / "freesound-rotor", noisy_paths.POOL / "audio")
        write_rotor_pool(noisy_paths.POOL / "rotor.json", clips, plugin=args.plugin, source=source)
        log(f"rotor pool: {len(clips)} clips")
    else:
        for name in names:
            path = noisy_paths.VERIFY / f"{name}.json"
            state = "verified" if path.exists() else "not verified"
            done = [f for f in SOURCES[name]["files"] if (noisy_paths.RAW / name / f"{f}.done").exists()]
            print(f"{name}: {state}; downloads done {len(done)}/{len(SOURCES[name]['files'])}")


if __name__ == "__main__":
    main()
