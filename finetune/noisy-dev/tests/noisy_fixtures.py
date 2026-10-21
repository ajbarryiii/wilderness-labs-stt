"""Shared synthetic fixtures for the noisy-dev contract tests (plans/build.md "Code and tests").

Everything is synthetic and small: speech-like harmonic bursts with pauses, noise clips, impulse
responses and dev manifests in the layout of parakeet-ternary data.py, written under
noisy_paths.TMP / "tests" (on /mnt/hd). Nothing reads the real downloads. Run the suite with

    ./python -m unittest discover -s tests -v

(CPU only; the real Codec2 binary is used, everything else is numpy/scipy/soundfile.)
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # nothing here may touch the GPU

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import noisy_paths  # noqa: E402
import noisy_sources  # noqa: E402

SR = noisy_paths.SAMPLE_RATE
TEST_ROOT = noisy_paths.TMP / "tests"


def fresh_dir(name: str) -> Path:
    noisy_paths.require_mount()
    path = TEST_ROOT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def speech_like(seconds: float, seed: int, level_db: float = -26.0, sr: int = SR,
                pauses: tuple[float, float] = (0.05, 0.3)) -> np.ndarray:
    """Harmonic bursts (100-250 Hz pitch, 80-400 ms) separated by pauses (default 50-300 ms), at level_db dBFS RMS."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sr))
    t = np.arange(n) / sr
    x = np.zeros(n)
    pos = rng.uniform(0.02, 0.15)
    while pos < seconds - 0.05:
        dur = min(rng.uniform(0.08, 0.4), seconds - pos)
        f0 = rng.uniform(100, 250)
        seg = (t >= pos) & (t < pos + dur)
        k = int(seg.sum())
        if k > 1:
            burst = sum(np.sin(2 * np.pi * f0 * h * t[seg] + rng.uniform(0, 6)) / h
                        for h in range(1, int(3800 / f0)))
            x[seg] = burst * np.hanning(k) ** 0.25
        pos += dur + rng.uniform(*pauses)
    x += 1e-4 * rng.standard_normal(n)  # a noise floor, as in real recordings
    return x * (10 ** (level_db / 20) / np.sqrt(np.mean(x ** 2)))


def noise_clip(seconds: float, seed: int, color: str = "white", sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(round(seconds * sr)))
    if color == "brown":
        x = np.cumsum(x)
        x -= np.convolve(x, np.ones(401) / 401, mode="same")  # remove the drift
    return 0.1 * x / np.sqrt(np.mean(x ** 2))


def bursts(seconds: float, onsets: list[float], seed: int, sr: int = SR) -> np.ndarray:
    """Impulsive clip: decaying noise bursts (about 150 ms each) at the given onsets, near silence elsewhere."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sr))
    x = 1e-5 * rng.standard_normal(n)
    for on in onsets:
        i = int(on * sr)
        k = min(int(0.15 * sr), n - i)
        x[i:i + k] += 0.5 * rng.standard_normal(k) * np.exp(-np.arange(k) / (0.03 * sr))
    return x


def write_audio(path: Path, x: np.ndarray, sr: int = SR, subtype: str = "PCM_16") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "FLAC" if path.suffix == ".flac" else "WAV"
    sf.write(path, np.asarray(x, dtype=np.float64), sr, format=fmt, subtype=subtype)
    return path


# Ids in the formats of the real dev manifests, so that cluster parsing is exercised.
DEV_IDS = {
    "librispeech_dev_clean": ["librispeech:1272-128104-0000", "librispeech:1272-128104-0001",
                              "librispeech:2277-149896-0000", "librispeech:3752-4944-0012"],
    "librispeech_dev_other": ["librispeech:116-288045-0000", "librispeech:116-288045-0001",
                              "librispeech:1255-138279-0003"],
    "ami_dev": ["ami:AMI_ES2011a_H00_FEE041_0003427_0003714", "ami:AMI_ES2011a_H01_FEE042_0004000_0004010",
                "ami:AMI_IB4010_H02_MIO066_0011000_0011500", "ami:AMI_TS3004c_H00_MTD009PM_0020000_0020300"],
    "voxpopuli_dev": ["voxpopuli:20090203-0900-PLENARY-11-en_20090203-19_30_15_2",
                      "voxpopuli:20160526-0900-PLENARY-3-en_20160526-10:18:50_2",
                      "voxpopuli:20180531-0900-PLENARY-15-en_20180531-15:00:20_3"],
    "yodas_dev": ["yodas:en000_00000316_gDJb_EmZIeQ_294_94_2_54", "yodas:en000_00000316_gDJb_EmZIeQ_300_00_3_10",
                  "yodas:en000_00000400_1FKsjISwPmu_10_00_4_00"],
}
SHORT_ID = "ami:AMI_ES2011a_H01_FEE042_0004000_0004010"   # 0.2 s: flagged "short"
SILENT_ID = "librispeech:2277-149896-0000"                  # -80 dBFS: flagged "silent"
VOX_SPEAKERS = {"20090203-0900-PLENARY-11-en_20090203-19_30_15_2": "96718",
                "20180531-0900-PLENARY-15-en_20180531-15:00:20_3": "124989"}  # the other has no speaker


def make_dev_sets(root: Path) -> Path:
    """MANIFESTS-style directory with <set>.jsonl for the five dev sets (synthetic audio, 0.2-7.5 s)."""
    manifests = Path(root) / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    seed = 100
    for name, ids in DEV_IDS.items():
        rows = []
        for k, uid in enumerate(ids):
            seed += 1
            seconds = 0.2 if uid == SHORT_ID else [1.3, 2.9, 4.4, 7.5][k % 4]
            x = speech_like(seconds, seed, level_db=-80.0 if uid == SILENT_ID else -26.0 + 3 * (k % 3))
            path = write_audio(Path(root) / "audio" / name / f"{k:02d}.flac", x)
            rows.append({"audio_filepath": str(path), "duration": len(x) / SR,
                         "text": f"reference {name} {k}", "id": uid, "source": name.split("_")[0]})
        rows.sort(key=lambda r: r["id"])
        (manifests / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifests


POOL_SOURCES = ("fsd50k", "demand", "musan", "gunshots-ipod", "but-reverbdb")


def source_dirs(pool_path: Path) -> tuple[Path, Path]:
    """(raw, verify) directories of a fixture pool: real download files and verification records."""
    root = Path(pool_path).parent
    return root / "raw", root / "verify"


def registry(pool_path: Path) -> dict:
    """The test registry the fixture pool's sources were verified against (injected explicitly into
    the build and checks; production SOURCES never vouch for test records)."""
    return json.loads((Path(pool_path).parent / "registry.json").read_text())


def members_of(root: Path) -> list[tuple[Path, str]]:
    """A fixture directory as a verified member list [(root, relative name)] (what extraction records)."""
    root = Path(root)
    return [(root, p.relative_to(root).as_posix()) for p in sorted(root.rglob("*")) if p.is_file()]


def make_sources(root: Path) -> dict:
    """A raw download per pool source (with its .done marker), verified against a test registry
    with noisy_sources.verify_source; returns the pool's source entries."""
    import hashlib
    raw, verify = Path(root) / "raw", Path(root) / "verify"
    registry, entries = {}, {}
    for name in POOL_SOURCES:
        payload = f"download of {name}".encode()
        (raw / name).mkdir(parents=True, exist_ok=True)
        (raw / name / f"{name}.bin").write_bytes(payload)
        (raw / name / f"{name}.bin.done").touch()
        registry[name] = {"dataset": name, "licence": "test", "url": f"https://example.org/{name}",
                          "checksum": "md5", "files": {f"{name}.bin": hashlib.md5(payload).hexdigest()}}
        record = noisy_sources.verify_source(name, raw=raw, verify_dir=verify, registry=registry)
        entries[name] = noisy_sources.source_entry(record, registry[name])
    (Path(root) / "registry.json").write_text(json.dumps(registry))
    return entries


def make_pool(root: Path, rotor: bool = False) -> Path:
    """A pool index (noisy_sources pool format) over synthetic clips; rotor clips go to rotor.json.
    Every source has a real verification record under root/verify (source_dirs)."""
    root = Path(root)
    audio = root / "pool-audio"
    rec = noisy_sources.clip_record

    def clip(key, source, category, x, sr=SR, subtype="PCM_16", **attrs):
        path = write_audio(audio / category / f"{key.replace(':', '_')}.wav", x, sr, subtype)
        return rec(path, key=key, source=source, category=category, original=f"{category}/{path.name}", **attrs)

    fs = {"licence": "CC0-1.0", "licence_url": "http://creativecommons.org/publicdomain/zero/1.0/"}
    clips = {
        "engine": [clip("fsd50k:1001", "fsd50k", "engine", noise_clip(9.0, 1, "brown"), freesound_id=1001,
                        uploader="u-engine", group="fsd50k:Car", **fs),
                   clip("fsd50k:1002", "fsd50k", "engine", noise_clip(1.2, 2), freesound_id=1002,
                        uploader="u-engine2", group="fsd50k:Truck", **fs)],  # shorter than most utterances: looped
        "wind": [clip("fsd50k:2001", "fsd50k", "wind", noise_clip(6.0, 3, "brown"), freesound_id=2001,
                      uploader="u-wind", group="fsd50k:Wind", **fs)],
        "crowd": [clip("fsd50k:3001", "fsd50k", "crowd", speech_like(5.0, 4, -20), freesound_id=3001,
                       uploader="u-crowd", group="fsd50k:Crowd", **fs)],
        "demand": [clip(f"demand:{env}/ch{c:02d}", "demand", "demand", noise_clip(12.0, 10 + c, "brown"),
                        licence="CC-BY-SA-3.0", group=f"demand:{env}") for c, env in ((1, "TBUS"), (2, "NPARK"))],
        "babble": [clip(f"musan:speech-us-gov-{i:04d}", "musan", "babble",
                        speech_like(20.0, 20 + i, -24, pauses=(0.02, 0.08)),
                        licence="public domain", group="musan:us-gov") for i in range(7)],
    }
    gun_clips = [clip("fsd50k:4001", "fsd50k", "gun", bursts(2.0, [0.3, 1.1], 30), freesound_id=4001,
                      uploader="u-gun", group="fsd50k:Gunshot_and_gunfire", **fs),
                 clip("gunshots-ipod:mic1raw", "gunshots-ipod", "gun", bursts(8.0, [1.0, 3.0, 5.5], 31),
                      licence="CC0-1.0", group="zenodo-3997406:mic1raw")]
    # Two clips with one sustained 2.5 s event each: longer than the 1.3 s utterances, so some
    # placements are truncated (with the 10 ms fade).
    for k, mic in enumerate(("mic2raw16b", "mic3raw")):
        x = 1e-5 * np.random.default_rng(32 + k).standard_normal(3 * SR)
        x[SR // 4: SR // 4 + int(2.5 * SR)] += 0.2 * np.random.default_rng(34 + k).standard_normal(int(2.5 * SR))
        gun_clips.append(clip(f"gunshots-ipod:{mic}", "gunshots-ipod", "gun", x, licence="CC0-1.0",
                              group=f"zenodo-3997406:{mic}"))
    events = noisy_sources.index_events(gun_clips)
    rirs = []
    for k, (rt, channels) in enumerate([(0.15, 1), (0.4, 2), (0.25, 1)]):
        rng = np.random.default_rng(40 + k)
        n = int(rt * 1.5 * SR)
        h = rng.standard_normal((n, channels)) * np.exp(-np.arange(n) / (rt / 6.9 * SR))[:, None] * 0.05
        h[:20] = 0.0
        h[20] = 1.0  # direct path after 20 samples of pre-delay
        path = write_audio(audio / "rir" / f"room{k}.wav", h, subtype="FLOAT")
        rirs.append(noisy_sources.rir_record(path, key=f"but-reverbdb:Room{k}/mic01", source="but-reverbdb",
                                             room=f"Room{k}", original=f"Room{k}/RIR/IR.wav",
                                             licence="CC-BY-4.0", group=f"but-reverbdb:Room{k}"))
    pool = noisy_sources.make_pool(clips=clips, event_clips=gun_clips, events=events, rirs=rirs,
                                   sources=make_sources(root),
                                   reserved={"fsd50k": {"sound_ids": [1001, 1002, 2001, 3001, 4001, 4002],
                                                        "uploaders": ["u-engine", "u-engine2", "u-wind", "u-crowd",
                                                                      "u-gun", "u-other"]}})
    noisy_sources.write_pool(root / "pool.json", pool)
    if rotor:  # through the real "freesound" plug-in: downloads plus sounds.json under raw/freesound-rotor
        import hashlib
        rotor_raw = root / "raw" / "freesound-rotor"
        sounds = []
        for sid, user, x in ((9001, "u-rotor", noise_clip(4.0, 50, "brown")), (9002, "u-rotor2", noise_clip(0.9, 51)),
                             (9003, "u-nc", noise_clip(2.0, 52))):  # 9003: not CC0, never enters
            path = write_audio(rotor_raw / f"{sid}.wav", x)
            sounds.append({"id": sid, "username": user, "split": "eval", "file": path.name,
                           "license": fs["licence_url"] if sid != 9003 else "http://creativecommons.org/licenses/by-nc/3.0/",
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        (rotor_raw / "sounds.json").write_text(json.dumps(sounds))
        rotor_clips, source = noisy_sources.index_freesound_rotor(rotor_raw, None)
        noisy_sources.write_rotor_pool(root / "rotor.json", rotor_clips, plugin="freesound", source=source)
    return root / "pool.json"


def tiny_build(root: Path, workers: int = 1, rotor: bool = False, sample: int | None = None, out: str = "out"):
    """A complete synthetic build under root/<out> (the real builder); returns (out, BuildConfig).
    With sample, a sample build of that many utterances (allow_missing, as the CLI's --sample)."""
    import noisy_build
    manifests = make_dev_sets(Path(root) / "dev") if not (Path(root) / "dev").exists() else Path(root) / "dev" / "manifests"
    pool = Path(root) / "pool" / "pool.json"
    if not pool.exists():
        pool = make_pool(Path(root) / "pool", rotor=rotor)
    raw, verify = source_dirs(pool)
    cfg = noisy_build.BuildConfig(out=Path(root) / out, manifests=manifests, pool=pool,
                                  rotor_pool=pool.parent / "rotor.json" if rotor else None, workers=workers,
                                  voxpopuli_speakers=VOX_SPEAKERS, raw=raw, verify_dir=verify, sample=sample,
                                  allow_missing=sample is not None, registry=registry(pool))
    noisy_build.build(cfg)
    return cfg.out, cfg


def read_pcm(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float64")
    assert sr == SR
    return x


def build_rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in (Path(out) / "build_manifest.jsonl").read_text().splitlines() if line]


def flac_files(out: Path) -> dict[str, bytes]:
    return {str(p.relative_to(out)): p.read_bytes() for p in sorted((Path(out) / "audio").rglob("*.flac"))}
