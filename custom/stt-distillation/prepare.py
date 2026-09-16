"""Construct independent pilot splits without using official final holdouts."""

import collections, concurrent.futures, hashlib, json, math, re
from pathlib import Path
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from scipy.signal import resample_poly
from common import ART, REPO, save, digest, norm, encode, storage

OUT = ART / "datasets/pilot"


def rank(s):
    return hashlib.sha256(str(s).encode()).hexdigest()


def feature(x):
    x = np.pad(x, (240, max(0, 160 - len(x))))
    frames = np.lib.stride_tricks.sliding_window_view(x, 400)[::160]
    window = np.hanning(400).astype(np.float32)
    power = abs(np.fft.rfft(frames * window, n=512)) ** 2 / (window**2).sum()
    hz = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 82) / 2595) - 1)
    f = np.fft.rfftfreq(512, 1 / 16000)
    bank = np.stack(
        [
            np.maximum(
                0,
                np.minimum(
                    (f - hz[i]) / (hz[i + 1] - hz[i]),
                    (hz[i + 2] - f) / (hz[i + 2] - hz[i + 1]),
                ),
            )
            for i in range(80)
        ]
    )
    return (np.log(np.maximum(power @ bank.T, 1e-8)).clip(-18, 6) / 6 + 1).T.astype(
        np.float32
    )


def write(row, x):
    name = row["id"]
    a = OUT / "audio" / f"{name}.wav"
    f = OUT / "features" / f"{name}.npy"
    if x is None:
        x, sr = sf.read(row.pop("source_audio"), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != 16000:
            x = resample_poly(
                x, 16000 // math.gcd(sr, 16000), sr // math.gcd(sr, 16000)
            ).astype(np.float32)
    if not 0.3 <= len(x) / 16000 <= 15:
        return None
    x = np.asarray(x, dtype=np.float32)
    feat = feature(x)
    y = encode(row["text"])
    if (feat.shape[-1] + 1) // 2 < len(y) + sum(a == b for a, b in zip(y, y[1:])):
        return None
    sf.write(a, x, 16000, subtype="FLOAT")
    np.save(f, feat)
    row.update(
        audio=str(a),
        features=str(f),
        audio_sha256=digest(a),
        feature_sha256=digest(f),
        seconds=len(x) / 16000,
        frames=feat.shape[-1],
    )
    return row


def prepare():
    storage()
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    (OUT / "features").mkdir(exist_ok=True)
    candidates = []
    excluded = set()
    old = REPO / "custom/stt/cache/manifest.json"
    if old.exists():
        excluded = {
            str(r["speaker_id"])
            for r in json.loads(old.read_text())["rows"]
            if r.get("speaker_id") is not None
        }
    libri = ART / "datasets/libri/LibriSpeech/train-clean-100"
    speakers = sorted(
        [p.name for p in libri.iterdir() if p.is_dir() and p.name not in excluded],
        key=rank,
    )
    assert len(speakers) > 30
    sp = {
        s: ("calibration" if i < 12 else "development" if i < 24 else "train")
        for i, s in enumerate(speakers)
    }
    groups = collections.defaultdict(list)
    for file in libri.rglob("*.trans.txt"):
        speaker = file.parent.parent.name
        if speaker not in sp:
            continue
        for line in file.read_text().splitlines():
            name, text = line.split(" ", 1)
            path = file.parent / (name + ".flac")
            info = sf.info(path)
            if 3 <= info.duration <= 12:
                groups[sp[speaker]].append(
                    (
                        dict(
                            id="libri-" + name,
                            domain="general",
                            split=sp[speaker],
                            speaker=speaker,
                            text=norm(text),
                            source_audio=str(path),
                            source="LibriSpeech train-clean-100",
                        ),
                        None,
                    )
                )
    for split, rows in groups.items():
        candidates.extend(
            sorted(rows, key=lambda x: rank(x[0]["id"]))[
                : (2400 if split == "train" else 64)
            ]
        )
    # Medical corpus lacks reliable speaker IDs. Use only a transcript-disjoint
    # diagnostic split and explicitly prohibit speaker-generalization claims.
    med = {}
    bad = 0
    for file in sorted((ART / "datasets/medical/data").glob("train-*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=32):
            for r in batch.to_pylist():
                text = norm(r["sentence"])
                audio = r["audio"]
                x = np.asarray(audio["array"], dtype=np.float32).squeeze()
                if x.ndim == 2 and x.shape[0] <= 2:
                    x = x.mean(axis=0)
                if x.ndim != 1 or not text:
                    bad += 1
                    continue
                sr = audio["sampling_rate"]
                if sr != 16000:
                    x = resample_poly(
                        x, 16000 // math.gcd(sr, 16000), sr // math.gcd(sr, 16000)
                    ).astype(np.float32)
                if not 1 <= len(x) / 16000 <= 12:
                    continue
                ph = rank(text)
                split = (
                    "calibration"
                    if int(ph[:8], 16) % 10 == 0
                    else "development"
                    if int(ph[:8], 16) % 10 == 1
                    else "train"
                )
                ident = "med-" + rank(str(audio["path"]) + text)[:20]
                med[ident] = (
                    dict(
                        id=ident,
                        domain="medical_symptoms",
                        split=split,
                        speaker=None,
                        text=text,
                        source="Hani89 medical_asr_recording_dataset; speaker identity unavailable",
                        phrase_group=ph,
                    ),
                    x,
                )
    for split, limit in [("train", 700), ("calibration", 48), ("development", 48)]:
        candidates.extend(
            sorted(
                [v for v in med.values() if v[0]["split"] == split],
                key=lambda x: rank(x[0]["id"]),
            )[:limit]
        )
    rev = json.loads((ART / "datasets/digits-revision.json").read_text())["revision"]
    df = ART / "datasets" / ("free-spoken-digit-dataset-" + rev) / "recordings"
    dg = collections.defaultdict(list)
    for p in sorted(df.glob("*.wav")):
        digit, speaker, idx = p.stem.split("_")
        if int(idx) >= 5:
            dg[speaker].append((p, digit))  # Official test indices 0–4 remain unused.
    speakers = sorted(dg, key=rank)
    assert len(speakers) >= 6
    for i, speaker in enumerate(speakers):
        split = "calibration" if i == 0 else "development" if i == 1 else "train"
        rng = np.random.default_rng(20260910 + i)
        pool = dg[speaker]
        for j in range(100 if split == "train" else 48):
            parts = []
            digits = []
            for ix in rng.choice(
                len(pool), size=int(rng.integers(3, 7)), replace=False
            ):
                p, digit = pool[ix]
                x, sr = sf.read(p, dtype="float32")
                x = resample_poly(
                    x, 16000 // math.gcd(sr, 16000), sr // math.gcd(sr, 16000)
                ).astype(np.float32)
                parts.extend([x, np.zeros(2400, dtype=np.float32)])
                digits.append(digit)
            candidates.append(
                (
                    dict(
                        id=f"digit-{speaker}-{j}",
                        domain="digits",
                        split=split,
                        speaker=speaker,
                        text=" ".join(digits),
                        source="FSDD training recordings; concatenated digit diagnostic",
                    ),
                    np.concatenate([np.zeros(3200, dtype=np.float32), *parts]),
                )
            )
    # Per-source exact audio duplicates cannot cross splits.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = [r for r in pool.map(lambda v: write(*v), candidates) if r]
    seen = {}
    for r in rows:
        h = r["audio_sha256"]
        if h in seen and seen[h] != r["split"]:
            raise ValueError("Cross-split audio duplicate")
        seen[h] = r["split"]
    counts = collections.Counter((r["split"], r["domain"]) for r in rows)
    sources = dict(
        libri=dict(
            url="https://www.openslr.org/12/",
            license="CC-BY-4.0",
            archive=digest(ART / "datasets/libri/train-clean-100.tar.gz"),
        ),
        medical=dict(
            repo="Hani89/medical_asr_recording_dataset",
            revision="8b2b2bd4233140705f1f7bb48411b49cd188d89c",
            license="Apache-2.0 as declared by dataset publisher",
            original="Kaggle/Figure Eight Medical Speech Transcription and Intent",
        ),
        digits=json.loads((ART / "datasets/digits-revision.json").read_text()),
    )
    summary = {
        s: {
            d: dict(
                count=counts[s, d],
                hours=sum(
                    r["seconds"] for r in rows if r["split"] == s and r["domain"] == d
                )
                / 3600,
            )
            for d in ["general", "medical_symptoms", "digits"]
        }
        for s in ["train", "calibration", "development"]
    }
    save(
        OUT / "manifest.json",
        dict(
            schema=1,
            rows=rows,
            sources=sources,
            summary=summary,
            excluded_historical_speakers=sorted(excluded),
            rejected_invalid_medical=bad,
            skipped_after_feature_checks=len(candidates) - len(rows),
            limitations=[
                "Medical symptoms are not TCCC/drug terminology; speaker identity unavailable in medical diagnostic split.",
                "Digit strings are assembled from isolated recordings, not natural quantities, decimals or doses.",
                "Labels inherited from public corpora, not newly human-verified.",
                "No clinical, final holdout, energy or streaming-runtime qualification.",
            ],
        ),
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    prepare()
