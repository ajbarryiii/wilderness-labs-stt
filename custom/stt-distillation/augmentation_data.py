"""Download verified DEMAND audio and freeze leakage-free augmentation fixtures."""

import argparse
import concurrent.futures
import hashlib
import json
import zipfile
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from common import ART, digest, encode, save, storage
from prepare import feature
from recovery_core import SOURCE_RUN, artifact, read

ROOT = ART / "augmentation-pilot"
RECOVERY_RUN = ART / "recovery-and-decoding/runs/recovery-8h-20260911T031803Z"
ENVIRONMENTS = {
    "DWASHING": "train", "NFIELD": "train", "STRAFFIC": "train", "TCAR": "train",
    "NPARK": "validation", "TBUS": "validation",
}


def download():
    storage()
    root = artifact(ROOT / "data/demand")
    root.mkdir(parents=True, exist_ok=True)
    metadata = ROOT / "demand-record.json"
    if not metadata.exists():
        response = requests.get("https://zenodo.org/api/records/1227121", timeout=60)
        response.raise_for_status()
        save(metadata, response.json())
    files = {x["key"]: x for x in read(metadata)["files"]}

    def fetch(env):
        record = files[env + "_16k.zip"]
        path = root / record["key"]
        if not path.exists():
            tmp = path.with_suffix(".part")
            for attempt in range(5):
                try:
                    with requests.get(record["links"]["self"], stream=True, timeout=(60, 90)) as response:
                        response.raise_for_status()
                        with tmp.open("wb") as out:
                            for chunk in response.iter_content(1024 * 1024):
                                out.write(chunk)
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(3 * (attempt + 1))
            tmp.replace(path)
        with path.open("rb") as handle:
            actual = "md5:" + hashlib.file_digest(handle, "md5").hexdigest()
        assert actual == record["checksum"], (env, actual, record["checksum"])
        # One channel per environment; no alternate channels of validation sessions in training.
        with zipfile.ZipFile(path) as archive:
            names = sorted(n for n in archive.namelist() if n.lower().endswith(".wav"))
            assert len(names) == 16
            audio = root / (env + ".wav")
            audio.write_bytes(archive.read(names[0]))
        info = sf.info(audio)
        assert info.samplerate == 16000 and info.channels == 1 and info.duration >= 290
        result = dict(id=env, split=ENVIRONMENTS[env], audio=str(audio),
                      sha256=digest(audio), seconds=info.duration,
                      source_url=record["links"]["self"], archive_md5=actual,
                      archive_member=names[0])
        print(json.dumps(result), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(fetch, ENVIRONMENTS))
    save(root / "manifest.json", dict(
        source="https://zenodo.org/records/1227121", authors="Joachim Thiemann, Nobutaka Ito, Emmanuel Vincent",
        license="CC BY-SA 3.0, as stated in the dataset description", rows=rows))


def prepare_validation():
    from augmentation_features import Mixer
    storage()
    rows = read(SOURCE_RUN / "manifest.json")["rows"]
    dev = [r for r in rows if r["split"] == "development"]
    mixer = Mixer(rows, read(ROOT / "data/demand/manifest.json")["rows"])
    definitions = {
        "noise15": ("noise", 15), "speech20": ("speech", 20),
        "noise5": ("noise", 5), "speech0": ("speech", 0),
    }
    manifest = {}
    for condition, (kind, db) in definitions.items():
        records = []
        for index, row in enumerate(dev):
            rng = np.random.default_rng(2026091100 + index)
            audio = mixer.audio(row, trim_digit=False)
            mixed, detail = mixer.mix(audio, row, rng, kind, db, validation=True)
            x = feature(mixed)
            y = encode(row["text"])
            assert (x.shape[-1] + 1) // 2 >= len(y) + sum(a == b for a, b in zip(y, y[1:]))
            path = artifact(ROOT / "data/validation" / condition / (row["id"] + ".npy"))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, x, allow_pickle=False)
            records.append(dict(row, features=str(path), feature_sha256=digest(path), augmentation=detail))
            if index < 2 or (row["domain"] == "digits" and sum(r["domain"] == "digits" for r in records) <= 2):
                sf.write(path.with_suffix(".wav"), mixed, 16000, subtype="FLOAT")
        manifest[condition] = records
    # Pure background tests expose false emissions from the amplitude-based onset mask.
    backgrounds = []
    for item in mixer.noise:
        if item["split"] != "validation":
            continue
        audio = mixer.background(item)[:16000 * 5]
        audio = audio * (0.05 / max(float(np.sqrt(np.mean(audio ** 2))), 1e-8))
        path = artifact(ROOT / "data/validation/noise_only" / (item["id"] + ".npy"))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, feature(audio), allow_pickle=False)
        backgrounds.append(dict(id="noise-only-" + item["id"], features=str(path), feature_sha256=digest(path)))
    save(ROOT / "data/validation.json", dict(conditions=manifest, noise_only=backgrounds,
        selection_conditions=["noise15", "speech20"], stress_only=["noise5", "speech0"],
        background_speech_split="calibration", target_split="development",
        scope="Reused development speech with unseen background sources; synthetic robustness, not field qualification"))
    print(json.dumps(dict(validation_conditions={k: len(v) for k, v in manifest.items()}, noise_only=len(backgrounds))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["download", "validation"])
    args = parser.parse_args()
    download() if args.action == "download" else prepare_validation()
