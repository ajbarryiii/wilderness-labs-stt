"""Download pinned inputs only; all artifacts stay on the verified data disk."""

import concurrent.futures, hashlib, json, os, pathlib, tarfile, urllib.request

ROOT = pathlib.Path("/mnt/hd/wilderness-labs-stt/stt-distillation")
assert os.path.ismount("/mnt/hd"), "/mnt/hd is not mounted"
from huggingface_hub import hf_hub_download, snapshot_download


def fetch_model():
    p = ROOT / "teachers/omi"
    hf_hub_download(
        "omi-health/omi-med-stt-v1",
        "omimedstt-v1.nemo",
        revision="75d3bdf176dd051b5a87df7752470230ec6dc1ed",
        local_dir=p,
    )
    hf_hub_download(
        "omi-health/omi-med-stt-v1",
        "README.md",
        revision="75d3bdf176dd051b5a87df7752470230ec6dc1ed",
        local_dir=p,
    )
    print("omi downloaded", flush=True)


def fetch_whisper():
    snapshot_download(
        "openai/whisper-medium.en",
        revision="2e98eb6279edf5095af0c8dedb36bdec0acd172b",
        local_dir=ROOT / "teachers/whisper",
        allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md"],
    )
    print("whisper downloaded", flush=True)


def archive(url, path, md5=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(".download")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    if md5:
        assert hashlib.file_digest(open(path, "rb"), "md5").hexdigest() == md5
    return path


def libri():
    folder = ROOT / "datasets/libri"
    f = archive(
        "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        folder / "train-clean-100.tar.gz",
        "2a93770f6d5c6c964bc36631d331a522",
    )
    if not (folder / "extracted.json").exists():
        with tarfile.open(f) as t:
            t.extractall(folder, filter="data")
        (folder / "extracted.json").write_text(
            json.dumps(
                {
                    "archive_sha256": hashlib.file_digest(
                        open(f, "rb"), "sha256"
                    ).hexdigest()
                }
            )
        )
    print("libri ready", flush=True)


def medical():
    repo = "Hani89/medical_asr_recording_dataset"
    rev = "8b2b2bd4233140705f1f7bb48411b49cd188d89c"
    for file in [
        "README.md",
        "data/train-00000-of-00007-b0c1bd87e30f8607.parquet",
        "data/train-00001-of-00007-bee1da05a7ded48d.parquet",
    ]:
        hf_hub_download(
            repo,
            file,
            repo_type="dataset",
            revision=rev,
            local_dir=ROOT / "datasets/medical",
        )
    print("medical downloaded", flush=True)


def digits():
    info = json.load(
        urllib.request.urlopen(
            "https://api.github.com/repos/Jakobovski/free-spoken-digit-dataset/commits/master"
        )
    )
    lock = ROOT / "datasets/digits-revision.json"
    rev = json.loads(lock.read_text())["revision"] if lock.exists() else info["sha"]
    lock.write_text(
        json.dumps(
            {
                "revision": rev,
                "license": "CC-BY-SA-4.0",
                "source": "https://github.com/Jakobovski/free-spoken-digit-dataset",
            }
        )
    )
    f = archive(
        "https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/tar.gz/"
        + rev,
        ROOT / "datasets/digits.tar.gz",
    )
    if not (ROOT / "datasets" / ("free-spoken-digit-dataset-" + rev)).exists():
        with tarfile.open(f) as t:
            t.extractall(ROOT / "datasets", filter="data")
    print("digits ready", flush=True)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for f in concurrent.futures.as_completed(
            [
                ex.submit(fn)
                for fn in [fetch_model, fetch_whisper, libri, medical, digits]
            ]
        ):
            f.result()
    print("all downloads complete", flush=True)
