"""Artifact storage is mandatory on the mounted data disk."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/mnt/hd/wilderness-labs-stt/inference-efficiency')


def storage():
    if not os.path.ismount('/mnt/hd'):
        raise RuntimeError('/mnt/hd is not mounted; refusing artifact writes')
    ROOT.mkdir(parents=True, exist_ok=True)
    return ROOT


def artifact(path):
    storage()
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f'Artifact must be under {ROOT}: {path}')
    return path


def save(path, value):
    path = artifact(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
