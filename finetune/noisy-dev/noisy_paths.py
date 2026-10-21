"""Fixed locations, pinned tools and constants of the noisy development-set copy. See plans/build.md.

Everything the build writes lives under ARTIFACTS (/mnt/hd/wilderness-labs-stt/noisy-dev):

    raw/<source>/          downloads (untrusted; written only by download.sh)
    publisher/             publisher checksum records as fetched (Zenodo API JSON, OpenSLR md5)
    verify/<source>.json   download verification records (noisy_sources.verify_source)
    extracted/<source>/    archive members, extracted with path checks (noisy_extract)
    pool/                  pool index (pool.json), prepared 16 kHz clips, rotor plug-in pools
    build/                 the production build (audio/, manifests/, build.lock.json, ...)
    samples/<name>/        sample builds (--sample), never the production path
    tmp/, caches/          interpreter temporary files and caches (./python)

The parakeet-ternary artifacts (dev manifests and audio, the pretrained model) are read only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import paths  # finetune/parakeet-ternary/paths.py

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

ARTIFACTS = Path("/mnt/hd/wilderness-labs-stt/noisy-dev")
RAW = ARTIFACTS / "raw"
PUBLISHER = ARTIFACTS / "publisher"
VERIFY = ARTIFACTS / "verify"
EXTRACTED = ARTIFACTS / "extracted"
POOL = ARTIFACTS / "pool"
BUILD = ARTIFACTS / "build"
SAMPLES = ARTIFACTS / "samples"
SCORES = ARTIFACTS / "scores"
TMP = ARTIFACTS / "tmp"

BUILD_SEED = 20261016
SAMPLE_RATE = paths.SAMPLE_RATE  # 16 kHz
DEV_SETS = list(paths.DEV_SETS)  # the five full development sets, plans/build.md "Utterances"
DEV_MANIFESTS = paths.MANIFESTS  # <set>.jsonl, parakeet-ternary data.py

# Codec2 1.2.0 from nixpkgs (LGPL-2.1), pinned by store path. Built with
# `nix build nixpkgs#codec2 --out-link /mnt/hd/wilderness-labs-stt/noisy-dev/gcroots/codec2`
# (2026-10-16; the out-link is a GC root, so the path is not collected).
CODEC2_STORE = Path("/nix/store/55pifb4bvhvqqhhg21faqm2czl80cz0x-codec2-1.2.0")
CODEC2_ENC = CODEC2_STORE / "bin" / "c2enc"
CODEC2_DEC = CODEC2_STORE / "bin" / "c2dec"
CODEC2_VERSION = "1.2.0"
CODEC2_GCROOT = ARTIFACTS / "gcroots" / "codec2"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def require_mount() -> None:
    paths.require_mount()


def named_dir(root: Path, name: str) -> Path:
    """root/name (not created); the name may not contain path separators."""
    if not _NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"invalid name {name!r}")
    return Path(root) / name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: object) -> str:
    """The byte-stable JSON used for every hashed record (sorted keys, no whitespace variance)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, obj: object) -> None:
    """Atomic: temp file, fsync, rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
