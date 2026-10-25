"""Pinned, verified downloads of stage 2c's sources (network, no GPU). Downloads are untrusted data.

    ./python -m synthetic fetch ALIAS [ALIAS ...]

Per alias (the private registry entry, read at run time):
- "files" [{name, url, size, md5}]: each file is downloaded (resumable, one at a time) to
  $RELABEL_ROOT/synthetic/sources/<alias>/download/<name> and kept only if its size and MD5 equal the pinned
  values; its SHA-256 is recorded.
- "hf_id" + "revision" (+ optional "files" {repo path: sha256}): the files are fetched from the Hub at the pinned
  revision (hf_hub_download into the Hugging Face cache, or for the text generator a full snapshot) and checked
  against the pinned SHA-256 where one is given.
- a "local_copy" {root, path, sha256}: a file already on the data mount (downloaded and verified by another
  project) is used read-only after its SHA-256 is checked;
- a TTS engine ("engine_install"): kind "zip" {name, url, size, sha256} is downloaded (size and SHA-256 checked) and
  extracted (noisy_extract, safe) into $RELABEL_ROOT/synthetic/engines/<alias>/model/; kind "hf" {hf_id, revision,
  files {repo path: sha256}} copies the verified Hub files into engines/<alias>/download/. Either way every file the
  engine loads must then match the registry's "engine_files" SHA-256 (tts.verify_files).
The record (files, sizes, hashes, revision) is written to <dir>/fetch.json; a rerun skips verified files.
Archives are extracted by synthetic/noise.py with finetune/noisy-dev's safe extractor (noisy_extract).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from synthetic import layout

USER_AGENT = "relabel-synthetic-fetch/1 (speech-recognition research; sequential downloads)"
CHUNK = 1 << 22


def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, size: int, md5: str | None, attempts: int = 6, sha256: str | None = None) -> str:
    """Download url to dest (resuming a partial .part file), check size and MD5 (or SHA-256); returns the SHA-256."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    algo, want = ("md5", md5) if md5 else ("sha256", sha256)
    if not want:
        raise ValueError("a pinned MD5 or SHA-256 is required")
    if dest.exists() and dest.stat().st_size == size and _hash(dest, algo) == want:
        return _hash(dest, "sha256")
    part = dest.with_name(dest.name + ".part")
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have < size:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **({"Range": f"bytes={have}-"} if have else {})})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp, open(part, "ab" if have else "wb") as out:
                    if have and resp.status != 206:
                        out.truncate(0)
                    for block in iter(lambda: resp.read(CHUNK), b""):
                        out.write(block)
            except OSError as error:
                print(f"fetch: {dest.name}: attempt {attempt + 1} failed: {error}", flush=True)
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
        if part.stat().st_size == size:
            if _hash(part, algo) != want:
                part.unlink()
                raise RuntimeError(f"{dest.name}: {algo} mismatch (expected {want}); partial file removed")
            os.replace(part, dest)
            return _hash(dest, "sha256")
    raise RuntimeError(f"{dest.name}: download failed after {attempts} attempts")


def fetch_files(alias: str, entry: dict) -> list[dict]:
    out = []
    target = layout.source_dir(alias) / "download"
    for f in entry["files"]:
        sha = download(f["url"], target / f["name"], int(f["size"]), f["md5"])
        out.append({"name": f["name"], "path": str(target / f["name"]), "size": int(f["size"]), "md5": f["md5"],
                    "sha256": sha})
        print(f"fetch: {alias} {f['name']} ok ({f['size']} bytes)", flush=True)
    return out


def fetch_local(alias: str, entry: dict) -> list[dict]:
    copy = entry["local_copy"]
    path = layout.root(copy["root"]) / copy["path"]
    sha = _hash(path, "sha256")
    if sha != copy["sha256"]:
        raise RuntimeError(f"{alias}: local copy SHA-256 {sha} is not the pinned {copy['sha256']}")
    return [{"name": path.name, "path": str(path), "size": path.stat().st_size, "sha256": sha, "local_copy": True}]


def fetch_hf(alias: str, entry: dict) -> list[dict]:
    from huggingface_hub import hf_hub_download, snapshot_download
    if not entry.get("files"):
        local = snapshot_download(entry["hf_id"], revision=entry["revision"])
        files = sorted(p for p in Path(local).rglob("*") if p.is_file())
        return [{"name": str(p.relative_to(local)), "path": str(p), "size": p.stat().st_size} for p in files]
    out = []
    for repo_path, sha in sorted(entry["files"].items()):
        local = Path(hf_hub_download(entry["hf_id"], repo_path, revision=entry["revision"]))
        got = _hash(local, "sha256")
        if got != sha:
            raise RuntimeError(f"{alias}: {repo_path} SHA-256 {got} is not the pinned {sha}")
        out.append({"name": repo_path, "path": str(local), "size": local.stat().st_size, "sha256": got})
    return out


def install_engine(alias: str, entry: dict) -> list[dict]:
    import shutil
    import sys
    from synthetic import tts
    inst = entry["engine_install"]
    d = layout.engine_dir(alias)
    if inst["kind"] == "zip":
        z = d / "download" / inst["name"]
        sha = download(inst["url"], z, int(inst["size"]), None, sha256=inst["sha256"])
        model = d / "model"
        if not model.exists():
            here = str(Path(__file__).resolve().parents[2] / "noisy-dev")
            if here not in sys.path:
                sys.path.append(here)
            import noisy_extract as nx
            nx.extract_zip([z], model, lambda n: True, nx.Budget(max_members=100, max_member_bytes=1 << 30,
                                                                 max_total_bytes=2 << 30))
        files = [{"name": inst["name"], "path": str(z), "size": z.stat().st_size, "sha256": sha}]
        loaded = [p for p in sorted(model.rglob("*")) if p.is_file()]
    elif inst["kind"] == "hf":
        from huggingface_hub import hf_hub_download
        files, loaded = [], []
        for repo_path, want in sorted(inst["files"].items()):
            src = Path(hf_hub_download(inst["hf_id"], repo_path, revision=inst["revision"]))
            if _hash(src, "sha256") != want:
                raise RuntimeError(f"{alias}: {repo_path} does not match its pinned SHA-256")
            dst = d / "download" / Path(repo_path).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or _hash(dst, "sha256") != want:
                shutil.copyfile(src, dst)
            files.append({"name": repo_path, "path": str(dst), "size": dst.stat().st_size, "sha256": want})
            loaded.append(dst)
    else:
        raise ValueError(f"{alias}: unknown engine_install kind {inst['kind']!r}")
    pinned = entry["engine_files"]
    tts.verify_files(alias, [p for p in loaded if p.name in pinned])
    missing = sorted(set(pinned) - {p.name for p in loaded})
    if missing:
        raise RuntimeError(f"{alias}: engine files missing after install: {missing}")
    return files


def fetch(alias: str) -> dict:
    layout.require_mount()
    entry = layout.source(alias)
    if "engine_install" in entry:
        files = install_engine(alias, entry)
    elif "local_copy" in entry:
        files = fetch_local(alias, entry)
    elif "hf_id" in entry:
        files = fetch_hf(alias, entry)
    else:
        files = fetch_files(alias, entry)
    record = {"alias": alias, "revision": entry.get("revision"), "files": files,
              "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = layout.source_dir(alias) / "fetch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(tmp, path)
    return record


def record(alias: str) -> dict:
    """The fetch record of a fetched alias (refuses an alias that was not fetched)."""
    path = layout.source_dir(alias) / "fetch.json"
    if not path.exists():
        raise RuntimeError(f"{alias} is not fetched ({path} missing; run ./python -m synthetic fetch {alias})")
    return json.loads(path.read_text())
