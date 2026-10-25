"""Pinned, verified downloads of the dev-set sources (network, no GPU).

    fetch(alias, entry, out_dir) -> record   (also written to out_dir/fetch.json)

- origin "hf": the repo's file list at the pinned revision (with LFS SHA-256) from the Hub API; files
  matching the entry's globs are downloaded with hf_hub_download at that revision into the Hugging Face
  cache ($RELABEL_ROOT/caches, set by relabel_boot.py) and their SHA-256 checked against the Hub's.
- origin "github": the tree at the pinned commit from the GitHub API; matching files are read from the raw
  host at that commit and checked against the tree's git blob hash. A Git LFS pointer is resolved through
  the media host at the same commit and the file is checked against the pointer's SHA-256 and size.
  Files are stored under out_dir/<repo path>.

- origin "ia" (a web archive's public items): the entry pins every file explicitly ("files": a list of
  {item, name, size, md5, sha1}, taken from the item's metadata when the slice was chosen). Each file is read
  from the archive's ordinary public download URL, one at a time with a descriptive User-Agent and a pause
  between files (the site's normal access terms; no authentication), and kept only if its size, MD5 and
  SHA-1 equal the pinned values. Files are stored under out_dir/<item>/<name>.

The record lists every file (repo path, local path, bytes, SHA-256) in sorted repo-path order, with the
revision; a rerun skips files already present with the recorded hash.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

USER_AGENT = "relabel-devsets-fetch/1 (held-out speech-recognition evaluation; sequential downloads)"
GITHUB_API = "https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"
GITHUB_RAW = "https://raw.githubusercontent.com/{repo}/{revision}/{path}"
GITHUB_MEDIA = "https://media.githubusercontent.com/media/{repo}/{revision}/{path}"
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
IA_DOWNLOAD = "https://archive.org/download/{item}/{name}"
IA_PAUSE_SECONDS = 2.0


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def parse_lfs_pointer(data: bytes) -> tuple[str, int] | None:
    """(sha256, size) of a Git LFS pointer file, or None if data is not one."""
    if not data.startswith(LFS_PREFIX) or len(data) > 1024:
        return None
    fields = dict(line.split(" ", 1) for line in data.decode().splitlines() if " " in line)
    oid = fields.get("oid", "")
    if not oid.startswith("sha256:"):
        return None
    return oid.split(":", 1)[1], int(fields["size"])


def _get(url: str, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError


def _download(url: str, dest: Path, sha256: str, size: int, attempts: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=300) as response, open(part, "wb") as handle:
                while chunk := response.read(1 << 20):
                    digest.update(chunk)
                    handle.write(chunk)
            if digest.hexdigest() != sha256 or part.stat().st_size != size:
                raise OSError(f"{url}: content does not match the LFS pointer")
            os.replace(part, dest)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def download_checked(url: str, dest: Path, size: int, md5: str, sha1: str, attempts: int = 4) -> str:
    """Download url to dest (via dest.part) and keep it only if size, MD5 and SHA-1 match; returns its SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            digests = {k: hashlib.new(k) for k in ("md5", "sha1", "sha256")}
            with urllib.request.urlopen(request, timeout=300) as response, open(part, "wb") as handle:
                while chunk := response.read(1 << 20):
                    for d in digests.values():
                        d.update(chunk)
                    handle.write(chunk)
            got = {k: d.hexdigest() for k, d in digests.items()}
            if part.stat().st_size != int(size) or got["md5"] != md5 or got["sha1"] != sha1:
                raise OSError(f"{url}: size or hash differs from the pinned values")
            os.replace(part, dest)
            return got["sha256"]
        except OSError:
            part.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError


def _fetch_ia(entry: dict, out_dir: Path, previous: dict) -> list[dict]:
    files = []
    for spec in sorted(entry["files"], key=lambda f: (f["item"], f["name"])):
        path = f"{spec['item']}/{spec['name']}"
        dest = out_dir / spec["item"] / spec["name"]
        old = previous.get(path)
        if old and dest.exists() and dest.stat().st_size == old["bytes"] and sha256_file(dest) == old["sha256"]:
            files.append(old)
            continue
        sha = download_checked(IA_DOWNLOAD.format(item=spec["item"], name=spec["name"]), dest, spec["size"],
                               spec["md5"], spec["sha1"])
        files.append({"path": path, "local": str(dest), "bytes": int(spec["size"]), "sha256": sha, "md5": spec["md5"],
                      "sha1": spec["sha1"], "lfs": False})
        time.sleep(IA_PAUSE_SECONDS)
    return files


def _fetch_github(entry: dict, out_dir: Path, previous: dict) -> list[dict]:
    repo, revision = entry["repo"], entry["revision"]
    tree = json.loads(_get(GITHUB_API.format(repo=repo, revision=revision)))
    if tree.get("truncated"):
        raise RuntimeError(f"{repo}@{revision}: the tree listing is truncated")
    files = []
    for node in sorted((n for n in tree["tree"] if n["type"] == "blob"), key=lambda n: n["path"]):
        if not matches(node["path"], entry["files"]):
            continue
        dest = out_dir / node["path"]
        old = previous.get(node["path"])
        if old and dest.exists() and dest.stat().st_size == old["bytes"] and sha256_file(dest) == old["sha256"]:
            files.append(old)
            continue
        data = _get(GITHUB_RAW.format(repo=repo, revision=revision, path=node["path"]))
        if git_blob_sha1(data) != node["sha"]:
            raise RuntimeError(f"{node['path']}: raw content does not match the tree's blob hash")
        pointer = parse_lfs_pointer(data)
        if pointer:
            sha, size = pointer
            _download(GITHUB_MEDIA.format(repo=repo, revision=revision, path=node["path"]), dest, sha, size)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            sha, size = hashlib.sha256(data).hexdigest(), len(data)
        files.append({"path": node["path"], "local": str(dest), "bytes": size, "sha256": sha, "lfs": bool(pointer)})
    return files


def _fetch_hf(entry: dict, previous: dict) -> list[dict]:
    from huggingface_hub import HfApi, hf_hub_download

    repo, revision = entry["repo"], entry["revision"]
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"{repo}: the Hub resolved revision {info.sha}, not the pinned {revision}")
    files = []
    for sibling in sorted(info.siblings, key=lambda s: s.rfilename):
        if not matches(sibling.rfilename, entry["files"]):
            continue
        lfs = sibling.lfs
        expected = (lfs.sha256 if hasattr(lfs, "sha256") else lfs["sha256"]) if lfs else None
        local = Path(hf_hub_download(repo, sibling.rfilename, repo_type="dataset", revision=revision))
        old = previous.get(sibling.rfilename)
        digest = old["sha256"] if old and old["local"] == str(local) and old["bytes"] == local.stat().st_size \
            and old["sha256"] == expected else sha256_file(local)
        if expected and digest != expected:
            raise RuntimeError(f"{sibling.rfilename}: SHA-256 {digest} != the Hub's {expected}")
        files.append({"path": sibling.rfilename, "local": str(local), "bytes": local.stat().st_size,
                      "sha256": digest, "lfs": bool(lfs)})
    return files


def fetch(alias: str, entry: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "fetch.json"
    previous = {}
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if old.get("revision") == entry["revision"] and old.get("repo") == entry["repo"]:
            previous = {f["path"]: f for f in old.get("files", [])}
    if entry["origin"] == "github":
        files = _fetch_github(entry, out_dir, previous)
    elif entry["origin"] == "ia":
        files = _fetch_ia(entry, out_dir, previous)
    else:
        files = _fetch_hf(entry, previous)
    if not files:
        raise RuntimeError(f"{alias}: no repository file matches {entry['files']}")
    record = {"alias": alias, "origin": entry["origin"], "repo": entry["repo"], "revision": entry["revision"],
              "patterns": entry["files"], "files": files, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = record_path.with_name(f"fetch.json.tmp{os.getpid()}")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, record_path)
    return record
