"""Pure helpers of the Kea-B 0.1 release kit (release-kea-b-0.1 plan "Staging, upload, verification"), without torch:
the Hub repository layout, file listings and their comparison, the approval digest, the manifests' release form,
template rendering and the scrubbing scan.

Approval: the stage report carries bundle_manifest (the repository id and every staged file's path, SHA-256 and size,
sorted) and bundle_sha256, its SHA-256. The upload refuses unless it is given exactly that digest and the folder still
hashes to it, so a restage that changes any byte invalidates an earlier approval (as parakeet-ternary hf/cardkit.py).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import socket
import struct
from pathlib import Path

REPO_ID = "wildernesslabs/kea-b-0.1"
RELEASE_NAME = "kea-b-0.1"
LOADER = "load_kea_b.py"
TOKENIZER = ("tokenizer/tokenizer.model", "tokenizer/tokenizer.vocab", "tokenizer/vocab.txt")
BUNDLE_WEIGHTS, VIEW_WEIGHTS, MANIFEST = "bundle.safetensors", "export.safetensors", "manifest.json"
COMPAT = "compat"
BUNDLE_FILES = (BUNDLE_WEIGHTS, MANIFEST, *TOKENIZER)
COMPAT_FILES = tuple(f"{COMPAT}/{f}" for f in (VIEW_WEIGHTS, MANIFEST, *TOKENIZER))
DOCS = ("README.md", "EVALUATION.md", "TRAINING.md", "NOTICE")
STATIC = ("LICENSE", LOADER)
ASSETS = "assets"
FIXED_FILES = tuple(sorted((*BUNDLE_FILES, *COMPAT_FILES, *DOCS, *STATIC)))
# Entries of a Hub download that are not published files.
IGNORED_FILES = frozenset({".gitattributes"})
IGNORED_DIRS = frozenset({".cache"})
# Machine paths that no published file may contain (plan check 6).
MACHINE_PATTERNS = ("/mnt/", "/home/", "/Users/", "/nix/store")
# Set aliases of held-out evaluation material (eval-only; never named in a published file).
HELDOUT_SET_ALIASES = ("eval-ext", "eval-stem", "dev-medical", "dev-technical", "medical-b", "lectures-a",
                       "ted-test-text")
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
UNFILLED = re.compile(r"\{\{|\}\}|\bTODO\b|\bTBD\b|\bFIXME\b|\bXXX\b")
TEXT_SUFFIXES = (".md", ".json", ".py", ".txt", ".vocab", "")


class RenderError(ValueError):
    pass


# ----------------------------------------------------------------------------- files

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _published(folder: Path, path: Path) -> bool:
    rel = path.relative_to(folder)
    return path.is_file() and rel.parts[0] not in IGNORED_DIRS and rel.as_posix() not in IGNORED_FILES


def listing(folder: Path) -> dict[str, dict]:
    """{rel path: {"sha256", "bytes"}} of every published file under folder (Hub extras ignored), sorted."""
    folder = Path(folder)
    return {p.relative_to(folder).as_posix(): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(folder.rglob("*")) if _published(folder, p)}


def compare_listing(report_files: dict, folder: Path) -> list[str]:
    """Differences between a stage report's {rel: {"sha256", "bytes"}} and folder (empty if identical). The only
    entries ignored are the Hub's .gitattributes and the download cache .cache/."""
    folder = Path(folder)
    found = {p.relative_to(folder).as_posix(): p for p in sorted(folder.rglob("*")) if _published(folder, p)}
    problems = [f"missing {rel}" for rel in sorted(set(report_files) - set(found))]
    problems += [f"unexpected {rel}" for rel in sorted(set(found) - set(report_files))]
    for rel in sorted(set(found) & set(report_files)):
        want = report_files[rel]
        if found[rel].stat().st_size != want["bytes"] or sha256_file(found[rel]) != want["sha256"]:
            problems.append(f"{rel} differs from the staged file")
    return problems


def expected_files(asset_names) -> list[str]:
    """The staged file set: the fixed layout plus assets/<name> for each graphic."""
    return sorted((*FIXED_FILES, *(f"{ASSETS}/{n}" for n in asset_names)))


def safetensors_header_bytes(path: Path) -> int:
    """8-byte length prefix plus the JSON header: the file bytes that are not tensor data."""
    with Path(path).open("rb") as handle:
        (n,) = struct.unpack("<Q", handle.read(8))
    return 8 + n


def safetensors_header(path: Path) -> bytes:
    with Path(path).open("rb") as handle:
        (n,) = struct.unpack("<Q", handle.read(8))
        return handle.read(n)


def write_json(path: Path, data) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- approval

def bundle_manifest(repo_id: str, files: dict) -> str:
    """The canonical text AJ's approval covers: the repository and every file's path, SHA-256 and size, sorted."""
    lines = [f"repo {repo_id}"] + [f"file {rel} {files[rel]['sha256']} {files[rel]['bytes']}" for rel in sorted(files)]
    return "\n".join(lines) + "\n"


def bundle_sha256(repo_id: str, files: dict) -> str:
    return sha256_bytes(bundle_manifest(repo_id, files).encode())


def approval_problems(report: dict, folder: Path, approved: str | None) -> list[str]:
    """Why an upload of folder under report must be refused (empty: go). The approved digest must be a SHA-256,
    equal the report's bundle_sha256, and equal the digest of the folder as it is now."""
    problems = []
    if report.get("status") != "PASSED":
        problems.append("the stage report is not PASSED")
    if not re.fullmatch(r"[0-9a-f]{64}", approved or ""):
        problems.append("no approved bundle digest (64 hex characters) was given")
    elif approved != report.get("bundle_sha256"):
        problems.append("the approved digest is not the stage report's bundle_sha256")
    if not Path(folder).is_dir():
        problems.append(f"{folder} is not a directory")
        return problems
    current = listing(folder)
    if current != report.get("files"):
        problems.append("the staged folder differs from the stage report's file list")
    if bundle_sha256(report.get("repo_id", ""), current) != approved:
        problems.append("the staged folder does not hash to the approved digest")
    return problems


# ----------------------------------------------------------------------------- release-form manifests

def is_machine_path(value) -> bool:
    return isinstance(value, str) and (value.startswith(("/", "~")) or any(p in value for p in MACHINE_PATTERNS)
                                       or re.match(r"^[A-Za-z]:\\", value) is not None)


def release_form(manifest: dict) -> tuple[dict, list[str]]:
    """(the manifest with every machine path under "extra" removed, the removed keys as dotted paths). Everything
    else is unchanged (the weights' and tokenizer's SHA-256 entries stay valid: they do not cover manifest.json)."""
    out = copy.deepcopy(manifest)
    removed: list[str] = []

    def strip(node, prefix: str) -> None:
        if isinstance(node, dict):
            for key in list(node):
                if is_machine_path(node[key]):
                    removed.append(f"{prefix}.{key}")
                    del node[key]
                else:
                    strip(node[key], f"{prefix}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                if is_machine_path(item):
                    raise ValueError(f"{prefix}[{i}] is a machine path inside a list")
                strip(item, f"{prefix}[{i}]")

    strip(out.get("extra", {}), "extra")
    return out, removed


def manifest_text(manifest: dict) -> str:
    return json.dumps(manifest, indent=2) + "\n"


# ----------------------------------------------------------------------------- templates

def render(template: str, values: dict[str, str]) -> str:
    """Every {{name}} replaced by values[name]; RenderError for a name without a value or anything left unfilled."""
    missing = sorted({m.group(1) for m in PLACEHOLDER.finditer(template)} - set(values))
    if missing:
        raise RenderError(f"no value for placeholders {missing}")
    text = PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)
    left = unfilled(text)
    if left:
        raise RenderError(f"unfilled after rendering: {left[:5]}")
    return text


def unfilled(text: str) -> list[str]:
    return [m.group(0) for m in UNFILLED.finditer(text)]


# ----------------------------------------------------------------------------- scrubbing

_CAMEL = re.compile(r"\b[A-Za-z][A-Za-z0-9-]*(?:[a-z][A-Z]|[A-Za-z]-?[0-9])[A-Za-z0-9-]*\b")


def _leading_phrase(name: str) -> str:
    return re.split(r" \(| / |, | via | - ", name, maxsplit=1)[0].strip()


def scrub_terms(registry: dict, allowed_aliases, extra_terms=()) -> list[str]:
    """Terms no published file may contain: for every registry entry not in allowed_aliases (the release's training
    sources and tools), its alias, its name, the name's leading phrase, its distinctive tokens (internal capitals or
    digits, e.g. "Corpus57"), and its repository, Hub id, mirror and URL; plus the held-out set aliases and
    extra_terms. Matching is case-insensitive."""
    terms = set(HELDOUT_SET_ALIASES) | {t for t in extra_terms if t}
    for alias, entry in registry.items():
        if alias in allowed_aliases or not isinstance(entry, dict):
            continue
        terms.add(alias)
        name = str(entry.get("name") or "")
        if name:
            terms.add(name)
            lead = _leading_phrase(name)
            if len(lead) >= 5:
                terms.add(lead)
            terms.update(t for t in _CAMEL.findall(name) if len(t) >= 5)
        for key in ("hf_id", "repo", "mirror", "url"):
            value = entry.get(key)
            if isinstance(value, str) and len(value) >= 5:
                terms.add(value.split("@")[0])
    return sorted(terms, key=str.lower)


def hostnames() -> list[str]:
    names = {socket.gethostname(), socket.getfqdn()}
    return sorted(n for n in names if n and n not in ("localhost", "localhost.localdomain") and len(n) >= 3)


def file_text(path: Path) -> str:
    """The searchable text of a published file: a safetensors file's header, any other file whole (bytes read as
    Latin-1, so a binary tokenizer model is searched too)."""
    path = Path(path)
    data = safetensors_header(path) if path.suffix == ".safetensors" else path.read_bytes()
    return data.decode("latin-1")


def scrub_findings(folder: Path, terms, hosts=()) -> list[dict]:
    """[{file, term}] of every machine path pattern, hostname or term found in a published file under folder. Every
    file gets every check, images included (their metadata chunks are searched as Latin-1 text)."""
    folder = Path(folder)
    found = []
    lowered = [(t, t.lower()) for t in terms]
    for p in sorted(folder.rglob("*")):
        if not _published(folder, p):
            continue
        rel = p.relative_to(folder).as_posix()
        text = file_text(p)
        low = text.lower()
        found += [{"file": rel, "term": pat} for pat in MACHINE_PATTERNS if pat in text]
        found += [{"file": rel, "term": "hostname"} for h in hosts
                  if re.search(r"(?<![A-Za-z0-9])" + re.escape(h.lower()) + r"(?![A-Za-z0-9])", low)]
        found += [{"file": rel, "term": t} for t, tl in lowered
                  if re.search(r"(?<![A-Za-z0-9])" + re.escape(tl) + r"(?![A-Za-z0-9])", low)]
    return found
