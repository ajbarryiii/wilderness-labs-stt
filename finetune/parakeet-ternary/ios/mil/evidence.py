"""Hash-verified full results outside Git; compact, cited summaries inside Git.

IOS_RESULTS_ROOT (environment or mil/local.json) is the archive root, containing
<commit>/<results-relative path> or live/<sha256>/<results-relative path>.
Default: artifacts.root()/results-archive. No repository fallback for results.
All APIs take logical paths under ios/results; other paths retain pathlib behavior.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

IOS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IOS))
RESULTS = IOS / "results"
INDEX = RESULTS / "INDEX.json"
MAX_SUMMARY_BYTES = 96_000


class EvidenceError(RuntimeError):
    pass


def _key(path):
    p = Path(os.path.abspath(Path(path).expanduser()))
    if not p.is_relative_to(RESULTS):
        return None
    key = str(p.relative_to(IOS))
    if "quarantine" in PurePosixPath(key).parts:
        raise EvidenceError("quarantined records are never evidence")
    if p.is_symlink():
        raise EvidenceError(f"symbolic result path: {p}")
    return key


def root() -> Path:
    import artifacts

    setting = os.environ.get("IOS_RESULTS_ROOT")
    config = IOS / "mil" / "local.json"
    if not setting and config.exists():
        setting = json.loads(config.read_text()).get("IOS_RESULTS_ROOT")
    # check enforces the mount and the machine's artifact area even for overrides.
    path = artifacts.check(Path(setting).expanduser() if setting else artifacts.root() / "results-archive")
    if path.is_relative_to(IOS.resolve()):
        raise EvidenceError("results archive cannot be inside the project")
    return path


def index() -> dict:
    try:
        doc = json.loads(INDEX.read_text())
        if doc["version"] != 1 or not isinstance(doc["records"], dict):
            raise ValueError("unsupported evidence index")
        return doc
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise EvidenceError(f"missing or malformed {INDEX}: {exc}") from exc


def _archive(entry) -> Path:
    try:
        name = entry["archive"]
        parts = PurePosixPath(name)
        if not isinstance(name, str) or parts.is_absolute() or ".." in parts.parts or not parts.parts:
            raise ValueError("unsafe archive path")
        base = root().resolve()
        path = base / name
        if not path.resolve().is_relative_to(base) or path.is_symlink():
            raise ValueError("archive path escapes root or is a symlink")
        return path
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"malformed archive entry: {exc}") from exc


def read_bytes(path) -> bytes:
    key = _key(path)
    if key is None:
        return Path(path).read_bytes()
    entry = index()["records"].get(key)
    if entry is None:
        raise EvidenceError(f"no indexed full record: {key}")
    return _read_entry(key, entry)


def _read_entry(key, entry) -> bytes:
    """Verify the immutable entry supplied by the caller, even if INDEX changes."""
    try:
        data = _archive(entry).read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise EvidenceError(f"SHA-256/size mismatch: {key}")
        return data
    except OSError as exc:
        raise EvidenceError(f"missing full record {key}: {exc}") from exc


def read_text(path, encoding="utf-8", errors=None) -> str:
    return read_bytes(path).decode(encoding or "utf-8", errors or "strict")


def sha(path) -> str:
    return hashlib.sha256(read_bytes(path)).hexdigest()


def exists(path) -> bool:
    key = _key(path)
    return Path(path).exists() if key is None else key in index()["records"]


def glob(path, pattern):
    key = _key(path)
    if key is None:
        return Path(path).glob(pattern)
    prefix = key + "/"
    return iter(sorted(IOS / k for k in index()["records"]
                       if k.startswith(prefix) and "/" not in k[len(prefix):]
                       and fnmatch.fnmatchcase(k[len(prefix):], pattern)))


def _prune(value, limit=16, depth=0):
    if isinstance(value, dict):
        if depth > 6 or len(value) > 48:
            return {"omitted_entries": len(value), "detail": "see full record"}
        return {k: _prune(v, limit, depth + 1) for k, v in value.items()
                if k != "_evidence"}
    if isinstance(value, list):
        if len(value) > limit:
            return {"omitted_entries": len(value), "detail": "see full record"}
        return [_prune(v, limit, depth + 1) for v in value]
    return value


def summary(data: bytes, entry: dict) -> bytes:
    """Deterministic display-only projection. Readers always consume the full record."""
    citation = {k: entry[k] for k in ("sha256", "bytes", "archive")}
    try:
        doc = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        text = data.decode("utf-8")
        footer = f"\nFull record: SHA-256 {citation['sha256']}; {citation['bytes']} bytes; {citation['archive']}\n"
        if len(data) > MAX_SUMMARY_BYTES - len(footer.encode()):
            text = text[:80_000] + "\n[Remaining text in full record]\n"
        return (text + footer).encode()
    # Retain curated tables/sweep summaries in full when they fit. Large per-clip records
    # expose their aggregate fields; array/dictionary omissions are explicitly labelled.
    curated = (any("/" + directory + "/" in entry["archive"] for directory in ("wp5", "wp7", "smoke"))
               or entry["archive"].endswith("/summary.json"))
    if len(data) < 12_000 or (curated and len(data) < MAX_SUMMARY_BYTES - 1000):
        view = doc if isinstance(doc, dict) else {"summary": doc}
    else:
        heavy = {"encoder_cases", "heads_per_clip", "free_decoding_tokens", "encoder_rows",
                 "cases", "calls", "results", "ops", "tensors", "files", "per_clip",
                 "program", "compute_plan"}
        view = ({k: ({"omitted_entries": len(v), "detail": "see full record"}
                      if k in heavy and isinstance(v, (list, dict)) else _prune(v))
                 for k, v in doc.items()} if isinstance(doc, dict) else {"summary": _prune(doc)})
    if isinstance(doc, dict) and doc.get("kind") == "pipeline":
        view = {k: doc[k] for k in ("design_revision", "model", "arm", "variant", "backend",
                "front_end", "decode", "timing_allowed", "selection_eligible", "reasons", "built")}
        view["checks"] = {k: {"pass": v.get("pass")} for k, v in doc["checks"].items()}
        view["wer"] = doc["checks"]["wer"]
        view["encoder"] = doc["checks"]["encoder"]
        view["free_decoding"] = doc["checks"]["free_decoding"]
        view["decisions"] = doc["checks"]["decisions"]
    if isinstance(doc, dict) and "decodes" in doc and "header" in doc:
        # The aggregate verdict and errors are the useful human evidence, rather than
        # a repeated component manifest or per-clip encoder diagnostics.
        view["header"] = {k: doc["header"][k] for k in ("model", "arm", "variant", "compute_units") if k in doc["header"]}
        view["decodes"] = {dec: {k: _prune(v[k]) for k in ("coverage", "finite", "decisions",
                                "head_errors", "free_decoding", "wer") if k in v}
                           for dec, v in doc["decodes"].items()}
    view = {**view, "_evidence": citation}
    def finite_display(v):
        import math
        if isinstance(v, float) and not math.isfinite(v):
            return str(v)  # original nonfinite numeric bytes remain in the full record
        if isinstance(v, dict):
            return {k: finite_display(x) for k, x in v.items()}
        if isinstance(v, list):
            return [finite_display(x) for x in v]
        return v
    out = (json.dumps(finite_display(view), indent=1, allow_nan=False) + "\n").encode()
    if len(out) > MAX_SUMMARY_BYTES:
        raise EvidenceError("summary exceeds 96 KB; add an explicit aggregate projection")
    return out


def _atomic(path: Path, data: bytes):
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".evidence-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def index_bytes(doc: dict) -> bytes:
    # One compact entry per line keeps the necessary global mapping reasonably small.
    lines = ['{', ' "version": 1,', ' "records": {']
    records = sorted(doc["records"].items())
    for i, (key, entry) in enumerate(records):
        lines.append("  " + json.dumps(key) + ": " + json.dumps(entry, sort_keys=True)
                     + ("," if i + 1 < len(records) else ""))
    return ("\n".join(lines + [" }", "}"]) + "\n").encode()


@contextmanager
def _locked():
    import fcntl

    base = root()
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".index.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def write_text(path, text, encoding="utf-8", errors=None) -> int:
    """Publish immutable full bytes, compact citation, then the current index under a lock."""
    key = _key(path)
    if key is None:
        return Path(path).write_text(text, encoding=encoding, errors=errors)
    data = text.encode(encoding or "utf-8", errors or "strict")
    digest = hashlib.sha256(data).hexdigest()
    entry = {"sha256": digest, "bytes": len(data),
             "archive": "live/" + digest + "/" + key.removeprefix("results/")}
    display = summary(data, entry)
    with _locked():
        doc = index()
        dest = _archive(entry)
        if dest.exists() and dest.read_bytes() != data:
            raise EvidenceError("immutable archive collision")
        _atomic(dest, data)
        doc["records"][key] = entry
        _atomic(Path(path), display)
        _atomic(INDEX, index_bytes(doc))
    return len(text)


def unlink(path, missing_ok=False):
    """Withdraw the active index entry; old immutable records remain for audit."""
    key = _key(path)
    if key is None:
        return Path(path).unlink(missing_ok=missing_ok)
    with _locked():
        doc = index()
        if key not in doc["records"] and not missing_ok:
            raise FileNotFoundError(path)
        doc["records"].pop(key, None)
        # Remove from the index first: interruption cannot re-enable old evidence.
        _atomic(INDEX, index_bytes(doc))
        Path(path).unlink(missing_ok=True)


def verify() -> dict:
    records = index()["records"]
    for key, entry in records.items():
        read_bytes(IOS / key)
        display = (IOS / key).read_bytes()
        if summary(read_bytes(IOS / key), entry) != display:
            raise EvidenceError(f"summary/citation differs from full record: {key}")
    return {"verified_records": len(records), "full_bytes": sum(e["bytes"] for e in records.values()),
            "committed_bytes": sum((IOS / k).stat().st_size for k in records) + INDEX.stat().st_size}


def export_bundle(out):
    """Portable bundle of precisely the current INDEX records (no models or audio)."""
    import artifacts
    import io
    import tarfile

    dest = artifacts.check(out)
    if dest.exists():
        raise EvidenceError(f"bundle destination exists: {dest}")
    doc = index()
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(dest, "w:gz") as tar:
            def add(name, data):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            add("INDEX.json", index_bytes(doc))
            for key, entry in doc["records"].items():
                add(entry["archive"], _read_entry(key, entry))
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return {"bundle": str(dest), "records": len(doc["records"])}


def import_bundle(src):
    """Import only the current Git index's records, verify before immutable publication."""
    import tarfile

    doc = index()
    expected = {entry["archive"]: entry for entry in doc["records"].values()}
    received = {}
    with tarfile.open(src, "r:gz") as tar:
        seen = set()
        for member in tar:
            if not member.isfile() or member.name in seen:
                raise EvidenceError("bundle has duplicate or non-file members")
            seen.add(member.name)
            if member.name == "INDEX.json":
                if member.size > INDEX.stat().st_size * 2 + 1024:
                    raise EvidenceError("oversized bundle index")
                incoming = json.load(tar.extractfile(member))
                if incoming != doc:
                    raise EvidenceError("bundle index differs from this checkout")
                continue
            entry = expected.get(member.name)
            if entry is None or member.size != entry["bytes"]:
                raise EvidenceError(f"unexpected bundle member: {member.name}")
            data = tar.extractfile(member).read(entry["bytes"] + 1)
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise EvidenceError(f"bundle SHA-256 mismatch: {member.name}")
            received[member.name] = data
    if seen != {"INDEX.json", *expected}:
        raise EvidenceError("incomplete bundle")
    with _locked():
        if index() != doc:
            raise EvidenceError("index changed during import")
        # Check all collisions before publishing any new files.
        for name, data in received.items():
            dest = _archive(expected[name])
            if dest.exists() and dest.read_bytes() != data:
                raise EvidenceError(f"immutable archive collision: {name}")
        for name, data in received.items():
            _atomic(_archive(expected[name]), data)
    return {"imported_records": len(received)}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    export = sub.add_parser("export")
    export.add_argument("--out", required=True)
    imp = sub.add_parser("import")
    imp.add_argument("--bundle", required=True)
    args = parser.parse_args()
    result = (verify() if args.command == "verify" else
              export_bundle(args.out) if args.command == "export" else import_bundle(args.bundle))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
