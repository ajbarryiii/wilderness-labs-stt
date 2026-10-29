"""A persisted label store (stage 3; b0prime plan v4.1): a training/labels.py LabelStore written once per label
identity and memory-mapped by every later run start, so a chunk job does not re-read and re-clean every label shard.

Key (key_of): everything a LabelStore's texts depend on, plus what the run applies on top of it, as one JSON object:
the label run's directory, the C1 directory, the label set, through_batch, the run contract's SHA-256, every needed
label shard with its meta SHA-256, the cleaning identity (version, tables, choices, layer 0; None = no cleaning), the
repetition rule (name, version, n, threshold, tokenizer; None = no rule), the read-time reject list's identity, the
real plan's SHA-256 (a version 2 plan's base plan: its real occurrences) and the SHA-256 of the code that computes
the texts (labels.py, cleaning.py, repetition.py and this file). key_sha256 is the SHA-256 of the key.

Directory (index_dir(root, key)): <root>/<key_sha256[:24]>/ with
    index.json   {"format", "key", "key_sha256", "identity" (the LabelStore's), "repetition_rejected", "records",
                  "files": {name: SHA-256}}
    ids.bin      the ids, UTF-8, concatenated, in the store's order; ids.off: uint64 offsets (records + 1)
    texts.bin    the texts, UTF-8, concatenated; texts.off: uint64 offsets (records + 1)
    state.u8     per record: 0 = no label (None), 1 = a label (an empty text is a label)
write() builds it in a temporary sibling and renames it into place (an existing index is kept). load() refuses an
index whose recorded key differs from the requested one in any field, an unknown format, or a file whose SHA-256
differs from index.json; it memory-maps texts.bin and decodes the ids into a dict. IndexedLabelStore answers like the
LabelStore it was written from: text(), kept(), len(), `in`, identity, label_set, repetition_rejected.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

FORMAT = 1
FILES = ("ids.bin", "ids.off", "texts.bin", "texts.off", "state.u8")
CODE = ("training/labels.py", "training/cleaning.py", "training/repetition.py", "training/label_index.py")


class IndexMismatch(RuntimeError):
    """A persisted label index of another identity, format or content."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256(base: Path | None = None) -> dict:
    """{relative path: SHA-256} of the code that computes the texts (CODE, relative to finetune/relabel)."""
    base = Path(__file__).resolve().parents[1] if base is None else Path(base)
    return {p: _sha256(base / p) for p in CODE}


def key_of(parts: dict) -> tuple[dict, str]:
    """(key, key_sha256): parts with the format, canonical JSON."""
    key = json.loads(json.dumps({"format": FORMAT, **parts}, sort_keys=True, default=str))
    return key, hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def index_dir(root: Path, key_sha256: str) -> Path:
    return Path(root) / key_sha256[:24]


def write(directory: Path, store, key: dict, key_sha256: str) -> Path:
    """Persist `store` (a LabelStore) under `key` at `directory` (atomic; an existing index is kept)."""
    directory = Path(directory)
    if (directory / "index.json").exists():
        return directory
    tmp = directory.with_name(f".{directory.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    items = list(store._texts.items())
    ids = [i.encode() for i, _ in items]
    texts = [b"" if t is None else t.encode() for _, t in items]
    arrays = {"ids.off": np.concatenate([[0], np.cumsum([len(x) for x in ids], dtype=np.uint64)]).astype(np.uint64),
              "texts.off": np.concatenate([[0], np.cumsum([len(x) for x in texts], dtype=np.uint64)]).astype(np.uint64),
              "state.u8": np.array([t is not None for _, t in items], dtype=np.uint8)}
    blobs = {"ids.bin": b"".join(ids), "texts.bin": b"".join(texts)}
    for name in FILES:
        with open(tmp / name, "wb") as f:
            f.write(blobs[name] if name in blobs else arrays[name].tobytes())
            f.flush()
            os.fsync(f.fileno())
    body = {"format": FORMAT, "key": key, "key_sha256": key_sha256, "identity": store.identity,
            "repetition_rejected": store.repetition_rejected, "records": len(items),
            "files": {name: _sha256(tmp / name) for name in FILES}}
    (tmp / "index.json").write_text(json.dumps(body, sort_keys=True, default=str))
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(tmp, directory)
    except OSError:
        if not (directory / "index.json").exists():
            raise
        shutil.rmtree(tmp, ignore_errors=True)
    return directory


def load(directory: Path, key: dict, key_sha256: str) -> "IndexedLabelStore":
    """The persisted store at `directory`, refused unless it was written under exactly `key` (module docstring)."""
    directory = Path(directory)
    body = json.loads((directory / "index.json").read_text())
    if body.get("format") != FORMAT:
        raise IndexMismatch(f"{directory}: label index format {body.get('format')}, not {FORMAT}")
    if body.get("key_sha256") != key_sha256 or body.get("key") != key:
        diff = sorted(k for k in set(key) | set(body.get("key") or {})
                      if (body.get("key") or {}).get(k) != key.get(k))
        raise IndexMismatch(f"{directory}: a label index of another identity (differs in {diff})")
    for name in FILES:
        if _sha256(directory / name) != body["files"].get(name):
            raise IndexMismatch(f"{directory}/{name}: SHA-256 differs from index.json")
    return IndexedLabelStore(directory, body)


class IndexedLabelStore:
    def __init__(self, directory: Path, body: dict):
        self.dir = Path(directory)
        self.identity = body["identity"]
        self.label_set = self.identity["label_set"]
        self.repetition_rejected = body["repetition_rejected"]
        n = int(body["records"])
        ids_off = np.fromfile(self.dir / "ids.off", dtype=np.uint64)
        self._toff = np.fromfile(self.dir / "texts.off", dtype=np.uint64)
        self._state = np.fromfile(self.dir / "state.u8", dtype=np.uint8)
        if len(ids_off) != n + 1 or len(self._toff) != n + 1 or len(self._state) != n:
            raise IndexMismatch(f"{self.dir}: arrays do not hold {n} records")
        blob = (self.dir / "ids.bin").read_bytes()
        bounds = ids_off.tolist()
        self._row = {blob[bounds[k]:bounds[k + 1]].decode(): k for k in range(n)}
        if len(self._row) != n:
            raise IndexMismatch(f"{self.dir}: an id is recorded twice")
        size = int(self._toff[-1])
        self._texts = np.memmap(self.dir / "texts.bin", dtype=np.uint8, mode="r") if size else np.zeros(0, np.uint8)
        if len(self._texts) != size:
            raise IndexMismatch(f"{self.dir}/texts.bin: {len(self._texts)} bytes, offsets say {size}")

    def text(self, item_id: str) -> str | None:
        """As LabelStore.text: the training text, None when T* rejected it, KeyError for an id without a record."""
        k = self._row[item_id]
        if not self._state[k]:
            return None
        return bytes(self._texts[int(self._toff[k]):int(self._toff[k + 1])]).decode()

    def kept(self, item_id: str) -> bool:
        return bool(self._state[self._row[item_id]])

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._row

    def __len__(self) -> int:
        return len(self._row)

    def ids(self):
        return iter(self._row)
