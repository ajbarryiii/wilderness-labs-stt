"""The offline label lookup by id (b0prime plan "No online teacher", decoder-only plan "Arms").

C2 (labeling/c2.py) writes one record per distinct id, in the label shard of the C1 shard where the id first
occurs (labels/<T*>/<run>/shards/<k>.labels.parquet, LABEL_COLUMNS, SHA-256 in <k>.json). A LabelStore holds,
for every id first seen in the C1 shards that cover the consumed batches [0, through_batch), the training text
of one label set:

    "T*"  the record's `text` (T*'s mapped label)
    "B0"  the record's `b0_raw` (B0's greedy label, as the online teacher produced it)

Both under T*'s stored disposition: an id whose disposition is not "kept" has no label (None) in either set,
with no re-filtering, so a B0-label control and a T* arm train on exactly the same utterances. A kept id
without a B0 label is an error (it would change the control's population); an empty B0 label is kept as is.

The store refuses: a run whose run.json names another C1 directory, a missing label shard of the range, a label
shard whose SHA-256 differs from its meta, an id recorded twice. A lookup of an id without a record raises
KeyError (every replayed id must have one).

Label cleaning (cleaning plan, stage 2a; training/cleaning.py): with `cleaning` (a loaded Cleaning spec), every kept
text of either label set is read through cleaning.clean, which never changes which ids are kept, never returns None
and never empties a non-empty text (an empty kept text stays empty and kept). The identity gains "cleaning"
({version, tables_sha256, choices_sha256}, which enters identity["sha256"]) and "cleaned_labels". The store refuses a
choices artifact of another label run contract or tables, or whose recorded spans do not match a label. With
cleaning=None the texts and the identity are exactly the uncleaned ones.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from labeling import c2, layout, shards

LABEL_SETS = {"T*": "text", "B0": "b0_raw"}
MASK = "T* disposition == kept (labeling/filters.py), applied to every label set"


def needed_shards(pool_dir: Path, through_batch: int) -> list[int]:
    """C1 shards whose batch range starts below through_batch (from their committed metas)."""
    if through_batch < 1:
        raise ValueError("through_batch must be positive")
    out, covered = [], 0
    for k in shards.committed(pool_dir):
        meta = shards.load_meta(pool_dir, k)
        lo, hi = meta["batches"]
        if lo != covered:
            raise RuntimeError(f"C1 shard {k} starts at batch {lo}, not {covered}")
        if lo >= through_batch:
            break
        out.append(k)
        covered = hi
    if covered < through_batch:
        raise RuntimeError(f"C1 covers {covered} consumed batches, fewer than {through_batch}")
    return out


class LabelStore:
    def __init__(self, texts: dict[str, str | None], identity: dict):
        self._texts = texts
        self.identity = identity
        self.label_set = identity["label_set"]

    @classmethod
    def open(cls, run_dir: Path, pool_dir: Path, *, label_set: str, through_batch: int,
             cleaning=None) -> "LabelStore":
        """cleaning: a training.cleaning.Cleaning (every kept text is read through cleaning.clean; the identity
        gains its block, which enters identity["sha256"]) or None (the stored texts and identity, unchanged)."""
        import pyarrow.parquet as pq
        if label_set not in LABEL_SETS:
            raise ValueError(f"label set {label_set!r} is not one of {sorted(LABEL_SETS)}")
        column = LABEL_SETS[label_set]
        run_dir, pool_dir = Path(run_dir), Path(pool_dir)
        run = c2.load_run(run_dir)
        if c2._abs(run["pool_dir"]).resolve() != pool_dir.resolve():
            raise RuntimeError(f"{run_dir} labels {run['pool_dir']}, not {pool_dir}")
        contract = c2.contract(run)
        contract_sha256 = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        if cleaning is not None:
            cleaning.bind_run(contract_sha256)
        texts: dict[str, str | None] = {}
        listed, kept, cleaned = [], 0, 0
        for k in needed_shards(pool_dir, through_batch):
            path = run_dir / "shards" / f"{shards.shard_name(k)}.labels.parquet"
            meta_path = path.with_name(f"{shards.shard_name(k)}.json")
            if not path.exists() or not meta_path.exists():
                raise RuntimeError(f"no label shard {path.name} (C2 has not labeled C1 shard {k} yet)")
            meta = json.loads(meta_path.read_text())
            if shards.sha256_file(path) != meta["sha256"]:
                raise RuntimeError(f"{path}: SHA-256 differs from its meta")
            table = pq.read_table(path, columns=["id", "disposition", "teacher", column]).to_pydict()
            if len(table["id"]) != meta["records"]:
                raise RuntimeError(f"{path}: {len(table['id'])} records, meta says {meta['records']}")
            for i, disposition, teacher, value in zip(table["id"], table["disposition"], table["teacher"],
                                                      table[column]):
                if i in texts:
                    raise RuntimeError(f"id {i} has two label records")
                if teacher != run["teacher"]:
                    raise RuntimeError(f"{path}: record of teacher {teacher!r} in a {run['teacher']} run")
                if disposition == "kept":
                    if value is None:
                        raise RuntimeError(f"{i}: kept by T* but no {column}")
                    if cleaning is not None:
                        text = cleaning.clean(value, i, label_set)
                        if text is None or (value != "" and text == ""):
                            raise RuntimeError(f"{i}: cleaning returned {text!r} for a kept label")
                        cleaned += text != value
                        value = text
                    texts[i] = value
                    kept += 1
                else:
                    texts[i] = None
            listed.append({"shard": k, "sha256": meta["sha256"], "records": meta["records"]})
        identity = {"run_dir": c2._rel(run_dir), "pool_dir": c2._rel(pool_dir), "teacher": run["teacher"],
                    "teacher_revision": run["teacher_revision"],
                    "contract_sha256": contract_sha256,
                    "label_set": label_set, "column": column, "mask": MASK, "through_batch": int(through_batch),
                    "shards": listed, "records": len(texts), "kept": kept}
        hashed = {k: identity[k] for k in ("label_set", "column", "contract_sha256", "shards")}
        if cleaning is not None:
            identity["cleaning"] = hashed["cleaning"] = cleaning.identity()
            identity["cleaned_labels"] = cleaned
        identity["sha256"] = hashlib.sha256(json.dumps(hashed, sort_keys=True).encode()).hexdigest()
        return cls(texts, identity)

    def text(self, item_id: str) -> str | None:
        """The training text of an id, or None when T* rejected it; KeyError for an id without a record."""
        return self._texts[item_id]

    def kept(self, item_id: str) -> bool:
        return self._texts[item_id] is not None

    def __len__(self) -> int:
        return len(self._texts)


def production_run_dir(run: str) -> Path:
    """$RELABEL_ROOT/labels/<T*>/<run> (labeling.layout)."""
    return layout.labels_dir(layout.TEACHER, run)
