"""Repetition rejection, rule "repetition" version 1: a read-time rule over a label run's T* labels.

Why (stage 2b, run B-T lr3e-5, NaN gradient norm at step 2839): a few T* labels are hallucinated repetitions that ran
to the teacher's generation cap ("Wurrrr…": one SentencePiece token 486 times in a 13 s clip). They pass the stage 2
filters (labeling/filters.py counts words, and "Wurrrr…" is one word) and label cleaning, and through the prediction
network's recurrence they give gradient norms of 1e4 to inf. The rule:

    tokens    the T* label as stored (the record's `text`, before label cleaning), encoded with the students'
              SentencePiece model (Kea-B's, Kea-T's and B0's; TOKENIZER_SHA256) as train.tokenize encodes it
    repeats   the longest run of an immediately repeated token n-gram, n = 1..N_MAX, counted in repeats of the
              n-gram (longest_repeat): "a b a b a b" is 3 repeats of "a b"
    rejected  repeats >= THRESHOLD, for an id T* kept

It reads the T* label only (B0's labels hold genuine repetitions, "no no no", up to 22 in the stage 2 run) and it
extends T*'s disposition mask: training/labels.py LabelStore.open(..., repetition=spec) gives a rejected id no label in
either label set, so a B0-label control and its T* arm still train on exactly the same utterances.

    spec = Repetition(tokenizer_file, cache_path)     or production(label_run)
    spec.repeats(text) -> (repeats, n)
    spec.evaluate(ids, dispositions, texts)           one label shard: {"rejected": [[id, repeats, n]], "kept",
                                                      "histogram"}
    spec.identity(rejected_ids)                       {name, version, n, threshold, tokenizer_sha256, label,
                                                       rejected, rejected_sha256}

The identity enters the store's identity["sha256"] (so a run's labels_sha256 and resume key): rejected_sha256 is the
SHA-256 of the sorted rejected ids of the store's label shards, one per line.

Cache (one file per label run, $RELABEL_ROOT/cleaning/<label run>/repetition-v<VERSION>.json): the rule, the label
run's contract SHA-256 and, per label shard, its SHA-256 and evaluate()'s result. A shard's entry is used only when the
rule, the contract and the shard's SHA-256 all match; otherwise the shard is evaluated again from its `text` column and
the entry rewritten (the file is replaced atomically, under a lock, merged with entries other processes wrote). A cached
rejected id that is not a kept record of its shard is refused. Recompute and compare every shard:

    ./python -m training.repetition check --labels-run RUN [--write]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

NAME = "repetition"
VERSION = 1
N_MAX = 6
THRESHOLD = 32                   # genuine labels reach 13 repeats in main-f3, the hallucinations 105 and more
LABEL = "T* text as stored (before label cleaning), kept ids"
# Kea-B's export tokenizer.model (binary_test_paths.PINNED_FILES); B0's .nemo holds the same file
TOKENIZER_SHA256 = "5a3a82c48998709f9bc9f5db2924d99a899d20bf82550ca52272da7c7557b0a0"


def longest_repeat(tokens, n_max: int = N_MAX) -> tuple[int, int]:
    """(repeats, n): the longest run of an immediately repeated n-gram, n = 1..n_max, in repeats of the n-gram, and
    its n (the smallest on ties). (0, 0) for an empty sequence, (1, 1) for one without repetition."""
    t = list(tokens)
    if not t:
        return 0, 0
    best = (1, 1)
    for n in range(1, n_max + 1):
        run = longest = 0
        for a, b in zip(t, t[n:]):                 # t[i] == t[i + n] over a stretch of m: m // n + 1 repeats
            run = run + 1 if a == b else 0
            longest = max(longest, run)
        if longest // n + 1 > best[0]:
            best = (longest // n + 1, n)
    return best


def longest_repeats(seqs, n_max: int = N_MAX) -> tuple[np.ndarray, np.ndarray]:
    """longest_repeat of every sequence (repeats, n arrays), vectorized: the sequences are laid out one after the
    other with n_max distinct negative separators between them, so no comparison at distance <= n_max crosses two
    sequences and no run of equal pairs spans a separator."""
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    count = len(lengths)
    repeats, ns = np.ones(count, dtype=np.int64), np.ones(count, dtype=np.int64)
    if count == 0:
        return repeats, ns
    starts = np.concatenate([[0], np.cumsum(lengths + n_max)[:-1]])
    flat = -np.arange(1, int(lengths.sum()) + n_max * count + 1, dtype=np.int64)
    if lengths.sum():
        within = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        flat[np.repeat(starts, lengths) + within] = np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs
                                                                    if len(s)])
    for n in range(1, n_max + 1):
        eq = np.concatenate([[False], flat[:-n] == flat[n:], [False]])
        edges = np.flatnonzero(np.diff(eq.astype(np.int8)))
        run_start, run_len = edges[0::2], edges[1::2] - edges[0::2]
        longest = np.zeros(count, dtype=np.int64)
        np.maximum.at(longest, np.searchsorted(starts, run_start, side="right") - 1, run_len)
        r = longest // n + 1
        better = r > repeats
        repeats[better], ns[better] = r[better], n
    empty = lengths == 0
    repeats[empty], ns[empty] = 0, 0
    return repeats, ns


def rejects(repeats: int, threshold: int = THRESHOLD) -> bool:
    return repeats >= threshold


def rejected_sha256(ids) -> str:
    return hashlib.sha256("".join(f"{i}\n" for i in sorted(ids)).encode()).hexdigest()


class Repetition:
    """The rule with a tokenizer and an optional cache file (module docstring)."""

    def __init__(self, tokenizer_file: Path, cache: Path | None = None, *, threshold: int = THRESHOLD,
                 n_max: int = N_MAX):
        import sentencepiece as spm
        data = Path(tokenizer_file).read_bytes()
        self.tokenizer_sha256 = hashlib.sha256(data).hexdigest()
        self.sp = spm.SentencePieceProcessor(model_proto=data)
        self.threshold, self.n_max = int(threshold), int(n_max)
        self.cache_path = None if cache is None else Path(cache)
        self.contract_sha256: str | None = None
        self._entries: dict[str, dict] = {}
        self._new: dict[str, dict] = {}

    def rule(self) -> dict:
        return {"name": NAME, "version": VERSION, "n": [1, self.n_max], "threshold": self.threshold,
                "tokenizer_sha256": self.tokenizer_sha256, "label": LABEL}

    def identity(self, rejected_ids) -> dict:
        ids = sorted(set(rejected_ids))
        return self.rule() | {"rejected": len(ids), "rejected_sha256": rejected_sha256(ids)}

    def repeats(self, text: str) -> tuple[int, int]:
        return longest_repeat(self.sp.encode(text), self.n_max)

    def evaluate(self, ids: list[str], dispositions: list[str], texts: list[str | None]) -> dict:
        """One label shard: the kept ids the rule rejects ([id, repeats, n], in record order), the number of kept
        ids, and the histogram of repeats over them."""
        kept = [(i, t) for i, d, t in zip(ids, dispositions, texts) if d == "kept"]
        if any(t is None for _, t in kept):
            raise RuntimeError("a kept record without a T* text")
        repeats, ns = longest_repeats(self.sp.encode([t for _, t in kept]), self.n_max)
        values, counts = np.unique(repeats, return_counts=True)
        return {"rejected": [[i, int(r), int(n)] for (i, _), r, n in zip(kept, repeats, ns)
                             if rejects(int(r), self.threshold)],
                "kept": len(kept), "histogram": {str(int(v)): int(c) for v, c in zip(values, counts)}}

    # ------------------------------------------------------------------ the label store's use

    def bind_run(self, contract_sha256: str) -> None:
        """Read the cache's entries of this rule and label run contract (none if the file is of another)."""
        self.contract_sha256 = contract_sha256
        self._entries, self._new = {}, {}
        cached = self._read()
        if cached.get("rule") == self.rule() and cached.get("contract_sha256") == contract_sha256:
            self._entries = cached.get("shards", {})

    def shard(self, name: str, sha256: str, ids: list[str], dispositions: list[str], read_texts) -> dict[str, list]:
        """{rejected id: [repeats, n]} of one label shard (`read_texts()` gives its T* texts, in record order, when
        the cache has no entry for this shard SHA-256)."""
        entry = self._entries.get(name)
        if entry is None or entry.get("sha256") != sha256:
            entry = {"sha256": sha256, **self.evaluate(ids, dispositions, read_texts())}
            self._entries[name] = self._new[name] = entry
        kept = {i for i, d in zip(ids, dispositions) if d == "kept"}
        out = {}
        for i, r, n in entry["rejected"]:
            if i not in kept:
                raise RuntimeError(f"repetition cache {self.cache_path}: {i} is not a kept record of label shard "
                                   f"{name}")
            out[i] = [r, n]
        return out

    def save(self) -> None:
        """Write the entries computed since bind_run into the cache file (atomically, merged, under a lock)."""
        if self.cache_path is None or not self._new:
            return
        import fcntl
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path.with_name(self.cache_path.name + ".lock"), "a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cached = self._read()
            same = cached.get("rule") == self.rule() and cached.get("contract_sha256") == self.contract_sha256
            entries = dict(cached.get("shards", {})) if same else {}
            entries.update(self._new)
            body = {"rule": self.rule(), "contract_sha256": self.contract_sha256,
                    "shards": dict(sorted(entries.items()))}
            tmp = self.cache_path.with_name(f"{self.cache_path.name}.tmp{os.getpid()}")
            tmp.write_text(json.dumps(body, sort_keys=True) + "\n")
            os.replace(tmp, self.cache_path)
        self._new = {}

    def _read(self) -> dict:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text())
        except ValueError:
            return {}


def cache_path(label_run: str) -> Path:
    """$RELABEL_ROOT/cleaning/<label run>/repetition-v<VERSION>.json."""
    from labeling import layout
    return layout.relabel_root() / "cleaning" / label_run / f"repetition-v{VERSION}.json"


def production(label_run: str) -> Repetition:
    """The rule with the students' tokenizer (B0's, extracted once from the pinned .nemo) and the label run's cache."""
    from labeling import filters, layout
    spec = Repetition(filters.b0_tokenizer_file(layout.relabel_root() / "caches" / "labeling"), cache_path(label_run))
    if spec.tokenizer_sha256 != TOKENIZER_SHA256:
        raise RuntimeError(f"the tokenizer's SHA-256 {spec.tokenizer_sha256} is not the students' {TOKENIZER_SHA256}")
    return spec


# ----------------------------------------------------------------------------- check

def check(label_run: str, write: bool = False) -> dict:
    """Evaluate every label shard of a label run afresh and compare with the cache (written when `write`)."""
    import pyarrow.parquet as pq
    from labeling import c2, shards
    from training import labels as tlabels
    run_dir = tlabels.production_run_dir(label_run)
    contract = c2.contract(c2.load_run(run_dir))
    contract_sha256 = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    spec = production(label_run)
    spec.bind_run(contract_sha256)
    cached = dict(spec._entries)
    t0 = time.perf_counter()
    rejected, sources, histogram, mismatched, kept, n_shards, read_s = {}, {}, {}, [], 0, 0, 0.0
    for meta_path in sorted((run_dir / "shards").glob("*.json")):
        name = meta_path.stem
        path = meta_path.with_name(f"{name}.labels.parquet")
        meta = json.loads(meta_path.read_text())
        t1 = time.perf_counter()
        if shards.sha256_file(path) != meta["sha256"]:
            raise RuntimeError(f"{path}: SHA-256 differs from its meta")
        t = pq.read_table(path, columns=["id", "source", "disposition", "text"]).to_pydict()
        read_s += time.perf_counter() - t1
        entry = {"sha256": meta["sha256"], **spec.evaluate(t["id"], t["disposition"], t["text"])}
        old = cached.get(name)
        if old is not None and old != entry:
            mismatched.append(name)
        spec._entries[name] = spec._new[name] = entry
        source = dict(zip(t["id"], t["source"]))
        for i, r, n in entry["rejected"]:
            rejected[i] = (r, n)
            sources[source[i]] = sources.get(source[i], 0) + 1
        for v, c in entry["histogram"].items():
            histogram[int(v)] = histogram.get(int(v), 0) + c
        kept += entry["kept"]
        n_shards += 1
    seconds = time.perf_counter() - t0
    if write:
        spec.save()
    below = max((v for v in histogram if v < spec.threshold), default=None)
    above = min((v for v in histogram if v >= spec.threshold), default=None)
    return {"label_run": label_run, "rule": spec.rule(), "shards": n_shards, "kept": kept,
            "rejected": len(rejected), "rejected_sha256": rejected_sha256(rejected), "by_source": sources,
            "largest_kept_repeats": below, "smallest_rejected_repeats": above,
            "histogram": dict(sorted(histogram.items())), "cached_shards": len(cached),
            "cache_mismatches": mismatched, "seconds": seconds, "read_seconds": read_s,
            "rejected_ids": {i: list(v) for i, v in sorted(rejected.items())}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.repetition", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="evaluate every label shard of a run afresh and compare with the cache")
    c.add_argument("--labels-run", required=True)
    c.add_argument("--write", action="store_true", help="write the cache (every shard)")
    c.add_argument("--ids", action="store_true", help="list the rejected ids")
    args = p.parse_args(argv)
    r = check(args.labels_run, write=args.write)
    if not args.ids:
        r.pop("rejected_ids")
    print(json.dumps(r, indent=1))
    return 1 if r["cache_mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
