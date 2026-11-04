"""The talk-text reject list of the main pool (b0prime plan "Extra sources", main-pool rule; the containment rule of
labeling plan Part B), run as `./python -m pool.tedreject build`.

Held-out text: registry alias ted-test-text (the `text` column only; one row per talk). Each talk is normalized with
the evaluation normalizer and cut into consecutive WINDOW-word windows, [0, 20), [20, 40), ... (the last one may be
shorter). Every window is a held-out utterance whose 8-grams are the word 8-grams entirely inside it.

Training text: every record of the label run (one per distinct id of the 782-pool plan, whatever its disposition),
under both label sets as training reads them (training/labels.py LABEL_SETS): T* = `text`, B0 = `b0_raw`. A label
run's shard is read only if its SHA-256 equals the run manifest's.

Recordings (recording_key), from the record id "<source>:<native id>":
    yodas, yodas-med, yodas-tech   "yodas:" + the YODAS original_audio_id: native fields 2..-5 joined by "_"
                                   (heldout.yodas_dev_recording's rule), ordered by the offset fields (-4, -3)
    peoples_speech, ps-med         "peoples_speech:" + the id without its "_<segment>.flac" suffix
                                   (common.ps_recording), ordered by the segment index
    librispeech                    "librispeech:<speaker>-<chapter>" (one chapter recording), by utterance number
    voxpopuli                      "voxpopuli:" + the field before the first "_" (the plenary item's audio,
                                   "<date>-<time>-PLENARY-<item>-<lang>"), by the segment's time stamp and index
    ami                            "ami:AMI_<meeting>_<headset>" (one headset recording), by the start field
    med-osce (local corpora)       the corpus manifest's `recording`, by its `start` (the denylist's rule)
Subset sources share the recordings of their corpus (a yodas-med recording is a yodas recording). Any other source
or id form is refused. Within a recording, records are ordered by (order, id).

A recording's transcript under one label set is its records' normalized labels concatenated in that order, so
8-grams crossing record boundaries count (pool/scan.py's convention). Word 8-grams are compared as 64-bit hashes
(pool/common.py).

Rule: a window 8-gram that occurs in more than BOILERPLATE_MAX recordings (under either label set) is boilerplate.
A window with fewer than 8 words, or with no non-boilerplate 8-gram, is not checked (counted). A checked window
with n distinct non-boilerplate 8-grams is contained in a recording's transcript if at least ceil(0.5 n) of them
occur in it. A recording with a contained window under either label set is rejected, and every record id of a
rejected recording is listed.

Outputs ($RELABEL_ROOT/pool/ted-reject/): rejected_ids.txt (sorted ids, one per line; empty if none), reject.json
(rule, inputs and their SHA-256, counts, the rejected recordings with their windows, ids_sha256 = SHA-256 of
rejected_ids.txt, the closest recordings for information). load() returns the id set and the identity a training
run hashes into its label identity.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from pool import common

NAME = "talk-text-reject"
VERSION = 1
ALIAS = "ted-test-text"
WINDOW = 20
HEAD = common.NGRAM - 1          # words of context a cross-record 8-gram can take from either side
LABEL_SETS = {"T*": "text", "B0": "b0_raw"}
YODAS = ("yodas", "yodas-med", "yodas-tech")
PEOPLE = ("peoples_speech", "ps-med")
_LS = re.compile(r"(\d+)-(\d+)-(\d+)")
_VOX = re.compile(r"([^_]+)_(\d{8})-(\d{2}):(\d{2}):(\d{2})_(\d+)")
_AMI = re.compile(r"(AMI_[A-Za-z0-9]+_[A-Za-z0-9]+)_[A-Za-z0-9]+_(\d+)_(\d+)")


def out_dir() -> Path:
    return common.pool_dir() / "ted-reject"


def recording_key(item_id: str, local: dict[str, tuple[str, float]]) -> tuple[str, tuple[float, float]]:
    """(recording key, order within the recording) of one record id (module docstring)."""
    source, sep, native = item_id.partition(":")
    if not sep or not native:
        raise ValueError(f"record id without a source prefix: {item_id!r}")
    if source in YODAS:
        parts = native.split("_")
        if len(parts) < 7 or not all(p.isdigit() for p in parts[-4:]) or len(parts[-3]) != 2:
            raise ValueError(f"unexpected yodas id {item_id!r}")       # offset = parts[-4] seconds, parts[-3] cs
        return "yodas:" + "_".join(parts[2:-4]), (float(int(parts[-4])), float(int(parts[-3])))
    if source in PEOPLE:
        rec, seg = common.ps_recording(native)
        return "peoples_speech:" + rec, (seg, 0.0)
    if source == "librispeech":
        m = _LS.fullmatch(native)
        if not m:
            raise ValueError(f"unexpected librispeech id {item_id!r}")
        return f"librispeech:{m.group(1)}-{m.group(2)}", (float(int(m.group(3))), 0.0)
    if source == "voxpopuli":
        m = _VOX.fullmatch(native)
        if not m:
            raise ValueError(f"unexpected voxpopuli id {item_id!r}")
        stamp = int(m.group(2)) * 1_000_000 + int(m.group(3)) * 10_000 + int(m.group(4)) * 100 + int(m.group(5))
        return "voxpopuli:" + m.group(1), (float(stamp), float(int(m.group(6))))
    if source == "ami":
        m = _AMI.fullmatch(native)
        if not m:
            raise ValueError(f"unexpected ami id {item_id!r}")
        return "ami:" + m.group(1), (float(int(m.group(2))), float(int(m.group(3))))
    if item_id in local:
        rec, start = local[item_id]
        return f"{source}:{rec}", (float(start), 0.0)
    raise ValueError(f"no recording rule for {item_id!r}")


def local_recordings(manifests: dict[str, Path]) -> dict[str, tuple[str, float]]:
    """id -> (recording, start) from local corpus manifests."""
    out = {}
    for alias, path in manifests.items():
        for r in common.read_jsonl(path):
            out[str(r["id"])] = (str(r["recording"]), float(r.get("start") or 0.0))
    return out


# --- held-out windows ---------------------------------------------------------------------------------------------

class Windows:
    """The talk windows: per window its words and distinct 8-gram hashes, indexed over all window 8-grams."""

    def __init__(self, talks: Sequence[str], normalize: Callable[[str], str], hasher=None):
        hasher = hasher or common.WordHasher()
        self.talk, self.start, self.words, grams = [], [], [], []
        for t, text in enumerate(talks):
            w = normalize(text).split()
            for s in range(0, len(w), WINDOW):
                win = w[s:s + WINDOW]
                self.talk.append(t)
                self.start.append(s)
                self.words.append(len(win))
                grams.append(np.unique(common.ngram_hashes(hasher(win))) if len(win) >= common.NGRAM
                             else np.empty(0, np.uint64))
        flat = np.concatenate(grams) if grams else np.empty(0, np.uint64)
        self.hashes, inverse = np.unique(flat, return_inverse=True)
        lens = [len(g) for g in grams]
        self.grams = np.split(inverse.astype(np.int64), np.cumsum(lens)[:-1]) if grams else []
        self.windows_of = defaultdict(list)    # gram index -> windows
        for w, g in enumerate(self.grams):
            for x in g.tolist():
                self.windows_of[x].append(w)
        self.talks = len(talks)

    def index(self, hashes: np.ndarray) -> np.ndarray:
        """Distinct indices into self.hashes of the given 8-gram hashes that are window 8-grams."""
        if not len(hashes) or not len(self.hashes):
            return np.empty(0, np.int64)
        idx = np.searchsorted(self.hashes, hashes)
        idx[idx >= len(self.hashes)] = 0
        return np.unique(idx[self.hashes[idx] == hashes]).astype(np.int64)


# --- the per-shard pass -------------------------------------------------------------------------------------------

_STATE: dict = {}


def _init(window_hashes: np.ndarray) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    _STATE.update(hashes=window_hashes, hasher=common.WordHasher())


def _label_pass(texts: Sequence[str | None]) -> tuple[list[tuple[int, np.ndarray]], np.ndarray, np.ndarray,
                                                        np.ndarray]:
    """(hits [(row, window-gram indices)], words per row, head [n, HEAD], tail [n, HEAD]) of one label set."""
    hashes, hasher = _STATE["hashes"], _STATE["hasher"]
    n = len(texts)
    head = np.zeros((n, HEAD), np.uint64)
    tail = np.zeros((n, HEAD), np.uint64)
    nwords = np.zeros(n, np.int64)
    hits = []
    for i, text in enumerate(texts):
        words = common.normalize(text).split() if text else []
        if not words:
            continue
        wh = hasher(words)
        nwords[i] = len(wh)
        k = min(HEAD, len(wh))
        head[i, :k] = wh[:k]
        tail[i, :k] = wh[len(wh) - k:]
        g = common.ngram_hashes(wh)
        if len(g) and len(hashes):
            idx = np.searchsorted(hashes, g)
            idx[idx >= len(hashes)] = 0
            ok = hashes[idx] == g
            if ok.any():
                hits.append((i, np.unique(idx[ok]).astype(np.int64)))
    return hits, nwords, head, tail


def scan_shard(task: tuple[str, str]) -> dict:
    """One label shard (path, expected SHA-256): ids and, per label set, the _label_pass output."""
    import pyarrow.parquet as pq
    path, sha = task
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError(f"{path}: SHA-256 differs from the label run manifest")
    t = pq.read_table(io.BytesIO(data), columns=["id", *LABEL_SETS.values()]).to_pydict()
    return {"ids": t["id"], "sets": {name: _label_pass(t[col]) for name, col in LABEL_SETS.items()}}


def _cross(buffer: np.ndarray, head: np.ndarray) -> np.ndarray:
    """8-gram hashes of buffer + head that start inside the buffer (the record-crossing 8-grams)."""
    seq = np.concatenate([buffer, head])
    g = common.ngram_hashes(seq)
    return g[:len(buffer)] if len(g) else g


def build(label_run: Path, talks: Sequence[str], normalize: Callable[[str], str], local: dict[str, tuple[str, float]],
          workers: int = 6, max_shards: int | None = None, log=common.log) -> tuple[dict, list[str]]:
    """(reject.json body, sorted rejected ids) for the label run (module docstring)."""
    manifest = json.loads((label_run / "manifest.json").read_text())
    if not manifest.get("complete") and max_shards is None:
        raise RuntimeError(f"{label_run}: the label run is not complete")
    shards = manifest["shards"][:max_shards] if max_shards else manifest["shards"]
    win = Windows(talks, normalize)
    log(f"tedreject: {win.talks} talks, {len(win.words)} windows, {len(win.hashes)} window 8-grams")
    tasks = [(str(label_run / "shards" / s["file"]), s["sha256"]) for s in shards]

    ids: list[str] = []
    per_set = {name: {"hits": [], "nwords": [], "head": [], "tail": []} for name in LABEL_SETS}
    if workers > 0:
        ex = cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init,
                                    initargs=(win.hashes,))
        results = ex.map(scan_shard, tasks, chunksize=2)
    else:
        _init(win.hashes)
        ex, results = None, map(scan_shard, tasks)
    try:
        for n, res in enumerate(results, 1):
            base = len(ids)
            ids += res["ids"]
            for name, (hits, nwords, head, tail) in res["sets"].items():
                per_set[name]["hits"] += [(base + i, h) for i, h in hits]
                per_set[name]["nwords"].append(nwords)
                per_set[name]["head"].append(head)
                per_set[name]["tail"].append(tail)
            if n % 200 == 0:
                log(f"tedreject: {n}/{len(tasks)} shards, {len(ids)} records")
    finally:
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)
    if len(set(ids)) != len(ids):
        raise RuntimeError("an id is recorded twice in the label run")

    gid_of: dict[str, int] = {}
    gid = np.empty(len(ids), np.int64)
    order_a, order_b = np.empty(len(ids)), np.empty(len(ids))
    for i, item in enumerate(ids):
        k, (a, b) = recording_key(item, local)
        gid[i] = gid_of.setdefault(k, len(gid_of))
        order_a[i], order_b[i] = a, b
    rec_names = list(gid_of)
    del gid_of
    id_rank = np.empty(len(ids), np.int64)
    id_rank[sorted(range(len(ids)), key=ids.__getitem__)] = np.arange(len(ids))
    order = np.lexsort((id_rank, order_b, order_a, gid))
    log(f"tedreject: {len(ids)} records in {len(rec_names)} recordings")

    found: dict[tuple[int, str], set[int]] = defaultdict(set)   # (recording, label set) -> window-gram indices
    for name, d in per_set.items():
        for row, h in d["hits"]:
            found[(int(gid[row]), name)].update(h.tolist())
        nwords = np.concatenate(d["nwords"])
        head = np.concatenate(d["head"])
        tail = np.concatenate(d["tail"])
        hashes = win.hashes
        prev_g, buffer = -1, np.empty(0, np.uint64)
        for row in order.tolist():
            g = int(gid[row])
            if g != prev_g:
                prev_g, buffer = g, np.empty(0, np.uint64)
            n = int(nwords[row])
            if n == 0:
                continue
            k = min(HEAD, n)
            if len(buffer):
                c = _cross(buffer, head[row, :k])
                if len(c) and len(hashes):
                    idx = np.searchsorted(hashes, c)
                    idx[idx >= len(hashes)] = 0
                    ok = hashes[idx] == c
                    if ok.any():
                        found[(g, name)].update(idx[ok].tolist())
            buffer = tail[row, :k] if n >= HEAD else np.concatenate([buffer, head[row, :k]])[-HEAD:]
        del nwords, head, tail
    log(f"tedreject: {len(found)} (recording, label set) pairs share a window 8-gram")

    df = Counter()
    by_rec = defaultdict(set)
    for (g, _), s in found.items():
        by_rec[g] |= s
    for s in by_rec.values():
        df.update(s)
    boiler = {x for x, c in df.items() if c > common.BOILERPLATE_MAX}
    nb = [len([x for x in g.tolist() if x not in boiler]) for g in win.grams]
    checked = [w for w in range(len(win.words)) if win.words[w] >= common.NGRAM and nb[w] > 0]
    need = {w: math.ceil(common.CONTAIN_FRACTION * nb[w]) for w in checked}
    checked_set = set(checked)

    rejected, best = defaultdict(list), []
    for (g, name), s in found.items():
        cnt = Counter()
        for x in s:
            if x in boiler:
                continue
            for w in win.windows_of[x]:
                if w in checked_set:
                    cnt[w] += 1
        if not cnt:
            continue
        w_best = max(cnt, key=lambda w: (cnt[w] / nb[w], -w))
        best.append({"recording": str(rec_names[g]), "label_set": name, "window": w_best, "talk": win.talk[w_best],
                     "matched": cnt[w_best], "of": nb[w_best], "fraction": cnt[w_best] / nb[w_best]})
        for w, c in cnt.items():
            if c >= need[w]:
                rejected[g].append({"label_set": name, "window": w, "talk": win.talk[w], "start": win.start[w],
                                    "matched": c, "of": nb[w]})
    rej_ids = sorted(ids[i] for i in np.flatnonzero(np.isin(gid, np.fromiter(rejected, np.int64)))) if rejected else []
    best.sort(key=lambda b: (-b["fraction"], b["recording"], b["label_set"]))
    by_corpus = Counter(str(r).split(":", 1)[0] for r in rec_names)
    body = {
        "name": NAME, "version": VERSION,
        "rule": {"window_words": WINDOW, "ngram": common.NGRAM, "boilerplate_if_recordings_gt": common.BOILERPLATE_MAX,
                 "contain_fraction": common.CONTAIN_FRACTION, "label_sets": LABEL_SETS,
                 "normalizer": "finetune/parakeet-ternary/evaluate.py normalize",
                 "recording_keys": "pool/tedreject.py recording_key (module docstring)"},
        "inputs": {"label_run": "/".join(label_run.parts[-3:]),
                   "manifest_sha256": common.sha256_file(label_run / "manifest.json"),
                   "shards": len(shards), "complete": max_shards is None},
        "windows": {"talks": win.talks, "windows": len(win.words), "checked": len(checked),
                    "skipped_short": sum(1 for w in win.words if w < common.NGRAM),
                    "skipped_boilerplate_only": sum(1 for w in range(len(win.words))
                                                    if win.words[w] >= common.NGRAM and nb[w] == 0),
                    "window_8grams": int(len(win.hashes)), "boilerplate_8grams": len(boiler)},
        "counts": {"records": len(ids), "recordings": int(len(rec_names)), "recordings_by_corpus": dict(sorted(by_corpus.items())),
                   "recording_label_sets_sharing_a_window_8gram": len(found),
                   "recordings_sharing_a_window_8gram": len(by_rec),
                   "rejected_recordings": len(rejected), "rejected_ids": len(rej_ids)},
        "rejected": [{"recording": str(rec_names[g]), "windows": v} for g, v in sorted(rejected.items(),
                                                                                        key=lambda kv: str(rec_names[kv[0]]))],
        "closest": best[:25]}
    return body, rej_ids


def write(directory: Path, body: dict, rejected_ids: list[str], talk_sha256: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    data = "".join(i + "\n" for i in rejected_ids).encode()
    common.atomic_write_bytes(directory / "rejected_ids.txt", data)
    body = {**body, "created": common.now(), "talk_text_sha256": talk_sha256,
            "ids_sha256": hashlib.sha256(data).hexdigest()}
    common.atomic_write_json(directory / "reject.json", body)
    return body


def load(directory: Path | None = None) -> tuple[frozenset[str], dict]:
    """(rejected ids, identity {name, version, ids_sha256, reject_sha256, rejected_ids}); refuses a damaged list."""
    directory = Path(directory) if directory is not None else out_dir()
    body = json.loads((directory / "reject.json").read_text())
    data = (directory / "rejected_ids.txt").read_bytes()
    if hashlib.sha256(data).hexdigest() != body["ids_sha256"] or body.get("name") != NAME:
        raise RuntimeError(f"{directory}: rejected_ids.txt does not match reject.json")
    if not body["inputs"].get("complete"):
        raise RuntimeError(f"{directory}: built from a partial label run (a smoke); not usable for training")
    ids = frozenset(line for line in data.decode().splitlines() if line)
    if len(ids) != body["counts"]["rejected_ids"]:
        raise RuntimeError(f"{directory}: {len(ids)} ids, reject.json says {body['counts']['rejected_ids']}")
    identity = {"name": NAME, "version": body["version"], "ids_sha256": body["ids_sha256"],
                "reject_sha256": common.sha256_file(directory / "reject.json"), "rejected_ids": len(ids)}
    return ids, identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pool.tedreject", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--labels-run", default="main-f3")
    b.add_argument("--teacher", default="granite-speech-5.0-2b")
    b.add_argument("--workers", type=int, default=6)
    b.add_argument("--max-shards", type=int, default=None, help="smoke: the first N shards, written under smoke/")
    args = parser.parse_args(argv)
    common.require_mount()
    from corpora import registry
    root = common.root("RELABEL_ROOT")
    talk_path = root / registry.source(ALIAS)["local"]["text_parquet"]
    import pyarrow.parquet as pq
    talks = [t or "" for t in pq.read_table(talk_path, columns=["text"]).column("text").to_pylist()]
    import corpora
    local = {}
    st = corpora.status("med-osce")
    if st.get("status") == "built":
        local = local_recordings({"med-osce": Path(st["manifest"])})
    label_run = root / "labels" / args.teacher / args.labels_run
    body, ids = build(label_run, talks, common.normalize, local, args.workers, args.max_shards)
    target = out_dir() if args.max_shards is None else root / "smoke" / "ted-reject"
    body = write(target, body, ids, common.sha256_file(talk_path))
    print(json.dumps({"out": str(target), "counts": body["counts"], "windows": body["windows"],
                      "ids_sha256": body["ids_sha256"], "closest": body["closest"][:5]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
