"""Held-out removal and the text rules of the read-speech source (b0prime plan "Extra sources", review p3).

Sentences are compared after the evaluation normalizer (finetune/parakeet-ternary/evaluate.py `normalize`, stripped),
as the global denylist does; words are its whitespace split. The normalizer deletes bracketed text, while the
training label (filters version 3, labeling/filters.py map_text) turns brackets into spaces and keeps their words, so
every text is compared in both forms (sentence_keys): the normalized sentence and the normalized filters version 3
text. A rule that matches either form applies (review r1).

1. Held-out sets (build_heldout), from the ESB test split of the same corpus (Kea test reporting) and the release's
   own test and dev splits (test.tsv, dev.tsv; dev is reserved for the dev sample):
   - clips: every held-out audio path (the ESB rows' audio.path, the test and dev rows' `path`);
   - sentences: both normalized forms of every held-out sentence;
   - speakers: the union of ESB's `id` values, the test and dev `client_id`s, and the `client_id` that any of the
     release's six tsvs (validated, other, invalidated, train, dev, test) records for a held-out clip path. A
     clip's speaker id can change between releases, so the path join catches a held-out speaker whose ESB id
     differs from this release's.
   A validated clip is dropped by the first rule that matches, in the order speaker, clip, sentence (REASONS).
2. The global denylist (labeling plan Part B), on every clip left (each clip is its own recording):
   - identity: a held-out set row naming a recording of this source (`identity` {alias: clip id}) drops that clip;
   - containment: a held-out utterance of >= 8 words whose distinct non-boilerplate 8-grams occur at least half
     (ceil(0.5 x n)) in either form of the clip's sentence drops the clip. The held-out sets are exactly the ones recorded in
     denylist.json (paths and SHA-256 verified), and the boilerplate 8-grams are the denylist's own (8-grams in
     more than 100 of its training recordings), reproduced by reproduce_boilerplate and verified against the
     counts denylist.json records. This source's clips do not add to the boilerplate counts.
3. Talk text (registry alias ted-test-text, the `text` column only): a clip is dropped if either form of its sentence
   shares any word 8-gram with any talk (all 8-grams of each talk's normalized text). A one-sentence clip is too short for
   the containment rule's 50% test, so this rule is stricter.

Untrusted input: paths read from the tsvs must match SAFE_PATH (a bare file name ending in .mp3) before they are
joined to the clips directory; nothing from the release is executed.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np

from cv27 import layout

REASONS = ("speaker", "clip", "sentence")
SAFE_PATH = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.mp3")
csv.field_size_limit(1 << 30)


class Normalizer:
    """The evaluation normalizer, cached per distinct text."""

    def __init__(self, fn: Callable[[str], str] | None = None):
        if fn is None:
            import evaluate  # finetune/parakeet-ternary
            fn = evaluate.normalize
        self.fn, self.cache = fn, {}

    def __call__(self, text: str | None) -> str:
        text = text or ""
        out = self.cache.get(text)
        if out is None:
            out = self.cache[text] = self.fn(text).strip()
        return out


def training_text(sentence: str | None) -> str:
    """The filters version 3 text of a sentence (the stored training label, before tokenizer coverage)."""
    from labeling import filters
    return filters.map_text(sentence or "", version=layout.FILTERS_VERSION)


def sentence_keys(sentence: str | None, normalize: Callable[[str], str]) -> tuple[str, ...]:
    """The distinct normalized forms of a sentence: of the sentence as written, then of its filters v3 text."""
    a, b = normalize(sentence or ""), normalize(training_text(sentence))
    return (a,) if a == b else (a, b)


def clip_stem(path: str) -> str:
    """The clip's file stem; refuses anything but a bare file name ending in .mp3 (untrusted input)."""
    if not SAFE_PATH.fullmatch(path or ""):
        raise ValueError(f"unexpected clip path {path!r}")
    return path[:-len(".mp3")]


def clip_id(path: str) -> str:
    return f"{layout.ALIAS}:{clip_stem(path)}"


def read_tsv(path: Path, columns: Sequence[str]) -> Iterator[dict]:
    """Rows of one release tsv (tab-separated, no quoting), with `columns` checked present."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"{path}: missing columns {missing}")
        for row in reader:
            yield row


def read_durations(path: Path) -> dict[str, int]:
    """clip_durations.tsv: clip path -> duration in ms."""
    out = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        if len(header) != 2:
            raise RuntimeError(f"{path}: unexpected header {header}")
        for clip, ms in reader:
            out[clip] = int(ms)
    return out


def esb_rows(files: Sequence[Path]) -> Iterator[tuple[str, str, str]]:
    """(id, text, audio path) of the ESB test split's parquet files (no audio bytes are read)."""
    import pyarrow.parquet as pq
    for path in files:
        t = pq.read_table(path, columns=["id", "text", "audio.path"])
        for uid, text, apath in zip(t.column("id").to_pylist(), t.column("text").to_pylist(),
                                    t.column("path").to_pylist()):
            yield str(uid), text or "", str(apath)


@dataclasses.dataclass
class HeldOut:
    speakers: frozenset[str]
    paths: frozenset[str]
    sentences: frozenset[str]
    esb_paths: frozenset[str]
    esb_speakers: frozenset[str]        # ESB ids and this release's client_id of every ESB path (both id forms)
    esb_sentences: frozenset[str]       # both normalized forms (sentence_keys)
    counts: dict

    def reason(self, client_id: str, path: str, keys: Sequence[str]) -> str | None:
        """The first held-out rule a clip matches; `keys` are its sentence's normalized forms (sentence_keys)."""
        if client_id in self.speakers:
            return "speaker"
        if path in self.paths:
            return "clip"
        if any(k in self.sentences for k in keys):
            return "sentence"
        return None


def build_heldout(esb: Iterable[tuple[str, str, str]], tsv_rows: Callable[[str], Iterable[dict]],
                  normalize: Callable[[str], str]) -> HeldOut:
    """The held-out sets (module docstring, rule 1). tsv_rows(name) yields the rows of one of layout.TSVS."""
    esb_spk, esb_path, esb_sent = set(), set(), set()
    n_esb = 0
    for uid, text, path in esb:
        n_esb += 1
        esb_spk.add(uid)
        esb_path.add(path)
        esb_sent.update(sentence_keys(text, normalize))
    speakers, paths, sentences = set(esb_spk), set(esb_path), set(esb_sent)
    counts = {"esb_test": n_esb, "esb_test_paths": len(esb_path)}
    for name in layout.HELD_SPLITS:
        n = 0
        for r in tsv_rows(name):
            n += 1
            speakers.add(r["client_id"])
            paths.add(r["path"])
            sentences.update(sentence_keys(r["sentence"], normalize))
        counts[name] = n
    ids_of_path: dict[str, set[str]] = defaultdict(set)
    for name in layout.TSVS:
        for r in tsv_rows(name):
            if r["path"] in paths:
                ids_of_path[r["path"]].add(r["client_id"])
    joined = set().union(*ids_of_path.values()) if ids_of_path else set()
    esb_current = set().union(*(ids_of_path.get(p, set()) for p in esb_path)) if esb_path else set()
    counts.update({
        "held_paths_found_in_release": len(ids_of_path),
        "esb_paths_found_in_release": sum(1 for p in esb_path if p in ids_of_path),
        "esb_paths_with_changed_id": sum(1 for p in esb_path if ids_of_path.get(p) and not (ids_of_path[p] & esb_spk)),
        "speaker_ids_added_by_path_join": len(joined - speakers),
    })
    speakers |= joined
    counts.update({"held_speakers": len(speakers), "held_paths": len(paths), "held_sentences": len(sentences)})
    return HeldOut(frozenset(speakers), frozenset(paths), frozenset(sentences), frozenset(esb_path),
                   frozenset(esb_spk | esb_current), frozenset(esb_sent), counts)


# --- the talk-text rule ------------------------------------------------------------------------------------------

def talk_grams(texts: Iterable[str], normalize: Callable[[str], str], n: int = layout.NGRAM) -> frozenset[tuple]:
    """Every word n-gram of every talk's normalized text."""
    out = set()
    for text in texts:
        w = normalize(text).split()
        out.update(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return frozenset(out)


def shares_gram(words: Sequence[str], grams: frozenset[tuple], n: int = layout.NGRAM) -> bool:
    return any(tuple(words[i:i + n]) in grams for i in range(len(words) - n + 1))


def read_talk_texts(path: Path) -> list[str]:
    import pyarrow.parquet as pq
    return [t or "" for t in pq.read_table(path, columns=["text"]).column("text").to_pylist()]


# --- the global denylist ------------------------------------------------------------------------------------------

def denylist_sets(denylist: dict) -> list[tuple[str, Path, str]]:
    """(name, manifest path, recorded SHA-256) of every held-out set denylist.json records."""
    return [(h["name"], Path(h["path"]), h["sha256"]) for h in denylist["heldout"]]


def load_denylist_index(denylist: dict):
    """The held-out index of exactly denylist.json's sets; refuses a manifest whose SHA-256 changed."""
    from pool import common
    from pool import heldout as ho
    sets = denylist_sets(denylist)
    for name, path, sha in sets:
        got = common.sha256_file(path)
        if got != sha:
            raise RuntimeError(f"held-out set {name} ({path}) changed since the denylist build: {got} != {sha}")
    return ho.Index(ho.load_sets([(name, path) for name, path, _ in sets]))


class _NoTerms:
    """The scan's term matchers are not needed to reproduce 8-gram hits (picklable, module level)."""

    def count(self, words) -> int:
        return 0


def reproduce_boilerplate(denylist: dict, index, caches: dict[str, Path], corpora: dict[str, Path],
                          workers: int) -> tuple[np.ndarray, dict]:
    """(sorted boilerplate 8-gram hashes, verification) from the pool scan over the denylist's training inputs,
    exactly as pool/denylist.py computed them; raises unless every recorded count is reproduced."""
    from pool import denylist as dl
    from pool import scan
    for alias, manifest in corpora.items():
        rec = denylist["training"][alias]
        from pool import common
        if common.sha256_file(manifest) != rec["manifest_sha256"]:
            raise RuntimeError(f"corpus {alias}: manifest changed since the denylist build")
    res = scan.scan(caches, corpora, index.hashes, {"medical": _NoTerms(), "technical": _NoTerms()}, workers)
    nh = len(index.hashes)
    h_all = (res.pairs & np.uint64(0xFFFFFFFF)).astype(np.int64)
    df = np.bincount(h_all, minlength=nh)
    boiler = df > dl.common.BOILERPLATE_MAX
    nb = np.array([int((~boiler[g]).sum()) for g in index.utt_grams], dtype=np.int64)
    c_gid = dl._containment(res, index, boiler, nb)[0]
    counts = denylist["counts"]
    got = {"boilerplate_8grams": int(boiler.sum()), "heldout_8grams": int(nh),
           "training_recordings_scanned": len(res.keys), "contained_pairs": int(len(c_gid))}
    want = {k: counts[k] for k in got}
    listing = {s: {k: res.sources[s].get(k) for k in ("files_sha256", "listing_sha256")}
               for s in caches}
    want_listing = {s: {k: denylist["training"][s].get(k) for k in ("files_sha256", "listing_sha256")}
                    for s in caches}
    if got != want or listing != want_listing:
        raise RuntimeError(f"the scan does not reproduce denylist.json: got {got} {listing}, "
                           f"recorded {want} {want_listing}")
    return np.sort(index.hashes[boiler]), {"reproduced": got, "training": listing}


def save_boilerplate(path: Path, hashes: np.ndarray, info: dict, denylist_sha256: str) -> None:
    from pool import common
    data = np.ascontiguousarray(hashes, dtype="<u8").tobytes()
    common.atomic_write_bytes(path.with_suffix(".u64"), data)
    common.atomic_write_json(path, {"denylist_sha256": denylist_sha256, "count": int(len(hashes)),
                                    "u64_sha256": hashlib.sha256(data).hexdigest(), **info})


def load_boilerplate(path: Path, denylist_sha256: str) -> np.ndarray:
    meta = json.loads(Path(path).read_text())
    if meta["denylist_sha256"] != denylist_sha256:
        raise RuntimeError(f"{path}: reproduced for another denylist.json ({meta['denylist_sha256']})")
    data = path.with_suffix(".u64").read_bytes()
    if hashlib.sha256(data).hexdigest() != meta["u64_sha256"] or len(data) != 8 * meta["count"]:
        raise RuntimeError(f"{path}: boilerplate hashes damaged")
    return np.frombuffer(data, dtype="<u8").astype(np.uint64)


class Containment:
    """The global denylist's identity and containment rules over single-clip recordings (module docstring)."""

    def __init__(self, index, boilerplate: np.ndarray):
        from pool import common
        self.index = index
        self.boiler = np.isin(index.hashes, np.asarray(boilerplate, dtype=np.uint64))
        self.nb = np.array([int((~self.boiler[g]).sum()) for g in index.utt_grams], dtype=np.int64) \
            if len(index) else np.zeros(0, np.int64)
        self.hasher = common.WordHasher()
        self.identity = {rec for s in index.sets for src, rec in s.identity if src == layout.ALIAS}

    def identity_hit(self, cid: str) -> bool:
        return cid in self.identity

    def contained(self, word_lists: Sequence[Sequence[str]]) -> dict[int, list[dict]]:
        """{position in word_lists: [{set, id, matched, of}, ...]} for every contained clip."""
        from types import SimpleNamespace

        from pool import common
        from pool import denylist as dl
        hashes = self.index.hashes
        chunks = []
        for gid, words in enumerate(word_lists):
            if len(words) < common.NGRAM or not len(hashes):
                continue
            g = np.unique(common.ngram_hashes(self.hasher(list(words))))
            idx = np.searchsorted(hashes, g)
            idx[idx >= len(hashes)] = 0
            ok = hashes[idx] == g
            if ok.any():
                chunks.append((np.uint64(gid) << np.uint64(32)) | idx[ok].astype(np.uint64))
        pairs = np.unique(np.concatenate(chunks)) if chunks else np.empty(0, np.uint64)
        c_gid, c_utt, c_cnt, c_of = dl._containment(SimpleNamespace(pairs=pairs), self.index, self.boiler, self.nb)
        out: dict[int, list[dict]] = defaultdict(list)
        for g, u, cnt, of in zip(c_gid.tolist(), c_utt.tolist(), c_cnt.tolist(), c_of.tolist()):
            out[int(g)].append({"set": self.index.sets[int(self.index.utt_set[u])].name,
                                "id": self.index.utt_id[u], "matched": int(cnt), "of": int(of)})
        return dict(out)
