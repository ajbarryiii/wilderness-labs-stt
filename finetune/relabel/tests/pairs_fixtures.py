"""Fixtures for the confusable-pair contract tests (synthetic plan v5; no tests here).

A small term split with chosen sides (v2), its pronunciation lexicon, label word counts and `full` clip counts, and
helpers that write them under a temporary $RELABEL_ROOT (synthetic_fixtures.roots). Nothing here touches the
production roots.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from synthetic import layout, pronounce, vocab
from synthetic.textgen import write_jsonl
from synthetic_fixtures import normalize

LISTS = {
    "chemistry": ["Schrodinger equation", "thermodynamic equilibrium", "laws of thermodynamics", "alkane", "alkene",
                  "enthalpy", "titration", "catalyst"],
    "mathematics": ["Cantor-Bernstein-Schroeder theorem", "Gauss law", "eigenvalue", "Fourier transform",
                    "Markov chain"],
    "finance": ["ebitda", "adjusted ebitda", "ebit", "roe", "roi", "opex", "capex", "cap ex", "free cash flow",
                "yield curve"],
    "software_engineering": ["gpu", "gpus", "tpu", "data set", "dataset", "hash table", "race condition",
                             "event loop"],
}
LEXICON = {"finance": "ebit\tee bit\nebitda\tee bit dah\nroe\nroi\nopex\top ex\ncapex\tcap ex\n",
           "software_engineering": "gpu\ntpu\n", "chemistry": "", "mathematics": ""}
# v2 sides by member phrase (a group takes its member's side); every other group is targeted
HELDOUT = ("ebitda", "schrodinger equation", "roe", "opex", "alkane", "alkene", "laws of thermodynamics")
# `full` clip counts by member phrase (a group takes its member's count); every other targeted group has 150
CLIPS = {"ebit": 214, "cantor bernstein schroeder theorem": 183, "capex": 372, "gpu": 300, "tpu": 120, "roi": 90}


def make_split(heldout=HELDOUT) -> vocab.Split:
    """The fixture split (v2): LISTS built by the split's own rules, sides chosen by group (HELDOUT)."""
    n = normalize()
    data = vocab.build_split(LISTS, n, vocab.medical_matcher(n))
    groups = {e["phrase"]: e["group"] for e in data["entries"]}
    held = {groups[p] for p in heldout}
    for e in data["entries"]:
        e["side"] = "heldout" if e["group"] in held else "targeted"
    data["lists"] = vocab.lists_record()             # the committed lists: Split.load accepts the fixture
    return vocab.Split(data, n)


def group_of(split: vocab.Split, phrase: str) -> str:
    return split.by_phrase[tuple(phrase.split())].group


def lexicon() -> pronounce.Lexicon:
    return pronounce.Lexicon.parse(LEXICON)


def clip_counts(split: vocab.Split, clips=CLIPS) -> dict[str, int]:
    out = {e.group: 150 for e in split.entries if e.side == "targeted"}
    for phrase, n in clips.items():
        out[group_of(split, phrase)] = n
    return out


def full_clips(split: vocab.Split, clips=CLIPS) -> list[dict]:
    """A `full` clips.jsonl whose per-group clip counts are clip_counts(split): one term per clip."""
    rows = []
    for g, n in sorted(clip_counts(split, clips).items()):
        phrase = min(" ".join(e.phrase) for e in split.entries if e.group == g)
        for k in range(n):
            rows.append({"id": f"full-{g}-{k:04d}", "label": f"We talked about the {phrase} today.", "terms": [phrase],
                         "duration": 2.0})
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_world(split: vocab.Split, words: dict[str, int] | None = None, clips=CLIPS) -> dict[str, Path]:
    """Under the current (temporary) $RELABEL_ROOT: the split as split.json, the label word counts as
    synthetic/v5/words.json, and the full run's clips.jsonl / clips.meta.json (hygiene under that split)."""
    paths = {"split": layout.split_path(), "words": layout.base() / "v5" / "words.json",
             "clips": layout.run_dir("full") / "clips.jsonl"}
    paths["split"].parent.mkdir(parents=True, exist_ok=True)
    paths["split"].write_text(json.dumps(split.data, indent=1) + "\n")
    paths["words"].parent.mkdir(parents=True, exist_ok=True)
    paths["words"].write_text(json.dumps({"format": 1, "labels": {"run": "fixture"}, "counts": words or {}}) + "\n")
    write_jsonl(paths["clips"], full_clips(split, clips))
    meta = {"clips": len(paths["clips"].read_text().splitlines()), "clips_sha256": sha256(paths["clips"]),
            "split_sha256": sha256(paths["split"]), "hygiene_split_sha256": sha256(paths["split"]),
            "provisional": False}
    (paths["clips"].parent / "clips.meta.json").write_text(json.dumps(meta) + "\n")
    return paths


def curation_text(candidates: dict, sha: str, decisions: dict[tuple[str, str], tuple[str, str]] | None = None,
                  default=("accept", "a shared onset with a short difference")) -> str:
    """A pairs.tsv for `candidates` (a candidate file's object, `sha` its file's SHA-256): every candidate accepted with
    `default`, except the ones in `decisions` ({(term_a, term_b): (decision, reason)})."""
    decisions = decisions or {}
    lines = ["# fixture curation", f"# candidates_sha256: {sha}",
             "# term_a\tterm_b\trule\tdecision\treason"]
    for c in candidates["candidates"]:
        a, b = (m["term"] for m in c["members"])
        decision, reason = decisions.get((a, b), decisions.get((b, a), default))
        lines.append("\t".join((a, b, c["rule"], decision, reason)))
    return "\n".join(lines) + "\n"


def find(candidates: dict, a: str, b: str) -> dict | None:
    """The candidate whose members hold the terms a and b (either order), or None."""
    for c in candidates["candidates"]:
        terms = [set(m["terms"]) for m in c["members"]]
        if (a in terms[0] and b in terms[1]) or (b in terms[0] and a in terms[1]):
            return c
    return None


def deep(obj):
    return copy.deepcopy(obj)


def make_v3(decisions=None, heldout=HELDOUT):
    """(split v2, split v3) built in memory from the fixture's candidates, every candidate accepted except `decisions`
    ({(term_a, term_b): (decision, reason)})."""
    from synthetic import confusable, pairs
    s = make_split(heldout)
    cands = confusable.detect(s, lexicon(), {}, clip_counts(s))
    sha = "ab" * 32
    cur = pairs.Curation.parse(curation_text(cands, sha, decisions), cands, sha, s)
    v3, _ = pairs.build_v3(s, cur, base={"file": "split.v2.json", "sha256": "cd" * 32})
    return s, vocab.Split(v3, normalize())
