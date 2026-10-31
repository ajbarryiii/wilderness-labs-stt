"""Confusable term pairs after curation (synthetic plan v5, "Confusable pairs"): the curation file, split v3, run
`pairs` (contrast sentences, balance, rounds) and earlier clip pools under split v3. The detector is confusable.py.

    ./python -m synthetic pairs-split                 split v3 from terms/synthetic/pairs.tsv (CPU)
    ./python -m synthetic pairs-check-full            full's clips under v3's held-out guard (CPU)
    ./python -m synthetic pairs-coverage              coverage under v3 (terms/coverage.v3.json; reads references)
    ./python -m synthetic pairs-prompts --round K     round K's prompts from the remaining need (CPU)
    ./python -m synthetic pairs-hygiene --round K     round K's hygiene under v3 and its selection (CPU)
    ./python -m synthetic pairs-finalize              run pairs: the rounds' clips joined, the report (CPU)
(textgen, tts-plan, tts, roundtrip and finalize run on the rounds as on any run: --run pairs-r1 .. pairs-r3.)

Curation file (terms/synthetic/pairs.tsv, committed, curated by hand by a blind curator from the candidate file
$RELABEL_ROOT/synthetic/v5/candidates.json alone). Blank lines and '#' comments are ignored, except one required line
"# candidates_sha256: <SHA-256 of the candidate file>". Every other line is one row of five TAB-separated fields,
COLUMNS: term_a, term_b (listed terms, as written in the term lists, naming the candidate's two groups in either
order), rule (the candidate's rule), decision ("accept" or "reject") and reason (free text, not empty). Every
candidate has exactly one row and nothing else has a row. The always-accepted pairs (FORCED: EBITDA/EBIT and
"Schrodinger equation"/"Cantor-Bernstein-Schroeder theorem", AJ's request) must be accepted. The dev-informed seed
(SEEDS: OPEX paired with CAPEX) is not a candidate and has no row: it is a plan constant, flagged as dev-informed.

Split v3 (build_v3, freeze_v3): split v2 (split.v2.json, written once as a byte copy of split.json and never changed
afterwards) with every held-out group of an accepted held-out-targeted pair, and the seed's held-out group, made
targeted. Groups move whole; nothing moves from targeted to held out; no term is added. So held-out(v3) is a subset of
held-out(v2), and full's clips stay clean. split.v3.json also lists every accepted pair and the seed ("pairs": id,
terms, groups, domains, sides under v2 and v3, rule, reason, the detector's token pairs ("hits"), forced, seed, and
"trained": both members targeted under v3), the moved groups with their reasons, and the dev-informed groups;
split-changes-v3.json lists each move with its pair or seed reason. Both are written once (a later build that differs
is refused).

Provenance (recheck, called by mixing.ClipPool.load): a pool whose hygiene ran under a split's base (v2 for v3) is
accepted under the narrower split if every clip's label passes its held-out guard (text only); a failing clip refuses
the pool. check_full records this for run full.

Run pairs. Spec: the trained pairs (both members targeted under v3) and, for every targeted group in an accepted pair,
its target T(g) = min(max(full clips of its partners, FLOOR), CAP) (FLOOR, CAP: the median and 90th-percentile full
clips of v2's targeted groups). State: counts per group over full and the run's kept clips (a clip counts once for
every group with a term occurrence in its label, so a contrast clip counts for both members), and the kept contrast
clips per pair. A round K (at most ROUNDS) issues ceil(FACTOR x need) prompts for each need left by the earlier rounds'
kept clips:
- contrast prompts per trained pair, need CONTRAST minus its kept contrast clips: both members' terms (the pair's
  terms), "as two different things" (textgen.prompt_text), registers taking turns per domain;
- single-term prompts per target group, need T(g) minus its count: one of the group's terms (taking turns over its
  terms) plus 0-2 other targeted terms of that term's domain, as in v1 (textgen.TERMS_PER_PROMPT, a seeded cycle over
  the domain's targeted terms, different groups).
Round hygiene is v1's (hygiene.run) under split v3, with full's kept sentences and the earlier rounds' selected
sentences as earlier kept sentences for the duplicate checks (prior_sentences). Selection (select) then walks the
hygiene-kept sentences, contrast ones first, each kind in a seeded order: a contrast sentence is kept while its pair
has fewer than CONTRAST kept (it must hold both members); a single-term sentence is kept while its group's count is
below T(g) (it must hold the group). Counts start from the state and grow with every kept sentence. Only selected
sentences are rendered (sentences.jsonl); their round trip is v1's, under split v3. Existing clips and contrast clips
are never dropped to meet T(g). finalize_run joins the rounds' kept clips into runs/pairs/clips.jsonl and reports the
shortfalls per pair and group; filters are never relaxed to meet a target.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import terms as committed_terms
from synthetic import layout, vocab

PAIRS_FILE = committed_terms.HERE / "synthetic" / "pairs.tsv"
COLUMNS = ("term_a", "term_b", "rule", "decision", "reason")
HEADER = ("# Confusable term pairs (synthetic plan v5, \"Curation and seeds\"): one row per detector candidate,",
          "# curated blind from the candidate file named below. Format: synthetic/pairs.py.")
DECISIONS = ("accept", "reject")
FORCED = ((("ebitda",), ("ebit",)), (("schrodinger", "equation"), ("cantor", "bernstein", "schroeder", "theorem")))
FORCED_REASON = "always accepted (AJ's request, 2026-10-30)"
SEEDS = ({"terms": (("opex",), ("capex",)),
          "reason": "dev-informed seed: B0' lost OPEX; the detector does not pair 'op ex' with 'cap ex' (edit distance "
                    "2 on 5-letter keys)"},)
FLOOR, CAP = 139, 264          # median and 90th percentile of full clips per targeted group (split v2)
CONTRAST = 40                  # kept contrast sentences per trained pair
FACTOR = 2.5                   # prompts per unit of remaining need
ROUNDS = 3
_SHA_LINE = re.compile(r"#\s*candidates_sha256:\s*([0-9a-f]{64})\s*")


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _json(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


# ----------------------------------------------------------------------------- curation file

class Curation:
    """The parsed curation file: rows [{id, terms, groups, rule, decision, reason, forced, hits}] in candidate order."""

    def __init__(self, rows: list[dict], candidates_sha256: str):
        self.rows, self.candidates_sha256 = rows, candidates_sha256

    @property
    def accepted(self) -> list[dict]:
        return [r for r in self.rows if r["decision"] == "accept"]

    @classmethod
    def parse(cls, text: str, candidates: dict, sha256: str, split: vocab.Split) -> "Curation":
        """From the file's text, the candidate file's object and SHA-256, and split v2 (ValueError on any violation of
        the format in the module docstring)."""
        group_of = {e.term: e.group for e in split.entries}
        by_groups = {frozenset(m["group"] for m in c["members"]): c for c in candidates["candidates"]}
        shas, rows, seen = [], {}, set()
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            if line.startswith("#"):
                m = _SHA_LINE.fullmatch(line.strip())
                if m:
                    shas.append(m.group(1))
                continue
            fields = [f.strip() for f in line.split("\t")]
            if len(fields) != len(COLUMNS) or not all(fields):
                raise ValueError(f"line {n}: expected {len(COLUMNS)} non-empty TAB-separated fields {COLUMNS}")
            a, b, rule, decision, reason = fields
            if decision not in DECISIONS:
                raise ValueError(f"line {n}: decision {decision!r} is not one of {DECISIONS}")
            for t in (a, b):
                if t not in group_of:
                    raise ValueError(f"line {n}: {t!r} is not a listed term")
            key = frozenset((group_of[a], group_of[b]))
            c = by_groups.get(key) if len(key) == 2 else None
            if c is None:
                raise ValueError(f"line {n}: {a!r} / {b!r} is not a candidate pair")
            if key in seen:
                raise ValueError(f"line {n}: a second row for candidate {c['id']}")
            seen.add(key)
            if rule != c["rule"]:
                raise ValueError(f"line {n}: rule {rule!r}, but candidate {c['id']}'s rule is {c['rule']!r}")
            if c["forced"] and decision != "accept":
                raise ValueError(f"line {n}: {a!r} / {b!r} is always accepted (AJ's request)")
            groups = [group_of[a], group_of[b]]
            order = [m["group"] for m in c["members"]]
            flip = groups != order
            hits = [{"tokens": h["tokens"][::-1] if flip else list(h["tokens"]), "rule": h["rule"]} for h in c["hits"]]
            rows[c["id"]] = {"id": c["id"], "terms": [a, b], "groups": groups, "rule": rule, "decision": decision,
                             "reason": reason, "forced": bool(c["forced"]), "hits": hits}
        if len(shas) != 1:
            raise ValueError("the file needs exactly one '# candidates_sha256: <hex>' line")
        if shas[0] != sha256:
            raise ValueError(f"curated from candidate file {shas[0][:12]}, not {sha256[:12]}")
        missing = [c["id"] for c in candidates["candidates"] if c["id"] not in rows]
        if missing:
            raise ValueError(f"{len(missing)} candidates have no row (first {missing[0]})")
        return cls([rows[c["id"]] for c in candidates["candidates"]], sha256)

    @classmethod
    def load(cls, path: Path = PAIRS_FILE, candidates_path: Path | None = None,
             split: vocab.Split | None = None) -> "Curation":
        from synthetic import confusable
        candidates_path = Path(candidates_path or confusable.out_dir() / "candidates.json")
        data = candidates_path.read_bytes()
        v2 = layout.split_version_path(2)
        split = split or vocab.Split.load(v2 if v2.exists() else layout.split_path())
        return cls.parse(Path(path).read_text(encoding="utf-8"), json.loads(data), hashlib.sha256(data).hexdigest(),
                         split)


# ----------------------------------------------------------------------------- split v3

def _seed_pairs(split: vocab.Split) -> list[dict]:
    out = []
    for k, s in enumerate(SEEDS, 1):
        es = [split.by_phrase.get(p) for p in s["terms"]]
        if not all(es):
            raise RuntimeError(f"seed {s['terms']}: a term is not listed")
        out.append({"id": f"seed{k}", "terms": [e.term for e in es], "groups": [e.group for e in es],
                    "rule": "seed", "reason": s["reason"], "forced": False, "seed": True,
                    "hits": [{"tokens": [" ".join(p).replace(" ", "") for p in s["terms"]], "rule": "seed"}]})
    return out


def build_v3(v2: vocab.Split, curation: Curation, base: dict, curation_record: dict | None = None) -> tuple[dict, dict]:
    """(split v3's object, its changes object) from split v2 and the curation (module docstring). base: {"file",
    "sha256"} of split v2."""
    side = {e.group: e.side for e in v2.entries}
    domain = {e.group: e.domain for e in v2.entries}
    terms_of = defaultdict(list)
    for e in v2.entries:
        terms_of[e.group].append(e.term)
    accepted = [r | {"seed": False} for r in curation.accepted] + _seed_pairs(v2)
    reasons: dict[str, list[dict]] = defaultdict(list)
    dev_informed = set()
    for p in accepted:
        sides = [side[g] for g in p["groups"]]
        if p["seed"]:
            for g, s in zip(p["groups"], sides):
                if s == "heldout":
                    reasons[g].append({"kind": "seed", "pair": p["id"], "terms": p["terms"], "reason": p["reason"]})
                    dev_informed.add(g)
        elif sorted(sides) == ["heldout", "targeted"]:
            g = p["groups"][sides.index("heldout")]
            reasons[g].append({"kind": "pair", "pair": p["id"], "terms": p["terms"], "rule": p["rule"],
                               "reason": p["reason"]})
    data = json.loads(json.dumps(v2.data))
    for e in data["entries"]:
        if e["group"] in reasons:
            e["side"] = "targeted"
    side3 = {e["group"]: e["side"] for e in data["entries"]}
    listed = []
    for p in accepted:
        listed.append({"id": p["id"], "terms": p["terms"], "groups": p["groups"],
                       "domains": [domain[g] for g in p["groups"]], "sides_v2": [side[g] for g in p["groups"]],
                       "sides_v3": [side3[g] for g in p["groups"]], "rule": p["rule"], "reason": p["reason"],
                       "hits": p["hits"], "forced": p["forced"], "seed": p["seed"],
                       "trained": all(side3[g] == "targeted" for g in p["groups"])})
    moved = [{"group": g, "terms": sorted(terms_of[g]), "domain": domain[g], "reasons": reasons[g],
              "dev_informed": g in dev_informed} for g in sorted(reasons)]
    data["version"] = 3
    data["base"] = dict(base)
    data["rule"] = (v2.data["rule"] + "; version 3 (synthetic plan v5): every held-out group of an accepted "
                    "held-out-targeted pair, and the seed's held-out group, is targeted")
    data["counts"] = {d: {s: sum(1 for e in data["entries"] if e["domain"] == d and e["side"] == s)
                          for s in vocab.SIDES} for d in layout.DOMAINS}
    data["groups"] = {"total": len(side3), "targeted": sum(1 for s in side3.values() if s == "targeted")}
    data["pairs"] = listed
    data["moved"] = moved
    data["dev_informed_groups"] = sorted(dev_informed)
    data["curation"] = dict(curation_record or {}) | {"candidates_sha256": curation.candidates_sha256}
    held2 = {g for g, s in side.items() if s == "heldout"}
    held3 = {g for g, s in side3.items() if s == "heldout"}
    if not held3 <= held2:
        raise AssertionError("held-out(v3) must be a subset of held-out(v2)")
    diff = vocab.changes(v2.data, data)
    changes = {"base": dict(base), "reason": "synthetic plan v5 (confusable pairs): curated held-out-targeted pairs "
                                             "and the dev-informed seed", "moved": moved,
               "changed": diff["changed"], "old_counts": diff["old_counts"], "new_counts": diff["new_counts"],
               "dev_informed_groups": sorted(dev_informed)}
    return data, changes


def freeze_v3(pairs_path: Path = PAIRS_FILE, candidates_path: Path | None = None, check_lists: bool = True) -> dict:
    """Write split.v2.json (once, a byte copy of split.json), split.v3.json and split-changes-v3.json (once; a later
    build that differs is refused). Returns split v3's object."""
    from synthetic import confusable
    current, v2_path, v3_path = layout.split_path(), layout.split_version_path(2), layout.split_version_path(3)
    if not v2_path.exists():
        _write(v2_path, current.read_text(encoding="utf-8"))
    elif v2_path.read_bytes() != current.read_bytes():
        raise RuntimeError(f"{v2_path} differs from {current} (split v2 is the split of record of stages 2c and 3)")
    v2 = vocab.Split.load(v2_path, check_lists=check_lists)
    if v2.data.get("version") != 2:
        raise RuntimeError(f"{v2_path} is not split version 2")
    candidates_path = Path(candidates_path or confusable.out_dir() / "candidates.json")
    cand_bytes = candidates_path.read_bytes()
    curation = Curation.parse(Path(pairs_path).read_text(encoding="utf-8"), json.loads(cand_bytes),
                              hashlib.sha256(cand_bytes).hexdigest(), v2)
    v3, changes = build_v3(v2, curation, {"file": v2_path.name, "sha256": _sha256(v2_path)},
                           {"file": "terms/synthetic/pairs.tsv", "sha256": _sha256(Path(pairs_path))})
    if v3_path.exists():
        old = json.loads(v3_path.read_text())
        if old == v3:
            return old
        raise RuntimeError(f"{v3_path} is frozen and differs from a fresh build (curation, candidates or split v2 "
                           "changed); a new split needs a plan decision")
    _write(v3_path.with_name("split-changes-v3.json"), _json(changes))
    _write(v3_path, _json(v3))
    return v3


def load_v3(check_lists: bool = True) -> vocab.Split:
    return vocab.Split.load(layout.split_version_path(3), check_lists=check_lists)


# ----------------------------------------------------------------------------- provenance under a narrower split

def recheck(clips: Sequence[dict], split_file: Path, hygiene_sha256: str | None) -> None:
    """A pool whose hygiene ran under split_file's base: every clip's label must pass split_file's held-out guard
    (RuntimeError otherwise, and for a split without a base, another base, or a held-out set that is not a subset of
    its base's)."""
    split_file = Path(split_file)
    data = json.loads(split_file.read_text()) if split_file.exists() else {}
    base = data.get("base") or {}
    if not hygiene_sha256 or base.get("sha256") != hygiene_sha256:
        raise RuntimeError(f"the pool's hygiene did not run under the current term split {split_file.name} or its "
                           f"base (split_sha256 {hygiene_sha256}); rerun hygiene and finalize")
    base_path = split_file.with_name(base["file"])
    if _sha256(base_path) != base["sha256"]:
        raise RuntimeError(f"{base_path} is not the base {split_file.name} names")
    held_base = {e["group"] for e in json.loads(base_path.read_text())["entries"] if e["side"] == "heldout"}
    split = vocab.Split(data)
    if not {e.group for e in split.heldout()} <= held_base:
        raise RuntimeError(f"{split_file.name}'s held-out set is not a subset of its base's")
    failing = [c["id"] for c in clips if split.has_heldout(c["label"])]
    if failing:
        raise RuntimeError(f"{len(failing)} clips fail the held-out guard of {split_file.name} (first {failing[0]}); "
                           "the pool is refused")


def check_full(run: str = "full") -> dict:
    """Run full's clips under split v3 (mixing.ClipPool.load), recorded in v5/provenance-<run>.json."""
    from synthetic import mixing
    v3 = layout.split_version_path(3)
    rec = {"run": run, "split": v3.name, "split_sha256": _sha256(v3), "rule": "every clip's label passes split v3's "
           "held-out guard (text only); a failing clip refuses the pool"}
    try:
        pool = mixing.ClipPool.load(run, split_path=v3, verify_audio=False)
    except RuntimeError as error:
        _write(layout.v5_dir() / f"provenance-{run}.json", _json(rec | {"pass": False, "error": str(error)}))
        raise
    meta = json.loads((layout.run_dir(run) / "clips.meta.json").read_text())
    rec |= {"pass": True, "clips": len(pool.clips), "clips_sha256": meta["clips_sha256"],
            "hygiene_split_sha256": meta.get("hygiene_split_sha256")}
    _write(layout.v5_dir() / f"provenance-{run}.json", _json(rec))
    return rec


# ----------------------------------------------------------------------------- run pairs: spec, state, prompts

def groups_in(label: str, split: vocab.Split) -> list[str]:
    """The groups with a term occurrence in a label (the scorer's matcher; sorted)."""
    return sorted({split.by_phrase[tuple(o["phrase"].split())].group for o in split.occurrences(label)})


class Spec:
    """Run pairs' fixed inputs (module docstring): split v3's pairs, the trained pairs, the targets and full's
    clips per group."""

    def __init__(self, split: vocab.Split, full: Mapping[str, int]):
        data = split.data
        if data.get("version") != 3 or "pairs" not in data:
            raise ValueError("run pairs needs split version 3")
        side = {e.group: e.side for e in split.entries}
        self.pairs = data["pairs"]
        self.by_id = {p["id"]: p for p in self.pairs}
        self.trained = [p for p in self.pairs if p["trained"]]
        self.full = Counter({g: int(n) for g, n in full.items() if n})
        partners = defaultdict(set)
        for p in self.pairs:
            ga, gb = p["groups"]
            partners[ga].add(gb)
            partners[gb].add(ga)
        self.targets = {g: min(max([self.full[q] for q in sorted(qs)] + [FLOOR]), CAP)
                        for g, qs in sorted(partners.items()) if side[g] == "targeted"}
        self.dev_informed = set(data.get("dev_informed_groups", []))

    def record(self) -> dict:
        return {"floor": FLOOR, "cap": CAP, "contrast": CONTRAST, "factor": FACTOR, "rounds": ROUNDS,
                "trained_pairs": [p["id"] for p in self.trained], "targets": dict(sorted(self.targets.items())),
                "full_clips": {g: self.full[g] for g in sorted(self.targets)},
                "dev_informed_groups": sorted(self.dev_informed)}


class State:
    """Counts per group over full and the kept clips [{id, label, kind, pair, target}], and contrast clips per pair."""

    def __init__(self, spec: Spec, split: vocab.Split, kept: Iterable[dict] = ()):
        self.counts = Counter(spec.full)
        self.contrast = Counter()
        for c in kept:
            for g in groups_in(c["label"], split):
                self.counts[g] += 1
            if c["kind"] == "contrast":
                self.contrast[c["pair"]] += 1


def _rng(*parts):
    from synthetic.textgen import _rng as rng
    return rng(*parts)


def round_prompts(spec: Spec, split: vocab.Split, state: State, round: int, seed: int = layout.SEED) -> list[dict]:
    """Round `round`'s prompts (module docstring): contrast prompts, then single-term prompts, ids pairs-r<K>-<i>."""
    from synthetic import textgen
    if not 1 <= round <= ROUNDS:
        raise ValueError(f"round {round} outside 1..{ROUNDS}")
    run = f"{layout.PAIRS_RUN}-r{round}"
    by_term = {e.term: e for e in split.entries}
    registers = Counter()
    out = []

    def add(kind, domain, requested, **extra):
        register = layout.REGISTERS[registers[domain] % len(layout.REGISTERS)]
        registers[domain] += 1
        out.append({"id": f"{run}-{len(out):06d}", "index": len(out), "kind": kind, "domain": domain,
                    "register": register, "requested": list(requested)} | extra)
    for p in spec.trained:
        for e in (by_term[t] for t in p["terms"]):
            if e.side != "targeted":
                raise RuntimeError(f"pair {p['id']}: {e.term!r} is not targeted under v3")
        need = max(0, CONTRAST - state.contrast[p["id"]])
        for _ in range(math.ceil(FACTOR * need)):
            add("contrast", by_term[p["terms"][0]].domain, p["terms"], pair=p["id"], target=None,
                domains=[by_term[t].domain for t in p["terms"]])
    cycles = {}
    for d in layout.DOMAINS:
        terms = split.targeted(d)
        cycles[d] = {"terms": terms, "order": [], "pass": 0, "groups": len({e.group for e in terms})}
    ks, weights = list(textgen.TERMS_PER_PROMPT), list(textgen.TERMS_PER_PROMPT.values())
    for g, target in spec.targets.items():
        need = max(0, target - state.counts[g])
        own = sorted((e for e in split.entries if e.group == g), key=lambda e: (e.domain, e.phrase))
        if any(e.side != "targeted" for e in own):
            raise RuntimeError(f"group {g} is not targeted under v3")
        _rng(seed, run, "group", g).shuffle(own)
        for j in range(math.ceil(FACTOR * need)):
            first = own[j % len(own)]
            c = cycles[first.domain]
            k = _rng(seed, run, "k", len(out)).choices(ks, weights)[0]
            chosen, groups, tries = [first.term], {g}, 0
            while len(chosen) < min(k, c["groups"]) and tries < 2 * len(c["terms"]):
                tries += 1
                if not c["order"]:
                    c["order"] = list(range(len(c["terms"])))
                    _rng(seed, run, first.domain, "pass", c["pass"]).shuffle(c["order"])
                    c["pass"] += 1
                e = c["terms"][c["order"].pop()]
                if e.group in groups:
                    continue
                groups.add(e.group)
                chosen.append(e.term)
            add("single", first.domain, chosen, pair=None, target=g)
    return out


# ----------------------------------------------------------------------------- run pairs: hygiene and selection

def hygiene(records, split: vocab.Split, medical, normalize, coverage, sets, heldout_record, prior: Sequence[str] = (),
            forms=None):
    """v1's hygiene (hygiene.run) under split v3, with `prior` (normalized texts: full's kept sentences and the earlier
    rounds' selected ones) as earlier kept sentences for the exact and near-duplicate checks."""
    from synthetic import hygiene as h
    if split.data.get("version") != 3:
        raise ValueError("run pairs' hygiene runs under split version 3")
    kept, report, rows = h.run(records, split, medical, normalize, coverage, sets, heldout_record, forms, prior=prior)
    report |= {"split_version": 3, "prior_sentences": len(prior)}
    return kept, report, rows


def prior_sentences(round: int) -> list[str]:
    """Normalized texts of full's kept sentences, then of the selected sentences of rounds before `round`."""
    from synthetic.textgen import read_jsonl
    out = [s["norm"] for s in read_jsonl(layout.run_dir("full") / "sentences.jsonl")]
    for k in range(1, round):
        out += [s["norm"] for s in read_jsonl(layout.run_dir(f"{layout.PAIRS_RUN}-r{k}") / "sentences.jsonl")]
    return out


def _order(seed: int, item_id: str) -> bytes:
    return hashlib.sha256(f"{seed}|{layout.PAIRS_RUN}|select|{item_id}".encode()).digest()


def select(sentences: Sequence[dict], prompts: Mapping[str, dict], spec: Spec, split: vocab.Split, state: State,
           seed: int = layout.SEED) -> tuple[list[dict], dict]:
    """(selected sentences in selection order, {id: {decision, groups}}) of a round's hygiene-kept sentences (module
    docstring). Decisions: selected, missing_member, contrast_full, missing_target, over_target."""
    counts, contrast = Counter(state.counts), Counter(state.contrast)
    order = sorted(sentences, key=lambda s: (prompts[s["id"]]["kind"] != "contrast", _order(seed, s["id"])))
    selected, decisions = [], {}
    for s in order:
        p = prompts[s["id"]]
        present = groups_in(s["label"], split)
        if p["kind"] == "contrast":
            pair = spec.by_id[p["pair"]]
            decision = ("missing_member" if not set(pair["groups"]) <= set(present) else
                        "contrast_full" if contrast[p["pair"]] >= CONTRAST else "selected")
            if decision == "selected":
                contrast[p["pair"]] += 1
        else:
            g = p["target"]
            decision = ("missing_target" if g not in present else
                        "over_target" if counts[g] >= spec.targets.get(g, 0) else "selected")
        if decision == "selected":
            selected.append(s)
            for g in present:
                counts[g] += 1
        decisions[s["id"]] = {"decision": decision, "groups": present}
    return selected, decisions


def shortfalls(spec: Spec, state: State) -> dict:
    return {"pairs": {p["id"]: {"target": CONTRAST, "count": state.contrast[p["id"]],
                                "shortfall": max(0, CONTRAST - state.contrast[p["id"]])} for p in spec.trained},
            "groups": {g: {"target": t, "count": state.counts[g], "shortfall": max(0, t - state.counts[g]),
                           "dev_informed": g in spec.dev_informed} for g, t in spec.targets.items()}}


# ----------------------------------------------------------------------------- run pairs: files

def round_dir(round: int) -> Path:
    return layout.run_dir(f"{layout.PAIRS_RUN}-r{round}")


def load_spec(split: vocab.Split | None = None) -> Spec:
    """The spec of run pairs from split v3 and full's clips; frozen in runs/pairs/spec.json at its first use."""
    from synthetic import confusable
    split = split or load_v3()
    full = confusable.full_clip_counts(layout.run_dir("full") / "clips.jsonl", split)
    spec = Spec(split, full)
    path = layout.run_dir(layout.PAIRS_RUN) / "spec.json"
    rec = spec.record() | {"split_sha256": _sha256(layout.split_version_path(3))}
    if path.exists():
        if json.loads(path.read_text()) != rec:
            raise RuntimeError(f"{path} differs from the spec built now (split v3 or full changed)")
    else:
        _write(path, _json(rec))
    return spec


def round_clips(round: int) -> list[dict]:
    """A finished round's kept clips with their prompt's kind, pair and target (refused unless its clip file matches
    its meta and its hygiene ran under split v3)."""
    from synthetic.textgen import read_jsonl
    d = round_dir(round)
    meta = json.loads((d / "clips.meta.json").read_text())
    if _sha256(d / "clips.jsonl") != meta["clips_sha256"]:
        raise RuntimeError(f"{d}/clips.jsonl does not match clips.meta.json")
    if meta.get("hygiene_split_sha256") != _sha256(layout.split_version_path(3)):
        raise RuntimeError(f"{d}: hygiene did not run under split v3")
    prompts = {p["id"]: p for p in read_jsonl(d / "prompts.jsonl")}
    return [c | {"kind": prompts[c["id"]]["kind"], "pair": prompts[c["id"]].get("pair"),
                 "target": prompts[c["id"]].get("target"), "round": round} for c in read_jsonl(d / "clips.jsonl")]


def finished_rounds() -> list[int]:
    done = [k for k in range(1, ROUNDS + 1) if (round_dir(k) / "clips.meta.json").exists()]
    if done != list(range(1, len(done) + 1)):
        raise RuntimeError(f"rounds {done} are not consecutive from 1")
    return done


def state_before(spec: Spec, split: vocab.Split, round: int) -> State:
    done = finished_rounds()
    if any(k not in done for k in range(1, round)):
        raise RuntimeError(f"round {round} needs rounds 1..{round - 1} finalized first")
    return State(spec, split, [c for k in range(1, round) for c in round_clips(k)])


def finalize_run(spec: Spec, split: vocab.Split) -> dict:
    """runs/pairs/clips.jsonl (the finished rounds' kept clips with kind, pair, target and round), clips.meta.json
    and report.json (prompts and clips per round, contrast clips per trained pair and counts per target group against
    their targets, with shortfalls)."""
    from synthetic.textgen import read_jsonl, write_jsonl
    rounds = finished_rounds()
    if not rounds:
        raise RuntimeError("no finished round")
    clips, per_round, provisional = [], {}, False
    for k in rounds:
        rc = round_clips(k)
        clips += rc
        meta = json.loads((round_dir(k) / "clips.meta.json").read_text())
        provisional |= bool(meta.get("provisional"))
        prompts = read_jsonl(round_dir(k) / "prompts.jsonl")
        per_round[k] = {"prompts": len(prompts), "contrast_prompts": sum(p["kind"] == "contrast" for p in prompts),
                        "single_prompts": sum(p["kind"] == "single" for p in prompts), "clips": len(rc),
                        "hours": sum(c["duration"] for c in rc) / 3600, "clips_sha256": meta["clips_sha256"]}
    clips.sort(key=lambda c: c["id"])
    state = State(spec, split, clips)
    over = [pid for pid, n in state.contrast.items() if n > CONTRAST]
    if over:
        raise RuntimeError(f"pairs {over} have more than {CONTRAST} contrast clips")
    d = layout.run_dir(layout.PAIRS_RUN)
    write_jsonl(d / "clips.jsonl", clips)
    v3 = _sha256(layout.split_version_path(3))
    meta = {"clips": len(clips), "hours": sum(c["duration"] for c in clips) / 3600,
            "clips_sha256": _sha256(d / "clips.jsonl"), "split_sha256": v3, "hygiene_split_sha256": v3,
            "split_stale": False, "provisional": provisional, "rounds": per_round,
            "roundtrip_max_wer": layout.ROUNDTRIP_MAX_WER}
    _write(d / "clips.meta.json", _json(meta))
    short = shortfalls(spec, state)
    report = {"run": layout.PAIRS_RUN, "rounds": per_round, "spec": spec.record(), **short,
              "total": {"clips": len(clips), "hours": meta["hours"],
                        "contrast_clips": sum(c["kind"] == "contrast" for c in clips),
                        "single_clips": sum(c["kind"] == "single" for c in clips),
                        "pairs_short": sum(1 for x in short["pairs"].values() if x["shortfall"]),
                        "groups_short": sum(1 for x in short["groups"].values() if x["shortfall"])},
              "dev_informed_groups": sorted(spec.dev_informed)}
    _write(d / "report.json", _json(report))
    return report
