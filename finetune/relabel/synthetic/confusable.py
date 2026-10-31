"""The confusable-pair detector (synthetic plan v5, "Confusable pairs": Detector; preregistered, run once).

    ./python -m synthetic pairs-words     word counts over the main pool's training labels (CPU job, about 10 min)
    ./python -m synthetic pairs-detect    the candidate file for the blind curator (CPU)

Inputs, and nothing else (require_input accepts exactly these paths and refuses every other one, so no evaluation
reference, hypothesis or score file can be read):
- split: the frozen term split v2 ($RELABEL_ROOT/synthetic/terms/split.json, or its copy split.v2.json): every
  listed term of both sides, with its group, side and domain;
- the pronunciation lexicon (terms/synthetic/pronunciation/, committed; synthetic/pronounce.py);
- words: word counts over the main pool's training labels ($RELABEL_ROOT/synthetic/v5/words.json, build_words):
  main-f3's T* label store over every C1 batch, read as stage 3 reads it (training/labels.py LabelStore at the
  cleaning version current at build time, training.cleaning.VERSION, with repetition rejection; then the read-time
  reject list), its kept texts counted by word_counts; the cleaning version and identity, the store's identity
  SHA-256 and the reject list's identity are recorded with the counts;
- full: the kept clips of run `full` (runs/full/clips.jsonl, with clips.meta.json, which must name the same split):
  clips per group, a clip counting once for every group with a term occurrence in its label (its "terms").

word_counts: each label is folded (NFKD, combining marks dropped), split into tokens (pronounce.TOKEN: maximal runs of
letters and digits), and every alphabetic token is counted in lower case. A word count is the token's own count
("alkanes" is not counted for "alkane").

Tokens (tokens()): each listed term's letter tokens (pronounce.TOKEN runs of letters only), folded and lower case,
without STOPWORDS. A token is compared by its spoken form: the pronunciation layer's rendering of the token as written
in the term (pronounce.Lexicon.render: a lexicon respelling, letter names for a spelled token, the default
all-capitals rule), spaces removed, folded and lower case ("roe" -> "arohee", "ebit" -> "eebit"). A token whose
spoken form has fewer than MIN_SPOKEN letters is skipped. One token rendered two ways in two listed terms ("ROE"
spelled by the all-capitals rule, "roe" said as a word) is an error.

Sound key (sound_key): the spoken form, diacritics folded and lower case, then FOLDS in order (each a plain string
replacement), then runs of one letter collapsed to one.

Pair rule (rule()), on the keys of two different tokens, in this order of precedence:
    same-sound      the keys are equal
    prefix          one is a prefix of the other, the shorter has >= 3 letters and the longer at most 3 more
    shared-prefix   a common prefix of >= 5 letters that is >= 70% of the shorter key
    edit            Levenshtein distance <= 1 (shorter key <= 6 letters) or <= 2 (longer keys)
Excluded:
    inflection      one token is the other plus a suffix of INFLECTIONS (the shorter's final "e" or "y" dropped first)
    common          either token occurs COMMON_MIN or more times in the word counts
Two tokens that meet the rule and are not excluded pair the groups of the terms they occur in, for every two
different groups (two tokens of one group never pair it with itself).

Candidates: one per pair of groups with at least one pair of tokens (its "hits"); its rule is the strongest of its
hits' rules. Side type: "heldout-targeted", "targeted-targeted" or "heldout-heldout" (split v2). The always-accepted
pairs (pairs.FORCED) are marked "forced"; one the detector does not find is added with rule "forced", and its hits
are the token pairs between its groups that met the rule but were excluded (each with its "excluded" reason).
Members: held-out first, else by group key; a member's "term" is its listed term that holds the strongest hit's token
(fewest words, then shortest, then alphabetical). Candidates are sorted by side type (in the order above), then by
the members' group keys, and numbered c0001, c0002, ...

Output ($RELABEL_ROOT/synthetic/v5/): candidates.json (the candidate file: terms, sides, domains, rules, spoken forms,
real-label counts and full clip counts, plus the rules and the inputs' SHA-256; no evaluation count and no model
output), candidates.tsv (one row per hit) and pairs.template.tsv (terms/synthetic/pairs.tsv's format, pairs.py, with
the decision and reason left blank except for the always-accepted pairs). There are no timestamps: the same inputs
give the same bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from synthetic import layout, pronounce, vocab

STOPWORDS = frozenset("a an the of and or in on for to with by at from as is are be into over under per vs via its"
                      .split())
FOLDS = (("sch", "sh"), ("ph", "f"), ("oe", "o"), ("ue", "u"), ("ae", "e"), ("ee", "e"), ("ck", "k"),
         ("q", "k"), ("x", "ks"), ("z", "s"), ("y", "i"), ("c", "k"))
INFLECTIONS = ("s", "es", "ic", "ics", "ity", "ities", "al", "ally", "ly", "ing", "ed", "er", "ers", "ation", "ations",
               "ion", "ions", "ive", "ism", "ist", "ists", "ness")
COMMON_MIN = 300
MIN_SPOKEN = 3
RULES = ("same-sound", "prefix", "shared-prefix", "edit")
SIDE_TYPES = ("heldout-targeted", "targeted-targeted", "heldout-heldout")
INPUTS = ("split", "words", "clips")
LABEL_RUN = "main-f3"
LABEL_SET = "T*"
REJECT_LIST = "pool/ted-reject"      # stage 3's read-time reject list (train_b0prime.REJECT_LIST)
TOKENIZER = "NFKD fold (combining marks dropped), pronounce.TOKEN runs, alphabetic tokens only, lower case"
FORMAT = 1


class InputRefused(RuntimeError):
    """A path that is not one of the detector's listed inputs (an evaluation or hypothesis file in particular)."""


def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()


def word_counts(texts: Iterable[str | None]) -> Counter:
    """Alphabetic token counts of `texts` (module docstring)."""
    out = Counter()
    for t in texts:
        if t:
            out.update(w for w in pronounce.TOKEN.findall(fold(t)) if w.isalpha())
    return out


def sound_key(spoken: str) -> str:
    t = fold(spoken)
    for a, b in FOLDS:
        t = t.replace(a, b)
    out = []
    for ch in t:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def levenshtein(a: str, b: str, cap: int) -> int:
    """The edit distance of a and b, or cap + 1 once it is known to exceed cap."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def rule(ka: str, kb: str) -> str | None:
    """The pair rule two sound keys meet (module docstring), or None."""
    if ka == kb:
        return "same-sound"
    s, l = sorted((ka, kb), key=len)
    if len(s) >= 3 and l.startswith(s) and len(l) - len(s) <= 3:
        return "prefix"
    p = _common_prefix(ka, kb)
    if p >= 5 and p >= 0.7 * len(s):
        return "shared-prefix"
    cap = 1 if len(s) <= 6 else 2
    if levenshtein(ka, kb, cap) <= cap:
        return "edit"
    return None


def inflection(a: str, b: str) -> bool:
    """Whether one token is the other plus an INFLECTIONS suffix (the shorter's final e or y dropped first)."""
    s, l = sorted((a, b), key=len)
    for stem in {s, s[:-1] if s[-1:] in ("e", "y") else s}:
        if l.startswith(stem) and l[len(stem):] in INFLECTIONS:
            return True
    return False


def spoken_form(token: str, lexicon: pronounce.Lexicon) -> str:
    return lexicon.render(token)[0].replace(" ", "")


def tokens(split: vocab.Split, lexicon: pronounce.Lexicon) -> dict[str, dict]:
    """{token: {"spoken", "key", "groups": {group: [terms]}}} over every listed term (module docstring)."""
    out: dict[str, dict] = {}
    for e in split.entries:
        for m in pronounce.TOKEN.finditer(e.term):
            raw = m.group()
            tok = fold(raw)
            if not raw.isalpha() or tok in STOPWORDS:
                continue
            sp = fold(spoken_form(raw, lexicon))
            if sum(c.isalpha() for c in sp) < MIN_SPOKEN:
                continue
            t = out.setdefault(tok, {"spoken": sp, "key": sound_key(sp), "groups": defaultdict(list)})
            if t["spoken"] != sp:
                raise ValueError(f"token {tok!r} is spoken {t['spoken']!r} and {sp!r} in different listed terms")
            if e.term not in t["groups"][e.group]:
                t["groups"][e.group].append(e.term)
    return out


def _representative(terms: list[str]) -> str:
    return min(terms, key=lambda t: (len(t.split()), len(t), t))


def _forced_groups(split: vocab.Split) -> dict[str, tuple[str, str] | None]:
    from synthetic import pairs
    out = {}
    for a, b in pairs.FORCED:
        name = f"{' '.join(a)} / {' '.join(b)}"
        ea, eb = split.by_phrase.get(a), split.by_phrase.get(b)
        out[name] = (ea.group, eb.group) if ea and eb else None
    return out


def detect(split: vocab.Split, lexicon: pronounce.Lexicon, counts: Mapping[str, int],
           clips: Mapping[str, int]) -> dict:
    """The candidate file's body (module docstring) from the split, the lexicon, the label word counts and the full
    run's clips per group. A pure function of its arguments."""
    toks = tokens(split, lexicon)
    side = {e.group: e.side for e in split.entries}
    domain = {e.group: e.domain for e in split.entries}
    terms_of = defaultdict(list)
    for e in split.entries:
        terms_of[e.group].append(e.term)
    found: dict[tuple[str, str], list[dict]] = defaultdict(list)
    excluded_hits: dict[tuple[str, str], list[dict]] = defaultdict(list)
    excluded = Counter()
    names = sorted(toks)
    for i, a in enumerate(names):
        ta = toks[a]
        for b in names[i + 1:]:
            tb = toks[b]
            group_pairs = sorted({tuple(sorted((ga, gb))) for ga in ta["groups"] for gb in tb["groups"] if ga != gb})
            if not group_pairs:
                continue
            r = rule(ta["key"], tb["key"])
            if r is None:
                continue
            reason = ("inflection" if inflection(a, b) else
                      "common" if max(counts.get(a, 0), counts.get(b, 0)) >= COMMON_MIN else None)
            if reason:
                excluded[reason] += 1
            for ga, gb in group_pairs:
                # oriented to the group pair: tokens[0] is a token of ga's terms, tokens[1] of gb's
                x, y = (a, b) if ga in ta["groups"] and gb in tb["groups"] else (b, a)
                hit = {"tokens": [x, y], "spoken": [toks[x]["spoken"], toks[y]["spoken"]],
                       "keys": [toks[x]["key"], toks[y]["key"]], "rule": r,
                       "label_counts": [int(counts.get(x, 0)), int(counts.get(y, 0))],
                       "terms": [sorted(toks[x]["groups"][ga]), sorted(toks[y]["groups"][gb])]}
                if reason:
                    excluded_hits[(ga, gb)].append(hit | {"excluded": reason})
                else:
                    found[(ga, gb)].append(hit)
    forced = _forced_groups(split)
    forced_pairs = {tuple(sorted(g)) for g in forced.values() if g}
    status = {name: ("not listed" if g is None else "detected" if tuple(sorted(g)) in found else "added")
              for name, g in forced.items()}
    for g in sorted(forced_pairs - set(found)):
        found[g] = []
    candidates = []
    for (ga, gb), hits in found.items():
        hits = sorted(hits, key=lambda h: (RULES.index(h["rule"]), h["tokens"]))
        is_forced = (ga, gb) in forced_pairs
        if not hits:
            hits = sorted(excluded_hits.get((ga, gb), []), key=lambda h: (RULES.index(h["rule"]), h["tokens"]))
        best = hits[0] if hits else None
        r = best["rule"] if best and "excluded" not in best else "forced"
        members = []
        for k, g in enumerate((ga, gb)):
            rep = _representative(best["terms"][k] if best else terms_of[g])
            members.append({"group": g, "term": rep, "terms": sorted(terms_of[g]), "side": side[g],
                            "domain": domain[g], "full_clips": int(clips.get(g, 0))})
        order = sorted(range(2), key=lambda k: (members[k]["side"] != "heldout", members[k]["group"]))
        members = [members[k] for k in order]
        hits = [h | {"tokens": [h["tokens"][k] for k in order], "spoken": [h["spoken"][k] for k in order],
                     "keys": [h["keys"][k] for k in order], "label_counts": [h["label_counts"][k] for k in order],
                     "terms": [h["terms"][k] for k in order]} for h in hits]
        side_type = "-".join(sorted(m["side"] for m in members))
        candidates.append({"side_type": side_type, "rule": r, "forced": is_forced, "members": members, "hits": hits})
    candidates.sort(key=lambda c: (SIDE_TYPES.index(c["side_type"]), [m["group"] for m in c["members"]]))
    for n, c in enumerate(candidates, 1):
        c["id"] = f"c{n:04d}"
    candidates = [{"id": c["id"], **{k: v for k, v in c.items() if k != "id"}} for c in candidates]
    moved = sorted({m["group"] for c in candidates if c["side_type"] == "heldout-targeted"
                    for m in c["members"] if m["side"] == "heldout"})
    targeted = sorted(int(clips.get(g, 0)) for g, s in side.items() if s == "targeted")
    summary = {"candidates": len(candidates),
               "by_side_type": {t: sum(c["side_type"] == t for c in candidates) for t in SIDE_TYPES},
               "by_rule": dict(sorted(Counter(c["rule"] for c in candidates).items())),
               "heldout_groups_in_heldout_targeted": len(moved),
               "heldout_groups_in_heldout_targeted_by_domain": dict(sorted(Counter(domain[g] for g in moved).items())),
               "forced": status, "excluded": {k: excluded[k] for k in ("inflection", "common")},
               "tokens": len(toks), "groups": len(side),
               "full_clips_per_targeted_group": {"groups": len(targeted),
                                                 "sorted[n // 2]": targeted[len(targeted) // 2] if targeted else None,
                                                 "sorted[9n // 10]": targeted[9 * len(targeted) // 10] if targeted else None}}
    return {"format": FORMAT, "plan": "synthetic plan v5, Confusable pairs: Detector", "rules": rules_record(),
            "summary": summary, "candidates": candidates}


def rules_record() -> dict:
    return {"stopwords": sorted(STOPWORDS), "min_spoken_letters": MIN_SPOKEN, "folds": [list(f) for f in FOLDS],
            "pair_rules": list(RULES), "inflections": list(INFLECTIONS), "common_min_label_count": COMMON_MIN,
            "spoken_form": "pronounce.Lexicon.render of the token as written, spaces removed",
            "word_counts": TOKENIZER}


# ----------------------------------------------------------------------------- inputs and outputs

def out_dir() -> Path:
    return layout.v5_dir()


def default_input(kind: str) -> Path:
    return {"split": layout.split_path(), "words": out_dir() / "words.json",
            "clips": layout.run_dir("full") / "clips.jsonl"}[kind]


def require_input(path: Path, kind: str) -> Path:
    """The resolved path if it is the detector's `kind` input (INPUTS), else InputRefused: the detector reads only its
    listed inputs, so any evaluation reference, hypothesis or score file is refused."""
    if kind not in INPUTS:
        raise ValueError(f"unknown detector input {kind!r} ({', '.join(INPUTS)})")
    allowed = {"split": (layout.split_path(), layout.split_version_path(2)), "words": (out_dir() / "words.json",),
               "clips": (layout.run_dir("full") / "clips.jsonl",)}[kind]
    p = Path(path).resolve()
    if p not in {a.resolve() for a in allowed}:
        raise InputRefused(f"{path} is not the detector's {kind} input ({', '.join(str(a) for a in allowed)}); it reads "
                           "nothing else, and never an evaluation reference, hypothesis or score file")
    return p


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def full_clip_counts(clips_path: Path, split: vocab.Split, split_sha256: str | None = None, total: bool = False):
    """Clips per group of a run's clips.jsonl; with split_sha256, its clips.meta.json must name that split. With total,
    (counts, Counter({"clips": number of clips}))."""
    clips_path = Path(clips_path)
    if split_sha256 is not None:
        meta = json.loads((clips_path.parent / "clips.meta.json").read_text())
        if meta.get("split_sha256") != split_sha256:
            raise RuntimeError(f"{clips_path} was finalized under another split ({meta.get('split_sha256')})")
    out, n = Counter(), Counter()
    with open(clips_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                n["clips"] += 1
                for g in {split.by_phrase[tuple(t.split())].group for t in c["terms"]}:
                    out[g] += 1
    return (out, n) if total else out


def _rel(path: Path) -> str:
    p = Path(path)
    root = layout.relabel_root().resolve()
    return str(p.relative_to(root)) if p.is_relative_to(root) else p.name


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def candidates_text(obj: dict) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def run(split_path: Path | None = None, words_path: Path | None = None, clips_path: Path | None = None,
        lexicon: pronounce.Lexicon | None = None) -> dict:
    """Read the listed inputs (each checked by require_input before anything is read), detect, and write the
    candidate file, its TSV view and the curation template to out_dir(). Returns the candidate file's object."""
    paths = {k: require_input(p or default_input(k), k)
             for k, p in (("split", split_path), ("words", words_path), ("clips", clips_path))}
    split = vocab.Split.load(paths["split"])
    if split.data.get("version") != 2:
        raise RuntimeError(f"{paths['split']}: the detector runs on split version 2")
    lexicon = lexicon or pronounce.Lexicon.load()
    words = json.loads(paths["words"].read_text())
    split_sha = _sha256(paths["split"])
    clips, clips_total = full_clip_counts(paths["clips"], split, split_sha, total=True)
    out = detect(split, lexicon, words["counts"], clips)
    out["inputs"] = {
        "split": {"file": _rel(paths["split"]), "sha256": split_sha, "version": split.data["version"]},
        "lexicon": lexicon.record(),
        "words": {"file": _rel(paths["words"]), "sha256": _sha256(paths["words"]), "labels": words.get("labels"),
                  "tokenizer": words.get("tokenizer")},
        "full": {"file": _rel(paths["clips"]), "sha256": _sha256(paths["clips"]), "clips": sum(clips_total.values()),
                 "split_sha256": split_sha}}
    d = out_dir()
    text = candidates_text(out)
    _write(d / "candidates.json", text)
    _write(d / "candidates.tsv", tsv(out))
    _write(d / "pairs.template.tsv", template(out, hashlib.sha256(text.encode()).hexdigest()))
    return out


def tsv(out: dict) -> str:
    cols = ["id", "side_type", "rule", "forced", "term_a", "side_a", "domain_a", "full_clips_a", "term_b", "side_b",
            "domain_b", "full_clips_b", "hit_rule", "tokens", "spoken", "keys", "label_counts", "excluded"]
    lines = ["\t".join(cols)]
    for c in out["candidates"]:
        a, b = c["members"]
        for h in c["hits"] or [{"rule": "", "tokens": [], "spoken": [], "keys": [], "label_counts": []}]:
            lines.append("\t".join(str(x) for x in (
                c["id"], c["side_type"], c["rule"], int(c["forced"]), a["term"], a["side"], a["domain"], a["full_clips"],
                b["term"], b["side"], b["domain"], b["full_clips"], h["rule"], " / ".join(h["tokens"]),
                " / ".join(h["spoken"]), " / ".join(h["keys"]), " / ".join(map(str, h["label_counts"])),
                h.get("excluded", ""))))
    return "\n".join(lines) + "\n"


def template(out: dict, sha256: str) -> str:
    """terms/synthetic/pairs.tsv's format (pairs.py) for this candidate file: one row per candidate, the decision and
    reason blank except for the always-accepted pairs."""
    from synthetic import pairs
    lines = list(pairs.HEADER) + [f"# candidates_sha256: {sha256}", "# " + "\t".join(pairs.COLUMNS)]
    for c in out["candidates"]:
        a, b = (m["term"] for m in c["members"])
        decision, reason = ("accept", pairs.FORCED_REASON) if c["forced"] else ("", "")
        lines.append("\t".join((a, b, c["rule"], decision, reason)))
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- word counts

def build_words(run: str = LABEL_RUN, version: int | None = None, log=print) -> dict:
    """Word counts over `run`'s T* training labels (module docstring), written to out_dir()/words.json."""
    from labeling import c2, shards
    from training import cleaning as tcleaning
    from training import label_sources, repetition as trep
    from training import labels as tlabels
    version = tcleaning.VERSION if version is None else int(version)
    run_dir = tlabels.production_run_dir(run)
    pool = c2._abs(c2.load_run(run_dir)["pool_dir"])
    committed = shards.committed(pool)
    through = int(shards.load_meta(pool, committed[-1])["batches"][1])
    cleaning = label_sources.cleaning_spec(run, version)
    store = tlabels.LabelStore.open(run_dir, pool, label_set=LABEL_SET, through_batch=through, cleaning=cleaning,
                                    repetition=trep.production(run))
    log(f"words: {len(store)} records through batch {through}, cleaning version {version}")
    reject = label_sources.RejectList.load(layout.relabel_root() / REJECT_LIST)
    texts = [t for i, t in store._texts.items() if t is not None and i not in reject.ids]
    counts = word_counts(texts)
    ident = store.identity
    out = {"format": FORMAT, "plan": "synthetic plan v5, Confusable pairs: Detector (word counts)",
           "labels": {"run": run, "run_dir": ident["run_dir"], "label_set": LABEL_SET, "through_batch": through,
                      "cleaning_version": version, "cleaning": ident.get("cleaning"),
                      "repetition": ident.get("repetition"), "reject": reject.identity, "mask": ident.get("mask"),
                      "records": len(store), "kept": ident.get("kept"), "labels_counted": len(texts),
                      "store_identity_sha256": ident["sha256"]},
           "tokenizer": TOKENIZER, "tokens": sum(counts.values()), "distinct": len(counts),
           "counts": dict(sorted(counts.items()))}
    _write(out_dir() / "words.json", json.dumps(out, ensure_ascii=False) + "\n")
    return out
