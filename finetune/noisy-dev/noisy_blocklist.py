"""Leakage controls: the reserved-unit blocklist and the TRAIN-pool check. See plans/build.md "Leakage controls".

make_blocklist(pool) lists every reserved unit of the evaluation pool, not only the files sampled:

- whole datasets and campaigns: Zenodo 3997406 (all four microphones), DEMAND (all environments
  and channels), MUSAN speech/us-gov, BUT ReverbDB by room;
- for FSD50K, every eval-split Freesound sound ID and uploader (and those of the rotor plug-in's
  clips, if any);
- the decoded-PCM SHA-256 of every file in the pool (noisy_sources.pcm_sha256), originals and
  prepared copies, and of every FSD50K eval clip;
- the excluded source families (noisy_sources.EXCLUDED_SOURCES), for the record.

check_train_pool(rows, blocklist) takes a later augmentation plan's candidate TRAIN sources, one
dict per file with the keys dataset and pcm_sha256 (required) and, where the dataset needs them,
subset, split, room, freesound_id and uploader. It raises LeakageError listing every row that
matches a reserved dataset/campaign (by dataset name and subset prefix), a reserved BUT room, a
reserved split (FSD50K eval), a reserved Freesound sound ID or uploader, or a reserved PCM hash,
and every row whose provenance is insufficient or ambiguous (review round 2, finding 6): no
dataset or PCM hash; a Freesound-derived row (FSD50K, Freesound, ESC-50, MUSAN noise/free-sound)
without sound ID and uploader; an FSD50K row without a dev split; a MUSAN row without a subset,
or naming a parent of a reserved subset ("speech" contains "speech/us-gov"); a BUT row without a
room.

This check is provisional (plans/build.md "Scope of the TRAIN check"): the authoritative one is
designed with the augmentation plan. Review round 5: an explicit origin is refused outright for
wrapper and MUSAN candidates, so it can never override a mapped subset.

Review round 4 (finding 2): resolve_chain gives the chain of links from the candidate to its
origin. An explicit origin may only carry ORIGIN_FIELDS (never a PCM hash) and may not contradict
a known wrapper mapping or a non-wrapper candidate's own dataset. The candidate's own PCM hash is
always checked, and reserved identities (units, splits, sound IDs, uploaders) are checked at
every link, so an origin cannot erase them.

Review round 3 (finding 2): checks run on the candidate's canonical origin. Collection wrappers
(WRAPPERS: DNS-Challenge, RIRS_NOISES/SLR28, SLR26) are resolved through their subset to the
dataset a file comes from (DNS "DEMAND/..." is DEMAND, reserved whole); a wrapper subset with
mixed origins (SLR28 point-source and real-RIR sets) needs an explicit "origin" {dataset, subset},
and an unknown wrapper subset is ambiguous. A resolved dataset outside CANONICAL_DATASETS fails
closed ("unknown_dataset"). A MUSAN subset must lie in one of MUSAN's groups ("noise" is
ambiguous: it contains noise/free-sound), and a BUT room may not be empty or whitespace. Re-encoded or edited derivatives that
change the PCM are not caught (accepted limitation; dataset and uploader blocking mitigate it).
"""
from __future__ import annotations

import json
import re

import noisy_sources

FORMAT = "noisy-dev-blocklist/1"
pcm_sha256 = noisy_sources.pcm_sha256  # the hash TRAIN candidates must be compared with
# Pool source -> the reserved unit(s) it stands for.
_DATASET_UNITS = {"gunshots-ipod": [{"dataset": "zenodo-3997406"}],
                  "demand": [{"dataset": "demand"}],
                  "musan": [{"dataset": "musan", "subset": "speech/us-gov"}]}
_ALIASES = {"gunshots-ipod": "zenodo-3997406", "zenodo3997406": "zenodo-3997406", "but_reverbdb": "but-reverbdb",
            "butreverbdb": "but-reverbdb", "but-reverb-db": "but-reverbdb", "dns": "dns-challenge",
            "dns-noise": "dns-challenge", "slr28": "rirs-noises", "rirs-noise": "rirs-noises"}


FREESOUND_DATASETS = ("fsd50k", "freesound", "esc50", "fsd-kaggle-2018", "fsd-kaggle-2019")
FSD50K_TRAIN_SPLITS = ("dev", "train", "val")   # FSD50K's dev split and its train/val parts
RESERVED_SPLITS = {"fsd50k": ("eval", "test")}  # FSD50K's eval split (also called test)
MUSAN_GROUPS = ("music/fma", "music/fma-western-art", "music/hd-classical", "music/jamendo", "music/rfm",
                "noise/free-sound", "noise/sound-bible", "speech/librivox", "speech/us-gov")
# Datasets a TRAIN file may come from (canonical origins). Extend deliberately; anything else fails closed.
CANONICAL_DATASETS = frozenset({
    "fsd50k", "freesound", "esc50", "fsd-kaggle-2018", "fsd-kaggle-2019", "musan", "demand", "zenodo-3997406",
    "but-reverbdb", "audioset", "urbansound8k", "wham-noise", "chime-background", "noisex-92", "sonniss",
    "rwcp", "reverb-challenge", "aachen-air", "ace-challenge", "mit-ir-survey", "ir64-car",
    "pyroomacoustics-simulated", "slr26-simulated", "librispeech", "vctk"})
# Collection wrappers: first subset component (lowercased) -> the wrapper or origin it comes from;
# None: mixed origins, an explicit "origin" is required; "*": the whole wrapper has one origin.
WRAPPERS = {
    "dns-challenge": {"demand": "demand", "freesound": "freesound", "audioset": "audioset", "slr26": "slr26",
                      "slr28": "rirs-noises", "rirs_noises": "rirs-noises"},
    "rirs-noises": {"simulated_rirs": "slr26-simulated", "real_rirs_isotropic_noises": None,
                    "pointsource_noises": None},
    "slr26": {"*": "slr26-simulated"},
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class LeakageError(ValueError):
    """A TRAIN candidate matches a reserved evaluation unit."""

    def __init__(self, matches: list[dict]) -> None:
        self.matches = matches
        super().__init__(f"{len(matches)} TRAIN candidate(s) match reserved evaluation units: "
                         + "; ".join(f"row {m['row']}: {m['kind']} {m['value']!r}" for m in matches[:10]))


def norm_dataset(name: str | None) -> str:
    key = re.sub(r"[\s_]+", "-", str(name or "").strip().lower())
    return _ALIASES.get(key, key)


def make_blocklist(pool: dict) -> dict:
    units = []
    for source in sorted(pool["sources"]):
        units += _DATASET_UNITS.get(source, [])
    rooms = sorted({r["room"] for r in pool["rirs"] if r["source"] == "but-reverbdb"})
    units += [{"dataset": "but-reverbdb", "room": room} for room in rooms]
    ids, uploaders = set(), set()
    for name in ("fsd50k", "rotor"):
        block = pool["reserved"].get(name) or {}
        ids.update(int(i) for i in block.get("sound_ids", []))
        uploaders.update(block.get("uploaders", []))
    for item in [c for items in pool["clips"].values() for c in items] + pool["event_clips"]:
        if "freesound_id" in item:
            ids.add(int(item["freesound_id"]))
        if "uploader" in item:
            uploaders.add(item["uploader"])
    hashes = sorted({f["pcm_sha256"] for f in noisy_sources.pool_files(pool)})
    return {"format": FORMAT,
            "units": sorted(units, key=lambda u: (u["dataset"], u.get("subset") or "", u.get("room") or "")),
            "freesound_ids": sorted(ids), "freesound_uploaders": sorted(uploaders), "pcm_sha256": hashes,
            "pcm_hash": "noisy_sources.pcm_sha256: SHA-256 of b'<rate>,<channels>\\n' + float32 LE samples",
            "excluded_families": dict(sorted(noisy_sources.EXCLUDED_SOURCES.items())),
            "note": "Reserved for evaluation. No TRAIN pool may be built until the augmentation plan's reviewed "
                    "TRAIN-pool checker exists; noisy_blocklist.check_train_pool is a provisional helper, not that "
                    "gate (plans/build.md 'Scope of the TRAIN check'). The five dev sets stay excluded from "
                    "training as before. Re-encoded derivatives are not caught by the PCM hash; dataset and "
                    "uploader blocking mitigate that."}


ORIGIN_FIELDS = frozenset({"dataset", "subset", "split", "room", "freesound_id", "uploader"})


def _wrapper_target(dataset: str, subset: str) -> tuple[str | None, str, str | None]:
    """(mapped origin or None, rest of the subset, problem) for a wrapper row."""
    table = WRAPPERS[dataset]
    if "*" in table:
        return table["*"], subset, None
    if not subset:
        return None, "", f"{dataset} without a subset naming the file's origin"
    head, _, rest = subset.partition("/")
    if head.lower() not in table:
        return None, rest, f"{dataset} subset {head!r} has no known origin"
    if table[head.lower()] is None:
        return None, rest, "mixed"
    return table[head.lower()], rest, None


def resolve_chain(row: dict) -> tuple[list[dict], str | None]:
    """The provenance chain from the candidate to its canonical origin (every link is checked for
    reserved identities), and a problem or None. Wrappers resolve through their subset. An
    explicit "origin" may only carry ORIGIN_FIELDS (never a PCM hash: the candidate's own hash is
    immutable), is required for a mixed-origin wrapper subset, and must agree with any known
    mapping (DNS "DEMAND/..." with an origin other than DEMAND conflicts); a non-wrapper row's
    origin must name its own dataset."""
    chain, current = [row], row
    for _ in range(6):
        dataset = norm_dataset(current.get("dataset"))
        origin = current.get("origin")
        if origin is not None:
            if not isinstance(origin, dict) or not norm_dataset(origin.get("dataset")):
                return chain, "origin without a dataset"
            extra = sorted(set(origin) - ORIGIN_FIELDS)
            if extra:
                return chain, f"origin carries fields {extra} (allowed: {sorted(ORIGIN_FIELDS)}; pcm_sha256 is the candidate's own)"
            if dataset in WRAPPERS or dataset == "musan":  # review round 5: origins never override mapped subsets
                return chain, (f"an explicit origin is not accepted for a {dataset} candidate: it conflicts with "
                               "(or could override) the subset mapping; this provisional check fails closed")
        if dataset in WRAPPERS:
            subset = str(current.get("subset") or "").strip().strip("/")
            target, rest, problem = _wrapper_target(dataset, subset)
            if problem and problem != "mixed":
                return chain, problem
            if origin is None:
                if target is None:
                    return chain, f"{dataset}/{subset.partition('/')[0]} has mixed origins; state the file's origin"
                nxt = {k: v for k, v in current.items() if k != "origin"}
                nxt.update(dataset=target, subset=rest or None)
            else:
                if target is not None and norm_dataset(origin["dataset"]) != target:
                    return chain, (f"origin {origin['dataset']!r} conflicts with the wrapper mapping "
                                   f"{dataset}/{subset.partition('/')[0] or '*'} -> {target}")
                nxt = {k: v for k, v in current.items() if k not in ("origin", "dataset", "subset", "split", "room")}
                nxt.update(origin)
        elif origin is not None:
            if norm_dataset(origin["dataset"]) != dataset:
                return chain, f"origin {origin['dataset']!r} conflicts with the candidate's dataset {dataset!r}"
            nxt = {k: v for k, v in current.items() if k != "origin"}
            nxt.update(origin)
        else:
            return chain, None
        chain.append(nxt)
        current = nxt
    return chain, "provenance chain too deep"


def resolve_origin(row: dict) -> tuple[dict, str | None]:
    """(the canonical origin of a candidate, a problem or None)."""
    chain, problem = resolve_chain(row)
    return chain[-1], problem


def _unit_match(unit: dict, row: dict) -> str | None:
    """"unit" if the row lies in the reserved unit, "ambiguous_provenance" if the row's subset
    contains the unit's (a parent) or its room is missing, None if disjoint."""
    if norm_dataset(row.get("dataset")) != unit["dataset"]:
        return None
    if unit.get("subset"):
        subset = str(row.get("subset") or "").strip().strip("/").lower()
        if not subset:
            return "ambiguous_provenance"  # the dataset without a subset may include the reserved one
        if unit["subset"].startswith(subset + "/"):
            return "ambiguous_provenance"  # a parent of the reserved subset
        if not (subset == unit["subset"] or subset.startswith(unit["subset"] + "/")):
            return None  # another part of the dataset (e.g. MUSAN noise)
    if unit.get("room"):
        room = str(row.get("room") or "").strip()
        if not room:
            return "ambiguous_provenance"  # a BUT row without a room fails closed
        if room != unit["room"]:
            return None
    return "unit"


def _provenance_problems(row: dict) -> list[tuple[str, str]]:
    """(kind, problem) of a resolved row's provenance."""
    problems = []
    dataset = norm_dataset(row.get("dataset"))
    if not dataset:
        problems.append(("insufficient_provenance", "no dataset"))
    elif dataset not in CANONICAL_DATASETS:
        problems.append(("unknown_dataset", f"{dataset!r} is not a known canonical origin"))
    if not _SHA256.fullmatch(str(row.get("pcm_sha256") or "")):
        problems.append(("insufficient_provenance", "no decoded-PCM SHA-256"))
    subset = str(row.get("subset") or "").strip().strip("/").lower()
    freesound = dataset in FREESOUND_DATASETS or (dataset == "musan" and subset.startswith("noise/free-sound"))
    if freesound and (row.get("freesound_id") is None or not str(row.get("uploader") or "").strip()):
        problems.append(("insufficient_provenance", "Freesound-derived without sound ID and uploader"))
    if dataset == "fsd50k" and str(row.get("split") or "").lower() not in FSD50K_TRAIN_SPLITS + RESERVED_SPLITS["fsd50k"]:
        problems.append(("insufficient_provenance", "FSD50K without a split"))
    if dataset == "musan" and not any(subset == g or subset.startswith(g + "/") for g in MUSAN_GROUPS):
        problems.append(("ambiguous_provenance", f"MUSAN subset {subset!r} is not within one MUSAN group"))
    if dataset == "but-reverbdb" and not str(row.get("room") or "").strip():
        problems.append(("ambiguous_provenance", "BUT ReverbDB without a room"))
    return problems


def check_train_pool(rows: list[dict], blocklist: dict) -> list[dict]:
    """The rows unchanged if every row has sufficient provenance and none matches a reserved unit;
    LeakageError listing every problem otherwise (checks run on each row's canonical origin)."""
    if blocklist.get("format") != FORMAT:
        raise ValueError("not a noisy-dev blocklist")
    ids = set(blocklist["freesound_ids"])
    uploaders = set(blocklist["freesound_uploaders"])
    hashes = set(blocklist["pcm_sha256"])
    matches = []
    for k, original in enumerate(rows):
        found: dict = {}

        def add(kind: str, value, found=found, k=k) -> None:  # one match per (kind, value)
            found.setdefault((kind, json.dumps(value, sort_keys=True, default=str)), {"row": k, "kind": kind,
                                                                                      "value": value})

        if original.get("pcm_sha256") in hashes:  # the candidate's own hash, whatever its origin says
            add("pcm_sha256", original["pcm_sha256"])
        chain, problem = resolve_chain(original)
        if problem:
            add("ambiguous_provenance", problem)
        else:
            for kind, value in _provenance_problems(chain[-1]):
                add(kind, value)
        for link in chain:  # reserved identities at every link of the provenance chain
            dataset = norm_dataset(link.get("dataset"))
            if str(link.get("split") or "").lower() in RESERVED_SPLITS.get(dataset, ()):
                add("reserved_split", f"{dataset}/{link['split']}")
            for unit in blocklist["units"]:
                kind = _unit_match(unit, link)
                if kind and (kind == "unit" or link is chain[-1]):
                    add(kind, unit)
            if link.get("freesound_id") is not None and int(link["freesound_id"]) in ids:
                add("freesound_id", int(link["freesound_id"]))
            if link.get("uploader") is not None and link["uploader"] in uploaders:
                add("uploader", link["uploader"])
        matches += found.values()
    if matches:
        raise LeakageError(matches)
    return rows
