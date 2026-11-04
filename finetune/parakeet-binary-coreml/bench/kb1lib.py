"""Pure logic of the kea-b1 encoder benchmark: statistics, pairing, counterbalancing, WER scoring, encoder-output
error, bundle verification and memory/cache accounting. Python 3.9+ standard library only (the Mac's
/usr/bin/python3 runs the driver). Contract tests: tests/test_kb1lib.py.

The WER normalizer is the one LocalFlow's LibriSpeech comparison used (LocalFlow local/apple-speech-benchmark/
benchmark-apple-speech.py, 2026-10-15): lowercase, punctuation removed, digits/currency/percent/Mr/Mrs/Dr spelled
out, optionally British spellings mapped to American with Whisper's english.json (pinned SHA-256 below).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

LOCALFLOW_BUCKETS = [4, 8, 15, 30]
REQUIRED_FILES = ("frontend.json", "frontend.f32bin", "decoder_joint.json", "decoder_joint.f32bin", "vocabulary.json",
                  "Encoder.mlmodelc/model.mil", "Encoder.mlmodelc/weights/weight.bin")
UNITS = {"ane": "cpuAndNeuralEngine", "gpu": "cpuAndGPU", "cpu": "cpuOnly", "all": "all", "ext": "external"}
SPELLING_SHA256 = "6607f948be9824d2e1b2fa2223cd94c06c45afa4e05ea0e3d5e1f2bdffde2465"


# ---------------------------------------------------------------- statistics

def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median of no values")
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (Hyndman-Fan type 7, numpy's default)."""
    if not values:
        raise ValueError("quantile of no values")
    s = sorted(values)
    h = (len(s) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def latency_summary(ms: Sequence[float]) -> dict:
    return {"n": len(ms), "median_ms": median(ms), "p95_ms": quantile(ms, 0.95), "min_ms": min(ms), "max_ms": max(ms)}


def _percentile_interval(values: List[float]) -> List[float]:
    return [quantile(values, 0.025), quantile(values, 0.975)]


def paired_ratio(arm: Dict[str, List[float]], base: Dict[str, List[float]], n_boot: int = 2000, seed: int = 0) -> dict:
    """Arm / baseline latency, paired by clip (the same input timed in both).

    typical_ratio = median over clips of (arm clip median / baseline clip median); p95_ratio = p95 of the arm's
    pooled calls / p95 of the baseline's. Percentile-bootstrap 95% intervals resample clips with replacement,
    keeping each clip's arm and baseline calls together."""
    ids = sorted(set(arm) & set(base))
    if not ids:
        raise ValueError("no clips in common")
    am = {i: median(arm[i]) for i in ids}
    bm = {i: median(base[i]) for i in ids}

    def stats(sample: List[str]) -> Tuple[float, float]:
        typical = median([am[i] / bm[i] for i in sample])
        pa = [v for i in sample for v in arm[i]]
        pb = [v for i in sample for v in base[i]]
        return typical, quantile(pa, 0.95) / quantile(pb, 0.95)

    typical, p95 = stats(ids)
    rng = random.Random(seed)
    boots = [stats([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(n_boot)]
    return {"clips": len(ids), "typical_ratio": typical,
            "typical_ratio_ci95": _percentile_interval([b[0] for b in boots]),
            "p95_ratio": p95, "p95_ratio_ci95": _percentile_interval([b[1] for b in boots]),
            "arm_typical_ms": median(list(am.values())), "baseline_typical_ms": median(list(bm.values()))}


def counterbalanced_orders(variants: Sequence[str], rounds: int) -> List[List[str]]:
    """Round 2k runs the variants rotated by k; round 2k + 1 runs that order reversed. Over every pair of rounds
    each variant's positions sum to n - 1, so with an even number of rounds no variant sits systematically early
    or late in the session (drift and heat affect all alike); rotation changes who follows whom."""
    v = list(variants)
    orders: List[List[str]] = []
    for r in range(rounds):
        k = (r // 2) % max(len(v), 1)
        forward = v[k:] + v[:k]
        orders.append(forward if r % 2 == 0 else list(reversed(forward)))
    return orders


def group_samples(records: Iterable[dict], key_fields: Sequence[str], clip_field: str = "clip",
                  value_field: str = "ms") -> Dict[tuple, Dict[str, List[float]]]:
    out: Dict[tuple, Dict[str, List[float]]] = {}
    for r in records:
        key = tuple(r[f] for f in key_fields)
        out.setdefault(key, {}).setdefault(r[clip_field], []).append(r[value_field])
    return out


def thermal_summary(states: Sequence[str]) -> dict:
    counts: Dict[str, int] = {}
    for s in states:
        counts[s] = counts.get(s, 0) + 1
    return {"samples": len(states), "counts": dict(sorted(counts.items())),
            "all_nominal": bool(states) and all(s == "nominal" for s in states)}


# ---------------------------------------------------------------- WER (LocalFlow's LibriSpeech scoring)

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen").split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"))
ORDINALS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
            "nine": "ninth", "twelve": "twelfth"}
ABBREVIATIONS = {"mr": "mister", "mrs": "missus", "dr": "doctor"}


def cardinal(number: int) -> str:
    if number < 20:
        return ONES[number]
    if number < 100:
        return TENS[number // 10] + ("" if number % 10 == 0 else " " + ONES[number % 10])
    if number < 1000:
        rest = number % 100
        return ONES[number // 100] + " hundred" + ("" if rest == 0 else " " + cardinal(rest))
    for scale, name in SCALES:
        if number >= scale:
            rest = number % scale
            return cardinal(number // scale) + " " + name + ("" if rest == 0 else " " + cardinal(rest))
    raise ValueError("Unsupported number")


def ordinal(number: int) -> str:
    words = cardinal(number).split()
    last = words[-1]
    if last in ORDINALS:
        words[-1] = ORDINALS[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def year(number: int) -> str:
    high, low = divmod(number, 100)
    if low == 0:
        return cardinal(high) + " hundred"
    return cardinal(high) + " " + ("o " + ONES[low] if low < 10 else cardinal(low))


def spoken_number(token: str) -> str:
    match = re.fullmatch(r"(\$?)(\d[\d,]*)(st|nd|rd|th|%)?", token)
    if not match or (len(match[2]) > 1 and match[2].startswith("0")):
        return token
    digits = match[2].replace(",", "")
    if not digits or len(digits) > 12:
        return token
    number, suffix = int(digits), match[3]
    if suffix in ("st", "nd", "rd", "th"):
        words = ordinal(number)
    elif not match[1] and not suffix and "," not in match[2] and 1100 <= number <= 1999:
        words = year(number)
    else:
        words = cardinal(number)
    if match[1]:
        words += " dollars"
    if suffix == "%":
        words += " percent"
    return words


def plain_normalize(text: str) -> List[str]:
    cleaned = "".join(c if c.isalnum() or c == "'" else " " for c in text.lower())
    return [word for word in (w.strip("'") for w in cleaned.split()) if word]


def normalize(text: str, spelling: dict | None = None) -> List[str]:
    spelling = spelling or {}
    words: List[str] = []
    for token in text.lower().replace("’", "'").split():
        token = token.strip(".,;:!?\"()[]{}“”'")
        if re.fullmatch(r"\$?\d[\d,]*(st|nd|rd|th|%)?", token):
            token = spoken_number(token)
        for word in plain_normalize(token):
            word = ABBREVIATIONS.get(word, word)
            words.append(spelling.get(word, word))
    return words


def load_spelling_map(path: Path) -> dict:
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SPELLING_SHA256:
        raise ValueError("Spelling map does not match the pinned SHA-256")
    return json.loads(data)


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> Tuple[int, int, int]:
    """Minimum word edits as (substitutions, deletions, insertions)."""
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, word in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, candidate in enumerate(hypothesis, 1):
            cost, sub, dele, ins = previous[j - 1]
            options = [(cost + (word != candidate), sub + (word != candidate), dele, ins)]
            cost, sub, dele, ins = previous[j]
            options.append((cost + 1, sub, dele + 1, ins))
            cost, sub, dele, ins = current[j - 1]
            options.append((cost + 1, sub, dele, ins + 1))
            current.append(min(options))
        previous = current
    return previous[-1][1], previous[-1][2], previous[-1][3]


def _bootstrap(rows: list, statistic, seed: int, n_boot: int) -> List[float]:
    rng = random.Random(seed)
    values = [statistic([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(n_boot)]
    return _percentile_interval(values)


def _ratio(rows) -> float:
    words = sum(r[0] for r in rows)
    return sum(r[1] for r in rows) / words if words else 0.0


def wer_summary(items: Sequence[dict], spelling: dict | None, seed: int = 1, n_boot: int = 1000) -> dict:
    """items: {"reference", "hypothesis"}. WER = (S + D + I) / reference words, pooled over utterances."""
    counted = []
    for item in items:
        ref = normalize(item["reference"], spelling)
        counted.append((len(ref), align(ref, normalize(item["hypothesis"], spelling))))
    plain = []
    for item in items:
        ref = plain_normalize(item["reference"])
        plain.append((len(ref), sum(align(ref, plain_normalize(item["hypothesis"])))))
    rows = [(w, sum(e)) for w, e in counted]
    return {"utterances": len(items), "reference_words": sum(w for w, _ in counted),
            "substitutions": sum(e[0] for _, e in counted), "deletions": sum(e[1] for _, e in counted),
            "insertions": sum(e[2] for _, e in counted), "wer": _ratio(rows),
            "wer_ci95": _bootstrap(rows, _ratio, seed, n_boot) if rows else [0.0, 0.0],
            "plain_wer": _ratio(plain),
            "empty_hypotheses": sum(not normalize(i["hypothesis"]) for i in items)}


def paired_wer_delta(baseline: Dict[str, dict], candidate: Dict[str, dict], spelling: dict | None,
                     seed: int = 1, n_boot: int = 1000) -> dict:
    """Candidate WER minus baseline WER on the same utterances, bootstrap over utterances."""
    if set(baseline) != set(candidate):
        raise ValueError("Paired comparison needs the same utterances")
    rows = []
    changed = 0
    for i in sorted(baseline):
        ref = normalize(baseline[i]["reference"], spelling)
        b = sum(align(ref, normalize(baseline[i]["hypothesis"], spelling)))
        c = sum(align(ref, normalize(candidate[i]["hypothesis"], spelling)))
        changed += baseline[i]["hypothesis"] != candidate[i]["hypothesis"]
        rows.append((len(ref), b, c))

    def delta(sample) -> float:
        words = sum(r[0] for r in sample)
        return (sum(r[2] for r in sample) - sum(r[1] for r in sample)) / words if words else 0.0

    return {"wer_delta": delta(rows), "wer_delta_ci95": _bootstrap(rows, delta, seed, n_boot),
            "utterances_changed": changed}


def agreement(reference: Dict[str, str], candidate: Dict[str, str]) -> dict:
    """Candidate transcripts vs the reference model's transcripts of the same audio."""
    if set(reference) != set(candidate):
        raise ValueError("Agreement needs the same utterances")
    ids = sorted(reference)
    exact = sum(reference[i] == candidate[i] for i in ids)
    normalized = sum(plain_normalize(reference[i]) == plain_normalize(candidate[i]) for i in ids)
    words = edits = 0
    for i in ids:
        r = plain_normalize(reference[i])
        words += len(r)
        edits += sum(align(r, plain_normalize(candidate[i])))
    n = len(ids)
    return {"utterances": n, "exact_match_rate": exact / n if n else 0.0,
            "normalized_match_rate": normalized / n if n else 0.0,
            "differing_ids": [i for i in ids if reference[i] != candidate[i]],
            "wer_vs_reference": edits / words if words else 0.0}


# ---------------------------------------------------------------- encoder output error

def encoder_error(ref: Sequence[float], cand: Sequence[float]) -> dict:
    """Relative L2 error ||c - r|| / ||r||, max |c - r| and cosine similarity of one chunk's encoder output
    (both time-major [frames, 1024] float32, valid frames only)."""
    if len(ref) != len(cand):
        raise ValueError(f"encoder outputs differ in size: {len(ref)} vs {len(cand)}")
    err_sq = ref_sq = cand_sq = dot = 0.0
    max_abs = 0.0
    for r, c in zip(ref, cand):
        d = c - r
        err_sq += d * d
        ref_sq += r * r
        cand_sq += c * c
        dot += r * c
        if abs(d) > max_abs:
            max_abs = abs(d)
    rel = math.sqrt(err_sq / ref_sq) if ref_sq > 0 else (0.0 if err_sq == 0 else math.inf)
    cos = dot / math.sqrt(ref_sq * cand_sq) if ref_sq > 0 and cand_sq > 0 else (1.0 if err_sq == 0 else 0.0)
    return {"rel_l2": rel, "max_abs": max_abs, "cosine": cos, "err_sq": err_sq, "ref_sq": ref_sq}


def aggregate_encoder_errors(errors: Sequence[dict]) -> dict:
    if not errors:
        raise ValueError("no encoder outputs compared")
    rels = [e["rel_l2"] for e in errors]
    ref_sq = sum(e["ref_sq"] for e in errors)
    return {"chunks": len(errors), "pooled_rel_l2": math.sqrt(sum(e["err_sq"] for e in errors) / ref_sq),
            "median_rel_l2": median(rels), "p95_rel_l2": quantile(rels, 0.95), "max_rel_l2": max(rels),
            "max_abs": max(e["max_abs"] for e in errors), "min_cosine": min(e["cosine"] for e in errors)}


# ---------------------------------------------------------------- bundles, variants, selection

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_order(buckets: Sequence[int], preferred: str) -> str:
    """The functions a bundle holds, in the preferred comma-separated order (e.g. "15,4,8,30")."""
    order = [b for b in (int(x) for x in preferred.split(",")) if b in buckets]
    if not order:
        raise ValueError(f"load order {preferred} names none of the bundle's functions {list(buckets)}")
    return ",".join(map(str, order))


def verify_bundle(directory: Path) -> dict:
    """Check a LocalFlow bundle as the app does (model id, required assets, every listed file's SHA-256, safe
    relative paths) plus LocalFlow's buckets: 15 s (the one the app loads), optionally with 4, 8 and 30 s. Returns sizes: encoder weight.bin, the Encoder.mlmodelc
    files, all listed files (what the app ships and loads), and everything in the directory (extras included)."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "bundle.json").read_text())
    if manifest.get("model") != "localflow":
        raise ValueError("bundle.json model is not localflow")
    buckets = manifest.get("buckets_seconds")
    if (not isinstance(buckets, list) or 15 not in buckets or buckets != sorted(set(buckets))
            or not set(buckets) <= set(LOCALFLOW_BUCKETS)):
        raise ValueError(f"bundle buckets {buckets} are not b15 plus any of {LOCALFLOW_BUCKETS}")
    files = manifest.get("files") or {}
    missing = [f for f in REQUIRED_FILES if f not in files]
    if missing:
        raise ValueError(f"bundle lacks {missing}")
    listed = encoder = 0
    for name, digest in sorted(files.items()):
        rel = Path(name)
        path = (directory / rel).resolve()
        if rel.is_absolute() or ".." in rel.parts or not str(path).startswith(str(directory) + "/"):
            raise ValueError(f"unsafe bundle path {name}")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"bundle file {name} fails its SHA-256")
        size = path.stat().st_size
        listed += size
        if name.startswith("Encoder.mlmodelc/"):
            encoder += size
    total = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file() and not p.is_symlink())
    return {"manifest": manifest, "bundle_json_sha256": sha256_file(directory / "bundle.json"),
            "encoder_weight_bytes": (directory / "Encoder.mlmodelc/weights/weight.bin").stat().st_size,
            "encoder_bytes": encoder, "listed_bytes": listed, "directory_bytes": total}


def parse_variant(text: str) -> Tuple[str, str, str]:
    """"NAME@UNITS" -> (name, unit code, MLComputeUnits name); ext marks an external (non-Core-ML) encoder."""
    parts = text.split("@")
    if len(parts) != 2 or not parts[0] or parts[1] not in UNITS:
        raise ValueError(f"variant {text!r} is not NAME@{{{','.join(UNITS)}}}")
    return parts[0], parts[1], UNITS[parts[1]]


def select_subset(items: Sequence[dict], n: int, seed: int = 0) -> List[dict]:
    ordered = sorted(items, key=lambda i: i["id"])
    if n >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, n), key=lambda i: i["id"])


def select_chunks(rows: Sequence[dict], per_bucket: int) -> Dict[int, List[str]]:
    """The first per_bucket non-silent chunk keys of each bucket, in input order."""
    out: Dict[int, List[str]] = {}
    for r in rows:
        if r.get("silent"):
            continue
        keys = out.setdefault(r["bucket"], [])
        if len(keys) < per_bucket:
            keys.append(r["key"])
    return dict(sorted(out.items()))


# ---------------------------------------------------------------- memory, caches, reports

def memory_verdict(candidate: int, ternary: int) -> dict:
    return {"candidate_bytes": candidate, "ternary_bytes": ternary, "delta_bytes": candidate - ternary,
            "ratio": candidate / ternary if ternary else math.inf, "less_than_ternary": candidate < ternary}


def cache_delta(before: Dict[str, int], after: Dict[str, int]) -> dict:
    new = [p for p in after if p not in before]
    changed = [p for p in after if p in before and after[p] != before[p]]
    added = sum(after[p] for p in new) + sum(max(0, after[p] - before[p]) for p in changed)
    return {"new_files": len(new), "changed_files": len(changed), "added_bytes": added,
            "removed_files": sum(p not in after for p in before)}


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    def cell(v) -> str:
        return "-" if v is None else str(v)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"
