"""Subset include lists, per-recording table and the user sample (labeling plan Part B).

A recording of a streamed source with at least MIN_WORDS normalized words joins
- yodas-med / ps-med if its medical density is >= MED_THRESHOLD per 1,000 words;
- yodas-tech (yodas only) if not medical and its technical density is >= TECH_THRESHOLD;
unless it is on the global denylist (counted as `denylisted_qualifying`).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from pool import common, fetch
from pool.scan import ScanResult
from pool.terms_match import density


def classify(source: str, words: int, medical: int, technical: int) -> str | None:
    if words < common.MIN_WORDS:
        return None
    if density(medical, words) >= common.MED_THRESHOLD:
        return {"yodas": "yodas-med", "peoples_speech": "ps-med"}.get(source)
    if source == "yodas" and density(technical, words) >= common.TECH_THRESHOLD:
        return "yodas-tech"
    return None


def select(scan: ScanResult, denied: set[tuple[str, str]]) -> tuple[dict[str, list[dict]], dict, list[dict]]:
    """(include rows per subset, exclusion counts, one row per scanned recording of a streamed source)."""
    subsets: dict[str, list[dict]] = {name: [] for name in common.SUBSETS}
    under = defaultdict(int)
    denied_q = defaultdict(int)
    table = []
    streamed = {src for src, _ in common.SUBSETS.values()}
    for gid, (source, rec) in enumerate(scan.keys):
        if source not in streamed:
            continue
        s = scan.stats[gid]
        name = classify(source, s["words"], s["medical"], s["technical"])
        is_denied = (source, rec) in denied
        if s["words"] < common.MIN_WORDS:
            under[source] += 1
        row = {"source": source, "recording": rec, "rows": s["rows"], "words": s["words"],
               "medical": s["medical"], "technical": s["technical"],
               "medical_per_1k": round(density(s["medical"], s["words"]), 4),
               "technical_per_1k": round(density(s["technical"], s["words"]), 4),
               "hours": s["seconds"] / 3600, "hours_1_30s": s["seconds_kept"] / 3600}
        table.append(row | {"denylisted": is_denied, "subset": None if is_denied else name})
        if name is None:
            continue
        if is_denied:
            denied_q[name] += 1
            continue
        locs = [{"file": f, "row_groups": sorted(g)} for f, g in sorted(scan.locators[gid].items())]
        subsets[name].append(row | {"locators": locs})
    for rows in subsets.values():
        rows.sort(key=lambda r: r["recording"])
    excluded = {"under_min_words": dict(sorted(under.items())), "denylisted_qualifying": dict(sorted(denied_q.items()))}
    return subsets, excluded, table


def subset_summary(subsets: Mapping[str, list[dict]]) -> dict:
    return {name: {"recordings": len(rows), "words": sum(r["words"] for r in rows),
                   "hours": sum(r["hours"] for r in rows), "hours_1_30s": sum(r["hours_1_30s"] for r in rows),
                   "source": common.SUBSETS[name][0]}
            for name, rows in subsets.items()}


def _first_words(rec: str, locators: list[dict], cache_dir: Path, n: int) -> list[str]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    rows = []
    for loc in locators:
        path, _ = fetch.cache_paths(cache_dir, {"path": loc["file"]})
        t = pq.read_table(path, columns=["recording", "order", "row", "text"])
        t = t.filter(pc.equal(t["recording"], rec))
        rows += [(r["order"], loc["file"], r["row"], r["text"]) for r in t.to_pylist()]
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    words = []
    for *_, text in rows:
        words += (text or "").split()
        if len(words) >= n:
            break
    return words[:n]


def user_sample(name: str, rows: list[dict], cache_dir: Path) -> str:
    """Seeded sample of SAMPLE_RECORDINGS recordings (all if fewer) with their first SAMPLE_WORDS words."""
    k = min(common.SAMPLE_RECORDINGS, len(rows))
    picked = random.Random(f"{common.SAMPLE_SEED}:{name}").sample(sorted(rows, key=lambda r: r["recording"]), k)
    out = [f"# {name}: {k} of {len(rows)} recordings (seed {common.SAMPLE_SEED}), first {common.SAMPLE_WORDS} "
           "words of each transcript"]
    for r in picked:
        words = _first_words(r["recording"], r["locators"], cache_dir, common.SAMPLE_WORDS)
        out.append(f"\n## {r['recording']} | {r['hours']:.3f} h | medical {r['medical_per_1k']:.1f}/1k | "
                   f"technical {r['technical_per_1k']:.1f}/1k | {r['words']} words\n{' '.join(words)}\n")
    return "\n".join(out)


def write_outputs(out_dir: Path, subsets: Mapping[str, list[dict]], excluded: dict, table: list[dict],
                  caches: Mapping[str, Path]) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    sub_dir = out_dir / "subsets"
    summary = subset_summary(subsets)
    for name, rows in subsets.items():
        path = sub_dir / f"{name}.jsonl"
        common.atomic_write_lines(path, (json.dumps(r, ensure_ascii=False) for r in rows))
        summary[name]["sha256"] = common.sha256_file(path)
        source = common.SUBSETS[name][0]
        sample = user_sample(name, rows, Path(caches[source])) if rows else user_sample(name, [], Path("."))
        common.atomic_write_bytes(out_dir / "sample" / f"{name}.txt", sample.encode())
    summary["excluded"] = excluded
    common.atomic_write_json(sub_dir / "summary.json", summary)
    if table:
        cols = {k: [r[k] for r in table] for k in table[0]}
        import io
        buf = io.BytesIO()
        pq.write_table(pa.table(cols), buf, compression="zstd")
        common.atomic_write_bytes(out_dir / "recordings.parquet", buf.getvalue())
    return summary
