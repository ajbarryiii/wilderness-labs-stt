"""The final build: global denylist first, then the subsets (one command: `./python -m pool finalize`).

finalize() is the whole local computation from its inputs; the CLI (pool/__main__.py) supplies the
production inputs: the text caches, every held-out manifest (parakeet-ternary dev sets and test_*.jsonl,
text only; every manifest under $RELABEL_ROOT/devsets/, which holds Part A's domain dev sets and later
held-out sets such as eval-ext; and any --heldout path), the reserved YODAS recordings, the local corpora and
the committed term lists. Every held-out manifest used is recorded with its SHA-256 in denylist.json. A final
build refuses to run unless every domain-dev part is present.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from pool import common, denylist, scan, select, terms_match
from pool import heldout as ho

DOMAIN_PARTS = ("med-dev-1", "med-dev-2", "med-dev-3", "tech-dev-1", "tech-dev-2")


def finalize(out_dir: Path, caches: Mapping[str, Path], heldout: Sequence[Path | tuple[str, Path]],
             reserved: Iterable[str],
             corpora: Mapping[str, Path], term_loader: Callable = terms_match.load_terms, workers: int = 0,
             final: bool = True, required_parts: Sequence[str] = DOMAIN_PARTS, notes: dict | None = None) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    sets = ho.load_sets(heldout)
    parts = set().union(*(s.parts for s in sets)) if sets else set()
    missing = [p for p in required_parts if p not in parts]
    if final and missing:
        raise RuntimeError(f"final build: domain dev parts {missing} are not in any held-out manifest")
    matchers = terms_match.matchers(term_loader)
    index = ho.Index(sets)
    common.log(f"finalize: {len(sets)} held-out sets, {len(index)} utterances, {len(index.hashes)} 8-grams")
    res = scan.scan(caches, corpora, index.hashes, matchers, workers)
    common.log(f"finalize: scanned {len(res.keys)} recordings, {len(res.pairs)} held-out 8-gram hits")
    deny, details = denylist.build(res, index, reserved)
    deny = {"final": final and not missing, "created": common.now(), "domain_parts_missing": missing,
            "terms_sha256": {k: m.sha256 for k, m in matchers.items()},
            "training": res.sources, "notes": notes or {}, **deny}
    common.atomic_write_json(out_dir / "denylist.json", deny)
    common.atomic_write_lines(out_dir / "denylist_details.jsonl", (json.dumps(d, ensure_ascii=False) for d in details))
    denied = {(s, r) for s, recs in deny["recordings"].items() for r in recs}
    subsets, excluded, table = select.select(res, denied)
    summ = select.write_outputs(out_dir, subsets, excluded, table, caches)
    summary = {"final": deny["final"], "created": common.now(), "seconds": round(time.time() - t0, 1),
               "denylist": {"counts": deny["counts"],
                            "heldout": [{k: v for k, v in s.items() if k != "path"} for s in deny["heldout"]],
                            "sha256": common.sha256_file(out_dir / "denylist.json")},
               "subsets": summ, "training": res.sources,
               "terms": {k: {"terms": len(m.terms), "sha256": m.sha256} for k, m in matchers.items()},
               "thresholds": {"medical_per_1k": common.MED_THRESHOLD, "technical_per_1k": common.TECH_THRESHOLD,
                              "min_words": common.MIN_WORDS}, "notes": notes or {}}
    common.atomic_write_json(out_dir / "summary.json", summary)
    return summary
