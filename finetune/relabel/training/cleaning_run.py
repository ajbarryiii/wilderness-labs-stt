"""Label cleaning runner (cleaning plan "Layer 2", "Checks"): spans, choose, assemble, check, audit.

    ./python -m training.cleaning_run spans --labels-run RUN [--workers N]          CPU
    ./python -m training.cleaning_run choose --labels-run RUN [--engine vllm|transformers] [--probe N]
                                     [--budget-minutes M] [--declared-vram-gb G] [--determinism F]   GPU
    ./python -m training.cleaning_run assemble --labels-run RUN                     CPU
    ./python -m training.cleaning_run check --labels-run RUN [--layer1-only] [--workers N]          CPU
    ./python -m training.cleaning_run audit --labels-run RUN [--workers N]          CPU (architect's sheets)

Everything is written under $RELABEL_ROOT/cleaning/<RUN>/v<VERSION>/ (training/cleaning.production_dir), the
architect's sheets under $RELABEL_ROOT/cleaning/audit/:

    spans.parquet       one row per layer 2 span of a kept label of either label set (label_set, id, span, start,
                        end, original, set, prompt_key), over every label shard of the run's manifest
    prompts.parquet     the unique prompts (prompt_key, set, text, start, end, user): dedup by the hash of (template,
                        label text, span, options), so identical texts of T* and B0 share one decision
    spans.json          counts per set and label set, the unique-prompt count, and what the spans were found under
                        (tables, template, label run contract)
    decisions/<n>.parquet, <n>.json
                        choose's output blocks (prompt_key, probs over the lettered options with "can't tell" last,
                        renormalized) and their meta (engine, model alias and revision, seconds, peak memory),
                        written atomically; choose resumes with the prompts no block has decided yet
    probe.json          choose --probe N: throughput, memory, the extrapolated time for every unique prompt
    determinism.json    choose --determinism F: a re-score of a fixed fraction F of the decided prompts
    choices.parquet, manifest.json
                        the artifact LabelStore reads (training/cleaning.py), written by assemble (choose runs it
                        once every prompt is decided)
    check.json          Check 3 (net harm per set, variant group and dialect entry; corpus WER of the
                        human-reference labels before and after) and Check 5 (spans found, chosen, left per set;
                        labels changed per source and label set)

choose scores each prompt with one prefill: the probabilities of the option letters at the first answer position,
renormalized over the options ("can't tell" included); cleaning.decide picks a spelling only if it is not "can't
tell" and has probability >= cleaning.THRESHOLD. The engine is an object with load(), info() and score(users,
n_options) (tests inject a fake); the real engines (VllmScorer, TransformersScorer) are at the end of this module
and only run on the GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from training import cleaning as C

LABEL_SETS = {"T*": "text", "B0": "b0_raw"}       # training/labels.py LABEL_SETS (not imported: no pyarrow there)
BLOCK = 4096
PROBE_DEFAULT = 2000
AUDIT_SAMPLE = 200
AUDIT_SEED = "cleaning-v1-audit"
MIN_JUDGED = 30
FORMS_AUDIT_MATCHES = 30
CONTEXT = 70
DECISION_KEYS = ("alias", "revision", "engine")   # a block's engine identity; every block must agree


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [cleaning] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_parquet(path: Path, columns: dict, schema=None) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    pq.write_table(pa.table(columns, schema=schema), tmp, compression="zstd")
    os.replace(tmp, path)
    return C._sha256(path.read_bytes())


# ----------------------------------------------------------------------------- the label run

class Run:
    """A finished label run: its manifest's shards, contract and pool (labeling/c2.py)."""

    def __init__(self, run_dir: Path, sample_shards: int | None = None):
        from labeling import c2
        self.dir = Path(run_dir)
        self.run = c2.load_run(self.dir)
        manifest = json.loads((self.dir / "manifest.json").read_text())
        if not manifest.get("complete"):
            raise RuntimeError(f"{self.dir}: the label run is not complete")
        self.shards = [(s["shard"], s["file"], s["sha256"], s["records"]) for s in manifest["shards"]]
        self.sampled = bool(sample_shards)
        if sample_shards:                    # evenly spaced shards: measurement runs only, never an artifact
            self.shards = self.shards[::max(1, len(self.shards) // sample_shards)][:sample_shards]
        self.contract_sha256 = hashlib.sha256(json.dumps(c2.contract(self.run), sort_keys=True).encode()).hexdigest()
        self.pool_dir = c2._abs(self.run["pool_dir"])
        self.rel = c2._rel(self.dir)

    def read(self, shard: tuple, columns: tuple) -> dict:
        import pyarrow.parquet as pq
        k, name, sha, records = shard
        path = self.dir / "shards" / name
        if C._sha256(path.read_bytes()) != sha:
            raise RuntimeError(f"{path}: SHA-256 differs from the run's manifest")
        t = pq.read_table(path, columns=list(columns)).to_pydict()
        if len(t["id"]) != records:
            raise RuntimeError(f"{path}: {len(t['id'])} records, the manifest says {records}")
        return t

    def references(self, k: int) -> dict[str, str]:
        """{id: human reference} of C1 shard k (the shard whose label records these are)."""
        from labeling import c2
        return {r["id"]: r["reference"] for r in c2.load_group(self.pool_dir, [k], with_audio=False)
                if r["reference_kind"] == "human" and r["reference"]}


def kept_texts(t: dict):
    """(id, source, reference_kind, {label set: text}) of every kept record (LabelStore's rule)."""
    for k, i in enumerate(t["id"]):
        if t["disposition"][k] != "kept":
            continue
        texts = {}
        for ls, col in LABEL_SETS.items():
            v = t[col][k]
            if v is None:
                raise RuntimeError(f"{i}: kept by T* but no {col}")
            texts[ls] = v
        yield i, t["source"][k], t["reference_kind"][k], texts


LABEL_COLUMNS = ("id", "disposition", "source", "reference_kind", "text", "b0_raw")


def _pool_map(fn, items, workers: int, init_args: tuple):
    if workers <= 1:
        _init(*init_args)
        for x in items:
            yield fn(x)
        return
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers, initializer=_init, initargs=init_args) as pool:
        yield from pool.imap(fn, items, chunksize=4)


_W: dict = {}


def _init(run_dir, tables_dir, with_spec: bool = False, choices_dir=None) -> None:
    """Per worker: the run, the tables and (with_spec) the cleaning spec with the choices artifact, if any."""
    _W["run"] = Run(run_dir)
    _W["tables"] = C.Tables.load(tables_dir)
    _W["spec"] = None if not with_spec else C.Cleaning(
        _W["tables"], None if choices_dir is None else C.Choices.load(choices_dir))


# ----------------------------------------------------------------------------- spans

def _spans_shard(shard):
    run, tables = _W["run"], _W["tables"]
    t = run.read(shard, LABEL_COLUMNS)
    rows, prompts = [], {}
    kept = Counter()
    with_span = Counter()
    for i, _source, _rk, texts in kept_texts(t):
        for ls, text in texts.items():
            kept[ls] += 1
            spans = tables.spans(text)
            if spans:
                with_span[ls] += 1
            for sp in spans:
                key = C.prompt_key(text, sp.start, sp.end, sp.set.options)
                rows.append((ls, i, sp.index, sp.start, sp.end, sp.original, sp.set.name, key))
                if key not in prompts:
                    prompts[key] = (sp.set.name, text, sp.start, sp.end)
    return rows, prompts, kept, with_span


def spans(run_dir: Path, out: Path, workers: int = 1, tables_dir: Path | None = None,
          sample_shards: int | None = None) -> dict:
    """Every layer 2 span of the run's kept labels, both label sets, and the unique prompts (CPU)."""
    t0 = time.time()
    run, tables = Run(run_dir, sample_shards), C.Tables.load(tables_dir)
    rows, prompts = [], {}
    kept, with_span = Counter(), Counter()
    for n, (r, p, k, w) in enumerate(_pool_map(_spans_shard, run.shards, workers, (run_dir, tables_dir))):
        rows += r
        for key, v in p.items():
            prompts.setdefault(key, v)
        kept.update(k)
        with_span.update(w)
        if (n + 1) % 200 == 0:
            log(f"spans: {n + 1}/{len(run.shards)} shards, {len(rows)} spans, {len(prompts)} unique prompts")
    rows.sort()
    cols = list(zip(*rows)) if rows else [()] * 8
    write_parquet(out / "spans.parquet", {
        "label_set": list(cols[0]), "id": list(cols[1]), "span": list(cols[2]), "start": list(cols[3]),
        "end": list(cols[4]), "original": list(cols[5]), "set": list(cols[6]), "prompt_key": list(cols[7])})
    sets = {cs.name: cs for cs in tables.sets}
    keys = sorted(prompts)
    write_parquet(out / "prompts.parquet", {
        "prompt_key": keys, "set": [prompts[k][0] for k in keys], "text": [prompts[k][1] for k in keys],
        "start": [prompts[k][2] for k in keys], "end": [prompts[k][3] for k in keys],
        "user": [C.prompt_user(prompts[k][1], prompts[k][2], prompts[k][3], sets[prompts[k][0]].options)
                 for k in keys]})
    per_set = {cs.name: {ls: 0 for ls in LABEL_SETS} for cs in tables.sets}
    for r in rows:
        per_set[r[6]][r[0]] += 1
    unique_per_set = Counter(v[0] for v in prompts.values())
    report = {"version": C.VERSION, "label_run": run.rel, "contract_sha256": run.contract_sha256,
              "tables_sha256": tables.sha256, "template_sha256": C.TEMPLATE_SHA256, "shards": len(run.shards),
              "sampled_shards": run.sampled,
              "kept": dict(kept), "labels_with_span": dict(with_span), "spans": len(rows),
              "spans_per_label_set": {ls: sum(v[ls] for v in per_set.values()) for ls in LABEL_SETS},
              "spans_per_set": per_set, "unique_prompts": len(prompts),
              "unique_prompts_per_set": {cs.name: unique_per_set.get(cs.name, 0) for cs in tables.sets},
              "seconds": round(time.time() - t0, 1)}
    write_json(out / "spans.json", report)
    return report


# ----------------------------------------------------------------------------- choose

def _read_prompts(work: Path, tables: C.Tables) -> list[dict]:
    import pyarrow.parquet as pq
    meta = json.loads((work / "spans.json").read_text())
    for key, want in (("tables_sha256", tables.sha256), ("template_sha256", C.TEMPLATE_SHA256),
                      ("version", C.VERSION)):
        if meta.get(key) != want:
            raise RuntimeError(f"{work}/spans.json was written under {key} {meta.get(key)!r}, not {want!r}; "
                               "re-run spans")
    sets = {cs.name: cs for cs in tables.sets}
    t = pq.read_table(work / "prompts.parquet").to_pylist()
    for p in t:
        p["n_options"] = len(sets[p["set"]].options)
    return t


def _decided(work: Path) -> tuple[dict, list[dict]]:
    import pyarrow.parquet as pq
    probs, metas = {}, []
    d = work / "decisions"
    for meta_path in sorted(d.glob("*.json")) if d.exists() else []:
        meta = json.loads(meta_path.read_text())
        path = meta_path.with_suffix(".parquet")
        if C._sha256(path.read_bytes()) != meta["sha256"]:
            raise RuntimeError(f"{path}: SHA-256 differs from its meta")
        t = pq.read_table(path).to_pydict()
        for k, p in zip(t["prompt_key"], t["probs"]):
            probs[k] = p
        metas.append(meta)
    return probs, metas


def _engine_identity(info: dict) -> dict:
    return {k: info.get(k) for k in DECISION_KEYS}


def _peak(scorer) -> dict:
    out = {"host_maxrss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20}
    if hasattr(scorer, "peak"):
        out.update(scorer.peak())
    return out


def _score(scorer, batch: list[dict]) -> list[list[float]]:
    got = scorer.score([p["user"] for p in batch], [p["n_options"] for p in batch])
    if len(got) != len(batch):
        raise RuntimeError(f"the scorer returned {len(got)} results for {len(batch)} prompts")
    for p, probs in zip(batch, got):
        if len(probs) != p["n_options"] or not all(x >= 0 for x in probs) or abs(sum(probs) - 1) > 1e-3:
            raise RuntimeError(f"prompt {p['prompt_key']}: bad probabilities {probs}")
    return [[float(x) for x in probs] for probs in got]


def probe(work: Path, scorer, n: int = PROBE_DEFAULT, tables_dir: Path | None = None) -> dict:
    """Score n prompts spread over the list (no decisions written): throughput, memory, extrapolated time."""
    prompts = _read_prompts(work, C.Tables.load(tables_dir))
    step = max(1, len(prompts) // max(1, n))
    sample = prompts[::step][:n]
    scorer.load()
    t0 = time.time()
    _score(scorer, sample)
    secs = time.time() - t0
    rate = len(sample) / secs if secs > 0 else None
    report = {"prompts": len(sample), "seconds": round(secs, 2), "prompts_per_second": rate,
              "unique_prompts": len(prompts),
              "extrapolated_hours": None if not rate else len(prompts) / rate / 3600,
              "engine": scorer.info(), "peak": _peak(scorer)}
    write_json(work / "probe.json", report)
    return report


def choose(work: Path, scorer, budget_minutes: float | None = None, block: int = BLOCK,
           tables_dir: Path | None = None, assemble_when_done: bool = True, run_dir: Path | None = None) -> dict:
    """Score every unique prompt not yet decided, block by block (resumable); assemble once all are decided."""
    tables = C.Tables.load(tables_dir)
    prompts = _read_prompts(work, tables)
    done, metas = _decided(work)
    todo = [p for p in prompts if p["prompt_key"] not in done]
    log(f"choose: {len(prompts)} unique prompts, {len(prompts) - len(todo)} decided, {len(todo)} to score")
    t0 = time.time()
    if todo:
        scorer.load()
        ident = _engine_identity(scorer.info())
        for m in metas:
            if _engine_identity(m["engine"]) != ident:
                raise RuntimeError(f"decision block {m['block']} was scored by {_engine_identity(m['engine'])}, "
                                   f"this engine is {ident}")
        nxt = 1 + max((m["block"] for m in metas), default=-1)
        for s in range(0, len(todo), block):
            if budget_minutes is not None and time.time() - t0 > budget_minutes * 60:
                log("choose: budget reached; the next run resumes")
                break
            batch = todo[s:s + block]
            b0 = time.time()
            probs = _score(scorer, batch)
            path = work / "decisions" / f"{nxt:05d}.parquet"
            sha = write_parquet(path, {"prompt_key": [p["prompt_key"] for p in batch], "probs": probs})
            write_json(path.with_suffix(".json"), {
                "block": nxt, "prompts": len(batch), "sha256": sha, "seconds": round(time.time() - b0, 2),
                "engine": scorer.info(), "template_sha256": C.TEMPLATE_SHA256, "peak": _peak(scorer)})
            nxt += 1
            log(f"choose: block {nxt - 1} ({len(batch)} prompts) in {time.time() - b0:.0f} s")
        done, metas = _decided(work)
    left = sum(p["prompt_key"] not in done for p in prompts)
    out = {"unique_prompts": len(prompts), "decided": len(prompts) - left, "left": left}
    if left == 0 and assemble_when_done:
        out["manifest"] = assemble(work, run_dir, tables_dir)
    return out


def determinism(work: Path, scorer, fraction: float = 0.01, tables_dir: Path | None = None) -> dict:
    """Check 2: re-score a fixed fraction of the decided prompts and compare the decisions."""
    tables = C.Tables.load(tables_dir)
    prompts = _read_prompts(work, tables)
    done, _ = _decided(work)
    cut = int(fraction * 2 ** 32)
    sample = [p for p in prompts if p["prompt_key"] in done and int(p["prompt_key"][:8], 16) < cut]
    scorer.load()
    probs = _score(scorer, sample)
    sets = {cs.name: cs for cs in tables.sets}
    same, max_diff = 0, 0.0
    for p, q in zip(sample, probs):
        opts = sets[p["set"]].options
        same += C.decide(done[p["prompt_key"]], opts) == C.decide(q, opts)
        max_diff = max(max_diff, max(abs(a - b) for a, b in zip(done[p["prompt_key"]], q)))
    report = {"fraction": fraction, "prompts": len(sample), "identical_decisions": same,
              "all_identical": same == len(sample), "max_probability_difference": max_diff, "engine": scorer.info()}
    write_json(work / "determinism.json", report)
    return report


# ----------------------------------------------------------------------------- assemble

def assemble(work: Path, run_dir: Path | None, tables_dir: Path | None = None) -> dict:
    """choices.parquet + manifest.json from spans.parquet and the decision blocks."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    tables = C.Tables.load(tables_dir)
    meta = json.loads((work / "spans.json").read_text())
    _read_prompts(work, tables)                                   # same tables and template as the spans
    if run_dir is not None and Run(run_dir).contract_sha256 != meta["contract_sha256"]:
        raise RuntimeError("the spans were found in another label run")
    done, metas = _decided(work)
    idents = {json.dumps(_engine_identity(m["engine"]), sort_keys=True) for m in metas}
    if len(idents) > 1:
        raise RuntimeError(f"decision blocks of different engines: {sorted(idents)}")
    sp = pq.read_table(work / "spans.parquet").to_pydict()
    sets = {cs.name: cs for cs in tables.sets}
    options, probs, chosen = [], [], []
    for name, key in zip(sp["set"], sp["prompt_key"]):
        if key not in done:
            raise RuntimeError(f"prompt {key} is not decided yet")
        opts = sets[name].options
        options.append(list(opts))
        probs.append(done[key])
        chosen.append(C.decide(done[key], opts))
    schema = pa.schema([("label_set", pa.string()), ("id", pa.string()), ("span", pa.int32()),
                        ("start", pa.int32()), ("end", pa.int32()), ("original", pa.string()), ("set", pa.string()),
                        ("options", pa.list_(pa.string())), ("probs", pa.list_(pa.float64())),
                        ("chosen", pa.string())])
    sha = write_parquet(work / "choices.parquet", {
        "label_set": sp["label_set"], "id": sp["id"], "span": sp["span"], "start": sp["start"], "end": sp["end"],
        "original": sp["original"], "set": sp["set"], "options": options, "probs": probs, "chosen": chosen},
        schema=schema)
    engine = metas[0]["engine"] if metas else {}
    manifest = {"version": C.VERSION, "label_run": meta["label_run"], "contract_sha256": meta["contract_sha256"],
                "tables_sha256": tables.sha256, "template_sha256": C.TEMPLATE_SHA256, "threshold": C.THRESHOLD,
                "model": {"alias": engine.get("alias"), "revision": engine.get("revision")}, "engine": engine,
                "rows": len(sp["id"]), "file_sha256": sha, "unique_prompts": len(done),
                "gpu_seconds": round(sum(m["seconds"] for m in metas), 1), "blocks": len(metas)}
    write_json(work / "manifest.json", manifest)
    return manifest


# ----------------------------------------------------------------------------- check (net harm)

def _tokens(normalize, text: str) -> list[str]:
    return normalize(text).split()


def span_tokens(normalize, text: str, start: int, end: int) -> tuple[list[str], int, int] | None:
    """The normalized tokens of text and the token range [a, b) of its characters [start, end); None when the
    span's words cannot be cut out of the normalized text (prefix + span + suffix differ from the whole)."""
    full = _tokens(normalize, text)
    pre, mid, post = _tokens(normalize, text[:start]), _tokens(normalize, text[start:end]), \
        _tokens(normalize, text[end:])
    if not mid or pre + mid + post != full:
        return None
    return full, len(pre), len(pre) + len(mid)


def side_matches(ref: list[str], hyp: list[str], a: int, b: int) -> bool:
    """Whether hyp[a:b] equals the reference tokens aligned to it: ref[lo..hi], lo and hi the first and last
    reference positions aligned (match or substitution) to a token of hyp[a:b], deletions between them included;
    no aligned reference token (all insertions) never matches."""
    import relabel_align
    al = relabel_align.align(ref, hyp)
    idx = [r for kind, r, h in al.pairs if kind in ("eq", "sub") and a <= h < b]
    if not idx:
        return False
    return ref[min(idx):max(idx) + 1] == hyp[a:b]


def screen_edit(normalize, ref: list[str], before: str, after: str, e: C.Edit) -> tuple[bool, bool] | None:
    """(original side matches, cleaned side matches) for one changed span against the aligned reference tokens,
    each side aligned in its own full label; None when a side's span cannot be mapped to normalized tokens."""
    b = span_tokens(normalize, before, e.start, e.end)
    a = span_tokens(normalize, after, e.out_start, e.out_end)
    if b is None or a is None:
        return None
    return side_matches(ref, b[0], b[1], b[2]), side_matches(ref, a[0], a[1], a[2])


def _group_key(e: C.Edit) -> str:
    return f"{e.table}:{e.group}"


def _rank(*parts) -> str:
    return hashlib.sha256("|".join(map(str, (AUDIT_SEED,) + parts)).encode()).hexdigest()


def _check_shard(shard):
    import evaluate
    run, spec = _W["run"], _W["spec"]
    t = run.read(shard, LABEL_COLUMNS)
    refs = None
    out = {"changed": Counter(), "labels": Counter(), "edits": Counter(), "screen": defaultdict(Counter),
           "wer": Counter(), "sample": []}
    for i, source, rk, texts in kept_texts(t):
        for ls, text in texts.items():
            out["labels"][ls] += 1
            after, edits = spec.edits(text, i, ls)
            if after != text:
                out["changed"][f"{ls}\t{source}"] += 1
            for n, e in enumerate(edits):
                out["edits"][f"{ls}\t{_group_key(e)}"] += 1
                out["sample"].append((_rank(ls, i, n), ls, i, e.table, e.group, e.original, e.replacement,
                                      text[max(0, e.start - CONTEXT):e.end + CONTEXT]))
            if rk != "human":
                continue
            if refs is None:
                refs = run.references(shard[0])
            ref_text = refs.get(i)
            if not ref_text:
                continue
            ref = _tokens(evaluate.normalize, ref_text)
            hb = _tokens(evaluate.normalize, text)
            s, d, ins = evaluate.edit_counts(ref, hb)
            out["wer"][f"{ls}\tbefore_errors"] += s + d + ins
            out["wer"][f"{ls}\twords"] += len(ref)
            out["wer"][f"{ls}\tlabels"] += 1
            if after == text:
                out["wer"][f"{ls}\tafter_errors"] += s + d + ins
                continue
            s2, d2, i2 = evaluate.edit_counts(ref, _tokens(evaluate.normalize, after))
            out["wer"][f"{ls}\tafter_errors"] += s2 + d2 + i2
            for e in edits:
                if e.table == "forms":
                    continue
                g = out["screen"][f"{ls}\t{_group_key(e)}"]
                g["changed"] += 1
                r = screen_edit(evaluate.normalize, ref, text, after, e)
                if r is None:
                    g["unmapped"] += 1
                    continue
                g["aligned"] += 1
                g["original_matches"] += r[0]
                g["cleaned_matches"] += r[1]
    out["sample"] = sorted(out["sample"])[:AUDIT_SAMPLE]
    return out


def judge(screen: dict[str, dict[str, Counter]]) -> dict:
    """Check 3's decision per group: judged on every cell (each label set, and both pooled) with at least
    MIN_JUDGED aligned changed spans; disabled when a judged cell's original matches more often than its cleaned
    output."""
    out = {}
    for group, cells in sorted(screen.items()):
        pooled = Counter()
        for c in cells.values():
            pooled.update(c)
        allc = dict(cells) | {"both": pooled}
        judged = {k: c for k, c in allc.items() if c["aligned"] >= MIN_JUDGED}
        harm = sorted(k for k, c in judged.items() if c["original_matches"] > c["cleaned_matches"])
        out[group] = {"cells": {k: dict(c) for k, c in allc.items()}, "judged": sorted(judged),
                      "harm_in": harm, "disabled": bool(harm), "reported_not_judged": not judged}
    return out


def check(run_dir: Path, work: Path, out_dir: Path | None = None, layer1_only: bool = False, workers: int = 1,
          tables_dir: Path | None = None, sample_shards: int | None = None) -> dict:
    """Checks 3 and 5 over every kept label of the run (CPU)."""
    t0 = time.time()
    run = Run(run_dir, sample_shards)
    choices_dir = None if layer1_only else work
    spec = C.Cleaning(C.Tables.load(tables_dir), None if choices_dir is None else C.Choices.load(choices_dir))
    spec.bind_run(run.contract_sha256)
    agg = {"changed": Counter(), "labels": Counter(), "edits": Counter(), "screen": defaultdict(Counter),
           "wer": Counter()}
    sample = []
    for n, part in enumerate(_pool_map(_check_shard, run.shards, workers, (run_dir, tables_dir, True, choices_dir))):
        for k in ("changed", "labels", "edits", "wer"):
            agg[k].update(part[k])
        for g, c in part["screen"].items():
            agg["screen"][g].update(c)
        sample = sorted(sample + part["sample"])[:AUDIT_SAMPLE]
        if (n + 1) % 200 == 0:
            log(f"check: {n + 1}/{len(run.shards)} shards")
    screen: dict[str, dict[str, Counter]] = defaultdict(dict)
    for key, c in agg["screen"].items():
        ls, group = key.split("\t", 1)
        screen[group][ls] = c
    verdict = judge(screen)
    wer = {}
    for ls in LABEL_SETS:
        words = agg["wer"][f"{ls}\twords"]
        wer[ls] = {"labels": agg["wer"][f"{ls}\tlabels"], "words": words,
                   "before": agg["wer"][f"{ls}\tbefore_errors"] / words if words else None,
                   "after": agg["wer"][f"{ls}\tafter_errors"] / words if words else None}
    changed = defaultdict(dict)
    for key, v in agg["changed"].items():
        ls, source = key.split("\t", 1)
        changed[ls][source] = v
    edits = defaultdict(dict)
    for key, v in agg["edits"].items():
        ls, group = key.split("\t", 1)
        edits[group][ls] = v
    report = {"version": C.VERSION, "label_run": run.rel, "tables_sha256": spec.tables.sha256,
              "choices_sha256": spec.identity()["choices_sha256"], "layer1_only": layer1_only,
              "check3": {"min_judged": MIN_JUDGED, "groups": verdict,
                         "disabled": sorted(g for g, v in verdict.items() if v["disabled"]),
                         "note": "a screen for net harm; it neither establishes calibrated correctness nor rules out "
                                 "harm below 1%"},
              "corpus_wer_human_reference_labels": wer,
              "check5": {"labels": dict(agg["labels"]), "labels_changed": dict(changed), "edits_per_group": edits,
                         "layer2": None if layer1_only else layer2_counts(work)},
              "seconds": round(time.time() - t0, 1)}
    if not layer1_only:
        manifest = json.loads((work / "manifest.json").read_text())
        probe_path = work / "probe.json"
        report["check5"]["gpu_seconds"] = {"run": manifest.get("gpu_seconds"), "probe": json.loads(
            probe_path.read_text())["seconds"] if probe_path.exists() else None}
    out_dir = work if out_dir is None else Path(out_dir)
    write_json(out_dir / ("check-layer1.json" if layer1_only else "check.json"), report)
    audit = C.audit_dir()
    audit.mkdir(parents=True, exist_ok=True)
    name = "check-sample-layer1.tsv" if layer1_only else "check-sample.tsv"
    with open(audit / name, "w", encoding="utf-8") as f:
        f.write("label_set\tid\ttable\tgroup\toriginal\treplacement\twrong (architect)\tcontext\n")
        for _, ls, i, table, group, o, r, ctx in sample:
            f.write("\t".join((ls, i, table, group, o, r, "", ctx.replace("\t", " ").replace("\n", " "))) + "\n")
    return report


def layer2_counts(work: Path) -> dict:
    """Check 5: per set, spans found, chosen (and changed), left as written (can't tell / below the threshold)."""
    import pyarrow.parquet as pq
    t = pq.read_table(work / "choices.parquet", columns=["label_set", "set", "original", "options", "probs",
                                                         "chosen"]).to_pydict()
    out: dict = defaultdict(Counter)
    for ls, name, original, options, probs, chosen in zip(t["label_set"], t["set"], t["original"], t["options"],
                                                          t["probs"], t["chosen"]):
        c = out[f"{name}\t{ls}"]
        c["found"] += 1
        if chosen is not None:
            c["chosen"] += 1
            c["changed"] += C.cased(chosen, original) != original
        else:
            best = max(range(len(probs)), key=lambda k: (probs[k], -k))
            c["left_cant_tell" if options[best] == C.CANT_TELL else "left_below_threshold"] += 1
    res: dict = defaultdict(dict)
    for key, c in out.items():
        name, ls = key.split("\t")
        res[name][ls] = dict(c)
    return res


# ----------------------------------------------------------------------------- audit (architect's sheets)

def _ctx(text: str, s: int, e: int) -> str:
    return (text[max(0, s - CONTEXT):s] + "[[" + text[s:e] + "]]" + text[e:e + CONTEXT]).replace("\t", " ") \
        .replace("\n", " ")


def _candidate_forms() -> dict[str, str]:
    """Every synthetic label form (the source of forms.tsv), written -> conventional."""
    out = {}
    base = C.committed_terms.HERE / "synthetic" / "forms"
    for p in sorted(base.glob("*.tsv")):
        for n, (w, f) in C._lines(p.read_text(encoding="utf-8"), p.name):
            out.setdefault(w.lower(), f)
    return out


def _term_groups() -> list[list[str]]:
    """The committed term lists' spelling-variant groups (terms.variant_key), each in its list's order."""
    import evaluate
    import terms
    from synthetic import layout
    lists = [terms.load(kind) for kind in terms.KINDS]
    syn = []
    for d in layout.DOMAINS:
        syn += terms.read_file(terms.HERE / "synthetic" / f"{d}.txt")
    lists.append(syn)
    groups, seen = [], set()
    for ts in lists:
        m = terms.Matcher(ts, evaluate.normalize)
        for key, phrases in m.groups().items():
            if key in seen:
                continue
            seen.add(key)
            order = {p: k for k, p in enumerate(m.phrases)}
            groups.append([m.phrases[p] for p in sorted(phrases, key=order.get)])
    return groups


_GDROP = None


def _top(samples: list, n: int = FORMS_AUDIT_MATCHES) -> list:
    """The n lowest-ranked samples with distinct contexts (T* and B0 often write the same sentence)."""
    out, seen = [], set()
    for s in sorted(samples):
        if s[-1] not in seen:
            seen.add(s[-1])
            out.append(s)
            if len(out) == n:
                break
    return out


def _audit_shard(shard):
    import re

    import evaluate
    global _GDROP
    run = _W["run"]
    A = _W["audit"]
    if _GDROP is None:
        _GDROP = re.compile(C._LEFT + r"[A-Za-z]+in'" + C._RIGHT)
    t = run.read(shard, LABEL_COLUMNS)
    refs = None
    res = {"forms": defaultdict(Counter), "forms_sample": defaultdict(list), "members": defaultdict(Counter),
           "member_sample": defaultdict(list), "gdrop": defaultdict(Counter), "gdrop_sample": defaultdict(list),
           "ref_counts": Counter(), "ref_labels": 0, "sets": defaultdict(Counter), "set_sample": defaultdict(list)}
    for i, source, rk, texts in kept_texts(t):
        for ls, text in texts.items():
            if not text:
                continue
            for m in A["forms_re"].finditer(text):
                w = m.group(0)
                conv = A["forms"][w.lower()]
                res["forms"][w.lower()][f"{ls} matches"] += 1
                res["forms"][w.lower()][f"{ls} non-canonical"] += w != conv
                res["forms_sample"][w.lower()].append((_rank("f", ls, i, m.start()), ls, _ctx(text, m.start(),
                                                                                              m.end())))
            for m in A["members_re"].finditer(text):
                w = m.group(0)
                res["members"][w.lower()][f"{ls} {w}"] += 1
                res["member_sample"][w.lower()].append((_rank("m", ls, i, m.start()), ls, _ctx(text, m.start(),
                                                                                               m.end())))
            for m in _GDROP.finditer(text):
                w = m.group(0)
                res["gdrop"][w.lower()][f"{ls} {'lower' if w == w.lower() else 'capitalized'}"] += 1
                res["gdrop_sample"][w.lower()].append((_rank("g", ls, i, m.start()), ls, _ctx(text, m.start(),
                                                                                              m.end())))
            for sp in _W["tables"].spans(text):
                res["sets"][sp.set.name][ls] += 1
                res["set_sample"][sp.set.name].append((_rank("s", ls, i, sp.start), ls, _ctx(text, sp.start,
                                                                                             sp.end)))
        if rk != "human":
            continue
        if refs is None:
            refs = run.references(shard[0])
        ref = refs.get(i)
        if not ref:
            continue
        res["ref_labels"] += 1
        for occ in A["ref_matcher"].find(evaluate.normalize(ref).split()):
            res["ref_counts"][" ".join(occ.phrase)] += 1
    for k in ("forms_sample", "member_sample", "gdrop_sample", "set_sample"):
        res[k] = {w: _top(v) for w, v in res[k].items()}
    return res


def audit(run_dir: Path, workers: int = 1, tables_dir: Path | None = None, out: Path | None = None,
          sample_shards: int | None = None) -> dict:
    """The architect's sheets for the tables (cleaning plan "Layer 1": the forms audit, the variant groups and their
    canonical forms by human-reference frequency, the dialect entries and g-dropped words, the layer 2 sets)."""
    import evaluate
    import terms
    tables = C.Tables.load(tables_dir)
    forms = _candidate_forms()
    groups = _term_groups()
    # groups of the tables beyond the term lists (ok / okay, alright / all right; two-member layer 2 sets)
    covered = {m.lower() for g in groups for m in g}
    table_groups: dict[str, set] = {}
    for e in tables.layer1.entries.values():
        if e.table == "variants":
            table_groups.setdefault(e.group, {e.group.lower()}).add(e.key)
    extra = [[c] + sorted(k for k in keys if k != c.lower()) for c, keys in table_groups.items()]
    extra += [list(cs.members) for cs in tables.sets if not cs.option_only and len(cs.members) == 2]
    for g in extra:
        if not covered & {m.lower() for m in g}:
            groups.append(g)
            covered |= {m.lower() for m in g}
    dialect = {k: e for k, e in tables.layer1.entries.items() if e.table == "dialect"}
    member_keys = {m.lower(): False for g in groups for m in g} | {k: e.lower_only for k, e in dialect.items()}
    for cs in tables.sets:
        member_keys.update({m.lower(): False for m in cs.members})
    phrases = {}
    for g in groups:
        for m in g:
            phrases[m] = tuple(evaluate.normalize(m).split())
    for k in dialect:
        phrases[k] = tuple(evaluate.normalize(k).split())
        phrases[dialect[k].output] = tuple(evaluate.normalize(dialect[k].output).split())
    ref_matcher = terms.Matcher(list(phrases), evaluate.normalize)
    A = {"forms": forms, "forms_re": C._compile({w: False for w in forms}),
         "members_re": C._compile(member_keys), "ref_matcher": ref_matcher}
    run = Run(run_dir, sample_shards)
    tot = {"forms": defaultdict(Counter), "forms_sample": defaultdict(list), "members": defaultdict(Counter),
           "member_sample": defaultdict(list), "gdrop": defaultdict(Counter), "gdrop_sample": defaultdict(list),
           "ref_counts": Counter(), "ref_labels": 0, "sets": defaultdict(Counter), "set_sample": defaultdict(list)}
    _W["audit"] = A                                        # inherited by the forked workers
    shards = run.shards
    for n, part in enumerate(_pool_map(_audit_shard, shards, workers, (run_dir, tables_dir))):
        for k in ("forms", "members", "gdrop", "sets"):
            for w, c in part[k].items():
                tot[k][w].update(c)
        for k in ("forms_sample", "member_sample", "gdrop_sample", "set_sample"):
            for w, v in part[k].items():
                tot[k][w] = _top(tot[k][w] + v)
        tot["ref_counts"].update(part["ref_counts"])
        tot["ref_labels"] += part["ref_labels"]
        if (n + 1) % 200 == 0:
            log(f"audit: {n + 1}/{len(shards)} shards")
    out = C.audit_dir() if out is None else Path(out)
    out.mkdir(parents=True, exist_ok=True)
    return _write_audit(out, tables, forms, groups, dialect, phrases, tot, run)


def _write_audit(out: Path, tables, forms, groups, dialect, phrases, tot, run) -> dict:
    kept_forms = {k for k, e in tables.layer1.entries.items() if e.table == "forms"}
    set_members = {m.lower(): cs.name for cs in tables.sets for m in cs.members}
    variant_of = {k: e.group for k, e in tables.layer1.entries.items() if e.table == "variants"}
    summary = {"label_run": run.rel, "tables_sha256": tables.sha256, "human_reference_labels": tot["ref_labels"]}
    with open(out / "forms.txt", "w", encoding="utf-8") as f:
        f.write("# Layer 1 forms audit: every synthetic label form, with up to 30 matches (both label sets, every\n"
                "# kept label of the run). status: in forms.tsv / moved to choices.tsv / left out. Proposal noted.\n")
        rows = []
        for w in sorted(forms, key=lambda w: -sum(v for k, v in tot["forms"][w].items() if "matches" in k)):
            c = tot["forms"][w]
            status = "forms.tsv" if w in kept_forms else (f"choices.tsv ({set_members[w]})" if w in set_members
                                                          else "left out")
            rows.append({"written": w, "conventional": forms[w], "status": status, **dict(c)})
            f.write(f"\n== {w!r} -> {forms[w]!r}  [{status}]  " + ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
                    + "\n")
            for _, ls, ctx in tot["forms_sample"].get(w, []):
                f.write(f"   {ls:3s} {ctx}\n")
        summary["forms"] = rows
    canon = []
    with open(out / "variants.txt", "w", encoding="utf-8") as f:
        f.write("# Variant groups: classification (pure = variants.tsv, contextual = choices.tsv), written counts per\n"
                "# label set, human-reference counts (normalized words, kept ids) and sample contexts.\n")
        for g in groups:
            keys = [m.lower() for m in g]
            if any(k in variant_of for k in keys):
                cls = f"pure (variants.tsv, canonical {variant_of[next(k for k in keys if k in variant_of)]!r})"
            elif any(k in set_members for k in keys):
                cls = f"contextual (choices.tsv: {set_members[next(k for k in keys if k in set_members)]})"
            else:
                cls = "not in the tables"
            refc = {m: tot["ref_counts"].get(" ".join(phrases[m]), 0) for m in g}
            best = max(g, key=lambda m: (refc[m], -g.index(m)))
            canon.append({"group": g, "classification": cls, "reference_counts": refc, "proposed_canonical": best,
                          "written_counts": {m: dict(tot["members"].get(m.lower(), {})) for m in g}})
            f.write(f"\n== {' | '.join(g)}  [{cls}]\n   human references (normalized): " +
                    ", ".join(f"{m!r} {refc[m]}" for m in g) + f"  -> canonical by frequency {best!r}\n")
            for m in g:
                c = tot["members"].get(m.lower(), {})
                f.write(f"   written {m!r}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) + "\n")
                for _, ls, ctx in tot["member_sample"].get(m.lower(), [])[:12]:
                    f.write(f"      {ls:3s} {ctx}\n")
    with open(out / "canonical.tsv", "w", encoding="utf-8") as f:
        f.write("group\tclassification\thuman reference counts (normalized words)\tcanonical by frequency\t"
                "written counts in the labels\n")
        for c in canon:
            f.write("\t".join((" | ".join(c["group"]), c["classification"],
                               ", ".join(f"{m}={n}" for m, n in c["reference_counts"].items()),
                               c["proposed_canonical"],
                               "; ".join(f"{m}: " + ", ".join(f"{k} {v}" for k, v in sorted(w.items()))
                                         for m, w in c["written_counts"].items()))) + "\n")
    summary["variant_groups"] = canon
    drows = []
    with open(out / "dialect.txt", "w", encoding="utf-8") as f:
        f.write("# Dialect entries (dialect.tsv): written counts per label set, human-reference counts of the\n"
                "# dialect form and its standard form (normalized words), sample contexts.\n")
        for k, e in sorted(dialect.items(), key=lambda kv: -sum(tot["members"].get(kv[0], {}).values())):
            c = tot["members"].get(k, {})
            r = (tot["ref_counts"].get(" ".join(phrases[k]), 0), tot["ref_counts"].get(" ".join(phrases[e.output]), 0))
            drows.append({"dialect": k, "standard": e.output, "lower_only": e.lower_only, "written": dict(c),
                          "reference_dialect": r[0], "reference_standard": r[1]})
            f.write(f"\n== {k!r} -> {e.output!r}{'  [lowercase only]' if e.lower_only else ''}  " +
                    ", ".join(f"{a} {v}" for a, v in sorted(c.items())) +
                    f"; human references: dialect {r[0]}, standard {r[1]}\n")
            for _, ls, ctx in tot["member_sample"].get(k, [])[:12]:
                f.write(f"   {ls:3s} {ctx}\n")
    summary["dialect"] = drows
    grows = []
    with open(out / "gdropped.tsv", "w", encoding="utf-8") as f:
        f.write("word\tallowlisted (dialect.tsv)\tT* lower\tT* capitalized\tB0 lower\tB0 capitalized\tcontexts\n")
        for w, c in sorted(tot["gdrop"].items(), key=lambda kv: -sum(kv[1].values())):
            allow = w in dialect
            grows.append({"word": w, "allowlisted": allow, **dict(c)})
            ctxs = " || ".join(f"{ls}: {ctx}" for _, ls, ctx in tot["gdrop_sample"].get(w, [])[:5])
            f.write("\t".join((w, "yes" if allow else "no", *(str(c.get(k, 0)) for k in (
                "T* lower", "T* capitalized", "B0 lower", "B0 capitalized")), ctxs)) + "\n")
    summary["gdropped"] = grows
    with open(out / "choices.txt", "w", encoding="utf-8") as f:
        f.write("# Layer 2 sets (choices.tsv): spans per label set (option-only members start no span) and samples.\n")
        for cs in tables.sets:
            c = tot["sets"].get(cs.name, {})
            f.write(f"\n== {cs.name}  option-only {sorted(cs.option_only)}  " +
                    ", ".join(f"{k} {v}" for k, v in sorted(c.items())) + "\n")
            for _, ls, ctx in tot["set_sample"].get(cs.name, [])[:15]:
                f.write(f"   {ls:3s} {ctx}\n")
    summary["sets"] = {cs.name: dict(tot["sets"].get(cs.name, {})) for cs in tables.sets}
    write_json(out / "audit.json", summary)
    return summary


# ----------------------------------------------------------------------------- engines (GPU only)

def _letter_ids(tok, n: int) -> list[int]:
    ids = []
    for letter in C.LETTERS[:n]:
        enc = tok.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise RuntimeError(f"option letter {letter!r} is not one token: {enc}")
        ids.append(enc[0])
    return ids


def _chat(tok, user: str) -> str:
    return tok.apply_chat_template([{"role": "system", "content": C.SYSTEM}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _softmax(xs: list[float]) -> list[float]:
    import math
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


class TransformersScorer:
    """syn-textgen under transformers (bf16): one forward pass per batch, float32 logits at the last prompt
    position, softmax over the option letters."""

    def __init__(self, device: str = "cuda", declared_vram_gb: float | None = None, batch: int = 32):
        self.device, self.declared_vram_gb, self.batch = device, declared_vram_gb, batch
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        from synthetic import layout, textgen
        place = textgen.placement(self.device)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from labeling.engines import vram_fraction
        entry = layout.source(layout.TEXTGEN)
        self.hf_id, self.revision = entry["hf_id"], entry["revision"]
        if self.device == "cuda":
            frac = vram_fraction(self.declared_vram_gb)
            if frac is not None:
                torch.cuda.set_per_process_memory_fraction(frac, 0)
        self.tok = AutoTokenizer.from_pretrained(self.hf_id, revision=self.revision, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_id, revision=self.revision, dtype=torch.bfloat16,
                                                          **place)
        self.model.eval()
        self.letters = _letter_ids(self.tok, len(C.LETTERS))

    def info(self) -> dict:
        from synthetic import layout
        import transformers
        return {"alias": layout.TEXTGEN, "revision": getattr(self, "revision", None), "engine": "transformers",
                "dtype": "bfloat16", "batch": self.batch, "transformers": transformers.__version__}

    def peak(self) -> dict:
        import torch
        return {"gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                "gpu_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30}

    def score(self, users: list[str], n_options: list[int]) -> list[list[float]]:
        import torch
        order = sorted(range(len(users)), key=lambda k: len(users[k]))
        out: list = [None] * len(users)
        for s in range(0, len(order), self.batch):
            idx = order[s:s + self.batch]
            enc = self.tok([_chat(self.tok, users[k]) for k in idx], return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                logits = self.model(**enc).logits[:, -1, :].float()
            for row, k in zip(logits, idx):
                out[k] = _softmax([float(row[t]) for t in self.letters[:n_options[k]]])
        return out


class VllmScorer:
    """syn-textgen under vLLM (run with labeling/vllm_env.sh python), batch-invariant: one generated position,
    restricted to the option letters (allowed_token_ids), whose processed log-probabilities are read
    (logprobs_mode "processed_logprobs", so every allowed letter is returned) and renormalized."""

    def __init__(self, gpu_memory_utilization: float = 0.85, max_model_len: int = 4096, seed: int = 0):
        self.gpu_memory_utilization, self.max_model_len, self.seed = gpu_memory_utilization, max_model_len, seed
        self.llm = None

    def load(self) -> None:
        if self.llm is not None:
            return
        import importlib
        from labeling.engines import apply_vllm_mode, check_vllm_mode
        apply_vllm_mode(True)
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        check_vllm_mode(importlib.import_module("vllm.envs"), True)
        from synthetic import layout
        entry = layout.source(layout.TEXTGEN)
        self.hf_id, self.revision = entry["hf_id"], entry["revision"]
        path = snapshot_download(self.hf_id, revision=self.revision, local_files_only=True)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.letters = _letter_ids(self.tok, len(C.LETTERS))
        self.llm = LLM(model=path, tokenizer=path, dtype="bfloat16", seed=self.seed,
                       gpu_memory_utilization=self.gpu_memory_utilization, max_model_len=self.max_model_len,
                       logprobs_mode="processed_logprobs")
        self.SamplingParams = SamplingParams

    def info(self) -> dict:
        from synthetic import layout
        import vllm
        return {"alias": layout.TEXTGEN, "revision": getattr(self, "revision", None), "engine": "vllm",
                "vllm": vllm.__version__, "batch_invariant": True, "logprobs_mode": "processed_logprobs"}

    def peak(self) -> dict:
        return {"gpu_memory_utilization": self.gpu_memory_utilization}

    def score(self, users: list[str], n_options: list[int]) -> list[list[float]]:
        import math
        params = [self.SamplingParams(temperature=0.0, max_tokens=1, logprobs=n,
                                      allowed_token_ids=self.letters[:n]) for n in n_options]
        outs = self.llm.generate([_chat(self.tok, u) for u in users], params, use_tqdm=False)
        res = []
        for o, n in zip(outs, n_options):
            lp = o.outputs[0].logprobs[0]
            missing = [t for t in self.letters[:n] if t not in lp]
            if missing:
                raise RuntimeError(f"vLLM returned no log-probability for option tokens {missing}")
            res.append(_softmax([lp[t].logprob if math.isfinite(lp[t].logprob) else -1e30
                                 for t in self.letters[:n]]))
        return res


ENGINES = {"vllm": VllmScorer, "transformers": TransformersScorer}


# ----------------------------------------------------------------------------- CLI

def _run_dir(name: str) -> Path:
    from training import labels as tlabels
    return tlabels.production_run_dir(name)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="training.cleaning_run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("spans", "choose", "assemble", "check", "audit"):
        s = sub.add_parser(name)
        s.add_argument("--labels-run", required=True, help="the C2 run name (labels/<T*>/<run>)")
        if name in ("spans", "check", "audit"):
            s.add_argument("--workers", type=int, default=8)
            s.add_argument("--sample-shards", type=int, metavar="N",
                           help="N evenly spaced shards only (a measurement; needs --out, never the artifact)")
            s.add_argument("--out", type=Path, help="output directory (default: the run's cleaning directory)")
        if name == "check":
            s.add_argument("--layer1-only", action="store_true", help="no choices artifact (layer 1 only)")
        if name == "choose":
            s.add_argument("--engine", choices=sorted(ENGINES), default="vllm")
            s.add_argument("--probe", type=int, metavar="N", help="score N prompts and report; write no decisions")
            s.add_argument("--determinism", type=float, metavar="F", help="re-score a fraction F (Check 2)")
            s.add_argument("--budget-minutes", type=float)
            s.add_argument("--declared-vram-gb", type=float, help="transformers: the job's VRAM lease")
            s.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vllm")
    a = p.parse_args(argv)
    import relabel_paths
    relabel_paths.require_mount()
    run_dir = _run_dir(a.labels_run)
    work = C.production_dir(a.labels_run)
    sample = getattr(a, "sample_shards", None)
    out = getattr(a, "out", None)
    if sample and out is None:
        p.error("--sample-shards needs --out (a sampled run is a measurement, never the run's artifact)")
    if a.cmd == "spans":
        r = spans(run_dir, out or work, a.workers, sample_shards=sample)
        print(json.dumps({k: r[k] for k in ("kept", "labels_with_span", "spans", "spans_per_label_set",
                                            "spans_per_set", "unique_prompts", "seconds")}, indent=1))
        return 0
    if a.cmd == "assemble":
        print(json.dumps(assemble(work, run_dir), indent=1))
        return 0
    if a.cmd == "check":
        r = check(run_dir, work, out_dir=out, layer1_only=a.layer1_only, workers=a.workers, sample_shards=sample)
        print(json.dumps({"disabled": r["check3"]["disabled"], "wer": r["corpus_wer_human_reference_labels"]},
                         indent=1))
        return 0
    if a.cmd == "audit":
        r = audit(run_dir, a.workers, out=out, sample_shards=sample)
        print(json.dumps({"human_reference_labels": r["human_reference_labels"], "out": str(out or C.audit_dir())}))
        return 0
    scorer = VllmScorer(a.gpu_memory_utilization) if a.engine == "vllm" else \
        TransformersScorer(declared_vram_gb=a.declared_vram_gb)
    if a.probe:
        print(json.dumps(probe(work, scorer, a.probe), indent=1))
        return 0
    if a.determinism:
        print(json.dumps(determinism(work, scorer, a.determinism), indent=1))
        return 0
    print(json.dumps(choose(work, scorer, a.budget_minutes, run_dir=run_dir), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
