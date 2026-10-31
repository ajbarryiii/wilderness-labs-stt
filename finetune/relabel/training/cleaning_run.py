"""Label cleaning runner (cleaning plan "Layer 2", "Checks"): spans, choose, assemble, check, audit.

    ./python -m training.cleaning_run spans --labels-run RUN [--workers N]          CPU
    ./python -m training.cleaning_run choose --labels-run RUN [--engine vllm|transformers] [--probe N]
                                     [--budget-minutes M] [--declared-vram-gb G] [--determinism F]
                                     [--gpu-memory-utilization U] [--lock-wait S]                   GPU
    ./python -m training.cleaning_run assemble --labels-run RUN                     CPU

The GPU commands (choose, --probe, --determinism; either engine) hold the parakeet GPU lock (score_teacher.gpu_lock)
while the model is loaded and run, and respect the job's VRAM lease (AGENT_JOBS_VRAM_FRACTION): vLLM's
gpu_memory_utilization defaults to the lease and a larger request is refused; transformers caps torch's per-process
memory fraction at the declared VRAM (refused above the lease) or else at the lease. Both are checked before any
model is loaded.
    ./python -m training.cleaning_run check --labels-run RUN [--layer1-only] [--workers N] [--audit-dir D]  CPU
    ./python -m training.cleaning_run audit --labels-run RUN [--workers N]          CPU (architect's sheets)
    ./python -m training.cleaning_run lexicon --labels-run RUN --version 3 [--workers N]          CPU
    ./python -m training.cleaning_run census --labels-run RUN --version 3 [--exclude TSV] [--out D]  CPU (Check 6)

Every command takes --version V (a cleaning version, cleaning.VERSIONS; default cleaning.VERSION). Everything is
written under $RELABEL_ROOT/cleaning/<RUN>/v<V>/ (training/cleaning.production_dir), the architect's sheets under
cleaning.audit_dir(V) ($RELABEL_ROOT/cleaning/audit/ for version 1, .../audit/v<V>/ after). Layer 2 is the same in
every version, so a version's decision blocks may be copied into another version's decisions/ before assemble (a
decision is keyed by its prompt):

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
                        labels changed per source and label set); from version 2 also the spelled-out acronym class
                        ("acronyms": changed spans and labels, shapes, residual-risk classes, compact forms)
    check-layer1.json   check --layer1-only (no choices artifact)

    audit_dir(V)/check-sample[-layer1].tsv   Check 4: AUDIT_SAMPLE changed spans to read (from version 2,
                        AUDIT_ACRONYMS of them spelled-out acronyms, and AUDIT_PUNCTUATION layer 0 edits added:
                        cleaning plan v6, 250 in all); audit_dir(V)/acronyms[-layer1].txt the acronym class's sheet
                        (version 2)

Version 2 (cleaning plan v6): T*'s labels are cleaned with the record's b0_raw as layer 0's donor; Check 3 never
screens layer 0 edits (punctuation and case change no normalized word) and reports the spelled-out acronym class
without judging it (NOT_JUDGED: AJ's decision), so it is never disabled.

Version 3 (cleaning plan v7) reads the run's acronym lexicon, built first by `lexicon` (lexicon.parquet and
lexicon.json in cleaning/<RUN>/v3/). Its layer 2 adds the OpEx set; the decisions of version 2 may be copied into
v3/decisions/ before `choose`, which then scores only the new prompts. `check` also reports, not judges, the
spelling-versus-acronym class (both groups) and the dose-unit class (NOT_JUDGED, NOT_JUDGED_TABLES), adds
AUDIT_SPELLING spelling outputs and AUDIT_UNITS unit edits to the audit sample (350 in all), and runs the size check
("version3": labels changed against version 2 per class, against EXPECTED_V3; the comparison reads version 2's
artifact of the run unless --layer1-only). `census` writes Check 6's blind sheet and its answer key.

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
import re
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
AUDIT_ACRONYMS = 50              # version 2: of the AUDIT_SAMPLE spans, drawn from the spelled-out acronyms (plan v5)
AUDIT_PUNCTUATION = 50           # version 2: layer 0 edits added to the AUDIT_SAMPLE spans (plan v6: 250 in all)
AUDIT_SPELLING = 50              # version 3: spelling outputs of the spelling-versus-acronym class added (plan v7)
AUDIT_UNITS = 50                 # version 3: dose-unit edits added (plan v7: 350 in all)
RISK_SAMPLES = 15                # examples per residual-risk class on the acronym sheet
COMPACT_LISTED = 150             # the most frequent compact forms on the acronym sheet and in the report
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


def _init(run_dir, tables_dir, with_spec: bool = False, choices_dir=None, version: int = C.VERSION,
          lexicon_dir=None, compare: tuple | None = None, exclude=None) -> None:
    """Per worker: the run, the tables of `version` and (with_spec) the cleaning spec of `version` with the choices
    artifact, if any, and (version 3) the acronym lexicon; compare = (tables_dir, choices_dir): version 2's spec, the
    size check's reference; exclude: the census's excluded (id, letters)."""
    _W["run"] = Run(run_dir)
    _W["tables"] = C.Tables.load(tables_dir, version)
    _W["spec"] = None if not with_spec else C.Cleaning(
        _W["tables"], None if choices_dir is None else C.Choices.load(choices_dir), version,
        None if lexicon_dir is None else C.Lexicon.load(lexicon_dir))
    _W["spec_v2"] = None if compare is None else C.Cleaning(
        C.Tables.load(compare[0], 2), None if compare[1] is None else C.Choices.load(compare[1]), 2)
    _W["exclude"] = frozenset(exclude or ())


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
          sample_shards: int | None = None, version: int = C.VERSION) -> dict:
    """Every layer 2 span of the run's kept labels, both label sets, and the unique prompts (CPU). Layer 2 is the same
    in every version; the version is recorded so that the artifact assembled from them is of that version."""
    t0 = time.time()
    run, tables = Run(run_dir, sample_shards), C.Tables.load(tables_dir, version)
    rows, prompts = [], {}
    kept, with_span = Counter(), Counter()
    for n, (r, p, k, w) in enumerate(_pool_map(_spans_shard, run.shards, workers,
                                               (run_dir, tables_dir, False, None, version))):
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
    report = {"version": version, "label_run": run.rel, "contract_sha256": run.contract_sha256,
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

def _read_prompts(work: Path, tables: C.Tables, version: int = C.VERSION) -> list[dict]:
    import pyarrow.parquet as pq
    meta = json.loads((work / "spans.json").read_text())
    for key, want in (("tables_sha256", tables.sha256), ("template_sha256", C.TEMPLATE_SHA256),
                      ("version", version)):
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


def probe(work: Path, scorer, n: int = PROBE_DEFAULT, tables_dir: Path | None = None,
          version: int = C.VERSION) -> dict:
    """Score n prompts spread over the list (no decisions written): throughput, memory, extrapolated time."""
    prompts = _read_prompts(work, C.Tables.load(tables_dir, version), version)
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
           tables_dir: Path | None = None, assemble_when_done: bool = True, run_dir: Path | None = None,
           version: int = C.VERSION) -> dict:
    """Score every unique prompt not yet decided, block by block (resumable); assemble once all are decided."""
    tables = C.Tables.load(tables_dir, version)
    prompts = _read_prompts(work, tables, version)
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
        out["manifest"] = assemble(work, run_dir, tables_dir, version)
    return out


def determinism(work: Path, scorer, fraction: float = 0.01, tables_dir: Path | None = None,
                version: int = C.VERSION) -> dict:
    """Check 2: re-score a fixed fraction of the decided prompts and compare the decisions."""
    tables = C.Tables.load(tables_dir, version)
    prompts = _read_prompts(work, tables, version)
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

def assemble(work: Path, run_dir: Path | None, tables_dir: Path | None = None, version: int = C.VERSION) -> dict:
    """choices.parquet + manifest.json from spans.parquet and the decision blocks (decisions of prompts no longer
    in the spans, e.g. of a disabled set, are ignored). A decision is keyed by its prompt (template, label text, span,
    options), not by the cleaning version, so another version's decision blocks may be copied into decisions/."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    tables = C.Tables.load(tables_dir, version)
    meta = json.loads((work / "spans.json").read_text())
    _read_prompts(work, tables, version)                          # same tables, template and version as the spans
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
    manifest = {"version": version, "label_run": meta["label_run"], "contract_sha256": meta["contract_sha256"],
                "tables_sha256": tables.sha256, "template_sha256": C.TEMPLATE_SHA256, "threshold": C.THRESHOLD,
                "model": {"alias": engine.get("alias"), "revision": engine.get("revision")}, "engine": engine,
                "rows": len(sp["id"]), "file_sha256": sha, "unique_prompts": len(set(sp["prompt_key"])),
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


def acronym_shape(run: str) -> tuple[int, str, bool, bool]:
    """(letters, separators, final period, plural) of a spelled-out acronym run as written (an acronym Edit's `run`):
    separators "dotted" ("A.P.I."), "dotted+spaced" ("A. P. I."), "spaced" ("A P I") or "mixed"."""
    caps = [k for k, c in enumerate(run) if "A" <= c <= "Z"]
    seps = {run[a + 1:b] for a, b in zip(caps, caps[1:])}
    kind = {frozenset({"."}): "dotted", frozenset({". "}): "dotted+spaced",
            frozenset({" "}): "spaced"}.get(frozenset(seps), "mixed")
    tail = run[caps[-1] + 1:] if caps else ""
    return len(caps), kind, tail.startswith("."), "s" in tail


def acronym_compact(run: str) -> str:
    """The compact form of a run as the acronym rule writes it, before any table (no final period)."""
    n, kind, final, plural = acronym_shape(run)
    return "".join(c for c in run if "A" <= c <= "Z") + ("s" if plural else "")


def acronym_risks(text: str, e: C.Edit) -> list[str]:
    """The residual-risk classes (cleaning plan v5, review p4) a changed acronym span falls in; heuristics that only
    count the class's changed spans for the audit, never a rule of the cleaning. Read from the edit's run as written
    (an edit may also hold table words merged into it)."""
    n, kind, final, plural = acronym_shape(e.run)
    letters = "".join(c for c in e.run if "A" <= c <= "Z")
    s = e.start + e.original.find(e.run)
    end = s + len(e.run)
    rest = text[:s] + " " + text[end:]
    out = []
    if n >= 3 and re.search(rf"(?<![^\W_]){letters}(?![^\W_])", rest, re.IGNORECASE):
        out.append("spelled word: the letters are a word of the label")
    if n >= 6 and kind in ("dotted", "dotted+spaced"):
        out.append("dotted, 6+ letters (letter-by-letter spelling)")
    if n >= 6 and kind == "spaced":
        out.append("spaced, 6+ letters (letter-by-letter spelling)")
    if n >= 3 and kind == "spaced" and _MATH.search(rest):
        out.append("variables: spaced, 3+ letters, mathematical words in the label")
    if n >= 3 and set(letters) <= set("ABCDEFG"):
        out.append("notes: 3+ letters of A-G" + (", music words in the label" if _MUSIC.search(rest) else ""))
    if n == 2 and kind == "dotted+spaced" and not final:
        out.append("two letters '. ' without a final period ('Plan A. B is', 'I. A lot')")
    if kind == "mixed":
        out.append("mixed separators ('I. A P I')")
    if text.strip(" .,!?") == e.run.strip(" ."):
        out.append("the whole label is the run (a spelling: 'M. I. N. E.')")
    if re.match(r" ?[a-z](?![^\W_])", text[end:]):
        out.append("a single lowercase letter follows (a partial run: 'L. C. d.')")
    return out


_MATH = re.compile(r"(?i)(?<![^\W_])(?:equals?|plus|minus|times|squared|cubed|sub|prime|matrix|matrices|vectors?|"
                   r"variables?|derivative|integral|sum|transpose|indices|index|modulo|mod)(?![^\W_])|[=+^]")
_MUSIC = re.compile(r"(?i)(?<![^\W_])(?:notes?|chords?|scales?|keys?|melody|sharp|flat|major|minor|guitar|piano|"
                    r"tune|songs?|sing|octave|riff|bass|play(?:s|ed|ing)?)(?![^\W_])")


def _acronym_totals() -> dict:
    return {"spans": Counter(), "labels": Counter(), "shape": Counter(), "letters": Counter(), "risks": Counter(),
            "compact": Counter(), "risk_sample": defaultdict(list)}


def _check_shard(shard):
    import evaluate
    run, spec, spec2 = _W["run"], _W["spec"], _W["spec_v2"]
    t = run.read(shard, LABEL_COLUMNS)
    refs = None
    out = {"changed": Counter(), "labels": Counter(), "edits": Counter(), "screen": defaultdict(Counter),
           "wer": Counter(), "sample": [], "acr_sample": [], "l0_sample": [], "spell_sample": [], "unit_sample": [],
           "acr": _acronym_totals(), "v3": Counter()}
    acr = out["acr"]
    added = set(spec.tables.added) if spec.version >= 3 else set()
    for i, source, rk, texts in kept_texts(t):
        for ls, text in texts.items():
            out["labels"][ls] += 1
            donor = texts["B0"] if spec.needs_donor(ls) else None
            trace = {} if spec2 is not None else None
            after, edits = spec.edits(text, i, ls, donor, trace=trace)
            if after != text:
                out["changed"][f"{ls}\t{source}"] += 1
            for n, e in enumerate(edits):
                out["edits"][f"{ls}\t{_group_key(e)}"] += 1
                acronym = e.table == C.ACRONYM_TABLE and e.group == C.ACRONYM_GROUP
                key = "acr_sample" if acronym else "l0_sample" if e.layer == 0 else \
                    "spell_sample" if e.table == C.ACRONYM_TABLE else "unit_sample" if e.table == C.UNITS_TABLE \
                    else "sample"
                out[key].append((_rank(ls, i, n), ls, i, e.table, e.group, e.original, e.replacement,
                                 text[max(0, e.start - CONTEXT):e.end + CONTEXT]))
                if acronym:
                    letters, kind, final, plural = acronym_shape(e.run)
                    acr["spans"][ls] += 1
                    acr["shape"][f"{ls}\t{kind}{', final period' if final else ''}{', plural' if plural else ''}"] += 1
                    acr["letters"][f"{ls}\t{letters if letters < 6 else '6+'}"] += 1
                    acr["compact"][f"{ls}\t{acronym_compact(e.run)}"] += 1
                    for risk in acronym_risks(text, e):
                        acr["risks"][f"{ls}\t{risk}"] += 1
                        acr["risk_sample"][risk].append((_rank("r", ls, i, n), ls, i, e.original, e.replacement,
                                                         _ctx(text, e.start, e.end)))
            if any(e.table == C.ACRONYM_TABLE and e.group == C.ACRONYM_GROUP for e in edits):
                acr["labels"][ls] += 1
            if spec2 is not None:
                _size_check(out["v3"], ls, text, after, edits, spec2.clean(text, i, ls, donor), trace, added,
                            spec.spans(text))
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
                if e.table == "forms" or e.layer == 0:      # casing and layer 0 change no normalized word
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
    for k, n in (("sample", AUDIT_SAMPLE), ("acr_sample", AUDIT_ACRONYMS), ("l0_sample", AUDIT_PUNCTUATION),
                 ("spell_sample", AUDIT_SPELLING), ("unit_sample", AUDIT_UNITS)):
        out[k] = sorted(out[k])[:n]
    acr["risk_sample"] = {k: sorted(v)[:RISK_SAMPLES] for k, v in acr["risk_sample"].items()}
    return out


SPELLING_CLASS, UNITS_CLASS = "spelling versus acronym", "dose units after a digit"


def _change_kind(d: C.Decision) -> str | None:
    """How the spelling rule's decision on a run differs from version 2 (None: it does not): a spelling where
    version 2 compacted ("undone", the calls that change normalized words), a spelling version 2 left as written
    ("rewritten"), a trim, a compaction of another run undone ("other undone"), or anything else ("other: <class>")."""
    if d.v3 == d.v2:
        return None
    if d.cls == "trim":
        return "trim"
    compacted = d.v2 != d.run
    if d.cls == "spelling":
        return "undone" if compacted else "rewritten"
    if d.cls == "other" and compacted:
        return "other undone"
    return f"other: {d.cls}"


def _size_check(c: Counter, ls: str, text: str, after: str, edits: list, after_v2: str, trace: dict, added: set,
                spans: list) -> None:
    """Version 3's size check (cleaning plan v7, Check 6 "Size check"): one label's changes against version 2, per
    class, and the spelling rule's classes (counts into c, keyed "<label set>\t<what>")."""
    labelled = set()
    if after != after_v2:
        c[f"{ls}\tlabels changed vs version 2"] += 1
    kinds = set()
    for d in trace.get("runs", []):
        if len(d.letters) >= 3 and d.cls not in ("pair", "i-repeat", "initials"):
            c[f"{ls}\truns\t{d.cls}"] += 1
        kind = _change_kind(d)
        if kind is not None:
            kinds.add(kind)
            c[f"{ls}\truns changed vs version 2\t{kind}"] += 1
    for kind in kinds:
        c[f"{ls}\tlabels changed vs version 2\t{SPELLING_CLASS}\t{kind}"] += 1
    if kinds:
        labelled.add(SPELLING_CLASS)
    for e in edits:
        if e.table == C.UNITS_TABLE:
            c[f"{ls}\toccurrences\t{UNITS_CLASS}"] += 1
            labelled.add(UNITS_CLASS)
        elif _group_key(e) in added:
            c[f"{ls}\toccurrences\t{_group_key(e)}"] += 1
            labelled.add(_group_key(e))
    for sp in spans:
        if f"choices:{sp.set.name}" in added:
            c[f"{ls}\tspans\tchoices:{sp.set.name}"] += 1
    for name in labelled:
        c[f"{ls}\tlabels\t{name}"] += 1
    if after != after_v2 and not labelled:
        c[f"{ls}\tlabels changed vs version 2\tunattributed"] += 1


# groups Check 3 reports without judging (cleaning plan v6, AJ 2026-10-29: the compact acronym is a deliberate style;
# cleaning plan v7, AJ 2026-10-29 and 2026-10-30: the spelling-versus-acronym class and the dose-unit class)
NOT_JUDGED = {f"{C.ACRONYM_TABLE}:{C.ACRONYM_GROUP}": "AJ's decision (cleaning plan v6): a deliberate style choice, "
                                                      "reported, not judged",
              f"{C.ACRONYM_TABLE}:{C.SPELLING_GROUP}": "AJ's decision (cleaning plan v7): the spelling-versus-acronym "
                                                       "class, reported, not judged"}
NOT_JUDGED_TABLES = {C.UNITS_TABLE: "AJ's decision (cleaning plan v7): the dose-unit class, reported, not judged"}
# the plan's expected changes against version 2 (cleaning plan v7 "Expected changes"; occurrences unless "labels";
# "at most" bounds are reported against the bound)
EXPECTED_V3 = {SPELLING_CLASS: {"what": "labels", "T*": 714, "B0": 1200, "B0 bound": "at most about"},
               UNITS_CLASS: {"what": "occurrences", "T*": 2500, "B0": 2200},
               "variants:Schrödinger": {"what": "occurrences (with the possessive group)", "T*": 52, "B0": 17},
               "choices:OpEx|OPX": {"what": "spans", "T*": 21, "B0": 6, "bound": "at most"}}
SIZE_TOLERANCE = 0.25


def judge(screen: dict[str, dict[str, Counter]]) -> dict:
    """Check 3's decision per group: judged on every cell (each label set, and both pooled) with at least
    MIN_JUDGED aligned changed spans; disabled when a judged cell's original matches more often than its cleaned
    output. The groups of NOT_JUDGED are reported (their cells and where the original matched more often) but never
    judged or disabled."""
    out = {}
    for group, cells in sorted(screen.items()):
        pooled = Counter()
        for c in cells.values():
            pooled.update(c)
        allc = dict(cells) | {"both": pooled}
        judged = {k: c for k, c in allc.items() if c["aligned"] >= MIN_JUDGED}
        harm = sorted(k for k, c in judged.items() if c["original_matches"] > c["cleaned_matches"])
        why = NOT_JUDGED.get(group) or NOT_JUDGED_TABLES.get(group.split(":", 1)[0])
        if why is not None:
            out[group] = {"cells": {k: dict(c) for k, c in allc.items()}, "judged": [], "harm_in": [],
                          "original_matches_more_in": harm, "disabled": False, "reported_not_judged": True,
                          "why": why}
            continue
        out[group] = {"cells": {k: dict(c) for k, c in allc.items()}, "judged": sorted(judged),
                      "harm_in": harm, "disabled": bool(harm), "reported_not_judged": not judged}
    return out


def check(run_dir: Path, work: Path, out_dir: Path | None = None, layer1_only: bool = False, workers: int = 1,
          tables_dir: Path | None = None, sample_shards: int | None = None, version: int = C.VERSION,
          lexicon_dir: Path | None = None, v2_choices_dir: Path | None = None, audit: Path | None = None) -> dict:
    """Checks 3 and 5 over every kept label of the run under cleaning `version` (CPU), and the Check 4 sample: the
    AUDIT_SAMPLE lowest-ranked changed spans, of which AUDIT_ACRONYMS (or all, if fewer) are spelled-out acronyms
    (version 2; plan v5), with the acronym class's sheet (shapes, residual-risk classes, compact forms). Version 3
    (plan v7) reads the acronym lexicon of lexicon_dir (default: work), adds AUDIT_SPELLING spelling outputs and
    AUDIT_UNITS dose-unit edits to the sample, and runs the size check against version 2 (with v2_choices_dir's
    artifact, or layer 2 left as written when None). audit: the sheets' directory (default cleaning.audit_dir)."""
    t0 = time.time()
    run = Run(run_dir, sample_shards)
    choices_dir = None if layer1_only else work
    lexicon_dir = (work if lexicon_dir is None else lexicon_dir) if version >= 3 else None
    compare = (tables_dir, v2_choices_dir) if version >= 3 else None
    spec = C.Cleaning(C.Tables.load(tables_dir, version), None if choices_dir is None else C.Choices.load(choices_dir),
                      version, None if lexicon_dir is None else C.Lexicon.load(lexicon_dir))
    spec.bind_run(run.contract_sha256)
    agg = {"changed": Counter(), "labels": Counter(), "edits": Counter(), "screen": defaultdict(Counter),
           "wer": Counter(), "v3": Counter()}
    acr = _acronym_totals()
    sample, acr_sample, l0_sample, spell_sample, unit_sample = [], [], [], [], []
    for n, part in enumerate(_pool_map(_check_shard, run.shards, workers,
                                       (run_dir, tables_dir, True, choices_dir, version, lexicon_dir, compare))):
        for k in ("changed", "labels", "edits", "wer", "v3"):
            agg[k].update(part[k])
        for g, c in part["screen"].items():
            agg["screen"][g].update(c)
        sample = sorted(sample + part["sample"])[:AUDIT_SAMPLE]
        acr_sample = sorted(acr_sample + part["acr_sample"])[:AUDIT_ACRONYMS]
        l0_sample = sorted(l0_sample + part["l0_sample"])[:AUDIT_PUNCTUATION]
        spell_sample = sorted(spell_sample + part["spell_sample"])[:AUDIT_SPELLING]
        unit_sample = sorted(unit_sample + part["unit_sample"])[:AUDIT_UNITS]
        for k in ("spans", "labels", "shape", "letters", "risks", "compact"):
            acr[k].update(part["acr"][k])
        for k, v in part["acr"]["risk_sample"].items():
            acr["risk_sample"][k] = sorted(acr["risk_sample"][k] + v)[:RISK_SAMPLES]
        if (n + 1) % 200 == 0:
            log(f"check: {n + 1}/{len(run.shards)} shards")
    sample = sorted(acr_sample + sample[:AUDIT_SAMPLE - len(acr_sample)]) + l0_sample   # v6: + layer 0's 50
    sample += spell_sample + unit_sample                                                 # v7: + 50 + 50
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
    report = {"version": version, "label_run": run.rel, "tables_sha256": spec.tables.sha256,
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
    if version >= 2:
        report["acronyms"] = _acronym_report(acr)
    if version >= 3:
        report["version3"] = size_report(agg["v3"], spec.tables.added)
    out_dir = work if out_dir is None else Path(out_dir)
    write_json(out_dir / ("check-layer1.json" if layer1_only else "check.json"), report)
    audit = C.audit_dir(version) if audit is None else Path(audit)
    audit.mkdir(parents=True, exist_ok=True)
    suffix = "-layer1" if layer1_only else ""
    with open(audit / f"check-sample{suffix}.tsv", "w", encoding="utf-8") as f:
        f.write("label_set\tid\ttable\tgroup\toriginal\treplacement\twrong (architect)\tcontext\n")
        for _, ls, i, table, group, o, r, ctx in sample:
            f.write("\t".join((ls, i, table, group, o, r, "", ctx.replace("\t", " ").replace("\n", " "))) + "\n")
    if version >= 2:
        _write_acronym_sheet(audit / f"acronyms{suffix}.txt", report, acr)
    return report


def size_report(c: Counter, added) -> dict:
    """Version 3's size check (cleaning plan v7): labels changed against version 2 per class and label set, the
    spelling rule's classes and changed runs, the added groups' occurrences, and each class against the plan's
    expected changes (EXPECTED_V3; flagged when the difference exceeds SIZE_TOLERANCE)."""
    per = _per_label_set(c)
    labels = {k.split("\t", 1)[1]: v for k, v in per.items() if k.startswith("labels\t")}
    occurrences = {k.split("\t", 1)[1]: v for k, v in per.items() if k.startswith("occurrences\t")}
    spans_found = {k.split("\t", 1)[1]: v for k, v in per.items() if k.startswith("spans\t")}
    got = {SPELLING_CLASS: labels.get(SPELLING_CLASS, {}), UNITS_CLASS: occurrences.get(UNITS_CLASS, {}),
           "variants:Schrödinger": {ls: sum(v.get(ls, 0) for g, v in occurrences.items()
                                            if g.startswith("variants:Schrödinger")) for ls in LABEL_SETS},
           "choices:OpEx|OPX": spans_found.get("choices:OpEx|OPX", {})}
    against = {}
    for name, want in EXPECTED_V3.items():
        row = {"what": want["what"]}
        for ls in LABEL_SETS:
            n, e = got.get(name, {}).get(ls, 0), want[ls]
            bound = want.get("bound") or (want.get("B0 bound") if ls == "B0" else None)
            diff = (n - e) / e if e else None
            flag = diff is not None and (n > e * (1 + SIZE_TOLERANCE) if bound else abs(diff) > SIZE_TOLERANCE)
            row[ls] = {"measured": n, "expected": e, "bound": bound, "difference": diff, "over_25_percent": flag}
        against[name] = row
    return {"labels_changed_vs_version_2": per.get("labels changed vs version 2", {}),
            "labels_changed_vs_version_2_per_kind": {k.split("\t", 2)[2]: v for k, v in per.items()
                                                     if k.startswith(f"labels changed vs version 2\t{SPELLING_CLASS}")},
            "unattributed_labels": per.get("labels changed vs version 2\tunattributed", {}),
            "labels_per_class": labels, "occurrences": occurrences, "layer2_spans_of_added_sets": spans_found,
            "runs_per_class": {k.split("\t", 1)[1]: v for k, v in per.items() if k.startswith("runs\t")},
            "runs_changed_vs_version_2": {k.split("\t", 1)[1]: v for k, v in per.items()
                                          if k.startswith("runs changed vs version 2\t")},
            "added_groups": list(added), "expected": against,
            "note": "runs: 3+ letters outside the pre-classes (pairs, I-repeats, initials), as the probe counted; "
                    "a run's class and its change against version 2 are the first round's, on the layer 1 input"}


def _per_label_set(c: Counter) -> dict:
    """{"<label set>\\t<key>": n} -> {key: {label set: n}}."""
    out: dict = defaultdict(dict)
    for key, v in c.items():
        ls, k = key.split("\t", 1)
        out[k][ls] = v
    return dict(out)


def _acronym_report(acr: dict) -> dict:
    compact = _per_label_set(acr["compact"])
    top = sorted(compact, key=lambda k: (-sum(compact[k].values()), k))[:COMPACT_LISTED]
    return {"changed_spans": dict(acr["spans"]), "labels_changed": dict(acr["labels"]),
            "shapes": _per_label_set(acr["shape"]), "letters": _per_label_set(acr["letters"]),
            "residual_risks": _per_label_set(acr["risks"]), "distinct_compact_forms": len(compact),
            "compact": {k: compact[k] for k in top},
            "note": "residual-risk classes are heuristics that count the class's changed spans for the audit"}


def _write_acronym_sheet(path: Path, report: dict, acr: dict) -> None:
    a = report["acronyms"]
    g = report["check3"]["groups"].get(f"{C.ACRONYM_TABLE}:{C.ACRONYM_GROUP}")

    def counts(d: dict) -> str:
        return ", ".join(f"{ls} {d.get(ls, 0)}" for ls in LABEL_SETS)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Spelled-out acronyms (cleaning v{report['version']}, plan v5): every changed span of the run's "
                f"kept labels.\n# label run {report['label_run']}, tables {report['tables_sha256']}, layer 1 only: "
                f"{report['layer1_only']}\n\n")
        f.write(f"changed spans: {counts(a['changed_spans'])}\nlabels changed: {counts(a['labels_changed'])}\n")
        if g is not None:
            f.write("Check 3 (aligned changed spans; original / cleaned matches): " + "; ".join(
                f"{k} {c.get('aligned', 0)} aligned, {c.get('original_matches', 0)} / {c.get('cleaned_matches', 0)}"
                for k, c in g["cells"].items()) + f"; disabled: {g['disabled']}\n")
        f.write("\n== shapes\n")
        for k, v in sorted(a["shapes"].items(), key=lambda kv: -sum(kv[1].values())):
            f.write(f"   {k}: {counts(v)}\n")
        f.write("\n== letters\n")
        for k, v in sorted(a["letters"].items()):
            f.write(f"   {k}: {counts(v)}\n")
        f.write("\n== residual-risk classes (heuristics; a span may be in several)\n")
        for k, v in sorted(a["residual_risks"].items(), key=lambda kv: -sum(kv[1].values())):
            f.write(f"\n-- {k}: {counts(v)}\n")
            for _, ls, i, o, r, ctx in acr["risk_sample"].get(k, []):
                f.write(f"   {ls:3s} {o!r} -> {r!r}   {ctx}\n")
        f.write(f"\n== compact forms ({a['distinct_compact_forms']} distinct; the {COMPACT_LISTED} most frequent)\n")
        for k, v in a["compact"].items():
            f.write(f"   {k}: {counts(v)}\n")


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


# ----------------------------------------------------------------------------- version 3: lexicon and census

def _lexicon_shard(shard):
    t = _W["run"].read(shard, LABEL_COLUMNS)
    acc, kept = {}, 0
    for i, _source, _rk, texts in kept_texts(t):
        C.count_lexicon(i, texts.values(), acc)
        kept += 1
    return acc, kept


def lexicon(run_dir: Path, out: Path, workers: int = 1, sample_shards: int | None = None) -> dict:
    """The acronym lexicon of the run (cleaning version 3; training/cleaning.py "The acronym lexicon"), from the raw
    labels of both label sets of every kept id, written to out (CPU)."""
    t0 = time.time()
    run = Run(run_dir, sample_shards)
    acc, kept = {}, 0
    for n, (part, k) in enumerate(_pool_map(_lexicon_shard, run.shards, workers, (run_dir, None))):
        C.merge_lexicon(acc, part)
        kept += k
        if (n + 1) % 200 == 0:
            log(f"lexicon: {n + 1}/{len(run.shards)} shards, {len(acc)} keys")
    lex = C.Lexicon.write(out, acc, run.contract_sha256, run.rel,
                          extra={"keys_counted": len(acc), "kept_ids": kept, "shards": len(run.shards),
                                 "sampled_shards": run.sampled, "seconds": round(time.time() - t0, 1)})
    return lex.manifest


CENSUS_SEED = "cleaning-v3-census"
CENSUS_COMPACTED = 100           # Check 6 stratum 2: runs the rule compacts (acronym or ambiguous)
CENSUS_TRIMS = 50                # Check 6 stratum 3: trims
CENSUS_BEFORE, CENSUS_AFTER = 160, 90
CENSUS_STRATA = ("changes words", "compacted", "trim")


def census_strata(d: C.Decision) -> list[str]:
    """The Check 6 strata of a T* run's decision (cleaning plan v7 Check 6): a spelling call where version 2 compacted
    (directly or as a trim's rest; the calls that change normalized words), a compaction by the rule (acronym or
    ambiguous), a trim."""
    out = []
    compacted_v2 = d.v2 != d.run
    if compacted_v2 and (d.cls == "spelling" or d.cls == "trim" and d.rest.cls == "spelling"):
        out.append("changes words")
    if d.cls in ("acronym", "ambiguous") and d.v3 != d.run:
        out.append("compacted")
    if d.cls == "trim":
        out.append("trim")
    return out


def _census_item(i: str, text: str, d: C.Decision, strata: list[str]) -> tuple:
    ctx = (text[max(0, d.start - CENSUS_BEFORE):d.start] + "[[" + d.run + "]]" + text[d.end:d.end + CENSUS_AFTER])
    rest = d.rest
    return (_rank(CENSUS_SEED, i, d.start), i, d.start, d.end, d.run, d.letters, d.cls, d.evidence,
            "" if rest is None else rest.cls, "" if rest is None else rest.evidence, d.v2, d.v3,
            ",".join(strata), ctx.replace("\t", " ").replace("\n", " "))


def _census_shard(shard):
    run, spec, exclude = _W["run"], _W["spec"], _W["exclude"]
    t = run.read(shard, LABEL_COLUMNS)
    out = {"items": {k: [] for k in CENSUS_STRATA}, "population": Counter(), "excluded": Counter()}
    for i, _source, _rk, texts in kept_texts(t):
        text, trace = texts["T*"], {}
        spec.edits(text, i, "T*", texts["B0"], trace=trace)
        for d in trace.get("runs", []):
            strata = census_strata(d)
            if not strata:
                continue
            item = _census_item(i, trace["text"], d, strata)
            for k in strata:
                out["population"][k] += 1
                if (i, d.letters) in exclude:
                    out["excluded"][k] += 1
                    continue
                out["items"][k].append(item)
    out["items"]["compacted"] = sorted(out["items"]["compacted"])[:CENSUS_COMPACTED]
    out["items"]["trim"] = sorted(out["items"]["trim"])[:CENSUS_TRIMS]
    return out


def census(run_dir: Path, out: Path, workers: int = 1, lexicon_dir: Path | None = None, tables_dir: Path | None = None,
           exclude: set | None = None, sample_shards: int | None = None, version: int = 3) -> dict:
    """Check 6's sheet (cleaning plan v7, before any training run reads version 3; CPU): over T*'s kept labels
    (layer 1 input after layer 0; layer 2 spans left as written), every run of the first stratum and, drawn with a
    fixed seed, CENSUS_COMPACTED compacted runs and CENSUS_TRIMS trims, without the tuning runs in `exclude` ((id,
    letters)). Writes census-sheet.tsv (blind: the run in its context only, shuffled), census-key.tsv (the class, the
    strata, version 2's and version 3's outputs) and census.json (populations, sample sizes) to out."""
    if version < 3:
        raise ValueError("the census is of cleaning version 3")
    t0 = time.time()
    run = Run(run_dir, sample_shards)
    spec = C.Cleaning(C.Tables.load(tables_dir, version), None, version, C.Lexicon.load(lexicon_dir))
    spec.bind_run(run.contract_sha256)
    items = {k: [] for k in CENSUS_STRATA}
    population, excluded = Counter(), Counter()
    for n, part in enumerate(_pool_map(_census_shard, run.shards, workers,
                                       (run_dir, tables_dir, True, None, version, lexicon_dir, None,
                                        sorted(exclude or ())))):
        population.update(part["population"])
        excluded.update(part["excluded"])
        items["changes words"] += part["items"]["changes words"]
        items["compacted"] = sorted(items["compacted"] + part["items"]["compacted"])[:CENSUS_COMPACTED]
        items["trim"] = sorted(items["trim"] + part["items"]["trim"])[:CENSUS_TRIMS]
        if (n + 1) % 200 == 0:
            log(f"census: {n + 1}/{len(run.shards)} shards")
    rows = {}
    for k in CENSUS_STRATA:
        for it in items[k]:
            rows.setdefault((it[1], it[2]), it)
    order = sorted(rows.values(), key=lambda it: _rank(CENSUS_SEED, "order", it[1], it[2]))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "census-sheet.tsv", "w", encoding="utf-8") as f:
        f.write("n\tcontext (the run in [[ ]])\tlabel (spelling / acronym / other / unsure; for a run ending in a word "
                "I or A, the letters before it)\tends in the word I or A? (y / n; blank if its last letter is not I or "
                "A)\tnote\n")
        for n, it in enumerate(order, 1):
            f.write(f"{n}\t{it[13]}\t\t\t\n")
    with open(out / "census-key.tsv", "w", encoding="utf-8") as f:
        f.write("n\tstrata\tid\tstart\tend\trun\tletters\tclass\tevidence\trest class\trest evidence\t"
                "version 2 output\tversion 3 output\n")
        for n, it in enumerate(order, 1):
            f.write("\t".join([str(n), it[12], it[1], str(it[2]), str(it[3])] + [x.replace("\t", " ") for x in it[4:12]])
                    + "\n")
    report = {"version": version, "label_run": run.rel, "label_set": "T*", "seed": CENSUS_SEED,
              "tables_sha256": spec.tables.sha256, "lexicon_sha256": spec.lexicon.file_sha256,
              "sampled_shards": run.sampled, "population": dict(population), "excluded_tuning_runs": dict(excluded),
              "sample": {k: len(items[k]) for k in CENSUS_STRATA}, "rows": len(order),
              "floor": "first stratum and trims: the lower bound of a 90% Wilson interval >= 0.5; unsure counts "
                       "against the rule", "seconds": round(time.time() - t0, 1)}
    write_json(out / "census.json", report)
    return report


def read_exclude(path: Path) -> set:
    """(id, letters) of the tuning runs to leave out of the census: a TSV with columns id and letters."""
    out = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            i, letters = line.split("\t")[:2]
            out.add((i, letters))
    return out


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


EXTRA_GROUPS = (("ok", "okay"), ("alright", "all right"))   # spelling variants with no fixed standard (plan v3)


def canonical_form(group: list[str], written: dict[str, dict[str, int]]) -> tuple[str, dict[str, int]]:
    """The canonical spelling of a pure variant group (cleaning plan v4): the member T* writes most often across all
    kept labels (case-insensitive count per member), ties to the group's first (term list) form; written in T*'s most
    frequent casing of it, where a form written in lowercase or with only its first letter capitalized counts as
    lowercase (the capitalization rule restores a sentence-initial capital). `written`: {member: {"T* <as written>":
    count, "B0 ...": count}}. Returns (canonical, {member: T* count})."""
    counts, casings = {}, {}
    for m in group:
        cas = Counter()
        for k, n in written.get(m, {}).items():
            if k.startswith("T* "):
                w = k[3:]
                cas[w.lower() if C.cased(w.lower(), w) == w else w] += n
        counts[m], casings[m] = sum(cas.values()), cas
    best = max(group, key=lambda m: (counts[m], -group.index(m)))
    if not casings[best]:
        return best, counts
    top = max(casings[best].values())
    tied = sorted(w for w, n in casings[best].items() if n == top)
    return (best if best in tied else tied[0]), counts


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
    import evaluate
    run = _W["run"]
    A = _W["audit"]
    t = run.read(shard, LABEL_COLUMNS)
    refs = None
    res = _audit_totals()
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
    for k in _SAMPLES:
        res[k] = {w: _top(v) for w, v in res[k].items()}
    return res


_SAMPLES = ("forms_sample", "member_sample", "set_sample")


def _audit_totals() -> dict:
    return {"forms": defaultdict(Counter), "forms_sample": defaultdict(list), "members": defaultdict(Counter),
            "member_sample": defaultdict(list), "ref_counts": Counter(), "ref_labels": 0,
            "sets": defaultdict(Counter), "set_sample": defaultdict(list)}


def audit(run_dir: Path, workers: int = 1, tables_dir: Path | None = None, out: Path | None = None,
          sample_shards: int | None = None, version: int = C.VERSION) -> dict:
    """The architect's sheets for the tables (cleaning plan "Layer 1": the forms audit, the variant groups and their
    canonical forms (canonical_form: T*'s majority form; human-reference counts shown for information), the dialect
    entries, the layer 2 sets)."""
    import evaluate
    import terms
    tables = C.Tables.load(tables_dir, version)
    forms = _candidate_forms()
    groups = _term_groups()
    # groups of the tables beyond the term lists (ok / okay, alright / all right; two-member layer 2 sets)
    covered = {m.lower() for g in groups for m in g}
    table_groups: dict[str, set] = {}
    for e in tables.layer1.entries.values():
        if e.table == "variants":
            table_groups.setdefault(e.group, {e.group.lower()}).add(e.key)
    extra = [[c] + sorted(k for k in keys if k != c.lower()) for c, keys in table_groups.items()]
    extra += [list(g) for g in EXTRA_GROUPS]
    extra += [list(cs.members) for cs in tables.sets if not cs.option_only and len(cs.members) == 2]
    for g in extra:
        keys = {m.lower() for m in g}
        if not covered & keys:
            groups.append(g)
            covered |= keys
        else:                       # a table group with more listed forms than its term list ("pre tax")
            for grp in groups:
                if {m.lower() for m in grp} & keys:
                    grp += [m for m in g if m.lower() not in {x.lower() for x in grp}]
                    covered |= keys
                    break
    dialect = {k: e for k, e in tables.layer1.entries.items() if e.table == "dialect"}
    member_keys = {m.lower() for g in groups for m in g} | set(dialect)
    for cs in tables.sets:
        member_keys |= {m.lower() for m in cs.members}
    phrases = {}
    for g in groups:
        for m in g:
            phrases[m] = tuple(evaluate.normalize(m).split())
    for k in dialect:
        phrases[k] = tuple(evaluate.normalize(k).split())
        phrases[dialect[k].output] = tuple(evaluate.normalize(dialect[k].output).split())
    ref_matcher = terms.Matcher(list(phrases), evaluate.normalize)
    A = {"forms": forms, "forms_re": C._compile(forms), "members_re": C._compile(member_keys),
         "ref_matcher": ref_matcher}
    run = Run(run_dir, sample_shards)
    tot = _audit_totals()
    _W["audit"] = A                                        # inherited by the forked workers
    shards = run.shards
    for n, part in enumerate(_pool_map(_audit_shard, shards, workers, (run_dir, tables_dir, False, None, version))):
        for k in ("forms", "members", "sets"):
            for w, c in part[k].items():
                tot[k][w].update(c)
        for k in _SAMPLES:
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
        f.write("# Variant groups: classification (pure = variants.tsv, contextual = choices.tsv), the canonical form by\n"
                "# T*'s count over all kept labels (cleaning plan v4), written counts per label set, human-reference\n"
                "# counts (normalized words, kept ids; information only) and sample contexts.\n")
        for g in groups:
            keys = [m.lower() for m in g]
            if any(k in variant_of for k in keys):
                cls = f"pure (variants.tsv, canonical {variant_of[next(k for k in keys if k in variant_of)]!r})"
            elif any(k in set_members for k in keys):
                cls = f"contextual (choices.tsv: {set_members[next(k for k in keys if k in set_members)]})"
            else:
                cls = "not in the tables"
            refc = {m: tot["ref_counts"].get(" ".join(phrases[m]), 0) for m in g}
            written = {m: dict(tot["members"].get(m.lower(), {})) for m in g}
            best, tcount = canonical_form(g, written)
            canon.append({"group": g, "classification": cls, "t_star_counts": tcount, "proposed_canonical": best,
                          "reference_counts": refc, "written_counts": written})
            f.write(f"\n== {' | '.join(g)}  [{cls}]\n   T* (all kept labels): " +
                    ", ".join(f"{m!r} {tcount[m]}" for m in g) + f"  -> canonical {best!r}; human references "
                    "(normalized): " + ", ".join(f"{m!r} {refc[m]}" for m in g) + "\n")
            for m in g:
                c = tot["members"].get(m.lower(), {})
                f.write(f"   written {m!r}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) + "\n")
                for _, ls, ctx in tot["member_sample"].get(m.lower(), [])[:12]:
                    f.write(f"      {ls:3s} {ctx}\n")
    with open(out / "canonical.tsv", "w", encoding="utf-8") as f:
        f.write("group\tclassification\tT* counts (all kept labels, case-insensitive)\tcanonical (T* majority, T* "
                "casing)\thuman reference counts (information only)\twritten counts in the labels\n")
        for c in canon:
            f.write("\t".join((" | ".join(c["group"]), c["classification"],
                               ", ".join(f"{m}={n}" for m, n in c["t_star_counts"].items()),
                               c["proposed_canonical"],
                               ", ".join(f"{m}={n}" for m, n in c["reference_counts"].items()),
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
            drows.append({"dialect": k, "standard": e.output, "written": dict(c),
                          "reference_dialect": r[0], "reference_standard": r[1]})
            f.write(f"\n== {k!r} -> {e.output!r}  " +
                    ", ".join(f"{a} {v}" for a, v in sorted(c.items())) +
                    f"; human references: dialect {r[0]}, standard {r[1]}\n")
            for _, ls, ctx in tot["member_sample"].get(k, [])[:12]:
                f.write(f"   {ls:3s} {ctx}\n")
    summary["dialect"] = drows
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


LEASE_ENV = "AGENT_JOBS_VRAM_FRACTION"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
LOCK_WAIT = 4 * 3600.0


def lease_fraction() -> float | None:
    """The job's VRAM lease as a fraction of the GPU (agent-jobs sets AGENT_JOBS_VRAM_FRACTION), or None."""
    v = os.environ.get(LEASE_ENV)
    return float(v) if v else None


def vllm_utilization(requested: float | None) -> float:
    """vLLM's gpu_memory_utilization: the requested fraction (refused above the lease), else the lease, else
    DEFAULT_GPU_MEMORY_UTILIZATION."""
    cap = lease_fraction()
    if requested is None:
        return DEFAULT_GPU_MEMORY_UTILIZATION if cap is None else min(DEFAULT_GPU_MEMORY_UTILIZATION, cap)
    if cap is not None and requested > cap + 1e-6:
        raise RuntimeError(f"gpu_memory_utilization {requested} is above the job's VRAM lease {cap} ({LEASE_ENV})")
    return requested


def transformers_fraction(declared_gb: float | None, total_gb: float) -> float | None:
    """torch's per-process memory fraction: the declared VRAM over the GPU's (refused above the lease), else the
    lease, else None (no cap)."""
    cap = lease_fraction()
    if declared_gb is None:
        return cap
    frac = min(1.0, declared_gb / total_gb)
    if cap is not None and frac > cap + 1e-6:
        raise RuntimeError(f"declared VRAM {declared_gb} GB is {frac:.3f} of the GPU, above the job's VRAM lease "
                           f"{cap} ({LEASE_ENV})")
    return frac


def make_scorer(engine: str, gpu_memory_utilization: float | None = None, declared_vram_gb: float | None = None):
    """The real scorer of `engine`, its memory cap checked against the lease before anything is loaded."""
    if engine == "vllm":
        return VllmScorer(vllm_utilization(gpu_memory_utilization))
    if engine == "transformers":
        if declared_vram_gb is not None and lease_fraction() is not None:
            transformers_fraction(declared_vram_gb, _gpu_total_gb())   # refuse now, not after a model load
        return TransformersScorer(declared_vram_gb=declared_vram_gb)
    raise ValueError(f"unknown engine {engine!r}")


def _gpu_total_gb() -> float:
    import torch
    return torch.cuda.get_device_properties(0).total_memory / 2 ** 30


class TransformersScorer:
    """syn-textgen under transformers (bf16): one forward pass per batch, float32 logits at the last prompt
    position, softmax over the option letters. Memory: torch's per-process fraction (transformers_fraction)."""

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
        entry = layout.source(layout.TEXTGEN)
        self.hf_id, self.revision = entry["hf_id"], entry["revision"]
        if self.device == "cuda":
            self.memory_fraction = transformers_fraction(self.declared_vram_gb, _gpu_total_gb())
            if self.memory_fraction is not None:
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction, 0)
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
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()   # not batch×length×vocab
            for row, k in zip(logits, idx):
                out[k] = _softmax([float(row[t]) for t in self.letters[:n_options[k]]])
        return out


class VllmScorer:
    """syn-textgen under vLLM (run with labeling/vllm_env.sh python), batch-invariant: one generated position,
    restricted to the option letters (allowed_token_ids), whose processed log-probabilities are read
    (logprobs_mode "processed_logprobs", so every allowed letter is returned) and renormalized."""

    def __init__(self, gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION, max_model_len: int = 4096,
                 seed: int = 0):
        self.gpu_memory_utilization, self.max_model_len, self.seed = gpu_memory_utilization, max_model_len, seed
        self.llm = None

    def load(self) -> None:
        if self.llm is not None:
            return
        vllm_utilization(self.gpu_memory_utilization)          # refuse above the lease before loading
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
    for name in ("spans", "choose", "assemble", "check", "audit", "lexicon", "census"):
        s = sub.add_parser(name)
        s.add_argument("--labels-run", required=True, help="the C2 run name (labels/<T*>/<run>)")
        s.add_argument("--version", type=int, choices=C.VERSIONS, default=C.VERSION,
                       help=f"cleaning version (default {C.VERSION}; stages 2b and 2c read version 1): the run's "
                            "cleaning/<run>/v<version>/ and the sheets in cleaning.audit_dir(version)")
        if name in ("spans", "check", "audit", "lexicon", "census"):
            s.add_argument("--workers", type=int, default=8)
            s.add_argument("--sample-shards", type=int, metavar="N",
                           help="N evenly spaced shards only (a measurement; needs --out, never the artifact)")
            s.add_argument("--out", type=Path, help="output directory (default: the run's cleaning directory)")
        if name == "check":
            s.add_argument("--layer1-only", action="store_true", help="no choices artifact (layer 1 only)")
            s.add_argument("--audit-dir", type=Path, help="the sheets' directory (default: cleaning.audit_dir)")
        if name == "census":
            s.add_argument("--exclude", type=Path, help="a TSV (id, letters) of tuning runs to leave out")
        if name == "choose":
            s.add_argument("--engine", choices=sorted(ENGINES), default="vllm")
            s.add_argument("--probe", type=int, metavar="N", help="score N prompts and report; write no decisions")
            s.add_argument("--determinism", type=float, metavar="F", help="re-score a fraction F (Check 2)")
            s.add_argument("--budget-minutes", type=float)
            s.add_argument("--declared-vram-gb", type=float, help="transformers: the job's VRAM (capped by the lease)")
            s.add_argument("--gpu-memory-utilization", type=float,
                           help=f"vllm (default: the job's VRAM lease, at most {DEFAULT_GPU_MEMORY_UTILIZATION})")
            s.add_argument("--lock-wait", type=float, default=LOCK_WAIT, help="seconds to wait for the GPU lock")
    a = p.parse_args(argv)
    import relabel_paths
    relabel_paths.require_mount()
    run_dir = _run_dir(a.labels_run)
    work = C.production_dir(a.labels_run, a.version)
    sample = getattr(a, "sample_shards", None)
    out = getattr(a, "out", None)
    if sample and out is None:
        p.error("--sample-shards needs --out (a sampled run is a measurement, never the run's artifact)")
    if a.cmd == "spans":
        r = spans(run_dir, out or work, a.workers, sample_shards=sample, version=a.version)
        print(json.dumps({k: r[k] for k in ("kept", "labels_with_span", "spans", "spans_per_label_set",
                                            "spans_per_set", "unique_prompts", "seconds")}, indent=1))
        return 0
    if a.cmd == "assemble":
        print(json.dumps(assemble(work, run_dir, version=a.version), indent=1))
        return 0
    if a.cmd == "check":
        v2_choices = None if a.layer1_only or a.version < 3 else C.production_dir(a.labels_run, 2)
        r = check(run_dir, work, out_dir=out, layer1_only=a.layer1_only, workers=a.workers, sample_shards=sample,
                  version=a.version, v2_choices_dir=v2_choices, audit=a.audit_dir)
        print(json.dumps({"disabled": r["check3"]["disabled"], "wer": r["corpus_wer_human_reference_labels"]}
                         | ({"expected": r["version3"]["expected"]} if a.version >= 3 else {}), indent=1))
        return 0
    if a.cmd == "lexicon":
        if a.version < 3:
            p.error("the acronym lexicon is of cleaning version 3 (--version 3)")
        print(json.dumps(lexicon(run_dir, out or work, a.workers, sample_shards=sample), indent=1))
        return 0
    if a.cmd == "census":
        if a.version < 3:
            p.error("the census is of cleaning version 3 (--version 3)")
        r = census(run_dir, out or work, a.workers, lexicon_dir=work,
                   exclude=None if a.exclude is None else read_exclude(a.exclude), sample_shards=sample,
                   version=a.version)
        print(json.dumps(r, indent=1))
        return 0
    if a.cmd == "audit":
        out = out or C.audit_dir(a.version)
        r = audit(run_dir, a.workers, out=out, sample_shards=sample, version=a.version)
        print(json.dumps({"human_reference_labels": r["human_reference_labels"], "out": str(out)}))
        return 0
    return gpu_command(a, work, run_dir)


def gpu_command(a, work: Path, run_dir: Path) -> int:
    """choose / --probe / --determinism: the memory cap is checked against the job's lease first, then the model
    is loaded and run only while holding the parakeet GPU lock (score_teacher.gpu_lock, as every relabel GPU entry
    point does); a lock not free within --lock-wait exits score_teacher.LOCK_BUSY_EXIT."""
    import score_teacher
    scorer = make_scorer(a.engine, a.gpu_memory_utilization, a.declared_vram_gb)
    mode = "probe" if a.probe else "determinism" if a.determinism else "choose"
    version = getattr(a, "version", C.VERSION)
    try:
        with score_teacher.gpu_lock(f"relabel cleaning {mode} {a.labels_run}", a.lock_wait):
            if a.probe:
                out = probe(work, scorer, a.probe, version=version)
            elif a.determinism:
                out = determinism(work, scorer, a.determinism, version=version)
            else:
                out = choose(work, scorer, a.budget_minutes, run_dir=run_dir, version=version)
    except score_teacher.LockBusy as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return score_teacher.LOCK_BUSY_EXIT
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
