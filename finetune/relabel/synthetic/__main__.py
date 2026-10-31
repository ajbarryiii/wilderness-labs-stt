"""CLI of stage 2c (synthetic plan). Jobs: synthetic/launch.sh. Pipeline order:

    fetch ALIAS...            pinned, verified downloads (network)
    termbuild [--fetch]       write terms/synthetic/<domain>.txt (once, by hand; then review and commit)
    split                     freeze the targeted / held-out split ($RELABEL_ROOT/synthetic/terms/split.json)
    coverage                  reference occurrences in eval-stem and dev-technical (underpowered domains)
    voicecheck                the voice-provenance check of both TTS engines
    noise-pool                the TRAIN noise / RIR pool and its disjointness check (CPU job)
    prompts  --run R --n N    the seeded prompts of a run
    textgen  --run R          generator chunk job (GPU, parakeet GPU lock held for the job)
    hygiene  --run R          sentence hygiene (CPU), label forms included (synthetic/forms.py)
    forms-dry-run --run R     label forms over a finished run's sentences: counts and checks only, writes nothing
    reuse-text --from R0 --run R   copy run R0's prompts, generator outputs and hygiene into R (no generation)
    tts-plan --run R          seeded, balanced engine / voice assignment, and each clip's tts_text (pronunciation)
    tts      --run R --engine A [--device cpu | --device cuda --declared-vram-gb GB]
                              rendering chunk job of one engine (RELABEL_DEPS=A)
    roundtrip --run R         T* transcription chunk job (GPU, lock held)
    finalize --run R          clips.jsonl, report.json, listening sample (CPU)

Confusable pairs (synthetic plan v5; synthetic/confusable.py, synthetic/pairs.py, synthetic/pair_metrics.py):
    pairs-words               word counts over main-f3's T* training labels (CPU job)
    pairs-detect              the detector: $RELABEL_ROOT/synthetic/v5/candidates.json for the blind curator (CPU)
    pairs-split               split v3 from the curated terms/synthetic/pairs.tsv (and split.v2.json, once)
    pairs-check-full          full's clips under v3's held-out guard (CPU)
    pairs-coverage            coverage under v3 (synthetic/terms/coverage.v3.json; reads references)
    pairs-prompts --round K   round K's prompts from the need left by rounds 1..K-1 (CPU)
    textgen, tts-plan, tts, roundtrip, finalize --run pairs-rK   as for any run
    pairs-hygiene --round K   round K's hygiene under v3 and its selection (CPU; instead of hygiene)
    pairs-finalize            run pairs: the rounds' clips joined, report.json with the shortfalls (CPU)
    (pairs-prompts, pairs-hygiene, pairs-finalize --run pairs-c: the contrast top-up, plan v5.2; rounds pairs-c-rK)
    pairs-metrics --model NAME TECH_DIR STEM_DIR ... [--contrast A B] --out NAME   metrics from saved hypotheses (CPU)

--run is "pilot", "pilot2" (the pilot's sentences re-rendered with the pronunciation layer), "full" or a round of run
pairs or pairs-c, "pairs-r1".."pairs-r3", "pairs-c-r1".."pairs-c-r3" (production, $RELABEL_ROOT/synthetic/runs/<run>/,
a round in runs/<pairs run>/r<K>/);
--smoke NAME instead writes everything to $RELABEL_ROOT/smoke/synthetic/<NAME>/ (never a production path).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from synthetic import layout

LOCK_BUSY_EXIT = 3


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg, flush=True)


def run_dir(args) -> Path:
    if getattr(args, "smoke", None):
        d = layout.smoke_dir(args.smoke)
        assert not layout.is_production(d)
        return d
    if args.run in layout.PAIRS_RUNS:
        raise SystemExit(f"run {args.run} is generated in rounds: --run {args.run}-r1 .. {args.run}-r3; "
                         f"pairs-finalize --run {args.run} joins them")
    return layout.run_dir(args.run)


def refuse_pairs(args, instead: str) -> None:
    if getattr(args, "run", None) in layout.ROUND_RUNS:
        raise SystemExit(f"{args.run}: run pairs uses {instead}")


def split_of(args) -> Path:
    """The split file a run's hygiene runs under (layout.run_split_path; split.json for smoke runs)."""
    return layout.run_split_path(args.run) if getattr(args, "run", None) else layout.split_path()


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def gpu_lock(device: str, label: str, wait: float):
    if device != "cuda":
        return _NoLock()
    import score_teacher
    return score_teacher.gpu_lock(label, wait)


def write_json(path: Path, obj) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def cmd_prompts(args) -> int:
    from synthetic import textgen, vocab
    refuse_pairs(args, "pairs-prompts --round K")
    d = run_dir(args)
    p = d / "prompts.jsonl"
    split = vocab.Split.load()
    prompts = textgen.make_prompts(split, args.n, layout.SEED, d.name)
    if p.exists():
        if textgen.read_jsonl(p) != prompts:
            raise SystemExit(f"{p} exists and differs (prompts are written once per run)")
        return 0
    textgen.write_jsonl(p, prompts)
    write_json(d / "run.json", {"run": d.name, "prompts": args.n, "seed": layout.SEED,
                                "split_sha256": _sha(layout.split_path()), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    log(f"{len(prompts)} prompts -> {p}")
    return 0


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_textgen(args) -> int:
    from synthetic import pairs, textgen
    textgen.placement(args.device)    # accelerate on sys.path before score_teacher imports transformers
    import score_teacher
    d = run_dir(args)
    declared, batch = args.declared_vram_gb, args.batch
    probes = load_probes() if declared == "auto" else None
    try:
        with gpu_lock(args.device, f"synthetic textgen {d.name}", args.lock_wait):
            if declared == "auto":   # a probe's measured peak (reserved) + 2 GB at the largest batch that fits
                batch, declared = textgen.declaration(probes, args.batch, fits_lease)
                log(f"batch {batch}, declared VRAM {declared} GB (probe peak + {textgen.DECLARE_MARGIN_GB} GB)")
            declared = None if declared is None else float(declared)
            gen = textgen.Generator(args.device, declared, batch)
            res = textgen.generate_chunk(d, gen, layout.SEED, args.budget_minutes, log,
                                         stream=pairs.text_stream(args.run) if args.run else None)
    except score_teacher.LockBusy as busy:
        print(f"refused: {busy}", file=sys.stderr)
        return LOCK_BUSY_EXIT
    log(json.dumps(res))
    return 0


def load_probes() -> dict:
    """The textgen probe's results by batch size (smoke/synthetic/textgen-probe/batch<N>.json)."""
    out = {}
    for p in sorted(layout.smoke_dir("textgen-probe").glob("batch*.json")):
        r = json.loads(p.read_text())
        out[int(r["batch"])] = r
    return out


def fits_lease(declared_gb: float) -> bool:
    """Whether a declaration fits the job's lease (labeling.engines.vram_fraction's check)."""
    from labeling.engines import vram_fraction
    try:
        vram_fraction(declared_gb)
    except RuntimeError:
        return False
    return True


def cmd_textgen_probe(args) -> int:
    from synthetic import textgen, vocab
    textgen.placement("cuda")         # accelerate on sys.path before score_teacher imports transformers
    import score_teacher
    out = layout.smoke_dir("textgen-probe")
    sizes = textgen.probe_sizes(args.batch, args.min_batch)
    prompts = textgen.make_prompts(vocab.Split.load(), 4 * args.batch * args.batches, layout.SEED, "probe")
    try:
        with gpu_lock("cuda", "synthetic textgen probe", args.lock_wait):
            res = textgen.probe(textgen.Generator("cuda", args.declared_vram_gb, args.batch), prompts, args.batches,
                                sizes=sizes)
    except score_teacher.LockBusy as busy:
        print(f"refused: {busy}", file=sys.stderr)
        return LOCK_BUSY_EXIT
    for size, r in res.items():
        write_json(out / f"batch{size}.json", r)
        log(json.dumps(r))
    return 0 if any(not r.get("oom") for r in res.values()) else 1


def cmd_hygiene(args) -> int:
    from labeling.filters import Coverage, b0_tokenizer_file
    from synthetic import hygiene, textgen, vocab
    refuse_pairs(args, "pairs-hygiene --round K")
    d = run_dir(args)
    prompts = textgen.read_jsonl(d / "prompts.jsonl")
    missing = [s for s, _ in textgen.blocks(len(prompts)) if not (d / "text" / f"{s:07d}.jsonl").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} text blocks missing (finish textgen first)")
    records = [r for s, _ in textgen.blocks(len(prompts)) for r in textgen.read_jsonl(d / "text" / f"{s:07d}.jsonl")]
    split = vocab.Split.load()
    normalize = vocab.default_normalize()
    coverage = Coverage(b0_tokenizer_file(layout.relabel_root() / "caches" / "b0-tokenizer"))
    sets, record = hygiene.heldout_sets()
    kept, report, rows = hygiene.run(records, split, vocab.medical_matcher(normalize), normalize, coverage, sets, record)
    textgen.write_jsonl(d / "hygiene_rows.jsonl", rows)
    textgen.write_jsonl(d / "sentences.jsonl", kept)
    write_json(d / "hygiene.json", report | {"split_sha256": _sha(layout.split_path())})
    log(f"hygiene: kept {report['kept']} of {report['generated']}; reasons {report['reasons']}; "
        f"provisional={record['provisional']}")
    return 0


def cmd_forms_dry_run(args) -> int:
    """Label forms over a finished run's hygiene-kept sentences, read-only: counts, and the checks that the scorer's
    words, the held-out and medical matches and the TTS input are unchanged. Writes nothing."""
    from collections import Counter
    from synthetic import forms, pronounce, textgen, vocab
    d = run_dir(args)
    f, lex, split = forms.Forms.load(), pronounce.Lexicon.load(), vocab.Split.load(split_of(args))
    n = vocab.default_normalize()
    med = vocab.medical_matcher(n)
    sents = textgen.read_jsonl(d / "sentences.jsonl")
    plan = {r["id"]: r for r in textgen.read_jsonl(d / "tts_plan.jsonl")} if (d / "tts_plan.jsonl").exists() else {}
    kept_ids = {c["id"] for c in textgen.read_jsonl(d / "clips.jsonl")} if (d / "clips.jsonl").exists() else set()
    subs, by_domain, problems = Counter(), Counter(), Counter()
    changed = changed_kept = tts_recorded = 0
    examples = []
    for s in sents:
        spoken = s.get("spoken", s["label"])
        label, sub = f.apply(spoken)
        if label != s["label"]:
            changed += 1
            changed_kept += s["id"] in kept_ids
            by_domain[s["domain"]] += 1
            if len(examples) < 8:
                examples.append({"id": s["id"], "before": s["label"], "after": label})
        subs.update(f"{x['written']} -> {x['form']}" for x in sub)
        w0, w1 = n(s["label"]).split(), n(label).split()
        problems["normalized_words_changed"] += w0 != w1
        problems["not_idempotent"] += f.apply(label)[0] != label
        problems["heldout_match_changed"] += split.heldout_guard(w0) != split.heldout_guard(w1)
        problems["medical_match_changed"] += bool(med.find(w0)) != bool(med.find(w1))
        problems["term_occurrences_changed"] += split.occurrences(s["label"]) != split.occurrences(label)
        # the TTS input with forms vs without: the run's recorded tts_text where it has one, else the layer's rendering
        # of the sentence as it is now
        rec = plan.get(s["id"], {})
        want = rec["tts_text"] if "tts_text" in rec else lex.render(spoken)[0]
        got = pronounce.apply([{"id": s["id"], "label": label}], lex, spoken={s["id"]: spoken})[0][0]["tts_text"]
        problems["tts_text_changed"] += got != want
        tts_recorded += "tts_text" in rec
    log(json.dumps({"run": d.name, "sentences": len(sents), "changed": changed, "changed_in_kept_clips": changed_kept,
                    "changed_by_domain": dict(by_domain), "substitutions": sum(subs.values()),
                    "by_form": dict(subs.most_common()), "problems": dict(problems),
                    "tts_text_compared_with_recorded": tts_recorded, "examples": examples,
                    "forms": f.record()}, ensure_ascii=False))
    return 0 if not any(problems.values()) else 1


REUSED_TEXT = ("prompts.jsonl",)


def cmd_reuse_text(args) -> int:
    """Copy a finished run's raw text (prompts and generator outputs, not its hygiene) into a new run, so it can be
    rendered again without generating text; hygiene then runs in the new run under the current split, lists and
    forms. Refuses a target whose text differs."""
    import shutil
    from synthetic import textgen
    refuse_pairs(args, "its own text")
    if args.source in layout.PAIRS_RUNS or args.source in layout.ROUND_RUNS:
        raise SystemExit("run pairs' text is not reused")
    src, d = layout.run_dir(args.source), run_dir(args)
    if src.resolve() == d.resolve():
        raise SystemExit("--from and the target are the same run")
    files = [src / f for f in REUSED_TEXT] + sorted((src / "text").glob("*.jsonl"))
    for f in files:
        if not f.exists():
            raise SystemExit(f"{f} is missing (finish text generation in run {args.source} first)")
    for f in files:
        rel = f.relative_to(src)
        t = d / rel
        if t.exists():
            if t.read_bytes() != f.read_bytes():
                raise SystemExit(f"{t} exists and differs from {args.source}'s")
            continue
        t.parent.mkdir(parents=True, exist_ok=True)
        tmp = t.with_name(t.name + ".tmp")
        shutil.copyfile(f, tmp)
        tmp.replace(t)
    meta = {"run": d.name, "text_from": args.source, "prompts_sha256": _sha(d / "prompts.jsonl")}
    if (d / "run.json").exists() and json.loads((d / "run.json").read_text()) | {"created": None} != meta | {"created": None}:
        raise SystemExit(f"{d / 'run.json'} exists and differs")
    if not (d / "run.json").exists():
        write_json(d / "run.json", meta | {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    log(f"reused raw text of run {args.source}: {len(textgen.read_jsonl(d / 'prompts.jsonl'))} prompts -> {d} "
        "(run hygiene next)")
    return 0


def cmd_tts_plan(args) -> int:
    from synthetic import pronounce, textgen, tts, voicecheck
    d = run_dir(args)
    for a in layout.TTS_ENGINES:
        voicecheck.require_pass(a)
    sentences = textgen.read_jsonl(d / "sentences.jsonl")
    voices = {a: tts.ENGINES[a]().voices() for a in layout.TTS_ENGINES}
    spoken = {s["id"]: s["spoken"] for s in sentences if "spoken" in s}     # before label forms (synthetic/forms.py)
    plan, summary = pronounce.apply(tts.assign(sentences, voices, layout.SEED), pronounce.Lexicon.load(), spoken)
    p, ps = d / "tts_plan.jsonl", d / "pronunciation.json"
    if p.exists() and textgen.read_jsonl(p) != plan:
        raise SystemExit(f"{p} exists and differs (a changed lexicon or sentences needs a new run)")
    if ps.exists() and json.loads(ps.read_text()) != summary:
        raise SystemExit(f"{ps} exists and differs")
    textgen.write_jsonl(p, plan)
    write_json(ps, summary)
    log(f"tts plan: {len(plan)} clips; " + ", ".join(f"{a}: {sum(r['engine'] == a for r in plan)}" for a in layout.TTS_ENGINES)
        + f"; tts_text differs from the label in {summary['clips_changed']} clips, by rule "
        + json.dumps(summary["substitutions_by_rule"]))
    return 0


def cmd_tts(args) -> int:
    import os
    from synthetic import tts
    if args.device == "cuda" and args.declared_vram_gb is None:
        raise SystemExit("--device cuda needs --declared-vram-gb")
    tts.prepare(args.engine, args.device)   # before score_teacher, which imports torch
    import score_teacher
    from synthetic import voicecheck
    d = run_dir(args)
    voicecheck.require_pass(args.engine)
    if os.environ.get("RELABEL_DEPS") != args.engine:
        raise SystemExit(f"run with RELABEL_DEPS={args.engine} (the engine's packages)")
    if args.engine == "syn-tts-a":
        os.environ["PATH"] = str(layout.deps_dir("espeak-ng-wrap") / "bin") + os.pathsep + os.environ.get("PATH", "")
    try:
        with gpu_lock(args.device, f"synthetic tts {args.engine} {d.name}", args.lock_wait):
            eng = tts.ENGINES[args.engine](args.device, args.threads, args.declared_vram_gb)
            res = tts.render_chunk(d, eng, args.budget_minutes, log)
    except score_teacher.LockBusy as busy:
        print(f"refused: {busy}", file=sys.stderr)
        return LOCK_BUSY_EXIT
    log(json.dumps(res))
    return 0


def make_teacher(kind: str, setting: int, declared_vram_gb: float | None, device: str = "cuda"):
    """Stage 2's T* engine (labeling/engines.py) at one of its probed settings (labeling/checks.py)."""
    from labeling import checks, engines
    if kind == "transformers":
        s = checks.TRANSFORMERS_SETTINGS[setting]
        return engines.TransformersTeacher(s["batch_size"], s["max_batch_seconds"], device, declared_vram_gb)
    if device != "cuda":
        raise SystemExit("the vLLM engine runs on the GPU only")
    s = checks.VLLM_SETTINGS[setting]
    return engines.VllmTeacher(s["max_num_seqs"], s["gpu_memory_utilization"])


def cmd_roundtrip(args) -> int:
    import score_teacher
    from synthetic import roundtrip, textgen
    d = run_dir(args)
    n = len(textgen.read_jsonl(d / "tts_plan.jsonl"))
    if args.device != "cuda" and not args.smoke:
        raise SystemExit("--device cpu is for --smoke runs only")
    try:
        with gpu_lock(args.device, f"synthetic roundtrip {d.name}", args.lock_wait):
            eng = make_teacher(args.engine_kind, args.setting, args.declared_vram_gb, args.device)
            res = roundtrip.transcribe_chunk(d, eng, args.budget_minutes, expected_total=n, log=log)
    except score_teacher.LockBusy as busy:
        print(f"refused: {busy}", file=sys.stderr)
        return LOCK_BUSY_EXIT
    log(json.dumps(res, default=str))
    return 0


def cmd_finalize(args) -> int:
    from synthetic import pronounce, roundtrip, vocab
    d = run_dir(args)
    sp = split_of(args)
    report = roundtrip.finalize(d, vocab.Split.load(sp), listen=not args.no_listen, lexicon=pronounce.Lexicon.load(),
                                split_path=sp)
    log(json.dumps(report["total"]))
    return 0


# ----------------------------------------------------------------------------- confusable pairs (plan v5)

def cmd_pairs_words(args) -> int:
    from synthetic import confusable
    res = confusable.build_words(version=args.cleaning_version, log=log)
    lab = res["labels"]
    log(json.dumps({"tokens": res["tokens"], "distinct": res["distinct"]} |
                   {k: lab[k] for k in ("run", "through_batch", "cleaning_version", "records", "kept", "labels_counted",
                                        "store_identity_sha256")}))
    return 0


def cmd_pairs_detect(args) -> int:
    from synthetic import confusable
    out = confusable.run()
    log(json.dumps(out["summary"]))
    return 0


def cmd_pairs_split(args) -> int:
    from synthetic import pairs
    v3 = pairs.freeze_v3()
    log(json.dumps({"moved": len(v3["moved"]), "pairs": len(v3["pairs"]),
                    "trained": sum(p["trained"] for p in v3["pairs"]), "counts": v3["counts"], "groups": v3["groups"],
                    "dev_informed_groups": v3["dev_informed_groups"]}))
    return 0


def cmd_pairs_check_full(args) -> int:
    from synthetic import pairs
    log(json.dumps(pairs.check_full()))
    return 0


def cmd_pairs_coverage(args) -> int:
    from synthetic import coverage, pairs
    res = coverage.run(pairs.load_v3(), out=layout.terms_dir() / "coverage.v3.json")
    log(json.dumps({"totals": res["totals"], "underpowered": res["underpowered"], "missing": res["missing_parts"]}))
    return 0


def cmd_pairs_prompts(args) -> int:
    from synthetic import pairs, textgen
    split = pairs.load_v3()
    spec = pairs.load_spec(split)
    state = pairs.state_before(spec, split, args.round, args.run)
    prompts = pairs.round_prompts(spec, split, state, args.round, run=args.run)
    if not prompts:
        log(f"round {args.round}: no remaining need; nothing written (run pairs-finalize)")
        return 0
    d = pairs.round_dir(args.round, args.run)
    p = d / "prompts.jsonl"
    if p.exists():
        if textgen.read_jsonl(p) != prompts:
            raise SystemExit(f"{p} exists and differs (prompts are written once per round)")
        return 0
    textgen.write_jsonl(p, prompts)
    write_json(d / "run.json", {"run": f"{args.run}-r{args.round}", "prompts": len(prompts),
                                "seed": layout.SEED, "split_sha256": _sha(layout.split_version_path(3)),
                                "need": pairs.shortfalls(spec, state),
                                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    kinds = {k: sum(x["kind"] == k for x in prompts) for k in ("contrast", "single")}
    log(f"round {args.round}: {len(prompts)} prompts {kinds} -> {p}")
    return 0


def cmd_pairs_hygiene(args) -> int:
    from collections import Counter
    from labeling.filters import Coverage, b0_tokenizer_file
    from synthetic import hygiene, pairs, textgen, vocab
    d = pairs.round_dir(args.round, args.run)
    prompts = textgen.read_jsonl(d / "prompts.jsonl")
    missing = [s for s, _ in textgen.blocks(len(prompts)) if not (d / "text" / f"{s:07d}.jsonl").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} text blocks missing (finish textgen first)")
    records = [r for s, _ in textgen.blocks(len(prompts)) for r in textgen.read_jsonl(d / "text" / f"{s:07d}.jsonl")]
    split = pairs.load_v3()
    spec = pairs.load_spec(split)
    state = pairs.state_before(spec, split, args.round, args.run)
    normalize = vocab.default_normalize()
    coverage = Coverage(b0_tokenizer_file(layout.relabel_root() / "caches" / "b0-tokenizer"))
    sets, record = hygiene.heldout_sets()
    kept, report, rows = pairs.hygiene(records, split, vocab.medical_matcher(normalize), normalize, coverage, sets,
                                       record, prior=pairs.prior_sentences(args.round, args.run))
    selected, decisions = pairs.select(kept, {p["id"]: p for p in prompts}, spec, split, state)
    summary = dict(Counter(x["decision"] for x in decisions.values()))
    textgen.write_jsonl(d / "hygiene_rows.jsonl", rows)
    textgen.write_jsonl(d / "hygiene_kept.jsonl", kept)
    textgen.write_jsonl(d / "sentences.jsonl", sorted(selected, key=lambda x: x["id"]))
    write_json(d / "selection.json", {"summary": summary, "decisions": decisions,
                                      "counts_before": pairs.shortfalls(spec, state)})
    write_json(d / "hygiene.json", report | {"split_sha256": _sha(layout.split_version_path(3)),
                                             "selection": summary})
    log(f"pairs hygiene round {args.round}: kept {report['kept']} of {report['generated']}; reasons "
        f"{report['reasons']}; selected {len(selected)} {summary}; provisional={record['provisional']}")
    return 0


def cmd_pairs_finalize(args) -> int:
    from synthetic import pairs
    split = pairs.load_v3()
    rep = pairs.finalize_run(pairs.load_spec(split), split, args.run)
    log(json.dumps(rep["total"]))
    return 0


def cmd_pairs_metrics(args) -> int:
    from synthetic import pair_metrics
    models = {name: (Path(tech), Path(stem)) for name, tech, stem in args.model}
    contrasts = {f"{a}-{b}": (a, b) for a, b in args.contrast or []}
    res = pair_metrics.main(models, contrasts, args.out)
    log(json.dumps({g: res["groups"][g]["recall"] for g in ("stable-heldout", "pair-recall", "pair-recall-no-seed",
                                                            "partner-substitution", "near-forms")}))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m synthetic", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("aliases", nargs="+")
    t = sub.add_parser("termbuild")
    t.add_argument("--fetch", action="store_true")
    q = sub.add_parser("split")
    q.add_argument("--replace-reason", help="replace a frozen split that differs (a plan decision; the old split is "
                                            "archived and the side changes recorded)")
    sub.add_parser("coverage")
    sub.add_parser("voicecheck")
    n = sub.add_parser("noise-pool")
    n.add_argument("--workers", type=int, default=6)

    def run_args(q):
        g = q.add_mutually_exclusive_group(required=True)
        g.add_argument("--run", choices=layout.RUNS)
        g.add_argument("--smoke")

    q = sub.add_parser("prompts")
    run_args(q)
    q.add_argument("--n", type=int, required=True)
    q = sub.add_parser("textgen")
    run_args(q)
    q.add_argument("--device", default="cuda")
    q.add_argument("--declared-vram-gb", help="GB, or auto (a textgen probe's peak + 2 GB at the largest probed "
                                              "batch <= --batch that fits the lease)")
    q.add_argument("--batch", type=int, default=64)
    q.add_argument("--budget-minutes", type=float, default=110.0)
    q.add_argument("--lock-wait", type=float, default=1800.0)
    q = sub.add_parser("textgen-probe")
    q.add_argument("--batch", type=int, default=64, help="the largest size; its halvings down to --min-batch too")
    q.add_argument("--min-batch", type=int, default=16)
    q.add_argument("--batches", type=int, default=3)
    q.add_argument("--declared-vram-gb", type=float, required=True)
    q.add_argument("--lock-wait", type=float, default=1800.0)
    q = sub.add_parser("hygiene")
    run_args(q)
    q = sub.add_parser("forms-dry-run")
    run_args(q)
    q = sub.add_parser("reuse-text")
    run_args(q)
    q.add_argument("--from", dest="source", choices=layout.RUNS, required=True)
    q = sub.add_parser("tts-plan")
    run_args(q)
    q = sub.add_parser("tts")
    run_args(q)
    q.add_argument("--engine", choices=layout.TTS_ENGINES, required=True)
    q.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    q.add_argument("--threads", type=int, default=4)
    q.add_argument("--declared-vram-gb", type=float, help="GB (required with --device cuda; the job's lease is this "
                                                          "rounded up + 1 GB)")
    q.add_argument("--budget-minutes", type=float, default=110.0)
    q.add_argument("--lock-wait", type=float, default=1800.0)
    q = sub.add_parser("roundtrip")
    run_args(q)
    q.add_argument("--engine-kind", choices=("transformers", "vllm"), default="transformers")
    q.add_argument("--device", default="cuda")
    q.add_argument("--setting", type=int, default=0)
    q.add_argument("--declared-vram-gb", type=float)
    q.add_argument("--budget-minutes", type=float, default=110.0)
    q.add_argument("--lock-wait", type=float, default=1800.0)
    q = sub.add_parser("finalize")
    run_args(q)
    q.add_argument("--no-listen", action="store_true")
    q = sub.add_parser("pairs-words")
    q.add_argument("--cleaning-version", type=int, help="default: training.cleaning.VERSION (current at build time)")
    for name in ("pairs-detect", "pairs-split", "pairs-check-full", "pairs-coverage"):
        sub.add_parser(name)
    q = sub.add_parser("pairs-finalize")
    q.add_argument("--run", choices=layout.PAIRS_RUNS, default=layout.PAIRS_RUN)
    for name in ("pairs-prompts", "pairs-hygiene"):
        q = sub.add_parser(name)
        q.add_argument("--round", type=int, required=True, choices=range(1, len(layout.PAIRS_ROUNDS) + 1))
        q.add_argument("--run", choices=layout.PAIRS_RUNS, default=layout.PAIRS_RUN)
    q = sub.add_parser("pairs-metrics")
    q.add_argument("--model", nargs=3, action="append", required=True, metavar=("NAME", "TECH_DIR", "STEM_DIR"))
    q.add_argument("--contrast", nargs=2, action="append", metavar=("A", "B"))
    q.add_argument("--out", required=True)
    args = p.parse_args(argv)
    layout.require_mount()
    if args.cmd == "fetch":
        from synthetic import fetch
        for alias in args.aliases:
            fetch.fetch(alias)
        return 0
    if args.cmd == "termbuild":
        from synthetic import termbuild
        log(json.dumps(termbuild.build(fetch=args.fetch)))
        return 0
    if args.cmd == "split":
        from synthetic import vocab
        s = vocab.freeze(replace_reason=args.replace_reason)
        log(json.dumps({"counts": s["counts"], "groups": s["groups"], "medical_excluded": len(s["medical_excluded"]),
                        "duplicates": len(s["duplicates"]), "dropped": len(s["dropped"])}))
        return 0
    if args.cmd == "coverage":
        from synthetic import coverage
        res = coverage.run()
        log(json.dumps({"totals": res["totals"], "underpowered": res["underpowered"], "missing": res["missing_parts"]}))
        return 0
    if args.cmd == "voicecheck":
        from synthetic import voicecheck
        res = voicecheck.run()
        log(json.dumps({a: r["verdict"] for a, r in res["engines"].items()}))
        return 0 if all(r["verdict"] == "pass" for r in res["engines"].values()) else 1
    if args.cmd == "noise-pool":
        from synthetic import noise
        res = noise.build(args.workers, log)
        log(json.dumps(res["train_check"] | {"noise": res["noise"], "rirs": res["rirs"], "hours": res["hours"]})[:2000])
        return 0 if res["train_check"]["pass"] else 1
    return {"prompts": cmd_prompts, "textgen": cmd_textgen, "textgen-probe": cmd_textgen_probe, "hygiene": cmd_hygiene, "forms-dry-run": cmd_forms_dry_run, "reuse-text": cmd_reuse_text,
            "tts-plan": cmd_tts_plan,
            "tts": cmd_tts, "roundtrip": cmd_roundtrip, "finalize": cmd_finalize,
            "pairs-words": cmd_pairs_words, "pairs-detect": cmd_pairs_detect, "pairs-split": cmd_pairs_split,
            "pairs-check-full": cmd_pairs_check_full, "pairs-coverage": cmd_pairs_coverage,
            "pairs-prompts": cmd_pairs_prompts, "pairs-hygiene": cmd_pairs_hygiene,
            "pairs-finalize": cmd_pairs_finalize, "pairs-metrics": cmd_pairs_metrics}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
