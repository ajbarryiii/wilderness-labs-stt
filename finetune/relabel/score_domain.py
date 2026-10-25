"""Score the student baselines on the domain dev sets (labeling plan Part A "Baselines").

    ./python score_domain.py --model {B0,M1,MB1} [--sets dev-medical dev-technical] [--limit N]
                             [--smoke-build N] [--lock-wait SECONDS] [--force]
    ./python score_domain.py --table [--sets ...]   # markdown table of the production summaries (aliases only)

Default sets: devsets.DEFAULT_SETS (dev-medical, dev-technical), output scores/<model>/. Other sets (eval-ext)
are scored only when named, into scores/<sets>/<model>/, with the same cross-check.

Models (teachers are not scored here):
  B0   the pinned FP32 Parakeet-TDT-0.6B-v2 (evaluate.load_pretrained, lock-file hash checked)
  M1   the ternary M1 export (published as Kea-T), rebuilt by parakeet-ternary export.load_export; its weights'
       SHA-256 must equal binary_test_paths.M1_EXPORT_SHA256
  MB1  the binary MB1 export (published as Kea-B), rebuilt by binary_export.load_export; SHA-256 must equal
       binary_test_paths.WEIGHTS_SHA256 (MB1's test-scoring identity)
Each model's greedy TDT decoder is pinned to binary_paths.DECODER_GRAPHS_MODE (the mode M1 and MB1 decoded
in; binary_train.pin_decoder_graphs_mode) and checked after every part.

Decoding and scoring are parakeet-ternary evaluate.py's, unchanged: evaluate.evaluate_set per part (strict
FP32, greedy TDT, Whisper normalizer, corpus WER over scored utterances) at batch 64 and 1200 s per batch
(the M1-test and MB1-test values). Per-utterance records keep evaluate.py's schema, plus "terms" on
dev-medical parts.

Critical-term recall (sets in devsets.TERM_KIND, with that set's list: medical for dev-medical and eval-ext,
technical for eval-stem): the committed term list (terms.matcher(kind), normalized
with the same normalizer) is matched greedily (longest first, non-overlapping) in each scored reference's
normalized words. An occurrence is reproduced if, in the word alignment of reference and hypothesis
(relabel_align.align: the edit_counts dynamic program with its backtrace), every one of its words is a
match and no hypothesis word is inserted between its words, or (spelling variants, 2026-10-24) another listed form
of the same term (terms.variant_key: "dataset" / "data set") is a contiguous run of the hypothesis words aligned
to the occurrence or inserted in its gaps, including one aligned word. An occurrence is counted once, whichever
listed form it has. Recall = reproduced / occurrences, per part.

Reported: per part WER (and S, D, I, words, utterances, hours) and term recall; per set the unweighted mean
of its parts' WERs, and for dev-medical the unweighted mean of its parts' term recalls (pooled counts too).

Before the domain parts, a cross-check decodes the full ami_dev set with the same loader and settings and
compares its WER with the model's recorded dev score (relabel_paths.STUDENT_RECORDS); a difference above
CROSSCHECK_POINTS refuses the run before anything is written (wrong weights, loader or runtime).

Outputs: $RELABEL_ROOT/devsets/scores/<model>/<part>.json, crosscheck.json and summary.json (written first
as "in_progress", final last). --limit N (first N id-sorted utterances per part) or --smoke-build N
(manifests of a smoke build) write to $RELABEL_ROOT/smoke/devsets-scores/<model>-..., never the production
path. A part already written with the same configuration is skipped unless --force.

GPU: the parakeet GPU lock (score_teacher.gpu_lock, relabel_paths.GPU_LOCK) is taken before CUDA is touched
and held until the end; it refuses to start if the lock is not free within --lock-wait seconds (exit 3).
The job's GPU lease is the caller's (devsets/launch.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import devsets
import evaluate  # finetune/parakeet-ternary/evaluate.py
import relabel_align
import relabel_paths
import terms
from devsets import layout

MODELS = ("B0", "M1", "MB1")
BATCH_SIZE = 64
MAX_BATCH_SECONDS = 1200.0
CROSSCHECK_SET = "ami_dev"
CROSSCHECK_POINTS = 0.10
M1_EXPORT = relabel_paths.PARAKEET_TERNARY / "runs" / "main-M1-P2-lr5e-4" / "export"
LOCK_BUSY_EXIT = 3


def binary_on_path() -> None:
    """finetune/parakeet-binary after parakeet-ternary on sys.path (its modules are named binary_*)."""
    here = str(Path(__file__).resolve().parents[1] / "parakeet-binary")
    if here not in sys.path:
        sys.path.append(here)


# --- term recall ----------------------------------------------------------------------------------

def _variant_at(a, hyp: list[str], start: int, end: int, forms) -> bool:
    """Whether a listed form appears in the hypothesis at the position aligned to ref[start:end]: as a contiguous
    run of the hypothesis words aligned to that span or inserted in its gaps (start..end), including at least one
    word aligned to the span."""
    aligned = [j for op, i, j in a.pairs if op in ("eq", "sub") and start <= i < end]
    if not aligned:
        return False
    window = aligned + [j for op, g, j in a.pairs if op == "ins" and start <= g <= end]
    lo, hi = min(window), max(window)
    for form in forms:
        n = len(form)
        for k in range(lo, hi - n + 2):
            if tuple(hyp[k:k + n]) == form and any(k <= j < k + n for j in aligned):
                return True
    return False


def term_occurrences(ref_norm: str, hyp_norm: str, matcher: terms.Matcher,
                     variants: bool = True) -> list[tuple[int, int, bool]]:
    """(start, end, reproduced) for every term occurrence in the reference (see module docstring). variants=False is
    the rule before 2026-10-24 (term_rescore.py's "old" column only)."""
    ref, hyp = ref_norm.split(), hyp_norm.split()
    found = matcher.find(ref)
    if not found:
        return []
    a = relabel_align.align(ref, hyp)
    out = []
    for occ in found:
        matched = all(i in a.matched for i in range(occ.start, occ.end))
        inserted = any(g in a.ins_gaps for g in range(occ.start + 1, occ.end))
        ok = matched and not inserted
        if not ok and variants:
            others = [v for v in matcher.variants(occ.phrase) if v != occ.phrase]
            ok = bool(others) and _variant_at(a, hyp, occ.start, occ.end, others)
        out.append((occ.start, occ.end, ok))
    return out


def term_recall(records: list[dict], matcher: terms.Matcher) -> dict:
    """Adds "terms" to each scored record; returns occurrences, reproduced and recall (None if no occurrence)."""
    occurrences = reproduced = 0
    for r in records:
        if not r.get("scored", True):
            continue
        found = term_occurrences(r["ref_norm"], r["hyp_norm"], matcher)
        r["terms"] = [list(t) for t in found]
        occurrences += len(found)
        reproduced += sum(t[2] for t in found)
    return {"occurrences": occurrences, "reproduced": reproduced,
            "recall": reproduced / occurrences if occurrences else None}


# --- summaries ------------------------------------------------------------------------------------

def part_entry(result: dict) -> dict:
    w = result["wer"]
    entry = {"wer": 100 * w["wer"], "substitutions": w["substitutions"], "deletions": w["deletions"],
             "insertions": w["insertions"], "ref_words": w["ref_words"], "utterances": result["utterances"],
             "scored_utterances": w["utterances"], "hours": result["audio_hours"], "limit": result["limit"]}
    if "term_recall" in result:
        entry["term_recall"] = result["term_recall"]
    return entry


def summarize(model: str, results: dict[str, dict], sets: list[str]) -> dict:
    parts = {name: part_entry(r) for name, r in results.items()}
    per_set = {}
    for name in sets:
        members = devsets.SETS[name]
        if not all(p in parts for p in members):
            continue
        entry = {"parts": list(members), "mean_wer": sum(parts[p]["wer"] for p in members) / len(members),
                 "mean_note": "unweighted mean of the parts' corpus WERs"}
        if name in devsets.TERM_KIND:
            recalls = [parts[p]["term_recall"]["recall"] for p in members]
            entry["mean_term_recall"] = (sum(recalls) / len(recalls)) if all(r is not None for r in recalls) else None
            occ = sum(parts[p]["term_recall"]["occurrences"] for p in members)
            rep = sum(parts[p]["term_recall"]["reproduced"] for p in members)
            entry["pooled_term_recall"] = {"occurrences": occ, "reproduced": rep, "recall": rep / occ if occ else None}
        per_set[name] = entry
    first = next(iter(results.values()))
    return {"model": model, "status": "complete", "source": first["source"], "parts": parts, "sets": per_set,
            "limit": first["limit"], "batch_size": BATCH_SIZE, "max_batch_seconds": MAX_BATCH_SECONDS}


# --- models -----------------------------------------------------------------------------------------

def expected_sha256(model: str) -> str | None:
    if model == "B0":
        return None  # describe_source("pretrained") checks the lock file
    binary_on_path()
    import binary_test_paths as tp

    return tp.M1_EXPORT_SHA256 if model == "M1" else tp.WEIGHTS_SHA256


def describe(model: str) -> dict:
    binary_on_path()
    if model == "B0":
        source = evaluate.describe_source("pretrained", None, "pretrained Parakeet-TDT-0.6B-v2, FP32, no training")
    elif model == "M1":
        source = evaluate.describe_source("export", M1_EXPORT, "M1 ternary export (Kea-T), rebuilt with dequantized "
                                          "FP32 weights")
    else:
        import binary_test_paths as tp

        source = evaluate.describe_source("export", tp.EXPORT, "MB1 binary export (Kea-B), rebuilt with dequantized "
                                          "FP32 weights")
    expected = expected_sha256(model)
    if expected is not None and source["sha256"] != expected:
        raise SystemExit(f"{model}: weights SHA-256 {source['sha256']} is not the pinned {expected}")
    source["model"] = model
    return source


def load_model(model: str, device):
    """(model on device in eval mode, its pinned decoder mode)."""
    binary_on_path()
    import binary_train  # noqa: I001  imports parakeet-ternary train.py first (CUDA/numba configuration)

    if model == "B0":
        net = evaluate.load_pretrained(device)
    elif model == "M1":
        import export

        net = export.load_export(M1_EXPORT, device)
    else:
        import binary_export
        import binary_test_paths as tp

        net = binary_export.load_export(tp.EXPORT, device)
    return net.eval(), binary_train.pin_decoder_graphs_mode(net, model)


def require_mode(net, model: str, where: str) -> str:
    binary_on_path()
    import binary_train

    return binary_train.require_decoder_graphs_mode(net, f"{model} {where}")


# --- inputs and outputs ---------------------------------------------------------------------------

def sets_key(sets) -> str | None:
    """None if every requested set is a default set (devsets.DEFAULT_SETS), else the sets joined with "+"."""
    sets = list(sets or devsets.DEFAULT_SETS)
    return None if set(sets) <= set(devsets.DEFAULT_SETS) else "+".join(sets)


def output_dir(model: str, limit: int | None, smoke_build: int | None, sets=None) -> Path:
    """scores/<model> for the default sets, scores/<sets>/<model> for others (e.g. scores/eval-ext/B0)."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    key = sets_key(sets)
    if limit is None and smoke_build is None:
        return layout.scores_root() / model if key is None else layout.scores_root() / key / model
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    name = model + (f"-limit{limit}" if limit is not None else "") + (f"-build{smoke_build}" if smoke_build else "")
    out = layout.smoke_scores_root() / (name if key is None else f"{key}/{name}")
    assert not out.is_relative_to(layout.production_root())
    return out


def part_manifests(parts: list[str], smoke_build: int | None) -> dict[str, tuple[Path, dict]]:
    """(manifest, meta) per part; refuses a part whose build is incomplete or whose manifest changed."""
    lay = layout.Layout.smoke(smoke_build) if smoke_build else layout.Layout.production()
    out = {}
    for part in parts:
        meta_path, manifest = lay.meta(part), lay.manifest(part)
        if not meta_path.exists() or not manifest.exists():
            raise SystemExit(f"{part}: no complete build ({meta_path}); run devsets.build first")
        meta = json.loads(meta_path.read_text())
        if not meta.get("complete") or meta.get("manifest_sha256") != evaluate.sha256_file(manifest):
            raise SystemExit(f"{part}: {manifest} does not match its completed build meta")
        out[part] = (manifest, meta)
    return out


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def run_config(model: str, source: dict, manifest_sha256: str, limit: int | None) -> dict:
    return {"model": model, "weights_sha256": source["sha256"], "manifest_sha256": manifest_sha256, "limit": limit,
            "batch_size": BATCH_SIZE, "max_batch_seconds": MAX_BATCH_SECONDS}


def crosscheck(net, model: str, source: dict, device) -> dict:
    """Full CROSSCHECK_SET with this loader and settings vs the model's recorded dev WER."""
    recorded_path = relabel_paths.STUDENT_RECORDS[model] / f"{CROSSCHECK_SET}.json"
    recorded = json.loads(recorded_path.read_text())
    name, manifest = evaluate.resolve_manifest(CROSSCHECK_SET)
    result = evaluate.evaluate_set(net, name, manifest, source, BATCH_SIZE, None, device, MAX_BATCH_SECONDS)
    old = {r["id"]: r["hyp"] for r in recorded["records"]}
    same = sum(old.get(r["id"]) == r["hyp"] for r in result["records"])
    wer_now = 100 * result["wer"]["wer"]
    wer_recorded = 100 * (recorded["wer"]["wer"] if isinstance(recorded["wer"], dict) else recorded["wer"])
    return {"set": name, "recorded": str(recorded_path), "recorded_wer": wer_recorded, "wer": wer_now,
            "difference_points": wer_now - wer_recorded, "tolerance_points": CROSSCHECK_POINTS,
            "identical_hypotheses": same, "utterances": len(result["records"]),
            "pass": abs(wer_now - wer_recorded) <= CROSSCHECK_POINTS}


def score(model: str, sets: list[str], limit: int | None, smoke_build: int | None, device_name: str,
          lock_wait: float, force: bool) -> int:
    import torch

    import score_teacher

    relabel_paths.require_mount()
    parts = [p for name in sets for p in devsets.SETS[name]]
    inputs = part_manifests(parts, smoke_build)
    out = output_dir(model, limit, smoke_build, sets)
    source = describe(model)
    configs = {p: run_config(model, source, meta["manifest_sha256"], limit) for p, (_, meta) in inputs.items()}
    todo = []
    for part in parts:
        path = out / f"{part}.json"
        if not force and path.exists() and json.loads(path.read_text()).get("config") == configs[part]:
            continue
        todo.append(part)
    out.mkdir(parents=True, exist_ok=True)
    if todo:
        write_json(out / "summary.json", {"model": model, "status": "in_progress", "complete": False})
        device = torch.device(device_name)
        label = f"score_domain {model}"
        lock = score_teacher.gpu_lock(label, lock_wait) if device.type == "cuda" else _nolock()
        try:
            with lock:
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise SystemExit("CUDA is required")
                net, mode = load_model(model, device)
                check = crosscheck(net, model, source, device) if limit is None and smoke_build is None else None
                if check is not None:
                    write_json(out / "crosscheck.json", check)
                    print(f"crosscheck {check['set']}: WER {check['wer']:.3f} vs recorded {check['recorded_wer']:.3f} "
                          f"({check['identical_hypotheses']}/{check['utterances']} identical)", flush=True)
                    if not check["pass"]:
                        raise SystemExit(f"{model}: cross-check failed ({check['difference_points']:+.3f} points)")
                matchers = {kind: terms.matcher(kind) for kind in set(devsets.TERM_KIND.values())}
                for part in todo:
                    manifest, meta = inputs[part]
                    result = evaluate.evaluate_set(net, part, manifest, source, BATCH_SIZE, limit, device,
                                                   MAX_BATCH_SECONDS)
                    after = require_mode(net, model, f"after {part}")
                    kind = devsets.TERM_KIND.get(devsets.set_of(part))
                    if kind is not None:
                        result["term_recall"] = term_recall(result["records"], matchers[kind])
                        result["term_kind"] = kind
                    result.update(config=configs[part], decoder_graphs_mode=after, part=part,
                                  set=devsets.set_of(part), build_config_hash=meta["config_hash"])
                    write_json(out / f"{part}.json", result)
                    print(f"{part}: n={result['utterances']} WER {100 * result['wer']['wer']:.2f}%"
                          + recall_text(result.get("term_recall")), flush=True)
                del net
        except score_teacher.LockBusy as busy:
            print(f"refused: {busy}", file=sys.stderr)
            return LOCK_BUSY_EXIT
    results = {p: json.loads((out / f"{p}.json").read_text()) for p in parts}
    summary = summarize(model, results, sets)
    summary["decoder_graphs_mode"] = mode if todo else results[parts[0]].get("decoder_graphs_mode")
    if (out / "crosscheck.json").exists():
        summary["crosscheck"] = json.loads((out / "crosscheck.json").read_text())
    write_json(out / "summary.json", summary)
    for name, entry in summary["sets"].items():
        extra = f", mean term recall {entry['mean_term_recall']:.4f}" if entry.get("mean_term_recall") is not None else ""
        print(f"{name}: mean WER {entry['mean_wer']:.2f}{extra}")
    return 0


def recall_text(tr: dict | None) -> str:
    """" term recall R (k/n)" for progress lines; recall is undefined (N/A) without occurrences."""
    if tr is None:
        return ""
    value = "N/A" if tr["recall"] is None else f"{tr['recall']:.4f}"
    return f" term recall {value} ({tr['reproduced']}/{tr['occurrences']})"


class _nolock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def table(sets=None) -> str:
    """Markdown tables (aliases only) from the production summaries of `sets` (default sets) that exist."""
    rows, sets_rows = [], []
    for model in MODELS:
        path = output_dir(model, None, None, sets) / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        if s.get("status") != "complete":
            continue
        for part, e in s["parts"].items():
            tr = e.get("term_recall")
            recall = "–" if tr is None else "N/A" if tr["recall"] is None else f"{100 * tr['recall']:.1f}"
            rows.append(f"| {model} | {part} | {e['utterances']} | {e['hours']:.2f} | {e['wer']:.2f} | "
                        + (f"{recall} ({tr['reproduced']}/{tr['occurrences']})" if tr else recall) + " |")
        for name, e in s["sets"].items():
            mtr = e.get("mean_term_recall")
            sets_rows.append(f"| {model} | {name} | {e['mean_wer']:.2f} | "
                             + (f"{100 * mtr:.1f}" if mtr is not None else "–") + " |")
    return "\n".join(["| Model | Part | Utterances | Hours | WER % | Term recall % |", "|---|---|---:|---:|---:|---:|",
                      *rows, "", "| Model | Set | Mean WER % (unweighted) | Mean term recall % |",
                      "|---|---|---:|---:|", *sets_rows])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--sets", nargs="+", default=list(devsets.DEFAULT_SETS), choices=list(devsets.SETS),
                        help="default: the domain dev sets; eval-ext only on request (own output directory)")
    parser.add_argument("--limit", type=int, help="first N id-sorted utterances per part (smoke path)")
    parser.add_argument("--smoke-build", type=int, help="score the manifests of `devsets.build --smoke N` (smoke path)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lock-wait", type=float, default=1800.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args(argv)
    if not args.table and not args.model:
        parser.error("--model is required")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.device == "cpu" and args.limit is None:
        parser.error("--device cpu is for --limit smoke runs only")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.table:
        print(table(args.sets))
        return 0
    return score(args.model, args.sets, args.limit, args.smoke_build, args.device, args.lock_wait, args.force)


if __name__ == "__main__":
    sys.exit(main())
