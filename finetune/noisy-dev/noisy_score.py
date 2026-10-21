"""Score models on the noisy development-set copy, and compare them. See plans/build.md "Scoring".

    ./python noisy_score.py score --model {b0,ternary,binary} [--path EXPORT_DIR] --name NAME
                                  [--build DIR] [--limit N] [--batch-size 32] [--overwrite]
    ./python noisy_score.py compare --names B0 M1 ... [--reference B0] [--resamples 2000]

(GPU: only when the GPU is free, inside ../parakeet-ternary/heavy; takes the shared GPU lock.)

Models: B0 is the pinned FP32 .nemo (train.verify_base, train.load_pretrained); M1 and every
binary pilot arm are scored on the model rebuilt from their export (parakeet-ternary export.py or
parakeet-binary binary_export.py through binary_train.FinishingExport, which pins the decoder of
the rebuilt model). Decoding is evaluate.decode (the transcribe() path, strict FP32, the Whisper
normalizer, leaderboard exclusions) with binary_eval's batch size 32, and NeMo's greedy TDT decoder
is pinned to binary_paths.DECODER_GRAPHS_MODE ("no_while_loops", M1's mode) before every decode and
checked after it (DecoderModeError otherwise). ./python omits NVRTC, as parakeet-binary/python.

Before scoring (condition_rows, review round 2): build.json must be complete; its lock SHA-256
must be that of build.lock.json and its manifest SHA-256 that of build_manifest.jsonl; every
condition manifest must hold exactly the build manifest's rows (IDs, references, durations, in
order), each pointing at the expected audio file (the original for clean, OUT/audio/<condition>/
... otherwise), and every audio file must have the SHA-256 the build recorded. The summary
records this build identity, including its provenance (sample size, skipped conditions,
missing pool sources, utterances; "full" only for a complete build of every condition on the
full dev sets). Only a full build without --limit may write scores/<name>; a sample or partial
build goes to scores/smoke-<name>-sample<N> (or -partial), --limit N to ...-limit<N> (review
round 3, finding 4), and the summary says "smoke": true. Writes
scores/<name>/<condition>.json (per-utterance records) and scores/<name>/summary.json last.

Aggregation (summarize): per condition, the WER of each set (corpus WER over scored rows, as
evaluate.wer_summary), the equal-weight mean of the five set WERs (the selection metric's formula)
and the four-set human-reference mean (YODAS, whose references are machine-generated, is labeled
"pseudo-reference agreement" and left out); the noisy mean over the noisy conditions; each
condition's degradation against the model's own clean row.

Comparison (compare): the gap of each model to the reference (B0) per condition and for the noisy
mean, and each model's degradation, with 95 % intervals from a paired cluster bootstrap: 2,000
resamples stratified by dev set, clusters (speakers, meetings, videos; from the build manifest)
resampled jointly for all conditions and models, so the pairing is preserved. Models are compared
only if they scored the same build (lock and manifest SHA-256, condition list) and every
utterance has the same normalized reference and exclusion flag in every model and condition.
The intervals cover speech sampling only: they are conditional on the frozen noise realization
(fixed DEMAND environments, hearings, FSD50K clips, BUT rooms). The per-condition WER difference
is also broken down by noise-source group (by_group: FSD50K class, DEMAND environment, Zenodo
microphone, BUT room; pooled corpus WER within a group), so a result driven by one source shows.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

import binary_paths  # parakeet-binary (light: paths only)
import noisy_paths

DECODER_GRAPHS_MODE = binary_paths.DECODER_GRAPHS_MODE
EVAL_BATCH_SIZE = 32       # parakeet-binary binary_eval / train.py --eval-batch-size default
PSEUDO_SETS = ("yodas_dev",)
PSEUDO_LABEL = "pseudo-reference agreement"
HUMAN_LABEL = "WER (human reference)"
RESAMPLES = 2000
KINDS = ("b0", "ternary", "binary")


class DecoderModeError(RuntimeError):
    """A model would decode (or decoded) in another NeMo CUDA-graph mode than M1's."""


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    kind: str            # b0 (FP32 original .nemo), ternary (export), binary (export)
    path: Path | None    # export directory (ternary, binary); None for b0
    name: str            # output directory name under scores/

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown model kind {self.kind!r}")
        if (self.kind == "b0") != (self.path is None):
            raise ValueError("an export path is required for ternary and binary models and not accepted for b0")
        noisy_paths.named_dir(noisy_paths.SCORES, self.name)


@dataclasses.dataclass
class Hooks:
    gpu_lock: Callable           # label -> context manager
    load_model: Callable         # (ModelSpec, device) -> model, decoder pinned
    pin_decoder: Callable        # (model, role) -> mode
    require_decoder: Callable    # (model, role) -> mode, or DecoderModeError
    decode: Callable             # (model, manifest rows, batch_size, device) -> evaluate.decode records
    deterministic: Callable      # () -> dict


def utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[noisy_score {dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------- production hooks

def pin_decoder_graphs_mode(model, role: str) -> str:
    """binary_train.pin_decoder_graphs_mode (parakeet-binary): force and verify M1's decoder mode."""
    import binary_train
    try:
        return binary_train.pin_decoder_graphs_mode(model, role)
    except binary_train.DecoderModeError as error:
        raise DecoderModeError(str(error)) from error


def require_decoder_graphs_mode(model, role: str) -> str:
    """binary_train.require_decoder_graphs_mode: the mode after decoding must still be M1's."""
    import binary_train
    try:
        return binary_train.require_decoder_graphs_mode(model, role)
    except binary_train.DecoderModeError as error:
        raise DecoderModeError(str(error)) from error


def _load_model(spec: ModelSpec, device: str):
    import binary_train  # noqa: I001  imports train first (CUDA/numba configuration)
    import train
    if spec.kind == "b0":
        train.verify_base()
        model = train.load_pretrained(device)
        pin_decoder_graphs_mode(model, "B0")
    else:
        module = train.local_module("export") if spec.kind == "ternary" else __import__("binary_export")
        model = binary_train.FinishingExport(module).load_export(Path(spec.path), device)  # pins the decoder
    return model.eval()


def _decode(model, rows: list[dict], batch_size: int, device: str) -> list[dict]:
    import train
    return train.local_module("evaluate").decode(model, rows, batch_size, device)


def _deterministic() -> dict:
    import train
    return train.deterministic_mode()


def _gpu_lock(label: str):
    import paths
    return paths.gpu_lock(label)


def default_hooks() -> Hooks:
    return Hooks(gpu_lock=_gpu_lock, load_model=_load_model, pin_decoder=pin_decoder_graphs_mode,
                 require_decoder=require_decoder_graphs_mode, decode=_decode, deterministic=_deterministic)


def describe_model(spec: ModelSpec) -> dict:
    out = {"kind": spec.kind, "name": spec.name, "path": str(spec.path) if spec.path else None}
    if spec.path is not None:
        manifest = Path(spec.path) / "manifest.json"
        if manifest.exists():
            m = json.loads(manifest.read_text())
            out.update(export_manifest_sha256=noisy_paths.sha256_file(manifest), export_format=m.get("format"),
                       export_files=m.get("files"))
        out["scored_on"] = "model rebuilt from its export"
    else:
        out["scored_on"] = "FP32 original .nemo (paths.MODEL_FILE, verified against lock.json)"
    return out


# --------------------------------------------------------------------------------------------- aggregation

def _counts(records: list[dict]) -> tuple[int, int, int, int]:
    """(errors, reference words, scored utterances, excluded) over scored records, as wer_summary."""
    errors = words = scored = 0
    for r in records:
        if not r.get("scored", True):
            continue
        errors += r["S"] + r["D"] + r["I"]
        words += len(r["ref_norm"].split())
        scored += 1
    return errors, words, scored, len(records) - scored


def set_entry(name: str, records: list[dict]) -> dict:
    errors, words, scored, excluded = _counts(records)
    if not words:
        raise ValueError(f"{name}: no reference words to score")
    return {"wer": errors / words, "errors": errors, "ref_words": words, "utterances": len(records),
            "scored_utterances": scored, "excluded_empty_reference": excluded,
            "label": PSEUDO_LABEL if name in PSEUDO_SETS else HUMAN_LABEL}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(results: dict[str, dict[str, list[dict]]]) -> dict:
    """Per-condition set WERs, five-set and four-set human means, noisy mean and degradations."""
    out = {}
    for cond, sets in results.items():
        entries = {name: set_entry(name, records) for name, records in sets.items()}
        human = [e["wer"] for n, e in entries.items() if n not in PSEUDO_SETS]
        out[cond] = {"sets": entries, "mean_wer": _mean([e["wer"] for e in entries.values()]),
                     "human_mean_wer": _mean(human), "mean_sets": list(entries),
                     "human_mean_sets": [n for n in entries if n not in PSEUDO_SETS]}
    noisy = [c for c in out if c != "clean"]
    if "clean" in out:
        for cond in noisy:
            out[cond]["degradation"] = out[cond]["mean_wer"] - out["clean"]["mean_wer"]
            out[cond]["human_degradation"] = out[cond]["human_mean_wer"] - out["clean"]["human_mean_wer"]
    summary = {"conditions": out, "noisy_conditions": noisy,
               "noisy_mean_wer": _mean([out[c]["mean_wer"] for c in noisy]),
               "noisy_human_mean_wer": _mean([out[c]["human_mean_wer"] for c in noisy]),
               "aggregation": "equal-weight mean of the set WERs (corpus WER per set); noisy mean: equal-weight "
                              "mean over the noisy conditions"}
    if "clean" in out and noisy:
        summary["noisy_degradation"] = summary["noisy_mean_wer"] - out["clean"]["mean_wer"]
    return summary


def _cluster_arrays(models: dict, clusters: dict) -> tuple[list[str], dict, dict]:
    """Per set: the sorted cluster list; per (model, condition, set): errors per cluster; per set:
    reference words per cluster (identical for every model and condition, else ValueError)."""
    first = next(iter(models.values()))
    sets = list(next(iter(first.values())))
    errors, words, index, refs = {}, {}, {}, {}
    for s in sets:
        ids = [r["id"] for r in next(iter(first.values()))[s]]
        names = sorted({clusters[(s, uid)] for uid in ids})
        index[s] = ({uid: names.index(clusters[(s, uid)]) for uid in ids}, len(names))
        refs[s] = {r["id"]: (r["ref_norm"], bool(r.get("scored", True))) for r in next(iter(first.values()))[s]}
    for m, conds in models.items():
        for c, per_set in conds.items():
            if list(per_set) != sets:
                raise ValueError(f"{m}/{c}: sets differ")
            for s, records in per_set.items():
                where, k = index[s]
                if sorted(r["id"] for r in records) != sorted(where):
                    raise ValueError(f"{m}/{c}/{s}: utterances differ from the first model's (pairing)")
                for r in records:
                    ref, scored = refs[s][r["id"]]
                    if r["ref_norm"] != ref:
                        raise ValueError(f"{m}/{c}/{s} {r['id']}: normalized reference differs (pairing)")
                    if bool(r.get("scored", True)) != scored:
                        raise ValueError(f"{m}/{c}/{s} {r['id']}: exclusion mask differs (pairing)")
                cl = np.array([where[r["id"]] for r in records])
                scored = np.array([bool(r.get("scored", True)) for r in records])
                e = np.array([r["S"] + r["D"] + r["I"] for r in records], dtype=np.float64) * scored
                w = np.array([len(r["ref_norm"].split()) for r in records], dtype=np.float64) * scored
                errors[(m, c, s)] = np.bincount(cl, weights=e, minlength=k)
                wk = np.bincount(cl, weights=w, minlength=k)
                if s in words and not np.array_equal(words[s], wk):
                    raise ValueError(f"{m}/{c}/{s}: reference words differ (pairing)")
                words[s] = wk
    return sets, errors, words


def bootstrap_set_wers(models: dict, clusters: dict, resamples: int = RESAMPLES,
                       seed: int = noisy_paths.BUILD_SEED) -> dict:
    """draws[model][condition][set]: corpus WER of each of `resamples` paired cluster resamples
    (clusters drawn with replacement within each set; the same draws for every model and condition)."""
    sets, errors, words = _cluster_arrays(models, clusters)
    rng = np.random.Generator(np.random.PCG64(seed))
    counts = {}
    for s in sets:
        k = len(words[s])
        counts[s] = rng.multinomial(k, np.full(k, 1.0 / k), size=resamples).astype(np.float64)
    draws: dict = {}
    for (m, c, s), e in errors.items():
        denom = counts[s] @ words[s]
        with np.errstate(invalid="ignore", divide="ignore"):
            draws.setdefault(m, {}).setdefault(c, {})[s] = np.where(denom > 0, (counts[s] @ e) / denom, np.nan)
    return draws


def _interval(point: float, draws: np.ndarray) -> dict:
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"point": float(point), "low": float(lo), "high": float(hi),
            "resolved": bool(lo > 0 or hi < 0)}


IDENTITY_KEYS = ("lock_sha256", "manifest_sha256", "conditions", "sample", "full", "skipped_conditions",
                 "pool_missing_sources", "limit")


def group_breakdown(models: dict, groups: dict, reference: str) -> dict:
    """by_group[condition][kind][group]: utterances, pooled corpus WER per model and its gap to the
    reference, over the rows of that noise-source group (point estimates)."""
    out: dict = {}
    for cond in models[reference]:
        acc: dict = {}
        for m, conds in models.items():
            for s, records in conds[cond].items():
                for r in records:
                    for kind, group in (groups.get((cond, s, r["id"])) or {}).items():
                        cell = acc.setdefault(kind, {}).setdefault(group, {}).setdefault(m, [0, 0, 0])
                        if r.get("scored", True):
                            cell[0] += r["S"] + r["D"] + r["I"]
                            cell[1] += len(r["ref_norm"].split())
                        cell[2] += 1
        for kind, by_group in acc.items():
            for group, per_model in sorted(by_group.items()):
                wer = {m: e / w if w else None for m, (e, w, _) in per_model.items()}
                out.setdefault(cond, {}).setdefault(kind, {})[group] = {
                    "utterances": per_model[reference][2], "wer": wer,
                    "gap_to_reference": {m: (v - wer[reference]) if v is not None and wer[reference] is not None
                                         else None for m, v in wer.items()}}
    return out


def compare(models: dict[str, dict[str, dict[str, list[dict]]]], clusters: dict, reference: str = "B0",
            resamples: int = RESAMPLES, seed: int = noisy_paths.BUILD_SEED, *, builds: dict,
            groups: dict | None = None) -> dict:
    """Gaps to the reference model and degradations, with paired cluster-bootstrap 95 % intervals,
    for models that scored the same build (builds: model -> summary["build"])."""
    if reference not in models:
        raise ValueError(f"reference model {reference!r} not among {sorted(models)}")
    if set(builds) != set(models):
        raise ValueError("a build identity is required for every model")
    ref_identity = {k: builds[reference].get(k) for k in IDENTITY_KEYS}
    for m, identity in builds.items():
        if {k: identity.get(k) for k in IDENTITY_KEYS} != ref_identity:
            raise ValueError(f"{m}: scored another build than {reference} (build identity differs)")
    smoke = not ref_identity.get("full", True) or ref_identity.get("limit") is not None
    for m, conds in models.items():
        if list(conds) != list(models[reference]):
            raise ValueError(f"{m}: condition set {list(conds)} differs from {reference}'s")
    draws = bootstrap_set_wers(models, clusters, resamples, seed)
    points = {m: summarize(r) for m, r in models.items()}

    def mean_draw(m, c, human=False):
        sets = [s for s in draws[m][c] if not (human and s in PSEUDO_SETS)]
        return np.mean([draws[m][c][s] for s in sets], axis=0)

    out = {}
    conds = list(models[reference])
    noisy = [c for c in conds if c != "clean"]
    for m in models:
        entry = {"gap_to_reference": {}, "human_gap_to_reference": {}, "degradation": {}}
        p = points[m]["conditions"]
        pr = points[reference]["conditions"]
        for c in conds:
            entry["gap_to_reference"][c] = _interval(p[c]["mean_wer"] - pr[c]["mean_wer"],
                                                     mean_draw(m, c) - mean_draw(reference, c))
            entry["human_gap_to_reference"][c] = _interval(
                p[c]["human_mean_wer"] - pr[c]["human_mean_wer"], mean_draw(m, c, True) - mean_draw(reference, c, True))
            if c != "clean" and "clean" in p:
                entry["degradation"][c] = _interval(p[c]["mean_wer"] - p["clean"]["mean_wer"],
                                                    mean_draw(m, c) - mean_draw(m, "clean"))
        if noisy:
            gap = np.mean([mean_draw(m, c) - mean_draw(reference, c) for c in noisy], axis=0)
            entry["noisy_mean_gap"] = _interval(points[m]["noisy_mean_wer"] - points[reference]["noisy_mean_wer"], gap)
            if "clean" in p:
                deg = np.mean([mean_draw(m, c) for c in noisy], axis=0) - mean_draw(m, "clean")
                entry["noisy_degradation"] = _interval(points[m]["noisy_degradation"], deg)
        entry["summary"] = points[m]
        out[m] = entry
    result = {"reference": reference, "resamples": resamples, "seed": seed, "interval": "percentile 2.5-97.5",
              "bootstrap": "paired cluster bootstrap stratified by dev set; clusters resampled jointly for every "
                           "condition and model",
              "interval_scope": "speech sampling only: every difference is conditional on the frozen noise "
                                "realization (fixed DEMAND environments, hearings, FSD50K clips, BUT rooms); "
                                "see by_group", "build": ref_identity, "smoke": smoke, "models": out}
    if groups is not None:
        result["by_group"] = group_breakdown(models, groups, reference)
    return result


# --------------------------------------------------------------------------------------------- scoring

def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


SUMMARY_FIELDS = ("complete", "conditions", "sample", "skipped_conditions", "pool_missing_sources", "utterances",
                  "lock_sha256", "manifest_sha256")


def build_identity(build: Path) -> dict:
    """The build's identity and provenance, derived from the validated lock and the manifest's
    actual coverage (review round 5), with build.json required to agree on every field.

    - build.lock.json must have the SHA-256 build.json records, and build_manifest.jsonl too.
    - Conditions, sample size, skipped conditions and missing pool sources come from the lock;
      utterances from the lock's per-set input hashes.
    - The manifest must hold, for every locked condition and dev set, exactly the locked utterances.
    - "full" only if there is no sample, nothing skipped or missing, every base condition is
      present, the dev sets are the five of paths.DEV_SETS, and each locked dev manifest still has
      its recorded SHA-256 and as many rows as the lock has utterances (the full set).
    A missing or contradictory build.json field raises ValueError."""
    import noisy_build
    build = Path(build)
    info = json.loads((build / "build.json").read_text())
    absent = [k for k in SUMMARY_FIELDS if k not in info]
    if absent:
        raise ValueError(f"{build}: build.json is missing {absent}")
    if info["complete"] is not True:
        raise ValueError(f"{build}: the build is not complete")
    lock = json.loads((build / "build.lock.json").read_text())
    if noisy_paths.sha256_bytes(noisy_paths.canonical_json(lock).encode()) != info["lock_sha256"]:
        raise ValueError(f"{build}: build.lock.json differs from the lock the build recorded")
    if noisy_paths.sha256_file(build / "build_manifest.jsonl") != info["manifest_sha256"]:
        raise ValueError(f"{build}: build_manifest.jsonl differs from the manifest SHA-256 the build recorded")
    conds = [c["id"] for c in lock["conditions"]]
    sample = lock.get("sample")
    skipped = list(lock.get("skipped_conditions") or [])
    missing = sorted((lock.get("pool") or {}).get("missing") or {})
    dev_sets = list(lock["dev_sets"])
    ids = {s: sorted(v["pcm_sha256"]) for s, v in lock["dev_sets"].items()}
    utterances = sum(len(v) for v in ids.values())
    derived = {"conditions": conds, "sample": sample, "skipped_conditions": skipped,
               "pool_missing_sources": missing, "utterances": utterances}
    stated = {"conditions": info["conditions"], "sample": info["sample"],
              "skipped_conditions": list(info["skipped_conditions"] or []),
              "pool_missing_sources": sorted(info["pool_missing_sources"] or []), "utterances": info["utterances"]}
    for key, value in derived.items():
        if stated[key] != value:
            raise ValueError(f"{build}: build.json disagrees with the lock on {key}: {stated[key]!r} != {value!r}")
    coverage: dict = {}
    for r in _jsonl(build / "build_manifest.jsonl"):
        coverage.setdefault((r["condition"], r["set"]), []).append(r["id"])
    if sorted(coverage) != sorted((c, s) for c in conds for s in dev_sets):
        raise ValueError(f"{build}: the manifest does not cover exactly the locked conditions and dev sets")
    for (c, s), got in coverage.items():
        if sorted(got) != ids[s]:
            raise ValueError(f"{build}: manifest rows of {c}/{s} are not the locked utterances")
    full_dev = dev_sets == list(noisy_paths.DEV_SETS)
    for s, v in lock["dev_sets"].items():
        manifest = Path(v["manifest"])
        if not manifest.is_file() or noisy_paths.sha256_file(manifest) != v["manifest_sha256"] \
                or len(_jsonl(manifest)) != len(ids[s]):
            full_dev = False
    base = {c.id for c in noisy_build.CONDITIONS}
    full = sample is None and not skipped and not missing and base <= set(conds) and full_dev
    return {"out": str(build), "lock_sha256": info["lock_sha256"], "manifest_sha256": info["manifest_sha256"],
            "conditions": conds, "rotor": info.get("rotor"), "complete": True, "sample": sample,
            "skipped_conditions": skipped, "pool_missing_sources": missing, "utterances": utterances,
            "dev_sets": dev_sets, "full": full}


def condition_rows(build: Path, limit: int | None = None,
                   verify_audio: bool = True) -> dict[str, dict[str, list[dict]]]:
    """Each condition's NeMo manifest rows per set, checked against the build identity and the
    build manifest: IDs, references, durations, expected audio paths and (verify_audio) audio SHA-256."""
    build = Path(build)
    identity = build_identity(build)
    expected: dict = {}
    for r in _jsonl(build / "build_manifest.jsonl"):
        path = r["audio"] if r["condition"] == "clean" else str((build / r["audio"]).resolve())
        expected.setdefault(r["condition"], {}).setdefault(r["set"], []).append(
            (r["id"], r["text"], path, r["duration"], r["sha256"]))
    if list(expected) != identity["conditions"]:
        raise ValueError(f"{build}: build manifest conditions differ from build.json")
    out = {}
    for cond, sets in expected.items():
        out[cond] = {}
        for s, want in sets.items():
            rows = _jsonl(build / "manifests" / cond / f"{s}.jsonl")
            if [(r["id"], r["text"]) for r in rows] != [(w[0], w[1]) for w in want]:
                raise ValueError(f"{cond}/{s}: manifest rows (IDs, references) differ from the build manifest")
            for r, (uid, _, path, duration, _digest) in zip(rows, want, strict=True):
                if r["audio_filepath"] != path:
                    raise ValueError(f"{cond}/{s} {uid}: audio_filepath {r['audio_filepath']} is not the build's {path}")
                if r["duration"] != duration:
                    raise ValueError(f"{cond}/{s} {uid}: duration differs from the build manifest")
            rows = rows[:limit] if limit is not None else rows
            if verify_audio:
                for r, w in zip(rows, want, strict=False):
                    if noisy_paths.sha256_file(Path(r["audio_filepath"])) != w[4]:
                        raise ValueError(f"{cond}/{s} {r['id']}: audio SHA-256 differs from the build manifest")
            out[cond][s] = rows
    return out


def output_name(spec: ModelSpec, limit: int | None, identity: dict | None = None) -> str:
    """The production name only for a full build without a limit; otherwise a marked smoke name."""
    full = identity is None or identity.get("full", False)
    if full and limit is None:
        return spec.name
    parts = ["smoke", spec.name]
    if not full:
        parts.append(f"sample{identity['sample']}" if identity.get("sample") is not None else "partial")
    if limit is not None:
        parts.append(f"limit{limit}")
    return "-".join(parts)


def score_model(spec: ModelSpec, build: Path = noisy_paths.BUILD, scores: Path = noisy_paths.SCORES,
                hooks: Hooks | None = None, device: str = "cuda", limit: int | None = None,
                batch_size: int = EVAL_BATCH_SIZE, overwrite: bool = False) -> dict:
    """Decode every condition of the build with the model, under the GPU lock; returns summary.json."""
    hooks = hooks or default_hooks()
    identity = build_identity(build)
    rows = condition_rows(build, limit)
    out = noisy_paths.named_dir(scores, output_name(spec, limit, identity))
    if (out / "summary.json").exists() and not overwrite:
        raise FileExistsError(f"{out / 'summary.json'} exists; pass --overwrite to rescore")
    started = utc()
    results: dict[str, dict[str, list[dict]]] = {}
    modes = set()
    with hooks.gpu_lock(f"noisy_score {out.name}"):
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.glob("*.json"):  # a rescore replaces every file; summary.json is written last
            stale.unlink()
        determinism = hooks.deterministic()
        model = hooks.load_model(spec, device)
        for cond, sets in rows.items():
            results[cond] = {}
            for s, manifest_rows in sets.items():
                role = f"{spec.name} {cond}/{s}"
                hooks.pin_decoder(model, role)
                records = hooks.decode(model, manifest_rows, batch_size, device)
                mode = hooks.require_decoder(model, role)
                if mode != DECODER_GRAPHS_MODE:
                    raise DecoderModeError(f"{role}: decoded in {mode!r}, not {DECODER_GRAPHS_MODE!r}")
                modes.add(mode)
                if [r["id"] for r in records] != [r["id"] for r in manifest_rows]:
                    raise RuntimeError(f"{role}: decoder returned other utterances")
                results[cond][s] = records
            entry = summarize({cond: results[cond]})["conditions"][cond]
            noisy_paths.write_json(out / f"{cond}.json", {"condition": cond, "model": spec.name, "summary": entry,
                                                          "records": results[cond]})
            log(f"{spec.name} {cond}: mean WER {100 * entry['mean_wer']:.2f}% "
                f"({', '.join(f'{n} {100 * e['wer']:.2f}' for n, e in entry['sets'].items())})")
    summary = summarize(results)
    summary.update(model=describe_model(spec), decoder_graphs_mode=modes.pop() if len(modes) == 1 else sorted(modes),
                   batch_size=batch_size, limit=limit, device=device, determinism=determinism, build=identity,
                   smoke=not identity["full"] or limit is not None, out_name=out.name,
                   started_utc=started, finished_utc=utc())
    noisy_paths.write_json(out / "summary.json", summary)  # last write
    return summary


def load_scores(name: str, scores: Path = noisy_paths.SCORES) -> dict[str, dict[str, list[dict]]]:
    out = Path(scores) / name
    summary = json.loads((out / "summary.json").read_text())
    return {cond: json.loads((out / f"{cond}.json").read_text())["records"] for cond in summary["conditions"]}


def build_groups(build: Path) -> dict:
    return {(r["condition"], r["set"], r["id"]): r.get("groups") or {}
            for r in _jsonl(Path(build) / "build_manifest.jsonl")}


def load_build_identity(name: str, scores: Path = noisy_paths.SCORES) -> dict:
    """The build identity a score was made on, with its scoring limit (compare requires both equal)."""
    summary = json.loads((Path(scores) / name / "summary.json").read_text())
    return {**summary["build"], "limit": summary.get("limit")}


def build_clusters(build: Path) -> dict:
    return {(r["set"], r["id"]): r["cluster"] for r in _jsonl(Path(build) / "build_manifest.jsonl")
            if r["condition"] == "clean"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--model", choices=KINDS, required=True)
    s.add_argument("--path", type=Path, help="export directory (ternary, binary)")
    s.add_argument("--name", required=True)
    s.add_argument("--build", type=Path, default=noisy_paths.BUILD)
    s.add_argument("--limit", type=int, help="first N rows per set (smoke only)")
    s.add_argument("--batch-size", type=int, default=EVAL_BATCH_SIZE)
    s.add_argument("--device", default="cuda")
    s.add_argument("--overwrite", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--names", nargs="+", required=True)
    c.add_argument("--reference", default="B0")
    c.add_argument("--build", type=Path, default=noisy_paths.BUILD)
    c.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args(argv)
    noisy_paths.require_mount()
    if args.command == "score":
        try:
            spec = ModelSpec(kind=args.model, path=args.path, name=args.name)
        except ValueError as error:
            parser.error(str(error))
        summary = score_model(spec, args.build, noisy_paths.SCORES, device=args.device, limit=args.limit,
                              batch_size=args.batch_size, overwrite=args.overwrite)
        log(f"{spec.name}: noisy mean {100 * summary['noisy_mean_wer']:.2f}%, clean "
            f"{100 * summary['conditions']['clean']['mean_wer']:.2f}%")
        return
    models = {name: load_scores(name) for name in args.names}
    builds = {name: load_build_identity(name) for name in args.names}
    identity = build_identity(args.build)
    for name, b in builds.items():
        if {k: b.get(k) for k in IDENTITY_KEYS if k != "limit"} != {k: identity.get(k) for k in IDENTITY_KEYS
                                                                       if k != "limit"}:
            raise SystemExit(f"{name} was scored on another build than {args.build}")
    result = compare(models, build_clusters(args.build), args.reference, args.resamples, builds=builds,
                     groups=build_groups(args.build))
    out = noisy_paths.SCORES / f"compare-{'-'.join(args.names)}.json"
    noisy_paths.write_json(out, result)
    for m, entry in result["models"].items():
        g = entry.get("noisy_mean_gap")
        if g:
            log(f"{m}: noisy-mean gap to {args.reference} {100 * g['point']:+.2f} "
                f"[{100 * g['low']:+.2f}, {100 * g['high']:+.2f}] points")
    log(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
