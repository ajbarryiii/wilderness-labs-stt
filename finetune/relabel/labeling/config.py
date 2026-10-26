"""The labeling stream's configuration: built from Part B's outputs, then pinned with W (labeling plan Part C).

    candidate = build_candidate(pool_dir, corpora_dir, ...)   everything but the worker count W
    pinned = pin(candidate, workers)                           adds W, the stream hash and the pinned hash

Pool (shares are of audio hours, labeling plan "Pool and stream"):
- existing mix 79%, plus the share of every local corpus Part B dropped, split over yodas, librispeech,
  peoples_speech, voxpopuli and ami in stream.py's SHARES proportions;
- yodas-med + ps-med 10%, split in proportion to their hours in 1-30 s rows (Part B's `hours_1_30s`, the
  audio the stream can deliver);
- yodas-tech 7%;
- local corpora: med-osce 1% and syn-dialog 3%, each only if Part B built it.
Routes: a subset source reads its include list's (file, row group) locators and keeps only its recordings;
`yodas` excludes yodas-med and yodas-tech recordings and `peoples_speech` excludes ps-med's (`in_subset`);
the global denylist ($RELABEL_ROOT/pool/denylist.json) is applied to every source that has recording keys
(`denylisted`). A denylist entry for a stream source without recording keys is an error.

Mean seconds (only to turn hour shares into per-item probabilities): stream.py's MEAN_SECONDS for the existing
five sources; for a subset, the mean duration of its 1-30 s rows in Part B's text cache; for a local corpus,
the mean manifest duration.

The pinned configuration records: the stream configuration (PoolStreamConfig as JSON), W, the stream's
configuration hash (stream.TrainingStream.config_hash = digest over the configuration and W), the batch
settings (600 s padded batches, train.py POOL_BATCHES, the seed), the number of batches, the SHA-256 of
every input file and of the code that determines the sequence, and `pinned_hash` over all of that. A
production build requires Part B's outputs to be final; `allow_provisional` is for smoke runs only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from labeling import layout
from labeling.pool_stream import PoolSourceSpec, PoolStreamConfig, config_from_dict

EXISTING = ("yodas", "librispeech", "peoples_speech", "voxpopuli", "ami")
SUBSETS = ("yodas-med", "ps-med", "yodas-tech")
CODE_FILES = ("finetune/parakeet-ternary/stream.py", "finetune/parakeet-ternary/data.py",
              "finetune/parakeet-ternary/train.py", "finetune/relabel/labeling/pool_stream.py",
              "finetune/relabel/labeling/batches.py")


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# --- shares -------------------------------------------------------------------------------------

def shares(subset_hours: Mapping[str, float], local_built: Mapping[str, bool]) -> dict[str, float]:
    """Per-source hour shares (see the module docstring). subset_hours: hours_1_30s of each subset."""
    import stream
    existing = layout.EXISTING_SHARE + sum(s for a, s in layout.LOCAL_SHARES.items() if not local_built.get(a))
    out = {name: existing * stream.SHARES[name] / sum(stream.SHARES.values()) for name in EXISTING}
    med = {s: float(subset_hours[s]) for s in layout.MEDICAL_SUBSETS}
    total = sum(med.values())
    if total <= 0:
        raise RuntimeError("the medical subsets have no hours")
    for name, hours in med.items():
        out[name] = layout.DOMAIN_SHARES["medical"] * hours / total
    out["yodas-tech"] = layout.DOMAIN_SHARES["yodas-tech"]
    for alias, share in layout.LOCAL_SHARES.items():
        if local_built.get(alias):
            out[alias] = share
    zero = [k for k, v in out.items() if v <= 0]
    if zero:
        raise RuntimeError(f"sources with no share: {zero}")
    if abs(sum(out.values()) - 1.0) > 1e-9:
        raise AssertionError(f"shares sum to {sum(out.values())}")
    return out


# --- inputs ---------------------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def hf_url(repo: str, revision: str, path: str) -> str:
    import stream
    return stream._hf_url(repo, revision, path)


def subset_units(rows: list[dict], repo: str, revision: str, url_fn=None) -> tuple[tuple[str, int], ...]:
    url_fn = url_fn or hf_url
    units = {(url_fn(repo, revision, loc["file"]), int(g)) for r in rows for loc in r["locators"] for g in loc["row_groups"]}
    return tuple(sorted(units))


def subset_mean_seconds(rows: list[dict], cache_dir: Path) -> float:
    """Mean duration of the subset recordings' 1-30 s rows, from Part B's text cache."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from pool import fetch
    by_file: dict[str, set[str]] = {}
    for r in rows:
        for loc in r["locators"]:
            by_file.setdefault(loc["file"], set()).add(r["recording"])
    total, n = 0.0, 0
    for path, recs in sorted(by_file.items()):
        cache, _ = fetch.cache_paths(cache_dir, {"path": path})
        t = pq.read_table(cache, columns=["recording", "duration"])
        t = t.filter(pc.is_in(t["recording"], value_set=pa.array(sorted(recs))))
        d = [x for x in t["duration"].to_pylist() if x is not None and 1.0 <= x <= 30.0]
        total += sum(d)
        n += len(d)
    if not n:
        raise RuntimeError("subset without 1-30 s rows in the text cache")
    return total / n


def local_rows(manifest: Path, corpus_dir: Path) -> tuple[tuple, float]:
    rows = _read_jsonl(manifest)
    out = []
    for r in sorted(rows, key=lambda r: r["id"]):
        audio = Path(r["audio_filepath"])
        if not audio.is_relative_to(corpus_dir):
            raise RuntimeError(f"{r['id']}: audio outside its corpus directory")
        out.append((r["id"], str(audio.relative_to(corpus_dir)), r.get("text") or "", r.get("recording")))
    if not out:
        raise RuntimeError(f"{manifest}: empty manifest")
    return tuple(out), sum(float(r["duration"]) for r in rows) / len(rows)


# --- candidate ------------------------------------------------------------------------------------

def build_candidate(pool_dir: Path, corpora_dir: Path, *, seed: int = layout.SEED, allow_provisional: bool = False,
                    source_files: Mapping[str, list[str]] | None = None, mean_seconds: Mapping[str, float] | None = None,
                    hf_sources: Mapping[str, dict] | None = None, reserved: set[str] | None = None,
                    url_fn=None) -> dict:
    """The candidate configuration (JSON) from Part B's outputs. The keyword arguments replace the
    parakeet-ternary inputs in tests (file lists, MEAN_SECONDS, HF_SOURCES, reserved recordings, and the
    locator-to-URL function, hf:// at the pinned revision by default)."""
    import data
    import stream
    pool_dir, corpora_dir = Path(pool_dir), Path(corpora_dir)
    summary_path, deny_path = pool_dir / "summary.json", pool_dir / "denylist.json"
    summary = json.loads(summary_path.read_text())
    deny = json.loads(deny_path.read_text())
    if not (summary.get("final") and deny.get("final")) and not allow_provisional:
        raise RuntimeError("Part B's pool is not final (pool/summary.json or denylist.json); "
                           "a production configuration needs `./python -m pool finalize`")
    files = dict(source_files) if source_files is not None else stream.source_files()
    hf = dict(hf_sources) if hf_sources is not None else {**data.HF_SOURCES, "librispeech": {
        "repo": stream.LIBRISPEECH["repo"], "revision": stream.LIBRISPEECH["revision"], "id_col": "id",
        "text_col": "text"}}
    means = dict(mean_seconds) if mean_seconds is not None else dict(stream.MEAN_SECONDS)
    reserved = set(reserved) if reserved is not None else set(data.reserved_recordings())

    denied: dict[str, set[str]] = {s: set(v) for s, v in deny["recordings"].items()}
    if not reserved <= denied.get("yodas", set()):
        raise RuntimeError("the denylist does not contain every reserved YODAS recording")

    subsets: dict[str, list[dict]] = {}
    inputs = {"pool_summary_sha256": sha256_file(summary_path), "denylist_sha256": sha256_file(deny_path),
              "subsets": {}, "corpora": {}}
    for name in SUBSETS:
        path = pool_dir / "subsets" / f"{name}.jsonl"
        subsets[name] = _read_jsonl(path)
        inputs["subsets"][name] = sha256_file(path)
    sub_hours = {n: summary["subsets"][n]["hours_1_30s"] for n in SUBSETS}

    local_built, local = {}, {}
    for alias in layout.LOCAL_SHARES:
        st_path = corpora_dir / alias / "status.json"
        st = json.loads(st_path.read_text()) if st_path.exists() else {}
        status = st.get("status")
        if status not in ("built", "dropped"):
            if not allow_provisional:
                raise RuntimeError(f"local corpus {alias} is {status or 'missing'!r}; it must be built or dropped")
            status = "missing"
        local_built[alias] = status == "built"
        inputs["corpora"][alias] = {"status": status}
        if status == "built":
            manifest = Path(st["manifest"])
            sha = sha256_file(manifest)
            if st.get("sha256") and st["sha256"] != sha:
                raise RuntimeError(f"{alias}: manifest SHA-256 differs from its status.json")
            local[alias] = (manifest, sha)
            inputs["corpora"][alias]["manifest_sha256"] = sha
    share = shares(sub_hours, local_built)

    sources: list[PoolSourceSpec] = []
    in_subset = {"yodas": {r["recording"] for n in ("yodas-med", "yodas-tech") for r in subsets[n]},
                 "peoples_speech": {r["recording"] for r in subsets["ps-med"]}}
    keyed = {"yodas": "group", "peoples_speech": "ps_id"}

    def excl(parent: str, extra: set[str] = frozenset()) -> tuple[tuple[str, str], ...]:
        reasons = {rec: "in_subset" for rec in extra}
        reasons.update({rec: "denylisted" for rec in denied.get(parent, ())})
        return tuple(sorted(reasons.items()))

    for name in EXISTING:
        spec = hf[name]
        is_yodas = name == "yodas"
        sources.append(PoolSourceSpec(
            name=name, kind="hf", units=tuple(sorted(files[name])), share=share[name], id_col=spec["id_col"],
            text_col=None if is_yodas else spec["text_col"], pseudo_col="text" if is_yodas else None,
            group_col="original_audio_id" if is_yodas else None, slots=2 if is_yodas else 1,
            recording_key=keyed.get(name), exclude=excl(name, in_subset.get(name, set())) if name in keyed else ()))
    for name in SUBSETS:
        parent = layout.SUBSET_PARENT[name]
        spec = hf[parent]
        is_yodas = parent == "yodas"
        rows = subsets[name]
        sources.append(PoolSourceSpec(
            name=name, kind="subset", units=subset_units(rows, spec["repo"], spec["revision"], url_fn), share=share[name],
            id_col=spec["id_col"], text_col=None if is_yodas else spec["text_col"],
            pseudo_col="text" if is_yodas else None, group_col="original_audio_id" if is_yodas else None,
            slots=1, recording_key=keyed[parent], include=tuple(sorted(r["recording"] for r in rows)),
            exclude=excl(parent)))
        means[name] = subset_mean_seconds(rows, pool_dir / "text" / parent)
    for alias, (manifest, _) in local.items():
        rows, mean = local_rows(manifest, corpora_dir / alias)
        sources.append(PoolSourceSpec(
            name=alias, kind="local", units=tuple(range(len(rows))), share=share[alias], id_col="id",
            text_col="text", slots=1, recording_key="local", exclude=excl(alias),
            local_dir=str((corpora_dir / alias).relative_to(layout.relabel_root())), local_rows=rows))
        means[alias] = mean
    for s in deny["recordings"]:
        if s in {x.name for x in sources if x.recording_key is None}:
            raise RuntimeError(f"the denylist names recordings of {s!r}, which has no recording key in the stream")
    names = [s.name for s in sources]
    cfg = PoolStreamConfig(sources=tuple(sources), seed=seed, weighting="hours",
                           mean_seconds=tuple((n, float(means[n])) for n in names))
    root = repo_root()
    code = {f: sha256_file(root / f) for f in CODE_FILES if (root / f).exists()}
    import train  # finetune/parakeet-ternary: the batch settings of the main runs
    return {"version": 1, "kind": "labeling-stream-candidate", "final": bool(summary.get("final") and deny.get("final")),
            "stream": json.loads(json.dumps(dataclasses.asdict(cfg))), "shares": share,
            "batch": {"batch_seconds": layout.BATCH_SECONDS, "pool_batches": train.POOL_BATCHES, "seed": seed,
                      "min_seconds": train.MIN_SECONDS, "max_seconds": train.MAX_SECONDS},
            "batches": layout.TOTAL_BATCHES, "inputs": inputs, "code_sha256": code,
            "denylist_sources_not_streamed": sorted(set(deny["recordings"]) - set(names))}


def pin(candidate: dict, workers: int) -> dict:
    """The pinned configuration: the candidate plus W, the stream's configuration hash and pinned_hash."""
    if int(workers) < 1:
        raise ValueError("W must be positive")
    cfg = config_from_dict(candidate["stream"])
    few = {s.name: len(s.units) for s in cfg.sources if len(s.units) < workers}
    if few:
        raise ValueError(f"sources with fewer read units than W={workers} (a worker would have none): {few}")
    stream_hash = cfg.digest(max(1, workers))
    body = {k: v for k, v in candidate.items() if k != "kind"} | {
        "kind": "labeling-stream-pinned", "workers": int(workers), "stream_config_hash": stream_hash}
    body["pinned_hash"] = sha256_json({k: v for k, v in body.items() if k != "pinned_hash"})
    return body


def verify(pinned: dict) -> PoolStreamConfig:
    """The stream configuration of a pinned file, after checking both hashes."""
    cfg = config_from_dict(pinned["stream"])
    if cfg.digest(max(1, pinned["workers"])) != pinned["stream_config_hash"]:
        raise RuntimeError("pinned configuration: stream hash mismatch")
    if sha256_json({k: v for k, v in pinned.items() if k != "pinned_hash"}) != pinned["pinned_hash"]:
        raise RuntimeError("pinned configuration: pinned_hash mismatch")
    return cfg


def config_id(pinned: dict) -> str:
    return f"w{pinned['workers']}-{pinned['pinned_hash'][:12]}"


def make_stream(pinned: dict, num_workers: int | None = None):
    """The pinned stream (the labeling pass's, and every training run's on this pool). num_workers must equal
    the pinned W (a different W is a different sequence); 0 (in-process) is accepted only for a W=1 pin."""
    from labeling.pool_stream import PoolTrainingStream
    cfg = verify(pinned)
    w = pinned["workers"] if num_workers is None else num_workers
    if max(1, w) != pinned["workers"]:
        raise ValueError(f"the pinned stream has W={pinned['workers']}, not {w}")
    s = PoolTrainingStream(cfg, num_workers=w)
    if s.config_hash != pinned["stream_config_hash"]:
        raise RuntimeError("stream hash differs from the pinned one")
    return s


def write_json(path: Path, obj) -> None:
    from pool import common
    common.atomic_write_json(path, obj)


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
