"""CLI of stage 2 Part B (labeling plan): run from finetune/relabel as `./python -m pool COMMAND`.

    fetch --source yodas|peoples_speech [--workers 6] [--threads 6]
        The text cache of every training file (network; resumable; run as a memory-capped job).
    probe --source yodas [--files 48] [--workers 6] [--threads 6]
        Times a seeded sample of files into $RELABEL_ROOT/smoke/pool-probe/ and projects the full pass.
    finalize [--workers 6] [--provisional] [--heldout NAME=PATH ...]
        The global denylist and then the subsets, from the text cache (local CPU). Held-out text: the
        parakeet-ternary dev and test manifests, every per-utterance manifest under $RELABEL_ROOT/devsets/
        (Part A's domain dev sets, eval-ext, ...), and any --heldout manifest. A final build needs all five
        domain dev parts and a decided status for both local corpora; --provisional builds without them and
        marks the outputs "final": false.
    status
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pool import build, common, fetch

SOURCES = ("yodas", "peoples_speech")
LOCAL_CORPORA = ("med-osce", "syn-dialog")


def heldout_manifests(extra: list[str]) -> tuple[list[tuple[str, Path]], list[str]]:
    """([(set name, manifest)], skipped JSONL files). Only `id`, `text` (and `part`, `identity`) are read."""
    import paths  # finetune/parakeet-ternary
    from pool import domain
    names = list(paths.DEV_SETS) + [f"test_{k}" for k in paths.TEST_SETS]
    out = [(n, common.ternary_manifests() / f"{n}.jsonl") for n in names]
    found, skipped = domain.manifests(common.devsets_dir())
    out += found
    for item in extra:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--heldout takes NAME=PATH, got {item!r}")
        if not domain.is_manifest(Path(path)):
            raise RuntimeError(f"--heldout {path}: not a per-utterance manifest (id, text, audio_filepath)")
        out.append((name, Path(path)))
    missing = [str(p) for _, p in out if not p.exists()]
    if missing:
        raise RuntimeError(f"missing held-out manifests: {missing}")
    return out, skipped


def corpora_inputs(final: bool) -> tuple[dict[str, Path], dict]:
    import corpora
    inputs, notes = {}, {}
    for alias in LOCAL_CORPORA:
        st = corpora.status(alias)
        notes[alias] = st.get("status", "missing")
        if st.get("status") == "built":
            inputs[alias] = Path(st["manifest"])
        elif st.get("status") != "dropped" and final:
            raise RuntimeError(f"final build: corpus {alias} is {notes[alias]!r} (build it or record it as dropped)")
    return inputs, notes


def cmd_finalize(args) -> None:
    import data
    final = not args.provisional
    heldout, skipped = heldout_manifests(args.heldout)
    corpora_paths, corpora_notes = corpora_inputs(final)
    caches = {s: common.pool_dir() / "text" / s for s in SOURCES}
    summary = build.finalize(common.pool_dir(), caches=caches, heldout=heldout,
                             reserved=sorted(data.reserved_recordings()), corpora=corpora_paths,
                             workers=args.workers, final=final,
                             notes={"corpora": corpora_notes, "devsets_jsonl_skipped": skipped})
    print(json.dumps({k: summary[k] for k in ("final", "subsets", "seconds")} |
                     {"denylist": summary["denylist"]["counts"]}, indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pool", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--source", choices=SOURCES, required=True)
    f.add_argument("--workers", type=int, default=6)
    f.add_argument("--threads", type=int, default=6)
    p = sub.add_parser("probe")
    p.add_argument("--source", choices=SOURCES, required=True)
    p.add_argument("--files", type=int, default=48)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--threads", type=int, default=6)
    z = sub.add_parser("finalize")
    z.add_argument("--workers", type=int, default=6)
    z.add_argument("--provisional", action="store_true")
    z.add_argument("--heldout", action="append", default=[], metavar="NAME=PATH",
                   help="an extra held-out manifest (repeatable)")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    common.require_mount()
    if args.command in ("fetch", "probe") and not 1 <= args.workers <= 12:
        parser.error("--workers must be 1..12 (shared machine)")
    if args.command == "fetch":
        status = fetch.run_fetch(args.source, common.training_files(args.source),
                                 common.pool_dir() / "text" / args.source, args.workers, args.threads)
        print(json.dumps(status, indent=1))
        return 1 if status["failed"] or status["done"] != status["files"] else 0
    if args.command == "probe":
        print(json.dumps(fetch.run_probe(args.source, common.training_files(args.source), args.files, args.workers,
                                         args.threads, common.probe_dir()), indent=1))
        return 0
    if args.command == "finalize":
        cmd_finalize(args)
        return 0
    for s in SOURCES:
        print(json.dumps(fetch.cache_status(s, common.training_files(s), common.pool_dir() / "text" / s)))
    summary = common.pool_dir() / "summary.json"
    if summary.exists():
        print(json.dumps(json.loads(summary.read_text())["subsets"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
