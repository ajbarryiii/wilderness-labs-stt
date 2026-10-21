"""Training entry point of the binary main run MB1 (plans/main.md): binary_train.py with pinned streams.

    python binary_main.py --init m1 --quantizer binary <train.py arguments> --train main --seed 20261017

Same arguments, same training and finishing as binary_train.py (which runs unchanged, inside its
own seam()), plus three replacements of binary_train attributes, made before binary_train.seam()
installs them in train.py and restored on exit (plans/main.md "Implementation notes
(2026-10-17)", item 1). binary_train.py, binary_paths.py and the other pilot sources stay
byte-identical, because the finished pilot arms recorded their hashes:

1. binary_train.parse_args: refuses (exit 2) a --train/--seed pair that has no stream pin in
   binary_main_paths.STREAM_PINS, before any model is loaded;
2. binary_train.open_stream: refuses an unpinned pair, opens the Hub stream with the pinned
   worker count and refuses any configuration hash other than the pinned one (binary_train
   checks the hash at the project seed only). stream.experiment_config's record of the stream
   (stream.record_sources, normally parakeet-ternary DATA/stream/sources.json and
   yodas_train_files.json, which carry the seed) is written under
   binary_main_paths.record_dir(seed) instead, and the two ternary files are hashed before and
   after: if they changed, the run stops before training;
3. binary_train.BINARY_SOURCES: plus binary_main.py and binary_main_paths.py, so config.json,
   every resume and summary.json record them.

And one train.py attribute (review round 1, finding 1):

4. train.load_resume_payload: after train.py's own checks (unreadable checkpoint exit 77,
   different arguments exit 78) and before anything is restored, a resume is refused with exit
   EXIT_SOURCES_CHANGED (79) unless every current training source (train.TRAINING_SOURCES under
   the seam: parakeet-ternary, whisper-ternary and the binary sources, binary_main.py and
   binary_main_paths.py included) hashes to what config.json recorded at the run's first start;
   a checkpoint without that record is refused too. sweep.run_train retries only exit 75, so a
   source edited during a stream outage stops the sweep instead of resuming under other code.
"""
from __future__ import annotations

import binary_train  # noqa: I001  first: it imports train.py, which configures CUDA/numba

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

import binary_main_paths
import binary_paths
import paths
import train

_BINARY_TRAIN = {name: getattr(binary_train, name) for name in ("parse_args", "open_stream", "BINARY_SOURCES")}
_LOAD_RESUME_PAYLOAD = train.load_resume_payload
EXIT_SOURCES_CHANGED = 79  # not train.py's 75-78; sweep.run_train retries 75 only


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """binary_train.parse_args, refusing (exit 2) a --train/--seed pair without a stream pin."""
    args = _BINARY_TRAIN["parse_args"](argv)
    try:
        binary_main_paths.stream_pin(args.train, args.seed)
    except ValueError as error:
        argparse.ArgumentParser(prog="binary_main.py").error(str(error))
    return args


def _record_hashes() -> dict[str, str | None]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
            for p in map(Path, binary_main_paths.TERNARY_STREAM_RECORDS)}


@contextlib.contextmanager
def redirected_record(module, seed: int):
    """stream.record_sources (module = the stream module) writing under record_dir(seed): the
    original function, with paths.DATA pointed there for the duration of the call only."""
    original = module.record_sources
    target = binary_main_paths.record_dir(seed)

    def record_sources(cfg):
        binary_paths.require_mount()
        saved = paths.DATA
        paths.DATA = target
        try:
            return original(cfg)
        finally:
            paths.DATA = saved

    module.record_sources = record_sources
    try:
        yield target
    finally:
        module.record_sources = original


def open_stream(args):
    """The training stream at a pinned (train, seed): the Hub stream with the pinned worker count
    and configuration hash (RuntimeError otherwise), its record redirected; local streams as
    train.open_stream."""
    try:
        workers, expected = binary_main_paths.stream_pin(args.train, args.seed)
    except ValueError as error:
        raise RuntimeError(str(error)) from None
    if args.stream != "hub":
        return train.open_stream(args)
    module = train.local_module("stream")
    before = _record_hashes()
    with redirected_record(module, args.seed):
        opened = module.make_training_stream(args.train, args.seed, num_workers=workers)
    try:
        after = _record_hashes()
        if after != before:
            raise RuntimeError(f"the parakeet-ternary stream record changed while the stream opened "
                               f"({before} -> {after}); restore it from the backup before training")
        if opened.config_hash != expected:
            raise RuntimeError(f"stream configuration hash {opened.config_hash} != {expected} pinned for "
                               f"--train {args.train} --seed {args.seed}: the data order would differ")
    except BaseException:
        close = getattr(opened, "close", None)
        if close is not None:
            close()
        raise
    return opened


def source_changes(run: Path) -> list[str] | None:
    """Training sources whose current hash differs from config.json's record of the run's first
    start (train.training_source_hashes, as the seam extends it); None if there is no record."""
    try:
        recorded = json.loads((run / "config.json").read_text()).get("training_source_hashes")
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(recorded, dict):
        return None
    now = train.training_source_hashes()
    return sorted(name for name in set(recorded) | set(now) if recorded.get(name) != now.get(name))


def load_resume_payload(run: Path, args) -> dict | None:
    """train.load_resume_payload, then (a checkpoint will be resumed) exit EXIT_SOURCES_CHANGED
    unless the training sources are those recorded at the run's first start."""
    payload = _LOAD_RESUME_PAYLOAD(run, args)
    if payload is None:
        return None
    changed = source_changes(run)
    if changed is None:
        problem = f"{run / 'config.json'} has no record of the training sources at the first start"
    elif changed:
        problem = f"training sources changed since the run's first start: {', '.join(changed)}"
    else:
        return payload
    train.log(f"refusing to resume {run} from {payload.get('_path')}: {problem}. Restore the recorded "
              "sources, or start a new run deliberately (another --run-name, or --overwrite)")
    sys.exit(EXIT_SOURCES_CHANGED)


@contextlib.contextmanager
def seam(quantizer: str):
    """binary_train.seam(quantizer) with this module's parse_args, open_stream, sources and resume
    check."""
    replacements = [(binary_train, "parse_args", parse_args), (binary_train, "open_stream", open_stream),
                    (binary_train, "BINARY_SOURCES", binary_main_paths.MAIN_SOURCES),
                    (train, "load_resume_payload", load_resume_payload)]
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in replacements]
    try:
        for obj, name, value in replacements:
            setattr(obj, name, value)
        with binary_train.seam(quantizer):
            yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--quantizer", choices=binary_train.QUANTIZERS)
    known, _ = pre.parse_known_args(argv)
    binary_paths.require_mount()
    with seam(known.quantizer or "binary"):
        train.main(argv)


if __name__ == "__main__":
    main()
