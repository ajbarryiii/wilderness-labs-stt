"""CLI for the new corpora of stage 2 Part B: run from finetune/relabel as `./python -m corpora COMMAND`.

    syn-dialog-check                 voice-provenance check; anything but a pass marks the corpus dropped
    med-osce-fetch                   download once, verify (size, MD5), extract audio and transcripts
    med-osce-segment [--threads 8] [--batch N] [--device cpu|cuda] [--lock-wait S]
                                     segment (forced alignment with the permissive training aligner, never
                                     the MMS model) at most N more recordings per call; the manifest is frozen
                                     by the call that finishes the last one. --device cuda computes the
                                     aligner's emissions on the GPU while holding the parakeet GPU lock
                                     (waiting at most S seconds for it); run it under a shared GPU lease
    med-osce-segment --smoke N       the first N recordings into $RELABEL_ROOT/smoke/corpora/med-osce (timing)
    aligner-validate [--threads 8] [--device cpu|cuda] [--lock-wait S]
                                     re-validate the training aligner's threshold on Part A's dev clips
    status
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys

import corpora
from corpora import ctc_align, med_osce, registry, syn_dialog, validate
from pool import common


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="corpora", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("syn-dialog-check")
    sub.add_parser("med-osce-fetch")
    s = sub.add_parser("med-osce-segment")
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--smoke", type=int, default=None, metavar="N")
    s.add_argument("--batch", type=int, default=None, metavar="N")
    v = sub.add_parser("aligner-validate")
    v.add_argument("--threads", type=int, default=8)
    for p in (s, v):
        p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        p.add_argument("--lock-wait", type=float, default=1800.0, metavar="S")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    common.require_mount()
    if args.command == "syn-dialog-check":
        out = corpora.corpus_dir("syn-dialog")
        if corpora.status("syn-dialog"):
            print(json.dumps(corpora.status("syn-dialog"), indent=1))
            return 0
        verdict = syn_dialog.run_check(out, registry.source("syn-dialog"))
        print(json.dumps({k: v for k, v in verdict.items() if k != "record_json_keys"}, indent=1))
        return 0
    if args.command == "med-osce-fetch":
        summary = med_osce.fetch(registry.source(med_osce.ALIAS), corpora.corpus_dir(med_osce.ALIAS))
        print(json.dumps({k: v for k, v in summary.items() if k != "recordings"}, indent=1))
        return 0
    if args.command in ("med-osce-segment", "aligner-validate"):
        if not 1 <= args.threads <= 16:
            parser.error("--threads must be 1..16 (shared machine)")
        with gpu_lock_if(args.device, args.lock_wait, f"relabel corpora {args.command}"):
            aligner = ctc_align.load(registry.source("train-aligner"), threads=args.threads, device=args.device)
            return run_aligned(args, aligner)
    for alias in corpora.ALIASES:
        print(alias, json.dumps(corpora.status(alias)))
    return 0


def gpu_lock_if(device: str, wait_s: float, label: str):
    """The parakeet GPU lock (score_teacher.gpu_lock) for a GPU run; nothing on the CPU."""
    if device == "cpu":
        return contextlib.nullcontext()
    import score_teacher
    return score_teacher.gpu_lock(label, wait_s)


def run_aligned(args: argparse.Namespace, aligner) -> int:
    if args.command == "aligner-validate":
        manifests = {p: common.devsets_dir() / "manifests" / f"{p}.jsonl" for p in validate.PARTS}
        res = validate.run(aligner, manifests, common.corpora_dir() / "aligner-validation.json")
        print(json.dumps(res["parts"], indent=1))
        return 0
    if args.command == "med-osce-segment":
        cdir = corpora.corpus_dir(med_osce.ALIAS)
        if args.smoke:
            smoke = common.root("RELABEL_ROOT") / "smoke" / "corpora" / med_osce.ALIAS
            common.atomic_write_bytes(smoke / "raw" / "fetch.done.json", (cdir / "raw" / "fetch.done.json").read_bytes())
            print(json.dumps(med_osce.segment_all(smoke, aligner, limit=args.smoke), indent=1))
            return 0
        print(json.dumps(med_osce.segment_all(cdir, aligner, batch=args.batch), indent=1))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
