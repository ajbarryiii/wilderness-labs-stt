"""Compare experimental recognition outputs with the unchanged packed control."""
import argparse
import contextlib
import json
import time
from pathlib import Path

import torch

import paths
from .benchmark import DEFAULT_EXPORT, energy_meter, metadata, records_for_sets, save
from .experiments import VARIANTS, experiment
from .runtime import load_packed


@contextlib.contextmanager
def variant(model, name):
    if name == "optimized":
        from .optimized import enable_optimizations, disable_optimizations
        enable_optimizations(model)
        try:
            yield
        finally:
            disable_optimizations(model)
    else:
        with experiment(model, **VARIANTS[name]):
            yield


def main():
    import evaluate as ev
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    p.add_argument("--variants", nargs="+", default=["baseline", "all_fp16x2", "all_bf16x2", "fp16_single", "bf16_single", "int8", "int8x2"])
    p.add_argument("--utterances-per-set", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if args.variants[0] != "baseline":
        p.error("first variant must be baseline")
    paths.require_mount()
    out = paths.ARTIFACTS / "kernel-runs" / f"{time.strftime('%Y%m%dT%H%M%S')}-quality2"
    out.mkdir(); print("RESULTS", out, flush=True)
    records = records_for_sets(args.utterances_per_set)
    rows = [dict(r, hypotheses={}) for r in records]
    summary = {}
    with paths.gpu_lock("recognition quality comparison"), torch.inference_mode(), energy_meter() as meter:
        meter._guard(); save(out, "metadata.json", metadata(args))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = load_packed(args.export)
        for name in args.variants:
            with variant(model, name):
                hypotheses = ev.transcribe_records(model, records, args.batch_size, "cuda", workers=0)
            totals = {}
            exact = 0
            for r, text in zip(rows, hypotheses):
                r["hypotheses"][name] = text
                exact += text == r["hypotheses"]["baseline"]
                score = ev.score(r["text"], text)
                if score["scored"]:
                    s = totals.setdefault(r["source"], dict(words=0, edits=0))
                    s["words"] += len(score["ref_norm"].split())
                    s["edits"] += score["S"]+score["D"]+score["I"]
            summary[name] = dict(exact_matches=exact, utterances=len(rows), per_set=totals,
                                 audio_seconds=sum(r["duration"] for r in records),
                                 total_edits=sum(s["edits"] for s in totals.values()),
                                 total_words=sum(s["words"] for s in totals.values()))
            save(out, "quality.json", dict(summary=summary, records=rows))
            print(name, json.dumps(summary[name]), flush=True)


if __name__ == "__main__":
    main()
