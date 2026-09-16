"""Separate training fit, reused development and new utterances from known speakers."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common import digest, digit_string, save, scores
from pilot4_train import configured_model, evaluate
from recovery_core import DOMAINS, SOURCE_RUN, artifact, read


def audit(run, job, seconds):
    started = time.monotonic()
    run = artifact(run)
    out = run / "evaluation" / ("recovery-" + job)
    checkpoint = run / "training" / job / "best.pt"
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    torch.set_num_threads(4)
    model = configured_model(saved["config"])(
        saved["config"]["model"], saved["arm"]["precision"]
    )
    model.load_state_dict(saved["model"], strict=True)
    model.cuda().eval()
    step = saved["step"]
    del saved
    manifest = read(run / "manifest.json")["rows"]
    known = read(
        SOURCE_RUN / "analysis/broad-investigation/unseen-known-speakers.json"
    )["rows"]
    datasets = [
        ("development", [r for r in manifest if r["split"] == "development"]),
        ("unseen_known_speakers", known),
        ("full_training", [r for r in manifest if r["split"] == "train"]),
    ]
    results = {}
    for name, rows in datasets:
        predictions = []
        for offset in range(0, len(rows), 32):
            if time.monotonic() - started > seconds - 30:
                break
            batch = rows[offset : offset + 32]
            features = {}
            for row in batch:
                assert digest(row["features"]) == row["feature_sha256"]
                features[row["id"]] = torch.from_numpy(
                    np.load(row["features"], allow_pickle=False)
                )
            predictions += evaluate(model, batch, features, step)["predictions"]
        metrics = {}
        for domain in DOMAINS:
            rs = [r for r in predictions if r["domain"] == domain]
            metrics[domain] = scores([(r["reference"], r["prediction"]) for r in rs])
            if domain == "digits":
                metrics[domain].update(
                    exact=sum(
                        digit_string(r["reference"]) == digit_string(r["prediction"])
                        for r in rs
                    ),
                    total=len(rs),
                )
        results[name] = dict(
            complete=len(predictions) == len(rows),
            requested=len(rows),
            examples=len(predictions),
            metrics=metrics,
        )
        save(out / (name + ".json"), dict(results[name], predictions=predictions))
    selected = read(run / "training" / job / "best.json")
    reload_match = results["development"]["complete"] and all(
        results["development"]["metrics"][d]["cer"] == selected["metrics"][d]["cer"]
        and results["development"]["metrics"][d]["wer"] == selected["metrics"][d]["wer"]
        for d in DOMAINS
    )
    metric = selected["metrics"]
    threshold = (
        metric["general"]["cer"] <= 0.42424
        and metric["medical_symptoms"]["cer"] <= 0.46584
        and metric["general"]["wer"] < 0.8
        and metric["medical_symptoms"]["wer"] < 0.8
        and metric["digits"]["exact"] >= 33
    )
    save(
        out / "generalization-audit.json",
        dict(
            checkpoint=str(checkpoint),
            sha256=digest(checkpoint),
            results=results,
            reload_matches_selection=reload_match,
            reused_development_threshold_met=threshold,
            independent_seed_confirmation="See per-seed jobs; one checkpoint cannot establish this",
            mission_specific_confirmation=False,
            elapsed_seconds=time.monotonic() - started,
        ),
    )
    if results["development"]["complete"] and not reload_match:
        raise RuntimeError("Reloaded selected checkpoint metrics differ")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("job")
    p.add_argument("--seconds", type=float, default=600)
    a = p.parse_args()
    audit(a.run, a.job, a.seconds)
