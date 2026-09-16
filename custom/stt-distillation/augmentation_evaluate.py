"""Reload a selected checkpoint and evaluate every frozen acoustic condition."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common import decode, digest, save
from pilot4_train import configured_model, evaluate
from recovery_core import DOMAINS, artifact, read


def evaluate_job(run, job, seconds=600):
    run = artifact(run)
    started = time.monotonic()
    torch.set_num_threads(4)
    checkpoint = run / "training" / job / "best.pt"
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model = configured_model(saved["config"])(saved["config"]["model"], saved["arm"]["precision"])
    model.load_state_dict(saved["model"], strict=True)
    model.cuda().eval()
    step = saved["step"]
    del saved
    sources = read(run / "augmentation-sources.json")
    fixture = read(sources["validation_manifest"])
    assert digest(sources["validation_manifest"]) == sources["validation_manifest_sha256"]
    clean = [r for r in read(run / "manifest.json")["rows"] if r["split"] == "development"]
    known_path = Path(sources["known_speakers_manifest"])
    assert digest(known_path) == sources["known_speakers_manifest_sha256"]
    conditions = {"clean": clean, **fixture["conditions"], "known_speakers": read(known_path)["rows"]}
    output = dict(job=job, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                  status="running", conditions={})
    destination = run / "evaluation" / job / "summary.json"
    for name, rows in conditions.items():
        if time.monotonic() - started > seconds - 30:
            output["status"] = "incomplete"
            save(destination, output)
            raise TimeoutError(f"Evaluation budget exhausted before {name}")
        features = {}
        for row in rows:
            assert digest(row["features"]) == row["feature_sha256"]
            features[row["id"]] = torch.from_numpy(np.load(row["features"], allow_pickle=False))
        result = evaluate(model, rows, features, step)
        save(destination.parent / (name + ".json"), result)
        output["conditions"][name] = result["metrics"]
        save(destination, output)
    selected = read(run / "training" / job / "best.json")
    output["reload_matches_selection"] = all(
        output["conditions"]["clean"][d][m] == selected["metrics"][d][m]
        for d in DOMAINS for m in ("cer", "wer"))
    assert output["reload_matches_selection"], "Checkpoint reload changed clean metrics"
    output["noise_only"] = []
    with torch.inference_mode():
        for row in fixture["noise_only"]:
            assert digest(row["features"]) == row["feature_sha256"]
            x = torch.from_numpy(np.load(row["features"], allow_pickle=False))[None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = decode(model(x)[0].argmax(-1).tolist())
            output["noise_only"].append(dict(id=row["id"], prediction=pred, false_emission=bool(pred)))
    output.update(status="completed", elapsed_seconds=time.monotonic() - started)
    save(destination, output)
    print(__import__("json").dumps(dict(job=job, status="completed", elapsed_seconds=output["elapsed_seconds"])), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("job")
    parser.add_argument("--seconds", type=float, default=600)
    args = parser.parse_args()
    evaluate_job(args.run, args.job, args.seconds)
