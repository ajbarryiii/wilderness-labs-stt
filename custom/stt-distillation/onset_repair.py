"""Isolated, fresh-initialization onset repair trials; no classifier training."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from common import ALPHABET, ART, decode, digest, norm, save, storage
from onset_model import OnsetModel, startup_mask
from pilot4_train import evaluate, worker


BASE = ART / "runs/pilot-4h-20260910T150425Z"
SOURCES = [
    "onset_repair.py",
    "onset_model.py",
    "onset_checks.py",
    "pilot4_train.py",
    "pilot4_model.py",
    "model.py",
    "common.py",
    "medical_terms.json",
    "prepare.py",
]


def prepare(name, hold_frames, seed):
    storage()
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Simple run name required")
    run = ART / "training-repair" / name
    run.mkdir(exist_ok=False)
    cfg = json.loads((BASE / "config.json").read_text())
    cfg["seed"] = seed
    cfg["model"].update(startup_confirmation_frames=3, startup_hold_frames=hold_frames)
    cfg["model_class"] = "onset_model.OnsetModel"
    cfg["scope"] = (
        "Fresh FP training-subset learning repair; signal onset, not speech/no-speech classification. No generalization claim."
    )
    save(run / "config.json", cfg)
    for name in ["manifest.json", "subsets.json"]:
        shutil.copy2(BASE / name, run / name)
    source = run / "source"
    source.mkdir()
    for name in SOURCES:
        shutil.copy2(Path(__file__).with_name(name), source / name)
    save(
        run / "provenance.json",
        {
            "reference_run": str(BASE),
            "initialization": "fresh; no checkpoint resume",
            "sources": {name: digest(source / name) for name in SOURCES},
            "manifest_sha256": digest(run / "manifest.json"),
            "subsets_sha256": digest(run / "subsets.json"),
        },
    )
    print(run, flush=True)


def audit(run):
    storage()
    torch.set_num_threads(4)
    saved = torch.load(
        run / "gate/fp_control/latest.pt", map_location="cpu", weights_only=False
    )
    model = OnsetModel(saved["config"]["model"], "fp")
    model.load_state_dict(saved["model"], strict=True)
    step = saved["step"]
    del saved
    model.cuda().eval()
    manifest = json.loads((run / "manifest.json").read_text())["rows"]
    gate = set(json.loads((run / "subsets.json").read_text())["gate"])
    rows = [r for r in manifest if r["id"] in gate]
    features = {r["id"]: torch.from_numpy(np.load(r["features"])) for r in rows}
    result = evaluate(model, rows, features, step)
    final = json.loads((run / "gate/fp_control/result.json").read_text())["final"][
        "training"
    ]
    assert result["metrics"] == final["metrics"], "Fresh reload metrics changed"
    events = []
    with torch.inference_mode():
        for row in rows:
            x = features[row["id"]][None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x)[0]
            path = z.argmax(-1)
            nonblank = torch.nonzero(path).flatten()
            first = int(nonblank[0]) if len(nonblank) else None
            opened = startup_mask(
                x,
                -1.95,
                model.cfg["startup_confirmation_frames"],
                model.cfg["startup_hold_frames"],
            )[0, ::2]
            where = torch.nonzero(opened).flatten()
            entry = {
                "id": row["id"],
                "reference": norm(row["text"]),
                "prediction": decode(path.tolist()),
                "emission_allowed_nominal_ms": int(where[0]) * 20
                if len(where)
                else None,
                "first_emission_nominal_ms": first * 20 if first is not None else None,
            }
            if row["domain"] == "digits":
                z32 = model(x)[0]
                entry["fp32_prediction"] = decode(z32.argmax(-1).tolist())
                if first is not None:
                    top = z[first].float().softmax(-1).topk(4)
                    entry["first_top4"] = [
                        (ALPHABET[int(i)], float(p))
                        for i, p in zip(top.indices, top.values)
                    ]
            events.append(entry)
    save(
        run / "audit.json",
        {
            "reload_matches": True,
            "training_metrics": result["metrics"],
            "events": events,
        },
    )
    print(
        json.dumps(
            {
                "reload_matches": True,
                "metrics": result["metrics"],
                "digit_events": [r for r in events if r["id"].startswith("digit-")],
            }
        ),
        flush=True,
    )
    digits = [r for r in rows if r["domain"] == "digits"]
    stress(run, model, digits)
    if "augmentation" in json.loads((run / "config.json").read_text()):
        stress(
            run,
            model,
            digits,
            prefixes=[35, 175, 415, 995],
            gains=[-3, -9, -21],
            filename="onset-interpolation-stress.json",
        )


def stress(
    run, model, rows, *, prefixes=None, gains=None, filename="onset-stress.json"
):
    # Recompute features from audio. These are transformed training clips, not
    # independent held-out examples. Remove only the known generated 200 ms zeros.
    import soundfile as sf
    from prepare import feature

    results = []
    with torch.inference_mode():
        for prefix_ms in [0, 80, 200, 600] if prefixes is None else prefixes:
            for gain_db in [0, -12, -24] if gains is None else gains:
                predictions = []
                for row in rows:
                    audio, sr = sf.read(row["audio"], dtype="float32")
                    assert sr == 16000 and not np.any(audio[:3200])
                    audio = np.concatenate(
                        [
                            np.zeros(prefix_ms * 16, np.float32),
                            audio[3200:] * (10 ** (gain_db / 20)),
                        ]
                    )
                    x = torch.from_numpy(feature(audio))[None].cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        ids = model(x)[0].argmax(-1)
                    pred = decode(ids.tolist())
                    hits = torch.nonzero(ids).flatten()
                    predictions.append(
                        {
                            "id": row["id"],
                            "reference": norm(row["text"]),
                            "prediction": pred,
                            "first_digit_correct": bool(pred)
                            and pred[0] == norm(row["text"])[0],
                            "exact": pred == norm(row["text"]),
                            "first_emission_from_recording_start_ms": int(hits[0]) * 20
                            - prefix_ms
                            if len(hits)
                            else None,
                        }
                    )
                results.append(
                    {
                        "prefix_ms": prefix_ms,
                        "gain_db": gain_db,
                        "exact": sum(p["exact"] for p in predictions),
                        "first_digit_correct": sum(
                            p["first_digit_correct"] for p in predictions
                        ),
                        "total": len(predictions),
                        "predictions": predictions,
                    }
                )
    save(
        run / filename,
        {
            "scope": "Transformations of the eight memorization-gate digit clips; not a generalization benchmark.",
            "conditions": results,
        },
    )
    print(
        "STRESS",
        json.dumps(
            [{k: v for k, v in r.items() if k != "predictions"} for r in results]
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "train", "audit"])
    parser.add_argument("name")
    parser.add_argument("--hold-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.name, args.hold_frames, args.seed)
    else:
        run = ART / "training-repair" / args.name
        if args.mode == "train":
            worker(run, "fp_control", "gate", 8000, 800, model_class=OnsetModel)
        else:
            audit(run)
