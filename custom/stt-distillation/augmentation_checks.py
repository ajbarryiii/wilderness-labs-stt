"""Fail-closed preflight for acoustic mixing, resume, full-size training and evaluation."""

import argparse
import copy
import datetime
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from augmentation_control import hashes, prepare
from augmentation_data import ROOT, SOURCE_RUN
from augmentation_evaluate import evaluate_job
from augmentation_features import AugmentedFeatures, Mixer, active_rms
from common import digest, save
from recovery_core import read
from recovery_train import Features, batch_loss, run_worker
from onset_model import OnsetModel


def contracts(existing=None):
    torch.set_num_threads(4)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path(existing) if existing else prepare(ROOT / "setup" / ("contracts-" + stamp), smoke=True)
    rows = read(run / "manifest.json")["rows"]
    train = [r for r in rows if r["split"] == "train"]
    examples = [next(r for r in train if r["domain"] == d) for d in ("general", "medical_symptoms", "digits")]
    noise = read(ROOT / "data/demand/manifest.json")["rows"]
    assert {r["id"] for r in noise if r["split"] == "train"}.isdisjoint(r["id"] for r in noise if r["split"] == "validation")
    for split in ("calibration", "development"):
        assert {r["speaker"] for r in train if r["domain"] == "general"}.isdisjoint(
            r["speaker"] for r in rows if r["split"] == split and r["domain"] == "general")
    fixture = read(ROOT / "data/validation.json")
    for condition, records in fixture["conditions"].items():
        assert len(records) == 160
        for row in records:
            assert row["split"] == "development"
            assert row["augmentation"]["source_split"] == ("validation" if condition.startswith("noise") else "calibration")
            assert abs(row["augmentation"]["db"] - row["augmentation"]["measured_db"]) < 1e-6
            assert digest(row["features"]) == row["feature_sha256"]
    base = Features(examples)
    for job in ("A0", "A1", "A2", "A3"):
        spec = read(run / "jobs" / (job + ".json"))
        spec["acoustic_augmentation"].update(probability=1, ramp_examples=1)
        aug = AugmentedFeatures(examples, run, spec)
        for j, row in enumerate(examples):
            for prefix, gain in ((0, 0), (60, -24), (100, -6)):
                first = aug.augmented(row, prefix, gain, presentation=100 + j)
                second = aug.augmented(row, prefix, gain, presentation=100 + j)
                assert torch.equal(first, second), "Augmentation replay changed"
                assert torch.isfinite(first).all()
                if job == "A0":
                    assert torch.equal(first, base.augmented(row, prefix, gain))
                else:
                    assert not torch.equal(first, base.augmented(row, prefix, gain))
                    detail = aug.last_details[row["id"]]
                    if job in ("A2", "A3"):
                        assert detail["source_split"] == "train"
                        assert abs(detail["measured_db"] - detail["db"]) < 1e-6
                    if job == "A1" and row["domain"] == "digits":
                        assert detail["time_width"] <= 3
        try:
            aug.augmented(next(r for r in rows if r["split"] == "development"), 0, 0, presentation=100)
        except AssertionError:
            pass
        else:
            raise AssertionError("Development audio was accepted for training augmentation")
        del aug
    waveform = np.sin(np.arange(16000, dtype=np.float32) * 0.03)
    assert abs(active_rms(waveform) - active_rms(np.pad(waveform, (16000, 16000)))) < 1e-7
    mixer = Mixer(rows, noise)
    for kind in ("noise", "speech"):
        mixture, detail = mixer.mix(waveform * 5, examples[0], np.random.default_rng(19), kind, 20)
        assert np.max(np.abs(mixture)) <= 0.980001 and detail["common_peak_scale"] < 1
    # Real augmented CTC, optimizer and random-state resume equivalence for every intervention.
    resume_results = {}
    for job in ("A1", "A2", "A3"):
        spec = read(run / "jobs" / (job + ".json"))
        spec["acoustic_augmentation"].update(probability=1, ramp_examples=1)
        save(run / "jobs" / (job + ".json"), spec)
        alternate = job + "-resume"
        save(run / "jobs" / (alternate + ".json"), dict(spec, name=alternate))
        if not existing:
            run_worker(run, job, 32, 600, feature_factory=AugmentedFeatures)
            run_worker(run, alternate, 16, 600, feature_factory=AugmentedFeatures)
            run_worker(run, alternate, 32, 600, resume=True, feature_factory=AugmentedFeatures)
        else:
            for name in (job, alternate):
                proof = read(run / "training" / name / "result.json")
                assert proof["status"] == "completed" and proof["presented"] == 32
        a = torch.load(run / "training" / job / "latest.pt", map_location="cpu", weights_only=False)
        b = torch.load(run / "training" / alternate / "latest.pt", map_location="cpu", weights_only=False)
        assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items()), job
        assert all(torch.equal(v, b["optimizer"]["state"][k][n]) for k, state in a["optimizer"]["state"].items() for n, v in state.items()), job
        assert a["presented"] == b["presented"] == 32
        resume_results[job] = "bit-exact model and optimizer"
        del a, b
        torch.cuda.empty_cache()
    # Nonfinite CTC must fail loudly rather than be hidden by zero_infinity.
    small = OnsetModel(read(run / "config.json")["model"], "fp").cuda()
    with torch.no_grad():
        small.head.weight.fill_(float("nan"))
    try:
        batch_loss(small, [torch.ones(80, 100)], [{"text": "ab"}], "fp32")
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Nonfinite CTC was silently accepted")
    del small
    torch.cuda.empty_cache()
    # Selected checkpoint reload and all clean/noisy/overlap/stress paths use the real evaluator.
    evaluated = evaluate_job(run, "A1", 600)
    assert evaluated["status"] == "completed" and evaluated["reload_matches_selection"]
    try:
        evaluate_job(run, "missing-checkpoint", 600)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing evaluation checkpoint accepted")
    result = dict(passed=True, run=str(run), resume=resume_results, source_hashes=hashes(),
        contracts=["baseline feature parity", "train/validation source and speaker separation", "measured level ratios",
                   "peak limiting", "digit mask limits", "presentation-keyed replay", "CTC rejects nonfinite",
                   "evaluation reload equality and missing-checkpoint failure"])
    save(ROOT / "setup/contracts.json", result)
    print(json.dumps(result), flush=True)


def fullsize(run):
    run = Path(run)
    torch.set_num_threads(4)
    results = {}
    for job in ("A0", "A1", "A2", "A3"):
        # Keep each subprocess independent so allocator/cache history does not bias memory estimates.
        command = [str(run / "code/augmentation-python"), str(run / "code/augmentation_train.py"),
                   str(run), job, "--target", "512", "--seconds", "900"]
        subprocess.run(command, check=True, timeout=930)
        r = read(run / "training" / job / "result.json")
        assert r["status"] == "completed" and r["presented"] == 512
        events = [json.loads(line) for path in (run / "training" / job).glob("metrics-*.jsonl") for line in path.read_text().splitlines()]
        assert len(events) == 128 and all(np.isfinite(e["loss"]) and np.isfinite(e["preclip_grad_norm"]) for e in events)
        seconds = [e["update_seconds"] for e in events if e["step"] > 8]
        overhead = r["elapsed_seconds"] - sum(e["update_seconds"] for e in events)
        projection = statistics.mean(seconds) * 27000 + overhead * 5
        results[job] = dict(update_seconds_mean=statistics.mean(seconds),
            projected_108k_seconds=projection, non_update_seconds=overhead,
            peak_reserved_gib=max(e["peak_reserved_gib"] for e in events))
        assert results[job]["peak_reserved_gib"] < 28
    evaluated = evaluate_job(run, "A3", 600)
    assert evaluated["reload_matches_selection"]
    budget = read(run / "config.json")["job_seconds"] - 180
    assert all(r["projected_108k_seconds"] < budget for r in results.values()), (
        "Training throughput does not fit the declared per-arm budget", results)
    save(ROOT / "setup/fullsize.json", dict(passed=True, run=str(run), jobs=results, source_hashes=hashes(),
        evaluation_seconds=evaluated["elapsed_seconds"]))
    print(json.dumps(dict(passed=True, fullsize=results)), flush=True)


def prepare_fullsize():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = prepare(ROOT / "setup" / ("fullsize-" + stamp))
    # Exercise the steady-state 50% rate immediately, rather than benchmarking mostly clean warmup.
    for job in ("A1", "A2", "A3"):
        path = run / "jobs" / (job + ".json")
        spec = read(path)
        spec["acoustic_augmentation"]["ramp_examples"] = 1
        save(path, spec)
    save(ROOT / "setup/fullsize-run.json", dict(run=str(run)))
    print(run, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["contracts", "prepare-fullsize", "fullsize"])
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.action == "contracts":
        contracts(args.run)
    elif args.action == "prepare-fullsize":
        prepare_fullsize()
    else:
        fullsize(args.run)
