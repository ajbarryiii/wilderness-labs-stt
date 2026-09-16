"""Meaningful preflight checks for exposure, augmentation, accumulation and resume."""

import copy
import datetime
import json
from pathlib import Path

import torch

from common import save
from pilot4_augmentation import PilotAugmentation
from pilot4_train import schedule
from recovery_core import (
    ROOT,
    SOURCE_RUN,
    exposure,
    lr_factor,
    read,
    source_hashes,
    tape,
)
from recovery_control import prepare
from recovery_train import Features, batch_loss, run_worker
from onset_model import OnsetModel


def main():
    torch.set_num_threads(4)
    rows = [
        r for r in read(SOURCE_RUN / "manifest.json")["rows"] if r["split"] == "train"
    ]
    cfg = read(SOURCE_RUN / "config.json")
    draws = tape(rows, cfg["seed"], 108000)
    prior = read(SOURCE_RUN / "train/ternary_matched/result.json")
    for key in ["sample_hash", "augmentation_hash"]:
        assert exposure(rows, draws, [0, -6, -12, -18, -24])[key] == prior[key], key
    for step in [1, 200, 3000, 6000, 27000]:
        actual = lr_factor(step * 4) * 1e-5
        expected = schedule(cfg["arms"][0], cfg, step, 27000)[0]
        assert abs(actual - expected) < 1e-18
    legacy = PilotAugmentation(SOURCE_RUN)
    selected = [rows[int(d[0])] for d in draws[:32]]
    fs = Features(list({r["id"]: r for r in selected}.values()))
    old = legacy(selected)
    new = [
        fs.augmented(r, int(d[1]), [0, -6, -12, -18, -24][int(d[2])])
        for r, d in zip(selected, draws[:32])
    ]
    assert all(torch.equal(a, b) for a, b in zip(old, new)), (
        "Augmentation differs from prior worker"
    )
    del legacy, fs
    # A real CTC loss with variable target lengths checks the accumulation math.
    small = dict(cfg["model"], width=32, depth=2, heads=4, context=16, checkpoint=False)
    torch.manual_seed(77)
    a = OnsetModel(small, "fp").cuda()
    torch.nn.init.xavier_uniform_(a.head.weight)
    b = copy.deepcopy(a)
    xs = [torch.randn(80, 80) for _ in range(4)]
    rs = [{"text": s} for s in ["ab", "aab", "bc", "abcd"]]
    raw, lengths = batch_loss(a, xs, rs, "fp32")
    (raw / lengths).mean().backward()
    for i in range(0, 4, 2):
        raw, lengths = batch_loss(b, xs[i : i + 2], rs[i : i + 2], "fp32")
        ((raw / lengths).sum() / 4).backward()
    max_gradient_difference = max(
        float((p.grad - q.grad).abs().max())
        for p, q in zip(a.parameters(), b.parameters())
    )
    assert max_gradient_difference < 2e-5, max_gradient_difference
    del a, b
    torch.cuda.empty_cache()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = prepare(ROOT / "setup" / ("smoke-" + stamp), smoke=True)
    # Full worker stop/resume: same data, RNG, optimizer, LR, and final parameters.
    spec = read(run / "jobs/R0.json")
    save(run / "jobs/resume.json", dict(spec, name="resume"))
    run_worker(run, "R0", 32, 300)
    run_worker(run, "resume", 16, 300)
    run_worker(run, "resume", 32, 300, resume=True)
    full = torch.load(
        run / "training/R0/latest.pt", map_location="cpu", weights_only=False
    )
    resumed = torch.load(
        run / "training/resume/latest.pt", map_location="cpu", weights_only=False
    )
    assert full["presented"] == resumed["presented"] == 32
    differences = {
        k: float((v - resumed["model"][k]).abs().max())
        for k, v in full["model"].items()
    }
    assert max(differences.values()) == 0, differences
    for p, values in full["optimizer"]["state"].items():
        for k, v in values.items():
            assert torch.equal(v, resumed["optimizer"]["state"][p][k])
    for k in ["sample_hash", "augmentation_hash"]:
        assert (
            read(run / "training/R0/checkpoint.json")[k]
            == read(run / "training/resume/checkpoint.json")[k]
        )
    save(
        ROOT / "setup/recovery-checks.json",
        dict(
            passed=True,
            smoke_run=str(run),
            contracts=[
                "historical_full_exposure_hashes",
                "historical_augmentation_feature_parity",
                "historical_learning_rates",
                "CTC_gradient_accumulation",
                "bit_exact_worker_resume_model_and_optimizer",
            ],
            max_accumulation_gradient_difference=max_gradient_difference,
            source_hashes={
                k: v
                for k, v in source_hashes(Path(__file__).parent).items()
                if k
                in {
                    "recovery_core.py",
                    "recovery_train.py",
                    "recovery_control.py",
                    "recovery_checks.py",
                    "recovery8.json",
                    "pilot4_train.py",
                    "onset_model.py",
                    "pilot4_model.py",
                    "model.py",
                    "common.py",
                    "prepare.py",
                    "python",
                }
            },
        ),
    )
    print(json.dumps(dict(passed=True, smoke_run=str(run))), flush=True)


if __name__ == "__main__":
    main()
