"""CPU-only controls audit and meaningful end-to-end tiny-model/resume checks."""

import collections
import copy
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import digest, encode, save
from onset_model import OnsetModel
from recovery_core import DOMAINS, artifact, read
from scaling_core import (ROOT, bootstrap_delta, checkpoint_reasons, common_tape,
                          exposure, groups_for_seed, lr_factor, selected_id, tier_ids)
from scaling_train import loss_and_logits, worker

HERE = Path(__file__).resolve().parent


def require_error(fn, kind=AssertionError):
    try:
        fn()
    except kind:
        return
    raise AssertionError("Expected validation to reject this input")


def tiny_fixture(path):
    path.mkdir()
    (path / "jobs").mkdir()
    cfg = read(HERE / "scaling.json")
    cfg.update(seeds=[101], duration_edges_seconds=[0, 1, 2], checkpoint_updates=[1, 2, 4, 8],
               checkpoint_training_seconds=[], recovery_every_updates=2, warmup_updates=2,
               peak_lr=.001, bootstrap_replicates=20)
    cfg["model"] = dict(cfg["model"], width=32, depth=2, heads=4, context=16, dropout=.1, checkpoint=False)
    rows = []
    rng = np.random.default_rng(711)
    for d in DOMAINS:
        for split, count in (("train", 16 if d == "general" else 2), ("development", 2)):
            for i in range(count):
                name = f"{split}-{d}-{i}"
                frames = 80 + i % 3 + (20 if d == "digits" else 0)
                x = rng.uniform(-1.5, 1., (80, frames)).astype(np.float32)
                if d == "digits":
                    x[:, :20] = -2
                fp = path / (name + ".npy")
                np.save(fp, x, allow_pickle=False)
                rows.append(dict(id=name, domain=d, split=split, frames=frames, seconds=frames / 100,
                    speaker=None if d == "medical_symptoms" else ("digit-heldout" if d == "digits" and split != "train" else split + str(i // 8 if split == "train" else i)),
                    text="one" if d == "digits" else "ab", features=str(fp), feature_sha256=digest(fp)))
    train = [r for r in rows if r["split"] == "train"]
    groups, _ = groups_for_seed(train, {train[0]["id"]}, 101, cfg)
    save(path / "groups.json", dict(groups=groups))
    draws = common_tape(groups, 101, cfg)
    np.save(path / "tape.npy", draws, allow_pickle=False)
    common = [r["id"] for d in DOMAINS for r in [v for v in train if v["domain"] == d][:1]]
    save(path / "subsets.json", dict(common_train_monitor=common, gate=common,
        development=[r["id"] for r in rows if r["split"] == "development"],
        known_speaker=[train[0]["id"]], long_development=[r["id"] for r in rows if r["split"] == "development" and r["domain"] == "general"]))
    save(path / "manifest.json", dict(rows=rows))
    save(path / "config.json", cfg)
    summary = {}
    for tier, size in cfg["sizes"].items():
        save(path / "jobs" / (f"101-{tier}.json"), dict(name=f"101-{tier}", tier=tier, size=size, seed=101,
            groups="groups.json", groups_sha256=digest(path / "groups.json"), tape="tape.npy", tape_sha256=digest(path / "tape.npy")))
        chosen = set(tier_ids(groups, size))
        summary[tier] = {d: dict(recordings=sum(r["id"] in chosen and r["domain"] == d for r in rows),
            hours=sum(r["seconds"] for r in rows if r["id"] in chosen and r["domain"] == d) / 3600) for d in DOMAINS}
    save(path / "data-ready.json", dict(summary={"101": summary}, synthetic_fixture=True))
    return cfg, rows, groups, draws


def test_contracts(root):
    cfg, rows, groups, draws = tiny_fixture(root / "fixture")
    assert np.array_equal(draws, common_tape(groups, 101, cfg))
    previous = set()
    for size in cfg["sizes"].values():
        ids = set(tier_ids(groups, size))
        assert previous <= ids
        previous = ids
    for start in range(0, len(draws) - 9, 10):
        assert collections.Counter(groups[int(d[0])]["domain"] for d in draws[start:start + 10]) == cfg["domain_counts_per_block"]
    for draw in draws:
        ids = [selected_id(groups, draw, size) for size in cfg["sizes"].values()]
        rr = [{r["id"]: r for r in rows}[i] for i in ids]
        assert len({(r["speaker"], r["domain"]) for r in rr}) == 1
        if draw[1] == 0:
            assert len(set(ids)) == 1
        for r in rr:
            assert r["frames"] - (20 if r["domain"] == "digits" else 0) + draw[2] <= draw[4]
    exposures = [exposure(rows, groups, draws, k, cfg["batch_size"]) for k in cfg["sizes"].values()]
    assert len({e["padded_input_frames"] for e in exposures}) == 1
    assert exposures[-1]["unique_recordings"]["general"] > exposures[0]["unique_recordings"]["general"]
    assert exposures[0]["domain_presentations"] == exposures[-1]["domain_presentations"]
    assert checkpoint_reasons(0, 0, cfg, []) == ["updates:0"]
    timed = dict(cfg, checkpoint_training_seconds=[1, 2, 3])
    assert checkpoint_reasons(3, 2.1, timed, []) == ["training_seconds:1", "training_seconds:2"]
    assert checkpoint_reasons(3, 2.9, timed, [1, 2]) == []
    assert lr_factor(max(cfg["checkpoint_updates"]), cfg) == cfg["lr_floor_fraction"]
    # Nonzero head ensures padding invariance actually tests the causal encoder.
    torch.manual_seed(502)
    model = OnsetModel(dict(cfg["model"], dropout=0), "fp").eval()
    torch.nn.init.normal_(model.head.weight, std=.1)
    x = torch.randn(80, 81)
    row = dict(text="abb")
    raw1, _, logits1, _ = loss_and_logits(model, [x], [row], 81, torch.device("cpu"))
    raw2, _, logits2, _ = loss_and_logits(model, [x], [row], 130, torch.device("cpu"))
    torch.testing.assert_close(logits1, logits2[:, :logits1.shape[1]], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(raw1, raw2, rtol=1e-5, atol=1e-5)
    require_error(lambda: loss_and_logits(model, [x], [dict(text="a" * 100)], 81, torch.device("cpu")))
    return cfg


def test_resume(root):
    original = root / "fixture"
    resumed = root / "resume"
    shutil.copytree(original, resumed)
    job = "101-small"
    worker(original, job, device_name="cpu")
    worker(resumed, job, device_name="cpu", stop_after=4)
    worker(resumed, job, device_name="cpu", resume=True)
    a = torch.load(original / "training" / job / "latest.pt", weights_only=False)
    b = torch.load(resumed / "training" / job / "latest.pt", weights_only=False)
    assert a["update"] == b["update"] == 8
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name]), name
    for idx, state in a["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, b["optimizer"]["state"][idx][name]), (idx, name)
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    # Reloaded saved weights must reproduce saved evaluation outputs/loss.
    from scaling_train import Features, evaluate
    cfg = read(original / "config.json")
    rows = read(original / "manifest.json")["rows"]
    model = OnsetModel(cfg["model"], "fp")
    saved = torch.load(original / "training" / job / "checkpoints/step-0000008.pt", weights_only=False)
    model.load_state_dict(saved["model"])
    dev = [r for r in rows if r["split"] == "development"]
    ev = evaluate(model, dev, Features(dev, 1), torch.device("cpu"))
    recorded = read(original / "training" / job / "evaluations/step-0000008.json")
    assert ev == recorded["evaluations"]["development"]
    assert len(read(resumed / "training" / job / "index.json")["records"]) == 5
    changed = read(resumed / "config.json")
    changed["peak_lr"] *= 2
    save(resumed / "config.json", changed)
    require_error(lambda: worker(resumed, job, device_name="cpu", resume=True))


def test_analysis(root):
    from scaling_analysis import analyze, fit_backtest
    summary = analyze(root / "fixture")
    assert len(summary["points"]) == 5 and summary["plots"]["status"] == "written"
    assert (root / "fixture/analysis/points.csv").exists()
    # Use actual development fixture with two general speakers and one digit speaker.
    preds = read(root / "fixture/training/101-small/evaluations/step-0000008.json")["evaluations"]["development"]["predictions"]
    for d in DOMAINS:
        interval = bootstrap_delta(preds, preds, d, 20)
        assert interval["delta"] == 0
        if d == "digits":
            assert interval["interval"] is None
        elif interval["interval"] is not None:
            assert interval["interval"] == [0., 0.]
    x = np.array([1, 2, 4, 8, 16, 32, 64.])
    result = fit_backtest(x, .2 + .8 * np.exp(-x / 10))
    assert result["models"]["exponential_floor"]["withheld_mae"] < .001
    assert fit_backtest([1, 2], [1, .5])["status"] == "insufficient_points"


def audit_real(run):
    cfg = read(run / "config.json")
    manifest = read(run / "manifest.json")["rows"]
    by_id = {r["id"]: r for r in manifest}
    assert len(by_id) == len(manifest)
    held = [r for r in manifest if r["split"] != "train"]
    train = [r for r in manifest if r["split"] == "train"]
    assert not ({r["pcm_sha256"] for r in held} & {r["pcm_sha256"] for r in train})
    assert not ({r["text"] for r in held} & {r["text"] for r in train if r["domain"] != "digits"})
    dev_speakers = {r["speaker"] for r in held if r["domain"] == "general" and r["split"] in ("development", "calibration", "long_development")}
    assert not dev_speakers & {r["speaker"] for r in train if r["domain"] == "general"}
    subsets = read(run / "subsets.json")
    summaries = {}
    for seed in cfg["seeds"]:
        previous, counters = set(), []
        for tier, size in cfg["sizes"].items():
            spec = read(run / "jobs" / f"{seed}-{tier}.json")
            groups = read(run / spec["groups"])["groups"]
            tape = np.load(run / spec["tape"], allow_pickle=False)
            assert np.array_equal(tape, common_tape(groups, seed, cfg))
            ids = tier_ids(groups, size)
            assert len(ids) == len(set(ids)) and previous <= set(ids)
            assert set(subsets["common_train_monitor"] + subsets["gate"]) <= set(ids)
            assert set(ids) <= {r["id"] for r in train}
            previous = set(ids)
            for group in groups:
                rr = [by_id[i] for i in group["members"]]
                assert len({(r["domain"], r["speaker"]) for r in rr}) == 1
                assert all(r["frames"] - (20 if r["domain"] == "digits" else 0) <= group["bound"] for r in rr)
            e = exposure(manifest, groups, tape, size, cfg["batch_size"])
            counters.append(e)
            summaries[f"{seed}-{tier}"] = e
        assert len({c["padded_input_frames"] for c in counters}) == 1
        assert all(c["domain_presentations"] == counters[0]["domain_presentations"] for c in counters)
        assert all(c["domain_presentations"] == {"general": 54000, "medical_symptoms": 32400, "digits": 21600} for c in counters)
    total_bytes = 0
    # All actual tensors/labels, including fixed medical/digit pools, are checked
    # on CPU before allowing even a full-size GPU preflight.
    for r in manifest:
        assert digest(artifact(r["features"])) == r["feature_sha256"]
        f = np.load(r["features"], allow_pickle=False)
        assert f.shape == (80, r["frames"]) and np.isfinite(f).all() and f.max() < 2 and f.min() >= -2
        y = encode(r["text"])
        assert y and (r["frames"] + 1) // 2 >= len(y) + sum(a == b for a, b in zip(y, y[1:]))
        total_bytes += f.nbytes
    assert total_bytes < cfg["feature_cache_gib"] * 2**30
    return dict(rows=len(manifest), feature_gib=total_bytes / 2**30, matched_exposure=summaries)


def main(run=None):
    torch.set_num_threads(1)
    artifact(ROOT / "setup").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cpu-check-", dir=ROOT / "setup") as temporary:
        root = Path(temporary)
        test_contracts(root)
        print("PASS nesting, paired sampling/padding, exposure, LR and milestone boundaries", flush=True)
        test_resume(root)
        print("PASS CPU training, dropout RNG / Adam resume, checkpoint reload and changed-input rejection", flush=True)
        test_analysis(root)
        print("PASS analysis artifacts, cluster bootstrap and withheld-checkpoint curve backtest", flush=True)
    result = dict(passed=True, gpu_work_performed=False)
    if run is not None:
        from scaling_control import fingerprint
        run = artifact(run)
        result.update(audit=audit_real(run), fingerprint=fingerprint(run))
        save(run / "cpu-preflight.json", result)
    else:
        save(ROOT / "setup/cpu-development-checks.json", result)
    print(json.dumps(dict(passed=True, gpu_work_performed=False)), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
