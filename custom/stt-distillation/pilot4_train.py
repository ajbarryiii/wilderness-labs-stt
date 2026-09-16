"""Four-hour pilot worker: full-size learning gates, timing, training and evaluation."""

import argparse
import collections
import hashlib
import json
import math
import signal
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import (
    decode,
    digest,
    digit_string,
    encode,
    medical_terms,
    norm,
    save,
    scores,
    storage,
)
from pilot4_model import PilotModel

STOP = False


def halt(*_):
    global STOP
    STOP = True


def schedule(arm, cfg, step, steps, gate=False):
    peak, wd = arm["peak_lr"], arm["weight_decay"]
    stage = 1
    local_step, local_steps = step, steps
    if arm["schedule"] == "two_stage" and step > steps // 2:
        peak, wd = arm["second_peak_lr"], arm["second_weight_decay"]
        stage = 2
        local_step, local_steps = step - steps // 2, steps - steps // 2
    elif arm["schedule"] == "two_stage":
        local_steps = steps // 2
    # Gate tests allow the complete two-stage recipe, but avoid an artificial
    # near-zero LR before basic CTC alignments emerge on the tiny subset.
    decay = (
        1.0
        if gate
        else 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * local_step / local_steps))
    )
    warm = min(1.0, step / cfg["warmup_steps"])
    return peak * warm * decay, wd, stage


def evaluate(model, rows, features, step):
    prior = model.training
    model.eval()
    predictions = []
    with torch.inference_mode():
        for row in rows:
            x = features[row["id"]][None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x)
            ids = z[0].argmax(-1)
            predictions.append(
                {
                    "id": row["id"],
                    "domain": row["domain"],
                    "reference": norm(row["text"]),
                    "prediction": decode(ids.tolist()),
                    "blank_fraction": float((ids == 0).float().mean()),
                }
            )
    result = {
        "step": step,
        "overall": scores([(r["reference"], r["prediction"]) for r in predictions]),
    }
    for domain in ["general", "medical_symptoms", "digits"]:
        rs = [r for r in predictions if r["domain"] == domain]
        result[domain] = scores([(r["reference"], r["prediction"]) for r in rs])
        if domain == "digits":
            result[domain].update(
                exact=sum(
                    digit_string(r["reference"]) == digit_string(r["prediction"])
                    for r in rs
                ),
                total=len(rs),
            )
        if domain == "medical_symptoms":
            result[domain]["terminology"] = medical_terms(
                [(r["reference"], r["prediction"]) for r in rs]
            )
    counts = collections.Counter(r["prediction"] for r in predictions)
    result.update(
        distinct_predictions=len(counts),
        modal_fraction=max(counts.values()) / len(rows),
        exact_utterances=sum(r["reference"] == r["prediction"] for r in predictions),
        blank_fraction=float(np.mean([r["blank_fraction"] for r in predictions])),
    )
    model.train(prior)
    return {"metrics": result, "predictions": predictions}


def passed(metric, cfg):
    g = cfg["gate"]
    return (
        metric["overall"]["wer"] <= g["max_wer"]
        and metric["overall"]["cer"] <= g["max_cer"]
        and metric["overall"]["empty_outputs"] == 0
        and metric["digits"]["total"] > 0
        and metric["digits"]["exact"] == metric["digits"]["total"]
    )


def is_collapsed(metric, cfg):
    return (
        metric["modal_fraction"] >= cfg["collapse"]["min_modal_fraction"]
        and metric["overall"]["cer"] >= cfg["collapse"]["min_cer"]
    )


def quality(metric):
    return metric["overall"]["cer"] + 0.5 * (
        1 - metric["digits"]["exact"] / metric["digits"]["total"]
    )


def atomic_torch_save(path, obj):
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(obj, pending)
    pending.replace(path)


def configured_model(cfg):
    name = cfg.get("model_class", "pilot4_model.PilotModel")
    if name == "onset_model.OnsetModel":
        from onset_model import OnsetModel

        return OnsetModel
    if name == "pilot4_model.PilotModel":
        return PilotModel
    raise ValueError(f"Unknown pilot model: {name}")


def worker(
    run,
    arm_name,
    mode,
    steps,
    seconds,
    *,
    model_class=None,
    batch_transform=None,
    extra_gate=None,
):
    storage()
    assert run.resolve().is_relative_to(
        Path("/mnt/hd/wilderness-labs-stt/stt-distillation")
    )
    cfg = json.loads((run / "config.json").read_text())
    arm = next(a for a in cfg["arms"] if a["name"] == arm_name)
    out = run / mode / arm_name
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run / "manifest.json").read_text())
    by_id = {r["id"]: r for r in manifest["rows"]}
    subset = json.loads((run / "subsets.json").read_text())
    training = [r for r in manifest["rows"] if r["split"] == "train"]
    gate_rows = [by_id[i] for i in subset["gate"]]
    monitor = [by_id[i] for i in subset["monitor"]]
    dev = [r for r in manifest["rows"] if r["split"] == "development"]
    assert len(dev) == 160 and len(monitor) == 48 and len(gate_rows) == 32
    assert all(r["split"] == "train" for r in gate_rows + monitor)
    if mode == "gate":
        training = gate_rows
    needed = {r["id"]: r for r in training + monitor + gate_rows + dev}
    features = {}
    for row in needed.values():
        assert digest(row["features"]) == row["feature_sha256"]
        x = torch.from_numpy(np.load(row["features"]))
        y = encode(row["text"])
        assert torch.isfinite(x).all() and x.ndim == 2 and x.shape[0] == 80 and y
        assert (x.shape[-1] + 1) // 2 >= len(y) + sum(a == b for a, b in zip(y, y[1:]))
        features[row["id"]] = x
    if model_class is None:
        model_class = configured_model(cfg)
    if batch_transform is None and cfg.get("augmentation"):
        from pilot4_augmentation import PilotAugmentation

        batch_transform = PilotAugmentation(run)
        if mode == "gate":
            extra_gate = batch_transform.gate
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(cfg["seed"])
    model = model_class(cfg["model"], arm["precision"]).cuda().train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=arm["peak_lr"],
        betas=tuple(cfg["betas"]),
        weight_decay=arm["weight_decay"],
        foreach=False,
    )
    initial = hashlib.sha256()
    for tensor in model.state_dict().values():
        initial.update(tensor.detach().cpu().numpy().tobytes())
    save(
        out / "initialization.json",
        {
            "sha256": initial.hexdigest(),
            "parameters": model.precision_counts(),
            "arm": arm,
        },
    )
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    started = time.monotonic()
    sample_hash = hashlib.sha256()
    groups = {
        d: [r for r in training if r["domain"] == d]
        for d in cfg["domain_probabilities"]
    }
    order = []
    history = []
    exposure = collections.Counter()
    best = math.inf
    gate_streak = collapse_streak = 0
    status = "step_limit"
    step = 0
    final = None
    checkpoint_step = 0

    def checkpoint():
        nonlocal checkpoint_step
        atomic_torch_save(
            out / "latest.pt",
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "config": cfg,
                "arm": arm,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "numpy_rng": rng.bit_generator.state,
                "pending_gate_order": [r["id"] for r in order],
                "augmentation_state": batch_transform.state_dict()
                if hasattr(batch_transform, "state_dict")
                else None,
                "sample_hash": sample_hash.hexdigest(),
                "exposure_seconds": dict(exposure),
                "best_selection_score": best if math.isfinite(best) else None,
            },
        )
        checkpoint_step = step
        save(
            out / "checkpoint.json",
            {"step": step, "optimizer_saved": True, "file": "latest.pt"},
        )

    def decode_check():
        nonlocal final, best
        rows = gate_rows if mode == "gate" else monitor
        tr = evaluate(model, rows, features, step)
        result = {
            "step": step,
            "elapsed_seconds": time.monotonic() - started,
            "training": tr,
        }
        if mode == "gate" and extra_gate is not None:
            result["onset_gate"] = extra_gate(model)
        if mode == "train":
            dv = evaluate(model, dev, features, step)
            result["development"] = dv
            q = quality(dv["metrics"])
            # Save the best selected validation checkpoint, not just the last one.
            if q < best:
                best = q
                atomic_torch_save(
                    out / "best.pt",
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "config": cfg,
                        "arm": arm,
                    },
                )
                save(
                    out / "best.json",
                    {
                        "step": step,
                        "selection_score": q,
                        "criterion": "development CER + 0.5 * digit-sequence error rate",
                        "metrics": dv["metrics"],
                    },
                )
        save(out / f"eval-{step:06d}.json", result)
        save(out / "progress.json", result)
        final = result
        print(
            json.dumps(
                {
                    "step": step,
                    "mode": mode,
                    "training": tr["metrics"],
                    "development": result.get("development", {}).get("metrics"),
                }
            ),
            flush=True,
        )
        return tr["metrics"]

    error = None
    try:
        if mode != "benchmark":
            decode_check()
        for step in range(1, steps + 1):
            if STOP or time.monotonic() - started > seconds - 90:
                step -= 1
                status = "cancelled" if STOP else "time_limit"
                break
            if mode == "gate":
                if not order:
                    order = [training[i] for i in rng.permutation(len(training))]
                batch, order = order[: cfg["batch_size"]], order[cfg["batch_size"] :]
            else:
                batch = []
                for _ in range(cfg["batch_size"]):
                    domain = rng.choice(
                        list(groups), p=list(cfg["domain_probabilities"].values())
                    )
                    batch.append(groups[domain][int(rng.integers(len(groups[domain])))])
            tick = time.monotonic()
            batch_features = (
                batch_transform(batch)
                if batch_transform is not None
                else [features[r["id"]] for r in batch]
            )
            length = max(x.shape[-1] for x in batch_features)
            x = torch.stack(
                [
                    F.pad(feature, (0, length - feature.shape[-1]))
                    for feature in batch_features
                ]
            ).cuda()
            target_lists = [encode(r["text"]) for r in batch]
            target = torch.tensor(
                [c for y in target_lists for c in y], device="cuda", dtype=torch.long
            )
            ilen = torch.tensor([(x.shape[-1] + 1) // 2 for x in batch_features])
            tlen = torch.tensor([len(y) for y in target_lists])
            lr, wd, stage = schedule(arm, cfg, step, steps, gate=mode == "gate")
            for group in opt.param_groups:
                group.update(lr=lr, weight_decay=wd)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = F.ctc_loss(
                logits.float().log_softmax(-1).transpose(0, 1),
                target,
                ilen,
                tlen,
                zero_infinity=False,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite CTC loss at update {step}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["grad_clip"], error_if_nonfinite=True
            )
            opt.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved() / 2**30
            if peak > cfg["max_vram_gib"]:
                raise MemoryError(f"VRAM limit exceeded: {peak:.2f} GiB")
            elapsed = time.monotonic() - tick
            history.append(elapsed)
            for row in batch:
                sample_hash.update((row["id"] + "\n").encode())
                exposure[row["domain"]] += row["seconds"]
            metric = {
                "step": step,
                "loss": float(loss.detach()),
                "grad_norm": float(grad),
                "lr": lr,
                "weight_decay": wd,
                "stage": stage,
                "update_seconds": elapsed,
                "elapsed_seconds": time.monotonic() - started,
                "peak_reserved_gib": peak,
            }
            with (out / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(metric) + "\n")
            if step % 100 == 0:
                save(out / "training-progress.json", metric)
            interval = 100 if mode == "gate" else cfg["eval_every"]
            if mode != "benchmark" and step % interval == 0:
                metric = decode_check()
                if mode == "gate":
                    gate_streak = (
                        gate_streak + 1
                        if passed(metric, cfg)
                        and final.get("onset_gate", {}).get("passed", True)
                        else 0
                    )
                    if gate_streak >= cfg["gate"]["consecutive_passes"]:
                        status = "passed_gate"
                        break
                elif step >= cfg["collapse"]["after_fraction"] * steps:
                    collapse_streak = (
                        collapse_streak + 1 if is_collapsed(metric, cfg) else 0
                    )
                    if collapse_streak >= cfg["collapse"]["consecutive_checks"]:
                        status = "collapsed"
                        break
            if mode == "train" and step % cfg["checkpoint_every"] == 0:
                checkpoint()
        if mode != "benchmark":
            if final is None or final["step"] != step:
                decode_check()
            checkpoint()
        if status == "step_limit":
            status = "gate_failed" if mode == "gate" else "completed"
    except Exception as exc:
        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"
        status = "failed"
        print(error, flush=True)
    result = {
        "status": status,
        "error": error,
        "mode": mode,
        "steps": step,
        "requested_steps": steps,
        "elapsed_seconds": time.monotonic() - started,
        "initial_hash": initial.hexdigest(),
        "sample_hash": sample_hash.hexdigest(),
        "augmentation_hash": batch_transform.trace.hexdigest()
        if batch_transform is not None
        else None,
        "augmentation_state": batch_transform.state_dict()
        if hasattr(batch_transform, "state_dict")
        else None,
        "exposure_hours": {d: v / 3600 for d, v in exposure.items()},
        "mean_update_seconds": float(np.mean(history[10:]))
        if len(history) > 10
        else None,
        "p90_update_seconds": float(np.quantile(history[10:], 0.9))
        if len(history) > 10
        else None,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "checkpoint_step": checkpoint_step,
        "parameters": model.precision_counts(),
        "final": final,
    }
    save(out / "result.json", result)
    if error:
        raise RuntimeError(error)


def final_evaluation(run, arm_name):
    storage()
    out = run / "train" / arm_name
    saved = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model = configured_model(saved["config"])(saved["config"]["model"], saved["arm"]["precision"])
    model.load_state_dict(saved["model"], strict=True)
    model.cuda().eval()
    manifest = json.loads((run / "manifest.json").read_text())
    rows = [r for r in manifest["rows"] if r["split"] == "development"]
    features = {r["id"]: torch.from_numpy(np.load(r["features"])) for r in rows}
    torch.set_num_threads(4)
    result = evaluate(model, rows, features, saved["step"])
    best = json.loads((out / "best.json").read_text())
    result["reload_matches_selection_metrics"] = result["metrics"] == best["metrics"]
    save(out / "final-development.json", result)
    if not result["reload_matches_selection_metrics"]:
        raise RuntimeError(
            "Reloaded best checkpoint metrics differ from selection evaluation"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("arm")
    p.add_argument("mode", choices=["gate", "benchmark", "train", "evaluate"])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seconds", type=float, default=300)
    a = p.parse_args()
    if a.mode == "evaluate":
        final_evaluation(a.run, a.arm)
    else:
        worker(a.run, a.arm, a.mode, a.steps, a.seconds)
