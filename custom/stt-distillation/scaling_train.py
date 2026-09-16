"""Paired scaling worker. GPU execution is only exposed through scaling_control."""

import collections
import contextlib
import math
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import decode, digest, encode, save
from onset_model import OnsetModel
from pilot4_train import atomic_torch_save
from recovery_core import artifact, read
from scaling_core import aggregate, checkpoint_reasons, exposure, lr_factor, selected_id, tier_ids

STOP = False


def halt(*_):
    global STOP
    STOP = True


class Features:
    def __init__(self, rows, max_gib):
        self.original = {}
        size = 0
        for r in rows:
            assert digest(artifact(r["features"])) == r["feature_sha256"]
            f = np.load(r["features"], allow_pickle=False)
            assert f.shape == (80, r["frames"]) and np.isfinite(f).all()
            assert f.max() < 2 and f.min() >= -2
            size += f.nbytes
            assert size <= max_gib * 2**30, "Preload exceeds configured RAM allowance"
            self.original[r["id"]] = torch.from_numpy(f)
        self.bytes = size

    def augmented(self, row, prefix, gain):
        assert gain <= 0
        f = self.original[row["id"]]
        if row["domain"] == "digits":
            f = f[:, 20:]
        f = (f + gain * math.log(10) / 60).clamp_min(-2)
        return F.pad(f, (prefix, 0), value=-2)


def loss_and_logits(model, xs, rows, padded_length, device):
    assert max(x.shape[-1] for x in xs) <= padded_length
    batch = torch.stack([F.pad(x, (0, padded_length - x.shape[-1]), value=-2) for x in xs]).to(device)
    labels = [encode(r["text"]) for r in rows]
    lengths = torch.tensor([(x.shape[-1] + 1) // 2 for x in xs], dtype=torch.long)
    targets = torch.tensor([c for y in labels for c in y], device=device, dtype=torch.long)
    target_lengths = torch.tensor([len(y) for y in labels], dtype=torch.long)
    assert min(target_lengths) > 0
    for length, y in zip(lengths.tolist(), labels):
        assert length >= len(y) + sum(a == b for a, b in zip(y, y[1:]))
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(batch)
    raw = F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), targets,
                     lengths, target_lengths, reduction="none", zero_infinity=False)
    if not torch.isfinite(raw).all():
        raise FloatingPointError("Nonfinite unreduced CTC loss")
    return raw, target_lengths.to(device), logits, lengths


def evaluate(model, rows, features, device):
    predictions = []
    prior = model.training
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        model.eval()
        with torch.inference_mode():
            for r in rows:
                x = features.original[r["id"]]
                raw, lengths, logits, _ = loss_and_logits(model, [x], [r], x.shape[-1], device)
                ids = logits[0].argmax(-1)
                predictions.append(dict(id=r["id"], domain=r["domain"], speaker=r.get("speaker"),
                    reference=r["text"], prediction=decode(ids.tolist()), ctc_sum=float(raw[0]),
                    target_length=int(lengths[0]), blank_fraction=float((ids == 0).float().mean())))
    finally:
        model.train(prior)
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return dict(metrics=aggregate(predictions), predictions=predictions)


def worker(run, job, *, device_name="cuda", resume=False, stop_after=None, preflight=False):
    global STOP
    STOP = False
    started = time.monotonic()
    started_epoch = time.time()
    run = artifact(run)
    cfg = read(run / "config.json")
    spec = read(run / "jobs" / (job + ".json"))
    if device_name == "cpu":
        assert cfg["model"]["width"] <= 64, "CPU smoke tests must use a tiny model"
    device = torch.device(device_name)
    torch.set_num_threads(4 if device.type == "cuda" else 1)
    torch.manual_seed(spec["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    rows = read(run / "manifest.json")["rows"]
    by_id = {r["id"]: r for r in rows}
    subsets = read(run / "subsets.json")
    assert digest(run / spec["groups"]) == spec["groups_sha256"]
    groups = read(run / spec["groups"])["groups"]
    assert digest(run / spec["tape"]) == spec["tape_sha256"]
    tape = np.load(run / spec["tape"], allow_pickle=False)
    needed = set(tier_ids(groups, spec["size"])) | {i for ids in subsets.values() for i in ids}
    features = Features([by_id[i] for i in sorted(needed)], cfg["feature_cache_gib"])
    model = OnsetModel(cfg["model"], "fp").to(device).train()
    if spec.get("initial_checkpoint") and not resume:
        assert digest(artifact(spec["initial_checkpoint"])) == spec["initial_checkpoint_sha256"]
        state = torch.load(spec["initial_checkpoint"], map_location="cpu", mmap=True, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        del state
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["peak_lr"], betas=tuple(cfg["betas"]),
                           weight_decay=cfg["weight_decay"], foreach=False)
    out = run / "training" / job
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "evaluations").mkdir(exist_ok=True)
    if (out / "latest.pt").exists() and not resume:
        raise RuntimeError("Existing worker state requires explicit resume")
    contracts = {n: digest(run / n) for n in ("config.json", "manifest.json", "subsets.json", "jobs/" + job + ".json")}
    update, emitted, records = 0, [], []
    timing = dict(training_work_seconds=0., cuda_interval_seconds=0., startup_seconds=0.,
                  evaluation_seconds=0., checkpoint_seconds=0.)
    if resume:
        state = torch.load(out / "latest.pt", map_location="cpu", mmap=True, weights_only=False)
        assert state["contracts"] == contracts, "Resume input changed"
        model.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
        update, emitted, records = state["update"], state["emitted_time_points"], state["records"]
        timing = state["timing"]
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
        meta = out / "latest.json"
        if meta.exists() and read(meta)["update"] == update:
            timing = read(meta)["timing"]
    timing["startup_seconds"] += time.monotonic() - started
    attempts_dir = out / "attempts"
    attempts_dir.mkdir(exist_ok=True)
    prior_attempts = [read(p) for p in attempts_dir.glob("*.json")]
    restored_work = timing["training_work_seconds"]
    retry_work = max(0., sum(a.get("training_work_delta", 0.) for a in prior_attempts) - restored_work)
    uncertain_attempts = sum(a.get("status") in ("running", "failed") for a in prior_attempts)
    attempt_path = attempts_dir / f"{time.time_ns()}.json"
    save(attempt_path, dict(status="running", started_epoch=started_epoch, resumed_from_update=update))
    target = max(cfg["checkpoint_updates"])
    if stop_after is not None:
        target = min(target, stop_after)
    assert update <= target
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in old_handlers:
        signal.signal(sig, halt)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def persist():
        sync()
        begin = time.monotonic()
        atomic_torch_save(out / "latest.pt", dict(model=model.state_dict(), optimizer=opt.state_dict(),
            update=update, timing=dict(timing), emitted_time_points=list(emitted), records=records,
            contracts=contracts, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None))
        timing["checkpoint_seconds"] += time.monotonic() - begin
        save(out / "latest.json", dict(update=update, timing=timing))

    def check(reasons):
        begin = time.monotonic()
        evaluation = {name: evaluate(model, [by_id[i] for i in ids], features, device)
                      for name, ids in subsets.items() if name != "gate"}
        sync()
        timing["evaluation_seconds"] += time.monotonic() - begin
        weights = out / "checkpoints" / f"step-{update:07d}.pt"
        before = time.monotonic()
        atomic_torch_save(weights, dict(model=model.state_dict(), update=update, contracts=contracts))
        timing["checkpoint_seconds"] += time.monotonic() - before
        emitted.extend(int(reason.split(":")[1]) for reason in reasons if reason.startswith("training_seconds:"))
        record = dict(job=job, tier=spec["tier"], seed=spec["seed"], update=update,
            presented=update * cfg["batch_size"], reasons=reasons, timing=dict(timing),
            resource_cost=dict(training_seconds_including_recorded_retries=timing["training_work_seconds"] + retry_work,
                measured_attempt_wall_seconds=sum(a.get("wall_seconds", 0.) for a in prior_attempts) + time.monotonic() - started,
                uncertain_prior_attempts=uncertain_attempts,
                note="Failed/killed partial updates may have unmeasured GPU work; exclude such jobs from clean time-efficiency claims."),
            weights=str(weights.relative_to(run)), evaluations=evaluation,
            exposure=exposure(rows, groups, tape[:update * cfg["batch_size"]], spec["size"], cfg["batch_size"]))
        path = out / "evaluations" / f"step-{update:07d}.json"
        save(path, record)
        # latest.pt is the commit point; analysis ignores orphan evaluations after
        # a crash. Resuming overwrites those points from the last committed state.
        records.append(str(path.relative_to(run)))
        persist()
        save(out / "index.json", dict(records=records, update=update, status="running"))

    status = "running"
    try:
        if not records:
            check(checkpoint_reasons(0, timing["training_work_seconds"], cfg, emitted))
        losses = collections.deque(maxlen=100)
        for step in range(update + 1, target + 1):
            if STOP:
                status = "interrupted"
                break
            sync()
            begin = time.monotonic()
            batch = tape[(step - 1) * cfg["batch_size"]:step * cfg["batch_size"]]
            batch_rows = [by_id[selected_id(groups, draw, spec["size"])] for draw in batch]
            xs = [features.augmented(r, int(draw[2]), cfg["gains_db"][int(draw[3])]) for r, draw in zip(batch_rows, batch)]
            for param_group in opt.param_groups:
                param_group["lr"] = cfg["peak_lr"] * lr_factor(step, cfg)
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
            raw, lengths, _, _ = loss_and_logits(model, xs, batch_rows, int(batch[:, 4].max()), device)
            loss = (raw / lengths).mean()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            opt.step()
            if device.type == "cuda":
                end_event.record()
            sync()
            timing["training_work_seconds"] += time.monotonic() - begin
            if device.type == "cuda":
                timing["cuda_interval_seconds"] += start_event.elapsed_time(end_event) / 1000
                if torch.cuda.max_memory_allocated() / 2**30 > cfg["max_vram_gib"]:
                    raise RuntimeError("VRAM exceeded configured preflight limit")
            update = step
            losses.append(float(loss.detach()))
            reasons = checkpoint_reasons(update, timing["training_work_seconds"], cfg, emitted)
            if reasons:
                check(reasons)
            elif update % cfg["recovery_every_updates"] == 0:
                persist()
                save(out / "index.json", dict(records=records, update=update, status="running"))
            if update % 100 == 0:
                save(out / "progress.json", dict(update=update, timing=timing,
                     rolling_training_ctc_loss=float(np.mean(losses)), gradient_norm=float(grad),
                     lr=opt.param_groups[0]["lr"]))
        status = "completed" if update == max(cfg["checkpoint_updates"]) else "interrupted" if STOP else "paused"
        persist()
        save(out / "index.json", dict(records=records, update=update, status=status))
        save(out / "result.json", dict(status=status, update=update, timing=timing, preflight=preflight,
             parameters=model.precision_counts(), feature_bytes=features.bytes,
             max_vram_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0))
    except Exception as exc:
        status = "failed"
        save(out / "failure.json", dict(update=update, error=f"{type(exc).__name__}: {exc}", timing=timing))
        raise
    finally:
        save(attempt_path, dict(status=status, started_epoch=started_epoch, ended_epoch=time.time(),
            wall_seconds=time.monotonic() - started, training_work_delta=timing["training_work_seconds"] - restored_work,
            last_update=update))
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return read(out / "result.json")
