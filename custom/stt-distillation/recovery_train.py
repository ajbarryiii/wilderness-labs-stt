"""Bounded FP recovery worker with exact exposure/resume and acoustic diagnostics."""

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
import soundfile as sf
import torch
from torch.nn import functional as F

from common import decode, digest, encode, save
from onset_model import OnsetModel, startup_mask
from pilot4_train import atomic_torch_save, evaluate, passed
from prepare import feature
from recovery_core import DOMAINS, artifact, exposure, lr_factor, read, selection

STOP = False


def halt(*_):
    global STOP
    STOP = True


class Features:
    def __init__(self, rows):
        self.original = {}
        self.cache = collections.OrderedDict()
        self.rows = {r["id"]: r for r in rows}
        for row in rows:
            assert digest(row["features"]) == row["feature_sha256"]
            x = torch.from_numpy(np.load(row["features"], allow_pickle=False))
            assert x.ndim == 2 and x.shape[0] == 80 and torch.isfinite(x).all()
            self.original[row["id"]] = x

    def augmented(self, row, prefix, gain, presentation=None):
        key = row["id"], gain
        if key not in self.cache:
            assert row["split"] == "train"
            assert digest(row["audio"]) == row["audio_sha256"]
            audio, sr = sf.read(row["audio"], dtype="float32")
            assert sr == 16000 and audio.ndim == 1
            if row["domain"] == "digits":
                assert not np.any(audio[:3200])
                audio = audio[3200:]
            self.cache[key] = torch.from_numpy(feature(audio * 10 ** (gain / 20)))
            if len(self.cache) > 18000:
                self.cache.popitem(last=False)
        return F.pad(self.cache[key], (int(prefix), 0), value=-2)


def batch_loss(model, xs, rows, precision="bf16"):
    length = max(x.shape[-1] for x in xs)
    batch = torch.stack([F.pad(x, (0, length - x.shape[-1])) for x in xs]).cuda()
    labels = [encode(r["text"]) for r in rows]
    ilen = torch.tensor([(x.shape[-1] + 1) // 2 for x in xs])
    tlen = torch.tensor([len(y) for y in labels])
    target = torch.tensor(
        [c for y in labels for c in y], device="cuda", dtype=torch.long
    )
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
        logits = model(batch)
    raw = F.ctc_loss(
        logits.float().log_softmax(-1).transpose(0, 1),
        target,
        ilen,
        tlen,
        reduction="none",
        zero_infinity=False,
    )
    if not torch.isfinite(raw).all():
        raise FloatingPointError("Nonfinite unreduced CTC loss")
    return raw, tlen.to(raw.device)


def acoustic_probe(model, rows, features):
    prior = model.training
    model.eval()
    records = []
    with torch.inference_mode():
        for row in rows:
            stats, handles = [], []

            def hook(module, inputs, output):
                z = output[:, 50:].float()
                stats.append(
                    float(z.std(dim=1, correction=0).mean()) if z.shape[1] > 1 else None
                )

            try:
                handles = [b.register_forward_hook(hook) for b in model.blocks]
                x = features[row["id"]][None].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model(x).float()
            finally:
                for handle in handles:
                    handle.remove()
            p = z.softmax(-1)
            variants = {}
            for name, alternate in [
                ("reverse", x.flip(-1)),
                ("constant", x.mean(-1, keepdim=True).expand_as(x)),
            ]:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    az = model(alternate).float()
                variants[name] = dict(
                    prediction=decode(az[0].argmax(-1).tolist()),
                    probability_delta=float(
                        (p[:, 50:] - az.softmax(-1)[:, 50:]).abs().mean()
                    ),
                )
            records.append(
                dict(
                    id=row["id"],
                    domain=row["domain"],
                    temporal_std=stats,
                    prediction=decode(z[0].argmax(-1).tolist()),
                    variants=variants,
                )
            )
    model.train(prior)
    mid = [r["temporal_std"][min(7, len(model.blocks) - 1)] for r in records]
    insensitive = sum(
        r["variants"]["reverse"]["probability_delta"] < 1e-4 for r in records
    )
    return dict(
        records=records,
        middle_temporal_std=float(np.median(mid)),
        insensitive_probes=insensitive,
        flagged=insensitive >= 10 and float(np.median(mid)) < 0.0075,
    )


def gradient_probe(model, rows, features):
    """Measured gradient norms and sampled directional comparisons by domain."""
    result, vectors = {}, {}
    prior = model.training
    model.train()
    params = [
        model.front.weight,
        model.blocks[len(model.blocks) // 2].qkv.weight,
        model.head.weight,
    ]
    for row in rows:
        model.zero_grad(set_to_none=True)
        raw, length = batch_loss(model, [features[row["id"]]], [row])
        loss = (raw / length).mean()
        loss.backward()
        norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    p.grad.detach().float().norm()
                    for p in model.parameters()
                    if p.grad is not None
                ]
            )
        )
        vector = torch.cat(
            [
                p.grad.detach().flatten()[:: max(1, p.numel() // 2048)].float()
                for p in params
            ]
        )
        vectors[row["domain"]] = vector
        result[row["domain"]] = dict(
            id=row["id"],
            normalized_loss=float(loss.detach()),
            gradient_norm=float(norm),
        )
    result["sampled_cosines"] = {
        a + "/" + b: float(F.cosine_similarity(vectors[a][None], vectors[b][None]))
        for i, a in enumerate(DOMAINS)
        for b in DOMAINS[i + 1 :]
    }
    model.zero_grad(set_to_none=True)
    model.train(prior)
    return result


def run_worker(run, job, target, seconds, resume=False, *, feature_factory=None):
    global STOP
    STOP = False
    run = artifact(run)
    cfg = read(run / "config.json")
    spec = read(run / "jobs" / f"{job}.json")
    out = run / "training" / job
    out.mkdir(parents=True, exist_ok=True)
    if (out / "latest.pt").exists() and not resume:
        raise RuntimeError("Existing checkpoint requires --resume")
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(spec["seed"])
    started = time.monotonic()
    rows = read(run / "manifest.json")["rows"]
    by_id = {r["id"]: r for r in rows}
    subsets = read(run / "subsets.json")
    train_rows = [by_id[i] for i in spec["row_ids"]]
    draws_path = Path(spec["tape"])
    assert digest(draws_path) == spec["tape_sha256"]
    draws = np.load(draws_path, allow_pickle=False, mmap_mode="r")
    assert target <= len(draws)
    dev = [r for r in rows if r["split"] == "development"]
    monitor = [by_id[i] for i in subsets["monitor"]]
    gate = [by_id[i] for i in subsets["gate"]]
    if spec.get("gate"):
        monitor = train_rows
    probes = [r for d in DOMAINS for r in [x for x in monitor if x["domain"] == d][:4]]
    needed = {r["id"]: r for r in train_rows + dev + monitor + gate}
    features = (Features(list(needed.values())) if feature_factory is None
                else feature_factory(list(needed.values()), run, spec))
    model = OnsetModel(cfg["model"], spec.get("precision", "fp")).cuda().train()
    if spec.get("initial_checkpoint"):
        assert digest(spec["initial_checkpoint"]) == spec["initial_checkpoint_sha256"]
        initial = torch.load(
            spec["initial_checkpoint"],
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        model.load_state_dict(initial["model"], strict=True)
        del initial
    opt = torch.optim.AdamW(
        [
            {
                "params": [
                    p for n, p in model.named_parameters() if not n.startswith("head.")
                ],
                "peak_lr": spec["encoder_lr"],
            },
            {"params": list(model.head.parameters()), "peak_lr": spec["head_lr"]},
        ],
        lr=spec["head_lr"],
        betas=(0.9, 0.95),
        weight_decay=0.01,
        foreach=False,
    )
    presented = updates = collapse_streak = gate_streak = 0
    best = math.inf
    status, error = "completed", None
    history, final = [], None
    initial_hash = None
    if resume:
        state = torch.load(
            out / "latest.pt", map_location="cpu", mmap=True, weights_only=False
        )
        assert state["job_hash"] == digest(run / "jobs" / f"{job}.json")
        assert state["config_hash"] == digest(run / "config.json")
        model.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
        presented, updates = state["presented"], state["step"]
        best = state["best"] if state["best"] is not None else math.inf
        collapse_streak, gate_streak = state["collapse_streak"], state["gate_streak"]
        history = state["speech_history"]
        initial_hash = state["initial_hash"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
    if initial_hash is None:
        h = hashlib.sha256()
        for tensor in model.state_dict().values():
            h.update(tensor.detach().cpu().numpy().tobytes())
        initial_hash = h.hexdigest()
        save(
            out / "initialization.json",
            dict(sha256=initial_hash, spec=spec, parameters=model.precision_counts()),
        )
    segment_start = presented
    events = out / f"metrics-from-{presented:07d}-{time.time_ns()}.jsonl"

    def checkpoint():
        atomic_torch_save(
            out / "latest.pt",
            dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                config=cfg,
                arm={"precision": spec.get("precision", "fp")},
                step=updates,
                presented=presented,
                best=best if math.isfinite(best) else None,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                initial_hash=initial_hash,
                collapse_streak=collapse_streak,
                gate_streak=gate_streak,
                speech_history=history,
                job_hash=digest(run / "jobs" / f"{job}.json"),
                config_hash=digest(run / "config.json"),
            ),
        )
        save(
            out / "checkpoint.json",
            dict(
                presented=presented,
                updates=updates,
                resumable=True,
                **exposure(train_rows, draws[:presented], spec["gains"]),
            ),
        )

    def check():
        nonlocal final, best, collapse_streak, gate_streak
        # Evaluation and diagnostics must not perturb training RNG state.
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        try:
            tr = evaluate(model, monitor, features.original, updates)
            dv = evaluate(model, dev, features.original, updates)
            acoustic = acoustic_probe(model, probes, features.original)
            final = dict(
                presented=presented,
                step=updates,
                training=tr,
                development=dv,
                acoustic=acoustic,
                elapsed_seconds=time.monotonic() - started,
            )
            if hasattr(features, "validation"):
                final["robustness"] = {
                    name: evaluate(model, items, feats, updates)
                    for name, (items, feats) in features.validation.items()
                }
            if presented in cfg.get("gradient_probe_at", []) and not spec.get("gate"):
                pr = [next(r for r in monitor if r["domain"] == d) for d in DOMAINS]
                final["domain_gradients"] = gradient_probe(model, pr, features.original)
            score = selection(dv["metrics"])
            if score < best and not spec.get("gate"):
                best = score
                atomic_torch_save(
                    out / "best.pt",
                    dict(
                        model=model.state_dict(),
                        config=cfg,
                        arm={"precision": spec.get("precision", "fp")},
                        step=updates,
                        presented=presented,
                    ),
                )
                save(
                    out / "best.json",
                    dict(
                        selection_score=score,
                        presented=presented,
                        metrics=dv["metrics"],
                        criterion="equal-domain speech CER",
                    ),
                )
            history.append(score)
            stagnant = len(history) >= 3 and min(history[-3:-1]) - score < 0.01
            collapsed = (
                presented >= cfg["collapse_grace"]
                and score >= 0.9
                and stagnant
                and acoustic["flagged"]
            )
            collapse_streak = collapse_streak + 1 if collapsed else 0
            if spec.get("gate"):
                transforms = []
                model.eval()
                with torch.inference_mode():
                    for row in [r for r in gate if r["domain"] == "digits"]:
                        for prefix in (0, 60):
                            for gain in (0, -24):
                                x = features.augmented(row, prefix, gain)[None].cuda()
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    pred = decode(model(x)[0].argmax(-1).tolist())
                                transforms.append(pred == row["text"])
                model.train()
                final["transformed_digits"] = dict(
                    exact=sum(transforms), total=len(transforms)
                )
                gate_streak = (
                    gate_streak + 1
                    if passed(tr["metrics"], cfg) and all(transforms)
                    else 0
                )
            save(out / f"eval-{presented:07d}.json", final)
            save(out / "progress.json", final)
            print(
                json.dumps(
                    dict(
                        job=job,
                        presented=presented,
                        speech_cer=score,
                        digits=dv["metrics"]["digits"]["exact"],
                        acoustic_flag=acoustic["flagged"],
                    )
                ),
                flush=True,
            )
        finally:
            model.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)

    try:
        if not resume:
            check()
        effective = spec.get("effective_batch", 4)
        assert effective % 4 == 0 and target % effective == 0
        with events.open("a", buffering=1) as logfile:
            while presented < target:
                if STOP or time.monotonic() - started > seconds - 120:
                    status = "cancelled" if STOP else "time_limit"
                    break
                tick = time.monotonic()
                selected = draws[presented : presented + effective]
                row_batch = [train_rows[int(v[0])] for v in selected]
                total_chars = sum(len(encode(r["text"])) for r in row_batch)
                factor = (
                    min(1.0, (presented + effective) / 800)
                    if spec.get("gate")
                    else lr_factor(presented + effective)
                )
                for group in opt.param_groups:
                    group["lr"] = group["peak_lr"] * factor
                opt.zero_grad(set_to_none=True)
                losses = []
                aug_stats = []
                for offset in range(0, effective, 4):
                    rs = row_batch[offset : offset + 4]
                    ds = selected[offset : offset + 4]
                    xs = [
                        features.augmented(r, int(d[1]), spec["gains"][int(d[2])],
                                           presentation=presented + offset + j)
                        for j, (r, d) in enumerate(zip(rs, ds))
                    ]
                    raw, lengths = batch_loss(
                        model, xs, rs, spec.get("training_dtype", "bf16")
                    )
                    loss = (
                        raw.sum() / total_chars
                        if spec.get("loss_reduction") == "total_characters"
                        else (raw / lengths).sum() / effective
                    )
                    loss.backward()
                    losses.extend(
                        (r["domain"], float(v), int(n))
                        for r, v, n in zip(rs, raw.detach(), lengths)
                    )
                    if (updates + 1) % 100 == 0:
                        for r, d, x in zip(rs, ds, xs):
                            allowed = startup_mask(x[None])[0]
                            aug_stats.append(
                                dict(
                                    domain=r["domain"],
                                    gain=spec["gains"][int(d[2])],
                                    floor_fraction=float((x <= -1.999).float().mean()),
                                    onset_ms=int((~allowed).sum()) * 10,
                                    extra=getattr(features, "last_details", {}).get(r["id"]),
                                )
                            )
                grad = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                tracked = [
                    model.front.weight,
                    model.blocks[len(model.blocks) // 2].qkv.weight,
                    model.head.weight,
                ]
                before = (
                    [
                        p.detach().flatten()[:: max(1, p.numel() // 4096)].clone()
                        for p in tracked
                    ]
                    if (updates + 1) % 100 == 0
                    else []
                )
                opt.step()
                torch.cuda.synchronize()
                presented += effective
                updates += 1
                peak = torch.cuda.max_memory_reserved() / 2**30
                if peak > cfg["max_vram_gib"]:
                    raise MemoryError(f"VRAM exceeds {cfg['max_vram_gib']} GiB: {peak}")
                record = dict(
                    presented=presented,
                    step=updates,
                    loss=sum(v / n for _, v, n in losses) / len(losses),
                    loss_reduction=spec.get("loss_reduction", "mean_label_normalized"),
                    preclip_grad_norm=float(grad),
                    lr=[g["lr"] for g in opt.param_groups],
                    update_seconds=time.monotonic() - tick,
                    peak_reserved_gib=peak,
                    domains={
                        d: dict(
                            examples=sum(a == d for a, _, _ in losses),
                            normalized_loss_sum=sum(
                                v / n for a, v, n in losses if a == d
                            ),
                        )
                        for d in DOMAINS
                    },
                )
                if before:
                    record["sampled_update_ratios"] = [
                        float(
                            (
                                p.detach().flatten()[:: max(1, p.numel() // 4096)] - b
                            ).norm()
                            / b.norm().clamp_min(1e-12)
                        )
                        for p, b in zip(tracked, before)
                    ]
                    record["augmentation"] = aug_stats
                logfile.write(json.dumps(record) + "\n")
                if updates % 100 == 0:
                    save(out / "training-progress.json", record)
                interval = 400 if spec.get("gate") else cfg["eval_every_examples"]
                if presented % interval == 0:
                    check()
                    if gate_streak >= 2:
                        status = "passed_gate"
                        break
                    if collapse_streak >= 2:
                        status = "collapsed"
                        break
                if presented % cfg["checkpoint_every_examples"] == 0:
                    checkpoint()
        if final is None or final["presented"] != presented:
            check()
        checkpoint()
        if spec.get("gate") and status == "completed":
            status = "gate_failed"
    except Exception as exc:
        traceback.print_exc()
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        # Keep the previous complete checkpoint; an interrupted optimizer update is not resumable.
    result = dict(
        status=status,
        error=error,
        job=job,
        presented=presented,
        updates=updates,
        segment_start=segment_start,
        requested_presentations=target,
        elapsed_seconds=time.monotonic() - started,
        initial_hash=initial_hash,
        best_selection_score=best if math.isfinite(best) else None,
        final=final,
        acoustic_augmentation=spec.get("acoustic_augmentation"),
        **exposure(train_rows, draws[:presented], spec["gains"]),
    )
    save(out / "result.json", result)
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("job")
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_worker(args.run, args.job, args.target, args.seconds, args.resume)
