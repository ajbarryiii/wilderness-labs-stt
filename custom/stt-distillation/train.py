"""Fixed-data, fixed-seed sequence distillation with matched FP/ternary arms."""

import argparse, collections, hashlib, json, math, signal, time, shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import ART, read_manifest, save, digest, encode, decode, scores
from model import Model
from export import export_model

stop = False


def halted(*args):
    global stop
    stop = True


def train(run, arm_name, steps, seconds, smoke=False):
    signal.signal(signal.SIGTERM, halted)
    signal.signal(signal.SIGINT, halted)
    cfg = json.loads((run / "config.json").read_text())
    arm = next(a for a in cfg["arms"] if a["name"] == arm_name)
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    rows = [r for r in read_manifest()["rows"] if r["split"] == "train"]
    targets = json.loads((run / "targets.json").read_text())
    rows = [r for r in rows if r["id"] in targets["targets"]]
    groups = {
        d: [r for r in rows if r["domain"] == d]
        for d in ["general", "medical_symptoms", "digits"]
    }
    assert all(groups.values())
    # Separate RNGs keep sample order independent of teacher routing decisions.
    sample_rng = np.random.default_rng(cfg["seed"])
    teacher_rng = np.random.default_rng(cfg["seed"] + 1)
    for r in rows:
        assert digest(r["features"]) == r["feature_sha256"], (
            f"Feature hash mismatch: {r['id']}"
        )
    features = {r["id"]: torch.from_numpy(np.load(r["features"])) for r in rows}
    out = run / arm_name
    out.mkdir(exist_ok=True)
    save(out / "arm.json", arm)
    m = Model(cfg["model"], arm["precision"]).cuda().train()
    opt = torch.optim.AdamW(
        m.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        foreach=False,
    )
    n = sum(p.numel() for p in m.parameters())
    torch.cuda.reset_peak_memory_stats()
    initial_hash = hashlib.sha256()
    for p in m.parameters():
        initial_hash.update(p.detach().cpu().numpy().tobytes())
    save(
        out / "initialization.json",
        dict(
            parameters=n,
            seed=cfg["seed"],
            latent_weight_sha256=initial_hash.hexdigest(),
        ),
    )
    started = time.monotonic()
    deadline = started + seconds
    history = []
    exposure = collections.Counter()
    routes = collections.Counter()
    sample_hash = hashlib.sha256()
    step = 0
    error = None
    checkpoint_step = 0

    def publish():
        nonlocal checkpoint_step
        checkpoint = out / "checkpoint"
        pending = out / "checkpoint-pending"
        previous = out / "checkpoint-previous"
        if pending.exists():
            shutil.rmtree(pending)
        meta = export_model(m, pending, step)
        if previous.exists():
            shutil.rmtree(previous)
        if checkpoint.exists():
            checkpoint.rename(previous)
        pending.rename(checkpoint)
        save(
            out / "checkpoint-state.json",
            dict(
                step=step, weights_sha256=meta["weights_sha256"], optimizer_saved=False
            ),
        )
        checkpoint_step = step

    try:
        for step in range(1, steps + 1):
            if stop or time.monotonic() > deadline - 120:
                step -= 1
                break
            lr = (
                cfg["learning_rate"]
                * min(1.0, step / 100)
                * (0.15 + 0.85 * 0.5 * (1 + math.cos(math.pi * step / steps)))
            )
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss_sum = gt_sum = 0.0
            batch_seconds = 0.0
            tick = time.monotonic()
            for micro in range(cfg["accumulation"]):
                domain = sample_rng.choice(list(groups), p=[0.65, 0.25, 0.1])
                r = groups[domain][int(sample_rng.integers(len(groups[domain])))]
                sample_hash.update((r["id"] + "\n").encode())
                x = features[r["id"]].unsqueeze(0).cuda()
                gt = torch.tensor(encode(r["text"]), device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = m(x)
                logp = z.float().log_softmax(-1).transpose(0, 1)
                ilen = torch.tensor([z.shape[1]])
                gl = F.ctc_loss(
                    logp, gt, ilen, torch.tensor([len(gt)]), zero_infinity=False
                )
                loss = gl
                accepted = targets["targets"].get(r["id"], {})
                kind = None
                if arm["supervision"] == "single":
                    kind = "omi" if "omi" in accepted else None
                elif accepted:
                    kinds = sorted(accepted)
                    w = np.array(
                        [targets["policy"][domain]["weights"][k] for k in kinds]
                    )
                    kind = teacher_rng.choice(kinds, p=w / w.sum())
                if kind:
                    y = torch.tensor(encode(accepted[kind]), device="cuda")
                    tl = F.ctc_loss(
                        logp, y, ilen, torch.tensor([len(y)]), zero_infinity=False
                    )
                    loss = (1 - cfg["teacher_weight"]) * gl + cfg["teacher_weight"] * tl
                    routes[kind] += 1
                else:
                    routes["ground_truth_only"] += 1
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite loss at step {step}, {r['id']}"
                    )
                (loss / cfg["accumulation"]).backward()
                loss_sum += float(loss.detach())
                gt_sum += float(gl.detach())
                batch_seconds += r["seconds"]
                exposure[domain] += r["seconds"]
            gn = float(
                torch.nn.utils.clip_grad_norm_(
                    m.parameters(), cfg["grad_clip"], error_if_nonfinite=True
                )
            )
            opt.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved() / 2**30
            if peak > cfg["limits"]["max_vram_gib"]:
                raise MemoryError(f"VRAM gate exceeded: {peak}")
            row = dict(
                step=step,
                loss=loss_sum / cfg["accumulation"],
                ground_truth_loss=gt_sum / cfg["accumulation"],
                grad_norm=gn,
                lr=lr,
                step_seconds=time.monotonic() - tick,
                audio_seconds=batch_seconds,
                elapsed_seconds=time.monotonic() - started,
                peak_reserved_gib=peak,
            )
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            history.append(row)
            if step == 1 or step % 25 == 0:
                save(out / "progress.json", row)
                print(json.dumps(row), flush=True)
            if step % 500 == 0:
                publish()
        publish()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(error, flush=True)
    finally:
        status = (
            "failed" if error else "completed" if step == steps else "budget_stopped"
        )
        save(
            out / "result.json",
            dict(
                status=status,
                error=error,
                steps=step,
                requested_steps=steps,
                checkpoint_step=checkpoint_step,
                elapsed_seconds=time.monotonic() - started,
                exposure_hours={k: v / 3600 for k, v in exposure.items()},
                teacher_routes=dict(routes),
                sample_order_sha256=sample_hash.hexdigest(),
                initial_weight_sha256=initial_hash.hexdigest(),
                parameters=n,
                mean_last_100_loss=float(np.mean([x["loss"] for x in history[-100:]]))
                if history
                else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            ),
        )
    if error:
        raise RuntimeError(error)
    # Final compact export is evaluated in a new process. No optimizer-state claim.
    if not stop:
        export_model(m, out / "export", step, packed=arm["precision"] == "ternary")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("arm")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seconds", type=float, required=True)
    a = p.parse_args()
    train(a.run, a.arm, a.steps, a.seconds)
