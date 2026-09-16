"""Bounded full-size FP learnability gate on a frozen, training-only speech subset.

This tests memorization, not generalization or clinical accuracy. It never reads
calibration/dev/test examples. Original pilot model and trainer are unchanged.
"""

import argparse
import collections
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import (
    ART,
    decode,
    digest,
    digit_string,
    encode,
    norm,
    read_manifest,
    save,
    scores,
    storage,
)
from model import Model


def plain(text):
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", norm(text)).split())


def select_rows(count):
    teachers = json.loads(
        (ART / "runs/pilot-8h-20260910T044750Z/targets.json").read_text()
    )["targets"]
    groups = collections.defaultdict(list)
    for row in read_manifest()["rows"]:
        if row["split"] != "train" or not 1 <= row["seconds"] <= 6:
            continue
        accepted = teachers.get(row["id"], {})
        # Teacher agreement screens obvious label problems; it is not a human audit.
        if "omi" not in accepted or plain(accepted["omi"]) != plain(row["text"]):
            continue
        if not encode(row["text"]):
            continue
        groups[row["domain"]].append(row)
    desired = (
        {"general": 1}
        if count == 1
        else {"general": 12, "medical_symptoms": 12, "digits": 8}
    )
    chosen = []
    for domain, n in desired.items():
        seen = set()
        # Fixed order, short examples first. No held-out performance-based selection.
        for row in sorted(groups[domain], key=lambda r: (r["seconds"], r["id"])):
            if plain(row["text"]) in seen:
                continue
            seen.add(plain(row["text"]))
            chosen.append(row)
            if len(seen) == n:
                break
        if len(seen) != n:
            raise RuntimeError(f"Not enough eligible training clips: {domain}")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--examples", type=int, choices=[1, 32], default=32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seconds", type=float, default=240)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--head-init-scale", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=50)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        parser.error("name must be a simple directory name")
    if (
        min(args.steps, args.seconds, args.batch, args.lr, args.warmup, args.eval_every)
        <= 0
    ):
        parser.error("training limits must be positive")
    storage()
    out = ART / "training-repair" / args.name
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path(__file__).with_name("pilot.json").read_text())["model"]
    cfg["dropout"] = args.dropout
    rows = select_rows(args.examples)
    xs = []
    ys = []
    for row in rows:
        assert row["split"] == "train"
        assert digest(row["features"]) == row["feature_sha256"]
        x = torch.from_numpy(np.load(row["features"]))
        y = encode(row["text"])
        assert x.ndim == 2 and x.shape[0] == 80 and torch.isfinite(x).all()
        assert (x.shape[1] + 1) // 2 >= len(y) + sum(a == b for a, b in zip(y, y[1:]))
        xs.append(x.cuda())
        ys.append(torch.tensor(y, dtype=torch.long, device="cuda"))
    seed = 20260910
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    model = Model(cfg, "fp").cuda().train()
    with torch.no_grad():
        model.head.weight.mul_(args.head_init_scale)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01, foreach=False
    )
    save(
        out / "config.json",
        {
            "arguments": vars(args),
            "model": cfg,
            "seed": seed,
            "parameters": sum(p.numel() for p in model.parameters()),
            "precision": "fp",
            "supervision": "ground_truth_only",
            "compute_dtype": "bfloat16_autocast",
            "gate": "Two consecutive checks: WER <= 2%, CER <= 1%, no empty outputs, all digit sequences exact.",
            "scope": "Training-subset memorization only; no held-out or medical terminology accuracy claim.",
            "model_source_sha256": digest(Path(__file__).with_name("model.py")),
            "probe_source_sha256": digest(__file__),
        },
    )
    save(
        out / "subset.json",
        {
            "rows": rows,
            "selection": "Shortest unique training transcripts with Omi agreement after punctuation normalization; inherited public labels, not human audited.",
            "subset_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True).encode()
            ).hexdigest(),
        },
    )

    def batch(indices):
        length = max(xs[i].shape[-1] for i in indices)
        x = torch.stack([F.pad(xs[i], (0, length - xs[i].shape[-1])) for i in indices])
        targets = torch.cat([ys[i] for i in indices])
        ilen = torch.tensor(
            [(xs[i].shape[-1] + 1) // 2 for i in indices], dtype=torch.long
        )
        tlen = torch.tensor([len(ys[i]) for i in indices], dtype=torch.long)
        return x, targets, ilen, tlen

    def evaluate(step):
        model.eval()
        predictions = []
        with torch.inference_mode():
            for i, row in enumerate(rows):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xs[i][None])
                ids = logits[0].argmax(-1)
                prediction = decode(ids.tolist())
                loss = F.ctc_loss(
                    logits.float().log_softmax(-1).transpose(0, 1),
                    ys[i],
                    torch.tensor([len(ids)]),
                    torch.tensor([len(ys[i])]),
                    zero_infinity=False,
                )
                predictions.append(
                    {
                        "id": row["id"],
                        "domain": row["domain"],
                        "reference": norm(row["text"]),
                        "prediction": prediction,
                        "ctc_loss": float(loss),
                        "blank_frame_fraction": float((ids == 0).float().mean()),
                    }
                )
        metric = scores([(r["reference"], r["prediction"]) for r in predictions])
        digits = [r for r in predictions if r["domain"] == "digits"]
        correct = sum(
            digit_string(r["reference"]) == digit_string(r["prediction"])
            for r in digits
        )
        metric.update(
            step=step,
            elapsed_seconds=time.monotonic() - started,
            distinct_predictions=len({r["prediction"] for r in predictions}),
            exact_utterances=sum(
                r["reference"] == r["prediction"] for r in predictions
            ),
            digit_exact=correct,
            digit_total=len(digits),
            blank_frame_fraction=float(
                np.mean([r["blank_frame_fraction"] for r in predictions])
            ),
            ctc_loss=float(np.mean([r["ctc_loss"] for r in predictions])),
        )
        metric["gate_pass"] = (
            metric["wer"] <= 0.02
            and metric["cer"] <= 0.01
            and not metric["empty_outputs"]
            and correct == len(digits)
        )
        save(
            out / f"eval-{step:06d}.json",
            {"metrics": metric, "predictions": predictions},
        )
        save(out / "progress.json", metric)
        print(json.dumps(metric), flush=True)
        model.train()
        return metric

    started = time.monotonic()
    step = 0
    streak = 0
    status = "step_limit"
    evaluate(0)
    order = []
    final_metric = None
    for step in range(1, args.steps + 1):
        if time.monotonic() - started >= args.seconds:
            step -= 1
            status = "time_limit"
            break
        if not order:
            order = rng.permutation(len(rows)).tolist()
        indices, order = order[: args.batch], order[args.batch :]
        x, y, ilen, tlen = batch(indices)
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group["lr"] = args.lr * min(1.0, step / args.warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.ctc_loss(
            logits.float().log_softmax(-1).transpose(0, 1),
            y,
            ilen,
            tlen,
            zero_infinity=False,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        opt.step()
        with (out / "training.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": float(loss.detach()),
                        "grad_norm": float(grad),
                        "lr": opt.param_groups[0]["lr"],
                        "ids": [rows[i]["id"] for i in indices],
                    }
                )
                + "\n"
            )
        if step % args.eval_every == 0:
            final_metric = evaluate(step)
            streak = streak + 1 if final_metric["gate_pass"] else 0
            if streak >= 2:
                status = "passed_memorization_gate"
                break
    if final_metric is None or final_metric["step"] != step:
        final_metric = evaluate(step)
    save(
        out / "result.json",
        {
            "status": status,
            "steps": step,
            "elapsed_seconds": time.monotonic() - started,
            "metrics": final_metric,
            "consecutive_passes": streak,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        },
    )
    if status == "passed_memorization_gate":
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model": cfg,
                "precision": "fp",
                "step": step,
                "optimizer_saved": False,
            },
            out / "weights.pt",
        )


if __name__ == "__main__":
    main()
