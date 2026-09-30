"""Fine-tune Whisper tiny.en on LibriSpeech training data: FP32 control or ternary QAT. See DESIGN.md.

One invocation is one (arm, learning rate) run. The best checkpoint is chosen by
greedy WER on the fixed dev-clean subset; full dev-clean is then scored once for
the learning-rate choice. Ternary arms are exported and scored on the model
rebuilt from the export, never on the training graph. No test split is read.

Revision 2 options (DESIGN.md "Revision 2"), all off by default so that a run
without them reproduces the v1 recipe exactly:

- --quant-ramp-fraction: progressive quantization. The ternary weight fraction
  f rises linearly to 1 over that fraction of the steps, then stays 1. Only
  dev-subset evaluations at f == 1 may select the best checkpoint; the export
  and the final dev-clean score always use f == 1. Ignored for the fp32 arm.
- --distill-weight / --distill-temperature: loss (1 - w) * CE + w * KD, where KD
  is KL(teacher || student) over label positions against a frozen FP32 copy of
  the pretrained checkpoint, teacher-forced on the same labels. All arms.
- --train-splits: concatenation of several LibriSpeech training splits.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
import transformers

import checkpoint
import data
import decoding
import export
import paths
import quant

ARMS = ("fp32", "ternary", "ternary-embed")
LOG_EVERY, PRINT_EVERY = 10, 50
# Everything a run writes; cleared before a (re)start so no stale output survives.
OUTPUTS = ("summary.json", "config.json", "metrics.jsonl", "best.pt", "last.pt", "best-hf",
           "export", "eval-dev-clean.json", "dev-subset")
# summary.json contract checked by sweep.py.
SUMMARY_TYPES = {"run_name": str, "arm": str, "lr": float, "max_steps": int, "steps": int,
                 "best_step": int, "dev_subset_wer": float, "dev_clean_wer": float, "seed": int,
                 "batch_size": int, "train_utterances": int, "quant_ramp_fraction": float,
                 "distill_weight": float, "distill_temperature": float,
                 "started_utc": str, "finished_utc": str}
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"  # ISO 8601; aligns runs with powerlog.py logs


def _training_splits(value: str) -> list[str]:
    try:
        return data.check_training_splits([s.strip() for s in value.split(",")])
    except (TypeError, ValueError) as err:
        raise argparse.ArgumentTypeError(str(err)) from None


def _unit_interval(value: str) -> float:
    x = float(value)
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is not in [0, 1]")
    return x


def _positive(value: str) -> float:
    x = float(value)
    if not (math.isfinite(x) and x > 0.0):
        raise argparse.ArgumentTypeError(f"{value} is not a positive finite number")
    return x


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--dev-subset", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=paths.SEED)
    parser.add_argument("--limit-train", type=int,
                        help="first N utterances of the concatenated id-sorted splits (smoke only)")
    parser.add_argument("--train-splits", type=_training_splits, default="train-clean-100",
                        help="comma-separated LibriSpeech training splits, concatenated in this "
                             "order (default: train-clean-100)")
    parser.add_argument("--quant-ramp-fraction", type=_unit_interval, default=0.0,
                        help="ternary arms: ramp the weight fraction f from 0 to 1 over this "
                             "fraction of --max-steps (default 0: fully ternary from step 1)")
    parser.add_argument("--distill-weight", type=_unit_interval, default=0.0,
                        help="weight w of KL(teacher || student) in (1 - w) * CE + w * KD "
                             "(default 0: plain CE, no teacher)")
    parser.add_argument("--distill-temperature", type=_positive, default=1.0,
                        help="softmax temperature T of the KD term, scaled by T^2 (default 1)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup reaching the peak on update `warmup`, then linear decay to 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def ramp_steps(fraction: float, max_steps: int) -> int:
    """Length of the progressive-quantization ramp in updates, round(fraction * max_steps).

    0 means no ramp (v1: fully ternary from the first update). A positive
    fraction always yields at least one step.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"ramp fraction {fraction} not in [0, 1]")
    return 0 if fraction == 0.0 else max(1, round(fraction * max_steps))


def ramp_weight_fraction(update: int, steps: int) -> float:
    """Weight fraction f for 1-based optimizer update k: min(1, k / ramp_steps); 1.0 without a ramp."""
    if update < 1:
        raise ValueError(f"updates are 1-based, got {update}")
    return 1.0 if steps <= 0 else min(1.0, update / steps)


def selectable(weight_fraction: float | None) -> bool:
    """Only fully quantized (f == 1) or FP32 (None) evaluations may select the best checkpoint."""
    return weight_fraction is None or weight_fraction == 1.0


def improves(score: float, weight_fraction: float | None, best: float) -> bool:
    """The best-checkpoint rule: a selectable evaluation with a strictly lower dev-subset WER."""
    return selectable(weight_fraction) and score < best


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor,
            temperature: float) -> torch.Tensor:
    """T^2 * mean over label positions (labels != -100) of KL(teacher || student) at temperature T.

    Computed in FP32. Rows are selected before the softmax, which gives the
    same per-position values as computing over [B, L, V] and masking afterwards
    while never materializing FP32 copies of the padding positions.
    """
    mask = labels != -100
    s = student_logits[mask].float() / temperature
    t = teacher_logits[mask].float() / temperature
    kl = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True,
                  reduction="none").sum(-1)
    return kl.mean() * (temperature * temperature)


def combine_loss(ce: torch.Tensor, kd: torch.Tensor | None, weight: float) -> torch.Tensor:
    """(1 - w) * ce + w * kd; exactly ce (the same tensor) when w == 0."""
    if weight == 0.0:
        return ce
    return (1.0 - weight) * ce + weight * kd


def compute_loss(model, teacher, features: torch.Tensor, labels: torch.Tensor,
                 distill_weight: float, temperature: float
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """(loss, ce, kd) for one batch; kd is None without a teacher, and then loss is ce.

    Student and teacher run the same BF16 autocast forward on the same features
    and labels, so Hugging Face builds identical shifted decoder inputs. Teacher
    logits live only inside this call.
    """
    device = features.device.type
    teacher_logits = None
    if teacher is not None:
        with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            teacher_logits = teacher(input_features=features, labels=labels).logits
    with torch.autocast(device, dtype=torch.bfloat16):
        outputs = model(input_features=features, labels=labels)
    ce = outputs.loss
    if teacher_logits is None:
        return ce, ce, None
    kd = kd_loss(outputs.logits, teacher_logits, labels, temperature)
    del outputs, teacher_logits
    return combine_loss(ce, kd, distill_weight), ce, kd


def load_teacher():
    """Frozen FP32 copy of the pinned pretrained checkpoint on the GPU: eval mode, no gradients,
    never quantized."""
    teacher = checkpoint.load_model("cuda")
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def teacher_source() -> dict:
    return {"model_dir": str(paths.MODEL_DIR), "revision": paths.MODEL_REVISION,
            **checkpoint.base_provenance(), "weights": "FP32, frozen, never quantized",
            "forward": "no_grad, BF16 autocast, teacher-forced on the same labels"}


def git_head() -> str:
    out = subprocess.run(["git", "-C", str(paths.REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def clear_outputs(run) -> None:
    for name in OUTPUTS:
        path = run / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def subset_wer(model, processor, subset: list[dict], step: int, run,
               weight_fraction: float | None) -> float:
    """Dev-subset WER at the model's current weight fraction; records saved to dev-subset/step-NNNNN.json."""
    records, info = decoding.decode(model, processor, subset, "cuda")
    result = {"step": step, "subset": f"data.dev_subset(dev-clean, {len(subset)})",
              "weight_fraction": weight_fraction, "selectable": selectable(weight_fraction),
              **decoding.report(records, info, "dev-clean", None, None)}
    data.write_json(run / "dev-subset" / f"step-{step:05d}.json", result)
    return result["wer"]["wer"]


def train(args, model, processor, manifest, run, metrics, teacher=None,
          ramp: int | None = None) -> dict:
    """The optimization loop; returns timing and counters. Writes best.pt and last.pt.

    metrics.jsonl gets, every 10 steps, the mean loss, CE, KD (null without a
    teacher) and pre-clip gradient norm over those steps, and the learning rate
    and weight fraction (null for fp32) of the latest step; eval lines carry
    the dev-subset WER, the weight fraction it was scored at, and whether it
    was eligible for checkpoint selection. `ramp` is the ramp length in steps
    for ternary arms (0 = none) and None for fp32.
    """
    dataset = data.LibriSpeechDataset(manifest, processor.feature_extractor, processor.tokenizer,
                                      model.config.decoder_start_token_id)
    loader, sampler = data.train_loader(dataset, args.batch_size, args.workers, args.seed)
    subset = data.dev_subset(data.build_manifest("dev-clean"), args.dev_subset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, args.warmup, args.max_steps))
    step = seen = epoch = 0
    best = math.inf
    eval_s = 0.0
    fraction = None
    loss_sum = torch.zeros((), device="cuda")
    ce_sum = torch.zeros((), device="cuda")
    kd_sum = torch.zeros((), device="cuda")
    norm_sum = torch.zeros((), device="cuda")
    start = mark = time.perf_counter()
    mark_step = 0
    model.train()
    while step < args.max_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            features = batch["input_features"].cuda(non_blocking=True)
            labels = batch["labels"].cuda(non_blocking=True)
            lr = optimizer.param_groups[0]["lr"]
            if ramp is not None:
                fraction = ramp_weight_fraction(step + 1, ramp)
                quant.set_weight_fraction(model, fraction)
            loss, ce, kd = compute_loss(model, teacher, features, labels, args.distill_weight,
                                        args.distill_temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step, seen = step + 1, seen + len(batch["ids"])
            loss_sum += loss.detach()
            ce_sum += ce.detach()
            if kd is not None:
                kd_sum += kd.detach()
            norm_sum += grad_norm
            del loss, ce, kd
            if step % LOG_EVERY == 0 or step == args.max_steps:
                n = step % LOG_EVERY or LOG_EVERY
                mean_loss, mean_norm = loss_sum.item() / n, norm_sum.item() / n
                mean_ce = ce_sum.item() / n
                mean_kd = kd_sum.item() / n if teacher is not None else None
                values = [mean_loss, mean_norm, mean_ce] + ([] if mean_kd is None else [mean_kd])
                if not all(math.isfinite(v) for v in values):
                    raise RuntimeError(f"non-finite loss or gradient norm at step {step}")
                for total in (loss_sum, ce_sum, kd_sum, norm_sum):
                    total.zero_()
                metrics.write(json.dumps({"step": step, "loss": mean_loss, "ce": mean_ce,
                                          "kd": mean_kd, "lr": lr, "grad_norm": mean_norm,
                                          "weight_fraction": fraction, "examples_seen": seen,
                                          "epoch": epoch,
                                          "elapsed_s": time.perf_counter() - start}) + "\n")
                metrics.flush()
                if step % PRINT_EVERY == 0:
                    per_step = (time.perf_counter() - mark) / (step - mark_step)
                    extra = "" if teacher is None else f" ce {mean_ce:.4f} kd {mean_kd:.4f}"
                    extra += "" if not ramp else f" f {fraction:.4f}"
                    print(f"step {step}/{args.max_steps} loss {mean_loss:.4f}{extra} lr {lr:.2e} "
                          f"gnorm {mean_norm:.2f} {per_step:.3f}s/step", flush=True)
                    mark, mark_step = time.perf_counter(), step
            if step % args.eval_every == 0 or step == args.max_steps:
                t = time.perf_counter()
                score = subset_wer(model, processor, subset, step, run, fraction)
                model.train()
                eligible = selectable(fraction)
                state = {"state_dict": model.state_dict(), "step": step, "dev_subset_wer": score,
                         "weight_fraction": fraction, "arm": args.arm, "args": vars(args)}
                if improves(score, fraction, best):
                    best = score
                    torch.save(state, run / "best.pt")
                torch.save(state, run / "last.pt")
                eval_s += time.perf_counter() - t
                metrics.write(json.dumps({"step": step, "dev_subset_wer": score,
                                          "dev_subset_utterances": len(subset),
                                          "best_dev_subset_wer": best if math.isfinite(best) else None,
                                          "weight_fraction": fraction, "selectable": eligible}) + "\n")
                metrics.flush()
                best_text = f"{100 * best:.2f}%" if math.isfinite(best) else "none yet"
                note = "" if eligible else f", not selectable at f={fraction:.4f}"
                print(f"eval step {step}: dev-subset WER {100 * score:.2f}% "
                      f"(best {best_text}{note})", flush=True)
                mark, mark_step = time.perf_counter(), step
            if step >= args.max_steps:
                break
        epoch += 1
    return {"train_time_s": time.perf_counter() - start, "dev_subset_eval_s": eval_s,
            "steps": step, "examples_seen": seen, "epochs_started": epoch,
            "steps_per_epoch": len(loader), "train_utterances": len(dataset)}


def finish(args, model, processor, run, config_sha: str) -> dict:
    """Reload best.pt, write the deployable artifact, check it, and score full dev-clean."""
    if not (run / "best.pt").exists():
        raise RuntimeError("no selectable (weight fraction 1) evaluation produced best.pt")
    best = torch.load(run / "best.pt", map_location="cpu")
    if not selectable(best.get("weight_fraction")):
        raise RuntimeError(f"best.pt was scored at weight fraction {best['weight_fraction']}")
    model.load_state_dict(best["state_dict"])
    model.eval()
    result = {"best_step": best["step"], "dev_subset_wer": best["dev_subset_wer"],
              "best_weight_fraction": best.get("weight_fraction")}
    if args.arm == "fp32":
        model.save_pretrained(run / "best-hf", safe_serialization=True)
        processor.save_pretrained(run / "best-hf")
        scorer = model
        source = decoding.describe_source("hf-dir", run / "best-hf",
                                          f"{args.run_name}: FP32 fine-tuned best checkpoint")
    else:
        quant.set_weight_fraction(model, 1.0)  # the export refuses f < 1
        extra = {"run_name": args.run_name, "arm": args.arm, "lr": args.lr,
                 "best_step": best["step"], "dev_subset_wer": best["dev_subset_wer"],
                 "config_sha256": config_sha, **checkpoint.base_provenance(),
                 "source_hashes": checkpoint.source_hashes()}
        manifest = export.export_model(model, run / "export", extra)
        scorer = export.load_export(run / "export", device="cuda")
        features, decoder_ids = data.reference_batch(processor, model.config)
        check = export.reconstruction_check(model, scorer, features, decoder_ids)
        data.write_json(run / "export" / "reconstruction.json", check)
        source = decoding.describe_source("export", run / "export",
                                          f"{args.run_name}: {args.arm} QAT, rebuilt from export")
        result.update(code_histogram=quant.code_histogram(model), bytes=manifest["bytes"],
                      reconstruction=check)
    records, info = decoding.decode(scorer, processor, data.build_manifest("dev-clean"), "cuda")
    evaluation = decoding.report(records, info, "dev-clean", source, None)
    data.write_json(run / "eval-dev-clean.json", evaluation)
    result["dev_clean_wer"] = evaluation["wer"]["wer"]
    return result


def main() -> None:
    args = parse_args()
    started_utc = time.strftime(UTC_FORMAT, time.gmtime())
    paths.require_mount()
    if not torch.cuda.is_available():
        sys.exit("CUDA is required")
    if (paths.RUNS / args.run_name / "summary.json").exists() and not args.overwrite:
        sys.exit(f"run {args.run_name} already has summary.json; pass --overwrite to replace it")
    run = paths.run_dir(args.run_name)
    clear_outputs(run)
    torch.manual_seed(args.seed)
    full = data.build_training_manifest(args.train_splits)
    split_counts = {s: sum(r["split"] == s for r in full) for s in args.train_splits}
    manifest = full[:args.limit_train]
    del full
    processor = checkpoint.load_processor()
    model = checkpoint.load_model("cpu")
    quantized = (quant.quantize_model(model, include_embedding=args.arm == "ternary-embed")
                 if args.arm != "fp32" else [])
    model.cuda()
    ramp = ramp_steps(args.quant_ramp_fraction, args.max_steps) if args.arm != "fp32" else None
    teacher = load_teacher() if args.distill_weight > 0 else None
    revision2 = {"train_splits": args.train_splits, "train_split_utterances": split_counts,
                 "quant_ramp_fraction": args.quant_ramp_fraction, "ramp_steps": ramp,
                 "distill_weight": args.distill_weight,
                 "distill_temperature": args.distill_temperature,
                 "teacher": teacher_source() if teacher is not None else None}
    config = {"args": vars(args), "arm": args.arm, "started_utc": started_utc, "git_head": git_head(),
              "source_hashes": checkpoint.source_hashes(), "model_dir": str(paths.MODEL_DIR),
              "model_revision": paths.MODEL_REVISION, "torch": torch.__version__,
              "transformers": transformers.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "quantized_modules": quantized,
              "parameter_accounting": quant.parameter_accounting(model),
              "train_utterances": len(manifest),
              "train_truncated_over_30s": sum(r["duration_s"] > data.MAX_SECONDS for r in manifest),
              **revision2}
    data.write_json(run / "config.json", config)
    print(f"{args.run_name}: arm {args.arm} lr {args.lr:g}, {len(quantized)} ternary modules, "
          f"{config['parameter_accounting']}; splits {split_counts} ({len(manifest)} used); "
          f"ramp steps {ramp}; distill w {args.distill_weight:g} T {args.distill_temperature:g}",
          flush=True)
    with open(run / "metrics.jsonl", "w") as metrics:
        stats = train(args, model, processor, manifest, run, metrics, teacher, ramp)
    del teacher
    torch.cuda.empty_cache()
    result = finish(args, model, processor, run, checkpoint.sha256(run / "config.json"))
    summary = {"run_name": args.run_name, "arm": args.arm, "lr": args.lr,
               "max_steps": args.max_steps, "seed": args.seed, "batch_size": args.batch_size,
               **result, **stats, **revision2,
               "parameter_accounting": config["parameter_accounting"],
               "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "started_utc": started_utc, "finished_utc": time.strftime(UTC_FORMAT, time.gmtime())}
    wrong = [k for k, t in SUMMARY_TYPES.items() if type(summary.get(k)) is not t]
    if wrong:
        raise TypeError(f"summary.json fields missing or mistyped: {wrong}")
    data.write_json(run / "summary.json", summary)  # last write; temp file + os.replace
    print(f"{args.run_name}: best step {result['best_step']}, dev-subset WER "
          f"{100 * result['dev_subset_wer']:.2f}%, dev-clean WER "
          f"{100 * result['dev_clean_wer']:.2f}%, train {stats['train_time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
