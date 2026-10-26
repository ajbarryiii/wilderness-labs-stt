"""Phase 1 (pilot) orchestration and the Phase 2 (main) launch, as in DESIGN.md "Arms and phases".

Pilot, one GPU job at a time:

1. recipes: P1, P2 and P3 at learning rate 5e-4, identical steps, schedule and stream.
2. select-recipe: the lowest full development mean WER (train.py's eval-dev.json, scored
   on the model rebuilt from the export of the FINAL checkpoint, step == max_steps at
   f == 1; train.py --select final, DESIGN.md "Selection").
3. lrs: the selected recipe at 2e-4 and 1e-3.
4. select: the lowest full development mean WER among the selected recipe's three
   learning rates; writes RUNS/pilot-selection.json, including the stop rule: if every
   pilot run's development mean WER exceeds 3x the FP32 original's (B0,
   EVAL/B0-dev/summary.json, validated before the first run), the selection says stop
   and the main phase refuses.

Main: `--phase main --max-steps N` launches M1 with the selected recipe and learning
rate; N must already be recorded in DESIGN.md (the design fixes it before launch).

Resumability and reuse. A run is reused or resumed only if it was made under this
protocol: summary.json and config.json must match arm, recipe, lr, steps, seed, batch
seconds, gradient checkpointing, evaluation interval, selection policy, stream kind,
phase and configuration hash, the recipe flag values (P1: no encoder matching, frozen
prediction/joint; P2: matching, frozen; P3: matching, trainable; A1: not quantized; M1:
per its recipe), the quantized module set (same hash as the other pilot runs), and the
SHA-256 of every source that affects training or scoring (TRAINING_SOURCES, including
the reused whisper-ternary quant.py and wer.py) as they are now, at the start, at every
resume recorded in resumes.jsonl (whose arguments must match too) and at finishing. Any mismatch stops the sweep with the reason; nothing is reused,
replaced or restarted silently. An unfinished run is relaunched with the same command
and train.py resumes it from its newest checkpoint (it refuses, exit 77, if checkpoint
files exist but none loads; this script never passes --overwrite). Exit 75 (the stream
failed after its own retries; train.py checkpointed) is retried with exponential backoff
up to MAX_STREAM_RETRIES; exit 76 (SIGTERM/SIGINT, checkpointed), 77 and any other
failure stop the sweep. Rerun the same command to continue.

Memory isolation. The whole sweep runs inside ONE memory-capped ./heavy unit; its
train.py children inherit the unit's cgroup, so this script never calls ./heavy itself
and refuses to start outside a parakeet-* systemd unit (--allow-unconfined overrides,
for dry runs). From finetune/parakeet-ternary/:

    ./heavy pilot --mem-max 36G -- python sweep.py --phase pilot
    ./heavy main --mem-max 36G -- python sweep.py --phase main --max-steps N

Check `systemctl --user list-units 'parakeet-*'` first: one heavy job at a time. Stop
with `systemctl --user stop parakeet-pilot` (train.py checkpoints on SIGTERM).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import paths

PY = paths.HERE / "python"
TRAIN = paths.HERE / "train.py"
DESIGN = paths.HERE / "DESIGN.md"
B0_DEV_SUMMARY = paths.EVAL / "B0-dev" / "summary.json"
SELECTION = paths.RUNS / "pilot-selection.json"

# Fixed in Phase 0 from the measured throughput and memory (README.md "Training").
PILOT_STEPS = 12000  # ~2.77 h at 0.83 s/step + ~7 min overhead ~= 3 GPU-hours (README)
BATCH_SECONDS = 600.0
GRAD_CHECKPOINTING = True
SELECT = "final"  # DESIGN.md: selection at the final f == 1 checkpoint
PILOT_LR = "5e-4"
FOLLOWUP_LRS = ("2e-4", "1e-3")
RECIPES = ("P1", "P2", "P3")
STOP_FACTOR = 3.0

EXIT_STREAM_FAILURE, EXIT_INTERRUPTED, EXIT_UNREADABLE_CHECKPOINT = 75, 76, 77  # train.py
MAX_STREAM_RETRIES = 20
BACKOFF_S, MAX_BACKOFF_S = 60.0, 1800.0

# Mirrors train.py (kept here so the sweep does not import torch): every source module whose
# change can alter training or scoring, relative to finetune/. powerlog.py is excluded on
# purpose: it runs as a separate recorder process and cannot affect results.
TRAINING_SOURCES = ("parakeet-ternary/train.py", "parakeet-ternary/quant.py",
                    "parakeet-ternary/stream.py", "parakeet-ternary/teacher.py",
                    "parakeet-ternary/evaluate.py", "parakeet-ternary/export.py",
                    "parakeet-ternary/data.py", "parakeet-ternary/paths.py",
                    "whisper-ternary/quant.py", "whisper-ternary/wer.py")
# DESIGN.md "Distillation variants": flag values per arm (M1 and A1 follow their recipe).
RECIPE_FLAGS = {"P1": {"encoder_matching": False, "train_pred_joint": False},
                "P2": {"encoder_matching": True, "train_pred_joint": False},
                "P3": {"encoder_matching": True, "train_pred_joint": True}}
SUMMARY_TYPES = {"run_name": str, "arm": str, "recipe": str, "lr": float, "max_steps": int,
                 "steps": int, "select": str, "selected_step": int, "eval_every": int,
                 "grad_checkpointing": bool, "source_hashes": dict,
                 "dev_subset_mean_wer": float,
                 "dev_mean_wer": float, "dev_wer": dict, "seed": int, "batch_seconds": float,
                 "train": str, "stream": dict, "quantized": bool, "encoder_matching": bool,
                 "train_pred_joint": bool, "ramp_steps": int, "warmup_steps": int,
                 "audio_seconds_seen": float, "smoke": bool, "started_utc": str,
                 "finished_utc": str}


def log(msg: str) -> None:
    print(f"[sweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_sources() -> dict[str, str]:
    """SHA-256 of the training sources as they are on disk now (a missing file is an error)."""
    root = paths.HERE.parent
    return {name: sha256(root / name) for name in TRAINING_SOURCES}


def read_log_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    """(records, problems) of an append-only JSONL log. A torn last line (power cut during
    an append) is ignored with a warning; a damaged line elsewhere is a problem."""
    if not path.exists():
        return [], []
    lines = path.read_bytes().split(b"\n")
    records, problems = [], []
    for i, raw in enumerate(lines):
        if not raw.strip(b" \x00\r"):
            continue
        try:
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError("not an object")
            records.append(record)
        except ValueError as error:
            if all(not x.strip(b" \x00\r") for x in lines[i + 1:]):
                log(f"warning: ignoring a torn last line in {path} (train.py drops it on resume)")
            else:
                problems.append(f"{path.name} line {i + 1} damaged ({error})")
    return records, problems


def run_name(arm: str, lr: str, phase: str = "pilot", recipe: str | None = None) -> str:
    return f"{phase}-{arm}{'-' + recipe if recipe and recipe != arm else ''}-lr{lr}"


class Plan:
    """The pilot protocol's fixed settings (overridable only in tests)."""

    def __init__(self, steps: int = PILOT_STEPS, batch_seconds: float = BATCH_SECONDS,
                 grad_checkpointing: bool = GRAD_CHECKPOINTING, eval_every: int | None = None,
                 stream: str = "hub", select: str = SELECT) -> None:
        self.steps, self.batch_seconds = steps, batch_seconds
        self.grad_checkpointing, self.stream, self.select = grad_checkpointing, stream, select
        self.eval_every = eval_every or max(1, steps // 10)

    def eval_interval(self, steps: int) -> int:
        return self.eval_every if steps == self.steps else max(1, steps // 20)

    def describe(self) -> dict:
        return {"steps": self.steps, "batch_seconds": self.batch_seconds,
                "grad_checkpointing": self.grad_checkpointing, "eval_every": self.eval_every,
                "select": self.select, "stream": self.stream, "seed": paths.SEED}


def train_command(plan: Plan, arm: str, lr: str, name: str, train: str = "pilot",
                  recipe: str | None = None, steps: int | None = None) -> list[str]:
    steps = steps or plan.steps
    cmd = [str(PY), str(TRAIN), "--arm", arm, "--lr", lr, "--run-name", name, "--train", train,
           "--max-steps", str(steps), "--batch-seconds", f"{plan.batch_seconds:g}",
           "--eval-every", str(plan.eval_interval(steps)), "--select", plan.select,
           "--stream", plan.stream]
    if recipe and recipe != arm:
        cmd += ["--recipe", recipe]
    if plan.grad_checkpointing:
        cmd.append("--grad-checkpointing")
    return cmd


class Expected:
    """What a run of this protocol must have recorded."""

    def __init__(self, plan: Plan, *, name: str, arm: str, recipe: str, lr: str, steps: int,
                 train: str, reference: dict | None = None) -> None:
        self.plan, self.name, self.arm, self.recipe, self.lr = plan, name, arm, recipe, lr
        self.steps, self.train = steps, train
        # Values every compared run must share (from the first run of the comparison).
        self.reference = reference or {}

    def flags(self) -> dict:
        recipe = RECIPE_FLAGS[self.recipe]
        if self.arm in RECIPE_FLAGS and self.arm != self.recipe:
            raise ValueError(f"arm {self.arm} must use recipe {self.arm}")
        return {**recipe, "quantized": self.arm != "A1"}

    def args(self) -> dict:
        return {"arm": self.arm, "recipe": self.recipe, "lr": float(self.lr),
                "max_steps": self.steps, "train": self.train, "seed": paths.SEED,
                "batch_seconds": self.plan.batch_seconds,
                "grad_checkpointing": self.plan.grad_checkpointing,
                "eval_every": self.plan.eval_interval(self.steps), "select": self.plan.select,
                "stream": self.plan.stream, "smoke": False, "dev_limit": None,
                "probe_steps": None}


def compare(label: str, recorded: dict, wanted: dict) -> list[str]:
    return [f"{label} {key} {recorded.get(key)!r} != {value!r}" for key, value in wanted.items()
            if recorded.get(key) != value]


def check_sources(label: str, recorded: dict | None, now: dict[str, str]) -> list[str]:
    recorded = recorded or {}
    return [f"{label}: {name} changed since the run (recorded {str(recorded.get(name))[:12]}, "
            f"now {digest[:12]})" for name, digest in now.items() if recorded.get(name) != digest]


def validate_config(run: Path, expected: Expected, sources: dict[str, str]) -> list[str]:
    """Problems in config.json and resumes.jsonl of a started run (empty: consistent)."""
    config = read_json(run / "config.json")
    problems = compare("config", config.get("args", {}), expected.args())
    problems += compare("config recipe", config.get("recipe", {}), expected.flags())
    modules = (config.get("trainable") or {}).get("trainable_modules")
    want_modules = ["encoder", "decoder", "joint"] if expected.flags()["train_pred_joint"] \
        else ["encoder"]
    if modules != want_modules:
        problems.append(f"config trainable modules {modules} != {want_modules}")
    problems += check_sources("config", config.get("training_source_hashes"), sources)
    resumes, damaged = read_log_jsonl(run / "resumes.jsonl")
    problems += damaged
    for i, entry in enumerate(resumes):
        problems += check_sources(f"resume {i + 1}", entry.get("training_source_hashes"), sources)
        problems += compare(f"resume {i + 1} args", entry.get("args") or {}, expected.args())
    stream = config.get("stream") or {}
    for key in ("config_hash", "stream_py_sha256", "teacher_py_sha256"):
        if key in expected.reference.get("stream", {}) and \
                stream.get(key) != expected.reference["stream"][key]:
            problems.append(f"stream {key} {stream.get(key)!r} differs from the compared runs")
    want_q = expected.reference.get("quantized_modules_sha256")
    if want_q and config.get("quantized_modules_sha256") != want_q:
        problems.append("quantized module set differs from the compared runs")
    return problems


def validate_summary(summary: dict, expected: Expected, sources: dict[str, str]) -> list[str]:
    """Problems that forbid reusing a finished run for this protocol (empty: reusable)."""
    problems = [f"{k} missing or not {t.__name__}" for k, t in SUMMARY_TYPES.items()
                if type(summary.get(k)) is not t]
    args = expected.args()
    wants = {"run_name": expected.name, "arm": expected.arm, "recipe": expected.recipe,
             "max_steps": expected.steps, "steps": expected.steps, "seed": paths.SEED,
             "train": expected.train, "smoke": False, "dev_limit": None,
             "batch_seconds": args["batch_seconds"],
             "grad_checkpointing": args["grad_checkpointing"],
             "eval_every": args["eval_every"], "select": args["select"]}
    problems += compare("summary", summary, wants)
    problems += compare("summary recipe flag", summary, expected.flags())
    if summary.get("lr") != float(expected.lr):
        problems.append(f"summary lr {summary.get('lr')!r} != {expected.lr}")
    if expected.plan.select == "final" and summary.get("selected_step") != expected.steps:
        problems.append(f"selected_step {summary.get('selected_step')!r} is not the final step")
    recorded = summary.get("stream") or {}
    if recorded.get("kind") != expected.plan.stream or recorded.get("phase") != expected.train:
        problems.append(f"stream {recorded} is not {expected.plan.stream}/{expected.train}")
    for key in ("config_hash", "stream_py_sha256", "teacher_py_sha256"):
        reference = expected.reference.get("stream", {})
        if key in reference and recorded.get(key) != reference[key]:
            problems.append(f"stream {key} {recorded.get(key)!r} differs from the other compared "
                            f"runs ({reference[key]!r}): data order or labels changed")
    want_q = expected.reference.get("quantized_modules_sha256")
    if want_q and summary.get("quantized_modules_sha256") != want_q:
        problems.append("quantized module set differs from the compared runs")
    problems += check_sources("summary", summary.get("source_hashes"), sources)
    return problems


def validate_run(run: Path, expected: Expected, sources: dict[str, str] | None = None) -> dict:
    sources = current_sources() if sources is None else sources
    summary = read_json(run / "summary.json")
    problems = validate_summary(summary, expected, sources)
    required = [run / "eval-dev.json", run / "config.json"]
    if summary.get("quantized", True):
        required += [run / "export" / "manifest.json", run / "export" / "export.safetensors"]
    else:
        required.append(run / "best.nemo")
    problems += [f"missing {p}" for p in required if not p.is_file()]
    if (run / "config.json").is_file():
        problems += validate_config(run, expected, sources)
    if (run / "eval-dev.json").is_file():
        dev = read_json(run / "eval-dev.json")
        if set(dev.get("dev_wer", {})) != set(paths.DEV_SETS) or dev.get("limit") is not None:
            problems.append(f"eval-dev.json is not the full {paths.DEV_SETS}")
        if dev.get("selected_step") != summary.get("selected_step"):
            problems.append("eval-dev.json scored a different step than summary.json selected")
    if problems:
        raise RuntimeError(f"run {run} cannot be reused: {'; '.join(problems)}. Rename or remove "
                           "it, or rerun train.py with --overwrite deliberately.")
    return summary


def run_train(cmd: list[str], dry: bool, sleep=time.sleep, runner=subprocess.run) -> None:
    """Run train.py to completion: retry stream failures (exit 75) with backoff; stop on
    anything else. Every retry resumes from train.py's newest checkpoint."""
    log(" ".join(cmd))
    if dry:
        return
    retries, delay = 0, BACKOFF_S
    while True:
        code = runner(cmd).returncode
        if code == 0:
            return
        if code == EXIT_STREAM_FAILURE and retries < MAX_STREAM_RETRIES:
            retries += 1
            log(f"stream failure (exit 75), checkpointed; retry {retries}/{MAX_STREAM_RETRIES} "
                f"in {delay:.0f}s (resumes from the checkpoint)")
            sleep(delay)
            delay = min(delay * 2, MAX_BACKOFF_S)
            continue
        if code == EXIT_STREAM_FAILURE:
            raise RuntimeError(f"stream failed {retries + 1} times; giving up (checkpoint kept, "
                               "rerun the sweep to resume)")
        if code == EXIT_INTERRUPTED:
            raise RuntimeError("train.py was interrupted (checkpoint kept); rerun to resume")
        if code == EXIT_UNREADABLE_CHECKPOINT:
            raise RuntimeError("train.py found checkpoint files but none is readable; inspect the "
                               "run's ckpt/ (the sweep never discards progress)")
        raise RuntimeError(f"train.py exited {code}: {' '.join(cmd)}")


def ensure_run(plan: Plan, arm: str, lr: str, dry: bool, *, train: str = "pilot",
               recipe: str | None = None, steps: int | None = None, phase: str = "pilot",
               reference: dict | None = None) -> dict | None:
    recipe = recipe or arm
    steps = steps or plan.steps
    name = run_name(arm, lr, phase, recipe)
    run = paths.RUNS / name
    expected = Expected(plan, name=name, arm=arm, recipe=recipe, lr=lr, steps=steps,
                        train=train, reference=reference)
    sources = current_sources()
    if (run / "summary.json").exists():
        summary = validate_run(run, expected, sources)
        log(f"complete and matches, reuse: {name}")
        return summary
    if (run / "config.json").exists():
        problems = validate_config(run, expected, sources)
        if problems:
            raise RuntimeError(f"run {run} was started under different settings or code: "
                               f"{'; '.join(problems)}. Rename or remove it deliberately.")
    if (run / "ckpt" / "latest.pt").exists() or (run / "ckpt" / "previous.pt").exists():
        log(f"{name}: checkpoint present, train.py resumes it")
    elif (run / "config.json").exists():
        log(f"{name}: started before but no checkpoint was written; train.py starts it at step 0")
    run_train(train_command(plan, arm, lr, name, train, recipe, steps), dry)
    if dry:
        return None
    return validate_run(run, expected, sources)


def candidate(summary: dict) -> dict:
    return {k: summary.get(k) for k in (
        "run_name", "arm", "recipe", "lr", "dev_mean_wer", "dev_wer", "dev_subset_mean_wer",
        "select", "selected_step", "best_interim_subset_step", "best_interim_subset_mean_wer",
        "audio_seconds_seen")}


def b0_dev_mean() -> dict:
    """The FP32 original's full development mean WER (fraction) from evaluate.py's summary.
    Raises unless it is a full, unlimited evaluation of the pretrained model on every dev set."""
    if not B0_DEV_SUMMARY.exists():
        raise FileNotFoundError(f"{B0_DEV_SUMMARY} missing: run evaluate.py --source pretrained "
                                "--sets dev --out-dir EVAL/B0-dev first (the stop rule needs it)")
    summary = read_json(B0_DEV_SUMMARY)
    sets = summary.get("sets", {})
    if summary.get("source", {}).get("kind") != "pretrained":
        raise ValueError(f"{B0_DEV_SUMMARY} is not an evaluation of the pretrained FP32 model")
    missing = [n for n in paths.DEV_SETS if n not in sets or sets[n].get("limit") is not None
               or not isinstance(sets[n].get("wer"), (int, float))]
    if missing:
        raise ValueError(f"{B0_DEV_SUMMARY} lacks full (unlimited) results for {missing}")
    per_set = {n: sets[n]["wer"] / 100 for n in paths.DEV_SETS}  # evaluate.py reports percent
    return {"dev_wer": per_set, "dev_mean_wer": sum(per_set.values()) / len(per_set),
            "path": str(B0_DEV_SUMMARY), "sha256": sha256(B0_DEV_SUMMARY)}


def reference_of(summary: dict) -> dict:
    return {"stream": summary["stream"],
            "quantized_modules_sha256": summary.get("quantized_modules_sha256")}


def phase_pilot(plan: Plan, dry: bool) -> dict | None:
    try:
        b0 = b0_dev_mean()  # validated before any GPU time is spent
        log(f"B0 full dev mean WER {100 * b0['dev_mean_wer']:.2f}% ({b0['path']})")
    except (FileNotFoundError, ValueError) as error:
        if not dry:
            raise SystemExit(f"refusing to start the pilot: {error}")
        log(f"dry run: {error}")
        b0 = None
    stage1, reference = [], None
    for recipe in RECIPES:
        summary = ensure_run(plan, recipe, PILOT_LR, dry, reference=reference)
        if summary is not None:
            stage1.append(summary)
            reference = reference or reference_of(summary)
    if dry and len(stage1) < len(RECIPES):
        best_recipe = "P2"
        log(f"dry run: assuming recipe {best_recipe} for the learning-rate stage")
    else:
        best = min(stage1, key=lambda s: s["dev_mean_wer"])
        best_recipe = best["recipe"]
        log(f"recipe stage: {[(s['recipe'], round(100 * s['dev_mean_wer'], 2)) for s in stage1]} "
            f"-> {best_recipe}")
    stage2 = [s for s in stage1 if s["recipe"] == best_recipe]
    for lr in FOLLOWUP_LRS:
        summary = ensure_run(plan, best_recipe, lr, dry, reference=reference)
        if summary is not None:
            stage2.append(summary)
    if dry:
        return None
    selected = min(stage2, key=lambda s: s["dev_mean_wer"])
    threshold = STOP_FACTOR * b0["dev_mean_wer"]
    every = stage1 + [s for s in stage2 if s not in stage1]
    stop = all(s["dev_mean_wer"] > threshold for s in every)
    selection = {
        "recipe": selected["recipe"], "lr": next(lr for lr in (PILOT_LR, *FOLLOWUP_LRS)
                                                 if float(lr) == selected["lr"]),
        "run": selected["run_name"], "dev_mean_wer": selected["dev_mean_wer"],
        "criterion": "lowest full development mean WER (unweighted over DEV_SETS) of the final "
                     "(step == max_steps, f == 1) model, scored on the model rebuilt from its export",
        "recipe_stage": [candidate(s) for s in stage1],
        "lr_stage": [candidate(s) for s in stage2],
        "b0": b0, "stop_rule": {"factor": STOP_FACTOR, "threshold_dev_mean_wer": threshold,
                                "every_pilot_run_above": stop},
        "protocol": plan.describe(), "reference": reference,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(SELECTION, selection)
    log(f"selected {selection['recipe']} at lr {selection['lr']} (dev mean WER "
        f"{100 * selected['dev_mean_wer']:.2f}%); B0 {100 * b0['dev_mean_wer']:.2f}%, stop rule "
        f"{'TRIGGERED' if stop else 'not triggered'}; wrote {SELECTION}")
    if stop:
        raise SystemExit("stop rule: every pilot run is above 3x the FP32 development mean WER; "
                         "reassess before spending the main budget (DESIGN.md Phase 1)")
    return selection


def design_records_steps(steps: int, text: str | None = None) -> bool:
    """DESIGN.md must state the M1 step count before launch (with or without separators)."""
    text = DESIGN.read_text() if text is None else text
    forms = {str(steps), f"{steps:,}", f"{steps:,}".replace(",", " ")}
    return any(re.search(rf"(?<![\d,]){re.escape(f)}(?![\d,])", text) for f in forms) and \
        "M1" in text


def phase_main(plan: Plan, steps: int | None, dry: bool, ignore_stop_rule: bool) -> None:
    if not steps:
        raise SystemExit("--phase main requires --max-steps (recorded in DESIGN.md first)")
    if not design_records_steps(steps):
        raise SystemExit(f"DESIGN.md does not record the M1 step count {steps}; record it first")
    if not SELECTION.exists():
        raise SystemExit(f"{SELECTION} missing; run the pilot first")
    selection = read_json(SELECTION)
    if selection["stop_rule"]["every_pilot_run_above"] and not ignore_stop_rule:
        raise SystemExit("the pilot triggered the stop rule; pass --ignore-stop-rule only after "
                         "the reassessment is recorded in DESIGN.md")
    reference = {"quantized_modules_sha256":
                 (selection.get("reference") or {}).get("quantized_modules_sha256")}
    ensure_run(plan, "M1", selection["lr"], dry, train="main", recipe=selection["recipe"],
               steps=steps, phase="main", reference=reference)


def heavy_unit(cgroup_text: str | None = None) -> str | None:
    """The parakeet-* systemd unit (./heavy) this process runs in, from /proc/self/cgroup."""
    if cgroup_text is None:
        try:
            cgroup_text = Path("/proc/self/cgroup").read_text()
        except OSError:
            return None
    for part in cgroup_text.replace("\n", "/").split("/"):
        if part.startswith("parakeet-") and part.endswith(".service"):
            return part
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--max-steps", type=int, help="M1 step count (main phase only)")
    parser.add_argument("--ignore-stop-rule", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-unconfined", action="store_true",
                        help="run outside a ./heavy systemd unit (dry runs only, normally)")
    args = parser.parse_args(argv)
    paths.require_mount()
    unit = heavy_unit()
    if unit is None and not args.allow_unconfined:
        raise SystemExit("sweep.py must run inside a memory-capped ./heavy unit, e.g.\n"
                         "  ./heavy pilot --mem-max 36G -- python sweep.py --phase pilot\n"
                         "(pass --allow-unconfined to override)")
    log(f"systemd unit: {unit or 'none (unconfined)'}")
    plan = Plan()
    log(f"phase {args.phase}; protocol {plan.describe()}")
    if args.phase == "pilot":
        phase_pilot(plan, args.dry_run)
    else:
        phase_main(plan, args.max_steps, args.dry_run, args.ignore_stop_rule)
    log("done")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        log(f"stopped: {error}")
        sys.exit(1)
