"""Orchestration of the binary pilot: gate, zero-step points, arms, selection and report. See plans/pilot.md.

One GPU job at a time, in this order (plans/pilot.md "Procedure"):

1. G-B (binary_eval.py --point G-B). Every phase starts here: if the gate did not pass, nothing
   else runs ("otherwise the init or ramp is wrong and nothing runs").
2. Z-A, Z-B (binary_eval.py): zero-step points, recorded for completeness.
3. Arms A, B5, B2, C (binary_train.py, train.py --arm P2): 12,000 steps, 600 s batches,
   gradient checkpointing, warmup 2% (240 steps), ramp 25% (3,000 steps; C has none and stays at
   f = 1), linear decay, dev-subset evaluation every 1,200 steps, the pilot Hub stream, selection
   on the final checkpoint (step 12,000, f = 1), full dev scored on the model rebuilt from its
   export.
4. decide(): the preregistered selection and reporting rules, written with every arm's report
   to RUNS/pilot-selection.json (binary RUNS). If the stop rule triggers, the file says so and
   the sweep exits nonzero.

Selection and reporting (plans/pilot.md "Selection and reporting", fixed before any run).
Metric: unweighted full-dev mean WER over paths.DEV_SETS of the final checkpoint, scored on the
model rebuilt from its export. References are read from their recorded files (exact values,
not the rounded numbers in the plan): B0 = parakeet-ternary EVAL/B0-dev/summary.json (5.574%),
the ternary pilot = RUNS/pilot-P2-lr5e-4 (6.850%), M1 = RUNS/main-M1-P2-lr5e-4 (6.086%).
- Stop rule: every candidate arm (A, B5, B2; C is a control) above STOP_FACTOR x B0.
- Starting point: B is the better of B5 and B2; |A - B| < 0.15 points is a tie and A wins;
  otherwise the lower of A and B wins.
- Binary cost: A - ternary pilot, B2 - C, B - M1, each with every per-set difference.
- Continuation control: C - M1; "M1 still improving" only if C is below M1 by more than 0.15
  points, "tie" within 0.15, else "worse than M1".
- Recommendation (to the user, not a decision): "stop" if the stop rule triggered; else a
  "binary main run" with the winning start if the winner is at most 1.5 points above the
  ternary pilot; else a "mixed pilot" (feed-forward binary, attention and pointwise ternary).
- Per arm: per-set and mean full dev, the dev-subset curve, code balance overall and per
  family, the fraction of latent signs changed since the arm's start, export size, wall time,
  training wall time (recorded by binary_train's BinaryContext in every checkpoint) and stream
  waits (from metrics.jsonl's per-step accumulators).

A finished arm or zero-step point is accepted only with an exact reconstruction record (codes
and scales) and with every model decoded in binary_paths.DECODER_GRAPHS_MODE (review round 1).

Reuse and resume follow parakeet-ternary/sweep.py (its validation is imported): a finished run
is reused only if summary.json, config.json and every resumes.jsonl entry match this protocol
(arguments including the binary ones, recipe flags, the quantized module set and the stream
configuration of the ternary pilot, and the SHA-256 of every training source, ternary and
binary, as they are now); an unfinished run is relaunched with the same command and train.py
resumes it; nothing is replaced or restarted silently. A scored point is reused only if its
sources match and it is a full (unlimited) evaluation. Rerun the same command to continue.

Main phase (plans/main.md, MB1): --phase main requires RUNS/pilot-selection.json with proposal
"binary main run", start m1 and winner B2, made from pilot arms that still validate under the
current sources and that still give the same decision (require_selection); then the G-B gate;
then the one arm MB1 (binary_main.py: binary_train.py with pinned streams, --train main --seed
20261017, 250,000 steps, dev-subset evaluation every 12,500), validated like a pilot arm with
train "main", seed 20261017, the pinned MB1 stream hash and MB1's source set (the pilot's plus
binary_main.py and binary_main_paths.py); finally RUNS/main-report.json (numbers only, no
decision rule). The pilot arms keep their own source set, so they still validate.

Memory isolation: run the whole sweep in ONE memory-capped ./heavy unit (its children inherit
the cgroup); this script refuses to start outside a parakeet-* unit unless --allow-unconfined
(dry runs). From finetune/parakeet-binary/:

    ./heavy binary-pilot --mem-max 36G --runtime 16h -- python binary_sweep.py --phase pilot
    ./python binary_sweep.py --phase pilot --dry-run --allow-unconfined
    ./heavy binary-main --mem-max 36G --runtime 100h -- /mnt/hd/wilderness-labs-stt/parakeet-binary/launch-main.sh
    ./python binary_sweep.py --phase main --dry-run --allow-unconfined
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import binary_main_paths
import binary_paths
import paths
import sweep  # finetune/parakeet-ternary/sweep.py, unchanged: validation, run_train, B0

Plan = sweep.Plan  # the pilot protocol: 12,000 steps, 600 s batches, eval every 1,200, select final
PY = binary_paths.HERE / "python"
TRAIN = binary_paths.HERE / "binary_train.py"
EVAL_SCRIPT = binary_paths.HERE / "binary_eval.py"
STOP_FACTOR = sweep.STOP_FACTOR  # 3 x B0
TIE_POINTS = 0.15  # |A - B| below this is a tie, A preferred
CONTROL_TIE_POINTS = 0.15  # |C - M1| within this is a tie
RECOMMEND_WITHIN_POINTS = 1.5  # winner at most this far above the ternary pilot: propose a main run
POINTS = ("G-B", "Z-A", "Z-B")
TRAINING_SOURCES = tuple(sweep.TRAINING_SOURCES) + binary_paths.BINARY_SOURCES
EVAL_SOURCES = TRAINING_SOURCES + ("parakeet-binary/binary_eval.py",)
SELECTION_NAME = "pilot-selection.json"
MAIN_TRAINING_SOURCES = TRAINING_SOURCES + binary_main_paths.MAIN_FILES  # MB1 only
MAIN_REPORT_NAME = "main-report.json"


@dataclass(frozen=True)
class Arm:
    name: str
    init: str
    quantizer: str
    ramp_kind: str
    lr: str
    run_name: str
    candidate: bool  # a start-point candidate (C is a control)
    train: str = "pilot"  # stream phase
    seed: int | None = None  # None: the project seed paths.SEED (no --seed argument)
    script: str = "binary_train.py"


# plans/pilot.md "Arms", in run order.
ARMS = {arm.name: arm for arm in (
    Arm("A", "fp32", "binary", "fp32", "5e-4", "pilot-binA-fp32-lr5e-4", True),
    Arm("B5", "m1", "binary", "ternary", "5e-4", "pilot-binB-m1-lr5e-4", True),
    Arm("B2", "m1", "binary", "ternary", "2e-4", "pilot-binB-m1-lr2e-4", True),
    Arm("C", "m1", "ternary", "none", "2e-4", "pilot-terC-m1-lr2e-4", False))}

# plans/main.md "The run".
MAIN_ARM = Arm("MB1", "m1", "binary", "ternary", binary_main_paths.LR, binary_main_paths.RUN_NAME, False,
               train="main", seed=binary_main_paths.SEED, script="binary_main.py")


class MainPlan(sweep.Plan):
    """The main run's protocol: sweep.Plan with MB1's steps, evaluation interval and seed."""

    def describe(self) -> dict:
        return {**super().describe(), "seed": binary_main_paths.SEED}


def main_plan() -> MainPlan:
    return MainPlan(steps=binary_main_paths.STEPS, eval_every=binary_main_paths.EVAL_EVERY)


RULES = {
    "metric": "unweighted full-dev mean WER over paths.DEV_SETS of the final checkpoint (step 12,000, f = 1), "
              "scored on the model rebuilt from its export",
    "stop_rule": f"every candidate arm (A, B5, B2) above {STOP_FACTOR:g} x B0's full-dev mean: stop and report; "
                 "no follow-up without reassessment",
    "starting_point": f"B = the better of B5 and B2; |A - B| < {TIE_POINTS} points is a tie and A is preferred "
                      "(standalone recipe); otherwise the lower one wins",
    "binary_cost": "A - ternary pilot P2 (same data and steps), B2 - C (same start, data, steps and learning "
                   "rate), B - M1 (its starting point), with every per-set number",
    "continuation_control": f"C - M1: 'M1 still improving' only if C is below M1 by more than {CONTROL_TIE_POINTS} "
                            "points; within that a tie",
    "recommendation": f"to the user, not a decision: winner within {RECOMMEND_WITHIN_POINTS} points of the ternary "
                      "pilot: propose a binary main run with the winning start; otherwise a mixed pilot "
                      "(feed-forward binary, attention and pointwise convolutions ternary)",
}


def log(msg: str) -> None:
    print(f"[binary_sweep {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def current_sources(names: tuple[str, ...] = TRAINING_SOURCES) -> dict[str, str]:
    root = paths.REPO / "finetune"
    return {name: sweep.sha256(root / name) for name in names}


# ----------------------------------------------------------------------------- commands

def train_command(plan: sweep.Plan, arm: Arm) -> list[str]:
    cmd = [str(PY), str(binary_paths.HERE / arm.script), "--init", arm.init, "--quantizer", arm.quantizer,
           "--arm", "P2", "--lr", arm.lr, "--run-name", arm.run_name, "--train", arm.train,
           "--max-steps", str(plan.steps)]
    if arm.seed is not None:
        cmd += ["--seed", str(arm.seed)]
    cmd += ["--batch-seconds", f"{plan.batch_seconds:g}", "--eval-every", str(plan.eval_interval(plan.steps)),
            "--select", plan.select, "--stream", plan.stream]
    if plan.grad_checkpointing:
        cmd.append("--grad-checkpointing")
    return cmd


def eval_command(point: str) -> list[str]:
    return [str(PY), str(EVAL_SCRIPT), "--point", point]


# ----------------------------------------------------------------------------- validation

class BinaryExpected(sweep.Expected):
    """sweep.Expected for a binary arm (pilot arm or MB1): train.py arm/recipe P2 plus the binary
    arguments, the arm's stream phase and seed."""

    def __init__(self, plan: sweep.Plan, arm: Arm, reference: dict | None) -> None:
        super().__init__(plan, name=arm.run_name, arm="P2", recipe="P2", lr=arm.lr, steps=plan.steps,
                         train=arm.train, reference=reference)
        self.binary = arm
        self.seed = paths.SEED if arm.seed is None else arm.seed

    def args(self) -> dict:
        arm = self.binary
        return {**super().args(), "seed": self.seed, "run_name": arm.run_name, "init": arm.init,
                "quantizer": arm.quantizer, "ramp_kind": arm.ramp_kind,
                "init_sha256": binary_paths.init_sha256(arm.init),
                "quantizer_sha256": binary_paths.quantizer_sha256(arm.quantizer)}


def arm_sources(arm: Arm) -> tuple[str, ...]:
    """The training sources an arm recorded: MB1 also binary_main.py and binary_main_paths.py."""
    return MAIN_TRAINING_SOURCES if arm.script == "binary_main.py" else TRAINING_SOURCES


def exact_reconstruction(label: str, check) -> list[str]:
    """Problems unless the reconstruction record has exact codes and scales (review round 1)."""
    if not isinstance(check, dict):
        return [f"{label}: no reconstruction record"]
    return [f"{label}: reconstruction {key} is {check.get(key)!r}" for key in ("codes_exact", "scales_exact")
            if check.get(key) is not True]


def decoder_modes(label: str, graphs, required: tuple[str, ...]) -> list[str]:
    """Problems unless every model in `required` (and every other one recorded) decoded in
    binary_paths.DECODER_GRAPHS_MODE (review round 1)."""
    if not isinstance(graphs, dict):
        return [f"{label}: no decoder mode record"]
    mode = binary_paths.DECODER_GRAPHS_MODE
    problems = [f"{label}: {key} decoder mode not recorded" for key in required if graphs.get(key) is None]
    return problems + [f"{label}: {key} decoded in CUDA-graph mode {value!r}, not {mode!r}"
                       for key, value in graphs.items() if value is not None and value != mode]


ARM_DECODERS = ("teacher", "student", "rebuilt", "full-dev")  # "dev-subset": None if finished in a later session
POINT_DECODERS = {"G-B": ("student", "full-dev"), "Z-A": ("qat", "rebuilt", "full-dev"),
                  "Z-B": ("qat", "rebuilt", "full-dev")}


def validate_binary(summary: dict, expected: BinaryExpected) -> list[str]:
    """The arm's own summary fields: binary block, ramp steps, exact reconstruction, decoder
    modes, training time and the report entries."""
    arm, problems = expected.binary, []
    block = summary.get("binary") or {}
    problems += exact_reconstruction("summary", summary.get("reconstruction"))
    problems += decoder_modes("summary", block.get("decoder_graphs"), ARM_DECODERS)
    timing = block.get("timing")
    if not (isinstance(timing, dict) and isinstance(timing.get("train_wall_s"), (int, float))):
        problems.append("summary binary timing (training wall time) missing")
    problems += sweep.compare("summary binary", block, {"init": arm.init, "quantizer": arm.quantizer,
                                                        "ramp_kind": arm.ramp_kind})
    ramp = 0 if arm.ramp_kind == "none" else max(1, round(0.25 * expected.steps))
    if summary.get("ramp_steps") != ramp:
        problems.append(f"summary ramp_steps {summary.get('ramp_steps')!r} != {ramp}")
    if not isinstance(block.get("code_balance"), dict):
        problems.append(f"summary binary code_balance missing ({block.get('report_error')})")
    return problems


def validate_run(run: Path, expected: BinaryExpected, sources: dict[str, str]) -> dict:
    """sweep.validate_run, also for a seed other than paths.SEED (MB1): sweep.validate_summary
    compares the summary's seed with paths.SEED, so the seed (and the stream record's seed) is
    checked here and validate_summary runs on a copy with the seed replaced. The other checks
    are sweep.validate_run's (validate_config compares the expected arguments, which carry the
    arm's seed)."""
    if expected.seed == paths.SEED:
        return sweep.validate_run(run, expected, sources)
    summary = sweep.read_json(run / "summary.json")
    problems = [f"summary {label} {value!r} != {expected.seed}" for label, value in (
        ("seed", summary.get("seed")), ("stream seed", (summary.get("stream") or {}).get("seed")))
        if type(value) is not int or value != expected.seed]
    problems += sweep.validate_summary({**summary, "seed": paths.SEED}, expected, sources)
    required = [run / "eval-dev.json", run / "config.json", run / "export" / "manifest.json",
                run / "export" / "export.safetensors"]
    problems += [f"missing {p}" for p in required if not p.is_file()]
    if (run / "config.json").is_file():
        problems += sweep.validate_config(run, expected, sources)
    if (run / "eval-dev.json").is_file():
        dev = sweep.read_json(run / "eval-dev.json")
        if set(dev.get("dev_wer", {})) != set(paths.DEV_SETS) or dev.get("limit") is not None:
            problems.append(f"eval-dev.json is not the full {paths.DEV_SETS}")
        if dev.get("selected_step") != summary.get("selected_step"):
            problems.append("eval-dev.json scored a different step than summary.json selected")
    if problems:
        raise RuntimeError(f"run {run} cannot be reused: {'; '.join(problems)}. Rename or remove it deliberately.")
    return summary


def validate_arm(run: Path, expected: BinaryExpected, sources: dict[str, str]) -> dict:
    summary = validate_run(run, expected, sources)
    problems = validate_binary(summary, expected)
    if problems:
        raise RuntimeError(f"run {run} cannot be reused: {'; '.join(problems)}")
    return summary


def validate_point(point: str, summary: dict, sources: dict[str, str]) -> list[str]:
    problems = []
    if summary.get("point") != point:
        problems.append(f"point {summary.get('point')!r} != {point}")
    if summary.get("limit") is not None or set(summary.get("dev_wer", {})) != set(paths.DEV_SETS):
        problems.append(f"not a full evaluation of {paths.DEV_SETS}")
    problems += sweep.check_sources(point, summary.get("source_hashes"), sources)
    problems += decoder_modes(point, summary.get("decoder_graphs"), POINT_DECODERS[point])
    if point == "G-B" and "gate" not in summary:
        problems.append("no gate verdict")
    if point != "G-B":
        problems += exact_reconstruction(point, summary.get("reconstruction"))
    return problems


def arm_reference() -> dict:
    """What every arm must share with ternary pilot P2 at 5e-4 (and with each other): the stream
    configuration (data order), the teacher source and the quantized module set."""
    summary = sweep.read_json(binary_paths.TERNARY_PILOT_RUN / "summary.json")
    stream = summary["stream"]
    if stream["config_hash"] != binary_paths.STREAM_CONFIG_HASH["pilot"]:
        raise RuntimeError(f"ternary pilot stream hash {stream['config_hash']} != binary_paths' pinned value")
    return {"stream": {"config_hash": stream["config_hash"], "teacher_py_sha256": stream["teacher_py_sha256"]},
            "quantized_modules_sha256": summary["quantized_modules_sha256"]}


def main_reference() -> dict:
    """What MB1 must have recorded: the pilot's teacher and quantized module set, and the stream
    configuration hash pinned for --train main --seed 20261017 (binary_main_paths)."""
    reference = arm_reference()
    _, config_hash = binary_main_paths.stream_pin(MAIN_ARM.train, MAIN_ARM.seed)
    return {**reference, "stream": {**reference["stream"], "config_hash": config_hash}}


# ----------------------------------------------------------------------------- steps

def ensure_point(point: str, dry: bool) -> dict | None:
    out = binary_paths.EVAL / point
    sources = current_sources(EVAL_SOURCES)
    if (out / "summary.json").exists():
        summary = sweep.read_json(out / "summary.json")
        problems = validate_point(point, summary, sources)
        if problems:
            raise RuntimeError(f"{out} cannot be reused: {'; '.join(problems)}. Rescore deliberately with "
                               f"binary_eval.py --point {point} --overwrite.")
        log(f"scored and matches, reuse: {point}")
        return summary
    sweep.run_train(eval_command(point), dry)
    if dry:
        return None
    summary = sweep.read_json(out / "summary.json")
    problems = validate_point(point, summary, sources)
    if problems:
        raise RuntimeError(f"{point}: {'; '.join(problems)}")
    return summary


def require_gate(dry: bool) -> dict | None:
    gate = ensure_point("G-B", dry)
    if gate is None:
        log("dry run: G-B not scored yet; the real run stops here unless it passes")
        return None
    verdict = gate["gate"]
    log(f"G-B: {verdict['difference_points']:+.3f} points vs M1, ternary twin "
        f"{'equal' if verdict['ternary_twin_encoder_equal'] else 'DIFFERENT'}: "
        f"{'PASS' if verdict['pass'] else 'FAIL'}")
    if verdict["pass"] is not True:
        raise SystemExit("G-B gate failed: the init or the ramp is wrong; nothing runs (plans/pilot.md)")
    return gate


def ensure_arm(plan: sweep.Plan, arm: Arm, dry: bool, reference: dict) -> dict | None:
    run = binary_paths.RUNS / arm.run_name
    expected = BinaryExpected(plan, arm, reference)
    sources = arm_sources(arm)
    if (run / "summary.json").exists():
        summary = validate_arm(run, expected, current_sources(sources))
        log(f"complete and matches, reuse: {arm.name} {arm.run_name}")
        return summary
    if (run / "config.json").exists():
        problems = sweep.validate_config(run, expected, current_sources(sources))
        config = sweep.read_json(run / "config.json")
        problems += decoder_modes("config", (config.get("binary") or {}).get("decoder_graphs"),
                                  ("teacher", "student"))
        if problems:
            raise RuntimeError(f"run {run} was started under different settings or code: "
                               f"{'; '.join(problems)}. Rename or remove it deliberately.")
    if (run / "ckpt" / "latest.pt").exists() or (run / "ckpt" / "previous.pt").exists():
        log(f"{arm.run_name}: checkpoint present, train.py resumes it")
    elif (run / "config.json").exists():
        log(f"{arm.run_name}: started before but no checkpoint was written; train.py starts it at step 0")
    sweep.run_train(train_command(plan, arm), dry)
    if dry:
        return None
    return validate_arm(run, expected, current_sources(sources))


# ----------------------------------------------------------------------------- selection

def _diff(x: dict, y: dict) -> dict:
    return {"mean": x["dev_mean_wer"] - y["dev_mean_wer"],
            "per_set": {s: x["dev_wer"][s] - y["dev_wer"][s] for s in paths.DEV_SETS}}


def decide(results: dict[str, dict], references: dict) -> dict:
    """The preregistered rules on full-dev results {arm: {"dev_mean_wer", "dev_wer"}} (WER as
    fractions) with references {"b0", "ternary_pilot", "m1"} of the same shape."""
    tp, m1 = references["ternary_pilot"], references["m1"]
    threshold = STOP_FACTOR * references["b0"]["dev_mean_wer"]
    candidates = {name: results[name] for name, arm in ARMS.items() if arm.candidate}
    stop = all(r["dev_mean_wer"] > threshold for r in candidates.values())
    b_name = min(("B5", "B2"), key=lambda k: results[k]["dev_mean_wer"])
    a, b = results["A"], results[b_name]
    gap = a["dev_mean_wer"] - b["dev_mean_wer"]
    tie = abs(gap) < TIE_POINTS / 100
    winner = "A" if tie or gap < 0 else b_name
    control = results["C"]["dev_mean_wer"] - m1["dev_mean_wer"]
    reading = ("M1 still improving" if control < -CONTROL_TIE_POINTS / 100
               else "tie" if abs(control) <= CONTROL_TIE_POINTS / 100 else "worse than M1")
    limit = tp["dev_mean_wer"] + RECOMMEND_WITHIN_POINTS / 100
    winner_wer = results[winner]["dev_mean_wer"]
    if stop:
        proposal = "stop"
    elif winner_wer <= limit:
        proposal = "binary main run"
    else:
        proposal = "mixed pilot"
    return {
        "stop_rule": {"triggered": stop, "threshold_dev_mean_wer": threshold, "factor": STOP_FACTOR,
                      "candidates": {k: r["dev_mean_wer"] for k, r in candidates.items()}},
        "starting_point": {"B": b_name, "A_minus_B_points": 100 * gap, "tie": tie,
                           "tie_points": TIE_POINTS, "winner": winner,
                           "start": "fp32" if winner == "A" else "m1"},
        "binary_cost": {"A_minus_ternary_pilot": _diff(a, tp), "B2_minus_C": _diff(results["B2"], results["C"]),
                        "B_minus_M1": {**_diff(b, m1), "B": b_name}},
        "continuation_control": {"C_minus_M1": _diff(results["C"], m1), "reading": reading,
                                 "tie_points": CONTROL_TIE_POINTS},
        "recommendation": {"proposal": proposal, "start": winner, "winner_dev_mean_wer": winner_wer,
                           "limit_dev_mean_wer": limit, "within_points": RECOMMEND_WITHIN_POINTS},
    }


def _utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def arm_report(run: Path, summary: dict) -> dict:
    """Per-arm numbers for the report (plans/pilot.md "Reported per arm")."""
    records, _ = sweep.read_log_jsonl(run / "metrics.jsonl")
    curve = [{"step": r["step"], "dev_subset_mean_wer": r["dev_subset_mean_wer"],
              "dev_subset_wer": r["dev_subset_wer"], "weight_fraction": r.get("weight_fraction")}
             for r in records if r.get("eval") == "dev-subset"]
    # Stream waits: train.py's data_wait_s_per_step is its accumulator over the steps since the
    # previous metrics line divided by their number; the accumulator is checkpointed and restored,
    # and lines after a resumed checkpoint are pruned, so mean x step difference is the exact total.
    # (step_time_s is not usable that way: its timer restarts at evaluations, checkpoints and
    # resumes; the training wall time comes from the seam's record instead.)
    wait_s = 0.0
    windows = waiting = 0
    previous = 0
    for r in records:
        if "loss" not in r:
            continue
        wait_s += r["data_wait_s_per_step"] * (r["step"] - previous)
        previous = r["step"]
        windows += 1
        waiting += r["data_wait_s_per_step"] > 0.05
    block = summary.get("binary") or {}
    timing = block.get("timing") or {}
    return {"run_name": summary["run_name"], "lr": summary["lr"], "dev_wer": summary["dev_wer"],
            "dev_mean_wer": summary["dev_mean_wer"], "dev_subset_curve": curve,
            "code_balance": block.get("code_balance"), "latent_sign_change": block.get("latent_sign_change"),
            "export_bytes": (summary.get("export_bytes") or {}).get("file_bytes"),
            "packed_code_bytes": (summary.get("export_bytes") or {}).get("packed_code_bytes"),
            "wall_time_s": (_utc(summary["finished_utc"]) - _utc(summary["started_utc"])).total_seconds(),
            "train_wall_s": timing.get("train_wall_s"), "train_sessions": timing.get("sessions"),
            "decoder_graphs": block.get("decoder_graphs"), "stream_wait_s": wait_s,
            "logged_windows": windows, "windows_waiting_over_50ms_per_step": waiting,
            "resumes": summary.get("resumes"), "audio_hours_seen": summary.get("audio_hours_seen")}


def references() -> dict:
    def full_dev(path: Path) -> dict:
        s = sweep.read_json(path)
        return {"dev_mean_wer": s["dev_mean_wer"], "dev_wer": s["dev_wer"], "path": str(path),
                "sha256": sweep.sha256(path)}
    return {"b0": sweep.b0_dev_mean(), "ternary_pilot": full_dev(binary_paths.TERNARY_PILOT_RUN / "summary.json"),
            "m1": full_dev(binary_paths.M1_RUN / "summary.json")}


# ----------------------------------------------------------------------------- main run

def require_selection(path: Path | None = None) -> dict:
    """The pilot's selection, if it proposes the binary main run from M1 with winner B2 and is not
    stale: every pilot arm still validates under the current sources, and decide() on those arms
    and the current references gives exactly the recorded decision. SystemExit otherwise."""
    path = binary_paths.RUNS / SELECTION_NAME if path is None else Path(path)
    if not path.exists():
        raise SystemExit(f"{path} missing: run the pilot (--phase pilot) first; the main run needs its selection")
    selection = sweep.read_json(path)
    decision = selection.get("decision") or {}
    point, rec = decision.get("starting_point") or {}, decision.get("recommendation") or {}
    problems = []
    if rec.get("proposal") != "binary main run":
        problems.append(f"proposal {rec.get('proposal')!r} is not 'binary main run'")
    if point.get("start") != "m1":
        problems.append(f"start {point.get('start')!r} is not 'm1'")
    if point.get("winner") != "B2" or rec.get("start") != "B2":
        problems.append(f"winner {point.get('winner')!r} (recommended start {rec.get('start')!r}) is not B2")
    if problems:
        raise SystemExit(f"{path}: {'; '.join(problems)}. plans/main.md runs B2's recipe from M1 only if the "
                         "pilot proposed it; refusing")
    plan, reference, sources = sweep.Plan(), arm_reference(), current_sources(TRAINING_SOURCES)
    summaries = {}
    for arm in ARMS.values():
        try:
            summaries[arm.name] = validate_arm(binary_paths.RUNS / arm.run_name,
                                               BinaryExpected(plan, arm, reference), sources)
        except (RuntimeError, OSError, ValueError, KeyError) as error:
            raise SystemExit(f"{path} is stale: pilot arm {arm.name} no longer validates ({error}); refusing")
    recomputed = json.loads(json.dumps(decide(summaries, references())))
    if recomputed != decision:
        raise SystemExit(f"{path} is stale: the validated pilot arms and current references give a different "
                         "decision; rerun --phase pilot and inspect before the main run")
    log(f"pilot selection {path} matches its arms: proposal {rec['proposal']}, start {point['start']}, "
        f"winner {point['winner']}")
    return selection


def _full_dev(summary: dict, path: Path) -> dict:
    return {"dev_mean_wer": summary["dev_mean_wer"], "dev_wer": summary["dev_wer"], "path": str(path),
            "sha256": sweep.sha256(path)}


def main_report(summary: dict, refs: dict, gate: dict, plan: sweep.Plan, reference: dict) -> dict:
    """RUNS/main-report.json (plans/main.md "Reported"): MB1's numbers and its differences from M1,
    B0 and pilot B2, mean and per set. No threshold decides anything."""
    b2_path = binary_paths.RUNS / ARMS["B2"].run_name / "summary.json"
    b2 = _full_dev(sweep.read_json(b2_path), b2_path)
    selection = binary_paths.RUNS / SELECTION_NAME
    workers, config_hash = binary_main_paths.stream_pin(MAIN_ARM.train, MAIN_ARM.seed)
    return {
        "mb1": {**arm_report(binary_paths.RUNS / MAIN_ARM.run_name, summary), "init": MAIN_ARM.init,
                "quantizer": MAIN_ARM.quantizer, "ramp_kind": MAIN_ARM.ramp_kind, "seed": summary.get("seed"),
                "steps": summary.get("steps"), "stream": summary.get("stream")},
        "comparisons": {"MB1_minus_M1": _diff(summary, refs["m1"]), "MB1_minus_B0": _diff(summary, refs["b0"]),
                        "MB1_minus_pilot_B2": _diff(summary, b2)},
        "readings": {"MB1_minus_M1": "net change after binary conversion and 250,000 further steps from M1 (not a "
                                     "causal binary cost; see plans/main.md 'Reported')",
                     "MB1_minus_pilot_B2": "what the 20x longer run bought (includes a seed effect)"},
        "references": {**refs, "pilot_B2": b2}, "gate": gate["gate"],
        "pilot_selection": {"path": str(selection), "sha256": sweep.sha256(selection)},
        "protocol": plan.describe(), "arm_reference": reference,
        "stream_pin": {"train": MAIN_ARM.train, "seed": MAIN_ARM.seed, "workers": workers, "config_hash": config_hash},
        "note": "numbers only; no threshold decides anything (plans/main.md 'Reported')",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def phase_main(plan: sweep.Plan, dry: bool, refs: dict | None) -> dict | None:
    """The pilot selection check, the gate, MB1, then RUNS/main-report.json."""
    require_selection()
    gate = require_gate(dry)
    reference = main_reference()
    summary = ensure_arm(plan, MAIN_ARM, dry, reference)
    if dry:
        return None
    report = main_report(summary, refs, gate, plan, reference)
    out = binary_paths.RUNS / MAIN_REPORT_NAME
    sweep.write_json(out, report)
    diff = report["comparisons"]
    log(f"MB1 full dev {100 * summary['dev_mean_wer']:.3f}%: {100 * diff['MB1_minus_M1']['mean']:+.3f} points vs M1, "
        f"{100 * diff['MB1_minus_pilot_B2']['mean']:+.3f} vs pilot B2, {100 * diff['MB1_minus_B0']['mean']:+.3f} vs B0; "
        f"wrote {out}")
    return report


# ----------------------------------------------------------------------------- phases

def phase(plan: sweep.Plan, name: str, dry: bool) -> dict | None:
    """gate: G-B only; zero: G-B, Z-A, Z-B; pilot: everything, then the selection; main: the
    selection check, G-B, MB1 and its report."""
    try:
        refs = references()  # validated before any GPU time is spent
        log(f"references: B0 {100 * refs['b0']['dev_mean_wer']:.3f}%, ternary pilot "
            f"{100 * refs['ternary_pilot']['dev_mean_wer']:.3f}%, M1 {100 * refs['m1']['dev_mean_wer']:.3f}%")
    except (FileNotFoundError, ValueError, KeyError) as error:
        if not dry:
            raise SystemExit(f"refusing to start: {error}")
        log(f"dry run: {error}")
        refs = None
    if name == "main":
        return phase_main(plan, dry, refs)
    gate = require_gate(dry)
    if name == "gate":
        return gate
    zero = {point: ensure_point(point, dry) for point in POINTS if point != "G-B"}
    if name == "zero":
        return zero
    reference = arm_reference()
    summaries = {}
    for arm in ARMS.values():
        summary = ensure_arm(plan, arm, dry, reference)
        if summary is not None:
            summaries[arm.name] = summary
    if dry:
        return None
    decision = decide(summaries, refs)
    selection = {
        "decision": decision, "rules": RULES,
        "arms": {name: {**arm_report(binary_paths.RUNS / ARMS[name].run_name, s),
                        "init": ARMS[name].init, "quantizer": ARMS[name].quantizer,
                        "ramp_kind": ARMS[name].ramp_kind, "candidate": ARMS[name].candidate}
                 for name, s in summaries.items()},
        "zero_step": {point: {k: s.get(k) for k in ("dev_wer", "dev_mean_wer", "code_histogram",
                                                     "export_bytes", "reconstruction")}
                      for point, s in zero.items()},
        "gate": gate["gate"], "references": refs, "protocol": plan.describe(), "arm_reference": reference,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = binary_paths.RUNS / SELECTION_NAME
    sweep.write_json(out, selection)
    sp, rec = decision["starting_point"], decision["recommendation"]
    log(f"B = {sp['B']}; A - B = {sp['A_minus_B_points']:+.2f} points ({'tie' if sp['tie'] else 'no tie'}); "
        f"winner {sp['winner']}; continuation control: {decision['continuation_control']['reading']}; "
        f"proposal: {rec['proposal']}; wrote {out}")
    if decision["stop_rule"]["triggered"]:
        raise SystemExit(f"stop rule: every candidate arm is above {STOP_FACTOR:g} x B0; reassess before any "
                         "follow-up (plans/pilot.md)")
    return selection


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("gate", "zero", "pilot", "main"), default="pilot")
    parser.add_argument("--dry-run", action="store_true", help="print the commands; run and write nothing")
    parser.add_argument("--allow-unconfined", action="store_true",
                        help="run outside a ./heavy systemd unit (dry runs only, normally)")
    args = parser.parse_args(argv)
    binary_paths.require_mount()
    unit = sweep.heavy_unit()
    if unit is None and not args.allow_unconfined:
        raise SystemExit("binary_sweep.py must run inside a memory-capped ./heavy unit, e.g.\n"
                         "  ./heavy binary-pilot --mem-max 36G --runtime 16h -- python binary_sweep.py --phase pilot\n"
                         "(pass --allow-unconfined to override)")
    log(f"systemd unit: {unit or 'none (unconfined)'}")
    plan = main_plan() if args.phase == "main" else sweep.Plan()
    arms = [MAIN_ARM.name] if args.phase == "main" else list(ARMS)
    log(f"phase {args.phase}; protocol {plan.describe()}; arms {arms}")
    phase(plan, args.phase, args.dry_run)
    log("done")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        log(f"stopped: {error}")
        sys.exit(1)
