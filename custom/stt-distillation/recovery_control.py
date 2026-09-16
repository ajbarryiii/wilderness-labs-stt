"""Launch and supervise the eight-hour recovery experiment on a single GPU."""

import argparse
import datetime
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path

from common import ART, REPO, digest, save, storage
from recovery_core import (
    ROOT,
    SOURCE_RUN,
    artifact,
    nested_subsets,
    read,
    source_hashes,
    write_tape,
)

HERE = Path(__file__).resolve().parent
STOP = False


def halt(*_):
    global STOP
    STOP = True


def utc(ts=None):
    return datetime.datetime.fromtimestamp(
        time.time() if ts is None else ts, datetime.timezone.utc
    ).isoformat()


def report(run):
    state = read(run / "state.json")
    results = {}
    lines = [
        "# Eight-hour training recovery experiment",
        "",
        f"Status: {state['status']}; stage: {state['stage']}.",
        f"Started: {state['started_utc']}. Deadline: {state['deadline_utc']}.",
        "",
        "| Job | Status | Presented examples | General CER / WER | Symptoms CER / WER | Digits exact |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for spec_path in sorted((run / "jobs").glob("*.json")):
        name = spec_path.stem
        path = run / "training" / name / "result.json"
        if path.exists():
            r = read(path)
            results[name] = {k: v for k, v in r.items() if k != "final"}
            metric = (r.get("final") or {}).get("development", {}).get("metrics")
            if metric:
                g, m, d = (metric[x] for x in ["general", "medical_symptoms", "digits"])
                lines.append(
                    f"| {name} | {r['status']} | {r['presented']} | {g['cer']:.2%} / {g['wer']:.2%} | {m['cer']:.2%} / {m['wer']:.2%} | {d['exact']}/{d['total']} |"
                )
        else:
            lines.append(f"| {name} | pending/running | — | — | — | — |")
    lines += [
        "",
        "Table shows last evaluated weights. Selection uses equally weighted general/symptom CER; see each best.json for the selected checkpoint.",
        "Existing development is reused validation. Medical terminology, natural quantities and field-device energy remain unqualified.",
        "Duration is an eight-hour ceiling, including evaluation/reporting. Missing independent seeds or full exposure are reported as incomplete.",
        "",
    ]
    if state.get("error"):
        lines += ["Error: " + state["error"]]
    save(run / "report.json", dict(state=state, jobs=results))
    (run / "REPORT.md").write_text("\n".join(lines))


def add_job(run, name, seed, variant, gate_checkpoint=None, gate=False):
    manifest = read(run / "manifest.json")["rows"]
    ids = set(read(run / "subsets.json")["gate"])
    rows = [
        r for r in manifest if r["split"] == "train" and (not gate or r["id"] in ids)
    ]
    if gate:
        by_id = {r["id"]: r for r in rows}
        rows = [by_id[i] for i in read(run / "subsets.json")["gate"]]
    path = run / "tapes" / f"{'gate' if gate else 'broad'}-{seed}.npy"
    if not path.exists():
        path = write_tape(
            run,
            rows,
            seed,
            48000 if gate else 108000,
            "gate" if gate else "broad",
            gate,
        )
    spec = dict(
        name=name,
        variant=variant,
        seed=seed,
        gate=gate,
        precision="fp",
        row_ids=[r["id"] for r in rows],
        tape=str(path),
        tape_sha256=digest(path),
        encoder_lr=1e-5,
        head_lr=1e-5,
        effective_batch=4,
        gains=[0, -6, -12, -18, -24],
    )
    if variant == "lower_encoder_lr":
        spec["encoder_lr"] = 3e-6
    elif variant == "batch16":
        spec["effective_batch"] = 16
    elif variant == "mild_gain":
        spec["gains"] = [0, -3, -6, -9, -12]
    elif variant == "character_weighting":
        spec["loss_reduction"] = "total_characters"
    if gate_checkpoint:
        spec.update(
            initial_checkpoint=str(gate_checkpoint),
            initial_checkpoint_sha256=digest(gate_checkpoint),
        )
    save(run / "jobs" / f"{name}.json", spec)
    return spec


def prepare(root=None, smoke=False):
    storage()
    run = artifact(
        root
        or ROOT
        / "runs"
        / datetime.datetime.now(datetime.timezone.utc).strftime(
            "recovery-8h-%Y%m%dT%H%M%SZ"
        )
    )
    run.mkdir(parents=True, exist_ok=False)
    cfg = read(HERE / "recovery8.json")
    if smoke:
        cfg["model"] = dict(
            cfg["model"], width=32, depth=2, heads=4, context=16, checkpoint=False
        )
        cfg.update(
            eval_every_examples=16,
            checkpoint_every_examples=16,
            gradient_probe_at=[],
            collapse_grace=12000,
        )
    save(run / "config.json", cfg)
    for name in ["manifest.json", "subsets.json"]:
        shutil.copy2(SOURCE_RUN / name, run / name)
    rows = [r for r in read(run / "manifest.json")["rows"] if r["split"] == "train"]
    subsets = read(run / "subsets.json")
    save(
        run / "representative-subsets.json",
        nested_subsets(rows, subsets["gate"], cfg["seed"]),
    )
    (run / "code").mkdir()
    for name in source_hashes(HERE):
        shutil.copy2(HERE / name, run / "code" / name)
    for name in ["python", "decoder-python"]:
        if (run / "code" / name).exists():
            (run / "code" / name).chmod(0o755)
    for name, variant in [
        ("R0", "control"),
        ("R1", "warmstart"),
        ("R2", "lower_encoder_lr"),
        ("R4", "mild_gain"),
        ("R3", "batch16"),
    ]:
        initial = (
            SOURCE_RUN / "gate/fp_control/latest.pt"
            if variant == "warmstart" and not smoke
            else None
        )
        add_job(run, name, cfg["seed"], variant, initial)
    save(
        run / "provenance.json",
        dict(
            source_run=str(SOURCE_RUN),
            source_hashes=source_hashes(run / "code"),
            frozen_files={
                name: digest(run / name)
                for name in [
                    "config.json",
                    "manifest.json",
                    "subsets.json",
                    "representative-subsets.json",
                ]
            },
            repo_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            smoke=smoke,
        ),
    )
    return run


def launch():
    storage()
    with (ART / "active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip():
            raise RuntimeError("GPU already has an active compute job")
        if shutil.disk_usage(ART).free < 200 * 2**30:
            raise RuntimeError("Need at least 200 GiB free for checkpoints and caches")
        proof = read(ROOT / "setup/recovery-checks.json")
        assert proof["passed"]
        for name, expected in proof["source_hashes"].items():
            assert digest(HERE / name) == expected, (
                f"Re-run checks after editing {name}"
            )
        decoder_proof = read(ROOT / "setup/decoder-checks.json")
        assert decoder_proof["passed"]
        for name in [
            "decoder-python",
            "decode_ctc.py",
            "cache_ctc.py",
            "prepare_lm.py",
            "decoder_checks.py",
        ]:
            assert digest(HERE / name) == decoder_proof["source_hashes"][name]
        fullsize = read(ROOT / "setup/fullsize-checks.json")
        assert fullsize["passed"]
        for name in ["recovery_train.py", "recovery_core.py"]:
            assert digest(HERE / name) == fullsize["source_hashes"][name]
        run = prepare()
        cfg = read(run / "config.json")
        assert cfg["duration_seconds"] == 28800
        start = time.time()
        save(
            run / "state.json",
            dict(
                status="starting",
                stage="launch",
                started_utc=utc(start),
                deadline_utc=utc(start + 28800),
                deadline_epoch=start + 28800,
                stages=[],
            ),
        )
        unit = "stt-" + run.name
        cmd = [
            "systemd-run",
            "--user",
            "--unit=" + unit,
            "--description=Eight hour STT acoustic recovery and local LM experiment",
            "--service-type=exec",
            "--property=RuntimeMaxSec=28800s",
            "--property=TimeoutStopSec=120s",
            "--property=KillMode=control-group",
            "--property=MemoryMax=52G",
            "--property=Restart=no",
            "--property=WorkingDirectory=" + str(run / "code"),
            "--property=StandardOutput=append:" + str(run / "supervisor.log"),
            "--property=StandardError=append:" + str(run / "supervisor.log"),
            str(run / "code/python"),
            str(run / "code/recovery_control.py"),
            "supervise",
            "--run",
            str(run),
        ]
        subprocess.run(cmd, check=True)
        save(run / "service.json", dict(unit=unit, command=cmd))
        save(ROOT / "latest.json", dict(run=str(run), unit=unit))
        print(
            json.dumps(dict(run=str(run), unit=unit, deadline_utc=utc(start + 28800))),
            flush=True,
        )


def supervise(run):
    global STOP
    storage()
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    lock = (ART / "active.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    cfg, prov = read(run / "config.json"), read(run / "provenance.json")
    for name, value in prov["frozen_files"].items():
        assert digest(run / name) == value
    for name, value in prov["source_hashes"].items():
        assert digest(run / "code" / name) == value
    state = read(run / "state.json")
    deadline = state["deadline_epoch"]
    training_end = deadline - cfg["report_reserve_seconds"]

    def update(**kwargs):
        state.update(kwargs, updated_utc=utc())
        save(run / "state.json", state)
        report(run)

    def child(stage, command, seconds, end=training_end):
        budget = min(seconds, end - time.time() - 15)
        if STOP or budget < 180:
            return False
        update(stage=stage, status="running")
        with (run / (stage + ".log")).open("a") as logfile:
            proc = subprocess.Popen(
                command + ["--seconds", str(budget - 30)],
                stdout=logfile,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            update(child_pid=proc.pid)
            cutoff = time.time() + budget
            while proc.poll() is None:
                if STOP or time.time() > cutoff:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    break
                try:
                    gpu = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=5,
                    ).strip()
                    with (run / "gpu.csv").open("a") as log:
                        log.write(f"{time.time()},{stage},{gpu}\n")
                except (OSError, subprocess.SubprocessError):
                    pass
                time.sleep(5)
        state["stages"].append(
            dict(stage=stage, exit_code=proc.returncode, finished_utc=utc())
        )
        update(child_pid=None)
        return proc.returncode == 0

    def train(name, target, seconds):
        prior = run / "training" / name / "result.json"
        if prior.exists() and read(prior)["status"] in {
            "collapsed",
            "failed",
            "gate_failed",
        }:
            return False
        args = [
            str(run / "code/python"),
            str(run / "code/recovery_train.py"),
            str(run),
            name,
            "--target",
            str(target),
        ]
        if (run / "training" / name / "latest.pt").exists():
            args += ["--resume"]
        return child(f"{name}-{target}", args, seconds)

    try:
        update()
        # R5 is triggered only by measured domain-gradient imbalance.
        for name in ["R0", "R1", "R2", "R4", "R3"]:
            train(name, 24000, cfg["screen_seconds"])
        gradient_evidence = []
        for path in (run / "training/R0").glob("eval-*.json"):
            evidence = read(path).get("domain_gradients")
            if evidence and evidence["general"]["gradient_norm"] > 0:
                gradient_evidence.append(
                    evidence["digits"]["gradient_norm"]
                    / evidence["general"]["gradient_norm"]
                )
        if (
            sum(x > 5 for x in gradient_evidence) >= 2
            and training_end - time.time() > 14400
        ):
            add_job(run, "R5", cfg["seed"], "character_weighting")
            train("R5", 24000, cfg["screen_seconds"])
        save(
            run / "conditional-decisions.json",
            dict(
                digit_general_gradient_ratios=gradient_evidence,
                R5_trigger="ratio >5 on at least two probes, with four hours remaining",
            ),
        )
        candidates = []
        for path in (run / "training").glob("*/result.json"):
            r = read(path)
            if r["status"] not in {"failed", "collapsed"} and r["presented"] >= 24000:
                candidates.append((r["best_selection_score"], r["job"]))
        candidates.sort()
        selected = [name for _, name in candidates[:2]]
        save(
            run / "selection.json",
            dict(
                screen_candidates=candidates,
                selected=selected,
                criterion="equal-domain general/symptom development CER; reused validation",
            ),
        )
        # Always give the measured warm-start lead a full-exposure chance if viable.
        for name in dict.fromkeys(["R0", "R1"] + selected):
            train(name, 108000, cfg["confirmation_seconds"])
        viable = []
        for name in dict.fromkeys(["R1"] + selected):
            p = run / "training" / name / "result.json"
            if p.exists():
                r = read(p)
                if r["status"] == "completed" and r["presented"] == 108000:
                    viable.append((r["best_selection_score"], name))
        if viable:
            _, winner = min(viable)
            winner_spec = read(run / "jobs" / (winner + ".json"))
            save(
                run / "winner.json",
                dict(job=winner, variant=winner_spec["variant"], provisional=True),
            )
            for seed in cfg["confirmation_seeds"]:
                if STOP or training_end - time.time() < 3600:
                    break
                initial = None
                if winner_spec["variant"] == "warmstart":
                    gate_name = f"gate-{seed}"
                    add_job(run, gate_name, seed, "control", gate=True)
                    train(gate_name, 48000, 1500)
                    p = run / "training" / gate_name / "result.json"
                    if not p.exists() or read(p)["status"] != "passed_gate":
                        continue
                    initial = run / "training" / gate_name / "latest.pt"
                name = f"confirm-{seed}"
                add_job(run, name, seed, winner_spec["variant"], initial)
                train(name, 108000, cfg["confirmation_seconds"])
                if training_end - time.time() > 1800:
                    name = f"control-{seed}"
                    add_job(run, name, seed, "control")
                    train(name, 108000, cfg["confirmation_seconds"])
        # All cache/decoder jobs are serialized under the same resource lock.
        child(
            "decoding",
            [
                str(run / "code/python"),
                str(run / "code/recovery_evaluate.py"),
                str(run),
            ],
            cfg["report_reserve_seconds"] - 120,
            end=deadline - 90,
        )
        update(
            status="cancelled" if STOP else "finished",
            stage="finished",
            finished_utc=utc(),
            note="See per-job exposure and status; finished does not assert quality gates passed.",
        )
    except Exception as exc:
        traceback.print_exc()
        update(
            status="failed",
            stage="finished",
            error=f"{type(exc).__name__}: {exc}",
            finished_utc=utc(),
        )
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=["prepare", "launch", "supervise", "status", "stop"]
    )
    p.add_argument("--run", type=Path)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.action == "prepare":
        print(prepare(a.run, a.smoke))
    elif a.action == "launch":
        launch()
    else:
        run = artifact(a.run or read(ROOT / "latest.json")["run"])
        if a.action == "supervise":
            supervise(run)
        elif a.action == "status":
            print((run / "REPORT.md").read_text())
            print(json.dumps(read(run / "state.json"), indent=2))
        else:
            subprocess.run(
                ["systemctl", "--user", "stop", read(run / "service.json")["unit"]],
                check=True,
            )
