"""Launch, supervise, inspect, and stop the bounded four-hour learning pilot."""

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

import numpy as np

from common import ART, REPO, digest, read_manifest, save, storage
from pilot4_probe import select_rows

HERE = Path(__file__).resolve().parent
STOP = False


def halt(*_):
    global STOP
    STOP = True


def utc(timestamp=None):
    return datetime.datetime.fromtimestamp(
        timestamp or time.time(), datetime.timezone.utc
    ).isoformat()


def make_report(run):
    cfg = json.loads((run / "config.json").read_text())
    state_path = run / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    rows = []
    summary = {"state": state, "arms": {}}
    for arm in cfg["arms"]:
        name = arm["name"]
        data = {}
        for mode in ["gate", "benchmark", "train"]:
            p = run / mode / name / "result.json"
            data[mode] = json.loads(p.read_text()) if p.exists() else None
        p = run / "train" / name / "final-development.json"
        data["development"] = json.loads(p.read_text()) if p.exists() else None
        summary["arms"][name] = data
        gate = data["gate"]["status"] if data["gate"] else "pending"
        train = data["train"]["status"] if data["train"] else "pending"
        metric = data["development"]["metrics"] if data["development"] else None
        rows.append(
            f"| {name} | {gate} | {train} | {data['train']['steps'] if data['train'] else '—'} | "
            + (
                f"{metric['general']['wer']:.2%} | {metric['medical_symptoms']['wer']:.2%} | {metric['digits']['exact']}/{metric['digits']['total']} |"
                if metric
                else "— | — | — |"
            )
        )
    trained = [d["train"] for d in summary["arms"].values() if d["train"]]
    summary["matched_three_arm_exposure"] = (
        len(trained) == 3
        and all(d["status"] == "completed" for d in trained)
        and len({d["steps"] for d in trained}) == 1
        and len({d["sample_hash"] for d in trained}) == 1
        and len({d["initial_hash"] for d in trained}) == 1
        and len({d.get("augmentation_hash") for d in trained}) == 1
    )
    save(run / "report.json", summary)
    body = f"# Four-hour training pilot\n\nStatus: {state.get('status', 'preparing')}; stage: {state.get('stage', 'preparing')}.\n\n"
    body += f"Started: {state.get('started_utc', 'pending')}. Deadline: {state.get('deadline_utc', 'pending')}.\n\n"
    body += (
        "| Arm | Learning gate | Training | Updates | General WER | Medical symptom WER | Exact digits |\n|---|---|---|---:|---:|---:|---:|\n"
        + "\n".join(rows)
    )
    body += f"\n\nMatched initialization and exposure for all three completed arms: {summary['matched_three_arm_exposure']}.\n\n"
    body += (
        cfg["scope"]
        + "\n\nValidation checkpoint selection: development CER + 0.5 × digit-sequence error rate. Final metrics reload that checkpoint in a fresh process.\n"
    )
    if state.get("error"):
        body += "\nFailure: " + state["error"] + "\n"
    (run / "REPORT.md").write_text(body)


def launch(config=None, root=None, service=True):
    storage()
    lock = (ART / "active.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPU has an active compute process")
    if service:
        proof = json.loads((ART / "preflight/pilot4-checks.json").read_text())
        assert proof["passed"]
        for name, expected in proof["source_hashes"].items():
            assert digest(HERE / name) == expected, (
                f"Re-run pilot4_checks.py after changing {name}"
            )
    if shutil.disk_usage(ART).free < 100 * 2**30:
        raise RuntimeError("Need 100 GiB free for resumable checkpoints")
    cfg = json.loads((Path(config) if config else HERE / "pilot4.json").read_text())
    assert cfg["duration_seconds"] <= 14400
    run = (
        Path(root)
        if root
        else ART
        / "runs"
        / datetime.datetime.now(datetime.timezone.utc).strftime(
            "pilot-4h-%Y%m%dT%H%M%SZ"
        )
    )
    assert run.resolve().is_relative_to(ART)
    run.mkdir(parents=True, exist_ok=False)
    code = run / "code"
    code.mkdir()
    for p in HERE.iterdir():
        if p.is_file() and (
            p.suffix in [".py", ".json", ".md", ".txt"] or p.name == "python"
        ):
            shutil.copy2(p, code / p.name)
    (code / "python").chmod(0o755)
    save(run / "config.json", cfg)
    manifest = read_manifest()
    save(run / "manifest.json", manifest)
    gate = select_rows(32)
    gate_ids = {r["id"] for r in gate}
    rng = np.random.default_rng(cfg["seed"] + 10)
    monitor = []
    for domain in cfg["domain_probabilities"]:
        candidates = [
            r
            for r in manifest["rows"]
            if r["split"] == "train"
            and r["domain"] == domain
            and r["id"] not in gate_ids
        ]
        monitor.extend(
            rng.choice([r["id"] for r in candidates], size=16, replace=False).tolist()
        )
    save(
        run / "subsets.json",
        {
            "gate": [r["id"] for r in gate],
            "monitor": monitor,
            "selection": "Gate: 12 general, 12 symptom, 8 digit training clips with Omi label agreement. Monitor: 16 other training clips per domain. Dev: existing 160-row development split.",
        },
    )
    save(
        run / "provenance.json",
        {
            "repo_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "source_hashes": {p.name: digest(p) for p in code.iterdir()},
            "config_sha256": digest(run / "config.json"),
            "manifest_sha256": digest(run / "manifest.json"),
            "subsets_sha256": digest(run / "subsets.json"),
            "teacher_screening_source": str(
                ART / "runs/pilot-8h-20260910T044750Z/targets.json"
            ),
            "teacher_screening_sha256": digest(
                ART / "runs/pilot-8h-20260910T044750Z/targets.json"
            ),
        },
    )
    if not service:
        print(json.dumps({"run": str(run), "prepared_only": True}), flush=True)
        return
    unit = "stt-distill-" + run.name
    cmd = [
        "systemd-run",
        "--user",
        "--unit=" + unit,
        "--description=Four hour STT learning and ternary optimizer pilot",
        "--service-type=exec",
        "--property=RuntimeMaxSec=" + str(cfg["duration_seconds"]) + "s",
        "--property=TimeoutStopSec=30s",
        "--property=KillMode=control-group",
        "--property=MemoryMax=52G",
        "--property=Restart=no",
        "--property=WorkingDirectory=" + str(REPO),
        "--property=StandardOutput=append:" + str(run / "supervisor.log"),
        "--property=StandardError=append:" + str(run / "supervisor.log"),
        str(code / "python"),
        str(code / "pilot4_control.py"),
        "supervise",
        "--run",
        str(run),
    ]
    subprocess.run(cmd, check=True)
    save(run / "service.json", {"unit": unit, "command": cmd})
    save(ART / "latest4.json", {"run": str(run), "unit": unit})
    fcntl.flock(lock, fcntl.LOCK_UN)
    print(json.dumps({"run": str(run), "unit": unit}), flush=True)


def supervise(run):
    storage()
    lock = (ART / "active.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    cfg = json.loads((run / "config.json").read_text())
    prov = json.loads((run / "provenance.json").read_text())
    assert digest(run / "config.json") == prov["config_sha256"]
    assert digest(run / "manifest.json") == prov["manifest_sha256"]
    assert digest(run / "subsets.json") == prov["subsets_sha256"]
    for name, expected in prov["source_hashes"].items():
        assert digest(run / "code" / name) == expected
    start = time.time()
    deadline = start + cfg["duration_seconds"]
    state = {
        "run": str(run),
        "status": "running",
        "stage": "preflight",
        "started_utc": utc(start),
        "deadline_utc": utc(deadline),
        "stages": [],
        "pid": os.getpid(),
    }

    def update(**kw):
        state.update(kw)
        state["updated_utc"] = utc()
        save(run / "state.json", state)
        make_report(run)

    def command(mode, arm, seconds, steps=100):
        name = arm["name"]
        stage = mode + "-" + name
        update(stage=stage)
        budget = min(seconds, deadline - time.time() - 30)
        if STOP or budget < 20:
            raise TimeoutError("Outer budget exhausted or stop requested")
        args = [
            str(run / "code/python"),
            str(run / "code/pilot4_train.py"),
            str(run),
            name,
            mode,
            "--steps",
            str(steps),
            "--seconds",
            str(max(1, budget - 45)),
        ]
        with (run / (stage + ".log")).open("a") as log:
            child = subprocess.Popen(
                args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            update(child_pid=child.pid)
            end = time.time() + budget
            reason = None
            try:
                while child.poll() is None:
                    try:
                        values = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=5,
                        ).strip()
                        with (run / "gpu.csv").open("a") as f:
                            f.write(f"{time.time()},{stage},{values}\n")
                    except (subprocess.SubprocessError, OSError):
                        pass
                    if STOP or time.time() >= end:
                        reason = "cancelled" if STOP else "time_budget"
                        os.killpg(child.pid, signal.SIGTERM)
                        break
                    time.sleep(5)
                if child.poll() is None:
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        result = {
            "stage": stage,
            "exit_code": child.returncode,
            "reason": reason,
            "finished_utc": utc(),
        }
        state["stages"].append(result)
        update(child_pid=None)
        return result

    try:
        update()
        eligible = []
        for arm in cfg["arms"]:
            result = command("gate", arm, cfg["gate_seconds"], cfg["gate_steps"])
            p = run / "gate" / arm["name"] / "result.json"
            passed = p.exists() and json.loads(p.read_text())["status"] == "passed_gate"
            if arm["precision"] == "fp" and not passed:
                raise RuntimeError(
                    "Full-size FP memorization gate failed; comparison training was not started"
                )
            if passed:
                eligible.append(arm)
            else:
                print(
                    f"Skipping {arm['name']}: learning gate failed, see {p}", flush=True
                )
        rates = {}
        for arm in eligible:
            result = command("benchmark", arm, 300, 100)
            if result["exit_code"]:
                raise RuntimeError("Benchmark failed for " + arm["name"])
            profile = json.loads(
                (run / "benchmark" / arm["name"] / "result.json").read_text()
            )
            assert profile["status"] == "completed" and profile["steps"] == 100
            rates[arm["name"]] = max(
                profile["mean_update_seconds"] * 1.25,
                profile["p90_update_seconds"] * 1.1,
            )
        available = (
            deadline - time.time() - cfg["report_reserve_seconds"] - 300 * len(eligible)
        )
        steps = (
            min(cfg["max_steps"], int(available / sum(rates.values()))) // 1000 * 1000
        )
        if steps < 2000:
            raise RuntimeError("Insufficient time for comparison after learning gates")
        save(
            run / "allocation.json",
            {
                "common_steps": steps,
                "eligible_arms": eligible,
                "conservative_seconds_per_update": rates,
                "reserve_seconds": cfg["report_reserve_seconds"],
                "budget_seconds": available,
            },
        )
        for index, arm in enumerate(eligible):
            remaining = eligible[index:]
            available = (
                deadline
                - time.time()
                - cfg["report_reserve_seconds"]
                - 180 * len(remaining)
            )
            budget = (
                available
                * rates[arm["name"]]
                / sum(rates[a["name"]] for a in remaining)
            )
            command("train", arm, max(180, budget), steps)
            if (run / "train" / arm["name"] / "best.pt").exists():
                command("evaluate", arm, 180)
            if STOP:
                break
        make_report(run)
        report = json.loads((run / "report.json").read_text())
        success = (
            len(eligible) == 3
            and report["matched_three_arm_exposure"]
            and all(
                d["development"]
                and d["development"]["reload_matches_selection_metrics"]
                for d in report["arms"].values()
            )
        )
        update(
            status="cancelled" if STOP else "completed" if success else "partial",
            stage="finished",
            finished_utc=utc(),
        )
    except Exception as exc:
        traceback.print_exc()
        update(
            status="cancelled" if STOP else "failed",
            stage="finished",
            error=f"{type(exc).__name__}: {exc}",
            finished_utc=utc(),
        )


def status(run=None, stop=False):
    latest = json.loads((ART / "latest4.json").read_text())
    run = Path(run) if run else Path(latest["run"])
    service = json.loads((run / "service.json").read_text())
    if stop:
        subprocess.run(["systemctl", "--user", "stop", service["unit"]], check=True)
    subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            service["unit"],
            "--property=ActiveState,SubState,RuntimeMaxUSec,MemoryMax,ExecMainStartTimestamp",
        ]
    )
    print(
        (run / "state.json").read_text()
        if (run / "state.json").exists()
        else "Starting"
    )
    print("Report:", run / "REPORT.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=["start", "prepare", "supervise", "status", "stop"]
    )
    p.add_argument("--config", type=Path)
    p.add_argument("--run", type=Path)
    a = p.parse_args()
    if a.action in ["start", "prepare"]:
        launch(a.config, a.run, service=a.action == "start")
    elif a.action == "supervise":
        supervise(a.run)
    else:
        status(a.run, stop=a.action == "stop")
