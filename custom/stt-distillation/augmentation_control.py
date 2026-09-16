"""Prepare, launch, supervise and report an eight-hour augmentation pilot."""

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

from augmentation_data import RECOVERY_RUN, ROOT
from common import ART, digest, save, storage
from recovery_core import DOMAINS, SOURCE_RUN, artifact, read, source_hashes, write_tape

HERE = Path(__file__).resolve().parent
STOP = False


def utc(epoch=None):
    return datetime.datetime.fromtimestamp(time.time() if epoch is None else epoch, datetime.timezone.utc).isoformat()


def halt(*_):
    global STOP
    STOP = True


def hashes():
    return dict(source_hashes(HERE), **{"augmentation-python": digest(HERE / "augmentation-python")})


def prepare(root=None, smoke=False):
    storage()
    run = artifact(root or ROOT / "runs" / datetime.datetime.now(datetime.timezone.utc).strftime("augmentation-8h-%Y%m%dT%H%M%SZ"))
    run.mkdir(parents=True, exist_ok=False)
    cfg = dict(read(HERE / "recovery8.json"), **read(HERE / "augmentation8.json"))
    if smoke:
        cfg["model"] = dict(cfg["model"], width=32, depth=2, heads=4, context=16, checkpoint=False)
        cfg.update(eval_every_examples=16, checkpoint_every_examples=16)
    save(run / "config.json", cfg)
    for name in ("manifest.json", "subsets.json"):
        shutil.copy2(SOURCE_RUN / name, run / name)
    sources = {}
    for name, path in {
        "noise_manifest": ROOT / "data/demand/manifest.json",
        "validation_manifest": ROOT / "data/validation.json",
        "known_speakers_manifest": SOURCE_RUN / "analysis/broad-investigation/unseen-known-speakers.json",
    }.items():
        sources[name], sources[name + "_sha256"] = str(path), digest(path)
    save(run / "augmentation-sources.json", sources)
    rows = [r for r in read(run / "manifest.json")["rows"] if r["split"] == "train"]
    for seed, prefix, checkpoint in [
        (cfg["seed"], "A", SOURCE_RUN / "gate/fp_control/latest.pt"),
        (cfg["confirmation_seed"], "B", RECOVERY_RUN / "training/gate-20260911/latest.pt"),
    ]:
        tape = write_tape(run, rows, seed, cfg["exposure_horizon"])
        checkpoint_hash = digest(checkpoint) if not smoke else None
        for index, policy in enumerate(cfg["arms"].values()):
            name = prefix + str(index)
            spec = dict(name=name, variant="augmentation", seed=seed, gate=False, precision="fp",
                row_ids=[r["id"] for r in rows], tape=str(tape), tape_sha256=digest(tape),
                encoder_lr=1e-5, head_lr=1e-5, effective_batch=4,
                gains=[0, -6, -12, -18, -24], acoustic_augmentation=policy)
            if not smoke:
                spec.update(initial_checkpoint=str(checkpoint), initial_checkpoint_sha256=checkpoint_hash)
            save(run / "jobs" / (name + ".json"), spec)
    (run / "code").mkdir()
    for name in hashes():
        shutil.copy2(HERE / name, run / "code" / name)
    for name in ("python", "decoder-python", "augmentation-python"):
        (run / "code" / name).chmod(0o755)
    save(run / "provenance.json", dict(source_hashes=hashes(), smoke=smoke,
        frozen_files={name: digest(run / name) for name in ("config.json", "manifest.json", "subsets.json", "augmentation-sources.json")},
        job_hashes={p.name: digest(p) for p in (run / "jobs").glob("*.json")}))
    return run


def comparison(run, prefix="A"):
    cfg = read(run / "config.json")["selection"]
    def get(name):
        path = run / "evaluation" / name / "summary.json"
        result = run / "training" / name / "result.json"
        if not path.exists() or not result.exists():
            return None
        ev, tr = read(path), read(result)
        if ev["status"] != "completed" or tr["status"] != "completed" or tr["presented"] != 108000:
            return None
        return ev["conditions"]
    base = get(prefix + "0")
    if base is None:
        raise RuntimeError("A complete matched-exposure baseline is required for selection")
    def robust(c):
        return sum(c[k][d]["wer"] for k in ("noise15", "speech20") for d in DOMAINS[:2]) / 4
    def clean(c):
        return sum(c["clean"][d]["wer"] for d in DOMAINS[:2]) / 2
    candidates = []
    for i in (1, 2, 3):
        name = prefix + str(i)
        metric = get(name)
        if metric is None:
            continue
        relative = (robust(base) - robust(metric)) / max(robust(base), 1e-12)
        clean_delta = {d: metric["clean"][d]["wer"] - base["clean"][d]["wer"] for d in DOMAINS[:2]}
        digit_delta = {k: metric[k]["digits"]["exact"] - base[k]["digits"]["exact"] for k in ("clean", "noise15", "speech20")}
        guarded = max(clean_delta.values()) <= cfg["maximum_clean_domain_wer_regression"] and min(digit_delta.values()) >= -cfg["maximum_clean_digit_sequence_regression"]
        # Either improved robustness or clean generalization warrants confirmation, subject to all guards.
        clean_relative = (clean(base) - clean(metric)) / max(clean(base), 1e-12)
        eligible = guarded and (relative >= cfg["minimum_relative_robust_wer_improvement"] or (clean_relative >= 0.05 and relative >= 0))
        candidates.append(dict(job=name, eligible=eligible, relative_robust_wer_improvement=relative,
            relative_clean_wer_improvement=clean_relative, clean_domain_wer_deltas=clean_delta,
            digit_sequence_deltas=digit_delta, score=(robust(metric) + clean(metric)) / 2))
    if not candidates:
        raise RuntimeError("No complete augmentation arm available")
    ranked = sorted(candidates, key=lambda c: (not c["eligible"], c["score"]))
    return dict(candidates=ranked, selected=ranked[0]["job"],
        status="candidate_for_confirmation" if ranked[0]["eligible"] else "exploratory_confirmation_no_arm_passed",
        criterion="Equal-domain clean/robust WER, per-domain clean WER and digit guards; stress conditions excluded")


def report(run):
    state = read(run / "state.json")
    lines = ["# Eight-hour augmentation pilot", "", f"Status: {state['status']}. Stage: {state['stage']}.",
             f"Started: {state['started_utc']}. Deadline: {state['deadline_utc']}.", "",
             "| Arm | State | Examples | Clean general WER | Clean medical WER | Clean digits | Noise15 speech WER | Speech20 speech WER |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    jobs = {}
    for path in sorted((run / "training").glob("*/result.json")):
        name = path.parent.name
        train = read(path)
        jobs[name] = {k: train.get(k) for k in ("status", "presented", "updates", "error", "elapsed_seconds", "sample_hash", "initial_hash")}
        ep = run / "evaluation" / name / "summary.json"
        if ep.exists() and read(ep)["status"] == "completed":
            ev = read(ep)
            jobs[name]["evaluation"] = ev
            c = ev["conditions"]
            g, m, d = (c["clean"][k] for k in DOMAINS)
            robust = [sum(c[k][domain]["wer"] for domain in DOMAINS[:2]) / 2 for k in ("noise15", "speech20")]
            lines.append(f"| {name} | {train['status']} | {train['presented']} | {g['wer']:.2%} | {m['wer']:.2%} | {d['exact']}/{d['total']} | {robust[0]:.2%} | {robust[1]:.2%} |")
        else:
            lines.append(f"| {name} | {train['status']}; evaluation pending | {train['presented']} | — | — | — | — | — |")
    lines += ["", "Metrics use reloaded clean-CER-selected checkpoints and fixed greedy CTC decoding.",
        "A arms share one passing-gate checkpoint and sample tape; B arms use an independent passing gate and sample seed.",
        "Validation speech is reused. Synthetic mixtures approximate relative speaker levels, not physical microphone distance.",
        "Severe noise and equal-volume overlap are stress diagnostics, not checkpoint or recipe selection inputs.",
        "A reported orchestration failure or incomplete comparison is not a successful experiment."]
    selection = read(run / "selection.json") if (run / "selection.json").exists() else None
    if selection:
        lines += ["", "Selection: " + selection["status"] + "; " + selection["selected"] + "."]
    if state.get("error"):
        lines += ["", "Error: " + state["error"]]
    save(run / "report.json", dict(state=state, jobs=jobs, selection=selection))
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")


def launch():
    storage()
    proof = read(ROOT / "setup/preflight.json")
    assert proof["passed"] and proof["service_passed"]
    assert proof["source_hashes"] == hashes(), "Code changed since preflight"
    for name, value in proof["asset_hashes"].items():
        assert digest(name) == value
    for root in ("training", "gcc"):
        target = (ROOT / "runtime" / root).resolve(strict=True)
        registered = subprocess.check_output(["nix-store", "--query", "--roots", str(target)], text=True)
        assert str(ROOT / "runtime" / root) in registered, "Runtime is not a registered GC root"
    assert shutil.disk_usage(ART).free > 150 * 2**30
    with (ART / "active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "GPU is busy"
        run = prepare()
        shutil.copy2(ROOT / "setup/preflight.json", run / "preflight.json")
        start = time.time()
        save(run / "state.json", dict(status="starting", stage="launch", started_utc=utc(start),
            deadline_utc=utc(start + 28800), deadline_epoch=start + 28800, stages=[]))
        unit = "stt-" + run.name
        command = ["systemd-run", "--user", "--unit=" + unit,
            "--description=Eight hour STT controlled acoustic augmentation pilot", "--service-type=exec",
            "--property=RuntimeMaxSec=28800s", "--property=TimeoutStopSec=120s",
            "--property=KillMode=control-group", "--property=MemoryMax=52G", "--property=Restart=no",
            "--property=WorkingDirectory=" + str(run / "code"),
            "--property=StandardOutput=append:" + str(run / "supervisor.log"),
            "--property=StandardError=append:" + str(run / "supervisor.log"),
            str(run / "code/augmentation-python"), str(run / "code/augmentation_control.py"), "supervise", "--run", str(run)]
        subprocess.run(command, check=True)
        save(run / "service.json", dict(unit=unit, command=command))
        save(ROOT / "latest.json", dict(run=str(run), unit=unit))
        print(json.dumps(dict(run=str(run), unit=unit, deadline_utc=utc(start + 28800))), flush=True)


def supervise(run):
    global STOP
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    lock = (ART / "active.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    state, cfg = read(run / "state.json"), read(run / "config.json")
    deadline = state["deadline_epoch"]
    training_end = deadline - cfg["training_reserve_seconds"]
    def update(**kw):
        state.update(kw, updated_utc=utc())
        save(run / "state.json", state)
        report(run)
    def child(stage, script, arguments, seconds, end=training_end):
        budget = min(seconds, end - time.time() - 20)
        if STOP:
            raise InterruptedError("Stop requested")
        if budget < 180:
            raise TimeoutError("Insufficient reserved time for " + stage)
        update(status="running", stage=stage)
        command = [str(run / "code/augmentation-python"), str(run / "code" / script), *arguments, "--seconds", str(budget - 30)]
        with (run / (stage + ".log")).open("a") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            update(child_pid=proc.pid)
            end_stage = time.time() + budget
            while proc.poll() is None:
                if STOP or time.time() > end_stage:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    break
                try:
                    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu", "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
                    with (run / "gpu.csv").open("a") as out:
                        out.write(f"{time.time()},{stage},{gpu}\n")
                except (OSError, subprocess.SubprocessError):
                    pass
                time.sleep(5)
        state["stages"].append(dict(stage=stage, exit_code=proc.returncode, finished_utc=utc()))
        update(child_pid=None)
        if STOP:
            raise InterruptedError("Stop requested")
        if proc.returncode:
            raise RuntimeError(f"{stage} failed with exit code {proc.returncode}; see {stage}.log")
    def train(name):
        child("train-" + name, "augmentation_train.py", [str(run), name, "--target", str(cfg["exposure_horizon"])], cfg["job_seconds"])
        result = read(run / "training" / name / "result.json")
        if result["status"] != "completed" or result["presented"] != cfg["exposure_horizon"]:
            raise RuntimeError(f"{name} did not finish matched exposure: {result['status']} / {result['presented']}")
        child("evaluate-" + name, "augmentation_evaluate.py", [str(run), name], 600, deadline)
    try:
        prov = read(run / "provenance.json")
        for name, expected in prov["frozen_files"].items():
            assert digest(run / name) == expected
        for name, expected in prov["source_hashes"].items():
            assert digest(run / "code" / name) == expected
        for name, expected in prov["job_hashes"].items():
            assert digest(run / "jobs" / name) == expected
        for name in ("A0", "A1", "A2", "A3"):
            train(name)
        selection = comparison(run)
        save(run / "selection.json", selection)
        chosen = "B" + selection["selected"][1:]
        for name in ("B0", chosen):
            train(name)
        confirmation = comparison(run, "B")
        save(run / "confirmation.json", confirmation)
        # Audits are bounded and their completion is checked, not inferred from exit code.
        for name in ("A0", selection["selected"]):
            child("audit-" + name, "recovery_audit.py", [str(run), name], 900, deadline)
            audit = read(run / "evaluation" / ("recovery-" + name) / "generalization-audit.json")
            assert audit["reload_matches_selection"] and all(v["complete"] for v in audit["results"].values()), "Incomplete generalization audit"
        update(status="completed", stage="finished", finished_utc=utc(), confirmation=confirmation)
    except Exception as exc:
        traceback.print_exc()
        update(status="cancelled" if isinstance(exc, InterruptedError) else "failed", stage="stopped",
               error=f"{type(exc).__name__}: {exc}", finished_utc=utc())
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "launch", "supervise", "status", "stop"])
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare(args.run))
    elif args.action == "launch":
        launch()
    else:
        pointer = read(ROOT / "latest.json") if args.run is None else None
        run = artifact(args.run or pointer["run"])
        if args.action == "supervise":
            supervise(run)
        elif args.action == "status":
            report(run)
            print((run / "REPORT.md").read_text())
        else:
            subprocess.run(["systemctl", "--user", "stop", read(run / "service.json")["unit"]], check=True)
