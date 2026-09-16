"""Explicit prepare / check-cpu / preflight-gpu / launch stages. No auto-start."""

import argparse
import datetime
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import ART, digest, save, storage
from recovery_core import SOURCE_RUN, artifact, read, source_hashes
from scaling_core import ROOT, common_tape, rank, tier_ids

HERE = Path(__file__).resolve().parent


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def fingerprint(run):
    paths = [run / n for n in ("config.json", "manifest.json", "subsets.json", "data-ready.json")]
    paths += sorted((run / "jobs").glob("*.json")) + sorted((run / "tapes").glob("*"))
    paths += sorted(p for p in (run / "code").iterdir() if p.is_file())
    return {str(p.relative_to(run)): digest(p) for p in paths}


def prepare(run, data):
    run, data = artifact(run), artifact(data)
    ready = read(data / "ready.json")
    assert ready["status"] == "data_prepared_no_training"
    assert ready["config_sha256"] == digest(HERE / "scaling.json")
    for name, h in ready["files"].items():
        assert digest(data / name) == h
    run.mkdir(parents=True, exist_ok=False)
    for name in ("jobs", "tapes", "code"):
        (run / name).mkdir()
    for name in ("manifest.json", "subsets.json"):
        shutil.copy2(data / name, run / name)
    shutil.copy2(data / "ready.json", run / "data-ready.json")
    shutil.copy2(HERE / "scaling.json", run / "config.json")
    cfg = read(run / "config.json")
    checkpoints = [SOURCE_RUN / "gate/fp_control/latest.pt",
                   ART / "recovery-and-decoding/runs/recovery-8h-20260911T031803Z/training/gate-20260911/latest.pt"]
    assert len(cfg["seeds"]) == len(checkpoints)
    designs = read(data / "designs.json")
    for seed, checkpoint in zip(cfg["seeds"], checkpoints):
        design = designs[str(seed)]
        groups_path = run / "tapes" / f"groups-{seed}.json"
        save(groups_path, design)
        tape_path = run / "tapes" / f"draws-{seed}.npy"
        np.save(tape_path, common_tape(design["groups"], seed, cfg), allow_pickle=False)
        checkpoint_hash = digest(artifact(checkpoint))
        for tier, size in cfg["sizes"].items():
            job = f"{seed}-{tier}"
            save(run / "jobs" / (job + ".json"), dict(name=job, tier=tier, size=size, seed=seed,
                groups=str(groups_path.relative_to(run)), groups_sha256=digest(groups_path),
                tape=str(tape_path.relative_to(run)), tape_sha256=digest(tape_path),
                initial_checkpoint=str(checkpoint), initial_checkpoint_sha256=checkpoint_hash))
    names = set(source_hashes(HERE)) | {"augmentation-python"}
    for name in sorted(names):
        shutil.copy2(HERE / name, run / "code" / name)
    (run / "code/augmentation-python").chmod(0o755)
    # Counterbalance the second replicate's order, without looking at outcomes.
    tiers = list(cfg["sizes"])
    order = [f"{seed}-{tier}" for index, seed in enumerate(cfg["seeds"])
             for tier in (tiers if index % 2 == 0 else list(reversed(tiers))) ]
    save(run / "provenance.json", dict(files=fingerprint(run), source=str(HERE),
        execution_order=order, created_utc=utc(), data_source=str(data)))
    save(run / "state.json", dict(status="prepared_not_started", gpu_preflight="pending", training_started=False))
    print(json.dumps(dict(run=str(run), status="prepared_not_started", jobs=order)), flush=True)


def verify(run, gpu_proof=False):
    prov = read(run / "provenance.json")
    assert fingerprint(run) == prov["files"], "Frozen code/config/data contract changed; prepare a fresh experiment"
    assert shutil.disk_usage(ART).free >= read(run / "config.json")["minimum_free_disk_gib"] * 2**30
    for name in ("training", "gcc"):
        path = ART / "augmentation-pilot/runtime" / name
        target = path.resolve(strict=True)
        roots = subprocess.check_output(["nix-store", "--query", "--roots", str(target)], text=True)
        assert str(path) in roots, "Pinned runtime GC root is missing"
    if gpu_proof:
        proof = read(run / "gpu-preflight.json")
        assert proof["passed"] and proof["fingerprint"] == prov["files"]
        assert read(run / "cpu-preflight.json")["passed"]
    return prov


def idle_lock(launch_handoff=False):
    lock = (ART / "active.lock").open("a")
    until = time.monotonic() + (5 if launch_handoff else 0)
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= until:
                lock.close()
                raise
            time.sleep(.05)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    assert not active.strip(), "GPU is busy; no experiment or preflight was started"
    return lock


def preflight(run):
    prov = verify(run)
    cpu = read(run / "cpu-preflight.json")
    assert cpu["passed"] and cpu["fingerprint"] == prov["files"]
    with idle_lock():
        # Run in a separate frozen fixture, preserving the actual experiment.
        # Four maximum-padded-length recordings exercise worst-case attention,
        # Adam state creation, finite gradients, evaluation and optimizer resume.
        check = run / "preflight-fullsize"
        if check.exists():
            raise RuntimeError("Preflight fixture exists; inspect it before retrying")
        check.mkdir()
        (check / "jobs").mkdir()
        shutil.copy2(run / "manifest.json", check / "manifest.json")
        cfg = read(run / "config.json")
        cfg.update(checkpoint_updates=[1, 2, 4], checkpoint_training_seconds=[], recovery_every_updates=1)
        save(check / "config.json", cfg)
        original_spec = read(run / "jobs" / (prov["execution_order"][0] + ".json"))
        groups = read(run / original_spec["groups"])["groups"]
        longest = max((g for g in groups if g["domain"] == "general"), key=lambda g: g["bound"])
        save(check / "groups.json", dict(groups=[longest]))
        tape = np.array([[0, j, 100, j % 5, longest["bound"] + 100] for j in range(16)], dtype=np.int32)
        np.save(check / "tape.npy", tape, allow_pickle=False)
        subsets = read(run / "subsets.json")
        by_id = {r["id"]: r for r in read(run / "manifest.json")["rows"]}
        save(check / "subsets.json", {name: [i for d in ("general", "medical_symptoms", "digits")
             for i in [v for v in ids if by_id[v]["domain"] == d][:2]] for name, ids in subsets.items()})
        spec = dict(original_spec, name="probe", size=8, groups="groups.json", groups_sha256=digest(check / "groups.json"),
                    tape="tape.npy", tape_sha256=digest(check / "tape.npy"))
        save(check / "jobs/probe.json", spec)
        command = [str(run / "code/augmentation-python"), str(run / "code/scaling_control.py"),
                   "worker", str(check), "--job", "probe", "--preflight"]
        subprocess.run(command + ["--stop-after", "2"], check=True)
        subprocess.run(command + ["--resume"], check=True)
        result = read(check / "training/probe/result.json")
        assert result["status"] == "completed" and result["update"] == 4
        assert result["max_vram_gib"] < cfg["max_vram_gib"]
        # Verify the second independent gate can also load, without training it.
        code = "import torch,sys; from onset_model import OnsetModel; from recovery_core import read; m=OnsetModel(read(sys.argv[1])['model'],'fp'); s=torch.load(sys.argv[2],map_location='cpu',mmap=True,weights_only=False); m.load_state_dict(s['model'],strict=True)"
        second = read(run / "jobs" / (prov["execution_order"][-1] + ".json"))
        assert digest(second["initial_checkpoint"]) == second["initial_checkpoint_sha256"]
        subprocess.run([str(run / "code/augmentation-python"), "-c", code, str(run / "config.json"), second["initial_checkpoint"]], cwd=run / "code", check=True)
        step_seconds = result["timing"]["training_work_seconds"] / 4
        save(run / "gpu-preflight.json", dict(passed=True, fingerprint=prov["files"], checked_utc=utc(),
            result=result, worst_case_seconds_per_update=step_seconds,
            conservative_total_training_hours=step_seconds * max(read(run / "config.json")["checkpoint_updates"]) * len(prov["execution_order"]) / 3600,
            note="Worst-length throughput after short warmup, not a precise wall-time forecast. Evaluation/checkpoint overhead is extra."))
        save(run / "state.json", dict(status="ready_not_started", training_started=False, gpu_preflight="passed"))


def run_all(run, resume=False, launch_handoff=False):
    prov = verify(run, gpu_proof=True)
    with idle_lock(launch_handoff):
        if read(run / "state.json").get("training_started") and not resume:
            raise RuntimeError("Experiment already started; use --resume")
        for job in prov["execution_order"]:
            result = run / "training" / job / "result.json"
            if result.exists() and read(result)["status"] == "completed":
                continue
            save(run / "state.json", dict(status="running", job=job, training_started=True, updated_utc=utc()))
            command = [str(run / "code/augmentation-python"), str(run / "code/scaling_control.py"), "worker", str(run), "--job", job]
            if (run / "training" / job / "latest.pt").exists():
                command.append("--resume")
            try:
                with (run / (job + ".log")).open("a") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                assert read(result)["status"] == "completed", "Worker stopped before matched update budget"
                subprocess.run([str(run / "code/augmentation-python"), str(run / "code/scaling_analysis.py"), str(run)], check=True)
            except Exception as exc:
                save(run / "state.json", dict(status="stopped", job=job, training_started=True,
                    error=f"{type(exc).__name__}: {exc}", updated_utc=utc()))
                raise
        save(run / "state.json", dict(status="completed", training_started=True, updated_utc=utc()))


def launch(run, resume=False):
    verify(run, gpu_proof=True)
    with idle_lock():
        unit = "stt-scaling-" + run.name
        command = ["systemd-run", "--user", "--unit=" + unit, "--service-type=exec",
            "--description=Controlled STT unique data and compute experiment",
            "--property=TimeoutStopSec=240s", "--property=KillMode=control-group",
            "--property=MemoryMax=52G", "--property=Restart=no",
            "--property=WorkingDirectory=" + str(run / "code"),
            "--property=StandardOutput=append:" + str(run / "supervisor.log"),
            "--property=StandardError=append:" + str(run / "supervisor.log"),
            str(run / "code/augmentation-python"), str(run / "code/scaling_control.py"), "run", str(run), "--launch-handoff"]
        if resume:
            command.append("--resume")
        # The service checks the same lock after this parent releases it.
        # systemd-run returns when exec starts, before application initialization.
        subprocess.run(command, check=True)
        save(run / "service.json", dict(unit=unit, command=command))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "check-cpu", "preflight-gpu", "launch", "run", "worker", "status"])
    p.add_argument("run", type=Path)
    p.add_argument("--data", type=Path, default=ROOT / "data/v1")
    p.add_argument("--job")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--launch-handoff", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--stop-after", type=int)
    args = p.parse_args()
    run = artifact(args.run)
    if args.action == "prepare":
        prepare(run, args.data)
    elif args.action == "check-cpu":
        verify(run)
        subprocess.run([str(run / "code/augmentation-python"), str(run / "code/scaling_checks.py"), str(run)], check=True)
    elif args.action == "preflight-gpu":
        preflight(run)
    elif args.action == "launch":
        launch(run, args.resume)
    elif args.action == "run":
        run_all(run, args.resume, args.launch_handoff)
    elif args.action == "worker":
        # Internal children only: parent holds the project lock. This route is
        # explicit, never reached by prepare/status/check-cpu.
        from scaling_train import worker
        worker(run, args.job, resume=args.resume, stop_after=args.stop_after, preflight=args.preflight)
    else:
        print(json.dumps(read(run / "state.json"), indent=2))
