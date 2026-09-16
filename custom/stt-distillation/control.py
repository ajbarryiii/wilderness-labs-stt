"""Start/status/stop a detached eight-hour run on the mounted data disk."""

import argparse, datetime, fcntl, json, os, shutil, subprocess
from pathlib import Path
from common import ART, REPO, save, digest, storage, read_manifest

HERE = Path(__file__).resolve().parent


def start(deadline=None, reuse_targets=None):
    storage()
    lock = (ART / "active.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPU has an active compute process")
    for check in ["pipeline-passed.json", "contracts-passed.json"]:
        assert (ART / "preflight" / check).exists(), f"Missing preflight gate: {check}"
    assert json.loads(
        (ART / "preflight/adapter/teacher-omi-adapter-check.json").read_text()
    )["executed"]
    manifest = read_manifest()
    cfg = json.loads((HERE / "pilot.json").read_text())
    if deadline:
        end = datetime.datetime.fromisoformat(deadline)
        cfg["duration_seconds"] = min(
            cfg["duration_seconds"],
            int((end - datetime.datetime.now(datetime.timezone.utc)).total_seconds()),
        )
        assert cfg["duration_seconds"] > 600, (
            "Not enough time before the requested deadline"
        )
        cfg["outer_deadline_utc"] = end.isoformat()
    if shutil.disk_usage(ART).free < 60 * 1024**3:
        raise RuntimeError("Need at least 60 GiB free on data disk")
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime(
        "pilot-8h-%Y%m%dT%H%M%SZ"
    )
    run = ART / "runs" / run_id
    run.mkdir(parents=True)
    code = run / "code"
    code.mkdir()
    for p in HERE.iterdir():
        if p.is_file() and (
            p.suffix in [".py", ".json", ".md", ".txt"] or p.name == "python"
        ):
            shutil.copy2(p, code / p.name)
    (code / "python").chmod(0o755)
    save(run / "config.json", cfg)
    for precision in ["fp", "ternary"]:
        profile = json.loads((ART / (precision + "-profile.json")).read_text())
        assert profile["parameters"] == 603806720 and profile["cuda_verified"]
        shutil.copy2(
            ART / (precision + "-profile.json"), run / (precision + "-profile.json")
        )
    provenance = dict(
        repo_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPO
        ).strip(),
        manifest_sha256=digest(ART / "datasets/pilot/manifest.json"),
        source_hashes={p.name: digest(p) for p in code.iterdir()},
        dataset_summary=manifest["summary"],
        limitations=manifest["limitations"],
        teachers=cfg["teachers"],
    )
    provenance["input_lock_sha256"] = digest(ART / "input-lock.json")
    shutil.copy2(ART / "input-lock.json", run / "input-lock.json")
    save(run / "provenance.json", provenance)
    shutil.copy2(ART / "datasets/pilot/manifest.json", run / "manifest.json")
    if reuse_targets:
        prior = Path(reuse_targets)
        prior_provenance = json.loads((prior / "provenance.json").read_text())
        assert prior_provenance["manifest_sha256"] == provenance["manifest_sha256"]
        assert prior_provenance["teachers"] == cfg["teachers"]
        assert digest(prior / "code/teachers.py") == digest(code / "teachers.py")
        for p in list(prior.glob("teacher-*.json")) + [prior / "targets.json"]:
            shutil.copy2(p, run / p.name)
        save(
            run / "target-reuse.json",
            dict(
                source=str(prior),
                reason="Same pinned inputs and teacher code; revised throughput allocation, no accuracy tuning",
            ),
        )
    unit = "stt-distill-" + run_id
    cmd = [
        "systemd-run",
        "--user",
        "--unit=" + unit,
        "--description=Eight hour ASR ternary distillation pilot",
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
        str(code / "run.py"),
        str(run),
    ]
    # Hold the launch guard until the unit and latest-run pointer exist.
    # The supervisor then acquires the guard for its entire lifetime.
    subprocess.run(cmd, check=True)
    save(ART / "latest.json", dict(run=str(run), unit=unit))
    save(run / "service.json", dict(unit=unit, command=cmd))
    fcntl.flock(lock, fcntl.LOCK_UN)
    print(json.dumps(dict(run=str(run), unit=unit)), flush=True)


def status(stop=False):
    latest = json.loads((ART / "latest.json").read_text())
    run = Path(latest["run"])
    if stop:
        subprocess.run(["systemctl", "--user", "stop", latest["unit"]], check=True)
    subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            latest["unit"],
            "--property=ActiveState,SubState,Result,RuntimeMaxUSec,ExecMainStartTimestamp",
        ]
    )
    p = run / "state.json"
    print(p.read_text() if p.exists() else "Service is starting")
    for p in sorted(run.glob("*/progress.json")):
        print(p.parent.name, p.read_text())
    print("Report:", run / "REPORT.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "status", "stop"])
    p.add_argument(
        "--deadline",
        help="Optional absolute ISO UTC deadline for an existing test window",
    )
    p.add_argument("--reuse-targets", type=Path)
    a = p.parse_args()
    start(a.deadline, a.reuse_targets) if a.action == "start" else status(
        a.action == "stop"
    )
