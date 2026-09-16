"""Time-bounded pilot supervisor. The systemd service enforces the outer deadline."""

import argparse, datetime, fcntl, json, os, signal, subprocess, time, traceback
from pathlib import Path
from common import ART, save, digest, read_manifest, storage
from report import report

stop = False


def halt(*args):
    global stop
    stop = True


def utc(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat()


def run_pilot(run):
    storage()
    lock = (ART / "active.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    os.environ["WILDERNESS_STT_MANIFEST"] = str(run / "manifest.json")
    cfg = json.loads((run / "config.json").read_text())
    code = run / "code"
    py = code / "python"
    started = time.time()
    deadline = started + cfg["duration_seconds"]
    if cfg.get("outer_deadline_utc"):
        deadline = min(
            deadline,
            datetime.datetime.fromisoformat(cfg["outer_deadline_utc"]).timestamp(),
        )
    state = dict(
        status="running",
        stage="preflight",
        started_utc=utc(started),
        deadline_utc=utc(deadline),
        run=str(run),
        pid=os.getpid(),
        stages=[],
    )

    def update(**kw):
        state.update(kw)
        state["updated_utc"] = utc(time.time())
        save(run / "state.json", state)

    def command(stage, args, seconds):
        if stop or time.time() >= deadline - 30:
            raise TimeoutError("Outer time budget exhausted")
        update(stage=stage)
        end = min(deadline - 30, time.time() + seconds)
        with (run / (stage + ".log")).open("a") as log:
            child = subprocess.Popen(
                [str(py), *map(str, args)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            update(child_pid=child.pid)
            reason = None
            try:
                while child.poll() is None:
                    try:
                        telemetry = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=5,
                        ).strip()
                        with (run / "gpu.csv").open("a") as f:
                            f.write(stage + "," + telemetry + "\n")
                    except Exception:
                        pass
                    if stop or time.time() >= end:
                        reason = "cancelled" if stop else "time_budget"
                        os.killpg(child.pid, signal.SIGTERM)
                        break
                    time.sleep(5)
                if reason:
                    try:
                        child.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                else:
                    child.wait()
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            result = dict(
                stage=stage,
                exit_code=child.returncode,
                reason=reason,
                finished_utc=utc(time.time()),
            )
            state["stages"].append(result)
            update(child_pid=None)
            report(run)
            return result

    error = None
    try:
        assert (
            digest(ART / "datasets/pilot/manifest.json")
            == json.loads((run / "provenance.json").read_text())["manifest_sha256"]
        )
        teacher_end = min(deadline - 1800, time.time() + cfg["teacher_seconds"])
        for i, kind in enumerate(["omi", "whisper"]):
            allocation = (teacher_end - time.time()) / (2 - i)
            result = command(
                "teacher-" + kind,
                [code / "teachers.py", kind, "--run", run],
                allocation,
            )
            if result["exit_code"] != 0 and result["reason"] != "time_budget":
                raise RuntimeError(f"Teacher {kind} failed; see its log")
        result = command(
            "teacher-calibration", [code / "teachers.py", "combine", "--run", run], 120
        )
        if result["exit_code"]:
            raise RuntimeError("Teacher calibration or coverage gate failed")
        profiles = {
            p: json.loads((run / (p + "-profile.json")).read_text())
            for p in ["fp", "ternary"]
        }
        # Measured real batches include gradient accumulation; budget adds
        # measured checkpoint cost and a ten-percent timing margin.
        rates = {p: v["calibrated_update_seconds"] for p, v in profiles.items()}
        remaining = deadline - time.time() - cfg["report_reserve_seconds"] - 4 * 180
        common_steps = min(
            cfg["max_steps"],
            int(remaining / sum(rates[a["precision"]] for a in cfg["arms"])),
        )
        common_steps = max(0, common_steps // 100 * 100)
        if common_steps < 500:
            raise RuntimeError("Insufficient time for a matched pilot")
        save(
            run / "allocation.json",
            dict(
                common_steps=common_steps,
                conservative_seconds_per_update=rates,
                remaining_training_seconds=remaining,
                report_reserve_seconds=cfg["report_reserve_seconds"],
            ),
        )
        for i, a in enumerate(cfg["arms"]):
            future = cfg["arms"][i:]
            remaining = (
                deadline
                - time.time()
                - cfg["report_reserve_seconds"]
                - 180 * len(future)
            )
            allocation = max(
                180,
                remaining
                * rates[a["precision"]]
                / sum(rates[x["precision"]] for x in future),
            )
            command(
                a["name"] + "-train",
                [
                    code / "train.py",
                    run,
                    a["name"],
                    "--steps",
                    common_steps,
                    "--seconds",
                    allocation,
                ],
                allocation + 100,
            )
            p = run / a["name"]
            folder = next(
                (
                    p / k
                    for k in ["export", "checkpoint", "checkpoint-previous"]
                    if (p / k / "export.json").exists()
                ),
                p / "checkpoint",
            )
            if (folder / "export.json").exists():
                command(
                    a["name"] + "-eval",
                    [code / "evaluate.py", folder, p / "development.json"],
                    180,
                )
            if stop:
                break
        report(run)
        out = json.loads((run / "report.json").read_text())
        update(
            status="cancelled"
            if stop
            else "completed"
            if out["matched_four_arm_comparison"]
            and all(v["development"] for v in out["arms"].values())
            else "partial",
            stage="finished",
            finished_utc=utc(time.time()),
        )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        update(
            status="cancelled" if stop else "failed",
            error=error,
            finished_utc=utc(time.time()),
        )
    finally:
        report(run)
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    a = p.parse_args()
    run_pilot(a.run)
