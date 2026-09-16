"""Out-of-process watchdog; never restarts a collapsed model automatically."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .notifications import Notifier
from .storage import ensure_artifact_path, write_json


def _stop_group(process: subprocess.Popen, grace: float) -> None:
    # The child owns its session/process group. No GPU PID enumeration or
    # process-name matching is used: unrelated experiments are never signaled.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Also reap children that outlived the worker's own graceful shutdown.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _timestamp(record: Mapping[str, Any]) -> float:
    value = record.get("time", record.get("timestamp", record.get("updated_at")))
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    if not math.isfinite(result):
        raise ValueError("Heartbeat timestamp must be finite")
    return result


def run_supervised(
    command: Sequence[str],
    run_dir: str | Path,
    timeout_seconds: float = 900,
    *,
    poll_interval: float = 1,
    terminate_grace_seconds: float = 15,
    stage_timeouts: Mapping[str, float] | None = None,
    notification_config: Mapping[str, Any] | None = None,
) -> int:
    """Launch a worker and return its conventional exit status (124 on stall).

    Worker writes atomic ``heartbeat.json`` with ``time`` (Unix seconds),
    ``stage`` and its ``pid`` at real progress points. A timer thread emitting
    heartbeats independently of training would conceal a hung CUDA operation.
    Long preprocessing/evaluation/checkpoint stages can have explicit timeouts.
    """
    if not command:
        raise ValueError("Supervisor command cannot be empty")
    if timeout_seconds <= 0 or poll_interval <= 0 or terminate_grace_seconds < 0:
        raise ValueError("Watchdog timeout/poll must be positive and grace nonnegative")
    stage_timeouts = dict(stage_timeouts or {})
    if any(float(value) <= 0 for value in stage_timeouts.values()):
        raise ValueError("Every stage timeout must be positive")
    run_dir = Path(ensure_artifact_path(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    notifier = Notifier(run_dir, notification_config)
    notifier.check_delivery_config()
    status_path = run_dir / "supervisor_status.json"
    previous_handlers: dict[int, Any] = {}
    requested_signal: list[int] = []

    def handle_signal(number: int, _frame: Any) -> None:
        requested_signal.append(number)

    parent_pid = os.getpid()
    preexec = None
    if sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)

        def child_setup() -> None:
            # If the watchdog is SIGKILLed, the CUDA-owning worker still gets
            # SIGTERM. A parent-death race is handled before executing training.
            if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
                os._exit(125)
            if os.getppid() != parent_pid:
                os.kill(os.getpid(), signal.SIGTERM)

        preexec = child_setup

    started_wall = time.time()
    worker_status_path = run_dir / "status.json"
    previous_status_identity = None
    try:
        stat = worker_status_path.stat()
        previous_status_identity = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    except OSError:
        pass
    last_progress = time.monotonic()
    last_timestamp = started_wall - 1
    stage = "startup"
    process = None
    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[number] = signal.signal(number, handle_signal)
        ensure_artifact_path(run_dir / "worker.log")
        with (run_dir / "worker.log").open("ab", buffering=0) as worker_log:
            env = {**os.environ, "BINARY_STT_SUPERVISED": "1", "PYTHONUNBUFFERED": "1"}
            process = subprocess.Popen(list(command), stdout=worker_log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=env, preexec_fn=preexec)
            write_json(status_path, {"status": "running", "pid": process.pid, "started_at": started_wall})
            while True:
                returncode = process.poll()
                if returncode is not None:
                    exit_code = returncode if returncode >= 0 else 128 - returncode
                    worker_status = None
                    try:
                        stat = worker_status_path.stat()
                        identity = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
                        if identity != previous_status_identity:
                            with worker_status_path.open() as handle:
                                worker_record = json.load(handle)
                                worker_status = worker_record.get("status", worker_record.get("state"))
                    except (OSError, ValueError, AttributeError):
                        pass
                    unexpected_clean_exit = exit_code == 0 and worker_status not in ("completed", "stopped")
                    if unexpected_clean_exit:
                        exit_code = 70
                    failed = exit_code != 0
                    detail = ("Training worker exited without a completed/stopped status record"
                              if unexpected_clean_exit else f"Training worker exited with status {exit_code}"
                              + (f" (signal {-returncode}; SIGKILL may indicate OOM or external termination)"
                                 if returncode == -signal.SIGKILL else ""))
                    write_json(status_path, {"status": "failed" if failed else worker_status, "exit_code": exit_code,
                                             "pid": process.pid, "ended_at": time.time(), "detail": detail})
                    # Clean up any descendants if the worker exits first.
                    _stop_group(process, terminate_grace_seconds)
                    if failed:
                        notifier.notify("worker_failed", detail, exit_code=exit_code, stage=stage)
                    return exit_code
                if requested_signal:
                    received = requested_signal[0]
                    _stop_group(process, terminate_grace_seconds)
                    write_json(status_path, {"status": "stopped", "signal": received, "pid": process.pid,
                                             "ended_at": time.time()})
                    notifier.notify("supervisor_stopped", f"Watchdog received signal {received}; worker group stopped.")
                    return 128 + received
                try:
                    with (run_dir / "heartbeat.json").open() as handle:
                        heartbeat = json.load(handle)
                    stamp = _timestamp(heartbeat)
                    if (heartbeat.get("pid", process.pid) == process.pid and stamp >= started_wall - 1
                            and last_timestamp < stamp <= time.time() + 60):
                        last_timestamp = stamp
                        # A stale heartbeat must not reset the timeout just
                        # because the supervisor only now read the file.
                        last_progress = time.monotonic() - max(0.0, time.time() - stamp)
                        stage = str(heartbeat.get("stage", "training"))
                except (OSError, ValueError, TypeError, AttributeError):
                    pass
                limit = float(stage_timeouts.get(stage, timeout_seconds))
                if time.monotonic() - last_progress > limit:
                    detail = f"No worker progress heartbeat for {limit:g} seconds during {stage}; worker group stopped."
                    _stop_group(process, terminate_grace_seconds)
                    write_json(status_path, {"status": "failed", "reason": "heartbeat_timeout", "stage": stage,
                                             "exit_code": 124, "pid": process.pid, "ended_at": time.time()})
                    notifier.notify("heartbeat_timeout", detail, stage=stage, timeout_seconds=limit)
                    return 124
                time.sleep(min(poll_interval, limit))
    except BaseException as exc:
        if process is not None:
            _stop_group(process, terminate_grace_seconds)
        try:
            write_json(status_path, {"status": "failed", "reason": "supervisor_error",
                                     "error_type": type(exc).__name__, "ended_at": time.time()})
        except Exception:
            pass
        notifier.notify("supervisor_failed", f"Watchdog failed: {type(exc).__name__}; worker group stopped.")
        raise
    finally:
        for number, previous in previous_handlers.items():
            signal.signal(number, previous)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--terminate-grace-seconds", type=float, default=15)
    parser.add_argument("--stage-timeouts", default="{}", help="JSON mapping of stage names to seconds")
    parser.add_argument("--email-to")
    parser.add_argument("--no-desktop", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run_supervised(command, args.run_dir, args.timeout_seconds,
                          terminate_grace_seconds=args.terminate_grace_seconds,
                          stage_timeouts=json.loads(args.stage_timeouts),
                          notification_config={"email_to": args.email_to, "desktop": not args.no_desktop})


if __name__ == "__main__":
    raise SystemExit(main())
