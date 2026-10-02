#!/usr/bin/python3
"""macguard: run one job on the shared Mac under memory, time and load guards. See DESIGN.md "Implementation decisions".

    macguard --rss-cap SIZE --timeout SECONDS [--min-free 40] [--abort-free 25] [--max-swap-growth 1G]
             [--threads 4] -- CMD ARGS...

Entry point: the POSIX-sh front end ios/macguard (this file is the implementation). The front end records
the caller's environment (the names of all variables, and the state of those the /usr/bin/python3 shim and
Python's start-up inject: SDKROOT, CPATH, LIBRARY_PATH, MANPATH, LC_CTYPE, __CF_USER_TEXT_ENCODING); the job
runs in exactly the caller's environment plus the thread caps (caller_environment).

Lock. One guarded job at a time: macguard takes fcntl.flock on the permanent file
<dir>/macguard.lockfile (refusing, exit 3, if it is held) and never unlocks it explicitly. The locked file
description is inherited by the sentinel and by the job (pass_fds), so the lock is held as long as the
supervisor, the sentinel, a watcher, or any job process that kept the descriptor is alive. It then refuses,
exit 3, unless the system-wide free memory percentage (memory_pressure -Q) is at least --min-free.

Control protocol. Before anything is launched, the supervisor forks a sentinel in its own session that holds
the lock and watches the supervisor (control pipe EOF, getppid every second). The supervisor tells it
"launch", then starts the job as `macguard.py --exec-after-gate GATE CTRL -- CMD`. That wrapper first
registers its own pid (= its process group) with the sentinel on the inherited control pipe and closes it,
so cleanup ownership exists from the job's first instruction on; then it blocks on the gate pipe. The
sentinel acknowledges "ok <pgid>" to the supervisor, which checks it against the pid it started and only
then opens the gate (the wrapper execs CMD). Every failure in this handshake enters cleanup: a missing or
wrong acknowledgement makes the supervisor kill the gated group (exit 125); a supervisor that dies after
"launch" leaves the sentinel waiting for the registration (or the pipe's EOF, which proves no wrapper
exists) and then killing the group; an acknowledgement the sentinel cannot deliver (supervisor gone) sends
it to cleanup as well. A wrapper whose gate pipe closes without "g" exits without running CMD.

Run. CMD runs in a new session and process group (pgid == pid, verified distinct from macguard's), under
nice 10, with OMP/BLAS/vecLib thread caps. Every second: the sentinel's liveness and the summed RSS of the
group from ps; every 5 s: free memory and swap use. Abort (exit 124) when the group RSS exceeds --rss-cap,
the job runs past --timeout, free memory falls below --abort-free, swap use grows by more than
--max-swap-growth since the start, the sentinel exits or is killed, or any monitoring or logging step fails
(fail closed: a failed or malformed ps, memory_pressure or sysctl reading, any parse or I/O error).

Cleanup runs for every outcome, with every signalling and bookkeeping step guarded: SIGTERM to the group,
5 s grace, then SIGKILL repeated until the kernel reports the group empty (os.killpg(pgid, 0) ->
ProcessLookupError; any other answer counts as alive), for at most 60 s. Only a group verified empty is
released. If it cannot be verified, macguard exits 125 and hands the group over: to the sentinel ("watch"),
or, if the sentinel cannot be reached, to a newly forked detached watcher holding the lock; if even that
fork fails, the supervisor keeps killing itself. Whoever holds it keeps the lock and keeps sending SIGKILL
until the group is empty. A leader that exits normally with descendants left in its group gets the same
cleanup. The sentinel catches its own errors: with a registered group it cleans up before it exits (status
1 after an error), so an erroring sentinel never leaves a live job behind an unlocked file. If the
supervisor dies (even by SIGKILL), the sentinel terminates the group the same way and then releases the lock.

Exit status: CMD's status if it ran to completion and everything after it succeeded (128 + signal if a signal
ended it); 124 the guard aborted the job (resource, timeout, sentinel lost, monitoring or logging failure),
whatever CMD returned; 130 macguard was interrupted (SIGINT/SIGTERM/SIGHUP; the job is cleaned up first);
125 the guard's own machinery failed: launch or handshake, cleanup errors, group not verified empty (the
lock then stays with the sentinel or a watcher), a sentinel that ended abnormally, or a failure while
finalizing (end probes or end log) when the status would otherwise be the job's own (an abort or interrupt
keeps 124/130) — never 0 in any of these; 3 refused to start; 2 usage error.

Residual risk (accepted, not hardened further): containment rests on the supervisor, the sentinel and the
inherited lock. If BOTH the supervisor and the sentinel are SIGKILLed, nothing monitors the job any more;
the lock is then held only as long as some job process keeps the inherited descriptor, so a job that closes
inherited descriptors (e.g. Python subprocess with close_fds) loses containment and a second job may start.

Test hooks (tests/macguard_tests.sh only; inert unless set): MACGUARD_TEST_FAULT, a comma-separated list of
"killpg-eperm" (os.killpg with a real signal raises PermissionError), "die-before-ack" (the supervisor
SIGKILLs itself right after starting the gated wrapper) and "end-log-fail" (the end log raises OSError);
with any fault set the SIGKILL deadline is 3 s instead of 60 s.

Log: <dir>/logs/macguard.log (dir = $MACGUARD_DIR, default
/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios): start and end, kill reason, peak group RSS, exit
status, free memory, swap growth, load average and the top 5 CPU processes at start and end, and the
sentinel's and watcher's actions. RSS polling can miss short peaks and memory used by system services on the
job's behalf. Standard library only (the Mac's /usr/bin/python3 is 3.9). The probes are macOS commands; on
Linux (tests only) /proc/meminfo stands in for memory and swap.
"""
from __future__ import annotations

import errno
import fcntl
import os
import select
import signal
import subprocess
import sys
import time

EXIT_USAGE, EXIT_REFUSED, EXIT_ABORTED, EXIT_INTERNAL, EXIT_INTERRUPTED = 2, 3, 124, 125, 130
DARWIN = sys.platform == "darwin"
DEFAULT_DIR = "/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios"
FAULTS = frozenset(f for f in os.environ.get("MACGUARD_TEST_FAULT", "").split(",") if f)
GRACE_S, PROBE_TIMEOUT_S, ACK_TIMEOUT_S = 5.0, 15.0, 10.0
KILL_DEADLINE_S = 3.0 if FAULTS else 60.0
USAGE = ("usage: macguard --rss-cap SIZE --timeout SECONDS [--min-free PCT] [--abort-free PCT] "
         "[--max-swap-growth SIZE] [--threads N] -- CMD ...")
THREAD_VARS = ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")
RESTORED = ("SDKROOT", "CPATH", "LIBRARY_PATH", "MANPATH", "LC_CTYPE", "__CF_USER_TEXT_ENCODING")


class ProbeError(Exception):
    pass


# --- arguments ------------------------------------------------------------------------------------

def to_kb(size: str) -> int:
    """SIZE with optional K/M/G suffix (binary units; no suffix = bytes) -> KB."""
    units = {"k": 1, "m": 1024, "g": 1024 * 1024}
    if size and size[-1].lower() in units:
        return int(size[:-1]) * units[size[-1].lower()]
    return int(size) // 1024


def parse(argv: list) -> tuple:
    opts = {"rss-cap": None, "timeout": None, "min-free": "40", "abort-free": "25", "max-swap-growth": "1G",
            "threads": "4"}
    i = 0
    while i < len(argv) and argv[i] != "--":
        key = argv[i][2:] if argv[i].startswith("--") else None
        if key not in opts or i + 1 >= len(argv):
            raise ValueError(argv[i])
        opts[key] = argv[i + 1]
        i += 2
    cmd = argv[i + 1:] if i < len(argv) else []
    if not cmd or opts["rss-cap"] is None or opts["timeout"] is None:
        raise ValueError("missing --rss-cap, --timeout or CMD")
    parsed = {"cap_kb": to_kb(opts["rss-cap"]), "cap": opts["rss-cap"], "timeout": int(opts["timeout"]),
              "min_free": int(opts["min-free"]), "abort_free": int(opts["abort-free"]),
              "growth_kb": to_kb(opts["max-swap-growth"]), "growth": opts["max-swap-growth"],
              "threads": int(opts["threads"])}
    if min(parsed["cap_kb"], parsed["timeout"], parsed["threads"]) <= 0:
        raise ValueError("cap, timeout and threads must be positive")
    return parsed, cmd


# --- probes (every failure or malformed reading raises ProbeError) ----------------------------------

def run_probe(args: list) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ProbeError(f"{args[0]}: {exc}") from exc
    if out.returncode != 0:
        raise ProbeError(f"{args[0]} exited {out.returncode}")
    return out.stdout


def free_pct() -> int:
    try:
        if DARWIN:
            lines = [l for l in run_probe(["memory_pressure", "-Q"]).splitlines() if "free percentage" in l]
            if len(lines) != 1:
                raise ProbeError("memory_pressure: no single free percentage line")
            value = int(lines[0].split(":")[1].strip().rstrip("%"))
        else:
            info = _meminfo()
            value = int(100 * info["MemAvailable"] / info["MemTotal"])
    except (ValueError, IndexError, KeyError, ZeroDivisionError) as exc:
        raise ProbeError(f"free memory: {exc!r}") from exc
    if not 0 <= value <= 100:
        raise ProbeError(f"free memory {value}% out of range")
    return value


def swap_used_kb() -> int:
    try:
        if DARWIN:
            fields = run_probe(["sysctl", "-n", "vm.swapusage"]).split()
            value = fields[fields.index("used") + 2]
            scale = {"K": 1, "M": 1024, "G": 1024 * 1024}[value[-1]]
            used = int(float(value[:-1]) * scale)
        else:
            info = _meminfo()
            used = info["SwapTotal"] - info["SwapFree"]
    except (ValueError, IndexError, KeyError) as exc:
        raise ProbeError(f"swap usage: {exc!r}") from exc
    if used < 0:
        raise ProbeError(f"swap usage {used}KB negative")
    return used


def _meminfo() -> dict:
    try:
        with open("/proc/meminfo") as handle:
            return {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in handle)}
    except (OSError, ValueError, IndexError) as exc:
        raise ProbeError(f"/proc/meminfo: {exc!r}") from exc


def group_rss_kb(pgid: int) -> int:
    """Summed RSS (KB) of the processes in process group pgid; every ps row must parse."""
    total, rows = 0, 0
    for line in run_probe(["ps", "-A", "-o", "pid=,pgid=,rss="]).splitlines():
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) != 3:
                raise ValueError(line)
            pid, group, rss = (int(p) for p in parts)
        except ValueError as exc:
            raise ProbeError(f"malformed ps row {line!r}") from exc
        if pid <= 0 or rss < 0:
            raise ProbeError(f"malformed ps row {line!r}")
        rows += 1
        if group == pgid:
            total += rss
    if rows == 0:
        raise ProbeError("ps listed no processes")
    return total


def group_alive(pgid: int) -> bool:
    """Whether any process (zombies included) is in group pgid, from the kernel; anything but ESRCH = alive."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def kill_group(pgid: int, sig: int) -> None:
    """Send sig to the group; errors other than "no such group" are swallowed (verification decides)."""
    try:
        if "killpg-eperm" in FAULTS:
            raise PermissionError(errno.EPERM, "injected by MACGUARD_TEST_FAULT")
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass
    except OSError:
        pass


def escalate(pgid: int, reap=lambda: None, kill_deadline: float | None = None) -> bool:
    """SIGTERM, up to GRACE_S, then SIGKILL every 0.5 s until the group is verified empty (True) or
    kill_deadline s pass (False; None = KILL_DEADLINE_S; float("inf") = until empty). Never raises."""
    deadline_s = KILL_DEADLINE_S if kill_deadline is None else kill_deadline

    def gone() -> bool:
        try:
            reap()
        except Exception:
            pass
        return not group_alive(pgid)

    kill_group(pgid, signal.SIGTERM)
    end = time.monotonic() + GRACE_S
    while time.monotonic() < end:
        if gone():
            return True
        time.sleep(0.1)
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        kill_group(pgid, signal.SIGKILL)
        for _ in range(5):
            if gone():
                return True
            time.sleep(0.1)
    return gone()


FOREVER = float("inf")


# --- logging ----------------------------------------------------------------------------------------

def log(path: str, tag: str, message: str, echo: bool = True) -> None:
    """Append one line to the log (OSError propagates) and echo it to stderr (errors ignored)."""
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{tag}] {message}"
    with open(path, "a") as handle:
        handle.write(line + "\n")
    if echo:
        try:
            print(f"macguard: {message}", file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass


def log_safe(path: str, tag: str, message: str, echo: bool = True) -> None:
    try:
        log(path, tag, message, echo)
    except OSError:
        pass


# --- sentinel and watcher -------------------------------------------------------------------------------

def _detach(keep: set) -> None:
    os.setsid()
    for signum in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
        signal.signal(signum, signal.SIG_IGN)
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    for fd in range(3, 1024):
        if fd not in keep:
            try:
                os.close(fd)
            except OSError:
                pass


def sentinel(ctrl_r: int, ack_w: int, lock_fd: int, supervisor: int, log_path: str) -> int:
    """Child of the supervisor in its own session; holds the lock until the job group is gone (module doc).
    Returns its exit status: 0 normally, 1 after an internal error (the group is cleaned up first)."""
    _detach({0, 1, 2, ctrl_r, ack_w, lock_fd})
    tag = f"sentinel {os.getpid()}"
    pgid, buffer, launched, supervisor_gone, error = None, b"", False, False, None
    while True:
        message = None
        try:
            ready, _, _ = select.select([ctrl_r], [], [], 1.0)
            if ready:
                data = os.read(ctrl_r, 4096)
                if not data:
                    message = "eof"  # no writer left: supervisor and any wrapper are gone or closed it
                buffer += data
                while b"\n" in buffer and message is None:
                    line, buffer = buffer.split(b"\n", 1)
                    words = line.decode(errors="replace").split()
                    if words[:1] == ["launch"]:
                        launched = True
                    elif words[:1] == ["pgid"] and len(words) == 2:
                        pgid = int(words[1])
                        try:
                            os.write(ack_w, f"ok {pgid}\n".encode())
                        except OSError as exc:  # supervisor gone between registration and ACK: clean up
                            log_safe(log_path, tag, f"cannot acknowledge group {pgid} ({exc!r}); cleaning up", False)
                            message = "orphan"
                    elif words[:1] in (["done"], ["watch"]):
                        message = words[0]
            if message is None and not supervisor_gone and os.getppid() != supervisor:
                supervisor_gone = True
            if message is None and supervisor_gone and (pgid is not None or not launched):
                message = "orphan"  # with a launch but no registration yet, wait for it (or for EOF)
        except Exception as exc:  # any internal failure: clean up whatever is registered, then exit 1
            error = exc
            message = "error"
        if message == "done":
            return 0
        if message is None:
            continue
        if pgid is None:
            log_safe(log_path, tag, f"{message}: no job group registered; nothing to clean up", False)
            return 1 if error else 0
        why = {"watch": f"supervisor could not verify group {pgid} empty; killing until it is",
               "eof": f"control pipe closed; terminating job group {pgid}",
               "orphan": f"supervisor {supervisor} gone; terminating job group {pgid}",
               "error": f"internal error {error!r}; terminating job group {pgid}"}[message]
        log_safe(log_path, tag, why, False)
        escalate(pgid, kill_deadline=FOREVER)
        log_safe(log_path, tag, f"group {pgid} empty; releasing the lock", False)
        return 1 if error else 0


def spawn_watcher(pgid: int, lock_fd: int, log_path: str) -> bool:
    """Fork a detached watcher that holds the lock and kills group pgid until it is empty."""
    try:
        pid = os.fork()
    except OSError:
        return False
    if pid == 0:
        try:
            _detach({0, 1, 2, lock_fd})
            if os.fork() != 0:
                os._exit(0)
            log_safe(log_path, f"watcher {os.getpid()}", f"killing group {pgid} until it is empty", False)
            escalate(pgid, kill_deadline=FOREVER)
            log_safe(log_path, f"watcher {os.getpid()}", f"group {pgid} empty; releasing the lock", False)
        finally:
            os._exit(0)
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass
    return True


# --- supervisor ---------------------------------------------------------------------------------------

class Guard:
    def __init__(self, opts: dict, cmd: list, directory: str) -> None:
        self.opts, self.cmd, self.dir = opts, cmd, directory
        self.log_path = os.path.join(directory, "logs", "macguard.log")
        self.tag = str(os.getpid())
        self.interrupted = 0

    def say(self, message: str) -> None:
        log(self.log_path, self.tag, message)

    def say_safe(self, message: str) -> None:
        log_safe(self.log_path, self.tag, message)

    def snapshot(self, label: str, free: object, swap: object, strict: bool) -> None:
        say = self.say if strict else self.say_safe
        load = " ".join(f"{x:.2f}" for x in os.getloadavg())
        say(f"{label}: free {free}% swap_used {swap}KB load {load}")
        try:
            args = (["ps", "-A", "-r", "-o", "pid=,pcpu=,rss=,comm="] if DARWIN
                    else ["ps", "-A", "-o", "pid=,pcpu=,rss=,comm=", "--sort=-pcpu"])
            for line in run_probe(args).splitlines()[:5]:
                say(f"{label}: top {line.strip()}")
        except ProbeError as exc:
            say(f"{label}: top unavailable ({exc})")

    def on_signal(self, signum, frame) -> None:
        self.interrupted = signum

    def run(self) -> int:
        lock_fd = os.open(os.path.join(self.dir, "macguard.lockfile"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(lock_fd)
            if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
                self.say_safe(f"refused: lock held by another guarded job: {' '.join(self.cmd)}")
                return EXIT_REFUSED
            raise
        try:
            free0, swap0 = free_pct(), swap_used_kb()
        except ProbeError as exc:
            self.say_safe(f"refused: cannot read memory state ({exc}): {' '.join(self.cmd)}")
            return EXIT_REFUSED
        if free0 < self.opts["min_free"]:
            self.say_safe(f"refused: free memory {free0}% < {self.opts['min_free']}%: {' '.join(self.cmd)}")
            return EXIT_REFUSED
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, self.on_signal)
        ctrl_r, ctrl_w = os.pipe()
        ack_r, ack_w = os.pipe()
        child = os.fork()
        if child == 0:
            code = 1
            try:
                code = sentinel(ctrl_r, ack_w, lock_fd, os.getppid(), self.log_path)
            finally:
                os._exit(code)
        os.close(ctrl_r)
        os.close(ack_w)
        return self.supervise(lock_fd, ctrl_w, ack_r, child, free0, swap0)

    def sentinel_exited(self, sentinel_pid: int):
        """The sentinel's wait status if it has exited (reaped now), else None; an unknowable state counts as exited."""
        try:
            pid, status = os.waitpid(sentinel_pid, os.WNOHANG)
        except ChildProcessError:
            return -1
        return status if pid else None

    def supervise(self, lock_fd: int, ctrl_w: int, ack_r: int, sentinel_pid: int, free0: int, swap0: int) -> int:
        opts = self.opts
        start = time.monotonic()
        reason, internal, proc, pgid, peak, phase = None, None, None, None, 0, "start"
        sentinel_status, problems = None, []  # problems: machinery failures during cleanup and finalization
        try:
            self.say(f"start: cap {opts['cap']} ({opts['cap_kb']}KB) timeout {opts['timeout']}s threads "
                     f"{opts['threads']} min_free {opts['min_free']}% abort_free {opts['abort_free']}% "
                     f"max_swap_growth {opts['growth']} sentinel {sentinel_pid}: {' '.join(self.cmd)}")
            self.snapshot("start", free0, swap0, strict=True)
            env = dict(os.environ)
            for key in THREAD_VARS:
                env[key] = str(opts["threads"])
            phase = "launch"
            os.write(ctrl_w, b"launch\n")
            gate_r, gate_w = os.pipe()
            try:
                proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--exec-after-gate", str(gate_r),
                                         str(ctrl_w), "--", *self.cmd], env=env, start_new_session=True,
                                        pass_fds=(lock_fd, gate_r, ctrl_w), preexec_fn=lambda: os.nice(10))
                pgid = proc.pid
                os.close(gate_r)
                if "die-before-ack" in FAULTS:
                    os.kill(os.getpid(), signal.SIGKILL)
                if os.getpgid(pgid) != pgid or pgid == os.getpgrp():
                    raise RuntimeError(f"job is not in its own process group (pgid {os.getpgid(pgid)})")
                ready, _, _ = select.select([ack_r], [], [], ACK_TIMEOUT_S)
                ack = os.read(ack_r, 64) if ready else b""
                if ack != f"ok {pgid}\n".encode():
                    raise RuntimeError(f"sentinel acknowledgement {ack!r} for group {pgid}")
                os.write(gate_w, b"g")
            finally:
                os.close(gate_w)
            phase = "monitor"
            tick = 0
            while True:
                if self.interrupted:
                    reason = f"macguard received {signal.Signals(self.interrupted).name}"
                    break
                if proc.poll() is not None:
                    break
                try:
                    sentinel_status = self.sentinel_exited(sentinel_pid)
                    if sentinel_status is not None:
                        reason = f"sentinel {sentinel_pid} lost (wait status {sentinel_status})"
                        break
                    rss = group_rss_kb(pgid)
                    peak = max(peak, rss)
                    elapsed = time.monotonic() - start
                    if rss > opts["cap_kb"]:
                        reason = f"rss {rss}KB > cap {opts['cap_kb']}KB"
                    elif elapsed > opts["timeout"]:
                        reason = f"timeout {elapsed:.1f}s > {opts['timeout']}s"
                    elif tick % 5 == 0:
                        free, swap = free_pct(), swap_used_kb()
                        if free < opts["abort_free"]:
                            reason = f"system free memory {free}% < {opts['abort_free']}%"
                        elif swap - swap0 > opts["growth_kb"]:
                            reason = f"swap grew {swap - swap0}KB > {opts['growth_kb']}KB"
                except Exception as exc:  # monitoring and logging failures abort the job (fail closed)
                    reason = f"monitoring failed: {exc!r}"
                if reason:
                    break
                tick += 1
                time.sleep(1.0)
        except Exception as exc:  # start-up logging or monitoring failure: abort; launch/handshake failure: internal
            if phase == "launch":
                internal = repr(exc)
            else:
                reason = f"{'logging' if phase == 'start' else 'monitoring'} failed: {exc!r}"
        finally:
            empty = self.cleanup(proc, pgid, reason or internal, problems)
            self.hand_off(empty, pgid, lock_fd, ctrl_w, sentinel_pid, sentinel_status, reason, problems)
            for fd in (ctrl_w, ack_r, lock_fd):  # never LOCK_UN: the lock lasts while any holder lives
                try:
                    os.close(fd)
                except OSError:
                    pass
        code = proc.returncode if proc is not None and proc.returncode is not None else 0
        child_status = 128 - code if code < 0 else code
        if internal or not empty or problems:
            status = EXIT_INTERNAL
        elif self.interrupted:
            status = EXIT_INTERRUPTED
        elif reason:
            status = EXIT_ABORTED
        else:
            status = child_status
        why = "; ".join(x for x in (reason, f"internal error: {internal}" if internal else None, *problems) if x)
        try:  # finalization: any failure here is reflected in the status (fail closed)
            free1, swap1 = free_pct(), swap_used_kb()
            if "end-log-fail" in FAULTS:
                raise OSError("end log write failed (injected by MACGUARD_TEST_FAULT)")
            self.say(f"end: status {status} (job {child_status}) after {time.monotonic() - start:.0f}s, peak group "
                     f"RSS {peak}KB, swap growth {swap1 - swap0}KB{', killed: ' + why if why else ''}")
            self.snapshot("end", free1, swap1, strict=True)
        except Exception as exc:  # a guard-decided status (124/125/130) already reports failure; keep its cause
            if status not in (EXIT_ABORTED, EXIT_INTERNAL, EXIT_INTERRUPTED):
                status = EXIT_INTERNAL
            self.say_safe(f"end: finalization failed ({exc!r}); status {status} (job {child_status})"
                          f"{', killed: ' + why if why else ''}")
        return status

    def cleanup(self, proc, pgid, abnormal, problems: list) -> bool:
        """Terminate the job group if needed; True only if it is verified empty. Never raises."""
        empty = True
        try:
            if pgid is not None:
                leftover = not abnormal and group_alive(pgid)
                if leftover:
                    self.say_safe(f"job leader exited with processes left in group {pgid}; terminating them")
                if abnormal or leftover:
                    empty = escalate(pgid, reap=proc.poll if proc is not None else (lambda: None))
                if empty and group_alive(pgid):  # re-verify after the leader was reaped
                    empty = escalate(pgid, reap=proc.poll if proc is not None else (lambda: None))
            elif proc is not None:
                proc.kill()
        except Exception as exc:
            problems.append(f"cleanup error {exc!r}")
            empty = pgid is None or not group_alive(pgid)
        if proc is not None:
            try:
                proc.wait(timeout=KILL_DEADLINE_S)
            except Exception as exc:
                problems.append(f"job leader not reaped: {exc!r}")
                empty = False
        return empty

    def hand_off(self, empty: bool, pgid, lock_fd: int, ctrl_w: int, sentinel_pid: int, sentinel_status, reason,
                 problems: list) -> None:
        """Release (group verified empty) or hand the group over to a lock holder that kills it until it is empty.
        Never raises; never lets go of the lock while the group may be alive."""
        if empty:
            try:
                os.write(ctrl_w, b"done\n")
            except OSError as exc:
                if sentinel_status is None:
                    problems.append(f"cannot reach the sentinel {sentinel_pid}: {exc!r}")
            if sentinel_status is None:
                try:
                    _, status = os.waitpid(sentinel_pid, 0)
                    if status != 0 and not reason:
                        problems.append(f"sentinel {sentinel_pid} ended with wait status {status}")
                except ChildProcessError:
                    pass
                except OSError as exc:
                    problems.append(f"cannot wait for the sentinel: {exc!r}")
            return
        message = f"process group {pgid} not verified empty after cleanup"
        try:
            os.write(ctrl_w, b"watch\n")
            if self.sentinel_exited(sentinel_pid) is not None:
                raise OSError("sentinel exited")
            self.say_safe(f"ERROR: {message}; the sentinel {sentinel_pid} keeps the lock and keeps killing it")
            return
        except OSError:
            pass
        if spawn_watcher(pgid, lock_fd, self.log_path):
            self.say_safe(f"ERROR: {message}; sentinel unreachable, a detached watcher keeps the lock and kills it")
            return
        self.say_safe(f"ERROR: {message}; no sentinel or watcher; macguard keeps the lock and kills it")
        escalate(pgid, kill_deadline=FOREVER)


def caller_environment(environ) -> dict:
    """The job's environment: environ without the variables the caller of the sh front end (ios/macguard) did not
    have (thread caps excepted), with every RESTORED variable put back to the caller's state, and the record removed.
    Without a record (macguard.py started directly) environ is returned unchanged."""
    env = dict(environ)
    names = env.pop("MACGUARD_CALLER_NAMES", None)
    if names is not None:
        keep = set(names.split()) | set(THREAD_VARS)
        for name in [n for n in env if n not in keep and not n.startswith("MACGUARD_CALLER_")]:
            del env[name]
    for name in RESTORED:
        flag = env.pop("MACGUARD_CALLER_SET_" + name, None)
        value = env.pop("MACGUARD_CALLER_VAL_" + name, "")
        if flag == "1":
            env[name] = value
        elif flag == "0":
            env.pop(name, None)
    return env


def exec_after_gate(argv: list) -> int:
    """Job wrapper: register this process group with the sentinel (inherited control pipe, then closed), wait for
    the supervisor's go on the gate pipe, then exec CMD (127 if not found) in the caller's environment
    (caller_environment) plus the thread caps. Any other outcome exits 125 without running CMD."""
    gate, ctrl = int(argv[0]), int(argv[1])
    cmd = argv[3:]
    try:
        os.write(ctrl, f"pgid {os.getpid()}\n".encode())
    except OSError:
        return EXIT_INTERNAL
    finally:
        os.close(ctrl)
    data = os.read(gate, 1)
    os.close(gate)
    if data != b"g":
        return EXIT_INTERNAL
    try:
        os.execvpe(cmd[0], cmd, caller_environment(os.environ))
    except OSError as exc:
        print(f"macguard: cannot run {cmd[0]}: {exc}", file=sys.stderr)
        return 127


def main() -> int:
    if sys.argv[1:2] == ["--exec-after-gate"]:
        return exec_after_gate(sys.argv[2:])
    try:
        opts, cmd = parse(sys.argv[1:])
    except ValueError:
        print(USAGE, file=sys.stderr)
        return EXIT_USAGE
    directory = os.environ.get("MACGUARD_DIR", DEFAULT_DIR if DARWIN else "")
    if not directory:
        print("macguard: set MACGUARD_DIR (no default outside macOS)", file=sys.stderr)
        return EXIT_USAGE
    try:
        os.makedirs(os.path.join(directory, "logs"), exist_ok=True)
    except OSError as exc:
        print(f"macguard: cannot create {directory}/logs: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    return Guard(opts, cmd, directory).run()


if __name__ == "__main__":
    sys.exit(main())
