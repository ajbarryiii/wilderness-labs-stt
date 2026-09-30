"""Power flight recorder: one fsync'd JSON line per sample, so the tail of the log
shows what the box drew until the breaker tripped. See README.md "Power logging".

est_wall_w = (gpu_w + cpu_w + baseline_watts) / psu_efficiency. cpu_w is RAPL package
power when intel-rapl:0/energy_uj is readable (root-only by default), else the proxy
cpu_busy * 170 + 30 W (about idle to all-core package power of the 170 W TDP 9950X3D);
est_wall_w_basis says which. hwmon amdgpu "PPT" is the iGPU's power, not the socket's.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import paths

POWER_DIR = paths.RUNS / "power"
PY = paths.HERE / "python"
RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"
RAPL_WRAP = 65532610987  # max_energy_range_uj on this machine; re-read when readable
RAPL_RECHECK_S, WARN_EVERY_S = 60.0, 30.0
PROXY_W_PER_BUSY, PROXY_IDLE_W = 170.0, 30.0
GPU_QUERY = ("timestamp,power.draw,power.draw.instant,power.draw.average,clocks.sm,clocks.mem,"
             "utilization.gpu,temperature.gpu,clocks_event_reasons.sw_power_cap,"
             "clocks_event_reasons.hw_slowdown,power.limit")
COLUMNS = ["t_utc", "t_mono", "gpu_w", "gpu_w_instant", "gpu_sm_mhz", "gpu_mem_mhz", "gpu_util",
           "gpu_temp_c", "gpu_sw_power_cap", "gpu_hw_slowdown", "cpu_pkg_w", "cpu_busy",
           "cpu_freq_mhz_mean", "cpu_tctl_c", "load1", "wall_w", "est_wall_w", "est_wall_w_basis"]


def utc_now() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


def _read(path: str) -> str:
    with open(path) as handle:
        return handle.read()


def _num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None  # "[N/A]", "[Not Supported]"


def parse_gpu(line: str) -> dict:
    """One `nvidia-smi --query-gpu=GPU_QUERY --format=csv,noheader,nounits` line."""
    v = [part.strip() for part in line.split(",")]
    if len(v) != GPU_QUERY.count(",") + 1:
        raise ValueError(f"unexpected nvidia-smi line: {line.strip()!r}")
    flag = {"Active": True, "Not Active": False}.get
    return {"gpu_w": _num(v[1]), "gpu_w_instant": _num(v[2]), "gpu_sm_mhz": _num(v[4]),
            "gpu_mem_mhz": _num(v[5]), "gpu_util": _num(v[6]), "gpu_temp_c": _num(v[7]),
            "gpu_sw_power_cap": flag(v[8]), "gpu_hw_slowdown": flag(v[9]),
            "gpu_power_limit_w": _num(v[10])}


def rapl_delta(prev_uj: int, cur_uj: int, wrap_uj: int = RAPL_WRAP) -> int:
    """Microjoules between two energy_uj readings; the counter wraps to 0 at wrap_uj."""
    return (cur_uj - prev_uj) % wrap_uj


def est_wall(gpu_w, cpu_pkg_w, cpu_busy, baseline_watts, psu_efficiency):
    """(estimated wall watts, basis), or (None, None) without the GPU or any CPU reading."""
    if cpu_pkg_w is not None:
        cpu, basis = cpu_pkg_w, "rapl"
    else:
        cpu, basis = (cpu_busy * PROXY_W_PER_BUSY + PROXY_IDLE_W if cpu_busy is not None
                      else None), "cpu_busy_proxy"
    if gpu_w is None or cpu is None:
        return None, None
    return round((gpu_w + cpu + baseline_watts) / psu_efficiency, 1), basis


def discover() -> tuple[list[str], str | None]:
    """cpufreq files and k10temp's Tctl input (the hwmon index is not stable across boots)."""
    freqs = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"))
    for label in sorted(glob.glob("/sys/class/hwmon/hwmon*/temp*_label")):
        if (_read(os.path.join(os.path.dirname(label), "name")).strip() == "k10temp"
                and _read(label).strip() == "Tctl"):
            return freqs, label[:-len("label")] + "input"
    return freqs, None


class Sensors:
    """Stateful readers. A failing sensor yields None and appends to self.errors."""

    def __init__(self, wall_command=None, baseline_watts=90.0, psu_efficiency=0.90,
                 interval=1.0, read=_read, run=subprocess.run, clock=time.monotonic):
        self.read, self.run, self.clock, self.errors = read, run, clock, []
        self.wall_command, self.baseline, self.efficiency = wall_command, baseline_watts, psu_efficiency
        self.wall_timeout, self.stat, self.energy, self.wrap = max(2.0, interval), None, None, RAPL_WRAP
        self.freq_paths, self.tctl_path = self.attempt("discover", discover) or ([], None)
        if self.tctl_path is None:
            self.errors.append("k10temp: no Tctl sensor found")
        self.cpu_energy = self.check_rapl()
        self.attempt("/proc/stat", self.cpu_busy)  # prime so the first sample has a delta

    def attempt(self, name, fn):
        try:
            return fn()
        except Exception as error:  # a sensor must never stop the recorder
            self.errors.append(f"{name}: {type(error).__name__}: {error}")
            return None

    def nvsmi(self, query: str) -> str:
        return self.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5, check=True).stdout.splitlines()[0]

    def check_rapl(self) -> str:
        """Status of the energy counter for the header; on success self.energy is primed."""
        self.rapl_checked, self.rapl_ok = self.clock(), False
        try:
            self.energy = (int(self.read(RAPL)), self.clock())
            self.wrap = int(self.read(RAPL_MAX))
        except PermissionError:
            return "unavailable (permission denied)"
        except (OSError, ValueError) as error:
            return f"unavailable ({type(error).__name__}: {error})"
        self.rapl_ok = True
        return "powercap"

    def cpu_pkg_w(self) -> float | None:
        now = self.clock()
        if not self.rapl_ok:  # re-check so a chmod takes effect without a restart
            if now - self.rapl_checked >= RAPL_RECHECK_S:
                self.cpu_energy = self.check_rapl()  # primes; the next sample has a delta
            return None
        try:
            uj = int(self.read(RAPL))
        except (OSError, ValueError):
            self.cpu_energy = self.check_rapl()
            raise
        (prev_uj, prev_t), self.energy = self.energy, (uj, now)
        return round(rapl_delta(prev_uj, uj, self.wrap) / 1e6 / (now - prev_t), 2) if now > prev_t else None

    def cpu_busy(self) -> float | None:
        fields = [int(x) for x in self.read("/proc/stat").split("\n", 1)[0].split()[1:9]]
        idle, total = fields[3] + fields[4], sum(fields)  # idle + iowait; guest is inside user
        prev, self.stat = self.stat, (idle, total)
        if prev is None or total <= prev[1]:
            return None
        return round(1 - (idle - prev[0]) / (total - prev[1]), 4)

    def sample(self) -> dict:
        read = self.read
        gpu = self.attempt("nvidia-smi", lambda: parse_gpu(self.nvsmi(GPU_QUERY))) or {}
        s = {"type": "sample", "t_utc": utc_now(), "t_mono": round(self.clock(), 3),
             **{key: gpu.get(key) for key in COLUMNS[2:10]},
             "cpu_pkg_w": self.attempt("cpu energy", self.cpu_pkg_w),
             "cpu_busy": self.attempt("/proc/stat", self.cpu_busy),
             "cpu_freq_mhz_mean": self.attempt("cpufreq", lambda: round(
                 sum(int(read(p)) for p in self.freq_paths) / len(self.freq_paths) / 1000, 1)),
             "cpu_tctl_c": (self.attempt("k10temp", lambda: int(read(self.tctl_path)) / 1000)
                            if self.tctl_path else None),
             "load1": self.attempt("/proc/loadavg", lambda: float(read("/proc/loadavg").split()[0])),
             "wall_w": self.attempt("wall-command", lambda: float(self.run(
                 wall_shell_command(self.wall_command, self.wall_timeout), shell=True,
                 capture_output=True, text=True, timeout=self.wall_timeout + 1,
                 check=True).stdout)) if self.wall_command else None}
        s["est_wall_w"], s["est_wall_w_basis"] = est_wall(
            s["gpu_w"], s["cpu_pkg_w"], s["cpu_busy"], self.baseline, self.efficiency)
        return s


def wall_shell_command(command: str, timeout: float) -> str:
    """Wrap the user's wall-power command so a hung pipeline dies as a whole.

    GNU timeout (without --foreground) runs the command in its own process
    group and signals the whole group on expiry, so `curl | sed` children cannot
    outlive the sample. The Python-side timeout is one second longer as backstop.
    """
    return f"exec timeout -s KILL {timeout:g} sh -c {shlex.quote(command)}"


def make_header(args, sensors: Sensors) -> dict:
    name, _, limit = (sensors.attempt("nvidia-smi", lambda: sensors.nvsmi("name,power.limit"))
                      or "").rpartition(",")
    cpuinfo = sensors.attempt("/proc/cpuinfo", lambda: sensors.read("/proc/cpuinfo")) or ""
    models = [line.split(":", 1)[1].strip() for line in cpuinfo.splitlines()
              if line.startswith("model name")]
    return {"type": "header", "started_utc": utc_now(), "hostname": socket.gethostname(),
            "interval_s": args.interval, "baseline_watts": args.baseline_watts,
            "psu_efficiency": args.psu_efficiency, "warn_wall_watts": args.warn_wall_watts,
            "wall_command": args.wall_command, "gpu_name": name.strip() or None,
            "gpu_power_limit_w": _num(limit.strip()), "cpu_model": models[0] if models else None,
            "cpu_energy": sensors.cpu_energy, "cpu_proxy_w": "cpu_busy * 170 + 30",
            "columns": COLUMNS}


def stop_reason(args, stop: dict) -> str | None:
    if args.stop_file and os.path.exists(args.stop_file):
        stop.setdefault("reason", "stop-file")
    if args.parent_pid and os.getppid() != args.parent_pid:
        stop.setdefault("reason", "parent-exited")
    return stop.get("reason")


def run_loop(out: Path, args, sensors: Sensors, stop: dict | None = None, sleep=time.sleep) -> str:
    """Header, then samples (plus warn/error/cpu_energy records) until stopped, then end."""
    stop = {} if stop is None else stop
    seen, last_warn, status, count = set(), None, sensors.cpu_energy, 0
    with open(out, "x") as handle:
        def write(record: dict) -> None:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())  # the tail must survive a hard power loss

        def drain_errors() -> None:
            errors, sensors.errors = sensors.errors, []
            for message in errors:
                if message not in seen:
                    seen.add(message)
                    write({"type": "error", "t_utc": utc_now(), "message": message})

        write(make_header(args, sensors))
        drain_errors()
        deadline = sensors.clock()
        while not stop_reason(args, stop):
            s = sensors.attempt("sample", sensors.sample)
            drain_errors()
            if sensors.cpu_energy != status:
                status = sensors.cpu_energy
                write({"type": "cpu_energy", "t_utc": utc_now(), "status": status})
            if s:
                write(s)
                count += 1
                watts, basis = ((s["wall_w"], "wall_w") if s["wall_w"] is not None
                                else (s["est_wall_w"], "est_wall_w"))
                if (watts is not None and watts > args.warn_wall_watts
                        and (last_warn is None or s["t_mono"] - last_warn >= WARN_EVERY_S)):
                    last_warn = s["t_mono"]
                    write({"type": "warn", "t_utc": s["t_utc"], "t_mono": s["t_mono"],
                           "watts": watts, "basis": basis, "threshold_w": args.warn_wall_watts})
                    print(f"WARN {s['t_utc']} {basis} {watts} W > {args.warn_wall_watts} W",
                          file=sys.stderr, flush=True)
            deadline = max(deadline + args.interval, sensors.clock())  # no burst after a stall
            while not stop.get("reason") and (left := deadline - sensors.clock()) > 0:
                sleep(min(left, 0.25))
        write({"type": "end", "t_utc": utc_now(), "t_mono": round(sensors.clock(), 3),
               "reason": stop["reason"], "samples": count})
    return stop["reason"]


def summarize(path, tail_seconds: float = 30.0) -> str:
    """Text summary; unparseable lines (a torn tail after power loss) are counted and skipped."""
    records, skipped = [], 0
    with open(path, errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                record = None
            if isinstance(record, dict):
                records.append(record)
            else:
                skipped += bool(line.strip("\x00 \n"))  # blank or NUL-filled tail is not a record
    kind = lambda k: [r for r in records if r.get("type") == k]  # noqa: E731
    header, samples, ends = (kind("header") or [{}])[0], kind("sample"), kind("end")
    later = kind("cpu_energy")
    out = [f"file: {path}",
           f"host {header.get('hostname')}; GPU {header.get('gpu_name')} (limit "
           f"{header.get('gpu_power_limit_w')} W); CPU {header.get('cpu_model')}",
           f"cpu_energy: {header.get('cpu_energy', 'unknown (no header)')}"
           + (f" (later: {later[-1].get('status')})" if later else ""),
           "end: " + (ends[-1].get("reason", "?") if ends
                      else "NONE - no end record (power loss, crash or SIGKILL)")]
    if skipped:
        out.append(f"skipped {skipped} unparseable line(s) (torn write at power loss?)")
    if not samples:
        return "\n".join(out + ["no samples"])
    last = samples[-1]["t_mono"]
    out.append(f"duration {last - samples[0]['t_mono']:.1f} s, {len(samples)} samples "
               f"({samples[0]['t_utc']} .. {samples[-1]['t_utc']})")
    for key in ("gpu_w", "gpu_w_instant", "cpu_pkg_w", "est_wall_w", "wall_w"):
        vals = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
        out.append(f"  {key:<14}" + (f" mean {sum(vals) / len(vals):7.1f}  max {max(vals):7.1f}"
                                     f"  (n={len(vals)})" if vals else " n/a"))
    bases = {b: sum(s.get("est_wall_w_basis") == b for s in samples) for b in ("rapl", "cpu_busy_proxy")}
    capped = sum(s.get("gpu_sw_power_cap") is True for s in samples) / len(samples)
    out.append(f"gpu_sw_power_cap active in {capped:.1%} of samples; est_wall_w basis {bases}; "
               f"warnings {len(kind('warn'))}; errors {len(kind('error'))}")
    out += [f"  error: {r.get('message')}" for r in kind("error")]
    fmt = lambda v: "-" if v is None else f"{v:.1f}"  # noqa: E731
    out.append(f"last {tail_seconds:g} s:\n  {'t_utc':<12} {'rel_s':>6} {'gpu_w':>7} "
               f"{'cpu_pkg_w':>9} {'est_wall_w':>10} {'wall_w':>7}")
    out += [f"  {s['t_utc'][11:23]:<12} {s['t_mono'] - last:>6.1f} {fmt(s.get('gpu_w')):>7} "
            f"{fmt(s.get('cpu_pkg_w')):>9} {fmt(s.get('est_wall_w')):>10} {fmt(s.get('wall_w')):>7}"
            for s in samples if s["t_mono"] >= last - tail_seconds]
    return "\n".join(out)


def start_recorder(name: str, **kwargs) -> subprocess.Popen:
    """Launch `record --name NAME` detached; kwargs become flags (warn_wall_watts=650)."""
    paths.require_mount()
    POWER_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [str(PY), str(Path(__file__).resolve()), "record", "--name", name,
           "--parent-pid", str(os.getpid())]
    for key, value in kwargs.items():
        if value is not None:
            cmd += [f"--{key.replace('_', '-')}", str(value)]
    with open(POWER_DIR / f"{name}.log", "ab") as log:
        return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)


def stop_recorder(proc: subprocess.Popen, timeout: float = 10) -> int | None:
    """SIGTERM, then SIGKILL, each with a bounded wait; never blocks the caller.

    Returns the exit code, or None if the process is still not reaped (for
    example stuck in uninterruptible disk I/O); its --parent-pid guard ends it
    once the caller exits.
    """
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                return None
    return proc.returncode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sampling = argparse.ArgumentParser(add_help=False)
    sampling.add_argument("--interval", type=float, default=1.0)
    sampling.add_argument("--baseline-watts", type=float, default=90.0, help="board, RAM, disks, fans")
    sampling.add_argument("--psu-efficiency", type=float, default=0.90)
    sampling.add_argument("--wall-command", help="shell command printing wall watts, run each sample")
    rec = sub.add_parser("record", parents=[sampling], help="log until SIGTERM/SIGINT/--stop-file")
    rec.add_argument("--name", required=True)
    rec.add_argument("--warn-wall-watts", type=float, default=700.0)
    rec.add_argument("--stop-file", help="stop cleanly once this path exists")
    rec.add_argument("--parent-pid", type=int, help="stop once this process is not the parent")
    summ = sub.add_parser("summary", help="summarize a log; tolerates a torn last line")
    summ.add_argument("file")
    summ.add_argument("--tail-seconds", type=float, default=30.0)
    sub.add_parser("status", parents=[sampling], help="print one live sample as JSON")
    args = parser.parse_args(argv)
    if args.command == "summary":
        print(summarize(args.file, args.tail_seconds))
        return
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.command == "status":
        sensors = Sensors(args.wall_command, args.baseline_watts, args.psu_efficiency, args.interval)
        time.sleep(0.5)  # a window for the /proc/stat and RAPL deltas
        sample = sensors.sample()
        print(json.dumps({**sample, "cpu_energy": sensors.cpu_energy, "errors": sensors.errors}, indent=1))
        return
    if not paths._RUN_NAME.fullmatch(args.name):
        parser.error(f"--name must match {paths._RUN_NAME.pattern!r}")
    stop: dict = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda num, _frame: stop.setdefault("reason", signal.Signals(num).name))
    paths.require_mount()
    POWER_DIR.mkdir(parents=True, exist_ok=True)
    out = POWER_DIR / f"{args.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
    sensors = Sensors(args.wall_command, args.baseline_watts, args.psu_efficiency, args.interval)
    print(f"{utc_now()} recording to {out}", flush=True)
    reason = run_loop(out, args, sensors, stop)
    print(f"{utc_now()} stopped ({reason}): {out}", flush=True)


if __name__ == "__main__":
    main()
