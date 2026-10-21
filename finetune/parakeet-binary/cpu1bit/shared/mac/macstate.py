"""Whole-block state monitoring on the Mac (plans/cpu-benchmark-mac.md "Quiet host", M6, M9, M11, M12).

Every second: AC power and battery (`pmset -g batt`), thermal state (`pmset -g therm`), Low Power Mode
(`pmset -g`), user input (HIDIdleTime, `ioreg -c IOHIDSystem`), load average, memory pressure
(`kern.memorystatus_vm_pressure_level` via sysctlbyname), swap-out counters (`vm_stat` Pageouts and
Swapouts), and the foreign-CPU estimate of M11: host busy cores from cumulative CPU-tick deltas
(`host_processor_info(PROCESSOR_CPU_LOAD_INFO)`) minus our own busy cores from cumulative CPU-time
deltas over the same interval, reconciled across process starts and exits by OwnCpuLedger (Codex code
review c2 finding 2): live pids via `proc_pid_rusage`, reaped children via `getrusage(RUSAGE_CHILDREN)`
minus what was already attributed to them; impossible deltas are probe failures. `ps %cpu` is not used. Any probe failure is recorded
and invalidates the window (M6). powermetrics runs as root and cannot be probed with proc_pid_rusage;
it is therefore counted as foreign load (the same in every arm; its share is reported).

Runs under /usr/bin/python3 (3.9) on the Mac; parsers are pure and tested on NixOS with fixtures.
"""
from __future__ import annotations

import ctypes
import errno
import ctypes.util
import os
import re
import resource
import subprocess
import threading
import time
from dataclasses import dataclass, field

HID_QUIET_S = 600          # no user input for 10 minutes before a block
LOAD_LIMIT = 1.5
FOREIGN_LIMIT_CORES = 0.5
STATHZ = 100                # host_processor_info tick rate (kern.clockrate stathz = 100, measured)
OWN_HOST_TOL_CORES = 0.05   # skew between the tick read and the rusage reads, and rounding
REAP_TOL_NS = 1_000_000     # RUSAGE_CHILDREN (us) vs proc_pid_rusage (mach ticks) rounding
PRESSURE_NORMAL = 1        # kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical


CLOCK = getattr(time, "CLOCK_UPTIME_RAW", None)   # CLOCK_MONOTONIC_RAW where absent (Linux tests)
if CLOCK is None:
    CLOCK = time.CLOCK_MONOTONIC_RAW


def now_ns() -> int:
    return time.clock_gettime_ns(CLOCK)


# --- pure parsers (fixtures in tests/test_state.py) ---------------------------------------------

def parse_batt(text: str) -> dict:
    ac = "'AC Power'" in text
    m = re.search(r"(\d+)%;\s*([^;]+);", text)
    return {"ac": ac, "battery_pct": int(m.group(1)) if m else None, "battery_state": m.group(2).strip() if m else None}


def parse_therm(text: str) -> dict:
    """Nominal iff every line about a warning level says none was recorded or reports level 0, and
    every CPU_Speed_Limit / CPU_Scheduler_Limit reported is 100 (conservative: anything else is a
    thermal transition)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bad = []
    for l in lines:
        low = l.lower()
        if "warning level" in low:
            if "no " in low and "has been recorded" in low:
                continue
            m = re.search(r"(\d+)\s*\.?$", l)
            if not (m and int(m.group(1)) == 0):
                bad.append(l)
        m = re.match(r"(CPU_Speed_Limit|CPU_Scheduler_Limit)\s*=\s*(\d+)", l)
        if m and int(m.group(2)) != 100:
            bad.append(l)
    return {"nominal": not bad, "not_nominal_lines": bad, "raw": text.strip()[:400]}


def parse_lowpower(text: str) -> int | None:
    m = re.search(r"lowpowermode\s+(\d+)", text)
    return int(m.group(1)) if m else None


def parse_hid_idle_ns(text: str) -> int | None:
    m = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', text)
    return int(m.group(1)) if m else None


def parse_vm_stat(text: str) -> dict:
    out = {}
    for key in ("Pageouts", "Swapouts"):
        m = re.search(rf"{key}:\s*(\d+)\.", text)
        out[key.lower()] = int(m.group(1)) if m else None
    return out


# --- native probes -----------------------------------------------------------------------------

class Native:
    """ctypes probes of libSystem (no third-party modules)."""

    PROCESSOR_CPU_LOAD_INFO = 2
    CPU_STATE_MAX = 4  # user, system, idle, nice
    RUSAGE_INFO_V2 = 2

    def __init__(self):
        self.lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        self.task = ctypes.c_uint.in_dll(self.lib, "mach_task_self_")
        self.lib.mach_host_self.restype = ctypes.c_uint
        self.host = self.lib.mach_host_self()

        class Timebase(ctypes.Structure):
            _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]
        tb = Timebase()
        self.lib.mach_timebase_info(ctypes.byref(tb))
        self.timebase = (tb.numer, tb.denom)
        # proc_pid_rusage times are in mach absolute-time units on Apple silicon; `rusage_scale`
        # converts to ns and is validated by `validate_rusage_units` (tests/test_state.py on the Mac).
        self.rusage_scale = tb.numer / tb.denom

    def cpu_ticks(self) -> list[tuple[int, int]]:
        """Per CPU (busy, total) cumulative ticks."""
        ncpu = ctypes.c_uint()
        info = ctypes.POINTER(ctypes.c_int)()
        cnt = ctypes.c_uint()
        kr = self.lib.host_processor_info(self.host, self.PROCESSOR_CPU_LOAD_INFO, ctypes.byref(ncpu),
                                          ctypes.byref(info), ctypes.byref(cnt))
        if kr != 0:
            raise OSError(f"host_processor_info kr={kr}")
        try:
            out = []
            for c in range(ncpu.value):
                user, system, idle, nice = (info[c * self.CPU_STATE_MAX + j] & 0xFFFFFFFF for j in range(4))
                out.append((user + system + nice, user + system + idle + nice))
            return out
        finally:
            self.lib.vm_deallocate(self.task, ctypes.cast(info, ctypes.c_void_p), ctypes.c_size_t(cnt.value * 4))

    def pid_cpu_ns(self, pid: int) -> int:
        return self.pid_rusage(pid)[0]

    def pid_rusage(self, pid: int) -> tuple[int, int]:
        """(user + system CPU ns, start time ns on CLOCK_UPTIME_RAW) of a live or zombie process
        (struct rusage_info_v2: user 16, system 24, ri_proc_start_abstime 80; mach units)."""
        buf = ctypes.create_string_buffer(512)
        if self.lib.proc_pid_rusage(ctypes.c_int(pid), self.RUSAGE_INFO_V2, buf) != 0:
            raise OSError(ctypes.get_errno(), f"proc_pid_rusage({pid})")
        g = lambda o: int.from_bytes(buf.raw[o:o + 8], "little")  # noqa: E731
        return int((g(16) + g(24)) * self.rusage_scale), int(g(80) * self.rusage_scale)

    def probe(self, pid: int) -> tuple:
        """("ok", cum_ns, start_ns) | ("gone",) (ESRCH: reaped) | ("error", message)."""
        try:
            cum, start = self.pid_rusage(pid)
            return ("ok", cum, start)
        except OSError as e:
            return ("gone",) if e.errno == errno.ESRCH else ("error", f"proc_pid_rusage({pid}): {e}")

    def sysctl_int(self, name: str) -> int:
        v = ctypes.c_int64(0)
        size = ctypes.c_size_t(8)
        if self.lib.sysctlbyname(name.encode(), ctypes.byref(v), ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
            raise OSError(ctypes.get_errno(), f"sysctlbyname({name})")
        return v.value if size.value == 8 else ctypes.c_int32(v.value & 0xFFFFFFFF).value


def children_cpu_ns() -> int:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return int((r.ru_utime + r.ru_stime) * 1e9)


def run_text(cmd: list[str], timeout: float = 5.0) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True).stdout


class OwnCpuLedger:
    """Own CPU time per interval from cumulative counters (Codex code review c2 finding 2). Each step
    gets a probe of every tracked pid (registered now, or alive at the previous step) and
    RUSAGE_CHILDREN read before and after the probes:
      own = sum over pids alive at both steps of (cum - cum_prev)
          + sum over pids first seen that started at or after the previous step of cum
          + (children_now - children_prev) - sum over pids gone since the previous step of cum_prev
    A reaped child's whole lifetime enters RUSAGE_CHILDREN at once; what earlier intervals already
    attributed to it (its last probe) is subtracted, so only its CPU since then counts here. Children
    never tracked (the sampler's own pmset/ioreg/vm_stat runs) count entirely when reaped (ours).
    Errors (the interval is invalid): a non-ESRCH probe error, a negative per-pid delta, RUSAGE_CHILDREN
    changing during the probes (a reap raced the probe), a pid first seen that started before the
    previous step (its earlier CPU cannot be attributed), or a reaped remainder below -REAP_TOL_NS
    (a gone pid was not reaped by us)."""

    def __init__(self):
        self.prev = None  # (t_ns, {pid: cum_ns}, children_ns)

    def tracked(self) -> list[int]:
        return sorted(self.prev[1]) if self.prev else []

    def step(self, t_ns: int, probes: dict, children_before_ns: int, children_after_ns: int):
        """Returns (own_ns or None on the first step, errors)."""
        errors = []
        if children_after_ns != children_before_ns:
            errors.append("a child was reaped during the probes")
        live = {pid: r[1] for pid, r in probes.items() if r[0] == "ok"}
        for pid, r in probes.items():
            if r[0] == "error":
                errors.append(r[1])
        if self.prev is None:
            self.prev = (t_ns, live, children_after_ns)
            return None, errors
        t0, prev_live, ch0 = self.prev
        own = 0
        for pid, r in probes.items():
            if r[0] != "ok":
                continue
            cum, start = r[1], r[2]
            if pid in prev_live:
                d = cum - prev_live[pid]
                if d < 0:
                    errors.append(f"pid {pid}: CPU time decreased by {-d} ns")
                own += d
            elif start >= t0:
                own += cum
            else:
                errors.append(f"pid {pid} first seen after it started (earlier CPU unattributable)")
        gone = [pid for pid in prev_live if pid not in live and probes.get(pid, ("gone",))[0] == "gone"]
        reaped = (children_after_ns - ch0) - sum(prev_live[pid] for pid in gone)
        if reaped < -REAP_TOL_NS:
            errors.append(f"reaped children inconsistent: {reaped} ns (a gone pid was not reaped by us)")
        own += max(reaped, 0)
        self.prev = (t_ns, live, children_after_ns)
        return own, errors


def cpu_fields(dt_s: float, host_busy_cores: float, own_ns: int | None, ncpu: int) -> tuple[dict, list[str]]:
    """Host, own and foreign busy cores for one interval, the tick-quantisation margin on host busy
    (one tick per CPU at STATHZ), and the impossible-delta checks (own above host, i.e. negative
    foreign beyond the margin)."""
    margin = ncpu / (STATHZ * dt_s) if dt_s > 0 else float("inf")
    out = {"host_busy_cores": host_busy_cores, "own_busy_cores": None, "foreign_busy_cores": None,
           "foreign_margin_cores": margin}
    errors = []
    if own_ns is None or dt_s <= 0:
        return out, errors
    own = own_ns / 1e9 / dt_s
    out["own_busy_cores"] = own
    out["foreign_busy_cores"] = host_busy_cores - own
    if own > host_busy_cores + margin + OWN_HOST_TOL_CORES:
        errors.append(f"impossible: own busy {own:.2f} > host busy {host_busy_cores:.2f} cores")
    return out, errors


@dataclass
class StateSample:
    t_ns: int
    ac: bool | None = None
    battery_pct: int | None = None
    therm_nominal: bool | None = None
    therm_raw: str = ""
    lowpower: int | None = None
    hid_idle_ns: int | None = None
    load1: float | None = None
    pressure: int | None = None
    pageouts: int | None = None
    swapouts: int | None = None
    host_busy_cores: float | None = None
    own_busy_cores: float | None = None
    foreign_busy_cores: float | None = None
    foreign_margin_cores: float = 0.0
    errors: list = field(default_factory=list)


class Sampler:
    """1 s state sampler in a thread. `own_pids()` returns our current process set (refreshed every
    interval: the orchestrator itself plus the running block process). Every sample covers a full
    interval (the first one is taken one interval after start()), so wait_after(t) can establish a
    completed sample before a boundary (before the powermetrics spawn) and after one (after the capture
    ends). `native` and `text` are the probes (tests inject fakes)."""

    def __init__(self, own_pids, interval_s: float = 1.0, native=None, text=None):
        self.native = native if native is not None else Native()
        self.text = text if text is not None else run_text
        self.own_pids = own_pids
        self.interval_s = interval_s
        self.samples: list[StateSample] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.prev = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.ledger = OwnCpuLedger()
        self.prev = self._cpu_state()
        self.thread.start()
        return self

    def _cpu_state(self):
        """(t_ns, ticks, ledger result): ticks, then RUSAGE_CHILDREN, the pid probes, RUSAGE_CHILDREN."""
        pids = sorted(set(self.own_pids()) | set(self.ledger.tracked()))
        t = now_ns()
        ticks = self.native.cpu_ticks()
        c0 = children_cpu_ns()
        probes = {pid: self.native.probe(pid) for pid in pids}
        c1 = children_cpu_ns()
        return t, ticks, self.ledger.step(t, probes, c0, c1)

    def sample_once(self) -> StateSample:
        s = StateSample(now_ns())
        probes = [
            ("pmset batt", lambda: parse_batt(self.text(["pmset", "-g", "batt"]))),
            ("pmset therm", lambda: parse_therm(self.text(["pmset", "-g", "therm"]))),
            ("pmset lowpower", lambda: parse_lowpower(self.text(["pmset", "-g"]))),
            ("ioreg HIDIdleTime", lambda: parse_hid_idle_ns(self.text(["ioreg", "-c", "IOHIDSystem", "-d", "4", "-r", "-k", "HIDIdleTime"]))),
            ("vm_stat", lambda: parse_vm_stat(self.text(["vm_stat"]))),
        ]
        for name, fn in probes:
            try:
                v = fn()
            except Exception as e:  # noqa: BLE001 - every probe failure is recorded
                s.errors.append(f"{name}: {type(e).__name__}: {e}")
                continue
            if name == "pmset batt":
                s.ac, s.battery_pct = v["ac"], v["battery_pct"]
            elif name == "pmset therm":
                s.therm_nominal, s.therm_raw = v["nominal"], v["raw"]
            elif name == "pmset lowpower":
                s.lowpower = v
            elif name == "ioreg HIDIdleTime":
                s.hid_idle_ns = v
            else:
                s.pageouts, s.swapouts = v["pageouts"], v["swapouts"]
            if v is None or (isinstance(v, dict) and None in v.values() and name != "pmset therm"):
                s.errors.append(f"{name}: unparsed")
        try:
            s.pressure = self.native.sysctl_int("kern.memorystatus_vm_pressure_level")
        except OSError as e:
            s.errors.append(f"pressure: {e}")
        s.load1 = os.getloadavg()[0]
        try:
            cur = self._cpu_state()
            (t0, ticks0, _), (t1, ticks1, (own_ns, ledger_errors)) = self.prev, cur
            dt = (t1 - t0) / 1e9
            host = sum((b1 - b0) / (tt1 - tt0) for (b0, tt0), (b1, tt1) in zip(ticks0, ticks1) if tt1 > tt0)
            fields, cpu_errors = cpu_fields(dt, host, own_ns, len(ticks1))
            for k, v in fields.items():
                setattr(s, k, v)
            s.errors += ledger_errors + cpu_errors
            self.prev = cur
        except OSError as e:
            s.errors.append(f"cpu probes: {e}")
        return s

    def _run(self):
        next_t = time.monotonic() + self.interval_s
        while not self.stop_event.wait(max(0.0, next_t - time.monotonic())):
            next_t += self.interval_s
            s = self.sample_once()
            with self.lock:
                self.samples.append(s)

    def wait_after(self, t_ns: int, timeout_s: float = 10.0) -> bool:
        """Blocks until a completed sample stamped at or after t_ns exists (True), or the timeout."""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            with self.lock:
                if any(x.t_ns >= t_ns for x in self.samples):
                    return True
            time.sleep(0.05)
        return False

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)

    def snapshot(self) -> list[StateSample]:
        with self.lock:
            return list(self.samples)


def quiet_now(native: Native | None = None) -> dict:
    """The pre-block conditions (plans/cpu-benchmark-mac.md "Before each block"), except the P-cluster
    idle check (from powermetrics) and the other-agent check (job --quiet-host)."""
    out = {"checks": {}, "errors": []}
    try:
        b = parse_batt(run_text(["pmset", "-g", "batt"]))
        out["checks"]["ac_power"] = b["ac"]
        out["checks"]["battery_100"] = b["battery_pct"] == 100
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"batt: {e}")
    try:
        out["checks"]["thermal_nominal"] = parse_therm(run_text(["pmset", "-g", "therm"]))["nominal"]
        out["checks"]["low_power_off"] = parse_lowpower(run_text(["pmset", "-g"])) == 0
        hid = parse_hid_idle_ns(run_text(["ioreg", "-c", "IOHIDSystem", "-d", "4", "-r", "-k", "HIDIdleTime"]))
        out["hid_idle_s"] = None if hid is None else hid / 1e9
        out["checks"]["no_input_10min"] = hid is not None and hid / 1e9 >= HID_QUIET_S
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"pmset/ioreg: {e}")
    out["load1"] = os.getloadavg()[0]
    out["checks"]["load_below_1.5"] = out["load1"] < LOAD_LIMIT
    try:
        native = native or Native()
        p = native.sysctl_int("kern.memorystatus_vm_pressure_level")
        out["pressure"] = p
        out["checks"]["memory_pressure_normal"] = p == PRESSURE_NORMAL
    except OSError as e:
        out["errors"].append(f"pressure: {e}")
    out["ok"] = not out["errors"] and all(out["checks"].values())
    return out


MAX_STATE_GAP_NS = 2_500_000_000


def window_state(samples: list[StateSample], t0: int, t1: int) -> dict:
    """Invalidation reasons for the interval [t0, t1] (a timed window, or a whole block phase:
    review c1 findings 5 and 6). A sample at t describes its measurement interval (t_prev, t]: CPU-time
    deltas, and HIDIdleTime reveals input since t - idle. Required: a bracketing sample at or before t0
    and one at or after t1, consecutive samples at most 2.5 s apart. Evaluated: the bracketing sample
    before (state at t0) and every sample with t0 < t <= t_after, t_after the FIRST sample at or after
    t1, so activity at the end of the window that is only observed afterwards is caught. Input: any
    input after t0 seen by any of these samples invalidates (conservative: input between t1 and
    t_after is counted, because a later input masks an earlier one in HIDIdleTime). Swap-out counters:
    the bracketing samples before and after are compared (M12)."""
    ss = sorted(samples, key=lambda x: x.t_ns)
    before = [x for x in ss if x.t_ns <= t0]
    after = [x for x in ss if x.t_ns >= t1]
    reasons = []
    if not before:
        reasons.append("no state sample at or before the start")
    if not after:
        reasons.append("no state sample at or after the end")
    t_after = after[0].t_ns if after else (ss[-1].t_ns if ss else t1)
    covering = [x for x in ss if t0 < x.t_ns <= t_after]
    span = ([before[-1]] if before else []) + covering
    for x, y in zip(span, span[1:]):
        if y.t_ns - x.t_ns > MAX_STATE_GAP_NS:
            reasons.append(f"state sampling gap {(y.t_ns - x.t_ns) / 1e9:.1f} s")
            break
    if not covering:
        reasons.append("no state sample inside the interval")
    for x in span:
        if x.errors:
            reasons.append(f"probe failure: {x.errors[0]}")
        if x.ac is False or (x.battery_pct is not None and x.battery_pct < 100):
            reasons.append("power transition (AC / battery)")
        if x.therm_nominal is False:
            reasons.append("thermal state not nominal")
        if x.lowpower not in (0, None):
            reasons.append("Low Power Mode on")
        if x.pressure is not None and x.pressure != PRESSURE_NORMAL:
            reasons.append(f"memory pressure level {x.pressure}")
    for x in covering:
        if x.hid_idle_ns is not None and x.t_ns - x.hid_idle_ns > t0:
            reasons.append("user input during the interval")
        if x.foreign_busy_cores is not None and x.foreign_busy_cores + x.foreign_margin_cores > FOREIGN_LIMIT_CORES:
            reasons.append(f"foreign busy {x.foreign_busy_cores:.2f} (+{x.foreign_margin_cores:.2f} tick margin) cores")
        for field_name in ("hid_idle_ns", "foreign_busy_cores", "pressure", "ac", "therm_nominal", "lowpower"):
            if getattr(x, field_name) is None and not x.errors:
                reasons.append(f"probe missing: {field_name}")
    if before and after:
        b0, a1 = before[-1], after[0]
        if None in (b0.pageouts, b0.swapouts, a1.pageouts, a1.swapouts):
            reasons.append("probe missing: swap counters at the bracketing samples")
        elif a1.pageouts > b0.pageouts or a1.swapouts > b0.swapouts:
            reasons.append("swap-out increase (vm_stat pageouts/swapouts)")
    uniq = list(dict.fromkeys(reasons))
    foreign = [x.foreign_busy_cores for x in covering if x.foreign_busy_cores is not None]
    return {"valid": not uniq, "reasons": uniq, "samples": len(covering),
            "foreign_busy_cores_max": max(foreign) if foreign else None,
            "foreign_busy_cores_mean": sum(foreign) / len(foreign) if foreign else None}


def validate_rusage_units(seconds: float = 0.5) -> dict:
    """Burns `seconds` of CPU in this process and compares proc_pid_rusage (scaled) with
    time.process_time(): the ratio must be ~1 (M11 unit check)."""
    n = Native()
    pid = os.getpid()
    c0, p0 = n.pid_cpu_ns(pid), time.process_time_ns()
    end = time.process_time() + seconds
    x = 0
    while time.process_time() < end:
        x += 1
    c1, p1 = n.pid_cpu_ns(pid), time.process_time_ns()
    return {"rusage_ns": c1 - c0, "process_time_ns": p1 - p0, "ratio": (c1 - c0) / max(1, p1 - p0),
            "timebase": n.timebase, "scale": n.rusage_scale}
