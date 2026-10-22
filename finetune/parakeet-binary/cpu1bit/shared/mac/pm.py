"""powermetrics reader, record schema, stream framing and provisional timeline
(plans/cpu-benchmark-mac.md "Measurement", "Energy", A1; history M3, M15.1). Runs on the Mac under
/usr/bin/python3 (3.9, stdlib only); the pure functions are also tested on NixOS. The gating window
energy bounds, the spawn-based mapping and the per-block A1-A3 checks are in guard.py.

Only this command is ever run, with exactly these arguments, read from a pipe and stopped by closing
it:  sudo -n /usr/bin/powermetrics --samplers cpu_power -i 100 -f plist
It needs non-interactive sudo for powermetrics; see the operator setup.

Measured format (probe 2026-10-21, fixture fixtures/powermetrics-3samples.plist0): NUL-terminated XML
plists ("</plist>\n\0"); `timestamp` is a <date> with 1 s resolution (UTC; used only by the A1 band
check, never for alignment); `elapsed_ns` is the sample's length (106-119 ms observed for -i 100);
`processor.cpu_energy` is an INTEGER in mJ (cpu_power = cpu_energy / elapsed, no extra precision);
per-cluster and per-CPU idle_ratio / freq_hz under processor.clusters.

Framing (M15.1, A1 deterministic part): check_stream requires every NUL-terminated chunk to parse; a
stream that ends (EOF) before the runner's stop fails; only the single unterminated chunk cut by the
runner's own stop is exempt. Reader.stop() records the verdict on the kept raw stream.

Provisional timeline (build_timeline; descriptive and for segmentation). Samples are contiguous: each
covers the time since the previous one. Within a run of good samples (a segment) the end of sample i is
b_i = B + cum_i (cum_i = sum of elapsed_ns up to i), so arrival_i - cum_i = B + d_i with d_i >= 0 the
output latency; the provisional anchor min_i(arrival_i - cum_i) makes the reconstruction late by
min_i d_i >= 0. A malformed record or a lost record (a persistent step in arrival - cum) ends a segment.
The p8 contract retired the calibration-burst absolute anchor: the mapping is spawn-based (guard.py).

window_energy gives eps-padded sums on the provisional timeline (descriptive):
  E_lower = sum over samples with [a_i, b_i] inside [t0 + eps, t1 - eps] of (e_i - q)
  E_upper = sum over samples with [a_i, b_i] meeting [t0 - eps, t1 + eps] of (e_i + q)
  E_est   = sum of e_i * |[a_i, b_i] ∩ [t0, t1]| / (b_i - a_i)    (prorated point value)
q = 1 mJ is the integer reporting granularity. The window must be covered: one segment spans
[t0 - eps, t1 + eps]. Energies are reported in joules.
"""
from __future__ import annotations

import datetime
import math
import os
import plistlib
import subprocess
import threading
import time
from dataclasses import dataclass, field

COMMAND = ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "cpu_power", "-i", "100", "-f", "plist"]
QUANTUM_MJ = 1.0
# A persistent step in arrival - cum larger than this ends a segment (a lost record).
GAP_TOLERANCE_NS = 50_000_000


# CLOCK_UPTIME_RAW on macOS; CLOCK_MONOTONIC_RAW where it does not exist (Linux: the end-to-end test),
# matching shared::bench::now_ns on each OS.
CLOCK = getattr(time, "CLOCK_UPTIME_RAW", None)
if CLOCK is None:
    CLOCK = time.CLOCK_MONOTONIC_RAW


def now_ns() -> int:
    """The clock the block process (shared::bench::now_ns) stamps windows with, in ns."""
    return time.clock_gettime_ns(CLOCK)


@dataclass
class Sample:
    index: int
    arrival_ns: int
    ok: bool
    error: str = ""
    elapsed_ns: int = 0
    cpu_energy_mj: float = 0.0
    cpu_power_mw: float = 0.0
    combined_power_mw: float = 0.0
    # cluster name -> {"idle_ratio": float, "cpus": {cpu: idle_ratio}, "freq_hz": float}
    clusters: dict = field(default_factory=dict)
    a_ns: int = 0
    b_ns: int = 0
    segment: int = -1
    # the plist `timestamp` (1 s resolution) as POSIX seconds; None for a malformed record
    timestamp_s: float | None = None


def _num(v, name: str, lo: float = 0.0, hi: float = math.inf, integer: bool = False):
    """A finite number in [lo, hi] (ints only when `integer`); bool, NaN, infinities and strings rejected."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not isinstance(v, int)):
        raise ValueError(f"{name} {v!r}: wrong type")
    if not math.isfinite(v) or not (lo <= v <= hi):
        raise ValueError(f"{name} {v!r}: out of range")
    return v


def _timestamp_s(v) -> float:
    """The plist <date> as POSIX seconds; plistlib returns naive datetimes in UTC (aware ones are honoured)."""
    if not isinstance(v, datetime.datetime):
        raise ValueError(f"timestamp {v!r}: not a date")
    if v.tzinfo is None:
        v = v.replace(tzinfo=datetime.timezone.utc)
    return _num(v.timestamp(), "timestamp", -math.inf)


def parse_record(raw: bytes, index: int, arrival_ns: int) -> Sample:
    """One NUL-delimited record -> Sample (ok=False with the reason when malformed). Enforces the
    measured schema (minor finding 8, M15.1): elapsed_ns and cpu_energy (mJ) integers, the `timestamp`
    date (kept as timestamp_s), cpu_power and combined_power finite and non-negative, every cluster /
    CPU idle_ratio finite in [0, 1]."""
    try:
        d = plistlib.loads(raw.strip(b"\0 \n\t"))
    except Exception as e:  # noqa: BLE001 - any parse failure is a malformed record
        return Sample(index, arrival_ns, False, f"plist: {type(e).__name__}")
    try:
        if not isinstance(d, dict):
            raise ValueError("record is not a dict")
        p = d["processor"]
        timestamp = _timestamp_s(d["timestamp"])
        elapsed = _num(d["elapsed_ns"], "elapsed_ns", 1, integer=True)
        energy = _num(p["cpu_energy"], "cpu_energy", 0, integer=True)
        power = _num(p["cpu_power"], "cpu_power")
        combined = _num(p["combined_power"], "combined_power")
        clusters = {}
        for c in p["clusters"]:
            name = c["name"]
            if not isinstance(name, str):
                raise ValueError("cluster name")
            cpus = {}
            for u in c["cpus"]:
                cpus[_num(u["cpu"], "cpu", 0, integer=True)] = _num(u["idle_ratio"], "cpu idle_ratio", 0, 1)
            clusters[name] = {"idle_ratio": _num(c["idle_ratio"], "idle_ratio", 0, 1),
                              "freq_hz": _num(c.get("freq_hz", 0.0), "freq_hz"), "cpus": cpus}
        if not clusters:
            raise ValueError("no clusters")
        return Sample(index, arrival_ns, True, "", elapsed, float(energy), float(power), float(combined), clusters,
                      timestamp_s=timestamp)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, OSError) as e:
        return Sample(index, arrival_ns, False, f"fields: {e}")


def split_records(buf: bytes) -> tuple[list[bytes], bytes]:
    """Complete NUL-terminated records and the unterminated remainder (the incremental reader's split:
    whitespace-only chunks are skipped; check_stream is the strict verdict)."""
    parts = buf.split(b"\0")
    return [p for p in parts[:-1] if p.strip()], parts[-1]


def check_stream(raw: bytes, stopped_by_runner: bool) -> tuple[bool, list[str], list[Sample]]:
    """M15.1 framing verdict on a kept raw stream -> (ok, reasons, records). Every NUL-terminated chunk
    (whitespace-only ones included) must parse into a record of the enforced schema; a stream that the
    runner did not stop (EOF: powermetrics ended on its own) fails; the single trailing unterminated chunk
    is exempt only when stopped_by_runner (it is the chunk cut by the runner's own stop). records are the
    parsed Samples of the terminated chunks, in order (index = position; arrival_ns = 0: raw bytes carry no
    arrival times, so they serve for counting and cross-checks, not for timing)."""
    if not isinstance(raw, (bytes, bytearray)):
        return False, ["stream: raw stream is not bytes"], []
    parts = bytes(raw).split(b"\0")
    records = [parse_record(p, i, 0) for i, p in enumerate(parts[:-1])]
    reasons = [f"stream: chunk {r.index} does not parse ({r.error})" for r in records if not r.ok]
    if not stopped_by_runner:
        reasons.append("stream: EOF before the runner stopped powermetrics (the producer ended on its own)")
    if not records:
        reasons.append("stream: no NUL-terminated records")
    return not reasons, reasons, records


def _split_lost(group: list[Sample], tol: int) -> list[list[Sample]]:
    """Splits a run of good samples where a record must have been lost. With offsets
    o_i = arrival_i - cum_i, a late arrival (latency spike) raises one o_i but not the minimum of the
    later ones, whereas a lost record shifts every later o_i up by the lost elapsed_ns (~100 ms). So a
    split is placed before the first i where min(o_i..) - min(..o_{i-1}) > tol."""
    out = []
    while group:
        cum, offs = 0, []
        for s in group:
            cum += s.elapsed_ns
            offs.append(s.arrival_ns - cum)
        suffix = offs[:]
        for i in range(len(suffix) - 2, -1, -1):
            suffix[i] = min(suffix[i], suffix[i + 1])
        prefix_min, cut = offs[0], None
        for i in range(1, len(group)):
            if suffix[i] - prefix_min > tol:
                cut = i
                break
            prefix_min = min(prefix_min, offs[i])
        if cut is None:
            out.append(group)
            break
        out.append(group[:cut])
        group = group[cut:]
    return out


def build_timeline(samples: list[Sample], gap_tolerance_ns: int = GAP_TOLERANCE_NS) -> list[Sample]:
    """Assigns segments and reconstructs [a_i, b_i] per segment (anchor = min(arrival - cum)).
    Segments end at malformed records and where a record must have been lost (_split_lost);
    late arrivals alone do not split (they only make the anchor's estimate conservative)."""
    runs: list[list[Sample]] = []
    current: list[Sample] = []
    for s in samples:
        if not s.ok:
            if current:
                runs.append(current)
            current = []
            continue
        current.append(s)
    if current:
        runs.append(current)
    segments = [seg for run in runs for seg in _split_lost(run, gap_tolerance_ns)]
    for seg, group in enumerate(segments):
        cum, offsets = 0, []
        for s in group:
            cum += s.elapsed_ns
            offsets.append(s.arrival_ns - cum)
        anchor = min(offsets)
        cum = 0
        for s in group:
            s.a_ns = anchor + cum
            cum += s.elapsed_ns
            s.b_ns = anchor + cum
            s.segment = seg
    return samples


def chain_stats(samples: list[Sample]) -> dict:
    """Per segment: output-latency spread (arrival - reconstructed end) and drift between the summed
    elapsed_ns and the arrival span (ppm)."""
    out = []
    for seg in sorted({s.segment for s in samples if s.ok}):
        g = [s for s in samples if s.ok and s.segment == seg]
        lat = [s.arrival_ns - s.b_ns for s in g]
        span_arr = g[-1].arrival_ns - g[0].arrival_ns
        span_chain = g[-1].b_ns - g[0].b_ns
        out.append({"segment": seg, "samples": len(g), "latency_ns_min": min(lat), "latency_ns_max": max(lat),
                    "latency_ns_median": sorted(lat)[len(lat) // 2],
                    "drift_ppm": 1e6 * (span_arr - span_chain) / span_chain if span_chain else 0.0,
                    "elapsed_ns": {"min": min(s.elapsed_ns for s in g), "max": max(s.elapsed_ns for s in g),
                                   "mean": sum(s.elapsed_ns for s in g) / len(g)}})
    return {"segments": out, "malformed": sum(1 for s in samples if not s.ok)}


@dataclass
class WindowEnergy:
    valid: bool
    reason: str
    e_lower_j: float
    e_est_j: float
    e_upper_j: float
    samples_inside: int
    samples_touching: int
    span_ns: int
    combined_power_mean_mw: float


def window_energy(samples: list[Sample], t0: int, t1: int, eps_ns: int, quantum_mj: float = QUANTUM_MJ) -> WindowEnergy:
    """M10 bounds for [t0, t1] (CLOCK_UPTIME_RAW ns) from a built timeline."""
    if t1 <= t0:
        return WindowEnergy(False, "empty window", 0, 0, 0, 0, 0, 0, 0)
    good = [s for s in samples if s.ok]
    lo, hi = t0 - eps_ns, t1 + eps_ns
    touching = [s for s in good if s.b_ns > lo and s.a_ns < hi]
    if not touching:
        return WindowEnergy(False, "no samples", 0, 0, 0, 0, 0, t1 - t0, 0)
    segs = {s.segment for s in touching}
    bad_between = [s for s in samples if not s.ok and touching[0].arrival_ns <= s.arrival_ns <= touching[-1].arrival_ns + 1]
    covered = min(s.a_ns for s in touching) <= lo and max(s.b_ns for s in touching) >= hi
    reason = ""
    if len(segs) != 1 or bad_between:
        reason = "gap: window spans a segment boundary or a malformed record"
    elif not covered:
        reason = "not covered: samples do not span [t0 - eps, t1 + eps]"
    inside = [s for s in touching if s.a_ns >= t0 + eps_ns and s.b_ns <= t1 - eps_ns]
    e_lower = sum(max(s.cpu_energy_mj - quantum_mj, 0.0) for s in inside)
    e_upper = sum(s.cpu_energy_mj + quantum_mj for s in touching)
    e_est = 0.0
    for s in touching:
        ov = min(s.b_ns, t1) - max(s.a_ns, t0)
        if ov > 0:
            e_est += s.cpu_energy_mj * ov / (s.b_ns - s.a_ns)
    pw = sum(s.combined_power_mw * s.elapsed_ns for s in touching) / sum(s.elapsed_ns for s in touching)
    return WindowEnergy(reason == "", reason, e_lower / 1000.0, e_est / 1000.0, e_upper / 1000.0, len(inside), len(touching),
                        t1 - t0, pw)


def cluster_busy(samples: list[Sample], t0: int, t1: int) -> dict:
    """Elapsed-weighted mean busy share (1 - idle_ratio, mean over the cluster's CPUs) per cluster over
    the samples overlapping [t0, t1] (descriptive residency, M4)."""
    acc: dict = {}
    for s in samples:
        if not s.ok or s.b_ns <= t0 or s.a_ns >= t1:
            continue
        w = min(s.b_ns, t1) - max(s.a_ns, t0)
        for name, c in s.clusters.items():
            cpus = c["cpus"]
            busy = sum(1.0 - r for r in cpus.values()) / len(cpus) if cpus else 1.0 - c["idle_ratio"]
            a = acc.setdefault(name, [0.0, 0])
            a[0] += busy * w
            a[1] += w
    return {k: (v[0] / v[1] if v[1] else None) for k, v in acc.items()}


class Reader:
    """Runs the permitted powermetrics command, stamps each record's arrival with CLOCK_UPTIME_RAW,
    keeps the raw stream (NUL-delimited) in `raw_path`, and parses records. spawn_ns is CLOCK_UPTIME_RAW
    read immediately before the spawn (s in the plan's mapping; started_ns is read just after). stop()
    marks the stop as the runner's, then closes the pipe; the reader thread records an EOF or read error
    that arrives before that (eof_before_stop: powermetrics ended on its own), and stop() returns the
    framing verdict (check_stream) of the kept raw stream."""

    def __init__(self, raw_path: str):
        self.raw_path = raw_path
        self.samples: list[Sample] = []
        self.lock = threading.Lock()
        self.stopped_by_runner = False
        self.eof_before_stop = False
        self.read_error = ""
        self.spawn_ns = now_ns()
        self.proc = subprocess.Popen(COMMAND, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        self.started_ns = now_ns()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _ended(self, why: str = "") -> None:
        with self.lock:
            if not self.stopped_by_runner:
                self.eof_before_stop = True
                self.read_error = why

    def _run(self) -> None:
        buf = b""
        with open(self.raw_path, "wb") as raw:
            while True:
                try:
                    chunk = self.proc.stdout.read1(1 << 16)
                except (ValueError, OSError) as e:      # ValueError: the pipe closed by stop()
                    self._ended(f"{type(e).__name__}: {e}")
                    break
                if not chunk:
                    self._ended("EOF")
                    break
                t = now_ns()
                raw.write(chunk)
                buf += chunk
                recs, buf = split_records(buf)
                with self.lock:
                    for r in recs:
                        self.samples.append(parse_record(r, len(self.samples), t))

    def wait_for(self, count: int, timeout_s: float = 10.0) -> bool:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            with self.lock:
                if len(self.samples) >= count:
                    return True
            time.sleep(0.05)
        return False

    def snapshot(self) -> list[Sample]:
        with self.lock:
            return [Sample(**s.__dict__) for s in self.samples]

    def _stream_verdict(self) -> dict:
        if self.thread.is_alive():
            return {"ok": False, "reasons": ["stream: the reader thread did not finish (raw stream incomplete)"],
                    "records": None}
        try:
            with open(self.raw_path, "rb") as f:
                raw = f.read()
        except OSError as e:
            return {"ok": False, "reasons": [f"stream: raw stream unreadable ({type(e).__name__}: {e})"], "records": None}
        ok, reasons, recs = check_stream(raw, stopped_by_runner=not self.eof_before_stop)
        if self.eof_before_stop and self.read_error not in ("", "EOF"):
            reasons.append(f"stream: read error before the runner's stop ({self.read_error})")
        if len(recs) != len(self.samples):
            reasons.append(f"stream: {len(recs)} records in the raw stream, {len(self.samples)} parsed by the reader")
        return {"ok": ok and not reasons, "reasons": reasons, "records": len(recs)}

    def stop(self, reap: bool = True) -> dict:
        """Ends the capture: marks the stop as the runner's, closes the pipe (powermetrics ends at its next
        write), joins the reader thread and returns the framing verdict. With reap=False the producer is
        NOT waited for (Codex c5 major): its lifetime CPU must not enter RUSAGE_CHILDREN while the state
        sampler still runs, so the runner calls reap() only after state collection has ended. No poll()
        here either: poll() reaps. An end of the stream before the stop is detected by the reader thread."""
        with self.lock:
            self.stopped_by_runner = True
        try:
            self.proc.stdout.close()
        except (OSError, ValueError):
            pass
        self.thread.join(timeout=5)
        out = {"samples": len(self.samples), "spawn_ns": self.spawn_ns, "started_ns": self.started_ns,
               "stopped_by_runner": self.stopped_by_runner, "eof_before_stop": self.eof_before_stop,
               "stream": self._stream_verdict(), "returncode": None, "stderr_tail": ""}
        if reap:
            out.update(self.reap())
        return out

    def reap(self) -> dict:
        """Waits for the producer (after stop()); returns its exit status and stderr tail."""
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            # Never signal a root process; closing the pipe ends it at its next write (SIGPIPE).
            self.proc.wait(timeout=30)
        err = ""
        if self.proc.stderr:
            err = self.proc.stderr.read().decode(errors="replace")[-2000:]
            self.proc.stderr.close()
        return {"returncode": self.proc.returncode, "stderr_tail": err}

def read_raw(path: str, arrivals: list[int] | None = None) -> list[Sample]:
    """Re-parse a kept raw stream (arrival times must come from the run's record)."""
    with open(path, "rb") as f:
        recs, _ = split_records(f.read())
    return [parse_record(r, i, arrivals[i] if arrivals else 0) for i, r in enumerate(recs)]


if __name__ == "__main__":
    import json
    import sys
    r = Reader(sys.argv[1] if len(sys.argv) > 1 else os.devnull)
    r.wait_for(int(sys.argv[2]) if len(sys.argv) > 2 else 20, 30)
    info = r.stop()
    s = build_timeline(r.snapshot())
    print(json.dumps({"reader": info, "chain": chain_stats(s)}, indent=1))
