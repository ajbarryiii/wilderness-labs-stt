"""Linux CPU topology and read-only cumulative package energy measurement.

No GPU/runtime dependency, privileged helpers, hardware settings, or disk writes.
Package RAPL includes other processes and uncore; it is not wall-plug energy.
Linux documents energy_uj and max_energy_range_uj at:
https://www.kernel.org/doc/html/latest/power/powercap/powercap.html
The kernel's perf RAPL driver exports its scale and package CPU mask in sysfs:
https://github.com/torvalds/linux/blob/master/arch/x86/events/rapl.c
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
import re
import struct
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def parse_cpu_list(value: str) -> list[int]:
    """Parse Linux CPU lists, rejecting malformed/reversed ranges."""
    result: set[int] = set()
    for item in value.strip().split(","):
        if not item:
            if not value.strip():
                return []
            raise ValueError(f"Malformed CPU list: {value!r}")
        if not re.fullmatch(r"\d+(?:-\d+)?", item):
            raise ValueError(f"Malformed CPU list: {value!r}")
        bounds = [int(part) for part in item.split("-")]
        start, end = bounds[0], bounds[-1]
        if end < start:
            raise ValueError(f"Reversed CPU range: {item}")
        result.update(range(start, end + 1))
    return sorted(result)


def _optional_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _cache_bytes(value: str) -> int:
    match = re.fullmatch(r"(\d+)\s*([KMG]?)", value.strip(), re.I)
    if not match:
        raise ValueError(f"Invalid sysfs cache size: {value!r}")
    return int(match[1]) * 1024 ** (" KMG".index(match[2].upper()) if match[2] else 0)


def discover_topology(
    sysfs_root: Path | str = "/sys/devices/system/cpu",
    *,
    allowed_cpus: Iterable[int] | None = None,
    cpuinfo_path: Path | str = "/proc/cpuinfo",
) -> dict[str, Any]:
    """Return allowed online cores and L3 groups without assuming CPU numbering.

    One permitted SMT sibling is selected for each physical core. L3 domains
    correspond to the two CCDs on this Ryzen; other CPUs may group differently.
    """
    root = Path(sysfs_root)
    online = parse_cpu_list((root / "online").read_text())
    allowed = set(os.sched_getaffinity(0) if allowed_cpus is None else allowed_cpus)
    usable = sorted(allowed.intersection(online))
    if not usable:
        raise RuntimeError("No online CPUs are available in this process's affinity")
    cpuinfo = _optional_text(Path(cpuinfo_path)) or ""
    info = dict(line.split(":", 1) for line in cpuinfo.split("\n\n", 1)[0].splitlines() if ":" in line)
    info = {key.strip(): value.strip() for key, value in info.items()}
    cores: dict[tuple[int, int], list[int]] = {}
    groups: dict[tuple[int, tuple[int, ...]], dict[str, Any]] = {}
    policies: dict[str, dict[str, Any]] = {}
    for cpu in usable:
        base = root / f"cpu{cpu}"
        package = int((base / "topology/physical_package_id").read_text())
        core = int((base / "topology/core_id").read_text())
        cores.setdefault((package, core), []).append(cpu)
        for cache in sorted((base / "cache").glob("index*")):
            if _optional_text(cache / "level") != "3":
                continue
            shared = tuple(parse_cpu_list((cache / "shared_cpu_list").read_text()))
            key = (package, shared)
            groups.setdefault(key, {
                "id": _optional_text(cache / "id") or str(min(shared)),
                "package_id": package,
                "size_bytes": _cache_bytes((cache / "size").read_text()),
                "shared_cpus": list(shared),
                "logical_cpus": sorted(allowed.intersection(online).intersection(shared)),
            })
        policy = base / "cpufreq"
        if policy.exists():
            key = str(policy.resolve())
            policies.setdefault(key, {
                name: _optional_text(policy / name)
                for name in ("scaling_driver", "scaling_governor", "energy_performance_preference",
                             "scaling_min_freq", "scaling_max_freq")
            })
    physical = sorted(min(siblings) for siblings in cores.values())
    l3_groups = sorted(groups.values(), key=lambda group: (group["package_id"], min(group["shared_cpus"])))
    presets = {"all_physical": physical}
    for index, group in enumerate(l3_groups):
        group["physical_cpus"] = sorted(set(physical).intersection(group["logical_cpus"]))
        # Ordinals give unique names even when the kernel repeats cache IDs per socket.
        group["affinity_name"] = f"l3_{index}_physical"
        presets[group["affinity_name"]] = group["physical_cpus"]
    if l3_groups:
        presets["largest_l3_physical"] = max(l3_groups, key=lambda group: group["size_bytes"])["physical_cpus"]
        presets["smallest_l3_physical"] = min(l3_groups, key=lambda group: group["size_bytes"])["physical_cpus"]
    return {
        "model_name": info.get("model name", platform.processor()),
        "architecture": platform.machine(),
        "kernel": platform.release(),
        "flags": sorted(info.get("flags", "").split()),
        "online_cpus": online,
        "allowed_cpus": usable,
        "physical_cpus": physical,
        "physical_core_count": len(physical),
        "logical_cpu_count": len(usable),
        "package_ids": sorted({package for package, _ in cores}),
        "physical_cores": [{"package_id": package, "core_id": core, "logical_cpus": siblings}
                           for (package, core), siblings in sorted(cores.items())],
        "l3_groups": l3_groups,
        "affinity_presets": presets,
        "cpufreq_policies": list(policies.values()),
        "load_average": list(os.getloadavg()),
        "energy_scope": "CPU package(s), including other processes and uncore; excludes discrete GPU and wall-plug losses",
    }


def counter_delta(before: int, after: int, maximum: int | None) -> int:
    """Counter delta; accept boundary wraps, reject apparent resets.

    Sampling must be frequent enough for at most one wrap. Resets near the
    wrap boundary cannot be distinguished by the hardware counter alone.
    """
    if before < 0 or after < 0 or (maximum is not None and (maximum <= 0 or max(before, after) > maximum)):
        raise RuntimeError("Invalid package energy counter value/range")
    if after >= before:
        return after - before
    if maximum is not None and before >= maximum * 0.9 and after <= maximum * 0.1:
        return maximum - before + after
    raise RuntimeError("Package energy counter reset or ambiguous wrap")


class _PowercapReader:
    method = "linux_powercap_package_energy"

    def __init__(self, root: Path) -> None:
        self.paths: list[Path] = []
        self.domains: list[dict[str, Any]] = []
        seen = set()
        # The class directory exposes zone symlinks directly. Also allow the
        # nested layout documented for /sys/devices/virtual/powercap.
        candidates = sorted(set(root.glob("*:*")) | set(root.glob("*/*:*")))
        for path in candidates:
            resolved = path.resolve()
            name = _optional_text(path / "name")
            if resolved in seen or name is None or re.fullmatch(r"package-\d+", name) is None:
                continue
            seen.add(resolved)
            maximum = int((path / "max_energy_range_uj").read_text())
            if maximum <= 0:
                raise RuntimeError(f"Invalid energy range for {path}")
            self.paths.append(path / "energy_uj")
            self.domains.append({"name": name, "path": str(path / "energy_uj"),
                                 "maximum": maximum, "joules_per_tick": 1e-6})
        if not self.paths:
            raise RuntimeError("No package RAPL energy zones in " + str(root))
        self.read()  # Fail discovery on inaccessible packages, never measure a partial set.

    def read(self) -> list[int]:
        return [int(path.read_text()) for path in self.paths]

    def close(self) -> None:
        pass


class _PerfReader:
    method = "linux_perf_package_energy"

    def __init__(self, root: Path) -> None:
        self.fds: list[int] = []
        self.domains: list[dict[str, Any]] = []
        if platform.machine() not in ("x86_64", "AMD64"):
            raise RuntimeError("The perf RAPL fallback supports x86_64 Linux only")
        event = (root / "events/energy-pkg").read_text().strip()
        match = re.fullmatch(r"event=(0x[0-9a-fA-F]+|\d+)", event)
        if not match:
            raise RuntimeError(f"Unsupported perf energy event encoding: {event}")
        scale = float((root / "events/energy-pkg.scale").read_text())
        unit = (root / "events/energy-pkg.unit").read_text().strip().lower()
        if unit != "joules" or not math.isfinite(scale) or scale <= 0:
            raise RuntimeError("Invalid perf package energy scale/unit")
        cpus = parse_cpu_list((root / "cpumask").read_text())
        if not cpus:
            raise RuntimeError("perf package CPU mask is empty")
        event_type = int((root / "type").read_text())
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        try:
            for cpu in cpus:
                # perf_event_attr v0 (64 bytes) suffices. Read total time enabled
                # and running so multiplexed/unscheduled counters are rejected.
                attr = bytearray(64)
                struct.pack_into("IIQ", attr, 0, event_type, len(attr), int(match[1], 0))
                struct.pack_into("Q", attr, 32, 3)
                buf = ctypes.create_string_buffer(bytes(attr))
                fd = libc.syscall(298, ctypes.byref(buf), -1, cpu, -1, 8)  # PERF_FLAG_FD_CLOEXEC
                if fd < 0:
                    error = ctypes.get_errno()
                    raise OSError(error, f"perf_event_open energy-pkg CPU {cpu}: {os.strerror(error)}")
                self.fds.append(int(fd))
                self.domains.append({"name": f"package_cpu_{cpu}", "cpu": cpu,
                                     "maximum": None, "joules_per_tick": scale})
            self.read()
        except BaseException:
            self.close()
            raise

    def read(self) -> list[int]:
        values = []
        for fd in self.fds:
            payload = os.read(fd, 24)
            if len(payload) != 24:
                raise RuntimeError("Incomplete perf package energy read")
            value, enabled, running = struct.unpack("QQQ", payload)
            if enabled != running:
                raise RuntimeError("perf package energy counter was multiplexed or unscheduled")
            values.append(value)
        return values

    def close(self) -> None:
        for fd in self.fds:
            os.close(fd)
        self.fds = []


class CpuEnergyMeter:
    """Repeat synchronous CPU work using package counters, or latency only.

    Missing permissions produce explicit unavailable metadata and null energy.
    Once a counter is selected, failed reads or invalid counters fail the whole
    measurement. Warmup and model construction must precede measure().
    """

    def __init__(
        self,
        reader: Any = None,
        *,
        unavailable_reason: str | None = None,
        sample_interval: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(sample_interval) or sample_interval <= 0:
            raise ValueError("sample_interval must be finite and positive")
        self.reader = reader
        self.unavailable_reason = unavailable_reason or ("No energy reader configured" if reader is None else None)
        self.sample_interval = sample_interval
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def detect(
        cls,
        powercap_root: Path | str = "/sys/class/powercap",
        perf_root: Path | str = "/sys/bus/event_source/devices/power",
        **kwargs: Any,
    ) -> CpuEnergyMeter:
        reasons = []
        for factory, path in ((_PowercapReader, powercap_root), (_PerfReader, perf_root)):
            try:
                reader = factory(Path(path))
            except (OSError, ValueError, RuntimeError) as exc:
                reasons.append(f"{factory.method}: {type(exc).__name__}: {exc}")
            else:
                return cls(reader, **kwargs)
        return cls(unavailable_reason="; ".join(reasons), **kwargs)

    def __enter__(self) -> CpuEnergyMeter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close CpuEnergyMeter during measurement")
        try:
            if self.reader is not None:
                self.reader.close()
            self._closed = True
        finally:
            self._lock.release()

    def metadata(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("CpuEnergyMeter is closed")
        return {
            "available": self.reader is not None,
            "method": self.reader.method if self.reader is not None else None,
            "unavailable_reason": self.unavailable_reason,
            "domains": self.reader.domains if self.reader is not None else [],
            "sample_interval_s": self.sample_interval,
            "energy_scope": "CPU package(s), including other processes and uncore; excludes discrete GPU and wall-plug losses",
            "counter_wrap_policy": "At most one wrap per sample interval; decreases away from range boundaries fail",
            "background_activity": "Package energy includes all host processes; process affinity does not isolate energy",
            "perf_event_paranoid": _optional_text(Path("/proc/sys/kernel/perf_event_paranoid")),
            "direct_msr_present": Path("/dev/cpu/0/msr").exists(),
        }

    def _sample(self) -> dict[str, Any]:
        begin = self._clock()
        values = self.reader.read() if self.reader is not None else None
        return {"timestamp_s": (begin + self._clock()) / 2, "counter_values": values}

    def measure(self, function: Callable[[], Any], min_seconds: float,
                min_iterations: int = 1) -> dict[str, Any]:
        self.metadata()
        if not math.isfinite(min_seconds) or min_seconds < 0:
            raise ValueError("min_seconds must be finite and nonnegative")
        if isinstance(min_iterations, bool) or not isinstance(min_iterations, int) or min_iterations < 1:
            raise ValueError("min_iterations must be a positive integer")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("CpuEnergyMeter measurements cannot overlap")
        try:
            return self._measure(function, min_seconds, min_iterations)
        finally:
            self._lock.release()

    def _measure(self, function: Callable[[], Any], minimum: float, count: int) -> dict[str, Any]:
        start = self._sample()
        samples = [start]
        errors: list[Exception] = []
        stop = threading.Event()

        def collect() -> None:
            while not stop.wait(self.sample_interval):
                try:
                    samples.append(self._sample())
                except Exception as exc:
                    errors.append(exc)
                    stop.set()

        sampler = None
        if self.reader is not None:
            sampler = threading.Thread(target=collect, name="cpu-package-energy-sampler", daemon=True)
            sampler.start()
        iterations = []
        try:
            while len(iterations) < count or self._clock() - start["timestamp_s"] < minimum:
                if errors:
                    raise RuntimeError("CPU energy measurement invalidated") from errors[0]
                begin = self._clock()
                function()
                iterations.append(self._clock() - begin)
            stop.set()
            if sampler is not None:
                sampler.join()
            if errors:
                raise RuntimeError("CPU energy measurement invalidated") from errors[0]
            end = self._sample()
            samples.append(end)
        finally:
            stop.set()
            if sampler is not None:
                sampler.join()
        elapsed = end["timestamp_s"] - start["timestamp_s"]
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("Measurement interval must be finite and positive")
        joules = None
        if self.reader is not None:
            joules = 0.0
            for left, right in zip(samples, samples[1:]):
                if right["timestamp_s"] < left["timestamp_s"]:
                    raise RuntimeError("Energy sample timestamps moved backwards")
                if len(left["counter_values"]) != len(self.reader.domains) or len(right["counter_values"]) != len(self.reader.domains):
                    raise RuntimeError("Energy package domain count changed")
                for before, after, domain in zip(left["counter_values"], right["counter_values"], self.reader.domains):
                    joules += counter_delta(before, after, domain["maximum"]) * domain["joules_per_tick"]
            if not math.isfinite(joules) or joules <= 0:
                raise RuntimeError("Package energy counter did not advance; use a longer measurement window")
        for sample in samples:
            sample["elapsed_s"] = sample["timestamp_s"] - start["timestamp_s"]
        return {
            "method": self.reader.method if self.reader is not None else None,
            "available": self.reader is not None,
            "unavailable_reason": self.unavailable_reason,
            "elapsed_seconds": elapsed,
            "energy_joules": joules,
            "average_watts": None if joules is None else joules / elapsed,
            "iterations": len(iterations),
            "iteration_seconds": iterations,
            "joules_per_iteration": None if joules is None else joules / len(iterations),
            "samples": samples,
            "telemetry_samples": len(samples) if self.reader is not None else 0,
            "energy_scope": self.metadata()["energy_scope"],
        }

    def idle(self, seconds: float) -> dict[str, Any]:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Idle duration must be finite and positive")
        result = self.measure(lambda: self._sleep(seconds), min_seconds=0)
        result["kind"] = "idle"
        return result
