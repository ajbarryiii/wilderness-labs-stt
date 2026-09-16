"""GPU board energy measurement, with no model/runtime dependency or disk writes.

Pass ``synchronize=`` for asynchronous workloads. Otherwise each measured callable
must block until its GPU work completes. Index is an NVML physical GPU index,
not an index remapped by CUDA_VISIBLE_DEVICES; compare metadata's UUID yourself.

NVML reports power in mW and cumulative energy in mJ:
https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html
"""

from __future__ import annotations

import importlib
import math
import os
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any


def integrate_power(samples: list[dict[str, Any]]) -> float:
    """Trapezoidal integral of ordered timestamp_s / power_w samples, in joules."""
    if len(samples) < 2:
        raise ValueError("Power integration requires at least two samples")
    energy = 0.0
    previous = None
    for sample in samples:
        stamp, power = sample["timestamp_s"], sample["power_w"]
        if power is None or not math.isfinite(power) or power < 0:
            raise ValueError("Power samples must contain finite, nonnegative watts")
        if not math.isfinite(stamp):
            raise ValueError("Power timestamps must be finite")
        if previous is not None:
            delta = stamp - previous[0]
            if delta < 0:
                raise ValueError("Power timestamps must be ordered")
            energy += delta * (power + previous[1]) / 2
        previous = (stamp, power)
    return energy


class EnergyMeter:
    """Read-only NVML metering; refuses measurements with foreign compute PIDs.

    ``nvml``, ``clock`` and ``sleep`` support deterministic tests without a GPU.
    The meter owns one NVML initialization reference; use a context manager or
    call close(). No power limits, clocks or persistence settings are changed.
    """

    def __init__(
        self,
        index: int = 0,
        *,
        synchronize: Callable[[], None] | None = None,
        sample_interval: float = 0.1,
        allowed_pids: Iterable[int] = (),
        nvml: Any = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not math.isfinite(sample_interval) or sample_interval <= 0:
            raise ValueError("sample_interval must be finite and positive")
        if nvml is None:
            try:
                nvml = importlib.import_module("pynvml")
            except ImportError as exc:
                raise RuntimeError("Install nvidia-ml-py to measure GPU energy") from exc
        self._nvml = nvml
        self.index = index
        self._synchronize = synchronize
        self.sample_interval = sample_interval
        self.allowed_pids = frozenset(int(pid) for pid in allowed_pids)
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._closed = True
        self._unsupported = tuple(
            cls for name in ("NVMLError_NotSupported", "NVMLError_FunctionNotFound")
            if isinstance(cls := getattr(nvml, name, None), type)
        )
        nvml.nvmlInit()
        try:
            self._handle = nvml.nvmlDeviceGetHandleByIndex(index)
        except BaseException:
            nvml.nvmlShutdown()
            raise
        self._closed = False

    def __enter__(self) -> EnergyMeter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            if not self._lock.acquire(blocking=False):
                raise RuntimeError("Cannot close an EnergyMeter during measurement")
            try:
                self._nvml.nvmlShutdown()
                self._closed = True
            finally:
                self._lock.release()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("EnergyMeter is closed")

    def _optional(self, name: str, *args: Any) -> Any:
        """Suppress unsupported features only; device/permission errors are fatal."""
        function = getattr(self._nvml, name, None)
        if function is None:
            return None
        try:
            return function(*args)
        except self._unsupported:
            return None

    @staticmethod
    def _text(value: Any) -> Any:
        return value.decode() if isinstance(value, bytes) else value

    @staticmethod
    def _milli(value: Any) -> float | None:
        return None if value is None else float(value) / 1000

    def metadata(self) -> dict[str, Any]:
        self._check_open()
        h, n = self._handle, self._nvml
        memory = self._optional("nvmlDeviceGetMemoryInfo", h)
        capability = self._optional("nvmlDeviceGetCudaComputeCapability", h)
        return {
            "gpu_index": self.index,
            "uuid": self._text(n.nvmlDeviceGetUUID(h)),
            "name": self._text(n.nvmlDeviceGetName(h)),
            "driver_version": self._text(self._optional("nvmlSystemGetDriverVersion")),
            "nvml_version": self._text(self._optional("nvmlSystemGetNVMLVersion")),
            "compute_capability": list(capability) if capability else None,
            "total_memory_bytes": int(memory.total) if memory else None,
            "power_limit_w": self._milli(self._optional("nvmlDeviceGetPowerManagementLimit", h)),
            "default_power_limit_w": self._milli(self._optional("nvmlDeviceGetPowerManagementDefaultLimit", h)),
            "temperature_c": self._optional("nvmlDeviceGetTemperature", h, n.NVML_TEMPERATURE_GPU),
            "graphics_clock_mhz": self._optional("nvmlDeviceGetClockInfo", h, n.NVML_CLOCK_GRAPHICS),
            "sm_clock_mhz": self._optional("nvmlDeviceGetClockInfo", h, n.NVML_CLOCK_SM),
            "memory_clock_mhz": self._optional("nvmlDeviceGetClockInfo", h, n.NVML_CLOCK_MEM),
            "sample_interval_s": self.sample_interval,
            "energy_scope": "GPU board; excludes host energy",
        }

    def foreign_processes(self, allowed_pids: Iterable[int] = ()) -> list[dict[str, Any]]:
        self._check_open()
        allowed = self.allowed_pids | {os.getpid()} | {int(pid) for pid in allowed_pids}
        try:
            processes = self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        except Exception as exc:
            raise RuntimeError("Cannot verify that the GPU has no foreign compute processes") from exc
        unavailable = getattr(self._nvml, "NVML_VALUE_NOT_AVAILABLE", (1 << 64) - 1)
        return [
            {
                "pid": int(process.pid),
                "used_gpu_memory_bytes": (
                    None if getattr(process, "usedGpuMemory", None) in (None, unavailable)
                    else int(process.usedGpuMemory)
                ),
            }
            for process in processes if int(process.pid) not in allowed
        ]

    def _guard(self) -> None:
        foreign = self.foreign_processes()
        if foreign:
            pids = ", ".join(str(process["pid"]) for process in foreign)
            raise RuntimeError(f"Refusing GPU energy measurement: foreign compute PIDs {pids}")

    def _sample(self) -> dict[str, Any]:
        h, n = self._handle, self._nvml
        before = self._clock()
        power = self._milli(self._optional("nvmlDeviceGetPowerUsage", h))
        energy = self._optional("nvmlDeviceGetTotalEnergyConsumption", h)
        stamp = (before + self._clock()) / 2
        if power is None and energy is None:
            raise RuntimeError("GPU exposes neither NVML energy nor power readings")
        memory = self._optional("nvmlDeviceGetMemoryInfo", h)
        return {
            "timestamp_s": stamp,
            "power_w": power,
            "total_energy_mj": int(energy) if energy is not None else None,
            "temperature_c": self._optional("nvmlDeviceGetTemperature", h, n.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz": self._optional("nvmlDeviceGetClockInfo", h, n.NVML_CLOCK_SM),
            "memory_clock_mhz": self._optional("nvmlDeviceGetClockInfo", h, n.NVML_CLOCK_MEM),
            "used_gpu_memory_bytes": int(memory.used) if memory is not None else None,
            "power_limit_w": self._milli(self._optional("nvmlDeviceGetPowerManagementLimit", h)),
        }

    def measure(
        self,
        function: Callable[[], Any],
        min_seconds: float,
        min_iterations: int = 1,
        *,
        synchronize: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """Repeat complete work; keep both total energy and raw telemetry.

        All compilation/loading/warmup must happen before this call. Results
        from the workload are discarded. A failed workload, telemetry read,
        counter reset or foreign PID invalidates the entire window.
        """
        self._check_open()
        if not math.isfinite(min_seconds) or min_seconds < 0:
            raise ValueError("min_seconds must be finite and nonnegative")
        if isinstance(min_iterations, bool) or not isinstance(min_iterations, int) or min_iterations < 1:
            raise ValueError("min_iterations must be a positive integer")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("EnergyMeter measurements cannot overlap")
        try:
            return self._measure(function, min_seconds, min_iterations, synchronize or self._synchronize)
        finally:
            self._lock.release()

    def _measure(self, function: Callable[[], Any], minimum: float, count: int,
                 synchronize: Callable[[], None] | None) -> dict[str, Any]:
        self._guard()
        if synchronize is not None:
            synchronize()
        start = self._sample()
        samples = [start]
        errors: list[Exception] = []
        stop = threading.Event()

        def collect() -> None:
            while not stop.wait(self.sample_interval):
                try:
                    self._guard()
                    samples.append(self._sample())
                except Exception as exc:
                    errors.append(exc)
                    stop.set()

        sampler = threading.Thread(target=collect, name="nvml-energy-sampler", daemon=True)
        sampler.start()
        iterations = []
        try:
            while len(iterations) < count or self._clock() - start["timestamp_s"] < minimum:
                if errors:
                    raise RuntimeError("NVML measurement window invalidated") from errors[0]
                begin = self._clock()
                function()
                if synchronize is not None:
                    synchronize()
                iterations.append(self._clock() - begin)
            stop.set()
            sampler.join()
            if errors:
                raise RuntimeError("NVML measurement window invalidated") from errors[0]
            end = self._sample()
            samples.append(end)
            self._guard()
        finally:
            stop.set()
            sampler.join()

        elapsed = end["timestamp_s"] - start["timestamp_s"]
        if elapsed <= 0:
            raise RuntimeError("Measurement interval must be positive")
        power_limits = [sample["power_limit_w"] for sample in samples]
        if power_limits[0] is not None and any(limit is None for limit in power_limits):
            raise RuntimeError("NVML power limit became unavailable during measurement")
        if len({limit for limit in power_limits if limit is not None}) > 1:
            raise RuntimeError("GPU power limit changed during measurement")
        integrated = integrate_power(samples) if all(s["power_w"] is not None for s in samples) else None
        counters = [s["total_energy_mj"] for s in samples]
        if counters[0] is not None:
            if any(value is None for value in counters):
                raise RuntimeError("NVML energy counter became unavailable during measurement")
            if any(right < left for left, right in zip(counters, counters[1:])):
                raise RuntimeError("NVML energy counter reset during measurement")
            joules = (counters[-1] - counters[0]) / 1000
            method = "nvml_total_energy"
        else:
            if integrated is None:
                raise RuntimeError("Cannot integrate incomplete NVML power telemetry")
            joules, method = integrated, "nvml_power_trapezoid"
        for sample in samples:
            sample["elapsed_s"] = sample["timestamp_s"] - start["timestamp_s"]
        telemetry_averages = {}
        for field in ("temperature_c", "sm_clock_mhz", "memory_clock_mhz"):
            telemetry_averages[field] = (
                sum(
                    (right["timestamp_s"] - left["timestamp_s"]) * (left[field] + right[field]) / 2
                    for left, right in zip(samples, samples[1:])
                ) / elapsed
                if all(sample[field] is not None for sample in samples) else None
            )
        sampled_memory = [sample["used_gpu_memory_bytes"] for sample in samples
                          if sample["used_gpu_memory_bytes"] is not None]
        return {
            "method": method,
            "elapsed_seconds": elapsed,
            "energy_joules": joules,
            "power_integral_joules": integrated,
            "average_watts": joules / elapsed,
            "iterations": len(iterations),
            "iteration_seconds": iterations,
            "joules_per_iteration": joules / len(iterations),
            "synchronization": "callback" if synchronize is not None else "workload must block",
            "samples": samples,
            "telemetry_samples": len(samples),
            "telemetry_averages": telemetry_averages,
            "max_sampled_gpu_memory_bytes": max(sampled_memory) if sampled_memory else None,
            "gpu_memory_scope": "NVML device memory including all GPU contexts; sampled maximum, not allocation peak",
        }

    def idle(self, seconds: float) -> dict[str, Any]:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Idle duration must be finite and positive")
        result = self.measure(lambda: self._sleep(seconds), min_seconds=0)
        result["kind"] = "idle"
        return result
