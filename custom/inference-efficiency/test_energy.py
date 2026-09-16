"""Deterministic metering tests; never initialize CUDA or real NVML."""

import math
import os
import time
import unittest
from types import SimpleNamespace

from energy import EnergyMeter, integrate_power


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeNVML:
    class NVMLError_NotSupported(Exception):
        pass

    class NVMLError_FunctionNotFound(Exception):
        pass

    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_GRAPHICS = 0
    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2
    NVML_VALUE_NOT_AVAILABLE = (1 << 64) - 1

    def __init__(self, clock):
        self.clock = clock
        self.references = 0
        self.energy_supported = True
        self.power_supported = True
        self.energy_error = False
        self.counter_offset = 0
        self.power_limit_mw = 575_000
        self.memory_used = 100_000
        self.processes = []
        self.process_error = False

    def nvmlInit(self):
        self.references += 1

    def nvmlShutdown(self):
        self.references -= 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        if self.energy_error:
            raise RuntimeError("device lost")
        if not self.energy_supported:
            raise self.NVMLError_NotSupported()
        # Deliberately much larger than 32 bits; retain integer precision.
        return 9_000_000_000_000 + self.counter_offset + int(self.clock() * 100_000)

    def nvmlDeviceGetPowerUsage(self, handle):
        if not self.power_supported:
            raise self.NVMLError_NotSupported()
        return 20_000

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        if self.process_error:
            raise PermissionError("query denied")
        return self.processes

    def nvmlDeviceGetUUID(self, handle):
        return b"GPU-test-uuid"

    def nvmlDeviceGetName(self, handle):
        return b"Fake 5090"

    def nvmlDeviceGetPowerManagementLimit(self, handle):
        return self.power_limit_mw

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(total=32 * 1024 ** 3, used=self.memory_used)


class EnergyTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.nvml = FakeNVML(self.clock)
        self.meter = EnergyMeter(nvml=self.nvml, clock=self.clock, sleep=self.clock.advance)
        self.addCleanup(self.meter.close)

    def test_trapezoidal_integration_uses_irregular_sample_intervals(self):
        samples = [
            {"timestamp_s": 10, "power_w": 10},
            {"timestamp_s": 11, "power_w": 30},
            {"timestamp_s": 14, "power_w": 50},
        ]
        self.assertEqual(integrate_power(samples), 140)

    def test_invalid_power_integral_is_rejected(self):
        with self.assertRaises(ValueError):
            integrate_power([])
        for samples in (
            [{"timestamp_s": 1, "power_w": 10}, {"timestamp_s": 0, "power_w": 10}],
            [{"timestamp_s": 0, "power_w": 10}, {"timestamp_s": 1, "power_w": math.nan}],
            [{"timestamp_s": 0, "power_w": 10}, {"timestamp_s": 1, "power_w": None}],
        ):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                integrate_power(samples)

    def test_counter_is_preferred_and_millijoules_are_converted(self):
        result = self.meter.measure(lambda: self.clock.advance(1), min_seconds=0, min_iterations=2)
        self.assertEqual(result["method"], "nvml_total_energy")
        self.assertEqual(result["energy_joules"], 200)
        self.assertEqual(result["power_integral_joules"], 40)
        self.assertEqual(result["average_watts"], 100)
        self.assertEqual(result["joules_per_iteration"], 100)
        self.assertEqual(result["iteration_seconds"], [1, 1])
        self.assertEqual(result["samples"][-1]["elapsed_s"], 2)

    def test_power_fallback_and_minimum_duration(self):
        self.nvml.energy_supported = False
        result = self.meter.measure(lambda: self.clock.advance(1), min_seconds=2.5)
        self.assertEqual(result["method"], "nvml_power_trapezoid")
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(result["energy_joules"], 60)
        self.assertEqual(result["average_watts"], 20)

    def test_counter_works_without_instantaneous_power(self):
        self.nvml.power_supported = False
        result = self.meter.measure(lambda: self.clock.advance(1), 0)
        self.assertEqual(result["energy_joules"], 100)
        self.assertIsNone(result["power_integral_joules"])

    def test_missing_both_energy_sources_rejected_before_work(self):
        self.nvml.energy_supported = self.nvml.power_supported = False
        with self.assertRaisesRegex(RuntimeError, "neither"):
            self.meter.measure(lambda: self.fail("work must not run"), 0)

    def test_synchronization_time_is_inside_each_iteration(self):
        result = self.meter.measure(
            lambda: self.clock.advance(0.75), 0, 2,
            synchronize=lambda: self.clock.advance(0.25),
        )
        self.assertEqual(result["iteration_seconds"], [1, 1])
        self.assertEqual(result["elapsed_seconds"], 2)
        self.assertEqual(result["energy_joules"], 200)
        self.assertEqual(result["synchronization"], "callback")

    def test_pid_filter_and_unknown_memory(self):
        foreign = os.getpid() + 1
        self.nvml.processes = [
            SimpleNamespace(pid=os.getpid(), usedGpuMemory=100),
            SimpleNamespace(pid=foreign, usedGpuMemory=self.nvml.NVML_VALUE_NOT_AVAILABLE),
        ]
        self.assertEqual(self.meter.foreign_processes(), [{"pid": foreign, "used_gpu_memory_bytes": None}])
        self.assertEqual(self.meter.foreign_processes(allowed_pids=[foreign]), [])
        with self.assertRaisesRegex(RuntimeError, "foreign compute PIDs"):
            self.meter.measure(lambda: self.fail("work must not run"), 0)

    def test_pid_appearing_during_window_invalidates_it(self):
        def work():
            self.clock.advance(1)
            self.nvml.processes = [SimpleNamespace(pid=os.getpid() + 1, usedGpuMemory=10)]
        with self.assertRaisesRegex(RuntimeError, "foreign compute PIDs"):
            self.meter.measure(work, 0)

    def test_background_sampler_catches_foreign_pid(self):
        self.meter.sample_interval = 0.002
        def work():
            self.clock.advance(1)
            self.nvml.processes = [SimpleNamespace(pid=os.getpid() + 1, usedGpuMemory=10)]
            time.sleep(0.025)
        with self.assertRaisesRegex(RuntimeError, "invalidated") as raised:
            self.meter.measure(work, 0)
        self.assertIn("foreign compute PIDs", str(raised.exception.__cause__))

    def test_unavailable_process_guard_is_fatal(self):
        self.nvml.process_error = True
        with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
            self.meter.measure(lambda: self.fail("work must not run"), 0)

    def test_driver_error_is_not_treated_as_unsupported_counter(self):
        self.nvml.energy_error = True
        with self.assertRaisesRegex(RuntimeError, "device lost"):
            self.meter.measure(lambda: self.clock.advance(1), 0)

    def test_counter_reset_is_fatal(self):
        def work():
            self.clock.advance(1)
            self.nvml.counter_offset = -1_000_000
        with self.assertRaisesRegex(RuntimeError, "counter reset"):
            self.meter.measure(work, 0)

    def test_power_limit_changed_during_window_invalidates_it(self):
        def work():
            self.clock.advance(1)
            self.nvml.power_limit_mw = 400_000
        with self.assertRaisesRegex(RuntimeError, "power limit changed"):
            self.meter.measure(work, 0)

    def test_sampled_memory_and_power_limit_have_explicit_units(self):
        def work():
            self.clock.advance(1)
            self.nvml.memory_used = 250_000
        result = self.meter.measure(work, 0)
        self.assertEqual(result["max_sampled_gpu_memory_bytes"], 250_000)
        self.assertEqual(result["samples"][0]["used_gpu_memory_bytes"], 100_000)
        self.assertEqual(result["samples"][-1]["power_limit_w"], 575)
        self.assertIn("sampled maximum, not allocation peak", result["gpu_memory_scope"])

    def test_workload_exception_propagates_and_meter_remains_usable(self):
        def fail():
            raise LookupError("workload failed")
        with self.assertRaisesRegex(LookupError, "workload failed"):
            self.meter.measure(fail, 0)
        self.assertEqual(self.meter.measure(lambda: self.clock.advance(1), 0)["iterations"], 1)

    def test_metadata_units_and_idle(self):
        self.assertEqual(self.meter.metadata()["power_limit_w"], 575)
        self.assertEqual(self.meter.metadata()["uuid"], "GPU-test-uuid")
        result = self.meter.idle(2)
        self.assertEqual(result["kind"], "idle")
        self.assertEqual(result["elapsed_seconds"], 2)

    def test_invalid_duration_and_iteration_count(self):
        for minimum, count in [(-1, 1), (math.nan, 1), (math.inf, 1), (0, 0), (0, 1.5), (0, True)]:
            with self.subTest(minimum=minimum, count=count), self.assertRaises(ValueError):
                self.meter.measure(lambda: None, minimum, count)

    def test_overlapping_measurement_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "cannot overlap"):
            self.meter.measure(lambda: self.meter.measure(lambda: None, 0), 0)

    def test_close_releases_nvml_and_prevents_reuse(self):
        self.assertEqual(self.nvml.references, 1)
        self.meter.close()
        self.assertEqual(self.nvml.references, 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.meter.metadata()


if __name__ == "__main__":
    unittest.main()
