"""Deterministic topology and energy tests; no real counters or inference."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hardware import CpuEnergyMeter, _PerfReader, _PowercapReader, counter_delta, discover_topology, parse_cpu_list


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeReader:
    method = "fake_package_counter"

    def __init__(self, clock):
        self.clock = clock
        self.domains = [{"name": "package-0", "maximum": 10**15, "joules_per_tick": 1e-6}]
        self.offset = 0
        self.failed = False
        self.closed = False

    def read(self):
        if self.failed:
            raise PermissionError("energy permission lost")
        return [9_000_000_000_000 + self.offset + int(self.clock() * 100_000_000)]

    def close(self):
        self.closed = True


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(value))


class TopologyTests(unittest.TestCase):
    def test_cpu_ranges(self):
        self.assertEqual(parse_cpu_list("0-2,8,16-17\n"), [0, 1, 2, 8, 16, 17])
        self.assertEqual(parse_cpu_list("3,1,3"), [1, 3])
        self.assertEqual(parse_cpu_list(""), [])
        for value in ["3-1", "1,,2", "-1", "2-3-4", "a", "1,"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_cpu_list(value)

    def test_asymmetric_l3_affinity_chooses_available_smt_sibling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root, "online", "0-3,8-11")
            write(root, "cpuinfo", "model name : Test asymmetric CPU\nflags : avx2 avx512_vnni\n")
            for cpu in [0, 1, 2, 3, 8, 9, 10, 11]:
                core = cpu % 8
                base = f"cpu{cpu}"
                write(root, f"{base}/topology/core_id", core)
                write(root, f"{base}/topology/physical_package_id", 0)
                write(root, f"{base}/cache/index3/level", 3)
                write(root, f"{base}/cache/index3/id", 0 if core < 2 else 1)
                write(root, f"{base}/cache/index3/size", "98304K" if core < 2 else "32768K")
                write(root, f"{base}/cache/index3/shared_cpu_list", "0-1,8-9" if core < 2 else "2-3,10-11")
            result = discover_topology(root, allowed_cpus=[1, 2, 3, 8, 9, 10, 11, 99], cpuinfo_path=root / "cpuinfo")
            self.assertEqual(result["physical_cpus"], [1, 2, 3, 8])
            self.assertEqual(result["logical_cpu_count"], 7)
            self.assertEqual(result["affinity_presets"]["largest_l3_physical"], [1, 8])
            self.assertEqual(result["affinity_presets"]["smallest_l3_physical"], [2, 3])
            self.assertEqual(result["l3_groups"][0]["size_bytes"], 96 * 1024**2)
            self.assertEqual(result["model_name"], "Test asymmetric CPU")
            self.assertIn("avx512_vnni", result["flags"])

    def test_no_available_online_cpu_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(root, "online", "0-3")
            with self.assertRaisesRegex(RuntimeError, "No online CPUs"):
                discover_topology(root, allowed_cpus=[8])


class ReaderTests(unittest.TestCase):
    def test_powercap_package_only_and_symlink_deduplication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, label in [("intel-rapl:0", "package-0"), ("intel-rapl:0:0", "core"),
                                ("intel-rapl:1", "package-1")]:
                write(root, f"{name}/name", label)
                write(root, f"{name}/energy_uj", 123_000_000)
                write(root, f"{name}/max_energy_range_uj", 65532610987)
                write(root, f"{name}/enabled", 0)
            (root / "alias:0").symlink_to(root / "intel-rapl:0", target_is_directory=True)
            reader = _PowercapReader(root)
            self.assertEqual(len(reader.domains), 2)
            self.assertEqual({domain["name"] for domain in reader.domains}, {"package-0", "package-1"})
            self.assertEqual(reader.read(), [123_000_000, 123_000_000])

    def test_counter_wrap_and_reset(self):
        self.assertEqual(counter_delta(950, 20, 1000), 70)
        self.assertEqual(counter_delta(2**60, 2**60 + 123, None), 123)
        for args in [(500, 20, 1000), (950, 200, 1000), (1000, 20, None),
                     (-1, 20, 1000), (10, 2000, 1000), (0, 1, 0)]:
            with self.subTest(args=args), self.assertRaises(RuntimeError):
                counter_delta(*args)

    def test_detect_missing_and_permission_denied_sources_reports_both(self):
        with patch("hardware._PowercapReader", side_effect=PermissionError("energy_uj denied")) as powercap, \
                patch("hardware._PerfReader", side_effect=PermissionError("perf_event_open denied")) as perf:
            powercap.method = "powercap"
            perf.method = "perf"
            meter = CpuEnergyMeter.detect()
            self.addCleanup(meter.close)
            self.assertFalse(meter.metadata()["available"])
            self.assertIn("energy_uj denied", meter.metadata()["unavailable_reason"])
            self.assertIn("perf_event_open denied", meter.metadata()["unavailable_reason"])

    def test_perf_rejects_multiplexed_energy_instead_of_scaling(self):
        import struct
        reader = _PerfReader.__new__(_PerfReader)
        reader.fds = [3]
        with patch("hardware.os.read", return_value=struct.pack("QQQ", 123, 100, 90)):
            with self.assertRaisesRegex(RuntimeError, "multiplexed"):
                reader.read()
        with patch("hardware.os.read", return_value=struct.pack("QQQ", 123, 100, 100)):
            self.assertEqual(reader.read(), [123])


class EnergyTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.reader = FakeReader(self.clock)
        self.meter = CpuEnergyMeter(self.reader, clock=self.clock, sleep=self.clock.advance)
        self.addCleanup(self.meter.close)

    def test_energy_unit_conversion_and_repeated_work(self):
        result = self.meter.measure(lambda: self.clock.advance(1), min_seconds=2.5, min_iterations=2)
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(result["iteration_seconds"], [1, 1, 1])
        self.assertEqual(result["energy_joules"], 300)
        self.assertEqual(result["average_watts"], 100)
        self.assertEqual(result["joules_per_iteration"], 100)
        self.assertEqual(result["samples"][-1]["elapsed_s"], 3)

    def test_unavailable_energy_is_null_with_real_latency(self):
        with CpuEnergyMeter(unavailable_reason="Permission denied", clock=self.clock) as meter:
            result = meter.measure(lambda: self.clock.advance(0.5), min_seconds=0)
        self.assertEqual(result["elapsed_seconds"], 0.5)
        self.assertFalse(result["available"])
        for field in ["energy_joules", "average_watts", "joules_per_iteration", "method"]:
            self.assertIsNone(result[field])
        self.assertEqual(result["unavailable_reason"], "Permission denied")

    def test_read_permission_loss_fails_window(self):
        def work():
            self.clock.advance(1)
            self.reader.failed = True
        with self.assertRaisesRegex(PermissionError, "permission lost"):
            self.meter.measure(work, 0)

    def test_reset_is_not_misreported_as_energy(self):
        def work():
            self.clock.advance(1)
            self.reader.offset = -1_000_000_000
        with self.assertRaisesRegex(RuntimeError, "reset"):
            self.meter.measure(work, 0)

    def test_stalled_counter_fails(self):
        with patch.object(self.reader, "read", return_value=[123]):
            with self.assertRaisesRegex(RuntimeError, "did not advance"):
                self.meter.measure(lambda: self.clock.advance(1), 0)

    def test_idle_and_close(self):
        result = self.meter.idle(2)
        self.assertEqual(result["energy_joules"], 200)
        self.assertEqual(result["kind"], "idle")
        self.meter.close()
        self.assertTrue(self.reader.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.meter.measure(lambda: None, 0)

    def test_workload_failure_propagates_and_unlocks(self):
        def work():
            raise LookupError("work failed")
        with self.assertRaisesRegex(LookupError, "work failed"):
            self.meter.measure(work, 0)
        self.assertEqual(self.meter.measure(lambda: self.clock.advance(1), 0)["iterations"], 1)

    def test_invalid_intervals_and_overlapping_measurement(self):
        for seconds, count in [(-1, 1), (math.inf, 1), (math.nan, 1), (0, 0), (0, True), (0, 1.5)]:
            with self.subTest(seconds=seconds, count=count), self.assertRaises(ValueError):
                self.meter.measure(lambda: None, seconds, count)
        with self.assertRaisesRegex(RuntimeError, "cannot overlap"):
            self.meter.measure(lambda: self.meter.measure(lambda: None, 0), 0)


if __name__ == "__main__":
    unittest.main()
