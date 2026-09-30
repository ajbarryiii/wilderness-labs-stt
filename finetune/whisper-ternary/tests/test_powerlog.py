"""powerlog.py: parsing, RAPL wrap, wall estimate, record shapes, summary, warnings, stopping.

No GPU and no /sys access: nvidia-smi, the wall command and every file read are faked.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import powerlog

GPU_LINE = ("2026/09/30 11:35:58.097, 312.45, 330.10, 300.00, 2610, 14001, 97, 61, "
            "Active, Not Active, 400.00")
FREQS = ["/fake/cpu0/cpufreq/scaling_cur_freq", "/fake/cpu1/cpufreq/scaling_cur_freq"]
TCTL = "/fake/hwmon1/temp1_input"


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class FakeHost:
    """Stands in for nvidia-smi, the wall command and the /proc and /sys files.

    /proc/stat advances 100 jiffies per read with 25 busy, so cpu_busy is 0.25.
    rapl=None means energy_uj is root-only; otherwise readings are popped in order.
    """

    def __init__(self, gpu_line: str = GPU_LINE, rapl: list[int] | None = None,
                 wall: str = "812.5\n") -> None:
        self.gpu_line, self.rapl, self.wall = gpu_line, rapl, wall
        self.stat_reads, self.rapl_reads, self.gpu_fail = 0, 0, None
        self.on_gpu_sample = None

    def run(self, cmd, **kwargs):
        if isinstance(cmd, str):  # --wall-command runs through the shell
            assert kwargs.get("shell") is True
            return subprocess.CompletedProcess(cmd, 0, self.wall, "")
        assert cmd[0] == "nvidia-smi" and kwargs.get("timeout")
        if "name,power.limit" in cmd[1]:
            return subprocess.CompletedProcess(cmd, 0, "NVIDIA GeForce RTX 5090, 400.00\n", "")
        if self.gpu_fail:
            raise self.gpu_fail
        if self.on_gpu_sample:
            self.on_gpu_sample()
        return subprocess.CompletedProcess(cmd, 0, self.gpu_line + "\n", "")

    def read(self, path: str) -> str:
        if path == powerlog.RAPL:
            if self.rapl is None:
                raise PermissionError(13, "Permission denied", path)
            self.rapl_reads += 1
            return f"{self.rapl.pop(0)}\n"
        if path == powerlog.RAPL_MAX:
            return f"{powerlog.RAPL_WRAP}\n"
        if path == "/proc/stat":
            self.stat_reads += 1
            n = self.stat_reads
            return f"cpu  {25 * n} 0 0 {75 * n} 0 0 0 0 0 0\ncpu0 1 2 3 4 5 6 7 8 0 0\n"
        if path in FREQS:
            return "3000000\n" if path == FREQS[0] else "5000000\n"
        if path == TCTL:
            return "45125\n"
        if path == "/proc/loadavg":
            return "1.35 0.48 0.44 18/745 6751\n"
        if path == "/proc/cpuinfo":
            return "processor\t: 0\nmodel name\t: AMD Ryzen 9 9950X3D 16-Core Processor\n"
        raise FileNotFoundError(path)


def make_sensors(host: FakeHost, clock: FakeClock, **kwargs) -> powerlog.Sensors:
    with mock.patch.object(powerlog, "discover", return_value=(FREQS, TCTL)):
        return powerlog.Sensors(read=host.read, run=host.run, clock=clock, **kwargs)


def record_args(**overrides) -> argparse.Namespace:
    base = {"interval": 1.0, "baseline_watts": 90.0, "psu_efficiency": 0.9,
            "warn_wall_watts": 700.0, "wall_command": None, "stop_file": None, "parent_pid": None}
    return argparse.Namespace(**{**base, **overrides})


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class ParsingTest(unittest.TestCase):
    def test_nvidia_smi_line(self):
        gpu = powerlog.parse_gpu(GPU_LINE)
        self.assertEqual(gpu, {"gpu_w": 312.45, "gpu_w_instant": 330.10, "gpu_sm_mhz": 2610.0,
                               "gpu_mem_mhz": 14001.0, "gpu_util": 97.0, "gpu_temp_c": 61.0,
                               "gpu_sw_power_cap": True, "gpu_hw_slowdown": False,
                               "gpu_power_limit_w": 400.0})

    def test_nvidia_smi_unsupported_fields_become_none(self):
        gpu = powerlog.parse_gpu(GPU_LINE.replace("330.10", "[N/A]").replace("Active,", "[N/A],", 1))
        self.assertIsNone(gpu["gpu_w_instant"])
        self.assertIsNone(gpu["gpu_sw_power_cap"])
        with self.assertRaises(ValueError):
            powerlog.parse_gpu("2026/09/30 11:35:58.097, 312.45")

    def test_rapl_delta_with_wrap(self):
        self.assertEqual(powerlog.rapl_delta(1_000_000, 4_000_000), 3_000_000)
        wrap = powerlog.RAPL_WRAP
        self.assertEqual(powerlog.rapl_delta(wrap - 1_000_000, 2_000_000), 3_000_000)
        self.assertEqual(powerlog.rapl_delta(9, 4, wrap_uj=10), 5)

    def test_cpu_package_watts_across_wrap(self):
        wrap, clock = powerlog.RAPL_WRAP, FakeClock()
        host = FakeHost(rapl=[wrap - 40_000_000, 60_000_000, 160_000_000])
        sensors = make_sensors(host, clock)
        self.assertEqual(sensors.cpu_energy, "powercap")
        clock.t += 1.0
        self.assertEqual(sensors.sample()["cpu_pkg_w"], 100.0)  # 100 J over 1 s, through the wrap
        clock.t += 2.0
        s = sensors.sample()
        self.assertEqual(s["cpu_pkg_w"], 50.0)
        self.assertEqual(s["est_wall_w_basis"], "rapl")

    def test_est_wall_both_bases(self):
        self.assertEqual(powerlog.est_wall(300.0, 100.0, 0.5, 90.0, 0.9), (544.4, "rapl"))
        # proxy: 0.5 * 170 + 30 = 115 W of CPU
        self.assertEqual(powerlog.est_wall(300.0, None, 0.5, 90.0, 0.9), (561.1, "cpu_busy_proxy"))
        self.assertEqual(powerlog.est_wall(None, 100.0, 0.5, 90.0, 0.9), (None, None))
        self.assertEqual(powerlog.est_wall(300.0, None, None, 90.0, 0.9), (None, None))


class SensorsTest(unittest.TestCase):
    def test_sample_without_rapl_uses_proxy(self):
        clock = FakeClock()
        sensors = make_sensors(FakeHost(), clock)
        self.assertEqual(sensors.cpu_energy, "unavailable (permission denied)")
        clock.t += 1.0
        s = sensors.sample()
        self.assertEqual(list(s), ["type", *powerlog.COLUMNS])
        self.assertEqual(s["type"], "sample")
        self.assertIsNone(s["cpu_pkg_w"])
        self.assertEqual(s["cpu_busy"], 0.25)
        self.assertEqual(s["cpu_freq_mhz_mean"], 4000.0)
        self.assertEqual(s["cpu_tctl_c"], 45.125)
        self.assertEqual(s["load1"], 1.35)
        self.assertIsNone(s["wall_w"])
        self.assertEqual(s["est_wall_w_basis"], "cpu_busy_proxy")
        self.assertAlmostEqual(s["est_wall_w"], round((312.45 + 0.25 * 170 + 30 + 90) / 0.9, 1))
        self.assertEqual(sensors.errors, [])

    def test_rapl_rechecked_every_60_s(self):
        clock, host = FakeClock(), FakeHost()
        sensors = make_sensors(host, clock)
        clock.t += 30
        self.assertIsNone(sensors.sample()["cpu_pkg_w"])
        host.rapl = [1_000_000, 51_000_000]  # the user ran chmod 444
        clock.t += 20  # 50 s after the first check: not re-checked yet
        self.assertIsNone(sensors.sample()["cpu_pkg_w"])
        self.assertEqual(host.rapl_reads, 0)
        clock.t += 15  # 65 s: re-check succeeds and primes the counter
        self.assertIsNone(sensors.sample()["cpu_pkg_w"])
        self.assertEqual(sensors.cpu_energy, "powercap")
        clock.t += 1
        s = sensors.sample()
        self.assertEqual((s["cpu_pkg_w"], s["est_wall_w_basis"]), (50.0, "rapl"))

    def test_sensor_failures_become_none_and_errors(self):
        clock, host = FakeClock(), FakeHost()
        host.gpu_fail = subprocess.CalledProcessError(9, ["nvidia-smi"])
        sensors = make_sensors(host, clock, wall_command="false")
        host.wall = "not a number"
        s = sensors.sample()
        self.assertIsNone(s["gpu_w"])
        self.assertIsNone(s["wall_w"])
        self.assertEqual((s["est_wall_w"], s["est_wall_w_basis"]), (None, None))
        self.assertEqual(s["cpu_busy"], 0.25)
        self.assertEqual(len(sensors.errors), 2)
        self.assertTrue(sensors.errors[0].startswith("nvidia-smi: CalledProcessError"))
        self.assertTrue(sensors.errors[1].startswith("wall-command: ValueError"))


class RecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.out = self.dir / "run.jsonl"
        # Count fsyncs without paying for them: TMPDIR is on a spinning disk.
        patcher = mock.patch.object(powerlog.os, "fsync")
        self.fsync = patcher.start()
        self.addCleanup(patcher.stop)

    def run_until(self, host, clock, args, seconds, stop=None, **kwargs) -> list[dict]:
        """Run the loop, touching the stop file once the fake clock passes `seconds`."""
        sensors = make_sensors(host, clock, **kwargs)
        start = clock.t
        stop_file = Path(args.stop_file)

        def sleep(dt):
            clock.sleep(dt)
            if clock.t - start >= seconds:
                stop_file.touch()
        reason = powerlog.run_loop(self.out, args, sensors, stop, sleep=sleep)
        records = read_records(self.out)
        self.assertEqual(records[-1]["reason"], reason)
        return records

    def test_record_shapes_fsync_and_stop_file(self):
        clock, args = FakeClock(), record_args(stop_file=str(self.dir / "stop"))
        records = self.run_until(FakeHost(), clock, args, seconds=3)
        self.assertEqual(self.fsync.call_count, len(records))  # every record is synced
        self.assertTrue(self.out.read_text().endswith("}\n"))
        header, samples, end = records[0], records[1:-1], records[-1]
        self.assertEqual(header["type"], "header")
        for key in ("started_utc", "hostname", "interval_s", "baseline_watts", "psu_efficiency",
                    "gpu_name", "gpu_power_limit_w", "cpu_model", "cpu_energy", "columns"):
            self.assertIn(key, header)
        self.assertEqual(header["gpu_name"], "NVIDIA GeForce RTX 5090")
        self.assertEqual(header["gpu_power_limit_w"], 400.0)
        self.assertEqual(header["cpu_model"], "AMD Ryzen 9 9950X3D 16-Core Processor")
        self.assertEqual(header["cpu_energy"], "unavailable (permission denied)")
        self.assertEqual(header["columns"], powerlog.COLUMNS)
        self.assertEqual([s["type"] for s in samples], ["sample"] * 3)
        self.assertEqual([s["t_mono"] for s in samples], [1000.0, 1001.0, 1002.0])
        for s in samples:
            self.assertEqual(list(s), ["type", *powerlog.COLUMNS])
            self.assertIsInstance(s["gpu_sw_power_cap"], bool)
        self.assertEqual(end, {"type": "end", "t_utc": end["t_utc"], "t_mono": 1003.0,
                               "reason": "stop-file", "samples": 3})

    def test_warn_rate_limited_to_once_per_30_s(self):
        clock = FakeClock()
        args = record_args(stop_file=str(self.dir / "stop"), warn_wall_watts=400.0)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            records = self.run_until(FakeHost(), clock, args, seconds=65)
        warns = [r for r in records if r["type"] == "warn"]
        self.assertEqual(sum(r["type"] == "sample" for r in records), 65)
        self.assertEqual([w["t_mono"] for w in warns], [1000.0, 1030.0, 1060.0])
        self.assertEqual(set(warns[0]), {"type", "t_utc", "t_mono", "watts", "basis", "threshold_w"})
        self.assertEqual((warns[0]["basis"], warns[0]["threshold_w"]), ("est_wall_w", 400.0))
        self.assertGreater(warns[0]["watts"], 400.0)
        self.assertEqual(err.getvalue().count("WARN"), 3)

    def test_warn_prefers_metered_wall_power(self):
        clock = FakeClock()
        args = record_args(stop_file=str(self.dir / "stop"), wall_command="plug-watts")
        with contextlib.redirect_stderr(io.StringIO()):
            records = self.run_until(FakeHost(wall="812.5\n"), clock, args, seconds=2,
                                     wall_command="plug-watts")
        samples = [r for r in records if r["type"] == "sample"]
        self.assertEqual([s["wall_w"] for s in samples], [812.5, 812.5])
        self.assertLess(samples[0]["est_wall_w"], 700)  # only the meter is over the threshold
        warn, = [r for r in records if r["type"] == "warn"]
        self.assertEqual((warn["basis"], warn["watts"]), ("wall_w", 812.5))

    def test_errors_logged_once_and_sampling_continues(self):
        clock, host = FakeClock(), FakeHost()
        host.gpu_fail = subprocess.TimeoutExpired(["nvidia-smi"], 5)
        records = self.run_until(host, clock, record_args(stop_file=str(self.dir / "stop")), seconds=5)
        errors = [r for r in records if r["type"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("nvidia-smi: TimeoutExpired", errors[0]["message"])
        samples = [r for r in records if r["type"] == "sample"]
        self.assertEqual(len(samples), 5)
        self.assertTrue(all(s["gpu_w"] is None and s["cpu_busy"] == 0.25 for s in samples))

    def test_signal_stops_with_end_record(self):
        clock, host = FakeClock(), FakeHost()
        stop: dict = {}
        host.on_gpu_sample = lambda: stop.setdefault("reason", "SIGTERM") if clock.t >= 1002 else None
        records = self.run_until(host, clock, record_args(stop_file=str(self.dir / "stop")),
                                 seconds=1e9, stop=stop)
        self.assertEqual(records[-1]["type"], "end")
        self.assertEqual((records[-1]["reason"], records[-1]["samples"]), ("SIGTERM", 3))

    def test_cpu_energy_change_is_recorded(self):
        clock, host = FakeClock(), FakeHost()
        host.on_gpu_sample = lambda: setattr(host, "rapl", [5_000_000] * 100) if clock.t >= 1030 else None
        records = self.run_until(host, clock, record_args(stop_file=str(self.dir / "stop")), seconds=63)
        change, = [r for r in records if r["type"] == "cpu_energy"]
        self.assertEqual(change["status"], "powercap")
        self.assertEqual(records[0]["cpu_energy"], "unavailable (permission denied)")


class SummaryTest(unittest.TestCase):
    def write_log(self, path: Path, torn: str) -> None:
        lines = [{"type": "header", "hostname": "nixos", "gpu_name": "NVIDIA GeForce RTX 5090",
                  "gpu_power_limit_w": 400.0, "cpu_model": "AMD Ryzen 9 9950X3D",
                  "cpu_energy": "unavailable (permission denied)", "columns": powerlog.COLUMNS}]
        for i in range(40):
            lines.append({"type": "sample", "t_utc": f"2026-09-30T12:00:{i:02d}.000Z",
                          "t_mono": 500.0 + i, "gpu_w": 100.0 + 10 * i, "gpu_w_instant": 110.0 + 10 * i,
                          "cpu_pkg_w": None, "cpu_busy": 0.5, "gpu_sw_power_cap": i >= 30,
                          "wall_w": None, "est_wall_w": 300.0 + 10 * i,
                          "est_wall_w_basis": "cpu_busy_proxy"})
            if i == 35:
                lines.append({"type": "warn", "t_mono": 535.0, "watts": 650.0})
        lines.append({"type": "error", "message": "k10temp: no Tctl sensor found"})
        path.write_text("".join(json.dumps(r) + "\n" for r in lines) + torn)

    def test_truncated_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "power.jsonl"
            self.write_log(path, '{"type": "sample", "t_utc": "2026-09-30T12:00:40.0')
            text = powerlog.summarize(path, tail_seconds=30)
        self.assertIn("cpu_energy: unavailable (permission denied)", text)
        self.assertIn("skipped 1 unparseable line(s)", text)
        self.assertIn("end: NONE", text)
        self.assertIn("duration 39.0 s, 40 samples", text)
        self.assertRegex(text, r"gpu_w +mean +295\.0 +max +490\.0 +\(n=40\)")
        self.assertRegex(text, r"gpu_w_instant +mean +305\.0 +max +500\.0")
        self.assertRegex(text, r"est_wall_w +mean +495\.0 +max +690\.0")
        self.assertRegex(text, r"cpu_pkg_w +n/a")
        self.assertRegex(text, r"wall_w +n/a")
        self.assertIn("gpu_sw_power_cap active in 25.0% of samples", text)
        self.assertIn("warnings 1; errors 1", text)
        self.assertIn("error: k10temp: no Tctl sensor found", text)
        table = text.split("last 30 s:\n")[1].splitlines()[1:]
        self.assertEqual(len(table), 31)  # t_mono 509 .. 539
        self.assertEqual(table[-1].split(), ["12:00:39.000", "0.0", "490.0", "-", "690.0", "-"])
        self.assertEqual(table[0].split()[:2], ["12:00:09.000", "-30.0"])

    def test_nul_filled_tail_and_clean_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "power.jsonl"
            self.write_log(path, json.dumps({"type": "end", "reason": "SIGTERM"}) + "\n" + "\x00" * 64)
            text = powerlog.summarize(path)
        self.assertIn("end: SIGTERM", text)
        self.assertNotIn("skipped", text)


class RecorderProcessTest(unittest.TestCase):
    def test_start_recorder_detaches_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(powerlog, "POWER_DIR", Path(tmp)), \
                mock.patch.object(powerlog.paths, "require_mount"), \
                mock.patch.object(powerlog.subprocess, "Popen") as popen:
            proc = powerlog.start_recorder("sweep-v2", warn_wall_watts=650, wall_command=None)
            self.assertTrue((Path(tmp) / "sweep-v2.log").exists())
        self.assertIs(proc, popen.return_value)
        cmd, kwargs = popen.call_args.args[0], popen.call_args.kwargs
        self.assertEqual(cmd[2:5], ["record", "--name", "sweep-v2"])
        self.assertEqual(cmd[5:7], ["--parent-pid", str(os.getpid())])
        self.assertEqual(cmd[7:], ["--warn-wall-watts", "650"])
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(Path(kwargs["stdout"].name).name, "sweep-v2.log")
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)

    def test_stop_recorder_terms_then_kills(self):
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired("record", 10), None]
        powerlog.stop_recorder(proc, timeout=10)
        proc.send_signal.assert_called_once_with(signal.SIGTERM)
        proc.kill.assert_called_once()
        clean = mock.Mock()
        clean.poll.return_value = None
        powerlog.stop_recorder(clean)
        clean.send_signal.assert_called_once_with(signal.SIGTERM)
        clean.kill.assert_not_called()


class SweepIntegrationTest(unittest.TestCase):
    """sweep.main starts the recorder after Protocol and always stops it; failures are non-fatal."""

    def run_main(self, argv: list[str], phase=None, start_error=None):
        import sweep
        phase = phase or mock.Mock()
        proto = mock.Mock(train_splits=["train-clean-100"], lrs={})
        proto.name = "v2"
        start = mock.Mock(side_effect=start_error)
        with mock.patch.object(sweep, "Protocol", return_value=proto), \
                mock.patch.dict(sweep.PHASES, {"zeroshot": phase}, clear=True), \
                mock.patch.object(sweep.paths, "require_mount"), \
                mock.patch.object(sweep.powerlog, "start_recorder", start), \
                mock.patch.object(sweep.powerlog, "stop_recorder") as stop, \
                mock.patch.object(sys, "argv", ["sweep.py", "--protocol", "v2", *argv]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.start, self.stop = start, stop
            try:
                sweep.main()
            finally:
                self.out = out.getvalue()
        return start, stop, phase

    def test_started_and_stopped(self):
        start, stop, phase = self.run_main(["--phase", "zeroshot"])
        start.assert_called_once_with("sweep-v2")
        stop.assert_called_once_with(start.return_value)
        phase.assert_called_once()

    def test_stopped_when_a_phase_fails(self):
        phase = mock.Mock(side_effect=subprocess.CalledProcessError(1, ["train.py"]))
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_main(["--phase", "zeroshot"], phase=phase)
        self.start.assert_called_once_with("sweep-v2")
        self.stop.assert_called_once_with(self.start.return_value)  # the finally block ran

    def test_not_started_for_dry_run_or_opt_out(self):
        for flag in ("--dry-run", "--no-powerlog"):
            start, stop, phase = self.run_main(["--phase", "zeroshot", flag])
            start.assert_not_called()
            stop.assert_not_called()
            phase.assert_called_once()

    def test_start_failure_is_non_fatal(self):
        start, stop, phase = self.run_main(["--phase", "zeroshot"], start_error=OSError("no disk"))
        phase.assert_called_once()
        stop.assert_not_called()
        self.assertIn("power recorder not started", self.out)


if __name__ == "__main__":
    unittest.main()
