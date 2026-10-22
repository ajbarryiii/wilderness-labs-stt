"""Tests of the state sampler's parsers and window invalidation (M6, M9, M11, M12). The live probes
(ctypes host_processor_info / proc_pid_rusage / sysctlbyname, rusage units) run only on macOS."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # test_e2e's fakes (python3 -I drops the script dir)
import macstate as M  # noqa: E402

BATT_AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=27656291)\t100%; charged; 0:00 remaining present: true\n"
BATT_BAT = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=27656291)\t97%; discharging; 5:12 remaining present: true\n"
THERM_OK = ("Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\n"
            "Note: No CPU power status has been recorded\n")
THERM_HOT = "Note: No thermal warning level has been recorded\nCPU_Scheduler_Limit \t= 100\nCPU_Available_CPUs \t= 10\nCPU_Speed_Limit \t= 72\n"
THERM_WARN = "Thermal warning level set to 1.\nNote: No performance warning level has been recorded\n"
VM = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPageouts:                                    2418550.\n"
      "Swapouts:                                  111354607.\n")
HID = '  |   "HIDIdleTime" = 5453139867583\n'


class Parsers(unittest.TestCase):
    def test_batt(self):
        self.assertEqual(M.parse_batt(BATT_AC), {"ac": True, "battery_pct": 100, "battery_state": "charged"})
        self.assertEqual(M.parse_batt(BATT_BAT)["ac"], False)

    def test_therm(self):
        self.assertTrue(M.parse_therm(THERM_OK)["nominal"])
        self.assertFalse(M.parse_therm(THERM_HOT)["nominal"])
        self.assertFalse(M.parse_therm(THERM_WARN)["nominal"])

    def test_misc(self):
        self.assertEqual(M.parse_lowpower(" lowpowermode         0\n"), 0)
        self.assertEqual(M.parse_lowpower(" lowpowermode         1\n"), 1)
        self.assertEqual(M.parse_hid_idle_ns(HID), 5453139867583)
        self.assertEqual(M.parse_vm_stat(VM), {"pageouts": 2418550, "swapouts": 111354607})


def S(t_s, **kw):
    base = dict(ac=True, battery_pct=100, therm_nominal=True, lowpower=0, hid_idle_ns=10**12, load1=0.5, pressure=1,
                pageouts=10, swapouts=20, host_busy_cores=8.1, own_busy_cores=8.0, foreign_busy_cores=0.1)
    base.update(kw)
    return M.StateSample(int(t_s * 1e9), **base)


GONE, S_ = ("gone",), 1_000_000_000


def ok(cum_s, start_s=0.0):
    return ("ok", int(cum_s * S_), int(start_s * S_))


class Ledger(unittest.TestCase):
    """Own CPU reconciliation across process starts and exits (Codex code review c2 finding 2)."""

    def run_steps(self, steps):
        led = M.OwnCpuLedger()
        out = []
        for t_s, probes, ch_s in steps:
            out.append(led.step(int(t_s * S_), probes, int(ch_s * S_), int(ch_s * S_)))
        return out

    def test_codex_repro_reaped_child_is_not_double_counted(self):
        """A child advancing from 100 to 101 CPU-seconds, then reaped: host busy 3 cores. The c2 code
        gave own 101 cores and foreign -98; the interval holds 1 own core and 2 foreign cores."""
        (_, e0), (own, e1) = self.run_steps([(10, {1: ok(5, 0), 7: ok(100, 1)}, 0.0),
                                             (11, {1: ok(5, 0), 7: GONE}, 101.0)])
        self.assertEqual((e0, e1), ([], []))
        self.assertAlmostEqual(own / S_, 1.0)
        fields, errs = M.cpu_fields(1.0, 3.0, own, 10)
        self.assertEqual(errs, [])
        self.assertAlmostEqual(fields["foreign_busy_cores"], 2.0)
        samples = [Sv(t, foreign_busy_cores=0.0) for t in range(0, 11)]
        samples.append(Sv(11, **{k: v for k, v in fields.items() if k != "host_busy_cores"}))
        r = M.window_state(samples, 0, int(10.5e9))
        self.assertFalse(r["valid"])
        self.assertTrue(any("foreign busy" in x for x in r["reasons"]), r["reasons"])

    def test_child_exit_with_foreign_load(self):
        """Self 0.1 core, a child 1 core for 3 s that exits during the 4th second (0.5 s CPU), with 2
        foreign cores throughout: every interval reports ~2 foreign cores."""
        steps = [(0, {1: ok(0.0, -5), 7: ok(0.0, -0.0)}, 0.0)]
        cum_self, cum_child = 0.0, 0.0
        for t in (1, 2, 3):
            cum_self += 0.1
            cum_child += 1.0
            steps.append((t, {1: ok(cum_self, -5), 7: ok(cum_child, -0.0)}, 0.0))
        cum_self += 0.1
        steps.append((4, {1: ok(cum_self, -5), 7: GONE}, cum_child + 0.5))   # reaped: lifetime 3.5 s
        res = self.run_steps(steps)
        owns = [o / S_ for o, _ in res[1:]]
        for o, (h, e) in zip(owns, ((3.1, None), (3.1, None), (3.1, None), (2.6, None))):
            fields, errs = M.cpu_fields(1.0, h, int(o * S_), 10)
            self.assertAlmostEqual(fields["foreign_busy_cores"], 2.0, places=6)
            self.assertEqual(errs, [])
        self.assertTrue(all(not e for _, e in res))

    def test_zombie_then_reap(self):
        res = self.run_steps([(0, {7: ok(1.0, -1)}, 0.0), (1, {7: ok(1.4, -1)}, 0.0),   # exited, not reaped
                              (2, {7: GONE}, 1.4)])                                  # reaped: nothing new
        self.assertAlmostEqual(res[1][0] / S_, 0.4)
        self.assertAlmostEqual(res[2][0] / S_, 0.0)
        self.assertEqual([e for _, e in res], [[], [], []])

    def test_untracked_children_count_when_reaped_and_new_pids_from_their_start(self):
        res = self.run_steps([(0, {1: ok(1.0, -1)}, 0.0),
                              (1, {1: ok(1.1, -1), 9: ok(0.3, 0.5)}, 0.04)])   # pmset runs (0.04 s) + new child
        self.assertAlmostEqual(res[1][0] / S_, 0.1 + 0.3 + 0.04)
        self.assertEqual(res[1][1], [])

    def test_impossible_deltas_are_errors(self):
        late = self.run_steps([(0, {1: ok(1.0, -1)}, 0.0), (1, {1: ok(1.1, -1), 9: ok(5.0, -3)}, 0.0)])
        self.assertTrue(any("first seen" in e for e in late[1][1]))
        back = self.run_steps([(0, {1: ok(1.0, -1)}, 0.0), (1, {1: ok(0.9, -1)}, 0.0)])
        self.assertTrue(any("decreased" in e for e in back[1][1]))
        not_ours = self.run_steps([(0, {1: ok(1.0, -1), 7: ok(2.0, -1)}, 0.0), (1, {1: ok(1.1, -1), 7: GONE}, 0.0)])
        self.assertTrue(any("not reaped by us" in e for e in not_ours[1][1]))
        err = self.run_steps([(0, {1: ok(1.0, -1), 7: ("error", "proc_pid_rusage(7): EPERM")}, 0.0)])
        self.assertTrue(err[0][1])
        led = M.OwnCpuLedger()
        led.step(0, {1: ok(1.0, -1)}, 0, 0)
        _, e = led.step(S_, {1: ok(1.1, -1)}, 0, int(0.5 * S_))   # a reap raced the probes
        self.assertTrue(any("reaped during the probes" in x for x in e))
        _, errs = M.cpu_fields(1.0, 2.0, int(2.5 * S_), 10)      # own above host beyond the margin
        self.assertTrue(errs)
        _, errs = M.cpu_fields(1.0, 2.0, int(2.12 * S_), 10)     # within the 0.1 tick margin + 0.05
        self.assertEqual(errs, [])


def Sv(t_s, **kw):
    base = dict(ac=True, battery_pct=100, therm_nominal=True, lowpower=0, hid_idle_ns=10**13, load1=0.5, pressure=1,
                pageouts=10, swapouts=20, host_busy_cores=1.0, own_busy_cores=1.0, foreign_busy_cores=0.0)
    base.update(kw)
    return M.StateSample(int(t_s * 1e9), **base)


class Window(unittest.TestCase):
    def test_clean_window(self):
        r = M.window_state([S(t) for t in range(0, 14)], 0, int(13e9))
        self.assertTrue(r["valid"], r["reasons"])

    def test_each_reason(self):
        cases = [
            (dict(ac=False), "power transition"), (dict(battery_pct=99), "power transition"),
            (dict(therm_nominal=False), "thermal"), (dict(lowpower=1), "Low Power Mode"),
            (dict(hid_idle_ns=int(1e9)), "user input"), (dict(pressure=2), "memory pressure"),
            (dict(foreign_busy_cores=0.7), "foreign busy"), (dict(swapouts=21), "swap-out"),
            (dict(pageouts=11), "swap-out"),
        ]
        for kw, why in cases:
            samples = [S(t) for t in range(0, 14)]
            samples[7] = S(7, **kw)
            if "swapouts" in kw or "pageouts" in kw:
                samples[8:] = [S(t, **kw) for t in range(8, 14)]
            r = M.window_state(samples, 0, int(13e9))
            self.assertFalse(r["valid"], kw)
            self.assertTrue(any(why in x for x in r["reasons"]), (kw, r["reasons"]))

    def test_probe_failure_and_sampling_gap(self):
        samples = [S(t) for t in range(0, 14)]
        samples[3].errors.append("ioreg HIDIdleTime: TimeoutExpired")
        self.assertFalse(M.window_state(samples, 0, int(13e9))["valid"])
        self.assertFalse(M.window_state([S(0), S(13)], 0, int(13e9))["valid"])

    def test_codex_repro_input_at_window_end_seen_afterwards(self):
        """Review c1 finding 6: input at 12.5 s inside [0.2, 12.7], observed only by the 13 s sample."""
        samples = [S(t, hid_idle_ns=int((t + 20) * 1e9)) for t in range(0, 13)]
        samples.append(S(13, hid_idle_ns=int(0.5e9)))
        r = M.window_state(samples, int(0.2e9), int(12.7e9))
        self.assertFalse(r["valid"])
        self.assertTrue(any("user input" in x for x in r["reasons"]), r["reasons"])

    def test_cpu_delta_ending_after_the_window_counts(self):
        samples = [S(t) for t in range(0, 13)] + [S(13, foreign_busy_cores=2.0)]
        r = M.window_state(samples, int(0.2e9), int(12.7e9))
        self.assertFalse(r["valid"])
        self.assertTrue(any("foreign busy" in x for x in r["reasons"]), r["reasons"])

    def test_bracketing_samples_required(self):
        samples = [S(t) for t in range(1, 13)]
        r = M.window_state(samples, int(0.5e9), int(12.7e9))
        self.assertFalse(r["valid"])
        self.assertIn("no state sample at or before the start", r["reasons"])
        self.assertIn("no state sample at or after the end", r["reasons"])

    def test_swap_compared_across_bracketing_samples(self):
        samples = [S(t) for t in range(0, 13)] + [S(13, swapouts=21)]
        self.assertFalse(M.window_state(samples, int(0.2e9), int(12.7e9))["valid"])

    def test_missing_probe_value_invalidates(self):
        samples = [S(t) for t in range(0, 14)]
        samples[5] = S(5, foreign_busy_cores=None)
        self.assertFalse(M.window_state(samples, 0, int(13e9))["valid"])

    def test_input_before_window_is_fine(self):
        # Idle for 20 s at window start (input before t0) is not input during the window.
        samples = [S(t, hid_idle_ns=int((t + 20) * 1e9)) for t in range(0, 14)]
        self.assertTrue(M.window_state(samples, 0, int(13e9))["valid"])


@unittest.skipUnless(sys.platform == "darwin", "live macOS probes")
class LiveMac(unittest.TestCase):
    def test_native_probes(self):
        n = M.Native()
        ticks = n.cpu_ticks()
        self.assertEqual(len(ticks), os.cpu_count())
        self.assertTrue(all(t >= b > 0 for b, t in ticks))
        self.assertGreater(n.pid_cpu_ns(os.getpid()), 0)
        self.assertIn(n.sysctl_int("kern.memorystatus_vm_pressure_level"), (1, 2, 4))

    def test_rusage_units(self):
        r = M.validate_rusage_units(0.4)
        self.assertGreater(r["ratio"], 0.9, r)
        self.assertLess(r["ratio"], 1.1, r)

    def test_sampler_attributes_a_child_burn_and_its_exit_to_own_load(self):
        """A registered child burns ~0.6 core-s and is reaped and deregistered (as the runner does)
        while sampled: own CPU over the run matches its CPU (no double count at the reap) and no
        impossible delta is reported."""
        import subprocess
        import time
        active = []
        s = M.Sampler(lambda: [os.getpid()] + list(active), interval_s=0.5).start()
        time.sleep(0.6)
        # Burns a fixed ~0.6 s of CPU (not of wall time: a background job on a loaded Mac gets little CPU).
        child = subprocess.Popen([sys.executable, "-I", "-c",
                                  "import time\nwhile time.process_time() < 0.6: pass"])
        active.append(child.pid)
        child.wait()
        active.remove(child.pid)
        time.sleep(1.6)
        s.stop()
        got = s.snapshot()
        errs = [e for g in got for e in g.errors]
        self.assertFalse([e for e in errs if "reaped during the probes" not in e], errs)
        self.assertLessEqual(len(errs), 1, errs)
        valid = [g for g in got if g.own_busy_cores is not None and not g.errors]
        own_s = sum(g.own_busy_cores * 0.5 for g in valid)
        self.assertTrue(all(g.own_busy_cores <= g.host_busy_cores + g.foreign_margin_cores + 0.05 for g in valid),
                        [g.__dict__ for g in got])
        self.assertLess(own_s, 3.0, [g.__dict__ for g in got])      # the c2 bug would add the lifetime again
        self.assertGreater(own_s, 0.15, [g.__dict__ for g in got])   # the child's CPU is attributed to us


class SamplerSchedule(unittest.TestCase):
    """The c4 plumbing run on the Mac (background priority) showed the sampler, after a stall, catching up
    with back-to-back samples ~0.2 s apart, whose tick margin (ncpu / (STATHZ x dt)) then fails the
    foreign-CPU gate by itself. After a late sample the next one must come at least half an interval
    later (portable: fake probes)."""

    def test_no_back_to_back_catch_up_after_a_stall(self):
        import time
        from test_e2e import FakeNative, fake_text
        calls = {"n": 0}

        def slow_text(cmd, timeout=5.0):
            if cmd[:3] == ["pmset", "-g", "batt"]:
                calls["n"] += 1
                if calls["n"] == 2:
                    time.sleep(1.2)                     # a stalled probe: three intervals late
            return fake_text(cmd, timeout)
        s = M.Sampler(lambda: [os.getpid()], interval_s=0.4, native=FakeNative(), text=slow_text).start()
        time.sleep(3.5)
        s.stop()
        ts = [x.t_ns for x in s.snapshot()]
        gaps = [(b - a) / 1e9 for a, b in zip(ts, ts[1:])]
        self.assertGreaterEqual(len(gaps), 4, gaps)
        self.assertGreaterEqual(min(gaps), 0.4 * 0.5 - 0.05, gaps)


if __name__ == "__main__":
    unittest.main()
