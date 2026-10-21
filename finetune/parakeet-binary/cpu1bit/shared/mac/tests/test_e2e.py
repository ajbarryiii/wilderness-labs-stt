"""End-to-end test of the real runner (Codex code review c4, blocker): runner.run_block drives a whole block
with its real lifecycle (lock, state sampler thread, powermetrics Reader, arm process, bracketing, shutdown)
and its real analysis. Only the environment is faked: the powermetrics producer
(tests/fakes/fake_powermetrics.py, via pm.COMMAND), the arm process (tests/fakes/fake_block.py, given the
real runner.block_argv arguments), the state probes (macstate.Sampler's injected `native` and `text`), the
quiet-host wait, caffeinate, and the lock path. A clean block must be accepted, and the state samples
must bracket the whole span: a completed sample before the spawn and one after the end of the capture.

Contract additions (c4): runner.STATE_PROBES (keyword arguments for macstate.Sampler), runner.CAFFEINATE,
macstate.Sampler(own_pids, interval_s, native=None, text=run_text), pm.now_ns / macstate.now_ns on
CLOCK_MONOTONIC_RAW where CLOCK_UPTIME_RAW does not exist (Linux).

    python3 tests/test_e2e.py
"""
from __future__ import annotations

import os
import resource
import shutil
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import guard  # noqa: E402
import macstate  # noqa: E402
import pm  # noqa: E402
import runner  # noqa: E402
from test_state import BATT_AC, THERM_OK  # noqa: E402

FAKES = os.path.join(HERE, "fakes")
IDLE_W, WORK_W = 0.4, 6.0          # the fake producer's power model


class FakeNative:
    """State probes that report no foreign load: 10 CPUs whose busy ticks are exactly our own CPU (this
    process plus reaped children), live pids probed as alive (this process with its real CPU time),
    memory pressure normal."""

    def __init__(self):
        self.first_seen = {}

    def cpu_ticks(self):
        s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        busy_ticks = (s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime) * macstate.STATHZ
        total_ticks = pm.now_ns() * macstate.STATHZ / 1e9
        return [(int(busy_ticks / 10), int(total_ticks)) for _ in range(10)]

    def probe(self, pid):
        if pid == os.getpid():
            s = resource.getrusage(resource.RUSAGE_SELF)
            return ("ok", int((s.ru_utime + s.ru_stime) * 1e9), 0)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return ("gone",)
        return ("ok", 0, self.first_seen.setdefault(pid, pm.now_ns()))

    def sysctl_int(self, name):
        return 1


def fake_text(cmd, timeout=5.0):
    if cmd[:3] == ["pmset", "-g", "batt"]:
        return BATT_AC
    if cmd[:3] == ["pmset", "-g", "therm"]:
        return THERM_OK
    if cmd[:2] == ["pmset", "-g"]:
        return " lowpowermode         0\n"
    if cmd[0] == "ioreg":
        return '  |   "HIDIdleTime" = 999000000000000000\n'
    if cmd[0] == "vm_stat":
        return "Pageouts:     10.\nSwapouts:     20.\n"
    raise AssertionError(f"unexpected probe {cmd}")


def work_intervals(path):
    ivs, start = [], None
    with open(path) as f:
        for line in f:
            k, t = line.split()
            if k == "S":
                start = int(t)
            else:
                ivs.append((start, int(t)))
    return ivs


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.worklog = os.path.join(self.tmp, "work.log")
        open(self.worklog, "w").close()
        self.saved = (pm.COMMAND, runner.LOCK, runner.CAFFEINATE, runner.STATE_PROBES, runner.wait_quiet, runner.IDLE_S,
                      os.environ.get("CPU1BIT_FAKE_WORKLOG"))
        pm.COMMAND = [sys.executable, "-I", os.path.join(FAKES, "fake_powermetrics.py"), self.worklog]
        runner.LOCK = os.path.join(self.tmp, "lock", "active.lock")
        runner.CAFFEINATE = [sys.executable, "-I", "-c", "import time; time.sleep(3600)"]
        runner.STATE_PROBES = {"native": FakeNative(), "text": fake_text}
        runner.wait_quiet = lambda log: {"ok": True, "checks": {"thermal_nominal": True}, "load1": 0.5, "waited_s": 0.0}
        runner.IDLE_S = 4.0
        os.environ["CPU1BIT_FAKE_WORKLOG"] = self.worklog

    def tearDown(self):
        (pm.COMMAND, runner.LOCK, runner.CAFFEINATE, runner.STATE_PROBES, runner.wait_quiet, runner.IDLE_S, wl) = self.saved
        if wl is None:
            os.environ.pop("CPU1BIT_FAKE_WORKLOG", None)
        else:
            os.environ["CPU1BIT_FAKE_WORKLOG"] = wl
        shutil.rmtree(self.tmp)

    def test_real_runner_accepts_a_clean_block(self):
        sess = {"clips": [0, 1], "arms": {"A": {"binary": "fake", "export": "x", "captures": "y", "tuning": None}}}
        blk = {"arm": "A", "condition": {"threads": 8, "components": 2}, "block_id": "e2e", "stage": "e2e"}
        replays = [8, 8]
        argv = [sys.executable, "-I", os.path.join(FAKES, "fake_block.py")] + runner.block_argv(sess, blk, replays=replays)[1:]
        logs = []
        rec = runner.run_block(sess, argv, os.path.join(self.tmp, "e2e"), logs.append, replays=replays, delta_a2_ns=0)
        a = rec["analysis"]
        reasons = a["block_invalid_reasons"] + [r for w in a["windows"] for r in w["invalid_reasons"]]
        # The lifecycle (c4 blocker): a completed state sample before the spawn and one after the capture's end.
        ph, st = rec["phases"], rec["state_samples"]
        self.assertTrue(any(s["t_ns"] <= ph["spawn_ns"] for s in st), (ph, [s["t_ns"] for s in st][:3]))
        self.assertTrue(any(s["t_ns"] >= ph["span1"] for s in st), (ph, [s["t_ns"] for s in st][-3:]))
        self.assertTrue(a["span_state"]["valid"], a["span_state"]["reasons"])
        self.assertTrue(a["stream"]["ok"] and a["band"]["valid"] and a["rate"]["valid"], (a["stream"], a["band"], a["rate"]))
        self.assertTrue(a["storage"] and a["storage"]["ok"])
        self.assertEqual(rec["exit_code"], 0)
        self.assertTrue(a["valid"], reasons)
        self.assertEqual([w["replays"] for w in a["windows"]], replays)
        ivs = work_intervals(self.worklog)
        g = guard.G_NS
        for w in a["windows"]:
            self.assertTrue(w["valid"], w["invalid_reasons"])
            lo, hi = w["t0_ns"] - g, w["t1_ns"] + g
            busy = sum(max(0, min(hi, e) - max(lo, s)) for s, e in ivs)
            truth = (IDLE_W * (hi - lo) + WORK_W * busy) / 1e9 / w["replays"]
            e = w["energy_j_per_replay"]
            self.assertLessEqual(e["lower"], truth + 1e-9, (e, truth))
            self.assertGreaterEqual(e["upper"], truth - 1e-9, (e, truth))
            self.assertIsNotNone(w["guard_j_per_replay"])
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "e2e.powermetrics.plist0")))


if __name__ == "__main__":
    unittest.main()
