"""Tests of the validate stage's pure parts and the sizing inputs (plans/cpu-benchmark-mac.md "Assumptions A1-A3
and validation", "Fixed N and sizing"):
  runner.validate_capture(cap) -> {"ok", "reasons", "band", "rate", "jitter_ns", "u_ns", "idle_reference",
                                   "energy_interval", "spin_excess_power_w", ...}
      cap = {"spawn_ns", "samples", "powermetrics": {"stream"}, "marks": {"idle0", "idle1", "spins"}}
  runner.delta_a2_ns(tol_mj, spin_excess_w) -> ns or None        (delta_A2 = tol / spin excess power)
  runner.instrument_ok(analysis) -> (ok, reasons, median relative width)
  runner.sizing_from_blocks({arm: block record}, clips) -> {"latency_s", "plan"}

    python3 tests/test_validate.py
"""
from __future__ import annotations

import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import guard  # noqa: E402
import pm  # noqa: E402
import runner  # noqa: E402
from test_pm import MS, S, synthetic  # noqa: E402

SPAWN = 10**12
IDLE0, IDLE1 = SPAWN + 1 * S, SPAWN + 11 * S
IDLE_W, SPIN_W = 0.4, 2.0


def capture(seed=1, rate_ppm=0, energy_lead=False, span_s=60):
    spins = []
    t = IDLE1
    for _ in range(3):
        t += int(runner.SPIN_GAP_S * S)
        spins.append({"on_ns": t, "off_ns": t + int(runner.SPIN_S * S)})
        t = spins[-1]["off_ns"]

    def power(x):
        return IDLE_W + (SPIN_W if any(s["on_ns"] <= x < s["off_ns"] for s in spins) else 0.0)
    n = int(span_s * S / (107 * MS))
    samples, _ = synthetic(power, n, seed, latency_max_ns=2 * MS, noise_mj=1.0, start_ns=SPAWN + 200 * MS)
    if rate_ppm:
        for x in samples:
            x.elapsed_ns = int(x.elapsed_ns * (1 + rate_ppm * 1e-6))
    if energy_lead:                       # each record carries the NEXT interval's energy
        es = [x.cpu_energy_mj for x in samples]
        for x, e in zip(samples, es[1:] + es[-1:]):
            x.cpu_energy_mj = e
    pm.build_timeline(samples)
    return {"spawn_ns": SPAWN, "samples": samples, "powermetrics": {"stream": {"ok": True, "reasons": []}},
            "marks": {"idle0": IDLE0, "idle1": IDLE1, "spins": spins}}


class ValidateCapture(unittest.TestCase):
    def test_clean_capture_passes_and_measures_the_spin(self):
        r = runner.validate_capture(capture())
        self.assertTrue(r["ok"], r["reasons"])
        self.assertTrue(r["band"]["valid"] and r["rate"]["valid"] and r["energy_interval"]["valid"])
        self.assertAlmostEqual(r["spin_excess_power_w"], SPIN_W, delta=0.1)
        self.assertLessEqual(r["u_ns"], guard.U_MAX_NS)
        self.assertIsNotNone(r["jitter_ns"])
        d = runner.delta_a2_ns(r["idle_reference"]["tol_mj"], r["spin_excess_power_w"])
        self.assertGreater(d, 0)
        self.assertLess(d, 10 * MS)                       # a few ms at about 5 mJ / 2 W

    def test_energy_reported_one_interval_early_fails(self):
        """The falsification test: energy and elapsed_ns covering different intervals shows up as excess in
        records that end before a spin starts."""
        r = runner.validate_capture(capture(seed=2, energy_lead=True))
        self.assertFalse(r["ok"])
        self.assertTrue(any("energy-interval" in x for x in r["reasons"]), r["reasons"])

    def test_rate_error_fails(self):
        r = runner.validate_capture(capture(seed=3, rate_ppm=1000))
        self.assertFalse(r["ok"])
        self.assertTrue(any("rate" in x for x in r["reasons"]), r["reasons"])

    def test_stream_failure_fails(self):
        cap = capture(seed=4)
        cap["powermetrics"]["stream"] = {"ok": False, "reasons": ["EOF before the runner stopped powermetrics"]}
        r = runner.validate_capture(cap)
        self.assertFalse(r["ok"])
        self.assertTrue(any("stream" in x for x in r["reasons"]), r["reasons"])


class DeltaA2(unittest.TestCase):
    def test_values(self):
        self.assertEqual(runner.delta_a2_ns(6.0, 2.0), 3 * MS)
        for bad in ((None, 2.0), (6.0, None), (6.0, math.nan), (6.0, 0.0), (6.0, -1.0), (0.0, 2.0)):
            self.assertIsNone(runner.delta_a2_ns(*bad), bad)


def window(width, u_ns=300 * MS, valid=True):
    return {"energy_j_window": {"lower": 10.0, "est": 10.0 + width * 5, "upper": 10.0 + width * 10.0 + width * 0},
            "u_ns": u_ns, "valid": valid, "invalid_reasons": [] if valid else ["x"]}


class Instrument(unittest.TestCase):
    def analysis(self, ws, valid=True):
        return {"valid": valid, "block_invalid_reasons": [], "windows": ws}

    def test_pass_and_fail(self):
        ok, reasons, med = runner.instrument_ok(self.analysis([window(0.005)] * 8))
        self.assertTrue(ok, reasons)
        self.assertLess(med, 0.01)
        ok, reasons, _ = runner.instrument_ok(self.analysis([window(0.03)] * 8))
        self.assertFalse(ok)
        ok, reasons, _ = runner.instrument_ok(self.analysis([window(0.005, u_ns=guard.U_MAX_NS + 1)] * 8))
        self.assertFalse(ok)
        ok, reasons, _ = runner.instrument_ok(self.analysis([window(0.005, valid=False)] * 8, valid=False))
        self.assertFalse(ok)


class SizingInputs(unittest.TestCase):
    def test_latency_and_plan_from_events(self):
        g, sp, spost = guard.G_NS, guard.S_NS, guard.S_POST_NS
        ev = [{"event": "start", "t_ns": 0}, {"event": "loaded", "t_ns": 1 * S}, {"event": "warmup_done", "t_ns": 20 * S},
              {"event": "storage", "t_ns": 20 * S}]
        t, lat = 20 * S, [0.4, 0.6]
        for i, l in enumerate(lat):
            t0 = t + 300 * MS + int(l * S) + sp + g                     # clip load, untimed replay, S + G
            t1 = t0 + int(l * S)
            ev += [{"event": "window_start", "window": i, "t_ns": t0}, {"event": "window_end", "window": i, "t_ns": t1, "replays": 1}]
            t = t1 + g + spost
        out = runner.sizing_from_blocks({"A": {"events": ev}}, [3, 5])
        self.assertAlmostEqual(out["latency_s"]["A"][3], 0.4)
        self.assertAlmostEqual(out["latency_s"]["A"][5], 0.6)
        self.assertGreaterEqual(out["plan"]["clip_load_s"], 0.3)
        self.assertGreaterEqual(out["plan"]["warmup_s"]["A"], 20.0)
        s = runner.size_condition(out["latency_s"], out["plan"])
        self.assertEqual(s["N"], {3: math.ceil(12 / (0.95 * 0.4)), 5: math.ceil(12 / (0.95 * 0.6))})


if __name__ == "__main__":
    unittest.main()
