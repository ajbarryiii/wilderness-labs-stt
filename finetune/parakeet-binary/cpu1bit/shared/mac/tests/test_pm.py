"""Tests of the powermetrics record schema, stream framing, provisional timeline and window sums
(plans/cpu-benchmark-mac.md "Energy", A1; review c1 findings 1 and 8). The gating energy bounds are
guard.window_bounds (tests/test_guard.py).

p8 contract change (from M14): the calibration-burst anchor tests (pm.estimate_anchor, pm.apply_anchor,
pm.combine) and their burst helpers are retired; the mapping is spawn-based and has no absolute anchor.

    python3 tests/test_pm.py      (Mac: /usr/bin/python3; NixOS: finetune/parakeet-binary/python)
"""
from __future__ import annotations

import datetime
import math
import os
import plistlib
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pm  # noqa: E402

FIXTURE = os.environ.get("CPU1BIT_PM_FIXTURE",
                         "/mnt/hd/wilderness-labs-stt/parakeet-binary/cpu1bit/fixtures/powermetrics-3samples.plist0")
MS = 1_000_000
S = 1_000_000_000


WALL0_S = 1_792_000_000          # wall-clock seconds at clock 0 for synthetic timestamps


def wall(t_ns: int) -> datetime.datetime:
    """The plist `timestamp` (1 s resolution, UTC) of a sample ending at t_ns on the synthetic clock."""
    return datetime.datetime.fromtimestamp(math.floor(WALL0_S + t_ns / 1e9), datetime.timezone.utc).replace(tzinfo=None)


def rec(elapsed_ns, energy_mj, power=0.0, combined=0.0, idle=0.9, extra=None, drop=None, timestamp=None) -> bytes:
    p = {"cpu_energy": energy_mj, "cpu_power": power, "combined_power": combined,
         "clusters": [{"name": "E-Cluster", "idle_ratio": 0.5, "freq_hz": 2e9,
                       "cpus": [{"cpu": 0, "idle_ratio": idle}, {"cpu": 1, "idle_ratio": 0.7}]}]}
    if drop in p:
        p.pop(drop)
    d = {"elapsed_ns": elapsed_ns, "processor": p, "is_delta": True,
         "timestamp": timestamp or datetime.datetime(2026, 10, 21, 0, 0, 0)}
    if drop in d:
        d.pop(drop)
    if extra:
        d.update(extra)
    return plistlib.dumps(d)


def synthetic(power_fn, n: int, seed: int, latency_max_ns: int = 2 * MS, start_ns: int = 10**12,
              jitter_ns: int = 15 * MS, drop: set | None = None, malformed: set | None = None,
              delay_ns: int = 0, noise_mj: float = 0.0):
    """True contiguous sample intervals with jittered lengths; energy = integral of power_fn (W, at
    absolute ns) over each interval in mJ, plus Gaussian noise, rounded to integer mJ (the measured
    schema); arrival = true end + constant delay + latency in [0, max]. Returns (samples, truth)."""
    rng = random.Random(seed)
    t = start_ns
    samples, truth = [], []
    for i in range(n):
        el = 100 * MS + rng.randrange(0, jitter_ns)
        a, b = t, t + el
        e = sum(power_fn(a + k * MS) for k in range(el // MS)) + power_fn(a) * ((el % MS) / MS)
        e_rep = max(0, round(e + rng.gauss(0.0, noise_mj))) if noise_mj else round(e)
        arrival = b + delay_ns + rng.randrange(0, latency_max_ns + 1)
        truth.append((a, b, e))
        t = b
        if drop and i in drop:
            continue
        raw = b"<not a plist" if malformed and i in malformed else rec(el, int(e_rep), timestamp=wall(b))
        samples.append(pm.parse_record(raw, len(samples), arrival))
    return samples, truth


def true_energy_j(power_fn, t0: int, t1: int) -> float:
    return sum(power_fn(t) for t in range(t0, t1, MS)) / 1000.0


class ParseTests(unittest.TestCase):
    def test_fixture_units(self):
        if not os.path.exists(FIXTURE):
            self.skipTest("fixture not present on this host")
        s = pm.read_raw(FIXTURE)
        self.assertEqual(len(s), 3)
        for x in s:
            self.assertTrue(x.ok, x.error)
            self.assertAlmostEqual(x.cpu_power_mw, x.cpu_energy_mj / (x.elapsed_ns / 1e9), delta=0.01)
            self.assertEqual(sorted(x.clusters), ["E-Cluster", "P0-Cluster", "P1-Cluster"])

    def test_malformed_and_non_finite_records(self):
        """Review c1 finding 8: NaN / infinite / non-integer energy and other schema violations are malformed."""
        bad = {
            "not a plist": b"<plist",
            "zero elapsed": rec(0, 5),
            "negative energy": rec(100 * MS, -1),
            "bool energy": rec(100 * MS, True),
            "string energy": rec(100 * MS, "12"),
            "NaN energy": rec(100 * MS, float("nan")),
            "inf energy": rec(100 * MS, float("inf")),
            "non-integer energy": rec(100 * MS, 12.5),
            "float energy": rec(100 * MS, 12.0),
            "float elapsed": rec(100.0 * MS, 5),
            "NaN power": rec(100 * MS, 5, power=float("nan")),
            "NaN combined": rec(100 * MS, 5, combined=float("nan")),
            "NaN cpu idle": rec(100 * MS, 5, idle=float("nan")),
            "idle above 1": rec(100 * MS, 5, idle=1.5),
            "missing cpu_power": rec(100 * MS, 5, drop="cpu_power"),
            "missing clusters": rec(100 * MS, 5, drop="clusters"),
            "missing processor": plistlib.dumps({"elapsed_ns": 5}),
            "array root": plistlib.dumps([1, 2]),
        }
        for name, r in bad.items():
            self.assertFalse(pm.parse_record(r, 0, 0).ok, name)
        self.assertTrue(pm.parse_record(rec(100 * MS, 7), 0, 0).ok)

    def test_timestamp_is_required_and_kept(self):
        """M15.1: the enforced schema includes the 1 s `timestamp` (the producer-continuity band uses it)."""
        self.assertFalse(pm.parse_record(rec(100 * MS, 7, drop="timestamp"), 0, 0).ok)
        s = pm.parse_record(rec(100 * MS, 7, timestamp=datetime.datetime(2026, 10, 21, 7, 30, 38)), 0, 0)
        self.assertTrue(s.ok, s.error)
        self.assertEqual(s.timestamp_s, datetime.datetime(2026, 10, 21, 7, 30, 38, tzinfo=datetime.timezone.utc).timestamp())

    def test_stream_framing(self):
        """M15.1: every NUL-terminated chunk must parse; EOF before the runner's stop invalidates; only
        the one unterminated chunk cut by the runner's own stop is exempt."""
        good = b"".join(rec(100 * MS, 5 + i, timestamp=wall(i * 100 * MS)) + b"\0" for i in range(5))
        ok, reasons, recs = pm.check_stream(good + b"<?xml partial", stopped_by_runner=True)
        self.assertTrue(ok, reasons)
        self.assertEqual(len(recs), 5)
        bad = good.replace(b"<integer>7</integer>", b"<integer>7</inte", 1)
        self.assertFalse(pm.check_stream(bad, stopped_by_runner=True)[0])
        self.assertFalse(pm.check_stream(good, stopped_by_runner=False)[0])          # EOF: producer ended
        self.assertFalse(pm.check_stream(good + b"\0garbage\0", stopped_by_runner=True)[0])

    def test_nan_cannot_reach_a_valid_window(self):
        raws = [rec(100 * MS, 10) for _ in range(30)]
        raws[15] = rec(100 * MS, float("nan"))
        s = [pm.parse_record(r, i, 10**12 + (i + 1) * 100 * MS) for i, r in enumerate(raws)]
        pm.build_timeline(s)
        w = pm.window_energy(s, s[10].a_ns, s[20].b_ns, 0)
        self.assertFalse(w.valid)
        self.assertTrue(all(math.isfinite(v) for v in (w.e_lower_j, w.e_est_j, w.e_upper_j)))

    def test_split_records(self):
        recs, rest = pm.split_records(b"a\0b\0\0c")
        self.assertEqual((recs, rest), ([b"a", b"b"], b"c"))

    def test_joules(self):
        s, _ = synthetic(lambda t: 2.0, 30, 1, latency_max_ns=0, jitter_ns=1)
        pm.build_timeline(s)
        w = pm.window_energy(s, s[5].a_ns, s[25].b_ns, 0)
        self.assertTrue(w.valid)
        self.assertAlmostEqual(w.e_lower_j, sum(x.cpu_energy_mj - 1 for x in s[5:26]) / 1000.0)
        self.assertAlmostEqual(w.e_est_j, sum(x.cpu_energy_mj for x in s[5:26]) / 1000.0, places=9)


class TimelineTests(unittest.TestCase):
    def test_reconstruction_error_is_bounded_by_latency(self):
        for seed in range(20):
            s, truth = synthetic(lambda t: 1.0, 60, seed, latency_max_ns=3 * MS)
            pm.build_timeline(s)
            for x, (a, b, _) in zip(s, truth):
                self.assertGreaterEqual(x.b_ns - b, 0)
                self.assertLessEqual(x.b_ns - b, 3 * MS)
                self.assertEqual(x.b_ns - x.a_ns, b - a)

    def test_gap_splits_segments(self):
        s, truth = synthetic(lambda t: 1.0, 40, 3, drop={20})
        pm.build_timeline(s)
        self.assertEqual(len({x.segment for x in s}), 2)
        w = pm.window_energy(s, truth[15][0], truth[25][1], 3 * MS)
        self.assertFalse(w.valid)
        self.assertIn("gap", w.reason)
        self.assertTrue(pm.window_energy(s, truth[3][0], truth[15][1], 3 * MS).valid)

    def test_latency_spikes_do_not_split_or_shift(self):
        s, truth = synthetic(lambda t: 1.0, 80, 11, latency_max_ns=1 * MS)
        for i in (20, 21, 45):
            s[i].arrival_ns += 150 * MS + 35 * MS * (i % 3)
        pm.build_timeline(s)
        self.assertEqual(len({x.segment for x in s}), 1)
        for x, (a, b, _) in zip(s, truth):
            self.assertTrue(0 <= x.b_ns - b <= 1 * MS)

    def test_malformed_record_invalidates_window(self):
        s, truth = synthetic(lambda t: 1.0, 40, 4, malformed={18})
        pm.build_timeline(s)
        self.assertFalse(pm.window_energy(s, truth[10][0], truth[30][1], 2 * MS).valid)
        self.assertTrue(pm.window_energy(s, truth[21][0], truth[35][1], 2 * MS).valid)

    def test_uncovered_window_is_invalid(self):
        s, truth = synthetic(lambda t: 1.0, 20, 5)
        pm.build_timeline(s)
        w = pm.window_energy(s, truth[0][0] - 50 * MS, truth[10][1], 2 * MS)
        self.assertFalse(w.valid)
        self.assertIn("not covered", w.reason)


class BoundTests(unittest.TestCase):
    def check(self, power_fn, seed, eps, n=150):
        s, truth = synthetic(power_fn, n, seed, latency_max_ns=eps)
        pm.build_timeline(s)
        rng = random.Random(seed)
        checked = 0
        for _ in range(40):
            t0 = truth[5][0] + rng.randrange(0, 4000 * MS)
            t1 = t0 + rng.randrange(1000 * MS, 9000 * MS)
            if t1 + eps >= truth[-5][1]:
                continue
            w = pm.window_energy(s, t0, t1, eps)
            self.assertTrue(w.valid, w.reason)
            true = true_energy_j(power_fn, t0, t1)
            self.assertLessEqual(w.e_lower_j, true + 1e-9, (seed, t0, t1))
            self.assertGreaterEqual(w.e_upper_j, true - 1e-9, (seed, t0, t1))
            self.assertLessEqual(w.e_lower_j, w.e_est_j + 1e-12)
            self.assertLessEqual(w.e_est_j, w.e_upper_j + 1e-12)
            checked += 1
        self.assertGreater(checked, 20)

    def test_bounds_contain_truth_constant_power(self):
        for seed in range(5):
            self.check(lambda t: 3.0, seed, 2 * MS)

    def test_bounds_contain_truth_with_activity_at_boundaries(self):
        burst = lambda t: 20.0 if (t // MS) % 997 < 30 else 0.4  # noqa: E731
        for seed in range(5):
            self.check(burst, 100 + seed, 3 * MS)

    def test_concentrated_activity_at_window_edges(self):
        s, truth = synthetic(lambda t: 0.5, 100, 7, latency_max_ns=0, jitter_ns=1)
        pm.build_timeline(s)
        a, b, _ = truth[30]
        s[30].cpu_energy_mj = 5000.0
        w = pm.window_energy(s, (a + b) // 2, truth[60][1], 0)
        self.assertLess(w.e_lower_j, 5.0)
        self.assertGreater(w.e_upper_j, 5.0)
        self.assertGreater(w.e_est_j, 2.4)
        s2, truth2 = synthetic(lambda t: 0.5, 100, 8, latency_max_ns=0, jitter_ns=1)
        pm.build_timeline(s2)
        s2[60].cpu_energy_mj = 5000.0
        a, b, _ = truth2[60]
        w2 = pm.window_energy(s2, truth2[30][0], (a + b) // 2, 0)
        self.assertLess(w2.e_lower_j, 5.0)
        self.assertGreater(w2.e_upper_j, 5.0)

    def test_zero_alignment_error_keeps_full_boundary_samples(self):
        s, truth = synthetic(lambda t: 1.0, 50, 9, latency_max_ns=0, jitter_ns=1)
        pm.build_timeline(s)
        t0 = (truth[10][0] + truth[10][1]) // 2
        t1 = (truth[40][0] + truth[40][1]) // 2
        w = pm.window_energy(s, t0, t1, 0)
        self.assertEqual(w.samples_inside, 29)
        self.assertEqual(w.samples_touching, 31)
        self.assertAlmostEqual(w.e_upper_j, sum(x.cpu_energy_mj + 1 for x in s[10:41]) / 1000.0)


if __name__ == "__main__":
    unittest.main()
