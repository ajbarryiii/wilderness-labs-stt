"""Contract tests for M14 as amended by M15 (plans/cpu-benchmark-mac.md; written before the
implementation, pending plan review p6). Changes from the M14 version, and why:
- fixtures (M15.7, p5 minor 3): start offsets are non-negative; the p4-major-2 fixture idles at 1 W;
- the rate-error case stays within the validated rho = 50 ppm (M15.6);
- "lost record before the window is contained" is removed: under M15.1 a lost record is either a framing
  failure or a producer skip outside A1; it is replaced by the per-block band check, which the producer
  skip must fail;
- new: Codex p5's interior-loss repro (fails the band check), a capture ending before the bracketing sample,
  the L <= 130 ms gate, the block-age gate, and the per-window delivery-span corroboration flag.
Constants are read from the module, not hard-coded.

Contract (shared/mac/guard.py):
  guard.G_NS, guard.U_MAX_NS, guard.S_NS, guard.S_POST_NS, guard.A_MAX_NS, guard.L_MAX_NS, guard.TAU_BAND_NS
  guard.idle_reference(idle_samples) -> ref                     (M13.3: p_ref and tol)
  guard.window_bounds(samples, spawn_ns, t0, t1, ref, delivery_jitter_ns=None)   (M14.2-M14.5, M15)
      samples: parsed records in arrival order (pm.parse_record, pm.build_timeline segments)
      -> {"valid", "reasons", "flags", "u_ns", "x_count", "e_lower_j", "e_est_j", "e_upper_j", ...}
  guard.band_check(samples) -> {"valid", "width_ns", "reasons"}   (M15.1 per-block producer-continuity band)
  guard.prespawn_trial(samples, spawn_ns, ref) -> {"valid", "first_excess_mj", "reasons"}   (M16.1, A2)
  guard.rate_check(samples) -> {"valid", "max_excess_ns", "jitter_ns", "reasons"}             (M16.2, A3)
M16 additions (p6): the A2 negative-origin contract, the A3 cancelling-rate and late-block contracts, and the
undetectable-skip fixture documenting the band check's limit (M16.4).
Truth (M14.1): E_pad = integral of P over [t0 - G, t1 + G] on the benchmark clock.

    python3 tests/test_guard.py
"""
from __future__ import annotations

import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import guard  # noqa: E402
import pm  # noqa: E402
from test_pm import MS, S, synthetic  # noqa: E402

SPAWN = 10**12                                  # s: runner clock just before spawning powermetrics
IDLE = (SPAWN + 1 * S, SPAWN + 11 * S)          # pre-block idle window
WORK_W = 6.0


class Scenario:
    """powermetrics spawned at SPAWN, sampling from SPAWN + offset; idle window; untimed replay; settle S
    and guard G; the timed window W (12 s); guard G and settle S'; the next clip load. Power is
    background(t) + work(t), unless override(t) returns an absolute value."""

    def __init__(self, seed=1, offset_ms=300.0, delay_ms=0.0, latency_ms=2, noise_mj=1.0, bg_w=0.4,
                 work_w=WORK_W, tail_ms=0, tail_w=3.0, extra=None, override=None, probes=True, rate_ppm=0,
                 replay_overrun_ns=0, t0_shift_ns=0, prespawn_spin=False):
        g = guard.G_NS
        self.u0, self.u1 = SPAWN + 12 * S, SPAWN + 14 * S
        self.t0 = self.u1 + guard.S_NS + g + t0_shift_ns
        self.t1 = self.t0 + 12 * S
        self.load0 = self.t1 + g + guard.S_POST_NS
        self.replay_end = self.u1 + replay_overrun_ns
        self.bg_w, self.work_w, self.tail_ms, self.tail_w = bg_w, work_w, tail_ms, tail_w
        self.extra, self.override, self.probes, self.prespawn_spin = extra, override, probes, prespawn_spin
        self.phase = random.Random(seed).randrange(0, S)
        start = SPAWN + int(offset_ms * MS)
        n = int((self.load0 + 3 * S - start) / (107 * MS))
        self.samples, self.truth_iv = synthetic(self.power, n, seed, latency_max_ns=latency_ms * MS,
                                                delay_ns=int(delay_ms * MS), noise_mj=noise_mj, start_ns=start)
        if rate_ppm:
            for x in self.samples:
                x.elapsed_ns = int(x.elapsed_ns * (1 + rate_ppm * 1e-6))
        pm.build_timeline(self.samples)
        idle = [x for x, (a, b, _) in zip(self.samples, self.truth_iv) if x.ok and a >= IDLE[0] and b <= IDLE[1]]
        self._ref = guard.idle_reference(idle)       # before any test edits the sample list

    def power(self, t):
        if self.override is not None:
            v = self.override(self, t)
            if v is not None:
                return v
        v = self.bg_w
        if self.prespawn_spin and SPAWN - 600 * MS <= t < SPAWN:   # A2 trial: a spin ending at the spawn call
            v += 2.0
        if self.probes and ((t - self.phase) // MS) % 1000 < 40:   # 1 Hz state-probe bursts
            v += 0.25
        if self.u0 <= t < self.replay_end or self.load0 <= t < self.load0 + 500 * MS:
            v += 5.0                                                # untimed replay / next clip load
        if self.t0 <= t < self.t1:
            v += self.work_w
        elif self.tail_ms and self.t1 <= t < self.t1 + self.tail_ms * MS:
            v += self.tail_w
        return v + (self.extra(t) if self.extra else 0.0)

    def truth_j(self):
        g = guard.G_NS
        return sum(self.power(t) for t in range(self.t0 - g, self.t1 + g, MS)) / 1000.0

    def bounds(self, spawn_ns=SPAWN, **kw):
        return guard.window_bounds(self.samples, spawn_ns, self.t0, self.t1, self._ref, **kw)


class M14Contract(unittest.TestCase):
    def assert_contains(self, sc, case, **kw):
        w = sc.bounds(**kw)
        self.assertTrue(w["valid"], (case, w["reasons"]))
        truth = sc.truth_j()
        self.assertLessEqual(w["e_lower_j"], truth + 1e-9, (case, w["e_lower_j"], truth))
        self.assertGreaterEqual(w["e_upper_j"], truth - 1e-9, (case, w["e_upper_j"], truth))
        self.assertLessEqual(w["e_lower_j"], w["e_est_j"] + 1e-12, case)
        self.assertLessEqual(w["e_est_j"], w["e_upper_j"] + 1e-12, case)
        return w, truth

    def assert_invalid(self, sc, case, **kw):
        w = sc.bounds(**kw)
        self.assertFalse(w["valid"], (case, w.get("e_lower_j"), w.get("e_upper_j")))
        self.assertTrue(w["reasons"], case)
        return w

    # --- containment -----------------------------------------------------------------------------

    def test_clean_windows_over_start_offsets(self):
        u_ms = guard.U_MAX_NS / MS
        for frac in (0.0, 0.2, 0.5, 0.85):         # non-negative offsets (M15.7)
            for seed in (1, 2):
                w, truth = self.assert_contains(Scenario(seed=seed, offset_ms=frac * u_ms), ("offset", frac, seed))
                self.assertLessEqual(w["u_ns"], guard.U_MAX_NS)
                self.assertLess((w["e_upper_j"] - w["e_lower_j"]) / truth, 0.05)   # sanity, not M14.6's 1%

    def test_delivery_jitter_and_noise(self):
        self.assert_contains(Scenario(seed=3, offset_ms=100, latency_ms=40, noise_mj=3.0), "jitter")

    def test_p4_major1_low_power_start_sliver(self):
        """2 W background, total power 0.1 W in the first 95 ms of W, 8 W after, 2.1 W for 95 ms after t1."""
        def ov(sc, t):
            if sc.t0 <= t < sc.t0 + 95 * MS:
                return 0.1
            if sc.t0 + 95 * MS <= t < sc.t1:
                return 8.0
            if sc.t1 <= t < sc.t1 + 95 * MS:
                return 2.1
            return None
        self.assert_contains(Scenario(seed=4, bg_w=2.0, override=ov), "p4 major 1")

    def test_p4_major2_guard_background_above_tail_background(self):
        """Idle and guards at 1 W (M15.7); 8 W in W; a 0.5 s tail at 1.1 W total."""
        def ov(sc, t):
            if sc.t0 <= t < sc.t1:
                return 8.0
            if sc.t1 <= t < sc.t1 + 500 * MS:
                return 1.1
            return None
        self.assert_contains(Scenario(seed=5, bg_w=1.0, override=ov, probes=False), "p4 major 2")

    def test_p4_major4_subthreshold_start_with_tail(self):
        """The first 0.5 s of W at idle level, a detectable 0.5 s tail: valid (work present) and contained."""
        def ov(sc, t):
            if sc.t0 <= t < sc.t0 + 500 * MS:
                return sc.bg_w
            return None
        self.assert_contains(Scenario(seed=6, override=ov, tail_ms=500, tail_w=1.5), "p4 major 4")

    def test_tails_before_the_edges_are_part_of_the_measurand(self):
        for tail_ms in (80, 300):
            sc = Scenario(seed=7, tail_ms=tail_ms, tail_w=3.0)
            w, truth = self.assert_contains(sc, ("tail", tail_ms))
            self.assertGreater(truth, Scenario(seed=7).truth_j())

    def test_guard_spikes(self):
        """1 Hz probe spikes everywhere, plus a large spike inside each guard away from its edges."""
        sc0 = Scenario()
        big = [sc0.t0 - 200 * MS, sc0.t1 + 300 * MS]
        spike = lambda t: 0.6 if any(h <= t < h + 50 * MS for h in big) else 0.0  # noqa: E731
        for seed in range(3):
            self.assert_contains(Scenario(seed=seed, extra=spike), ("guard spikes", seed))

    def test_rate_error_within_rho(self):
        for ppm in (-40, 40):
            self.assert_contains(Scenario(seed=8, rate_ppm=ppm), ("rate", ppm))

    def test_corroboration_flag_does_not_gate(self):
        sc = Scenario(seed=12)
        w = sc.bounds(delivery_jitter_ns=10 * MS)
        self.assertTrue(w["valid"], w["reasons"])
        self.assertFalse([f for f in w["flags"] if "delivery" in f], w["flags"])
        sc2 = Scenario(seed=12)
        mid = next(i for i, (a, _, _) in enumerate(sc2.truth_iv) if a >= sc2.t0 + 6 * S)
        for x in sc2.samples[mid:]:
            x.arrival_ns += 40 * MS                 # delivery latency steps up by 40 ms mid-window
        pm.build_timeline(sc2.samples)
        w2, _ = self.assert_contains(sc2, "latency step", delivery_jitter_ns=10 * MS)
        self.assertTrue([f for f in w2["flags"] if "delivery" in f], w2["flags"])

    # --- producer continuity (M15.1) --------------------------------------------------------------

    def test_band_of_a_complete_capture(self):
        for seed in (1, 2, 3):
            b = guard.band_check(Scenario(seed=seed).samples)
            self.assertTrue(b["valid"], b["reasons"])
            self.assertLessEqual(b["width_ns"], 1 * S + guard.TAU_BAND_NS)

    def test_codex_p5_interior_loss_with_latency_drop(self):
        """p5 major 1: 100 ms startup offset, 0.4 W idle, one interior record lost while delivery latency
        drops from 200 to 100 ms. Per window it is invisible (d - c stays constant); it violates A1, and
        the per-block band check must invalidate the block."""
        sc = Scenario(seed=13, offset_ms=100, delay_ms=200, latency_ms=0, probes=False)
        cut = next(i for i, (a, _, _) in enumerate(sc.truth_iv) if a >= sc.t0 + 6 * S)
        lost_el = sc.samples[cut].elapsed_ns
        del sc.samples[cut]
        for x in sc.samples[cut:]:
            x.arrival_ns -= 100 * MS
        pm.build_timeline(sc.samples)
        self.assertLess(abs(lost_el - 100 * MS), 20 * MS)
        b = guard.band_check(sc.samples)
        self.assertFalse(b["valid"], b)
        self.assertGreater(b["width_ns"], 1 * S + guard.TAU_BAND_NS)

    def test_band_check_limit_undetectable_skip(self):
        """M16.4: with exactly 100 ms samples and whole-second timestamps the complete band is 0.9 s and one
        skipped record makes it 1.0 s, below 1 s + tau: the check cannot see it (documented limit)."""
        samples, truth = synthetic(lambda t: 0.4, 400, 1, latency_max_ns=0, jitter_ns=1, start_ns=SPAWN)
        pm.build_timeline(samples)
        full = guard.band_check(samples)
        self.assertAlmostEqual(full["width_ns"] / 1e9, 0.9, delta=0.002)
        del samples[200]
        pm.build_timeline(samples)
        skipped = guard.band_check(samples)
        self.assertAlmostEqual(skipped["width_ns"] / 1e9, 1.0, delta=0.002)
        self.assertTrue(skipped["valid"])          # undetected by design of the check

    # --- post-spawn origin (M16.1, A2) ------------------------------------------------------------

    def test_prespawn_trial_passes_with_a_fresh_baseline(self):
        for off in (0, 50, 300):
            sc = Scenario(seed=17, offset_ms=off, prespawn_spin=True)
            r = guard.prespawn_trial(sc.samples, SPAWN, sc._ref)
            self.assertTrue(r["valid"], (off, r))

    def test_negative_origin_is_caught_by_the_prespawn_trial_only(self):
        """p6 major 1: the first interval starts 100 ms before spawn, delivery latency 200 ms. U >= 0 and the
        window checks cannot see it; the pre-spawn spin trial must."""
        sc = Scenario(seed=18, offset_ms=-100, delay_ms=200, latency_ms=0, prespawn_spin=True)
        w = sc.bounds()
        self.assertGreaterEqual(w["u_ns"], 0)       # the per-block causality check passes
        r = guard.prespawn_trial(sc.samples, SPAWN, sc._ref)
        self.assertFalse(r["valid"], r)
        self.assertGreater(r["first_excess_mj"], 50)

    # --- constant rate (M16.2, A3) ----------------------------------------------------------------

    def long_capture(self, seed, rate_fn):
        """610 s of idle-level samples; rate_fn(c_ns) gives the relative rate error of elapsed_ns at
        cumulative time c, so elapsed drifts while delivery stays on the true clock."""
        samples, truth = synthetic(lambda t: 0.4, 5600, seed, latency_max_ns=5 * MS, start_ns=SPAWN + 50 * MS)
        c = 0
        for x, (a, b, _) in zip(samples, truth):
            c += b - a
            x.elapsed_ns = int(x.elapsed_ns * (1 + rate_fn(c)))
        pm.build_timeline(samples)
        return samples

    def test_constant_rate_within_rho_passes(self):
        for ppm in (0, 40, -40):
            r = guard.rate_check(self.long_capture(19, lambda c, ppm=ppm: ppm * 1e-6))
            self.assertTrue(r["valid"], (ppm, r))

    def test_cancelling_rate_error_fails(self):
        """p6 major 2: +100 ppm for 300 s, then -100 ppm: endpoint error ~0, 30 ms at 300 s."""
        r = guard.rate_check(self.long_capture(20, lambda c: 100e-6 if c < 300 * S else -100e-6))
        self.assertFalse(r["valid"], r)
        self.assertGreater(r["max_excess_ns"], 10 * MS)

    def test_late_block_rate_error_fails(self):
        r = guard.rate_check(self.long_capture(21, lambda c: 100e-6))
        self.assertFalse(r["valid"], r)

    # --- validity gates (M15.1, M15.6) ------------------------------------------------------------

    def test_capture_ending_before_the_bracketing_sample_is_invalid(self):
        sc = Scenario(seed=14)
        keep = [x for x, (_, b, _) in zip(sc.samples, sc.truth_iv) if b <= sc.t1 + 400 * MS]
        sc.samples = keep                          # p5 major 1: the capture stops at t1 + 0.4 s
        self.assert_invalid(sc, "capture ends early")

    def test_long_sample_is_invalid(self):
        sc = Scenario(seed=15)
        i = next(i for i, (a, _, _) in enumerate(sc.truth_iv) if a >= sc.t0 + 5 * S)
        a, b = sc.samples[i], sc.samples[i + 1]
        merged = pm.parse_record(__import__("test_pm").rec(a.elapsed_ns + b.elapsed_ns,
                                                            int(a.cpu_energy_mj + b.cpu_energy_mj),
                                                            timestamp=__import__("test_pm").wall(sc.truth_iv[i + 1][1])),
                                 a.index, b.arrival_ns)
        sc.samples[i:i + 2] = [merged]             # one contiguous ~215 ms sample
        pm.build_timeline(sc.samples)
        self.assertGreater(merged.elapsed_ns, guard.L_MAX_NS)
        self.assert_invalid(sc, "sample longer than L_MAX")

    def test_window_beyond_block_age_is_invalid(self):
        sc = Scenario(seed=16, t0_shift_ns=guard.A_MAX_NS, probes=False, noise_mj=0.0)
        self.assert_invalid(sc, "beyond A_MAX")

    # --- invalid ---------------------------------------------------------------------------------

    def test_alignment_beyond_u_max_is_invalid(self):
        self.assert_invalid(Scenario(offset_ms=guard.U_MAX_NS / MS + 100), "start offset beyond U_MAX")
        self.assert_invalid(Scenario(delay_ms=13_500), "p4 major 3: 13.5 s constant delivery delay")

    def test_causality_contradiction_is_invalid(self):
        sc = Scenario(offset_ms=100)
        self.assert_invalid(sc, "spawn stamped after sampling began", spawn_ns=SPAWN + 300 * MS)

    def test_edge_contamination_is_invalid(self):
        """Activity inside either edge of the padded interval, or outside it but within reach of the
        offset uncertainty (start offset 0 with a 300 ms delivery delay reaches earlier; a 300 ms start
        offset reaches later)."""
        g = guard.G_NS
        sc0 = Scenario()
        cases = {"inside the leading edge": ((sc0.t0 - g + 30 * MS, sc0.t0 - g + 90 * MS), {}),
                 "inside the trailing edge": ((sc0.t1 + g - 90 * MS, sc0.t1 + g - 30 * MS), {}),
                 "before, within the uncertainty": ((sc0.t0 - g - 200 * MS, sc0.t0 - g - 120 * MS),
                                                    dict(offset_ms=0, delay_ms=300)),
                 "after, within the uncertainty": ((sc0.t1 + g + 100 * MS, sc0.t1 + g + 180 * MS),
                                                   dict(offset_ms=300))}
        for name, ((lo, hi), kw) in cases.items():
            act = lambda t, lo=lo, hi=hi: 4.0 if lo <= t < hi else 0.0  # noqa: E731
            self.assert_invalid(Scenario(extra=act, **kw), ("edge", name))

    def test_tail_reaching_the_edge_is_invalid(self):
        self.assert_invalid(Scenario(tail_ms=int(guard.G_NS / MS), tail_w=1.0), "tail to t1 + G")

    def test_untimed_replay_overrunning_the_settle_wait_is_invalid(self):
        self.assert_invalid(Scenario(replay_overrun_ns=guard.S_NS + 20 * MS), "untimed replay overrun into the guard")

    def test_malformed_or_lost_record_inside_the_padded_region_is_invalid(self):
        sc = Scenario(seed=10)
        mid = next(i for i, (a, _, _) in enumerate(sc.truth_iv) if a >= sc.t0 + 5 * S)
        sc.samples[mid] = pm.parse_record(b"<not a plist", sc.samples[mid].index, sc.samples[mid].arrival_ns)
        pm.build_timeline(sc.samples)
        self.assert_invalid(sc, "malformed")
        sc2 = Scenario(seed=11)
        cut = next(i for i, (a, _, _) in enumerate(sc2.truth_iv) if a >= sc2.t0 + 3 * S)
        del sc2.samples[cut]
        pm.build_timeline(sc2.samples)
        self.assert_invalid(sc2, "lost")

    def test_c3_background_only_window_is_invalid(self):
        """p4 minor: the c3 fixture (0.4 W background only in W, 4.800 J over W) has no work."""
        sc = Scenario(work_w=0.0, probes=False)
        w = self.assert_invalid(sc, "no work")
        self.assertTrue(any("work" in r for r in w["reasons"]), w["reasons"])


# --- helper contracts (added with the implementation, written before the helpers) -------------------
# guard.window_bounds(..., delta_a2_ns=0) also returns "guard_j" (plan "Energy": the mean power of the samples
#   lying wholly inside a guard for every admissible offset, times 2G; None without such a sample), and
#   "p_in_count" / "p_out_count" (x_count = p_out_count - p_in_count).
# guard.samples_within(samples, spawn_ns, lo_ns, hi_ns, delta_a2_ns=0) -> [Sample]: the parsed samples whose
#   mapped interval lies inside [lo, hi] for every admissible offset; [] when the mapping fails or the parsed
#   run does not reach past hi.
# guard.energy_interval_check(samples, spawn_ns, spins, ref, delta_a2_ns=0) -> {"valid", "max_excess_mj",
#   "reasons", ...}: M15.1's energy-interval falsification test over known one-thread spins [(on, off)].
# guard.spin_excess_power_w(samples, spawn_ns, spins, ref, delta_a2_ns=0) -> float (W; NaN when undefined):
#   the median over spins of the excess power of the samples wholly inside the spin, for delta_A2 = tol / it
#   (M16.1; tol in mJ over W gives ms).
# Every guard function is total: bad data gives an invalid result with a reason ([] / NaN for the list and
#   float helpers), never an exception.
import math  # noqa: E402

SPIN_W = 2.0


class SpinCapture:
    """powermetrics spawned at SPAWN, first sample at SPAWN + offset, constant delivery delay (by default 0 and
    300 ms, so U is about 300 ms above the true offset: the widest uncertainty for that delay); 0.4 W idle,
    the idle window IDLE, then one-thread spins of known [on, off] at +SPIN_W."""

    def __init__(self, seed=41, spins=None, offset_ms=0, delay_ms=300):
        self.spins = spins if spins is not None else [
            (SPAWN + (13 + 4 * i) * S, SPAWN + (13 + 4 * i) * S + 1500 * MS) for i in range(6)]
        start = SPAWN + int(offset_ms * MS)
        n = int((max(off for _, off in self.spins) + 3 * S - start) / (107 * MS))
        self.samples, self.truth_iv = synthetic(self.power, n, seed, latency_max_ns=2 * MS,
                                                delay_ns=int(delay_ms * MS), noise_mj=1.0, start_ns=start)
        pm.build_timeline(self.samples)
        idle = [x for x, (a, b, _) in zip(self.samples, self.truth_iv) if a >= IDLE[0] and b <= IDLE[1]]
        self.ref = guard.idle_reference(idle)

    def power(self, t):
        return 0.4 + (SPIN_W if any(a <= t < b for a, b in self.spins) else 0.0)


class HelperContract(unittest.TestCase):
    def test_guard_energy_and_sample_counts(self):
        sc = Scenario(seed=34, probes=False)
        w = sc.bounds()
        self.assertTrue(w["valid"], w["reasons"])
        g = guard.G_NS
        true_guard = (sum(sc.power(t) for t in range(sc.t0 - g, sc.t0, MS))
                      + sum(sc.power(t) for t in range(sc.t1, sc.t1 + g, MS))) / 1000.0
        self.assertAlmostEqual(w["guard_j"], true_guard, delta=0.03 * true_guard)
        self.assertEqual(w["x_count"], w["p_out_count"] - w["p_in_count"])
        self.assertGreater(w["p_in_count"], 2 * guard.K)
        far = Scenario(seed=34, offset_ms=950, probes=False).bounds()  # U ~ G: no sample wholly in a guard
        self.assertFalse(far["valid"])
        self.assertIsNone(far["guard_j"])

    def test_admissible_offsets_extend_below_zero_by_delta_a2(self):
        """M16.1: o in [-delta_A2, U]. With delta_A2 covering the first interval's pre-spawn start (100 ms), the
        window is valid and its bounds contain the measurand; the edge-uncertain set can only grow."""
        sc = Scenario(seed=18, offset_ms=-100, delay_ms=200, latency_ms=0, prespawn_spin=True)
        w0 = sc.bounds()
        w = sc.bounds(delta_a2_ns=150 * MS)
        self.assertTrue(w["valid"], w["reasons"])
        truth = sc.truth_j()
        self.assertLessEqual(w["e_lower_j"], truth + 1e-9)
        self.assertGreaterEqual(w["e_upper_j"], truth - 1e-9)
        self.assertGreaterEqual(w["x_count"], w0["x_count"])

    def test_samples_within_is_inside_for_every_offset(self):
        for off, delay in ((300, 0), (0, 300), (100, 40)):
            sc = Scenario(seed=31, offset_ms=off, delay_ms=delay)
            truth = {id(x): (a, b) for x, (a, b, _) in zip(sc.samples, sc.truth_iv)}
            sel = guard.samples_within(sc.samples, SPAWN, IDLE[0], IDLE[1])
            ids = {id(x) for x in sel}
            for x in sel:
                a, b = truth[id(x)]
                self.assertTrue(x.ok and IDLE[0] <= a and b <= IDLE[1], (off, delay, a, b))
            core = [x for x in sc.samples
                    if IDLE[0] + 500 * MS <= truth[id(x)][0] and truth[id(x)][1] <= IDLE[1] - 500 * MS]
            self.assertTrue(core)
            self.assertTrue(all(id(x) in ids for x in core), (off, delay))   # nothing well inside is dropped
            narrower = guard.samples_within(sc.samples, SPAWN, IDLE[0], IDLE[1], delta_a2_ns=100 * MS)
            self.assertTrue({id(x) for x in narrower} <= ids)

    def test_samples_within_fails_closed(self):
        sc = Scenario(seed=32)
        self.assertEqual(guard.samples_within(sc.samples, SPAWN + 400 * MS, IDLE[0], IDLE[1]), [])   # U < 0
        self.assertEqual(guard.samples_within(sc.samples, SPAWN, IDLE[1], IDLE[0]), [])
        mid = next(i for i, (a, _, _) in enumerate(sc.truth_iv) if a >= IDLE[0] + 5 * S)
        sc.samples[mid] = pm.parse_record(b"<not a plist", sc.samples[mid].index, sc.samples[mid].arrival_ns)
        pm.build_timeline(sc.samples)
        self.assertEqual(guard.samples_within(sc.samples, SPAWN, IDLE[0], IDLE[1]), [])   # run breaks inside
        for bad in ([], None, [None], ["x"]):
            self.assertEqual(guard.samples_within(bad, SPAWN, IDLE[0], IDLE[1]), [])

    def test_energy_interval_check_passes_on_a_consistent_capture(self):
        cap = SpinCapture()
        self.assertTrue(cap.ref["valid"], cap.ref)
        r = guard.energy_interval_check(cap.samples, SPAWN, cap.spins, cap.ref)
        self.assertTrue(r["valid"], r["reasons"])
        self.assertLessEqual(r["max_excess_mj"], cap.ref["tol_mj"])

    def test_energy_reported_before_a_spin_fails(self):
        """Excess energy in a sample that ends, under every admissible offset, before a spin's start (energy and
        elapsed_ns covering different intervals) fails the check."""
        cap = SpinCapture()
        on = cap.spins[2][0]
        i = max(i for i, (_, b, _) in enumerate(cap.truth_iv) if b <= on - 400 * MS)
        cap.samples[i].cpu_energy_mj += 60
        r = guard.energy_interval_check(cap.samples, SPAWN, cap.spins, cap.ref)
        self.assertFalse(r["valid"], r)
        self.assertTrue(r["reasons"])
        self.assertGreater(r["max_excess_mj"], cap.ref["tol_mj"])

    def test_energy_interval_check_reach_is_limited_by_u(self):
        """Documented limit: a sample that truly ends before `on` but may end after it under some admissible
        offset (here U is about 300 ms) is outside the check."""
        cap = SpinCapture()
        on = cap.spins[2][0]
        i = max(i for i, (_, b, _) in enumerate(cap.truth_iv) if b <= on)
        cap.samples[i].cpu_energy_mj += 60
        r = guard.energy_interval_check(cap.samples, SPAWN, cap.spins, cap.ref)
        self.assertTrue(r["valid"], r["reasons"])

    def test_energy_interval_check_fails_closed(self):
        cap = SpinCapture()
        on, off = cap.spins[0]
        cases = {"no spins": (cap.samples, SPAWN, [], cap.ref),
                 "spins too close": (cap.samples, SPAWN, [(on, off), (off + 500 * MS, off + 1000 * MS)], cap.ref),
                 "spin after the capture": (cap.samples, SPAWN, [(off + 100 * S, off + 101 * S)], cap.ref),
                 "spin with off <= on": (cap.samples, SPAWN, [(off, on)], cap.ref),
                 "malformed spin": (cap.samples, SPAWN, [(on, "x")], cap.ref),
                 "invalid reference": (cap.samples, SPAWN, cap.spins, {"valid": False}),
                 "no reference": (cap.samples, SPAWN, cap.spins, None),
                 "causality (U < 0)": (cap.samples, SPAWN + 400 * MS, cap.spins, cap.ref),
                 "no samples": ([], SPAWN, cap.spins, cap.ref),
                 "garbage samples": ([None, "x"], SPAWN, cap.spins, cap.ref),
                 "samples None": (None, SPAWN, cap.spins, cap.ref)}
        for name, args in cases.items():
            r = guard.energy_interval_check(*args)
            self.assertFalse(r["valid"], name)
            self.assertTrue(r["reasons"], name)

    def test_spin_excess_power(self):
        cap = SpinCapture()
        p = guard.spin_excess_power_w(cap.samples, SPAWN, cap.spins, cap.ref)
        self.assertAlmostEqual(p, SPIN_W, delta=0.05 * SPIN_W)
        delta_a2_ms = cap.ref["tol_mj"] / p                       # mJ / W = ms
        self.assertTrue(0 < delta_a2_ms < 10, delta_a2_ms)
        last_off = cap.spins[-1][1]
        short = cap.spins + [(last_off + 2 * S, last_off + 2 * S + 150 * MS)]   # holds no sample for every offset
        self.assertAlmostEqual(guard.spin_excess_power_w(cap.samples, SPAWN, short, cap.ref), p, places=9)

    def test_spin_excess_power_is_nan_when_undefined(self):
        cap = SpinCapture()
        for args in (([], SPAWN, cap.spins, cap.ref), (None, SPAWN, cap.spins, cap.ref),
                     ([None], SPAWN, cap.spins, cap.ref), (cap.samples, SPAWN, [], cap.ref),
                     (cap.samples, SPAWN, cap.spins, None), (cap.samples, SPAWN, cap.spins, {"valid": False}),
                     (cap.samples, SPAWN + 400 * MS, cap.spins, cap.ref)):
            self.assertTrue(math.isnan(guard.spin_excess_power_w(*args)), args[1:3])


class TotalityContract(unittest.TestCase):
    """No guard function raises on bad data: each returns an invalid result with a reason."""

    BAD = ([], None, [None], ["x"], [object()], [pm.parse_record(b"<bad", 0, SPAWN + S)])

    def test_never_raises_on_bad_samples(self):
        sc = Scenario(seed=33)
        for bad in self.BAD:
            w = guard.window_bounds(bad, SPAWN, sc.t0, sc.t1, sc._ref)
            self.assertFalse(w["valid"], bad)
            self.assertTrue(w["reasons"], bad)
            for r in (guard.band_check(bad), guard.rate_check(bad), guard.prespawn_trial(bad, SPAWN, sc._ref),
                      guard.idle_reference(bad)):
                self.assertFalse(r["valid"], (bad, r))
                self.assertTrue(r["reasons"], (bad, r))

    def test_never_raises_on_bad_reference_or_times(self):
        sc = Scenario(seed=33)
        for ref in (None, {}, {"valid": False}, {"valid": True, "p_ref_w": float("nan"), "tol_mj": 5.0}, "x"):
            w = guard.window_bounds(sc.samples, SPAWN, sc.t0, sc.t1, ref)
            self.assertFalse(w["valid"], ref)
            self.assertTrue(w["reasons"], ref)
            self.assertFalse(guard.prespawn_trial(sc.samples, SPAWN, ref)["valid"], ref)
        for spawn, t0, t1 in ((SPAWN, sc.t1, sc.t0), (None, sc.t0, sc.t1), (SPAWN, "x", sc.t1)):
            w = guard.window_bounds(sc.samples, spawn, t0, t1, sc._ref)
            self.assertFalse(w["valid"])
            self.assertTrue(w["reasons"])
        self.assertFalse(guard.window_bounds(sc.samples, SPAWN, sc.t0, sc.t1, sc._ref, delta_a2_ns=-1)["valid"])

    def test_corrupted_fields_are_malformed_not_a_crash(self):
        sc = Scenario(seed=33)
        i = next(i for i, (a, _, _) in enumerate(sc.truth_iv) if a >= sc.t0 + 4 * S)
        sc.samples[i].cpu_energy_mj = float("nan")
        w = guard.window_bounds(sc.samples, SPAWN, sc.t0, sc.t1, sc._ref)
        self.assertFalse(w["valid"])
        self.assertTrue(all(v is None or math.isfinite(v) for v in (w["e_lower_j"], w["e_est_j"], w["e_upper_j"])))
        self.assertFalse(guard.band_check(sc.samples)["valid"])
        self.assertFalse(guard.rate_check(sc.samples)["valid"])


if __name__ == "__main__":
    unittest.main()
