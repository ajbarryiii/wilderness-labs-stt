"""Contract tests: the WSD schedule on the consumed-batch index (b0prime plan "Schedule (WSD)",
decoder-only plan "Schedule")."""
from __future__ import annotations

import unittest

from training import schedule


class Trunk(unittest.TestCase):
    def test_warmup_is_two_percent_of_the_planned_trunk(self):
        self.assertEqual(schedule.trunk(1e-4, 85_000).warmup, 1_700)
        self.assertEqual(schedule.trunk(1e-4, 10_000).warmup, 200)
        self.assertEqual(schedule.trunk(1e-4, 20_000).warmup, 400)

    def test_warmup_is_linear_and_reaches_the_peak_at_batch_w(self):
        s = schedule.trunk(2e-4, 10_000)     # w = 200
        self.assertEqual(s.lr(0), 0.0)
        self.assertAlmostEqual(s.lr(100), 1e-4)
        self.assertAlmostEqual(s.lr(199), 2e-4 * 199 / 200)
        self.assertLess(s.lr(199), 2e-4)
        self.assertEqual(s.lr(200), 2e-4)

    def test_trunk_holds_the_peak_without_an_end(self):
        s = schedule.trunk(3e-5, 85_000)
        for k in (1_700, 50_000, 84_999, 85_000, 200_000):
            self.assertEqual(s.lr(k), 3e-5)
        self.assertIsNone(s.last_batch)

    def test_negative_index_is_refused(self):
        with self.assertRaises(ValueError):
            schedule.trunk(1e-4, 100).lr(-1)


class UniformPlan:
    """The pool starts of a plan with `per` planned batches in each of `n_pools` pools."""

    def __init__(self, n_pools, per):
        self.n_pools, self.per = n_pools, per

    def pool_start(self, p):
        return self.per * p


class Branch(unittest.TestCase):
    def test_decoder_only_boundaries(self):
        # Contract change (2026-10-24, decoder-only plan "Schedule"): phases are pools of the batch plan on the
        # planned-batch index: warmup 2% of the planned batches of pools 0-155, stable through pool 132, cooldown
        # over pools 133-155 (previously 400 / 17,000-19,999 consumed batches). A plan of 106 planned batches per
        # pool, 16,536 in all; pool 133 starts at planned batch 14,098.
        s = schedule.decoder_only(1e-4, UniformPlan(156, 106))
        self.assertEqual((s.warmup, s.cooldown_start, s.cooldown, s.last_batch), (331, 14_098, 2_438, 16_535))
        self.assertEqual(s.lr(0), 0.0)
        self.assertLess(s.lr(330), 1e-4)
        self.assertEqual(s.lr(331), 1e-4)
        self.assertEqual(s.lr(14_097), 1e-4)       # stable through pool 132
        self.assertEqual(s.lr(14_098), 1e-4)       # cooldown starts from the peak at pool 133's first batch
        self.assertLess(s.lr(14_099), 1e-4)
        self.assertEqual(s.lr(16_535), 0.0)        # 0 at the last planned batch
        with self.assertRaises(ValueError):
            s.lr(16_536)                           # past the last batch: the run has stopped
        self.assertEqual(s.stop_batch, 16_536)
        with self.assertRaises(ValueError):
            schedule.decoder_only(1e-4, UniformPlan(155, 106))   # another span

    def test_cooldown_is_linear(self):
        s = schedule.branch(schedule.trunk(1.0, 1_000), start=500, cooldown=11)
        values = [s.lr(k) for k in range(500, 511)]
        self.assertEqual(values[0], 1.0)
        self.assertEqual(values[-1], 0.0)
        steps = [a - b for a, b in zip(values, values[1:])]
        for d in steps:
            self.assertAlmostEqual(d, 0.1)

    def test_b0prime_branches(self):
        trunk = schedule.trunk(1e-4, 85_000)
        for total, start, cooldown in ((25_000, 21_250, 3_750), (50_000, 42_500, 7_500), (100_000, 85_000, 15_000)):
            b = schedule.fraction_branch(trunk, total)
            self.assertEqual((b.cooldown_start, b.cooldown, b.last_batch + 1), (start, cooldown, total))
            self.assertEqual(b.lr(start), trunk.lr(start))     # from the trunk's peak
            self.assertEqual(b.lr(total - 1), 0.0)             # to 0 at its last batch
            for k in (0, 1_000, trunk.warmup, start - 1):       # identical to the trunk before the branch point
                self.assertEqual(b.lr(k), trunk.lr(k))
        pilot = schedule.branch(schedule.trunk(3e-4, 10_000), start=8_500, cooldown=1_500)
        self.assertEqual((pilot.warmup, pilot.lr(8_499), pilot.lr(9_999)), (200, 3e-4, 0.0))

    def test_branch_does_not_change_the_trunk(self):
        trunk = schedule.trunk(1e-4, 85_000)
        before = trunk.to_dict()
        schedule.branch(trunk, start=42_500, cooldown=7_500)
        self.assertEqual(trunk.to_dict(), before)
        self.assertIsNone(trunk.cooldown_start)

    def test_invalid_branches_are_refused(self):
        trunk = schedule.trunk(1e-4, 10_000)
        with self.assertRaises(ValueError):
            schedule.branch(trunk, start=100, cooldown=10)          # inside the warmup
        with self.assertRaises(ValueError):
            schedule.branch(trunk, start=5_000, cooldown=1)         # no linear decay over one batch
        with self.assertRaises(ValueError):
            schedule.branch(schedule.branch(trunk, 5_000, 100), 6_000, 100)   # a branch of a branch

    def test_round_trip(self):
        s = schedule.decoder_only(3e-5, UniformPlan(156, 106))
        self.assertEqual(schedule.WSD.from_dict(s.to_dict()), s)


if __name__ == "__main__":
    unittest.main()
