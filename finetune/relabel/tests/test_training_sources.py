"""Contract tests: the multi-source version 2 batch plan (b0prime plan v4 "Extra sources", "Code and review") and
its replay, on the batch-plan fixture world (real C1 shards, a complete journal of 4 pools) with two clip stores:
an augmented source at a fixed share and a source drawn exactly once.

- every real occurrence once; each source's per-pool share within one clip, the running total within half a clip;
- the source drawn once: every clip exactly once over the complete journal, never repeated in a prefix;
- the repeating source repeats a clip only after its whole pool is used;
- the budget held at the augmented keys; deterministic, and a prefix of the longer plan;
- the replay serves each source's rows with its prefix, tag, label and transform; stage 2c's single-source plan is
  still served and verified as before (its real production plan too, when present)."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from synthetic import mixing
from test_training_batchplan import BUDGET, N_POOLS, P, SEED, World
from training import batchplan as bp
from training import label_sources
from training import replay as treplay


def store(root: Path, n: int, identity: str) -> mixing.ClipPool:
    from test_synthetic_mixing import make_pool
    return mixing.ClipPool(make_pool(root, n=n).clips, identity=identity)


class Sources(World):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.syn = store(cls.root / "src-syn", 12, "syn-store")
        cls.once = store(cls.root / "src-once", 30, "once-store")
        cls.sources = [bp.Source("synthetic", "syn:", cls.syn, 11, 0.2, True),
                       bp.Source("read-speech", "rs:", cls.once, 13, bp.ONCE, False)]
        cls.kw = dict(pool_batches=P, budget=BUDGET, seed=SEED)
        cls.plan3 = bp.build(cls.w.pool, cls.root / "plans" / "multi", N_POOLS, sources=cls.sources, **cls.kw)

    def draws(self, plan, s: int) -> list[list[int]]:
        """Per pool, source s's clip indices in entry (draw) order."""
        out = []
        for a, b in zip(plan.pools[:-1].tolist(), plan.pools[1:].tolist()):
            out.append([bp.extra_parts(e)[1] for e in plan.entries[a:b].tolist()
                        if e >= bp.SYN_FLAG and bp.extra_parts(e)[0] == s])
        return out

    def real_seconds(self) -> list[float]:
        return [float(self.plan.samples[a:b].sum()) / bp.SAMPLE_RATE
                for a, b in zip(self.plan.pools[:-1].tolist(), self.plan.pools[1:].tolist())]

    def test_entry_encoding(self):
        self.assertEqual(bp.extra_entry(0, 5), bp.SYN_FLAG + 5)                # source 0: stage 2c's entries
        for s, c in ((0, 0), (1, 941_999), (3, (1 << bp.SOURCE_BITS) - 1)):
            self.assertEqual(bp.extra_parts(bp.extra_entry(s, c)), (s, c))
        with self.assertRaises(bp.PlanError):
            bp.extra_entry(0, 1 << bp.SOURCE_BITS)

    def test_every_real_occurrence_appears_exactly_once(self):
        ents = self.plan3.entries
        self.assertTrue(np.array_equal(ents[ents < bp.SYN_FLAG], np.arange(len(self.seq))))
        served = sorted(e for t in range(self.plan3.n_batches) for e in self.plan3.batch(t).tolist()
                        if e < bp.SYN_FLAG)
        self.assertEqual(served, list(range(len(self.seq))))
        self.assertEqual((self.plan3.n_pools, self.plan3.consumed_batches), (self.plan.n_pools, self.plan.consumed_batches))
        self.assertEqual(self.plan3.meta["base_plan_sha256"], self.plan.sha256)

    def test_each_sources_share_per_pool_and_running_total(self):
        real = self.real_seconds()
        for s, src in enumerate(self.sources):
            secs = [c["duration"] for c in sorted(src.clips.clips, key=lambda c: c["id"])]
            biggest = max(secs)
            share = self.plan3.meta["sources"][s]["share_value"]
            got = [sum(secs[c] for c in pool) for pool in self.draws(self.plan3, s)]
            pools = range(len(real) - 1) if src.share == bp.ONCE else range(len(real))   # the last takes the rest
            for p in pools:
                self.assertLessEqual(abs(got[p] - share * real[p]), biggest + 1e-9, (src.name, p))
                self.assertLessEqual(abs(sum(got[:p + 1]) - share * sum(real[:p + 1])), biggest / 2 + 1e-9,
                                     (src.name, p))
            self.assertGreater(sum(got), 0)
        self.assertEqual(self.plan3.meta["sources"][0]["share_value"], 0.2)

    def test_the_once_source_is_drawn_exactly_once(self):
        drawn = [c for pool in self.draws(self.plan3, 1) for c in pool]
        self.assertEqual(sorted(drawn), list(range(len(self.once.clips))))
        total = sum(c["duration"] for c in self.once.clips)
        self.assertAlmostEqual(self.plan3.meta["sources"][1]["share_value"],
                               total * bp.SAMPLE_RATE / float(self.plan.samples.sum()))
        self.assertTrue(self.plan3.meta["journal_complete"])
        short = bp.build(self.w.pool, self.root / "plans" / "multi-two", 2, sources=self.sources, **self.kw)
        part = [c for pool in self.draws(short, 1) for c in pool]
        self.assertEqual(len(part), len(set(part)))                        # never repeated, nothing taken early
        self.assertLess(len(part), len(self.once.clips))
        self.assertEqual(short.meta["sources"][1]["share_value"], self.plan3.meta["sources"][1]["share_value"])

    def test_the_repeating_source_repeats_only_after_its_whole_pool(self):
        drawn = [c for pool in self.draws(self.plan3, 0) for c in pool]
        n = len(self.syn.clips)
        self.assertGreater(len(drawn), n)                                  # drawn more than once over the run
        for k in range(0, len(drawn) - n + 1, n):
            self.assertEqual(sorted(drawn[k:k + n]), list(range(n)))
        tail = drawn[len(drawn) // n * n:]
        self.assertEqual(len(tail), len(set(tail)))

    def test_the_budget_holds_at_the_augmented_length(self):
        secs = {s: bp.clip_samples(src.clips) for s, src in enumerate(self.sources)}
        for e, k in zip(self.plan3.entries.tolist(), self.plan3.samples.tolist()):
            if e >= bp.SYN_FLAG:
                s, c = bp.extra_parts(e)
                self.assertEqual(k, bp.synthetic_key(secs[s][c]) if s == 0 else secs[s][c])
        for b in range(self.plan3.n_batches):
            rows = self.plan3.order[self.plan3.offsets[b]:self.plan3.offsets[b + 1]]
            self.assertLessEqual(len(rows) * int(self.plan3.samples[rows].max()), BUDGET)

    def test_deterministic_prefix_and_refusals(self):
        a = bp.compute_sources(self.w.pool, N_POOLS, self.sources, **self.kw)
        self.assertEqual(a, bp.compute_sources(self.w.pool, N_POOLS, self.sources, **self.kw))
        other = [self.sources[0], bp.Source("read-speech", "rs:", self.once, 14, bp.ONCE, False)]
        self.assertNotEqual(bp.compute_sources(self.w.pool, N_POOLS, other, **self.kw)[1]["entries.u32"],
                            a[1]["entries.u32"])
        short = bp.build(self.w.pool, self.root / "plans" / "multi-prefix", 2, sources=self.sources, **self.kw)
        self.assertTrue(bp.is_prefix(short, self.plan3))
        self.assertFalse(bp.is_prefix(short, self.plan))
        self.assertEqual(bp.Plan.load(self.plan3.dir).sha256, self.plan3.sha256)
        self.assertEqual(self.plan3.config_id(), self.plan.config_id())
        self.plan3.verify_journal(self.w.pool, sources=self.sources)
        with self.assertRaises(bp.PlanError):
            self.plan3.verify_journal(self.w.pool, sources=other)
        with self.assertRaises(bp.PlanError):
            self.plan3.verify_journal(self.w.pool)
        with self.assertRaises(bp.PlanError):
            self.plan.verify_journal(self.w.pool, sources=self.sources)
        with self.assertRaises(bp.PlanError):
            bp.compute_sources(self.w.pool, N_POOLS, [self.sources[0], self.sources[0]], **self.kw)
        with self.assertRaises(ValueError):                               # a once source of an incomplete journal
            mixing.plan([1.0], [0.5], 1, 0.1, once=False, take_rest=True)

    def test_the_replay_serves_every_sources_rows(self):
        from synthetic import augment
        from test_synthetic_mixing import make_aug_pool
        aug = augment.Augmenter(make_aug_pool(self.root / "src-aug"), seed=3)
        extras = [treplay.ExtraSource(self.sources[0], aug), treplay.ExtraSource(self.sources[1], None)]
        r = treplay.PlannedReplay(self.plan3, self.w.pool, store=self.store, sources=extras)
        labels = label_sources.Labels(label_sources.StoreLabels(self.labels()),
                                      {"syn:": label_sources.ClipLabels(self.syn, "syn:"),
                                       "rs:": label_sources.ClipLabels(self.once, "rs:")})
        by = {"syn:": {c["id"]: c for c in self.syn.clips}, "rs:": {c["id"]: c for c in self.once.clips}}
        seen = {"synthetic": 0, "read-speech": 0}
        bt = label_sources.Batcher(r, labels, start=0, stop=r.through_batch)
        try:
            for t in range(r.through_batch):
                got = bt.next(timeout=60)
                items = r.read(t)
                for row, (i, src) in enumerate(zip(got["ids"], got["sources"])):
                    if src in seen:
                        prefix = "syn:" if src == "synthetic" else "rs:"
                        self.assertTrue(i.startswith(prefix))
                        clip = by[prefix][i[len(prefix):]]
                        self.assertEqual(got["labels"][row], clip["label"])
                        if src == "read-speech":                           # no transform: the stored audio
                            self.assertTrue(np.array_equal(items[row]["audio"], self.once.read(clip["id"])))
                        seen[src] += 1
                    else:
                        self.assertEqual(got["labels"][row], self.labels().text(i))
        finally:
            bt.close()
        self.assertEqual(seen["synthetic"] + seen["read-speech"], self.plan3.meta["n_extra"])
        self.assertTrue(all(seen.values()))
        acc = treplay.plan(r, labels, r.through_batch)
        self.assertEqual(acc["synthetic_rows"], self.plan3.meta["n_extra"])
        self.assertEqual(r.identity["sources"][0]["name"], "synthetic")
        with self.assertRaises(ValueError):
            treplay.PlannedReplay(self.plan3, self.w.pool, store=self.store)
        with self.assertRaises(ValueError):
            treplay.PlannedReplay(self.plan, self.w.pool, store=self.store, sources=extras)


REAL = Path(os.environ.get("RELABEL_ROOT", "/nonexistent"))


@unittest.skipUnless((REAL / "training" / "batch-plan").exists() and (REAL / "synthetic" / "runs" / "full").exists(),
                     "needs the real stage 2c plan and clip pool")
class RealStage2cPlan(unittest.TestCase):
    """The generalized code recomputes stage 2c's production version 2 plan byte for byte."""

    def test_the_production_synthetic_plan_verifies(self):
        from labeling import config as lconfig
        from labeling.__main__ import c1_dir, pinned
        clips = mixing.ClipPool.load("full")
        d = bp.production_dir(lconfig.config_id(pinned()), 156, clips=clips)
        if not (d / "plan.json").exists():
            self.skipTest("no stage 2c version 2 plan")
        plan = bp.Plan.load(d)
        j = plan.verify_journal(c1_dir(), clips=clips)
        self.assertEqual(len(j["ids"]), int(plan.meta["n_occurrences"]))


if __name__ == "__main__":
    unittest.main()
