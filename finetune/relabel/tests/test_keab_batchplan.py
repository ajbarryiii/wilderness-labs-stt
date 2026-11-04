"""Contract tests: batch plan version 3 (keab-prime plan "Batch plan version 3", "Code and review": "as stage 3's
version 2 tests, plus the AMI share and pairs") and its replay, on the batch-plan fixture world (real C1 shards, a
complete journal of 4 pools) with three sources: a synthetic union of two clip stores drawn about K times (TS keys),
a source drawn exactly once, and an AMI pair store drawn exactly K times (keys at the SDM samples).

- every real occurrence once; each source's per-pool share within one clip and the running total within half a clip;
- "times:K": every clip exactly K times over the complete journal, a clip repeating only after the whole source in
  each pass, never more than K in a prefix plan; "about:K": share = K x seconds / the journal's real seconds, the
  repeating rule; ONCE unchanged;
- the budget held at the keys (augmented for the synthetic union, the SDM samples for the pairs);
- deterministic, a prefix of the longer plan, verified against the journal and its sources, refusals (version 2
  takes no times/about share, a version 3 plan needs its sources, the plan id carries -v3x);
- the synthetic union: one source of several runs, ids distinct, its identity changes with its parts;
- the replay serves a pair row with the SDM audio for the student, the IHM audio for the teacher and no stored
  label; its text is the arm's label of its IHM id (rejected with it); plan() counts steps with the pair rows' labels;
- version 2 plans (stage 3's) unchanged: their plan.json bytes and ids."""
from __future__ import annotations

import types
import unittest
from pathlib import Path

import numpy as np

from synthetic import mixing
from test_ami_sdm import make_clips, pcm_of
from test_training_batchplan import BUDGET, N_POOLS, P, SEED, World
from training import ami_store
from training import batchplan as bp
from training import label_sources
from training import replay as treplay
from training.clip_union import ClipUnion

K_AMI, K_SYN = 3, 2


def store(root: Path, n: int, identity: str, prefix: str) -> mixing.ClipPool:
    from test_synthetic_mixing import make_pool
    clips = make_pool(root, n=n).clips
    for c in clips:
        c["id"] = c["id"].replace("pilot", prefix)
    return mixing.ClipPool(clips, identity=identity)


class V3(World):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.full = store(cls.root / "src-full", 8, "full-store", "full")
        cls.pairs = store(cls.root / "src-pairs", 5, "pairs-store", "pairs-r1")
        cls.union = ClipUnion([("full", cls.full), ("pairs", cls.pairs)])
        cls.once = store(cls.root / "src-once", 25, "once-store", "cv")
        lab = cls.labels(cls)                                   # World.labels reads only class attributes
        kept = sorted(i for i in lab._texts if lab.text(i) is not None)
        ami = [i for i in kept if i.startswith("ami:")]
        cls.kept_ami = (ami + [i for i in kept if i not in ami])[:4]          # the pairs' IHM ids (in memory, any id)
        cls.rejected_ami = sorted(i for i in lab._texts if lab.text(i) is None)
        assert len(cls.kept_ami) >= 4 and cls.rejected_ami, "the fixture has kept and rejected ids"
        chosen = cls.kept_ami[:4] + cls.rejected_ami[:1]
        specs = [(f"AMI_ES2002a_sdm_FEE005_{k:07d}_{k + 100:07d}", hid, pcm_of(k, 8000 + 997 * k),
                  pcm_of(50 + k, 8000 + 997 * k)) for k, hid in enumerate(chosen)]
        cls.pair_specs = specs
        cls.ami = ami_store.PairStore(make_clips(cls.root / "pair-audio", specs), identity="pair-store")
        cls.sources = [bp.Source("synthetic", "syn:", cls.union, 11, f"about:{K_SYN}", True),
                       bp.Source("cv27-en", "cv27-en:", cls.once, 13, bp.ONCE, False),
                       bp.Source("ami-sdm", "ami-sdm:", cls.ami, 17, f"times:{K_AMI}", False)]
        cls.kw = dict(pool_batches=P, budget=BUDGET, seed=SEED, version=bp.V3)
        cls.plan3 = bp.build(cls.w.pool, cls.root / "plans" / "v3", N_POOLS, sources=cls.sources, **cls.kw)

    def draws(self, plan, s: int) -> list[list[int]]:
        out = []
        for a, b in zip(plan.pools[:-1].tolist(), plan.pools[1:].tolist()):
            out.append([bp.extra_parts(e)[1] for e in plan.entries[a:b].tolist()
                        if e >= bp.SYN_FLAG and bp.extra_parts(e)[0] == s])
        return out

    def real_seconds(self) -> list[float]:
        return [float(self.plan.samples[a:b].sum()) / bp.SAMPLE_RATE
                for a, b in zip(self.plan.pools[:-1].tolist(), self.plan.pools[1:].tolist())]

    def secs(self, s: int) -> list[float]:
        return [c["duration"] for c in sorted(self.sources[s].clips.clips, key=lambda c: c["id"])]

    def test_version_identity_and_every_real_occurrence_once(self):
        p = self.plan3
        self.assertEqual(p.version, bp.V3)
        self.assertIn(f"-v3x{bp.sources_id(self.sources)}", p.meta["plan_id"])
        self.assertEqual(p.config_id(), self.plan.config_id())
        self.assertEqual(p.meta["base_plan_sha256"], self.plan.sha256)
        ents = p.entries
        self.assertTrue(np.array_equal(ents[ents < bp.SYN_FLAG], np.arange(len(self.seq))))
        served = sorted(e for t in range(p.n_batches) for e in p.batch(t).tolist() if e < bp.SYN_FLAG)
        self.assertEqual(served, list(range(len(self.seq))))
        self.assertEqual(bp.Plan.load(p.dir).sha256, p.sha256)
        self.assertEqual([m["kind"] for m in p.meta["sources"]], ["about", "once", "times"])

    def test_the_ami_pairs_are_drawn_exactly_k_times(self):
        drawn = [c for pool in self.draws(self.plan3, 2) for c in pool]
        n = len(self.ami.clips)
        self.assertEqual(sorted(drawn), sorted(list(range(n)) * K_AMI))
        for k in range(0, K_AMI * n, n):                         # each pass is the whole source, then the next
            self.assertEqual(sorted(drawn[k:k + n]), list(range(n)))
        m = self.plan3.meta["sources"][2]
        self.assertEqual(m["draws_per_clip"], {"min": K_AMI, "max": K_AMI})
        self.assertAlmostEqual(m["share_value"], K_AMI * sum(self.secs(2)) * bp.SAMPLE_RATE / float(self.plan.samples.sum()))
        short = bp.build(self.w.pool, self.root / "plans" / "v3-two", 2, sources=self.sources, **self.kw)
        part = [c for pool in self.draws(short, 2) for c in pool]
        self.assertLess(len(part), K_AMI * n)
        self.assertTrue(max(np.bincount(part, minlength=n)) <= K_AMI)
        self.assertEqual(part, drawn[:len(part)])                # the same passes, nothing taken early
        self.assertEqual(short.meta["sources"][2]["share_value"], m["share_value"])

    def test_the_synthetic_union_is_drawn_about_k_times(self):
        drawn = [c for pool in self.draws(self.plan3, 0) for c in pool]
        n = len(self.union.clips)
        m = self.plan3.meta["sources"][0]
        self.assertAlmostEqual(m["share_value"], K_SYN * sum(self.secs(0)) * bp.SAMPLE_RATE / float(self.plan.samples.sum()))
        self.assertLessEqual(abs(sum(self.secs(0)[c] for c in drawn) - K_SYN * sum(self.secs(0))), max(self.secs(0)) / 2 + 1e-9)
        for k in range(0, len(drawn) - n + 1, n):                # repeats only after the whole union
            self.assertEqual(sorted(drawn[k:k + n]), list(range(n)))
        self.assertIn(m["draws_per_clip"]["min"], (K_SYN - 1, K_SYN))
        self.assertIn(m["draws_per_clip"]["max"], (K_SYN, K_SYN + 1))
        once = [c for pool in self.draws(self.plan3, 1) for c in pool]
        self.assertEqual(sorted(once), list(range(len(self.once.clips))))     # Common Voice: exactly once

    def test_each_sources_share_per_pool_and_running_total(self):
        real = self.real_seconds()
        for s, src in enumerate(self.sources):
            secs = self.secs(s)
            share = self.plan3.meta["sources"][s]["share_value"]
            got = [sum(secs[c] for c in pool) for pool in self.draws(self.plan3, s)]
            fixed = src.share == bp.ONCE or str(src.share).startswith("times")
            for p in range(len(real) - 1 if fixed else len(real)):            # a fixed draw's last pool takes the rest
                self.assertLessEqual(abs(got[p] - share * real[p]), max(secs) + 1e-9, (src.name, p))
                self.assertLessEqual(abs(sum(got[:p + 1]) - share * sum(real[:p + 1])), max(secs) / 2 + 1e-9,
                                     (src.name, p))

    def test_keys_and_the_budget(self):
        samples = {s: bp.clip_samples(src.clips) for s, src in enumerate(self.sources)}
        for e, k in zip(self.plan3.entries.tolist(), self.plan3.samples.tolist()):
            if e >= bp.SYN_FLAG:
                s, c = bp.extra_parts(e)
                self.assertEqual(k, bp.synthetic_key(samples[s][c]) if s == 0 else samples[s][c])
        ami = sorted(self.ami.clips, key=lambda c: c["id"])
        self.assertEqual(samples[2], [c["samples"] for c in ami])          # the SDM (student) samples
        for b in range(self.plan3.n_batches):
            rows = self.plan3.order[self.plan3.offsets[b]:self.plan3.offsets[b + 1]]
            self.assertLessEqual(len(rows) * int(self.plan3.samples[rows].max()), BUDGET)

    def test_deterministic_prefix_verification_and_refusals(self):
        a = bp.compute_sources(self.w.pool, N_POOLS, self.sources, **self.kw)
        self.assertEqual(a, bp.compute_sources(self.w.pool, N_POOLS, self.sources, **self.kw))
        short = bp.build(self.w.pool, self.root / "plans" / "v3-prefix", 2, sources=self.sources, **self.kw)
        self.assertTrue(bp.is_prefix(short, self.plan3))
        self.plan3.verify_journal(self.w.pool, sources=self.sources)
        other = self.sources[:2] + [bp.Source("ami-sdm", "ami-sdm:", self.ami, 17, "times:4", False)]
        with self.assertRaises(bp.PlanError):
            self.plan3.verify_journal(self.w.pool, sources=other)
        with self.assertRaises(bp.PlanError):
            self.plan3.verify_journal(self.w.pool)
        for share in ("times:2", "about:2"):                                  # version 2 has no such shares
            with self.assertRaises(bp.PlanError):
                bp.compute_sources(self.w.pool, N_POOLS, [bp.Source("x", "x:", self.once, 1, share, False)],
                                   pool_batches=P, budget=BUDGET, seed=SEED)
        for share in ("times:0", "about:x", "sometimes", 1.5):
            with self.assertRaises(bp.PlanError):
                bp.share_kind(share, bp.V3)
        with self.assertRaises(bp.PlanError):
            bp.compute_sources(self.w.pool, N_POOLS, self.sources, pool_batches=P, budget=BUDGET, seed=SEED, version=4)
        self.assertNotEqual(bp.production_dir("c", 4, sources=self.sources, version=bp.V3),
                            bp.production_dir("c", 4, sources=self.sources))
        with self.assertRaises(ValueError):
            mixing.plan([1.0], [0.5], 1, 0.1, once=True, passes=2)

    def test_the_synthetic_union(self):
        self.assertEqual(len(self.union.clips), len(self.full.clips) + len(self.pairs.clips))
        self.assertEqual(self.union.owner(self.pairs.clips[0]["id"]), "pairs")
        c = self.full.clips[2]
        self.assertTrue(np.array_equal(self.union.read(c["id"]), self.full.read(c["id"])))
        swapped = ClipUnion([("pairs", self.pairs), ("full", self.full)])
        self.assertNotEqual(swapped.identity, self.union.identity)
        self.assertNotEqual(ClipUnion([("full", self.full)]).identity, self.union.identity)
        with self.assertRaises(ValueError):
            ClipUnion([("full", self.full), ("again", self.full)])            # an id in two parts
        self.assertEqual([p[0] for p in self.union.parts], ["full", "pairs"])

    def test_the_replay_serves_pair_rows_with_both_recordings_and_the_label_stores_text(self):
        extras = [treplay.ExtraSource(self.sources[0], None), treplay.ExtraSource(self.sources[1], None),
                  treplay.ExtraSource(self.sources[2], None)]
        with self.assertRaises(ValueError):
            treplay.ExtraSource(self.sources[2], lambda item, b: item)       # a pair source takes no transform
        r = treplay.PlannedReplay(self.plan3, self.w.pool, store=self.store, sources=extras)
        real = label_sources.StoreLabels(self.labels())
        labels = label_sources.Labels(real, {"syn:": label_sources.ClipLabels(self.union, "syn:"),
                                             "cv27-en:": label_sources.ClipLabels(self.once, "cv27-en:"),
                                             "ami-sdm:": ami_store.PairLabels(self.ami, real)})
        by_sdm = {sid: (hid, spcm, hpcm) for sid, hid, spcm, hpcm in self.pair_specs}
        seen_pairs, rejected_pairs, steps = 0, 0, 0
        bt = label_sources.Batcher(r, labels, start=0, stop=r.through_batch)
        try:
            for t in range(r.through_batch):
                got = bt.next(timeout=60)
                items = r.read(t)
                mask = r.label_mask(t)
                for row, (i, src) in enumerate(zip(got["ids"], got["sources"])):
                    if src != "ami-sdm":
                        self.assertEqual(mask[row], not i.startswith(("syn:", "cv27-en:")))
                        continue
                    self.assertTrue(mask[row])
                    sid = i[len("ami-sdm:"):]
                    hid, spcm, hpcm = by_sdm[sid]
                    self.assertNotIn("label", items[row])
                    np.testing.assert_array_equal(items[row]["audio"], spcm.astype(np.float32) / 32768.0)
                    np.testing.assert_array_equal(got["teacher_audio"][row], hpcm.astype(np.float32) / 32768.0)
                    self.assertEqual(got["labels"][row], self.labels().text(hid))
                    seen_pairs += 1
                    rejected_pairs += got["labels"][row] is None
                if any(x is not None for x in got["labels"]):
                    steps += 1
                if "ami-sdm" not in got["sources"]:
                    self.assertNotIn("teacher_audio", got)
        finally:
            bt.close()
        self.assertEqual(seen_pairs, K_AMI * len(self.pair_specs))
        self.assertEqual(rejected_pairs, K_AMI)                                # the rejected IHM id, each draw
        acc = treplay.plan(r, labels, r.through_batch)
        self.assertEqual(acc["steps"], steps)
        self.assertEqual((acc["pair_rows"], acc["kept_pair_rows"]), (seen_pairs, seen_pairs - rejected_pairs))
        self.assertEqual(acc["synthetic_rows"], self.plan3.meta["n_extra"] - seen_pairs)
        self.assertIn("pairs", r.identity["sources"][2])

    def test_the_pooled_read_equals_the_sequential_read(self):
        """The extra rows' files are read in the decode pool (the batcher's): every item, the TS-augmented ones and
        both recordings of a pair included, is the sequential read's."""
        import concurrent.futures as cf
        from synthetic import augment
        from test_synthetic_mixing import make_aug_pool
        aug = augment.Augmenter(make_aug_pool(self.root / "src-aug-pooled"), seed=3)
        extras = [treplay.ExtraSource(self.sources[0], aug), treplay.ExtraSource(self.sources[1], None),
                  treplay.ExtraSource(self.sources[2], None)]
        r = treplay.PlannedReplay(self.plan3, self.w.pool, store=self.store, sources=extras)
        compared = 0
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            for t in range(r.through_batch):
                seq, par = r.read(t), r.read(t, pool)
                self.assertEqual([sorted(i) for i in seq], [sorted(i) for i in par])
                for x, y in zip(seq, par):
                    for k in x:
                        if isinstance(x[k], np.ndarray):
                            self.assertTrue(np.array_equal(x[k], y[k]), (t, x["id"], k))
                        else:
                            self.assertEqual(x[k], y[k], (t, x["id"], k))
                    compared += x["source"] in ("synthetic", "ami-sdm")
        self.assertGreater(compared, 0)
        r.close()

    def test_a_pair_batch_with_only_rejected_rows_takes_no_step(self):
        """plan()'s accounting counts a pair row only through its label: a batch whose rows are all rejected (real
        and pair rows) is consumed without a step."""
        fake = types.SimpleNamespace(through_batch=2, batch_ids=lambda b: ["ami-sdm:x", "real-1"],
                                     synthetic_mask=lambda b: [True, False], label_mask=lambda b: [True, True])
        labels = types.SimpleNamespace(text=lambda i: None)
        acc = treplay.plan(fake, labels, 2)
        self.assertEqual((acc["steps"], acc["empty_batches"], acc["pair_rows"], acc["kept_pair_rows"]), (0, [0, 1], 2, 0))


class Version2Unchanged(World):
    """Stage 3's multi-source version 2 plan is the same bytes as before version 3 (its meta has no version 3 keys)."""

    def test_a_version_2_plan_keeps_its_meta_and_id(self):
        from test_training_sources import store as v2store
        syn = v2store(self.root / "v2-syn", 6, "syn-store")
        sources = [bp.Source("synthetic", "syn:", syn, 11, 0.2, True)]
        body, _ = bp.compute_sources(self.w.pool, N_POOLS, sources, pool_batches=P, budget=BUDGET, seed=SEED)
        self.assertEqual(body["version"], bp.SYNTHETIC_VERSION)
        self.assertTrue(body["plan_id"].endswith(f"-x{bp.sources_id(sources)}"))
        self.assertNotIn("kind", body["sources"][0])
        self.assertNotIn("draws_per_clip", body["sources"][0])
        self.assertEqual(sorted(body["sources"][0]), sorted(
            ["config", "share_value", "share_rule", "key_rule", "n_clips", "clip_order", "draw_rule", "draws",
             "distinct", "seconds", "share_of_real", "pool_share_min", "pool_share_max", "max_pool_error_seconds"]))


if __name__ == "__main__":
    unittest.main()
