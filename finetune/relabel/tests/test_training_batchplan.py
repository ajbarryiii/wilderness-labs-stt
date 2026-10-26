"""Contract tests: the batch plan (b0prime plan "Batch plan" and "Units", 2026-10-24), its materialized pool
audio and the planned replay, on real C1 shards and C2 label records (training_fixtures), plus checks on the real
journal when it is present."""
from __future__ import annotations

import collections
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import train  # finetune/parakeet-ternary
from labeling import shards
from labeling_fixtures import relabel_root
from training import batchplan as bp
from training import labels as tlabels
from training import poolaudio
from training import replay as treplay
from training import schedule
from training_fixtures import TOTAL, LabeledWorld

P = 4                    # fixture pools: batches 0-3, 4-7, 8-11, 12-13 (the journal's last pool is short)
N_POOLS = 4
BUDGET = 160_000         # the fixture stream's 10 s batches
SEED = 7


def cut_batches(plan: bp.Plan) -> list[list[int]]:
    return [plan.order[plan.offsets[b]:plan.offsets[b + 1]].tolist() for b in range(plan.n_batches)]


class World(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.w = LabeledWorld(cls.root)
        cls.seq = cls.w.journal                       # (batch, pos) order across shards: index = sequence number
        cls.plan = bp.build(cls.w.pool, cls.root / "plans" / "full", N_POOLS, pool_batches=P, budget=BUDGET, seed=SEED)
        cls.store = cls.root / "pool-audio-store"
        poolaudio.materialize(cls.plan, cls.w.pool, cls.store)
        ref = treplay.JournalReplay(cls.w.pool, TOTAL)
        cls.audio = {}                                # sequence number -> the stream's audio of that occurrence
        k = 0
        for b in range(TOTAL):
            for item in ref.read(b):
                cls.audio[k] = item["audio"]
                k += 1

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def labels(self, label_set="T*"):
        return tlabels.LabelStore.open(self.w.labels_dir, self.w.pool, label_set=label_set, through_batch=TOTAL)

    def replay(self, audio="materialized", plan=None):
        return treplay.PlannedReplay(plan or self.plan, self.w.pool, audio=audio, store=self.store)

    def copy_world(self, name):
        d = self.root / "copies" / name
        shutil.copytree(self.w.pool, d / "pool")
        return d / "pool"

    def rewrite_journal(self, pool, k, change):
        """Rewrite C1 shard k's journal through `change(rows)` and update its meta, as C1 would have written it."""
        p = shards.paths(pool, k)
        rows = shards.read_journal(p["journal"])
        change(rows)
        sha, size = shards.write_journal(p["journal"], rows)
        meta = shards.load_meta(pool, k)
        meta["files"]["journal"].update(sha256=sha, bytes=size)
        p["meta"].write_text(json.dumps(meta))


class Cut(World):
    def test_covers_every_journal_occurrence_exactly_once(self):
        n = len(self.seq)
        self.assertEqual(self.plan.meta["n_occurrences"], n)
        served = [s for t in range(self.plan.n_batches) for s in self.plan.batch(t).tolist()]
        self.assertEqual(sorted(served), list(range(n)))
        repeated = [i for i, c in collections.Counter(r["id"] for r in self.seq).items() if c > 1]
        self.assertTrue(repeated)                     # repeats are separate occurrences, each planned once
        for i in repeated:
            want = sum(r["id"] == i for r in self.seq)
            self.assertEqual(sum(self.seq[s]["id"] == i for s in served), want)

    def test_pools_are_consecutive_consumed_batches_and_stay_whole(self):
        self.assertEqual(self.plan.consumed_batches, TOTAL)
        for p in range(N_POOLS):
            lo, hi = p * P, min((p + 1) * P, TOTAL)
            want = [k for k, r in enumerate(self.seq) if lo <= r["batch"] < hi]
            got = [s for t in range(self.plan.pool_start(p), self.plan.pool_start(p + 1)) for s in self.plan.batch(t).tolist()]
            self.assertEqual(sorted(got), want)
        self.assertEqual(self.plan.pool_start(N_POOLS), self.plan.n_batches)

    def test_every_batch_is_within_the_budget(self):
        samples = self.plan.samples
        for rows in cut_batches(self.plan):
            self.assertLessEqual(len(rows) * int(samples[rows].max()), BUDGET)
        per_pool = np.diff(self.plan.batch_pools)
        self.assertTrue(np.any(per_pool > 1))         # the budget really splits pools

    def test_cut_equals_train_cut_pool_float_rule(self):
        batches = cut_batches(self.plan)
        for p in range(N_POOLS):
            lo, hi = int(self.plan.pools[p]), int(self.plan.pools[p + 1])
            durations = [float(self.seq[k]["duration"]) for k in range(lo, hi)]
            want = [[lo + i for i in idx] for idx in train.cut_pool(durations, BUDGET / bp.SAMPLE_RATE)]
            got = batches[self.plan.batch_pools[p]:self.plan.batch_pools[p + 1]]
            self.assertEqual(got, want)

    def test_key_is_whole_samples(self):
        for k, r in enumerate(self.seq):
            self.assertEqual(int(self.plan.samples[k]), round(r["duration"] * 16000))

    def test_sort_key_ties_and_greedy_rule(self):
        samples = np.array([5, 3, 5, 3, 5, 9, 4, 4, 4], dtype=np.int64)
        pools = np.array([0, 5, 9])
        order, offsets, bpools = bp.cut(samples, pools, budget=100)
        self.assertEqual(order[:5].tolist(), [1, 3, 0, 2, 4])          # by samples, ties by sequence number
        self.assertEqual(order[5:].tolist(), [6, 7, 8, 5])
        _, offsets, _ = bp.cut(samples, pools, budget=8)
        got = [order[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
        # pool 0: 3,3 (2x3=6) | 5 (2x5=10>8) | 5 | 5; pool 1: 4,4 (=8 fits) | 4 | 9 alone (longer than the budget)
        self.assertEqual(got, [[1, 3], [0], [2], [4], [6, 7], [8], [5]])
        for a, b, c in zip(offsets[:-2], offsets[1:-1], offsets[2:]):  # greedy: the next clip would not fit
            if c <= b:
                continue
            first_next = int(samples[order[b]])
            same_pool = np.searchsorted(pools, a, side="right") == np.searchsorted(pools, b, side="right")
            if same_pool:
                self.assertGreater((b - a + 1) * first_next, 8)

    def test_cut_is_deterministic_and_the_plan_depends_only_on_the_journal(self):
        a = bp.compute(self.w.pool, N_POOLS, pool_batches=P, budget=BUDGET, seed=SEED)
        b = bp.compute(self.w.pool, N_POOLS, pool_batches=P, budget=BUDGET, seed=SEED)
        self.assertEqual(a, b)
        self.assertEqual(bp.build(self.w.pool, self.plan.dir, N_POOLS, pool_batches=P, budget=BUDGET,
                                  seed=SEED).sha256, self.plan.sha256)
        with self.assertRaises(bp.PlanError):
            bp.build(self.w.pool, self.plan.dir, N_POOLS, pool_batches=P, budget=BUDGET, seed=SEED + 1)

    def test_shuffle_is_batchers_generator_within_each_pool(self):
        for p in range(N_POOLS):
            lo, hi = self.plan.pool_start(p), self.plan.pool_start(p + 1)
            perm = np.random.default_rng([SEED, 2, p]).permutation(hi - lo)
            self.assertEqual(self.plan.shuffle[lo:hi].tolist(), (lo + perm).tolist())
        other = bp.compute(self.w.pool, N_POOLS, pool_batches=P, budget=BUDGET, seed=SEED + 1)[1]
        self.assertEqual(other["order.u32"], (self.plan.dir / "order.u32").read_bytes())   # the cut ignores the seed

    def test_files_are_little_endian_u32_as_documented(self):
        for name in bp.FILES:
            raw = (self.plan.dir / name).read_bytes()
            self.assertEqual(len(raw) % 4, 0)
        offsets = bp.read_u32(self.plan.dir / "offsets.u32")
        self.assertEqual(int.from_bytes((self.plan.dir / "offsets.u32").read_bytes()[4:8], "little"), int(offsets[1]))
        self.assertEqual(len(offsets), self.plan.n_batches + 1)
        meta = json.loads((self.plan.dir / "plan.json").read_text())
        self.assertEqual((meta["pool_batches"], meta["budget_samples"], meta["seed"]), (P, BUDGET, SEED))
        self.assertEqual([s["shard"] for s in meta["journal_shards"]], shards.committed(self.w.pool))


class Files(World):
    def copy_plan(self, name):
        d = self.root / "plan-copies" / name
        shutil.copytree(self.plan.dir, d)
        return d

    def test_every_file_is_verified_against_plan_json(self):
        for name in bp.FILES:
            d = self.copy_plan(f"bad-{name}")
            raw = bytearray((d / name).read_bytes())
            raw[0] ^= 1
            (d / name).write_bytes(bytes(raw))
            with self.assertRaises(bp.PlanError):
                bp.Plan.load(d)

    def test_plan_json_is_the_identity(self):
        d = self.copy_plan("edited-json")
        meta = json.loads((d / "plan.json").read_text())
        meta["seed"] = SEED + 1
        (d / "plan.json").write_text(json.dumps(meta))
        other = bp.Plan.load(d)
        self.assertNotEqual(other.sha256, self.plan.sha256)
        with self.assertRaises(bp.PlanError):           # its shuffle no longer follows from its seed
            other.verify_journal(self.w.pool)

    def test_verify_journal_accepts_its_journal_and_refuses_another(self):
        self.plan.verify_journal(self.w.pool)
        pool = self.copy_world("changed-journal")

        def longer(rows):
            rows[0]["duration"] = rows[0]["duration"] + 1.0
        self.rewrite_journal(pool, 1, longer)
        with self.assertRaises(bp.PlanError):
            self.plan.verify_journal(pool)

    def test_refuses_a_duration_that_is_not_whole_samples(self):
        pool = self.copy_world("inexact")

        def inexact(rows):
            rows[2]["duration"] = rows[2]["duration"] + 1e-5
        self.rewrite_journal(pool, 0, inexact)
        with self.assertRaises(bp.PlanError):
            bp.compute(pool, N_POOLS, pool_batches=P, budget=BUDGET, seed=SEED)

    def test_prefix_plans(self):
        short = bp.build(self.w.pool, self.root / "plans" / "two", 2, pool_batches=P, budget=BUDGET, seed=SEED)
        self.assertEqual(short.consumed_batches, 2 * P)
        self.assertTrue(bp.is_prefix(short, self.plan))
        other = bp.build(self.w.pool, self.root / "plans" / "two-seed", 2, pool_batches=P, budget=BUDGET, seed=SEED + 1)
        self.assertFalse(bp.is_prefix(other, self.plan))

    def test_only_the_complete_journals_last_pool_may_be_short(self):
        with self.assertRaises(bp.PlanError):
            bp.compute(self.w.pool, N_POOLS + 1, pool_batches=P, budget=BUDGET, seed=SEED)
        pool = self.copy_world("incomplete")
        (pool / "manifest.json").unlink()
        with self.assertRaises(RuntimeError):
            bp.compute(pool, N_POOLS, pool_batches=P, budget=BUDGET, seed=SEED)
        self.assertEqual(bp.compute(pool, 3, pool_batches=P, budget=BUDGET, seed=SEED)[0]["consumed_batches"], 12)


class Serving(World):
    def test_replay_serves_exactly_the_planned_clips_with_their_audio(self):
        for audio in ("materialized", "c1"):
            r = self.replay(audio)
            self.assertEqual(r.through_batch, self.plan.n_batches)
            for t in range(self.plan.n_batches):
                want = self.plan.batch(t).tolist()
                got = r.read(t)
                self.assertEqual([x["id"] for x in got], [self.seq[k]["id"] for k in want])
                self.assertEqual(r.batch_ids(t), [self.seq[k]["id"] for k in want])
                self.assertEqual([x["source"] for x in got], [self.seq[k]["source"] for k in want])
                self.assertEqual([x["duration"] for x in got], [self.seq[k]["duration"] for k in want])
                for x, k in zip(got, want):
                    self.assertEqual(x["audio"].dtype, np.float32)
                    self.assertTrue(np.array_equal(x["audio"], self.audio[k]), f"{audio} t={t} {x['id']}")

    def test_batcher_yields_the_planned_batches_with_their_labels(self):
        r, s = self.replay(), self.labels("B0")
        b = treplay.ReplayBatcher(r, s, start=0, stop=r.through_batch)
        try:
            for t in range(r.through_batch):
                got = b.next(timeout=30)
                items = r.read(t)
                want = train.collate(items)
                self.assertEqual((got["index"], got["position_after"]), (t, {"batch": t + 1}))
                self.assertEqual(got["ids"], want["ids"])
                self.assertTrue(torch.equal(got["audio"], want["audio"]))
                self.assertEqual(got["labels"], [s.text(i) for i in got["ids"]])
            with self.assertRaises(treplay.ReplayStopped):
                b.next(timeout=30)
        finally:
            b.close()

    def test_resume_at_a_planned_batch_boundary(self):
        r, s = self.replay(), self.labels()
        n = r.through_batch
        k = self.plan.pool_start(2) + 1                 # inside a pool, after a shuffled batch
        full = treplay.ReplayBatcher(r, s, start=0, stop=n)
        part = treplay.ReplayBatcher(r, s, start=k, stop=n)
        try:
            a = [full.next(timeout=30) for _ in range(n)]
            c = [part.next(timeout=30) for _ in range(n - k)]
            self.assertEqual([x["ids"] for x in a[k:]], [x["ids"] for x in c])
            self.assertEqual([x["labels"] for x in a[k:]], [x["labels"] for x in c])
            self.assertTrue(all(torch.equal(x["audio"], y["audio"]) for x, y in zip(a[k:], c)))
        finally:
            full.close()
            part.close()

    def test_rejected_rows_are_dropped_after_the_cut(self):
        r = self.replay()
        t_star, b0 = self.labels("T*"), self.labels("B0")
        mixed = 0
        for t in range(r.through_batch):
            ids = r.batch_ids(t)
            kept = [t_star.text(i) is not None for i in ids]
            self.assertEqual(kept, [b0.text(i) is not None for i in ids])   # one mask for both label sets
            mixed += any(kept) and not all(kept)
        self.assertGreater(mixed, 0)                    # rejected rows sit inside planned batches
        acc = treplay.plan(r, t_star, r.through_batch)
        nonempty = sum(any(t_star.text(i) is not None for i in r.batch_ids(t)) for t in range(r.through_batch))
        self.assertEqual(acc["steps"], nonempty)
        self.assertEqual(acc["occurrences"], len(self.seq))
        self.assertEqual(acc["kept_occurrences"], sum(t_star.text(x["id"]) is not None for x in self.seq))

    def test_extra_clips_are_appended_after_the_cut(self):
        r, s = self.replay(), self.labels()

        class Extra:
            def items(self, index, real):
                pcm = np.full(16000, index + 1, dtype=np.float32) / 32768.0
                return [{"audio": pcm, "duration": 1.0, "id": f"syn:{index}", "source": "syn", "label": f"syn {index}"}]

        plain = treplay.ReplayBatcher(r, s, start=0, stop=r.through_batch)
        extra = treplay.ReplayBatcher(r, s, start=0, stop=r.through_batch, extra=Extra())
        try:
            for t in range(r.through_batch):
                a, e = plain.next(timeout=30), extra.next(timeout=30)
                n = len(a["ids"])
                self.assertEqual(e["ids"], a["ids"] + [f"syn:{t}"])
                self.assertEqual(e["labels"], a["labels"] + [f"syn {t}"])
                self.assertTrue(torch.equal(e["audio"][:n, :a["audio"].shape[1]], a["audio"]))
        finally:
            plain.close()
            extra.close()

    def test_synthetic_plan_is_built_on_the_planned_batches(self):
        from synthetic import mixing
        r, s = self.replay(), self.labels()
        clips = [{"id": f"c{i}", "duration": 1.0 + i / 10, "label": f"clip {i}", "audio_filepath": "", "sha256": ""}
                 for i in range(5)]
        src = mixing.source_for("TSn", r, s, r.through_batch, pool=mixing.ClipPool(clips, "fixture", verify_audio=False))
        self.assertEqual(src.real, [[(x["id"], x["duration"]) for x in r.read(t)] for t in range(r.through_batch)])
        kept = [sum(x["duration"] for x in r.read(t) if s.text(x["id"]) is not None) for t in range(r.through_batch)]
        self.assertEqual(src.real_seconds, kept)

    def test_identity_names_the_plan_and_the_audio(self):
        a, c = self.replay("materialized").identity, self.replay("c1").identity
        self.assertEqual((a["plan_sha256"], a["through_batch"]), (self.plan.sha256, self.plan.n_batches))
        self.assertEqual(a["plan_sha256"], c["plan_sha256"])
        self.assertNotEqual(a["sha256"], c["sha256"])


class Seam(World):
    def test_offline_data_serves_the_plan_from_a_planned_batch_position(self):
        from training import seam as tseam
        r, s = self.replay(), self.labels()
        sched = schedule.pool_branch(schedule.pool_trunk(1e-4, self.plan, N_POOLS), self.plan, 2, N_POOLS)
        data = tseam.OfflineData(r, s, sched, r.through_batch)
        self.assertEqual(data.plan["stop_batch"], self.plan.n_batches)
        self.assertEqual(data.state_dict()["replay_sha256"], r.identity["sha256"])
        k = self.plan.pool_start(1) + 1
        b = data.batcher({"batch": k})
        try:
            got = b.next(timeout=30)
            self.assertEqual((got["index"], got["ids"]), (k, r.batch_ids(k)))
        finally:
            b.close()
        with self.assertRaises(ValueError):
            tseam.OfflineData(r, s, sched, r.through_batch + 1)


class Materialized(World):
    def test_each_pool_holds_each_of_its_ids_once(self):
        store = poolaudio.PoolAudio(self.store, self.plan)
        for p in range(N_POOLS):
            lo, hi = int(self.plan.pools[p]), int(self.plan.pools[p + 1])
            want = {self.seq[k]["id"] for k in range(lo, hi)}
            index = store.index(p)
            self.assertEqual(sorted(index["id"]), sorted(want))

    def test_resumed_materialization_is_byte_identical(self):
        store = self.root / "store-resumed"
        status = {"complete": False}
        rounds = 0
        while not status["complete"]:
            status = poolaudio.materialize(self.plan, self.w.pool, store, checkpoint_shards=1, stop_after_shards=1)
            for part in (store / "work").glob("*.part"):   # a crash after the checkpoint leaves extra bytes
                with open(part, "ab") as handle:
                    handle.write(b"junk")
            rounds += 1
        self.assertGreater(rounds, 2)
        for p in range(N_POOLS):
            for suffix in (".flac", ".index.parquet", ".json"):
                name = f"{poolaudio.pool_name(p)}{suffix}"
                self.assertEqual((store / "pools" / name).read_bytes(), (self.store / "pools" / name).read_bytes(), name)

    def test_refuses_c1_audio_that_differs_from_its_meta(self):
        pool = self.copy_world("bad-audio")
        p = shards.paths(pool, 1)["audio"]
        raw = bytearray(p.read_bytes())
        raw[len(raw) // 2] ^= 1
        p.write_bytes(bytes(raw))
        with self.assertRaises(RuntimeError):
            poolaudio.materialize(self.plan, pool, self.root / "store-bad")

    def test_reader_refuses_a_missing_or_foreign_pool(self):
        store = self.root / "store-copy"
        shutil.copytree(self.store, store)
        (store / "pools" / f"{poolaudio.pool_name(1)}.json").unlink()
        with self.assertRaises(RuntimeError):
            poolaudio.PoolAudio(store, self.plan)
        shutil.rmtree(store)
        shutil.copytree(self.store, store)
        meta = json.loads((store / "store.json").read_text())
        meta["pool_batches"] = P + 1
        (store / "store.json").write_text(json.dumps(meta))
        with self.assertRaises(RuntimeError):
            poolaudio.PoolAudio(store, self.plan)

    def test_reader_checks_every_clip_against_c1s_hash(self):
        store = self.root / "store-flipped"
        shutil.copytree(self.store, store)
        f = store / "pools" / f"{poolaudio.pool_name(0)}.flac"
        raw = bytearray(f.read_bytes())
        raw[-10] ^= 0x55
        f.write_bytes(bytes(raw))
        r = treplay.PlannedReplay(self.plan, self.w.pool, audio="materialized", store=store)
        with self.assertRaises(Exception):
            for t in range(self.plan.pool_start(0), self.plan.pool_start(1)):
                r.read(t)


class Reader(World):
    def test_at_most_cache_pools_are_resident_while_a_pool_loads(self):
        seen = []
        real = poolaudio.PoolAudio._read

        def spy(audio, p):
            with audio._lock:
                seen.append(len(audio._cache))
            return real(audio, p)
        with mock.patch.object(poolaudio.PoolAudio, "_read", spy):
            r = self.replay()
            b = treplay.ReplayBatcher(r, self.labels(), start=0, stop=r.through_batch)
            try:
                for _ in range(r.through_batch):
                    b.next(timeout=30)
            finally:
                b.close()
                r.close()
        self.assertGreaterEqual(len(seen), N_POOLS)
        self.assertLessEqual(max(seen), poolaudio.CACHE_POOLS - 1)


class Probe(World):
    def test_required_rate_follows_the_fastest_student(self):
        from training import replay_probe
        d = self.root / "decoder-only" / "probe"
        d.mkdir(parents=True, exist_ok=True)
        (d / "B.json").write_text(json.dumps({"step_time_s": 0.5}))
        (d / "T.json").write_text(json.dumps({"step_time_s": 0.4}))
        rate, info = replay_probe.required_rate()
        self.assertAlmostEqual(rate, 1.25 / 0.4)
        self.assertEqual(info["step_time_s"], {"B": 0.5, "T": 0.4})

    def test_every_region_starts_with_a_fresh_reader(self):
        from training import replay_probe
        reads = []
        real = poolaudio.PoolAudio._read

        def spy(audio, p):
            reads.append((id(audio), p))
            return real(audio, p)
        with mock.patch.object(poolaudio.PoolAudio, "_read", spy):
            out = replay_probe.run(self.plan, self.w.pool, self.store, [(0, 1), (1, 2)], ["materialized", "c1"], 60.0)
        self.assertEqual([(x["reader"], x["pools"]) for x in out["regions"]],
                         [("materialized", [0, 1]), ("materialized", [1, 2]), ("c1", [0, 1]), ("c1", [1, 2])])
        readers = list(dict.fromkeys(a for a, _ in reads))
        self.assertEqual(len(readers), 2)
        self.assertIn((readers[1], 1), reads)          # region 2 read pool 1 itself (not region 1's prefetch)
        self.assertTrue(all(x["planned_batches"] == x["of"] for x in out["regions"]))


class PoolSchedule(World):
    def test_wsd_boundaries_are_pool_starts(self):
        trunk = schedule.pool_trunk(1e-4, self.plan, n_pools=N_POOLS)
        self.assertEqual(trunk.warmup, round(0.02 * self.plan.n_batches))
        s = schedule.pool_branch(trunk, self.plan, start_pool=2, end_pool=N_POOLS)
        self.assertEqual(s.cooldown_start, self.plan.pool_start(2))
        self.assertEqual(s.stop_batch, self.plan.n_batches)
        self.assertEqual(s.lr(s.cooldown_start), 1e-4)
        self.assertEqual(s.lr(s.last_batch), 0.0)
        with self.assertRaises(ValueError):
            schedule.pool_branch(trunk, self.plan, start_pool=3, end_pool=N_POOLS + 1)

    def test_decoder_only_span_and_phases(self):
        self.assertEqual((schedule.DECODER_ONLY_POOLS, schedule.DECODER_ONLY_COOLDOWN_POOL), (156, 133))


SYN_SEED, SHARE = 11, 0.1


class Synthetic(World):
    """Version 2 plans (synthetic plan "Mixing", 2026-10-24): synthetic clips join each pool before the cut."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from test_synthetic_mixing import make_aug_pool, make_pool
        cls.clips = make_pool(cls.root / "syn", n=40)
        cls.aug_pool = make_aug_pool(cls.root / "aug")
        cls.kw = dict(pool_batches=P, budget=BUDGET, seed=SEED, syn_seed=SYN_SEED, share=SHARE)
        cls.plan2 = bp.build(cls.w.pool, cls.root / "plans" / "syn", N_POOLS, clips=cls.clips, **cls.kw)

    def syn_replay(self, transform=None):
        return treplay.PlannedReplay(self.plan2, self.w.pool, audio="materialized", store=self.store,
                                     synthetic=self.clips, synthetic_transform=transform)

    def clip_samples(self):
        return bp.clip_samples(self.clips)

    def test_every_real_occurrence_appears_exactly_once(self):
        rows = [e for t in range(self.plan2.n_batches) for e in self.plan2.batch(t).tolist()]
        real = sorted(e for e in rows if e < bp.SYN_FLAG)
        self.assertEqual(real, list(range(len(self.seq))))
        self.assertEqual(len(rows) - len(real), self.plan2.meta["n_synthetic"])
        self.assertGreater(self.plan2.meta["n_synthetic"], N_POOLS)
        for p in range(N_POOLS):                       # a pool's batches hold its real occurrences and its draw
            got = [e for t in range(self.plan2.pool_start(p), self.plan2.pool_start(p + 1))
                   for e in self.plan2.batch(t).tolist() if e < bp.SYN_FLAG]
            self.assertEqual(sorted(got), list(range(int(self.plan.pools[p]), int(self.plan.pools[p + 1]))))

    def test_budget_holds_for_every_batch_even_at_the_longest_augmentation(self):
        keys = self.plan2.samples
        for b in range(self.plan2.n_batches):
            rows = self.plan2.order[self.plan2.offsets[b]:self.plan2.offsets[b + 1]]
            self.assertLessEqual(len(rows) * int(keys[rows].max()), BUDGET)
        longest_synthetic = 0                          # batches whose longest row is synthetic: the stretch matters
        for b in range(self.plan2.n_batches):
            rows = self.plan2.order[self.plan2.offsets[b]:self.plan2.offsets[b + 1]]
            top = rows[np.argmax(keys[rows])]
            longest_synthetic += int(self.plan2.entries[top] >= bp.SYN_FLAG)
        self.assertGreater(longest_synthetic, 0)

        def stretch(item, index):                      # the worst case the key allows
            n = bp.synthetic_key(len(item["audio"]))
            return {**item, "audio": np.resize(item["audio"], n).astype(np.float32)}
        from synthetic import augment
        for transform in (None, stretch, augment.Augmenter(self.aug_pool, seed=3)):
            r = self.syn_replay(transform)
            b = treplay.ReplayBatcher(r, self.labels(), start=0, stop=r.through_batch)
            try:
                for _ in range(r.through_batch):
                    got = b.next(timeout=60)
                    self.assertLessEqual(got["audio"].shape[0] * got["audio"].shape[1], BUDGET)
            finally:
                b.close()

    def test_synthetic_share_per_pool_is_within_one_clip(self):
        cs = self.clip_samples()
        longest = max(cs) / 16000
        total_syn = total_real = 0.0
        for p in range(N_POOLS):
            real = int(self.plan.samples[self.plan.pools[p]:self.plan.pools[p + 1]].sum()) / 16000
            ent = self.plan2.entries[self.plan2.pools[p]:self.plan2.pools[p + 1]]
            syn = sum(cs[e - bp.SYN_FLAG] for e in ent.tolist() if e >= bp.SYN_FLAG) / 16000
            self.assertLessEqual(abs(syn - SHARE * real), longest)
            total_syn, total_real = total_syn + syn, total_real + real
            self.assertLessEqual(abs(total_syn - SHARE * total_real), longest / 2 + 1e-9)
        self.assertAlmostEqual(self.plan2.meta["stats"]["synthetic"]["share_of_real"], total_syn / total_real)
        lengths = np.asarray([cs[e - bp.SYN_FLAG] if e >= bp.SYN_FLAG else int(self.plan.samples[e])
                              for e in self.plan2.entries.tolist()])
        st = bp.stats(lengths, self.plan2.order, self.plan2.offsets, self.plan2.consumed_batches)
        self.assertAlmostEqual(self.plan2.meta["stats"]["padding"], st["padding"])      # served audio, not keys
        self.assertAlmostEqual(self.plan2.meta["stats"]["real_seconds"], (lengths.sum()) / 16000)
        self.assertGreater(self.plan2.meta["stats"]["by_key"]["padded_seconds"], 0)

    def test_synthetic_keys_are_the_longest_augmented_length(self):
        from synthetic import augment
        bp.check_stretch()
        cs = self.clip_samples()
        for e, k in zip(self.plan2.entries.tolist(), self.plan2.samples.tolist()):
            if e >= bp.SYN_FLAG:
                self.assertEqual(k, -(-cs[e - bp.SYN_FLAG] * 10 // 9))
        for n in (1, 9, 16000, 16001, 48017):
            for f in (0.9, 0.90001, 0.93, 1.1):
                self.assertLessEqual(len(augment.speed(np.zeros(n, np.float32), f)), bp.synthetic_key(n))

    def test_deterministic_and_the_base_plan_is_untouched(self):
        a = bp.compute_synthetic(self.w.pool, N_POOLS, self.clips, **self.kw)
        b = bp.compute_synthetic(self.w.pool, N_POOLS, self.clips, **self.kw)
        self.assertEqual(a, b)
        self.assertEqual(self.plan2.meta["base_plan_sha256"], self.plan.sha256)
        self.assertEqual(bp.Plan.load(self.plan.dir).sha256, self.plan.sha256)
        other = bp.compute_synthetic(self.w.pool, N_POOLS, self.clips, **{**self.kw, "syn_seed": SYN_SEED + 1})
        self.assertNotEqual(other[1]["entries.u32"], a[1]["entries.u32"])
        order, offsets, _ = bp.cut(self.plan2.samples, self.plan2.pools, BUDGET)   # the version 1 cut contract
        self.assertTrue(np.array_equal(order, self.plan2.order) and np.array_equal(offsets, self.plan2.offsets))
        short = bp.build(self.w.pool, self.root / "plans" / "syn-two", 2, clips=self.clips, **self.kw)
        self.assertTrue(bp.is_prefix(short, self.plan2))
        self.assertFalse(bp.is_prefix(short, self.plan))

    def test_replay_serves_real_and_synthetic_rows_in_cut_order(self):
        r, s = self.syn_replay(), self.labels()
        by_id = {c["id"]: c for c in self.clips.clips}
        bt = treplay.ReplayBatcher(r, s, start=0, stop=r.through_batch)
        try:
            for t in range(r.through_batch):
                got = bt.next(timeout=60)
                ents = self.plan2.batch(t).tolist()
                want_ids = [f"syn:{sorted(by_id)[e - bp.SYN_FLAG]}" if e >= bp.SYN_FLAG else self.seq[e]["id"]
                            for e in ents]
                self.assertEqual(got["ids"], want_ids)
                for row, (i, e) in enumerate(zip(got["ids"], ents)):
                    if e >= bp.SYN_FLAG:
                        self.assertEqual(got["labels"][row], by_id[i[4:]]["label"])
                    else:
                        self.assertEqual(got["labels"][row], s.text(i))     # None: rejected, dropped by the seam
                        n = int(got["audio_len"][row])
                        self.assertTrue(torch.equal(got["audio"][row, :n], torch.from_numpy(self.audio[e])))
        finally:
            bt.close()
        acc = treplay.plan(r, s, r.through_batch)
        self.assertEqual(acc["occurrences"], len(self.seq))
        self.assertEqual(acc["synthetic_rows"], self.plan2.meta["n_synthetic"])
        self.assertEqual(acc["kept_occurrences"], sum(s.text(x["id"]) is not None for x in self.seq))

    def test_ts_and_tsn_get_identical_clips_in_identical_batches(self):
        from synthetic import augment
        a, b = self.syn_replay(), self.syn_replay(augment.Augmenter(self.aug_pool, seed=3))
        changed = 0
        for t in range(a.through_batch):
            x, y = a.read(t), b.read(t)
            self.assertEqual([i["id"] for i in x], [i["id"] for i in y])
            self.assertEqual([i.get("label") for i in x], [i.get("label") for i in y])
            for i, j in zip(x, y):
                if "label" not in i:
                    self.assertTrue(np.array_equal(i["audio"], j["audio"]))
                else:
                    changed += not np.array_equal(i["audio"], j["audio"])
        self.assertGreater(changed, 0)                 # augmentation really ran, on synthetic rows only

    def test_refusals(self):
        def too_long(item, index):
            n = bp.synthetic_key(len(item["audio"])) + 1
            return {**item, "audio": np.zeros(n, np.float32)}
        r = self.syn_replay(too_long)
        syn_t = next(t for t in range(r.through_batch) if any(r.synthetic_mask(t)))
        with self.assertRaises(RuntimeError):
            r.read(syn_t)
        from test_synthetic_mixing import make_pool
        with self.assertRaises(bp.PlanError):           # another clip pool
            other = make_pool(self.root / "syn-other", n=41)
            treplay.PlannedReplay(self.plan2, self.w.pool, store=self.store, synthetic=other)
        with self.assertRaises(ValueError):
            treplay.PlannedReplay(self.plan2, self.w.pool, store=self.store)
        with self.assertRaises(ValueError):
            treplay.PlannedReplay(self.plan, self.w.pool, store=self.store, synthetic=self.clips)
        d = self.root / "plan-copies" / "syn-entries"
        shutil.copytree(self.plan2.dir, d)
        raw = bytearray((d / "entries.u32").read_bytes())
        raw[0] ^= 1
        (d / "entries.u32").write_bytes(bytes(raw))
        with self.assertRaises(bp.PlanError):
            bp.Plan.load(d)


REAL = Path(os.environ.get("RELABEL_ROOT", "/nonexistent")) / "pool-audio"


@unittest.skipUnless(REAL.exists() and any(REAL.iterdir()), "needs the real C1 journal")
class RealJournal(unittest.TestCase):
    """The key is exact and the integer cut equals train.cut_pool's float rule on the real journal."""

    def test_integer_key_and_cut_on_the_real_journal(self):
        pool = sorted(REAL.iterdir())[0]
        ks = shards.committed(pool)
        through = (shards.load_meta(pool, ks[-1])["batches"][1] // bp.POOL_BATCHES) * bp.POOL_BATCHES
        j = bp.read_journal(pool, through)               # refuses any inexact duration
        n_pools = through // bp.POOL_BATCHES
        pools = bp.pool_starts(j["batch"], n_pools, bp.POOL_BATCHES)
        order, offsets, bpools = bp.cut(j["samples"], pools, bp.BUDGET_SAMPLES)
        d = np.asarray(j["durations"])
        self.assertTrue(np.array_equal(j["samples"] / 16000.0, d))
        for p in range(n_pools):
            lo, hi = int(pools[p]), int(pools[p + 1])
            want = [[lo + i for i in idx] for idx in train.cut_pool(d[lo:hi].tolist(), 600.0)]
            got = [order[a:b].tolist() for a, b in zip(offsets[bpools[p]:bpools[p + 1]], offsets[bpools[p] + 1:bpools[p + 1] + 1])]
            self.assertEqual(got, want, f"pool {p}")
        if n_pools >= 16:                                # the independent reference's figures (first 2,000 batches)
            first = int(np.searchsorted(j["batch"], 2000))
            p2000 = bp.pool_starts(j["batch"][:first], 16, bp.POOL_BATCHES)
            p2000[-1] = first
            o, off, _ = bp.cut(j["samples"][:first], p2000, bp.BUDGET_SAMPLES)
            st = bp.stats(j["samples"][:first], o, off, 2000)
            self.assertEqual(len(off) - 1, 1657)
            self.assertAlmostEqual(st["padding"], 0.0161, places=4)


if __name__ == "__main__":
    unittest.main()
