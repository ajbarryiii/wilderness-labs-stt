"""Contract tests: the synthetic clip source and the augmentation (synthetic plan "Mixing", "Augmentation").

On real C1 shards and C2 labels (training_fixtures.LabeledWorld) through training/replay.py's ReplayBatcher:
- TS and TSn receive identical clips in identical batches;
- the real batch sequence is unchanged (real rows first, identical ids, labels and audio);
- the synthetic share stays at about 10% of each batch's trained real audio; a fully rejected batch gets none;
- augmentation never changes labels and is off in TSn (TSn audio is the stored audio);
and the augmentation itself: seeded per (batch, clip), probability 0.8, one to three distinct operations in a
fixed order, parameters in the plan's ranges, label and id never changed.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_synthetic_*' -v
"""
from __future__ import annotations

import collections
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from labeling_fixtures import relabel_root
from synthetic import augment, layout, mixing
from synthetic_fixtures import tone, write_clip
from training import labels as tlabels
from training import replay as treplay
from training_fixtures import REJECTED_BATCH, LabeledWorld

THROUGH = 10


def make_pool(root: Path, n: int = 40) -> mixing.ClipPool:
    rng = np.random.default_rng(7)
    clips = []
    for i in range(n):
        dur = float(rng.uniform(0.2, 0.9))
        path = write_clip(root / "clips" / f"c{i:03d}.flac", tone(dur, 200 + 10 * i))
        import hashlib
        clips.append({"id": f"pilot-{i:06d}", "audio_filepath": str(path), "duration": len(tone(dur)) / 16000,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "label": f"Synthetic sentence {i}.",
                      "domain": layout.DOMAINS[i % 4], "engine": "syn-tts-a", "voice": "p225"})
    return mixing.ClipPool(clips, identity="fixture")


def make_aug_pool(root: Path) -> augment.Pool:
    rng = np.random.default_rng(3)
    noise = [str(write_clip(root / "noise" / f"n{i}.flac", (0.1 * rng.standard_normal(16000 * 2)).astype(np.float32)))
             for i in range(3)]
    rirs = []
    for i in range(3):
        r = np.zeros(4000, np.float32)
        r[100] = 1.0
        r[101:] = (0.3 * np.exp(-np.arange(3899) / 400.0) * rng.standard_normal(3899)).astype(np.float32)
        rirs.append(str(write_clip(root / "rir" / f"r{i}.flac", np.clip(r, -1, 1))))
    return augment.Pool(noise, rirs, identity="fixture")


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.w = LabeledWorld(cls.root)
        cls.replay = treplay.JournalReplay(cls.w.pool, THROUGH)
        cls.store = tlabels.LabelStore.open(cls.w.labels_dir, cls.w.pool, label_set="T*", through_batch=THROUGH)
        cls.pool = make_pool(cls.root)
        cls.aug_pool = make_aug_pool(cls.root)
        cls.real = [[(i, d) for i, _, d in cls.replay.batches[b]] for b in range(THROUGH)]
        cls.keep = staticmethod(lambda i: cls.store.text(i) is not None)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def sources(self):
        ts = mixing.SyntheticSource(self.pool, self.real, self.keep, seed=11, augment=augment.Augmenter(self.aug_pool, 5))
        tsn = mixing.SyntheticSource(self.pool, self.real, self.keep, seed=11, augment=None)
        return ts, tsn

    def batches(self, extra=None, start=0, stop=THROUGH):
        b = treplay.ReplayBatcher(self.replay, self.store, start=start, stop=stop, extra=extra)
        try:
            return [b.next(timeout=60) for _ in range(start, stop)]
        finally:
            b.close()


class Mixing(Fixture):
    def test_ts_and_tsn_receive_identical_clips_in_identical_batches(self):
        ts, tsn = self.sources()
        self.assertEqual(ts.batches, tsn.batches)
        a, b = self.batches(ts), self.batches(tsn)
        for x, y in zip(a, b):
            self.assertEqual(x["ids"], y["ids"])
            self.assertEqual(x["labels"], y["labels"])
            self.assertEqual(x["extra_rows"], y["extra_rows"])
        self.assertGreater(sum(x["extra_rows"] for x in a), 0)

    def test_the_real_batch_sequence_is_unchanged(self):
        ts, _ = self.sources()
        plain, mixed = self.batches(), self.batches(ts)
        for k, (p, m) in enumerate(zip(plain, mixed)):
            n = len(p["ids"])
            self.assertEqual(m["index"], p["index"])
            self.assertEqual(m["ids"][:n], p["ids"])
            self.assertEqual(m["labels"][:n], p["labels"])
            self.assertTrue(torch.equal(m["audio"][:n, :p["audio"].shape[1]], p["audio"]))
            self.assertEqual(m["ids"][n:], [f"syn:{c}" for c in ts.clip_ids(k)])
            self.assertTrue(all(s == "synthetic" for s in m["sources"][n:]))

    def test_the_synthetic_share_stays_at_about_ten_percent(self):
        ts, _ = self.sources()
        longest = max(c["duration"] for c in self.pool.clips)
        real_total = syn_total = 0.0
        for b in range(THROUGH):
            real, syn = ts.real_seconds[b], ts.synthetic_seconds(b)
            real_total += real
            syn_total += syn
            self.assertLessEqual(abs(syn_total - 0.10 * real_total), longest / 2 + 1e-9, b)
            if real == 0:
                self.assertEqual(syn, 0.0)
        self.assertEqual(ts.real_seconds[REJECTED_BATCH], 0.0)
        self.assertEqual(ts.batches[REJECTED_BATCH], [])

    def test_augmentation_never_changes_labels_and_is_off_in_tsn(self):
        ts, tsn = self.sources()
        self.assertIsNone(tsn.transform)
        labels = {f"syn:{c['id']}": c["label"] for c in self.pool.clips}
        differs = 0
        for x, y in zip(self.batches(ts), self.batches(tsn)):
            n = len(x["ids"]) - x["extra_rows"]
            for row in range(n, len(x["ids"])):
                cid = x["ids"][row]
                self.assertEqual(x["labels"][row], labels[cid])
                self.assertEqual(y["labels"][row], labels[cid])
                stored = torch.from_numpy(self.pool.read(cid[4:]))
                self.assertTrue(torch.equal(y["audio"][row, :len(stored)], stored))
                self.assertEqual(int(y["audio_len"][row]), len(stored))
                L = min(int(x["audio_len"][row]), len(stored))
                differs += int(int(x["audio_len"][row]) != len(stored) or not torch.equal(x["audio"][row, :L], stored[:L]))
        self.assertGreater(differs, 0)

    def test_the_plan_is_fixed_and_a_resumed_run_gets_the_same_batches(self):
        ts, _ = self.sources()
        again = mixing.SyntheticSource(self.pool, self.real, self.keep, seed=11)
        self.assertEqual(again.batches, ts.batches)
        self.assertEqual(again.identity["plan_sha256"], ts.identity["plan_sha256"])
        full, resumed = self.batches(ts), self.batches(ts, start=5)
        for x, y in zip(full[5:], resumed):
            self.assertEqual(x["ids"], y["ids"])
            self.assertTrue(torch.equal(x["audio"], y["audio"]))
        other = mixing.SyntheticSource(self.pool, self.real, self.keep, seed=12)
        self.assertNotEqual(other.batches, ts.batches)

    def test_arm_factory_gives_identical_plans_and_augments_only_ts(self):
        ts = mixing.source_for("TS", self.replay, self.store, THROUGH, pool=self.pool, augment_pool=self.aug_pool)
        tsn = mixing.source_for("TSn", self.replay, self.store, THROUGH, pool=self.pool, augment_pool=self.aug_pool)
        self.assertEqual(ts.batches, tsn.batches)
        self.assertIsNotNone(ts.transform)
        self.assertIsNone(tsn.transform)
        self.assertEqual(ts.real_seconds[REJECTED_BATCH], 0.0)
        with self.assertRaises(ValueError):
            mixing.source_for("T", self.replay, self.store, THROUGH, pool=self.pool)

    def test_items_refuse_real_rows_other_than_the_planned_ones(self):
        ts, _ = self.sources()
        items = self.replay.read(1)
        with self.assertRaises(RuntimeError):
            ts.items(1, items[:-1])
        with self.assertRaises(RuntimeError):
            ts.items(2, items)


class Plan(unittest.TestCase):
    def test_share_error_diffusion_and_clip_order_over_many_batches(self):
        rng = np.random.default_rng(0)
        real = [float(x) for x in rng.uniform(500, 600, 3000)]
        real[10] = 0.0
        clips = [float(x) for x in rng.uniform(1, 30, 500)]
        p = mixing.plan(real, clips, seed=3)
        self.assertEqual(p[10], [])
        tot_r = tot_s = 0.0
        for r, ks in zip(real, p):
            s = sum(clips[k] for k in ks)
            tot_r, tot_s = tot_r + r, tot_s + s
            self.assertLessEqual(abs(tot_s - 0.1 * tot_r), 15.0 + 1e-9)
            if r:
                self.assertLessEqual(abs(s - 0.1 * r), 45.0)
        flat = [k for ks in p for k in ks]
        self.assertEqual(sorted(flat[:500]), list(range(500)))            # each clip once before any repeat
        counts = collections.Counter(flat)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertAlmostEqual(tot_s / tot_r, 0.10, delta=0.001)


class Augmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pool = make_aug_pool(Path(cls.tmp.name))
        cls.aug = augment.Augmenter(cls.pool, seed=9)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def item(self, i=0):
        return {"audio": tone(2.0, 300), "duration": 2.0, "id": f"syn:pilot-{i:06d}", "source": "synthetic",
                "label": "The enthalpy rose sharply."}

    def test_labels_ids_and_source_never_change_and_duration_follows_the_audio(self):
        for idx in range(60):
            it = self.item(idx)
            out = self.aug(it, idx)
            self.assertEqual(out["label"], it["label"])
            self.assertEqual(out["id"], it["id"])
            self.assertEqual(out["source"], it["source"])
            self.assertAlmostEqual(out["duration"], len(out["audio"]) / 16000)
            self.assertEqual(out["audio"].dtype, np.float32)
            self.assertLessEqual(float(np.max(np.abs(out["audio"]))), 1.0)
            self.assertTrue(np.array_equal(it["audio"], tone(2.0, 300)))   # input untouched

    def test_seeded_per_batch_and_clip(self):
        a, b = self.aug(self.item(), 4), self.aug(self.item(), 4)
        self.assertTrue(np.array_equal(a["audio"], b["audio"]))
        self.assertEqual(a["augment"], b["augment"])
        draws = {repr(self.aug.draw(augment.rng_for(9, k, "syn:pilot-000000"))) for k in range(30)}
        self.assertGreater(len(draws), 10)

    def test_probability_count_order_and_ranges(self):
        n, applied, sizes = 4000, 0, collections.Counter()
        for k in range(n):
            plan = self.aug.draw(augment.rng_for(1, k, "x"))
            if plan:
                applied += 1
                sizes[len(plan)] += 1
                ops = [op for op, _ in plan]
                self.assertEqual(len(set(ops)), len(ops))
                self.assertEqual(ops, sorted(ops, key=augment.OPS.index))
                for op, p in plan:
                    if op == "speed":
                        self.assertTrue(0.9 <= p["factor"] <= 1.1)
                    elif op == "pitch":
                        self.assertTrue(-2 <= p["semitones"] <= 2)
                    elif op == "noise":
                        self.assertTrue(5 <= p["snr_db"] <= 30)
                    elif op == "channel":
                        self.assertIn(p["kind"], ("telephone", "codec"))
        self.assertAlmostEqual(applied / n, 0.8, delta=0.03)
        self.assertEqual(set(sizes), {1, 2, 3})
        for k in (1, 2, 3):
            self.assertAlmostEqual(sizes[k] / applied, 1 / 3, delta=0.04)

    def test_each_operation(self):
        x = tone(2.0, 300)
        self.assertAlmostEqual(len(augment.speed(x, 1.1)), len(x) / 1.1, delta=2)
        self.assertAlmostEqual(len(augment.speed(x, 0.9)), len(x) / 0.9, delta=2)
        self.assertEqual(len(augment.pitch(x, 2.0)), len(x))
        noise = self.pool.read(self.pool.noise_paths[0])
        y = augment.add_noise(x, noise, 10.0, 123)
        snr = 10 * np.log10(np.mean(x.astype(np.float64) ** 2) / np.mean((y - x).astype(np.float64) ** 2))
        self.assertAlmostEqual(snr, 10.0, delta=0.2)
        r = augment.reverb(x, self.pool.read(self.pool.rir_paths[0]))
        self.assertEqual(len(r), len(x))
        self.assertAlmostEqual(augment.rms(r), augment.rms(x), delta=1e-3)
        delta = np.zeros(10, np.float32)
        delta[3] = 1.0
        self.assertTrue(np.allclose(augment.reverb(x, delta), x, atol=1e-5))
        self.assertEqual(len(augment.telephone(x)), len(x))
        self.assertEqual(len(augment.codec(x, 0.95)), len(x))
        g = augment.gain(x, 15.0)
        self.assertLessEqual(float(np.max(np.abs(g))), 1.0)

    def test_p_apply_zero_returns_the_stored_audio(self):
        off = augment.Augmenter(self.pool, seed=9, p_apply=0.0)
        it = self.item()
        out = off(it, 3)
        self.assertTrue(np.array_equal(out["audio"], it["audio"]))
        self.assertEqual(out["augment"], [])


if __name__ == "__main__":
    unittest.main()
