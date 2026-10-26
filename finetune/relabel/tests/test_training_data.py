"""Contract tests: the label lookup and the journal replay (decoder-only plan "Data", b0prime plan
"No online teacher" and "Data"), on real C1 shards and real C2 label records (training_fixtures)."""
from __future__ import annotations

import json
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
from training import labels as tlabels
from training import replay as treplay
from training_fixtures import PER_SHARD, REJECTED_BATCH, TOTAL, LabeledWorld

THROUGH = 10     # batches 0-9: C1 shards 0-2 (shard 2 covers 8-11)


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.w = LabeledWorld(cls.root)
        cls.fresh = cls.w.stream_batches(TOTAL)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def store(self, label_set="T*", through=THROUGH, run_dir=None, pool=None):
        return tlabels.LabelStore.open(run_dir or self.w.labels_dir, pool or self.w.pool, label_set=label_set,
                                       through_batch=through)

    def copy_world(self, name):
        """A copy of the C1 and C2 directories, to damage."""
        d = self.root / "copies" / name
        pool, lab = d / "pool", d / "labels"
        shutil.copytree(self.w.pool, pool)
        shutil.copytree(self.w.labels_dir, lab)
        run = json.loads((lab / "run.json").read_text())
        run["pool_dir"] = str(pool.relative_to(self.root))
        (lab / "run.json").write_text(json.dumps(run))
        return pool, lab


class Labels(Fixture):
    def test_t_star_text_for_kept_ids_and_none_for_rejected(self):
        s = self.store("T*")
        for b in range(THROUGH):
            for i in self.w.batch_ids(b):
                rec = self.w.records[i]
                if rec["disposition"] == "kept":
                    self.assertEqual(s.text(i), rec["text"])
                    self.assertTrue(s.kept(i))
                else:
                    self.assertIsNone(s.text(i))
                    self.assertFalse(s.kept(i))
        self.assertTrue(any(not s.kept(i) for i in self.w.batch_ids(REJECTED_BATCH)))

    def test_b0_labels_under_the_t_star_mask(self):
        t, b0 = self.store("T*"), self.store("B0")
        for b in range(THROUGH):
            for i in self.w.batch_ids(b):
                rec = self.w.records[i]
                self.assertEqual(b0.kept(i), t.kept(i))
                self.assertEqual(b0.text(i), rec["b0_raw"] if rec["disposition"] == "kept" else None)
                if rec["disposition"] == "kept":
                    self.assertNotEqual(b0.text(i), t.text(i))   # really the other label

    def test_control_and_t_star_arm_see_the_same_utterances(self):
        rt = treplay.JournalReplay(self.w.pool, THROUGH)
        t, b0 = self.store("T*"), self.store("B0")
        for b in range(THROUGH):
            ids = rt.batch_ids(b)
            self.assertEqual([t.text(i) is not None for i in ids], [b0.text(i) is not None for i in ids])
        self.assertEqual(treplay.plan(rt, t, THROUGH), treplay.plan(rt, b0, THROUGH))

    def test_unknown_id_is_an_error(self):
        with self.assertRaises(KeyError):
            self.store().text("nope:0")

    def test_needs_every_label_shard_of_the_range(self):
        pool, lab = self.copy_world("missing-label-shard")
        (lab / "shards" / f"{shards.shard_name(2)}.labels.parquet").unlink()
        self.store(run_dir=lab, pool=pool, through=8)          # shards 0-1 suffice
        with self.assertRaises(RuntimeError):
            self.store(run_dir=lab, pool=pool, through=9)      # batch 8 is in shard 2

    def test_refuses_a_label_shard_that_differs_from_its_meta(self):
        pool, lab = self.copy_world("bad-label-shard")
        p = lab / "shards" / f"{shards.shard_name(1)}.labels.parquet"
        p.write_bytes(p.read_bytes() + b"x")
        with self.assertRaises(RuntimeError):
            self.store(run_dir=lab, pool=pool)

    def test_refuses_labels_of_another_pool(self):
        pool, _ = self.copy_world("other-pool")
        with self.assertRaises(RuntimeError):
            self.store(pool=pool)

    def test_identity_names_the_label_set_and_shards(self):
        a, b = self.store("T*").identity, self.store("B0").identity
        self.assertEqual((a["label_set"], a["column"], b["column"]), ("T*", "text", "b0_raw"))
        self.assertEqual(a["shards"], b["shards"])
        self.assertNotEqual(a["sha256"], b["sha256"])


class Replay(Fixture):
    def test_batches_are_the_labeled_sequence(self):
        r = treplay.JournalReplay(self.w.pool, TOTAL)
        for b in range(TOTAL):
            got = r.read(b)
            want = self.fresh[b]
            self.assertEqual([x["id"] for x in got], [x["id"] for x in want])
            self.assertEqual([x["id"] for x in got], self.w.batch_ids(b))
            self.assertEqual([x["source"] for x in got], [x["source"] for x in want])
            for g, s in zip(got, want):
                self.assertEqual(g["audio"].dtype, np.float32)
                self.assertTrue(np.array_equal(g["audio"], s["audio"]), f"batch {b} {g['id']}")

    def test_collated_batches_equal_train_batcher_collation(self):
        r = treplay.JournalReplay(self.w.pool, TOTAL)
        b = treplay.ReplayBatcher(r, self.store(through=TOTAL), start=0, stop=TOTAL)
        try:
            for k in range(TOTAL):
                got = b.next(timeout=30)
                want = train.collate(self.fresh[k])
                self.assertEqual(got["index"], k)
                self.assertEqual(got["position_after"], {"batch": k + 1})
                self.assertEqual(got["ids"], want["ids"])
                self.assertTrue(torch.equal(got["audio"], want["audio"]))
                self.assertTrue(torch.equal(got["audio_len"], want["audio_len"]))
                self.assertEqual(got["seconds"], want["seconds"])
        finally:
            b.close()

    def test_start_resumes_exactly_and_stop_ends_the_sequence(self):
        r = treplay.JournalReplay(self.w.pool, THROUGH)
        s = self.store()
        full = treplay.ReplayBatcher(r, s, start=0, stop=THROUGH)
        part = treplay.ReplayBatcher(r, s, start=6, stop=THROUGH)
        try:
            a = [full.next(timeout=30) for _ in range(THROUGH)]
            c = [part.next(timeout=30) for _ in range(THROUGH - 6)]
            self.assertEqual([x["ids"] for x in a[6:]], [x["ids"] for x in c])
            self.assertEqual([x["labels"] for x in a[6:]], [x["labels"] for x in c])
            with self.assertRaises(treplay.ReplayStopped):
                full.next(timeout=30)
        finally:
            full.close()
            part.close()

    def test_labels_travel_with_their_rows(self):
        r = treplay.JournalReplay(self.w.pool, THROUGH)
        s = self.store("B0")
        bt = treplay.ReplayBatcher(r, s, start=0, stop=THROUGH)
        try:
            for k in range(THROUGH):
                got = bt.next(timeout=30)
                self.assertEqual(got["labels"], [s.text(i) for i in got["ids"]])
                if k == REJECTED_BATCH:
                    self.assertTrue(all(x is None for x in got["labels"]))
        finally:
            bt.close()

    def test_plan_counts_consumed_batches_and_steps(self):
        r = treplay.JournalReplay(self.w.pool, THROUGH)
        p = treplay.plan(r, self.store(), THROUGH)
        self.assertEqual(p["stop_batch"], THROUGH)
        self.assertEqual(p["empty_batches"], [REJECTED_BATCH])
        self.assertEqual(p["steps"], THROUGH - 1)
        self.assertEqual(p["occurrences"], sum(len(self.w.batch_ids(b)) for b in range(THROUGH)))
        self.assertLess(p["kept_occurrences"], p["occurrences"])

    def test_refuses_a_range_beyond_the_committed_shards(self):
        with self.assertRaises(RuntimeError):
            treplay.JournalReplay(self.w.pool, TOTAL + 1)

    def test_refuses_a_journal_that_differs_from_its_meta(self):
        pool, _ = self.copy_world("bad-journal")
        p = shards.paths(pool, 1)["journal"]
        p.write_bytes(p.read_bytes()[:-3] + b"abc")
        with self.assertRaises(RuntimeError):
            treplay.JournalReplay(pool, THROUGH)

    def test_refuses_audio_whose_samples_differ_from_the_shard_hash(self):
        r = treplay.JournalReplay(self.w.pool, PER_SHARD)
        real = shards.decode_flac
        with mock.patch.object(shards, "decode_flac", lambda data: real(data) // 2):
            with self.assertRaises(RuntimeError):
                r.read(0)
        self.assertEqual([x["id"] for x in r.read(0)], self.w.batch_ids(0))

    def test_refuses_an_audio_file_whose_size_differs_from_its_meta(self):
        pool, _ = self.copy_world("bad-audio")
        p = shards.paths(pool, 0)["audio"]
        p.write_bytes(p.read_bytes() + b"\0")
        with self.assertRaises(RuntimeError):
            treplay.JournalReplay(pool, PER_SHARD)

    def test_extra_clip_source_is_appended_with_its_labels_and_transform(self):
        r = treplay.JournalReplay(self.w.pool, THROUGH)
        s = self.store()

        class Extra:
            def items(self, index, real):
                pcm = np.full(16000, index + 1, dtype=np.float32) / 32768.0
                return [{"audio": pcm, "duration": 1.0, "id": f"syn:{index}", "source": "syn", "label": f"syn {index}"}]

            def transform(self, item, index):
                return {**item, "audio": item["audio"] * 2}

        plain = treplay.ReplayBatcher(r, s, start=0, stop=4)
        extra = treplay.ReplayBatcher(r, s, start=0, stop=4, extra=Extra())
        try:
            for k in range(4):
                a, e = plain.next(timeout=30), extra.next(timeout=30)
                n = len(a["ids"])
                self.assertEqual(e["ids"], a["ids"] + [f"syn:{k}"])
                self.assertEqual(e["labels"], a["labels"] + [f"syn {k}"])
                self.assertEqual(e["extra_rows"], 1)
                self.assertEqual(e["position_after"], a["position_after"])
                self.assertTrue(torch.equal(e["audio"][:n, :a["audio"].shape[1]], a["audio"]))
                self.assertTrue(torch.equal(e["audio"][n, :16000], torch.full((16000,), 2 * (k + 1) / 32768.0)))
        finally:
            plain.close()
            extra.close()


if __name__ == "__main__":
    unittest.main()
