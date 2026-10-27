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
from labeling import filters, shards
from labeling_fixtures import relabel_root
from training import labels as tlabels
from training import repetition as trep
from training import replay as treplay
from training_fixtures import HALLUCINATED, HALLUCINATED_BATCH, NEAR_MISS, PER_SHARD, REJECTED_BATCH, TOTAL, LabeledWorld

THROUGH = 10     # batches 0-9: C1 shards 0-2 (shard 2 covers 8-11)
_TOKENIZER: dict = {}


def repetition_spec(cache: Path | None, **kw) -> trep.Repetition:
    """Rule "repetition" with the students' tokenizer (extracted once per process from the pinned .nemo)."""
    if "file" not in _TOKENIZER:
        _TOKENIZER["dir"] = tempfile.TemporaryDirectory()
        _TOKENIZER["file"] = filters.b0_tokenizer_file(Path(_TOKENIZER["dir"].name))
    return trep.Repetition(_TOKENIZER["file"], cache, **kw)


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.w = LabeledWorld(cls.root, hallucinate=True)
        cls.fresh = cls.w.stream_batches(TOTAL)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def store(self, label_set="T*", through=THROUGH, run_dir=None, pool=None, **kw):
        return tlabels.LabelStore.open(run_dir or self.w.labels_dir, pool or self.w.pool, label_set=label_set,
                                       through_batch=through, **kw)

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
        spec = repetition_spec(self.root / "rep-same.json")
        for kw in ({}, {"repetition": spec}):
            t, b0 = self.store("T*", **kw), self.store("B0", **kw)
            for b in range(THROUGH):
                ids = rt.batch_ids(b)
                self.assertEqual([t.text(i) is not None for i in ids], [b0.text(i) is not None for i in ids])
            self.assertEqual(treplay.plan(rt, t, THROUGH), treplay.plan(rt, b0, THROUGH))
        # with the rule, both drop the hallucinated id (and nothing else)
        plain, ruled = treplay.plan(rt, self.store("B0"), THROUGH), treplay.plan(rt, b0, THROUGH)
        occurrences = sum(i == self.w.hallucinated for b in range(THROUGH) for i in rt.batch_ids(b))
        self.assertGreater(occurrences, 0)
        self.assertEqual(ruled["kept_occurrences"], plain["kept_occurrences"] - occurrences)
        self.assertFalse(b0.kept(self.w.hallucinated))

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


class Repetition(Fixture):
    """Rule "repetition" v1 (training/repetition.py) in the label store: decided on the T* label as stored, applied
    to every label set as part of T*'s disposition mask, named in the identity, cached per label shard."""

    def cache(self) -> Path:
        return self.root / "rep" / f"{self.id().rsplit('.', 1)[-1]}.json"

    def test_the_fixture_has_repetitions_that_passed_the_filters(self):
        h, n = self.w.records[self.w.hallucinated], self.w.records[self.w.near_miss]
        self.assertEqual((h["disposition"], n["disposition"]), ("kept", "kept"))
        self.assertTrue(h["text"].endswith(HALLUCINATED) and n["text"].endswith(NEAR_MISS))
        spec = repetition_spec(None)
        self.assertEqual(spec.repeats(h["text"]), (32, 1))
        self.assertEqual(spec.repeats(n["text"]), (31, 1))
        self.assertEqual(spec.repeats(h["b0_raw"])[0], 1)
        self.assertIn(self.w.hallucinated, self.w.batch_ids(HALLUCINATED_BATCH))

    def test_a_hallucinated_t_star_label_is_rejected_for_every_label_set(self):
        spec = repetition_spec(self.cache())
        for label_set in ("T*", "B0"):
            plain, ruled = self.store(label_set), self.store(label_set, repetition=spec)
            self.assertTrue(plain.kept(self.w.hallucinated))
            self.assertFalse(ruled.kept(self.w.hallucinated), label_set)
            self.assertIsNone(ruled.text(self.w.hallucinated))
            self.assertTrue(ruled.kept(self.w.near_miss), label_set)              # 31 repeats: below the threshold
            others = [i for b in range(THROUGH) for i in self.w.batch_ids(b) if i != self.w.hallucinated]
            self.assertEqual([ruled.text(i) for i in others], [plain.text(i) for i in others])
            self.assertEqual(ruled.identity["kept"], plain.identity["kept"] - 1)

    def test_the_identity_names_the_rule_and_its_rejected_ids(self):
        spec = repetition_spec(self.cache())
        t, b0, plain = self.store("T*", repetition=spec), self.store("B0", repetition=spec), self.store("T*")
        block = t.identity["repetition"]
        self.assertEqual(block, b0.identity["repetition"])
        self.assertEqual(block, spec.identity([self.w.hallucinated]))
        self.assertEqual((block["name"], block["version"], block["threshold"], block["n"], block["rejected"]),
                         ("repetition", 1, 32, [1, 6], 1))
        self.assertNotIn("repetition", plain.identity)
        self.assertNotEqual(t.identity["sha256"], plain.identity["sha256"])
        # another rule: another identity, with another rejected set (threshold 31) or the same one (n up to 1)
        lower = self.store("T*", repetition=repetition_spec(self.cache().with_suffix(".t31"), threshold=31))
        self.assertEqual(lower.identity["repetition"]["rejected"], 2)
        self.assertNotEqual(lower.identity["repetition"]["rejected_sha256"], block["rejected_sha256"])
        unigram = self.store("T*", repetition=repetition_spec(self.cache().with_suffix(".n1"), n_max=1))
        self.assertEqual(unigram.identity["repetition"]["rejected_sha256"], block["rejected_sha256"])
        self.assertEqual(len({t.identity["sha256"], lower.identity["sha256"], unigram.identity["sha256"]}), 3)
        # only the store's shards count: batches 0-3 (C1 shard 0) hold no rejected id
        self.assertEqual(self.store("T*", through=4, repetition=spec).identity["repetition"]["rejected"], 0)

    def test_decided_on_the_stored_t_star_text_whatever_the_cleaning(self):
        from training import cleaning
        spec = repetition_spec(self.cache())
        layer1 = cleaning.Cleaning.load()
        for label_set in ("T*", "B0"):
            s = self.store(label_set, repetition=spec, cleaning=layer1)
            self.assertFalse(s.kept(self.w.hallucinated))
            self.assertEqual(s.identity["repetition"], self.store(label_set, repetition=spec).identity["repetition"])

    def test_cached_per_label_shard_and_reused(self):
        path = self.cache()
        first = self.store("B0", repetition=repetition_spec(path))
        cached = json.loads(path.read_text())
        self.assertEqual(cached["rule"], repetition_spec(None).rule())
        self.assertEqual(sorted(cached["shards"]), [shards.shard_name(k) for k in range(3)])
        self.assertEqual(cached["shards"][shards.shard_name(1)]["rejected"], [[self.w.hallucinated, 32, 1]])
        with mock.patch.object(trep.Repetition, "evaluate", side_effect=AssertionError("recomputed")):
            again = self.store("T*", repetition=repetition_spec(path))     # the T* store reads the same cache
        self.assertEqual(again.identity["repetition"], first.identity["repetition"])
        self.assertFalse(again.kept(self.w.hallucinated))
        # a wider store computes only the shard it adds
        calls = []
        real = trep.Repetition.evaluate
        with mock.patch.object(trep.Repetition, "evaluate", lambda s, *a: calls.append(a[0]) or real(s, *a)):
            self.store("T*", through=TOTAL, repetition=repetition_spec(path))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(json.loads(path.read_text())["shards"]), 4)

    def test_a_stale_cache_entry_is_recomputed(self):
        path = self.cache()
        want = self.store("T*", repetition=repetition_spec(path)).identity["repetition"]
        cached = json.loads(path.read_text())
        name = shards.shard_name(1)
        cached["shards"][name].update(sha256="0" * 64, rejected=[])        # another shard's result
        path.write_text(json.dumps(cached))
        self.assertEqual(self.store("T*", repetition=repetition_spec(path)).identity["repetition"], want)
        self.assertNotEqual(json.loads(path.read_text())["shards"][name]["sha256"], "0" * 64)
        for change in ({"rule": {**cached["rule"], "threshold": 99}}, {"contract_sha256": "0" * 64}):
            stale = {**json.loads(path.read_text()), **change}
            stale["shards"][name]["rejected"] = []
            path.write_text(json.dumps(stale))
            self.assertEqual(self.store("T*", repetition=repetition_spec(path)).identity["repetition"], want)

    def test_a_cached_rejected_id_must_be_a_kept_record_of_its_shard(self):
        path = self.cache()
        self.store("T*", repetition=repetition_spec(path))
        cached = json.loads(path.read_text())
        rejected_record = sorted(self.w.reject)[0]
        k = next(k for k in range(3) if rejected_record in {i for b in range(k * PER_SHARD, (k + 1) * PER_SHARD)
                                                            for i in self.w.batch_ids(b)})
        cached["shards"][shards.shard_name(k)]["rejected"] = [[rejected_record, 12, 1]]
        path.write_text(json.dumps(cached))
        with self.assertRaises(RuntimeError):
            self.store("T*", repetition=repetition_spec(path))
        cached["shards"][shards.shard_name(k)]["rejected"] = [["nope:0", 12, 1]]
        path.write_text(json.dumps(cached))
        with self.assertRaises(RuntimeError):
            self.store("T*", repetition=repetition_spec(path))

    def test_without_a_cache_file(self):
        s = self.store("T*", repetition=repetition_spec(None))
        self.assertFalse(s.kept(self.w.hallucinated))


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
