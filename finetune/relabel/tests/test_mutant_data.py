"""mutant_data contracts (mutant plan "Training data" and "Training protocol"): which clips are selected (sources,
human references, denylist), tokenizer coverage, the held-out slice by recording, the batch plan (pools, cut, epochs,
pilot positions, identity), the audio store (C1's bytes, checked decode, resume) and smoke-path isolation.

CPU only, on a fake C1 store written with labeling/shards.py's AudioWriter and a stand-in tokenizer.

Run from finetune/relabel:  ./python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

import mutant_data as md  # noqa: E402
import relabel_paths  # noqa: E402
from labeling import shards as lshards  # noqa: E402

TMP = relabel_paths.ARTIFACTS / "tmp" / "tests" / "mutant-data"


class StandInTokenizer:
    """Characters a-z, space and apostrophe are covered (one id each); anything else is the unknown id 0."""
    alphabet = " abcdefghijklmnopqrstuvwxyz'"
    unk_id = 0

    def text_to_ids(self, text: str) -> list[int]:
        return [self.alphabet.index(ch) + 1 if ch in self.alphabet else 0 for ch in text.lower()]


def ls_id(spk: int, ch: int, utt: int) -> str:
    return f"librispeech:{spk}-{ch}-{utt:04d}"


def vox_id(item: int, seg: int) -> str:
    return f"voxpopuli:20180528-0900-PLENARY-{item}-en_20180528-21:13:15_{seg}"


def ami_id(meeting: str, seg: int) -> str:
    return f"ami:AMI_{meeting}_H0{seg % 4}_MEE00{seg % 4}_{seg:07d}_{seg + 50:07d}"


def row(item_id: str, source: str, seconds: float = 1.5, kind: str | None = "human", text: str | None = "a b",
        seed: int = 0) -> dict:
    rng = np.random.default_rng(abs(hash(item_id)) % 2**32 + seed)
    pcm = (rng.standard_normal(int(seconds * 16000)) * 2000).astype(np.int16)
    return {"id": item_id, "source": source, "duration": len(pcm) / 16000, "samples": len(pcm),
            "reference": text if kind else None, "reference_kind": kind, "pcm_sha256": lshards.pcm_sha256(pcm),
            "flac": lshards.encode_flac(pcm)}


def fake_c1(directory: Path, shards: list[list[dict]]) -> Path:
    for k, rows in enumerate(shards):
        w = lshards.AudioWriter(lshards.paths(directory, k)["audio"])
        for r in rows:
            w.add(r)
        w.commit()
    return directory


def clip(i: int, source: str, recording: str, shard: int, row_: int, samples: int, split: str = "train") -> dict:
    return {"id": f"{source}:{i}", "source": source, "recording": recording, "shard": shard, "row": row_,
            "samples": samples, "duration": samples / 16000, "split": split}


class TestRecordings(unittest.TestCase):
    def test_recording_of_each_source(self):
        self.assertEqual(md.recording_of("librispeech", ls_id(103, 1240, 5)), "103-1240")
        self.assertEqual(md.recording_of("voxpopuli", vox_id(22, 6)), "20180528-0900-PLENARY-22")
        self.assertEqual(md.recording_of("voxpopuli", "voxpopuli:20200212-0900-PLENARY-en_20200212-13:43:16_1"),
                         "20200212-0900-PLENARY")
        self.assertEqual(md.recording_of("voxpopuli", "voxpopuli:20180911-0900-PLENARY-witholdRO-en_20180911-16:56:33_7"),
                         "20180911-0900-PLENARY-witholdRO")
        self.assertEqual(md.recording_of("ami", "ami:AMI_IN1002_H02_MIO050_0242908_0243104"), "IN1002")
        self.assertEqual(md.recording_of("ami", "ami:AMI_TS3009a_H00_MTD033PM_0103448_0104838"), "TS3009a")
        self.assertEqual(md.recording_of("ami", "ami:AMI_TS3009a_H03_MTD036UID_0000001_0000100"), "TS3009a")

    def test_unreadable_id_is_an_error(self):
        with self.assertRaises(md.DataError):
            md.recording_of("librispeech", "librispeech:abc")
        with self.assertRaises(md.DataError):
            md.recording_of("yodas", "yodas:xyz")


class TestSelect(unittest.TestCase):
    def rows(self):
        return [
            {"id": ls_id(1, 2, 0), "source": "librispeech", "reference": "x", "reference_kind": "human"},
            {"id": vox_id(3, 0), "source": "voxpopuli", "reference": "x", "reference_kind": "human"},
            {"id": "ami:AMI_ES2002a_H00_X_0000001_0000002", "source": "ami", "reference": "x", "reference_kind": "human"},
            {"id": "yodas:abc-1", "source": "yodas", "reference": "x", "reference_kind": "pseudo"},
            {"id": "peoples_speech:q", "source": "peoples_speech", "reference": "x", "reference_kind": "human"},
            {"id": ls_id(1, 2, 1), "source": "librispeech", "reference": "x", "reference_kind": "pseudo"},
            {"id": ls_id(5, 6, 0), "source": "librispeech", "reference": None, "reference_kind": None},
        ]

    def test_sources_and_human_flag(self):
        out, counts = md.select_rows(self.rows(), {})
        self.assertEqual([r["id"] for r in out], [ls_id(1, 2, 0), vox_id(3, 0), "ami:AMI_ES2002a_H00_X_0000001_0000002"])
        self.assertEqual(counts["librispeech"], {"rows": 3, "human": 1, "denylisted": 0, "selected": 1})
        self.assertEqual(out[0]["recording"], "1-2")

    def test_denylist_removes_recordings(self):
        out, counts = md.select_rows(self.rows(), {"voxpopuli": {"20180528-0900-PLENARY-3"}, "yodas": {"abc"}})
        self.assertNotIn(vox_id(3, 0), [r["id"] for r in out])
        self.assertEqual(counts["voxpopuli"]["denylisted"], 1)

    def test_duplicate_id_is_refused(self):
        rows = self.rows()[:1] * 2
        with self.assertRaises(md.DataError):
            md.select_rows(rows, {})

    def test_coverage(self):
        tok = StandInTokenizer()
        ids, ok = md.coverage(tok, "it's fine")
        self.assertTrue(ok)
        self.assertEqual(len(ids), len("it's fine"))
        self.assertFalse(md.coverage(tok, "café")[1])
        self.assertFalse(md.coverage(tok, "20 €")[1])


class TestHeldout(unittest.TestCase):
    def clips(self):
        out, rng = [], np.random.default_rng(1)
        for s, recs in (("librispeech", 300), ("voxpopuli", 200), ("ami", 40)):
            for r in range(recs):
                for j in range(int(rng.integers(1, 30))):
                    out.append({"id": f"{s}:{r}:{j}", "source": s, "recording": f"{s}-rec{r}"})
        return out

    def test_about_one_percent_per_source_by_whole_recordings(self):
        clips = self.clips()
        held = md.heldout_recordings(clips)
        for s in md.SOURCES:
            per = [c for c in clips if c["source"] == s]
            n_held = sum(c["recording"] in held[s] for c in per)
            target = round(0.01 * len(per))
            largest = max(sum(c["recording"] == r for c in per) for r in {c["recording"] for c in per})
            self.assertGreaterEqual(n_held, target)
            self.assertLess(n_held, target + largest)
            # whole recordings: every clip of a held-out recording is held out
            self.assertTrue(all(c["recording"] in held[s] for c in per if c["recording"] in held[s]))
        self.assertGreater(len(held["ami"]), 0)

    def test_deterministic_and_salted(self):
        clips = self.clips()
        self.assertEqual(md.heldout_recordings(clips), md.heldout_recordings(list(reversed(clips))))
        self.assertNotEqual(md.heldout_recordings(clips), md.heldout_recordings(clips, salt="another salt"))


class TestPlan(unittest.TestCase):
    def train(self, n=600, shards=10, seed=2):
        rng = np.random.default_rng(seed)
        out = []
        for i in range(n):
            out.append(clip(i, md.SOURCES[i % 3], f"r{i % 50}", int(rng.integers(0, shards)), i,
                            int(rng.integers(16000, 30 * 16000))))
        return out

    def test_every_clip_once_pools_and_budget(self):
        train = self.train()
        plan = md.build_plan(train, n_shards=10, shards_per_pool=3, budget=60 * 16000)
        self.assertEqual(plan.n_pools, 4)
        seen = np.concatenate([plan.batch(b) for b in range(plan.n_batches)])
        self.assertEqual(sorted(seen.tolist()), list(range(len(train))))
        shard = {c["id"]: c["shard"] for c in train}
        for b in range(plan.n_batches):
            seqs = plan.batch(b)
            p = plan.pool_of_batch(b)
            self.assertTrue(all(shard[plan.ids[s]] // 3 == p for s in seqs))
            samples = plan.samples[seqs]
            if len(seqs) > 1:
                self.assertLessEqual(len(seqs) * samples.max(), 60 * 16000)
        # sequence = C1 order
        keys = [(shard[i], next(c["row"] for c in train if c["id"] == i)) for i in plan.ids]
        self.assertEqual(keys, sorted(keys))

    def test_identity_is_deterministic_and_covers_seed_and_budget(self):
        train = self.train()
        a = md.build_plan(train, 10, 3, 60 * 16000)
        b = md.build_plan(list(reversed(train)), 10, 3, 60 * 16000)
        self.assertEqual(a.sha256, b.sha256)
        self.assertNotEqual(a.sha256, md.build_plan(train, 10, 3, 60 * 16000, seed=7).sha256)
        self.assertNotEqual(a.sha256, md.build_plan(train, 10, 3, 50 * 16000).sha256)

    def test_epochs_serve_every_batch_once_in_a_new_order(self):
        plan = md.build_plan(self.train(), 10, 3, 60 * 16000)
        orders = [plan.epoch_order(e) for e in range(3)]
        for o in orders:
            self.assertEqual(sorted(o.tolist()), list(range(plan.n_batches)))
            pools = [plan.pool_of_batch(int(b)) for b in o]
            runs = [pools[0]] + [q for p, q in zip(pools, pools[1:]) if q != p]
            self.assertEqual(len(runs), len(set(runs)), "each pool's batches are served together")
        self.assertFalse(np.array_equal(orders[0], orders[1]))
        self.assertTrue(np.array_equal(orders[1], plan.epoch_order(1)))

    def test_global_steps_map_to_epochs_and_positions(self):
        plan = md.build_plan(self.train(), 10, 3, 60 * 16000)
        n = plan.n_batches
        self.assertEqual(plan.position(0), (0, 0, int(plan.epoch_order(0)[0])))
        self.assertEqual(plan.position(n + 2), (1, 2, int(plan.epoch_order(1)[2])))
        self.assertEqual(plan.position(2 * n), (2, 0, int(plan.epoch_order(2)[0])))

    def test_save_load_and_tamper(self):
        d = TMP / "plan"
        shutil.rmtree(d, ignore_errors=True)
        plan = md.build_plan(self.train(), 10, 3, 60 * 16000)
        md.save_plan(plan, d)
        back = md.load_plan(d)
        self.assertEqual(back.sha256, plan.sha256)
        self.assertTrue(np.array_equal(back.order, plan.order))
        arrays = dict(np.load(d / "plan-arrays.npz"))
        arrays["order"] = arrays["order"][::-1].copy()
        np.savez(d / "plan-arrays.npz", **arrays)
        with self.assertRaises(md.DataError):
            md.load_plan(d)


class TestLayout(unittest.TestCase):
    def test_smoke_paths_are_under_smoke_only(self):
        prod = md.layout()
        smoke = md.layout(40)
        for p in (smoke.root, smoke.features, smoke.records, smoke.data, smoke.audio, smoke.runs):
            self.assertTrue(p.is_relative_to(relabel_paths.SMOKE))
            self.assertFalse(p.is_relative_to(md.production_root()))
        for p in (prod.data, prod.audio, prod.runs, prod.features, prod.records):
            self.assertTrue(p.is_relative_to(relabel_paths.ARTIFACTS / "encoder-probe"))
            self.assertFalse(p.is_relative_to(relabel_paths.SMOKE))
        self.assertEqual(prod.records, relabel_paths.ARTIFACTS / "encoder-probe" / "decoder")


class TestSelectAndAudio(unittest.TestCase):
    """select() and the audio store on a fake C1 of 4 shards (two pools of 2), with row groups of 64."""

    @classmethod
    def setUpClass(cls):
        shutil.rmtree(TMP / "world", ignore_errors=True)
        rows = []
        for k in range(4):
            shard = []
            for j in range(70):          # more than one 64-row group per shard
                i = 70 * k + j
                kind = j % 7
                if kind == 0:
                    shard.append(row(f"yodas:x{i}-1", "yodas", kind="pseudo"))
                elif kind == 1:
                    shard.append(row(ls_id(10 + i % 5, 100 + i % 3, i), "librispeech", text="café"))
                elif kind == 2:
                    shard.append(row(vox_id(i % 9, i), "voxpopuli", text="the cut"))
                elif kind == 3:
                    shard.append(row(ami_id(f"ES20{i % 6:02d}a", i), "ami", text="yeah okay"))
                else:
                    shard.append(row(ls_id(20 + i % 7, 200 + i % 11, i), "librispeech", seconds=1 + (i % 5),
                                     text="hello world"))
            rows.append(shard)
        cls.c1 = fake_c1(TMP / "world" / "c1", rows)
        cls.deny = TMP / "world" / "relabel"
        (cls.deny / "pool").mkdir(parents=True, exist_ok=True)
        (cls.deny / "pool" / "denylist.json").write_text(json.dumps({"recordings": {"yodas": ["x1"]}}))
        cls.lay = md.Layout(TMP / "world" / "mutant", TMP / "world" / "features", TMP / "world" / "decoder", None)
        with mock.patch.object(md.relabel_paths, "ARTIFACTS", cls.deny), \
                mock.patch.object(md, "SHARDS_PER_POOL", 2), mock.patch.object(md.relabel_paths, "require_mount"):
            cls.summary = md.select(cls.lay, tokenizer=StandInTokenizer(), tokenizer_sha256="stand-in", c1=cls.c1,
                                    n_shards=4, hygiene=False)
        cls.data = md.Data.load(cls.lay)

    def test_selection_counts_and_coverage(self):
        s = self.summary
        clips = self.data.clips
        self.assertTrue(all(c["source"] in md.SOURCES for c in clips.values()))
        self.assertFalse(any(c["text"] == "café" for c in clips.values()))
        self.assertEqual(s["uncovered"]["count"], sum(1 for k in range(4) for j in range(70) if j % 7 == 1))
        self.assertEqual(s["counts"]["librispeech"]["uncovered"], s["uncovered"]["count"])
        self.assertTrue(all(c["tokens"] == StandInTokenizer().text_to_ids(c["text"]) for c in clips.values()))
        self.assertEqual(s["denylist"]["recordings_for_these_sources"], {"librispeech": 0, "voxpopuli": 0, "ami": 0})

    def test_heldout_is_disjoint_from_the_plan(self):
        held = set(self.data.heldout_ids())
        self.assertTrue(held)
        self.assertFalse(held & set(self.data.plan.ids))
        self.assertEqual(held | set(self.data.plan.ids), set(self.data.clips))

    def test_reselect_with_another_plan_is_refused(self):
        with mock.patch.object(md.relabel_paths, "ARTIFACTS", self.deny), \
                mock.patch.object(md, "SHARDS_PER_POOL", 1), mock.patch.object(md.relabel_paths, "require_mount"):
            with self.assertRaises(md.DataError):
                md.select(self.lay, tokenizer=StandInTokenizer(), tokenizer_sha256="x", c1=self.c1, n_shards=4,
                          hygiene=False)

    def test_denylist_entry_for_a_selected_source_stops_select(self):
        (self.deny / "pool" / "denylist.json").write_text(json.dumps({"recordings": {"ami": ["ES2001a"]}}))
        try:
            with mock.patch.object(md.relabel_paths, "ARTIFACTS", self.deny), \
                    mock.patch.object(md.relabel_paths, "require_mount"):
                with self.assertRaises(md.DataError):
                    md.select(md.Layout(TMP / "world" / "other", TMP / "x", TMP / "y", None),
                              tokenizer=StandInTokenizer(), c1=self.c1, n_shards=4, hygiene=False)
        finally:
            (self.deny / "pool" / "denylist.json").write_text(json.dumps({"recordings": {"yodas": ["x1"]}}))

    def test_audio_store_holds_c1_bytes_and_decodes_checked(self):
        with mock.patch.object(md.relabel_paths, "require_mount"):
            md.build_audio(self.lay)
        c1_rows = {r["id"]: r for k in range(4) for r in lshards.read_audio(lshards.paths(self.c1, k)["audio"])}
        for p in range(self.data.plan.n_pools):
            store = md.PoolAudio(self.lay, self.data, p)
            self.assertEqual([e["id"] for e in store.meta["entries"]], self.data.pool_ids(p))
            for i in self.data.pool_ids(p):
                self.assertEqual(store.raw(i), c1_rows[i]["flac"])
                pcm = lshards.decode_flac(c1_rows[i]["flac"])
                self.assertTrue(np.array_equal(store.clip(i), pcm.astype(np.float32) / 32768.0))
        # every selected clip (training and held-out) is in exactly one pool
        stored = [i for p in range(self.data.plan.n_pools) for i in self.data.pool_ids(p)]
        self.assertEqual(sorted(stored), sorted(self.data.clips))

    def test_damaged_audio_is_refused(self):
        with mock.patch.object(md.relabel_paths, "require_mount"):
            md.build_audio(self.lay)
        store = md.PoolAudio(self.lay, self.data, 0)
        i = self.data.pool_ids(0)[0]
        entry = dict(store.entries[i], pcm_sha256="0" * 64)
        with self.assertRaises(md.DataError):
            md.decode_checked(i, store.raw(i), entry)

    def test_audio_pass_resumes_at_the_first_uncommitted_pool(self):
        with mock.patch.object(md.relabel_paths, "require_mount"):
            md.build_audio(self.lay)
            first = md.audio_paths(self.lay, 0)["meta"].stat().st_mtime_ns
            md.audio_paths(self.lay, 1)["meta"].unlink()
            with mock.patch.object(md, "write_pool_audio", wraps=md.write_pool_audio) as spy:
                md.build_audio(self.lay)
        self.assertEqual([c.args[2] for c in spy.call_args_list], [1])
        self.assertEqual(md.audio_paths(self.lay, 0)["meta"].stat().st_mtime_ns, first)


if __name__ == "__main__":
    unittest.main()
