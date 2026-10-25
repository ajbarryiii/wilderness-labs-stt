"""Contract tests: C1, the pool replay (journal, lossless distinct-id audio, shards, resume, chunking)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from labeling import config, replay, shards
from labeling_fixtures import FakeStream, World, relabel_root

TOTAL, PER_SHARD = 14, 4


def files_of(d: Path) -> dict[str, bytes]:
    """Every shard file's content (metas without their timestamps)."""
    out = {}
    for p in sorted((d / "shards").iterdir()):
        if p.suffix == ".json":
            m = json.loads(p.read_text())
            m.pop("created", None)
            out[p.name] = json.dumps(m, sort_keys=True).encode()
        else:
            out[p.name] = json.dumps(shards.read_journal(p) if "journal" in p.name else shards.read_audio(p),
                                     sort_keys=True, default=lambda b: b.hex()).encode()
    return out


class Replay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.world = World(cls.root)
        cls.pin = cls.world.pinned(workers=1, batch_seconds=10.0)
        make = lambda: config.make_stream(cls.pin, 0)  # noqa: E731
        cls.full = cls.root / "full"
        cls.status = replay.replay(cls.pin, cls.full, total_batches=TOTAL, shard_batches=PER_SHARD,
                                   stream_factory=make)
        cls.fresh = replay.journal_of(cls.pin, TOTAL, stream_factory=make)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def factory(self):
        return config.make_stream(self.pin, 0)

    def journal(self, d):
        return [r for k in shards.committed(d) for r in shards.read_journal(shards.paths(d, k)["journal"])]

    def test_complete_with_manifest(self):
        self.assertTrue(self.status["complete"])
        self.assertEqual(shards.committed(self.full), [0, 1, 2, 3])
        m = json.loads((self.full / "manifest.json").read_text())
        self.assertEqual(m["batches"], TOTAL)
        self.assertEqual([s["batches"] for s in m["shard_files"]], [[0, 4], [4, 8], [8, 12], [12, 14]])
        for k in shards.committed(self.full):
            shards.verify_shard(self.full, k)

    def test_journal_is_every_occurrence_in_consumption_order(self):
        keys = ("batch", "pos", "id", "source", "duration", "first")
        self.assertEqual(self.journal(self.full), [{k: r[k] for k in keys} for r in self.fresh])
        self.assertEqual(sorted({r["batch"] for r in self.fresh}), list(range(TOTAL)))

    def test_one_lossless_clip_per_distinct_id(self):
        firsts = {r["id"]: r for r in self.fresh if r["first"]}
        rows = {}
        for k in shards.committed(self.full):
            meta = shards.load_meta(self.full, k)
            for r in shards.read_audio(shards.paths(self.full, k)["audio"]):
                self.assertNotIn(r["id"], rows)
                rows[r["id"]] = r
                self.assertTrue(meta["batches"][0] <= firsts[r["id"]]["batch"] < meta["batches"][1])
        self.assertEqual(set(rows), set(firsts))
        for i, r in rows.items():
            pcm = shards.decode_flac(r["flac"])
            self.assertEqual(shards.pcm_sha256(pcm), firsts[i]["pcm_sha256"])
            self.assertEqual(r["samples"], pcm.size)

    def test_references_and_kinds_are_kept(self):
        src = {}
        for k in shards.committed(self.full):
            for r in shards.read_audio(shards.paths(self.full, k)["audio"]):
                src[r["id"]] = r
                native = r["id"] if r["source"] == "med-osce" else r["id"].split(":", 1)[1]
                row = self.world.rows[native]
                if r["source"].startswith("yodas"):
                    self.assertEqual((r["reference"], r["reference_kind"]), (row["text"], "pseudo"))
                else:
                    self.assertEqual(r["reference_kind"], "human")
                    self.assertEqual(r["reference"], row.get("text") or row.get("normalized_text"))
        self.assertTrue({"yodas", "peoples_speech"} <= {r["source"] for r in src.values()})

    def test_resume_in_chunks_gives_identical_output(self):
        d = self.root / "chunks"
        st = replay.replay(self.pin, d, total_batches=TOTAL, shard_batches=PER_SHARD, stream_factory=self.factory,
                           stop_after_shards=1)
        self.assertFalse(st["complete"])
        # an interrupted shard: files without a commit marker are removed on resume
        (d / "shards" / "00001.audio.parquet").write_bytes(b"partial")
        replay.replay(self.pin, d, total_batches=TOTAL, shard_batches=PER_SHARD, stream_factory=self.factory,
                      stop_after_shards=2)
        st = replay.replay(self.pin, d, total_batches=TOTAL, shard_batches=PER_SHARD, stream_factory=self.factory)
        self.assertTrue(st["complete"])
        self.assertEqual(files_of(d), files_of(self.full))

    def test_resume_detects_a_journal_that_is_not_reproduced(self):
        d = self.root / "tamper"
        replay.replay(self.pin, d, total_batches=TOTAL, shard_batches=PER_SHARD, stream_factory=self.factory,
                      stop_after_shards=2)
        p = shards.paths(d, 1)
        rows = shards.read_journal(p["journal"])
        rows[0]["duration"] += 0.5
        sha, size = shards.write_journal(p["journal"], rows)
        meta = shards.load_meta(d, 1)
        meta["files"]["journal"].update(sha256=sha, bytes=size)
        shards.write_meta(p["meta"], meta)
        with self.assertRaises(replay.ResumeMismatch):
            replay.replay(self.pin, d, total_batches=TOTAL, shard_batches=PER_SHARD, stream_factory=self.factory)

    def test_another_configuration_is_refused(self):
        other = self.world.pinned(workers=1, batch_seconds=12.0)
        with self.assertRaises(RuntimeError):
            replay.replay(other, self.full, total_batches=TOTAL, shard_batches=PER_SHARD,
                          stream_factory=lambda: config.make_stream(other, 0))


class Conflicts(unittest.TestCase):
    def test_same_id_with_other_audio_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp, relabel_root(Path(tmp)):
            fake = {"pinned_hash": "x" * 64, "workers": 1, "batch": {"batch_seconds": 20.0, "seed": 5}}
            fs = FakeStream(n_ids=13)
            ids = [fs.item(i)["id"] for i in range(200)]
            at = next(i for i in range(60, 200) if ids[i] in ids[:i] and 1.0 <= fs.item(i)["duration"] <= 30.0)
            st = replay.replay(fake, Path(tmp) / "c", total_batches=10, shard_batches=5,
                               stream_factory=lambda: FakeStream(n_ids=13, conflict_at=at))
            m = json.loads((Path(tmp) / "c" / "manifest.json").read_text())
            self.assertTrue(st["complete"])
            self.assertIn(ids[at], m["id_conflicts"])


class Lossless(unittest.TestCase):
    def test_to_int16_round_trip_and_refusal(self):
        pcm = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
        np.testing.assert_array_equal(shards.to_int16(pcm.astype(np.float32) / 32768.0), pcm)
        with self.assertRaises(ValueError):
            shards.to_int16(np.array([0.1234567], dtype=np.float32))
        np.testing.assert_array_equal(shards.decode_flac(shards.encode_flac(pcm)), pcm)


class ChooseWorkers(unittest.TestCase):
    def r(self, rate, peak):
        return {"audio_hours_per_wall_hour": rate, "host_peak_gb": peak}

    def test_rule(self):
        c = replay.choose_workers({8: self.r(1000, 20), 12: self.r(1150, 25), 16: self.r(1300, 29)})
        self.assertEqual(c["workers"], 16)
        c = replay.choose_workers({8: self.r(1000, 20), 12: self.r(1090, 25), 16: self.r(1150, 29)})
        self.assertEqual(c["workers"], 16)          # 16 is >= 10% faster than 8 (12 was not)
        c = replay.choose_workers({8: self.r(1000, 20), 12: self.r(1200, 25), 16: self.r(1250, 29)})
        self.assertEqual(c["workers"], 12)          # 16 not 10% faster than 12
        c = replay.choose_workers({8: self.r(1000, 20), 12: self.r(1500, 31), 16: self.r(1600, 33)})
        self.assertEqual(c["workers"], 8)           # over the 30G peak
        c = replay.choose_workers({8: self.r(1000, 30.5)})
        self.assertIsNone(c["workers"])
        self.assertTrue(c["stop"])


if __name__ == "__main__":
    unittest.main()
