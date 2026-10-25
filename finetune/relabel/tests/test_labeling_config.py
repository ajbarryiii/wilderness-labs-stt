"""Contract tests: the labeling stream's configuration (labeling plan Part C "Pool and stream")."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import stream  # finetune/parakeet-ternary
from labeling import config
from labeling_fixtures import World, relabel_root


class Shares(unittest.TestCase):
    def test_dropped_corpus_share_returns_to_the_existing_mix(self):
        s = config.shares({"yodas-med": 30.0, "ps-med": 10.0, "yodas-tech": 50.0},
                          {"med-osce": True, "syn-dialog": False})
        existing = 0.79 + 0.03
        total = sum(stream.SHARES.values())
        for name in config.EXISTING:
            self.assertAlmostEqual(s[name], existing * stream.SHARES[name] / total)
        self.assertAlmostEqual(s["yodas-med"], 0.10 * 30 / 40)
        self.assertAlmostEqual(s["ps-med"], 0.10 * 10 / 40)
        self.assertAlmostEqual(s["yodas-tech"], 0.07)
        self.assertAlmostEqual(s["med-osce"], 0.01)
        self.assertNotIn("syn-dialog", s)
        self.assertAlmostEqual(sum(s.values()), 1.0)

    def test_built_corpus_keeps_its_share(self):
        s = config.shares({"yodas-med": 1.0, "ps-med": 1.0, "yodas-tech": 1.0}, {"med-osce": True, "syn-dialog": True})
        self.assertAlmostEqual(s["syn-dialog"], 0.03)
        self.assertAlmostEqual(sum(s[n] for n in config.EXISTING), 0.79)

    def test_both_corpora_dropped(self):
        s = config.shares({"yodas-med": 1.0, "ps-med": 1.0, "yodas-tech": 1.0}, {})
        self.assertAlmostEqual(sum(s[n] for n in config.EXISTING), 0.83)
        self.assertAlmostEqual(sum(s.values()), 1.0)


class Candidate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ctx = relabel_root(self.root)
        self.ctx.__enter__()

    def tearDown(self):
        self.ctx.__exit__(None, None, None)
        self.tmp.cleanup()

    def sources(self, cand):
        return {s["name"]: s for s in cand["stream"]["sources"]}

    def test_routes_include_exclude_and_local_sources(self):
        w = World(self.root)
        cand = w.candidate()
        src = self.sources(cand)
        self.assertEqual(set(src), {"yodas", "librispeech", "peoples_speech", "voxpopuli", "ami", "yodas-med",
                                    "ps-med", "yodas-tech", "med-osce"})
        excl = dict(map(tuple, src["yodas"]["exclude"]))
        self.assertEqual(excl, {"yA": "denylisted", "yG": "denylisted", "yB": "in_subset", "yD": "in_subset",
                                "yE": "in_subset", "yH": "in_subset"})
        self.assertEqual(dict(map(tuple, src["peoples_speech"]["exclude"])), {"pA": "in_subset", "pB": "denylisted"})
        self.assertEqual(src["yodas-med"]["include"], ["yB", "yH"])
        f0 = w.url("r/y", "1", "data/en000/asr_only/00000000.parquet")
        self.assertEqual([tuple(u) for u in src["yodas-med"]["units"]], [(f0, 1), (f0, 2)])
        self.assertEqual(dict(map(tuple, src["yodas-med"]["exclude"])), {"yA": "denylisted", "yG": "denylisted"})
        self.assertEqual(len(src["ps-med"]["units"]), 2)
        self.assertEqual(src["med-osce"]["kind"], "local")
        self.assertEqual(len(src["med-osce"]["local_rows"]), 5)
        self.assertEqual(src["yodas"]["pseudo_col"], "text")
        self.assertIsNone(src["yodas"]["text_col"])
        self.assertEqual(src["peoples_speech"]["text_col"], "text")
        self.assertAlmostEqual(sum(cand["shares"].values()), 1.0)
        self.assertAlmostEqual(cand["shares"]["med-osce"], 0.01)
        self.assertTrue(cand["final"])

    def test_local_corpus_dropped_or_built_is_required(self):
        w = World(self.root, osce="dropped")
        cand = w.candidate()
        self.assertNotIn("med-osce", self.sources(cand))
        self.assertAlmostEqual(sum(cand["shares"][n] for n in config.EXISTING), 0.83)
        w2root = self.root / "w2"
        w2 = World(w2root, osce="missing")
        with self.assertRaises(RuntimeError):
            config.build_candidate(w2.pool, w2.corpora, source_files=w2.files, mean_seconds={},
                                   hf_sources={}, reserved={"yA"}, url_fn=w2.url)

    def test_non_final_pool_is_refused_unless_provisional(self):
        w = World(self.root, final=False)
        with self.assertRaises(RuntimeError):
            w.candidate()
        cand = w.candidate(allow_provisional=True)
        self.assertFalse(cand["final"])

    def test_reserved_recordings_must_be_on_the_denylist(self):
        w = World(self.root)
        with self.assertRaises(RuntimeError):
            w.candidate(reserved={"yA", "yZ"})

    def test_denylist_entry_for_a_source_without_recording_keys_is_refused(self):
        w = World(self.root, deny_extra={"librispeech": ["lsR"]})
        with self.assertRaises(RuntimeError):
            w.candidate()

    def test_pinned_hash_covers_workers_seed_and_content(self):
        w = World(self.root)
        cand = w.candidate()
        p8, p12, p8b = config.pin(cand, 1), config.pin(cand, 2), config.pin(copy.deepcopy(cand), 1)
        self.assertNotEqual(p8["stream_config_hash"], p12["stream_config_hash"])
        self.assertNotEqual(p8["pinned_hash"], p12["pinned_hash"])
        self.assertEqual(p8["pinned_hash"], p8b["pinned_hash"])
        config.verify(p8)
        other = w.candidate(seed=7)
        self.assertNotEqual(config.pin(other, 1)["stream_config_hash"], p8["stream_config_hash"])
        tampered = copy.deepcopy(p8)
        tampered["stream"]["sources"][0]["share"] += 0.01
        with self.assertRaises(RuntimeError):
            config.verify(tampered)
        tampered = copy.deepcopy(p8)
        tampered["batch"]["batch_seconds"] = 300.0
        with self.assertRaises(RuntimeError):
            config.verify(tampered)

    def test_every_source_needs_a_read_unit_per_worker(self):
        w = World(self.root)
        with self.assertRaises(ValueError):
            config.pin(w.candidate(), 3)      # the fixture's sources have two read units

    def test_make_stream_refuses_another_worker_count(self):
        w = World(self.root)
        p = config.pin(w.candidate(), 2)
        with self.assertRaises(ValueError):
            config.make_stream(p, 4)
        with self.assertRaises(ValueError):
            config.make_stream(p, 0)
        s = config.make_stream(config.pin(w.candidate(), 1), 0)
        self.assertEqual(s.config_hash, config.pin(w.candidate(), 1)["stream_config_hash"])


if __name__ == "__main__":
    unittest.main()
