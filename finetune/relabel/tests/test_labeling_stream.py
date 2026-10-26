"""Contract tests: the labeling stream (include filter by row-group locator, exclusions, local sources, resume)."""
from __future__ import annotations

import itertools
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np

from labeling import config
from labeling_fixtures import World, recording, relabel_root


def take(s, n):
    return list(itertools.islice(iter(s), n))


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.world = World(cls.root)
        cls.pin = config.pin(cls.world.candidate(), 1)
        s = config.make_stream(cls.pin, 0)
        cls.items = take(s, 600)
        cls.counters = s.counters()
        s.close()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def recs(self):
        out = defaultdict(set)
        for it in self.items:
            out[it["source"]].add(recording(it["id"]))
        return out

    def test_every_source_is_drawn(self):
        self.assertEqual(set(self.recs()), {s["name"] for s in self.pin["stream"]["sources"]})

    def test_subsets_hold_only_their_recordings(self):
        r = self.recs()
        self.assertEqual(r["yodas-med"], {"yB", "yH"})
        self.assertEqual(r["yodas-tech"], {"yD", "yE"})
        self.assertEqual(r["ps-med"], {"pA"})

    def test_subset_reads_only_listed_row_groups(self):
        ids = {it["id"] for it in self.items}
        self.assertIn("yodas-med:yB-0", ids)
        self.assertNotIn("yodas-med:yB-8", ids)   # unlisted row group 0 of a file the subset reads
        self.assertNotIn("yodas-med:yB-9", ids)   # unlisted row group of another file
        self.assertNotIn("yodas:yB-9", ids)       # and the general source excludes yB
        self.assertNotIn("yodas:yB-8", ids)

    def test_general_sources_exclude_subsets_and_denylist(self):
        r = self.recs()
        self.assertTrue(r["yodas"] <= {"yC", "yF"}, r["yodas"])
        self.assertIn("yC", r["yodas"])
        self.assertTrue(r["peoples_speech"] <= {"pC", "pD"}, r["peoples_speech"])
        every = set().union(*r.values())
        self.assertFalse(every & {"yA", "yG", "pB"})

    def test_drop_reasons_are_counted(self):
        c = self.counters["sources"]
        self.assertGreater(c["yodas"]["dropped"].get("in_subset", 0), 0)
        self.assertGreater(c["yodas"]["dropped"].get("denylisted", 0), 0)
        self.assertGreater(c["yodas-med"]["dropped"].get("not_in_subset", 0), 0)
        self.assertGreater(c["yodas"]["dropped"].get("too_short", 0), 0)

    def test_references_and_kinds(self):
        for it in self.items:
            row = self.world.rows[it["id"] if it["source"] == "med-osce" else it["id"].split(":", 1)[1]]
            if it["source"] in ("yodas", "yodas-med", "yodas-tech"):
                self.assertIsNone(it["text"])
                self.assertEqual(it["reference"], row["text"])
                self.assertEqual(it["reference_kind"], "pseudo")
            else:
                text = row.get("text") or row.get("normalized_text")
                self.assertEqual(it["text"], text)
                self.assertEqual(it["reference"], text)
                self.assertEqual(it["reference_kind"], "human")

    def test_local_ids_are_the_manifest_ids(self):
        ids = {it["id"] for it in self.items if it["source"] == "med-osce"}
        self.assertTrue(ids)
        self.assertTrue(all(i.startswith("med-osce:osc") for i in ids))

    def test_audio_is_the_decoded_pcm16(self):
        import io
        import soundfile as sf
        for it in self.items[:40]:
            row = self.world.rows[it["id"] if it["source"] == "med-osce" else it["id"].split(":", 1)[1]]
            data = Path(row["audio_filepath"]).read_bytes() if "audio_filepath" in row else row["audio"]["bytes"]
            pcm, _ = sf.read(io.BytesIO(data), dtype="int16")
            np.testing.assert_array_equal(it["audio"], pcm.astype(np.float32) / 32768.0)


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ctx = relabel_root(self.root)
        self.ctx.__enter__()
        self.world = World(self.root)

    def tearDown(self):
        self.ctx.__exit__(None, None, None)
        self.tmp.cleanup()

    def check(self, workers):
        pin = config.pin(self.world.candidate(), max(1, workers))
        a = config.make_stream(pin, workers)
        it = iter(a)
        for _ in range(37):
            next(it)
        state = a.state_dict()
        rest = [next(it) for _ in range(45)]
        a.close()
        b = config.make_stream(pin, workers)
        b.load_state_dict(state)
        again = take(b, 45)
        b.close()
        self.assertEqual([x["id"] for x in rest], [x["id"] for x in again])
        for x, y in zip(rest, again):
            np.testing.assert_array_equal(x["audio"], y["audio"])
            self.assertEqual(x["reference"], y["reference"])

    def test_resume_in_process(self):
        self.check(0)

    def test_resume_with_worker_processes(self):
        self.check(2)

    def test_state_of_another_configuration_is_refused(self):
        pin = config.pin(self.world.candidate(), 1)
        a = config.make_stream(pin, 0)
        take(a, 5)
        state = a.state_dict()
        other = config.pin(self.world.candidate(seed=99), 1)
        b = config.make_stream(other, 0)
        with self.assertRaises(ValueError):
            b.load_state_dict(state)

    def test_same_configuration_same_sequence(self):
        pin = config.pin(self.world.candidate(), 1)
        a = [x["id"] for x in take(config.make_stream(pin, 0), 80)]
        b = [x["id"] for x in take(config.make_stream(pin, 0), 80)]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
