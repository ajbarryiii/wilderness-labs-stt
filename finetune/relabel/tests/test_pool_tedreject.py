"""Contract tests: the main pool's talk-text reject list (b0prime plan "Extra sources", the main-pool rule; the
containment rule of labeling plan Part B, applied per source recording and per label set).

- recording keys from record ids for every source form of the label run (subset sources share their corpus's);
- talks cut into 20-word windows; windows under 8 words or with only boilerplate 8-grams are not checked;
- a recording is rejected when one label set's transcript holds at least half of a window's non-boilerplate
  8-grams, counting 8-grams that cross record boundaries in recording order (not shard order), and every record
  of a rejected recording is listed; half under T* and half under B0 is not containment;
- boilerplate (a window 8-gram in more than 100 recordings) leaves the denominator;
- shard SHA-256 checked; the written list is hashed and a damaged or partial list is refused.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_pool_tedreject*' -v
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pool import tedreject

# 40 distinct non-number words: two full 20-word windows
WORDS = ("amber basil cedar dahlia ember fennel garnet hazel iris juniper kestrel laurel maple nettle olive "
         "pepper quartz raven sage thistle umber violet willow yarrow zinnia acorn bramble clover dune "
         "elder fjord glade heath inlet jasper knoll lichen marsh nook oxbow").split()
TALK = " ".join(WORDS[:40] + ["peony", "quill", "reed"])          # windows: [0,20) [20,40) [40,43)


def norm(text: str) -> str:
    return " ".join((text or "").lower().replace(".", " ").split())


def write_run(root: Path, shards: list[list[dict]], complete: bool = True) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    run = root / "labels" / "t" / "run"
    (run / "shards").mkdir(parents=True)
    entries = []
    for k, rows in enumerate(shards):
        path = run / "shards" / f"{k:05d}.labels.parquet"
        pq.write_table(pa.table({c: [r.get(c) for r in rows] for c in ("id", "source", "text", "b0_raw")}), path)
        entries.append({"shard": k, "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "records": len(rows)})
    (run / "manifest.json").write_text(json.dumps({"complete": complete, "shards": entries}))
    return run


def yid(video: str, offset: int, i: int = 0) -> str:
    return f"yodas:en000_{i:08d}_{video}_{offset}_00_5_00"


def rec(item: str, t: str = "", b: str = "") -> dict:
    return {"id": item, "source": item.split(":")[0], "text": t, "b0_raw": b}


def build(run: Path, local=None, **kw):
    return tedreject.build(run, [TALK], norm, local or {}, workers=0, log=lambda m: None, **kw)


class RecordingKeyTest(unittest.TestCase):
    def test_every_source_form(self):
        k = tedreject.recording_key
        self.assertEqual(k("yodas:en127_00000038_JPNWKbJgqxA_1042_30_29_02", {}),
                         ("yodas:JPNWKbJgqxA", (1042.0, 30.0)))
        self.assertEqual(k("yodas-med:en126_00000222_4Nnc_Y5q_112_06_6_34", {})[0], "yodas:4Nnc_Y5q")
        self.assertEqual(k("yodas-tech:en117_00000442_DwNrOg8PXRy_3962_58_29_76", {})[0], "yodas:DwNrOg8PXRy")
        self.assertEqual(k("peoples_speech:a_SLASH_b_DOT_mp3_00027.flac", {}), ("peoples_speech:a_SLASH_b_DOT_mp3", (27.0, 0.0)))
        self.assertEqual(k("ps-med:h_SLASH_h_DOT_mp3_00066.flac", {})[0], "peoples_speech:h_SLASH_h_DOT_mp3")
        self.assertEqual(k("librispeech:4856-20410-0027", {}), ("librispeech:4856-20410", (27.0, 0.0)))
        self.assertEqual(k("voxpopuli:20180528-0900-PLENARY-22-en_20180528-21:13:15_6", {}),
                         ("voxpopuli:20180528-0900-PLENARY-22-en", (20180528211315.0, 6.0)))
        self.assertEqual(k("ami:AMI_TS3009a_H00_MTD033PM_0103448_0104838", {}),
                         ("ami:AMI_TS3009a_H00", (103448.0, 104838.0)))
        self.assertEqual(k("med-osce:CAR0001-0002", {"med-osce:CAR0001-0002": ("CAR0001", 12.5)}),
                         ("med-osce:CAR0001", (12.5, 0.0)))
        for bad in ("yodas:short_id", "nosource", "unknown:x", "voxpopuli:bad", "med-osce:CAR0001-0002"):
            with self.assertRaises(ValueError):
                k(bad, {})


class WindowsTest(unittest.TestCase):
    def test_windows(self):
        w = tedreject.Windows([TALK, "too short a talk"], norm)
        self.assertEqual(w.words, [20, 20, 3, 4])
        self.assertEqual([len(g) for g in w.grams], [13, 13, 0, 0])
        self.assertEqual(w.start, [0, 20, 40, 0])


class BuildTest(unittest.TestCase):
    def test_containment_per_recording_and_label_set(self):
        w = WORDS
        first = " ".join(w[0:10])         # window 0's first half ...
        second = " ".join(w[10:20])       # ... and second half: together all 13 8-grams, apart none
        with tempfile.TemporaryDirectory() as t:
            run = write_run(Path(t), [
                # A: window 0 split across two records in two shards, out of shard order -> rejected (T*)
                [rec(yid("vidA", 20), t=second), rec(yid("vidB", 0), b=" ".join(w[20:30])),
                 rec(yid("vidC", 10), t=" ".join(w[20:27]), b=" ".join(w[27:40]))],
                [rec(yid("vidA", 10), t=first), rec(yid("vidA", 99), t="nothing to see"),
                 rec(yid("vidB", 5), b=" ".join(w[30:40])),
                 # D: the halves in reversed recording order -> the crossing 8-grams never form
                 rec(yid("vidD", 50), t=first), rec(yid("vidD", 40), t=second)]])
            body, ids = build(run)
            rejected = {r["recording"]: r["windows"] for r in body["rejected"]}
            self.assertEqual(set(rejected), {"yodas:vidA", "yodas:vidB"})
            self.assertEqual({x["label_set"] for x in rejected["yodas:vidA"]}, {"T*"})
            self.assertEqual(rejected["yodas:vidA"][0]["matched"], 13)
            self.assertEqual({x["label_set"] for x in rejected["yodas:vidB"]}, {"B0"})   # B0's 20..39 cross records
            self.assertEqual(ids, sorted([yid("vidA", 10), yid("vidA", 20), yid("vidA", 99), yid("vidB", 0),
                                          yid("vidB", 5)]))
            # C holds 7 words of window 1 under T* and 13 under B0: B0 alone has 6 of 13 8-grams (< 7)
            self.assertNotIn("yodas:vidC", rejected)
            self.assertEqual(body["counts"]["rejected_recordings"], 2)
            self.assertEqual(body["windows"]["checked"], 2)
            self.assertEqual(body["windows"]["skipped_short"], 1)

    def test_boilerplate_leaves_the_denominator(self):
        w = WORDS
        stock = " ".join(w[0:8])                          # window 0's first 8-gram, in 101 recordings
        six = " ".join(w[5:18])                           # 8-grams 5..10 of window 0: six of them
        with tempfile.TemporaryDirectory() as t:
            rows = [rec(yid(f"st{i:03d}", 0), t=stock) for i in range(101)] + [rec(yid("six", 0), t=six)]
            body, ids = build(write_run(Path(t), [rows]))
            # boilerplate: n = 12, need 6 -> rejected; without it, 6 < ceil(6.5)
            self.assertEqual([r["recording"] for r in body["rejected"]], ["yodas:six"])
            self.assertEqual(body["windows"]["boilerplate_8grams"], 1)
            body100, _ = build(write_run(Path(t) / "b", [rows[1:]]))
            self.assertEqual(body100["rejected"], [])     # 100 recordings: not boilerplate

    def test_shard_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as t:
            run = write_run(Path(t), [[rec(yid("v", 0), t="x")]])
            m = json.loads((run / "manifest.json").read_text())
            m["shards"][0]["sha256"] = "0" * 64
            (run / "manifest.json").write_text(json.dumps(m))
            with self.assertRaises(RuntimeError):
                build(run)

    def test_write_and_load(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "out"
            run = write_run(Path(t), [[rec(yid("vidA", 0), t=" ".join(WORDS[0:20]))]])
            body, ids = build(run)
            tedreject.write(out, body, ids, "talk-sha")
            got, identity = tedreject.load(out)
            self.assertEqual(got, frozenset(ids))
            self.assertEqual(identity["rejected_ids"], 1)
            self.assertEqual(identity["ids_sha256"], hashlib.sha256((ids[0] + "\n").encode()).hexdigest())
            (out / "rejected_ids.txt").write_text("")
            with self.assertRaises(RuntimeError):
                tedreject.load(out)
            partial = dict(body, inputs=dict(body["inputs"], complete=False))
            tedreject.write(out, partial, ids, "talk-sha")
            with self.assertRaises(RuntimeError):
                tedreject.load(out)


if __name__ == "__main__":
    unittest.main()
