"""Global denylist contracts (labeling plan Part B "Global denylist").

- the reserved YODAS dev recordings are listed (rule `reserved`);
- recordings whose id is published by a held-out utterance are listed (rule `identity`; yodas_dev ids);
- text containment: evaluation-normalized word 8-grams; 8-grams in more than 100 training recordings are
  boilerplate and ignored; a held-out utterance of at least 8 words is contained in a recording if at least
  50% of its distinct non-boilerplate 8-grams occur in that recording's transcript (rows concatenated in
  recording order); utterances under 8 words, or with no 8-grams left, are skipped and counted;
- every training route is checked (yodas, peoples_speech, local corpora); boilerplate counts recordings
  across all of them;
- counts per rule and per held-out set; each held-out manifest is recorded with its SHA-256;
- a final build refuses missing domain-dev parts or an incomplete text cache.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_pool_*' -v
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pool import build
from pool import scan as scan_mod
from test_pool_fixtures import Words, phrase, write_cache, write_listing, write_manifest

F1, F2 = "data/en000/asr_only/00000001.parquet", "data/en000/asr_only/00000002.parquet"
P1 = "clean/train-00000-of-00804.parquet"


def terms(kind):
    return ["medterm"] if kind == "medical" else ["techterm"]


def rows_of(recording: str, words: list[str], per_row: int = 50, start: float = 0.0) -> list[dict]:
    return [{"recording": recording, "order": start + i, "text": " ".join(words[k:k + per_row])}
            for i, k in enumerate(range(0, len(words), per_row))]


class DenylistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        w = Words(seed=1)
        u = {k: phrase(n, t) for k, n, t in (("u1", 20, "ua"), ("u2", 20, "ub"), ("u3", 19, "uc"),
                                             ("u7", 7, "ue"), ("u8", 12, "uf"))}
        B, B100, ud = phrase(8, "bp"), phrase(8, "bq"), phrase(2, "ud")
        u["u4"], u["u5"], u["u6"] = B + ud, B100, B
        y1, y2, p1 = [], [], []
        y1 += rows_of("R1", w.take(1000) + u["u1"] + w.take(980))
        # u2 split over two rows, written out of order; the order key restores the sequence.
        r2 = [{"recording": "R2", "order": 3.0, "text": " ".join(u["u2"][10:] + w.take(30))},
              {"recording": "R2", "order": 0.0, "text": " ".join(w.take(40))},
              {"recording": "R2", "order": 2.0, "text": " ".join(w.take(30) + u["u2"][:10])},
              {"recording": "R2", "order": 1.0, "text": " ".join(w.take(40))}]
        y1 += r2
        y2 += rows_of("R3", w.take(100) + u["u3"][:13] + w.take(100))    # 6 of 12 8-grams: contained
        p1 += rows_of("R4", w.take(100) + u["u3"][:12] + w.take(100))    # 5 of 12: not contained
        y2 += rows_of("R5", w.take(50) + u["u4"][1:9] + w.take(50))      # 1 of 2 non-boilerplate (1 of 3 raw)
        y2 += rows_of("R6", w.take(50) + u["u7"] + w.take(50))           # short utterance: never checked
        for i in range(60):
            y2 += rows_of(f"BY{i}", w.take(20) + B + w.take(20))
        for i in range(41):
            p1 += rows_of(f"BP{i}", w.take(20) + B + w.take(20))        # B in 101 recordings: boilerplate
        for i in range(100):
            y2 += rows_of(f"BQ{i}", w.take(20) + B100 + w.take(20))     # B100 in exactly 100: not boilerplate
        cls.caches = {"yodas": d / "cache" / "yodas", "peoples_speech": d / "cache" / "peoples_speech"}
        write_cache(cls.caches["yodas"], "yodas", F1, y1)
        write_cache(cls.caches["yodas"], "yodas", F2, y2)
        write_cache(cls.caches["peoples_speech"], "peoples_speech", P1, p1)
        write_listing(cls.caches["yodas"], "yodas", [F1, F2])
        write_listing(cls.caches["peoples_speech"], "peoples_speech", [P1])
        dev = write_manifest(d / "heldout" / "devA.jsonl",
                             [{"id": f"devA:{k}", "text": " ".join(u[k]), "part": "med-dev-1"}
                              for k in ("u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8")])
        ydev = write_manifest(d / "heldout" / "yodas_dev.jsonl",
                              [{"id": "yodas:en000_00000316_idRec_x_10_20_3_45", "text": "short words only"}])
        osce = write_manifest(d / "corpora" / "med-osce.jsonl",
                              [{"id": "med-osce:CAR9_0000", "recording": "CAR9", "start": 0.0, "end": 4.0,
                                "text": " ".join(w.take(30) + u["u8"][:6])},
                               {"id": "med-osce:CAR9_0001", "recording": "CAR9", "start": 4.0, "end": 8.0,
                                "text": " ".join(u["u8"][6:] + w.take(30))}])
        cls.heldout = [dev, ydev]
        cls.out = d / "pool"
        cls.summary = build.finalize(cls.out, caches=cls.caches, heldout=cls.heldout, reserved=["resA"],
                                     corpora={"med-osce": osce}, term_loader=terms, workers=0, final=False)
        cls.deny = json.loads((cls.out / "denylist.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_recordings(self):
        recs = self.deny["recordings"]
        want_y = {"R1", "R2", "R3", "R5", "idRec_x", "resA"} | {f"BQ{i}" for i in range(100)}
        self.assertEqual(set(recs["yodas"]), want_y)
        self.assertEqual(recs.get("peoples_speech", []), [])
        self.assertEqual(recs["med-osce"], ["CAR9"])

    def test_counts_by_rule(self):
        c = self.deny["counts"]
        self.assertEqual(c["by_rule"], {"reserved": 1, "identity": 1, "containment": 105})
        self.assertEqual(c["total"], 107)
        self.assertEqual(c["boilerplate_8grams"], 1)

    def test_per_heldout_set(self):
        sets = {s["name"]: s for s in self.deny["heldout"]}
        a = sets["devA"]
        self.assertEqual((a["utterances"], a["checked"], a["skipped_short"], a["skipped_empty"],
                          a["contained_utterances"]), (8, 6, 1, 1, 6))
        y = sets["yodas_dev"]
        self.assertEqual((y["utterances"], y["checked"], y["skipped_short"], y["identity_recordings"]), (1, 0, 1, 1))
        for s, path in zip(("devA", "yodas_dev"), self.heldout):
            self.assertEqual(sets[s]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_details_name_the_held_out_utterance(self):
        details = {(r["source"], r["recording"]): r for r in map(json.loads, open(self.out / "denylist_details.jsonl"))}
        self.assertIn("devA:u1", [h["id"] for h in details[("yodas", "R1")]["heldout"]])
        self.assertEqual(details[("yodas", "resA")]["rules"], ["reserved"])
        self.assertEqual(details[("yodas", "idRec_x")]["rules"], ["identity"])

    def test_final_build_requires_domain_parts(self):
        with self.assertRaises(RuntimeError):
            build.finalize(Path(self.tmp.name) / "pool2", caches=self.caches, heldout=self.heldout, reserved=[],
                           corpora={}, term_loader=terms, workers=0, final=True,
                           required_parts=("med-dev-1", "tech-dev-1"))

    def test_incomplete_cache_refused(self):
        d = Path(self.tmp.name) / "partial"
        write_cache(d, "yodas", F1, rows_of("Z", ["a"] * 10))
        write_listing(d, "yodas", [F1, F2])  # F2 has no done marker
        with self.assertRaises(RuntimeError):
            build.finalize(Path(self.tmp.name) / "pool3", caches={"yodas": d}, heldout=self.heldout, reserved=[],
                           corpora={}, term_loader=terms, workers=0, final=False)


class CrossFileTest(unittest.TestCase):
    """A recording's transcript is all its rows in recording order, across files: 8-grams spanning a file
    boundary count, and rows that files interleave are put back in order (no false adjacency)."""

    def test_recordings_spanning_files(self):
        F3 = "data/en000/asr_only/00000003.parquet"
        w = Words(seed=3)
        u, v, x, y = phrase(8, "xa"), phrase(8, "xb"), phrase(8, "xc"), phrase(8, "xd")
        f1, f2, f3 = [], [], []
        # S: u split 4 + 4 over two files.
        f1 += [{"recording": "S", "order": 0.0, "text": " ".join(w.take(20) + u[:4])}]
        f2 += [{"recording": "S", "order": 1.0, "text": " ".join(u[4:] + w.take(20))}]
        # T: v over three files, the middle piece only 3 words.
        f1 += [{"recording": "T", "order": 0.0, "text": " ".join(w.take(20) + v[:2])}]
        f2 += [{"recording": "T", "order": 1.0, "text": " ".join(v[2:5])}]
        f3 += [{"recording": "T", "order": 2.0, "text": " ".join(v[5:] + w.take(20))}]
        # I: rows interleaved over two files; x spans rows 0-1 in recording order.
        f1 += [{"recording": "I", "order": 0.0, "text": " ".join(w.take(20) + x[:4])},
               {"recording": "I", "order": 2.0, "text": " ".join(w.take(20))}]
        f2 += [{"recording": "I", "order": 1.0, "text": " ".join(x[4:] + w.take(20))},
               {"recording": "I", "order": 3.0, "text": " ".join(w.take(20))}]
        # J: y's halves are in rows 0 and 2 (one file); row 1 (another file) lies between them: not contained.
        f1 += [{"recording": "J", "order": 0.0, "text": " ".join(w.take(20) + y[:4])},
               {"recording": "J", "order": 2.0, "text": " ".join(y[4:] + w.take(20))}]
        f2 += [{"recording": "J", "order": 1.0, "text": " ".join(w.take(20))}]
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            cache = d / "cache"
            for path, rows in ((F1, f1), (F2, f2), (F3, f3)):
                write_cache(cache, "yodas", path, rows)
            write_listing(cache, "yodas", [F1, F2, F3])
            held = write_manifest(d / "h.jsonl", [{"id": f"h:{k}", "text": " ".join(p)}
                                                  for k, p in (("u", u), ("v", v), ("x", x), ("y", y))])
            for workers in (2, 0):
                scan_mod._STATE.clear()   # as in a fresh process: nothing initialized in the parent
                build.finalize(d / f"pool{workers}", caches={"yodas": cache}, heldout=[held], reserved=[],
                               corpora={}, term_loader=terms, workers=workers, final=False)
                deny = json.loads((d / f"pool{workers}" / "denylist.json").read_text())
                self.assertEqual(deny["recordings"]["yodas"], ["I", "S", "T"])


class DiscoveryTest(unittest.TestCase):
    """Held-out manifests under the devsets root are found at any depth (e.g. a later eval-ext set), non-manifest
    JSONL is skipped, and a row may publish recording ids for the identity rule."""

    def test_discovery_and_row_identity(self):
        from pool import domain
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            root = d / "devsets"
            write_manifest(root / "dev-medical.jsonl", [{"id": "m:1", "text": "a b c", "part": "med-dev-1"}])
            ext = write_manifest(root / "eval-ext" / "medical-b.jsonl",
                                 [{"id": "e:1", "text": "x y", "identity": {"yodas": "VID123"}},
                                  {"id": "e:2", "text": "z", "identity": {"yodas": ["VID9", "VID8"]}}])
            (root / "records").mkdir()
            (root / "records" / "scores.jsonl").write_text(json.dumps({"utt": 1, "hyp": "q"}) + "\n")
            found, skipped = domain.manifests(root)
            self.assertEqual([n for n, _ in found], ["dev-medical", "eval-ext/medical-b"])
            self.assertEqual(skipped, [str(root / "records" / "scores.jsonl")])
            cache = d / "cache"
            write_cache(cache, "yodas", F1, rows_of("VID123", ["w"] * 20))
            write_listing(cache, "yodas", [F1])
            build.finalize(d / "pool", caches={"yodas": cache}, heldout=found, reserved=[], corpora={},
                           term_loader=terms, workers=0, final=False)
            deny = json.loads((d / "pool" / "denylist.json").read_text())
            self.assertEqual(deny["recordings"]["yodas"], ["VID123", "VID8", "VID9"])
            sets = {s["name"]: s for s in deny["heldout"]}
            self.assertEqual(sets["eval-ext/medical-b"]["identity_recordings"], 3)
            self.assertEqual(sets["eval-ext/medical-b"]["sha256"], hashlib.sha256(ext.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
