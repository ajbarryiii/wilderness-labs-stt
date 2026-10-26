"""Subset selection contracts (labeling plan Part B "Score", "Thresholds", "Output").

- yodas-med / ps-med: >= 10 medical terms per 1,000 normalized words; yodas-tech: >= 10 technical terms per
  1,000 words and not medical; recordings under 200 words are excluded; denylisted recordings never enter a
  subset; peoples_speech has no technical subset.
- each included recording lists every (file, row group) holding one of its rows; counts, words and hours
  per subset are the sums over included recordings (hours also within the stream's 1-30 s filter).
- a seeded 50-recording sample per subset (all if fewer) with the first 300 words of the transcript.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_pool_*' -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pool import build
from test_pool_fixtures import Words, write_cache, write_listing, write_manifest

F1, F2 = "data/en000/asr_only/00000001.parquet", "data/en000/asr_only/00000002.parquet"
P1 = "clean/train-00000-of-00804.parquet"


def terms(kind):
    return ["medterm", "blood pressure"] if kind == "medical" else ["techterm"]


def text_with(w: Words, n: int, med: int = 0, tech: int = 0, multi: int = 0) -> list[str]:
    """n words: `med` x medterm, `tech` x techterm, `multi` x 'blood pressure', the rest filler."""
    words = ["medterm"] * med + ["techterm"] * tech + ["blood", "pressure"] * multi
    words += w.take(n - len(words))
    w.rng.shuffle(words)
    # keep each 'blood pressure' pair adjacent after the shuffle
    out = [x for x in words if x not in ("blood", "pressure")]
    for _ in range(multi):
        k = w.rng.randrange(len(out) + 1)
        out[k:k] = ["blood", "pressure"]
    return out


def rows(rec: str, words: list[str], per: int = 50, **kw) -> list[dict]:
    return [{"recording": rec, "order": float(i), "text": " ".join(words[k:k + per]), **kw}
            for i, k in enumerate(range(0, len(words), per))]


def read_subset(out: Path, name: str) -> dict[str, dict]:
    path = out / "subsets" / f"{name}.jsonl"
    return {r["recording"]: r for r in map(json.loads, open(path))} if path.exists() else {}


class SelectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        w = Words(seed=2)
        y1, y2, p1 = [], [], []
        y1 += rows("Ymed", text_with(w, 1000, med=10))
        y1 += rows("Ymed9", text_with(w, 1000, med=9))
        y1 += rows("Yshort", text_with(w, 199, med=50))
        y1 += rows("Y200", text_with(w, 200, med=2))
        y1 += rows("Ytech", text_with(w, 1000, med=3, tech=12))
        y1 += rows("Yboth", text_with(w, 1000, med=10, tech=20))
        y1 += rows("Ymulti", text_with(w, 1000, multi=10))
        y1 += rows("Ydeny", text_with(w, 1000, med=15))
        split = text_with(w, 1000, med=10)
        # Ysplit: 600 words in F1 row group 0, 400 words in F2 row groups 1 and 2; durations 10, 20, 40, 0.5 s.
        s = rows("Ysplit", split[:600], per=600, row_group=0, duration=10.0)
        s += [{"recording": "Ysplit", "order": 1.0, "text": " ".join(split[600:700]), "row_group": 1, "duration": 20.0},
              {"recording": "Ysplit", "order": 2.0, "text": " ".join(split[700:900]), "row_group": 2, "duration": 40.0},
              {"recording": "Ysplit", "order": 3.0, "text": " ".join(split[900:]), "row_group": 2, "duration": 0.5}]
        y1 += s[:1]
        y2 += s[1:]
        for i in range(60):
            y2 += rows(f"M{i:02d}", text_with(w, 400, med=8), duration=7.0)
        p1 += rows("Pmed", text_with(w, 1000, med=10))
        p1 += rows("Ptech", text_with(w, 1000, tech=20))
        caches = {"yodas": d / "cache" / "yodas", "peoples_speech": d / "cache" / "peoples_speech"}
        write_cache(caches["yodas"], "yodas", F1, y1)
        write_cache(caches["yodas"], "yodas", F2, y2)
        write_cache(caches["peoples_speech"], "peoples_speech", P1, p1)
        write_listing(caches["yodas"], "yodas", [F1, F2])
        write_listing(caches["peoples_speech"], "peoples_speech", [P1])
        heldout = [write_manifest(d / "heldout" / "devA.jsonl", [{"id": "devA:1", "text": "nothing in common"}])]
        cls.split_words = split
        cls.out = d / "pool"
        cls.args = dict(caches=caches, heldout=heldout, reserved=["Ydeny"], corpora={}, term_loader=terms,
                        workers=0, final=False)
        cls.summary = build.finalize(cls.out, **cls.args)
        cls.med, cls.tech, cls.ps = (read_subset(cls.out, n) for n in ("yodas-med", "yodas-tech", "ps-med"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_membership(self):
        self.assertEqual(set(self.med), {"Ymed", "Y200", "Yboth", "Ymulti", "Ysplit"} | {f"M{i:02d}" for i in range(60)})
        self.assertEqual(set(self.tech), {"Ytech"})
        self.assertEqual(set(self.ps), {"Pmed"})
        self.assertFalse((self.out / "subsets" / "ps-tech.jsonl").exists())

    def test_densities(self):
        self.assertEqual(self.med["Ymed"]["medical_per_1k"], 10.0)
        self.assertEqual(self.med["Y200"]["words"], 200)
        self.assertEqual(self.med["Ymulti"]["medical"], 10)

    def test_locators_and_hours(self):
        r = self.med["Ysplit"]
        self.assertEqual(r["locators"], [{"file": F1, "row_groups": [0]}, {"file": F2, "row_groups": [1, 2]}])
        self.assertEqual(r["words"], 1000)
        self.assertAlmostEqual(r["hours"], 70.5 / 3600, places=6)
        self.assertAlmostEqual(r["hours_1_30s"], 30.0 / 3600, places=6)
        summ = json.loads((self.out / "subsets" / "summary.json").read_text())
        for name, subset in (("yodas-med", self.med), ("yodas-tech", self.tech), ("ps-med", self.ps)):
            self.assertEqual(summ[name]["recordings"], len(subset))
            self.assertEqual(summ[name]["words"], sum(x["words"] for x in subset.values()))
            self.assertAlmostEqual(summ[name]["hours"], sum(x["hours"] for x in subset.values()), places=6)

    def test_exclusions_counted(self):
        summ = json.loads((self.out / "subsets" / "summary.json").read_text())
        ex = summ["excluded"]
        self.assertEqual(ex["under_min_words"]["yodas"], 1)
        self.assertEqual(ex["denylisted_qualifying"]["yodas-med"], 1)

    def test_sample(self):
        text = (self.out / "sample" / "yodas-med.txt").read_text()
        blocks = [b for b in text.split("\n## ")[1:]]
        self.assertEqual(len(blocks), 50)
        names = [b.split()[0] for b in blocks]
        self.assertEqual(len(set(names)), 50)
        self.assertTrue(set(names) <= set(self.med))
        if "Ysplit" in names:
            body = blocks[names.index("Ysplit")].split("\n", 1)[1].split()
            self.assertEqual(body, self.split_words[:300])
        tech = (self.out / "sample" / "yodas-tech.txt").read_text()
        self.assertEqual(tech.count("\n## "), 1)
        # deterministic
        out2 = Path(self.tmp.name) / "pool-again"
        build.finalize(out2, **self.args)
        self.assertEqual((out2 / "sample" / "yodas-med.txt").read_text(), text)

    def test_worker_processes_give_the_same_outputs(self):
        out2 = Path(self.tmp.name) / "pool-workers"
        build.finalize(out2, **(self.args | {"workers": 2}))
        for name in ("yodas-med", "yodas-tech", "ps-med"):
            self.assertEqual((out2 / "subsets" / f"{name}.jsonl").read_text(),
                             (self.out / "subsets" / f"{name}.jsonl").read_text())

    def test_first_300_words(self):
        text = (self.out / "sample" / "yodas-tech.txt").read_text()
        body = text.split("\n## ")[1].split("\n", 1)[1].split()
        self.assertEqual(len(body), 300)


if __name__ == "__main__":
    unittest.main()
