"""Contract tests: the cluster-disjoint dev split and the paired cluster bootstrap (decoder-only plan "Dev
split for selection and evaluation", b0prime plan "Intervals")."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training import bootstrap, devsplit


def manifest(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def ls_rows(prefix: str, speakers: int, per: int) -> list[dict]:
    return [{"id": f"librispeech:{s}-{100 + s}-{u:04d}", "text": f"WORD {s} {u}", "duration": 1.0,
             "audio_filepath": "x"} for s in range(1000, 1000 + speakers) for u in range(per)]


class Clusters(unittest.TestCase):
    def test_cluster_of_each_dev_set(self):
        self.assertEqual(devsplit.cluster_of("librispeech_dev_clean", "librispeech:1272-128104-0000"), "1272")
        self.assertEqual(devsplit.cluster_of("librispeech_dev_other", "librispeech:116-288045-0001"), "116")
        self.assertEqual(devsplit.cluster_of("ami_dev", "ami:AMI_ES2011a_H00_FEE041_0003427_0003714"), "ES2011a")
        self.assertEqual(devsplit.cluster_of("voxpopuli_dev", "voxpopuli:20090203-0900-PLENARY-11-en_20090203-19:30:15_2"),
                         "20090203-0900-PLENARY")
        with self.assertRaises(ValueError):
            devsplit.cluster_of("voxpopuli_dev", "voxpopuli:something-else")
        with self.assertRaises(ValueError):
            devsplit.cluster_of("yodas_dev", "yodas:x")


class Split(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.d = d
        self.manifests = {
            "librispeech_dev_clean": manifest(d / "a.jsonl", ls_rows("a", 40, 3)),
            "librispeech_dev_other": manifest(d / "b.jsonl", ls_rows("b", 33, 2)),
            "ami_dev": manifest(d / "c.jsonl", [{"id": f"ami:AMI_M{m:02d}a_H00_X_{u:07d}_{u + 1:07d}",
                                                 "text": "OKAY SO", "duration": 1.0, "audio_filepath": "x"}
                                                for m in range(18) for u in range(4)]),
            "voxpopuli_dev": manifest(d / "e.jsonl", [{"id": f"voxpopuli:2009{m:04d}-0900-PLENARY-{i}-en_x_{u}",
                                                       "text": "a b c", "duration": 1.0, "audio_filepath": "x"}
                                                      for m in range(30) for i in (3, 7) for u in range(2)]),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_parts_are_cluster_disjoint_and_cover_every_utterance(self):
        s = devsplit.build(self.manifests)
        for name, path in self.manifests.items():
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            sel, ev = set(s["sets"][name]["selection"]["ids"]), set(s["sets"][name]["evaluation"]["ids"])
            self.assertFalse(sel & ev)
            self.assertEqual(sel | ev, {r["id"] for r in rows})
            csel = {devsplit.cluster_of(name, i) for i in sel}
            cev = {devsplit.cluster_of(name, i) for i in ev}
            self.assertFalse(csel & cev, name)
            self.assertEqual(sorted(csel), s["sets"][name]["selection"]["clusters"])
            n = len(csel | cev)
            self.assertEqual(len(csel), round(0.30 * n))
        self.assertEqual(len(s["sets"]["ami_dev"]["selection"]["clusters"]), 5)      # 18 meetings
        self.assertEqual(len(s["sets"]["ami_dev"]["evaluation"]["clusters"]), 13)

    def test_split_is_fixed_by_the_hash_rule(self):
        a = devsplit.build(self.manifests)
        rows = [json.loads(x) for x in self.manifests["librispeech_dev_clean"].read_text().splitlines()]
        manifest(self.manifests["librispeech_dev_clean"], list(reversed(rows)))     # order does not matter
        b = devsplit.build(self.manifests)
        self.assertEqual(a["sets"]["librispeech_dev_clean"]["selection"]["clusters"],
                         b["sets"]["librispeech_dev_clean"]["selection"]["clusters"])
        c = devsplit.build(self.manifests, salt="another")
        self.assertNotEqual(a["sets"]["librispeech_dev_clean"]["selection"]["clusters"],
                            c["sets"]["librispeech_dev_clean"]["selection"]["clusters"])
        self.assertEqual(a["rule"]["fraction"], 0.30)
        self.assertIn("sha256", a["rule"]["hash"])

    def test_counts_per_part(self):
        s = devsplit.build(self.manifests)
        p = s["sets"]["librispeech_dev_clean"]
        self.assertEqual(p["selection"]["utterances"] + p["evaluation"]["utterances"], 120)
        self.assertEqual(p["selection"]["reference_words"], 3 * p["selection"]["utterances"])
        self.assertEqual(p["selection"]["cluster_count"], 12)

    def test_frozen_manifest_is_written_once_and_verified(self):
        path = self.d / "split.json"
        s = devsplit.freeze(path, devsplit.build(self.manifests))
        loaded = devsplit.load(path, self.manifests)
        self.assertEqual(loaded.sha256, s["sha256"])
        some = s["sets"]["ami_dev"]["selection"]["ids"][0]
        self.assertEqual(loaded.part("ami_dev", some), "selection")
        devsplit.freeze(path, devsplit.build(self.manifests))       # identical: accepted
        rows = [json.loads(x) for x in self.manifests["ami_dev"].read_text().splitlines()]
        manifest(self.manifests["ami_dev"], rows[:-1])
        with self.assertRaises(RuntimeError):
            devsplit.freeze(path, devsplit.build(self.manifests))   # a different split never replaces the frozen one
        with self.assertRaises(RuntimeError):
            devsplit.load(path, self.manifests)                     # the manifest changed under the frozen split
        manifest(self.manifests["ami_dev"], rows)
        raw = json.loads(path.read_text())
        raw["sets"]["ami_dev"]["selection"]["ids"].append("ami:extra")
        path.write_text(json.dumps(raw))
        with self.assertRaises(RuntimeError):
            devsplit.load(path, self.manifests)


def records(errors: list[int], words: list[int], ids: list[str]) -> list[dict]:
    return [{"id": i, "S": e, "D": 0, "I": 0, "ref_norm": " ".join(["w"] * n), "scored": True}
            for i, e, n in zip(ids, errors, words)]


class Bootstrap(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.sets = {}
        for s, n_clusters in (("s1", 5), ("s2", 8)):
            ids = [f"{s}-{c}-{u}" for c in range(n_clusters) for u in range(6)]
            words = rng.integers(5, 15, len(ids)).tolist()
            a = rng.integers(0, 3, len(ids)).tolist()
            b = [max(0, x - (1 if k % 3 == 0 else 0)) for k, x in enumerate(a)]
            self.sets[s] = (ids, words, a, b)

    def tables(self, same: bool = False):
        out = {}
        for s, (ids, words, a, b) in self.sets.items():
            out[s] = bootstrap.SetTable.from_records(
                {"A": records(a, words, ids), "B": records(a if same else b, words, ids)},
                cluster_of=lambda i: i.split("-")[1])
        return out

    def test_point_estimates_are_corpus_wer_differences_in_points(self):
        res = bootstrap.paired(self.tables(), {"d": ("A", "B")}, mean_sets=["s1", "s2"], replicates=200, seed=1)
        wers = {}
        for s, (ids, words, a, b) in self.sets.items():
            wers[s] = (100 * sum(a) / sum(words), 100 * sum(b) / sum(words))
            self.assertAlmostEqual(res["sets"][s]["d"]["point"], wers[s][0] - wers[s][1])
            self.assertAlmostEqual(res["wer"][s]["A"], wers[s][0])
        h4 = np.mean([w[0] for w in wers.values()]) - np.mean([w[1] for w in wers.values()])
        self.assertAlmostEqual(res["mean"]["d"]["point"], h4)
        lo, hi = res["mean"]["d"]["ci"]
        self.assertLessEqual(lo, hi)
        self.assertEqual(res["replicates"], 200)

    def test_identical_models_have_a_degenerate_interval(self):
        res = bootstrap.paired(self.tables(same=True), {"d": ("A", "B")}, mean_sets=["s1", "s2"], replicates=300)
        self.assertEqual(res["mean"]["d"]["point"], 0.0)
        self.assertEqual(res["mean"]["d"]["ci"], [0.0, 0.0])

    def test_clusters_not_utterances_are_resampled(self):
        """With one cluster per set every replicate is the original data: the interval collapses to the point.
        An utterance bootstrap would not."""
        out = {}
        for s, (ids, words, a, b) in self.sets.items():
            out[s] = bootstrap.SetTable.from_records({"A": records(a, words, ids), "B": records(b, words, ids)},
                                                     cluster_of=lambda i: "one")
        res = bootstrap.paired(out, {"d": ("A", "B")}, mean_sets=["s1", "s2"], replicates=300)
        p = res["mean"]["d"]["point"]
        self.assertNotEqual(p, 0.0)
        self.assertAlmostEqual(res["mean"]["d"]["ci"][0], p)
        self.assertAlmostEqual(res["mean"]["d"]["ci"][1], p)

    def test_seeded_and_ten_thousand_by_default(self):
        a = bootstrap.paired(self.tables(), {"d": ("A", "B")}, mean_sets=["s1", "s2"])
        b = bootstrap.paired(self.tables(), {"d": ("A", "B")}, mean_sets=["s1", "s2"])
        self.assertEqual(a, b)
        self.assertEqual(a["replicates"], 10_000)
        self.assertEqual(a["ci_level"], 0.90)

    def test_models_must_cover_the_same_utterances(self):
        ids, words, a, b = self.sets["s1"]
        with self.assertRaises(ValueError):
            bootstrap.SetTable.from_records({"A": records(a, words, ids), "B": records(b[1:], words[1:], ids[1:])},
                                            cluster_of=lambda i: i.split("-")[1])

    def test_restricted_to_a_part(self):
        ids, words, a, b = self.sets["s1"]
        keep = {i for i in ids if i.split("-")[1] in {"0", "1"}}
        t = bootstrap.SetTable.from_records({"A": records(a, words, ids), "B": records(b, words, ids)},
                                            cluster_of=lambda i: i.split("-")[1], ids=keep)
        self.assertEqual(t.clusters, ["0", "1"])
        self.assertEqual(int(t.words.sum()), sum(w for i, w in zip(ids, words) if i in keep))


if __name__ == "__main__":
    unittest.main()
