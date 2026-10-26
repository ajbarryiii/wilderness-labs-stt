"""Contract tests: re-filtering a finished C2 run under the current filters version (c2.refilter; labeling plan Part C
"Filters", amended 2026-10-25)."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from labeling import c2, checks, filters, shards
from labeling_fixtures import relabel_root
from test_labeling_c2 import fake_signals, table
from test_labeling_checks import clip_row, write_outputs

FAST = " ".join(f"wor{chr(97 + i)}" for i in range(14))      # 14 words in 2 s: 7 words per second


def substituted(text: str, n: int) -> str:
    w = text.split()
    return " ".join(["other"] * n + w[n:])


# id -> (source, reference, reference_kind, seconds, T* label, B0 label)
CLIPS = {
    "a:h-ok": ("a", FAST, "human", 2.0, FAST, "nothing alike"),                 # confirmed by the human transcript
    "a:h-bad": ("a", FAST, "human", 2.0, substituted(FAST, 4), FAST),            # human WER 0.29: unconfirmed
    "a:h-slow": ("a", "the patient is stable", "human", 2.0, "the patient is stable", "the patient is stable"),
    "a:loop": ("a", "yes " * 14, "human", 2.0, "yes " * 14, "yes " * 14),          # confirmed, but a loop
    "a:short": ("a", "hi", "human", 0.5, "hi", "hi"),
    "b:n-ok": ("b", None, None, 2.0, FAST, substituted(FAST, 1)),               # confirmed by B0
    "b:n-bad": ("b", None, None, 2.0, FAST, substituted(FAST, 4)),              # B0 WER 0.29: unconfirmed
    "b:p-ok": ("b", "a pseudo label that disagrees", "pseudo", 2.0, FAST, FAST),  # B0 confirms, not the pseudo label
    "b:slow": ("b", None, None, 2.0, "hello there friend", "hello there friend"),
}
CHANGED = {"a:h-ok", "b:n-ok", "b:p-ok"}                    # rejected under version 1, kept under version 2
ENGINE = {"kind": "fake", "confidence": True}


def cover(_text):
    return True


def files(run_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted((run_dir / "shards").glob("*"))}


def records(run_dir: Path) -> dict[str, dict]:
    return {r["id"]: r for p in sorted((run_dir / "shards").glob("*.labels.parquet")) for r in table(p)}


class Refilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.pool = cls.root / "pool"
        rows = [clip_row(i, src, ref, kind, seconds) for i, (src, ref, kind, seconds, _, _) in CLIPS.items()]
        checks.write_rows_as_shards(rows, cls.pool, 2)                     # 5 shards, 3 groups of 2 shards
        c2.write_json(cls.pool / "materialized.json", {})
        cls.v1 = cls.labeled("v1", version=1)
        cls.v2 = cls.labeled("v2")
        cls.v1_files = files(cls.v1)
        cls.out = cls.root / "labels" / "v1-f2"
        cls.result = c2.refilter(cls.v1, cls.out, coverage=cover)

    @classmethod
    def labeled(cls, name: str, version: int | None = None, finish: bool = True) -> Path:
        """A C2 run over the pool from fixed outputs (filters version: init_run's, or `version`)."""
        d = cls.root / "labels" / name
        c2.init_run(d, pool_dir=cls.pool, engine=ENGINE, group_shards=2, b0_batch_seconds=60.0, declared_vram_gb=None,
                    extra=None if version is None else {"filters_version": version})
        write_outputs(d, cls.pool, {i: c[4] for i, c in CLIPS.items()}, b0={i: c[5] for i, c in CLIPS.items()},
                      signals={i: fake_signals(c[4]) for i, c in CLIPS.items()})
        c2.merge_step(d, coverage=cover)
        if finish:
            assert c2.finish(d)["complete"]
        return d

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def test_a_version_1_run_keeps_version_1_dispositions(self):
        """merge applies the run's recorded filters version, so a run labeled under 1 stays consistent."""
        v1, v2 = records(self.v1), records(self.v2)
        for i in CHANGED:
            self.assertEqual((v1[i]["disposition"], v1[i]["reasons"]), ("rejected", ["words_per_second_gt_6"]), i)
            self.assertEqual((v2[i]["disposition"], v2[i]["reasons"]), ("kept", []), i)
        for i in set(CLIPS) - CHANGED:
            self.assertEqual(v1[i]["disposition"], v2[i]["disposition"], i)
        self.assertEqual(v1["a:loop"]["reasons"], ["words_per_second_gt_6", "ngram_loop"])
        self.assertEqual(v2["a:loop"]["reasons"], ["ngram_loop"])
        self.assertEqual(v2["a:h-bad"]["reasons"], ["words_per_second_gt_6"])
        self.assertEqual(v2["b:n-bad"]["reasons"], ["words_per_second_gt_6"])

    def test_refiltered_records_equal_a_version_2_merge_of_the_same_outputs(self):
        self.assertEqual(sorted(files(self.out)), sorted(files(self.v2)))
        for p in sorted((self.v2 / "shards").glob("*.labels.parquet")):
            self.assertEqual(table(self.out / "shards" / p.name), table(p), p.name)
        for p in sorted((self.v2 / "shards").glob("*.json")):
            want = json.loads(p.read_text())
            got = json.loads((self.out / "shards" / p.name).read_text())
            self.assertEqual({k: v for k, v in got.items() if k not in ("filters_version", "source_sha256")},
                             {k: v for k, v in want.items() if k != "sha256"} | {"sha256": got["sha256"]})

    def test_refilter_changes_only_dispositions(self):
        v1, out = records(self.v1), records(self.out)
        self.assertEqual(set(v1), set(out))
        decision = {"disposition", "reasons"}
        for i, r in v1.items():
            self.assertEqual({k: v for k, v in out[i].items() if k not in decision},
                             {k: v for k, v in r.items() if k not in decision}, i)
            self.assertTrue(out[i]["t_token_ids"])
        self.assertEqual({i for i in out if out[i]["disposition"] != v1[i]["disposition"]}, CHANGED)

    def test_the_new_run_records_its_filters_version_and_source(self):
        run = c2.load_run(self.out)
        src = c2.load_run(self.v1)
        self.assertEqual(run["filters_version"], filters.VERSION)
        self.assertEqual(run["refilter"]["source_filters_version"], 1)
        self.assertEqual(run["refilter"]["source_run_dir"], "labels/v1")
        self.assertEqual(run["refilter"]["source_manifest_sha256"], shards.sha256_file(self.v1 / "manifest.json"))
        self.assertEqual(c2.contract(run), c2.contract(src) | {"filters_version": 2})
        m = json.loads((self.out / "manifest.json").read_text())
        self.assertTrue(m["complete"])
        self.assertEqual(m["run"], run)
        self.assertEqual(m["records"], len(CLIPS))
        by = m["by_source_distinct_ids"]
        self.assertEqual((by["a"]["kept"], by["b"]["kept"]), (2, 3))
        self.assertEqual((by["a"]["rate_confirmed"], by["b"]["rate_confirmed"]), (2, 2))   # incl. the loop
        old = json.loads((self.v1 / "manifest.json").read_text())["by_source_distinct_ids"]
        self.assertEqual((old["a"]["kept"], old["b"]["kept"]), (1, 1))
        self.assertEqual((old["a"]["rate_confirmed"], old["b"]["rate_confirmed"]), (0, 0))
        for f in m["shards"]:
            meta = json.loads((self.out / "shards" / f"{shards.shard_name(f['shard'])}.json").read_text())
            src_meta = json.loads((self.v1 / "shards" / f"{shards.shard_name(f['shard'])}.json").read_text())
            self.assertEqual(shards.sha256_file(self.out / "shards" / f["file"]), meta["sha256"])
            self.assertEqual(meta["filters_version"], 2)
            self.assertEqual(meta["source_sha256"], src_meta["sha256"])
        self.assertEqual(self.result["refiltered"], [f["shard"] for f in m["shards"]])
        self.assertEqual(self.result["kept"], {"a": [1, 2], "b": [1, 3]})

    def test_the_source_run_is_untouched(self):
        self.assertEqual(files(self.v1), self.v1_files)
        self.assertEqual(c2.load_run(self.v1)["filters_version"], 1)

    def test_idempotent_and_resumable_per_shard(self):
        out = self.root / "labels" / "v1-f2-resume"
        c2.refilter(self.v1, out, coverage=cover)
        before = files(out) | {"manifest.json": (out / "manifest.json").read_bytes()}
        again = c2.refilter(self.v1, out, coverage=cover)
        self.assertEqual((again["refiltered"], again["skipped"]), ([], 5))
        self.assertEqual(files(out) | {"manifest.json": (out / "manifest.json").read_bytes()}, before)
        (out / "shards" / "00003.json").unlink()                  # interrupted after shard 3's records, before its meta
        (out / "manifest.json").unlink()
        resumed = c2.refilter(self.v1, out, coverage=cover)
        self.assertEqual(resumed["refiltered"], [3])
        self.assertEqual(files(out) | {"manifest.json": (out / "manifest.json").read_bytes()}, before)

    def test_refuses_an_incomplete_run(self):
        d = self.root / "labels" / "partial"
        c2.init_run(d, pool_dir=self.pool, engine=ENGINE, group_shards=2, b0_batch_seconds=60.0, declared_vram_gb=None,
                    extra={"filters_version": 1})
        write_outputs(d, self.pool, {i: c[4] for i, c in CLIPS.items()}, b0={i: c[5] for i, c in CLIPS.items()},
                      signals={i: fake_signals(c[4]) for i, c in CLIPS.items()})
        (d / "work" / "b0" / "00002.json").unlink()               # the last group has no B0 output yet
        c2.merge_step(d, coverage=cover)
        self.assertFalse(c2.finish(d)["complete"])
        out = self.root / "labels" / "partial-f2"
        with self.assertRaisesRegex(RuntimeError, "not complete"):
            c2.refilter(d, out, coverage=cover)
        self.assertFalse(out.exists())

    def test_refuses_a_run_already_at_the_current_version(self):
        with self.assertRaisesRegex(RuntimeError, "already"):
            c2.refilter(self.v2, self.root / "labels" / "v2-f2", coverage=cover)

    def test_refuses_its_own_directory_and_another_runs_output(self):
        with self.assertRaises(ValueError):
            c2.refilter(self.v1, self.v1, coverage=cover)
        other = self.root / "labels" / "v1-other"
        shutil.copytree(self.v1, other)
        run = json.loads((other / "run.json").read_text()) | {"b0": {"batch_seconds": 30.0}}
        (other / "run.json").write_text(json.dumps(run))
        with self.assertRaisesRegex(RuntimeError, "different settings"):
            c2.refilter(other, self.out, coverage=cover)

    def test_refuses_a_label_shard_that_differs_from_its_meta(self):
        bad = self.root / "labels" / "v1-tampered"
        shutil.copytree(self.v1, bad)
        p = bad / "shards" / "00001.labels.parquet"
        rows = table(p)
        rows[0]["raw"] = rows[0]["text"] = rows[0]["norm"] = "changed"
        c2._write_parquet(p, rows, c2.label_schema())
        with self.assertRaisesRegex(RuntimeError, "SHA-256"):
            c2.refilter(bad, self.root / "labels" / "v1-tampered-f2", coverage=cover)

    def test_labeling_steps_refuse_a_refiltered_run(self):
        for step in (lambda: c2.teacher_step(self.out, engine=c2.FakeTeacher(ENGINE)),
                     lambda: c2.b0_step(self.out, labeler=object()), lambda: c2.merge_step(self.out, coverage=cover)):
            with self.assertRaisesRegex(RuntimeError, "refiltered"):
                step()

    def test_default_output_is_a_sibling_named_for_the_version(self):
        self.assertEqual(c2.refilter_dir(Path("labels/t/main")), Path("labels/t/main-f2"))


if __name__ == "__main__":
    unittest.main()
