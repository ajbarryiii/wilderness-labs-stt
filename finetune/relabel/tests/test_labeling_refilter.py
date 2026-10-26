"""Contract tests: re-filtering a finished C2 run under the current filters version (c2.refilter; labeling plan Part C
"Filters", amended 2026-10-25)."""
from __future__ import annotations

import hashlib
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
    """Version 1 to version 2 (the rate confirmation); refilter's target version is named, so these semantics hold
    whatever filters.VERSION is."""
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
        cls.v2 = cls.labeled("v2", version=2)
        cls.v1_files = files(cls.v1)
        cls.out = cls.root / "labels" / "v1-f2"
        cls.result = c2.refilter(cls.v1, cls.out, coverage=cover, version=2)

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
        self.assertEqual(run["filters_version"], 2)
        self.assertEqual(run["refilter"]["source_filters_version"], 1)
        self.assertEqual(run["refilter"]["source_run_dir"], "labels/v1")
        listed = json.loads((self.v1 / "manifest.json").read_text())["shards"]
        self.assertEqual(run["refilter"]["source_shards_sha256"],
                         hashlib.sha256(json.dumps(listed, sort_keys=True).encode()).hexdigest())
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
        c2.refilter(self.v1, out, coverage=cover, version=2)
        before = files(out) | {"manifest.json": (out / "manifest.json").read_bytes()}
        again = c2.refilter(self.v1, out, coverage=cover, version=2)
        self.assertEqual((again["refiltered"], again["skipped"]), ([], 5))
        self.assertEqual(files(out) | {"manifest.json": (out / "manifest.json").read_bytes()}, before)
        (out / "shards" / "00003.json").unlink()                  # interrupted after shard 3's records, before its meta
        (out / "manifest.json").unlink()
        resumed = c2.refilter(self.v1, out, coverage=cover, version=2)
        self.assertEqual(resumed["refiltered"], [3])
        self.assertEqual(files(out) | {"manifest.json": (out / "manifest.json").read_bytes()}, before)

    def test_a_rerendered_source_manifest_does_not_block_resuming(self):
        """finish re-renders a complete run's manifest on every later chunk (other code may add keys); the output is
        pinned to the source's label shards, not to the manifest file's bytes."""
        src = self.root / "labels" / "v1-rerendered"
        shutil.copytree(self.v1, src)
        out = self.root / "labels" / "v1-rerendered-f2"
        c2.refilter(src, out, coverage=cover, version=2)
        m = json.loads((src / "manifest.json").read_text())
        (src / "manifest.json").write_text(json.dumps(m | {"note": "re-rendered"}, indent=2))
        self.assertEqual(c2.refilter(src, out, coverage=cover, version=2)["refiltered"], [])

    def test_refuses_an_output_outside_the_relabel_root(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            with self.assertRaisesRegex(ValueError, "RELABEL_ROOT"):
                c2.refilter(self.v1, Path(elsewhere) / "v1-f2", coverage=cover)
            self.assertFalse((Path(elsewhere) / "v1-f2").exists())

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

    def test_refuses_a_run_already_at_the_target_version(self):
        with self.assertRaisesRegex(RuntimeError, "already"):
            c2.refilter(self.v2, self.root / "labels" / "v2-f2", coverage=cover, version=2)
        current = self.labeled("current")                               # init_run: filters.VERSION
        self.assertEqual(c2.load_run(current)["filters_version"], filters.VERSION)
        with self.assertRaisesRegex(RuntimeError, "already"):
            c2.refilter(current, self.root / "labels" / "current-f", coverage=cover)
        with self.assertRaisesRegex(ValueError, "version"):
            c2.refilter(self.v1, self.root / "labels" / "v1-f9", coverage=cover, version=9)
        self.assertFalse((self.root / "labels" / "v1-f9").exists())

    def test_refuses_its_own_directory_and_another_runs_output(self):
        with self.assertRaises(ValueError):
            c2.refilter(self.v1, self.v1, coverage=cover)
        other = self.root / "labels" / "v1-other"
        shutil.copytree(self.v1, other)
        run = json.loads((other / "run.json").read_text()) | {"b0": {"batch_seconds": 30.0}}
        (other / "run.json").write_text(json.dumps(run))
        with self.assertRaisesRegex(RuntimeError, "different settings"):
            c2.refilter(other, self.out, coverage=cover, version=2)

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
        self.assertEqual(c2.refilter_dir(Path("labels/t/main")), Path(f"labels/t/main-f{filters.VERSION}"))
        self.assertEqual(c2.refilter_dir(Path("labels/t/main")), Path("labels/t/main-f3"))
        self.assertEqual(c2.refilter_dir(Path("labels/t/main"), version=2), Path("labels/t/main-f2"))


# --- filters version 3: punctuation mapping (labeling plan Part C "Filters", amended 2026-10-26) ---

B0_LACKS = set('"#&()*+;<=>@[\\]_{|}~`')


def b0_like(text: str) -> bool:
    """A stand-in for B0's tokenizer coverage: ASCII without the characters B0's vocabulary lacks."""
    return all(c.isascii() and c not in B0_LACKS for c in text)


QUOTE = 'She said, "I\'m bored."'
PAREN = "seventy two percent (of Denver) said yes"
# id -> (source, reference, reference_kind, seconds, T* label, B0 label)
CLIPS3 = {
    "c:quote": ("c", "she said i'm bored", "human", 3.0, QUOTE, "she said i'm bored"),
    "c:paren": ("c", None, None, 3.0, PAREN, "seventy two percent of denver said yes"),
    "c:both": ("c", None, None, 3.0, f'"{PAREN}" [sic]', "seventy two percent of denver said yes sic"),
    "c:hash": ("c", None, None, 3.0, "item #5 is ready", "item five is ready"),
    "c:jp": ("c", None, None, 3.0, "これは100円です。", "nothing alike here at all"),
    "c:loop": ("c", None, None, 5.0, '"yes yes yes yes yes"', "yes yes yes yes yes"),
    "d:plain": ("d", "hello there friend", "human", 2.0, "hello there friend", "hello there friend"),
    "d:amp": ("d", "salt and pepper please", "human", 2.0, "salt & pepper please", "salt and pepper please"),
}
KEPT3 = {"c:quote", "c:paren", "c:both", "d:plain", "d:amp"}      # under version 3


class RefilterVersion3(unittest.TestCase):
    """Re-filtering to version 3 recomputes the stored training text (`text`) of every record, kept or rejected, with
    version 3's mapping; filters 1-3 and every other column are unchanged."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.pool = cls.root / "pool"
        rows = [clip_row(i, src, ref, kind, seconds) for i, (src, ref, kind, seconds, _, _) in CLIPS3.items()]
        checks.write_rows_as_shards(rows, cls.pool, 3)                     # 3 shards, 2 groups
        c2.write_json(cls.pool / "materialized.json", {})
        cls.v1 = cls.labeled("main", 1)
        cls.v2 = cls.labeled("v2", 2)
        cls.v3 = cls.labeled("v3", 3)
        cls.v1_files = files(cls.v1)
        cls.result = c2.refilter(cls.v1, coverage=b0_like)              # the default: main-f3, filters.VERSION
        cls.out = cls.root / "labels" / "main-f3"
        cls.v12 = cls.root / "labels" / "main-f2"
        c2.refilter(cls.v1, cls.v12, coverage=b0_like, version=2)
        cls.v123 = cls.root / "labels" / "main-f2-f3"
        cls.result_v123 = c2.refilter(cls.v12, coverage=b0_like)

    @classmethod
    def labeled(cls, name: str, version: int) -> Path:
        d = cls.root / "labels" / name
        c2.init_run(d, pool_dir=cls.pool, engine=ENGINE, group_shards=2, b0_batch_seconds=60.0, declared_vram_gb=None,
                    extra={"filters_version": version})
        write_outputs(d, cls.pool, {i: c[4] for i, c in CLIPS3.items()}, b0={i: c[5] for i, c in CLIPS3.items()},
                      signals={i: fake_signals(c[4]) for i, c in CLIPS3.items()})
        c2.merge_step(d, coverage=b0_like)
        assert c2.finish(d)["complete"]
        return d

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def test_merge_maps_under_the_runs_version(self):
        for run, version in ((self.v1, 1), (self.v2, 2), (self.v3, 3)):
            for i, r in records(run).items():
                self.assertEqual(r["text"], filters.map_text(CLIPS3[i][4], version=version), (version, i))
        v1, v3 = records(self.v1), records(self.v3)
        self.assertEqual(v1["c:quote"]["text"], QUOTE)                         # version 1 keeps the quotes
        self.assertEqual(v1["c:quote"]["reasons"], ["tokenizer_coverage"])
        self.assertEqual(v3["c:quote"]["text"], "She said, I'm bored.")
        self.assertEqual({i for i, r in v3.items() if r["disposition"] == "kept"}, KEPT3)
        self.assertEqual({i for i, r in v1.items() if r["disposition"] == "kept"}, {"d:plain"})

    def test_refilter_to_version_3_recomputes_the_text_of_every_record(self):
        self.assertEqual(self.result["out"], str(self.out))
        out, v1 = records(self.out), records(self.v1)
        self.assertEqual(set(out), set(CLIPS3))
        for i, r in out.items():
            self.assertEqual(r["text"], filters.map_text(r["raw"], version=3), i)
        self.assertEqual(out["c:quote"]["text"], "She said, I'm bored.")
        self.assertEqual(out["c:paren"]["text"], "seventy two percent of Denver said yes")
        self.assertEqual(out["c:both"]["text"], "seventy two percent of Denver said yes sic")
        self.assertEqual(out["c:loop"]["text"], "yes yes yes yes yes")                         # rejected, mapped too
        self.assertEqual(out["d:amp"]["text"], "salt and pepper please")
        self.assertEqual({i for i, r in out.items() if r["disposition"] == "kept"}, KEPT3)
        self.assertEqual(out["c:hash"]["reasons"], ["tokenizer_coverage"])
        self.assertEqual(out["c:jp"]["reasons"], ["b0_wer_gt_0.8", "tokenizer_coverage"])
        self.assertEqual(out["c:loop"]["reasons"], ["ngram_loop"])
        changed = {"disposition", "reasons", "text"}
        for i, r in v1.items():
            self.assertEqual({k: v for k, v in out[i].items() if k not in changed},
                             {k: v for k, v in r.items() if k not in changed}, i)

    def test_refiltered_records_equal_a_version_3_merge_of_the_same_outputs(self):
        for p in sorted((self.v3 / "shards").glob("*.labels.parquet")):
            self.assertEqual(table(self.out / "shards" / p.name), table(p), p.name)
            self.assertEqual(table(self.v123 / "shards" / p.name), table(p), p.name)
        self.assertEqual(c2.load_run(self.out)["filters_version"], 3)
        by = json.loads((self.out / "manifest.json").read_text())["by_source_distinct_ids"]
        self.assertEqual((by["c"]["kept"], by["d"]["kept"]), (3, 2))
        self.assertEqual(self.result["kept"], {"c": [0, 3], "d": [1, 2]})

    def test_refilter_from_a_version_2_refiltered_run(self):
        run = c2.load_run(self.v123)
        self.assertEqual(run["filters_version"], 3)
        self.assertEqual(run["refilter"]["source_run_dir"], "labels/main-f2")
        self.assertEqual(run["refilter"]["source_filters_version"], 2)
        self.assertEqual(run["refilter"]["source_refilter"], c2.load_run(self.v12)["refilter"])
        self.assertNotIn("source_refilter", c2.load_run(self.out)["refilter"])     # a labeled source has none
        self.assertEqual(c2.contract(run), c2.contract(c2.load_run(self.v1)) | {"filters_version": 3})

    def test_the_source_run_is_untouched(self):
        self.assertEqual(files(self.v1), self.v1_files)

    def test_refuses_a_source_whose_stored_text_is_not_its_versions_mapping(self):
        """The stored text must be the source version's mapping of the stored raw label (here: a version 1 run whose
        text was written under version 3's mapping)."""
        bad = self.root / "labels" / "main-badtext"
        shutil.copytree(self.v1, bad)
        p = bad / "shards" / "00000.labels.parquet"
        rows = table(p)
        rows[0]["text"] = filters.map_text(rows[0]["raw"], version=3)
        self.assertNotEqual(rows[0]["text"], rows[0]["raw"])
        sha = c2._write_parquet(p, rows, c2.label_schema())
        meta_path = p.with_name("00000.json")
        meta_path.write_text(json.dumps(json.loads(meta_path.read_text()) | {"sha256": sha}))
        m = json.loads((bad / "manifest.json").read_text())
        m["shards"][0]["sha256"] = sha
        (bad / "manifest.json").write_text(json.dumps(m))
        with self.assertRaisesRegex(RuntimeError, "stored label text"):
            c2.refilter(bad, coverage=b0_like)


if __name__ == "__main__":
    unittest.main()
