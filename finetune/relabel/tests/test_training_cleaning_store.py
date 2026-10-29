"""Contract tests (CPU): label cleaning through the label store and the layer 2 runner (cleaning plan "Where it
applies", "Layer 2", Checks 1-5), on real C1 shards and C2 label records (training_fixtures) with test tables: spans
and prompt dedup, choose (injected scorer: resumable, atomic blocks, probe, determinism), the artifact and its
refusals, LabelStore with and without cleaning, and the net-harm check."""
from __future__ import annotations

import hashlib
import os
import json
import re
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from labeling import c2, shards
from labeling_fixtures import relabel_root
from training import cleaning as C
from training import cleaning_run as R
from training import labels as tlabels
from training_fixtures import TOTAL, LabeledWorld

TABLES = {"forms.tsv": "b0\tB0\n", "variants.tsv": "", "dialect.tsv": "", "choices.tsv": "human\thumane\n"}


def number(user: str) -> int | None:
    line = next(ln for ln in user.splitlines() if ln.startswith("Transcript: "))
    m = re.findall(r"\d+", line)
    return None if not m else int(m[-1])


def fake_probs(user: str, n: int) -> list[float]:
    """"human xY k": k % 3 == 0 -> "humane" (0.95); 1 -> can't tell; 2 -> below the threshold; no number ->
    "human"."""
    k = number(user)
    if k is None:
        return [0.96, 0.02, 0.02]
    return [[0.02, 0.95, 0.03], [0.1, 0.1, 0.8], [0.3, 0.6, 0.1]][k % 3]


class FakeScorer:
    def __init__(self, engine="fake", fail_after=None):
        self.engine, self.fail_after, self.calls, self.loads = engine, fail_after, 0, 0

    def load(self):
        self.loads += 1

    def info(self):
        return {"alias": "fake-textgen", "revision": "r1", "engine": self.engine}

    def score(self, users, n_options):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise KeyboardInterrupt("interrupted")
        self.calls += 1
        return [fake_probs(u, n) for u, n in zip(users, n_options)]


class World(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.w = LabeledWorld(cls.root)
        assert c2.finish(cls.w.labels_dir)["complete"]
        cls.tables = cls.write_tables(cls.root / "tables", TABLES)
        cls.work = cls.root / "cleaning" / "fixture" / "v1"
        cls.spans = R.spans(cls.w.labels_dir, cls.work, tables_dir=cls.tables)
        R.choose(cls.work, FakeScorer(), block=4, tables_dir=cls.tables, run_dir=cls.w.labels_dir)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    @staticmethod
    def write_tables(d: Path, texts: dict) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        for f, t in texts.items():
            (d / f).write_text(t)
        return d

    def spec(self, work=None, tables=None):
        return C.Cleaning.load(tables or self.tables, work or self.work)

    def store(self, label_set="T*", cleaning=None, run_dir=None, pool=None, through=TOTAL):
        return tlabels.LabelStore.open(run_dir or self.w.labels_dir, pool or self.w.pool, label_set=label_set,
                                       through_batch=through, cleaning=cleaning)

    def copy(self, name: str) -> Path:
        d = self.root / "copies" / name
        shutil.copytree(self.work, d)
        return d

    def rewrite_choices(self, work: Path, fn) -> None:
        path = work / "choices.parquet"
        t = pq.read_table(path)
        rows = fn(t.to_pylist())
        pq.write_table(pa.Table.from_pylist(rows, schema=t.schema), path)
        m = json.loads((work / "manifest.json").read_text())
        m["file_sha256"], m["rows"] = hashlib.sha256(path.read_bytes()).hexdigest(), len(rows)
        (work / "manifest.json").write_text(json.dumps(m))


class Spans(World):
    def test_counts_cover_both_label_sets_and_dedup_identical_prompts(self):
        sp = pq.read_table(self.work / "spans.parquet").to_pylist()
        prompts = pq.read_table(self.work / "prompts.parquet").to_pylist()
        want = 0
        for rec in self.w.records.values():
            if rec["disposition"] == "kept":
                want += len(re.findall(r"\bhuman\b", rec["text"])) + len(re.findall(r"\bhuman\b", rec["b0_raw"]))
        self.assertEqual(len(sp), want)
        self.assertEqual(self.spans["spans"], want)
        self.assertEqual(set(self.spans["spans_per_set"]["human|humane"]), {"T*", "B0"})
        self.assertEqual(self.spans["unique_prompts"], len({r["prompt_key"] for r in sp}))
        self.assertEqual(len(prompts), self.spans["unique_prompts"])
        texts = {}
        for r in sp:
            rec = self.w.records[r["id"]]
            text = rec["text"] if r["label_set"] == "T*" else rec["b0_raw"]
            self.assertEqual(text[r["start"]:r["end"]], r["original"])
            texts.setdefault(r["prompt_key"], set()).add(text)
        self.assertTrue(all(len(v) == 1 for v in texts.values()))        # one key, one text
        for p in prompts:
            self.assertEqual(p["prompt_key"], C.prompt_key(p["text"], p["start"], p["end"], ("human", "humane",
                                                                                           C.CANT_TELL)))
            self.assertIn(C.marked(p["text"], p["start"], p["end"]), p["user"])
        self.assertEqual(self.spans["tables_sha256"], C.Tables.load(self.tables).sha256)
        self.assertEqual(self.spans["contract_sha256"], R.Run(self.w.labels_dir).contract_sha256)


class Choose(World):
    def test_artifact_and_manifest(self):
        m = json.loads((self.work / "manifest.json").read_text())
        for key in ("version", "label_run", "contract_sha256", "tables_sha256", "template_sha256", "threshold",
                    "model", "rows", "file_sha256"):
            self.assertIn(key, m)
        self.assertEqual(m["model"], {"alias": "fake-textgen", "revision": "r1"})
        self.assertEqual(m["file_sha256"], hashlib.sha256((self.work / "choices.parquet").read_bytes()).hexdigest())
        rows = pq.read_table(self.work / "choices.parquet").to_pylist()
        self.assertEqual(m["rows"], len(rows))
        self.assertEqual(set(rows[0]), set(C.CHOICE_COLUMNS))
        for r in rows:
            rec = self.w.records[r["id"]]
            text = rec["text"] if r["label_set"] == "T*" else rec["b0_raw"]
            want = C.decide(fake_probs(C.prompt_user(text, r["start"], r["end"], tuple(r["options"])), 3),
                            tuple(r["options"]))
            self.assertEqual(r["chosen"], want)
        self.assertTrue({"humane", None} <= {r["chosen"] for r in rows})

    def test_resumable_atomic_blocks_and_one_engine(self):
        work = self.copy("resume")
        for p in list((work / "decisions").iterdir()) + [work / "choices.parquet", work / "manifest.json"]:
            p.unlink()
        with self.assertRaises(KeyboardInterrupt):
            R.choose(work, FakeScorer(fail_after=2), block=3, tables_dir=self.tables)
        self.assertEqual(len(list((work / "decisions").glob("*.parquet"))), 2)
        self.assertEqual([p for p in (work / "decisions").iterdir() if ".tmp" in p.name], [])
        self.assertFalse((work / "choices.parquet").exists())
        with self.assertRaises(RuntimeError):
            R.choose(work, FakeScorer(engine="other"), block=3, tables_dir=self.tables)
        s = FakeScorer()
        out = R.choose(work, s, block=3, tables_dir=self.tables, run_dir=self.w.labels_dir)
        self.assertEqual(out["left"], 0)
        total = self.spans["unique_prompts"]
        self.assertEqual(s.calls, -(-(total - 6) // 3))
        self.assertEqual(pq.read_table(work / "choices.parquet").to_pylist(),
                         pq.read_table(self.work / "choices.parquet").to_pylist())
        s2 = FakeScorer()
        R.choose(work, s2, block=3, tables_dir=self.tables, run_dir=self.w.labels_dir)
        self.assertEqual((s2.calls, s2.loads), (0, 0))                   # nothing left: no model load

    def test_probe_writes_no_decisions(self):
        work = self.copy("probe")
        before = sorted(p.name for p in (work / "decisions").iterdir())
        r = R.probe(work, FakeScorer(), n=5, tables_dir=self.tables)
        self.assertEqual(r["prompts"], 5)
        self.assertEqual(r["unique_prompts"], self.spans["unique_prompts"])
        self.assertIn("extrapolated_hours", r)
        self.assertIn("host_maxrss_gib", r["peak"])
        self.assertTrue((work / "probe.json").exists())
        self.assertEqual(sorted(p.name for p in (work / "decisions").iterdir()), before)

    def test_determinism_rescore(self):
        r = R.determinism(self.copy("det"), FakeScorer(), fraction=1.0, tables_dir=self.tables)
        self.assertEqual(r["prompts"], self.spans["unique_prompts"])
        self.assertTrue(r["all_identical"])

    def test_reassembly_after_disabling_a_set_counts_only_current_prompts(self):
        both = self.write_tables(self.root / "tables-two-sets",
                                 {**TABLES, "choices.tsv": "human\thumane\npseudo\tsudo\n"})
        work = self.root / "cleaning" / "two-sets" / "v1"
        R.spans(self.w.labels_dir, work, tables_dir=both)
        R.choose(work, FakeScorer(), block=4, tables_dir=both, run_dir=self.w.labels_dir)
        spans = R.spans(self.w.labels_dir, work, tables_dir=self.tables)       # "pseudo" disabled
        self.assertGreater(len(R._decided(work)[0]), spans["unique_prompts"])  # cached decisions of the removed set
        m = R.assemble(work, self.w.labels_dir, tables_dir=self.tables)
        self.assertEqual(m["unique_prompts"], spans["unique_prompts"])
        self.assertEqual(pq.read_table(work / "choices.parquet").to_pylist(),
                         pq.read_table(self.work / "choices.parquet").to_pylist())

    def test_spans_of_other_tables_are_refused(self):
        other = self.write_tables(self.root / "tables-other", {**TABLES, "forms.tsv": "b0\tB0\n# changed\n"})
        with self.assertRaises(RuntimeError):
            R.choose(self.copy("other-tables"), FakeScorer(), tables_dir=other)


class Store(World):
    def test_none_reproduces_the_uncleaned_texts_and_identity(self):
        a = self.store("T*")
        b = tlabels.LabelStore.open(self.w.labels_dir, self.w.pool, label_set="T*", through_batch=TOTAL)
        self.assertEqual(a._texts, b._texts)
        self.assertEqual(json.dumps(a.identity, sort_keys=True), json.dumps(b.identity, sort_keys=True))
        self.assertNotIn("cleaning", a.identity)
        want = hashlib.sha256(json.dumps({k: a.identity[k] for k in ("label_set", "column", "contract_sha256",
                                                                      "shards")}, sort_keys=True).encode()).hexdigest()
        self.assertEqual(a.identity["sha256"], want)
        for i, rec in self.w.records.items():
            self.assertEqual(a.text(i), rec["text"] if rec["disposition"] == "kept" else None)

    def test_cleaned_texts_identity_and_kept_set_for_both_label_sets(self):
        spec = self.spec()
        for ls, col in (("T*", "text"), ("B0", "b0_raw")):
            plain, clean = self.store(ls), self.store(ls, spec)
            changed = 0
            for i, rec in self.w.records.items():
                self.assertEqual(clean.kept(i), plain.kept(i))
                if rec["disposition"] != "kept":
                    self.assertIsNone(clean.text(i))
                    continue
                want = spec.clean(rec[col], i, ls)
                self.assertEqual(clean.text(i), want)
                self.assertIsNotNone(clean.text(i))
                changed += want != rec[col]
            self.assertGreater(changed, 0, ls)
            self.assertEqual(clean.identity["cleaning"], spec.identity())
            self.assertEqual(clean.identity["cleaned_labels"], changed)
            self.assertNotEqual(clean.identity["sha256"], plain.identity["sha256"])
            hashed = {k: clean.identity[k] for k in ("label_set", "column", "contract_sha256", "shards")}
            hashed["cleaning"] = spec.identity()
            self.assertEqual(clean.identity["sha256"],
                             hashlib.sha256(json.dumps(hashed, sort_keys=True).encode()).hexdigest())
        b0 = self.store("B0", spec)
        self.assertTrue(any(b0.text(i).startswith("B0 humane") for i in self.w.records if b0.kept(i)))

    def test_identity_follows_the_tables_and_the_artifact(self):
        base = self.store("T*", self.spec()).identity["sha256"]
        layer1 = self.store("T*", C.Cleaning.load(self.tables)).identity
        self.assertIsNone(layer1["cleaning"]["choices_sha256"])
        self.assertNotEqual(layer1["sha256"], base)
        work = self.copy("flip")
        self.rewrite_choices(work, lambda rows: [r | {"chosen": None} for r in rows])
        self.assertNotEqual(self.store("T*", self.spec(work)).identity["sha256"], base)

    def test_an_empty_kept_label_stays_empty_and_kept(self):
        d = self.root / "copies" / "empty-b0"
        pool, lab = d / "pool", d / "labels"
        shutil.copytree(self.w.pool, pool)
        shutil.copytree(self.w.labels_dir, lab)
        run = json.loads((lab / "run.json").read_text())
        run["pool_dir"] = str(pool.relative_to(self.root))
        (lab / "run.json").write_text(json.dumps(run))
        path = lab / "shards" / f"{shards.shard_name(0)}.labels.parquet"
        t = pq.read_table(path)
        rows = t.to_pylist()
        k = next(n for n, r in enumerate(rows) if r["disposition"] == "kept")
        rows[k]["b0_raw"] = ""
        pq.write_table(pa.Table.from_pylist(rows, schema=t.schema), path)
        meta = json.loads(path.with_name(f"{shards.shard_name(0)}.json").read_text())
        meta["sha256"] = shards.sha256_file(path)
        path.with_name(f"{shards.shard_name(0)}.json").write_text(json.dumps(meta))
        s = tlabels.LabelStore.open(lab, pool, label_set="B0", through_batch=TOTAL,
                                    cleaning=C.Cleaning.load(self.tables))
        self.assertEqual(s.text(rows[k]["id"]), "")
        self.assertTrue(s.kept(rows[k]["id"]))

    def test_refuses_an_artifact_of_another_contract(self):
        work = self.copy("contract")
        m = json.loads((work / "manifest.json").read_text())
        m["contract_sha256"] = "0" * 64
        (work / "manifest.json").write_text(json.dumps(m))
        with self.assertRaises(C.ArtifactMismatch):
            self.store("T*", self.spec(work))

    def test_refuses_an_artifact_of_other_tables(self):
        other = self.write_tables(self.root / "tables-store", {**TABLES, "dialect.tsv": "gonna\tgoing to\n"})
        with self.assertRaises(C.ArtifactMismatch):
            self.spec(tables=other)

    def test_refuses_a_recorded_original_that_differs_from_the_label(self):
        work = self.copy("original")
        self.rewrite_choices(work, lambda rows: [r | ({"original": "Human"} if n == 0 else {})
                                                 for n, r in enumerate(rows)])
        ls = pq.read_table(work / "choices.parquet").to_pylist()[0]["label_set"]
        with self.assertRaises(C.ArtifactMismatch):
            self.store(ls, self.spec(work))

    def test_refuses_a_choice_outside_the_set_and_missing_spans(self):
        work = self.copy("member")
        self.rewrite_choices(work, lambda rows: [r | {"chosen": "tex"} for r in rows])
        with self.assertRaises(C.ArtifactMismatch):
            self.store("T*", self.spec(work))
        work = self.copy("missing")
        self.rewrite_choices(work, lambda rows: [r for r in rows if r["label_set"] != "T*"])
        with self.assertRaises(C.ArtifactMismatch):
            self.store("T*", self.spec(work))

    def test_refuses_a_damaged_artifact_file(self):
        work = self.copy("damaged")
        p = work / "choices.parquet"
        p.write_bytes(p.read_bytes() + b"x")
        with self.assertRaises(C.ArtifactMismatch):
            self.spec(work)
        work = self.copy("rows")
        m = json.loads((work / "manifest.json").read_text())
        m["rows"] += 1
        (work / "manifest.json").write_text(json.dumps(m))
        with self.assertRaises(C.ArtifactMismatch):
            self.spec(work)


class Check(World):
    def test_net_harm_screen_wer_and_report(self):
        with mock.patch.object(R, "MIN_JUDGED", 1):
            r = R.check(self.w.labels_dir, self.work, tables_dir=self.tables)
        g = r["check3"]["groups"]["choices:human|humane"]
        t = g["cells"]["T*"]
        self.assertGreater(t["aligned"], 0)
        self.assertEqual(t["original_matches"], t["aligned"])      # T* writes the reference: every change harms
        self.assertEqual(t["cleaned_matches"], 0)
        self.assertTrue(g["disabled"])
        self.assertIn("choices:human|humane", r["check3"]["disabled"])
        self.assertNotIn("forms:B0", r["check3"]["groups"])         # casing is audited, not screened
        w = r["corpus_wer_human_reference_labels"]["T*"]
        self.assertEqual(w["before"], 0.0)
        self.assertGreater(w["after"], 0.0)
        c5 = r["check5"]
        found = sum(v["found"] for v in c5["layer2"]["human|humane"].values())
        self.assertEqual(found, self.spans["spans"])
        for v in c5["layer2"]["human|humane"].values():
            self.assertEqual(v["found"], v.get("chosen", 0) + v.get("left_cant_tell", 0) +
                             v.get("left_below_threshold", 0))
        self.assertTrue(set(c5["labels_changed"]) <= {"T*", "B0"})
        self.assertTrue((self.work / "check.json").exists())
        sample = (C.audit_dir() / "check-sample.tsv").read_text().splitlines()
        self.assertGreater(len(sample), 1)
        self.assertLessEqual(len(sample) - 1, R.AUDIT_SAMPLE)

    def test_floor_reports_without_judging(self):
        r = R.check(self.w.labels_dir, self.work, tables_dir=self.tables, out_dir=self.root / "floor")
        g = r["check3"]["groups"]["choices:human|humane"]
        self.assertTrue(g["reported_not_judged"])
        self.assertFalse(g["disabled"])


ACRONYM_T, ACRONYM_B0 = "the A. P. I. of {}", "b0 the A P I of {}"


class Versions(World):
    """Cleaning plan v5: version 2 adds spelled-out acronyms to layer 1 for both label sets; version 1 (stages 2b and
    2c) reads today's texts; the version is in the store identity, the artifact and the runner's outputs."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        d = cls.root / "copies" / "acronym"
        cls.pool, cls.lab = d / "pool", d / "labels"
        shutil.copytree(cls.w.pool, cls.pool)
        shutil.copytree(cls.w.labels_dir, cls.lab)
        run = json.loads((cls.lab / "run.json").read_text())
        run["pool_dir"] = str(cls.pool.relative_to(cls.root))
        (cls.lab / "run.json").write_text(json.dumps(run))
        cls.raw = {}
        manifest = json.loads((cls.lab / "manifest.json").read_text())
        for k in range(2):                       # one kept label with a human reference in each of two label shards
            path = cls.lab / "shards" / f"{shards.shard_name(k)}.labels.parquet"
            t = pq.read_table(path)
            rows = t.to_pylist()
            n = next(n for n, r in enumerate(rows) if r["disposition"] == "kept" and r["reference_kind"] == "human")
            rows[n]["text"], rows[n]["b0_raw"] = ACRONYM_T.format(k), ACRONYM_B0.format(k)
            cls.raw[rows[n]["id"]] = (rows[n]["text"], rows[n]["b0_raw"])
            pq.write_table(pa.Table.from_pylist(rows, schema=t.schema), path)
            meta_path = path.with_name(f"{shards.shard_name(k)}.json")
            meta = json.loads(meta_path.read_text())
            meta["sha256"] = shards.sha256_file(path)
            meta_path.write_text(json.dumps(meta))
            for s in manifest["shards"]:
                if s["file"] == path.name:
                    s["sha256"] = meta["sha256"]
        (cls.lab / "manifest.json").write_text(json.dumps(manifest))

    def open(self, label_set, cleaning):
        return tlabels.LabelStore.open(self.lab, self.pool, label_set=label_set, through_batch=TOTAL,
                                       cleaning=cleaning)

    def test_store_texts_and_identity_per_version(self):
        for ls, col in (("T*", 0), ("B0", 1)):
            plain = self.open(ls, None)
            v1 = self.open(ls, C.Cleaning.load(self.tables, version=1))
            v2 = self.open(ls, C.Cleaning.load(self.tables, version=2))
            for i, texts in self.raw.items():
                self.assertEqual(plain.text(i), texts[col])
                self.assertEqual(v1.text(i), C.Cleaning.load(self.tables, version=1).clean(texts[col], i, ls))
                self.assertIn("A. P. I." if ls == "T*" else "A P I", v1.text(i))         # version 1: as written
                self.assertIn("the API of", v2.text(i))
            for i in plain._texts:
                self.assertEqual((v1.kept(i), v2.kept(i)), (plain.kept(i), plain.kept(i)))
                if i not in self.raw:
                    self.assertEqual(v2.text(i), v1.text(i))
            self.assertEqual((v1.identity["cleaning"]["version"], v2.identity["cleaning"]["version"]), (1, 2))
            self.assertEqual(v2.identity["cleaned_labels"] - v1.identity["cleaned_labels"],
                             len(self.raw) if ls == "T*" else 0)    # B0 labels are all cleaned already ("b0")
            self.assertEqual(len({plain.identity["sha256"], v1.identity["sha256"], v2.identity["sha256"]}), 3)

    def test_artifact_per_version_reuses_the_layer2_decisions(self):
        work = self.root / "cleaning" / "fixture" / "v1-of-version-test"
        R.spans(self.w.labels_dir, work, tables_dir=self.tables, version=1)
        shutil.copytree(self.work / "decisions", work / "decisions")
        m1 = R.assemble(work, self.w.labels_dir, tables_dir=self.tables, version=1)
        self.assertEqual(m1["version"], 1)
        self.assertEqual(json.loads((work / "spans.json").read_text())["version"], 1)
        self.assertEqual(C.Cleaning.load(self.tables, work, version=1).identity()["version"], 1)
        with self.assertRaises(C.ArtifactMismatch):
            C.Cleaning.load(self.tables, work, version=2)
        with self.assertRaises(RuntimeError):                       # spans of another version
            R.assemble(work, self.w.labels_dir, tables_dir=self.tables, version=2)
        self.assertEqual(pq.read_table(work / "choices.parquet").to_pylist(),
                         pq.read_table(self.work / "choices.parquet").to_pylist())

    def test_check_screens_the_class_and_reserves_its_audit_share(self):
        work = self.root / "cleaning" / "acronym" / "v2"
        with mock.patch.object(R, "AUDIT_SAMPLE", 6), mock.patch.object(R, "AUDIT_ACRONYMS", 2):
            r = R.check(self.lab, work, out_dir=work, layer1_only=True, tables_dir=self.tables, version=2)
        self.assertEqual(r["version"], 2)
        g = r["check3"]["groups"]["acronyms:spelled-out"]
        self.assertEqual({k: g["cells"][k]["changed"] for k in ("T*", "B0")}, {"T*": 2, "B0": 2})
        self.assertEqual(r["check5"]["edits_per_group"]["acronyms:spelled-out"], {"T*": 2, "B0": 2})
        a = r["acronyms"]
        self.assertEqual(a["changed_spans"], {"T*": 2, "B0": 2})
        self.assertEqual(sum(a["compact"]["API"].values()), 4)
        sample = (C.audit_dir(2) / "check-sample-layer1.tsv").read_text().splitlines()[1:]
        tables = [row.split("\t")[2] for row in sample]
        self.assertEqual(len(sample), 6)
        self.assertEqual(tables.count("acronyms"), 2)
        sheet = (C.audit_dir(2) / "acronyms-layer1.txt").read_text()
        self.assertIn("changed spans: T* 2, B0 2", sheet)
        self.assertIn("API: T* 2, B0 2", sheet)
        self.assertTrue((work / "check-layer1.json").exists())
        v1 = R.check(self.lab, work, out_dir=self.root / "v1-check", layer1_only=True, tables_dir=self.tables,
                     version=1)
        self.assertNotIn("acronyms:spelled-out", v1["check3"]["groups"])
        self.assertNotIn("acronyms", v1)

    def test_the_acronym_class_is_reported_not_judged(self):
        """Cleaning plan v6 (AJ): Check 3 reports the class, never disables it, even where the original matches the
        references more often."""
        screen = {"acronyms:spelled-out": {"T*": {"aligned": 40, "original_matches": 39, "cleaned_matches": 1}},
                  "dialect:gonna": {"T*": {"aligned": 40, "original_matches": 39, "cleaned_matches": 1}}}
        from collections import Counter
        v = R.judge({g: {ls: Counter(c) for ls, c in cells.items()} for g, cells in screen.items()})
        self.assertEqual((v["acronyms:spelled-out"]["disabled"], v["acronyms:spelled-out"]["reported_not_judged"]),
                         (False, True))
        self.assertEqual(v["acronyms:spelled-out"]["original_matches_more_in"], ["T*", "both"])
        self.assertTrue(v["dialect:gonna"]["disabled"])


L0_T, L0_B0 = "so we went home then we slept {}", "So, we went home. Then we slept {}."


class Layer0Store(World):
    """Cleaning plan v6 "Layer 0" through the label store: version 2 cleans T*'s labels with the same record's b0_raw
    as the donor; B0's labels and version 1 are untouched; the runner audits 50 layer 0 edits and never screens them."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        d = cls.root / "copies" / "layer0"
        cls.pool, cls.lab = d / "pool", d / "labels"
        shutil.copytree(cls.w.pool, cls.pool)
        shutil.copytree(cls.w.labels_dir, cls.lab)
        run = json.loads((cls.lab / "run.json").read_text())
        run["pool_dir"] = str(cls.pool.relative_to(cls.root))
        (cls.lab / "run.json").write_text(json.dumps(run))
        cls.raw = {}
        manifest = json.loads((cls.lab / "manifest.json").read_text())
        for k in range(2):
            path = cls.lab / "shards" / f"{shards.shard_name(k)}.labels.parquet"
            t = pq.read_table(path)
            rows = t.to_pylist()
            n = next(n for n, r in enumerate(rows) if r["disposition"] == "kept" and r["reference_kind"] == "human")
            rows[n]["text"], rows[n]["b0_raw"] = L0_T.format(k), L0_B0.format(k)
            cls.raw[rows[n]["id"]] = (rows[n]["text"], rows[n]["b0_raw"])
            pq.write_table(pa.Table.from_pylist(rows, schema=t.schema), path)
            meta_path = path.with_name(f"{shards.shard_name(k)}.json")
            meta = json.loads(meta_path.read_text())
            meta["sha256"] = shards.sha256_file(path)
            meta_path.write_text(json.dumps(meta))
            for s in manifest["shards"]:
                if s["file"] == path.name:
                    s["sha256"] = meta["sha256"]
        (cls.lab / "manifest.json").write_text(json.dumps(manifest))

    def open(self, label_set, version):
        return tlabels.LabelStore.open(self.lab, self.pool, label_set=label_set, through_batch=TOTAL,
                                       cleaning=C.Cleaning.load(self.tables, version=version))

    def test_t_star_takes_b0s_marks_and_b0_is_untouched(self):
        t2, t1, b2 = self.open("T*", 2), self.open("T*", 1), self.open("B0", 2)
        for k, (i, (t, b)) in enumerate(sorted(self.raw.items())):
            self.assertEqual(t2.text(i), f"so, we went home. Then we slept {t.split()[-1]}.")
            self.assertEqual(t1.text(i), t)
            self.assertEqual(b2.text(i), b)
        self.assertEqual(t2.identity["cleaning"]["layer0"]["rule"], "union")
        self.assertNotEqual(t2.identity["sha256"], t1.identity["sha256"])

    def test_the_runner_audits_layer0_and_never_screens_it(self):
        work = self.root / "cleaning" / "layer0" / "v2"
        with mock.patch.object(R, "AUDIT_SAMPLE", 4), mock.patch.object(R, "AUDIT_ACRONYMS", 1), \
                mock.patch.object(R, "AUDIT_PUNCTUATION", 3):
            r = R.check(self.lab, work, out_dir=work, layer1_only=True, tables_dir=self.tables, version=2)
        self.assertEqual(r["check5"]["edits_per_group"]["punctuation:mark"], {"T*": 6})
        self.assertEqual(r["check5"]["edits_per_group"]["punctuation:capital"], {"T*": 2})
        self.assertFalse(any(g.startswith("punctuation") for g in r["check3"]["groups"]))
        sample = (C.audit_dir(2) / "check-sample-layer1.tsv").read_text().splitlines()[1:]
        self.assertEqual(sum(row.split("\t")[2] == "punctuation" for row in sample), 3)


class AcronymRisks(unittest.TestCase):
    """The audit counts the residual-risk classes the plan names (review p4) among the class's changed spans."""

    def risks(self, text: str) -> list[str]:
        spec = C.Cleaning(C.Tables({"forms.tsv": "", "variants.tsv": "", "dialect.tsv": "", "choices.tsv": ""}))
        out, edits = spec.edits(text, "i", "T*")
        self.assertEqual(len(edits), 1, text)
        return R.acronym_risks(text, edits[0])

    def test_named_classes(self):
        self.assertEqual(R.acronym_shape("G. P. U. s"), (3, "dotted+spaced", True, True))
        self.assertEqual(R.acronym_shape("A P I"), (3, "spaced", False, False))
        self.assertTrue(any(r.startswith("dotted, 6+") for r in self.risks("it's I.L.L.U.S.T.R.I.O.U.S. there")))
        self.assertTrue(any(r.startswith("spelled word") for r in self.risks("cat, that's C A T")))
        self.assertTrue(any(r.startswith("variables") for r in self.risks("A I J equals the sum")))
        self.assertTrue(any(r.startswith("notes") and "music" in r for r in self.risks("the chord E G D E G D")))
        self.assertTrue(any(r.startswith("two letters") for r in self.risks("and I. A lot of it")))
        self.assertTrue(any(r.startswith("mixed") for r in self.risks("than I. A P I is")))
        self.assertTrue(any(r.startswith("the whole label") for r in self.risks("M. I. N. E.")))
        self.assertTrue(any(r.startswith("a single lowercase") for r in self.risks("uh L. C. d. screen")))
        self.assertEqual(self.risks("the A. P. I. call"), [])
        # statistics of a composed edit come from its run as written, not the table words merged into it (review r1)
        spec = C.Cleaning(C.Tables({"forms.tsv": "chi squared\tchi-squared\n", "variants.tsv": "", "dialect.tsv": "",
                                    "choices.tsv": ""}))
        text = "the C H I squared test"
        e = spec.edits(text, "i", "T*")[1][0]
        self.assertEqual((e.original, e.run), ("C H I squared", "C H I"))
        self.assertEqual(R.acronym_shape(e.run), (3, "spaced", False, False))


class Lease(unittest.TestCase):
    """GPU commands respect the agent-jobs VRAM lease (AGENT_JOBS_VRAM_FRACTION) before any model is loaded."""

    def env(self, lease):
        return mock.patch.dict(os.environ, {} if lease is None else {R.LEASE_ENV: str(lease)}, clear=False)

    def setUp(self):
        self._saved = os.environ.pop(R.LEASE_ENV, None)

    def tearDown(self):
        if self._saved is not None:
            os.environ[R.LEASE_ENV] = self._saved

    def test_vllm_utilization_defaults_to_the_lease_and_refuses_more(self):
        with self.env(None):
            self.assertEqual(R.vllm_utilization(None), R.DEFAULT_GPU_MEMORY_UTILIZATION)
            self.assertEqual(R.vllm_utilization(0.95), 0.95)
        with self.env(0.5):
            self.assertEqual(R.vllm_utilization(None), 0.5)
            self.assertEqual(R.vllm_utilization(0.4), 0.4)
            with self.assertRaises(RuntimeError):
                R.vllm_utilization(0.85)
        with self.env(0.95):
            self.assertEqual(R.vllm_utilization(None), R.DEFAULT_GPU_MEMORY_UTILIZATION)

    def test_transformers_fraction_is_the_declared_vram_capped_by_the_lease(self):
        with self.env(None):
            self.assertIsNone(R.transformers_fraction(None, 32.0))
            self.assertEqual(R.transformers_fraction(8.0, 32.0), 0.25)
        with self.env(0.5):
            self.assertEqual(R.transformers_fraction(None, 32.0), 0.5)
            self.assertEqual(R.transformers_fraction(16.0, 32.0), 0.5)
            with self.assertRaises(RuntimeError):
                R.transformers_fraction(20.0, 32.0)

    def test_engines_refuse_before_loading(self):
        with self.env(0.5):
            with self.assertRaises(RuntimeError):
                R.make_scorer("vllm", 0.85)
            with mock.patch.object(R, "_gpu_total_gb", lambda: 32.0), self.assertRaises(RuntimeError):
                R.make_scorer("transformers", declared_vram_gb=20.0)
            self.assertEqual(R.make_scorer("vllm").gpu_memory_utilization, 0.5)
            with self.assertRaises(RuntimeError):                  # refused before vLLM is imported
                R.VllmScorer(0.85).load()


class GpuCommands(World):
    """choose / probe / determinism load and run the model only while holding the parakeet GPU lock."""

    def run_cmd(self, work, events, scorer=None, busy=False, **kw):
        import contextlib

        import score_teacher

        @contextlib.contextmanager
        def lock(label, wait_s):
            if busy:
                raise score_teacher.LockBusy("held")
            events.append(("lock", label))
            try:
                yield
            finally:
                events.append(("unlock", label))
        s = scorer or EventScorer(events)
        a = types.SimpleNamespace(engine="vllm", gpu_memory_utilization=None, declared_vram_gb=None, probe=None,
                                  determinism=None, budget_minutes=None, labels_run="fixture", lock_wait=1.0)
        vars(a).update(kw)
        with mock.patch.object(score_teacher, "gpu_lock", lock), mock.patch.object(R, "make_scorer",
                                                                                  lambda *x, **y: s), \
                mock.patch.object(C, "TABLES_DIR", self.tables):
            return R.gpu_command(a, work, self.w.labels_dir)

    def test_model_runs_inside_the_lock(self):
        for kw in ({"probe": 3}, {"determinism": 1.0}, {}):
            work = self.copy(f"gpu-{len(kw)}-{list(kw)}")
            if not kw:
                for p in list((work / "decisions").iterdir()):
                    p.unlink()
            events = []
            self.assertEqual(self.run_cmd(work, events, **kw), 0)
            self.assertEqual(events[0][0], "lock", kw)
            self.assertEqual(events[-1][0], "unlock", kw)
            inner = [e for e in events[1:-1]]
            self.assertIn("load", inner, kw)
            self.assertTrue(all(e[0] not in ("lock", "unlock") for e in inner))

    def test_a_busy_lock_runs_nothing(self):
        import score_teacher
        events = []
        self.assertEqual(self.run_cmd(self.copy("busy"), events, busy=True, probe=3), score_teacher.LOCK_BUSY_EXIT)
        self.assertEqual(events, [])

    def test_a_lease_refusal_takes_no_lock(self):
        events = []
        a = types.SimpleNamespace(engine="vllm", gpu_memory_utilization=0.9, declared_vram_gb=None, probe=3,
                                  determinism=None, budget_minutes=None, labels_run="fixture", lock_wait=1.0)
        import score_teacher
        with mock.patch.dict(os.environ, {R.LEASE_ENV: "0.5"}), \
                mock.patch.object(score_teacher, "gpu_lock", lambda *x: events.append("lock")), \
                self.assertRaises(RuntimeError):
            R.gpu_command(a, self.copy("lease"), self.w.labels_dir)
        self.assertEqual(events, [])


class EventScorer(FakeScorer):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def load(self):
        self.events.append("load")
        super().load()

    def score(self, users, n_options):
        self.events.append("score")
        return super().score(users, n_options)


class Canonical(unittest.TestCase):
    """Cleaning plan v4: a pure group's canonical form is T*'s majority form over all kept labels (case-insensitive
    count per form), in T*'s most frequent casing, ties to the term list's first form; B0 and the human references
    do not vote."""

    def test_t_star_majority_wins_and_b0_does_not_vote(self):
        written = {"ok": {"T* OK": 1344, "T* ok": 128, "T* Ok": 23, "B0 OK": 90000},
                   "okay": {"T* Okay": 68096, "T* okay": 44368, "T* OKAY": 3}}
        self.assertEqual(C_run_canonical(["ok", "okay"], written), ("okay", {"ok": 1495, "okay": 112467}))

    def test_casing_is_t_stars_with_sentence_capitals_folded(self):
        self.assertEqual(C_run_canonical(["ptsd", "p t s d"], {"ptsd": {"T* PTSD": 477, "T* ptsd": 30}})[0], "PTSD")
        self.assertEqual(C_run_canonical(["dataset", "data set"],
                                         {"dataset": {"T* Dataset": 50, "T* dataset": 40},
                                          "data set": {"T* data set": 60}})[0], "dataset")

    def test_ties_go_to_the_first_form(self):
        self.assertEqual(C_run_canonical(["heap sort", "heapsort"], {})[0], "heap sort")
        self.assertEqual(C_run_canonical(["a b", "ab"], {"a b": {"T* a b": 3}, "ab": {"T* ab": 3}})[0], "a b")


def C_run_canonical(group, written):
    return R.canonical_form(group, written)


class Screen(unittest.TestCase):
    @staticmethod
    def norm(text: str) -> str:
        return " ".join(re.findall(r"[a-z']+", text.lower()))

    def screen(self, ref, before, after, start, end, out_start, out_end):
        e = C.Edit(2, "choices", "g", start, end, before[start:end], after[out_start:out_end], out_start, out_end)
        return R.screen_edit(self.norm, ref.split(), before, after, e)

    def test_joined_and_split_forms_are_judged_as_whole_token_sequences(self):
        # a correct "may be" changed to "maybe" is harm; the reverse is a fix
        self.assertEqual(self.screen("it may be so", "It may be so", "It maybe so", 3, 9, 3, 8), (True, False))
        self.assertEqual(self.screen("it may be so", "It maybe so", "It may be so", 3, 8, 3, 9), (False, True))

    def test_substitutions_and_deletions_count_as_misses(self):
        self.assertEqual(self.screen("it might be so", "It may be so", "It maybe so", 3, 9, 3, 8), (False, False))
        self.assertEqual(self.screen("set it up now", "set up now", "setup now", 0, 6, 0, 5), (False, False))

    def test_unmappable_spans(self):
        self.assertIsNone(self.screen("x", "a , b", "a , b", 2, 3, 2, 3))


if __name__ == "__main__":
    unittest.main()
