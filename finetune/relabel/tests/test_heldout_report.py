"""heldout_report contracts (heldout-panel plan "Models and records" and "Reported"):

- each model's record of each set is read from the plan's location (existing records reused, the gaps from
  $RELABEL_ROOT/heldout-panel/<model>/), the reference rows from the existing dev, AMI SDM and CV records;
- a cell counts only if its record is of the expected model (identity), on the current manifest, unlimited and complete;
  otherwise, or without a record, it is "-"; each counted cell names its record, the record's SHA-256, the model
  identity and the code commit when the record has one;
- panel means are unweighted means of the panel's set WERs, "-" if any set is "-"; H4 likewise over its four sets;
- differences are A - B per set and per panel mean, "-" if either side is "-";
- error profile: substitution, deletion and insertion rates (per 100 reference words);
- term recall on eval-stem is recomputed from the stored hypotheses as score_domain computes it;
- sets whose counted records disagree on utterances or reference words are flagged.

CPU only. Run from finetune/relabel:  ./python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import shutil
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import evaluate  # noqa: E402
import heldout_panel as hp  # noqa: E402
import heldout_report as hr  # noqa: E402
import relabel_paths  # noqa: E402
import score_domain  # noqa: E402
import terms  # noqa: E402

TMP = relabel_paths.ARTIFACTS / "tmp" / "tests" / "heldout-report"
R = relabel_paths.ARTIFACTS
WERS = {"tech-dev-1": 10.0, "tech-dev-2": 9.0, "stem-chem": 4.0, "stem-cs": 5.0, "stem-math": 8.0,
        "med-dev-1": 12.0, "med-dev-2": 19.0, "med-dev-3": 11.0, "lectures-a": 4.0, "medical-b": 14.0}
REF = {"librispeech_dev_clean": 1.6, "librispeech_dev_other": 2.9, "ami_dev": 12.1, "voxpopuli_dev": 5.8,
       "yodas_dev": 5.5, "ami-sdm-dev": 25.3, "cv27-en-dev": 7.1}


def cell(wer: float, words: int = 1000, utterances: int = 100, s: int = 50, d: int = 30, i: int = 20) -> dict:
    return {"status": "ok", "wer": wer, "substitutions": s, "deletions": d, "insertions": i, "ref_words": words,
            "utterances": utterances}


def full(shift: float = 0.0) -> dict:
    return {name: cell(w + shift) for name, w in {**WERS, **REF}.items()}


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


class TableTest(unittest.TestCase):
    def cells(self, **models) -> dict:
        out = {m: {} for m in hr.MODELS}
        out.update(models)
        return out

    def test_panel_means_are_unweighted_set_means(self) -> None:
        table = hr.tabulate(self.cells(B0=full()))
        panels = table["main"]["B0"]["panels"]
        self.assertAlmostEqual(panels["T5"], mean(WERS[s] for s in hp.PANELS["T5"]))
        self.assertAlmostEqual(panels["T5-no-tech-dev-2"], (10 + 4 + 5 + 8) / 4)
        self.assertAlmostEqual(panels["M3"], (12 + 19 + 11) / 3)
        self.assertAlmostEqual(panels["X2"], (4 + 14) / 2)
        self.assertEqual(table["main"]["B0"]["sets"], WERS)
        ref = table["main"]["B0"]["reference"]
        self.assertAlmostEqual(ref["H4"], (1.6 + 2.9 + 12.1 + 5.8) / 4)
        for name in ("yodas_dev", "ami-sdm-dev", "cv27-en-dev"):
            self.assertEqual(ref[name], REF[name])

    def test_a_missing_or_uncounted_set_makes_its_panel_and_differences_dash(self) -> None:
        b0p = full(-1.0)
        del b0p["lectures-a"]
        b0p["med-dev-2"] = {**b0p["med-dev-2"], "status": "not_comparable"}
        del b0p["ami_dev"]
        table = hr.tabulate(self.cells(B0=full(), B0p=b0p))
        main = table["main"]["B0p"]
        self.assertIsNone(main["sets"]["lectures-a"])
        self.assertIsNone(main["sets"]["med-dev-2"])
        self.assertIsNone(main["panels"]["X2"])
        self.assertIsNone(main["panels"]["M3"])
        self.assertIsNotNone(main["panels"]["T5"])
        self.assertIsNone(main["reference"]["H4"])
        self.assertEqual(main["reference"]["yodas_dev"], REF["yodas_dev"] - 1.0)
        diff = table["differences"]["B0p - B0"]
        self.assertIsNone(diff["sets"]["lectures-a"])
        self.assertIsNone(diff["panels"]["X2"])
        self.assertIsNone(diff["panels"]["M3"])
        self.assertAlmostEqual(diff["sets"]["medical-b"], -1.0)
        self.assertAlmostEqual(diff["panels"]["T5"], -1.0)

    def test_differences_are_the_plans_pairs_a_minus_b(self) -> None:
        cells = self.cells(**{"B0": full(), "b100-control": full(1.0), "b100-main": full(0.5), "B0p": full(-0.25),
                              "tstar": full(-2.0), "tstar-encoder": full(-1.5), "MB1": full(3.0)})
        table = hr.tabulate(cells)
        self.assertEqual(list(table["differences"]), ["b100-main - b100-control", "B0p - B0", "tstar - B0",
                                                      "tstar-encoder - B0", "tstar-encoder - tstar"])
        expect = {"b100-main - b100-control": -0.5, "B0p - B0": -0.25, "tstar - B0": -2.0,
                  "tstar-encoder - B0": -1.5, "tstar-encoder - tstar": 0.5}
        for key, value in expect.items():
            for name in hp.SETS:
                self.assertAlmostEqual(table["differences"][key]["sets"][name], value)
            for panel in hp.PANELS:
                self.assertAlmostEqual(table["differences"][key]["panels"][panel], value)

    def test_error_profile_rates(self) -> None:
        b0 = full()
        b0["stem-cs"] = cell(5.0, words=2000, s=60, d=20, i=20)
        table = hr.tabulate(self.cells(B0=b0))
        self.assertEqual(table["error_profile"]["B0"]["stem-cs"],
                         {"substitutions": 3.0, "deletions": 1.0, "insertions": 1.0, "wer": 5.0})
        self.assertEqual(set(table["error_profile"]), set(hr.ERROR_PROFILE_MODELS))
        self.assertIsNone(table["error_profile"]["tstar"]["stem-cs"])

    def test_sets_whose_records_disagree_on_size_are_flagged(self) -> None:
        other = full()
        other["tech-dev-2"] = cell(9.0, words=999)
        table = hr.tabulate(self.cells(B0=full(), tstar=other))
        flagged = [f for f in table["flags"] if f["kind"] == "size_disagreement"]
        self.assertEqual([f["set"] for f in flagged], ["tech-dev-2"])

    def test_term_recall_means_over_the_stem_parts(self) -> None:
        stem = {p: {**cell(4.0), "term_recall": {"occurrences": 10 * (k + 1), "reproduced": 9 * (k + 1),
                                                   "recall": 0.9}}
                for k, p in enumerate(("stem-chem", "stem-math", "stem-cs"))}
        stem["stem-cs"]["term_recall"] = {"occurrences": 30, "reproduced": 15, "recall": 0.5}
        table = hr.tabulate(self.cells(tstar=stem))
        tr = table["term_recall"]["tstar"]
        self.assertAlmostEqual(tr["mean"], (0.9 + 0.9 + 0.5) / 3)
        self.assertEqual(tr["pooled"], {"occurrences": 60, "reproduced": 42, "recall": 42 / 60})
        self.assertIsNone(table["term_recall"]["B0p"]["mean"])

    def test_markdown_shows_dash_and_signed_differences(self) -> None:
        b0p = full(-0.25)
        del b0p["lectures-a"]
        report = {"tables": hr.tabulate(self.cells(B0=full(), B0p=b0p)), "sizes": {}, "cells": {}, "flags": []}
        text = hr.markdown(report)
        b0p_row = next(line for line in text.splitlines() if line.startswith("| B0′ |"))
        self.assertIn("| - |", b0p_row)
        diff_row = next(line for line in text.splitlines() if line.startswith("| B0′ − B0 |"))
        self.assertIn("-0.25", diff_row)
        self.assertIn("| - |", diff_row)


class LocationTest(unittest.TestCase):
    def path(self, model, name):
        loc = hr.location(model, name)
        return None if loc is None else loc.path

    def test_domain_records_come_from_the_plans_locations(self) -> None:
        heldout = R / "heldout-panel"
        self.assertEqual(self.path("B0", "tech-dev-1"), R / "devsets" / "scores" / "B0" / "tech-dev-1.json")
        self.assertEqual(self.path("B0", "med-dev-3"), R / "devsets" / "scores" / "B0" / "med-dev-3.json")
        self.assertEqual(self.path("B0", "stem-chem"), R / "devsets" / "scores" / "eval-stem" / "B0" / "stem-chem.json")
        ext = R / "devsets" / "scores" / "eval-ext"
        self.assertEqual(self.path("B0", "lectures-a"), ext / "B0" / "lectures-a.json")
        self.assertEqual(self.path("MB1", "medical-b"), ext / "MB1" / "medical-b.json")
        for arm in ("b100-main", "b100-control"):
            self.assertEqual(self.path(arm, "med-dev-2"), R / "keab-prime" / "scores" / arm / "x1" / "med-dev-2.json")
            self.assertEqual(self.path(arm, "stem-cs"), R / "keab-prime" / "scores" / arm / "x1" / "stem-cs.json")
            for name in ("lectures-a", "medical-b"):
                self.assertEqual(self.path(arm, name), heldout / arm / f"{name}.json")
        b100 = R / "b0prime" / "runs" / "b100"
        self.assertEqual(self.path("B0p", "tech-dev-2"), b100 / "domain" / "tech-dev-2.json")
        self.assertEqual(self.path("B0p", "stem-math"), b100 / "eval-stem" / "stem-math.json")
        self.assertEqual(self.path("B0p", "medical-b"), heldout / "B0p" / "medical-b.json")
        for model in ("tstar", "tstar-encoder"):
            for name in hp.SETS:
                self.assertEqual(self.path(model, name), heldout / model / f"{name}.json")

    def test_every_gap_scored_is_read_from_the_panels_output(self) -> None:
        for model, names in hp.SCORED.items():
            for name in names:
                self.assertEqual(self.path(model, name), hp.output_dir(model, None) / f"{name}.json")

    def test_reference_rows_come_from_existing_records(self) -> None:
        self.assertEqual(self.path("B0", "ami_dev"), relabel_paths.STUDENT_RECORDS["B0"] / "ami_dev.json")
        self.assertEqual(self.path("MB1", "yodas_dev"), relabel_paths.STUDENT_RECORDS["MB1"] / "yodas_dev.json")
        keab, b100 = R / "keab-prime" / "scores", R / "b0prime" / "runs" / "b100"
        self.assertEqual(self.path("B0", "ami-sdm-dev"), keab / "B0" / "fp32" / "ami-sdm-dev.json")
        self.assertEqual(self.path("B0", "cv27-en-dev"), R / "b0prime" / "scores" / "B0" / "cv-dev" / "cv27-en-dev.json")
        self.assertEqual(self.path("B0p", "voxpopuli_dev"), b100 / "eval-dev" / "voxpopuli_dev.json")
        self.assertEqual(self.path("B0p", "cv27-en-dev"), b100 / "cv-dev" / "cv27-en-dev.json")
        self.assertEqual(self.path("B0p", "ami-sdm-dev"), keab / "B0p" / "fp32" / "ami-sdm-dev.json")
        self.assertEqual(self.path("b100-main", "yodas_dev"), keab / "b100-main" / "x1" / "yodas_dev.json")
        self.assertEqual(self.path("tstar", "ami_dev"), relabel_paths.DIAGNOSTIC / hp.TEACHER / "ami_dev.json")
        self.assertEqual(self.path("tstar-encoder", "yodas_dev"), R / "encoder-probe" / "ctc" / "bpe" / "yodas_dev.json")
        for model in ("tstar", "tstar-encoder"):
            for name in ("ami-sdm-dev", "cv27-en-dev"):
                self.assertIsNone(hr.location(model, name))

    def test_a_record_scored_another_way_than_the_rows_path_is_marked(self) -> None:
        loc = hr.location("MB1", "ami-sdm-dev")
        self.assertEqual(loc.path, R / "keab-prime" / "scores" / "MB1" / "x1" / "ami-sdm-dev.json")
        self.assertNotEqual(loc.kind, hr.MAIN_KIND["MB1"])
        for model in hr.MODELS:
            for name in hp.SETS:
                self.assertEqual(hr.location(model, name).kind, hr.MAIN_KIND[model])


class ReadCellTest(unittest.TestCase):
    def setUp(self) -> None:
        relabel_paths.require_mount()
        self.root = TMP / "cells"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        rows = [{"id": f"u{k}", "ref": t, "hyp": h, **evaluate.score(t, h), "duration": 1.0}
                for k, (t, h) in enumerate([("the gradient of a matrix", "the gradient of a matrix"),
                                            ("an eigenvalue problem", "an eigen value problem"),
                                            ("use a hash table here", "use a hash cable here")])]
        self.rows = rows
        self.record = {"source": {"sha256": "a" * 64}, "manifest_sha256": "m" * 64, "limit": None,
                       "utterances": 3, "wer": evaluate.wer_summary(rows),
                       "term_recall": {"occurrences": 0, "reproduced": 0, "recall": None}, "records": rows,
                       "panel": {"code": {"commit": "c" * 40, "dirty": False}}}
        self.expect = {"identity": "a" * 64, "manifest_sha256": "m" * 64, "utterances": 3}

    def read(self, record=None, name="stem-cs", expect=None):
        path = self.root / f"{name}.json"
        if record is not False:
            path.write_text(json.dumps(record or self.record))
        loc = hr.Location(path, hr.MAIN_KIND["tstar"])
        return hr.read_cell("tstar", name, loc, expect or self.expect), path

    def test_a_record_of_the_expected_model_and_manifest_counts_with_its_identity(self) -> None:
        got, path = self.read()
        self.assertEqual(got["status"], "ok")
        self.assertAlmostEqual(got["wer"], 100 * self.record["wer"]["wer"])
        self.assertEqual(got["record_sha256"], evaluate.sha256_file(path))
        self.assertEqual(got["identity"], "a" * 64)
        self.assertEqual(got["manifest_sha256"], "m" * 64)
        self.assertEqual(got["code"], {"commit": "c" * 40, "dirty": False})
        self.assertEqual((got["substitutions"], got["deletions"], got["insertions"], got["ref_words"]),
                         tuple(self.record["wer"][k] for k in ("substitutions", "deletions", "insertions",
                                                                "ref_words")))

    def test_wrong_model_manifest_limit_or_size_is_not_counted(self) -> None:
        for change in ({"source": {"sha256": "b" * 64}}, {"manifest_sha256": "n" * 64}, {"limit": 2},
                       {"utterances": 2}):
            got, _ = self.read({**self.record, **change})
            self.assertEqual(got["status"], "not_comparable", change)
            self.assertTrue(got["problems"])

    def test_identity_from_each_scorers_layout(self) -> None:
        layouts = [({"config": {"model": {"name": "x", "view_sha256": "a" * 64}}}, "x1"),
                   ({"config": {"model": {"name": "B0p", "sha256": "a" * 64}}}, "stage 4 fp32"),
                   ({"config": {"weights_sha256": "a" * 64}}, "stage 3 and score_domain"),
                   ({"source": {"sha256": "a" * 64}}, "evaluate.py"),
                   ({"config": {"revision": "a" * 64}}, "score_teacher and encoder_probe_ctc")]
        for extra, name in layouts:
            record = {k: v for k, v in self.record.items() if k != "source"} | extra
            got, _ = self.read(record)
            self.assertEqual(got["status"], "ok", name)

    def test_absent_record_is_missing(self) -> None:
        got, _ = self.read(record=False, name="stem-chem")
        self.assertEqual(got["status"], "missing")

    def test_term_recall_is_recomputed_on_eval_stem(self) -> None:
        got, _ = self.read()
        expected = score_domain.term_recall(json.loads(json.dumps(self.rows)), terms.matcher("technical"))
        self.assertEqual(got["term_recall"], expected)
        self.assertEqual(got["term_recall_stored"], self.record["term_recall"])
        other, _ = self.read(name="tech-dev-1")
        self.assertNotIn("term_recall", other)


class BuildTest(unittest.TestCase):
    """build() end to end on a tiny stand-in tree: every cell's record identity and the "-" cells are reported."""

    def test_unscored_gaps_are_dash_and_reported_missing(self) -> None:
        relabel_paths.require_mount()
        root = TMP / "build"
        if root.exists():
            shutil.rmtree(root)
        with mock.patch.object(hr, "location", lambda model, name: hr.Location(root / model / f"{name}.json",
                                                                              hr.MAIN_KIND[model])), \
             mock.patch.object(hr, "expectations", lambda: ({m: "a" * 64 for m in hr.MODELS},
                                                            {n: {"manifest_sha256": "m" * 64, "utterances": 1,
                                                                 "hours": 0.1} for n in hr.ALL_SETS})):
            report = hr.build()
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["tables"]["main"]["B0"]["sets"]["stem-chem"], None)
        self.assertTrue(all(c["status"] == "missing" for m in report["cells"].values() for c in m.values()))


if __name__ == "__main__":
    unittest.main()
