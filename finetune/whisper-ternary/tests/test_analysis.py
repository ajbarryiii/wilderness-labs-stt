"""Post-hoc secondary analyses: runaway criteria, secondary WERs, duration cap, trajectory, CLI.

Synthetic records and manifests only; no model, no GPU. The primary WER must be copied untouched.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analysis
import paths
import wer


def rec(uid: str, ref: str, hyp: str) -> dict:
    """An evaluation record whose S/D/I come from wer.edit_counts, as decoding.decode stores them."""
    s, d, i = wer.edit_counts(ref.split(), hyp.split())
    return {"id": uid, "ref": ref.upper(), "hyp": hyp, "ref_norm": ref, "hyp_norm": hyp,
            "S": s, "D": d, "I": i, "truncated": False}


def doc_of(records: list[dict], split: str = "dev-clean") -> dict:
    return {"split": split, "utterances": len(records), "wer": wer.corpus_wer(records),
            "records": records}


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}{k}" for k in range(n))


def write_manifest(directory: Path, split: str, durations: dict[str, float]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{split}.json").write_text(json.dumps(
        [{"id": uid, "path": f"/nowhere/{uid}.flac", "text": "X", "duration_s": s}
         for uid, s in durations.items()]))


RUNAWAY_REF = "he hoped there would be stew for dinner"
RUNAWAY_HYP = "he hoped there would be still for dinner " + words("continuation", 30)
# Duration cap ceil(4.5 * s) + 5: 2.0 s -> 14 words, 3.0 s -> 19 words.
DURATIONS = {"100-1-0000": 2.0, "100-1-0001": 2.0, "100-1-0002": 2.0, "100-1-0003": 3.0}


def synthetic_doc() -> dict:
    """Three normal utterances (one S, one I, one empty hypothesis) and one runaway (S 1, I 30)."""
    return doc_of([rec("100-1-0000", "the cat sat on the mat", "the cat sat on a mat"),
                   rec("100-1-0001", "a quick brown fox jumps", "a quick brown fox jumps high"),
                   rec("100-1-0002", "we went home early today", ""),
                   rec("100-1-0003", RUNAWAY_REF, RUNAWAY_HYP)])


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifests = self.root / "manifests"
        write_manifest(self.manifests, "dev-clean", DURATIONS)

    def tearDown(self) -> None:
        self.tmp.cleanup()


class RunawayCriteriaTest(unittest.TestCase):
    def check(self, record: dict, by_ins: bool, by_ratio: bool) -> None:
        self.assertIs(analysis.runaway_by_insertions(record), by_ins)
        self.assertIs(analysis.runaway_by_ratio(record), by_ratio)
        self.assertIs(analysis.runaway(record), by_ins or by_ratio)

    def test_neither(self) -> None:
        self.check(rec("n", words("w", 10), words("w", 10)), False, False)
        self.check(rec("n", words("w", 10), words("x", 10)), False, False)  # 10 S, no runaway

    def test_insertions_alone_with_boundary(self) -> None:
        ref = words("w", 40)  # length limit 1.5 * 40 + 10 = 70 words
        at = rec("a", ref, ref + " " + words("x", 20))
        below = rec("b", ref, ref + " " + words("x", 19))
        self.assertEqual((at["I"], len(at["hyp_norm"].split())), (20, 60))
        self.assertEqual(below["I"], 19)
        self.check(at, True, False)  # exactly 20 insertions counts
        self.check(below, False, False)

    def test_ratio_alone_with_boundary(self) -> None:
        ref = words("w", 4)  # length limit 1.5 * 4 + 10 = 16 words
        over = rec("o", ref, ref + " " + words("x", 13))
        exact = rec("e", ref, ref + " " + words("x", 12))
        self.assertEqual((len(over["hyp_norm"].split()), over["I"]), (17, 13))
        self.assertEqual(len(exact["hyp_norm"].split()), 16)
        self.check(over, False, True)
        self.check(exact, False, False)  # exactly ratio * len + 10 is not runaway

    def test_both(self) -> None:
        self.check(rec("b", words("w", 4), words("w", 4) + " " + words("x", 30)), True, True)

    def test_parameters(self) -> None:
        record = rec("p", words("w", 4), words("w", 4) + " " + words("x", 8))  # I 8, 12 words
        self.assertFalse(analysis.runaway(record))
        self.assertTrue(analysis.runaway(record, min_insertions=8))
        self.assertFalse(analysis.runaway(record, ratio=0.5))  # 12 > 0.5 * 4 + 10 is false
        self.assertTrue(analysis.runaway(record, ratio=0.25))  # 12 > 0.25 * 4 + 10

    def test_empty_reference(self) -> None:
        self.check(rec("e", "", words("x", 11)), False, True)  # 11 > 1.5 * 0 + 10
        self.check(rec("e", "", words("x", 10)), False, False)


class DiagnoseTest(TempDirTest):
    def setUp(self) -> None:
        super().setUp()
        self.doc = synthetic_doc()
        self.normal, self.run = self.doc["records"][:3], self.doc["records"][3]

    def diagnose(self, doc: dict, **options) -> dict:
        return analysis.diagnose(doc, manifests=self.manifests, **options)

    def test_every_field(self) -> None:
        before = copy.deepcopy(self.doc)
        d = self.diagnose(self.doc)
        self.assertEqual(self.doc, before)  # the evaluation document is not modified
        self.assertEqual(set(d), {
            "split", "utterances", "wer", "substitutions", "deletions", "insertions",
            "runaway_count", "runaway_by_insertions", "runaway_by_ratio", "runaway_insertions",
            "wer_excluding_runaway", "wer_runaway_capped", "wer_duration_capped",
            "duration_capped_truncated", "duration_capped_truncated_words", "reference_truncated",
            "duration_cap_status", "insertion_share_of_errors", "empty_hypotheses",
            "median_hyp_ref_ratio", "worst", "worst_runaway", "criteria"})
        self.assertEqual((self.run["S"], self.run["D"], self.run["I"]), (1, 0, 30))
        self.assertEqual((d["split"], d["utterances"]), ("dev-clean", 4))
        self.assertEqual(d["wer"], self.doc["wer"]["wer"])
        self.assertAlmostEqual(d["wer"], (2 + 5 + 31) / 24)
        self.assertEqual((d["substitutions"], d["deletions"], d["insertions"]), (2, 5, 31))
        self.assertEqual((d["runaway_count"], d["runaway_by_insertions"], d["runaway_by_ratio"]),
                         (1, 1, 1))
        self.assertEqual(d["runaway_insertions"], 30)
        self.assertAlmostEqual(d["wer_excluding_runaway"], (1 + 5 + 1) / 16)
        self.assertAlmostEqual(d["wer_runaway_capped"], (2 + 5 + 6) / 24)
        # Duration cap: only the runaway (3.0 s -> 19 words of 38) is cut: S 1, I 11 remain.
        self.assertAlmostEqual(d["wer_duration_capped"], (2 + 5 + 12) / 24)
        self.assertEqual((d["duration_capped_truncated"], d["duration_capped_truncated_words"],
                          d["reference_truncated"]), (1, 19, 0))
        self.assertTrue(d["duration_cap_status"].startswith("ok"))
        for secondary in ("wer_excluding_runaway", "wer_runaway_capped", "wer_duration_capped"):
            self.assertLess(d[secondary], d["wer"])
        self.assertAlmostEqual(d["insertion_share_of_errors"], 31 / 38)
        self.assertEqual(d["empty_hypotheses"], 1)
        self.assertAlmostEqual(d["median_hyp_ref_ratio"], (1.0 + 1.2) / 2)  # 0, 1.0, 1.2, 4.75
        self.assertEqual([w["id"] for w in d["worst_runaway"]], ["100-1-0003"])
        self.assertEqual(d["criteria"], {"min_insertions": 20, "ratio": 1.5,
                                         "length_slack_words": 10, "runaway_cap_extra_words": 5,
                                         "duration_cap_words_per_s": 4.5,
                                         "duration_cap_extra_words": 5})

    def test_worst(self) -> None:
        worst = self.diagnose(self.doc)["worst"]
        self.assertEqual([w["id"] for w in worst],  # by I, ties by id; fewer than 5 records
                         ["100-1-0003", "100-1-0001", "100-1-0000", "100-1-0002"])
        top = worst[0]
        self.assertGreater(len(self.run["hyp_norm"]), 240)
        self.assertEqual(top["hyp_norm"], self.run["hyp_norm"][:240])
        self.assertEqual(top["ref_norm"], self.run["ref_norm"][:120])
        self.assertEqual((top["S"], top["D"], top["I"], top["runaway"]), (1, 0, 30, True))
        self.assertEqual((top["ref_words"], top["hyp_words"]), (8, 38))
        self.assertFalse(worst[1]["runaway"])
        many = {**self.doc, "records": self.doc["records"] * 2}
        self.assertEqual(len(self.diagnose(many)["worst"]), 5)

    def test_primary_is_copied_not_recomputed(self) -> None:
        doc = copy.deepcopy(self.doc)
        doc["wer"] = {**doc["wer"], "wer": 0.4242}
        d = self.diagnose(doc)
        self.assertEqual(d["wer"], 0.4242)
        self.assertAlmostEqual(d["wer_runaway_capped"], 13 / 24)

    def test_secondary_wers_use_edit_counts(self) -> None:
        with patch.object(wer, "edit_counts", wraps=wer.edit_counts) as counted:
            d = self.diagnose(self.doc)
        calls = [tuple(c.args) for c in counted.call_args_list]
        ref, hyp = RUNAWAY_REF.split(), RUNAWAY_HYP.split()
        capped, duration_capped = hyp[:len(ref) + 5], hyp[:19]
        self.assertIn((ref, capped), calls)
        self.assertIn((ref, duration_capped), calls)
        # Independent recomputation with wer.edit_counts gives the same numbers.
        excl = [wer.edit_counts(r["ref_norm"].split(), r["hyp_norm"].split()) for r in self.normal]
        self.assertAlmostEqual(d["wer_excluding_runaway"], sum(map(sum, excl)) / 16)
        self.assertAlmostEqual(d["wer_runaway_capped"],
                               sum(map(sum, excl + [wer.edit_counts(ref, capped)])) / 24)
        self.assertAlmostEqual(d["wer_duration_capped"],
                               sum(map(sum, excl + [wer.edit_counts(ref, duration_capped)])) / 24)

    def test_capped_truncation_rule(self) -> None:
        capped = analysis.capped_hypothesis(self.run).split()
        self.assertEqual(len(capped), len(RUNAWAY_REF.split()) + 5)
        self.assertEqual(capped, RUNAWAY_HYP.split()[:13])
        short = rec("s", "a b c", "a b c d")
        self.assertEqual(analysis.capped_hypothesis(short), "a b c d")  # shorter than the cap

    def test_no_runaway_and_all_runaway(self) -> None:
        d = self.diagnose(doc_of(self.normal))
        self.assertEqual(d["runaway_count"], 0)
        self.assertEqual(d["worst_runaway"], [])
        self.assertAlmostEqual(d["wer_excluding_runaway"], d["wer"])
        self.assertAlmostEqual(d["wer_runaway_capped"], d["wer"])
        self.assertAlmostEqual(d["wer_duration_capped"], d["wer"])
        self.assertIsNone(self.diagnose(doc_of([self.run]))["wer_excluding_runaway"])


class DurationCapTest(TempDirTest):
    def test_cap_words(self) -> None:
        self.assertEqual(analysis.DEFAULT_CAP_WORDS_PER_S, 4.5)
        self.assertEqual(analysis.DEFAULT_CAP_EXTRA_WORDS, 5)
        self.assertEqual(analysis.duration_cap_words(2.0), 14)  # 9 exactly, + 5
        self.assertEqual(analysis.duration_cap_words(1.0), 10)  # ceil(4.5) = 5, + 5
        self.assertEqual(analysis.duration_cap_words(2.01), 15)  # ceil(9.045) = 10, + 5
        self.assertEqual(analysis.duration_cap_words(2.0, words_per_s=3.0, extra=0), 6)

    def test_boundary(self) -> None:
        ref = words("w", 10)
        at = rec("u-at", ref, words("w", 14))  # exactly the 2.0 s cap
        over = rec("u-over", ref, words("w", 15))  # one word more
        write_manifest(self.manifests, "dev-clean", {"u-at": 2.0, "u-over": 2.0})
        d = analysis.diagnose(doc_of([at]), manifests=self.manifests)
        self.assertEqual((d["duration_capped_truncated"], d["duration_capped_truncated_words"]),
                         (0, 0))
        self.assertAlmostEqual(d["wer_duration_capped"], d["wer"])
        d = analysis.diagnose(doc_of([over]), manifests=self.manifests)
        self.assertEqual((d["duration_capped_truncated"], d["duration_capped_truncated_words"]),
                         (1, 1))
        self.assertAlmostEqual(d["wer"], 5 / 10)
        self.assertAlmostEqual(d["wer_duration_capped"], 4 / 10)  # w10..w14 -> w10..w13
        both = analysis.duration_capped([at, over], {"u-at": 2.0, "u-over": 2.0})
        self.assertEqual(both["duration_capped_truncated"], 1)

    def test_overrides(self) -> None:
        record = rec("u", words("w", 10), words("w", 14))
        write_manifest(self.manifests, "dev-clean", {"u": 2.0})
        d = analysis.diagnose(doc_of([record]), manifests=self.manifests, cap_words_per_s=3.0,
                              cap_extra_words=2)  # cap 8 words
        self.assertEqual((d["duration_capped_truncated"], d["duration_capped_truncated_words"],
                          d["reference_truncated"]), (1, 6, 1))
        self.assertEqual((d["criteria"]["duration_cap_words_per_s"],
                          d["criteria"]["duration_cap_extra_words"]), (3.0, 2))

    def test_reference_truncated(self) -> None:
        # 0.5 s -> ceil(2.25) + 5 = 8 words; a 9-word reference would be cut, an 8-word one not.
        records = [rec("long", words("w", 9), words("w", 9)), rec("fits", words("w", 8), words("w", 8)),
                   rec("long2", words("w", 12), words("w", 3))]
        write_manifest(self.manifests, "test-other", {"long": 0.5, "fits": 0.5, "long2": 0.5})
        d = analysis.diagnose(doc_of(records, "test-other"), manifests=self.manifests)
        self.assertEqual(d["reference_truncated"], 2)
        self.assertEqual((d["duration_capped_truncated"], d["duration_capped_truncated_words"]),
                         (1, 1))

    def test_missing_manifest(self) -> None:
        d = analysis.diagnose(doc_of(synthetic_doc()["records"], "test-clean"),
                              manifests=self.manifests)  # no test-clean.json there
        for field in analysis.DURATION_FIELDS:
            self.assertIsNone(d[field])
        self.assertIn("manifest not found", d["duration_cap_status"])
        self.assertIsNotNone(d["wer_excluding_runaway"])  # the other diagnostics still run
        (self.manifests / "test-clean.json").write_text("not json")
        d = analysis.diagnose(doc_of(synthetic_doc()["records"], "test-clean"),
                              manifests=self.manifests)
        self.assertIsNone(d["wer_duration_capped"])
        self.assertIn("manifest unreadable", d["duration_cap_status"])

    def test_missing_id(self) -> None:
        write_manifest(self.manifests, "dev-clean", {"100-1-0000": 2.0})
        d = analysis.diagnose(synthetic_doc(), manifests=self.manifests)
        for field in analysis.DURATION_FIELDS:
            self.assertIsNone(d[field])
        self.assertIn("3 of 4 ids missing", d["duration_cap_status"])

    def test_default_manifest_location(self) -> None:
        with patch.object(paths, "MANIFESTS", self.manifests):
            d = analysis.diagnose(synthetic_doc())
        self.assertEqual(d["duration_capped_truncated"], 1)
        self.assertIn(str(self.manifests / "dev-clean.json"), d["duration_cap_status"])

    def test_split_of(self) -> None:
        self.assertEqual(analysis.split_of({"split": "test-other"}), "test-other")
        self.assertEqual(analysis.split_of({"subset": "data.dev_subset(dev-clean, 400)"}),
                         "dev-clean")
        self.assertEqual(analysis.split_of({"subset": "data.dev_subset(test-other, 16)"}),
                         "test-other")
        self.assertEqual(analysis.split_of({}), "dev-clean")


class FileTest(TempDirTest):
    def setUp(self) -> None:
        super().setUp()
        manifests = patch.object(paths, "MANIFESTS", self.manifests)
        manifests.start()
        self.addCleanup(manifests.stop)
        self.run = self.root / "v2-test-run"
        subset = self.run / "dev-subset"
        subset.mkdir(parents=True)
        base = synthetic_doc()
        clean_records = base["records"][:3]
        # Written out of order; the non-step file must be ignored.
        self.late = {**base, "step": 1000}
        self.early = {**doc_of(clean_records), "step": 500, "weight_fraction": 0.5}
        del self.early["split"]
        self.early["subset"] = "data.dev_subset(dev-clean, 3)"
        (subset / "step-01000.json").write_text(json.dumps(self.late))
        (subset / "step-00500.json").write_text(json.dumps(self.early))
        (subset / "notes.json").write_text("{}")
        self.files = [subset / "step-00500.json", subset / "step-01000.json"]

    def run_main(self, argv: list[str]) -> str:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(analysis.main(argv), 0)
        return stdout.getvalue()

    def test_trajectory_order(self) -> None:
        rows = analysis.trajectory(self.run)
        self.assertEqual([r["step"] for r in rows], [500, 1000])
        self.assertEqual([r["weight_fraction"] for r in rows], [0.5, None])
        self.assertEqual([r["runaway_count"] for r in rows], [0, 1])
        self.assertEqual([r["duration_capped_truncated"] for r in rows], [0, 1])
        self.assertEqual(rows[1]["wer"], self.late["wer"]["wer"])
        lines = analysis.trajectory_text(self.run, rows).splitlines()
        self.assertIn("SECONDARY", lines[0])
        self.assertIn("primary metric is the uncapped corpus WER", lines[0])
        header = lines[2]
        for column in ("WER dur-capped", "truncated (n)", "refs truncated"):
            self.assertIn(column, header)
        body = [line.split() for line in lines[4:]]
        self.assertEqual([b[0] for b in body], ["500", "1000"])
        self.assertEqual([b[1] for b in body], ["0.500", "-"])
        self.assertEqual([b[-3:] for b in body],
                         [[analysis._pct(rows[0]["wer"]), "0", "0"],
                          [f"{100 * 19 / 24:.2f}%", "1", "0"]])

    def test_cli_files(self) -> None:
        before = [f.read_bytes() for f in self.files]
        out_json = self.root / "out" / "diag.json"
        text = self.run_main([str(f) for f in self.files] + ["--json", str(out_json)])
        self.assertEqual([f.read_bytes() for f in self.files], before)  # inputs untouched
        headings = [line for line in text.splitlines()
                    if line.startswith(("Error analysis", "Worst runaway"))]
        self.assertEqual(len(headings), 2)
        for heading in headings:
            self.assertIn("SECONDARY", heading)
            self.assertIn("primary metric is the uncapped corpus WER", heading)
        for column in ("WER dur-capped", "truncated (n)", "refs truncated"):
            self.assertIn(column, text)
        self.assertIn("dev-subset/step-00500.json", text)
        self.assertIn("dev-subset/step-01000.json", text)
        self.assertIn("100-1-0003", text)  # the runaway utterance is listed
        saved = json.loads(out_json.read_text())
        self.assertIn("primary metric", saved["note"])
        self.assertEqual([r["label"] for r in saved["results"]],
                         ["dev-subset/step-00500.json", "dev-subset/step-01000.json"])
        self.assertEqual(saved["results"][1]["diagnostics"],
                         json.loads(json.dumps(analysis.diagnose(self.late))))

    def test_cli_missing_manifest_reports_null(self) -> None:
        (self.manifests / "dev-clean.json").unlink()
        text = self.run_main([str(f) for f in self.files])
        self.assertIn("manifest not found", text)
        self.assertIn("dur-capped fields are null", text)
        row = next(line for line in text.splitlines() if line.startswith("dev-subset/step-01000"))
        self.assertEqual(row.split().count("null"), 3)  # WER dur-capped, truncated, refs truncated
        text = self.run_main(["trajectory", str(self.run)])
        self.assertIn("step 500: duration cap unavailable", text)

    def test_cli_markdown(self) -> None:
        out_md = self.root / "out" / "SECONDARY.md"
        self.run_main([str(f) for f in self.files] + ["--markdown", str(out_md)])
        markdown = out_md.read_text()
        self.assertEqual(markdown.splitlines()[0],
                         "# Secondary analyses (post hoc). Primary metric is the uncapped corpus "
                         "WER; see DESIGN.md 'Secondary analyses'.")
        table = [line for line in markdown.splitlines() if line.startswith("| ")]
        self.assertEqual(len(table), 2 + len(self.files))  # header, alignment, one row per file
        self.assertTrue(table[0].startswith("| file | utts | WER |"))
        self.assertIn("WER dur-capped", table[0])
        self.assertEqual([row.split(" | ")[0] for row in table[2:]],
                         ["| dev-subset/step-00500.json", "| dev-subset/step-01000.json"])
        self.assertIn("## Worst runaway utterances", markdown)
        self.assertIn("`100-1-0003  S 1 D 0 I 30  ref 8 w, hyp 38 w`", markdown)

    def test_cli_trajectory(self) -> None:
        text = self.run_main(["trajectory", str(self.run), "--cap-words-per-second", "3",
                              "--cap-extra-words", "0"])
        self.assertIn("SECONDARY", text.splitlines()[0])
        self.assertIn("ceil(3.0 * audio s) + 0 words", text)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            analysis.main(["trajectory", str(self.root)])  # no dev-subset files


if __name__ == "__main__":
    unittest.main()
