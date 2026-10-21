"""Contract tests for kb1lib, the harness's pure logic (statistics, pairing, scoring, bundle checks).

Invented data only. Run with Python 3.9 or later from the bench directory:
    python3 -m unittest discover -s tests
"""
from __future__ import annotations

import array
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kb1lib as lib  # noqa: E402


class Quantiles(unittest.TestCase):
    def test_median_and_type7_quantile(self):
        self.assertEqual(lib.median([3, 1, 2]), 2)
        self.assertEqual(lib.median([4, 1, 3, 2]), 2.5)
        self.assertAlmostEqual(lib.quantile(list(range(1, 21)), 0.95), 19.05)
        self.assertEqual(lib.quantile([7], 0.95), 7)
        self.assertEqual(lib.quantile([1, 2, 3], 0.0), 1)
        self.assertEqual(lib.quantile([1, 2, 3], 1.0), 3)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            lib.median([])
        with self.assertRaises(ValueError):
            lib.quantile([], 0.5)

    def test_latency_summary(self):
        s = lib.latency_summary([10.0, 12.0, 11.0, 50.0])
        self.assertEqual(s["n"], 4)
        self.assertEqual(s["median_ms"], 11.5)
        self.assertEqual(s["min_ms"], 10.0)
        self.assertEqual(s["max_ms"], 50.0)
        self.assertAlmostEqual(s["p95_ms"], lib.quantile([10.0, 12.0, 11.0, 50.0], 0.95))


class PairedRatios(unittest.TestCase):
    base = {"a": [10.0, 11.0, 9.0], "b": [20.0, 21.0, 19.0], "c": [40.0, 40.0, 41.0]}

    def test_identical_arms_give_ratio_one_with_degenerate_interval(self):
        r = lib.paired_ratio(self.base, self.base, n_boot=200, seed=0)
        self.assertEqual(r["clips"], 3)
        self.assertEqual(r["typical_ratio"], 1.0)
        self.assertEqual(r["typical_ratio_ci95"], [1.0, 1.0])
        self.assertEqual(r["p95_ratio"], 1.0)

    def test_uniform_slowdown_is_recovered_exactly(self):
        arm = {k: [2 * v for v in vs] for k, vs in self.base.items()}
        r = lib.paired_ratio(arm, self.base, n_boot=200, seed=0)
        self.assertAlmostEqual(r["typical_ratio"], 2.0)
        self.assertAlmostEqual(r["typical_ratio_ci95"][0], 2.0)
        self.assertAlmostEqual(r["typical_ratio_ci95"][1], 2.0)
        self.assertAlmostEqual(r["p95_ratio"], 2.0)
        self.assertEqual(r["arm_typical_ms"], 2 * r["baseline_typical_ms"])

    def test_ratio_uses_per_clip_medians_so_one_outlier_call_does_not_move_it(self):
        base = {"a": [10.0, 10.0, 1000.0]}
        arm = {"a": [20.0, 20.0, 20.0]}
        self.assertAlmostEqual(lib.paired_ratio(arm, base, n_boot=50, seed=0)["typical_ratio"], 2.0)

    def test_only_common_clips_are_paired(self):
        arm = {"a": [5.0], "z": [99.0]}
        r = lib.paired_ratio(arm, self.base, n_boot=50, seed=0)
        self.assertEqual(r["clips"], 1)
        self.assertAlmostEqual(r["typical_ratio"], 5.0 / 10.0)
        with self.assertRaises(ValueError):
            lib.paired_ratio({"z": [1.0]}, self.base, n_boot=50, seed=0)

    def test_bootstrap_is_deterministic_and_brackets_the_estimate(self):
        noisy = {f"c{i}": [10.0 + i, 11.0 + i, 12.0 + 2 * i] for i in range(12)}
        arm = {k: [v * (1.1 + 0.02 * (i % 5)) for v in vs] for i, (k, vs) in enumerate(sorted(noisy.items()))}
        first = lib.paired_ratio(arm, noisy, n_boot=300, seed=7)
        second = lib.paired_ratio(arm, noisy, n_boot=300, seed=7)
        self.assertEqual(first, second)
        lo, hi = first["typical_ratio_ci95"]
        self.assertLessEqual(lo, first["typical_ratio"])
        self.assertGreaterEqual(hi, first["typical_ratio"])
        self.assertGreater(lo, 1.0)


class Counterbalancing(unittest.TestCase):
    def test_each_round_is_a_permutation_and_positions_balance_over_round_pairs(self):
        variants = ["A", "B", "C", "D", "E"]
        for rounds in (2, 4, 6):
            orders = lib.counterbalanced_orders(variants, rounds)
            self.assertEqual(len(orders), rounds)
            for order in orders:
                self.assertEqual(sorted(order), sorted(variants))
            totals = {v: sum(order.index(v) for order in orders) for v in variants}
            self.assertEqual(len(set(totals.values())), 1, totals)

    def test_single_round_keeps_the_listed_order(self):
        self.assertEqual(lib.counterbalanced_orders(["A", "B", "C"], 1), [["A", "B", "C"]])

    def test_later_round_pairs_change_neighbours(self):
        orders = lib.counterbalanced_orders(["A", "B", "C", "D"], 4)
        self.assertNotEqual(orders[0], orders[2])


class Grouping(unittest.TestCase):
    def test_records_group_by_key_then_clip(self):
        records = [
            {"variant": "x@ane", "bucket": 4, "clip": "k1", "ms": 1.0},
            {"variant": "x@ane", "bucket": 4, "clip": "k1", "ms": 2.0},
            {"variant": "x@ane", "bucket": 8, "clip": "k2", "ms": 3.0},
            {"variant": "y@ane", "bucket": 4, "clip": "k1", "ms": 4.0},
        ]
        g = lib.group_samples(records, ("variant", "bucket"))
        self.assertEqual(g[("x@ane", 4)], {"k1": [1.0, 2.0]})
        self.assertEqual(g[("x@ane", 8)], {"k2": [3.0]})
        self.assertEqual(g[("y@ane", 4)], {"k1": [4.0]})


class Normalization(unittest.TestCase):
    def test_spoken_numbers(self):
        self.assertEqual(lib.spoken_number("1850"), "eighteen fifty")
        self.assertEqual(lib.spoken_number("1905"), "nineteen o five")
        self.assertEqual(lib.spoken_number("$5"), "five dollars")
        self.assertEqual(lib.spoken_number("10%"), "ten percent")
        self.assertEqual(lib.spoken_number("21st"), "twenty first")
        self.assertEqual(lib.spoken_number("2,000"), "two thousand")
        self.assertEqual(lib.spoken_number("007"), "007")

    def test_normalize_applies_titles_numbers_punctuation_and_spelling_map(self):
        words = lib.normalize("Mr. Brown's 3rd colour, he said!", {"colour": "color"})
        self.assertEqual(words, ["mister", "brown's", "third", "color", "he", "said"])
        self.assertEqual(lib.plain_normalize("Hello, World -- it's 'fine'."), ["hello", "world", "it's", "fine"])

    def test_alignment_counts_substitutions_deletions_insertions(self):
        self.assertEqual(lib.align("a b c".split(), "a x c d".split()), (1, 0, 1))
        self.assertEqual(lib.align("a b c".split(), "a c".split()), (0, 1, 0))
        self.assertEqual(lib.align([], "a".split()), (0, 0, 1))
        self.assertEqual(lib.align("a b".split(), []), (0, 2, 0))


class WordErrorRate(unittest.TestCase):
    items = [
        {"id": "u1", "reference": "THE CAT SAT", "hypothesis": "The cat sat."},
        {"id": "u2", "reference": "ON THE MAT", "hypothesis": "on a mat"},
        {"id": "u3", "reference": "A RED COLOUR", "hypothesis": "A red color"},
    ]

    def test_wer_counts_edits_over_reference_words(self):
        s = lib.wer_summary(self.items, spelling={"colour": "color"}, seed=1, n_boot=200)
        self.assertEqual(s["utterances"], 3)
        self.assertEqual(s["reference_words"], 9)
        self.assertEqual((s["substitutions"], s["deletions"], s["insertions"]), (1, 0, 0))
        self.assertAlmostEqual(s["wer"], 1 / 9)
        self.assertAlmostEqual(s["plain_wer"], 2 / 9)
        lo, hi = s["wer_ci95"]
        self.assertLessEqual(lo, s["wer"])
        self.assertGreaterEqual(hi, s["wer"])

    def test_paired_delta_is_candidate_minus_baseline(self):
        base = {i["id"]: i for i in self.items}
        cand = {k: dict(v) for k, v in base.items()}
        cand["u1"]["hypothesis"] = "the dog sat"
        d = lib.paired_wer_delta(base, cand, spelling={"colour": "color"}, seed=1, n_boot=200)
        self.assertAlmostEqual(d["wer_delta"], 1 / 9)
        self.assertEqual(d["utterances_changed"], 1)
        with self.assertRaises(ValueError):
            lib.paired_wer_delta(base, {"u1": cand["u1"]}, spelling={}, seed=1, n_boot=10)


class Agreement(unittest.TestCase):
    def test_exact_and_normalized_agreement_with_the_reference_model(self):
        ref = {"u1": "Hello there.", "u2": "Good morning", "u3": "Fine, thanks"}
        cand = {"u1": "Hello there.", "u2": "good morning", "u3": "Fine thanks a lot"}
        a = lib.agreement(ref, cand)
        self.assertEqual(a["utterances"], 3)
        self.assertAlmostEqual(a["exact_match_rate"], 1 / 3)
        self.assertAlmostEqual(a["normalized_match_rate"], 2 / 3)
        self.assertEqual(a["differing_ids"], ["u2", "u3"])
        # Word error of the candidate against the reference model's words (u3: 2 insertions over 6 words).
        self.assertAlmostEqual(a["wer_vs_reference"], 2 / 6)

    def test_agreement_needs_the_same_utterances(self):
        with self.assertRaises(ValueError):
            lib.agreement({"u1": "a"}, {"u2": "a"})


class EncoderError(unittest.TestCase):
    def test_identical_outputs_have_zero_error(self):
        ref = array.array("f", [1.0, 0.0, -2.0, 0.5])
        e = lib.encoder_error(ref, array.array("f", ref))
        self.assertEqual(e["rel_l2"], 0.0)
        self.assertEqual(e["max_abs"], 0.0)
        self.assertAlmostEqual(e["cosine"], 1.0)

    def test_scaled_output_has_relative_error_and_unit_cosine(self):
        ref = array.array("f", [1.0, 2.0, 2.0])
        e = lib.encoder_error(ref, array.array("f", [2 * v for v in ref]))
        self.assertAlmostEqual(e["rel_l2"], 1.0)
        self.assertAlmostEqual(e["max_abs"], 2.0)
        self.assertAlmostEqual(e["cosine"], 1.0)
        self.assertAlmostEqual(e["err_sq"], 9.0)
        self.assertAlmostEqual(e["ref_sq"], 9.0)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            lib.encoder_error(array.array("f", [1.0]), array.array("f", [1.0, 2.0]))

    def test_aggregate_pools_squared_errors(self):
        a = {"rel_l2": 0.0, "max_abs": 0.0, "cosine": 1.0, "err_sq": 0.0, "ref_sq": 4.0}
        b = {"rel_l2": 1.0, "max_abs": 3.0, "cosine": 0.5, "err_sq": 12.0, "ref_sq": 12.0}
        agg = lib.aggregate_encoder_errors([a, b])
        self.assertEqual(agg["chunks"], 2)
        self.assertAlmostEqual(agg["pooled_rel_l2"], math.sqrt(12.0 / 16.0))
        self.assertEqual(agg["max_rel_l2"], 1.0)
        self.assertEqual(agg["median_rel_l2"], 0.5)
        self.assertEqual(agg["max_abs"], 3.0)
        self.assertEqual(agg["min_cosine"], 0.5)


def write_bundle(root: Path, *, model="localflow", buckets=(4, 8, 15, 30), extra_files=None) -> Path:
    files = {
        "frontend.json": b"{}", "frontend.f32bin": b"\x00" * 8, "decoder_joint.json": b"{}",
        "decoder_joint.f32bin": b"\x01" * 16, "vocabulary.json": b"{}",
        "Encoder.mlmodelc/model.mil": b"program", "Encoder.mlmodelc/weights/weight.bin": b"\x02" * 100,
        "Encoder.mlmodelc/coremldata.bin": b"cd",
    }
    files.update(extra_files or {})
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    manifest = {"model": model, "buckets_seconds": list(buckets), "encoder": "X", "compute_units": "cpuAndNeuralEngine",
                "files": {n: hashlib.sha256(d).hexdigest() for n, d in files.items()}}
    (root / "bundle.json").write_text(json.dumps(manifest))
    (root / "conversion").mkdir()
    (root / "conversion" / "log.txt").write_bytes(b"x" * 1000)  # not listed: not part of what the app loads
    return root


class BundleVerification(unittest.TestCase):
    def test_valid_bundle_reports_listed_and_encoder_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = write_bundle(Path(d))
            info = lib.verify_bundle(root)
            self.assertEqual(info["encoder_weight_bytes"], 100)
            self.assertEqual(info["encoder_bytes"], 100 + len(b"program") + 2)
            listed = 2 + 8 + 2 + 16 + 2 + 7 + 100 + 2
            self.assertEqual(info["listed_bytes"], listed)
            self.assertEqual(info["directory_bytes"], listed + 1000 + (root / "bundle.json").stat().st_size)
            self.assertEqual(info["manifest"]["model"], "localflow")
            self.assertEqual(len(info["bundle_json_sha256"]), 64)

    def test_tampered_file_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = write_bundle(Path(d))
            (root / "vocabulary.json").write_bytes(b"{ }")
            with self.assertRaises(ValueError):
                lib.verify_bundle(root)

    def test_wrong_model_or_buckets_fail(self):
        for kwargs in ({"model": "other"}, {"buckets": (2, 4, 8, 15)}):
            with tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):
                    lib.verify_bundle(write_bundle(Path(d), **kwargs))

    def test_unsafe_or_missing_paths_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = write_bundle(Path(d))
            manifest = json.loads((root / "bundle.json").read_text())
            manifest["files"]["../outside.bin"] = "0" * 64
            (root / "bundle.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                lib.verify_bundle(root)
        with tempfile.TemporaryDirectory() as d:
            root = write_bundle(Path(d))
            manifest = json.loads((root / "bundle.json").read_text())
            del manifest["files"]["vocabulary.json"]
            (root / "bundle.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                lib.verify_bundle(root)


class Variants(unittest.TestCase):
    def test_compute_unit_codes(self):
        self.assertEqual(lib.parse_variant("kea-b@ane"), ("kea-b", "ane", "cpuAndNeuralEngine"))
        self.assertEqual(lib.parse_variant("B1L@gpu"), ("B1L", "gpu", "cpuAndGPU"))
        self.assertEqual(lib.parse_variant("B1L@cpu"), ("B1L", "cpu", "cpuOnly"))
        self.assertEqual(lib.parse_variant("B1L@all"), ("B1L", "all", "all"))
        self.assertEqual(lib.parse_variant("mlx1@ext"), ("mlx1", "ext", "external"))
        for bad in ("B1L", "B1L@npu", "@ane", "a@b@ane"):
            with self.assertRaises(ValueError):
                lib.parse_variant(bad)


class Selection(unittest.TestCase):
    def test_subset_is_deterministic_sorted_and_bounded(self):
        items = [{"id": f"u{i:03d}"} for i in range(50)]
        a = lib.select_subset(items, 10, seed=0)
        self.assertEqual(a, lib.select_subset(list(reversed(items)), 10, seed=0))
        self.assertEqual(len(a), 10)
        self.assertEqual([i["id"] for i in a], sorted(i["id"] for i in a))
        self.assertEqual(len(lib.select_subset(items, 500, seed=0)), 50)

    def test_chunks_per_bucket_skip_silence_and_are_stable(self):
        rows = [{"key": f"k{i}", "bucket": b, "silent": i == 3} for i, b in enumerate([4, 4, 8, 4, 30, 4, 8])]
        chosen = lib.select_chunks(rows, per_bucket=2)
        self.assertEqual(chosen, {4: ["k0", "k1"], 8: ["k2", "k6"], 30: ["k4"]})


class Memory(unittest.TestCase):
    def test_verdict_against_the_ternary_model(self):
        v = lib.memory_verdict(candidate=900, ternary=1000)
        self.assertEqual(v, {"candidate_bytes": 900, "ternary_bytes": 1000, "delta_bytes": -100, "ratio": 0.9,
                             "less_than_ternary": True})
        self.assertFalse(lib.memory_verdict(candidate=1000, ternary=1000)["less_than_ternary"])

    def test_cache_delta_counts_new_and_grown_files(self):
        before = {"a": 10, "b": 5}
        after = {"a": 10, "b": 8, "c/d": 100}
        self.assertEqual(lib.cache_delta(before, after), {"new_files": 1, "changed_files": 1, "added_bytes": 103,
                                                          "removed_files": 0})
        self.assertEqual(lib.cache_delta({"a": 1}, {})["removed_files"], 1)


class Thermal(unittest.TestCase):
    def test_counts_states(self):
        t = lib.thermal_summary(["nominal", "nominal", "fair"])
        self.assertEqual(t, {"samples": 3, "counts": {"fair": 1, "nominal": 2}, "all_nominal": False})


class Markdown(unittest.TestCase):
    def test_table(self):
        self.assertEqual(lib.markdown_table(["a", "b"], [[1, "x"], [2.5, None]]),
                         "| a | b |\n| --- | --- |\n| 1 | x |\n| 2.5 | - |\n")


if __name__ == "__main__":
    unittest.main()
