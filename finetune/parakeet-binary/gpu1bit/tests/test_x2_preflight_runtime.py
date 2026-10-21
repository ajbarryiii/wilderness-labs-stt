"""Contract tests of the x2 preflight (gpu1bit/x2/preflight.py; plans/x2-default.md Y6, Z3) and of the x2
runtime-contract logic (gpu1bit/x2/runtime_contracts.py; Z6, Z8), on fixtures (CPU):
  preflight
    - shapes: four distinct cached shapes of 104, 100, 96 and 92 s, then the 105.7 s maximum;
    - the route check passes only if the cache holds 4 entries before the fifth input and the fifth
      input's route is eager (no new entry, the eager fallback called once). A cache that captures the
      fifth, or holds fewer than 4 entries, fails;
    - the lease is peak + max(25%, 1 GiB);
    - the time check needs cold start-up + capture + 2 x the longest utterance within the hold cap and
      capture + 2 x the longest within the work budget.
  runtime contracts at x2
    - recognition: the reference is g2's rows (text, token IDs, timestamps) on the 24 timing workloads,
      identity-bound to the gate run. A difference fails. Equality with x3 is reported as a
      hypothesis, never asserted.
    - memory: test_runtime_fresh's contract with b1 as the reference: per clip the fresh peak is at most
      1.02 x b1's recorded P-memory peak (max over its runs) and at least 500 MB below R-expanded's (b3),
      transcripts equal b1's. It refuses a missing benchmark, a b1 record of another identity, b1 runs with
      different transcripts, an R-expanded record not labelled b3, or a missing clip.

    gpu1bit/shared/python -m gpu1bit.tests.run --cpu gpu1bit.tests.test_x2_preflight_runtime
"""
from __future__ import annotations

import unittest

from gpu1bit.x2 import preflight as PF
from gpu1bit.x2 import runtime_contracts as RC


class _Cache:
    def __init__(self, entries, capture_fifth=False):
        self.entries = {i: None for i in range(entries)}
        self.max_graphs = 4
        self.calls = 0
        self.capture_fifth = capture_fifth

    def original_forward(self, *a, **k):
        self.calls += 1

    def __call__(self, key):
        if key not in self.entries:
            if len(self.entries) >= self.max_graphs and not self.capture_fifth:
                return self.original_forward()
            self.original_forward()
            self.original_forward()
            self.entries[key] = None


class Preflight(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(PF.CACHED_SECONDS, (104.0, 100.0, 96.0, 92.0))
        self.assertEqual(PF.EAGER_SECONDS, 105.7)

    def test_route(self):
        c = _Cache(4)
        self.assertEqual(PF.fifth_route(c, lambda: c("fifth"))["route"], "eager")
        c = _Cache(4, capture_fifth=True)
        with self.assertRaisesRegex(AssertionError, "eager"):
            PF.fifth_route(c, lambda: c("fifth"))
        c = _Cache(3)
        with self.assertRaisesRegex(AssertionError, "4 entries"):
            PF.fifth_route(c, lambda: c("fifth"))

    def test_lease_and_time(self):
        self.assertEqual(PF.lease_bytes(2 * 2**30), 3 * 2**30)                       # +1 GiB beats +25%
        self.assertEqual(PF.lease_bytes(8 * 2**30), 10 * 2**30)                      # +25%
        self.assertTrue(PF.time_ok(startup_s=30, capture_s=20, longest_s=60)["ok"])
        self.assertFalse(PF.time_ok(startup_s=30, capture_s=20, longest_s=240)["ok"])   # 2 x 240 > budget
        self.assertFalse(PF.time_ok(startup_s=200, capture_s=60, longest_s=150)["ok"])  # hold cap


class DurationBudget(unittest.TestCase):
    """X7's job duration budget, from the gate run's measured batch-1 throughput: the full-dev work time per
    audio second and per utterance, each extrapolated to the test sets (the larger is kept), plus a cold
    start-up and capture per chunk (preflight) for the chunks the work budget implies; the job timeout is
    twice that. Missing measurements are refused."""

    def test_budget(self):
        b = PF.compute_budget(work_s=[300.0, 100.0], dev_utterances=20000, dev_audio_s=100000.0,
                              test_utterances=100000, test_audio_s=400000.0, startup_s=30.0, capture_s=20.0)
        self.assertAlmostEqual(b["per_audio_second_s"], 400.0 / 100000)
        self.assertAlmostEqual(b["per_utterance_s"], 400.0 / 20000)
        self.assertAlmostEqual(b["predicted_work_s"], max(400.0 / 100000 * 400000, 400.0 / 20000 * 100000))   # 2000 s
        self.assertEqual(b["chunks"], 5)                                   # ceil(2000 / 480)
        self.assertAlmostEqual(b["predicted_s"], 2000.0 + 5 * 50.0)
        self.assertEqual(b["timeout_s"], int(2 * (2000.0 + 5 * 50.0)))
        for bad in ({"work_s": []}, {"dev_audio_s": 0.0}, {"test_utterances": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                PF.compute_budget(**{**dict(work_s=[1.0], dev_utterances=1, dev_audio_s=1.0, test_utterances=1,
                                            test_audio_s=1.0, startup_s=1.0, capture_s=1.0), **bad})


class Recognition(unittest.TestCase):
    def rows(self, n=24):
        return {f"u{i}": {"text": "a b", "tokens": [1, 2], "timestamps": [0, 3]} for i in range(n)}

    def test_equal_and_differences(self):
        ref = self.rows()
        out = RC.compare_recognition(self.rows(), ref, x3_rows=self.rows())
        self.assertTrue(out["ok"])
        self.assertEqual(out["x3_hypothesis"]["status"], "reported, not asserted")
        bad = self.rows()
        bad["u5"]["timestamps"] = [0, 4]
        out = RC.compare_recognition(bad, ref)
        self.assertFalse(out["ok"])
        self.assertEqual(out["differences"], ["u5"])
        x3 = self.rows()
        x3["u1"]["tokens"] = [9]
        out = RC.compare_recognition(self.rows(), ref, x3_rows=x3)
        self.assertTrue(out["ok"], "x3 differences never fail the x2 contract")
        self.assertEqual(out["x3_hypothesis"]["differences"], ["u1"])
        with self.assertRaisesRegex(ValueError, "24"):
            RC.compare_recognition(self.rows(23), self.rows(23))


class Memory(unittest.TestCase):
    """test_runtime_fresh's contract with the references replaced (Z6): per dev clip, the fresh-process peak
    including first capture is at most 1.02 x b1's recorded P-memory peak (the max over its two P-clips
    recordings, clips1 and clips2) and at least 500 MB below R-expanded's (b3), and the transcripts equal b1's.
    Coverage (Codex x2 c1, finding 6): all three frozen dev clips must be in every required recording (both
    orders of b1 and of b3), in both reference maps and in the fresh measurement; anything less is refused."""
    MB = 1_000_000
    CLIPS = ("c3", "c10", "c30")

    def bench(self, roster_id="b1", identity="I", peaks=(1000, 1200, 1500), texts=("a", "b", "c"), clips=CLIPS):
        variant = {"b1": "MB1-B-hybrid-x2", "b3": "R-expanded"}[roster_id]
        return {"variant": variant, "variant_info": {"identity": identity, "roster_id": roster_id},
                "workloads": [{"record": {"id": c}, "hypothesis": t, "process_sampled_peak_including_first_bytes": b * self.MB}
                              for c, b, t in zip(self.CLIPS, peaks, texts) if c in clips]}

    def fresh(self, peaks=(1000, 1200, 1500), texts=("a", "b", "c")):
        return {c: {"peak": b * self.MB, "texts": [t] * 3} for c, b, t in zip(self.CLIPS, peaks, texts)}

    def b1(self, **kw):
        return {"clips1": self.bench(**kw), "clips2": self.bench(peaks=(1010, 1190, 1500))}

    def b3(self, **kw):
        base = dict(roster_id="b3", identity="R", peaks=(2000, 2400, 2900), texts=("x", "y", "z"))
        return {"clips1": self.bench(**{**base, **kw}), "clips2": self.bench(**{**base, **{k: v for k, v in kw.items() if k == "peaks"}})}

    def refs(self):
        return (RC.pmemory_reference(self.b1(), runtime_identity="I", clip_ids=self.CLIPS),
                RC.r_expanded_reference(self.b3(), clip_ids=self.CLIPS))

    def test_reference_is_the_max_over_runs(self):
        b1, b3 = self.refs()
        self.assertEqual(b1, {"c3": {"peak": 1010 * self.MB, "text": "a"}, "c10": {"peak": 1200 * self.MB, "text": "b"},
                              "c30": {"peak": 1500 * self.MB, "text": "c"}})
        self.assertEqual(b3, {"c3": 2000 * self.MB, "c10": 2400 * self.MB, "c30": 2900 * self.MB})

    def test_bounds(self):
        b1, b3 = self.refs()
        self.assertTrue(RC.compare_pmemory(self.fresh(), b1, b3, clip_ids=self.CLIPS)["ok"])
        self.assertTrue(RC.compare_pmemory(self.fresh(peaks=(1030, 1224, 1530)), b1, b3, clip_ids=self.CLIPS)["ok"])
        out = RC.compare_pmemory(self.fresh(peaks=(1000, 1225, 1500)), b1, b3, clip_ids=self.CLIPS)   # 1225 > 1.02 x 1200
        self.assertFalse(out["ok"])
        self.assertEqual(sorted(out["differences"]), ["c10"])
        tight = RC.r_expanded_reference(self.b3(peaks=(1400, 2400, 2900)), clip_ids=self.CLIPS)
        self.assertFalse(RC.compare_pmemory(self.fresh(), b1, tight, clip_ids=self.CLIPS)["ok"])     # not 500 MB below b3
        self.assertFalse(RC.compare_pmemory(self.fresh(texts=("a", "B", "c")), b1, b3, clip_ids=self.CLIPS)["ok"])

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            RC.pmemory_reference(self.b1(identity="other"), runtime_identity="I", clip_ids=self.CLIPS)
        with self.assertRaisesRegex(ValueError, "identity"):
            RC.pmemory_reference({"clips1": self.bench(), "clips2": self.bench(identity="other")}, runtime_identity="I",
                                 clip_ids=self.CLIPS)
        for missing in (None, {}, {"clips1": self.bench()}, {"clips1": self.bench(), "clips2": None}):
            with self.subTest(b1=missing), self.assertRaisesRegex(ValueError, "benchmark|recording"):
                RC.pmemory_reference(missing, runtime_identity="I", clip_ids=self.CLIPS)
        with self.assertRaisesRegex(ValueError, "b3"):
            RC.r_expanded_reference(self.b1(), clip_ids=self.CLIPS)
        with self.assertRaisesRegex(ValueError, "transcript"):
            RC.pmemory_reference({"clips1": self.bench(), "clips2": self.bench(texts=("a", "b", "C"))}, runtime_identity="I",
                                 clip_ids=self.CLIPS)
        b1, b3 = self.refs()
        with self.assertRaisesRegex(ValueError, "clip"):
            RC.compare_pmemory({k: v for k, v in self.fresh().items() if k != "c30"}, b1, b3, clip_ids=self.CLIPS)

    def test_incomplete_coverage_is_refused(self):
        """Finding 6: an interrupted recording with only the shortest clip, or empty maps, never passes."""
        with self.assertRaisesRegex(ValueError, "clip"):
            RC.pmemory_reference(self.b1(clips=("c3",)), runtime_identity="I", clip_ids=self.CLIPS)
        with self.assertRaisesRegex(ValueError, "clip"):
            RC.r_expanded_reference(self.b3(clips=("c3", "c10")), clip_ids=self.CLIPS)
        b1, b3 = self.refs()
        for name, (ref, rexp) in {"empty b1": ({}, b3), "empty b3": (b1, {}), "both empty": ({}, {}),
                                  "b1 missing c30": ({k: v for k, v in b1.items() if k != "c30"}, b3),
                                  "b3 missing c10": (b1, {k: v for k, v in b3.items() if k != "c10"})}.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, "clip"):
                RC.compare_pmemory(self.fresh(), ref, rexp, clip_ids=self.CLIPS)
        with self.assertRaisesRegex(ValueError, "clip"):
            RC.compare_pmemory(self.fresh(), b1, b3, clip_ids=())
        self.assertEqual(RC.frozen_clip_ids(), tuple(r["id"] for r in __import__("gpu1bit.shared.workloads", fromlist=["x"]).load()["dev_clips"]))
