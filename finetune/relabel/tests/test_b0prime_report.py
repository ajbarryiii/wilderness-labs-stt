"""Contract tests (CPU): stage 3's v4 reporting (b0prime plan v4 "Reporting"): style-neutral WER (the standard
variant equals the scorer's errors; each canonicalizer applied identically to reference and hypothesis; per-utterance
errors summed into the bootstrap tables), formatted metrics (any-mark and per-mark precision, recall and F1, case
errors, formatted WER), the any-mark F1 flag and the gate note (standard fails, style-neutral passes)."""
from __future__ import annotations

import collections
import unittest
from unittest import mock

import b0prime_report as rep
import style_metrics as sm


def record(i: str, ref: str, hyp: str) -> dict:
    import evaluate
    rn, hn = evaluate.normalize(ref), evaluate.normalize(hyp)
    s, d, ins = evaluate.edit_counts(rn.split(), hn.split())
    return {"id": i, "ref": ref, "hyp": hyp, "ref_norm": rn, "hyp_norm": hn, "S": s, "D": d, "I": ins,
            "scored": True}


class StyleNeutral(unittest.TestCase):
    def test_std_is_the_scorers_errors(self):
        for ref, hyp in (("The A. P. I. call is here.", "the API call is here"), ("uh I I think so", "I think so"),
                         ("a data set of health care costs", "a dataset of healthcare costs")):
            r = record("x", ref, hyp)
            self.assertEqual(sm.utterance_errors(ref, hyp)["std"], (r["S"] + r["D"] + r["I"], len(r["ref_norm"].split())))

    def test_each_group_neutralizes_its_style_only(self):
        e = sm.utterance_errors("call the A. P. I. now", "call the API now")
        self.assertGreater(e["std"][0], 0)
        self.assertEqual(e["acr"][0], 0)
        self.assertEqual(e["acr"], e["all"])
        self.assertEqual(e["cmp"][0], e["std"][0])
        e = sm.utterance_errors("a data set of health care costs", "a dataset of healthcare costs")
        self.assertEqual((e["cmp"][0], e["all"][0]), (0, 0))
        self.assertGreater(e["std"][0], 0)
        e = sm.utterance_errors("I think so", "I I think er so")
        self.assertEqual(e["dis"][0], 0)
        self.assertGreater(e["std"][0], 0)
        # a real word error stays an error in every variant
        self.assertTrue(all(x[0] >= 1 for x in sm.utterance_errors("the cat sat", "the bat sat").values()))
        # the same canonicalizers on both sides: identical texts have no error in any variant
        for text in ("The A. P. I., data set, uh-huh, I I, er", "N A S A and the U.S. and C T X-ray"):
            self.assertTrue(all(x[0] == 0 for x in sm.utterance_errors(text, text).values()), text)

    def test_score_records_and_contributions(self):
        recs = [record("a", "call the A. P. I. now", "call the API now"), record("b", "the cat", "the cat"),
                {**record("c", "", "noise"), "scored": False}]
        s = sm.score_records(recs)
        self.assertEqual(s["ids"], ["a", "b"])
        self.assertAlmostEqual(s["contribution"]["acr"], s["wer"]["std"] - s["wer"]["acr"])
        self.assertGreater(s["contribution"]["acr"], 0)
        self.assertEqual(s["contribution"]["std"], 0)
        table = rep.variant_table({"B0": s, "B0p": sm.score_records(recs[:2])}, "std", lambda i: i)
        self.assertEqual((table.clusters, table.utterances), (["a", "b"], 2))
        self.assertAlmostEqual(table.wer("B0"), s["wer"]["std"])


class Formatted(unittest.TestCase):
    def test_mark_precision_recall_and_case(self):
        c = sm.formatted_counts("Hello, world. How are you?", "hello world. How are you.")
        s = sm.formatted_summary(c)
        self.assertEqual((s["any_mark"]["tp"], s["any_mark"]["ref"], s["any_mark"]["hyp"]), (2, 3, 2))
        self.assertAlmostEqual(s["any_mark"]["precision"], 1.0)
        self.assertAlmostEqual(s["any_mark"]["recall"], 2 / 3)
        self.assertAlmostEqual(s["any_mark"]["f1"], 0.8)
        self.assertEqual((s["per_mark"]["?"]["tp"], s["per_mark"]["?"]["hyp"]), (0, 0))
        self.assertEqual((s["per_mark"]["."]["tp"], s["per_mark"]["."]["hyp"]), (1, 2))
        self.assertAlmostEqual(s["case_error_rate"], 1 / 5)                       # "Hello" / "hello"
        self.assertGreater(s["formatted_wer"], 0)
        self.assertEqual(sm.formatted_summary(sm.formatted_counts("Yes.", "Yes."))["formatted_wer"], 0.0)
        self.assertNotIn("case_error_rate", sm.formatted_summary(c, punct_only=True))

    def test_the_f1_flag(self):
        sets = {"tech-dev-1": ("Hello, world. Bye.", "Hello world bye"), "med-dev-1": ("A, b.", "A, b.")}

        def records(model_dir, name):
            if name not in sets:
                return None
            ref, hyp = sets[name]
            return [record("x", ref, ref if model_dir == "B0" else hyp)]
        with mock.patch.object(rep, "set_records", records), mock.patch.object(rep, "t_star_set_records", lambda n: None):
            f = rep.formatted_section({"B0": "B0", "B0p": "B0p"})
        self.assertEqual(f["f1_flags"], ["tech-dev-1"])


class GateNote(unittest.TestCase):
    def test_report_both_when_standard_fails_and_style_neutral_passes(self):
        h4 = {"point": -0.30, "ci": [-0.45, -0.15]}                       # B0' - B0 on "all": better by 0.30
        style = {"contrast": {"all": {"H4": h4, "sets": {n: {"point": -0.3} for n in rep.tb.H4}}}}
        ok = {"pass": True}
        g = rep.style_gate(style, ok, {"pass": False})
        self.assertTrue(g["decision"]["pass"])
        self.assertTrue(g["report_both_to_aj"])
        self.assertEqual(g["delta"], {"point": 0.30, "ci": [0.15, 0.45]})
        self.assertFalse(rep.style_gate(style, ok, {"pass": True})["report_both_to_aj"])
        worse = {"contrast": {"all": {"H4": {"point": 0.1, "ci": [0.0, 0.2]}, "sets": {n: {"point": 0.1}
                                                                                          for n in rep.tb.H4}}}}
        self.assertFalse(rep.style_gate(worse, ok, {"pass": False})["report_both_to_aj"])
        self.assertEqual(rep.style_gate({"contrast": {}}, ok, {"pass": False})["status"], "pending")


if __name__ == "__main__":
    unittest.main()
