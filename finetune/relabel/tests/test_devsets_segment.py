"""Segmenter contracts (labeling plan Part A "Segments are 1 to 30 s ..."; reused by Part B):
1-30 s segments cut at the source's own utterance timestamps, else at forced-alignment sentence
boundaries; failed alignments dropped and counted; stable ids that drops never renumber; references
without markup. The aligner is a stand-in that returns given unit times and scores.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p "test_devsets_*" -v
"""
from __future__ import annotations

import unittest

import numpy as np

from devsets import segment as sg
from devsets import transcript as tx

SR = sg.SAMPLE_RATE


class FakeAligner:
    """Returns the given (start, end, score) per unit, in order; checks the unit count."""

    min_score = 0.5

    def __init__(self, spans: list[tuple[float, float, float]]) -> None:
        self.spans, self.calls = spans, 0

    def align(self, audio, units):
        from devsets.align import UnitSpan

        self.calls += 1
        assert len(units) == len(self.spans), (len(units), len(self.spans))
        return [UnitSpan(a, b, None if u is None else s) for (a, b, s), u in zip(self.spans, units)]


def timed_words(sentences: list[tuple[float, float, str]], score: float = 0.9):
    """(text, spans): words of each (start, end, sentence) spread evenly over [start, end]."""
    words, spans = [], []
    for start, end, sentence in sentences:
        ws = sentence.split()
        step = (end - start) / len(ws)
        for i, w in enumerate(ws):
            words.append(w)
            spans.append((start + i * step, start + (i + 0.8) * step, score))
    return " ".join(words), spans


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


class InvariantsMixin:
    def check(self, result: sg.Result, audio: np.ndarray) -> None:
        last_end = -1.0
        for seg in sorted(result.segments, key=lambda s: s.start):
            seconds = sg.cut_seconds(seg.start, seg.end)
            self.assertGreaterEqual(seconds, sg.MIN_SECONDS, seg)
            self.assertLessEqual(seconds, sg.MAX_SECONDS, seg)
            self.assertGreaterEqual(seg.start, 0.0)
            self.assertLessEqual(sg.samples(seg.end), len(audio))
            self.assertGreaterEqual(seg.start, last_end - 1e-9, "segments overlap")
            last_end = seg.end
            self.assertNotIn("<", seg.text)
        ids = [s.id for s in result.segments]
        self.assertEqual(len(ids), len(set(ids)))


class TimestampedTest(InvariantsMixin, unittest.TestCase):
    def test_cut_exactly_at_source_timestamps_with_index_ids(self) -> None:
        audio = silence(60)
        utts = [sg.Utterance(10.0, 14.5, "Second one."), sg.Utterance(1.0, 5.0, "<UNSURE>Hello</UNSURE> there."),
                sg.Utterance(20.0, 20.6, "Yes."), sg.Utterance(30.0, 33.0, "<UNIN/>"),
                sg.Utterance(40.0, 45.0, "Okay <UNIN/> fine.")]
        result = sg.segment(audio, "day1 consult/01", utts)
        self.check(result, audio)
        by_id = {s.id: s for s in result.segments}
        self.assertEqual(sorted(by_id), ["day1_consult_01-0000", "day1_consult_01-0001", "day1_consult_01-0004"])
        first = by_id["day1_consult_01-0000"]
        self.assertEqual((first.start, first.end, first.text, first.aligned), (1.0, 5.0, "Hello there.", False))
        self.assertFalse(first.untranscribed)
        self.assertTrue(by_id["day1_consult_01-0004"].untranscribed)
        self.assertEqual(result.drops, {"too_short": 1, "empty_reference": 1})
        self.assertAlmostEqual(result.dropped_seconds["too_short"], 0.6)

    def test_long_utterance_needs_the_aligner(self) -> None:
        audio = silence(80)
        text, spans = timed_words([(0.0, 12.0, "Alpha beta gamma delta epsilon."),
                                   (13.0, 25.0, "Zeta eta theta iota kappa."),
                                   (26.0, 38.0, "Lambda mu nu xi omicron.")])
        utts = [sg.Utterance(2.0, 3.5, "First."), sg.Utterance(10.0, 48.0, text)]
        without = sg.segment(audio, "r", utts)
        self.assertEqual(without.drops, {"too_long": 1})
        self.assertEqual([s.id for s in without.segments], ["r-0000"])
        aligner = FakeAligner(spans)
        result = sg.segment(audio, "r", utts, aligner=aligner)
        self.check(result, audio)
        pieces = [s for s in result.segments if s.id.startswith("r-0001-")]
        self.assertGreaterEqual(len(pieces), 2)
        self.assertTrue(all(s.aligned and 10.0 <= s.start and s.end <= 48.0 for s in pieces))
        self.assertEqual(" ".join(s.text for s in pieces), text)


class AlignedTest(InvariantsMixin, unittest.TestCase):
    SENTENCES = [(0.5, 4.5, "One two three four."), (5.5, 9.5, "Five six seven eight."),
                 (10.5, 14.0, "Nine ten eleven twelve."), (15.5, 19.5, "Thirteen fourteen fifteen."),
                 (20.5, 24.5, "Sixteen seventeen eighteen."), (25.5, 29.5, "Nineteen twenty done.")]

    def run_segment(self, sentences, audio_seconds=32.0, score=0.9, **kwargs):
        text, spans = timed_words(sentences, score)
        audio = silence(audio_seconds)
        return sg.segment(audio, "rec", text, aligner=FakeAligner(spans), **kwargs), audio, text, spans

    def test_sentences_are_packed_up_to_the_target(self) -> None:
        result, audio, text, _ = self.run_segment(self.SENTENCES)
        self.check(result, audio)
        self.assertEqual([s.id for s in result.segments], ["rec-0000", "rec-0001"])
        first, second = result.segments
        self.assertTrue(first.text.endswith("twelve.") and second.text.startswith("Thirteen"))
        self.assertAlmostEqual(first.start, 0.5 - sg.PAD_SECONDS)
        self.assertLessEqual(first.end - first.start, sg.TARGET_SECONDS + 2 * sg.PAD_SECONDS)
        self.assertEqual(" ".join(s.text for s in result.segments), text)
        self.assertTrue(all(s.aligned for s in result.segments))

    def test_failed_alignment_is_dropped_counted_and_never_renumbers(self) -> None:
        text, spans = timed_words(self.SENTENCES)
        spans = [(a, b, 0.1) if i < 12 else (a, b, s) for i, (a, b, s) in enumerate(spans)]
        audio = silence(32)
        result = sg.segment(audio, "rec", text, aligner=FakeAligner(spans))
        self.assertEqual([s.id for s in result.segments], ["rec-0001"])
        self.assertEqual(result.drops, {"alignment_failed": 1})
        self.assertGreater(result.dropped_seconds["alignment_failed"], 10)

    def test_a_stretched_word_fails_the_alignment(self) -> None:
        text, spans = timed_words(self.SENTENCES)
        a, b, s = spans[2]
        spans[2] = (a, a + sg.MAX_WORD_SECONDS + 0.5, s)
        spans[3] = (a + sg.MAX_WORD_SECONDS + 0.6, a + sg.MAX_WORD_SECONDS + 0.7, s)
        result = sg.segment(silence(32), "rec", text, aligner=FakeAligner(spans))
        self.assertEqual(result.drops["alignment_failed"], 1)

    def test_long_sentence_is_split_at_its_largest_pause(self) -> None:
        words = [f"w{i}" for i in range(40)]
        spans = [(i * 0.9, i * 0.9 + 0.6, 0.9) for i in range(20)] + \
                [(25.0 + i * 0.9, 25.0 + i * 0.9 + 0.6, 0.9) for i in range(20)]
        audio = silence(45)
        result = sg.segment(audio, "rec", " ".join(words), aligner=FakeAligner(spans))
        self.check(result, audio)
        self.assertEqual(len(result.segments), 2)
        self.assertTrue(result.segments[0].text.endswith("w19") and result.segments[1].text.startswith("w20"))

    def test_dictated_boundary_words_end_sentences(self) -> None:
        sentences = [(0.5, 9.5, "Chart number one two three period"), (10.5, 19.5, "Date of birth four five period"),
                     (20.5, 29.5, "The patient is well period")]
        plain, audio, _, _ = self.run_segment(sentences)
        dictated, _, _, _ = self.run_segment(sentences, boundary_words=["period"])
        self.check(dictated, audio)
        self.assertEqual(len(plain.segments), 1)  # no sentence end, and it fits in 30 s
        self.assertEqual(len(dictated.segments), 3)
        self.assertTrue(all(s.text.endswith("period") for s in dictated.segments))

    def test_a_short_sentence_joins_its_nearer_neighbour(self) -> None:
        sentences = [(0.5, 14.0, "A long first sentence with many words in it."), (14.3, 14.6, "Yes."),
                     (20.0, 34.0, "Another long sentence follows after a pause here.")]
        result, audio, _, _ = self.run_segment(sentences, audio_seconds=36.0)
        self.check(result, audio)
        self.assertEqual(len(result.segments), 2)
        self.assertTrue(result.segments[0].text.endswith("Yes."))

    def test_ids_are_stable_across_runs(self) -> None:
        a, _, _, _ = self.run_segment(self.SENTENCES)
        b, _, _, _ = self.run_segment(self.SENTENCES)
        self.assertEqual(a.segments, b.segments)

    def test_markup_only_transcript_is_an_empty_reference(self) -> None:
        result = sg.segment(silence(10), "rec", "<inaudible> [laughter]", aligner=FakeAligner([]))
        self.assertEqual(result.segments, [])
        self.assertEqual(result.drops, {"empty_reference": 1})


class TranscriptTest(unittest.TestCase):
    def test_tokens(self) -> None:
        toks = tx.tokens("<UNSURE>Hello</UNSURE> Dr. O'Brien — 9% of $1.5 <inaudible> 2020 café")
        self.assertEqual([t.text for t in toks], ["Hello", "Dr.", "O'Brien", "9%", "of", "$1.5", "", "2020", "café"])
        self.assertEqual([t.chars for t in toks], ["hello", "dr", "o'brien", "ninepercent", "of",
                                                   "onepointfivedollars", None, "twentytwenty", "cafe"])

    def test_number_words(self) -> None:
        self.assertEqual(tx.number_words("1998"), "nineteen ninety eight")
        self.assertEqual(tx.number_words("2005"), "two thousand five")
        self.assertEqual(tx.number_words("1,234.5"), "one thousand two hundred thirty four point five")
        self.assertEqual(tx.number_words("21st"), "twenty first")
        self.assertEqual(tx.number_words("1990s"), "nineteen nineties")

    def test_sentence_ends(self) -> None:
        self.assertTrue(tx.ends_sentence(tx.Token("done.", "done")))
        self.assertTrue(tx.ends_sentence(tx.Token('"Why?"', "why")))
        self.assertFalse(tx.ends_sentence(tx.Token("period", "period")))
        self.assertTrue(tx.ends_sentence(tx.Token("period", "period"), frozenset({"period"})))

    def test_safe_id(self) -> None:
        self.assertEqual(sg.safe_id("a b/c:d"), "a_b_c_d")
        with self.assertRaises(ValueError):
            sg.safe_id("///")


if __name__ == "__main__":
    unittest.main()
