"""Contract tests for the Bend WER engine: S, D, I identical to wer.edit_counts and to relabel_align.align, the same
corpus_wer dict, and malformed input rejected without writing output.

    finetune/relabel/python -m unittest discover -s finetune/bend-wer/tests -v

The binary is build/bend-wer (or $BEND_WER); it is rebuilt through the repo flake when missing or older than a .bend
or .c source it is built from.
"""
from __future__ import annotations

import itertools
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.append(str(REPO / "finetune" / "relabel"))
sys.path.append(str(REPO / "finetune" / "whisper-ternary"))
import bend_wer  # noqa: E402
import relabel_align  # noqa: E402
import wer  # noqa: E402

_BUILT = False


def build() -> Path:
    """build/bend-wer, rebuilt when a source is newer (or $BEND_WER as given)."""
    global _BUILT
    if os.environ.get("BEND_WER") or _BUILT:
        return bend_wer.binary()
    out = HERE / "build" / "bend-wer"
    sources = [HERE / "scorer.bend", HERE / "wer.bend", HERE.parent / "bend-planner" / "plan.bend",
               *(HERE.parent / "bend-planner" / "effs").glob("*")]
    if not out.exists() or any(s.stat().st_mtime > out.stat().st_mtime for s in sources):
        out.parent.mkdir(exist_ok=True)
        subprocess.run(["nix", "develop", str(REPO), "-c", "bend", "scorer.bend", "-o", str(out)], cwd=HERE,
                       check=True)
    _BUILT = True
    return out


def reference(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """wer.edit_counts, after checking that relabel_align.align agrees with it."""
    expected = wer.edit_counts(ref, hyp)
    a = relabel_align.align(ref, hyp)
    assert (a.S, a.D, a.I) == expected, (ref, hyp)
    return expected


EDGES = {
    "both empty": ([], []),
    "empty ref": ([], ["a", "b", "c"]),
    "empty hyp": (["a", "b", "c"], []),
    "one word, equal": (["a"], ["a"]),
    "one word, different": (["a"], ["b"]),
    "identical": ("the cat sat on the mat".split(), "the cat sat on the mat".split()),
    "repeats": ("a a a a".split(), "a a".split()),
    "repeats inserted": ("a a".split(), "a a a a a".split()),
    "swap (sub or del+ins tie)": ("a b".split(), "b a".split()),
    "shift tie": ("a b c".split(), "b c d".split()),
    "all different": ("a b c".split(), "x y z w".split()),
    "deletion at either end": ("x a b x".split(), "a b".split()),
    "interleaved": ("a b a b a b".split(), "b a b a".split()),
    "deletion before insertion": ("a b a b".split(), "b a a b a".split()),
    "unicode and empty-looking words": (["café", "naïve", "a"], ["cafe", "naïve", "a", "​"]),
}


def exhaustive_pairs(alphabet: str = "ab", longest: int = 5) -> list[tuple[list[str], list[str]]]:
    """Every pair of word lists over the alphabet up to the length: every tie the backtrace can meet there."""
    lists = [list(w) for n in range(longest + 1) for w in itertools.product(alphabet, repeat=n)]
    return [(r, h) for r in lists for h in lists]


def random_pairs(rng: random.Random, n: int) -> list[tuple[list[str], list[str]]]:
    pairs = []
    for _ in range(n):
        vocab = rng.choice([1, 2, 3, 5, 50])
        words = [f"w{k}" for k in range(vocab)]
        ref = [rng.choice(words) for _ in range(rng.choice([0, 1, 2, rng.randint(0, 12), rng.randint(0, 60)]))]
        kind = rng.random()
        if kind < 0.4:  # a noisy copy: substitutions, deletions and insertions of the reference
            hyp = []
            for w in ref:
                r = rng.random()
                if r < 0.1:
                    continue
                hyp.append(rng.choice(words) if r < 0.25 else w)
                if rng.random() < 0.1:
                    hyp.append(rng.choice(words))
        else:
            hyp = [rng.choice(words) for _ in range(rng.choice([0, 1, rng.randint(0, 12), rng.randint(0, 60)]))]
        pairs.append((ref, hyp))
    return pairs


def raw(tmp: Path, ids, offsets, extra: list[str] | None = None) -> tuple[subprocess.CompletedProcess, Path]:
    ip, op, cp = tmp / "ids.u32", tmp / "offsets.u32", tmp / "counts.u32"
    cp.unlink(missing_ok=True)
    for path, values in ((ip, ids), (op, offsets)):
        if isinstance(values, bytes):
            path.write_bytes(values)
        else:
            np.asarray(values, dtype="<u4").tofile(path)
    proc = subprocess.run([str(build()), str(ip), str(op), str(cp), *(extra or [])], capture_output=True, text=True,
                          timeout=600)
    return proc, cp


REJECT = {
    "ids not a multiple of 4": (b"\x01\x00\x00", [0, 0, 0]),
    "offsets not a multiple of 4": ([1, 2], b"\x00\x00\x00\x00\x01"),
    "offsets empty": ([1, 2], b""),
    "offsets even": ([1, 2], [0, 1]),
    "offsets even, longer": ([1, 2, 3, 4], [0, 1, 2, 4]),
    "offsets not from 0": ([1, 2], [1, 1, 2]),
    "offsets not to n": ([1, 2, 3], [0, 1, 2]),
    "offsets past n": ([1, 2], [0, 1, 3]),
    "offsets decrease": ([1, 2, 3, 4], [0, 2, 1, 3, 4]),
}


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build()

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def assert_matches(self, pairs, **kwargs):
        got = bend_wer.edit_counts_many(pairs, **kwargs)
        self.assertEqual(got, [reference(r, h) for r, h in pairs])

    def test_edges_match_reference(self):
        for name, (ref, hyp) in EDGES.items():
            with self.subTest(name):
                self.assertEqual(bend_wer.edit_counts(ref, hyp), reference(ref, hyp))
        self.assert_matches(list(EDGES.values()))

    def test_random_matches_reference(self):
        rng = random.Random(20261026)
        for batch in range(8):
            with self.subTest(batch=batch):
                self.assert_matches(random_pairs(rng, rng.choice([1, 2, 7, 300])))

    def test_exhaustive_small_matches_reference(self):
        self.assert_matches(exhaustive_pairs())

    def test_long_utterance(self):
        rng = random.Random(7)
        ref = [f"w{rng.randrange(30)}" for _ in range(400)]
        hyp = [w if rng.random() < 0.7 else f"w{rng.randrange(30)}" for w in ref][:350]
        self.assert_matches([(ref, hyp), (hyp, ref)])

    def test_threads_and_chunks_do_not_change_counts(self):
        pairs = random_pairs(random.Random(3), 500)
        expected = [reference(r, h) for r, h in pairs]
        for kwargs in ({"threads": 1}, {"threads": 8}, {"chunk_ids": 1}, {"chunk_ids": 97, "threads": 2},
                       {"chunk_utts": 7}):
            with self.subTest(**kwargs):
                self.assertEqual(bend_wer.edit_counts_many(pairs, **kwargs), expected)

    def test_ids_are_per_call(self):
        """Only equality of words matters: the same pair scores the same whatever else is in the batch."""
        pair = ("a b c d".split(), "a c d e".split())
        alone = bend_wer.edit_counts(*pair)
        mixed = bend_wer.edit_counts_many([("e d c b a".split(), "x".split()), pair])[1]
        self.assertEqual(alone, mixed)
        self.assertEqual(alone, reference(*pair))

    def test_corpus_wer_matches(self):
        rng = random.Random(11)
        records = [{"ref_norm": " ".join(r), "hyp_norm": " ".join(h), "id": k}
                   for k, (r, h) in enumerate(random_pairs(rng, 200))]
        self.assertEqual(bend_wer.corpus_wer(records), wer.corpus_wer(records))
        for empty in ([], [{"ref_norm": "", "hyp_norm": "a b"}]):
            with self.subTest(records=empty):
                with self.assertRaises(ValueError):
                    wer.corpus_wer(empty)
                with self.assertRaises(ValueError):
                    bend_wer.corpus_wer(empty)

    def test_output_format(self):
        ids, offsets = bend_wer.encode([("a b".split(), "a c x".split()), ([], ["q"])])
        proc, cp = raw(self.tmp, ids, offsets)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(cp.read_bytes(), np.asarray([1, 0, 1, 0, 0, 1], dtype="<u4").tobytes())
        proc, cp = raw(self.tmp, b"", [0])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(cp.read_bytes(), b"")

    def test_rejects_malformed_input(self):
        for name, (ids, offsets) in REJECT.items():
            with self.subTest(name):
                proc, cp = raw(self.tmp, ids, offsets)
                self.assertEqual(proc.returncode, 1, proc.stderr)
                self.assertTrue(proc.stderr.startswith("bend-wer: "), proc.stderr)
                self.assertFalse(cp.exists())

    def test_missing_input_and_usage(self):
        proc = subprocess.run([str(build()), str(self.tmp / "none.u32"), str(self.tmp / "none.u32"),
                               str(self.tmp / "out.u32")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 1)
        self.assertFalse((self.tmp / "out.u32").exists())
        proc = subprocess.run([str(build()), "a", "b"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("usage", proc.stderr)

    def test_wrapper_reports_failures(self):
        with self.assertRaises(bend_wer.BendWerError):
            bend_wer.run(np.asarray([1], dtype="<u4"), np.asarray([0, 1], dtype="<u4"))


if __name__ == "__main__":
    unittest.main()
