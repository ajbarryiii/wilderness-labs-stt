"""Mutation tests: deliberately wrong variants of wer.bend, each of which the proof gate and the fidelity tests must
reject. They show the laws are not vacuous (a wrong cost breaks them, numerically as well as in PROOF.bend) and why
the fidelity tests are needed beside them (a different tie-break keeps every law true).

    finetune/relabel/python -m unittest discover -s finetune/bend-wer/tests -p 'test_mutants.py' -v

Each mutant is a copy of bend-wer and bend-planner's sources in a temporary directory, with one edit to wer.bend.
"""
from __future__ import annotations

import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(REPO / "finetune" / "whisper-ternary"))
import bend_wer  # noqa: E402
import wer  # noqa: E402
from test_contract import EDGES, exhaustive_pairs, random_pairs  # noqa: E402

# name: (text in wer.bend, its replacement, laws it must break on some input; none for a tie-break)
MUTANTS = {
    "tie-break: insertion before deletion": (
        "Nat.is_eq(c, uc), eq, ku, kg, kl)",
        "Bool.not(Nat.is_eq(c, 1n+l)), eq, ku, kg, kl)",
        set()),
    "tie-break: deletion before diagonal": (
        "Cell{c, pick(Nat.is_eq(c, dc), ",
        "Cell{c, pick(Bool.and(Nat.is_eq(c, dc), Bool.not(Nat.is_eq(c, uc))), ",
        set()),
    "off by one: an insertion costs 2": (
        "+c = Nat.min(dc, Nat.min(uc, 1n+l))",
        "+c = Nat.min(dc, Nat.min(uc, 2n+l))",
        {"total_is_distance"}),
    "a match counted as a substitution": (
        "    case True{} Counts{m, s, d, i}:\n      Counts{1n+m, s, d, i}",
        "    case True{} Counts{m, s, d, i}:\n      Counts{m, 1n+s, d, i}",
        {"total_is_distance", "identical"}),
}


def levenshtein(a: list, b: list) -> int:
    """The distance alone, by the textbook recurrence (no backtrace)."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(prev + (x != y), row[j] + 1, row[j - 1] + 1)
    return row[-1]


def violated_laws(pairs, counts) -> set[str]:
    """The laws of LAWS.bend that the binary's (S, D, I) break on these pairs (the length laws in their M-free form)."""
    bad = set()
    for (ref, hyp), (s, d, i) in zip(pairs, counts):
        n, m, t = len(ref), len(hyp), s + d + i
        if t != levenshtein(ref, hyp):
            bad.add("total_is_distance")
        if s + d > n or s + i > m or n - s - d != m - s - i:
            bad.add("ref_length/hyp_length")
        if ref == hyp and (s, d, i) != (0, 0, 0):
            bad.add("identical")
        if not abs(n - m) <= t <= max(n, m):
            bad.add("bounds")
    return bad


def mutant(root: Path, old: str, new: str) -> Path:
    for name in ("bend-wer", "bend-planner"):
        shutil.copytree(REPO / "finetune" / name, root / name,
                        ignore=shutil.ignore_patterns("build", "tests", "lean", "__pycache__", "*.py", "*.md"))
    if old:
        src = root / "bend-wer" / "wer.bend"
        text = src.read_text()
        assert text.count(old) == 1, old
        src.write_text(text.replace(old, new))
    return root / "bend-wer"


def bend(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["nix", "develop", str(REPO), "-c", "bend", *args], cwd=cwd, capture_output=True,
                          text=True, timeout=900)


class MutantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(5)
        cls.pairs = (list(EDGES.values()) + exhaustive_pairs() + random_pairs(rng, 400)
                     + [(r, r) for r, _ in random_pairs(rng, 50)])
        cls.expected = [wer.edit_counts(r, h) for r, h in cls.pairs]

    def test_original_passes(self):
        """The unmutated program: proofs check, no law broken, identical to edit_counts."""
        with tempfile.TemporaryDirectory() as d:
            here = mutant(Path(d), "", "")
            self.assertIn("ALL PROOFS CHECK", bend(here, "PROOF.bend").stdout)
            self.assertEqual(bend(here, "scorer.bend", "-o", "bend-wer").returncode, 0)
            counts = self.score(here / "bend-wer")
            self.assertEqual(violated_laws(self.pairs, counts), set())
            self.assertEqual(counts, self.expected)

    def score(self, binary: Path) -> list[tuple[int, int, int]]:
        old = bend_wer.os.environ.get("BEND_WER")
        bend_wer.os.environ["BEND_WER"] = str(binary)
        try:
            return bend_wer.edit_counts_many(self.pairs)
        finally:
            if old is None:
                del bend_wer.os.environ["BEND_WER"]
            else:
                bend_wer.os.environ["BEND_WER"] = old

    def test_mutants_are_rejected(self):
        for name, (old, new, laws) in MUTANTS.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                here = mutant(Path(d), old, new)
                proof = bend(here, "PROOF.bend")
                self.assertIn("SOME PROOFS FAIL", proof.stdout + proof.stderr, "PROOF.bend accepted a mutant")
                build = bend(here, "scorer.bend", "-o", "bend-wer")
                self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
                counts = self.score(here / "bend-wer")
                self.assertNotEqual(counts, self.expected, "the fidelity tests accepted a mutant")
                broken = violated_laws(self.pairs, counts)
                if laws:
                    self.assertLessEqual(laws, broken)
                else:
                    self.assertEqual(broken, set(), "a tie-break should keep every law")


if __name__ == "__main__":
    unittest.main()
