"""Alignment contracts (diagnostic plan Step 2 "Check"): the backtrace reproduces edit_counts's S, D, I
on random word strings and on every recorded B0, M1 and MB1 record; events are multisets.

Run from finetune/relabel:  ./python -m unittest discover -s tests -v   (CPU only)
"""
from __future__ import annotations

import os
import random
import unittest
from collections import Counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import evaluate  # noqa: E402
import relabel_paths  # noqa: E402
from relabel_align import align, fixed_by, only_in, shared, size  # noqa: E402


def rebuild(ref: list[str], hyp: list[str], a) -> bool:
    """Whether hyp is ref edited by the alignment: per gap its inserted words (as a multiset), then per
    ref position its match, substitution word or deletion, consuming hyp left to right."""
    subs = {k: w for (op, k, w) in a.events if op == "sub"}
    dels = {k for (op, k, _) in a.events if op == "del"}
    gaps = {}
    for (op, g, w), n in a.events.items():
        if op == "ins":
            gaps.setdefault(g, Counter())[w] += n
    p = 0
    for k in range(len(ref) + 1):
        want = gaps.get(k, Counter())
        n = sum(want.values())
        if Counter(hyp[p:p + n]) != want:
            return False
        p += n
        if k == len(ref):
            break
        if k in a.matched:
            if p >= len(hyp) or hyp[p] != ref[k]:
                return False
            p += 1
        elif k in subs:
            if p >= len(hyp) or hyp[p] != subs[k] or subs[k] == ref[k]:
                return False
            p += 1
        elif k not in dels:
            return False
    return p == len(hyp)


class RandomStringsTest(unittest.TestCase):
    def test_sdi_equals_edit_counts(self) -> None:
        rng = random.Random(20261021)
        for trial in range(5000):
            vocab = ["a", "b", "c", "d"][: rng.randint(1, 4)]
            ref = [rng.choice(vocab) for _ in range(rng.randint(0, 9))]
            hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 9))]
            a = align(ref, hyp)
            with self.subTest(ref=ref, hyp=hyp):
                self.assertEqual((a.S, a.D, a.I), evaluate.edit_counts(ref, hyp))
                self.assertEqual(size(a.events), a.S + a.D + a.I)

    def test_alignment_is_a_valid_edit_script(self) -> None:
        rng = random.Random(7)
        for trial in range(2000):
            ref = [rng.choice("abc") for _ in range(rng.randint(0, 8))]
            hyp = [rng.choice("abc") for _ in range(rng.randint(0, 8))]
            a = align(ref, hyp)
            with self.subTest(ref=ref, hyp=hyp):
                self.assertTrue(rebuild(ref, hyp, a))
                self.assertTrue(all(0 <= g <= len(ref) for g in a.ins_gaps))


class EventSemanticsTest(unittest.TestCase):
    def test_substitution_deletion_insertion_events(self) -> None:
        a = align("a b c d".split(), "a x c".split())
        self.assertEqual(a.events, Counter({("sub", 1, "x"): 1, ("del", 3, ""): 1}))
        self.assertEqual(a.matched, frozenset({0, 2}))
        b = align("a b".split(), "a u u b".split())
        self.assertEqual(b.events, Counter({("ins", 1, "u"): 2}))  # repeated insertions count separately
        self.assertEqual(b.ins_gaps, frozenset({1}))

    def test_multiset_intersection_and_difference(self) -> None:
        b0 = align("a b".split(), "a u u b".split()).events        # u, u in gap 1
        m1 = align("a b".split(), "a u b".split()).events          # u in gap 1
        mb1 = align("a b".split(), "a u u u b".split()).events     # u, u, u in gap 1
        self.assertEqual(size(shared(m1, b0)), 1)
        self.assertEqual(size(shared(mb1, b0)), 2)
        self.assertEqual(size(only_in(mb1, b0)), 1)
        self.assertEqual(size(only_in(b0, mb1)), 0)
        self.assertEqual(size(only_in(m1, b0)), 0)

    def test_fixed_by(self) -> None:
        teacher = align("a b c".split(), "a b q c".split())  # matches 0, 1, 2; inserts q in gap 2
        self.assertTrue(fixed_by(("sub", 1, "x"), teacher))
        self.assertTrue(fixed_by(("del", 2, ""), teacher))
        self.assertTrue(fixed_by(("ins", 1, "u"), teacher))
        self.assertFalse(fixed_by(("ins", 2, "u"), teacher))  # the teacher inserts something in gap 2
        wrong = align("a b c".split(), "a y c".split())
        self.assertFalse(fixed_by(("sub", 1, "x"), wrong))     # another substitution is not a fix


@unittest.skipUnless(all(d.exists() for d in relabel_paths.STUDENT_RECORDS.values()), "real B0/M1/MB1 records not found under PARAKEET_TERNARY_ROOT / PARAKEET_BINARY_ROOT (private/local.env)")
class RecordedModelsTest(unittest.TestCase):
    def test_every_recorded_b0_m1_mb1_record_is_reproduced(self) -> None:
        import analyze

        total = 0
        for model in analyze.RECORDED:
            records, _ = analyze.load_model(relabel_paths.STUDENT_RECORDS[model], relabel_paths.SETS)
            for name, rows in records.items():
                for r in rows:
                    a = align(r["ref_norm"].split(), r["hyp_norm"].split())
                    self.assertEqual((a.S, a.D, a.I), (r["S"], r["D"], r["I"]), f"{model} {name} {r['id']}")
                    total += 1
        self.assertEqual(total, 3 * (2703 + 2864 + 13098 + 1753 + 2000))


if __name__ == "__main__":
    unittest.main()
