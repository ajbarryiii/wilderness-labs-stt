"""Contract test: labeling.batches reproduces train.Batcher's consumed-batch sequence and positions."""
from __future__ import annotations

import copy
import unittest
from unittest import mock

import train  # finetune/parakeet-ternary
from labeling.batches import iter_batches
from labeling_fixtures import FakeStream


def light_collate(items):
    return {"ids": [it["id"] for it in items], "seconds": sum(it["duration"] for it in items)}


def batcher_sequence(stream, batch_seconds, seed, position, n):
    with mock.patch.object(train, "collate", light_collate):
        b = train.Batcher(stream, batch_seconds, seed, position)
        try:
            out = []
            for _ in range(n):
                batch = b.next(timeout=30)
                out.append((batch["ids"], copy.deepcopy(batch["position_after"])))
            return out
        finally:
            b.close()


def ours(stream, batch_seconds, seed, position, n):
    it = iter_batches(stream, batch_seconds, seed, position)
    return [([x["id"] for x in items], copy.deepcopy(after)) for items, after in (next(it) for _ in range(n))]


class SameAsTrainBatcher(unittest.TestCase):
    def test_sequence_and_positions(self):
        a = batcher_sequence(FakeStream(), 30.0, 11, None, 40)
        b = ours(FakeStream(), 30.0, 11, None, 40)
        self.assertEqual([x[0] for x in a], [x[0] for x in b])
        self.assertEqual([x[1] for x in a], [x[1] for x in b])
        self.assertGreater(len({x[1]["pool"] for x in b}), 3)

    def test_resume_from_a_mid_pool_position(self):
        full = ours(FakeStream(), 30.0, 11, None, 40)
        pos = full[12][1]
        self.assertGreater(pos["consumed"], 0)
        a = batcher_sequence(FakeStream(), 30.0, 11, pos, 20)
        b = ours(FakeStream(), 30.0, 11, pos, 20)
        self.assertEqual([x[0] for x in a], [x[0] for x in b])
        self.assertEqual([x[0] for x in b], [x[0] for x in full[13:33]])

    def test_out_of_range_durations_never_batched(self):
        for items, _ in (x for x, _ in zip(iter_batches(FakeStream(), 30.0, 3), range(30))):
            for it in items:
                self.assertTrue(1.0 <= it["duration"] <= 30.0)

    def test_seed_changes_the_order(self):
        a = [x[0] for x in ours(FakeStream(), 30.0, 1, None, 10)]
        b = [x[0] for x in ours(FakeStream(), 30.0, 2, None, 10)]
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
