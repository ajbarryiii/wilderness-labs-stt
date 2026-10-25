"""Contract tests for the Bend batch planner (DESIGN.md "Acceptance"): the binary must match reference.py byte for
byte, satisfy the laws, and reject malformed input without writing output.

    finetune/relabel/python -m unittest discover -s finetune/bend-planner/tests -v

The binary is build/planner (or $BEND_PLANNER); it is rebuilt through the repo flake when missing or older than a
.bend or .c source. The real-data tests skip when $RELABEL_ROOT does not hold the C1 shards or the stage 2b plan.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import reference as ref  # noqa: E402

REPO = HERE.parents[1]
BUDGET = 600 * ref.SAMPLE_RATE
# $RELABEL_ROOT is machine-specific (finetune/relabel's ./python sets it from its private local.env)
RELABEL_ROOT = Path(os.environ.get("RELABEL_ROOT", "/nonexistent"))
C1_SHARDS = RELABEL_ROOT / "pool-audio/w12-7ff17f270fd6/shards"
STAGE_2B_PLAN = RELABEL_ROOT / "training/batch-plan/w12-7ff17f270fd6-p128-n156"
_PLANNER: Path | None = None


def planner() -> Path:
    global _PLANNER
    if _PLANNER is None:
        if os.environ.get("BEND_PLANNER"):
            _PLANNER = Path(os.environ["BEND_PLANNER"])
        else:
            binary = HERE / "build" / "planner"
            sources = list(HERE.rglob("*.bend")) + list(HERE.rglob("*.c"))
            if not binary.exists() or any(s.stat().st_mtime > binary.stat().st_mtime for s in sources):
                binary.parent.mkdir(exist_ok=True)
                subprocess.run(["nix", "develop", str(REPO), "-c", "bend", "planner.bend", "-o", str(binary)],
                               cwd=HERE, check=True)
            _PLANNER = binary
    return _PLANNER


def run(tmp: Path, samples, pools, budget) -> tuple[subprocess.CompletedProcess, Path, Path]:
    s, p = tmp / "samples.u32", tmp / "pools.u32"
    o, f = tmp / "order.u32", tmp / "offsets.u32"
    for path in (o, f):
        path.unlink(missing_ok=True)
    for path, values in ((s, samples), (p, pools)):
        if isinstance(values, bytes):
            path.write_bytes(values)
        else:
            ref.write_u32(path, values)
    proc = subprocess.run([str(planner()), str(s), str(p), str(budget), str(o), str(f)],
                          capture_output=True, text=True, timeout=600)
    return proc, o, f


def check_laws(test: unittest.TestCase, samples, pools, budget, order, offsets) -> None:
    n = len(samples)
    test.assertEqual(sorted(order.tolist()), list(range(n)), "law 1: order is not a permutation")
    test.assertTrue(offsets[0] == 0 and offsets[-1] == n and np.all(np.diff(offsets.astype(np.int64)) > 0))
    pool_of = np.searchsorted(pools, np.arange(n), side="right") - 1
    batches = [order[lo:hi] for lo, hi in zip(offsets[:-1].tolist(), offsets[1:].tolist())]
    for b, batch in enumerate(batches):
        test.assertEqual(len(set(pool_of[batch].tolist())), 1, "law 1: a batch crosses pools")
        lengths = samples[batch].astype(np.int64)
        test.assertTrue(np.all(np.diff(lengths) >= 0), "law 3: a batch is not sorted")
        test.assertTrue(len(batch) == 1 or len(batch) * int(lengths.max()) <= budget, "law 2: over budget")
        if b + 1 < len(batches) and pool_of[batches[b + 1][0]] == pool_of[batch[0]]:
            nxt = int(samples[batches[b + 1][0]])
            test.assertGreaterEqual(nxt, int(lengths.max()), "law 3: batches in a pool are not in order")
            test.assertGreater((len(batch) + 1) * nxt, budget, "law 3: the cut is not greedy")


def random_case(rng: np.random.Generator, n: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "speech":
        samples = rng.integers(16_000, 30 * 16_000 + 1, n)
    elif kind == "ties":
        samples = rng.choice([16_000, 80_000, 160_000, 480_000], n)
    else:  # "wide": zero-length clips and clips over the budget
        samples = rng.choice([0, 1, 7, 9_599_999, 9_600_000, 9_600_001, 50_000_000], n)
    cuts = np.sort(rng.integers(0, n + 1, rng.integers(0, 6)))
    return samples.astype(np.uint32), np.concatenate([[0], cuts, [n]]).astype(np.uint32)


EDGES = {
    "empty": ([], [0]),
    "one clip": ([16_000], [0, 1]),
    "all equal": ([160_000] * 200, [0, 200]),
    "over budget": ([20_000_000, 1, 2, 20_000_000], [0, 4]),
    "empty pools": ([5, 3, 9], [0, 0, 2, 2, 3, 3]),
}

REJECT = {
    "samples not a multiple of 4": (b"\x01\x00\x00", [0, 0], BUDGET),
    "pools not from 0": ([1, 2, 3], [1, 3], BUDGET),
    "pools not to n": ([1, 2, 3], [0, 2], BUDGET),
    "pools decrease": ([1, 2, 3], [0, 2, 1, 3], BUDGET),
    "pools empty": ([1, 2, 3], b"", BUDGET),
    "budget not decimal": ([1], [0, 1], "6e5"),
    "budget negative": ([1], [0, 1], "-1"),
    "budget non-ASCII digit": ([1], [0, 1], "\u0663"),
    "budget empty": ([1], [0, 1], ""),
    "budget over u32": ([1], [0, 1], str(2**32)),
}


class ContractTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def assert_matches_reference(self, samples, pools, budget):
        proc, o, f = run(self.tmp, samples, pools, budget)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        order, offsets = ref.plan(samples, pools, budget)
        self.assertEqual(o.read_bytes(), order.astype("<u4").tobytes())
        self.assertEqual(f.read_bytes(), offsets.astype("<u4").tobytes())
        check_laws(self, samples, pools, budget, order, offsets)

    def test_edges_match_reference(self):
        for name, (samples, pools) in EDGES.items():
            with self.subTest(name):
                self.assert_matches_reference(np.asarray(samples, dtype=np.uint32),
                                              np.asarray(pools, dtype=np.uint32), BUDGET)

    def test_random_matches_reference(self):
        for k, kind in enumerate(["speech", "ties", "wide"]):
            for seed in range(12):
                with self.subTest(kind=kind, seed=seed):
                    rng = np.random.default_rng([seed, k])
                    samples, pools = random_case(rng, int(rng.integers(1, 3000)), kind)
                    budget = int(rng.choice([BUDGET, 1_000_000, 0, ref.U32_MAX]))
                    self.assert_matches_reference(samples, pools, budget)

    def test_rejects_malformed_input(self):
        for name, (samples, pools, budget) in REJECT.items():
            with self.subTest(name):
                proc, o, f = run(self.tmp, samples, pools, budget)
                self.assertNotEqual(proc.returncode, 0)
                self.assertTrue(proc.stderr.strip())
                self.assertFalse(o.exists() or f.exists())

    def test_c1_journal_128_batch_pools(self):
        journals = sorted(C1_SHARDS.glob("*.journal.parquet"))[:40]
        if len(journals) < 40:
            self.skipTest("C1 shards are not available")
        samples, batch = ref.journal_samples(journals)
        pools = ref.pools_by_batches(batch, 128)
        order, offsets = ref.plan(samples, pools, BUDGET)
        self.assertLess(ref.padding(samples, order, offsets), 0.02)
        proc, o, f = run(self.tmp, samples, pools, BUDGET)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(o.read_bytes(), order.astype("<u4").tobytes())
        self.assertEqual(f.read_bytes(), offsets.astype("<u4").tobytes())

    def test_reproduces_stage_2b_plan(self):
        """The relabel thread's v1 plan (training/batchplan.py): the cut files, checked against plan.json first."""
        if not STAGE_2B_PLAN.is_dir():
            self.skipTest("the stage 2b batch plan is not available")
        meta = json.loads((STAGE_2B_PLAN / "plan.json").read_text())
        for name in ("samples.u32", "pools.u32", "order.u32", "offsets.u32"):
            digest = hashlib.sha256((STAGE_2B_PLAN / name).read_bytes()).hexdigest()
            self.assertEqual(digest, meta["files"][name]["sha256"], f"{name} does not match plan.json")
        o, f = self.tmp / "order.u32", self.tmp / "offsets.u32"
        proc = subprocess.run([str(planner()), str(STAGE_2B_PLAN / "samples.u32"), str(STAGE_2B_PLAN / "pools.u32"),
                               str(meta["budget_samples"]), str(o), str(f)],
                              capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(o.read_bytes(), (STAGE_2B_PLAN / "order.u32").read_bytes())
        self.assertEqual(f.read_bytes(), (STAGE_2B_PLAN / "offsets.u32").read_bytes())

    def test_reproduces_every_plan_on_disk(self):
        """Every v1 or v2 plan under batch-plan/ (v2 indexes entries rather than sequence numbers; same cut)."""
        plans = sorted(p.parent for p in STAGE_2B_PLAN.parent.glob("*/plan.json"))
        if not plans:
            self.skipTest("no batch plans are available")
        for plan_dir in plans:
            with self.subTest(plan_dir.name):
                meta = json.loads((plan_dir / "plan.json").read_text())
                self.assertIn(meta["version"], (1, 2))
                for name in ("samples.u32", "pools.u32", "order.u32", "offsets.u32"):
                    digest = hashlib.sha256((plan_dir / name).read_bytes()).hexdigest()
                    self.assertEqual(digest, meta["files"][name]["sha256"], f"{name} does not match plan.json")
                o, f = self.tmp / "order.u32", self.tmp / "offsets.u32"
                proc = subprocess.run([str(planner()), str(plan_dir / "samples.u32"), str(plan_dir / "pools.u32"),
                                       str(meta["budget_samples"]), str(o), str(f)],
                                      capture_output=True, text=True, timeout=600)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(o.read_bytes(), (plan_dir / "order.u32").read_bytes())
                self.assertEqual(f.read_bytes(), (plan_dir / "offsets.u32").read_bytes())


if __name__ == "__main__":
    unittest.main()
