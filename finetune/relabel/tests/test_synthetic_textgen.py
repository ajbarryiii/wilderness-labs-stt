"""Contract tests: the text generator's GPU memory declaration and placement (synthetic plan "Resources":
text generation is a shared-lease GPU job estimated from a probe).

- The declared VRAM is a probe's measured peak (reserved) + 2 GB, rounded up to 0.1 GB, at the largest probed
  batch whose declaration fits the job's lease; smaller batches are used only when a larger one does not fit.
- The probe measures each batch size separately, smallest first, and records an out-of-memory size instead of
  failing, so the sizes that fit are still measured.
- On the GPU the weights are placed straight on the device; the generator refuses to stage them in host memory.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_synthetic_*' -v
"""
from __future__ import annotations

import sys
import unittest

import torch

from synthetic import textgen
from synthetic_fixtures import roots


def measured(peak_reserved_gb: float, batch: int) -> dict:
    return {"batch": batch, "peak_reserved_gb": peak_reserved_gb, "peak_allocated_gb": peak_reserved_gb - 0.5}


class Declaration(unittest.TestCase):
    def test_the_largest_batch_that_fits_is_declared_at_peak_plus_two(self):
        probes = {16: measured(17.03, 16), 32: measured(18.51, 32), 64: measured(21.04, 64)}
        self.assertEqual(textgen.declaration(probes, 64, lambda gb: gb <= 23.5), (64, 23.1))

    def test_a_smaller_batch_is_used_only_when_the_larger_does_not_fit(self):
        probes = {16: measured(17.03, 16), 32: measured(18.51, 32), 64: measured(21.6, 64)}
        self.assertEqual(textgen.declaration(probes, 64, lambda gb: gb <= 23.5), (32, 20.6))

    def test_out_of_memory_and_larger_than_requested_sizes_are_never_chosen(self):
        probes = {16: measured(17.0, 16), 32: measured(18.0, 32), 64: {"batch": 64, "oom": True},
                  128: measured(19.0, 128)}
        self.assertEqual(textgen.declaration(probes, 64, lambda gb: True), (32, 20.0))
        self.assertEqual(textgen.declaration(probes, 16, lambda gb: True), (16, 19.0))

    def test_nothing_fits_or_nothing_probed_is_refused(self):
        with self.assertRaises(SystemExit):
            textgen.declaration({16: measured(22.0, 16)}, 64, lambda gb: gb <= 23.5)
        with self.assertRaises(SystemExit):
            textgen.declaration({}, 64, lambda gb: True)


class FakeTok:
    def __call__(self, text):
        return {"input_ids": text.split()}


class FakeGenerator:
    """Records every phase; generating a batch of `oom_at` or more prompts raises CUDA out of memory."""

    def __init__(self, batch: int, oom_at: int | None = None):
        self.batch, self.oom_at, self.declared_vram_gb = batch, oom_at, 23.0
        self.tok, self.calls, self.current = FakeTok(), [], 0.0

    def load(self):
        self.calls.append(("load",))

    def reset_peak(self):
        self.calls.append(("reset", self.batch))
        self.current = 16.0

    def peak(self):
        return self.current - 0.2, self.current

    def generate(self, users, seed):
        self.calls.append(("generate", self.batch, len(users)))
        if self.oom_at is not None and len(users) >= self.oom_at:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
        self.current = max(self.current, 16.0 + len(users) * 0.07)
        return [("a sentence of words", False) for _ in users]


def prompts(n: int) -> list[dict]:
    return [{"id": f"probe-{i:06d}", "index": i, "domain": "chemistry", "register": "explanatory",
             "requested": ["mole fraction"] * (1 + i % 3)} for i in range(n)]


class Probe(unittest.TestCase):
    def test_sizes_are_the_halvings_of_the_batch_smallest_first(self):
        self.assertEqual(textgen.probe_sizes(64), [16, 32, 64])
        self.assertEqual(textgen.probe_sizes(64, min_batch=32), [32, 64])
        self.assertEqual(textgen.probe_sizes(16), [16])

    def test_each_size_is_measured_separately_with_full_batches(self):
        gen = FakeGenerator(64)
        res = textgen.probe(gen, prompts(64 * 3), batches=2, sizes=[16, 32, 64])
        self.assertEqual(sorted(res), [16, 32, 64])
        self.assertEqual(gen.calls.count(("load",)), 1)
        for b in (16, 32, 64):
            self.assertEqual([c for c in gen.calls if c[0] == "generate" and c[1] == b], [("generate", b, b)] * 2)
            self.assertAlmostEqual(res[b]["peak_reserved_gb"], 16.0 + b * 0.07)
            self.assertEqual(res[b]["batch"], b)
            self.assertFalse(res[b].get("oom", False))
        order = [c[1] for c in gen.calls if c[0] == "reset"]
        self.assertEqual(order, [16, 32, 64])

    def test_an_out_of_memory_size_is_recorded_and_larger_sizes_are_skipped(self):
        gen = FakeGenerator(64, oom_at=32)
        res = textgen.probe(gen, prompts(64 * 3), batches=2, sizes=[16, 32, 64])
        self.assertFalse(res[16].get("oom", False))
        self.assertTrue(res[32]["oom"])
        self.assertTrue(res[64]["oom"])
        self.assertFalse(any(c[0] == "generate" and c[1] == 64 for c in gen.calls))
        self.assertEqual(textgen.declaration(res, 64, lambda gb: True)[0], 16)


class Placement(unittest.TestCase):
    def test_on_the_gpu_weights_go_straight_to_the_device(self):
        self.assertEqual(textgen.placement("cuda", lambda: True), {"device_map": "cuda"})

    def test_on_the_gpu_without_accelerate_loading_is_refused(self):
        with self.assertRaises(SystemExit):
            textgen.placement("cuda", lambda: False)

    def test_cpu_loads_in_place(self):
        self.assertEqual(textgen.placement("cpu", lambda: False), {})

    def test_the_generator_packages_dir_is_used_when_accelerate_is_not_otherwise_importable(self):
        with roots():
            from synthetic import layout
            d = layout.deps_dir(layout.TEXTGEN)
            (d / "fakeaccel_pkg").mkdir(parents=True)
            (d / "fakeaccel_pkg" / "__init__.py").write_text("")
            saved = list(sys.path)
            try:
                self.assertTrue(textgen.ensure_package("fakeaccel_pkg"))
                self.assertIn(str(d), sys.path)
                self.assertEqual(sys.path[-1], str(d))
                self.assertFalse(textgen.ensure_package("no_such_package_xyz"))
            finally:
                sys.path[:] = saved
                sys.modules.pop("fakeaccel_pkg", None)


# Runs the real CLI commands in a fresh interpreter (transformers caches whether accelerate is importable the first
# time it is asked, and the commands import evaluation code that imports transformers). A stand-in accelerate and
# psutil (with package metadata) live only in the temporary generator package dir; the GPU parts are replaced.
CLI_SCRIPT = r"""
import json, sys
from synthetic import __main__ as cli, layout, textgen, vocab
layout.require_mount = lambda: None
vocab.Split.load = staticmethod(lambda: None)
textgen.make_prompts = lambda *a, **k: []
cli.gpu_lock = lambda *a, **k: cli._NoLock()
cli.write_json = lambda *a, **k: None
seen = {}
def check(*a, **k):
    from transformers.utils import import_utils
    seen["transformers_sees_accelerate"] = import_utils.is_accelerate_available()
    seen["score_teacher_imported"] = "score_teacher" in sys.modules
    return {16: {"batch": 16, "peak_reserved_gb": 1.0}}
textgen.probe = check
textgen.generate_chunk = lambda *a, **k: (check(), {"blocks_done": 0, "blocks_remaining": 0})[1]
cli.fits_lease = lambda gb: True
cli.load_probes = lambda: {16: {"batch": 16, "peak_reserved_gb": 1.0}}
cli.run_dir = lambda args: layout.relabel_root()
rc = cli.main(sys.argv[1:])
print("RESULT " + json.dumps(seen | {"rc": rc}))
"""


def fake_package(root, name: str, version: str) -> None:
    (root / name).mkdir(parents=True)
    (root / name / "__init__.py").write_text(f"__version__ = {version!r}\n")
    info = root / f"{name}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


class ImportOrder(unittest.TestCase):
    def run_cli(self, *argv):
        import os
        import subprocess
        from pathlib import Path
        from synthetic import layout
        here = Path(__file__).resolve().parents[1]
        with roots():
            d = layout.deps_dir(layout.TEXTGEN)
            fake_package(d, "accelerate", "9.0.0")
            fake_package(d, "psutil", "9.0.0")
            env = {k: v for k, v in os.environ.items() if k != "RELABEL_DEPS"}
            out = subprocess.run([str(here / "python"), "-c", CLI_SCRIPT, *argv], cwd=here, env=env,
                                 capture_output=True, text=True, timeout=300)
        lines = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
        self.assertTrue(lines, out.stdout[-2000:] + out.stderr[-3000:])
        import json
        return json.loads(lines[-1][len("RESULT "):])

    def test_the_probe_command_makes_accelerate_visible_to_transformers(self):
        res = self.run_cli("textgen-probe", "--batch", "16", "--declared-vram-gb", "1")
        self.assertTrue(res["score_teacher_imported"])
        self.assertTrue(res["transformers_sees_accelerate"])

    def test_the_textgen_command_makes_accelerate_visible_to_transformers(self):
        res = self.run_cli("textgen", "--smoke", "x", "--declared-vram-gb", "auto", "--batch", "16")
        self.assertTrue(res["score_teacher_imported"])
        self.assertTrue(res["transformers_sees_accelerate"])
        self.assertEqual(res["rc"], 0)


class RowProvenance(unittest.TestCase):
    def test_every_row_records_the_generator_batch(self):
        import tempfile
        from pathlib import Path

        class Gen(FakeGenerator):
            def info(self):
                return {"alias": "syn-textgen", "revision": "r", "batch": self.batch}

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            textgen.write_jsonl(d / "prompts.jsonl", prompts(40))
            res = textgen.generate_chunk(d, Gen(16), 7, 60.0, log=lambda *a: None)
            self.assertEqual(res["blocks_remaining"], 0)
            rows = textgen.read_jsonl(d / "text" / f"{0:07d}.jsonl")
            self.assertEqual(len(rows), 40)
            self.assertTrue(all(r["generator"] == {"alias": "syn-textgen", "revision": "r", "batch": 16}
                                for r in rows))


if __name__ == "__main__":
    unittest.main()
