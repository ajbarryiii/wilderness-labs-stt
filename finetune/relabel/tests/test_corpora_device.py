"""Aligner device contracts for the training aligner (corpora/ctc_align.py) and the corpora CLI:

- the model forward used for emissions takes CPU audio, normalizes it on the CPU (so both devices see the
  same input), runs the model on the chosen device and returns CPU float32 log-probabilities;
- `med-osce-segment --device cuda` loads the aligner on the GPU and segments while holding the parakeet GPU
  lock (waiting at most --lock-wait seconds); on the CPU (the default) no lock is taken.

No model is downloaded. The GPU case of the forward runs only with RELABEL_ALIGN_GPU_TEST=1 (under the GPU
lock and a lease: see tests/test_corpora_ctc_align.py for the CPU contracts).

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_corpora_*' -v
"""
from __future__ import annotations

import contextlib
import os
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

import score_teacher
from corpora import __main__ as cli
from corpora import ctc_align, med_osce
from devsets import align as part_a


class Tiny(torch.nn.Module):
    """Frame k's logits: (sample k*HOP, 0, -sample k*HOP); records the input it was given."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, x):
        self.seen = x.detach().cpu().clone()
        frames = part_a.frame_count(x.shape[1])
        v = x[:, :frames * part_a.HOP:part_a.HOP] * self.scale
        return types.SimpleNamespace(logits=torch.stack([v, torch.zeros_like(v), -v], dim=-1))


def wave(n: int = 16000) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return (3.0 + 2.0 * torch.randn(1, n, generator=g)).float()


class ForwardTest(unittest.TestCase):
    def check(self, device: str) -> torch.Tensor:
        model = Tiny().to(device)
        x = wave()
        out = ctc_align.make_forward(model, True, device)(x)
        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.dtype, torch.float32)
        z = (x - x.mean()) / torch.sqrt(x.var() + 1e-7)
        self.assertTrue(torch.equal(model.seen, z))   # normalized on the CPU, identically for every device
        v = z[:, :part_a.frame_count(x.shape[1]) * part_a.HOP:part_a.HOP]
        expect = torch.stack([v, torch.zeros_like(v), -v], dim=-1).log_softmax(-1)
        self.assertTrue(torch.allclose(out, expect, atol=1e-6))
        self.assertTrue(torch.allclose(out.exp().sum(-1), torch.ones(out.shape[:2]), atol=1e-5))
        return out

    def test_cpu(self):
        self.check("cpu")

    def test_without_normalization_the_input_is_unchanged(self):
        model = Tiny()
        x = wave()
        ctc_align.make_forward(model, False, "cpu")(x)
        self.assertTrue(torch.equal(model.seen, x))

    @unittest.skipUnless(os.environ.get("RELABEL_ALIGN_GPU_TEST") == "1" and torch.cuda.is_available(),
                         "GPU forward test runs only with RELABEL_ALIGN_GPU_TEST=1")
    def test_cuda_matches_cpu(self):
        self.assertTrue(torch.allclose(self.check("cuda"), self.check("cpu"), atol=1e-5))


class SegmentCliTest(unittest.TestCase):
    def run_cli(self, argv: list[str]) -> tuple[list, mock.MagicMock]:
        events: list = []

        @contextlib.contextmanager
        def lock(label, wait_s, *a, **k):
            events.append(("lock", float(wait_s)))
            yield
            events.append(("unlock",))

        def segment_all(cdir, aligner, limit=None, batch=None):
            events.append(("segment", batch))
            return {"status": "partial"}

        with mock.patch.object(score_teacher, "gpu_lock", lock), \
                mock.patch.object(ctc_align, "load", return_value=object()) as load, \
                mock.patch.object(med_osce, "segment_all", segment_all), \
                mock.patch.object(cli.corpora, "corpus_dir", return_value=Path("/nonexistent/med-osce")), \
                mock.patch.object(cli.registry, "source", return_value={}), \
                mock.patch.object(cli.common, "require_mount"):
            self.assertEqual(cli.main(argv), 0)
        return events, load

    def test_cuda_segments_under_the_gpu_lock(self):
        events, load = self.run_cli(["med-osce-segment", "--device", "cuda", "--lock-wait", "7", "--batch", "3"])
        self.assertEqual(events, [("lock", 7.0), ("segment", 3), ("unlock",)])
        self.assertEqual(load.call_args.kwargs.get("device"), "cuda")

    def test_cpu_is_the_default_and_takes_no_lock(self):
        events, load = self.run_cli(["med-osce-segment", "--batch", "3"])
        self.assertEqual(events, [("segment", 3)])
        self.assertIn(load.call_args.kwargs.get("device", "cpu"), ("cpu",))


if __name__ == "__main__":
    unittest.main()
