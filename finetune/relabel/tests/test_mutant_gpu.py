"""mutant_train contracts that need CUDA (the fused joint+TDT loss and B0's CUDA-graph greedy decoder are CUDA-only):
with the real B0 prediction network, a fresh joint and the real loss seam, a run interrupted and resumed equals an
uninterrupted one; the held-out metric is deterministic in the pinned decoder mode; the loss accepts any frame count
(D-Q's 13 Hz changes only T); frozen-encoder features carry no gradient into training.

Skipped without CUDA (the full suite hides the GPU); run alone on a GPU lease:
    ./python -m unittest discover -s tests -p "test_mutant_gpu.py" -v
"""
from __future__ import annotations

import shutil
import unittest

import torch

import evaluate
import mutant_train as mt
import relabel_paths
from mutant_fixtures import FakeBatches, heldout_items

HAS_CUDA = torch.cuda.is_available()
TMP = relabel_paths.ARTIFACTS / "tmp" / "tests" / "mutant-gpu"


@unittest.skipUnless(HAS_CUDA, "CUDA-only: the fused TDT loss and the CUDA-graph decoder")
class TestCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import train
        train.deterministic_mode()
        shutil.rmtree(TMP, ignore_errors=True)

    def trainer(self, run, stop_after=None):
        model = evaluate.load_pretrained("cpu")
        mt.init_decoder(model)
        sched = mt.schedule("main", 3e-4, n_batches=5)        # 10 steps, evals at 2 (round 2.5), 5, 8 (7.5), 10
        return mt.Trainer(model, FakeBatches(dim=1024, n_batches=5), sched, TMP / run, {"gpu-test": True},
                          torch.device("cuda"), heldout=heldout_items(dim=1024), stop_after=stop_after)

    def test_resume_equals_uninterrupted_with_the_fused_loss(self):
        full = self.trainer("full")
        self.assertEqual(full.run(), "complete")
        part = self.trainer("part", stop_after=4)
        self.assertEqual(part.run(), "stopped")
        again = self.trainer("part")
        self.assertEqual(again.step, 4)
        self.assertEqual(again.run(), "complete")
        a, b = mt.trained_state(full.model), mt.trained_state(again.model)
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        self.assertEqual(full.evals, again.evals)
        self.assertEqual(again.nonfinite, 0)

    def test_heldout_metric_is_deterministic(self):
        t = self.trainer("det")
        t.run_steps(3)
        items = heldout_items(dim=1024)
        a = mt.heldout_metric(t.model, items, torch.device("cuda"))
        b = mt.heldout_metric(t.model, items, torch.device("cuda"))
        self.assertEqual(a, b)

    def test_decoding_after_training_uses_the_current_weights(self):
        """The label-looping decoder caches the start symbol's prediction-network state and projected output when it
        builds its CUDA graphs; a held-out pass after more training must decode as a fresh model with those weights."""
        t = self.trainer("cache")
        items = heldout_items(dim=1024)
        t.run_steps(2)
        mt.heldout_metric(t.model, items, torch.device("cuda"))          # builds the graph state at step-2 weights
        t.run_steps(3)
        got = mt.heldout_metric(t.model, items, torch.device("cuda"))
        fresh = evaluate.load_pretrained("cpu")
        mt.init_decoder(fresh)
        mt.load_trained(fresh, mt.trained_state(t.model))
        fresh.decoder.cuda()
        fresh.joint.cuda()
        want = mt.heldout_metric(fresh, items, torch.device("cuda"))
        self.assertEqual(got, want)
        cached = t.model.decoding.decoding.decoding_computer.state.decoder_output_after_sos
        ref = fresh.decoding.decoding.decoding_computer.state.decoder_output_after_sos
        self.assertTrue(torch.equal(cached, ref))

    def test_loss_at_13_hz_frame_counts(self):
        model = evaluate.load_pretrained("cpu")
        mt.init_decoder(model)
        model.decoder.cuda().train()
        model.joint.cuda().train()
        g = torch.Generator().manual_seed(0)
        for frames in (13, 26, 131):
            feats = torch.randn(2, 1024, frames, generator=g).to(torch.bfloat16).cuda()
            batch = {"feats": feats, "feat_len": torch.tensor([frames, frames - 3]).cuda(),
                     "tokens": torch.randint(0, 1000, (2, 7), generator=g).cuda(),
                     "token_len": torch.tensor([7, 4]).cuda()}
            loss = mt.fused_loss(model, batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNone(feats.grad)
            self.assertTrue(all(p.grad is not None for p in model.joint.parameters()))
            model.zero_grad(set_to_none=True)


if __name__ == "__main__":
    unittest.main()
