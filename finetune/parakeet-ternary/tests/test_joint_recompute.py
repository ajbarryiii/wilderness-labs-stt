"""--joint-recompute (joint_recompute.py): the transducer joint's hidden activations are
recomputed in the backward pass instead of being kept from the forward.

Contract under test:
- the seam: add_arguments registers --joint-recompute, off by default; configure leaves
  the model untouched with the flag off and returns the provenance record; the train
  module is never modified (its attributes are the same objects after every test);
- loss, joint output and the gradients of every joint parameter and of the joint's
  inputs equal the flag-off ones exactly (torch.equal), in FP32 and under BF16 autocast,
  and through train.compute_losses (NeMo's fused sub-batch loop, the TDT loss, encoder
  grad checkpointing) for the full-model, frozen-joint and decoder-only recipes, and when
  the backward is a torch.autograd.grad taken with autocast off (as the per-sub-batch
  backward tdt_chunked.joint_loss takes it). Exact,
  not within a tolerance: the recompute re-runs the same kernels on the same inputs (same
  shapes, so the same GEMM algorithms) and backward is the same graph, so nothing is
  reassociated; any difference is a bug;
- joint dropout with p > 0 draws the same mask in the recompute (the CUDA RNG is
  preserved), and the step leaves the RNG where the flag-off step does;
- the memory kept from the forward for the backward drops by the joint activations NeMo
  saves today: at p = 0 (every recipe) one copy of the [B, T, U, H] ReLU output in the
  compute dtype (ReLU and the output Linear save the same tensor); at p > 0 also the
  dropout output and its 1-byte mask. The projections f [B, T, H] and g [B, U, H] are
  kept instead;
- refusals: a module without joint_after_projection, a joint already wrapped, a joint
  containing BatchNorm (the recompute would update its running statistics twice).

Not tested here: resume refusal of a flag change, which is the consuming trainer's
(joint_recompute's module docstring). train.py and its tests are byte-frozen (M1
provenance, finetune/parakeet-binary/tests/test_provenance.py).

Run: finetune/parakeet-ternary/python -m unittest discover -s finetune/parakeet-ternary/tests -v
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import types
import unittest
from pathlib import Path

import torch

import joint_recompute
import paths
import train
from test_train import HAS_CUDA, tiny_model

TRAIN_ATTRIBUTES = dict(vars(train))  # identity snapshot: the seam must never modify train

HIDDEN, VOCAB, EXTRA, ENC, PRED = 640, 1024, 5, 64, 32  # the pinned joint's H and outputs


def synthetic_joint(p: float = 0.0):
    """NeMo's RNNTJoint at the pinned joint width; built with the pinned 20% dropout and then
    set to p, as train.disable_pred_joint_dropout does (the Dropout module stays)."""
    from nemo.collections.asr.modules.rnnt import RNNTJoint
    joint = RNNTJoint({"encoder_hidden": ENC, "pred_hidden": PRED, "joint_hidden": HIDDEN,
                       "activation": "relu", "dropout": 0.2}, num_classes=VOCAB,
                      num_extra_outputs=EXTRA)
    dropouts = [m for m in joint.modules() if isinstance(m, torch.nn.Dropout)]
    assert len(dropouts) == 1
    dropouts[0].p = p
    return joint


def pair(p: float = 0.0, device: str = "cuda"):
    """(reference joint, recomputing joint) with identical weights."""
    torch.manual_seed(0)
    ref = synthetic_joint(p).to(device).train()
    rec = copy.deepcopy(ref)
    joint_recompute.enable(rec)
    return ref, rec


def joint_step(joint, enc, dec, weight, autocast: bool, seed: int) -> dict:
    """One forward + backward of joint.joint (as RNNTJoint.forward calls it) with a fixed
    linear loss; everything a caller can observe, on the CPU."""
    torch.manual_seed(seed)
    for p in joint.parameters():
        p.grad = None
    e, d = enc.clone().requires_grad_(True), dec.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = joint.joint(e, d)
    loss = (out.float() * weight).sum()
    loss.backward()
    grads = {"enc": e.grad, "dec": d.grad, **{n: p.grad for n, p in joint.named_parameters()}}
    return {"loss": loss.detach().cpu(), "out": out.detach().cpu(),
            "grads": {k: v.cpu() for k, v in grads.items()},
            "next_draw": torch.rand(8, device="cuda").cpu()}


def synthetic_inputs(b: int, t: int, u: int, seed: int = 1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    enc = torch.randn(b, t, ENC, device="cuda", generator=g)
    dec = torch.randn(b, u, PRED, device="cuda", generator=g)
    weight = torch.randn(b, t, u, VOCAB + 1 + EXTRA, device="cuda", generator=g)
    return enc, dec, weight


class SeamTest(unittest.TestCase):
    def test_add_arguments(self) -> None:
        parser = argparse.ArgumentParser()
        joint_recompute.add_arguments(parser)
        self.assertFalse(parser.parse_args([]).joint_recompute)
        self.assertTrue(parser.parse_args(["--joint-recompute"]).joint_recompute)

    def test_configure(self) -> None:
        student = types.SimpleNamespace(joint=synthetic_joint())
        self.assertEqual(joint_recompute.configure(student, False), {"joint_recompute": False})
        self.assertNotIn("joint_after_projection", vars(student.joint))
        record = joint_recompute.configure(student, True)
        self.assertTrue(joint_recompute.is_enabled(student.joint))
        digest = hashlib.sha256(Path(joint_recompute.__file__).read_bytes()).hexdigest()
        self.assertEqual(record, {"joint_recompute": True, "module_sha256": digest})
        for name in joint_recompute.SOURCES:
            self.assertTrue((paths.HERE.parent / name).samefile(joint_recompute.__file__))
        joint_recompute.disable(student.joint)
        self.assertNotIn("joint_after_projection", vars(student.joint))
        with self.assertRaises(RuntimeError):
            joint_recompute.disable(student.joint)

    def test_refusals(self) -> None:
        with self.assertRaises(TypeError):
            joint_recompute.enable(torch.nn.Linear(2, 2))
        joint = synthetic_joint()
        joint_recompute.enable(joint)
        with self.assertRaises(RuntimeError):
            joint_recompute.enable(joint)
        normed = synthetic_joint()
        normed.joint_net.insert(0, torch.nn.BatchNorm1d(HIDDEN))
        with self.assertRaises(ValueError):
            joint_recompute.enable(normed)
        self.assertFalse(joint_recompute.is_enabled(normed))

    def test_cpu_and_no_grad_unchanged(self) -> None:
        """Outside a CUDA training step (CPU, no_grad: decoding, export) the outputs are the
        flag-off ones."""
        ref, rec = pair(device="cpu")
        enc, dec = torch.randn(2, 5, ENC), torch.randn(2, 3, PRED)
        with torch.no_grad():
            self.assertTrue(torch.equal(ref.joint(enc, dec), rec.joint(enc, dec)))
        ref.eval(), rec.eval()
        with torch.inference_mode():
            self.assertTrue(torch.equal(ref.joint(enc, dec), rec.joint(enc, dec)))


@unittest.skipUnless(HAS_CUDA, "CUDA joint and TDT loss")
class SyntheticJointTest(unittest.TestCase):
    def assert_same(self, a: dict, b: dict, label: str) -> None:
        self.assertTrue(torch.equal(a["loss"], b["loss"]), f"{label}: loss")
        self.assertTrue(torch.equal(a["out"], b["out"]), f"{label}: joint output")
        self.assertEqual(a["grads"].keys(), b["grads"].keys())
        for k in a["grads"]:
            self.assertIsNotNone(a["grads"][k], f"{label}: {k} has no gradient")
            self.assertTrue(torch.equal(a["grads"][k], b["grads"][k]), f"{label}: gradient {k}")
        self.assertTrue(torch.equal(a["next_draw"], b["next_draw"]), f"{label}: RNG after the step")

    def test_exact(self) -> None:
        enc, dec, weight = synthetic_inputs(3, 40, 15)
        for autocast in (False, True):
            ref, rec = pair()
            self.assert_same(joint_step(ref, enc, dec, weight, autocast, seed=5),
                             joint_step(rec, enc, dec, weight, autocast, seed=5),
                             f"autocast={autocast}")

    def test_grad_with_autocast_off(self) -> None:
        """The per-sub-batch backward's pattern (tdt_chunked.joint_loss): forward under BF16
        autocast, then torch.autograd.grad with autocast off; the recompute must replay the
        forward's autocast state."""
        enc, dec, weight = synthetic_inputs(3, 40, 15)
        results = []
        for joint in pair(p=0.3):
            torch.manual_seed(5)
            e, d = enc.clone().requires_grad_(True), dec.clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = joint.joint(e, d)
                loss = (out.float() * weight).sum()
            params = list(joint.parameters())
            with torch.autocast("cuda", enabled=False):
                grads = torch.autograd.grad(loss, [e, d] + params)
            results.append([loss.detach()] + [g.cpu() for g in grads])
        self.assertEqual(results[0][1].dtype, torch.float32)
        for i, (a, b) in enumerate(zip(*results)):
            self.assertTrue(torch.equal(a.cpu(), b.cpu()), f"item {i}")

    def test_dropout_mask_reproduced(self) -> None:
        enc, dec, weight = synthetic_inputs(3, 40, 15)
        for autocast in (False, True):
            ref, rec = pair(p=0.3)
            a = joint_step(ref, enc, dec, weight, autocast, seed=5)
            self.assert_same(a, joint_step(rec, enc, dec, weight, autocast, seed=5),
                             f"p=0.3 autocast={autocast}")
            # Negative control: another mask (seed) changes the gradients, so the equality
            # above does test the mask.
            other = joint_step(ref, enc, dec, weight, autocast, seed=6)
            self.assertFalse(torch.equal(a["grads"]["enc"], other["grads"]["enc"]))

    def held_bytes(self, joint, enc, dec) -> int:
        """CUDA memory kept from the forward for the backward (forward done, before backward)."""
        e, d = enc.clone().requires_grad_(True), dec.clone().requires_grad_(True)
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.bfloat16):  # as train.compute_losses
            out = joint.joint(e, d)
        loss = out.float().sum()  # sum's backward saves no tensor
        del out
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - before
        loss.backward()
        return held

    def test_saved_memory_drops(self) -> None:
        b, t, u = 4, 200, 50
        enc, dec, _ = synthetic_inputs(b, t, u)
        cells, bf16 = b * t * u * HIDDEN, 2
        # Bytes per [B, T, U, H] element saved today: p = 0, the ReLU output (one tensor,
        # saved by ReLU and by the output Linear); p > 0, also the dropout output and mask.
        for p, per_cell in ((0.0, bf16), (0.3, 2 * bf16 + 1)):
            ref, rec = pair(p)
            self.held_bytes(ref, enc, dec)  # the first GEMMs allocate cuBLAS workspace (counted)
            drop = self.held_bytes(ref, enc, dec) - self.held_bytes(rec, enc, dec)
            expected = cells * per_cell
            # What else the two sides keep differently is small and known: the recompute
            # keeps f and g (BF16) and no longer keeps the output Linear's BF16 weight cast;
            # allocator rounding is 512 bytes per tensor. Their sum bounds the deviation.
            slack = bf16 * (b * t + b * u) * HIDDEN + bf16 * HIDDEN * (VOCAB + 1 + EXTRA) + 16 * 512
            self.assertLess(slack, expected / 10)
            self.assertLessEqual(abs(drop - expected), slack,
                                 f"p={p}: held memory dropped by {drop} bytes, expected {expected}")


@unittest.skipUnless(HAS_CUDA, "CUDA joint and TDT loss")
class TrainingPathTest(unittest.TestCase):
    """train.compute_losses (BF16 autocast, NeMo's fused sub-batch loop, the TDT loss) on the
    tiny model of test_train; six rows, so two sub-batches of the model's fused_batch_size 4."""

    @classmethod
    def setUpClass(cls) -> None:
        paths.require_mount()
        train.deterministic_mode()
        train.prepare_tdt_loss()
        torch.manual_seed(321)
        cls.init = copy.deepcopy(tiny_model().state_dict())
        g = torch.Generator().manual_seed(9)
        seconds = [2.5, 2.1, 1.7, 1.4, 1.2, 1.0]
        cls.audio_len = torch.tensor([int(s * 16000) for s in seconds])
        cls.audio = torch.randn(len(seconds), int(cls.audio_len.max()), generator=g) * 0.1
        cls.token_len = torch.tensor([9, 7, 8, 4, 5, 3])
        cls.tokens = torch.randint(1, 1000, (len(seconds), 9), generator=g)

    def student(self, recipe: str, recompute: bool):
        model = tiny_model()
        model.load_state_dict(self.init)
        flags = {"encoder_matching": False, "train_pred_joint": recipe != "frozen-joint",
                 "quantized": False}
        train.configure_trainable(model, flags)
        if recipe == "decoder-only":  # as relabel/train_decoder.configure
            model.encoder.requires_grad_(False)
            model.encoder.eval()
        model.cuda()
        if recipe != "decoder-only":
            train.enable_grad_checkpointing(model.encoder)
        self.assertEqual(joint_recompute.configure(model, recompute)["joint_recompute"], recompute)
        return model, flags

    def step(self, recipe: str, recompute: bool) -> dict:
        model, flags = self.student(recipe, recompute)
        self.assertEqual(model.joint.fused_batch_size, 4)
        inputs = {}

        def keep_inputs(_module, _args, kwargs):
            for name in ("encoder_outputs", "decoder_outputs"):
                if kwargs[name].requires_grad:
                    kwargs[name].retain_grad()
                    inputs[name] = kwargs[name]

        handle = model.joint.register_forward_pre_hook(keep_inputs, with_kwargs=True)
        train.seed_everything(11, model)
        train.reseed_cudnn_rnn_dropout()
        loss, _, _ = train.compute_losses(model, self.audio.cuda(), self.audio_len.cuda(),
                                          self.tokens.cuda(), self.token_len.cuda(), flags)
        loss.backward()
        handle.remove()
        return {"loss": loss.detach().cpu(),
                "grads": {n: p.grad.cpu() for n, p in model.named_parameters() if p.requires_grad},
                "inputs": {n: t.grad.cpu() for n, t in inputs.items()},
                "buffers": {n: b.cpu() for n, b in model.named_buffers()},
                "next_draw": torch.rand(8, device="cuda").cpu()}

    def test_compute_losses_exact(self) -> None:
        for recipe, inputs in (("full", {"encoder_outputs", "decoder_outputs"}),
                               ("frozen-joint", {"encoder_outputs"}),
                               ("decoder-only", {"decoder_outputs"})):
            a, b = self.step(recipe, False), self.step(recipe, True)
            self.assertTrue(torch.isfinite(a["loss"]))
            self.assertTrue(torch.equal(a["loss"], b["loss"]), f"{recipe}: loss")
            self.assertEqual(set(a["inputs"]), inputs, recipe)
            for part in ("grads", "inputs", "buffers"):
                self.assertEqual(a[part].keys(), b[part].keys(), f"{recipe}: {part}")
                for k in a[part]:
                    self.assertTrue(torch.equal(a[part][k], b[part][k]), f"{recipe}: {part} {k}")
            self.assertTrue(any(k.startswith("joint.") for k in a["grads"]) == (recipe != "frozen-joint"))
            self.assertTrue(torch.equal(a["next_draw"], b["next_draw"]), f"{recipe}: RNG")


class ZTrainRestoredTest(unittest.TestCase):
    """Runs last (unittest orders classes by name): train's attributes are the objects it had
    when this module was imported, after every test above."""

    def test_train_module_untouched(self) -> None:
        self.assertEqual(TRAIN_ATTRIBUTES.keys(), vars(train).keys())
        changed = [k for k, v in vars(train).items() if TRAIN_ATTRIBUTES[k] is not v]
        self.assertEqual(changed, [])


if __name__ == "__main__":
    unittest.main()
