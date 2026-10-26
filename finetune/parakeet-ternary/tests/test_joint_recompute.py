"""--joint-recompute (joint_recompute.py): the transducer joint's hidden activations are
recomputed in the backward pass instead of being kept from the forward.

Contract under test:
- off by default; with the flag off build_context leaves the joint untouched; a resume
  that would change the setting is refused (a checkpoint written before the flag existed
  counts as off);
- loss, joint output and the gradients of every joint parameter and of the joint's
  inputs equal the flag-off ones exactly (torch.equal), in FP32 and under BF16 autocast,
  and through train.compute_losses (NeMo's fused sub-batch loop, the TDT loss, encoder
  grad checkpointing) for the full-model, frozen-joint and decoder-only recipes. Exact,
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

Run: finetune/parakeet-ternary/python -m unittest discover -s finetune/parakeet-ternary/tests -v
"""
from __future__ import annotations

import copy
import gc
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import joint_recompute
import paths
import train
from test_train import HAS_CUDA, TEST_ROOT, SyntheticStream, make_args, tiny_model

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


class FlagTest(unittest.TestCase):
    def test_default_off(self) -> None:
        self.assertFalse(make_args().joint_recompute)
        argv = ["--arm", "P2", "--lr", "1e-3", "--run-name", "x", "--max-steps", "2",
                "--joint-recompute"]
        self.assertTrue(train.parse_args(argv).joint_recompute)

    def test_resume_refuses_a_change(self) -> None:
        off, on = make_args(), make_args(joint_recompute=True)
        self.assertEqual(train.resume_mismatches(vars(off), off), [])
        self.assertEqual(train.resume_mismatches(vars(on), on), [])
        self.assertTrue(train.resume_mismatches(vars(off), on))
        self.assertTrue(train.resume_mismatches(vars(on), off))
        # A checkpoint written before the flag existed trained without it.
        legacy = {k: v for k, v in vars(off).items() if k != "joint_recompute"}
        self.assertEqual(train.resume_mismatches(legacy, off), [])
        self.assertTrue(train.resume_mismatches(legacy, on))

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
        if recompute:
            joint_recompute.enable(model.joint)
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

    def test_build_context_hook(self) -> None:
        """Flag off: the joint is untouched; on: wrapped. Encoder checkpointing either way."""
        teacher = train.load_teacher(tiny_model())
        for flag in (False, True):
            argv = ["--arm", "A1", "--recipe", "P3", "--train", "pilot", "--lr", "1e-3",
                    "--run-name", "hook", "--max-steps", "3", "--probe-steps", "1",
                    "--stream", "local-librispeech", "--grad-checkpointing", "--no-powerlog"]
            args = train.parse_args(argv + (["--joint-recompute"] if flag else []))
            with mock.patch.object(train, "verify_base", return_value={}), \
                    mock.patch.object(train, "load_pretrained", side_effect=lambda *_: tiny_model()), \
                    mock.patch.object(train, "load_teacher", return_value=teacher), \
                    mock.patch.object(train, "open_stream", return_value=SyntheticStream()):
                TEST_ROOT.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=TEST_ROOT) as tmp:
                    ctx, _ = train.build_context(args, Path(tmp))
            self.assertEqual(joint_recompute.is_enabled(ctx.student.joint), flag)
            self.assertEqual("joint_after_projection" in vars(ctx.student.joint), flag)
            self.assertTrue(all("forward" in vars(layer) for layer in ctx.student.encoder.layers))
            del ctx
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
