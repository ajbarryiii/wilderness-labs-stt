"""Contract tests for tdt_chunked (train.py --tdt-chunked-backward and --fused-batch-size).

The reference is NeMo's fused joint (`student.joint(..., compute_wer=False)`) at the same
fused_batch_size. Observable contract, flag on:

1. the scalar loss equals the reference's for the model's reduction (NeMo's `loss.reduce`;
   the pinned config's is mean_volume); other reductions are refused;
2. the gradients w.r.t. the encoder output, the prediction-network output and every joint
   parameter equal the reference's (whichever of them require gradients);
3. train.compute_losses / train.train_step work unchanged (one loss.backward());
4. peak CUDA memory stays flat as the number of sub-batches grows (the reference's grows);
5. the TDT loss's omega draw: one random.uniform per sub-batch loss, in order, so the same
   RNN-T/TDT branches are chosen as in the reference;
6. an exception mid-loop leaves the loss reduction at its configured value;
7. double backward and compute_wer=True are refused.
Flag off, the trainer's path is NeMo's fused joint, untouched.

Equality vs tolerance. Both paths run the same NeMo calls on the same sub-batches, so the
loss and every gradient upstream of the joint's parameters (w.r.t. the encoder and
prediction-network outputs and, through them, every encoder and prediction-network
parameter) are the same deterministic computation: asserted bitwise equal (measured equal
at every sub-batch size tried, fp32 and bf16). The joint's parameter gradients are summed
over sub-batches in another order (and, under autocast, in FP32 where NeMo's single
backward sums the bf16 gradients of autocast's cached weight casts), so they are compared
with tolerances derived from the reference's own variation across sub-batch sizes
(mathematically equivalent with omega = 0: every utterance's loss is computed alone; only
GEMM shapes and that accumulation change). NegativeControlTest shows the checks catch a 1%
loss-scale error, a 1% gradient-scale error and a dropped sub-batch.

GPU tests use the tiny model of test_train (the pinned config: vocabulary 1024 + blank, 5
durations, TDT loss with sigma 0.02, omega 0.1, reduction mean_volume, joint 64/32 -> 32).
RealModelTest (bf16 autocast on the pinned Parakeet-TDT-0.6B-v2's joint, real dev-clean
encoder and prediction-network outputs) runs only with TDT_CHUNKED_REAL_MODEL=1.
"""
from __future__ import annotations

import math
import os
import random
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

import paths
import tdt_chunked
import train
from test_train import HAS_CUDA, WORDS, make_args, tiny_model

# Tolerances for the joint's parameter gradients: worst over the joint's tensors of the
# relative L2 difference, 1 - cosine, and max |diff| / max |reference| (per-element, scaled
# by the tensor's magnitude). Each is >= 5x the reference's own floor, measured as the
# reference at fused_batch_size 4 vs 1, 2, 3, 8, 16 (test_reference_noise_floor re-checks it):
# fp32, tiny joint (16 x 96 frames x 24 tokens): floor rel L2 5.9e-6, 1 - cos 1.5e-11,
# per-element 5.9e-6; the chunked path measured 1.0e-7, 4.9e-15, 1.8e-7.
TOL_FP32 = {"rel_l2": 4e-5, "one_minus_cos": 2e-10, "elem": 4e-5}
# bf16 autocast, tiny joint: floor 9.7e-3, 4.7e-5, 1.5e-2 (fused_batch_size 1, the most bf16
# accumulations); the chunked path measured <= 5.0e-3, 1.2e-5, 9.0e-3. Too loose to see a 1%
# gradient scale on the joint alone; the bitwise checks of the loss and input gradients do.
TOL_BF16 = {"rel_l2": 5e-2, "one_minus_cos": 3e-4, "elem": 8e-2}
# bf16 autocast, pinned Parakeet joint on 12 real dev-clean outputs: floor (fused_batch_size
# 4 vs 1, 2, 3, 6, 12) 3.6e-2, 6.6e-4, 5.9e-2 (the reference's loss itself moves 1.3e-3
# relative across sizes: bf16 GEMMs of other trimmed shapes); the chunked path measured
# <= 4.0e-3, 8.1e-6, 6.0e-3.
TOL_REAL = {"rel_l2": 0.2, "one_minus_cos": 3.5e-3, "elem": 0.3}
# Whole compute_losses on the tiny model (bf16; encoder with grad checkpointing), the joint's
# parameter gradients: floor (fused_batch_size 2 vs 3 and 7) 4.2e-3, 6.1e-6, 7.8e-3; the
# chunked path measured <= 2.5e-3, 3.1e-6, 5.2e-3.
TOL_PIPELINE = {"rel_l2": 2.5e-2, "one_minus_cos": 4e-5, "elem": 4e-2}

ENC_DIM, DEC_DIM, VOCAB = 64, 32, 1024


def is_joint(name: str) -> bool:
    return name.startswith("joint.")


def compare(ref: tuple, got: tuple) -> dict:
    """nonfinite: gradients with a NaN or inf on either side; loss_equal; unequal: the non-joint
    gradients that are not bitwise equal; and over the joint's parameter gradients the worst
    relative L2, 1 - cosine and max |diff| / max |ref|."""
    (loss_a, grads_a), (loss_b, grads_b) = ref, got
    if grads_a.keys() != grads_b.keys():
        raise AssertionError(f"gradient sets differ: {sorted(grads_a)} vs {sorted(grads_b)}")
    out = {"nonfinite": sorted(k for k in grads_a if not (torch.isfinite(grads_a[k]).all()
                                                          and torch.isfinite(grads_b[k]).all())),
           "loss_equal": torch.equal(loss_a, loss_b),
           "loss_rel": abs(float(loss_b) - float(loss_a)) / abs(float(loss_a)),
           "unequal": sorted(k for k in grads_a if not is_joint(k)
                             and not torch.equal(grads_a[k], grads_b[k])),
           "rel_l2": 0.0, "one_minus_cos": 0.0, "elem": 0.0}
    for name in filter(is_joint, grads_a):
        a, b = grads_a[name].double(), grads_b[name].double()
        na = a.norm().item()
        if na == 0.0:
            raise AssertionError(f"reference gradient {name} is zero; the test batch is degenerate")
        out["rel_l2"] = max(out["rel_l2"], (a - b).norm().item() / na)
        cos = (a * b).sum().item() / (na * b.norm().item() + 1e-300)
        out["one_minus_cos"] = max(out["one_minus_cos"], 1.0 - cos)
        out["elem"] = max(out["elem"], (a - b).abs().max().item() / a.abs().max().item())
    return out


def identical(ref: tuple, got: tuple) -> bool:
    return (torch.equal(ref[0], got[0]) and ref[1].keys() == got[1].keys()
            and all(torch.equal(ref[1][k], got[1][k]) for k in ref[1]))


def within(ref: tuple, got: tuple, tol: dict) -> tuple[bool, dict]:
    found = compare(ref, got)
    ok = (not found["nonfinite"] and found["loss_equal"] and not found["unequal"]
          and all(found[k] <= tol[k] for k in tol))
    return ok, found


def synthetic(rows: int, frames: int, tokens: int, seed: int, equal: bool = False) -> dict:
    """Random encoder [B, 64, T] and prediction-network [B, 32, U+1] outputs, with varied
    lengths (so every sub-batch is trimmed to its own maxima) unless `equal`."""
    g = torch.Generator().manual_seed(seed)
    enc = torch.randn(rows, ENC_DIM, frames, generator=g)
    dec = torch.randn(rows, DEC_DIM, tokens + 1, generator=g)
    if equal:
        enc_len = torch.full((rows,), frames)
        tr_len = torch.full((rows,), tokens)
    else:
        enc_len = torch.randint(frames // 2, frames + 1, (rows,), generator=g)
        tr_len = torch.randint(1, tokens + 1, (rows,), generator=g)
        enc_len[0], tr_len[-1] = frames, tokens
    transcripts = torch.randint(0, VOCAB, (rows, tokens), generator=g)
    return {k: v.cuda() for k, v in dict(enc=enc, dec=dec, enc_len=enc_len, tr_len=tr_len,
                                          transcripts=transcripts).items()}


def loss_of(path: str, joint, enc, dec, batch: dict, **kw) -> torch.Tensor:
    if path == "fused":
        loss, _, _, _ = joint(encoder_outputs=enc, decoder_outputs=dec,
                              encoder_lengths=batch["enc_len"], transcripts=batch["transcripts"],
                              transcript_lengths=batch["tr_len"], compute_wer=False)
        return loss
    return tdt_chunked.joint_loss(joint, enc, dec, batch["enc_len"], batch["transcripts"],
                                  batch["tr_len"], **kw)


def run(path: str, joint, batch: dict, fbs: int, omega: float = 0.0, seed: int = 0,
        bf16: bool = False, enc_grad: bool = True, dec_grad: bool = True) -> tuple:
    """(loss, {name: gradient}) of one forward+backward: path 'fused' (NeMo) or 'chunked'."""
    joint.set_fused_batch_size(fbs)
    joint.loss._loss.omega = omega
    enc = batch["enc"].clone().requires_grad_(enc_grad)
    dec = batch["dec"].clone().requires_grad_(dec_grad)
    for p in joint.parameters():
        p.grad = None
    random.seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        loss = loss_of(path, joint, enc, dec, batch)
    loss.backward()
    grads = {"enc": enc.grad, "dec": dec.grad}
    grads.update({f"joint.{n}": p.grad for n, p in joint.named_parameters()})
    out = {k: v.detach().clone() for k, v in grads.items() if v is not None}
    for p in joint.parameters():
        p.grad = None
    return loss.detach().clone(), out


def peak_bytes(path: str, joint, batch: dict, fbs: int) -> int:
    """Peak CUDA memory allocated above the inputs during one forward+backward."""
    joint.set_fused_batch_size(fbs)
    joint.loss._loss.omega = 0.0
    enc = batch["enc"].clone().requires_grad_(True)
    dec = batch["dec"].clone().requires_grad_(True)
    for p in joint.parameters():
        p.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    loss = loss_of(path, joint, enc, dec, batch)
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del loss, enc, dec
    for p in joint.parameters():
        p.grad = None
    return peak


class TinyJoint:
    """The tiny model's joint and loss on the GPU, pred/joint dropout 0, training mode."""

    @classmethod
    def setUpClass(cls) -> None:
        paths.require_mount()
        train.deterministic_mode()
        train.prepare_tdt_loss()
        torch.manual_seed(0)
        cls.model = tiny_model()
        train.disable_pred_joint_dropout(cls.model)
        cls.joint = cls.model.joint.cuda().train()
        cls.configured = cls.joint.loss.reduction
        cls.default_fbs = cls.joint.fused_batch_size

    def setUp(self) -> None:
        self.joint.loss.reduction = self.configured
        self.joint.set_fused_batch_size(self.default_fbs)
        self.joint.requires_grad_(True)

    def assertWithin(self, ref: tuple, got: tuple, tol: dict, label: str = "") -> None:
        ok, found = within(ref, got, tol)
        self.assertTrue(ok, f"{label}: {found} exceeds {tol}")


# ----------------------------------------------------------------------------- CPU

class ArgsTest(unittest.TestCase):
    def test_defaults_are_off_and_validated(self) -> None:
        args = make_args()
        self.assertFalse(args.tdt_chunked_backward)
        self.assertIsNone(args.fused_batch_size)
        on = train.parse_args(
            ["--arm", "P2", "--lr", "1e-3", "--run-name", "t", "--max-steps", "6",
             "--tdt-chunked-backward", "--fused-batch-size", "16"])
        self.assertTrue(on.tdt_chunked_backward)
        self.assertEqual(on.fused_batch_size, 16)
        for bad in ("0", "-2"):
            with self.assertRaises(SystemExit):
                train.parse_args(["--arm", "P2", "--lr", "1e-3", "--run-name", "t",
                                  "--max-steps", "6", "--fused-batch-size", bad])

    def test_resume_refuses_a_change_and_accepts_older_checkpoints(self) -> None:
        args = make_args()
        for change in ({"tdt_chunked_backward": True}, {"fused_batch_size": 8}):
            self.assertTrue(train.resume_mismatches(vars(args), make_args(**change)), change)
        older = {k: v for k, v in vars(args).items()
                 if k not in ("tdt_chunked_backward", "fused_batch_size")}
        self.assertEqual(train.resume_mismatches(older, args), [])
        self.assertTrue(train.resume_mismatches(older, make_args(tdt_chunked_backward=True)))
        self.assertTrue(train.resume_mismatches(older, make_args(fused_batch_size=8)))

    def test_legacy_checkpoint_without_args_refuses_the_new_settings(self) -> None:
        """A checkpoint with only a resume_key (no recorded args) was written with both off."""
        import shutil
        import tempfile
        from test_train import TEST_ROOT
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(dir=TEST_ROOT))
        try:
            args = make_args()
            train.save_checkpoint(run, {"step": 3, "resume_key": train.resume_key(args)})
            self.assertEqual(train.load_resume_payload(run, args)["step"], 3)
            for change in ({"tdt_chunked_backward": True}, {"fused_batch_size": 16}):
                with self.assertRaises(SystemExit) as stop:
                    train.load_resume_payload(run, make_args(**change))
                self.assertEqual(stop.exception.code, train.EXIT_RESUME_MISMATCH, change)
        finally:
            shutil.rmtree(run, ignore_errors=True)


class SourcesTest(unittest.TestCase):
    def test_module_is_a_training_source(self) -> None:
        """Its hash is recorded and checked with the other training sources (sweep.py mirrors
        the list; test_train checks the two are equal)."""
        self.assertIn("parakeet-ternary/tdt_chunked.py", train.TRAINING_SOURCES)
        self.assertIn("parakeet-ternary/tdt_chunked.py", train.training_source_hashes())


class ConfigureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        paths.require_mount()
        cls.model = tiny_model()

    def test_default_leaves_the_model_untouched(self) -> None:
        joint = self.model.joint
        before = joint.fused_batch_size
        record = tdt_chunked.configure(self.model, None, False)
        self.assertEqual(joint.fused_batch_size, before)
        self.assertFalse(tdt_chunked.enabled(joint))
        self.assertEqual(record, {"chunked_backward": False, "fused_batch_size": before,
                                  "fused_batch_size_default": before,
                                  "reduction": joint.loss.reduction})

    def test_sets_size_and_flag_and_refuses_unsupported(self) -> None:
        model = tiny_model()
        record = tdt_chunked.configure(model, 8, True)
        self.assertEqual(model.joint.fused_batch_size, 8)
        self.assertTrue(tdt_chunked.enabled(model.joint))
        self.assertTrue(record["chunked_backward"])
        self.assertEqual(len(record["module_sha256"]), 64)
        for bad in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                tdt_chunked.configure(tiny_model(), bad, False)
        model = tiny_model()
        model.joint.loss.reduction = None
        with self.assertRaises(ValueError):
            tdt_chunked.configure(model, None, True)


# ----------------------------------------------------------------------------- GPU

@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class EquivalenceTest(TinyJoint, unittest.TestCase):
    def test_reference_noise_floor(self) -> None:
        """The joint tolerances are >= 5x the reference's own variation across sub-batch sizes."""
        for tol, bf16 in ((TOL_FP32, False), (TOL_BF16, True)):
            batch = synthetic(16, 96, 24, seed=1)
            ref = run("fused", self.joint, batch, 4, bf16=bf16)
            for fbs in (1, 2, 3, 8, 16):
                found = compare(ref, run("fused", self.joint, batch, fbs, bf16=bf16))
                for k in tol:
                    self.assertLessEqual(5 * found[k], tol[k], f"bf16={bf16} fbs={fbs} {k}: {found}")

    def test_fp32_loss_and_gradients_match_the_fused_joint(self) -> None:
        batch = synthetic(14, 96, 24, seed=2)
        for fbs in (1, 3, 4, 8, 14, 64):
            ref = run("fused", self.joint, batch, fbs)
            got = run("chunked", self.joint, batch, fbs)
            self.assertEqual(got[0].dim(), 0)
            self.assertEqual(got[1].keys(), ref[1].keys())
            self.assertWithin(ref, got, TOL_FP32, f"fbs={fbs}")

    def test_bf16_autocast_matches_the_fused_joint(self) -> None:
        batch = synthetic(14, 96, 24, seed=3)
        for fbs in (2, 4, 8):
            self.assertWithin(run("fused", self.joint, batch, fbs, bf16=True),
                              run("chunked", self.joint, batch, fbs, bf16=True), TOL_BF16,
                              f"fbs={fbs}")

    def test_every_supported_reduction(self) -> None:
        batch = synthetic(10, 64, 16, seed=4)
        for reduction in tdt_chunked.SUPPORTED_REDUCTIONS:
            self.joint.loss.reduction = reduction
            self.assertWithin(run("fused", self.joint, batch, 3), run("chunked", self.joint, batch, 3),
                              TOL_FP32, reduction)
        self.joint.loss.reduction = self.configured

    def test_only_the_inputs_that_require_gradients(self) -> None:
        """Encoder-only (frozen joint and prediction network, P1/P2), decoder-only (frozen
        encoder), and no gradient at all (the reference's value, no graph)."""
        batch = synthetic(9, 64, 16, seed=5)
        self.joint.requires_grad_(False)
        ref = run("fused", self.joint, batch, 4, dec_grad=False)
        got = run("chunked", self.joint, batch, 4, dec_grad=False)
        self.assertEqual(sorted(got[1]), ["enc"])
        self.assertWithin(ref, got, TOL_FP32, "encoder only")
        self.joint.requires_grad_(True)
        ref = run("fused", self.joint, batch, 4, enc_grad=False)
        got = run("chunked", self.joint, batch, 4, enc_grad=False)
        self.assertNotIn("enc", got[1])
        self.assertIn("dec", got[1])
        self.assertWithin(ref, got, TOL_FP32, "decoder only")
        with torch.no_grad():
            value = loss_of("chunked", self.joint, batch["enc"], batch["dec"], batch)
            fused = loss_of("fused", self.joint, batch["enc"], batch["dec"], batch)
        self.assertFalse(value.requires_grad)
        self.assertTrue(torch.equal(value, fused))

    def test_omega_branch_draws_match_the_fused_joint(self) -> None:
        """One random.uniform per sub-batch loss, in order: forced RNN-T (omega 1), forced
        TDT (omega 0) and a seeded mix choose the same branches as the reference."""
        batch = synthetic(12, 64, 16, seed=6)
        fbs, calls = 2, math.ceil(12 / 2)
        results = {}
        for omega, seed in ((1.0, 0), (0.0, 0), (0.5, 11)):
            draws, states = {}, {}
            for path in ("fused", "chunked"):
                record = []
                real = random.uniform

                def spy(a, b, _record=record, _real=real):
                    value = _real(a, b)
                    _record.append(value)
                    return value

                with mock.patch("random.uniform", side_effect=spy):
                    results[omega, path] = run(path, self.joint, batch, fbs, omega=omega, seed=seed)
                draws[path], states[path] = record, random.getstate()
            self.assertEqual(len(draws["fused"]), calls)
            self.assertEqual(draws["chunked"], draws["fused"], f"omega={omega}")
            self.assertEqual(states["chunked"], states["fused"], f"omega={omega}")
            if omega == 0.5:
                branches = [d < omega for d in draws["fused"]]
                self.assertTrue(any(branches) and not all(branches), "seed 11 must give a mix")
            self.assertWithin(results[omega, "fused"], results[omega, "chunked"], TOL_FP32,
                              f"omega={omega}")
        # The branches differ far beyond the tolerance, so a wrong choice cannot pass.
        self.assertFalse(within(results[0.0, "fused"], results[1.0, "fused"], TOL_FP32)[0])
        self.assertFalse(within(results[0.0, "fused"], results[0.5, "fused"], TOL_FP32)[0])
        self.assertFalse(within(results[1.0, "fused"], results[0.5, "fused"], TOL_FP32)[0])


@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class MemoryTest(TinyJoint, unittest.TestCase):
    def test_peak_memory_is_flat_in_the_number_of_sub_batches(self) -> None:
        """Fixed sub-batches (4 rows, equal lengths) x N. The reference keeps every
        sub-batch's FP32 TDT gradient (saved by the numba loss's forward) until backward, so
        its peak grows by at least that much per sub-batch; the chunked path holds one
        sub-batch's graph at a time plus gradients of size B x T x D, B x U x H and the
        joint's parameters."""
        peaks = {}
        for n in (1, 2, 4, 8):
            batch = synthetic(4 * n, 120, 40, seed=7, equal=True)
            for path in ("fused", "chunked"):
                peaks[path, n] = peak_bytes(path, self.joint, batch, 4)
            del batch
        # One sub-batch's FP32 TDT gradient: 4 x 120 x 41 x (1025 + 5) x 4 B = 81 MB. The
        # reference must grow by >= 90% of that per extra sub-batch (measured 84 MB per
        # sub-batch: 246 MB at N = 1, 835 MB at N = 8). The chunked path may grow only by its
        # B-proportional gradients (< 0.2 MB per sub-batch here) and allocator rounding:
        # <= 16 MiB in all (measured +1.3 MB at N = 8); at N = 1 it matches the reference.
        per_sub_batch = 4 * 120 * 41 * (VOCAB + 1 + 5) * 4
        for n in (2, 4, 8):
            self.assertGreaterEqual(peaks["fused", n] - peaks["fused", 1],
                                    0.9 * (n - 1) * per_sub_batch, peaks)
            self.assertLessEqual(peaks["chunked", n], peaks["chunked", 1] + 16 * 2**20, peaks)
        self.assertLessEqual(peaks["chunked", 1], peaks["fused", 1] + 16 * 2**20, peaks)


@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class SafetyTest(TinyJoint, unittest.TestCase):
    def test_reduction_restored_after_an_exception_mid_loop(self) -> None:
        batch = synthetic(12, 64, 16, seed=8)
        self.joint.set_fused_batch_size(4)
        inner = self.joint.loss._loss
        real = inner.forward
        calls = []

        def second_call_fails(*a, **kw):
            calls.append(1)
            if len(calls) == 2:
                raise torch.OutOfMemoryError("simulated")
            return real(*a, **kw)

        enc = batch["enc"].clone().requires_grad_(True)
        with mock.patch.object(inner, "forward", side_effect=second_call_fails):
            with self.assertRaises(torch.OutOfMemoryError):
                loss_of("chunked", self.joint, enc, batch["dec"], batch)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.joint.loss.reduction, self.configured)
        self.assertWithin(run("fused", self.joint, batch, 4), run("chunked", self.joint, batch, 4),
                          TOL_FP32, "after the exception")

    def test_refusals(self) -> None:
        batch = synthetic(6, 32, 8, seed=9)
        enc = batch["enc"].clone().requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "compute_wer"):
            loss_of("chunked", self.joint, enc, batch["dec"], batch, compute_wer=True)
        for reduction in (None, "mean_tokens"):
            self.joint.loss.reduction = reduction
            with self.assertRaisesRegex(ValueError, "reduction"):
                loss_of("chunked", self.joint, enc, batch["dec"], batch)
        self.joint.loss.reduction = self.configured
        loss = loss_of("chunked", self.joint, enc, batch["dec"], batch)
        with self.assertRaisesRegex(RuntimeError, "double backward"):
            torch.autograd.grad(loss, enc, create_graph=True)


@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class NegativeControlTest(TinyJoint, unittest.TestCase):
    def test_checks_catch_scale_errors_and_a_dropped_sub_batch(self) -> None:
        for tol, bf16 in ((TOL_FP32, False), (TOL_BF16, True)):
            batch = synthetic(12, 96, 24, seed=10)
            ref = run("fused", self.joint, batch, 4, bf16=bf16)
            self.assertTrue(identical(ref, run("fused", self.joint, batch, 4, bf16=bf16)))
            scaled = (ref[0] * 1.01, {k: v * 1.01 for k, v in ref[1].items()})
            self.assertFalse(within(ref, scaled, tol)[0], f"bf16={bf16}: 1% loss scale passed")
            grads_only = (ref[0], {k: v * 1.01 for k, v in ref[1].items()})
            self.assertFalse(within(ref, grads_only, tol)[0], f"bf16={bf16}: 1% gradient scale passed")
            dropped = run("fused", self.joint, {k: v[:8] for k, v in batch.items()}, 4, bf16=bf16)
            grads = {}
            for k, v in dropped[1].items():
                if is_joint(k):
                    grads[k] = v
                else:
                    grads[k] = torch.zeros_like(ref[1][k])
                    grads[k][:8, :, :v.shape[-1]] = v
            self.assertFalse(within(ref, (dropped[0], grads), tol)[0],
                             f"bf16={bf16}: dropped sub-batch passed")
        # In FP32 the joint tolerance alone sees a 1% error confined to the joint's gradients;
        # a non-finite joint gradient fails at any tolerance.
        batch = synthetic(12, 96, 24, seed=10)
        ref = run("fused", self.joint, batch, 4)
        joint_only = (ref[0], {k: v * 1.01 if is_joint(k) else v for k, v in ref[1].items()})
        self.assertFalse(within(ref, joint_only, TOL_FP32)[0], "1% joint gradient scale passed")
        name = next(filter(is_joint, ref[1]))
        for bad in (float("nan"), float("inf")):
            broken = (ref[0], {**ref[1], name: torch.full_like(ref[1][name], bad)})
            self.assertFalse(within(ref, broken, TOL_REAL)[0], f"{bad} joint gradient passed")


@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class TrainerTest(unittest.TestCase):
    """train.compute_losses and train.train_step on the tiny model: the full-model recipe
    (encoder, prediction and joint networks trainable, encoder gradient checkpointing), and
    a frozen prediction/joint recipe (P1/P2: only the encoder trains).

    The preprocessor's dither is off: its noise is not reproduced by train.seed_everything
    (measured: two seeded runs differ, and are bitwise equal without dither), and the
    comparison needs a repeatable reference. SpecAugment and encoder dropout stay on."""

    RECIPE = {"encoder_matching": False, "train_pred_joint": True, "quantized": False}
    FROZEN = {"encoder_matching": False, "train_pred_joint": False, "quantized": False}

    @classmethod
    def setUpClass(cls) -> None:
        paths.require_mount()
        train.deterministic_mode()
        train.prepare_tdt_loss()
        torch.manual_seed(0)
        cls.init = tiny_model().state_dict()
        g = torch.Generator().manual_seed(12)
        rows = 7
        lengths = torch.randint(16000, 48000, (rows,), generator=g)
        audio = torch.randn(rows, int(lengths.max()), generator=g) * 0.1
        for i, n in enumerate(lengths.tolist()):
            audio[i, n:] = 0.0
        rng = random.Random(12)
        cls.texts = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 12)))
                     for _ in range(rows)]
        cls.audio, cls.audio_len = audio.cuda(), lengths.cuda()

    def student(self, fbs: int | None, chunked: bool, recipe: dict | None = None):
        model = tiny_model()
        model.load_state_dict(self.init)
        model.preprocessor.featurizer.dither = 0.0
        train.configure_trainable(model, recipe or self.RECIPE)
        model.cuda()
        train.enable_grad_checkpointing(model.encoder)
        record = tdt_chunked.configure(model, fbs, chunked)
        self.assertEqual(record["chunked_backward"], chunked)
        return model

    def labeled(self, model) -> dict:
        train.seed_everything(5, model)  # every RNG, SpecAugment's private ones included
        train.reseed_cudnn_rnn_dropout()
        tokens, token_len = train.tokenize(model, self.texts, "cuda")
        return {"audio": self.audio, "audio_len": self.audio_len, "tokens": tokens,
                "token_len": token_len, "teacher_enc": None, "teacher_len": None}

    def losses(self, model, recipe: dict | None = None) -> tuple:
        lab = self.labeled(model)
        for p in model.parameters():
            p.grad = None
        loss, tdt, _ = train.compute_losses(model, lab["audio"], lab["audio_len"], lab["tokens"],
                                            lab["token_len"], recipe or self.RECIPE)
        self.assertEqual(loss.dim(), 0)
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        return loss.detach(), grads

    def test_reference_is_repeatable_and_its_floor_is_within_tolerance(self) -> None:
        ref = self.losses(self.student(2, False))
        self.assertTrue(identical(ref, self.losses(self.student(2, False))))
        for fbs in (3, 7):
            floor = compare(ref, self.losses(self.student(fbs, False)))
            for k in TOL_PIPELINE:
                self.assertLessEqual(5 * floor[k], TOL_PIPELINE[k], f"fbs={fbs}: {floor}")

    def test_compute_losses_matches_the_fused_path(self) -> None:
        """Loss and every encoder and prediction-network gradient (the encoder's through the
        joint's gradient w.r.t. its output, under checkpointing) bitwise; the joint's within
        TOL_PIPELINE."""
        for fbs in (2, 3):
            ref = self.losses(self.student(fbs, False))
            got = self.losses(self.student(fbs, True))
            self.assertTrue(any(n.startswith("encoder.") for n in got[1]))
            self.assertTrue(any(n.startswith("decoder.") for n in got[1]))
            ok, found = within(ref, got, TOL_PIPELINE)
            self.assertTrue(ok, f"fbs={fbs}: {found}")
        ref = self.losses(self.student(2, False, self.FROZEN), self.FROZEN)
        got = self.losses(self.student(2, True, self.FROZEN), self.FROZEN)
        self.assertFalse(any(n.startswith(("decoder.", "joint.")) for n in got[1]))
        ok, found = within(ref, got, TOL_PIPELINE)
        self.assertTrue(ok, f"frozen prediction/joint: {found}")

    def test_flag_off_never_reaches_the_module(self) -> None:
        model = self.student(None, False)
        self.assertEqual(model.joint.fused_batch_size, tiny_model().joint.fused_batch_size)
        with mock.patch.object(tdt_chunked, "joint_loss", side_effect=AssertionError("called")):
            first = self.losses(model)
        second = self.losses(self.student(None, False))
        self.assertTrue(identical(first, second))

    def test_train_step(self) -> None:
        """train.train_step unchanged: its one loss.backward(), clipping, optimizer step and
        logged values work with the flag on and agree with the flag off."""
        values = {}
        for chunked in (False, True):
            model = self.student(2, chunked)
            optimizer, scheduler = train.make_optimizer(model, 1e-3, 10)
            ctx = types.SimpleNamespace(student=model, recipe=self.RECIPE, optimizer=optimizer,
                                        scheduler=scheduler, state=types.SimpleNamespace(step=0))
            trainable = [p for g in optimizer.param_groups for p in g["params"]]
            before = [p.detach().clone() for p in trainable]
            values[chunked] = train.train_step(ctx, self.labeled(model), trainable)
            self.assertEqual(set(values[chunked]), {"loss", "tdt", "enc", "norm"})
            self.assertTrue(all(math.isfinite(v) for v in values[chunked].values()))
            self.assertTrue(all(not torch.equal(a, p) for a, p in zip(before, trainable)
                                if p.grad is not None and p.grad.abs().sum() > 0))
        self.assertEqual(values[True]["loss"], values[False]["loss"])
        self.assertEqual(values[True]["tdt"], values[False]["tdt"])
        # The norm differs only through the joint's gradients (rel L2 <= TOL_PIPELINE).
        self.assertLessEqual(abs(values[True]["norm"] - values[False]["norm"]),
                             TOL_PIPELINE["rel_l2"] * values[False]["norm"], values)


@unittest.skipUnless(HAS_CUDA and os.environ.get("TDT_CHUNKED_REAL_MODEL") == "1",
                     "set TDT_CHUNKED_REAL_MODEL=1 (loads the pinned 0.6B model)")
class RealModelTest(unittest.TestCase):
    """bf16 autocast on the pinned model's joint (1024/640 -> 640 -> 1025 + 5) with real
    encoder and prediction-network outputs of 12 dev-clean utterances (6-14 s)."""

    @classmethod
    def setUpClass(cls) -> None:
        import evaluate
        paths.require_mount()
        train.deterministic_mode()
        train.prepare_tdt_loss()
        model = train.load_pretrained("cpu")
        train.disable_pred_joint_dropout(model)
        model.cuda().eval()
        records = [r for r in evaluate.read_manifest(paths.MANIFESTS / "librispeech_dev_clean.jsonl")
                   if 6.0 <= r["duration"] <= 14.0][:12]
        audio, lengths = evaluate.audio_batch(records)
        tokens, token_len = train.tokenize(model, [r["text"] for r in records], "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats, feat_len = model.preprocessor(input_signal=audio.cuda(), length=lengths.cuda())
            enc, enc_len = model.encoder(audio_signal=feats, length=feat_len)
            dec, _, _ = model.decoder(targets=tokens, target_length=token_len)
        cls.batch = {"enc": enc.detach(), "dec": dec.detach(), "enc_len": enc_len,
                     "tr_len": token_len, "transcripts": tokens}
        cls.joint = model.joint.train()
        cls.joint.requires_grad_(True)
        del model, feats, enc, dec
        torch.cuda.empty_cache()

    def test_bf16_real_shapes(self) -> None:
        ref = run("fused", self.joint, self.batch, 4, bf16=True)
        for fbs in (1, 2, 3, 6, 12):
            floor = compare(ref, run("fused", self.joint, self.batch, fbs, bf16=True))
            for k in TOL_REAL:
                self.assertLessEqual(5 * floor[k], TOL_REAL[k], f"floor fbs={fbs}: {floor}")
        for fbs in (1, 4, 6, 12):
            ok, found = within(run("fused", self.joint, self.batch, fbs, bf16=True),
                               run("chunked", self.joint, self.batch, fbs, bf16=True), TOL_REAL)
            self.assertTrue(ok, f"fbs={fbs}: {found}")
        scaled = (ref[0] * 1.01, {k: v * 1.01 for k, v in ref[1].items()})
        self.assertFalse(within(ref, scaled, TOL_REAL)[0], "1% scale passed")
        grads_only = (ref[0], {k: v * 1.01 for k, v in ref[1].items()})
        self.assertFalse(within(ref, grads_only, TOL_REAL)[0], "1% gradient scale passed")


if __name__ == "__main__":
    unittest.main()
