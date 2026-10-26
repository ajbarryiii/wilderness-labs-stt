"""Contract tests for tdt_fused (train.py --tdt-fused-kernel): NeMo's fused joint and TDT loss in
one call per batch, without the [B, T, U, V] joint output.

Reference: NeMo's own RNNTJoint fused path (fuse_loss_wer, sub-batches of fused_batch_size rows)
with the numba TDT loss as train.prepare_tdt_loss() patches it, on the same module and inputs;
small shapes also against NeMo's pure-PyTorch TDTLossPytorch. Compared: the reduced loss and the
gradients w.r.t. every joint parameter and both inputs (the encoder and prediction-network
outputs, so df and dg through the projections). Synthetic cases cover variable T and U with
padding, empty transcripts, T < U (duration-0 emissions), single-row batches, sigma, other
duration sets, chunk splits, the omega (plain RNN-T) branch and the reductions. Peak CUDA memory
must grow with B*T*U per-cell scalars, not with the joint's 1030 outputs per cell. The trainer
hook (train.compute_losses under BF16 autocast) and the flag plumbing (default off, recorded,
resume check, refusals) run on test_train's tiny model.

Tolerances (TOL_*): each is >= 5x the reference's own measured noise floor (comment next to it);
NegativeControlTest shows each still catches a real defect.

Run (GPU): finetune/parakeet-ternary/python -m unittest discover -s finetune/parakeet-ternary/tests -p test_tdt_fused.py -v
"""
from __future__ import annotations

import argparse
import math
import os
import random
import unittest
from unittest import mock

import torch

import paths
import train

HAS_CUDA = torch.cuda.is_available()
DURATIONS = (0, 1, 2, 3, 4)  # the model's (Kea-T NeMo log); sigma and omega likewise
SIGMA, OMEGA = 0.02, 0.1
# (T, U) per row: padding in both directions, empty transcripts, T == 1, T < U (duration-0
# emissions only), a long row that the tiny chunk budget splits along T.
ROWS = ((17, 6), (1, 0), (1, 3), (2, 5), (7, 0), (9, 4), (12, 2))
SMALL = dict(vocab=8, enc_hidden=12, pred_hidden=10, joint_hidden=16)
REAL = dict(vocab=1024, enc_hidden=1024, pred_hidden=640, joint_hidden=640)  # Parakeet-TDT 0.6B
LONG_ROWS = ((375, 150), (375, 0), (40, 150), (300, 120))  # ~30 s clips (T frames, U tokens)
NEGATIVE_ROWS = ((120, 40), (100, 30), (110, 0), (90, 35), (60, 50))

# Tolerances. Noise floors measured 2026-10-26 on the 5090 from NeMo's reference alone: the
# reference against itself under mathematically equivalent re-groupings (fused_batch_size 1, 2,
# 3 and B against the model's 4; the rows permuted; a plain repeat) and, in FP32, NeMo's numba
# loss against its pure-PyTorch TDTLossPytorch; the floor is the largest difference over those
# variants and 2-4 seeds per case. Each tolerance is about 10x (FP32) or 5x (BF16) that floor.
# FP32, the synthetic cases (ROWS, seeds 0-3; sigma 0, durations (0,1,2,4,8), (0,1); omega = 1
# for the RNN-T branch; TDTLossPytorch on 3 rows): floors loss 2.1e-7, whole-gradient relative
# L2 1.7e-6, per-tensor max element 3.7e-6, cosine gap 3.8e-13 (all from regrouping; numba vs
# PyTorch stays below them: 9.4e-8, 9.2e-7, 1.3e-6, 7.1e-14).
TOL_FP32 = dict(loss_rel=2.5e-6, grad_rel_l2=2e-5, grad_max_rel=4e-5, cosine_gap=4e-12)
# BF16 autocast, real joint sizes (LONG_ROWS and NEGATIVE_ROWS, seeds 0-1): floors loss 2.4e-4,
# whole-gradient L2 1.45e-2, per-tensor max element 0.108, cosine gap 1.05e-4. The gradients
# w.r.t. the encoder input and projection inherit df = sum_u dh, NeMo's BF16 dh summed with
# cancellation; a repeat with the same grouping is bitwise identical.
TOL_BF16_REAL = dict(loss_rel=1.2e-3, grad_rel_l2=7.5e-2, grad_max_rel=0.55, cosine_gap=5.5e-4)
# BF16 autocast through train.compute_losses, tiny decoder-only model (tiny_batch seeds 0-3):
# floors loss 6.0e-5, whole-gradient L2 5.6e-3, per-tensor max element 8.2e-3, cosine gap 1.5e-5;
# a plain repeat alone reaches 4.5e-5, 4.1e-3, 8.1e-3, 8.4e-6 (the GPU path is not deterministic).
TOL_BF16_MODEL = dict(loss_rel=3e-4, grad_rel_l2=2.8e-2, grad_max_rel=4.1e-2, cosine_gap=7.7e-5)
# Peak-memory slope (bytes per lattice cell): the fused loss keeps FP32 scalars per cell (alpha,
# beta, 8 statistics, 8 coefficients: 72 B), measured 92 B/cell with the per-token tensors; NeMo
# keeps at least the FP32 [.., 1030] gradient (4120 B/cell) per cell, measured 15.7 KB/cell.
MAX_FUSED_BYTES_PER_CELL = 512
MIN_MATERIALIZED_BYTES_PER_CELL = 4 * 1030


def setUpModule() -> None:
    if HAS_CUDA and "AGENT_JOBS_VRAM_FRACTION" in os.environ:
        torch.cuda.set_per_process_memory_fraction(float(os.environ["AGENT_JOBS_VRAM_FRACTION"]))


# ----------------------------------------------------------------------------- helpers

class _NoWer:
    """RNNTJoint's fused path insists on a WER module; compute_wer=False never uses it."""


def make_joint(vocab: int, enc_hidden: int, pred_hidden: int, joint_hidden: int,
               durations=DURATIONS, sigma: float = SIGMA, omega: float = 0.0,
               reduction: str = "mean_batch", fused_batch_size: int = 4, seed: int = 0,
               logit_scale: float = 1.0, dropout: float = 0.0):
    """A NeMo RNNTJoint wired as in the model (fused loss, TDT numba loss), on the GPU, training mode.
    logit_scale multiplies the output layer so logits have a realistic spread."""
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    from nemo.collections.asr.modules.rnnt import RNNTJoint
    torch.manual_seed(seed)
    joint = RNNTJoint(jointnet={"encoder_hidden": enc_hidden, "pred_hidden": pred_hidden,
                                "joint_hidden": joint_hidden, "activation": "relu", "dropout": dropout},
                      num_classes=vocab, num_extra_outputs=len(durations), fuse_loss_wer=True,
                      fused_batch_size=fused_batch_size)
    joint.set_loss(RNNTLoss(num_classes=vocab, reduction=reduction, loss_name="tdt",
                            loss_kwargs={"durations": list(durations), "sigma": sigma, "omega": omega,
                                         "clamp": -1.0, "fastemit_lambda": 0.0}))
    joint.set_wer(_NoWer())
    with torch.no_grad():
        joint.joint_net[-1].weight.mul_(logit_scale)
    return joint.cuda().train()


def make_batch(rows, vocab: int, enc_hidden: int, pred_hidden: int, seed: int = 0,
               pad_t: int = 2, pad_u: int = 1, scale: float = 1.0, **_) -> dict:
    """Encoder output [B, D, T] and prediction-network output [B, D, U + 1] with padding past
    every row's length (as the trainer's batches have), transcripts zero-padded like train.tokenize."""
    g = torch.Generator().manual_seed(seed)
    t_max, u_max = max(t for t, _ in rows) + pad_t, max(u for _, u in rows) + pad_u
    tokens = torch.zeros(len(rows), max(1, u_max), dtype=torch.long)
    for i, (_, u) in enumerate(rows):
        tokens[i, :u] = torch.randint(0, vocab, (u,), generator=g)
    return {"enc": (scale * torch.randn(len(rows), enc_hidden, t_max, generator=g)).cuda(),
            "dec": (scale * torch.randn(len(rows), pred_hidden, tokens.shape[1] + 1, generator=g)).cuda(),
            "enc_len": torch.tensor([t for t, _ in rows]).cuda(),
            "tokens": tokens.cuda(), "token_len": torch.tensor([u for _, u in rows]).cuda()}


def run(joint, batch: dict, fused: bool, fused_batch_size: int | None = None, autocast: bool = False,
        loss_scale: float = 1.0) -> dict:
    """One forward + backward of NeMo's fused joint (fused=False) or tdt_fused.joint_loss; the loss
    and the gradients w.r.t. both inputs and every joint parameter (FP32 copies)."""
    train.prepare_tdt_loss()
    if fused_batch_size is not None:
        joint.set_fused_batch_size(fused_batch_size)
    joint.zero_grad(set_to_none=True)
    enc = batch["enc"].clone().requires_grad_(True)
    dec = batch["dec"].clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        if fused:
            import tdt_fused
            loss = tdt_fused.joint_loss(joint, enc, dec, batch["enc_len"], batch["tokens"], batch["token_len"])
        else:
            loss = joint(encoder_outputs=enc, decoder_outputs=dec, encoder_lengths=batch["enc_len"],
                         transcripts=batch["tokens"], transcript_lengths=batch["token_len"],
                         compute_wer=False)[0]
    (loss * loss_scale).backward()
    grads = {"enc": enc.grad, "dec": dec.grad} | {n: p.grad for n, p in joint.named_parameters()}
    return {"loss": float(loss.detach()) * loss_scale, "dtype": loss.dtype, "shape": tuple(loss.shape),
            "grads": {k: v.detach().double().cpu() for k, v in grads.items()}}


def compare(ref: dict, out: dict) -> dict:
    """Scale-aware differences: the loss's relative difference; over the whole gradient (every
    tensor together) the relative L2 difference and 1 - cosine similarity; per tensor the largest
    element difference relative to that tensor's largest element (grad_max_rel, the maximum over
    tensors). per_grad_rel_l2 is reported, not checked: in BF16 single tensors whose elements are
    sums with cancellation (df, dg) differ by percents between NeMo's own sub-batchings."""
    rel_l2, max_rel = {}, {}
    dot = na = nb = nd = 0.0
    for name, a in ref["grads"].items():
        b = out["grads"][name]
        rel_l2[name] = float((a - b).norm() / a.norm().clamp_min(1e-300))
        max_rel[name] = float((a - b).abs().max() / a.abs().max().clamp_min(1e-300))
        dot, na, nb = dot + float((a * b).sum()), na + float((a * a).sum()), nb + float((b * b).sum())
        nd += float(((a - b) ** 2).sum())
    return {"loss_rel": abs(out["loss"] - ref["loss"]) / abs(ref["loss"]),
            "grad_rel_l2": math.sqrt(nd / na), "grad_max_rel": max(max_rel.values()),
            "cosine_gap": 1.0 - dot / math.sqrt(na * nb), "per_grad_rel_l2": rel_l2}


def within(diff: dict, tol: dict) -> list[str]:
    return [f"{k} {diff[k]:.3g} > {tol[k]:.3g}" for k in tol if not diff[k] <= tol[k]]


class _Case(unittest.TestCase):
    def assert_matches(self, ref: dict, out: dict, tol: dict, label: str = "") -> None:
        diff = compare(ref, out)
        self.assertEqual(within(diff, tol), [], f"{label}: {diff}")
        self.assertTrue(all(torch.isfinite(g).all() for g in out["grads"].values()), label)


# ----------------------------------------------------------------------------- flag plumbing (CPU)

class FlagTest(unittest.TestCase):
    def test_default_off_recorded_and_resume_checked(self) -> None:
        base = ["--arm", "P3", "--lr", "1e-4", "--run-name", "x", "--max-steps", "2"]
        off, on = train.parse_args(base), train.parse_args(base + ["--tdt-fused-kernel"])
        self.assertIs(off.tdt_fused_kernel, False)
        self.assertIs(on.tdt_fused_kernel, True)
        # A checkpoint written before the flag existed resumes with the flag off, and only then.
        older = {k: v for k, v in vars(off).items() if k != "tdt_fused_kernel"}
        self.assertEqual(train.resume_mismatches(older, off), [])
        self.assertTrue(train.resume_mismatches(older, on))
        self.assertTrue(train.resume_mismatches(vars(off), on))
        self.assertIn("parakeet-ternary/tdt_fused.py", train.TRAINING_SOURCES)

    def test_refuses_the_other_joint_memory_flags(self) -> None:
        import tdt_fused
        for other in ({"tdt_chunked_backward": True}, {"joint_recompute": True}, {"fused_batch_size": 8}):
            args = argparse.Namespace(tdt_fused_kernel=True, **other)
            with self.assertRaisesRegex(ValueError, "tdt-fused-kernel"):
                tdt_fused.configure(mock.Mock(), args)

    def test_off_configures_nothing(self) -> None:
        import tdt_fused
        student = argparse.Namespace(joint=torch.nn.Module())
        self.assertIsNone(tdt_fused.configure(student, argparse.Namespace(tdt_fused_kernel=False)))
        self.assertIsNone(tdt_fused.configure(student, argparse.Namespace()))
        self.assertFalse(tdt_fused.enabled(student.joint))


@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class RefusalTest(unittest.TestCase):
    def test_refuses_what_it_does_not_implement(self) -> None:
        import tdt_fused
        batch = make_batch(ROWS[:2], **SMALL)
        with self.assertRaisesRegex(ValueError, "dropout"):
            tdt_fused.enable(make_joint(**SMALL, dropout=0.2))
        joint = make_joint(**SMALL)
        loss = joint.loss._loss
        for attr, value, pattern in (("fastemit_lambda", 0.01, "fastemit"), ("clamp", 1.0, "clamp")):
            with mock.patch.object(loss, attr, value), self.assertRaisesRegex(ValueError, pattern):
                tdt_fused.enable(joint)
        with mock.patch.object(joint, "masking_prob", 0.1), self.assertRaisesRegex(ValueError, "masking"):
            tdt_fused.enable(joint)
        with mock.patch.object(joint, "log_softmax", True), self.assertRaisesRegex(ValueError, "log_softmax"):
            tdt_fused.enable(joint)
        # Dropout switched back on after enabling is refused at the call, in training mode.
        joint = make_joint(**SMALL)
        tdt_fused.enable(joint)
        self.assertTrue(tdt_fused.enabled(joint))
        joint.joint_net.insert(1, torch.nn.Dropout(0.1))
        with self.assertRaisesRegex(ValueError, "dropout"):
            run(joint, batch, fused=True)


# ----------------------------------------------------------------------------- equivalence (FP32)

@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class EquivalenceTest(_Case):
    def check(self, rows, label: str, chunk_cells: int | None = None, seed: int = 0, **joint_kw) -> None:
        import tdt_fused
        joint = make_joint(**SMALL, seed=seed, **joint_kw)
        batch = make_batch(rows, **SMALL, seed=seed)
        ref = run(joint, batch, fused=False)
        with mock.patch.object(tdt_fused, "CHUNK_CELLS", chunk_cells or tdt_fused.CHUNK_CELLS):
            out = run(joint, batch, fused=True)
        self.assertEqual((out["dtype"], out["shape"]), (ref["dtype"], ref["shape"]))
        self.assert_matches(ref, out, TOL_FP32, label)

    def test_matches_nemo_fused_joint(self) -> None:
        for seed in (0, 1):
            self.check(ROWS, f"default chunks seed {seed}", seed=seed)
            # 24 cells: the 17 x 7 row is split along T and the others are grouped.
            self.check(ROWS, f"tiny chunks seed {seed}", chunk_cells=24, seed=seed)

    def test_single_rows_and_empty_transcripts(self) -> None:
        for rows in (((5, 0),), ((1, 0),), ((3, 4),), ((1, 0), (1, 0)), ((6, 0), (4, 0))):
            self.check(rows, f"rows {rows}")

    def test_sigma_durations_and_reductions(self) -> None:
        self.check(ROWS, "sigma 0", sigma=0.0)
        self.check(ROWS, "sigma 0.05", sigma=0.05)
        self.check(ROWS, "durations 0,1,2,4,8", durations=(0, 1, 2, 4, 8))
        self.check(ROWS, "durations 0,1", durations=(0, 1))
        for reduction in ("mean_batch", "sum", "mean_volume"):
            self.check(ROWS, f"reduction {reduction}", reduction=reduction)
        # NeMo's "mean" divides by the transcript length: rows with tokens only.
        self.check(tuple(r for r in ROWS if r[1]), "reduction mean", reduction="mean")

    def test_matches_pytorch_reference(self) -> None:
        """NeMo's pure-PyTorch TDT loss (no omega, no fused joint): logits from the same joint."""
        from nemo.collections.asr.losses.rnnt import RNNTLoss
        rows = ((6, 3), (4, 1), (2, 3))
        joint = make_joint(**SMALL, seed=3)
        batch = make_batch(rows, **SMALL, seed=3, pad_t=0, pad_u=0)
        ref_loss = RNNTLoss(num_classes=SMALL["vocab"], reduction="mean_batch", loss_name="tdt_pytorch",
                            loss_kwargs={"durations": list(DURATIONS), "sigma": SIGMA})
        joint.zero_grad(set_to_none=True)
        enc = batch["enc"].clone().requires_grad_(True)
        dec = batch["dec"].clone().requires_grad_(True)
        logits = joint.joint(enc.transpose(1, 2), dec.transpose(1, 2))
        loss = ref_loss(log_probs=logits, targets=batch["tokens"], input_lengths=batch["enc_len"],
                        target_lengths=batch["token_len"])
        loss.backward()
        grads = {"enc": enc.grad, "dec": dec.grad} | {n: p.grad for n, p in joint.named_parameters()}
        ref = {"loss": loss.item(), "grads": {k: v.double().cpu() for k, v in grads.items()}}
        self.assert_matches(ref, run(joint, batch, fused=True), TOL_FP32, "TDTLossPytorch")


@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class OmegaTest(_Case):
    """omega: one uniform draw per loss call (as NeMo's gpu_rnnt.py); below omega the whole call
    is the plain RNN-T loss (no durations, no sigma). One call per batch: one draw per batch."""

    def test_rnnt_branch_matches_numba(self) -> None:
        for seed in (0, 1):
            joint = make_joint(**SMALL, seed=seed, omega=1.0)  # every NeMo sub-batch draws RNN-T
            batch = make_batch(ROWS, **SMALL, seed=seed)
            self.assert_matches(run(joint, batch, fused=False), run(joint, batch, fused=True), TOL_FP32,
                                f"RNN-T seed {seed}")

    def test_one_draw_per_call_selects_the_branch(self) -> None:
        batch = make_batch(ROWS, **SMALL)
        tdt = run(make_joint(**SMALL, omega=0.0), batch, fused=False)
        rnnt = run(make_joint(**SMALL, omega=1.0), batch, fused=False)
        joint = make_joint(**SMALL, omega=0.3)
        for draw, ref in ((0.29, rnnt), (0.31, tdt)):
            with mock.patch.object(random, "uniform", return_value=draw) as uniform:
                out = run(joint, batch, fused=True)
            self.assertEqual(uniform.call_count, 1)
            self.assertEqual(uniform.call_args.args, (0, 1))
            self.assert_matches(ref, out, TOL_FP32, f"draw {draw}")


# ----------------------------------------------------------------------------- memory and real shapes

def _peak_bytes(joint, batch: dict, fused: bool) -> int:
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    run(joint, batch, fused=fused, autocast=True)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class MemoryTest(unittest.TestCase):
    def test_peak_memory_grows_with_cells_not_vocabulary(self) -> None:
        """Peak CUDA memory of forward + backward (BF16 autocast, real joint sizes) for transcripts of
        two lengths at the same B and T, so per-frame costs cancel: the slope per extra lattice cell
        (B*T*(U+1)) separates per-cell scalars from a per-cell [1030] tensor. Each path runs once
        first, so one-time allocations (cuBLAS workspace) do not count. The same measurement of
        NeMo's path is the negative control."""
        import tdt_fused
        joint = make_joint(**REAL, logit_scale=8.0, fused_batch_size=4)
        sizes = [((300, 40),) * 4, ((300, 160),) * 4]  # 49k and 193k cells, 1200 frames each
        cells = [sum(t * (u + 1) for t, u in rows) for rows in sizes]
        slope = {}
        with mock.patch.object(tdt_fused, "CHUNK_CELLS", 4096):  # full chunks at both sizes
            for fused in (True, False):
                batches = [make_batch(rows, **REAL, pad_t=0, pad_u=0) for rows in sizes]
                _peak_bytes(joint, batches[0], fused)
                peaks = [_peak_bytes(joint, batch, fused) for batch in batches]
                self.assertGreater(min(peaks), 0)
                slope[fused] = (peaks[1] - peaks[0]) / (cells[1] - cells[0])
        self.assertGreater(slope[True], 0, slope)
        self.assertLess(slope[True], MAX_FUSED_BYTES_PER_CELL, slope)
        self.assertGreater(slope[False], MIN_MATERIALIZED_BYTES_PER_CELL, slope)


@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class RealShapeTest(_Case):
    def test_long_clips_bf16(self) -> None:
        """~30 s clips at the real joint sizes under BF16 autocast: finite, and NeMo's result
        within the BF16 tolerances (the joint's padded rows and an empty transcript included)."""
        joint = make_joint(**REAL, logit_scale=8.0, fused_batch_size=4)
        batch = make_batch(LONG_ROWS, **REAL, pad_t=3, pad_u=2)
        ref = run(joint, batch, fused=False, autocast=True)
        out = run(joint, batch, fused=True, autocast=True)
        self.assertTrue(math.isfinite(out["loss"]))
        self.assertEqual((out["dtype"], out["shape"]), (torch.float32, ()))
        self.assert_matches(ref, out, TOL_BF16_REAL, "long clips")


# ----------------------------------------------------------------------------- trainer hook

def tiny_batch(student, rows: int = 5, seed: int = 0) -> dict:
    from test_train import WORDS
    g = torch.Generator().manual_seed(seed)
    lengths = [int(16000 * (1.0 + 2.0 * float(torch.rand((), generator=g)))) for _ in range(rows)]
    audio = torch.zeros(rows, max(lengths))
    for i, n in enumerate(lengths):
        audio[i, :n] = 0.1 * torch.randn(n, generator=g)
    texts = [" ".join(WORDS[(seed + i + j) % len(WORDS)] for j in range(1 + 3 * i)) for i in range(rows)]
    texts[1] = ""  # an empty transcript
    tokens, token_len = train.tokenize(student, texts, "cuda")
    return {"audio": audio.cuda(), "audio_len": torch.tensor(lengths).cuda(), "tokens": tokens,
            "token_len": token_len}


def decoder_only_tiny_model():
    from test_train import tiny_model
    paths.require_mount()
    train.deterministic_mode()
    student = tiny_model()
    train.configure_trainable(student, TrainerHookTest.RECIPE)
    student.encoder.requires_grad_(False)
    student.encoder.eval()
    return student.cuda()


def hook_run(student, batch: dict, fused: bool, fused_batch_size: int = 4, loss_scale: float = 1.0) -> dict:
    """train.compute_losses + backward with the fused loss enabled or not; same seeds every call."""
    import tdt_fused
    student.joint.set_fused_batch_size(fused_batch_size)
    if fused:
        tdt_fused.enable(student.joint)
    try:
        train.seed_everything(7, student)
        student.zero_grad(set_to_none=True)
        loss, tdt, enc = train.compute_losses(student, batch["audio"], batch["audio_len"], batch["tokens"],
                                              batch["token_len"], TrainerHookTest.RECIPE)
        (loss * loss_scale).backward()
    finally:
        tdt_fused.disable(student.joint)
    assert enc is None and loss is tdt
    return {"loss": float(loss) * loss_scale, "dtype": loss.dtype, "shape": tuple(loss.shape),
            "grads": {n: p.grad.double().cpu() for n, p in student.named_parameters() if p.requires_grad}}


@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class TrainerHookTest(_Case):
    """The decoder-only recipe (relabel/train_decoder.py: trainable prediction and joint networks,
    frozen encoder in eval mode), whose GPU path is deterministic; df still reaches joint.enc."""
    RECIPE = {"encoder_matching": False, "train_pred_joint": True, "quantized": False}

    @classmethod
    def setUpClass(cls) -> None:
        cls.student = decoder_only_tiny_model()
        for module in cls.omega_modules(cls.student):
            module.omega = 0.0  # compared branch by branch; OmegaTest covers the draw

    @staticmethod
    def omega_modules(student) -> list:
        return [student.loss._loss, student.joint.loss._loss]

    def test_flag_on_matches_flag_off_under_bf16_autocast(self) -> None:
        for seed in (0, 1):
            batch = tiny_batch(self.student, seed=seed)
            ref, out = hook_run(self.student, batch, fused=False), hook_run(self.student, batch, fused=True)
            self.assertEqual((out["dtype"], out["shape"]), (ref["dtype"], ref["shape"]))
            self.assertEqual(out["dtype"], torch.float32)
            self.assertEqual(out["grads"].keys(), ref["grads"].keys())
            self.assert_matches(ref, out, TOL_BF16_MODEL, f"tiny model seed {seed}")

    def test_train_step_runs_with_the_flag(self) -> None:
        import types
        import tdt_fused
        student = decoder_only_tiny_model()  # NeMo models cannot be deep-copied
        record = tdt_fused.configure(student, argparse.Namespace(tdt_fused_kernel=True))
        self.assertTrue(tdt_fused.enabled(student.joint))
        self.assertEqual(record["chunk_cells"], tdt_fused.CHUNK_CELLS)
        optimizer, scheduler = train.make_optimizer(student, 1e-4, 10)
        ctx = types.SimpleNamespace(student=student, recipe=self.RECIPE, optimizer=optimizer,
                                    scheduler=scheduler, state=types.SimpleNamespace(step=0))
        batch = tiny_batch(student, seed=2) | {"teacher_enc": None, "teacher_len": None}
        trainable = [p for group in optimizer.param_groups for p in group["params"]]
        before = [p.detach().clone() for p in trainable]
        with mock.patch.object(type(student.joint), "forward", side_effect=AssertionError("joint called")):
            values = train.train_step(ctx, batch, trainable)
        self.assertTrue(all(math.isfinite(v) for v in values.values()), values)
        self.assertTrue(any(not torch.equal(a, p) for a, p in zip(before, trainable)))

    def test_flag_off_never_consults_the_fused_loss(self) -> None:
        import tdt_fused
        batch = tiny_batch(self.student, seed=3)
        with mock.patch.object(tdt_fused, "joint_loss", side_effect=AssertionError("fused path used")):
            calls = []
            original = type(self.student.joint).forward

            def counting(module, *a, **kw):
                calls.append(1)
                return original(module, *a, **kw)

            with mock.patch.object(type(self.student.joint), "forward", counting):
                train.compute_losses(self.student, batch["audio"], batch["audio_len"], batch["tokens"],
                                     batch["token_len"], self.RECIPE)
        self.assertEqual(len(calls), 1)


# ----------------------------------------------------------------------------- negative controls

@unittest.skipUnless(HAS_CUDA, "the TDT loss is CUDA-only")
class NegativeControlTest(unittest.TestCase):
    """The tolerances catch defects of the size that matters: a 1% loss-scale error (loss), a row
    (one of NeMo's sub-batches when fused_batch_size = 1) dropped from the batch (loss, whole
    gradient L2 and cosine) and the blank logit's gradient missing (the per-element check)."""

    def dropped_row(self, joint, batch: dict, autocast: bool) -> dict:
        """NeMo's result with the last row's loss and gradient missing (mean over all rows kept)."""
        rows = batch["enc_len"].numel()
        part = {k: v[:-1] for k, v in batch.items()}
        out = run(joint, part, fused=False, autocast=autocast, loss_scale=(rows - 1) / rows)
        for name in ("enc", "dec"):
            out["grads"][name] = torch.cat([out["grads"][name], torch.zeros_like(out["grads"][name][:1])])
        return out

    def check(self, joint, batch: dict, tol: dict, autocast: bool) -> dict:
        """Each defect must fail the tolerance; returns the 1%-scaled comparison."""
        ref = run(joint, batch, fused=False, autocast=autocast)
        scaled = compare(ref, run(joint, batch, fused=False, autocast=autocast, loss_scale=1.01))
        self.assertTrue(within(scaled, {"loss_rel": tol["loss_rel"]}), scaled)
        dropped = compare(ref, self.dropped_row(joint, batch, autocast))
        for metric in ("loss_rel", "grad_rel_l2", "cosine_gap"):
            self.assertTrue(within(dropped, {metric: tol[metric]}), (metric, dropped))
        no_blank = {"loss": ref["loss"], "grads": dict(ref["grads"])}
        out = f"joint_net.{len(joint.joint_net) - 1}"
        blank = joint.num_classes_with_blank - 1 - joint.num_extra_outputs
        for name in (f"{out}.weight", f"{out}.bias"):
            no_blank["grads"][name] = no_blank["grads"][name].clone()
            no_blank["grads"][name][blank] = 0.0
        missing = compare(ref, no_blank)
        self.assertTrue(within(missing, {"grad_max_rel": tol["grad_max_rel"]}), missing)
        return scaled

    def test_fp32_tolerances(self) -> None:
        scaled = self.check(make_joint(**SMALL), make_batch(ROWS, **SMALL), TOL_FP32, autocast=False)
        self.assertTrue(within(scaled, {"grad_rel_l2": TOL_FP32["grad_rel_l2"]}), scaled)

    def test_bf16_model_tolerances(self) -> None:
        student = decoder_only_tiny_model()
        for module in TrainerHookTest.omega_modules(student):
            module.omega = 0.0
        batch = tiny_batch(student, seed=0)
        ref = hook_run(student, batch, fused=False)
        scaled = compare(ref, hook_run(student, batch, fused=False, loss_scale=1.01))
        self.assertTrue(within(scaled, {"loss_rel": TOL_BF16_MODEL["loss_rel"]}), scaled)
        rows = batch["audio"].shape[0]
        part = {k: v[:-1] for k, v in batch.items()}
        dropped = compare(ref, hook_run(student, part, fused=False, loss_scale=(rows - 1) / rows))
        for metric in ("loss_rel", "grad_rel_l2", "cosine_gap"):
            self.assertTrue(within(dropped, {metric: TOL_BF16_MODEL[metric]}), (metric, dropped))
        no_blank = {"loss": ref["loss"], "grads": dict(ref["grads"])}
        blank = student.joint.num_classes_with_blank - 1 - student.joint.num_extra_outputs
        out = f"joint.joint_net.{len(student.joint.joint_net) - 1}"
        for name in (f"{out}.weight", f"{out}.bias"):
            no_blank["grads"][name] = no_blank["grads"][name].clone()
            no_blank["grads"][name][blank] = 0.0
        missing = compare(ref, no_blank)
        self.assertTrue(within(missing, {"grad_max_rel": TOL_BF16_MODEL["grad_max_rel"]}), missing)

    def test_bf16_tolerances(self) -> None:
        self.check(make_joint(**REAL, logit_scale=8.0), make_batch(NEGATIVE_ROWS, **REAL, pad_t=3, pad_u=2),
                   TOL_BF16_REAL, autocast=True)


if __name__ == "__main__":
    unittest.main()
