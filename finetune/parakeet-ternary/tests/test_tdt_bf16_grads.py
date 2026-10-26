"""Contract tests for --tdt-bf16-grads (tdt_bf16_grads.install).

NeMo's TDT loss computes the full fp32 gradient w.r.t. the label logits during the forward
pass and keeps it until backward. Under the trainer's bf16 autocast the joint's logits are
bf16; RNNTLoss casts them to fp32, and the backward of that cast rounds the fp32 gradient
back to bf16. Keeping the saved label gradient in bf16 therefore moves one bf16 rounding:

    flag off: g = bf16(L * s)          L: the kernel's fp32 label gradient
    flag on:  g = bf16(bf16(L) * s)    s: the upstream gradient (fp32, per utterance)

The product is formed in fp32 (s is not rounded to bf16). |bf16(L) * s - L * s| < 1 bf16 ulp
of L * s, and two reals less than one grid step apart round to grid points at most one step
apart, so every label element of the gradient w.r.t. the joint's output changes by at most
one bf16 ulp. The five duration columns keep their fp32 saved gradient and must not change.

Tolerances (both reference arms are bitwise repeatable on the GPU: a repeated flag-off run
gives identical loss and gradients, which test_reference_is_deterministic checks; the
difference under test is therefore the deliberate rounding, not run-to-run noise):
- loss: bitwise equal (the forward math is untouched).
- gradient w.r.t. the joint's output: label columns within 1 bf16 ulp elementwise (the bound
  above, exact); duration columns bitwise equal.
- gradients w.r.t. the joint's parameters and inputs: (a) bitwise what the unchanged autograd
  graph gives when the flag-on output gradient is fed into the flag-off graph (so the flag
  changes nothing but that rounding), and (b) relative L2 <= 2**-7 and elementwise
  |diff| <= 2**-6 * max|ref|. 2**-7 is the bound's largest relative change of any output
  gradient element (1 ulp <= 2**-7 |g|), carried to the whole tensor; the elementwise bound is
  two bf16 ulps of the tensor's largest element (the backward GEMMs round their bf16 outputs
  once more). Measured on this case (2026-10-26): relative L2 0.9e-4 to 4.1e-3, the same size
  as re-running the flag-off reference with a different, mathematically equivalent sub-batch
  size (up to 5.8e-3). A 5x margin over either would exceed 1e-2 and could not catch the
  1 % defect required of the negative controls, so the analytic bound is the tolerance and
  (a) plus the output-gradient bound carry the exact proof. Negative controls: a 1 % scaled
  and a dropped-sub-batch gradient fail both checks.
- held memory after the forward pass (after a warm-up call): falls by sum(B*T*U*1025) * 2
  bytes (half of the fp32 label gradients) within max(5x the flag-off repeat difference,
  512 B per sub-batch: the allocator rounds each block to 512 B). Measured: repeat
  difference 0 B, drop 256 B under the expectation.

Shapes are the pinned model's joint (encoder 1024, prediction 640, joint hidden 640,
1024 tokens + blank + 5 durations, sub-batches of 4, sigma 0.02, mean_volume reduction) with
random weights and inputs; omega is forced to 0 (TDT kernels) and 1 (the sampled plain
RNN-T kernels), which share the saved-gradient path.

Run: finetune/parakeet-ternary/python -m unittest discover -s finetune/parakeet-ternary/tests -v
"""
from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

import torch

import paths
import train

HAS_CUDA = torch.cuda.is_available()
VOCAB = 1024  # tokens; + blank = 1025 label logits
DURATIONS = [0, 1, 2, 3, 4]
JOINT = {"encoder_hidden": 1024, "pred_hidden": 640, "joint_hidden": 640}
REL_L2_TOL = 2.0 ** -7  # see the module docstring
ELEMENT_TOL = 2.0 ** -6  # x max|ref|; see the module docstring


def tdt_loss(omega: float):
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    return RNNTLoss(num_classes=VOCAB, loss_name="tdt", reduction="mean_volume",
                    loss_kwargs={"durations": DURATIONS, "sigma": 0.02, "omega": omega,
                                 "clamp": -1.0, "fastemit_lambda": 0.0})


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of bf16 numbers at |x|: |x| = m * 2**e with m in [0.5, 1) -> 2**(e - 8)."""
    _, e = torch.frexp(x.float().abs())
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), (e - 8).clamp_min(-133))


def within_one_ulp(ref: torch.Tensor, other: torch.Tensor) -> bool:
    a, b = ref.float(), other.float()
    return bool(((a - b).abs() <= bf16_ulp(torch.maximum(a.abs(), b.abs()))).all())


def close(ref: torch.Tensor, other: torch.Tensor) -> tuple[bool, float]:
    """(within REL_L2_TOL and ELEMENT_TOL, relative L2 difference)."""
    a, b = ref.double(), other.double()
    rel = float((a - b).norm() / a.norm())
    worst = float((a - b).abs().max())
    return rel <= REL_L2_TOL and worst <= ELEMENT_TOL * float(a.abs().max()), rel


class JointCase:
    """NeMo's fused RNNTJoint at the pinned model's sizes on seeded random inputs."""

    def __init__(self, seed: int = 0, rows: int = 8, lengths: tuple | None = None) -> None:
        from nemo.collections.asr.modules.rnnt import RNNTJoint
        torch.manual_seed(seed)
        self.joint = RNNTJoint(jointnet={**JOINT, "activation": "relu", "dropout": 0.0},
                               num_classes=VOCAB, num_extra_outputs=len(DURATIONS),
                               fuse_loss_wer=True, fused_batch_size=4).cuda().train()
        self.joint.set_wer(object())  # the fused path requires one; compute_wer=False never uses it
        g = torch.Generator().manual_seed(seed + 1)
        if lengths is None:
            t = torch.randint(20, 60, (rows,), generator=g)
            u = torch.randint(5, 20, (rows,), generator=g)
        else:
            t, u = torch.full((rows,), lengths[0]), torch.full((rows,), lengths[1])
        self.enc = torch.randn(rows, JOINT["encoder_hidden"], int(t.max()), generator=g).cuda()
        self.dec = torch.randn(rows, JOINT["pred_hidden"], int(u.max()) + 1, generator=g).cuda()
        self.tokens = torch.randint(0, VOCAB, (rows, int(u.max())), generator=g).cuda()
        self.t, self.u = t.cuda(), u.cuda()

    def forward(self, loss, dtype=torch.bfloat16, keep_logits: bool = True):
        """(loss, joint outputs per sub-batch, enc, dec); dtype None runs without autocast."""
        self.joint.set_loss(loss)
        for p in self.joint.parameters():
            p.grad = None
        enc, dec = self.enc.clone().requires_grad_(True), self.dec.clone().requires_grad_(True)
        logits = []
        handle = self.joint.joint_net[-1].register_forward_hook(
            lambda module, inputs, output: logits.append(output) if keep_logits else None)
        try:
            with torch.autocast("cuda", dtype=dtype or torch.bfloat16, enabled=dtype is not None):
                value = self.joint(encoder_outputs=enc, decoder_outputs=dec, encoder_lengths=self.t,
                                   transcripts=self.tokens, transcript_lengths=self.u,
                                   compute_wer=False)[0]
        finally:
            handle.remove()
        return value, logits, enc, dec

    def grads(self, enc, dec) -> dict:
        out = {name: p.grad.detach().clone() for name, p in self.joint.named_parameters()}
        out.update(enc=enc.grad.detach().clone(), dec=dec.grad.detach().clone())
        return out

    def run(self, loss) -> dict:
        value, logits, enc, dec = self.forward(loss)
        for x in logits:
            x.retain_grad()
        value.backward()
        return {"loss": value.detach(), "output_grads": [x.grad for x in logits],
                "grads": self.grads(enc, dec)}

    def backward_from(self, loss, output_grads: list) -> dict:
        """Gradients of the flag-off graph when fed `output_grads` at the joint's outputs."""
        _, logits, enc, dec = self.forward(loss)
        torch.autograd.backward(logits, output_grads)
        return self.grads(enc, dec)

    def held_bytes(self, loss) -> tuple[int, int]:
        """(bytes held after the forward pass, peak bytes over forward and backward)."""
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        value, _, enc, dec = self.forward(loss, keep_logits=False)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - before
        value.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        del value, enc, dec
        return held, peak


@unittest.skipUnless(HAS_CUDA, "TDT loss is CUDA-only")
class Bf16GradsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        paths.require_mount()
        train.deterministic_mode()
        cls.kernel_sha = train.prepare_tdt_loss()
        cls.mod = train.local_module("tdt_bf16_grads")
        cls.case = JointCase()

    def installed(self, omega: float):
        loss = tdt_loss(omega)
        self.mod.install(loss)
        return loss

    def test_reference_is_deterministic(self) -> None:
        """The flag-off reference repeats bitwise, so the tolerances below need no noise term."""
        for omega in (0.0, 1.0):
            a, b = self.case.run(tdt_loss(omega)), self.case.run(tdt_loss(omega))
            self.assertTrue(torch.equal(a["loss"], b["loss"]))
            for name in a["grads"]:
                self.assertTrue(torch.equal(a["grads"][name], b["grads"][name]), name)

    def test_loss_and_gradients(self) -> None:
        for omega in (0.0, 1.0):
            with self.subTest(omega=omega):
                ref = self.case.run(tdt_loss(omega))
                on = self.case.run(self.installed(omega))
                self.assertTrue(torch.equal(ref["loss"], on["loss"]), "forward loss changed")
                differ = 0
                for r, o in zip(ref["output_grads"], on["output_grads"], strict=True):
                    self.assertEqual((r.dtype, o.dtype), (torch.bfloat16, torch.bfloat16))
                    labels = VOCAB + 1
                    self.assertTrue(within_one_ulp(r[..., :labels], o[..., :labels]))
                    self.assertTrue(torch.equal(r[..., labels:], o[..., labels:]),
                                    "duration gradients changed")
                    differ += int((r != o).sum())
                # Not vacuous: mean_volume's upstream 1/sum(U) is no power of two, so the moved
                # rounding changes some elements (with a power of two it would change none).
                self.assertGreater(differ, 0)
                exact = self.case.backward_from(tdt_loss(omega), on["output_grads"])
                for name, g in on["grads"].items():
                    self.assertTrue(torch.equal(exact[name], g), f"{name}: not the graph's image")
                    ok, rel = close(ref["grads"][name], g)
                    self.assertTrue(ok, f"{name}: relative L2 {rel:.2e}")

    def test_negative_controls(self) -> None:
        """A 1 % scaled or a dropped-sub-batch gradient fails the checks above."""
        ref = self.case.run(tdt_loss(0.0))
        r = ref["output_grads"][0]
        self.assertFalse(within_one_ulp(r, (r.float() * 1.01).to(r.dtype)))
        self.assertFalse(within_one_ulp(r, torch.zeros_like(r)))
        dropped = self.case.backward_from(
            tdt_loss(0.0), [torch.zeros_like(r)] + ref["output_grads"][1:])
        for name, g in ref["grads"].items():
            self.assertFalse(close(g, g * 1.01)[0], f"{name}: 1 % scale not caught")
            self.assertFalse(close(g, dropped[name])[0], f"{name}: dropped sub-batch not caught")

    def test_refuses_logits_that_are_not_bf16(self) -> None:
        """The one-rounding argument needs bf16 logits: fp32 (no autocast) and fp16 are refused;
        the flag-off loss accepts them as before."""
        for dtype in (None, torch.float16):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(RuntimeError, "bf16"):
                    self.case.forward(self.installed(0.0), dtype=dtype)
                value, *_ = self.case.forward(tdt_loss(0.0), dtype=dtype)
                self.assertTrue(torch.isfinite(value))

    def test_install_is_idempotent_and_coexists_with_the_kernel_patch(self) -> None:
        once, twice = tdt_loss(0.0), tdt_loss(0.0)
        sha = self.mod.install(once)
        self.assertEqual(self.mod.install(twice), sha)
        self.assertEqual(self.mod.install(twice), sha)
        self.assertRegex(sha, r"^[0-9a-f]{64}$")
        self.assertEqual(train.prepare_tdt_loss(), self.kernel_sha)
        a, b = self.case.run(once), self.case.run(twice)
        self.assertTrue(torch.equal(a["loss"], b["loss"]))
        for name in a["grads"]:
            self.assertTrue(torch.equal(a["grads"][name], b["grads"][name]), name)

    def test_flag_off_is_unchanged(self) -> None:
        """Installing on one loss leaves every other loss on today's path, bitwise."""
        before = self.case.run(tdt_loss(0.0))
        self.installed(0.0)
        after = self.case.run(tdt_loss(0.0))
        self.assertTrue(torch.equal(before["loss"], after["loss"]))
        for name in before["grads"]:
            self.assertTrue(torch.equal(before["grads"][name], after["grads"][name]), name)

    def test_saved_memory_halves(self) -> None:
        """16 rows of T = 200, U = 40 in 4 sub-batches: the fp32 label gradients held between
        forward and backward are 16 * 200 * 41 * 1025 * 4 bytes; the flag holds half."""
        case = JointCase(seed=3, rows=16, lengths=(200, 40))
        expected = 16 * 200 * 41 * (VOCAB + 1) * 2
        case.held_bytes(tdt_loss(0.0))  # warm-up: first-call workspaces (cuBLAS, numba) persist
        off_a, peak_off = case.held_bytes(tdt_loss(0.0))
        off_b, _ = case.held_bytes(tdt_loss(0.0))
        on, peak_on = case.held_bytes(self.installed(0.0))
        tolerance = max(5 * abs(off_a - off_b), 512 * 4)
        self.assertLessEqual(abs((off_a - on) - expected), tolerance,
                             f"held off {off_a} on {on}: drop {off_a - on}, expected {expected}")
        # Over the whole step the last sub-batch's fp32 gradient still exists transiently
        # (forward: next to its bf16 copy; backward: the fp32 product), so with n = 4
        # sub-batches the peak falls by about (n - 2) / n of the held saving; assert half of it.
        self.assertGreaterEqual(peak_off - peak_on, 0.5 * expected * (4 - 2) / 4,
                                f"step peak off {peak_off} on {peak_on}")

    def test_context_installs_only_with_the_flag(self) -> None:
        """train.Context applies the patch for --tdt-bf16-grads and records its SHA-256."""
        from test_train import make_args
        run = Path(tempfile.mkdtemp())
        for flag in (None, True):
            with self.subTest(flag=flag):
                student = torch.nn.ModuleDict({"loss": tdt_loss(0.0), "proj": torch.nn.Linear(2, 2)})
                ctx = train.Context(make_args(tdt_bf16_grads=flag), run, student, None, None, None)
                self.assertEqual(ctx.tdt_kernel_sha256, self.kernel_sha)
                if flag:
                    self.assertEqual(ctx.tdt_bf16_grads_sha256, self.mod.install(tdt_loss(0.0)))
                    with self.assertRaisesRegex(RuntimeError, "bf16"):
                        self.case.forward(student["loss"], dtype=None)
                else:
                    self.assertIsNone(ctx.tdt_bf16_grads_sha256)
                    self.case.forward(student["loss"], dtype=None)


if __name__ == "__main__":
    unittest.main()
