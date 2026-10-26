"""--joint-recompute: recompute the transducer joint's hidden activations in the backward pass.

NeMo's RNNTJoint.joint_after_projection(f, g) computes relu(f + g) -> dropout -> Linear on a
[B, T, U, H] grid. Autograd keeps the ReLU output (with dropout p > 0 also the dropout output
and its mask) until backward, for every sub-batch of NeMo's fused loss loop, so the fused
loop bounds only transient joint memory. enable() wraps the method in non-reentrant
activation checkpointing: the forward keeps only f [B, T, H] and g [B, U, H], and backward
recomputes the grid when it first needs it (checkpoint's early stop ends the recompute once
the output Linear's saved input exists, so the [B, T, U, V] GEMM is not repeated). The RNG
state is preserved, so dropout masks match; the recompute runs the same kernels on the same
inputs and backward is the same graph, so loss and gradients equal the plain joint's
exactly wherever the plain joint is itself reproducible (tests/test_joint_recompute.py; on
real batches neither is bitwise run to run, and the deviation stays within that spread).

The method is replaced on the joint instance only; train.py and NeMo are not modified. Every
caller of the joint (NeMo's fused loop, a per-sub-batch backward, decoding) goes through it;
without gradients (decoding, export) it is the plain method.

Seam for a consuming trainer (train.py itself is frozen):
1. add_arguments(parser) registers --joint-recompute (default off) on the consumer's parser;
2. after the model is built and on its device, configure(student, args.joint_recompute)
   applies it and returns the provenance record, which the consumer stores with its run
   configuration; with the flag off the model is untouched;
3. the consumer appends SOURCES to train.TRAINING_SOURCES through its seam;
4. a resume must refuse a change of the flag, and a checkpoint recorded before the flag
   existed counts as off (also in any fallback that compares only a stored resume key).

Composition with the other TDT-memory seams: a per-sub-batch backward
(tdt_chunked.joint_loss) still calls joint.joint(...) per sub-batch, so the recompute applies
inside it; its torch.autograd.grad runs with autocast off, and the recompute replays the
forward's autocast state (checkpoint records it), as tested here. A patch of the TDT loss
instance acts downstream of the joint output and is independent. Order of application
does not matter.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import types

import torch
from torch.utils.checkpoint import checkpoint

SOURCES = ("parakeet-ternary/joint_recompute.py",)  # relative to finetune/, for train.TRAINING_SOURCES


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--joint-recompute", action="store_true",
                        help="recompute the joint's [B, T, U, H] hidden activations in the backward "
                             "pass (joint_recompute.py)")


def configure(student, enabled: bool) -> dict:
    """Apply --joint-recompute to a built model; returns the provenance record."""
    record = {"joint_recompute": bool(enabled)}
    if enabled:
        enable(student.joint)
        record["module_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return record


def _joint_after_projection(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    inner = type(self).joint_after_projection
    if not torch.is_grad_enabled():
        return inner(self, f, g)
    # f and g are positional tensors: checkpoint preserves the RNG of their device.
    return checkpoint(inner, self, f, g, use_reentrant=False, preserve_rng_state=True)


def is_enabled(joint: torch.nn.Module) -> bool:
    method = vars(joint).get("joint_after_projection")
    return getattr(method, "__func__", None) is _joint_after_projection


def enable(joint: torch.nn.Module) -> None:
    """Recompute joint.joint_after_projection in the backward pass (see the module docstring)."""
    if not callable(getattr(type(joint), "joint_after_projection", None)):
        raise TypeError(f"{type(joint).__name__} has no joint_after_projection; "
                        "not a NeMo transducer joint")
    if "joint_after_projection" in vars(joint):
        raise RuntimeError("this joint's joint_after_projection is already replaced"
                           + (" (joint recompute is enabled)" if is_enabled(joint) else ""))
    norms = [name for name, m in joint.named_modules()
             if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if norms:
        raise ValueError(f"the joint has BatchNorm modules {norms}; the recompute would "
                         "update their running statistics twice per step")
    # A bound method (not a closure): deepcopy rebinds it to the copy.
    joint.joint_after_projection = types.MethodType(_joint_after_projection, joint)


def disable(joint: torch.nn.Module) -> None:
    """Undo enable(): the joint is the plain NeMo joint again."""
    if not is_enabled(joint):
        raise RuntimeError("joint recompute is not enabled on this joint")
    del joint.joint_after_projection
