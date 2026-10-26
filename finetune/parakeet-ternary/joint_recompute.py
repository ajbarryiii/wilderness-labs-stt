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

The method is replaced on the instance, so NeMo's joint() and forward(), and every caller
of the joint (the fused loop, a per-sub-batch backward, decoding), go through it. Without
gradients (decoding, export) it is the plain method.
"""
from __future__ import annotations

import types

import torch
from torch.utils.checkpoint import checkpoint


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
