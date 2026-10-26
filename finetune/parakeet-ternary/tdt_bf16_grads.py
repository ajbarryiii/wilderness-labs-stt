"""--tdt-bf16-grads: keep the TDT loss's saved label gradient in bf16 (opt-in runtime patch).

NeMo's _TDTNumba computes the full fp32 gradient w.r.t. the label logits (B x T x U x 1025)
in the forward pass and saves it until backward; it is the largest tensor the joint keeps
alive. Under the trainer's bf16 autocast the joint's logits are bf16 and RNNTLoss casts them
to fp32, so the backward of that cast rounds the gradient to bf16 anyway. Saving it in bf16
moves that rounding before the multiplication by the upstream gradient:

    today:   bf16(L * s)          with this patch:   bf16(bf16(L) * s)

The product stays in fp32 (the original backward casts the upstream gradient to the saved
tensor's dtype, which would round s to bf16 as well; the patch multiplies in fp32 instead),
which bounds the change of every label-gradient element to one bf16 ulp. The saved duration
gradients feed an fp32 log_softmax backward and stay fp32, so the duration columns of the
gradient w.r.t. the joint's output are unchanged. The argument needs bf16
logits: install() makes each patched loss refuse logits of any other dtype.

The patch is a source rebuild of _TDTNumba (asserted text replacements, nothing else
changed, as train.prepare_tdt_loss rebuilds the gradient kernels) installed on given
TDTLossNumba instances only; other losses keep NeMo's class. The TDT and the sampled plain
RNN-T kernels (omega) share this saved-gradient path. tests/test_tdt_bf16_grads.py holds the
contract.
"""
from __future__ import annotations

import hashlib
import inspect

import torch

_REPLACEMENTS = (
    ("class _TDTNumba(Function):", "class _TDTNumbaBf16Grads(Function):"),
    ("ctx.save_for_backward(label_grads, duration_grads)",
     "ctx.save_for_backward(None if label_grads is None else label_grads.to(torch.bfloat16),\n"
     "                              duration_grads)"),
    ("grad_output = grad_output.view(-1, 1, 1, 1).to(label_grads)",
     "grad_output = grad_output.view(-1, 1, 1, 1).to(duration_grads)"),
    ("label_grads.mul_(grad_output),", "label_grads * grad_output,"),  # bf16 * fp32 -> fp32
)


def patched_function():
    """(the rebuilt autograd Function, SHA-256 of its source); built once per process."""
    from nemo.collections.asr.parts.numba.rnnt_loss import rnnt_pytorch as rp
    cached = getattr(rp, "_parakeet_ternary_bf16_grads", None)
    if cached is not None:
        return cached
    source = inspect.getsource(rp._TDTNumba)
    for old, new in _REPLACEMENTS:
        if source.count(old) != 1:
            raise RuntimeError(f"NeMo _TDTNumba changed; expected one {old!r}")
        source = source.replace(old, new)
    namespace: dict = {}
    exec(compile(source, rp.__file__, "exec"), rp.__dict__, namespace)
    rp._parakeet_ternary_bf16_grads = (namespace["_TDTNumbaBf16Grads"],
                                       hashlib.sha256(source.encode()).hexdigest())
    return rp._parakeet_ternary_bf16_grads


def _require_bf16_logits(module, args, kwargs) -> None:
    logits = kwargs["log_probs"] if "log_probs" in kwargs else args[0]
    if logits.dtype != torch.bfloat16 or not module._force_float32:
        raise RuntimeError(f"--tdt-bf16-grads needs bf16 joint logits cast to fp32 by the loss "
                           f"(bf16 autocast); got {logits.dtype} logits, force_float32="
                           f"{module._force_float32}")


def install(model: torch.nn.Module) -> str:
    """Patch every TDT loss in `model` (an RNNTLoss or a module containing one); idempotent.

    Returns the SHA-256 of the rebuilt _TDTNumba source (run provenance).
    """
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import TDTLossNumba
    function, digest = patched_function()
    losses = [m for m in model.modules() if isinstance(m, RNNTLoss) and isinstance(m._loss, TDTLossNumba)]
    if not losses:
        raise RuntimeError("--tdt-bf16-grads: no NeMo TDT loss (RNNTLoss over TDTLossNumba) in the model")
    for loss in losses:
        if getattr(loss, "_parakeet_ternary_bf16_grads", False):
            continue
        loss._loss.loss = function.apply
        loss.register_forward_pre_hook(_require_bf16_logits, with_kwargs=True)
        loss._parakeet_ternary_bf16_grads = True
    return digest
