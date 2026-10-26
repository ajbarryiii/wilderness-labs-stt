"""Per-sub-batch backward through the TDT joint and loss (train.py --tdt-chunked-backward).

NeMo's fused joint (RNNTJoint.forward with fuse_loss_wer) splits a batch into sub-batches of
fused_batch_size rows, trims each to its own longest encoder output and transcript, and
calls the TDT loss on each; but every sub-batch's saved tensors (the joint's activations and
the FP32 gradient the numba TDT loss computes in its forward) stay alive until
loss.backward(), so peak memory grows with the whole batch.

joint_loss computes the same sub-batches with the same NeMo calls (joint.joint, joint.loss
with reduction None, joint.loss.reduce), in the same order and under the caller's autocast,
but inside one autograd Function: each sub-batch runs on detached inputs, its gradients
w.r.t. the encoder output, the prediction-network output and the joint's parameters are
taken at once (torch.autograd.grad, autocast off, as a backward pass runs) and accumulated,
and its graph is freed before the next sub-batch. The Function's backward scales the stored
gradients by the incoming gradient. Peak memory is then one sub-batch's graph plus gradients
of size B x T x D, B x U x H and the joint's parameters.

Each utterance's weight in the reduction is d reduce / d loss_i, taken from NeMo's own
reduce up front (every supported reduction is linear in the per-utterance losses), so it is
exactly what the fused path's backward feeds the TDT loss for a unit incoming gradient.
Equal to the fused path up to floating-point order: the joint's parameter gradients are
summed over sub-batches in FP32 (NeMo's single backward sums the bf16 gradients of
autocast's cached weight casts, then casts once). The TDT loss's omega draw (one
random.uniform per loss call) happens per sub-batch in the same order, so the same
RNN-T/TDT branches are chosen. Not supported: double backward, compute_wer.
"""
from __future__ import annotations

import torch

ATTR = "tdt_chunked_backward"  # set on the joint by configure; train.compute_losses reads it
SUPPORTED_REDUCTIONS = ("mean_batch", "mean_volume", "mean", "sum")


def configure(student, fused_batch_size: int | None, chunked: bool) -> dict:
    """Set the joint's sub-batch size (None: the model's own) and the chunked backward;
    returns the provenance record. With (None, False) the model is left untouched."""
    joint = student.joint
    default = joint.fused_batch_size
    if fused_batch_size is not None:
        if type(fused_batch_size) is not int or fused_batch_size < 1:
            raise ValueError(f"fused_batch_size must be a positive int, not {fused_batch_size!r}")
        joint.set_fused_batch_size(fused_batch_size)
    if chunked:
        check_supported(joint)
        setattr(joint, ATTR, True)
    return {"chunked_backward": bool(chunked), "fused_batch_size": joint.fused_batch_size,
            "fused_batch_size_default": default, "reduction": joint.loss.reduction}


def enabled(joint) -> bool:
    return bool(getattr(joint, ATTR, False))


def check_supported(joint) -> None:
    if not joint.fuse_loss_wer or joint.loss is None:
        raise ValueError("the chunked TDT backward needs NeMo's fused joint (fuse_loss_wer with a loss)")
    if not joint.fused_batch_size or joint.fused_batch_size < 1:
        raise ValueError(f"fused_batch_size must be a positive int, not {joint.fused_batch_size!r}")
    if joint.loss.reduction not in SUPPORTED_REDUCTIONS:
        raise ValueError(f"the chunked TDT backward supports the loss reductions {SUPPORTED_REDUCTIONS}, "
                         f"not {joint.loss.reduction!r}")


def joint_loss(joint, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor,
               encoder_lengths: torch.Tensor, transcripts: torch.Tensor,
               transcript_lengths: torch.Tensor, compute_wer: bool = False) -> torch.Tensor:
    """The fused joint's reduced TDT loss (a scalar), with the per-sub-batch backward.

    encoder_outputs [B, D, T], decoder_outputs [B, H, U + 1] as for RNNTJoint.forward.
    Without any gradient to compute (grad mode off, or no input or joint parameter requiring
    one) this is NeMo's fused path itself.
    """
    if compute_wer:
        raise ValueError("the chunked TDT backward does not compute WER (compute_wer=True)")
    check_supported(joint)
    params = [p for p in joint.parameters() if p.requires_grad]
    if not (torch.is_grad_enabled() and (encoder_outputs.requires_grad or decoder_outputs.requires_grad
                                         or params)):
        loss, _, _, _ = joint(encoder_outputs=encoder_outputs, decoder_outputs=decoder_outputs,
                              encoder_lengths=encoder_lengths, transcripts=transcripts,
                              transcript_lengths=transcript_lengths, compute_wer=False)
        return loss
    return _ChunkedJointLoss.apply(joint, encoder_lengths, transcripts, transcript_lengths,
                                   encoder_outputs, decoder_outputs, *params)


def _reduction_weights(loss, sub_lengths: list[torch.Tensor], dtype, device) -> list[torch.Tensor]:
    """d loss.reduce / d loss_i per sub-batch (independent of the losses: the reductions are linear)."""
    probes = [torch.zeros(len(n), dtype=dtype, device=device, requires_grad=True) for n in sub_lengths]
    with torch.enable_grad():
        reduced = loss.reduce(list(probes), list(sub_lengths))
    with torch.autocast(device.type, enabled=False):
        return list(torch.autograd.grad(reduced, probes))


class _ChunkedJointLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, joint, encoder_lengths, transcripts, transcript_lengths, enc, dec, *params):
        want_enc, want_dec = ctx.needs_input_grad[4], ctx.needs_input_grad[5]
        loss = joint.loss
        reduction = loss.reduction
        batch, step = int(enc.size(0)), int(joint.fused_batch_size)
        enc_t, dec_t = enc.detach().transpose(1, 2), dec.detach().transpose(1, 2)  # [B, T, D], [B, U+1, H]
        spans = [(b, min(b + step, batch)) for b in range(0, batch, step)]
        sub_lengths = [transcript_lengths[b:e] for b, e in spans]
        grad_enc = torch.zeros_like(enc) if want_enc else None
        grad_dec = torch.zeros_like(dec) if want_dec else None
        grad_params: list[torch.Tensor | None] = [None] * len(params)
        losses, weights = [], None
        try:
            for (begin, end), sub_tr_lens in zip(spans, sub_lengths):
                rows = end - begin
                sub_enc_lens = encoder_lengths[begin:end]
                t, u = int(sub_enc_lens.max()), int(sub_tr_lens.max())
                sub_enc = enc_t.narrow(0, begin, rows)
                if sub_enc.shape[1] != t:
                    sub_enc = sub_enc.narrow(1, 0, t)
                sub_dec = dec_t.narrow(0, begin, rows)
                if sub_dec.shape[1] != u + 1:
                    sub_dec = sub_dec.narrow(1, 0, u + 1)
                sub_tr = transcripts.narrow(0, begin, rows)
                if sub_tr.shape[1] != u:
                    sub_tr = sub_tr.narrow(1, 0, u)
                sub_enc = sub_enc.detach().requires_grad_(want_enc)
                sub_dec = sub_dec.detach().requires_grad_(want_dec)
                with torch.enable_grad():
                    sub_joint = joint.joint(sub_enc, sub_dec)  # [rows, T', U'+1, V + 1 + durations]
                    loss.reduction = None
                    sub_loss = loss(log_probs=sub_joint, targets=sub_tr, input_lengths=sub_enc_lens,
                                    target_lengths=sub_tr_lens)
                    loss.reduction = reduction
                del sub_joint
                if weights is None:
                    weights = _reduction_weights(loss, sub_lengths, sub_loss.dtype, sub_loss.device)
                inputs = [x for x, w in ((sub_enc, want_enc), (sub_dec, want_dec)) if w] + list(params)
                with torch.autocast(sub_loss.device.type, enabled=False):
                    grads = list(torch.autograd.grad(sub_loss, inputs, grad_outputs=weights[len(losses)],
                                                     allow_unused=True))
                losses.append(sub_loss.detach())
                del sub_loss
                if want_enc:
                    g = grads.pop(0)
                    if g is not None:
                        grad_enc.transpose(1, 2).narrow(0, begin, rows).narrow(1, 0, t).copy_(g)
                if want_dec:
                    g = grads.pop(0)
                    if g is not None:
                        grad_dec.transpose(1, 2).narrow(0, begin, rows).narrow(1, 0, u + 1).copy_(g)
                for i, g in enumerate(grads):
                    if g is not None:
                        grad_params[i] = g if grad_params[i] is None else grad_params[i].add_(g)
                del grads, inputs, sub_enc, sub_dec
        finally:
            loss.reduction = reduction
        ctx.grads = (grad_enc, grad_dec, *grad_params)
        return loss.reduce(losses, sub_lengths)

    @staticmethod
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError("the chunked TDT backward does not support double backward (create_graph=True)")
        scaled = [None if g is None else g * grad_output for g in ctx.grads]
        return (None, None, None, None, *scaled)
