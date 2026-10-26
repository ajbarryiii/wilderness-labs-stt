"""Per-sub-batch backward through the TDT joint and loss (--tdt-chunked-backward), a seam for
consumers of finetune/parakeet-ternary/train.py (which stays unmodified).

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
reduce up front (every supported reduction is linear in the per-utterance losses; the pinned
config's is mean_volume), so it is exactly what the fused path's backward feeds the TDT loss
for a unit incoming gradient. Equal to the fused path up to floating-point order: the
joint's parameter gradients are summed over sub-batches in FP32 (NeMo's single backward
sums the bf16 gradients of autocast's cached weight casts, then casts once). The TDT loss's
omega draw (one random.uniform per loss call) happens per sub-batch in the same order, so
the same RNN-T/TDT branches are chosen. Not supported: double backward, compute_wer.

Seam. Nothing in the train module is replaced: enable(joint) replaces the forward method of
one joint instance, and train.compute_losses, which calls student.joint(..., compute_wer=False),
then gets joint_loss's scalar (as (loss, None, None, None)); train.train_step's single
loss.backward() applies the stored gradients. Every other call of the joint (no
fuse_loss_wer, no decoder outputs, no gradient: decoding, export, evaluation) goes to NeMo's
own forward. A consumer:

1. add_arguments(parser): --tdt-chunked-backward (default off) and --fused-batch-size N
   (default None: the model's own, 4);
2. after building the training model, every session (first start and every resume; the
   patch is not in checkpoints), record = configure(student, args.fused_batch_size,
   args.tdt_chunked_backward), and record `record` in its provenance (config.json,
   summary.json): the settings, the model's default size, the reduction and, with the flag
   on, this module's SHA-256;
3. a resume must refuse a change of either argument (a checkpoint recorded without them
   means off / None);
4. SOURCES are appended to train.TRAINING_SOURCES through the consumer's seam, so the source
   guards cover this module;
5. a replacement of train.compute_losses must keep calling student.joint(...) for the TDT loss,
   or the chunked path is bypassed.

Composition. The patches are on different attributes of different instances and compose in
any order: this module replaces joint.forward; joint recompute (joint_recompute.enable)
replaces joint.joint_after_projection, which NeMo's joint.joint calls for every sub-batch here
(its checkpoint recomputes inside this module's per-sub-batch autograd.grad); the bf16 TDT
gradients (tdt_bf16_grads.install) replace the TDT loss instance's autograd Function and add a
pre-hook on RNNTLoss, both reached through the joint.loss(...) call made per sub-batch here.
"""
from __future__ import annotations

import argparse
import hashlib
import types
from pathlib import Path

import torch

SOURCES = ("parakeet-ternary/tdt_chunked.py",)  # relative to finetune/, for train.TRAINING_SOURCES
SUPPORTED_REDUCTIONS = ("mean_batch", "mean_volume", "mean", "sum")


def positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, not {value}")
    return n


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tdt-chunked-backward", action="store_true",
                        help="backward through the TDT joint and loss one sub-batch at a time "
                             "(tdt_chunked.py): peak memory of one sub-batch, not the batch")
    parser.add_argument("--fused-batch-size", type=positive_int,
                        help="rows per sub-batch of the fused TDT joint (default: the model's)")


def configure(student, fused_batch_size: int | None, chunked: bool) -> dict:
    """Set the joint's sub-batch size (None: the model's own) and enable the chunked backward;
    returns the provenance record. With (None, False) the model is left untouched."""
    joint = student.joint
    default = joint.fused_batch_size
    if fused_batch_size is not None:
        if type(fused_batch_size) is not int or fused_batch_size < 1:
            raise ValueError(f"fused_batch_size must be a positive int, not {fused_batch_size!r}")
        joint.set_fused_batch_size(fused_batch_size)
    record = {"chunked_backward": bool(chunked), "fused_batch_size": joint.fused_batch_size,
              "fused_batch_size_default": default, "reduction": joint.loss.reduction}
    if chunked:
        enable(joint)
        record["module_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return record


def _forward(self, encoder_outputs, decoder_outputs, encoder_lengths=None, transcripts=None,
             transcript_lengths=None, compute_wer=False, keep_hypotheses=False):
    """RNNTJoint.forward with the fused loss computed by joint_loss. NeMo's signature: its
    typecheck asks for keywords, but export disables typechecking and RNNTDecoderJoint calls
    the joint positionally (after _prepare_for_export turned fuse_loss_wer off)."""
    if not self.fuse_loss_wer or decoder_outputs is None:
        return _nemo_forward(self, encoder_outputs=encoder_outputs, decoder_outputs=decoder_outputs,
                             encoder_lengths=encoder_lengths, transcripts=transcripts,
                             transcript_lengths=transcript_lengths, compute_wer=compute_wer,
                             keep_hypotheses=keep_hypotheses)
    if encoder_lengths is None or transcript_lengths is None:
        raise ValueError("the fused joint needs encoder and transcript lengths")
    loss = joint_loss(self, encoder_outputs, decoder_outputs, encoder_lengths, transcripts,
                      transcript_lengths, compute_wer=compute_wer)
    self.hypotheses = None
    return loss, None, None, None


def _nemo_forward(joint, **kwargs):
    """NeMo's own (typechecked) forward, whether or not the instance's is replaced."""
    return type(joint).forward.__get__(joint, type(joint))(**kwargs)


def enabled(joint) -> bool:
    return getattr(vars(joint).get("forward"), "__func__", None) is _forward


def enable(joint) -> None:
    """Route the joint's fused loss through joint_loss (see the module docstring)."""
    check_supported(joint)
    if "forward" in vars(joint):
        raise RuntimeError("this joint's forward is already replaced"
                           + (" (the chunked TDT backward is enabled)" if enabled(joint) else ""))
    joint.forward = types.MethodType(_forward, joint)


def disable(joint) -> None:
    if enabled(joint):
        del joint.forward


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
        loss, _, _, _ = _nemo_forward(joint, encoder_outputs=encoder_outputs,
                                      decoder_outputs=decoder_outputs, encoder_lengths=encoder_lengths,
                                      transcripts=transcripts, transcript_lengths=transcript_lengths,
                                      compute_wer=False)
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
