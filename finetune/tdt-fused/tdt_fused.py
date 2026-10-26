"""Fused joint + TDT loss as a train.py seam: NeMo's fused joint and numba TDT loss for a whole batch
in one call, without the [B, T, U+1, V] joint output (V = 1030: 1024 tokens, blank, 5 durations).

The seam (finetune/parakeet-ternary is never modified; the pattern of relabel/training/seam.py):
replacements() gives the (module, attribute, value) list a consuming trainer installs, and
installed() installs it and restores the originals on exit:

1. train.compute_losses: compute_losses below, the original train.compute_losses (features,
   SpecAugment, BF16 autocast, encoder matching, the scalar check: all unchanged) with only the
   joint's fused loss call, student.joint(..., compute_wer=False), answered by joint_loss for the
   duration of the call;
2. train.TRAINING_SOURCES: plus SOURCE, so train.py records this file's SHA-256 with its own (at
   the start, at every resume and at finishing). settings() is the record for the consumer's
   configuration and resume key.

Another seam on train.compute_losses, or any replaced method of the joint or the TDT loss (they
would be bypassed silently), is refused.

NeMo (RNNTJoint.forward with fuse_loss_wer) runs the joint and the TDT loss once per sub-batch of
fused_batch_size rows, and every sub-batch keeps its FP32 [.., V] gradient and joint activations
until loss.backward(). Here the TDT recursions need only per-cell scalars: the log-sum-exp over
the 1025 label logits, the blank and target logits and the duration log-softmax. So:

- forward: the lattice cells (b, t, u) are cut into chunks of at most CHUNK_CELLS cells (rows of
  similar lengths grouped, long rows split along T; never less than one frame); per chunk the
  joint hidden state h = relu(f + g) and z = Linear_out(h) are computed exactly as NeMo does
  (same ops and dtypes) and reduced by a Triton kernel to the per-cell scalars; then one Triton
  launch runs the alpha recursion of every row (NeMo's compute_tdt_alphas_kernel, or
  compute_alphas_kernel on the omega branch, on those scalars).
- backward: one launch for the beta recursion, one for the per-cell occupancies; then per chunk
  z is recomputed, turned into dz in place (Triton; the gradient NeMo's grad kernels write,
  including the duration log-softmax backward, rounded to the logits' dtype as NeMo's cast
  backward does), and dW += dz^T h, db += sum dz, dh = dz W (masked by relu) reduced into df
  and dg. Only the per-cell scalars (alpha, beta, 8 statistics, 8 coefficients) grow with the
  batch; chunk buffers are bounded by CHUNK_CELLS.

Semantics follow NeMo: the same transitions, sigma (logit under-normalization), the duration-0
rules, mean_batch (or any) reduction through the joint's RNNTLoss.reduce, and omega (one
random.uniform(0, 1) draw per loss call; below omega the call is the plain RNN-T loss). With one
call per batch the omega branch is chosen per batch, not per sub-batch of 4 rows: the expected
loss is unchanged, its step-to-step variance is not. Projections, the prediction network and the
encoder stay in PyTorch. Not implemented (refused): joint dropout p > 0, masking_prob, adapters,
an explicit log_softmax in the joint, fastemit_lambda, gradient clamping.
"""
from __future__ import annotations

import train  # noqa: I001  finetune/parakeet-ternary/train.py first: it configures numba for NeMo

import contextlib
import hashlib
import inspect
import random
import types
from dataclasses import dataclass
from pathlib import Path

import nemo
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import TDTLossNumba
from triton.language.extra import libdevice

SOURCE = "tdt-fused/tdt_fused.py"  # relative to finetune/, as train.TRAINING_SOURCES entries are
CHUNK_CELLS = 16384  # lattice cells per joint chunk: h, z/dz and dh of a chunk stay in the 5090's L2
STATS = 3  # per-cell statistics before the duration log-probs: lse, logp(blank), logp(target)
# Joint methods this path stands in for or relies on: any replacement of them is refused.
JOINT_METHODS = ("forward", "joint", "joint_after_projection", "project_encoder", "project_prednet")
# The originals as this module found them. A replacement is refused by identity (functools.wraps
# keeps a wrapper's module and name, a wrapt wrapper even its __code__) and by code location (a
# replacement made before this import).
_COMPUTE_LOSSES = train.compute_losses
_NEMO_DIR = Path(nemo.__file__).resolve().parent
_NEMO_METHODS = {(cls, name): inspect.getattr_static(cls, name)
                 for cls, names in ((RNNTJoint, JOINT_METHODS), (RNNTLoss, ("forward",)), (TDTLossNumba, ("forward",)))
                 for name in names}


@dataclass(frozen=True)
class Spec:
    blank: int
    durations: tuple[int, ...]
    sigma: float
    omega: float


# ----------------------------------------------------------------------------- the seam

def replacements() -> list[tuple]:
    """(object, attribute, value) for the consumer's seam; train.compute_losses must be train.py's own."""
    current = train.compute_losses
    if (current is not _COMPUTE_LOSSES or not _defined_in(current, Path(train.__file__))
            or SOURCE in train.TRAINING_SOURCES):
        raise ValueError(f"train.compute_losses is already replaced ({current!r}) or tdt_fused is installed: "
                         "tdt_fused cannot be combined with another compute_losses seam")
    return [(train, "compute_losses", compute_losses),
            (train, "TRAINING_SOURCES", tuple(train.TRAINING_SOURCES) + (SOURCE,))]


@contextlib.contextmanager
def installed():
    """Install replacements(); restore the originals on exit."""
    items = replacements()
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in items]
    try:
        for obj, name, value in items:
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


def settings() -> dict:
    """What a consumer records with the run (and should make part of its resume key)."""
    path = Path(__file__).resolve()
    return {"source": SOURCE, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "chunk_cells": CHUNK_CELLS, "triton": triton.__version__,
            "replaces": "train.compute_losses: the joint's fused loss call, one loss call per batch",
            "omega_draw": "one random.uniform(0, 1) per loss call = one per batch"}


def compute_losses(student, *args, **kwargs):
    """train.compute_losses with student.joint's fused loss call answered by joint_loss."""
    joint = student.joint
    _check_not_replaced(joint)
    joint.forward = types.MethodType(_fused_forward, joint)
    try:
        return _COMPUTE_LOSSES(student, *args, **kwargs)
    finally:
        del joint.forward


def _fused_forward(joint, encoder_outputs, decoder_outputs, encoder_lengths=None, transcripts=None,
                   transcript_lengths=None, compute_wer=False, keep_hypotheses=False):
    """RNNTJoint.forward's fused loss path: (loss, wer, wer_num, wer_denom) with compute_wer=False."""
    if compute_wer or keep_hypotheses or decoder_outputs is None or transcripts is None:
        raise RuntimeError("tdt_fused answers only the joint's loss call (compute_wer=False, with transcripts)")
    loss = joint_loss(joint, encoder_outputs, decoder_outputs, encoder_lengths, transcripts, transcript_lengths)
    return loss, None, None, None


def _check_not_replaced(joint) -> None:
    """ValueError unless the joint and its losses are NeMo's RNNTJoint, RNNTLoss and TDTLossNumba with
    their own methods (none replaced on the class or the instance, no subclass override)."""
    loss = getattr(joint, "loss", None)
    for obj, cls in ((joint, RNNTJoint), (loss, RNNTLoss), (getattr(loss, "_loss", None), TDTLossNumba)):
        if not isinstance(obj, cls):
            raise ValueError(f"tdt_fused needs NeMo's {cls.__name__} on the joint, not {type(obj).__name__}")
        for name in JOINT_METHODS if cls is RNNTJoint else ("forward",):
            own = vars(obj).get(name)
            if own is not None and getattr(own, "__func__", None) is not _fused_forward:
                raise ValueError(f"{type(obj).__name__}.{name} is replaced on the instance: tdt_fused cannot be "
                                 "combined with another seam on the joint or the loss")
            method = inspect.getattr_static(type(obj), name)
            if method is not _NEMO_METHODS[(cls, name)] or not _defined_in(method, _NEMO_DIR):
                raise ValueError(f"{type(obj).__name__}.{name} is not NeMo's own ({method!r}): tdt_fused cannot be "
                                 "combined with another seam on the joint or the loss")


def _defined_in(function, path: Path) -> bool:
    """Whether the function's code lives in path (a file, or a directory tree)."""
    code = getattr(function, "__code__", None)
    if code is None:
        return False
    where = Path(code.co_filename).resolve()
    return where == path.resolve() or where.is_relative_to(path.resolve())


def loss_spec(joint) -> Spec:
    """The joint's TDT loss configuration; ValueError for anything this path does not implement."""
    rnnt_loss = getattr(joint, "loss", None)
    inner = getattr(rnnt_loss, "_loss", None)
    if type(inner).__name__ != "TDTLossNumba":
        raise ValueError(f"tdt_fused needs NeMo's numba TDT loss on the joint, not {type(inner).__name__}")
    if inner.fastemit_lambda:
        raise ValueError("tdt_fused does not implement fastemit_lambda != 0")
    if inner.clamp > 0:
        raise ValueError("tdt_fused does not implement gradient clamp > 0")
    durations = tuple(int(d) for d in inner.durations)
    if (list(durations) != sorted(set(durations)) or 1 not in durations or durations[0] < 0):
        raise ValueError(f"tdt_fused needs ascending distinct durations including 1, not {durations}")
    if inner.blank != joint.num_classes_with_blank - 1 - len(durations) or joint.num_extra_outputs != len(durations):
        raise ValueError("tdt_fused needs the blank as the last label logit, then one logit per duration")
    layers = list(joint.joint_net)
    if (not layers or type(layers[0]) is not torch.nn.ReLU or type(layers[-1]) is not torch.nn.Linear
            or any(type(m) is not torch.nn.Dropout for m in layers[1:-1])):
        raise ValueError(f"tdt_fused needs joint_net = ReLU, [Dropout], Linear, not {joint.joint_net}")
    if any(m.p > 0 for m in layers[1:-1]):
        raise ValueError("tdt_fused does not implement joint dropout p > 0")
    if joint.masking_prob > 0:
        raise ValueError("tdt_fused does not implement the joint's masking_prob")
    if joint.log_softmax:
        raise ValueError("tdt_fused needs raw joint logits (log_softmax None/False)")
    if joint.is_adapter_available():
        raise ValueError("tdt_fused does not implement joint adapters")
    return Spec(int(inner.blank), durations, float(inner.sigma), float(inner.omega))


# ----------------------------------------------------------------------------- loss

def joint_loss(joint, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor,
               encoder_lengths: torch.Tensor, transcripts: torch.Tensor,
               transcript_lengths: torch.Tensor) -> torch.Tensor:
    """What joint(encoder_outputs=..., decoder_outputs=..., ..., compute_wer=False)[0] returns
    (the reduced TDT loss), in one loss call: encoder outputs [B, D, T], prediction-network
    outputs [B, D, U+1], transcripts [B, >= U]."""
    _check_not_replaced(joint)
    spec = loss_spec(joint)
    f = joint.project_encoder(encoder_outputs.transpose(1, 2))
    g = joint.project_prednet(decoder_outputs.transpose(1, 2))
    out = joint.joint_net[-1]
    losses = tdt_losses(f, g, out.weight, out.bias, transcripts, encoder_lengths, transcript_lengths, spec)
    return joint.loss.reduce(losses, transcript_lengths.long())


def tdt_losses(f: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
               labels: torch.Tensor, f_len: torch.Tensor, label_len: torch.Tensor, spec: Spec) -> torch.Tensor:
    """Per-row TDT losses [B] (FP32) of the joint Linear(relu(f[:, :, None] + g[:, None])) for
    projected encoder f [B, T, H] and prediction network g [B, U+1, H]. Under CUDA autocast the
    joint runs in the autocast dtype, as autocast runs NeMo's joint."""
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else weight.dtype
    rnnt = random.uniform(0, 1) < spec.omega  # NeMo's gpu_rnnt.py: one draw per loss call
    return _FusedTDT.apply(f, g, weight, bias, labels.long(), f_len.long(), label_len.long(), spec, dtype, rnnt)


def plan_chunks(lens_t: list[int], lens_u1: list[int], budget: int) -> list[tuple[int, int, int, int, int]]:
    """(b0, b1, t0, t1, u1) chunks over rows sorted by length: consecutive rows while their padded
    cells fit the budget, a row larger than the budget split along T. A chunk holds at least one
    frame of one row, so its size is at most max(budget, U+1) cells."""
    chunks, b, rows = [], 0, len(lens_t)
    while b < rows:
        t, u = lens_t[b], lens_u1[b]
        if t * u > budget:
            step = max(1, budget // u)
            chunks += [(b, b + 1, t0, min(t, t0 + step), u) for t0 in range(0, t, step)]
            b += 1
            continue
        e = b + 1
        while e < rows and (e + 1 - b) * max(t, lens_t[e]) * max(u, lens_u1[e]) <= budget:
            t, u, e = max(t, lens_t[e]), max(u, lens_u1[e]), e + 1
        chunks.append((b, e, 0, t, u))
        b = e
    return chunks


def _hidden(f: torch.Tensor, g: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """NeMo's joint_after_projection up to the output layer: relu(f + g), then autocast's cast."""
    return torch.relu(f[:, :, None, :] + g[:, None, :, :]).to(dtype)


def _pow2(n: int) -> int:
    return triton.next_power_of_2(max(1, n))


class _FusedTDT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, g, weight, bias, labels, f_len, label_len, spec: Spec, dtype, rnnt: bool):
        with torch.autocast("cuda", enabled=False):
            lens_t, lens_u = f_len.tolist(), label_len.tolist()
            rows, t_max, u1_max = len(lens_t), max(lens_t), max(lens_u) + 1
            if min(lens_t) < 1 or min(lens_u) < 0 or t_max > f.shape[1] or u1_max > g.shape[1] \
                    or labels.shape[0] != rows or (u1_max > 1 and labels.shape[1] < u1_max - 1):
                raise ValueError(f"tdt_fused: lengths T {lens_t} / U {lens_u} do not fit f {tuple(f.shape)}, "
                                 f"g {tuple(g.shape)}, labels {tuple(labels.shape)}")
            order = sorted(range(rows), key=lambda i: (lens_u[i], lens_t[i]), reverse=True)
            perm = torch.tensor(order, device=f.device)
            s_t, s_u1 = [lens_t[i] for i in order], [lens_u[i] + 1 for i in order]
            f_s, g_s = f[perm, :t_max], g[perm, :u1_max]
            labels_s = labels[perm, :max(1, u1_max - 1)].contiguous()
            t_s = torch.tensor(s_t, device=f.device, dtype=torch.int32)
            u1_s = torch.tensor(s_u1, device=f.device, dtype=torch.int32)
            durations = torch.tensor(spec.durations, device=f.device, dtype=torch.int32)
            plan = plan_chunks(s_t, s_u1, CHUNK_CELLS)
            n_dur, vocab = len(spec.durations), weight.shape[0]
            stats = torch.zeros(rows, t_max, u1_max, STATS + n_dur, device=f.device)
            w, b = weight.to(dtype), bias.to(dtype)
            for b0, b1, t0, t1, u1 in plan:
                z = F.linear(_hidden(f_s[b0:b1, t0:t1], g_s[b0:b1, :u1], dtype), w, b)
                cells = (b1 - b0) * (t1 - t0) * u1
                block_v = _pow2(vocab)
                block_c = max(1, min(64, 4096 // block_v))
                _stats_kernel[(triton.cdiv(cells, block_c),)](
                    z, labels_s, t_s, u1_s, stats, b0, t0, t1 - t0, u1, cells, labels_s.stride(0),
                    stats.stride(0), stats.stride(1), stats.stride(2), vocab, spec.blank,
                    N_DUR=n_dur, BLOCK_C=block_c, BLOCK_V=block_v, BLOCK_D=_pow2(n_dur))
                del z
            alpha = torch.empty(rows, t_max, u1_max, device=f.device)
            ll = torch.empty(rows, device=f.device)
            block_u = _pow2(u1_max)
            _alpha_kernel[(rows,)](stats, alpha, ll, t_s, u1_s, durations, spec.sigma, stats.stride(0),
                                   stats.stride(1), stats.stride(2), alpha.stride(0), alpha.stride(1),
                                   N_DUR=n_dur, RNNT=rnnt, BLOCK_U=block_u, num_warps=_warps(block_u))
            losses = torch.empty_like(ll)
            losses[perm] = -ll
        ctx.save_for_backward(f_s, g_s, weight, bias, labels_s, t_s, u1_s, durations, perm, stats, alpha, ll)
        ctx.plan, ctx.spec, ctx.dtype, ctx.rnnt = plan, spec, dtype, rnnt
        ctx.f_meta, ctx.g_meta = (f.shape, f.dtype), (g.shape, g.dtype)
        return losses

    @staticmethod
    def backward(ctx, grad_losses):
        f_s, g_s, weight, bias, labels_s, t_s, u1_s, durations, perm, stats, alpha, ll = ctx.saved_tensors
        spec, dtype = ctx.spec, ctx.dtype
        need_f, need_g, need_w, need_b = ctx.needs_input_grad[:4]
        with torch.autocast("cuda", enabled=False):
            rows, t_max, u1_max, width = stats.shape
            n_dur, vocab, hidden = len(spec.durations), weight.shape[0], weight.shape[1]
            block_u = _pow2(u1_max)
            beta = torch.empty_like(alpha)
            _beta_kernel[(rows,)](stats, beta, t_s, u1_s, durations, spec.sigma, stats.stride(0),
                                  stats.stride(1), stats.stride(2), beta.stride(0), beta.stride(1),
                                  N_DUR=n_dur, RNNT=ctx.rnnt, BLOCK_U=block_u, num_warps=_warps(block_u))
            coef = torch.empty_like(stats)
            cells = rows * t_max * u1_max
            _coef_kernel[(triton.cdiv(cells, 128),)](
                stats, alpha, beta, ll, coef, t_s, u1_s, durations, spec.sigma, cells, t_max, u1_max,
                stats.stride(0), stats.stride(1), stats.stride(2), alpha.stride(0), alpha.stride(1),
                N_DUR=n_dur, RNNT=ctx.rnnt, BLOCK=128, BLOCK_D=_pow2(n_dur))
            del beta
            scale = grad_losses.float()[perm].contiguous()
            w, b = weight.to(dtype), bias.to(dtype)
            d_w = torch.zeros(vocab, hidden, device=weight.device) if need_w else None
            d_b = torch.zeros(vocab, device=weight.device) if need_b else None
            d_f = torch.zeros(rows, t_max, hidden, device=weight.device) if need_f else None
            d_g = torch.zeros(rows, u1_max, hidden, device=weight.device) if need_g else None
            for b0, b1, t0, t1, u1 in ctx.plan:
                h = _hidden(f_s[b0:b1, t0:t1], g_s[b0:b1, :u1], dtype)
                dz = F.linear(h, w, b)
                n = (b1 - b0) * (t1 - t0) * u1
                block_v = _pow2(vocab)
                block_c = max(1, min(64, 4096 // block_v))
                _dz_kernel[(triton.cdiv(n, block_c),)](
                    dz, labels_s, t_s, u1_s, coef, scale, b0, t0, t1 - t0, u1, n, labels_s.stride(0),
                    coef.stride(0), coef.stride(1), coef.stride(2), vocab, spec.blank,
                    N_DUR=n_dur, BLOCK_C=block_c, BLOCK_V=block_v)
                dz2, h2 = dz.view(n, vocab), h.view(n, hidden)
                if need_w:
                    d_w += _mm(dz2.t(), h2)
                if need_b:
                    d_b += dz2.sum(0, dtype=torch.float32)
                if need_f or need_g:
                    dh = _mm(dz2, w).view(h.shape).masked_fill_(h <= 0, 0.0)
                    if need_f:
                        d_f[b0:b1, t0:t1] += dh.sum(2)
                    if need_g:
                        d_g[b0:b1, :u1] += dh.sum(1)
                del h, dz, dz2, h2
            inv = torch.argsort(perm)
            grads = [None] * 4
            for i, (part, (shape, dt)) in enumerate(((d_f, ctx.f_meta), (d_g, ctx.g_meta))):
                if part is not None:
                    full = torch.zeros(shape, dtype=dt, device=part.device)
                    full[:, :part.shape[1]] = part[inv]
                    grads[i] = full
            grads[2] = None if d_w is None else d_w.to(weight.dtype)
            grads[3] = None if d_b is None else d_b.to(bias.dtype)
        return (*grads, None, None, None, None, None, None)


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b accumulated and returned in FP32."""
    return a @ b if a.dtype == torch.float32 else torch.mm(a, b, out_dtype=torch.float32)


def _warps(block_u: int) -> int:
    return min(8, max(1, block_u // 32))


# ----------------------------------------------------------------------------- Triton kernels
# Per-cell layout (stats): [lse, z_blank - lse, z_target - lse, log_softmax(z_durations)...];
# (coef): [alpha + beta - ll - lse, blank occupancy, target occupancy, duration-logit gradients...].
# Arithmetic in the recursions follows NeMo's kernels term by term (gpu_rnnt_kernel.py).

@triton.jit
def _lse(a, b):
    """rnnt_helper.log_sum_exp: max + log1p(exp(min - max)), -inf if both are -inf."""
    mx = tl.maximum(a, b)
    mn = tl.minimum(a, b)
    return tl.where(mx == float("-inf"), mx, mx + libdevice.log1p(libdevice.exp(mn - mx)))


@triton.jit(do_not_specialize=["b0", "t0", "tc", "uc", "n_cells", "lab_b", "s_b", "s_t", "s_u"])
def _stats_kernel(z_ptr, lab_ptr, t_ptr, u1_ptr, stats_ptr, b0, t0, tc, uc, n_cells, lab_b, s_b, s_t, s_u,
                  vocab, blank, N_DUR: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_V: tl.constexpr,
                  BLOCK_D: tl.constexpr):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    in_chunk = c < n_cells
    b = b0 + c // (tc * uc)
    t = t0 + (c // uc) % tc
    u = c % uc
    big_t = tl.load(t_ptr + b, mask=in_chunk, other=0)
    big_u = tl.load(u1_ptr + b, mask=in_chunk, other=0)
    valid = in_chunk & (t < big_t) & (u < big_u)
    has_target = valid & (u < big_u - 1)
    row = z_ptr + c.to(tl.int64) * vocab
    v = tl.arange(0, BLOCK_V)
    z = tl.load(row[:, None] + v[None, :], mask=valid[:, None] & (v[None, :] <= blank),
                other=float("-inf")).to(tl.float32)
    m = tl.max(z, axis=1)
    m = tl.where(m == float("-inf"), 0.0, m)
    lse = m + tl.log(tl.sum(libdevice.exp(z - m[:, None]), axis=1))
    z_blank = tl.load(row + blank, mask=valid, other=0.0).to(tl.float32)
    target = tl.load(lab_ptr + b * lab_b + u, mask=has_target, other=0)
    z_target = tl.load(row + target, mask=has_target, other=0.0).to(tl.float32)
    d = tl.arange(0, BLOCK_D)
    zd = tl.load(row[:, None] + blank + 1 + d[None, :], mask=valid[:, None] & (d[None, :] < N_DUR),
                 other=float("-inf")).to(tl.float32)
    md = tl.max(zd, axis=1)
    md = tl.where(md == float("-inf"), 0.0, md)
    lsed = md + tl.log(tl.sum(libdevice.exp(zd - md[:, None]), axis=1))
    out = stats_ptr + b * s_b + t * s_t + u * s_u
    tl.store(out, lse, mask=valid)
    tl.store(out + 1, z_blank - lse, mask=valid)
    tl.store(out + 2, tl.where(has_target, z_target - lse, 0.0), mask=valid)
    tl.store(out[:, None] + 3 + d[None, :], zd - lsed[:, None], mask=valid[:, None] & (d[None, :] < N_DUR))


@triton.jit(do_not_specialize=["sigma", "s_b", "s_t", "s_u", "a_b", "a_t"])
def _alpha_kernel(stats_ptr, alpha_ptr, ll_ptr, t_ptr, u1_ptr, dur_ptr, sigma, s_b, s_t, s_u, a_b, a_t,
                  N_DUR: tl.constexpr, RNNT: tl.constexpr, BLOCK_U: tl.constexpr):
    """One program per row; lane u walks the anti-diagonals t + u = n (NeMo: one thread per u,
    syncthreads between diagonals). compute_tdt_alphas_kernel, or compute_alphas_kernel (RNNT)."""
    row = tl.program_id(0)
    big_t = tl.load(t_ptr + row)
    big_u = tl.load(u1_ptr + row)
    st = stats_ptr + row * s_b
    al = alpha_ptr + row * a_b
    u = tl.arange(0, BLOCK_U)
    tl.store(al + u, tl.zeros([BLOCK_U], tl.float32), mask=u == 0)
    tl.debug_barrier()
    for n in range(1, big_t + big_u - 1):
        t = n - u
        valid = (u < big_u) & (t >= 0) & (t < big_t)
        no_emit = tl.full([BLOCK_U], float("-inf"), tl.float32)
        emit = tl.full([BLOCK_U], float("-inf"), tl.float32)
        if RNNT:
            ok = valid & (t >= 1)
            prev = tl.load(al + (t - 1) * a_t + u, mask=ok, other=float("-inf"), cache_modifier=".cg")
            lp = tl.load(st + (t - 1) * s_t + u * s_u + 1, mask=ok, other=0.0)
            no_emit = tl.where(ok, prev + lp, no_emit)
            ok = valid & (u >= 1)
            prev = tl.load(al + t * a_t + u - 1, mask=ok, other=float("-inf"), cache_modifier=".cg")
            lp = tl.load(st + t * s_t + (u - 1) * s_u + 2, mask=ok, other=0.0)
            emit = tl.where(ok, prev + lp, emit)
        else:
            for i in tl.static_range(N_DUR):
                d = tl.load(dur_ptr + i)
                ok = valid & (d > 0) & (t >= d)
                prev = tl.load(al + (t - d) * a_t + u, mask=ok, other=float("-inf"), cache_modifier=".cg")
                cell = st + (t - d) * s_t + u * s_u
                lp = tl.load(cell + 1, mask=ok, other=0.0)
                lpd = tl.load(cell + 3 + i, mask=ok, other=0.0)
                no_emit = tl.where(ok, _lse(no_emit, prev + lp - sigma + lpd), no_emit)
            for i in tl.static_range(N_DUR):
                d = tl.load(dur_ptr + i)
                ok = valid & (u >= 1) & (t >= d)
                prev = tl.load(al + (t - d) * a_t + u - 1, mask=ok, other=float("-inf"), cache_modifier=".cg")
                cell = st + (t - d) * s_t + (u - 1) * s_u
                lp = tl.load(cell + 2, mask=ok, other=0.0)
                lpd = tl.load(cell + 3 + i, mask=ok, other=0.0)
                emit = tl.where(ok, _lse(emit, prev + lp - sigma + lpd), emit)
        tl.store(al + t * a_t + u, _lse(emit, no_emit), mask=valid)
        tl.debug_barrier()
    last = st + (big_u - 1) * s_u
    if RNNT:
        ll = tl.load(al + (big_t - 1) * a_t + big_u - 1, cache_modifier=".cg") + \
            tl.load(last + (big_t - 1) * s_t + 1)
    else:
        ll = float("-inf")
        for i in tl.static_range(N_DUR):
            d = tl.load(dur_ptr + i)
            ok = (d > 0) & (big_t >= d)
            prev = tl.load(al + (big_t - d) * a_t + big_u - 1, mask=ok, other=float("-inf"), cache_modifier=".cg")
            lp = tl.load(last + (big_t - d) * s_t + 1, mask=ok, other=0.0)
            lpd = tl.load(last + (big_t - d) * s_t + 3 + i, mask=ok, other=0.0)
            ll = tl.where(ok, _lse(ll, prev + lp - sigma + lpd), ll)
    tl.store(ll_ptr + row, ll)


@triton.jit(do_not_specialize=["sigma", "s_b", "s_t", "s_u", "a_b", "a_t"])
def _beta_kernel(stats_ptr, beta_ptr, t_ptr, u1_ptr, dur_ptr, sigma, s_b, s_t, s_u, a_b, a_t,
                 N_DUR: tl.constexpr, RNNT: tl.constexpr, BLOCK_U: tl.constexpr):
    """compute_tdt_betas_kernel, or compute_betas_kernel (RNNT); same lane layout as _alpha_kernel."""
    row = tl.program_id(0)
    big_t = tl.load(t_ptr + row)
    big_u = tl.load(u1_ptr + row)
    st = stats_ptr + row * s_b
    be = beta_ptr + row * a_b
    u = tl.arange(0, BLOCK_U)
    corner = st + (big_t - 1) * s_t + (big_u - 1) * s_u
    init = tl.load(corner + 1)
    if not RNNT:
        one = 0
        for i in tl.static_range(N_DUR):
            one = tl.where(tl.load(dur_ptr + i) == 1, i, one)
        init = init - sigma + tl.load(corner + 3 + one)
    tl.store(be + (big_t - 1) * a_t + u, tl.zeros([BLOCK_U], tl.float32) + init, mask=u == big_u - 1)
    tl.debug_barrier()
    for n in range(big_t + big_u - 2, -1, -1):
        t = n - u
        valid = (u < big_u) & (t >= 0) & (t < big_t) & ((t < big_t - 1) | (u < big_u - 1))
        here = st + t * s_t + u * s_u
        no_emit = tl.full([BLOCK_U], float("-inf"), tl.float32)
        emit = tl.full([BLOCK_U], float("-inf"), tl.float32)
        if RNNT:
            ok = valid & (t + 1 < big_t)
            nxt = tl.load(be + (t + 1) * a_t + u, mask=ok, other=float("-inf"), cache_modifier=".cg")
            no_emit = tl.where(ok, nxt + tl.load(here + 1, mask=ok, other=0.0), no_emit)
            ok = valid & (u < big_u - 1)
            nxt = tl.load(be + t * a_t + u + 1, mask=ok, other=float("-inf"), cache_modifier=".cg")
            emit = tl.where(ok, nxt + tl.load(here + 2, mask=ok, other=0.0), emit)
        else:
            lp_blank = tl.load(here + 1, mask=valid, other=0.0)
            lp_target = tl.load(here + 2, mask=valid, other=0.0)
            for i in tl.static_range(N_DUR):
                d = tl.load(dur_ptr + i)
                lpd = tl.load(here + 3 + i, mask=valid, other=0.0)
                ok = valid & (d > 0) & (t + d < big_t)
                nxt = tl.load(be + (t + d) * a_t + u, mask=ok, other=float("-inf"), cache_modifier=".cg")
                no_emit = tl.where(ok, _lse(no_emit, nxt + lp_blank + lpd - sigma), no_emit)
                ok = valid & (d > 0) & (t + d == big_t) & (u == big_u - 1)
                no_emit = tl.where(ok, _lse(no_emit, lp_blank + lpd - sigma), no_emit)
            for i in tl.static_range(N_DUR):
                d = tl.load(dur_ptr + i)
                lpd = tl.load(here + 3 + i, mask=valid, other=0.0)
                ok = valid & (u < big_u - 1) & (t + d < big_t)
                nxt = tl.load(be + (t + d) * a_t + u + 1, mask=ok, other=float("-inf"), cache_modifier=".cg")
                emit = tl.where(ok, _lse(emit, nxt + lp_target + lpd - sigma), emit)
        tl.store(be + t * a_t + u, _lse(emit, no_emit), mask=valid)
        tl.debug_barrier()


@triton.jit(do_not_specialize=["sigma", "n_cells", "t_max", "u_max", "s_b", "s_t", "s_u", "a_b", "a_t"])
def _coef_kernel(stats_ptr, alpha_ptr, beta_ptr, ll_ptr, coef_ptr, t_ptr, u1_ptr, dur_ptr, sigma, n_cells,
                 t_max, u_max, s_b, s_t, s_u, a_b, a_t, N_DUR: tl.constexpr, RNNT: tl.constexpr,
                 BLOCK: tl.constexpr, BLOCK_D: tl.constexpr):
    """Per cell: log(occupancy) - lse (so occupancy * softmax_v = exp(z_v + coef[0])), the blank
    and target arc occupancies, and the gradient w.r.t. the duration logits (the arcs' duration
    occupancies through the duration log-softmax). Zero (coef[0] = -inf) outside the lattice."""
    c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_grid = c < n_cells
    b = c // (t_max * u_max)
    t = (c // u_max) % t_max
    u = c % u_max
    big_t = tl.load(t_ptr + b, mask=in_grid, other=0)
    big_u = tl.load(u1_ptr + b, mask=in_grid, other=0)
    valid = in_grid & (t < big_t) & (u < big_u)
    here = stats_ptr + b * s_b + t * s_t + u * s_u
    pos = b * a_b + t * a_t + u
    alpha = tl.load(alpha_ptr + pos, mask=valid, other=float("-inf"))
    beta = tl.load(beta_ptr + pos, mask=valid, other=float("-inf"))
    ll = tl.load(ll_ptr + b, mask=in_grid, other=0.0)
    lse = tl.load(here, mask=valid, other=0.0)
    lp_blank = tl.load(here + 1, mask=valid, other=0.0)
    lp_target = tl.load(here + 2, mask=valid, other=0.0)
    d_idx = tl.arange(0, BLOCK_D)
    in_d = d_idx < N_DUR
    out = coef_ptr + b * s_b + t * s_t + u * s_u
    if RNNT:
        ok = valid & (t < big_t - 1)
        nxt = tl.load(beta_ptr + pos + a_t, mask=ok, other=0.0)
        occ_blank = tl.where(ok, libdevice.exp(alpha + lp_blank + nxt - ll), 0.0)
        ok = valid & (t == big_t - 1) & (u == big_u - 1)
        occ_blank += tl.where(ok, libdevice.exp(alpha + lp_blank - ll), 0.0)
        ok = valid & (u < big_u - 1)
        nxt = tl.load(beta_ptr + pos + 1, mask=ok, other=0.0)
        occ_target = tl.where(ok, libdevice.exp(alpha + lp_target + nxt - ll), 0.0)
        d_logits = tl.zeros([BLOCK, BLOCK_D], tl.float32)
    else:
        d = tl.load(dur_ptr + d_idx, mask=in_d, other=0)
        lpd = tl.load(here[:, None] + 3 + d_idx[None, :], mask=valid[:, None] & in_d[None, :], other=0.0)
        a2, ll2, td = alpha[:, None], ll[:, None], t[:, None] + d[None, :]
        cell_ok = valid[:, None] & in_d[None, :]
        ok = cell_ok & (d[None, :] > 0) & (td < big_t[:, None])
        nxt = tl.load(beta_ptr + pos[:, None] + d[None, :] * a_t, mask=ok, other=0.0)
        arc_blank = tl.where(ok, libdevice.exp(a2 + lp_blank[:, None] - sigma - ll2 + nxt + lpd), 0.0)
        ok = cell_ok & (d[None, :] > 0) & (td == big_t[:, None]) & (u[:, None] == big_u[:, None] - 1)
        arc_blank += tl.where(ok, libdevice.exp(a2 + lp_blank[:, None] - sigma - ll2 + lpd), 0.0)
        ok = cell_ok & (u[:, None] < big_u[:, None] - 1) & (td < big_t[:, None])
        nxt = tl.load(beta_ptr + pos[:, None] + d[None, :] * a_t + 1, mask=ok, other=0.0)
        arc_target = tl.where(ok, libdevice.exp(a2 + lp_target[:, None] - sigma + lpd - ll2 + nxt), 0.0)
        occ_blank = tl.sum(arc_blank, axis=1)
        occ_target = tl.sum(arc_target, axis=1)
        d_logprob = -(arc_blank + arc_target)  # gradient w.r.t. the duration log-probabilities
        d_logits = d_logprob - libdevice.exp(lpd) * tl.sum(d_logprob, axis=1)[:, None]
        d_logits = tl.where(cell_ok, d_logits, 0.0)
    log_flow = tl.where(valid, alpha + beta - ll - lse, float("-inf"))
    tl.store(out, log_flow, mask=in_grid)
    tl.store(out + 1, tl.where(valid, occ_blank, 0.0), mask=in_grid)
    tl.store(out + 2, tl.where(valid, occ_target, 0.0), mask=in_grid)
    tl.store(out[:, None] + 3 + d_idx[None, :], d_logits, mask=in_grid[:, None] & in_d[None, :])


@triton.jit(do_not_specialize=["b0", "t0", "tc", "uc", "n_cells", "lab_b", "s_b", "s_t", "s_u"])
def _dz_kernel(z_ptr, lab_ptr, t_ptr, u1_ptr, coef_ptr, scale_ptr, b0, t0, tc, uc, n_cells, lab_b, s_b, s_t,
               s_u, vocab, blank, N_DUR: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_V: tl.constexpr):
    """Overwrite a chunk's logits z with dLoss/dz: for labels and blank occupancy * softmax - the
    occupancy of the arcs emitting that symbol (NeMo's compute_tdt_grad_kernel / compute_grad_kernel),
    for durations coef's duration-logit gradient; times the row's incoming gradient."""
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    in_chunk = c < n_cells
    b = b0 + c // (tc * uc)
    t = t0 + (c // uc) % tc
    u = c % uc
    big_t = tl.load(t_ptr + b, mask=in_chunk, other=0)
    big_u = tl.load(u1_ptr + b, mask=in_chunk, other=0)
    valid = in_chunk & (t < big_t) & (u < big_u)
    here = coef_ptr + b * s_b + t * s_t + u * s_u
    log_flow = tl.load(here, mask=valid, other=float("-inf"))
    occ_blank = tl.load(here + 1, mask=valid, other=0.0)
    occ_target = tl.load(here + 2, mask=valid, other=0.0)
    target = tl.load(lab_ptr + b * lab_b + u, mask=valid & (u < big_u - 1), other=-1)
    scale = tl.load(scale_ptr + b, mask=in_chunk, other=0.0)
    row = z_ptr + c.to(tl.int64) * vocab
    v = tl.arange(0, BLOCK_V)
    in_row = in_chunk[:, None] & (v[None, :] < vocab)
    z = tl.load(row[:, None] + v[None, :], mask=in_row, other=0.0)
    g = libdevice.exp(z.to(tl.float32) + log_flow[:, None])
    g -= tl.where(v[None, :] == blank, occ_blank[:, None], 0.0)
    g -= tl.where(v[None, :] == target[:, None], occ_target[:, None], 0.0)
    is_dur = (v[None, :] > blank) & (v[None, :] < vocab)
    g_dur = tl.load(here[:, None] + 3 + (v[None, :] - blank - 1), mask=valid[:, None] & is_dur, other=0.0)
    g = tl.where(is_dur, g_dur, g)
    g = tl.where(valid[:, None], g * scale[:, None], 0.0)
    tl.store(row[:, None] + v[None, :], g.to(z.dtype), mask=in_row)
