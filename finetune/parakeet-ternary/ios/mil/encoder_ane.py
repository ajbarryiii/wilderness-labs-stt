"""The "ANE" graph layout of the encoder (DESIGN.md "D. Graph layout"), same numerics as encoder.py's plain layout.

- Channels-first activations (1, C, 1, T) in the 24 layers; every linear is a 1x1 conv2d (rank-4 constexpr
  weights [out, in, 1, 1], the arm's encoding chain unchanged); LayerNorm over axis 1.
- Attention split per head: q, k, v are split along channels into 8 (1, 128, 1, T) tensors; per head
  ac = (q + u)^T k and bd = rel_shift((q + v)^T P_h) with matmuls on rank-4 transposes, softmax over the key
  axis, out_h = attn v^T; heads concatenated along channels. No rank-changing reshapes in the layers (the
  rel_shift reshapes keep rank 4; transposes only otherwise).
- Masking contract of mil/MASKING.md unchanged: the subsampling front end is the plain one; the attention mask
  (query and key) is (1, 1, T, T); padded frames are zeroed (select) before each depthwise conv, which is a
  (1, 9) conv2d over T with BatchNorm folded.
- Per-bucket folded linear_pos tables as in the plain layout, split per head ([1, 1, 128, 2T - 1] each).
- Rank changes only at the boundaries: subsampling output [1, 256, T, 16] -> (1, 4096, 1, T) (transpose +
  rank-4 reshape), and the final (1, 1024, 1, T) -> [1, 1024, T] squeeze.
Static shapes only (fixed and multifunction variants).
"""
from __future__ import annotations

import math

import numpy as np

from .encoder import (BUCKETS, D_K, D_MODEL, HEADS, KERNEL, LN_EPS, MEL, NEG, SUB_CH, _Dims, _halve_length,
                      _time_mask)


def _rel_shift4(x, t: int, name: str, zero):
    """rel_shift + [..., :T] on one head: x [1, 1, T, 2T - 1] -> [1, 1, T, T] (rank-4 reshapes only)."""
    from coremltools.converters.mil import Builder as mb

    x = mb.pad(x=x, pad=[0, 0, 0, 0, 0, 0, 1, 0], mode="constant", constant_val=zero, name=name + "_pad")
    x = mb.reshape(x=x, shape=[1, 1, 2 * t, t], name=name + "_view")
    x = mb.slice_by_index(x=x, begin=[0, 0, 1, 0], end=[1, 1, 2 * t, t], name=name + "_drop")
    x = mb.reshape(x=x, shape=[1, 1, t, 2 * t - 1], name=name + "_back")
    return mb.slice_by_index(x=x, begin=[0, 0, 0, 0], end=[1, 1, t, t], name=name)


def _layer(P, i: int, x, t: int, att_bad, pad_bad):
    from coremltools.converters.mil import Builder as mb

    npdt = P.np

    def ln(v, norm):
        return mb.layer_norm(x=v, axes=[1], gamma=P.dense(f"layers.{i}.{norm}.weight"),
                             beta=P.dense(f"layers.{i}.{norm}.bias"), epsilon=npdt(LN_EPS), name=f"l{i}_{norm}")

    def ff(v, which):
        h = P.project(v, i, f"feed_forward{which[-1]}.linear1")
        h = mb.silu(x=h, name=f"l{i}_{which}_silu")
        h = P.project(h, i, f"feed_forward{which[-1]}.linear2")
        return mb.mul(x=h, y=npdt(0.5), name=f"l{i}_{which}_half")

    res = mb.add(x=x, y=ff(ln(x, "norm_feed_forward1"), "ff1"), name=f"l{i}_res_ff1")
    # attention, split per head
    xa = ln(res, "norm_self_att")
    q = mb.split(x=P.project(xa, i, "self_attn.linear_q"), num_splits=HEADS, axis=1, name=f"l{i}_q_heads")
    k = mb.split(x=P.project(xa, i, "self_attn.linear_k"), num_splits=HEADS, axis=1, name=f"l{i}_k_heads")
    v = mb.split(x=P.project(xa, i, "self_attn.linear_v"), num_splits=HEADS, axis=1, name=f"l{i}_v_heads")
    u_all = P.dense(f"layers.{i}.self_attn.pos_bias_u").reshape(HEADS, D_K)
    w_all = P.dense(f"layers.{i}.self_attn.pos_bias_v").reshape(HEADS, D_K)
    table = P.pos_table(i, t)  # [1, 8, 128, 2T - 1]
    scale = npdt(1.0 / math.sqrt(D_K))
    outs = []
    for h in range(HEADS):
        n = f"l{i}_h{h}"
        qu = mb.transpose(x=mb.add(x=q[h], y=u_all[h].reshape(1, D_K, 1, 1)), perm=[0, 2, 3, 1], name=n + "_qu")
        qv = mb.transpose(x=mb.add(x=q[h], y=w_all[h].reshape(1, D_K, 1, 1)), perm=[0, 2, 3, 1], name=n + "_qv")
        kt = mb.transpose(x=k[h], perm=[0, 2, 1, 3], name=n + "_k")                        # (1, 1, 128, T)
        ac = mb.matmul(x=qu, y=kt, name=n + "_ac")                                           # (1, 1, T, T)
        pos = np.ascontiguousarray(table[:, h:h + 1])                                        # (1, 1, 128, 2T-1)
        bd = _rel_shift4(mb.matmul(x=qv, y=pos, name=n + "_bd_raw"), t, n + "_bd", npdt(0))
        sc = mb.mul(x=mb.add(x=ac, y=bd), y=scale, name=n + "_scores")
        sc = mb.select(cond=att_bad, a=npdt(NEG), b=sc, name=n + "_masked")
        attn = mb.select(cond=att_bad, a=npdt(0), b=mb.softmax(x=sc, axis=-1), name=n + "_attn")
        vt = mb.transpose(x=v[h], perm=[0, 2, 3, 1], name=n + "_v")                         # (1, 1, T, 128)
        o = mb.matmul(x=attn, y=vt, name=n + "_out")                                         # (1, 1, T, 128)
        outs.append(mb.transpose(x=o, perm=[0, 3, 1, 2], name=n + "_out_cf"))               # (1, 128, 1, T)
    att = mb.concat(values=outs, axis=1, name=f"l{i}_heads_cat")
    res = mb.add(x=res, y=P.project(att, i, "self_attn.linear_out"), name=f"l{i}_res_att")
    # convolution module
    hc = P.project(ln(res, "norm_conv"), i, "conv.pointwise_conv1")
    a, b = mb.split(x=hc, num_splits=2, axis=1, name=f"l{i}_glu_split")
    hc = mb.mul(x=a, y=mb.sigmoid(x=b), name=f"l{i}_glu")
    hc = mb.select(cond=pad_bad, a=npdt(0), b=hc, name=f"l{i}_conv_masked")
    hc = mb.pad(x=hc, pad=[0, 0, 0, 0, 0, 0, (KERNEL - 1) // 2, (KERNEL - 1) // 2], mode="constant",
                constant_val=npdt(0), name=f"l{i}_dw_pad")
    dw_w, dw_b = P.depthwise(i)
    hc = mb.conv(x=hc, weight=dw_w.reshape(D_MODEL, 1, 1, KERNEL), bias=dw_b, groups=D_MODEL, pad_type="valid",
                 name=f"l{i}_dw")
    hc = mb.silu(x=hc, name=f"l{i}_dw_silu")
    res = mb.add(x=res, y=P.project(hc, i, "conv.pointwise_conv2"), name=f"l{i}_res_conv")
    res = mb.add(x=res, y=ff(ln(res, "norm_feed_forward2"), "ff2"), name=f"l{i}_res_ff2")
    return ln(res, "norm_out")


def forward(P, mel, mel_length):
    from coremltools.converters.mil import Builder as mb

    d = _Dims(True)
    x = mb.cast(x=mel, dtype=P.dt, name="mel_" + P.dt)
    x = mb.expand_dims(x=mb.transpose(x=x, perm=[0, 2, 1]), axes=[1], name="mel_nchw")
    L = mel_length
    x = mb.mul(x=x, y=_time_mask(d, d.dim(x, 2), L, "mask0", P.dt), name="sub_in_masked")
    x = mb.conv(x=x, weight=P.weight("pre_encode.conv.0.weight"), bias=P.dense("pre_encode.conv.0.bias"),
                strides=[2, 2], pad_type="custom", pad=[1, 1, 1, 1], name="sub_conv0")
    x = mb.relu(x=x, name="sub_relu0")
    L = _halve_length(L, "len1")
    for stage, (dw, pw) in enumerate(((2, 3), (5, 6)), start=1):
        x = mb.mul(x=x, y=_time_mask(d, d.dim(x, 2), L, f"mask{stage}", P.dt), name=f"sub_masked{stage}")
        x = mb.conv(x=x, weight=P.weight(f"pre_encode.conv.{dw}.weight"), bias=P.dense(f"pre_encode.conv.{dw}.bias"),
                    strides=[2, 2], pad_type="custom", pad=[1, 1, 1, 1], groups=SUB_CH, name=f"sub_dw{stage}")
        x = mb.conv(x=x, weight=P.weight(f"pre_encode.conv.{pw}.weight"), bias=P.dense(f"pre_encode.conv.{pw}.bias"),
                    name=f"sub_pw{stage}")
        x = mb.relu(x=x, name=f"sub_relu{stage}")
        L = _halve_length(L, f"len{stage + 1}")
    x = mb.mul(x=x, y=_time_mask(d, d.dim(x, 2), L, "mask3", P.dt), name="sub_out_masked")  # [1, 256, T, 16]
    t = int(x.shape[2])
    x = mb.reshape(x=mb.transpose(x=x, perm=[0, 1, 3, 2]), shape=[1, SUB_CH * 16, 1, t], name="sub_cf")
    w_out = P.weight("pre_encode.out.weight")
    x = mb.conv(x=x, weight=w_out.reshape(D_MODEL, SUB_CH * 16, 1, 1), bias=P.dense("pre_encode.out.bias"),
                name="sub_out")                                                                   # (1, 1024, 1, T)
    length = mb.identity(x=L, name="encoder_length")
    valid = mb.less(x=np.arange(t, dtype=np.int32), y=L, name="valid")
    pair = mb.logical_and(x=mb.expand_dims(x=valid, axes=[1]), y=mb.expand_dims(x=valid, axes=[0]))
    att_bad = mb.expand_dims(x=mb.logical_not(x=pair), axes=[0, 1], name="att_mask")          # (1, 1, T, T)
    pad_bad = mb.expand_dims(x=mb.logical_not(x=valid), axes=[0, 1, 2], name="pad_mask")      # (1, 1, 1, T)
    for i in range(P.n_layers):
        x = _layer(P, i, x, t, att_bad, pad_bad)
    out = mb.cast(x=mb.squeeze(x=x, axes=[2]), dtype="fp32", name="encoder")
    return out, length


def function(P, mel_frames: int, opset):
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    if mel_frames is None:
        raise ValueError("the ANE layout is built for static shapes only (fixed, multifunction)")

    @mb.function(input_specs=[mb.TensorSpec(shape=(1, MEL, mel_frames), dtype=types.fp32),
                              mb.TensorSpec(shape=(1,), dtype=types.int32)], opset_version=opset)
    def encoder(mel, mel_length):
        return forward(P, mel, mel_length)

    return encoder
