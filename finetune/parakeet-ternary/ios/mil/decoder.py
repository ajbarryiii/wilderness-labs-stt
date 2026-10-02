"""TDT prediction and joint networks as FP16 Core ML programs, built from a benchmark model's weights.

Contracts (mil/contract.json; the first two equal C0's, so the Swift harness drives them like C0):
- Decoder: targets int32 [1, 1], target_length int32 [1] (unused, kept for C0's contract), h_in / c_in
  fp32 [2, 1, 640] -> decoder fp32 [1, 640, 1] (LSTM output, before joint.pred), h_out / c_out fp32 [2, 1, 640].
  Embedding gather (row 1024 = blank/SOS is the zero padding row), then two LSTM cells written out as
  linear + sigmoid/tanh (PyTorch gate order i, f, g, o; b_ih + b_hh summed in FP32 before the cast).
- JointDecision: encoder_step fp32 [1, 1024, 1], decoder_step fp32 [1, 640, 1] -> token_id int32 [1, 1, 1]
  (argmax over the 1,025 token + blank logits), token_prob fp32 [1, 1, 1] (softmax over the same 1,025,
  at the argmax), duration int32 [1, 1, 1] (argmax bin of the 5 duration logits; value = durations[bin]).
  joint.enc and joint.pred projections, add, ReLU, joint_net linear 640 -> 1030 (as C0: per step).
- JointLogits (diagnostic): same inputs -> token_logits fp32 [1, 1025], duration_logits fp32 [1, 5] (raw).
- DecoderJoint (fused): targets int32 [1, 1], h_in, c_in fp32 [2, 1, 640], encoder_step fp32 [1, 1024, 1] ->
  token_id, token_prob, duration (as JointDecision), h_out, c_out: the prediction network on targets from
  (h_in, c_in), then the joint with its output. After a blank the caller keeps (targets, h_in, c_in) and
  only advances encoder_step; after an emission it passes the emitted token with the h_out/c_out of the
  call that consumed the previous token (see contract.json).
"""
from __future__ import annotations

import numpy as np

H = 640
LAYERS = 2
VOCAB_BLANK = 1025
DURATIONS = 5
ENC = 1024


def weights(source) -> dict:
    """FP16 arrays of the prediction and joint networks (FP32 sums of the two LSTM biases, then cast)."""
    f = source.floating
    W = {"embed": f("decoder.prediction.embed.weight").astype(np.float16)}
    if W["embed"].shape != (VOCAB_BLANK, H) or W["embed"][VOCAB_BLANK - 1].any():
        raise ValueError("embedding must be [1025, 640] with a zero blank row")
    for l in range(LAYERS):
        p = f"decoder.prediction.dec_rnn.lstm."
        W[f"w_ih{l}"] = f(p + f"weight_ih_l{l}").astype(np.float16)
        W[f"w_hh{l}"] = f(p + f"weight_hh_l{l}").astype(np.float16)
        W[f"b{l}"] = (f(p + f"bias_ih_l{l}") + f(p + f"bias_hh_l{l}")).astype(np.float16)
    for k, key in (("enc", "joint.enc"), ("pred", "joint.pred"), ("out", "joint.joint_net.2")):
        W[f"{k}_w"] = f(key + ".weight").astype(np.float16)
        W[f"{k}_b"] = f(key + ".bias").astype(np.float16)
    return W


def _decoder_body(W, targets, h_in, c_in):
    from coremltools.converters.mil import Builder as mb

    x = mb.gather(x=W["embed"], indices=mb.reshape(x=targets, shape=[1]), axis=0, name="embed")  # [1, 640]
    h16 = mb.cast(x=h_in, dtype="fp16", name="h_in_fp16")
    c16 = mb.cast(x=c_in, dtype="fp16", name="c_in_fp16")
    hs, cs = [], []
    for l in range(LAYERS):
        h = mb.squeeze(x=mb.slice_by_index(x=h16, begin=[l, 0, 0], end=[l + 1, 1, H]), axes=[0], name=f"h{l}")
        c = mb.squeeze(x=mb.slice_by_index(x=c16, begin=[l, 0, 0], end=[l + 1, 1, H]), axes=[0], name=f"c{l}")
        gates = mb.add(x=mb.linear(x=x, weight=W[f"w_ih{l}"], bias=W[f"b{l}"]),
                       y=mb.linear(x=h, weight=W[f"w_hh{l}"]), name=f"gates{l}")
        i, f, g, o = mb.split(x=gates, num_splits=4, axis=-1, name=f"gate_split{l}")
        c_new = mb.add(x=mb.mul(x=mb.sigmoid(x=f), y=c), y=mb.mul(x=mb.sigmoid(x=i), y=mb.tanh(x=g)), name=f"c_new{l}")
        h_new = mb.mul(x=mb.sigmoid(x=o), y=mb.tanh(x=c_new), name=f"h_new{l}")
        hs.append(h_new)
        cs.append(c_new)
        x = h_new
    h_out = mb.cast(x=mb.stack(values=hs, axis=0), dtype="fp32", name="h_out")
    c_out = mb.cast(x=mb.stack(values=cs, axis=0), dtype="fp32", name="c_out")
    return x, h_out, c_out  # x: fp16 [1, 640]


def _joint_logits(W, f16, g16):
    """f16 [1, 1024], g16 [1, 640] fp16 -> logits fp16 [1, 1030]."""
    from coremltools.converters.mil import Builder as mb

    f = mb.linear(x=f16, weight=W["enc_w"], bias=W["enc_b"], name="joint_enc")
    g = mb.linear(x=g16, weight=W["pred_w"], bias=W["pred_b"], name="joint_pred")
    h = mb.relu(x=mb.add(x=f, y=g), name="joint_hidden")
    return mb.linear(x=h, weight=W["out_w"], bias=W["out_b"], name="joint_logits")


def _decision(logits):
    from coremltools.converters.mil import Builder as mb

    tok = mb.slice_by_index(x=logits, begin=[0, 0], end=[1, VOCAB_BLANK], name="tok_logits")
    dur = mb.slice_by_index(x=logits, begin=[0, VOCAB_BLANK], end=[1, VOCAB_BLANK + DURATIONS], name="dur_logits")
    token_id = mb.reshape(x=mb.reduce_argmax(x=tok, axis=-1, keep_dims=True), shape=[1, 1, 1], name="token_id")
    prob = mb.reduce_max(x=mb.softmax(x=tok, axis=-1), axes=[-1], keep_dims=True)
    token_prob = mb.cast(x=mb.reshape(x=prob, shape=[1, 1, 1]), dtype="fp32", name="token_prob")
    duration = mb.reshape(x=mb.reduce_argmax(x=dur, axis=-1, keep_dims=True), shape=[1, 1, 1], name="duration")
    return token_id, token_prob, duration


def _step_inputs(enc_step, dec_step=None):
    from coremltools.converters.mil import Builder as mb

    f16 = mb.reshape(x=mb.cast(x=enc_step, dtype="fp16"), shape=[1, ENC], name="encoder_step_fp16")
    if dec_step is None:
        return f16
    g16 = mb.reshape(x=mb.cast(x=dec_step, dtype="fp16"), shape=[1, H], name="decoder_step_fp16")
    return f16, g16


def _specs(*shapes_types):
    from coremltools.converters.mil import Builder as mb

    return [mb.TensorSpec(shape=s, dtype=t) for s, t in shapes_types]


def decoder_program(W):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    @mb.program(input_specs=_specs(((1, 1), types.int32), ((1,), types.int32), ((LAYERS, 1, H), types.fp32),
                                   ((LAYERS, 1, H), types.fp32)), opset_version=ct.target.iOS26)
    def prog(targets, target_length, h_in, c_in):
        x, h_out, c_out = _decoder_body(W, targets, h_in, c_in)
        dec = mb.cast(x=mb.expand_dims(x=x, axes=[-1]), dtype="fp32", name="decoder")  # [1, 640, 1]
        return dec, h_out, c_out

    return prog


def joint_decision_program(W):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    @mb.program(input_specs=_specs(((1, ENC, 1), types.fp32), ((1, H, 1), types.fp32)), opset_version=ct.target.iOS26)
    def prog(encoder_step, decoder_step):
        f16, g16 = _step_inputs(encoder_step, decoder_step)
        return _decision(_joint_logits(W, f16, g16))

    return prog


def joint_logits_program(W):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    @mb.program(input_specs=_specs(((1, ENC, 1), types.fp32), ((1, H, 1), types.fp32)), opset_version=ct.target.iOS26)
    def prog(encoder_step, decoder_step):
        f16, g16 = _step_inputs(encoder_step, decoder_step)
        logits = _joint_logits(W, f16, g16)
        tok = mb.slice_by_index(x=logits, begin=[0, 0], end=[1, VOCAB_BLANK])
        dur = mb.slice_by_index(x=logits, begin=[0, VOCAB_BLANK], end=[1, VOCAB_BLANK + DURATIONS])
        return mb.cast(x=tok, dtype="fp32", name="token_logits"), mb.cast(x=dur, dtype="fp32", name="duration_logits")

    return prog


def decoder_joint_program(W):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    @mb.program(input_specs=_specs(((1, 1), types.int32), ((LAYERS, 1, H), types.fp32), ((LAYERS, 1, H), types.fp32),
                                   ((1, ENC, 1), types.fp32)), opset_version=ct.target.iOS26)
    def prog(targets, h_in, c_in, encoder_step):
        x, h_out, c_out = _decoder_body(W, targets, h_in, c_in)
        token_id, token_prob, duration = _decision(_joint_logits(W, _step_inputs(encoder_step), x))
        return token_id, token_prob, duration, h_out, c_out

    return prog


BUILDERS = {"Decoder": decoder_program, "JointDecision": joint_decision_program, "JointLogits": joint_logits_program,
            "DecoderJoint": decoder_joint_program}
