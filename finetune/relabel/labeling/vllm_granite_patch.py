# The encoder functions below (attention, conv_module, block, encode and embed) are adapted from vLLM 0.32.0,
# vllm/model_executor/models/granite_speech.py (GraniteSpeechConformerAttention, GraniteSpeechConformerConvModule,
# GraniteSpeechConformerBlock, GraniteSpeechCTCEncoder and GraniteSpeechForConditionalGeneration._process_audio_input).
# Copyright contributors to the vLLM project. Licensed under the Apache License, Version 2.0
# (https://www.apache.org/licenses/LICENSE-2.0); this file's other code is under the repository's licence.
# Modified 2026-10-25: each clip in a batch is masked and zero-padded to its own length, the attention output projection
# gets a contiguous input, and the convolutions use shape-independent kernels (see the module docstring).
"""Per-clip Granite speech encoder for vLLM (labeling plan "Engine", 2026-10-25): a clip's audio embedding no longer
depends on the clips it shares a scheduler step with.

vLLM 0.32.0 encodes each scheduler step's audio items as one batch, zero-padded to the longest item's feature frames,
and its encoder only masks the padding of the batch's own last block. A shorter clip in that batch therefore differs
from the same clip encoded alone in three places: its last attention block attends to the padded frames after its end
(they are neither zero nor masked), the depthwise convolution reads those frames at the clip's right edge, and the
projector's last window pools them. Two more differences are numerical and appear even between clips of equal length
(measured on the GPU, 2026-10-25, with VLLM_BATCH_INVARIANT=1, which covers matrix products and softmax): the attention
output is sliced to the batch's frames before its output projection, and for more than one clip that slice is not
contiguous, so the projection runs as another matrix product than for one clip; and the convolution module's
convolutions run in cuDNN, whose algorithm choice varies with the batch shape (the pointwise ones differed for clips up
to ~8 s). Labels changed with the batch (labeling plan "GPU probe", 2026-10-25).

The patch keeps vLLM's layout ([batch, longest, dim], the same modules and weights) and gives every clip the inputs it
has when encoded alone (a batch of one, vLLM's and transformers' semantics for a single clip):
- attention: the block's input frames after the clip's end are zero, and a query or key at or after the clip's end is
  masked, block by block (alone, only the clip's own last block is partial: the same mask);
- convolution: the depthwise convolution's input frames after the clip's end are zero (alone: its zero padding);
- every layer's output, and the mid-layer CTC branch's sum, is zero after the clip's end (no padded values carry over);
- projector: the encoder output after the clip's end is zero (alone: the projector's zero padding of its last window);
and computes the shape-dependent ops with shape-independent kernels:
- the attention output projection gets a contiguous input;
- the pointwise (kernel 1) convolutions are matrix products on contiguous [batch, frames, channels] inputs (the GEMMs
  every other linear layer uses), and the depthwise convolution is PyTorch's native CUDA depthwise kernel (one thread per
  output, the taps summed in a fixed order in float32) instead of cuDNN.
The masking steps are no-ops for a batch of one; the convolution kernels are not the unpatched ones, so a clip alone
differs from the unpatched encoder by kernel rounding only (and is compared with transformers' encoder on the GPU).
Bitwise equality of a batched clip and the clip alone rests on those kernels being shape invariant; the GPU check
(alone vs batched) measures it.

Activation (labeling/engines.py VllmTeacher): vLLM loads the entry points of group vllm.general_plugins in every process
it starts, the spawned engine core and its worker included. labeling/vllm_env.sh installs a metadata-only distribution
in the isolated environment whose entry point is register() below. register() does nothing unless the environment
variable ENV holds this file's sha256 (VllmTeacher sets it before vLLM starts, and the engine core inherits it); with a
different value it raises, so the engine fails to start. When active it registers PerClipGraniteSpeech for T*'s
architecture and a worker method (STATUS_METHOD) that reports, from the process holding the model, this file's identity
and the loaded model's class. VllmTeacher refuses to run unless that report matches what its spec expects.

The patch identity (NAME, VERSION, sha256 of this file) is part of the vLLM engine spec (checks.vllm_engine), so it is
part of c2.contract: patched and unpatched outputs never share a contract.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

NAME = "granite-speech-per-clip-encoder"
VERSION = 1
ENV = "RELABEL_GRANITE_PATCH"            # the expected sha256 of this file
ARCH = "GraniteSpeechForConditionalGeneration"
STATUS_METHOD = "relabel_granite_patch_status"
VLLM_VERSION = "0.32.0"                  # the vLLM whose Granite speech code this mirrors
PLUGIN = "relabel_granite_encoder"       # the entry point's name (labeling/vllm_env.sh)
ENTRY_POINT = "labeling.vllm_granite_patch:register"

_state: dict = {"active": False, "pid": None, "identity": None}


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def identity() -> dict:
    return {"name": NAME, "version": VERSION, "sha256": source_sha256()}


def apply_env(expected: dict | None) -> None:
    """Request the patch (expected: its identity) or vLLM's own encoder (None) for this process and the engine core it
    spawns; before vLLM is imported."""
    if expected:
        os.environ[ENV] = expected["sha256"]
    else:
        os.environ.pop(ENV, None)


def check_status(expected: dict, reports: list[dict] | None, parent_pid: int) -> list[dict]:
    """Refuse an engine whose workers do not run the expected patch. reports: one STATUS_METHOD result per worker."""
    if not reports:
        raise RuntimeError("the vLLM Granite encoder patch is expected but no vLLM worker reports it (is the plugin "
                           "installed? run: labeling/vllm_env.sh plugin)")
    for r in reports:
        if not r.get("active"):
            raise RuntimeError(f"the vLLM Granite encoder patch is not active in worker pid {r.get('pid')}")
        if r.get("identity") != expected:
            raise RuntimeError(f"worker pid {r.get('pid')} runs patch {r.get('identity')}, the spec expects {expected}")
        if r.get("model_class") != f"{__name__}.PerClipGraniteSpeech":
            raise RuntimeError(f"worker pid {r.get('pid')} loaded {r.get('model_class')}, not the patched model")
        if r.get("pid") == parent_pid:
            raise RuntimeError("the patch status came from this process, not from a spawned engine core")
    return reports


# --- the per-clip encoder (vLLM 0.32.0 granite_speech.py, with the clip's own padding) ---------------------------------

def _tensor(y):
    """vLLM's parallel linear layers return (output, bias); plain modules return the output."""
    return y[0] if isinstance(y, tuple) else y


def feature_lengths(input_features) -> list[int]:
    """Each item's number of feature frames, from vLLM's batched input_features: a tensor [b, n, 160] or
    [b, 1, n, 160] (items of equal length) or a list of [n_i, 160] tensors."""
    if isinstance(input_features, (list, tuple)):
        if any(f.ndim != 2 for f in input_features):
            raise ValueError(f"expected 2-D feature items, got {[tuple(f.shape) for f in input_features]}")
        return [int(f.shape[0]) for f in input_features]
    shape = tuple(input_features.shape)
    if len(shape) == 4 and shape[1] == 1:
        shape = (shape[0],) + shape[2:]
    if len(shape) != 3:
        raise ValueError(f"unexpected input_features shape {tuple(input_features.shape)}")
    return [shape[1]] * shape[0]


def valid_frames(lengths: list[int], frames: int, device):
    """[b, frames] bool: frame t of item b is inside the clip."""
    import torch
    return torch.arange(frames, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]


def attention(attn, x, attention_dists, valid):
    import torch
    import torch.nn.functional as F
    h = attn.pre_norm(x)
    bsz, num_features, _ = h.shape
    ctx = attn.context_size
    num_blocks = math.ceil(num_features / ctx)
    pad = num_blocks * ctx - num_features
    if pad:
        h = F.pad(h, (0, 0, 0, pad))
        valid = F.pad(valid, (0, pad), value=False)
    h = h.masked_fill(~valid[..., None], 0)
    q = attn.to_q(h)
    k, v = attn.to_kv(h).chunk(2, dim=-1)
    q = q.reshape(bsz, num_blocks, ctx, attn.num_heads, -1).transpose(2, 3)
    k = k.reshape(bsz, num_blocks, ctx, attn.num_heads, -1).transpose(2, 3)
    v = v.reshape(bsz, num_blocks, ctx, attn.num_heads, -1).transpose(2, 3)
    rel_pos_emb = attn.rel_pos_emb(attention_dists.to(h.device))
    pos_attn = torch.einsum("bnhid,ijd->bnhij", q, rel_pos_emb) * attn.scale
    vb = valid.view(bsz, num_blocks, ctx)
    masked = ~(vb[..., :, None] & vb[..., None, :])                  # [b, n, i, j]
    pos_attn.masked_fill_(masked[:, :, None], -torch.finfo(pos_attn.dtype).max)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=pos_attn, scale=attn.scale)
    out = out.transpose(2, 3).reshape(bsz, h.shape[1], -1)
    # contiguous: for more than one clip the slice is not, and to_out would run as another GEMM (module docstring)
    return attn.to_out(out[:, :num_features, :].contiguous())


def check_conv(conv) -> None:
    """The convolution module conv_module computes: pointwise up/down convolutions, a GLU over channels and a bias-free
    depthwise convolution with stride and dilation 1."""
    up, down, dw = conv.up_conv, conv.down_conv, conv.depth_conv.conv
    ok = (up.kernel_size == (1,) and down.kernel_size == (1,) and up.groups == down.groups == 1
          and up.stride == down.stride == dw.stride == (1,) and up.padding == down.padding == dw.padding == (0,)
          and dw.groups == dw.in_channels == dw.out_channels and dw.bias is None and dw.dilation == (1,)
          and conv.glu.dim == 1)
    if not ok:
        raise RuntimeError(f"unexpected Granite convolution module {conv}")


def _depthwise(h, weight):
    """Depthwise convolution of [b, c, n] (already padded) with weight [c, 1, k]: on CUDA PyTorch's native depthwise
    kernel (shape invariant), elsewhere F.conv1d (CPU tests)."""
    import torch
    import torch.nn.functional as F
    if not h.is_cuda:
        return F.conv1d(h, weight, groups=weight.shape[0])
    k = weight.shape[-1]
    out = torch.ops.aten._conv_depthwise2d(h[:, :, None, :], weight[:, :, None, :], [1, k], None, [1, 1], [0, 0], [1, 1])
    return out[:, :, 0, :]


def conv_module(conv, x, valid):
    import torch.nn.functional as F
    h = conv.norm(x)
    h = F.linear(h, conv.up_conv.weight[..., 0], conv.up_conv.bias)     # [b, n, 2c]: the pointwise conv as a GEMM
    h = F.glu(h, dim=-1)                                               # GLU over channels
    h = h.masked_fill(~valid[..., None], 0)
    h = F.pad(h.transpose(1, 2).contiguous(), conv.depth_conv.padding)  # [b, c, n + k - 1]
    h = _depthwise(h, conv.depth_conv.conv.weight)
    h = conv.silu(conv.batch_norm(h))
    return F.linear(h.transpose(1, 2).contiguous(), conv.down_conv.weight[..., 0], conv.down_conv.bias)


def block(layer, x, attention_dists, valid):
    x = 0.5 * layer.ff1(x) + x
    x = attention(layer.attn, x, attention_dists, valid) + x
    x = conv_module(layer.conv, x, valid) + x
    x = 0.5 * layer.ff2(x) + x
    x = layer.post_norm(x)
    return x.masked_fill(~valid[..., None], 0)


def encode(encoder, features, lengths: list[int]):
    """GraniteSpeechCTCEncoder.forward on [b, longest, input_dim] features with each item's own length; frames after an
    item's end are zero in the output."""
    import torch.nn.functional as F
    valid = valid_frames(lengths, features.shape[1], features.device)
    h = encoder.input_linear(features).masked_fill(~valid[..., None], 0)
    for idx, layer in enumerate(encoder.layers, start=1):
        h = block(layer, h, encoder.attention_dists, valid)
        if idx == encoder.num_layers // 2:
            mid = _tensor(encoder.out(h.clone()))
            mid = encoder.softmax(mid) if hasattr(encoder, "softmax") else F.softmax(mid, dim=-1)
            h += _tensor(encoder.out_mid(mid))
            h = h.masked_fill(~valid[..., None], 0)
    return h


def projector_tokens(n_frames: int, window_size: int, downsample_rate: int) -> int:
    """The projector's output tokens for a clip of n_frames encoder frames (its windows, each window_size //
    downsample_rate queries); the feature extractor's _get_num_audio_features for the clip."""
    return math.ceil(n_frames / window_size) * (window_size // downsample_rate)


def embed(encoder, projector, features, lengths: list[int], embed_sizes: list[int]):
    """Per clip: its projector tokens (embed_sizes[b] of them, checked against its length), as in
    GraniteSpeechForConditionalGeneration._process_audio_input."""
    want = [projector_tokens(n, projector.window_size, projector.downsample_rate) for n in lengths]
    if list(want) != [int(s) for s in embed_sizes]:
        raise RuntimeError(f"audio token counts {list(embed_sizes)} do not match the feature lengths {lengths}")
    projected = projector(encode(encoder, features, lengths))
    mask = valid_frames(want, projected.shape[1], projected.device)
    import torch
    return torch.split(projected[mask], want)


# --- the vLLM plugin --------------------------------------------------------------------------------------------------

def _status(worker) -> dict:
    model = worker.get_model()
    model = getattr(model, "unwrap", lambda: model)()
    cls = type(model)
    return {"pid": os.getpid(), "active": _state["active"], "identity": _state["identity"],
            "model_class": f"{cls.__module__}.{cls.__qualname__}"}


def register() -> None:
    """Entry point (vllm.general_plugins): runs in every vLLM process; see the module docstring."""
    import logging
    want = os.environ.get(ENV)
    if not want:
        return
    have = source_sha256()
    if want != have:
        raise RuntimeError(f"{ENV}={want} but this patch is {have}: refusing to start vLLM with another patch")
    if _state["active"]:
        return
    import vllm
    if vllm.__version__ != VLLM_VERSION:
        raise RuntimeError(f"the Granite encoder patch mirrors vLLM {VLLM_VERSION}, not {vllm.__version__}")
    from vllm import ModelRegistry
    from vllm.model_executor.models import granite_speech as gs
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.v1.worker.worker_base import WorkerBase

    @MULTIMODAL_REGISTRY.register_processor(gs.GraniteSpeechMultiModalProcessor, info=gs.GraniteSpeechProcessingInfo,
                                            dummy_inputs=gs.GraniteSpeechDummyInputsBuilder)
    class PerClipGraniteSpeech(gs.GraniteSpeechForConditionalGeneration):
        def __init__(self, *, vllm_config, prefix: str = ""):    # vLLM passes these by keyword (its signature)
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            if type(self.encoder) is not gs.GraniteSpeechCTCEncoder:
                raise RuntimeError(f"unexpected Granite encoder {type(self.encoder)}")
            for layer in self.encoder.layers:
                check_conv(layer.conv)

        def embed_multimodal(self, **kwargs):
            features = kwargs.get("input_features")
            if features is None:
                return []
            lengths = feature_lengths(features)
            audio = self._parse_and_validate_audio_input(**kwargs)
            return embed(self.encoder, self.projector, audio["input_features"], lengths, audio["audio_embed_sizes"])

    PerClipGraniteSpeech.__module__, PerClipGraniteSpeech.__qualname__ = __name__, "PerClipGraniteSpeech"
    globals()["PerClipGraniteSpeech"] = PerClipGraniteSpeech
    ModelRegistry.register_model(ARCH, PerClipGraniteSpeech)
    setattr(WorkerBase, STATUS_METHOD, _status)
    _state.update(active=True, pid=os.getpid(), identity=identity())
    logging.getLogger("vllm").warning("relabel: Granite encoder patch %s v%d (%s) active in pid %d", NAME, VERSION,
                                      have[:12], os.getpid())
