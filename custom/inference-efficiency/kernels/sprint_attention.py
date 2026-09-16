"""One-query self-attention candidate for the short growing decoder cache.

One CTA owns a batch/head, performing QK, softmax, and PV without materializing
scores. It preserves FP16 Q/K/V and output with FP32 products/reductions. The
unnormalized softmax probabilities are rounded to FP16 before PV, following
the Tensor Core attention convention. Numerical error is checked independently
against the existing SDPA implementation; this is not a quantization change.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _self_attention(Q, K, V, Y, H: tl.constexpr, L, D: tl.constexpr,
                    QB: tl.constexpr, QH: tl.constexpr,
                    KB: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
                    VB: tl.constexpr, VH: tl.constexpr, VS: tl.constexpr,
                    SCALE: tl.constexpr, BS: tl.constexpr,
                    HALF_PROB: tl.constexpr):
    head = tl.program_id(0)
    batch, h = head // H, head % H
    s = tl.arange(0, BS)
    d = tl.arange(0, D)
    q = tl.load(Q + batch * QB + h * QH + d).to(tl.float32)
    k = tl.load(K + batch * KB + h * KH + s[:, None] * KS + d[None, :],
                s[:, None] < L, other=0).to(tl.float32)
    score = tl.sum(k * q[None, :], 1) * SCALE
    score = tl.where(s < L, score, -float("inf"))
    p = tl.exp(score - tl.max(score, 0))
    denominator = tl.sum(p, 0)
    if HALF_PROB:
        p = p.to(tl.float16).to(tl.float32)
    v = tl.load(V + batch * VB + h * VH + s[:, None] * VS + d[None, :],
                s[:, None] < L, other=0).to(tl.float32)
    out = tl.sum(v * p[:, None], 0) / denominator
    tl.store(Y + head * D + d, out.to(tl.float16))


def variants():
    return ["attention_self", "attention_self_fp32p", "attention_self_fusion"]


def _call(query, key, value, scale, half_prob):
    b, h, _, d = query.shape
    length = key.shape[-2]
    result = torch.empty((b, h, 1, d), dtype=query.dtype, device=query.device)
    _self_attention[(b * h,)](
        query, key, value, result, h, length, d, *query.stride()[:2],
        *key.stride()[:3], *value.stride()[:3],
        d ** -.5 if scale is None else scale, triton.next_power_of_2(length),
        half_prob, num_warps=4, enable_fp_fusion=False,
    )
    return result


def install(name, artifact_dir):
    if name not in variants():
        raise ValueError(name)
    restore_other = None
    if name == "attention_self_fusion":
        from . import sprint_qkv
        restore_other = sprint_qkv.install("qkv_cache_fusion", artifact_dir)
    original = F.scaled_dot_product_attention
    half_prob = name != "attention_self_fp32p"

    def attention(query, key, value, attn_mask=None, dropout_p=0.,
                  is_causal=False, *, scale=None, enable_gqa=False):
        if (query.ndim == 4 and query.shape[-2] == 1 and 0 < key.shape[-2] <= 129
                and query.shape[-1] in (16, 32, 64, 128)
                and query.dtype == key.dtype == value.dtype == torch.float16
                and query.is_cuda and key.device == query.device == value.device
                and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
                and key.shape == value.shape
                and key.shape[:2] == query.shape[:2]
                and key.shape[-1] == query.shape[-1]
                and attn_mask is None and dropout_p == 0 and not is_causal
                and not enable_gqa):
            return _call(query, key, value, scale, half_prob)
        return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                        is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)

    F.scaled_dot_product_attention = attention

    def restore():
        F.scaled_dot_product_attention = original
        if restore_other is not None:
            restore_other()

    return restore


def compile_only(artifact_dir):
    """AOT-compile representative short/long cache shapes, without GPU use."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    target = GPUTarget("cuda", 120, 32)
    signature = {"Q": "*fp16", "K": "*fp16", "V": "*fp16", "Y": "*fp16", "L": "i32"}
    folder = Path(artifact_dir)
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for length in (1, 32, 129):
        constexprs = dict(H=16, D=64, QB=1024, QH=64,
                         KB=132096, KH=8256, KS=64,
                         VB=132096, VH=8256, VS=64,
                         SCALE=0.125, BS=triton.next_power_of_2(length), HALF_PROB=True)
        kernel = triton.compile(ASTSource(_self_attention, signature, constexprs),
                                target=target,
                                options={"num_warps": 4, "enable_fp_fusion": False})
        (folder / f"self-attention-{length}.ptx").write_text(kernel.asm["ptx"])
        (folder / f"self-attention-{length}.cubin").write_bytes(kernel.asm["cubin"])
        results.append({"length": length, "metadata": kernel.metadata._asdict()})
    (folder / "compile.json").write_text(json.dumps(results, indent=2, default=str))
    return results


def check(name, artifact_dir):
    from runtime_model import ReplayWhisper, WhisperConfig
    torch.set_grad_enabled(False)
    torch.manual_seed(414)
    cases = []
    original = F.scaled_dot_product_attention
    half_prob = name != "attention_self_fp32p"
    for width, heads in ((16, 4), (64, 16)):
        for batch in (1, 4):
            qkv = torch.randn(batch, 1, 3 * heads * width,
                              device="cuda", dtype=torch.float16)
            q = qkv.chunk(3, dim=-1)[0].reshape(batch, 1, heads, width).transpose(1, 2)
            k = torch.randn(batch, heads, 129, width, device="cuda", dtype=torch.float16)
            v = torch.randn_like(k)
            for length in (1, 3, 32, 73, 129):
                expected = original(q, k[:, :, :length], v[:, :, :length])
                actual = _call(q, k[:, :, :length], v[:, :, :length], None, half_prob)
                torch.testing.assert_close(actual, expected, atol=.003, rtol=.01)
                cases.append({"head_dim": width, "heads": heads, "batch": batch,
                              "length": length,
                              "max_abs_error": (actual - expected).abs().max().item()})
            if width == 64 and batch == 1:
                # Capture with strided Q and prefix views, then change all
                # sources in place to verify the graph does not reuse scores.
                _call(q, k[:, :, :73], v[:, :, :73], None, half_prob)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replay_out = _call(q, k[:, :, :73], v[:, :, :73], None, half_prob)
                q.mul_(-.5)
                k.mul_(.2)
                v.add_(.17)
                graph.replay()
                expected = original(q, k[:, :, :73], v[:, :, :73])
                torch.testing.assert_close(replay_out, expected, atol=.003, rtol=.01)
                cases.append({"changed_qkv_graph": "passed",
                              "max_abs_error": (replay_out - expected).abs().max().item()})
    for distribution in ("binary", "ternary"):
        config = WhisperConfig.tiny_smoke()
        model = ReplayWhisper(config, distribution, "packed")
        mel = torch.randn(1, config.n_mels, 2 * config.n_audio_ctx,
                          dtype=torch.float16, device="cuda")
        tokens = [3, 4, 5, 6]
        expected = model.run(mel, tokens).clone()
        restore = install(name, artifact_dir)
        try:
            actual = model.run(mel, tokens)
            torch.testing.assert_close(actual, expected, atol=.003, rtol=.01)
            model.capture(mel, tokens)
            torch.testing.assert_close(model.replay_graph(mel), expected, atol=.003, rtol=.01)
            changed = model.replay_graph(mel * .31).clone()
            torch.cuda.synchronize()
            if torch.equal(changed, actual):
                raise AssertionError("Changed-input graph replay was insensitive")
            cases.append({"distribution": distribution, "tiny_model_graph": "passed",
                          "max_abs_error": (actual - expected).abs().max().item()})
        finally:
            restore()
    return {"variant": name, "cases": len(cases), "checks": cases,
            "half_softmax_probabilities": half_prob,
            "max_abs_error": max(row["max_abs_error"] for row in cases)}
