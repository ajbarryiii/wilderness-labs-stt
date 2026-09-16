"""Direct Q/K/V projection into decoder query and growing KV-cache buffers.

Only destination addresses change. Projection accumulation, scaling, bias and
FP16 rounding are inherited verbatim from the reviewed CUDA GEMV. The kernel
writes the same cache slice that the two original PyTorch copies populate.
"""
from __future__ import annotations

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

from . import cuda_gemv, sprint_fusion


def variants():
    return ["qkv_cache", "qkv_cache_fusion"]


def source():
    text = cuda_gemv._SOURCE
    args = "fp16_t* Y, fp16_t* KC, fp16_t* VC, int Capacity, int Step, int HeadDim, int"
    text = text.replace("fp16_t* Y, int", args)
    text = text.replace("S, Bias, Y, row", "S, Bias, Y, KC, VC, Capacity, Step, HeadDim, row")
    text = text.replace("S, Bias, Y, N", "S, Bias, Y, KC, VC, Capacity, Step, HeadDim, N")
    old = "    Y[(long long)m * n + row] = f2h(value);"
    assert text.count(old) == 1
    text = text.replace(old, r"""
    const int width = n / 3;
    if (row < width) {
        Y[(long long)m * width + row] = f2h(value);
    } else {
        const int offset = row >= 2 * width ? row - 2 * width : row - width;
        const int head = offset / HeadDim, dim = offset % HeadDim;
        const long long target = (long long)m * width * Capacity
            + (long long)head * Capacity * HeadDim + Step * HeadDim + dim;
        if (row < 2 * width) KC[target] = f2h(value);
        else VC[target] = f2h(value);
    }""")
    return text


def compile_only(artifact_dir):
    cuda_gemv._preload_nvrtc()
    from torch.cuda._utils import _nvrtc_compile
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    kernel_source = source()
    (artifact_dir / "qkv.cu").write_text(kernel_source)
    cubin, _ = _nvrtc_compile(kernel_source, "gemv_b1_w1", compute_capability="120")
    (artifact_dir / "qkv.cubin").write_bytes(cubin)
    return cubin


@functools.lru_cache(maxsize=1)
def _load(artifact_dir):
    from torch.cuda._utils import _cuda_load_module
    path = Path(artifact_dir) / "qkv.cubin"
    cubin = path.read_bytes() if path.is_file() else compile_only(artifact_dir)
    names = [f"gemv_b{b}_w{w}" for b, ws in ((1, (1, 2, 4)), (2, (1, 2, 4, 8)))
             for w in ws] + [f"gemv_b{b}_gen" for b in (1, 2)]
    return _cuda_load_module(cubin, names)


def _project(attn, x, cache, step, kernels):
    p = attn.qkv.packed
    if x.shape[1] != 1 or step is None:
        raise ValueError("Direct-cache projection requires a single sequential token")
    if (cache[0].shape != cache[1].shape or not cache[0].is_contiguous()
            or not cache[1].is_contiguous() or cache[0].shape[0] != x.shape[0]
            or cache[0].shape[1] != attn.heads
            or cache[0].shape[-1] != attn.width // attn.heads
            or not 0 <= step < cache[0].shape[2]):
        raise ValueError("Unexpected KV-cache layout")
    out = torch.empty((*x.shape[:-1], attn.width), device=x.device, dtype=x.dtype)
    kw, m = p.words.shape[1], x.shape[0]
    wpl = (kw + 31) // 32
    name = f"gemv_b{p.bits}_w{wpl}"
    if p.k % 32 or kw != wpl * 32 or name not in kernels:
        name = f"gemv_b{p.bits}_gen"
    kernels[name](grid=((p.n + 3) // 4, m, 1), block=(128, 1, 1),
                  args=[x.contiguous(), p.words, p.scales, attn.qkv_bias, out,
                        cache[0], cache[1], cache[0].shape[2], step,
                        attn.width // attn.heads, p.n, p.k, kw, 1])
    return (attn.split_heads(out), cache[0][:, :, :step + 1, :],
            cache[1][:, :, :step + 1, :])


def install(name, artifact_dir):
    if name not in variants():
        raise ValueError(name)
    from runtime_model import _Attention
    restore_fusion = (sprint_fusion.install("fusion_all", artifact_dir)
                      if name == "qkv_cache_fusion" else None)
    kernels = _load(str(Path(artifact_dir)))
    original_attention = _Attention.__call__
    original_residual = sprint_fusion._attention_residual

    def attention(self, x, *, memory_kv=None, cache=None, step=None):
        if self.cross or cache is None or self.qkv.packed is None:
            return original_attention(self, x, memory_kv=memory_kv, cache=cache, step=step)
        q, k, v = _project(self, x, cache, step, kernels)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0, is_causal=False)
        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.width)
        return self.out.linear(y, self.out_bias)

    def attention_residual(attn, x, residual, epilogue_kernels,
                           *, memory_kv=None, cache=None, step=None):
        if attn.cross or cache is None or attn.qkv.packed is None:
            return original_residual(attn, x, residual, epilogue_kernels,
                                     memory_kv=memory_kv, cache=cache, step=step)
        q, k, v = _project(attn, x, cache, step, kernels)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0, is_causal=False)
        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], attn.width)
        return sprint_fusion._linear_epilogue(attn.out, y, attn.out_bias, 1,
                                              residual, epilogue_kernels)

    _Attention.__call__ = attention
    sprint_fusion._attention_residual = attention_residual

    def restore():
        _Attention.__call__ = original_attention
        sprint_fusion._attention_residual = original_residual
        if restore_fusion is not None:
            restore_fusion()

    return restore


def check(name, artifact_dir):
    from runtime_model import ReplayWhisper, WhisperConfig, _Factory, _Attention
    if name not in variants():
        raise ValueError(name)
    kernels = _load(str(Path(artifact_dir)))
    torch.set_grad_enabled(False)
    torch.manual_seed(313)
    cases = []
    for distribution in ("binary", "ternary"):
        factory = _Factory(distribution, "packed", 444, "cuda", torch.float16)
        for width, heads in ((64, 4), (1024, 16)):
            attn = _Attention(factory, width, heads)
            for batch in (1, 4):
                x = torch.randn(batch, 1, width, device="cuda", dtype=torch.float16)
                expected = list(map(attn.split_heads,
                                    attn.qkv.linear(x, attn.qkv_bias).chunk(3, dim=-1)))
                shape = (batch, heads, 9, width // heads)
                cache = (torch.full(shape, 57., device="cuda", dtype=torch.float16),
                         torch.full(shape, 57., device="cuda", dtype=torch.float16))
                for step in (0, 4, 8):
                    expected_cache = tuple(c.clone() for c in cache)
                    for c, value in zip(expected_cache, expected[1:]):
                        c[:, :, step:step + 1].copy_(value)
                    q, _, _ = _project(attn, x, cache, step, kernels)
                    actual = (q, cache[0][:, :, step:step + 1],
                              cache[1][:, :, step:step + 1])
                    for a, e in zip(actual, expected):
                        torch.testing.assert_close(a, e, atol=0, rtol=0)
                    for actual_cache, reference_cache in zip(cache, expected_cache):
                        torch.testing.assert_close(actual_cache, reference_cache,
                                                   atol=0, rtol=0)
                    cases.append({"distribution": distribution, "width": width,
                                  "batch": batch, "step": step, "max_abs_error": 0})
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
            cases.append({"distribution": distribution, "full_model_graph": "passed",
                          "max_abs_error": (actual - expected).abs().max().item()})
        finally:
            restore()
    return {"variant": name, "checks": cases, "cases": len(cases),
            "cache_canaries": "passed",
            "max_abs_error": max(row["max_abs_error"] for row in cases)}
