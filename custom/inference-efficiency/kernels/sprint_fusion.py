"""Isolated decoder GEMV epilogue fusion candidates for the one-hour sprint.

The model, token loop, cache length, attention, encoder, FP16 intermediate
rounding, and FP32 GEMV accumulation are preserved. ``install`` patches only
packed decoder blocks in this process and returns a callable that restores the
original runtime. No production dispatcher is changed by importing this module.
"""
from __future__ import annotations

import functools
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from . import cuda_gemv


def variants():
    return ["fusion_residual", "fusion_gelu", "fusion_all"]


def source():
    """Reuse the reviewed GEMV arithmetic and add an FP16-rounded epilogue."""
    text = cuda_gemv._SOURCE
    # Every internal function and wrapper receives the same two new arguments.
    text = text.replace("fp16_t* Y, int", "fp16_t* Y, const fp16_t* Residual, int Mode, int")
    text = text.replace("S, Bias, Y, row", "S, Bias, Y, Residual, Mode, row")
    text = text.replace("S, Bias, Y, N", "S, Bias, Y, Residual, Mode, N")
    old = "    Y[(long long)m * n + row] = f2h(value);"
    assert text.count(old) == 1
    text = text.replace(old, r"""
    // Original GEMV output is FP16 before either PyTorch operation.
    value = h2f(f2h(value));
    if (Mode == 1) {
        value += h2f(__ldg(Residual + (long long)m * n + row));
    } else if (Mode == 2) {
        // Match torch.nn.functional.gelu(..., approximate="none").
        const float cdf = 0.5f * (1.0f + erff(value * 0.70710678118654752440f));
        value = value * cdf;
    }
    Y[(long long)m * n + row] = f2h(value);""")
    return text


def compile_only(artifact_dir):
    """Compile for SM120 without launching a kernel or creating a GPU context."""
    cuda_gemv._preload_nvrtc()
    from torch.cuda._utils import _nvrtc_compile
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    kernel_source = source()
    (artifact_dir / "fusion.cu").write_text(kernel_source)
    cubin, _ = _nvrtc_compile(kernel_source, "gemv_b1_w1", compute_capability="120")
    (artifact_dir / "fusion.cubin").write_bytes(cubin)
    return cubin


@functools.lru_cache(maxsize=1)
def _load(artifact_dir):
    from torch.cuda._utils import _cuda_load_module
    path = Path(artifact_dir) / "fusion.cubin"
    cubin = path.read_bytes() if path.is_file() else compile_only(artifact_dir)
    names = [f"gemv_b{bits}_w{wpl}"
             for bits, wpls in ((1, (1, 2, 4)), (2, (1, 2, 4, 8)))
             for wpl in wpls]
    names += [f"gemv_b{bits}_gen" for bits in (1, 2)]
    return _cuda_load_module(cubin, names)


def _linear_epilogue(weight, x, bias, mode, residual, kernels):
    p = weight.packed
    if p is None or math.prod(x.shape[:-1]) > 4:
        value = weight.linear(x, bias)
        return value + residual if mode == 1 else F.gelu(value)
    x = x.contiguous()
    out = torch.empty((*x.shape[:-1], p.n), dtype=x.dtype, device=x.device)
    m, kw = math.prod(x.shape[:-1]), p.words.shape[1]
    if residual is not None and (residual.shape != out.shape or not residual.is_contiguous()):
        raise ValueError("Fused residual must match the contiguous GEMV output")
    wpl = (kw + 31) // 32
    name = f"gemv_b{p.bits}_w{wpl}"
    if p.k % 32 or kw != wpl * 32 or name not in kernels:
        name = f"gemv_b{p.bits}_gen"
    kernels[name](
        grid=((p.n + 3) // 4, m, 1), block=(128, 1, 1),
        args=[x, p.words, p.scales, bias if bias is not None else out, out,
              residual if residual is not None else out, mode,
              p.n, p.k, kw, int(bias is not None)],
    )
    return out


def _attention_residual(attn, x, residual, kernels, *, memory_kv=None, cache=None, step=None):
    # This is the original attention implementation; only its final projection
    # absorbs the immediately following, FP16-rounded residual addition.
    if attn.cross:
        q = attn.split_heads(attn.q.linear(x, attn.q_bias))
        k, v = memory_kv
    else:
        q, k, v = attn.qkv.linear(x, attn.qkv_bias).chunk(3, dim=-1)
        q, k, v = map(attn.split_heads, (q, k, v))
        if cache is not None:
            if step is None or x.shape[1] != 1:
                raise ValueError("Cached attention requires one sequential input token")
            cache[0][:, :, step:step + 1, :].copy_(k)
            cache[1][:, :, step:step + 1, :].copy_(v)
            k, v = cache[0][:, :, :step + 1, :], cache[1][:, :, :step + 1, :]
    y = F.scaled_dot_product_attention(q, k, v, dropout_p=0, is_causal=False)
    y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], attn.width)
    return _linear_epilogue(attn.out, y, attn.out_bias, 1, residual, kernels)


def install(name, artifact_dir):
    """Install one candidate in an isolated worker; return restore callable."""
    if name not in variants():
        raise ValueError(f"Unknown fusion candidate {name!r}")
    from runtime_model import _Block
    original = _Block.__call__
    kernels = _load(str(Path(artifact_dir)))
    residual_fusion = name in {"fusion_residual", "fusion_all"}
    gelu_fusion = name in {"fusion_gelu", "fusion_all"}

    def block_call(self, x, *, memory_kv=None, cache=None, step=None):
        if (self.cross_attn is None or self.mlp_up.packed is None
                or math.prod(x.shape[:-1]) > 4):
            return original(self, x, memory_kv=memory_kv, cache=cache, step=step)
        if residual_fusion:
            x = _attention_residual(self.attn, self.attn_ln(x), x, kernels,
                                    cache=cache, step=step)
            x = _attention_residual(self.cross_attn, self.cross_attn_ln(x), x, kernels,
                                    memory_kv=memory_kv)
        else:
            x = x + self.attn(self.attn_ln(x), cache=cache, step=step)
            x = x + self.cross_attn(self.cross_attn_ln(x), memory_kv=memory_kv)
        normalized = self.mlp_ln(x)
        if gelu_fusion:
            y = _linear_epilogue(self.mlp_up, normalized, self.mlp_up_bias,
                                 2, None, kernels)
        else:
            y = F.gelu(self.mlp_up.linear(normalized, self.mlp_up_bias))
        if residual_fusion:
            return _linear_epilogue(self.mlp_down, y, self.mlp_down_bias,
                                    1, x, kernels)
        return x + self.mlp_down.linear(y, self.mlp_down_bias)

    _Block.__call__ = block_call

    def restore():
        _Block.__call__ = original

    return restore


def check(name, artifact_dir):
    """GPU checks; caller must hold the shared experiment GPU lock."""
    from runtime_model import ReplayWhisper, WhisperConfig, _Factory
    torch.set_grad_enabled(False)
    torch.manual_seed(212)
    kernels = _load(str(Path(artifact_dir)))
    if name not in variants():
        raise ValueError(name)
    records = []
    for distribution in ("binary", "ternary"):
        factory = _Factory(distribution, "packed", 109, "cuda", torch.float16)
        for n, k in ((19, 73), (1024, 1024), (4096, 1024), (1024, 4096)):
            weight, bias = factory.matrix(n, k), factory.vector(n)
            for m in (1, 4):
                x = torch.randn((m, k), device="cuda", dtype=torch.float16)
                residual = torch.randn((m, n), device="cuda", dtype=torch.float16)
                linear = weight.linear(x, bias)
                for mode in (1, 2):
                    expected = linear + residual if mode == 1 else F.gelu(linear)
                    actual = _linear_epilogue(weight, x, bias, mode,
                                              residual if mode == 1 else None, kernels)
                    # Accumulation and rounding of residual fusion are bitwise
                    # unchanged. CUDA erff vs PyTorch GELU may differ by one ulp.
                    torch.testing.assert_close(actual, expected,
                                               atol=0 if mode == 1 else 0.002,
                                               rtol=0 if mode == 1 else 0.001)
                    records.append({"distribution": distribution, "n": n, "k": k,
                                    "m": m, "mode": mode,
                                    "max_abs_error": (actual - expected).abs().max().item()})
                # An exactly cancelling FP16 residual must produce exact zero;
                # folding it before the linear's FP16 round would fail this.
                cancelled = _linear_epilogue(weight, x, bias, 1, -linear, kernels)
                if torch.count_nonzero(cancelled).item():
                    raise AssertionError("Fused residual changed intermediate FP16 rounding")
        config = WhisperConfig.tiny_smoke()
        model = ReplayWhisper(config, distribution, "packed")
        mel = torch.randn(1, config.n_mels, 2 * config.n_audio_ctx,
                          dtype=torch.float16, device="cuda")
        tokens = [3, 4, 5, 6]
        expected = model.run(mel, tokens).clone()
        for variant in (name,):
            restore = install(variant, artifact_dir)
            try:
                actual = model.run(mel, tokens)
                torch.testing.assert_close(actual, expected, atol=.003, rtol=.01)
                model.capture(mel, tokens)
                torch.testing.assert_close(model.replay_graph(mel), expected,
                                           atol=.003, rtol=.01)
                changed = model.replay_graph(mel * .31).clone()
                torch.cuda.synchronize()
                if torch.equal(changed, actual):
                    raise AssertionError("Changed-input graph replay was insensitive")
                records.append({"distribution": distribution, "variant": variant,
                                "tiny_model_graph_check": "passed",
                                "max_abs_error": (actual - expected).abs().max().item()})
            finally:
                restore()
    return {"variant": name, "checks": records, "cases": len(records),
            "residual_cancellation": "passed",
            "max_abs_error": max(row["max_abs_error"] for row in records)}
