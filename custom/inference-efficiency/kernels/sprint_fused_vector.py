"""Compose epilogue/cache fusion with vectorized FP16 activation loads.

Vector loads change FP32 summation order but retain its precision. Fusion still
rounds the complete linear result to FP16 before GELU or residual addition.
An offset FP16 view that lacks 8-byte alignment uses the reviewed scalar load
kernel. Installation is process-local and returns a complete restore callable.
"""
from __future__ import annotations

import functools
from pathlib import Path

import torch
import torch.nn.functional as F

from . import cuda_gemv, sprint_cuda, sprint_fusion, sprint_qkv


def variants():
    return ["fusion_vec4", "qkv_fusion_vec4", "qkv_fusion_owned4", "qkv_fusion_hybrid",
            "qkv_fusion_hybrid_shift"]


def compile_only(name, artifact_dir):
    if name not in variants():
        raise ValueError(name)
    from torch.cuda._utils import _nvrtc_compile
    cuda_gemv._preload_nvrtc()
    folder = Path(artifact_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    families = (("pair", 2, False), ("quad", 4, False), ("owned", 4, True)) if (
        name.startswith("qkv_fusion_hybrid")) else (("default", 4, name.endswith("owned4")),)
    for family, vector, owned in families:
        for kind, generate in (("epilogue", sprint_fusion.source), ("qkv", sprint_qkv.source)):
            source = generate()
            if name.endswith("_shift") and kind == "qkv":
                old = "const int head = offset / HeadDim, dim = offset % HeadDim;"
                assert source.count(old) == 1
                source = source.replace(old,
                    "const int head = offset >> (31 - __clz(HeadDim));\n"
                    "        const int dim = offset & (HeadDim - 1);")
            source = sprint_cuda.vectorize_warp_source(source, vector=vector, owned=owned)
            cubin, _ = _nvrtc_compile(source, "gemv_b1_w1", compute_capability="120")
            (folder / f"{kind}-{family}.cu").write_text(source)
            (folder / f"{kind}-{family}.cubin").write_bytes(cubin)
    return folder


@functools.lru_cache(None)
def _load(name, artifact_dir):
    from torch.cuda._utils import _cuda_load_module
    folder = Path(artifact_dir) / name
    families = ("pair", "quad", "owned") if name.startswith("qkv_fusion_hybrid") else ("default",)
    if not all((folder / f"{kind}-{family}.cubin").is_file()
               for kind in ("epilogue", "qkv") for family in families):
        folder = compile_only(name, artifact_dir)
    names = [f"gemv_b{b}_w{w}" for b, ws in ((1, (1, 2, 4)), (2, (1, 2, 4, 8)))
             for w in ws] + [f"gemv_b{b}_gen" for b in (1, 2)]
    result = []
    for kind in ("epilogue", "qkv"):
        modules = {family: _cuda_load_module((folder / f"{kind}-{family}.cubin").read_bytes(), names)
                   for family in families}
        if not name.startswith("qkv_fusion_hybrid"):
            result.append(modules["default"])
            continue
        # Shared-input GPU diagnostics favor pair loads at K1024, quad for
        # binary K4096 and direct owned-word loads for ternary K4096.
        merged = dict(modules["pair"])
        merged["gemv_b1_w4"] = modules["quad"]["gemv_b1_w4"]
        merged["gemv_b2_w8"] = modules["owned"]["gemv_b2_w8"]
        result.append(merged)
    return tuple(result)


def install(name, artifact_dir):
    if name not in variants():
        raise ValueError(name)
    vector_epilogue, vector_qkv = _load(name, str(Path(artifact_dir)))
    original_epilogue = sprint_fusion._linear_epilogue
    original_project = sprint_qkv._project
    restore_base = (sprint_fusion.install("fusion_all", Path(artifact_dir) / "base")
                    if name == "fusion_vec4" else
                    sprint_qkv.install("qkv_cache_fusion", Path(artifact_dir) / "base"))

    def epilogue(weight, x, bias, mode, residual, kernels):
        selected = vector_epilogue if x.data_ptr() % 8 == 0 else kernels
        return original_epilogue(weight, x, bias, mode, residual, selected)

    def project(attn, x, cache, step, kernels):
        head_dim = attn.width // attn.heads
        can_index = not name.endswith("_shift") or (head_dim & (head_dim - 1)) == 0
        selected = vector_qkv if x.data_ptr() % 8 == 0 and can_index else kernels
        return original_project(attn, x, cache, step, selected)

    sprint_fusion._linear_epilogue = epilogue
    sprint_qkv._project = project

    def restore():
        sprint_fusion._linear_epilogue = original_epilogue
        sprint_qkv._project = original_project
        restore_base()

    return restore


def check(name, artifact_dir):
    from runtime_model import ReplayWhisper, WhisperConfig, _Factory, _Attention
    torch.set_grad_enabled(False)
    torch.manual_seed(515)
    epilogue_kernels, qkv_kernels = _load(name, str(Path(artifact_dir)))
    scalar_kernels = sprint_fusion._load(str(Path(artifact_dir) / "base"))
    scalar_qkv_kernels = sprint_qkv._load(str(Path(artifact_dir) / "base"))
    cases = []
    for distribution in ("binary", "ternary"):
        factory = _Factory(distribution, "packed", 516, "cuda", torch.float16)
        for n, k in ((19, 73), (1024, 1024), (4096, 1024), (1024, 4096)):
            weight, bias = factory.matrix(n, k), factory.vector(n)
            for m in (1, 4):
                x = torch.randn(m, k, device="cuda", dtype=torch.float16)
                residual = torch.randn(m, n, device="cuda", dtype=torch.float16)
                # Mode zero uses the identical vector accumulation without a
                # fused operation, isolating the required FP16 rounding point.
                vector_linear = sprint_fusion._linear_epilogue(
                    weight, x, bias, 0, None, epilogue_kernels)
                baseline = weight.linear(x, bias)
                torch.testing.assert_close(vector_linear, baseline, atol=.003, rtol=.003)
                for mode in (1, 2):
                    expected = vector_linear + residual if mode == 1 else F.gelu(vector_linear)
                    actual = sprint_fusion._linear_epilogue(
                        weight, x, bias, mode, residual if mode == 1 else None,
                        epilogue_kernels)
                    torch.testing.assert_close(actual, expected,
                                               atol=0 if mode == 1 else .002,
                                               rtol=0 if mode == 1 else .001)
                cancelled = sprint_fusion._linear_epilogue(
                    weight, x, bias, 1, -vector_linear, epilogue_kernels)
                if torch.count_nonzero(cancelled).item():
                    raise AssertionError("Vector fusion skipped intermediate FP16 rounding")
                cases.append({"distribution": distribution, "n": n, "k": k, "m": m,
                              "max_abs_error": (vector_linear - baseline).abs().max().item()})
        attn = _Attention(factory, 1024, 16)
        x = torch.randn(1, 1, 1024, dtype=torch.float16, device="cuda")
        linear = sprint_fusion._linear_epilogue(attn.qkv, x, attn.qkv_bias,
                                               0, None, epilogue_kernels)
        expected_qkv = list(map(attn.split_heads, linear.chunk(3, -1)))
        cache = (torch.full((1, 16, 9, 64), 57., device="cuda", dtype=torch.float16),
                 torch.full((1, 16, 9, 64), 57., device="cuda", dtype=torch.float16))
        q, _, _ = sprint_qkv._project(attn, x, cache, 4, qkv_kernels)
        for actual, expected in zip((q, cache[0][:, :, 4:5], cache[1][:, :, 4:5]),
                                    expected_qkv):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for c in cache:
            if not ((c[:, :, :4] == 57).all() & (c[:, :, 5:] == 57).all()).item():
                raise AssertionError("Vector QKV projection touched another token")
        # Exercise alignment fallback using an offset but contiguous FP16 view.
        offset = torch.randn(1025, dtype=torch.float16, device="cuda")[1:].view(1, 1024)
        weight, bias = factory.matrix(1024, 1024), factory.vector(1024)
        expected = F.gelu(weight.linear(offset, bias))
        restore = install(name, artifact_dir)
        try:
            actual = sprint_fusion._linear_epilogue(weight, offset, bias, 2,
                                                   None, scalar_kernels)
            torch.testing.assert_close(actual, expected, atol=.002, rtol=.001)
        finally:
            restore()
        # QKV alignment fallback and the shift variant's generic head-dimension
        # fallback each use the original scalar projection, without vector loads.
        for width, heads, batch, offset_elements in ((1024, 16, 4, 1), (96, 4, 1, 0)):
            fallback_attn = _Attention(factory, width, heads)
            raw = torch.randn(batch * width + offset_elements,
                              device="cuda", dtype=torch.float16)
            fallback_x = raw[offset_elements:].view(batch, 1, width)
            shape = (batch, heads, 7, width // heads)
            reference_cache = tuple(torch.full(shape, 57., device="cuda", dtype=torch.float16)
                                    for _ in range(2))
            actual_cache = tuple(c.clone() for c in reference_cache)
            expected_q, _, _ = sprint_qkv._project(fallback_attn, fallback_x,
                                                  reference_cache, 3, scalar_qkv_kernels)
            restore = install(name, artifact_dir)
            try:
                actual_q, _, _ = sprint_qkv._project(fallback_attn, fallback_x,
                                                    actual_cache, 3, scalar_qkv_kernels)
                torch.testing.assert_close(actual_q, expected_q, atol=0, rtol=0)
                for a, e in zip(actual_cache, reference_cache):
                    torch.testing.assert_close(a, e, atol=0, rtol=0)
            finally:
                restore()
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
            cases.append({"distribution": distribution, "model_graph": "passed",
                          "max_abs_error": (actual - expected).abs().max().item()})
        finally:
            restore()
    return {"variant": name, "cases": len(cases), "checks": cases,
            "alignment_fallback": "passed", "intermediate_fp16_rounding": "passed",
            "cache_canaries": "passed",
            "max_abs_error": max(row["max_abs_error"] for row in cases)}
