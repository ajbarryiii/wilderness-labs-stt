"""Bounded CUDA source mutations and dispatch policies for kernel autoresearch.

Only the five decoder GEMV shapes are changed. Encoder GEMM, convolutions,
attention, precision, weights, and inference scheduling stay in the evaluator.
"""
import functools
import math
from pathlib import Path
import statistics

import torch
import triton

from paths import digest, save
from . import cuda_gemv, packed
from .investigate_cuda import accumulated_source, transposed_source, _MULTIWARP

AXES = {"point": (1024, 1024), "qkv": (3072, 1024),
        "up": (4096, 1024), "down": (1024, 4096), "vocab": (51864, 1024)}
CALLS = {"point": 3 * 24 * 129, "qkv": 24 * 129,
         "up": 24 * 129, "down": 24 * 129, "vocab": 128}
METHODS = {axis: ("base", "acc2", "acc4", "split4", "warp32", "warp64", "triton")
           for axis in AXES}
METHODS["vocab"] = ("base", "transposed", "rpl", "tc32", "tc64", "tc128", "tc64k128")
BASE = dict.fromkeys(AXES, "base")
ORIGINAL = packed.PackedWeight.linear
NAMES = ([f"gemv_b{b}_w{w}" for b, ws in ((1, (1, 2, 4)), (2, (1, 2, 4, 8))) for w in ws]
         + [f"gemv_b{b}_{v}" for b in (1, 2) for v in ("rpl", "gen")])


def validate_policy(policy):
    if set(policy) != set(AXES) or any(policy[a] not in METHODS[a] for a in AXES):
        raise ValueError(f"Policy outside frozen search space: {policy}")
    return policy


def sources():
    return {"base": cuda_gemv._SOURCE, "acc2": accumulated_source(2),
            "acc4": accumulated_source(4), "transposed": transposed_source(),
            "split4": cuda_gemv._SOURCE + _MULTIWARP}


def compile_modules(folder):
    from torch.cuda._utils import _nvrtc_compile
    cuda_gemv._preload_nvrtc()
    folder.mkdir(parents=True, exist_ok=True)
    for name, source in sources().items():
        (folder / f"{name}.cu").write_text(source)
        cubin, _ = _nvrtc_compile(source, "gemv_b2_w8", compute_capability="120")
        (folder / f"{name}.cubin").write_bytes(cubin)
    save(folder / "hashes.json", {p.name: digest(p) for p in sorted(folder.iterdir())
                                if p.suffix in (".cu", ".cubin")})


@functools.lru_cache(None)
def module(folder, name):
    from torch.cuda._utils import _cuda_load_module
    names = NAMES + ([f"gemv_b{b}_split4" for b in (1, 2)] if name == "split4" else [])
    return _cuda_load_module((Path(folder) / f"{name}.cubin").read_bytes(), names)


def launch(method, folder, p, x, bias, out):
    m, n, k, bits = x.numel() // p.k, p.n, p.k, p.bits
    kw = p.words.shape[1]
    if method == "base":
        return ORIGINAL(p, x, bias, out)
    args = [x, p.words, p.scales, bias if bias is not None else out, out,
            n, k, kw, int(bias is not None)]
    if method.startswith("tc"):
        bn, bk = {"tc32": (32, 64), "tc64": (64, 64),
                  "tc128": (128, 64), "tc64k128": (64, 128)}[method]
        packed._gemm[(triton.cdiv(n, bn),)](
            x, p.words_t, p.scales, args[3], out, m, n, k, kw, bits,
            bias is not None, 16, bn, bk, 8, num_warps=4, num_stages=3)
    elif method == "triton":
        bn, bkc, warps = packed._gemv_config(n, k, bits)
        packed._gemv[(triton.cdiv(n, bn), m)](
            x, p.words, p.scales, args[3], out, n, k, kw, bits,
            bias is not None, bn, bkc, num_warps=warps)
    else:
        source = method if method in ("acc2", "acc4", "split4", "transposed") else "base"
        kernels = module(str(folder), source)
        block, rows, shared = 128, 4, 0
        if method in ("transposed", "rpl"):
            if k % (32 // bits):
                raise ValueError("RPL requires complete packed words")
            name, rows = "rpl", 128
            shared = 4 * k if method == "transposed" else 4 * (k + 128 * (kw | 1))
            if method == "transposed":
                args[1] = p.words_t
        elif method == "split4":
            name, rows = "split4", 1
        else:
            if method in ("warp32", "warp64"):
                block = int(method[4:])
                rows = block // 32
            wpl = kw // 32
            name = f"w{wpl}" if (k % 32 == 0 and kw == wpl * 32
                                     and f"gemv_b{bits}_w{wpl}" in kernels) else "gen"
        kernels[f"gemv_b{bits}_{name}"](grid=(triton.cdiv(n, rows), m, 1),
                                        block=(block, 1, 1), args=args, shared_mem=shared)
    return out


def install(policy, folder):
    validate_policy(policy)
    cuda_gemv.load()  # Compilation must finish before any graph capture.
    for method in set(policy.values()) & {"acc2", "acc4", "split4", "transposed"}:
        module(str(folder), method)
    module(str(folder), "base")

    def linear(p, x, bias=None, out=None):
        axis = next((a for a, shape in AXES.items() if shape == (p.n, p.k)), None)
        if (axis is None or x.ndim < 1 or math.prod(x.shape[:-1]) > 4
                or not math.prod(x.shape[:-1]) or policy[axis] == "base"):
            return ORIGINAL(p, x, bias, out)
        if x.shape[-1] != p.k or x.dtype != torch.float16 or not x.is_cuda or x.device != p.words.device:
            raise ValueError("Research GEMV requires FP16 CUDA input [...,K]")
        if bias is not None and (bias.shape != (p.n,) or bias.dtype != x.dtype
                                 or bias.device != x.device or not bias.is_contiguous()):
            raise ValueError("Invalid bias")
        shape = (*x.shape[:-1], p.n)
        if out is None:
            out = torch.empty(shape, device=x.device, dtype=x.dtype)
        elif (out.shape != shape or out.dtype != x.dtype or out.device != x.device
              or not out.is_contiguous()):
            raise ValueError("Invalid output")
        return launch(policy[axis], folder, p, x.contiguous(), bias, out)

    packed.PackedWeight.linear = ORIGINAL if policy == BASE else linear


def check_all(folder):
    """Full shapes, row tails, non-dyadic scales, Gaussian/dyadic inputs, graphs."""
    from .verify_cuda import CudaGemvTests
    helper = CudaGemvTests()
    torch.set_num_threads(4)
    torch.manual_seed(23941)
    cuda_gemv.load()
    count, worst = 0, 0.
    for bits in (1, 2):
        for axis, (n, k) in AXES.items():
            for rows, m in ((n, 1), (129 if axis == "vocab" else 19, 4)):
                p, x, bias, expected = helper.inputs(bits, rows, k, m, bias=m == 4)
                # A second distribution catches reduction-order error on real FP16 inputs.
                x.normal_()
                host_weights = p.dense().cpu().float()
                expected = x.cpu().float() @ host_weights.T
                if bias is not None:
                    expected += bias.cpu().float()
                expected = expected.half()
                for method in METHODS[axis]:
                    guarded = torch.full((m * rows + 16,), 321., device="cuda", dtype=torch.float16)
                    out = guarded[8:-8].view(m, rows)
                    launch(method, folder, p, x, bias, out)
                    torch.testing.assert_close(out.cpu(), expected, atol=.003, rtol=.003)
                    worst = max(worst, float((out.cpu().float() - expected.float()).abs().max()))
                    if not bool((guarded[:8] == 321).all() & (guarded[-8:] == 321).all()):
                        raise AssertionError("Output canary overwritten")
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        launch(method, folder, p, x, bias, out)
                    original = x.clone()
                    x.zero_()
                    graph.replay()
                    zero_expected = torch.zeros_like(out) if bias is None else bias.expand_as(out)
                    torch.testing.assert_close(out, zero_expected, atol=0, rtol=0)
                    x.copy_(original)
                    graph.replay()
                    torch.testing.assert_close(out.cpu(), expected, atol=.003, rtol=.003)
                    del graph
                    count += 1
                del p, x, bias, out, guarded, host_weights
    # Regression for the earlier unsigned-code cancellation bug, including new RPL.
    for bits in (1, 2):
        k, n = 1024, 129
        x = torch.full((1, k), 1 / 1024, dtype=torch.float16)
        codes = torch.zeros((n, k), dtype=torch.int8)
        expected_value = 0.
        if bits == 1:
            x[0, :2] = 32752
            codes.fill_(1)
            codes[:, 0] = -1
            expected_value = (k - 2) / 1024
        else:
            x[0, 0] = 65504
        p, x, bias, _ = helper.inputs(bits, n, k, x=x, codes=codes, scale=1.)
        for method in ("transposed", "rpl"):
            out = torch.empty((1, n), device="cuda", dtype=torch.float16)
            launch(method, folder, p, x, bias, out)
            torch.testing.assert_close(out, torch.full_like(out, expected_value), atol=0, rtol=0)
            count += 1
    return {"passed": True, "cases": count, "maximum_absolute_error": worst,
            "checks": ["FP32 dense reference", "full shapes", "row tails", "FP16 scale rounding",
                       "output canaries", "nondefault capture stream", "changed-input graph replay",
                       "RPL cancellation regression"]}


def profile(folder):
    """Rotating-order event timings guide proposals; never used as energy scores."""
    from .verify_cuda import CudaGemvTests
    helper = CudaGemvTests()
    torch.set_num_threads(4)
    torch.manual_seed(74821)
    cuda_gemv.load()
    result = {}
    for bits in (1, 2):
        result[str(bits)] = {}
        for axis, (n, k) in AXES.items():
            p, x, bias, _ = helper.inputs(bits, n, k, bias=True)
            out = torch.empty((1, n), device="cuda", dtype=torch.float16)
            graphs, times = {}, {method: [] for method in METHODS[axis]}
            for method in METHODS[axis]:
                for _ in range(3):
                    launch(method, folder, p, x, bias, out)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(32):
                        launch(method, folder, p, x, bias, out)
                graphs[method] = graph
            methods = list(graphs)
            for repeat in range(9):
                for method in methods[repeat % len(methods):] + methods[:repeat % len(methods)]:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(16):
                        graphs[method].replay()
                    end.record()
                    end.synchronize()
                    times[method].append(start.elapsed_time(end) * 1000 / (32 * 16))
            result[str(bits)][axis] = {m: {"median_us": statistics.median(t), "rounds_us": t}
                                       for m, t in times.items()}
            del graphs, p, x, bias, out
    return result
