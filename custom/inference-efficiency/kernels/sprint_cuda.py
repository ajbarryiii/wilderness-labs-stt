"""Agent-led SM120 experiments: bitwise signs and vector activation loads.

This module is isolated from production dispatch. Candidates change only small-M
GEMV with K=1024/4096 and N<8192; all other operations use the original kernels.
The signed half values are converted individually to FP32 and summed in FP32.
Compilation artifacts must be on the mounted experiment disk.
"""
import functools
import math
from pathlib import Path
import statistics

import torch
import triton

from paths import artifact, digest, save
from . import cuda_gemv, packed

ORIGINAL = packed.PackedWeight.linear
_METHODS = {
    "sign": (1, 1, 1, 1, 0),
    "pair": (2, 1, 2, 1, 0),
    "quad": (4, 1, 4, 1, 0),
    "pair_rows2": (2, 2, 2, 1, 0),
    "pair_fma": (2, 1, 2, 0, 0),
    "quad_fma": (4, 1, 4, 0, 0),
    "quad_ptx": (4, 1, 4, 2, 0),
    "quad_rows2_fma": (4, 2, 4, 0, 0),
    "ownedword": (4, 1, 4, 0, 1),
}

_BODY = r'''
struct __align__(8) halves4 { u32 lo, hi; };

template<int BITS>
__device__ __forceinline__ float signed_half(fp16_t h, u32 code) {
    // -1 toggles the input sign. Ternary zero clears both sign and magnitude.
    // No coefficient conversion and no half-precision arithmetic.
    u32 raw = (u32)h ^ ((~code << (16 - BITS)) & 0x8000u);
    if (BITS == 2) raw &= (code & 1u) - 1u;
    return h2f((fp16_t)raw);
}

template<int BITS>
__device__ __forceinline__ float signed_half_ptx(fp16_t h, u32 code) {
    u32 sign = ((~code << (16 - BITS)) & 0x8000u);
    u32 raw;
    if (BITS == 2) {
        u32 mask;
        asm("{ .reg .u32 low; and.b32 low, %1, 1; sub.u32 %0, low, 1; }"
            : "=r"(mask) : "r"(code));
        asm("lop3.b32 %0, %1, %2, %3, 0x28;"
            : "=r"(raw) : "r"((u32)h), "r"(sign), "r"(mask));
    } else {
        raw = (u32)h ^ sign;
    }
    return h2f((fp16_t)raw);
}

template<int BITS, int KFIX, int VEC, int ROWS, int ACCS, int SIGNED, int OWNED>
__device__ __forceinline__ void sprint_impl(
        const fp16_t* X, const u32* W, const float* S, const fp16_t* Bias,
        fp16_t* Y, int N, int K, int KW, int has_bias) {
    constexpr int WPL = KFIX * BITS / 1024;
    constexpr int STEPS = 32 / BITS / VEC;
    const int lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * ROWS;
    if (row0 >= N) return;
    const int m = blockIdx.y;
    const fp16_t* xm = X + (long long)m * KFIX;
    u32 words[ROWS][WPL];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        #pragma unroll
        for (int j = 0; j < WPL; ++j)
            words[r][j] = row0 + r < N
                ? __ldg(W + (long long)(row0 + r) * KW + lane + 32 * j) : 0u;
    }
    float acc[ROWS][ACCS] = {};
    #pragma unroll
    for (int j = 0; j < WPL; ++j) {
        #pragma unroll
        for (int t = 0; t < STEPS; ++t) {
            constexpr int VPW = 32 / BITS;
            const int offset = OWNED ? (j * 32 + lane) * VPW + t * VEC
                                     : (j * STEPS + t) * 32 * VEC + lane * VEC;
            const int src = t * BITS * VEC + lane / (VPW / VEC);
            const int shift = OWNED ? t * BITS * VEC : (lane * BITS * VEC) & 31;
            fp16_t xh[VEC];
            if (VEC == 1) {
                xh[0] = __ldg(xm + offset);
            } else if (VEC == 2) {
                const u32 xx = __ldg((const u32*)(xm + offset));
                xh[0] = (fp16_t)xx;
                xh[1] = (fp16_t)(xx >> 16);
            } else {
                // Aligned eight-byte load: four adjacent FP16 activations.
                const unsigned long long xx = __ldg((const unsigned long long*)(xm + offset));
                xh[0] = (fp16_t)xx;
                xh[1] = (fp16_t)(xx >> 16);
                xh[2] = (fp16_t)(xx >> 32);
                xh[3] = (fp16_t)(xx >> 48);
            }
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const u32 w = OWNED ? words[r][j]
                                   : __shfl_sync(0xffffffffu, words[r][j], src);
                #pragma unroll
                for (int q = 0; q < VEC; ++q) {
                    const u32 c = (w >> (shift + q * BITS)) & ((1u << BITS) - 1u);
                    if (SIGNED == 1)
                        acc[r][q % ACCS] += signed_half<BITS>(xh[q], c);
                    else if (SIGNED == 2)
                        acc[r][q % ACCS] += signed_half_ptx<BITS>(xh[q], c);
                    else {
                        const float coefficient = BITS == 1
                            ? (c ? 1.f : -1.f) : (float)c - 1.f;
                        acc[r][q % ACCS] = fmaf(h2f(xh[q]), coefficient, acc[r][q % ACCS]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        float sum = acc[r][0];
        #pragma unroll
        for (int a = 1; a < ACCS; ++a) sum += acc[r][a];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, off);
        if (lane == 0 && row0 + r < N)
            store_result(sum, S, Bias, Y, row0 + r, m, N, has_bias);
    }
}
'''


def variants():
    return ["sign", "pair", "quad", "pair_rows2", "pair_down", "quad_down",
            "pair_fma", "quad_fma", "quad_ptx", "quad_rows2_fma", "ownedword",
            "quad_fma_b32", "ownedword_b32", "hybrid_vector", "hybrid"]


def _source():
    header = cuda_gemv._SOURCE[:cuda_gemv._SOURCE.index("template <int BITS, int WPL>")]
    wrappers = []
    for method, (vec, rows, accs, signed, owned) in _METHODS.items():
        for bits in (1, 2):
            for k in (1024, 4096):
                wrappers.append(f'''extern "C" __global__ void __launch_bounds__(128)
                sprint_{method}_b{bits}_k{k}(
                    const fp16_t* X, const u32* W, const float* S, const fp16_t* Bias,
                    fp16_t* Y, int N, int K, int KW, int has_bias) {{
                    sprint_impl<{bits}, {k}, {vec}, {rows}, {accs}, {signed}, {owned}>(
                        X, W, S, Bias, Y, N, K, KW, has_bias);
                }}''')
    return header + _BODY + "\n".join(wrappers)


def vectorize_warp_source(text, vector=4, owned=False):
    """Transform the production warp body, preserving a caller's epilogue.

    This can be applied after sprint_fusion.source() adds Residual/Mode: original
    function names, arguments and final epilogue call stay intact. The dispatcher
    must enforce natural 2*vector byte activation alignment for this variant.
    """
    if vector not in (2, 4):
        raise ValueError("Activation vector must contain two or four FP16 values")
    start = text.index("template <int BITS, int WPL>")
    end = text.index("// Row-per-lane kernel", start)
    body = text[start:end]

    def replace(old, new):
        nonlocal body
        if body.count(old) != 1:
            raise RuntimeError(f"Production warp source changed at {old!r}")
        body = body.replace(old, new, 1)

    replace("const fp16_t* xm = X + (long long)m * K + gi;",
            "const fp16_t* xm = X + (long long)m * K;")
    replace("const int bpos = BITS == 1 ? gi : ((gi & 15) << 1);",
            f"const int bpos = (gi * BITS * {vector}) & 31;")
    replace("float acc = 0.0f;", f"float accs[{vector}] = {{}};")
    replace("for (int t = 0; t < 32 / BITS; ++t)",
            f"for (int t = 0; t < 32 / BITS / {vector}; ++t)")
    replace("const int s = (j << (BITS == 1 ? 5 : 4)) + t;",
            f"const int s = j * (32 / BITS / {vector}) + t;")
    replace("const int src = BITS == 1 ? t : (((s << 1) | wsel) & 31);",
            f"const int src = t * BITS * {vector} + gi / (32 / BITS / {vector});")
    if owned:
        replace("const u32 wv = __shfl_sync(0xffffffffu, w[j], src);",
                "const u32 wv = w[j];")
    begin = body.index("            const float xv =")
    finish = body.index("        }\n    }\n    epilogue", begin)
    offset = (f"(j * 32 + gi) * (32 / BITS) + t * {vector}" if owned
              else f"(s * 32 + gi) * {vector}")
    typename = "u32" if vector == 2 else "unsigned long long"
    shift = "t * BITS * " + str(vector) if owned else "bpos"
    inner = f'''            const {typename} xx = __ldg((const {typename}*)(xm + {offset}));
            #pragma unroll
            for (int q = 0; q < {vector}; ++q) {{
                const fp16_t h = (fp16_t)(xx >> (q * 16));
                const u32 code = (wv >> ({shift} + q * BITS)) & ((1u << BITS) - 1u);
                const float coefficient = BITS == 1
                    ? (code ? 1.0f : -1.0f) : (float)code - 1.0f;
                accs[q] = fmaf(h2f(h), coefficient, accs[q]);
            }}
'''
    body = body[:begin] + inner + body[finish:]
    total = " + ".join(f"accs[{q}]" for q in range(vector))
    replace("    epilogue(acc,", f"    float acc = {total};\n    epilogue(acc,")
    return text[:start] + body + text[end:]


def compile_only(artifact_dir):
    from torch.cuda._utils import _nvrtc_compile
    folder = artifact(Path(artifact_dir) / "sprint-cuda")
    folder.mkdir(parents=True, exist_ok=True)
    source = _source()
    src = folder / "sprint.cu"
    cub = folder / "sprint.cubin"
    if not src.exists() or src.read_text() != source or not cub.exists():
        cuda_gemv._preload_nvrtc()
        cubin, _ = _nvrtc_compile(source, "sprint_sign_b1_k1024", compute_capability="120")
        src.write_text(source)
        cub.write_bytes(cubin)
    save(folder / "hashes.json", {p.name: digest(p) for p in (src, cub)})
    return folder


@functools.lru_cache(None)
def _module(folder):
    from torch.cuda._utils import _cuda_load_module
    names = [f"sprint_{method}_b{bits}_k{k}" for method in _METHODS
             for bits in (1, 2) for k in (1024, 4096)]
    return _cuda_load_module((Path(folder) / "sprint.cubin").read_bytes(), names)


def _method(name, n, k, bits=None):
    if name == "hybrid":
        return ("ownedword" if bits == 2 else "quad_fma") if k == 4096 else "pair_fma"
    if name == "hybrid_vector":
        return "ownedword" if k == 4096 else "pair_fma"
    method = name.removesuffix("_b32").removesuffix("_down")
    if name.endswith("_down") and k != 4096:
        return None
    if method == "pair_rows2" and n < 3072:
        return "pair"
    if method == "quad_rows2_fma" and n < 3072:
        return "quad_fma"
    return method


def install(name, artifact_dir):
    if name not in variants():
        raise ValueError(f"Unknown CUDA sprint candidate {name}")
    cuda_gemv.load()
    kernels = _module(str(compile_only(artifact_dir)))

    def linear(p, x, bias=None, out=None):
        method = _method(name, p.n, p.k, p.bits)
        if (method is None or p.k not in (1024, 4096) or p.n >= 8192
                or x.ndim < 1 or not 0 < math.prod(x.shape[:-1]) <= 4):
            return ORIGINAL(p, x, bias, out)
        if x.shape[-1] != p.k or x.dtype != torch.float16 or not x.is_cuda or x.device != p.words.device:
            raise ValueError("Sprint GEMV requires FP16 CUDA input [...,K]")
        if bias is not None and (bias.shape != (p.n,) or bias.dtype != x.dtype
                                 or bias.device != x.device or not bias.is_contiguous()):
            raise ValueError("Invalid bias")
        shape = (*x.shape[:-1], p.n)
        if out is None:
            out = torch.empty(shape, dtype=x.dtype, device=x.device)
        elif out.shape != shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
            raise ValueError("Invalid output")
        vec, rows, accs, signed, owned = _METHODS[method]
        x = x.contiguous()
        # A contiguous view may start at an odd FP16 storage offset. Wide loads
        # require natural alignment, so retain the scalar production path there.
        if x.data_ptr() % (2 * vec):
            return ORIGINAL(p, x, bias, out)
        threads = 32 if name.endswith("_b32") or (name == "hybrid_vector" and p.k == 4096) else 128
        kernels[f"sprint_{method}_b{p.bits}_k{p.k}"](
            grid=(triton.cdiv(p.n, (threads//32) * rows), math.prod(x.shape[:-1]), 1),
            block=(threads, 1, 1), args=[x, p.words, p.scales,
                                    bias if bias is not None else out, out,
                                    p.n, p.k, p.words.shape[1], int(bias is not None)])
        return out

    packed.PackedWeight.linear = linear
    return {"name": name, "precision": "FP16 input/output, FP32 accumulation",
            "source_sha256": digest(Path(__file__))}


def check(name, artifact_dir):
    from .verify_cuda import CudaGemvTests
    install(name, artifact_dir)
    helper = CudaGemvTests()
    torch.set_num_threads(4)
    torch.manual_seed(37825)
    worst, cases = 0., 0
    try:
        for bits in (1, 2):
            for n, k, m in ((19, 1024, 4), (1024, 1024, 1), (3073, 1024, 1),
                            (4096, 1024, 1), (19, 4096, 4), (1024, 4096, 1),
                            (19, 73, 1)):
                p, x, bias, _ = helper.inputs(bits, n, k, m, bias=True)
                x.normal_()
                expected = (x.cpu().float() @ p.dense().cpu().float().T
                            + bias.cpu().float()).half()
                guard = torch.full((m*n+16,), 321., device="cuda", dtype=torch.float16)
                out = guard[8:-8].view(m, n)
                p.linear(x, bias, out)
                torch.testing.assert_close(out.cpu(), expected, atol=.003, rtol=.003)
                worst = max(worst, float((out.cpu().float() - expected.float()).abs().max()))
                assert bool((guard[:8] == 321).all() & (guard[-8:] == 321).all())
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    p.linear(x, bias, out)
                original = x.clone()
                x.zero_()
                graph.replay()
                torch.testing.assert_close(out, bias.expand_as(out), atol=0, rtol=0)
                x.copy_(original)
                graph.replay()
                torch.testing.assert_close(out.cpu(), expected, atol=.003, rtol=.003)
                cases += 1
                del graph, p, x, out, guard, bias, expected, original
        report = {"passed": True, "cases": cases, "maximum_absolute_error": worst,
                  "checks": ["FP32 dense reference", "non-dyadic scales", "row tails",
                             "changed-input graph replay", "nondefault stream", "output canaries",
                             "ragged-K fallback"]}
        if name == "ownedword":
            report["cuda_method_diagnostics"] = profile_all(artifact_dir)
        return report
    finally:
        packed.PackedWeight.linear = ORIGINAL


def profile(name, artifact_dir):
    """Rotating baseline/candidate launch timings; energy comes from full model."""
    from .verify_cuda import CudaGemvTests
    helper = CudaGemvTests()
    torch.set_num_threads(4)
    torch.manual_seed(738251)
    report = {}
    try:
        for bits in (1, 2):
            report[str(bits)] = {}
            for n, k in ((1024, 1024), (3072, 1024), (4096, 1024), (1024, 4096)):
                p, x, bias, _ = helper.inputs(bits, n, k, bias=True)
                out = torch.empty((1, n), device="cuda", dtype=torch.float16)
                graphs = {}
                for arm in ("base", "candidate"):
                    packed.PackedWeight.linear = ORIGINAL
                    if arm == "candidate":
                        install(name, artifact_dir)
                    for _ in range(3):
                        p.linear(x, bias, out)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(32):
                            p.linear(x, bias, out)
                    graphs[arm] = graph
                samples = {"base": [], "candidate": []}
                for repeat in range(7):
                    for arm in (("base", "candidate") if repeat % 2 == 0 else ("candidate", "base")):
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record()
                        for _ in range(8):
                            graphs[arm].replay()
                        b.record(); b.synchronize()
                        samples[arm].append(a.elapsed_time(b)*1000/(32*8))
                report[str(bits)][f"{n}x{k}"] = {a: statistics.median(v) for a,v in samples.items()}
                del graphs, p, x, bias, out
        return report
    finally:
        packed.PackedWeight.linear = ORIGINAL


def profile_all(artifact_dir):
    """One rotating sweep over all promising CUDA layouts, with shared inputs.

    This never calls check(), and restores original dispatch before returning.
    Latency-only diagnostics are not acceptance scores; full-model energy is.
    """
    from .verify_cuda import CudaGemvTests
    helper = CudaGemvTests()
    torch.set_num_threads(4)
    torch.manual_seed(781539)
    names = ("base", "pair_fma", "quad_fma", "quad_rows2_fma", "ownedword",
             "quad_fma_b32", "ownedword_b32")
    report = {"timings_us": {}, "weighted_decoder_gemv_ms": {}}
    shapes = ((1024, 1024, 3*24*129), (3072, 1024, 24*129),
              (4096, 1024, 24*129), (1024, 4096, 24*129))
    try:
        for bits in (1, 2):
            report["timings_us"][str(bits)] = {}
            weighted = dict.fromkeys(names, 0.)
            for n, k, count in shapes:
                p, x, bias, _ = helper.inputs(bits, n, k, bias=True)
                out = torch.empty((1, n), device="cuda", dtype=torch.float16)
                graphs = {}
                for name in names:
                    packed.PackedWeight.linear = ORIGINAL
                    if name != "base":
                        install(name, artifact_dir)
                    for _ in range(3):
                        p.linear(x, bias, out)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(32):
                            p.linear(x, bias, out)
                    graphs[name] = graph
                samples = {name: [] for name in names}
                for repeat in range(7):
                    order = names[repeat:] + names[:repeat]
                    for name in order:
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        a.record()
                        for _ in range(8):
                            graphs[name].replay()
                        b.record(); b.synchronize()
                        samples[name].append(a.elapsed_time(b)*1000/(32*8))
                medians = {name: statistics.median(v) for name,v in samples.items()}
                report["timings_us"][str(bits)][f"{n}x{k}"] = medians
                for name in names:
                    weighted[name] += medians[name] * count / 1000
                del graphs, p, x, bias, out
            report["weighted_decoder_gemv_ms"][str(bits)] = weighted
        report["scope"] = "Repeated-weight CUDA graph microtiming; excludes vocab and all other model work"
        return report
    finally:
        packed.PackedWeight.linear = ORIGINAL


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--check", choices=variants())
    parser.add_argument("--profile", choices=variants())
    args = parser.parse_args()
    print(compile_only(args.artifacts), flush=True)
    if args.check:
        report = check(args.check, args.artifacts)
        save(Path(args.artifacts)/f"cuda-{args.check}-check.json", report)
        print(report, flush=True)
    if args.profile:
        report = profile(args.profile, args.artifacts)
        save(Path(args.artifacts)/f"cuda-{args.profile}-profile.json", report)
        print(report, flush=True)
