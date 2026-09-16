"""Compile-only CUDA GEMV ablations; --check adds short correctness checks.

These candidates are isolated from inference dispatch. Instruction counts and
resource usage are static diagnostics, not latency or power measurements.
Artifacts go to the mandatory data disk. Run with ./python -m kernels.investigate_cuda.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

import torch
import triton

from paths import ROOT, digest, save, storage
from . import cuda_gemv


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("CUDA source changed; review the ablation before rerunning")
    return source.replace(old, new, 1)


def accumulated_source(count):
    source = cuda_gemv._SOURCE
    start = source.index("template <int BITS, int WPL>")
    end = source.index("// Row-per-lane kernel")
    body = source[start:end]
    body = replace_once(body, "float acc = 0.0f;", f"float accs[{count}] = {{0.0f}};")
    for coeff in ("sgn", "c - 1.0f"):
        body = replace_once(body, f"acc = fmaf(xv, {coeff}, acc);",
                            f"accs[t % {count}] = fmaf(xv, {coeff}, accs[t % {count}]);")
    total = " + ".join(f"accs[{i}]" for i in range(count))
    body = replace_once(body, "    epilogue(acc,", f"    float acc = {total};\n    epilogue(acc,")
    return source[:start] + body + source[end:]


def transposed_source():
    source = cuda_gemv._SOURCE
    start = source.index("// Row-per-lane kernel")
    end = source.index("// General fallback")
    body = source[start:end]
    body = replace_once(body, "    u32* wt = (u32*)(smem + K);\n", "")
    begin = body.index("    const int gi = threadIdx.x & 31;")
    stop = body.index("    __syncthreads();", begin)
    body = body[:begin] + body[stop:]
    body = replace_once(body, "    const u32* wrow = wt + rl * stride;\n", "")
    body = replace_once(body, "        const u32 wv = wrow[t];",
                        "        const u32 wv = row < N ? __ldg(W + (long long)t * N + row) : 0u;")
    return source[:start] + body + source[end:]


_MULTIWARP = r"""
// Four warps cooperate on one output, with an in-block deterministic reduction.
// No global workspace, atomics, or extra kernel launch. General K is supported.
template <int BITS>
__device__ __forceinline__ void gemv_split4_impl(GEMV_ARGS) {
    const int row = blockIdx.x, m = blockIdx.y;
    const int gi = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int VPW = 32 / BITS;
    __shared__ float partials[4];
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += 128) {
        const u32 w = __ldg(W + (long long)row * KW + k / VPW);
        const u32 c = (w >> ((k % VPW) * BITS)) & ((1u << BITS) - 1u);
        const float code = BITS == 1 ? 2.0f * (float)c - 1.0f : (float)c - 1.0f;
        acc = fmaf(h2f(__ldg(X + (long long)m * K + k)), code, acc);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (gi == 0) partials[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        const float total = (partials[0] + partials[1]) + (partials[2] + partials[3]);
        store_result(total, S, Bias, Y, row, m, N, has_bias);
    }
}
extern "C" __global__ void __launch_bounds__(128) gemv_b1_split4(GEMV_ARGS) {
    gemv_split4_impl<1>(X, W, S, Bias, Y, N, K, KW, has_bias);
}
extern "C" __global__ void __launch_bounds__(128) gemv_b2_split4(GEMV_ARGS) {
    gemv_split4_impl<2>(X, W, S, Bias, Y, N, K, KW, has_bias);
}
"""


def static_metrics(resources, sass):
    metrics = {}
    chunks = re.split(r"Function : (\w+)", sass)
    for name, body in zip(chunks[1::2], chunks[2::2]):
        instructions = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)\s", body)
        attrs = re.search(rf"Function {name}:\s+([^\n]+)", resources)[1]
        metrics[name] = {
            "registers_per_thread": int(re.search(r"REG:(\d+)", attrs)[1]),
            "local_bytes": int(re.search(r"LOCAL:(\d+)", attrs)[1]),
            "stack_bytes": int(re.search(r"STACK:(\d+)", attrs)[1]),
            "static_instruction_count": len(instructions),
            "instructions": dict(Counter(instructions)),
        }
    return metrics


def check_candidate(candidate, cubin):
    from torch.cuda._utils import _cuda_load_module
    from .verify_cuda import CudaGemvTests
    helper = CudaGemvTests()
    names = ([f"gemv_b{b}_split4" for b in (1, 2)] if candidate == "split4" else
             [f"gemv_b{b}_rpl" for b in (1, 2)] if candidate == "transposed_rpl" else
             [f"gemv_b{b}_w{w}" for b, ws in ((1, (1, 2, 4)), (2, (1, 2, 4, 8))) for w in ws])
    kernels = _cuda_load_module(cubin, names)
    count, worst = 0, 0.0
    for name, kernel in kernels.items():
        bits = int(name[6])
        ks = ([32, 1024] if candidate == "transposed_rpl" else
              [73, 1024, 4096] if candidate == "split4" else [int(name.rsplit("w", 1)[1]) * 1024 // bits])
        for k in ks:
            for m in (1, 4):
                n = 129 if candidate == "transposed_rpl" else 19
                p, x, bias, expected = helper.inputs(bits, n, k, m, bias=True)
                out = torch.empty((m, n), device="cuda", dtype=torch.float16)
                words = p.words_t if candidate == "transposed_rpl" else p.words
                rows = 128 if candidate == "transposed_rpl" else 1 if candidate == "split4" else 4
                shared = 4 * k if candidate == "transposed_rpl" else 0

                def launch():
                    kernel(grid=((n + rows - 1) // rows, m, 1), block=(128, 1, 1),
                           args=[x, words, p.scales, bias, out, n, k, p.words.shape[1], 1],
                           shared_mem=shared)

                launch()
                actual = out.cpu()
                torch.testing.assert_close(actual, expected, atol=.003, rtol=.003)
                worst = max(worst, float((actual.float() - expected.float()).abs().max()))
                # The global buffer-free reduction must remain capture-safe.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                x.zero_()
                graph.replay()
                torch.testing.assert_close(out.cpu(), bias.cpu().expand_as(expected), atol=0, rtol=0)
                count += 1
    return {"cases": count, "max_absolute_error": worst, "changed_input_graph_replay": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    storage()
    folder = ROOT / "investigations" / "cuda-gemv-followup"
    folder.mkdir(parents=True, exist_ok=True)
    cuda_gemv._preload_nvrtc()
    from torch.cuda._utils import _nvrtc_compile
    disassembler = Path(triton.__file__).parent / "backends/nvidia/bin/cuobjdump"
    sources = {"baseline": cuda_gemv._SOURCE, "acc2": accumulated_source(2),
               "acc4": accumulated_source(4), "transposed_rpl": transposed_source(),
               "split4": cuda_gemv._SOURCE + _MULTIWARP}
    report = {"target": "sm_120", "performance_measured": False,
              "source_sha256": digest(Path(cuda_gemv.__file__)), "candidates": {}}
    for name, source in sources.items():
        cubin, _ = _nvrtc_compile(source, "gemv_b2_w8", compute_capability="120")
        path = folder / f"{name}.cubin"
        path.write_bytes(cubin)
        (folder / f"{name}.cu").write_text(source)
        dumps = []
        for flag, suffix in (("--dump-resource-usage", "resources.txt"), ("--dump-sass", "sass")):
            text = subprocess.run([str(disassembler), flag, str(path)],
                                  text=True, capture_output=True, check=True).stdout
            (folder / f"{name}.{suffix}").write_text(text)
            dumps.append(text)
        row = {"kernels": static_metrics(*dumps)}
        if args.check:
            row["correctness"] = check_candidate(name, cubin)
        report["candidates"][name] = row
        print(name, row.get("correctness", "compiled; GPU execution disabled"), flush=True)
    if args.check:
        props = torch.cuda.get_device_properties(0)
        report["device"] = {"name": props.name, "sm_count": props.multi_processor_count}
    save(folder / "inspection.json", report)
    print(folder / "inspection.json")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
