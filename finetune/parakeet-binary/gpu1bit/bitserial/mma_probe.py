"""Does SM120 (RTX 5090) execute mma.sync .b1 natively, and how fast vs INT8 / BF16 / FP8 mma?

Agent C, step 1 of plans/gpu-kernel.md. For each warp-level MMA variant, a kernel issues ILP
independent accumulator chains per warp over many iterations (operands in registers, no memory
traffic) and the grid fills every SM; throughput = warp MMAs per second, also reported as dense
"ops" (2*m*n*k) and as K-elements per warp-instruction. CUDA-core FFMA / LOP3 / POPC issue rates
are measured the same way for the small-M SIMT design. The b1 .and.popc / .xor.popc semantics
and the fragment layout used later are checked against a CPU reference. SASS of every probe is
saved and its tensor instructions are listed, which shows whether b1 is native (BMMA) or lowered
to something else.

Run (needs NVRTC; the ternary wrapper provides it):
  finetune/parakeet-ternary/python finetune/parakeet-binary/gpu1bit/bitserial/mma_probe.py
Holds paths.gpu_lock("C:mma-probe") for the whole process (well under 10 minutes).
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'parakeet-ternary'))
sys.path.insert(0, str(HERE.parents[1]))  # parakeet-binary: gpu1bit is a namespace package
import paths  # noqa: E402  (parakeet-ternary paths.py: gpu_lock)
from gpu1bit.bitserial import nvrtc  # noqa: E402

# name: (ptx shape/type string, a regs, b regs, c regs, c type, m, n, k, operand init kind)
MMAS = {
    's8_k32': ('m16n8k32.row.col.s32.s8.s8.s32', 4, 2, 4, 'int', 16, 8, 32, 'int'),
    's4_k64': ('m16n8k64.row.col.s32.s4.s4.s32', 4, 2, 4, 'int', 16, 8, 64, 'int'),
    'b1_and_k256': ('m16n8k256.row.col.s32.b1.b1.s32.and.popc', 4, 2, 4, 'int', 16, 8, 256, 'int'),
    'b1_xor_k256': ('m16n8k256.row.col.s32.b1.b1.s32.xor.popc', 4, 2, 4, 'int', 16, 8, 256, 'int'),
    'b1_and_k128': ('m16n8k128.row.col.s32.b1.b1.s32.and.popc', 2, 1, 4, 'int', 16, 8, 128, 'int'),
    'bf16_k16_f32acc': ('m16n8k16.row.col.f32.bf16.bf16.f32', 4, 2, 4, 'float', 16, 8, 16, 'bf16'),
    'f16_k16_f16acc': ('m16n8k16.row.col.f16.f16.f16.f16', 4, 2, 2, 'half2', 16, 8, 16, 'f16'),
    'e4m3_k32_f32acc': ('m16n8k32.row.col.f32.e4m3.e4m3.f32', 4, 2, 4, 'float', 16, 8, 32, 'fp8'),
    'tf32_k8_f32acc': ('m16n8k8.row.col.f32.tf32.tf32.f32', 4, 2, 4, 'float', 16, 8, 8, 'tf32'),
}

ILP = 8


def mma_source(name: str) -> str:
    shape, na, nb, nc, ctype, *_rest, init = MMAS[name]
    c_kind = {'int': 'r', 'float': 'f', 'half2': 'r'}[ctype]
    c_decl = {'int': 'int', 'float': 'float', 'half2': 'u32'}[ctype]
    regs = [f'%{i}' for i in range(nc + na + nb)]
    c_ops = '{' + ','.join(regs[:nc]) + '}'
    a_ops = '{' + ','.join(regs[nc:nc + na]) + '}'
    b_ops = '{' + ','.join(regs[nc + na:]) + '}'
    outs = ', '.join(f'"+{c_kind}"(c[{i}])' for i in range(nc))
    ins = ', '.join([f'"r"(a[{i}])' for i in range(na)] + [f'"r"(b[{i}])' for i in range(nb)])
    # Valid finite operands for float types (bit patterns of small normal numbers).
    base = {'int': '0x9E3779B9u', 'bf16': '0x3F803F80u', 'f16': '0x3C003C00u',
            'fp8': '0x38383838u', 'tf32': '0x3F800000u'}[init]
    jitter = '(t & 7u)' if init != 'int' else '(t * 0x85EBCA6Bu)'
    return f'''
typedef unsigned int u32;
extern "C" __global__ void probe(u32 *out, int iters) {{
    const u32 t = threadIdx.x + blockIdx.x * blockDim.x;
    u32 a[{na}], b[{nb}];
    #pragma unroll
    for (int i = 0; i < {na}; ++i) a[i] = {base} ^ ({jitter} << i);
    #pragma unroll
    for (int i = 0; i < {nb}; ++i) b[i] = {base} ^ ({jitter} << (i + 2));
    {c_decl} acc[{ILP}][{nc}];
    #pragma unroll
    for (int j = 0; j < {ILP}; ++j)
        #pragma unroll
        for (int i = 0; i < {nc}; ++i) acc[j][i] = 0;
    for (int it = 0; it < iters; ++it) {{
        #pragma unroll
        for (int j = 0; j < {ILP}; ++j) {{
            {c_decl} *c = acc[j];
            asm volatile("mma.sync.aligned.{shape} {c_ops}, {a_ops}, {b_ops}, {c_ops};"
                         : {outs} : {ins});
        }}
    }}
    u32 s = 0;
    #pragma unroll
    for (int j = 0; j < {ILP}; ++j)
        #pragma unroll
        for (int i = 0; i < {nc}; ++i) {{
            {c_decl} v = acc[j][i];
            s ^= *reinterpret_cast<u32 *>(&v);
        }}
    out[t] = s;
}}
'''


# CUDA-core issue-rate probes: each "op" is one instruction per thread on ILP chains.
SIMT = {
    'ffma': ('float', 'x[j] = fmaf(x[j], 0.999f, y);'),
    'lop3': ('u32', 'asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(x[j]) : "r"(y), "r"(z));'),
    'popc': ('u32', 'asm volatile("popc.b32 %0, %0;" : "+r"(x[j]));'),
    'iadd': ('u32', 'asm volatile("add.u32 %0, %0, %1;" : "+r"(x[j]) : "r"(y));'),
}


def simt_source(name: str) -> str:
    ctype, body = SIMT[name]
    decl = 'float' if ctype == 'float' else 'u32'
    init = '1.0f + 1e-3f * (float)(t & 15)' if ctype == 'float' else '0x9E3779B9u ^ t'
    return f'''
typedef unsigned int u32;
extern "C" __global__ void probe(u32 *out, int iters) {{
    const u32 t = threadIdx.x + blockIdx.x * blockDim.x;
    {decl} x[{ILP}];
    {decl} y = ({decl})(t & 3); u32 z = t * 7u;
    (void)z;
    #pragma unroll
    for (int j = 0; j < {ILP}; ++j) x[j] = {init} + ({decl})j;
    for (int it = 0; it < iters; ++it) {{
        #pragma unroll
        for (int j = 0; j < {ILP}; ++j) {{ {body} }}
    }}
    u32 s = 0;
    #pragma unroll
    for (int j = 0; j < {ILP}; ++j) s ^= *reinterpret_cast<u32 *>(&x[j]);
    out[t] = s;
}}
'''


B1_CHECK = r'''
typedef unsigned int u32;
extern "C" __global__ void check(const u32 *A, const u32 *B, int *C, int op) {
    // One warp. A: 16 rows x 256 bits (8 words/row); B: 8 columns x 256 bits (8 words/col).
    // Thread fragment (PTX m16n8k256 .b1): a0 row g, bits [32t, 32t+32); a1 row g+8 same bits;
    // a2/a3 the same rows at bits +128; b0 column g bits [32t, 32t+32), b1 at bits +128.
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    u32 a0 = A[g * 8 + t], a1 = A[(g + 8) * 8 + t], a2 = A[g * 8 + 4 + t], a3 = A[(g + 8) * 8 + 4 + t];
    u32 b0 = B[g * 8 + t], b1 = B[g * 8 + 4 + t];
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    if (op == 0)
        asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    else
        asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    // C fragment: c0,c1 row g, columns 2t, 2t+1; c2,c3 row g+8.
    C[g * 8 + 2 * t] = c0; C[g * 8 + 2 * t + 1] = c1;
    C[(g + 8) * 8 + 2 * t] = c2; C[(g + 8) * 8 + 2 * t + 1] = c3;
}
'''


def tensor_instructions(text: str) -> dict[str, int]:
    """Count SASS opcodes that look like tensor-core or bit ops."""
    counts: dict[str, int] = {}
    for line in text.splitlines():
        m = re.search(r'/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)', line)
        if not m:
            continue
        op = m.group(1)
        if re.match(r'(HMMA|IMMA|BMMA|QMMA|OMMA|HGMMA|UTC|MMA|POPC|LOP3|FFMA|IADD3|IMAD|BMSK|FLO)', op):
            counts[op] = counts.get(op, 0) + 1
    return counts


def timed(kernel, grid, block, iters, out, repeats=5):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    kernel(grid=(grid, 1, 1), block=(block, 1, 1), args=[out, 2])  # warm
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start.record()
        kernel(grid=(grid, 1, 1), block=(block, 1, 1), args=[out, iters])
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) / 1e3)
    return min(times), statistics.median(times)


def gpu_state() -> dict:
    query = 'name,power.limit,clocks.sm,clocks.max.sm,clocks.applications.graphics,temperature.gpu,driver_version'
    text = subprocess.run(['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,name', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip()
    return {'query': query, 'values': text, 'compute_apps': apps}


def main():
    nvrtc.setup()
    out_dir = nvrtc.ARTIFACTS / f'{time.strftime("%Y%m%dT%H%M%S")}-mma-probe'
    out_dir.mkdir(parents=True, exist_ok=False)
    print('RESULTS', out_dir, flush=True)
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    results = {'device': props.name, 'sms': sms, 'capability': list(torch.cuda.get_device_capability()),
               'torch': torch.__version__, 'ilp': ILP, 'wrapper': 'finetune/parakeet-ternary/python',
               'mma': {}, 'simt': {}, 'b1_semantics': {}}
    with paths.gpu_lock('C:mma-probe'):
        t0 = time.time()
        results['gpu_before'] = gpu_state()
        if '400.00 W' not in results['gpu_before']['values']:
            raise RuntimeError(f'unexpected power limit: {results["gpu_before"]}')
        out = torch.empty(sms * 8 * 1024, dtype=torch.int32, device='cuda')
        # Semantics + layout check of b1 .and.popc / .xor.popc against a CPU reference.
        g = torch.Generator().manual_seed(20261020)
        A = torch.randint(0, 2**31, (16, 8), generator=g, dtype=torch.int64) * 2 + torch.randint(0, 2, (16, 8), generator=g)
        B = torch.randint(0, 2**31, (8, 8), generator=g, dtype=torch.int64) * 2 + torch.randint(0, 2, (8, 8), generator=g)
        bits = lambda w: ((w[..., None] >> torch.arange(32)) & 1).flatten(-2)  # noqa: E731
        Ab, Bb = bits(A), bits(B)  # [16,256], [8,256]
        expect = {'and': Ab @ Bb.T, 'xor': (Ab[:, None, :] ^ Bb[None, :, :]).sum(-1)}
        as_i32 = lambda w: ((w + 2**31) % 2**32 - 2**31).to(torch.int32).cuda()  # noqa: E731
        A32, B32 = as_i32(A), as_i32(B)
        for op_index, op in enumerate(('and', 'xor')):
            try:
                kernel, cubin = nvrtc.load(B1_CHECK, 'check')
                C = torch.zeros(16, 8, dtype=torch.int32, device='cuda')
                kernel(grid=(1, 1, 1), block=(32, 1, 1), args=[A32, B32, C, op_index])
                torch.cuda.synchronize()
                ok = bool(torch.equal(C.cpu().long(), expect[op]))
                results['b1_semantics'][op] = {'matches_cpu_popc': ok}
            except Exception as error:  # compile or launch failure is itself a finding
                results['b1_semantics'][op] = {'error': repr(error)[:2000]}
            print('B1-SEMANTICS', op, results['b1_semantics'][op], flush=True)
        # Throughput.
        for name, spec in MMAS.items():
            m, n, k = spec[5:8]
            row = {'ptx': 'mma.sync.aligned.' + spec[0], 'm': m, 'n': n, 'k': k}
            try:
                kernel, cubin = nvrtc.load(mma_source(name), 'probe')
                text = nvrtc.sass(cubin)
                (out_dir / f'{name}.sass').write_text(text)
                row['sass_counts'] = tensor_instructions(text)
                best = None
                for warps_per_block, blocks_per_sm in ((4, 4), (8, 2), (8, 4), (16, 2)):
                    block = warps_per_block * 32
                    grid = blocks_per_sm * sms
                    iters = 4096
                    tmin, tmed = timed(kernel, grid, block, iters, out)
                    warp_mmas = grid * warps_per_block * iters * ILP
                    rate = warp_mmas / tmin
                    cand = {'warps_per_block': warps_per_block, 'blocks_per_sm': blocks_per_sm,
                            'seconds_min': tmin, 'seconds_median': tmed,
                            'warp_mma_per_s': rate, 'warp_mma_per_sm_per_us': rate / sms / 1e6,
                            'dense_tops': rate * 2 * m * n * k / 1e12,
                            'k_elements_per_s_per_row_col': rate * k}
                    row.setdefault('configs', []).append(cand)
                    if best is None or rate > best['warp_mma_per_s']:
                        best = cand
                row['best'] = best
            except Exception as error:
                row['error'] = repr(error)[:3000]
            results['mma'][name] = row
            b = row.get('best')
            print('MMA', name, row.get('sass_counts'), 'TOPS=%.1f' % b['dense_tops'] if b else row.get('error', '')[:400],
                  'warpMMA/SM/us=%.2f' % b['warp_mma_per_sm_per_us'] if b else '', flush=True)
        for name in SIMT:
            kernel, cubin = nvrtc.load(simt_source(name), 'probe')
            text = nvrtc.sass(cubin)
            (out_dir / f'simt_{name}.sass').write_text(text)
            best = None
            for warps_per_block, blocks_per_sm in ((8, 4), (16, 2), (32, 1)):
                block = warps_per_block * 32
                grid = blocks_per_sm * sms
                iters = 8192
                tmin, tmed = timed(kernel, grid, block, iters, out)
                thread_ops = grid * block * iters * ILP / tmin
                cand = {'warps_per_block': warps_per_block, 'blocks_per_sm': blocks_per_sm, 'seconds_min': tmin,
                        'thread_ops_per_s': thread_ops, 'per_sm_per_clock_at_max_clock': None}
                if best is None or thread_ops > best['thread_ops_per_s']:
                    best = cand
            results['simt'][name] = {'best': best, 'sass_counts': tensor_instructions(text)}
            print('SIMT', name, '%.2f T/s' % (best['thread_ops_per_s'] / 1e12), results['simt'][name]['sass_counts'], flush=True)
        results['gpu_after'] = gpu_state()
        results['lock_seconds'] = time.time() - t0
    (out_dir / 'results.json').write_text(json.dumps(results, indent=1))
    (out_dir / 'mma_probe.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: (v['best']['dense_tops'] if v.get('best') else v.get('error', '')[:200])
                      for k, v in results['mma'].items()}, indent=1))
    print('gpu_after', results['gpu_after'])


if __name__ == '__main__':
    main()
