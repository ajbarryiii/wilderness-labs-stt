"""Agent C screening: per-shape config sweep of the bitserial FP32 SIMT kernel, and M1 baselines
on the same MB1 weights (derived 2-bit layout), as warm CUDA-graph replay latency.

This is C's own screening tool (phase 1/2), not H's recorded harness. Inputs are Gaussian with the
pipeline's input layouts (linear contiguous [1, M, K]; pointwise_conv1 a transposed view of
[1, T, K]; pointwise_conv2 contiguous [1, K, T]); weights are real MB1 layer-0 codes. Every op is
timed as called (INT8x3 baselines include their activation-quantization kernel; no norm fusion).

  # CPU only, no lock: fill the CUBIN cache
  finetune/parakeet-binary/gpu1bit/shared/python -m gpu1bit.bitserial.tune precompile
  # GPU: one shape group per process, under the lock
  timeout 600 finetune/parakeet-binary/gpu1bit/shared/python finetune/parakeet-binary/gpu1bit/bitserial/run_locked.py \
      C:tune -m gpu1bit.bitserial.tune sweep --shapes qkv --m 38 126 376
"""
from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import statistics
import time
from pathlib import Path

import torch

from gpu1bit.shared import api, env

from . import kernels, nvrtc
from .kernels import Config
from .layer import BitSerialLayer, choose

# name -> (kind, N, K, layer-0 module names whose codes are concatenated along N, input layout)
SHAPES = {
    'qkv': ('linear', 3072, 1024, ['self_attn.linear_q', 'self_attn.linear_k', 'self_attn.linear_v'], 'rows'),
    'out': ('linear', 1024, 1024, ['self_attn.linear_out'], 'rows'),
    'up': ('linear', 4096, 1024, ['feed_forward1.linear1'], 'rows'),
    'down': ('linear', 1024, 4096, ['feed_forward1.linear2'], 'rows'),
    'pw1': ('conv', 2048, 1024, ['conv.pointwise_conv1'], 'conv_view'),
    'pw2': ('conv', 1024, 1024, ['conv.pointwise_conv2'], 'conv_contiguous'),
}


def sweep_configs(m: int) -> list[Config]:
    out = []
    for (bm, tm), (bn, tn), bk, split in itertools.product(
            [(16, 4), (32, 4), (32, 8), (40, 4), (40, 8), (48, 4), (64, 8), (96, 8), (128, 8)],
            [(64, 8), (128, 8), (256, 8), (128, 4), (64, 4)],
            [32, 64], [1, 2, 3, 4, 6, 8]):
        c = Config(bm, bn, bk, tm, tn, split)
        if not c.valid() or not 64 <= c.threads <= 512:
            continue
        if bm > 2 * max(16, m) or (m > 64 and bm < 32):  # tiles far taller than M waste work
            continue
        out.append(c)
    return out


def xmode_for(layout: str) -> int:
    return {'rows': 1, 'conv_view': 1, 'conv_contiguous': 2}[layout]


def _precompile_one(args):
    config, xmode = args
    return kernels.precompile(config, xmode=xmode, wfast=True, has_bias=False)


def precompile(ms: list[int], workers: int) -> None:
    jobs = sorted({(c, xm) for m in ms for c in sweep_configs(m) for xm in (1, 2)}, key=repr)
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        for i, size in enumerate(pool.map(_precompile_one, jobs, chunksize=4)):
            if i % 50 == 0:
                print(f'precompiled {i}/{len(jobs)} ({time.time() - started:.0f} s)', flush=True)
    print(f'precompiled {len(jobs)} kernels in {time.time() - started:.0f} s', flush=True)


def load_codes(modules: list[str], layer: int = 0):
    from safetensors import safe_open
    manifest = json.loads((env.MB1_EXPORT / 'manifest.json').read_text())
    codes, scales = [], []
    with safe_open(env.MB1_EXPORT / manifest['file'], framework='pt') as f:
        for module in modules:
            name = f'encoder.layers.{layer}.{module}'
            codes.append(f.get_tensor(name + '.codes'))
            scales.append(f.get_tensor(name + '.scale'))
    return torch.cat(codes), torch.cat(scales)


def make_input(layout: str, m: int, k: int) -> torch.Tensor:
    if layout == 'rows':
        return torch.randn(1, m, k, device='cuda')
    if layout == 'conv_view':
        return torch.randn(1, m, k, device='cuda').transpose(1, 2)
    return torch.randn(1, k, m, device='cuda')


def graph_time(fn, *, calls: int = 20, windows: int = 7, replays: int = 10) -> dict:
    """Median per-call microseconds of `calls` back-to-back ops captured in one CUDA graph."""
    fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    per_call = []
    for _ in range(windows):
        start.record()
        for _ in range(replays):
            graph.replay()
        stop.record()
        stop.synchronize()
        per_call.append(start.elapsed_time(stop) * 1e3 / (replays * calls))
    del graph
    return {'median_us': statistics.median(per_call), 'min_us': min(per_call), 'windows_us': per_call}


class Baselines:
    """M1's kernels on MB1's derived 2-bit layout (code 01 = +1, 10 = -1; no zeros)."""

    def __init__(self):
        from inference.encoder4 import ExpandedProjection
        from inference.ptx_cuda import PtxCudaProjection
        self.expanded = ExpandedProjection('triton_col_tuned')
        self.hybrid = PtxCudaProjection(tuning='native', row_warps='adaptive')

    @staticmethod
    def module(codes, scale, k, kind):
        from export import pack_codes
        from inference.runtime import PackedLinear, PackedPointwiseConv1d
        signs = api.unpack_signs(codes, k)
        cls = PackedPointwiseConv1d if kind == 'conv' else PackedLinear
        return cls(pack_codes(signs), scale.clone(), k).cuda()

    def ops(self, mod, kind):
        from inference.fused_kernels import int8_matmul
        from inference.kernels import matmul
        conv = kind == 'conv'
        return {
            'R-bf16x3': lambda x: matmul(x, mod.packed_t, mod.scale, mod.bias, mod.in_features, conv=conv, mode='bf16x3'),
            'R-compact(int8x3)': lambda x: int8_matmul(x, mod, conv, 3),
            'R-expanded(int8x3)': lambda x: self.expanded(x, mod, conv, 3),
            'R-hybrid-dot(ptx_row_best)': lambda x: self.hybrid(x, mod, conv, 3),
        }


def sweep(shape_names: list[str], ms: list[int], out_dir: Path, baselines: bool, top: int, state) -> list[dict]:
    rows = []
    base = Baselines() if baselines else None
    for name in shape_names:
        kind, n, k, modules, layout = SHAPES[name]
        codes, scale = load_codes(modules)
        layer = BitSerialLayer.prepare(codes, scale, None, k, 'cuda')
        mod = base.module(codes, scale, k, kind) if base else None
        for m in ms:
            env.check_hold(state, margin_s=60)
            x = make_input(layout, m, k)
            ref = api.reference(kind, x, codes, scale, None, k)
            call = (lambda x: layer.conv(x)) if kind == 'conv' else (lambda x: layer.linear(x))
            results = []
            for config in sweep_configs(m):
                layer.config_override = config
                y = call(x)
                err = api.relative_error(y, ref)
                timing = graph_time(lambda: call(x), windows=3, replays=5)
                results.append({'config': list(config.__dict__.values()), 'median_us': timing['median_us'], 'rel_err': err})
            results.sort(key=lambda r: r['median_us'])
            # Re-time the leaders more carefully.
            for r in results[:top]:
                layer.config_override = Config(*r['config'])
                r.update({'retimed': graph_time(lambda: call(x))})
            best = min(results[:top], key=lambda r: r['retimed']['median_us'])
            layer.config_override = None
            default = choose(m, n, k)
            row = {'shape': name, 'kind': kind, 'M': m, 'N': n, 'K': k, 'layout': layout,
                   'input_stride': list(x.stride()), 'best': best, 'default_config': list(default.__dict__.values()),
                   'default_us': graph_time(lambda: call(x))['median_us'], 'candidates': results}
            if base:
                row['baselines'] = {}
                for label, op in base.ops(mod, kind).items():
                    try:
                        y = op(x)
                        row['baselines'][label] = {'rel_err': api.relative_error(y, ref), **graph_time(lambda: op(x))}
                    except Exception as error:  # report, never hide
                        row['baselines'][label] = {'error': repr(error)[:500]}
            rows.append(row)
            summary = {label: round(v.get('median_us', float('nan')), 2) for label, v in row.get('baselines', {}).items()}
            print(f"SHAPE {name} M={m} N={n} K={k} best={best['config']} {best['retimed']['median_us']:.2f} us "
                  f"err={best['rel_err']:.2e} default={row['default_us']:.2f} us baselines={summary}", flush=True)
            (out_dir / 'sweep.json').write_text(json.dumps(rows, indent=1))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['precompile', 'sweep'])
    parser.add_argument('--shapes', nargs='+', default=list(SHAPES))
    parser.add_argument('--m', nargs='+', type=int, default=[38, 126, 368])  # dev clips (H, workloads.json)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--no-baselines', action='store_true')
    parser.add_argument('--top', type=int, default=5)
    args = parser.parse_args()
    if args.mode == 'precompile':
        precompile(args.m, args.workers)
        return
    if not getattr(env.gpu_lock, '_held', False):
        raise SystemExit('run under run_locked.py (holds the GPU lock)')
    out_dir = nvrtc.ARTIFACTS / f'{time.strftime("%Y%m%dT%H%M%S")}-tune-{"-".join(args.shapes)}'
    out_dir.mkdir(parents=True)
    print('RESULTS', out_dir, flush=True)
    (out_dir / 'meta.json').write_text(json.dumps({
        'args': vars(args), 'wrapper': env.wrapper(), 'torch': torch.__version__,
        'sources': env.source_hashes(Path(__file__).parent), 'scope': 'C screening; warm CUDA-graph replay of '
        '20 back-to-back calls; Gaussian inputs; MB1 layer-0 weights; L2-resident weights'}, indent=1))
    import os
    state = {'acquired': float(os.environ['C_LOCK_ACQUIRED_MONOTONIC'])}
    with torch.inference_mode():
        torch.backends.cuda.matmul.allow_tf32 = False
        sweep(args.shapes, args.m, out_dir, not args.no_baselines, args.top, state)


if __name__ == '__main__':
    main()
