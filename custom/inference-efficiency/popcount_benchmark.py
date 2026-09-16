"""Compare experimental binary activations with the existing W1/W2A16 kernels.

Run with ./inference-efficiency/python inference-efficiency/popcount_benchmark.py.
Acquires the shared training lock before any CUDA work. Results are kernel
diagnostics, not whole-model energy/accuracy or deployment-hardware predictions.
"""
import argparse
import datetime as dt
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time

import torch

from energy import EnergyMeter
from paths import ROOT, artifact, digest, save, storage
from kernels.packed import PackedWeight
from kernels.popcount import BitWeight, SOURCE, load, pack, pack_plane


HERE = Path(__file__).resolve().parent
LOCK = Path('/mnt/hd/wilderness-labs-stt/stt-distillation/active.lock')
SHAPES = [(1, 1024, 1024), (1, 3072, 1024), (1, 4096, 1024),
          (1, 1024, 4096), (1, 51864, 1024), (4, 1024, 1024),
          (1500, 1024, 1024), (1500, 2048, 1024), (1500, 3072, 1024),
          (1500, 4096, 1024), (1500, 1024, 4096)]
ENERGY_SHAPES = {(1, 1024, 1024), (1, 1024, 4096),
                 (1, 51864, 1024), (1500, 4096, 1024)}


def capture(op, count=128):
    for _ in range(3):
        op()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count):
            op()
    return graph


def timing(graph, count, samples=9):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(samples):
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end) * 1000 / count)
    return times


def verify():
    """Independent CPU dot products, tails, zeros, bias, canaries, and replay."""
    cases = []
    for bits in (1, 2):
        for m, n, k in [(1, 7, 1), (3, 35, 31), (5, 19, 33), (2, 9, 1031),
                        (3, 37, 1024), (4, 33, 4096)]:
            codes = torch.randint(0, 2 if bits == 1 else 3, (n, k), dtype=torch.int8)
            codes = codes * 2 - 1 if bits == 1 else codes - 1
            if bits == 2:
                codes[0].zero_()
            scales = torch.linspace(-.7, 1.3, n) / math.sqrt(k)
            bias = torch.linspace(-.25, .25, n).half()
            x_cpu = torch.randn(m, k).half()
            x_cpu[:, ::7] = 0  # zero -> +1; includes masked ragged tail words
            x = x_cpu.cuda()
            w = BitWeight(codes.cuda(), scales.cuda(), bits)
            a = torch.empty(m, (k+31)//32, dtype=torch.int32, device='cuda')
            guarded = torch.full((m*n+16,), 321., device='cuda', dtype=torch.float16)
            out = guarded[8:-8].view(m, n)
            b = bias.cuda()
            for variant in ('warp', 'tile'):
                def op():
                    pack(x, a)
                    w.linear(a, out, variant, b)
                op()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    op()
                stream.synchronize()
                for change in (False, True):
                    if change:
                        x_cpu.neg_()
                        x.copy_(x_cpu)
                    graph.replay()
                    torch.cuda.synchronize()
                    signs = torch.where(x_cpu >= 0, 1., -1.)
                    expected = ((signs @ codes.float().T) * scales.half().float() + bias.float()).half()
                    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
                    torch.testing.assert_close(a.cpu(), pack_plane(x_cpu >= 0), atol=0, rtol=0)
                    assert bool((guarded[:8] == 321).all() & (guarded[-8:] == 321).all())
                cases.append(dict(bits=bits, m=m, n=n, k=k, variant=variant, exact=True))
    return cases


def write_report(job, rows):
    lines = ['# Binary-activation popcount kernel benchmark', '',
             'Diagnostic only: changes activation precision; no full-model accuracy or energy claim.',
             'CUDA graph timings on repeated, cache-resident weights. Baseline is unchanged PackedWeight.linear with FP16 random inputs.',
             'Candidates use sign(x), with zero mapped to +1. Pack+dot includes FP16 sign packing; prepacked excludes it.',
             'Warp/tile candidate selected in a separate short timing pass before reported measurements.', '',
             '| Bits | M,N,K | Baseline µs | Prepacked µs | Pack+dot µs | Pack+dot speedup | Energy saving, pack+dot |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        t = r['median_us']
        e = r.get('energy', {})
        saving = '—'
        if e:
            vals = [1 - e['pack_dot'][i]['joules_per_call'] / e['baseline'][i]['joules_per_call']
                    for i in range(len(e['baseline']))]
            saving = f'{100*statistics.mean(vals):.1f}% (paired range {100*min(vals):.1f}–{100*max(vals):.1f}%)'
        lines.append(f"| {r['bits']} | {r['m']},{r['n']},{r['k']} | {t['baseline']:.2f} | {t['prepacked']:.2f} | {t['pack_dot']:.2f} | {t['baseline']/t['pack_dot']:.2f}× | {saving} |")
    lines += ['', 'Energy uses three alternating-order sustained windows per arm, with raw NVML telemetry saved.',
              'Energy is GPU board joules per repeated projection, including graph replay/synchronization overhead, not joules per transcription.',
              'These software XOR+popcount prototypes do not establish the best achievable binary Tensor Core performance.',
              'Ternary nonzero/sign planes still cost two bits per weight; binary activations enable popcount arithmetic.',
              'Rotating-weight latency diagnostics are recorded separately in results.json; see working-set bytes and samples there.']
    (job / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def run_case(job, meter, bits, shape, seconds, rounds, cold_mib):
    m, n, k = shape
    codes = torch.randint(0, 2 if bits == 1 else 3, (n, k), device='cuda', dtype=torch.int8)
    codes = codes * 2 - 1 if bits == 1 else codes - 1
    p = PackedWeight.from_codes(codes, bits, 1 / math.sqrt(k))
    w = BitWeight(codes, p.scales, bits)
    x = torch.randn(m, k, device='cuda', dtype=torch.float16)
    a = torch.empty(m, (k+31)//32, device='cuda', dtype=torch.int32)
    out = torch.empty(m, n, device='cuda', dtype=torch.float16)
    pack(x, a)
    x_sign = torch.where(x >= 0, 1., -1.).half()
    expected = p.linear(x_sign)
    selection = {}
    for variant in ('warp', 'tile'):
        w.linear(a, out, variant)
        torch.testing.assert_close(out, expected, atol=.003, rtol=.003)
        g = capture(lambda: w.linear(a, out, variant), 16)
        selection[variant] = statistics.median(timing(g, 16, 5))
        del g
    variant = min(selection, key=selection.get)
    del codes, expected, x_sign

    def pack_dot():
        pack(x, a)
        w.linear(a, out, variant)

    ops = dict(baseline=lambda: p.linear(x, out=out),
               prepacked=lambda: w.linear(a, out, variant), pack_dot=pack_dot,
               pack_only=lambda: pack(x, a))
    # Milliseconds of GPU work per replay amortize host/synchronization costs,
    # including for very short popcount GEMVs.
    count = 2048 if m <= 4 else 32
    graphs = {name: capture(op, count) for name, op in ops.items()}
    samples = {name: [] for name in ops}
    for repeat in range(3):
        names = list(ops)
        if repeat % 2:
            names.reverse()
        meter._guard()
        for name in names:
            samples[name].extend(timing(graphs[name], count))
    result = dict(bits=bits, m=m, n=n, k=k, variant=variant, selection_us=selection,
                  median_us={name: statistics.median(v) for name, v in samples.items()},
                  latency_samples_us=samples, graph_calls=count, energy={})

    if seconds > 0:
        names = ['baseline', 'prepacked', 'pack_dot']
        result['energy'] = {name: [] for name in names}
        for repeat in range(rounds):
            order = names if repeat % 2 == 0 else list(reversed(names))
            for name in order:
                # Warm for a second immediately before each sustained window.
                start = time.monotonic()
                while time.monotonic() - start < 1:
                    graphs[name].replay()
                    torch.cuda.synchronize()
                measurement = meter.measure(graphs[name].replay, min_seconds=seconds,
                                            synchronize=torch.cuda.synchronize)
                measurement['joules_per_call'] = measurement['joules_per_iteration'] / count
                result['energy'][name].append(measurement)
                print(f'{bits=} {shape=} round={repeat} {name}: '
                      f"{measurement['joules_per_call']*1e6:.2f} uJ/call", flush=True)
    del graphs

    if cold_mib:
        # Separate sets for each arm, each > L2 even for the smallest packed
        # matrix. Only the selected layout is read; byte count excludes scales.
        matrix_bytes = n * ((k+31)//32) * 4 * bits
        copies = max(2, math.ceil(cold_mib * 2**20 / matrix_bytes))
        cold = {}
        for name in ('baseline', 'prepacked', 'pack_dot'):
            if name == 'baseline':
                weights = [PackedWeight(p.words.clone(), p.scales, n, k, bits, p.words_t.clone())
                           for _ in range(copies)]
            else:
                weights = []
                for _ in range(copies):
                    clone = object.__new__(BitWeight)
                    clone.__dict__ = w.__dict__.copy()
                    for attr in ('sign', 'sign_t', 'nonzero', 'nonzero_t'):
                        if bits == 1 and attr.startswith('nonzero'):
                            continue
                        setattr(clone, attr, getattr(w, attr).clone())
                    weights.append(clone)
            def cycle():
                for weight in weights:
                    if name == 'baseline':
                        weight.linear(x, out=out)
                    else:
                        if name == 'pack_dot':
                            pack(x, a)
                        weight.linear(a, out, variant)
            g = capture(cycle, 1)
            meter._guard()
            cold[name] = timing(g, copies, 9)
            del g, weights
            gc.collect()
        result['rotating'] = dict(copies=copies, read_weight_bytes=copies*matrix_bytes,
                                 samples_us=cold,
                                 median_us={name: statistics.median(v) for name, v in cold.items()})
    torch.cuda.synchronize()
    meter._guard()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--cold-mib', type=int, default=128)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--energy-all', action='store_true')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    storage()
    job = artifact(args.job or ROOT / 'popcount-benchmark' / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    job.mkdir(parents=True, exist_ok=False)
    sources = ['popcount_benchmark.py', 'kernels/popcount.py', 'kernels/packed.py',
               'kernels/cuda_gemv.py', 'energy.py', 'paths.py']
    for source in sources:
        dest = job / 'source' / source
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / source, dest)
    (job / 'kernel.cu').write_text(SOURCE)
    save(job / 'status.json', dict(status='waiting_for_gpu', pid=os.getpid()))
    print(f'Artifacts: {job}; waiting for shared GPU lock', flush=True)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with EnergyMeter() as meter:
            meter._guard()
            os.environ['EFFICIENCY_REQUIRE_CUDA_GEMV'] = '1'
            torch.manual_seed(9317)
            torch.cuda.init()
            config = dict(gpu=meter.metadata(), source_hashes={s: digest(HERE / s) for s in sources},
                          seconds=args.seconds, rounds=args.rounds, cold_mib=args.cold_mib,
                          energy_shapes=SHAPES if args.energy_all else sorted(ENERGY_SHAPES),
                          shapes=SHAPES, seed=9317,
                          activation_quantization='sign(x), zero -> +1; no activation scale',
                          scope='kernel diagnostics; not full-model inference or accuracy')
            save(job / 'config.json', config)
            save(job / 'status.json', dict(status='checking', pid=os.getpid()))
            load()
            save(job / 'correctness.json', verify())
            if args.check_only:
                save(job / 'status.json', dict(status='checked'))
                return
            rows = []
            for bits in (1, 2):
                for shape in SHAPES:
                    save(job / 'status.json', dict(status='running', bits=bits, shape=shape))
                    seconds = args.seconds if args.energy_all or shape in ENERGY_SHAPES else 0
                    row = run_case(job, meter, bits, shape, seconds, args.rounds, args.cold_mib)
                    rows.append(row)
                    save(job / 'results.json', rows)
                    write_report(job, rows)
                    print(json.dumps({k:v for k,v in row.items() if k in ('bits','m','n','k','median_us')}), flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
            save(job / 'status.json', dict(status='completed', gpu_after=meter.metadata()))
    print(f'Completed: {job / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
