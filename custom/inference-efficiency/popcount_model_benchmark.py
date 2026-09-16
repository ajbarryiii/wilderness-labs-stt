"""Fixed-work whole-model screen of binary-activation popcount projections.

Uses the same seeded weights, 16 clips and 128 forced tokens as benchmark.py.
Activations entering packed linear/convolution projections become sign(x).
Embedding lookup, attention, normalization and residuals retain their runtime.
This is a structurally changed random model, not equivalent/accurate inference.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from paths import ROOT, artifact, digest, save, storage

HERE = Path(__file__).resolve().parent
LOCK = Path('/mnt/hd/wilderness-labs-stt/stt-distillation/active.lock')


def install(policy, ternary_activations=False):
    import torch
    from kernels.packed import PackedWeight
    if ternary_activations:
        from kernels.popcount_ternary_activation import BitWeight, load, pack
    else:
        from kernels.popcount import BitWeight, load, pack
    load()
    weights = {}

    def linear(self, x, bias=None, out=None):
        if id(self) not in weights:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Popcount preparation must precede graph capture')
            kidx = torch.arange(self.k, device=self.words.device)
            encoded = (self.words[:, kidx // (32 // self.bits)] >>
                       ((kidx % (32 // self.bits)) * self.bits)) & ((1 << self.bits) - 1)
            codes = (encoded * 2 - 1 if self.bits == 1 else encoded - 1).to(torch.int8)
            # PackedWeight is a frozen dataclass. Keep a strong reference with
            # each cached conversion so object IDs cannot be recycled.
            weights[id(self)] = (self, BitWeight(codes, self.scales, self.bits))
        shape = (*x.shape[:-1], self.n)
        x = x.reshape(-1, self.k).contiguous()
        if out is None:
            out = torch.empty(shape, device=x.device, dtype=x.dtype)
        a = torch.empty(x.shape[0], (self.k+31)//32, device=x.device,
                        dtype=torch.int64 if ternary_activations else torch.int32)
        pack(x, a)
        key = f'{self.bits},{x.shape[0]},{self.n},{self.k}'
        variant = policy.get(key, 'warp' if x.shape[0] <= 4 else 'tile')
        weights[id(self)][1].linear(a, out.view(-1, self.n), variant, bias)
        return out

    PackedWeight.linear = linear


def worker(job, bits, arm, repeat):
    import numpy as np
    import torch
    import benchmark
    from energy import EnergyMeter
    cfg = json.loads((job / 'config.json').read_text())
    for source, expected in cfg['source_hashes'].items():
        if digest(HERE / source) != expected:
            raise RuntimeError(f'Source changed: {source}')
    if digest(ROOT / 'data/manifest.json') != cfg['manifest_sha256']:
        raise RuntimeError('Workload manifest changed')
    os.environ['EFFICIENCY_REQUIRE_CUDA_GEMV'] = '1'
    torch.set_grad_enabled(False)
    with EnergyMeter(synchronize=torch.cuda.synchronize) as meter:
        meter._guard()
        before = meter.metadata()
        if any(before[k] != cfg['gpu'][k] for k in ('uuid', 'power_limit_w')):
            raise RuntimeError('GPU identity or power limit changed')
        if arm != 'baseline':
            install(cfg['ternary_policy'] if arm == 'popcount_a2' else cfg['policy'],
                    ternary_activations=arm == 'popcount_a2')
        manifest = benchmark.workload_manifest()
        model, call, infer, mel = benchmark.make_runner(
            ('binary' if bits == 1 else 'ternary') + '-packed', manifest,
            graph=True, seed=20260911)
        for _ in manifest['clips']:
            output = call()
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError('Nonfinite full-model output')
        torch.cuda.synchronize()
        times = []
        def cycle():
            for _ in manifest['clips']:
                start = time.perf_counter()
                call()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        result = meter.measure(cycle, min_seconds=cfg['seconds'])
        row = dict(bits=bits, arm=arm, repeat=repeat, clips=len(times),
                   median_ms=statistics.median(times)*1000,
                   p95_ms=float(np.percentile(times, 95))*1000,
                   joules_per_clip=result['energy_joules']/len(times),
                   average_watts=result['average_watts'], clip_seconds=times,
                   gpu_before=before, gpu_after=meter.metadata(), measurement=result,
                   scope='Changed activation precision; no recognition accuracy claim')
        save(job / f'{repeat:02d}-b{bits}-{arm}.json', row)
        print(json.dumps({k: row[k] for k in ('bits','arm','repeat','median_ms','joules_per_clip')}), flush=True)


def report(job):
    rows = [json.loads(p.read_text()) for p in sorted(job.glob('[0-9][0-9]-b*.json'))]
    result = {}
    lines = ['# Whole-model binary-activation screen', '',
             'Same seeded random Whisper medium.en weights, 16 frozen 30-second clips and 128 forced decoder steps.',
             'popcount applies sign(x); popcount_a2 applies +1 for x>=.5, -1 for x<=-.5, else zero.',
             'Both include activation packing before every packed projection/convolution.',
             'Attention, embeddings, residuals and normalization remain in the original runtime.',
             'Changed activation precision: no accuracy or numerically equivalent inference claim.', '',
             '| Bits | Arm | Windows | Pooled median ms | Pooled p95 ms | Mean J/clip | Mean W |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    import numpy as np
    for bits in (1,2):
        for arm in ('baseline','popcount','popcount_a2'):
            selected = [r for r in rows if r['bits']==bits and r['arm']==arm]
            if not selected:
                continue
            times = [t for r in selected for t in r['clip_seconds']]
            r = dict(windows=len(selected), median_ms=statistics.median(times)*1000,
                     p95_ms=float(np.percentile(times,95))*1000,
                     joules_per_clip=statistics.mean(r['joules_per_clip'] for r in selected),
                     average_watts=statistics.mean(r['average_watts'] for r in selected))
            result[f'b{bits}-{arm}'] = r
            lines.append(f"| {bits} | {arm} | {len(selected)} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['joules_per_clip']:.2f} | {r['average_watts']:.2f} |")
    lines += ['', 'Each arm runs in a fresh process. Three rounds alternate forward/reverse arm order.',
              'Each energy window lasts at least 60 seconds and completes whole 16-clip cycles.',
              'Preparation, weight-plane conversion, compilation and graph capture are excluded; frontend, transfers and sign packing are included.',
              'GPU board energy only. Original packed weights remain allocated for embedding/runtime compatibility; candidate bit planes add allocation.',
              'This measures these prototypes, not the maximum attainable binary-activation performance.']
    save(job / 'summary.json', result)
    (job / 'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--micro-job', type=Path)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--bits', type=int)
    parser.add_argument('--arm')
    parser.add_argument('--repeat', type=int)
    args = parser.parse_args()
    storage()
    if args.worker:
        worker(artifact(args.job), args.bits, args.arm, args.repeat)
        return
    job = artifact(args.job or ROOT/'popcount-model'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    job.mkdir(parents=True, exist_ok=False)
    save(job/'status.json', dict(status='waiting_for_microbenchmark'))
    print(f'Full-model artifacts: {job}', flush=True)
    while not (args.micro_job/'status.json').exists() or json.loads((args.micro_job/'status.json').read_text())['status'] != 'completed':
        time.sleep(10)
    micro = json.loads((args.micro_job/'results.json').read_text())
    policy = {f"{r['bits']},{r['m']},{r['n']},{r['k']}": r['variant'] for r in micro}
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from energy import EnergyMeter
        with EnergyMeter() as meter:
            # A CUDA-owning process can release flock just before its context
            # disappears from NVML. Wait for teardown; never whitelist it.
            deadline = time.monotonic() + 30
            while meter.foreign_processes() and time.monotonic() < deadline:
                time.sleep(.1)
            meter._guard()
            gpu = meter.metadata()
        save(job/'initial-policy.json',policy)
        subprocess.run([str(HERE/'python'),str(HERE/'popcount_ternary_checks.py'),
                        '--job',str(job)],check=True)
        ternary_policy = json.loads((job/'w2a2-policy.json').read_text())
        sources = ['popcount_model_benchmark.py','kernels/popcount.py','kernels/packed.py',
                   'kernels/popcount_ternary_activation.py','popcount_ternary_checks.py',
                   'kernels/cuda_gemv.py','runtime_model.py','benchmark.py','energy.py','paths.py']
        cfg = dict(policy=policy, ternary_policy=ternary_policy, gpu=gpu, seconds=60, repeats=3,
                   source_hashes={s:digest(HERE/s) for s in sources}, micro_job=str(args.micro_job),
                   manifest_sha256=digest(ROOT/'data/manifest.json'))
        save(job/'config.json',cfg)
        import shutil
        for s in sources:
            dest=job/'source'/s
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(HERE/s,dest)
        for repeat in range(3):
            for bits in (1,2):
                order = ['baseline', 'popcount'] + (['popcount_a2'] if bits == 2 else [])
                if repeat % 2:
                    order.reverse()
                for arm in order:
                    save(job/'status.json',dict(status='running',repeat=repeat,bits=bits,arm=arm))
                    subprocess.run([str(HERE/'python'),str(HERE/'popcount_model_benchmark.py'),
                                    '--worker','--job',str(job),'--bits',str(bits),'--arm',arm,
                                    '--repeat',str(repeat)],check=True,
                                   env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
                    report(job)
        save(job/'status.json',dict(status='completed'))
    print(f'Completed: {job / "REPORT.md"}',flush=True)


if __name__=='__main__':
    main()
