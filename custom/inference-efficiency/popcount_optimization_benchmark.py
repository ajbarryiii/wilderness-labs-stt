"""Correctness-gated popcount optimization ablation and full-model benchmark.

The CPU-only controller owns the shared GPU lock. Each CUDA worker exits before
the next arm starts. Frozen sources, workload hashes and raw samples are saved.
"""
import argparse
import datetime as dt
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from paths import ROOT, artifact, digest, save, storage

HERE=Path(__file__).resolve().parent
LOCK=Path('/mnt/hd/wilderness-labs-stt/stt-distillation/active.lock')
BASE_JOB=ROOT/'popcount-model/20260912T171543Z'
SPRINT_JOB=ROOT/'agent-kernel-sprint/20260912T160024Z'
SOURCES=['popcount_optimization_benchmark.py','popcount_optimized_runtime.py',
         'test_popcount_optimized.py','kernels/popcount_optimized.py',
         'popcount_model_benchmark.py','popcount_benchmark.py',
         'kernels/popcount.py','kernels/popcount_ternary_activation.py',
         'kernels/packed.py','kernels/cuda_gemv.py','runtime_model.py',
         'benchmark.py','energy.py','paths.py','kernels/sprint_cuda.py',
         'kernels/sprint_fusion.py','kernels/sprint_qkv.py','kernels/sprint_fused_vector.py']


def install(cfg,bits,a2,arm,job):
    from kernels.packed import PackedWeight
    original=PackedWeight.linear
    policy=cfg['ternary_policy'] if a2 else cfg['policy']
    if arm=='popcount':
        from popcount_model_benchmark import install as baseline
        baseline(policy,a2)
        return lambda: setattr(PackedWeight,'linear',original)
    if arm=='sprint':
        restores=[]
        for plugin in cfg['sprint']['policies']['binary' if bits==1 else 'ternary']['plugins']:
            restores.append(importlib.import_module(plugin['module']).install(plugin['name'],job/'compiled-sprint'))
        def restore():
            for f in reversed(restores): f()
        return restore
    from popcount_optimized_runtime import install as optimized
    return optimized(policy,a2=a2,candidate=arm)


def full_checks(job,cfg,bits,a2):
    import numpy as np
    import torch
    import benchmark
    manifest=benchmark.workload_manifest()
    restore=install(cfg,bits,a2,'popcount',job)
    model,call,infer,mel=benchmark.make_runner('binary-packed' if bits==1 else 'ternary-packed',manifest)
    refs=[]
    for clip in manifest['clips']:
        x=np.load(clip['mel'],allow_pickle=False)
        logits=infer(x).clone()
        encoded=model.encode(torch.from_numpy(x).cuda().half()).clone()
        refs.append((logits,model.last_predictions.clone(),encoded))
    restore()
    restore=install(cfg,bits,a2,cfg.get('candidate','selected'),job)
    model.capture(torch.from_numpy(mel).cuda().half(),manifest['replay_tokens'])
    rows=[]
    for i,clip in enumerate(manifest['clips']):
        x=np.load(clip['mel'],allow_pickle=False)
        actual=infer(x)
        encoded=model.encode(torch.from_numpy(x).cuda().half())
        for name,value,expected in zip(('logits','predictions','encoded'),
                (actual,model.last_predictions,encoded),refs[i]):
            torch.testing.assert_close(value,expected,atol=0,rtol=0,msg=f'clip {i} {name}')
        rows.append(dict(clip=i,exact=True,forced_tokens=128))
    save(job/f'full-check-b{bits}-a{2 if a2 else 1}.json',rows)
    restore()
    print(f'Full model exact on all 16 clips: W{bits}A{2 if a2 else 1}',flush=True)


def micro(job,cfg):
    import math
    import torch
    import torch.nn.functional as F
    from kernels import popcount, popcount_ternary_activation as pa2, popcount_optimized as opt
    from kernels.packed import PackedWeight
    from popcount_benchmark import capture,timing
    from popcount_optimized_runtime import FEATURES
    from runtime_model import _Factory,_Block
    torch.set_grad_enabled(False)
    torch.manual_seed(20260912)
    rows=[]
    def measure(ops):
        result={}
        for name,op in ops.items():
            graph=capture(op,64)
            samples=timing(graph,64)
            result[name]=dict(median_us=statistics.median(samples),samples_us=samples)
        return result
    for bits,a2 in ((1,False),(2,False),(2,True)):
        base=pa2 if a2 else popcount
        policy=cfg['ternary_policy'] if a2 else cfg['policy']
        for m,n,k in ((1,1024,1024),(1,1024,4096),(1,51864,1024),
                      (1500,1024,1024),(1500,3072,1024),(1500,4096,1024),(1500,1024,4096)):
            codes=torch.randint(-1,2,(n,k),device='cuda',dtype=torch.int8)
            if bits==1: codes=torch.where(codes==0,1,codes)
            p=PackedWeight.from_codes(codes,bits,1/math.sqrt(k))
            w0=base.BitWeight(codes,p.scales,bits)
            w1=opt.BitWeight(codes,p.scales,bits)
            x=torch.randn(m,k,device='cuda',dtype=torch.float16)
            a=opt.pack(x,a2).words
            out=torch.empty(m,n,device='cuda',dtype=torch.float16)
            layout=policy.get(f'{bits},{m},{n},{k}','warp' if m<=4 else 'tile')
            w0.linear(a,out,layout)
            expected=out.clone()
            w1.linear(a,out,layout,a2=a2)
            torch.testing.assert_close(out,expected,atol=0,rtol=0)
            opt.quantized_gemm(p,x,out=out,a2=a2)
            torch.testing.assert_close(out,expected,atol=0,rtol=0)
            def old_pack_dot():
                base.pack(x,a)
                w0.linear(a,out,layout)
            def new_pack_dot():
                w1.linear(opt.pack(x,a2).words,out,layout,a2=a2)
            ops={'old_prepacked':lambda:w0.linear(a,out,layout),
                 'counts_prepacked':lambda:w1.linear(a,out,layout,a2=a2),
                 'old_pack_dot':old_pack_dot,'counts_pack_dot':new_pack_dot}
            if m>4:
                ops['quantized_tensor_core']=lambda:opt.quantized_gemm(p,x,out=out,a2=a2)
            rows.append(dict(test='projection',bits=bits,a2=a2,m=m,n=n,k=k,layout=layout,timing=measure(ops)))
        f=_Factory('binary' if bits==1 else 'ternary','packed',20260912,'cuda',torch.float16)
        block=_Block(f,1024,16,decoder=True)
        x=torch.randn(1,1,1024,device='cuda',dtype=torch.float16)
        memory=tuple(torch.randn(1,16,1500,64,device='cuda',dtype=torch.float16) for _ in range(2))
        cache=tuple(torch.randn(1,16,129,64,device='cuda',dtype=torch.float16) for _ in range(2))
        timings={}
        expected=None
        for arm in ('popcount',*FEATURES):
            restore=install(cfg,bits,a2,arm,job)
            op=lambda:block(x,memory_kv=memory,cache=cache,step=128)
            actual=op()
            if expected is None: expected=actual.clone()
            else: torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            timings[arm]=measure({arm:op})[arm]
            restore()
        rows.append(dict(test='decoder_block',bits=bits,a2=a2,timing=timings))
        save(job/'micro.json',rows)
    print('Projection and decoder-block ablations passed and measured',flush=True)


def model_screen(job,cfg,bits,a2):
    """Same model and inputs, cumulative features; CUDA graph time only."""
    import torch
    import benchmark
    from popcount_optimized_runtime import FEATURES
    manifest=benchmark.workload_manifest()
    restore=install(cfg,bits,a2,'popcount',job)
    model,call,infer,mel=benchmark.make_runner('binary-packed' if bits==1 else 'ternary-packed',manifest)
    mel_gpu=torch.from_numpy(mel).cuda().half()
    expected=model.replay_graph(mel_gpu).clone()
    torch.cuda.synchronize()
    restore()
    rows={}
    for arm in ('popcount',*FEATURES):
        restore=install(cfg,bits,a2,arm,job)
        model.capture(mel_gpu,manifest['replay_tokens'])
        actual=model.replay_graph(mel_gpu)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        for _ in range(8): model.replay_graph(mel_gpu)
        torch.cuda.synchronize()
        times=[]
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        for _ in range(24):
            start.record()
            model.replay_graph(mel_gpu)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        rows[arm]=dict(median_ms=statistics.median(times),samples_ms=times,exact=True)
        print(f'Screen W{bits}A{2 if a2 else 1} {arm}: {statistics.median(times):.3f} ms',flush=True)
        save(job/f'model-screen-b{bits}-a{2 if a2 else 1}.json',rows)
        restore()


def worker(job,cfg,bits,a2,arm,repeat):
    import numpy as np
    import torch
    import benchmark
    from energy import EnergyMeter
    torch.set_grad_enabled(False)
    with EnergyMeter(synchronize=torch.cuda.synchronize) as meter:
        meter._guard()
        before=meter.metadata()
        if any(before[k]!=cfg['gpu'][k] for k in ('uuid','power_limit_w')):
            raise RuntimeError('GPU identity/power limit changed')
        if arm=='checks':
            full_checks(job,cfg,bits,a2)
            return
        if arm=='micro':
            micro(job,cfg)
            return
        if arm=='screen':
            model_screen(job,cfg,bits,a2)
            return
        install(cfg,bits,a2,arm,job)
        manifest=benchmark.workload_manifest()
        model,call,infer,mel=benchmark.make_runner('binary-packed' if bits==1 else 'ternary-packed',manifest)
        for _ in manifest['clips']:
            if not torch.isfinite(call()).all().item():
                raise RuntimeError('Nonfinite output')
        torch.cuda.synchronize()
        times=[]
        def cycle():
            for _ in manifest['clips']:
                start=time.perf_counter()
                call()
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
        measurement=meter.measure(cycle,min_seconds=cfg['seconds'])
        row=dict(bits=bits,a2=a2,arm=arm,repeat=repeat,clips=len(times),
                 median_ms=statistics.median(times)*1000,p95_ms=float(np.percentile(times,95))*1000,
                 joules_per_clip=measurement['energy_joules']/len(times),
                 average_watts=measurement['average_watts'],clip_seconds=times,
                 gpu_before=before,gpu_after=meter.metadata(),measurement=measurement)
        save(job/f'{repeat:02d}-b{bits}-a{2 if a2 else 1}-{arm}.json',row)
        print(json.dumps({k:row[k] for k in ('bits','a2','arm','repeat','median_ms','joules_per_clip')}),flush=True)


def report(job):
    import numpy as np
    raw=[json.loads(p.read_text()) for p in sorted(job.glob('[0-9][0-9]-b*.json'))]
    summary={}
    lines=['# Popcount optimization results','',
           'RTX 5090 at 400 W; seeded Whisper medium.en; 16 frozen 30-second clips; 128 forced tokens.',
           'Frontend, input transfers and activation quantization are included. GPU board energy only.',
           'Three fresh-process windows per arm, each at least 60 seconds; whole clip cycles.',
           'Sprint uses FP16 activations. Popcount/all use the same quantized activations as each other.',
           'Random models: this validates arithmetic and performance, not recognition quality.','',
           '| Model | Arm | Windows | Median ms | p95 ms | J/clip | W |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for bits,a2 in ((1,False),(2,False),(2,True)):
        for arm in ('popcount','all','selected','sprint'):
            rows=[r for r in raw if (r['bits'],r['a2'],r['arm'])==(bits,a2,arm)]
            if not rows: continue
            samples=[t for r in rows for t in r['clip_seconds']]
            label=f'W{bits}A{2 if a2 else 1}' if arm!='sprint' else f'W{bits}A16'
            r=dict(windows=len(rows),median_ms=statistics.median(samples)*1000,
                   p95_ms=float(np.percentile(samples,95))*1000,
                   joules_per_clip=statistics.mean(r['joules_per_clip'] for r in rows),
                   average_watts=statistics.mean(r['average_watts'] for r in rows))
            summary[f'{label}-{arm}']=r
            lines.append(f'| {label} | {arm} | {len(rows)} | {r["median_ms"]:.3f} | {r["p95_ms"]:.3f} | {r["joules_per_clip"]:.3f} | {r["average_watts"]:.2f} |')
    micro_file=job/'micro.json'
    if micro_file.exists():
        lines+=['','Cumulative decoder-block ablation (cached weights, graph timing; microseconds):','',
                '| Model | Old popcount | Counts | + Tensor Core encoder | + Output/cache fusion | + Norm/GELU packing | Selected |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in json.loads(micro_file.read_text()):
            if r['test']=='decoder_block':
                v=r['timing']
                lines.append(f'| W{r["bits"]}A{2 if r["a2"] else 1} | '+ ' | '.join(f'{v[a]["median_us"]:.3f}' for a in ('popcount','counts','hybrid','fusion','all','selected'))+' |')
    lines+=['','`counts` precomputes the weight nonzero count only for A1. A2 still counts the activation/weight nonzero intersection.',
            '`hybrid` quantizes FP16 values inside Tensor Core encoder tiles; decoder stays on popcount.',
            '`fusion` adds FP16-rounded residual/GELU epilogues and direct QKV cache writes.',
            '`all` also packs directly from decoder LayerNorm and combines GELU with packing.',
            '`selected` retains popcount for the binary encoder, uses Tensor Core ternary encoder tiles, enables output/cache fusion and leaves fused packing disabled.',
            'Original packed weights remain allocated for embeddings/encoder; decoder bit planes and row counts add allocation.',
            'Preparation, bit-plane conversion, compilation and graph capture are excluded from measurements.',
            'See component, tiny-model and full-model check JSON files; full-model checks require exact encoder/logit/predicted-ID agreement on all 16 clips.']
    save(job/'summary.json',summary)
    (job/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--bits',type=int,default=1)
    p.add_argument('--a2',action='store_true')
    p.add_argument('--arm',default='all')
    p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--checks-only',action='store_true')
    p.add_argument('--screen-only',action='store_true')
    p.add_argument('--candidate',choices=('all','selected'),default='selected')
    args=p.parse_args()
    storage()
    if args.worker:
        job=artifact(args.job)
        cfg=json.loads((job/'config.json').read_text())
        for s,h in cfg['source_hashes'].items():
            if digest(HERE/s)!=h: raise RuntimeError(f'Source changed: {s}')
        if digest(ROOT/'data/manifest.json')!=cfg['manifest_sha256']:
            raise RuntimeError('Workload changed')
        worker(job,cfg,args.bits,args.a2,args.arm,args.repeat)
        return
    job=artifact(args.job or ROOT/'popcount-optimization'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    job.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((BASE_JOB/'config.json').read_text())
    cfg.update(seconds=60,repeats=3,candidate=args.candidate,baseline_job=str(BASE_JOB),sprint=json.loads((SPRINT_JOB/'confirmed-receipt.json').read_text()),
               source_hashes={s:digest(HERE/s) for s in SOURCES})
    for s,h in cfg['sprint']['source_hashes'].items():
        if digest(HERE/s)!=h: raise RuntimeError(f'Sprint source changed: {s}')
    for s in SOURCES:
        dest=job/'source'/s
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(HERE/s,dest)
    save(job/'status.json',dict(status='waiting_for_gpu'))
    print(f'Artifacts: {job}',flush=True)
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','EFFICIENCY_REQUIRE_CUDA_GEMV':'1'}
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from energy import EnergyMeter
        def guard():
            with EnergyMeter() as meter:
                deadline=time.monotonic()+30
                while meter.foreign_processes() and time.monotonic()<deadline: time.sleep(.1)
                meter._guard()
                return meter.metadata()
        cfg['gpu']=guard()
        save(job/'config.json',cfg)
        try:
            save(job/'status.json',dict(status='checking'))
            subprocess.run([str(HERE/'python'),str(HERE/'test_popcount_optimized.py'),'--job',str(job)],check=True,env=env)
            def run(bits,a2,arm,repeat=0):
                guard()
                save(job/'status.json',dict(status='running',bits=bits,a2=a2,arm=arm,repeat=repeat))
                subprocess.run([str(HERE/'python'),str(HERE/'popcount_optimization_benchmark.py'),
                    '--worker','--job',str(job),'--bits',str(bits),'--arm',arm,'--repeat',str(repeat)]+
                    (['--a2'] if a2 else []),check=True,env=env)
            for bits,a2 in ((1,False),(2,False),(2,True)): run(bits,a2,'checks')
            run(1,False,'micro')
            if args.screen_only:
                for bits,a2 in ((1,False),(2,False),(2,True)): run(bits,a2,'screen')
            if not (args.checks_only or args.screen_only):
                arms=[(1,False,'popcount'),(1,False,args.candidate),(1,False,'sprint'),
                      (2,False,'popcount'),(2,False,args.candidate),(2,False,'sprint'),
                      (2,True,'popcount'),(2,True,args.candidate)]
                for repeat in range(cfg['repeats']):
                    for bits,a2,arm in (arms if repeat%2==0 else reversed(arms)):
                        run(bits,a2,arm,repeat)
                        report(job)
            report(job)
            status='screen_completed' if args.screen_only else 'checks_completed' if args.checks_only else 'completed'
            save(job/'status.json',dict(status=status))
            save(ROOT/'popcount-optimization/latest.json',dict(job=str(job),status=status))
        except Exception as exc:
            save(job/'status.json',dict(status='failed',error=str(exc)))
            raise
    print(f'Completed: {job/"REPORT.md"}',flush=True)


if __name__=='__main__': main()
