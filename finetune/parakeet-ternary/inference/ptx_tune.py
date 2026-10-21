"""Offline CUDA-from-PTX compiler and launch-geometry experiments."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import paths

from .benchmark import DEFAULT_EXPORT, energy_meter, metadata, save
from .encoder4 import expanded_configuration
from .fused_kernels import _int8_dot
from .kernels import _finish
from .ptx_cuda import cuda_from_ptx
from .sprint4 import require_power_cap


VARIANTS = ['baseline', 'no_sat', 'cache_ca', 'prefetch128', 'prefetch256',
            'regs64', 'regs80', 'regs96', 'regs128', 'blocks2', 'blocks3', 'blocks4']

# Experimental choices from the first compiler/geometry sweeps. Public
# runtime dispatch remains unchanged; validate these on distinct utterances.
COMPILER_CHOICES = {
    (1024,1024): ['baseline','baseline','baseline'],
    (3072,1024): ['regs96','baseline','blocks2'],
    (4096,1024): ['regs128','baseline','baseline'],
    (2048,1024): ['regs128','cache_ca','blocks4'],
    (1024,4096): ['baseline','baseline','blocks3'],
}
GEOMETRY_CHOICES = {
    (1024,1024): [(32,64,256,4,4,2),(32,64,256,4,4,2),(32,64,128,1,4,2)],
    (3072,1024): [(16,64,128,1,4,3),(32,64,128,1,4,2),(64,64,64,1,4,2)],
    (4096,1024): [(16,64,128,1,4,2),(64,64,128,1,4,4),(32,128,64,1,4,2)],
    (2048,1024): [(32,64,64,4,4,2),(16,64,128,1,4,2),(32,64,128,1,4,2)],
    (1024,4096): [(64,64,128,8,4,3),(64,64,64,8,4,2),(32,128,64,8,4,2)],
}


def choice(mode,m,n,k):
    band=0 if m<=64 else 1 if m<=192 else 2
    if m>512:return expanded_configuration(m,n,k),'baseline'
    config=GEOMETRY_CHOICES.get((n,k),[expanded_configuration(m,n,k)]*3)[band] if mode=='geometry' else expanded_configuration(m,n,k)
    variant=COMPILER_CHOICES.get((n,k),['baseline']*3)[band] if mode=='compiler' else 'baseline'
    return config,variant


def cuda_variant(ptx, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if variant != 'baseline':
        # With ternary weights and K <= 4096 even the maximum integer sum
        # is 520192, so saturation is unnecessary. Retain identical sums.
        ptx = ptx.replace('.row.col.satfinite.s32.s8.s8.s32', '.row.col.s32.s8.s8.s32')
    if variant == 'cache_ca':
        ptx = ptx.replace('cp.async.cg.', 'cp.async.ca.')
    if variant.startswith('prefetch'):
        ptx = ptx.replace('cp.async.cg.shared.global ',
                          f'cp.async.cg.shared.global.L2::{variant[8:]}B ')
    source, count, threads = cuda_from_ptx(ptx)
    options = ['--std=c++17', '--fmad=false']
    if variant.startswith('regs'):
        source = source.replace(f'__launch_bounds__({threads})', '')
        options.append(f'--maxrregcount={variant[4:]}')
    if variant.startswith('blocks'):
        source = source.replace(f'__launch_bounds__({threads})',
                                f'__launch_bounds__({threads},{variant[6:]})')
    return source, count, threads, options


def geometry_candidates(base):
    candidates = [base]
    bm, bn, bk, split, warps, stages = base
    for value in [1, 2, 3, 4]:
        candidates.append((bm,bn,bk,split,warps,value))
    for value in [1, 2, 4, 8]:
        candidates.append((bm,bn,bk,value,warps,stages))
    for value in [64,128,256]:
        candidates.append((bm,bn,value,split,warps,stages))
    for value in [16,32,64]:
        candidates.append((value,bn,bk,split,warps,stages))
    for value in [32,64,128]:
        candidates.append((bm,value,bk,split,warps,stages))
    for value in [2,8]:
        candidates.append((bm,bn,bk,split,value,stages))
    return list(dict.fromkeys(candidates))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,default=DEFAULT_EXPORT)
    p.add_argument('--mode',choices=['compiler','geometry','compressed4','compressed2','grouped2','grouped4','native2','native4'],default='compiler')
    p.add_argument('--rows',type=int,nargs='+',default=[38,126,376])
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    paths.require_mount()
    out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-ptx-{args.mode}'
    out.mkdir();print('RESULTS',out,flush=True)
    (out/'ptx_tune.py').write_bytes(Path(__file__).read_bytes())
    with paths.gpu_lock('CUDA PTX tuning'),torch.inference_mode(),energy_meter() as meter:
        meter._guard();require_power_cap(meter);save(out,'gpu.json',meter.metadata())
        save(out,'metadata.json',metadata(args))
        source_dir=out/'source';source_dir.mkdir()
        for f in Path(__file__).parent.glob('*.py'):(source_dir/f.name).write_bytes(f.read_bytes())
        from .cuda_matmul import kernel
        from torch.cuda._utils import _nvrtc_compile,_cuda_load_module
        kernel('direct')
        null=torch.empty(0,device='cuda',dtype=torch.uint8)
        torch.manual_seed(781)
        torch.backends.cuda.matmul.allow_tf32=False
        results=[]
        for m in args.rows:
            for n,k in [(1024,1024),(3072,1024),(4096,1024),(2048,1024),(1024,4096)]:
                require_power_cap(meter)
                q=torch.randint(-127,128,(3,m,k),device='cuda',dtype=torch.int8)
                qs=torch.rand(3,m,device='cuda')*.01
                w=torch.randint(-1,2,(n,k),device='cuda',dtype=torch.int8).T
                bits=int(args.mode[-1]) if args.mode.startswith('compressed') else 8
                candidate_weight=w
                if bits!=8:
                    from .compressed_weights import pack_dense
                    candidate_weight=pack_dense(w.T,bits)
                ws=torch.rand(n,device='cuda')*.03
                bias=torch.randn(n,device='cuda')*.01
                base=expanded_configuration(m,n,k)
                configs=[base] if args.mode=='compiler' else geometry_candidates(base)
                groups=int(args.mode[-1]) if args.mode.startswith('grouped') else 1
                native_groups=int(args.mode[-1]) if args.mode.startswith('native') else 1
                if groups>1:
                    configs=list(dict.fromkeys([(bm,bn,bk,base[3],4*groups,stage)
                        for bm,bn,bk in [(base[0],base[1],base[2]),(16,64,64),(16,64,128),(16,32,128)]
                        for stage in [2,3]]))
                if native_groups>1:
                    configs=list(dict.fromkeys([(bm,bn,bk,split,4,stage)
                        for bm,bn,bk,split in [(base[0],base[1],base[2],base[3]),
                                               (16,64,128,1),(16,32,128,1),(32,64,128,1)]
                        for stage in [2,3] if k%(bk*split*native_groups)==0]))
                variants=args.variants if args.mode=='compiler' else ['baseline']
                yref=torch.empty(m,n,device='cuda');pref=torch.empty(base[3],m,n,device='cuda')
                def triton_fn():
                    bm,bn,bk,split,warps,stages=base
                    compiled=_int8_dot[(triton.cdiv(m,bm),triton.cdiv(n,bn),split)](
                        q,qs,w,ws,bias,yref,pref,m,n,k,True,bm,bn,bk,split,3,
                        DENSE=True,COLUMN=True,num_warps=warps,num_stages=stages)
                    if split>1:_finish[(triton.cdiv(m*n,256),)](pref,ws,bias,yref,m,n,m,m*n,n,1,True,split,256)
                    return compiled
                triton_fn();torch.cuda.synchronize()
                functions={'triton':triton_fn};owners=[];entries=[]
                for config in configs:
                    bm,bn,bk,split,warps,stages=config
                    y=torch.empty_like(yref);partial=torch.empty(split,m,n,device='cuda')
                    try:
                        from .grouped_dot import _grouped_int8_dot
                        dot_function=_grouped_int8_dot if groups>1 else _int8_dot
                        extra={'GROUPS':groups} if groups>1 else {}
                        if native_groups>1:extra['CTA_GROUPS']=native_groups
                        compiled=dot_function[(triton.cdiv(m,bm),triton.cdiv(n,bn),split)](
                            q,qs,candidate_weight,ws,bias,y,partial,m,n,k,True,bm,bn,bk,split,3,
                            DENSE=True,COLUMN=True,WEIGHT_BITS=bits,num_warps=warps,num_stages=stages,**extra)
                    except Exception as exc:
                        entries.append({'config':config,'status':'compile_failed','error':str(exc)[:2000]});continue
                    for variant in variants:
                        label=variant if args.mode=='compiler' else '-'.join(map(str,config))
                        folder=out/f'm{m}-n{n}-k{k}'/label;folder.mkdir(parents=True)
                        candidate_ptx=compiled.asm['ptx'];candidate_shared=compiled.metadata.shared
                        if native_groups>1:
                            from .native_groups import grouped_ptx
                            candidate_ptx,candidate_shared=grouped_ptx(candidate_ptx,groups=native_groups,
                                shared=candidate_shared,bk=bk,split=split,k=k)
                        source,count,threads,options=cuda_variant(candidate_ptx,variant)
                        (folder/'candidate.cu').write_text(source)
                        (folder/'original.ttgir').write_text(compiled.asm['ttgir'])
                        try:
                            binary,symbol=_nvrtc_compile(source,'encoder_from_triton','120a',nvcc_options=options)
                            native=_cuda_load_module(binary,[symbol])[symbol]
                            shared=candidate_shared
                            if shared>=48*1024:native.set_shared_memory_config(shared)
                            (folder/'candidate.cubin').write_bytes(binary)
                        except Exception as exc:
                            entries.append({'variant':label,'config':config,'status':'compile_failed','error':str(exc)[:2000]});continue
                        pointers=[q,qs,candidate_weight,ws,bias,y,partial]+[null]*(count-7)
                        def fn(native=native,pointers=pointers,threads=threads,shared=shared,config=config,y=y,partial=partial):
                            bm,bn,bk,split,warps,stages=config
                            native(grid=(triton.cdiv(m,bm),triton.cdiv(n,bn),split),
                                   block=(threads,1,1),shared_mem=shared,args=pointers)
                            if split>1:_finish[(triton.cdiv(m*n,256),)](partial,ws,bias,y,m,n,m,m*n,n,1,True,split,256)
                        fn();torch.cuda.synchronize()
                        maxabs=float((y-yref).abs().max())
                        # Different split-K can change FP32 summation order,
                        # so test against the original float32 tolerance.
                        torch.testing.assert_close(y,yref,atol=2e-6,rtol=2e-5)
                        entries.append({'variant':label,'config':config,'status':'ok','shared_bytes':shared,
                                        'max_abs':maxabs,'bitwise_match':torch.equal(y,yref),'options':options})
                        owners.append((native,pointers,y,partial));functions[label]=fn
                timings={name:[] for name in functions}
                for repeat in range(args.repeats):
                    names=list(functions)
                    if repeat%2:names.reverse()
                    for name in names:
                        timings[name].append(triton.testing.do_bench_cudagraph(functions[name],rep=25)*1000)
                medians={name:statistics.median(v) for name,v in timings.items()}
                row={'m':m,'n':n,'k':k,'base_config':base,'entries':entries,'timings_us':timings,'median_us':medians,
                     'scope':'synthetic prequantized resident dot + finish; no quantization, encoder, or energy claim'}
                results.append(row);save(out,'sweep.json',results)
                winner=min(medians,key=medians.get)
                print(json.dumps({'m':m,'n':n,'k':k,'triton_us':medians['triton'],
                    'winner':winner,'winner_us':medians[winner],'change_percent':100*(medians[winner]/medians['triton']-1),
                    'compile_failures':sum(e['status']!='ok' for e in entries)}),flush=True)
                del functions,owners


if __name__=='__main__':main()
