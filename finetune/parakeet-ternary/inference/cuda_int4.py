"""Exact INT8 residual arithmetic via native signed/unsigned INT4 MMA.

q = 16 * (q >> 4) + (q & 15), with signed high and unsigned low
nibbles. Ternary weights use signed nibbles. This is a representation
experiment, not a reduction in activation precision.
"""
import argparse
import functools
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
import paths

from .benchmark import DEFAULT_EXPORT, energy_meter, metadata, save
from .cuda_matmul import CONFIGS, _SOURCE, kernel as int8_kernel
from .encoder4 import expanded_configuration
from .fused_kernels import _int8_dot
from .kernels import _finish
from .sprint4 import require_power_cap


@triton.jit
def _pack_q(Q, P, M:tl.constexpr, K:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=i//(K//2); col=i%(K//2)
    a=tl.load(Q+row*K+2*col,i<3*M*(K//2),other=0).to(tl.int32)
    b=tl.load(Q+row*K+2*col+1,i<3*M*(K//2),other=0).to(tl.int32)
    component=row//M; m=row%M
    low=(a&15)|((b&15)<<4)
    high=((a>>4)&15)|(((b>>4)&15)<<4)
    tl.store(P+(2*component*M+m)*(K//2)+col,low,i<3*M*(K//2))
    tl.store(P+((2*component+1)*M+m)*(K//2)+col,high,i<3*M*(K//2))


@triton.jit
def _pack_w(W, P, N:tl.constexpr, K:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(W+2*i,i<N*K//2,other=0).to(tl.int32)
    b=tl.load(W+2*i+1,i<N*K//2,other=0).to(tl.int32)
    tl.store(P+i,(a&15)|((b&15)<<4),i<N*K//2)


def source_for(config, merged=False):
    wm,wn,rm,rn,bk,split,mode=config
    start=_SOURCE.index('__device__ __forceinline__ void mma(')
    end=_SOURCE.index('__device__ __forceinline__ u32 load_word',start)
    mma=r'''
__device__ __forceinline__ void mma(bool LOW, int *c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    if (LOW) {
        asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
}
'''
    source=_SOURCE[:start]+mma+_SOURCE[end:]
    source=source.replace('3 * BM','6 * BM').replace('component < 3','component < 6')
    source=source.replace('int accum[3]','int accum[6]').replace('u32 af[3]','u32 af[6]')
    source=source.replace('mma(accum[component]', 'mma(component % 2 == 0,accum[component]')
    for component in reversed(range(3)):
        source=source.replace(f'float(accum[{component}][i][j][element])',
                              f'float(accum[{2*component}][i][j][element] + 16 * accum[{2*component+1}][i][j][element])')
    if merged:
        # Reuse three full accumulators and only a four-register temporary
        # for the high nibble of one MMA atom. Test the instruction cost.
        source=source.replace('int accum[6]','int accum[3]')
        old='''for (int component = 0; component < 6; ++component)
                #pragma unroll
                for (int i = 0; i < RM; ++i)
                    #pragma unroll
                    for (int j = 0; j < RN; ++j)
                        mma(component % 2 == 0,accum[component][i][j],af[component][i][0],af[component][i][1],
                            af[component][i][2],af[component][i][3],bf[j][0],bf[j][1]);'''
        new='''for (int component = 0; component < 3; ++component)
                #pragma unroll
                for (int i = 0; i < RM; ++i)
                    #pragma unroll
                    for (int j = 0; j < RN; ++j) {
                        int hi[4] = {};
                        mma(false,hi,af[2*component+1][i][0],af[2*component+1][i][1],
                            af[2*component+1][i][2],af[2*component+1][i][3],bf[j][0],bf[j][1]);
                        #pragma unroll
                        for (int e = 0; e < 4; ++e) accum[component][i][j][e] += 16 * hi[e];
                        mma(true,accum[component][i][j],af[2*component][i][0],af[2*component][i][1],
                            af[2*component][i][2],af[2*component][i][3],bf[j][0],bf[j][1]);
                    }'''
        if old not in source:raise ValueError('INT4 loop template changed')
        source=source.replace(old,new)
        for component in range(3):
            source=source.replace(f'float(accum[{2*component}][i][j][element] + 16 * accum[{2*component+1}][i][j][element])',
                                  f'float(accum[{component}][i][j][element])')
    source+=f'\nWRAP(ternary_int4,{wm},{wn},{rm},{rn},{bk},{split},{mode})\n'
    return source


@functools.lru_cache(None)
def kernel(config,merged=False):
    paths.require_mount();int8_kernel('direct')
    if torch.cuda.is_current_stream_capturing():raise RuntimeError('warm INT4 kernels before capture')
    return torch.cuda._compile_kernel(source_for(config,merged),'ternary_int4',compute_capability='120a',
                                     nvcc_options=['--std=c++17','--fmad=false'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,default=DEFAULT_EXPORT)
    p.add_argument('--rows',type=int,nargs='+',default=[38,126,376])
    p.add_argument('--names',nargs='+',choices=CONFIGS,default=['direct_m16','shared','async','async_split4','async_split8','async_large','async_deep','pipeline','pipeline_split4','pipeline_m16','pipeline_m16_split4'])
    args=p.parse_args();paths.require_mount()
    out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-cuda-int4';out.mkdir()
    print('RESULTS',out,flush=True)
    (out/'cuda_int4.py').write_bytes(Path(__file__).read_bytes())
    with paths.gpu_lock('Exact INT4 CUDA experiments'),torch.inference_mode(),energy_meter() as meter:
        meter._guard();require_power_cap(meter);save(out,'gpu.json',meter.metadata())
        save(out,'metadata.json',metadata(args))
        torch.manual_seed(793);torch.backends.cuda.matmul.allow_tf32=False
        results=[]
        for m in args.rows:
            for n,k in [(1024,1024),(3072,1024),(4096,1024),(2048,1024),(1024,4096)]:
                require_power_cap(meter)
                q=torch.randint(-127,128,(3,m,k),device='cuda',dtype=torch.int8)
                qs=torch.rand(3,m,device='cuda')*.01
                w=torch.randint(-1,2,(n,k),device='cuda',dtype=torch.int8)
                qp=torch.empty(6,m,k//2,device='cuda',dtype=torch.int8)
                wp=torch.empty(n,k//2,device='cuda',dtype=torch.int8)
                _pack_q[(triton.cdiv(q.numel()//2,256),)](q,qp,m,k,256)
                _pack_w[(triton.cdiv(w.numel()//2,256),)](w,wp,n,k,256)
                ws=torch.rand(n,device='cuda')*.03;bias=torch.randn(n,device='cuda')*.01
                y0=torch.empty(m,n,device='cuda');base=expanded_configuration(m,n,k)
                p0=torch.empty(base[3],m,n,device='cuda')
                def triton_fn():
                    bm,bn,bk,split,warps,stages=base
                    _int8_dot[(triton.cdiv(m,bm),triton.cdiv(n,bn),split)](q,qs,w.T,ws,bias,y0,p0,
                        m,n,k,True,bm,bn,bk,split,3,DENSE=True,COLUMN=True,num_warps=warps,num_stages=stages)
                    if split>1:_finish[(triton.cdiv(m*n,256),)](p0,ws,bias,y0,m,n,m,m*n,n,1,True,split,256)
                triton_fn();torch.cuda.synchronize()
                functions={'triton':triton_fn};entries=[];owners=[]
                for name in args.names:
                    orig=CONFIGS[name];config=(*orig[:4],orig[4]//2,*orig[5:])
                    wm,wn,rm,rn,bk,split,mode=config
                    bm=wm*rm*16;bn=wn*rn*8
                    shared=(6*bm+bn)*(bk+16)*(2 if mode==3 else 1) if mode else 0
                    for merged in [False,True]:
                        native=kernel(config,merged)
                        if shared>=48*1024:native.set_shared_memory_config(shared)
                        y=torch.empty_like(y0);partial=torch.empty(split,m,n,device='cuda')
                        pointers=[qp,qs,wp,ws,bias,y,partial,m,n,k//2,1,0]
                        def fn(native=native,pointers=pointers,shared=shared,bm=bm,bn=bn,split=split,y=y,partial=partial,threads=wm*wn*32):
                            native(grid=(triton.cdiv(n,bn),triton.cdiv(m,bm),split),block=(threads,1,1),shared_mem=shared,args=pointers)
                            if split>1:_finish[(triton.cdiv(m*n,256),)](partial,ws,bias,y,m,n,m,m*n,n,1,True,split,256)
                        fn();torch.cuda.synchronize();torch.testing.assert_close(y,y0,atol=2e-6,rtol=2e-5)
                        label=name+('-merged' if merged else '')
                        entries.append({'name':label,'config':config,'shared_bytes':shared,'max_abs':float((y-y0).abs().max()),'bitwise_match':torch.equal(y,y0)})
                        functions[label]=fn;owners.append((native,pointers,y,partial))
                timings={name:[] for name in functions}
                for repeat in range(3):
                    names=list(functions)
                    if repeat%2:names.reverse()
                    for name in names:timings[name].append(triton.testing.do_bench_cudagraph(functions[name],rep=25)*1000)
                medians={name:statistics.median(v) for name,v in timings.items()}
                row={'m':m,'n':n,'k':k,'entries':entries,'timings_us':timings,'median_us':medians,
                     'scope':'exact prepacked INT4 dot + same finish; excludes nibble packing/quantization; no energy claim'}
                results.append(row);save(out,'micro.json',results)
                winner=min(medians,key=medians.get)
                print(json.dumps({'m':m,'n':n,'k':k,'triton_us':medians['triton'],'winner':winner,'winner_us':medians[winner],
                     'change_percent':100*(medians[winner]/medians['triton']-1)}),flush=True)
                del functions,owners


if __name__=='__main__':main()
