"""Experimental CUDA + inline-PTX starting point from compiled Triton kernels.

This preserves the generated instruction body, not readable recovered C++.
Only the current encoder's single-entry, pointer-only, shared-memory kernels
are supported. No production runtime dispatch changes are made.
"""
import argparse
import json
import re
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
from .sprint4 import require_power_cap


def cuda_from_ptx(ptx, *, name='encoder_from_triton'):
    """Wrap the original instruction body in one CUDA inline-assembly block."""
    entries=list(re.finditer(r'\.visible\s+\.entry\s+(\w+)\s*\((.*?)\)',ptx,re.S))
    if len(entries)!=1:
        raise ValueError('expected one visible PTX entry')
    entry=entries[0]
    params=[p.strip() for p in entry.group(2).split(',')]
    if any(not re.match(r'\.param\s+\.[ub]64\b',p) for p in params):
        raise ValueError('only 64-bit pointer parameters are supported')
    param_names=[p.split()[-1] for p in params]
    start=ptx.index('{',entry.end())
    depth=1
    for end in range(start+1,len(ptx)):
        depth+=(ptx[end]=='{')-(ptx[end]=='}')
        if depth==0:break
    else:
        raise ValueError('unbalanced PTX entry body')
    body=ptx[start+1:end]
    body=re.sub(r'^\s*\.loc[^\n]*','',body,flags=re.M)
    # NVRTC substitutes input operands with registers such as %rd1. Original
    # LLVM register names would shadow those inputs inside the assembly scope.
    # Give every declared register family its own namespace before substitution.
    families=re.findall(r'\.reg\s+\.\w+\s+%(\w+)(?:<\d+>)?\s*;',body)
    for family in sorted(families,key=len,reverse=True):
        body=re.sub(r'%('+re.escape(family)+r'\d*)\b',r'%wl_\1',body)
    for i,param in enumerate(param_names):
        body=re.sub(r'ld\.param\.[ub]64\s+([^,;]+),\s*\['+re.escape(param)+r'\]\s*;',
                    lambda m:f'mov.u64 {m.group(1)}, __WL_ARG_{i}__;',body)
        if re.search(r'\b'+re.escape(param)+r'\b',body):
            raise ValueError('unsupported parameter access')
    # LLVM's shared symbol is a 32-bit shared-space address. Pass the same
    # address explicitly from CUDA rather than relying on NVRTC symbol names.
    if re.search(r'mov\.(?:u64|b64)\s+[^;]*\bglobal_smem\b',body):
        raise ValueError('unsupported 64-bit shared-symbol access')
    body=re.sub(r'\bglobal_smem\b','__WL_SMEM__',body)
    body=body.replace('%','%%')
    for i in range(len(params)):
        body=body.replace(f'__WL_ARG_{i}__',f'%{i}')
    body=body.replace('__WL_SMEM__',f'%{len(params)}')
    threads=re.search(r'\.reqntid\s+(\d+)',ptx[entry.end():start])
    if threads is None:
        raise ValueError('expected a fixed one-dimensional thread block')
    threads=int(threads.group(1))
    signature=', '.join(f'const void *arg{i}' for i in range(len(params)))
    operands=', '.join([f'"l"(arg{i})' for i in range(len(params))]+['"r"(shared_base)'])
    # JSON string quoting also produces valid C++ string literals here. It is
    # used for source generation only, never interpolated into shell commands.
    quoted='\n'.join('        '+json.dumps(line+'\n') for line in ('{\n'+body+'\n}').splitlines())
    source=f'''// Generated from Triton PTX; preserve its thread mapping and instruction body.
extern "C" __global__ __launch_bounds__({threads})
void {name}({signature}) {{
    extern __shared__ __align__(16) unsigned char cuda_smem[];
    unsigned long long shared_address;
    asm("cvta.to.shared.u64 %0, %1;" : "=l"(shared_address) : "l"(cuda_smem));
    unsigned shared_base = (unsigned)shared_address;
    asm volatile(
{quoted}
        : : {operands} : "memory");
}}
'''
    return source,len(params),threads


class PtxCudaProjection:
    """Experiment-only encoder adapter with CUDA compiled from its Triton PTX."""
    def __init__(self, artifacts=None, *, tuning='baseline', weight_bits=8, row_warps=4):
        from .encoder4 import ExpandedProjection
        from .cuda_matmul import kernel
        self.fallback=ExpandedProjection('triton_col_tuned')
        self.kernels={}
        self.artifacts=artifacts
        self.tuning=tuning
        if tuning not in ('baseline','compiler','geometry','native'):raise ValueError(tuning)
        if weight_bits not in (2,4,8):raise ValueError(weight_bits)
        self.weight_bits=weight_bits
        self.defer_epilogue=False
        self.prequantized={}
        self.no_defer_weights=set()
        if row_warps not in (1,2,4,8,'adaptive'):raise ValueError(row_warps)
        self.row_warps=row_warps
        self.null=torch.empty(0,device='cuda',dtype=torch.uint8)
        if self.null.data_ptr()!=0:raise RuntimeError('expected a null empty-tensor pointer')
        kernel('direct')  # Reuse the pinned headerless NVRTC environment.

    @property
    def bytes(self):return self.fallback.bytes

    def configuration(self,m,n,k,config=None):
        from .ptx_tune import choice
        selected,variant=choice(self.tuning,m,n,k);groups=1
        if self.tuning=='native' and m<=64 and k==1024 and config is None:
            if n in (1024,2048):selected=(16,32 if n==1024 else 64,128,1,4,3);groups=2
            elif n==3072:selected=(*expanded_configuration(m,n,k)[:5],3)
        return config or selected,variant,groups

    def _compile_native(self,compiled,key,variant):
        from .ptx_tune import cuda_variant
        m,n,k,bm,bn,bk,split,warps,stages,has_bias,activation,groups=key
        if compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
            raise ValueError('scratch-using kernels require a separate launcher')
        ptx=compiled.asm['ptx'];shared=compiled.metadata.shared
        if groups>1:
            from .native_groups import grouped_ptx
            ptx,shared=grouped_ptx(ptx,groups=groups,shared=shared,bk=bk,split=split,k=k)
        source,count,threads,options=cuda_variant(ptx,variant)
        from torch.cuda._utils import _nvrtc_compile,_cuda_load_module
        binary,symbol=_nvrtc_compile(source,'encoder_from_triton','120a',nvcc_options=options)
        native=_cuda_load_module(binary,[symbol])[symbol]
        if shared>=48*1024:native.set_shared_memory_config(shared)
        if threads!=warps*32*groups:raise ValueError('unexpected CUDA thread count')
        self.kernels[key]=(native,count,threads,shared)
        if self.artifacts is not None:
            paths.require_mount()
            import hashlib
            tag=hashlib.sha256(repr(key).encode()).hexdigest()[:16]
            folder=self.artifacts/('kernel-'+tag);folder.mkdir(parents=True,exist_ok=True)
            (folder/'original.ptx').write_text(compiled.asm['ptx'])
            (folder/'transformed.ptx').write_text(ptx)
            (folder/'reconstructed.cu').write_text(source)
            (folder/'reconstructed.cubin').write_bytes(binary)
            save(folder,'launch.json',{'key':list(key),'pointer_arguments':count,
                 'threads':threads,'shared_bytes':shared,'tuning':self.tuning,'weight_bits':self.weight_bits,
                 'compiler_variant':variant,'options':options,'cta_groups':groups})

    def prewarm(self,m,n,k,has_bias,activation=''):
        """Compile without launching a matmul; useful for large quality sets.

        Warmup uses null tensors solely to supply aligned pointer types to
        Triton's compiler. No kernel may execute with these arguments.
        """
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('prewarm before capture')
        (bm,bn,bk,split,warps,stages),variant,groups=self.configuration(m,n,k)
        key=(m,n,k,bm,bn,bk,split,warps,stages,has_bias,activation,groups)
        if key in self.kernels:return
        q=torch.empty(0,device='cuda',dtype=torch.int8)
        w=q if self.weight_bits==8 else self.null
        f=torch.empty(0,device='cuda',dtype=torch.float32)
        compiled=_int8_dot.warmup(q,f,w,f,f if has_bias else None,f,f,m,n,k,has_bias,
            bm,bn,bk,split,3,grid=(triton.cdiv(m,bm),triton.cdiv(n,bn),split),ACTIVATION=activation,
            DENSE=True,COLUMN=True,WEIGHT_BITS=self.weight_bits,CTA_GROUPS=groups,num_warps=warps,num_stages=stages)
        self._compile_native(compiled,key,variant)

    def __call__(self,x,mod,conv=False,components=3,activation='',norm=None,
                 config=None,input_activation=''):
        from .encoder4 import _expand
        from .fused_kernels import _quantize
        xx=x.transpose(1,2) if conv else x
        if xx.ndim==2:xx=xx.unsqueeze(0)
        b,t,k=xx.shape;m=b*t;n=mod.out_features
        if components!=3 or k%128:
            return self.fallback(x,mod,conv,components,activation,norm,config,input_activation)
        weight_key=(mod.packed_t.data_ptr(),k,n,True,self.weight_bits)
        if weight_key not in self.fallback.weights:
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('warm weights before capture')
            if self.weight_bits==8:
                weight=torch.empty((n,k),device=x.device,dtype=torch.int8)
                _expand[(triton.cdiv(k*n,512),)](mod.packed_t,weight,k,n,True,512)
                weight=weight.T
            else:
                from .compressed_weights import pack_export
                weight=pack_export(mod,k,n,self.weight_bits)
            self.fallback.weights[weight_key]=weight
        weight=self.fallback.weights[weight_key]
        (bm,bn,bk,split,warps,stages),variant,groups=self.configuration(m,n,k,config)
        cached=self.prequantized.pop((xx.data_ptr(),m,k,id(norm)),None) if norm is not None else None
        q,qs=cached if cached is not None else (torch.empty((3,m,k),device=x.device,dtype=torch.int8),
                                               torch.empty((3,m),device=x.device,dtype=torch.float32))
        y=torch.empty((b,t,n),device=x.device,dtype=torch.float32)
        partial=torch.empty((split,m,n),device=x.device) if split>1 else y
        if cached is None:
            _quantize[(m,)](xx,q,qs,norm.weight if norm is not None else None,
                           norm.bias if norm is not None else None,m,k,t,*xx.stride(),
                           triton.next_power_of_2(k),3,norm is not None,
                           norm.eps if norm is not None else 0.,
                           INPUT_SILU=input_activation=='silu',enable_fp_fusion=False)
        key=(m,n,k,bm,bn,bk,split,warps,stages,mod.bias is not None,activation,groups)
        grid=(triton.cdiv(m,bm),triton.cdiv(n,bn),split)
        if key not in self.kernels:
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('warm CUDA kernels before capture')
            compiled=_int8_dot[grid](q,qs,weight,mod.scale,mod.bias,y,partial,m,n,k,
                        mod.bias is not None,bm,bn,bk,split,3,ACTIVATION=activation,
                        DENSE=True,COLUMN=True,WEIGHT_BITS=self.weight_bits,CTA_GROUPS=groups,num_warps=warps,num_stages=stages)
            self._compile_native(compiled,key,variant)
        native,count,threads,shared=self.kernels[key]
        # Triton specializes None arguments away, removing a bias pointer from
        # the PTX ABI. Its two unused scratch pointers must still be supplied.
        pointers=[q,qs,weight,mod.scale]+([mod.bias] if mod.bias is not None else [])+[y,partial]
        if len(pointers)>count:raise ValueError('unexpected Triton PTX pointer signature')
        pointers+=[self.null]*(count-len(pointers))
        native(grid=grid,block=(threads,1,1),shared_mem=shared,args=pointers)
        if self.defer_epilogue and n==1024 and mod.packed_t.data_ptr() not in self.no_defer_weights and activation=='':
            from .row_fusion import DeferredProjection
            return DeferredProjection(partial,mod.scale,mod.bias,split,(b,t,n))
        if split>1:
            _finish[(triton.cdiv(m*n,256),)](partial,mod.scale,mod.bias,y,
                    m,n,t,t*n,n,1,mod.bias is not None,split,256,ACTIVATION=activation)
        return y.transpose(1,2) if conv else y.reshape(*x.shape[:-1],n)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path,default=DEFAULT_EXPORT,
                        help='model manifest for provenance only; fixtures are synthetic')
    parser.add_argument('--rows',type=int,nargs='+',default=[38,126,376])
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--check-only',action='store_true',help='correctness only, suitable for Compute Sanitizer')
    args=parser.parse_args()
    if args.limit<0 or any(m<1 for m in args.rows):
        parser.error('rows must be positive and limit nonnegative')
    paths.require_mount()
    out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-ptx-cuda'
    out.mkdir();print('RESULTS',out,flush=True)
    with paths.gpu_lock('CUDA reconstruction from Triton PTX'),torch.inference_mode(),energy_meter() as meter:
        meter._guard();require_power_cap(meter)
        save(out,'gpu.json',meter.metadata())
        save(out,'metadata.json',metadata(args))
        (out/'ptx_cuda.py').write_bytes(Path(__file__).read_bytes())
        # Reuse the headerless NVRTC environment setup from the CUDA experiments.
        from .cuda_matmul import kernel
        kernel('direct')
        from torch.cuda._utils import _nvrtc_compile,_cuda_load_module
        torch.manual_seed(419)
        torch.backends.cuda.matmul.allow_tf32=False
        results=[]
        shapes=[(m,n,k) for m in args.rows for n,k in
                [(1024,1024),(3072,1024),(4096,1024),(2048,1024),(1024,4096)]]
        if args.limit:shapes=shapes[:args.limit]
        for m,n,k in shapes:
            require_power_cap(meter)
            bm,bn,bk,split,warps,stages=expanded_configuration(m,n,k)
            q=torch.randint(-127,128,(3,m,k),device='cuda',dtype=torch.int8)
            qs=torch.rand(3,m,device='cuda')*.01
            w=torch.randint(-1,2,(n,k),device='cuda',dtype=torch.int8).T
            ws=torch.rand(n,device='cuda')*.03
            bias=torch.randn(n,device='cuda')*.01
            y0=torch.empty(m,n,device='cuda');p0=torch.empty(split,m,n,device='cuda')
            y1=torch.empty_like(y0);p1=torch.empty_like(p0)
            grid=(triton.cdiv(m,bm),triton.cdiv(n,bn),split)
            def original():
                compiled=_int8_dot[grid](q,qs,w,ws,bias,y0,p0,m,n,k,True,bm,bn,bk,split,3,
                          DENSE=True,COLUMN=True,num_warps=warps,num_stages=stages)
                if split>1:
                    _finish[(triton.cdiv(m*n,256),)](p0,ws,bias,y0,m,n,m,m*n,n,1,True,split,256)
                return compiled
            compiled=original();torch.cuda.synchronize()
            if compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
                raise ValueError('scratch-using kernels require a separate launcher')
            ptx=compiled.asm['ptx']
            source,count,threads=cuda_from_ptx(ptx)
            assert count>=7 and threads==warps*32
            folder=out/f'm{m}-n{n}-k{k}';folder.mkdir()
            (folder/'original.ptx').write_text(ptx)
            (folder/'reconstructed.cu').write_text(source)
            binary,symbol=_nvrtc_compile(source,'encoder_from_triton','120a',nvcc_options=['--std=c++17','--fmad=false'])
            (folder/'reconstructed.cubin').write_bytes(binary)
            (folder/'original.cubin').write_bytes(compiled.asm['cubin'])
            native=_cuda_load_module(binary,[symbol])[symbol]
            null=torch.empty(0,device='cuda',dtype=torch.uint8)
            assert null.data_ptr()==0
            pointers=[q,qs,w,ws,bias,y1,p1]+[null]*(count-7)
            shared=compiled.metadata.shared
            if shared>=48*1024:native.set_shared_memory_config(shared)
            def reconstructed():
                native(grid=grid,block=(threads,1,1),shared_mem=shared,args=pointers)
                if split>1:
                    _finish[(triton.cdiv(m*n,256),)](p1,ws,bias,y1,m,n,m,m*n,n,1,True,split,256)
            reconstructed();torch.cuda.synchronize()
            torch.testing.assert_close(y1,y0,atol=0,rtol=0)
            reference=sum((q[i].float() @ w.float())*qs[i,:,None] for i in range(3))*ws+bias
            torch.testing.assert_close(y1,reference,atol=2e-6,rtol=2e-5)
            timings={'triton':[],'cuda_inline_ptx':[]}
            functions={'triton':original,'cuda_inline_ptx':reconstructed}
            if args.check_only:
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):reconstructed()
                # Reuse the captured pointers with different input contents.
                q.copy_(torch.randint(-127,128,q.shape,device='cuda',dtype=torch.int8))
                original();graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(y1,y0,atol=0,rtol=0)
                del graph
            else:
                for repeat in range(3):
                    for name in (list(functions) if repeat%2==0 else list(functions)[::-1]):
                        timings[name].append(triton.testing.do_bench_cudagraph(functions[name],rep=25)*1000)
            row={'m':m,'n':n,'k':k,'config':[bm,bn,bk,split,warps,stages],
                 'shared_bytes':shared,'pointer_arguments':count,'bitwise_output_match':True,
                 'original_ptx_lines':len(ptx.splitlines()),
                 'original_cubin_matches_reconstructed':binary==compiled.asm['cubin'],
                 'median_us':{name:statistics.median(v) for name,v in timings.items() if v},
                 'timing_windows_us':timings,
                 'scope':'synthetic resident prequantized dot + same finish kernel; CUDA-graph microbenchmark or check-only; no energy or full encoder measurement'}
            results.append(row);save(out,'results.json',results)
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
