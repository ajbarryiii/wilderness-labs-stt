"""GPU correctness gates for each optimized popcount component and runtime."""
import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from kernels import popcount, popcount_ternary_activation as pa2, popcount_optimized as opt
from kernels.packed import PackedWeight
from paths import save


def exact(a,b):
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def components():
    from types import SimpleNamespace
    torch.set_grad_enabled(False)
    torch.manual_seed(20260912)
    rows=[]
    for a2 in (False,True):
        base=pa2 if a2 else popcount
        # Every finite FP16 value exercises signed zero, underflow and thresholds.
        values=torch.arange(65536,dtype=torch.int32,device='cuda').to(torch.int16).view(torch.float16)
        values=values[values.isfinite()].reshape(1,-1)
        for gelu in (False,True):
            expected=torch.empty((1,(values.numel()+31)//32),dtype=torch.int64 if a2 else torch.int32,device='cuda')
            base.pack(F.gelu(values) if gelu else values,expected)
            exact(opt.pack(values,a2,gelu=gelu).words,expected)
        rows.append(dict(test='all_finite_fp16_pack_and_gelu',a2=a2,values=values.numel()))
        for k in (33,64,1024):
            x=torch.randn(4*k+1,device='cuda',dtype=torch.float16)[1:].view(4,k)
            norm=SimpleNamespace(weight=torch.randn(k+1,device='cuda',dtype=torch.float16)[1:],
                                 bias=torch.randn(k+1,device='cuda',dtype=torch.float16)[1:])
            y=F.layer_norm(x,(k,),norm.weight,norm.bias,1e-5)
            exact(opt.pack(x,a2,norm=norm).words,opt.pack(y,a2).words)
            rows.append(dict(test='unaligned_norm_fallback',a2=a2,k=k))
        for k in (4,32,64,128,240,1024,4096):
            for m in (1,4,17):
                x=torch.randn(m,k,device='cuda',dtype=torch.float16)
                gamma=torch.randn(k,device='cuda',dtype=torch.float16)
                bias=torch.randn(k,device='cuda',dtype=torch.float16)
                norm=SimpleNamespace(weight=gamma,bias=bias)
                y=F.layer_norm(x,(k,),gamma,bias,1e-5)
                expected=torch.empty((m,(k+31)//32),dtype=torch.int64 if a2 else torch.int32,device='cuda')
                base.pack(y,expected)
                actual=opt.pack(x,a2,norm=norm).words
                exact(actual,expected)
                rows.append(dict(test='norm_pack',a2=a2,m=m,k=k))
        for bits in ((2,) if a2 else (1,2)):
            for m,n,k in ((1,19,73),(4,67,65),(1,1024,1024),(1,1024,4096),
                          (17,67,73),(16,1024,1024),(33,64,240)):
                codes=torch.randint(-1,2,(n,k),dtype=torch.int8,device='cuda')
                if bits==1:
                    codes=torch.where(codes==0,1,codes)
                else:
                    codes[0].zero_()
                scales=torch.randn(n,device='cuda',dtype=torch.float32)*.07
                bias=torch.randn(n,device='cuda',dtype=torch.float16)
                x=torch.randn(m,k,device='cuda',dtype=torch.float16)
                x[:,0:3]=torch.tensor([0.,.5,-.5],device='cuda',dtype=torch.float16)
                a=opt.pack(x,a2).words
                reference=base.BitWeight(codes,scales,bits)
                weight=opt.BitWeight(codes,scales,bits)
                expected=torch.empty(m,n,device='cuda',dtype=torch.float16)
                reference.linear(a,expected,'warp',bias)
                for layout in ('warp','tile'):
                    for counts in (False,True):
                        backing=torch.full((m*n+32,),91.,device='cuda',dtype=torch.float16)
                        actual=backing[16:-16].view(m,n)
                        weight.linear(a,actual,layout,bias,a2=a2,counts=counts)
                        exact(actual,expected)
                        exact(backing[:16],backing[-16:])
                        assert (backing[:16]==91).all()
                        residual=torch.randn_like(actual)
                        weight.linear(a,actual,layout,bias,a2=a2,counts=counts,mode=1,residual=residual)
                        exact(actual,expected+residual)
                        weight.linear(a,actual,layout,bias,a2=a2,counts=counts,mode=1,residual=-expected)
                        assert not actual.count_nonzero().item(), 'Skipped FP16 intermediate rounding'
                        weight.linear(a,actual,layout,bias,a2=a2,counts=counts,mode=2)
                        torch.testing.assert_close(actual,F.gelu(expected),atol=.002,rtol=.001)
                        exact(opt.pack(actual,a2).words,opt.pack(F.gelu(expected),a2).words)
                packed=PackedWeight.from_codes(codes,bits,scales)
                exact(opt.quantized_gemm(packed,x,bias,a2=a2),expected)
                rows.append(dict(test='dot_epilogues_hybrid',a2=a2,bits=bits,m=m,n=n,k=k))
            # Direct QKV writes, two batches, ragged N and head width.
            m,width,heads,capacity=2,35,5,9
            codes=torch.randint(-1,2,(3*width,73),device='cuda',dtype=torch.int8)
            if bits==1: codes=torch.where(codes==0,1,codes)
            weight=opt.BitWeight(codes,torch.randn(3*width,device='cuda'),bits)
            a=opt.pack(torch.randn(m,73,device='cuda',dtype=torch.float16),a2).words
            expected=torch.empty(m,3*width,device='cuda',dtype=torch.float16)
            weight.linear(a,expected,a2=a2)
            for layout in ('warp','tile'):
                for step in (0,4,8):
                    cache=tuple(torch.full((m,heads,capacity,width//heads),57.,device='cuda',dtype=torch.float16) for _ in range(2))
                    q=torch.empty(m,width,device='cuda',dtype=torch.float16)
                    weight.linear(a,q,layout,a2=a2,mode=3,cache=cache,step=step)
                    exact(q,expected[:,:width])
                    for i,c in enumerate(cache):
                        exact(c[:,:,step,:].reshape(m,width),expected[:,(i+1)*width:(i+2)*width])
                        assert (c[:,:,:step]==57).all() and (c[:,:,step+1:]==57).all()
            rows.append(dict(test='qkv_cache_canaries',a2=a2,bits=bits))
    return rows


def models(policy,a2_policy):
    from runtime_model import ReplayWhisper,WhisperConfig,_Norm,_Block
    from popcount_model_benchmark import install as baseline_install
    from popcount_optimized_runtime import install,FEATURES
    original=PackedWeight.linear
    rows=[]
    for bits,a2 in ((1,False),(2,False),(2,True)):
        selected=a2_policy if a2 else policy
        model=ReplayWhisper(WhisperConfig.tiny_smoke(),distribution='binary' if bits==1 else 'ternary')
        mel=torch.randn(1,8,32,device='cuda',dtype=torch.float16)
        tokens=[3,4,5,6]
        baseline_install(selected,a2)
        encoded=model.encode(mel).clone()
        expected=model.decode(encoded,tokens).clone()
        predictions=model.last_predictions.clone()
        PackedWeight.linear=original
        for candidate in FEATURES:
            restore=install(selected,a2=a2,candidate=candidate)
            try:
                actual_encoded=model.encode(mel)
                actual=model.decode(actual_encoded,tokens)
                exact(actual_encoded,encoded)
                exact(actual,expected)
                exact(model.last_predictions,predictions)
                if candidate in ('all','selected'):
                    model.capture(mel,tokens)
                    changed=mel*.75+.1
                    eager=model.run(changed,tokens).clone()
                    actual=model.replay_graph(changed)
                    torch.cuda.synchronize()
                    exact(actual,eager)
                    stream=torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        streamed=model.run(mel,tokens).clone()
                    torch.cuda.current_stream().wait_stream(stream)
                    exact(streamed,expected)
                rows.append(dict(test='whole_tiny_model_exact',bits=bits,a2=a2,candidate=candidate))
            finally:
                restore()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--components-only',action='store_true')
    args=parser.parse_args()
    from energy import EnergyMeter
    with EnergyMeter(synchronize=torch.cuda.synchronize) as meter:
        meter._guard()
        rows=components()
        save(args.job/'components.json',rows)
        print(f'{len(rows)} component groups passed',flush=True)
        if not args.components_only:
            cfg=json.loads((args.job/'config.json').read_text())
            rows=models(cfg['policy'],cfg['ternary_policy'])
            save(args.job/'tiny-model-checks.json',rows)
            print(f'{len(rows)} model checks passed',flush=True)


if __name__=='__main__':
    main()
