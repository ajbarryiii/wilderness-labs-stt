"""W2A2 correctness and fresh latency comparisons, called under the GPU lock."""
import math
import statistics

import torch

from kernels.packed import PackedWeight
from kernels.popcount import pack_plane
from kernels.popcount_ternary_activation import BitWeight, load, pack
from paths import save
from popcount_benchmark import SHAPES, capture, timing


def quantize(x):
    return torch.where(x >= .5, 1., torch.where(x <= -.5, -1., 0.))


def run(job, policy):
    load()
    torch.manual_seed(9318)
    cases = []
    for m, n, k in [(1,7,1),(3,35,31),(5,19,33),(2,9,1031),(3,37,1024),(4,33,4096)]:
        codes = torch.randint(-1,2,(n,k),dtype=torch.int8)
        codes[0].zero_()
        scales = torch.linspace(-.7,1.3,n)/math.sqrt(k)
        bias = torch.linspace(-.25,.25,n).half()
        x_cpu = torch.randn(m,k).half()
        x_cpu[:,::7] = .5
        x_cpu[:,::11] = -.5
        x_cpu[:,::13] = 0
        if m > 1:
            x_cpu[0].zero_()
        x = x_cpu.cuda()
        w = BitWeight(codes.cuda(),scales.cuda(),2)
        a = torch.empty(m,(k+31)//32,device='cuda',dtype=torch.int64)
        guard = torch.full((m*n+16,),321.,device='cuda',dtype=torch.float16)
        out = guard[8:-8].view(m,n)
        b = bias.cuda()
        for variant in ('warp','tile'):
            def op():
                pack(x,a)
                w.linear(a,out,variant,b)
            op()
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                op()
            stream.synchronize()
            for change in (False,True):
                if change:
                    x_cpu.neg_()
                    x.copy_(x_cpu)
                graph.replay()
                torch.cuda.synchronize()
                expected=((quantize(x_cpu)@codes.float().T)*scales.half().float()+bias.float()).half()
                torch.testing.assert_close(out.cpu(),expected,atol=0,rtol=0)
                torch.testing.assert_close(a.cpu().to(torch.int32),pack_plane(x_cpu>=0),atol=0,rtol=0)
                torch.testing.assert_close((a.cpu()>>32).to(torch.int32),pack_plane(x_cpu.abs()>=.5),atol=0,rtol=0)
                assert bool((guard[:8]==321).all() & (guard[-8:]==321).all())
            cases.append(dict(m=m,n=n,k=k,variant=variant,exact=True))
    save(job/'w2a2-correctness.json',cases)
    policy=dict(policy)
    rows=[]
    for m,n,k in SHAPES:
        codes=torch.randint(-1,2,(n,k),device='cuda',dtype=torch.int8)
        p=PackedWeight.from_codes(codes,2,1/math.sqrt(k))
        w=BitWeight(codes,p.scales,2)
        x=torch.randn(m,k,device='cuda',dtype=torch.float16)
        a=torch.empty(m,(k+31)//32,device='cuda',dtype=torch.int64)
        out=torch.empty(m,n,device='cuda',dtype=torch.float16)
        pack(x,a)
        expected=p.linear(quantize(x).half())
        selection={}
        for variant in ('warp','tile'):
            w.linear(a,out,variant)
            torch.testing.assert_close(out,expected,atol=.003,rtol=.003)
            g=capture(lambda:w.linear(a,out,variant),16)
            selection[variant]=statistics.median(timing(g,16,5))
            del g
        variant=min(selection,key=selection.get)
        policy[f'2,{m},{n},{k}']=variant
        def packed_op():
            pack(x,a)
            w.linear(a,out,variant)
        ops=dict(baseline=lambda:p.linear(x,out=out),prepacked=lambda:w.linear(a,out,variant),pack_dot=packed_op)
        count=2048 if m<=4 else 32
        graphs={name:capture(op,count) for name,op in ops.items()}
        samples={name:[] for name in ops}
        for repeat in range(3):
            for name in (list(ops) if repeat%2==0 else list(reversed(ops))):
                samples[name].extend(timing(graphs[name],count))
        rows.append(dict(m=m,n=n,k=k,variant=variant,selection_us=selection,
                         median_us={name:statistics.median(v) for name,v in samples.items()},
                         samples_us=samples))
        save(job/'w2a2-kernel-latency.json',rows)
        print('W2A2 kernel',m,n,k,rows[-1]['median_us'],flush=True)
        del graphs
    return policy


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    from energy import EnergyMeter
    from paths import artifact
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args()
    job=artifact(args.job)
    with EnergyMeter() as meter, torch.inference_mode():
        meter._guard()
        policy=run(job,json.loads((job/'initial-policy.json').read_text()))
        meter._guard()
        save(job/'w2a2-policy.json',policy)
