"""Offline INT8x3 tile sweep including preparation and reduction costs."""
import json
import statistics
import time

import torch
import triton
from safetensors import safe_open

import paths
from export import unpack_codes
from .benchmark import DEFAULT_EXPORT, energy_meter, save
from .fused_kernels import _quantize, _int8_dot
from .kernels import _finish
from .runtime import PackedLinear


def main():
    paths.require_mount()
    out = paths.ARTIFACTS / "kernel-runs" / f"{time.strftime('%Y%m%dT%H%M%S')}-integer-tiles"
    out.mkdir(); print("RESULTS", out, flush=True)
    manifest = json.loads((DEFAULT_EXPORT / "manifest.json").read_text())
    torch.manual_seed(571)
    rows, seen = [], set()
    with paths.gpu_lock("integer tile sweep"), torch.inference_mode(), energy_meter() as meter:
        meter._guard(); torch.backends.cuda.matmul.allow_tf32 = False
        with safe_open(DEFAULT_EXPORT / manifest["file"], framework="pt") as f:
            for name, layer in manifest["quantized_layers"].items():
                n, k = layer['shape']
                if (n,k) in seen: continue
                seen.add((n,k))
                codes, scales = f.get_tensor(name+'.codes'), f.get_tensor(name+'.scale')
                mod = PackedLinear(codes,scales,k).cuda()
                w = (unpack_codes(codes,k).float()*scales[:,None]).cuda()
                for m in [38,126,376]:
                    x = torch.randn(m,k,device='cuda'); ref = torch.nn.functional.linear(x,w)
                    q = torch.empty((3,m,k),dtype=torch.int8,device='cuda')
                    qs = torch.empty((3,m),device='cuda'); y = torch.empty((m,n),device='cuda')
                    configs = [(32,64,64,4,4,3)] + [(bm,bn,bk,s,warps,2)
                        for bm,bn,bk,warps in [(32,64,64,4),(32,64,128,4),(64,64,64,4),(64,64,128,4),(32,128,64,4)]
                        for s in [1,2,4,8]]
                    candidates=[]
                    for config in configs:
                        bm,bn,bk,split,warps,stages=config
                        partial=torch.empty((split,m,n),device='cuda')
                        def run():
                            _quantize[(m,)](x,q,qs,None,None,m,k,m,m*k,k,1,triton.next_power_of_2(k),3,False,0.,enable_fp_fusion=False)
                            _int8_dot[(triton.cdiv(m,bm),triton.cdiv(n,bn),split)](
                                q,qs,mod.packed_t,mod.scale,None,y,partial,m,n,k,False,bm,bn,bk,split,3,num_warps=warps,num_stages=stages)
                            if split > 1:
                                _finish[(triton.cdiv(m*n,256),)](partial,mod.scale,None,y,m,n,m,m*n,n,1,False,split,256)
                            return y
                        try:
                            actual=run();torch.testing.assert_close(actual,ref,atol=1e-4,rtol=2e-5)
                            us=[triton.testing.do_bench_cudagraph(run,rep=30)*1000 for _ in range(2)]
                            candidates.append(dict(config=config,us=us,median_us=statistics.median(us)))
                        except Exception as exc:candidates.append(dict(config=config,error=repr(exc)))
                    good=sorted((v for v in candidates if 'median_us' in v),key=lambda v:v['median_us'])
                    row=dict(m=m,n=n,k=k,baseline=candidates[0],best=good[0],candidates=candidates)
                    rows.append(row);save(out,'tiles.json',rows)
                    print(json.dumps({k:v for k,v in row.items() if k!='candidates'}),flush=True)


if __name__ == '__main__':main()
