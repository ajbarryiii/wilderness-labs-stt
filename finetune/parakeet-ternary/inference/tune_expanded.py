"""Offline expanded-column INT8 sweep; includes quantization and epilogue."""
import json,statistics,time
import torch,triton
from safetensors import safe_open
import paths
from export import unpack_codes
from .benchmark import DEFAULT_EXPORT,energy_meter,save
from .runtime import PackedLinear
from .encoder4 import ExpandedProjection
from .fused_kernels import integer_configuration
from .sprint4 import require_power_cap


def main():
 paths.require_mount();out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-expanded-tiles';out.mkdir()
 print('RESULTS',out,flush=True);manifest=json.loads((DEFAULT_EXPORT/'manifest.json').read_text());rows=[]
 with paths.gpu_lock('expanded integer tile tuning'),torch.inference_mode(),energy_meter() as meter:
  meter._guard();require_power_cap(meter);save(out,'gpu.json',meter.metadata());torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(417)
  with safe_open(DEFAULT_EXPORT/manifest['file'],framework='pt') as f:
   choices={}
   for name,layer in manifest['quantized_layers'].items():choices.setdefault(tuple(layer['shape']),[name])
   choices[(3072,1024)]=[f'encoder.layers.0.self_attn.linear_{v}' for v in 'qkv']
   for (n,k),names in choices.items():
    packed=torch.cat([f.get_tensor(name+'.codes') for name in names]);scales=torch.cat([f.get_tensor(name+'.scale') for name in names])
    mod=PackedLinear(packed,scales,k).cuda();w=(unpack_codes(packed,k).float()*scales[:,None]).cuda()
    provider=ExpandedProjection('triton_col')
    for m in [38,126,376]:
     require_power_cap(meter);x=torch.randn(m,k,device='cuda');ref=torch.nn.functional.linear(x,w)
     configs=[integer_configuration(m,n,k)]+[(bm,bn,bk,s,nw,st)
             for bm,bn,bk,nw,st in [(32,64,128,4,2),(32,64,256,4,2),(32,128,128,4,2),
                                    (64,64,128,4,2),(64,64,128,8,3),(16,64,128,4,2)] for s in [1,2,4,8]]
     configs=list(dict.fromkeys(configs));candidates=[]
     for config in configs:
      provider.config=config
      def run():return provider(x,mod)
      try:
       y=run();torch.testing.assert_close(y,ref,atol=1e-4,rtol=2e-5)
       us=[triton.testing.do_bench_cudagraph(run,rep=15)*1000 for _ in range(2)]
       candidates.append(dict(config=config,us=us,median_us=statistics.median(us)))
      except Exception as exc:candidates.append(dict(config=config,error=repr(exc)))
     good=sorted((c for c in candidates if 'median_us' in c),key=lambda c:c['median_us'])
     row=dict(m=m,n=n,k=k,baseline=candidates[0],best=good[0],candidates=candidates)
     rows.append(row);save(out,'tiles.json',rows);print(json.dumps({k:v for k,v in row.items() if k!='candidates'}),flush=True)
    del provider,mod,w


if __name__=='__main__':main()
