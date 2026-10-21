"""Current encoder CUDA-kernel profile plus graph-timed Conformer layers."""
import argparse,collections,contextlib,json,time
from pathlib import Path
import torch
import paths
from .benchmark import DEFAULT_EXPORT,capture,energy_meter,metadata,save
from .runtime import load_packed
from .sprint4 import variant,require_power_cap


def main():
 import evaluate as ev
 from nemo.utils import logging
 logging.setLevel(logging.ERROR)
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',type=Path,default=DEFAULT_EXPORT)
 p.add_argument('--variants',nargs='+',default=['baseline','torch_col']);p.add_argument('--duration',type=float,default=10)
 args=p.parse_args();paths.require_mount()
 out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-profile4';out.mkdir();print('RESULTS',out,flush=True)
 with paths.gpu_lock('layer-level encoder profile'),torch.inference_mode(),energy_meter() as meter:
  meter._guard();require_power_cap(meter);save(out,'metadata.json',metadata(args))
  model=load_packed(args.export)
  rs=[json.loads(s) for s in (paths.MANIFESTS/'test_librispeech_clean.jsonl').read_text().splitlines()]
  rec=min(rs,key=lambda r:abs(r['duration']-args.duration));a,l=ev.audio_batch([rec]);a,l=a.cuda(),l.cuda()
  with ev.inference_settings(model),ev.strict_fp32('cuda'):
   f,fl=model.preprocessor(input_signal=a,length=l)
   for name in args.variants:
    with variant(model,name):
     graph,output=capture(lambda:model.encoder(audio_signal=f,length=fl))
     for _ in range(10):graph.replay()
     torch.cuda.synchronize()
     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
      for _ in range(3):graph.replay()
      torch.cuda.synchronize()
     prof.export_chrome_trace(str(out/(name+'-trace.json')))
     events=json.loads((out/(name+'-trace.json')).read_text())['traceEvents'];groups=collections.defaultdict(lambda:[0,0])
     for e in events:
      if e.get('cat')=='kernel':groups[e['name']][0]+=e.get('dur',0)/3;groups[e['name']][1]+=1/3
     rows=sorted([dict(name=n,us=v[0],count=round(v[1])) for n,v in groups.items()],key=lambda r:-r['us'])
     save(out,name+'-kernels.json',rows);print(name,json.dumps(rows[:14]),flush=True)
     del graph,output
     hooks=[];stamps=[]
     for layer in model.encoder.layers:
      start=torch.cuda.Event(enable_timing=True,external=True);end=torch.cuda.Event(enable_timing=True,external=True)
      stamps.append((start,end))
      hooks.append(layer.register_forward_pre_hook(lambda module,inputs,event=start:event.record()))
      hooks.append(layer.register_forward_hook(lambda module,inputs,outputs,event=end:event.record()))
     try:
      graph,output=capture(lambda:model.encoder(audio_signal=f,length=fl))
      for _ in range(10):graph.replay()
      torch.cuda.synchronize();save(out,name+'-layers.json',[dict(layer=i,ms=a.elapsed_time(b)) for i,(a,b) in enumerate(stamps)])
      del graph,output
     finally:
      for hook in hooks:hook.remove()


if __name__=='__main__':main()
