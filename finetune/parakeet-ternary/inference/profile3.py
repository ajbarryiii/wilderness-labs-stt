"""Profile the second-pass optimized pipeline and save raw CUDA kernel events."""
import collections, json, time
import torch
import paths
from .benchmark import DEFAULT_EXPORT, energy_meter, save
from .optimized import enable_optimizations, disable_optimizations
from .runtime import load_packed


def main():
 import evaluate as ev
 from nemo.utils import logging
 logging.setLevel(logging.ERROR)
 paths.require_mount()
 out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-profile3';out.mkdir()
 print('RESULTS',out,flush=True)
 with paths.gpu_lock('profile current optimized pipeline'),torch.inference_mode(),energy_meter() as meter:
  meter._guard()
  model=load_packed(DEFAULT_EXPORT)
  with ev.inference_settings(model),ev.strict_fp32('cuda'):
   enable_optimizations(model,decoder=False,position_dot=False)
   print('DECODER',model.decoder,'JOINT',model.joint,flush=True)
   comp=model.decoding.decoding.decoding_computer
   comp.force_cuda_graphs_mode('full_graph')
   rs=[json.loads(l) for l in (paths.MANIFESTS/'test_librispeech_clean.jsonl').read_text().splitlines()]
   rec=min(rs,key=lambda r:abs(r['duration']-10));a,l=ev.audio_batch([rec]);a,l=a.cuda(),l.cuda()
   def enc():return model(input_signal=a,input_signal_length=l)
   e,el=enc()
   def dec():return model.decoding.rnnt_decoder_predictions_tensor(e,el,return_hypotheses=False)
   def full():
    x,xl=enc();return model.decoding.rnnt_decoder_predictions_tensor(x,xl,return_hypotheses=False)
   for _ in range(5):full()
   data={}
   for name,fn in [('encoder',enc),('decoder',dec),('full',full)]:
    w=meter.measure(fn,min_seconds=3);save(out,name+'.json',w)
    data[name]={'ms':sum(w['iteration_seconds'])/len(w['iteration_seconds'])*1000,'J':w['joules_per_iteration']}
   print(data,flush=True);save(out,'summary.json',data)
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(3):full()
    torch.cuda.synchronize()
   prof.export_chrome_trace(str(out/'trace.json'))
   (out/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=45))
   events=json.loads((out/'trace.json').read_text())['traceEvents'];groups=collections.defaultdict(lambda:[0,0])
   for event in events:
    if event.get('cat')=='kernel':
     groups[event['name']][0]+=event.get('dur',0);groups[event['name']][1]+=1
   rows=sorted([{'name':n,'us':v[0]/3,'count':v[1]/3} for n,v in groups.items()],key=lambda r:-r['us'])
   save(out,'kernels.json',rows);print(json.dumps(rows[:25],indent=2),flush=True)
   disable_optimizations(model)
if __name__=='__main__':main()
