"""Screen decoder fusions against the unchanged second-pass optimized runtime."""
import argparse, contextlib, json, statistics, time, traceback
from pathlib import Path
import torch
import paths
from .benchmark import DEFAULT_EXPORT, energy_meter, save, metadata
from .runtime import load_packed
from .optimized import enable_optimizations,disable_optimizations
from .decoder_kernels import decoder_transform, control_transform

VARIANTS={
 'baseline':None,'joint':{'joint':True},'lstm':{'joint':False,'lstm':True},
 'lookup':{'joint':False,'lstm':True,'precompute':True},
 'combined':{'joint':True,'lstm':True,'precompute':True},
 'combined_b1':{'joint':True,'lstm':True,'precompute':True,'block':1},
 'combined_b4':{'joint':True,'lstm':True,'precompute':True,'block':4},
 'combined_b8':{'joint':True,'lstm':True,'precompute':True,'block':8},
 'control':{'joint':False,'control':True},
 'all':{'joint':True,'lstm':True,'precompute':True,'block':1,'control':True},
 'all_storage':{'joint':True,'lstm':True,'precompute':True,'block':1,'control':True,'storage':True},
}

@contextlib.contextmanager
def variant(model,name):
 options=dict(VARIANTS[name] or {})
 control=options.pop('control',False)
 storage=options.pop('storage',False)
 with contextlib.ExitStack() as stack:
  if options:stack.enter_context(decoder_transform(model,**options))
  if control:stack.enter_context(control_transform(model,storage=storage))
  yield

def text(out):
 if isinstance(out,tuple):out=out[0]
 return [h.text if hasattr(h,'text') else str(h) for h in out]

def main():
 import evaluate as ev
 from nemo.utils import logging
 logging.setLevel(logging.ERROR)
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variants',nargs='+',default=list(VARIANTS))
 p.add_argument('--durations',nargs='+',type=float,default=[10]);p.add_argument('--seconds',type=float,default=3)
 p.add_argument('--export',type=Path,default=DEFAULT_EXPORT);args=p.parse_args()
 paths.require_mount();out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-sprint3';out.mkdir()
 print('RESULTS',out,flush=True);rows=[]
 with paths.gpu_lock('decoder kernel screen'),torch.inference_mode(),energy_meter() as meter:
  meter._guard();save(out,'metadata.json',metadata(args));model=load_packed(args.export)
  with ev.inference_settings(model),ev.strict_fp32('cuda'):
   enable_optimizations(model,decoder=False,position_dot=False);model.decoding.decoding.decoding_computer.force_cuda_graphs_mode('full_graph')
   rs=[json.loads(l) for l in (paths.MANIFESTS/'test_librispeech_clean.jsonl').read_text().splitlines()]
   # Direct predictor check catches state/layout errors independently of text.
   torch.manual_seed(18)
   states=[torch.randn(2,2,640,device='cuda')*.1 for _ in range(2)]
   tok=torch.tensor([[13],[1024]],device='cuda')
   ref=model.decoder.predict(tok,states,False)
   for name in args.variants:
    try:
     with variant(model,name):
      actual=model.decoder.predict(tok,states,False)
      torch.testing.assert_close(actual,ref,atol=2e-6,rtol=2e-4)
     print('numeric',name,'PASS',flush=True)
    except Exception as exc:print('numeric',name,repr(exc),flush=True);raise
   for duration in args.durations:
    rec=min(rs,key=lambda r:abs(r['duration']-duration));a,l=ev.audio_batch([rec]);a,l=a.cuda(),l.cuda()
    def enc():return model(input_signal=a,input_signal_length=l)
    e,el=enc()
    def dec():return model.decoding.rnnt_decoder_predictions_tensor(e,el,return_hypotheses=False)
    def full():
     x,xl=enc();return model.decoding.rnnt_decoder_predictions_tensor(x,xl,return_hypotheses=False)
    reference=text(full())
    for name in args.variants:
     try:
      with variant(model,name):
       hypothesis=text(full());assert hypothesis==reference,(name,hypothesis,reference)
       for label,fn in [('decoder',dec),('full',full)]:
        start=time.perf_counter()
        while time.perf_counter()-start<1:fn();torch.cuda.synchronize()
        w=meter.measure(fn,min_seconds=args.seconds)
        save(out,f'{duration:g}-{name}-{label}.json',w)
        row={'variant':name,'seconds':rec['duration'],'scope':label,'median_ms':statistics.median(w['iteration_seconds'])*1000,
             'J':w['joules_per_iteration'],'W':w['average_watts'],'text':hypothesis}
        rows.append(row);print(json.dumps(row),flush=True);save(out,'screen.json',rows)
     except Exception as exc:
      rows.append({'variant':name,'seconds':duration,'error':repr(exc)});save(out,'screen.json',rows);traceback.print_exc()
   disable_optimizations(model)
if __name__=='__main__':main()
