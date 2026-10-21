"""Validate new decoder path at batch one on shared/batched encoder inputs."""
import argparse,contextlib,json,time
from pathlib import Path
import torch
import paths
from .benchmark import DEFAULT_EXPORT,records_for_sets,energy_meter,metadata,save
from .runtime import load_packed
from .optimized import enable_optimizations,disable_optimizations
from .graphs import enable_pipeline_graphs,disable_pipeline_graphs
from .transforms import encoder_transform
from .decoder_kernels import decoder_transform,control_transform


def main():
 import evaluate as ev
 from nemo.utils import logging
 logging.setLevel(logging.ERROR)
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',type=Path,default=DEFAULT_EXPORT)
 p.add_argument('--utterances-per-set',type=int,default=256);p.add_argument('--position-dot',action='store_true')
 p.add_argument('--reference-only',action='store_true');p.add_argument('--limit-batches',type=int,default=0)
 p.add_argument('--decoder-variant',default='all_storage')
 p.add_argument('--capture-pool',action='store_true')
 args=p.parse_args();paths.require_mount()
 out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-quality3';out.mkdir();print('RESULTS',out,flush=True)
 records=records_for_sets(args.utterances_per_set);rows=[];errors=[]
 with paths.gpu_lock('third-pass quality validation'),torch.inference_mode(),energy_meter() as meter,contextlib.ExitStack() as stack:
  meter._guard();save(out,'metadata.json',metadata(args));stack.enter_context(ev.strict_fp32('cuda'))
  old=load_packed(args.export);new=load_packed(args.export)
  for model in [old,new]:
   stack.enter_context(ev.inference_settings(model));model.decoding.decoding.decoding_computer.force_cuda_graphs_mode('full_graph')
   if model is new:
    from .graph_memory import decoder_graph_pool
    stack.enter_context(decoder_graph_pool(model))
  enable_optimizations(old,decoder=False,position_dot=False);stack.callback(disable_optimizations,old)
  stack.enter_context(encoder_transform(new,mode='int8x3',qkv=True,silu=True,position=True,fold_bn=True,norm=True,
                      attention=True,norm_quant=True,position_dot=args.position_dot))
  enable_pipeline_graphs(new);stack.callback(disable_pipeline_graphs,new)
  if not args.reference_only:
   from .sprint3 import variant
   stack.enter_context(variant(new,args.decoder_variant))
  def decode(model,e,l):
   result=model.decoding.rnnt_decoder_predictions_tensor(e,l,return_hypotheses=True)
   h=(result[0] if isinstance(result,tuple) else result)[0]
   state=model.decoding.decoding.decoding_computer.state
   bh=state.batched_hyps;n=int(bh.current_lengths[0])
   return dict(text=h.text,tokens=bh.transcript[0,:n].tolist(),timestamps=bh.timestamps[0,:n].tolist(),score=float(bh.scores[0]))
  for batch_number,indices in enumerate(ev.make_batches(records,8)):
   if args.limit_batches and batch_number>=args.limit_batches:break
   batch=[records[i] for i in indices];a,l=ev.audio_batch(batch);a,l=a.cuda(),l.cuda()
   e0,el0=old(input_signal=a,input_signal_length=l)
   if args.position_dot:
    e1,el1=new(input_signal=a,input_signal_length=l)
    torch.testing.assert_close((e1,el1),(e0,el0),atol=3e-6,rtol=3e-5)
    errors.append(dict(max_abs=float((e1-e0).abs().max()),relative_l2=float((e1-e0).norm()/e0.norm())))
   else:e1,el1=e0,el0
   for i,rec in enumerate(batch):
    baseline=decode(old,e0[i:i+1],el0[i:i+1])
    candidate=decode(new,e1[i:i+1],el1[i:i+1])
    rows.append({**rec,'baseline':baseline,'candidate':candidate,'scores':{n:ev.score(rec['text'],v['text']) for n,v in [('baseline',baseline),('candidate',candidate)]}})
   summary={key:sum(r['baseline'][key]==r['candidate'][key] for r in rows) for key in ['text','tokens','timestamps']}
   summary.update(utterances=len(rows),audio_seconds=sum(r['duration'] for r in rows),encoder_errors=errors,
                  max_score_difference=max(abs(r['baseline']['score']-r['candidate']['score']) for r in rows))
   totals={}
   for name in ['baseline','candidate']:
    totals[name]={}
    for r in rows:
     s=r['scores'][name]
     if s['scored']:
      t=totals[name].setdefault(r['source'],dict(words=0,edits=0));t['words']+=len(s['ref_norm'].split());t['edits']+=s['S']+s['D']+s['I']
   summary['per_set']=totals
   save(out,'quality.json',dict(summary=summary,records=rows))
   print('quality',len(rows),summary['text'],summary['tokens'],summary['timestamps'],flush=True)
  assert summary['text']==summary['tokens']==summary['timestamps']==len(rows),summary
if __name__=='__main__':main()
