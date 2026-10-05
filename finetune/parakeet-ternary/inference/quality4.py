"""Validate expanded encoder against the third-pass packed runtime."""
import argparse,contextlib,time
from pathlib import Path
import torch
import paths
from .benchmark import DEFAULT_EXPORT,records_for_sets,energy_meter,metadata,save
from .runtime import load_packed
from .optimized import enable_optimizations,disable_optimizations


def main():
 import evaluate as ev
 from nemo.utils import logging
 logging.setLevel(logging.ERROR)
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export',type=Path,default=DEFAULT_EXPORT)
 p.add_argument('--utterances-per-set',type=int,default=256);p.add_argument('--batch-size',type=int,default=8)
 p.add_argument('--limit-batches',type=int,default=0)
 args=p.parse_args();paths.require_mount()
 if args.batch_size<1 or args.utterances_per_set<1:p.error('positive batch size and utterance count required')
 out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-quality4';out.mkdir();print('RESULTS',out,flush=True)
 records=records_for_sets(args.utterances_per_set);rows=[];errors=[]
 with paths.gpu_lock('expanded encoder quality validation'),torch.inference_mode(),energy_meter() as meter,contextlib.ExitStack() as stack:
  from .sprint4 import require_power_cap
  meter._guard();require_power_cap(meter);save(out,'metadata.json',metadata(args));stack.enter_context(ev.strict_fp32('cuda'))
  old=load_packed(args.export);new=load_packed(args.export)
  for model in [old,new]:
   stack.enter_context(ev.inference_settings(model));model.decoding.decoding.decoding_computer.force_cuda_graphs_mode('full_graph')
  enable_optimizations(old);stack.callback(disable_optimizations,old)
  enable_optimizations(new,encoder_storage='expanded');stack.callback(disable_optimizations,new)
  def decode(model,e,l):
   result=model.decoding.rnnt_decoder_predictions_tensor(e,l,return_hypotheses=True)
   h=(result[0] if isinstance(result,tuple) else result)[0]
   state=model.decoding.decoding.decoding_computer.state
   bh=state.batched_hyps;n=int(bh.current_lengths[0])
   return dict(text=h.text,tokens=bh.transcript[0,:n].tolist(),timestamps=bh.timestamps[0,:n].tolist(),score=float(bh.scores[0]))
  for batch_number,indices in enumerate(ev.make_batches(records,args.batch_size)):
   if args.limit_batches and batch_number>=args.limit_batches:break
   require_power_cap(meter)
   batch=[records[i] for i in indices];a,l=ev.audio_batch(batch);a,l=a.cuda(),l.cuda()
   e0,el0=old(input_signal=a,input_signal_length=l)
   e1,el1=new(input_signal=a,input_signal_length=l)
   torch.testing.assert_close((e1,el1),(e0,el0),atol=3e-6,rtol=3e-5)
   errors.append(dict(max_abs=float((e1-e0).abs().max()),relative_l2=float((e1-e0).norm()/e0.norm())))
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
