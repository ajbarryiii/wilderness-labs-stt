"""Validate experimental encoders with text/token/timestamp checks."""
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
 p.add_argument('--reference',choices=['packed','expanded'],default='packed')
 p.add_argument('--candidate',choices=['expanded','ptx_rowfusion','ptx_compiler','ptx_geometry','ptx_row_best'],default='expanded')
 p.add_argument('--compile-workers',type=int,default=0,help='precompile predicted batch-one CUDA shapes on CPU')
 args=p.parse_args();paths.require_mount()
 if args.batch_size<1 or args.utterances_per_set<1 or args.compile_workers<0:p.error('positive batch size/count; nonnegative compiler workers required')
 if args.compile_workers and (args.batch_size!=1 or args.candidate=='expanded'):p.error('CPU precompile is for batch-one CUDA validation')
 out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-quality4';out.mkdir();print('RESULTS',out,flush=True)
 records=records_for_sets(args.utterances_per_set);rows=[];errors=[]
 with paths.gpu_lock('expanded encoder quality validation'),torch.inference_mode(),energy_meter() as meter,contextlib.ExitStack() as stack:
  from .sprint4 import require_power_cap
  meter._guard();require_power_cap(meter);save(out,'metadata.json',metadata(args));stack.enter_context(ev.strict_fp32('cuda'))
  old=load_packed(args.export);new=load_packed(args.export)
  for model in [old,new]:
   stack.enter_context(ev.inference_settings(model));model.decoding.decoding.decoding_computer.force_cuda_graphs_mode('full_graph')
  enable_optimizations(old,encoder_storage=args.reference);stack.callback(disable_optimizations,old)
  if args.candidate=='expanded':
   enable_optimizations(new,encoder_storage='expanded');stack.callback(disable_optimizations,new)
  else:
   from .cuda_pipeline import experimental_pipeline
   provider=stack.enter_context(experimental_pipeline(new,args.candidate))
   if args.compile_workers:
    import concurrent.futures
    import soundfile as sf
    from .runtime import PackedLinear
    activations={id(ff.linear1) for layer in new.encoder.layers for ff in [layer.feed_forward1,layer.feed_forward2]}
    specs={(mod.out_features,mod.in_features,mod.bias is not None,'silu' if id(mod) in activations else '')
           for mod in new.encoder.modules() if isinstance(mod,PackedLinear) and mod.in_features%128==0}
    specs.add((3072,1024,new.encoder.layers[0].self_attn.linear_q.bias is not None,''))
    hop=round(new.cfg.preprocessor.window_stride*new.cfg.preprocessor.sample_rate)
    factor=new.encoder.subsampling_factor
    frames={((sf.info(rec['audio_filepath']).frames//hop+1)+factor-1)//factor for rec in records}
    tasks={(m,*spec) for m in frames for spec in specs}
    tasks.update((2*m-1,1024,1024,new.encoder.layers[0].self_attn.linear_pos.bias is not None,'') for m in frames)
    print('CPU precompile',len(tasks),'kernels;',args.compile_workers,'workers; no matmul launches',flush=True)
    def precompile(task):
     torch.cuda.set_device(0)
     provider.prewarm(*task)
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.compile_workers) as executor:
     futures=[executor.submit(precompile,task) for task in sorted(tasks)]
     for i,future in enumerate(concurrent.futures.as_completed(futures),1):
      future.result()
      if i%64==0:print('CPU precompiled',i,'/',len(tasks),flush=True)
    save(out,'precompile.json',dict(kernels=len(provider.kernels),seconds=time.perf_counter()-start,
                                  workers=args.compile_workers,predicted_frames=sorted(frames)))
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
