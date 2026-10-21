"""CUDA matmul experiments and cache ablation under the 400 W GPU cap.

All artifacts are written to the mounted project disk. Microbenchmarks use
representative shapes; full-encoder tests use distinct real utterances sampled
from every existing test corpus. These are warm measurements, not cold starts.
"""
import argparse
import contextlib
import json
import statistics
import time
from pathlib import Path

import torch
import triton
import paths

from .benchmark import DEFAULT_EXPORT, capture, energy_meter, metadata, save
from .cuda_matmul import CONFIGS, CudaProjection, dot, kernel
from .encoder4 import expanded_configuration
from .fused_kernels import _int8_dot
from .kernels import _finish
from .sprint4 import require_power_cap


def sample_records():
    """24 different utterances: eight corpora x three duration targets."""
    records = []
    for source in paths.TEST_SETS:
        candidates = [json.loads(line) for line in
                      (paths.MANIFESTS / f'test_{source}.jsonl').read_text().splitlines()]
        used = set()
        for duration in (3, 10, 30):
            rec = min((r for r in candidates if r['id'] not in used and r['duration'] > 0),
                      key=lambda r: abs(r['duration'] - duration))
            used.add(rec['id'])
            records.append({**rec, 'source': source, 'target_seconds': duration})
    return records


def micro(args, out, meter):
    torch.manual_seed(731)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for name in args.candidates:
        if name != 'cublas':
            kernel(name)
    for m in args.rows:
        for n, k in [(1024, 1024), (3072, 1024), (4096, 1024), (2048, 1024), (1024, 4096)]:
            require_power_cap(meter)
            q = torch.randint(-127, 128, (3, m, k), dtype=torch.int8, device='cuda')
            qs = torch.rand(3, m, device='cuda') * .01
            w = torch.randint(-1, 2, (n, k), dtype=torch.int8, device='cuda').T
            ws = torch.rand(n, device='cuda') * .03
            bias = torch.randn(n, device='cuda') * .01
            reference = sum((q[i].float() @ w.float()) * qs[i, :, None] for i in range(3))
            reference = reference * ws + bias
            functions = {}
            outputs = {}
            bm, bn, bk, split, warps, stages = expanded_configuration(m, n, k)
            y = torch.empty(m, n, device='cuda')
            p = torch.empty(split, m, n, device='cuda')
            def triton_fn(y=y, p=p):
                _int8_dot[(triton.cdiv(m,bm),triton.cdiv(n,bn),split)](
                    q,qs,w,ws,bias,y,p,m,n,k,True,bm,bn,bk,split,3,
                    DENSE=True,COLUMN=True,num_warps=warps,num_stages=stages)
                if split > 1:
                    _finish[(triton.cdiv(m*n,256),)](p,ws,bias,y,m,n,m,m*n,n,1,True,split,256)
            functions['triton'] = triton_fn
            outputs['triton'] = y
            for name in args.candidates:
                split_cuda = CONFIGS[name][5]
                y = torch.empty(m, n, device='cuda')
                p = torch.empty(split_cuda, m, n, device='cuda')
                def cuda_fn(name=name, y=y, p=p, split_cuda=split_cuda):
                    dot(q,qs,w,ws,bias,y,p,name=name)
                    if split_cuda > 1:
                        _finish[(triton.cdiv(m*n,256),)](p,ws,bias,y,m,n,m,m*n,n,1,True,split_cuda,256)
                functions[name] = cuda_fn
                outputs[name] = y
            row = {'m':m,'n':n,'k':k,'scope':'prequantized residual INT8 dot + epilogue; excludes activation quantization'}
            measurements = []
            for name, fn in functions.items():
                fn()
                torch.cuda.synchronize()
                actual = outputs[name]
                torch.testing.assert_close(actual,reference,atol=2e-6,rtol=2e-5)
                measurements.append({'name':name,'max_abs':float((actual-reference).abs().max()),
                    'us':statistics.median(triton.testing.do_bench_cudagraph(fn,rep=25)*1000 for _ in range(3))})
            row['measurements'] = measurements
            results.append(row)
            print(json.dumps(row),flush=True)
            save(out,'micro.json',results)
            del functions,outputs


@contextlib.contextmanager
def encoder_provider(model, name, out=None):
    from .transforms import encoder_transform
    from .encoder4 import ExpandedProjection
    from .layer_fusions4 import layer_fusions
    if name=='triton':
        provider=ExpandedProjection('triton_col_tuned')
    elif name in ('ptx_cuda','ptx_compiler','ptx_geometry','ptx_weights4','ptx_weights2','ptx_rowfusion','ptx_rowwarp1','ptx_rowwarp2','ptx_rowwarp8','ptx_row_native','ptx_row_best'):
        from .ptx_cuda import PtxCudaProjection
        tuning={'ptx_compiler':'compiler','ptx_geometry':'geometry','ptx_row_native':'native','ptx_row_best':'native'}.get(name,'baseline')
        bits={'ptx_weights4':4,'ptx_weights2':2}.get(name,8)
        row_warps={'ptx_rowwarp1':1,'ptx_rowwarp2':2,'ptx_rowwarp8':8,'ptx_row_best':'adaptive'}.get(name,4)
        provider=PtxCudaProjection(out/(name+'-kernels') if out is not None else None,tuning=tuning,weight_bits=bits,row_warps=row_warps)
    else:
        provider=CudaProjection(name)
    with encoder_transform(model,mode='int8x3',qkv=True,silu=True,position=True,fold_bn=True,
                           norm=True,attention=True,norm_quant=True,position_dot=True,projection=provider), \
         layer_fusions(model,provider,residual=not name.startswith('ptx_row')),contextlib.ExitStack() as stack:
        if name.startswith('ptx_row'):
            from .row_fusion import row_fusions
            stack.enter_context(row_fusions(model,provider))
        yield provider


def multi_encoder(args, out, meter):
    import evaluate as ev
    from .runtime import load_packed
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    records=sample_records()
    if args.limit:
        records=records[:args.limit]
    names=['triton',*args.candidates]
    results=[]
    with contextlib.ExitStack() as stack:
        stack.enter_context(ev.strict_fp32('cuda'))
        models={name:load_packed(args.export) for name in names}
        for name,model in models.items():
            stack.enter_context(ev.inference_settings(model))
            stack.enter_context(encoder_provider(model,name,out))
        # Distinct models keep folded convolution weights and captured raw
        # arguments owned for the full graph lifetime. Never restore a model
        # while its graph is live.
        for index,rec in enumerate(records):
            audio,length=ev.audio_batch([rec]);audio,length=audio.cuda(),length.cuda()
            features,feature_length=models['triton'].preprocessor(input_signal=audio,length=length)
            graphs={};outputs={}
            try:
                for name in names:
                    model=models[name]
                    graph,output=capture(lambda:model.encoder(audio_signal=features,length=feature_length))
                    graphs[name]=graph;outputs[name]=tuple(t.clone() for t in output)
                ref=outputs['triton']
                checks={}
                for name in names:
                    torch.testing.assert_close(outputs[name],ref,atol=3e-6,rtol=3e-5)
                    checks[name]={'max_abs':float((outputs[name][0]-ref[0]).abs().max()),
                                  'bitwise_match':all(torch.equal(a,b) for a,b in zip(outputs[name],ref))}
                windows={name:[] for name in names}
                for repeat in range(args.repeats):
                    order=names if (repeat+index)%2==0 else names[::-1]
                    for name in order:
                        require_power_cap(meter)
                        started=time.perf_counter()
                        while time.perf_counter()-started<.3:graphs[name].replay();torch.cuda.synchronize()
                        w=meter.measure(graphs[name].replay,min_seconds=args.seconds)
                        w['median_ms']=statistics.median(w['iteration_seconds'])*1000
                        windows[name].append(w)
                row={'id':rec['id'],'source':rec['source'],'audio_seconds':rec['duration'],
                     'target_seconds':rec['target_seconds'],'checks':checks,
                     'summary':{name:{key:statistics.median(w[key] for w in ws)
                                 for key in ['median_ms','joules_per_iteration','average_watts']}
                                 for name,ws in windows.items()},'windows':windows,
                     'scope':'GPU features -> encoder output; raw CUDA-graph replay; excludes frontend/decoder/first capture'}
                results.append(row);print(json.dumps({k:v for k,v in row.items() if k!='windows'}),flush=True)
                save(out,'encoder.json',results)
            finally:
                graphs.clear();outputs.clear()


def multi_pipeline(args,out,meter):
    """Paired warm CPU waveform -> text; cache one utterance shape at a time."""
    import evaluate as ev
    from .runtime import load_packed
    from .cuda_pipeline import experimental_pipeline
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    records=sample_records()
    if args.limit:records=records[:args.limit]
    names=['triton',*args.candidates];results=[]
    with contextlib.ExitStack() as stack:
        stack.enter_context(ev.strict_fp32('cuda'))
        models={name:load_packed(args.export) for name in names}
        for name,model in models.items():
            stack.enter_context(ev.inference_settings(model))
            model.decoding.decoding.decoding_computer.force_cuda_graphs_mode('full_graph')
            stack.enter_context(experimental_pipeline(model,name,out))
        for index,rec in enumerate(records):
            audio,length=ev.audio_batch([rec])
            for model in models.values():model.forward.entries.clear()
            def recognize(name):
                model=models[name]
                e,el=model(input_signal=audio.cuda(),input_signal_length=length.cuda())
                result=model.decoding.rnnt_decoder_predictions_tensor(e,el,return_hypotheses=False)
                if isinstance(result,tuple):result=result[0]
                return result[0].text if hasattr(result[0],'text') else str(result[0])
            checks={}
            for name in names:
                text=recognize(name)
                state=models[name].decoding.decoding.decoding_computer.state
                bh=state.batched_hyps;n=int(bh.current_lengths[0])
                checks[name]={'text':text,'tokens':bh.transcript[0,:n].tolist(),'timestamps':bh.timestamps[0,:n].tolist()}
            for name in names:assert checks[name]==checks['triton'],(rec['id'],name,checks)
            windows={name:[] for name in names}
            for repeat in range(args.repeats):
                for name in (names if (repeat+index)%2==0 else names[::-1]):
                    require_power_cap(meter)
                    started=time.perf_counter()
                    while time.perf_counter()-started<.3:recognize(name);torch.cuda.synchronize()
                    w=meter.measure(lambda:recognize(name),min_seconds=args.seconds)
                    w['median_ms']=statistics.median(w['iteration_seconds'])*1000
                    windows[name].append(w)
            row={'id':rec['id'],'source':rec['source'],'audio_seconds':rec['duration'],'target_seconds':rec['target_seconds'],
                 'checks':checks,'summary':{name:{key:statistics.median(w[key] for w in ws)
                         for key in ['median_ms','joules_per_iteration','average_watts']} for name,ws in windows.items()},
                 'windows':windows,'scope':'CPU waveform -> CPU text, transfers/frontend/encoder/optimized decoder; one warmed shape cached per model; no disk I/O/compile/capture; two resident models, not deployment VRAM'}
            results.append(row);save(out,'pipeline.json',results)
            print(json.dumps({k:v for k,v in row.items() if k not in ('windows','checks')}),flush=True)


def decoder_cache(args, out, meter):
    import evaluate as ev
    from .runtime import load_packed
    from .optimized import enable_optimizations, disable_optimizations
    from .decoder_kernels import decoder_transform, control_transform
    from .graph_memory import decoder_graph_pool
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    model=load_packed(args.export)
    records=sample_records()
    if args.limit:records=records[:args.limit]
    comp=model.decoding.decoding.decoding_computer
    results=[]
    with ev.inference_settings(model),ev.strict_fp32('cuda'):
        enable_optimizations(model,decoder=False,encoder_storage='expanded')
        try:
            fixtures=[]
            for rec in records:
                a,l=ev.audio_batch([rec]);e,el=model(input_signal=a.cuda(),input_signal_length=l.cuda())
                fixtures.append((rec,e.clone(),el.clone()))
        finally:disable_optimizations(model)
        rnn=model.decoder.prediction['dec_rnn'].lstm
        table_bytes=model.decoder.prediction['embed'].weight.shape[0]*rnn.weight_ih_l0.shape[0]*4
        for index,(rec,e,el) in enumerate(fixtures):
            names=['uncached','cached'] if index%2==0 else ['cached','uncached']
            measurements={};hyps={}
            for name in names:
                with decoder_graph_pool(model),decoder_transform(model,joint=True,lstm=True,precompute=name=='cached',block=1), \
                     control_transform(model,storage=True):
                    comp.force_cuda_graphs_mode('full_graph')
                    def decode():
                        out=model.decoding.rnnt_decoder_predictions_tensor(e,el,return_hypotheses=True)
                        h=(out[0] if isinstance(out,tuple) else out)[0]
                        return h.text
                    hyp=decode();hyps[name]=hyp
                    torch.cuda.synchronize()
                    state=comp.state;bh=state.batched_hyps;n=int(bh.current_lengths[0])
                    tokens=bh.transcript[0,:n].tolist();timestamps=bh.timestamps[0,:n].tolist()
                    windows=[]
                    for repeat in range(args.repeats):
                        require_power_cap(meter)
                        started=time.perf_counter()
                        while time.perf_counter()-started<.3:decode()
                        w=meter.measure(decode,min_seconds=args.seconds)
                        w['median_ms']=statistics.median(w['iteration_seconds'])*1000
                        windows.append(w)
                    measurements[name]={'tokens':tokens,'timestamps':timestamps,
                        'summary':{key:statistics.median(w[key] for w in windows)
                                  for key in ['median_ms','joules_per_iteration','average_watts']},'windows':windows}
            assert hyps['cached']==hyps['uncached'],rec['id']
            assert measurements['cached']['tokens']==measurements['uncached']['tokens']
            assert measurements['cached']['timestamps']==measurements['uncached']['timestamps']
            row={'id':rec['id'],'source':rec['source'],'audio_seconds':rec['duration'],
                 'target_seconds':rec['target_seconds'],'table_bytes':table_bytes,
                 'text_tokens_timestamps_match':True,'measurements':measurements,
                 'scope':'identical GPU encoder tensors -> CPU text; cache creation excluded; existing optimized decoder with cache toggled'}
            results.append(row);save(out,'decoder-cache.json',results)
            print(json.dumps({'id':rec['id'],'source':rec['source'],'audio_seconds':rec['duration'],
                 'summary':{name:v['summary'] for name,v in measurements.items()}}),flush=True)


def decoder_profile(args,out,meter):
    import collections
    import evaluate as ev
    from .runtime import load_packed
    from .optimized import enable_optimizations,disable_optimizations
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    rec=next(r for r in sample_records() if r['source']=='librispeech_clean' and r['target_seconds']==10)
    model=load_packed(args.export)
    with ev.inference_settings(model),ev.strict_fp32('cuda'):
        comp=model.decoding.decoding.decoding_computer
        comp.force_cuda_graphs_mode('full_graph')
        enable_optimizations(model,encoder_storage='expanded')
        try:
            a,l=ev.audio_batch([rec]);e,el=model(input_signal=a.cuda(),input_signal_length=l.cuda())
            def decode():return model.decoding.rnnt_decoder_predictions_tensor(e,el,return_hypotheses=False)
            for _ in range(3):decode()
            torch.cuda.synchronize()
            token_count=int(comp.state.batched_hyps.current_lengths[0])
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(3):decode()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(out/'decoder-trace.json'))
            groups=collections.defaultdict(lambda:[0,0])
            for event in json.loads((out/'decoder-trace.json').read_text())['traceEvents']:
                if event.get('cat')=='kernel':
                    groups[event['name']][0]+=event.get('dur',0)/3
                    groups[event['name']][1]+=1/3
            rows=[{'name':name,'us':v[0],'count':round(v[1])} for name,v in groups.items()]
            rows.sort(key=lambda r:-r['us'])
            total=sum(r['us'] for r in rows)
            predictor=sum(r['us'] for r in rows if r['name'].startswith(('_lstm_step','_linear')))
            lstm_calls=sum(r['count'] for r in rows if r['name'].startswith('_lstm_step'))
            # Conditional CUDA graphs can expose only their static node
            # definitions in this profiler. Two LSTM nodes for a many-token
            # transcript do not account for their repeated execution. Keep the
            # trace for diagnosis, but never use it as prefix-cache headroom.
            report={'record':rec,'token_count':token_count,
                    'traced_kernel_us':total,'traced_predictor_kernel_us':predictor,
                    'traced_lstm_calls_per_decode':lstm_calls,'kernels':rows,
                    'prefix_cache_headroom':None,
                    'scope':'warm cached decoder trace; conditional-graph repetitions may be missing; cannot estimate latency shares or prefix-cache gains'}
            save(out,'decoder-profile.json',report)
            print(json.dumps({k:v for k,v in report.items() if k!='kernels'}),flush=True)
        finally:disable_optimizations(model)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['micro','encoder','pipeline','decoder-cache','decoder-profile'])
    p.add_argument('--export',type=Path,default=DEFAULT_EXPORT)
    p.add_argument('--candidates',nargs='+')
    p.add_argument('--rows',type=int,nargs='+',default=[38,126,376])
    p.add_argument('--seconds',type=float,default=1)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args();paths.require_mount()
    if args.seconds <= 0 or args.repeats < 1 or args.limit < 0 or any(m < 1 for m in args.rows):
        p.error('positive timing windows, repeats and rows required; limit must be nonnegative')
    if args.candidates is None:
        args.candidates=['auto','cublas'] if args.mode=='encoder' else list(CONFIGS)
    if any(name not in CONFIGS and name not in ('auto','ptx_cuda','ptx_compiler','ptx_geometry','ptx_weights4','ptx_weights2','ptx_rowfusion','ptx_rowwarp1','ptx_rowwarp2','ptx_rowwarp8','ptx_row_native','ptx_row_best') for name in args.candidates):
        p.error('unknown CUDA candidate')
    if args.mode=='micro' and any(name=='auto' or name.startswith('ptx_') for name in args.candidates):
        p.error('micro sweep needs explicit CUDA candidates; use inference.ptx_cuda for the PTX proof')
    out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-cuda-{args.mode}'
    out.mkdir();print('RESULTS',out,flush=True)
    with paths.gpu_lock('CUDA encoder and token-cache experiments'),torch.inference_mode(),energy_meter() as meter:
        meter._guard();require_power_cap(meter)
        save(out,'metadata.json',metadata(args));save(out,'gpu.json',meter.metadata())
        source_dir=out/'source';source_dir.mkdir()
        for source in Path(__file__).parent.glob('*.py'):
            (source_dir/source.name).write_bytes(source.read_bytes())
        {'micro':micro,'encoder':multi_encoder,'pipeline':multi_pipeline,'decoder-cache':decoder_cache,
         'decoder-profile':decoder_profile}[args.mode](args,out,meter)


if __name__=='__main__':main()
