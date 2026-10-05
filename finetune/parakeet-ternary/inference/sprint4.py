"""Encoder-only fourth-pass experiments under the existing 400 W GPU limit."""
import argparse,contextlib,json,statistics,time
from pathlib import Path
import torch
import paths
from .benchmark import DEFAULT_EXPORT,capture,energy_meter,metadata,save
from .runtime import load_packed
from .transforms import encoder_transform
from .encoder4 import ExpandedProjection


def require_power_cap(meter):
    limit=meter._nvml.nvmlDeviceGetPowerManagementLimit(meter._handle)/1000
    if limit>400:raise RuntimeError(f'GPU power limit is {limit} W; refusing benchmark above 400 W')
    return limit


@contextlib.contextmanager
def variant(model,name):
    pieces=name.split('+');name=pieces[0];fusions=set(pieces[1:])
    provider=None if name=='baseline' else ExpandedProjection(name)
    with encoder_transform(model,mode='int8x3',qkv=True,silu=True,position=True,fold_bn=True,norm=True,
                           attention=True,norm_quant=True,position_dot=True,projection=provider),contextlib.ExitStack() as stack:
        if fusions:
            from .layer_fusions4 import layer_fusions
            stack.enter_context(layer_fusions(model,provider,convolution='conv' in fusions,residual='residual' in fusions))
        yield provider


def main():
    import evaluate as ev
    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,default=DEFAULT_EXPORT)
    p.add_argument('--variants',nargs='+',default=['baseline','triton','torch_row','torch_col'])
    p.add_argument('--durations',type=float,nargs='+',default=[3,10,30]);p.add_argument('--seconds',type=float,default=3)
    args=p.parse_args();paths.require_mount()
    out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-sprint4';out.mkdir()
    print('RESULTS',out,flush=True);rows=[]
    with paths.gpu_lock('fourth encoder optimization screen'),torch.inference_mode(),energy_meter() as meter:
        meter._guard();require_power_cap(meter);save(out,'metadata.json',metadata(args));save(out,'gpu.json',meter.metadata())
        model=load_packed(args.export)
        records=[json.loads(l) for l in (paths.MANIFESTS/'test_librispeech_clean.jsonl').read_text().splitlines()]
        with ev.inference_settings(model),ev.strict_fp32('cuda'):
            for duration in args.durations:
                rec=min(records,key=lambda r:abs(r['duration']-duration));a,l=ev.audio_batch([rec]);a,l=a.cuda(),l.cuda()
                f,fl=model.preprocessor(input_signal=a,length=l)
                with variant(model,'baseline'):
                    graph,(ref,ref_l)=capture(lambda:model.encoder(audio_signal=f,length=fl))
                    graph.replay();ref=ref.clone();ref_l=ref_l.clone();del graph
                for name in args.variants:
                    require_power_cap(meter)
                    with variant(model,name) as provider:
                        graph,(y,yl)=capture(lambda:model.encoder(audio_signal=f,length=fl))
                        graph.replay();torch.cuda.synchronize()
                        err={'max_abs':float((y-ref).abs().max()),'relative_l2':float((y-ref).norm()/ref.norm())}
                        try:
                            torch.testing.assert_close((y,yl),(ref,ref_l),atol=3e-6,rtol=3e-5)
                            err['numerical_pass']=True
                        except AssertionError as exc:
                            err.update(numerical_pass=False,numerical_error=str(exc))
                        start=time.perf_counter()
                        while time.perf_counter()-start<1:graph.replay();torch.cuda.synchronize()
                        w=meter.measure(graph.replay,min_seconds=args.seconds)
                        save(out,f'{duration:g}-{name}.json',w)
                        row={'variant':name,'duration':rec['duration'],'ms':statistics.median(w['iteration_seconds'])*1000,
                             'J':w['joules_per_iteration'],'watts':w['average_watts'],**err,
                             'expanded_bytes':provider.bytes if provider else 0,'live_bytes':torch.cuda.memory_allocated(),
                             'power_limit_w':require_power_cap(meter)}
                        rows.append(row);print(json.dumps(row),flush=True);save(out,'screen.json',rows)
                        del graph,y,yl
                    del provider
                del ref,ref_l


if __name__=='__main__':main()
