"""Independent paired transcription/gating evaluation; never imports candidate training code."""
import argparse
import ctypes
import json
import time
from pathlib import Path
import numpy as np
import torch
from core import ROOT, SR, decode, digest, features, load_model, save, score_pairs
from data import audio, load
from gating import Gate


class Energy:
    """Optional host CPU package counters. Null is not a zero-energy measurement."""
    def __init__(self):
        self.counters=[]
        for p in sorted(Path('/sys/class/powercap').glob('intel-rapl:[0-9]*/energy_uj')):
            # Only package domains, not overlapping child domains.
            if p.parent.name.count(':') != 1:
                continue
            try:
                self.counters.append((p,int((p.parent/'max_energy_range_uj').read_text())))
            except (OSError,ValueError):
                pass

    def read(self):
        try:
            return [int(p.read_text()) for p,_ in self.counters] or None
        except (OSError,ValueError):
            return None

    def delta(self,a,b):
        if a is None or b is None:
            return None
        return sum((y-x)%limit for x,y,(_,limit) in zip(a,b,self.counters))/1e6


def scenes(manifest,split,limit):
    speech=[r for r in manifest['rows'] if r['split']==split and r['label']]
    noise=[r for r in manifest['rows'] if r['split']==split and not r['label']]
    if limit:
        # Deterministic spread across the complete speaker/source list, not first speaker only.
        speech=[speech[i] for i in np.linspace(0,len(speech)-1,min(limit,len(speech)),dtype=int)]
        noise=[noise[i] for i in np.linspace(0,len(noise)-1,min(limit,len(noise)),dtype=int)]
    result=[]
    for i,row in enumerate(speech):
        x=audio(row)
        n=audio(noise[i%len(noise)])
        n=np.resize(n,len(x)+4*SR).astype(np.float32)
        # Paired clean and mixed-noise scenes, same speaker/source split.
        for snr in (None,10):
            y=x.copy()
            if snr is not None:
                interference=n[2*SR:2*SR+len(x)]
                gain=np.sqrt(np.mean(x*x)+1e-12)/(np.sqrt(np.mean(interference**2)+1e-12)*10**(snr/20))
                y+=interference*gain
            recording=np.concatenate([n[:2*SR],y,n[-2*SR:]])
            result.append(dict(id=f'speech-{i}-snr-{snr}',audio=recording,text=row['text'],
                condition='clean' if snr is None else 'mixed_10db',speech_start=2*SR,speech_end=2*SR+len(x)))
    for i,row in enumerate(noise):
        result.append(dict(id=f'noise-{i}',audio=audio(row),text='',condition='noise_only'))
    return result


def recognize(model,pcm,device):
    parts=[]
    # Fixed bounded chunking shared by both arms; no oracle utterance boundaries.
    for start in range(0,len(pcm),20*SR):
        with torch.inference_mode():
            x=torch.from_numpy(features(pcm[start:start+20*SR])[None]).to(device)
            logits=model(x)
            parts.append(decode(logits[0].argmax(-1).tolist()))
    return ' '.join(p for p in parts if p)


def evaluate(args):
    torch.set_num_threads(1)
    manifest,_=load(args.cache,verify_audio=True)
    if args.backend == 'custom':
        model=load_model(args.candidate,args.device)
        run_asr=lambda x:recognize(model,x,args.device)
    else:
        # Optional pretrained engineering reference, outside strict ternary claims.
        from pretrained import Recognizer
        model=Recognizer(args.candidate,args.device)
        run_asr=model.transcribe
    cases=scenes(manifest,args.split,args.limit)
    if not cases:
        raise ValueError('Empty evaluation')
    gates={name:Gate(ROOT/'custom/autoresearch/results/20260909T224149Z/trials'/name) for name in ['008','010']}
    # Warm up the same runtime; loading and compilation do not enter steady-state timing.
    run_asr(cases[0]['audio'][:SR])
    for gate in gates.values():
        gate.intervals(cases[0]['audio'][:SR])
    energy=Energy()
    results=[]
    arms=['always_on','gate_008','gate_010']
    for repeat in range(args.repeats):
        for i,case in enumerate(cases):
            # Rotate execution order to reduce consistent thermal/order bias.
            order=arms[(i+repeat)%3:]+arms[:(i+repeat)%3]
            for arm in order:
                pcm=case['audio']
                before=energy.read()
                started=time.perf_counter()
                if arm=='always_on':
                    intervals=[(0,len(pcm))]
                else:
                    intervals,_=gates[arm[-3:]].intervals(pcm)
                gate_seconds=time.perf_counter()-started
                text=' '.join(filter(None,(run_asr(pcm[a:b]) for a,b in intervals)))
                if args.device=='cuda':
                    torch.cuda.synchronize()
                elapsed=time.perf_counter()-started
                after=energy.read()
                retained=sum(b-a for a,b in intervals)
                row=dict(id=case['id'],condition=case['condition'],repeat=repeat,arm=arm,
                    reference=case['text'],hypothesis=text,audio_seconds=len(pcm)/SR,
                    asr_audio_seconds=retained/SR,wall_seconds=elapsed,gate_seconds=gate_seconds,
                    cpu_package_joules=energy.delta(before,after),intervals_samples=intervals)
                if case['text']:
                    a,b=case['speech_start'],case['speech_end']
                    row['utterance_audio_omitted_seconds']=((b-a)-sum(max(0,min(b,v)-max(a,u)) for u,v in intervals))/SR
                results.append(row)
        print(f'Completed paired repeat {repeat+1}/{args.repeats}',flush=True)
    summary={}
    for arm in arms:
        rows=[r for r in results if r['arm']==arm]
        scores=score_pairs([(r['reference'],r['hypothesis']) for r in rows])
        total=sum(r['audio_seconds'] for r in rows)
        wall=sum(r['wall_seconds'] for r in rows)
        joules=[r['cpu_package_joules'] for r in rows]
        summary[arm]=dict(**scores,wall_seconds=wall,rtf=wall/total,
            asr_audio_fraction=sum(r['asr_audio_seconds'] for r in rows)/total,
            cpu_package_joules=sum(joules) if all(j is not None for j in joules) else None,
            iphone_joules=None,
            utterance_audio_omitted_seconds=sum(r.get('utterance_audio_omitted_seconds',0) for r in rows),
            by_condition={c:score_pairs([(r['reference'],r['hypothesis']) for r in rows if r['condition']==c])
                          for c in ('clean','mixed_10db')})
    baseline=summary['always_on']
    for arm in arms[1:]:
        summary[arm]['wer_delta']=summary[arm]['wer']-baseline['wer']
        summary[arm]['wall_time_reduction_fraction']=1-summary[arm]['wall_seconds']/baseline['wall_seconds']
    output=dict(schema=1,backend=args.backend,device=args.device,split=args.split,
        scenes=len(cases),repeats=args.repeats,summary=summary,predictions=results,
        manifest_sha256=digest(Path(args.cache)/'manifest.json'),
        limitations=['Composed continuous playback from real clips; no frame activity labels.',
          'Original full-clip gate thresholds applied to causal windows without recalibration.',
          'Development evidence, not untouched holdout or battlefield/medical validation.',
          'Elapsed time/audio skipped are cost proxies, not power measurements.',
          'CPU package energy, if available, excludes whole-device and GPU energy.',
          'Offline replay omits real-time idle, capture and device wake overhead.'])
    save(args.output,output)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',required=True)
    p.add_argument('--backend',choices=['custom','pretrained'],default='custom')
    p.add_argument('--cache',default=str(Path(__file__).with_name('cache')))
    p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['calibration','development'],default='development')
    p.add_argument('--limit',type=int,default=24)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=p.parse_args()
    if args.limit < 0 or args.repeats < 1:
        p.error('Nonnegative limit and positive repeats required')
    evaluate(args)
