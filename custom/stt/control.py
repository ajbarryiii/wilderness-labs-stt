"""Prepared, frozen, bounded STT autoresearch with supervisor-owned training/scoring."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from core import ROOT, digest, save, validate_recipe
from data import load

BASE=ROOT/'custom/stt'
RUNS=BASE/'runs'
CFG=json.loads((BASE/'experiment.json').read_text())
PYTHON=ROOT/'custom/autoresearch/runtime-python'


def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def fingerprint():
    files=[BASE/n for n in ['core.py','data.py','gating.py','train.py','evaluate.py','control.py']]
    files+=list(BASE.glob('*.json'))+[BASE/'program.md',BASE/'control']
    files=[p for p in files if p.name!='readiness.json']
    files += [BASE/'cache/lock.json', ROOT/'custom/vad/prepare.py',
              ROOT/'custom/autoresearch/runtime_lock.json', PYTHON]
    for trial in ['008','010']:
        folder=ROOT/'custom/autoresearch/results/20260909T224149Z/trials'/trial
        files += [folder/n for n in ['weights.npz','weights.2bit','model.json','metrics.json']]
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def verify(hashes):
    for name,h in hashes.items():
        if digest(ROOT/name)!=h:
            raise ValueError('Frozen dependency changed: '+name)


def bounded(argv, cwd, seconds, log, stdin=None):
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',
               WILDERNESS_STT_ROOT=str(ROOT))
    env.pop('PYTHONPATH',None)
    env.pop('PYTHONHOME',None)
    with open(log,'w') as handle:
        p=subprocess.Popen([str(x) for x in argv],cwd=cwd,stdout=handle,stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            text=True,start_new_session=True,env=env)
        try:
            p.communicate(stdin,timeout=seconds)
        except BaseException:
            try:
                os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
            raise
    if p.returncode:
        raise RuntimeError(f'Process exit {p.returncode}; see {log}')


def snapshot(run):
    hashes=fingerprint()
    for name in hashes:
        dest=run/'frozen'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
    save(run/'hashes.json',hashes)
    return hashes


def check_frozen(run):
    hashes=json.loads((run/'hashes.json').read_text())
    verify(hashes)
    for name,h in hashes.items():
        if digest(run/'frozen'/name)!=h:
            raise ValueError('Snapshot changed: '+name)


def valid_metrics(metrics):
    if metrics['backend']!='custom' or metrics['split']!='development':
        raise ValueError('Wrong scoring protocol')
    for arm in ['always_on','gate_008','gate_010']:
        s=metrics['summary'][arm]
        for key in ['wer','cer','wall_seconds','asr_audio_fraction']:
            if not isinstance(s[key],(float,int)) or not math.isfinite(s[key]) or s[key]<0:
                raise ValueError('Nonfinite/invalid metric')
        if s['reference_words']<=0:
            raise ValueError('Missing speech denominator')
    b=metrics['summary']['always_on']
    return b['wer']<CFG['max_baseline_wer'] and b['cer']<CFG['max_baseline_cer']


def useful(metrics):
    valid_metrics(metrics)
    b=metrics['summary']['always_on']
    # A decoding pilot can be informative before it meets the quality gate.
    # Empty/constant output has CER ~1 and cannot pass this diagnostic gate.
    predictions=[r['hypothesis'] for r in metrics.get('predictions',[]) if r['reference']]
    return (b['wer']<1.5 and b['cer']<0.95 and len(set(predictions))>=3
            and sum(bool(x.strip()) for x in predictions)>=len(predictions)*0.5)


def improves(candidate,best):
    if not useful(candidate):
        return False
    if best is None or not useful(best):
        return True
    if valid_metrics(candidate)!=valid_metrics(best):
        return valid_metrics(candidate)
    a,b=candidate['summary']['always_on'],best['summary']['always_on']
    # Any strict WER improvement counts; avoid the former VAD ranking blind spot.
    if a['wer'] < b['wer']:
        return True
    return a['wer']==b['wer'] and a['cer'] < b['cer']


def report(run):
    rows=json.loads((run/'results.json').read_text()) if (run/'results.json').exists() else []
    lines=['# STT autoresearch','', 'Development results; prototype scale, expanded FP32 inference.',
           'Cost proxies do not establish phone energy savings.','',
           '| Trial | Status | Always-on WER | Gate 008 WER | Gate 010 WER |',
           '|---|---|---:|---:|---:|']
    for r in rows:
        scores=r.get('metrics',{}).get('summary',{})
        values=[str(scores.get(a,{}).get('wer','')) for a in ['always_on','gate_008','gate_010']]
        lines.append('| '+' | '.join([r['trial'],r['status']]+values)+' |')
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n')


def trial(run,recipe,seconds,hypothesis):
    check_frozen(run)
    rows=json.loads((run/'results.json').read_text()) if (run/'results.json').exists() else []
    out=run/'trials'/f'{len(rows):03d}'
    out.mkdir(parents=True)
    save(out/'recipe.json',validate_recipe(recipe))
    row=dict(trial=out.name,status='error',hypothesis=hypothesis,recipe=recipe)
    started=time.monotonic()
    frozen=run/'frozen/custom/stt'
    with open(out/'gpu.csv','w') as telemetry:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw',
                 '--format=csv,noheader,nounits','-l','1'],stdout=telemetry,stderr=subprocess.DEVNULL)
        try:
            bounded([PYTHON,frozen/'train.py','--recipe',out/'recipe.json','--cache',BASE/'cache',
                '--output',out,'--train-seconds',seconds,'--seed',CFG['seed']],frozen,seconds+10,out/'train.log')
            check_frozen(run)
            t=json.loads((out/'training.json').read_text())
            if not t['training_cuda_verified'] or t['training_steps']<10 or t['optimizer_seconds']<seconds*0.5 or t['overfit_samples']:
                raise ValueError('Insufficient independent training work')
            if not math.isfinite(t['final_loss']) or t['master_weight_l1_change']<=0:
                raise ValueError('Invalid learning evidence')
            bounded([PYTHON,frozen/'evaluate.py','--candidate',out,'--cache',BASE/'cache',
                '--output',out/'metrics.json','--limit',CFG['development_speech_limit'],
                '--repeats',CFG['evaluation_repeats']],frozen,CFG['evaluation_seconds'],out/'evaluate.log')
            check_frozen(run)
            m=json.loads((out/'metrics.json').read_text())
            qualified=valid_metrics(m)
            diagnostic=useful(m)
            best=json.loads((run/'best.json').read_text()) if (run/'best.json').exists() else None
            keep=improves(m,best['metrics'] if best else None)
            row.update(metrics=m,status=('keep' if qualified else 'infeasible_best') if keep else ('discard' if diagnostic else 'unqualified'))
            if keep:
                save(run/'best.json',row)
        except Exception as e:
            row['error']=str(e)
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill(); monitor.wait()
    row['seconds']=time.monotonic()-started
    save(out/'result.json',row)
    rows.append(row)
    save(run/'results.json',rows)
    report(run)
    return row


def proposal(run):
    ws=run/'proposal'
    ws.mkdir(exist_ok=True)
    best=json.loads((run/'best.json').read_text())
    rows=json.loads((run/'results.json').read_text())
    history=[{k:r[k] for k in ('trial','status','hypothesis','recipe','error') if k in r} |
             {'scores':r.get('metrics',{}).get('summary',{})} for r in rows]
    prompt=(BASE/'program.md').read_text()+'\nRetained recipe:\n'+json.dumps(best['recipe'])
    prompt+='\nValidation bounds/function:\n'+(BASE/'core.py').read_text().split('def validate_recipe(r):')[1].split('class TernaryConv')[0]
    prompt+='\nTrial history:\n'+json.dumps(history)
    schema={'type':'object','additionalProperties':False,'required':['hypothesis','recipe'],
      'properties':{'hypothesis':{'type':'string'},'recipe':{'type':'object','additionalProperties':False,
      'required':list(best['recipe']), 'properties':{k:{'type':'integer' if type(v)is int else 'number'} for k,v in best['recipe'].items()}}}}
    save(ws/'schema.json',schema)
    output=ws/'proposal.json'
    output.unlink(missing_ok=True)
    bounded(['codex','exec','--sandbox','read-only','-c','approval_policy="never"',
        '--skip-git-repo-check','--cd',ws,'--output-schema',ws/'schema.json',
        '--output-last-message',output,'-'],ws,CFG['proposal_seconds'],run/f'proposal-{len(rows):03d}.log',prompt)
    obj=json.loads(output.read_text())
    if not isinstance(obj.get('hypothesis'),str) or not obj['hypothesis'].strip():
        raise ValueError('Missing hypothesis')
    validate_recipe(obj['recipe'])
    check_frozen(run)
    return obj


def preflight(seconds):
    load(verify_audio=True)
    run=RUNS/('preflight-'+timestamp())
    run.mkdir(parents=True)
    snapshot(run)
    recipe=json.loads((BASE/'recipe.json').read_text())
    result=trial(run,recipe,seconds,'Baseline qualification, outside research timer')
    ready=result['status'] in ('keep','infeasible_best') and seconds>=CFG['training_seconds']
    # Full qualification also exercises the actual bounded proposal path.
    if ready:
        try:
            obj=proposal(run)
            save(run/'proposal-smoke.json',obj)
        except Exception as e:
            ready=False
            save(run/'proposal-error.json',{'error':str(e)})
    hashes=json.loads((run/'hashes.json').read_text())
    if fingerprint()!=hashes:
        ready=False
    save(BASE/'readiness.json',dict(ready=ready,hashes=hashes,run=str(run),train_seconds=seconds,
        quality_qualified=result['status']=='keep',
        baseline_artifacts={n:digest(run/'trials/000'/n) for n in ['metrics.json','training.json','weights.npz','weights.2bit','model.json']
                            if (run/'trials/000'/n).exists()}))
    print(json.dumps(dict(ready=ready,run=str(run),status=result['status'])))


def start():
    r=json.loads((BASE/'readiness.json').read_text())
    if not r['ready']:
        raise ValueError('Full baseline and proposal qualification required')
    verify(r['hashes'])
    for name,h in r['baseline_artifacts'].items():
        if digest(Path(r['run'])/'trials/000'/name)!=h:
            raise ValueError('Readiness evidence changed')
    load(verify_audio=True)
    latest=RUNS/'latest.json'
    if latest.exists():
        old=json.loads(latest.read_text())
        status=subprocess.run(['systemctl','--user','is-active',old['unit']],capture_output=True,text=True)
        if status.stdout.strip() in ('active','activating'):
            raise ValueError('Existing STT research service active')
    run=RUNS/timestamp()
    run.mkdir(parents=True)
    snapshot(run)
    unit='wilderness-stt-'+run.name.lower()
    save(run/'state.json',dict(status='starting',unit=unit))
    save(latest,dict(run=str(run),unit=unit))
    subprocess.run(['systemd-run','--user','--unit',unit,'--property=Type=exec',
       '--property=RuntimeMaxSec=3600','--property=TimeoutStopSec=1','--property=KillMode=control-group',
       '--property=MemoryMax=52G','--property=WorkingDirectory='+str(ROOT),
       '--property=StandardOutput=append:'+str(run/'supervisor.log'),
       '--property=StandardError=append:'+str(run/'supervisor.log'),
       '--setenv=PATH='+os.environ['PATH'],str(PYTHON),str(BASE/'control.py'),'_supervise','--run',str(run)],check=True)
    print('Started '+str(run))


def supervise(run):
    deadline=time.monotonic()+CFG['duration_seconds']
    save(run/'state.json',dict(status='running',started_at=timestamp()))
    envelope=CFG['training_seconds']+10+CFG['evaluation_seconds']+5
    try:
        recipe=json.loads((run/'frozen/custom/stt/recipe.json').read_text())
        result=trial(run,recipe,CFG['training_seconds'],'Frozen initial baseline')
        if result['status'] not in ('keep','infeasible_best'):
            raise ValueError('Baseline failed independent qualification')
        errors=0
        while deadline-time.monotonic()>envelope+CFG['proposal_seconds']+CFG['report_reserve_seconds']:
            try:
                obj=proposal(run)
                result=trial(run,obj['recipe'],CFG['training_seconds'],obj['hypothesis'])
                errors=errors+1 if result['status']=='error' else 0
            except Exception as e:
                errors+=1
                save(run/f'proposal-failure-{timestamp()}.json',{'error':str(e)})
            if errors>=3:
                raise RuntimeError('Three consecutive failures; stopping rather than fabricating progress')
        save(run/'state.json',dict(status='completed',finished_at=timestamp()))
    except Exception as e:
        save(run/'state.json',dict(status='failed',error=str(e),finished_at=timestamp()))
    finally:
        report(run)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['preflight','start','status','stop','_supervise'])
    p.add_argument('--train-seconds',type=int,default=300)
    p.add_argument('--authorize-one-hour',action='store_true')
    p.add_argument('--run',type=Path)
    args=p.parse_args()
    if args.command=='preflight':
        preflight(args.train_seconds)
    elif args.command=='start':
        if not args.authorize_one_hour:
            p.error('Explicit --authorize-one-hour required')
        start()
    elif args.command=='_supervise':
        supervise(args.run)
    elif args.command=='stop':
        r=json.loads((RUNS/'latest.json').read_text())
        subprocess.run(['systemctl','--user','stop',r['unit']],check=True)
        save(Path(r['run'])/'state.json',dict(status='stopped',finished_at=timestamp()))
    else:
        for name in [BASE/'readiness.json',RUNS/'latest.json']:
            if name.exists():
                obj=json.loads(name.read_text())
                print(json.dumps({k:v for k,v in obj.items() if k not in ('hashes','baseline_artifacts')}))
        if (RUNS/'latest.json').exists():
            r=json.loads((RUNS/'latest.json').read_text())
            print((Path(r['run'])/'state.json').read_text())
            subprocess.run(['systemctl','--user','show',r['unit'],'-p','ActiveState','-p','Result'],check=False)


if __name__=='__main__':
    main()
