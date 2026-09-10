#!/usr/bin/env python3
"""Preflight-gated, externally supervised GPU research; starts only explicitly."""
import argparse, csv, hashlib, json, math, os, shutil, signal, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
BASE = Path(__file__).resolve().parent
REPO = BASE.parent.parent
RUNS = BASE / 'runs'
CFG = json.loads((BASE / 'experiment.json').read_text())
FIELDS = ['trial','source_sha256','status','fnr','fpr','cpu_p95_ms','num_parameters','seconds','hypothesis','error']

def save(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(obj,indent=2)+'\n'); t.replace(p)
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def clean_env():
    env=os.environ.copy()
    for k in ('GIT_DIR','GIT_WORK_TREE','PYTHONPATH','PYTHONHOME'): env.pop(k,None)
    env['PYTHONDONTWRITEBYTECODE']='1'
    return env
def cmd(argv,**kw):
    kw.setdefault('env',clean_env())
    return subprocess.run(argv,check=True,text=True,capture_output=True,**kw).stdout
def remaining(s): return s['deadline_epoch']-time.time()
def resolve_run(value=None):
    p=Path(value).resolve() if value else (Path(json.loads((RUNS/'latest.json').read_text())['run_dir']) if (RUNS/'latest.json').exists() else None)
    if p and p.parent!=RUNS.resolve(): raise ValueError('Run must be a direct child of runs')
    return p

def validate_metrics(m):
    for k in ('fnr','fpr','cpu_p95_ms','score_range','training_seconds'):
        if type(m.get(k)) not in (int,float) or not math.isfinite(m[k]) or m[k]<0: raise ValueError('Invalid metric: '+k)
    if m['fnr']>1 or m['fpr']>1: raise ValueError('Rates must be in [0,1]')
    if type(m.get('num_parameters')) is not int or not 0<m['num_parameters']<=CFG['vad_max_parameters']: raise ValueError('Parameter contract failed')
    if m.get('precision_ok') is not True or m.get('label_granularity') not in ('frame','clip'): raise ValueError('Precision/label contract failed')
    for k in ('speech_samples','nonspeech_samples'):
        if type(m.get(k)) is not int or m[k]<=0: raise ValueError('Positive denominator required: '+k)
    if m.get('data_kind')!='real_audio': raise ValueError('Scored trials require real audio')
    if m.get('training_cuda_verified') is not True or type(m.get('training_steps')) is not int or m['training_steps']<1 or m['training_seconds']<=0: raise ValueError('Actual CUDA training evidence missing')

def feasible(m): return m['fnr']<=CFG.get('max_fnr',.01) and m['fpr']<=CFG.get('max_fpr',.2)
def useful(m):
    return m['fpr']<.95 and m['fnr']<.95 and (m['fnr']+m['fpr'])/2<.49 and m.get('score_range',0)>1e-8

def improves(m,b):
    if not useful(m): return False
    if b is None or not useful(b): return True
    a_ok,b_ok=feasible(m),feasible(b)
    if a_ok!=b_ok: return a_ok
    if not a_ok:
        def penalty(x): return max(x['fnr']/CFG.get('max_fnr',.01),x['fpr']/CFG.get('max_fpr',.2))
        pa,pb=penalty(m),penalty(b)
        return pa<pb-.01 or (abs(pa-pb)<.01 and m['fnr']+m['fpr']<b['fnr']+b['fpr']-.005)
    return m['fpr']<=b['fpr']-.005 or (m['fnr']<=b['fnr']+.001 and m['fpr']<=b['fpr']+.001 and m['cpu_p95_ms']<b['cpu_p95_ms']*.9)

def integrity(run):
    if (run/'external_frozen.json').exists():
        for path,h in json.loads((run/'external_frozen.json').read_text()).items():
            if digest(path)!=h: raise ValueError('Frozen runtime dependency changed: '+path)
    for name,h in json.loads((run/'frozen.json').read_text()).items():
        if digest(run/'workspace'/name)!=h: raise ValueError('Frozen file changed: '+name)

def freeze(run):
    if (run/'frozen.json').exists(): raise ValueError('Already frozen')
    ws=run/'workspace'
    for name in CFG['protected_after_freeze']+CFG['candidate_files']:
        if not (ws/name).is_file(): raise ValueError('Missing prepared file: '+name)
    files=cmd(['git','ls-files','-z'],cwd=ws).split('\0')
    hashes={x:digest(ws/x) for x in files if x and x not in CFG['candidate_files']}
    if not all(x in hashes for x in CFG['protected_after_freeze']): raise ValueError('Protected files must be snapshotted')
    save(run/'frozen.json',hashes)
    save(run/'external_frozen.json',{str(REPO/name):digest(REPO/name) for name in runtime().get('dependency_files',[])})
    # Evaluator code lives outside the agent's writable workspace.
    shutil.copytree(ws/'custom/vad',run/'evaluator',ignore=shutil.ignore_patterns('__pycache__','train.py'))

def terminate(p):
    try:
        os.killpg(p.pid,signal.SIGTERM); p.wait(timeout=1)
    except ProcessLookupError: pass
    except subprocess.TimeoutExpired:
        try: os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        p.wait()

def bounded(argv,cwd,seconds,log,pid_file=None):
    if seconds<=0: raise TimeoutError('No trial time remains')
    with open(log,'w') as output:
        p=subprocess.Popen(argv,cwd=cwd,stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=clean_env())
        if pid_file: save(pid_file,dict(pid=p.pid,started_at=now()))
        try: rc=p.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            terminate(p); raise TimeoutError('Trial process exceeded its time budget')
    if rc: raise RuntimeError(f'Exit {rc}; see {log}')

def runtime(): return json.loads((BASE/'runtime_contract.json').read_text())
def python_prefix(run=None):
    return json.loads((run/'workspace/runtime_contract.json').read_text())['python_command'] if run else runtime()['python_command']
def gpu_monitor(out):
    h=open(out/'gpu.csv','w')
    p=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits','-l','1'],stdout=h,stderr=subprocess.DEVNULL,start_new_session=True)
    ph=open(out/'gpu-processes.csv','w')
    pp=subprocess.Popen(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits','-l','1'],stdout=ph,stderr=subprocess.DEVNULL,start_new_session=True)
    return [(p,h),(pp,ph)]

def phase(run,name,**extra): save(run/'progress.json',dict(phase=name,updated_at=now(),**extra))

def trial(run,label,hypothesis,train_seconds=None,seed=20260909,rank=True):
    s=json.loads((run/'state.json').read_text())
    training=CFG['trial_training_seconds'] if train_seconds is None else train_seconds
    budget=training+120
    if s['status'] not in ('running','preflight') or remaining(s)<budget+CFG['report_reserve_seconds']: raise ValueError('Insufficient authorized time for trial plus reporting')
    integrity(run); ws=run/'workspace'; trials=run/'trials'; trials.mkdir(exist_ok=True)
    count=len(list(trials.glob('[0-9][0-9][0-9]')))
    if count==0 and label!='baseline': raise ValueError('First trial must be baseline')
    out=trials/f'{count:03d}'; out.mkdir(); src=ws/'custom/vad/train.py'
    shutil.copy2(src,out/'train.py'); save(out/'request.json',dict(label=label,hypothesis=hypothesis,started_at=now(),train_seconds=training,seed=seed,rank=rank))
    started=time.monotonic(); row={k:'' for k in FIELDS}; row.update(trial=out.name,source_sha256=digest(src),status='error',hypothesis=hypothesis)
    monitors=gpu_monitor(out)
    try:
        phase(run,'training',trial=out.name,training_seconds=training)
        bounded(python_prefix(run)+[str(src),'--output',str(out),'--train-seconds',str(training),'--seed',str(seed)],ws,training,out/'train.log',out/'training-process.json')
        train_wall=time.monotonic()-started
        if train_wall<training*.5: raise ValueError('Training finished prematurely instead of using the allotted budget')
        integrity(run); phase(run,'evaluating',trial=out.name)
        metrics_path=out/'metrics.json'
        if metrics_path.exists(): metrics_path.unlink()
        bounded(python_prefix(run)+[str(run/'evaluator/prepare.py'),'evaluate','--candidate',str(out),'--output',str(metrics_path)],ws,budget-(time.monotonic()-started),out/'evaluate.log')
        integrity(run)
        if digest(src)!=row['source_sha256']: raise ValueError('Candidate changed during trial')
        m=json.loads(metrics_path.read_text()); validate_metrics(m)
        if m['training_seconds']<training*.5 or m['training_steps']<10: raise ValueError('Insufficient measured learning work')
        pid=json.loads((out/'training-process.json').read_text())['pid']
        if not any(line.split(',')[0].strip()==str(pid) for line in (out/'gpu-processes.csv').read_text().splitlines()): raise ValueError('NVIDIA telemetry did not observe the training process on GPU')
        b=json.loads((ws/'custom/vad/benchmark.json').read_text())
        if m['label_granularity']!=b['label_granularity']: raise ValueError('Label granularity changed')
        best_path=run/'best.json'; best=json.loads(best_path.read_text()) if best_path.exists() else None
        keep=rank and improves(m,best['metrics'] if best else None)
        row['status']=('keep' if feasible(m) else 'infeasible_best') if keep else ('discard' if useful(m) else 'rejected_degenerate')
        if not rank: row['status']='confirmation' if useful(m) else 'confirmation_degenerate'
        row.update({k:m[k] for k in ('fnr','fpr','cpu_p95_ms','num_parameters')})
        if keep:
            save(best_path,dict(trial=out.name,source_sha256=row['source_sha256'],metrics=m)); shutil.copy2(out/'train.py',run/'best_train.py')
    except Exception as e:
        row.update(status='timeout' if isinstance(e,TimeoutError) else 'error',error=str(e))
    finally:
        for monitor,handle in monitors: terminate(monitor); handle.close()
    row['seconds']=round(time.monotonic()-started,3); save(out/'result.json',row)
    with open(run/'results.tsv','a',newline='') as h: csv.DictWriter(h,FIELDS,delimiter='\t').writerow(row)
    phase(run,'trial_finished',result=row); write_report(run); print(json.dumps(row),flush=True)
    return row

def snapshot(ws):
    (ws/'custom').mkdir(parents=True)
    for name in ('AGENTS.md','LICENSE','.gitignore'): shutil.copy2(REPO/name,ws/name)
    shutil.copy2(REPO/'custom/PLAN.md',ws/'custom/PLAN.md')
    shutil.copytree(REPO/'custom/vad',ws/'custom/vad',ignore=shutil.ignore_patterns('__pycache__','artifacts','cache','*.pt','*.npz','*.npy'))
    if (REPO/'custom/tests').exists(): shutil.copytree(REPO/'custom/tests',ws/'custom/tests',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('program.md','experiment.json','runtime_contract.json'): shutil.copy2(BASE/name,ws/name)
    for name in runtime().get('dependency_files',[]):
        dest=ws/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(REPO/name,dest)
    with open(ws/'.gitignore','a') as h: h.write('\n.cache/\ndata/\nruns/\n*.pt\n*.npz\n*.npy\n__pycache__/\njournal.md\nREPORT.md\nBLOCKED.md\nproposal.json\n')
    cmd(['git','init','-b','codex/autoresearch-hour1'],cwd=ws)
    cmd(['git','config','user.name','Autoresearch experiment'],cwd=ws)
    cmd(['git','config','user.email','autoresearch@localhost'],cwd=ws)
    cmd(['git','add','.'],cwd=ws); cmd(['git','commit','-m','Snapshot validated GPU experiment before timer'],cwd=ws)

def fingerprint():
    names=['control.py','experiment.json','program.md','runtime_contract.json']
    hashes={'custom/autoresearch/'+x:digest(BASE/x) for x in names}
    for p in sorted((REPO/'custom/vad').glob('*')):
        if p.is_file() and p.suffix in ('.py','.json'): hashes[str(p.relative_to(REPO))]=digest(p)
    for name in runtime().get('dependency_files',[]): hashes[name]=digest(REPO/name)
    return hashes

def check():
    if CFG['duration_seconds']!=3600 or CFG['auto_start'] is not False: raise ValueError('Duration/authorization contract failed')
    for x in ('codex','systemctl','systemd-run','nix','git','nvidia-smi'):
        if not shutil.which(x): raise ValueError('Missing prerequisite: '+x)
    cmd(['codex','login','status']); cmd(['systemctl','--user','show-environment'])
    r=runtime()
    print(cmd(r['verify_command'],cwd=REPO,timeout=120).strip())
    b=json.loads((REPO/'custom/vad/benchmark.json').read_text())
    if b.get('data_kind')!='real_audio': raise ValueError('Real audio benchmark required')
    for name in CFG['protected_after_freeze']+CFG['candidate_files']:
        if not (REPO/name).is_file(): raise ValueError('Missing prepared file: '+name)
    # The fixed evaluator verifies cached data hashes/provenance without scoring a test set.
    print(cmd(python_prefix()+[str(REPO/'custom/vad/prepare.py'),'validate-data'],cwd=REPO,timeout=120).strip())
    print('GPU/runtime/data prerequisites verified. No research timer started.')

def preflight(seconds):
    check(); verified_hashes=fingerprint(); RUNS.mkdir(exist_ok=True)
    run=RUNS/('preflight-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')); run.mkdir(); snapshot(run/'workspace'); freeze(run)
    save(run/'state.json',dict(status='preflight',run_dir=str(run),started_at=now(),deadline_epoch=time.time()+seconds+600))
    with open(run/'results.tsv','w',newline='') as h: csv.DictWriter(h,FIELDS,delimiter='\t').writeheader()
    row=trial(run,'baseline','Pre-timer actual CUDA training and independent real-audio evaluation',seconds)
    good=row['status'] in ('keep','infeasible_best')
    state=json.loads((run/'state.json').read_text()); state['status']='preflight' if good else 'preflight_failed'; save(run/'state.json',state)
    if not good: raise ValueError('Baseline preflight failed; inspect '+str(run))
    # Exercise the real sandboxed edit -> supervisor train/evaluate handoff before the timer.
    before=digest(run/'workspace/custom/vad/train.py')
    proposal=propose(run,0,CFG.get('agent_proposal_seconds',120))
    if digest(run/'workspace/custom/vad/train.py')==before: raise ValueError('Preflight proposal did not change candidate')
    state=json.loads((run/'state.json').read_text())
    state.update(status='preflight',deadline_epoch=time.time()+600)
    save(run/'state.json',state)
    candidate=trial(run,proposal['label'],proposal['hypothesis'],60,rank=False)
    if candidate['status'] in ('error','timeout','rejected_degenerate','confirmation_degenerate'): raise ValueError('Preflight proposal handoff failed: '+str(candidate))
    state.update(status='preflight_passed',finished_at=now()); save(run/'state.json',state); write_report(run)
    if fingerprint()!=verified_hashes: raise ValueError('Prepared code changed during preflight; readiness withheld')
    save(BASE/'readiness.json',dict(status='ready',verified_at=now(),verified_epoch=time.time(),baseline_run=str(run),baseline_result=row,proposal_smoke_result=candidate,hashes=verified_hashes,preflight_training_seconds=seconds))
    print('READY: validated baseline; one-hour timer has not started. '+str(run))

def readiness():
    r=json.loads((BASE/'readiness.json').read_text())
    if r['status']!='ready' or time.time()-r['verified_epoch']>86400: raise ValueError('Readiness missing or stale; run preflight')
    if r['hashes']!=fingerprint(): raise ValueError('Prepared code/config changed; rerun preflight')
    if not r.get('proposal_smoke_result'): raise ValueError('Sandboxed proposal handoff is not verified')
    if r.get('preflight_training_seconds',0)<60: raise ValueError('At least 60 seconds of preflight training required')
    return r

def start(authorized):
    if not authorized: raise ValueError('Explicit start --authorize-one-hour required')
    previous=resolve_run()
    if previous:
        old=json.loads((previous/'state.json').read_text())
        active=cmd(['systemctl','--user','show',old['unit'],'-p','ActiveState']).strip() if old.get('unit') else ''
        if active in ('ActiveState=active','ActiveState=activating'): raise ValueError('Existing research service is active')
    r=readiness(); check()
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); run=RUNS/tag; run.mkdir(); snapshot(run/'workspace'); freeze(run)
    if fingerprint()!=r['hashes']: raise ValueError('Prepared files changed during launch; service not started')
    for name,h in r['hashes'].items():
        p=run/'workspace'/name
        if p.exists() and digest(p)!=h: raise ValueError('Snapshot differs from verified source: '+name)
    save(run/'readiness.json',r)
    unit='wilderness-autoresearch-'+tag.lower()
    save(run/'state.json',dict(status='starting',created_at=now(),unit=unit,duration_seconds=3600,run_dir=str(run)))
    save(RUNS/'latest.json',{'run_dir':str(run)})
    with open(run/'results.tsv','w',newline='') as h: csv.DictWriter(h,FIELDS,delimiter='\t').writeheader()
    argv=['systemd-run','--user','--unit',unit,'--property=Type=exec','--property=RuntimeMaxSec=3600','--property=TimeoutStopSec=1','--property=KillMode=control-group','--property=KillSignal=SIGKILL','--property=SendSIGKILL=yes','--property=MemoryMax=52G','--property=WorkingDirectory='+str(REPO),'--property=StandardOutput=append:'+str(run/'supervisor.log'),'--property=StandardError=append:'+str(run/'supervisor.log'),'--setenv=PATH='+os.environ['PATH'],sys.executable,str(BASE/'control.py'),'_supervise','--run',str(run)]
    try: print(cmd(argv))
    except Exception:
        s=json.loads((run/'state.json').read_text()); s['status']='launch_failed'; save(run/'state.json',s); raise
    print('Started validated one-hour GPU research session: '+str(run))

def propose(run,turn,seconds,report_only=False):
    ws=run/'workspace'; s=json.loads((run/'state.json').read_text())
    proposal=ws/'proposal.json'
    if proposal.exists(): proposal.unlink()
    phase(run,'reporting' if report_only else 'proposing',turn=turn,time_limit_seconds=seconds)
    rows=(run/'results.tsv').read_text()
    best=(run/'best.json').read_text() if (run/'best.json').exists() else 'None yet'
    prompt=(ws/'program.md').read_text()+f'\nAUTHORIZED ACTIVE SESSION. Workspace {ws}. Run {run}. Remaining {int(remaining(s))} seconds.\nResults:\n{rows}\nBest:\n{best}\n'
    prompt+= ('REPORT ONLY. Write REPORT.md synthesis. Do not edit code or propose a trial.' if report_only else 'Make one concrete candidate change to custom/vad/train.py. Then write proposal.json with nonempty string fields label and hypothesis. Do not run training, evaluator, controller, git mutations, or change other code. The supervisor runs GPU training immediately after your response. You have '+str(seconds)+' seconds. Read errors/logs as needed. Finish promptly.')
    argv=['codex','exec','--sandbox','workspace-write','-c','approval_policy="never"','-c','sandbox_workspace_write.network_access=false','--cd',str(ws),'--json','--color','never','--output-last-message',str(ws/'agent-final.md'),'-']
    with open(run/f'agent-{turn:02d}.jsonl','w') as log,open(run/f'agent-{turn:02d}.stderr','w') as err:
        p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=log,stderr=err,text=True,start_new_session=True,env=clean_env())
        try: p.communicate(prompt,timeout=seconds)
        except subprocess.TimeoutExpired:
            terminate(p); raise TimeoutError('Agent proposal exceeded bounded editing window')
    if p.returncode: raise RuntimeError('Agent failed; inspect stderr')
    integrity(run)
    if report_only: return
    obj=json.loads(proposal.read_text())
    if not all(isinstance(obj.get(k),str) and obj[k].strip() for k in ('label','hypothesis')): raise ValueError('Invalid proposal.json')
    return obj

def write_report(run):
    rows=list(csv.DictReader((run/'results.tsv').open(),delimiter='\t'))
    best=json.loads((run/'best.json').read_text()) if (run/'best.json').exists() else None
    s=json.loads((run/'state.json').read_text())
    text='# Supervised GPU experiment status\n\nStatus: '+s['status']+'. Updated: '+now()+'.\n\n'
    text+='Real-audio clip classification is a feasibility proxy; frame VAD, iPhone energy and a 0.5–1B recognizer remain unvalidated. Remote CPU timing is a reference only.\n\n'
    text+='| Trial | Status | FNR | FPR | CPU p95 ms | Seconds |\n|---|---|---:|---:|---:|---:|\n'
    for x in rows: text+='| '+' | '.join(x[k] for k in ('trial','status','fnr','fpr','cpu_p95_ms','seconds'))+' |\n'
    text+='\nRetained candidate: '+(best['trial'] if best else 'none')+'.\n'
    if best and not feasible(best['metrics']): text+='The retained candidate remains infeasible against the provisional joint FNR/FPR gates.\n'
    if (run/'failure.json').exists(): text+='\nRecorded interruption: '+json.loads((run/'failure.json').read_text())['error']+'\n'
    text+='\nSee per-trial checkpoints, independent metrics, train/evaluation logs and gpu.csv for evidence.\n'
    (run/'REPORT.md').write_text(text)

def supervise(run):
    s=json.loads((run/'state.json').read_text()); s.update(status='running',started_at=now(),started_epoch=time.time()); s['deadline_epoch']=s['started_epoch']+CFG['duration_seconds']; save(run/'state.json',s)
    errors=0; outcome='completed'; turn=0
    try:
        row=trial(run,'baseline','Validated initial ternary model on frozen real-audio splits')
        if row['status'] in ('error','timeout'): raise RuntimeError('Previously validated baseline failed in service: '+row['error'])
        while remaining(s)>CFG['report_reserve_seconds']+CFG['trial_timeout_seconds']+15:
            turn+=1
            seconds=min(CFG.get('agent_proposal_seconds',120),int(remaining(s)-CFG['report_reserve_seconds']-CFG['trial_timeout_seconds']-5))
            if (run/'best_train.py').exists(): shutil.copy2(run/'best_train.py',run/'workspace/custom/vad/train.py')
            before=digest(run/'workspace/custom/vad/train.py')
            try:
                proposal=propose(run,turn,max(10,seconds))
                if digest(run/'workspace/custom/vad/train.py')==before: raise ValueError('Proposal did not change candidate; refusing duplicate GPU trial')
                row=trial(run,proposal['label'],proposal['hypothesis'])
                if row['status'] in ('error','timeout'): raise RuntimeError(row['error'])
                errors=0
            except Exception as e:
                errors+=1; save(run/f'agent-{turn:02d}-error.json',dict(error=str(e),at=now())); print('Candidate error: '+str(e),flush=True)
                integrity(run)
                if (run/'best_train.py').exists(): shutil.copy2(run/'best_train.py',run/'workspace/custom/vad/train.py')
                if errors>=CFG['max_consecutive_agent_errors']: raise RuntimeError('Consecutive candidate failures: '+str(e))
        # Use a short final window for fresh-seed robustness evidence, never for ranking.
        confirm_seconds=min(CFG['trial_training_seconds'],int(remaining(s)-120-CFG['report_reserve_seconds']-5))
        if confirm_seconds>=60 and (run/'best_train.py').exists():
            shutil.copy2(run/'best_train.py',run/'workspace/custom/vad/train.py')
            trial(run,'confirmation-short','Unranked retained-recipe fresh-seed confirmation; shorter budget is not a controlled candidate comparison',confirm_seconds,seed=20260910,rank=False)
        if remaining(s)>10:
            try: propose(run,turn+1,min(90,int(remaining(s)-5)),report_only=True)
            except Exception as e: print('Report synthesis failed: '+str(e),flush=True)
    except Exception as e:
        outcome='failed'; save(run/'failure.json',dict(error=str(e),at=now())); print('Research interruption: '+str(e),flush=True)
        # Keep the GPU useful if proposal generation fails but the validated recipe works.
        # Fresh-seed repetitions are robustness evidence, never ranked as discoveries.
        retry=0
        while (run/'best_train.py').exists() and remaining(s)>CFG['report_reserve_seconds']+180+5:
            retry+=1
            seconds=min(CFG['trial_training_seconds'],int(remaining(s)-120-CFG['report_reserve_seconds']-5))
            shutil.copy2(run/'best_train.py',run/'workspace/custom/vad/train.py')
            try:
                row=trial(run,'recovery-confirmation',f'Unranked retained-recipe seed repeat after proposal interruption; seed {20262000+retry}',seconds,seed=20262000+retry,rank=False)
            except Exception as failure:
                print('Validated recovery could not run: '+str(failure),flush=True); break
            if row['status'] in ('error','timeout'): break
            outcome='completed_with_proposal_failures'
    s.update(status=outcome,finished_at=now()); save(run/'state.json',s); phase(run,outcome); write_report(run); print(json.dumps(s),flush=True)

def status(run):
    if run is None: print(json.dumps(dict(status='prepared_not_started',readiness_present=(BASE/'readiness.json').exists()))); return
    s=json.loads((run/'state.json').read_text())
    if s.get('unit'):
        out=cmd(['systemctl','--user','show',s['unit'],'-p','ActiveState','-p','Result']); s['systemd']=out.strip()
        if s['status'] in ('starting','running') and ('ActiveState=failed' in out or 'ActiveState=inactive' in out):
            s['status']='deadline' if 'Result=timeout' in out else 'terminated'; s['finished_at']=now(); save(run/'state.json',{k:v for k,v in s.items() if k!='systemd'}); write_report(run)
    if 'deadline_epoch' in s: s['remaining_seconds']=max(0,round(remaining(s))) if s['status']=='running' else 0
    if (run/'progress.json').exists(): s['progress']=json.loads((run/'progress.json').read_text())
    if (run/'results.tsv').exists(): s['trials']=list(csv.DictReader((run/'results.tsv').open(),delimiter='\t'))
    print(json.dumps(s,indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['check','preflight','status','start','stop','freeze','trial','_supervise']); p.add_argument('--run'); p.add_argument('--authorize-one-hour',action='store_true'); p.add_argument('--train-seconds',type=int,default=60); p.add_argument('--label',default='candidate'); p.add_argument('--hypothesis',default=''); a=p.parse_args()
    if a.action=='start': start(a.authorize_one_hour)
    elif a.action=='check': check()
    elif a.action=='preflight':
        if not 60<=a.train_seconds<=300: raise ValueError('Preflight must be 60–300 seconds')
        preflight(a.train_seconds)
    else:
        run=resolve_run(a.run)
        if a.action=='status': status(run)
        elif run is None: raise ValueError('No session exists')
        elif a.action=='stop':
            s=json.loads((run/'state.json').read_text()); cmd(['systemctl','--user','stop',s['unit']]); s.update(status='stopped_by_user',finished_at=now()); save(run/'state.json',s)
        elif a.action=='freeze': raise ValueError('Freeze is performed by supervisor before timer; agents must not invoke it')
        elif a.action=='trial': raise ValueError('Trials are owned by supervisor; agent writes proposal.json')
        elif a.action=='_supervise': supervise(run)
if __name__=='__main__':
    try: main()
    except Exception as e: print(str(e),file=sys.stderr); sys.exit(1)
