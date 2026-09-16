"""Confirm an unchanged, correctness-gated screen without repeating its tests."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from paths import ROOT,artifact,digest,save,storage
from popcount_optimization_benchmark import HERE,LOCK,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--job',type=Path)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    storage()
    screen=artifact(args.screen)
    cfg=json.loads((screen/'config.json').read_text())
    if json.loads((screen/'status.json').read_text())['status']!='screen_completed':
        raise RuntimeError('A completed screen is required')
    if cfg['candidate']!='selected': raise RuntimeError('Expected the selected profile')
    for source,expected in cfg['source_hashes'].items():
        if digest(HERE/source)!=expected: raise RuntimeError(f'Screened source changed: {source}')
    for bits,a in ((1,1),(2,1),(2,2)):
        rows=json.loads((screen/f'full-check-b{bits}-a{a}.json').read_text())
        if len(rows)!=16 or not all(r['exact'] for r in rows):
            raise RuntimeError('All full-model correctness gates must pass')
    job=artifact(args.job or ROOT/'popcount-optimization'/dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    cfg.update(validated_screen=str(screen),source_hashes={**cfg['source_hashes'],
               'popcount_optimization_confirm.py':digest(HERE/'popcount_optimization_confirm.py')})
    if args.resume:
        existing=json.loads((job/'config.json').read_text())
        for key in ('source_hashes','manifest_sha256','candidate','validated_screen','seconds','repeats'):
            if existing[key]!=cfg[key]: raise RuntimeError(f'Resume mismatch: {key}')
        cfg=existing
    else:
        job.mkdir(parents=True,exist_ok=False)
        for source in cfg['source_hashes']:
            dest=job/'source'/source
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(HERE/source,dest)
        evidence=[screen/'components.json',screen/'tiny-model-checks.json',screen/'micro.json']
        evidence+=list(screen.glob('full-check-*.json'))+list(screen.glob('model-screen-*.json'))
        for source in evidence: shutil.copy2(source,job/source.name)
        save(job/'validation-receipt.json',dict(screen=str(screen),
             config_sha256=digest(screen/'config.json'),
             evidence={source.name:digest(source) for source in evidence}))
        save(job/'config.json',cfg)
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','EFFICIENCY_REQUIRE_CUDA_GEMV':'1'}
    print(f'Confirmation artifacts: {job}',flush=True)
    save(job/'status.json',dict(status='waiting_for_gpu'))
    with LOCK.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from energy import EnergyMeter
        def guard():
            with EnergyMeter() as meter:
                deadline=time.monotonic()+30
                while meter.foreign_processes() and time.monotonic()<deadline: time.sleep(.1)
                meter._guard()
                now=meter.metadata()
                if any(now[k]!=cfg['gpu'][k] for k in ('uuid','power_limit_w')):
                    raise RuntimeError('GPU identity or power limit changed')
        arms=[(1,False,'popcount'),(1,False,'selected'),(1,False,'sprint'),
              (2,False,'popcount'),(2,False,'selected'),(2,False,'sprint'),
              (2,True,'popcount'),(2,True,'selected')]
        try:
            for repeat in range(cfg['repeats']):
                for bits,a2,arm in (arms if repeat%2==0 else reversed(arms)):
                    dest=job/f'{repeat:02d}-b{bits}-a{2 if a2 else 1}-{arm}.json'
                    if dest.exists(): continue
                    guard()
                    if (job/'STOP').exists():
                        save(job/'status.json',dict(status='paused_between_windows'))
                        return
                    save(job/'status.json',dict(status='running',repeat=repeat,bits=bits,a2=a2,arm=arm))
                    subprocess.run([str(HERE/'python'),str(HERE/'popcount_optimization_benchmark.py'),
                        '--worker','--job',str(job),'--bits',str(bits),'--arm',arm,'--repeat',str(repeat)]+
                        (['--a2'] if a2 else []),check=True,env=env)
                    report(job)
            save(job/'status.json',dict(status='completed'))
            save(ROOT/'popcount-optimization/latest.json',dict(job=str(job),status='completed'))
        except Exception as exc:
            save(job/'status.json',dict(status='failed',error=str(exc)))
            raise
    print(f'Completed: {job/"REPORT.md"}',flush=True)


if __name__=='__main__': main()
