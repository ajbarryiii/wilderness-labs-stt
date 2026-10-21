"""Aggregate complete isolated comparison workers without loading CUDA."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args=p.parse_args()
    from .compare_v2 import VARIANTS
    reports={v:json.loads((args.directory/f'{v}.json').read_text()) for v in VARIANTS
             if (args.directory/f'{v}.json').exists()}
    rows=[]
    for name, d in reports.items():
        assert len(d['workloads']) == len(d['args']['durations'])
        for row in d['workloads']:
            assert len(row['windows']) == d['args']['repeats']
            rows.append({'variant':name, 'audio_seconds':row['record']['duration'], **row['summary'],
                'process_warm_peak_bytes':max(w['process_sampled_peak_bytes'] for w in row['windows']),
                'process_peak_including_first_bytes':row['process_sampled_peak_including_first_bytes'],
                'torch_peak_allocated':max((w['memory'].get('torch_peak_allocated',0) for w in row['windows']),default=0) or None,
                'torch_reserved':max((w['memory'].get('torch_reserved',0) for w in row['windows']),default=0) or None})
    reference=reports.get('nemo_fp32',{}).get('quality',[])
    quality={}
    for name,d in reports.items():
        if not reference or 'quality' not in d:continue
        q=d['quality']; assert len(q)==len(reference)>0
        per_set={}
        for a,b in zip(q,reference):
            assert a['audio_filepath']==b['audio_filepath'] and a['text']==b['text']
            score=a['score']
            if not score['scored']: continue
            s=per_set.setdefault(a['source'],{'words':0,'edits':0})
            s['words']+=len(score['ref_norm'].split())
            s['edits']+=score['S']+score['D']+score['I']
        words=sum(s['words'] for s in per_set.values()); edits=sum(s['edits'] for s in per_set.values())
        quality[name]={'utterances':len(q), 'words':words,'edits':edits,'wer_percent':edits/words*100,
            'exact_matches_to_nvidia_fp32':sum(a['hypothesis']==b['hypothesis'] for a,b in zip(q,reference)),
            'per_set':per_set}
    summary={'scope':reports['packed_optimized']['scope'], 'performance':rows,'quality':quality,
             'memory_loaded':{v:d['loaded'] for v,d in reports.items()},
             'process_peak_including_load':{v:d['performance_phase_process_peak_bytes'] for v,d in reports.items()}}
    (args.directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for seconds in reports['packed_optimized']['args']['durations']:
        print(f'\nAudio {seconds}s\n| Variant | ms | GPU J | Process GB | Live peak GB | Reserved GB |\n|---|---:|---:|---:|---:|---:|')
        for r in rows:
            if abs(r['audio_seconds']-seconds) > .1: continue
            live=f"{r['torch_peak_allocated']/1e9:.3f}" if r['torch_peak_allocated'] else '—'
            reserve=f"{r['torch_reserved']/1e9:.3f}" if r['torch_reserved'] else '—'
            print(f"| {r['variant']} | {r['median_ms']:.2f} | {r['joules_per_iteration']:.2f} | {r['process_warm_peak_bytes']/1e9:.3f} | {live} | {reserve} |")
    print('\nQuality',json.dumps(quality,indent=2))

if __name__=='__main__': main()
