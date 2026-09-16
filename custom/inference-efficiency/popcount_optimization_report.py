"""Audit and summarize a completed popcount optimization confirmation."""
import argparse
import json
from pathlib import Path
import shutil

from paths import ROOT,artifact,digest,save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',type=Path,required=True)
    args=p.parse_args()
    job=artifact(args.job)
    cfg=json.loads((job/'config.json').read_text())
    if digest(ROOT/'data/manifest.json')!=cfg['manifest_sha256']:
        raise RuntimeError('Workload manifest changed')
    if json.loads((job/'status.json').read_text())['status']!='completed':
        raise RuntimeError('Confirmation has not completed')
    receipt=json.loads((job/'validation-receipt.json').read_text())
    for name,h in receipt['evidence'].items():
        if digest(job/name)!=h: raise RuntimeError(f'Validation evidence changed: {name}')
    for name,h in cfg['source_hashes'].items():
        if digest(job/'source'/name)!=h: raise RuntimeError(f'Frozen source changed: {name}')
    rows=[json.loads(p.read_text()) for p in sorted(job.glob('[0-9][0-9]-b*.json'))]
    expected={(bits,a2,arm,r) for bits,a2,arm in
        ((1,False,'popcount'),(1,False,'selected'),(1,False,'sprint'),
         (2,False,'popcount'),(2,False,'selected'),(2,False,'sprint'),
         (2,True,'popcount'),(2,True,'selected')) for r in range(3)}
    if len(rows)!=24 or {(r['bits'],r['a2'],r['arm'],r['repeat']) for r in rows}!=expected:
        raise RuntimeError('Expected exactly 24 specified measurement windows')
    for r in rows:
        if (r['measurement']['elapsed_seconds']<60 or r['clips']%16
                or len(r['clip_seconds'])!=r['clips']):
            raise RuntimeError('Incomplete measurement window')
        for gpu in (r['gpu_before'],r['gpu_after']):
            if any(gpu[k]!=cfg['gpu'][k] for k in ('uuid','power_limit_w')):
                raise RuntimeError('GPU/power-limit mismatch')
    summary=json.loads((job/'summary.json').read_text())
    comparisons={}
    lines=['# Confirmed popcount optimization','',
           'Selected dispatch: output/cache fusion and precomputed A1 counts; Tensor Core large-batch projections only for ternary; fused LayerNorm/GELU packing disabled.',
           'All three selected quantized models exactly match their previous popcount references on 16 full-size clips each.',
           'Measurement: RTX 5090, 400 W cap, seeded Whisper medium.en, 16 frozen 30-second clips, 128 forced decoder tokens.',
           'Each arm has three fresh-process windows of at least 60 seconds, completing whole clip cycles. Frontend, input transfer and quantization are included.',
           'Energy is GPU board energy; CPU energy and recognition accuracy are not measured.','',
           '| Model | Previous median ms | Selected median ms | p95 before → after ms | Latency change | J/clip before → after | Energy change |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for bits,a2 in ((1,False),(2,False),(2,True)):
        label=f'W{bits}A{2 if a2 else 1}'
        old,new=summary[label+'-popcount'],summary[label+'-selected']
        latency=100*(new['median_ms']/old['median_ms']-1)
        energy=100*(new['joules_per_clip']/old['joules_per_clip']-1)
        pairs=[]
        for repeat in range(3):
            pair={r['arm']:r for r in rows if (r['bits'],r['a2'],r['repeat'])==(bits,a2,repeat)}
            pairs.append(dict(repeat=repeat,
                latency_change_percent=100*(pair['selected']['median_ms']/pair['popcount']['median_ms']-1),
                energy_change_percent=100*(pair['selected']['joules_per_clip']/pair['popcount']['joules_per_clip']-1)))
        comparisons[label]=dict(latency_change_percent=latency,energy_change_percent=energy,paired_windows=pairs)
        lines.append(f'| {label} | {old["median_ms"]:.3f} | {new["median_ms"]:.3f} | {old["p95_ms"]:.3f} → {new["p95_ms"]:.3f} | {latency:+.2f}% | {old["joules_per_clip"]:.3f} → {new["joules_per_clip"]:.3f} | {energy:+.2f}% |')
    lines+=['','Comparison with fresh reruns of the completed sprint winners (which use FP16 activations):','',
            '| Quantized model | Selected / sprint median ms | Latency change vs sprint | Selected / sprint J/clip | Energy change vs sprint |',
            '| --- | ---: | ---: | ---: | ---: |']
    for bits,a2 in ((1,False),(2,False),(2,True)):
        label=f'W{bits}A{2 if a2 else 1}'
        new,sprint=summary[label+'-selected'],summary[f'W{bits}A16-sprint']
        latency=100*(new['median_ms']/sprint['median_ms']-1)
        energy=100*(new['joules_per_clip']/sprint['joules_per_clip']-1)
        comparisons[label]['versus_sprint']=dict(latency_change_percent=latency,energy_change_percent=energy)
        lines.append(f'| {label} | {new["median_ms"]:.3f} / {sprint["median_ms"]:.3f} | {latency:+.2f}% | {new["joules_per_clip"]:.3f} / {sprint["joules_per_clip"]:.3f} | {energy:+.2f}% |')
    lines+=['','Full-model CUDA-graph ablation (one fixed clip, 24 samples, frontend excluded; screening only):','',
            '| Model | Old popcount | Counts | + Tensor Core encoder | + Output/cache fusion | + Norm/GELU packing | Selected |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for bits,a in ((1,1),(2,1),(2,2)):
        v=json.loads((job/f'model-screen-b{bits}-a{a}.json').read_text())
        lines.append(f'| W{bits}A{a} | '+' | '.join(f'{v[arm]["median_ms"]:.3f}' for arm in ('popcount','counts','hybrid','fusion','all','selected'))+' |')
    lines+=['','Output/cache fusion supplies most of the latency improvement. Counts alone have little whole-model impact.',
            'Tensor Core tiles help the ternary encoder and cross-attention memory projections, but slow the binary encoder.',
            'The current producer-packing implementation regresses full-model latency; it is implemented and correctness-tested, but disabled in selected dispatch.',
            'The faster selected kernels can draw more power while active, so latency savings and energy savings differ.',
            'Quantized activations remain a structural model change relative to the sprint models. Seeded random weights cannot establish speech recognition quality.',
            'Preparation, weight conversion, compilation and graph capture are excluded. Original packed weights coexist with decoder bit planes and counts.','',
            'Artifacts: [raw aggregate report](REPORT.md), [paired comparisons](comparison.json), [validation receipt](validation-receipt.json), [audit](audit.json).']
    save(job/'comparison.json',comparisons)
    shutil.copy2(Path(__file__),job/'source'/Path(__file__).name)
    save(job/'audit.json',dict(passed=True,windows=len(rows),clips=sum(r['clips'] for r in rows),
        min_window_seconds=min(r['measurement']['elapsed_seconds'] for r in rows),
        source_files=len(cfg['source_hashes']),validation_evidence_files=len(receipt['evidence']),
        gpu_uuid=cfg['gpu']['uuid'],power_limit_w=cfg['gpu']['power_limit_w'],
        report_script_sha256=digest(Path(__file__))))
    (job/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__': main()
