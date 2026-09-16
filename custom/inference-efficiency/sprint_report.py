"""Readable per-step watts, latency, throughput and energy from sprint artifacts."""
import argparse
import json
from pathlib import Path
import statistics

from paths import artifact, save


def report(job):
    rows = [json.loads(p.read_text()) for p in (job / 'results').glob('*.json')]
    rows.sort(key=lambda r: r['started_utc'])
    lines = ['# Kernel sprint measurements', '',
             'All energy is GPU board energy. Each successful step contains a fresh baseline/candidate pair; each window lasts at least 60 seconds and completes whole 16-clip cycles. One clip is 30 seconds of audio plus 128 forced decoder outputs.', '',
             '| Step | Model | Phase | Candidate W | Baseline W | Candidate J/clip | Baseline J/clip | Energy change | p95 ms (candidate / baseline) | Clips/s (candidate / baseline) |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['status'] != 'measured':
            lines.append(f"| {row['name']} | {row['distribution']} | rejected: {row.get('error','')} | | | | | | | |")
            continue
        a, b = row['windows']['candidate'], row['windows']['baseline']
        lines.append(f"| {row['name']} | {row['distribution']} | {row['phase']} | {a['avg_watts']:.2f} | {b['avg_watts']:.2f} | {30*a['gpu_j_per_audio_second']:.3f} | {30*b['gpu_j_per_audio_second']:.3f} | {100*(row['energy_ratio']-1):+.2f}% | {1000*a['p95_seconds']:.2f} / {1000*b['p95_seconds']:.2f} | {1/(30*a['real_time_factor']):.3f} / {1/(30*b['real_time_factor']):.3f} |")
    confirmations = {}
    for row in rows:
        if row['phase'] == 'confirm' and row['status'] == 'measured':
            k = (row['distribution'], row.get('source_request', json.dumps(row['plugins'], sort_keys=True)))
            confirmations.setdefault(k, []).append(row)
    final = []
    for (distribution, source), group in confirmations.items():
        energy = [r['energy_ratio'] for r in group]
        latency = [r['p95_ratio'] for r in group]
        result = dict(distribution=distribution, source_request=source, plugins=group[0]['plugins'],
                      pairs=len(group), energy_ratio_mean=statistics.mean(energy), energy_ratio_range=[min(energy),max(energy)],
                      latency_ratio_mean=statistics.mean(latency),
                      latency_ratio_range=[min(latency), max(latency)],
                      watts_ratio_mean=statistics.mean(r['watts_ratio'] for r in group),
                      requests=[r['name'] for r in group],
                      confirmed=len(group)>=3 and statistics.mean(energy)<=.98 and max(energy)<1 and statistics.mean(latency)<=1.02,
                      arms={arm:{field:statistics.mean(r['windows'][arm][field] for r in group)
                                 for field in ('gpu_j_per_audio_second','avg_watts','p95_seconds','real_time_factor')}
                            for arm in ('baseline','candidate')})
        final.append(result)
    save(job / 'final-comparison.json', dict(completed_steps=len(rows), confirmation=final,
         numerical_scope='Fixed random binary/ternary Whisper medium.en, no transcription accuracy claim',
         comparison_scope='Changes versus unchanged packed runtime; no new dense/CT2 control measurement'))
    lines += ['', '## Independent confirmation', '',
              'These rows use only new confirmation pairs. Ratios are averaged within paired measurements; reported p95 values are means of window-level p95 values, not a pooled percentile. Confirmation requires at least three pairs, at least 2% mean energy savings, energy savings in every pair, and no more than 2% mean p95 regression. This practical gate is not a confidence interval.', '',
              '| Model | Pairs | Confirmed | Mean energy change (pair range) | Mean p95 change | Mean watts change |',
              '| --- | ---: | --- | ---: | ---: | ---: |']
    for row in final:
        lo, hi = row['energy_ratio_range']
        lines.append(f"| {row['distribution']} | {row['pairs']} | {row['confirmed']} | {100*(row['energy_ratio_mean']-1):+.2f}% ({100*(lo-1):+.2f}% to {100*(hi-1):+.2f}%) | {100*(row['latency_ratio_mean']-1):+.2f}% | {100*(row['watts_ratio_mean']-1):+.2f}% |")
    lines += ['', 'Final selection is evaluated only on independent confirmation pairs, grouped by the exact source request. Full source hashes and raw telemetry remain in each request/trial directory. Encoder tile selection and alternate weight decoding did not establish an energy improvement; unmeasured ideas are not included as wins. The comparison is against the previous packed implementation, without a new dense or CTranslate2 measurement.']
    (job / 'MEASUREMENTS.md').write_text('\n'.join(lines) + '\n')
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    print(json.dumps(report(artifact(args.job)), indent=2))
