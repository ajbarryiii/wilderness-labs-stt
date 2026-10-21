"""Run correctness-gated paired CPU comparisons and retain all timed calls.

../python cpu-bitplanes/run.py --out /mnt/hd/.../cpu-bitplanes/RUN
The snapshot is created by prepare.py. Only public speech is processed.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import selectors
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = Path('/mnt/hd/wilderness-labs-stt/parakeet-ternary')


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def counters(pid):
    v = [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:]]
    busy = sum(v[i] for i in [0, 1, 2, 5, 6, 7])
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return time.monotonic(), busy, int(fields[11]) + int(fields[12])


def worker(command, out, label, env):
    monitoring = []
    with (out / f'{label}.log').open('w') as log:
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env=env, bufsize=1)
        selector = selectors.DefaultSelector()
        selector.register(p.stdout, selectors.EVENT_READ)
        previous = counters(p.pid)
        last_sample = time.monotonic()
        while p.poll() is None:
            events = selector.select(timeout=.5)
            for key, _ in events:
                line = key.fileobj.readline()
                if line:
                    log.write(line)
                    log.flush()
                    print(f'{label}: {line.rstrip()}', flush=True)
            if time.monotonic() - last_sample >= 1:
                try:
                    current = counters(p.pid)
                except FileNotFoundError:
                    continue
                dt = current[0] - previous[0]
                hz = os.sysconf('SC_CLK_TCK')
                host = (current[1]-previous[1]) / hz / dt
                process = (current[2]-previous[2]) / hz / dt
                monitoring.append({'time_monotonic_s': current[0], 'host_busy_cores': host,
                                   'process_busy_cores': process, 'other_busy_cores_estimate': max(0, host-process),
                                   'load_average': list(os.getloadavg())})
                previous = current
                last_sample = time.monotonic()
        remaining = p.stdout.read()
        log.write(remaining)
        if remaining:
            print(f'{label}: {remaining.rstrip()}', flush=True)
        selector.close()
    save(out / f'{label}-host-activity.json', monitoring)
    if p.returncode:
        raise RuntimeError(f'{label} failed with exit {p.returncode}; see its log')


def paired_ci(values):
    rng = random.Random(81509)
    boots = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(10000))
    return [boots[249], boots[9749]]


def summarize(data):
    times = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for window in data['windows']:
        for sample in window['samples']:
            for scope in ['full_ms', 'encoder_ms', 'projections_ms', 'decoder_ms']:
                times[window['backend']][sample['index']][scope].append(sample[scope])
    result = {'cpus':data['cpus'], 'components':data['components'], 'tiles':data['tiles'],
              'exact_encoder_text_tokens_all_utterances':all(r['exact_all_backends'] for r in data['validation']),
              'utterances':len(data['validation']), 'backends':{}}
    for name, records in times.items():
        report = {}
        for scope in ['full_ms', 'encoder_ms', 'projections_ms', 'decoder_ms']:
            medians = {index:statistics.median(r[scope]) for index,r in records.items()}
            report[scope] = {'median_across_utterances_ms':statistics.median(medians.values()),
                             'mean_of_utterance_medians_ms':statistics.mean(medians.values())}
            if name != 'vnni':
                changes = [100*(v/statistics.median(times['vnni'][index][scope])-1) for index,v in medians.items()]
                report[scope].update({'paired_median_change_percent':statistics.median(changes),
                                     'paired_bootstrap_95_percent_interval':paired_ci(changes),
                                     'faster_utterances':sum(v<0 for v in changes),
                                     'per_utterance_change_percent':changes})
        report['timed_calls'] = sum(len(r['full_ms']) for r in records.values())
        result['backends'][name] = report
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=10)
    p.add_argument('--windows', type=int, default=4)
    p.add_argument('--configs', default='16:2,16:3,8:3', help='THREADS:COMPONENTS comma-separated')
    p.add_argument('--skip-build', action='store_true')
    p.add_argument('--micro-only', action='store_true')
    p.add_argument('--skip-micro', action='store_true')
    a = p.parse_args()
    if not os.path.ismount('/mnt/hd') or not a.out.resolve().is_relative_to(ROOT):
        raise RuntimeError('A mounted project disk is required')
    out = a.out
    metadata = json.loads((out / 'metadata.json').read_text())
    snapshot = Path(metadata['source_snapshot'])
    env = {**os.environ, 'CARGO_HOME':str(ROOT / 'cpu-bitplanes/cargo-home'),
           'CARGO_TARGET_DIR':str(out / 'target'), 'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1'}
    if not a.skip_build:
        subprocess.run(['cargo','test','--release','--manifest-path',str(snapshot/'Cargo.toml'),'-p','lf-cpu','-p','lf-asr'], env=env, check=True)
        subprocess.run(['cargo','build','--release','--manifest-path',str(snapshot/'Cargo.toml'),'-p','bitplane-bench'], env=env, check=True)
    binary = out / 'target/release/bitplane-bench'
    sys.path.insert(0, str(PROJECT.parents[1] / 'custom/cpu-inference'))
    from hardware import CpuEnergyMeter, discover_topology
    with CpuEnergyMeter.detect() as meter:
        energy = meter.metadata()
    hardware = discover_topology()
    cpu_description = subprocess.check_output(['lscpu','--json'], text=True)
    save(out/'hardware.json', {'topology':hardware,'lscpu':json.loads(cpu_description), 'energy':energy,
                              'gpu_power_limit':subprocess.check_output(['nvidia-smi','--query-gpu=power.limit','--format=csv,noheader'], text=True).strip()})
    allowed = os.sched_getaffinity(0)
    physical = []
    seen = set()
    for cpu in sorted(allowed):
        core = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/core_id').read_text().strip()
        if core not in seen:
            seen.add(core)
            physical.append(cpu)
    if len(physical) < 16:
        raise RuntimeError('This experiment expects 16 available physical cores')
    lock_path = Path('/mnt/hd/wilderness-labs-stt/cpu-inference/active.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        micro_path = out/'micro.json'
        if not a.skip_micro:
            worker([str(binary),'micro',metadata['export'],str(micro_path),','.join(map(str,physical))],out,'micro',env)
        cases = json.loads(micro_path.read_text())['cases']
        tiles = {}
        for name in ['pn','masked','swar']:
            candidates = sorted({r['tile'] for r in cases if r['backend']==name})
            costs = {tile:sum(r['median_ms'] for r in cases if r['backend']==name and r['tile']==tile) for tile in candidates}
            tiles[name] = min(costs, key=costs.get)
        save(out/'tile-selection.json', {'tiles':tiles,'scope':'sum of 12 independent synthetic shape/component medians per tile; selected before speech timing'})
        if a.micro_only:
            print('TILES', json.dumps(tiles), flush=True)
            return
        summaries = []
        for threads,components in [tuple(map(int, v.split(':'))) for v in a.configs.split(',')]:
            label=f'{threads}cores-i8x{components}'
            raw=out/f'{label}.json'
            worker([str(binary),'speech',metadata['export'],str(out/'corpus.json'),','.join(map(str,physical[:threads])),
                    str(components),','.join(f'{name}={tile}' for name,tile in tiles.items()),str(a.seconds),str(a.windows),str(raw)],out,label,env)
            summaries.append(summarize(json.loads(raw.read_text())))
            save(out/'summary.json', {'status':'running','results':summaries,'energy':energy})
        # Snapshot hashes identify the code that actually produced these results.
        hashes = {str(f.relative_to(snapshot)):hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in snapshot.rglob('*') if f.is_file() and f.suffix in ('.rs','.toml','.lock')}
        save(out/'snapshot-sha256.json', hashes)
        save(out/'summary.json', {'status':'complete','results':summaries,'energy':energy,
                                'method':f'16 complete public utterances, eight corpora, 15-30s; {a.windows} rotated backend windows per configuration; >={a.seconds}s and >=2 complete corpus cycles per window; paired bootstrap resamples utterances, not repeated calls',
                                'scope':'warm waveform-to-text and component timings; no trained-weight or activation-precision changes; benchmark holds all backend caches, so RSS is not deployment memory'})
    print(f'RESULTS {out / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
