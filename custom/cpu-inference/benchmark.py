"""Offline CPU kernel and fixed-work Whisper efficiency benchmarks."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import time

from storage import ROOT, GPU_ROOT, artifact, digest, save, storage

HERE = Path(__file__).resolve().parent
VARIANTS = ['ct2-fp32', 'ct2-int8'] + [
    f'{family}-{backend}' for family in ('w1a1', 'w2a1', 'w2a2')
    for backend in ('dense', 'scalar', 'avx512', 'avx512_opt')]
RUNTIME_ENV = ('OMP_WAIT_POLICY', 'OMP_DYNAMIC', 'CUDA_VISIBLE_DEVICES',
               'CT2_USE_MKL', 'CT2_PACKED_GEMM', 'CT2_FORCE_CPU_ISA',
               'MKL_ENABLE_INSTRUCTIONS', 'DNNL_MAX_CPU_ISA', 'CXX')


def package_versions():
    return {name: importlib.metadata.version(name)
            for name in ('torch', 'numpy', 'transformers', 'ctranslate2')}


def source_hashes():
    paths = sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.cpp')) + [HERE / 'python']
    paths += [HERE.parent / 'inference-efficiency' / name
              for name in ('runtime_model.py', 'control.py')]
    return {str(p): digest(p) for p in paths}


def workload():
    manifest = json.loads((GPU_ROOT / 'data/manifest.json').read_text())
    if len(manifest['clips']) != 16 or manifest['decoder_steps'] != 128:
        raise ValueError('Require the original 16-clip, 128-step GPU workload')
    tokens = manifest['replay_tokens']
    if len(tokens) != 128:
        raise ValueError('Require 128 forced tokens')
    for clip in manifest['clips']:
        if clip['seconds'] != 30 or clip['replay_tokens'] != tokens:
            raise ValueError('Workload duration or tokens differ')
        for field in ('pcm', 'mel'):
            path = Path(clip[field]).resolve()
            if not path.is_relative_to(GPU_ROOT) or digest(path) != clip[field + '_sha256']:
                raise ValueError(f'Invalid frozen workload file: {path}')
    return manifest


def verify_models(*, include_ct2=True):
    lock = json.loads((GPU_ROOT / 'models/model-lock.json').read_text())
    required = {'openai/whisper-medium.en': 'preprocessor_config.json'}
    if include_ct2:
        required['Systran/faster-whisper-medium.en'] = 'model.bin'
    for model_id, required_file in required.items():
        if model_id not in lock or required_file not in lock[model_id]['files']:
            raise ValueError(f'Model lock lacks required artifact: {model_id}/{required_file}')
        model = lock[model_id]
        for name, spec in model['files'].items():
            path = Path(model['path']) / name
            if not path.resolve().is_relative_to(GPU_ROOT):
                raise ValueError('Model outside mounted artifact root')
            if path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
                raise ValueError(f'Model hash mismatch: {path}')


def proc_cpu():
    """Whole-host non-idle ticks; records possible background contention."""
    fields = Path('/proc/stat').read_text().splitlines()[0].split()[1:9]
    values = [int(x) for x in fields]
    return sum(values) - values[3] - values[4]


def telemetry():
    result = {'loadavg': list(os.getloadavg()), 'affinity': sorted(os.sched_getaffinity(0))}
    result['frequencies_khz'] = {}
    for cpu in result['affinity']:
        p = Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq')
        if p.exists():
            result['frequencies_khz'][str(cpu)] = int(p.read_text())
    result['temperatures_c'] = {}
    for p in Path('/sys/class/hwmon').glob('hwmon*/temp*_input'):
        name = p.parent / 'name'
        if name.exists() and name.read_text().strip() == 'k10temp':
            result['temperatures_c'][p.name] = int(p.read_text()) / 1000
    return result


def energy_metadata():
    from hardware import CpuEnergyMeter
    with CpuEnergyMeter.detect() as meter:
        return meter.metadata()


def configure(threads, cpus):
    # Must precede importing torch, CT2, or the native OpenMP library.
    os.sched_setaffinity(0, cpus)
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    return torch


def output_digest(model, output):
    import torch
    if not isinstance(output, torch.Tensor):
        if not model.replay_verified:
            raise RuntimeError('CTranslate2 forced replay was not verified')
        return {'forced_prefix_verified': True}
    if output.numel() < 2 or not bool(torch.isfinite(output).all()) or output.float().std().item() == 0:
        raise RuntimeError('Nonfinite or degenerate model output')
    return {key: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
            for key, value in [('logits', output), ('predictions', model.last_predictions)]}


def make_runner(variant, threads, manifest):
    import numpy as np
    import torch
    from transformers import WhisperFeatureExtractor
    from runtime import CPUReplayWhisper, CPUCTranslate2Control
    extractor = WhisperFeatureExtractor.from_pretrained(
        GPU_ROOT / 'models/whisper-medium.en', local_files_only=True)
    pcm = [np.load(c['pcm'], allow_pickle=False) for c in manifest['clips']]
    tokens = manifest['replay_tokens']
    if variant.startswith('ct2-'):
        model = CPUCTranslate2Control(GPU_ROOT / 'models/ct2-medium.en',
                                     compute_type='float32' if variant == 'ct2-fp32' else 'int8_float32',
                                     threads=threads)
    else:
        family, backend = variant.split('-')
        model = CPUReplayWhisper(distribution='binary' if family == 'w1a1' else 'ternary',
                                 activation_bits=2 if family == 'w2a2' else 1,
                                 implementation=backend, threads=threads)

    def call(index):
        mel = extractor(pcm[index], sampling_rate=16000, return_tensors='np').input_features
        return model.run(mel if variant.startswith('ct2-') else torch.from_numpy(mel), tokens)
    return model, call


def model_worker(job, config, spec):
    import numpy as np
    from hardware import CpuEnergyMeter
    manifest = workload()
    verify_models(include_ct2=spec['variant'].startswith('ct2-'))
    start = time.perf_counter()
    model, call = make_runner(spec['variant'], spec['threads'], manifest)
    setup_seconds = time.perf_counter() - start
    # Validate each measured clip outside the energy/latency window. This also
    # warms the exact graph, including input preparation and all KV cache slots.
    outputs = {manifest['clips'][index]['id']: output_digest(model, call(index))
               for index in range(config['clips'])}
    for _ in range(config['warmup'] - 1):
        if output_digest(model, call(0)) != outputs[manifest['clips'][0]['id']]:
            raise RuntimeError('Non-deterministic warmup output')
    meter = CpuEnergyMeter.detect()
    idle = meter.idle(config['idle_seconds']) if config['idle_seconds'] else None
    if config['idle_seconds']:
        output_digest(model, call(0))
    latencies = []
    clip_count = config['clips']
    before = telemetry()
    cpu_before, wall_before = proc_cpu(), time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)

    def cycle():
        for index in range(clip_count):
            start = time.perf_counter()
            call(index)
            latencies.append(time.perf_counter() - start)

    measured = meter.measure(cycle, min_seconds=config['seconds'],
                             min_iterations=config['cycles'])
    meter_info = meter.metadata()
    meter.close()
    elapsed = time.perf_counter() - wall_before
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    host_cpu = (proc_cpu() - cpu_before) / os.sysconf('SC_CLK_TCK')
    own_cpu = (usage_after.ru_utime + usage_after.ru_stime
               - usage_before.ru_utime - usage_before.ru_stime)
    energy = measured['energy_joules']
    row = {**spec, 'kind': config['kind'], 'setup_seconds': setup_seconds,
           'completed_clips': len(latencies), 'clip_latency_seconds': latencies,
           'median_ms': statistics.median(latencies) * 1000,
           'p95_ms': float(np.percentile(latencies, 95)) * 1000,
           'real_time_factor': measured['elapsed_seconds'] / (30 * len(latencies)),
           'joules_per_clip': None if energy is None else energy / len(latencies),
           'cpu_j_per_audio_second': None if energy is None else energy / (30 * len(latencies)),
           'average_watts': measured['average_watts'], 'measurement': measured,
           'idle': idle, 'energy_meter': meter_info, 'model': model.metadata(),
           'outputs': outputs, 'before': before, 'after': telemetry(),
           'peak_rss_bytes': usage_after.ru_maxrss * 1024,
           'host_cpu_seconds': host_cpu, 'worker_cpu_seconds': own_cpu,
           'estimated_other_busy_cores': max(0, (host_cpu - own_cpu) / elapsed),
           'qualified_energy': config['kind'] == 'run' and energy is not None
                               and config['seconds'] >= 60 and clip_count == 16}
    if idle and idle['average_watts'] is not None and energy is not None:
        row['incremental_joules_per_clip'] = (
            energy - idle['average_watts'] * measured['elapsed_seconds']) / len(latencies)
    else:
        row['incremental_joules_per_clip'] = None
    save(job / f"{spec['id']}.json", row)
    print(json.dumps({k: row[k] for k in ('id', 'median_ms', 'p95_ms', 'joules_per_clip',
                                         'average_watts', 'estimated_other_busy_cores')}), flush=True)


def micro_worker(job, config, spec):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from native import PackedWeight
    rows = []
    generator = torch.Generator().manual_seed(20260914)
    # M=1 includes decoder projections and tied vocabulary; M=1500 is encoder.
    shapes = [(1, 1024, 1024), (1, 4096, 1024), (1, 1024, 4096),
              (1, 51864, 1024), (1500, 1024, 1024), (1500, 4096, 1024)]
    for family in config['families']:
        bits = 2 if family == 'w2a2' else 1
        for m, n, k in shapes:
            codes = (torch.randint(0, 2, (n, k), dtype=torch.int8, generator=generator) * 2 - 1
                     if family == 'w1a1' else
                     torch.randint(-1, 2, (n, k), dtype=torch.int8, generator=generator))
            x = torch.randn((m, k), generator=generator)
            bias = torch.randn(n, generator=generator) * .01
            scale = k ** -.5
            dense_codes = codes.float()
            def dense():
                q = (torch.where(x >= 0, 1., -1.) if bits == 1 else
                     torch.where(x >= .5, 1., torch.where(x <= -.5, -1., 0.)))
                return F.linear(q, dense_codes) * scale + bias
            reference = dense()
            for backend in ('dense', 'scalar', 'avx512', 'avx512_opt'):
                packed = None if backend == 'dense' else PackedWeight(
                    codes, scale, activation_bits=bits, backend=backend, threads=spec['threads'])
                call = dense if packed is None else lambda: packed.linear(x, bias)
                torch.testing.assert_close(call(), reference, rtol=0, atol=0)
                for _ in range(3):
                    call()
                times = []
                start = time.perf_counter()
                while len(times) < 10 or time.perf_counter() - start < config['seconds']:
                    t = time.perf_counter()
                    call()
                    times.append(time.perf_counter() - t)
                rows.append({**spec, 'family': family, 'backend': backend,
                             'm': m, 'n': n, 'k': k, 'samples': len(times),
                             'median_us': statistics.median(times) * 1e6,
                             'p95_us': float(np.percentile(times, 95)) * 1e6,
                             'latency_seconds': times, 'exact': True,
                             'weight_bytes': codes.numel() * 4 if packed is None else packed.storage_bytes})
            print(f"micro {spec['placement']} t={spec['threads']} {family} {m},{n},{k}", flush=True)
    save(job / f"{spec['id']}.json", {'kind': 'micro', 'spec': spec, 'rows': rows})


def verify_config(config):
    if 'package_versions' in config and package_versions() != config['package_versions']:
        raise RuntimeError('Runtime package versions changed since run was frozen')
    if 'runtime_environment' in config:
        actual = {key: os.environ.get(key) for key in RUNTIME_ENV}
        if actual != config['runtime_environment']:
            raise RuntimeError('Runtime backend environment changed since run was frozen')
    for path, sha in config['source_hashes'].items():
        if digest(path) != sha:
            raise RuntimeError(f'Source changed after run was frozen: {path}')
    if config['kind'] != 'micro':
        if digest(GPU_ROOT / 'data/manifest.json') != config['manifest_sha256']:
            raise RuntimeError('Workload manifest changed')
        if digest(GPU_ROOT / 'models/model-lock.json') != config['model_lock_sha256']:
            raise RuntimeError('Model lock changed')


def report(job):
    import numpy as np
    config = json.loads((job / 'config.json').read_text())
    scheduled = {spec['id']: spec for spec in config['schedule']}
    if len(scheduled) != len(config['schedule']):
        raise RuntimeError('Duplicate window IDs in frozen schedule')
    rows, seen = [], set()
    for path in sorted(job.glob('window-*.json')):
        row = json.loads(path.read_text())
        actual = row['spec'] if config['kind'] == 'micro' else row
        row_id = actual['id']
        if row_id in seen or row_id not in scheduled or path.stem != row_id:
            raise RuntimeError(f'Duplicate or unscheduled measurement window: {path.name}')
        if any(actual.get(key) != value for key, value in scheduled[row_id].items()):
            raise RuntimeError(f'Measurement differs from frozen schedule: {row_id}')
        if row['kind'] != config['kind']:
            raise RuntimeError(f'Measurement kind differs from frozen schedule: {row_id}')
        seen.add(row_id)
        rows.append(row)
    if config['kind'] == 'micro':
        save(job / 'summary.json', [r for row in rows for r in row['rows']])
        return
    manifest = json.loads((job / 'workload.json').read_text())
    expected_clips = {clip['id'] for clip in manifest['clips'][:config['clips']]}
    if len(expected_clips) != config['clips']:
        raise RuntimeError('Frozen workload contains duplicate or missing clip IDs')
    for row in rows:
        if set(row['outputs']) != expected_clips:
            raise RuntimeError(f"Missing or unexpected output evidence: {row['id']}")
        if (not row['clip_latency_seconds']
                or len(row['clip_latency_seconds']) != row['completed_clips']
                or row['completed_clips'] != config['clips'] * row['measurement']['iterations']):
            raise RuntimeError(f"Incomplete measurement cycle: {row['id']}")
        if row['variant'].startswith('ct2-') and any(
                evidence != {'forced_prefix_verified': True} for evidence in row['outputs'].values()):
            raise RuntimeError(f"Unverified CTranslate2 forced replay: {row['id']}")
        if not row['variant'].startswith('ct2-') and any(
                set(evidence) != {'logits', 'predictions'}
                or any(not isinstance(value, str) or len(value) != 64
                       or any(c not in '0123456789abcdef' for c in value)
                       for value in evidence.values()) for evidence in row['outputs'].values()):
            raise RuntimeError(f"Invalid custom model output hashes: {row['id']}")
    groups = {}
    planned_groups = {}
    for spec in config['schedule']:
        key = f"{spec['variant']}@{spec['placement']}:{spec['threads']}"
        planned_groups.setdefault(key, []).append(spec)
    for row in rows:
        key = f"{row['variant']}@{row['placement']}:{row['threads']}"
        groups.setdefault(key, []).append(row)
    result = {}
    reference_backend = config.get('reference_backend', 'dense')
    if reference_backend not in ('dense', 'scalar', 'avx512'):
        raise RuntimeError(f'Invalid reference backend: {reference_backend}')
    reference = {}
    for row in rows:
        if row['variant'].endswith('-' + reference_backend):
            family = row['variant'].split('-')[0]
            for clip, hashes in row['outputs'].items():
                key = (family, clip)
                if key in reference and reference[key] != hashes:
                    raise RuntimeError(f'{reference_backend.capitalize()} reference differs across thread placements: {key}')
                reference[key] = hashes
    lines = ['# CPU inference measurements', '',
             f"Run type: **{config['kind']}**. CPU: {config['topology'].get('model_name', 'see config.json')}.", '',
             'Same frozen 30-second clips and 128 forced decoder steps as the RTX 5090 experiment.',
             'Custom models have seeded random weights and quantized projection activations; no recognition accuracy claim.',
             'CPU FP32 attention/norm/residual arithmetic differs from the GPU FP16 runtime.',
             'CTranslate2 uses pretrained weights in a separate engine; it is an external throughput control.', '',
             f'Same-model arithmetic reference: **{reference_backend}**.', '',
             f'| Variant / placement / threads | Windows | Median ms | p95 ms | RTF | CPU J/clip | CPU W | Exact {reference_backend} match |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for key, selected in groups.items():
        times = [v for r in selected for v in r['clip_latency_seconds']]
        energies = [r['joules_per_clip'] for r in selected if r['joules_per_clip'] is not None]
        watts = [r['average_watts'] for r in selected if r['average_watts'] is not None]
        family = selected[0]['variant'].split('-')[0]
        checks = [(reference.get((family, clip)), hashes)
                  for row in selected for clip, hashes in row['outputs'].items()]
        exact = None if family.startswith('ct2') or not all(a is not None for a, b in checks) else all(a == b for a, b in checks)
        if exact is False:
            raise RuntimeError(f'Full-model output mismatch for {key}')
        complete_group = {r['id'] for r in selected} == {s['id'] for s in planned_groups[key]}
        complete_repeats = {r['repeat'] for r in selected} == set(range(config['repeats']))
        qualified = (config['kind'] == 'run' and config['repeats'] >= 3
                     and config['clips'] == 16 and config['seconds'] >= 60
                     and complete_group and complete_repeats
                     and (family == 'ct2' or exact is True)
                     and all(r['qualified_energy'] and r['measurement']['elapsed_seconds'] >= 60
                             and r['joules_per_clip'] is not None
                             and r['measurement']['energy_joules'] is not None for r in selected))
        r = {'windows': len(selected), 'median_ms': statistics.median(times) * 1000,
             'p95_ms': float(np.percentile(times, 95)) * 1000,
             'real_time_factor': sum(r['measurement']['elapsed_seconds'] for r in selected) / (30 * len(times)),
             'joules_per_clip': statistics.mean(energies) if energies else None,
             'average_watts': statistics.mean(watts) if watts else None,
             'energy_window_range': [min(energies), max(energies)] if energies else None,
             'reference_backend': reference_backend, 'exact_reference_match': exact,
             'exact_dense_match': exact if reference_backend == 'dense' else None,
             'completed_clips': len(times),
             'peak_rss_bytes': max(r['peak_rss_bytes'] for r in selected),
             'max_estimated_other_busy_cores': max(r['estimated_other_busy_cores'] for r in selected),
             'planned_windows': len(planned_groups[key]), 'complete_group': complete_group,
             'qualified_energy': qualified}
        result[key] = r
        j = 'unavailable' if r['joules_per_clip'] is None else f"{r['joules_per_clip']:.3f}"
        w = 'unavailable' if r['average_watts'] is None else f"{r['average_watts']:.2f}"
        lines.append(f"| {key} | {len(selected)} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['real_time_factor']:.4f} | {j} | {w} | {exact} |")
    lines += ['', 'Each worker runs in a fresh process with explicit physical-core affinity, one inter-op thread, and passive OpenMP waiting.',
              'Timing includes the CPU audio frontend, activation packing, encoder, growing KV cache, 128 vocabulary projections and reductions.',
              'Construction, weight packing, compilation and initial warmup are excluded. Weights and runtime memory remain resident.',
              'RAPL energy, when readable, covers the entire CPU package, including background tasks; it excludes GPU and whole-machine power.',
              'Unavailable energy stays null; TDP and utilization are never converted to watts.',
              f"Requested minimum per window: {config['seconds']} seconds, {config['clips']} clips per full cycle; repeats: {config['repeats']}.",
              'Screen results are tuning evidence, not sustained energy results. Raw times, output hashes, CPU telemetry and source snapshots accompany this report.']
    save(job / 'summary.json', result)
    (job / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def placements(topology, requests):
    presets = topology['affinity_presets']
    result = []
    for request in requests:
        name, count = request.rsplit(':', 1)
        threads = int(count)
        cpus = presets[name]
        if not 1 <= threads <= len(cpus):
            raise ValueError(f'Invalid physical thread count: {request}')
        result.append({'placement': name, 'threads': threads, 'cpus': cpus[:threads]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'micro', 'screen', 'run', 'report', 'worker'])
    parser.add_argument('--job', type=Path)
    parser.add_argument('--id')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS,
                        default=['w1a1-dense', 'w1a1-scalar', 'w1a1-avx512', 'w1a1-avx512_opt',
                                 'w2a2-dense', 'w2a2-scalar', 'w2a2-avx512', 'w2a2-avx512_opt',
                                 'ct2-fp32', 'ct2-int8'])
    parser.add_argument('--reference-backend', choices=['dense', 'scalar', 'avx512'], default='dense',
                        help='Same-model output reference required for custom run qualification')
    parser.add_argument('--placements', nargs='+', default=['largest_l3_physical:4', 'largest_l3_physical:8',
                                                         'smallest_l3_physical:8', 'all_physical:16'])
    parser.add_argument('--families', nargs='+', choices=['w1a1', 'w2a1', 'w2a2'], default=['w1a1', 'w2a1', 'w2a2'])
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--clips', type=int)
    parser.add_argument('--cycles', type=int)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--idle-seconds', type=float, default=2)
    args = parser.parse_args()
    storage()
    if args.command == 'worker':
        job = artifact(args.job)
        cfg = json.loads((job / 'config.json').read_text())
        verify_config(cfg)
        spec = next(s for s in cfg['schedule'] if s['id'] == args.id)
        configure(spec['threads'], spec['cpus'])
        (micro_worker if cfg['kind'] == 'micro' else model_worker)(job, cfg, spec)
        return
    if args.command == 'report':
        report(artifact(args.job))
        return
    from hardware import CpuEnergyMeter, discover_topology
    topology = discover_topology()
    if args.command == 'inspect':
        value = {'topology': topology, 'energy': energy_metadata()}
        save(ROOT / 'hardware.json', value)
        print(json.dumps(value, indent=2))
        return
    seconds = args.seconds if args.seconds is not None else {'micro': .15, 'screen': 0, 'run': 60}[args.command]
    repeats = args.repeats if args.repeats is not None else (3 if args.command == 'run' else 1)
    clips = args.clips if args.clips is not None else (16 if args.command == 'run' else 1)
    cycles = args.cycles if args.cycles is not None else (3 if args.command == 'screen' else 1)
    if seconds < 0 or repeats < 1 or not 1 <= clips <= 16 or cycles < 1 or args.warmup < 1 or args.idle_seconds < 0:
        parser.error('Invalid measurement dimensions')
    if args.command == 'run' and (seconds < 60 or clips != 16 or repeats < 3):
        parser.error('Sustained run requires >=60 seconds, all 16 clips and >=3 windows; use screen for shorter tests')
    selected_placements = placements(topology, args.placements)
    job = artifact(args.job or ROOT / args.command / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    job.mkdir(parents=True, exist_ok=False)
    variants = [None] if args.command == 'micro' else args.variants
    schedule = []
    base = [{**p, 'variant': v} for p in selected_placements for v in variants]
    for repeat in range(repeats):
        # Alternate arm order to reduce fixed-order thermal drift.
        for spec in (base if repeat % 2 == 0 else list(reversed(base))):
            schedule.append({**spec, 'repeat': repeat, 'id': f'window-{len(schedule):03d}'})
    cfg = {'kind': args.command, 'seconds': seconds, 'repeats': repeats, 'clips': clips,
           'cycles': cycles, 'warmup': args.warmup, 'idle_seconds': args.idle_seconds,
           'families': args.families, 'schedule': schedule, 'topology': topology,
           'reference_backend': args.reference_backend,
           'source_hashes': source_hashes(), 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
           'energy_meter': energy_metadata(),
           'package_versions': package_versions(),
           'runtime_environment': {key: os.environ.get(key) for key in RUNTIME_ENV}}
    if args.command != 'micro':
        manifest = workload()
        verify_models()
        cfg.update(manifest_sha256=digest(GPU_ROOT / 'data/manifest.json'),
                   model_lock_sha256=digest(GPU_ROOT / 'models/model-lock.json'))
        save(job / 'workload.json', manifest)
    save(job / 'config.json', cfg)
    for path in cfg['source_hashes']:
        p = Path(path)
        destination = job / 'source' / p.parent.name / p.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, destination)
    print(f'Artifacts: {job}', flush=True)
    with (ROOT / 'active.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            for spec in schedule:
                save(job / 'status.json', {'status': 'running', 'spec': spec})
                subprocess.run([str(HERE / 'python'), str(HERE / 'benchmark.py'), 'worker',
                                '--job', str(job), '--id', spec['id']], check=True,
                               env={**os.environ, 'CPU_THREADS': str(spec['threads'])})
                report(job)
        except BaseException as exc:
            save(job / 'status.json', {'status': 'failed', 'error': repr(exc)})
            raise
    save(job / 'status.json', {'status': 'completed'})
    print(f'Completed: {job}', flush=True)


if __name__ == '__main__':
    main()
