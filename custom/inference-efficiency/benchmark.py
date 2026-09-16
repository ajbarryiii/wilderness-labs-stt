"""Reproducible fixed-work Whisper comparison. Full measurements require a free GPU."""
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from paths import ROOT, artifact, digest, save, storage

HERE = Path(__file__).resolve().parent
VARIANTS = ('ct2-fp16', 'ct2-int8', 'ternary-dense', 'ternary-packed', 'binary-dense', 'binary-packed')


def source_hashes():
    return {str(p.relative_to(HERE)): digest(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc',)}


def workload_manifest():
    manifest_path = ROOT / 'data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if len(manifest['clips']) != 16 or manifest['decoder_steps'] != 128:
        raise ValueError('Main experiment requires 16 clips and 128 sequential decoder steps')
    tokens = manifest['replay_tokens']
    if len(tokens) != 128 or any(c['replay_tokens'] != tokens for c in manifest['clips']):
        raise ValueError('All clips must share the frozen 128-token replay for CUDA graph reuse')
    for clip in manifest['clips']:
        if clip['seconds'] != 30:
            raise ValueError('Every clip must contain 30 seconds of audio')
        for field in ('pcm', 'mel'):
            path = artifact(clip[field])
            if digest(path) != clip[field + '_sha256']:
                raise ValueError(f'Workload hash mismatch: {path}')
    return manifest


def verify_models():
    lock = json.loads((ROOT / 'models/model-lock.json').read_text())
    for model in lock.values():
        if not model.get('files'):
            raise RuntimeError('Incomplete model download in lock')
        for filename, spec in model['files'].items():
            path = artifact(Path(model['path']) / filename)
            if path.stat().st_size != spec['bytes'] or digest(path) != spec['sha256']:
                raise RuntimeError(f'Model file no longer matches pinned hash: {path}')


def preflight():
    from energy import EnergyMeter
    storage()
    with EnergyMeter() as meter:
        result = dict(gpu=meter.metadata(), foreign_processes=meter.foreign_processes(),
                      runtime=json.loads((ROOT / 'runtime.json').read_text()),
                      model_lock=json.loads((ROOT / 'models/model-lock.json').read_text()),
                      workload_sha256=digest(ROOT / 'data/manifest.json'))
    workload_manifest()
    verify_models()
    save(ROOT / 'preflight.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def make_runner(variant, manifest, *, graph=True, seed=20260911):
    import numpy as np
    import torch
    from transformers import WhisperFeatureExtractor
    from runtime_model import ReplayWhisper, WhisperConfig
    from control import CTranslate2Control
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = True
    extractor = WhisperFeatureExtractor.from_pretrained(ROOT / 'models/whisper-medium.en', local_files_only=True)
    pcm = [np.load(c['pcm'], allow_pickle=False) for c in manifest['clips']]
    tokens = manifest['replay_tokens']
    example_mel = np.load(manifest['clips'][0]['mel'], allow_pickle=False)
    if variant.startswith('ct2-'):
        precision = 'float16' if variant == 'ct2-fp16' else 'int8_float16'
        model = CTranslate2Control(ROOT / 'models/ct2-medium.en', compute_type=precision)
        model.run(example_mel, tokens)
        def infer(mel):
            return model.run(mel, tokens)
    else:
        distribution, implementation = variant.split('-')
        model = ReplayWhisper(WhisperConfig.medium_en(), distribution=distribution,
                              implementation=implementation, seed=seed)
        mel_gpu = torch.from_numpy(example_mel).to('cuda', dtype=torch.float16)
        if graph:
            model.capture(mel_gpu, tokens)
            def infer(mel):
                # Host-to-device transfer and graph input copy are timed.
                return model.replay_graph(torch.from_numpy(mel).to('cuda', dtype=torch.float16))
        else:
            def infer(mel):
                return model.run(torch.from_numpy(mel).to('cuda', dtype=torch.float16), tokens)
    cursor = 0
    def call():
        nonlocal cursor
        audio = pcm[cursor % len(pcm)]
        cursor += 1
        mel = extractor(audio, sampling_rate=16000, return_tensors='np').input_features
        return infer(mel)
    return model, call, infer, example_mel


def worker(args):
    import numpy as np
    import torch
    from energy import EnergyMeter
    config_path = artifact(args.run_dir / 'config.json')
    config = json.loads(config_path.read_text())
    if config['source_hashes'] != source_hashes():
        raise RuntimeError('Source changed since experiment configuration was frozen')
    if config['workload_sha256'] != digest(ROOT / 'data/manifest.json'):
        raise RuntimeError('Workload changed since run was frozen')
    for name, key in (('runtime.json', 'runtime_sha256'), ('models/model-lock.json', 'model_lock_sha256')):
        if config[key] != digest(ROOT / name):
            raise RuntimeError(f'Frozen artifact changed: {name}')
    verify_models()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '0'):
        raise RuntimeError('GPU visibility remapping is unsupported; use physical GPU 0')
    with EnergyMeter(synchronize=torch.cuda.synchronize) as meter:
        foreign = meter.foreign_processes()
        if foreign and not args.smoke:
            raise RuntimeError(f'GPU busy; clean measurement refused: {foreign}')
        manifest = workload_manifest()
        before = meter.metadata()
        if any(before[k] != config['gpu'][k] for k in ('uuid', 'power_limit_w')):
            raise RuntimeError('GPU identity or configured power limit changed since run was frozen')
        torch.cuda.set_device(0)
        model, call, infer, mel = make_runner(args.variant, manifest, graph=config['cuda_graph'], seed=config['seed'])
        result = call()
        torch.cuda.synchronize()
        if isinstance(result, torch.Tensor):
            if not bool(torch.isfinite(result).all()) or not float(result.float().std()) > 0:
                raise RuntimeError('Candidate outputs are not finite and nondegenerate')
        if args.smoke:
            save(args.run_dir / f'smoke-{args.variant}.json', dict(
                variant=args.variant, passed=True, model=model.metadata(),
                foreign_processes=foreign, qualified_energy=False,
                note='Correctness/launch check only; no energy estimate from this run.',
                peak_torch_reserved_bytes=torch.cuda.max_memory_reserved()))
            print(f'Smoke passed: {args.variant}', flush=True)
            return
        idle = meter.idle(config['idle_seconds'])
        # Warm after idle so clocks/temperature are not reset immediately before timing.
        for _ in manifest['clips']:
            call()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        def cycle():
            # Complete whole sets: every variant sees exactly the same clip mix.
            for _ in manifest['clips']:
                start = time.perf_counter()
                call()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
        measured = meter.measure(cycle, min_seconds=config['seconds'], min_iterations=1)
        count = measured['iterations'] * len(manifest['clips'])
        after = meter.metadata()
        if any(after[k] != before[k] for k in ('uuid', 'power_limit_w')):
            raise RuntimeError('GPU identity or power limit changed during measurement')
        energy = measured['energy_joules']
        duration = measured['elapsed_seconds']
        row = dict(variant=args.variant, repeat=args.repeat, model=model.metadata(),
                   audio_seconds=30 * count, gpu_j_per_audio_second=energy / (30 * count),
                   avg_watts=energy / duration, p50_seconds=float(np.percentile(times, 50)),
                   p95_seconds=float(np.percentile(times, 95)), real_time_factor=duration / (30 * count),
                   incremental_j_per_audio_second=(energy-idle['average_watts']*duration)/(30*count),
                   peak_torch_allocated_bytes=torch.cuda.max_memory_allocated() if not args.variant.startswith('ct2') else None,
                   peak_torch_reserved_bytes=torch.cuda.max_memory_reserved() if not args.variant.startswith('ct2') else None,
                   completed_clips=count, clip_latency_seconds=times,
                   gpu_before=before, gpu_after=after,
                   sampled_peak_gpu_memory_bytes=measured['max_sampled_gpu_memory_bytes'],
                   sampled_memory_scope=measured['gpu_memory_scope'],
                   idle=idle, measurement=measured)
        save(args.run_dir / f'{args.repeat:02d}-{args.variant}.json', row)
        print(json.dumps({k: row[k] for k in ('variant', 'repeat', 'gpu_j_per_audio_second', 'p95_seconds')}), flush=True)


def report(run_dir):
    config = json.loads((run_dir / 'config.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted(run_dir.glob('[0-9][0-9]-*.json'))]
    summary = {}
    for variant in config['variants']:
        group = [r for r in rows if r['variant'] == variant]
        if group:
            energies = [r['gpu_j_per_audio_second'] for r in group]
            all_times = [t for r in group for t in r['clip_latency_seconds']]
            sorted_times = sorted(all_times)
            position = .95 * (len(sorted_times) - 1)
            lo = int(position)
            p95 = sorted_times[lo] + (position-lo)*(sorted_times[min(lo+1,len(sorted_times)-1)]-sorted_times[lo])
            summary[variant] = dict(windows=len(group), gpu_j_per_audio_second=statistics.mean(energies),
                                    minimum=min(energies), maximum=max(energies),
                                    p95_seconds=p95, window_p95_seconds=[r['p95_seconds'] for r in group],
                                    avg_watts=statistics.mean(r['avg_watts'] for r in group))
    complete = all(summary.get(v, {}).get('windows') == config['repeats'] for v in VARIANTS)
    qualified = (complete and config['repeats'] >= 5 and config['seconds'] >= 60
                 and all(r['measurement']['elapsed_seconds'] >= 60 for r in rows)
                 and all(r['completed_clips'] >= 16 and r['completed_clips'] % 16 == 0 for r in rows))
    decision = 'incomplete'
    comparison = {}
    if all(v in summary for v in ('ct2-fp16', 'ct2-int8', 'ternary-packed')):
        control = min(('ct2-fp16', 'ct2-int8'), key=lambda v: summary[v]['gpu_j_per_audio_second'])
        baseline, candidate = summary[control], summary['ternary-packed']
        comparison = dict(control=control,
                          ternary_energy_ratio=candidate['gpu_j_per_audio_second']/baseline['gpu_j_per_audio_second'],
                          conservative_energy_ratio=candidate['maximum']/baseline['minimum'],
                          latency_ratio=candidate['p95_seconds']/baseline['p95_seconds'])
        if 'ternary-dense' in summary:
            dense = summary['ternary-dense']
            comparison['same_runtime_energy_ratio'] = candidate['gpu_j_per_audio_second']/dense['gpu_j_per_audio_second']
            comparison['same_runtime_conservative_energy_ratio'] = candidate['maximum']/dense['minimum']
            comparison['same_runtime_latency_ratio'] = candidate['p95_seconds']/dense['p95_seconds']
        if qualified:
            external_pass = comparison['conservative_energy_ratio'] <= 0.5 and comparison['latency_ratio'] <= 1
            precision_pass = comparison.get('same_runtime_conservative_energy_ratio', 1) < 1
            decision = 'go' if external_pass and precision_pass else ('runtime_gain_only' if external_pass else 'no_go_for_this_implementation')
    result = dict(qualified=qualified, decision=decision, comparison=comparison, variants=summary,
                  scope='GPU board energy on fixed Whisper work; no accuracy or phone-energy conclusion.',
                  runtime_caveat='CTranslate2 controls; CUDA-graph PyTorch/SDPA candidates. Dense twins expose engine overhead.')
    save(run_dir / 'summary.json', result)
    lines = ['# Inference efficiency result', '', f'Decision: **{decision}**. Qualified: {qualified}.', '',
             '| Variant | Windows | GPU J/audio-second | Average W | Aggregate p95 seconds |',
             '| --- | ---: | ---: | ---: | ---: |']
    for v, r in summary.items():
        lines.append(f"| {v} | {r['windows']} | {r['gpu_j_per_audio_second']:.5f} | {r['avg_watts']:.2f} | {r['p95_seconds']:.4f} |")
    lines.extend(['', result['scope'], '', result['runtime_caveat'], '', json.dumps(comparison, indent=2)])
    artifact(run_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'run', 'smoke', 'worker', 'report'))
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--idle-seconds', type=float, default=5)
    parser.add_argument('--no-cuda-graph', action='store_true')
    parser.add_argument('--smoke', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    storage()
    if args.command == 'preflight':
        preflight()
        return
    if args.command == 'worker':
        worker(args)
        return
    if args.command == 'report':
        report(artifact(args.run_dir))
        return
    if args.repeats < 1 or args.seconds <= 0 or args.idle_seconds <= 0:
        parser.error('Repeats and measurement/idle durations must be positive')
    from energy import EnergyMeter
    with EnergyMeter() as meter:
        foreign = meter.foreign_processes()
        gpu = meter.metadata()
    if args.command == 'run' and foreign:
        raise RuntimeError(f'GPU busy; measurement refused without changing the training process: {foreign}')
    manifest = workload_manifest()
    verify_models()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_dir = artifact(args.run_dir or ROOT / 'runs' / f'{args.command}-{stamp}')
    if (run_dir / 'config.json').exists():
        raise FileExistsError('Use a new run directory; frozen experiment outputs are never overwritten')
    config = dict(schema=1, seed=20260911, created_utc=stamp, gpu=gpu,
                  variants=args.variants, repeats=args.repeats, seconds=args.seconds,
                  idle_seconds=args.idle_seconds, cuda_graph=not args.no_cuda_graph,
                  gate=dict(maximum_energy_ratio=0.5, maximum_latency_ratio=1.0),
                  source_hashes=source_hashes(), workload_sha256=digest(ROOT / 'data/manifest.json'),
                  runtime_sha256=digest(ROOT / 'runtime.json'),
                  model_lock_sha256=digest(ROOT / 'models/model-lock.json'))
    save(run_dir / 'config.json', config)
    save(run_dir / 'workload.json', manifest)
    for repeat in range(1 if args.command == 'smoke' else args.repeats):
        order = args.variants[repeat % len(args.variants):] + args.variants[:repeat % len(args.variants)]
        for variant in order:
            command = [str(HERE / 'python'), str(HERE / 'benchmark.py'), 'worker',
                       '--run-dir', str(run_dir), '--variant', variant, '--repeat', str(repeat)]
            if args.command == 'smoke':
                command.append('--smoke')
            subprocess.run(command, check=True, env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    if args.command == 'run':
        report(run_dir)
    print(f'Artifacts: {run_dir}', flush=True)


if __name__ == '__main__':
    main()
