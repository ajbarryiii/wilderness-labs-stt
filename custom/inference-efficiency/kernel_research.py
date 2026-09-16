"""One-hour, isolated, adaptive keep/discard search over decoder kernel mutations.

./python kernel_research.py setup
Setup validates readiness, then starts a systemd user service with a hard 3600s
limit. All data, compiled candidates, logs, and winning policies live on /mnt/hd.
"""
import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

from paths import ROOT, artifact, digest, save, storage

HERE = Path(__file__).resolve().parent
LOCK = Path('/mnt/hd/wilderness-labs-stt/stt-distillation/active.lock')
DISTRIBUTIONS = ('ternary', 'binary')
HOUR = 3600
SEARCH_SECONDS = 2280
SOFT_SECONDS = 3540
WORKER_TIMEOUT = 180


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def hashes(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def key(policy):
    return json.dumps(policy, sort_keys=True)


def ratios(candidate, incumbent):
    return (candidate['gpu_j_per_audio_second'] / incumbent['gpu_j_per_audio_second'],
            candidate['p95_seconds'] / incumbent['p95_seconds'])


def improves(candidate, incumbent, energy_ratio=.99):
    energy, latency = ratios(candidate, incumbent)
    return math.isfinite(energy) and math.isfinite(latency) and energy <= energy_ratio and latency <= 1.02


def proposal(policy, seen, profile):
    """Rebuild the neighborhood around the current energy winner after each keep."""
    from kernels.research import AXES, BASE, CALLS, METHODS
    candidates = []
    for axis in AXES:
        for method in METHODS[axis]:
            child = {**policy, axis: method}
            candidates.append((child, f'Change {axis} from {policy[axis]} to {method}'))
    # Try complementary changes together, then test individual neighbors to ablate them.
    fastest = {a: min(profile[a], key=lambda m: profile[a][m]['median_us']) for a in AXES}
    candidates.append((fastest, 'Combine the fastest measured implementation for each decoder shape'))
    for method in ('acc2', 'acc4', 'split4', 'warp32', 'warp64'):
        child = {**policy, **{a: method for a in AXES if a != 'vocab'}}
        candidates.append((child, f'Apply {method} to all internal decoder projections'))
    unique = {key(p): (p, why) for p, why in candidates if key(p) not in seen and p != policy}
    if not unique:
        return None
    def estimated(p):
        return sum(CALLS[a] * profile[a][p[a]]['median_us'] for a in AXES)
    p, why = min(unique.values(), key=lambda item: estimated(item[0]))
    return p, why, estimated(p) / estimated(BASE)


def aggregate(rows):
    return {field: statistics.mean(r[field] for r in rows)
            for field in ('gpu_j_per_audio_second', 'p95_seconds', 'avg_watts')}


def confirmation(rows):
    """Independent alternating windows, with conservative extrema against noise."""
    old = [r['result'] for r in rows if r['role'] == 'baseline']
    new = [r['result'] for r in rows if r['role'] == 'winner']
    if len(old) < 3 or len(new) < 3:
        return {'confirmed': False, 'reason': 'Fewer than three fresh windows per arm',
                'baseline_windows': len(old), 'winner_windows': len(new)}
    a, b = aggregate(new), aggregate(old)
    energy, latency = ratios(a, b)
    conservative = max(r['gpu_j_per_audio_second'] for r in new) / min(r['gpu_j_per_audio_second'] for r in old)
    return {'confirmed': improves(a, b, .98) and conservative < 1,
            'baseline_windows': len(old), 'winner_windows': len(new),
            'energy_ratio': energy, 'p95_ratio': latency,
            'conservative_energy_ratio': conservative, 'baseline': b, 'winner': a,
            'rule': 'Mean GPU energy at least 2% lower; worst winner below best baseline; p95 at most +2%'}


def verify_ready(job):
    cfg = read(job / 'config.json')
    if hashes(job / 'code') != cfg['source_hashes']:
        raise RuntimeError('Frozen source changed')
    for name, expected in read(job / 'ready.json')['hashes'].items():
        if digest(job / name) != expected:
            raise RuntimeError(f'Readiness artifact changed: {name}')
    return cfg


def run_child(job, command, log, timeout):
    env = {**os.environ, 'EFFICIENCY_REQUIRE_CUDA_GEMV': '1'}
    started = time.monotonic()
    with (job / log).open('w') as output:
        proc = subprocess.Popen([str(job / 'code/python'), *command], cwd=job / 'code',
                                stdout=output, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except BaseException:
            # Killing the worker's process group also cleans up any compiler children.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise
    if rc:
        raise RuntimeError(f'Worker exited {rc}; see {job / log}')
    return time.monotonic() - started


def worker(args):
    import numpy as np
    import torch
    import benchmark
    from energy import EnergyMeter
    from kernels import research
    job = artifact(args.job)
    cfg = read(job / 'config.json')
    if hashes(job / 'code') != cfg['source_hashes']:
        raise RuntimeError('Frozen source changed')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    with EnergyMeter() as meter:
        if meter.foreign_processes():
            raise RuntimeError('Foreign GPU process before research worker')
        gpu = meter.metadata()
        if any(gpu[k] != cfg['gpu'][k] for k in ('uuid', 'power_limit_w', 'compute_capability')):
            raise RuntimeError('GPU configuration changed')
    if args.mode == 'kernels':
        research.compile_modules(job / 'modules')
        save(job / 'kernel-correctness.json', research.check_all(job / 'modules'))
        return
    if args.mode == 'profile':
        verify_ready(job)
        save(job / 'profile.json', research.profile(job / 'modules'))
        return
    benchmark.verify_models()
    manifest = benchmark.workload_manifest()
    if args.mode == 'reference':
        model, _, infer, _ = benchmark.make_runner(args.distribution + '-packed', manifest)
        arrays = {}
        for i, clip in enumerate(manifest['clips']):
            arrays[f'logits_{i}'] = infer(np.load(clip['mel'], allow_pickle=False)).cpu().numpy().copy()
            arrays[f'predictions_{i}'] = model.last_predictions.cpu().numpy().copy()
        np.savez(job / f'reference-{args.distribution}.npz', **arrays)
        save(job / f'reference-{args.distribution}.json', model.metadata())
        return
    if args.mode != 'smoke':
        verify_ready(job)
    run_dir = artifact(args.run_dir)
    policy = research.validate_policy(read(run_dir / 'config.json')['kernel_policy'])
    research.install(policy, job / 'modules')
    original_make = benchmark.make_runner

    def checked_runner(*a, **kw):
        model, call, infer, mel = original_make(*a, **kw)
        checks = []
        with np.load(job / f'reference-{args.distribution}.npz', allow_pickle=False) as reference:
            for i, clip in enumerate(manifest['clips']):
                actual = infer(np.load(clip['mel'], allow_pickle=False)).cpu().float().numpy()
                expected = reference[f'logits_{i}'].astype(np.float32)
                delta = actual - expected
                nrms = float(np.sqrt(np.mean(delta ** 2)) / max(np.sqrt(np.mean(expected ** 2)), 1e-8))
                nmax = float(np.max(np.abs(delta)) / max(np.max(np.abs(expected)), 1e-8))
                agreement = float(np.mean(model.last_predictions.cpu().numpy() == reference[f'predictions_{i}']))
                if not np.isfinite(actual).all() or nrms > .005 or nmax > .02:
                    raise RuntimeError(f'Full-model numerical regression on clip {i}: NRMS={nrms}, NMAX={nmax}')
                checks.append(dict(clip=i, normalized_rms_error=nrms,
                                   normalized_max_error=nmax, prediction_agreement=agreement))
        save(run_dir / 'correctness.json', {'passed': True, 'clips': checks,
             'reference': 'Frozen original packed kernels, identical seed/weights and forced tokens',
             'limits': {'normalized_rms_error': .005, 'normalized_max_error': .02}})
        original_metadata = model.metadata
        def metadata():
            row = original_metadata()
            row['kernel_research_policy'] = policy
            if policy != research.BASE:
                row['decode_backend'] = 'kernel-research-policy (see kernel_research_policy)'
            return row
        model.metadata = metadata
        return model, call, infer, mel

    benchmark.make_runner = checked_runner
    benchmark.worker(SimpleNamespace(run_dir=run_dir, variant=args.distribution + '-packed',
                                     repeat=0, smoke=args.mode == 'smoke'))


def setup():
    from energy import EnergyMeter
    import benchmark
    from kernels.research import BASE, METHODS
    storage()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job = artifact(ROOT / 'kernel-research' / stamp)
    job.mkdir(parents=True)
    with LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with EnergyMeter() as meter:
            if meter.foreign_processes():
                raise RuntimeError('GPU busy; readiness checks refused')
            gpu = meter.metadata()
        if gpu['compute_capability'] != [12, 0] or gpu['power_limit_w'] != 400:
            raise RuntimeError('This experiment is frozen to SM120 at the existing 400 W limit')
        benchmark.workload_manifest()
        benchmark.verify_models()
        shutil.copytree(HERE, job / 'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (job / 'autoresearch').mkdir()
        (job / 'autoresearch/.training-runtime').symlink_to((HERE.parent / 'autoresearch/.training-runtime').resolve())
        cfg = dict(schema=1, created_utc=utc(), source_hashes=hashes(job / 'code'), gpu=gpu,
                   cuda_graph=True, seed=20260911, seconds=60, idle_seconds=5,
                   workload_sha256=digest(ROOT / 'data/manifest.json'),
                   runtime_sha256=digest(ROOT / 'runtime.json'),
                   model_lock_sha256=digest(ROOT / 'models/model-lock.json'),
                   budget_seconds=HOUR, search_seconds=SEARCH_SECONDS, soft_seconds=SOFT_SECONDS,
                   methods=METHODS, shared_lock=str(LOCK),
                   objective='GPU board joules per fixed 30-second clip; p95 latency at most +2%',
                   search_keep_energy_ratio=.99, confirmation_energy_ratio=.98,
                   confirmation_windows_per_arm=3,
                   prior_control_run=str(ROOT / 'runs/cuda-gemv-20260912T054024Z'),
                   scope='Bounded adaptive CUDA source/launch/dispatch search; no model or evaluator edits')
        save(job / 'config.json', cfg)
        print(f'Preparing {job}', flush=True)
        common = [str(job / 'code/kernel_research.py'), 'worker', '--job', str(job)]
        run_child(job, common + ['--mode', 'kernels'], 'preflight-kernels.log', 600)
        run_child(job, ['-m', 'kernels.verify_cuda'], 'preflight-original-cuda.log', 180)
        for distribution in DISTRIBUTIONS:
            print(f'Preparing {distribution} full-model reference', flush=True)
            run_child(job, common + ['--mode', 'reference', '--distribution', distribution],
                      f'preflight-reference-{distribution}.log', 300)
            smoke = job / f'smoke-{distribution}'
            smoke.mkdir()
            policy = {a: 'acc2' if a != 'vocab' else 'transposed' for a in BASE}
            save(smoke / 'config.json', {**cfg, 'kernel_policy': policy})
            run_child(job, common + ['--mode', 'smoke', '--distribution', distribution,
                                    '--run-dir', str(smoke)], f'preflight-smoke-{distribution}.log', 300)
        frozen = [job / 'config.json', job / 'kernel-correctness.json']
        frozen += list((job / 'modules').iterdir()) + list(job.glob('reference-*'))
        save(job / 'ready.json', {'passed': True, 'completed_utc': utc(),
                                 'hashes': {str(p.relative_to(job)): digest(p) for p in frozen}})
    unit = 'stt-kernel-research-' + stamp.lower()
    save(job / 'state.json', dict(status='ready', job=str(job), unit=unit, updated_utc=utc()))
    subprocess.run(['systemd-run', '--user', '--unit', unit,
                    '--property=RuntimeMaxSec=3600', '--property=TimeoutStopSec=5',
                    '--property=KillMode=control-group', '--property=Restart=no',
                    f'--property=WorkingDirectory={job / "code"}',
                    f'--property=StandardOutput=append:{job / "controller.log"}',
                    f'--property=StandardError=append:{job / "controller.log"}',
                    str(job / 'code/python'), str(job / 'code/kernel_research.py'),
                    'run', '--job', str(job)], check=True)
    save(ROOT / 'kernel-research/latest.json', dict(job=str(job), unit=unit, started_utc=utc()))
    print(json.dumps({'job': str(job), 'unit': unit, 'hard_limit_seconds': HOUR}), flush=True)


def run(job):
    from energy import EnergyMeter
    from kernels.research import BASE
    started = time.monotonic()
    cfg = verify_ready(job)
    if (job / 'ledger.jsonl').exists():
        raise RuntimeError('Refusing to restart a spent one-hour budget; create a new run')
    ledger, incumbents, seen, confirmation_rows = [], {}, {}, {d: [] for d in DISTRIBUTIONS}
    state = read(job / 'state.json')
    state.update(started_utc=utc(), deadline_utc=(dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=HOUR)).isoformat())
    number = 0

    def update(status, **details):
        state.update(status=status, elapsed_seconds=time.monotonic() - started, updated_utc=utc(), **details)
        save(job / 'state.json', state)
        print(json.dumps(state), flush=True)

    def measure(distribution, policy, role, reason):
        nonlocal number
        number += 1
        trial = job / 'trials' / f'{number:03d}-{distribution}-{role}'
        trial.mkdir(parents=True)
        save(trial / 'config.json', {**cfg, 'kernel_policy': policy})
        row = dict(number=number, distribution=distribution, policy=policy, role=role,
                   reason=reason, trial=str(trial), started_utc=utc())
        update('running', stage=role, distribution=distribution, trial=number, policy=policy)
        try:
            remaining = SOFT_SECONDS - (time.monotonic() - started)
            if remaining < WORKER_TIMEOUT + 5:
                raise TimeoutError('Insufficient budget to admit another worker')
            command = [str(job / 'code/kernel_research.py'), 'worker', '--job', str(job),
                       '--mode', 'measure', '--distribution', distribution, '--run-dir', str(trial)]
            row['wall_seconds'] = run_child(job, command, str(trial.relative_to(job) / 'worker.log'), WORKER_TIMEOUT)
            result = read(trial / f'00-{distribution}-packed.json')
            if result['measurement']['elapsed_seconds'] < 60 or result['completed_clips'] % 16:
                raise RuntimeError('Incomplete energy window')
            row['result'] = {k: result[k] for k in ('gpu_j_per_audio_second', 'p95_seconds', 'avg_watts')}
            row['status'] = 'measured'
        except Exception as exc:
            row.update(status='rejected-error', error=f'{type(exc).__name__}: {exc}')
        row['finished_utc'] = utc()
        return row

    def record(row):
        ledger.append(row)
        with (job / 'ledger.jsonl').open('a') as output:
            output.write(json.dumps(row, allow_nan=False) + '\n')
            output.flush()
            os.fsync(output.fileno())
        save(job / 'incumbents.json', incumbents)
        print(json.dumps(row), flush=True)

    def report(status, error=None):
        results = {d: confirmation(rows) for d, rows in confirmation_rows.items()}
        summary = dict(status=status, elapsed_seconds=time.monotonic() - started, error=error,
                       trials=len(ledger), incumbents=incumbents, confirmation=results,
                       qualified_external_control_comparison=False,
                       scope='Kernel optimization against original packed runtime. Prior CT2/dense controls are historical; no new accuracy or external-control gate claim.')
        save(job / 'summary.json', summary)
        lines = ['# One-hour kernel research', '', f'Status: {status}. Recorded trials: {len(ledger)}.', '',
                 '| Model | Confirmed | Energy change vs original packed | p95 change |',
                 '| --- | --- | ---: | ---: |']
        for d, result in results.items():
            energy = f"{100 * (result['energy_ratio'] - 1):+.2f}%" if 'energy_ratio' in result else 'pending'
            latency = f"{100 * (result['p95_ratio'] - 1):+.2f}%" if 'p95_ratio' in result else 'pending'
            lines.append(f"| {d} | {result['confirmed']} | {energy} | {latency} |")
            if result['confirmed']:
                best = job / 'best' / d
                best.mkdir(parents=True, exist_ok=True)
                save(best / 'policy.json', incumbents[d]['policy'])
                save(best / 'confirmation.json', result)
                (best / 'REPRODUCE.txt').write_text(
                    f'Frozen implementation: {job / "code/kernels/research.py"}\n'
                    f'CUDA sources and cubins: {job / "modules"}\n'
                    f'Install policy before constructing/capturing the model with:\n'
                    f'  kernels.research.install(policy, Path({str(job / "modules")!r}))\n')
        lines.extend(['', summary['scope'], '',
                      'Search keeps are provisional (>=1% energy gain, p95 <=+2%). Confirmation requires three fresh windows per arm, >=2% mean energy gain, and separated energy extrema.',
                      '', 'Source mutations and policies are exported under this job; the working kernels are not automatically replaced.'])
        (job / 'REPORT.md').write_text('\n'.join(lines) + '\n')

    def stop(signum, frame):
        raise InterruptedError(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with EnergyMeter() as meter:
                if meter.foreign_processes():
                    raise RuntimeError('GPU busy at timed start')
            update('running', stage='profiling')
            run_child(job, [str(job / 'code/kernel_research.py'), 'worker', '--job', str(job),
                            '--mode', 'profile'], 'profile.log', WORKER_TIMEOUT)
            profiles = read(job / 'profile.json')
            for d in DISTRIBUTIONS:
                row = measure(d, BASE.copy(), 'baseline', 'Fresh baseline for the unchanged packed kernels')
                if 'result' not in row:
                    record(row)
                    raise RuntimeError('Baseline failed; search has no valid objective')
                incumbents[d] = row
                seen[d] = {key(BASE)}
                record(row)
            turn = 0
            while time.monotonic() - started < SEARCH_SECONDS - WORKER_TIMEOUT:
                d = DISTRIBUTIONS[turn % len(DISTRIBUTIONS)]
                turn += 1
                suggested = proposal(incumbents[d]['policy'], seen[d], profiles['1' if d == 'binary' else '2'])
                if suggested is None:
                    break
                policy, reason, estimate = suggested
                seen[d].add(key(policy))
                row = measure(d, policy, 'search', reason)
                row.update(parent_trial=incumbents[d]['number'], estimated_kernel_latency_ratio=estimate)
                if 'result' in row:
                    row['energy_ratio_to_incumbent'], row['p95_ratio_to_incumbent'] = ratios(row['result'], incumbents[d]['result'])
                    if improves(row['result'], incumbents[d]['result']):
                        row['status'] = 'keep-provisional'
                        incumbents[d] = row
                    else:
                        row['status'] = 'discard'
                record(row)
                report('searching')
            # Fixed selected policies; fresh A/B then B/A then A/B, both distributions.
            for repeat in range(3):
                for d in DISTRIBUTIONS[repeat % 2:] + DISTRIBUTIONS[:repeat % 2]:
                    for role in (('baseline', 'winner') if repeat % 2 == 0 else ('winner', 'baseline')):
                        if time.monotonic() - started >= SOFT_SECONDS - WORKER_TIMEOUT - 5:
                            break
                        policy = BASE.copy() if role == 'baseline' else incumbents[d]['policy']
                        row = measure(d, policy, 'confirm-' + role, f'Independent confirmation pair {repeat + 1}')
                        if 'result' in row:
                            confirmation_rows[d].append({**row, 'role': role})
                        record(row)
                        report('confirming')
            report('completed')
            update('completed', stage='finished', summary=str(job / 'summary.json'))
    except BaseException as exc:
        report('stopped' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed', str(exc))
        update('stopped' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed', error=str(exc))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('setup', 'run', 'worker'))
    p.add_argument('--job', type=Path)
    p.add_argument('--mode', choices=('kernels', 'reference', 'smoke', 'profile', 'measure'))
    p.add_argument('--distribution', choices=DISTRIBUTIONS)
    p.add_argument('--run-dir', type=Path)
    args = p.parse_args()
    storage()
    if args.command == 'setup':
        setup()
    elif args.command == 'run':
        run(artifact(args.job))
    else:
        worker(args)


if __name__ == '__main__':
    main()
