"""Agent-led kernel sprint: immutable per-step snapshots and fresh paired energy windows."""
import argparse
import datetime as dt
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time
from types import SimpleNamespace

from paths import ROOT, artifact, digest, save, storage

HERE = Path(__file__).resolve().parent
LOCK = Path('/mnt/hd/wilderness-labs-stt/stt-distillation/active.lock')
ORIGINAL_RUN = ROOT / 'kernel-research/20260912T141649Z'
CORE = ('benchmark.py', 'runtime_model.py', 'energy.py', 'paths.py',
        'kernels/packed.py', 'kernels/cuda_gemv.py')


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def hashes(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def remaining(job):
    return read(job / 'config.json')['deadline_epoch'] - time.time()


def inspect_gpu(cfg):
    from energy import EnergyMeter
    with EnergyMeter() as meter:
        if meter.foreign_processes():
            raise RuntimeError('Foreign GPU compute process')
        gpu = meter.metadata()
        if any(gpu[k] != cfg['gpu'][k] for k in ('uuid', 'power_limit_w')):
            raise RuntimeError('GPU identity or power limit changed')


def initialize(job, deadline):
    from energy import EnergyMeter
    storage()
    job.mkdir(parents=True, exist_ok=False)
    for name in CORE:
        if digest(HERE / name) != digest(ORIGINAL_RUN / 'code' / name):
            raise RuntimeError(f'Cannot reuse prior correctness reference: changed {name}')
    shutil.copytree(HERE, job / 'code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (job / 'autoresearch').mkdir()
    (job / 'autoresearch/.training-runtime').symlink_to((HERE.parent / 'autoresearch/.training-runtime').resolve())
    for d in ('binary', 'ternary'):
        shutil.copy2(ORIGINAL_RUN / f'reference-{d}.npz', job / f'reference-{d}.npz')
    with EnergyMeter() as meter:
        if meter.foreign_processes():
            raise RuntimeError('GPU busy')
        gpu = meter.metadata()
    cfg = {**read(ORIGINAL_RUN / 'config.json'), 'source_hashes': hashes(job / 'code'),
           'gpu': gpu, 'created_utc': utc(), 'deadline_epoch': deadline,
           'deadline_utc': dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat(),
           'reference_hashes': {d: digest(job / f'reference-{d}.npz') for d in ('binary', 'ternary')},
           'seconds': 60, 'idle_seconds': 5,
           'objective': 'GPU board joules per complete inference with fresh paired unchanged-kernel baseline',
           'protocol': 'Alternating AB/BA pairs; >=60s full-clip-cycle windows; independent final pairs'}
    save(job / 'config.json', cfg)
    for name in ('queue', 'requests', 'trials', 'results'):
        (job / name).mkdir()
    unit = 'stt-agent-kernel-sprint-' + job.name.lower()
    save(job / 'state.json', dict(status='ready', unit=unit, deadline_utc=cfg['deadline_utc']))
    limit = max(1, int(deadline - time.time()))
    subprocess.run(['systemd-run', '--user', '--unit', unit,
                    f'--property=RuntimeMaxSec={limit}', '--property=TimeoutStopSec=5',
                    '--property=KillMode=control-group', '--property=Restart=no',
                    f'--property=WorkingDirectory={job / "code"}',
                    f'--property=StandardOutput=append:{job / "controller.log"}',
                    f'--property=StandardError=append:{job / "controller.log"}',
                    str(job / 'code/python'), str(job / 'code/kernel_sprint.py'), 'controller',
                    '--job', str(job)], check=True)
    save(ROOT / 'agent-kernel-sprint/latest.json', dict(job=str(job), unit=unit, deadline_utc=cfg['deadline_utc']))
    print(json.dumps({'job': str(job), 'unit': unit, 'deadline_utc': cfg['deadline_utc']}), flush=True)


def submit(job, name, distribution, plugins, phase='search', priority=50):
    if remaining(job) < 200:
        raise RuntimeError('Too late to admit a paired experiment')
    request_dir = job / 'requests' / name
    request_dir.mkdir(exist_ok=False)
    shutil.copytree(job / 'code', request_dir / 'code')
    (request_dir / 'autoresearch').mkdir()
    (request_dir / 'autoresearch/.training-runtime').symlink_to((HERE.parent / 'autoresearch/.training-runtime').resolve())
    # Add only agent-owned implementation files; the evaluator and baseline are frozen.
    for path in (HERE / 'kernels').glob('sprint_*.py'):
        before = digest(path)
        shutil.copy2(path, request_dir / 'code/kernels' / path.name)
        if before != digest(path) or before != digest(request_dir / 'code/kernels' / path.name):
            raise RuntimeError('Agent file changed during snapshot; submit again after writer finishes')
    for plugin in plugins:
        if not plugin['module'].startswith('kernels.sprint_'):
            raise ValueError('Candidate must use an agent-owned sprint module')
    cfg = {**read(job / 'config.json'), 'source_hashes': hashes(request_dir / 'code')}
    save(request_dir / 'config.json', cfg)
    request = dict(name=name, distribution=distribution, plugins=plugins, phase=phase,
                   priority=priority, submitted_utc=utc(), directory=str(request_dir))
    save(job / 'queue' / f'{name}.json', request)
    print(json.dumps(request), flush=True)


def submit_frozen(job, name, source_name, priority=5, *, distribution=None,
                  plugins=None, screening_request=None):
    """Confirm the exact selected source, independently of later agent edits."""
    if remaining(job) < 200:
        raise RuntimeError('Too late to admit a paired confirmation')
    source = read(job / 'queue' / f'{source_name}.json')
    old = Path(source['directory'])
    directory = job / 'requests' / name
    directory.mkdir(exist_ok=False)
    shutil.copytree(old / 'code', directory / 'code')
    shutil.copy2(old / 'config.json', directory / 'config.json')
    (directory / 'autoresearch').mkdir()
    (directory / 'autoresearch/.training-runtime').symlink_to(
        (HERE.parent / 'autoresearch/.training-runtime').resolve())
    if hashes(directory / 'code') != read(directory / 'config.json')['source_hashes']:
        raise RuntimeError('Selected candidate source hash mismatch')
    request = {**source, 'name': name, 'phase': 'confirm', 'priority': priority,
               'source_request': source_name, 'directory': str(directory), 'submitted_utc': utc()}
    if distribution is not None:
        if distribution not in ('binary', 'ternary'):
            raise ValueError('Invalid model distribution')
        request['distribution'] = distribution
    if plugins is not None:
        request['plugins'] = plugins
    if screening_request is not None:
        request['screening_request'] = screening_request
    save(job / 'queue' / f'{name}.json', request)
    print(json.dumps(request), flush=True)


def worker(job, trial):
    import numpy as np
    import torch
    import benchmark
    cfg = read(trial / 'config.json')
    request = read(trial / 'request.json')
    distribution = request['distribution']
    inspect_gpu(cfg)
    if hashes(HERE) != cfg['source_hashes']:
        raise RuntimeError('Per-step frozen source changed')
    if digest(job / f'reference-{distribution}.npz') != cfg['reference_hashes'][distribution]:
        raise RuntimeError('Frozen reference changed')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    kernel_checks = []
    for plugin in request['plugins']:
        module = importlib.import_module(plugin['module'])
        directory = trial / 'candidate-artifacts' / plugin['module'].split('.')[-1]
        directory.mkdir(parents=True, exist_ok=True)
        if plugin['name'] not in module.variants():
            raise ValueError(f'Unknown candidate {plugin}')
        if hasattr(module, 'check'):
            kernel_checks.append({'plugin': plugin, 'result': module.check(plugin['name'], directory)})
        module.install(plugin['name'], directory)
    save(trial / 'kernel-checks.json', kernel_checks)
    original_make = benchmark.make_runner
    manifest = benchmark.workload_manifest()

    def checked_runner(*args, **kwargs):
        model, call, infer, mel = original_make(*args, **kwargs)
        checks = []
        with np.load(job / f'reference-{distribution}.npz', allow_pickle=False) as reference:
            for i, clip in enumerate(manifest['clips']):
                actual = infer(np.load(clip['mel'], allow_pickle=False)).cpu().float().numpy()
                expected = reference[f'logits_{i}'].astype(np.float32)
                delta = actual - expected
                nrms = float(np.sqrt(np.mean(delta ** 2)) / max(np.sqrt(np.mean(expected ** 2)), 1e-8))
                nmax = float(np.max(np.abs(delta)) / max(np.max(np.abs(expected)), 1e-8))
                if not np.isfinite(actual).all() or nrms > .005 or nmax > .02:
                    raise RuntimeError(f'Full-model numerical regression clip {i}: NRMS={nrms}, NMAX={nmax}')
                checks.append(dict(clip=i, normalized_rms_error=nrms, normalized_max_error=nmax,
                                   prediction_agreement=float(np.mean(model.last_predictions.cpu().numpy() == reference[f'predictions_{i}']))))
        save(trial / 'correctness.json', dict(passed=True, clips=checks,
              limits=dict(normalized_rms_error=.005, normalized_max_error=.02)))
        original_metadata = model.metadata
        def metadata():
            row = original_metadata()
            row['sprint_plugins'] = request['plugins']
            if request['plugins']:
                row['decode_backend'] = 'agent-sprint (see sprint_plugins and source snapshot)'
            return row
        model.metadata = metadata
        return model, call, infer, mel

    benchmark.make_runner = checked_runner
    benchmark.worker(SimpleNamespace(run_dir=trial, variant=distribution + '-packed', repeat=0, smoke=False))


def report(job):
    rows = [read(p) for p in sorted((job / 'results').glob('*.json'))]
    by_candidate = {}
    for row in rows:
        if row.get('status') != 'measured':
            continue
        label = json.dumps([row['distribution'], row['plugins']], sort_keys=True)
        by_candidate.setdefault(label, []).append(row)
    summary = []
    for group in by_candidate.values():
        final = [r for r in group if r['phase'] == 'confirm']
        use = final or group
        ratios = [r['energy_ratio'] for r in use]
        latency = [r['p95_ratio'] for r in use]
        summary.append(dict(distribution=group[0]['distribution'], plugins=group[0]['plugins'],
                            measured_pairs=len(group), confirmation_pairs=len(final),
                            energy_ratio=statistics.mean(ratios), p95_ratio=statistics.mean(latency),
                            confirmed=len(final) >= 3 and statistics.mean(ratios) <= .98
                                      and max(ratios) < 1 and statistics.mean(latency) <= 1.02))
    save(job / 'summary.json', dict(updated_utc=utc(), candidates=summary, completed_requests=len(rows),
                                   deadline_utc=read(job / 'config.json')['deadline_utc'],
                                   scope='GPU board energy, same fixed model/work; no accuracy or new CT2 control claim'))
    lines = ['# Agent-led kernel sprint', '',
             '| Candidate | Model | Pairs / final | Energy change | p95 change | Confirmed |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for row in summary:
        label = ' + '.join(p['module'].split('.')[-1] + ':' + p['name'] for p in row['plugins']) or 'baseline calibration'
        lines.append(f"| {label} | {row['distribution']} | {row['measured_pairs']} / {row['confirmation_pairs']} | {100*(row['energy_ratio']-1):+.2f}% | {100*(row['p95_ratio']-1):+.2f}% | {row['confirmed']} |")
    lines += ['', 'Every candidate is paired with a fresh original-kernel baseline. AB/BA order alternates. Each window covers at least60seconds and complete16-clip cycles.',
              '', 'Confirmation requires three independent final pairs, mean energy reduction >=2%, each energy ratio below1, and mean p95 regression <=2%. Screening results are provisional.']
    (job / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def controller(job):
    cfg = read(job / 'config.json')
    started = time.monotonic()
    with (job / 'timed-start.json').open('x') as out:
        json.dump(dict(started_utc=utc(), deadline_utc=cfg['deadline_utc']), out)
    state = read(job / 'state.json')
    seen, count = set(), 0
    def update(status, **details):
        state.update(status=status, updated_utc=utc(), remaining_seconds=max(0, remaining(job)), **details)
        save(job / 'state.json', state)
        print(json.dumps(state), flush=True)
    def stop(signum, frame):
        raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        with LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            inspect_gpu(cfg)
            update('running', stage='waiting-for-candidate')
            while remaining(job) >= 175 and not (job / 'FINISH').exists():
                pending = [read(p) for p in (job / 'queue').glob('*.json') if p.stem not in seen]
                if not pending:
                    time.sleep(1)
                    continue
                request = min(pending, key=lambda r: (r['priority'], r['submitted_utc']))
                seen.add(request['name'])
                count += 1
                row = {**request, 'started_utc': utc(), 'windows': {}, 'status': 'running'}
                snapshot = Path(request['directory'])
                # Alternate arm order to reduce monotonic thermal/clock drift bias.
                order = ('baseline', 'candidate') if count % 2 else ('candidate', 'baseline')
                if request['phase'] == 'calibration':
                    order = ('baseline', 'candidate')
                try:
                    for arm in order:
                        if remaining(job) < 85:
                            raise TimeoutError('Insufficient remaining budget for a complete window')
                        trial = job / 'trials' / (request['name'] + '-' + arm)
                        trial.mkdir()
                        save(trial / 'config.json', read(snapshot / 'config.json'))
                        save(trial / 'request.json', {**request, 'plugins': request['plugins'] if arm == 'candidate' else []})
                        update('running', stage=arm, request=request['name'], distribution=request['distribution'],
                               plugins=request['plugins'], trial=str(trial))
                        env = {**os.environ, 'EFFICIENCY_REQUIRE_CUDA_GEMV': '1'}
                        with (trial / 'worker.log').open('w') as output:
                            proc = subprocess.Popen([str(snapshot / 'code/python'),
                                    str(snapshot / 'code/kernel_sprint.py'), 'worker', '--job', str(job), '--trial', str(trial)],
                                    cwd=snapshot / 'code', env=env, stdout=output, stderr=subprocess.STDOUT,
                                    start_new_session=True)
                            try:
                                rc = proc.wait(timeout=min(175, remaining(job) - 5))
                            except BaseException:
                                try:
                                    os.killpg(proc.pid, signal.SIGKILL)
                                except ProcessLookupError:
                                    pass
                                proc.wait()
                                raise
                        if rc:
                            raise RuntimeError(f'{arm} worker exited {rc}; see {trial / "worker.log"}')
                        result = read(trial / f'00-{request["distribution"]}-packed.json')
                        if result['measurement']['elapsed_seconds'] < 60 or result['completed_clips'] % 16:
                            raise RuntimeError('Incomplete measurement window')
                        row['windows'][arm] = {k: result[k] for k in
                            ('gpu_j_per_audio_second', 'p95_seconds', 'p50_seconds', 'avg_watts', 'real_time_factor',
                             'completed_clips', 'gpu_before', 'gpu_after')}
                        row['windows'][arm]['trial'] = str(trial)
                    a, b = row['windows']['candidate'], row['windows']['baseline']
                    row.update(status='measured', energy_ratio=a['gpu_j_per_audio_second']/b['gpu_j_per_audio_second'],
                               p95_ratio=a['p95_seconds']/b['p95_seconds'], watts_ratio=a['avg_watts']/b['avg_watts'])
                except Exception as exc:
                    row.update(status='rejected-error', error=f'{type(exc).__name__}: {exc}')
                row['finished_utc'] = utc()
                save(job / 'results' / (request['name'] + '.json'), row)
                with (job / 'ledger.jsonl').open('a') as output:
                    output.write(json.dumps(row) + '\n')
                    output.flush()
                    os.fsync(output.fileno())
                print(json.dumps(row), flush=True)
                report(job)
                update('running', stage='waiting-for-candidate')
            report(job)
            update('completed', stage='finished', elapsed_controller_seconds=time.monotonic()-started)
    except BaseException as exc:
        report(job)
        update('stopped', error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('initialize', 'submit', 'controller', 'worker', 'report'))
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--name')
    parser.add_argument('--distribution', choices=('binary', 'ternary'))
    parser.add_argument('--plugins', default='[]')
    parser.add_argument('--phase', default='search', choices=('search', 'confirm', 'calibration'))
    parser.add_argument('--priority', type=int, default=50)
    parser.add_argument('--trial', type=Path)
    args = parser.parse_args()
    job = artifact(args.job)
    if args.command == 'initialize':
        initialize(job, args.deadline)
    elif args.command == 'submit':
        submit(job, args.name, args.distribution, json.loads(args.plugins), args.phase, args.priority)
    elif args.command == 'controller':
        controller(job)
    elif args.command == 'report':
        report(job)
    else:
        worker(job, artifact(args.trial))


if __name__ == '__main__':
    main()
