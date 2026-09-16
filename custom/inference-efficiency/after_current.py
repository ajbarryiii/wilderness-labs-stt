"""Run the frozen efficiency benchmark after a systemd experiment supervisor ends."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import time

from energy import EnergyMeter
from paths import artifact, save, storage


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def service_state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit,
                             '-p', 'LoadState', '-p', 'ActiveState', '-p', 'SubState',
                             '-p', 'MainPID', '-p', 'Result'],
                            capture_output=True, text=True, check=True)
    state = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if 'ActiveState' not in state:
        raise RuntimeError(f'Cannot determine predecessor state: {result.stdout}')
    return state


def predecessor_running(state):
    return state['ActiveState'] not in ('inactive', 'failed') or int(state.get('MainPID', '0')) != 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--check', action='store_true', help='Read readiness without launching work')
    args = parser.parse_args()
    storage()
    config = json.loads(artifact(args.config).read_text())
    status_path = artifact(args.config.parent / 'state.json')
    def update(status, **details):
        save(status_path, dict(status=status, updated_utc=utc(),
                               predecessor_unit=config['predecessor_unit'],
                               output_directory=config['output_directory'], **details))
    if args.check:
        with EnergyMeter() as meter:
            print(json.dumps(dict(predecessor=service_state(config['predecessor_unit']),
                                  foreign_processes=meter.foreign_processes()), indent=2))
        return
    try:
        while True:
            previous = service_state(config['predecessor_unit'])
            if not predecessor_running(previous):
                break
            update('waiting_for_current_experiment', predecessor=previous)
            time.sleep(config['poll_seconds'])
        # This is the same lock held by the current training supervisor. Hold it
        # through all benchmark subprocesses so other STT launchers cannot race us.
        with Path(config['shared_lock']).open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    update('waiting_for_experiment_lock', predecessor=previous)
                    time.sleep(config['poll_seconds'])
            with EnergyMeter() as meter:
                clear_since = None
                while True:
                    previous = service_state(config['predecessor_unit'])
                    if predecessor_running(previous):
                        raise RuntimeError('Predecessor restarted while queued benchmark acquired the lock')
                    foreign = meter.foreign_processes()
                    if foreign:
                        clear_since = None
                        update('waiting_for_clear_gpu', foreign_processes=foreign, predecessor=previous)
                    else:
                        if clear_since is None:
                            clear_since = time.monotonic()
                        if time.monotonic() - clear_since >= config['idle_grace_seconds']:
                            break
                        update('waiting_for_idle_grace', predecessor=previous)
                    time.sleep(config['poll_seconds'])
            update('running', predecessor=previous, command=config['command'])
            subprocess.run(config['command'], check=True, cwd=config['code_directory'])
            update('completed', predecessor=previous)
    except BaseException as exc:
        update('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
