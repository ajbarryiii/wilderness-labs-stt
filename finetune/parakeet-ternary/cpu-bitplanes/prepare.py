"""Build an isolated LocalFlow CPU benchmark snapshot; never change app dispatch.

Run with ../python cpu-bitplanes/prepare.py. Models, public audio, build caches
and benchmark artifacts are kept on the mounted project disk.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/hd/wilderness-labs-stt/parakeet-ternary')
DEFAULT_LOCALFLOW = Path.home() / 'workspace/github.com/LocalFlow/.claude/worktrees/linux-port/linux'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'LocalFlow integration changed: {old[:80]!r}')
    return text.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--localflow', type=Path, default=DEFAULT_LOCALFLOW)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if not os.path.ismount('/mnt/hd'):
        raise RuntimeError('/mnt/hd is not mounted')
    out = a.out or ROOT / 'cpu-bitplanes' / time.strftime('%Y%m%dT%H%M%S')
    if not out.resolve().is_relative_to(ROOT):
        raise ValueError('Artifacts must be under the project disk')
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / 'snapshot'
    snapshot.mkdir()
    hashes = {}
    for name in ['lf-model', 'lf-cpu', 'lf-asr']:
        source = a.localflow / 'crates' / name
        for f in source.rglob('*'):
            if f.is_file():
                hashes[str(f.relative_to(a.localflow))] = digest(f)
        shutil.copytree(source, snapshot / 'crates' / name)
    cargo = (a.localflow / 'Cargo.toml').read_text()
    cargo = cargo.replace('members = ["crates/*"]', 'members = ["crates/lf-model", "crates/lf-cpu", "crates/lf-asr", "crates/bitplane-bench"]')
    (snapshot / 'Cargo.toml').write_text(cargo)
    shutil.copy2(a.localflow / 'Cargo.lock', snapshot / 'Cargo.lock')
    cpu = snapshot / 'crates/lf-cpu/src'
    shutil.copy2(HERE / 'gemm_bitplanes.rs', cpu / 'gemm_bitplanes.rs')
    with (cpu / 'lib.rs').open('a') as f:
        f.write('\npub mod gemm_bitplanes;\n')
    encoder = snapshot / 'crates/lf-asr/src/encoder.rs'
    text = encoder.read_text()
    text = replace_once(text, '    q_pos: QuantizedActs,', '    q_pos: QuantizedActs,\n    bitplanes: Option<lf_cpu::gemm_bitplanes::Engine>,')
    text = replace_once(text, '            q_pos: QuantizedActs::with_capacity(0, D_MODEL, n),',
                        '            q_pos: QuantizedActs::with_capacity(0, D_MODEL, n),\n            bitplanes: lf_cpu::gemm_bitplanes::Engine::from_env(),')
    text = replace_once(text, '                gemm_i8::gemm(pool, &self.scratch, w, qa, y);',
                        '                if let Some(engine) = &mut self.bitplanes {\n'
                        '                    let si = match slot { Slot::Model => 0, Slot::Ff => 1, Slot::Pos => 2 };\n'
                        '                    engine.gemm(pool, w, qa, y, si, reuse);\n'
                        '                } else {\n'
                        '                    gemm_i8::gemm(pool, &self.scratch, w, qa, y);\n'
                        '                }')
    encoder.write_text(text)
    bench = snapshot / 'crates/bitplane-bench'
    (bench / 'src').mkdir(parents=True)
    shutil.copy2(HERE / 'bench.rs', bench / 'src/main.rs')
    (bench / 'Cargo.toml').write_text('''[package]
name = "bitplane-bench"
version.workspace = true
edition.workspace = true
license.workspace = true
publish.workspace = true
[dependencies]
lf-asr.workspace = true
lf-cpu.workspace = true
lf-model.workspace = true
serde_json.workspace = true
sha2.workspace = true
libc.workspace = true
''')
    # Select complete public utterances, without cutting or concatenating them.
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    audio = out / 'audio'
    audio.mkdir()
    records = []
    for manifest in sorted((ROOT / 'manifests').glob('test_*.jsonl')):
        source = manifest.stem.removeprefix('test_')
        candidates = [r for r in map(json.loads, manifest.read_text().splitlines()) if 15 <= r['duration'] <= 30]
        used = set()
        for target in [16, 28]:
            rec = min((r for r in candidates if r['id'] not in used), key=lambda r: abs(r['duration']-target))
            used.add(rec['id'])
            samples, rate = sf.read(rec['audio_filepath'], dtype='float32', always_2d=True)
            samples = samples.mean(axis=1)
            if rate != 16000:
                g = gcd(rate, 16000)
                samples = resample_poly(samples, 16000//g, rate//g).astype(np.float32)
            duration = len(samples) / 16000
            if not 15 <= duration <= 30 or not np.isfinite(samples).all():
                raise RuntimeError(f'Invalid selected public audio: {source}')
            pcm = audio / f'{len(records):02d}.f32'
            pcm.write_bytes(np.asarray(samples, dtype='<f4').tobytes())
            records.append({'id': str(rec['id']), 'source': source, 'duration_s': duration,
                            'samples': len(samples), 'pcm': str(pcm), 'pcm_sha256': digest(pcm),
                            'original_audio_sha256': digest(Path(rec['audio_filepath']))})
    (out / 'corpus.json').write_text(json.dumps(records, indent=2))
    metadata = {'localflow': str(a.localflow), 'localflow_revision': subprocess.check_output(['git','-C',str(a.localflow),'rev-parse','HEAD'], text=True).strip(),
                'localflow_source_sha256': hashes, 'experiment_source_sha256': {f.name: digest(f) for f in HERE.iterdir() if f.suffix in ('.py','.rs')},
                'source_snapshot': str(snapshot), 'corpus_count': len(records), 'audio_seconds': sum(r['duration_s'] for r in records),
                'export': str(ROOT / 'runs/main-M1-P2-lr5e-4/export'),
                'export_manifest_sha256': digest(ROOT / 'runs/main-M1-P2-lr5e-4/export/manifest.json'),
                'scope': 'Exact existing residual INT8 components; only ternary projection implementation changes; default LocalFlow source untouched'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(out, flush=True)


if __name__ == '__main__':
    main()
