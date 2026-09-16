"""Provision immutable model revisions and a fixed local speech workload."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from paths import ROOT, artifact, digest, save, storage


def runtime():
    import torch
    import zlib
    library_dirs = set()
    libraries = [Path(torch.__file__).parent / 'lib' / name for name in
                 ('libtorch_cuda.so', 'libtorch_global_deps.so', 'libtorch_python.so', 'libtorch_cpu.so')]
    for library in libraries + [Path(zlib.__file__)]:
        listing = subprocess.check_output(['ldd', str(library)], text=True)
        for line in listing.splitlines():
            if '=>' in line:
                resolved = line.split('=>', 1)[1].strip().split()[0]
                if resolved.startswith('/nix/store/'):
                    library_dirs.add(str(Path(resolved).parent))
    artifact(ROOT / 'library-path.txt').write_text(':'.join(sorted(library_dirs)))
    packages = {p: importlib.metadata.version(p) for p in
                ('torch', 'triton', 'numpy', 'transformers', 'ctranslate2', 'nvidia-ml-py', 'soundfile')}
    save(ROOT / 'runtime.json', dict(python=sys.version, executable=str(Path(sys.executable).resolve()),
                                   packages=packages, torch_cuda=torch.version.cuda,
                                   architecture_list=torch.cuda.get_arch_list(),
                                   requirements_sha256=digest(Path(__file__).with_name('requirements.lock'))))
    print(json.dumps(packages), flush=True)


def models():
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    specs = [('openai/whisper-medium.en', 'whisper-medium.en',
              ['*.json', 'merges.txt', '*.tiktoken']),
             ('Systran/faster-whisper-medium.en', 'ct2-medium.en',
              ['model.bin', '*.json', 'vocabulary.txt'])]
    lock_path = ROOT / 'models/model-lock.json'
    previous = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    records = dict(previous)
    for repo, folder, patterns in specs:
        revision = previous.get(repo, {}).get('revision') or api.model_info(repo).sha
        # Persist the selected revision before a potentially interrupted download.
        records[repo] = dict(revision=revision, path=str(ROOT / 'models' / folder))
        save(lock_path, records)
        target = artifact(ROOT / 'models' / folder)
        snapshot_download(repo_id=repo, revision=revision, local_dir=target,
                          cache_dir=ROOT / 'cache/huggingface/hub', allow_patterns=patterns,
                          max_workers=4)
        files = {str(p.relative_to(target)): dict(bytes=p.stat().st_size, sha256=digest(p))
                 for p in sorted(target.rglob('*')) if p.is_file() and '.cache' not in p.parts}
        records[repo]['files'] = files
        save(lock_path, records)
        print(f'Prepared {repo}@{revision}', flush=True)
    cfg = json.loads((ROOT / 'models/whisper-medium.en/config.json').read_text())
    expected = {'d_model': 1024, 'encoder_layers': 24, 'decoder_layers': 24,
                'encoder_attention_heads': 16, 'decoder_attention_heads': 16,
                'vocab_size': 51864, 'num_mel_bins': 80}
    for key, value in expected.items():
        if cfg[key] != value:
            raise ValueError(f'Model architecture mismatch: {key}={cfg[key]} != {value}')


def data(source, count):
    import numpy as np
    import soundfile as sf
    from transformers import AutoTokenizer, WhisperFeatureExtractor
    source = Path(source).resolve()
    if not source.is_relative_to(Path('/mnt/hd')):
        raise ValueError('Source audio must be on /mnt/hd')
    out = artifact(ROOT / 'data')
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/whisper-medium.en', local_files_only=True)
    extractor = WhisperFeatureExtractor.from_pretrained(ROOT / 'models/whisper-medium.en', local_files_only=True)
    files = iter(sorted(source.rglob('*.flac')))
    sample_count = 30 * 16000
    records = []
    transcript_cache = {}
    for i in range(count):
        waveform = np.empty(sample_count, np.float32)
        cursor, provenance, texts = 0, [], []
        while cursor < sample_count:
            try:
                path = next(files)
            except StopIteration as exc:
                raise RuntimeError('Not enough local speech to construct workload') from exc
            samples, rate = sf.read(path, dtype='float32')
            if rate != 16000 or samples.ndim != 1:
                raise ValueError(f'Expected mono 16 kHz: {path}')
            n = min(len(samples), sample_count - cursor)
            waveform[cursor:cursor+n] = samples[:n]
            provenance.append(dict(path=str(path), sha256=digest(path), start_sample=0,
                                   samples=n, destination_start=cursor))
            trans = path.parent / ('-'.join(path.stem.split('-')[:2]) + '.trans.txt')
            if trans not in transcript_cache:
                transcript_cache[trans] = dict(line.split(' ', 1) for line in trans.read_text().splitlines())
            texts.append(transcript_cache[trans][path.stem])
            cursor += n
        tokens = tokenizer.encode(' '.join(texts), add_special_tokens=False)
        if not tokens:
            raise ValueError('Empty transcript tokens')
        replay = (tokens * ((128 + len(tokens) - 1) // len(tokens)))[:128]
        pcm_path, mel_path = out / f'clip-{i:02d}.npy', out / f'mel-{i:02d}.npy'
        np.save(pcm_path, waveform, allow_pickle=False)
        mel = extractor(waveform, sampling_rate=16000, return_tensors='np').input_features
        if mel.shape != (1, 80, 3000) or not np.isfinite(mel).all():
            raise ValueError(f'Unexpected mel features: {mel.shape}')
        np.save(mel_path, mel, allow_pickle=False)
        records.append(dict(id=f'clip-{i:02d}', seconds=30, pcm=str(pcm_path),
                            pcm_sha256=digest(pcm_path), mel=str(mel_path), mel_sha256=digest(mel_path),
                            replay_tokens=replay, source_audio=provenance))
    shared_tokens = records[0]['replay_tokens']
    for record in records:
        record['replay_tokens'] = shared_tokens
    save(out / 'manifest.json', dict(schema=1, source=str(source), clips=records,
                                   sample_rate=16000, decoder_steps=128,
                                   replay_tokens=shared_tokens,
                                   model_lock_sha256=digest(ROOT / 'models/model-lock.json'),
                                   note='Concatenated local training speech, trimmed to 30 seconds; fixed transcript token replay.'))
    print(f'Prepared {count} fixed 30-second speech clips', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('runtime', 'models', 'data', 'all'))
    parser.add_argument('--source', default='/mnt/hd/wilderness-labs-stt/stt-distillation/datasets/libri/LibriSpeech/train-clean-100')
    parser.add_argument('--clips', type=int, default=16)
    args = parser.parse_args()
    storage()
    if args.stage in ('runtime', 'all'):
        runtime()
    if args.stage in ('models', 'all'):
        models()
    if args.stage in ('data', 'all'):
        data(args.source, args.clips)


if __name__ == '__main__':
    main()
