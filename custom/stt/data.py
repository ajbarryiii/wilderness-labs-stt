"""Join cached real audio to transcripts, retaining the established speaker splits."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
from core import ROOT, digest, encode, features, normalize, save

BASE = Path(__file__).resolve().parent


def audio(row):
    path = ROOT / row['path']
    x, sr = sf.read(path, start=row.get('offset_samples',0), frames=row['num_samples'],
                    dtype='float32', always_2d=True)
    if sr != 16000 or len(x) != row['num_samples']:
        raise ValueError('Audio length/rate mismatch')
    return x.mean(axis=1)


def prepare():
    source = ROOT/'custom/vad/data_manifest.json'
    m = json.loads(source.read_text())
    rows, files, groups, identities = [], {}, {}, set()
    for c in m['clips']:
        path = Path(c['path'])
        # Historical manifest is host-specific; resolve the portable cached suffix.
        path = ROOT/'custom/autoresearch/.data/audio'/str(path).split('/.data/audio/',1)[1]
        key = str(path.relative_to(ROOT))
        if key not in files:
            if digest(path) != c['sha256']:
                raise ValueError('Audio checksum mismatch: '+key)
            files[key] = c['sha256']
        group = ('speech', c['speaker_id']) if c['label'] else ('noise', c['source_id'])
        if group in groups and groups[group] != c['split']:
            raise ValueError('Split leakage')
        groups[group] = c['split']
        identity = (key,c['offset_samples'],c['num_samples'])
        if identity in identities:
            raise ValueError('Duplicate sample')
        identities.add(identity)
        row = {k:c[k] for k in ['split','label','speaker_id','source_id','offset_samples','num_samples','condition']}
        row.update(path=key, sha256=c['sha256'], text='')
        if c['label']:
            transcript = path.parent/('-'.join(path.stem.split('-')[:2])+'.trans.txt')
            transkey = str(transcript.relative_to(ROOT))
            files[transkey] = digest(transcript)
            transcripts = dict(line.split(' ',1) for line in transcript.read_text().splitlines())
            row['text'] = normalize(transcripts[path.stem])
            target = encode(row['text'])
            frames = features(audio(row)).shape[-1]
            if (frames+1)//2 < len(target)+sum(a==b for a,b in zip(target,target[1:])):
                raise ValueError('Infeasible CTC alignment: '+key)
        rows.append(row)
    cache = BASE/'cache'
    cache.mkdir(exist_ok=True)
    manifest = dict(schema=1, source_manifest_sha256=digest(source), sources=m['sources'],
        split_policy=m['split_policy'], files=files, rows=rows,
        limitations=['Development reused during search; no final holdout measurement.',
                     'Read English speech, not medical or field validation.'])
    save(cache/'manifest.json', manifest)
    arrays = {}
    for i,row in enumerate(rows):
        arrays[str(i)] = features(audio(row))
    np.savez(cache/'features.npz', **arrays)
    save(cache/'lock.json',dict(manifest_sha256=digest(cache/'manifest.json'),
                              features_sha256=digest(cache/'features.npz')))
    print(json.dumps(dict(rows=len(rows), speech_splits=dict(Counter(r['split'] for r in rows if r['label'])))))


def load(cache=None, verify_audio=False):
    cache = Path(cache) if cache else BASE/'cache'
    lock = json.loads((cache/'lock.json').read_text())
    for name in ['manifest','features']:
        path = cache/(name + ('.json' if name=='manifest' else '.npz'))
        if digest(path) != lock[name+'_sha256']:
            raise ValueError('Cache checksum mismatch')
    manifest = json.loads((cache/'manifest.json').read_text())
    if verify_audio:
        for path, h in manifest['files'].items():
            if digest(ROOT/path) != h:
                raise ValueError('Source checksum mismatch: '+path)
    return manifest, np.load(cache/'features.npz', allow_pickle=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','verify'])
    if p.parse_args().command == 'prepare':
        prepare()
    else:
        m,_ = load(verify_audio=True)
        print('Verified',len(m['rows']),'real audio/transcript records')
