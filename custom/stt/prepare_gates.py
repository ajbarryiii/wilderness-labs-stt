"""Restore local gate exports from an existing run, verifying archived hashes."""
import argparse
import json
from pathlib import Path
import shutil
from core import ROOT, digest

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=ROOT/'custom/autoresearch/runs/20260909T224149Z')
    args=p.parse_args()
    for name in ['008','010']:
        dest=ROOT/'custom/autoresearch/results/20260909T224149Z/trials'/name
        hashes=json.loads((dest/'artifact_hashes.json').read_text())
        for filename in ['weights.npz','weights.2bit']:
            source=args.run/'trials'/name/filename
            if digest(source)!=hashes[filename]:
                raise ValueError('Original gate export checksum mismatch: '+str(source))
            shutil.copy2(source,dest/filename)
        print('Restored local gate',name)
