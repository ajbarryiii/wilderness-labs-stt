#!/usr/bin/env python3
"""Independent Phase 2 Bend checks; mutations and edited laws live only in /tmp."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
REAL = HERE.parent
# Scratch outputs go under $BEND_LEAN_SCRATCH (default: <system tmp>/bend-lean-review)
SCRATCH_ROOT = Path(os.environ.get('BEND_LEAN_SCRATCH', Path(tempfile.gettempdir()) / 'bend-lean-review'))
SCRATCH = SCRATCH_ROOT / 'phase2'
ORIGINAL = SCRATCH / 'original'
BEND_LIB = Path.home() / '.bend' / 'lib'   # where bend caches hub packages
NAT = BEND_LIB / '0xcd4682caa53da19986acd472c2181473' / 'nat.bend'


def declarations(text, kind='law'):
    result = []
    for m in re.finditer(rf'^{kind} ([^:\n]+):\n', text, re.M):
        end = re.search(r'^\S', text[m.end():], re.M)
        stop = m.end()+end.start() if end else len(text)
        result.append(dict(name=m[1], start=m.start(), stop=stop,
                           statement=text[m.start():stop].rstrip()))
    return result


def clone(name):
    target = SCRATCH / name
    target.mkdir(parents=True, exist_ok=True)
    for p in ORIGINAL.glob('*.bend'):
        shutil.copy2(p, target/p.name)
    return target


def check(file, verdict=True, timeout=180):
    cmd = ['bend', str(file)] + (['--verdict'] if verdict else [])
    started = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stdout, stderr = p.stdout, p.stderr
        result = dict(command=cmd, returncode=p.returncode, stdout=stdout, stderr=stderr,
                      passed='ALL PROOFS CHECK' in stdout and 'SOME PROOFS FAIL' not in stdout+stderr,
                      rejected='SOME PROOFS FAIL' in stdout+stderr)
    except subprocess.TimeoutExpired as e:
        result = dict(command=cmd, timed_out_seconds=timeout, passed=False, rejected=False,
                      stdout=(e.stdout or b'').decode(), stderr=(e.stderr or b'').decode())
    result['elapsed_seconds'] = time.monotonic()-started
    file.with_suffix('.verdict.log' if verdict else '.check.log').write_text(result['stdout']+result['stderr']) if SCRATCH in file.parents else None
    return result


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not ORIGINAL.exists():
        ORIGINAL.mkdir()
        for name in ['u32.bend','PROOF.bend','LAWS.bend','plan.bend']:
            shutil.copy2(REAL/name,ORIGINAL/name)
    source = (ORIGINAL/'u32.bend').read_text()
    direct = sorted(set(re.findall(r'\bN\.([\w.]+)', re.sub(r'#.*', '', source))))
    result = dict(source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in ORIGINAL.glob('*.bend')}, nat_sha256=hashlib.sha256(NAT.read_bytes()).hexdigest(),
                  direct_nat_references=direct, checks=[])
    for name in ['u32.bend','PROOF.bend']:
        for verdict in [False, True]:
            r = check(REAL/name, verdict)
            result['checks'].append(r)
            print(json.dumps(r), flush=True)
    subset = clone('ten-laws')
    laws = (subset/'LAWS.bend').read_text()
    cover = next(d for d in declarations(laws) if d['name']=='pools_cover')
    (subset/'LAWS.bend').write_text(laws[:cover['start']]+laws[cover['stop']:])
    for verdict in [False,True]:
        r = check(subset/'PROOF.bend',verdict)
        result['checks'].append(r); print(json.dumps(r),flush=True)
    symbolic = clone('symbolic-small')
    laws = (symbolic/'LAWS.bend').read_text()
    old = 'Holds(Nat.is_le(List.length(&2, U32, ss), 4294967295n))'
    new = 'Holds(Nat.is_le(List.length(&2, U32, ss), W.Word.max_nat(32n)))'
    assert laws.count(old)==1
    laws = laws.replace('import Base\n','import Base\nimport ./u32.bend as W\n',1).replace(old,new)
    (symbolic/'LAWS.bend').write_text(laws)
    proof = (symbolic/'PROOF.bend').read_text()
    proof += '\n\ndef Laws.pools_cover(ss, ps, w, v):\n  pools_cover_bounded(ss, ps, w, v)\n'
    (symbolic/'PROOF.bend').write_text(proof)
    diff = ''
    for name in ['LAWS.bend','PROOF.bend']:
        diff += ''.join(difflib.unified_diff((ORIGINAL/name).read_text().splitlines(True),
            (symbolic/name).read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (HERE/'results/phase2_symbolic_small.diff').write_text(diff)
    for verdict in [False, True]:
        r = check(symbolic/'PROOF.bend', verdict)
        result['checks'].append(r); print(json.dumps(r),flush=True)
    assert all(hashlib.sha256((REAL/n).read_bytes()).hexdigest()==h for n,h in result['source_hashes'].items())
    (HERE/'results/phase2_bend_checks.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
