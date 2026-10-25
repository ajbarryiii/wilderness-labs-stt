#!/usr/bin/env python3
"""Phase 3 real-file checks, immutable snapshot and reachable-source audit."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from phase2_bend import BEND_LIB, HERE, REAL, NAT, SCRATCH_ROOT, declarations
from phase2_audit import BASE

SCRATCH = SCRATCH_ROOT / 'phase3'
ORIGINAL = SCRATCH / 'original'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd, timeout=30, kernel=False):
    started = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
            env=dict(os.environ, LEAN_STACK_SIZE_KB='131072'))
        out = p.stdout + p.stderr
        result = dict(command=cmd, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr,
            passed=p.returncode == 0 and 'ALL PROOFS CHECK' in out,
            rejected=p.returncode == 1 and 'SOME PROOFS FAIL' in out
                and not re.search(r'overflow|RangeError|TODO|incomplete', out, re.I))
    except subprocess.TimeoutExpired as e:
        def dec(x):
            return x.decode() if isinstance(x, bytes) else x or ''
        result = dict(command=cmd, timed_out_seconds=timeout, stdout=dec(e.stdout),
            stderr=dec(e.stderr), passed=False, rejected=False)
    result['elapsed_seconds'] = time.monotonic() - started
    print(json.dumps(result), flush=True)
    return result


def source_audit():
    pending = [REAL/'LAWS.bend', REAL/'PROOF.bend']
    files = {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in files:
            continue
        raw = path.read_text()
        clean = re.sub(r'#.*', '', raw)
        imports = re.findall(r'^import (\S+)(?: as (\S+))?', clean, re.M)
        resolved = []
        for target, alias in imports:
            if target == 'Base':
                dest = BASE
            elif target.startswith('.'):
                dest = (path.parent/target).resolve()
            else:
                dest = BEND_LIB/target
            assert dest.exists(), (path, target, dest)
            resolved.append(dict(imported=target, alias=alias, file=str(dest)))
            pending.append(dest)
        defs = list(re.finditer(r'^def ([\w.?]+)(?:\(|:)', clean, re.M))
        unsafe = re.findall(r'@unsafe\s*\ndef ([\w.?]+)(?:\(|:)', clean)
        assert len(unsafe) == len(re.findall(r'@unsafe\b', clean))
        foreign = []
        for i, m in enumerate(defs):
            start = m.start()
            stop = defs[i+1].start() if i+1 < len(defs) else len(clean)
            body = clean[start:stop]
            if re.search(r'^\s+import [\"\']', body, re.M):
                foreign.append(m[1])
        files[str(path)] = dict(sha256=digest(path), imports=resolved,
            holes=[dict(line=i, text=l.strip()) for i,l in enumerate(clean.splitlines(),1)
                if re.search(r'\?(?:TODO\b|\s|$)', l)],
            question_definitions=[m[1] for m in defs if '?' in m[1]],
            unsafe_markers=[i for i,l in enumerate(clean.splitlines(),1) if '@unsafe' in l],
            unsafe_definitions=unsafe, foreign_definitions=foreign,
            foreign_import_lines=[dict(line=i, text=l.strip()) for i,l in enumerate(clean.splitlines(),1)
                if re.match(r'\s+import [\"\']', l)])
    return files


def kernel_audit(text, sources):
    marks = list(re.finditer(r'^(\S+) : ', text, re.M))
    blocks = {m[1]:text[m.start():marks[i+1].start() if i+1 < len(marks) else len(text)]
        for i,m in enumerate(marks)}
    graph = {n:set(re.findall(r'[A-Za-z_][A-Za-z_0-9.]*', s)).intersection(blocks)-{n}
        for n,s in blocks.items()}
    roots = ['small_bound', 'LAWS.pools_cover']
    roots += ['LAWS.'+d['name'] for d in declarations((REAL/'LAWS.bend').read_text())]
    closures = {}
    unsafe = set(sources[str(BASE)]['unsafe_definitions'])
    foreign = set(sources[str(BASE)]['foreign_definitions'])
    for root in dict.fromkeys(roots):
        seen = set()
        pending = [root]
        while pending:
            n = pending.pop()
            if n in seen:
                continue
            assert n in blocks, n
            seen.add(n)
            pending.extend(graph[n]-seen)
        closures[root] = dict(declarations=sorted(seen),
            unsafe=sorted(unsafe.intersection(seen)), foreign=sorted(foreign.intersection(seen)),
            TODO_tokens=sorted(n for n in seen if re.search(r'\?TODO|\bTODO\b', blocks[n])))
    law_names = [d['name'] for d in declarations((REAL/'LAWS.bend').read_text())]
    proof = re.sub(r'#.*', '', (REAL/'PROOF.bend').read_text())
    open_laws = [n for n in law_names if not re.search(r'^def Laws\.'+re.escape(n)+r'\(',proof,re.M)]
    return dict(closures=closures, planner_law_count=len(law_names), open_laws=open_laws,
        kernel_headers={n:blocks[n].splitlines()[0] for n in ['LAWS.Small','LAWS.Holds','small_bound','LAWS.pools_cover']})


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not ORIGINAL.exists():
        ORIGINAL.mkdir()
        for n in ['LAWS.bend','PROOF.bend','u32.bend','plan.bend']:
            shutil.copy2(REAL/n, ORIGINAL/n)
    hashes = {n:digest(ORIGINAL/n) for n in ['LAWS.bend','PROOF.bend','u32.bend','plan.bend']}
    assert all(digest(REAL/n) == h for n,h in hashes.items())
    result = dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REAL,text=True).strip(),
        source_hashes=hashes, sources=source_audit(), checks=[])
    diff = subprocess.check_output(['git','diff','--','LAWS.bend','PROOF.bend'],cwd=REAL,text=True)
    (HERE/'results/phase3_reviewed.diff').write_text(diff)
    for verdict in [False, True]:
        result['checks'].append(run(['bend',str(REAL/'PROOF.bend')]+(['--verdict'] if verdict else [])))
    export = SCRATCH/'planner-eleven.bendtt'
    result['export'] = run(['bend',str(ORIGINAL/'PROOF.bend'),'-o',str(export)])
    assert export.exists() and result['export']['returncode'] == 0
    result['kernel_baseline'] = run([os.environ['BENDTT'],str(export)],kernel=True)
    result['kernel_export_sha256'] = digest(export)
    result.update(kernel_audit(export.read_text(),result['sources']))
    minimal = HERE/'minimal.bend'
    if minimal.exists():
        shutil.copy2(minimal,SCRATCH/'minimal.bend')
        result['minimal_reproducer'] = dict(source=minimal.read_text(),sha256=digest(minimal),
            check=run(['bend',str(SCRATCH/'minimal.bend'),'--verdict']))
    assert all(digest(REAL/n) == h for n,h in hashes.items())
    (HERE/'results/phase3_bend_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Planner laws:',result['planner_law_count'],'open:',result['open_laws'],flush=True)
    for n in ['small_bound','LAWS.pools_cover']:
        c=result['closures'][n]
        print(n,'closure=',len(c['declarations']),'unsafe=',c['unsafe'],'foreign=',c['foreign'],flush=True)


if __name__ == '__main__':
    main()
