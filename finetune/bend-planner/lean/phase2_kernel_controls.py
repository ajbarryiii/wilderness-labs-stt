#!/usr/bin/env python3
"""Supplement frontend-crashed mutants with direct kernel negative controls.

These are explicitly kernel-text controls, not successful Bend frontend runs.
"""
import json
import os
import re
import subprocess
import time
from phase2_bend import SCRATCH,HERE


def run(path):
    cmd=[os.environ['BENDTT'],str(path)]
    start=time.monotonic()
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=15,
                         env=dict(os.environ,LEAN_STACK_SIZE_KB='131072'))
        r=dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
            passed=p.returncode==0 and p.stdout.strip()=='ALL PROOFS CHECK',
            rejected=p.returncode==1 and p.stdout.startswith('SOME PROOFS FAIL'))
    except subprocess.TimeoutExpired as e:
        r=dict(command=cmd,timed_out_seconds=15,passed=False,rejected=False,
            stdout=(e.stdout or b'').decode(),stderr=(e.stderr or b'').decode())
    r['elapsed_seconds']=time.monotonic()-start
    return r


def main():
    original=SCRATCH/'u32.bendtt'
    source=original.read_text()
    results=[dict(law='baseline',**run(original))]
    directory=SCRATCH/'kernel-controls';directory.mkdir(exist_ok=True)
    for name in ['U32.to_nat_bound','U32.to_nat_inc','U32.to_nat_from_nat']:
        m=re.search(r'^'+re.escape(name)+r' : (.+)$',source,re.M)
        header=m[1]
        for mode in [0,1]:
            if mode==0:
                # Bend's != is the live function from equality to Empty.
                j=header.rfind(' -> ')+4
                assert header[j]=='{' and header.endswith(' =')
                new=header[:j]+'∀x99 : '+header[j:-2]+' -> Empty ='
            elif name=='U32.to_nat_bound':
                new=header.replace('(Nat.is_le ','(Nat.is_lt ',1)
            else:
                # The final proof binder is always x1 for these two laws.
                start=header.index(' : {',header.index(' -> ∀'))+3
                end=header.index(' -> ',start)
                new=header[:start]+'Unit'+header[end:]
            assert new!=header
            file=directory/f'{name.replace(".","-")}-{mode+1}.bendtt'
            file.write_text(source[:m.start(1)]+new+source[m.end(1):])
            r=dict(law=name,mutant=mode+1,old_header=header,new_header=new,**run(file))
            results.append(r); print(json.dumps(r),flush=True)
    (HERE/'results/phase2_kernel_controls.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__': main()
