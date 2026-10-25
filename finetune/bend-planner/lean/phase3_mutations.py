#!/usr/bin/env python3
"""Source and direct BendTT controls for the newly completed pools_cover law."""
import difflib
import json
import os
import re
import shutil

from phase3_bend import HERE, SCRATCH, ORIGINAL, digest, run
from phase2_bend import declarations


def nat(n):
    r = '(.Zero, ())'
    for _ in range(n):
        r = '(.Succ, ('+r+', ()))'
    return r


def main():
    original = SCRATCH/'planner-eleven.bendtt'
    source = original.read_text()
    marker = re.search(r'^LAWS\.pools_cover : (.+)$',source,re.M)
    header = marker[1]
    assert header.count('(LAWS.Small +x0)') == 1
    small_body = re.search(r'^LAWS\.Small : .+\n  λ\+x0 => (.+)$',source,re.M)[1]
    assert small_body.count(nat(32)) == 1
    small33 = small_body.replace(nat(32),nat(33),1)
    j = header.rfind(' -> ')+4
    assert header[j] == '{' and header.endswith(' =')
    mutants = [
        ('a-start-one', 'Shift the expected clip start from U32 0 to U32 1.',
         header.replace('(plan.clips x0 +U32.lit0)','(plan.clips x0 +U32.lit1)',1)),
        ('b-negate-equality', 'Replace the conclusion equality by its negation.',
         header[:j]+'∀x99 : '+header[j:-2]+' -> Empty ='),
        ('c-small-width33', 'Change only the cover assumption to Small with width 33.',
         header.replace('(LAWS.Small +x0)',small33,1)),
        ('d-drop-valid', 'Remove offset validity, retaining its binder with type Unit.',
         header.replace('∀x3 : (LAWS.Valid +x0 +x1)','∀x3 : Unit',1)),
    ]
    directory = SCRATCH/'kernel-controls'
    directory.mkdir(exist_ok=True)
    result = dict(export_sha256=digest(original),old_header=header,
        width32_term=nat(32),width33_term=nat(33),kernel=[],frontend=[])
    result['kernel'].append(dict(control='baseline',**run([os.environ['BENDTT'],str(original)],timeout=15)))
    expanded = header.replace('(LAWS.Small +x0)',small_body,1)
    path = directory/'positive-small-expanded-width32.bendtt'
    path.write_text(source[:marker.start(1)]+expanded+source[marker.end(1):])
    result['kernel'].append(dict(control='positive-small-expanded-width32',new_header=expanded,
        **run([os.environ['BENDTT'],str(path)],timeout=15)))
    for name,intent,new in mutants:
        assert new != header
        path = directory/(name+'.bendtt')
        path.write_text(source[:marker.start(1)]+new+source[marker.end(1):])
        diff = ''.join(difflib.unified_diff([header+'\n'],[new+'\n'],
            fromfile='original pools_cover header',tofile=name))
        r = dict(mutant=name,intent=intent,new_header=new,diff=diff,
            **run([os.environ['BENDTT'],str(path)],timeout=15))
        result['kernel'].append(r)
    laws = (ORIGINAL/'LAWS.bend').read_text()
    cover = next(x for x in declarations(laws) if x['name']=='pools_cover')
    block = laws[cover['start']:cover['stop']]
    for name,intent,_ in mutants:
        dest = SCRATCH/'source-mutants'/name
        dest.mkdir(parents=True,exist_ok=True)
        for p in ORIGINAL.glob('*.bend'):
            shutil.copy2(p,dest/p.name)
        if name == 'a-start-one':
            changed = block.replace('Plan.clips(ss, 0)','Plan.clips(ss, 1)',1)
            new = laws[:cover['start']]+changed+laws[cover['stop']:]
        elif name == 'b-negate-equality':
            changed = block.replace(' == ',' != ',1)
            new = laws[:cover['start']]+changed+laws[cover['stop']:]
        elif name == 'c-small-width33':
            # The source mutation changes shared Small exactly as AJ's experiment.
            new = laws.replace('W.Word.max_nat(32n)','W.Word.max_nat(33n)',1)
        else:
            changed = block.replace('for v  : Valid(ss, ps)','for v  : Unit',1)
            new = laws[:cover['start']]+changed+laws[cover['stop']:]
        assert new != laws
        (dest/'LAWS.bend').write_text(new)
        diff = ''.join(difflib.unified_diff(laws.splitlines(True),new.splitlines(True),
            fromfile='a/LAWS.bend',tofile='b/LAWS.bend'))
        result['frontend'].append(dict(mutant=name,intent=intent,diff=diff,
            **run(['bend',str(dest/'PROOF.bend'),'--verdict'],timeout=30)))
    (HERE/'results/phase3_mutations.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
