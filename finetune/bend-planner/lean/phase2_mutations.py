#!/usr/bin/env python3
"""Two independent, well-formed statement mutants per arithmetic law.

Unit replaces a dropped proof hypothesis, retaining its binder for the original
proof body. This is a logically vacuous hypothesis, not an unknown identifier.
An unchanged body rejecting a true stronger statement does not establish that
the old hypothesis was necessary. All variants keep their original proof bodies.
"""
import difflib
import json
from pathlib import Path
import re
import shutil

from phase2_bend import HERE, ORIGINAL, SCRATCH, declarations, check, clone


def goal_parts(statement):
    lines = statement.splitlines(True)
    start = max(i for i,l in enumerate(lines) if l.startswith('  for '))+1
    return ''.join(lines[:start]), ''.join(lines[start:]).strip()


def equality_type(goal):
    return goal.rsplit(' : ',1)[1][:-1]


def mutate_goal(goal, mode):
    if ' == ' in goal:
        if mode == 0:
            return goal.replace(' == ', ' != ', 1), 'negate conclusion equality'
        ty = equality_type(goal)
        left,right = goal[1:-1].rsplit(' : ',1)[0].split(' == ',1)
        if ty=='Nat': left='1n+'+left
        elif ty=='U32': left='U32.inc('+left+')'
        elif ty.startswith('Word('): left='Word.inc(n, '+left+')'
        elif ty=='Cmp': left='Word.cmp.fin(False{}, True{}, '+left+')'
        elif ty=='Bool':
            for old,new in [('Nat.is_le(', 'Nat.is_lt('),('Nat.is_eq(', 'Nat.is_lt(')]:
                if old in goal:
                    return goal.replace(old,new,1), 'make non-strict/equality test strict'
            return '{Bool.not('+left+') == '+right+' : Bool}', 'negate comparison result'
        else: raise ValueError((ty,goal))
        return '{'+left+' == '+right+' : '+ty+'}', 'shift natural/word result by one; break equal-compare tie'
    if goal.startswith('N.le('):
        return goal.replace('N.le(', 'N.lt(',1), 'make order conclusion strict'
    if goal.startswith('N.lt('):
        body=goal[5:-1]
        depth=0
        for i,c in enumerate(body):
            if c=='(': depth+=1
            if c==')': depth-=1
            if c==',' and depth==0:
                return 'N.le('+body[i+1:].strip()+', '+body[:i]+')', 'reverse strict order into its negation'
        raise ValueError(goal)
    if goal.startswith('Word.Shift('):
        if mode==0: return '('+goal+' -> Empty)', 'negate shift certificate'
        return goal.replace('Word.Shift(n, c,', 'Word.Shift(n, Bool.not(c),',1), 'flip incoming bit in claimed shift'
    if goal.startswith('U32.DivSpec('):
        return '('+goal+' -> Empty)', 'negate quotient/remainder certificate'
    raise ValueError(goal)


def arithmetic_mutants():
    source = (ORIGINAL/'u32.bend').read_text()
    entries = declarations(source)
    assert len(entries)==41
    result=[]
    for i,entry in enumerate(entries):
        header,goal=goal_parts(entry['statement'])
        for mode in [0,1]:
            new_header=header
            hmatches=list(re.finditer(r'^  for ([+]?)(h|ab|bc|width|positive): (.+)$',header,re.M))
            if mode==1 and hmatches:
                hmatch=next((m for m in hmatches if m[2]=='width'),hmatches[-1])
                new_header=header[:hmatch.start(3)]+'Unit'+header[hmatch.end(3):]
                new_goal=goal
                intent=f'drop hypothesis {hmatch[2]} by replacing its proposition with Unit'
            else:
                new_goal,intent=mutate_goal(goal,mode)
            statement=new_header+'  '+new_goal+'\n\n'
            mutated=source[:entry['start']]+statement+source[entry['stop']:]
            directory=SCRATCH/'mutations'/f'{i+1:02}-{entry["name"].replace(".","-")}-{mode+1}'
            directory.mkdir(parents=True,exist_ok=True)
            (directory/'u32.bend').write_text(mutated)
            (directory/'PROOF.bend').write_text('import Base\nimport ./u32.bend as W\n')
            diff=''.join(difflib.unified_diff(source.splitlines(True),mutated.splitlines(True),fromfile='a/u32.bend',tofile='b/u32.bend'))
            r=check(directory/'PROOF.bend')
            r.update(law=entry['name'], mutant=mode+1, intent=intent, diff=diff,
                     baseline='original/u32.bend: both checkers pass', eligible=True)
            result.append(r)
            print(json.dumps({k:r[k] for k in ['law','mutant','intent','passed','rejected','elapsed_seconds']}),flush=True)
    return result


def planner_mutants():
    source=(ORIGINAL/'LAWS.bend').read_text()
    result=[]
    for name in ['pool_budget','pools_sizes','pools_cover']:
        for mode in ([0,1,2] if name=='pools_sizes' else [0,1]):
            directory=clone(f'mutations/planner-{name}-{mode+1}')
            if name!='pools_cover':
                shutil.copy2(SCRATCH/'ten-laws/LAWS.bend',directory/'LAWS.bend')
            laws=(directory/'LAWS.bend').read_text()
            entry=next(d for d in declarations(laws) if d['name']==name)
            old=entry['statement']
            if name=='pool_budget':
                new=old.replace('Budgets(b, Plan.pool(b, xs))',
                    'Budgets(0, Plan.pool(b, xs))' if mode==0 else 'Budgets(U32.sub(b, 1), Plan.pool(b, xs))')
                intent='claim zero budget' if mode==0 else 'decrease the claimed budget by one'
            elif name=='pools_sizes':
                if mode==0:
                    new=old.replace(' == ',' != ',1); intent='negate sizes equality'
                elif mode==1:
                    new=old.replace('== offsets.gaps(ps)','== 0n <> offsets.gaps(ps)'); intent='insert an extra zero-size pool in the expected lengths'
                else:
                    new=re.sub(r'^  for w  : Small\(ss\)\n','',old,flags=re.M)
                    intent='drop Small and its unused proof parameter; proof body unchanged'
            else:
                new=old.replace(' == ',' != ',1) if mode==0 else old.replace('Plan.clips(ss, 0)','Plan.clips(ss, 1)')
                intent='negate exact cover' if mode==0 else 'shift every expected clip index by one'
            assert new!=old
            mutated=laws[:entry['start']]+new+'\n\n'+laws[entry['stop']:]
            (directory/'LAWS.bend').write_text(mutated)
            proof_diff=''
            if name=='pools_sizes' and mode==2:
                file=directory/'PROOF.bend'; old_proof=file.read_text()
                new_proof=old_proof.replace('def Laws.pools_sizes(ss, ps, w, v):','def Laws.pools_sizes(ss, ps, v):')
                assert new_proof!=old_proof
                file.write_text(new_proof)
                proof_diff=''.join(difflib.unified_diff(old_proof.splitlines(True),new_proof.splitlines(True),fromfile='a/PROOF.bend',tofile='b/PROOF.bend'))
            r=check(directory/'PROOF.bend')
            r.update(law=name,mutant=mode+1,intent=intent,
                diff=''.join(difflib.unified_diff(laws.splitlines(True),mutated.splitlines(True),fromfile='a/LAWS.bend',tofile='b/LAWS.bend'))+proof_diff,
                eligible=name!='pools_cover',
                baseline='ten-laws: both checkers pass' if name!='pools_cover' else 'original: one TODO; no proof exists to mutation-test')
            result.append(r)
            print(json.dumps({k:r[k] for k in ['law','mutant','intent','passed','rejected','eligible']}),flush=True)
    # A non-vacuous single-clip baseline and the same two exact-cover mutations.
    probe=clone('cover-instance')
    common='''import Base
import ./plan.bend as Plan

law valid:
  {Plan.check([0, 1], 1) == None{} : Maybe<&2, String>}

law cover:
  {List.concat(&2, Plan.Clip, Plan.pools([7], [0, 1])) == Plan.clips([7], 0) : List<&2, Plan.Clip>}
'''
    proof='''import Base
import ./LAWS.bend as Laws

def Laws.valid():
  {==}

def Laws.cover():
  {==}
'''
    for mode in [None,0,1]:
        file=probe/'PROOF.bend'
        text=common
        if mode==0: text=text.replace(')) == Plan.clips', ')) != Plan.clips')
        if mode==1: text=text.replace('Plan.clips([7], 0)','Plan.clips([7], 1)')
        (probe/'LAWS.bend').write_text(text)
        file.write_text(proof)
        r=check(file)
        r.update(law='pools_cover_concrete_instance',mutant=mode,eligible=True,
                 intent='baseline' if mode is None else ('negate exact cover' if mode==0 else 'shift clip index by one'))
        result.append(r)
        (probe/f'LAWS-{mode}.bend').write_text(text)
        print(json.dumps(r),flush=True)
    return result


def main():
    arith=arithmetic_mutants()
    planner=planner_mutants()
    result=dict(arithmetic=arith,planner=planner)
    (HERE/'results/phase2_mutations.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Arithmetic survivors:',[(r['law'],r['mutant']) for r in arith if r['passed']],flush=True)
    print('Non-rejection failures:',[(r['law'],r['mutant']) for r in arith if not r['passed'] and not r['rejected']],flush=True)


if __name__=='__main__': main()
