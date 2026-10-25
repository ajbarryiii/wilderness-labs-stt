#!/usr/bin/env python3
"""Dependency and placeholder audit, using emitted, resolved kernel names."""
import hashlib
import json
from pathlib import Path
import re
import shutil

from phase2_bend import HERE, ORIGINAL, SCRATCH, NAT, REAL, declarations

# Base of the bend on PATH (the bend2 package keeps it at share/bend2/bend2/base.bend)
BASE=Path(shutil.which('bend')).resolve().parent.parent/'share'/'bend2'/'bend2'/'base.bend'
PREFIX='_30_xcd4682caa53da19986acd472c2181473_2f_nat.'


def main():
    src=(ORIGINAL/'u32.bend').read_text()
    direct=sorted(set(re.findall(r'\bN\.([\w.]+)', re.sub(r'#.*','',src))))
    nat=NAT.read_text()
    nat_laws={d['name']:d['statement'] for d in declarations(nat)}
    direct_entries=[dict(name=n,kind='law' if n in nat_laws else 'helper definition',
                         statement=nat_laws.get(n)) for n in direct]
    text=(SCRATCH/'u32.bendtt').read_text()
    marks=list(re.finditer(r'^(\S+) : ',text,re.M))
    blocks={m[1]:text[m.start():marks[i+1].start() if i+1<len(marks) else len(text)]
            for i,m in enumerate(marks)}
    graph={n:set(re.findall(r'[A-Za-z_][A-Za-z_0-9.]*',s)).intersection(blocks)-{n}
           for n,s in blocks.items()}
    roots=[d['name'] for d in declarations(src)]
    seen=set()
    todo=roots[:]
    while todo:
        n=todo.pop()
        if n in seen: continue
        seen.add(n);todo.extend(graph[n]-seen)
    transitive=sorted(n[len(PREFIX):] for n in seen if n.startswith(PREFIX))
    assert set(direct).issubset(transitive)
    headers={n:blocks[n].splitlines()[0] for n in roots}
    (HERE/'results/phase2_kernel_headers.json').write_text(json.dumps(headers,indent=2)+'\n')
    scans=[]
    for path in [ORIGINAL/n for n in ['LAWS.bend','PROOF.bend','u32.bend','plan.bend']]+[NAT,BASE]:
        raw=path.read_text()
        clean=re.sub(r'#.*','',raw)
        defs=set(re.findall(r'^def ([\w.?]+)\(',clean,re.M))
        foreign=[]
        for i,l in enumerate(clean.splitlines(),1):
            if re.match(r'\s+import ["\']',l): foreign.append(dict(line=i,text=l.strip()))
        scans.append(dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            imports=re.findall(r'^import .+$',clean,re.M),
            todo_tokens=re.findall(r'\?TODO\b',clean), unsafe_markers=re.findall(r'@unsafe\b',clean),
            question_defs=sorted(n for n in defs if '?' in n),
            open_laws=[d['name'] for d in declarations(clean) if d['name'] not in defs],
            foreign_imports=foreign))
    result=dict(direct_nat_references=direct_entries,
                direct_law_count=sum(d['kind']=='law' for d in direct_entries),
                transitive_nat_dependencies=transitive,
                transitive_dependency_count=len(transitive),
                transitive_method='Reachability from all 41 emitted laws through resolved global names in emitted BendTT types and bodies; not all imported definitions.',
                scans=scans,kernel_export_sha256=hashlib.sha256(text.encode()).hexdigest(),
                planner_open_laws=[d['name'] for d in declarations((ORIGINAL/'LAWS.bend').read_text())
                    if 'def Laws.'+d['name']+'(' not in (ORIGINAL/'PROOF.bend').read_text()])
    result['arithmetic_proof_references_to_unsafe_base'] = sorted(n for n in seen
        if n.startswith('Array.fork') or n.startswith('Array.join'))
    planner_text=(SCRATCH/'planner-ten.bendtt').read_text()
    marks=list(re.finditer(r'^(\S+) : ',planner_text,re.M))
    pblocks={m[1]:planner_text[m.start():marks[i+1].start() if i+1<len(marks) else len(planner_text)]
             for i,m in enumerate(marks)}
    pgraph={n:set(re.findall(r'[A-Za-z_][A-Za-z_0-9.]*',s)).intersection(pblocks)-{n}
            for n,s in pblocks.items()}
    pseen=set(); todo=['LAWS.'+d['name'] for d in declarations((ORIGINAL/'LAWS.bend').read_text())
                       if d['name']!='pools_cover']
    while todo:
        n=todo.pop()
        if n in pseen: continue
        pseen.add(n);todo.extend(pgraph[n]-pseen)
    result['planner_proof_references_to_unsafe_base']=sorted(n for n in pseen
        if n.startswith('Array.fork') or n.startswith('Array.join'))
    base_text=BASE.read_text()
    defs=list(re.finditer(r'^def ([\w.]+)\(',base_text,re.M))
    foreign_defs=[]
    for i,m in enumerate(defs):
        body=base_text[m.start():defs[i+1].start() if i+1<len(defs) else len(base_text)]
        if re.search(r'^\s+import ["\']',body,re.M): foreign_defs.append(m[1])
    result['base_foreign_definitions']=foreign_defs
    result['arithmetic_proof_references_to_foreign_base']=sorted(set(foreign_defs).intersection(seen))
    result['planner_proof_references_to_foreign_base']=sorted(set(foreign_defs).intersection(pseen))
    for s in scans:
        if s['file']==str(BASE):
            result['base_opaque_intrinsic_declarations']=s['open_laws']
    result['planner_kernel_export_sha256']=hashlib.sha256(planner_text.encode()).hexdigest()
    (HERE/'results/phase2_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['scans','direct_nat_references']},indent=2))
    print('Direct laws:', ', '.join(d['name'] for d in direct_entries if d['kind']=='law'))
    print('Direct helpers:', ', '.join(d['name'] for d in direct_entries if d['kind']!='law'))
    for s in scans:
        print(s['file'], 'open=',s['open_laws'],'holes=',s['todo_tokens'],'unsafe=',s['unsafe_markers'],'question_defs=',s['question_defs'],'foreign=',len(s['foreign_imports']))


if __name__=='__main__': main()
