"""Replicate the known-good PTX warp program and reduce INT32 inside a CTA.

The transform is intentionally strict: complete K groups only, pointer-only
ABI, one conversion point, and no extra scratch. No production selection.
"""
import re


def grouped_ptx(ptx,*,groups,shared,bk,split,k):
    if groups not in (2,4) or k%(groups*bk*split):raise ValueError('K must contain complete CTA groups')
    entry=re.search(r'\.visible\s+\.entry\s+\w+\s*\((.*?)\)\s*\.reqntid\s+(\d+)',ptx,re.S)
    if entry is None:raise ValueError('expected single entry with reqntid')
    params=[p.strip().split()[-1] for p in entry.group(1).split(',')]
    threads=int(entry.group(2));start=ptx.index('{',entry.end())
    if threads!=128:raise ValueError('only the original four-warp program is supported')
    if '%warpid' in ptx or '%ntid.x' in ptx:raise ValueError('unhandled physical warp/block dependency')
    conversions=list(re.finditer(r'cvt\.rn\.f32\.s32\s+[^,;]+,\s*(%\w+)\s*;',ptx))
    regs=list(dict.fromkeys(c.group(1) for c in conversions))
    if not regs or len(regs)!=len(conversions):raise ValueError('unexpected conversion pattern')
    conversion_start=conversions[0].start()
    if 'mma.sync' in ptx[conversion_start:]:raise ValueError('integer work after conversion')
    # Give every replicated warp program its own shared-memory staging area.
    ptx=ptx[:start]+ptx[start:].replace('%tid.x','%grp_tid').replace('global_smem','%grp_smem')
    definitions='\n'+''.join(f' .reg .b32 %{name};\n' for name in
        ['grp_raw','grp_tid','grp_id','grp_base','grp_smem','grp_row','grp_addr','grp_tmp'])
    definitions+=' .reg .b64 %grp_pointer;\n .reg .pred %grp_other;\n'
    ptx=ptx[:start+1]+definitions+ptx[start+1:]
    declarations=list(re.finditer(r'\.reg[^;]+;',ptx))
    pos=declarations[-1].end()
    initialize=f'''
 mov.u32 %grp_raw, %tid.x;
 and.b32 %grp_tid, %grp_raw, {threads-1};
 shr.u32 %grp_id, %grp_raw, 7;
 mov.u32 %grp_base, global_smem;
 mad.lo.u32 %grp_smem, %grp_id, {shared}, %grp_base;
 setp.ne.u32 %grp_other, %grp_id, 0;
 mad.lo.u32 %grp_row, %grp_tid, {4*len(regs)}, %grp_base;
 add.u32 %grp_row, %grp_row, {groups*shared};
 mad.lo.u32 %grp_addr, %grp_id, {threads*4*len(regs)}, %grp_row;
'''
    ptx=ptx[:pos]+initialize+ptx[pos:]
    for parameter in [params[0],params[2]]:
        pattern=r'ld\.param\.[ub]64\s+(%\w+),\s*\['+re.escape(parameter)+r'\]\s*;'
        def shift(match):
            register=match.group(1)
            return match.group(0)+f'\n mad.wide.u32 %grp_pointer, %grp_id, {bk*split}, {register};\n mov.b64 {register}, %grp_pointer;'
        ptx,count=re.subn(pattern,shift,ptx)
        if count!=1:raise ValueError('unexpected pointer load count')
    reduction='\n'
    for i,register in enumerate(regs):reduction+=f' @%grp_other st.shared.b32 [%grp_addr+{4*i}], {register};\n'
    reduction+=' bar.sync 0;\n @%grp_other ret;\n'
    for group in range(1,groups):
        for i,register in enumerate(regs):
            reduction+=f' ld.shared.b32 %grp_tmp, [%grp_row+{group*threads*4*len(regs)+4*i}];\n add.s32 {register}, {register}, %grp_tmp;\n'
    pos=ptx.index('cvt.rn.f32.s32')
    before,after=ptx[:pos],ptx[pos:]
    after=re.sub(r'bar\.sync\s+0\s*;',f'bar.sync 1, {threads};',after)
    ptx=before+reduction+after
    ptx=re.sub(r'(\.reqntid\s+)'+str(threads)+r'\b',lambda m:m.group(1)+str(threads*groups),ptx,count=1)
    return ptx,groups*shared+groups*threads*4*len(regs)
