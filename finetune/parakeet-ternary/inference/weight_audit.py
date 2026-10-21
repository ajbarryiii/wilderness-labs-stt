"""CPU-only exact ternary sign/sparsity audit of the exported encoder."""
import json
import time
import numpy as np
from safetensors import safe_open
import paths


def main():
    paths.require_mount()
    export=paths.RUNS/'main-M1-P2-lr5e-4/export'
    manifest=json.loads((export/'manifest.json').read_text())
    out=paths.ARTIFACTS/'kernel-runs'/f'{time.strftime("%Y%m%dT%H%M%S")}-weight-audit'
    out.mkdir();rows=[]
    with safe_open(export/manifest['file'],framework='np') as f:
        for name,layer in manifest['quantized_layers'].items():
            n,k=layer['shape'];packed=f.get_tensor(name+'.codes')
            c=((packed[:,:,None] >> np.array([0,2,4,6],dtype=np.uint8))&3).reshape(n,-1)[:,:k]
            assert not np.any(c==3)
            counts=np.bincount(c.ravel(),minlength=3)
            positive=(c==1);negative=(c==2);zero=(c==0)
            one_sided=~positive.any(1)|~negative.any(1)
            groups={}
            for width in [4,16,64,128]:
                v=c[:,:k//width*width].reshape(n,-1,width)
                groups[str(width)]={
                    'groups':int(v.shape[0]*v.shape[1]),
                    'one_sided':int((~(v==1).any(-1)|~(v==2).any(-1)).sum()),
                    'zero_free':int((v!=0).all(-1).sum()),
                    'all_equal':int((v==v[...,:1]).all(-1).sum()),
                    'all_zero':int((v==0).all(-1).sum())}
                if width==4:groups[str(width)]['two_four_eligible']=int(((v==0).sum(-1)>=2).sum())
            row={'name':name,'shape':[n,k],'counts_zero_positive_negative':counts.tolist(),
                 'one_sided_rows':int(one_sided.sum()),'zero_free_rows':int((~zero.any(1)).sum()),
                 'all_zero_rows':int(zero.all(1).sum()),
                 'one_sided_columns':int((~positive.any(0)|~negative.any(0)).sum()),'groups':groups}
            rows.append(row)
    counts=np.array([r['counts_zero_positive_negative'] for r in rows]).sum(0)
    summary={'export_sha256':manifest['sha256'],'layers':len(rows),'weights':int(counts.sum()),
             'counts_zero_positive_negative':counts.tolist(),
             'fractions_zero_positive_negative':(counts/counts.sum()).tolist(),
             'one_sided_layers':sum(r['counts_zero_positive_negative'][1]==0 or r['counts_zero_positive_negative'][2]==0 for r in rows),
             'one_sided_rows':sum(r['one_sided_rows'] for r in rows),
             'zero_free_rows':sum(r['zero_free_rows'] for r in rows),
             'one_sided_columns':sum(r['one_sided_columns'] for r in rows),
             'groups':{str(w):{key:sum(r['groups'][str(w)][key] for r in rows)
                              for key in rows[0]['groups'][str(w)]} for w in [4,16,64,128]}}
    (out/'audit.json').write_text(json.dumps({'summary':summary,'layers':rows},indent=2)+'\n')
    print('RESULTS',out);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
