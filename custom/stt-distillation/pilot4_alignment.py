import json
from pathlib import Path
import numpy as np
import torch
from common import ALPHABET, ART
from pilot4_model import PilotModel
p=ART/'training-repair/pilot4-fp-blankinit'
obj=torch.load(p/'weights.pt',map_location='cpu',weights_only=False)
m=PilotModel(obj['model'],'fp');m.load_state_dict(obj['state_dict']);m.cuda().eval()
rows=json.loads((p/'subset.json').read_text())['rows']
with torch.inference_mode():
 for r in rows:
  x=torch.from_numpy(np.load(r['features'])).cuda()[None]
  with torch.autocast('cuda',dtype=torch.bfloat16):z=m(x)
  ids=z[0].argmax(-1).tolist();last=None;events=[]
  for t,c in enumerate(ids):
   if c and c!=last:events.append([t*20,ALPHABET[c],round(float(z[0,t].float().softmax(-1)[c]),3)])
   last=c
  pred=''.join(e[1] for e in events)
  if pred.strip()!=r['text'].lower().strip():
   print(json.dumps({'id':r['id'],'text':r['text'],'pred':pred,'early_events':events[:12],'first_feature_std':float(x[0,:,0].std()),'first_feature_mean':float(x[0,:,0].mean())}))
