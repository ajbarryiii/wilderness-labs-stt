#!/usr/bin/env python3
"""Editable strict ternary CUDA clip-presence classifier; fixed evaluator in prepare.py."""
import time
PROCESS_START=time.monotonic()
import argparse
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from prepare import HERE,SCHEMA,FEATURES,load_cache,read_json,forward_reference

LAYERS=[{"channels":32,"kernel":5},{"channels":48,"kernel":3},{"channels":48,"kernel":3}]
LEARNING_RATE=0.003
BATCH_SIZE=128
WEIGHT_DECAY=1e-4
POS_WEIGHT=2.5
POS_MARGIN=6.0
POS_MARGIN_WEIGHT=0.02
POSITIVE_BATCH_FRACTION=0.70

def ternary(weight):
    scale=weight.detach().abs().mean().clamp_min(1e-8)
    codes=(weight.detach()/scale).round().clamp(-1,1)
    effective=codes*scale
    return weight+(effective-weight).detach(),codes,scale

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        parameters=[]
        previous=FEATURES
        for layer in LAYERS:
            value=torch.empty(layer["channels"],previous,layer["kernel"])
            nn.init.kaiming_uniform_(value,a=math.sqrt(5))
            parameters.append(nn.Parameter(value))
            previous=layer["channels"]
        head=torch.empty(1,previous*2)
        nn.init.kaiming_uniform_(head,a=math.sqrt(5))
        parameters.append(nn.Parameter(head))
        self.weights=nn.ParameterList(parameters)

    def forward(self,x,lengths):
        weights=[ternary(w)[0] for w in self.weights]
        return forward_reference(x,lengths,weights,{"layers":LAYERS})

def export(model,output,training):
    arrays={}
    packed=[]
    names=[f"conv{i}" for i in range(len(LAYERS))]+["head"]
    for name,weight in zip(names,model.weights):
        _,codes,scale=ternary(weight)
        code=codes.cpu().numpy().astype(np.int8)
        arrays[name+"_codes"]=code
        arrays[name+"_scale"]=np.asarray(scale.cpu(),dtype=np.float32)
        unsigned=(code.reshape(-1)+1).astype(np.uint8)
        unsigned=np.pad(unsigned,(0,(-len(unsigned))%4),constant_values=1)
        packed.append((unsigned[0::4]|unsigned[1::4]<<2|unsigned[2::4]<<4|unsigned[3::4]<<6).tobytes())
    np.savez(output/"weights.npz",**arrays)
    (output/"weights.2bit").write_bytes(b"".join(packed))
    config={"schema_version":SCHEMA,"model_type":"causal_ternary_clip_cnn","layers":LAYERS,
        "activation_precision":"fp32_reference","learned_bias":False,"normalization":"nonaffine channel RMS",
        "pooling":"valid-length masked whole-clip mean+max; not a streaming VAD decision",
        "scale_metadata":"one FP32 absmean scale per tensor, derived from master weights; no learned affine parameters",
        "packed_format":"2bits/code, mapping -1=0,0=1,+1=2,3 invalid; layer order thenhead, each tensor padded to4",
        "training":training}
    (output/"model.json").write_text(json.dumps(config,indent=2,allow_nan=False)+"\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--train-seconds",type=float,required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--max-steps",type=int,default=0,help="Smoke verification only; scored trials leave zero.")
    args=parser.parse_args()
    deadline=PROCESS_START+args.train_seconds
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory; no CPU/synthetic fallback")
    device=torch.device("cuda:0")
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    benchmark=read_json(HERE/"benchmark.json")
    data=load_cache(benchmark)
    indices=np.flatnonzero(data["splits"]=="train")
    x=torch.from_numpy(data["x"][indices]).to(device)
    lengths=torch.from_numpy(data["lengths"][indices]).to(device)
    y=torch.from_numpy(data["y"][indices]).to(device)
    positive_indices=torch.nonzero(y>0.5,as_tuple=False).flatten()
    negative_indices=torch.nonzero(y<=0.5,as_tuple=False).flatten()
    if len(positive_indices)<=0 or len(negative_indices)<=0:
        raise RuntimeError("Training split must contain both speech and non-speech")
    mask=torch.arange(x.shape[-1],device=device)[None,:]<lengths[:,None]
    model=Classifier().to(device)
    num_parameters=sum(p.numel() for p in model.parameters())
    if num_parameters>50000:raise ValueError("Classifier exceeds50k parameter budget")
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    pos_weight=torch.tensor(POS_WEIGHT,device=device)
    torch.cuda.reset_peak_memory_stats()
    steps=0;losses=[];started=time.monotonic();next_log=started
    initial_weights=[p.detach().clone() for p in model.parameters()]
    while time.monotonic()<deadline-20:
        positive_count=max(1,min(BATCH_SIZE-1,int(round(BATCH_SIZE*POSITIVE_BATCH_FRACTION))))
        negative_count=BATCH_SIZE-positive_count
        positive_batch=positive_indices[torch.randint(len(positive_indices),(positive_count,),device=device)]
        negative_batch=negative_indices[torch.randint(len(negative_indices),(negative_count,),device=device)]
        batch=torch.cat([positive_batch,negative_batch])
        batch=batch[torch.randperm(BATCH_SIZE,device=device)]
        features=x[batch]
        # Random clip-wide gain varies speech and noise equally; no dev-derived statistics.
        gain=torch.empty(BATCH_SIZE,1,1,device=device).uniform_(-0.5,0.5)
        features=(features+gain)*mask[batch,None,:]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=model(features,lengths[batch])
            logits_f=logits.float()
            labels=y[batch]
            bce=F.binary_cross_entropy_with_logits(logits_f,labels,pos_weight=pos_weight)
            positive_margin=(F.relu(POS_MARGIN-logits_f)*labels).mean()
            loss=bce+POS_MARGIN_WEIGHT*positive_margin
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        steps+=1
        if steps%10==0:
            value=float(loss.detach())
            if not math.isfinite(value):raise RuntimeError("Nonfinite CUDA training loss")
            losses.append(value)
        if time.monotonic()>=next_log:
            print(json.dumps({"step":steps,"loss":float(loss.detach()),"elapsed_seconds":time.monotonic()-PROCESS_START}),flush=True)
            next_log=time.monotonic()+15
        if args.max_steps and steps>=args.max_steps:break
    torch.cuda.synchronize()
    if steps<=0:raise RuntimeError("No CUDA updates completed in training budget")
    change=sum(float((p.detach()-initial).abs().sum()) for p,initial in zip(model.parameters(),initial_weights))
    if change<=0:raise RuntimeError("Training did not change any parameters")
    training={"training_cuda_verified":True,"training_device":str(device),"gpu_name":torch.cuda.get_device_name(device),
        "training_steps":steps,"training_seconds":time.monotonic()-started,
        "process_seconds":time.monotonic()-PROCESS_START,"peak_vram_mb":torch.cuda.max_memory_allocated()/2**20,
        "initial_train_loss":losses[0] if losses else float(loss.detach()),"final_train_loss":float(loss.detach()),
        "master_weight_l1_change":change,"seed":args.seed,"batch_size":BATCH_SIZE,
        "optimizer":"AdamW","learning_rate":LEARNING_RATE,"training_activation_precision":"BF16 autocast",
        "positive_loss_weight":POS_WEIGHT,"positive_margin":POS_MARGIN,
        "positive_margin_weight":POS_MARGIN_WEIGHT,
        "positive_batch_fraction":POSITIVE_BATCH_FRACTION,
        "num_parameters":num_parameters,"torch_version":torch.__version__,
        "cuda_version":torch.version.cuda,"train_samples":len(indices)}
    model.to("cpu")
    export(model,output,training)
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"training":training},output/"training-state.pt")
    (output/"training.json").write_text(json.dumps(training,indent=2,allow_nan=False)+"\n")
    print(json.dumps(training,allow_nan=False),flush=True)

if __name__=="__main__":main()
