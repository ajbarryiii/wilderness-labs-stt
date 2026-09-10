"""Bounded CUDA CTC training. Only recipe.json is proposed during research."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core import AcousticModel, digest, encode, export, save, validate_recipe
from data import load


def train(args):
    started = time.monotonic()
    if args.train_seconds < 30:
        raise ValueError('Training envelope must be >=30s')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA training required')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    recipe = validate_recipe(json.loads(Path(args.recipe).read_text()))
    manifest, cache = load(args.cache)
    ids = [i for i,r in enumerate(manifest['rows']) if r['split']=='train' and r['label']]
    # Overfit mode is a separate engineering check, never ranked as accuracy evidence.
    if args.overfit_samples:
        ids = sorted(ids, key=lambda i:manifest['rows'][i]['num_samples'])[:args.overfit_samples]
    xs = {i:torch.from_numpy(cache[str(i)]).cuda() for i in ids}
    ys = {i:encode(manifest['rows'][i]['text']) for i in ids}
    model = AcousticModel(recipe).cuda()
    initial = [p.detach().clone() for p in model.parameters()]
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'])
    first = final = None
    steps = 0
    rng = np.random.default_rng(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    work_start = time.monotonic()
    deadline = started + args.train_seconds - 10
    while time.monotonic() < deadline:
        batch = rng.choice(ids, size=min(recipe['batch_size'],len(ids)), replace=False).tolist()
        lengths = torch.tensor([xs[i].shape[-1] for i in batch],dtype=torch.long)
        x = torch.zeros(len(batch),80,int(lengths.max()),device='cuda')
        targets = []
        for j,i in enumerate(batch):
            x[j,:,:lengths[j]] = xs[i]
            targets.extend(ys[i])
        if rng.random() < recipe['noise_probability']:
            # Feature perturbation is training augmentation, not a field-noise claim.
            x += torch.randn_like(x)*0.05
        target_lengths = torch.tensor([len(ys[i]) for i in batch],dtype=torch.long)
        output_lengths = (lengths+1)//2
        for i,n in zip(batch,output_lengths.tolist()):
            if n < len(ys[i])+sum(a==b for a,b in zip(ys[i],ys[i][1:])):
                raise ValueError('CTC path infeasible')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits = model(x)
        loss = F.ctc_loss(logits.float().log_softmax(-1).transpose(0,1),
                         torch.tensor(targets,dtype=torch.long,device='cuda'),
                         output_lengths,target_lengths,zero_infinity=False)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite CTC loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['grad_clip'],error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):
                a.lerp_(b,1-recipe['ema_decay'])
        final = float(loss.detach())
        first = final if first is None else first
        steps += 1
        if steps % 1000 == 0:
            print(json.dumps(dict(step=steps,ctc_loss=final,seconds=time.monotonic()-started)),flush=True)
    torch.cuda.synchronize()
    change = sum(float((p-q).abs().sum()) for p,q in zip(model.parameters(),initial))
    if not steps or not change:
        raise RuntimeError('No actual optimizer work')
    export(ema,out)
    save(out/'training.json',dict(seed=args.seed,training_cuda_verified=True,
        gpu=torch.cuda.get_device_name(), training_steps=steps,
        optimizer_seconds=time.monotonic()-work_start, initial_loss=first,final_loss=final,
        master_weight_l1_change=change, peak_vram_mb=torch.cuda.max_memory_allocated()/2**20,
        train_samples=len(ids),overfit_samples=args.overfit_samples,
        cache_lock_sha256=digest(Path(args.cache)/'lock.json'),
        torch_version=torch.__version__,recipe=recipe))
    print(json.dumps(dict(steps=steps,initial_loss=first,final_loss=final)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--recipe',default=str(Path(__file__).with_name('recipe.json')))
    p.add_argument('--cache',default=str(Path(__file__).with_name('cache')))
    p.add_argument('--output',required=True)
    p.add_argument('--train-seconds',type=int,default=300)
    p.add_argument('--seed',type=int,default=20260909)
    p.add_argument('--overfit-samples',type=int,default=0)
    train(p.parse_args())
