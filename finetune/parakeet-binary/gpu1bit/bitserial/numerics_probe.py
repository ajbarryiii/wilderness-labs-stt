"""Where does the bitserial kernel's error come from on real inputs? (C diagnostic, dev fixtures only)

Runs one fixture layer through the kernel under several configs and through CPU emulations of
summation orders, and reports A3 metrics (relative Frobenius, elementwise guard) vs FP64, next to
M1's thresholds for that layer and utterance.

  timeout 600 .../shared/python .../bitserial/run_locked.py C:numerics -m gpu1bit.bitserial.numerics_probe \
      --utterance <i> --layer encoder.layers.10.feed_forward1.linear1
"""
from __future__ import annotations

import argparse
import json

import torch

from gpu1bit.shared import api, fixtures

from .kernels import Config
from .layer import BitSerialLayer


def emulate(x: torch.Tensor, signs: torch.Tensor, scale: torch.Tensor, *, split: int, bk: int, inner: int,
            two_sum: bool = False, middle: int = 0, fast2sum: bool = False) -> torch.Tensor:
    """CPU FP32 emulation of the kernel's order: per split, k-tiles of bk, inner blocks of `inner`
    summed sequentially into a fresh partial; partials added to acc (plain, 2Sum- or
    Fast2Sum-compensated), or, with middle > 0, first into a middle accumulator flushed into acc
    every `middle` k; splits summed in rank order; times scale."""
    m, k = x.shape
    prod = (x[:, None, :] * signs[None, :, :].float())  # exact in FP32
    tiles = -(-k // bk)
    per = -(-tiles // split)
    total = None
    shape = prod.shape[:2]
    for s in range(split):
        lo, hi = s * per * bk, min(k, (s + 1) * per * bk)
        acc, comp, mid = torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)
        for b0 in range(lo, hi, inner):
            part = torch.zeros(shape)
            for j in range(b0, min(hi, b0 + inner)):
                part = part + prod[:, :, j]
            if middle:
                mid = mid + part
                if (b0 - lo + inner) % middle == 0 or b0 + inner >= hi:
                    acc, mid = acc + mid, torch.zeros(shape)
            elif two_sum:
                t = acc + part
                bb = t - acc
                comp = comp + ((acc - (t - bb)) + (part - bb))
                acc = t
            elif fast2sum:
                t = acc + part
                comp = comp + (part - (t - acc))
                acc = t
            else:
                acc = acc + part
        acc = acc + comp
        total = acc if total is None else total + acc
    return total * scale[None, :]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--utterance', type=int, default=None)
    p.add_argument('--layer', default='encoder.layers.10.feed_forward1.linear1')
    p.add_argument('--rows', type=int, default=32, help='rows for the CPU emulation')
    p.add_argument('--cols', type=int, default=128, help='columns for the CPU emulation')
    args = p.parse_args()
    idx = fixtures.index()['utterances']
    if args.utterance is None:  # the longest fixture utterance
        args.utterance = max(range(len(idx)), key=lambda i: idx[i].get('T', 0))
    w = fixtures.mb1_layers([args.layer])[args.layer]
    x = fixtures.utterance_inputs(args.utterance)[args.layer]
    kind = 'conv' if w['kind'] in ('conv', 'pointwise_conv1d') else 'linear'
    limits = fixtures.m1_thresholds()[idx[args.utterance]['file']][args.layer]
    codes, scale, k = w['codes'], w['scale'], w['K']
    ref = api.reference(kind, x, codes, scale, None, k)
    layer = BitSerialLayer.prepare(codes, scale, None, k, 'cuda')
    out = {'layer': args.layer, 'utterance': idx[args.utterance]['id'], 'x_shape': list(x.shape), 'limits': limits,
           'gpu': {}}
    with torch.inference_mode():
        for cfg in [None, Config(64, 128, 32, 8, 4, 1), Config(64, 128, 32, 8, 4, 8), Config(16, 64, 64, 4, 4, 6)]:
            layer.config_override = cfg
            y = api.call(layer, kind, x)
            out['gpu'][str(cfg)] = api.errors(kind, x, y, ref, scale)
        layer.config_override = None
        y = api.call(layer, kind, x)
        rows_x = (x.transpose(1, 2) if kind == 'conv' else x).reshape(-1, k)
        rows_y = (y.transpose(1, 2) if kind == 'conv' else y).reshape(-1, codes.shape[0]).double()
        rows_ref = (ref.transpose(1, 2) if kind == 'conv' else ref).reshape(-1, codes.shape[0])
        l1 = rows_x.abs().sum(1).double()
        ratio = (rows_y - rows_ref).abs() / (scale.cuda().double().abs()[None, :] * l1[:, None])
        flat = ratio.flatten().topk(5)
        worst = []
        for v, i in zip(flat.values.tolist(), flat.indices.tolist()):
            r, c = divmod(i, codes.shape[0])
            xr = rows_x[r]
            worst.append({'row': r, 'col': c, 'ratio': v, 'y': rows_y[r, c].item(), 'ref': rows_ref[r, c].item(),
                          'row_l1': l1[r].item(), 'row_max': xr.abs().max().item(), 'row_rms': xr.pow(2).mean().sqrt().item(),
                          'row_mean': xr.mean().item()})
        out['worst'] = worst
        # CPU emulation on the worst rows plus a sample, worst columns plus a sample.
        rws = sorted({w_['row'] for w_ in worst} | set(range(0, rows_x.shape[0], max(1, rows_x.shape[0] // args.rows))))[:args.rows]
        cls = sorted({w_['col'] for w_ in worst} | set(range(0, codes.shape[0], max(1, codes.shape[0] // args.cols))))[:args.cols]
        xs = rows_x[rws].cpu()
        signs = api.unpack_signs(codes, k)[cls]
        sc = scale[cls]
        ref_sub = rows_ref[rws][:, cls].cpu()
        l1s = l1[rws].cpu()
        def metrics(yy):
            d = (yy.double() - ref_sub)
            return {'frob': (d.norm() / ref_sub.norm()).item(),
                    'guard': (d.abs() / (sc.double().abs()[None, :] * l1s[:, None])).max().item()}
        out['gpu_subset'] = metrics(rows_y[rws][:, cls].cpu())
        out['emulated'] = {}
        schemes = {}
        for split in (1, 2, 3):
            schemes[f'b16_split{split}'] = dict(split=split, bk=32, inner=16)
            schemes[f'b16_mid128_split{split}'] = dict(split=split, bk=32, inner=16, middle=128)
            schemes[f'b16_mid256_split{split}'] = dict(split=split, bk=32, inner=16, middle=256)
            schemes[f'b16_fast2sum_split{split}'] = dict(split=split, bk=32, inner=16, fast2sum=True)
            schemes[f'b32_fast2sum_split{split}'] = dict(split=split, bk=32, inner=32, fast2sum=True)
            schemes[f'b16_2sum_split{split}'] = dict(split=split, bk=32, inner=16, two_sum=True)
        for name, kw in schemes.items():
            out['emulated'][name] = metrics(emulate(xs, signs, sc, **kw))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
