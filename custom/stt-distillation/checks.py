"""Meaningful correctness gates and full-size CUDA resource profiling."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import ART, ALPHABET, encode, decode, digit_string, save
from model import Model, set_precision, Weight


def small():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    cfg = dict(width=32, depth=2, heads=2, context=16, dropout=0.0, checkpoint=False)
    m = Model(cfg, "ternary").eval()
    x = torch.randn(1, 80, 80)
    with torch.no_grad():
        a = m(x)
        b = m(x[:, :, :40])
        assert torch.allclose(a[:, :20], b, atol=2e-5), "Future audio leaked"
        for p in m.modules():
            if isinstance(p, Weight):
                q = p.effective().clone()
                p.weight.copy_(q)
                p.precision = "fp"
        assert torch.allclose(a, m(x), atol=2e-5), (
            "Export reconstruction changed outputs"
        )
    assert digit_string("zero five 007") == "05007"
    assert digit_string("zero five") != digit_string("zero fifty")
    assert decode([1, 1, 0, 1]) == "aa"
    # Check CTC actually learns a tiny target under the quantized forward.
    m = Model(cfg, "ternary").train()
    opt = torch.optim.AdamW(m.parameters(), lr=0.003)
    target = torch.tensor(encode("one"))
    losses = []
    for i in range(80):
        opt.zero_grad()
        z = m(x).float().log_softmax(-1).transpose(0, 1)
        loss = F.ctc_loss(
            z,
            target,
            torch.tensor([40]),
            torch.tensor([len(target)]),
            zero_infinity=False,
        )
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.5, losses[-1]
    print(
        json.dumps(
            dict(correctness="passed", initial_loss=losses[0], final_loss=losses[-1])
        ),
        flush=True,
    )


def profile(cfg, precision, out):
    torch.set_num_threads(4)
    torch.manual_seed(20260910)
    torch.cuda.reset_peak_memory_stats()
    m = Model(cfg, precision).cuda().train()
    n = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, foreach=False)
    times = []
    last = None
    for frames in [400, 1200, 1200]:
        x = torch.randn(1, 80, frames, device="cuda")
        target = torch.tensor(encode("this is a test of one two three"), device="cuda")
        started = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m(x)
        loss = F.ctc_loss(
            z.float().log_softmax(-1).transpose(0, 1),
            target,
            torch.tensor([(frames + 1) // 2]),
            torch.tensor([len(target)]),
            zero_infinity=False,
        )
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        last = float(loss.detach())
    peak = torch.cuda.max_memory_reserved() / 2**30
    obj = dict(
        precision=precision,
        parameters=n,
        step_seconds=times,
        peak_reserved_gib=peak,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        finite_loss=last,
        cuda_verified=True,
    )
    assert peak < 28, obj
    save(out, obj)
    print(json.dumps(obj), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["small", "fp", "ternary"])
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    small() if a.mode == "small" else profile(
        json.loads(Path(__file__).with_name("pilot.json").read_text())["model"],
        a.mode,
        a.output,
    )
