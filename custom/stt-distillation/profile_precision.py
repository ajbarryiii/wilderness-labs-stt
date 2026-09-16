"""Profile identical dense training operations with and without ternary simulation."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import ART, encode, save, storage
from model import Model


def profile_arm(cfg, precision):
    torch.manual_seed(912)
    model = Model(cfg, precision).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    x = torch.randn(1, 80, 600, device="cuda")
    y = torch.tensor(encode("this is a test of speech recognition"), device="cuda")

    def update():
        opt.zero_grad(set_to_none=True)
        for _ in range(4):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x)
            loss = F.ctc_loss(
                z.float().log_softmax(-1).transpose(0, 1),
                y,
                torch.tensor([z.shape[1]]),
                torch.tensor([len(y)]),
            )
            (loss / 4).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()

    for _ in range(4):
        update()
    torch.cuda.synchronize()
    times = []
    for _ in range(8):
        tick = time.perf_counter()
        update()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - tick)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        update()
        torch.cuda.synchronize()
    events = [
        dict(
            name=e.key,
            count=e.count,
            self_cuda_us=getattr(e, "self_device_time_total", 0),
            self_cpu_us=e.self_cpu_time_total,
        )
        for e in prof.key_averages()
    ]
    return dict(
        mean_update_seconds=float(np.mean(times)),
        timings=times,
        top_cuda=sorted(events, key=lambda e: e["self_cuda_us"], reverse=True)[:25],
        # These operators also occur outside quantization. Compare the FP control;
        # do not sum both aten operator rows and the underlying CUDA kernel rows.
        quantization_ops=[
            e
            for e in events
            if e["name"]
            in [
                "aten::abs",
                "aten::mean",
                "aten::round",
                "aten::clamp",
                "aten::div",
                "aten::mul",
                "aten::sub",
                "aten::add",
                "aten::_to_copy",
            ]
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="kernel-profile")
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a simple directory name")
    storage()
    out = ART / "training-repair" / args.name
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path(__file__).with_name("pilot.json").read_text())["model"]
    torch.set_num_threads(4)
    results = {}
    for precision in ["fp", "ternary"]:
        results[precision] = profile_arm(cfg, precision)
        save(out / "profile.json", results)
        print(json.dumps({precision: results[precision]}), flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
