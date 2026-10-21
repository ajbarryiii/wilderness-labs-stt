"""Fresh GPU process for test_runtime_fresh: P-memory of the integrated runtime (A6), the way
bench.p_clips samples it. Takes the gpu lock before any CUDA call and exits inside it (A9).

    timeout 600 gpu1bit/shared/python -m gpu1bit.tests.runtime_child --runtime hybrid --out F.json \
        [--kernel NAME] [--parts N] [--selection PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

from gpu1bit.shared import env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", required=True)
    p.add_argument("--kernel", default=None)
    p.add_argument("--parts", type=int, default=None)
    p.add_argument("--selection", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    code = 0
    with env.gpu_lock(f"H:runtime fresh-process VRAM {args.runtime}"):
        try:
            from inference.benchmark import energy_meter
            from nemo.utils import logging
            from gpu1bit import runtime
            from gpu1bit.shared import workloads
            logging.setLevel(logging.ERROR)
            meter = energy_meter()
            deadline = time.monotonic() + 20  # a previous holder's context may still be tearing down
            while meter.foreign_processes() and time.monotonic() < deadline:
                time.sleep(0.5)
            meter._guard()
            samples, errors, stop = [], [], threading.Event()

            def memory():
                ps = meter._nvml.nvmlDeviceGetComputeRunningProcesses(meter._handle)
                return max([int(q.usedGpuMemory) for q in ps if q.pid == os.getpid()] or [0])

            def sample():
                while not stop.wait(.01):
                    try:
                        samples.append((time.perf_counter(), memory()))
                    except Exception as exc:  # noqa: BLE001
                        errors.append(str(exc))
                        return
            thread = threading.Thread(target=sample, daemon=True)
            thread.start()
            started = time.perf_counter()
            model = runtime.load_binary_packed(env.MB1_EXPORT, runtime=args.runtime, kernel=args.kernel, parts=args.parts,
                                               selection=args.selection)
            import torch
            torch.cuda.synchronize()
            report = {"runtime": args.runtime, "kernel": runtime.runtime_of(model).kernel,
                      "sources": runtime.runtime_of(model).sources, "load_seconds": time.perf_counter() - started,
                      "parts": runtime.runtime_of(model).parts, "identity": runtime.runtime_of(model).identity,
                      "selection_sha256": runtime.runtime_of(model).selection_sha256,
                      "loaded_process_bytes": memory(), "clips": []}
            for rec in workloads.load()["dev_clips"]:
                begin = len(samples)
                texts = [runtime.transcribe(model, rec["audio_filepath"]) for _ in range(3)]
                torch.cuda.synchronize()
                report["clips"].append({"id": rec["id"], "duration": rec["duration"], "M": rec["M"], "texts": texts,
                                        **runtime.decoded(model), "after_bytes": memory(),
                                        "process_peak_including_first_bytes": max([v for _, v in samples[begin:]] + [memory()])})
            stop.set()
            thread.join()
            if errors:
                raise RuntimeError(errors)
            report["process_peak_bytes"] = max(v for _, v in samples)
            report["samples"] = len(samples)
            runtime.close_binary_packed(model)
            torch.cuda.synchronize()
            meter.close()
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"error: {exc}", file=sys.stderr)
            code = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":
    main()
