"""Fresh GPU process for test_x2_loaders_fresh: M1's deployment loader (gpu1bit.runtime_m1) called the way a
user calls it, under its own gpu lock (A9). Writes what the loader resolved: part count, runtime identity,
selection hash, the provider chain, one dev clip's transcript, and the frontend state during and after.

    timeout 600 gpu1bit/shared/python -m gpu1bit.tests.x2_loader_child --out F.json [--parts N] [--selection PATH]

The selection document otherwise comes from GPU1BIT_SELECTION or the pointer shared/selected.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from gpu1bit.shared import env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--parts", type=int, default=None)
    p.add_argument("--selection", default=None)
    args = p.parse_args()
    code = 0
    with env.gpu_lock("H:x2 M1 loader fresh process"):
        try:
            from nemo.utils import logging
            from gpu1bit import runtime_m1
            from gpu1bit.shared import fulldev, workloads
            logging.setLevel(logging.ERROR)
            kw = {k: v for k, v in (("parts", args.parts), ("selection", args.selection)) if v is not None}
            model = runtime_m1.load_ternary_packed(**kw)
            rt = runtime_m1.runtime_of(model)
            prov = rt.provider
            clip = workloads.load()["dev_clips"][0]
            fz = model.preprocessor.featurizer
            report = {"kwargs": kw, "env_selection": os.environ.get("GPU1BIT_SELECTION"), "parts": rt.parts,
                      "identity": rt.identity, "selection_sha256": rt.selection_sha256, "export": rt.export,
                      "provider": type(prov).__name__,
                      "partcount_n": prov.n if isinstance(prov, fulldev.PartCount) else None,
                      "inner": type(prov.provider).__name__ if isinstance(prov, fulldev.PartCount) else type(prov).__name__,
                      "frontend_during": {"dither": fz.dither, "pad_to": fz.pad_to},
                      "clip": clip["id"], "text": runtime_m1.transcribe(model, clip["audio_filepath"])}
            runtime_m1.close_ternary_packed(model)
            report["frontend_after"] = {"dither": fz.dither, "pad_to": fz.pad_to}
            report["record_removed"] = not hasattr(model, "_gpu1bit_m1_runtime")
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
