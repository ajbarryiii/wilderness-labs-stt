"""A5 unique-storage audit of B-hybrid-x2 (plans/x2-default.md; budget 83,047,219 B), H's method unchanged.

The same procedure as gpu1bit.shared.selection.audit_variant: build the variant (here hybrid_x2.exact_model_x2
with H's fulldev.PartCount), transcribe
the three dev clips twice (eager warm-up + capture, then replay), run gpu1bit.shared.audit.audit with the
provider as an extra root and the MB1 layer scales, then selection.reclassify_decoder_side. The report goes
to /mnt/hd/.../gpu1bit/B/x2/audit-B-hybrid-x2.json (B's artifacts; H's selection runs write their own).

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.audit_x2
"""
from __future__ import annotations

import json


def main() -> None:
    import torch
    import evaluate as ev
    from nemo.utils import logging
    from gpu1bit.int8_mma import gpulock, hybrid_x2
    from gpu1bit.int8_mma.test_exact_pipeline_x2 import adapter_class
    from gpu1bit.shared import audit, bench, env, fixtures, selection, workloads
    env.require_shared_wrapper()
    logging.setLevel(logging.ERROR)
    hybrid_x2.parts()
    clips = workloads.load()["dev_clips"]
    scales = {n: w["scale"] for n, w in fixtures.mb1_layers().items()}
    out = env.ARTIFACTS_ROOT / "B" / "x2"
    out.mkdir(parents=True, exist_ok=True)
    adapter, adapter_name = adapter_class()
    with gpulock.locked("B:A5 audit B-hybrid-x2") as lock, torch.inference_mode(), ev.strict_fp32("cuda"):
        with hybrid_x2.exact_model_x2(partcount=adapter) as (model, info):
            for clip in clips:
                audio, length = ev.audio_batch([clip])
                for _ in range(2):
                    bench.recognize(model, audio, length)
            torch.cuda.synchronize()
            entries = len(model.forward.entries) if hasattr(model.forward, "entries") else None
            report = audit.audit(model, list(scales), {"provider": info["projection"]}, scales=scales)
            report["exact_slice_calls"] = info["inner"].calls.get("exact_slices", 0)
            held = env.check_hold(lock, margin_s=0)
        selection.reclassify_decoder_side(report)
        report.update(variant="B-hybrid-x2", adapter=adapter_name, graph_entries=entries, lock_held_s=held,
                      wrapper=env.wrapper())
        path = out / "audit-B-hybrid-x2.json"
        path.write_text(json.dumps(report, indent=1, default=str) + "\n")
        print(json.dumps({k: report[k] for k in ("variant", "adapter", "weight_code_bytes", "code_budget_bytes", "dense_bytes",
                                                  "scale_bytes", "within_budget", "graph_entries", "exact_slice_calls")}))
        print("RESULTS", path)


if __name__ == "__main__":
    main()
