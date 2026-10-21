"""B-hybrid smoke check on the dev clips (workloads.json dev_clips; dev audio only): encoder output vs
R-hybrid and R-expanded (lengths, rel. Frobenius on valid frames, max abs), full recognition (text,
tokens, timestamps), and how HybridProjection served its calls (deferred / prequantized / quantized).

    timeout 600 gpu1bit/shared/python -m gpu1bit.int8_mma.smoke_hybrid
"""
from __future__ import annotations

import contextlib
import json

import torch

from gpu1bit.int8_mma import gpulock


def main():
    with gpulock.locked("B:B-hybrid smoke"), torch.inference_mode(), contextlib.ExitStack() as stack:
        import evaluate as ev
        from nemo.utils import logging
        from gpu1bit.int8_mma import paired  # noqa: F401  (patches bench.variant_model)
        from gpu1bit.shared import bench, workloads
        logging.setLevel(logging.ERROR)
        stack.enter_context(ev.strict_fp32("cuda"))
        names = ["R-expanded", "R-hybrid", "B-hybrid"]
        models, info = {}, {}
        for n in names:
            models[n], info[n] = stack.enter_context(bench.variant_model(n, max_graphs=4))
        for clip in workloads.load()["dev_clips"]:
            audio, length = ev.audio_batch([clip])
            feats, flen = models["R-expanded"].preprocessor(input_signal=audio.cuda(), length=length.cuda())
            outs = {n: tuple(t.clone() for t in models[n].encoder(audio_signal=feats, length=flen)) for n in names}
            t = int(outs["R-expanded"][1][0])
            rows = {}
            for n in names:
                ref = outs["R-expanded"][0][..., :t]
                e = outs[n][0][..., :t]
                rows[n] = {"lengths_equal": bool(torch.equal(outs[n][1], outs["R-expanded"][1])),
                           "rel_frobenius_vs_R-expanded": ((e - ref).norm() / ref.norm()).item(),
                           "max_abs_vs_R-expanded": (e - ref).abs().max().item()}
            for m in models.values():
                m.forward.entries.clear()
            texts = {}
            for n in names:
                texts[n] = {"text": bench.recognize(models[n], audio, length), **bench.decoded_state(models[n])}
            for n in names:
                rows[n]["transcript_identical_to_R-expanded"] = texts[n] == texts["R-expanded"]
            print(json.dumps({"clip": clip["id"], "M": clip.get("M"), "rows": rows,
                              "b_hybrid_calls": dict(info["B-hybrid"]["projection"].calls),
                              "text": texts["B-hybrid"]["text"][:80]}), flush=True)


if __name__ == "__main__":
    main()
