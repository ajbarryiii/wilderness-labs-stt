"""Independent process evaluates only a hashed exported model and frozen dev rows."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from common import read_manifest, save, scores, decode, digit_string, medical_terms
from export import load_export


@torch.inference_mode()
def evaluate(folder, output, limit=0):
    torch.set_num_threads(4)
    m, meta = load_export(folder)
    rows = [r for r in read_manifest()["rows"] if r["split"] == "development"]
    if limit:
        rows = rows[:limit]
    preds = []
    start = time.monotonic()
    for r in rows:
        x = torch.from_numpy(np.load(r["features"])).unsqueeze(0).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m(x)
        h = decode(z.argmax(-1)[0].tolist())
        preds.append(
            dict(id=r["id"], domain=r["domain"], reference=r["text"], hypothesis=h)
        )
    domains = {}
    for d in sorted({r["domain"] for r in preds}):
        p = [r for r in preds if r["domain"] == d]
        s = scores([(r["reference"], r["hypothesis"]) for r in p])
        if d == "medical_symptoms":
            s["medical_terms"] = medical_terms(
                [(r["reference"], r["hypothesis"]) for r in p]
            )
        if d == "digits":
            s["digit_sequence_accuracy"] = sum(
                digit_string(r["reference"]) == digit_string(r["hypothesis"]) for r in p
            ) / len(p)
        domains[d] = s
    obj = dict(
        step=meta["step"],
        export_sha256=meta["weights_sha256"],
        export_bytes=meta["bytes"],
        domains=domains,
        predictions=preds,
        evaluation_seconds=time.monotonic() - start,
        scope="Development diagnostics; no clinical entity labels, confidence interval, noise, deployment energy or natural quantity qualification.",
    )
    save(output, obj)
    print(json.dumps({k: v for k, v in obj.items() if k != "predictions"}), flush=True)
    return obj


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    evaluate(a.folder, a.output, a.limit)
