"""Read-only train/eval-mode diagnostic; no optimizer or weight updates."""

import collections, gc, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

run = Path(
    "/mnt/hd/wilderness-labs-stt/stt-distillation/runs/pilot-8h-20260910T044750Z"
)
sys.path.insert(0, str(run / "code"))
from common import encode, decode, save
from export import load_export

rows = json.loads((run / "manifest.json").read_text())["rows"]
chosen = [
    r
    for d in ["general", "medical_symptoms", "digits"]
    for r in sorted(
        [x for x in rows if x["split"] == "train" and x["domain"] == d],
        key=lambda r: r["id"],
    )[:4]
]
result = {}
torch.set_num_threads(4)
for name in ["fp_single", "ternary_single"]:
    m, _ = load_export(run / name / "export")
    m.cfg["checkpoint"] = False
    values = {}
    for mode in ["eval", "train_dropout", "train_without_dropout"]:
        m.train(mode != "eval")
        if mode == "train_without_dropout":
            for b in m.blocks:
                b.dropout = 0.0
        pp = []
        with torch.inference_mode():
            for rep in range(5 if mode == "train_dropout" else 1):
                torch.manual_seed(813 + rep)
                for r in chosen:
                    x = torch.from_numpy(np.load(r["features"])).unsqueeze(0).cuda()
                    gt = torch.tensor(encode(r["text"]), device="cuda")
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        z = m(x)
                    loss = F.ctc_loss(
                        z.float().log_softmax(-1).transpose(0, 1),
                        gt,
                        torch.tensor([z.shape[1]]),
                        torch.tensor([len(gt)]),
                        zero_infinity=False,
                    )
                    pp.append(
                        dict(
                            id=r["id"],
                            loss=float(loss),
                            hypothesis=decode(z.argmax(-1)[0].tolist()),
                        )
                    )
        values[mode] = dict(
            mean_ctc_loss=float(np.mean([p["loss"] for p in pp])),
            prediction_counts=dict(collections.Counter(p["hypothesis"] for p in pp)),
            observations=len(pp),
        )
    result[name] = values
    print(json.dumps({name: values}), flush=True)
    del m
    gc.collect()
    torch.cuda.empty_cache()
save(run / "analysis/dropout-diagnostics.json", result)
