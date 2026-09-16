"""Read-only checks of saved student weights, acoustics, and output collapse."""

import collections, contextlib, gc, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

run = Path(
    "/mnt/hd/wilderness-labs-stt/stt-distillation/runs/pilot-8h-20260910T044750Z"
)
sys.path.insert(0, str(run / "code"))
from common import save, scores, decode, encode
from export import load_export
from torch.nn import functional as F

read = lambda p: json.loads(p.read_text())
rows = read(run / "manifest.json")["rows"]
out = run / "analysis"
torch.set_num_threads(4)
chosen = [
    r
    for split in ["train", "development"]
    for domain in ["general", "medical_symptoms", "digits"]
    for r in sorted(
        [x for x in rows if x["split"] == split and x["domain"] == domain],
        key=lambda x: x["id"],
    )[:12]
]
features = {r["id"]: np.load(r["features"]) for r in chosen}
feature_stats = dict(
    all_finite=all(np.isfinite(x).all() for x in features.values()),
    min=min(float(x.min()) for x in features.values()),
    max=max(float(x.max()) for x in features.values()),
    mean_temporal_std=float(np.mean([x.std(axis=1).mean() for x in features.values()])),
    distinct_arrays=len({x.tobytes() for x in features.values()}),
    examples=len(features),
)
result = dict(feature_stats=feature_stats, arms={})
for name in ["fp_single", "ternary_single", "fp_multi", "ternary_multi"]:
    m, meta = load_export(run / name / "export")
    preds = []
    layer_stats = []
    hooks = []

    def capture(label):
        def hook(module, args, output):
            z = output.float()
            layer_stats.append(
                dict(
                    layer=label,
                    temporal_std=float(z.std(dim=1).mean()),
                    mean_norm=float(z.norm(dim=-1).mean()),
                )
            )

        return hook

    for i, b in enumerate(m.blocks):
        hooks.append(b.register_forward_hook(capture(f"block_{i + 1}")))
    with torch.inference_mode():
        for i, r in enumerate(chosen):
            x = torch.from_numpy(features[r["id"]]).unsqueeze(0).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = m(x)
            if i == 0:
                for h in hooks:
                    h.remove()
            p = z.float().softmax(-1)
            gt = torch.tensor(encode(r["text"]), device="cuda")
            lp = z.float().log_softmax(-1).transpose(0, 1)
            loss = F.ctc_loss(
                lp,
                gt,
                torch.tensor([z.shape[1]]),
                torch.tensor([len(gt)]),
                zero_infinity=False,
            )
            preds.append(
                dict(
                    id=r["id"],
                    split=r["split"],
                    domain=r["domain"],
                    reference=r["text"],
                    hypothesis=decode(p.argmax(-1)[0].tolist()),
                    ctc_loss=float(loss),
                    blank_frame_fraction=float((p.argmax(-1) == 0).float().mean()),
                    mean_blank_probability=float(p[:, :, 0].mean()),
                )
            )
        probe = chosen[0]
        x = torch.from_numpy(features[probe["id"]]).unsqueeze(0).cuda()
        # Compare different precision and an acoustically unrelated all-zero input.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            baseline = m(x).float()
            zero = m(torch.zeros_like(x)).float()
        full = m(x).float()
        sanity = dict(
            probe=probe["id"],
            bf16_hypothesis=decode(baseline.argmax(-1)[0].tolist()),
            fp32_hypothesis=decode(full.argmax(-1)[0].tolist()),
            zero_feature_hypothesis=decode(zero.argmax(-1)[0].tolist()),
            mean_abs_logit_change_for_zero_features=float(
                (baseline - zero).abs().mean()
            ),
        )
        # Load the final expanded checkpoint into the same model. This rules out
        # ternary packing/unpacking as the source of the failed recognition.
        cp = load_file(str(run / name / "checkpoint/weights.safetensors"))
        maxdiff = max(
            float((p.detach().cpu() - cp[k]).abs().max())
            for k, p in m.named_parameters()
        )
        m.load_state_dict(cp, strict=True)
        del cp
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expanded = m(x).float()
        sanity.update(
            max_weight_difference_export_vs_checkpoint=maxdiff,
            max_logit_difference_export_vs_checkpoint=float(
                (baseline - expanded).abs().max()
            ),
            checkpoint_hypothesis=decode(expanded.argmax(-1)[0].tolist()),
        )
    by_split = {}
    for split in ["train", "development"]:
        rr = [r for r in preds if r["split"] == split]
        by_split[split] = dict(
            scores=scores([(r["reference"], r["hypothesis"]) for r in rr]),
            prediction_counts=dict(collections.Counter(r["hypothesis"] for r in rr)),
            mean_ctc_loss=float(np.mean([r["ctc_loss"] for r in rr])),
            mean_blank_frame_fraction=float(
                np.mean([r["blank_frame_fraction"] for r in rr])
            ),
            mean_blank_probability=float(
                np.mean([r["mean_blank_probability"] for r in rr])
            ),
        )
    result["arms"][name] = dict(
        by_split=by_split,
        sanity=sanity,
        layer_stats_for_first_training_example=layer_stats,
        predictions=preds,
    )
    print(
        json.dumps(
            dict(
                arm=name,
                by_split=by_split,
                sanity=sanity,
                first_and_last_layer=[layer_stats[0], layer_stats[-1]],
            )
        ),
        flush=True,
    )
    del m
    gc.collect()
    torch.cuda.empty_cache()
    save(out / "collapse-diagnostics.json", result)
