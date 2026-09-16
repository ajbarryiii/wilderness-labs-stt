"""Static, exportable research figure from unmodified pilot logs."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = Path(
    "/mnt/hd/wilderness-labs-stt/stt-distillation/runs/pilot-8h-20260910T044750Z"
)
out = run / "analysis"
summary = json.loads((out / "audit-summary.json").read_text())
fig, axes = plt.subplots(
    1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.65, 1]}
)
fig.patch.set_facecolor("white")
labels = {
    "fp_single": "FP · Omi",
    "ternary_single": "Ternary · Omi",
    "fp_multi": "FP · mixed teachers",
    "ternary_multi": "Ternary · mixed teachers",
}
for n in labels:
    r = [json.loads(s) for s in (run / n / "metrics.jsonl").read_text().splitlines()]
    bins = [r[i : i + 250] for i in range(0, len(r), 250)]
    color = "#cf6c25" if n.startswith("ternary") else "#2676a6"
    style = "--" if n.endswith("multi") else "-"
    axes[0].plot(
        [np.mean([a["step"] for a in b]) for b in bins],
        [np.mean([a["loss"] for a in b]) for b in bins],
        style,
        color=color,
        lw=1.7,
        label=labels[n],
    )
axes[0].set(
    xlabel="Optimizer updates",
    ylabel="Mean CTC training loss (250-update bins)",
    title="Loss falls without usable transcription",
    xlim=(0, 27500),
)
axes[0].legend(fontsize=8, frameon=False)
axes[0].grid(alpha=0.18)
names = list(labels)
minutes = [summary["arms"][n]["training"]["elapsed_seconds"] / 60 for n in names]
axes[1].barh(
    [labels[n] for n in names],
    minutes,
    color=["#cf6c25" if n.startswith("ternary") else "#2676a6" for n in names],
    height=0.6,
)
for i, v in enumerate(minutes):
    axes[1].text(v + 1.3, i, f"{v:.1f} min", va="center", fontsize=9)
axes[1].invert_yaxis()
axes[1].set(
    xlabel="Training time", title="Ternary training took ≈32% longer", xlim=(0, 140)
)
axes[1].grid(axis="x", alpha=0.18)
axes[1].set_axisbelow(True)
for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle(
    "604M-parameter ASR pilot: matched training, failed recognition",
    fontsize=14,
    x=0.02,
    ha="left",
)
fig.text(
    0.02,
    0.02,
    "Each arm: 27,100 updates and 108,400 example presentations. Final outputs: blank or “i” on every development clip.",
    fontsize=9,
    color="#555555",
)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(out / "training-curves.png", dpi=180)
fig.savefig(out / "training-curves.svg")
print("Saved training-curves.png and .svg")
