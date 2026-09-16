"""Plot measured pilot learning, representation and generalization diagnostics."""

import json
import sys
from pathlib import Path

ROOT = Path(
    "/mnt/hd/wilderness-labs-stt/stt-distillation/runs/pilot-4h-20260910T180602Z/analysis/broad-investigation"
)
sys.path.insert(0, str(ROOT / "plot-dependencies"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import storage


def main():
    storage()
    root = ROOT
    read = lambda name: json.loads((root / name).read_text())
    static = read("static.json")
    checkpoints = read("checkpoints.json")
    generalization = read("generalization.json")
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for arm, label, color in [
        ("fp_control", "FP", "#b23d35"),
        ("ternary_matched", "Ternary", "#176b9a"),
    ]:
        data = static["timelines"]["train/" + arm]["evaluations"]
        for split, style, suffix in [
            ("training", "-", "training monitor"),
            ("development", "--", "development"),
        ]:
            ax.plot(
                [r["step"] / 1000 for r in data],
                [r[split]["overall"]["cer"] * 100 for r in data],
                style,
                color=color,
                label=f"{label}: {suffix}",
            )
    ax.set(
        title="A. FP fails to learn; ternary separates train and development",
        xlabel="Broad-training updates (thousands)",
        ylabel="Overall character error (%)",
        ylim=(-2, 105),
    )
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    data = static["timelines"]["train/ternary_matched"]["evaluations"]
    for domain, label, color in [
        ("general", "General", "#176b9a"),
        ("medical_symptoms", "Symptoms", "#a46a16"),
    ]:
        for split, style, suffix in [
            ("training", "-", "training monitor"),
            ("development", "--", "development"),
        ]:
            ax.plot(
                [r["step"] / 1000 for r in data],
                [r[split]["domains"][domain]["cer"] * 100 for r in data],
                style,
                color=color,
                label=f"{label}: {suffix}",
            )
    ax.set(
        title="B. Ternary training improvement does not transfer",
        xlabel="Broad-training updates (thousands)",
        ylabel="Character error (%)",
        ylim=(-2, 105),
    )
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for key, label, color, style in [
        ("gate/fp_control/latest", "FP successful gate", "#828282", "--"),
        ("train/fp_control/best", "FP selected (18k)", "#b23d35", "-"),
        ("train/fp_control/latest", "FP final (25k)", "#d9958b", "-"),
        ("train/ternary_matched/latest", "Ternary final (27k)", "#176b9a", "-"),
    ]:
        values = np.array(
            [
                [s["mean_channel_temporal_std"] for s in p["layer_statistics_after_1s"]]
                for p in checkpoints[key]["probes"]
            ]
        )
        ax.semilogy(
            range(1, 17), np.median(values, axis=0), style, color=color, label=label
        )
    ax.set(
        title="C. Failed FP encoder loses acoustic variation",
        xlabel="Encoder block",
        ylabel="Temporal standard deviation (log scale)",
        xticks=[1, 4, 8, 12, 16],
    )
    ax.legend(fontsize=9)
    ax.text(
        0.02,
        0.02,
        "Median of 12 clips; frames after 1 s; mean across channels",
        transform=ax.transAxes,
        fontsize=8,
    )

    ax = axes[1, 1]
    groups = ["all_training", "unseen_known_speakers", "development"]
    labels = [
        "Training recordings\n2,400 clips",
        "New recordings, known\nspeakers: 64 clips",
        "Development, unseen\nspeakers: 64 clips",
    ]
    for delta, name, color, label in [
        (-0.19, "best", "#85b2ce", "Selected checkpoint (14k)"),
        (0.19, "latest", "#176b9a", "Final checkpoint (27k)"),
    ]:
        values = [
            generalization[name][g]["metrics"]["general"]["cer"] * 100 for g in groups
        ]
        bars = ax.bar(
            np.arange(3) + delta, values, width=0.36, color=color, label=label
        )
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set(
        title="D. Ternary: unseen recordings versus unseen speakers",
        ylabel="General-speech character error (%)",
        xticks=range(3),
        xticklabels=labels,
        ylim=(0, 75),
    )
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle(
        "Four-hour pilot diagnosis · 10 September 2026 · revised 18:06 UTC run",
        fontsize=15,
    )
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(root / ("findings." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
