"""Observed compute/data curves and paired comparisons; no long-run promises."""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from common import save
from recovery_core import DOMAINS, SOURCE_RUN, artifact, read
from scaling_core import bootstrap_delta


def fit_backtest(x, y):
    """Fit early checkpoints, predict two withheld later checkpoints.

    Report both plausible forms and fit instability. Three dataset tiers cannot
    identify a reliable joint scaling law or a two-week asymptote.
    """
    pairs = sorted((float(a), float(b)) for a, b in zip(x, y) if a > 0 and np.isfinite(b))
    if len(pairs) < 6:
        return dict(status="insufficient_points", needed=6, available=len(pairs))
    x, y = map(np.array, zip(*pairs))
    scale = x[0]
    x = x / scale
    functions = {
        "exponential_floor": lambda z, c, a, k: c + a * np.exp(-z * k),
        "power_floor": lambda z, c, a, k: c + a * z ** (-k),
    }
    result = {}
    for name, fun in functions.items():
        try:
            def fit(xx, yy):
                candidates = []
                for rate in (.01, .1, 1.):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        try:
                            p, covariance = curve_fit(fun, xx, yy, p0=[max(0, yy.min() * .5), max(yy.max(), .01), rate],
                                bounds=([0, 0, .000001], [np.inf, np.inf, 10]), maxfev=20000)
                            candidates.append((float(np.mean((fun(xx, *p) - yy) ** 2)), p, covariance))
                        except (RuntimeError, ValueError, Warning):
                            continue
                if not candidates:
                    raise ValueError("No stable numerical fit")
                return min(candidates, key=lambda item: item[0])[1:]
            p, _ = fit(x[:-2], y[:-2])
            predicted = fun(x[-2:], *p)
            full, covariance = fit(x, y)
            result[name] = dict(status="fitted", fitted_floor=float(full[0]),
                early_fitted_floor=float(p[0]), parameters=full.tolist(), x_scale_seconds=scale,
                covariance_condition=float(np.linalg.cond(covariance)) if np.isfinite(covariance).all() and np.linalg.cond(covariance) < 1e300 else None,
                withheld_actual=y[-2:].tolist(), withheld_predicted=predicted.tolist(),
                withheld_mae=float(np.mean(abs(predicted - y[-2:]))))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            result[name] = dict(status="unstable", reason=str(exc))
    return dict(status="diagnostic_only", models=result,
        warning="Monotone decay is an assumption. Inspect worsening curves, floors, covariance and withheld errors; no two-week forecast is produced.")


def pareto(points, metric):
    candidates = [p for p in points if p[metric] is not None and p["update"] > 0 and not p.get("uncertain_prior_attempts", 0)]
    return [p["id"] for p in candidates if not any(
        q["training_seconds"] <= p["training_seconds"] and q["available_hours"] <= p["available_hours"] and q[metric] <= p[metric]
        and (q["training_seconds"] < p["training_seconds"] or q["available_hours"] < p["available_hours"] or q[metric] < p[metric])
        for q in candidates)]


def collect(run):
    cfg = read(run / "config.json")
    ready = read(run / "data-ready.json")
    points, records = [], {}
    for index in sorted((run / "training").glob("*/index.json")):
        for path in read(index)["records"]:
            record = read(run / path)
            # Index contains only evaluations committed with resumable state.
            assert (run / record["weights"]).is_file()
            key = f"{record['job']}:{record['update']}"
            records[key] = record
            nominal = ready["summary"][str(record["seed"])][record["tier"]]
            point = dict(id=key, job=record["job"], tier=record["tier"], seed=record["seed"], update=record["update"],
                training_seconds=record["resource_cost"]["training_seconds_including_recorded_retries"],
                trajectory_training_seconds=record["timing"]["training_work_seconds"], cuda_interval_seconds=record["timing"]["cuda_interval_seconds"],
                measured_attempt_wall_seconds=record["resource_cost"]["measured_attempt_wall_seconds"],
                uncertain_prior_attempts=record["resource_cost"]["uncertain_prior_attempts"],
                cumulative_accounted_seconds=sum(v for k, v in record["timing"].items() if k != "cuda_interval_seconds"),
                available_hours=sum(v["hours"] for v in nominal.values()), available_general_hours=nominal["general"]["hours"],
                seen_hours=sum(record["exposure"]["unique_hours"].values()),
                seen_general_hours=record["exposure"]["unique_hours"]["general"],
                padded_input_frames=record["exposure"]["padded_input_frames"],
                training_monitor_ctc=float(np.mean([m["ctc_loss"] for m in record["evaluations"]["common_train_monitor"]["metrics"].values()])))
            for d in DOMAINS:
                m = record["evaluations"]["development"]["metrics"][d]
                for metric in ("wer", "cer", "ctc_loss"):
                    point[d + "_" + metric] = m[metric]
                if d == "digits":
                    point["digits_sequence_error"] = 1 - m["exact"] / m["total"]
            for name in ("known_speaker", "long_development"):
                point[name + "_wer"] = record["evaluations"][name]["metrics"]["general"]["wer"]
            points.append(point)
    return cfg, points, records


def plot(run, points):
    if not points:
        return None
    try:
        import matplotlib
    except ImportError:
        deps = SOURCE_RUN / "analysis/broad-investigation/plot-dependencies"
        if not deps.is_dir():
            return dict(status="unavailable", reason="matplotlib is not installed")
        sys.path.append(str(deps))
        import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = {"small": "#2474b5", "medium": "#db8218", "large": "#34945b"}
    views = [("general_wer", "General WER"), ("medical_symptoms_wer", "Medical WER"),
             ("digits_sequence_error", "Digit sequence error"), ("general_ctc_loss", "General validation CTC"),
             ("training_monitor_ctc", "Common training monitor CTC"), ("seen_general_hours", "Unique general audio seen (hours)")]
    for ax, (metric, title) in zip(axes.flat, views):
        for job in sorted({p["job"] for p in points}):
            ps = sorted((p for p in points if p["job"] == job), key=lambda p: p["update"])
            ax.plot([p["training_seconds"] / 3600 for p in ps], [p[metric] for p in ps],
                    marker=".", color=colors[ps[0]["tier"]], alpha=.8,
                    linestyle="-" if ps[0]["seed"] == min(p["seed"] for p in points) else "--", label=job)
        ax.set(title=title, xlabel="Training work (hours)")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Observed trajectories; fixed recipe and LR horizon, two paired replicates")
    for ext in ("png", "svg"):
        fig.savefig(run / "analysis" / ("compute-data-curves." + ext), dpi=160)
    plt.close(fig)
    # Separate resource surface: nominal unique data vs measured compute.
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    plotted = ax.scatter([p["training_seconds"] / 3600 for p in points if p["update"]],
               [p["available_general_hours"] for p in points if p["update"]],
               c=[p["general_wer"] for p in points if p["update"]], cmap="viridis_r", s=55)
    fig.colorbar(plotted, ax=ax, label="General validation WER")
    ax.set(xlabel="Training work (hours)", ylabel="Available unique general audio (hours)",
           title="Measured compute/data grid (no interpolated scaling-law surface)")
    fig.savefig(run / "analysis/compute-data-grid.svg")
    plt.close(fig)
    return dict(status="written", matplotlib_version=matplotlib.__version__)


def analyze(run):
    run = artifact(run)
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    cfg, points, records = collect(run)
    comparisons = []
    for seed in cfg["seeds"]:
        for step in cfg["checkpoint_updates"]:
            a = records.get(f"{seed}-small:{step}")
            if a is None:
                continue
            for tier in cfg["sizes"]:
                b = records.get(f"{seed}-{tier}:{step}")
                if tier == "small" or b is None:
                    continue
                assert a["exposure"]["padded_input_frames"] == b["exposure"]["padded_input_frames"]
                comparisons.append(dict(seed=seed, update=step, tier=tier,
                    timing_ratio=b["timing"]["training_work_seconds"] / a["timing"]["training_work_seconds"],
                    domains={d: bootstrap_delta(a["evaluations"]["development"]["predictions"],
                        b["evaluations"]["development"]["predictions"], d, cfg["bootstrap_replicates"]) for d in DOMAINS}))
    fits = {}
    for job in sorted({p["job"] for p in points}):
        # Fit only the preregistered update grid: time-triggered extras must not
        # change a tier's statistical weight in a curve fit.
        ps = [p for p in points if p["job"] == job and p["update"] in cfg["checkpoint_updates"] and not p["uncertain_prior_attempts"]]
        fits[job] = {metric: fit_backtest([p["training_seconds"] for p in ps], [p[metric] for p in ps])
                     for metric in ("general_ctc_loss", "medical_symptoms_ctc_loss", "general_wer")}
    fronts = {m: pareto(points, m) for m in ("general_wer", "medical_symptoms_wer", "digits_sequence_error")}
    limitations = ["Only general-speech recording diversity varies; medical/digit pools stay fixed.",
        "Same source and training speakers, duration bins and padded shapes; exact useful audio/label counts can differ and are logged.",
        "Two paired seeds include different gate initialization and nested subset realization. Seed spread is descriptive, not a reliable training-seed CI.",
        "Bootstrap intervals condition on trained models. Medical clusters are transcripts; digit holdout has one speaker.",
        "Development/known-speaker data were reused in earlier experiments. These are validation results, not final-test/deployment estimates.",
        "Fixed LR horizon means early points are partial trajectories, not separately tuned shorter training runs.",
        "Pareto fronts are observed per-domain candidates, not certified winners or global compute optima. No two-week extrapolation."]
    primary = []
    for tier in cfg["sizes"]:
        if tier == "small":
            continue
        matched = [c for c in comparisons if c["tier"] == tier and c["update"] == max(cfg["checkpoint_updates"])]
        complete = len(matched) == len(cfg["seeds"])
        primary.append(dict(tier=tier, complete=complete,
            consistent_general_improvement=all(c["domains"]["general"]["delta"] < 0 for c in matched) if complete else None,
            medical_guard_passed=all(c["domains"]["medical_symptoms"]["delta"] <= .01 for c in matched) if complete else None,
            digit_guard_passed=all(c["domains"]["digits"]["delta"] <= 1 / 48 + 1e-12 for c in matched) if complete else None,
            scope="Final matched-update general WER is primary. Guards: at most +1 percentage point medical WER / one additional wrong digit sequence per seed. Candidate evidence only."))
    summary = dict(status="observed_results" if points else "prepared_no_results", points=points, primary_endpoint=primary,
        paired_comparisons=comparisons, curve_backtests=fits, observed_pareto=fronts,
        limitations=limitations, plots=plot(run, points))
    save(output / "summary.json", summary)
    if points:
        with (output / "points.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)
    lines = ["# Unique data / compute experiment", "", "Status: " + summary["status"], "",
             "| Job | Updates | Training hours | General hours available / seen | General WER | Medical WER | Digit sequence error |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for p in points:
        lines.append(f"| {p['job']} | {p['update']} | {p['training_seconds']/3600:.3f} | {p['available_general_hours']:.2f} / {p['seen_general_hours']:.2f} | {p['general_wer']:.2%} | {p['medical_symptoms_wer']:.2%} | {p['digits_sequence_error']:.2%} |")
    lines += ["", *["- " + line for line in limitations]]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    result = analyze(args.run)
    print(json.dumps(dict(status=result["status"], points=len(result["points"]))))
