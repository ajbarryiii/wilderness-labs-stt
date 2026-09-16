"""Post-run audit; preserve the original report and predictions unchanged."""

import collections, datetime, json, statistics, sys, re
from pathlib import Path

run = Path(
    "/mnt/hd/wilderness-labs-stt/stt-distillation/runs/pilot-8h-20260910T044750Z"
)
sys.path.insert(0, str(run / "code"))
from common import save, digest, scores, distance

out = run / "analysis"
out.mkdir(exist_ok=True)
read = lambda p: json.loads(p.read_text())
cfg = read(run / "config.json")
manifest = read(run / "manifest.json")
state = read(run / "state.json")
targets = read(run / "targets.json")
rows = manifest["rows"]
summary = dict(
    run=str(run),
    status=state["status"],
    wall_seconds=(
        datetime.datetime.fromisoformat(state["finished_utc"])
        - datetime.datetime.fromisoformat(state["started_utc"])
    ).total_seconds(),
    dataset=manifest["summary"],
    arms={},
    teachers={},
    target_policy=targets["policy"],
    target_acceptance=targets["acceptance_counts"],
    target_differences=targets["targets_different_from_ground_truth"],
)
for arm in cfg["arms"]:
    name = arm["name"]
    p = run / name
    r = read(p / "result.json")
    dev = read(p / "development.json")
    metrics = [json.loads(s) for s in (p / "metrics.jsonl").read_text().splitlines()]
    curve = []
    for start, end in [
        (1, 100),
        (101, 1000),
        (1001, 5000),
        (5001, 10000),
        (10001, 20000),
        (20001, 26100),
        (26101, 27100),
    ]:
        m = [x for x in metrics if start <= x["step"] <= end]
        curve.append(
            dict(
                start=start,
                end=end,
                loss=statistics.mean(x["loss"] for x in m),
                ground_truth_loss=statistics.mean(x["ground_truth_loss"] for x in m),
            )
        )
    verify = (
        digest(p / "export/weights.safetensors")
        == read(p / "export/export.json")["weights_sha256"]
        == dev["export_sha256"]
    )
    summary["arms"][name] = dict(
        training=r,
        domains={
            d: {k: v for k, v in v.items() if k != "medical_terms"}
            for d, v in dev["domains"].items()
        },
        prediction_counts=dict(
            collections.Counter(x["hypothesis"] for x in dev["predictions"])
        ),
        examples=dev["predictions"][:3] + dev["predictions"][-3:],
        curve=curve,
        export_verified=verify,
        export_bytes=dev["export_bytes"],
        evaluation_seconds=dev["evaluation_seconds"],
        mean_step_seconds=statistics.mean(m["step_seconds"] for m in metrics),
        examples_seen=len(metrics) * cfg["accumulation"],
    )
for kind in ["omi", "whisper"]:
    pred = read(run / ("teacher-" + kind + ".json"))
    t = read(run / ("teacher-" + kind + "-summary.json"))
    summary["teachers"][kind] = dict(
        scores={
            s: {
                d: {k: v for k, v in sc.items() if k != "medical_terms"}
                for d, sc in ds.items()
            }
            for s, ds in t.items()
        }
    )
    bad = []
    for r in rows:
        if r["split"] != "development":
            continue
        h = pred[r["id"]]["text"]
        we = distance(r["text"].split(), h.split())
        bad.append(
            dict(
                id=r["id"],
                domain=r["domain"],
                reference=r["text"],
                hypothesis=h,
                word_errors=we,
                wer=we / len(r["text"].split()),
            )
        )
    summary["teachers"][kind]["worst_general"] = sorted(
        [r for r in bad if r["domain"] == "general"],
        key=lambda r: r["word_errors"],
        reverse=True,
    )[:6]
    strip = lambda s: " ".join(re.sub(r"[^a-z0-9' ]", " ", s.lower()).split())
    summary["teachers"][kind]["punctuation_stripped_wer"] = {
        d: scores(
            [
                (strip(r["text"]), strip(pred[r["id"]]["text"]))
                for r in rows
                if r["domain"] == d and r["split"] == "development"
            ]
        )["wer"]
        for d in ["general", "medical_symptoms"]
    }
# Approximate GPU board energy by trapezoidal integration of regularly sampled power.
telemetry = collections.defaultdict(list)
for line in (run / "gpu.csv").read_text().splitlines():
    stage, ts, util, mem, power, temp = [x.strip() for x in line.split(",")]
    telemetry[stage].append(
        (
            datetime.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp(),
            float(util),
            float(mem),
            float(power),
            float(temp),
        )
    )
summary["telemetry"] = {}
for stage, rr in telemetry.items():
    energy = sum(
        (b[0] - a[0]) * (a[3] + b[3]) / 2
        for a, b in zip(rr, rr[1:])
        if 0 < b[0] - a[0] < 30
    )
    summary["telemetry"][stage] = dict(
        samples=len(rr),
        mean_utilization=statistics.mean(r[1] for r in rr),
        mean_power_watts=statistics.mean(r[3] for r in rr),
        peak_gpu_mib=max(r[2] for r in rr),
        peak_temperature_c=max(r[4] for r in rr),
        sampled_board_energy_kwh=energy / 3.6e6,
    )
# Require agreement across source run records, not just the convenience report flag.
trs = [x["training"] for x in summary["arms"].values()]
summary["matched"] = (
    all(r["status"] == "completed" and r["steps"] == 27100 for r in trs)
    and len({r["sample_order_sha256"] for r in trs}) == 1
    and len({r["initial_weight_sha256"] for r in trs}) == 1
    and all(r["exposure_hours"] == trs[0]["exposure_hours"] for r in trs)
)
save(out / "audit-summary.json", summary)
print(json.dumps(summary, indent=2))
