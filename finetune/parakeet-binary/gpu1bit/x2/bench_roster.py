"""The recorded x2 benchmark roster (plans/x2-default.md X5, Y4; agent H). Every arm names model, export,
stack and parts; the report uses these labels and refuses an arm whose recorded model or complete export
identity does not match its row. The comparison pairs are frozen. Each pair reports median paired energy
and median latency separately, with no automatic consequence (Y4); the report states plainly where M1's
new default (b4) uses more energy or has higher latency than M1's previously deployed runtime (b6).
"""
from __future__ import annotations

from pathlib import Path

from gpu1bit.shared import env, fulldev

BENCH_ROSTER = [
    {"id": "b1", "model": "MB1", "export": "MB1 (1-bit export)", "export_key": "mb1", "stack": "B-hybrid", "parts": 2,
     "variant": "MB1-B-hybrid-x2"},
    {"id": "b2", "model": "MB1", "export": "MB1 (1-bit export)", "export_key": "mb1", "stack": "B-hybrid", "parts": 3,
     "variant": "MB1-B-hybrid-x3"},
    {"id": "b3", "model": "MB1", "export": "MB1 (H's derived 2-bit layout)", "export_key": "derived", "stack": "R-expanded",
     "parts": 3, "variant": "R-expanded"},
    {"id": "b4", "model": "M1", "export": "M1 (M1 export a287e977…)", "export_key": "m1", "stack": "R-expanded", "parts": 2,
     "variant": "M1-x2"},
    {"id": "b5", "model": "M1", "export": "M1 (M1 export a287e977…)", "export_key": "m1", "stack": "R-expanded", "parts": 3,
     "variant": "M1-x3"},
    {"id": "b6", "model": "M1", "export": "M1 (M1 export a287e977…)", "export_key": "m1", "stack": "R-hybrid (ptx_row_best)",
     "parts": 3, "variant": "M1-R-hybrid-x3"},
]
PAIRS = [("b1", "b2", "same stack, part count only"), ("b4", "b5", "same stack, part count only"),
         ("b1", "b6", "change of deployed runtime: the MB1 default against M1's previously deployed runtime"),
         ("b4", "b6", "change of deployed runtime: M1's new default against its previously deployed runtime"),
         ("b1", "b4", "change of deployed runtime: both defaults"),
         ("b1", "b3", "change of deployed runtime: the MB1 default against the selection baseline")]
ROW = {b["id"]: b for b in BENCH_ROSTER}
VARIANT_TO_ID = {b["variant"]: b["id"] for b in BENCH_ROSTER}


def export_dir(bid: str) -> Path:
    return {"mb1": env.MB1_EXPORT, "derived": env.DERIVED_EXPORT, "m1": env.M1_EXPORT}[ROW[bid]["export_key"]]


def verify_labels(infos: dict[str, dict], variants: list[str] | None = None) -> None:
    """infos[variant] = the recorded {"roster_id", "model", "export": fulldev.export_identity(...)};
    ValueError unless every roster arm (or every arm in `variants`, e.g. a P-clips artifact's one) is
    present and matches its row's roster id, model and complete export."""
    for b in BENCH_ROSTER:
        if variants is not None and b["variant"] not in variants:
            continue
        info = infos.get(b["variant"])
        if info is None:
            raise ValueError(f"{b['id']} ({b['variant']}): no recorded variant info")
        if info.get("roster_id") != b["id"] or info.get("model") != b["model"]:
            raise ValueError(f"{b['id']}: recorded roster id / model {info.get('roster_id')} / {info.get('model')} "
                             f"is not {b['id']} / {b['model']}")
        want = fulldev.export_identity(export_dir(b["id"]))
        if info.get("export") != want:
            raise ValueError(f"{b['id']}: recorded export does not match its roster row ({b['export']})")


def describe_pairs(full_rows: list[dict]) -> dict:
    """Median paired energy and latency per frozen pair (first arm against second), with counts; plain
    statements where b4 is worse than b6. Descriptive only."""
    from gpu1bit.shared import selection
    out = {"pairs": {}, "statements": []}
    for a, b, label in PAIRS:
        va, vb = ROW[a]["variant"], ROW[b]["variant"]
        out["pairs"][f"{a}-{b}"] = {"label": label, "first": va, "second": vb,
                                    "energy": selection.paired(full_rows, va, vb, "joules_per_iteration"),
                                    "latency": selection.paired(full_rows, va, vb, "median_ms")}
    p = out["pairs"]["b4-b6"]
    if p["energy"]["median_pct"] > 0:
        out["statements"].append(f"b4 (M1's new default, R-expanded x2) uses more energy per full transcription than b6 "
                                 f"(M1's previously deployed R-hybrid x3): median {p['energy']['median_pct']:+.2f}%")
    if p["latency"]["median_pct"] > 0:
        out["statements"].append(f"b4 (M1's new default) has higher latency than b6 (M1's previously deployed runtime): "
                                 f"median {p['latency']['median_pct']:+.2f}%")
    return out
