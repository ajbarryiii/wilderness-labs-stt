"""Always produce a partial report; budget limits are not accuracy cliffs."""

import json
from pathlib import Path
from common import save


def report(run):
    def read(p):
        return json.loads(p.read_text()) if p.exists() else None

    cfg = read(run / "config.json")
    state = read(run / "state.json")
    arms = {}
    for a in cfg["arms"]:
        p = run / a["name"]
        arms[a["name"]] = dict(
            training=read(p / "result.json"), development=read(p / "development.json")
        )
    completed = [
        v["training"]
        for v in arms.values()
        if v["training"] and v["training"]["status"] == "completed"
    ]
    matched = (
        len(completed) == 4
        and len({r["steps"] for r in completed}) == 1
        and len({r["sample_order_sha256"] for r in completed}) == 1
        and len({r["initial_weight_sha256"] for r in completed}) == 1
    )
    teachers = {
        k: read(run / ("teacher-" + k + "-summary.json")) for k in ["omi", "whisper"]
    }
    obj = dict(
        teachers=teachers,
        state=state,
        matched_four_arm_comparison=matched,
        arms=arms,
        interpretation="This short run tests implementation and early learning. Poor or blank-heavy results are inconclusive about the accuracy cliff of a converged distilled model. Medical symptoms and composed digits do not establish TCCC, medication/dose or natural-number accuracy.",
    )
    save(run / "report.json", obj)
    lines = [
        "# Eight-hour distillation pilot",
        "",
        f"Status: {state['status'] if state else 'unknown'}",
        f"Matched four-arm comparison: {matched}",
        "",
        "| Arm | Steps | General WER | Medical symptom WER | Digit exact accuracy |",
        "|---|---:|---:|---:|---:|",
    ]

    def fmt(x):
        return "pending" if x is None else f"{x:.2%}"

    for k, v in arms.items():
        dev = v["development"]
        d = dev["domains"] if dev else {}
        t = v["training"]
        lines.append(
            f"| {k} | {t['steps'] if t else 'pending'} | {fmt(d.get('general', {}).get('wer'))} | {fmt(d.get('medical_symptoms', {}).get('wer'))} | {fmt(d.get('digits', {}).get('digit_sequence_accuracy'))} |"
        )
    for k, v in teachers.items():
        if v:
            d = v["development"]
            lines.append(
                f"| Teacher: {k} | frozen | {fmt(d.get('general', {}).get('wer'))} | {fmt(d.get('medical_symptoms', {}).get('wer'))} | {fmt(d.get('digits', {}).get('digit_sequence_accuracy'))} |"
            )
    lines += [
        "",
        obj["interpretation"],
        "",
        "Read per-arm result.json for failures, actual audio exposure, teacher routes, and memory use. Exports execute as floating-point references; packed bytes do not demonstrate device energy savings.",
    ]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import sys

    report(Path(sys.argv[1]))
