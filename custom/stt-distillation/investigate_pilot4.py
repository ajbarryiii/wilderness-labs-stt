"""Read-only diagnosis of the stopped four-hour pilot and its two digit errors."""

import hashlib
import json

import numpy as np
import torch

from common import ALPHABET, ART, save, storage
from pilot4_model import PilotModel


def main():
    storage()
    run = ART / "runs/pilot-4h-20260910T150425Z"
    folder = run / "gate/fp_control"
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    timeline = []
    for path in sorted(folder.glob("eval-*.json")):
        obj = json.loads(path.read_text())
        metrics = obj["training"]["metrics"]
        timeline.append(
            {
                "step": obj["step"],
                "wer": metrics["overall"]["wer"],
                "cer": metrics["overall"]["cer"],
                "exact_utterances": metrics["exact_utterances"],
                "digit_exact": metrics["digits"]["exact"],
                "mistakes": [
                    {k: r[k] for k in ["id", "reference", "prediction"]}
                    for r in obj["training"]["predictions"]
                    if r["reference"] != r["prediction"]
                ],
            }
        )
    save(out / "gate-timeline.json", timeline)
    print("TIMELINE", json.dumps([t for t in timeline if t["step"] % 1000 == 0]))
    print(
        "BEST",
        json.dumps(
            {
                "max_digits": max(t["digit_exact"] for t in timeline),
                "min_wer": min(t["wer"] for t in timeline),
                "all_digit_steps": [
                    t["step"] for t in timeline if t["digit_exact"] == 8
                ],
            }
        ),
    )
    obj = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
    model = PilotModel(obj["config"]["model"], obj["arm"]["precision"])
    model.load_state_dict(obj["model"], strict=True)
    del obj
    model.cuda().eval()
    torch.set_num_threads(4)
    rows = json.loads((run / "manifest.json").read_text())["rows"]
    ids = json.loads((run / "subsets.json").read_text())["gate"]
    rows = [r for r in rows if r["id"] in ids]
    outputs = []
    xs = {}
    with torch.inference_mode():
        for row in rows:
            feature = np.load(row["features"])
            xs[row["id"]] = feature
            x = torch.from_numpy(feature)[None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(x)
            probabilities = z[0].float().softmax(-1)
            path = z[0].argmax(-1).tolist()
            previous = None
            events = []
            for t, symbol in enumerate(path):
                if symbol and symbol != previous:
                    events.append(
                        {
                            "frame": t,
                            "milliseconds": t * 20,
                            "symbol": ALPHABET[symbol],
                            "probability": float(probabilities[t, symbol]),
                        }
                    )
                previous = symbol
            prediction = "".join(e["symbol"] for e in events).strip()
            if row["domain"] == "digits":
                onset = int(np.argmax(feature.max(axis=0) > -1.95))
                first = events[0]["frame"] if events else 0
                top = probabilities[first].topk(5)
                summary = {
                    "id": row["id"],
                    "reference": row["text"],
                    "prediction": prediction,
                    "signal_onset_ms": onset * 10,
                    "events": events,
                    "first_event_top5": [
                        (ALPHABET[int(i)], float(p))
                        for i, p in zip(top.indices, top.values)
                    ],
                    "feature_prefix_sha256": hashlib.sha256(
                        feature[:, : first * 2 + 1].tobytes()
                    ).hexdigest(),
                }
                z32 = model(x)
                ids32 = z32[0].argmax(-1).tolist()
                chars32 = []
                previous32 = None
                for c in ids32:
                    if c and c != previous32:
                        chars32.append(ALPHABET[c])
                    previous32 = c
                summary["fp32_prediction"] = "".join(chars32).strip()
                outputs.append(summary)
                print("DIGIT", json.dumps(summary))
    pairs = []
    for a, b in [
        ("digit-nicolas-24", "digit-nicolas-26"),
        ("digit-nicolas-6", "digit-nicolas-26"),
        ("digit-nicolas-40", "digit-nicolas-26"),
    ]:
        x, y = xs[a], xs[b]
        n = min(x.shape[-1], y.shape[-1])
        difference = np.abs(x[:, :n] - y[:, :n]).max(axis=0)
        pairs.append(
            {
                "a": a,
                "b": b,
                "first_nonidentical_frame_ms": int(np.argmax(difference > 0)) * 10,
                "first_difference_gt_001_ms": int(np.argmax(difference > 0.01)) * 10,
                "max_difference_first_100ms": float(difference[:10].max()),
            }
        )
    save(
        out / "alignment-diagnostics.json",
        {"digits": outputs, "feature_prefix_comparisons": pairs},
    )
    print("PREFIXES", json.dumps(pairs))


if __name__ == "__main__":
    main()
