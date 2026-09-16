"""Gates for routing, missing metric coverage, and matched-comparison reporting."""

import json, os
from pathlib import Path
from common import ART, save, medical_terms
from teachers import build_targets
from report import report

out = ART / "preflight/contracts"
out.mkdir(parents=True, exist_ok=True)
rows = []
pred = {k: {} for k in ["omi", "whisper"]}
for domain in ["general", "medical_symptoms", "digits"]:
    for split, n in [("train", 30), ("calibration", 2), ("development", 2)]:
        for i in range(n):
            ident = f"{domain}-{split}-{i}"
            text = "1 2 3" if domain == "digits" else "one two three four"
            rows.append(
                dict(
                    id=ident,
                    split=split,
                    domain=domain,
                    text=text,
                    frames=200,
                    audio=str(out / "unused.wav"),
                    features=str(out / "unused.npy"),
                )
            )
            pred["omi"][ident] = dict(text=text)
            pred["whisper"][ident] = dict(
                text="one two three" if domain != "digits" else "one two three"
            )
save(out / "manifest.json", dict(rows=rows))
os.environ["WILDERNESS_STT_MANIFEST"] = str(out / "manifest.json")
for k, p in pred.items():
    save(out / ("teacher-" + k + ".json"), p)
build_targets(out)
t = json.loads((out / "targets.json").read_text())
assert t["common_training_coverage"] == dict(general=30, medical_symptoms=30, digits=30)
assert (
    t["policy"]["general"]["weights"]["omi"]
    > t["policy"]["general"]["weights"]["whisper"]
)
assert t["targets"]["digits-train-0"]["whisper"] == "one two three"
assert t["targets_different_from_ground_truth"]["whisper"] == 90
p = medical_terms([("oxygen and ketamine", "oxygen")])
assert p["reference_mentions"] == 2 and p["exact_lexical_recall"] == 0.5
assert medical_terms([("hello", "hello")])["exact_lexical_recall"] is None
# A partially completed run cannot be reported as a matched comparison.
save(out / "config.json", dict(arms=[dict(name=n) for n in ["a", "b", "c", "d"]]))
save(out / "state.json", dict(status="running"))
report(out)
assert not json.loads((out / "report.json").read_text())["matched_four_arm_comparison"]
# The actual FP/ternary pipeline smoke used identical initial weights/data order.
r = ART / "preflight/pipeline"
a = json.loads((r / "fp_multi/result.json").read_text())
b = json.loads((r / "ternary_multi/result.json").read_text())
assert a["initial_weight_sha256"] == b["initial_weight_sha256"]
assert a["sample_order_sha256"] == b["sample_order_sha256"]
save(
    ART / "preflight/contracts-passed.json",
    dict(
        routing=True,
        zero_coverage_not_success=True,
        partial_run_not_matched=True,
        matched_initialization_and_data_order=True,
    ),
)
print("CONTRACT CHECKS PASSED", flush=True)
