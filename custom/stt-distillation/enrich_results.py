import collections, json, re, sys
from pathlib import Path

run = Path(
    "/mnt/hd/wilderness-labs-stt/stt-distillation/runs/pilot-8h-20260910T044750Z"
)
sys.path.insert(0, str(run / "code"))
from common import save, scores, distance, digit_string

rows = json.loads((run / "manifest.json").read_text())["rows"]
targets = json.loads((run / "targets.json").read_text())
out = {}
clean = lambda x: " ".join(re.sub(r"[^a-z0-9' ]", " ", x.lower()).split())
for k in ["omi", "whisper"]:
    ps = json.loads((run / f"teacher-{k}.json").read_text())
    rr = [r for r in rows if r["split"] == "development" and r["domain"] == "general"]
    per = [(distance(r["text"].split(), ps[r["id"]]["text"].split()), r) for r in rr]
    worst = max(per, key=lambda x: x[0])
    rest = [r for r in rr if r["id"] != worst[1]["id"]]
    train = [r for r in rows if k in targets["targets"].get(r["id"], {})]
    out[k] = dict(
        accepted_training_targets=len(train),
        targets_differ_after_punctuation_stripping=sum(
            clean(r["text"]) != clean(targets["targets"][r["id"]][k]) for r in train
        ),
        targets_identical_after_punctuation_stripping=sum(
            clean(r["text"]) == clean(targets["targets"][r["id"]][k]) for r in train
        ),
        worst_development_general_id=worst[1]["id"],
        worst_word_errors=worst[0],
        general_wer_excluding_worst=scores(
            [(r["text"], ps[r["id"]]["text"]) for r in rest]
        )["wer"],
    )
p = {k: json.loads((run / f"teacher-{k}.json").read_text()) for k in ["omi", "whisper"]}
paired = collections.Counter()
for r in rows:
    if r["split"] == "development" and r["domain"] == "digits":
        good = {
            k: digit_string(p[k][r["id"]]["text"]) == digit_string(r["text"]) for k in p
        }
        paired[
            "both_correct"
            if all(good.values())
            else "omi_only"
            if good["omi"]
            else "whisper_only"
            if good["whisper"]
            else "both_wrong"
        ] += 1
out["paired_digit_outcomes"] = dict(paired)
save(run / "analysis/teacher-sensitivity.json", out)
print(json.dumps(out))
