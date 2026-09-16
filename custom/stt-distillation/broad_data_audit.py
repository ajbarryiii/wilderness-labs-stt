"""Audit frozen four-hour data, cached teacher transcripts and onset feasibility."""

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RUN = Path(
    "/mnt/hd/wilderness-labs-stt/stt-distillation/runs/pilot-4h-20260910T180602Z"
)
sys.path.insert(0, str(RUN / "code"))
from common import digest, digit_string, encode, norm, save, scores, storage
from prepare import feature


def main():
    storage()
    rows = json.loads((RUN / "manifest.json").read_text())["rows"]
    subsets = json.loads((RUN / "subsets.json").read_text())
    check_ids = set(subsets["gate"] + subsets["monitor"])
    result = {
        "hash_failures": [],
        "feature_failures": [],
        "recomputed_max_error": 0,
        "gain_onset": {},
        "cached_teachers": {},
        "label_statistics": {},
    }
    stats = collections.defaultdict(list)
    for row in rows:
        for key in ["audio", "feature"]:
            path = row["features" if key == "feature" else key]
            if digest(path) != row[key + "_sha256"]:
                result["hash_failures"].append([row["id"], key])
        x = np.load(row["features"])
        if not np.isfinite(x).all() or x.shape != (80, row["frames"]):
            result["feature_failures"].append(row["id"])
        if row["id"] in check_ids:
            audio, sr = sf.read(row["audio"], dtype="float32")
            assert sr == 16000
            err = float(np.max(abs(x - feature(audio))))
            result["recomputed_max_error"] = max(result["recomputed_max_error"], err)
        # No upper-clipped stored value is present, so gain reduction can be
        # computed exactly up to float rounding in log-power space.
        assert x.max() < 2
        if row["domain"] == "digits":
            x = x[:, 20:]
        y = encode(row["text"])
        required = len(y) + sum(a == b for a, b in zip(y, y[1:]))
        for gain in [0, -6, -12, -18, -24]:
            transformed = np.maximum(-2, x + gain * np.log(10) / 60)
            active = transformed.max(axis=0) > -1.95
            confirmed = np.flatnonzero(
                np.convolve(active.astype(int), np.ones(3, dtype=int), mode="full")[
                    : len(active)
                ]
                == 3
            )
            first = int(confirmed[0] + 10) if len(confirmed) else len(active)
            available = (len(active) + 1) // 2 - (first + 1) // 2
            stats[row["split"], row["domain"], gain].append(
                {
                    "id": row["id"],
                    "available": available,
                    "margin": available - required,
                    "first_ms": first * 10,
                    "floor_fraction": float(np.mean(transformed == -2)),
                }
            )
    for (split, dom, gain), values in stats.items():
        result["gain_onset"][f"{split}/{dom}/{gain}"] = {
            "count": len(values),
            "infeasible": [v for v in values if v["margin"] < 0],
            "no_allowed_frames": sum(v["available"] <= 0 for v in values),
            "minimum_ctc_margin": min(v["margin"] for v in values),
            "open_ms_quantiles": np.quantile(
                [v["first_ms"] for v in values], [0, 0.5, 0.95, 1]
            ).tolist(),
            "mean_floor_fraction": np.mean(
                [v["floor_fraction"] for v in values]
            ).item(),
            "latest_opening_clips": sorted(
                values, key=lambda v: v["first_ms"], reverse=True
            )[:5],
        }
    plain = lambda s: " ".join(re.sub(r"[^a-z0-9' ]", " ", norm(s)).split())
    for name in ["omi", "whisper"]:
        teacher = json.loads(
            (
                RUN.parent / "pilot-8h-20260910T044750Z" / f"teacher-{name}.json"
            ).read_text()
        )
        result["cached_teachers"][name] = {}
        for dom in ["general", "medical_symptoms", "digits"]:
            rs = [
                r
                for r in rows
                if r["split"] == "train" and r["domain"] == dom and r["id"] in teacher
            ]
            pairs = [(r["text"], teacher[r["id"]]["text"]) for r in rs]
            metric = {
                "raw_scores": scores(pairs),
                "punctuation_normalized_scores": scores(
                    [(plain(a), plain(b)) for a, b in pairs]
                ),
                "plain_exact": sum(plain(a) == plain(b) for a, b in pairs),
            }
            if dom == "digits":
                metric["digit_sequences_exact"] = sum(
                    digit_string(a) == digit_string(b) for a, b in pairs
                )
            result["cached_teachers"][name][dom] = metric
    for dom in ["general", "medical_symptoms", "digits"]:
        train = [r for r in rows if r["split"] == "train" and r["domain"] == dom]
        dev = [r for r in rows if r["split"] == "development" and r["domain"] == dom]
        train_words = set(w for r in train for w in plain(r["text"]).split())
        dev_words = [w for r in dev for w in plain(r["text"]).split()]
        result["label_statistics"][dom] = {
            "train_unique_words": len(train_words),
            "dev_word_tokens": len(dev_words),
            "dev_oov_tokens": sum(w not in train_words for w in dev_words),
            "train_unique_phrases": len({plain(r["text"]) for r in train}),
            "dev_phrases_present_in_train": sum(
                plain(r["text"]) in {plain(t["text"]) for t in train} for r in dev
            ),
            "mean_inverse_target_length": np.mean(
                [1 / len(encode(r["text"])) for r in train]
            ).item(),
        }
    save(RUN / "analysis/broad-investigation/data-audit.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "hash_failures",
                    "feature_failures",
                    "recomputed_max_error",
                    "cached_teachers",
                    "label_statistics",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
