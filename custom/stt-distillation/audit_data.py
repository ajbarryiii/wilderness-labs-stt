"""Verify the actual prepared split and immutable audio/feature inputs before launch."""

import collections, json
from common import ART, read_manifest, save, digest, medical_terms

m = read_manifest()
rows = m["rows"]
assert len({r["id"] for r in rows}) == len(rows)
for domain in ["general", "digits"]:
    speakers = {
        s: {r["speaker"] for r in rows if r["domain"] == domain and r["split"] == s}
        for s in ["train", "calibration", "development"]
    }
    assert not (
        speakers["train"] & speakers["calibration"]
        or speakers["train"] & speakers["development"]
        or speakers["calibration"] & speakers["development"]
    )
phrases = {
    s: {
        r["phrase_group"]
        for r in rows
        if r["domain"] == "medical_symptoms" and r["split"] == s
    }
    for s in ["train", "calibration", "development"]
}
assert not (
    phrases["train"] & phrases["calibration"]
    or phrases["train"] & phrases["development"]
    or phrases["calibration"] & phrases["development"]
)
for r in rows:
    assert digest(r["audio"]) == r["audio_sha256"]
    assert digest(r["features"]) == r["feature_sha256"]
medical = medical_terms(
    [
        (r["text"], "")
        for r in rows
        if r["domain"] == "medical_symptoms" and r["split"] == "development"
    ]
)
obj = dict(
    rows=len(rows),
    train=sum(r["split"] == "train" for r in rows),
    general_and_digit_speakers_disjoint=True,
    medical_phrases_disjoint=True,
    medical_speakers_unknown=True,
    all_audio_and_feature_hashes_verified=True,
    medical_development_reference_mentions=medical["reference_mentions"],
    medical_development_terms_present={
        t: v["reference"] for t, v in medical["per_term"].items() if v["reference"]
    },
    manifest_sha256=digest(ART / "datasets/pilot/manifest.json"),
)
save(ART / "preflight/data-audit.json", obj)
print(json.dumps(obj), flush=True)
