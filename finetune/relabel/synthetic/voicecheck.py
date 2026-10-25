"""The voice-provenance check (synthetic plan "Speech": the rule that dropped syn-dialog).

An engine is used only if its voices can be shown not to be dev or test speakers, from the engine's documented
training data and voice provenance; otherwise it is dropped. Weights, outputs and voice assets are checked
separately (their license fields in the private registry must be present and permissive).

Per engine (writes $RELABEL_ROOT/synthetic/voice_check.json, read by the TTS jobs, which refuse a failed engine):
- the speaker table is read from the model files; only entries that are documented corpus speaker ids
  (tts.VOICE_ID) are voices; any other entry is listed as excluded and never used;
- syn-tts-b (documented: trained from scratch on one LibriSpeech training split, re-processed for synthesis, whose
  speaker ids are LibriSpeech reader ids): every voice id must be a speaker of that split (registry
  "voice_split") in LibriSpeech's SPEAKERS.TXT, and none may be a speaker of the LibriSpeech dev/test sets, taken
  both from SPEAKERS.TXT (dev-*/test-* subsets) and from the speaker ids of this project's LibriSpeech dev and
  test manifests;
- syn-tts-a (documented: trained only on a separate studio corpus recorded for speech synthesis): every voice
  must be a documented speaker id of that corpus (registry "voice_id_range"); the corpus is not a source of any
  dev or test set (LibriSpeech, AMI, VoxPopuli, Common Voice, TED-LIUM, GigaSpeech, SPGISpeech, Earnings-22,
  the domain dev sets, YouTube-derived sets), so no voice is a dev or test speaker by provenance.
Residual risk (recorded, not checkable by id): a person who is a speaker of two unrelated corpora.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from synthetic import layout

SPEAKERS_TXT = ("whisper-ternary", "data", "LibriSpeech", "SPEAKERS.TXT")   # under $DATA_ROOT (read only)
LIBRISPEECH_MANIFESTS = ("librispeech_dev_clean", "librispeech_dev_other", "test_librispeech_clean",
                         "test_librispeech_other")
DOCUMENTED = {
    "syn-tts-a": {"training_data": "one studio multi-speaker corpus, not a dev/test source (registry: training_data)",
                  "voice_ids": "the corpus's speaker ids (registry: voice_id_range)"},
    "syn-tts-b": {"training_data": "one LibriSpeech training split (registry: voice_split)",
                  "voice_ids": "LibriSpeech reader ids"},
}


def librispeech_speakers(path: Path) -> dict[str, str]:
    """reader id -> subset, from SPEAKERS.TXT."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(";") or not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        out[parts[0]] = parts[2]
    return out


def manifest_speakers(manifest_dir: Path) -> set[str]:
    out = set()
    for name in LIBRISPEECH_MANIFESTS:
        with open(manifest_dir / f"{name}.jsonl", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rid = json.loads(line)["id"].split(":")[-1]
                    out.add(rid.split("-")[0])
    return out


def license_fields(alias: str) -> dict:
    e = layout.source(alias)
    fields = {"weights": e.get("license"), "outputs": e.get("outputs"), "voice_assets": e.get("voice_assets"),
              "training_data": e.get("training_data") or e.get("license")}
    missing = [k for k, v in fields.items() if not v]
    bad = [k for k, v in fields.items() if v and re.search(r"\bnc\b|non-?commercial|research only|no ai", v, re.I)]
    return {"fields": fields, "missing": missing, "restrictive": bad}


def check_engine(alias: str, table: list[str], speakers: dict[str, str], dev_test_manifest: set[str]) -> dict:
    from synthetic.tts import VOICE_ID
    voices = sorted(s for s in table if VOICE_ID[alias].fullmatch(s))
    excluded = sorted(s for s in table if not VOICE_ID[alias].fullmatch(s))
    lic = license_fields(alias)
    problems = [f"license field missing: {k}" for k in lic["missing"]] + \
               [f"restrictive license: {k}" for k in lic["restrictive"]]
    result = {"alias": alias, "voices": len(voices), "excluded_entries": excluded, "documented": DOCUMENTED[alias],
              "licenses": lic["fields"]}
    if not voices:
        problems.append("no documented voices")
    if alias == "syn-tts-b":
        ids = [v[1:] for v in voices]
        split = layout.source(alias)["voice_split"]
        dev_test_ls = {k for k, s in speakers.items() if s.startswith(("dev-", "test-"))}
        not_in_split = [i for i in ids if speakers.get(i) != split]
        in_dev_test = sorted(set(ids) & (dev_test_ls | dev_test_manifest))
        result.update(in_documented_split=len(ids) - len(not_in_split), not_in_documented_split=not_in_split[:50],
                      dev_test_speakers_checked=len(dev_test_ls | dev_test_manifest), in_dev_or_test=in_dev_test)
        if not_in_split:
            problems.append(f"{len(not_in_split)} voices are not speakers of {split}")
        if in_dev_test:
            problems.append(f"{len(in_dev_test)} voices are LibriSpeech dev/test speakers")
    elif alias == "syn-tts-a":
        lo, hi = layout.source(alias)["voice_id_range"]
        bad = [v for v in voices if not lo <= int(v[1:]) <= hi]
        result.update(outside_documented_range=bad)
        if bad:
            problems.append(f"{len(bad)} voices outside the documented speaker ids")
    result["problems"] = problems
    result["verdict"] = "pass" if not problems else "fail"
    result["residual_risk"] = "a person who also speaks in an unrelated dev/test corpus cannot be excluded by id"
    return result


def run(out: Path | None = None) -> dict:
    from synthetic.tts import ENGINES
    layout.require_mount()
    speakers = librispeech_speakers(layout.root("DATA_ROOT").joinpath(*SPEAKERS_TXT))
    manifests = layout.root("PARAKEET_TERNARY_ROOT") / "manifests"
    dev_test = manifest_speakers(manifests)
    res = {"engines": {a: check_engine(a, ENGINES[a]().speaker_table(), speakers, dev_test) for a in layout.TTS_ENGINES},
           "librispeech_dev_test_speakers_from_manifests": len(dev_test)}
    out = layout.base() / "voice_check.json" if out is None else out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(res, indent=1) + "\n")
    os.replace(tmp, out)
    return res


def require_pass(alias: str) -> None:
    path = layout.base() / "voice_check.json"
    if not path.exists():
        raise RuntimeError(f"{path} missing (run: ./python -m synthetic voicecheck)")
    r = json.loads(path.read_text())["engines"].get(alias)
    if not r or r["verdict"] != "pass":
        raise RuntimeError(f"{alias} did not pass the voice-provenance check; it is not used")
