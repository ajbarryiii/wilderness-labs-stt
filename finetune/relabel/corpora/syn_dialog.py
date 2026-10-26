"""`syn-dialog`: the voice-provenance check (labeling plan "New corpora").

The corpus is used only if every reference voice is confirmed to come from the voice corpus's TRAINING
speakers (that corpus's dev/test speakers are this project's dev/test speakers). The check:

1. reads the voice bank's metadata at its pinned revision (registry `voice_bank`): speaker id and subset;
2. inspects the corpus's own records for anything that names a voice (keys matching VOICE_KEY in the
   per-dialog JSON): the first RECORDS_PER_SHARD JSON members of every training shard, found by walking the
   tar headers with range requests (no audio member is read);
3. decides:
   - records name voices, every record was inspected, and every named voice is a bank training speaker: pass;
   - records name voices outside the training subsets, or unknown voices: fail;
   - records name no voices: pass only if the bank itself holds training speakers only; otherwise the
     check cannot be established;
   - records name voices but only a sample was inspected: not established (a full scan would be needed).
   Anything but a pass marks the corpus dropped (status.json) and nothing else is done for it.

If the check passes, status.json says "check_passed"; segmenting the corpus is then the next step (not part
of this module: the finalize step refuses a corpus that is neither built nor dropped).
"""
from __future__ import annotations

import collections
import csv
import io
import json
import re
import tarfile
from collections.abc import Callable, Sequence

from pool import common

VOICE_KEY = re.compile(r"(?i)(voice|speaker_?id|ref(erence)?_?(audio|wav|clip|speaker)|libri|clone|tts_?speaker)")
RECORDS_PER_SHARD = 8    # JSON members inspected per training shard
MAX_MEMBERS = 200        # tar members walked per shard at most


def decide(bank: Sequence[dict], id_field: str, subset_field: str, training_subsets: Sequence[str],
           record_voices: Sequence[set[str]], records_inspected: int, all_records_inspected: bool) -> dict:
    train = set(training_subsets)
    subsets = collections.Counter(str(r[subset_field]) for r in bank)
    ids = {str(r[id_field]) for r in bank}
    non_train = {str(r[id_field]) for r in bank if str(r[subset_field]) not in train}
    named = [v for v in record_voices if v]
    info = {"bank_speakers": len(ids), "bank_subsets": dict(sorted(subsets.items())),
            "bank_non_training_speakers": len(non_train), "records_inspected": records_inspected,
            "records_naming_voices": len(named), "all_records_inspected": all_records_inspected}
    if named:
        used = set().union(*named)
        unknown, bad = used - ids, used & non_train
        info.update(voices_named=len(used), voices_unknown=len(unknown), voices_non_training=len(bad))
        if bad or unknown:
            return info | {"passed": False, "verdict": "failed",
                           "reason": f"{len(bad)} named voices are not training speakers, {len(unknown)} unknown"}
        if not all_records_inspected or len(named) != records_inspected:
            return info | {"passed": False, "verdict": "not_established",
                           "reason": "voices are named, but not every record was inspected / names a voice"}
        return info | {"passed": True, "verdict": "passed", "reason": "every named voice is a training speaker"}
    if not non_train:
        return info | {"passed": True, "verdict": "passed",
                       "reason": "the records name no voices, and every bank voice is a training speaker"}
    return info | {"passed": False, "verdict": "not_established",
                   "reason": (f"the records name no voices (the voice-to-speaker mapping is not published), and the "
                              f"voice bank holds {len(non_train)} non-training speakers of {len(ids)}, so it cannot "
                              "be established that every reference voice is a training speaker")}


def record_voice_ids(meta: dict) -> set[str]:
    """Values of voice-identifying keys anywhere in one record's JSON."""
    out: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if VOICE_KEY.search(str(k)) and isinstance(v, (str, int)) and str(v).strip():
                    out.add(str(v))
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(meta)
    return out


def json_members(read: Callable[[int, int], bytes], limit: int = RECORDS_PER_SHARD) -> list[dict]:
    """The first `limit` JSON members of a tar file, reading only headers and JSON contents
    (`read(start, end)` returns bytes [start, end) of the tar)."""
    out, offset = [], 0
    for _ in range(MAX_MEMBERS):
        header = read(offset, offset + 512)
        if len(header) < 512 or header == b"\0" * 512:
            break
        info = tarfile.TarInfo.frombuf(header, tarfile.ENCODING, "surrogateescape")
        start = offset + 512
        if info.isfile() and info.name.endswith(".json"):
            out.append(json.loads(read(start, start + info.size)))
            if len(out) >= limit:
                break
        offset = start + (info.size + 511) // 512 * 512
    return out


def _hub_get(hf_id: str, revision: str, path: str, byte_range: tuple[int, int] | None = None) -> bytes:
    import requests
    from huggingface_hub import get_token, hf_hub_url
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1] - 1}"
    r = requests.get(hf_hub_url(hf_id, path, repo_type="dataset", revision=revision), headers=headers, timeout=120)
    r.raise_for_status()
    return r.content


def fetch_inputs(reg: dict, get: Callable = _hub_get) -> tuple[list[dict], list[set[str]], int, list[str]]:
    vb = reg["voice_bank"]
    bank = list(csv.DictReader(io.StringIO(get(vb["hf_id"], vb["revision"], vb["metadata_file"]).decode())))
    voices, keys = [], set()
    for shard in reg["train_files"]:
        def read(start, end, shard=shard):
            return get(reg["hf_id"], reg["revision"], shard, (start, end))
        for meta in json_members(read):
            voices.append(record_voice_ids(meta))
            stack = [meta]
            while stack:
                o = stack.pop()
                if isinstance(o, dict):
                    keys.update(map(str, o))
                    stack.extend(o.values())
                elif isinstance(o, list):
                    stack.extend(o)
    return bank, voices, len(voices), sorted(keys)


def run_check(out_dir, reg: dict, get: Callable = _hub_get) -> dict:
    bank, voices, n, keys = fetch_inputs(reg, get)
    if n == 0:
        raise RuntimeError("no corpus records could be inspected")
    vb = reg["voice_bank"]
    verdict = decide(bank, vb["id_field"], vb["subset_field"], vb["training_subsets"], voices, n,
                     all_records_inspected=False)
    verdict |= {"record_json_keys": keys, "checked": common.now(), "records_per_shard": RECORDS_PER_SHARD,
                "shards": len(reg["train_files"])}
    common.atomic_write_json(out_dir / "voice_check.json", verdict)
    if verdict["passed"]:
        status = {"status": "check_passed", "voice_check": str(out_dir / "voice_check.json"), "created": common.now()}
    else:
        status = {"status": "dropped", "reason": f"voice provenance {verdict['verdict']}: {verdict['reason']}",
                  "voice_check": str(out_dir / "voice_check.json"), "created": common.now()}
    common.atomic_write_json(out_dir / "status.json", status)
    return verdict
