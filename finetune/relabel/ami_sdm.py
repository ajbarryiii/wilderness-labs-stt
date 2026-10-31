"""The AMI distant-microphone (SDM) pair store of stage 4 (keab-prime plan "Data", "AMI distant microphone (new)").
CPU only; every step skips work already done, so a rerun after an interruption resumes.

    ./python ami_sdm.py download             the Hub files at the pinned revision into the HF cache (network)
    ./python ami_sdm.py index [--workers N]  every row's metadata; C1's AMI ids; the label store's AMI dispositions
    ./python ami_sdm.py select               pairing, held-out exclusion, label coverage
    ./python ami_sdm.py decode [--workers N] FLAC per selected recording (SDM and IHM)
    ./python ami_sdm.py finalize             frame check, C1 PCM check, the store (clips.jsonl, clips.meta.json)
    ./python ami_sdm.py dev [--workers N]    the AMI SDM dev manifest (informational, never trained on)
    ./python ami_sdm.py draws                draws per clip from C1's journal, and the SDM hours
    ./python ami_sdm.py all [--workers N]    every step in order
    ./python ami_sdm.py status

Everything is under $RELABEL_ROOT/corpora/ami-sdm (root()).

Source (download): `edinburghcstr/ami` at the pool's pinned revision REVISION (CC BY 4.0, not gated), config `sdm`
(train and validation) and config `ihm` (train: the rows the pool streamed). The listing (path, size, LFS SHA-256)
is written once to fetch/files.json; each file goes into the HF cache under $RELABEL_ROOT/caches (relabel_boot.py
sets HF_HOME, HF_HUB_CACHE and HF_DATASETS_CACHE there) and is checked against its size and SHA-256 before its
fetch/done/<key>.json is written. A gated or private revision stops the download: terms are never accepted on AJ's
behalf.

Index: index/<key>.jsonl per Hub file, one line per row (file, row, audio_id, meeting_id, speaker_id,
microphone_id, begin_time, end_time as stored (float32), text, and the audio header's frames, rate and channels).
index/c1-ami.jsonl: every AMI id ("ami:" + IHM audio_id, labeling/pool_stream.py) occurring in C1's journal over
consumed batches [0, THROUGH_BATCH) (pools 0-781), with its occurrence count and C1's stored samples and PCM
SHA-256. index/labels-ami.jsonl: the stored disposition of every AMI id in the stage 2 label run LABELS_RUN (the
label store's own checks: run tied to C1, label shard SHA-256 against its meta, record counts, teacher).

Select (selection/): pair() each SDM train row with the IHM train row of the same (meeting_id, speaker_id,
begin_time, end_time), compared exactly; unpaired rows by reason (UNPAIRED). Then select(): held out (a meeting of
the ami_dev manifest or of the ESB AMI test manifest; the train meetings must be disjoint from both, check_disjoint),
then labels (the IHM id must have a label record kept under T*'s disposition). The global denylist names no AMI
recording (AMI has no recording key in the stream; checked).

Decode (store/): the selected SDM rows of each SDM train file and the selected IHM rows of each IHM train file,
decoded by the pool's path (parakeet-ternary data.convert: soundfile, int16 for PCM_16 at 16 kHz mono) and written
as 16 kHz mono PCM_16 FLAC (store/audio/<config>/<key>/<audio_id>.flac); store/decode/<key>.jsonl (written last)
holds each row's record: samples, SHA-256 of the FLAC bytes, PCM SHA-256, or the drop reason. store/decode.json ties
the units to the selection; a unit is done when its record file lists exactly its ids.

Finalize: assemble() drops a pair with an undecodable side or unequal encoder frame counts (training/ami_store.py
encoder_frames, the stage 3 trainer's NeMo formula), and checks each kept IHM clip found in C1 against C1's stored
samples and PCM SHA-256 (any mismatch stops the build: require_c1_match). Every FLAC file is re-read and checked
against its record. Then training/ami_store.write_store writes store/clips.jsonl and store/clips.meta.json, and
REPORT.json summarizes the counts.

Dev (dev/): the SDM validation rows of the ami_dev manifest's meetings with a nonempty transcript, decoded the same
way into dev/audio/, and dev/ami-sdm-dev.jsonl in the dev manifests' format (parakeet-ternary data.MANIFEST_KEYS:
audio_filepath, duration, text = the row's human transcript as ami_dev's, id "ami-sdm:<audio_id>", source
"ami-sdm"), with dev/ami-sdm-dev.meta.json.

Draws (draws.json): the plan's share rule, "each SDM clip is drawn as many times as the average AMI IHM train id
occurs in pools 0-781 (rounded)": the AMI occurrences of C1's journal over [0, THROUGH_BATCH) divided by the distinct
AMI ids occurring there, rounded half up; with the resulting SDM hours (draws x the store's SDM hours). Reported
only; the batch plan is built later.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

REPO = "edinburghcstr/ami"
REVISION = "46f28f2503e2ec48f8867a84eef356c70476beab"      # the pool's pinned revision (parakeet-ternary data.py)
LICENSE = "CC-BY-4.0"
FILES = (("sdm", "train"), ("ihm", "train"), ("sdm", "validation"))   # (config, split) downloaded
EXPECTED_ROWS = {("sdm", "train"): 107_319, ("ihm", "train"): 108_502}   # keab-prime plan "AMI distant microphone"
DOWNLOAD_THREADS = 4
WORKERS = 8
LABELS_RUN = "main-f3"                 # stage 2's label run (keab-prime plan "Inputs")
THROUGH_BATCH = 100_000                # consumed batches [0, 100,000) = pools 0-781 of 128 batches
DEV_MANIFEST = "ami_dev"               # $PARAKEET_TERNARY_ROOT/manifests/ami_dev.jsonl (H4's AMI dev, IHM)
TEST_MANIFEST = "test_ami"             # $PARAKEET_TERNARY_ROOT/manifests/test_ami.jsonl (the ESB AMI test set)
DEV_NAME = "ami-sdm-dev"
SR = 16000
INDEX_COLUMNS = ("audio_id", "meeting_id", "speaker_id", "microphone_id", "begin_time", "end_time", "text")
UNPAIRED = ("no_ihm_meeting", "no_ihm_speaker", "no_ihm_time", "ambiguous_ihm_key", "ambiguous_sdm_key")
SELECT_DROPS = ("held_out_dev", "held_out_test", "no_label_record", "not_kept")
ASSEMBLE_DROPS = ("sdm_undecodable", "ihm_undecodable", "frames_differ")
_AMI_ID = re.compile(r"^(?:[A-Za-z0-9-]+:)?AMI_([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)_(\d+)_(\d+)$")


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"{stamp} [ami-sdm pid {os.getpid()}] {msg}", flush=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _sha_json(obj) -> str:
    return _sha256_bytes(json.dumps(obj, sort_keys=True).encode())


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    from pool import common
    common.atomic_write_lines(path, (json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows))


def _read_jsonl(path: Path) -> list[dict]:
    from pool import common
    return common.read_jsonl(path)


# --- layout --------------------------------------------------------------------------------------------------------

def root() -> Path:
    """$RELABEL_ROOT/corpora/ami-sdm (on the data mount)."""
    from labeling import layout
    layout.require_mount()
    return layout.corpora_dir() / "ami-sdm"


def fetch_dir() -> Path:
    return root() / "fetch"


def file_key(path: str) -> str:
    """'sdm/train-00003-of-00027.parquet' -> 'sdm-train-00003-of-00027'."""
    config, name = path.split("/")
    return f"{config}-{name.removesuffix('.parquet')}"


# --- ids and rules (the contract) ----------------------------------------------------------------------------------

def meeting_of(audio_id: str) -> str:
    """The meeting of an AMI audio id, bare or with a source prefix ("ami:AMI_ES2011a_H00_FEE041_0003427_0003714")."""
    m = _AMI_ID.match(audio_id or "")
    if not m:
        raise ValueError(f"not an AMI audio id: {audio_id!r}")
    return m.group(1)


def ihm_pool_id(audio_id: str) -> str:
    """The pool's, the label store's and C1's id of an IHM row: "ami:" + its audio_id."""
    from training import ami_store
    return ami_store.IHM_PREFIX + audio_id


def held_out_meetings(dev_manifest: Path, test_manifest: Path) -> dict[str, set[str]]:
    """{"dev": the ami_dev manifest's meetings, "test": the ESB AMI test manifest's}, from their row ids."""
    out = {}
    for kind, path in (("dev", dev_manifest), ("test", test_manifest)):
        out[kind] = {meeting_of(r["id"]) for r in _read_jsonl(Path(path))}
        if not out[kind]:
            raise RuntimeError(f"{path}: no rows")
    return out


def check_disjoint(train_meetings: set[str], held: dict[str, set[str]]) -> None:
    """RuntimeError if a train meeting is a held-out (dev or test) meeting (stop and report)."""
    both = {k: sorted(set(train_meetings) & v) for k, v in held.items()}
    if any(both.values()):
        raise RuntimeError(f"train meetings that are held out: {both}")


def pair_key(r: dict) -> tuple:
    return (r["meeting_id"], r["speaker_id"], float(r["begin_time"]), float(r["end_time"]))


def pair(sdm_rows: Sequence[dict], ihm_rows: Sequence[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    """([{"sdm": row, "ihm": row}], {reason: [SDM audio ids]} for every reason of UNPAIRED). An SDM row pairs with the
    one IHM row of the same pair_key; a key held by two SDM rows, or by two IHM rows, pairs none of them."""
    by_key = collections.defaultdict(list)
    for r in ihm_rows:
        by_key[pair_key(r)].append(r)
    meetings = {r["meeting_id"] for r in ihm_rows}
    speakers = {(r["meeting_id"], r["speaker_id"]) for r in ihm_rows}
    sdm_count = collections.Counter(pair_key(r) for r in sdm_rows)
    pairs, unpaired = [], {k: [] for k in UNPAIRED}
    for r in sdm_rows:
        k = pair_key(r)
        if sdm_count[k] > 1:
            unpaired["ambiguous_sdm_key"].append(r["audio_id"])
        elif k in by_key:
            if len(by_key[k]) > 1:
                unpaired["ambiguous_ihm_key"].append(r["audio_id"])
            else:
                pairs.append({"sdm": r, "ihm": by_key[k][0]})
        elif r["meeting_id"] not in meetings:
            unpaired["no_ihm_meeting"].append(r["audio_id"])
        elif (r["meeting_id"], r["speaker_id"]) not in speakers:
            unpaired["no_ihm_speaker"].append(r["audio_id"])
        else:
            unpaired["no_ihm_time"].append(r["audio_id"])
    return pairs, unpaired


def select(pairs: Sequence[dict], held: dict[str, set[str]], dispositions: dict[str, str]
           ) -> tuple[list[dict], dict[str, int]]:
    """(selected pairs sorted by SDM id, {reason: count} for every reason of SELECT_DROPS). In order: a pair from a
    dev or test meeting is held out; then its IHM id must have a label record (dispositions: the label store's id ->
    T*'s stored disposition) that is "kept"."""
    selected, drops = [], dict.fromkeys(SELECT_DROPS, 0)
    for p in pairs:
        s, h = p["sdm"], p["ihm"]
        meeting = s["meeting_id"]
        if meeting in held["dev"]:
            drops["held_out_dev"] += 1
            continue
        if meeting in held["test"]:
            drops["held_out_test"] += 1
            continue
        hid = ihm_pool_id(h["audio_id"])
        disposition = dispositions.get(hid)
        if disposition is None:
            drops["no_label_record"] += 1
            continue
        if disposition != "kept":
            drops["not_kept"] += 1
            continue
        selected.append({"sdm_id": s["audio_id"], "sdm_file": s.get("file"), "sdm_row": s.get("row"),
                         "ihm_audio_id": h["audio_id"], "ihm_id": hid, "ihm_file": h.get("file"),
                         "ihm_row": h.get("row"), "meeting_id": meeting, "speaker_id": s["speaker_id"],
                         "begin_time": float(s["begin_time"]), "end_time": float(s["end_time"])})
    return sorted(selected, key=lambda x: x["sdm_id"]), drops


def assemble(selected: Sequence[dict], sdm_records: dict[str, dict], ihm_records: dict[str, dict],
             c1: dict[str, dict]) -> tuple[list[dict], dict]:
    """(clip records of the store, sorted by id; report). A pair is dropped if its SDM or IHM recording did not
    decode, or if their encoder frame counts differ (ASSEMBLE_DROPS, all counted). Each kept pair's IHM clip found
    in C1 (c1: pool id -> {samples, pcm_sha256}) is checked against C1's samples and PCM SHA-256; the report lists
    the mismatched ids (require_c1_match refuses them)."""
    from training import ami_store
    clips, drops = [], dict.fromkeys(ASSEMBLE_DROPS, 0)
    reasons = {"sdm": collections.Counter(), "ihm": collections.Counter()}
    differ = []
    c1_report = {"checked": 0, "matched": 0, "mismatched": [], "not_in_c1": 0}
    for p in selected:
        s, h = sdm_records[p["sdm_id"]], ihm_records[p["ihm_audio_id"]]
        if s["status"] != "ok":
            drops["sdm_undecodable"] += 1
            reasons["sdm"][s.get("reason")] += 1
            continue
        if h["status"] != "ok":
            drops["ihm_undecodable"] += 1
            reasons["ihm"][h.get("reason")] += 1
            continue
        fs, fh = ami_store.encoder_frames(s["samples"]), ami_store.encoder_frames(h["samples"])
        if fs != fh:
            drops["frames_differ"] += 1
            differ.append({"id": p["sdm_id"], "samples": s["samples"], "ihm_samples": h["samples"],
                           "frames": fs, "ihm_frames": fh})
            continue
        ref = c1.get(p["ihm_id"])
        if ref is None:
            c1_report["not_in_c1"] += 1
        else:
            c1_report["checked"] += 1
            if ref["samples"] == h["samples"] and ref["pcm_sha256"] == h["pcm_sha256"]:
                c1_report["matched"] += 1
            else:
                c1_report["mismatched"].append(p["ihm_id"])
        clips.append({"id": p["sdm_id"], "audio_filepath": s["audio_filepath"], "duration": s["samples"] / SR,
                      "samples": s["samples"], "sha256": s["sha256"], "pcm_sha256": s["pcm_sha256"], "frames": fs,
                      "ihm_id": p["ihm_id"], "ihm_audio_filepath": h["audio_filepath"], "ihm_samples": h["samples"],
                      "ihm_sha256": h["sha256"], "ihm_pcm_sha256": h["pcm_sha256"], "meeting_id": p["meeting_id"],
                      "speaker_id": p["speaker_id"], "begin_time": p["begin_time"], "end_time": p["end_time"]})
    c1_report["mismatched"].sort()
    report = {"drops": drops, "undecodable_reasons": {k: dict(sorted(v.items(), key=str)) for k, v in reasons.items()},
              "frames_differ": differ, "c1": c1_report}
    return sorted(clips, key=lambda c: c["id"]), report


def require_c1_match(report: dict) -> None:
    bad = report["c1"]["mismatched"]
    if bad:
        raise RuntimeError(f"{len(bad)} IHM clips differ from C1's stored PCM (first: {bad[:5]}); stop and report")


def draws_per_clip(occurrences: Iterable[int]) -> dict:
    """{"ids", "occurrences", "mean", "draws"}: the mean occurrences per id, rounded half up."""
    occ = [int(x) for x in occurrences]
    if not occ:
        raise ValueError("no occurrences")
    total = sum(occ)
    mean = total / len(occ)
    return {"ids": len(occ), "occurrences": total, "mean": mean, "draws": math.floor(mean + 0.5)}


def dev_wanted(rows: Sequence[dict], meetings: set[str]) -> list[str]:
    """The audio ids of the SDM validation rows of the dev meetings with a nonempty transcript (to decode)."""
    return [r["audio_id"] for r in rows if r["meeting_id"] in meetings and (r.get("text") or "").strip()]


def dev_rows(rows: Sequence[dict], records: dict[str, dict], meetings: set[str]) -> tuple[list[dict], dict]:
    """(manifest rows sorted by id, counts) of the AMI SDM dev set: the rows of the dev meetings with a nonempty
    transcript that decoded (ami_dev's filters: decodable, nonempty, finite, not all zeros; nonempty text)."""
    from training import ami_store
    counts = collections.Counter()
    out = []
    for r in rows:
        if r["meeting_id"] not in meetings:
            counts["not_dev_meeting"] += 1
            continue
        text = (r.get("text") or "").strip()
        if not text:
            counts["empty_text"] += 1
            continue
        rec = records[r["audio_id"]]
        if rec["status"] != "ok":
            counts[rec.get("reason") or "undecodable"] += 1
            continue
        out.append({"audio_filepath": rec["audio_filepath"], "duration": rec["samples"] / SR, "text": text,
                    "id": ami_store.PREFIX + r["audio_id"], "source": ami_store.ALIAS})
    return sorted(out, key=lambda x: x["id"]), {"rows": len(rows), **dict(sorted(counts.items())), "kept": len(out)}


# --- reading the Hub files -----------------------------------------------------------------------------------------

def index_file(local: Path, hub_path: str) -> list[dict]:
    """One record per row of a Hub parquet file (module docstring "Index"); the audio is only probed."""
    import pyarrow.parquet as pq
    import soundfile as sf
    out = []
    pf = pq.ParquetFile(local)
    for batch in pf.iter_batches(batch_size=64, columns=list(INDEX_COLUMNS) + ["audio"], use_threads=False):
        for r in batch.to_pylist():
            data = (r.pop("audio") or {}).get("bytes")
            frames = rate = channels = None
            if data:
                try:
                    info = sf.info(io.BytesIO(data))
                    frames, rate, channels = int(info.frames), int(info.samplerate), int(info.channels)
                except Exception:
                    pass
            r.update(file=hub_path, row=len(out), header_frames=frames, header_rate=rate, header_channels=channels,
                     begin_time=float(r["begin_time"]), end_time=float(r["end_time"]))
            out.append(r)
    if len(out) != pf.metadata.num_rows:
        raise RuntimeError(f"{hub_path}: read {len(out)} rows of {pf.metadata.num_rows}")
    return out


def _safe(audio_id: str) -> str:
    import data  # finetune/parakeet-ternary
    return data.safe_name(audio_id)


def decode_file(local: Path, wanted: Sequence[str], out_dir: Path) -> list[dict]:
    """Decode the wanted rows of a Hub parquet file (by audio id) to FLAC files in out_dir; one record per wanted
    row, in file order: {"id", "status": "ok", "audio_filepath", "samples", "sha256", "pcm_sha256"} or
    {"id", "status": "undecodable", "reason"} (data.convert's reason). RuntimeError if a wanted id is not in the
    file exactly once."""
    import numpy as np
    import pyarrow.parquet as pq

    import data  # finetune/parakeet-ternary: the pool's decoding
    from labeling import shards
    from pool import common
    want = set(wanted)
    if len(want) != len(wanted):
        raise RuntimeError("a wanted id is listed twice")
    out_dir = Path(out_dir)
    recs, seen = [], set()
    pf = pq.ParquetFile(local)
    for batch in pf.iter_batches(batch_size=32, columns=["audio_id", "audio"], use_threads=False):
        for r in batch.to_pylist():
            aid = r["audio_id"]
            if aid not in want:
                continue
            if aid in seen:
                raise RuntimeError(f"{local}: {aid} twice")
            seen.add(aid)
            raw = (r.get("audio") or {}).get("bytes")
            if not raw:
                recs.append({"id": aid, "status": "undecodable", "reason": "empty_audio"})
                continue
            pcm, reason, _ = data.convert(raw)
            if reason is not None:
                recs.append({"id": aid, "status": "undecodable", "reason": reason})
                continue
            flac = shards.encode_flac(pcm)
            if not np.array_equal(shards.decode_flac(flac), pcm):
                raise RuntimeError(f"{aid}: FLAC round trip changed the samples")
            path = out_dir / f"{_safe(aid)}.flac"
            common.atomic_write_bytes(path, flac)
            recs.append({"id": aid, "status": "ok", "audio_filepath": str(path), "samples": int(len(pcm)),
                         "sha256": _sha256_bytes(flac), "pcm_sha256": shards.pcm_sha256(pcm)})
    missing = want - seen
    if missing:
        raise RuntimeError(f"{local}: {len(missing)} wanted ids not in the file (first {sorted(missing)[:3]})")
    return recs


# --- the C1 journal and the label store ----------------------------------------------------------------------------

def c1_ami(pool_dir: Path, through_batch: int, prefix: str = "ami:") -> dict[str, dict]:
    """{id: {"occurrences", "samples", "pcm_sha256"}} of every id with `prefix` occurring in C1's journal over
    consumed batches [0, through_batch): occurrences counted in the journal, samples and PCM SHA-256 from the audio
    file of the shard where the id first occurs. Journals are checked against their metas' SHA-256 and the audio
    files against their sizes (shards.verify_shard without hashing ~0.7 TB of audio)."""
    import pyarrow.parquet as pq

    from labeling import shards
    from training.labels import needed_shards
    pool_dir = Path(pool_dir)
    occ: collections.Counter = collections.Counter()
    audio: dict[str, dict] = {}
    for k in needed_shards(pool_dir, through_batch):
        meta = shards.verify_shard(pool_dir, k, audio_hash=False)
        if meta.get("kind") != "replay" or meta.get("audio_deleted"):
            raise RuntimeError(f"C1 shard {k} is not a replay shard with audio")
        p = shards.paths(pool_dir, k)
        j = pq.read_table(p["journal"], columns=["batch", "id"]).to_pydict()
        for b, i in zip(j["batch"], j["id"]):
            if b < through_batch and i.startswith(prefix):
                occ[i] += 1
        a = pq.read_table(p["audio"], columns=["id", "samples", "pcm_sha256"]).to_pydict()
        for i, n, sha in zip(a["id"], a["samples"], a["pcm_sha256"]):
            if i.startswith(prefix):
                if i in audio:
                    raise RuntimeError(f"C1: {i} has audio in two shards")
                audio[i] = {"samples": int(n), "pcm_sha256": sha, "shard": k}
    missing = set(occ) - set(audio)
    if missing:
        raise RuntimeError(f"C1: {len(missing)} journal ids without audio (first {sorted(missing)[:3]})")
    return {i: {"occurrences": occ[i], **audio[i]} for i in sorted(occ)}


def label_dispositions(run_dir: Path, pool_dir: Path, through_batch: int, prefix: str = "ami:") -> dict[str, str]:
    """{id: T*'s stored disposition} of every label record with `prefix` in the label run over the C1 shards of
    consumed batches [0, through_batch), with the label store's checks (training/labels.py LabelStore.open): the
    run labels this C1 directory, every label shard of the range exists and matches its meta's SHA-256 and record
    count, every record is the run's teacher's, no id is recorded twice."""
    import pyarrow.parquet as pq

    from labeling import c2, shards
    from training.labels import needed_shards
    run_dir, pool_dir = Path(run_dir), Path(pool_dir)
    run = c2.load_run(run_dir)
    if c2._abs(run["pool_dir"]).resolve() != pool_dir.resolve():
        raise RuntimeError(f"{run_dir} labels {run['pool_dir']}, not {pool_dir}")
    out: dict[str, str] = {}
    for k in needed_shards(pool_dir, through_batch):
        path = run_dir / "shards" / f"{shards.shard_name(k)}.labels.parquet"
        meta_path = path.with_name(f"{shards.shard_name(k)}.json")
        if not path.exists() or not meta_path.exists():
            raise RuntimeError(f"no label shard {path.name}")
        meta = json.loads(meta_path.read_text())
        if shards.sha256_file(path) != meta["sha256"]:
            raise RuntimeError(f"{path}: SHA-256 differs from its meta")
        t = pq.read_table(path, columns=["id", "disposition", "teacher"]).to_pydict()
        if len(t["id"]) != meta["records"]:
            raise RuntimeError(f"{path}: {len(t['id'])} records, meta says {meta['records']}")
        for i, d, teacher in zip(t["id"], t["disposition"], t["teacher"]):
            if teacher != run["teacher"]:
                raise RuntimeError(f"{path}: record of teacher {teacher!r} in a {run['teacher']} run")
            if i.startswith(prefix):
                if i in out:
                    raise RuntimeError(f"id {i} has two label records")
                out[i] = d
    return out


# --- download ------------------------------------------------------------------------------------------------------

def _hub_cache() -> Path:
    from labeling import layout
    cache = Path(os.environ.get("HF_HUB_CACHE", ""))
    if not cache.is_absolute() or not cache.resolve().is_relative_to(layout.relabel_root().resolve()):
        raise RuntimeError(f"HF_HUB_CACHE {cache} is not under $RELABEL_ROOT (run through ./python)")
    return cache


def check_not_gated() -> dict:
    """The revision's dataset info; RuntimeError if the Hub gates it (stop and report; never accept terms)."""
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(REPO, revision=REVISION)
    if info.gated or info.private:
        raise RuntimeError(f"{REPO}@{REVISION} is gated or private ({info.gated!r}): stop and report to AJ; "
                           "terms are never accepted on his behalf")
    if info.sha != REVISION:
        raise RuntimeError(f"{REPO}: revision resolved to {info.sha}, not {REVISION}")
    return {"gated": info.gated, "private": info.private, "sha": info.sha}


def listing() -> dict:
    """fetch/files.json: the files of FILES at REVISION (path, size, sha256), written once and then reused."""
    from pool import common
    path = fetch_dir() / "files.json"
    if path.exists():
        return json.loads(path.read_text())
    from huggingface_hub import HfApi
    api = HfApi()
    gate = check_not_gated()
    files = []
    for config, split in FILES:
        entries = [f for f in api.list_repo_tree(REPO, path_in_repo=config, repo_type="dataset", revision=REVISION)
                   if f.path.endswith(".parquet") and f.path.split("/")[-1].startswith(f"{split}-")]
        if not entries:
            raise RuntimeError(f"no {config}/{split} parquet files at {REVISION}")
        files += [{"config": config, "split": split, "path": f.path, "size": int(f.size),
                   "sha256": f.lfs.sha256 if f.lfs else None} for f in entries]
    if any(f["sha256"] is None for f in files):
        raise RuntimeError("a listed parquet file has no LFS SHA-256")
    body = {"repo": REPO, "revision": REVISION, "license": LICENSE, "gate": gate, "listed": common.now(),
            "files": sorted(files, key=lambda f: f["path"])}
    common.atomic_write_json(path, body)
    return body


def done_path(entry: dict) -> Path:
    return fetch_dir() / "done" / f"{file_key(entry['path'])}.json"


def local_file(entry: dict) -> Path:
    """The checked local copy of a listed file (RuntimeError if it was not downloaded and checked)."""
    marker = done_path(entry)
    if not marker.exists():
        raise RuntimeError(f"{entry['path']} is not downloaded yet (./python ami_sdm.py download)")
    body = json.loads(marker.read_text())
    local = Path(body["local"])
    if body["sha256"] != entry["sha256"] or not local.exists() or local.stat().st_size != entry["size"]:
        raise RuntimeError(f"{entry['path']}: the local copy no longer matches the listing")
    return local


def download_one(entry: dict, attempts: int = 8) -> dict:
    """Download one listed file into the HF cache and check it; skipped when its done marker exists."""
    from huggingface_hub import hf_hub_download

    from pool import common
    marker = done_path(entry)
    if marker.exists():
        return json.loads(marker.read_text()) | {"skipped": True}
    for attempt in range(attempts):
        try:
            t0 = time.monotonic()
            local = Path(hf_hub_download(REPO, entry["path"], repo_type="dataset", revision=REVISION,
                                         cache_dir=str(_hub_cache())))
            if local.stat().st_size != entry["size"]:
                raise OSError(f"size {local.stat().st_size} != {entry['size']}")
            if _sha256(local) != entry["sha256"]:
                raise OSError("SHA-256 differs from the listing")
            body = {"path": entry["path"], "size": entry["size"], "sha256": entry["sha256"], "local": str(local),
                    "seconds": round(time.monotonic() - t0, 1), "checked": common.now()}
            common.atomic_write_json(marker, body)
            return body
        except Exception as exc:   # network errors, truncated files: retry with backoff
            if attempt == attempts - 1:
                raise
            wait = min(600, 15 * 2 ** attempt)
            log(f"download {entry['path']} failed ({exc!r}); retry {attempt + 1} in {wait} s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def cmd_download(threads: int = DOWNLOAD_THREADS) -> int:
    body = listing()
    files = body["files"]
    if all(done_path(f).exists() for f in files):
        log(f"download: all {len(files)} files already checked")
        return 0
    check_not_gated()
    log(f"{len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB at {REVISION}")
    done = 0
    with cf.ThreadPoolExecutor(threads) as pool:
        for result in pool.map(download_one, files):
            done += 1
            log(f"{done}/{len(files)} {result['path']} {'already checked' if result.get('skipped') else 'checked'}")
    return 0


# --- steps ---------------------------------------------------------------------------------------------------------

def _files(config: str, split: str) -> list[dict]:
    return [f for f in listing()["files"] if f["config"] == config and f["split"] == split]


def _executor(workers: int):
    import multiprocessing as mp
    return cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker)


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(2)


def index_path(entry: dict) -> Path:
    return root() / "index" / f"{file_key(entry['path'])}.jsonl"


def _index_task(entry: dict) -> tuple[str, int]:
    path = index_path(entry)
    if not path.exists():
        rows = index_file(local_file(entry), entry["path"])
        _write_jsonl(path, rows)
    return entry["path"], sum(1 for _ in open(path, encoding="utf-8"))


def read_index(config: str, split: str) -> list[dict]:
    rows = []
    for f in _files(config, split):
        path = index_path(f)
        if not path.exists():
            raise RuntimeError(f"no index of {f['path']} yet (./python ami_sdm.py index)")
        rows += _read_jsonl(path)
    return rows


def c1_dir() -> Path:
    from labeling.__main__ import c1_dir as c1
    return c1()


def labels_run_dir() -> Path:
    from training import labels as tlabels
    return tlabels.production_run_dir(LABELS_RUN)


def cmd_index(workers: int) -> int:
    from pool import common
    files = listing()["files"]
    with _executor(workers) as pool:
        for hub_path, n in pool.map(_index_task, files):
            log(f"index {hub_path}: {n} rows")
    for (config, split), want in EXPECTED_ROWS.items():
        got = sum(1 for f in _files(config, split) for _ in open(index_path(f), encoding="utf-8"))
        if got != want:
            raise RuntimeError(f"{config} {split}: {got} rows, the plan says {want}")
        log(f"{config} {split}: {got} rows (as the plan says)")
    base = root() / "index"
    if not (base / "c1-ami.json").exists():
        t0 = time.monotonic()
        pool_dir = c1_dir()
        c1 = c1_ami(pool_dir, THROUGH_BATCH)
        _write_jsonl(base / "c1-ami.jsonl", ({"id": i, **v} for i, v in c1.items()))
        common.atomic_write_json(base / "c1-ami.json", {
            "pool_dir": str(pool_dir), "through_batch": THROUGH_BATCH, "ids": len(c1),
            "occurrences": sum(v["occurrences"] for v in c1.values()),
            "jsonl_sha256": _sha256(base / "c1-ami.jsonl"), "seconds": round(time.monotonic() - t0, 1)})
        log(f"C1: {len(c1)} AMI ids over consumed batches [0, {THROUGH_BATCH})")
    if not (base / "labels-ami.json").exists():
        t0 = time.monotonic()
        disp = label_dispositions(labels_run_dir(), c1_dir(), THROUGH_BATCH)
        _write_jsonl(base / "labels-ami.jsonl", ({"id": i, "disposition": d} for i, d in sorted(disp.items())))
        common.atomic_write_json(base / "labels-ami.json", {
            "labels_run": LABELS_RUN, "run_dir": str(labels_run_dir()), "through_batch": THROUGH_BATCH,
            "records": len(disp), "by_disposition": dict(collections.Counter(disp.values()).most_common()),
            "jsonl_sha256": _sha256(base / "labels-ami.jsonl"), "seconds": round(time.monotonic() - t0, 1)})
        log(f"labels {LABELS_RUN}: {len(disp)} AMI records")
    return 0


def _read_checked(base: Path, name: str) -> list[dict]:
    meta = json.loads((base / f"{name}.json").read_text())
    if _sha256(base / f"{name}.jsonl") != meta["jsonl_sha256"]:
        raise RuntimeError(f"{base / name}.jsonl differs from its meta")
    return _read_jsonl(base / f"{name}.jsonl")


def c1_index() -> dict[str, dict]:
    return {r["id"]: r for r in _read_checked(root() / "index", "c1-ami")}


def dispositions_index() -> dict[str, str]:
    return {r["id"]: r["disposition"] for r in _read_checked(root() / "index", "labels-ami")}


def manifests() -> tuple[Path, Path]:
    from pool import common
    base = common.ternary_manifests()
    return base / f"{DEV_MANIFEST}.jsonl", base / f"{TEST_MANIFEST}.jsonl"


def _check_index_ids(rows: Sequence[dict], what: str) -> None:
    for r in rows:
        if meeting_of(r["audio_id"]) != r["meeting_id"]:
            raise RuntimeError(f"{what}: {r['audio_id']} is not of meeting {r['meeting_id']}")


def _no_time_diagnostics(sdm_rows: Sequence[dict], ihm_rows: Sequence[dict], ids: Sequence[str]) -> dict:
    """For SDM rows whose meeting and speaker have IHM rows but no time match: the nearest IHM row's |begin| + |end|
    difference (seconds), as a histogram, and whether the audio ids' centisecond stamps match some IHM row's."""
    by_ms = collections.defaultdict(list)
    stamps = collections.defaultdict(set)
    for r in ihm_rows:
        by_ms[(r["meeting_id"], r["speaker_id"])].append((r["begin_time"], r["end_time"]))
        m = _AMI_ID.match(r["audio_id"])
        stamps[(r["meeting_id"], r["speaker_id"])].add((m.group(4), m.group(5)))
    sdm = {r["audio_id"]: r for r in sdm_rows}
    hist, stamp_match, examples = collections.Counter(), 0, []
    edges = (0.0, 0.0105, 0.05, 0.1, 0.5, 1.0, 5.0, math.inf)
    for aid in ids:
        r = sdm[aid]
        best = min(abs(b - r["begin_time"]) + abs(e - r["end_time"]) for b, e in by_ms[(r["meeting_id"], r["speaker_id"])])
        for lo, hi in zip(edges, edges[1:]):
            if lo < best <= hi or (best == 0 and lo == 0):
                hist[f"({lo}, {hi}]"] += 1
                break
        m = _AMI_ID.match(aid)
        if (m.group(4), m.group(5)) in stamps[(r["meeting_id"], r["speaker_id"])]:
            stamp_match += 1
        if len(examples) < 10:
            examples.append({"id": aid, "nearest_seconds": round(best, 4)})
    return {"nearest_ihm_begin_plus_end_seconds": dict(hist), "same_centisecond_stamps": stamp_match,
            "examples": examples}


def build_selection() -> tuple[list[dict], dict]:
    """(selected pairs, selection report) from the index (module docstring "Select")."""
    from pool import common
    sdm_rows, ihm_rows = read_index("sdm", "train"), read_index("ihm", "train")
    _check_index_ids(sdm_rows, "sdm train")
    _check_index_ids(ihm_rows, "ihm train")
    dev_path, test_path = manifests()
    held = held_out_meetings(dev_path, test_path)
    train_meetings = {r["meeting_id"] for r in sdm_rows} | {r["meeting_id"] for r in ihm_rows}
    check_disjoint(train_meetings, held)
    deny = json.loads((common.pool_dir() / "denylist.json").read_text())
    if deny.get("recordings", {}).get("ami"):
        raise RuntimeError("the global denylist names AMI recordings; the plan's assumption does not hold (report)")
    pairs, unpaired = pair(sdm_rows, ihm_rows)
    disp = dispositions_index()
    selected, drops = select(pairs, held, disp)
    c1 = c1_index()
    ihm_ids = {ihm_pool_id(r["audio_id"]) for r in ihm_rows}
    paired_ihm = {ihm_pool_id(p["ihm"]["audio_id"]) for p in pairs}
    no_record = [ihm_pool_id(p["ihm"]["audio_id"]) for p in pairs
                 if p["sdm"]["meeting_id"] not in held["dev"] | held["test"]
                 and ihm_pool_id(p["ihm"]["audio_id"]) not in disp]
    ihm_by_id = {ihm_pool_id(r["audio_id"]): r for r in ihm_rows}

    def in_stream_range(i: str) -> bool:   # the stream's 1-30 s filter, on the audio header's frames
        n = ihm_by_id[i]["header_frames"]
        return n is not None and SR <= n <= 30 * SR

    hours = lambda rows: sum((r["header_frames"] or 0) for r in rows) / SR / 3600   # noqa: E731
    report = {
        "rows": {"sdm_train": len(sdm_rows), "ihm_train": len(ihm_rows)},
        "hours_header": {"sdm_train": hours(sdm_rows), "ihm_train": hours(ihm_rows)},
        "meetings": {"sdm_train": len({r["meeting_id"] for r in sdm_rows}),
                     "ihm_train": len({r["meeting_id"] for r in ihm_rows}),
                     "sdm_not_in_ihm": sorted({r["meeting_id"] for r in sdm_rows} - {r["meeting_id"] for r in ihm_rows}),
                     "ihm_not_in_sdm": sorted({r["meeting_id"] for r in ihm_rows} - {r["meeting_id"] for r in sdm_rows})},
        "held_out": {"dev_meetings": len(held["dev"]), "test_meetings": len(held["test"]),
                     "dev_manifest": str(dev_path), "test_manifest": str(test_path),
                     "dev_manifest_sha256": _sha256(dev_path), "test_manifest_sha256": _sha256(test_path),
                     "train_meetings_disjoint": True},
        "denylist_ami_recordings": 0,
        "paired": len(pairs),
        "unpaired": {k: len(v) for k, v in unpaired.items()},
        "unpaired_examples": {k: v[:5] for k, v in unpaired.items()},
        "no_ihm_time": _no_time_diagnostics(sdm_rows, ihm_rows, unpaired["no_ihm_time"]),
        "paired_text_differs": sum((p["sdm"]["text"] or "").strip() != (p["ihm"]["text"] or "").strip() for p in pairs),
        "paired_header_frames_differ": sum(
            p["sdm"]["header_frames"] is None or p["ihm"]["header_frames"] is None
            or p["sdm"]["header_frames"] != p["ihm"]["header_frames"] for p in pairs),
        "select_drops": drops,
        "no_label_record_by_ihm_duration": {
            "outside_1_30s": sum(not in_stream_range(i) for i in no_record),
            "inside_1_30s_not_in_c1": sum(in_stream_range(i) and i not in c1 for i in no_record),
            "inside_1_30s_in_c1": sum(in_stream_range(i) and i in c1 for i in no_record)},
        "selected": len(selected),
        "labels": {"labels_run": LABELS_RUN, "ami_records": len(disp),
                   "ami_records_kept": sum(d == "kept" for d in disp.values()),
                   "ami_records_not_ihm_train": sum(i not in ihm_ids for i in disp),
                   "c1_ami_ids": len(c1), "c1_ami_ids_not_ihm_train": sum(i not in ihm_ids for i in c1),
                   "ihm_train_ids_in_c1": sum(i in c1 for i in ihm_ids),
                   "paired_ihm_ids": len(paired_ihm),
                   "coverage_of_paired_not_held_out":
                       len(selected) / max(1, len(pairs) - drops["held_out_dev"] - drops["held_out_test"])},
    }
    return selected, report


def selection_dir() -> Path:
    return root() / "selection"


def cmd_select() -> int:
    from pool import common
    selected, report = build_selection()
    base = selection_dir()
    data = "".join(json.dumps(s, sort_keys=True) + "\n" for s in selected).encode()
    sha = _sha256_bytes(data)
    path = base / "selection.jsonl"
    if path.exists() and _sha256(path) != sha:
        raise RuntimeError(f"{path} exists with another selection; the decoded store belongs to it (move it aside)")
    common.atomic_write_bytes(path, data)
    common.atomic_write_json(base / "selection.json", {"selection_sha256": sha, **report})
    log(f"select: {report['paired']} paired of {report['rows']['sdm_train']} SDM rows, unpaired {report['unpaired']}; "
        f"drops {report['select_drops']}; selected {len(selected)}")
    return 0


def read_selection() -> tuple[list[dict], str]:
    base = selection_dir()
    meta = json.loads((base / "selection.json").read_text())
    if _sha256(base / "selection.jsonl") != meta["selection_sha256"]:
        raise RuntimeError("selection.jsonl differs from selection.json")
    return _read_jsonl(base / "selection.jsonl"), meta["selection_sha256"]


def store_dir() -> Path:
    return root() / "store"


def dev_dir() -> Path:
    return root() / "dev"


def _units(base: Path, wanted: dict[str, list[str]], selection_sha256: str) -> list[dict]:
    """Decode units (one per Hub file with wanted rows) tied to the selection in <base>/decode.json."""
    from pool import common
    entries = {f["path"]: f for f in listing()["files"]}
    units = [{"path": p, "key": file_key(p), "ids": sorted(ids), "ids_sha256": _sha_json(sorted(ids))}
             for p, ids in sorted(wanted.items()) if ids]
    body = {"selection_sha256": selection_sha256,
            "units": [{k: u[k] for k in ("path", "key", "ids_sha256")} | {"rows": len(u["ids"])} for u in units]}
    plan = base / "decode.json"
    if plan.exists():
        if json.loads(plan.read_text()) != body:
            raise RuntimeError(f"{plan} belongs to another selection; refusing to mix stores (move it aside)")
    else:
        common.atomic_write_json(plan, body)
    for u in units:
        u["entry"] = entries[u["path"]]
        config = u["path"].split("/")[0]
        u["out_dir"] = str(base / "audio" / config / u["key"]) if base == store_dir() else str(base / "audio" / u["key"])
        u["records"] = str(base / "decode" / f"{u['key']}.jsonl")
    return units


def _unit_records(u: dict) -> list[dict] | None:
    path = Path(u["records"])
    if not path.exists():
        return None
    recs = _read_jsonl(path)
    if sorted(r["id"] for r in recs) != u["ids"]:
        raise RuntimeError(f"{path}: its ids are not the unit's")
    return recs


def _decode_task(u: dict) -> tuple[str, int, int, float]:
    t0 = time.monotonic()
    recs = _unit_records(u)
    if recs is None:
        recs = decode_file(local_file(u["entry"]), u["ids"], Path(u["out_dir"]))
        _write_jsonl(Path(u["records"]), recs)
    return u["path"], len(recs), sum(r["status"] != "ok" for r in recs), time.monotonic() - t0


def run_units(units: list[dict], workers: int, what: str) -> None:
    todo = [u for u in units if _unit_records(u) is None]
    log(f"{what}: {len(units)} units, {len(units) - len(todo)} done, {sum(len(u['ids']) for u in todo)} rows to decode")
    if not todo:
        return
    with _executor(min(workers, len(todo))) as pool:
        for path, n, bad, secs in pool.map(_decode_task, todo):
            log(f"{what} {path}: {n} rows, {bad} undecodable, {secs:.0f} s")


def store_units(selected: list[dict], selection_sha256: str) -> list[dict]:
    wanted: dict[str, list[str]] = collections.defaultdict(list)
    for s in selected:
        wanted[s["sdm_file"]].append(s["sdm_id"])
        wanted[s["ihm_file"]].append(s["ihm_audio_id"])
    return _units(store_dir(), wanted, selection_sha256)


def cmd_decode(workers: int) -> int:
    selected, sha = read_selection()
    run_units(store_units(selected, sha), workers, "decode")
    return 0


def _verify_file(rec: dict) -> str | None:
    path = Path(rec["audio_filepath"])
    if not path.exists():
        return f"{path}: missing"
    if _sha256(path) != rec["sha256"]:
        return f"{path}: SHA-256 differs from its record"
    return None


def informational_rejects(clips: Sequence[dict]) -> dict:
    """How many store pairs lose their label at read time in stage 3's rules (repetition v1's cached rejections of
    the label run, the TED-LIUM read-time reject list): counted, not applied (the trainer applies them)."""
    from labeling import layout
    ids = {c["ihm_id"] for c in clips}
    out = {}
    cache = layout.relabel_root() / "cleaning" / LABELS_RUN / "repetition-v1.json"
    if cache.exists():
        body = json.loads(cache.read_text())
        rejected = {r[0] for s in body.get("shards", {}).values() for r in s.get("rejected", [])}
        out["repetition_v1_rejected"] = len(ids & rejected)
    reject = layout.relabel_root() / "pool" / "ted-reject" / "rejected_ids.txt"
    if reject.exists():
        out["ted_reject_listed"] = len(ids & set(reject.read_text().split()))
    return out


def cmd_finalize(workers: int) -> int:
    from pool import common
    from training import ami_store
    selected, sha = read_selection()
    units = store_units(selected, sha)
    recs: dict[str, dict] = {}
    for u in units:
        got = _unit_records(u)
        if got is None:
            raise RuntimeError(f"unit {u['path']} is not decoded yet (./python ami_sdm.py decode)")
        recs.update({r["id"]: r for r in got})
    sdm_rec = {s["sdm_id"]: recs[s["sdm_id"]] for s in selected}
    ihm_rec = {s["ihm_audio_id"]: recs[s["ihm_audio_id"]] for s in selected}
    clips, report = assemble(selected, sdm_rec, ihm_rec, c1_index())
    require_c1_match(report)
    files = [r for c in clips for r in (sdm_rec[c["id"]], recs[c["ihm_id"].removeprefix(ami_store.IHM_PREFIX)])]
    with cf.ThreadPoolExecutor(max(1, workers)) as pool:
        bad = [e for e in pool.map(_verify_file, files, chunksize=256) if e]
    if bad:
        raise RuntimeError(f"{len(bad)} FLAC files differ from their records (first: {bad[:3]})")
    selection = json.loads((selection_dir() / "selection.json").read_text())
    meta = {"source": {"repo": REPO, "revision": REVISION, "license": LICENSE, "student": "sdm train",
                       "teacher": "ihm train"},
            "selection_sha256": sha, "decode_plan_sha256": _sha256(store_dir() / "decode.json"),
            "labels_run": LABELS_RUN, "through_batch": THROUGH_BATCH,
            "frames": "training/ami_store.py encoder_frames (train_b0prime.encoder_frames), equal on both sides",
            "counts": {"selected": len(selected), **report["drops"], "clips": len(clips)},
            "c1_check": {k: v for k, v in report["c1"].items() if k != "mismatched"} | {"mismatched": 0}}
    out = ami_store.write_store(store_dir(), clips, meta)
    store = ami_store.PairStore.load(store_dir())
    rejects = informational_rejects(store.clips)
    common.atomic_write_json(root() / "REPORT.json", {
        "created": common.now(), "store": str(store_dir()), "clips": out["clips"], "sdm_hours": out["hours"],
        "ihm_hours": out["ihm_hours"], "clips_sha256": out["clips_sha256"],
        "rows": selection["rows"], "paired": selection["paired"], "unpaired": selection["unpaired"],
        "select_drops": selection["select_drops"], "selected": len(selected), "assemble_drops": report["drops"],
        "undecodable_reasons": report["undecodable_reasons"], "frames_differ_examples": report["frames_differ"][:10],
        "c1_check": out["c1_check"], "read_time_rejects_informational": rejects,
        "labels": selection["labels"], "held_out": selection["held_out"],
        "no_label_record_by_ihm_duration": selection["no_label_record_by_ihm_duration"]})
    log(f"store: {out['clips']} pairs, {out['hours']:.2f} h SDM ({out['ihm_hours']:.2f} h IHM); drops "
        f"{report['drops']}; C1 {out['c1_check']}; read-time rejects (informational) {rejects}")
    return 0


def cmd_dev(workers: int) -> int:
    from pool import common
    rows = read_index("sdm", "validation")
    _check_index_ids(rows, "sdm validation")
    dev_path, _ = manifests()
    meetings = held_out_meetings(*manifests())["dev"]
    wanted: dict[str, list[str]] = collections.defaultdict(list)
    keep = set(dev_wanted(rows, meetings))
    for r in rows:
        if r["audio_id"] in keep:
            wanted[r["file"]].append(r["audio_id"])
    units = _units(dev_dir(), wanted, _sha_json(sorted(keep)))
    run_units(units, workers, "dev decode")
    recs = {r["id"]: r for u in units for r in _unit_records(u)}
    out, counts = dev_rows(rows, recs, meetings)
    path = dev_dir() / f"{DEV_NAME}.jsonl"
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out).encode()
    common.atomic_write_bytes(path, data)
    sdm_meetings = {r["meeting_id"] for r in rows}
    common.atomic_write_json(dev_dir() / f"{DEV_NAME}.meta.json", {
        "name": DEV_NAME, "path": str(path), "utterances": len(out), "hours": sum(r["duration"] for r in out) / 3600,
        "source": "ami-sdm", "repo": REPO, "config": "sdm", "split": "validation", "revision": REVISION,
        "license": LICENSE, "text_field": "text", "text_kind": "human transcript (uppercase, no punctuation)",
        "kind": "dev", "informational": True,
        "selection": f"the SDM validation rows of the {DEV_MANIFEST} manifest's {len(meetings)} meetings with a "
                     "nonempty transcript that decode (decodable, nonempty, finite, not all zeros)",
        "counts": counts, "dev_meetings_without_sdm_rows": sorted(meetings - sdm_meetings),
        "dev_manifest_sha256": _sha256(dev_path), "manifest_sha256": _sha256_bytes(data),
        "audio": "16 kHz mono PCM_16 FLAC (decoded by the pool's path, parakeet-ternary data.convert)"})
    log(f"dev: {len(out)} utterances, {sum(r['duration'] for r in out) / 3600:.2f} h; {counts}")
    return 0


def cmd_draws() -> int:
    from pool import common
    from training import ami_store
    c1 = c1_index()
    rule = draws_per_clip(v["occurrences"] for v in c1.values())
    ihm_rows = read_index("ihm", "train")
    selected, _ = read_selection()
    alt = {"occurrences_over_all_ihm_train_rows": rule["occurrences"] / len(ihm_rows),
           "mean_over_selected_ihm_ids": sum(c1[s["ihm_id"]]["occurrences"] for s in selected if s["ihm_id"] in c1)
                                         / max(1, len(selected))}
    body = {"rule": "each SDM clip is drawn as many times as the average AMI IHM train id occurs in pools 0-781 "
                    "(rounded): C1's AMI occurrences over consumed batches [0, 100000) / the distinct AMI ids there, "
                    "rounded half up", "through_batch": THROUGH_BATCH, **rule, "alternatives_informational": alt}
    if (store_dir() / "clips.meta.json").exists():
        store = ami_store.PairStore.load(store_dir(), verify_audio=False)
        body.update(store_clips=len(store.clips), store_sdm_hours=store.meta["hours"],
                    sdm_hours_drawn=rule["draws"] * store.meta["hours"], store_clips_sha256=store.identity)
    common.atomic_write_json(root() / "draws.json", body)
    log(f"draws: {rule['occurrences']} occurrences / {rule['ids']} ids = {rule['mean']:.4f} -> {rule['draws']}"
        + (f"; {body['sdm_hours_drawn']:.1f} h of SDM audio drawn" if "sdm_hours_drawn" in body else ""))
    return 0


def cmd_status() -> int:
    if not (fetch_dir() / "files.json").exists():
        print("download: not listed")
        return 0
    files = listing()["files"]
    print(f"download: {sum(done_path(f).exists() for f in files)}/{len(files)} files checked")
    print(f"index: {sum(index_path(f).exists() for f in files)}/{len(files)} files; "
          f"c1 {'done' if (root() / 'index' / 'c1-ami.json').exists() else 'not done'}; "
          f"labels {'done' if (root() / 'index' / 'labels-ami.json').exists() else 'not done'}")
    for name, path in (("selection", selection_dir() / "selection.json"), ("store", store_dir() / "clips.meta.json"),
                       ("dev", dev_dir() / f"{DEV_NAME}.meta.json"), ("draws", root() / "draws.json")):
        print(f"{name}: {'done' if path.exists() else 'not done'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download")
    p.add_argument("--threads", type=int, default=DOWNLOAD_THREADS)
    for name in ("index", "decode", "finalize", "dev", "all"):
        sub.add_parser(name).add_argument("--workers", type=int, default=WORKERS)
    for name in ("select", "draws", "status"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    t0 = time.monotonic()
    if args.command == "download":
        rc = cmd_download(args.threads)
    elif args.command == "index":
        rc = cmd_index(args.workers)
    elif args.command == "select":
        rc = cmd_select()
    elif args.command == "decode":
        rc = cmd_decode(args.workers)
    elif args.command == "finalize":
        rc = cmd_finalize(args.workers)
    elif args.command == "dev":
        rc = cmd_dev(args.workers)
    elif args.command == "draws":
        rc = cmd_draws()
    elif args.command == "all":
        for step in (lambda: cmd_download(), lambda: cmd_index(args.workers), cmd_select,
                     lambda: cmd_decode(args.workers), lambda: cmd_finalize(args.workers),
                     lambda: cmd_dev(args.workers), cmd_draws):
            rc = step()
            if rc:
                return rc
    else:
        rc = cmd_status()
    if args.command != "status":
        log(f"{args.command} done in {time.monotonic() - t0:.0f} s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
