"""The TRAIN noise / RIR pool of the augmentation, and its disjointness check (synthetic plan "Augmentation").

Sources (aliases; downloads verified by fetch.py, extracted with finetune/noisy-dev's safe extractor):
- syn-noise-a: a sound-event collection's DEVELOPMENT split: clips whose per-clip license is CC0 or CC BY (any
  version) and whose labels include none of EXCLUDED_LABELS (speech, voice, singing, music, crowd and chatter, so
  that no intelligible speech is added to a clip whose label stays unchanged), at least MIN_NOISE_SECONDS long;
- syn-noise-b: one subset of a noise/music/speech corpus whose license file names each file's license, title and
  recorder; only files with a single license kind, "Attribution" or "Public Domain", and whose title suggests no
  voices (VOICE_TITLE_WORDS);
- syn-rir-a: simulated room impulse responses (image method; no recorded rooms).

The noisy dev set's evaluation pool is reserved (finetune/noisy-dev, plans/build.md "Scope of the TRAIN check"):
its blocklist ($DATA_ROOT/noisy-dev/build/eval_pool_blocklist.json) lists reserved datasets / subsets / rooms,
excluded families, every reserved sound id and uploader, and the decoded-PCM SHA-256 of every reserved file.
check_train_pool(pool, blocklist) is this plan's TRAIN-pool check: it FAILS (pass false, every violation listed)
if any pool file's dataset matches a reserved unit (dataset, and subset or room when the unit names one) or an
excluded family, or its sound id or uploader is reserved, or its decoded-PCM SHA-256 (of the original file,
noisy_sources.pcm_sha256's definition) is reserved. Selection drops reserved ids and uploaders up front; a file
whose PCM hash is reserved is dropped and recorded; the written pool must then pass the check, and
augment.Pool.load refuses a pool that did not.

Prepared noise: originals decoded, mixed to mono, resampled to 16 kHz (scipy resample_poly), written as FLAC
to $RELABEL_ROOT/synthetic/noise/audio/<alias>/<key>.flac. RIRs are used as extracted (16 kHz mono).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np

from synthetic import fetch, layout

EXCLUDED_LABELS = frozenset({
    "Human_voice", "Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking",
    "Child_speech_and_kid_speaking", "Conversation", "Speech_synthesizer", "Singing", "Male_singing",
    "Female_singing", "Choir", "Music", "Musical_instrument", "Crowd", "Chatter", "Cheering", "Laughter", "Giggle",
    "Chuckle_and_chortle", "Shout", "Yell", "Screaming", "Whispering", "Human_group_actions"})
LICENSES_A = ("http://creativecommons.org/publicdomain/zero/1.0/", "http://creativecommons.org/licenses/by/")
MIN_NOISE_SECONDS = 1.0
BLOCKLIST = ("noisy-dev", "build", "eval_pool_blocklist.json")      # under $DATA_ROOT
# Archive layouts and the blocklist's dataset / subset names of each source come from the private registry
# ("layout" and "blocklist_unit" of the alias), so that no source is named here.


def unit(alias: str) -> tuple[str, str | None]:
    u = layout.source(alias)["blocklist_unit"]
    return u["dataset"], u.get("subset")


def src_layout(alias: str) -> dict:
    return layout.source(alias)["layout"]


def noisy_dev_on_path() -> None:
    here = str(Path(__file__).resolve().parents[2] / "noisy-dev")
    if here not in sys.path:
        sys.path.append(here)


def pcm_sha256(path: Path) -> str:
    """SHA-256 of b"<rate>,<channels>\\n" + the decoded samples as little-endian float32, interleaved (the noisy dev
    set's noisy_sources.pcm_sha256 definition, which its blocklist uses)."""
    import soundfile as sf
    digest = hashlib.sha256()
    with sf.SoundFile(str(path)) as f:
        digest.update(f"{f.samplerate},{f.channels}\n".encode())
        while True:
            block = f.read(1 << 18, dtype="float32", always_2d=True)
            if not len(block):
                break
            digest.update(np.ascontiguousarray(block, dtype="<f4").tobytes())
    return digest.hexdigest()


def load_blocklist() -> tuple[dict, str]:
    p = layout.root("DATA_ROOT").joinpath(*BLOCKLIST)
    data = p.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def check_train_pool(files: list[dict], blocklist: dict) -> dict:
    """{pass, checked, violations[]} for pool files {key, dataset, subset, freesound_id, uploader, pcm_sha256}."""
    ids = {str(i) for i in blocklist.get("freesound_ids", [])}
    uploaders = set(blocklist.get("freesound_uploaders", []))
    hashes = set(blocklist.get("pcm_sha256", []))
    families = set(blocklist.get("excluded_families", {}))
    units = blocklist.get("units", [])
    violations = []
    for f in files:
        reasons = []
        ds = f["dataset"]
        for u in units:
            if u.get("dataset") != ds:
                continue
            if "subset" in u and u["subset"] != f.get("subset"):
                continue
            if "room" in u and u["room"] != f.get("room"):
                continue
            reasons.append(f"reserved unit {u}")
        if ds in families:
            reasons.append(f"excluded family {ds}")
        if f.get("freesound_id") is not None and str(f["freesound_id"]) in ids:
            reasons.append("reserved sound id")
        if f.get("uploader") is not None and f["uploader"] in uploaders:
            reasons.append("reserved uploader")
        if f.get("pcm_sha256") in hashes:
            reasons.append("reserved PCM hash")
        if not f.get("pcm_sha256"):
            reasons.append("no PCM hash")
        if reasons:
            violations.append({"key": f["key"], "reasons": reasons})
    return {"pass": not violations, "checked": len(files), "violations": violations[:200],
            "violation_count": len(violations)}


def _resample16(x: np.ndarray, rate: int) -> np.ndarray:
    from scipy.signal import resample_poly
    if rate == layout.SAMPLE_RATE:
        return x.astype(np.float32)
    f = Fraction(layout.SAMPLE_RATE, rate)
    return resample_poly(x.astype(np.float64), f.numerator, f.denominator).astype(np.float32)


def prepare(original: Path, out: Path) -> float:
    """Decode, mono, 16 kHz, FLAC; returns seconds."""
    import soundfile as sf
    a, rate = sf.read(str(original), dtype="float32", always_2d=True)
    y = _resample16(a.mean(axis=1), rate)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    sf.write(str(tmp), np.clip(y, -1, 1), layout.SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    os.replace(tmp, out)
    return len(y) / layout.SAMPLE_RATE


def _files(alias: str) -> dict[str, Path]:
    return {f["name"]: Path(f["path"]) for f in fetch.record(alias)["files"]}


def select_a(blocklist: dict) -> tuple[dict[str, dict], dict]:
    files, lay = _files("syn-noise-a"), src_layout("syn-noise-a")
    meta = json.loads(zipfile.ZipFile(files[lay["metadata_zip"]]).read(lay["metadata_member"]))
    gt = zipfile.ZipFile(files[lay["labels_zip"]]).read(lay["labels_member"]).decode()
    labels = {r["fname"]: set(r["labels"].split(",")) for r in csv.DictReader(io.StringIO(gt))}
    ids, ups = {str(i) for i in blocklist.get("freesound_ids", [])}, set(blocklist.get("freesound_uploaders", []))
    chosen, why = {}, {"license": 0, "labels": 0, "reserved_id": 0, "reserved_uploader": 0, "no_labels": 0}
    for fid, info in sorted(meta.items(), key=lambda kv: int(kv[0])):
        if not info["license"].startswith(LICENSES_A):
            why["license"] += 1
        elif fid not in labels:
            why["no_labels"] += 1
        elif labels[fid] & EXCLUDED_LABELS:
            why["labels"] += 1
        elif fid in ids:
            why["reserved_id"] += 1
        elif info["uploader"] in ups:
            why["reserved_uploader"] += 1
        else:
            chosen[fid] = {"license": info["license"], "uploader": info["uploader"], "labels": sorted(labels[fid])}
    return chosen, {"candidates": len(meta), "selected": len(chosen), "dropped": why}


VOICE_TITLE_WORDS = ("crowd", "church", "city", "skype", "police", "talk", "voice", "speech", "people", "laugh",
                     "audience", "sing", "song", "music", "piano")


def read_license_blocks(path: Path, id_prefix: str) -> dict[str, dict]:
    """The subset's per-file license file: blocks separated by lines of '=' characters, each with the file id, a
    'Title:' line, one or more 'License:' lines, a 'Recorded by' line and the source URL. Returns
    {file id: {title, license ("attribution" | "public domain" | None), licenses (as written), recorded_by, url}}."""
    out = {}
    block: list[str] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() + ["===="]
    for line in lines:
        if line.strip() and set(line.strip()) == {"="}:
            ids = [b.strip() for b in block if b.strip().startswith(id_prefix)]
            if len(ids) == 1:
                lic = [b.split(":", 1)[1].strip() for b in block if b.startswith("License:")]
                kinds = {"public domain" if x.lower().startswith("public domain") else
                         "attribution" if x.lower().startswith("attribution") else None for x in lic}
                title = next((b.split(":", 1)[1].strip() for b in block if b.startswith("Title:")), "")
                out[ids[0]] = {"title": title, "licenses": lic,
                               "license": kinds.pop() if len(kinds) == 1 else None,
                               "recorded_by": next((b[len("Recorded by"):].strip() for b in block
                                                    if b.startswith("Recorded by")), None),
                               "url": next((b.strip() for b in block if b.strip().startswith("http")), None)}
            block = []
        else:
            block.append(line)
    return out


def voice_title(title: str) -> bool:
    t = title.lower()
    return any(w in t for w in VOICE_TITLE_WORDS)


def extract_all(chosen_a: set[str], log=print) -> dict[str, Path]:
    """Extract the selected clips of syn-noise-a (by clip id = file stem), the subset of syn-noise-b and the RIRs."""
    noisy_dev_on_path()
    import noisy_extract as nx
    out = {}
    a = _files("syn-noise-a")
    dest_a = layout.source_dir("syn-noise-a") / "extracted" / "dev_audio"
    if not dest_a.exists():
        segs = [a[n] for n in src_layout("syn-noise-a")["audio_segments"]]
        log("extracting syn-noise-a dev audio")
        nx.extract_zip(segs, dest_a, lambda n: n.endswith(".wav") and Path(n).stem in chosen_a,
                       nx.Budget(max_members=45_000, max_member_bytes=256 << 20, max_total_bytes=40 << 30,
                                 max_metadata_bytes=16 << 20))
    out["syn-noise-a"] = dest_a
    b = _files("syn-noise-b")
    dest_b = layout.source_dir("syn-noise-b") / "extracted" / "subset"
    if not dest_b.exists():
        log("extracting syn-noise-b subset")
        lb = src_layout("syn-noise-b")
        nx.extract_tar(b[lb["archive"]], dest_b, lambda n: n.startswith(lb["member_prefix"]),
                       nx.Budget(max_members=5_000, max_member_bytes=1 << 30, max_total_bytes=24 << 30))
    out["syn-noise-b"] = dest_b
    r = _files("syn-rir-a")
    dest_r = layout.source_dir("syn-rir-a") / "extracted" / "sim_rir_16k"
    if not dest_r.exists():
        log("extracting syn-rir-a")
        nx.extract_zip([r[src_layout("syn-rir-a")["archive"]]], dest_r, lambda n: n.endswith((".wav", "rir_list", "room_info")),
                       nx.Budget(max_members=70_000, max_member_bytes=16 << 20, max_total_bytes=8 << 30,
                                 max_metadata_bytes=16 << 20))
    out["syn-rir-a"] = dest_r
    return out


def build(workers: int = 6, log=print) -> dict:
    import concurrent.futures as cf
    import soundfile as sf
    layout.require_mount()
    blocklist, bl_sha = load_blocklist()
    chosen_a, sel_a = select_a(blocklist)
    roots = extract_all(set(chosen_a), log)
    noise, rirs, dropped = [], [], []
    wavs = {p.stem: p for p in roots["syn-noise-a"].rglob("*.wav")}
    jobs = []
    for fid, info in chosen_a.items():
        if fid not in wavs:
            raise RuntimeError(f"syn-noise-a clip {fid} missing from the extracted audio")
        ds, sub = unit("syn-noise-a")
        jobs.append({"key": f"syn-noise-a:{fid}", "source": "syn-noise-a", "dataset": ds, "subset": sub,
                     "original": str(wavs[fid]), "license": info["license"], "uploader": info["uploader"],
                     "freesound_id": int(fid), "labels": info["labels"]})
    lb = src_layout("syn-noise-b")
    lic_files = list(roots["syn-noise-b"].rglob(lb["license_file"]))
    if len(lic_files) != 1:
        raise RuntimeError(f"syn-noise-b: expected one {lb['license_file']} in the extracted subset")
    sb_root = lic_files[0].parent
    ann = read_license_blocks(lic_files[0], lb["id_prefix"])
    for wav in sorted(sb_root.glob("*.wav")):
        a = ann.get(wav.stem)
        if not a or a["license"] is None:
            dropped.append({"key": f"syn-noise-b:{wav.stem}", "reason": "no single permissive license entry"})
            continue
        if voice_title(a["title"]):
            dropped.append({"key": f"syn-noise-b:{wav.stem}", "reason": f"title suggests voices: {a['title']}"})
            continue
        ds, sub = unit("syn-noise-b")
        # The blocklist's uploaders are accounts of another platform; "recorded_by" is kept, not compared.
        jobs.append({"key": f"syn-noise-b:{wav.stem}", "source": "syn-noise-b", "dataset": ds, "subset": sub,
                     "original": str(wav), "license": a["license"], "uploader": None, "freesound_id": None,
                     "recorded_by": a["recorded_by"], "url": a["url"], "title": a["title"]})

    def work(j):
        info = sf.info(j["original"])
        if info.duration < MIN_NOISE_SECONDS:
            return j, None, "shorter than MIN_NOISE_SECONDS"
        j["pcm_sha256"] = pcm_sha256(Path(j["original"]))
        out = layout.noise_dir() / "audio" / j["source"] / f"{j['key'].split(':', 1)[1]}.flac"
        if not out.exists():
            j["seconds"] = prepare(Path(j["original"]), out)
        else:
            j["seconds"] = sf.info(str(out)).duration
        j["path"] = str(out)
        return j, out, None

    reserved = set(blocklist.get("pcm_sha256", []))
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for j, out, why in ex.map(work, jobs):
            if why:
                dropped.append({"key": j["key"], "reason": why})
            elif j["pcm_sha256"] in reserved:
                dropped.append({"key": j["key"], "reason": "reserved PCM hash"})
            else:
                noise.append(j)
    log(f"noise: {len(noise)} clips, {sum(n['seconds'] for n in noise) / 3600:.1f} h")
    rir_wavs = sorted(roots["syn-rir-a"].rglob("*.wav"))
    ds, sub = unit("syn-rir-a")

    def rwork(p):
        return {"key": f"syn-rir-a:{p.relative_to(roots['syn-rir-a']).as_posix()}", "source": "syn-rir-a",
                "dataset": ds, "subset": sub, "path": str(p), "room": p.parent.name, "pcm_sha256": pcm_sha256(p)}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(rwork, rir_wavs):
            (dropped.append({"key": r["key"], "reason": "reserved PCM hash"}) if r["pcm_sha256"] in reserved
             else rirs.append(r))
    noise.sort(key=lambda n: n["key"])
    rirs.sort(key=lambda r: r["key"])
    check = check_train_pool(noise + rirs, blocklist)
    pool = {"noise": noise, "rirs": rirs, "dropped": dropped, "selection": {"syn-noise-a": sel_a},
            "blocklist_sha256": bl_sha, "train_check": check,
            "excluded_labels": sorted(EXCLUDED_LABELS), "min_noise_seconds": MIN_NOISE_SECONDS,
            "hours": sum(n["seconds"] for n in noise) / 3600}
    path = layout.noise_dir() / "pool.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(pool, indent=0) + "\n")
    os.replace(tmp, path)
    log(f"pool: {len(noise)} noise clips, {len(rirs)} RIRs, TRAIN check pass={check['pass']}")
    return {k: pool[k] for k in ("hours", "train_check", "selection")} | {"noise": len(noise), "rirs": len(rirs)}
