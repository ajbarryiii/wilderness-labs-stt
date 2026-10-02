"""Benchmark clips: selection from LibriSpeech dev-clean, committed manifest clips.json, PCM materialization.

See DESIGN.md "Surrogate models, clips and traces" (Clips) and README.md (WP2). Runs on NixOS and on the
Mac (numpy + soundfile only, no repository-local imports).

Source: OpenSLR SLR12 dev-clean.tar.gz, read directly from the archive (no extracted tree). The archive is
checked against OpenSLR's published MD5 (https://www.openslr.org/resources/12/md5sum.txt) and the SHA-256
recorded in clips.json before anything is read.

PCM: 16 kHz mono float32 little-endian, value = int16 sample / 32768 (exact; FLAC is lossless, decoded as
int16 by libsndfile), written raw as <id>.f32. Every clip's SHA-256 is over exactly those bytes.

Front-end arithmetic (reference.py, NeMo FilterbankFeatures with n_fft 512, hop 160, center=True,
exact_pad False, pad_to 0; dw_striding subsampling with three kernel-3 stride-2 pad-1 convolutions):
  valid mel frames      M(N) = (N + 2 * (512 // 2) - 512) // 160 = N // 160
  allocated STFT frames for an input buffer of N_alloc samples = N_alloc // 160 + 1
  one stride-2 stage    L -> (L + 2 - 3) // 2 + 1 = ceil(L / 2), so encoder frames E(M) = ceil(M / 8)
  FluidAudio (C0)       actualAudioFrames = ceil(N / 1280) (ASRConstants.calculateEncoderFrames), used as
                        min(encoder_length, ceil(N / 1280)); it exceeds E(M(N)) by one when
                        N mod 1280 is in 1..159 (e.g. N = 32001: 26 vs 25).
So the encoder frame count steps between M = 8k and M = 8k + 1, i.e. between N = 1280k + 159 and
N = 1280k + 160.

Buckets: valid duration (0, 2], (2, 4], (4, 8], (8, 15] s, i.e. N in (0, 32000], (32000, 64000],
(64000, 128000], (128000, 240000] samples; allocated encoder frames 26, 51, 101, 188 for inputs of
32000, 64000, 128000, 240000 samples.

Selection (deterministic, seed SEED):
- natural: per bucket, dev-clean utterances whose full length lies in the bucket, sorted by id and
  permuted by numpy default_rng([SEED, bucket index]); the first 16 with distinct speakers are taken,
  then (if needed) the next ones in permutation order. If a bucket had fewer than 16 natural utterances,
  prefixes of longer utterances (length drawn uniformly in the bucket) would fill it, marked kind
  "prefix" without transcript; dev-clean has enough in every bucket (counts in clips.json).
- boundary: crops of length N (no transcript) for each bucket edge N_e = 16000 * b, b in 2, 4, 8, 15:
  sample edge N_e (last length inside the bucket) and N_e + 1 (first length of the next bucket; not for
  15 s, which has no next bucket), and stride lengths N = 160 M for M in {8k - 1, 8k, 8k + 1} with
  k = M_e // 8, M_e = N_e // 160 (2/4/8 s: M_e = 8k, so 8k + 1 lies in the next bucket; 15 s: M_e = 1500
  is not a multiple of 8, k = 187, all three lie inside and 8 * 188 + 1 would exceed 15 s). Duplicate
  lengths are merged (their tags combined). Each crop comes from an utterance at least N samples long,
  chosen by default_rng([SEED, 100 + clip index]) among those sorted by id, at an offset drawn uniformly
  in [0, len - N] by the same generator.
- synthetic: "silence" = 48000 zero samples (3 s); "impulse" = 48000 samples, all zero except sample
  24000 = 0.5.

  python clips.py select [--archive PATH]                # NixOS: (re)writes clips.json next to this file
  python clips.py materialize --out DIR [--archive PATH]  # any machine: DIR/<id>.f32, hashes verified
  python clips.py verify-archive [--archive PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "clips.json"
SEED = 20261002
SAMPLE_RATE = 16000
HOP = 160
SUBSAMPLING = 8
FLUIDAUDIO_SAMPLES_PER_FRAME = 1280
BUCKETS = (2, 4, 8, 15)  # seconds; bucket b holds N in (prev edge, 16000 b]
PER_BUCKET = 16
ARCHIVE_NAME = "dev-clean.tar.gz"
ARCHIVE_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
ARCHIVE_MD5_PUBLISHED = "42e2234ba48799c1f50f24a7926300a1"  # https://www.openslr.org/resources/12/md5sum.txt
ARCHIVE_MD5_SOURCE = "https://www.openslr.org/resources/12/md5sum.txt"
DEFAULT_ARCHIVES = (Path("/mnt/hd/wilderness-labs-stt/whisper-ternary/data/dev-clean.tar.gz"),
                    Path("/Users/ajbarry/wilderness-labs-stt-artifacts/parakeet-ios/data/dev-clean.tar.gz"))
SYNTH_SAMPLES = 48000
IMPULSE_AT = 24000
IMPULSE_VALUE = 0.5


# --- arithmetic --------------------------------------------------------------------------------------------

def mel_frames(n: int) -> int:
    return n // HOP


def encoder_frames(n: int) -> int:
    return -(-mel_frames(n) // SUBSAMPLING)


def fluidaudio_frames(n: int) -> int:
    return -(-n // FLUIDAUDIO_SAMPLES_PER_FRAME)


def bucket_of(n: int) -> int:
    for b in BUCKETS:
        if n <= b * SAMPLE_RATE:
            return b
    raise ValueError(f"{n} samples exceed the largest bucket")


def bucket_table() -> list[dict]:
    out, lo = [], 0
    for b in BUCKETS:
        hi = b * SAMPLE_RATE
        alloc_mel = hi // HOP + 1
        out.append({"bucket": b, "min_samples_exclusive": lo, "max_samples": hi, "allocated_mel_frames": alloc_mel,
                    "allocated_encoder_frames": -(-alloc_mel // SUBSAMPLING), "max_valid_encoder_frames": encoder_frames(hi)})
        lo = hi
    return out


# --- archive -----------------------------------------------------------------------------------------------

def find_archive(path: str | None) -> Path:
    if path:
        return Path(path)
    for candidate in DEFAULT_ARCHIVES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("no dev-clean.tar.gz found; pass --archive")


def archive_hashes(path: Path) -> dict:
    md5, sha = hashlib.md5(), hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 22):
            md5.update(chunk); sha.update(chunk); size += len(chunk)
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest(), "bytes": size}


def verify_archive(path: Path, expected: dict | None) -> dict:
    hashes = archive_hashes(path)
    if hashes["md5"] != ARCHIVE_MD5_PUBLISHED:
        raise ValueError(f"{path}: MD5 {hashes['md5']} != OpenSLR's {ARCHIVE_MD5_PUBLISHED}")
    if expected and (hashes["sha256"] != expected["sha256"] or hashes["bytes"] != expected["bytes"]):
        raise ValueError(f"{path}: SHA-256/size differ from clips.json")
    return hashes


def decode_flac(data: bytes) -> np.ndarray:
    """int16 FLAC bytes -> float32 PCM (int16 / 32768, exact)."""
    import soundfile as sf

    audio, rate = sf.read(io.BytesIO(data), dtype="int16", always_2d=True)
    if rate != SAMPLE_RATE or audio.shape[1] != 1:
        raise ValueError("expected 16 kHz mono")
    return (audio[:, 0].astype(np.float32) / np.float32(32768.0)).astype("<f4")


def scan_archive(path: Path, wanted: set[str] | None = None):
    """Yield (member name, bytes) of .flac and .trans.txt members (only `wanted` .flac members if given),
    streaming the gzip once."""
    with tarfile.open(path, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if name.endswith(".trans.txt") or (name.endswith(".flac") and (wanted is None or name in wanted)):
                yield name, tar.extractfile(member).read()


def utterance_table(path: Path) -> list[dict]:
    """Every dev-clean utterance: id, member, speaker, samples, transcript (one pass over the archive)."""
    import soundfile as sf

    utts: dict[str, dict] = {}
    texts: dict[str, str] = {}
    for name, data in scan_archive(path):
        if name.endswith(".trans.txt"):
            for line in data.decode().splitlines():
                if line.strip():
                    uid, text = line.split(" ", 1)
                    texts[uid] = text.strip()
        else:
            info = sf.info(io.BytesIO(data))
            if info.samplerate != SAMPLE_RATE or info.channels != 1:
                raise ValueError(f"{name}: not 16 kHz mono")
            uid = Path(name).name[:-len(".flac")]
            utts[uid] = {"utterance": uid, "member": name, "speaker": uid.split("-")[0], "samples": int(info.frames)}
    for uid, u in utts.items():
        u["transcript"] = texts[uid]
    return [utts[k] for k in sorted(utts)]


# --- selection ---------------------------------------------------------------------------------------------

def pcm_sha256(pcm: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pcm, dtype="<f4").tobytes()).hexdigest()


def clip_entry(cid: str, kind: str, n: int, member: str | None, utterance: str | None, offset: int,
               transcript: str | None, tags: list[str] | None = None) -> dict:
    return {"id": cid, "kind": kind, "bucket": bucket_of(n), "length": n, "seconds": round(n / SAMPLE_RATE, 5),
            "mel_frames": mel_frames(n), "encoder_frames": encoder_frames(n), "fluidaudio_frames": fluidaudio_frames(n),
            "source": member, "utterance": utterance, "offset": offset, "transcript": transcript,
            "tags": tags or [], "sha256": None}


def select(utts: list[dict]) -> tuple[list[dict], dict]:
    clips: list[dict] = []
    counts = {}
    lo = 0
    for bi, b in enumerate(BUCKETS):
        hi = b * SAMPLE_RATE
        candidates = [u for u in utts if lo < u["samples"] <= hi]
        counts[str(b)] = len(candidates)
        order = np.random.default_rng([SEED, bi]).permutation(len(candidates))
        chosen, speakers = [], set()
        for i in order:
            if len(chosen) < PER_BUCKET and candidates[i]["speaker"] not in speakers:
                chosen.append(int(i)); speakers.add(candidates[i]["speaker"])
        for i in order:
            if len(chosen) < PER_BUCKET and int(i) not in chosen:
                chosen.append(int(i))
        for i in sorted(chosen, key=lambda j: candidates[j]["utterance"]):
            u = candidates[i]
            clips.append(clip_entry(f"n{b:02d}-{u['utterance']}", "natural", u["samples"], u["member"], u["utterance"],
                                    0, u["transcript"]))
        missing = PER_BUCKET - len(chosen)
        if missing > 0:  # not reached for dev-clean (see counts); kept so the rule is complete
            longer = [u for u in utts if u["samples"] > hi]
            rng = np.random.default_rng([SEED, 50 + bi])
            for j in rng.permutation(len(longer))[:missing]:
                u, n = longer[j], int(rng.integers(lo + 1, hi + 1))
                clips.append(clip_entry(f"p{b:02d}-{u['utterance']}", "prefix", n, u["member"], u["utterance"], 0, None))
        lo = hi
    lengths: dict[int, list[str]] = {}
    for b in BUCKETS:
        n_e = b * SAMPLE_RATE
        m_e = n_e // HOP
        k = m_e // SUBSAMPLING
        lengths.setdefault(n_e, []).append(f"edge{b}s_inside")
        if b != BUCKETS[-1]:
            lengths.setdefault(n_e + 1, []).append(f"edge{b}s_outside")
        for m in (8 * k - 1, 8 * k, 8 * k + 1):
            if HOP * m <= BUCKETS[-1] * SAMPLE_RATE:
                lengths.setdefault(HOP * m, []).append(f"stride_M{m}")
    for ci, n in enumerate(sorted(lengths)):
        rng = np.random.default_rng([SEED, 100 + ci])
        pool = [u for u in utts if u["samples"] >= n]
        u = pool[int(rng.integers(len(pool)))]
        offset = int(rng.integers(0, u["samples"] - n + 1))
        clips.append(clip_entry(f"b{bucket_of(n):02d}-N{n}", "boundary", n, u["member"], u["utterance"], offset, None,
                                sorted(lengths[n])))
    clips.append(clip_entry("silence-3s", "silence", SYNTH_SAMPLES, None, None, 0, None))
    clips.append(clip_entry("impulse-3s", "impulse", SYNTH_SAMPLES, None, None, 0, None, [f"impulse_at_{IMPULSE_AT}"]))
    return clips, counts


def synthetic_pcm(clip: dict) -> np.ndarray:
    pcm = np.zeros(clip["length"], dtype="<f4")
    if clip["kind"] == "impulse":
        pcm[IMPULSE_AT] = IMPULSE_VALUE
    return pcm


def build_pcm(clips: list[dict], archive: Path):
    """Yield (clip, pcm) for every clip (archive read once)."""
    need: dict[str, list[dict]] = {}
    for c in clips:
        if c["source"]:
            need.setdefault(c["source"], []).append(c)
    for c in clips:
        if not c["source"]:
            yield c, synthetic_pcm(c)
    seen = set()
    for name, data in scan_archive(archive, set(need)):
        if name.endswith(".trans.txt"):
            continue
        audio = decode_flac(data)
        seen.add(name)
        for c in need[name]:
            pcm = audio[c["offset"]:c["offset"] + c["length"]]
            if len(pcm) != c["length"]:
                raise ValueError(f"{c['id']}: source too short")
            yield c, pcm
    if seen != set(need):
        raise ValueError(f"members missing from the archive: {sorted(set(need) - seen)}")


def cmd_select(args) -> None:
    archive = find_archive(args.archive)
    t0 = time.time()
    hashes = verify_archive(archive, None)
    utts = utterance_table(archive)
    clips, counts = select(utts)
    for c, pcm in build_pcm(clips, archive):
        c["sha256"] = pcm_sha256(pcm)
    manifest = {
        "schema": 1,
        "description": "Parakeet iOS benchmark clips (DESIGN.md 'Clips'); selection rules in clips.py docstring",
        "archive": {"name": ARCHIVE_NAME, "url": ARCHIVE_URL, "bytes": hashes["bytes"], "md5": hashes["md5"],
                    "md5_published": ARCHIVE_MD5_PUBLISHED, "md5_published_source": ARCHIVE_MD5_SOURCE,
                    "sha256": hashes["sha256"], "license": "CC BY 4.0 (OpenSLR SLR12, Panayotov et al. 2015)",
                    "utterances": len(utts)},
        "seed": SEED,
        "pcm": "16 kHz mono float32 little-endian raw (<id>.f32), value = int16 / 32768; sha256 over those bytes",
        "formulas": {
            "mel_frames": "M = N // 160 (valid frames; NeMo get_seq_len with n_fft 512, hop 160, exact_pad False)",
            "allocated_stft_frames": "N_alloc // 160 + 1 (center=True)",
            "encoder_frames": "E = ceil(M / 8) (three stride-2, kernel-3, pad-1 convolutions: L -> ceil(L / 2))",
            "fluidaudio_frames": "ceil(N / 1280) (FluidAudio 0.7.8 ASRConstants.calculateEncoderFrames)",
            "stride_step": "E steps between M = 8k and 8k + 1, i.e. N = 1280k + 159 -> 1280k + 160",
        },
        "buckets": bucket_table(),
        "natural_candidates_per_bucket": counts,
        "rules": {
            "natural": f"{PER_BUCKET} per bucket from utterances fully inside it; ids sorted, default_rng([seed, bucket index]) "
                       "permutation, distinct speakers first",
            "prefix": "only if a bucket has fewer natural candidates (not the case here)",
            "boundary": "N_e and N_e + 1 per edge (no N_e + 1 at 15 s) and N = 160 M, M in {8k - 1, 8k, 8k + 1}, "
                        "k = (N_e // 160) // 8; crop source and offset from default_rng([seed, 100 + index])",
            "synthetic": f"silence: {SYNTH_SAMPLES} zeros; impulse: {SYNTH_SAMPLES} zeros with sample {IMPULSE_AT} = {IMPULSE_VALUE}",
        },
        "clips": clips,
    }
    write_manifest(manifest)
    kinds = {}
    for c in clips:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(json.dumps({"clips": len(clips), "kinds": kinds, "candidates": counts, "archive": hashes,
                      "seconds": round(time.time() - t0, 1)}))


def write_manifest(manifest: dict) -> None:
    """One clip per line, so diffs stay readable."""
    head = {k: v for k, v in manifest.items() if k != "clips"}
    text = json.dumps(head, indent=1)[:-2] + ',\n "clips": [\n'
    text += ",\n".join("  " + json.dumps(c, separators=(", ", ": ")) for c in manifest["clips"])
    MANIFEST.write_text(text + "\n ]\n}\n")


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(Path(path).read_text())


def read_pcm(directory: Path, clip: dict) -> np.ndarray:
    return np.fromfile(Path(directory) / f"{clip['id']}.f32", dtype="<f4")


def cmd_materialize(args) -> None:
    manifest = load_manifest()
    archive = find_archive(args.archive)
    t0 = time.time()
    hashes = verify_archive(archive, manifest["archive"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mismatches, written = [], 0
    for c, pcm in build_pcm(manifest["clips"], archive):
        digest = pcm_sha256(pcm)
        if digest != c["sha256"]:
            mismatches.append(c["id"])
            continue
        np.ascontiguousarray(pcm, dtype="<f4").tofile(out / f"{c['id']}.f32")
        written += 1
    result = {"platform": sys.platform, "archive_md5": hashes["md5"], "archive_sha256": hashes["sha256"],
              "archive_md5_matches_openslr": hashes["md5"] == ARCHIVE_MD5_PUBLISHED, "clips": len(manifest["clips"]),
              "written": written, "sha256_mismatches": mismatches, "all_sha256_match": not mismatches,
              "out": str(out), "seconds": round(time.time() - t0, 1)}
    (out / "materialize.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))
    if mismatches:
        sys.exit(1)


def cmd_verify_archive(args) -> None:
    archive = find_archive(args.archive)
    print(json.dumps({"archive": str(archive), **verify_archive(archive, load_manifest()["archive"] if MANIFEST.exists() else None),
                      "md5_published": ARCHIVE_MD5_PUBLISHED}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select"); p.add_argument("--archive"); p.set_defaults(func=cmd_select)
    p = sub.add_parser("materialize"); p.add_argument("--out", required=True); p.add_argument("--archive")
    p.set_defaults(func=cmd_materialize)
    p = sub.add_parser("verify-archive"); p.add_argument("--archive"); p.set_defaults(func=cmd_verify_archive)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
