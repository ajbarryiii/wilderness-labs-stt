"""M1's historical test workloads for the recorded x2 benchmark (plans/x2-default.md "Recorded GPU benchmark";
plans/gpu-kernel.md A13). They are timed only after the freeze and after the benchmark's dev results are
written, for comparability with M1's recorded numbers; nothing is selected or tuned on them.

Rules, exactly M1's (applied to M1's test manifests, paths.MANIFESTS/test_<set>.jsonl):
- clips: parakeet-ternary/inference/compare_v2.py (default durations 3, 10, 30): for each duration the
  first record of test_librispeech_clean.jsonl minimising |duration - target|;
- 24 utterances: parakeet-ternary/inference/cuda_experiments.py sample_records(): for each test set in
  paths.TEST_SETS order and each target 3, 10, 30 s, the first unused record with duration > 0 minimising
  |duration - target|.
The rules are restated here (importing cuda_experiments would build its CUDA extensions); the selection
must reproduce the ids M1 recorded (RECORDED_CLIPS: comparison/cuda-iteration-20261007; RECORDED_24:
kernel-runs/20261007T141539-cuda-pipeline/pipeline.json), else it is refused.
Records carry the fields the bench protocols read (id, corpus, duration, audio_filepath, target_seconds,
samples, M) computed as for the dev workloads (shared/workloads.py).
"""
from __future__ import annotations

import json

TARGETS = (3, 10, 30)
RECORDED_CLIPS = ("3575-170457-0052", "1320-122617-0010", "121-123859-0002")
RECORDED_24 = (
    "3575-170457-0052", "1320-122617-0010", "121-123859-0002",
    "3528-168669-0026", "3080-5032-0002", "5442-41168-0009",
    "AMI_EN2002a_H01_FEO070_0170294_0170594", "AMI_ES2004c_H02_MEE014_0045596_0046596",
    "AMI_IS1009b_H03_FIO089_0023485_0026102",
    "4479741/143.wav", "4483338/125.wav", "4450488/235.wav",
    "POD1000000021_S0000276", "POD1000000040_S0000654", "POD1000000005_S0000339",
    "403903eb119bfc1cec44d86ec907077a/123.wav", "002d12258ff802d65f79ae2eef99e4ab/75.wav",
    "044da592acc01ff7baacb3dbf2a302a9/54.wav",
    "20110706-0900-PLENARY-10-en_20110706-17:24:55_2", "20181022-0900-PLENARY-en_20181022-19:42:13_4",
    "20140716-0900-PLENARY-7-en_20140716-10:22:18_15",
    "02db6434275133df0c23a4adcd81f00a643887ff9355b49cab7da96b8bd3dfa0b84fc855c39d395a0f7f9c8a4a33b4ee10e0675a882b91201d606457ae7ae728#1",
    "08979b185cf9503b193715c289628acd417c344cb97033f9452bc82f2d5588792865e036093e25064532451f9ee57069ef591a9dd70f5fba4bd0ec0fad856124",
    "38acaf2488bf254930421515622c046d7d70aa64eee6f24ee402b20a6142af4a9b3ecaab517a05adeffa4999a1a01a750cb0b087144c9a7439df381ce926795c",
)


def _manifest(source: str) -> list[dict]:
    import paths
    return [json.loads(line) for line in (paths.MANIFESTS / f"test_{source}.jsonl").read_text().splitlines()]


def select() -> dict:
    """{"clips": [...], "utterances": [...]} by M1's rules; refused unless they are M1's recorded ids."""
    import paths
    clean = _manifest("librispeech_clean")
    clips = [dict(min(clean, key=lambda r: abs(r["duration"] - t)), corpus="librispeech_clean", target_seconds=t)
             for t in TARGETS]
    utterances = []
    for source in paths.TEST_SETS:
        candidates, used = _manifest(source), set()
        for t in TARGETS:
            rec = min((r for r in candidates if r["id"] not in used and r["duration"] > 0),
                      key=lambda r: abs(r["duration"] - t))
            used.add(rec["id"])
            utterances.append(dict(rec, corpus=source, target_seconds=t))
    if tuple(r["id"] for r in clips) != RECORDED_CLIPS or tuple(r["id"] for r in utterances) != RECORDED_24:
        raise ValueError("M1's rules no longer reproduce M1's recorded historical workloads")
    return {"clips": clips, "utterances": utterances}


def records(dev_run: str) -> dict:
    """The historical workloads with frame counts, for the bench protocols; SystemExit unless the x2
    benchmark run `dev_run` has written its dev results (A13)."""
    import soundfile as sf
    from gpu1bit.shared import workloads
    from gpu1bit.x2 import bench_session
    bench_session.require_dev_results(dev_run)
    out = {}
    for key, recs in select().items():
        rows = []
        for i, r in enumerate(recs):
            info = sf.info(r["audio_filepath"])
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError(f"{r['id']}: {info.samplerate} Hz, {info.channels} channels")
            rows.append({**workloads._slim(r, target_seconds=r["target_seconds"]), "index": i,
                         **workloads._frames(int(info.frames))})
        out[key] = rows
    return out
