"""Measure the signal trigger against known inserted silence boundaries.

Uses training audio only. Recording start is not a manually annotated phonetic
speech onset, and these results do not measure speech/noise discrimination.
"""

import collections
import json

import numpy as np
import soundfile as sf
import torch

from common import ART, digest, read_manifest, save, storage
from onset_model import startup_mask
from prepare import feature


def main():
    storage()
    torch.set_num_threads(2)
    run = ART / "training-repair/onset-fp-h100-20260910"
    manifest = read_manifest()["rows"]
    gate = set(json.loads((run / "subsets.json").read_text())["gate"])
    groups = collections.defaultdict(list)
    for row in manifest:
        if row["split"] == "train" and row["domain"] == "digits" and row["id"] not in gate:
            groups[row["speaker"]].append(row)
    rows = [r for r in manifest if r["id"] in gate and r["domain"] == "digits"]
    rows += [r for rs in groups.values() for r in sorted(rs, key=lambda r: r["id"])[:8]]
    results = []
    for row in rows:
        assert digest(row["audio"]) == row["audio_sha256"]
        audio, sr = sf.read(row["audio"], dtype="float32")
        assert sr == 16000 and not np.any(audio[:3200])
        for prefix in [0, 80, 200, 600]:
            for gain_db in [0, -12, -24]:
                signal = np.concatenate([np.zeros(prefix * 16, np.float32), audio[3200:] * 10 ** (gain_db / 20)])
                x = torch.from_numpy(feature(signal))[None]
                hits = torch.nonzero(startup_mask(x)[0, ::2]).flatten()
                # Feature j includes audio up through nominal j*10+10 ms.
                available_ms = int(hits[0]) * 20 + 10 if len(hits) else None
                results.append({"id": row["id"], "prefix_ms": prefix, "gain_db": gain_db,
                                "allowed_audio_available_ms": available_ms,
                                "delay_from_recording_start_ms": available_ms - prefix if available_ms is not None else None})
    negative = []
    for amplitude in [0.0, 0.01, 0.5]:
        audio = np.zeros(16000, np.float32)
        audio[1600] = amplitude
        allowed = startup_mask(torch.from_numpy(feature(audio))[None])
        negative.append({"single_sample_impulse_amplitude": amplitude, "opened": bool(allowed.any())})
    delays = [r["delay_from_recording_start_ms"] for r in results if r["delay_from_recording_start_ms"] is not None]
    by_recording_gain = collections.defaultdict(list)
    for r in results:
        by_recording_gain[r["id"], r["gain_db"]].append(r["delay_from_recording_start_ms"])
    drift = [max(v) - min(v) for v in by_recording_gain.values() if None not in v]
    summary = {"training_recordings": len(rows), "conditions": len(results),
               "missed_signal": len(results) - len(delays),
               "opened_before_recording": sum(d < 0 for d in delays),
               "delay_from_recording_start_ms_quantiles": dict(zip(["min", "median", "p95", "max"], np.quantile(delays, [0, .5, .95, 1]).tolist())),
               "max_prefix_shift_drift_ms": max(drift), "silence_impulse_checks": negative,
               "scope": "Known inserted silence and original FSDD recording starts; not human-labeled speech onset or noisy-field VAD evaluation."}
    save(run / "signal-audit.json", {"summary": summary, "conditions": results, "source_sha256": digest(__file__)})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
