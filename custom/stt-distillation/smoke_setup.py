import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from common import ART, save
from scipy.signal import resample_poly

out = ART / "preflight"
out.mkdir(exist_ok=True)
batch = next(
    pq.ParquetFile(
        next((ART / "datasets/medical/data").glob("*.parquet"))
    ).iter_batches(batch_size=2)
).to_pylist()
rows = []
for i, r in enumerate(batch):
    a = r["audio"]
    x = np.asarray(a["array"], dtype=np.float32).squeeze()
    sr = a["sampling_rate"]
    if x.ndim == 2:
        x = x.mean(axis=0)
    print(dict(shape=x.shape, sample_rate=sr, text=r["sentence"]), flush=True)
    if sr != 16000:
        x = resample_poly(x, 16000, sr)
    p = out / f"smoke-{i}.wav"
    sf.write(p, x, 16000)
    rows.append(
        dict(
            id=f"smoke-{i}",
            audio=str(p),
            text=r["sentence"],
            split="train",
            domain="medical_symptoms",
        )
    )
save(out / "smoke-input.json", rows)
