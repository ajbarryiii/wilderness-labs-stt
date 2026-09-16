"""Real-audio plumbing check; its tiny diagnostic scores are not experiment results."""

import copy, json, os, subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from common import ART, save, digest
from prepare import feature
from model import Model
from export import export_model, load_export

out = ART / "preflight/pipeline"
out.mkdir(parents=True, exist_ok=True)
# Exercise packed serialization, reloading and causality with actual audio.
x, sr = sf.read(ART / "preflight/smoke-1.wav", dtype="float32")
f = feature(x)
fp = out / "feature.npy"
np.save(fp, f)
cfg = json.loads(Path(__file__).with_name("pilot.json").read_text())
small = copy.deepcopy(cfg)
small["model"].update(width=32, depth=2, heads=2, context=16, dropout=0.0)
small["accumulation"] = 2
m = Model(small["model"], "ternary").eval()
a = torch.from_numpy(f).unsqueeze(0)
with torch.no_grad():
    expected = m(a)
export_model(m, out / "packed-check", 0, packed=True)
restored, meta = load_export(out / "packed-check", device="cpu")
with torch.no_grad():
    actual = restored(a)
assert torch.allclose(expected, actual, atol=3e-5), "Packed export changed logits"
rows = []
for domain in ["general", "medical_symptoms", "digits"]:
    for split in ["train", "calibration", "development"]:
        rows.append(
            dict(
                id=domain + "-" + split,
                domain=domain,
                split=split,
                text="one",
                audio=str(ART / "preflight/smoke-1.wav"),
                features=str(fp),
                feature_sha256=digest(fp),
                seconds=len(x) / sr,
                frames=f.shape[-1],
            )
        )
save(
    out / "manifest.json",
    dict(
        rows=rows,
        scope="Repeated real audio with plumbing-only labels; never used by main experiment.",
    ),
)
save(
    out / "targets.json",
    dict(
        policy={
            d: dict(weights=dict(omi=0.6, whisper=0.4))
            for d in ["general", "medical_symptoms", "digits"]
        },
        targets={
            r["id"]: dict(omi="one", whisper="a") for r in rows if r["split"] == "train"
        },
    ),
)
env = dict(os.environ, WILDERNESS_STT_MANIFEST=str(out / "manifest.json"))
py = Path(__file__).with_name("python")
here = Path(__file__).parent
save(out / "config.json", small)
for name in ["fp_multi", "ternary_multi"]:
    subprocess.run(
        [
            str(py),
            str(here / "train.py"),
            str(out),
            name,
            "--steps",
            "5",
            "--seconds",
            "300",
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            str(py),
            str(here / "evaluate.py"),
            str(out / name / "export"),
            str(out / name / "development.json"),
        ],
        env=env,
        check=True,
    )
# One actual full-size accumulated optimizer update + reload/evaluation.
save(out / "config.json", cfg)
subprocess.run(
    [
        str(py),
        str(here / "train.py"),
        str(out),
        "ternary_single",
        "--steps",
        "2",
        "--seconds",
        "300",
    ],
    env=env,
    check=True,
)
subprocess.run(
    [
        str(py),
        str(here / "evaluate.py"),
        str(out / "ternary_single/export"),
        str(out / "ternary_single/development.json"),
    ],
    env=env,
    check=True,
)
save(
    ART / "preflight/pipeline-passed.json",
    dict(
        packed_export_logits_equal=True,
        small_fp_and_ternary_training=True,
        full_size_accumulated_training=True,
        independent_export_evaluation=True,
        production_data_used=False,
    ),
)
print("PIPELINE CHECK PASSED", flush=True)
