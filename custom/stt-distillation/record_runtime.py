import importlib.metadata as md, json, platform, subprocess
from pathlib import Path
import numpy, torch, torchaudio
from common import ART, save, digest

out = Path(__file__).parent
pkgs = sorted(
    {
        (d.metadata["Name"], d.version)
        for d in md.distributions(path=[str(ART / "deps")])
    }
)
save(
    out / "runtime-lock.json",
    dict(
        python=platform.python_version(),
        torch=torch.__version__,
        torchaudio=torchaudio.__version__,
        numpy=numpy.__version__,
        cuda=torch.version.cuda,
        gpu=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        dependencies=[dict(name=n, version=v) for n, v in pkgs],
        base_runtime=str(
            Path("custom/autoresearch/.training-runtime/bin/python3").resolve()
        ),
        wrapper_sha256=digest(out / "python"),
    ),
)
files = [
    ART / "teachers/omi/omimedstt-v1.nemo",
    *[p for p in (ART / "teachers/whisper").iterdir() if p.is_file()],
]
save(
    ART / "input-lock.json",
    dict(
        teachers={
            str(p.relative_to(ART)): dict(bytes=p.stat().st_size, sha256=digest(p))
            for p in files
        }
    ),
)
print("Recorded runtime and teacher file hashes", flush=True)
