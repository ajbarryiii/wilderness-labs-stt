"""Entry point run by ./python inside the shared runtime: caches, sys.path, then the script or module.

    relabel_boot.py SCRIPT [ARGS...]      run SCRIPT as __main__
    relabel_boot.py -m MODULE [ARGS...]   run MODULE as __main__ (e.g. -m unittest discover -s tests)

The shared runtime's own wrapper sets its caches; they are replaced here, before anything imports
huggingface_hub, transformers, torch or numba, by directories under $RELABEL_ROOT/caches (and TMPDIR by
$RELABEL_ROOT/tmp). Roots are read from the environment (./python fills it from private/local.env).
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _need(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        sys.exit(f"relabel_boot: {key} is not set (environment or finetune/relabel/private/local.env)")
    return value


def configure() -> None:
    root = Path(_need("RELABEL_ROOT"))
    caches = root / "caches"
    dirs = {"HF_HOME": caches / "hf", "HF_HUB_CACHE": caches / "hf" / "hub", "HF_DATASETS_CACHE": caches / "datasets",
            "XDG_CACHE_HOME": caches, "TORCH_HOME": caches / "torch", "NUMBA_CACHE_DIR": caches / "numba",
            "TORCHINDUCTOR_CACHE_DIR": caches / "inductor", "TRITON_CACHE_DIR": caches / "triton",
            "TORCH_EXTENSIONS_DIR": caches / "extensions", "CUDA_CACHE_PATH": caches / "cuda",
            "NEMO_CACHE_DIR": caches / "nemo", "TMPDIR": root / "tmp"}
    for key, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    import tempfile

    tempfile.tempdir = None
    # The token saved by `hf auth login` (read only; HF_HOME no longer points at its default location).
    os.environ.setdefault("HF_TOKEN_PATH", os.path.expanduser("~/.cache/huggingface/token"))
    os.environ.update(HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", WANDB_MODE="disabled",
                      TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    first = [str(HERE), str(REPO / "finetune" / "parakeet-ternary")]
    last = [str(REPO / "finetune" / "whisper-ternary")]
    if os.environ.get("RELABEL_DEPS"):
        last.append(str(root / "deps" / os.environ["RELABEL_DEPS"]))
    rest = [p for p in sys.path if p not in first + last and Path(p or ".").resolve() != HERE]
    sys.path[:] = first + rest + last


def main() -> None:
    configure()
    args = sys.argv[1:]
    if not args:
        sys.exit("usage: ./python SCRIPT [ARGS] | ./python -m MODULE [ARGS]")
    if args[0] == "-m":
        sys.argv = [args[1]] + args[2:]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    elif args[0] == "-c":
        sys.argv = ["-c"] + args[2:]
        exec(compile(args[1], "<string>", "exec"), {"__name__": "__main__"})
    else:
        sys.argv = args
        runpy.run_path(args[0], run_name="__main__")


if __name__ == "__main__":
    main()
