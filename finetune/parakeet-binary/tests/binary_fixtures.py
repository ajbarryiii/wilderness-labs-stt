"""Shared fixtures for the binary pilot tests (plans/pilot.md "Contract tests").

Every test runs on the CPU: CUDA is hidden before anything can initialise it. Small NeMo
models are built from the pinned checkpoint's own config and tokenizer (2 conformer layers,
d_model 64, 22 quantized modules), as in parakeet-ternary/tests/test_train.py; the full-size
architecture (random weights, about 2.5 GB) is built only by the quantized-set test, so run
the suite inside ./heavy:

    ./heavy binary-tests --mem-max 24G --runtime 30min --wait -- python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, before torch can initialise CUDA

import contextlib  # noqa: E402
import hashlib  # noqa: E402
import tarfile  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

import torch  # noqa: E402

import binary_paths  # noqa: E402
import paths  # noqa: E402

TEST_ROOT = binary_paths.TMP / "tests"
TINY_LAYERS = 2
PER_LAYER = 11  # parakeet-ternary quant.PER_LAYER
TINY_MODULES = TINY_LAYERS * PER_LAYER


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_dir() -> Path:
    """Config and tokenizer files of the pinned .nemo (no weights), extracted once."""
    paths.require_mount()
    out = TEST_ROOT / "base"
    if not (out / "done").exists():
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(paths.MODEL_FILE, "r:*") as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if member.isfile() and not name.endswith(".ckpt"):
                    (out / name).write_bytes(tar.extractfile(member).read())
        (out / "done").write_text("ok")
    return out


def _config(tiny: bool):
    from omegaconf import OmegaConf, open_dict
    base = base_dir()
    cfg = OmegaConf.load(base / "model_config.yaml")
    refs = {k: str(cfg.tokenizer[k]) for k in ("model_path", "vocab_path", "spe_tokenizer_vocab")}
    with open_dict(cfg):
        cfg.tokenizer.dir = str(base)
        for key, ref in refs.items():
            cfg.tokenizer[key] = str(base / ref.removeprefix("nemo:"))
        for ds in ("train_ds", "validation_ds", "test_ds"):
            cfg.pop(ds, None)
        if tiny:
            cfg.encoder.update(n_layers=TINY_LAYERS, d_model=64, n_heads=2, subsampling_conv_channels=16)
            cfg.model_defaults.update(enc_hidden=64, pred_hidden=32, joint_hidden=32)
            cfg.decoder.prednet.update(pred_hidden=32, pred_rnn_layers=2)
            cfg.joint.jointnet.update(joint_hidden=32, encoder_hidden=64, pred_hidden=32)
        # The pinned config's greedy decoder (CUDA graphs allowed, as for M1); on the CPU NeMo
        # decodes without graphs whatever the mode, but the mode is set and checked.
    return cfg, refs


def nemo_model(tiny: bool = True, seed: int = 0, cuda_graph_decoder: bool = True):
    """An FP32 EncDecRNNTBPEModel with seeded random weights (tiny, or the full architecture)."""
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    from omegaconf import open_dict
    cfg, refs = _config(tiny)
    if not cuda_graph_decoder:
        with open_dict(cfg):
            cfg.decoding.greedy.use_cuda_graph_decoder = False
    torch.manual_seed(seed)
    model = EncDecRNNTBPEModel(cfg=cfg)
    with open_dict(model.cfg):  # the exports look the tokenizer up in the .nemo by these names
        for key, ref in refs.items():
            model.cfg.tokenizer[key] = ref
    return model.float().eval()


def tiny_model(seed: int = 0, cuda_graph_decoder: bool = True):
    return nemo_model(tiny=True, seed=seed, cuda_graph_decoder=cuda_graph_decoder)


@contextlib.contextmanager
def tiny_module_count():
    """The 264-module count checks of both quantizers, set to the tiny model's 22."""
    import binary_quant
    import quant
    with mock.patch.object(binary_quant, "EXPECTED_MODULES", TINY_MODULES), \
            mock.patch.object(quant, "EXPECTED_MODULES", TINY_MODULES):
        yield


def write_checkpoint(path: Path, student_state: dict, *, weight_fraction: float | None = 1.0,
                     step: int | None = None, complete: bool = True) -> str:
    """A train.py checkpoint file (layout of train.save_checkpoint) with this student state;
    returns its SHA-256. The optimizer entry is a placeholder the init loader must ignore."""
    import train
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"step": binary_paths.M1_STEP if step is None else step, "student": student_state,
               "optimizer": {"state": {}, "param_groups": []}, "weight_fraction": weight_fraction,
               "format": train.CKPT_FORMAT, "complete": complete}
    torch.save(payload, path)
    return sha256_file(path)


def perturbed_state(model: torch.nn.Module, seed: int) -> dict:
    """The model's state_dict with every floating-point tensor perturbed (a stand-in for M1)."""
    generator = torch.Generator().manual_seed(seed)
    out = {}
    for key, value in model.state_dict().items():
        value = value.detach().clone()
        if value.is_floating_point() and "featurizer" not in key:
            value += 0.05 * torch.randn(value.shape, generator=generator)
        out[key] = value
    return out


@contextlib.contextmanager
def pinned_init(path: Path, digest: str):
    """binary_paths' pinned M1 checkpoint, hash and hash cache, pointed at a test file."""
    cache = TEST_ROOT / "sha256-cache"
    with mock.patch.object(binary_paths, "M1_CHECKPOINT", path), \
            mock.patch.object(binary_paths, "M1_SHA256", digest), \
            mock.patch.object(binary_paths, "CACHES", cache):
        yield


class StubEvaluator:
    """train.DevEvaluator stand-in over four noise clips split into dev sets. WER per set is
    `values[set]` if given, else a deterministic function of the model's encoder output."""

    def __init__(self, sets: tuple[str, ...] = ("dev_a", "dev_b"), values: dict | None = None) -> None:
        import numpy as np
        import soundfile as sf

        import evaluate
        root = TEST_ROOT / "stub-dev"
        root.mkdir(parents=True, exist_ok=True)
        self.evaluate, self.values = evaluate, values
        records = []
        for i in range(4):
            path = root / f"{i}.wav"
            audio = (0.1 * np.random.default_rng([7, i]).standard_normal(16000 + 4000 * i)).astype(np.float32)
            if not path.exists():
                sf.write(path, audio, 16000, subtype="FLOAT")
            records.append({"audio_filepath": str(path), "duration": len(audio) / 16000, "text": "x",
                            "id": f"dev:{i}"})
        self.subsets = {name: records[j % 2 * 2:j % 2 * 2 + 2] for j, name in enumerate(sets)}

    def full(self, model) -> dict:
        audio, lengths = self.evaluate.audio_batch([r for rs in self.subsets.values() for r in rs][:4])
        device = next(model.parameters()).device
        with torch.no_grad():
            enc, _ = model.forward(input_signal=audio.to(device), input_signal_length=lengths.to(device))
        value = float(enc.abs().mean())
        return {name: {"wer": self.values[name] if self.values else value * (j + 1), "utterances": 2,
                       "records": []} for j, name in enumerate(self.subsets)}

    subset = full


@contextlib.contextmanager
def stubbed_build(fp32_state: dict, m1_path: Path, m1_sha: str, evaluator_factory=None):
    """binary_train.build_context on the CPU: the pretrained model and the teacher are tiny models
    with `fp32_state`, M1 is the checkpoint at m1_path, the dev evaluator is a StubEvaluator, the
    stream is not opened, the CUDA TDT kernel is not built, the base .nemo is not hashed."""
    import binary_train
    import train

    def pretrained(device: str = "cpu"):
        model = tiny_model(seed=99)
        model.load_state_dict(fp32_state)
        return model

    factory = evaluator_factory or (lambda *a: StubEvaluator())
    with contextlib.ExitStack() as stack:
        for cm in (pinned_init(m1_path, m1_sha), tiny_module_count(),
                   mock.patch.object(train, "prepare_tdt_loss", return_value="stub"),
                   mock.patch.object(train, "verify_base", return_value={"base_sha256": "stub"}),
                   mock.patch.object(train, "load_pretrained", side_effect=pretrained),
                   mock.patch.object(train, "DevEvaluator", side_effect=factory),
                   mock.patch.object(binary_train, "open_stream", return_value=None)):
            stack.enter_context(cm)
        yield
