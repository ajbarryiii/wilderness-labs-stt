"""Assemble a Hugging Face upload folder for a ternary export, then check it standalone.

    finetune/whisper-ternary/python finetune/whisper-ternary/hf/stage.py USER/NAME [--export DIR]

Copies export.safetensors and manifest.json (SHA-256 re-checked), the base model's
tokenizer and feature-extractor files (hash-verified against the pinned lock), and
load_ternary.py, and renders MODEL_CARD.md as README.md. Then loads the folder with
load_ternary.py and requires logits identical to the repository's export.load_export
on a fixed input. Writes only under /mnt/hd/wilderness-labs-stt/whisper-ternary/hf-staging/.
Uploading is a separate, explicit step (see the printed command).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import torch

import checkpoint
import export
import paths

HERE = Path(__file__).resolve().parent
STAGING = paths.ARTIFACTS / "hf-staging"
# MODEL_CARD.md states this run's numbers (12.2 MB, 12.12% / 28.42%); only it may be staged.
EXPECTED = {"run_name": "v2-ternary-embed-lr1e-3", "arm": "ternary-embed", "file_bytes": 12192744}
DEFAULT_EXPORT = paths.RUNS / EXPECTED["run_name"] / "export"
LICENSE_TEXT = paths.REPO / "finetune" / "stt" / "deps" / "huggingface_hub-0.36.2.dist-info" / "licenses" / "LICENSE"
PROCESSOR_FILES = ("added_tokens.json", "merges.txt", "normalizer.json",
                   "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
                   "tokenizer_config.json", "vocab.json")


def render_card(repo_id: str, manifest: dict) -> str:
    hist = manifest["code_histogram"]
    summary = (f"{100 * hist['minus_one']:.1f}% -1, {100 * hist['zero']:.1f}% 0, "
               f"{100 * hist['plus_one']:.1f}% +1")
    text = (HERE / "MODEL_CARD.md").read_text()
    for key, value in {"{repo_id}": repo_id, "{repo_name}": repo_id.split("/")[1],
                       "{code_histogram}": summary}.items():
        text = text.replace(key, value)
    return text


def standalone_check(folder: Path, reference_export: Path) -> dict:
    spec = importlib.util.spec_from_file_location("load_ternary", folder / "load_ternary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, processor = module.load(folder)
    reference = export.load_export(reference_export)
    torch.manual_seed(0)
    features = torch.randn(1, 80, 3000)
    decoder_ids = torch.tensor([[50257, 50362, 1169, 1049, 13]])
    with torch.no_grad():
        a = model(input_features=features, decoder_input_ids=decoder_ids).logits
        b = reference(input_features=features, decoder_input_ids=decoder_ids).logits
    ids = processor.tokenizer("hello world").input_ids
    return {"logits_identical": bool(torch.equal(a, b)),
            "tokenizer_ids_ok": ids == [50257, 50362, 31373, 995, 50256],
            "generation_config_equal": model.generation_config.to_dict()
            == reference.generation_config.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id", help="USER/NAME on huggingface.co")
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    args = parser.parse_args()
    if args.repo_id.count("/") != 1:
        parser.error("repo_id must be USER/NAME")
    paths.require_mount()
    manifest = json.loads((args.export / export.MANIFEST).read_text())
    if checkpoint.sha256(args.export / manifest["file"]) != manifest["sha256"]:
        sys.exit("export does not match its manifest SHA-256")
    found = {"run_name": manifest["extra"].get("run_name"), "arm": manifest["extra"].get("arm"),
             "file_bytes": manifest["bytes"]["file_bytes"]}
    if found != EXPECTED:
        sys.exit(f"the model card describes {EXPECTED}, but this export is {found}")
    if "Apache License" not in LICENSE_TEXT.read_text() or "END OF TERMS" not in LICENSE_TEXT.read_text():
        sys.exit(f"{LICENSE_TEXT} is not the full Apache-2.0 text")
    checkpoint.verify_lock(files=[n for n in PROCESSOR_FILES
                                  if n in json.loads((paths.MODEL_DIR / "lock.json").read_text())["files"]])
    folder = STAGING / args.repo_id.replace("/", "__")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for name in (manifest["file"], export.MANIFEST):
        shutil.copy2(args.export / name, folder / name)
    for name in PROCESSOR_FILES:
        shutil.copy2(paths.MODEL_DIR / name, folder / name)
    shutil.copy2(HERE / "load_ternary.py", folder / "load_ternary.py")
    shutil.copy2(LICENSE_TEXT, folder / "LICENSE")
    shutil.copy2(HERE / "NOTICE", folder / "NOTICE")
    card = render_card(args.repo_id, manifest)
    from huggingface_hub import ModelCard
    data = ModelCard(card).data  # raises on invalid YAML front matter
    if data.model_name != args.repo_id.split("/")[1] or data.license != "apache-2.0":
        sys.exit(f"model card metadata did not render as expected: {data.model_name!r}")
    if "{" + "repo_id}" in card or "{" + "repo_name}" in card or "{" + "code_histogram}" in card:
        sys.exit("unrendered placeholder in the model card")
    (folder / "README.md").write_text(card)
    check = standalone_check(folder, args.export)
    print(json.dumps({"folder": str(folder), "files": sorted(p.name for p in folder.iterdir()),
                      "bytes": sum(p.stat().st_size for p in folder.iterdir()), **check}, indent=1))
    if not all(check.values()):
        sys.exit("standalone check failed; do not upload")
    print(f"\nUpload with:\n  PYTHONPATH=finetune/stt/deps custom/autoresearch/.training-runtime/bin/python3 "
          f"-m huggingface_hub.cli.hf upload {args.repo_id} {folder} . --repo-type model")


if __name__ == "__main__":
    main()
