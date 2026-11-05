"""Stage the Hugging Face repository of Kea-B 0.1 (stage 4's run b100-main) and check it (release-kea-b-0.1 plan
"Staging, upload, verification"). Never uploads.

    job start kea-b-0.1-stage --needs gpu:shared --vram 4G --mem 16G --priority normal \\
        --env HF_HUB_OFFLINE=1 -- ./python -m hf.stage            (from finetune/relabel)

Measured 2026-11-04: host peak 12.8 GB, GPU peak 3.2 GB, under a minute once the shared GPU lock is free.

Outputs, under $RELABEL_ROOT/release/kea-b-0.1/: staging/ (the Hub repository layout), stage-report.json (PASSED)
or stage-report.FAILED.json, provenance.json (the derived training-source mapping; private). Graphics are copied from
assets/ there (any regular files; none is fine). The private display file display.json gives the public names of the
registry sources the release names and extra scrub terms.

Every check is recorded by name; success is the conjunction of all of them, and the set of names is fixed before the
first check runs (required_checks), so none can be skipped silently:

1. Source pins: the bundle's and the view's files are keab_test_paths.PINNED_FILES; the bundle's tokenizer equals the
   view's; K-fp's test scoring was of that view and its per-set files are the ones its summary records.
2. The view derived from the bundle (keab_export.derive_view, into a temporary directory) is the shipped view, byte
   for byte, and the shipped view's tensors are the bundle's widened exactly (keab_export.check_view).
3. The staged loader (imported by file path): the bundle's state == the compat view's state ==
   binary_export.load_export(source view)'s, every tensor bit for bit, all FP32, from the release-form manifests.
4. Transcripts: the staged loader's model from the bundle, on CUDA under the shared GPU lock, reproduces K-fp's
   recorded hypothesis of each check clip: the first CLIPS_PER_SET utterances (by id, evaluate.read_manifest's order)
   of each of the seven mean sets, decoded per set.
5. Documents: rendered from templates with every number computed from the result files (test.json, the run summary,
   results.json, the staged files); no placeholder is left; the card's YAML and its table and model-index are the
   computed ones.
6. Scrubbing: no staged file holds a machine path, this host's name, or a name, alias or URL of a private registry
   source that the release does not train on; manifests ship in release form (machine paths under "extra" removed;
   the report records each original's SHA-256 and the removed keys).
7. Size facts (bundle tensor bytes, file sizes) are computed from the staged files.
Plus: the staged file set is the expected layout, each copied file equals its source, and the training sources are
derived and permissive (hf/provenance.py).

On success the temporary folder becomes staging/ and the report carries bundle_sha256, the approval token for
hf/upload.py. On any failure no staging/ folder is left and only the FAILED report is written.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import traceback
from pathlib import Path

import keab_test_paths as tp
import paths
import relabel_paths
from hf import kit
from hf import provenance as pv

HERE = Path(__file__).resolve().parent
RELEASE = relabel_paths.ARTIFACTS / "release" / kit.RELEASE_NAME
STAGING = RELEASE / "staging"
REPORT, REPORT_FAILED = RELEASE / "stage-report.json", RELEASE / "stage-report.FAILED.json"
ASSETS_SRC = RELEASE / "assets"
PROVENANCE = RELEASE / "provenance.json"
DISPLAY = RELEASE / "display.json"
TMP_ROOT = RELEASE / "tmp"
RUN_SUMMARY, RUN_CONFIG = tp.RUN / "summary.json", tp.RUN / "config.json"
FP_TEST = tp.test_dir("fp")
REGISTRY = relabel_paths.ARTIFACTS / "private" / "sources.json"
PLANS = relabel_paths.ARTIFACTS / "training" / "batch-plan"
CLIPS_PER_SET = 2
CLIP_SETS = tuple(paths.MEAN_SETS)
KEA_B0, KEA_T = "wildernesslabs/kea-b", "wildernesslabs/kea-t"
GITHUB = "https://github.com/ajbarryiii/wilderness-labs-stt"
BASE_ID, BASE_REVISION = tp.BASE_MODEL["id"], tp.BASE_MODEL["revision"]
LABELS = {"librispeech_clean": "LS clean", "librispeech_other": "LS other", "ami": "AMI", "earnings22": "Earnings-22",
          "gigaspeech": "GigaSpeech", "spgispeech": "SPGISpeech", "voxpopuli": "VoxPopuli", "mean": "Mean of 7",
          "common_voice": "Common Voice"}
DATASET_NAMES = {"librispeech_clean": "LibriSpeech test-clean", "librispeech_other": "LibriSpeech test-other",
                 "ami": "AMI test", "earnings22": "Earnings-22 test", "gigaspeech": "GigaSpeech test",
                 "spgispeech": "SPGISpeech test", "voxpopuli": "VoxPopuli test", "common_voice": "Common Voice test"}
CARD_ROWS = (("NVIDIA published", "[Original](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (NVIDIA's numbers)"),
             ("M1", f"[Kea-T](https://huggingface.co/{KEA_T})"),
             ("MB1", f"[Kea-B 0.0](https://huggingface.co/{KEA_B0})"),
             ("Kea-B' K-fp", "**Kea-B 0.1 (this model)**"))
EVAL_ROWS = (("NVIDIA published", "NVIDIA published"), ("B0", "Original, FP32 (our scoring)"),
             ("M1", "Kea-T"), ("MB1", "Kea-B 0.0"), ("Kea-B' K-fp", "Kea-B 0.1, FP32 rebuild (K-fp)"),
             ("Kea-B' K-x1", "Kea-B 0.1, i8x1 kernels (K-x1)"))
SOURCE_PINS = ("export/bundle/bundle.safetensors", "export/bundle/manifest.json", "export/view/export.safetensors",
               "export/view/manifest.json", "export/view/tokenizer/tokenizer.model",
               "export/view/tokenizer/tokenizer.vocab", "export/view/tokenizer/vocab.txt")

FIXED_CHECKS = (
    "1 source pins: bundle and view files == keab_test_paths.PINNED_FILES",
    "1 bundle tokenizer files == view tokenizer files == both manifests",
    "1 K-fp scoring is of the pinned view; its per-set files == its summary's SHA-256",
    "2 bundle-derived view export.safetensors == shipped view, byte for byte",
    "2 shipped view tensors == bundle tensors widened exactly",
    "3 loader: bundle state == compat state (keys, order, dtypes, every tensor bit for bit)",
    "3 loader: bundle state == binary_export.load_export(source view) state",
    "3 loader: every floating tensor FP32, model in eval mode",
    "4 decoder CUDA-graph mode no_while_loops",
    "5 no unfilled placeholder in any rendered document",
    "5 card YAML: license, base_model, relation, library, datasets == trained Hub corpora",
    "5 card model-index == round(K-fp, 2) for every test set",
    "5 card results table == the computed rows",
    "6 scrub: no machine path, hostname or private source name in any staged file",
    "6 manifests in release form: only machine paths under extra removed, loader inputs unchanged",
    "7 size facts from staged files: bundle tensor bytes == manifest == keab_test_paths.BUNDLE_TENSOR_BYTES",
    "staged file set == the expected layout",
    "staged weights and tokenizer files == their sources (SHA-256)",
    "training sources derived from the run's records, all permissive",
    "5 the documents' image links == the staged assets, each linked at least once",
)


class StageFailed(Exception):
    pass


def required_checks(clips: list[dict]) -> set[str]:
    return set(FIXED_CHECKS) | {clip_check(c) for c in clips}


def clip_check(clip: dict) -> str:
    return f"4 clip {clip['set']}/{clip['id']}: staged transcript == K-fp hypothesis"


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def utc() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------- inputs

def plan_json(config: dict) -> Path:
    """The batch plan the run trained on: the plan.json whose SHA-256 is the run config's plan_sha256."""
    want = config["args"]["plan_sha256"]
    found = [p for p in sorted(PLANS.glob("*/plan.json")) if kit.sha256_file(p) == want]
    if len(found) != 1:
        raise StageFailed(f"expected one batch plan with SHA-256 {want}, found {found}")
    return found[0]


def check_clips() -> list[dict]:
    """The fixed check clips (module docstring, check 4), with their K-fp hypotheses."""
    import evaluate

    clips = []
    for name in CLIP_SETS:
        records = evaluate.read_manifest(tp.manifest_path(name))[:CLIPS_PER_SET]
        clips += [{"set": name, "id": r["id"], "audio_filepath": r["audio_filepath"], "duration": r["duration"]}
                  for r in records]
    return clips


def kfp_hypotheses(clips: list[dict], summary: dict, checks: dict | None = None) -> dict:
    """{(set, id): K-fp hypothesis}, reading each set's file once and checking it against the summary's SHA-256."""
    out, ok = {}, True
    for name in sorted({c["set"] for c in clips}):
        path = FP_TEST / f"{name}.json"
        ok &= kit.sha256_file(path) == summary.get("per_set_sha256", {}).get(name)
        hyps = {r["id"]: r["hyp"] for r in read_json(path)["records"]}
        out.update({(name, c["id"]): hyps.get(c["id"]) for c in clips if c["set"] == name})
    if checks is not None:
        checks["_kfp_files_ok"] = ok
    return out


# ----------------------------------------------------------------------------- numbers and documents

def f2(x: float) -> str:
    return f"{x:.2f}"


def signed(x: float) -> str:
    return f"{x:+.2f}"


def row_cells(row: dict, columns: list[str]) -> list[str]:
    return [f2(row[c]) if c in row else "n/a" for c in columns]


def md_table(header: list[str], rows: list[list[str]], align_first_left: bool = True) -> str:
    sep = ["---"] + ["---:"] * (len(header) - 1) if align_first_left else ["---:"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def results_rows(test: dict, spec) -> list[list[str]]:
    columns = test["columns"]
    return [[label, *row_cells(test["rows"][arm], columns)] for arm, label in spec]


def results_header(test: dict) -> list[str]:
    return ["Model", *(LABELS[c] for c in test["columns"])]


def model_index_yaml(test: dict, repo_name: str) -> str:
    row = test["rows"]["Kea-B' K-fp"]
    lines = ["model-index:", f"- name: {repo_name}", "  results:"]
    for name, (config, split) in paths.TEST_SETS.items():
        lines += ["  - task:", "      type: automatic-speech-recognition", "      name: Automatic Speech Recognition",
                  "    dataset:", f"      name: {DATASET_NAMES[name]} (ESB test bundle)", f"      type: {paths.ESB_REPO}",
                  f"      config: {config}", f"      split: {split}", f"      revision: {paths.ESB_REVISION}",
                  "      args:", "        language: en", "    metrics:", "    - type: wer",
                  f"      value: {f2(row[name])}", "      name: Test WER"]
    return "\n".join(lines)


def values(*, test: dict, summary: dict, config: dict, results: dict, prov: dict, sizes: dict, view_manifest: dict,
           repo_id: str, assets: list[str]) -> dict[str, str]:
    """Every value the templates use, computed from its source (plan check 5)."""
    rows, cols = test["rows"], test["columns"]
    kfp, kx1, mb1, m1, nv = (rows[k] for k in ("Kea-B' K-fp", "Kea-B' K-x1", "MB1", "M1", "NVIDIA published"))
    x1_max = max(abs(kx1[c] - kfp[c]) for c in cols if c != "mean" and c in kx1 and c in kfp)
    gate = results["gate"]
    gate_rows = [[s, f2(gate["delta"]["wer"][s]["a"]), f2(gate["delta"]["wer"][s]["b"]),
                  f"{signed(-gate['delta']['sets'][s]['point'])} [{signed(-gate['delta']['sets'][s]['ci'][1])}, "
                  f"{signed(-gate['delta']['sets'][s]['ci'][0])}]"] for s in gate["delta"]["sets"]]
    dev = test["dev_check"]
    empty = test["empty_hypothesis_fraction"]
    empty_sets = [c for c in cols if c != "mean"]
    hours = (datetime.datetime.strptime(summary["finished_utc"], "%Y-%m-%dT%H:%M:%SZ")
             - datetime.datetime.strptime(summary["started_utc"], "%Y-%m-%dT%H:%M:%SZ")).total_seconds() / 3600
    sched = config["args"]["schedule"]
    datasets = pv.card_datasets(prov)
    return {
        "repo_id": repo_id, "repo_name": repo_id.split("/")[1], "kea_b0": KEA_B0, "kea_t": KEA_T, "github": GITHUB,
        "base_id": BASE_ID, "base_revision": BASE_REVISION,
        "datasets_yaml": "\n".join(f"- {d}" for d in datasets),
        "model_index": model_index_yaml(test, repo_id.split("/")[1]),
        "card_table": md_table(results_header(test), results_rows(test, CARD_ROWS)),
        "eval_table": md_table(results_header(test), results_rows(test, EVAL_ROWS)),
        "diff_table": md_table(["Difference", *(LABELS[c] for c in cols)],
                               [[f"Kea-B 0.1 {m} - Kea-B 0.0 (points)", *(signed(test["differences"][k][c]["points"])
                                 if c in test["differences"][k] else "n/a" for c in cols)]
                                for m, k in (("K-fp", "Kea-B' K-fp - MB1"), ("K-x1", "Kea-B' K-x1 - MB1"))]),
        "empty_table": md_table(["Model", *(LABELS[c] for c in empty_sets)],
                                [[label, *(f"{100 * empty[arm][c]:.2f}%" for c in empty_sets)]
                                 for arm, label in EVAL_ROWS if arm in empty]),
        "gate_table": md_table(["Dev set", "Control", "Kea-B 0.1", "Kea-B 0.1 - control [90% interval]"], gate_rows),
        "gate_delta": f"{gate['delta']['h4']['point']:.2f}",
        "gate_ci": f"[{gate['delta']['h4']['ci'][0]:.2f}, {gate['delta']['h4']['ci'][1]:.2f}]",
        "dev_x1_mean": f"{dev['x1']['mean_percent']:.2f}", "dev_x1_recorded": f"{dev['x1']['recorded_mean_percent']:.2f}",
        "dev_mb1_mean": f"{dev['fp']['mb1_mean_percent']:.2f}",
        "dev_token_identity": f"{100 * dev['x1']['token_identity_rate']:.2f}%",
        "dev_fp_sets": ", ".join(f"{k} {f2(v)}" for k, v in dev["fp"]["keab_fp_dev_percent"].items()),
        "keab_mean": f2(kfp["mean"]), "keab_x1_mean": f2(kx1["mean"]), "nvidia_mean": f2(nv["mean"]),
        "kea_b0_mean": f2(mb1["mean"]), "kea_t_mean": f2(m1["mean"]), "b0_mean": f2(rows["B0"]["mean"]),
        "keab_cv": f2(kfp["common_voice"]), "kea_b0_cv": f2(mb1["common_voice"]),
        "keab_ls_clean": f2(kfp["librispeech_clean"]),
        "spgi_delta": f2(test["differences"]["Kea-B' K-fp - MB1"]["spgispeech"]["points"]),
        "x1_max_set_diff": f"{math.ceil(round(x1_max * 100, 9)) / 100:.2f}",   # an upper bound, rounded up
        "bundle_mb": f"{sizes['bundle_tensor_bytes'] / 1e6:.2f}", "bundle_file_bytes": f"{sizes['bundle_file_bytes']:,}",
        "view_file_bytes": f"{sizes['view_file_bytes']:,}", "view_mb": f"{sizes['view_tensor_bytes'] / 1e6:.1f}",
        "kea_b0_mb": f"{test['sizes_mb']['MB1']:.1f}", "kea_t_mb": f"{test['sizes_mb']['M1']:.1f}",
        "original_mb": f"{test['sizes_mb']['B0']:,.0f}",
        "steps": f"{summary['steps']:,}", "lr": f"{summary['lr']:.0e}".replace("e-0", "e-"),
        "warmup": f"{sched['warmup']:,}", "cooldown": f"{sched['cooldown']:,}",
        "batch_seconds": f"{summary['batch_seconds']:.0f}", "audio_hours": f"{summary['audio_hours_seen']:,.0f}",
        "train_hours": f"{hours:.1f}", "seed": str(summary["seed"]),
        "teacher_sha256": config["args"]["teacher"]["sha256"],
        "nemo_version": view_manifest["versions"]["nemo"], "torch_version": view_manifest["versions"]["torch"],
        "training_table": pv.training_table(prov), "synthetic_parts": pv.synthetic_parts(prov),
        "notice_data": pv.notice_data(prov), "notice_tools": pv.notice_tools(prov),
    }


def asset_links(docs: dict[str, str]) -> set[str]:
    """The asset file names that the documents' Markdown images and links point at (relative assets/ paths)."""
    return {m for text in docs.values() for m in re.findall(r"\]\(assets/([^)\s]+)\)", text)}


def card_checks(card: str, test: dict, prov: dict, repo_id: str, checks: dict, info: dict) -> None:
    from huggingface_hub import ModelCard

    data = ModelCard(card).data   # raises on invalid YAML
    checks["5 card YAML: license, base_model, relation, library, datasets == trained Hub corpora"] = (
        data.license == "cc-by-4.0" and data.base_model == KEA_B0
        and data.to_dict().get("base_model_relation") == "finetune" and data.library_name == "nemo"
        and list(data.datasets or []) == pv.card_datasets(prov))
    kfp = test["rows"]["Kea-B' K-fp"]
    by_dataset = {v: k for k, v in paths.TEST_SETS.items()}
    seen, ok = {}, True
    for r in data.eval_results or []:
        key = by_dataset.get((r.dataset_config, r.dataset_split))
        ok &= (key is not None and key not in seen and r.dataset_type == paths.ESB_REPO
               and r.dataset_revision == paths.ESB_REVISION and r.metric_type == "wer"
               and r.metric_value == round(kfp[key], 2))
        seen[key] = r.metric_value
    checks["5 card model-index == round(K-fp, 2) for every test set"] = ok and set(seen) == set(paths.TEST_SETS)
    table = md_table(results_header(test), results_rows(test, CARD_ROWS))
    checks["5 card results table == the computed rows"] = card.count(table) == 1
    info["card"] = {"model_name": repo_id.split("/")[1], "datasets": list(data.datasets or []),
                    "model_index": seen, "sha256": kit.sha256_bytes(card.encode())}


# ----------------------------------------------------------------------------- the stage

def load_loader(path: Path):
    spec = importlib.util.spec_from_file_location("staged_load_kea_b", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_states(a: dict, b: dict) -> list[str]:
    import torch

    if list(a) != list(b):
        return ["key order or set differs"]
    return [k for k in a if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not torch.equal(a[k], b[k])]


def stage(repo_id: str, *, gpu: bool = True) -> tuple[bool, dict, Path]:
    """Run every check and stage repo_id's folder; (passed, report, report path)."""
    relabel_paths.require_mount()
    RELEASE.mkdir(parents=True, exist_ok=True)
    tmp = RELEASE / f".staging.tmp-{os.getpid()}"
    for stale in (REPORT, REPORT_FAILED):
        stale.unlink(missing_ok=True)
    for stale in (STAGING, *RELEASE.glob(".staging.tmp-*"), TMP_ROOT):
        if stale.exists():
            shutil.rmtree(stale)
    checks: dict[str, bool] = {}
    info: dict = {"repo_id": repo_id, "started_utc": utc(), "run": tp.RUN_NAME}
    clips: list[dict] = []
    try:
        import train_keab as tk   # noqa: F401  first, as keab_test: train.py's CUDA/numba settings before torch
        import torch

        import binary_export
        import binary_train
        from training import keab_export as ke

        if gpu and not torch.cuda.is_available():
            raise StageFailed("CUDA is required for the transcript check (K-fp was decoded on CUDA)")
        test, summary, config = read_json(tp.TEST_JSON), read_json(RUN_SUMMARY), read_json(RUN_CONFIG)
        fp_summary, results = read_json(FP_TEST / "summary.json"), read_json(tp.STAGE4_RESULTS_JSON)
        clips = check_clips()
        info["clips_rule"] = f"first {CLIPS_PER_SET} utterances by id of each of {list(CLIP_SETS)}"

        # 1. source pins
        digests = {rel: kit.sha256_file(tp.RUN / rel) for rel in SOURCE_PINS}
        info["source_sha256"] = digests
        checks[FIXED_CHECKS[0]] = all(digests[rel] == tp.PINNED_FILES[rel] for rel in SOURCE_PINS)
        bman, vman = read_json(tp.BUNDLE / kit.MANIFEST), read_json(tp.VIEW / kit.MANIFEST)
        tok = {rel: (kit.sha256_file(tp.BUNDLE / rel), kit.sha256_file(tp.VIEW / rel)) for rel in kit.TOKENIZER}
        checks[FIXED_CHECKS[1]] = all(b == v == bman["files"][rel] == vman["files"][rel] for rel, (b, v) in tok.items())
        hyps = kfp_hypotheses(clips, fp_summary, checks)
        checks[FIXED_CHECKS[2]] = (checks.pop("_kfp_files_ok") and fp_summary["source"]["sha256"] == tp.VIEW_SHA256
                                   and test["identity"]["view_sha256"] == tp.VIEW_SHA256
                                   and test["identity"]["bundle_sha256"] == tp.BUNDLE_SHA256
                                   and fp_summary["mean_wer"] == test["rows"]["Kea-B' K-fp"]["mean"])
        for c in clips:
            c["kfp"] = hyps[(c["set"], c["id"])]

        # provenance (private mapping; public text for NOTICE and TRAINING.md)
        prov = pv.derive(**pv.gather(RUN_SUMMARY, RUN_CONFIG, plan_json(config), REGISTRY, DISPLAY))
        kit.write_json(PROVENANCE, prov)
        display = read_json(DISPLAY)
        info["provenance"] = {"path": str(PROVENANCE), "sha256": kit.sha256_file(PROVENANCE),
                              "display_sha256": kit.sha256_file(DISPLAY), "registry_sha256": kit.sha256_file(REGISTRY)}
        checks[FIXED_CHECKS[18]] = True

        # 2. bundle-derived view == shipped view
        TMP_ROOT.mkdir(parents=True)
        ke.derive_view(tp.BUNDLE, TMP_ROOT / "view", root=TMP_ROOT)
        checks[FIXED_CHECKS[3]] = kit.sha256_file(TMP_ROOT / "view" / kit.VIEW_WEIGHTS) == tp.VIEW_SHA256
        manifest, tensors = ke.read_bundle(tp.BUNDLE)
        try:
            ke.check_view(tp.VIEW, manifest, tensors)
            checks[FIXED_CHECKS[4]] = True
        except ValueError as error:
            checks[FIXED_CHECKS[4]] = False
            info["check_view_error"] = str(error)
        del tensors
        shutil.rmtree(TMP_ROOT)

        # staging: weights, tokenizer, release-form manifests, loader, LICENSE, assets
        tmp.mkdir()
        copies = {**{rel: tp.BUNDLE / rel for rel in (kit.BUNDLE_WEIGHTS, *kit.TOKENIZER)},
                  **{f"{kit.COMPAT}/{rel}": tp.VIEW / rel for rel in (kit.VIEW_WEIGHTS, *kit.TOKENIZER)},
                  kit.LOADER: HERE / kit.LOADER, "LICENSE": HERE / "LICENSE"}
        assets = sorted(p.name for p in ASSETS_SRC.glob("*") if p.is_file() and not p.name.startswith(".")) \
            if ASSETS_SRC.is_dir() else []
        copies.update({f"{kit.ASSETS}/{a}": ASSETS_SRC / a for a in assets})
        for rel, src in copies.items():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, tmp / rel)
        info["manifests"] = {}
        release_ok = True
        for rel, original in ((kit.MANIFEST, bman), (f"{kit.COMPAT}/{kit.MANIFEST}", vman)):
            form, removed = kit.release_form(original)
            (tmp / rel).write_text(kit.manifest_text(form))
            src = tp.BUNDLE if rel == kit.MANIFEST else tp.VIEW
            info["manifests"][rel] = {"original_sha256": kit.sha256_file(src / kit.MANIFEST),
                                      "release_sha256": kit.sha256_file(tmp / rel), "removed_keys": removed}
            loader_keys = ("format", "file", "sha256", "files", "config", "binary_layers", "int8", "tensors",
                           "quantized_layers")
            release_ok &= (bool(removed) and all(k.startswith("extra.") for k in removed)
                           and {k: v for k, v in form.items() if k != "extra"}
                           == {k: v for k, v in original.items() if k != "extra"}
                           and all(form.get(k) == original.get(k) for k in loader_keys))
        checks[FIXED_CHECKS[14]] = release_ok

        # 7. size facts from the staged files
        sizes = {"bundle_file_bytes": (tmp / kit.BUNDLE_WEIGHTS).stat().st_size,
                 "bundle_tensor_bytes": (tmp / kit.BUNDLE_WEIGHTS).stat().st_size
                 - kit.safetensors_header_bytes(tmp / kit.BUNDLE_WEIGHTS),
                 "view_file_bytes": (tmp / kit.COMPAT / kit.VIEW_WEIGHTS).stat().st_size,
                 "view_tensor_bytes": (tmp / kit.COMPAT / kit.VIEW_WEIGHTS).stat().st_size
                 - kit.safetensors_header_bytes(tmp / kit.COMPAT / kit.VIEW_WEIGHTS)}
        info["size_facts"] = sizes
        checks[FIXED_CHECKS[15]] = (sizes["bundle_tensor_bytes"] == bman["bytes"]["tensor_bytes"]
                                    == tp.BUNDLE_TENSOR_BYTES and sizes["bundle_file_bytes"] == bman["bytes"]["file_bytes"])

        # 5. documents
        vals = values(test=test, summary=summary, config=config, results=results, prov=prov, sizes=sizes,
                      view_manifest=vman, repo_id=repo_id, assets=assets)
        docs = {}
        for rel, template in (("README.md", "MODEL_CARD.md"), ("EVALUATION.md", "EVALUATION.md"),
                              ("TRAINING.md", "TRAINING.md"), ("NOTICE", "NOTICE")):
            docs[rel] = kit.render((HERE / template).read_text(), vals)
            (tmp / rel).write_text(docs[rel])
        checks[FIXED_CHECKS[9]] = not any(kit.unfilled(t) for t in docs.values())
        card_checks(docs["README.md"], test, prov, repo_id, checks, info)
        checks[FIXED_CHECKS[19]] = asset_links(docs) == set(assets)

        listing = kit.listing(tmp)
        checks[FIXED_CHECKS[16]] = sorted(listing) == kit.expected_files(assets)
        checks[FIXED_CHECKS[17]] = all(listing.get(rel, {}).get("sha256") == kit.sha256_file(src)
                                       for rel, src in copies.items())

        # 6. scrubbing
        terms = kit.scrub_terms(read_json(REGISTRY), set(prov["allowed_aliases"]), display.get("scrub_extra", []))
        findings = kit.scrub_findings(tmp, terms, kit.hostnames())
        checks[FIXED_CHECKS[13]] = not findings
        info["scrub"] = {"terms": len(terms), "hosts": len(kit.hostnames()), "findings": findings}
        failed = sorted(k for k, ok in checks.items() if ok is not True)
        if failed:
            raise StageFailed(f"{len(failed)} check(s) failed before loading: {failed}")

        # 3. the staged loader, both formats, against binary_export.load_export
        module = load_loader(tmp / kit.LOADER)
        with torch.no_grad():
            staged = module.load(tmp, device="cpu")
            compat = module.load(tmp / kit.COMPAT, device="cpu")
            a, b = staged.state_dict(), compat.state_dict()
            diff = compare_states(a, b)
            checks[FIXED_CHECKS[5]] = not diff and len(a) > 0
            info["loader"] = {"state_keys": len(a), "bundle_vs_compat_differing": diff[:10]}
            del compat, b
            reference = binary_export.load_export(tp.VIEW, device="cpu")
            diff = compare_states(a, reference.state_dict())
            checks[FIXED_CHECKS[6]] = not diff
            info["loader"]["bundle_vs_load_export_differing"] = diff[:10]
            checks[FIXED_CHECKS[7]] = all(v.dtype == torch.float32 for v in a.values() if v.is_floating_point()) \
                and not staged.training
            del reference, a

        # 4. transcripts on CUDA under the shared GPU lock
        if gpu:
            import keab_step0 as k0
            tk.vram_lease()
            device = torch.device("cuda")
            with k0.waiting_lock("kea-b-0.1-stage", 6 * 3600.0):
                staged.to(device)
                for name in CLIP_SETS:
                    group = [c for c in clips if c["set"] == name]
                    for c, text in zip(group, module.transcribe(staged, [c["audio_filepath"] for c in group]),
                                       strict=True):
                        c["staged"] = text
                mode = binary_train.decoder_graphs_mode(staged, "staged model")
                del staged
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            checks[FIXED_CHECKS[8]] = mode == binary_train.DECODER_GRAPHS_MODE
            info["decoder_graphs_mode"] = mode
            info["cuda_device"] = torch.cuda.get_device_name(0)
            for c in clips:
                checks[clip_check(c)] = c.get("staged") is not None and c["staged"] == c["kfp"]
        info["clips"] = clips

        required = required_checks(clips)
        missing, unexpected = sorted(required - set(checks)), sorted(set(checks) - required)
        if missing or unexpected:
            raise StageFailed(f"check set mismatch: missing {missing}, unexpected {unexpected}")
        failed = sorted(k for k, ok in checks.items() if ok is not True)
        if failed:
            raise StageFailed(f"{len(failed)} check(s) failed: {failed}")
        validated = listing
        os.rename(tmp, STAGING)
        listing = kit.listing(STAGING)
        if listing != validated:
            raise StageFailed("staged files changed between validation and promotion")
        report = {"status": "PASSED", "folder": str(STAGING), **info, "finished_utc": utc(), "files": listing,
                  "total_bytes": sum(v["bytes"] for v in listing.values()),
                  "bundle_manifest": kit.bundle_manifest(repo_id, listing),
                  "bundle_sha256": kit.bundle_sha256(repo_id, listing),
                  "checks_passed": f"{len(checks)}/{len(checks)}", "checks": checks,
                  "upload_command": "./python -m hf.upload --approved-bundle-sha256 <bundle_sha256>",
                  "verify_command": "./python -m hf.verify <download dir>"}
        kit.write_json(REPORT, report)
        return True, report, REPORT
    except BaseException as error:  # noqa: BLE001 -- any failure, including SystemExit, leaves no staging folder
        for leftover in (tmp, STAGING, TMP_ROOT):
            if leftover.exists():
                shutil.rmtree(leftover)
        REPORT.unlink(missing_ok=True)
        report = {"status": "FAILED", "error": "".join(traceback.format_exception_only(error)).strip(),
                  "traceback": traceback.format_exc(), **info, "clips": clips, "checks": checks,
                  "failed_checks": sorted(k for k, ok in checks.items() if ok is not True)}
        kit.write_json(REPORT_FAILED, report)
        return False, report, REPORT_FAILED


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=kit.REPO_ID)
    args = parser.parse_args(argv)
    passed, report, path = stage(args.repo_id)
    summary = {k: report.get(k) for k in ("status", "error", "folder", "checks_passed", "failed_checks",
                                          "bundle_sha256", "total_bytes")}
    print(json.dumps(summary, indent=1))
    if not passed:
        print(f"\nSTAGING FAILED; no staging folder was left. Report: {path}", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll {len(report['checks'])} checks passed. Report: {path}\nBundle SHA-256 (the approval token): "
          f"{report['bundle_sha256']}\n{report['bundle_manifest']}Nothing was uploaded.")


if __name__ == "__main__":
    main()
