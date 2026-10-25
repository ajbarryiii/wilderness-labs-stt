"""The cluster-disjoint dev split (decoder-only plan "Dev split for selection and evaluation") and the dev
clusters (b0prime plan "Intervals").

Clusters (one per correlated group of utterances):
    LibriSpeech dev-clean / dev-other   the speaker: the first field of the utterance id
    AMI dev                             the meeting (AMI_<meeting>_<headset>_...)
    VoxPopuli dev                       the plenary session: the manifest has no speaker id, so the id's
                                        "<date>-<time>-PLENARY" prefix (the sitting; the agenda item that
                                        follows is not a session)
YODAS dev has no cluster definition: it takes no part in decisions and is not split.

Hash rule: each H4 set's clusters are ranked by SHA-256(SALT, set, cluster) (ties impossible; then by name);
the first round(FRACTION x clusters) form the selection part (about 30%), the rest the evaluation part. The split
depends only on the cluster ids, not on record order. It is written once, with its rule, the manifests' SHA-256,
the clusters, ids and per-part cluster, utterance and reference-word counts, and its own SHA-256; it is never
replaced (freeze refuses a different split), and load refuses a changed manifest or a damaged split file.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

FRACTION = 0.30
SALT = "relabel-decoder-only-dev-split-v1"
H4 = ("librispeech_dev_clean", "librispeech_dev_other", "ami_dev", "voxpopuli_dev")
PARTS = ("selection", "evaluation")
_AMI = re.compile(r"^ami:AMI_([A-Za-z0-9]+)_")
_VOX = re.compile(r"^voxpopuli:(\d{8}-\d{4}-PLENARY)-")
_LS = re.compile(r"^(?:librispeech:)?(\d+)-\d+-\d+$")


def cluster_of(set_name: str, item_id: str) -> str:
    if set_name.startswith("librispeech_dev"):
        m = _LS.match(item_id)
    elif set_name == "ami_dev":
        m = _AMI.match(item_id)
    elif set_name == "voxpopuli_dev":
        m = _VOX.match(item_id)
    else:
        raise ValueError(f"no cluster definition for {set_name}")
    if m is None:
        raise ValueError(f"{set_name}: cannot read the cluster of {item_id!r}")
    return m.group(1)


def cluster_hash(salt: str, set_name: str, cluster: str) -> str:
    return hashlib.sha256(f"{salt}\0{set_name}\0{cluster}".encode()).hexdigest()


def reference_words(text: str) -> int:
    """Normalized reference words of a scored utterance (0 for an excluded reference), as evaluate.score."""
    import evaluate  # finetune/parakeet-ternary
    norm = evaluate.normalize(text)
    return 0 if norm.strip() in evaluate.EXCLUDED_REFERENCES else len(norm.split())


def _sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _body_sha(split: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in split.items() if k != "sha256"}, sort_keys=True).encode()).hexdigest()


def build(manifests: dict[str, Path], fraction: float = FRACTION, salt: str = SALT) -> dict:
    """The split of the given H4 manifests ({set: path})."""
    out = {"rule": {"fraction": fraction, "salt": salt, "selection_clusters": "round(fraction x clusters)",
                    "hash": "clusters ranked by sha256(salt \\0 set \\0 cluster) hex, then by name; the first form the "
                            "selection part",
                    "clusters": {"librispeech": "speaker (first field of the utterance id)", "ami": "meeting",
                                 "voxpopuli": "plenary session (<date>-<time>-PLENARY prefix of the id)"}},
           "sets": {}}
    for name in sorted(manifests):
        path = Path(manifests[name])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_cluster: dict[str, list[dict]] = {}
        for r in rows:
            by_cluster.setdefault(cluster_of(name, r["id"]), []).append(r)
        ranked = sorted(by_cluster, key=lambda c: (cluster_hash(salt, name, c), c))
        n_sel = round(fraction * len(ranked))
        if not 0 < n_sel < len(ranked):
            raise RuntimeError(f"{name}: {len(ranked)} clusters cannot be split at {fraction}")
        entry = {"manifest": path.name, "manifest_sha256": _sha256_file(path)}
        for part, clusters in (("selection", ranked[:n_sel]), ("evaluation", ranked[n_sel:])):
            members = [r for c in clusters for r in by_cluster[c]]
            entry[part] = {"clusters": sorted(clusters), "cluster_count": len(clusters), "utterances": len(members),
                           "reference_words": sum(reference_words(r["text"]) for r in members),
                           "hours": sum(float(r["duration"]) for r in members) / 3600,
                           "ids": sorted(r["id"] for r in members)}
        out["sets"][name] = entry
    out["sha256"] = _body_sha(out)
    return out


def freeze(path: Path, split: dict) -> dict:
    """Write the split once; an existing file must hold exactly this split (else RuntimeError)."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        if old != split:
            raise RuntimeError(f"{path} holds another split (sha256 {old.get('sha256')}); the frozen split is never "
                               "replaced")
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(split, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return split


class Split:
    def __init__(self, data: dict):
        self.data, self.sha256 = data, data["sha256"]
        self._part = {(s, i): p for s, e in data["sets"].items() for p in PARTS for i in e[p]["ids"]}

    @property
    def sets(self) -> list[str]:
        return list(self.data["sets"])

    def part(self, set_name: str, item_id: str) -> str:
        return self._part[(set_name, item_id)]

    def ids(self, set_name: str, part: str) -> set[str]:
        return set(self.data["sets"][set_name][part]["ids"])

    def counts(self) -> dict:
        return {s: {p: {k: e[p][k] for k in ("cluster_count", "utterances", "reference_words", "hours")} for p in PARTS}
                for s, e in self.data["sets"].items()}


def load(path: Path, manifests: dict[str, Path] | None = None) -> Split:
    """The frozen split, after checking its SHA-256 and (if given) that the manifests are the ones it split."""
    data = json.loads(Path(path).read_text())
    if _body_sha(data) != data.get("sha256"):
        raise RuntimeError(f"{path}: content differs from its sha256")
    for name, mpath in (manifests or {}).items():
        if _sha256_file(Path(mpath)) != data["sets"][name]["manifest_sha256"]:
            raise RuntimeError(f"{name}: manifest {mpath} changed since the split was frozen")
    return Split(data)


def dev_manifests() -> dict[str, Path]:
    """The stage 1 H4 dev manifests (parakeet-ternary paths.MANIFESTS)."""
    import paths  # finetune/parakeet-ternary
    return {name: paths.MANIFESTS / f"{name}.jsonl" for name in H4}
