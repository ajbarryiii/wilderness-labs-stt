"""Training labels by row, from one provider per clip source (stage 3 and later; b0prime plan "No online teacher",
"Extra sources").

A row's training text is read by one function, Labels.text(item_id), so everything that decides a label is in one
place and in one identity:

    StoreLabels   the stage 2 label store (training/labels.py: T*'s text under T*'s disposition mask, read through label
                  cleaning at a recorded version and repetition rejection), then the read-time reject list (`reject`,
                  a RejectList: a listed id has no label, as a rejected one), then an optional read-time `transform`
                  (an object with __call__(text, item_id) -> text and an `identity` dict; deterministic). A rejected
                  id stays None and is never transformed; a transform may not return None, nor empty a non-empty text.
                  An id without a store record is a KeyError, listed or not.
    ClipLabels    a clip store's labels (synthetic.mixing.ClipPool clips: the stage 2c synthetic store, or another
                  source's store in the same format): row id "<prefix><clip id>" -> the clip's "label" as stored (raw(),
                  what the replay serves with the row), or with a `transform` (__call__(label, clip id) -> text and an
                  `identity`; deterministic, memoized) that label transformed (text()). Layer1 is the read-speech
                  source's transform: cleaning layer 1 alone at the run's cleaning version (b0prime plan v4 "Labels":
                  training/cleaning.py with no choices artifact, so layer 2 leaves every span as written; layer 0 needs
                  a T* label and its B0 donor and never applies; the label set is the source's name).
    Labels        the composite: a row id with an extra provider's prefix is that provider's, every other id is the
                  store's. identity["sha256"] is the run's labels_sha256: the store's own sha256 when there is no
                  reject list, transform or extra provider (so it equals stage 2b's for the same store), else the
                  SHA-256 of {"version": COMPOSITE_VERSION, store sha256, the reject list's identity, the transform's,
                  each provider's}.
    RejectList    a hashed list of ids rejected at read time (b0prime plan v4: the ted-test-text containment rule's
                  main-pool rejects), pool/tedreject.py's artifact directory: rejected_ids.txt (sorted ids, one per
                  line; empty if none) and reject.json (name REJECT_NAME, version, ids_sha256 = SHA-256 of
                  rejected_ids.txt, inputs.complete, counts.rejected_ids). Refused: another name, a list whose SHA-256
                  differs, a partial (smoke) build, a count that differs. Its identity {name, version, ids_sha256,
                  reject_sha256 (of reject.json), rejected_ids} enters labels_sha256. An empty list is valid.

Batcher (a training/replay.py ReplayBatcher) takes every row's text from Labels.text; a row the replay serves with
its own "label" (an extra row) must carry the provider's stored label (raw). Cleaning versions (cleaning plan; the version enters
the store identity, hence labels_sha256): 0 = no cleaning; 1 and 2 = training/cleaning.py's committed tables with the
label run's choices artifact of that version (2: spelled-out acronyms and layer 0's punctuation transfer into T*'s
labels, cleaning plan v6; stage 3 reads it; CLEANING_VERSIONS are stage 3's); 3 = cleaning plan v7 (spellings keep their letters, dose units; the label
run's acronym lexicon, in the choices artifact's directory; stages 4-5 read it). Layer1 at version 3 reads that lexicon
too (layer 1's spelling rule; still no choices artifact and no donor).

A pair row (training/replay.py: an AMI distant-microphone pair, served with "teacher_audio" and no label) takes its
text from Labels.text like a real row (training/ami_store.PairLabels: its IHM id's label); the Batcher passes the
teacher recordings on as batch["teacher_audio"] = {row: float32 samples} (only when the batch has a pair row).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import replay as treplay

COMPOSITE_VERSION = 1
CLEANING_VERSIONS = (0, 1, 2)       # stage 3's; stage 4 (train_keab.py) reads version 3 (cleaning_spec's versions)
REJECT_NAME = "talk-text-reject"


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def cleaning_spec(labels_run: str, version: int, versions: tuple[int, ...] = CLEANING_VERSIONS):
    """The label cleaning a run reads through at `version` (module docstring); None for version 0. `versions`: the
    versions the calling stage reads (stage 3: CLEANING_VERSIONS; stage 4: (3,))."""
    from training import cleaning as tcleaning
    if version not in versions:
        raise ValueError(f"cleaning version {version} is not available in this checkout (versions {versions})")
    if version == 0:
        return None
    spec = tcleaning.production(labels_run, version)
    got = spec.identity().get("version")
    if got != version:
        raise RuntimeError(f"the cleaning spec of {labels_run} is version {got}, not {version}")
    return spec


class RejectList:
    def __init__(self, ids, identity: dict):
        self.ids, self.identity = frozenset(ids), identity

    @classmethod
    def load(cls, directory: Path) -> "RejectList":
        """The reject list artifact at `directory` (module docstring); RuntimeError for a damaged or partial one."""
        directory = Path(directory)
        raw = (directory / "reject.json").read_bytes()
        body = json.loads(raw)
        data = (directory / "rejected_ids.txt").read_bytes()
        if body.get("name") != REJECT_NAME or hashlib.sha256(data).hexdigest() != body.get("ids_sha256"):
            raise RuntimeError(f"{directory}: rejected_ids.txt does not match reject.json (or not a {REJECT_NAME})")
        if not (body.get("inputs") or {}).get("complete"):
            raise RuntimeError(f"{directory}: built from a partial label run (a smoke); not usable for training")
        ids = frozenset(line for line in data.decode().splitlines() if line)
        count = (body.get("counts") or {}).get("rejected_ids")
        if len(ids) != count:
            raise RuntimeError(f"{directory}: {len(ids)} distinct ids, reject.json says {count}")
        return cls(ids, {"name": REJECT_NAME, "version": body.get("version"), "ids_sha256": body["ids_sha256"],
                         "reject_sha256": hashlib.sha256(raw).hexdigest(), "rejected_ids": len(ids)})


class StoreLabels:
    def __init__(self, store, transform=None, reject: RejectList | None = None):
        self.store, self.transform, self.reject = store, transform, reject
        self.label_set = store.label_set

    def text(self, item_id: str) -> str | None:
        value = self.store.text(item_id)
        if value is None or (self.reject is not None and item_id in self.reject.ids):
            return None
        if self.transform is None:
            return value
        out = self.transform(value, item_id)
        if not isinstance(out, str) or (value != "" and out == ""):
            raise RuntimeError(f"{item_id}: the label transform returned {out!r} for a kept label")
        return out

    def transform_identity(self) -> dict | None:
        return None if self.transform is None else dict(self.transform.identity)

    def reject_counts(self) -> dict | None:
        """The reject list's ids that are kept records of the store (they lose their label), or None."""
        if self.reject is None:
            return None
        known = [i for i in self.reject.ids if i in self.store]
        return {"listed": len(self.reject.ids), "in_store": len(known),
                "kept_in_store": sum(self.store.text(i) is not None for i in known)}


class ClipLabels:
    def __init__(self, clips, prefix: str = "syn:", transform=None):
        self.clips, self.prefix, self.transform = clips, prefix, transform
        self._labels = {f"{prefix}{c['id']}": c["label"] for c in clips.clips}
        self._done: dict[str, str] = {}
        self.identity = {"kind": "clip store", "prefix": prefix, "clip_pool": str(clips.identity),
                         "labels_sha256": _sha(sorted(self._labels.items()))}
        if transform is not None:
            self.identity["transform"] = dict(transform.identity)

    def raw(self, item_id: str) -> str:
        """The clip's label as stored (what the replay serves with the row)."""
        return self._labels[item_id]

    def text(self, item_id: str) -> str:
        """The row's training text: the stored label, through the transform if there is one."""
        value = self._labels[item_id]
        if self.transform is None:
            return value
        out = self._done.get(item_id)
        if out is None:
            out = self.transform(value, item_id[len(self.prefix):])
            if not isinstance(out, str) or (value != "" and out == ""):
                raise RuntimeError(f"{item_id}: the label transform returned {out!r}")
            self._done[item_id] = out
        return out


class Layer1:
    """Cleaning layer 1 alone at `version` as a label transform (module docstring, ClipLabels); version 3 reads the
    acronym lexicon at lexicon_dir (the label run's, cleaning.production_dir(run, 3))."""

    def __init__(self, version: int, label_set: str, lexicon_dir: Path | None = None):
        from training import cleaning
        if version not in cleaning.VERSIONS:
            raise ValueError(f"cleaning version {version} has no layer 1 here (versions {cleaning.VERSIONS})")
        if (version >= 3) != (lexicon_dir is not None):
            raise ValueError(f"cleaning version {version}'s layer 1 {'needs' if version >= 3 else 'reads no'} "
                             "acronym lexicon")
        self.spec = cleaning.Cleaning.load(version=version, lexicon_dir=lexicon_dir)
        if self.spec.choices is not None or self.spec.needs_donor(label_set):
            raise RuntimeError("layer 1 alone reads no choices artifact and no B0 donor")
        self.label_set = label_set
        self.identity = self.spec.identity() | {"layers": [1], "label_set": label_set}

    def __call__(self, text: str, clip_id: str) -> str:
        return self.spec.clean(text, clip_id, self.label_set)


def layer1(version: int, label_set: str, lexicon_dir: Path | None = None) -> Layer1 | None:
    """Layer1 at the run's cleaning version; None at version 0 (no cleaning)."""
    return None if version == 0 else Layer1(version, label_set, lexicon_dir)


class Labels:
    def __init__(self, store: StoreLabels, extra: dict | None = None):
        self.real, self.extra = store, dict(extra or {})
        for prefix, provider in self.extra.items():
            if not prefix or not prefix.endswith(":") or getattr(provider, "prefix", prefix) != prefix:
                raise ValueError(f"an extra provider's prefix must end with ':' and be its own ({prefix!r})")
        self.label_set = store.label_set
        base = store.store.identity
        transform = store.transform_identity()
        reject = None if store.reject is None else dict(store.reject.identity)
        extras = {p: dict(v.identity) for p, v in sorted(self.extra.items())}
        if transform is None and reject is None and not extras:
            sha = base["sha256"]
        else:
            sha = _sha({"version": COMPOSITE_VERSION, "store_sha256": base["sha256"], "reject": reject,
                        "transform": transform, "extra": extras})
        self.identity = {**base, "store_sha256": base["sha256"], "reject": reject, "transform": transform,
                         "extra": extras, "composite_version": COMPOSITE_VERSION, "sha256": sha}

    def text(self, item_id: str) -> str | None:
        """The training text of a row (module docstring); None for a rejected real row."""
        for prefix, provider in self.extra.items():
            if item_id.startswith(prefix):
                return provider.text(item_id)
        return self.real.text(item_id)

    def stored(self, item_id: str) -> str:
        """An extra row's label as its store holds it (what the replay serves); KeyError for a real row."""
        for prefix, provider in self.extra.items():
            if item_id.startswith(prefix):
                return provider.raw(item_id)
        raise KeyError(item_id)

    def kept(self, item_id: str) -> bool:
        return self.text(item_id) is not None


class Batcher(treplay.ReplayBatcher):
    """ReplayBatcher whose rows' texts all come from Labels.text (a row served with its own label must agree)."""

    def assemble(self, b: int) -> dict:
        import train  # finetune/parakeet-ternary
        items = self.replay.read(b, self.pool)
        texts = []
        for it in items:
            text = self.labels.text(it["id"])
            if "label" in it and it["label"] != self.labels.stored(it["id"]):
                raise RuntimeError(f"batch {b}: row {it['id']} carries a label its provider does not hold")
            texts.append(text)
        if self.extra is not None:
            raise RuntimeError("extra clips come through the plan (version 2), not appended after the cut")
        batch = train.collate(items)
        batch.update(index=b, position_after={"batch": b + 1}, labels=texts, extra_rows=0)
        teacher = {r: it["teacher_audio"] for r, it in enumerate(items) if "teacher_audio" in it}
        if teacher:
            batch["teacher_audio"] = teacher
        return batch
