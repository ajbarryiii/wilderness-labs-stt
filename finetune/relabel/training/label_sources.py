"""Training labels by row, from one provider per clip source (stage 3 and later; b0prime plan "No online teacher").

A row's training text is read by one function, Labels.text(item_id), so everything that decides a label is in one
place and in one identity:

    StoreLabels   the stage 2 label store (training/labels.py: T*'s text under T*'s disposition mask, read through label
                  cleaning at a recorded version and repetition rejection), then an optional read-time `transform`
                  (an object with __call__(text, item_id) -> text and an `identity` dict; deterministic). A rejected
                  id stays None and is never transformed; a transform may not return None, nor empty a non-empty text.
    ClipLabels    a synthetic clip pool's labels (synthetic.mixing.ClipPool clips): row id "syn:<clip id>" -> the clip's
                  label (the version 2 plan's synthetic rows, training/replay.py PlannedReplay).
    Labels        the composite: a row id with an extra provider's prefix is that provider's, every other id is the
                  store's. identity["sha256"] is the run's labels_sha256: the store's own sha256 when there is no
                  transform and no extra provider (so it equals stage 2b's for the same store), else the SHA-256 of
                  {"version": COMPOSITE_VERSION, store sha256, transform identity, each provider's identity}.

Batcher (a training/replay.py ReplayBatcher) takes every row's text from Labels.text; a row the replay serves with
its own "label" (a synthetic row) must carry the provider's label. Cleaning versions (cleaning plan; the version
enters the store identity, hence labels_sha256): 0 = no cleaning, 1 = training/cleaning.py's committed tables with the
label run's choices artifact; any other version is refused until its code is in this checkout.
"""
from __future__ import annotations

import hashlib
import json

from training import replay as treplay

COMPOSITE_VERSION = 1
CLEANING_VERSIONS = (0, 1)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def cleaning_spec(labels_run: str, version: int):
    """The label cleaning a run reads through at `version` (module docstring); None for version 0."""
    from training import cleaning as tcleaning
    if version not in CLEANING_VERSIONS:
        raise ValueError(f"cleaning version {version} is not available in this checkout (versions {CLEANING_VERSIONS})")
    if version == 0:
        return None
    spec = tcleaning.production(labels_run)
    got = spec.identity().get("version")
    if got != version:
        raise RuntimeError(f"the cleaning spec of {labels_run} is version {got}, not {version}")
    return spec


class StoreLabels:
    def __init__(self, store, transform=None):
        self.store, self.transform = store, transform
        self.label_set = store.label_set

    def text(self, item_id: str) -> str | None:
        value = self.store.text(item_id)
        if value is None or self.transform is None:
            return value
        out = self.transform(value, item_id)
        if not isinstance(out, str) or (value != "" and out == ""):
            raise RuntimeError(f"{item_id}: the label transform returned {out!r} for a kept label")
        return out

    def transform_identity(self) -> dict | None:
        return None if self.transform is None else dict(self.transform.identity)


class ClipLabels:
    PREFIX = "syn:"

    def __init__(self, clips):
        self.clips = clips
        self._labels = {f"{self.PREFIX}{c['id']}": c["label"] for c in clips.clips}
        self.identity = {"kind": "synthetic clip pool", "clip_pool": str(clips.identity),
                         "labels_sha256": _sha(sorted(self._labels.items()))}

    def text(self, item_id: str) -> str:
        return self._labels[item_id]


class Labels:
    def __init__(self, store: StoreLabels, extra: dict | None = None):
        self.real, self.extra = store, dict(extra or {})
        for prefix in self.extra:
            if not prefix or ":" not in prefix:
                raise ValueError(f"an extra provider's prefix must end its source tag with ':' ({prefix!r})")
        self.label_set = store.label_set
        base = store.store.identity
        transform = store.transform_identity()
        extras = {p: dict(v.identity) for p, v in sorted(self.extra.items())}
        if transform is None and not extras:
            sha = base["sha256"]
        else:
            sha = _sha({"version": COMPOSITE_VERSION, "store_sha256": base["sha256"], "transform": transform,
                        "extra": extras})
        self.identity = {**base, "store_sha256": base["sha256"], "transform": transform, "extra": extras,
                         "composite_version": COMPOSITE_VERSION, "sha256": sha}

    def text(self, item_id: str) -> str | None:
        """The training text of a row (module docstring); None for a rejected real row."""
        for prefix, provider in self.extra.items():
            if item_id.startswith(prefix):
                return provider.text(item_id)
        return self.real.text(item_id)

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
            if "label" in it and it["label"] != text:
                raise RuntimeError(f"batch {b}: row {it['id']} carries a label its provider does not give")
            texts.append(text)
        if self.extra is not None:
            raise RuntimeError("extra clips come through the plan (version 2), not appended after the cut")
        batch = train.collate(items)
        batch.update(index=b, position_after={"batch": b + 1}, labels=texts, extra_rows=0)
        return batch
