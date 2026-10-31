"""A union of clip stores as one clip store (keab-prime plan "Data": the synthetic source is the runs full, pairs and
pairs-c of synthetic plan v5.2, under split v3, as one source).

ClipUnion(parts) takes [(name, store)] (synthetic.mixing.ClipPool's interface: .clips [{"id", "duration", "label", ...}],
.identity, .read(clip_id)) and answers the same interface:
    .clips      every part's clips, in part order (training/batchplan.py sorts a source's clips by id)
    .by_id      clip id -> clip
    .identity   SHA-256 of [[name, str(part identity)], ...] in part order: a union of other runs, other contents or
                another order is another source (its plan and labels identities change with it)
    .read(id)   the owning part's read (its SHA-256 check included)
    .parts      [(name, number of clips, hours, identity)]
A clip id held by two parts is refused (the synthetic runs' ids carry their run's name, so they never collide).
"""
from __future__ import annotations

import hashlib
import json


class ClipUnion:
    def __init__(self, parts: list[tuple[str, object]]):
        if not parts:
            raise ValueError("a union needs at least one clip store")
        names = [n for n, _ in parts]
        if len(set(names)) != len(names):
            raise ValueError(f"part names repeat: {names}")
        self.clips, self.by_id, self._owner = [], {}, {}
        for name, store in parts:
            for c in store.clips:
                if c["id"] in self.by_id:
                    raise ValueError(f"clip {c['id']} is in parts {self._owner[c['id']]} and {name}")
                self.by_id[c["id"]] = c
                self._owner[c["id"]] = name
                self.clips.append(c)
        self._stores = dict(parts)
        self.identity = hashlib.sha256(json.dumps([[n, str(s.identity)] for n, s in parts]).encode()).hexdigest()
        self.parts = [(n, len(s.clips), sum(float(c["duration"]) for c in s.clips) / 3600, str(s.identity))
                      for n, s in parts]
        self.verify_audio = all(getattr(s, "verify_audio", True) for _, s in parts)

    def read(self, clip_id: str):
        return self._stores[self._owner[clip_id]].read(clip_id)

    def owner(self, clip_id: str) -> str:
        return self._owner[clip_id]
