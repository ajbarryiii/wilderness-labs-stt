"""Fixture world for the test_training_* and test_decoder_* contract tests (no tests here).

On top of labeling_fixtures.World (a temporary $RELABEL_ROOT with local "Hub" files): C1 replays the pinned
stream into real shards (labeling.replay), and C2 labels them into real label records (labeling.c2, with fake
engines): T* outputs the clip's reference, except "" (rejected: empty_label) for the ids in `reject`; B0 outputs
"b0 " + the reference. Every id of batch REJECTED_BATCH is rejected, so that batch is fully rejected.

With hallucinate=True, T* also hallucinates a repetition that passes the filters (one word, as the real ones):
on the first id of batch HALLUCINATED_BATCH the reference + HALLUCINATED (10 repeats of the token "r" under the
students' tokenizer), on its second the reference + NEAR_MISS (9 repeats). The ids are `hallucinated`, `near_miss`.
"""
from __future__ import annotations

from pathlib import Path

from labeling import c2, config, replay, shards
from labeling.batches import iter_batches
from labeling_fixtures import World

TOTAL, PER_SHARD = 14, 4
REJECTED_BATCH = 3
HALLUCINATED_BATCH = 5
HALLUCINATED = " Wu" + "r" * 11          # "▁W ur" + "r" x 10
NEAR_MISS = " Wu" + "r" * 10             # "▁W ur" + "r" x 9


class ScriptedEngine:
    """A C2 engine whose output is a function of the clip's id (found through its samples)."""
    kind = "fake"

    def __init__(self, rows_by_pcm: dict[str, dict], text_of):
        self.rows, self.text_of = rows_by_pcm, text_of

    def load(self):
        pass

    def plan(self, durations):
        return [list(range(i, min(len(durations), i + 3))) for i in range(0, len(durations), 3)]

    def run(self, clips):
        return [self.text_of(self.rows[shards.pcm_sha256(shards.to_int16(c))]) for c in clips]

    def info(self):
        return {"engine": "scripted"}


class LabeledWorld:
    """World + C1 shards (TOTAL batches) + one C2 run (labels/<T*>/<run>) over every C1 shard."""

    def __init__(self, root: Path, run: str = "fixture", hallucinate: bool = False):
        self.root = Path(root)
        self.world = World(self.root)
        self.pin = self.world.pinned(workers=1, batch_seconds=10.0)
        self.pool = self.root / "pool-audio" / config.config_id(self.pin)
        replay.replay(self.pin, self.pool, total_batches=TOTAL, shard_batches=PER_SHARD,
                      stream_factory=self.stream)
        self.journal = [r for k in shards.committed(self.pool) for r in shards.read_journal(shards.paths(self.pool, k)["journal"])]
        self.reject = {r["id"] for r in self.journal if r["batch"] == REJECTED_BATCH}
        self.hallucinated = self.near_miss = None
        suffix = {}
        if hallucinate:
            self.hallucinated, self.near_miss = self.batch_ids(HALLUCINATED_BATCH)[:2]
            suffix = {self.hallucinated: HALLUCINATED, self.near_miss: NEAR_MISS}
        by_pcm = {}
        for k in shards.committed(self.pool):
            for r in shards.read_audio(shards.paths(self.pool, k)["audio"]):
                by_pcm[r["pcm_sha256"]] = r
        self.labels_dir = self.root / "labels" / "granite-speech-5.0-2b" / run
        c2.init_run(self.labels_dir, pool_dir=self.pool, engine={"kind": "fake"}, group_shards=2,
                    b0_batch_seconds=60.0, declared_vram_gb=None)
        teacher = ScriptedEngine(by_pcm, lambda r: ("" if r["id"] in self.reject else
                                                    (r["reference"] or "text") + suffix.get(r["id"], ""), False))
        b0 = ScriptedEngine(by_pcm, lambda r: "b0 " + (r["reference"] or "text"))
        c2.teacher_step(self.labels_dir, engine=teacher)
        c2.b0_step(self.labels_dir, labeler=b0)
        c2.merge_step(self.labels_dir, coverage=lambda t: True)
        self.records = {}
        import pyarrow.parquet as pq
        for p in sorted((self.labels_dir / "shards").glob("*.labels.parquet")):
            for rec in pq.read_table(p).to_pylist():
                self.records[rec["id"]] = rec

    def stream(self):
        return config.make_stream(self.pin, 0)

    def stream_batches(self, n: int) -> list[list[dict]]:
        """The first n consumed batches of a fresh stream (labeling.batches = train.Batcher's sequence)."""
        s = self.stream()
        try:
            it = iter_batches(s, self.pin["batch"]["batch_seconds"], self.pin["batch"]["seed"])
            return [next(it)[0] for _ in range(n)]
        finally:
            s.close()

    def batch_ids(self, b: int) -> list[str]:
        return [r["id"] for r in self.journal if r["batch"] == b]
