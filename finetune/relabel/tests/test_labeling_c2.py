"""Contract tests: C2, the labeling pass over C1's shards (groups, order, records, resume, lock, engines)."""
from __future__ import annotations

import fcntl
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import evaluate  # finetune/parakeet-ternary
import relabel_paths
from labeling import c2, config, engines, replay, shards
from labeling_fixtures import World, relabel_root

TOTAL, PER_SHARD = 14, 4


class RefTeacher:
    """Returns each clip's reference (looked up by its samples); records the durations it was given."""
    kind = "fake"

    def __init__(self, refs: dict[str, str], batch: int = 3):
        self.refs, self.batch, self.seen = refs, batch, []

    def load(self):
        pass

    def plan(self, durations):
        return [list(range(i, min(len(durations), i + self.batch))) for i in range(0, len(durations), self.batch)]

    def run(self, clips):
        out = []
        for c in clips:
            self.seen.append(len(c) / 16000)
            out.append((self.refs[shards.pcm_sha256(shards.to_int16(c))], False))
        return out

    def info(self):
        return {"engine": "fake-ref"}


class RefB0(RefTeacher):
    def run(self, clips):
        return [t for t, _ in super().run(clips)]


def fake_signals(text: str) -> dict:
    """Deterministic signals for a label: one token per word plus an end token, log-probabilities <= 0, the
    runner-up below the chosen token (ties included), values not representable in float16."""
    n = len(text.split()) + 1
    rng = np.random.default_rng([sum(map(ord, text)) % 9973, n])
    chosen = -rng.exponential(0.3, n)
    chosen[0] = 0.0
    runner = chosen - rng.exponential(2.0, n)
    runner[-1] = chosen[-1]                    # a tie
    return {"token_ids": [int(x) for x in rng.integers(0, 100_000, n)], "logprob": [float(x) for x in chosen],
            "runner_up": [float(x) for x in runner]}


class SigTeacher(RefTeacher):
    """RefTeacher that also returns confidence signals (as an engine with confidence on does)."""

    def __init__(self, refs, batch=3, empty=()):
        super().__init__(refs, batch)
        self.empty, self.given = set(empty), {}

    def run(self, clips):
        out = []
        for c in clips:
            h = shards.pcm_sha256(shards.to_int16(c))
            text = "" if h in self.empty else self.refs[h]
            s = fake_signals(text)
            self.given[h] = s
            out.append((text, False, s))
        return out


def table(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


class C2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.ctx = relabel_root(cls.root)
        cls.ctx.__enter__()
        cls.world = World(cls.root)
        cls.pin = cls.world.pinned(workers=1, batch_seconds=10.0)
        cls.pool = cls.root / "pool-audio"
        replay.replay(cls.pin, cls.pool, total_batches=TOTAL, shard_batches=PER_SHARD,
                      stream_factory=lambda: config.make_stream(cls.pin, 0))
        cls.rows = {}
        for k in shards.committed(cls.pool):
            for r in shards.read_audio(shards.paths(cls.pool, k)["audio"]):
                cls.rows[r["pcm_sha256"]] = r
        cls.refs = {h: (r["reference"] or "") for h, r in cls.rows.items()}

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)
        cls.tmp.cleanup()

    def init(self, name, group_shards=2, engine=None):
        d = self.root / "runs" / name
        c2.init_run(d, pool_dir=self.pool, engine=engine or {"kind": "fake"}, group_shards=group_shards,
                    b0_batch_seconds=60.0, declared_vram_gb=None)
        return d

    def full_run(self, d, teacher=None, stop=None):
        teacher = teacher or RefTeacher(self.refs)
        c2.teacher_step(d, engine=teacher, stop_after_groups=stop)
        c2.b0_step(d, labeler=RefB0(self.refs))
        c2.merge_step(d, coverage=lambda t: True)
        return teacher

    def test_groups_are_contiguous_and_the_last_short_group_waits_for_completion(self):
        self.assertEqual(c2.groups(self.pool, 2), [[0, 1], [2, 3]])
        self.assertEqual(c2.groups(self.pool, 3), [[0, 1, 2], [3]])
        partial = self.root / "partial"
        replay.replay(self.pin, partial, total_batches=TOTAL, shard_batches=PER_SHARD, stop_after_shards=3,
                      stream_factory=lambda: config.make_stream(self.pin, 0))
        self.assertEqual(c2.groups(partial, 2), [[0, 1]])
        self.assertEqual(c2.groups(partial, 3), [[0, 1, 2]])

    def test_every_distinct_id_gets_exactly_one_record_per_c1_shard(self):
        d = self.init("e2e")
        self.full_run(d)
        self.assertTrue(c2.finish(d)["complete"])
        for k in shards.committed(self.pool):
            ids = [r["id"] for r in shards.read_audio(shards.paths(self.pool, k)["audio"], columns=("id",))]
            recs = table(d / "shards" / f"{shards.shard_name(k)}.labels.parquet")
            self.assertEqual(sorted(r["id"] for r in recs), sorted(ids))
            for r in recs:
                self.assertTrue(set(c2.LABEL_COLUMNS) <= set(r))
                self.assertEqual(r["teacher"], "granite-speech-5.0-2b")
                self.assertTrue(r["teacher_revision"])
                self.assertIn(r["disposition"], ("kept", "rejected"))
                self.assertEqual(r["disposition"] == "kept", not r["reasons"])
        m = json.loads((d / "manifest.json").read_text())
        self.assertEqual(m["records"], len(self.rows))

    def test_each_group_is_labeled_in_descending_duration_order(self):
        d = self.init("order", group_shards=2)
        t = self.full_run(d)
        sizes = [sum(1 for _ in shards.iter_audio(shards.paths(self.pool, k)["audio"], columns=("id",)))
                 for k in shards.committed(self.pool)]
        first = sizes[0] + sizes[1]
        for part in (t.seen[:first], t.seen[first:]):
            self.assertEqual(part, sorted(part, reverse=True))
        self.assertEqual(len(t.seen), len(self.rows))

    def test_interrupted_and_resumed_gives_identical_labels(self):
        a = self.init("straight")
        self.full_run(a)
        b = self.init("resumed")
        c2.teacher_step(b, engine=RefTeacher(self.refs), stop_after_groups=1)
        self.assertEqual(len(list((b / "work" / "teacher").glob("*.parquet"))), 1)
        self.full_run(b)
        for p in sorted((a / "shards").glob("*.labels.parquet")):
            self.assertEqual(table(p), table(b / "shards" / p.name))

    def test_run_settings_are_pinned(self):
        d = self.init("pinned")
        self.init("pinned")
        with self.assertRaises(RuntimeError):
            self.init("pinned", group_shards=3)

    def test_vllm_memory_fraction_cannot_exceed_the_declared_vram(self):
        with self.assertRaises(ValueError):
            c2.init_run(self.root / "runs" / "v", pool_dir=self.pool, group_shards=2, b0_batch_seconds=60.0,
                        engine={"kind": "vllm", "max_num_seqs": 64, "gpu_memory_utilization": 0.5,
                                "total_vram_gb": 32.0}, declared_vram_gb=14.0)
        c2.init_run(self.root / "runs" / "v2", pool_dir=self.pool, group_shards=2, b0_batch_seconds=60.0,
                    engine={"kind": "vllm", "max_num_seqs": 64, "gpu_memory_utilization": 0.4,
                            "total_vram_gb": 32.0}, declared_vram_gb=14.8)

    def test_merge_refuses_a_work_file_that_differs_from_its_meta(self):
        d = self.init("tamper")
        c2.teacher_step(d, engine=RefTeacher(self.refs))
        c2.b0_step(d, labeler=RefB0(self.refs))
        p = d / "work" / "teacher" / "00000.parquet"
        rows = table(p)
        rows[0]["raw"] = "changed"
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), p)
        with self.assertRaises(RuntimeError):
            c2.merge_step(d, coverage=lambda t: True)

    def test_chunk_holds_the_gpu_lock_only_for_its_duration(self):
        d = self.init("chunk")
        lock = self.root / "gpu.lock"
        held = []

        def runner(cmd, cwd=None):
            with open(lock, "a+") as h:
                try:
                    fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held.append(False)
                    fcntl.flock(h, fcntl.LOCK_UN)
                except BlockingIOError:
                    held.append(True)
            step = cmd[cmd.index("labeling.c2") + 1]
            if step == "teacher":
                c2.teacher_step(d, engine=RefTeacher(self.refs))
            else:
                c2.b0_step(d, labeler=RefB0(self.refs))
            return types.SimpleNamespace(returncode=0)

        with mock.patch.object(relabel_paths, "GPU_LOCK", lock), \
                mock.patch("labeling.filters.Coverage", lambda _p: (lambda t: True)), \
                mock.patch("labeling.filters.b0_tokenizer_file", lambda _d: None):
            st = c2.chunk(d, budget_minutes=5, lock_wait=5, runner=runner)
        self.assertEqual(held, [True, True])
        with open(lock, "a+") as h:
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)   # released afterwards
        self.assertTrue(st["complete"])

    def test_teacher_runs_in_the_vllm_environment_and_b0_in_the_shared_runtime(self):
        run = {"engine": {"kind": "vllm"}}
        self.assertTrue(c2.step_command(run, "teacher", Path("r"))[0].endswith("labeling/vllm_env.sh"))
        self.assertTrue(c2.step_command(run, "b0", Path("r"))[0].endswith("/python"))
        self.assertTrue(c2.step_command({"engine": {"kind": "transformers"}}, "teacher", Path("r"))[0].endswith("/python"))

    # --- confidence signals (labeling plan "Confidence signals", 2026-10-24) -----------------------------

    SIG = ("t_token_ids", "t_logprob", "t_runner_up")

    def signal_run(self, name, empty=(), stop=None, teacher=None):
        d = self.init(name, engine={"kind": "fake", "confidence": True})
        teacher = teacher or SigTeacher(self.refs, empty=empty)
        if stop is not None:
            c2.teacher_step(d, engine=SigTeacher(self.refs, empty=empty), stop_after_groups=stop)
        c2.teacher_step(d, engine=teacher)
        c2.b0_step(d, labeler=RefB0(self.refs))
        c2.merge_step(d, coverage=lambda t: True)
        return d, teacher

    def records(self, d):
        return [r for p in sorted((d / "shards").glob("*.labels.parquet")) for r in table(p)]

    def test_signals_are_stored_for_every_id_including_rejected(self):
        empty = set(sorted(self.rows)[:3])               # empty labels: rejected
        d, t = self.signal_run("sig", empty=empty)
        self.assertTrue(c2.finish(d)["complete"])
        recs = self.records(d)
        self.assertEqual(len(recs), len(self.rows))
        by_id = {r["id"]: r for r in self.rows.values()}
        rejected = 0
        for r in recs:
            self.assertTrue(set(c2.LABEL_COLUMNS) <= set(r))
            ids, lp, ru = (r[k] for k in self.SIG)
            self.assertTrue(ids, r["id"])
            self.assertEqual(len(ids), len(lp))
            self.assertEqual(len(ids), len(ru))
            self.assertTrue(all(x <= 0 for x in lp))
            self.assertTrue(all(b <= a for a, b in zip(lp, ru)))
            given = t.given[by_id[r["id"]]["pcm_sha256"]]
            self.assertEqual(ids, given["token_ids"])
            np.testing.assert_array_equal(np.array(lp, np.float16), np.array(given["logprob"], np.float16))
            np.testing.assert_array_equal(np.array(ru, np.float16), np.array(given["runner_up"], np.float16))
            self.assertEqual(r["b0_norm"], evaluate.normalize(r["b0_raw"]).strip())
            rejected += r["disposition"] == "rejected"
        self.assertGreaterEqual(rejected, len(empty))
        import pyarrow.parquet as pq
        schema = pq.read_schema(sorted((d / "shards").glob("*.labels.parquet"))[0])
        self.assertEqual(str(schema.field("t_token_ids").type), "list<element: int32>")
        self.assertEqual(str(schema.field("t_logprob").type), "list<element: halffloat>")
        self.assertEqual(str(schema.field("t_runner_up").type), "list<element: halffloat>")
        m = json.loads((d / "manifest.json").read_text())
        self.assertEqual(m["signals"]["tokens"], sum(len(r["t_token_ids"]) for r in recs))
        self.assertGreater(m["signals"]["bytes"], 0)
        self.assertGreater(m["signals"]["bytes_per_audio_hour"], 0)

    def test_labels_and_dispositions_are_identical_with_signals_on_and_off(self):
        empty = set(sorted(self.rows)[:2])
        on, _ = self.signal_run("sig-on", empty=empty)
        off = self.init("sig-off")

        class Off(SigTeacher):
            def run(self, clips):
                return [(t, h) for t, h, _ in super().run(clips)]

        c2.teacher_step(off, engine=Off(self.refs, empty=empty))
        c2.b0_step(off, labeler=RefB0(self.refs))
        c2.merge_step(off, coverage=lambda t: True)
        a, b = self.records(on), self.records(off)
        self.assertEqual(len(a), len(b))
        strip = lambda r: {k: v for k, v in r.items() if k not in self.SIG}
        self.assertEqual([strip(r) for r in a], [strip(r) for r in b])
        self.assertTrue(all(r[k] is None for r in b for k in self.SIG))
        self.assertTrue(any(r["disposition"] == "rejected" for r in a))

    def test_interrupted_and_resumed_reproduces_the_signals(self):
        a, _ = self.signal_run("sig-straight")
        b, _ = self.signal_run("sig-resumed", stop=1)
        for p in sorted((a / "shards").glob("*.labels.parquet")):
            self.assertEqual(table(p), table(b / "shards" / p.name))

    def test_a_run_with_signals_on_refuses_outputs_without_them(self):
        d = self.init("sig-missing", engine={"kind": "fake", "confidence": True})
        with self.assertRaises(RuntimeError):
            c2.teacher_step(d, engine=RefTeacher(self.refs))
        bad = SigTeacher(self.refs)
        orig = bad.run
        bad.run = lambda clips: [(t, h, s | {"runner_up": s["runner_up"][:-1]}) for t, h, s in orig(clips)]
        with self.assertRaises(RuntimeError):
            c2.teacher_step(d, engine=bad)
        d2 = self.init("sig-unwanted")
        with self.assertRaises(RuntimeError):
            c2.teacher_step(d2, engine=SigTeacher(self.refs))

    def test_fake_engine_honours_the_confidence_setting(self):
        clips = [np.zeros(16000, np.float32), np.zeros(8000, np.float32)]
        on = c2.FakeTeacher({"kind": "fake", "confidence": True}).run(clips)
        off = c2.FakeTeacher({"kind": "fake"}).run(clips)
        self.assertEqual([o[:2] for o in on], [o[:2] for o in off])
        self.assertTrue(all(len(o) == 2 or o[2] is None for o in off))
        for _, _, s in on:
            self.assertEqual(len(s["token_ids"]), len(s["logprob"]))
            self.assertTrue(all(b <= a <= 0 for a, b in zip(s["logprob"], s["runner_up"])))

    def test_label_rows_refuses_a_plan_that_misses_clips(self):
        t = RefTeacher(self.refs)
        t.plan = lambda durations: [[0]]
        rows = list(self.rows.values())[:3]
        with self.assertRaises(RuntimeError):
            c2.label_rows(t, rows)


def teachers_max() -> int:
    import teachers
    return teachers.MAX_NEW_TOKENS


def tiny_granite():
    """A tiny randomly initialized GraniteSpeechForConditionalGeneration (the class T* loads), seeded."""
    import torch
    from transformers import GraniteSpeechConfig, GraniteSpeechForConditionalGeneration
    torch.manual_seed(0)
    cfg = GraniteSpeechConfig(
        encoder_config=dict(input_dim=160, num_layers=1, hidden_dim=32, feedforward_mult=2, num_heads=2,
                            dim_head=16, output_dim=8, context_size=20, max_pos_emb=64, conv_kernel_size=3),
        projector_config=dict(model_type="blip_2_qformer", hidden_size=32, num_hidden_layers=1,
                              num_attention_heads=2, intermediate_size=64, encoder_hidden_size=32,
                              cross_attention_frequency=1),
        text_config=dict(model_type="granite", hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, intermediate_size=64, vocab_size=128),
        audio_token_index=127, window_size=4, downsample_rate=2, has_lora_adapter=False)
    return GraniteSpeechForConditionalGeneration(cfg).eval()


class TinyProcessor:
    """Stands in for T*'s processor: the chat prompt with 8 audio tokens, features taken from the clip's samples."""

    def __call__(self, prompts, wavs, device=None, return_tensors=None):
        import torch
        from transformers import BatchFeature
        ids = torch.tensor([[1, 2] + [127] * 8 + [3]] * len(wavs))
        feats = torch.stack([w[:16 * 160].reshape(16, 160) * 8 for w in wavs])
        return BatchFeature({"input_ids": ids, "input_features": feats, "attention_mask": torch.ones_like(ids)})


class TinyTokenizer:
    def __init__(self, special):
        self.special = set(special)

    def batch_decode(self, rows, add_special_tokens=False, skip_special_tokens=True):
        return [" ".join(f"t{int(t)}" for t in row.tolist() if int(t) not in self.special) for row in rows]


class Engines(unittest.TestCase):
    def tiny_teacher(self, eos=None, pad=0):
        """teachers.GraniteSpeech (the stage 1 decoder's transcribe) around the tiny model, in bf16 on the CPU."""
        import torch
        import teachers
        t = teachers.GraniteSpeech()
        t.device = torch.device("cpu")
        t.model = tiny_granite()
        torch.manual_seed(1)
        with torch.no_grad():   # wide weights: the default init's tiny weights repeat one token forever
            for prm in t.model.parameters():
                prm.normal_(0, 0.5)
        t.model = t.model.to(torch.bfloat16)
        t.model.generation_config.eos_token_id = eos
        t.model.generation_config.pad_token_id = pad
        t.processor = TinyProcessor()
        t.eos_ids = {eos} if eos is not None else set()
        t.tokenizer = TinyTokenizer(t.eos_ids | {pad})
        t.chat_prompt = "p"
        return t

    def clips(self, n=3):
        rng = np.random.default_rng(7)
        return [rng.uniform(-1, 1, 3000).astype(np.float32) for _ in range(n)]

    def test_transformers_signals_leave_greedy_unchanged_and_match_the_logits(self):
        """TransformersTeacher with and without the signals: identical texts and cap flags; the signals cover the
        generated tokens up to and including the first end token, and equal the log-softmax of the step's float32
        logits (from generate's own output_logits) at the chosen token and its best competitor."""
        import torch
        clips = self.clips()
        probe = self.tiny_teacher()
        # an end token that row 0 emits at step 4 or later, so that rows stop at different lengths
        out = {}
        probe.transcribe(clips, out=out)
        row0 = out["new"][0].tolist()
        eos = next(t for t in row0[4:] if t not in row0[:4])
        teacher = self.tiny_teacher(eos=eos, pad=0 if eos else 1)
        on = engines.TransformersTeacher(8, 60.0, device="cpu", confidence=True)
        off = engines.TransformersTeacher(8, 60.0, device="cpu")
        on.teacher = off.teacher = teacher
        a, b = on.run(clips), off.run(clips)
        self.assertEqual([x[:2] for x in a], [x[:2] for x in b])
        self.assertTrue(all(len(x) == 2 or x[2] is None for x in b))
        # reference: generate with output_logits on the same inputs
        wavs = [torch.from_numpy(c) for c in clips]
        inputs = teacher.processor(["p"] * 3, wavs)
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
        with torch.inference_mode():
            g = teacher.model.generate(**inputs, max_new_tokens=teachers_max(), do_sample=False, num_beams=1,
                                       output_logits=True, return_dict_in_generate=True)
        new = g.sequences[:, inputs["input_ids"].shape[-1]:]
        lp = torch.log_softmax(torch.stack(g.logits, 1).float(), -1)
        lengths = []
        for row, (text, hit, s) in enumerate(a):
            ids = new[row].tolist()
            n = ids.index(eos) + 1 if eos in ids else len(ids)
            lengths.append(n)
            self.assertEqual(s["token_ids"], ids[:n])
            self.assertEqual(hit, eos not in ids and len(ids) >= teachers_max())
            chosen = lp[row, torch.arange(n), torch.tensor(ids[:n])]
            other = lp[row, :n].clone()
            other[torch.arange(n), torch.tensor(ids[:n])] = -float("inf")
            np.testing.assert_allclose(s["logprob"], chosen.numpy(), atol=1e-5)
            np.testing.assert_allclose(s["runner_up"], other.max(-1).values.numpy(), atol=1e-5)
            self.assertTrue(all(y <= x <= 0 for x, y in zip(s["logprob"], s["runner_up"])))
        self.assertLess(lengths[0], teachers_max())
        self.assertGreater(len(set(lengths)), 1, "rows should stop at different steps")

    def test_vllm_signals_from_request_outputs(self):
        """engines.vllm_signals on vLLM-shaped outputs (per token a dict of token id -> Logprob, top-2 plus the sampled
        token): the chosen token's log-probability and the best other one, ties included."""
        L = lambda v: types.SimpleNamespace(logprob=v, rank=None, decoded_token=None)
        ids = [5, 9, 2]
        lps = [{5: L(-0.1), 7: L(-2.5)},
               {9: L(-1.0), 3: L(-1.0)},                 # a tie
               {4: L(-0.2), 8: L(-0.9), 2: L(-0.2)}]     # chosen tied with the top-1 (3 entries)
        s = engines.vllm_signals(ids, lps)
        self.assertEqual(s["token_ids"], ids)
        self.assertEqual(s["logprob"], [-0.1, -1.0, -0.2])
        self.assertEqual(s["runner_up"], [-2.5, -1.0, -0.2])
        with self.assertRaises(RuntimeError):
            engines.vllm_signals([5, 9], lps[:1])        # misaligned
        with self.assertRaises(RuntimeError):
            engines.vllm_signals([6], [{5: L(-0.1), 7: L(-2.5)}])   # chosen token missing

    def test_production_vllm_settings_are_batch_invariant_and_recorded(self):
        """labeling plan 2026-10-24: every production vLLM spec has batch-invariant mode; it is in run.json and in the
        contract the checks must match."""
        from labeling import checks
        for setting in checks.VLLM_SETTINGS:
            self.assertIs(checks.vllm_engine(setting, 32.0)["batch_invariant"], True)
        with tempfile.TemporaryDirectory() as tmp, relabel_root(Path(tmp)):
            pool = Path(tmp) / "pool"
            (pool / "shards").mkdir(parents=True)
            run = c2.init_run(Path(tmp) / "run", pool_dir=pool, engine=checks.vllm_engine(checks.VLLM_SETTINGS[0], 32.0),
                              group_shards=4, b0_batch_seconds=1200.0, declared_vram_gb=21.0)
            self.assertIs(c2.load_run(Path(tmp) / "run")["engine"]["batch_invariant"], True)
            self.assertIs(c2.contract(run)["engine"]["batch_invariant"], True)
        eng = c2.make_engine(checks.vllm_engine(checks.VLLM_SETTINGS[1], 32.0), {})
        self.assertIs(eng.batch_invariant, True)

    def test_vllm_mode_is_set_before_vllm_starts_and_checked_against_the_installed_vllm(self):
        import os
        with mock.patch.dict(os.environ, {}, clear=False):
            engines.apply_vllm_mode(True)
            self.assertEqual(os.environ[engines.VLLM_MODE_ENV], "1")
            engines.apply_vllm_mode(False)
            self.assertEqual(os.environ[engines.VLLM_MODE_ENV], "0")
        lacking = types.SimpleNamespace(environment_variables={})
        with self.assertRaises(RuntimeError):
            engines.check_vllm_mode(lacking, True)
        has = lambda v: types.SimpleNamespace(environment_variables={engines.VLLM_MODE_ENV: None},
                                              **{engines.VLLM_MODE_ENV: v})
        engines.check_vllm_mode(has(True), True)
        engines.check_vllm_mode(has(False), False)
        with self.assertRaises(RuntimeError):
            engines.check_vllm_mode(has(False), True)
        with self.assertRaises(RuntimeError):
            engines.check_vllm_mode(has(True), False)

        class Sentinel(Exception):
            pass

        def first(mode):
            raise Sentinel(mode)

        with mock.patch.object(engines, "apply_vllm_mode", first):
            with self.assertRaises(Sentinel) as cm:
                engines.VllmTeacher(64, 0.4, batch_invariant=True).load()
        self.assertEqual(cm.exception.args, (True,))

    def test_vllm_info_records_the_mode(self):
        import sys
        fake = types.SimpleNamespace(__version__="0.32.0")
        with mock.patch.dict(sys.modules, {"vllm": fake}):
            info = engines.VllmTeacher(64, 0.4, batch_invariant=True).info()
        self.assertIs(info["batch_invariant"], True)

    def test_only_the_vllm_teacher_constructs_an_llm(self):
        """Every vLLM entry point (C2 teacher step, probe split, checks) goes through VllmTeacher.load, where the mode is
        applied and checked: no other code in the labeling package or teachers.py constructs vllm.LLM."""
        import ast
        here = Path(c2.__file__).resolve().parent
        found = []
        for path in sorted(here.glob("*.py")) + [here.parent / "teachers.py"]:
            tree = ast.parse(path.read_text())
            for cls in [n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.Module))]:
                for fn in [n for n in ast.iter_child_nodes(cls) if isinstance(n, ast.FunctionDef)]:
                    for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
                        f = call.func
                        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                        if name == "LLM":
                            found.append((path.name, getattr(cls, "name", None), fn.name))
        self.assertEqual(found, [("engines.py", "VllmTeacher", "load")])

    def test_production_engine_settings_include_the_signals(self):
        from labeling import checks
        self.assertTrue(checks.vllm_engine(checks.VLLM_SETTINGS[0], 32.0)["confidence"])
        self.assertTrue(checks.tf_engine(checks.TRANSFORMERS_SETTINGS[0])["confidence"])
        self.assertTrue(checks.tf_engine(checks.STAGE1_BATCH)["confidence"])

    def test_timing_hooks_on_the_real_model_structure(self):
        """Attach the probe's timing hooks to a tiny randomly initialized GraniteSpeechForConditionalGeneration (the
        class T* loads) and check that encoder, prefill and decode time are all measured."""
        import torch
        model = tiny_granite()
        eng = engines.TransformersTeacher(8, 60.0, device="cpu", timing=True)
        eng.teacher = types.SimpleNamespace(model=model)
        eng._hooks()
        ids = torch.tensor([[1, 2] + [127] * 8 + [3]])
        eng._lm_state["calls"] = 0
        with torch.inference_mode():
            model.generate(input_ids=ids, input_features=torch.randn(1, 16, 160), attention_mask=torch.ones_like(ids),
                           max_new_tokens=4, min_new_tokens=4, do_sample=False)
        self.assertGreater(eng.times["encoder_s"], 0)
        self.assertGreater(eng.times["prefill_s"], 0)
        self.assertGreater(eng.times["decode_s"], 0)
        self.assertEqual(eng._lm_state["calls"], 4)

    def test_batches_are_stage_one_batches(self):
        rng = np.random.default_rng(0)
        recs = [{"id": f"u{i:03d}", "duration": float(rng.uniform(0.5, 30))} for i in range(300)]
        expect = evaluate.make_batches(recs, 32, 600.0)
        order = sorted(range(len(recs)), key=lambda i: (-recs[i]["duration"], recs[i]["id"]))
        got = engines.make_batches([recs[i]["duration"] for i in order], 32, 600.0)
        self.assertEqual([[order[j] for j in b] for b in got], expect)

    def test_short_clip_padding_matches_stage_one(self):
        self.assertEqual(engines.MIN_DECODE_SAMPLES, evaluate.MIN_DECODE_SAMPLES)
        clip = np.ones(100, dtype=np.float32)
        np.testing.assert_array_equal(engines.pad_short(clip), evaluate.pad_short(clip)[0])


if __name__ == "__main__":
    unittest.main()
