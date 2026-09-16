"""Data replay, real offline HF streaming, input guards and token contracts."""

from copy import deepcopy
import io
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from binary_stt.data import DataPipelineError, StreamingSpeechDataset, iter_source_texts, resolve_sources, _shard_epoch
from binary_stt.storage import storage
from binary_stt.tokenizer import CharacterTokenizer, artifact_path, normalize_text, train_tokenizer


def audio_bytes(index=0, *, rate=16000, seconds=.3, stereo=False, nonfinite=False):
    samples = .1 * np.sin(np.arange(int(rate * seconds)) * (0.02 + index * .001)).astype(np.float32)
    if stereo:
        samples = np.stack([samples, samples * .5], axis=1)
    if nonfinite:
        samples[20] = np.nan
    output = io.BytesIO()
    sf.write(output, samples, rate, format="WAV", subtype="FLOAT")
    return output.getvalue()


def rows(count=12):
    return [{"audio": {"bytes": audio_bytes(i), "path": None}, "text": f"DO NOT use {i} mg.",
             "uid": str(i), "speaker": "speaker-1"} for i in range(count)]


def fixture_source(name="fixture"):
    return {"id": name, "config": None, "split": "train", "revision": "fixture", "text_column": "text",
            "audio_column": "audio", "id_column": "uid", "speaker_column": "speaker", "weight": 1.0}


class StatefulRows:
    def __init__(self, examples):
        self.examples = examples
        self.position = 0
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position == len(self.examples):
            raise StopIteration
        example = self.examples[self.position]
        self.position += 1
        return deepcopy(example)

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]

    def close(self):
        self.close_calls += 1


class StreamingTests(unittest.TestCase):
    def dataset(self, examples=None, **kwargs):
        examples = examples if examples is not None else rows()
        return StreamingSpeechDataset([fixture_source()], factory=lambda source, epoch: StatefulRows(examples), **kwargs)

    def test_exact_replay_preserves_prefetch_and_rng_across_epochs(self):
        examples = {"first": rows(7), "second": rows(9)}
        sources = [fixture_source(name) for name in examples]
        sources[1]["weight"] = 3
        factory = lambda source, epoch: StatefulRows(examples[source["id"]])
        original = StreamingSpeechDataset(sources, factory=factory, shuffle_buffer=4, seed=41)
        list(itertools.islice(original, 11))
        checkpoint = original.state_dict()
        expected = [(row["id"], row["content_id"]) for row in itertools.islice(original, 55)]
        resumed = StreamingSpeechDataset(sources, factory=factory, shuffle_buffer=4, seed=41)
        resumed.load_state_dict(checkpoint)
        actual = [(row["id"], row["content_id"]) for row in itertools.islice(resumed, 55)]
        self.assertEqual(expected, actual)
        self.assertEqual(original.stats, resumed.stats)
        self.assertNotEqual(checkpoint["stats"], original.stats)

    def test_finite_eval_every_row_once_and_resume_exhaustion(self):
        stream = self.dataset(repeat=False, shuffle_buffer=3)
        first = list(itertools.islice(stream, 5))
        checkpoint = stream.state_dict()
        remaining = list(stream)
        resumed = self.dataset(repeat=False, shuffle_buffer=3)
        resumed.load_state_dict(checkpoint)
        self.assertEqual([r["id"] for r in remaining], [r["id"] for r in resumed])
        self.assertEqual(len({row["id"] for row in first + remaining}), 12)
        self.assertEqual(list(stream), [])

    def test_resample_mono_normalize_and_pcm_identity(self):
        example = rows(1)[0]
        example["audio"]["bytes"] = audio_bytes(rate=48000, stereo=True)
        sample = next(self.dataset([example], shuffle_buffer=1))
        self.assertEqual(sample["audio"].dtype, torch.float32)
        self.assertEqual(sample["audio"].shape, (4800,))
        self.assertEqual(sample["sample_rate"], 16000)
        self.assertEqual(sample["text"], "do not use 0 mg.")
        self.assertEqual(sample["speaker"], "speaker-1")
        self.assertEqual(len(sample["content_id"]), 64)

    def test_nonfinite_audio_and_rejection_guard(self):
        examples = rows(4)
        for example in examples:
            example["audio"]["bytes"] = audio_bytes(nonfinite=True)
        stream = self.dataset(examples, shuffle_buffer=1, max_consecutive_bad=2)
        with self.assertRaisesRegex(DataPipelineError, "Excessive rejected"):
            next(stream)
        self.assertEqual(stream.stats["rejected"], 2)
        self.assertEqual(stream.stats["reasons"]["nonfinite_audio"], 2)

    def test_rejection_fraction_and_empty_source_fail(self):
        examples = rows(4)
        examples[0]["text"] = ""
        examples[1]["text"] = ""
        stream = self.dataset(examples, shuffle_buffer=1, rejection_fraction_min_samples=2,
                              max_rejection_fraction=.5)
        with self.assertRaisesRegex(DataPipelineError, "Excessive rejected"):
            next(stream)
        with self.assertRaisesRegex(DataPipelineError, "Empty source"):
            next(self.dataset([], shuffle_buffer=1))
        invalid = rows(1)
        invalid[0]["text"] = ""
        with self.assertRaisesRegex(DataPipelineError, "entirely rejected"):
            next(self.dataset(invalid, shuffle_buffer=1))

    def test_long_input_rejected_without_truncating_transcript(self):
        examples = rows(2)
        examples[0]["audio"]["bytes"] = audio_bytes(seconds=1.1)
        stream = self.dataset(examples, max_seconds=1, shuffle_buffer=1)
        accepted = next(stream)
        self.assertEqual(accepted["text"], "do not use 1 mg.")
        self.assertEqual(stream.stats["reasons"]["duration_out_of_range"], 1)

    def test_unpinned_and_changed_config_rejected(self):
        source = fixture_source()
        source["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "commit SHA"):
            StreamingSpeechDataset([source])
        stream = self.dataset(seed=1)
        state = stream.state_dict()
        with self.assertRaisesRegex(DataPipelineError, "mismatch"):
            self.dataset(seed=2).load_state_dict(state)

    def test_io_failure_is_not_skipped(self):
        class FailedRows(StatefulRows):
            def __next__(self):
                raise OSError("upstream unavailable")
        stream = StreamingSpeechDataset([fixture_source()], factory=lambda s, e: FailedRows([]))
        with self.assertRaisesRegex(DataPipelineError, "Streaming read failed"):
            next(stream)
        self.assertEqual(stream.stats["rejected"], 0)

    def test_close_is_idempotent_preserves_checkpoint_and_can_resume(self):
        opened = []
        def factory(source, epoch):
            stream = StatefulRows(rows(9))
            opened.append(stream)
            return stream
        stream = StreamingSpeechDataset([fixture_source()], factory=factory, shuffle_buffer=3)
        list(itertools.islice(stream, 2))
        checkpoint = stream.state_dict()
        expected = next(stream)["id"]
        stream.load_state_dict(checkpoint)
        active = opened[-1]
        before = stream.state_dict()
        stream.close()
        stream.close()
        self.assertEqual(active.close_calls, 1)
        self.assertEqual(before, stream.state_dict())
        with self.assertRaises(StopIteration):
            next(stream)
        stream.load_state_dict(before)
        self.assertEqual(next(stream)["id"], expected)
        stream.close()

    def test_context_manager_closes_source_on_early_error(self):
        underlying = StatefulRows(rows())
        stream = StreamingSpeechDataset([fixture_source()], factory=lambda s, e: underlying, shuffle_buffer=1)
        with self.assertRaisesRegex(ValueError, "consumer failed"):
            with stream:
                next(stream)
                raise ValueError("consumer failed")
        self.assertEqual(underlying.position, 1)
        self.assertEqual(underlying.close_calls, 1)

    def test_bounded_and_explicitly_closed_text_streams_release_sources(self):
        for bounded in (False, True):
            underlying = StatefulRows(rows())
            with patch("binary_stt.data._load_hf_source", return_value=underlying):
                texts = iter_source_texts([fixture_source()], max_records=2)
                if bounded:
                    self.assertEqual(len(list(texts)), 2)
                    self.assertEqual(underlying.position, 2)
                else:
                    next(texts)
                    texts.close()
                    self.assertEqual(underlying.position, 1)
                self.assertEqual(underlying.close_calls, 1)

    def test_real_hf_parquet_resume_and_text_projection(self):
        from datasets import Audio, Features, Value
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory(prefix="stream-test-", dir=storage()) as directory:
            path = Path(directory) / "speech.parquet"
            features = Features({"audio": Audio(decode=False), "text": Value("string"),
                                 "uid": Value("string"), "speaker": Value("string")})
            # Write the Arrow storage schema directly; HF's Audio.encode_example
            # imports torchcodec even when given already encoded WAV bytes.
            pq.write_table(pa.Table.from_pylist(rows(13), schema=features.arrow_schema), path)
            source = fixture_source("parquet")
            source["data_files"] = {"train": str(path)}
            source = resolve_sources([source])[0]
            stream = StreamingSpeechDataset([source], repeat=False, shuffle_buffer=4, seed=22)
            list(itertools.islice(stream, 5))
            checkpoint = stream.state_dict()
            expected = [(row["id"], row["content_id"]) for row in stream]
            resumed = StreamingSpeechDataset([source], repeat=False, shuffle_buffer=4, seed=22)
            resumed.load_state_dict(checkpoint)
            self.assertEqual(expected, [(row["id"], row["content_id"]) for row in resumed])
            self.assertEqual(len(expected), 8)
            texts = list(iter_source_texts([source], max_records=3))
            self.assertEqual(texts, [f"do not use {i} mg." for i in range(3)])
            with path.open("ab") as handle:
                handle.write(b"modified")
            with self.assertRaisesRegex(ValueError, "revision"):
                StreamingSpeechDataset([source])

    def test_real_shard_shuffle_first_pass_and_cross_shard_resume(self):
        from datasets import Audio, Features, Value
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory(prefix="shard-test-", dir=storage()) as directory:
            features = Features({"audio": Audio(decode=False), "text": Value("string"),
                                 "uid": Value("string"), "speaker": Value("string")})
            examples = rows(20)
            paths = []
            for index in range(4):
                path = Path(directory) / f"shard-{index}.parquet"
                pq.write_table(pa.Table.from_pylist(examples[index * 5:(index + 1) * 5],
                                                   schema=features.arrow_schema), path, row_group_size=2)
                paths.append(str(path))
            source = fixture_source("parquet")
            source["data_files"] = {"train": paths}
            source = resolve_sources([source])[0]

            # Pick seeds whose independently predicted file permutations differ;
            # avoid a probabilistic assertion that one arbitrary shuffle changes
            # order, since an identity permutation is a legitimate outcome.
            seed = next(s for s in range(100)
                        if np.random.default_rng(_shard_epoch(source, s, 0)).permutation(4)[0] != 0
                        and not np.array_equal(
                            np.random.default_rng(_shard_epoch(source, s, 0)).permutation(4),
                            np.random.default_rng(_shard_epoch(source, s, 1)).permutation(4)))
            other_seed = next(s for s in range(seed + 1, seed + 100)
                              if not np.array_equal(
                                  np.random.default_rng(_shard_epoch(source, s, 0)).permutation(4),
                                  np.random.default_rng(_shard_epoch(source, seed, 0)).permutation(4)))
            first_pass = list(StreamingSpeechDataset([source], seed=seed, shuffle_buffer=1, repeat=False))
            other_pass = list(StreamingSpeechDataset([source], seed=other_seed, shuffle_buffer=1, repeat=False))
            order = [int(row["text"].split()[3]) for row in first_pass]
            self.assertNotEqual(order[0], 0)
            self.assertNotEqual(order, [int(row["text"].split()[3]) for row in other_pass])
            self.assertEqual(sorted(order), list(range(20)))
            projected = list(iter_source_texts([source], max_records=20, seed=seed))
            self.assertEqual(projected, [row["text"] for row in first_pass])
            repeated = list(itertools.islice(
                StreamingSpeechDataset([source], seed=seed, shuffle_buffer=1), 40))
            self.assertNotEqual([row["id"] for row in repeated[:20]],
                                [row["id"] for row in repeated[20:]])
            self.assertEqual({row["id"] for row in repeated[:20]},
                             {row["id"] for row in repeated[20:]})

            # Checkpoint before, on and after shard boundaries with pending raw
            # examples. Continuations cross the end of a pass and a new epoch.
            for cut in (1, 4, 5, 6, 12, 19, 20, 24):
                original = StreamingSpeechDataset([source], seed=seed, shuffle_buffer=3)
                list(itertools.islice(original, cut))
                checkpoint = original.state_dict()
                expected = [(row["id"], row["content_id"]) for row in itertools.islice(original, 26)]
                resumed = StreamingSpeechDataset([source], seed=seed, shuffle_buffer=3)
                resumed.load_state_dict(checkpoint)
                self.assertEqual(expected, [(row["id"], row["content_id"])
                                             for row in itertools.islice(resumed, 26)])

    def test_shard_replay_rejects_unverified_hf_version(self):
        source = fixture_source("remote")
        source["revision"] = "a" * 40
        with patch("datasets.__version__", "5.1.0"):
            with self.assertRaisesRegex(DataPipelineError, "datasets==5.0.1"):
                next(StreamingSpeechDataset([source]))

    def test_missing_source_id_falls_back_to_stable_pcm_identity(self):
        example = rows(1)[0]
        del example["uid"]
        stream = self.dataset([example], shuffle_buffer=1)
        first = next(stream)
        second = next(stream)
        self.assertEqual(first["id"], second["id"])
        self.assertIn(first["content_id"], first["id"])


class TokenizerTests(unittest.TestCase):
    def test_preserve_numbers_negation_and_punctuation(self):
        self.assertEqual(normalize_text("  Do NOT give ０.５\u00a0mg!  "), "do not give 0.5 mg!")
        self.assertEqual(normalize_text("US", casefold=False), "US")

    def test_ctc_blank_and_repeat_order(self):
        tokenizer = CharacterTokenizer(" ab")
        self.assertEqual(tokenizer.encode("abb"), [2, 3, 3])
        self.assertEqual(tokenizer.decode_ctc([2, 2, 0, 2, 3, 3, 0, 3]), "aabb")
        with self.assertRaises(ValueError):
            tokenizer.encode("unknown")

    def test_real_sentencepiece_training_and_byte_fallback(self):
        with tempfile.TemporaryDirectory(prefix="tokenizer-test-", dir=storage()) as directory:
            texts = [f"do not give {i} mg. use oxygen now!" for i in range(100)]
            prefix = Path(directory) / "pieces.v1"
            tokenizer = train_tokenizer(texts, prefix, vocab_size=320, max_samples=100)
            self.assertEqual(tokenizer.vocab_size, 320)
            text = "do not give 0.5 mg! uncommon café λ"
            encoded = tokenizer.encode(text)
            self.assertNotIn(0, encoded)
            self.assertNotIn(tokenizer.unk_id, encoded)
            self.assertEqual(tokenizer.decode(encoded), text)
            with self.assertRaises(FileExistsError):
                train_tokenizer(texts, prefix, vocab_size=320)

    def test_drive_guard(self):
        with patch("binary_stt.storage.os.path.ismount", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "not mounted"):
                artifact_path("/mnt/hd/wilderness-labs-stt/binary-stt/test.model")
        with self.assertRaises(ValueError):
            artifact_path("/tmp/wrong-drive.model")


if __name__ == "__main__":
    unittest.main()
