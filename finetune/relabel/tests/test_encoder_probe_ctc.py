"""encoder_probe_ctc contracts (encoder-probe-ctc plan "Method", "Implementation checks" and "Code"): greedy CTC
collapse; posterior-weighted pooling with the reference's final partial window; the BPE (output i -> tokenizer id
i - 1, blank 0) and character maps on known labels; the middle-layer capture returns the pre-update logits; the
conventions in summary.json; the record schema; resume and skip; smoke-path isolation; the GPU lock.

CPU only: a tiny encoder with the installed encoder's in-place self-conditioning update, a stand-in feature
extractor and tokenizer, and real dev audio. The snapshot's own tokenizer is used where it is cached.

Run from finetune/relabel:  ./python -m unittest discover -s tests -v
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import encoder_probe_ctc as ep  # noqa: E402
import evaluate  # noqa: E402
import relabel_paths  # noqa: E402

TMP = relabel_paths.ARTIFACTS / "tmp" / "tests" / "encoder-probe-ctc"
RECORD_KEYS = {"id", "ref", "hyp", "ref_norm", "hyp_norm", "S", "D", "I", "scored", "duration"}


# --- stand-ins ----------------------------------------------------------------------------------------------------

class TinyEncoder(nn.Module):
    """The installed GraniteSpeechCTCEncoder's forward, shrunk: after layer num_layers // 2 (1-based) the character
    head reads a clone, and the layer output is then updated in place (hidden_states += out_mid(softmax(...)))."""

    def __init__(self, dim: int = 8, vocab: int = 7, layers: int = 4, input_dim: int = 160, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.input_linear = nn.Linear(input_dim, dim)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.Tanh()) for _ in range(layers)])
        self.out = nn.Linear(dim, vocab)
        self.out_mid = nn.Linear(vocab, dim)
        self.num_layers = layers
        with torch.no_grad():
            self.out_mid.bias.add_(0.5)  # make the in-place update visible

    def forward(self, hidden_states, return_dict=True):
        hidden_states = self.input_linear(hidden_states)
        for idx, layer in enumerate(self.layers, start=1):
            hidden_states = layer(hidden_states)
            if idx == self.num_layers // 2:
                hidden_states_mid = hidden_states.clone()
                hidden_states_mid = self.out(hidden_states_mid)
                hidden_states += self.out_mid(nn.Softmax(dim=-1)(hidden_states_mid))
        return SimpleNamespace(last_hidden_state=hidden_states)


class StandInProcessor:
    """160-d frames at 50 Hz like GraniteSpeechFeatureExtractor for one clip: (samples // 160 + 1) // 2 frames."""

    def __call__(self, texts, audios, device=None, return_tensors=None):
        wav = audios[0].float()
        frames = (len(wav) // 160 + 1) // 2
        padded = torch.nn.functional.pad(wav, (0, frames * 320 - len(wav))) if frames * 320 > len(wav) else wav
        chunks = padded[: frames * 320].reshape(frames, 320)
        energy = chunks.abs().mean(dim=1, keepdim=True)
        ramp = torch.linspace(-1, 1, 160).view(1, 160)
        feats = torch.sin(energy * 50 * ramp + torch.arange(frames).view(-1, 1) * 0.3)
        return {"input_features": feats.unsqueeze(0)}


class StandInTokenizer:
    pieces = ["a", "b", " c", "d", "<|end|>"]
    all_special_ids = [4]

    def __len__(self):
        return len(self.pieces)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[i] for i in ids if not (skip_special_tokens and i in self.all_special_ids))


def stand_in_probe(seed: int = 0) -> "ep.Probe":
    encoder = TinyEncoder(seed=seed).eval()
    torch.manual_seed(seed + 1)
    out_llm = nn.Linear(8, len(StandInTokenizer.pieces) + 1).eval()
    return ep.Probe(processor=StandInProcessor(), prompt="<|audio|>p", tokenizer=StandInTokenizer(), encoder=encoder,
                    out_llm=out_llm, device=torch.device("cpu"))


def stand_in_loader(device):
    return stand_in_probe(), {"name": "stand-in", "revision": "0" * 40}


# --- pure contracts -----------------------------------------------------------------------------------------------

class CollapseTest(unittest.TestCase):
    def test_repeats_are_merged_then_blanks_dropped(self) -> None:
        self.assertEqual(ep.ctc_collapse([1, 1, 0, 1, 2, 2, 0, 0, 3]), [1, 1, 2, 3])
        self.assertEqual(ep.ctc_collapse([5, 5, 5]), [5])
        self.assertEqual(ep.ctc_collapse([5, 0, 5]), [5, 5])  # a blank separates a doubled label

    def test_empty_and_all_blank(self) -> None:
        self.assertEqual(ep.ctc_collapse([]), [])
        self.assertEqual(ep.ctc_collapse([0, 0, 0]), [])

    def test_greedy_takes_the_frame_argmax(self) -> None:
        logits = torch.tensor([[0.0, 0.0, 3.0], [0.0, 1.0, 2.0], [5.0, 0.0, 0.0], [0.0, 4.0, 1.0]],
                              dtype=torch.bfloat16)
        frames, labels = ep.greedy(logits)
        self.assertEqual(frames, [2, 2, 0, 1])
        self.assertEqual(labels, [2, 1])


class PoolTest(unittest.TestCase):
    def test_hand_computed_with_a_partial_final_window(self) -> None:
        h = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [4.0, 0.0], [0.0, 8.0], [4.0, 4.0]]).unsqueeze(0)
        w = torch.tensor([[1.0, 0.0, 1.0, 2.0, 3.0, 1.0]])
        pooled = ep.posterior_weighted_pool(h, w, 4)
        self.assertEqual(tuple(pooled.shape), (1, 2, 2))
        # window 0: (1*h0 + 0*h1 + 1*h2 + 2*h3) / 4; window 1 (frames 4, 5 and two zero-padded frames): (3*h4 + h5) / 4
        expected = torch.tensor([[(1 + 2 + 8) / 4, (0 + 2 + 0) / 4], [(0 + 4) / 4, (24 + 4) / 4]])
        torch.testing.assert_close(pooled[0], expected, rtol=1e-6, atol=1e-6)

    def test_whole_windows_need_no_padding(self) -> None:
        h = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
        w = torch.tensor([[0.0, 0.0, 0.0, 2.0]])
        torch.testing.assert_close(ep.posterior_weighted_pool(h, w, 4)[0, 0], h[0, 3])

    def test_an_all_blank_window_pools_to_about_zero(self) -> None:
        h = torch.ones(1, 4, 3)
        self.assertLess(float(ep.posterior_weighted_pool(h, torch.zeros(1, 4), 4).abs().max()), 1e-6)

    def test_bf16_hidden_states_pool_in_float32(self) -> None:
        pooled = ep.posterior_weighted_pool(torch.ones(1, 5, 2, dtype=torch.bfloat16), torch.ones(1, 5), 4)
        self.assertEqual(pooled.dtype, torch.float32)
        self.assertEqual(pooled.shape[1], 2)

    def test_pooled_length_is_the_reference_formula(self) -> None:
        # enc_len = min(samples // 160 // 2 + 1, frames); pooled = ceil(enc_len / 4)
        self.assertEqual(ep.pooled_length(16000, 50), 13)
        self.assertEqual(ep.pooled_length(16000, 48), 12)
        self.assertEqual(ep.pooled_length(480, 2), 1)
        self.assertEqual(ep.pooled_length(160 * 9, 100), 2)  # formula below frames: 9 // 2 + 1 = 5 -> 2

    def test_importance_is_one_minus_the_middle_blank_probability(self) -> None:
        logits = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]], dtype=torch.bfloat16)
        imp = ep.importance(logits)
        self.assertEqual(imp.dtype, torch.float32)
        p = torch.softmax(torch.tensor([2.0, 0.0]), dim=-1)
        torch.testing.assert_close(imp, torch.tensor([[0.5, 1 - float(p[0])]]))


SNAPSHOT_TOKENIZER = None


def snapshot_tokenizer():
    global SNAPSHOT_TOKENIZER
    if SNAPSHOT_TOKENIZER is None:
        try:
            from transformers import AutoTokenizer

            SNAPSHOT_TOKENIZER = AutoTokenizer.from_pretrained(ep.HF_ID, revision=ep.REVISION, local_files_only=True)
        except Exception:  # noqa: BLE001  (not cached on this machine)
            SNAPSHOT_TOKENIZER = False
    return SNAPSHOT_TOKENIZER


class MappingTest(unittest.TestCase):
    def test_bpe_output_i_is_tokenizer_id_i_minus_1(self) -> None:
        tok = StandInTokenizer()
        # frame labels: blank, 'b' (id 1 -> output 2) twice, blank, ' c' (output 3), 'a' (output 1)
        frames = [0, 2, 2, 0, 3, 1, 0]
        self.assertEqual(ep.bpe_token_ids(ep.ctc_collapse(frames)), [1, 2, 0])
        self.assertEqual(ep.bpe_text(ep.ctc_collapse(frames), tok), "b ca")
        self.assertEqual(ep.bpe_text([], tok), "")
        self.assertEqual(ep.bpe_text([5, 1], tok), "a")  # output 5 = id 4, a special token: skipped

    @unittest.skipUnless(snapshot_tokenizer(), "granite-speech-5.0-2b tokenizer not cached here")
    def test_bpe_known_labels_with_the_snapshot_tokenizer(self) -> None:
        tok = snapshot_tokenizer()
        ids = tok.encode(" hello world", add_special_tokens=False)
        frames = [0] + [ids[0] + 1] * 3 + [0] + [i + 1 for i in ids[1:]] + [0]
        self.assertEqual(ep.bpe_text(ep.ctc_collapse(frames), tok), "hello world")
        eot = tok.convert_tokens_to_ids("<|end_of_text|>")
        self.assertEqual(ep.bpe_text([ids[0] + 1, eot + 1], tok), "hello")
        self.assertEqual(tok.vocab_size, ep.BPE_OUTPUTS - 1)

    def test_character_map_on_known_labels(self) -> None:
        frames = [0, ord("h"), ord("h"), 0, ord("i"), ord(" "), ord(" "), 0, ord("y"), 0, ord("o"), ord("o")]
        self.assertEqual(ep.char_text(ep.ctc_collapse(frames)), "hi yo")
        self.assertEqual(ep.char_text(ep.ctc_collapse([ord("l"), 0, ord("l")])), "ll")
        self.assertEqual(ep.char_text([]), "")
        self.assertEqual(ep.char_text([ord("a"), 300]), "a" + chr(300))  # the reference's chr(i), also above 255


class MidCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.enc = TinyEncoder().eval()
        torch.manual_seed(3)
        self.x = torch.randn(1, 9, 160)

    def test_capture_returns_the_pre_update_logits(self) -> None:
        saved = {}

        def keep(module, args, output):
            saved["post"], saved["pre"] = output, output.detach().clone()

        hook = self.enc.layers[self.enc.num_layers // 2 - 1].register_forward_hook(keep)
        with torch.inference_mode():
            final, mid = ep.encode(self.enc, self.x)
            hook.remove()
            pre, post = self.enc.out(saved["pre"]), self.enc.out(saved["post"])
        self.assertTrue(torch.equal(mid, pre))
        self.assertFalse(torch.equal(mid, post))  # the notebook's saved tensor is the post-update state

    def test_hooks_are_passive(self) -> None:
        with torch.inference_mode():
            plain = self.enc(self.x).last_hidden_state
            final, _ = ep.encode(self.enc, self.x)
        self.assertTrue(torch.equal(plain, final))
        self.assertEqual(len(self.enc.out._forward_hooks), 0)  # removed after the pass

    def test_exactly_one_middle_call_is_required(self) -> None:
        with ep.HeadCapture(self.enc.out) as cap, torch.inference_mode():
            pass
        with self.assertRaises(RuntimeError):
            cap.logits()
        with ep.HeadCapture(self.enc.out) as cap, torch.inference_mode():
            self.enc(self.x)
            self.enc.out(torch.zeros(1, 8))
        with self.assertRaises(RuntimeError):
            cap.logits()


class ClipChecksTest(unittest.TestCase):
    """Implementation checks 1-3 as functions, on the tiny encoder and real dev audio."""

    @classmethod
    def setUpClass(cls) -> None:
        relabel_paths.require_mount()
        records = evaluate.read_manifest(evaluate.resolve_manifest("ami_dev")[1])
        short = min(records, key=lambda r: (r["duration"], r["id"]))
        cls.clips = [("ami_dev", r) for r in [records[0], records[1], short]]

    def test_checks_pass_and_check_2_is_not_vacuous(self) -> None:
        probe = stand_in_probe()
        c1 = ep.check_hooks_passive(probe, self.clips)
        c2 = ep.check_pre_feedback(probe, self.clips)
        c3 = ep.check_determinism(probe, self.clips)
        self.assertTrue(c1["passed"] and c2["passed"] and c3["passed"], (c1, c2, c3))
        self.assertEqual(c1["clips"], 3)
        self.assertGreater(c2["post_update_max_abs_logit_diff"], 0.0)
        self.assertEqual(c2["post_update_equal_clips"], 0)

    def test_check_2_fails_if_the_capture_reads_the_post_update_state(self) -> None:
        probe = stand_in_probe()

        def post_update(encoder, features):  # the notebook's way: encoder.out on the saved, later-updated tensor
            saved = {}
            hook = encoder.layers[encoder.num_layers // 2 - 1].register_forward_hook(
                lambda m, a, o: saved.__setitem__("h", o))
            try:
                final = encoder(features, return_dict=True).last_hidden_state
            finally:
                hook.remove()
            return final, encoder.out(saved["h"])

        with mock.patch.object(ep, "encode", post_update):
            self.assertFalse(ep.check_pre_feedback(probe, self.clips)["passed"])

    def test_check_3_fails_on_nondeterministic_output(self) -> None:
        probe = stand_in_probe()
        real = ep.Probe.decode
        calls = {"n": 0}

        def flaky(self, clip):
            out = real(self, clip)
            calls["n"] += 1
            if calls["n"] > len(ClipChecksTest.clips):
                out = {**out, "bpe_frames": out["bpe_frames"][:-1] + [out["bpe_frames"][-1] + 1]}
            return out

        with mock.patch.object(ep.Probe, "decode", flaky):
            self.assertFalse(ep.check_determinism(probe, self.clips)["passed"])

    def test_check_clip_selection(self) -> None:
        manifests = {n: evaluate.read_manifest(evaluate.resolve_manifest(n)[1]) for n in relabel_paths.SETS}
        picked = ep.check_clips(manifests, 4)
        self.assertEqual(len(picked), 20)
        self.assertEqual(sorted({n for n, _ in picked}), sorted(relabel_paths.SETS))
        for name in relabel_paths.SETS:
            shortest = min(manifests[name], key=lambda r: (r["duration"], r["id"]))
            self.assertIn(shortest["id"], {r["id"] for n, r in picked if n == name})
        self.assertEqual(picked, ep.check_clips(manifests, 4))
        self.assertEqual(len(ep.check_clips(manifests, 10)), 50)


class ConventionsTest(unittest.TestCase):
    REQUIRED = {"char_blank", "char_layer", "char_map", "bpe_blank", "bpe_index_offset", "bpe_layer", "mid_layer",
                "mid_posteriors", "pool_weights", "pool_formula", "pool_window", "pool_alignment",
                "pool_final_partial_window", "pool_length", "bpe_detokenization", "char_detokenization", "greedy"}

    def test_every_convention_has_a_value_and_a_reference_source(self) -> None:
        self.assertLessEqual(self.REQUIRED, set(ep.CONVENTIONS))
        files = set(ep.REFERENCE["files"])
        for name, entry in ep.CONVENTIONS.items():
            self.assertTrue(entry["value"], name)
            self.assertTrue(any(f in entry["source"] for f in files) or "plan" in entry["source"], name)


# --- entry point --------------------------------------------------------------------------------------------------

class OutputPathTest(unittest.TestCase):
    def test_smoke_path_never_equals_production_path(self) -> None:
        self.assertEqual(ep.output_dir(None), ep.OUT)
        self.assertFalse(ep.OUT.is_relative_to(relabel_paths.SMOKE))
        for limit in (1, 8, 100000):
            smoke = ep.output_dir(limit)
            self.assertTrue(smoke.is_relative_to(relabel_paths.SMOKE))
            self.assertFalse(smoke.is_relative_to(ep.OUT))
        with self.assertRaises(ValueError):
            ep.output_dir(0)

    def test_cpu_is_smoke_only_and_sets_are_dev_only(self) -> None:
        with self.assertRaises(SystemExit):
            ep.parse_args(["--device", "cpu"])
        with self.assertRaises(SystemExit):
            ep.parse_args(["--shortest", "2"])
        with self.assertRaises(SystemExit):
            ep.main(["--sets", "librispeech_clean", "--limit", "1", "--device", "cpu"])


class EntryTestBase(unittest.TestCase):
    def setUp(self) -> None:
        relabel_paths.require_mount()
        self.root = TMP / self.__class__.__name__
        if self.root.exists():
            shutil.rmtree(self.root)
        self.patches = [mock.patch.object(relabel_paths, "SMOKE", self.root / "smoke"),
                        mock.patch.object(ep, "OUT", self.root / "production"),
                        mock.patch.object(relabel_paths, "GPU_LOCK", self.root / "gpu.lock"),
                        mock.patch.object(ep, "load_probe", stand_in_loader)]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()

    def run_main(self, *extra):
        return ep.main(["--limit", "3", "--shortest", "1", "--sets", "ami_dev", "yodas_dev", "--device", "cpu",
                        "--min-available-gb", "0", *extra])

    @property
    def out(self) -> Path:
        return self.root / "smoke" / "encoder-probe-ctc-limit3-cpu"


class EntryPointTest(EntryTestBase):
    def test_smoke_writes_both_heads_and_the_summary(self) -> None:
        self.assertEqual(self.run_main(), 0)
        for head in ep.HEADS:
            for name in ("ami_dev", "yodas_dev"):
                data = json.loads((self.out / head / f"{name}.json").read_text())
                self.assertEqual(data["head"], head)
                self.assertEqual(data["set"], name)
                self.assertEqual(data["limit"], 3)
                self.assertEqual(len(data["records"]), 4)  # 3 first + the shortest
                ids = [r["id"] for r in data["records"]]
                self.assertEqual(ids, sorted(ids))
                for r in data["records"]:
                    self.assertLessEqual(RECORD_KEYS, set(r))
                    self.assertEqual({k: r[k] for k in ("ref_norm", "hyp_norm", "S", "D", "I", "scored")},
                                     evaluate.score(r["ref"], r["hyp"]))
                    self.assertIsInstance(r["labels"], list)
                self.assertEqual(data["wer"], evaluate.wer_summary(data["records"]))
                for key in ("audio_hours", "inference_wall_s", "audio_hours_per_gpu_hour", "peak_gpu_memory_gb"):
                    self.assertIn(key, data["throughput"])
            ami = json.loads((self.out / head / "ami_dev.json").read_text())
            self.assertEqual(len(ami["padded_short_ids"]), 1)  # ami_dev's 0.02 s clip, padded as B0's scorer does
        bpe = json.loads((self.out / "bpe" / "ami_dev.json").read_text())
        tok = StandInTokenizer()
        for r in bpe["records"]:
            self.assertEqual(r["hyp"], tok.decode([i - 1 for i in r["labels"]], skip_special_tokens=True).strip())
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["status"], "smoke")
        self.assertFalse(summary["complete"])
        self.assertIsNone(summary["gpu_lock"])
        self.assertEqual(summary["conventions"], ep.CONVENTIONS)
        self.assertEqual(summary["reference"], ep.REFERENCE)
        for key in ("revision", "versions", "precision", "timings", "peak_memory", "checks", "heads"):
            self.assertIn(key, summary)
        self.assertTrue(all(summary["checks"][k]["passed"] for k in ("hooks_passive", "pre_feedback",
                                                                       "determinism")))
        self.assertEqual(summary["checks"]["hooks_passive"]["clips"], 20)
        self.assertEqual(summary["checks"]["determinism"]["clips"], 50)
        self.assertFalse((self.root / "production").exists())

    def test_resume_skips_identical_sets_and_force_redoes_them(self) -> None:
        self.assertEqual(self.run_main(), 0)
        with mock.patch.object(ep.Probe, "decode", side_effect=AssertionError("re-decoded")):
            self.assertEqual(self.run_main(), 0)
        self.assertEqual(json.loads((self.out / "summary.json").read_text())["status"], "smoke")
        seen = []
        real = ep.Probe.decode

        def count(probe, clip):
            seen.append(len(clip))
            return real(probe, clip)

        with mock.patch.object(ep.Probe, "decode", count):
            self.assertEqual(self.run_main("--force"), 0)
        self.assertGreaterEqual(len(seen), 8)

    def test_a_set_missing_one_head_is_redone(self) -> None:
        self.assertEqual(self.run_main(), 0)
        (self.out / "bpe" / "yodas_dev.json").unlink()
        seen = []
        real = ep.Probe.decode

        def count(probe, clip):
            seen.append(len(clip))
            return real(probe, clip)

        with mock.patch.object(ep, "run_checks", return_value={"passed": True}), \
             mock.patch.object(ep.Probe, "decode", count):
            self.assertEqual(self.run_main(), 0)
        self.assertEqual(len(seen), 4)  # yodas_dev only
        self.assertTrue((self.out / "bpe" / "yodas_dev.json").exists())

    def test_failed_checks_stop_before_scoring(self) -> None:
        bad = {"passed": False, "hooks_passive": {"passed": False}}
        with mock.patch.object(ep, "run_checks", return_value=bad):
            self.assertEqual(self.run_main(), ep.CHECKS_FAILED_EXIT)
        self.assertFalse((self.out / "bpe").exists())
        self.assertFalse((self.out / "char").exists())
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["status"], "checks_failed")
        self.assertEqual(summary["checks"], bad)

    def test_interrupted_run_does_not_leave_a_finished_summary(self) -> None:
        self.assertEqual(self.run_main(), 0)
        with mock.patch.object(ep.Probe, "decode", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_main("--force")
        self.assertEqual(json.loads((self.out / "summary.json").read_text())["status"], "in_progress")


class LockTest(EntryTestBase):
    def test_refuses_to_run_while_the_lock_is_held(self) -> None:
        lock = self.root / "gpu.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with open(lock, "a+") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            with mock.patch.object(ep, "load_probe", side_effect=AssertionError("loaded without the lock")):
                rc = ep.main(["--limit", "1", "--lock-wait", "0.2"])
            fcntl.flock(holder, fcntl.LOCK_UN)
        self.assertEqual(rc, ep.LOCK_BUSY_EXIT)

    def test_the_model_is_loaded_inside_the_lock(self) -> None:
        lock = self.root / "gpu.lock"
        seen = {}

        def loader(device):
            with open(lock, "a+") as other:
                try:
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    seen["held"] = False
                    fcntl.flock(other, fcntl.LOCK_UN)
                except BlockingIOError:
                    seen["held"] = True
            raise KeyboardInterrupt  # stop before CUDA would be needed

        with mock.patch.object(ep, "load_probe", loader), mock.patch.object(ep, "_require_cuda", lambda device: None):
            with self.assertRaises(KeyboardInterrupt):
                ep.main(["--limit", "1", "--lock-wait", "1", "--min-available-gb", "0"])
        self.assertTrue(seen["held"])


if __name__ == "__main__":
    unittest.main()
