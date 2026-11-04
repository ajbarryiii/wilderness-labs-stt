"""heldout_panel scoring contracts (heldout-panel plan "Sets", "Models and records", "Outputs"):

- the panel's sets are the ten domain parts, read only from their complete production builds; anything else is refused,
  and the existing scorers' dev-set (H4) populations and output paths are unchanged;
- the gaps scored per model are the plan's;
- a scored set's record is the existing scorer's own record for that input (T* through score_teacher.score_set, T*'s
  encoder through encoder_probe_ctc.score_set, BPE head), plus critical-term recall as score_domain computes it and
  the panel's provenance (model identity, set, manifest SHA-256, code commit);
- the scorer's configuration must equal the one its existing records were scored with, or nothing is scored;
- a cross-check against an existing record refuses a production run that does not reproduce it;
- smoke runs write only under relabel_paths.SMOKE; resume skips what is written; the GPU lock and host-memory floor.

CPU only: stand-in scorers and teachers, the real domain manifests and dev audio.
Run from finetune/relabel:  ./python -m unittest discover -s tests -v
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

import devsets  # noqa: E402
import encoder_probe_ctc as ep  # noqa: E402
import evaluate  # noqa: E402
import heldout_panel as hp  # noqa: E402
import relabel_paths  # noqa: E402
import score_domain  # noqa: E402
import score_teacher  # noqa: E402
import teachers  # noqa: E402
import terms  # noqa: E402
from devsets import layout  # noqa: E402

TMP = relabel_paths.ARTIFACTS / "tmp" / "tests" / "heldout-panel"
RECORD_KEYS = {"id", "ref", "hyp", "ref_norm", "hyp_norm", "S", "D", "I", "scored", "duration"}
T5 = ("tech-dev-1", "tech-dev-2", "stem-chem", "stem-cs", "stem-math")
M3 = ("med-dev-1", "med-dev-2", "med-dev-3")
X2 = ("lectures-a", "medical-b")
VOLATILE = {"wall_s", "throughput", "model_load_s_this_run", "panel", "term_recall", "term_kind", "records"}


def manifest_of(part: str) -> Path:
    return layout.Layout.production().manifest(part)


def stand_in_rows(manifest: Path, limit: int | None, upper: bool = False) -> list[dict]:
    rows = evaluate.read_manifest(manifest)[:limit]
    return [{"id": r["id"], "ref": r["text"], "hyp": (r["text"].upper() if upper else r["text"][::-1]),
             **evaluate.score(r["text"], r["text"].upper() if upper else r["text"][::-1]), "duration": r["duration"]}
            for r in rows]


class StandInScorer(hp.Scorer):
    """Scores by reversing each reference (deterministic, imperfect), so WER, term recall and records are real."""

    calls: list = []

    def __init__(self, model: str = "b100-main", cross: tuple | None = None):
        self.model, self.cross, self.loaded = model, cross, False

    def prepare(self, limit, device_name):
        self.identity = {"stand_in": "0" * 8}
        self.scorer = {"kind": "stand-in"}

    def crosscheck_input(self):
        return self.cross

    def load(self, device):
        self.loaded = True

    def score(self, part, manifest, limit):
        StandInScorer.calls.append(part)
        rows = stand_in_rows(manifest, limit)
        return {"set": part, "utterances": len(rows), "wer": evaluate.wer_summary(rows),
                "config": {"stand_in": True, "limit": limit}, "records": rows}

    def close(self):
        pass


class SetsTest(unittest.TestCase):
    def test_the_ten_sets_and_the_panels(self) -> None:
        self.assertEqual(set(hp.SETS), set(devsets.PARTS))
        self.assertEqual(len(hp.SETS), 10)
        self.assertEqual(tuple(hp.PANELS["T5"]), T5)
        self.assertEqual(tuple(hp.PANELS["T5-no-tech-dev-2"]), tuple(s for s in T5 if s != "tech-dev-2"))
        self.assertEqual(tuple(hp.PANELS["M3"]), M3)
        self.assertEqual(tuple(hp.PANELS["X2"]), X2)

    def test_the_gaps_scored_are_the_plans(self) -> None:
        for arm in ("b100-main", "b100-control", "B0p"):
            self.assertEqual(set(hp.SCORED[arm]), set(X2))
        for model in ("tstar", "tstar-encoder"):
            self.assertEqual(set(hp.SCORED[model]), set(hp.SETS))
        self.assertEqual(set(hp.SCORED), {"b100-main", "b100-control", "B0p", "tstar", "tstar-encoder"})


class ManifestTest(unittest.TestCase):
    def test_accepts_the_ten_domain_parts_from_their_builds(self) -> None:
        relabel_paths.require_mount()
        got = hp.domain_manifests(hp.SETS)
        self.assertEqual(list(got), list(hp.SETS))
        for part, (manifest, sha) in got.items():
            self.assertEqual(manifest, manifest_of(part))
            meta = json.loads(layout.Layout.production().meta(part).read_text())
            self.assertEqual(sha, meta["manifest_sha256"])
            self.assertEqual(sha, evaluate.sha256_file(manifest))

    def test_refuses_anything_else(self) -> None:
        for names in (["ami_dev"], ["librispeech_dev_clean"], ["dev-medical"], ["tech-dev-1", "yodas_dev"]):
            with self.assertRaises(SystemExit):
                hp.domain_manifests(names)

    def test_refuses_a_manifest_that_differs_from_its_build(self) -> None:
        root = TMP / "changed"
        if root.exists():
            shutil.rmtree(root)
        lay = layout.Layout(root / "devsets")
        lay.manifests().mkdir(parents=True)
        shutil.copy(manifest_of("stem-chem"), lay.manifest("stem-chem"))
        meta = json.loads(layout.Layout.production().meta("stem-chem").read_text())
        with open(lay.manifest("stem-chem"), "a") as handle:
            handle.write("\n")  # same rows, different bytes
        lay.meta("stem-chem").write_text(json.dumps(meta))
        with mock.patch.dict(os.environ, {"RELABEL_ROOT": str(root)}), self.assertRaises(SystemExit):
            hp.domain_manifests(["stem-chem"])

    def test_existing_scorers_keep_their_dev_sets_and_paths(self) -> None:
        self.assertEqual(score_teacher.expand_sets(["dev"]), relabel_paths.SETS)
        with self.assertRaises(SystemExit):
            score_teacher.expand_sets(["tech-dev-1"])
        self.assertEqual(relabel_paths.output_dir(hp.TEACHER, None), relabel_paths.DIAGNOSTIC / hp.TEACHER)
        self.assertEqual(ep.output_dir(None), ep.OUT)


class OutputPathTest(unittest.TestCase):
    def test_smoke_never_lands_on_the_production_path(self) -> None:
        for model in hp.SCORED:
            production = hp.output_dir(model, None)
            self.assertEqual(production, relabel_paths.ARTIFACTS / "heldout-panel" / model)
            for limit in (1, 50):
                smoke = hp.output_dir(model, limit)
                self.assertTrue(smoke.is_relative_to(relabel_paths.SMOKE))
                self.assertFalse(smoke.is_relative_to(hp.out_root()))
        for bad in ("B0", "MB1", "report.json", "../x"):
            with self.assertRaises(ValueError):
                hp.output_dir(bad, None)
        with self.assertRaises(ValueError):
            hp.output_dir("tstar", 0)


class FlowBase(unittest.TestCase):
    def setUp(self) -> None:
        relabel_paths.require_mount()
        self.root = TMP / self.id().split(".")[-1]
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        self.patches = [mock.patch.object(relabel_paths, "ARTIFACTS", self.root / "relabel"),
                        mock.patch.object(relabel_paths, "SMOKE", self.root / "relabel" / "smoke"),
                        mock.patch.object(relabel_paths, "GPU_LOCK", self.root / "gpu.lock")]
        for p in self.patches:
            p.start()
        StandInScorer.calls = []

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()

    def recorded(self, part: str, limit: int | None, upper: bool = False, name: str = "recorded.json") -> Path:
        rows = stand_in_rows(manifest_of(part), limit, upper)
        path = self.root / name
        path.write_text(json.dumps({"config": {"stand_in": True, "limit": limit}, "records": rows}))
        return path

    def run_panel(self, scorer, model="b100-main", limit=3, **kw):
        with mock.patch.dict(hp.SCORERS, {model: lambda: scorer}):
            return hp.run(model, limit=limit, lock_wait=kw.pop("lock_wait", 1.0), device="cpu",
                          min_available_gb=kw.pop("min_available_gb", 0.0), **kw)


class FlowTest(FlowBase):
    def test_smoke_writes_records_with_provenance_and_term_recall(self) -> None:
        cross = ("stem-chem", manifest_of("stem-chem"), self.recorded("stem-chem", 3))
        with mock.patch.dict(hp.SCORED, {"b100-main": ("stem-chem", "lectures-a", "tech-dev-1")}):
            self.assertEqual(self.run_panel(StandInScorer(cross=cross)), 0)
        out = hp.output_dir("b100-main", 3)
        self.assertTrue(out.is_relative_to(relabel_paths.SMOKE))
        self.assertFalse((relabel_paths.ARTIFACTS / "heldout-panel").exists())
        inputs = hp.domain_manifests(["stem-chem", "lectures-a", "tech-dev-1"])
        for part in ("stem-chem", "lectures-a", "tech-dev-1"):
            data = json.loads((out / f"{part}.json").read_text())
            self.assertEqual(len(data["records"]), 3)
            self.assertTrue(RECORD_KEYS <= set(data["records"][0]))
            cfg = data["panel"]["config"]
            self.assertEqual((cfg["model"], cfg["set"], cfg["limit"]), ("b100-main", part, 3))
            self.assertEqual(cfg["identity"], {"stand_in": "0" * 8})
            self.assertEqual(cfg["manifest_sha256"], inputs[part][1])
            self.assertRegex(data["panel"]["code"]["commit"], r"^[0-9a-f]{40}$")
            self.assertIn("dirty", data["panel"]["code"])
            kind = devsets.TERM_KIND.get(devsets.set_of(part))
            if kind is None:
                self.assertNotIn("term_recall", data)
            else:
                self.assertEqual(data["term_kind"], kind)
                rows = stand_in_rows(manifest_of(part), 3)
                self.assertEqual(data["term_recall"], score_domain.term_recall(rows, terms.matcher(kind)))
        check = json.loads((out / "crosscheck.json").read_text())
        self.assertTrue(check["pass"])
        self.assertEqual(check["identical_hypotheses"], 3)
        run = json.loads((out / "run.json").read_text())
        self.assertEqual(run["model"], "b100-main")
        self.assertIn("host_max_rss_gb", run["memory"])

    def test_resume_skips_written_sets_and_force_rescores(self) -> None:
        cross = ("stem-chem", manifest_of("stem-chem"), self.recorded("stem-chem", 3))
        with mock.patch.dict(hp.SCORED, {"b100-main": ("stem-chem", "lectures-a")}):
            self.assertEqual(self.run_panel(StandInScorer(cross=cross)), 0)
            self.assertEqual(StandInScorer.calls, ["stem-chem", "stem-chem", "lectures-a"])  # the cross-check first
            StandInScorer.calls = []
            scorer = StandInScorer(cross=cross)
            self.assertEqual(self.run_panel(scorer), 0)
            self.assertEqual(StandInScorer.calls, [])
            self.assertFalse(scorer.loaded)
            self.assertEqual(self.run_panel(StandInScorer(cross=cross), force=True), 0)
            self.assertEqual(StandInScorer.calls, ["stem-chem", "stem-chem", "lectures-a"])

    def test_failed_crosscheck_refuses_a_production_run(self) -> None:
        cross = ("stem-chem", manifest_of("stem-chem"), self.recorded("stem-chem", None, upper=True))
        with mock.patch.dict(hp.SCORED, {"b100-main": ("stem-chem",)}):
            rc = self.run_panel(StandInScorer(cross=cross), limit=None)
        self.assertEqual(rc, hp.CROSSCHECK_FAILED_EXIT)
        out = hp.output_dir("b100-main", None)
        self.assertFalse((out / "stem-chem.json").exists())
        check = json.loads((out / "crosscheck.json").read_text())
        self.assertFalse(check["pass"])
        self.assertGreater(abs(check["difference_points"]), score_domain.CROSSCHECK_POINTS)

    def test_reproduced_crosscheck_lets_a_production_run_score(self) -> None:
        cross = ("stem-chem", manifest_of("stem-chem"), self.recorded("stem-chem", None))
        with mock.patch.dict(hp.SCORED, {"b100-main": ("stem-chem",)}):
            self.assertEqual(self.run_panel(StandInScorer(cross=cross), limit=None), 0)
        data = json.loads((hp.output_dir("b100-main", None) / "stem-chem.json").read_text())
        self.assertIsNone(data["panel"]["config"]["limit"])
        self.assertEqual(data["utterances"], len(evaluate.read_manifest(manifest_of("stem-chem"))))

    def test_crosscheck_with_another_configuration_fails(self) -> None:
        path = self.recorded("stem-chem", None)
        data = json.loads(path.read_text())
        data["config"]["stand_in"] = "another scorer"
        path.write_text(json.dumps(data))
        with mock.patch.dict(hp.SCORED, {"b100-main": ("stem-chem",)}):
            rc = self.run_panel(StandInScorer(cross=("stem-chem", manifest_of("stem-chem"), path)), limit=None)
        self.assertEqual(rc, hp.CROSSCHECK_FAILED_EXIT)

    def test_smoke_crosscheck_failure_is_recorded_not_refused(self) -> None:
        cross = ("stem-chem", manifest_of("stem-chem"), self.recorded("stem-chem", 3, upper=True))
        with mock.patch.dict(hp.SCORED, {"b100-main": ("stem-chem",)}):
            self.assertEqual(self.run_panel(StandInScorer(cross=cross)), 0)
        self.assertFalse(json.loads((hp.output_dir("b100-main", 3) / "crosscheck.json").read_text())["pass"])

    def test_refuses_while_the_lock_is_held(self) -> None:
        cross = ("stem-chem", manifest_of("stem-chem"), self.recorded("stem-chem", 3))
        scorer = StandInScorer(cross=cross)
        with open(relabel_paths.GPU_LOCK, "a+") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            rc = self.run_panel(scorer, lock_wait=0.2)
            fcntl.flock(holder, fcntl.LOCK_UN)
        self.assertEqual(rc, hp.LOCK_BUSY_EXIT)
        self.assertFalse(scorer.loaded)

    def test_scores_inside_the_lock(self) -> None:
        seen = {}

        class Probing(StandInScorer):
            def load(self, device):
                with open(relabel_paths.GPU_LOCK, "a+") as other:
                    try:
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        seen["held"] = False
                        fcntl.flock(other, fcntl.LOCK_UN)
                    except BlockingIOError:
                        seen["held"] = True
                super().load(device)

        cross = ("stem-chem", manifest_of("stem-chem"), self.recorded("stem-chem", 3))
        with mock.patch.dict(hp.SCORED, {"b100-main": ("stem-chem",)}):
            self.assertEqual(self.run_panel(Probing(cross=cross)), 0)
        self.assertTrue(seen["held"])

    def test_low_host_memory_refuses_to_load(self) -> None:
        cross = ("stem-chem", manifest_of("stem-chem"), self.recorded("stem-chem", 3))
        scorer = StandInScorer(cross=cross)
        with mock.patch.object(score_teacher, "available_memory_gb", return_value=1.0):
            rc = self.run_panel(scorer, min_available_gb=16.0)
        self.assertEqual(rc, hp.LOW_MEMORY_EXIT)
        self.assertFalse(scorer.loaded)


# --- T* and T*'s encoder: the existing scorers' own records ---------------------------------------------------------

class GraniteStandIn(teachers.Teacher):
    name = hp.TEACHER
    hf_id = "stand-in"
    revision = "1" * 40
    license = "none"
    library = "test"
    batch_size = 2
    max_batch_seconds = 600.0

    def _load(self) -> None:
        pass

    def transcribe(self, clips):
        return [f"Hello, world {len(c) % 7}." for c in clips], [False] * len(clips)

    def info(self) -> dict:
        return {"name": self.name, "revision": self.revision}


class TStarPathTest(FlowBase):
    def diagnostic(self, **change) -> Path:
        """A stand-in for the diagnostic's T* records: its summary's configuration and a yodas_dev record."""
        teacher = GraniteStandIn()
        args = argparse.Namespace(limit=None, shortest=0, batch_size=teacher.batch_size,
                                  max_batch_seconds=teacher.max_batch_seconds, device="cpu")
        config = {**score_teacher.run_config(teacher, args), **change}
        out = relabel_paths.output_dir(hp.TEACHER, None)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps({"status": "scored", "config": config}))
        teacher.load("cpu")
        args.limit = 2
        record = score_teacher.score_set(teacher, "yodas_dev", evaluate.resolve_manifest("yodas_dev")[1], args,
                                         torch.device("cpu"), teacher.info())
        (out / "yodas_dev.json").write_text(json.dumps(record))
        return out

    def run_tstar(self):
        with mock.patch.object(relabel_paths, "DIAGNOSTIC", self.root / "diagnostic"), \
             mock.patch.object(teachers, "get", lambda name: GraniteStandIn()), \
             mock.patch.dict(hp.SCORED, {"tstar": ("stem-chem",)}):
            self.diagnostic_dir = self.diagnostic(**self.change)
            return hp.run("tstar", limit=2, lock_wait=1.0, device="cpu", min_available_gb=0.0)

    def test_record_is_score_teachers_record(self) -> None:
        self.change = {}
        self.assertEqual(self.run_tstar(), 0)
        data = json.loads((hp.output_dir("tstar", 2) / "stem-chem.json").read_text())
        teacher = GraniteStandIn()
        teacher.load("cpu")
        args = argparse.Namespace(limit=2, shortest=0, batch_size=2, max_batch_seconds=600.0, device="cpu")
        direct = score_teacher.score_set(teacher, "stem-chem", manifest_of("stem-chem"), args, torch.device("cpu"),
                                         data["source"])
        self.assertEqual({k: v for k, v in data.items() if k not in VOLATILE}, {
            k: v for k, v in direct.items() if k not in VOLATILE})
        self.assertEqual([{k: v for k, v in r.items() if k != "terms"} for r in data["records"]], direct["records"])
        self.assertEqual(data["panel"]["config"]["identity"]["revision"], GraniteStandIn.revision)
        self.assertEqual(data["term_kind"], "technical")
        check = json.loads((hp.output_dir("tstar", 2) / "crosscheck.json").read_text())
        self.assertEqual(check["set"], "yodas_dev")
        self.assertEqual(check["identical_hypotheses"], 2)

    def test_another_configuration_than_the_diagnostics_is_refused(self) -> None:
        self.change = {"batch_size": 16}
        with self.assertRaises(SystemExit):
            self.run_tstar()
        self.assertFalse(hp.output_dir("tstar", 2).exists())


def stand_in_loader(device):
    import test_encoder_probe_ctc as tep
    return tep.stand_in_probe(), {"name": "stand-in", "revision": ep.REVISION,
                                  "out_llm": {"sha256": "f" * 64}}


class EncoderPathTest(FlowBase):
    def probe_records(self, **change) -> None:
        """A stand-in for probe 1's records: summary (configuration, out_llm hash) and the bpe yodas_dev record."""
        args = argparse.Namespace(limit=None, shortest=0, device="cpu")
        config = {**ep.run_config(args), **change}
        ep_out = self.root / "encoder-probe"
        (ep_out / "bpe").mkdir(parents=True)
        (ep_out / "summary.json").write_text(json.dumps({"status": "scored", "config": config,
                                                         "source": {"out_llm": {"sha256": "f" * 64}}}))
        probe, info = stand_in_loader("cpu")
        args.limit = 2
        result = ep.score_set(probe, "yodas_dev", evaluate.resolve_manifest("yodas_dev")[1], args,
                              ep.run_config(args) | change, info)
        (ep_out / "bpe" / "yodas_dev.json").write_text(json.dumps(result[ep.BPE]))
        self.ep_out = ep_out

    def run_encoder(self, change=None, checks=None):
        change = change or {}
        patches = [mock.patch.object(ep, "OUT", self.root / "encoder-probe"),
                   mock.patch.object(ep, "load_probe", stand_in_loader),
                   mock.patch.dict(hp.SCORED, {"tstar-encoder": ("stem-chem",)})]
        if checks is not None:
            patches.append(mock.patch.object(ep, "run_checks", return_value=checks))
        for p in patches:
            p.start()
        try:
            self.probe_records(**change)
            return hp.run("tstar-encoder", limit=2, lock_wait=1.0, device="cpu", min_available_gb=0.0)
        finally:
            for p in patches:
                p.stop()

    def test_record_is_the_probes_bpe_record(self) -> None:
        self.assertEqual(self.run_encoder(), 0)
        data = json.loads((hp.output_dir("tstar-encoder", 2) / "stem-chem.json").read_text())
        probe, info = stand_in_loader("cpu")
        args = argparse.Namespace(limit=2, shortest=0, device="cpu")
        direct = ep.score_set(probe, "stem-chem", manifest_of("stem-chem"), args, ep.run_config(args),
                              data["source"])[ep.BPE]
        self.assertEqual(data["head"], ep.BPE)
        self.assertEqual({k: v for k, v in data.items() if k not in VOLATILE | {"diagnostics"}},
                         {k: v for k, v in direct.items() if k not in VOLATILE | {"diagnostics"}})
        self.assertEqual([{k: v for k, v in r.items() if k != "terms"} for r in data["records"]], direct["records"])
        self.assertEqual(data["panel"]["config"]["identity"]["revision"], ep.REVISION)
        self.assertFalse((hp.output_dir("tstar-encoder", 2) / ep.CHAR).exists())
        run = json.loads((hp.output_dir("tstar-encoder", 2) / "run.json").read_text())
        self.assertTrue(run["scorer_info"]["checks"]["passed"])

    def test_failed_implementation_checks_score_nothing(self) -> None:
        bad = {"passed": False, "hooks_passive": {"passed": False}}
        self.assertEqual(self.run_encoder(checks=bad), hp.CHECKS_FAILED_EXIT)
        self.assertFalse((hp.output_dir("tstar-encoder", 2) / "stem-chem.json").exists())

    def test_another_configuration_than_probe_1s_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self.run_encoder(change={"conventions_sha256": "0" * 64})
        self.assertFalse((hp.output_dir("tstar-encoder", 2) / "stem-chem.json").exists())



# --- the x1 arms and B0': the recorded identities --------------------------------------------------------------------

class X1IdentityTest(unittest.TestCase):
    """The arm's scoring view must be the one its stage 4 x1 records name, and the loaded runtime the recorded one."""

    def setUp(self) -> None:
        relabel_paths.require_mount()
        import train_keab as tk
        from training import deployed
        self.tk, self.deployed = tk, deployed
        self.root = TMP / "x1"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.view = self.root / "runs" / "b100-main" / "export" / "view"
        self.view.mkdir(parents=True)
        self.scores = self.root / "scores" / "b100-main" / "x1"
        self.scores.mkdir(parents=True)
        self.runtime = {"export": str(self.view), "export_sha256": "a" * 64, "kernel": "int8_mma:x1", "parts": 1,
                        "runtime": "provider", "selection_sha256": "s" * 64, "decoder_cuda_graphs_mode": "full_graph"}
        self.write(view_sha="a" * 64, recorded_sha="a" * 64)
        self.patches = [mock.patch.object(tk, "view_of", lambda model: self.view),
                        mock.patch.object(tk, "scores_dir", lambda model, mode: self.scores),
                        mock.patch.object(deployed, "require_wrapper", lambda: {}),
                        mock.patch.dict(os.environ, {}, clear=False)]
        for p in self.patches:
            p.start()
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    def tearDown(self) -> None:
        for p in reversed(self.patches):
            p.stop()

    def write(self, view_sha: str, recorded_sha: str) -> None:
        (self.view / "manifest.json").write_text(json.dumps({"sha256": view_sha}))
        model = {"name": "b100-main", "view": str(self.view), "view_sha256": recorded_sha}
        for part in ("med-dev-1", "stem-chem"):
            (self.scores / f"{part}.json").write_text(json.dumps({"config": {"model": model, "mode": "x1"}}))
        (self.scores / "runtime.json").write_text(json.dumps({"runtime": self.runtime}))

    def loaded(self, **change):
        net = mock.MagicMock()
        net._keab_x1 = {**self.runtime, "sources": {}, **change}
        return net

    def test_the_recorded_view_is_accepted(self) -> None:
        arm = hp.X1Arm("b100-main")
        arm.prepare(None, "cuda")
        self.assertEqual(arm.identity["view_sha256"], "a" * 64)

    def test_a_view_other_than_the_recorded_one_is_refused(self) -> None:
        self.write(view_sha="b" * 64, recorded_sha="a" * 64)
        with self.assertRaises(SystemExit):
            hp.X1Arm("b100-main").prepare(None, "cuda")

    def test_cublas_workspace_config_is_refused(self) -> None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        arm = hp.X1Arm("b100-main")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        with self.assertRaises(SystemExit):
            arm.prepare(None, "cuda")

    def test_a_runtime_other_than_the_recorded_one_is_refused(self) -> None:
        arm = hp.X1Arm("b100-main")
        arm.prepare(None, "cuda")
        closed = []
        with mock.patch.object(self.deployed, "load_x1", lambda view: self.loaded(export_sha256="c" * 64)), \
             mock.patch.object(self.deployed, "close", lambda net: closed.append(net)):
            with self.assertRaises(SystemExit):
                arm.load(torch.device("cuda"))
        self.assertEqual(len(closed), 1)

    def test_records_carry_stage_4s_configuration_and_identity(self) -> None:
        import hashlib
        arm = hp.X1Arm("b100-main")
        arm.prepare(2, "cuda")
        seen = {}

        def score_records(net, records, label, identity):
            seen.update(n=len(records), label=label, identity=identity)
            return {"set": label, "utterances": len(records), "records": []}

        with mock.patch.object(self.deployed, "load_x1", lambda view: self.loaded()), \
             mock.patch.object(self.deployed, "score_records", score_records):
            arm.load(torch.device("cuda"))
            result = arm.score("lectures-a", manifest_of("lectures-a"), 2)
        spec = {"name": "b100-main", "view": str(self.view), "view_sha256": "a" * 64}
        self.assertEqual(seen, {"n": 2, "label": "lectures-a",
                                "identity": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()})
        self.assertEqual(result["config"], self.tk.set_config(spec, "x1", "lectures-a", "eval-ext", 2))
        self.assertEqual(result["group"], "eval-ext")


class B0PrimeIdentityTest(unittest.TestCase):
    """B0' is stage 3's b100 model.nemo: its SHA-256 must equal model.json's and every stage 3 record's."""

    def setUp(self) -> None:
        relabel_paths.require_mount()
        import train_b0prime as tb
        self.tb = tb
        self.root = TMP / "b0p"
        if self.root.exists():
            shutil.rmtree(self.root)
        self.run_dir = self.root / "runs" / "b100"
        (self.run_dir / "domain").mkdir(parents=True)
        (self.run_dir / "eval-stem").mkdir(parents=True)
        (self.run_dir / "model.nemo").write_bytes(b"stand-in nemo")
        self.sha = evaluate.sha256_file(self.run_dir / "model.nemo")
        self.write(model_json=self.sha, records=self.sha)
        self.patch = mock.patch.object(tb, "run_dir", lambda name: self.root / "runs" / name)
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()

    def write(self, model_json: str, records: str) -> None:
        (self.run_dir / "model.json").write_text(json.dumps({"sha256": model_json, "run": "b100"}))
        for sub, part in (("domain", "med-dev-1"), ("eval-stem", "stem-chem")):
            (self.run_dir / sub / f"{part}.json").write_text(json.dumps({"config": {"weights_sha256": records}}))

    def test_the_recorded_model_is_accepted(self) -> None:
        b0p = hp.B0Prime()
        b0p.prepare(None, "cuda")
        self.assertEqual(b0p.identity["weights_sha256"], self.sha)

    def test_model_json_other_than_the_file_is_refused(self) -> None:
        self.write(model_json="0" * 64, records=self.sha)
        with self.assertRaises(SystemExit):
            hp.B0Prime().prepare(None, "cuda")

    def test_stage_3_records_of_another_model_are_refused(self) -> None:
        self.write(model_json=self.sha, records="0" * 64)
        with self.assertRaises(SystemExit):
            hp.B0Prime().prepare(None, "cuda")


if __name__ == "__main__":
    unittest.main()
