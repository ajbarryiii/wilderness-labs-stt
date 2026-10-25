"""Contract tests: the pronunciation layer and GPU rendering (synthetic plan "Speech", Pronunciation;
"Resources", TTS on the GPU).

- The label never changes; the TTS input (tts_text) is the label with whole tokens replaced: lexicon tokens
  (case-insensitive, whole tokens only; spelled-out entries also take a plural "s"), and otherwise all-capital tokens
  of two or more letters spelled out by default. Everything else is unchanged.
- Clips record both the label and tts_text; the engine is given tts_text; round-trip WER and term agreement use the
  label.
- The report splits term agreement into acronym and other terms, per domain and pooled.
- `--device cuda` is selectable (with a declared VRAM) and CPU rendering still works.

Run from finetune/relabel:  ./python -m unittest discover -s tests -p 'test_synthetic_*' -v
"""
from __future__ import annotations

import hashlib
import json
import re
import unittest

from synthetic import layout, pronounce, roundtrip, tts
from synthetic.textgen import read_jsonl, write_jsonl
from synthetic_fixtures import env, make_split, roots, tone, write_clip

LEX = {
    "finance": "# a comment\n\ncapex\tcap ex\neps\ngaap\tgap\nebitda\tee bit dah\n",
    "software_engineering": "gpu\nsolid\tsolid\nmysql\tmy ess cue el\n",
    "chemistry": "",
    "mathematics": "qr\n",
}


def lexicon(files=LEX) -> pronounce.Lexicon:
    return pronounce.Lexicon.parse(files)


TOKEN = re.compile(r"[A-Za-z0-9]+")


def pieces(text: str) -> tuple[list[str], list[str]]:
    """(tokens, the separators around them)."""
    return TOKEN.findall(text), TOKEN.split(text)


class Rendering(unittest.TestCase):
    def test_lexicon_tokens_are_replaced_and_everything_else_is_kept(self):
        lex = lexicon()
        self.assertEqual(lex.render("Our capex rose, but EPS fell under non-GAAP rules.")[0],
                         "Our cap ex rose, but ee pee ess fell under non-gap rules.")
        self.assertEqual(lex.render("The GPU ran MySQL; SOLID code, QR steps.")[0],
                         "The jee pee you ran my ess cue el; solid code, cue ar steps.")

    def test_matching_is_case_insensitive_and_on_whole_tokens_only(self):
        lex = lexicon()
        for form in ("capex", "Capex", "CAPEX", "CapEx"):
            self.assertEqual(lex.render(f"the {form} plan")[0], "the cap ex plan")
        for form in ("eps", "EPS", "Eps"):
            self.assertEqual(lex.render(f"{form} grew")[0], "ee pee ess grew")
        unchanged = "Steps and epsilon, gaapish recapex, capexes, gpuX and the 3eps model."
        self.assertEqual(lex.render(unchanged)[0], unchanged)

    def test_spelled_entries_take_a_plural_and_respelled_entries_do_not(self):
        lex = lexicon()
        self.assertEqual(lex.render("two gpus and two GPUs")[0], "two jee pee yous and two jee pee yous")
        self.assertEqual(lex.render("capexs")[0], "capexs")

    def test_unlisted_all_capital_tokens_are_spelled_by_default(self):
        lex = lexicon()
        self.assertEqual(lex.render("GDP and the NYSE and two ETFs.")[0],
                         "jee dee pee and the en why ess ee and two ee tee efs.")
        self.assertEqual(lex.render("I saw A plan; Type II errors; McLafferty; iPhone; L'Hopital's rule.")[0],
                         "I saw A plan; Type II errors; McLafferty; iPhone; L'Hopital's rule.")
        self.assertEqual(lex.render("GAAP's rules")[0], "gap's rules")

    def test_the_tts_input_differs_from_the_label_only_by_token_substitutions(self):
        lex = lexicon()
        label = "Under GAAP, capex (not EPS!) drove the GPUs' cost: 3.5 bps, i.e. roughly half; CAGR rose."
        text, subs = lex.render(label)
        toks, seps = pieces(label)
        out = []
        last = 0
        for s in subs:
            self.assertEqual(label[s["start"]:s["end"]], s["token"])
            self.assertIn(s["token"], toks)
            out.append(label[last:s["start"]] + s["spoken"])
            last = s["end"]
        self.assertEqual("".join(out) + label[last:], text)
        # every untouched token and every separator is identical
        unchanged = [t for t in toks if t not in {s["token"] for s in subs}]
        self.assertEqual([t for t in TOKEN.findall(text) if t in unchanged], unchanged)
        self.assertEqual({s["rule"] for s in subs} <= {"lexicon", "lexicon_plural", "caps", "caps_plural"}, True)
        self.assertEqual(lex.render(label), (text, subs))

    def test_lexicon_files_are_validated_and_merged_across_domains(self):
        with self.assertRaises(ValueError):          # one token, two renderings
            lexicon({"finance": "eps\n", "software_engineering": "eps\teps\n"})
        with self.assertRaises(ValueError):          # not a single token
            lexicon({"finance": "non-gaap\tnon gap\n"})
        with self.assertRaises(ValueError):          # spelled entries are letters only
            lexicon({"finance": "s3\n"})
        same = lexicon({"finance": "eps\n", "software_engineering": "EPS\n"})
        self.assertEqual(same.render("eps")[0], "ee pee ess")

    def test_the_committed_lexicon_loads_and_covers_the_listed_all_capital_tokens(self):
        lex = pronounce.Lexicon.load()
        self.assertTrue(lex.record()["files"])
        from synthetic import vocab
        for terms in vocab.read_lists().values():
            for term in terms:
                for tok in TOKEN.findall(term):
                    if re.fullmatch(r"[A-Z]{2,}s?", tok):
                        self.assertTrue(lex.covers(tok.lower()), tok)


class AcronymTerms(unittest.TestCase):
    def test_a_term_is_an_acronym_term_by_lexicon_or_its_listed_capitals(self):
        lex = lexicon()
        self.assertTrue(pronounce.is_acronym_term("capex", ("capex",), lex))
        self.assertTrue(pronounce.is_acronym_term("non-GAAP earnings", ("non", "gaap", "earnings"), lex))
        self.assertTrue(pronounce.is_acronym_term("gpus", ("gpus",), lex))
        self.assertTrue(pronounce.is_acronym_term("EBITDA margin", ("ebitda", "margin"), lex))
        self.assertTrue(pronounce.is_acronym_term("NMR spectroscopy", ("nmr", "spectroscopy"), lex))  # listed capitals
        self.assertFalse(pronounce.is_acronym_term("solid solution", ("solid", "solution"), lex))      # said as itself
        self.assertFalse(pronounce.is_acronym_term("McLafferty rearrangement", ("mclafferty", "rearrangement"), lex))
        self.assertFalse(pronounce.is_acronym_term("Cauchy sequence", ("cauchy", "sequence"), lex))


class FakeEngine:
    alias = "syn-tts-a"

    def __init__(self, device="cpu", threads=4, declared_vram_gb=None):
        self.device, self.threads, self.declared_vram_gb = device, threads, declared_vram_gb
        self.texts = []

    def load(self):
        pass

    def synth(self, text, voice, seed):
        self.texts.append(text)
        return tone(1.0 + 0.05 * len(text.split())), layout.SAMPLE_RATE

    def info(self):
        return {"alias": self.alias, "device": self.device}


class Plan(unittest.TestCase):
    def sentences(self):
        labels = ["Our capex rose sharply this quarter.", "The EPS beat was small.", "Plain words only here.",
                  "We moved the GPU jobs.", "Non-GAAP margins widened again."]
        return [{"id": f"r-{i:06d}", "domain": layout.DOMAINS[i % 4], "label": s} for i, s in enumerate(labels)]

    def test_the_plan_keeps_every_label_and_records_its_tts_text(self):
        sents = self.sentences()
        voices = {"syn-tts-a": ["p225", "p226"], "syn-tts-b": ["p1"]}
        plan, summary = pronounce.apply(tts.assign(sents, voices, seed=3), lexicon())
        by_id = {s["id"]: s["label"] for s in sents}
        self.assertEqual({r["id"]: r["label"] for r in plan}, by_id)
        for r in plan:
            self.assertEqual(r["tts_text"], lexicon().render(r["label"])[0])
        self.assertEqual(summary["clips_changed"], sum(r["tts_text"] != r["label"] for r in plan))
        self.assertEqual(summary["clips_changed"], 4)
        self.assertEqual(summary["lexicon"], lexicon().record())
        # assignment (engine, voice, seed) is the same as without the layer
        self.assertEqual([{k: v for k, v in r.items() if k != "tts_text"} for r in plan],
                         tts.assign(sents, voices, seed=3))

    def test_rendering_gives_the_engine_tts_text_and_records_both_texts(self):
        sents = self.sentences()
        with roots() as t:
            d = t / "run"
            plan, _ = pronounce.apply(tts.assign(sents, {"syn-tts-a": ["p225"], "syn-tts-b": ["p1"]}, seed=3),
                                      lexicon())
            write_jsonl(d / "tts_plan.jsonl", plan)
            eng = FakeEngine()
            tts.render_chunk(d, eng, budget_minutes=10, log=lambda m: None)
            rows = read_jsonl(d / "tts" / "syn-tts-a" / "00000.jsonl")
            mine = [r for r in plan if r["engine"] == "syn-tts-a"]
            self.assertEqual(eng.texts, [r["tts_text"] for r in mine])
            self.assertEqual([(r["label"], r["tts_text"]) for r in rows], [(r["label"], r["tts_text"]) for r in mine])
            # a plan without tts_text is refused (never rendered from the label silently)
            write_jsonl(d / "tts_plan.jsonl", [{k: v for k, v in r.items() if k != "tts_text"} for r in plan])
            for p in (d / "tts" / "syn-tts-a").glob("*.jsonl"):
                p.unlink()
            with self.assertRaises(RuntimeError):
                tts.render_chunk(d, FakeEngine(), budget_minutes=10, log=lambda m: None)


class FailingEngine(FakeEngine):
    def synth(self, text, voice, seed):
        if len(self.texts) == 2:
            raise RuntimeError("CUDA failure (fake)")
        return super().synth(text, voice, seed)


class EngineFailure(unittest.TestCase):
    def test_a_failure_while_rendering_commits_no_block(self):
        sents = Plan().sentences()
        with roots() as t:
            d = t / "run"
            plan, _ = pronounce.apply(tts.assign(sents, {"syn-tts-a": ["p225"], "syn-tts-b": ["p1"]}, seed=3),
                                      lexicon())
            write_jsonl(d / "tts_plan.jsonl", plan)
            with self.assertRaises(RuntimeError):
                tts.render_chunk(d, FailingEngine(), budget_minutes=10, log=lambda m: None)
            self.assertFalse(list((d / "tts").glob("*/*.jsonl")))


class Teacher:
    """Transcribes the clips, in order, as the texts in `next`."""

    def __init__(self, texts):
        self.next = list(texts)

    def load(self):
        pass

    def plan(self, durations):
        return [[i] for i in range(len(durations))]

    def run(self, audios):
        return [(self.next.pop(0), False) for _ in audios]

    def info(self):
        return {}


class Scoring(unittest.TestCase):
    def world(self, t, split, lex, hyp_of):
        """A rendered run whose clips contain one targeted term each; T* says hyp_of(label, tts_text)."""
        d = t / "run"
        prompts, sents, plan, rt = [], [], [], []
        for i in range(8):
            dom = layout.DOMAINS[i % 4]
            term = split.targeted(dom)[0].term
            pid = f"r-{i:06d}"
            label = f"Today the {term} came up again in our meeting."
            prompts.append({"id": pid, "domain": dom, "register": "explanatory", "requested": [term]})
            sents.append({"id": pid, "domain": dom, "label": label, "requested": [term]})
            plan.append({"id": pid, "domain": dom, "label": label, "engine": layout.TTS_ENGINES[i % 2],
                         "voice": "p225", "seed": i})
        plan, summary = pronounce.apply(plan, lex)
        for i, r in enumerate(plan):
            path = write_clip(d / "audio" / f"{r['id']}.flac", tone(2.0))
            write_jsonl(d / "tts" / r["engine"] / f"{i:05d}.jsonl", [r | {
                "audio_filepath": str(path), "duration": 2.0, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}])
            hyp = hyp_of(r["label"], r["tts_text"])
            kept, reason, wer = roundtrip.decide(r["label"], hyp, False, 2.0)
            rt.append({"id": r["id"], "hyp": hyp, "hit_cap": False, "wer": wer, "kept": kept, "reason": reason})
        write_jsonl(d / "prompts.jsonl", prompts)
        write_jsonl(d / "sentences.jsonl", sents)
        write_jsonl(d / "tts_plan.jsonl", plan)
        (d / "pronunciation.json").write_text(json.dumps(summary))
        write_jsonl(d / "roundtrip" / "00000.jsonl", rt)
        (d / "hygiene.json").write_text(json.dumps({"generated": 8, "kept": 8, "reasons": {},
                                                    "heldout": {"provisional": True}}))
        layout.split_path().parent.mkdir(parents=True, exist_ok=True)
        layout.split_path().write_text(json.dumps(split.data))
        return d

    def test_round_trip_wer_is_against_the_label(self):
        lex = lexicon()
        label = "Our capex and EPS rose sharply in the second quarter of the year."
        spoken = lex.render(label)[0]
        self.assertTrue(roundtrip.decide(label, label, False, 3.0)[0])
        self.assertFalse(roundtrip.decide(label, spoken, False, 3.0)[0])
        with roots() as t:
            d = t / "run"
            path = write_clip(d / "audio" / "a.flac", tone(3.0))
            row = {"id": "r-000000", "domain": "finance", "label": label, "tts_text": spoken, "engine": "syn-tts-a",
                   "voice": "p225", "seed": 0, "audio_filepath": str(path), "duration": 3.0, "sha256": "x"}
            write_jsonl(d / "tts" / "syn-tts-a" / "00000.jsonl", [row])
            teacher = Teacher([label])
            roundtrip.transcribe_chunk(d, teacher, budget_minutes=10, log=lambda m: None)
            self.assertTrue(read_jsonl(d / "roundtrip" / "00000.jsonl")[0]["kept"])
            (d / "roundtrip" / "00000.jsonl").unlink()
            teacher.next = [spoken]
            roundtrip.transcribe_chunk(d, teacher, budget_minutes=10, log=lambda m: None)
            self.assertFalse(read_jsonl(d / "roundtrip" / "00000.jsonl")[0]["kept"])

    def test_clips_carry_label_and_tts_text_and_the_report_splits_acronym_terms(self):
        lists = {"chemistry": ["enthalpy", "titration"], "mathematics": ["eigenvalue", "determinant"],
                 "finance": ["capex", "EBITDA", "eps"], "software_engineering": ["deadlock", "hash table"]}
        split = make_split(lists, share=1.0)
        lex = lexicon()
        acronym = {e.phrase for e in split.entries if pronounce.is_acronym_term(e.term, e.phrase, lex)}
        self.assertEqual(acronym, {("capex",), ("ebitda",), ("eps",)})
        with roots() as t:
            # T* reproduces every term except acronyms, which it hears as their spoken rendering
            d = self.world(t, split, lex, lambda label, spoken: spoken)
            rep = roundtrip.finalize(d, split, listen=False, lexicon=lex)
            clips = read_jsonl(d / "clips.jsonl")
            for c in clips:
                self.assertEqual(c["tts_text"], lex.render(c["label"])[0])
            fin = rep["domains"]["finance"]["term_agreement_by_kind"]["all/targeted"]
            self.assertEqual((fin["acronym"]["occurrences"], fin["acronym"]["reproduced"]), (2, 0))
            self.assertEqual(fin["other"]["occurrences"], 0)
            tot = rep["total"]["term_agreement_by_kind"]["all/targeted"]
            self.assertEqual((tot["acronym"]["occurrences"], tot["acronym"]["reproduced"]), (2, 0))
            self.assertEqual((tot["other"]["occurrences"], tot["other"]["reproduced"]), (6, 6))
            self.assertEqual(rep["total"]["term_agreement"]["all/targeted"]["occurrences"], 8)
            self.assertEqual(rep["total"]["term_agreement"]["all/targeted"]["reproduced"], 6)
            self.assertEqual(rep["pronunciation"]["clips_changed"], 2)
            # unchanged when T* says the label
            d2 = self.world(t / "b", split, lex, lambda label, spoken: label)
            rep2 = roundtrip.finalize(d2, split, listen=False, lexicon=lex)
            self.assertEqual(rep2["total"]["term_agreement_by_kind"]["all/targeted"]["acronym"]["rate"], 1.0)

    def test_a_run_rendered_before_the_layer_reports_its_label_as_the_tts_input(self):
        split = make_split({"chemistry": ["enthalpy"], "mathematics": ["eigenvalue"], "finance": ["capex"],
                            "software_engineering": ["deadlock"]}, share=1.0)
        lex = lexicon()
        with roots() as t:
            d = self.world(t, split, lex, lambda label, spoken: label)
            for p in (d / "tts").glob("*/*.jsonl"):        # legacy records: no tts_text, no pronunciation.json
                write_jsonl(p, [{k: v for k, v in r.items() if k != "tts_text"} for r in read_jsonl(p)])
            (d / "pronunciation.json").unlink()
            rep = roundtrip.finalize(d, split, listen=True, lexicon=lex)
            self.assertTrue(all(c["tts_text"] == c["label"] for c in read_jsonl(d / "clips.jsonl")))
            self.assertTrue(rep["pronunciation"]["legacy_no_layer"])
            self.assertEqual(rep["pronunciation"]["clips_changed"], 0)
            # with the layer's summary present, a record without tts_text is refused
            (d / "pronunciation.json").write_text(json.dumps(pronounce.apply([], lex)[1]))
            with self.assertRaises(RuntimeError):
                roundtrip.finalize(d, split, listen=False, lexicon=lex)

    def test_finalize_refuses_a_lexicon_other_than_the_one_that_rendered_the_run(self):
        split = make_split({"chemistry": ["enthalpy"], "mathematics": ["eigenvalue"], "finance": ["capex"],
                            "software_engineering": ["deadlock"]}, share=1.0)
        with roots() as t:
            d = self.world(t, split, lexicon(), lambda label, spoken: label)
            other = lexicon(LEX | {"chemistry": "nmr\n"})
            with self.assertRaises(RuntimeError):
                roundtrip.finalize(d, split, listen=False, lexicon=other)


class Device(unittest.TestCase):
    def run_cli(self, argv, engines):
        import synthetic.__main__ as cli
        from synthetic import voicecheck
        saved = (dict(tts.ENGINES), cli.gpu_lock, voicecheck.require_pass, tts.render_chunk)
        made, locks = [], []

        def factory(*a, **k):
            e = FakeEngine(*a, **k)
            made.append(e)
            return e
        tts.ENGINES.update({a: factory for a in engines})
        cli.gpu_lock = lambda device, label, wait: (locks.append(device), cli._NoLock())[1]
        voicecheck.require_pass = lambda alias: None
        layout.require_mount, saved_mount = (lambda: None), layout.require_mount
        tts.render_chunk = lambda d, eng, budget, log: {"engine": eng.alias, "info": eng.info()}
        try:
            with env(RELABEL_DEPS=argv[argv.index("--engine") + 1]):
                code = cli.main(argv)
        finally:
            tts.ENGINES.clear()
            tts.ENGINES.update(saved[0])
            cli.gpu_lock, voicecheck.require_pass, tts.render_chunk = saved[1:]
            layout.require_mount = saved_mount
        return code, made, locks

    def test_cuda_is_selectable_with_a_declared_vram_and_cpu_still_works(self):
        with roots():
            code, made, locks = self.run_cli(["tts", "--smoke", "dev", "--engine", "syn-tts-a", "--device", "cuda",
                                              "--declared-vram-gb", "3.5"], ["syn-tts-a"])
            self.assertEqual(code, 0)
            self.assertEqual((made[0].device, made[0].declared_vram_gb), ("cuda", 3.5))
            self.assertEqual(locks, ["cuda"])
            code, made, locks = self.run_cli(["tts", "--smoke", "dev", "--engine", "syn-tts-a", "--device", "cpu",
                                              "--threads", "2"], ["syn-tts-a"])
            self.assertEqual(code, 0)
            self.assertEqual((made[0].device, made[0].threads, made[0].declared_vram_gb), ("cpu", 2, None))
            self.assertEqual(locks, ["cpu"])
            with self.assertRaises(SystemExit):          # the GPU path needs a VRAM declaration
                self.run_cli(["tts", "--smoke", "dev", "--engine", "syn-tts-a", "--device", "cuda"], ["syn-tts-a"])


if __name__ == "__main__":
    unittest.main()
