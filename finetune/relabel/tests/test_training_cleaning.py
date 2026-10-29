"""Contract tests (CPU): label cleaning's rules (cleaning plan "Layer 1", "Layer 2", Check 1): the tables' load-time
checks, whole-token longest-first matching, idempotent layer 1, disjoint layers, the dialect class and its quote
guard, option-only members, the capitalization rule, the read-time order, empty texts, the identity block and the
layer 2 decision rule; spelled-out acronyms and cleaning versions 1 and 2 (plan v5). Store, artifact and runner
contracts are in test_training_cleaning_store.py."""
from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from training import cleaning as C

FORMS = "html\tHTML\ndouble check\tdouble-check\nlithium ion\tlithium-ion\nnon gaap\tnon-GAAP\n"
VARIANTS = "dataset\tdata set\nOK\tokay\nall right\talright\nfree cash flow\tfree cashflow\ncash flow\tcashflow\n"
DIALECT = ("gonna\tgoing to\ni'ma\tI'm going to\n'cause\tbecause\n'em\tthem\ny'all\tyou all\n"
           "nothin'\tnothing\ngoin'\tgoing\n")
CHOICES = "maybe\tmay be\nset up\tsetup\nget*\tgit\tGit\nyou*\tyeah*\tya\nbecause*\tcos\ngap\tGAAP2\n"


def tables(forms=FORMS, variants=VARIANTS, dialect=DIALECT, choices=CHOICES) -> C.Tables:
    return C.Tables({"forms.tsv": forms, "variants.tsv": variants, "dialect.tsv": dialect, "choices.tsv": choices})


class Artifact:
    """An in-memory choices artifact: {(label set, id): [(span, start, end, original, set name, chosen)]}."""

    def __init__(self, tabs: C.Tables, picks: dict, sha: str = "x" * 64):
        self.manifest = {"version": C.VERSION, "tables_sha256": tabs.sha256, "template_sha256": C.TEMPLATE_SHA256,
                         "threshold": C.THRESHOLD, "contract_sha256": "c" * 64, "file_sha256": sha}
        self.rows, self.dir, self.file_sha256 = picks, Path("mem"), sha


def spec_with(tabs: C.Tables, text: str, chosen: list, item="i1", label_set="T*") -> C.Cleaning:
    spans = tabs.spans(text)
    rows = [(sp.index, sp.start, sp.end, sp.original, sp.set.name, c) for sp, c in zip(spans, chosen)]
    return C.Cleaning(tabs, Artifact(tabs, {(label_set, item): rows}))


L1 = None


def layer1(text: str) -> str:
    global L1
    if L1 is None:
        L1 = tables()
    return L1.layer1.apply(text)[0]


class Layer1Rules(unittest.TestCase):
    def test_case_insensitive_whole_token(self):
        self.assertEqual(layer1("the Html page and xhtml and html5"), "the HTML page and xhtml and html5")
        self.assertEqual(layer1("index.html and config.HTML."), "index.html and config.HTML.")   # a dotted token
        self.assertEqual(layer1("it's html's"), "it's html's")                                 # internal apostrophe
        self.assertEqual(layer1("éhtml htmlé html"), "éhtml htmlé HTML")                       # letters of any script

    def test_longest_match_first_and_no_overlap(self):
        self.assertEqual(layer1("free cashflow and cashflow"), "free cash flow and cash flow")
        self.assertEqual(layer1("non gaap"), "non-GAAP")

    def test_forms_change_only_case_and_hyphens(self):
        with self.assertRaises(ValueError):
            tables(forms="colour\tcolor\n")
        with self.assertRaises(ValueError):
            tables(forms="htm\tHTML\n")

    def test_no_diacritic_folding(self):
        for t in ("Schrödinger's equation", "L'Hôpital", "café html"):
            out = layer1(t)
            self.assertEqual([c for c in out if ord(c) > 127], [c for c in t if ord(c) > 127])

    def test_capitalization_rule(self):
        self.assertEqual(layer1("Double check it"), "Double-check it")
        self.assertEqual(layer1("Gonna go"), "Going to go")
        self.assertEqual(layer1("GONNA go"), "Going to go")
        self.assertEqual(layer1("Okay, okay, ok, OK"), "OK, OK, OK, OK")                # fixed case as listed
        self.assertEqual(layer1("I'ma go, i'ma go"), "I'm going to go, I'm going to go")
        self.assertEqual(layer1("Data set"), "Dataset")
        self.assertEqual(layer1("DATASET and Dataset"), "DATASET and Dataset")          # canonical casing kept

    def test_idempotent(self):
        rng = random.Random(7)
        words = ["html", "Html", "double", "check", "gonna", "Gonna", "'cause", "'em", "y'all", "nothin'", "goin'",
                 "okay", "OK", "alright", "all", "right", "data", "set", "dataset", "free", "cash", "flow", "cashflow",
                 "non", "gaap", "lithium", "ion", "i'ma", "'", ",", ".", "the", "Nothin'", "'Robin'"]
        for _ in range(3000):
            t = " ".join(rng.choice(words) for _ in range(rng.randint(1, 12)))
            once = layer1(t)
            self.assertEqual(layer1(once), once, t)

    def test_committed_tables_load_and_are_idempotent_on_their_outputs(self):
        tabs = C.Tables.load()
        for e in tabs.layer1.entries.values():
            self.assertEqual(tabs.layer1.apply(e.output)[0], e.output)
        self.assertGreater(len(tabs.sets), 10)
        # nouns that are not dialect spellings (cleaning plan, architect decision): left as written
        for t in ("That shot was a gimme.", "That is my cuz."):
            self.assertEqual(tabs.layer1.apply(t)[0], t)
        for f in C.TABLE_FILES:
            text = (C.TABLES_DIR / f).read_text()
            self.assertNotIn("/mnt", text)
            self.assertNotIn("/home", text)


class Dialect(unittest.TestCase):
    def test_fixed_dialect_forms(self):
        self.assertEqual(layer1("'Cause y'all gonna get 'em, cuz"), "Because you all going to get them, cuz")
        self.assertEqual(layer1("I was goin' home with nothin'."), "I was going home with nothing.")

    def test_dialect_forms_match_in_any_case(self):
        # Contract change (cleaning plan v4): the lowercase-only flag of the dropped g-dropped allowlist is gone.
        self.assertEqual(layer1("Nothin' at all, nothin'"), "Nothing at all, nothing")
        with self.assertRaises(ValueError):
            tables(dialect=DIALECT + "dunkin'\tdunking\tlower\n")

    def test_quoted_words_are_not_rewritten(self):
        tabs = tables(dialect=DIALECT + "mountain'\tmountaing\n")
        apply = lambda t: tabs.layer1.apply(t)[0]    # noqa: E731
        self.assertEqual(apply("the word 'mountain' here"), "the word 'mountain' here")
        self.assertEqual(apply("she said 'I was goin' and left"), "she said 'I was goin' and left")
        self.assertEqual(apply("he wrote 'nothin'"), "he wrote 'nothin'")
        self.assertEqual(apply("'Cause I was goin' home"), "Because I was going home")   # 'Cause is no quote

    def test_leading_apostrophe_forms_keep_quote_marks(self):
        self.assertEqual(layer1("the word 'em' means"), "the word 'em' means")
        self.assertEqual(layer1("say 'gonna' twice"), "say 'going to' twice")
        self.assertEqual(layer1("Get 'em."), "Get them.")
        self.assertEqual(layer1("tell'em"), "tell'em")

    def test_a_quoted_word_keeps_both_quote_marks(self):
        tabs = tables(dialect=DIALECT + "robin'\trobing\n")
        for t in ("I said 'robin' twice", "'robin'", "x 'nothin' y", "a 'goin' b 'em' c"):
            out, edits = tabs.layer1.apply(t)
            self.assertEqual(out, t)
            self.assertEqual(edits, [])


class Disjoint(unittest.TestCase):
    def test_member_equal_to_a_form_is_refused(self):
        with self.assertRaises(ValueError):
            tables(choices=CHOICES + "html\tHTMLX\n")

    def test_form_containing_a_member_is_refused(self):
        with self.assertRaises(ValueError):
            tables(forms=FORMS + "set up box\tset-up box\n")

    def test_member_in_two_sets_is_refused(self):
        with self.assertRaises(ValueError):
            tables(choices=CHOICES + "maybe\tmaybee\n")

    def test_bad_sets_are_refused(self):
        for bad in ("solo\n", "a\ta\n", "x*\ty*\n", "git it\t\n"):
            with self.assertRaises(ValueError, msg=bad):
                tables(choices=bad)

    def test_written_form_in_two_tables_is_refused(self):
        with self.assertRaises(ValueError):
            tables(dialect=DIALECT + "html\thypertext\n")

    def test_layer1_never_touches_a_layer2_span(self):
        tabs = tables(forms="be the\tbe-the\n")
        spec = spec_with(tabs, "it may be the end", [None])
        self.assertEqual(spec.clean("it may be the end", "i1", "T*"), "it may be the end")
        self.assertEqual(tabs.layer1.apply("it will be the end")[0], "it will be-the end")


class Spans(unittest.TestCase):
    def setUp(self):
        self.t = tables()

    def test_whole_token_case_insensitive_longest_first(self):
        sp = self.t.spans("Maybe we set up the SETUP, maybee, may be")
        self.assertEqual([(s.original, s.set.name) for s in sp],
                         [("Maybe", "maybe|may be"), ("set up", "set up|setup"), ("SETUP", "set up|setup"),
                          ("may be", "maybe|may be")])
        self.assertEqual([s.index for s in sp], [0, 1, 2, 3])

    def test_option_only_members_start_no_span(self):
        self.assertEqual(self.t.spans("you get it, yeah, because"), [])
        got = self.t.spans("ya git it cos Git")
        self.assertEqual([s.original for s in got], ["ya", "git", "cos", "Git"])
        self.assertEqual(got[1].set.options, ("get", "git", "Git", C.CANT_TELL))
        self.assertEqual(self.t.spans("ya'll"), [])

    def test_prompt_shows_the_label_the_marked_span_and_lettered_options(self):
        text = "you git me that money"
        sp = self.t.spans(text)[0]
        user = C.prompt_user(text, sp.start, sp.end, sp.set.options)
        self.assertIn("you [[git]] me that money", user)
        for line in ("A. get", "B. git", "C. Git", f"D. {C.CANT_TELL}"):
            self.assertIn(line, user)

    def test_dedup_key(self):
        k = C.prompt_key("a maybe", 2, 7, ("maybe", "may be", C.CANT_TELL))
        self.assertEqual(k, C.prompt_key("a maybe", 2, 7, ("maybe", "may be", C.CANT_TELL)))
        self.assertNotEqual(k, C.prompt_key("b maybe", 2, 7, ("maybe", "may be", C.CANT_TELL)))
        self.assertNotEqual(k, C.prompt_key("a maybe", 2, 6, ("maybe", "may be", C.CANT_TELL)))


class Decide(unittest.TestCase):
    OPTS = ("maybe", "may be", C.CANT_TELL)

    def test_threshold_and_cant_tell(self):
        self.assertEqual(C.decide([0.9, 0.05, 0.05], self.OPTS), "maybe")
        self.assertIsNone(C.decide([0.89, 0.06, 0.05], self.OPTS))
        self.assertIsNone(C.decide([0.02, 0.03, 0.95], self.OPTS))
        self.assertEqual(C.decide([1.8, 0.1, 0.1], self.OPTS), "maybe")              # renormalized

    def test_bad_probabilities(self):
        with self.assertRaises(ValueError):
            C.decide([0.5, 0.5], self.OPTS)


class ReadTime(unittest.TestCase):
    def setUp(self):
        self.t = tables()

    def test_layer2_then_layer1_with_capitalization_and_fixed_case(self):
        text = "Maybe the data set, git it and gap"
        spec = spec_with(self.t, text, ["may be", "get", "GAAP2"])
        out, edits = spec.edits(text, "i1", "T*")
        self.assertEqual(out, "May be the dataset, get it and GAAP2")
        self.assertEqual([(e.layer, e.table, e.original, e.replacement) for e in edits],
                         [(2, "choices", "Maybe", "May be"), (1, "variants", "data set", "dataset"),
                          (2, "choices", "git", "get"), (2, "choices", "gap", "GAAP2")])
        for e in edits:                                          # coordinates in the original and the result
            self.assertEqual(text[e.start:e.end], e.original)
            self.assertEqual(out[e.out_start:e.out_end], e.replacement)
        self.assertEqual(spec_with(self.t, "Git it", ["get"]).clean("Git it", "i1", "T*"), "Get it")
        self.assertEqual(spec_with(self.t, "Git it", ["git"]).clean("Git it", "i1", "T*"), "Git it")

    def test_unchosen_spans_are_left_as_written(self):
        text = "maybe GIT"
        self.assertEqual(spec_with(self.t, text, [None, None]).clean(text, "i1", "T*"), text)
        self.assertEqual(C.Cleaning(self.t).clean(text, "i1", "T*"), text)              # no artifact: layer 1 only

    def test_outside_span_bytes_unchanged(self):
        rng = random.Random(11)
        words = ["maybe", "may", "be", "set", "up", "setup", "git", "ya", "cos", "html", "gonna", ",", "x", "'em"]
        for n in range(500):
            text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 9)))
            spans = self.t.spans(text)
            picks = [rng.choice(list(sp.set.members) + [None]) for sp in spans]
            out, edits = spec_with(self.t, text, picks, item=f"i{n}").edits(text, f"i{n}", "T*")
            rebuilt, pos = [], 0
            for e in sorted(edits, key=lambda e: e.start):
                rebuilt.append(text[pos:e.start])
                rebuilt.append(e.replacement)
                pos = e.end
            rebuilt.append(text[pos:])
            self.assertEqual("".join(rebuilt), out, text)
            self.assertTrue(out, text)

    def test_replacement_must_be_a_member_of_the_set(self):
        spec = spec_with(self.t, "maybe so", ["perhaps"])
        with self.assertRaises(C.ArtifactMismatch):
            spec.clean("maybe so", "i1", "T*")

    def test_recorded_original_and_spans_must_match(self):
        spec = spec_with(self.t, "maybe so", ["may be"])
        with self.assertRaises(C.ArtifactMismatch):
            spec.clean("Maybe so", "i1", "T*")                 # the label differs at [start, end)
        with self.assertRaises(C.ArtifactMismatch):
            spec.clean("so maybe", "i1", "T*")                 # other spans
        with self.assertRaises(C.ArtifactMismatch):
            spec.clean("maybe so", "i1", "B0")                 # no rows for this label set
        self.assertEqual(spec.clean("nothing here", "i9", "T*"), "nothing here")   # no spans, no rows: fine

    def test_an_empty_label_is_checked_against_the_artifact(self):
        rows = {("B0", "e1"): [(0, 0, 5, "maybe", "maybe|may be", None)]}
        with self.assertRaises(C.ArtifactMismatch):
            C.Cleaning(self.t, Artifact(self.t, rows)).clean("", "e1", "B0")
        self.assertEqual(C.Cleaning(self.t, Artifact(self.t, rows)).clean("", "e2", "B0"), "")

    def test_empty_text_stays_empty_and_non_empty_stays_non_empty(self):
        spec = C.Cleaning(self.t)
        self.assertEqual(spec.clean("", "i1", "B0"), "")
        for t in ("gonna", "'em", "okay", "maybe", "a"):
            self.assertTrue(spec.clean(t, "i1", "T*"))

    def test_both_label_sets_are_cleaned_identically(self):
        spec = C.Cleaning(self.t)
        for t in ("Gonna get the data set, okay", "nothin' html"):
            self.assertEqual(spec.clean(t, "i", "T*"), spec.clean(t, "i", "B0"))


class Identity(unittest.TestCase):
    def test_identity_block_changes_with_any_table_and_the_artifact(self):
        base = C.Cleaning(tables()).identity()
        self.assertEqual(set(base), {"version", "tables_sha256", "choices_sha256"})
        self.assertEqual(base["version"], C.VERSION)
        self.assertIsNone(base["choices_sha256"])
        for key in ("forms", "variants", "dialect", "choices"):
            kw = {"forms": FORMS, "variants": VARIANTS, "dialect": DIALECT, "choices": CHOICES}
            kw[key] += "# a comment\n"
            self.assertNotEqual(C.Cleaning(tables(**kw)).identity()["tables_sha256"], base["tables_sha256"], key)
        t = tables()
        a = C.Cleaning(t, Artifact(t, {}, sha="a" * 64)).identity()
        b = C.Cleaning(t, Artifact(t, {}, sha="b" * 64)).identity()
        self.assertNotEqual(a, b)
        self.assertEqual(a["tables_sha256"], base["tables_sha256"])

    def test_artifact_of_other_tables_template_or_threshold_is_refused(self):
        t = tables()
        for key, value in (("tables_sha256", "0" * 64), ("template_sha256", "0" * 64), ("threshold", 0.5),
                           ("version", 0)):
            art = Artifact(t, {})
            art.manifest[key] = value
            with self.assertRaises(C.ArtifactMismatch, msg=key):
                C.Cleaning(t, art)

    def test_artifact_of_another_run_contract_is_refused(self):
        t = tables()
        spec = C.Cleaning(t, Artifact(t, {}))
        spec.bind_run("c" * 64)
        with self.assertRaises(C.ArtifactMismatch):
            spec.bind_run("d" * 64)

    def test_tables_load_from_a_directory(self):
        with tempfile.TemporaryDirectory() as d:
            for f, text in zip(C.TABLE_FILES, (FORMS, VARIANTS, DIALECT, CHOICES)):
                (Path(d) / f).write_text(text)
            self.assertEqual(C.Tables.load(d).sha256, tables().sha256)
            (Path(d) / "dialect.tsv").unlink()
            with self.assertRaises(RuntimeError):
                C.Tables.load(d)


ACR = None


def v2(text: str) -> str:
    """Cleaning version 2 without an artifact (layer 1 with spelled-out acronyms)."""
    global ACR
    if ACR is None:
        ACR = C.Cleaning(tables(forms=FORMS + "macos\tmacOS\n"), version=2)
    return ACR.clean(text, "i1", "T*")


class Acronyms(unittest.TestCase):
    """Cleaning plan v5 "Spelled-out acronyms": layer 1 of cleaning version 2."""

    def test_spelled_out_runs_become_the_compact_acronym(self):
        for t in ("an A. P. I. call", "an A.P.I. call", "an A P I call", "an A. P.I. call"):
            self.assertEqual(v2(t), "an API call", t)
        self.assertEqual(v2("A. P. I. calls"), "API calls")
        self.assertEqual(v2("the F B I, the C. I. A. and N.A.S.A.'s rover"), "the FBI, the CIA and NASAs rover")
        self.assertEqual(v2("two T. V. sets"), "two TV sets")              # two letters dotted with a space

    def test_plural(self):
        for t in ("two G. P. U. s here", "two G.P.U.s here", "two G P U's here", "two G P Us here",
                  "two G P U s here", "two G.P.U.'s here"):
            self.assertEqual(v2(t), "two GPUs here", t)
        self.assertEqual(v2("the G.P.U.s."), "the GPUs.")                   # the sentence's period is not the run's
        # after a dotted run, a capital with an attached "s" is a word starting a sentence, not a plural
        for t in ("hard R. Is that why", "the R. As such", "the A. B. Us and"):
            self.assertEqual(v2(t), t, t)
        self.assertEqual(v2("the G.P.U's"), "the GPUs")

    def test_final_period(self):
        self.assertEqual(v2("…the A. P. I."), "…the API.")                  # the label's last character: kept
        self.assertEqual(v2("the A.P.I."), "the API.")
        self.assertEqual(v2("the A.P.I. and"), "the API and")              # else dropped with the others
        self.assertEqual(v2("the A. P. I., then"), "the API, then")
        self.assertEqual(v2("the A P I."), "the API.")
        self.assertEqual(v2("the A P I. then"), "the API. then")           # undotted: the period is the sentence's

    def test_exclusions_are_left_as_written(self):
        for t in ("the U.S. economy", "in D.C.", "the U.S", "300 B.C. and 9 A.M.",     # two letters, dotted, no space
                  "the T V is on", "N I 2 R", "grades A B and", "a T V.",            # two letters, undotted
                  "I I think", "I. I. I said",                                          # only "I": a stutter
                  "J. A. Mellon wrote", "J.R.R. Tolkien", "J A Mellon", "by J. R. R. O'Brien",  # initials
                  "A-V-E-N-S-A-R", "it's A-P-I"):                                       # hyphens are not separators
            self.assertEqual(v2(t), t, t)

    def test_exclusions_take_precedence(self):
        self.assertEqual(v2("…the A. P. I. We went"), "…the A. P. I. We went")   # initials, a stated loss
        self.assertEqual(v2("the A P I Gateway"), "the A P I Gateway")
        self.assertEqual(v2("the A P I API"), "the API API")               # an all-capitals word is no name

    def test_whole_token_boundaries(self):
        for t in ("xA B C", "e.A B C", "an a p i call", "it's É P I", "the A B C-D", "co-A B C"):
            self.assertEqual(v2(t), t, t)
        self.assertEqual(v2("the A B C D1 way"), "the ABC D1 way")         # "D1" is not a single letter
        self.assertEqual(v2("the A P Is"), "the APIs")
        self.assertEqual(v2("the A P I's docs"), "the APIs docs")          # a possessive reads as a plural
        self.assertEqual(v2("(A P I)"), "(API)")
        self.assertEqual(v2("A P I"), "API")

    def test_before_the_forms_table(self):
        self.assertEqual(v2("the H T M L page"), "the HTML page")
        self.assertEqual(v2("on M A C O S today"), "on macOS today")       # compact, then the table's casing
        out, edits = ACR.edits("on M A C O S, h t m l", "i1", "T*")
        self.assertEqual(out, "on macOS, h t m l")                         # lowercase letters are no run
        self.assertEqual([(e.layer, e.table, e.group, e.original, e.replacement) for e in edits],
                         [(1, "acronyms", "spelled-out", "M A C O S", "macOS")])

    def test_edit_coordinates_and_identical_label_sets(self):
        text = "Maybe the A. P. I., gonna use G P U s and the H T M L."
        tabs = tables()
        spec = spec_with(tabs, text, ["may be"])
        spec = C.Cleaning(tabs, spec.choices, version=2)
        out, edits = spec.edits(text, "i1", "T*")
        self.assertEqual(out, "May be the API, going to use GPUs and the HTML.")
        self.assertEqual([(e.layer, e.table) for e in edits],
                         [(2, "choices"), (1, "acronyms"), (1, "dialect"), (1, "acronyms"), (1, "acronyms")])
        for e in edits:
            self.assertEqual(text[e.start:e.end], e.original)
            self.assertEqual(out[e.out_start:e.out_end], e.replacement)
        b0 = C.Cleaning(tabs, version=2)
        for t in ("the A. P. I. and G P U s", "J. A. Mellon and the C P U"):
            self.assertEqual(b0.clean(t, "i", "T*"), b0.clean(t, "i", "B0"))

    def test_layer1_never_touches_a_layer2_span(self):
        tabs = tables(choices=CHOICES + "x y\txy\n")
        text = "the X Y Z here"
        spec = C.Cleaning(tabs, spec_with(tabs, text, [None]).choices, version=2)
        self.assertEqual(spec.clean(text, "i1", "T*"), text)
        self.assertEqual(C.Cleaning(tabs, version=2).clean("the W Y Z here", "i1", "T*"), "the WYZ here")

    def test_idempotent_never_empty(self):
        rng = random.Random(5)
        pieces = ["A", "P", "I", "G", "U", "T", "V", "J", "M", "s", "'s", "Mellon", "API", "the", "U.S.", "html",
                  "gonna", "x", "1", "-", ",", "é", "macos", "O'Brien", "okay", "Okay", "lithium", "ion", "Is", "As"]
        seps = [" ", ". ", ".", "", "-", "  "]
        for _ in range(4000):
            t = "".join(rng.choice(pieces) + rng.choice(seps) for _ in range(rng.randint(1, 10))).strip()
            once = v2(t)
            self.assertEqual(v2(once), once, t)
            self.assertEqual(bool(once), bool(t), t)
        self.assertEqual(v2(""), "")

    def test_committed_table_outputs_are_left_by_the_acronym_rule(self):
        tabs = C.Tables.load()
        spec = C.Cleaning(tabs, version=2)
        for e in tabs.layer1.entries.values():
            self.assertEqual(spec.clean(e.output, "i", "T*"), e.output)
        self.assertEqual(spec.clean("the H T M L and the P T S D", "i", "T*"), "the HTML and the PTSD")
        # the initials exclusion reads the next word as the tables write it, so a second pass agrees (review r1)
        for t in ("the A P I uv visible", "the A P I r tree", "J. A. Gaussian", "the A P I gaussian"):
            once = spec.clean(t, "i", "T*")
            self.assertEqual(spec.clean(once, "i", "T*"), once, t)
        self.assertEqual(spec.clean("the A P I uv visible", "i", "T*"), "the A P I UV-visible")
        # a table edit that splits an excluded chain exposes a run, cleaned in the same call (review r2)
        text = "P T S D. M R Is"
        out, edits = spec.edits(text, "i", "T*")
        self.assertEqual(out, "PTSD. MRIs")
        self.assertEqual(spec.clean(out, "i", "T*"), out)
        self.assertEqual([(e.table, e.original, e.replacement) for e in edits],
                         [("variants", "P T S D", "PTSD"), ("acronyms", "M R Is", "MRIs")])
        for e in edits:
            self.assertEqual((text[e.start:e.end], out[e.out_start:e.out_end]), (e.original, e.replacement))
        rng = random.Random(9)
        pieces = ["P", "T", "S", "D", "M", "R", "I", "Is", "s", "uv", "visible", "r", "tree", "x", "ray", "X", "H",
                  "html", "gonna", "Mellon", "chi", "squared", "C", "lithium", "ion", "gpu", "G"]
        for _ in range(3000):
            t = "".join(rng.choice(pieces) + rng.choice([" ", ". ", ".", "-"]) for _ in range(rng.randint(1, 9)))
            once, edits = spec.edits(t.strip(), "i", "T*")
            self.assertEqual(spec.clean(once, "i", "T*"), once, t)
            pos, rebuilt = 0, []
            for e in edits:
                rebuilt += [t.strip()[pos:e.start], e.replacement]
                pos = e.end
            self.assertEqual("".join(rebuilt) + t.strip()[pos:], once, t)
        for t in ("the A P I okay", "the A P I Okay"):                  # test tables: okay -> OK, all capitals
            self.assertEqual(v2(t), "the API OK", t)
            self.assertEqual(v2(v2(t)), "the API OK", t)

    def test_an_acronym_edit_keeps_its_run_as_written(self):
        spec = C.Cleaning(tables(variants=VARIANTS + "chest x-ray\tchest xray\n"), version=2)
        out, edits = spec.edits("a CHEST X R A Y today", "i", "T*")
        self.assertEqual(out, "a Chest x-ray today")
        self.assertEqual([(e.table, e.original, e.replacement, e.run) for e in edits],
                         [("acronyms", "CHEST X R A Y", "Chest x-ray", "X R A Y")])
        self.assertEqual(spec.edits("the G. P. U. s", "i", "T*")[1][0].run, "G. P. U. s")
        self.assertIsNone(spec.edits("Gonna", "i", "T*")[1][0].run)


class Versions(unittest.TestCase):
    """Cleaning plan v5 "Version": version 2 adds the acronym rule; version 1 (stages 2b and 2c) is unchanged."""

    def test_versions(self):
        self.assertEqual(C.VERSIONS, (1, 2))
        self.assertEqual(C.VERSION, 2)
        for bad in (0, 3):
            with self.assertRaises(ValueError):
                C.Cleaning(tables(), version=bad)

    def test_version_1_is_layer1_without_the_acronym_rule(self):
        t = tables()
        v1 = C.Cleaning(t, version=1)
        self.assertEqual(v1.clean("an A. P. I. call, gonna", "i", "T*"), "an A. P. I. call, going to")
        rng = random.Random(3)
        words = ["A", "P.", "I.", "G", "U", "s", "html", "gonna", "okay", "data", "set", "the", ",", "'em", "T", "V"]
        for _ in range(2000):
            text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 10)))
            self.assertEqual(v1.edits(text, "i", "T*"), t.layer1.apply(text), text)

    def test_identity_differs_between_versions(self):
        t = tables()
        a, b = C.Cleaning(t, version=1).identity(), C.Cleaning(t, version=2).identity()
        self.assertEqual((a["version"], b["version"]), (1, 2))
        self.assertEqual(a["tables_sha256"], b["tables_sha256"])
        self.assertNotEqual(a, b)
        self.assertEqual(C.Cleaning(t).identity(), b)                      # the default is the latest version

    def test_an_artifact_of_the_other_version_is_refused(self):
        t = tables()
        for made, read in ((1, 2), (2, 1)):
            art = Artifact(t, {})
            art.manifest["version"] = made
            with self.assertRaises(C.ArtifactMismatch):
                C.Cleaning(t, art, version=read)
            self.assertEqual(C.Cleaning(t, art, version=made).identity()["version"], made)

    def test_directories_per_version(self):
        from unittest import mock
        with mock.patch("labeling.layout.relabel_root", lambda: Path("/R")):
            self.assertEqual(C.production_dir("main-f3", 1), Path("/R/cleaning/main-f3/v1"))
            self.assertEqual(C.production_dir("main-f3", 2), Path("/R/cleaning/main-f3/v2"))
            self.assertEqual(C.audit_dir(1), Path("/R/cleaning/audit"))       # the version 1 sheets stay put
            self.assertEqual(C.audit_dir(2), Path("/R/cleaning/audit/v2"))
            with mock.patch.object(C.Cleaning, "load", lambda tables_dir=None, choices_dir=None, version=None:
                                   (choices_dir, version)):
                self.assertEqual(C.production("main-f3"), (Path("/R/cleaning/main-f3/v1"), 1))   # stages 2b, 2c
                self.assertEqual(C.production("main-f3", 2), (Path("/R/cleaning/main-f3/v2"), 2))


if __name__ == "__main__":
    unittest.main()
