"""Contract tests (CPU): label cleaning's rules (cleaning plan "Layer 1", "Layer 2", Check 1): the tables' load-time
checks, whole-token longest-first matching, idempotent layer 1, disjoint layers, the dialect class and its quote
guard, option-only members, the capitalization rule, the read-time order, empty texts, the identity block and the
layer 2 decision rule. Store, artifact and runner contracts are in test_training_cleaning_store.py."""
from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from training import cleaning as C

FORMS = "html\tHTML\ndouble check\tdouble-check\nlithium ion\tlithium-ion\nnon gaap\tnon-GAAP\n"
VARIANTS = "dataset\tdata set\nOK\tokay\nall right\talright\nfree cash flow\tfree cashflow\ncash flow\tcashflow\n"
DIALECT = ("gonna\tgoing to\ni'ma\tI'm going to\n'cause\tbecause\n'em\tthem\ny'all\tyou all\n"
           "nothin'\tnothing\tlower\ngoin'\tgoing\tlower\n")
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
        for f in C.TABLE_FILES:
            text = (C.TABLES_DIR / f).read_text()
            self.assertNotIn("/mnt", text)
            self.assertNotIn("/home", text)


class Dialect(unittest.TestCase):
    def test_fixed_dialect_forms(self):
        self.assertEqual(layer1("'Cause y'all gonna get 'em, cuz"), "Because you all going to get them, cuz")
        self.assertEqual(layer1("I was goin' home with nothin'."), "I was going home with nothing.")

    def test_g_dropped_allowlist_is_lowercase_only(self):
        self.assertEqual(layer1("Nothin' at all, nothin'"), "Nothin' at all, nothing")

    def test_names_and_quotes_are_not_rewritten(self):
        tabs = tables(dialect=DIALECT + "dunkin'\tdunking\tlower\nmountain'\tmountaing\tlower\n")
        apply = lambda t: tabs.layer1.apply(t)[0]    # noqa: E731
        self.assertEqual(apply("Let's get Dunkin' Donuts"), "Let's get Dunkin' Donuts")
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
        tabs = tables(dialect=DIALECT + "robin'\trobing\tlower\n")
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


if __name__ == "__main__":
    unittest.main()
