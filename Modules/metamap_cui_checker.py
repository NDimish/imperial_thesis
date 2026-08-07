"""Pure, engine-only CUI set-difference checker -- the "just normalize both
documents to UMLS Concept Unique Identifiers and diff the two sets" baseline
that MetaMapLite and QuickUMLS both exist to make fast and deterministic.

Unlike Modules/medspacy_umls_checker.py, this checker does NOT run medspaCy's
ConText pipe over the matched spans -- no negation, no uncertainty, no family
history, no hypothetical filtering, no cross-turn question/answer carry-over.
It only asks "does this CUI appear anywhere in the other document at all",
per-line, and reports omission/hallucination on that basis alone. That's a
deliberate scope cut, not an oversight -- it isolates how much of
MedspacyUmlsChecker's precision comes from the ConText layer specifically
versus from concept normalization alone, and it mirrors what a MetaMapLite-
or QuickUMLS-only pipeline (with no assertion-status layer bolted on) would
actually produce in practice. See the accompanying HTML write-up for the
full comparison.

Backend selection: MetaMapLite (NLM's official lightweight MetaMap
reimplementation, Java-based, run as a local batch process -- see
MetaMapLiteClient below) is tried first if METAMAPLITE_INSTALL_DIR is
configured. If it isn't (no local install, no UMLS license wired up here),
this falls back automatically to the project's shared QuickUMLS matcher
(Medical condensor/umls_matching.py, already configured via
QUICKUMLS_INSTALL_DIR and used by every condenser). Both are "fast,
deterministic CUI lookup engines built by/for NLM" per the same UMLS
Metathesaurus, so treating them as interchangeable backends behind one
checker -- rather than writing two near-identical modules -- matches how
they're actually used in practice: whichever one is locally installed.
Falling back rather than failing hard also means this checker is runnable
in this repo out of the box (QuickUMLS is already configured here), while
still being the "real" MetaMapLite path for anyone who sets up the install.

Raises RuntimeError from __init__ if NEITHER backend is configured, so
main.py's load_checker_modules() try/except skips this module with a
message instead of crashing the whole run -- the same pattern already used
for AlignScoreChecker (missing checkpoint) and QuickUMLSCondenser (missing
QUICKUMLS_INSTALL_DIR).
"""
import glob
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from base import split_turns  # noqa: E402 -- path insert must run first
from umls_matching import get_matcher, is_common_word  # noqa: E402 -- path insert must run first

from Modules.base import CheckerModule

# Set this to a local MetaMapLite install directory (the one containing
# target/ and data/) to use MetaMapLite instead of QuickUMLS. Requires a UMLS
# Metathesaurus license and a local MetaMapLite build:
#   1. Register for a UMLS license: https://www.nlm.nih.gov/research/umls/
#   2. Download MetaMapLite + a UMLS data index:
#      https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/run-locally/MetaMapLite.html
#   3. `mvn clean install` to build target/metamaplite-*-standalone.jar
# Left unset here -- this repo already has QuickUMLS configured (see
# QUICKUMLS_INSTALL_DIR in Medical condensor/umls_matching.py), so this
# checker falls back to that backend automatically (see module docstring).
METAMAPLITE_INSTALL_DIR = os.environ.get("METAMAPLITE_INSTALL_DIR")

# Copied from Modules/medspacy_umls_checker.py's JUNK_CONCEPT_DENYLIST (itself
# copied from Modules/old/concept_checker.py) -- real, validated exact
# (similarity=1.0) UMLS/generic-English collisions in this dataset (e.g. this
# project's own "Plan:" SOAP heading matching a real UMLS concept). Duplicated
# here rather than imported so this checker stays independent of
# medspacy_umls_checker.py's heavier medspacy/spacy import chain -- the whole
# point of this module is to be the lightweight, context-free alternative.
# "patient", "diagnosed", "history", "complaints" added after a direct audit
# (see Modules/cql_checker.py's copy of this list for the full case) found
# each an exact coincidental UMLS collision -- generic words ("the patient",
# "no complaints") that happen to have their own CUI, not genuine clinical
# content, and common enough on both sides of a real file to still cost
# real precision even in a plain set-difference.
JUNK_CONCEPT_DENYLIST = {
    "hand", "controll", "control", "other things", "move", "mind", "said",
    "close", "test", "able", "times", "life", "difficult", "little",
    "stage", "recap", "keen", "remember", "feels", "much", "normal",
    "nice", "listened", "sampled", "quite often", "examined", "therex",
    "etests", "couplet", "coinfection",
    "plan", "patient", "diagnosed", "history", "complaints",
}


def _is_junk_concept(term):
    return term.strip().lower() in JUNK_CONCEPT_DENYLIST


class MetaMapLiteClient:
    """Thin subprocess wrapper around a local MetaMapLite batch install.
    Runs MetaMapLite's MMI (MetaMap Machine Indexing) output format over a
    chunk of text and parses out (cui, preferred_name, semtypes) tuples.

    MetaMapLite's own project layout (after `mvn clean install`) puts the
    runnable jar under target/ and the UMLS data index + models under
    data/ivf/<release>/USAbase and data/models -- see the NLM docs linked in
    METAMAPLITE_INSTALL_DIR's comment above. The UMLS release folder name
    under data/ivf/ varies by which release you installed, so it's globbed
    for rather than hardcoded.
    """

    def __init__(self, install_dir=None):
        self._install_dir = install_dir or METAMAPLITE_INSTALL_DIR
        if not self._install_dir or not os.path.isdir(self._install_dir):
            raise RuntimeError(
                "MetaMapLite is not configured. Set METAMAPLITE_INSTALL_DIR to a "
                "local MetaMapLite install (requires a UMLS license + Java 8+) -- "
                "see https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/run-locally/MetaMapLite.html"
            )
        self._jar = self._find_jar()
        self._index_dir = self._find_index_dir()
        self._models_dir = os.path.join(self._install_dir, "data", "models")

    def _find_jar(self):
        matches = glob.glob(os.path.join(self._install_dir, "target", "metamaplite-*-standalone.jar"))
        if not matches:
            raise RuntimeError(
                f"No metamaplite-*-standalone.jar found under {self._install_dir}/target -- "
                "run `mvn clean install` in the MetaMapLite install first."
            )
        return matches[0]

    def _find_index_dir(self):
        matches = glob.glob(os.path.join(self._install_dir, "data", "ivf", "*", "USAbase"))
        if not matches:
            raise RuntimeError(
                f"No UMLS data index found under {self._install_dir}/data/ivf/*/USAbase -- "
                "install a MetaMapLite UMLS data set first."
            )
        return matches[0]

    def extract(self, text):
        """Runs MetaMapLite over `text` and returns [(cui, term, semtypes), ...]
        for every concept it recognizes."""
        if not text.strip():
            return []
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "input.txt")
            with open(in_path, "w", encoding="utf-8") as f:
                f.write(text)
            cmd = [
                "java", "-cp", self._jar, "gov.nih.nlm.nls.ner.MetaMapLite",
                f"--indexdir={self._index_dir}", f"--modelsdir={self._models_dir}",
                "--outputformat=mmi", in_path,
            ]
            subprocess.run(cmd, cwd=tmp, check=True, capture_output=True, timeout=120)
            out_path = os.path.join(tmp, "input.mmi")
            if not os.path.exists(out_path):
                return []
            with open(out_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        return [self._parse_mmi_line(line) for line in lines if line.strip()]

    @staticmethod
    def _parse_mmi_line(line):
        # MMI format: id|mmi|score|conceptname|CUI|semtypes|triggerinfo|location|length
        fields = line.rstrip("\n").split("|")
        cui = fields[4] if len(fields) > 4 else ""
        term = fields[3] if len(fields) > 3 else cui
        semtypes_raw = fields[5] if len(fields) > 5 else ""
        semtypes = set(t for t in semtypes_raw.strip("[]").split(",") if t)
        return cui, term, semtypes


class MetaMapCuiChecker(CheckerModule):
    """Extracts a {CUI: term} set from the transcript and from the SOAP note
    (one line/turn at a time -- see split_turns) and reports a plain CUI
    set-difference: a CUI in the transcript with no match anywhere in the
    note is an omission; a CUI in the note with no match anywhere in the
    transcript is a hallucination. No assertion-status modeling -- see
    module docstring."""

    def __init__(self, metamaplite_dir=None, quickumls_install_dir=None):
        try:
            self._client = MetaMapLiteClient(metamaplite_dir)
            self._backend = "metamaplite"
        except Exception:
            self._client = None
            self._matcher = get_matcher(quickumls_install_dir)  # raises RuntimeError if unconfigured too
            self._backend = "quickumls"

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        transcript_concepts = self._extract_concepts(transcript)
        soap_concepts = self._extract_concepts(soap_note)

        errors = []
        for cui, info in transcript_concepts.items():
            if cui not in soap_concepts:
                errors.append(("omission", self._describe(info)))
        for cui, info in soap_concepts.items():
            if cui not in transcript_concepts:
                errors.append(("hallucination", self._describe(info)))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    def _extract_concepts(self, text):
        """Returns {cui: {"term": str, "sentence": str}}, one entry per
        distinct CUI (first mention wins -- unlike MedspacyUmlsChecker, no
        per-mention assertion state to track, so only one representative
        sentence per concept is kept)."""
        concepts = {}
        for _speaker, line_text in split_turns(text):
            if not line_text.strip():
                continue
            for cui, term in self._extract_line(line_text):
                concepts.setdefault(cui, {"term": term, "sentence": line_text.strip()})
        return concepts

    def _extract_line(self, text):
        if self._backend == "metamaplite":
            raw = self._client.extract(text)
            return [(cui, term) for cui, term, _semtypes in raw if cui and not _is_junk_concept(term)]

        matches = self._matcher.match(text, best_match=True, ignore_syntax=False)
        return [
            (m["cui"], m["term"])
            for group in matches
            for m in group
            if not is_common_word(m["term"])
            and not is_common_word(m["ngram"])
            and not _is_junk_concept(m["term"])
            and not _is_junk_concept(m["ngram"])
        ]

    @staticmethod
    def _describe(info):
        return f"{info['sentence']} ({info['term']})"
