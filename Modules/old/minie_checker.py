"""Deterministic structured-fact checker based on MinIE (Minimizing Facts in
Open Information Extraction, Gashteovski et al. 2017 -- see drafts/
MinIE_Section.pdf, S8.1.19-24 for the source material this implements).

This is a from-scratch Python reimplementation of MinIE's method using
spaCy's dependency parser, NOT a wrapper around the original Java tool
(uma-pi1/minie, built on ClausIE + Stanford CoreNLP). The real MinIE needs a
Maven build, a downloaded CoreNLP model, and a py4j gateway process -- none
of which this project has set up. Reimplementing the three phases directly
in Python keeps this checker deterministic, dependency-light (spaCy only,
already a project dependency via Modules/medspacy_umls_checker.py), and
consistent with S8.1.23's own observation that MinIE's cost profile belongs
with the rule-based NLP in S7, not the neural checkers elsewhere in S8.1.

Implements the PDF's "standalone structured checker" integration path
(S8.1.22, second bullet), the general, less UMLS-vocabulary-dependent
sibling of Modules/medspacy_umls_checker.py's CUI set-difference approach:
extract (subject, relation, object) triples from the transcript and the
SOAP note independently, then compare the two triple sets directly instead
of comparing bare concept presence. Structured this way so it can also
catch a relational error a pure concept-presence check cannot -- the right
finding attached to the wrong subject (S8.1.22's own example: family
history misattributed to the patient or vice versa).

Three phases per S8.1.20 of the source material:

Phase 1 (_extract_triples_from_sentence) -- clause-based extraction. Each
finite verb (ROOT/conj/ccomp/xcomp/advcl/relcl) becomes one triple, with
prepositions pushed into the relation text and its object pulled from the
matching pobj (mirrors ClausIE's SV/SVO/SVOA clause typology and MinIE's own
"push constituents into the relation" rewriting). Sentences with no finite
verb -- extremely common in this dataset's telegraphic SOAP style ("No
blood in stool.", "Opening bowels x6/day.") -- fall back to a synthetic
(patient, has, <fragment>) triple per fragment/apposition segment instead
(_build_fragment_triples); see its docstring for why appositions are split
out as their own segment rather than folded into the parent noun's span.

Phase 2 (_detect_polarity/_detect_modality/_detect_attribution) -- strips
negation, modal/possibility cues, and attribution ("X believes/reports
that...") out of the triple text and records them as annotations instead,
same as MinIE's own polarity/modality/attribution split. Quantity (S8.1.20's
fourth annotation) is recorded for display only (_span_details) --
deliberately NOT replaced with a Q placeholder in the matching text the way
the paper describes, because this dataset's real number-edit corruptions
(prim57/bad notes labels lib/*.json detail_type "number edit") are exactly
the kind of change a Q-normalized comparison would silently stop catching.

Phase 3 (_span_details' minimization) -- MinIE-S ("Safe") minimization only,
the conservative mode the source material recommends starting with
(S8.1.24): drops determiners and possessive pronouns unconditionally, and
adjectives/adverbs only when they modify a person-like noun (PERSON_WORDS).

Known limitation, not fixed here: comma-bundled clinical shorthand with no
appositive dependency relation at all (e.g. "Associated LLQ pain - crampy,
intermittent, nil radiation.") still extracts as a single fragment, so a
negation cue anywhere in it (here "nil", genuinely only negating
"radiation") marks the whole fragment negative. Splitting on syntactic
appositions (_build_fragment_triples) recovers the paper's own worked
example ("Nil smoking, social EtOH" -> two independent facts) but not this
harder case, which has no appos relation to split on -- would need genuine
clause-level segmentation (real ClausIE) to fix properly.
"""
import os
import re
import sys
import time
from dataclasses import dataclass, field

import spacy

from Modules.base import CheckerModule

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from base import split_turns  # noqa: E402 -- path insert must run first
from umls_matching import has_real_concept  # noqa: E402 -- path insert must run first

SPACY_MODEL = "en_core_web_sm"

# Clause-boundary dependencies: a noun's own subtree can contain a nested
# clause (a relative clause, a conjunct, an attributed complement...) that
# spaCy still returns from doc.token.subtree. Minimization must NOT pull
# those in as if they were plain modifying words -- each is either its own
# independent triple already (extracted separately by the main clause loop)
# or, for "conj"/"cc", just clutter. Also excludes "neg"/"det"/"poss" here
# even though they get their own explicit handling below, so a stray nested
# occurrence deep in the subtree can't slip past the direct-child checks.
CLAUSE_BOUNDARY_DEPS = {"relcl", "acl", "ccomp", "xcomp", "advcl", "conj", "cc", "punct", "appos"}

# MinIE-S drops determiners and possessive pronouns unconditionally, and
# adjectives/adverbs only when they modify a person (S8.1.20's table). This
# is the set of head-noun lemmas treated as "a person" for that third rule --
# small and dataset-specific (this project's own transcript/SOAP-note pairs
# use exactly these referents for people), not a general person-detector.
PERSON_WORDS = {
    "patient", "doctor", "gp", "nurse", "he", "she", "they", "him", "her",
    "them", "mother", "father", "mum", "dad", "wife", "husband", "son",
    "daughter", "man", "woman", "gentleman", "lady", "i", "you", "we",
}

# Phase 2 (modality): modal verbs / adverbs that flag a statement as a
# possibility (PS) rather than a certainty (CT) -- S8.1.20's table gives
# "can"/"may" and "probably" as its own examples. "will"/"shall"/"must"/
# "should"/"would" are deliberately excluded: those mark obligation or
# futurity, not the speaker's own uncertainty about whether the fact holds.
POSSIBILITY_MODAL_LEMMAS = {"can", "could", "may", "might"}
POSSIBILITY_ADVERB_LEMMAS = {"probably", "possibly", "perhaps", "maybe"}

# Phase 2 (attribution): "X believes/reports/says that Y" -- verbs whose
# ccomp complement is who-said-it context rather than a fact about the
# patient in its own right (S8.1.20's table + demo).
ATTRIBUTION_VERB_LEMMAS = {
    "believe", "think", "report", "state", "say", "claim", "feel",
    "suspect", "mention", "note", "recall", "describe",
}

# Phase 2 (polarity): lexically-negative verbs whose negation isn't a
# syntactic "neg" child (spaCy tags "denies" as a plain transitive verb, not
# an auxiliary negator) -- confirmed directly against this project's own
# transcripts, e.g. "She denies any fever."/"Patient reports no blood..." --
# see module docstring's Phase 2 paragraph.
NEGATION_VERB_LEMMAS = {"deny", "refuse", "reject", "rule out"}

# Fragment-level negation cue words (S8.1.20's "polarity" row, extended past
# just "not"/"never" to this dataset's own SOAP-note shorthand). Checked by
# lemma, not dependency label -- confirmed necessary directly against this
# dataset's own parses: spaCy tags "No" in "No blood in stool." as a plain
# det, and "Nil" in "Nil smoking, social EtOH" as a compound, neither one a
# syntactic "neg" relation, so a dep_=="neg"-only check misses both.
NEGATION_CUE_LEMMAS = {"no", "nil", "none", "without", "absent", "deny", "denies"}

# Content-word filter for triple matching (Step 3) -- excludes function
# words so two triples match on their real clinical content instead of
# shared "a"/"the"/"in". Mirrors the is_common_word/junk-concept filtering
# already validated for Modules/medspacy_umls_checker.py's CUI matching.
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM", "PRON", "ADV"}

# Real-content gate (_has_real_content) -- confirmed necessary by a direct
# 10-file run: without it, this checker extracted a triple from EVERY
# declarative line, including pure disfluency/social speech that carries no
# checkable clinical fact at all ("Thank you very much." -> (patient | Thank
# | you), "That's fine." -> (that | 's | fine), "OK, yep, I will do, that's
# fine." -> (patient | do | )). Since these never appear in a SOAP note (by
# construction -- a note doesn't transcribe pleasantries), every one becomes
# a guaranteed omission false positive; that first run scored 3 TP / 1343 FP
# overall, 1289 of the FPs specifically omission. A triple is only emitted
# if its subject/relation/object lemmas contain at least one word outside
# these three closed classes:
#   - filler/discourse markers ("um", "so", "just", "kind of", ...)
#   - social speech-acts ("thank", "sorry", "nice", "fine", "alright", ...)
#   - bare pronouns/light verbs with no semantic content of their own
#     ("it", "that", "someone" / "need", "think", "can", "get", ...)
# A pronoun or light verb attached to a REAL content word survives fine --
# e.g. "(patient | feel | bit dizzy)" keeps "dizzy" even with "bit" and
# "feel" filtered out -- this only suppresses triples where NOTHING real is
# left after removing all three classes.
FILLER_LEMMAS = {
    "um", "uh", "uhh", "erm", "ah", "oh", "yeah", "yep", "yup", "nah",
    "ok", "okay", "well", "so", "just", "like", "know", "mean", "gonna",
    "wanna", "gotta", "right", "actually", "basically", "literally",
    "kind", "sort", "really", "quite",
}
SOCIAL_ACT_LEMMAS = {
    "thank", "thanks", "sorry", "welcome", "nice", "glad", "please",
    "hello", "hi", "bye", "goodbye", "pleasure", "good", "great", "fine",
    "alright", "sure",
}
PRONOUN_LEMMAS = {
    "i", "you", "we", "it", "that", "this", "he", "she", "they",
    "someone", "something", "anything", "nothing", "me", "him", "her",
    "them", "who", "what", "which",
}
LIGHT_VERB_LEMMAS = {
    "need", "think", "know", "try", "trying", "can", "could", "will",
    "would", "go", "going", "want", "guess", "mean", "suppose", "get",
    "got", "have", "be", "do", "did", "does",
}
CONTENT_STOP_LEMMAS = FILLER_LEMMAS | SOCIAL_ACT_LEMMAS | PRONOUN_LEMMAS | LIGHT_VERB_LEMMAS | {"patient"}

# Same "shares a stem" fallback already validated in
# Modules/medspacy_umls_checker.py's _cui_present (see its docstring for the
# real-file audit behind the >=4-char floor) -- reused here at the lemma
# level so e.g. "vomit"/"vomiting" still match after independent parses.
MIN_STEM_LEN = 4
MAX_STEM_LEN = 5

# Sentence-boundary splitting for the SOAP note side. Same validated
# behavior as Modules/alignscore_checker.py's split_full_stop_sentences
# (splits on ./!/?, on newlines, and on "/" immediately before a recognized
# clinical field label so a single unpunctuated line like "PMH: Asthma / DH:
# Inhalers" doesn't glue three unrelated facts into one span -- see that
# module's docstring for the full rationale and a confirmed real failure
# case) -- duplicated here rather than imported so this checker doesn't pull
# in alignscore's own (heavy, torch-based) import chain just for a regex.
MIN_SENTENCE_WORDS = 4
_FIELD_LABEL_RE = (
    r"(?:pmh\w*|dh\w*|sh\w*|fh\w*|psh\w*|ice|hx|hpc|"
    r"imp|impression|dx|diagnosis|assessment|"
    r"plan|rx|management|mx|"
    r"o/e|exam\w*|obs|vitals?|bloods?|ix|investigations?)\s*:"
)
_SOAP_SENTENCE_SPLIT_RE = re.compile(
    rf"(?<=[.!?])\s+|\n+|\s*/\s*(?={_FIELD_LABEL_RE})",
    re.IGNORECASE,
)


def _split_soap_sentences(text):
    return [
        s.strip() for s in _SOAP_SENTENCE_SPLIT_RE.split(text.strip())
        if s and len(s.strip().split()) >= MIN_SENTENCE_WORDS
    ]


# Transcript-side sentence splitting: ./!/? boundaries only (turn
# boundaries are already handled by split_turns before this ever runs, and
# unlike the SOAP note, transcript turns are conversational speech, not
# field-labeled shorthand).
_TRANSCRIPT_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_transcript_sentences(turn_text):
    return [
        s.strip() for s in _TRANSCRIPT_SENTENCE_SPLIT_RE.split(turn_text.strip())
        if s and len(s.strip().split()) >= MIN_SENTENCE_WORDS
    ]


@dataclass
class Triple:
    """One MinIE-style extraction: a compact (subject, relation, object)
    with its Phase 2 annotations attached, plus the source sentence (kept so
    Modules/evaluate.py's substring matching against ground-truth labels has
    real text to match against -- see Modules/medspacy_umls_checker.py's
    docstring for why a bare "term/state" summary can never match a label).
    """

    subject: str
    relation: str
    object: str
    polarity: str  # "+" or "-"
    modality: str  # "CT" (certainty) or "PS" (possibility)
    attribution: str | None
    quantities: list
    sentence: str
    subject_lemmas: frozenset = field(default_factory=frozenset)
    relation_lemmas: frozenset = field(default_factory=frozenset)
    object_lemmas: frozenset = field(default_factory=frozenset)


class MinIEChecker(CheckerModule):
    """See module docstring for the full Phase 1-3 extraction pipeline and
    the Step 3 (triple-set comparison) judging this implements."""

    def __init__(self, spacy_model=None):
        self._nlp = spacy.load(spacy_model or SPACY_MODEL, exclude=["ner"])

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        transcript_triples = self._extract_transcript_triples(transcript)
        soap_triples = self._extract_soap_triples(soap_note)

        errors = []
        errors.extend(self._judge_omissions(transcript_triples, soap_triples))
        errors.extend(self._judge_hallucinations(transcript_triples, soap_triples))
        errors.extend(self._judge_status_flips(transcript_triples, soap_triples))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    # ------------------------------------------------------------------
    # Document -> triples
    # ------------------------------------------------------------------

    def _extract_transcript_triples(self, transcript):
        triples = []
        for speaker, line_text in split_turns(transcript):
            if not line_text.strip():
                continue
            for sentence in _split_transcript_sentences(line_text):
                if sentence.endswith("?"):
                    continue  # a question doesn't assert anything -- same rule as medspacy_umls_checker
                triples.extend(self._extract_triples_from_sentence(sentence))
        return triples

    def _extract_soap_triples(self, soap_note):
        triples = []
        for sentence in _split_soap_sentences(soap_note):
            triples.extend(self._extract_triples_from_sentence(sentence))
        return triples

    # ------------------------------------------------------------------
    # Phase 1: clause-based extraction (+ Phase 2 annotations inline)
    # ------------------------------------------------------------------

    def _extract_triples_from_sentence(self, sentence):
        doc = self._nlp(sentence)
        triples = []

        for token in doc:
            if token.pos_ in ("VERB", "AUX") and token.dep_ in (
                "ROOT", "conj", "ccomp", "advcl", "relcl", "xcomp",
            ):
                triple = self._build_verb_triple(token, sentence)
                if triple is not None:
                    triples.append(triple)

        if not triples:
            triples.extend(self._build_fragment_triples(doc, sentence))

        return triples

    def _build_verb_triple(self, verb, sentence):
        subject_head = self._find_subject(verb)
        if subject_head is None and verb.dep_ in ("ccomp", "xcomp", "advcl", "relcl"):
            return None  # subordinate clause with no recoverable subject -- can't attribute this fact confidently

        if subject_head is not None:
            subject_text, subject_lemmas, _ = self._span_details(subject_head)
            subject_text, subject_lemmas = self._normalize_subject(subject_text, subject_lemmas)
        else:
            subject_text, subject_lemmas = "patient", frozenset({"patient"})

        relation_tokens = [verb]
        relation_parts = [verb.text]
        for child in verb.children:
            if child.dep_ == "prt":
                relation_parts.append(child.text)
                relation_tokens.append(child)

        obj_texts, obj_lemma_sets, quantities = [], [], []
        for child in verb.children:
            if child.dep_ in ("dobj", "attr", "acomp", "oprd"):
                text, lemmas, quants = self._span_details(child)
                if text:
                    obj_texts.append(text)
                    obj_lemma_sets.append(lemmas)
                    quantities.extend(quants)
            elif child.dep_ == "prep":
                pobjs = [g for g in child.children if g.dep_ == "pobj"]
                if pobjs:
                    relation_parts.append(child.text)
                    for pobj in pobjs:
                        text, lemmas, quants = self._span_details(pobj)
                        if text:
                            obj_texts.append(text)
                            obj_lemma_sets.append(lemmas)
                            quantities.extend(quants)

        has_ccomp = any(c.dep_ == "ccomp" for c in verb.children)
        if not obj_texts and has_ccomp:
            return None  # this verb's whole "object" is the ccomp clause, already its own triple below

        object_text = " / ".join(obj_texts)
        object_lemmas = frozenset().union(*obj_lemma_sets) if obj_lemma_sets else frozenset()
        relation_text = " ".join(relation_parts)
        relation_lemmas = frozenset(t.lemma_.lower() for t in relation_tokens if t.pos_ in CONTENT_POS)

        if not self._should_emit(
            subject_lemmas, relation_lemmas, object_lemmas,
            f"{subject_text} {relation_text} {object_text}",
        ):
            return None  # not a checkable clinical fact -- see _should_emit's docstring

        polarity = self._detect_polarity(verb, subject_lemmas | object_lemmas)
        modality = self._detect_modality(verb)
        attribution = self._detect_attribution(verb)

        return Triple(
            subject=subject_text, relation=relation_text, object=object_text,
            polarity=polarity, modality=modality, attribution=attribution,
            quantities=quantities, sentence=sentence,
            subject_lemmas=subject_lemmas, relation_lemmas=relation_lemmas,
            object_lemmas=object_lemmas,
        )

    @staticmethod
    def _find_subject(verb):
        subject = next((c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if subject is not None:
            return subject
        if verb.dep_ == "conj":  # a conjunct verb ("it comes and goes") inherits its head's subject
            return next((c for c in verb.head.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        return None

    @staticmethod
    def _normalize_subject(subject_text, subject_lemmas):
        """Collapses first/second-person deictic pronouns ("I"/"we"/"you",
        stripped of possessives by minimization already) to the canonical
        "patient" -- both sides of this checker's comparison mean the same
        real-world referent (the transcript's patient speaking in first
        person vs. the SOAP note's implicit "patient" subject), so leaving
        them as distinct literal strings would make every such triple
        unmatchable against its SOAP-note counterpart. Third-person "she"/
        "he"/named family members are deliberately left alone -- those are
        exactly the ambiguous-referent cases (patient vs. family history)
        this checker's subject comparison exists to catch (see module
        docstring's opening paragraph). Normalizes BOTH the display text and
        the lemma set used for Step 3 matching -- fixing only the text and
        leaving subject_lemmas as {"i"} would silently break matching
        against a SOAP-note triple whose subject literally lemmatizes to
        {"patient"}, since neither an exact-lemma nor a stem match can ever
        bridge "i" and "patient"."""
        if subject_text.strip().lower() in ("i", "we", "you", "me", "us"):
            return "patient", frozenset({"patient"})
        return subject_text, subject_lemmas

    @staticmethod
    def _has_real_content(subject_lemmas, relation_lemmas, object_lemmas):
        """True if this triple has at least one lemma outside
        CONTENT_STOP_LEMMAS -- see that constant's comment for the real-file
        false-positive audit behind this gate. A pronoun or light verb next
        to a genuine content word still passes (only the union across all
        three fields has to clear the bar, not each field individually)."""
        content = (subject_lemmas | relation_lemmas | object_lemmas) - CONTENT_STOP_LEMMAS
        return bool(content)

    def _should_emit(self, subject_lemmas, relation_lemmas, object_lemmas, combined_text):
        """Two independent gates a triple must clear before it's treated as
        a checkable fact at all:

        1. _has_real_content -- not pure filler/social/pronoun noise (fast,
           lemma-set only, checked first so the slower gate below never runs
           on obvious junk).
        2. has_real_concept (Medical condensor/umls_matching.py) -- the SAME
           tuned QuickUMLS relevance filter Modules/medspacy_umls_checker.py
           already uses, reused here rather than re-derived. Necessary
           because gate 1 alone still passed through a huge amount of real,
           grammatically well-formed, but clinically IRRELEVANT content --
           "I work", "that's under control", "everything else is fine" --
           that a real SOAP note correctly and routinely omits by design (a
           note summarizes, it doesn't transcribe). This dataset's own
           "omission" ground truth labels are specifically the sentences a
           corruption pipeline deliberately deleted (see
           datamakerfiles/prim_lib_injection.py), not "every transcript
           detail missing from the note" -- so a checker that flags EVERY
           syntactically-valid fact as checkable cannot tell normal
           summarization apart from an injected error. Confirmed directly:
           gate 1 alone still scored 3 TP / 1031 FP on a real 10-file run
           (957 of those omission); restricting "checkable" to clinically
           relevant content is the same fix already validated for
           Modules/medspacy_umls_checker.py's CUI-based approach, applied
           here to MinIE's triples instead of bare concept spans."""
        if not self._has_real_content(subject_lemmas, relation_lemmas, object_lemmas):
            return False
        return has_real_concept(combined_text)

    def _build_fragment_triples(self, doc, sentence):
        """Phase 1 fallback for a clause with no finite verb at all -- this
        dataset's own SOAP-note shorthand ("No blood in stool.", "Opening
        bowels x6/day.") is full of these (see module docstring).

        Splits off each syntactic apposition attached to the sentence's
        root as its OWN fragment rather than folding it into the root's
        minimized span. Directly recovers the source material's own worked
        example (S8.1.22): "Nil smoking, social EtOH" parses with "EtOH" as
        an appos child of root "smoking" -- treating them as one fragment
        would produce the nonsensical (patient, has, "Nil smoking social
        EtOH") with negation from "Nil" incorrectly spreading onto EtOH;
        splitting them yields the paper's own two independent facts instead
        -- (patient, has, "Nil smoking") negative, (patient, has, "social
        EtOH") positive. See module docstring's "Known limitation" for the
        comma-bundled case this split does NOT catch (no appos relation to
        split on)."""
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        if root is None or root.pos_ in ("VERB", "AUX"):
            return []

        segments = [root] + [c for c in root.children if c.dep_ == "appos"]
        triples = []
        for segment in segments:
            text, lemmas, quantities = self._span_details(segment)
            if not text:
                continue
            if not self._should_emit(frozenset({"patient"}), frozenset({"have"}), lemmas, text):
                continue  # not a checkable clinical fact -- see _should_emit's docstring
            polarity = "-" if lemmas & NEGATION_CUE_LEMMAS else "+"
            triples.append(Triple(
                subject="patient", relation="has", object=text,
                polarity=polarity, modality="CT", attribution=None,
                quantities=quantities, sentence=sentence,
                subject_lemmas=frozenset({"patient"}),
                relation_lemmas=frozenset({"have"}),
                object_lemmas=lemmas,
            ))
        return triples

    # ------------------------------------------------------------------
    # Phase 2: semantic annotations
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_polarity(verb, related_lemmas):
        if any(c.dep_ == "neg" for c in verb.children):
            return "-"
        if verb.lemma_.lower() in NEGATION_VERB_LEMMAS:
            return "-"
        if related_lemmas & NEGATION_CUE_LEMMAS:
            return "-"
        return "+"

    @staticmethod
    def _detect_modality(verb):
        for child in verb.children:
            if child.dep_ in ("aux", "auxpass") and child.lemma_.lower() in POSSIBILITY_MODAL_LEMMAS:
                return "PS"
            if child.dep_ == "advmod" and child.lemma_.lower() in POSSIBILITY_ADVERB_LEMMAS:
                return "PS"
        return "CT"

    @staticmethod
    def _detect_attribution(verb):
        if verb.dep_ != "ccomp":
            return None
        head = verb.head
        if head.pos_ not in ("VERB", "AUX") or head.lemma_.lower() not in ATTRIBUTION_VERB_LEMMAS:
            return None
        head_subject = next((c for c in head.children if c.dep_ == "nsubj"), None)
        subject_text = head_subject.text if head_subject is not None else "someone"
        return f"{subject_text} {head.lemma_}"

    # ------------------------------------------------------------------
    # Phase 3: MinIE-S minimization (used by both the verb path's subject/
    # object spans and the fragment path)
    # ------------------------------------------------------------------

    def _span_details(self, token):
        """Walks token's subtree, applying MinIE-S's minimization rules
        (drop determiners and possessive pronouns unconditionally; drop
        adjective/adverb modifiers only of a person-like head noun -- see
        PERSON_WORDS) and stopping at clause boundaries
        (CLAUSE_BOUNDARY_DEPS, which includes "appos" -- so a sentence's
        root never swallows an apposition attached to it; see
        _build_fragment_triples, which handles appositions as their own
        separate segments instead). Returns (minimized_text,
        content_lemma_set, [cardinal-number annotations]) -- the lemma set
        is content words only (CONTENT_POS), used for Step 3 matching; the
        number list is Phase 2's quantity annotation, display-only (see
        module docstring for why it isn't folded into the matching text)."""
        included = []
        quantities = []

        def visit(t):
            included.append(t)
            if t.pos_ == "NUM":
                quantities.append(t.text)
            for child in t.children:
                if child.dep_ in CLAUSE_BOUNDARY_DEPS:
                    continue
                if child.dep_ == "det":
                    continue
                if child.dep_ == "poss" and child.pos_ == "PRON":
                    continue
                if child.dep_ == "neg":
                    continue
                if child.dep_ in ("amod", "advmod") and t.lemma_.lower() in PERSON_WORDS:
                    continue
                visit(child)

        visit(token)
        included.sort(key=lambda t: t.i)
        text = " ".join(t.text for t in included if t.pos_ != "PUNCT")
        lemmas = frozenset(t.lemma_.lower() for t in included if t.pos_ in CONTENT_POS)
        return text, lemmas, quantities

    # ------------------------------------------------------------------
    # Step 3: triple-set comparison / judging
    # ------------------------------------------------------------------

    def _judge_omissions(self, transcript_triples, soap_triples):
        """A transcript triple with no matching (subject+object, any
        polarity) triple anywhere in the SOAP note -- content the note
        dropped entirely."""
        return [
            ("omission", self._describe(t))
            for t in transcript_triples
            if not any(self._triples_match(t, s) for s in soap_triples)
        ]

    def _judge_hallucinations(self, transcript_triples, soap_triples):
        """A SOAP-note triple with no matching triple anywhere in the
        transcript -- content the note invented."""
        return [
            ("hallucination", self._describe(s))
            for s in soap_triples
            if not any(self._triples_match(s, t) for t in transcript_triples)
        ]

    def _judge_status_flips(self, transcript_triples, soap_triples):
        """A triple present in BOTH documents (subject+object match) whose
        polarity never agrees between any matching transcript triple and any
        matching SOAP triple -- the fact itself is neither missing nor
        invented, just asserted with the opposite polarity. Same
        deliberately-separate-third-judgment structure as
        Modules/medspacy_umls_checker.py's _judge_status_flips -- see its
        docstring for why this isn't folded into omission/hallucination
        instead."""
        errors = []
        for s in soap_triples:
            matches = [t for t in transcript_triples if self._triples_match(s, t)]
            if not matches:
                continue  # not present in both -- already a hallucination above
            if any(t.polarity == s.polarity for t in matches):
                continue  # at least one match agrees -- not a flip
            errors.append(("status_flip", self._describe_flip(s, matches[0])))
        return errors

    @classmethod
    def _triples_match(cls, a, b):
        """Two triples "match" (refer to the same real-world fact,
        independent of polarity -- polarity agreement is _judge_status_flips'
        job, not this one's, same split as
        Modules/medspacy_umls_checker.py's _cui_present/_judge_status_flips)
        if their subjects overlap AND their PREDICATES overlap, where a
        predicate is relation_lemmas UNION object_lemmas rather than the two
        compared separately.

        Deliberately not object-vs-object with a relation-only fallback (an
        earlier version of this method) -- that split assumes both triples
        put the clinical concept in the same slot, which doesn't hold
        between this checker's two extraction paths: a transcript verb
        clause encodes "doesn't smoke" as relation="smoke" with NO object,
        while a SOAP-note fragment encodes "Nil smoking" as object="smoking"
        with a synthetic relation="has" (see _build_fragment_triples). Under
        the old split, comparing object_lemmas={} against object_lemmas=
        {"smoking"} short-circuited to the relation-only fallback, which
        then compared relation_lemmas {"smoke"} against {"have"} and never
        matched -- confirmed directly on a real file: transcript "Uh, no, I
        don't smoke."/"I don't drink alcohol" both went unmatched against
        the SOAP note's own "Nil smoking, social EtOH" this way, becoming
        pure false-positive omissions despite the note correctly recording
        both facts. Unioning relation+object into one predicate set fixes
        this regardless of which slot either side happened to put the
        concept in.

        CONTENT_STOP_LEMMAS is subtracted from both predicates before
        comparing -- without this, every fragment-path triple's relation is
        the same fixed literal "have" (see _build_fragment_triples), so ANY
        two fragment triples would trivially "match" on that shared generic
        verb alone regardless of what their actual objects were, silently
        treating every fragment as mutually supporting every other one."""
        if not cls._lemma_sets_overlap(a.subject_lemmas, b.subject_lemmas):
            return False
        a_predicate = (a.relation_lemmas | a.object_lemmas) - CONTENT_STOP_LEMMAS
        b_predicate = (b.relation_lemmas | b.object_lemmas) - CONTENT_STOP_LEMMAS
        return cls._lemma_sets_overlap(a_predicate, b_predicate)

    @staticmethod
    def _lemma_sets_overlap(lemmas_a, lemmas_b):
        if lemmas_a & lemmas_b:
            return True
        for a in lemmas_a:
            for b in lemmas_b:
                stem_len = min(len(a), len(b), MAX_STEM_LEN)
                if stem_len >= MIN_STEM_LEN and a[:stem_len] == b[:stem_len]:
                    return True
        return False

    @staticmethod
    def _describe_state(polarity, modality):
        parts = []
        if polarity == "-":
            parts.append("negated")
        if modality == "PS":
            parts.append("possible")
        return " & ".join(parts) if parts else "affirmed"

    @classmethod
    def _describe(cls, t):
        state = cls._describe_state(t.polarity, t.modality)
        attribution_part = f", attributed to {t.attribution}" if t.attribution else ""
        quantity_part = f", quantity={'/'.join(t.quantities)}" if t.quantities else ""
        return f"{t.sentence} ({t.subject} | {t.relation} | {t.object}: {state}{attribution_part}{quantity_part})"

    @classmethod
    def _describe_flip(cls, soap_triple, transcript_triple):
        t_state = cls._describe_state(transcript_triple.polarity, transcript_triple.modality)
        s_state = cls._describe_state(soap_triple.polarity, soap_triple.modality)
        return (
            f"{transcript_triple.sentence} {soap_triple.sentence} "
            f"({soap_triple.subject} | {soap_triple.relation} | {soap_triple.object}: "
            f"{t_state} in transcript vs {s_state} in SOAP note)"
        )
