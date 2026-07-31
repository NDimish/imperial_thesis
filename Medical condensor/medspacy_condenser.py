import os
import sys
import time

from loguru import logger

# medspacy's PyRuSH sentencizer logs at DEBUG by default, which floods the
# terminal/log files with per-token sentence-boundary traces. Quiet it down.
logger.remove()
logger.add(sys.stderr, level="WARNING")

import medspacy
import spacy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import CondenserModule, join_turns, split_turns
from umls_matching import get_real_concept_spans


class MedspacyCondenser(CondenserModule):
    """Removes non-clinical filler using the shared UMLS concept matcher (see
    umls_matching.py) for concept detection, and medspacy's ConText algorithm
    for assertion status -- negation, uncertainty, historical, hypothetical,
    family-context.

    This used to run on a hand-typed list of ~60 clinical terms
    (CLINICAL_TARGET_RULES). That was medspacy's *only* source of medical
    knowledge -- the library itself ships none, by design (it's meant to be
    seeded with your own rules or an external terminology). A real example
    confirmed the hand list was a genuine, unbounded coverage gap, not a
    hypothetical one: on prim28.txt, a patient's underactive thyroid,
    thyroxine medication, and a depression/mood/acid-reflux screening were
    dropped entirely, simply because none of those words happened to be
    typed into the list. No fixed list can ever be complete, so the hand list
    is gone entirely now -- UMLS (millions of real terms and synonyms via the
    shared matcher) is the sole concept source.

    What differentiates this from the other UMLS-grounded condensers:
      - QuickUMLSCondenser: is there a real concept here at all (no assertion
        awareness).
      - NegspacyCondenser: that, plus it must not be negated (one axis).
      - MedspacyCondenser (this): that, plus the full ConText picture -- a
        turn is kept only if it has a concept that's both NOT negated and NOT
        purely hypothetical. Historical and family-context mentions are
        deliberately still kept (a family history of cancer, or a resolved
        past symptom, is genuine clinical content a real SOAP note would
        record -- unlike a negated or hypothetical mention, which describes
        something that isn't actually the case).
    """

    def __init__(self):
        self._nlp = medspacy.load()

    def condense(self, transcript):
        start = time.perf_counter()

        # A "keep the turn after a kept question" lookback was tried here
        # too (see quickumls_condenser.py's condense() for the full story)
        # and reverted for the same reason: it improved the KDE-based
        # groundedness score a lot, but that score turned out to penalize
        # condensing almost regardless of content quality. Under
        # check_omissions_cosine (validated to actually track content
        # quality), this condenser's real coverage of the SOAP note was
        # about the same with or without it, so the lookback was only
        # costing condensing aggressiveness (21.2% words reduced without it
        # vs 9.7% with it) for no real coverage benefit.
        turns = split_turns(transcript)
        kept = [(speaker, text) for speaker, text in turns if self._is_clinical(text, speaker)]
        condensed = join_turns(kept)

        elapsed = time.perf_counter() - start
        return condensed, elapsed

    def _is_clinical(self, text, speaker=None):
        if not text.strip():
            return False

        spans = get_real_concept_spans(text)
        if not spans:
            return False

        # Full pipeline first (sentence boundaries matter to ConText's scope
        # rules), with target_matcher finding nothing since it has no rules --
        # then swap in the UMLS-derived spans as the entities ConText assesses.
        doc = self._nlp(text)
        candidate_ents = []
        for match_start, match_end in spans:
            span = doc.char_span(match_start, match_end, label="CONCEPT", alignment_mode="expand")
            if span is not None:
                candidate_ents.append(span)
        if not candidate_ents:
            return False

        doc.ents = spacy.util.filter_spans(candidate_ents)
        self._nlp.get_pipe("medspacy_context")(doc)

        # ConText's negation/hypothetical rules were built for declarative
        # clinical notes ("denies fever", "if symptoms worsen"), not spoken
        # interrogatives. Found via a full-dataset audit: doctor turns like
        # "Have you had any other symptoms, like fever or temperature?" were
        # being tagged NEGATED -- "any" is a real ConText negation trigger in
        # written notes ("denies any pain"), but here it's just a question,
        # not an assertion that fever is absent. Asking about a symptom is
        # itself real clinical content, so a question turn is kept on concept
        # presence alone; assertion status only makes sense to apply to
        # statements. This is a syntactic distinction (question vs statement),
        # not a word list, so it doesn't reopen the hand-typing problem.
        if text.strip().endswith("?"):
            return True

        # A full-dataset audit of statement turns dropped for is_hypothetical
        # found all 73 were genuine conditional language ("if symptoms
        # worsen, come back", "take paracetamol if you're feverish") -- and
        # the overwhelming majority were the DOCTOR giving conditional advice
        # ("if X, do Y"), not the patient hedging about their own condition.
        # ConText's "hypothetical" rule is designed to exclude a patient's
        # hypothetical symptom (something that isn't actually the case), but
        # a doctor's conditional advice was genuinely given -- it's real Plan
        # content, not a hypothetical claim about the patient. So doctor
        # turns are judged on negation alone; hypothetical still applies to
        # patient turns, where the original rationale holds.
        if speaker == "d":
            return any(not ent._.is_negated for ent in doc.ents)

        return any(not ent._.is_negated and not ent._.is_hypothetical for ent in doc.ents)
