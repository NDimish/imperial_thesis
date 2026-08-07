"""Deterministic hallucination/omission checker via UMLS concept + negation
matching, instead of per-sentence NLI/factuality scoring against a whole
other document (the approach every other checker in this project uses).

Why: AlignScoreChecker/FactKBChecker/KdbeChecker/SummaCChecker all split one
document into sentences and score each sentence *in isolation* against the
*entire other document* as an undifferentiated blob, using models trained on
generic (often news/Wikipedia-style) entailment data. Real testing this
session found three structural problems with that approach, not just
under-tuned thresholds:
  1. Context fragmentation -- an isolated transcript sentence often doesn't
     carry meaning without its preceding turn.
  2. Domain mismatch -- SOAP notes use clinical shorthand ("Nil smoking",
     "SH: lives alone, software engineer") that a generic NLI model doesn't
     reliably recognize as equivalent to the transcript's conversational
     phrasing ("no I don't smoke", "I live alone... I'm a software
     engineer"), even when the clinical content matches exactly.
  3. No synthesis -- a SOAP line often compresses facts scattered across
     several distant transcript turns; a per-sentence checker can only
     compare one candidate sentence against the whole other document, with
     no mechanism to trace a claim back across multiple disconnected
     mentions.
  FactKBChecker's threshold, tuned on 5 labeled files, was then measured to
  produce ZERO true positives on 5 fresh files (see main_run10_v4 results)
  -- a concrete demonstration that per-sentence scoring on raw text is
  fragile for this specific task, not just imprecisely calibrated.

This checker sidesteps all three by comparing at the CONCEPT level instead
of the sentence-text level, reusing the same shared UMLS matcher and
medspacy ConText assertion machinery already built and validated for the
condensers this session (see Medical condensor/umls_matching.py and
medspacy_condenser.py):
  - Concepts are matched by UMLS CUI (concept unique identifier), not
    surface text, so "smoking" (transcript) and "Nil smoking" (SOAP note)
    resolve to the same underlying concept regardless of register --
    directly fixing the domain-mismatch problem, since CUIs are
    phrasing-independent by construction.
  - Each concept's assertion status (negated / affirmed / hypothetical) is
    extracted via medspacy's ConText, the same algorithm used for the
    condensers -- and SOAP notes are exactly ConText's intended domain
    (declarative clinical prose: "denies fever", "nil smoking"), unlike the
    spoken transcript side, where this project already found and fixed a
    real mismatch (ConText reading a doctor's question as a negation).
  - A concept is a HALLUCINATION if it's asserted in the SOAP note at some
    polarity (affirmed or negated) with no matching CUI+polarity anywhere
    in the transcript. It's an OMISSION if the reverse is true.
  - Matching on CUI+polarity together (not just CUI) means "I smoke" and "I
    don't smoke" -- which can score deceptively close on raw embedding
    cosine similarity, since they're both "about smoking" -- are correctly
    treated as different facts, not the same one.

Deliberately deterministic and free of any external LLM/API dependency:
same two inputs always produce the same concept sets and the same errors,
no network calls, no sampling.
"""
import os
import sys
import time

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

import medspacy
import spacy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from umls_matching import get_matcher, is_common_word  # noqa: E402 -- path insert must run first

from Modules.base import CheckerModule

# umls_matching.py's own GENERIC_WORD_DENYLIST (start/well/good/fine/right/
# help/birth) is intentionally small -- it was tuned for the condensers,
# where the cost of a false positive is diluted (a turn survives as long as
# ANY of its words is a real concept) and every extra word denylisted risks
# dropping real recall. This checker has the opposite risk profile: it
# treats every matched concept as an independent fact to compare 1:1, so a
# coincidental collision here doesn't cost a little recall, it directly
# manufactures a spurious hallucination/omission. A single-file smoke test
# surfaced a long tail of exact (similarity=1.0) UMLS collisions on
# completely ordinary words the condensers' smaller list never needed to
# cover -- "hand" is a real T023 Body Part entry, "life" a real T060
# Diagnostic Procedure entry, "said"/"times" real T047 Disease/Syndrome
# entries, none of which are used in any clinical sense in this dataset.
# Checked directly: every one is an exact sim=1.0 match, not a borderline
# fuzzy one, so raising the match threshold would not have filtered any of
# them -- a denylist is the only lever that addresses this specific failure
# mode. Kept local to this checker (not added to the shared
# GENERIC_WORD_DENYLIST) so it doesn't reopen the recall cost already
# measured and reverted for the condensers. Expect this list to need
# expansion as more files are tested -- it was built from one file's smoke
# test, not the full dataset.
JUNK_CONCEPT_DENYLIST = {
    "hand", "controll", "control", "other things", "move", "mind", "said",
    "close", "test", "able", "times", "life", "difficult", "little",
    "stage", "recap", "keen", "remember", "feels", "much", "normal",
    "nice", "listened", "sampled", "quite often", "examined", "therex",
    "etests", "couplet", "coinfection",
    # "plan" specifically because it matches the SOAP note's own "Plan:"
    # section header -- a document-formatting artifact, not patient content.
    "plan",
}
# Deliberately NOT included, even though they also showed up as noise in
# the same smoke test: "smoker", "vitamins", "triggers", "stools",
# "bowels"/"bowel problems", "back", "food", "liquid", "eating", "drinking",
# "lives". These are genuinely clinical concepts -- their appearance as
# false positives is a symptom of the separate CUI-granularity problem
# (see _has_match's docstring), not a coincidental word collision.
# Denylisting them would hide that real gap instead of fixing it.


def _is_junk_concept(term):
    return term.strip().lower() in JUNK_CONCEPT_DENYLIST


class ConceptChecker(CheckerModule):
    """Flags hallucinated/omitted SOAP content by comparing UMLS concept +
    negation-status sets between the transcript and the SOAP note, instead
    of scoring raw sentence text. See module docstring for the full
    rationale and how this differs from every other checker in this project.
    """

    def __init__(self, install_dir=None):
        self._matcher = get_matcher(install_dir)
        self._nlp = medspacy.load()

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        transcript_concepts = self._extract_concepts(transcript)
        soap_concepts = self._extract_concepts(soap_note)

        errors = []
        for cui, term, negated in soap_concepts:
            if not self._has_match(transcript_concepts, cui, negated, term):
                errors.append(("hallucination", self._describe(term, negated)))
        for cui, term, negated in transcript_concepts:
            if not self._has_match(soap_concepts, cui, negated, term):
                errors.append(("omission", self._describe(term, negated)))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    def _extract_concepts(self, text):
        """Returns a set of (cui, term, negated) tuples for every real UMLS
        concept in text that's genuinely asserted -- i.e. not inside a
        question (a question doesn't assert its concept is present or
        absent; that's the answer's job -- same syntactic distinction
        already validated for the condensers this session) and not
        hypothetical (ConText's own notion of "not actually the case")."""
        if not text.strip():
            return set()

        # Checking BOTH the matched canonical term and the raw surface text
        # ("ngram") against the denylist, not just the term. This is known
        # to cost real recall for the condensers (a short common word like
        # "right" fuzzy-matching an unrelated canonical term like "bright"/
        # "fright" can be a turn's only real-concept trigger, so narrowing
        # it there dropped whole turns of real content -- see
        # umls_matching.py). That tradeoff doesn't apply here: this checker
        # treats every matched concept as an independent fact to compare, so
        # a junk match here doesn't cost recall, it directly manufactures a
        # spurious error. A first single-file smoke test surfaced this
        # immediately -- "fright"/"life"/"hand"/"much"/"little" alongside
        # real concepts, all coincidental collisions inflating the flagged
        # count. Stricter filtering is the right call for this specific use.
        matches = self._matcher.match(text, best_match=True, ignore_syntax=False)
        candidates = [
            (m["start"], m["end"], m["cui"], m["term"])
            for group in matches
            for m in group
            if not is_common_word(m["term"])
            and not is_common_word(m["ngram"])
            and not _is_junk_concept(m["term"])
            and not _is_junk_concept(m["ngram"])
        ]
        if not candidates:
            return set()

        doc = self._nlp(text)
        span_to_match = {}
        candidate_ents = []
        for match_start, match_end, cui, term in candidates:
            span = doc.char_span(match_start, match_end, label="CONCEPT", alignment_mode="expand")
            if span is None:
                continue
            candidate_ents.append(span)
            span_to_match[(span.start_char, span.end_char)] = (cui, term)

        if not candidate_ents:
            return set()

        doc.ents = spacy.util.filter_spans(candidate_ents)
        self._nlp.get_pipe("medspacy_context")(doc)

        concepts = set()
        for ent in doc.ents:
            cui, term = span_to_match.get((ent.start_char, ent.end_char), (None, ent.text))
            if cui is None:
                continue
            sent_text = ent.sent.text.strip() if ent.sent is not None else ent.text
            if sent_text.endswith("?"):
                continue  # a question doesn't assert the concept either way
            if ent._.is_hypothetical:
                continue  # "if you have chest pain" isn't a real assertion either way
            concepts.add((cui, term, bool(ent._.is_negated)))

        return concepts

    @staticmethod
    def _has_match(concept_set, cui, negated, term=None):
        """A concept counts as matched if either its CUI matches exactly, or
        (fallback) its term shares a >=5-character word stem with a term in
        the other set at the same polarity.

        Exact-CUI-only matching was tested first and found too strict: a
        single-file smoke test on a real transcript/SOAP pair produced 70+
        flagged "omissions" for concepts like "vomit"/"vomiting" and
        "cramp"/"muscular cramp" that plausibly ARE represented in the SOAP
        note, just resolved by QuickUMLS to a slightly different (though
        clearly related) CUI for the inflected or phrased-differently form.
        This is a coarse fix (a shared 5+ character prefix, not real
        stemming or UMLS synonym/relationship data), but directly targets
        that failure mode without reopening the junk-collision problem --
        short common words are already filtered out before reaching this
        point, so a 5-character minimum stays well clear of them."""
        if any(c_cui == cui and c_negated == negated for c_cui, _, c_negated in concept_set):
            return True
        if term is None:
            return False
        term_key = term.strip().lower()[:5]
        if len(term_key) < 5:
            return False
        return any(
            c_negated == negated and c_term.strip().lower()[:5] == term_key
            for _, c_term, c_negated in concept_set
        )

    @staticmethod
    def _describe(term, negated):
        return f"{term} ({'negated' if negated else 'affirmed'})"
