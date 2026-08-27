"""NLPMEDANDconverstaionNLP: two-tool clinical NLP comparison checker.

SOAP-note side: medspaCy -- the shared QuickUMLS matcher for concept/CUI
extraction plus medspaCy's ConText algorithm for negation tagging. Reuses
Modules/medspacy_umls_checker.py's MedspacyUmlsChecker._extract_concepts
directly (see _soap_extractor below) rather than re-deriving it: a written
SOAP note is exactly the register medspaCy's rule sets (ConText,
sectionizer) were built and validated for in this project already.

Transcript side: Clinical-AI-Apollo/Medical-NER -- a DeBERTa-v3 model
fine-tuned for 41-type clinical NER (SIGN_SYMPTOM, DISEASE_DISORDER,
MEDICATION, HISTORY, ...), the closest public match to "a fine-tuned
BioLinkBERT/PubMedBERT NER" -- is the span-finder: it decides WHERE the
concept-worthy phrases are in each turn. It has no UMLS-linking head and no
negation output of its own, so two more tools still do the rest of the job:
the project's shared QuickUMLS matcher resolves each span it finds to a real
CUI (so both sides land in the same CUI space the SOAP side already uses),
and negspaCy's NegEx algorithm (Chapman, Bridewell, Hanbury, Cooper &
Buchanan 2001) tags each resolved span negated/not -- the only negation
source available on the transcript side either way, since neither QuickUMLS
nor the transformer detect negation themselves.

This is the SECOND transcript-side design tried in this module -- see
extra/explainer/conversationalNLP.html for the full history:
  1. MedCAT -- ruled out outright: hard-requires numpy>=2.0, binary-
     incompatible with the numpy-1.x build every spaCy/medspaCy/QuickUMLS
     component in this project's shared conda env depends on (confirmed by
     installing it, breaking spacy/medspacy/quickumls, then rolling numpy
     back and confirming they recovered).
  2. QuickUMLS's own n-gram/dictionary scan for span-finding, negspaCy NegEx
     for negation -- the original working baseline. Evaluated against both
     prim57/bad notes labels lib and .../lib extra.
  3. This version: the transformer takes over span-finding entirely (no
     more QuickUMLS n-gram scan of the raw turn text) -- QuickUMLS is now
     used ONLY as a per-span CUI dictionary lookup, not an independent
     candidate source. Chosen and built at the user's direct request after
     an earlier smoke test found the transformer added zero NEW candidates
     on top of QuickUMLS's own scan in an additive role (see
     conversationalNLP.html) -- this version tests the opposite regime:
     what the transcript side looks like when the transformer is the ONLY
     thing deciding where to look, not an addition to a wider scan. Given
     that earlier finding, this is expected to find a SUBSET of what
     baseline (2) found, not a superset -- verified, not assumed, against
     prim57/bad notes labels lib (see conversationalNLP.html for the run).

Comparison (Step 3): CUI set-difference, same presence-only judgment
Modules/medspacy_umls_checker.py uses for its own omission/hallucination
pair (_cui_present is reused verbatim, stem-fallback included) -- these are
the two error types prim57/bad notes labels lib's ground truth ever uses
(confirmed directly by inspecting prim1.txt), so the comparison here is
scoped to exactly what's scorable against it. Negation state is extracted
on both sides but deliberately NOT folded into the omission/hallucination
match key or surfaced as a third status_flip type -- medspaCy's ConText
also tags uncertainty/family/historical state alongside negation, while
negspaCy's NegEx tags negation only, so the two sides are not capturing the
same assertion dimensions; comparing them as if they were would either
manufacture spurious flips on every affirmed-but-hedged concept (if
compared as full state tuples) or silently double-count one flipped concept
as one omission plus one hallucination (if folded into a (cui, negated)
match key, the exact failure mode medspacy_umls_checker.py's own docstring
already documents and rejects for Modules/old/concept_checker.py). Instead,
negation_agreement() reports negation-state agreement as its own
independent statistic over CUIs present on both sides -- see
extra/explainer/conversationalNLP.html for that analysis.

Deterministic throughout -- no LLM, no sampling; the transformer is a fixed
extractive classifier (same input always produces the same span/score
output), same as this project's other neural components (AlignScore,
SummaC).
"""
import os
import re
import sys
import time

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

import spacy
from negspacy.negation import Negex  # noqa: F401 -- import registers the "negex" spaCy factory

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from base import split_turns  # noqa: E402 -- path insert must run first
from umls_matching import get_matcher, is_common_word  # noqa: E402 -- path insert must run first

from Modules.base import CheckerModule
from Modules.medspacy_umls_checker import (  # noqa: E402
    MedspacyUmlsChecker,
    _is_junk_concept,
    _NEGATIVE_ANSWER_RE,
    _AFFIRMATIVE_ANSWER_RE,
)

# Transcript-side span-finder -- see module docstring for the full
# reasoning and history. 41 entity types trained on the MACCROBAT clinical
# case-report corpus; only the types below overlap the clinical scope
# QuickUMLS is itself scoped to elsewhere in this project (diseases/
# findings, procedures, drugs, history) -- SEVERITY/DETAILED_DESCRIPTION/
# BIOLOGICAL_STRUCTURE/etc. are real tags this model has but aren't
# standalone "concepts" in the CUI-diff sense used here, so they're left
# out of scope rather than treated as their own comparable entities.
TRANSFORMER_NER_MODEL = "Clinical-AI-Apollo/Medical-NER"
TRANSFORMER_NER_ENTITY_GROUPS = {
    "SIGN_SYMPTOM", "DISEASE_DISORDER", "MEDICATION",
    "DIAGNOSTIC_PROCEDURE", "THERAPEUTIC_PROCEDURE",
    "HISTORY", "FAMILY_HISTORY",
}
TRANSFORMER_NER_MIN_SCORE = 0.3


class NLPMEDANDconverstaionNLP(CheckerModule):
    """See module docstring for the full two-tool Extraction -> Comparison
    pipeline this implements."""

    def __init__(self, install_dir=None):
        self._matcher = get_matcher(install_dir)
        # SOAP-note side: reuse MedspacyUmlsChecker's own extractor instance
        # (its _matcher and its medspacy.load() pipeline) rather than
        # building a second, separate one.
        self._soap_extractor = MedspacyUmlsChecker(install_dir)

        # Transcript side: negspaCy NegEx, scoped to only negate spans we've
        # labeled "CONCEPT" ourselves (the transformer-found, QuickUMLS-
        # resolved candidates below) -- same reasoning as
        # medspacy_umls_checker.py restricting medspaCy's ConText to its own
        # injected entities rather than trusting a general-purpose NER.
        self._nlp = spacy.load("en_core_web_sm")
        self._nlp.add_pipe("negex", config={"neg_termset": {}, "ent_types": ["CONCEPT"]})

        self._ner_pipeline = None  # lazy -- see _get_ner_pipeline

    def _get_ner_pipeline(self):
        """Lazily loads the transformer span-finder -- only imports
        transformers and downloads/loads the DeBERTa-v3 checkpoint the
        first time a transcript actually needs it, not on every
        NLPMEDANDconverstaionNLP() construction."""
        if self._ner_pipeline is None:
            from transformers import pipeline as hf_pipeline
            self._ner_pipeline = hf_pipeline(
                "token-classification", model=TRANSFORMER_NER_MODEL, aggregation_strategy="simple",
            )
        return self._ner_pipeline

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        transcript_concepts = self._extract_transcript_concepts(transcript)
        soap_concepts = self._soap_extractor._extract_concepts(soap_note)

        errors = []
        errors.extend(self._judge_omissions(transcript_concepts, soap_concepts))
        errors.extend(self._judge_hallucinations(transcript_concepts, soap_concepts))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    # ------------------------------------------------------------------
    # Transcript-side extraction: transformer NER spans + QuickUMLS CUI
    # lookup + negspaCy NegEx
    # ------------------------------------------------------------------

    def _extract_transcript_concepts(self, text):
        """Transcript-side counterpart to MedspacyUmlsChecker._extract_concepts
        -- same turn-scoped, question-bridging structure (see that method's
        docstring for why: feeding a whole multi-turn transcript through a
        single NLP call merges a doctor's question with the patient's next
        turn and lets negation scope bleed across unrelated clauses), same
        return shape ({cui: {"term", "mentions": [(is_negated, False, False),
        ...], "sentences": [...]}) so the two sides plug into the same
        _judge_omissions/_judge_hallucinations. The mentions tuple's 2nd/3rd
        slots (uncertain, family) are always False here -- NegEx has no
        equivalent to ConText's is_uncertain/is_family, see module docstring.
        """
        concepts = {}
        pending_question = None
        for speaker, line_text in split_turns(text):
            if not line_text.strip():
                continue
            if speaker == "d" and line_text.strip().endswith("?"):
                pending_question = self._question_candidates(line_text)
                continue
            if speaker == "p" and pending_question:
                self._apply_pending_question(line_text, pending_question, concepts)
                pending_question = None
            self._extract_line_concepts(line_text, concepts)
        return concepts

    def _question_candidates(self, text):
        if not text.strip():
            return []
        return [(cui, term) for _, _, cui, term in self._transformer_candidates(text)]

    def _apply_pending_question(self, answer_text, pending, concepts):
        negated = bool(_NEGATIVE_ANSWER_RE.search(answer_text))
        affirmed = bool(_AFFIRMATIVE_ANSWER_RE.match(answer_text.strip()))
        if negated == affirmed:
            return
        state = (negated, False, False)
        sentence = answer_text.strip()
        for cui, term in pending:
            entry = concepts.setdefault(cui, {"term": term, "mentions": [], "sentences": []})
            entry["mentions"].append(state)
            entry["sentences"].append(sentence)

    def _extract_line_concepts(self, text, concepts):
        if not text.strip():
            return

        candidates = self._transformer_candidates(text)
        if not candidates:
            return

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
            return

        doc.ents = spacy.util.filter_spans(candidate_ents)

        # NegEx, scoped to our own injected CONCEPT ents (ent_types config
        # above) -- re-run manually (same pattern as medspacy_umls_checker.py
        # re-running medspacy_context) since the automatic pipeline pass
        # earlier in self._nlp(text) saw en_core_web_sm's own (wrong/empty
        # for clinical spans) NER ents, not these.
        self._nlp.get_pipe("negex")(doc)

        for ent in doc.ents:
            cui, term = span_to_match.get((ent.start_char, ent.end_char), (None, ent.text))
            if cui is None:
                continue
            sent_text = ent.sent.text.strip() if ent.sent is not None else ent.text
            if sent_text.endswith("?"):
                continue

            state = (bool(ent._.negex), False, False)
            entry = concepts.setdefault(cui, {"term": term, "mentions": [], "sentences": []})
            entry["mentions"].append(state)
            entry["sentences"].append(sent_text)

    def _transformer_candidates(self, text):
        """(start, end, cui, term) candidates from the transformer span-
        finder -- see module docstring. Only entity groups covering
        diseases/findings/procedures/drugs/history, only above
        TRANSFORMER_NER_MIN_SCORE, and only if the span text itself
        resolves to a real UMLS CUI via the shared QuickUMLS matcher -- a
        span this model finds with zero UMLS presence still can't enter
        the CUI-diff comparison (see module docstring)."""
        pipeline = self._get_ner_pipeline()
        try:
            entities = pipeline(text)
        except Exception:
            return []

        candidates = []
        for ent in entities:
            if ent["entity_group"] not in TRANSFORMER_NER_ENTITY_GROUPS:
                continue
            if ent["score"] < TRANSFORMER_NER_MIN_SCORE:
                continue
            start, end = ent["start"], ent["end"]
            span_text = text[start:end]
            matches = self._matcher.match(span_text, best_match=True, ignore_syntax=False)
            for group in matches:
                for m in group:
                    if is_common_word(m["term"]) or is_common_word(m["ngram"]):
                        continue
                    if _is_junk_concept(m["term"]) or _is_junk_concept(m["ngram"]):
                        continue
                    candidates.append((start, end, m["cui"], m["term"]))
                    break
                break  # one CUI per transformer-found span -- its own best QuickUMLS match
        return candidates

    # ------------------------------------------------------------------
    # Step 3: Deterministic Judging (CUI presence only, see module docstring)
    # ------------------------------------------------------------------

    def _judge_omissions(self, transcript_concepts, soap_concepts):
        errors = []
        for cui, info in transcript_concepts.items():
            if not MedspacyUmlsChecker._cui_present(soap_concepts, cui, info["term"]):
                errors.append(("omission", self._describe(info)))
        return errors

    def _judge_hallucinations(self, transcript_concepts, soap_concepts):
        errors = []
        for cui, info in soap_concepts.items():
            if not MedspacyUmlsChecker._cui_present(transcript_concepts, cui, info["term"]):
                errors.append(("hallucination", self._describe(info)))
        return errors

    @staticmethod
    def _describe(info):
        sentence = info["sentences"][0]
        negated = info["mentions"][0][0]
        state_desc = "negated" if negated else "affirmed"
        return f"{sentence} ({info['term']}: {state_desc})"

    # ------------------------------------------------------------------
    # Extra analysis (not fed to Modules/evaluate.py -- the label set has
    # no negation-agreement label type; this is reported directly in
    # extra/explainer/conversationalNLP.html instead): of every CUI present
    # on BOTH sides, how often do the two tools' independent negation
    # detectors -- medspaCy ConText vs negspaCy NegEx -- agree?
    # ------------------------------------------------------------------

    def negation_agreement(self, transcript, soap_note):
        """Returns {"agree": int, "disagree": int, "examples": [...]} over
        every CUI extracted on both the transcript (this module's negspaCy
        pipeline) and SOAP note (medspaCy ConText) sides, comparing each
        side's FIRST mention's is_negated flag only (see
        MedspacyUmlsChecker._describe's docstring for why "first mention" is
        the deterministic, common-case choice already used project-wide)."""
        transcript_concepts = self._extract_transcript_concepts(transcript)
        soap_concepts = self._soap_extractor._extract_concepts(soap_note)

        agree = 0
        disagree = 0
        examples = []
        for cui, t_info in transcript_concepts.items():
            s_info = soap_concepts.get(cui)
            if s_info is None:
                continue
            t_negated = t_info["mentions"][0][0]
            s_negated = s_info["mentions"][0][0]
            if t_negated == s_negated:
                agree += 1
            else:
                disagree += 1
                examples.append({
                    "term": t_info["term"],
                    "transcript_sentence": t_info["sentences"][0],
                    "transcript_negated": t_negated,
                    "soap_sentence": s_info["sentences"][0],
                    "soap_negated": s_negated,
                })
        return {"agree": agree, "disagree": disagree, "examples": examples}
