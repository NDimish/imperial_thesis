"""Three-step deterministic SOAP-note QA checker: UMLS-grounded concept
extraction, medspaCy ConText assertion tagging, then three independent
judgments -- omission, hallucination, and status flip.

Step 1 (Entity Extraction) -- _extract_concepts() runs the project's shared
QuickUMLS matcher (Medical condensor/umls_matching.py) against both the
transcript and the SOAP note. It's tuned to UMLS semantic types covering
diseases/findings (T047/T033/T184/T191), procedures (T059-T061), and drugs
(T121 Pharmacologic Substance, T195 Antibiotic, T200 Clinical Drug -- UMLS's
own drug types, used here in place of a separate RxNorm-only lookup since
RxNorm concepts are themselves included in the UMLS Metathesaurus this
QuickUMLS index is already built from).

Step 2 (Context Validation) -- the matched spans are injected into a
medspaCy Doc as entities and run through the medspacy_context pipe (the
ConText algorithm), which tags each one with its assertion state. medspaCy's
default rule set (loaded automatically by medspacy.load(), confirmed by
inspecting nlp.get_pipe("medspacy_context").rules directly) covers five
categories -- NEGATED_EXISTENCE, POSSIBLE_EXISTENCE, HISTORICAL,
HYPOTHETICAL, FAMILY -- which set the is_negated, is_uncertain,
is_historical, is_hypothetical, and is_family span attributes respectively.
This checker judges on is_negated/is_uncertain/is_family; purely
hypothetical mentions ("if you get chest pain, come back") are dropped
before Step 3 since they don't assert the concept either way -- the same
reasoning already validated for MedspacyCondenser, see Medical
condensor/medspacy_condenser.py.

Step 3 (Deterministic Judging) -- compares the transcript's and SOAP note's
concept sets by CUI to raise three independent, non-overlapping error types:
  - omission: a concept asserted in the transcript with no matching CUI
    anywhere in the SOAP note.
  - hallucination: a concept asserted in the SOAP note with no matching CUI
    anywhere in the transcript.
  - status_flip: a concept present in BOTH documents (same CUI) whose
    assertion state never agrees between any transcript mention and any
    SOAP note mention of it -- e.g. "denies diabetes" in the transcript but
    diabetes asserted affirmed in the SOAP note. This is deliberately a
    THIRD, separate judgment rather than folding negation into the CUI
    match key itself (as Modules/old/concept_checker.py does, matching on
    (cui, negated) pairs) -- that approach can only ever report a flipped
    concept as one hallucination plus one omission, never as the single,
    more specific "this got its polarity reversed" fact status_flip states
    directly.

Reuses Modules/old/concept_checker.py's JUNK_CONCEPT_DENYLIST (real-file
exact-similarity UMLS/English collisions, e.g. "plan" matching this
project's own "Plan:" SOAP heading) and its 5-character term-stem fallback
for CUI matching (QuickUMLS resolving inflected forms like "vomit"/
"vomiting" to slightly different CUIs was found to otherwise manufacture
dozens of spurious flags per file) -- both validated there, reused as-is
rather than re-derived here.

Deterministic throughout: QuickUMLS + medspaCy ConText only, no LLM, no
sampling, no external API call. Same two inputs always produce the same
concept sets and the same errors.
"""
import os
import re
import sys
import time

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

import medspacy
import spacy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from base import split_turns  # noqa: E402 -- path insert must run first
from umls_matching import get_matcher, is_common_word  # noqa: E402 -- path insert must run first

from Modules.base import CheckerModule

# See module docstring -- copied from Modules/old/concept_checker.py's
# JUNK_CONCEPT_DENYLIST, which found and validated every one of these as an
# exact (similarity=1.0) UMLS/generic-English collision with no clinical
# sense in this dataset, via direct inspection of real-file smoke tests.
JUNK_CONCEPT_DENYLIST = {
    "hand", "controll", "control", "other things", "move", "mind", "said",
    "close", "test", "able", "times", "life", "difficult", "little",
    "stage", "recap", "keen", "remember", "feels", "much", "normal",
    "nice", "listened", "sampled", "quite often", "examined", "therex",
    "etests", "couplet", "coinfection",
    # matches this project's own "Plan:" SOAP section header, not patient content.
    "plan",
    # Found via a direct audit of a real 10-file hallucination run: each of
    # these is a generic English word (or SOAP-formatting fragment like "of
    # note"/"Hx:") that QuickUMLS coincidentally resolves to a real UMLS
    # concept, firing on nearly every file ("live"/"lives" alone fired in 6+
    # of 10) with no clinical sense in context -- same collision pattern as
    # "hand"/"life"/"plan" above, not a borderline call.
    "live", "lives", "note", "assoc", "rule", "possible", "present",
    "patient", "flag", "attention", "secondary", "probable", "history",
}


def _is_junk_concept(term):
    return term.strip().lower() in JUNK_CONCEPT_DENYLIST


# A doctor's closed question ("Any blood in your stools?") names a concept the
# patient's terse next-turn answer ("No, not that I could see.") usually doesn't
# repeat -- see _apply_pending_question's docstring for the full rationale and a
# real confirmed example (prim12). These detect a plain yes/no answer from simple
# lexical cues rather than ConText (whose negation scope doesn't span turns);
# genuinely ambiguous answers (neither or both match) are left alone rather than
# guessed at.
_NEGATIVE_ANSWER_RE = re.compile(r"\b(no|not|n't|nope|nah|never|none)\b", re.IGNORECASE)
_AFFIRMATIVE_ANSWER_RE = re.compile(
    r"^(?:\s*(?:uh+|um+|well|so|right|ok(?:ay)?)[,.]?\s*)*(yes|yeah|yep|yup|correct)\b",
    re.IGNORECASE,
)


class MedspacyUmlsChecker(CheckerModule):
    """See module docstring for the full Extraction -> Context -> Judging
    pipeline this implements."""

    def __init__(self, install_dir=None):
        self._matcher = get_matcher(install_dir)
        self._nlp = medspacy.load()

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        transcript_concepts = self._extract_concepts(transcript)
        soap_concepts = self._extract_concepts(soap_note)

        errors = []
        errors.extend(self._judge_omissions(transcript_concepts, soap_concepts))
        errors.extend(self._judge_hallucinations(transcript_concepts, soap_concepts))
        errors.extend(self._judge_status_flips(transcript_concepts, soap_concepts))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    # ------------------------------------------------------------------
    # Steps 1-2: Entity Extraction + Context Validation
    # ------------------------------------------------------------------

    def _extract_concepts(self, text):
        """Steps 1-2 combined: finds every real UMLS concept span in text
        (Step 1), then tags each one with its ConText assertion state
        (Step 2) -- one turn/line at a time (see split_turns), not the whole
        document in a single spaCy call.

        Found necessary via a direct audit of 20 status_flip detections on
        real files: feeding a whole multi-turn transcript (or a whole SOAP
        note, whose lines are just as terse -- "No SOB / chest pain") through
        spaCy's automatic sentence segmenter in one call let it merge a
        doctor's question with the patient's next turn, or one clinical
        fragment with the next, into a single oversized "sentence". That
        merging broke two things: the question filter below only checks the
        END of the merged span, so a question buried mid-sentence wasn't
        filtered; and ConText's negation/family scope then bled across
        clauses that only looked adjacent because of the merge -- confirmed
        directly on real files, e.g. "No SOB / chest pain" wrongly negating
        the unrelated next clause "Feeling tired, weak and myalgia" in a
        SOAP note, and "Have you got any blood in your stools?" surviving
        into the patient's next-line answer instead of being filtered as a
        question. Same fix already validated for MedspacyCondenser (see
        Medical condensor/medspacy_condenser.py), applied here to both the
        transcript AND the SOAP-note side -- split_turns works on any
        newline-delimited text, SOAP note lines included, since a line with
        no "d:"/"p:" prefix just comes back with speaker=None.

        Also carries a doctor question's concepts over to the patient's next
        turn (see _apply_pending_question) -- a plain per-turn pass drops a
        question's concepts entirely (correctly -- a question doesn't assert
        anything) but has no way to recover them from a terse answer that
        doesn't repeat the words ("No, not that I could see."), which a
        full-transcript audit confirmed was manufacturing hallucination
        false positives on completely real transcript content (prim12:
        "blood in stools", "abdo pain", "fever/temp", and "headaches/muscle
        pain" were all genuinely established this way, not hallucinated).

        Returns {cui: {"term": str, "mentions": [(is_negated, is_uncertain,
        is_family), ...], "sentences": [str, ...]}} -- one mentions/sentences
        entry per distinct mention of that concept (same index = same
        mention), so a concept mentioned more than once with different
        assertions keeps all of them (see _judge_status_flips, which needs
        every mention, not just the first). "sentences" holds the actual
        source sentence each mention came from -- see _describe/_describe_flip,
        which echo it back as the reported error text instead of a bare
        "term (state)" summary, matching the convention every other checker
        module in this project uses (see e.g. AlignScoreChecker: it reports
        the whole flagged sentence, not just the concept inside it). This
        isn't just a style match -- Modules/evaluate.py's compare() matches a
        prediction to a ground-truth label by substring, and every label in
        prim57/bad notes labels lib/*.json is itself a full sentence (see
        evaluate.py's module docstring), so a bare term/state summary can
        essentially never substring-match a label; confirmed directly on a
        real 10-file run, which scored 0 TP / 430 FP before this change.
        """
        concepts = {}
        pending_question = None  # (cui, term) candidates from the last unanswered "d:" question
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
        """Step 1 only (no ConText) for a doctor's question line -- the raw
        (cui, term) UMLS matches it contains, held onto so _apply_pending_question
        can attach them to the patient's next-turn answer at whatever polarity
        that answer implies, instead of just discarding them (see
        _extract_concepts's docstring for why the question itself never asserts
        them directly)."""
        if not text.strip():
            return []
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

    def _apply_pending_question(self, answer_text, pending, concepts):
        """Attaches a preceding doctor question's concepts to the patient's
        answer at the polarity that answer's own lead-in cues imply (see
        _NEGATIVE_ANSWER_RE/_AFFIRMATIVE_ANSWER_RE) -- e.g. "Any blood in your
        stools?" / "No, not that I could see." attaches (blood, stools) as
        negated. Skips entirely, rather than guessing, when the answer has
        neither cue (a substantive answer with its own content -- already
        handled by the normal per-turn extraction that runs on it regardless)
        or both (genuinely ambiguous)."""
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
        """Runs Steps 1-2 on a single turn/line of text, adding any concepts
        found into the (mutated in place) concepts dict. See
        _extract_concepts for why this is scoped to one line at a time."""
        if not text.strip():
            return

        # Step 1: UMLS concept spans, independent of medspaCy's own (empty
        # by design -- medspacy.load() ships no target rules of its own,
        # see MedspacyCondenser's docstring) target-matcher rules.
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

        # Step 2: ConText assertion tagging.
        self._nlp.get_pipe("medspacy_context")(doc)

        for ent in doc.ents:
            cui, term = span_to_match.get((ent.start_char, ent.end_char), (None, ent.text))
            if cui is None:
                continue
            sent_text = ent.sent.text.strip() if ent.sent is not None else ent.text
            if sent_text.endswith("?"):
                continue  # a question doesn't assert the concept either way
            if ent._.is_hypothetical:
                continue  # "if you get chest pain" isn't a real assertion either way

            state = (bool(ent._.is_negated), bool(ent._.is_uncertain), bool(ent._.is_family))
            entry = concepts.setdefault(cui, {"term": term, "mentions": [], "sentences": []})
            entry["mentions"].append(state)
            entry["sentences"].append(sent_text)

    # ------------------------------------------------------------------
    # Step 3: Deterministic Judging
    # ------------------------------------------------------------------

    def _judge_omissions(self, transcript_concepts, soap_concepts):
        """A concept genuinely present in the transcript (any CUI, any
        assertion state) with no matching CUI anywhere in the SOAP note --
        content the SOAP note dropped entirely."""
        errors = []
        for cui, info in transcript_concepts.items():
            if not self._cui_present(soap_concepts, cui, info["term"]):
                errors.append(("omission", self._describe(info)))
        return errors

    def _judge_hallucinations(self, transcript_concepts, soap_concepts):
        """A concept the SOAP note asserts (any state) with no matching CUI
        anywhere in the transcript -- content the SOAP note invented."""
        errors = []
        for cui, info in soap_concepts.items():
            if not self._cui_present(transcript_concepts, cui, info["term"]):
                errors.append(("hallucination", self._describe(info)))
        return errors

    def _judge_status_flips(self, transcript_concepts, soap_concepts):
        """A concept present in BOTH documents (matched by CUI) whose
        assertion state never agrees between any transcript mention and any
        SOAP note mention of it -- the entity itself is neither missing nor
        invented, just asserted with the opposite polarity/context. See
        module docstring for why this is a separate judgment from
        omission/hallucination rather than folded into either."""
        errors = []
        for cui, soap_info in soap_concepts.items():
            transcript_info = transcript_concepts.get(cui)
            if transcript_info is None:
                continue  # not present in both -- already a hallucination above
            if set(soap_info["mentions"]) & set(transcript_info["mentions"]):
                continue  # at least one mention on each side agrees -- not a flip
            errors.append(("status_flip", self._describe_flip(soap_info, transcript_info)))
        return errors

    @staticmethod
    def _cui_present(concept_set, cui, term):
        """A concept counts as present if either its CUI matches exactly, or
        (fallback) its term shares a word stem with a term in the other set
        -- see module docstring for why this fallback exists. Deliberately
        NOT conditioned on assertion state here -- comparing states is
        _judge_status_flips's job, not this one's.

        The stem length compared is min(len(both terms), 5), not a fixed 5
        taken from just one side -- the original fixed-5 version compared
        term[:5] against candidate[:5] directly, which silently never
        matches whenever either term is under 5 characters (Python slicing a
        4-char string to [:5] just returns the 4-char string unchanged, so
        "live"[:5] == "live" can never equal "lives"[:5] == "lives", even
        though they're the same word). Confirmed on a real 10-file
        hallucination run: exactly this asymmetry was blocking "live"
        (transcript) from matching "lives" (SOAP) and "temp" from matching
        "temperature" -- both real content, both short/abbreviated on one
        side only, both invisible to the old fixed-5 comparison. 4-character
        minimum (not lower) matches MIN_MATCH_LENGTH already used for
        QuickUMLS matching elsewhere in this project, chosen so this doesn't
        reopen the false-match risk a shorter floor would carry.
        """
        if cui in concept_set:
            return True
        term_key = term.strip().lower()
        if len(term_key) < 4:
            return False
        for info in concept_set.values():
            candidate = info["term"].strip().lower()
            stem_len = min(len(term_key), len(candidate), 5)
            if stem_len >= 4 and term_key[:stem_len] == candidate[:stem_len]:
                return True
        return False

    @staticmethod
    def _describe_state(is_negated, is_uncertain, is_family):
        parts = []
        if is_negated:
            parts.append("negated")
        if is_uncertain:
            parts.append("uncertain")
        if is_family:
            parts.append("family history")
        return " & ".join(parts) if parts else "affirmed"

    @classmethod
    def _describe(cls, info):
        # A concept can carry more than one mention/state/sentence (see
        # _extract_concepts); describe the first one -- deterministic
        # (extraction order), and the common case is exactly one mention
        # per concept in a note this short. The sentence leads so it's a
        # contiguous, matchable substring for Modules/evaluate.py; the
        # term/state annotation after it is just for a human reading the log.
        sentence = info["sentences"][0]
        state_desc = cls._describe_state(*info["mentions"][0])
        return f"{sentence} ({info['term']}: {state_desc})"

    @classmethod
    def _describe_flip(cls, soap_info, transcript_info):
        t_sentence = transcript_info["sentences"][0]
        s_sentence = soap_info["sentences"][0]
        t_desc = cls._describe_state(*transcript_info["mentions"][0])
        s_desc = cls._describe_state(*soap_info["mentions"][0])
        return (
            f"{t_sentence} {s_sentence} "
            f"({soap_info['term']}: {t_desc} in transcript vs {s_desc} in SOAP note)"
        )
