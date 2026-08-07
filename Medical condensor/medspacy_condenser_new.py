import os
import re
import sys
import time

from loguru import logger

# medspacy's PyRuSH sentencizer logs at DEBUG by default, which floods the
# terminal/log files with per-token sentence-boundary traces. Quiet it down.
logger.remove()
logger.add(sys.stderr, level="WARNING")

import medspacy
import spacy
from medspacy.preprocess import Preprocessor, PreprocessingRule

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import CondenserModule, join_turns, split_turns
from umls_matching import QUICKUMLS_INSTALL_DIR, get_real_concept_spans

# ---------------------------------------------------------------------------
# Concept-source switch.
#
#   True (default) -- routes concept detection through the same hand-tuned
#   QuickUMLS matcher the other three condensers share (umls_matching.py:
#   THRESHOLD=1 by default, the coincidental-collision denylist for
#   "well"/"start"/"right"/"related"/"always"/etc, the added T040/T041/T201
#   semtypes for mood/appetite/sleep) -- CUSTOM_MATCHER_THRESHOLD below runs
#   it slightly relaxed instead of that default THRESHOLD=1, without
#   touching the shared instance the other condensers depend on. Switched to
#   the default here after direct testing (see ISSUES #1) confirmed the
#   medspacy_quickumls path below is substantially noisier on real data.
#
#   False -- medspacy's own inbuilt UMLS integration ("medspacy_quickumls", a
#   spaCy factory shipped by the medspacy_quickumls package -- confirmed this
#   is the SAME "quickumls" module umls_matching.py imports from;
#   medspacy_quickumls's install *is* the quickumls package in this
#   environment, it just doesn't register its spaCy factory until medspacy
#   imports it). Added straight into the medspacy pipeline via
#   medspacy.load()'s own quickumls_path=; doc.ents is populated by medspacy
#   itself, with none of umls_matching.py's tuning applied.
# ---------------------------------------------------------------------------
USE_CUSTOM_UMLS_MATCHER = True

# Only used when USE_CUSTOM_UMLS_MATCHER is True. A bit below the other three
# condensers' THRESHOLD=1 (exact match only), to catch near-exact surface
# variants without reopening the door to the fuzzy-match flood that
# THRESHOLD=1 exists to prevent (see umls_matching.py's tuning history).
# Tested directly against QUICKUMLS_INSTALL_DIR: does NOT fix every
# morphological gap -- "bled" still doesn't match any "bleeding" concept even
# down to threshold=0.5 (the only match at 0.5 is an unrelated term
# "bledta"), because QuickUMLS similarity is surface-string Jaccard, not
# lemma-aware. That specific class of gap would need a lemmatizing
# preprocessing step, which is a bigger change than "reduce the threshold"
# and out of scope here.
CUSTOM_MATCHER_THRESHOLD = 0.9

# Gates the ISSUES #4 lookback (see _condense_turn_text) so it only fires
# when the kept sentence actually opens with a referential/continuation
# word -- i.e. looks like it's continuing a thought from the sentence before
# it, not starting a fresh one. Deliberately conservative/non-exhaustive:
# under-triggering just leaves the ORIGINAL dangling-reference problem in
# rare cases; over-triggering (the unconditional version this replaced)
# concretely broke plain filler removal ("Hi hello. You have chest pain."
# stopped condensing at all).
REFERENTIAL_START = re.compile(
    r"^(it|this|that|he|she|they|so|and|but)\b", re.IGNORECASE
)


class MedspacyCondenserNew(CondenserModule):
    """Removes non-clinical filler at SENTENCE granularity (not turn
    granularity like every other condenser in this folder), using medspacy
    for sentence splitting, medspacy's ConText for assertion status (all 5
    modifier attributes -- see _is_affirmed below), and either medspacy's own
    UMLS integration or the shared tuned matcher (USE_CUSTOM_UMLS_MATCHER
    above) for concept detection.

    A turn is rebuilt from only its clinically-relevant, affirmed sentences
    instead of being kept or dropped whole, e.g.:

        d: Hi hello. You have a broken arm.
        -> d: You have a broken arm.

    ISSUES:

    1. medspacy's own inbuilt UMLS path (USE_CUSTOM_UMLS_MATCHER = False) is
       unfiltered -- CONFIRMED on real data, not just a theoretical risk.
       medspacy_quickumls's own defaults accept EVERY UMLS semantic type
       (umls_matching.py's ACCEPTED_SEMTYPES exists precisely because the
       unfiltered set is too broad). Run directly against
       QUICKUMLS_INSTALL_DIR (the real licensed install, not even
       medspacy_quickumls's tiny bundled demo sample) on the sentence "You
       have a broken arm.": both "You" and "arm" matched as UMLS concepts --
       "You" alone would keep an otherwise-empty sentence. Same test also
       matched "worse" and "back" (from "if the pain gets worse come back")
       as concepts. This is why USE_CUSTOM_UMLS_MATCHER now defaults to True
       -- confirmed that removes all four false positives above in the same
       test. Still an open gap on the custom-matcher path: found "related"
       and "always" as two more coincidental sim=1.0 collisions during this
       testing round (now added to umls_matching.py's GENERIC_WORD_DENYLIST,
       shared with the other three condensers) -- there is no reason to
       think that list is exhaustive, just that it's caught everything found
       by testing so far.

    2. ConText's 5 modifier attributes are split into two groups, not all
       treated as exclusionary -- this is a judgment call, not something
       medspacy decides for you:
         - existence axis (is_negated, is_hypothetical, is_uncertain): a
           concept under any of these describes something that may NOT
           actually be the case, so the sentence is dropped unless it has
           another, affirmed concept.
         - subject/temporal axis (is_historical, is_family): these describe
           WHO or WHEN, not IF, so a family or past mention is kept -- same
           rationale medspacy_condenser.py documents for keeping historical/
           family mentions. This is a real design choice a reviewer could
           disagree with (e.g. is an "uncertain" finding like "possible
           pneumonia" genuinely non-clinical, or worth keeping the way
           historical/family is?) -- flagging it rather than presenting it
           as objectively correct.
       Unlike medspacy_condenser.py, this file does NOT special-case doctor
       turns to skip the hypothetical check (that was a documented,
       dataset-audited call there -- doctor's conditional advice vs a
       patient's hedge -- not replicated here since it wasn't asked for and
       hasn't been separately verified against this per-sentence version).
       The "?" exception (keep a question on concept presence alone,
       assertion status doesn't apply to a question) IS kept, but now
       evaluated per SENTENCE rather than per turn, since one turn can mix a
       question sentence with a statement sentence.

    3. Sentence splitting depends on medspacy's PyRuSH sentencizer, which is
       tuned for punctuated clinical prose, not transcribed speech. Spoken
       dialogue lines with run-on clauses, false starts, or missing
       punctuation ("yeah no I'm fine so anyway my arm still hurts") may
       segment as one giant sentence -- at which point sentence-level
       filtering degrades back to turn-level filtering for that line, since
       one affirmed concept anywhere in the "sentence" keeps the whole
       thing. On real transcripts (prim1.txt, prim15.txt) PyRuSH actually
       handled disfluent speech reasonably ("Um, should we start?" and "You
       have a broken arm." split correctly) -- this held up better than
       expected, but wasn't checked past 2 of the 10 trial files.

    4. FIXED, in two parts -- was: dropping individual sentences could orphan
       a pronoun/referent in the sentence kept next to it. The originally
       CONFIRMED case on prim15.txt ("...I'm not always very good about
       wearing gloves. So that could be related." losing its middle
       sentence, surfacing as a dangling "So that could be related.") turned
       out to be caused by "related" itself being a coincidental sim=1.0
       UMLS collision (ISSUES #1) -- now that it's denylisted, that sentence
       no longer independently survives at all, so the dangling case is
       gone at the source, not by lookback.
       Implemented the general lookback anyway (a real dangling-reference
       case is still plausible even without a collision causing it), keeping
       the sentence immediately BEFORE any kept sentence, one step only --
       but gated on the kept sentence actually STARTING with a referential
       or continuation word (REFERENTIAL_START: it/this/that/he/she/they/
       so/and/but). This gate was NOT optional/defensive -- an unconditional
       first version of this rule was tested and concretely broke plain
       filler removal: "Hi hello. You have chest pain." stopped condensing
       at all, because "Hi hello." got pulled back in front of "You have
       chest pain." merely for being its immediate predecessor, with no
       actual referential link between them. REFERENTIAL_START fixes that
       specific regression and is deliberately a short, non-exhaustive list
       -- under-triggering just leaves the original dangling-reference
       problem in rare cases, which is the safer failure direction.
       Chosen over "keep full quote" (the whole turn) because that would
       give up sentence-granularity condensing entirely for any turn that
       trips it. Residual gap: an antecedent MORE than one sentence back
       still dangles, and the referential-word list is a heuristic, not a
       real coreference resolver.

    5. FIXED (was: a kept question's one-word "No." answer right after it
       had no UMLS concept of its own and was silently dropped) --
       CONFIRMED and fixed on prim15.txt, where an entire negative
       review-of-systems block (fever, nausea/vomiting, bowel, urinary,
       allergy, smoking, skin-problem, asthma/bowel-history -- all "No.")
       was vanishing, leaving each screening question dangling with no
       visible answer. condense() now tracks whether the last kept sentence
       ENDS in "?"; if the very next turn would otherwise produce nothing at
       all, that turn's original, unfiltered text is kept as-is instead --
       i.e. "keep full quote", but scoped ONLY to this specific trigger
       (turn immediately follows a kept question AND would otherwise be
       fully dropped), not applied generally the way it would be under
       issue #4. This is deliberately the raw quote rather than an attempt
       to guess which fragment is "the answer" -- once the concept filter
       has found nothing, there's no reliable signal left to trim on.
       A similar "keep the turn after a kept question" lookback was tried
       and reverted in the other three (turn-level) condensers (see
       quickumls_condenser.py) -- but that verdict was reached under a KDE
       groundedness metric later found to be an unreliable judge of real
       quality (see umls_matching.py's tuning history), and the concrete
       loss found here (whole pertinent-negative ROS blocks vanishing) is a
       sharper case than what motivated re-testing it there, which is why
       it's implemented here despite that history.
       Tracking this by "did the immediately preceding TURN end in a
       question" was itself found broken on prim10.txt/prim11.txt: a
       <UNIN/>/<INAUDIBLE_SPEECH/> tag that strips to a fully empty turn, or
       a bare filler turn like "Um" (itself rescued by this same fallback),
       routinely sits BETWEEN the doctor's question and the patient's real
       answer as separate transcript lines -- "when was your last period?"
       -> "two weeks ago" and "any blood in your stools?" -> "no...nothing
       like that" were both silently dropped this way. condense() now
       tracks a pending_question_asker across turns instead: the window
       stays open through any number of empty turns or same-speaker filler
       turns, and only closes once the OTHER speaker produces a turn where
       the concept filter itself found something (has_own_content), not
       merely once *a* turn from them appears. Residual gap: if the
       answering speaker never produces a turn with its own detected
       concept (a long run of pure filler), the window can stay open
       further into the conversation than strictly the one exchange it was
       meant for -- not seen in the 10-file test round, but a real
       possibility given how it's implemented.

    6. NEW REGRESSION found while re-testing after switching the default to
       USE_CUSTOM_UMLS_MATCHER -- umls_matching.py's shared MIN_MATCH_LENGTH
       is 4, so 3-letter clinical words never match at all. Confirmed this
       breaks the exact worked example from the top of this docstring: "You
       have a broken arm." now condenses to nothing, because "arm" (3
       chars) is below MIN_MATCH_LENGTH and CUSTOM_MATCHER_THRESHOLD doesn't
       change that -- lowering the similarity threshold and raising/lowering
       the minimum match length are two different knobs. Confirmed directly
       that min_match_length=3 does let "arm" match (as C1140618, a real
       T023 Body Part concept) -- but also lets "You" match again (a
       spurious sim=1.0 T033 collision, not currently denylisted), so
       lowering it isn't free. Left MIN_MATCH_LENGTH untouched here since
       changing it affects the shared matcher instance's tuning for the
       other three condensers too (or, if given its own override the way
       threshold now has one, would need its own denylist audit first) --
       flagging this rather than silently deciding it for you. Other short
       clinical words (leg, hip, eye, ear, rib) likely have the same gap.

    7. The medspacy Preprocessor rules below duplicate base.clean_transcript
       (which already strips <UNSURE>/<UNIN/>/<INAUDIBLE_SPEECH/> upstream,
       once, before any condenser sees the transcript -- see NLP_Main.py and
       main.py). They're wired in anyway per the brief ("run preprocessing
       rule from medspacy"), and as a side benefit make this file safe to
       run standalone on raw, uncleaned transcript text -- but that's now
       two independently-maintained regex sets for the same three tags, and
       they need to be kept in sync by hand if the annotation format ever
       changes.
    """

    def __init__(self, quickumls_install_dir=None):
        self._install_dir = quickumls_install_dir or QUICKUMLS_INSTALL_DIR

        if USE_CUSTOM_UMLS_MATCHER:
            # No target_matcher/quickumls component -- entities are injected
            # by hand in _condense_turn_text from the shared matcher instead,
            # same pattern medspacy_condenser.py uses.
            self._nlp = medspacy.load(medspacy_enable=["medspacy_pyrush", "medspacy_context"])
        else:
            if not self._install_dir:
                logger.warning(
                    "QUICKUMLS_INSTALL_DIR is not configured -- medspacy_quickumls "
                    "will fall back to its own bundled demo UMLS sample, not a "
                    "real install."
                )
            # medspacy.load() builds the pipeline in a fixed internal order
            # (pyrush -> target_matcher -> quickumls -> context -> ...)
            # regardless of this list's order, so quickumls's entities are
            # guaranteed to exist before context assesses them.
            self._nlp = medspacy.load(
                medspacy_enable=["medspacy_pyrush", "medspacy_quickumls", "medspacy_context"],
                quickumls_path=self._install_dir,
            )

        self._nlp.tokenizer = Preprocessor(self._nlp.tokenizer)
        self._nlp.tokenizer.add(self._preprocessing_rules())

    @staticmethod
    def _preprocessing_rules():
        """medspacy preprocessing rules -- run on raw turn text before
        tokenization, same three annotation tags base.clean_transcript
        already handles upstream (see ISSUES #7 above)."""
        return [
            PreprocessingRule(
                r"<UNSURE>(.*?)</UNSURE>",
                repl=r"\1",
                flags=re.IGNORECASE | re.DOTALL,
                desc="Unwrap <UNSURE>...</UNSURE>, keeping the enclosed words.",
            ),
            PreprocessingRule(
                r"<UNIN\s*/>",
                repl="",
                flags=re.IGNORECASE,
                desc="Drop <UNIN/> (unintelligible speech marker).",
            ),
            PreprocessingRule(
                r"<INAUDIBLE_SPEECH\s*/>",
                repl="",
                flags=re.IGNORECASE,
                desc="Drop <INAUDIBLE_SPEECH/> marker.",
            ),
        ]

    def condense(self, transcript):
        start = time.perf_counter()

        turns = split_turns(transcript)
        kept = []
        # Tracks who's owed an answer, not just "did the last line end in a
        # question mark" -- CONFIRMED necessary on prim10.txt/prim11.txt: a
        # <UNIN/>/<INAUDIBLE_SPEECH/> tag that strips to a completely empty
        # turn (base.clean_transcript leaves an empty line, split_turns still
        # emits it as its own turn), or a bare filler turn like "Um" that
        # itself gets kept via the ISSUES #5 fallback, sits BETWEEN the
        # doctor's question and the patient's real answer in the raw
        # transcript. The original version reset the "?" flag after every
        # single turn, so it fired for the empty/filler turn immediately
        # after the question and then went back to False, missing the real
        # answer one or two turns later -- silently dropping "when was your
        # last period?" -> "two weeks ago" and "any blood in your stools?"
        # -> "no ... nothing like that" entirely. Now the grace period
        # stays open across any number of empty or same-speaker turns, and
        # only closes once the OTHER speaker actually produces a non-empty,
        # non-question turn (a real answer).
        pending_question_asker = None
        for speaker, text in turns:
            is_answer_turn = bool(
                text.strip() and pending_question_asker is not None and speaker != pending_question_asker
            )
            kept_text, ends_in_question, has_own_content = self._condense_turn_text(text, is_answer_turn)
            if kept_text:
                kept.append((speaker, kept_text))

            if not text.strip():
                continue
            if ends_in_question:
                pending_question_asker = speaker
            elif is_answer_turn and has_own_content:
                # Only a turn with SOME independently-detected concept closes
                # the window -- confirmed necessary on prim11.txt: "Have you
                # got any blood in your stools?" was answered "Um" (its own
                # turn, no concept, rescued by the fallback above) THEN "no
                # I don't... nothing like that" (the real answer, a second,
                # separate turn from the same speaker). A pure filler answer
                # closing the window immediately would leave that second
                # turn stranded exactly like the bug this was meant to fix.
                pending_question_asker = None
        condensed = join_turns(kept)

        elapsed = time.perf_counter() - start
        return condensed, elapsed

    def _condense_turn_text(self, text, prev_ends_in_question):
        """Returns (kept_text, ends_in_question, has_own_content) for one turn.
        ends_in_question feeds into the NEXT call as prev_ends_in_question --
        see the "keep the answer to a kept question" fallback below (ISSUES
        #5). has_own_content is True only when the concept filter itself
        found something in THIS turn (as opposed to kept_text coming from the
        raw-quote fallback, or being empty) -- condense() uses it to decide
        whether the answer-to-a-question window is really closed yet."""
        text = text.strip()
        if not text:
            return "", False, False

        doc = self._nlp(text)

        if USE_CUSTOM_UMLS_MATCHER:
            spans = get_real_concept_spans(text, self._install_dir, threshold=CUSTOM_MATCHER_THRESHOLD)
            candidate_ents = []
            for match_start, match_end in spans:
                span = doc.char_span(match_start, match_end, label="CONCEPT", alignment_mode="expand")
                if span is not None:
                    candidate_ents.append(span)
            doc.ents = spacy.util.filter_spans(candidate_ents)
            self._nlp.get_pipe("medspacy_context")(doc)

        sentences = list(doc.sents)
        keep_flags = [self._sentence_is_clinical(doc, sent) for sent in sentences]

        # Lookback -- keep the sentence immediately before a kept sentence
        # too, one step only (no cascading further back), so a pronoun or
        # referent split across a sentence boundary keeps its antecedent
        # (ISSUES #4). Gated on the kept sentence actually STARTING with a
        # referential/continuation word -- confirmed necessary, not just
        # cautious: an unconditional version of this rule pulled "Hi hello."
        # back in front of ANY kept sentence merely for being its immediate
        # predecessor ("Hi hello. You have chest pain." stopped condensing
        # at all), reintroducing exactly the greeting filler this file
        # exists to remove. Computed from the ORIGINAL flags, not the flags
        # as they're being mutated, so a sentence kept only because of THIS
        # rule doesn't itself trigger pulling in one more sentence before it.
        original_flags = list(keep_flags)
        for i in range(1, len(sentences)):
            if (
                original_flags[i]
                and not original_flags[i - 1]
                and REFERENTIAL_START.match(sentences[i].text.strip())
            ):
                keep_flags[i - 1] = True

        if any(keep_flags):
            kept_sentences = [
                sent.text.strip() for sent, keep in zip(sentences, keep_flags) if keep
            ]
            kept_text = " ".join(s for s in kept_sentences if s)
            return kept_text, kept_sentences[-1].endswith("?"), True

        # Lookahead -- nothing in this turn cleared the concept filter on its
        # own. If the immediately preceding kept output ended in a question,
        # this turn is very likely the direct answer to it, so keep the
        # ORIGINAL, unfiltered turn text rather than dropping it (ISSUES #5)
        # -- a bare "No." to a screening question has no UMLS concept of its
        # own, but dropping it leaves the question looking unanswered/never
        # asked. Keeping the full raw quote here (vs. trying to guess which
        # fragment is "the answer") is deliberate: there's no reliable signal
        # for that once the concept filter has already found nothing.
        # has_own_content=False here (not True) is deliberate: this text
        # survived on the STRENGTH OF THE QUESTION, not its own content, so
        # condense() keeps the window open for a possible continuation turn.
        if prev_ends_in_question:
            return text, text.endswith("?"), False

        return "", False, False

    def _sentence_is_clinical(self, doc, sent):
        sent_ents = [ent for ent in doc.ents if ent.start >= sent.start and ent.end <= sent.end]
        if not sent_ents:
            return False

        # Assertion status doesn't apply to a question -- asking about a
        # symptom is real clinical content even if "any"/"no" trigger words
        # appear in it (same syntactic distinction medspacy_condenser.py
        # uses, applied per sentence here instead of per turn).
        if sent.text.strip().endswith("?"):
            return True

        return any(self._is_affirmed(ent) for ent in sent_ents)

    @staticmethod
    def _is_affirmed(ent):
        """True if this entity is on the "affirmed" side of ALL 5 ConText
        modifier attributes that matter for existence -- see ISSUES #2 for
        why is_historical/is_family are deliberately excluded from this
        check rather than treated as disqualifying."""
        return not (ent._.is_negated or ent._.is_hypothetical or ent._.is_uncertain)
