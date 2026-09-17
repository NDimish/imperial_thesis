"""Number-aware extension of Modules/medspacy_umls_checker.py: adds a fourth,
deterministic judgment on top of the original three (omission, hallucination,
status_flip) -- finds every number mentioned in the transcript and in the
SOAP note, matches a note-side number to its best-context-matching
transcript-side number by shared surrounding content words, and flags a
hallucination (detail_type "number edit") for every note-side number that
isn't POSITIVELY confirmed correct -- i.e. it flags a note number unless a
context-matching transcript number was found stating the same value.
Deliberately asymmetric: silence requires evidence, a flag does not. This
means a note number with no transcript anchor at all (dataset fact #1/#2
below) is flagged, not skipped -- the earlier version of this checker (see
git history) required positive evidence of a MISMATCH before flagging, which
left every "no anchor either way" case (the majority of this dataset's
number edits, see below) silently uncaught. Confirmed by direct case audit
(prim1 "Opening bowels x60/day", prim33 "cough more than 300 weeks", prim43
"80 weeks.") that the old version's silence-by-default was the single
biggest source of missed number edits, bigger than the concept-matching
threshold itself -- this version trades a large amount of precision for
recall on exactly that failure mode, deliberately, on the reasoning that an
unverifiable number in a note is itself something worth a clinician's eyes,
not something to stay quiet about.

Why the original checker can never catch this on its own: a quantity
("300mg", "60/day", "38.5 degrees") is not a UMLS concept and carries no
CUI, so MedspacyUmlsChecker's CUI-set comparison (Step 3 in that module's
docstring) is structurally blind to it -- confirmed directly, that checker's
own number-edit row is 0/33 caught on the full 57-file dataset (see
data_jsons/fig25_medspacy_tp_fn_by_corruption_type.json). This module targets
exactly that blind spot; everything else (concept extraction, ConText
tagging, omission/hallucination/status_flip judging) is inherited unchanged
from MedspacyUmlsChecker.

Two dataset facts this design had to be built around, both confirmed
directly against prim57/cleaned transcripts/*.txt and prim57/bad notes
labels lib/*.txt before writing a single line of matching logic:

  1. Zero of the 57 cleaned transcripts contain a single digit character
     (`re.search(r"\\d", text)` is False on all 57 files). Every spoken
     number, if said at all, is transcribed as a word ("sixty", "one hundred
     and forty"). A checker that only looks for \\d+ on the transcript side
     has no possible anchor to match against, ever -- so this module ships
     its own deterministic word-number parser (_consume_number_words) and
     runs it over the transcript, not just the note.

  2. Even after converting every number word to its digit value, only 18 of
     the dataset's 33 number-edit ground-truth labels have their corrupted
     value appearing ANYWHERE in the transcript at all -- checked with no
     context restriction whatsoever, the most permissive possible test (any
     substring match, anywhere in the file). The remaining 15 are
     clinician-inferred quantifications with no numeric utterance on either
     side to recover from at all, e.g. prim1's SOAP note states "Opening
     bowels x60/day" where the transcript's own words are just "I go quite
     frequently" -- no number, spoken or spelled, appears anywhere near it.
     No deterministic lexical method, this one included, can positively
     CONFIRM those are wrong -- but per the flag-by-default policy above,
     "cannot confirm" is exactly the condition that gets a number flagged
     now, not silently passed over.

A third thing this version fixes: Modules/base.split_sentences's
MIN_SENTENCE_WORDS=4 floor exists to drop meaningless transcript filler
turns ("Yeah.", "Fine."), but the SOAP note side of this checker was
reusing that same floor and, with it, silently dropping short-but-real
clinical fragments before a number in them was ever even extracted -- e.g.
prim43's note sentence "80 weeks." (2 words) never reached the matching
logic at all, despite the transcript stating a real, comparable value
("six weeks or so"). _extract_number_spans_note now splits the note on the
same sentence-boundary regex Modules/base.py uses but WITHOUT that word-count
floor -- see _split_note_sentences.

Deterministic throughout, same guarantee as the parent checker: no LLM, no
sampling, no external API call -- pure regex/token-window number extraction,
pure Python word-number parsing, pure set-overlap context matching. Same two
inputs always produce the same number spans and the same errors.
"""
import json
import os
import re
import sys
import time

from Modules.base import SENTENCE_SPLIT_PATTERN
from Modules.medspacy_umls_checker import MedspacyUmlsChecker
from Modules.risk_taxonomy import classify_severity

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from base import split_turns  # noqa: E402 -- path insert must run first

_STOPWORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spacy_stopwords.json")
try:
    with open(_STOPWORDS_PATH, "r", encoding="utf-8") as _f:
        _STOPWORDS = set(json.load(_f))
except (FileNotFoundError, json.JSONDecodeError, OSError):
    _STOPWORDS = set()

_WORD_TOKEN_RE = re.compile(r"[A-Za-z]+")
_DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000}
_NUMBER_WORDS = set(_ONES) | set(_TENS) | set(_SCALES)

# How many shared content-word stems a note-side and transcript-side number's
# contexts must have before a matching value there is trusted enough to KEEP
# THE NOTE'S NUMBER SILENT (see _judge_number_edits's flag-by-default
# policy) -- not, as in the previous version of this module, the bar for
# raising a flag. Kept at 1 deliberately: dataset fact #2 in the module
# docstring already shows the candidate pool is sparse (spoken numbers are
# rare and short-lived per file), so raising this further would mostly
# silence-by-mistake fewer genuinely-correct numbers rather than catch more
# edits -- since under this policy, raising the bar makes MORE numbers get
# flagged, not fewer.
MIN_SHARED_CONTEXT_WORDS = 1
MIN_CONTEXT_WORD_LEN = 3


def _stem(word):
    """Same min(len, 5)-char stem fallback MedspacyUmlsChecker._cui_present
    uses for CUI terms (see that module's docstring for why), reused here so
    "cigarette"/"cigarettes" or "week"/"weeks" still count as the same
    context anchor on both sides of the comparison."""
    word = word.strip().lower()
    if len(word) < MIN_CONTEXT_WORD_LEN:
        return None
    return word[: min(len(word), 5)]


def _consume_number_words(tokens, start_idx):
    """Greedy English number-word parser -- the standard "accumulate ones/
    tens into the current hundred-group, roll into the total on 'thousand'"
    grammar every text2num-style implementation uses. Starting at
    tokens[start_idx], consumes the longest run of number words (optionally
    glued by "and", e.g. "one hundred and forty") and returns
    (value, end_idx_exclusive). Returns None if tokens[start_idx] isn't
    itself a number word.

    Known, accepted limitation: an idiom containing a number word ("no one",
    "one of the") parses as a literal number with no way to tell idiom from
    quantity at the token level alone. This is left uncorrected deliberately
    -- the context-overlap requirement in _judge_number_edits is what keeps
    an idiomatic false extraction from ever turning into a false flag in
    practice (it has nothing plausible to context-match against), rather
    than trying to hand-list every idiom that contains a number word.
    """
    word = tokens[start_idx]
    if word not in _NUMBER_WORDS:
        return None

    total = 0
    current = 0
    i = start_idx
    consumed = False
    while i < len(tokens):
        w = tokens[i]
        if w in _ONES:
            current += _ONES[w]
            consumed = True
            i += 1
        elif w in _TENS:
            current += _TENS[w]
            consumed = True
            i += 1
        elif w in _SCALES:
            scale = _SCALES[w]
            current = (current or 1) * scale
            if scale >= 1000:
                total += current
                current = 0
            consumed = True
            i += 1
        elif w == "and" and consumed and i + 1 < len(tokens) and tokens[i + 1] in _NUMBER_WORDS:
            i += 1  # skip the glue word, keep parsing the next number word
        else:
            break

    if not consumed:
        return None
    return total + current, i


def _numbers_in_line(text):
    """Returns [(value, raw_text), ...] for every digit run (\\d+(?:\\.\\d+)?)
    and every spelled-out number-word run (_consume_number_words) found in
    text. Run over both the transcript (where only word-numbers will ever
    fire -- see module docstring, dataset fact #1) and the SOAP note (where
    only digits are expected in this dataset, but word-numbers are checked
    too for generality/robustness rather than assuming)."""
    results = []
    for m in _DIGIT_RE.finditer(text):
        try:
            results.append((float(m.group()), m.group()))
        except ValueError:
            continue

    words = _WORD_TOKEN_RE.findall(text.lower())
    i = 0
    while i < len(words):
        parsed = _consume_number_words(words, i)
        if parsed is None:
            i += 1
            continue
        value, end_idx = parsed
        raw = " ".join(words[i:end_idx])
        results.append((float(value), raw))
        i = end_idx
    return results


def _split_note_sentences(text):
    """Same sentence-boundary regex Modules.base.split_sentences uses
    (split on ./!/? or any run of newlines), WITHOUT that function's
    MIN_SENTENCE_WORDS>=4 floor. That floor is there to drop meaningless
    transcript filler ("Yeah.", "Fine.") -- applied to a SOAP note instead,
    it also drops real, short clinical fragments a number can live in
    entirely alone (e.g. "80 weeks.", 2 words) before a number in them is
    ever extracted. See module docstring's third fix for the confirmed case
    this was silently losing (prim43)."""
    return [s for s in SENTENCE_SPLIT_PATTERN.split(text.strip()) if s.strip()]


def _line_context_words(text):
    """Content-word stems (see _stem) for one line/sentence, with number
    tokens and stopwords removed -- the "topic" a number's value is supposed
    to be answering, e.g. {"alcoh", "week"} for "I don't drink more than
    three evenings a week"."""
    words = _WORD_TOKEN_RE.findall(text.lower())
    stems = set()
    for w in words:
        if w in _NUMBER_WORDS or w in _STOPWORDS:
            continue
        stem = _stem(w)
        if stem:
            stems.add(stem)
    return stems


class MedspacyUmlsNumberChecker(MedspacyUmlsChecker):
    """MedspacyUmlsChecker plus a fourth deterministic judgment: flags every
    SOAP-note number that isn't positively confirmed by a context-matching,
    same-valued transcript number. See module docstring for the mechanism,
    the flag-by-default policy this implies, and the dataset facts (zero
    transcript digits; most number edits have no transcript anchor at all)
    that motivated it."""

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        transcript_concepts = self._extract_concepts(transcript)
        soap_concepts = self._extract_concepts(soap_note)

        errors = []
        errors.extend(self._judge_omissions(transcript_concepts, soap_concepts))
        errors.extend(self._judge_hallucinations(transcript_concepts, soap_concepts))
        errors.extend(self._judge_status_flips(transcript_concepts, soap_concepts))
        errors.extend(self._judge_number_edits(transcript, soap_note))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    # ------------------------------------------------------------------
    # Number extraction (new Step, parallel to the parent's Steps 1-2)
    # ------------------------------------------------------------------

    def _extract_number_spans_transcript(self, transcript):
        """One span per number mention in the transcript. Context is the
        line's own non-stopword content words; for a patient turn, the
        immediately preceding doctor turn's content words are folded in too
        -- the doctor's question usually names the topic ("what about
        alcohol?"), the patient's answer states the value ("three evenings a
        week"), and neither line alone carries both halves. Same "the
        question sets the topic, the answer states the fact" reasoning
        already validated for concepts in MedspacyUmlsChecker's
        _apply_pending_question, applied here to numbers instead."""
        spans = []
        prev_doctor_context = set()
        for speaker, line_text in split_turns(transcript):
            if not line_text.strip():
                continue
            if speaker == "d":
                prev_doctor_context = _line_context_words(line_text)
                continue
            line_context = _line_context_words(line_text)
            context = line_context | (prev_doctor_context if speaker == "p" else set())
            for value, raw in _numbers_in_line(line_text):
                spans.append({"value": value, "raw": raw, "context": context, "sentence": line_text.strip()})
        return spans

    def _extract_number_spans_note(self, soap_note):
        """One span per number mention in the SOAP note, one sentence
        (_split_note_sentences -- same boundary regex as
        AlignScoreChecker/EmbedKDECheck's split_sentences, but without its
        4-word floor -- see module docstring's third fix) at a time, so a
        span's context never crosses into an unrelated clinical fact two
        sentences over."""
        spans = []
        for sentence in _split_note_sentences(soap_note):
            found = _numbers_in_line(sentence)
            if not found:
                continue
            context = _line_context_words(sentence)
            for value, raw in found:
                spans.append({"value": value, "raw": raw, "context": context, "sentence": sentence.strip()})
        return spans

    # ------------------------------------------------------------------
    # Judgment (new Step, alongside the parent's Step 3)
    # ------------------------------------------------------------------

    def _judge_number_edits(self, transcript, soap_note):
        """For every number in the SOAP note, finds its best-context-matching
        number in the transcript (most shared content-word stems, ties broken
        by extraction order) and flags the note's sentence as a hallucinated
        (edited) number UNLESS that match positively confirms it: a
        transcript number sharing at least MIN_SHARED_CONTEXT_WORDS of
        context AND stating the identical value. Everything else gets
        flagged -- a context match with a different value (a confirmed
        edit), a context match too thin to trust either way, or no
        transcript number in the same topic at all (see module docstring
        for why this is most of this dataset's real number edits). Silence
        requires evidence now; a flag does not -- see module docstring's
        opening paragraph for why this is a deliberate precision-for-recall
        trade, not an oversight.

        A note number with a completely empty context (see
        _line_context_words -- typically bare plan/list-item numbering like
        "1.", which the corruption generator itself never targets, see
        datamakerfiles/prim_lib_injection.py's _swap_number) is skipped
        entirely rather than flagged: there's no clinical content to check
        it against and no way this policy could ever confirm it either, so
        flagging it would be pure noise, not a real "cannot verify" case.

        Emits (type, severity, detail_type, error_text) 4-tuples -- unlike
        the parent checker's bare (type, error_text) 2-tuples -- so severity
        (via the same Modules/risk_taxonomy.classify_severity rule the
        corruption generator itself grades number edits by) and detail_type
        land in Modules/evaluate.py's by_severity/by_detail_type breakdowns
        the same way Modules/high_risk_checker.py's errors already do.
        Modules/evaluate.py's compare() branches on tuple length per error,
        so mixing 4-tuples into the same errors sequence as the parent's
        2-tuples is safe.
        """
        transcript_spans = self._extract_number_spans_transcript(transcript)
        note_spans = self._extract_number_spans_note(soap_note)
        if not note_spans:
            return []

        errors = []
        seen_sentences = set()
        for note_span in note_spans:
            if not note_span["context"]:
                continue  # nothing to check this number against either way

            best_overlap = 0
            best_match = None
            for t_span in transcript_spans:
                overlap = len(note_span["context"] & t_span["context"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = t_span

            confirmed_correct = (
                best_match is not None
                and best_overlap >= MIN_SHARED_CONTEXT_WORDS
                and best_match["value"] == note_span["value"]
            )
            if confirmed_correct:
                continue

            sentence = note_span["sentence"]
            if sentence in seen_sentences:
                continue  # a sentence with 2+ numbers only gets flagged once
            seen_sentences.add(sentence)

            severity = classify_severity("number edit", sentence)
            errors.append(("hallucination", severity, "number edit", sentence))
        return errors
