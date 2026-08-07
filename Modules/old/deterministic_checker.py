"""Full four-layer deterministic hallucination/omission checker, extending
ConceptChecker (Modules/concept_checker.py, which already implements layers
1-2 below) with two more layers that specifically target the class of error
concept-matching alone cannot catch: a correctly-identified drug or vital
sign whose *number* is wrong.

The four layers, and what each one alone would miss:

  1. Dictionary & Ontology Entity Set Matching. Map both documents onto
     UMLS via QuickUMLS, compute set differences E_soap \\ E_transcript
     (hallucination) and E_transcript \\ E_soap (omission). This alone
     cannot tell "started Ibuprofen" from "stopped Ibuprofen", or "400mg"
     from "40mg" -- the drug concept is identical either way.

  2. Rule-Based Assertion & Negation Parsing (ConText/NegEx-style). Extract
     (entity, negation, certainty, temporality) instead of bare entity
     presence, so "denies chest pain" and "reports chest pain" are treated
     as different facts even though they share an entity. This alone still
     can't catch a right entity/right polarity/wrong number error, since
     assertion status says nothing about a dosage or vital sign's value.

  3. Numerical & Attribute Regex Constraint Validation (NEW here). Extract
     number+unit pairs -- dosages (mg/mcg/g/ml), vitals (blood pressure,
     temperature), frequency (x/day, od/bd/tds/qds, "twice a day"), and
     duration (days/weeks/months, clinical shorthand n/7, n/52, n/12) --
     from both documents independently of any entity they're near, and flag
     a SOAP-note numeric claim with no matching value anywhere in the
     transcript. Catches wrong-dosage/wrong-frequency/wrong-duration errors
     regardless of which drug or vital sign they're attached to, but on its
     own has no way to associate a number with the specific entity it
     modifies -- a transcript mentioning "400mg" for drug A and a SOAP note
     saying "400mg" for unrelated drug B would incorrectly look consistent.

  4. Deterministic Slot-Filling & Frame Verification (NEW here). For every
     drug/medication concept found via layer 1, extract a frame -- {drug:
     dosage, frequency, duration} -- from a text window around that
     specific mention, using the same regex machinery as layer 3. Comparing
     frames by drug CUI is what closes the gap layer 3 leaves open: it
     verifies the number is attached to the RIGHT drug, not just that the
     number appears somewhere in the document.

Layers 1-2 are inherited unchanged from ConceptChecker (including its
tested denylist and CUI+stem matching fixes -- see that module for the full
history of what was tried and why). Layers 3-4 are additive: every error
layers 1-2 would have raised is still raised here.

Deliberately deterministic throughout: regex and dictionary lookups only,
no LLM, no sampling, no external API. Same two inputs always produce the
same errors.
"""
import re
import time

from Modules.concept_checker import ConceptChecker, _is_junk_concept
from umls_matching import is_common_word  # noqa: F401 -- already on sys.path via concept_checker's import

# A closed, bounded set of English number words -- not open-ended clinical
# vocabulary, so this doesn't reopen the "hand-typing doesn't scale" problem
# flagged earlier this session. Spoken transcripts overwhelmingly say
# frequency/duration as words ("two tablets... four times a day"), while
# SOAP notes overwhelmingly use digits or clinical shorthand ("2 tabs qds") --
# normalizing words to digits before running the numeric regexes below is
# what lets the same patterns catch both registers instead of needing two
# parallel sets of patterns.
NUMBER_WORDS = {
    # "once"/"twice"/"thrice" expand to "N times" (not just "N"), since
    # FREQUENCY_COUNT_PATTERN below requires an "x"/"times" keyword between
    # the number and "a day" -- found via a direct unit test: normalizing
    # "twice a day" to "2 a day" (without "times") silently failed to match
    # anything, dropping a real frequency instead of extracting it.
    "once": "1 times", "twice": "2 times", "thrice": "3 times",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_NUMBER_WORD_PATTERN = re.compile(r"\b(" + "|".join(NUMBER_WORDS.keys()) + r")\b", re.IGNORECASE)

DOSAGE_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|micrograms?|milligrams?|g|grams?|ml|units?|mmol)\b", re.IGNORECASE)
BP_PATTERN = re.compile(r"\b(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg)?\b")
# Decimal part must be INSIDE the capturing group, not a non-capturing
# lookalike after it -- found via a direct unit test: the first version had
# "(?:\.\d)?" as a separate non-capturing group, so "38.5C" only ever
# captured "38", silently truncating every temperature with a decimal.
TEMP_PATTERN = re.compile(r"\b(3[5-9](?:\.\d)?|4[0-2](?:\.\d)?)\s*°?\s*c\b", re.IGNORECASE)
# n/7 = n days, n/52 = n weeks, n/12 = n months -- standard UK clinical
# shorthand, confirmed present in this project's real SOAP notes (e.g.
# "3/7 hx", "review in 1/52").
DURATION_SHORTHAND_PATTERN = re.compile(r"\b(\d+)\s*/\s*(7|52|12)\b")
DURATION_WORD_PATTERN = re.compile(r"\b(\d+)\s*(day|days|week|weeks|month|months)\b", re.IGNORECASE)
FREQUENCY_COUNT_PATTERN = re.compile(r"\b(\d+)\s*(?:x|times)\s*(?:a|per|/)\s*day\b", re.IGNORECASE)
FREQUENCY_SHORTHAND_PATTERN = re.compile(r"\b(od|bd|tds|qds)\b", re.IGNORECASE)
FREQUENCY_SHORTHAND_MAP = {"od": 1, "bd": 2, "tds": 3, "qds": 4}

_DURATION_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}


def _normalize_number_words(text):
    return _NUMBER_WORD_PATTERN.sub(lambda m: NUMBER_WORDS[m.group(0).lower()], text)


def extract_numeric_attributes(text):
    """Returns a set of (category, *values) tuples -- e.g. ("dosage", 400.0,
    "mg"), ("bp", 120, 80), ("duration_days", 3), ("frequency_per_day", 2)
    -- for every clinical numeric attribute found in text, in either digit
    or spoken-word form (see _normalize_number_words)."""
    normalized = _normalize_number_words(text)
    attrs = set()

    for value, unit in DOSAGE_PATTERN.findall(normalized):
        unit_norm = unit.lower().rstrip("s")
        if unit_norm == "microgram":
            unit_norm = "mcg"
        elif unit_norm == "milligram":
            unit_norm = "mg"
        elif unit_norm == "gram":
            unit_norm = "g"
        attrs.add(("dosage", round(float(value), 2), unit_norm))

    for sys_bp, dia_bp in BP_PATTERN.findall(normalized):
        # BP_PATTERN also matches unrelated "n/7"-style fractions (e.g.
        # "3/7" could parse as sys=3 dia=7), so only keep pairs in a
        # plausible blood-pressure range to avoid double-booking duration
        # shorthand as a vital sign.
        if 60 <= int(sys_bp) <= 260 and 30 <= int(dia_bp) <= 200:
            attrs.add(("bp", int(sys_bp), int(dia_bp)))

    for temp in TEMP_PATTERN.findall(normalized):
        attrs.add(("temp_c", round(float(temp), 1)))

    for value, denom in DURATION_SHORTHAND_PATTERN.findall(normalized):
        days = int(value) * (1 if denom == "7" else 7 if denom == "52" else 30)
        attrs.add(("duration_days", days))

    for value, unit in DURATION_WORD_PATTERN.findall(normalized):
        attrs.add(("duration_days", int(value) * _DURATION_UNIT_DAYS[unit.lower()]))

    for value in FREQUENCY_COUNT_PATTERN.findall(normalized):
        attrs.add(("frequency_per_day", int(value)))

    for shorthand in FREQUENCY_SHORTHAND_PATTERN.findall(normalized):
        attrs.add(("frequency_per_day", FREQUENCY_SHORTHAND_MAP[shorthand.lower()]))

    return attrs


DRUG_SEMTYPES = {"T121", "T195", "T200"}  # Pharmacologic Substance, Antibiotic, Clinical Drug
FRAME_WINDOW_CHARS = 100  # characters of context on each side of a drug mention searched for its dosage/frequency/duration

# "water" is a real T121 Pharmacologic Substance UMLS entry (dual-use --
# deliberately left un-denylisted in ConceptChecker's general JUNK list,
# since fluid-intake mentions can be genuinely clinically relevant there).
# For the drug-FRAME check specifically, treating "drink plenty of water"
# as a prescribed medication with a dosage/frequency/duration to verify is
# a clear category error, found immediately in the first smoke test (a
# duration mention near "water" was flagged as a hallucinated fact about a
# "drug"). Scoped to this one narrow, frame-specific use rather than
# touching the shared or general denylists.
FRAME_DRUG_DENYLIST = {"water"}


class DeterministicChecker(ConceptChecker):
    """ConceptChecker (layers 1-2: entity set matching + assertion parsing)
    plus numeric attribute validation and drug-frame verification (layers
    3-4). See module docstring for what each layer catches that the others
    can't.
    """

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        concept_errors, _ = super().check(transcript, soap_note)
        numeric_errors = self._check_numeric_attributes(transcript, soap_note)
        frame_errors = self._check_drug_frames(transcript, soap_note)

        errors = tuple(concept_errors) + tuple(numeric_errors) + tuple(frame_errors)
        elapsed = time.perf_counter() - start
        return errors, elapsed

    def _check_numeric_attributes(self, transcript, soap_note):
        """Layer 3: every numeric attribute asserted in the SOAP note (a
        dosage, vital sign, frequency, or duration) should have a matching
        value SOMEWHERE in the transcript, independent of which entity it's
        attached to. Only checks the hallucination direction (SOAP claims a
        number the transcript never mentions) -- the transcript naturally
        contains many numbers (ages, dates, repeated mentions) that a
        compact SOAP note has no obligation to restate individually, so
        flagging the omission direction here would mostly be noise."""
        transcript_attrs = extract_numeric_attributes(transcript)
        soap_attrs = extract_numeric_attributes(soap_note)

        errors = []
        for attr in soap_attrs:
            if attr not in transcript_attrs:
                errors.append(("hallucination", self._describe_attr(attr)))
        return errors

    def _check_drug_frames(self, transcript, soap_note):
        """Layer 4: for every drug concept present in BOTH documents (same
        CUI), compare the numeric attributes found in a text window around
        each mention. A dosage/frequency/duration claimed in the SOAP
        note's version of a drug frame that's absent from the transcript's
        version of the SAME drug's frame is a more specific, higher-value
        flag than layer 3 alone -- it confirms the number is attached to
        the wrong fact, not just present nowhere in the document."""
        transcript_frames = self._extract_drug_frames(transcript)
        soap_frames = self._extract_drug_frames(soap_note)

        errors = []
        for cui, soap_frame in soap_frames.items():
            transcript_frame = transcript_frames.get(cui)
            if transcript_frame is None:
                continue  # the drug itself is already covered by layer 1's entity matching
            missing = soap_frame["attrs"] - transcript_frame["attrs"]
            for attr in missing:
                errors.append((
                    "hallucination",
                    f"{soap_frame['term']}: {self._describe_attr(attr)} (not found near this drug in transcript)",
                ))
        return errors

    def _extract_drug_frames(self, text):
        if not text.strip():
            return {}
        matches = self._matcher.match(text, best_match=True, ignore_syntax=False)
        frames = {}
        for group in matches:
            for m in group:
                if not (m["semtypes"] & DRUG_SEMTYPES):
                    continue
                if is_common_word(m["term"]) or _is_junk_concept(m["term"]):
                    continue
                if m["term"].strip().lower() in FRAME_DRUG_DENYLIST:
                    continue
                window_start = max(0, m["start"] - FRAME_WINDOW_CHARS)
                window_end = min(len(text), m["end"] + FRAME_WINDOW_CHARS)
                window_attrs = extract_numeric_attributes(text[window_start:window_end])
                frame = frames.setdefault(m["cui"], {"term": m["term"], "attrs": set()})
                frame["attrs"] |= window_attrs
        return frames

    @staticmethod
    def _describe_attr(attr):
        category = attr[0]
        values = attr[1:]
        if category == "dosage":
            return f"dosage {values[0]}{values[1]}"
        if category == "bp":
            return f"blood pressure {values[0]}/{values[1]}"
        if category == "temp_c":
            return f"temperature {values[0]}°C"
        if category == "duration_days":
            return f"duration {values[0]} day(s)"
        if category == "frequency_per_day":
            return f"frequency {values[0]}x/day"
        return f"{category} {values}"
