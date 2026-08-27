"""Deterministic, note-only content extractor -- a "quick basic summary" of a
SOAP note's key content, structured the same lightweight way as
Modules/risk_taxonomy.py (plain functions/constants, no transcript needed,
no checker class): pull out every negation, every prescribed medicine, and
every inference/impression-style statement, tagged with the SOAP section it
falls in.

Unlike Modules/high_risk_checker.py, there is no transcript to compare
against here -- "inference" can't mean "content that goes beyond what the
transcript said" the way it did in prim57/labels_inferences. Instead it's a
single-document, lexical definition: a statement that reads as the doctor's
own synthesized judgment (the whole Assessment/Impression section, plus any
line elsewhere carrying a hedging/differential cue like "likely", "?", or
"r/o") rather than a plain recorded fact.

Reuses SECTION_PATTERNS from Modules/high_risk_checker.py (module-level
regexes, safe to import without triggering that module's medspaCy/QuickUMLS
model load, which only happens inside HighRiskChecker.__init__) and
DRUG_SEMTYPES from Modules/risk_taxonomy.py so all three modules agree on
what counts as a section header / a drug, instead of three drifting copies.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from umls_matching import get_matcher, is_common_word  # noqa: E402

from Modules.high_risk_checker import SECTION_PATTERNS  # noqa: E402
from Modules.risk_taxonomy import DRUG_SEMTYPES  # noqa: E402

# Same coincidental-collision pattern documented at length in
# Medical condensor/umls_matching.py's GENERIC_WORD_DENYLIST, just for this
# module's own DRUG_SEMTYPES scan instead of that file's broader
# ACCEPTED_SEMTYPES one -- confirmed directly against this project's own
# QuickUMLS install and a full 57-file audit of every distinct term this
# extractor flagged as a "medicine" (checked context for every borderline
# one -- see scratchpad diag_matcher.py / audit_medicines.py / audit_
# context.py this session). Two different failure shapes, both bounded and
# verified rather than guessed:
#   (1) a plain coincidental T109/T121 (Organic Chemical / Pharmacologic
#       Substance) collision with an everyday word that means something
#       else entirely in a prim57 note (alcohol history, distress, a
#       symptom, blood glucose, housing, a body organ, a throat exam...).
#   (2) a generic category/descriptor word ("medication", "drugs",
#       "prophylactic") that's genuinely medicine-adjacent but never names
#       an actual medicine on its own, so keeping it would just be noise --
#       same reasoning as excluding it applies to "malaria"/"influenza",
#       which in this dataset are always the DISEASE being differentiated
#       ("r/o malaria"), never a vaccine being given.
MEDICINE_DENYLIST = {
    "etoh", "stress", "sore throat", "sugars", "sugar",
    "active", "alcohol", "balance", "bite", "chocolate", "cotton",
    "drug", "drugs", "duration", "food", "glass", "halls", "influenza",
    "link", "liver", "malaria", "medication", "medications", "meds",
    "pace", "pharmaceutical", "posture", "prophylactic", "relief",
    "support", "throat", "today", "urban", "water",
}

# A line that's ONLY a section header (e.g. a bare "Imp:" with the actual
# content following on the next line) carries no content of its own to
# extract -- matched separately from SECTION_PATTERNS so switching section on
# it doesn't also emit an empty inference/negation item for the header text
# itself.
HEADER_ONLY_RE = re.compile(
    r"^\s*#{0,6}\s*(plan|rx|management|mx|imp|impression|dx|diagnosis|assessment|"
    r"o/e|exam\w*|obs|vitals?|bloods?|ix|investigations?|pmh\w*|dh\w*|sh\w*|fh\w*|"
    r"psh\w*|ice|hx|hpc|objective|subjective)\s*:?\s*$",
    re.IGNORECASE,
)

# Negation cues seen across the prim57 note style -- terse clinical
# shorthand ("No SOB", "NKDA", "Nil PMHx", "Non smoker"), not the fuller
# conversational negation medspaCy's ConText is tuned for elsewhere in this
# project. A plain cue-word regex is the better fit for THIS text register:
# short, bulleted, non-conversational.
NEGATION_CUE_RE = re.compile(
    r"\b(no|nil|nkda|nad|denies|denied|not|non[\s-]|never|negative(?:\s+for)?|"
    r"absent|unremarkable|no known)\b",
    re.IGNORECASE,
)

# Inference/impression cues -- hedging or differential-diagnosis language
# ("likely", "?migraine", "r/o MS", "suggestive of") that marks a statement
# as the doctor's synthesized judgment rather than a plain recorded fact.
# The bare "?word" form (e.g. "?UTI", "??hypothyroid") is this dataset's
# most common differential shorthand, confirmed across the manually-read
# prim1-57 notes earlier in this project -- matched separately since it
# isn't a word-boundary token the main alternation can catch.
INFERENCE_CUE_RE = re.compile(
    r"\b(likely|probable|probably|possible|possibly|suggestive of|consistent with|"
    r"differential|ddx|r/o|rule out|working diagnosis|impression|query|presumed|"
    r"exclude|\?\?)\b",
    re.IGNORECASE,
)
INFERENCE_QUERY_RE = re.compile(r"\?[A-Za-z]")

CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;])\s+|\n+")


def _split_clauses(line):
    parts = [c.strip(" .;") for c in CLAUSE_SPLIT_RE.split(line)]
    return [c for c in parts if c]


# "number" is a fourth extraction category, added for the browser-extension
# checklist (oogle api/extension) -- a dose, frequency, duration, or vital
# is exactly the kind of verbatim fact an ambient-AI note generator can get
# subtly wrong (right drug, wrong dose) without the sentence reading as
# obviously off, so it's worth a clinician's dedicated second look the same
# way a drug name or a negation already gets one. Two shapes are matched
# separately since they don't share a common word-boundary form: a plain
# NUMBER + UNIT/FREQUENCY phrase ("2 tablets", "four times a day", "3-4
# days"), and a bare vital-sign shape with no unit word of its own
# ("120/80", "37.5°C").
_NUMBER_TOKEN = (
    r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|once|twice|thrice)"
)
_NUMBER_RANGE_SEP = r"(?:\s*(?:[-–]|to)\s*)"
_NUMBER_UNIT = (
    r"(?:mg|mcg|micrograms?|milligrams?|grams?|g|ml|units?|mmol\w*|tablets?|"
    r"caps?|capsules?|puffs?|drops?|times?(?:\s+(?:a|per)\s+day)?|days?|"
    r"hours?|hrs?|weeks?|months?|years?|bpm|%)"
)
NUMBER_PHRASE_RE = re.compile(
    rf"\b{_NUMBER_TOKEN}(?:{_NUMBER_RANGE_SEP}{_NUMBER_TOKEN})?(?:\s+|-)\s*{_NUMBER_UNIT}\b",
    re.IGNORECASE,
)
VITAL_SHAPE_RE = re.compile(
    r"\b\d{2,3}\s*/\s*\d{2,3}\b|\b\d{2,3}(?:\.\d+)?\s*°?\s*c\b", re.IGNORECASE
)


def _extract_numbers_from_clause(clause):
    """Every distinct dose/frequency/duration/vital phrase in clause,
    deduplicated case-insensitively within the clause."""
    phrases = []
    seen = set()
    for pattern in (VITAL_SHAPE_RE, NUMBER_PHRASE_RE):
        for m in pattern.finditer(clause):
            phrase = m.group(0).strip()
            key = phrase.lower()
            if key and key not in seen:
                seen.add(key)
                phrases.append(phrase)
    return phrases


_matcher = None


def _get_matcher(install_dir=None):
    global _matcher
    if _matcher is None:
        _matcher = get_matcher(install_dir)
    return _matcher


def _extract_medicines_from_line(line, section, install_dir=None):
    """Every UMLS drug-semtype match in line (DRUG_SEMTYPES, same scope as
    Modules/risk_taxonomy.py/Modules/high_risk_checker.py), deduplicated
    within the line by matched surface text."""
    items = []
    matcher = _get_matcher(install_dir)
    matches = matcher.match(line, best_match=True, ignore_syntax=False)
    seen = set()
    for group in matches:
        for m in group:
            if is_common_word(m["term"]) or is_common_word(m["ngram"]):
                continue
            if not (m.get("semtypes", set()) & DRUG_SEMTYPES):
                continue
            key = m["ngram"].strip().lower()
            if not key or key in seen or key in MEDICINE_DENYLIST:
                continue
            seen.add(key)
            items.append(("medicine", section, line, m["ngram"]))
    return items


def extract_note_content(note_text, install_dir=None):
    """Returns a list of (type, section, detail, highlighted) 4-tuples --
    type is one of "inference"/"negation"/"medicine", section is the SOAP
    section the line falls in (Modules/high_risk_checker.py's
    SECTION_PATTERNS), detail is the clause/line the item was found in, and
    highlighted is the specific matched drug term for "medicine" items (None
    for inference/negation, whose whole clause already is the tight scope)."""
    items = []
    current_section = "subjective"

    for raw_line in note_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        for section, pattern in SECTION_PATTERNS:
            if pattern.match(line):
                current_section = section
                break

        if HEADER_ONLY_RE.match(line):
            continue

        items.extend(_extract_medicines_from_line(line, current_section, install_dir))

        if current_section == "assessment":
            # The whole Assessment/Impression line IS the doctor's
            # synthesized judgment by definition -- no cue word needed.
            items.append(("inference", current_section, line, None))
        else:
            for clause in _split_clauses(line):
                if INFERENCE_CUE_RE.search(clause) or INFERENCE_QUERY_RE.search(clause):
                    items.append(("inference", current_section, clause, None))

        for clause in _split_clauses(line):
            if NEGATION_CUE_RE.search(clause):
                items.append(("negation", current_section, clause, None))

    return items


def extract_from_sections(sections, install_dir=None):
    """Same four-tuple output as extract_note_content, plus the "number"
    category, but for a caller that already knows each line's SOAP section
    from the UI itself (Modules/note_extractor.py's own docstring's target
    use case: oogle api/extension reads the Clinical Note tab's four
    separate Subjective/Objective/Assessment/Plan fields directly, so there
    is no header text to detect a section from -- SECTION_PATTERNS/
    HEADER_ONLY_RE are for a single blob of note text and don't apply here.

    sections: dict of {section_name: raw_text}, e.g.
        {"Subjective": "...", "Objective": "...", ...}."""
    items = []
    for section_name, text in sections.items():
        section = (section_name or "").strip().lower()
        if not text or not text.strip():
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip(" \t-*•")
            if not line:
                continue

            items.extend(_extract_medicines_from_line(line, section, install_dir))

            if section == "assessment":
                items.append(("inference", section, line, None))
            else:
                for clause in _split_clauses(line):
                    if INFERENCE_CUE_RE.search(clause) or INFERENCE_QUERY_RE.search(clause):
                        items.append(("inference", section, clause, None))

            for clause in _split_clauses(line):
                if NEGATION_CUE_RE.search(clause):
                    items.append(("negation", section, clause, None))
                for phrase in _extract_numbers_from_clause(clause):
                    items.append(("number", section, clause, phrase))

    return items
