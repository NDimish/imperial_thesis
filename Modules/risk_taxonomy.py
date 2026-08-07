"""Shared severity / detail-type taxonomy for the prim57 corruption generator
(datamakerfiles/prim_lib_injection.py) and the checker built to catch its
highest-risk error classes (Modules/high_risk_checker.py) -- kept in one
place so both sides grade severity by the identical rule instead of two
independently-drifting copies, and so their detail_type vocabularies overlap
where the underlying concept is genuinely the same (a "drug switch" injected
by the generator and a "drug switch" flagged by the checker mean the same
thing).

Severity rule, in one sentence: a drug- or allergy-related error is always
critical/high (wrong substance or dose is the shortest, most mechanistic
harm path); everything else scales with what the changed/affected text
actually touches, using the shared QuickUMLS matcher's semantic types --
not the corruption mechanism -- to judge that, since "how dangerous is this"
is a property of the clinical content, not of how the error was introduced.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from umls_matching import get_matcher, is_common_word  # noqa: E402 -- path insert must run first

SEVERITY_LEVELS = ("low", "moderate", "high", "critical")
SEVERITY_RANK = {"low": 1, "moderate": 2, "high": 3, "critical": 4}

# Corruption-mechanism types (generator) and clinical-content types (checker)
# share this one vocabulary where the concept is identical (drug switch,
# number edit, negation flip); the generator-only mechanism types (entity
# swap / inserted sentence / omitted detail) and checker-only content types
# (lasa confusion / diagnosis mismatch / allergy mismatch) don't have a
# natural counterpart on the other side -- see each module's own docstring.
DETAIL_TYPES = (
    "drug switch", "number edit", "negation flip",
    "entity swap", "inserted sentence", "omitted detail",
    "lasa confusion", "diagnosis mismatch", "allergy mismatch",
)

DRUG_SEMTYPES = {"T121", "T195", "T200"}
DISEASE_SEMTYPES = {"T047", "T191"}
ALLERGY_TRIGGER_RE = re.compile(r"\b(allerg\w*|nkda|anaphyla\w*)\b", re.IGNORECASE)
DOSAGE_UNIT_RE = re.compile(r"\b(mg|mcg|microgram\w*|milligram\w*|g|gram\w*|ml|units?|mmol)\b", re.IGNORECASE)

_matcher = None


def _get_matcher(install_dir=None):
    global _matcher
    if _matcher is None:
        _matcher = get_matcher(install_dir)
    return _matcher


def text_risk_tags(text, install_dir=None):
    """Returns the set of high-risk tags ({"DRUG", "DISEASE", "ALLERGY"} any
    subset) found anywhere in text -- DRUG/DISEASE via the shared QuickUMLS
    matcher's semantic types, ALLERGY via a plain trigger-word regex (an
    allergy is a relation, not a UMLS semantic type, same reasoning as
    Modules/high_risk_checker.py's own allergy handling)."""
    tags = set()
    if not text or not text.strip():
        return tags
    if ALLERGY_TRIGGER_RE.search(text):
        tags.add("ALLERGY")

    matcher = _get_matcher(install_dir)
    matches = matcher.match(text, best_match=True, ignore_syntax=False)
    for group in matches:
        for m in group:
            if is_common_word(m["term"]) or is_common_word(m["ngram"]):
                continue
            semtypes = m.get("semtypes", set())
            if semtypes & DRUG_SEMTYPES:
                tags.add("DRUG")
            if semtypes & DISEASE_SEMTYPES:
                tags.add("DISEASE")
    return tags


def classify_severity(detail_type, *texts, install_dir=None):
    """Deterministic severity assignment for one injected/detected error.

    detail_type: one of DETAIL_TYPES.
    texts: one or more strings to scan for high-risk content -- e.g. the
        swapped entity text, the inserted/omitted sentence, the sentence a
        negation/number was changed within.

    Returns one of SEVERITY_LEVELS.
    """
    combined = " ".join(t for t in texts if t)
    risk = text_risk_tags(combined, install_dir=install_dir)

    if detail_type in ("drug switch", "lasa confusion"):
        return "critical"
    if detail_type == "allergy mismatch":
        return "critical"
    if detail_type == "number edit":
        return "critical" if DOSAGE_UNIT_RE.search(combined) else "high"
    if detail_type == "negation flip":
        # Always critical, not just when it happens to touch a drug/allergy
        # in the same (sometimes merged) sentence -- a flipped assertion
        # changes whether a finding is present or absent at all, which is
        # the single fact most SOAP-note consumers rely on being right,
        # regardless of which concept it's attached to.
        return "critical"
    if detail_type == "diagnosis mismatch":
        return "critical" if "ALLERGY" in risk else "high"
    if detail_type == "omitted detail":
        if "DRUG" in risk or "ALLERGY" in risk:
            return "critical"
        if "DISEASE" in risk:
            return "high"
        return "low"
    if detail_type == "inserted sentence":
        if "DRUG" in risk or "ALLERGY" in risk:
            return "high"
        if "DISEASE" in risk:
            return "moderate"
        return "moderate"
    if detail_type == "entity swap":
        return "low"

    return "moderate"
