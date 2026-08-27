"""Edit-diff classifier for the browser-extension capture pipeline (oogle
api/extension + oogle api/server) -- given the AI's original SOAP-note draft
and the clinician's edited/saved version, judges each changed span as
clinically CRITICAL (changed what the note asserts) or not (tidied wording
without changing meaning).

Reuses Modules/risk_taxonomy.py's classify_severity as-is for the actual
severity judgment rather than building a second, independently-drifting
scale -- same reasoning Modules/note_extractor.py already applies by reusing
SECTION_PATTERNS/DRUG_SEMTYPES: "how dangerous is a drug switch / negation
flip / number edit" should be answered identically everywhere in this
project. high/critical severities bucket to critical=True here; low/
moderate bucket to critical=False.

Per changed span (difflib opcode over the note's lines -- SOAP notes are
mostly one bullet or one paragraph per line, so line-level diffing is the
natural, stable unit): detect what changed, in this priority order --
  1. a drug present in the old span missing from the new one -> "drug switch"
  2. a negation cue present in one span but not the other -> "negation flip"
  3. a dose/frequency/duration/vital phrase differs -> "number edit"
  4. the whole span is new (nothing removed) -> "inserted sentence"
  5. the whole span was deleted (nothing added) -> "omitted detail"
  6. otherwise -- both sides non-empty, no drug/negation/number difference
     detected -- "reworded", always non-critical: this is the "just to make
     the note cleaner" case, and unlike 1-5 above there's no existing
     risk_taxonomy detail_type for plain rewording to route through, so it's
     judged directly rather than forced through classify_severity's fallback.
"""
import difflib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from umls_matching import get_matcher, is_common_word  # noqa: E402

from Modules.note_extractor import (  # noqa: E402
    MEDICINE_DENYLIST,
    NEGATION_CUE_RE,
    NUMBER_PHRASE_RE,
    VITAL_SHAPE_RE,
)
from Modules.risk_taxonomy import DRUG_SEMTYPES, SEVERITY_RANK, classify_severity  # noqa: E402

CRITICAL_THRESHOLD = SEVERITY_RANK["high"]

_matcher = None


def _get_matcher(install_dir=None):
    global _matcher
    if _matcher is None:
        _matcher = get_matcher(install_dir)
    return _matcher


def _drug_terms(text, install_dir=None):
    """Lowercased set of distinct drug-semtype surface terms in text, same
    matching rule as Modules/note_extractor.py's own medicine extraction so
    the two modules never disagree on "is this a drug"."""
    if not text or not text.strip():
        return set()
    matcher = _get_matcher(install_dir)
    matches = matcher.match(text, best_match=True, ignore_syntax=False)
    terms = set()
    for group in matches:
        for m in group:
            if is_common_word(m["term"]) or is_common_word(m["ngram"]):
                continue
            if not (m.get("semtypes", set()) & DRUG_SEMTYPES):
                continue
            key = m["ngram"].strip().lower()
            if key and key not in MEDICINE_DENYLIST:
                terms.add(key)
    return terms


def _number_phrases(text):
    if not text:
        return set()
    phrases = set()
    for pattern in (VITAL_SHAPE_RE, NUMBER_PHRASE_RE):
        phrases.update(m.group(0).strip().lower() for m in pattern.finditer(text))
    return phrases


def _has_negation(text):
    return bool(text and NEGATION_CUE_RE.search(text))


def _classify_span(old_text, new_text, install_dir=None):
    """Returns (detail_type, severity, critical) for one changed span."""
    old_drugs = _drug_terms(old_text, install_dir)
    new_drugs = _drug_terms(new_text, install_dir)
    old_negated = _has_negation(old_text)
    new_negated = _has_negation(new_text)
    old_numbers = _number_phrases(old_text)
    new_numbers = _number_phrases(new_text)

    if old_drugs - new_drugs:
        detail_type = "drug switch"
    elif old_negated != new_negated:
        detail_type = "negation flip"
    elif old_numbers != new_numbers:
        detail_type = "number edit"
    elif not old_text.strip():
        detail_type = "inserted sentence"
    elif not new_text.strip():
        detail_type = "omitted detail"
    else:
        return "reworded", "low", False

    severity = classify_severity(detail_type, old_text, new_text, install_dir=install_dir)
    critical = SEVERITY_RANK[severity] >= CRITICAL_THRESHOLD
    return detail_type, severity, critical


def _align_block(old_block, new_block):
    """Pairs each line in a changed block (a difflib opcode span, which can
    hold several lines on each side) with its best-matching line on the
    other side by plain text similarity, instead of treating the whole
    block as one span.

    Without this, a bullet list where several lines change in the same
    edit (e.g. dose AND allergy status AND follow-up window, all in one
    contiguous run of changed lines) collapses into a single difflib
    'replace' opcode -- and _classify_span's priority order (drug > negation
    > number) then reports only the highest-priority category for the
    whole block, silently dropping the other real changes. Pairing by
    per-line similarity keeps each bullet's own before/after together so
    each gets classified on its own.

    Returns a list of (old_line_or_None, new_line_or_None) pairs. Falls
    back to leaving a line unpaired (an omission or an insertion) when
    nothing on the other side is even a weak textual match."""
    old_lines = [l for l in old_block if l.strip()]
    new_lines = [l for l in new_block if l.strip()]
    used_new = set()
    pairs = []
    for old_line in old_lines:
        best_j, best_ratio = None, 0.0
        for j, new_line in enumerate(new_lines):
            if j in used_new:
                continue
            ratio = difflib.SequenceMatcher(None, old_line, new_line, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio, best_j = ratio, j
        if best_j is not None and best_ratio >= 0.35:
            pairs.append((old_line, new_lines[best_j]))
            used_new.add(best_j)
        else:
            pairs.append((old_line, None))
    for j, new_line in enumerate(new_lines):
        if j not in used_new:
            pairs.append((None, new_line))
    return pairs


def diff_edit(note_before, note_after, install_dir=None):
    """Compares two full SOAP-note strings line by line and returns:
      {
        "similarity": float 0..1 (difflib ratio over the raw text),
        "edit_ratio": 1 - similarity,
        "critical": bool (any changed span judged critical),
        "changes": [
          {"before": str, "after": str, "detail_type": str,
           "severity": str, "critical": bool},
          ...
        ],
      }
    Only changed lines (difflib 'replace'/'delete'/'insert' opcodes, each
    re-aligned per-line via _align_block) are judged -- unchanged lines
    carry no signal either way."""
    before_lines = (note_before or "").splitlines()
    after_lines = (note_after or "").splitlines()

    sm = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    changes = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        for old_line, new_line in _align_block(before_lines[i1:i2], after_lines[j1:j2]):
            old_text = old_line or ""
            new_text = new_line or ""
            if not old_text.strip() and not new_text.strip():
                continue
            detail_type, severity, critical = _classify_span(old_text, new_text, install_dir)
            changes.append(
                {
                    "before": old_text,
                    "after": new_text,
                    "detail_type": detail_type,
                    "severity": severity,
                    "critical": critical,
                }
            )

    text_sm = difflib.SequenceMatcher(None, note_before or "", note_after or "", autojunk=False)
    similarity = text_sm.ratio()
    return {
        "similarity": similarity,
        "edit_ratio": 1 - similarity,
        "critical": any(c["critical"] for c in changes),
        "changes": changes,
    }
