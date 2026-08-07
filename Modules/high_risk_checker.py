"""Precision-first checker: only flags drug / diagnosis / allergy errors --
the categories a real audit of every other checker in this project (see
extra/explainer/checker_audit_report.html) found actually carry patient-
safety risk, as opposed to the generic body-part/clinical-attribute/
discourse-adjacent vocabulary that dominated every other checker's false
positives.

Deliberately narrower than Modules/medspacy_umls_checker.py, and built to
only ever flag something it's reasonably certain about -- it will miss real
errors outside its scope (that's the intended trade named in conversation:
"don't want too many FP", "will skip stuff"), not attempt full recall.

Four judgments, each independently scoped to drugs/diagnoses/allergies only:
  - omission / hallucination: plain CUI presence, same as
    Modules/metamap_cui_checker.py's proven-highest-recall approach, but
    restricted to DRUG_SEMTYPES/DISEASE_SEMTYPES + a regex allergy trigger
    instead of the full clinical vocabulary.
  - status_flip: ConText negation/uncertainty/family disagreement between a
    transcript mention and a SOAP mention of the same concept -- but ONLY
    fired when each side has exactly one mention with no ambiguity (see
    _judge_status_flips), and only after a same-line contrastive-conjunction
    guard (see _extract_line_concepts) that specifically targets the
    negation-scope-bleed bug found this session ("not really... but I feel
    warm" wrongly negating "warm").
  - number_edit: dosage/frequency/duration mismatch, bound to whatever the
    number is actually describing on its own line -- a drug, a disease/
    finding, or a family-relation marker ("mother", "FH:") -- not bare
    presence-checking, which Modules/old/deterministic_checker.py's own
    real run (0.000 precision, 0 TPs across 10 files) already proved isn't
    enough on its own (see _judge_numeric_frames / _find_numeric_anchor).
    Started drug-only; widened after a direct audit found real number
    edits this dataset injects -- a follow-up duration, a symptom
    duration, a family-history detail -- have nothing to do with a drug at
    all, so a drug-only frame structurally couldn't ever catch them.
  - lasa_confusion: a static, bounded look-alike/sound-alike drug-name
    pairlist (a representative subset of the ISMP list, not the full
    official one) -- deterministic and false-positive-bounded by
    construction, unlike an open-ended text search.

Every flag carries (error_type, severity, detail_type, detail) -- a 4-tuple,
not the 2-tuple every other checker in this project returns -- so severity/
detail_type can be scored the same way Modules/evaluate.py scores type
(see that module's by_severity/by_detail_type additions). Severity is
assigned by Modules/risk_taxonomy.py, the same rule the dataset generator
(datamakerfiles/prim_lib_injection.py) uses to grade its own injected
errors, so both sides grade "how bad is this" identically.
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
from umls_matching import get_matcher, is_common_word  # noqa: E402

from Modules.base import CheckerModule
from Modules.risk_taxonomy import DISEASE_SEMTYPES, DRUG_SEMTYPES, classify_severity

# Global on/off switch for omission judgments -- both _judge_omissions (plain
# CUI presence) and the omission direction of _judge_allergies (a documented
# transcript allergy missing from the SOAP note). Set False: omissions were
# the single largest false-positive source by far (146 of 203 FP in the last
# 10-file run, vs 57 for hallucination) -- almost entirely driven by two
# structural causes documented in this session (the adjacency mechanism
# reporting a bare reply clause like "yeah" as its evidence sentence instead
# of the concept's real context, and FINDING_SEMTYPES catching granular
# spoken symptom fragments a terse SOAP note paraphrases away rather than
# repeats verbatim) -- neither fixable by a quick toggle, so turning
# omissions off entirely is the direct way to stop that noise rather than
# filtering it label-by-label. hallucination/status_flip/allergy-
# hallucination/number-edit/lasa judgments are unaffected.
ENABLE_OMISSIONS = False

# Copied from Modules/medspacy_umls_checker.py's JUNK_CONCEPT_DENYLIST (itself
# validated in Modules/old/concept_checker.py's real-file smoke tests) -- kept
# local rather than imported, same reasoning as Modules/metamap_cui_checker.py
# and Modules/cql_checker.py's own copies (keeps this checker's dependency
# chain independent). Confirmed directly this run: without this, "times"
# alone (T047, Disease or Syndrome) matched "six or seven times a day" and
# produced a spurious drug/disease omission -- the same coincidental-
# collision class every UMLS-based checker in this project has already had
# to filter, resurfacing here because this module built its own extraction
# from scratch instead of reusing an existing one.
JUNK_CONCEPT_DENYLIST = {
    "hand", "controll", "control", "other things", "move", "mind", "said",
    "close", "test", "able", "times", "life", "difficult", "little",
    "stage", "recap", "keen", "remember", "feels", "much", "normal",
    "nice", "listened", "sampled", "quite often", "examined", "therex",
    "etests", "couplet", "coinfection",
    "plan", "patient", "diagnosed", "history", "complaints",
    # Generic drug-CLASS words, not coincidental collisions like the rest of
    # this list -- added after a direct test found the doctor-question
    # adjacency path picking up "medications" (from "any allergies to
    # medications?") as its own pending DRUG concept, spuriously flagged
    # omitted since the specific drug named in the SOAP note is a different
    # CUI. Same specificity-mismatch class already found this session
    # ("antibiotic" vs "nitrofurantoin", "the pill" vs a named contraceptive)
    # -- too generic to usefully anchor a comparison either way.
    "medication", "medications", "antibiotic", "antibiotics",
}


def _is_junk_concept(term):
    return term.strip().lower() in JUNK_CONCEPT_DENYLIST


# A negation trigger's ConText scope can bleed across a clause boundary
# within the same line/sentence -- confirmed directly this session on real
# files: "Uh, so, not, really I, I feel a little bit warm..." (answering
# "any OTHER symptoms?" negatively) wrongly negated "warm", the affirmed
# clause AFTER the contrastive conjunction. Splitting the line at these
# conjunctions before matching means a trigger before the split can no
# longer apply to a concept after it.
#
# Widened from contrastive-conjunctions-only to ALSO split on sentence-
# ending punctuation within a line -- confirmed directly this session that
# scoping to the whole LINE (not just across "but"/"however") was its own
# real bug for numeric-anchor detection specifically: a SOAP note line like
# "No blood in stool. Opening bowels x60/day." has no newline between its
# two sentences, so the frequency number in the second sentence was
# anchoring to "blood in stool" -- the nearest concept in the merged block
# -- rather than correctly finding no anchor of its own. Phrase-level
# scoping is also what the checker's own reported detail text is built
# from now: a short phrase ("no blood in stool") instead of the whole
# merged line, even though the ground-truth LABEL still keeps the full
# original sentence (see datamakerfiles/prim_lib_injection.py -- unchanged).
PHRASE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\b(?:but|however|although|yet)\b", re.IGNORECASE)


def _split_phrases(text):
    return [p.strip() for p in PHRASE_SPLIT_RE.split(text) if p and p.strip()]

# SOAP-section tagging: a deterministic line-header scan, not a general-
# purpose NLP sectionizer -- confirmed directly against this dataset's own
# notes (prim57/bad notes lib/prim1.txt, prim2.txt, prim20.txt) that they
# use UK-GP free-text convention (PMH:/DHx:/SH:/FH:/Imp:/Plan:, no explicit
# "Subjective:"/"Assessment:" headers, and no Objective section at all in
# most of these -- they're telephone consults with no physical exam
# documented), so a note-specific header regex is both simpler and more
# accurate here than medspaCy's general sectionizer, which is tuned for US
# hospital-note headers this dataset doesn't use. A line before the first
# recognized header defaults to "subjective" since these notes always open
# with free-text presenting complaint / history before any header appears.
SECTION_PATTERNS = [
    ("plan", re.compile(r"^\s*(plan|rx|management|mx)\s*:", re.IGNORECASE)),
    ("assessment", re.compile(r"^\s*(imp|impression|dx|diagnosis|assessment)\s*:", re.IGNORECASE)),
    ("objective", re.compile(r"^\s*(o/e|exam\w*|obs|vitals?|bloods?|ix|investigations?)\s*:", re.IGNORECASE)),
    ("subjective", re.compile(r"^\s*(pmh\w*|dh\w*|sh\w*|fh\w*|psh\w*|ice|hx|hpc)\s*:", re.IGNORECASE)),
]


def _line_sections(text):
    """Returns [(start_offset, end_offset, section), ...] covering every
    line of text, so a raw snippet's section can be found by locating its
    offset in text and looking up which line range it falls in."""
    sections = []
    current = "subjective"
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            for section, pattern in SECTION_PATTERNS:
                if pattern.match(stripped):
                    current = section
                    break
        sections.append((pos, pos + len(line), current))
        pos += len(line)
    return sections


def _section_for_snippet(full_text, snippet, line_sections):
    """Locates snippet (expected to be exact/near-exact text copied from
    full_text -- a raw sentence, phrase, or term) and returns the SOAP
    section it falls in, or None if it can't be found (e.g. a transcript-
    sourced omission, which has no SOAP-note position at all). Case-
    insensitive since a couple of callers (e.g. _judge_lasa) only have a
    lowercased drug name to search with, not the original-case text."""
    if not snippet or not line_sections:
        return None
    full_lower = full_text.lower()
    idx = full_lower.find(snippet.lower())
    if idx == -1:
        words = snippet.split()
        if len(words) > 6:
            idx = full_lower.find(" ".join(words[:6]).lower())
    if idx == -1:
        return None
    for start, end, section in line_sections:
        if start <= idx < end:
            return section
    return None


# A handful of common conversational symptom phrasings this dataset uses
# have no UMLS synonym string at all in this project's install -- confirmed
# directly: "blood in vomit" matches NOTHING (checked with both
# ignore_syntax=False and True), even though the standard clinical synonym
# "vomiting blood" matches C0018926 (haematemesis) immediately, and "blood
# in stool"/"blood in urine" both DO already match directly (UMLS happens
# to carry those exact strings as synonyms, just not this one). Without
# this override, "blood in vomit." collapses onto the generic "vomiting"
# CUI instead of getting its own concept -- confirmed this is exactly why
# a real negation-flip label was unfixable by the status_flip last-mention
# logic: the SOAP note's specific "blood in vomit" mention and several
# unrelated "patient vomited" mentions all shared one CUI, so the targeted
# fact's own assertion state was invisible, drowned out by the concept's
# other, unrelated mentions. A small, bounded override table (same shape as
# LASA_PAIRS below), not a general synonym engine -- only phrases directly
# confirmed missing from this project's own UMLS install.
SPECIAL_COMPOUND_FINDINGS = [
    (re.compile(r"\bblood in (?:\w+\s+)?vomit\b", re.IGNORECASE), "C0018926", "vomiting blood", "FINDING"),
    (re.compile(r"\bblood in (?:\w+\s+)?(?:sputum|phlegm)\b", re.IGNORECASE), "C0019079", "haemoptysis", "FINDING"),
]


def _special_finding_candidates(text):
    """(start, end, cui, term, category) tuples for SPECIAL_COMPOUND_FINDINGS
    matches in text -- same shape as the matcher-derived candidates list
    each caller already builds, so it can just be concatenated in before
    span-overlap resolution (spacy.util.filter_spans prefers the longer
    compound span over the matcher's separate shorter "blood"/"vomit"
    spans automatically)."""
    candidates = []
    for pattern, cui, term, category in SPECIAL_COMPOUND_FINDINGS:
        for m in pattern.finditer(text):
            candidates.append((m.start(), m.end(), cui, term, category))
    return candidates


ALLERGY_TRIGGER_RE = re.compile(r"\b(allerg\w*|nkda|anaphyla\w*)\b", re.IGNORECASE)
NKDA_RE = re.compile(r"\bnkda\b|no known (drug )?allerg\w*", re.IGNORECASE)
ALLERGIC_TO_RE = re.compile(
    r"allerg(?:ic|y|ies)\s*(?:to)?\s*[:\-]?\s*([a-zA-Z][a-zA-Z /-]{2,40}?)(?=[.,;\n]|$)",
    re.IGNORECASE,
)

# Numeric attribute extraction, scoped down from Modules/old/deterministic_
# checker.py's fuller version to dosage/frequency/duration/age -- the
# attributes that matter once bound to an anchor (see _find_numeric_anchor).
# Vitals/BP intentionally dropped: this checker's scope is drugs/diagnoses/
# allergies/family-history, not general numeric fact-checking. AGE_PATTERN
# added specifically for family-history numbers ("mother diagnosed at 45",
# "father died of MI at 55") -- the anchor-widening this whole check exists
# for is pointless for that case without also recognizing "age" as an
# extractable attribute in the first place.
DOSAGE_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|micrograms?|milligrams?|g|grams?|ml|units?|mmol)\b", re.IGNORECASE)
DURATION_SHORTHAND_PATTERN = re.compile(r"\b(\d+)\s*/\s*(7|52|12)\b")  # n/7=days, n/52=weeks, n/12=months
DURATION_WORD_PATTERN = re.compile(r"\b(\d+)\s*(day|days|week|weeks|month|months)\b", re.IGNORECASE)
AGE_PATTERN = re.compile(r"\b(?:age\w*\s+(\d{1,3})|aged\s+(\d{1,3})|(\d{1,3})\s*(?:years?|yrs?)?[\s-]*old)\b", re.IGNORECASE)
FREQUENCY_SHORTHAND_PATTERN = re.compile(r"\b(od|bd|tds|qds)\b", re.IGNORECASE)
FREQUENCY_SHORTHAND_MAP = {"od": 1, "bd": 2, "tds": 3, "qds": 4}
# "N times a day"/"N x/day" -- normalized from spoken "twice a day" via
# _normalize_number_words below, so this one pattern catches both the
# transcript's spoken register and a note's written "2 x/day" register
# instead of needing two parallel patterns (same fix already validated for
# this in Modules/old/deterministic_checker.py -- found there that without
# it, "twice a day" (transcript) and "BD" (SOAP note) looked like a real
# frequency mismatch even when they mean the same thing).
FREQUENCY_COUNT_PATTERN = re.compile(r"\b(\d+)\s*(?:x|times)\s*(?:a|per|/)\s*day\b", re.IGNORECASE)
# "x6/day" -- the number AFTER the "x", not before it. Confirmed directly
# this session: this project's own real notes use this exact clinical
# shorthand ("Opening bowels x6/day"), which FREQUENCY_COUNT_PATTERN above
# never matches since it only handles the number-then-"x" order.
FREQUENCY_COUNT_PREFIX_PATTERN = re.compile(r"\bx\s*(\d+)\s*/\s*day\b", re.IGNORECASE)

# Widened beyond DRUG_SEMTYPES/DISEASE_SEMTYPES for numeric-anchor detection
# ONLY (see _find_numeric_anchor) -- the main omission/hallucination concept
# judgments stay on the narrower DRUG_SEMTYPES/DISEASE_SEMTYPES scope so
# precision there doesn't regress. A duration/frequency is very often
# attached to a symptom or finding ("weakness for 3 days"), not a hard
# disease code, so T033 (Finding)/T184 (Sign or Symptom) are included here
# even though they're deliberately excluded from the main concept scope.
FINDING_SEMTYPES = {"T033", "T184"}
# T023 (Body Part, Organ, or Organ Component) added after a direct audit of
# a real missed number-edit label ("Opening bowels x60/day.") found the
# anchor concept itself was the problem, not the number extraction: the
# matcher resolves "bowels" to C0021853, semtype T023 -- an ANATOMY
# concept, not a Finding/Sign-or-Symptom -- so a very common frequency
# phrasing ("opening bowels", "passing urine") had no anchor at all under
# the old scope. Anchor-only, same reasoning as FINDING_SEMTYPES's own
# widening above: this doesn't touch the main omission/hallucination
# concept scope, only which concepts a bare number is allowed to attach to.
NUMERIC_ANCHOR_SEMTYPES = DRUG_SEMTYPES | DISEASE_SEMTYPES | FINDING_SEMTYPES | {"T023"}

# A number in a family-history line ("mother diagnosed at 45", "FH: father -
# HTN, died of bowel Ca age 60") has no UMLS concept of its own to anchor to
# -- "family history" is a relation, not a semantic type (same reasoning as
# the allergy handling above) -- so it gets its own lightweight anchor here.
FAMILY_RELATION_RE = re.compile(
    r"\b(mother|father|brother|sister|parent\w*|sibling\w*|son|daughter|"
    r"grandmother|grandfather|family history|fh\s*:)\b",
    re.IGNORECASE,
)

# Representative subset of the ISMP Confused Drug Names list (not the full
# official list -- see module docstring) -- the same example pairs named in
# Modules/AI_checker.py's own LASA prompt, plus a few more well-known ones.
# Static and bounded on purpose: a false positive here can only come from
# stale list coverage, never from open-ended matching.
LASA_PAIRS = [
    ("hydralazine", "hydroxyzine"),
    ("celebrex", "celexa"),
    ("clonidine", "clonazepam"),
    ("prednisone", "prednisolone"),
    ("metformin", "metronidazole"),
    ("lamictal", "lamisil"),
    ("humalog", "humulin"),
    ("zantac", "zyrtec"),
    ("citalopram", "escitalopram"),
    ("hydrocodone", "hydrocortisone"),
    ("clomiphene", "clomipramine"),
    ("dopamine", "dobutamine"),
]


# Widened from one-five to one-twenty plus the tens words after a direct
# audit found a real transcript number this checker was silently blind to:
# "Six, seven times a day" (prim1.txt) -- "six"/"seven" weren't in the old
# map at all, so FREQUENCY_COUNT_PATTERN (which requires a DIGIT before
# "times") never even saw a number there, and the frequency was dropped
# before extraction could run, not just failed to match the SOAP side.
# Same gap would have hit a spoken dosage ("twenty milligrams") or age
# ("aged fifty") the exact same way -- this fixes all three at once since
# they all normalize through this one function first.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}


def _normalize_number_words(text):
    # "once"/"twice"/"thrice" expand to "N times" (not just "N") so
    # FREQUENCY_COUNT_PATTERN's required "x"/"times" keyword still matches --
    # confirmed in Modules/old/deterministic_checker.py that normalizing to a
    # bare digit without "times" silently drops the frequency instead of
    # extracting it. Must run BEFORE the general word map below since
    # "once"/"twice"/"thrice" aren't in it and need the "N times" form, not
    # a bare digit.
    for word, replacement in {"once": "1 times", "twice": "2 times", "thrice": "3 times"}.items():
        text = re.sub(rf"\b{word}\b", replacement, text, flags=re.IGNORECASE)
    for word, value in _NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", str(value), text, flags=re.IGNORECASE)
    return text


def extract_numeric_attributes(text):
    """Returns a set of (category, *values) tuples for dosage/duration/
    frequency found in text -- scoped-down port of Modules/old/
    deterministic_checker.py's extract_numeric_attributes."""
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

    for value, denom in DURATION_SHORTHAND_PATTERN.findall(normalized):
        days = int(value) * (1 if denom == "7" else 7 if denom == "52" else 30)
        attrs.add(("duration_days", days))

    for value, unit in DURATION_WORD_PATTERN.findall(normalized):
        unit_days = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}
        attrs.add(("duration_days", int(value) * unit_days[unit.lower()]))

    for shorthand in FREQUENCY_SHORTHAND_PATTERN.findall(normalized):
        attrs.add(("frequency_per_day", FREQUENCY_SHORTHAND_MAP[shorthand.lower()]))

    for value in FREQUENCY_COUNT_PATTERN.findall(normalized):
        attrs.add(("frequency_per_day", int(value)))

    for value in FREQUENCY_COUNT_PREFIX_PATTERN.findall(normalized):
        attrs.add(("frequency_per_day", int(value)))

    for match in AGE_PATTERN.finditer(normalized):
        value = next(g for g in match.groups() if g)
        attrs.add(("age", int(value)))

    return attrs


# Adjacency Pair Context Propagation: naive per-line extraction silently
# loses a real fact when a doctor's question carries the clinical concept
# but the patient's reply is a bare "Yes"/"No" that never repeats it --
# confirmed directly this session ("any allergies?"/"No." lost the allergy
# fact entirely, since neither line alone has both a concept AND an
# assertion). Instead of extracting each line in isolation, a doctor
# question ending in "?" holds its concept(s) pending and links them
# forward to the patient's immediate next turn: a clear affirmative/
# negative reply resolves the pending concept's polarity directly; an
# ambiguous or elaborate reply ("only when I eat dairy") falls back to
# running standard extraction on that reply instead of guessing. A run of
# several consecutive doctor questions followed by a run of patient replies
# is handled by splitting the reply text into clauses and pairing them
# positionally with the pending questions in order -- any question left
# over once clauses run out is dropped, not guessed at; any clause left
# over once questions run out still gets standard extraction.
AFFIRMATIVE_LEXICON = ["yes", "yeah", "yep", "i do", "i have", "a bit", "definitely", "correct", "a lot"]
# "not that i know of"/"not that i'm aware of"/"none" added after a direct
# audit found a real transcript reply -- "Um, not that I know of." to "Do
# you have any allergies to medications?" -- falling through as ambiguous
# under the original 7-phrase set, losing the allergy fact the same way a
# bare "No." used to before that fix. Widened again after a second direct
# audit (this time on _judge_allergies specifically) found "Um, so I don't
# have any allergies that I know of." ALSO fell through as ambiguous --
# "i don't have any" is an extremely common negative-reply shape this list
# didn't cover at all, so an allergy fact both sides genuinely agreed on
# was silently dropped from the transcript side, reading as a hallucinated
# NKDA. Every phrase here is still a strict LEADING-phrase match (see
# _classify_reply), never a substring search, so widening this list can't
# turn a genuinely ambiguous reply into a false negative/affirmative --
# it only recognizes MORE ways of clearly saying no.
NEGATIVE_LEXICON = [
    "no", "nope", "nah", "never", "none",
    "i haven't", "i've not", "not really", "don't think so", "doesn't think so",
    "not that i know of", "not that i'm aware of", "not that im aware of",
    "nothing that i know of", "nothing i know of",
    "not to my knowledge", "not aware of any", "not aware of",
    "i don't have any", "don't have any", "i do not have any", "do not have any",
    "i don't have", "don't have", "i do not have", "do not have",
    "i've never had any", "never had any", "i haven't had any", "haven't had any",
    "no known", "not known",
    "no i don't", "no i haven't", "no i do not", "no i've never",
    "none that i know of", "none that i'm aware of", "none that im aware of",
]
# "mm+"/"hmm+" added after a direct audit of a real allergy-mismatch FP
# (prim13.txt: "Mm not that I know of." to "Do you have any allergies to
# any medications?") found this exact hesitation sound wasn't covered --
# "Mm" isn't stripped by the old filler set, so "mm not that i know of"
# never starts with any NEGATIVE_LEXICON phrase (they all start with "not"),
# and the reply fell through as ambiguous, silently losing a clear negative
# the same way bare "No."/"Um, not that I know of." used to before their
# own fixes earlier this session.
FILLER_PREFIX_RE = re.compile(r"^[\s,.\-]*\b(um+|uh+|h?mm+|well|so|okay|ok)\b[\s,.\-]*", re.IGNORECASE)
CLAUSE_SPLIT_RE = re.compile(r"[.,;]\s+|\band\b", re.IGNORECASE)


FILLER_ONLY_RE = re.compile(r"^(um+|uh+|h?mm+|well|so|okay|ok)$", re.IGNORECASE)


def _split_reply_clauses(text):
    """Splits a reply into clauses, dropping any that are PURELY a filler
    word ("Um") with nothing else in them. Confirmed directly this session:
    without this, CLAUSE_SPLIT_RE's own comma boundary turns "Um, not that
    I know of." into two clauses (["Um", "not that I know of."]) -- and
    since callers pair clauses positionally with pending questions, the
    real content clause silently gets pushed out of alignment (or, for a
    single pending question, never even looked at -- only the leading "Um"
    clause is), rather than the filler being absorbed into the same clause
    as the content that follows it."""
    raw = [c.strip() for c in CLAUSE_SPLIT_RE.split(text) if c.strip()]
    return [c for c in raw if not FILLER_ONLY_RE.match(c)]


def _classify_reply(clause):
    """True=affirmative, False=negative, None=ambiguous/elaborate (caller
    falls back to standard extraction on this clause). Strict leading-
    phrase match against the lexicons above, not substring search -- "not
    sure" must NOT match "no" or any negative phrase, since it's genuinely
    ambiguous, not a clear negative."""
    # .rstrip(".!?") matters a lot in practice: confirmed directly this
    # session that a bare "No." -- an extremely common standalone patient
    # turn -- fell through as ambiguous without it, since CLAUSE_SPLIT_RE
    # only splits a trailing period when it's followed by whitespace, so a
    # period at the very end of a turn never gets separated from the word
    # before it.
    stripped = FILLER_PREFIX_RE.sub("", clause.strip()).strip().lower().rstrip(".!?").strip()
    for phrase in NEGATIVE_LEXICON:
        if stripped == phrase or stripped.startswith(phrase + " ") or stripped.startswith(phrase + ","):
            return False
    for phrase in AFFIRMATIVE_LEXICON:
        if stripped == phrase or stripped.startswith(phrase + " ") or stripped.startswith(phrase + ","):
            return True
    return None


def _describe_attr(attr):
    category, values = attr[0], attr[1:]
    if category == "dosage":
        return f"dosage {values[0]}{values[1]}"
    if category == "duration_days":
        return f"duration {values[0]} day(s)"
    if category == "frequency_per_day":
        return f"frequency {values[0]}x/day"
    if category == "age":
        return f"age {values[0]}"
    return f"{category} {values}"


class HighRiskChecker(CheckerModule):
    """See module docstring for the full scope/judgment design."""

    def __init__(self, install_dir=None):
        self._matcher = get_matcher(install_dir)
        self._nlp = medspacy.load()

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        transcript_concepts, transcript_links = self._extract_concepts_and_links(transcript)
        soap_concepts, soap_links = self._extract_concepts_and_links(soap_note)
        soap_line_sections = _line_sections(soap_note)

        errors = []
        if ENABLE_OMISSIONS:
            errors.extend(self._judge_omissions(transcript_concepts, soap_concepts))
        errors.extend(self._judge_hallucinations(transcript_concepts, soap_concepts, soap_note, soap_line_sections))
        errors.extend(self._judge_status_flips(transcript_concepts, soap_concepts, soap_note, soap_line_sections))
        errors.extend(self._judge_allergies(transcript, soap_note, soap_line_sections))
        errors.extend(self._judge_numeric_frames(transcript, soap_note, transcript_links, soap_links, soap_line_sections))
        errors.extend(self._judge_lasa(transcript, soap_note, soap_line_sections))

        # _judge_allergies also emits the "omission" direction (a
        # transcript allergy missing from the SOAP note) inline with its
        # hallucination direction -- filtered out here rather than
        # threading ENABLE_OMISSIONS into that method too, so there's one
        # place that decides what "omissions off" means for every producer.
        if not ENABLE_OMISSIONS:
            errors = [e for e in errors if e[0] != "omission"]

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    # ------------------------------------------------------------------
    # Extraction (drug/disease concepts only, line-scoped, guarded ConText,
    # Adjacency Pair Context Propagation across doctor Q / patient A turns)
    # ------------------------------------------------------------------

    def _extract_concepts_and_links(self, text):
        """Returns (concepts, numeric_links):
          concepts: {cui: {"term": str, "category": "DRUG"|"DISEASE"|
            "FINDING", "mentions": [(is_negated, is_uncertain, is_family,
            sentence_text), ...]}}
          numeric_links: {anchor_key: {"anchor_term", "sentence", "attrs"}}
            -- numbers found in a patient reply clause that's linked back to
            a doctor question's concept, merged into _judge_numeric_frames's
            per-line frames (see that method) so a dose/duration stated in
            answer to a question naming the drug/finding is still bound to
            it even though the reply itself never repeats the concept.

        See the Adjacency Pair Context Propagation comment above
        AFFIRMATIVE_LEXICON for the full mechanism this implements.
        """
        concepts = {}
        numeric_links = {}
        pending = []  # [(cui, term, category), ...] from unanswered doctor questions

        for speaker, line_text in split_turns(text):
            if not line_text.strip():
                continue
            speaker_norm = (speaker or "").lower()
            is_doctor = speaker_norm == "d"
            is_patient = speaker_norm == "p"

            if is_doctor and line_text.strip().endswith("?"):
                pending.extend(self._extract_question_concepts(line_text))
                continue

            if is_patient and pending:
                clauses = _split_reply_clauses(line_text)
                n = min(len(pending), len(clauses))
                for (cui, term, category), clause in zip(pending[:n], clauses[:n]):
                    reply = _classify_reply(clause)

                    if reply is False:
                        entry = concepts.setdefault(cui, {"term": term, "category": category, "mentions": []})
                        entry["mentions"].append((True, False, False, clause))
                        continue

                    # True (clear "yes") and None (ambiguous/elaborate, e.g.
                    # "500mg twice a day") are both treated as the patient
                    # having engaged with and confirmed the doctor's
                    # question concept -- confirmed directly this session
                    # that skipping this for the ambiguous case is a real
                    # bug, not just caution: "How much metformin do you
                    # take?"/"500mg twice a day" was leaving "metformin"
                    # unconfirmed entirely (its name isn't repeated in the
                    # reply, so the fallback extraction alone never finds
                    # it), making it read as hallucinated regardless of
                    # whether the dose matched or not. A specific,
                    # substantive answer is itself evidence of affirmation
                    # the same way a bare "yes" is; a patient who doesn't
                    # take the drug would decline instead (a negative
                    # reply, handled above), not elaborate on a dose for it.
                    entry = concepts.setdefault(cui, {"term": term, "category": category, "mentions": []})
                    entry["mentions"].append((False, False, False, clause))

                    if reply is None:
                        # Also run standard extraction on the same clause,
                        # in case it independently asserts something
                        # ConText can read more precisely than the
                        # affirmed-by-default inference above (e.g. a
                        # negation this simple lexicon didn't cover).
                        self._extract_line_concepts(clause, concepts)

                    clause_attrs = extract_numeric_attributes(clause)
                    if clause_attrs:
                        anchor_key = f"CONCEPT:{cui}"
                        frame = numeric_links.setdefault(
                            anchor_key, {"anchor_term": term, "sentence": clause, "attrs": set()}
                        )
                        frame["attrs"] |= clause_attrs

                # More replies than pending questions -- the extra clauses
                # are volunteered info, not answers to a specific question.
                for clause in clauses[n:]:
                    self._extract_line_concepts(clause, concepts)

                pending = []
                continue

            if is_doctor:
                # A non-question doctor statement breaks the adjacency
                # chain -- whatever comes next isn't necessarily answering
                # an earlier unanswered question anymore.
                pending = []

            self._extract_line_concepts(line_text, concepts)

        return concepts, numeric_links

    def _extract_question_concepts(self, line_text):
        """Candidate concepts from a doctor's question turn, held pending
        for linking to the patient's next reply. Uses
        NUMERIC_ANCHOR_SEMTYPES (drug/disease/finding) -- wider than the
        main per-line concept scope below, safe here specifically because
        it's gated by the question+immediate-reply structure rather than a
        blind scan.

        Runs the SAME span-overlap resolution (spacy.util.filter_spans) as
        _extract_clause_concepts -- confirmed directly this session that
        skipping it is a real bug, not just simpler code: a word matching
        MULTIPLE overlapping UMLS CUIs at the same span (e.g. "rash"/
        "rashes" resolves to two distinct CUIs here) let this method return
        both as separate pending concepts while _extract_clause_concepts
        (which DOES resolve the overlap) keeps only one -- so the two sides
        of a later comparison could end up keyed on different CUIs for the
        exact same real-world word, always reading as a mismatch."""
        matches = self._matcher.match(line_text, best_match=True, ignore_syntax=False)
        candidates = []
        for group in matches:
            for m in group:
                if is_common_word(m["term"]) or is_common_word(m["ngram"]):
                    continue
                if _is_junk_concept(m["term"]) or _is_junk_concept(m["ngram"]):
                    continue
                semtypes = m.get("semtypes", set())
                if semtypes & DRUG_SEMTYPES:
                    candidates.append((m["start"], m["end"], m["cui"], m["term"], "DRUG"))
                elif semtypes & DISEASE_SEMTYPES:
                    candidates.append((m["start"], m["end"], m["cui"], m["term"], "DISEASE"))
                elif semtypes & FINDING_SEMTYPES:
                    candidates.append((m["start"], m["end"], m["cui"], m["term"], "FINDING"))
        candidates.extend(_special_finding_candidates(line_text))
        if not candidates:
            return []

        doc = self._nlp(line_text)
        span_to_match = {}
        candidate_ents = []
        for start, end, cui, term, category in candidates:
            span = doc.char_span(start, end, label="CONCEPT", alignment_mode="expand")
            if span is None:
                continue
            candidate_ents.append(span)
            span_to_match[(span.start_char, span.end_char)] = (cui, term, category)
        if not candidate_ents:
            return []

        resolved = spacy.util.filter_spans(candidate_ents)
        return [span_to_match[(span.start_char, span.end_char)] for span in resolved]

    def _extract_line_concepts(self, text, concepts):
        if not text.strip():
            return

        # Phrase-level guard: process each phrase of the line separately (see
        # PHRASE_SPLIT_RE's comment) -- both so a negation trigger before
        # "but"/"however"/"although"/"yet" can't reach a concept after it,
        # and so a merged multi-sentence line doesn't let one sentence's
        # concept get borrowed as context for an unrelated adjacent one.
        for phrase in _split_phrases(text):
            self._extract_clause_concepts(phrase, concepts)

    def _extract_clause_concepts(self, text, concepts):
        if not text.strip():
            return

        matches = self._matcher.match(text, best_match=True, ignore_syntax=False)
        candidates = []
        for group in matches:
            for m in group:
                if is_common_word(m["term"]) or is_common_word(m["ngram"]):
                    continue
                if _is_junk_concept(m["term"]) or _is_junk_concept(m["ngram"]):
                    continue
                semtypes = m.get("semtypes", set())
                if semtypes & DRUG_SEMTYPES:
                    candidates.append((m["start"], m["end"], m["cui"], m["term"], "DRUG"))
                elif semtypes & DISEASE_SEMTYPES:
                    candidates.append((m["start"], m["end"], m["cui"], m["term"], "DISEASE"))
                elif semtypes & FINDING_SEMTYPES:
                    # Widened from DRUG/DISEASE-only after a direct test found
                    # the two sides of a comparison could never agree on a
                    # Finding-type concept (fever/rash/haematuria/dysphagia)
                    # otherwise: _extract_question_concepts (doctor-question
                    # adjacency path) already looked at FINDING_SEMTYPES, but
                    # this method -- used for EVERY SOAP note line, since SOAP
                    # notes have no doctor/patient turns to gate on -- didn't,
                    # so a Finding concept recovered via adjacency on the
                    # transcript side had structurally nothing to match
                    # against on the SOAP side, always reading as omitted
                    # regardless of whether the note actually agreed.
                    candidates.append((m["start"], m["end"], m["cui"], m["term"], "FINDING"))
        candidates.extend(_special_finding_candidates(text))
        if not candidates:
            return

        doc = self._nlp(text)
        span_to_match = {}
        candidate_ents = []
        for start, end, cui, term, category in candidates:
            span = doc.char_span(start, end, label="CONCEPT", alignment_mode="expand")
            if span is None:
                continue
            candidate_ents.append(span)
            span_to_match[(span.start_char, span.end_char)] = (cui, term, category)
        if not candidate_ents:
            return

        doc.ents = spacy.util.filter_spans(candidate_ents)
        self._nlp.get_pipe("medspacy_context")(doc)

        for ent in doc.ents:
            cui, term, category = span_to_match.get((ent.start_char, ent.end_char), (None, None, None))
            if cui is None:
                continue
            sent_text = ent.sent.text.strip() if ent.sent is not None else ent.text
            if sent_text.endswith("?"):
                continue
            if ent._.is_hypothetical:
                continue

            state = (bool(ent._.is_negated), bool(ent._.is_uncertain), bool(ent._.is_family), sent_text)
            entry = concepts.setdefault(cui, {"term": term, "category": category, "mentions": []})
            entry["mentions"].append(state)

    # ------------------------------------------------------------------
    # Judgments
    # ------------------------------------------------------------------

    def _judge_omissions(self, transcript_concepts, soap_concepts):
        """Omissions describe content missing FROM the SOAP note -- sourced
        from the transcript side, which has no SOAP-section structure, so
        section is always None here (see _section_for_snippet's docstring)."""
        errors = []
        for cui, info in transcript_concepts.items():
            if cui in soap_concepts:
                continue
            errors.append(self._make_error("omission", info))
        return errors

    def _judge_hallucinations(self, transcript_concepts, soap_concepts, soap_note, soap_line_sections):
        errors = []
        for cui, info in soap_concepts.items():
            if cui in transcript_concepts:
                continue
            errors.append(self._make_error("hallucination", info, soap_note, soap_line_sections))
        return errors

    def _judge_status_flips(self, transcript_concepts, soap_concepts, soap_note, soap_line_sections):
        """Compares the LAST mention of a concept on each side, not a
        single-mention-only match -- confirmed directly this session that
        requiring exactly one mention on both sides was the reason recall
        was 0: real conversation naturally repeats/restates a symptom many
        times across a consultation (the same concept can easily rack up
        5+ mentions in a transcript) while a SOAP note states it once, so
        the old guard almost never fired at all. The LAST mention is what a
        conversation actually settles on ("I had vomiting but I've stopped
        now" -- the final statement is the one that matters), the same way
        a SOAP note's single mention is its final word on the topic.

        This will also flag a hedge-language mismatch (transcript says
        "probably X", SOAP note's Impression states X unhedged) as a status
        flip via is_uncertain -- an accepted, known false positive (per
        direct instruction), not filtered out here, since a written
        Impression is conventionally unhedged regardless of how it was
        spoken and there's no reliable way to tell that apart from a real
        uncertainty disagreement without discarding real signal too.

        Reports type "hallucination", not "status_flip" -- confirmed
        directly this session that this was the actual reason status_flip
        scored 0 TP / 0 FN in every prior run, independent of the mention-
        count logic above: datamakerfiles/prim_lib_injection.py's
        _flip_negative ALWAYS labels a negation flip {"type":
        "hallucination", ...} (a flipped negation asserts a new false fact,
        which IS a hallucination by this project's own schema -- there is
        no "status_flip" type anywhere in the ground truth), and
        Modules/evaluate.py's compare() only matches a prediction against a
        label of the SAME type. A "status_flip"-typed prediction could
        therefore never match ANY label, no matter how accurate the
        judgment underneath it was. detail_type "negation flip" (see
        by_detail_type in Modules/evaluate.py's results) is what actually
        distinguishes this judgment from a plain concept-presence
        hallucination now, the same way the label schema distinguishes it.
        """
        errors = []
        for cui, soap_info in soap_concepts.items():
            transcript_info = transcript_concepts.get(cui)
            if transcript_info is None:
                continue
            t_state = transcript_info["mentions"][-1][:3]
            s_state = soap_info["mentions"][-1][:3]
            if t_state == s_state:
                continue

            detail_type = "negation flip"
            sentence = soap_info["mentions"][-1][3]
            severity = classify_severity(detail_type, sentence, soap_info["term"])
            detail = f"{soap_info['term']}: {sentence}"
            section = _section_for_snippet(soap_note, sentence, soap_line_sections)
            errors.append(("hallucination", severity, detail_type, detail, section))
        return errors

    def _make_error(self, error_type, info, soap_note=None, soap_line_sections=None):
        sentence = info["mentions"][0][3]
        detail_type = "drug switch" if info["category"] == "DRUG" else "diagnosis mismatch"
        severity = classify_severity(detail_type, sentence, info["term"])
        section = _section_for_snippet(soap_note, sentence, soap_line_sections) if soap_note is not None else None
        return (error_type, severity, detail_type, sentence, section)

    def _judge_allergies(self, transcript, soap_note, soap_line_sections):
        """Allergy status is a relation ("allergic to X" / "NKDA"), not a
        UMLS concept -- extracted directly via regex per line rather than
        through the concept pipeline above. The most severe case (a
        documented allergy silently dropped, or NKDA overwritten with a
        specific allergy) is graded critical regardless of which direction
        it goes.

        The hallucination direction (SOAP note states an allergy fact the
        transcript doesn't) is only checked when the transcript raises the
        topic of allergies AT ALL (t_allergies non-empty) -- confirmed
        directly this session that the dominant real-world cause of these
        FPs isn't a fabricated fact, it's a call where allergy status was
        never discussed (this specific consultation, at least) and NKDA was
        carried into the note from an existing patient record instead --
        exactly the same "not discussed on this call" pattern real medical
        scribes rely on chart context for, which this checker has no
        access to and can't tell apart from a genuine hallucination without
        this guard. If the transcript has NO allergy signal at all, there's
        nothing to compare the note's allergy line against, so it's no
        longer flagged either way."""
        t_allergies = self._extract_allergy_facts(transcript)
        s_allergies = self._extract_allergy_facts(soap_note)
        errors = []

        for substance, sentence in t_allergies.items():
            if substance not in s_allergies:
                detail_type = "allergy mismatch"
                severity = classify_severity(detail_type, sentence)
                errors.append(("omission", severity, detail_type, sentence, None))

        if t_allergies:
            for substance, sentence in s_allergies.items():
                if substance not in t_allergies:
                    detail_type = "allergy mismatch"
                    severity = classify_severity(detail_type, sentence)
                    section = _section_for_snippet(soap_note, sentence, soap_line_sections)
                    errors.append(("hallucination", severity, detail_type, sentence, section))

        return errors

    @staticmethod
    def _extract_allergy_facts(text):
        """{substance_or_"none": sentence}, one entry per line with an
        allergy trigger -- "none" for an NKDA-style line, the matched
        substance name (lowercased) for an "allergic to X" line.

        Also runs Adjacency Pair Context Propagation for the doctor-asks/
        patient-answers case, same lexicon/mechanism as
        _extract_concepts_and_links -- confirmed directly this session that
        without it, "Any allergies at all?" / "No." loses the allergy fact
        entirely, since the patient's bare "No." never repeats "allerg"/
        "nkda" for ALLERGY_TRIGGER_RE to catch. A clear negative reply
        resolves to "none"; an affirmative reply is skipped rather than
        guessed at, since a generic "any allergies?" question doesn't name
        a specific substance to attribute the answer to.
        """
        facts = {}
        pending_allergy_question = False

        for speaker, line_text in split_turns(text):
            if not line_text.strip():
                continue
            speaker_norm = (speaker or "").lower()
            is_doctor = speaker_norm == "d"
            is_patient = speaker_norm == "p"

            if is_doctor and line_text.strip().endswith("?") and ALLERGY_TRIGGER_RE.search(line_text):
                pending_allergy_question = True
                continue

            if is_patient and pending_allergy_question:
                pending_allergy_question = False
                clause = _split_reply_clauses(line_text)
                reply = _classify_reply(clause[0]) if clause else None
                if reply is False:
                    facts.setdefault("none", line_text.strip())
                    continue
                if reply is True:
                    continue  # confirmed SOME allergy exists, but not which -- don't guess
                # Ambiguous/elaborate reply -- fall through to standard
                # trigger-word scanning below on this same line.

            if is_doctor:
                pending_allergy_question = False

            if not ALLERGY_TRIGGER_RE.search(line_text):
                continue
            if NKDA_RE.search(line_text):
                facts.setdefault("none", line_text.strip())
                continue
            m = ALLERGIC_TO_RE.search(line_text)
            if m:
                substance = m.group(1).strip().lower()
                facts.setdefault(substance, line_text.strip())

        return facts

    def _judge_numeric_frames(self, transcript, soap_note, transcript_links, soap_links, soap_line_sections):
        """Frame-bound numeric check, generalized beyond drug dosages: for
        every LINE with a number in it, find what that number is actually
        describing -- a DRUG/DISEASE/FINDING concept, or a family-relation
        marker ("mother", "FH:") -- and compare against the same anchor's
        numeric attributes on the other side. Widening WHAT can anchor a
        number (not just a drug mention) catches real cases a drug-only
        version misses entirely -- a follow-up duration ("review in 10/52"),
        a symptom duration ("10 week history of..."), a family history
        detail ("mother diagnosed at 45") -- while still requiring SOME
        identifiable context per number, never bare "is this number anywhere
        in the other document" presence-checking, which Modules/old/
        deterministic_checker.py's real run already showed produces 0 true
        positives on this dataset (see module docstring). A line with a
        number but no recognizable anchor at all is skipped, not guessed at.

        transcript_links/soap_links: numbers recovered via Adjacency Pair
        Context Propagation (see _extract_concepts_and_links) -- a dose or
        duration stated in reply to a doctor's question naming the drug/
        finding, even though the reply itself never repeats the concept
        word ("How much metformin do you take?" / "500mg twice a day").
        Merged into the same per-line frames below before comparing.
        """
        transcript_frames = self._extract_numeric_frames(transcript)
        soap_frames = self._extract_numeric_frames(soap_note)

        for anchor_key, link_frame in transcript_links.items():
            frame = transcript_frames.setdefault(
                anchor_key, {"anchor_term": link_frame["anchor_term"], "sentence": link_frame["sentence"], "attrs": set()}
            )
            frame["attrs"] |= link_frame["attrs"]
        for anchor_key, link_frame in soap_links.items():
            frame = soap_frames.setdefault(
                anchor_key, {"anchor_term": link_frame["anchor_term"], "sentence": link_frame["sentence"], "attrs": set()}
            )
            frame["attrs"] |= link_frame["attrs"]

        errors = []
        for anchor_key, soap_frame in soap_frames.items():
            transcript_frame = transcript_frames.get(anchor_key)
            if transcript_frame is None:
                continue  # the concept itself is already covered by _judge_hallucinations
            missing = soap_frame["attrs"] - transcript_frame["attrs"]
            for attr in missing:
                sentence = soap_frame["sentence"]
                detail_type = "number edit"
                severity = classify_severity(detail_type, sentence)
                detail = f"{soap_frame['anchor_term']}: {_describe_attr(attr)} -- {sentence}"
                section = _section_for_snippet(soap_note, sentence, soap_line_sections)
                errors.append(("hallucination", severity, detail_type, detail, section))
        return errors

    def _extract_numeric_frames(self, text):
        """Phrase-scoped, not line-scoped (see PHRASE_SPLIT_RE's comment) --
        confirmed directly this session that line-scoping let a number in
        one sentence of a merged SOAP-note line anchor to an unrelated
        concept from a different sentence in the same line."""
        frames = {}
        for _speaker, line_text in split_turns(text):
            if not line_text.strip():
                continue
            for phrase in _split_phrases(line_text):
                phrase_attrs = extract_numeric_attributes(phrase)
                if not phrase_attrs:
                    continue  # no number in this phrase at all -- nothing to anchor

                anchor_key, anchor_term = self._find_numeric_anchor(phrase)
                if anchor_key is None:
                    continue  # a number with no identifiable context -- skip, don't guess

                frame = frames.setdefault(anchor_key, {"anchor_term": anchor_term, "sentence": phrase, "attrs": set()})
                frame["attrs"] |= phrase_attrs
        return frames

    def _find_numeric_anchor(self, line_text):
        """Returns (anchor_key, anchor_term), or (None, None) if line_text
        has nothing recognizable to anchor a number to. Concept anchors key
        on CUI (so the same concept mentioned in different words still
        matches on both sides); family-relation anchors key on the
        normalized relation word since there's no CUI for a relation."""
        matches = self._matcher.match(line_text, best_match=True, ignore_syntax=False)
        for group in matches:
            for m in group:
                if is_common_word(m["term"]) or is_common_word(m["ngram"]):
                    continue
                if _is_junk_concept(m["term"]) or _is_junk_concept(m["ngram"]):
                    continue
                if m.get("semtypes", set()) & NUMERIC_ANCHOR_SEMTYPES:
                    return f"CONCEPT:{m['cui']}", m["term"]

        family_match = FAMILY_RELATION_RE.search(line_text)
        if family_match:
            word = family_match.group(1).lower()
            return f"FAMILY:{word}", word

        return None, None

    def _judge_lasa(self, transcript, soap_note, soap_line_sections):
        """Flags a SOAP-note drug mention that's a known LASA counterpart of
        a DIFFERENT drug the transcript actually mentions -- e.g. transcript
        says "hydroxyzine", SOAP note says "hydralazine". Static pairlist,
        see LASA_PAIRS."""
        transcript_lower = transcript.lower()
        soap_lower = soap_note.lower()
        errors = []

        for drug_a, drug_b in LASA_PAIRS:
            soap_has_a, soap_has_b = drug_a in soap_lower, drug_b in soap_lower
            transcript_has_a, transcript_has_b = drug_a in transcript_lower, drug_b in transcript_lower

            confused = None
            if soap_has_a and transcript_has_b and not transcript_has_a:
                confused = (drug_a, drug_b)
            elif soap_has_b and transcript_has_a and not transcript_has_b:
                confused = (drug_b, drug_a)

            if confused is None:
                continue
            soap_drug, transcript_drug = confused
            detail_type = "lasa confusion"
            detail = f'SOAP note says "{soap_drug}", transcript says "{transcript_drug}" -- known look-alike/sound-alike pair'
            severity = classify_severity(detail_type, detail)
            # Locate the drug NAME itself, not the constructed detail string
            # (which never appears verbatim in soap_note) -- the anchor a
            # section lookup can actually find.
            section = _section_for_snippet(soap_note, soap_drug, soap_line_sections)
            errors.append(("hallucination", severity, detail_type, detail, section))

        return errors
