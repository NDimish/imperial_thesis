"""Shared UMLS concept-matching, used by QuickUMLSCondenser directly and by
MedspacyCondenser/NegspacyCondenser as a cross-check on top of their own
detection layer. Centralizing this in one place means all three condensers'
"is this actually a clinical concept" question is answered the same,
consistently-tuned way instead of drifting separately.
"""
import os

from quickumls import QuickUMLS

# Set this to your local QuickUMLS install directory. Requires a UMLS Metathesaurus
# license from NLM and a local build first:
#   1. Register for a UMLS license: https://www.nlm.nih.gov/research/umls/
#   2. Download a UMLS release and install it with MetamorphoSys
#   3. python -m quickumls.install <umls_install_dir> <this_destination_dir>
# See: https://github.com/Georgetown-IR-Lab/QuickUMLS
QUICKUMLS_INSTALL_DIR = r"C:\Users\natha\OneDrive\Documents\Uni\Impreial\modules\Thesis\code\data\quickumls_install"

# QuickUMLS defaults (threshold=0.7, min_match_length=3, ~27 accepted semtypes)
# are too permissive on spoken dialogue -- short common words (e.g. "hi", "sir")
# fuzzy-match spurious UMLS concepts at 0.7 Jaccard similarity, so almost every
# turn (including pure greetings/sign-offs) was being kept as "clinical".
#
# THRESHOLD was first raised to 0.85, but a real full-dataset NLP_Main.py run
# showed the fix that followed (a denylist) had made filtering much more
# aggressive at a real cost: the condensed-vs-original groundedness score got
# roughly 10x worse (-1.1 -> -11.3). Direct inspection of what an unrestricted
# QuickUMLS matcher actually returns for dropped-but-relevant phrases found why:
# "smoke"->"smoker" only reaches similarity=0.75, "peanuts"->"peanut" reaches
# 0.80, and "allergic"->"allergican" reaches 0.75 -- all below the 0.85 bar,
# even though their semantic types were already accepted. The real fix for the
# "start"/"well" false positives (both exact similarity=1.0, immune to ANY
# threshold) was always a denylist, not the threshold raise -- so 0.85 was
# mostly just costing real recall without doing the job it was raised for.
# Lowered back to 0.75: still well above the original too-permissive 0.7
# default, but low enough to keep these legitimate fuzzy matches.
#
# Also found two real semantic-type gaps the same way: "lung cancer" (family
# history) only matches under T191 (Neoplastic Process), and "loss of appetite"
# partly matches under T184 (Sign or Symptom) -- neither was in the accepted
# set below, a straightforward oversight given how central "sign or symptom"
# should be. Added both. ("travel" and "contacts" remain uncovered, under
# T058/T067/T170 -- deliberately NOT added, since those are broad/abstract
# types that were the original source of spurious matches; a known residual
# gap rather than a fix worth the false-positive risk.)
#
# A third gap, found while diagnosing NegspacyCondenser turn-level regressions
# on real transcripts: "mood's low" (a genuine depression-screening question)
# was being dropped entirely, even though "mood" itself matches UMLS at
# similarity=1.0. Direct inspection showed why -- "mood" only matches under
# T201 (Clinical Attribute) / T041 (Mental Process), "appetite" only under
# T040 (Organism Function), and neither type was accepted. These aren't edge
# cases: sleep, concentration, memory, motivation, irritability, and stress
# all match the same two missing types (T040/T041), and all are core
# depression/anxiety screening content for this dataset. Added T040/T041/T201.
# Verified this doesn't reopen the door to junk: re-tested every non-clinical
# word from the regression list (hi, ok, years, hello, welcome, jacob, peter,
# email, sir, alright, ohh, started, trying, tell) against just these three
# new types -- none match except "birth" (T040, exact similarity=1.0, from
# "date of birth"), which is the same kind of bounded coincidental collision
# as the words already in GENERIC_WORD_DENYLIST below, so it was added there
# instead of narrowing the semtypes back down.
THRESHOLD = 0.75
MIN_MATCH_LENGTH = 4
ACCEPTED_SEMTYPES = {
    "T023",  # Body Part, Organ, or Organ Component
    "T029",  # Body Location or Region
    "T031",  # Body Substance
    "T033",  # Finding
    "T034",  # Laboratory or Test Result
    "T037",  # Injury or Poisoning
    "T040",  # Organism Function (appetite, sleep)
    "T041",  # Mental Process (mood, concentration, memory, motivation)
    "T046",  # Pathologic Function
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction
    "T059",  # Laboratory Procedure
    "T060",  # Diagnostic Procedure
    "T061",  # Therapeutic or Preventive Procedure
    "T121",  # Pharmacologic Substance
    "T184",  # Sign or Symptom
    "T191",  # Neoplastic Process
    "T195",  # Antibiotic
    "T200",  # Clinical Drug
    "T201",  # Clinical Attribute (mood, stress)
}

# UMLS contains ordinary English words as literal clinical terminology -- e.g.
# "start" is a real T061 concept and "well" is a real T033 concept, both at
# similarity=1.0 (immune to any THRESHOLD).
#
# First attempt: a wordfreq-based Zipf-frequency cutoff (reject any match at
# or above 4.7), reasoning that real clinical terms and generic words sit in
# separate frequency bands. Verified WRONG at real full-dataset scale: a fresh
# NLP_Main.py run showed QuickUMLS/Negspacy's groundedness got worse, not
# better, after adding it. Root cause: raw English-wide word frequency doesn't
# distinguish "generic discourse filler" from "a symptom word that's also
# common in everyday speech" -- confirmed directly, "pain"=5.03, "blood"=5.10,
# "tired"=4.71, "sick"=4.88, "hurt"=4.94, and "sleep"=5.05 ALL sit at or above
# the cutoff, right alongside "hi"=5.00 and "birth"=4.77. No single frequency
# threshold can separate these -- some of the most important symptom words in
# this whole dataset were being silently discarded as "too common".
#
# This is a genuinely different, BOUNDED problem from scispacy's unbounded
# discourse-marker denylist (any word-shaped span can get NER-tagged there).
# Here, a false positive only happens when a specific common word happens to
# have an exact coincidental UMLS entry -- a small, stable, enumerable set.
# Every one found so far (start, well, good, fine, right, help) is an exact
# similarity=1.0 match with no real clinical usage anywhere in this dataset,
# unlike "pain"/"tired"/"sick" which are being used in their genuine clinical
# sense here. A short hardcoded list for this specific, bounded collision
# case is not the same unscalable pattern as hand-typing clinical vocabulary.
# A full 57-file audit of NegspacyCondenser also found several more recurring
# collisions (nothing/else/talk/think/aware/mean/symptom(s)/problem(s)/medical
# history), all real sim=1.0 UMLS matches with no clinical specificity. Tried
# adding them here and re-ran the full 57-file pass: groundedness got WORSE
# for Negspacy specifically (-11.81 -> -13.40), even though every added word
# was genuinely non-clinical. Root cause: unlike removing pure filler WITHIN
# a turn (which only ever helps, per the earlier synthetic test), denylisting
# a word that was a TURN'S ONLY qualifying real-concept match drops that
# whole turn -- so widening this list can cost real recall.
#
# Revisited later at much larger scale, in case the smaller test above just
# didn't have enough volume to be conclusive: counting every kept UMLS match
# across all 57 files found "like" (552x) and "think" (393x) matching UMLS at
# similarity=1.0, and "right" fuzzy-matching the UNRELATED canonical terms
# "bright"/"fright" (818x combined) since the denylist only checked the
# canonical "term", not QuickUMLS's "ngram" (actual matched surface text).
# Added "like"/"think"/"mean"/"talk"/"don't know" plus an ngram-aware
# denylist check, and re-ran the full 57-file pass: every condenser's KDE
# groundedness score got WORSE (QuickUMLS -3.81 -> -4.81, Medspacy -4.69 ->
# -5.79, Negspacy -9.00 -> -10.01). Reverted at the time, term-only check
# restored.
#
# Revisited AGAIN once check_omissions_cosine existed (see kdbe_check.py) --
# the KDE-based groundedness score used for every decision above was later
# found to be length-biased (see the manual-condensation test documented in
# the report), so "made the KDE score worse" wasn't actually evidence this
# was a bad change. Re-applied the same denylist words plus the ngram-aware
# check and re-ran the full 57-file pass under BOTH metrics: KDE groundedness
# still got worse as before (expected, and no longer the metric being
# trusted), but cosine_coverage -- which doesn't have the length bias --
# was UNCHANGED for every condenser (QuickUMLS/Medspacy -0.01, Negspacy
# -0.02, identical before and after), while percent-reduced went up 4-5
# points across the board (QuickUMLS 16.3% -> 20.7%, Medspacy 21.2% ->
# 25.3%, Negspacy 32.7% -> 36.8%). More condensing, same real coverage.
# Kept this time -- the earlier "reverted" verdicts above were judged
# against a metric now known to be an unreliable judge of this specific
# kind of change.
GENERIC_WORD_DENYLIST = {
    "start", "well", "good", "fine", "right", "help", "birth",
    "like", "think", "mean", "talk", "don't know", "i don't know",
    "nothing", "else", "aware", "symptom", "symptoms", "problem",
    "problems", "medical history",
}

_matcher = None


def get_matcher(install_dir=None):
    """Returns a shared, lazily-constructed QuickUMLS matcher instance (tuned
    per the constants above), so multiple condensers reuse one loaded index
    instead of each loading their own copy."""
    global _matcher
    if _matcher is None:
        target_dir = install_dir or QUICKUMLS_INSTALL_DIR
        if not target_dir:
            raise RuntimeError(
                "QuickUMLS is not configured. Set QUICKUMLS_INSTALL_DIR in "
                "umls_matching.py to your local QuickUMLS install directory "
                "(requires a UMLS license -- see Georgetown-IR-Lab/QuickUMLS)."
            )
        _matcher = QuickUMLS(
            target_dir,
            threshold=THRESHOLD,
            min_match_length=MIN_MATCH_LENGTH,
            accepted_semtypes=ACCEPTED_SEMTYPES,
        )
    return _matcher


def is_common_word(term):
    """True if term is a known coincidental UMLS/generic-English collision
    (see GENERIC_WORD_DENYLIST) rather than a genuine clinical mention."""
    return term.strip().lower() in GENERIC_WORD_DENYLIST


def _is_common_match(m):
    """A match is a coincidental collision if EITHER the canonical UMLS term
    it matched to, or the actual surface text it matched from (QuickUMLS's
    "ngram"), is denylisted -- these can differ, e.g. "right" fuzzy-matches
    the unrelated canonical terms "bright"/"fright" at similarity 0.75, so a
    term-only check never caught it even with "right" denylisted. See the
    GENERIC_WORD_DENYLIST comment above for why this is kept now."""
    return is_common_word(m["term"]) or is_common_word(m["ngram"])


def has_real_concept(text, install_dir=None):
    """True if text contains at least one UMLS concept match (within the
    tuned threshold/semtypes above) whose surface form isn't just a common
    English word coincidentally colliding with a UMLS entry."""
    if not text.strip():
        return False
    matcher = get_matcher(install_dir)
    matches = matcher.match(text, best_match=True, ignore_syntax=False)
    return any(not _is_common_match(m) for group in matches for m in group)


def get_real_concept_spans(text, install_dir=None):
    """Returns [(start, end), ...] character offsets for every UMLS concept
    match in text that isn't just a common English word (see is_common_word),
    for callers that need the match location, not just a yes/no answer (e.g.
    MedspacyCondenser injecting these as spaCy entities for ConText to assess)."""
    if not text.strip():
        return []
    matcher = get_matcher(install_dir)
    matches = matcher.match(text, best_match=True, ignore_syntax=False)
    return [
        (m["start"], m["end"])
        for group in matches
        for m in group
        if not _is_common_match(m)
    ]
