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
THRESHOLD = 1
MIN_MATCH_LENGTH = 4
ACCEPTED_SEMTYPES = {
    # --- DISORDERS & SYMPTOMS (Core Clinical Signal) ---
    "T047",  # Disease or Syndrome (e.g., Gastroenteritis, Asthma)
    "T184",  # Sign or Symptom (e.g., Diarrhea, Fever, Cramps)
    "T046",  # Pathologic Function (e.g., Inflammation)
    "T037",  # Injury or Poisoning (e.g., Fracture, Burn)
    "T048",  # Mental or Behavioral Dysfunction (e.g., Depression)
    "T191",  # Neoplastic Process (e.g., Tumor)

    # --- DRUGS & CHEMICALS ---
    "T121",  # Pharmacologic Substance (e.g., Paracetamol)
    "T200",  # Clinical Drug (e.g., Inhaler, Dioralyte)
    "T195",  # Antibiotic

    # --- PROCEDURES & MANAGEMENT ---
    "T060",  # Diagnostic Procedure (e.g., Stool Test, X-ray)
    "T061",  # Therapeutic or Preventive Procedure (e.g., Hydration)
    "T059",  # Laboratory Procedure
    "T058",  # Health Care Activity (ADD THIS: e.g., Consultation, Follow-up, Off work)

    # --- ANATOMY & SUBSTANCES ---
    "T023",  # Body Part, Organ, or Organ Component (e.g., Stomach)
    "T029",  # Body Location or Region (e.g., Abdomen, Lower Left Quadrant)
    "T031",  # Body Substance (e.g., Stool, Blood, Vomit)
    "T034",  # Laboratory or Test Result

    # --- PHYSIOLOGY & FUNCTIONS ---
    "T040",  # Organism Function (e.g., Appetite, Sleep)
    
    # --- REMOVED FOR TRANSCRIPT PRECISION ---
    # "T033",  # Finding (REMOVED: Too much conversational noise like "well", "work")
    # "T201",  # Clinical Attribute (REMOVED: Matches non-clinical words like "left")
    # "T041",  # Mental Process (REMOVED unless evaluating psychiatric transcripts)
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
# "related" and "always" found the same way, via medspacy_condenser_new.py's
# per-sentence granularity making a coincidental collision visible as a
# dangling non-sequitur rather than just riding along inside an otherwise
# clinical turn: "So that could be related." (no real clinical content) was
# surviving on its own -- "related" is an exact sim=1.0 UMLS match, same
# bounded-collision pattern as "well"/"start"/"right" above. "always"
# collided the same way inside "I'm not always very good about wearing
# gloves" (only excluded from that sentence because it also got tagged
# is_negated by "not always", not because the denylist caught it).
GENERIC_WORD_DENYLIST = {
    "start", "well", "good", "fine", "right", "help", "birth",
    "like", "think", "mean", "talk", "don't know", "i don't know",
    "nothing", "else", "aware", "symptom", "symptoms", "problem",
    "problems", "medical history", "related", "always",

    # Pure conversational filler ("Um, not that I know of.") -- added after
    # Modules/high_risk_checker.py's Adjacency Pair Context Propagation
    # needed its own local filler-word filter (FILLER_ONLY_RE) to stop a
    # leading "Um," from splitting off as its own empty clause and
    # swallowing the real reply text. None of these matched anything via a
    # direct standalone check against the current UMLS build, but kept here
    # too, project-wide, as a zero-cost defensive measure -- a real
    # coincidental match for one of these in some sentence context (the
    # same way "well"/"start" only surfaced in context, not in isolation)
    # costs nothing to pre-empt and nothing to keep if it never fires.
    "um", "uh", "okay", "ok", "so",

    # Below: three more rounds of the exact same bounded-collision pattern
    # as everything above, added together rather than derived one at a time.
    #
    # (1) NLM's own MetaMap/MetaMapLite ships an analogous "suppress" list --
    # concept-independent cui|term pairs excluded regardless of confidence,
    # for exactly this "ordinary English word with a coincidental UMLS
    # entry" reason (their own examples: "was"/Wasp, "all"/Acute Lymphocytic
    # Leukemia, "it"/Information Technology, "die"/DIE tool). Confirmed the
    # mechanism is real via NLM's public gov/lhncbc/metamaplite GitHub repo
    # (SpecialTerms.java implements the exclusion; MIN_MATCH_LENGTH=4 above
    # means anything under 4 characters -- "was", "it", "all", "die" -- can
    # never surface as a match here regardless of this list, so they're
    # included below for completeness/parity with NLM's list but are inert
    # in this project's matcher). The full production list itself ships
    # inside NLM's UMLS-licensed data package, not the open-source repo, so
    # isn't fetchable without UMLS credentials -- what IS directly verified,
    # from metamaplite's own test fixture (TestExcludedTerms.java, 8 real
    # cui:term suppression pairs used as its unit test), is: idea, commit,
    # bars, still, cars, ran, fact, trade.
    "was", "all", "it", "die",
    "idea", "commit", "bars", "still", "cars", "ran", "fact", "trade",

    # (2) Confirmed directly on this project's own pipeline.py run (10-file
    # false-positive audit, see the FP-diagnostic investigation): reading
    # through every MedspacyUmlsChecker "text absent everywhere" false
    # positive across 10 real files found these recurring as the exact same
    # collision pattern as "start"/"well"/"right" above -- hedge/filler
    # words with a coincidental UMLS entry, not real clinical content.
    # "maybe"/"probably" alone accounted for ~10 of the 154 false positives
    # audited.
    "maybe", "probably", "sort", "thought", "definitely", "basis",
    "screen", "call", "back", "find", "practice", "worried", "worry",
    "usually", "same", "none", "settle", "flush", "not much", "not sure",
    "very rare", "very much", "one thing",

    # (3) spaCy's built-in English stopword list (spacy.lang.en.stop_words.
    # STOP_WORDS, 326 words -- already a project dependency, spacy==3.7.5),
    # filtered down before adding rather than dumped in whole. A generic
    # linguistic stopword list is a different curation goal than (1)/(2)
    # above -- most of it is safe by definition (function words can't be
    # standalone clinical content), but some entries ARE real clinical
    # terms as bare words ("serious", "move", "alone", "used"...) and
    # blindly adding all 326 risked costing real recall the same way every
    # over-eager addition earlier in this file's history did.
    #
    # Filtered using this project's own local UMLS install (2026AA
    # Metathesaurus -- MRCONSO.RRF + MRSTY.RRF) rather than guessing: for
    # each of the 293 spaCy stopwords not already covered above, checked
    # whether it has ANY UMLS English atom under one of this project's own
    # ACCEPTED_SEMTYPES (the same ~20-TUI set QuickUMLS is scoped to
    # below). If it doesn't -- either no UMLS entry at all, or only under
    # semtypes outside this project's accepted set -- it can never surface
    # as a match here regardless of this list, so it's safe to add. 248 of
    # 293 cleared that bar; kept below. The other 45 ("a", "an", "is", "no",
    # "one", "so", "will", "you", "move", "alone", "rather", "serious",
    # "used", "various", ...) have a real entry under an accepted clinical
    # semtype and were deliberately left OUT -- most are also under
    # MIN_MATCH_LENGTH=4 and inert either way, but the handful that aren't
    # ("alone", "either", "rather", "serious", "move", "used", "various")
    # are genuinely plausible as real clinical content on their own (e.g.
    # "lives alone" is real, significant social-history content elsewhere
    # in this project's own audited files) and weren't worth the same risk
    # already reverted once for "smoker"/"triggers"/"stools" above.
    'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'almost',
    'along', 'already', 'also', 'although', 'among', 'amongst', 'amount', 'and', 'another',
    'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'became',
    'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind',
    'being', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'bottom', 'by',
    'cannot', 'could', 'do', 'does', 'doing', 'done', 'down', 'due', 'during', 'each',
    'eight', 'eleven', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'every', 'everyone',
    'everything', 'everywhere', 'except', 'few', 'fifteen', 'fifty', 'first', 'former',
    'formerly', 'forty', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'has',
    'have', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers',
    'herself', 'him', 'himself', 'how', 'however', 'hundred', 'indeed', 'into', 'its',
    'itself', 'just', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'made', 'make',
    'many', 'may', 'meanwhile', 'might', 'more', 'moreover', 'most', 'mostly', 'must', 'my',
    'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine',
    'nobody', 'noone', 'nor', 'not', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once',
    'only', 'onto', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'part',
    'perhaps', 'please', 'put', 'quite', 'really', 'regarding', 'say', 'seem', 'seemed',
    'seeming', 'seems', 'several', 'she', 'should', 'show', 'side', 'since', 'six', 'sixty',
    'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'such',
    'take', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there',
    'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'third',
    'this', 'those', 'though', 'through', 'throughout', 'thru', 'thus', 'to', 'together',
    'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'under', 'unless',
    'until', 'up', 'upon', 'using', 'very', 'via', 'we', 'were', 'what', 'whatever', 'when',
    'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein',
    'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever',
    'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'would', 'yet', 'your',
    'yours', 'yourself', 'yourselves',

    # (4) Trial group -- the 45 "clinical hit" exclusions from (3) above were tiered:
    # 31 of 45 are under MIN_MATCH_LENGTH=4 and therefore already inert (excluding them
    # cost nothing, including them changes nothing -- e.g. "a", "an", "be", "his"), leaving
    # only 14 words actually long enough to ever match here. Of those 14, only "five"
    # (T047 Disease/Syndrome) and "mine" (T061 Procedure) are anchored by a semtype this
    # project's own history hasn't flagged as overly broad; the other 12 are protected
    # ONLY by "risky" TUIs already documented above as promiscuous (T033 Finding --
    # by far the single biggest source of the 45 exclusions -- plus T040/T041/T121/T201).
    # "alone" held out of this group deliberately: same T033-only profile as the rest,
    # but already directly confirmed as real content in a real file ("lives alone",
    # genuine social history) during the FP-diagnostic audit -- proof the TUI tier alone
    # isn't a fully reliable signal, not just a theoretical caveat.
    #
    # The remaining 12 -- "either", "move", "much", "other", "others", "rather",
    # "serious", "three", "used", "various", "will" -- were added as a TRIAL, some at
    # real volume in this dataset (234/145/126/89 occurrences for other/much/three/will
    # respectively -- an order of magnitude more than any previously-added word). A
    # pipeline.py rerun immediately after adding these held every checker's TP count
    # exactly steady (MedspacyUmlsChecker 4, MetaMapCuiChecker 5, DeterministicChecker 0,
    # merged 4 -- identical to before this group) with FP still dropping slightly further
    # -- kept.
    'either', 'move', 'much', 'other', 'others', 'rather', 'serious', 'three',
    'used', 'various', 'will',
}

# Keyed by (install_dir, threshold) so a caller asking for a non-default
# threshold (e.g. medspacy_condenser_new.py's CUSTOM_MATCHER_THRESHOLD) gets
# its own matcher instance instead of silently reusing -- or silently
# overwriting -- the shared THRESHOLD=1 instance the other three condensers
# depend on.
_matchers = {}


def get_matcher(install_dir=None, threshold=None):
    """Returns a shared, lazily-constructed QuickUMLS matcher instance for
    the given threshold (defaults to the module-level THRESHOLD, tuned for
    the other three condensers -- see the tuning history above), so multiple
    callers at the SAME threshold reuse one loaded index instead of each
    loading their own copy."""
    target_dir = install_dir or QUICKUMLS_INSTALL_DIR
    resolved_threshold = THRESHOLD if threshold is None else threshold
    cache_key = (target_dir, resolved_threshold)
    if cache_key not in _matchers:
        if not target_dir:
            raise RuntimeError(
                "QuickUMLS is not configured. Set QUICKUMLS_INSTALL_DIR in "
                "umls_matching.py to your local QuickUMLS install directory "
                "(requires a UMLS license -- see Georgetown-IR-Lab/QuickUMLS)."
            )
        _matchers[cache_key] = QuickUMLS(
            target_dir,
            threshold=resolved_threshold,
            min_match_length=MIN_MATCH_LENGTH,
            accepted_semtypes=ACCEPTED_SEMTYPES,
        )
    return _matchers[cache_key]


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


def has_real_concept(text, install_dir=None, threshold=None):
    """True if text contains at least one UMLS concept match (within the
    tuned threshold/semtypes above) whose surface form isn't just a common
    English word coincidentally colliding with a UMLS entry."""
    if not text.strip():
        return False
    matcher = get_matcher(install_dir, threshold)
    matches = matcher.match(text, best_match=True, ignore_syntax=False)
    return any(not _is_common_match(m) for group in matches for m in group)


def get_real_concept_spans(text, install_dir=None, threshold=None):
    """Returns [(start, end), ...] character offsets for every UMLS concept
    match in text that isn't just a common English word (see is_common_word),
    for callers that need the match location, not just a yes/no answer (e.g.
    MedspacyCondenser injecting these as spaCy entities for ConText to assess).

    threshold overrides the module-level THRESHOLD for this call only (see
    get_matcher) -- e.g. medspacy_condenser_new.py runs a bit below 1.0 to
    catch near-exact morphological variants without touching the other three
    condensers' tuned exact-match behavior."""
    if not text.strip():
        return []
    matcher = get_matcher(install_dir, threshold)
    matches = matcher.match(text, best_match=True, ignore_syntax=False)
    return [
        (m["start"], m["end"])
        for group in matches
        for m in group
        if not _is_common_match(m)
    ]
