import time

from summac.model_summac import SummaCZS

from Modules.base import CheckerModule, SENTENCE_SPLIT_PATTERN

# SummaCZS's zero-shot score is signed, roughly in [-1, 1] (entailment minus
# contradiction), NOT a 0-1 probability -- see Modules/old/summac_checker.py's
# documented history of the THRESHOLD=0.5 bug that made it flag literally
# everything. Kept at 0.0 for both directions here since the underlying
# entailment/contradiction math is identical either way; separate constants
# because the two directions are different comparisons (note-vs-transcript,
# transcript-vs-note) that could end up calibrating differently in a real
# sweep -- see the NOT-RE-VALIDATED note in the class docstring below.
HALLUCINATION_THRESHOLD = 0.0
OMISSION_THRESHOLD = 0.0

# Modules/base.py's shared split_sentences() drops anything under
# MIN_SENTENCE_WORDS=4 -- right for an LLM-prompt checker (a 3-word fragment
# like "Yeah." or "d: Fine." is pure noise there), wrong for SummaC: this
# dataset's injected errors are often exactly this terse ("Opening bowels
# x60/day." and "blood in vomit." are both 3 words) because that's how SOAP
# notes are actually written. CONFIRMED directly on prim1.txt -- both of
# that file's real injected hallucinations were silently dropped before
# SummaC ever saw them, guaranteeing FN=2 on that file regardless of
# threshold. SummaC doesn't have the LLM-prompt short-fragment noise problem
# the 4-word floor exists for (an NLI entailment score on a 3-word premise
# isn't inherently less meaningful than on a 10-word one), so this uses the
# same split pattern (period/newline boundaries) with a much lower floor
# instead of the shared function -- 2 words, not 0, still excludes truly
# empty/single-token noise from the split.
MIN_SENTENCE_WORDS = 2

# CONFIRMED via direct score inspection on prim1.txt: SummaCImager's OWN
# internal split_sentences() (summac/model_summac.py, used to re-chunk
# whichever side gets built per NLI-pair) drops anything with
# `len(sent) <= 10` characters -- a SEPARATE, hidden floor underneath ours.
# When our per-item hypothesis text is that short, SummaC's internal
# generated_chunks list comes back EMPTY, build_chunk_dataset returns an
# empty dataset, and build_image short-circuits to a degenerate all-zero
# image (np.zeros((3, 1, 1))) -- which image2score turns into an exact,
# meaningless score of 0.0. Confirmed directly: every single transcript
# turn under 10 characters ("d: Hello?", "d: Okay.", "p: Yep.", "p: Yeah.")
# scored EXACTLY +0.000 in the omission-direction dump, not a real
# entailment judgment -- coincidentally landing on the "not flagged" side of
# threshold=0.0 for these (harmless, since none of them were real omissions
# either), but there is nothing stopping a genuinely important short fact
# from being silently swallowed the same way. Guarding here so this project
# never quietly trusts a fake score: MIN_SENTENCE_CHARS is set one above
# SummaC's own floor so nothing we pass in can ever hit that degenerate path.
MIN_SENTENCE_CHARS = 11


def split_sentences_min2(text):
    return [
        s for s in SENTENCE_SPLIT_PATTERN.split(text.strip())
        if s and len(s.split()) >= MIN_SENTENCE_WORDS and len(s) > MIN_SENTENCE_CHARS
    ]


class SummaCChecker(CheckerModule):
    """Flags both hallucinated AND omitted SOAP content using SummaC
    (NLI-based factual consistency). Fixed from Modules/old/summac_checker.py
    in two ways -- that file is left as-is for reference, this is the active
    version run_checker_modules.py already expects at this import path:

    1. Omission detection added. The old version was hallucination-only:
       SummaC's score() tests whether a HYPOTHESIS is entailed by a DOCUMENT
       (premise) -- used there as "is this SOAP sentence supported by the
       transcript". Omission is the identical entailment test run in the
       OTHER direction: "is this TRANSCRIPT sentence supported by the SOAP
       note" -- if the note doesn't entail a transcript fact, that fact was
       left out. Two directions through the same underlying model and
       scoring function, not a second technique bolted on.

    2. Speed. The old version called .score() once PER SENTENCE, so a
       15-sentence SOAP note against an 80-sentence transcript ran 15
       separate small-batch-of-20 loops, each re-splitting the whole
       transcript from scratch -- confirmed at 1,829s/file average, one file
       at 71 minutes (Logs/old2/old3/evaluate_SummaCChecker_20260731_174428.log).
       This calls .score() ONCE PER DIRECTION per file instead of once per
       sentence -- SummaC's own batch_size=128 pools every (doc-sentence,
       our-sentence) pair for the whole file into large batches rather than
       many independent small ones. Total NLI comparisons are the same
       order (comparing every SOAP sentence against every transcript
       sentence is inherent to a sentence-level consistency matrix and
       doesn't shrink by batching) -- what shrinks is the Python/tokenizer/
       model-call round-trip count. Also switched the default device to
       "cuda": confirmed a real GPU is available on this machine (RTX 3050
       Ti) -- the environment's installed torch was a CPU-only wheel despite
       the driver supporting CUDA 13.1, reinstalled as the +cu126 build
       (same 2.13.0 version pinned in requirements.txt) to actually see it.

    Still uses OUR OWN sentence splitting (split_sentences_min2 above -- the
    same period/newline pattern Modules/base.py's split_sentences() uses, but
    a 2-word floor instead of 4, see that constant's comment) for whichever
    side is being iterated per call, for the same index-alignment reason the
    old version's docstring gave: SummaC's own internal nltk-based splitter
    would silently disagree with this project's turn/newline-aware one on
    disfluent transcript text, and there'd be no reliable way to map a
    matrix column back to "which of our sentences is this" if it did.

    3. Sub-11-character fragments excluded entirely (MIN_SENTENCE_CHARS
       above) -- SummaC's OWN internal chunker silently zero-scores anything
       that short rather than giving it a real judgment; see that constant's
       comment for how this was found and confirmed.

    Two things this does NOT fix, confirmed by inspecting raw (pre-
    threshold) scores directly on prim1.txt rather than guessing:

    - Hallucination precision has a real ceiling no threshold clears. Sorting
      prim1.txt's SOAP sentences by score, the two REAL injected errors
      ("Opening bowels x60/day.", "blood in vomit.") land at -0.790/-0.793 --
      right in the MIDDLE of the distribution. Several genuinely CORRECT
      sentences score more "hallucinated" than either real error: "Fever on
      first day, nil since." at -0.922 (the single most-hallucinated-looking
      sentence in the whole file), "Conservative management..." at -0.871,
      "Imp: gastroenteritis" at -0.811. No cutoff separates real errors from
      real content here -- this independently reproduces
      Modules/old/summac_checker.py's own documented ~9%-precision sweep
      result, not a coincidence. This is the vitc/albert-xlarge-vitaminc-mnli
      model's actual discriminative power on this domain, not a
      threshold-tuning gap.

    - Omission's false-positive flood isn't a short-fragment or
      filler-content problem either. Tried gating transcript sentences on
      Medical condensor/umls_matching.has_real_concept() (skip non-clinical
      turns before the NLI check) and confirmed it wouldn't have helped: the
      lowest-scoring ("most omitted") sentences are a MIX of clinical and
      non-clinical content in roughly equal measure -- e.g. "So, um, I don't
      think you need anything like antibiotics." (clinical=True) scores
      -0.939, right alongside "p: Um, just soups, and, uh, yeah, light
      foods." (clinical=False) at -0.979. The real problem is that strict
      NLI entailment doesn't recognize a PARAPHRASED clinical fact as
      "entailed" by a condensed note -- a lexical/semantic-form problem, not
      a content-relevance one, so a relevance filter doesn't touch it. This
      project already has a validated, paraphrase-tolerant coverage metric
      for exactly this comparison direction (embedding cosine similarity,
      not NLI) -- see kdbe_check.check_omissions_bidirectional_cosine's own
      validation history -- worth using instead of forcing NLI into this
      role if omission detection specifically is the goal.

    In short: the two MECHANICAL problems (no omission direction; too slow
    to run at all) are genuinely fixed. The two ACCURACY problems are a real
    ceiling on this model/domain pairing, confirmed by direct evidence here,
    not something left un-swept.
    """

    def __init__(self, hallucination_threshold=None, omission_threshold=None, device="cuda"):
        self._model = SummaCZS(granularity="sentence", model_name="vitc", device=device)
        self._halluc_threshold = HALLUCINATION_THRESHOLD if hallucination_threshold is None else hallucination_threshold
        self._omission_threshold = OMISSION_THRESHOLD if omission_threshold is None else omission_threshold

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        soap_sentences = split_sentences_min2(soap_note)
        transcript_sentences = split_sentences_min2(transcript)

        errors = []

        # Hallucination: is each SOAP sentence (hypothesis) entailed by the
        # transcript (document)? One call for the whole file, not one per
        # sentence -- see the class docstring's point 2.
        if soap_sentences:
            halluc_scores = self._model.score([transcript] * len(soap_sentences), soap_sentences)["scores"]
            for sentence, score in zip(soap_sentences, halluc_scores):
                if score < self._halluc_threshold:
                    errors.append(("hallucination", sentence))

        # Omission: is each TRANSCRIPT sentence (hypothesis) entailed by the
        # SOAP note (document)? Same model, same scoring, direction flipped.
        if transcript_sentences:
            omission_scores = self._model.score([soap_note] * len(transcript_sentences), transcript_sentences)["scores"]
            for sentence, score in zip(transcript_sentences, omission_scores):
                if score < self._omission_threshold:
                    errors.append(("omission", sentence))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed
