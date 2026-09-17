"""AlignScoreChecker -- the project's default AlignScore-based hallucination
checker. Flags hallucinated SOAP sentences by scoring each one-fact SOAP
claim against the transcript, using AlignScore's NLI head.

This file carries three generations of fix, in order, all still present:

Edit 1 -- claim-side sentence splitting (split_full_stop_sentences below).
AlignScore's own claim splitter is punctuation-only, which glues a whole
bullet-style SOAP line into one claim ("PMH: Asthma / DH: Inhalers / SH:
works as an accountant." has no period until the end). split_full_stop_sentences
adds newline and clinical-field-label ("PMH:", "DH:", ...) boundaries on top
of ./!/?, so each fact gets its own claim. This alone moved F1 from 0.077 to
0.348 on the original 10-file calibration sample (prim1, prim10-prim18) --
the single largest improvement of anything tried on this checker.

Edit 2 -- transcript-side Q&A bridging (superseded, see Modules/old/
alignscorechecker2.py). A doctor-question/patient-reply pair like "any arm
weakness?" / "No." never appears in the transcript as a declarative sentence
for the SOAP claim "No arm or leg weakness" to match against. The first
attempted fix generated a synthetic assertion ("No arm weakness.") and
appended it to the END of the transcript. That was additive -- the original
transcript untouched, new text bolted on -- and it backfired: appending at
the end reshuffled AlignScore's own ~350-token chunk boundaries for the
WHOLE file, not just the claims the bridging was meant to help (F1 fell
0.348 -> 0.320 on the same 10 files, all four new errors were false
positives with zero new true positives). Superseded by the pair-based
context chunking below, which targets the same problem structurally instead
of additively -- see PAIRS_PER_CHUNK and build_context_chunks.

Edit 3 -- THRESHOLD recalibration (0.30, not the earlier 0.25). Edit 1
changed what a "claim" looks like, so the pre-Edit-1 threshold (calibrated
on 6 hand-picked clean phrases) was never provably right for Edit 1's
output. Re-swept on the same 10-file sample (hallucination-type labels
only, every claim scored once and cached, then every threshold tried
against those cached scores):
  threshold=0.20: precision=0.381 recall=0.364 f1=0.372 (tp=8  fp=13 fn=14)
  threshold=0.25: precision=0.333 recall=0.364 f1=0.348 (tp=8  fp=16 fn=14)  <- old default
  threshold=0.30: precision=0.367 recall=0.500 f1=0.423 (tp=11 fp=19 fn=11)  <- adopted, best F1
  threshold=0.35: precision=0.324 recall=0.545 f1=0.407 (tp=12 fp=25 fn=10)
  threshold=0.50: precision=0.235 recall=0.545 f1=0.329 (tp=12 fp=39 fn=10)

Pair-based context chunking (current default, replacing AlignScore's own
context chunking) -- the structural fix that supersedes Edit 2. AlignScore's
own AlignScore.score()/Inferencer.inference_per_example splits CONTEXT
(the transcript) via nltk.sent_tokenize (punctuation only, no concept of
this dataset's "d:"/"p:" turn tags or newlines) and then groups those
sentences into coarse ~350-token blocks by word count alone. Confirmed
failure mode: two turns conditionally opposite each other (e.g. "go to A&E
if X" / "wait for your appointment if not X") can land inside the same
350-token chunk as an undifferentiated block, so a SOAP claim describing
only one side of that conditional gets scored against a chunk that also
contains its own contradiction. This checker replaces that chunking
mechanism entirely instead of layering fixes on top of it. No synthetic
text is ever created -- every chunk is built only from turns that were
actually said, kept in their original wording. The atomic unit is a
"pair": a maximal run of consecutive doctor turns immediately followed by
a maximal run of consecutive patient turns (not strictly one "d" + one
"p" -- if the doctor asks two things in a row before the patient replies,
that whole run is still one pair). A run that doesn't fit that shape (a
leading patient run with no preceding doctor turn, or a trailing run left
over at the end of the transcript with no partner) still forms its own
pair, just an incomplete one (kept, never dropped). PAIRS_PER_CHUNK pairs
are then joined to form one context chunk -- "any arm weakness?" / "No."
now sits, verbatim, inside its own small, tightly-scoped chunk, rather
than glued into a 350-token block alongside unrelated or contradictory
turns; the model infers the negation from the raw adjacency itself rather
than being handed a pre-written assertion.

Because AlignScore's own AlignScore.score()/Inferencer.inference_per_example
does its OWN sentence splitting and chunking on whatever string it is given
as "premise", there is no way to hand it pre-built chunks through the
public API -- it would just re-split them. This checker instead calls the
lower-level Inferencer.inference() directly (the same method
inference_per_example itself calls) and reimplements only the
max-over-chunks-then-mean-over-claim-sentences aggregation
inference_per_example already does (see that method's nli_sp branch) -- so
the only thing that changes relative to a stock AlignScore call is how the
context is split into chunks; claim splitting, the NLI head used, and the
aggregation formula are all left exactly as they were.

Re-swept on the same 10-file sample with THRESHOLD re-tuned for the new
chunking scheme (necessary -- the score distribution shifts along with the
chunk boundaries, the same reason Edit 3 re-swept after Edit 1):
  threshold=0.10: precision=0.400 recall=0.455 f1=0.426 (tp=10 fp=15 fn=12)  <- best F1, adopted
  threshold=0.30: precision=0.316 recall=0.545 f1=0.400 (tp=12 fp=26 fn=10)  <- old THRESHOLD, unchanged chunking-adjusted score
A marginal improvement over Edit 3's 0.423, not yet distinguishable from
noise on 10 files, but achieved without fabricating any content and without
Edit 2's collateral false positives. A full 57-file re-run is the next
validation step before treating either the approach or 0.10 as final.

Hallucination direction only -- the omission direction (transcript sentence
as claim, SOAP note as context) is not implemented, for the same structural
reason Edit 2 fell short: it would need the transcript split into
individual, well-formed CLAIMS (not just well-scoped context chunks), and
neither Edit 2 nor the pair-chunking above solves that -- pair-chunking
produces chunks of *grouped* turns for use as context, not the per-turn
claim-splitting the omission direction would need on its own.
"""

import os
import re
import sys
import time

from alignscore import AlignScore
from nltk.tokenize import sent_tokenize

from Modules.base import MIN_SENTENCE_WORDS, CheckerModule

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Medical condensor"))
from base import split_turns  # noqa: E402 -- path insert must run first

# Set to a downloaded AlignScore checkpoint (see yuh-zha/AlignScore -- checkpoints
# are on HuggingFace, not bundled with the package).
CKPT_PATH = r"C:\Users\natha\OneDrive\Documents\Uni\Impreial\modules\Thesis\code\data\AlignScore-base.ckpt"
MODEL = "roberta-base"

# Kept at Edit 3's original 0.30 rather than the 0.10 found in the 10-file
# re-sweep for this chunking scheme (see module docstring's "Pair-based
# context chunking" section) -- that re-sweep is not yet validated at scale
# (a full 57-file run is the pending next step), so the active threshold
# stays at the value already established across the rest of this project
# until that validation lands, rather than adopting an unconfirmed,
# small-sample result as the new default.
THRESHOLD = 0.30

# How many adjacency-pairs (see _build_pairs) are joined to form one context
# chunk. Chosen as a starting point, not swept -- see module docstring.
PAIRS_PER_CHUNK = 2

# Splits on ./!/? (real sentence endings) and on newlines (this dataset's
# notes put one fact per line at least as often as they use punctuation).
# FIELD_LABEL_RE adds a third boundary, a "/" specifically when followed by
# a recognized clinical field header, so a single-line note like "PMH:
# Asthma / DH: Inhalers / SH: works as an accountant." splits at each field
# instead of surviving as one glued claim. Deliberately NOT a blind split on
# every "/": this dataset also uses "/" to join two findings under the SAME
# assertion ("No SOB / chest pain"), and splitting that blindly would strand
# "chest pain" without the "No" negating it.
FIELD_LABEL_RE = (
    r"(?:pmh\w*|dh\w*|sh\w*|fh\w*|psh\w*|ice|hx|hpc|"
    r"imp|impression|dx|diagnosis|assessment|"
    r"plan|rx|management|mx|"
    r"o/e|exam\w*|obs|vitals?|bloods?|ix|investigations?)\s*:"
)
SENTENCE_SPLIT_RE = re.compile(
    rf"(?<=[.!?])\s+|\n+|\s*/\s*(?={FIELD_LABEL_RE})",
    re.IGNORECASE,
)


def split_full_stop_sentences(text):
    """Splits text on sentence/newline/field-label boundaries (see
    SENTENCE_SPLIT_RE above), dropping fragments under MIN_SENTENCE_WORDS
    words (too short to carry any checkable clinical content)."""
    return [
        s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip())
        if s and len(s.strip().split()) >= MIN_SENTENCE_WORDS
    ]


def _speaker_runs(turns):
    """Groups consecutive same-speaker turns into runs: [(speaker, [text, ...]), ...].
    A run boundary is any change in the normalized speaker label (including
    to/from None for an unrecognized turn prefix)."""
    runs = []
    for speaker, text in turns:
        if runs and runs[-1][0] == speaker:
            runs[-1][1].append(text)
        else:
            runs.append((speaker, [text]))
    return runs


def _build_pairs(turns):
    """Groups speaker runs into doctor-run+patient-run pairs.

    A pair is a maximal run of consecutive "d" turns immediately followed by
    a maximal run of consecutive "p" turns. A run that does not fit that
    shape -- a leading "p" run with no preceding "d" run, a trailing run
    left over at the end with no partner, or a run whose speaker tag was not
    recognized (None) -- still forms its own pair, just an incomplete one
    (kept, not dropped or merged into a neighbour).

    Verified against dpdpdpppdpppdddppddp -> dp/dp/dppp/dppp/dddpp/ddp.
    """
    runs = _speaker_runs(turns)
    pairs = []
    i = 0
    while i < len(runs):
        speaker, texts = runs[i]
        if speaker == "d" and i + 1 < len(runs) and runs[i + 1][0] == "p":
            pairs.append(texts + runs[i + 1][1])
            i += 2
        else:
            pairs.append(texts)
            i += 1
    return pairs


def build_context_chunks(transcript, pairs_per_chunk=PAIRS_PER_CHUNK):
    """Splits transcript into a list of context-chunk strings: pairs_per_chunk
    adjacency-pairs per chunk (see _build_pairs), each pair's turns joined by
    spaces, in original order. A trailing chunk with fewer than
    pairs_per_chunk pairs (when the pair count is not an exact multiple) is
    kept as its own, smaller chunk rather than dropped or merged backward."""
    turns = split_turns(transcript)
    pairs = _build_pairs(turns)
    chunks = []
    for i in range(0, len(pairs), pairs_per_chunk):
        group = pairs[i:i + pairs_per_chunk]
        chunk_text = " ".join(text for pair in group for text in pair)
        chunks.append(chunk_text)
    return chunks


def _score_claim_against_chunks(inferencer, claim, chunks):
    """Reimplements alignscore.inference.Inferencer.inference_per_example's
    nli_sp scoring (premise_sents -> chunks, cross product against claim
    sentences, NLI-head tri-label column 0, max over chunks then mean over
    claim sentences) but with caller-supplied chunks instead of the
    library's own nltk.sent_tokenize + word-count chunking. Claim splitting
    (sent_tokenize on an already-claim-split sentence) is left identical to
    the library's own behaviour, to isolate context-chunking as the only
    changed variable."""
    if not chunks:
        chunks = [""]
    hypo_sents = sent_tokenize(claim) or [claim]

    premise_mat = []
    hypo_mat = []
    for chunk in chunks:
        for hypo in hypo_sents:
            premise_mat.append(chunk)
            hypo_mat.append(hypo)

    tri_label_scores = inferencer.inference(premise_mat, hypo_mat)[2][:, 0]
    score = tri_label_scores.view(len(chunks), len(hypo_sents)).max(dim=0).values.mean().item()
    return score


class AlignScoreChecker(CheckerModule):
    """Flags hallucinated SOAP sentences using AlignScore: each one-fact SOAP
    claim (Edit 1's splitter) is scored against the transcript, re-chunked
    around its own turn structure (pair-based context chunking, see module
    docstring) rather than AlignScore's own punctuation-only chunking.

    Hallucination direction only -- see module docstring's closing note for
    why the omission direction isn't implemented.

    Not usable until CKPT_PATH points at a downloaded AlignScore checkpoint --
    raises immediately on construction if it isn't configured.
    """

    def __init__(self, ckpt_path=None, model=None, threshold=None, device="cpu", pairs_per_chunk=PAIRS_PER_CHUNK):
        ckpt_path = ckpt_path or CKPT_PATH
        if not ckpt_path:
            raise RuntimeError(
                "AlignScore is not configured. Set CKPT_PATH in Modules/alignscore_checker.py "
                "to a downloaded AlignScore checkpoint (see yuh-zha/AlignScore)."
            )
        self._scorer = AlignScore(
            model=model or MODEL,
            batch_size=32,
            device=device,
            ckpt_path=ckpt_path,
            evaluation_mode="nli_sp",
        )
        # Same quieting AlignScore.score()/inference_example_batch() does
        # internally -- calling Inferencer.inference() directly below
        # bypasses that, so it's set explicitly here instead.
        self._scorer.model.disable_progress_bar_in_inference = True
        self._threshold = THRESHOLD if threshold is None else threshold
        self._pairs_per_chunk = pairs_per_chunk

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        chunks = build_context_chunks(transcript, self._pairs_per_chunk)
        claims = split_full_stop_sentences(soap_note)
        errors = self._flag(claims=claims, chunks=chunks, error_type="hallucination")

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    def _flag(self, claims, chunks, error_type):
        if not claims:
            return []
        flagged = []
        for claim in claims:
            score = _score_claim_against_chunks(self._scorer.model, claim, chunks)
            if score < self._threshold:
                flagged.append((error_type, claim))
        return flagged
