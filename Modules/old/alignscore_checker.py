import re
import time

from alignscore import AlignScore

from Modules.base import MIN_SENTENCE_WORDS, CheckerModule

# Set to a downloaded AlignScore checkpoint (see yuh-zha/AlignScore -- checkpoints
# are on HuggingFace, not bundled with the package).
CKPT_PATH = r"C:\Users\natha\OneDrive\Documents\Uni\Impreial\modules\Thesis\code\data\AlignScore-base.ckpt"
MODEL = "roberta-base"

# 0.30, not the earlier 0.25: re-swept after the field-label/newline
# splitter fix above (see that comment) changed what actually reaches
# AlignScore as a claim -- the 0.25 default was calibrated on 6 hand-picked
# clean phrases (see the old comment this replaced), not on real post-fix
# SOAP claims, so it was never provably right for this splitter's output.
# Real 10-file sweep (hallucination-type labels only, all claims scored
# once and cached, then every threshold tried against those cached scores):
#   threshold=0.20: precision=0.381 recall=0.364 f1=0.372 (tp=8  fp=13 fn=14)
#   threshold=0.25: precision=0.333 recall=0.364 f1=0.348 (tp=8  fp=16 fn=14)  <- old default
#   threshold=0.30: precision=0.367 recall=0.500 f1=0.423 (tp=11 fp=19 fn=11)  <- new default, best F1
#   threshold=0.35: precision=0.324 recall=0.545 f1=0.407 (tp=12 fp=25 fn=10)
#   threshold=0.50: precision=0.235 recall=0.545 f1=0.329 (tp=12 fp=39 fn=10)
# 0.30 is a strict improvement over 0.25 (higher precision AND higher
# recall, not a tradeoff), and sits at the peak of the sweep -- F1 falls off
# on both sides of it. Still only 10 files -- re-sweep across more of the 57
# before treating this as final, same caveat as every other threshold here.
THRESHOLD = 0.30

# Splits on ./!/? (real sentence endings) and on newlines (this dataset's
# notes put one fact per line at least as often as they use punctuation --
# same fix Modules.base.split_sentences already made for exactly this
# reason). Confirmed real failure this splitter alone doesn't fix: a note
# like "PMH: Asthma / DH: Inhalers / SH: works as an accountant." has no
# period until the very end, so even with newline-splitting a single-line
# note still glues three unrelated facts into one claim -- which then fails
# entailment as a whole even when every individual fact in it is true (see
# extra/explainer/checker_audit_report.html). FIELD_LABEL_RE below adds a
# third boundary, a "/" specifically when followed by a recognized clinical
# field header, to split exactly that case at each new field.
#
# Deliberately NOT a blind split on every "/": this dataset also uses "/"
# to join two findings under the SAME assertion ("No SOB / chest pain" =
# "no shortness of breath, no chest pain") -- splitting that blindly would
# separate "chest pain" from the "No" negating it, silently flipping a
# negative claim into a positive one. Requiring a field label after the
# slash (not just any word) keeps that shorthand intact while still
# catching the field-separator case.
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
    words (same validated fix as Modules.base.split_sentences -- see its
    docstring for why short fragments are guaranteed false positives)."""
    return [
        s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip())
        if s and len(s.strip().split()) >= MIN_SENTENCE_WORDS
    ]


class AlignScoreChecker(CheckerModule):
    """Flags hallucinated SOAP sentences using AlignScore, scored one
    full-stop-terminated sentence at a time against the whole transcript.

    Hallucination direction only for now (SOAP sentence = claim, transcript =
    context) -- the omission direction (transcript sentence vs SOAP note) is
    not run here.

    Not usable until CKPT_PATH points at a downloaded AlignScore checkpoint --
    raises immediately on construction if it isn't configured.
    """

    def __init__(self, ckpt_path=None, model=None, threshold=None, device="cpu"):
        ckpt_path = ckpt_path or CKPT_PATH
        if not ckpt_path:
            raise RuntimeError(
                "AlignScore is not configured. Set CKPT_PATH in alignscore_checker.py "
                "to a downloaded AlignScore checkpoint (see yuh-zha/AlignScore)."
            )
        self._scorer = AlignScore(
            model=model or MODEL,
            batch_size=32,
            device=device,
            ckpt_path=ckpt_path,
            evaluation_mode="nli_sp",
        )
        self._threshold = THRESHOLD if threshold is None else threshold

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        claims = split_full_stop_sentences(soap_note)
        errors = self._flag(claims=claims, context=transcript, error_type="hallucination")

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    def _flag(self, claims, context, error_type):
        if not claims:
            return []
        scores = self._scorer.score(contexts=[context] * len(claims), claims=claims)
        return [(error_type, sentence) for sentence, score in zip(claims, scores) if score < self._threshold]
