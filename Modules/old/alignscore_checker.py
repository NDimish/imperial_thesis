import time

from alignscore import AlignScore

from Modules.base import CheckerModule, split_sentences

# Set to a downloaded AlignScore checkpoint (see yuh-zha/AlignScore -- checkpoints
# are on HuggingFace, not bundled with the package).
CKPT_PATH = r"C:\Users\natha\OneDrive\Documents\Uni\Impreial\modules\Thesis\code\data\AlignScore-base.ckpt"
MODEL = "roberta-base"
# THRESHOLD=0.5 assumed AlignScore's raw score centers near the 0-1 midpoint on
# this domain -- it doesn't. A labeled 5-file/8-hallucination threshold sweep
# (condensed transcript vs bad SOAP note, hallucination direction) found
# threshold=0.5 gives precision=0.03 (near useless: 60 FP for 2 TP), while
# threshold=0.05 gives precision=0.50 and roughly triples F1 (0.06 -> 0.20).
# Moved down to 0.1 accordingly.
#
# Re-swept later, both directions this time (hallucination + omission
# together) and after fixing Modules/base.py's sentence splitter to drop
# <4-word fragments ("d: Okay." style turn fragments were guaranteed false
# positives). At 0.1 that gave precision=0.043 f1=0.078 -- worse than hoped.
# A finer sweep (0.02-0.5) found the real optimum is lower still: 0.02 gives
# precision=0.136 recall=0.250 f1=0.176, more than double the F1 at 0.1, and
# every threshold above 0.05 does progressively worse (recall keeps climbing
# but precision collapses faster). Moved down to 0.02. Still only a 5-file
# calibration -- re-sweep across more of the 57 files before treating this as
# final.
THRESHOLD = 0.02


class AlignScoreChecker(CheckerModule):
    """Flags hallucinated SOAP sentences and omitted transcript sentences using
    AlignScore, scored one sentence at a time against the other full document.

    AlignScore is the only one of the 5 factual-consistency libraries checked in
    this project that genuinely supports both directions -- its score() takes
    separate contexts/claims lists, so swapping which document is which gives a
    real omission check, not an unvalidated hack.

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

        errors = []
        errors.extend(self._flag(claims=split_sentences(soap_note), context=transcript, error_type="hallucination"))
        errors.extend(self._flag(claims=split_sentences(transcript), context=soap_note, error_type="omission"))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    def _flag(self, claims, context, error_type):
        if not claims:
            return []
        scores = self._scorer.score(contexts=[context] * len(claims), claims=claims)
        return [(error_type, sentence) for sentence, score in zip(claims, scores) if score < self._threshold]
