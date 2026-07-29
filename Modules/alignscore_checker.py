import time

from alignscore import AlignScore

from Modules.base import CheckerModule, split_sentences

# Set to a downloaded AlignScore checkpoint (see yuh-zha/AlignScore -- checkpoints
# are on HuggingFace, not bundled with the package).
CKPT_PATH = r"C:\Users\natha\OneDrive\Documents\Uni\Impreial\modules\Thesis\code\data\AlignScore-base.ckpt"
MODEL = "roberta-base"
# Uncalibrated on this data -- tune once you see real scores.
THRESHOLD = 0.5


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
