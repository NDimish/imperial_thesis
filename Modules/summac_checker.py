import time

from summac.model_summac import SummaCZS

from Modules.base import CheckerModule, split_sentences

# THRESHOLD=0.5 was a copy-pasted assumption that this behaves like AlignScore's
# 0-1 alignment probability. It doesn't: SummaCZS's zero-shot score is signed,
# roughly in [-1, 1]. Confirmed directly on prim42.txt -- the HIGHEST score
# across all 20 SOAP sentences was 0.09, meaning 0.5 could never be satisfied by
# anything, ever (100% of every file gets flagged as hallucination, always).
# Moved into the metric's real range -- but a labeled 5-file/8-hallucination
# threshold sweep found precision stays poor (~9%) even at the sweep's best
# setting (-0.6), so this fixes the "always flags everything" bug without
# claiming SummaC is actually a strong hallucination detector on this domain.
THRESHOLD = 0.0


class SummaCChecker(CheckerModule):
    """Flags hallucinated SOAP sentences using SummaC (NLI-based consistency).

    Hallucination-only: SummaC is a one-directional (document -> summary)
    consistency model. Its score() returns a "images" NLI matrix internally, but
    doesn't expose the sentence text it split into -- re-implementing its internal
    splitter to read the matrix would be fragile, so this scores one sentence at
    a time instead, which is slower but keeps the module in full control of what
    "sentence" means and avoids index-misalignment risk.
    """

    def __init__(self, threshold=None, device="cpu"):
        self._model = SummaCZS(granularity="sentence", model_name="vitc", device=device)
        self._threshold = THRESHOLD if threshold is None else threshold

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        errors = []
        for sentence in split_sentences(soap_note):
            result = self._model.score([transcript], [sentence])
            score = result["scores"][0]
            if score < self._threshold:
                errors.append(("hallucination", sentence))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed
