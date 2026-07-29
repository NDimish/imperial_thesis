import time

from summac.model_summac import SummaCZS

from Modules.base import CheckerModule, split_sentences

# Uncalibrated on this data -- tune once you see real scores.
THRESHOLD = 0.5


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
