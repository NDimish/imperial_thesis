import time

from factsumm import FactSumm

from Modules.base import CheckerModule, split_sentences

# Uncalibrated on this data -- tune once you see real scores.
THRESHOLD = 0.5


class FactSummChecker(CheckerModule):
    """Flags hallucinated SOAP sentences using FactSumm's fact_score.

    Hallucination-only: FactSumm's QAGS module generates questions from the
    summary and checks them against the source, with no reverse/omission
    direction. Its extracted (subject, predicate, object) triples aren't exposed
    through the public API either (only printed with verbose=True), so this uses
    the single fact_score float per call, run once per SOAP sentence.
    """

    def __init__(self, threshold=None):
        self._factsumm = FactSumm()
        self._threshold = THRESHOLD if threshold is None else threshold

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        errors = []
        for sentence in split_sentences(soap_note):
            result = self._factsumm(transcript, sentence, verbose=False)
            score = result.get("fact_score")
            if score is not None and score < self._threshold:
                errors.append(("hallucination", sentence))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed
