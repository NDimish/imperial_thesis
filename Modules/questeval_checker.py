import time

from questeval.questeval_metric import QuestEval

from Modules.base import CheckerModule, split_sentences

# Uncalibrated on this data -- tune once you see real scores.
THRESHOLD = 0.5


class QuestEvalChecker(CheckerModule):
    """Flags hallucinated SOAP sentences using QuestEval's blended QG/QA score.

    Hallucination-only for now: QuestEval computes both precision and recall
    internally, but the public API (corpus_questeval, open_log_from_text) only
    exposes one blended score per pair and doesn't tie generated questions back
    to specific sentences -- checked directly against the source, no sentence-span
    or "unanswerable" data is actually returned. So this can't cleanly separate
    hallucination from omission; treated as a single consistency-style signal.
    """

    def __init__(self, threshold=None):
        self._questeval = QuestEval(no_cuda=True)
        self._threshold = THRESHOLD if threshold is None else threshold

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        errors = []
        sentences = split_sentences(soap_note)
        if sentences:
            result = self._questeval.corpus_questeval(hypothesis=sentences, sources=[transcript] * len(sentences))
            for sentence, score in zip(sentences, result["ex_level_scores"]):
                if score < self._threshold:
                    errors.append(("hallucination", sentence))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed
