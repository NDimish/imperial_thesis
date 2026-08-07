import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Modules.base import CheckerModule, split_sentences

MODEL = "bunsenfeng/FactKB"
BASE_TOKENIZER = "roberta-base"
# Was uncalibrated (0.5, the untuned default). A real 5-file threshold sweep,
# run after fixing Modules/base.py's sentence splitter to drop <4-word
# fragments (short turn-fragments like "d: Okay." were guaranteed false
# positives with no real content to be hallucinated or omitted), found
# threshold=0.9 gives the best F1 in the swept range (0.197 vs 0.172 at the
# old 0.5, plus better recall: 0.50 vs 0.42). The full swept range (0.1-0.99)
# only moved F1 between 0.17-0.20 -- a shallow, not sharply-peaked curve, so
# this is a real but modest improvement, not a dramatic recalibration like
# AlignScore/SummaC/Kdbe needed.
THRESHOLD = 0.9


class FactKBChecker(CheckerModule):
    """Flags hallucinated SOAP sentences using FactKB's factuality classifier.

    Hallucination-only: FactKB is trained purely as a (summary, article)
    consistency classifier, with no omission/coverage capability at all.
    """

    def __init__(self, threshold=None):
        self._tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
        self._model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
        self._model.eval()
        self._threshold = THRESHOLD if threshold is None else threshold

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        errors = []
        for sentence in split_sentences(soap_note):
            tokens = self._tokenizer(
                [[sentence, transcript]], return_tensors="pt", padding="max_length", truncation=True
            )
            with torch.no_grad():
                logits = self._model(**tokens).logits
            score = torch.softmax(logits, dim=1)[0][1].item()
            if score < self._threshold:
                errors.append(("hallucination", sentence))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed
