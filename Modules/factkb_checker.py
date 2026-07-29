import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Modules.base import CheckerModule, split_sentences

MODEL = "bunsenfeng/FactKB"
BASE_TOKENIZER = "roberta-base"
# Uncalibrated on this data -- tune once you see real scores.
THRESHOLD = 0.5


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
