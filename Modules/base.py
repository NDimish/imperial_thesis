import re
from abc import ABC, abstractmethod

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


class CheckerModule(ABC):
    """Base class for all error-checking modules."""

    @abstractmethod
    def check(self, transcript, soap_note):
        """Compares a transcript against a SOAP note.

        Returns (((type, error), (type, error), ...), elapsed).
        """
        raise NotImplementedError


def split_sentences(text):
    """Splits text into sentences on ./!/? boundaries."""
    return [s for s in SENTENCE_SPLIT_PATTERN.split(text.strip()) if s]
