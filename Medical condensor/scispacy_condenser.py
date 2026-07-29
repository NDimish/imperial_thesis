import os
import sys
import time

import spacy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import CondenserModule, join_turns, split_turns

MODEL = "en_core_sci_sm"


class SciSpacyCondenser(CondenserModule):
    """Removes non-clinical filler from a transcript using scispacy's pretrained biomedical NER."""

    def __init__(self, model=None):
        self._nlp = spacy.load(model or MODEL)

    def condense(self, transcript):
        start = time.perf_counter()

        turns = split_turns(transcript)
        kept = [(speaker, text) for speaker, text in turns if self._is_clinical(text)]
        condensed = join_turns(kept)

        elapsed = time.perf_counter() - start
        return condensed, elapsed

    def _is_clinical(self, text):
        if not text.strip():
            return False
        return len(self._nlp(text).ents) > 0
