import os
import sys
import time

import spacy
from negspacy.negation import Negex  # noqa: F401 -- registers the "negex" spaCy factory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import CondenserModule, is_generic_entity, join_turns, split_turns

# negspacy only classifies negation for entities an upstream NER component already
# found -- it has no concept detection of its own. Pairing it with plain
# en_core_web_sm would give no clinical entities to negate, so this uses scispacy's
# biomedical NER as the upstream recognizer.
MODEL = "en_core_sci_sm"


class NegspacyCondenser(CondenserModule):
    """Removes non-clinical filler using scispacy NER + negspacy negation detection.

    Keeps a turn only if it contains at least one clinical entity that is NOT
    negated (e.g. keeps "chest pain", drops a turn whose only clinical mention is
    "no chest pain") -- a stricter condensation than SciSpacyCondenser, which keeps
    any turn with a clinical mention regardless of negation.
    """

    def __init__(self, model=None):
        self._nlp = spacy.load(model or MODEL)
        self._nlp.add_pipe("negex")

    def condense(self, transcript):
        start = time.perf_counter()

        turns = split_turns(transcript)
        kept = [(speaker, text) for speaker, text in turns if self._has_affirmed_entity(text)]
        condensed = join_turns(kept)

        elapsed = time.perf_counter() - start
        return condensed, elapsed

    def _has_affirmed_entity(self, text):
        if not text.strip():
            return False
        doc = self._nlp(text)
        return any(not ent._.negex and not is_generic_entity(ent.text) for ent in doc.ents)
