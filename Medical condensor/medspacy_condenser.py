import os
import sys
import time

from loguru import logger

# medspacy's PyRuSH sentencizer logs at DEBUG by default, which floods the
# terminal/log files with per-token sentence-boundary traces. Quiet it down.
logger.remove()
logger.add(sys.stderr, level="WARNING")

import medspacy
from medspacy.ner import TargetRule

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import CondenserModule, join_turns, split_turns

# medspacy ships no default concept list — its target matcher is meant to be seeded
# with your own rules. This is a compact general clinical-relevance list, not a
# clinical knowledge base.
CLINICAL_TARGET_RULES = [
    TargetRule("pain", "SYMPTOM"),
    TargetRule("fever", "SYMPTOM"),
    TargetRule("nausea", "SYMPTOM"),
    TargetRule("vomiting", "SYMPTOM"),
    TargetRule("diarrhea", "SYMPTOM"),
    TargetRule("cough", "SYMPTOM"),
    TargetRule("rash", "SYMPTOM"),
    TargetRule("swelling", "SYMPTOM"),
    TargetRule("bleeding", "SYMPTOM"),
    TargetRule("headache", "SYMPTOM"),
    TargetRule("shortness of breath", "SYMPTOM"),
    TargetRule("chest pain", "SYMPTOM"),
    TargetRule("dizziness", "SYMPTOM"),
    TargetRule("fatigue", "SYMPTOM"),
    TargetRule("weight loss", "SYMPTOM"),
    TargetRule("asthma", "CONDITION"),
    TargetRule("diabetes", "CONDITION"),
    TargetRule("hypertension", "CONDITION"),
    TargetRule("eczema", "CONDITION"),
    TargetRule("infection", "CONDITION"),
    TargetRule("medication", "MEDICATION"),
    TargetRule("tablet", "MEDICATION"),
    TargetRule("allergy", "PROBLEM"),
    TargetRule("blood pressure", "VITAL"),
    TargetRule("temperature", "VITAL"),
]


class MedspacyCondenser(CondenserModule):
    """Removes non-clinical filler from a transcript using medspacy's rule-based target matcher."""

    def __init__(self):
        self._nlp = medspacy.load()
        target_matcher = self._nlp.get_pipe("medspacy_target_matcher")
        target_matcher.add(CLINICAL_TARGET_RULES)

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
