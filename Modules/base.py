import re
from abc import ABC, abstractmethod

# Splits after ./!/? (original behavior) OR on any run of newlines, even with no
# preceding punctuation. Newline-only splitting was added after a direct audit
# (see Modules/medspacy_umls_checker.py's docstring for the full investigation)
# found that terse, unpunctuated lines -- this project's own turn-tagged
# transcripts ("d:"/"p:" lines) and its own terse SOAP-note fragments ("No SOB /
# chest pain" with no period before the next line) -- were being merged into one
# oversized "sentence" by punctuation-only splitting. That merging let a doctor's
# question survive into the patient's next-line answer, and let one clinical
# fragment's negation bleed into an unrelated adjacent one -- confirmed directly
# on a real SOAP note: "No SOB / chest pain" wrongly negating the very next,
# unrelated line "Feeling tired, weak and myalgia" once fed to a checker as a
# single merged unit. Splitting on newlines too keeps each turn/line the
# independent unit it actually is.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


class CheckerModule(ABC):
    """Base class for all error-checking modules."""

    @abstractmethod
    def check(self, transcript, soap_note):
        """Compares a transcript against a SOAP note.

        Returns (((type, error), (type, error), ...), elapsed).
        """
        raise NotImplementedError


# A raw regex sentence split on disfluent, turn-tagged transcript text produces a
# lot of near-empty fragments -- "d: Okay.", "Yeah.", "d: Fine." -- each becomes
# its own candidate claim to every checker below. These carry no checkable
# clinical content, so a real-label run always finds them "unsupported" by the
# other document (there's nothing in them to support), and every checker flags
# them as a false-positive omission/hallucination. Confirmed directly: a 5-file
# real-label run showed AlignScoreChecker at precision=0.04 (5 TP / 128 FP) and
# KdbeChecker at precision=0.03 (5 TP / 142 FP), with a large fraction of the
# false positives being exactly these short fragments. Filtering them out
# before they ever reach a checker is the same fix this project already
# validated for the NLP condensers (a turn with no real content shouldn't be
# treated as content to check), just applied on the checker side instead.
MIN_SENTENCE_WORDS = 4


def split_sentences(text):
    """Splits text into sentences on ./!/? boundaries, dropping fragments under
    MIN_SENTENCE_WORDS words (see the comment above) -- too short to carry any
    checkable clinical content, so keeping them only produces guaranteed false
    positives in every checker that uses this."""
    return [
        s for s in SENTENCE_SPLIT_PATTERN.split(text.strip())
        if s and len(s.split()) >= MIN_SENTENCE_WORDS
    ]
