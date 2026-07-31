import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import CondenserModule, join_turns, split_turns
from umls_matching import get_matcher, has_real_concept

# See umls_matching.py for the tuning history (threshold, semtypes, and the
# wordfreq-based common-word filter that replaced an earlier hand-typed
# denylist -- kept there since QuickUMLS/threshold tuning is shared with
# MedspacyCondenser and NegspacyCondenser now, not specific to this file.


class QuickUMLSCondenser(CondenserModule):
    """Removes non-clinical filler from a transcript using QuickUMLS concept matching.

    Not usable until QUICKUMLS_INSTALL_DIR (in umls_matching.py) points at a
    local QuickUMLS install -- raises immediately on construction if it isn't
    configured.
    """

    def __init__(self, quickumls_install_dir=None):
        self._matcher = get_matcher(quickumls_install_dir)

    def condense(self, transcript):
        start = time.perf_counter()

        # Tried keeping the turn right after a kept clinical question
        # regardless of its own concept match ("Yep.", "It comes and goes."
        # often has no UMLS concept of its own but is part of the same
        # clinical exchange) -- it did improve the KDE-based groundedness
        # score a lot. But that turned out to be exploiting a bias in that
        # metric, not a real improvement: a manual, clinically-complete
        # condensation of real transcripts (see kdbe_check.check_omissions'
        # docstring) proved the KDE score penalizes condensing almost in
        # proportion to how much text survives, regardless of content
        # quality. Under check_omissions_cosine (validated to track content
        # quality instead), this condenser's real coverage of the SOAP note
        # barely changes with or without the lookback (diff_cosine_coverage
        # -0.01 either way) -- so the lookback wasn't recovering real content,
        # it was just keeping more text. Removed, since it only cost
        # condensing aggressiveness (16.3% words reduced without it vs 7.9%
        # with it) for no real coverage benefit.
        turns = split_turns(transcript)
        kept = [(speaker, text) for speaker, text in turns if self._is_clinical(text)]
        condensed = join_turns(kept)

        elapsed = time.perf_counter() - start
        return condensed, elapsed

    def _is_clinical(self, text):
        return has_real_concept(text)
