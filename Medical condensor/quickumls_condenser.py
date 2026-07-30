import os
import sys
import time

from quickumls import QuickUMLS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import CondenserModule, join_turns, split_turns

# Set this to your local QuickUMLS install directory. Requires a UMLS Metathesaurus
# license from NLM and a local build first:
#   1. Register for a UMLS license: https://www.nlm.nih.gov/research/umls/
#   2. Download a UMLS release and install it with MetamorphoSys
#   3. python -m quickumls.install <umls_install_dir> <this_destination_dir>
# See: https://github.com/Georgetown-IR-Lab/QuickUMLS
QUICKUMLS_INSTALL_DIR = r"C:\Users\natha\OneDrive\Documents\Uni\Impreial\modules\Thesis\code\data\quickumls_install"

# QuickUMLS defaults (threshold=0.7, min_match_length=3, ~27 accepted semtypes)
# are too permissive on spoken dialogue -- short common words (e.g. "hi", "sir")
# fuzzy-match spurious UMLS concepts at 0.7 Jaccard similarity, so almost every
# turn (including pure greetings/sign-offs) was being kept as "clinical".
# Tightened here: higher similarity threshold, longer minimum match length, and
# a narrower set of semantic types focused on symptoms/findings/procedures/drugs
# (dropping broader/more abstract types like Mental Process or Intellectual
# Product that were the likely source of spurious matches).
THRESHOLD = 0.85
MIN_MATCH_LENGTH = 4
ACCEPTED_SEMTYPES = {
    "T023",  # Body Part, Organ, or Organ Component
    "T029",  # Body Location or Region
    "T031",  # Body Substance
    "T033",  # Finding
    "T034",  # Laboratory or Test Result
    "T037",  # Injury or Poisoning
    "T046",  # Pathologic Function
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction
    "T059",  # Laboratory Procedure
    "T060",  # Diagnostic Procedure
    "T061",  # Therapeutic or Preventive Procedure
    "T121",  # Pharmacologic Substance
    "T195",  # Antibiotic
    "T200",  # Clinical Drug
}

# UMLS contains ordinary English words as literal clinical terminology --
# e.g. "start" is a real T061 (Therapeutic/Preventive Procedure) concept and
# "well" is a real T033 (Finding) concept, both at similarity=1.0 (an exact
# match, so no THRESHOLD setting above can ever filter them out). Confirmed by
# directly inspecting match output on greeting/sign-off turns that kept
# surviving despite the semtype/threshold tightening above. Denylist these by
# their matched surface form rather than trying to threshold-tune them away.
GENERIC_WORD_DENYLIST = {"start", "well", "good", "fine", "right", "help"}


class QuickUMLSCondenser(CondenserModule):
    """Removes non-clinical filler from a transcript using QuickUMLS concept matching.

    Not usable until QUICKUMLS_INSTALL_DIR points at a local QuickUMLS install --
    raises immediately on construction if it isn't configured.
    """

    def __init__(self, quickumls_install_dir=None):
        install_dir = quickumls_install_dir or QUICKUMLS_INSTALL_DIR
        if not install_dir:
            raise RuntimeError(
                "QuickUMLS is not configured. Set QUICKUMLS_INSTALL_DIR in "
                "quickumls_condenser.py to your local QuickUMLS install directory "
                "(requires a UMLS license -- see Georgetown-IR-Lab/QuickUMLS)."
            )
        self._matcher = QuickUMLS(
            install_dir,
            threshold=THRESHOLD,
            min_match_length=MIN_MATCH_LENGTH,
            accepted_semtypes=ACCEPTED_SEMTYPES,
        )

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
        matches = self._matcher.match(text, best_match=True, ignore_syntax=False)
        return any(
            m["term"].lower() not in GENERIC_WORD_DENYLIST
            for group in matches
            for m in group
        )
