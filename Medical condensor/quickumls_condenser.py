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
        self._matcher = QuickUMLS(install_dir)

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
        return len(self._matcher.match(text, best_match=True, ignore_syntax=False)) > 0
