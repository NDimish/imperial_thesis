import os
import sys
import time

import spacy
from negspacy.negation import Negex  # noqa: F401 -- registers the "negex" spaCy factory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base import CondenserModule, join_turns, split_turns
from umls_matching import has_real_concept

# negspacy only classifies negation for entities an upstream NER component already
# found -- it has no concept detection of its own. Pairing it with plain
# en_core_web_sm would give no clinical entities to negate, so this uses scispacy's
# biomedical NER as the upstream recognizer.
MODEL = "en_core_sci_sm"


class NegspacyCondenser(CondenserModule):
    """Removes non-clinical filler using scispacy NER + negspacy negation detection,
    cross-checked against the shared UMLS matcher (see umls_matching.py).

    Keeps a turn only if it contains at least one clinical entity that is NOT
    negated (e.g. keeps "chest pain", drops a turn whose only clinical mention is
    "no chest pain") -- a stricter condensation than SciSpacyCondenser, which keeps
    any turn with a clinical mention regardless of negation.

    en_core_sci_sm exposes only one flat entity label ("ENTITY") with no semantic
    type to filter on, and was confirmed (directly, on prim28.txt) tagging plain
    discourse markers -- "Hi", "I", "OK", "years", "birth" -- as entities. A
    hand-typed denylist for these doesn't scale (every new file surfaces new
    junk words), so each surviving non-negated entity is now also required to
    be a real UMLS concept via the shared matcher, instead of just "any entity
    spaCy happened to tag".
    """

    def __init__(self, model=None):
        self._nlp = spacy.load(model or MODEL)
        self._nlp.add_pipe("negex")

    def condense(self, transcript):
        start = time.perf_counter()

        # A "keep the turn after a kept question" lookback was tried here
        # too (see quickumls_condenser.py's condense() for the full story)
        # and reverted for the same reason: it improved the KDE-based
        # groundedness score a lot, but that score turned out to penalize
        # condensing almost regardless of content quality. Under
        # check_omissions_cosine (validated to actually track content
        # quality), this condenser's real coverage of the SOAP note was
        # about the same with or without it, so the lookback was only
        # costing condensing aggressiveness (32.7% words reduced without it
        # vs 24.5% with it) for no real coverage benefit.
        turns = split_turns(transcript)
        kept = [(speaker, text) for speaker, text in turns if self._has_affirmed_entity(text)]
        condensed = join_turns(kept)

        elapsed = time.perf_counter() - start
        return condensed, elapsed

    def _has_affirmed_entity(self, text):
        if not text.strip():
            return False

        # Tried two fixes for a confirmed negex scope-bleeding bug (found via
        # full-dataset audit): re-parsing each sentence in isolation (so
        # negation can't cross a full stop) was neutral (-9.00 -> -8.98);
        # additionally stripping a leading response particle ("No, I feel
        # ...tired." -> "I feel...tired.") went the other way (-9.00 ->
        # -9.22), because this dataset also uses "No," as genuine ELLIPTICAL
        # negation ("No, headaches or dizziness.") just as often -- stripping
        # it removed the only negation trigger in those turns, flipping
        # correct exclusions into wrong inclusions. Neither beat the
        # unmodified pipeline below, so both were reverted.
        doc = self._nlp(text)
        real_ents = [ent for ent in doc.ents if has_real_concept(ent.text)]
        if not real_ents:
            return False

        # negex's rules were built for negation in declarative clinical prose
        # ("denies any pain"), not spoken questions. A full-dataset audit
        # found the same mismatch as MedspacyCondenser's ConText fix: "any
        # other symptoms?", "any allergies to any medication?" get flagged
        # negex=True purely because "any" is a negation trigger word, even
        # though asking about a symptom isn't a claim that it's absent. Kept
        # on concept presence alone for questions; same syntactic (not
        # word-list) distinction as the Medspacy fix.
        if text.strip().endswith("?"):
            return True

        return any(not ent._.negex for ent in real_ents)
