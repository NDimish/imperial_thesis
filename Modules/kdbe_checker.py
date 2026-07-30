"""Per-sentence KDE-over-static-embeddings checker, adapted from Embed2KDE
(achok7893/EmbedKDECheck_hallucination_detection) -- see Medical condensor/kdbe_check.py
for the whole-document version this was originally built from.

Unlike that whole-document version (one omission-style score per file), this fits a
KDE on the *reference* document once per direction, then scores each candidate
sentence from the other document against it individually -- giving flaggable
(type, sentence) pairs like the other 5 checker modules, instead of a bare score.
"""

import re
import time

import gensim.downloader as gensim_api
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from Modules.base import CheckerModule, split_sentences

EMBEDDING_MODEL_NAME = "glove-wiki-gigaword-100"
# Ratio of a sentence's density under the reference KDE to the reference's own
# weakest self-density point. < 1 means the sentence's words are literally less
# typical than anything already in the reference document -- a grounded (if still
# uncalibrated) cutoff, unlike an arbitrary 0-1 probability threshold.
THRESHOLD = 1.0
N_COMPONENTS = 5

_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = gensim_api.load(EMBEDDING_MODEL_NAME)
    return _embeddings


def _clean_words(text):
    return [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if len(w) > 2]


def _embed_words(text):
    embeddings = _get_embeddings()
    return [embeddings[word] for word in _clean_words(text) if word in embeddings]


class KdbeChecker(CheckerModule):
    """Flags hallucinated SOAP sentences and omitted transcript sentences by scoring
    each sentence's word embeddings against a KDE fit on the *other* full document.
    """

    def __init__(self, threshold=None, n_components=None):
        self._threshold = THRESHOLD if threshold is None else threshold
        self._n_components = N_COMPONENTS if n_components is None else n_components

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        errors = []
        errors.extend(self._flag(candidates=split_sentences(soap_note), reference=transcript, error_type="hallucination"))
        errors.extend(self._flag(candidates=split_sentences(transcript), reference=soap_note, error_type="omission"))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    def _flag(self, candidates, reference, error_type):
        if not candidates:
            return []

        reference_vectors = _embed_words(reference)
        if len(reference_vectors) < 6:
            return []

        n_components = min(self._n_components, len(reference_vectors), len(reference_vectors[0]))

        scaler = StandardScaler().fit(reference_vectors)
        reference_scaled = scaler.transform(reference_vectors)

        pca = PCA(n_components=n_components, random_state=0).fit(reference_scaled)
        reference_pca = pca.transform(reference_scaled)

        bandwidth = self._select_bandwidth(reference_pca)
        kde = KernelDensity(bandwidth=bandwidth).fit(reference_pca)
        min_self_density = np.exp(kde.score_samples(reference_pca)).min()

        flagged = []
        for sentence in candidates:
            sentence_vectors = _embed_words(sentence)
            if not sentence_vectors:
                continue

            sentence_scaled = scaler.transform(sentence_vectors)
            sentence_pca = pca.transform(sentence_scaled)
            density = np.exp(kde.score_samples(sentence_pca))
            # mean, not max: we want "does this sentence overall fit", not "does its
            # single best word fit" (max would almost never flag anything).
            ratio = float((density / min_self_density).mean())

            if ratio < self._threshold:
                flagged.append((error_type, sentence))

        return flagged

    @staticmethod
    def _select_bandwidth(x):
        bandwidths = 10 ** np.linspace(-1, 1, 20)
        n_splits = min(5, len(x))
        grid = GridSearchCV(KernelDensity(), {"bandwidth": bandwidths}, cv=n_splits)
        grid.fit(x)
        return grid.best_params_["bandwidth"]
