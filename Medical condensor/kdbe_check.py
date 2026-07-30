"""KDE-over-static-embeddings omission check, adapted from Embed2KDE
(achok7893/EmbedKDECheck_hallucination_detection).

The original repo's static-embeddings backend is hardcoded to local French
word2vec/fastText pickle files that aren't shipped or downloadable, and its text
cleaning drops every word <=5 characters and filters with French stopwords on
English text. This keeps the same KDE math (which is embedding-agnostic) but
swaps in English GloVe vectors pulled on demand via gensim, and fixes the
preprocessing for English.
"""

import re

import gensim.downloader as gensim_api
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

# Static word embeddings (GloVe, 100-dim, ~128MB download, cached by gensim after
# the first call). Swap for a larger gensim model (e.g. "word2vec-google-news-300")
# for higher-quality vectors at the cost of a much bigger download.
EMBEDDING_MODEL_NAME = "glove-wiki-gigaword-100"

_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = gensim_api.load(EMBEDDING_MODEL_NAME)
    return _embeddings


def _clean_words(text):
    """Lowercases and strips to alphabetic tokens longer than 2 characters."""
    return [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if len(w) > 2]


def _embed_words(text):
    """Returns a DataFrame with one static embedding vector per recognized word."""
    embeddings = _get_embeddings()
    vectors = [embeddings[word] for word in _clean_words(text) if word in embeddings]
    return pd.DataFrame(vectors)


def _select_bandwidth(x):
    bandwidths = 10 ** np.linspace(-1, 1, 20)
    n_splits = min(5, len(x))
    grid = GridSearchCV(KernelDensity(), {"bandwidth": bandwidths}, cv=n_splits)
    grid.fit(x)
    return grid.best_params_["bandwidth"]


def _max_min_density(x, kde):
    scores = np.exp(kde.score_samples(x))
    return scores.max(), scores.min()


def check_omissions(text_i, text_o, n_components=5):
    """Compares two texts via KDE over static word embeddings.

    text_i: the "input" text, treated as ground truth (e.g. the condensed transcript)
    text_o: the "output" text being checked for completeness against it (e.g. the
        ground-truth SOAP note)

    Returns a dict with an omission-style score in each direction. Each score is the
    *best-case* coverage among that side's words -- how well even the single most-
    covered word is explained by the other text's distribution:
      - "omission_transcript_to_soap": best-case coverage of the transcript's words
        by the SOAP's distribution. LOW means even the best-matching transcript word
        wasn't well covered -- strong evidence the SOAP omits transcript content.
        HIGH is a weak/ambiguous signal -- it only means one word matched well, and
        says nothing about the rest.
      - "omission_soap_to_transcript": best-case coverage of the SOAP's words by the
        transcript's distribution -- same logic, mirrored (a low score here points
        at SOAP content not grounded in the transcript, i.e. hallucination).

    LOWER scores are the stronger/more confident omission signal (matches the
    original Embed2KDE repo's own calibration: predicted_omission = score < threshold).
    Both scores are None if either text doesn't have enough recognized words to fit
    a density estimate. There is no validated threshold for "yes/no omission" on this
    embedding space -- these are relative scores, not calibrated probabilities.
    """
    xi = _embed_words(text_i)
    xo = _embed_words(text_o)

    if len(xi) < 6 or len(xo) < 6:
        return {"omission_transcript_to_soap": None, "omission_soap_to_transcript": None}

    scaler = StandardScaler().fit(pd.concat([xi, xo], axis=0))
    xi_scaled = pd.DataFrame(scaler.transform(xi))
    xo_scaled = pd.DataFrame(scaler.transform(xo))

    n_components = min(n_components, xi_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=0).fit(pd.concat([xi_scaled, xo_scaled], axis=0))
    xi_pca = pd.DataFrame(pca.transform(xi_scaled))
    xo_pca = pd.DataFrame(pca.transform(xo_scaled))

    bandwidth = min(_select_bandwidth(xi_pca), _select_bandwidth(xo_pca))

    kde_i = KernelDensity(bandwidth=bandwidth).fit(xi_pca)
    kde_o = KernelDensity(bandwidth=bandwidth).fit(xo_pca)

    _, min_kde_i = _max_min_density(xi_pca, kde_i)
    _, min_kde_o = _max_min_density(xo_pca, kde_o)

    xi_density_under_o = np.exp(kde_o.score_samples(xi_pca))
    omission_transcript_to_soap = float((xi_density_under_o / min_kde_o).max())

    xo_density_under_i = np.exp(kde_i.score_samples(xo_pca))
    omission_soap_to_transcript = float((xo_density_under_i / min_kde_i).max())

    return {
        "omission_transcript_to_soap": omission_transcript_to_soap,
        "omission_soap_to_transcript": omission_soap_to_transcript,
    }
