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

    Returns a dict with a coverage-style score in each direction. Each score is the
    *best-case* coverage among that side's words -- how well even the single most-
    covered word is explained by the other text's distribution:
      - "omission_transcript_to_soap": best-case coverage of the transcript's words
        by the SOAP's distribution. LOW means even the best-matching transcript word
        wasn't well covered -- strong evidence the SOAP omits transcript content.
        HIGH is a weak/ambiguous signal -- it only means one word matched well, and
        says nothing about the rest. This is the genuine omission-direction signal.
      - "groundedness_soap_in_transcript": best-case coverage of the SOAP's words by
        the transcript's distribution -- same logic, mirrored. Deliberately NOT named
        "omission_soap_to_transcript" (an earlier version of this function used that
        name) -- a low score here means SOAP content isn't grounded in the transcript,
        which is a hallucination-style signal, not an omission one. Keeping the two
        directions distinctly named avoids a downstream reader assuming both measure
        the same thing just mirrored.

    LOWER scores are the stronger/more confident signal in both directions (matches
    the original Embed2KDE repo's own calibration: predicted_omission = score <
    threshold). Both scores are None if either text doesn't have enough recognized
    words to fit a density estimate. There is no validated threshold for "yes/no
    omission" on this embedding space -- these are relative scores, not calibrated
    probabilities.

    Note on methodology vs. Modules/kdbe_checker.py's per-sentence version: that
    version fits scaler/PCA on the *reference document only*, since it needs one
    stable projection space to test many candidate sentences against efficiently.
    This whole-document version fits on the *concatenation of both texts* instead,
    since it's a one-shot symmetric comparison and that's how the original Embed2KDE
    repo (which this was ported from) did it. Both choices are deliberate for their
    own context -- scores from the two files aren't on directly comparable scales.
    """
    xi = _embed_words(text_i)
    xo = _embed_words(text_o)

    if len(xi) < 6 or len(xo) < 6:
        return {"omission_transcript_to_soap": None, "groundedness_soap_in_transcript": None}

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
    groundedness_soap_in_transcript = float((xo_density_under_i / min_kde_i).max())

    return {
        "omission_transcript_to_soap": omission_transcript_to_soap,
        "groundedness_soap_in_transcript": groundedness_soap_in_transcript,
    }


def check_omissions_cosine(text_i, text_o):
    """Nearest-neighbor cosine-similarity alternative to check_omissions, added
    after check_omissions was found to have a severe length bias: a genuinely
    excellent, clinically-complete manual condensation (real transcripts,
    real SOAP notes) that removed ~45-60% of words scored FAR worse
    (avg diff -20.00 across 10 files) than several of this project's
    automated condensers, purely because check_omissions refits a KDE from
    scratch on however many words survive -- fewer points structurally
    yields lower density estimates at any given query point, independent of
    whether the removed words were meaningless filler or vital content.
    Confirmed via controlled tests: fixing the embedding projection space
    didn't help (same result to 3 decimal places); neither did replacing the
    normalizing floor with a percentile instead of a strict minimum (same
    qualitative pattern at every percentile). The bias is intrinsic to
    density estimation over small point clouds, not a fixable implementation
    detail of check_omissions.

    This function sidesteps that entirely: no density estimate is fit at
    all. For each word in text_o (typically the SOAP note), it finds that
    word's single most similar word in text_i (typically the transcript or
    condensed transcript) by cosine similarity, then averages those
    best-match scores across every word in text_o. A "best match" for a
    given SOAP word exists as long as at least one semantically similar word
    survives anywhere in text_i -- the size of text_i otherwise barely
    matters, which is exactly the property check_omissions lacked.

    Validated on the same real 10-file test: the manual condensation's
    average coverage barely moved (0.8360 -> 0.8265, -0.0095), while random
    word deletion at the SAME retention percentage dropped much further
    (0.8360 -> 0.7800, -0.0560) -- in every one of the 10 files, not just on
    average. That's the behavior a coverage metric should have: content-aware
    condensing should cost little, careless deletion should cost more, and
    neither should be dominated by how much text is left.

    Returns a dict with "cosine_coverage" -- UNLIKE check_omissions, this is
    a genuine coverage score where HIGHER is better (not an inverted
    omission signal), or {"cosine_coverage": None} if either text has too
    few recognized words to compare. Since removing words can only hold
    coverage steady or reduce it, 0 is the realistic ceiling for a
    condensed-vs-original diff, not some positive target -- see
    check_omissions_bidirectional_cosine for the precision-side counterpart.
    """
    xi = _embed_words(text_i)
    xo = _embed_words(text_o)
    if len(xi) < 3 or len(xo) < 3:
        return {"cosine_coverage": None}

    xi_vectors = xi.to_numpy()
    xo_vectors = xo.to_numpy()
    xi_unit = xi_vectors / np.linalg.norm(xi_vectors, axis=1, keepdims=True)
    xo_unit = xo_vectors / np.linalg.norm(xo_vectors, axis=1, keepdims=True)

    similarity_matrix = xo_unit @ xi_unit.T
    best_match_per_word = similarity_matrix.max(axis=1)

    return {"cosine_coverage": float(best_match_per_word.mean())}


def check_omissions_bidirectional_cosine(transcript, soap_ground):
    """cosine_coverage only checks recall: is every SOAP word matched
    somewhere in the transcript? It can't tell a condenser that keeps every
    SOAP-relevant word AND a pile of unrelated junk apart from one that keeps
    only the relevant words -- both would score the same cosine_coverage.

    This adds the mirror direction -- precision: for each word in the
    TRANSCRIPT, find its best cosine match anywhere in the SOAP note, and
    average those best-match scores. A condenser that keeps a lot of content
    unrelated to anything in the SOAP note (the transcript's own filler,
    tangents, admin chatter) will show a LOWER precision even if its
    coverage/recall is perfect, because that filler's best match in the SOAP
    note's vocabulary is a poor one. Combines the two into an F1-style
    harmonic mean, same idea as ROUGE/BERTScore precision-recall-F1 triples.

    Returns a dict with "cosine_recall" (same value as check_omissions_cosine's
    "cosine_coverage", recomputed here for a single shared similarity matrix),
    "cosine_precision", and "cosine_f1". All three are higher-is-better, and
    all three have 0 as the ceiling for a condensed-vs-original diff, for the
    same reason as cosine_coverage. Returns all None if either text has too
    few recognized words to compare.
    """
    xi = _embed_words(transcript)
    xo = _embed_words(soap_ground)
    if len(xi) < 3 or len(xo) < 3:
        return {"cosine_recall": None, "cosine_precision": None, "cosine_f1": None}

    xi_vectors = xi.to_numpy()
    xo_vectors = xo.to_numpy()
    xi_unit = xi_vectors / np.linalg.norm(xi_vectors, axis=1, keepdims=True)
    xo_unit = xo_vectors / np.linalg.norm(xo_vectors, axis=1, keepdims=True)

    similarity_matrix = xo_unit @ xi_unit.T  # (n_soap_words, n_transcript_words)

    recall = float(similarity_matrix.max(axis=1).mean())  # each SOAP word's best transcript match
    precision = float(similarity_matrix.max(axis=0).mean())  # each transcript word's best SOAP match
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"cosine_recall": recall, "cosine_precision": precision, "cosine_f1": float(f1)}


def check_omissions_rouge1(transcript, soap_ground):
    """Pure lexical (no embeddings at all) cross-check on the cosine metrics
    above: standard multiset-overlap ROUGE-1 between the transcript and the
    SOAP note, using the same word tokenization as the rest of this module
    (lowercased, alphabetic, length > 2) for consistency with the other
    scores here.

    This exists to answer a specific question: are the cosine-based
    conclusions in this project an artifact of GloVe's particular embedding
    geometry, or does a completely different, decades-old, embedding-free
    method tell the same story? ROUGE-1 shares no machinery with cosine_*
    or check_omissions -- no vectors, no learned representations, just exact
    word-overlap counts -- so agreement between it and the cosine metrics is
    real independent triangulation, not two views of the same computation.

    recall = how much of the SOAP note's words appear in the transcript
    (the direct lexical analogue of cosine_recall/cosine_coverage).
    precision = how much of the transcript's words also appear in the SOAP
    note (the lexical analogue of cosine_precision). f1 is their harmonic
    mean. All are higher-is-better with 0 as the diff ceiling, same as the
    cosine metrics, and for the same reason (removing words can't manufacture
    new overlap).
    """
    from collections import Counter

    transcript_counts = Counter(_clean_words(transcript))
    soap_counts = Counter(_clean_words(soap_ground))

    total_soap_words = sum(soap_counts.values())
    total_transcript_words = sum(transcript_counts.values())
    if total_soap_words == 0 or total_transcript_words == 0:
        return {"rouge1_recall": None, "rouge1_precision": None, "rouge1_f1": None}

    overlap = sum(min(count, transcript_counts.get(word, 0)) for word, count in soap_counts.items())

    recall = overlap / total_soap_words
    precision = overlap / total_transcript_words
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"rouge1_recall": recall, "rouge1_precision": precision, "rouge1_f1": f1}
