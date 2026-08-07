"""Per-sentence omission/hallucination checker based on EmbedKDECheck (Oukelmoun et al.,
2025, "Detecting Omissions in LLM-Generated Medical Summaries", EMNLP Industry Track --
github.com/achok7893/EmbedKDECheck_hallucination_detection).

The paper's method (one direction only -- omission): embed every word of the INPUT and
OUTPUT documents, fit a KDE on the OUTPUT's word embeddings, then score every INPUT word
under that KDE. Each word's omission score is

    OMscore = (min density over the OUTPUT's own words) / (that word's KDE density)

-- i.e. how much less typical a word is than the least-typical word already in the
reference document. A high OMscore means that word (and the source content around it)
isn't represented in the output. The paper's *global* omission score for a whole
document is the MAX token-level OMscore across it; tokens with scores near that max are
the ones "responsible" for the flagged omission.

This checker keeps that formula, generalized two ways to fit this project's
per-sentence CheckerModule interface (matching how every other checker here --
AlignScoreChecker, FactKBChecker, KdbeChecker -- works):
  - scored per SENTENCE, not per whole document: a candidate sentence's OMscore is an
    aggregate over its own words' OMscores (the per-sentence analogue of the paper's
    per-document max), flagged if that aggregate exceeds THRESHOLD.
  - run in BOTH directions, not just omission: a KDE fit on the transcript scores SOAP
    sentences for hallucination, and a KDE fit on the SOAP note scores transcript
    sentences for omission -- exactly the "extend to other forms of hallucination"
    the paper names as future work (Section 6).

Two further deviations from the paper's own choices, both made after real-file testing
showed the literal versions misbehaving:
  - Aggregation is MEAN, not the paper's own MAX. The raw OMscore range is extreme on
    this checker's word embeddings (roughly 1e-5 to 1e+34 -- a single word's KDE
    density can land numerically close to zero), and MAX let that one outlier word
    alone decide a whole sentence's score, which is what caused the over-flagging this
    checker went through during development (see explainer/embedkde_checker_report.html
    for that history). MEAN still weights every word, but no longer lets one word's
    density underflow dominate the sentence outright -- matching the choice
    kdbe_checker.py (this project's other, GloVe-based port of the same paper) already
    made for the same reason.
  - Words are LEMMATIZED before embedding (via spaCy's en_core_sci_sm, already a
    project dependency for the clinical domain), not just regex-split and
    length-filtered: "lives"/"living" collapse onto the same "live" embedding lookup,
    and stopwords are dropped entirely rather than counted as content words with no
    clinical signal of their own.
"""

import os
import time

import gensim.downloader as gensim_api
import numpy as np
import spacy
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from Modules.base import CheckerModule, split_sentences

# Clinical-domain spaCy model (scispaCy), already a project dependency -- used only for
# tagging + lemmatization here (parser/NER disabled: not needed and meaningfully slower).
LEMMATIZER_MODEL_NAME = "en_core_sci_sm"

# The paper's own FTW2V is FastText+Word2Vec fine-tuned on a 32M-word medical corpus
# from the collaborating hospital -- that fine-tuned model isn't available here. This
# uses the same FTW2V *architecture* (FastText and Word2Vec combined) instead of the
# GloVe substitute this project's other port of this paper/repo uses (kdbe_checker.py),
# built from the two public English pretrained gensim models rather than a from-scratch
# fine-tune. Each word's vector is the CONCATENATION of its FastText and Word2Vec
# vectors, not an average: the two models are trained independently, so their axes
# aren't aligned -- averaging would blend two unrelated coordinate systems into
# something with no clean meaning, whereas concatenation just widens the feature space
# and lets the PCA step already in this pipeline find the useful combined structure.
# Note: gensim's "fasttext-wiki-news-subwords-300" is the plain pretrained word-vector
# file, not the subword-capable .bin -- it does NOT synthesize vectors for
# out-of-vocabulary words, so it does not fix the "LLQ" OOV gap noted elsewhere in this
# project (Modules/old/evaluate.py's docstring); a word only gets embedded here if
# BOTH models recognize it.
FASTTEXT_MODEL_NAME = "fasttext-wiki-news-subwords-300"
WORD2VEC_MODEL_NAME = "word2vec-google-news-300"
# Real sweep, hallucination direction only, 10 real files (prim1/2/3/4/5/10/11/12/13/42)
# against their real labels -- 128 candidate sentences, 11 true hallucination labels among
# them. The raw OMscore range is extreme (roughly 1e-5 to 1e+34: a single word's density
# can get numerically close to zero, and max-aggregation lets that one word's score alone
# decide the whole sentence), so threshold=1.0 (the paper's literal boundary) and
# threshold=100 flag almost the same set -- tp stays flat at 9/11 (recall=0.818) all the
# way from threshold=10 up through threshold=100, since none of the true labels' own
# scores fall in that range. Past 100, tp starts dropping (8/11 at threshold=300). So
# threshold=100 is the top of that plateau: same recall as 1.0 (0.818, no cost), but fewer
# false positives (53 vs 65) and higher precision (0.145 vs 0.122) -- a strict improvement
# within the sample, not a tradeoff pick. Omission direction not evaluated at this
# threshold yet (out of scope for this pass). Still only 10 files / 11 labels -- re-sweep
# across more of the 57 before treating this as final, same caveat every other checker's
# initial calibration in this project carries.
THRESHOLD = 100.0
# Was 5, tuned in kdbe_checker.py for 100-dim GloVe vectors. FTW2V's concatenated
# FastText+Word2Vec vectors are 600-dim raw -- 6x wider -- so 5 components risks
# collapsing mostly onto whichever of the two embedding spaces happens to have more
# variance, losing the other's signal. Bumped to 8 as a reasoned guess (not measured):
# enough headroom to let both halves contribute, still far below where the paper's own
# text warns KDE reliability degrades in higher dimensions.
N_COMPONENTS = 8
MIN_REFERENCE_WORDS = 6
# See _leave_one_out_min_density's docstring: the low end of the reference's own
# leave-one-out self-density distribution, not the strict min.
SELF_DENSITY_PERCENTILE = 10

# gensim.downloader.load() parses the ORIGINAL word2vec/FastText text-or-binary format
# from scratch every time -- for word2vec-google-news-300 (3M words) that's a ~15-minute
# cold start, confirmed directly (918s cold vs. 5.64s once already loaded in the same
# process). Two things fix that:
#   1. LOAD_LIMIT: both files are frequency-ordered, so only parsing the first N words
#      keeps virtually all vocabulary real transcript/SOAP-note text ever hits, while
#      cutting parse work roughly in proportion to the vocab cut (3M -> 300K is a 10x
#      reduction for word2vec).
#   2. Caching the limited result in gensim's own native format (KeyedVectors.save()),
#      so every load after the first for a given LOAD_LIMIT uses KeyedVectors.load(...,
#      mmap="r") -- a memory-mapped file open, not a re-parse -- instead of paying the
#      text/binary parse cost again.
LOAD_LIMIT = 300_000

_fasttext = None
_word2vec = None


def _load_cached(name, binary):
    """Loads a gensim-downloader word-vector model, using a LOAD_LIMIT-capped,
    mmap-cached copy after the first call (see comment above)."""
    cache_dir = os.path.join(gensim_api.base_dir, name)
    cache_path = os.path.join(cache_dir, f"{name}-limit{LOAD_LIMIT}.kv")

    if os.path.exists(cache_path):
        return KeyedVectors.load(cache_path, mmap="r")

    source_path = gensim_api.load(name, return_path=True)
    model = KeyedVectors.load_word2vec_format(source_path, binary=binary, limit=LOAD_LIMIT)
    os.makedirs(cache_dir, exist_ok=True)
    model.save(cache_path)
    return model


def _get_embeddings():
    global _fasttext, _word2vec
    if _fasttext is None:
        _fasttext = _load_cached(FASTTEXT_MODEL_NAME, binary=False)
    if _word2vec is None:
        _word2vec = _load_cached(WORD2VEC_MODEL_NAME, binary=True)
    return _fasttext, _word2vec


_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(LEMMATIZER_MODEL_NAME, disable=["parser", "ner"])
    return _nlp


# spaCy's default English stopword list includes core negation terms ("no", "not",
# "never", "without", "nor", "neither") -- confirmed directly: without this carve-out,
# "no blood in stools" and "blood in stools" produced the IDENTICAL cleaned word list,
# erasing exactly the distinction that matters most for a clinical checker. Negation
# flips are a real, dangerous hallucination type this project's own label generator
# produces (_flip_negative() in datamakerfiles/prim_lib_injection_extra.py) -- silently
# losing that signal would be a correctness bug, not just lost recall.
NEGATION_LEMMAS = {"no", "not", "never", "without", "nor", "neither"}


def _clean_words(text):
    """Lemmatizes text and drops stopwords/non-alphabetic tokens (keeping negation
    terms regardless -- see NEGATION_LEMMAS above) -- collapses inflected forms
    ("lives"/"living") onto the same lemma ("live") before the embedding lookup, and
    removes function words that carry no clinical content of their own, instead of
    the previous plain regex-split + length>2 filter."""
    doc = _get_nlp()(text.lower())
    return [
        token.lemma_ for token in doc
        if token.is_alpha and (not token.is_stop or token.lemma_ in NEGATION_LEMMAS)
    ]


def _embed_words(text):
    """FTW2V vector per recognized word: FastText and Word2Vec vectors concatenated
    (see module comment for why concatenation, not averaging). A word only
    contributes if BOTH models recognize it -- zero-padding a missing half would
    cluster every such word artificially near the origin on that half, distorting
    the KDE, rather than just being absent."""
    fasttext, word2vec = _get_embeddings()
    return [
        np.concatenate([fasttext[word], word2vec[word]])
        for word in _clean_words(text)
        if word in fasttext and word in word2vec
    ]


class EmbedKdeChecker(CheckerModule):
    """Flags hallucinated SOAP sentences and omitted transcript sentences using
    EmbedKDECheck's own OMscore formula, mean-aggregated per sentence (see module
    docstring), applied in both directions.
    """

    def __init__(self, threshold=None, n_components=None):
        self._threshold = THRESHOLD if threshold is None else threshold
        self._n_components = N_COMPONENTS if n_components is None else n_components

    def check(self, transcript, soap_note):
        start = time.perf_counter()

        errors = []
        errors.extend(self._flag(split_sentences(soap_note), transcript, "hallucination"))
        errors.extend(self._flag(split_sentences(transcript), soap_note, "omission"))

        elapsed = time.perf_counter() - start
        return tuple(errors), elapsed

    def _flag(self, candidate_sentences, reference_text, error_type):
        """Scores each of candidate_sentences by the MEAN OMscore among its own words
        (see module docstring for why mean, not the paper's own max) against a KDE fit
        on reference_text's words, flagging (error_type, sentence) for every candidate
        whose score exceeds self._threshold."""
        if not candidate_sentences:
            return []

        reference_vectors = _embed_words(reference_text)
        if len(reference_vectors) < MIN_REFERENCE_WORDS:
            return []

        n_components = min(self._n_components, len(reference_vectors), len(reference_vectors[0]))

        scaler = StandardScaler().fit(reference_vectors)
        reference_scaled = scaler.transform(reference_vectors)

        pca = PCA(n_components=n_components, random_state=0).fit(reference_scaled)
        reference_pca = pca.transform(reference_scaled)

        bandwidth = self._select_bandwidth(reference_pca)
        kde = KernelDensity(bandwidth=bandwidth).fit(reference_pca)
        min_self_log_density = self._leave_one_out_min_log_density(reference_pca, bandwidth)

        flagged = []
        for sentence in candidate_sentences:
            word_vectors = _embed_words(sentence)
            if not word_vectors:
                continue

            words_scaled = scaler.transform(word_vectors)
            words_pca = pca.transform(words_scaled)
            log_densities = kde.score_samples(words_pca)

            # log(OMscore) = log(min_self_density / density) = min_self_log_density -
            # log_density -- computed as a subtraction in log-space, not exp() then
            # divide, so a word whose density underflows to exactly 0.0 in linear
            # space (confirmed happening on real files: a "divide by zero" warning,
            # then `inf` silently poisoning the MEAN below for the whole sentence --
            # exactly the one-word-dominates failure mode mean-aggregation was meant
            # to avoid) instead just produces a large but finite log-ratio. Clipped
            # before exponentiating so even a genuinely astronomical ratio can't
            # produce inf and poison the mean; 700 is just under float64's overflow
            # point (~709) for exp().
            log_om_scores = np.clip(min_self_log_density - log_densities, None, 700.0)
            sentence_score = float(np.mean(np.exp(log_om_scores)))

            if sentence_score > self._threshold:
                flagged.append((error_type, sentence))

        return flagged

    @staticmethod
    def _leave_one_out_min_log_density(points, bandwidth):
        """The reference's own "self-density" floor, scored leave-one-out: each
        reference point is scored under a KDE fit on every OTHER reference point, not
        itself. Scoring a point under a KDE that includes its own kernel inflates its
        density (that point's own contribution boosts it) -- an unseen candidate word
        never gets that same self-boost, so comparing candidate density against an
        in-sample min is systematically unfair to the candidate. Leaving each point out
        of its own fit puts the anchor on the same in-sample/out-of-sample footing as
        the candidate words actually being scored against it.

        Returns a LOG density (not exponentiated) -- see _flag's log-space comment for
        why: exponentiating individual leave-one-out density values here risked the
        same underflow-to-zero as the per-word densities in _flag.

        Uses the SELF_DENSITY_PERCENTILE-th percentile of those leave-one-out scores,
        not the strict min the paper's formula literally names ("min density over
        output words"): the paper's own reference sets are whole medical reports
        (hundreds of words), where a true minimum is a stable statistic, but a
        transcript/SOAP sentence here is a few dozen words at most -- small enough
        that a bare min is dominated by whichever single reference word happens to be
        the most unusual one (confirmed directly: with a strict min this flagged 0/94
        candidates on a real file, the opposite failure from the in-sample-inflated
        min it replaced, which flagged nearly everything). A low percentile keeps the
        "near the floor" intent without being hostage to one point.
        """
        n = len(points)
        self_log_densities = np.empty(n)
        for i in range(n):
            others = np.delete(points, i, axis=0)
            loo_kde = KernelDensity(bandwidth=bandwidth).fit(others)
            self_log_densities[i] = loo_kde.score_samples(points[i : i + 1])[0]
        return float(np.percentile(self_log_densities, SELF_DENSITY_PERCENTILE))

    @staticmethod
    def _select_bandwidth(x):
        bandwidths = 10 ** np.linspace(-1, 1, 20)
        n_splits = min(5, len(x))
        grid = GridSearchCV(KernelDensity(), {"bandwidth": bandwidths}, cv=n_splits)
        grid.fit(x)
        return grid.best_params_["bandwidth"]
