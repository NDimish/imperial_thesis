import numpy as np
import spacy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
# NEW: needed for _select_bandwidth (cross-validated bandwidth, see below).
from sklearn.model_selection import GridSearchCV
# NEW: needed for _rank_windows_by_tfidf -- same technique already used by
# datamakerfiles/prim_lib_injection.py's get_similar_notes() to find similar
# notes, reused here to find similar WINDOWS within one transcript.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# NEW: AlignScore -- a real NLI/entailment model (not word-density), the fix
# discussed in conversation for the ceiling every KDE-based fix kept hitting
# (paraphrase blindness: "abdominal" vs "tummy" scoring as unrelated; generic
# medical vocabulary like "broken arm" scoring as well-supported). Same
# config already validated in Modules/old/alignscore_checker.py.
from alignscore import AlignScore

# 1. Load a biomedical embedding model (scispacy) instead of generic-domain word2vec.
# en_core_sci_md, not en_core_sci_sm -- confirmed directly that _sm ships NO static word
# vectors at all (nlp.vocab.vectors.shape == (0, 0), every token.vector just a tok2vec
# hash-embedding fallback, not real distributional similarity). _md carries genuine
# pretrained biomedical word vectors instead (confirmed: vocab.vectors.shape ==
# (50000, 200), is_oov=False on real clinical words), a much closer match in kind to the
# word2vec vectors this replaced. Installed via:
#   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
print("Loading scispacy model...")
nlp = spacy.load("en_core_sci_md", exclude=["ner", "parser", "lemmatizer"])
print("scispacy model loaded.")

# NEW: AlignScore setup, same config as Modules/old/alignscore_checker.py
# (that file's own comments have the full threshold-sweep history -- 0.02
# was the calibrated cutoff there, kept as the default here too).
ALIGNSCORE_CKPT_PATH = r"C:\Users\natha\OneDrive\Documents\Uni\Impreial\modules\Thesis\code\data\AlignScore-base.ckpt"
ALIGNSCORE_MODEL = "roberta-base"
ALIGNSCORE_THRESHOLD = 0.02

print("Loading AlignScore model (this loads a ~1.9GB checkpoint, may take a moment)...")
_alignscore = AlignScore(
    model=ALIGNSCORE_MODEL,
    batch_size=32,
    device="cpu",
    ckpt_path=ALIGNSCORE_CKPT_PATH,
    evaluation_mode="nli_sp",
)
print("AlignScore model loaded.")


def compute_alignscore(source_text, target_text, metric_type="hallucination"):
    """Scores target_text's entailment by source_text using AlignScore -- a
    real NLI model, not word-density -- so a paraphrase ("abdominal pain" vs
    "pain in my lower abdomen") is recognized as supported, and unrelated-
    but-common medical vocabulary ("broken arm") is correctly scored low.

    Returns a score in [0, 1] -- HIGH means well-supported/entailed (opposite
    scale from the KDE OMscore above, where high means hallucinated) --
    matching AlignScore's own convention and Modules/old/alignscore_checker.py's
    usage (flags claims where score < ALIGNSCORE_THRESHOLD as unsupported).

    - metric_type='hallucination': is target_text (the claim) supported by
      source_text (the context)?
    - metric_type='omission': is source_text (the claim) supported by
      target_text (the context)?
    """
    if metric_type == "hallucination":
        contexts, claims = [source_text], [target_text]
    else:  # omission
        contexts, claims = [target_text], [source_text]
    scores = _alignscore.score(contexts=contexts, claims=claims)
    return float(scores[0])


def get_word_vectors(text):
    """Extracts word vectors for words longer than 2 characters via scispacy."""
    doc = nlp(text.lower())
    return [tok.vector for tok in doc if tok.is_alpha and len(tok.text) > 2 and tok.has_vector]


# NEW: cross-validated bandwidth, ported from Modules/embedkde_checker.py's
# _select_bandwidth. The OLD code below used a single fixed bandwidth=1.0 for
# every reference set regardless of size/spread -- confirmed directly (see
# conversation) that this is what let a handful of tiny 2-line reference
# windows collapse to a numerically unstable near-zero self-density floor,
# making compute_best_pair_score's "lowest score" land on the same broken
# windows for every query rather than reflecting real support. Tuning
# bandwidth per reference set is the fix already validated in the real checker.
def _select_bandwidth(x):
    bandwidths = 10 ** np.linspace(-1, 1, 20)
    n_splits = min(5, len(x))
    grid = GridSearchCV(KernelDensity(), {"bandwidth": bandwidths}, cv=n_splits)
    grid.fit(x)
    return grid.best_params_["bandwidth"]


def compute_embed_kde_score(source_text, target_text, metric_type="hallucination"):
    """
    Computes EmbedKDECheck score between source and target.
    - metric_type='hallucination': Fits KDE on Source, scores Target.
    - metric_type='omission': Fits KDE on Target, scores Source.
    """
    if metric_type == "hallucination":
        ref_vectors = get_word_vectors(source_text)
        cand_vectors = get_word_vectors(target_text)
    else:  # omission
        ref_vectors = get_word_vectors(target_text)
        cand_vectors = get_word_vectors(source_text)

    if len(ref_vectors) < 5 or len(cand_vectors) < 1:
        return 0.0

    # Dimension reduction (PCA)
    scaler = StandardScaler().fit(ref_vectors)
    ref_scaled = scaler.transform(ref_vectors)
    cand_scaled = scaler.transform(cand_vectors)

    n_components = min(5, len(ref_vectors), ref_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42).fit(ref_scaled)
    ref_pca = pca.transform(ref_scaled)
    cand_pca = pca.transform(cand_scaled)

    # OLD: fixed bandwidth, never tuned to the reference set's own size/spread.
    # bandwidth = 1.0
    # NEW: cross-validated per reference set (see _select_bandwidth above).
    bandwidth = _select_bandwidth(ref_pca)
    kde = KernelDensity(bandwidth=bandwidth).fit(ref_pca)

    # OLD: leave-one-out self-density computed in LINEAR space (np.exp inside the
    # loop), then divided directly -- underflows silently to 0.0 for small/sparse
    # reference sets instead of raising, which is exactly what let the per-pair
    # score collapse to near-zero regardless of the candidate.
    # n = len(ref_pca)
    # self_densities = np.empty(n)
    # for i in range(n):
    #     others = np.delete(ref_pca, i, axis=0)
    #     loo_kde = KernelDensity(bandwidth=bandwidth).fit(others)
    #     self_densities[i] = np.exp(loo_kde.score_samples(ref_pca[i : i + 1]))[0]
    # min_self_density = np.percentile(self_densities, 10)
    # cand_densities = np.exp(kde.score_samples(cand_pca))
    # om_scores = min_self_density / (cand_densities + 1e-8)
    # return float(np.mean(om_scores))

    # NEW: everything computed in LOG space until the very last step (ported
    # from Modules/embedkde_checker.py's _flag/_leave_one_out_min_log_density),
    # so a density that underflows to 0.0 in linear space instead just becomes
    # a large-but-finite log-ratio -- no divide-by-(near)zero, no silent inf.
    n = len(ref_pca)
    self_log_densities = np.empty(n)
    for i in range(n):
        others = np.delete(ref_pca, i, axis=0)
        loo_kde = KernelDensity(bandwidth=bandwidth).fit(others)
        self_log_densities[i] = loo_kde.score_samples(ref_pca[i : i + 1])[0]
    min_self_log_density = float(np.percentile(self_log_densities, 10))

    cand_log_densities = kde.score_samples(cand_pca)

    # log(OMscore) = min_self_log_density - log_density, clipped below 700 so
    # exp() can't overflow to inf even for an astronomically unsupported word.
    log_om_scores = np.clip(min_self_log_density - cand_log_densities, None, 700.0)
    return float(np.mean(np.exp(log_om_scores)))


# OLD: 5 matched compute_embed_kde_score's own len(ref_vectors) < 5 guard, but
# that guard is the bare minimum PCA/KDE can run at all, not enough for a
# STABLE leave-one-out self-density estimate -- confirmed directly (see
# conversation): even with bandwidth cross-validation fixed, tiny 2-line
# windows still produced a near-zero "best" score for every query, because a
# handful of words is too little for the self-density floor to be a reliable
# statistic regardless of how well-tuned the bandwidth is.
# MIN_PAIR_REF_WORDS = 5
# NEW: raised alongside the wider windows below.
MIN_PAIR_REF_WORDS = 15


# OLD: fixed, non-overlapping 2-line (one doctor + one patient turn) chunks --
# confirmed the root cause of compute_best_pair_score always landing near 0:
# with 51 independent tiny (often 5-20 word) samples, taking the MINIMUM
# across them is a multiple-comparisons problem -- by chance, some window
# looks like a good match for almost ANY query, true or not, regardless of
# bandwidth tuning. Ported nothing from embedkde_checker.py here since it has
# no per-window metric at all, only whole-document scoring -- this fix is
# specific to test.py's own per-pair addition.
# def get_turn_pairs(text):
#     """Splits a turn-tagged transcript into consecutive, non-overlapping two-line
#     (doctor + patient) chunks -- lines 0-1, lines 2-3, ... -- for per-pair scoring
#     instead of scoring the whole transcript as one block."""
#     lines = [line for line in text.splitlines() if line.strip()]
#     return [lines[i] + "\n" + lines[i + 1] for i in range(0, len(lines) - 1, 2)]

# NEW: wider (WINDOW_LINES-line), OVERLAPPING (stride WINDOW_STRIDE) windows.
# Widening gives each window's KDE roughly 4x the words to fit on -- far more
# stable leave-one-out self-density -- and overlap (stride < window size)
# means no real turn-pair boundary sits split across two windows the way a
# fixed non-overlapping split could. Window/stride are a reasoned guess (not
# swept): 8 lines was chosen to comfortably clear MIN_PAIR_REF_WORDS=15 on
# real transcript turns without pulling in multiple unrelated topic changes
# in one window.
WINDOW_LINES = 8
WINDOW_STRIDE = 2

# NEW: replaces the bare min() in compute_best_pair_score below with a low
# percentile across windows -- the same "percentile, not strict min" fix
# already validated in compute_embed_kde_score's own self-density floor (see
# its min_self_log_density), applied here for the identical reason: a strict
# min over many windows is hostage to whichever single window's leave-one-out
# estimate happens to spike lowest by chance, even after widening the
# windows reduces (but doesn't eliminate) that per-window noise.
BEST_PAIR_PERCENTILE = 10


def get_turn_windows(text):
    """Splits a turn-tagged transcript into overlapping WINDOW_LINES-line
    windows, stepping by WINDOW_STRIDE lines -- see the comment above for why
    this replaced the old fixed 2-line, non-overlapping pairing."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < WINDOW_LINES:
        return ["\n".join(lines)] if lines else []
    return [
        "\n".join(lines[i : i + WINDOW_LINES])
        for i in range(0, len(lines) - WINDOW_LINES + 1, WINDOW_STRIDE)
    ]


# NEW: how many windows the TF-IDF retrieval step keeps before the (expensive,
# and still generic-word-biased on its own -- see conversation) KDE scoring
# runs. Small on purpose: the point of retrieval is to hand compute_embed_kde_
# score a genuinely topical neighborhood instead of all 51 windows, so it's
# no longer searching broadly enough for a common word to find a lucky match
# somewhere.
TOP_K_WINDOWS = 5


def _rank_windows_by_tfidf(windows, target_text):
    """Ranks windows by TF-IDF cosine similarity to target_text, most similar
    first -- same technique datamakerfiles/prim_lib_injection.py's
    get_similar_notes() already uses to find similar notes, applied here to
    find similar WINDOWS within one transcript.

    This is the actual fix for the generic-word bias found in the whole-
    transcript AND per-window KDE scores (see conversation): TF-IDF's IDF
    term specifically DOWN-weights common words and UP-weights rare/specific
    ones -- the opposite bias from KDE density, which treats "common in the
    reference vocabulary" as unremarkable regardless of whether it's the
    right topic. "broken arm" won't score high similarity against any window
    here, because "arm"/"broken" don't appear anywhere in this transcript at
    all; "diarrhea" will correctly rank the windows that actually discuss it
    above the ones that don't.
    """
    if not windows:
        return []
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(windows + [target_text])
    target_vector = matrix[-1]
    window_vectors = matrix[:-1]
    similarities = cosine_similarity(target_vector, window_vectors)[0]
    ranked = sorted(zip(similarities, windows), key=lambda pair: pair[0], reverse=True)
    return [window for _similarity, window in ranked]


def compute_best_pair_score(source_text, target_text, metric_type="hallucination"):
    """Scores target_text against the TOP_K_WINDOWS windows (see
    get_turn_windows) of source_text most lexically similar to it (see
    _rank_windows_by_tfidf), and returns the BEST_PAIR_PERCENTILE-th
    percentile score across those -- the region of the transcript the target
    text is genuinely topically closest to, not whichever window out of ALL
    of them a common word happened to get lucky against (see the module-level
    comments on TOP_K_WINDOWS/_rank_windows_by_tfidf and the constants above
    for the full history of why this two-stage retrieve-then-score design
    replaced scoring every window blind).

    Windows with fewer than MIN_PAIR_REF_WORDS extractable word vectors are
    skipped rather than scored -- compute_embed_kde_score can't fit a
    meaningful KDE on that little text and just returns a trivial 0.0
    short-circuit (see its own len(ref_vectors) < 5 guard).
    """
    all_windows = get_turn_windows(source_text)
    neighborhood = _rank_windows_by_tfidf(all_windows, target_text)[:TOP_K_WINDOWS]

    scored = []
    for window_text in neighborhood:
        if len(get_word_vectors(window_text)) < MIN_PAIR_REF_WORDS:
            continue
        score = compute_embed_kde_score(window_text, target_text, metric_type=metric_type)
        scored.append((score, window_text))

    if not scored:
        return None

    scores = np.array([s for s, _ in scored])
    percentile_score = float(np.percentile(scores, BEST_PAIR_PERCENTILE))
    # Report the actual window closest to that percentile value, for context.
    closest_window = min(scored, key=lambda pair: abs(pair[0] - percentile_score))[1]
    print(f"Best Pair Score ({BEST_PAIR_PERCENTILE}th percentile of top-{TOP_K_WINDOWS} TF-IDF-similar windows): {closest_window}")
    return percentile_score


INPUT_DIR = "prim57/cleaned transcripts"
import os

# Example Usage
if __name__ == "__main__":

    path = os.path.join(INPUT_DIR, "prim1.txt")
    with open(path, "r", encoding="utf-8") as f:
            transcript = f.read()

    source_transcript = transcript

    x = "x"

    while x != "Q":
        x = input("Enter a target SOAP record (or press Q to exit): ")
        if x == "Q":
            break
        target_soap = x

        score = compute_embed_kde_score(source_transcript, target_soap, metric_type="hallucination")
        print(f"Hallucination Score (whole transcript vs target): {score:.4f}")

        # pair_score = compute_best_pair_score(source_transcript, target_soap, metric_type="hallucination")
        # if pair_score is None:
        #     print("Best Pair Score: n/a (no window had enough text to score)")
        # else:
        #     print(f"Best Pair Score ({BEST_PAIR_PERCENTILE}th percentile across windows): {pair_score:.6f}")

        # NEW: AlignScore, for direct comparison against the two KDE-based scores
        # above. Note the scale is OPPOSITE the KDE scores: high AlignScore =
        # well-supported, low/negative = unsupported (flagged below threshold).
        align_score = compute_alignscore(source_transcript, target_soap, metric_type="hallucination")
        verdict = "SUPPORTED" if align_score >= ALIGNSCORE_THRESHOLD else "UNSUPPORTED (hallucination)"
        print(f"AlignScore (whole transcript vs target): {align_score:.4f}  -> {verdict}")
