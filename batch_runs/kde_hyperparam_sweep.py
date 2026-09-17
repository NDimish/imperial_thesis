"""Hyperparameter sweep for EmbedKdeChecker: N_COMPONENTS x SELF_DENSITY_PERCENTILE
x THRESHOLD, swept on the project's existing 10-file KDE tuning sample (the same
10 files -- prim1/2/3/4/5/10/11/12/13/42 -- already named in embedkde_checker.py's
own THRESHOLD=100 comment), then the single best combination by overall F1 is
re-run against the full 57-file set for a real, directly comparable result against
the published (untuned) defaults (N_COMPONENTS=8, SELF_DENSITY_PERCENTILE=10,
THRESHOLD=100 -> P=4.4% R=44.7% F1=8.0%, TP=76 FP=1661 FN=94).

Deliberately a standalone script: Modules/embedkde_checker.py and
batch_runs/kde_batch57.py are both left exactly as published. The final
confirmation run overrides SELF_DENSITY_PERCENTILE via the module's own global
at call time (the class itself only exposes threshold/n_components as
constructor args) rather than editing the source file.

Design note on cost: PCA/KDE fitting is the expensive part and only depends on
(file, direction, n_components) -- not on percentile or threshold. So the sweep
precomputes, once per (file, direction, n_components): the leave-one-out
self-density array, and every candidate sentence's raw per-word log-densities.
Every (percentile, threshold) combination on top of that is then just cheap
numpy array math, not a re-fit. Evaluate.compare() is called per combo (pure
in-memory accumulation -- confirmed directly that file I/O only happens inside
Evaluate.results(), which this script deliberately never calls during the sweep
itself, to avoid writing 500+ throwaway log/json files); only the winning
combo's 10-file result and the final 57-file confirmation run call results().
"""
import itertools
import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.base import split_sentences
from Modules.evaluate import Evaluate
import Modules.embedkde_checker as ekc

sys.path.insert(0, "Medical condensor")
from base import clean_transcript

INPUT_DIR = "prim57/cleaned transcripts"
OUTPUT_DIR = "prim57/bad notes lib"
LABELS_DIR = "prim57/bad notes labels lib"

TUNE_FILES = [
    "prim1.txt", "prim2.txt", "prim3.txt", "prim4.txt", "prim5.txt",
    "prim10.txt", "prim11.txt", "prim12.txt", "prim13.txt", "prim42.txt",
]

N_COMPONENTS_GRID = [4, 5, 6, 8, 10, 12, 16, 20]
PERCENTILE_GRID = [5, 10, 15, 20, 25]
THRESHOLD_GRID = [float(t) for t in np.logspace(0, 4, 13)]  # 1 .. 10,000

MIN_REFERENCE_WORDS = ekc.MIN_REFERENCE_WORDS


def select_bandwidth(x):
    bandwidths = 10 ** np.linspace(-1, 1, 20)
    n_splits = min(5, len(x))
    grid = GridSearchCV(KernelDensity(), {"bandwidth": bandwidths}, cv=n_splits)
    grid.fit(x)
    return grid.best_params_["bandwidth"]


def leave_one_out_log_densities(points, bandwidth):
    n = len(points)
    out = np.empty(n)
    for i in range(n):
        others = np.delete(points, i, axis=0)
        loo_kde = KernelDensity(bandwidth=bandwidth).fit(others)
        out[i] = loo_kde.score_samples(points[i:i + 1])[0]
    return out


def precompute_direction(reference_text, candidate_sentences, n_components):
    """Everything for one (file, direction, n_components) triple that does NOT
    depend on percentile/threshold: the leave-one-out self-density array, and
    each candidate sentence's raw per-word log densities under the fitted KDE.
    Mirrors EmbedKdeChecker._flag exactly, just split so the expensive part is
    reusable across the whole (percentile, threshold) grid."""
    reference_vectors = ekc._embed_words(reference_text)
    if len(reference_vectors) < MIN_REFERENCE_WORDS:
        return None

    n_comp = min(n_components, len(reference_vectors), len(reference_vectors[0]))
    scaler = StandardScaler().fit(reference_vectors)
    reference_scaled = scaler.transform(reference_vectors)
    pca = PCA(n_components=n_comp, random_state=0).fit(reference_scaled)
    reference_pca = pca.transform(reference_scaled)

    bandwidth = select_bandwidth(reference_pca)
    kde = KernelDensity(bandwidth=bandwidth).fit(reference_pca)
    self_log_densities = leave_one_out_log_densities(reference_pca, bandwidth)

    per_sentence = []
    for sentence in candidate_sentences:
        word_vectors = ekc._embed_words(sentence)
        if not word_vectors:
            continue
        words_scaled = scaler.transform(word_vectors)
        words_pca = pca.transform(words_scaled)
        log_densities = kde.score_samples(words_pca)
        per_sentence.append((sentence, log_densities))

    return self_log_densities, per_sentence


def flag_sentences(precomputed, error_type, percentile, threshold):
    if precomputed is None:
        return []
    self_log_densities, per_sentence = precomputed
    floor = float(np.percentile(self_log_densities, percentile))
    flagged = []
    for sentence, log_densities in per_sentence:
        log_om = np.clip(floor - log_densities, None, 700.0)
        score = float(np.mean(np.exp(log_om)))
        if score > threshold:
            flagged.append((error_type, sentence))
    return flagged


def load_files(filenames):
    loaded = []
    for fname in filenames:
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            transcript = clean_transcript(f.read())
        with open(os.path.join(OUTPUT_DIR, fname), "r", encoding="utf-8") as f:
            soap_note = f.read()
        loaded.append((fname, transcript, soap_note))
    return loaded


def run_sweep():
    loaded = load_files(TUNE_FILES)
    soap_sentences = {fname: split_sentences(soap) for fname, _, soap in loaded}
    transcript_sentences = {fname: split_sentences(t) for fname, t, _ in loaded}

    t0 = time.time()
    precomputed = {}  # (n_components, fname) -> (hallu_pre, omit_pre)
    for n_components in N_COMPONENTS_GRID:
        for fname, transcript, soap in loaded:
            hallu_pre = precompute_direction(transcript, soap_sentences[fname], n_components)
            omit_pre = precompute_direction(soap, transcript_sentences[fname], n_components)
            precomputed[(n_components, fname)] = (hallu_pre, omit_pre)
        print(f"  precomputed n_components={n_components} ({time.time() - t0:.1f}s elapsed)", flush=True)
    print(f"Precompute done in {time.time() - t0:.1f}s\n", flush=True)

    all_results = []
    combos = list(itertools.product(N_COMPONENTS_GRID, PERCENTILE_GRID, THRESHOLD_GRID))
    print(f"Sweeping {len(combos)} (n_components, percentile, threshold) combos over {len(loaded)} files...", flush=True)
    t1 = time.time()
    for n_components, percentile, threshold in combos:
        evaluator = Evaluate(LABELS_DIR, "sweep")  # never call .results() here -- no file I/O
        for fname, _, _ in loaded:
            hallu_pre, omit_pre = precomputed[(n_components, fname)]
            errors = flag_sentences(hallu_pre, "hallucination", percentile, threshold)
            errors += flag_sentences(omit_pre, "omission", percentile, threshold)
            evaluator.compare(errors, fname, 0.0)
        tp, fp, fn = evaluator.total_tp, evaluator.total_fp, evaluator.total_fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        all_results.append((n_components, percentile, threshold, precision, recall, f1, tp, fp, fn))
    print(f"Sweep done in {time.time() - t1:.1f}s\n", flush=True)

    all_results.sort(key=lambda r: -r[5])
    print("Top 15 combos by F1 on the 10-file tuning sample:")
    for r in all_results[:15]:
        nc, pct, thr, p, r_, f1, tp, fp, fn = r
        print(f"  n_components={nc:>2}  percentile={pct:>2}  threshold={thr:>8.1f}  "
              f"P={p:.3f} R={r_:.3f} F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")

    best_n, best_p, best_t = all_results[0][0], all_results[0][1], all_results[0][2]
    print(f"\nBest on tuning sample: n_components={best_n} percentile={best_p} "
          f"threshold={best_t:.2f}  F1={all_results[0][5]:.3f} "
          f"(vs. published defaults n_components=8 percentile=10 threshold=100)")
    return best_n, best_p, best_t


def run_full57_confirmation(best_n, best_p, best_t):
    print(f"\n=== Confirming best combo (n_components={best_n}, percentile={best_p}, "
          f"threshold={best_t:.2f}) against the full 57-file set ===\n")

    # SELF_DENSITY_PERCENTILE is only ever read as a module global inside
    # _leave_one_out_min_log_density, not exposed as a constructor arg -- this
    # overrides it for this run only, without editing embedkde_checker.py.
    ekc.SELF_DENSITY_PERCENTILE = best_p

    checker = ekc.EmbedKdeChecker(threshold=best_t, n_components=best_n)
    evaluator = Evaluate(LABELS_DIR, "EmbedKdeChecker_full57_tuned")

    input_files = sorted(os.listdir(INPUT_DIR))
    output_files = sorted(os.listdir(OUTPUT_DIR))
    for i, (in_f, out_f) in enumerate(zip(input_files, output_files)):
        with open(os.path.join(INPUT_DIR, in_f), "r", encoding="utf-8") as f:
            transcript = clean_transcript(f.read())
        with open(os.path.join(OUTPUT_DIR, out_f), "r", encoding="utf-8") as f:
            soap_note = f.read()
        try:
            errors, elapsed = checker.check(transcript, soap_note)
        except Exception as e:
            print(f"[{i + 1}/57] {in_f} FAILED: {type(e).__name__}: {e}")
            continue
        evaluator.compare(errors, in_f, elapsed)
        print(f"[{i + 1}/57] {in_f}: {len(errors)} flagged, {elapsed:.2f}s")

    overall = evaluator.results()
    print("\ntuned full-57 overall:", overall.get("overall"))
    print("by_type:", overall.get("by_type"))
    print("\npublished (untuned) full-57 overall for comparison: "
          "{'tp': 76, 'fp': 1661, 'fn': 94, 'precision': 0.0438, 'recall': 0.4471, 'f1': 0.0797}")


if __name__ == "__main__":
    best_n, best_p, best_t = run_sweep()
    run_full57_confirmation(best_n, best_p, best_t)
