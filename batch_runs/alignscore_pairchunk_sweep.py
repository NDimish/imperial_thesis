"""Threshold re-sweep for Modules/alignscore_checker.py, same
method as the original Edit 3 sweep in Modules/old/alignscore_checker.py:
every claim across the 10-file calibration sample (prim1, prim10-prim18) is
scored ONCE (one model pass, cached), then every candidate threshold is
tried against those cached scores for free -- THRESHOLD=0.30 was inherited
unchanged from the old chunking scheme's own sweep and was never re-tuned
for this one.

Usage: python batch_runs/alignscore_pairchunk_sweep.py [limit]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.evaluate import Evaluate
from Modules.alignscore_checker import (
    AlignScoreChecker,
    build_context_chunks,
    _score_claim_against_chunks,
    split_full_stop_sentences,
)

sys.path.insert(0, "Medical condensor")
from base import clean_transcript

INPUT_DIR = "prim57/cleaned transcripts"
OUTPUT_DIR = "prim57/bad notes lib"
LABELS_DIR = "prim57/bad notes labels lib"
DEFAULT_LIMIT = 10

CANDIDATE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

RESULTS_DIR = os.environ.get("RESULTS_DIR", "Logs")
CACHE_PATH = os.path.join(RESULTS_DIR, "alignscore_pairchunk_sweep_cache.json")


def score_all_claims(limit):
    """One model pass: returns {filename: [[claim, score], ...]}."""
    input_files = sorted(os.listdir(INPUT_DIR))
    output_files = sorted(os.listdir(OUTPUT_DIR))
    file_count = min(limit, len(input_files))

    print("Loading AlignScoreChecker (roberta-base, cpu)...")
    load_start = time.perf_counter()
    checker = AlignScoreChecker()
    print(f"Loaded in {time.perf_counter() - load_start:.1f}s")

    cache = {}
    for i in range(file_count):
        input_filename = input_files[i]
        output_filename = output_files[i]
        with open(os.path.join(INPUT_DIR, input_filename), "r", encoding="utf-8") as f:
            transcript = clean_transcript(f.read())
        with open(os.path.join(OUTPUT_DIR, output_filename), "r", encoding="utf-8") as f:
            soap_note = f.read()

        start = time.perf_counter()
        chunks = build_context_chunks(transcript)
        claims = split_full_stop_sentences(soap_note)
        scored = [
            [claim, _score_claim_against_chunks(checker._scorer.model, claim, chunks)]
            for claim in claims
        ]
        elapsed = time.perf_counter() - start

        cache[input_filename] = scored
        print(f"[{i + 1}/{file_count}] {input_filename}: {len(scored)} claims scored in {elapsed:.1f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"\nCached scores written to {CACHE_PATH}")
    return cache


def sweep(cache):
    print("\n=== Threshold sweep (cached scores, no re-scoring) ===")
    results = []
    for threshold in CANDIDATE_THRESHOLDS:
        evaluator = Evaluate(LABELS_DIR, f"AlignScoreChecker_t{threshold:.2f}")
        for filename, scored in cache.items():
            errors = [("hallucination", claim) for claim, score in scored if score < threshold]
            evaluator.compare(errors, filename)
        # by_type["hallucination"], NOT results()["overall"] -- this checker
        # only ever predicts "hallucination", but the ground-truth label
        # files also carry "omission" labels it never attempts. Evaluate's
        # global total_fn sums FN across every label TYPE present, so every
        # omission label silently counts as a missed detection in "overall",
        # deflating recall/F1 for a metric this checker was never trying to
        # answer. by_type isolates the type this checker actually predicts,
        # matching how Modules/old/alignscore_checker.py's own 0.423 baseline
        # was reported. (Confirmed directly: an earlier version of this sweep
        # used "overall" and got fn=15 at threshold=0.30 where the equivalent
        # single-run report -- and by_type here -- both give fn=10; the
        # difference is exactly the 5 omission labels in this 10-file sample.)
        hallucination = evaluator.results()["by_type"]["hallucination"]
        tp, fp, fn = hallucination["tp"], hallucination["fp"], hallucination["fn"]
        results.append((threshold, hallucination["precision"], hallucination["recall"], hallucination["f1"], tp, fp, fn))
        print(
            f"threshold={threshold:.2f}: precision={hallucination['precision']:.3f} "
            f"recall={hallucination['recall']:.3f} f1={hallucination['f1']:.3f} (tp={tp} fp={fp} fn={fn})"
        )

    best = max(results, key=lambda r: r[3])
    print(f"\nBest F1: threshold={best[0]:.2f} -> precision={best[1]:.3f} recall={best[2]:.3f} f1={best[3]:.3f} "
          f"(tp={best[4]} fp={best[5]} fn={best[6]})")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("limit", nargs="?", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()

    cache = score_all_claims(args.limit)
    sweep(cache)
