"""Full 57-file run of Modules/alignscore_checker.py (d/p pair chunking), in two
transcript-preprocessing variants -- "raw" (cleaned transcript, as used in
the 10-file calibration run) and "medspacy" (transcript run through
MedspacyCondenserNew.condense() before pairing/chunking, the same condenser
Section 9.2.1 selected as this project's default) -- so the pair-chunking
approach can be checked against the full corpus rather than the 10-file
calibration sample, and against a condensed-transcript variant to see
whether stripping non-clinical turns before pairing helps or hurts.

Every claim in every file, in both variants, is scored ONCE and cached to
JSON (the expensive part -- 57 files x 2 variants, ~220s/file on CPU based
on the 10-file sample, so several hours total). Threshold sweeping against
those cached scores is then free, same method as the 10-file re-sweep.

MedspacyCondenserNew.condense() returns turns re-joined via join_turns
(confirmed by reading Medical condensor/medspacy_condenser_new.py directly),
so it preserves the "d:"/"p:" tags build_context_chunks()'s pairing logic
depends on -- condensing only removes some turns' text, it doesn't restructure
the surviving ones.

Usage: python batch_runs/alignscore_pairchunk_57_sweep.py [limit]
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
from medspacy_condenser_new import MedspacyCondenserNew

INPUT_DIR = "prim57/cleaned transcripts"
OUTPUT_DIR = "prim57/bad notes lib"
LABELS_DIR = "prim57/bad notes labels lib"
DEFAULT_LIMIT = 57

CANDIDATE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

RESULTS_DIR = os.environ.get("RESULTS_DIR", "Logs")
CACHE_PATH = os.path.join(RESULTS_DIR, "alignscore_pairchunk_57_cache.json")


def score_all(limit):
    """One model pass per (file, variant): returns
    {"raw": {filename: [[claim, score], ...]}, "medspacy": {...}}."""
    input_files = sorted(os.listdir(INPUT_DIR))
    output_files = sorted(os.listdir(OUTPUT_DIR))
    file_count = min(limit, len(input_files))

    print("Loading AlignScoreChecker (roberta-base, cpu)...", flush=True)
    load_start = time.perf_counter()
    checker = AlignScoreChecker()
    inferencer = checker._scorer.model
    print(f"Loaded in {time.perf_counter() - load_start:.1f}s", flush=True)

    condenser = MedspacyCondenserNew()

    cache = {"raw": {}, "medspacy": {}}
    run_start = time.perf_counter()
    for i in range(file_count):
        input_filename = input_files[i]
        output_filename = output_files[i]
        with open(os.path.join(INPUT_DIR, input_filename), "r", encoding="utf-8") as f:
            raw_transcript = clean_transcript(f.read())
        with open(os.path.join(OUTPUT_DIR, output_filename), "r", encoding="utf-8") as f:
            soap_note = f.read()

        claims = split_full_stop_sentences(soap_note)

        for variant_name, transcript in (
            ("raw", raw_transcript),
            ("medspacy", condenser.condense(raw_transcript)[0]),
        ):
            start = time.perf_counter()
            chunks = build_context_chunks(transcript)
            scored = [[claim, _score_claim_against_chunks(inferencer, claim, chunks)] for claim in claims]
            elapsed = time.perf_counter() - start
            cache[variant_name][input_filename] = scored
            print(
                f"[{i + 1}/{file_count}] {input_filename} [{variant_name}]: "
                f"{len(chunks)} chunks, {len(scored)} claims scored in {elapsed:.1f}s",
                flush=True,
            )

        # Checkpoint after every file -- a 6-7 hour run should survive an
        # interruption without losing everything scored so far.
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

        total_elapsed = time.perf_counter() - run_start
        print(f"  -- running total: {total_elapsed / 60:.1f} min elapsed, {file_count - i - 1} files left", flush=True)

    print(f"\nAll scores cached to {CACHE_PATH}", flush=True)
    return cache


def sweep_variant(variant_name, per_file_scores):
    print(f"\n=== Threshold sweep: {variant_name} ===", flush=True)
    results = []
    for threshold in CANDIDATE_THRESHOLDS:
        evaluator = Evaluate(LABELS_DIR, f"pairchunk57_{variant_name}_t{threshold:.2f}")
        for filename, scored in per_file_scores.items():
            errors = [("hallucination", claim) for claim, score in scored if score < threshold]
            evaluator.compare(errors, filename)
        hallucination = evaluator.results()["by_type"].get(
            "hallucination", {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        )
        tp, fp, fn = hallucination["tp"], hallucination["fp"], hallucination["fn"]
        results.append((threshold, hallucination["precision"], hallucination["recall"], hallucination["f1"], tp, fp, fn))
        print(
            f"threshold={threshold:.2f}: precision={hallucination['precision']:.3f} "
            f"recall={hallucination['recall']:.3f} f1={hallucination['f1']:.3f} (tp={tp} fp={fp} fn={fn})",
            flush=True,
        )
    best = max(results, key=lambda r: r[3])
    print(
        f"\nBest F1 [{variant_name}]: threshold={best[0]:.2f} -> precision={best[1]:.3f} "
        f"recall={best[2]:.3f} f1={best[3]:.3f} (tp={best[4]} fp={best[5]} fn={best[6]})",
        flush=True,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("limit", nargs="?", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()

    cache = score_all(args.limit)
    sweep_variant("raw", cache["raw"])
    sweep_variant("medspacy", cache["medspacy"])
