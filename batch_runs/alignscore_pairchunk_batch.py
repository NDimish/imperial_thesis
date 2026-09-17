"""Smoke test for Modules/alignscore_checker.py (CPU,
hallucination only) against prim57's "lib" dataset -- run with no argument
to reproduce the exact 10-file calibration sample (prim1, prim10-prim18)
used to tune THRESHOLD in Modules/old/alignscore_checker.py, for a direct
before/after comparison against that checker's own numbers on the same
files (precision=0.367 recall=0.500 f1=0.423, tp=11 fp=19 fn=11).

Usage: python batch_runs/alignscore_pairchunk_batch.py [limit]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.evaluate import Evaluate
from Modules.alignscore_checker import AlignScoreChecker

sys.path.insert(0, "Medical condensor")
from base import clean_transcript

INPUT_DIR = "prim57/cleaned transcripts"
OUTPUT_DIR = "prim57/bad notes lib"
LABELS_DIR = "prim57/bad notes labels lib"
DEFAULT_LIMIT = 10


def main(limit):
    input_files = sorted(os.listdir(INPUT_DIR))
    output_files = sorted(os.listdir(OUTPUT_DIR))
    file_count = min(limit, len(input_files))

    print("Loading AlignScoreChecker (roberta-base, cpu)...")
    load_start = time.perf_counter()
    checker = AlignScoreChecker()
    print(f"Loaded in {time.perf_counter() - load_start:.1f}s")

    module_name = "AlignScoreChecker"
    evaluator = Evaluate(LABELS_DIR, module_name)

    for i in range(file_count):
        input_filename = input_files[i]
        output_filename = output_files[i]
        with open(os.path.join(INPUT_DIR, input_filename), "r", encoding="utf-8") as f:
            transcript = clean_transcript(f.read())
        with open(os.path.join(OUTPUT_DIR, output_filename), "r", encoding="utf-8") as f:
            soap_note = f.read()

        errors, elapsed = checker.check(transcript, soap_note)

        print(f"\n=== [{i + 1}/{file_count}] {module_name} on {input_filename} ===")
        for error_type, detail in errors:
            print(f"{error_type}: {detail}")
        print(f"Flagged: {len(errors)} | Time: {elapsed:.1f}s")

        evaluator.compare(errors, input_filename, elapsed)

    overall = evaluator.results()
    print("\noverall:", overall.get("overall"))
    print("by_type:", overall.get("by_type"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("limit", nargs="?", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    main(args.limit)
