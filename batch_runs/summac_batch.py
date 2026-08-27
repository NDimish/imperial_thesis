"""One-file smoke test for the fixed Modules/summac_checker.py (GPU,
hallucination + omission) against prim57's "lib" dataset.

Usage: python batch_runs/summac_batch.py [limit]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.evaluate import Evaluate
from Modules.summac_checker import SummaCChecker

sys.path.insert(0, "Medical condensor")
from base import clean_transcript

INPUT_DIR = "prim57/cleaned transcripts"
OUTPUT_DIR = "prim57/bad notes lib"
LABELS_DIR = "prim57/bad notes labels lib"
DEFAULT_LIMIT = 1


def main(limit):
    input_files = sorted(os.listdir(INPUT_DIR))
    output_files = sorted(os.listdir(OUTPUT_DIR))
    file_count = min(limit, len(input_files))

    print("Loading SummaCZS (vitc, cuda)...")
    load_start = time.perf_counter()
    checker = SummaCChecker()
    print(f"Loaded in {time.perf_counter() - load_start:.1f}s")

    module_name = "SummaCChecker"
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
