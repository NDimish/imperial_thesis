"""Batch runner for NLIAlignScoreChecker (Modules/NLI_alignscore.py) --
AlignScoreChecker2's own claim-splitting + Q&A bridging, scored by a
MedNLI-fine-tuned classifier instead of AlignScore's general-domain
checkpoint. Same file layout/dataset as the other alignscore-family batch
scripts, so its Logs/ output is directly comparable to theirs.

Usage: python batch_runs/nli_alignscore_batch.py [limit]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.evaluate import Evaluate
from Modules.NLI_alignscore import NLIAlignScoreChecker

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading pritamdeka/PubMedBERT-MNLI-MedNLI ({device})...")
    load_start = time.perf_counter()
    checker = NLIAlignScoreChecker(device=device)
    print(f"Loaded in {time.perf_counter() - load_start:.1f}s")

    module_name = "NLIAlignScoreChecker"
    evaluator = Evaluate(LABELS_DIR, module_name)

    run_start = time.perf_counter()
    for i in range(file_count):
        input_filename = input_files[i]
        output_filename = output_files[i]
        with open(os.path.join(INPUT_DIR, input_filename), "r", encoding="utf-8") as f:
            transcript = clean_transcript(f.read())
        with open(os.path.join(OUTPUT_DIR, output_filename), "r", encoding="utf-8") as f:
            soap_note = f.read()

        try:
            errors, elapsed = checker.check(transcript, soap_note)
        except Exception as e:
            print(f"\n=== [{i + 1}/{file_count}] {module_name} on {input_filename} -- FAILED ===")
            print(f"  {type(e).__name__}: {e}")
            continue

        print(f"\n=== [{i + 1}/{file_count}] {module_name} on {input_filename} ===")
        for error_type, detail in errors:
            print(f"{error_type}: {detail}")
        print(f"Flagged: {len(errors)} | Time: {elapsed:.1f}s")

        evaluator.compare(errors, input_filename, elapsed)

    total_elapsed = time.perf_counter() - run_start
    overall = evaluator.results()
    print(f"\nDone. {file_count} files in {total_elapsed:.1f}s.")
    print("overall:", overall.get("overall"))
    print("by_type:", overall.get("by_type"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("limit", nargs="?", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    main(args.limit)
