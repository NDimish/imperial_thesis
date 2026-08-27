"""Batch runner for Stage2JudgeChecker (stage2_inference_judge.py's paper-Stage-2 Inference-
Aware Judge, see Modules/old/stage2_judge_checker.py) over prim57 + its "lib"
injected-error dataset -- one Gemini call per SOAP-note sentence/claim,
throttled across processes via gemini_rate_limiter.py.

This is the SMOKE TEST configuration: 20 of the 57 files, the "lib" dataset
only (prim57/bad notes lib + prim57/bad notes labels lib), a single run --
not the full 57-file x2-dataset x2-run evaluation. At ~237 claims for 20
files and a 18s/claim throttle, this takes roughly 70-80 minutes.

Usage: python batch_runs/stage2_judge_batch.py [limit]
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.evaluate import Evaluate
from Modules.old.stage2_judge_checker import Stage2JudgeChecker

sys.path.insert(0, "Medical condensor")
from base import clean_transcript

INPUT_DIR = "prim57/cleaned transcripts"
OUTPUT_DIR = "prim57/bad notes lib"
LABELS_DIR = "prim57/bad notes labels lib"
DEFAULT_LIMIT = 20

RESULTS_DIR = os.environ.get("RESULTS_DIR", "Logs")


def read_input_file(filename):
    with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
        return clean_transcript(f.read())


def read_output_file(filename):
    with open(os.path.join(OUTPUT_DIR, filename), "r", encoding="utf-8") as f:
        return f.read()


def main(limit):
    input_files = sorted(os.listdir(INPUT_DIR))
    output_files = sorted(os.listdir(OUTPUT_DIR))
    file_count = min(limit, len(input_files))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = os.path.join(RESULTS_DIR, f"stage2_judge_smoke_test_{timestamp}.json")

    checker = Stage2JudgeChecker()
    module_name = "Stage2JudgeChecker"
    evaluator = Evaluate(LABELS_DIR, module_name)

    file_records = []
    run_start = time.perf_counter()

    for i in range(file_count):
        input_filename = input_files[i]
        output_filename = output_files[i]

        transcript = read_input_file(input_filename)
        soap_note = read_output_file(output_filename)

        file_start = time.perf_counter()
        try:
            errors, elapsed = checker.check(transcript, soap_note)
        except Exception as e:
            print(f"\n=== {module_name} on {input_filename} -- FAILED ===")
            print(f"  {type(e).__name__}: {e}")
            continue
        file_elapsed = time.perf_counter() - file_start

        print(f"\n=== [{i + 1}/{file_count}] {module_name} on {input_filename} ===")
        for error in errors:
            # Stage2JudgeChecker returns (type, severity, detail_type, detail)
            # 4-tuples now (see Modules/stage2_judge_checker.py) -- same shape
            # Modules/high_risk_checker.py already uses and Evaluate.compare()
            # already understands. Handle both shapes rather than assuming 2.
            if len(error) == 4:
                error_type, severity, detail_type, detail = error
                print(f"{error_type} [{severity}/{detail_type}]: {detail}")
            else:
                error_type, detail = error
                print(f"{error_type}: {detail}")
        print(f"Claims judged: {len(checker.last_claims)} | Flagged hallucinated: {len(errors)} | "
              f"Time: {file_elapsed:.1f}s")

        record = evaluator.compare(errors, input_filename, elapsed)
        file_records.append({
            "filename": input_filename,
            "claims": checker.last_claims,
            "eval": record,
            "file_elapsed": file_elapsed,
        })

        # Flush progress after every file -- this run takes over an hour;
        # a crash or interruption partway through shouldn't lose everything
        # judged so far.
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump({
                "module": module_name,
                "labels_dir": LABELS_DIR,
                "output_dir": OUTPUT_DIR,
                "file_count_planned": file_count,
                "files_completed": len(file_records),
                "files": file_records,
            }, f, indent=2)

    total_elapsed = time.perf_counter() - run_start
    overall = evaluator.results()

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump({
            "module": module_name,
            "labels_dir": LABELS_DIR,
            "output_dir": OUTPUT_DIR,
            "file_count_planned": file_count,
            "files_completed": len(file_records),
            "total_elapsed": total_elapsed,
            "overall": overall,
            "files": file_records,
        }, f, indent=2)

    print(f"\nDone. {len(file_records)}/{file_count} files completed in {total_elapsed:.1f}s.")
    print(f"Detail JSON: {detail_path}")
    return detail_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-run Stage2JudgeChecker over prim57's lib dataset.")
    parser.add_argument("limit", nargs="?", type=int, default=DEFAULT_LIMIT,
                         help=f"Number of files to process, starting from the first (default: {DEFAULT_LIMIT})")
    args = parser.parse_args()

    main(args.limit)
