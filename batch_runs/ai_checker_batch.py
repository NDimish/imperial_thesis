"""One-off comparison run: AIChecker (the paper's naive Stage-1 whole-note
judge, see Modules/AI_checker.py and stage2_inference_judge.py's module docstring) over the
SAME files, SAME current Evaluate class as batch_runs/stage2_judge_batch.py's Stage2JudgeChecker
runs -- so the two are directly comparable, not eyeballed against stale
numbers from an old dataset/Evaluate version. Supports either prim57 dataset
("lib" or "lib_extra") via the --dataset flag.

Model history on this key, in order tried (see stage2_inference_judge.py's MODEL comment for
the first two):
  - gemini-3.6-flash: 20 requests/DAY free cap. Unusable for any real run.
  - gemini-3.5-flash-lite: 500/day, confirmed via the 429 error's own
    quotaValue. Used for the first "lib" AIChecker run (13/20 files before
    exhausting it -- the earlier Stage2JudgeChecker runs today had already
    used most of that 500).
  - gemini-flash-lite-latest: confirmed to be an ALIAS onto the same
    gemini-3.5-flash-lite quota bucket -- also already exhausted when tried.
  - gemini-3.1-flash-lite: a distinct, older/established model name (quotas
    are tracked per-model-name, "GenerateRequestsPerDayPerProjectPerModel"),
    confirmed callable with quota untouched by anything run today. Used here.

_throttle is monkeypatched to the shared cross-process file-lock limiter
(gemini_rate_limiter.py) instead of AI_checker.py's own in-memory-only one,
so this can't double up against any other script sharing the key.

Usage: python batch_runs/ai_checker_batch.py [lib|lib_extra] [limit]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Modules.AI_checker as ai_checker_module
import gemini_rate_limiter
from Modules.AI_checker import AIChecker
from Modules.evaluate import Evaluate

ai_checker_module._throttle = gemini_rate_limiter.throttle

MODEL = "gemini-3.1-flash-lite"
INPUT_DIR = "prim57/cleaned transcripts"
DATASETS = {
    "lib": ("prim57/bad notes lib", "prim57/bad notes labels lib", "AIChecker_lib"),
    "lib_extra": ("prim57/bad notes lib extra", "prim57/bad notes labels lib extra", "AIChecker_lib_extra"),
}
DEFAULT_LIMIT = 20
RESULTS_DIR = os.environ.get("RESULTS_DIR", "Logs")


def read_input_file(filename):
    sys.path.insert(0, "Medical condensor")
    from base import clean_transcript
    with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
        return clean_transcript(f.read())


def read_output_file(output_dir, filename):
    with open(os.path.join(output_dir, filename), "r", encoding="utf-8") as f:
        return f.read()


def main(dataset, limit):
    output_dir, labels_dir, module_name = DATASETS[dataset]

    input_files = sorted(os.listdir(INPUT_DIR))
    output_files = sorted(os.listdir(output_dir))
    file_count = min(limit, len(input_files))

    checker = AIChecker(model=MODEL)
    evaluator = Evaluate(labels_dir, module_name)

    run_start = time.perf_counter()
    for i in range(file_count):
        input_filename = input_files[i]
        output_filename = output_files[i]
        transcript = read_input_file(input_filename)
        soap_note = read_output_file(output_dir, output_filename)

        try:
            errors, elapsed = checker.check(transcript, soap_note)
        except Exception as e:
            print(f"\n=== {module_name} on {input_filename} -- FAILED ===")
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
    print(json.dumps(overall.get("overall"), indent=2))
    print(json.dumps(overall.get("by_type"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", choices=list(DATASETS), default="lib")
    parser.add_argument("limit", nargs="?", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    main(args.dataset, args.limit)
