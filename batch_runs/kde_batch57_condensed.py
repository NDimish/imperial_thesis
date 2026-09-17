"""Full 57-file run of EmbedKdeChecker, but with the transcript pre-filtered
through MedspacyCondenserNew before it reaches the checker -- tests the first
proposed fix in the thesis's Possible Fixes for Further Research (Sec 9.3.6):
"Pre-filter the transcript before scoring the omission direction, reusing the
medspaCy condenser already validated in Sec 9.2.1 to strip conversational,
non-clinical turns before they ever reach the KDE."

Directly comparable to batch_runs/kde_batch57.py (the published, uncondensed
result: TP=76 FP=1661 FN=94, P=4.4% R=44.7% F1=8.0%) -- same checker, same
files, same labels, the ONLY difference is this file's extra condense() call.
Confirmed directly that kde_batch57.py and run_checker_modules.py both skip
condensing entirely (the latter's condenser.condense() call is commented out,
despite printing "Using MedspacyCondenserNew to condense transcripts"), so
the published number was never actually tested against this fix before now.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.evaluate import Evaluate
from Modules.embedkde_checker import EmbedKdeChecker

sys.path.insert(0, "Medical condensor")
from base import clean_transcript
from medspacy_condenser_new import MedspacyCondenserNew

INPUT_DIR = "prim57/cleaned transcripts"
OUTPUT_DIR = "prim57/bad notes lib"
LABELS_DIR = "prim57/bad notes labels lib"

input_files = sorted(os.listdir(INPUT_DIR))
output_files = sorted(os.listdir(OUTPUT_DIR))
print(f"{len(input_files)} transcripts, {len(output_files)} notes")

checker = EmbedKdeChecker()
condenser = MedspacyCondenserNew()
evaluator = Evaluate(LABELS_DIR, "EmbedKdeChecker_full57_condensed")

total_condense_time = 0.0
for i, (in_f, out_f) in enumerate(zip(input_files, output_files)):
    with open(os.path.join(INPUT_DIR, in_f), "r", encoding="utf-8") as f:
        transcript = clean_transcript(f.read())
    with open(os.path.join(OUTPUT_DIR, out_f), "r", encoding="utf-8") as f:
        soap_note = f.read()

    condensed_transcript, condense_elapsed = condenser.condense(transcript)
    total_condense_time += condense_elapsed

    try:
        errors, check_elapsed = checker.check(condensed_transcript, soap_note)
    except Exception as e:
        print(f"[{i+1}/57] {in_f} FAILED: {type(e).__name__}: {e}")
        continue
    evaluator.compare(errors, in_f, condense_elapsed + check_elapsed)
    print(f"[{i+1}/57] {in_f}: {len(errors)} flagged, condense={condense_elapsed:.2f}s check={check_elapsed:.2f}s")

overall = evaluator.results()
print("\noverall:", overall.get("overall"))
print("by_type:", overall.get("by_type"))
print(f"\ntotal condense time: {total_condense_time:.1f}s ({total_condense_time/len(input_files):.2f}s/file avg)")
