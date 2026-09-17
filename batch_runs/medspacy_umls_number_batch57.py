"""Full 57-file run of MedspacyUmlsNumberChecker -- the number-context
extension of the "S9.2.2" UMLS/ConText checker (see
Modules/medspacy_umls_number_checker.py's module docstring for the added
deterministic number-edit judgment and the two dataset facts that bound its
recall). Mirrors batch_runs/medspacy_umls_batch57.py exactly, swapped to the
new checker class and a distinct module_name so its own Logs/*.json doesn't
collide with the original checker's run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.evaluate import Evaluate
from Modules.medspacy_umls_number_checker import MedspacyUmlsNumberChecker

sys.path.insert(0, "Medical condensor")
from base import clean_transcript

INPUT_DIR = "prim57/cleaned transcripts"
OUTPUT_DIR = "prim57/bad notes lib"
LABELS_DIR = "prim57/bad notes labels lib"

input_files = sorted(os.listdir(INPUT_DIR))
output_files = sorted(os.listdir(OUTPUT_DIR))
print(f"{len(input_files)} transcripts, {len(output_files)} notes")

checker = MedspacyUmlsNumberChecker()
evaluator = Evaluate(LABELS_DIR, "MedspacyUmlsNumberChecker_full57")

for i, (in_f, out_f) in enumerate(zip(input_files, output_files)):
    with open(os.path.join(INPUT_DIR, in_f), "r", encoding="utf-8") as f:
        transcript = clean_transcript(f.read())
    with open(os.path.join(OUTPUT_DIR, out_f), "r", encoding="utf-8") as f:
        soap_note = f.read()
    try:
        errors, elapsed = checker.check(transcript, soap_note)
    except Exception as e:
        print(f"[{i+1}/57] {in_f} FAILED: {type(e).__name__}: {e}")
        continue
    evaluator.compare(errors, in_f, elapsed)
    print(f"[{i+1}/57] {in_f}: {len(errors)} flagged, {elapsed:.2f}s")

overall = evaluator.results()
print("\noverall:", overall.get("overall"))
print("by_type:", overall.get("by_type"))
print("by_detail_type:", overall.get("by_detail_type"))
