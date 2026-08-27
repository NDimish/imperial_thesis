"""Full 57-file run of EmbedKdeChecker (the KDE-based groundedness/omission
checker) -- no local log/JSON with n=57 for this checker could be found, only
a 3-file sample (prim1/prim10/prim11). This settles it with a real run rather
than reporting the 3-file number as if it were the full dataset.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Modules.evaluate import Evaluate
from Modules.embedkde_checker import EmbedKdeChecker

sys.path.insert(0, "Medical condensor")
from base import clean_transcript

INPUT_DIR = "prim57/cleaned transcripts"
OUTPUT_DIR = "prim57/bad notes lib"
LABELS_DIR = "prim57/bad notes labels lib"

input_files = sorted(os.listdir(INPUT_DIR))
output_files = sorted(os.listdir(OUTPUT_DIR))
print(f"{len(input_files)} transcripts, {len(output_files)} notes")

checker = EmbedKdeChecker()
evaluator = Evaluate(LABELS_DIR, "EmbedKdeChecker_full57")

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
