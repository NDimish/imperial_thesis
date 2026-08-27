import argparse
import os
import time

from Modules.evaluate import Evaluate
from Modules.note_extractor import extract_note_content

NOTE_DIR = "prim57/notes cleaned"  # note-only extractor -- no transcript needed
LABELS_DIR = "prim57/labels_extraction"
RESULTS_DIR = os.environ.get("RESULTS_DIR", "Logs")


def main(limit=None):
    note_files = sorted(os.listdir(NOTE_DIR))
    file_count = len(note_files) if limit is None else min(limit, len(note_files))

    evaluator = Evaluate(LABELS_DIR, "NoteExtractor")

    for i in range(file_count):
        filename = note_files[i]
        with open(os.path.join(NOTE_DIR, filename), "r", encoding="utf-8") as f:
            note = f.read()

        if filename == "prim54.txt":
            # Source file bug (same fix applied for the labels_inferences
            # eval): prim54's note has a second consultation's note
            # (belongs to prim55 -- "blinding headache", Mercilon)
            # accidentally appended after a "10.  13:30-13:50" record
            # marker. prim57/labels_extraction/prim54.txt was built from
            # only the genuine portion, so score against that same portion.
            marker = "10.  13:30-13:50"
            if marker in note:
                note = note.split(marker)[0]

        start = time.perf_counter()
        items = extract_note_content(note)
        elapsed = time.perf_counter() - start

        errors = [(item_type, detail) for item_type, section, detail, highlighted in items]

        print(f"\n=== NoteExtractor on {filename} ===")
        for item_type, section, detail, highlighted in items:
            tail = f"  [[{highlighted}]]" if highlighted else ""
            print(f"{item_type} [{section}]: {detail}{tail}")
        print(f"Time to complete: {elapsed:.4f}s")

        evaluator.compare(errors, filename, elapsed)

    evaluator.results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NoteExtractor over prim57's real SOAP notes and score against prim57/labels_extraction.")
    parser.add_argument(
        "limit",
        nargs="?",
        type=int,
        default=None,
        help="Number of files to process, starting from the first (default: all files)",
    )
    args = parser.parse_args()

    print("started run_note_extractor")
    main(args.limit)
