import argparse
import os
import sys

from Modules.evaluate import Evaluate

sys.path.insert(0, "Medical condensor")
from base import clean_transcript

from run_checker_modules import CONDENSER, load_checker_modules


def read_transcript(path):
    with open(path, "r", encoding="utf-8") as f:
        return clean_transcript(f.read())


def read_note(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main(transcript_path, note_path, label_path):
    modules = load_checker_modules()
    print(f"Loaded {len(modules)} checker module(s).")

    condenser = CONDENSER()
    print(f"Using {condenser.__class__.__name__} to condense transcripts.")

    transcript = read_transcript(transcript_path)
    soap_note = read_note(note_path)

    has_labels = bool(label_path) and os.path.isfile(label_path)
    if label_path and not has_labels:
        print(f"Label file not found at {label_path} -- proceeding with no labels.")

    for module in modules:
        module_name = module.__class__.__name__
        print(f"\n=== Running {module_name} ===")

        try:
            errors, elapsed = module.check(transcript, soap_note)
        except Exception as e:
            print(f"  {type(e).__name__}: {e}")
            continue

        for error in errors:
            # HighRiskChecker (and any future severity-aware checker) returns
            # (type, severity, detail_type, detail, section, highlighted) 6-tuples --
            # "highlighted" is the specific term/phrase within detail that triggered
            # the flag, display-only (never fed to Evaluate's matching) -- or the
            # earlier (type, severity, detail_type, detail, section) 5-tuples/
            # (type, severity, detail_type, detail) 4-tuples, instead of the
            # (type, detail) 2-tuples every other checker here returns.
            if len(error) == 6:
                error_type, severity, detail_type, detail, section, highlighted = error
                text = f"{detail} :::: {highlighted}" if highlighted else detail
                print(f"{error_type} [{severity}/{detail_type}/{section}]: {text}")
            elif len(error) == 5:
                error_type, severity, detail_type, detail, section = error
                print(f"{error_type} [{severity}/{detail_type}/{section}]: {detail}")
            elif len(error) == 4:
                error_type, severity, detail_type, detail = error
                print(f"{error_type} [{severity}/{detail_type}]: {detail}")
            else:
                error_type, detail = error
                print(f"{error_type}: {detail}")
        print(f"Time to complete: {elapsed:.2f}s")

        if has_labels:
            evaluator = Evaluate(os.path.dirname(label_path), module_name)
            evaluator.compare(errors, os.path.basename(label_path), elapsed)
            evaluator.results()
            print(f"Wrote {evaluator.log_path} / {evaluator.json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run every checker module in run_checker_modules.py against a single transcript/note pair, "
        "instead of the whole prim57 dataset."
    )
    parser.add_argument("transcript_path", help="Path to the transcript file")
    parser.add_argument("note_path", help="Path to the SOAP note file")
    parser.add_argument(
        "label_path",
        nargs="?",
        default=None,
        help="Path to the label file for this pair (optional -- if omitted or not found, runs with no labels/scoring)",
    )
    args = parser.parse_args()

    print("started run_single_pair")

    main(args.transcript_path, args.note_path, args.label_path)
